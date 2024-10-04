import Mathlib

namespace initial_number_of_employees_l106_106264

variables (E : ℕ)
def hourly_rate : ℕ := 12
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def extra_employees : ℕ := 200
def total_payroll : ℕ := 1680000

-- Total hours worked by each employee per month
def monthly_hours_per_employee : ℕ := hours_per_day * days_per_week * weeks_per_month

-- Monthly salary per employee
def monthly_salary_per_employee : ℕ := monthly_hours_per_employee * hourly_rate

-- Condition expressing the constraint given in the problem
def payroll_equation : Prop :=
  (E + extra_employees) * monthly_salary_per_employee = total_payroll

-- The statement we are proving
theorem initial_number_of_employees :
  payroll_equation E → E = 500 :=
by
  -- Proof not required
  intros
  sorry

end initial_number_of_employees_l106_106264


namespace div_a2_plus_2_congr_mod8_l106_106756

variable (a d : ℤ)
variable (h_odd : a % 2 = 1)
variable (h_pos : a > 0)

theorem div_a2_plus_2_congr_mod8 :
  (d ∣ (a ^ 2 + 2)) → (d % 8 = 1 ∨ d % 8 = 3) :=
by
  sorry

end div_a2_plus_2_congr_mod8_l106_106756


namespace important_countries_l106_106605

-- Define the main theorem
theorem important_countries
  (N d : ℕ)
  (hN : N ≥ d + 2)
  (G : SimpleGraph (Fin N))
  [decidable_rel G.adj]
  (h_regular : ∀ v : Fin N, G.degree v = d)
  (h_connected : Connected G)
  (h_important : ∀ v : Fin N, ∃ u1 u2 : Fin N, u1 ≠ u2 ∧ Connected (G.delete_vertices {v, v.neighbors}) u1 u2 = false) :
  ∃ (u w : Fin N), (G.neighbor_finset u ∩ G.neighbor_finset w).card > 2 * d / 3 :=
sorry

end important_countries_l106_106605


namespace sqrt_cos_sin_relation_l106_106588

variable {a b c θ : ℝ}

theorem sqrt_cos_sin_relation 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * (Real.cos θ) ^ 2 + b * (Real.sin θ) ^ 2 < c) :
  Real.sqrt a * (Real.cos θ) ^ 2 + Real.sqrt b * (Real.sin θ) ^ 2 < Real.sqrt c :=
sorry

end sqrt_cos_sin_relation_l106_106588


namespace abs_fraction_inequality_solution_l106_106772

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l106_106772


namespace max_value_of_reciprocal_powers_l106_106619

variable {R : Type*} [CommRing R]
variables (s q r₁ r₂ : R)

-- Condition: the roots of the polynomial
def is_roots_of_polynomial (s q r₁ r₂ : R) : Prop :=
  r₁ + r₂ = s ∧ r₁ * r₂ = q ∧ (r₁ + r₂ = r₁ ^ 2 + r₂ ^ 2) ∧ (r₁ + r₂ = r₁^10 + r₂^10)

-- The theorem that needs to be proven
theorem max_value_of_reciprocal_powers (s q r₁ r₂ : ℝ) (h : is_roots_of_polynomial s q r₁ r₂):
  (∃ r₁ r₂, r₁ + r₂ = s ∧ r₁ * r₂ = q ∧
             r₁ + r₂ = r₁^2 + r₂^2 ∧
             r₁ + r₂ = r₁^10 + r₂^10) →
  (r₁^ 11 ≠ 0 ∧ r₂^11 ≠ 0 ∧
  ((1 / r₁^11) + (1 / r₂^11) = 2)) :=
by
  sorry

end max_value_of_reciprocal_powers_l106_106619


namespace friday_lending_tuesday_vs_thursday_total_lending_l106_106496

def standard_lending_rate : ℕ := 50
def monday_excess : ℤ := 0
def tuesday_excess : ℤ := 8
def wednesday_excess : ℤ := 6
def thursday_shortfall : ℤ := -3
def friday_shortfall : ℤ := -7

theorem friday_lending : (standard_lending_rate + friday_shortfall) = 43 := by
  sorry

theorem tuesday_vs_thursday : (tuesday_excess - thursday_shortfall) = 11 := by
  sorry

theorem total_lending : 
  (5 * standard_lending_rate + (monday_excess + tuesday_excess + wednesday_excess + thursday_shortfall + friday_shortfall)) = 254 := by
  sorry

end friday_lending_tuesday_vs_thursday_total_lending_l106_106496


namespace odd_power_of_7_plus_1_divisible_by_8_l106_106081

theorem odd_power_of_7_plus_1_divisible_by_8 (n : ℕ) (h : n % 2 = 1) : (7 ^ n + 1) % 8 = 0 :=
by
  sorry

end odd_power_of_7_plus_1_divisible_by_8_l106_106081


namespace integer_solutions_count_2009_l106_106446

theorem integer_solutions_count_2009 :
  ∃ s : Finset (ℤ × ℤ × ℤ), (∀ (x y z : ℤ), (x, y, z) ∈ s ↔ x * y * z = 2009) ∧ s.card = 72 :=
  sorry

end integer_solutions_count_2009_l106_106446


namespace evaluate_expression_l106_106538

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l106_106538


namespace simplify_expression_l106_106370

-- Define the given conditions
def pow_2_5 : ℕ := 32
def pow_4_4 : ℕ := 256
def pow_2_2 : ℕ := 4
def pow_neg_2_3 : ℤ := -8

-- State the theorem to prove
theorem simplify_expression : 
  (pow_2_5 + pow_4_4) * (pow_2_2 - pow_neg_2_3)^8 = 123876479488 := 
by
  sorry

end simplify_expression_l106_106370


namespace quadratic_sum_l106_106698

theorem quadratic_sum (b c : ℤ) : 
  (∃ b c : ℤ, (x^2 - 10*x + 15 = 0) ↔ ((x + b)^2 = c)) → b + c = 5 :=
by
  intros h
  sorry

end quadratic_sum_l106_106698


namespace simplify_polynomials_l106_106226

theorem simplify_polynomials :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 3 * x - 8) = x^3 + 2 * x^2 + 3 * x + 3 :=
by
  sorry

end simplify_polynomials_l106_106226


namespace problem_not_equivalent_l106_106508

theorem problem_not_equivalent :
  (0.0000396 ≠ 3.9 * 10^(-5)) ∧ 
  (0.0000396 = 3.96 * 10^(-5)) ∧ 
  (0.0000396 = 396 * 10^(-7)) ∧ 
  (0.0000396 = (793 / 20000) * 10^(-5)) ∧ 
  (0.0000396 = 198 / 5000000) :=
by
  sorry

end problem_not_equivalent_l106_106508


namespace smallest_positive_n_l106_106554

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l106_106554


namespace sum_divisibility_l106_106768

theorem sum_divisibility (a b : ℤ) (h : 6 * a + 11 * b ≡ 0 [ZMOD 31]) : a + 7 * b ≡ 0 [ZMOD 31] :=
sorry

end sum_divisibility_l106_106768


namespace sprinkler_days_needed_l106_106130

-- Definitions based on the conditions
def morning_water : ℕ := 4
def evening_water : ℕ := 6
def daily_water : ℕ := morning_water + evening_water
def total_water_needed : ℕ := 50

-- The proof statement
theorem sprinkler_days_needed : total_water_needed / daily_water = 5 := by
  sorry

end sprinkler_days_needed_l106_106130


namespace train_speed_l106_106271

noncomputable def train_length : ℝ := 65 -- length of the train in meters
noncomputable def time_to_pass : ℝ := 6.5 -- time to pass the telegraph post in seconds
noncomputable def speed_conversion_factor : ℝ := 18 / 5 -- conversion factor from m/s to km/h

theorem train_speed (h_length : train_length = 65) (h_time : time_to_pass = 6.5) :
  (train_length / time_to_pass) * speed_conversion_factor = 36 :=
by
  simp [h_length, h_time, train_length, time_to_pass, speed_conversion_factor]
  sorry

end train_speed_l106_106271


namespace minimum_value_l106_106871

-- Define geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * ((a 2 / a 1) ^ n)

-- Define the condition for positive geometric sequence
def positive_geometric_sequence (a : ℕ → ℝ) :=
  is_geometric_sequence a ∧ ∀ n : ℕ, a n > 0

-- Condition given in the problem
def condition (a : ℕ → ℝ) :=
  2 * a 4 + a 3 = 2 * a 2 + a 1 + 8

-- Define the problem statement to be proved
theorem minimum_value (a : ℕ → ℝ) (h1 : positive_geometric_sequence a) (h2 : condition a) :
  2 * a 6 + a 5 = 32 :=
sorry

end minimum_value_l106_106871


namespace ball_weights_l106_106311

-- Define the weights of red and white balls we are going to use in our conditions and goal
variables (R W : ℚ)

-- State the conditions as hypotheses
axiom h1 : 7 * R + 5 * W = 43
axiom h2 : 5 * R + 7 * W = 47

-- State the theorem we want to prove, given the conditions
theorem ball_weights :
  4 * R + 8 * W = 49 :=
by
  sorry

end ball_weights_l106_106311


namespace problem_l106_106571

noncomputable def f (a b x : ℝ) := a * x^2 - b * x + 1

theorem problem (a b : ℝ) (h1 : 4 * a - b^2 = 3)
                (h2 : ∀ x : ℝ, f a b (x + 1) = f a b (-x))
                (h3 : b = a + 1) 
                (h4 : 0 ≤ a ∧ a ≤ 1) 
                (h5 : ∀ x ∈ Set.Icc 0 2, ∃ m : ℝ, m ≥ abs (f a b x)) :
  (∀ x : ℝ, f a b x = x^2 - x + 1) ∧ (∃ m : ℝ, m = 1 ∧ ∀ x ∈ Set.Icc 0 2, m ≥ abs (f a b x)) :=
  sorry

end problem_l106_106571


namespace impossible_to_convince_logical_jury_of_innocence_if_guilty_l106_106091

theorem impossible_to_convince_logical_jury_of_innocence_if_guilty :
  (guilty : Prop) →
  (jury_is_logical : Prop) →
  guilty →
  (∀ statement : Prop, (logical_deduction : Prop) → (logical_deduction → ¬guilty)) →
  False :=
by
  intro guilty jury_is_logical guilty_premise logical_argument
  sorry

end impossible_to_convince_logical_jury_of_innocence_if_guilty_l106_106091


namespace odd_function_expression_l106_106305

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 3 * x - 4 else - (x^2 - 3 * x - 4)

theorem odd_function_expression (x : ℝ) (h : x < 0) : 
  f x = -x^2 + 3 * x + 4 :=
by
  sorry

end odd_function_expression_l106_106305


namespace teacher_age_l106_106397

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (new_avg_with_teacher : ℕ) (num_total : ℕ) 
  (total_age_students : ℕ)
  (h1 : avg_age_students = 10)
  (h2 : num_students = 15)
  (h3 : new_avg_with_teacher = 11)
  (h4 : num_total = 16)
  (h5 : total_age_students = num_students * avg_age_students) :
  num_total * new_avg_with_teacher - total_age_students = 26 :=
by sorry

end teacher_age_l106_106397


namespace volume_water_needed_l106_106930

noncomputable def radius_sphere : ℝ := 0.5
noncomputable def radius_cylinder : ℝ := 1
noncomputable def height_cylinder : ℝ := 2

theorem volume_water_needed :
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
  let total_volume_spheres := 4 * volume_sphere
  let volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  volume_cylinder - total_volume_spheres = (4 * Real.pi) / 3 :=
by
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
  let total_volume_spheres := 4 * volume_sphere
  let volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  have h : volume_cylinder - total_volume_spheres = (4 * Real.pi) / 3 := sorry
  exact h

end volume_water_needed_l106_106930


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l106_106687

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l106_106687


namespace find_formula_and_range_l106_106915

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := 4^x + a * 2^x + b

theorem find_formula_and_range
  (a b : ℝ)
  (h₀ : f 0 a b = 1)
  (h₁ : f (-1) a b = -5 / 4) :
  f x (-3) 3 = 4^x - 3 * 2^x + 3 ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ 2 → 1 ≤ f x (-3) 3 ∧ f x (-3) 3 ≤ 25) :=
by
  sorry

end find_formula_and_range_l106_106915


namespace supplement_twice_angle_l106_106313

theorem supplement_twice_angle (α : ℝ) (h : 180 - α = 2 * α) : α = 60 := by
  admit -- This is a placeholder for the actual proof

end supplement_twice_angle_l106_106313


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l106_106688

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l106_106688


namespace base_b_of_200_has_5_digits_l106_106517

theorem base_b_of_200_has_5_digits : ∃ (b : ℕ), (b^4 ≤ 200) ∧ (200 < b^5) ∧ (b = 3) := by
  sorry

end base_b_of_200_has_5_digits_l106_106517


namespace sam_money_left_l106_106223

-- Assuming the cost per dime and quarter
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Given conditions
def dimes : ℕ := 19
def quarters : ℕ := 6
def cost_per_candy_bar_in_dimes : ℕ := 3
def candy_bars : ℕ := 4
def lollipops : ℕ := 1

-- Calculate the initial money in cents
def initial_money : ℕ := (dimes * dime_value) + (quarters * quarter_value)

-- Calculate the cost of candy bars in cents
def candy_bars_cost : ℕ := candy_bars * cost_per_candy_bar_in_dimes * dime_value

-- Calculate the cost of lollipops in cents
def lollipop_cost : ℕ := lollipops * quarter_value

-- Calculate the total cost of purchases in cents
def total_cost : ℕ := candy_bars_cost + lollipop_cost

-- Calculate the final money left in cents
def final_money : ℕ := initial_money - total_cost

-- Theorem to prove
theorem sam_money_left : final_money = 195 := by
  sorry

end sam_money_left_l106_106223


namespace cubical_box_edge_length_l106_106127

noncomputable def edge_length_of_box_in_meters : ℝ :=
  let number_of_cubes := 999.9999999999998
  let edge_length_cube_cm := 10
  let volume_cube_cm := edge_length_cube_cm^3
  let total_volume_box_cm := volume_cube_cm * number_of_cubes
  let total_volume_box_meters := total_volume_box_cm / (100^3)
  (total_volume_box_meters)^(1/3)

theorem cubical_box_edge_length :
  edge_length_of_box_in_meters = 1 := 
sorry

end cubical_box_edge_length_l106_106127


namespace f_prime_at_1_l106_106757

def f (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 10 * x - 5

theorem f_prime_at_1 : (deriv f 1) = 11 :=
by
  sorry

end f_prime_at_1_l106_106757


namespace power_inequality_l106_106760

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) := 
by 
  sorry

end power_inequality_l106_106760


namespace current_women_count_l106_106964

variable (x : ℕ) -- Let x be the common multiplier.
variable (initial_men : ℕ := 4 * x)
variable (initial_women : ℕ := 5 * x)

-- Conditions
variable (men_after_entry : ℕ := initial_men + 2)
variable (women_after_leave : ℕ := initial_women - 3)
variable (current_women : ℕ := 2 * women_after_leave)
variable (current_men : ℕ := 14)

-- Theorem statement
theorem current_women_count (h : men_after_entry = current_men) : current_women = 24 := by
  sorry

end current_women_count_l106_106964


namespace weekly_goal_l106_106259

theorem weekly_goal (a : ℕ) (d : ℕ) (n : ℕ) (h1 : a = 20) (h2 : d = 5) (h3 : n = 5) :
  ∑ i in finset.range n, a + i * d = 150 :=
by
  sorry

end weekly_goal_l106_106259


namespace chocolates_sold_l106_106186

theorem chocolates_sold (C S : ℝ) (n : ℕ) (h1 : 165 * C = n * S) (h2 : ((S - C) / C) * 100 = 10) : n = 150 :=
by
  sorry

end chocolates_sold_l106_106186


namespace sequence_of_numbers_exists_l106_106898

theorem sequence_of_numbers_exists :
  ∃ (a b : ℤ), (a + 2 * b > 0) ∧ (7 * a + 13 * b < 0) :=
sorry

end sequence_of_numbers_exists_l106_106898


namespace triangle_incircle_ratio_l106_106139

theorem triangle_incircle_ratio (r s q : ℝ) (h1 : r + s = 8) (h2 : r < s) (h3 : r + q = 13) (h4 : s + q = 17) (h5 : 8 + 13 > 17 ∧ 8 + 17 > 13 ∧ 13 + 17 > 8):
  r / s = 1 / 3 := by sorry

end triangle_incircle_ratio_l106_106139


namespace distance_apart_after_skating_l106_106001

theorem distance_apart_after_skating :
  let Ann_speed := 6 -- Ann's speed in miles per hour
  let Glenda_speed := 8 -- Glenda's speed in miles per hour
  let skating_time := 3 -- Time spent skating in hours
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  Total_Distance = 42 :=
by
  let Ann_speed := 6
  let Glenda_speed := 8
  let skating_time := 3
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  sorry

end distance_apart_after_skating_l106_106001


namespace smallest_n_l106_106556

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l106_106556


namespace total_cost_after_discounts_l106_106703

theorem total_cost_after_discounts 
    (price_iphone : ℝ)
    (discount_iphone : ℝ)
    (price_iwatch : ℝ)
    (discount_iwatch : ℝ)
    (cashback_percentage : ℝ) :
    (price_iphone = 800) →
    (discount_iphone = 0.15) →
    (price_iwatch = 300) →
    (discount_iwatch = 0.10) →
    (cashback_percentage = 0.02) →
    let discounted_iphone := price_iphone * (1 - discount_iphone),
        discounted_iwatch := price_iwatch * (1 - discount_iwatch),
        total_discounted := discounted_iphone + discounted_iwatch,
        cashback := total_discounted * cashback_percentage 
    in total_discounted - cashback = 931 :=
by {
  intros,
  sorry
}

end total_cost_after_discounts_l106_106703


namespace melissa_solves_equation_l106_106763

theorem melissa_solves_equation : 
  ∃ b c : ℤ, (∀ x : ℝ, x^2 - 6 * x + 9 = 0 ↔ (x + b)^2 = c) ∧ b + c = -3 :=
by
  sorry

end melissa_solves_equation_l106_106763


namespace smallest_positive_integer_n_l106_106565

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l106_106565


namespace travel_time_on_third_day_l106_106990

-- Definitions based on conditions
def speed_first_day : ℕ := 5
def time_first_day : ℕ := 7
def distance_first_day : ℕ := speed_first_day * time_first_day

def speed_second_day_part1 : ℕ := 6
def time_second_day_part1 : ℕ := 6
def distance_second_day_part1 : ℕ := speed_second_day_part1 * time_second_day_part1

def speed_second_day_part2 : ℕ := 3
def time_second_day_part2 : ℕ := 3
def distance_second_day_part2 : ℕ := speed_second_day_part2 * time_second_day_part2

def distance_second_day : ℕ := distance_second_day_part1 + distance_second_day_part2
def total_distance_first_two_days : ℕ := distance_first_day + distance_second_day

def total_distance : ℕ := 115
def distance_third_day : ℕ := total_distance - total_distance_first_two_days

def speed_third_day : ℕ := 7
def time_third_day : ℕ := distance_third_day / speed_third_day

-- The statement to be proven
theorem travel_time_on_third_day : time_third_day = 5 := by
  sorry

end travel_time_on_third_day_l106_106990


namespace weights_system_l106_106818

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end weights_system_l106_106818


namespace intersection_A_B_l106_106761

def setA : Set ℝ := {x | x^2 - 1 > 0}
def setB : Set ℝ := {x | Real.log x / Real.log 2 < 1}

theorem intersection_A_B :
  {x | x ∈ setA ∧ x ∈ setB} = {x | 1 < x ∧ x < 2} :=
sorry

end intersection_A_B_l106_106761


namespace partnership_investment_l106_106411

theorem partnership_investment (A B C : ℕ) (x m : ℝ) 
    (H1 : B = 2 * A) 
    (H2 : C = 3 * A) 
    (total_annual_gain A_share : ℝ) 
    (H3 : total_annual_gain = 21000) 
    (H4 : A_share = 7000) 
    (investment_ratio : (A * 12) / (A * 12 + B * 6 + C * (12 - m)) = 1 / 3) :
  m = 8 :=
sorry

end partnership_investment_l106_106411


namespace mike_peaches_eq_120_l106_106363

def original_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def total_peaches (orig : ℝ) (picked : ℝ) : ℝ := orig + picked

theorem mike_peaches_eq_120 : total_peaches original_peaches picked_peaches = 120.0 := 
by
  sorry

end mike_peaches_eq_120_l106_106363


namespace find_positive_x_l106_106172

theorem find_positive_x (x y z : ℝ) 
  (h1 : x * y = 15 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 2 * y - 4 * z)
  (h3 : x * z = 56 - 5 * x - 6 * z) : x = 8 := 
sorry

end find_positive_x_l106_106172


namespace john_initial_investment_in_alpha_bank_is_correct_l106_106902

-- Definition of the problem conditions
def initial_investment : ℝ := 2000
def alpha_rate : ℝ := 0.04
def beta_rate : ℝ := 0.06
def final_amount : ℝ := 2398.32
def years : ℕ := 3

-- Alpha Bank growth factor after 3 years
def alpha_growth_factor : ℝ := (1 + alpha_rate) ^ years

-- Beta Bank growth factor after 3 years
def beta_growth_factor : ℝ := (1 + beta_rate) ^ years

-- The main theorem
theorem john_initial_investment_in_alpha_bank_is_correct (x : ℝ) 
  (hx : x * alpha_growth_factor + (initial_investment - x) * beta_growth_factor = final_amount) : 
  x = 246.22 :=
sorry

end john_initial_investment_in_alpha_bank_is_correct_l106_106902


namespace pizza_area_increase_l106_106885

theorem pizza_area_increase 
  (r : ℝ) 
  (A_medium A_large : ℝ) 
  (h_medium_area : A_medium = Real.pi * r^2)
  (h_large_area : A_large = Real.pi * (1.40 * r)^2) : 
  ((A_large - A_medium) / A_medium) * 100 = 96 := 
by 
  sorry

end pizza_area_increase_l106_106885


namespace contrapositive_of_neg_and_inverse_l106_106457

theorem contrapositive_of_neg_and_inverse (p r s : Prop) (h1 : r = ¬p) (h2 : s = ¬r) : s = (¬p → false) :=
by
  -- We have that r = ¬p
  have hr : r = ¬p := h1
  -- And we have that s = ¬r
  have hs : s = ¬r := h2
  -- Now we need to show that s is the contrapositive of p, which is ¬p → false
  sorry

end contrapositive_of_neg_and_inverse_l106_106457


namespace min_value_is_8_plus_4_sqrt_3_l106_106173

noncomputable def min_value_of_expression (a b : ℝ) : ℝ :=
  2 / a + 1 / b

theorem min_value_is_8_plus_4_sqrt_3 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + 2 * b = 1) :
  min_value_of_expression a b = 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_is_8_plus_4_sqrt_3_l106_106173


namespace volume_of_cuboctahedron_l106_106875

def points (i j : ℕ) (A : ℕ → ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x0, y0, z0) := A 0
  let (xi, yi, zi) := A i
  let (xj, yj, zj) := A j
  (xi - xj, yi - yj, zi - zj)

def is_cuboctahedron (points_set : Set (ℝ × ℝ × ℝ)) : Prop :=
  -- Insert specific conditions that define a cuboctahedron
  sorry

theorem volume_of_cuboctahedron : 
  let A := fun 
    | 0 => (0, 0, 0)
    | 1 => (1, 0, 0)
    | 2 => (0, 1, 0)
    | 3 => (0, 0, 1)
    | _ => (0, 0, 0)
  let P_ij := 
    {p | ∃ i j : ℕ, i ≠ j ∧ p = points i j A}
  ∃ v : ℝ, is_cuboctahedron P_ij ∧ v = 10 / 3 :=
sorry

end volume_of_cuboctahedron_l106_106875


namespace math_problem_l106_106143
open Real

noncomputable def problem_statement : Prop :=
  let a := 99
  let b := 3
  let c := 20
  let area := (99 * sqrt 3) / 20
  a + b + c = 122 ∧ 
  ∃ (AB: ℝ) (QR: ℝ), AB = 14 ∧ QR = 3 * sqrt 3 ∧ area = (1 / 2) * QR * (QR / (2 * (sqrt 3 / 2))) * (sqrt 3 / 2)

theorem math_problem : problem_statement := by
  sorry

end math_problem_l106_106143


namespace domain_of_f_univ_l106_106552

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 1)^(1 / 3) + (9 - x^2)^(1 / 3)

theorem domain_of_f_univ : ∀ x : ℝ, true :=
by
  intro x
  sorry

end domain_of_f_univ_l106_106552


namespace part1_part2_l106_106171

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem part1 (a : ℝ) (h : (2 * a - (a + 2) + 1) = 0) : a = 1 :=
by
  sorry

theorem part2 (a x : ℝ) (ha : a ≥ 1) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : (2 * a * x - (a + 2) + 1 / x) ≥ 0 :=
by
  sorry

end part1_part2_l106_106171


namespace range_of_a_l106_106306

variable (a : ℝ) (x : ℝ) (x₀ : ℝ)

def p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ (x₀ : ℝ), ∃ (x : ℝ), x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l106_106306


namespace profit_percentage_example_l106_106338

noncomputable def selling_price : ℝ := 100
noncomputable def cost_price (sp : ℝ) : ℝ := 0.75 * sp
noncomputable def profit (sp cp : ℝ) : ℝ := sp - cp
noncomputable def profit_percentage (profit cp : ℝ) : ℝ := (profit / cp) * 100

theorem profit_percentage_example :
  profit_percentage (profit selling_price (cost_price selling_price)) (cost_price selling_price) = 33.33 :=
by
  -- Proof will go here
  sorry

end profit_percentage_example_l106_106338


namespace sin_beta_acute_l106_106332

theorem sin_beta_acute (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcosα : Real.cos α = 4 / 5)
  (hcosαβ : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end sin_beta_acute_l106_106332


namespace min_value_4x_plus_inv_l106_106262

noncomputable def min_value_function (x : ℝ) := 4 * x + 1 / (4 * x - 5)

theorem min_value_4x_plus_inv (x : ℝ) (h : x > 5 / 4) : min_value_function x = 7 :=
sorry

end min_value_4x_plus_inv_l106_106262


namespace solve_linear_eqns_x3y2z_l106_106207

theorem solve_linear_eqns_x3y2z (x y z : ℤ) 
  (h1 : x - 3 * y + 2 * z = 1) 
  (h2 : 2 * x + y - 5 * z = 7) : 
  z = 4 ^ 111 := 
sorry

end solve_linear_eqns_x3y2z_l106_106207


namespace count_triangles_in_3x3_grid_l106_106449

/--
In a 3x3 grid of dots, the number of triangles formed by connecting the dots is 20.
-/
def triangles_in_3x3_grid : Prop :=
  let num_rows := 3
  let num_cols := 3
  let total_triangles := 20
  ∃ (n : ℕ), n = total_triangles ∧ n = 20

theorem count_triangles_in_3x3_grid : triangles_in_3x3_grid :=
by {
  -- Insert the proof here
  sorry
}

end count_triangles_in_3x3_grid_l106_106449


namespace data_division_into_groups_l106_106521

-- Conditions
def data_set_size : Nat := 90
def max_value : Nat := 141
def min_value : Nat := 40
def class_width : Nat := 10

-- Proof statement
theorem data_division_into_groups : (max_value - min_value) / class_width + 1 = 11 :=
by
  sorry

end data_division_into_groups_l106_106521


namespace initial_pieces_count_l106_106040

theorem initial_pieces_count (people : ℕ) (pieces_per_person : ℕ) (leftover_pieces : ℕ) :
  people = 6 → pieces_per_person = 7 → leftover_pieces = 3 → people * pieces_per_person + leftover_pieces = 45 :=
by
  intros h_people h_pieces_per_person h_leftover_pieces
  sorry

end initial_pieces_count_l106_106040


namespace range_of_c_l106_106429

-- Definitions of p and q based on conditions
def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (c > 1 / 2)

-- The theorem states the required condition on c
theorem range_of_c (c : ℝ) (h : c > 0) :
  ¬(p c ∧ q c) ∧ (p c ∨ q c) ↔ (0 < c ∧ c ≤ 1 / 2) ∨ (c ≥ 1) :=
sorry

end range_of_c_l106_106429


namespace gcd_50421_35343_l106_106712

theorem gcd_50421_35343 : Int.gcd 50421 35343 = 23 := by
  sorry

end gcd_50421_35343_l106_106712


namespace quadratic_root_sum_product_l106_106316

theorem quadratic_root_sum_product (m n : ℝ)
  (h1 : m + n = 4)
  (h2 : m * n = -1) :
  m + n - m * n = 5 :=
sorry

end quadratic_root_sum_product_l106_106316


namespace paco_cookies_proof_l106_106220

-- Define the initial conditions
def initial_cookies : Nat := 40
def cookies_eaten : Nat := 2
def cookies_bought : Nat := 37
def free_cookies_per_bought : Nat := 2

-- Define the total number of cookies after all operations
def total_cookies (initial_cookies cookies_eaten cookies_bought free_cookies_per_bought : Nat) : Nat :=
  let remaining_cookies := initial_cookies - cookies_eaten
  let free_cookies := cookies_bought * free_cookies_per_bought
  let cookies_from_bakery := cookies_bought + free_cookies
  remaining_cookies + cookies_from_bakery

-- The target statement that needs to be proved
theorem paco_cookies_proof : total_cookies initial_cookies cookies_eaten cookies_bought free_cookies_per_bought = 149 :=
by
  sorry

end paco_cookies_proof_l106_106220


namespace total_water_carried_l106_106132

/-- Define the capacities of the four tanks in each truck -/
def tank1_capacity : ℝ := 200
def tank2_capacity : ℝ := 250
def tank3_capacity : ℝ := 300
def tank4_capacity : ℝ := 350

/-- The total capacity of one truck -/
def total_truck_capacity : ℝ := tank1_capacity + tank2_capacity + tank3_capacity + tank4_capacity

/-- Define the fill percentages for each truck -/
def fill_percentage (truck_number : ℕ) : ℝ :=
if truck_number = 1 then 1
else if truck_number = 2 then 0.75
else if truck_number = 3 then 0.5
else if truck_number = 4 then 0.25
else 0

/-- Define the amounts of water each truck carries -/
def water_carried_by_truck (truck_number : ℕ) : ℝ :=
(fill_percentage truck_number) * total_truck_capacity

/-- Prove that the total amount of water the farmer can carry in his trucks is 2750 liters -/
theorem total_water_carried : 
  water_carried_by_truck 1 + water_carried_by_truck 2 + water_carried_by_truck 3 +
  water_carried_by_truck 4 + water_carried_by_truck 5 = 2750 :=
by sorry

end total_water_carried_l106_106132


namespace find_integer_n_cos_l106_106868

theorem find_integer_n_cos : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ (Real.cos (n * Real.pi / 180) = Real.cos (1124 * Real.pi / 180)) ∧ n = 44 := by
  sorry

end find_integer_n_cos_l106_106868


namespace increasing_on_interval_l106_106047

noncomputable def f (x a : ℝ) : ℝ := x^3 + a * x - 2

theorem increasing_on_interval (a : ℝ) : 
  (a ≥ -3) ↔ (∀ x, 1 ≤ x → 0 ≤ deriv (λ x : ℝ, f x a) x) :=
begin
  sorry
end

end increasing_on_interval_l106_106047


namespace hamburger_combinations_l106_106738

theorem hamburger_combinations : 
  let condiments := 10  -- Number of available condiments
  let patty_choices := 4 -- Number of meat patty options
  2^condiments * patty_choices = 4096 :=
by sorry

end hamburger_combinations_l106_106738


namespace arcsin_sqrt3_div_2_l106_106684

theorem arcsin_sqrt3_div_2 :
  ∃ θ : ℝ, θ ∈ Icc (-(Real.pi / 2)) (Real.pi / 2) ∧ Real.sin θ = (Real.sqrt 3) / 2 ∧ Real.arcsin ((Real.sqrt 3) / 2) = θ ∧ θ = (Real.pi / 3) :=
by
  sorry

end arcsin_sqrt3_div_2_l106_106684


namespace solution_exists_l106_106956

namespace EquationSystem
-- Given the conditions of the equation system:
def eq1 (a b c d : ℝ) := a * b + a * c = 3 * b + 3 * c
def eq2 (a b c d : ℝ) := b * c + b * d = 5 * c + 5 * d
def eq3 (a b c d : ℝ) := a * c + c * d = 7 * a + 7 * d
def eq4 (a b c d : ℝ) := a * d + b * d = 9 * a + 9 * b

-- We need to prove that the solutions are as described:
theorem solution_exists (a b c d : ℝ) :
  eq1 a b c d → eq2 a b c d → eq3 a b c d → eq4 a b c d →
  (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) ∨ ∃ t : ℝ, a = t ∧ b = -t ∧ c = t ∧ d = -t :=
  by
    sorry
end EquationSystem

end solution_exists_l106_106956


namespace Andrew_spent_1395_dollars_l106_106872

-- Define the conditions
def cookies_per_day := 3
def cost_per_cookie := 15
def days_in_may := 31

-- Define the calculation
def total_spent := cookies_per_day * cost_per_cookie * days_in_may

-- State the theorem
theorem Andrew_spent_1395_dollars :
  total_spent = 1395 := 
by
  sorry

end Andrew_spent_1395_dollars_l106_106872


namespace sequence_is_arithmetic_l106_106072

noncomputable def f (x : ℝ) (n : ℕ) : ℝ := (x - 1)^2 + n

def a_n (n : ℕ) : ℝ := n

def b_n (n : ℕ) : ℝ := n + 4

def c_n (n : ℕ) : ℝ := (b_n n)^2 - (a_n n) * (b_n n)

theorem sequence_is_arithmetic :
  ∀ n : ℕ, c_n (n + 1) - c_n n = 4 := 
by
  intros n
  sorry

end sequence_is_arithmetic_l106_106072


namespace find_angle_l106_106866

theorem find_angle :
  ∃ (x : ℝ), (90 - x = 0.4 * (180 - x)) → x = 30 :=
by
  sorry

end find_angle_l106_106866


namespace ratio_problem_l106_106570

-- Given condition: a, b, c are in the ratio 2:3:4
theorem ratio_problem (a b c : ℝ) (h1 : a / b = 2 / 3) (h2 : a / c = 2 / 4) : 
  (a - b + c) / b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end ratio_problem_l106_106570


namespace find_incorrect_expression_l106_106041

variable {x y : ℚ}

theorem find_incorrect_expression
  (h : x / y = 5 / 6) :
  ¬ (
    (x + 3 * y) / x = 23 / 5
  ) := by
  sorry

end find_incorrect_expression_l106_106041


namespace correct_product_l106_106600

theorem correct_product (a b : ℚ) (calc_incorrect : a = 52 ∧ b = 735)
                        (incorrect_product : a * b = 38220) :
  (0.52 * 7.35 = 3.822) :=
by
  sorry

end correct_product_l106_106600


namespace rhombus_side_length_l106_106591

theorem rhombus_side_length (a b m : ℝ) 
  (h1 : a + b = 10) 
  (h2 : a * b = 22) 
  (h3 : a^2 - 10 * a + m = 0) 
  (h4 : b^2 - 10 * b + m = 0) 
  (h_area : 1/2 * a * b = 11) : 
  ∃ s : ℝ, s = √14 := 
sorry

end rhombus_side_length_l106_106591


namespace rhombus_side_length_l106_106593

noncomputable def quadratic_roots (a b c : ℝ) := 
  (b * b - 4 * a * c) ≥ 0

theorem rhombus_side_length (a b : ℝ) (m : ℝ)
  (h1 : quadratic_roots 1 (-10) m)
  (h2 : a + b = 10)
  (h3 : a * b = 22)
  (area : 0.5 * a * b = 11) :
  (1 / 2) * real.sqrt (a * a + b * b) = real.sqrt 14 :=
by
  sorry

end rhombus_side_length_l106_106593


namespace sum_of_coefficients_3x_minus_1_pow_7_l106_106355

theorem sum_of_coefficients_3x_minus_1_pow_7 :
  let f (x : ℕ) := (3 * x - 1) ^ 7
  (f 1) = 128 :=
by
  sorry

end sum_of_coefficients_3x_minus_1_pow_7_l106_106355


namespace calvin_haircut_goal_percentage_l106_106526

theorem calvin_haircut_goal_percentage :
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  (completed_haircuts / total_haircuts_needed) * 100 = 80 :=
by
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  show (completed_haircuts / total_haircuts_needed) * 100 = 80
  sorry

end calvin_haircut_goal_percentage_l106_106526


namespace max_true_statements_l106_106473

theorem max_true_statements :
  ∃ x : ℝ, 
  (0 < x ∧ x < 1) ∧ -- Statement 4
  (0 < x^3 ∧ x^3 < 1) ∧ -- Statement 1
  (0 < x - x^3 ∧ x - x^3 < 1) ∧ -- Statement 5
  ¬(x^3 > 1) ∧ -- Not Statement 2
  ¬(-1 < x ∧ x < 0) := -- Not Statement 3
sorry

end max_true_statements_l106_106473


namespace polynomial_a5_coefficient_l106_106883

noncomputable theory

theorem polynomial_a5_coefficient :
  let p := (X^2 - 2*X + 2) ^ 5 in
  p.coeff 5 = -592 :=
by
  sorry

end polynomial_a5_coefficient_l106_106883


namespace snickers_bars_needed_l106_106197

-- Definitions of the conditions
def points_needed : ℕ := 2000
def chocolate_bunny_points : ℕ := 100
def number_of_chocolate_bunnies : ℕ := 8
def snickers_points : ℕ := 25

-- Derived conditions
def points_from_bunnies : ℕ := number_of_chocolate_bunnies * chocolate_bunny_points
def remaining_points : ℕ := points_needed - points_from_bunnies

-- Statement to prove
theorem snickers_bars_needed : ∀ (n : ℕ), n = remaining_points / snickers_points → n = 48 :=
by 
  sorry

end snickers_bars_needed_l106_106197


namespace find_n_for_quadratic_roots_l106_106333

noncomputable def quadratic_root_properties (d c e n : ℝ) : Prop :=
  let A := (n + 2)
  let B := -((n + 2) * d + (n - 2) * c)
  let C := e * (n - 2)
  ∃ y1 y2 : ℝ, (A * y1 * y1 + B * y1 + C = 0) ∧ (A * y2 * y2 + B * y2 + C = 0) ∧ (y1 = -y2) ∧ (y1 + y2 = 0)

theorem find_n_for_quadratic_roots (d c e : ℝ) (h : d ≠ c) : 
  (quadratic_root_properties d c e (-2)) :=
sorry

end find_n_for_quadratic_roots_l106_106333


namespace gum_candy_ratio_l106_106202

theorem gum_candy_ratio
  (g c : ℝ)  -- let g be the cost of a stick of gum and c be the cost of a candy bar.
  (hc : c = 1.5)  -- the cost of each candy bar is $1.5
  (h_total_cost : 2 * g + 3 * c = 6)  -- total cost of 2 sticks of gum and 3 candy bars is $6
  : g / c = 1 / 2 := -- the ratio of the cost of gum to candy is 1:2
sorry

end gum_candy_ratio_l106_106202


namespace needed_angle_BPC_l106_106208

-- Definitions needed from problem conditions

variables (A B C D P Q M N : Type)
variables [geometry.AB A B] [geometry.CD C D]
variables [MidpointSegment M AB] [MidpointSegment N CD]
variables [CloserToSegment P BC Q]
variables [angle.MPN M P N (40 : angle)]

theorem needed_angle_BPC 
  (h_rect : Rectangle ABCD)
  (h_circles : CirclesWithDiameters A B C D P Q) :
  angle BPC = 80 :=
by
  sorry

end needed_angle_BPC_l106_106208


namespace misha_current_dollars_l106_106919

variable (x : ℕ)

def misha_needs_more : ℕ := 13
def total_amount : ℕ := 47

theorem misha_current_dollars : x = total_amount - misha_needs_more → x = 34 :=
by
  sorry

end misha_current_dollars_l106_106919


namespace log_eq_one_l106_106399

theorem log_eq_one (log : ℝ → ℝ) (h1 : ∀ a b, log (a ^ b) = b * log a) (h2 : ∀ a b, log (a * b) = log a + log b) :
  (log 5) ^ 2 + log 2 * log 50 = 1 :=
sorry

end log_eq_one_l106_106399


namespace desired_markup_percentage_l106_106138

theorem desired_markup_percentage
  (initial_price : ℝ) (markup_rate : ℝ) (wholesale_price : ℝ) (additional_increase : ℝ) 
  (h1 : initial_price = wholesale_price * (1 + markup_rate)) 
  (h2 : initial_price = 34) 
  (h3 : markup_rate = 0.70) 
  (h4 : additional_increase = 6) 
  : ( (initial_price + additional_increase - wholesale_price) / wholesale_price * 100 ) = 100 := 
by
  sorry

end desired_markup_percentage_l106_106138


namespace min_value_of_expression_min_value_achieved_l106_106160

noncomputable def f (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_of_expression : ∀ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6452.25 :=
by sorry

theorem min_value_achieved : ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6452.25 :=
by sorry

end min_value_of_expression_min_value_achieved_l106_106160


namespace min_value_xy_k_l106_106808

theorem min_value_xy_k (x y k : ℝ) : ∃ x y : ℝ, (xy - k)^2 + (x + y - 1)^2 = 1 := by
  sorry

end min_value_xy_k_l106_106808


namespace range_of_a_l106_106916

variable (f : ℝ → ℝ)
variable (a : ℝ)

theorem range_of_a (h1 : ∀ a : ℝ, (f (1 - 2 * a) / 2 ≥ f a))
                  (h2 : ∀ (x1 x2 : ℝ), x1 < x2 ∧ x1 + x2 ≠ 0 → f x1 > f x2) : a > (1 / 2) :=
by
  sorry

end range_of_a_l106_106916


namespace lauras_european_stamps_cost_l106_106658

def stamp_cost (count : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  count * cost_per_stamp

def total_stamps_cost (stamps80 : ℕ) (stamps90 : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  stamp_cost stamps80 cost_per_stamp + stamp_cost stamps90 cost_per_stamp

def european_stamps_cost_80_90 :=
  total_stamps_cost 10 12 0.09 + total_stamps_cost 18 16 0.07

theorem lauras_european_stamps_cost : european_stamps_cost_80_90 = 4.36 :=
by
  sorry

end lauras_european_stamps_cost_l106_106658


namespace lunks_needed_for_bananas_l106_106584

theorem lunks_needed_for_bananas :
  (7 : ℚ) / 4 * (20 * 3 / 5) = 21 :=
by
  sorry

end lunks_needed_for_bananas_l106_106584


namespace evaluate_expression_l106_106534

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l106_106534


namespace wayne_needs_30_more_blocks_l106_106947

def initial_blocks : ℕ := 9
def additional_blocks : ℕ := 6
def total_blocks : ℕ := initial_blocks + additional_blocks
def triple_total : ℕ := 3 * total_blocks

theorem wayne_needs_30_more_blocks :
  triple_total - total_blocks = 30 := by
  sorry

end wayne_needs_30_more_blocks_l106_106947


namespace alok_total_payment_l106_106671

def cost_of_chapatis : Nat := 16 * 6
def cost_of_rice_plates : Nat := 5 * 45
def cost_of_mixed_vegetable_plates : Nat := 7 * 70
def total_cost : Nat := cost_of_chapatis + cost_of_rice_plates + cost_of_mixed_vegetable_plates

theorem alok_total_payment :
  total_cost = 811 := by
  unfold total_cost
  unfold cost_of_chapatis
  unfold cost_of_rice_plates
  unfold cost_of_mixed_vegetable_plates
  calc
    16 * 6 + 5 * 45 + 7 * 70 = 96 + 5 * 45 + 7 * 70 := by rfl
                      ... = 96 + 225 + 7 * 70 := by rfl
                      ... = 96 + 225 + 490 := by rfl
                      ... = 96 + (225 + 490) := by rw Nat.add_assoc
                      ... = (96 + 225) + 490 := by rw Nat.add_assoc
                      ... = 321 + 490 := by rfl
                      ... = 811 := by rfl

end alok_total_payment_l106_106671


namespace percentage_increase_l106_106522

theorem percentage_increase (R W : ℕ) (hR : R = 36) (hW : W = 20) : 
  ((R - W) / W : ℚ) * 100 = 80 := 
by 
  sorry

end percentage_increase_l106_106522


namespace romanov_family_savings_l106_106512

theorem romanov_family_savings :
  let cost_multi_tariff_meter := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let night_consumption := 230
  let day_consumption := monthly_consumption - night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let yearly_cost_multi_tariff :=
    (night_consumption * night_rate * 12) +
    (day_consumption * day_rate * 12)
  let total_cost_multi_tariff :=
    cost_multi_tariff_meter + installation_cost + (yearly_cost_multi_tariff * 3)
  let yearly_cost_standard :=
    monthly_consumption * standard_rate * 12
  let total_cost_standard :=
    yearly_cost_standard * 3
  total_cost_standard - total_cost_multi_tariff = 3824 := 
by {
  sorry -- Proof goes here
}

end romanov_family_savings_l106_106512


namespace tile_coverage_fraction_l106_106421

structure Room where
  rect_length : ℝ
  rect_width : ℝ
  tri_base : ℝ
  tri_height : ℝ
  
structure Tiles where
  square_tiles : ℕ
  triangular_tiles : ℕ
  triangle_base : ℝ
  triangle_height : ℝ
  tile_area : ℝ
  triangular_tile_area : ℝ
  
noncomputable def fractionalTileCoverage (room : Room) (tiles : Tiles) : ℝ :=
  let rect_area := room.rect_length * room.rect_width
  let tri_area := (room.tri_base * room.tri_height) / 2
  let total_room_area := rect_area + tri_area
  let total_tile_area := (tiles.square_tiles * tiles.tile_area) + (tiles.triangular_tiles * tiles.triangular_tile_area)
  total_tile_area / total_room_area

theorem tile_coverage_fraction
  (room : Room) (tiles : Tiles)
  (h1 : room.rect_length = 12)
  (h2 : room.rect_width = 20)
  (h3 : room.tri_base = 10)
  (h4 : room.tri_height = 8)
  (h5 : tiles.square_tiles = 40)
  (h6 : tiles.triangular_tiles = 4)
  (h7 : tiles.tile_area = 1)
  (h8 : tiles.triangular_tile_area = (1 * 1) / 2) :
  fractionalTileCoverage room tiles = 3 / 20 :=
by 
  sorry

end tile_coverage_fraction_l106_106421


namespace find_m_l106_106887

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x - 5 * y + 20 = 0
def l2 (m x y : ℝ) : Prop := m * x + 2 * y - 10 = 0

-- Define the condition of perpendicularity
def lines_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Proving the value of m given the conditions
theorem find_m (m : ℝ) :
  (∃ x y : ℝ, l1 x y) → (∃ x y : ℝ, l2 m x y) → lines_perpendicular 2 (-5 : ℝ) m 2 → m = 5 :=
sorry

end find_m_l106_106887


namespace ramu_profit_percent_l106_106119

noncomputable def profitPercent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ) : ℝ :=
  ((selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost)) * 100

theorem ramu_profit_percent :
  profitPercent 42000 13000 61900 = 12.55 :=
by
  sorry

end ramu_profit_percent_l106_106119


namespace problem_statement_l106_106616

def f (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n - 3

theorem problem_statement : f (f (f 3)) = 31 :=
by
  sorry

end problem_statement_l106_106616


namespace father_cards_given_l106_106064

-- Defining the conditions
def Janessa_initial_cards : Nat := 4
def eBay_cards : Nat := 36
def bad_cards : Nat := 4
def dexter_cards : Nat := 29
def janessa_kept_cards : Nat := 20

-- Proving the number of cards father gave her
theorem father_cards_given : ∃ n : Nat, n = 13 ∧ (Janessa_initial_cards + eBay_cards - bad_cards + n = dexter_cards + janessa_kept_cards) := 
by
  sorry

end father_cards_given_l106_106064


namespace no_rational_solution_l106_106085

/-- Prove that the only rational solution to the equation x^3 + 3y^3 + 9z^3 = 9xyz is x = y = z = 0. -/
theorem no_rational_solution : ∀ (x y z : ℚ), x^3 + 3 * y^3 + 9 * z^3 = 9 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z h
  sorry

end no_rational_solution_l106_106085


namespace neg_p_l106_106045

open Nat -- Opening natural number namespace

-- Definition of the proposition p
def p := ∃ (m : ℕ), ∃ (k : ℕ), k * k = m * m + 1

-- Theorem statement for the negation of proposition p
theorem neg_p : ¬p ↔ ∀ (m : ℕ), ¬ ∃ (k : ℕ), k * k = m * m + 1 :=
by {
  -- Provide the proof here
  sorry
}

end neg_p_l106_106045


namespace evaluate_expression_l106_106545

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l106_106545


namespace ratio_josh_to_selena_l106_106481

def total_distance : ℕ := 36
def selena_distance : ℕ := 24

def josh_distance (td sd : ℕ) : ℕ := td - sd

theorem ratio_josh_to_selena : (josh_distance total_distance selena_distance) / selena_distance = 1 / 2 :=
by
  sorry

end ratio_josh_to_selena_l106_106481


namespace find_p_l106_106407

-- Define the conditions for the problem.
-- Random variable \xi follows binomial distribution B(n, p).
axiom binomial_distribution (n : ℕ) (p : ℝ) : Type
variables (ξ : binomial_distribution n p)

-- Given conditions: Eξ = 300 and Dξ = 200.
axiom Eξ (ξ : binomial_distribution n p) : ℝ
axiom Dξ (ξ : binomial_distribution n p) : ℝ

-- Given realizations of expectations and variance.
axiom h1 : Eξ ξ = 300
axiom h2 : Dξ ξ = 200

-- Prove that p = 1/3
theorem find_p (n : ℕ) (p : ℝ) (ξ : binomial_distribution n p)
  (h1 : Eξ ξ = 300) (h2 : Dξ ξ = 200) : p = 1 / 3 :=
sorry

end find_p_l106_106407


namespace max_t_subsets_of_base_set_l106_106206

theorem max_t_subsets_of_base_set (n : ℕ)
  (A : Fin (2 * n + 1) → Set (Fin n))
  (h : ∀ i j k : Fin (2 * n + 1), i < j → j < k → (A i ∩ A k) ⊆ A j) : 
  ∃ t : ℕ, t = 2 * n + 1 :=
by
  sorry

end max_t_subsets_of_base_set_l106_106206


namespace farmer_earns_from_runt_pig_l106_106828

def average_bacon_per_pig : ℕ := 20
def price_per_pound : ℕ := 6
def runt_pig_bacon : ℕ := average_bacon_per_pig / 2
def total_money_made (bacon_pounds : ℕ) (price_per_pound : ℕ) : ℕ := bacon_pounds * price_per_pound

theorem farmer_earns_from_runt_pig :
  total_money_made runt_pig_bacon price_per_pound = 60 :=
sorry

end farmer_earns_from_runt_pig_l106_106828


namespace solve_equation_l106_106087

theorem solve_equation (x : ℝ) : 
  (x - 1) / 2 - (2 * x + 3) / 3 = 1 ↔ 3 * (x - 1) - 2 * (2 * x + 3) = 6 := 
sorry

end solve_equation_l106_106087


namespace solutions_to_equation_l106_106709

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 10*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 12*x - 8)) = 0

theorem solutions_to_equation :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = -19 ∨ x = (5 + Real.sqrt 57) / 2 ∨ x = (5 - Real.sqrt 57) / 2) :=
sorry

end solutions_to_equation_l106_106709


namespace count_integer_solutions_xyz_eq_2009_l106_106447

theorem count_integer_solutions_xyz_eq_2009 :
  let equations2009 := [
    (1, 1, 2009),
    (1, 7, 287),
    (7, 7, 41),
    (1, 49, 41)
  ]
  ∃ n : ℕ, n = 72 ∧
  ∀ (x y z : ℤ), (x, y, z) ∈ equations2009 ∨ (x, z, y) ∈ equations2009 ∨ (y, x, z) ∈ equations2009 ∨
    (y, z, x) ∈ equations2009 ∨ (z, x, y) ∈ equations2009 ∨ (z, y, x) ∈ equations2009 ∨ 
    ((-x), (-y), z) ∈ equations2009 ∨ ((-x), y, (-z)) ∈ equations2009 ∨ (x, (-y), (-z)) ∈ equations2009 ∨ 
    ((-y), (-z), x) ∈ equations2009 ∨ ((-y), z, (-x)) ∈ equations2009 ∨ (y, (-z), (-x)) ∈ equations2009 ∨ 
    ((-z), (-x), y) ∈ equations2009 ∨ ((-z), x, (-y)) ∈ equations2009 ∨ (z, (-x), (-y)) ∈ equations2009 → 
    x * y * z = 2009 := 
by
  intros
  let equations2009 := [
    (1, 1, 2009),
    (1, 7, 287),
    (7, 7, 41),
    (1, 49, 41)
  ]
  existsi (72 : ℕ)
  split
  · refl
  intro x y z H
  cases H;
  { repeat { cases H; 
      { repeat {
          cases H with (eq : xyz = _),
          { 
            rw eq,
            simp,
            sorry
          } }
        } } }

end count_integer_solutions_xyz_eq_2009_l106_106447


namespace chinese_mathematical_system_l106_106057

noncomputable def problem_statement : Prop :=
  ∃ (x : ℕ) (y : ℕ),
    7 * x + 7 = y ∧ 
    9 * (x - 1) = y

theorem chinese_mathematical_system :
  problem_statement := by
  sorry

end chinese_mathematical_system_l106_106057


namespace abs_inequality_l106_106781

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l106_106781


namespace divisibility_condition_l106_106020

theorem divisibility_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ab ∣ (a^2 + b^2 - a - b + 1) → (a = 1 ∧ b = 1) :=
by sorry

end divisibility_condition_l106_106020


namespace flowers_bouquets_l106_106604

theorem flowers_bouquets (tulips: ℕ) (roses: ℕ) (extra: ℕ) (total: ℕ) (used_for_bouquets: ℕ) 
(h1: tulips = 36) 
(h2: roses = 37) 
(h3: extra = 3) 
(h4: total = tulips + roses)
(h5: used_for_bouquets = total - extra) :
used_for_bouquets = 70 := by
  sorry

end flowers_bouquets_l106_106604


namespace total_miles_l106_106529

theorem total_miles (miles_Darius : Int) (miles_Julia : Int) (h1 : miles_Darius = 679) (h2 : miles_Julia = 998) :
  miles_Darius + miles_Julia = 1677 :=
by
  sorry

end total_miles_l106_106529


namespace Theresa_helper_hours_l106_106389

theorem Theresa_helper_hours :
  ∃ x : ℕ, (7 + 10 + 8 + 11 + 9 + 7 + x) / 7 = 9 ∧ x ≥ 10 := by
  sorry

end Theresa_helper_hours_l106_106389


namespace num_positive_four_digit_integers_of_form_xx75_l106_106882

theorem num_positive_four_digit_integers_of_form_xx75 : 
  ∃ n : ℕ, n = 90 ∧ ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → (∃ x: ℕ, x = 1000 * a + 100 * b + 75 ∧ 1000 ≤ x ∧ x < 10000) → n = 90 :=
sorry

end num_positive_four_digit_integers_of_form_xx75_l106_106882


namespace problem_l106_106607

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) (α1 : ℝ) (α2 : ℝ) :=
  m * Real.sin (Real.pi * x + α1) + n * Real.cos (Real.pi * x + α2)

variables (m n α1 α2 : ℝ) (h_m : m ≠ 0) (h_n : n ≠ 0) (h_α1 : α1 ≠ 0) (h_α2 : α2 ≠ 0)

theorem problem (h : f 2008 m n α1 α2 = 1) : f 2009 m n α1 α2 = -1 :=
  sorry

end problem_l106_106607


namespace calc_expr_eq_l106_106679

-- Define the polynomial and expression
def expr (x : ℝ) : ℝ := x * (x * (x * (3 - 2 * x) - 4) + 8) + 3 * x^2

theorem calc_expr_eq (x : ℝ) : expr x = -2 * x^4 + 3 * x^3 - x^2 + 8 * x := 
by
  sorry

end calc_expr_eq_l106_106679


namespace solve_inequality_l106_106793

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l106_106793


namespace weights_system_l106_106819

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end weights_system_l106_106819


namespace find_x_value_l106_106454

noncomputable def check_x (x : ℝ) : Prop :=
  (0 < x) ∧ (Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (10 * x) = 10)

theorem find_x_value (x : ℝ) (h : check_x x) : x = 1 / 6 :=
by 
  sorry

end find_x_value_l106_106454


namespace range_of_b_l106_106617

noncomputable def f (b x : ℝ) : ℝ := -x^3 + b * x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → -3 * x^2 + b ≥ 0) ↔ b ≥ 3 := sorry

end range_of_b_l106_106617


namespace paul_diner_total_cost_l106_106675

/-- At Paul's Diner, sandwiches cost $5 each and sodas cost $3 each. If a customer buys
more than 4 sandwiches, they receive a $10 discount on the total bill. Calculate the total
cost if a customer purchases 6 sandwiches and 3 sodas. -/
def totalCost (num_sandwiches num_sodas : ℕ) : ℕ :=
  let sandwich_cost := 5
  let soda_cost := 3
  let discount := if num_sandwiches > 4 then 10 else 0
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) - discount

theorem paul_diner_total_cost : totalCost 6 3 = 29 :=
by
  sorry

end paul_diner_total_cost_l106_106675


namespace num_lists_correct_l106_106121

def num_balls : ℕ := 18
def num_draws : ℕ := 4

theorem num_lists_correct : (num_balls ^ num_draws) = 104976 :=
by
  sorry

end num_lists_correct_l106_106121


namespace complement_of_A_in_U_l106_106324

def U : Set ℝ := {x | x ≤ 1}
def A : Set ℝ := {x | x < 0}

theorem complement_of_A_in_U : (U \ A) = {x | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end complement_of_A_in_U_l106_106324


namespace ilya_arithmetic_l106_106346

theorem ilya_arithmetic (v t : ℝ) (h : v + t = v * t ∧ v + t = v / t) : False :=
by
  sorry

end ilya_arithmetic_l106_106346


namespace train_platform_length_l106_106124

theorem train_platform_length (train_length : ℕ) (platform_crossing_time : ℕ) (pole_crossing_time : ℕ) (length_of_platform : ℕ) :
  train_length = 300 →
  platform_crossing_time = 27 →
  pole_crossing_time = 18 →
  ((train_length * platform_crossing_time / pole_crossing_time) = train_length + length_of_platform) →
  length_of_platform = 150 :=
by
  intros h_train_length h_platform_time h_pole_time h_eq
  -- Proof omitted
  sorry

end train_platform_length_l106_106124


namespace triangle_ABC_right_angled_l106_106154
open Real

theorem triangle_ABC_right_angled (A B C : ℝ) (a b c : ℝ)
  (h1 : cos (2 * A) - cos (2 * B) = 2 * sin C ^ 2)
  (h2 : a = sin A) (h3 : b = sin B) (h4 : c = sin C)
  : a^2 + c^2 = b^2 :=
by sorry

end triangle_ABC_right_angled_l106_106154


namespace find_x_l106_106394

theorem find_x (x : ℤ) (h : x + -27 = 30) : x = 57 :=
sorry

end find_x_l106_106394


namespace evaluate_expression_l106_106543

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l106_106543


namespace probability_complement_l106_106941

theorem probability_complement (p : ℝ) (h : p = 0.997) : 1 - p = 0.003 :=
by
  rw [h]
  norm_num

end probability_complement_l106_106941


namespace spinner_probability_l106_106982

theorem spinner_probability (P_D P_E : ℝ) (hD : P_D = 2/5) (hE : P_E = 1/5) 
  (hTotal : P_D + P_E + P_F = 1) : P_F = 2/5 :=
by
  sorry

end spinner_probability_l106_106982


namespace product_of_two_integers_l106_106655

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 18) (h2 : x^2 - y^2 = 36) : x * y = 80 :=
by
  sorry

end product_of_two_integers_l106_106655


namespace largest_integral_solution_l106_106713

noncomputable def largest_integral_value : ℤ :=
  let a : ℚ := 1 / 4
  let b : ℚ := 7 / 11 
  let lower_bound : ℚ := 7 * a
  let upper_bound : ℚ := 7 * b
  let x := 3  -- The largest integral value within the bounds
  x

-- A theorem to prove that x = 3 satisfies the inequality conditions and is the largest integer.
theorem largest_integral_solution (x : ℤ) (h₁ : 1 / 4 < x / 7) (h₂ : x / 7 < 7 / 11) : x = 3 := by
  sorry

end largest_integral_solution_l106_106713


namespace count_numbers_leaving_remainder_7_when_divided_by_59_l106_106329

theorem count_numbers_leaving_remainder_7_when_divided_by_59 :
  ∃ n, n = 3 ∧ ∀ k, (k ∣ 52) ∧ (k > 7) ↔ k ∈ {13, 26, 52} :=
by
  sorry

end count_numbers_leaving_remainder_7_when_divided_by_59_l106_106329


namespace john_makes_money_l106_106066

-- Definitions of the conditions
def num_cars := 5
def time_first_3_cars := 3 * 40 -- 3 cars each take 40 minutes
def time_remaining_car := 40 * 3 / 2 -- Each remaining car takes 50% longer
def time_remaining_cars := 2 * time_remaining_car -- 2 remaining cars
def total_time_min := time_first_3_cars + time_remaining_cars
def total_time_hr := total_time_min / 60 -- Convert total time from minutes to hours
def rate_per_hour := 20

-- Theorem statement
theorem john_makes_money : total_time_hr * rate_per_hour = 80 := by
  sorry

end john_makes_money_l106_106066


namespace roots_sum_and_product_l106_106314

theorem roots_sum_and_product (m n : ℝ) (h : (x^2 - 4*x - 1 = 0).roots = [m, n]) : m + n - m*n = 5 :=
sorry

end roots_sum_and_product_l106_106314


namespace evaluate_expression_l106_106532

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l106_106532


namespace instantaneous_velocity_at_t_is_4_l106_106035

variable (t : ℝ)

def position (t : ℝ) : ℝ := t + (1 / 9) * t^3

theorem instantaneous_velocity_at_t_is_4 :
  deriv position 3 = 4 := by
sorry

end instantaneous_velocity_at_t_is_4_l106_106035


namespace nancy_soap_bars_l106_106077

def packs : ℕ := 6
def bars_per_pack : ℕ := 5

theorem nancy_soap_bars : packs * bars_per_pack = 30 := by
  sorry

end nancy_soap_bars_l106_106077


namespace inclination_angle_x_equals_3_is_90_l106_106096

-- Define the condition that line x = 3 is vertical
def is_vertical_line (x : ℝ) : Prop := x = 3

-- Define the inclination angle property for a vertical line
def inclination_angle_of_vertical_line_is_90 (x : ℝ) (h : is_vertical_line x) : ℝ :=
90   -- The angle is 90 degrees

-- Theorem statement to prove the inclination angle of the line x = 3 is 90 degrees
theorem inclination_angle_x_equals_3_is_90 :
  inclination_angle_of_vertical_line_is_90 3 (by simp [is_vertical_line]) = 90 :=
sorry  -- proof goes here


end inclination_angle_x_equals_3_is_90_l106_106096


namespace alicia_bought_more_markers_l106_106174

theorem alicia_bought_more_markers (price_per_marker : ℝ) (n_h : ℝ) (n_a : ℝ) (m : ℝ) 
    (h_hector : n_h * price_per_marker = 2.76) 
    (h_alicia : n_a * price_per_marker = 4.07)
    (h_diff : n_a - n_h = m) : 
  m = 13 :=
sorry

end alicia_bought_more_markers_l106_106174


namespace residue_of_neg_1001_mod_37_l106_106997

theorem residue_of_neg_1001_mod_37 : (-1001 : ℤ) % 37 = 35 :=
by
  sorry

end residue_of_neg_1001_mod_37_l106_106997


namespace nina_weekend_earnings_l106_106078

noncomputable def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℕ)
                                   (necklaces_sold bracelets_sold individual_earrings_sold ensembles_sold : ℕ) : ℕ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * (individual_earrings_sold / 2) +
  ensemble_price * ensembles_sold

theorem nina_weekend_earnings :
  total_money_made 25 15 10 45 5 10 20 2 = 465 :=
by
  sorry

end nina_weekend_earnings_l106_106078


namespace total_cost_after_discounts_and_cashback_l106_106702

def iPhone_original_price : ℝ := 800
def iWatch_original_price : ℝ := 300
def iPhone_discount_rate : ℝ := 0.15
def iWatch_discount_rate : ℝ := 0.10
def cashback_rate : ℝ := 0.02

theorem total_cost_after_discounts_and_cashback :
  (iPhone_original_price * (1 - iPhone_discount_rate) + iWatch_original_price * (1 - iWatch_discount_rate)) * (1 - cashback_rate) = 931 :=
by sorry

end total_cost_after_discounts_and_cashback_l106_106702


namespace boris_can_achieve_7_60_cents_l106_106285

/-- Define the conditions as constants -/
def penny_value : ℕ := 1
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def quarter_value : ℕ := 25

def penny_to_dimes : ℕ := 69
def dime_to_pennies : ℕ := 5
def nickel_to_quarters : ℕ := 120

/-- Function to determine if a value can be produced by a sequence of machine operations -/
def achievable_value (start: ℕ) (target: ℕ) : Prop :=
  ∃ k : ℕ, target = start + k * penny_to_dimes

theorem boris_can_achieve_7_60_cents : achievable_value penny_value 760 :=
  sorry

end boris_can_achieve_7_60_cents_l106_106285


namespace num_counting_numbers_dividing_52_leaving_remainder_7_l106_106330

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem num_counting_numbers_dividing_52_leaving_remainder_7 (n : ℕ) :
  (∃ n : ℕ, 59 ≡ 7 [MOD n]) → (n > 7 ∧ divides n 52) → n = 3 := 
sorry

end num_counting_numbers_dividing_52_leaving_remainder_7_l106_106330


namespace smallest_positive_n_l106_106555

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

theorem smallest_positive_n (n : ℕ) :
  (n > 0) ∧ (rotation_matrix ^ n = 1) ↔ n = 3 := sorry

end smallest_positive_n_l106_106555


namespace roots_of_quadratic_l106_106319

theorem roots_of_quadratic :
  ∃ m n : ℝ, (∀ x : ℝ, x^2 - 4 * x - 1 = 0 → (x = m ∨ x = n)) ∧
            (m + n = 4) ∧
            (m * n = -1) ∧
            (m + n - m * n = 5) :=
by
  sorry

end roots_of_quadratic_l106_106319


namespace max_paths_from_A_to_F_l106_106354

-- Define the points and line segments.
inductive Point
| A | B | C | D | E | F

-- Define the edges of the graph as pairs of points.
def edges : List (Point × Point) :=
  [(Point.A, Point.B), (Point.A, Point.E), (Point.A, Point.D),
   (Point.B, Point.C), (Point.B, Point.E),
   (Point.C, Point.F),
   (Point.D, Point.E), (Point.D, Point.F),
   (Point.E, Point.F)]

-- A path is valid if it passes through each point and line segment only once.
def valid_path (path : List (Point × Point)) : Bool :=
  -- Check that each edge in the path is unique and forms a sequence from A to F.
  sorry

-- Calculate the maximum number of different valid paths from point A to point F.
def max_paths : Nat :=
  List.length (List.filter valid_path (List.permutations edges))

theorem max_paths_from_A_to_F : max_paths = 9 :=
by sorry

end max_paths_from_A_to_F_l106_106354


namespace smallest_k_digit_number_l106_106211

theorem smallest_k_digit_number (a n : ℕ) (h1: a > 0) (h2 : (nat.log 10 (a ^ n) + 1) = 2014) : 2014 = Inf {k : ℕ | ¬(10^(k-1) ≤ a ∧ a < 10^k)} :=
by sorry

end smallest_k_digit_number_l106_106211


namespace diversity_values_l106_106516

theorem diversity_values (k : ℕ) (h : 1 ≤ k ∧ k ≤ 4) :
  ∃ (D : ℕ), D = 1000 * (k - 1) := by
  sorry

end diversity_values_l106_106516


namespace range_of_f_l106_106440

noncomputable def f (x : ℕ) : ℤ := x^2 - 3 * x

def domain : Finset ℕ := {1, 2, 3}

def range : Finset ℤ := {-2, 0}

theorem range_of_f :
  Finset.image f domain = range :=
by
  sorry

end range_of_f_l106_106440


namespace quadratic_function_range_l106_106715

theorem quadratic_function_range (x : ℝ) (y : ℝ) (h1 : y = x^2 - 2*x - 3) (h2 : -2 ≤ x ∧ x ≤ 2) :
  -4 ≤ y ∧ y ≤ 5 :=
sorry

end quadratic_function_range_l106_106715


namespace probability_at_least_one_five_or_six_l106_106649

theorem probability_at_least_one_five_or_six
  (P_neither_five_nor_six: ℚ)
  (h: P_neither_five_nor_six = 4 / 9) :
  (1 - P_neither_five_nor_six) = 5 / 9 :=
by
  sorry

end probability_at_least_one_five_or_six_l106_106649


namespace lines_parallel_l106_106032

noncomputable def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a, 2, 6)
noncomputable def line2 (a : ℝ) : ℝ × ℝ × ℝ := (1, a-1, a^2-1)

def are_parallel (line1 line2 : ℝ × ℝ × ℝ) : Prop :=
  let ⟨a1, b1, _⟩ := line1
  let ⟨a2, b2, _⟩ := line2
  a1 * b2 = a2 * b1

theorem lines_parallel (a : ℝ) :
  are_parallel (line1 a) (line2 a) → a = -1 :=
sorry

end lines_parallel_l106_106032


namespace average_score_is_correct_l106_106920

-- Define the given conditions
def numbers_of_students : List ℕ := [12, 28, 40, 35, 20, 10, 5]
def scores : List ℕ := [95, 85, 75, 65, 55, 45, 35]

-- Function to calculate the total score
def total_score (scores numbers : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ a b => a * b) scores numbers)

-- Calculate the average percent score
def average_percent_score (total number_of_students : ℕ) : ℕ :=
  total / number_of_students

-- Prove that the average percentage score is 70
theorem average_score_is_correct :
  average_percent_score (total_score scores numbers_of_students) 150 = 70 :=
by
  sorry

end average_score_is_correct_l106_106920


namespace contradiction_proof_l106_106253

theorem contradiction_proof (a b : ℝ) (h : a ≥ b) (h_pos : b > 0) (h_contr : a^2 < b^2) : false :=
by {
  sorry
}

end contradiction_proof_l106_106253


namespace average_GPA_of_whole_class_l106_106239

variable (n : ℕ)

def GPA_first_group : ℕ := 54 * (n / 3)
def GPA_second_group : ℕ := 45 * (2 * n / 3)
def total_GPA : ℕ := GPA_first_group n + GPA_second_group n

theorem average_GPA_of_whole_class : total_GPA n / n = 48 := by
  sorry

end average_GPA_of_whole_class_l106_106239


namespace certain_amount_l106_106424

theorem certain_amount (n : ℤ) (x : ℤ) : n = 5 ∧ 7 * n - 15 = 2 * n + x → x = 10 :=
by
  sorry

end certain_amount_l106_106424


namespace solve_inequality_l106_106791

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l106_106791


namespace width_of_sheet_of_paper_l106_106268

theorem width_of_sheet_of_paper (W : ℝ) (h1 : ∀ (W : ℝ), W > 0) (length_paper : ℝ) (margin : ℝ)
  (width_picture_area : ∀ (W : ℝ), W - 2 * margin = (W - 3)) 
  (area_picture : ℝ) (length_picture_area : ℝ) :
  length_paper = 10 ∧ margin = 1.5 ∧ area_picture = 38.5 ∧ length_picture_area = 7 →
  W = 8.5 :=
by
  sorry

end width_of_sheet_of_paper_l106_106268


namespace product_of_binomials_l106_106680

theorem product_of_binomials :
  (2*x^2 + 3*x - 4) * (x + 6) = 2*x^3 + 15*x^2 + 14*x - 24 :=
by {
  sorry
}

end product_of_binomials_l106_106680


namespace minValue_at_least_9_minValue_is_9_l106_106474

noncomputable def minValue (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) : ℝ :=
  1 / a + 4 / b + 9 / c

theorem minValue_at_least_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) :
  minValue a b c h_pos h_sum ≥ 9 :=
by
  sorry

theorem minValue_is_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4)
  (h_abc : a = 2/3 ∧ b = 4/3 ∧ c = 2) : minValue a b c h_pos h_sum = 9 :=
by
  sorry

end minValue_at_least_9_minValue_is_9_l106_106474


namespace water_left_l106_106468

theorem water_left (initial_water: ℚ) (science_experiment_use: ℚ) (plant_watering_use: ℚ)
  (h1: initial_water = 3)
  (h2: science_experiment_use = 5 / 4)
  (h3: plant_watering_use = 1 / 2) :
  (initial_water - science_experiment_use - plant_watering_use = 5 / 4) :=
by
  rw [h1, h2, h3]
  norm_num

end water_left_l106_106468


namespace original_price_is_135_l106_106176

-- Problem Statement:
variable (P : ℝ)  -- Let P be the original price of the potion

-- Conditions
axiom potion_cost : (1 / 15) * P = 9

-- Proof Goal
theorem original_price_is_135 : P = 135 :=
by
  sorry

end original_price_is_135_l106_106176


namespace smallest_positive_perfect_cube_has_divisor_l106_106759

theorem smallest_positive_perfect_cube_has_divisor (p q r s : ℕ) (hp : Prime p) (hq : Prime q)
  (hr : Prime r) (hs : Prime s) (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  ∃ n : ℕ, n = (p * q * r * s^2)^3 ∧ ∀ m : ℕ, (m = p^2 * q^3 * r^4 * s^5 → m ∣ n) :=
by
  sorry

end smallest_positive_perfect_cube_has_divisor_l106_106759


namespace largest_even_number_with_given_conditions_l106_106647

open Finset
open BigOperators

-- Define the conditions formally
def digits_sum_to_seventeen (n : ℕ) : Prop :=
  (n.digits 10).sum = 17

def all_distinct_digits (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n.digits 10).nth i ≠ (n.digits 10).nth j

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def largest_even_number_with_distinct_digits (n : ℕ) : Prop :=
  ∀ m, digits_sum_to_seventeen m → all_distinct_digits m → is_even m → m ≤ n

-- The statement to be proven
theorem largest_even_number_with_given_conditions :
  largest_even_number_with_distinct_digits 62108 :=
sorry

end largest_even_number_with_given_conditions_l106_106647


namespace fernanda_total_time_eq_90_days_l106_106862

-- Define the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Define the total time calculation
def total_time_to_finish_audiobooks (a h r : ℕ) : ℕ :=
  (h / r) * a

-- The assertion we need to prove
theorem fernanda_total_time_eq_90_days :
  total_time_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 :=
by sorry

end fernanda_total_time_eq_90_days_l106_106862


namespace solve_inequality_l106_106779

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l106_106779


namespace volume_of_square_pyramid_l106_106566

theorem volume_of_square_pyramid (a r : ℝ) : 
  a > 0 → r > 0 → volume = (1 / 3) * a^2 * r :=
by 
    sorry

end volume_of_square_pyramid_l106_106566


namespace stanley_walk_distance_l106_106927

variable (run_distance walk_distance : ℝ)

theorem stanley_walk_distance : 
  run_distance = 0.4 ∧ run_distance = walk_distance + 0.2 → walk_distance = 0.2 :=
by
  sorry

end stanley_walk_distance_l106_106927


namespace alok_total_payment_l106_106672

def cost_of_chapatis : Nat := 16 * 6
def cost_of_rice_plates : Nat := 5 * 45
def cost_of_mixed_vegetable_plates : Nat := 7 * 70
def total_cost : Nat := cost_of_chapatis + cost_of_rice_plates + cost_of_mixed_vegetable_plates

theorem alok_total_payment :
  total_cost = 811 := by
  unfold total_cost
  unfold cost_of_chapatis
  unfold cost_of_rice_plates
  unfold cost_of_mixed_vegetable_plates
  calc
    16 * 6 + 5 * 45 + 7 * 70 = 96 + 5 * 45 + 7 * 70 := by rfl
                      ... = 96 + 225 + 7 * 70 := by rfl
                      ... = 96 + 225 + 490 := by rfl
                      ... = 96 + (225 + 490) := by rw Nat.add_assoc
                      ... = (96 + 225) + 490 := by rw Nat.add_assoc
                      ... = 321 + 490 := by rfl
                      ... = 811 := by rfl

end alok_total_payment_l106_106672


namespace game_result_l106_106494

theorem game_result (a : ℤ) : ((2 * a + 6) / 2 - a = 3) :=
by
  sorry

end game_result_l106_106494


namespace eval_expression_l106_106541

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l106_106541


namespace boy_lap_time_l106_106177

noncomputable def total_time_needed
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) : ℝ :=
(side_lengths.zip running_speeds).foldl (λ (acc : ℝ) ⟨len, speed⟩ => acc + (len / (speed / 60))) 0
+ obstacle_time

theorem boy_lap_time
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) :
  side_lengths = [80, 120, 140, 100, 60] →
  running_speeds = [250, 200, 300, 166.67, 266.67] →
  obstacle_time = 5 →
  total_time_needed side_lengths running_speeds obstacle_time = 7.212 := by
  intros h_lengths h_speeds h_obstacle_time
  rw [h_lengths, h_speeds, h_obstacle_time]
  sorry

end boy_lap_time_l106_106177


namespace determine_exponent_l106_106442

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a

theorem determine_exponent (a : ℝ) (hf : power_function a 4 = 8) : power_function (3/2) = power_function a := by
  sorry

end determine_exponent_l106_106442


namespace henry_change_l106_106175

theorem henry_change (n : ℕ) (p m : ℝ) (h_n : n = 4) (h_p : p = 0.75) (h_m : m = 10) : 
  m - (n * p) = 7 := 
by 
  sorry

end henry_change_l106_106175


namespace div_36_of_n_ge_5_l106_106181

noncomputable def n := Nat

theorem div_36_of_n_ge_5 (n : ℕ) (hn : n ≥ 5) (h2 : ¬ (n % 2 = 0)) (h3 : ¬ (n % 3 = 0)) : 36 ∣ (n^2 - 1) :=
by
  sorry

end div_36_of_n_ge_5_l106_106181


namespace value_of_expression_l106_106310

theorem value_of_expression (x y : ℝ) (h₁ : x * y = -3) (h₂ : x + y = -4) :
  x^2 + 3 * x * y + y^2 = 13 :=
by
  sorry

end value_of_expression_l106_106310


namespace solve_abs_eq_l106_106865

theorem solve_abs_eq (x : ℝ) : (|x + 2| = 3*x - 6) → x = 4 :=
by
  intro h
  sorry

end solve_abs_eq_l106_106865


namespace jack_sugar_remaining_l106_106357

-- Define the initial amount of sugar and all daily transactions
def jack_initial_sugar : ℝ := 65
def jack_use_day1 : ℝ := 18.5
def alex_borrow_day1 : ℝ := 5.3
def jack_buy_day2 : ℝ := 30.2
def jack_use_day2 : ℝ := 12.7
def emma_give_day2 : ℝ := 4.75
def jack_buy_day3 : ℝ := 20.5
def jack_use_day3 : ℝ := 8.25
def alex_return_day3 : ℝ := 2.8
def alex_borrow_day3 : ℝ := 1.2
def jack_use_day4 : ℝ := 9.5
def olivia_give_day4 : ℝ := 6.35
def jack_use_day5 : ℝ := 10.75
def emma_borrow_day5 : ℝ := 3.1
def alex_return_day5 : ℝ := 3

-- Calculate the remaining sugar each day
def jack_sugar_day1 : ℝ := jack_initial_sugar - jack_use_day1 - alex_borrow_day1
def jack_sugar_day2 : ℝ := jack_sugar_day1 + jack_buy_day2 - jack_use_day2 + emma_give_day2
def jack_sugar_day3 : ℝ := jack_sugar_day2 + jack_buy_day3 - jack_use_day3 + alex_return_day3 - alex_borrow_day3
def jack_sugar_day4 : ℝ := jack_sugar_day3 - jack_use_day4 + olivia_give_day4
def jack_sugar_day5 : ℝ := jack_sugar_day4 - jack_use_day5 - emma_borrow_day5 + alex_return_day5

-- Final proof statement: Jack ends up with 63.3 pounds of sugar
theorem jack_sugar_remaining : jack_sugar_day5 = 63.3 := 
by sorry

end jack_sugar_remaining_l106_106357


namespace BoatsRUs_total_canoes_l106_106145

def totalCanoesBuiltByJuly (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem BoatsRUs_total_canoes :
  totalCanoesBuiltByJuly 5 3 7 = 5465 :=
by
  sorry

end BoatsRUs_total_canoes_l106_106145


namespace roots_sum_and_product_l106_106315

theorem roots_sum_and_product (m n : ℝ) (h : (x^2 - 4*x - 1 = 0).roots = [m, n]) : m + n - m*n = 5 :=
sorry

end roots_sum_and_product_l106_106315


namespace square_area_with_circles_l106_106665

theorem square_area_with_circles 
  (r : ℝ)
  (nrows : ℕ)
  (ncols : ℕ)
  (circle_radius : r = 3)
  (rows : nrows = 2)
  (columns : ncols = 3)
  (num_circles : nrows * ncols = 6)
  : ∃ (side_length area : ℝ), side_length = ncols * 2 * r ∧ area = side_length ^ 2 ∧ area = 324 := 
by sorry

end square_area_with_circles_l106_106665


namespace parabola_min_value_incorrect_statement_l106_106166

theorem parabola_min_value_incorrect_statement
  (m : ℝ)
  (A B : ℝ × ℝ)
  (P Q : ℝ × ℝ)
  (parabola : ℝ → ℝ)
  (on_parabola : ∀ (x : ℝ), parabola x = x^2 - 2*m*x + m^2 - 9)
  (A_intersects_x_axis : A.2 = 0)
  (B_intersects_x_axis : B.2 = 0)
  (A_on_parabola : parabola A.1 = A.2)
  (B_on_parabola : parabola B.1 = B.2)
  (P_on_parabola : parabola P.1 = P.2)
  (Q_on_parabola : parabola Q.1 = Q.2)
  (P_coordinates : P = (m + 1, parabola (m + 1)))
  (Q_coordinates : Q = (m - 3, parabola (m - 3))) :
  ∃ (min_y : ℝ), min_y = -9 ∧ min_y ≠ m^2 - 9 := 
sorry

end parabola_min_value_incorrect_statement_l106_106166


namespace smallest_square_side_length_l106_106527

theorem smallest_square_side_length (s : ℕ) :
  (∃ s, s > 3 ∧ s ≤ 4 ∧ (s - 1) * (s - 1) = 5) ↔ s = 4 := by
  sorry

end smallest_square_side_length_l106_106527


namespace train_speed_correct_l106_106410

theorem train_speed_correct :
  ∀ (L : ℝ) (V_man : ℝ) (T : ℝ) (V_train : ℝ),
    L = 220 ∧ V_man = 6 * (1000 / 3600) ∧ T = 11.999040076793857 ∧ 
    L / T - V_man = V_train ↔ V_train * 3.6 = 60 :=
by
  intros L V_man T V_train
  sorry

end train_speed_correct_l106_106410


namespace find_a_from_inclination_l106_106267

open Real

theorem find_a_from_inclination (a : ℝ) :
  (∃ (k : ℝ), k = (2 - (-3)) / (1 - a) ∧ k = tan (135 * pi / 180)) → a = 6 :=
by
  sorry

end find_a_from_inclination_l106_106267


namespace cliff_total_rocks_l106_106051

theorem cliff_total_rocks (I S : ℕ) (h1 : S = 2 * I) (h2 : I / 3 = 30) :
  I + S = 270 :=
sorry

end cliff_total_rocks_l106_106051


namespace benny_initial_comics_l106_106677

variable (x : ℕ)

def initial_comics (x : ℕ) : ℕ := x

def comics_after_selling (x : ℕ) : ℕ := (2 * x) / 5

def comics_after_buying (x : ℕ) : ℕ := (comics_after_selling x) + 12

def traded_comics (x : ℕ) : ℕ := (comics_after_buying x) / 4

def comics_after_trading (x : ℕ) : ℕ := (3 * (comics_after_buying x)) / 4 + 18

theorem benny_initial_comics : comics_after_trading x = 72 → x = 150 := by
  intro h
  sorry

end benny_initial_comics_l106_106677


namespace perpendicular_tangent_l106_106101

noncomputable def f (x a : ℝ) := (x + a) * Real.exp x -- Defines the function

theorem perpendicular_tangent (a : ℝ) : 
  ∀ (tangent_slope perpendicular_slope : ℝ), 
  (tangent_slope = 1) → 
  (perpendicular_slope = -1) →
  tangent_slope = Real.exp 0 * (a + 1) →
  tangent_slope + perpendicular_slope = 0 → 
  a = 0 := by 
  intros tangent_slope perpendicular_slope htangent hperpendicular hderiv hperpendicular_slope
  sorry

end perpendicular_tangent_l106_106101


namespace yangmei_1_yangmei_2i_yangmei_2ii_l106_106289

-- Problem 1: Prove that a = 20
theorem yangmei_1 (a : ℕ) (h : 160 * a + 270 * a = 8600) : a = 20 := by
  sorry

-- Problem 2 (i): Prove x = 44 and y = 36
theorem yangmei_2i (x y : ℕ) (h1 : 160 * x + 270 * y = 16760) (h2 : 8 * x + 18 * y = 1000) : x = 44 ∧ y = 36 := by
  sorry

-- Problem 2 (ii): Prove b = 9 or 18
theorem yangmei_2ii (m n b : ℕ) (h1 : 8 * (m + b) + 18 * n = 1000) (h2 : 160 * m + 270 * n = 16760) (h3 : 0 < b)
: b = 9 ∨ b = 18 := by
  sorry

end yangmei_1_yangmei_2i_yangmei_2ii_l106_106289


namespace roads_with_five_possible_roads_with_four_not_possible_l106_106896

-- Problem (a)
theorem roads_with_five_possible :
  ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 5) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

-- Problem (b)
theorem roads_with_four_not_possible :
  ¬ ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 4) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

end roads_with_five_possible_roads_with_four_not_possible_l106_106896


namespace g_50_zero_l106_106611

noncomputable def g : ℕ → ℝ → ℝ
| 0, x     => x + |x - 50| - |x + 50|
| (n+1), x => |g n x| - 2

theorem g_50_zero :
  ∃! x : ℝ, g 50 x = 0 :=
sorry

end g_50_zero_l106_106611


namespace proof_problem_l106_106027

noncomputable def p : Prop := ∃ (α : ℝ), Real.cos (Real.pi - α) = Real.cos α
def q : Prop := ∀ (x : ℝ), x ^ 2 + 1 > 0

theorem proof_problem : p ∨ q := 
by
  sorry

end proof_problem_l106_106027


namespace sad_girls_count_l106_106364

-- Statement of the problem in Lean 4
theorem sad_girls_count :
  ∀ (total_children happy_children sad_children neither_happy_nor_sad children boys girls happy_boys boys_neither_happy_nor_sad : ℕ),
    total_children = 60 →
    happy_children = 30 →
    sad_children = 10 →
    neither_happy_nor_sad = 20 →
    children = total_children →
    boys = 19 →
    girls = total_children - boys →
    happy_boys = 6 →
    boys_neither_happy_nor_sad = 7 →
    girls = 41 →
    sad_children = 10 →
    (sad_children = 6 + (total_children - boys - girls - neither_happy_nor_sad - happy_children)) → 
    ∃ sad_girls, sad_girls = 4 := by
  sorry

end sad_girls_count_l106_106364


namespace coefficient_x3y_in_expansion_of_2x_minus_y_power_4_l106_106460

theorem coefficient_x3y_in_expansion_of_2x_minus_y_power_4 :
  ∃ (r : ℕ), r = 1 ∧ 
    ∃ (C : ℤ), 
    (4.choose r) * (2 ^ (4 - r)) * (-1)^r = C ∧ 
    C = -32 :=
by
  sorry

end coefficient_x3y_in_expansion_of_2x_minus_y_power_4_l106_106460


namespace jimmy_needs_4_packs_of_bread_l106_106467

theorem jimmy_needs_4_packs_of_bread
  (num_sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (slices_per_pack : ℕ)
  (initial_slices : ℕ)
  (h1 : num_sandwiches = 8)
  (h2 : slices_per_sandwich = 2)
  (h3 : slices_per_pack = 4)
  (h4 : initial_slices = 0) :
  (num_sandwiches * slices_per_sandwich) / slices_per_pack = 4 := by
  sorry

end jimmy_needs_4_packs_of_bread_l106_106467


namespace find_ab_l106_106455

theorem find_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 31) : a * b = 3 := by
  sorry

end find_ab_l106_106455


namespace Y_3_2_eq_1_l106_106043

def Y (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem Y_3_2_eq_1 : Y 3 2 = 1 := by
  sorry

end Y_3_2_eq_1_l106_106043


namespace solve_inequality_l106_106777

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l106_106777


namespace lice_checks_time_in_hours_l106_106498

-- Define the number of students in each grade
def kindergarteners : ℕ := 26
def first_graders : ℕ := 19
def second_graders : ℕ := 20
def third_graders : ℕ := 25

-- Define the time each check takes (in minutes)
def time_per_check : ℕ := 2

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- The theorem states that the total time in hours is 3
theorem lice_checks_time_in_hours : 
  ((kindergarteners + first_graders + second_graders + third_graders) * time_per_check) / minutes_per_hour = 3 := 
by
  sorry

end lice_checks_time_in_hours_l106_106498


namespace g_of_f_of_3_is_1852_l106_106758

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 - x + 2

theorem g_of_f_of_3_is_1852 : g (f 3) = 1852 := by
  sorry

end g_of_f_of_3_is_1852_l106_106758


namespace beau_age_today_l106_106848

theorem beau_age_today (sons_age : ℕ) (triplets : ∀ i j : ℕ, i ≠ j → sons_age = 16) 
                       (beau_age_three_years_ago : ℕ) 
                       (H : 3 * (sons_age - 3) = beau_age_three_years_ago) :
  beau_age_three_years_ago + 3 = 42 :=
by
  -- Normally this is the place to write the proof,
  -- but it's enough to outline the theorem statement as per the instructions.
  sorry

end beau_age_today_l106_106848


namespace student_correct_answers_l106_106746

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 140) : C = 40 :=
by
  sorry

end student_correct_answers_l106_106746


namespace value_of_x_m_minus_n_l106_106300

variables {x : ℝ} {m n : ℝ}

theorem value_of_x_m_minus_n (hx_m : x^m = 6) (hx_n : x^n = 3) : x^(m - n) = 2 := 
by 
  sorry

end value_of_x_m_minus_n_l106_106300


namespace carvings_per_shelf_l106_106089

def total_wood_carvings := 56
def num_shelves := 7

theorem carvings_per_shelf : total_wood_carvings / num_shelves = 8 := by
  sorry

end carvings_per_shelf_l106_106089


namespace fernanda_total_time_to_finish_l106_106859

noncomputable def fernanda_days_to_finish_audiobooks
  (num_audiobooks : ℕ) (hours_per_audiobook : ℕ) (hours_listened_per_day : ℕ) : ℕ :=
num_audiobooks * hours_per_audiobook / hours_listened_per_day

-- Definitions based on the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Statement to prove
theorem fernanda_total_time_to_finish :
  fernanda_days_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 := 
sorry

end fernanda_total_time_to_finish_l106_106859


namespace beau_age_today_l106_106847

theorem beau_age_today (sons_age : ℕ) (triplets : ∀ i j : ℕ, i ≠ j → sons_age = 16) 
                       (beau_age_three_years_ago : ℕ) 
                       (H : 3 * (sons_age - 3) = beau_age_three_years_ago) :
  beau_age_three_years_ago + 3 = 42 :=
by
  -- Normally this is the place to write the proof,
  -- but it's enough to outline the theorem statement as per the instructions.
  sorry

end beau_age_today_l106_106847


namespace find_number_l106_106511

theorem find_number (n : ℤ) (h : 7 * n - 15 = 2 * n + 10) : n = 5 :=
sorry

end find_number_l106_106511


namespace checkerboard_inequivalent_color_schemes_l106_106639

/-- 
  We consider a 7x7 checkerboard where two squares are painted yellow, and the remaining 
  are painted green. Two color schemes are equivalent if one can be obtained from 
  the other by rotations of 0°, 90°, 180°, or 270°. We aim to prove that the 
  number of inequivalent color schemes is 312. 
-/
theorem checkerboard_inequivalent_color_schemes : 
  let n := 7
  let total_squares := n * n
  let total_pairs := total_squares.choose 2
  let symmetric_pairs := 24
  let nonsymmetric_pairs := total_pairs - symmetric_pairs
  let unique_symmetric_pairs := symmetric_pairs 
  let unique_nonsymmetric_pairs := nonsymmetric_pairs / 4
  unique_symmetric_pairs + unique_nonsymmetric_pairs = 312 :=
by sorry

end checkerboard_inequivalent_color_schemes_l106_106639


namespace triangle_area_via_line_eq_l106_106092

theorem triangle_area_via_line_eq (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  area = 1 / (2 * |a * b|) :=
by
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  sorry

end triangle_area_via_line_eq_l106_106092


namespace sum_first_75_terms_arith_seq_l106_106255

theorem sum_first_75_terms_arith_seq (a_1 d : ℕ) (n : ℕ) (h_a1 : a_1 = 3) (h_d : d = 4) (h_n : n = 75) : 
  (n * (2 * a_1 + (n - 1) * d)) / 2 = 11325 := 
by
  subst h_a1
  subst h_d
  subst h_n
  sorry

end sum_first_75_terms_arith_seq_l106_106255


namespace students_only_one_activity_l106_106807

theorem students_only_one_activity 
  (total : ℕ) (both : ℕ) (neither : ℕ)
  (h_total : total = 317) 
  (h_both : both = 30) 
  (h_neither : neither = 20) : 
  (total - both - neither) = 267 :=
by 
  sorry

end students_only_one_activity_l106_106807


namespace diff_squares_example_l106_106952

theorem diff_squares_example :
  (311^2 - 297^2) / 14 = 608 :=
by
  -- The theorem statement directly follows from the conditions and question.
  sorry

end diff_squares_example_l106_106952


namespace problem_statement_l106_106073

noncomputable def f (x : ℝ) := 2 * cos x * (cos x + sqrt 3 * sin x)

theorem problem_statement :
  (∃ T : ℝ, (T = π) ∧
    (∀ k : ℤ, (∀ x : ℝ, (k * π - π / 3 < x ∧ x < k * π + π / 6) → (f x = 2 * sin (2 * x + π / 6) + 1)))) ∧
  (x ∈ (set.Icc 0 (π / 2)) → (∃ m : ℝ, (m = 3) ∧ (∀ x : ℝ, f x ≤ m))) :=
begin
  sorry
end

end problem_statement_l106_106073


namespace solve_inequality_l106_106788

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l106_106788


namespace triangle_inequality_l106_106023

variables {a b c h : ℝ}
variable {n : ℕ}

theorem triangle_inequality
  (h_triangle : a^2 + b^2 = c^2)
  (h_height : a * b = c * h)
  (h_cond : a + b < c + h)
  (h_pos_n : n > 0) :
  a^n + b^n < c^n + h^n :=
sorry

end triangle_inequality_l106_106023


namespace sequence_general_term_l106_106737

/-- Given the sequence {a_n} defined by a_n = 2^n * a_{n-1} for n > 1 and a_1 = 1,
    prove that the general term a_n = 2^((n^2 + n - 2) / 2) -/
theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n > 1, a n = 2^n * a (n-1)) :
  ∀ n, a n = 2^((n^2 + n - 2) / 2) :=
sorry

end sequence_general_term_l106_106737


namespace same_color_pair_exists_l106_106137

-- Define the coloring of a point on a plane
def is_colored (x y : ℝ) : Type := ℕ  -- Assume ℕ represents two colors 0 and 1

-- Prove there exists two points of the same color such that the distance between them is 2006 meters
theorem same_color_pair_exists (colored : ℝ → ℝ → ℕ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ colored x1 y1 = colored x2 y2 ∧ (x2 - x1)^2 + (y2 - y1)^2 = 2006^2) :=
sorry

end same_color_pair_exists_l106_106137


namespace find_angle_A_l106_106050

noncomputable def angle_A (a b c S : ℝ) := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))

theorem find_angle_A (a b c S : ℝ) (hb : 0 < b) (hc : 0 < c) (hS : S = (1/2) * b * c * Real.sin (angle_A a b c S)) 
    (h_eq : b^2 + c^2 = (1/3) * a^2 + (4 * Real.sqrt 3 / 3) * S) : 
    angle_A a b c S = π / 6 := by 
  sorry

end find_angle_A_l106_106050


namespace ferry_P_travel_time_l106_106428

-- Definitions of conditions
def speed_P : ℝ := 6 -- speed of ferry P in km/h
def speed_diff_PQ : ℝ := 3 -- speed difference between ferry Q and ferry P in km/h
def travel_longer_Q : ℝ := 2 -- ferry Q travels a route twice as long as ferry P
def time_diff_PQ : ℝ := 1 -- time difference between ferry Q and ferry P in hours

-- Distance traveled by ferry P
def distance_P (t_P : ℝ) : ℝ := speed_P * t_P

-- Distance traveled by ferry Q
def distance_Q (t_P : ℝ) : ℝ := travel_longer_Q * (speed_P * t_P)

-- Speed of ferry Q
def speed_Q : ℝ := speed_P + speed_diff_PQ

-- Time taken by ferry Q
def time_Q (t_P : ℝ) : ℝ := t_P + time_diff_PQ

-- Main theorem statement
theorem ferry_P_travel_time (t_P : ℝ) : t_P = 3 :=
by
  have eq_Q : speed_Q * (time_Q t_P) = distance_Q t_P := sorry
  have eq_P : speed_P * t_P = distance_P t_P := sorry
  sorry

end ferry_P_travel_time_l106_106428


namespace tan_2x_value_l106_106321

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) := deriv f x

theorem tan_2x_value (x : ℝ) (h : f' x = 3 * f x) : Real.tan (2 * x) = (4/3) := by
  sorry

end tan_2x_value_l106_106321


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l106_106694

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l106_106694


namespace range_of_a_l106_106631

theorem range_of_a (f : ℝ → ℝ) (h_mono_dec : ∀ x1 x2, -2 ≤ x1 ∧ x1 ≤ 2 ∧ -2 ≤ x2 ∧ x2 ≤ 2 → x1 < x2 → f x1 > f x2) 
  (h_cond : ∀ a, -2 ≤ a + 1 ∧ a + 1 ≤ 2 ∧ -2 ≤ 2 * a ∧ 2 * a ≤ 2 → f (a + 1) < f (2 * a)) :
  { a : ℝ | -1 ≤ a ∧ a < 1 } :=
sorry

end range_of_a_l106_106631


namespace find_four_digit_numbers_l106_106548

noncomputable def four_digit_number_permutations_sum (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) : Prop :=
  6 * (x + y + z + t) * (1000 + 100 + 10 + 1) = 10 * (1111 * x)

theorem find_four_digit_numbers (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) :
  four_digit_number_permutations_sum x y z t distinct nonzero :=
  sorry

end find_four_digit_numbers_l106_106548


namespace negation_of_forall_prop_l106_106242

theorem negation_of_forall_prop :
  ¬ (∀ x : ℝ, x^2 + x > 0) ↔ ∃ x : ℝ, x^2 + x ≤ 0 :=
by
  sorry

end negation_of_forall_prop_l106_106242


namespace smallest_n_l106_106557

def matrix_rotation := 
  (matrix 2 2 ℝ)
    !![(1 / 2), (- (real.sqrt 3) / 2);
       (real.sqrt 3 / 2), (1 / 2)]

noncomputable def smallest_positive_integer (n : ℕ) : Prop :=
  matrix_rotation ^ n = 1

theorem smallest_n : smallest_positive_integer 3 :=
by
  sorry

end smallest_n_l106_106557


namespace total_tickets_l106_106369

theorem total_tickets (R K : ℕ) (hR : R = 12) (h_income : 2 * R + (9 / 2) * K = 60) : R + K = 20 :=
sorry

end total_tickets_l106_106369


namespace equation_solutions_equivalence_l106_106469

theorem equation_solutions_equivalence {n k : ℕ} (hn : 1 < n) (hk : 1 < k) (hnk : n > k) :
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^k) ↔
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^(n - k)) :=
by
  sorry

end equation_solutions_equivalence_l106_106469


namespace find_a_minus_b_l106_106356

theorem find_a_minus_b (a b : ℝ) :
  (∀ (x : ℝ), x^4 - 8 * x^3 + a * x^2 + b * x + 16 = 0 → x > 0) →
  a - b = 56 :=
by
  sorry

end find_a_minus_b_l106_106356


namespace danny_bottle_caps_after_collection_l106_106853

-- Definitions for the conditions
def initial_bottle_caps : ℕ := 69
def bottle_caps_thrown : ℕ := 60
def bottle_caps_found : ℕ := 58

-- Theorem stating the proof problem
theorem danny_bottle_caps_after_collection : 
  initial_bottle_caps - bottle_caps_thrown + bottle_caps_found = 67 :=
by {
  -- Placeholder for proof
  sorry
}

end danny_bottle_caps_after_collection_l106_106853


namespace three_numbers_sum_l106_106381

theorem three_numbers_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 10)
  (h4 : (a + b + c) / 3 = a + 8) (h5 : (a + b + c) / 3 = c - 20) : 
  a + b + c = 66 :=
sorry

end three_numbers_sum_l106_106381


namespace alex_silver_tokens_l106_106279

theorem alex_silver_tokens :
  let R : Int -> Int -> Int := fun x y => 100 - 3 * x + 2 * y
  let B : Int -> Int -> Int := fun x y => 50 + 2 * x - 4 * y
  let x := 61
  let y := 42
  100 - 3 * x + 2 * y < 3 → 50 + 2 * x - 4 * y < 4 → x + y = 103 :=
by
  intro hR hB
  sorry

end alex_silver_tokens_l106_106279


namespace sufficient_and_not_necessary_condition_l106_106569

theorem sufficient_and_not_necessary_condition (a b : ℝ) (hb: a < 0 ∧ b < 0) : a + b < 0 :=
by
  sorry

end sufficient_and_not_necessary_condition_l106_106569


namespace total_test_subjects_l106_106664

-- Defining the conditions as mathematical entities
def number_of_colors : ℕ := 5
def unique_two_color_codes : ℕ := number_of_colors * number_of_colors
def excess_subjects : ℕ := 6

-- Theorem stating the question and correct answer
theorem total_test_subjects :
  unique_two_color_codes + excess_subjects = 31 :=
by
  -- Leaving the proof as sorry, since the task only requires statement creation
  sorry

end total_test_subjects_l106_106664


namespace sum_a5_a6_a7_l106_106048

def S (n : ℕ) : ℕ :=
  n^2 + 2 * n + 5

theorem sum_a5_a6_a7 : S 7 - S 4 = 39 :=
  by sorry

end sum_a5_a6_a7_l106_106048


namespace total_population_l106_106006

-- Defining the populations of Springfield and the difference in population
def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

-- The definition of Greenville's population in terms of Springfield's population
def greenville_population : ℕ := springfield_population - population_difference

-- The statement that we want to prove: the total population of Springfield and Greenville
theorem total_population :
  springfield_population + greenville_population = 845640 := by
  sorry

end total_population_l106_106006


namespace bicycles_difference_on_october_1_l106_106622

def initial_inventory : Nat := 200
def february_decrease : Nat := 4
def march_decrease : Nat := 6
def april_decrease : Nat := 8
def may_decrease : Nat := 10
def june_decrease : Nat := 12
def july_decrease : Nat := 14
def august_decrease : Nat := 16 + 20
def september_decrease : Nat := 18
def shipment : Nat := 50

def total_decrease : Nat := february_decrease + march_decrease + april_decrease + may_decrease + june_decrease + july_decrease + august_decrease + september_decrease
def stock_increase : Nat := shipment
def net_decrease : Nat := total_decrease - stock_increase

theorem bicycles_difference_on_october_1 : initial_inventory - net_decrease = 58 := by
  sorry

end bicycles_difference_on_october_1_l106_106622


namespace sequence_count_l106_106148

theorem sequence_count :
  ∃ n : ℕ, 
  (∀ a : Fin 101 → ℤ, 
    a 1 = 0 ∧ 
    a 100 = 475 ∧ 
    (∀ k : ℕ, 1 ≤ k ∧ k < 100 → |a (k + 1) - a k| = 5) → 
    n = 4851) := 
sorry

end sequence_count_l106_106148


namespace exists_k_in_octahedron_l106_106195

theorem exists_k_in_octahedron
  (x0 y0 z0 : ℚ)
  (h : ∀ n : ℤ, x0 + y0 + z0 ≠ n ∧ 
                 x0 + y0 - z0 ≠ n ∧ 
                 x0 - y0 + z0 ≠ n ∧ 
                 x0 - y0 - z0 ≠ n) :
  ∃ k : ℕ, ∃ (xk yk zk : ℚ), 
    k ≠ 0 ∧ 
    xk = k * x0 ∧ 
    yk = k * y0 ∧ 
    zk = k * z0 ∧
    ∀ n : ℤ, 
      (xk + yk + zk < ↑n → xk + yk + zk > ↑(n - 1)) ∧ 
      (xk + yk - zk < ↑n → xk + yk - zk > ↑(n - 1)) ∧ 
      (xk - yk + zk < ↑n → xk - yk + zk > ↑(n - 1)) ∧ 
      (xk - yk - zk < ↑n → xk - yk - zk > ↑(n - 1)) :=
sorry

end exists_k_in_octahedron_l106_106195


namespace problem_statement_l106_106150

theorem problem_statement {n d : ℕ} (hn : 0 < n) (hd : 0 < d) (h1 : d ∣ n) (h2 : d^2 * n + 1 ∣ n^2 + d^2) :
  n = d^2 :=
sorry

end problem_statement_l106_106150


namespace range_of_a_l106_106576

noncomputable def f (a x : ℝ) : ℝ := (x^2 + 2 * a * x) * Real.log x - (1/2) * x^2 - 2 * a * x

theorem range_of_a (a : ℝ) : (∀ x > 0, 0 < (f a x)) ↔ -1 ≤ a :=
sorry

end range_of_a_l106_106576


namespace perfect_squares_between_50_and_1000_l106_106740

theorem perfect_squares_between_50_and_1000 :
  ∃ (count : ℕ), count = 24 ∧ ∀ (n : ℕ), 50 < n * n ∧ n * n < 1000 ↔ 8 ≤ n ∧ n ≤ 31 :=
by {
  -- proof goes here
  sorry
}

end perfect_squares_between_50_and_1000_l106_106740


namespace trigonometric_product_identity_l106_106415

theorem trigonometric_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) = 2.25 :=
by
  -- Let x = Real.pi / 12 and y = 5 * Real.pi / 12
  have h1 : Real.sin (11 * Real.pi / 12) = Real.sin (Real.pi - (Real.pi / 12)) := by sorry,
  have h2 : Real.sin (7 * Real.pi / 12) = Real.sin (Real.pi - (5 * Real.pi / 12)) := by sorry,
  have h3 : Real.sin (5 * Real.pi / 12) = Real.cos (Real.pi / 12) := by sorry,
  have h4 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry,
  have h5 : Real.sin (2 * Real.pi / 12) = 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) := by sorry,
  sorry

end trigonometric_product_identity_l106_106415


namespace sqrt_112_consecutive_integers_product_l106_106249

theorem sqrt_112_consecutive_integers_product : 
  (∃ (a b : ℕ), a * a < 112 ∧ 112 < b * b ∧ b = a + 1 ∧ a * b = 110) :=
by 
  use 10, 11
  repeat { sorry }

end sqrt_112_consecutive_integers_product_l106_106249


namespace bus_speed_l106_106936

noncomputable def radius : ℝ := 35 / 100  -- Radius in meters
noncomputable def rpm : ℝ := 500.4549590536851

noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def distance_in_one_minute : ℝ := circumference * rpm
noncomputable def distance_in_km_per_hour : ℝ := (distance_in_one_minute / 1000) * 60

theorem bus_speed :
  distance_in_km_per_hour = 66.037 :=
by
  -- The proof is skipped here as it is not required
  sorry

end bus_speed_l106_106936


namespace measure_4_minutes_with_hourglasses_l106_106657

/-- Prove that it is possible to measure exactly 4 minutes using hourglasses of 9 minutes and 7 minutes and the minimum total time required is 18 minutes -/
theorem measure_4_minutes_with_hourglasses : 
  ∃ (a b : ℕ), (9 * a - 7 * b = 4) ∧ (a + b) * 1 ≤ 2 ∧ (a * 9 ≤ 18 ∧ b * 7 <= 18) :=
by {
  sorry
}

end measure_4_minutes_with_hourglasses_l106_106657


namespace correct_age_equation_l106_106505

variable (x : ℕ)

def age_older_brother (x : ℕ) : ℕ := 2 * x
def age_younger_brother_six_years_ago (x : ℕ) : ℕ := x - 6
def age_older_brother_six_years_ago (x : ℕ) : ℕ := 2 * x - 6

theorem correct_age_equation (h1 : age_younger_brother_six_years_ago x + age_older_brother_six_years_ago x = 15) :
  (x - 6) + (2 * x - 6) = 15 :=
by
  sorry

end correct_age_equation_l106_106505


namespace eval_expression_l106_106422

theorem eval_expression :
  let x := 2
  let y := -3
  let z := 1
  x^2 + y^2 - z^2 + 2 * x * y + 3 * z = 0 := by
sorry

end eval_expression_l106_106422


namespace train_speed_l106_106409

theorem train_speed (train_length platform_length total_time : ℕ) 
  (h_train_length : train_length = 150) 
  (h_platform_length : platform_length = 250) 
  (h_total_time : total_time = 8) : 
  (train_length + platform_length) / total_time = 50 := 
by
  -- Proof goes here
  -- Given: train_length = 150, platform_length = 250, total_time = 8
  -- We need to prove: (train_length + platform_length) / total_time = 50
  -- So we calculate
  --  (150 + 250)/8 = 400/8 = 50
  sorry

end train_speed_l106_106409


namespace algebraic_expression_value_l106_106648

variable (a b : ℝ)
axiom h1 : a = 3
axiom h2 : a - b = 1

theorem algebraic_expression_value :
  a^2 - a * b = 3 :=
by
  sorry

end algebraic_expression_value_l106_106648


namespace cos_value_l106_106741

theorem cos_value (α : ℝ) (h : Real.sin (π/4 + α) = 1/3) :
  Real.cos (π/2 - 2*α) = -7/9 :=
sorry

end cos_value_l106_106741


namespace equilateral_triangle_area_increase_l106_106412

theorem equilateral_triangle_area_increase (A : ℝ) (k : ℝ) (s : ℝ) (s' : ℝ) (A' : ℝ) (ΔA : ℝ) :
  A = 36 * Real.sqrt 3 →
  A = (Real.sqrt 3 / 4) * s^2 →
  s' = s + 3 →
  A' = (Real.sqrt 3 / 4) * s'^2 →
  ΔA = A' - A →
  ΔA = 20.25 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_increase_l106_106412


namespace eval_expr_l106_106156

theorem eval_expr : 3 + 3 * (3 ^ (3 ^ 3)) - 3 ^ 3 = 22876792454937 := by
  sorry

end eval_expr_l106_106156


namespace find_d_l106_106162

noncomputable def median (x : ℕ) : ℕ := x + 4
noncomputable def mean (x d : ℕ) : ℕ := x + (13 + d) / 5

theorem find_d (x d : ℕ) (h : mean x d = median x + 5) : d = 32 := by
  sorry

end find_d_l106_106162


namespace probability_not_within_B_l106_106796

-- Definition representing the problem context
structure Squares where
  areaA : ℝ
  areaA_pos : areaA = 65
  perimeterB : ℝ
  perimeterB_pos : perimeterB = 16

-- The theorem to be proved
theorem probability_not_within_B (s : Squares) : 
  let sideA := Real.sqrt s.areaA
  let sideB := s.perimeterB / 4
  let areaB := sideB^2
  let area_not_covered := s.areaA - areaB
  let probability := area_not_covered / s.areaA
  probability = 49 / 65 := 
by
  sorry

end probability_not_within_B_l106_106796


namespace exist_positive_integers_summing_to_one_l106_106246

theorem exist_positive_integers_summing_to_one :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (1 / (x:ℚ) + 1 / (y:ℚ) + 1 / (z:ℚ) = 1)
    ∧ ((x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end exist_positive_integers_summing_to_one_l106_106246


namespace total_pieces_is_100_l106_106502

-- Definitions based on conditions
def total_pieces_of_bread (B : ℕ) : Prop :=
  let duck1 := B / 2 in      -- The first duck eats half of all the pieces of bread
  let duck2 := 13 in         -- The second duck eats 13 pieces
  let duck3 := 7 in          -- The third duck eats 7 pieces
  let left_in_water := 30 in -- There are 30 pieces left in the water
  duck1 + duck2 + duck3 + left_in_water = B

-- The statement to be proved
theorem total_pieces_is_100 (B : ℕ) : total_pieces_of_bread B → B = 100 :=
by
  -- Proof would be provided here
  sorry

end total_pieces_is_100_l106_106502


namespace min_value_g_l106_106615

noncomputable def f (x₁ x₂ x₃ : ℝ) : ℝ :=
  -2 * (x₁^3 + x₂^3 + x₃^3) + 3 * (x₁^2 * (x₂ + x₃) + x₂^2 * (x₁ + x₃) + x₃^2 * (x₁ + x₂)) - 12 * x₁ * x₂ * x₃

noncomputable def g (r s t : ℝ) : ℝ :=
  (λ x₃, |f r (r + 2) x₃ + s|) '' (Set.Icc t (t + 2)).toMax

theorem min_value_g : ∀ r s t : ℝ, ∃ a : ℝ, g r s t = 12 * Real.sqrt 3 := sorry

end min_value_g_l106_106615


namespace crabapple_recipients_sequences_l106_106764

-- Define the number of students in Mrs. Crabapple's class
def num_students : ℕ := 12

-- Define the number of class meetings per week
def num_meetings : ℕ := 5

-- Define the total number of different sequences
def total_sequences : ℕ := num_students ^ num_meetings

-- The target theorem to prove
theorem crabapple_recipients_sequences :
  total_sequences = 248832 := by
  sorry

end crabapple_recipients_sequences_l106_106764


namespace Shekar_marks_in_Science_l106_106483

theorem Shekar_marks_in_Science (S : ℕ) (h : (76 + S + 82 + 67 + 85) / 5 = 75) : S = 65 :=
sorry

end Shekar_marks_in_Science_l106_106483


namespace factorial_div_add_two_l106_106993

def factorial (n : ℕ) : ℕ :=
match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

theorem factorial_div_add_two :
  (factorial 50) / (factorial 48) + 2 = 2452 :=
by
  sorry

end factorial_div_add_two_l106_106993


namespace system_of_equations_solution_system_of_inequalities_no_solution_l106_106824

-- Problem 1: Solving system of linear equations
theorem system_of_equations_solution :
  ∃ x y : ℝ, x - 3*y = -5 ∧ 2*x + 2*y = 6 ∧ x = 1 ∧ y = 2 := by
  sorry

-- Problem 2: Solving the system of inequalities
theorem system_of_inequalities_no_solution :
  ¬ (∃ x : ℝ, 2*x < -4 ∧ (1/2)*x - 5 > 1 - (3/2)*x) := by
  sorry

end system_of_equations_solution_system_of_inequalities_no_solution_l106_106824


namespace math_problem_l106_106718

variable (a b c m : ℝ)

-- Quadratic equation: y = ax^2 + bx + c
def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Opens downward
axiom a_neg : a < 0
-- Passes through A(1, 0)
axiom passes_A : quadratic a b c 1 = 0
-- Passes through B(m, 0) with -2 < m < -1
axiom passes_B : quadratic a b c m = 0
axiom m_range : -2 < m ∧ m < -1

-- Prove the conclusions
theorem math_problem : b < 0 ∧ (a + b + c = 0) ∧ (a * (m+1) - b + c > 0) ∧ ¬(4 * a * c - b^2 > 4 * a) :=
by
  sorry

end math_problem_l106_106718


namespace number_of_ways_to_select_team_l106_106191

def calc_binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem number_of_ways_to_select_team : calc_binomial_coefficient 17 4 = 2380 := by
  sorry

end number_of_ways_to_select_team_l106_106191


namespace total_paths_A_to_D_l106_106870

-- Given conditions
def paths_from_A_to_B := 2
def paths_from_B_to_C := 2
def paths_from_C_to_D := 2
def direct_path_A_to_C := 1
def direct_path_B_to_D := 1

-- Proof statement
theorem total_paths_A_to_D : 
  paths_from_A_to_B * paths_from_B_to_C * paths_from_C_to_D + 
  direct_path_A_to_C * paths_from_C_to_D + 
  paths_from_A_to_B * direct_path_B_to_D = 12 := 
  by
    sorry

end total_paths_A_to_D_l106_106870


namespace f_xh_sub_f_x_l106_106606

def f (x : ℝ) (k : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + k * x - 4

theorem f_xh_sub_f_x (x h : ℝ) (k : ℝ := -5) : 
    f (x + h) k - f x k = h * (6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h - 5) := by
  sorry

end f_xh_sub_f_x_l106_106606


namespace inequality_proof_l106_106908

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a + 1 / b + 9 / c + 25 / d) ≥ (100 / (a + b + c + d)) :=
by
  sorry

end inequality_proof_l106_106908


namespace combined_distance_is_twelve_l106_106643

-- Definitions based on the conditions
def distance_second_lady : ℕ := 4
def distance_first_lady : ℕ := 2 * distance_second_lady
def total_distance : ℕ := distance_second_lady + distance_first_lady

-- Theorem statement
theorem combined_distance_is_twelve : total_distance = 12 := by
  sorry

end combined_distance_is_twelve_l106_106643


namespace solve_inequality_l106_106795

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l106_106795


namespace system_of_equations_l106_106444

theorem system_of_equations (x y k : ℝ) 
  (h1 : x + 2 * y = k + 2) 
  (h2 : 2 * x - 3 * y = 3 * k - 1) : 
  x + 9 * y = 7 :=
  sorry

end system_of_equations_l106_106444


namespace cos2theta_sin2theta_l106_106743

theorem cos2theta_sin2theta (θ : ℝ) (h : 2 * Real.cos θ + Real.sin θ = 0) :
  Real.cos (2 * θ) + (1 / 2) * Real.sin (2 * θ) = -1 :=
sorry

end cos2theta_sin2theta_l106_106743


namespace prove_proposition_l106_106574

-- Define the propositions p and q
def p : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def q : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Define the main theorem to prove
theorem prove_proposition : (¬ p) ∨ q :=
by { sorry }

end prove_proposition_l106_106574


namespace initial_men_in_camp_l106_106487

theorem initial_men_in_camp (M F : ℕ) 
  (h1 : F = M * 50)
  (h2 : F = (M + 10) * 25) : 
  M = 10 :=
by
  sorry

end initial_men_in_camp_l106_106487


namespace total_number_of_coins_l106_106752

variable (nickels dimes total_value : ℝ)
variable (total_nickels : ℕ)

def value_of_nickel : ℝ := 0.05
def value_of_dime : ℝ := 0.10

theorem total_number_of_coins :
  total_value = 3.50 → total_nickels = 30 → total_value = total_nickels * value_of_nickel + dimes * value_of_dime → 
  total_nickels + dimes = 50 :=
by
  intros h_total_value h_total_nickels h_value_equation
  sorry

end total_number_of_coins_l106_106752


namespace divisibility_by_n5_plus_1_l106_106909

theorem divisibility_by_n5_plus_1 (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  n^5 + 1 ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) :=
sorry

end divisibility_by_n5_plus_1_l106_106909


namespace zeros_of_shifted_function_l106_106577

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_shifted_function :
  {x : ℝ | f (x - 1) = 0} = {0, 2} :=
sorry

end zeros_of_shifted_function_l106_106577


namespace smallest_k_for_a_n_digital_l106_106212

theorem smallest_k_for_a_n_digital (a n : ℕ) (h : 10^2013 ≤ a^n ∧ a^n < 10^2014) : 
  ∀ k : ℕ, (∀ b : ℕ, 10^(k-1) ≤ b → b < 10^k → (¬(10^2013 ≤ b^n ∧ b^n < 10^2014))) ↔ k = 2014 :=
by 
  sorry

end smallest_k_for_a_n_digital_l106_106212


namespace arithmetic_sequence_term_number_l106_106749

theorem arithmetic_sequence_term_number
  (a : ℕ → ℤ)
  (ha1 : a 1 = 1)
  (ha2 : a 2 = 3)
  (n : ℕ)
  (hn : a n = 217) :
  n = 109 :=
sorry

end arithmetic_sequence_term_number_l106_106749


namespace smallest_k_exists_l106_106161

theorem smallest_k_exists : ∃ (k : ℕ), k > 0 ∧ (∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ k = 19^n - 5^m) ∧ k = 14 :=
by 
  sorry

end smallest_k_exists_l106_106161


namespace value_of_polynomial_l106_106722

theorem value_of_polynomial (x y : ℝ) (h : x - y = 5) : (x - y)^2 + 2 * (x - y) - 10 = 25 :=
by sorry

end value_of_polynomial_l106_106722


namespace calculate_expression_l106_106681

theorem calculate_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end calculate_expression_l106_106681


namespace cricket_avg_score_l106_106404

theorem cricket_avg_score
  (avg_first_two : ℕ)
  (num_first_two : ℕ)
  (avg_all_five : ℕ)
  (num_all_five : ℕ)
  (avg_first_two_eq : avg_first_two = 40)
  (num_first_two_eq : num_first_two = 2)
  (avg_all_five_eq : avg_all_five = 22)
  (num_all_five_eq : num_all_five = 5) :
  ((num_all_five * avg_all_five - num_first_two * avg_first_two) / (num_all_five - num_first_two) = 10) :=
by
  sorry

end cricket_avg_score_l106_106404


namespace radius_of_circle_l106_106978

theorem radius_of_circle
  (d PQ QR : ℝ) (h1 : d = 15) (h2 : PQ = 7) (h3 : QR = 8) :
  ∃ r : ℝ, r = 2 * Real.sqrt 30 ∧ (PQ * (PQ + QR) = (d - r) * (d + r)) :=
by
  -- All necessary non-proof related statements
  sorry

end radius_of_circle_l106_106978


namespace Tom_sold_games_for_240_l106_106944

-- Define the value of games and perform operations as per given conditions
def original_value : ℕ := 200
def tripled_value : ℕ := 3 * original_value
def sold_percentage : ℕ := 40
def sold_value : ℕ := (sold_percentage * tripled_value) / 100

-- Assert the proof problem
theorem Tom_sold_games_for_240 : sold_value = 240 := 
by
  sorry

end Tom_sold_games_for_240_l106_106944


namespace solve_for_a_l106_106298

variable (x y a : ℤ)
variable (hx : x = 1)
variable (hy : y = -3)
variable (eq : a * x - y = 1)
 
theorem solve_for_a : a = -2 := by
  -- Placeholder to satisfy the lean prover, no actual proof steps
  sorry

end solve_for_a_l106_106298


namespace jim_ran_16_miles_in_2_hours_l106_106065

-- Given conditions
variables (j f : ℝ) -- miles Jim ran in 2 hours, miles Frank ran in 2 hours
variables (h1 : f = 20) -- Frank ran 20 miles in 2 hours
variables (h2 : f / 2 = (j / 2) + 2) -- Frank ran 2 miles more than Jim in an hour

-- Statement to prove
theorem jim_ran_16_miles_in_2_hours (j f : ℝ) (h1 : f = 20) (h2 : f / 2 = (j / 2) + 2) : j = 16 :=
by
  sorry

end jim_ran_16_miles_in_2_hours_l106_106065


namespace trig_product_identity_l106_106414

theorem trig_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) =
  (1 + Real.sin (Real.pi / 12))^2 * (1 + Real.sin (5 * Real.pi / 12))^2 :=
by
  sorry

end trig_product_identity_l106_106414


namespace trig_expression_l106_106451

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := 
  sorry

end trig_expression_l106_106451


namespace ratio_of_a_to_b_l106_106384

variables (a b x m : ℝ)
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variables (h_x : x = a + 0.25 * a)
variables (h_m : m = b - 0.80 * b)
variables (h_ratio : m / x = 0.2)

theorem ratio_of_a_to_b (h_pos_a : 0 < a) (h_pos_b : 0 < b)
                        (h_x : x = a + 0.25 * a)
                        (h_m : m = b - 0.80 * b)
                        (h_ratio : m / x = 0.2) :
  a / b = 5 / 4 := by
  sorry

end ratio_of_a_to_b_l106_106384


namespace distance_between_truck_and_car_l106_106836

noncomputable def speed_truck : ℝ := 65
noncomputable def speed_car : ℝ := 85
noncomputable def time : ℝ := 3 / 60

theorem distance_between_truck_and_car : 
  let Distance_truck := speed_truck * time,
      Distance_car := speed_car * time in
  Distance_car - Distance_truck = 1 :=
by {
  sorry
}

end distance_between_truck_and_car_l106_106836


namespace simplify_expression_l106_106707

variable (y : ℝ)
variable (h : y ≠ 0)

theorem simplify_expression : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry

end simplify_expression_l106_106707


namespace quadratic_has_two_distinct_real_roots_l106_106294

theorem quadratic_has_two_distinct_real_roots (p : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - 3) * (x1 - 2) - p^2 = 0 ∧ (x2 - 3) * (x2 - 2) - p^2 = 0 :=
by
  -- This part will be replaced with the actual proof
  sorry

end quadratic_has_two_distinct_real_roots_l106_106294


namespace solve_q_l106_106770

theorem solve_q (n m q : ℤ) 
  (h₁ : 5/6 = n/72) 
  (h₂ : 5/6 = (m + n)/90) 
  (h₃ : 5/6 = (q - m)/150) : 
  q = 140 := by
  sorry

end solve_q_l106_106770


namespace watch_loss_percentage_l106_106140

noncomputable def initial_loss_percentage : ℝ :=
  let CP := 350
  let SP_new := 364
  let delta_SP := 140
  show ℝ from 
  sorry

theorem watch_loss_percentage (CP SP_new delta_SP : ℝ) (h₁ : CP = 350)
  (h₂ : SP_new = 364) (h₃ : delta_SP = 140) : 
  initial_loss_percentage = 36 :=
by
  -- Use the hypothesis and solve the corresponding problem
  sorry

end watch_loss_percentage_l106_106140


namespace average_visitors_per_day_l106_106652

theorem average_visitors_per_day (average_sunday : ℕ) (average_other : ℕ) (days_in_month : ℕ) (begins_with_sunday : Bool) :
  average_sunday = 600 → average_other = 240 → days_in_month = 30 → begins_with_sunday = true → (8640 / 30 = 288) :=
by
  intros h1 h2 h3 h4
  sorry

end average_visitors_per_day_l106_106652


namespace intersection_point_of_curve_and_line_l106_106935

theorem intersection_point_of_curve_and_line : 
  ∃ (e : ℝ), (0 < e) ∧ (e = Real.exp 1) ∧ ((e, e) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), x ^ y = y ^ x ∧ 0 ≤ x ∧ 0 ≤ y}) :=
by {
  sorry
}

end intersection_point_of_curve_and_line_l106_106935


namespace find_m_l106_106340

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^3 + 6*x^2 - m

theorem find_m (m : ℝ) (h : ∃ x : ℝ, f x m = 12) : m = 20 :=
by
  sorry

end find_m_l106_106340


namespace sale_in_fourth_month_l106_106133

variable (sale1 sale2 sale3 sale5 sale6 sale4 : ℕ)

def average_sale (total : ℕ) (months : ℕ) : ℕ := total / months

theorem sale_in_fourth_month
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7391)
  (avg : average_sale (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) 6 = 6900) :
  sale4 = 7230 := 
sorry

end sale_in_fourth_month_l106_106133


namespace polynomial_no_strictly_positive_roots_l106_106070

-- Define the necessary conditions and prove the main statement

variables (n : ℕ)
variables (a : Fin n → ℕ) (k : ℕ) (M : ℕ)

-- Axioms/Conditions
axiom pos_a (i : Fin n) : 0 < a i
axiom pos_k : 0 < k
axiom pos_M : 0 < M
axiom M_gt_1 : M > 1

axiom sum_reciprocals : (Finset.univ.sum (λ i => (1 : ℚ) / a i)) = k
axiom product_a : (Finset.univ.prod a) = M

noncomputable def polynomial_has_no_positive_roots : Prop :=
  ∀ x : ℝ, 0 < x →
    M * (1 + x)^k > (Finset.univ.prod (λ i => x + a i))

theorem polynomial_no_strictly_positive_roots (h : polynomial_has_no_positive_roots n a k M) : 
  ∀ x : ℝ, 0 < x → (M * (1 + x)^k - (Finset.univ.prod (λ i => x + a i)) ≠ 0) :=
by
  sorry

end polynomial_no_strictly_positive_roots_l106_106070


namespace smallest_b_not_divisible_by_5_l106_106425

theorem smallest_b_not_divisible_by_5 :
  ∃ b : ℕ, b > 2 ∧ ¬ (5 ∣ (2 * b^3 - 1)) ∧ ∀ b' > 2, ¬ (5 ∣ (2 * (b'^3) - 1)) → b = 6 :=
by
  sorry

end smallest_b_not_divisible_by_5_l106_106425


namespace find_f_log_l106_106727

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom f_def : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2

-- Theorem to be proved
theorem find_f_log : f (Real.log 6 / Real.log (1/2)) = 1 / 2 :=
by
  sorry

end find_f_log_l106_106727


namespace solve_inequality_l106_106785

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l106_106785


namespace range_independent_variable_l106_106461

theorem range_independent_variable (x : ℝ) (h : x + 1 > 0) : x > -1 :=
sorry

end range_independent_variable_l106_106461


namespace abs_inequality_l106_106780

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l106_106780


namespace variance_transformed_is_8_l106_106343

variables {n : ℕ} (x : Fin n → ℝ)

-- Given: the variance of x₁, x₂, ..., xₙ is 2.
def variance_x (x : Fin n → ℝ) : ℝ := sorry

axiom variance_x_is_2 : variance_x x = 2

-- Variance of 2 * x₁ + 3, 2 * x₂ + 3, ..., 2 * xₙ + 3
def variance_transformed (x : Fin n → ℝ) : ℝ :=
  variance_x (fun i => 2 * x i + 3)

-- Prove that the variance is 8.
theorem variance_transformed_is_8 : variance_transformed x = 8 :=
  sorry

end variance_transformed_is_8_l106_106343


namespace rectangles_single_row_7_rectangles_grid_7_4_l106_106825

def rectangles_in_single_row (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem rectangles_single_row_7 :
  rectangles_in_single_row 7 = 28 :=
by
  -- Add the proof here
  sorry

def rectangles_in_grid (rows cols : ℕ) : ℕ :=
  ((cols + 1) * cols / 2) * ((rows + 1) * rows / 2)

theorem rectangles_grid_7_4 :
  rectangles_in_grid 4 7 = 280 :=
by
  -- Add the proof here
  sorry

end rectangles_single_row_7_rectangles_grid_7_4_l106_106825


namespace total_number_of_birds_l106_106351

theorem total_number_of_birds (B C G S W : ℕ) (h1 : C = 2 * B) (h2 : G = 4 * B)
  (h3 : S = (C + G) / 2) (h4 : W = 8) (h5 : B = 2 * W) :
  C + G + S + W + B = 168 :=
  by
  sorry

end total_number_of_birds_l106_106351


namespace product_eq_one_l106_106578

noncomputable def f (x : ℝ) : ℝ := |Real.logb 3 x|

theorem product_eq_one (a b : ℝ) (h_diff : a ≠ b) (h_eq : f a = f b) : a * b = 1 := by
  sorry

end product_eq_one_l106_106578


namespace calculation_2015_l106_106948

theorem calculation_2015 :
  2015 ^ 2 - 2016 * 2014 = 1 :=
by
  sorry

end calculation_2015_l106_106948


namespace snowfall_difference_l106_106989

-- Defining all conditions given in the problem
def BaldMountain_snowfall_meters : ℝ := 1.5
def BillyMountain_snowfall_meters : ℝ := 3.5
def MountPilot_snowfall_centimeters : ℝ := 126
def RockstonePeak_snowfall_millimeters : ℝ := 5250
def SunsetRidge_snowfall_meters : ℝ := 2.25

-- Conversion constants
def meters_to_centimeters : ℝ := 100
def millimeters_to_centimeters : ℝ := 0.1

-- Converting snowfall amounts to centimeters
def BaldMountain_snowfall_centimeters : ℝ := BaldMountain_snowfall_meters * meters_to_centimeters
def BillyMountain_snowfall_centimeters : ℝ := BillyMountain_snowfall_meters * meters_to_centimeters
def RockstonePeak_snowfall_centimeters : ℝ := RockstonePeak_snowfall_millimeters * millimeters_to_centimeters
def SunsetRidge_snowfall_centimeters : ℝ := SunsetRidge_snowfall_meters * meters_to_centimeters

-- Defining total combined snowfall
def combined_snowfall_centimeters : ℝ :=
  BillyMountain_snowfall_centimeters + MountPilot_snowfall_centimeters + RockstonePeak_snowfall_centimeters + SunsetRidge_snowfall_centimeters

-- Stating the proof statement
theorem snowfall_difference :
  combined_snowfall_centimeters - BaldMountain_snowfall_centimeters = 1076 := 
  by
    sorry

end snowfall_difference_l106_106989


namespace largest_common_term_in_range_l106_106840

def seq1 (n : ℕ) : ℕ := 5 + 9 * n
def seq2 (m : ℕ) : ℕ := 3 + 8 * m

theorem largest_common_term_in_range :
  ∃ (a : ℕ) (n m : ℕ), seq1 n = a ∧ seq2 m = a ∧ 1 ≤ a ∧ a ≤ 200 ∧ (∀ b, (∃ nf mf, seq1 nf = b ∧ seq2 mf = b ∧ 1 ≤ b ∧ b ≤ 200) → b ≤ a) :=
sorry

end largest_common_term_in_range_l106_106840


namespace romanov_family_savings_l106_106514

theorem romanov_family_savings :
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption := monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  savings = 3824 :=
by
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption :=monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2 
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  show savings = 3824 
  sorry

end romanov_family_savings_l106_106514


namespace denise_removed_bananas_l106_106995

theorem denise_removed_bananas (initial_bananas remaining_bananas : ℕ) 
  (h_initial : initial_bananas = 46) (h_remaining : remaining_bananas = 41) : 
  initial_bananas - remaining_bananas = 5 :=
by
  sorry

end denise_removed_bananas_l106_106995


namespace find_sin_beta_l106_106453

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π/2) -- α is acute
variable (hβ : 0 < β ∧ β < π/2) -- β is acute

variable (hcosα : Real.cos α = 4/5)
variable (hcosαβ : Real.cos (α + β) = 5/13)

theorem find_sin_beta (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
    (hcosα : Real.cos α = 4/5) (hcosαβ : Real.cos (α + β) = 5/13) : 
    Real.sin β = 33/65 := 
sorry

end find_sin_beta_l106_106453


namespace linear_eq_zero_l106_106969

variables {a b c d x y : ℝ}

theorem linear_eq_zero (h1 : a * x + b * y = 0) (h2 : c * x + d * y = 0) (h3 : a * d - c * b ≠ 0) :
  x = 0 ∧ y = 0 :=
by
  sorry

end linear_eq_zero_l106_106969


namespace equilateral_triangle_of_altitude_sum_l106_106062

def triangle (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def altitude (a b c : ℝ) (S : ℝ) : ℝ := 
  2 * S / a

noncomputable def inradius (S : ℝ) (s : ℝ) : ℝ := 
  S / s

def shape_equilateral (a b c : ℝ) : Prop := 
  a = b ∧ b = c

theorem equilateral_triangle_of_altitude_sum (a b c h_a h_b h_c r S s : ℝ) 
  (habc : triangle a b c)
  (ha : h_a = altitude a b c S)
  (hb : h_b = altitude b a c S)
  (hc : h_c = altitude c a b S)
  (hr : r = inradius S s)
  (h_sum : h_a + h_b + h_c = 9 * r)
  (h_area : S = s * r)
  (h_semi : s = (a + b + c) / 2) : 
  shape_equilateral a b c := 
sorry

end equilateral_triangle_of_altitude_sum_l106_106062


namespace even_function_sum_eval_l106_106676

variable (v : ℝ → ℝ)

theorem even_function_sum_eval (h_even : ∀ x : ℝ, v x = v (-x)) :
    v (-2.33) + v (-0.81) + v (0.81) + v (2.33) = 2 * (v 2.33 + v 0.81) :=
by
  sorry

end even_function_sum_eval_l106_106676


namespace cakes_baked_yesterday_l106_106981

noncomputable def BakedToday : ℕ := 5
noncomputable def SoldDinner : ℕ := 6
noncomputable def Left : ℕ := 2

theorem cakes_baked_yesterday (CakesBakedYesterday : ℕ) : 
  BakedToday + CakesBakedYesterday - SoldDinner = Left → CakesBakedYesterday = 3 := 
by 
  intro h 
  sorry

end cakes_baked_yesterday_l106_106981


namespace length_OP_l106_106368

noncomputable def right_triangle_length_OP (XY XZ YZ : ℝ) (rO rP : ℝ) : ℝ :=
  let O := rO
  let P := rP
  -- Coordinates of point Y and Z can be O = (0, r), P = (OP, r)
  25 -- directly from the given correct answer

theorem length_OP (XY XZ YZ : ℝ) (rO rP : ℝ) (hXY : XY = 7) (hXZ : XZ = 24) (hYZ : YZ = 25) 
  (hO : rO = YZ - rO) (hP : rP = YZ - rP) : 
  right_triangle_length_OP XY XZ YZ rO rP = 25 :=
sorry

end length_OP_l106_106368


namespace alok_paid_rs_811_l106_106670

/-
 Assume Alok ordered the following items at the given prices:
 - 16 chapatis, each costing Rs. 6
 - 5 plates of rice, each costing Rs. 45
 - 7 plates of mixed vegetable, each costing Rs. 70
 - 6 ice-cream cups

 Prove that the total cost Alok paid is Rs. 811.
-/
theorem alok_paid_rs_811 :
  let chapati_cost := 6
  let rice_plate_cost := 45
  let mixed_vegetable_plate_cost := 70
  let chapatis := 16 * chapati_cost
  let rice_plates := 5 * rice_plate_cost
  let mixed_vegetable_plates := 7 * mixed_vegetable_plate_cost
  chapatis + rice_plates + mixed_vegetable_plates = 811 := by
  sorry

end alok_paid_rs_811_l106_106670


namespace number_of_biscuits_l106_106945

theorem number_of_biscuits (dough_length dough_width biscuit_length biscuit_width : ℕ)
    (h_dough : dough_length = 12) (h_dough_width : dough_width = 12)
    (h_biscuit_length : biscuit_length = 3) (h_biscuit_width : biscuit_width = 3)
    (dough_area : ℕ := dough_length * dough_width)
    (biscuit_area : ℕ := biscuit_length * biscuit_width) :
    dough_area / biscuit_area = 16 :=
by
  -- assume dough_area and biscuit_area are calculated from the given conditions
  -- dough_area = 144 and biscuit_area = 9
  sorry

end number_of_biscuits_l106_106945


namespace sprinkler_system_days_l106_106129

theorem sprinkler_system_days 
  (morning_water : ℕ) (evening_water : ℕ) (total_water : ℕ) 
  (h_morning : morning_water = 4) 
  (h_evening : evening_water = 6) 
  (h_total : total_water = 50) :
  total_water / (morning_water + evening_water) = 5 := 
by 
  sorry

end sprinkler_system_days_l106_106129


namespace fraction_in_between_l106_106470

variable {r u s v : ℤ}

/-- Assumes r, u, s, v be positive integers such that su - rv = 1 --/
theorem fraction_in_between (h1 : r > 0) (h2 : u > 0) (h3 : s > 0) (h4 : v > 0) (h5 : s * u - r * v = 1) :
  ∀ ⦃x num denom : ℤ⦄, r * denom = num * u → s * denom = (num + 1) * v → r * v ≤ num * denom - 1 / u * v * denom
   ∧ num * denom - 1 / u * v * denom ≤ s * v :=
sorry

end fraction_in_between_l106_106470


namespace likes_spinach_not_music_lover_l106_106843

universe u

variable (Person : Type u)
variable (likes_spinach is_pearl_diver is_music_lover : Person → Prop)

theorem likes_spinach_not_music_lover :
  (∃ x, likes_spinach x ∧ ¬ is_pearl_diver x) →
  (∀ x, is_music_lover x → (is_pearl_diver x ∨ ¬ likes_spinach x)) →
  (∀ x, (¬ is_pearl_diver x → is_music_lover x) ∨ (is_pearl_diver x → ¬ is_music_lover x)) →
  (∀ x, likes_spinach x → ¬ is_music_lover x) :=
by
  sorry

end likes_spinach_not_music_lover_l106_106843


namespace solution_set_of_inequality_l106_106733

theorem solution_set_of_inequality (a b x : ℝ) :
  (2 < x ∧ x < 3) → (x^2 - a * x - b < 0) →
  (a = 5 ∧ b = -6) →
  (bx^2 - a * x - 1 > 0) → ( - (1:ℝ)/2 < x ∧ x < - (1:ℝ)/3 ) :=
begin
  sorry
end

end solution_set_of_inequality_l106_106733


namespace force_for_18_inch_wrench_l106_106488

theorem force_for_18_inch_wrench (F : ℕ → ℕ → ℕ) : 
  (∀ L : ℕ, ∃ k : ℕ, F 300 12 = F (F L k) L) → 
  ((F 12 300) = 3600) → 
  (∀ k : ℕ, F (F 6 k) 6 = 3600) → 
  (∀ k : ℕ, F (F 18 k) 18 = 3600) → 
  (F 18 200 = 3600) :=
by
  sorry

end force_for_18_inch_wrench_l106_106488


namespace min_value_of_f_in_interval_l106_106443

def f (x k : ℝ) : ℝ := x^2 - k * x - 1

theorem min_value_of_f_in_interval (k : ℝ) :
  (f 1 k = -k ∧ k ≤ 2) ∨ 
  (∃ k', k' = 2 ∧ f (k'/2) k = - (k'^2) / 4 - 1 ∧ 2 < k ∧ k < 8) ∨ 
  (f 4 k = 15 - 4 * k ∧ k ≥ 8) :=
by sorry

end min_value_of_f_in_interval_l106_106443


namespace max_divisors_with_remainder_10_l106_106943

theorem max_divisors_with_remainder_10 (m : ℕ) :
  (m > 0) → (∀ k, (2008 % k = 10) ↔ k < m) → m = 11 :=
by
  sorry

end max_divisors_with_remainder_10_l106_106943


namespace bill_money_left_l106_106524

def bill_remaining_money (merchantA_qty : Int) (merchantA_rate : Int) 
                        (merchantB_qty : Int) (merchantB_rate : Int)
                        (fine : Int) (merchantC_qty : Int) (merchantC_rate : Int) 
                        (protection_costs : Int) (passerby_qty : Int) 
                        (passerby_rate : Int) : Int :=
let incomeA := merchantA_qty * merchantA_rate
let incomeB := merchantB_qty * merchantB_rate
let incomeC := merchantC_qty * merchantC_rate
let incomeD := passerby_qty * passerby_rate
let total_income := incomeA + incomeB + incomeC + incomeD
let total_expenses := fine + protection_costs
total_income - total_expenses

theorem bill_money_left 
    (merchantA_qty : Int := 8) 
    (merchantA_rate : Int := 9) 
    (merchantB_qty : Int := 15) 
    (merchantB_rate : Int := 11) 
    (fine : Int := 80)
    (merchantC_qty : Int := 25) 
    (merchantC_rate : Int := 8) 
    (protection_costs : Int := 30) 
    (passerby_qty : Int := 12) 
    (passerby_rate : Int := 7) : 
    bill_remaining_money merchantA_qty merchantA_rate 
                         merchantB_qty merchantB_rate 
                         fine merchantC_qty merchantC_rate 
                         protection_costs passerby_qty 
                         passerby_rate = 411 := by 
  sorry

end bill_money_left_l106_106524


namespace problems_per_worksheet_l106_106984

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ)
    (h1 : total_worksheets = 16) (h2 : graded_worksheets = 8) (h3 : remaining_problems = 32) :
    remaining_problems / (total_worksheets - graded_worksheets) = 4 :=
by
  sorry

end problems_per_worksheet_l106_106984


namespace mira_weekly_distance_l106_106476

noncomputable def total_distance_jogging : ℝ :=
  let monday_distance := 4 * 2
  let thursday_distance := 5 * 1.5
  monday_distance + thursday_distance

noncomputable def total_distance_swimming : ℝ :=
  2 * 1

noncomputable def total_distance_cycling : ℝ :=
  12 * 1

noncomputable def total_distance : ℝ :=
  total_distance_jogging + total_distance_swimming + total_distance_cycling

theorem mira_weekly_distance : total_distance = 29.5 := by
  unfold total_distance
  unfold total_distance_jogging
  unfold total_distance_swimming
  unfold total_distance_cycling
  sorry

end mira_weekly_distance_l106_106476


namespace good_numbers_l106_106269

/-- Definition of a good number -/
def is_good (n : ℕ) : Prop :=
  ∃ (k_1 k_2 k_3 k_4 : ℕ), 
    (1 ≤ k_1 ∧ 1 ≤ k_2 ∧ 1 ≤ k_3 ∧ 1 ≤ k_4) ∧
    (n + k_1 ∣ n + k_1^2) ∧ 
    (n + k_2 ∣ n + k_2^2) ∧ 
    (n + k_3 ∣ n + k_3^2) ∧ 
    (n + k_4 ∣ n + k_4^2) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ 
    (k_3 ≠ k_4)

/-- The main theorem to prove -/
theorem good_numbers : 
  is_good 58 ∧ 
  ∀ (p : ℕ), p > 2 → 
  (Prime p ∧ Prime (2 * p + 1) ↔ is_good (2 * p)) :=
by
  sorry

end good_numbers_l106_106269


namespace evaluate_expression_l106_106537

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l106_106537


namespace selection_at_most_one_l106_106568

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_at_most_one (A B : ℕ) :
  (combination 5 3) - (combination 3 1) = 7 :=
by
  sorry

end selection_at_most_one_l106_106568


namespace ratio_a_b_c_l106_106244

theorem ratio_a_b_c (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a + b + c) / 3 = 42) (h5 : a = 28) : 
  ∃ y z : ℕ, a / 28 = 1 ∧ b / (ky) = 1 / k ∧ c / (kz) = 1 / k ∧ (b + c) = 98 :=
by sorry

end ratio_a_b_c_l106_106244


namespace max_area_of_garden_l106_106769

theorem max_area_of_garden (l w : ℝ) 
  (h : 2 * l + w = 400) : 
  l * w ≤ 20000 :=
sorry

end max_area_of_garden_l106_106769


namespace total_cost_is_correct_l106_106699

-- Definitions of the conditions given
def price_iphone12 : ℝ := 800
def price_iwatch : ℝ := 300
def discount_iphone12 : ℝ := 0.15
def discount_iwatch : ℝ := 0.1
def cashback_discount : ℝ := 0.02

-- The final total cost after applying all discounts and cashback
def total_cost_after_discounts_and_cashback : ℝ :=
  let discount_amount_iphone12 := price_iphone12 * discount_iphone12
  let new_price_iphone12 := price_iphone12 - discount_amount_iphone12
  let discount_amount_iwatch := price_iwatch * discount_iwatch
  let new_price_iwatch := price_iwatch - discount_amount_iwatch
  let initial_total_cost := new_price_iphone12 + new_price_iwatch
  let cashback_amount := initial_total_cost * cashback_discount
  initial_total_cost - cashback_amount

-- Statement to be proved
theorem total_cost_is_correct :
  total_cost_after_discounts_and_cashback = 931 := by
  sorry

end total_cost_is_correct_l106_106699


namespace mixture_cost_in_july_l106_106375

theorem mixture_cost_in_july :
  (∀ C : ℝ, C > 0 → 
    (cost_green_tea_july : ℝ) = 0.1 → 
    (cost_green_tea_july = 0.1 * C) →
    (equal_quantities_mixture:  ℝ) = 1.5 →
    (cost_coffee_july: ℝ) = 2 * C →
    (total_mixture_cost: ℝ) = equal_quantities_mixture * cost_green_tea_july + equal_quantities_mixture * cost_coffee_july →
    total_mixture_cost = 3.15) :=
by
  sorry

end mixture_cost_in_july_l106_106375


namespace find_angle_B_l106_106029

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {m n : ℝ × ℝ}
variable (h1 : m = (Real.cos A, Real.sin A))
variable (h2 : n = (1, Real.sqrt 3))
variable (h3 : m.1 / n.1 = m.2 / n.2)
variable (h4 : a * Real.cos B + b * Real.cos A = c * Real.sin C)

theorem find_angle_B (h_conditions : a * Real.cos B + b * Real.cos A = c * Real.sin C) : B = Real.pi / 6 :=
sorry

end find_angle_B_l106_106029


namespace work_completion_l106_106402

theorem work_completion (A : ℝ) (B : ℝ) (work_duration : ℝ) (total_days : ℝ) (B_days : ℝ) :
  B_days = 28 ∧ total_days = 8 ∧ (A * 2 + (A * 6 + B * 6) = work_duration) →
  A = 84 / 11 :=
by
  sorry

end work_completion_l106_106402


namespace smallest_n_for_identity_matrix_l106_106559

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l106_106559


namespace vitya_convinced_of_12_models_l106_106111

noncomputable def min_offers_needed (n : ℕ) (k : ℕ) : ℕ :=
  if h : n = 13 then
    let ln100 := Real.log 100
    let ln13 := Real.log 13
    let ln12 := Real.log 12
    let req_k := Real.log 100 / (Real.log 13 - Real.log 12)
    if k > req_k then k else req_k.toNat + 1
  else k

theorem vitya_convinced_of_12_models (k : ℕ) : ∀ n, (n >= 13) → (min_offers_needed n k > 58) :=
by
  intros n h
  apply sorry

end vitya_convinced_of_12_models_l106_106111


namespace symmetric_circle_equation_l106_106014

theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (-x ^ 2 + y^2 + 4 * x = 0) :=
sorry

end symmetric_circle_equation_l106_106014


namespace abs_inequality_solution_l106_106245

theorem abs_inequality_solution (x : ℝ) :
  |x + 2| + |x - 2| ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end abs_inequality_solution_l106_106245


namespace marginal_cost_proof_l106_106628

theorem marginal_cost_proof (fixed_cost : ℕ) (total_cost : ℕ) (n : ℕ) (MC : ℕ)
  (h1 : fixed_cost = 12000)
  (h2 : total_cost = 16000)
  (h3 : n = 20)
  (h4 : total_cost = fixed_cost + MC * n) :
  MC = 200 :=
  sorry

end marginal_cost_proof_l106_106628


namespace Vitya_needs_58_offers_l106_106108

theorem Vitya_needs_58_offers :
  ∃ k : ℕ, (log 0.01 / log (12 / 13) < k) ∧ k = 58 :=
by
  sorry

end Vitya_needs_58_offers_l106_106108


namespace quadratic_expression_l106_106609

-- Definitions of roots and their properties
def quadratic_roots (r s : ℚ) : Prop :=
  (r + s = 5 / 3) ∧ (r * s = -8 / 3)

theorem quadratic_expression (r s : ℚ) (h : quadratic_roots r s) :
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
by
  sorry

end quadratic_expression_l106_106609


namespace square_side_length_l106_106834

theorem square_side_length {s : ℝ} (h1 : 4 * s = 60) : s = 15 := 
by
  linarith

end square_side_length_l106_106834


namespace triangle_altitude_l106_106233

theorem triangle_altitude (b : ℕ) (h : ℕ) (area : ℕ) (h_area : area = 800) (h_base : b = 40) (h_formula : area = (1 / 2) * b * h) : h = 40 :=
by
  sorry

end triangle_altitude_l106_106233


namespace arcsin_sqrt_three_over_two_l106_106691

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l106_106691


namespace range_of_a_l106_106339

noncomputable def f (x a : ℝ) : ℝ := 
  (1 / 2) * (Real.cos x + Real.sin x) * (Real.cos x - Real.sin x - 4 * a) + (4 * a - 3) * x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → 
  0 ≤ (Real.cos (2 * x) - 2 * a * (Real.sin x - Real.cos x) + 4 * a - 3)) ↔ (a ≥ 1.5) :=
sorry

end range_of_a_l106_106339


namespace probability_x_plus_2y_lt_6_l106_106663

noncomputable def prob_x_plus_2y_lt_6 : ℚ :=
  let rect_area : ℚ := (4 : ℚ) * 3
  let quad_area : ℚ := (4 : ℚ) * 1 + (1 / 2 : ℚ) * 4 * 2
  quad_area / rect_area

theorem probability_x_plus_2y_lt_6 :
  prob_x_plus_2y_lt_6 = 2 / 3 :=
by
  sorry

end probability_x_plus_2y_lt_6_l106_106663


namespace solution_l106_106549

noncomputable def polynomial_has_real_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - a * x^2 + a * x - 1 = 0

theorem solution (a : ℝ) : polynomial_has_real_root a :=
sorry

end solution_l106_106549


namespace apples_jackie_l106_106899

theorem apples_jackie (A : ℕ) (J : ℕ) (h1 : A = 8) (h2 : J = A + 2) : J = 10 := by
  -- Adam has 8 apples
  sorry

end apples_jackie_l106_106899


namespace exponential_decreasing_range_l106_106629

theorem exponential_decreasing_range (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x y : ℝ, x < y → a^y < a^x) : 0 < a ∧ a < 1 :=
by sorry

end exponential_decreasing_range_l106_106629


namespace sum_of_first_3n_terms_l106_106304

def arithmetic_geometric_sequence (n : ℕ) (s : ℕ → ℕ) :=
  (s n = 10) ∧ (s (2 * n) = 30)

theorem sum_of_first_3n_terms (n : ℕ) (s : ℕ → ℕ) :
  arithmetic_geometric_sequence n s → s (3 * n) = 70 :=
by
  intro h
  sorry

end sum_of_first_3n_terms_l106_106304


namespace average_number_of_visitors_is_25_l106_106205

-- Define the sequence parameters
def a : ℕ := 10  -- First term
def d : ℕ := 5   -- Common difference
def n : ℕ := 7   -- Number of days

-- Define the sequence for the number of visitors on each day
def visitors (i : ℕ) : ℕ := a + (i - 1) * d

-- Define the average number of visitors
def avg_visitors : ℕ := (List.sum (List.map visitors [1, 2, 3, 4, 5, 6, 7])) / n

-- Prove the average
theorem average_number_of_visitors_is_25 : avg_visitors = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end average_number_of_visitors_is_25_l106_106205


namespace boy_run_time_l106_106445

section
variables {d1 d2 d3 d4 : ℝ} -- distances
variables {v1 v2 v3 v4 : ℝ} -- velocities
variables {t : ℝ} -- time

-- Define conditions
def distances_and_velocities (d1 d2 d3 d4 v1 v2 v3 v4 : ℝ) :=
  d1 = 25 ∧ d2 = 30 ∧ d3 = 40 ∧ d4 = 35 ∧
  v1 = 3.33 ∧ v2 = 3.33 ∧ v3 = 2.78 ∧ v4 = 2.22

-- Problem statement
theorem boy_run_time
  (h : distances_and_velocities d1 d2 d3 d4 v1 v2 v3 v4) :
  t = (d1 / v1) + (d2 / v2) + (d3 / v3) + (d4 / v4) := 
sorry
end

end boy_run_time_l106_106445


namespace jennifer_interviews_both_clubs_l106_106463

theorem jennifer_interviews_both_clubs :
  let total_students := 30
  let chess_club := 22
  let drama_club := 20
  let both_clubs := chess_club + drama_club - total_students
  let only_chess := chess_club - both_clubs
  let only_drama := drama_club - both_clubs
  let total_ways := Nat.choose total_students 2
  let chess_ways := Nat.choose only_chess 2
  let drama_ways := Nat.choose only_drama 2
  let neither_ways := chess_ways + drama_ways
  1 - (neither_ways / total_ways) = 362 / 435 :=
by
  sorry

end jennifer_interviews_both_clubs_l106_106463


namespace solve_quadratic_l106_106771

noncomputable def quadratic_roots (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic : ∀ x : ℝ, quadratic_roots 1 (-4) (-5) x ↔ (x = -1 ∨ x = 5) :=
by
  intro x
  rw [quadratic_roots]
  sorry

end solve_quadratic_l106_106771


namespace ramu_selling_price_l106_106083

theorem ramu_selling_price (P R : ℝ) (profit_percent : ℝ) 
  (P_def : P = 42000)
  (R_def : R = 13000)
  (profit_percent_def : profit_percent = 17.272727272727273) :
  let total_cost := P + R
  let selling_price := total_cost * (1 + (profit_percent / 100))
  selling_price = 64500 := 
by
  sorry

end ramu_selling_price_l106_106083


namespace angle_bisector_theorem_l106_106985

noncomputable def angle_bisector_length (a b : ℝ) (C : ℝ) (CX : ℝ) : Prop :=
  C = 120 ∧
  CX = (a * b) / (a + b)

theorem angle_bisector_theorem (a b : ℝ) (C : ℝ) (CX : ℝ) :
  angle_bisector_length a b C CX :=
by
  sorry

end angle_bisector_theorem_l106_106985


namespace mark_owes_820_l106_106075

-- Definitions of the problem conditions
def base_fine : ℕ := 50
def over_speed_fine (mph_over : ℕ) : ℕ := mph_over * 2
def school_zone_multiplier : ℕ := 2
def court_costs : ℕ := 300
def lawyer_cost_per_hour : ℕ := 80
def lawyer_hours : ℕ := 3

-- Calculation of the total fine
def total_fine (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let mph_over := actual_speed - speed_limit
  let additional_fine := over_speed_fine mph_over
  let fine_before_multipliers := base_fine + additional_fine
  let fine_after_multipliers := fine_before_multipliers * school_zone_multiplier
  fine_after_multipliers

-- Calculation of the total costs
def total_costs (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let fine := total_fine speed_limit actual_speed
  fine + court_costs + (lawyer_cost_per_hour * lawyer_hours)

theorem mark_owes_820 : total_costs 30 75 = 820 := 
by
  sorry

end mark_owes_820_l106_106075


namespace man_is_older_by_20_l106_106136

variables (M S : ℕ)
axiom h1 : S = 18
axiom h2 : M + 2 = 2 * (S + 2)

theorem man_is_older_by_20 :
  M - S = 20 :=
by {
  sorry
}

end man_is_older_by_20_l106_106136


namespace drive_photos_storage_l106_106979

theorem drive_photos_storage (photo_size: ℝ) (num_photos_with_videos: ℕ) (photo_storage_with_videos: ℝ) (video_size: ℝ) (num_videos_with_photos: ℕ) : 
  num_photos_with_videos * photo_size + num_videos_with_photos * video_size = 3000 → 
  (3000 / photo_size) = 2000 :=
by
  sorry

end drive_photos_storage_l106_106979


namespace slow_car_speed_l106_106386

theorem slow_car_speed (x : ℝ) (hx : 0 < x) (distance : ℝ) (delay : ℝ) (fast_factor : ℝ) :
  distance = 60 ∧ delay = 0.5 ∧ fast_factor = 1.5 ∧ 
  (distance / x) - (distance / (fast_factor * x)) = delay → 
  x = 40 :=
by
  intros h
  sorry

end slow_car_speed_l106_106386


namespace quadratic_no_real_roots_l106_106342

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 - 3 * x - k ≠ 0) → k < -9 / 4 :=
by
  sorry

end quadratic_no_real_roots_l106_106342


namespace min_value_ineq_l106_106723

noncomputable def a_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a 2018 = a 2017 + 2 * a 2016

theorem min_value_ineq (a : ℕ → ℝ) (m n : ℕ) 
  (h : a_sequence a) 
  (h2 : a m * a n = 16 * (a 1) ^ 2) :
  (4 / m) + (1 / n) ≥ 5 / 3 :=
sorry

end min_value_ineq_l106_106723


namespace variance_transformation_example_l106_106595

def variance (X : List ℝ) : ℝ := sorry -- Assuming some definition of variance

theorem variance_transformation_example {n : ℕ} (X : List ℝ) (h_len : X.length = 2021) (h_var : variance X = 3) :
  variance (X.map (fun x => 3 * (x - 2))) = 27 := 
sorry

end variance_transformation_example_l106_106595


namespace moles_H2O_formed_l106_106582

-- Define the conditions
def moles_HCl : ℕ := 6
def moles_CaCO3 : ℕ := 3
def moles_CaCl2 : ℕ := 3
def moles_CO2 : ℕ := 3

-- Proposition that we need to prove
theorem moles_H2O_formed : moles_CaCl2 = 3 ∧ moles_CO2 = 3 ∧ moles_CaCO3 = 3 ∧ moles_HCl = 6 → moles_CaCO3 = 3 := by
  sorry

end moles_H2O_formed_l106_106582


namespace annual_sales_profit_relationship_and_maximum_l106_106827

def cost_per_unit : ℝ := 6
def selling_price (x : ℝ) := x > 6
def sales_volume (u : ℝ) := u * 10000
def proportional_condition (x u : ℝ) := (585 / 8) - u = 2 * (x - 21 / 4) ^ 2
def sales_volume_condition : Prop := proportional_condition 10 28

theorem annual_sales_profit_relationship_and_maximum (x u y : ℝ) 
    (hx : selling_price x) 
    (hu : proportional_condition x u) 
    (hs : sales_volume_condition) :
    (y = (-2 * x^3 + 33 * x^2 - 108 * x - 108)) ∧ 
    (x = 9 → y = 135) := 
sorry

end annual_sales_profit_relationship_and_maximum_l106_106827


namespace maximum_correct_answers_l106_106193

theorem maximum_correct_answers (c w u : ℕ) :
  c + w + u = 25 →
  4 * c - w = 70 →
  c ≤ 19 :=
by
  sorry

end maximum_correct_answers_l106_106193


namespace cos_identity_arithmetic_sequence_in_triangle_l106_106891

theorem cos_identity_arithmetic_sequence_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h1 : 2 * b = a + c)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : A + B + C = Real.pi)
  : 5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 4 := 
  sorry

end cos_identity_arithmetic_sequence_in_triangle_l106_106891


namespace paving_cost_l106_106815

variable (L : ℝ) (W : ℝ) (R : ℝ)

def area (L W : ℝ) := L * W
def cost (A R : ℝ) := A * R

theorem paving_cost (hL : L = 5) (hW : W = 4.75) (hR : R = 900) : cost (area L W) R = 21375 :=
by
  sorry

end paving_cost_l106_106815


namespace total_money_correct_l106_106682

def total_money_in_cents : ℕ :=
  let Cindy := 5 * 10 + 3 * 50
  let Eric := 3 * 25 + 2 * 100 + 1 * 50
  let Garrick := 8 * 5 + 7 * 1
  let Ivy := 60 * 1 + 5 * 25
  let TotalBeforeRemoval := Cindy + Eric + Garrick + Ivy
  let BeaumontRemoval := 2 * 10 + 3 * 5 + 10 * 1
  let EricRemoval := 1 * 25 + 1 * 50
  TotalBeforeRemoval - BeaumontRemoval - EricRemoval

theorem total_money_correct : total_money_in_cents = 637 := by
  sorry

end total_money_correct_l106_106682


namespace eval_expression_l106_106540

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l106_106540


namespace molecular_weight_of_acetic_acid_l106_106949

-- Define the molecular weight of 7 moles of acetic acid
def molecular_weight_7_moles_acetic_acid := 420 

-- Define the number of moles of acetic acid
def moles_acetic_acid := 7

-- Define the molecular weight of 1 mole of acetic acid
def molecular_weight_1_mole_acetic_acid := molecular_weight_7_moles_acetic_acid / moles_acetic_acid

-- The theorem stating that given the molecular weight of 7 moles of acetic acid, we have the molecular weight of the acetic acid
theorem molecular_weight_of_acetic_acid : molecular_weight_1_mole_acetic_acid = 60 := by
  -- proof to be solved
  sorry

end molecular_weight_of_acetic_acid_l106_106949


namespace quadratic_ineq_real_solutions_l106_106008

theorem quadratic_ineq_real_solutions (d : ℝ) (h₀ : 0 < d) :
  (∀ x : ℝ, x^2 - 8 * x + d < 0 → 0 < d ∧ d < 16) :=
by
  sorry

end quadratic_ineq_real_solutions_l106_106008


namespace smallest_is_B_l106_106105

def A : ℕ := 32 + 7
def B : ℕ := (3 * 10) + 3
def C : ℕ := 50 - 9

theorem smallest_is_B : min A (min B C) = B := 
by 
  have hA : A = 39 := by rfl
  have hB : B = 33 := by rfl
  have hC : C = 41 := by rfl
  rw [hA, hB, hC]
  exact sorry

end smallest_is_B_l106_106105


namespace arithmetic_sequence_tenth_term_l106_106636

noncomputable def prove_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : Prop :=
  a + 9*d = 38

theorem arithmetic_sequence_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : prove_tenth_term a d h1 h2 :=
by
  sorry

end arithmetic_sequence_tenth_term_l106_106636


namespace work_completion_l106_106501

theorem work_completion (W : ℕ) (n : ℕ) (h1 : 0 < n) (H1 : 0 < W) :
  (∀ w : ℕ, w ≤ W / n) → 
  (∀ k : ℕ, k = (7 * n) / 10 → k * (3 * W) / (10 * n) ≥ W / 3) → 
  (∀ m : ℕ, m = (3 * n) / 10 → m * (7 * W) / (10 * n) ≥ W / 3) → 
  ∃ g1 g2 g3 : ℕ, g1 + g2 + g3 < W / 3 :=
by
  sorry

end work_completion_l106_106501


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l106_106686

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l106_106686


namespace time_spent_on_type_a_problems_l106_106814

-- Define the conditions
def total_questions := 200
def examination_duration_hours := 3
def type_a_problems := 100
def type_b_problems := total_questions - type_a_problems
def type_a_time_coeff := 2

-- Convert examination duration to minutes
def examination_duration_minutes := examination_duration_hours * 60

-- Variables for time per problem
variable (x : ℝ)

-- The total time spent
def total_time_spent : ℝ := type_a_problems * (type_a_time_coeff * x) + type_b_problems * x

-- Statement we need to prove
theorem time_spent_on_type_a_problems :
  total_time_spent x = examination_duration_minutes → type_a_problems * (type_a_time_coeff * x) = 120 :=
by
  sorry

end time_spent_on_type_a_problems_l106_106814


namespace probability_in_interval_l106_106751

open Probability

noncomputable def probability_event_in_interval (a : ℝ) : Prop :=
  2 + a - a^2 > 0

theorem probability_in_interval : 
  (1/2 : ℝ) = P (λ a, a ∈ set.Icc (-1) 2) (uniform (set.Icc (-3) 3)) :=
begin
  simp only [uniform, interval_integral, set_integral, Icc, Ioc, Pi.smul_apply],
  sorry
-- the details of the integral and probability calculation would go here, omitted by "sorry"
end

end probability_in_interval_l106_106751


namespace eight_in_C_l106_106209

def C : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

theorem eight_in_C : 8 ∈ C :=
by {
  sorry
}

end eight_in_C_l106_106209


namespace solve_inequality_l106_106794

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l106_106794


namespace symmetrical_character_l106_106650

def symmetrical (char : String) : Prop :=
  -- Define a predicate symmetrical which checks if a given character
  -- is a symmetrical figure somehow. This needs to be implemented
  -- properly based on the graphical property of the character.
  sorry 

theorem symmetrical_character :
  ∀ (c : String), (c = "幸" → symmetrical c) ∧ 
                  (c = "福" → ¬ symmetrical c) ∧ 
                  (c = "惠" → ¬ symmetrical c) ∧ 
                  (c = "州" → ¬ symmetrical c) :=
by
  sorry

end symmetrical_character_l106_106650


namespace triangle_angles_and_side_l106_106347

noncomputable def triangle_properties : Type := sorry

variables {A B C : ℝ}
variables {a b c : ℝ}

theorem triangle_angles_and_side (hA : A = 60)
    (ha : a = 4 * Real.sqrt 3)
    (hb : b = 4 * Real.sqrt 2)
    (habc : triangle_properties)
    : B = 45 ∧ C = 75 ∧ c = 2 * Real.sqrt 2 + 2 * Real.sqrt 6 := 
sorry

end triangle_angles_and_side_l106_106347


namespace rhombus_side_length_l106_106594

theorem rhombus_side_length
  (a b : ℝ)
  (h_eq : ∀ x, x^2 - 10*x + ((x - a) * (x - b)) = 0)
  (h_area : (1/2) * a * b = 11) :
  sqrt ((a + b)^2 / 4 - ab / 2) = sqrt 14 :=
by
  sorry

end rhombus_side_length_l106_106594


namespace total_books_l106_106817

theorem total_books (shelves_mystery shelves_picture : ℕ) (books_per_shelf : ℕ) 
    (h_mystery : shelves_mystery = 5) (h_picture : shelves_picture = 4) (h_books_per_shelf : books_per_shelf = 6) : 
    shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf = 54 := 
by 
  sorry

end total_books_l106_106817


namespace find_x_of_series_eq_16_l106_106589

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x ^ n

theorem find_x_of_series_eq_16 (x : ℝ) (h : series_sum x = 16) : x = (4 - Real.sqrt 2) / 4 :=
by
  sorry

end find_x_of_series_eq_16_l106_106589


namespace line_equation_l106_106383

theorem line_equation (P : ℝ × ℝ) (hP : P = (-2, 1)) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -5 := by
  sorry

end line_equation_l106_106383


namespace sci_not_218000_l106_106196

theorem sci_not_218000 : 218000 = 2.18 * 10^5 :=
by
  sorry

end sci_not_218000_l106_106196


namespace find_original_wage_l106_106523

theorem find_original_wage (W : ℝ) (h : 1.50 * W = 51) : W = 34 :=
sorry

end find_original_wage_l106_106523


namespace alok_paid_rs_811_l106_106669

/-
 Assume Alok ordered the following items at the given prices:
 - 16 chapatis, each costing Rs. 6
 - 5 plates of rice, each costing Rs. 45
 - 7 plates of mixed vegetable, each costing Rs. 70
 - 6 ice-cream cups

 Prove that the total cost Alok paid is Rs. 811.
-/
theorem alok_paid_rs_811 :
  let chapati_cost := 6
  let rice_plate_cost := 45
  let mixed_vegetable_plate_cost := 70
  let chapatis := 16 * chapati_cost
  let rice_plates := 5 * rice_plate_cost
  let mixed_vegetable_plates := 7 * mixed_vegetable_plate_cost
  chapatis + rice_plates + mixed_vegetable_plates = 811 := by
  sorry

end alok_paid_rs_811_l106_106669


namespace rhombus_diagonal_sum_l106_106833

theorem rhombus_diagonal_sum
  (d1 d2 : ℝ)
  (h1 : d1 ≤ 6)
  (h2 : 6 ≤ d2)
  (side_len : ℝ)
  (h_side : side_len = 5)
  (rhombus_relation : d1^2 + d2^2 = 4 * side_len^2) :
  d1 + d2 ≤ 14 :=
sorry

end rhombus_diagonal_sum_l106_106833


namespace max_possible_K_l106_106284

theorem max_possible_K :
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let total_sum := ∑ i in nums, i
  total_sum = 55 →
  (∃ (A B : Finset ℕ), A ∪ B = nums ∧ A ∩ B = ∅ ∧ total_sum = (∑ i in A, i) + (∑ i in B, i) ∧ 
    ∃ (K : ℕ), K = min (∑ i in A, i * ∑ i in B, i) ∧ K ≤ 756) :=
by 
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let total_sum := ∑ i in nums, i
  show total_sum = 55 → (∃ A B : Finset ℕ, A ∪ B = nums ∧ A ∩ B = ∅ 
    ∧ total_sum = ((∑ i in A, i) + (∑ i in B, i)) 
    ∧ ∃ (K : ℕ), K = min ((∑ i in A, i) * (∑ i in B, i)) ∧ K ≤ 756)
  from sorry

end max_possible_K_l106_106284


namespace Vitya_needs_58_offers_l106_106109

theorem Vitya_needs_58_offers :
  ∃ k : ℕ, (log 0.01 / log (12 / 13) < k) ∧ k = 58 :=
by
  sorry

end Vitya_needs_58_offers_l106_106109


namespace largest_determinable_1986_l106_106816

-- Define main problem with conditions
def largest_determinable_cards (total : ℕ) (select : ℕ) : ℕ :=
  total - 27

-- Statement we need to prove
theorem largest_determinable_1986 :
  largest_determinable_cards 2013 10 = 1986 :=
by
  sorry

end largest_determinable_1986_l106_106816


namespace romanov_family_savings_l106_106513

theorem romanov_family_savings :
  let cost_multi_tariff_meter := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let night_consumption := 230
  let day_consumption := monthly_consumption - night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let yearly_cost_multi_tariff :=
    (night_consumption * night_rate * 12) +
    (day_consumption * day_rate * 12)
  let total_cost_multi_tariff :=
    cost_multi_tariff_meter + installation_cost + (yearly_cost_multi_tariff * 3)
  let yearly_cost_standard :=
    monthly_consumption * standard_rate * 12
  let total_cost_standard :=
    yearly_cost_standard * 3
  total_cost_standard - total_cost_multi_tariff = 3824 := 
by {
  sorry -- Proof goes here
}

end romanov_family_savings_l106_106513


namespace system1_solution_system2_solution_l106_106227

-- System (1)
theorem system1_solution {x y : ℝ} : 
  x + y = 3 → 
  x - y = 1 → 
  (x = 2 ∧ y = 1) :=
by
  intros h1 h2
  -- proof goes here
  sorry

-- System (2)
theorem system2_solution {x y : ℝ} :
  2 * x + y = 3 →
  x - 2 * y = 1 →
  (x = 7 / 5 ∧ y = 1 / 5) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end system1_solution_system2_solution_l106_106227


namespace find_an_find_n_l106_106387

noncomputable def a_n (n : ℕ) : ℤ := 12 + (n - 1) * 2

noncomputable def S_n (n : ℕ) : ℤ := n * 12 + (n * (n - 1) / 2) * 2

theorem find_an (n : ℕ) : a_n n = 2 * n + 10 :=
by sorry

theorem find_n (n : ℕ) (S_n : ℤ) : S_n = 242 → n = 11 :=
by sorry

end find_an_find_n_l106_106387


namespace andy_remaining_demerits_l106_106000

-- Definitions based on conditions
def max_demerits : ℕ := 50
def demerits_per_late_instance : ℕ := 2
def late_instances : ℕ := 6
def joke_demerits : ℕ := 15

-- Calculation of total demerits for the month
def total_demerits : ℕ := (demerits_per_late_instance * late_instances) + joke_demerits

-- Proof statement: Andy can receive 23 more demerits without being fired
theorem andy_remaining_demerits : max_demerits - total_demerits = 23 :=
by
  -- Placeholder for proof
  sorry

end andy_remaining_demerits_l106_106000


namespace parabola_intersection_ratios_l106_106325

noncomputable def parabola_vertex_x1 (a b c : ℝ) := -b / (2 * a)
noncomputable def parabola_vertex_y1 (a b c : ℝ) := (4 * a * c - b^2) / (4 * a)
noncomputable def parabola_vertex_x2 (a d e : ℝ) := d / (2 * a)
noncomputable def parabola_vertex_y2 (a d e : ℝ) := (4 * a * e + d^2) / (4 * a)

theorem parabola_intersection_ratios
  (a b c d e : ℝ)
  (h1 : 144 * a + 12 * b + c = 21)
  (h2 : 784 * a + 28 * b + c = 3)
  (h3 : -144 * a + 12 * d + e = 21)
  (h4 : -784 * a + 28 * d + e = 3) :
  (parabola_vertex_x1 a b c + parabola_vertex_x2 a d e) / 
  (parabola_vertex_y1 a b c + parabola_vertex_y2 a d e) = 5 / 3 := by
  sorry

end parabola_intersection_ratios_l106_106325


namespace dilation_image_l106_106376

theorem dilation_image (z : ℂ) (c : ℂ) (k : ℝ) (w : ℂ) (h₁ : c = 0 + 5 * I) 
  (h₂ : k = 3) (h₃ : w = 3 + 2 * I) : z = 9 - 4 * I :=
by
  -- Given conditions
  have hc : c = 0 + 5 * I := h₁
  have hk : k = 3 := h₂
  have hw : w = 3 + 2 * I := h₃

  -- Dilation formula
  let formula := (w - c) * k + c

  -- Prove the result
  -- sorry for now, the proof is not required as per instructions
  sorry

end dilation_image_l106_106376


namespace calculate_fraction_l106_106678

theorem calculate_fraction : (5 / (8 / 13) / (10 / 7) = 91 / 16) :=
by
  sorry

end calculate_fraction_l106_106678


namespace sprinkler_system_days_l106_106128

theorem sprinkler_system_days 
  (morning_water : ℕ) (evening_water : ℕ) (total_water : ℕ) 
  (h_morning : morning_water = 4) 
  (h_evening : evening_water = 6) 
  (h_total : total_water = 50) :
  total_water / (morning_water + evening_water) = 5 := 
by 
  sorry

end sprinkler_system_days_l106_106128


namespace total_earnings_first_three_months_l106_106326

-- Definitions
def earning_first_month : ℕ := 350
def earning_second_month : ℕ := 2 * earning_first_month + 50
def earning_third_month : ℕ := 4 * (earning_first_month + earning_second_month)

-- Question restated as a theorem
theorem total_earnings_first_three_months : 
  (earning_first_month + earning_second_month + earning_third_month = 5500) :=
by 
  -- Placeholder for the proof
  sorry

end total_earnings_first_three_months_l106_106326


namespace problem_3034_1002_20_04_div_sub_l106_106123

theorem problem_3034_1002_20_04_div_sub:
  3034 - (1002 / 20.04) = 2984 :=
by
  sorry

end problem_3034_1002_20_04_div_sub_l106_106123


namespace quadratic_root_sum_product_l106_106317

theorem quadratic_root_sum_product (m n : ℝ)
  (h1 : m + n = 4)
  (h2 : m * n = -1) :
  m + n - m * n = 5 :=
sorry

end quadratic_root_sum_product_l106_106317


namespace vitya_convinced_of_12_models_l106_106110

noncomputable def min_offers_needed (n : ℕ) (k : ℕ) : ℕ :=
  if h : n = 13 then
    let ln100 := Real.log 100
    let ln13 := Real.log 13
    let ln12 := Real.log 12
    let req_k := Real.log 100 / (Real.log 13 - Real.log 12)
    if k > req_k then k else req_k.toNat + 1
  else k

theorem vitya_convinced_of_12_models (k : ℕ) : ∀ n, (n >= 13) → (min_offers_needed n k > 58) :=
by
  intros n h
  apply sorry

end vitya_convinced_of_12_models_l106_106110


namespace point_on_line_and_equidistant_l106_106441

theorem point_on_line_and_equidistant {x y : ℝ} :
  (4 * x + 3 * y = 12) ∧ (x = y) ∧ (x ≥ 0) ∧ (y ≥ 0) ↔ x = 12 / 7 ∧ y = 12 / 7 :=
by
  sorry

end point_on_line_and_equidistant_l106_106441


namespace system_of_equations_correct_l106_106821

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end system_of_equations_correct_l106_106821


namespace multiplication_even_a_b_multiplication_even_a_a_l106_106897

def a : Int := 4
def b : Int := 3

theorem multiplication_even_a_b : a * b = 12 := by sorry
theorem multiplication_even_a_a : a * a = 16 := by sorry

end multiplication_even_a_b_multiplication_even_a_a_l106_106897


namespace total_teachers_correct_l106_106482

-- Define the number of departments and the total number of teachers
def num_departments : ℕ := 7
def total_teachers : ℕ := 140

-- Proving that the total number of teachers is 140
theorem total_teachers_correct : total_teachers = 140 := 
by
  sorry

end total_teachers_correct_l106_106482


namespace weights_system_l106_106820

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end weights_system_l106_106820


namespace penny_dime_halfdollar_probability_l106_106931

-- Define the universe of outcomes for five coin flips
def coin_flip_outcomes : Finset (vector Bool 5) := Finset.univ

-- Define a predicate that checks if the penny, dime, and half-dollar come up the same
def same_penny_dime_halfdollar (v : vector Bool 5) : Prop :=
  v.head = v.nth 2 ∧ v.head = v.nth 4

-- The proof problem: prove that the probability of the penny, dime, and half-dollar being the same is 1/4
theorem penny_dime_halfdollar_probability :
  (Finset.filter same_penny_dime_halfdollar coin_flip_outcomes).card / coin_flip_outcomes.card = 1 / 4 :=
by
  sorry

end penny_dime_halfdollar_probability_l106_106931


namespace seating_arrangements_l106_106281

def person := {Alice, Bob, Carla, Derek, Eric}

def conditions (seating : list person) : Prop :=
  ¬( (seating.indexOf Alice = 1 ∧ (seating.indexOf Derek = 2 ∨ seating.indexOf Eric = 2))
   ∨ (seating.indexOf Alice = 2 ∧ (seating.indexOf Derek = 1 ∨ seating.indexOf Eric = 3))
   ∨ (seating.indexOf Alice = 3 ∧ (seating.indexOf Derek = 2 ∨ seating.indexOf Eric = 4))
   ∨ (seating.indexOf Alice = 4 ∧ (seating.indexOf Derek = 3 ∨ seating.indexOf Eric = 5))
   ∨ (seating.indexOf Alice = 5 ∧ (seating.indexOf Derek = 4 ∨ seating.indexOf Eric = 4))
   ∨ (seating.indexOf Carla = 1 ∧ seating.indexOf Derek = 2)
   ∨ (seating.indexOf Carla = 2 ∧ (seating.indexOf Derek = 1 ∨ seating.indexOf Derek = 3))
   ∨ (seating.indexOf Carla = 3 ∧ (seating.indexOf Derek = 2 ∨ seating.indexOf Derek = 4))
   ∨ (seating.indexOf Carla = 4 ∧ (seating.indexOf Derek = 3 ∨ seating.indexOf Derek = 5))
   ∨ (seating.indexOf Carla = 5 ∧ seating.indexOf Derek = 4))

theorem seating_arrangements : (finset.persons.attach.filter conditions).card = 20 :=
sorry

end seating_arrangements_l106_106281


namespace soccer_match_outcome_l106_106937

theorem soccer_match_outcome :
  ∃ n : ℕ, n = 4 ∧
  (∃ (num_wins num_draws num_losses : ℕ),
     num_wins * 3 + num_draws * 1 + num_losses * 0 = 19 ∧
     num_wins + num_draws + num_losses = 14) :=
sorry

end soccer_match_outcome_l106_106937


namespace pairs_sold_l106_106510

theorem pairs_sold (total_sales : ℝ) (avg_price_per_pair : ℝ) (h1 : total_sales = 490) (h2 : avg_price_per_pair = 9.8) :
  total_sales / avg_price_per_pair = 50 :=
by
  rw [h1, h2]
  norm_num

end pairs_sold_l106_106510


namespace agatha_initial_money_60_l106_106278

def Agatha_initial_money (spent_frame : ℕ) (spent_front_wheel: ℕ) (left_over: ℕ) : ℕ :=
  spent_frame + spent_front_wheel + left_over

theorem agatha_initial_money_60 :
  Agatha_initial_money 15 25 20 = 60 :=
by
  -- This line assumes $15 on frame, $25 on wheel, $20 left translates to a total of $60.
  sorry

end agatha_initial_money_60_l106_106278


namespace number_of_pizzas_l106_106391

-- Define the conditions
def slices_per_pizza := 8
def total_slices := 168

-- Define the statement we want to prove
theorem number_of_pizzas : total_slices / slices_per_pizza = 21 :=
by
  -- Proof goes here
  sorry

end number_of_pizzas_l106_106391


namespace sum_arithmetic_sequence_l106_106940

theorem sum_arithmetic_sequence (m : ℕ) (S : ℕ → ℕ) 
  (h1 : S m = 30) 
  (h2 : S (3 * m) = 90) : 
  S (2 * m) = 60 := 
sorry

end sum_arithmetic_sequence_l106_106940


namespace sum_of_decimals_l106_106416

theorem sum_of_decimals : (5.76 + 4.29 = 10.05) :=
by
  sorry

end sum_of_decimals_l106_106416


namespace abs_sum_eq_abs_add_iff_ab_gt_zero_l106_106575

theorem abs_sum_eq_abs_add_iff_ab_gt_zero (a b : ℝ) :
  (|a + b| = |a| + |b|) → (a = 0 ∧ b = 0 ∨ ab > 0) :=
sorry

end abs_sum_eq_abs_add_iff_ab_gt_zero_l106_106575


namespace digit_possibilities_for_mod4_count_possibilities_is_3_l106_106520

theorem digit_possibilities_for_mod4 (N : ℕ) (h : N < 10): 
  (80 + N) % 4 = 0 → N = 0 ∨ N = 4 ∨ N = 8 → true := 
by
  -- proof is not needed
  sorry

def count_possibilities : ℕ := 
  (if (80 + 0) % 4 = 0 then 1 else 0) +
  (if (80 + 1) % 4 = 0 then 1 else 0) +
  (if (80 + 2) % 4 = 0 then 1 else 0) +
  (if (80 + 3) % 4 = 0 then 1 else 0) +
  (if (80 + 4) % 4 = 0 then 1 else 0) +
  (if (80 + 5) % 4 = 0 then 1 else 0) +
  (if (80 + 6) % 4 = 0 then 1 else 0) +
  (if (80 + 7) % 4 = 0 then 1 else 0) +
  (if (80 + 8) % 4 = 0 then 1 else 0) +
  (if (80 + 9) % 4 = 0 then 1 else 0)

theorem count_possibilities_is_3: count_possibilities = 3 := 
by
  -- proof is not needed
  sorry

end digit_possibilities_for_mod4_count_possibilities_is_3_l106_106520


namespace particle_hits_origin_l106_106406

def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x, 0 => 0
| 0, y => 0
| x+1, y+1 => 0.25 * P x (y+1) + 0.25 * P (x+1) y + 0.5 * P x y

theorem particle_hits_origin :
    ∃ m n : ℕ, m ≠ 0 ∧ m % 4 ≠ 0 ∧ P 5 5 = m / 4^n :=
sorry

end particle_hits_origin_l106_106406


namespace count_integers_M_3_k_l106_106581

theorem count_integers_M_3_k (M : ℕ) (hM : M < 500) :
  (∃ k : ℕ, k ≥ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ M = 2 * k * (m + k - 1)) ∧
  (∃ k1 k2 k3 k4 : ℕ, k1 ≠ k2 ∧ k1 ≠ k3 ∧ k1 ≠ k4 ∧
    k2 ≠ k3 ∧ k2 ≠ k4 ∧ k3 ≠ k4 ∧
    (M / 2 = (k1 + k2 + k3 + k4) ∨ M / 2 = (k1 * k2 * k3 * k4))) →
  (∃ n : ℕ, n = 6) :=
by
  sorry

end count_integers_M_3_k_l106_106581


namespace emily_sixth_quiz_score_l106_106706

-- Define the scores Emily has received
def scores : List ℕ := [92, 96, 87, 89, 100]

-- Define the number of quizzes
def num_quizzes : ℕ := 6

-- Define the desired average score
def desired_average : ℕ := 94

-- The theorem to prove the score Emily needs on her sixth quiz to achieve the desired average
theorem emily_sixth_quiz_score : ∃ (x : ℕ), List.sum scores + x = desired_average * num_quizzes := by
  sorry

end emily_sixth_quiz_score_l106_106706


namespace negation_of_proposition_l106_106725

theorem negation_of_proposition (a b : ℝ) : 
  ¬(a + b = 1 → a^2 + b^2 ≥ 1/2) ↔ (a + b ≠ 1 → a^2 + b^2 < 1/2) :=
by sorry

end negation_of_proposition_l106_106725


namespace difference_nickels_is_8q_minus_20_l106_106280

variable (q : ℤ)

-- Define the number of quarters for Alice and Bob
def alice_quarters : ℤ := 7 * q - 3
def bob_quarters : ℤ := 3 * q + 7

-- Define the worth of a quarter in nickels
def quarter_to_nickels (quarters : ℤ) : ℤ := 2 * quarters

-- Define the difference in quarters
def difference_quarters : ℤ := alice_quarters q - bob_quarters q

-- Define the difference in their amount of money in nickels
def difference_nickels (q : ℤ) : ℤ := quarter_to_nickels (difference_quarters q)

theorem difference_nickels_is_8q_minus_20 : difference_nickels q = 8 * q - 20 := by
  sorry

end difference_nickels_is_8q_minus_20_l106_106280


namespace incorrect_statement_A_l106_106167

theorem incorrect_statement_A (m : ℝ) :
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  ∃ x : ℝ, y x = m^2 - 9 → False :=
by
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  have minY : ∀ x, y x ≥ -9 := 
    sorry  
  have h : ∀ x, y x = m^2 - 9 → x = m ∧ -9 = m^2 - 9 :=
    sorry
  obtain ⟨x, hx⟩ := h
  have eq := minY x
  rw [hx] at eq
  exact eq, 
  sorry


termination_with
termination_axiom _ := 
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  ∃ x : ℝ, y x = -9 :=
by 
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  have minY : (x - m)^2 ≥ 0 :=
    by 
      sorry 
  ∃ x : ℝ, y x = -9 :=
    sorry  
ē

end incorrect_statement_A_l106_106167


namespace algebraic_expression_value_l106_106434

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : -2 * a^2 + 8 * a - 5 = 1 := 
by 
  sorry 

end algebraic_expression_value_l106_106434


namespace verify_solution_l106_106250

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x - y = 9
def condition2 : Prop := 4 * x + 3 * y = 1

-- Proof problem statement
theorem verify_solution
  (h1 : condition1 x y)
  (h2 : condition2 x y) :
  x = 4 ∧ y = -5 :=
sorry

end verify_solution_l106_106250


namespace combined_distance_is_twelve_l106_106642

-- Definitions based on the conditions
def distance_second_lady : ℕ := 4
def distance_first_lady : ℕ := 2 * distance_second_lady
def total_distance : ℕ := distance_second_lady + distance_first_lady

-- Theorem statement
theorem combined_distance_is_twelve : total_distance = 12 := by
  sorry

end combined_distance_is_twelve_l106_106642


namespace solution_set_of_inequality_l106_106938

theorem solution_set_of_inequality:
  {x : ℝ | 3 ≤ |2 - x| ∧ |2 - x| < 9} = {x : ℝ | (-7 < x ∧ x ≤ -1) ∨ (5 ≤ x ∧ x < 11)} :=
by
  sorry

end solution_set_of_inequality_l106_106938


namespace colors_used_l106_106142

theorem colors_used (total_blocks number_per_color : ℕ) (h1 : total_blocks = 196) (h2 : number_per_color = 14) : 
  total_blocks / number_per_color = 14 :=
by
  sorry

end colors_used_l106_106142


namespace arcsin_sqrt3_div_2_l106_106683

theorem arcsin_sqrt3_div_2 :
  ∃ θ : ℝ, θ ∈ Icc (-(Real.pi / 2)) (Real.pi / 2) ∧ Real.sin θ = (Real.sqrt 3) / 2 ∧ Real.arcsin ((Real.sqrt 3) / 2) = θ ∧ θ = (Real.pi / 3) :=
by
  sorry

end arcsin_sqrt3_div_2_l106_106683


namespace arc_length_sector_l106_106336

-- Given conditions
def theta := 90
def r := 6

-- Formula for arc length of a sector
def arc_length (theta : ℕ) (r : ℕ) : ℝ :=
  (theta : ℝ) / 360 * 2 * Real.pi * r

-- Proving the arc length for given theta and radius
theorem arc_length_sector : arc_length theta r = 3 * Real.pi :=
sorry

end arc_length_sector_l106_106336


namespace exists_n_sum_reciprocals_lt_2022_l106_106623

theorem exists_n_sum_reciprocals_lt_2022 :
  ∃ (n : ℕ), 91 ≤ n ∧ (∑ i in Finset.range (n + 1), (nat.choose n i)⁻¹) < 2.022 :=
begin
  -- To be proved
  sorry
end

end exists_n_sum_reciprocals_lt_2022_l106_106623


namespace find_x_l106_106037

def vector (α : Type*) := α × α

def parallel (a b : vector ℝ) : Prop :=
a.1 * b.2 - a.2 * b.1 = 0

theorem find_x (x : ℝ) (a b : vector ℝ)
  (ha : a = (1, 2))
  (hb : b = (x, 4))
  (h : parallel a b) : x = 2 :=
by sorry

end find_x_l106_106037


namespace train_length_l106_106266

theorem train_length (speed_kmph : ℕ) (time_s : ℕ) (platform_length_m : ℕ) (h1 : speed_kmph = 72) (h2 : time_s = 26) (h3 : platform_length_m = 260) :
  ∃ train_length_m : ℕ, train_length_m = 260 := by
  sorry

end train_length_l106_106266


namespace cos2α_plus_sin2α_neg_7_over_5_l106_106880

theorem cos2α_plus_sin2α_neg_7_over_5 (α : ℝ) :
  let a := (2, 1) in
  let b := (Real.sin α - Real.cos α, Real.sin α + Real.cos α) in 
  let parallel (u v : ℝ × ℝ) := ∃ k : ℝ, u = (k * v.1, k * v.2) in

  parallel a b → Real.cos (2 * α) + Real.sin (2 * α) = -7 / 5 :=
by
  intros
  sorry

end cos2α_plus_sin2α_neg_7_over_5_l106_106880


namespace both_questions_correct_l106_106961

def total_students := 100
def first_question_correct := 75
def second_question_correct := 30
def neither_question_correct := 20

theorem both_questions_correct :
  (first_question_correct + second_question_correct - (total_students - neither_question_correct)) = 25 :=
by
  sorry

end both_questions_correct_l106_106961


namespace arithmetic_sequence_properties_l106_106025

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}
variable {a1 : ℝ}

theorem arithmetic_sequence_properties 
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a1 > 0) 
  (h3 : a 9 + a 10 = a 11) :
  (∀ m n, m < n → a m > a n) ∧ (∀ n, S n = n * (a1 + (d * (n - 1) / 2))) ∧ S 14 > 0 :=
by 
  sorry

end arithmetic_sequence_properties_l106_106025


namespace arithmetic_sequence_y_solution_l106_106417

theorem arithmetic_sequence_y_solution : 
  ∃ y : ℚ, (y + 2 - - (1 / 3)) = (4 * y - (y + 2)) ∧ y = 13 / 6 :=
by
  sorry

end arithmetic_sequence_y_solution_l106_106417


namespace lice_checks_time_in_hours_l106_106497

-- Define the number of students in each grade
def kindergarteners : ℕ := 26
def first_graders : ℕ := 19
def second_graders : ℕ := 20
def third_graders : ℕ := 25

-- Define the time each check takes (in minutes)
def time_per_check : ℕ := 2

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- The theorem states that the total time in hours is 3
theorem lice_checks_time_in_hours : 
  ((kindergarteners + first_graders + second_graders + third_graders) * time_per_check) / minutes_per_hour = 3 := 
by
  sorry

end lice_checks_time_in_hours_l106_106497


namespace true_statements_l106_106320

-- Define the fixed points F₁ and F₂
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the moving point M on the curve
def M (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1 ∧ x ≠ 3 ∧ x ≠ -3

-- Define the statements to check
def statement_1 : Prop := F₁ = (-5, 0) ∧ F₂ = (5, 0)
def statement_2 (M : ℝ × ℝ) : Prop := 
  let (x, y) := M in x < 0 → y = 0 → x = -3
def statement_3 (M : ℝ × ℝ) : Prop := 
  let (x, y) := M in
  ∠ (F₁.1, F₁.2) (x, y) (F₂.1, F₂.2) = 90 → 
    let m := dist (x, y) (F₁.1, F₁.2) in
    let n := dist (x, y) (F₂.1, F₂.2) in
    (1/2) * m * n = 32 
def statement_4 (A M : ℝ × ℝ) : Prop := 
  let (ax, ay) := A in
  ax = 6 ∧ ay = 1 → 
  min (dist M A + dist M F₂) (|√2)

-- Define the theorem
theorem true_statements (M : ℝ × ℝ) :
  M (M.1, M.2) →
  (statement_1 ∧ statement_2 M) :=
sorry

end true_statements_l106_106320


namespace x_minus_y_eq_2_l106_106187

theorem x_minus_y_eq_2 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : 3 * x + 2 * y = 11) : x - y = 2 :=
sorry

end x_minus_y_eq_2_l106_106187


namespace length_of_XY_correct_l106_106015

noncomputable def length_of_XY (XZ : ℝ) (angleY : ℝ) (angleZ : ℝ) :=
  if angleZ = 90 ∧ angleY = 30 then 8 * Real.sqrt 3 else panic! "Invalid triangle angles"

theorem length_of_XY_correct : length_of_XY 12 30 90 = 8 * Real.sqrt 3 :=
by
  sorry

end length_of_XY_correct_l106_106015


namespace union_of_A_and_B_l106_106163

open Set

def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3, 4} :=
  sorry

end union_of_A_and_B_l106_106163


namespace common_point_eq_l106_106802

theorem common_point_eq (a b c d : ℝ) (h₀ : a ≠ b) 
  (h₁ : ∃ x y : ℝ, y = a * x + a ∧ y = b * x + b ∧ y = c * x + d) : 
  d = c :=
by
  sorry

end common_point_eq_l106_106802


namespace convinced_of_twelve_models_vitya_review_58_offers_l106_106115

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l106_106115


namespace gcd_98_140_245_l106_106645

theorem gcd_98_140_245 : Nat.gcd (Nat.gcd 98 140) 245 = 7 := 
by 
  sorry

end gcd_98_140_245_l106_106645


namespace system_of_equations_correct_l106_106823

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end system_of_equations_correct_l106_106823


namespace abscissa_of_A_is_3_l106_106060

-- Definitions of the points A, B, line l and conditions
def in_first_quadrant (A : ℝ × ℝ) := (A.1 > 0) ∧ (A.2 > 0)

def on_line_l (A : ℝ × ℝ) := A.2 = 2 * A.1

def point_B : ℝ × ℝ := (5, 0)

def diameter_circle (A B : ℝ × ℝ) (P : ℝ × ℝ) :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Vectors AB and CD
def vector_AB (A B : ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

def vector_CD (C D : ℝ × ℝ) := (D.1 - C.1, D.2 - C.2)

def dot_product_zero (A B C D : ℝ × ℝ) := (vector_AB A B).1 * (vector_CD C D).1 + (vector_AB A B).2 * (vector_CD C D).2 = 0

-- Statement to prove
theorem abscissa_of_A_is_3 (A : ℝ × ℝ) (D : ℝ × ℝ) (a : ℝ) :
  in_first_quadrant A →
  on_line_l A →
  diameter_circle A point_B D →
  dot_product_zero A point_B (a, a) D →
  A.1 = 3 :=
by
  sorry

end abscissa_of_A_is_3_l106_106060


namespace question_l106_106971
-- Importing necessary libraries

-- Stating the problem
theorem question (x : ℤ) (h : (x + 12) / 8 = 9) : 35 - (x / 2) = 5 :=
by {
  sorry
}

end question_l106_106971


namespace mul_exponents_l106_106002

theorem mul_exponents (m : ℝ) : 2 * m^3 * 3 * m^4 = 6 * m^7 :=
by sorry

end mul_exponents_l106_106002


namespace circle_center_coordinates_l106_106374

theorem circle_center_coordinates :
  ∀ x y, (x^2 + y^2 - 4 * x - 2 * y - 5 = 0) → (x, y) = (2, 1) :=
by
  sorry

end circle_center_coordinates_l106_106374


namespace time_for_A_alone_l106_106960

variable {W : ℝ}
variable {x : ℝ}

theorem time_for_A_alone (h1 : (W / x) + (W / 24) = W / 12) : x = 24 := 
sorry

end time_for_A_alone_l106_106960


namespace evaluate_expression_l106_106546

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l106_106546


namespace percentage_gain_second_week_l106_106624

variables (initial_investment final_value after_first_week_value gain_percentage first_week_gain second_week_gain second_week_gain_percentage : ℝ)

def pima_investment (initial_investment: ℝ) (first_week_gain_percentage: ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage)

def second_week_investment (initial_investment first_week_gain_percentage second_week_gain_percentage : ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage) * (1 + second_week_gain_percentage)

theorem percentage_gain_second_week
  (initial_investment : ℝ)
  (first_week_gain_percentage : ℝ)
  (final_value : ℝ)
  (h1: initial_investment = 400)
  (h2: first_week_gain_percentage = 0.25)
  (h3: final_value = 750) :
  second_week_gain_percentage = 0.5 :=
by
  let after_first_week_value := pima_investment initial_investment first_week_gain_percentage
  let second_week_gain := final_value - after_first_week_value
  let second_week_gain_percentage := second_week_gain / after_first_week_value * 100
  sorry

end percentage_gain_second_week_l106_106624


namespace solve_inequality_l106_106776

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l106_106776


namespace probability_two_dice_sum_greater_than_9_eq_1_over_6_l106_106967

-- Define the sample space for a die roll
def die := {1, 2, 3, 4, 5, 6}

-- Define the event of rolling two dice
def event_space := die × die

-- Define the favorable event where the sum is greater than 9
def favorable_event (x : ℕ × ℕ) : Prop := (x.1 + x.2) > 9

-- Find the number of elements in a set that satisfy a predicate
def count {α : Type*} (s : Finset α) (p : α → Prop) [DecidablePred p] : ℕ :=
  (s.filter p).card

-- Define the probability that the sum on the top faces of both dice is greater than 9
def probability_sum_greater_than_9 : ℚ :=
  (count event_space favorable_event) / event_space.card

-- State the theorem
theorem probability_two_dice_sum_greater_than_9_eq_1_over_6 :
  probability_sum_greater_than_9 = 1 / 6 := by
  sorry

end probability_two_dice_sum_greater_than_9_eq_1_over_6_l106_106967


namespace school_problem_proof_l106_106352

noncomputable def solve_school_problem (B G x y z : ℕ) :=
  B + G = 300 ∧
  B * y = x * G ∧
  G = (x * 300) / 100 →
  z = 300 - 3 * x - (300 * x) / (x + y)

theorem school_problem_proof (B G x y z : ℕ) :
  solve_school_problem B G x y z :=
by
  sorry

end school_problem_proof_l106_106352


namespace xyz_cubed_over_xyz_eq_21_l106_106912

open Complex

theorem xyz_cubed_over_xyz_eq_21 {x y z : ℂ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 18)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 21 :=
sorry

end xyz_cubed_over_xyz_eq_21_l106_106912


namespace sum_of_cubes_pattern_l106_106620

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 = 3^2) ->
  (1^3 + 2^3 + 3^3 = 6^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 = 10^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  intros h1 h2 h3
  -- Proof follows here
  sorry

end sum_of_cubes_pattern_l106_106620


namespace mohamed_donated_more_l106_106905

-- Definitions of the conditions
def toysLeilaDonated : ℕ := 2 * 25
def toysMohamedDonated : ℕ := 3 * 19

-- The theorem stating Mohamed donated 7 more toys than Leila
theorem mohamed_donated_more : toysMohamedDonated - toysLeilaDonated = 7 :=
by
  sorry

end mohamed_donated_more_l106_106905


namespace bus_speed_l106_106184

theorem bus_speed (d t : ℕ) (h1 : d = 201) (h2 : t = 3) : d / t = 67 :=
by sorry

end bus_speed_l106_106184


namespace curve_is_line_l106_106159

def curve := {p : ℝ × ℝ | ∃ (θ : ℝ), (p.1 = (1 / (Real.sin θ + Real.cos θ)) * Real.cos θ
                                        ∧ p.2 = (1 / (Real.sin θ + Real.cos θ)) * Real.sin θ)}

-- Problem: Prove that the curve defined by the polar equation is a line.
theorem curve_is_line : ∀ (p : ℝ × ℝ), p ∈ curve → p.1 + p.2 = 1 :=
by
  -- The proof is omitted.
  sorry

end curve_is_line_l106_106159


namespace converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l106_106117

variable (a b : ℝ)

theorem converse_of_proposition :
  (ab > 0 → a > 0 ∧ b > 0) = false := sorry

theorem inverse_of_proposition :
  (a ≤ 0 ∨ b ≤ 0 → ab ≤ 0) = false := sorry

theorem contrapositive_of_proposition :
  (ab ≤ 0 → a ≤ 0 ∨ b ≤ 0) = true := sorry

end converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l106_106117


namespace combined_distance_l106_106641

theorem combined_distance (second_lady_distance : ℕ) (first_lady_distance : ℕ) 
  (h1 : second_lady_distance = 4) 
  (h2 : first_lady_distance = 2 * second_lady_distance) : 
  first_lady_distance + second_lady_distance = 12 :=
by 
  sorry

end combined_distance_l106_106641


namespace mean_proportional_49_64_l106_106654

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end mean_proportional_49_64_l106_106654


namespace pqrs_predicate_l106_106164

noncomputable def P (a b c : ℝ) := a + b - c
noncomputable def Q (a b c : ℝ) := b + c - a
noncomputable def R (a b c : ℝ) := c + a - b

theorem pqrs_predicate (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c) * (Q a b c) * (R a b c) > 0 ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end pqrs_predicate_l106_106164


namespace average_age_of_girls_l106_106459

theorem average_age_of_girls (total_students : ℕ) (avg_age_boys : ℕ) (num_girls : ℕ) (avg_age_school : ℚ) 
  (h1 : total_students = 604) 
  (h2 : avg_age_boys = 12) 
  (h3 : num_girls = 151) 
  (h4 : avg_age_school = 11.75) : 
  (total_age_of_girls / num_girls) = 11 :=
by
  -- Definitions
  let num_boys := total_students - num_girls
  let total_age := avg_age_school * total_students
  let total_age_boys := avg_age_boys * num_boys
  let total_age_girls := total_age - total_age_boys
  -- Proof goal
  have : total_age_of_girls = total_age_girls := sorry
  have : total_age_of_girls / num_girls = 11 := sorry
  sorry

end average_age_of_girls_l106_106459


namespace worst_player_is_son_or_sister_l106_106973

axiom Family : Type
axiom Woman : Family
axiom Brother : Family
axiom Son : Family
axiom Daughter : Family
axiom Sister : Family

axiom are_chess_players : ∀ f : Family, Prop
axiom is_twin : Family → Family → Prop
axiom is_best_player : Family → Prop
axiom is_worst_player : Family → Prop
axiom same_age : Family → Family → Prop
axiom opposite_sex : Family → Family → Prop
axiom is_sibling : Family → Family → Prop

-- Conditions
axiom all_are_chess_players : ∀ f, are_chess_players f
axiom worst_best_opposite_sex : ∀ w b, is_worst_player w → is_best_player b → opposite_sex w b
axiom worst_best_same_age : ∀ w b, is_worst_player w → is_best_player b → same_age w b
axiom twins_relationship : ∀ t1 t2, is_twin t1 t2 → (is_sibling t1 t2 ∨ (t1 = Woman ∧ t2 = Sister))

-- Goal
theorem worst_player_is_son_or_sister :
  ∃ w, (is_worst_player w ∧ (w = Son ∨ w = Sister)) :=
sorry

end worst_player_is_son_or_sister_l106_106973


namespace incorrect_rounding_statement_l106_106116

def rounded_to_nearest (n : ℝ) (accuracy : ℝ) : Prop :=
  ∃ (k : ℤ), abs (n - k * accuracy) < accuracy / 2

theorem incorrect_rounding_statement :
  ¬ rounded_to_nearest 23.9 10 :=
sorry

end incorrect_rounding_statement_l106_106116


namespace marco_might_need_at_least_n_tables_n_tables_are_sufficient_l106_106419
open Function

variables (n : ℕ) (friends_sticker_sets : Fin n → Finset (Fin n))

-- Each friend is missing exactly one unique sticker
def each_friend_missing_one_unique_sticker :=
  ∀ i : Fin n, ∃ j : Fin n, friends_sticker_sets i = (Finset.univ \ {j})

-- A pair of friends is wholesome if their combined collection has all stickers
def is_wholesome_pair (i j : Fin n) :=
  ∀ s : Fin n, s ∈ friends_sticker_sets i ∨ s ∈ friends_sticker_sets j

-- Main problem statements
-- Problem 1: Marco might need to reserve at least n different tables
theorem marco_might_need_at_least_n_tables 
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) : 
  ∃ i j : Fin n, i ≠ j ∧ is_wholesome_pair n friends_sticker_sets i j :=
sorry

-- Problem 2: n tables will always be enough for Marco to achieve his goal
theorem n_tables_are_sufficient
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) :
  ∃ arrangement : Fin n → Fin n, ∀ i j, i ≠ j → arrangement i ≠ arrangement j :=
sorry

end marco_might_need_at_least_n_tables_n_tables_are_sufficient_l106_106419


namespace solve_inequality_l106_106790

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l106_106790


namespace total_population_of_Springfield_and_Greenville_l106_106004

theorem total_population_of_Springfield_and_Greenville :
  let Springfield := 482653
  let diff := 119666
  let Greenville := Springfield - diff
  Springfield + Greenville = 845640 := by
  sorry

end total_population_of_Springfield_and_Greenville_l106_106004


namespace tetrahedron_inequality_l106_106275

theorem tetrahedron_inequality (t1 t2 t3 t4 τ1 τ2 τ3 τ4 : ℝ) 
  (ht1 : t1 > 0) (ht2 : t2 > 0) (ht3 : t3 > 0) (ht4 : t4 > 0)
  (hτ1 : τ1 > 0) (hτ2 : τ2 > 0) (hτ3 : τ3 > 0) (hτ4 : τ4 > 0)
  (sphere_inscribed : ∀ {x y : ℝ}, x > 0 → y > 0 → x^2 / y^2 ≤ (x - 2 * y) ^ 2 / x ^ 2) :
  (τ1 / t1 + τ2 / t2 + τ3 / t3 + τ4 / t4) ≥ 1 
  ∧ (τ1 / t1 + τ2 / t2 + τ3 / t3 + τ4 / t4 = 1 ↔ t1 = t2 ∧ t2 = t3 ∧ t3 = t4) := by
  sorry

end tetrahedron_inequality_l106_106275


namespace find_a_squared_plus_b_squared_l106_106367

theorem find_a_squared_plus_b_squared 
  (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 104) : 
  a^2 + b^2 = 1392 := 
by 
  sorry

end find_a_squared_plus_b_squared_l106_106367


namespace solve_inequality_system_l106_106635

theorem solve_inequality_system (x : ℝ) :
  (x + 1 < 4 ∧ 1 - 3 * x ≥ -5) ↔ (x ≤ 2) :=
by
  sorry

end solve_inequality_system_l106_106635


namespace circumference_of_minor_arc_l106_106068

-- Given:
-- 1. Three points (D, E, F) are on a circle with radius 25
-- 2. The angle ∠EFD = 120°

-- We need to prove that the length of the minor arc DE is 50π / 3
theorem circumference_of_minor_arc 
  (D E F : Point) 
  (r : ℝ) (h : r = 25) 
  (angleEFD : ℝ) 
  (hAngle : angleEFD = 120) 
  (circumference : ℝ) 
  (hCircumference : circumference = 2 * Real.pi * r) :
  arc_length_DE = 50 * Real.pi / 3 :=
by
  sorry

end circumference_of_minor_arc_l106_106068


namespace exercise_l106_106301

noncomputable def f : ℝ → ℝ := sorry

theorem exercise
  (h_even : ∀ x : ℝ, f (x + 1) = f (-(x + 1)))
  (h_increasing : ∀ ⦃a b : ℝ⦄, 1 ≤ a → a ≤ b → f a ≤ f b)
  (x1 x2 : ℝ)
  (h_x1_neg : x1 < 0)
  (h_x2_pos : x2 > 0)
  (h_sum_neg : x1 + x2 < -2) :
  f (-x1) > f (-x2) :=
sorry

end exercise_l106_106301


namespace vitya_needs_58_offers_l106_106112

noncomputable def smallest_integer_k (P : ℝ → ℝ) : ℝ :=
  if H : ∃ k, k > P (100), then classical.some H else 0

theorem vitya_needs_58_offers :
  ∀ n : ℕ, n ≥ 13 → 
  (12:ℝ/13:ℝ) ^ smallest_integer_k (fun x => Real.log x / (Real.log 13 - Real.log 12)) < 0.01 :=
begin
  assume n h,
  rw smallest_integer_k,
  split_ifs,
  { sorry }, -- proof would go here
  { exfalso, exact sorry }, -- no proof steps provided
end

end vitya_needs_58_offers_l106_106112


namespace compute_expression_l106_106906

-- The definition and conditions
def is_nonreal_root_of_unity (ω : ℂ) : Prop := ω ^ 3 = 1 ∧ ω ≠ 1

-- The statement
theorem compute_expression (ω : ℂ) (hω : is_nonreal_root_of_unity ω) : 
  (1 - 2 * ω + 2 * ω ^ 2) ^ 6 + (1 + 2 * ω - 2 * ω ^ 2) ^ 6 = 0 :=
sorry

end compute_expression_l106_106906


namespace min_value_expression_l106_106724

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  ∃ c, c = (1/(a+1) + 4/(b+1)) ∧ c ≥ 9/4 :=
by
  sorry

end min_value_expression_l106_106724


namespace range_of_a_l106_106214

def p (x : ℝ) : Prop := x ≤ 1/2 ∨ x ≥ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

def not_q (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, not_q x a → p x) ∧ (∃ x : ℝ, ¬ (p x → not_q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l106_106214


namespace find_jordana_and_james_age_l106_106601

variable (current_age_of_Jennifer : ℕ) (current_age_of_Jordana : ℕ) (current_age_of_James : ℕ)

-- Conditions
axiom jennifer_40_in_twenty_years : current_age_of_Jennifer + 20 = 40
axiom jordana_twice_jennifer_in_twenty_years : current_age_of_Jordana + 20 = 2 * (current_age_of_Jennifer + 20)
axiom james_ten_years_younger_in_twenty_years : current_age_of_James + 20 = 
  (current_age_of_Jennifer + 20) + (current_age_of_Jordana + 20) - 10

-- Prove that Jordana is currently 60 years old and James is currently 90 years old
theorem find_jordana_and_james_age : current_age_of_Jordana = 60 ∧ current_age_of_James = 90 :=
  sorry

end find_jordana_and_james_age_l106_106601


namespace solve_inequality_l106_106778

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3*x - 2) / (x - 2)) > 3) ↔ (x ∈ set.Ioo (4/3 : ℝ) 2 ∪ set.Ioi 2) :=
by  -- Proof to be provided
  sorry

end solve_inequality_l106_106778


namespace wire_cut_equal_area_l106_106276

theorem wire_cut_equal_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a / b = 2 / Real.sqrt Real.pi) ↔ (a^2 / 16 = b^2 / (4 * Real.pi)) :=
by
  sorry

end wire_cut_equal_area_l106_106276


namespace find_initial_number_l106_106426

theorem find_initial_number (x : ℕ) (h : ∃ y : ℕ, x * y = 4 ∧ y = 2) : x = 2 :=
by
  sorry

end find_initial_number_l106_106426


namespace money_lent_to_C_is_3000_l106_106405

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def time_C : ℕ := 4
def rate_of_interest : ℕ := 12
def total_interest : ℕ := 2640
def interest_rate : ℚ := (rate_of_interest : ℚ) / 100
def interest_B : ℚ := principal_B * interest_rate * time_B
def interest_C (P_C : ℚ) : ℚ := P_C * interest_rate * time_C

theorem money_lent_to_C_is_3000 :
  ∃ P_C : ℚ, interest_B + interest_C P_C = total_interest ∧ P_C = 3000 :=
by
  use 3000
  unfold interest_B interest_C interest_rate principal_B time_B time_C rate_of_interest total_interest
  sorry

end money_lent_to_C_is_3000_l106_106405


namespace solve_inequality_l106_106784

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l106_106784


namespace quotient_when_divided_by_44_l106_106662

theorem quotient_when_divided_by_44 (N Q P : ℕ) (h1 : N = 44 * Q) (h2 : N = 35 * P + 3) : Q = 12 :=
by {
  -- Proof
  sorry
}

end quotient_when_divided_by_44_l106_106662


namespace date_behind_D_correct_l106_106353

noncomputable def date_behind_B : ℕ := sorry
noncomputable def date_behind_E : ℕ := date_behind_B + 2
noncomputable def date_behind_F : ℕ := date_behind_B + 15
noncomputable def date_behind_D : ℕ := sorry

theorem date_behind_D_correct :
  date_behind_B + date_behind_D = date_behind_E + date_behind_F := sorry

end date_behind_D_correct_l106_106353


namespace total_cost_after_discounts_and_cashback_l106_106701

def iPhone_original_price : ℝ := 800
def iWatch_original_price : ℝ := 300
def iPhone_discount_rate : ℝ := 0.15
def iWatch_discount_rate : ℝ := 0.10
def cashback_rate : ℝ := 0.02

theorem total_cost_after_discounts_and_cashback :
  (iPhone_original_price * (1 - iPhone_discount_rate) + iWatch_original_price * (1 - iWatch_discount_rate)) * (1 - cashback_rate) = 931 :=
by sorry

end total_cost_after_discounts_and_cashback_l106_106701


namespace charlotte_avg_speed_l106_106413

def distance : ℕ := 60  -- distance in miles
def time : ℕ := 6       -- time in hours

theorem charlotte_avg_speed : (distance / time) = 10 := by
  sorry

end charlotte_avg_speed_l106_106413


namespace pet_store_initial_puppies_l106_106831

theorem pet_store_initial_puppies
  (sold: ℕ) (cages: ℕ) (puppies_per_cage: ℕ)
  (remaining_puppies: ℕ)
  (h1: sold = 30)
  (h2: cages = 6)
  (h3: puppies_per_cage = 8)
  (h4: remaining_puppies = cages * puppies_per_cage):
  (sold + remaining_puppies) = 78 :=
by
  sorry

end pet_store_initial_puppies_l106_106831


namespace ned_pieces_left_l106_106477

def boxes_bought : ℝ := 14.0
def boxes_given : ℝ := 7.0
def pieces_per_box : ℝ := 6.0
def boxes_left (bought : ℝ) (given : ℝ) : ℝ := bought - given
def total_pieces (boxes : ℝ) (pieces_per_box : ℝ) : ℝ := boxes * pieces_per_box

theorem ned_pieces_left : total_pieces (boxes_left boxes_bought boxes_given) pieces_per_box = 42.0 := by
  sorry

end ned_pieces_left_l106_106477


namespace dealer_gross_profit_l106_106812

theorem dealer_gross_profit (P S G : ℝ) (hP : P = 150) (markup : S = P + 0.5 * S) :
  G = S - P → G = 150 :=
by
  sorry

end dealer_gross_profit_l106_106812


namespace BoxC_in_BoxA_l106_106146

-- Define the relationship between the boxes
def BoxA_has_BoxB (A B : ℕ) : Prop := A = 4 * B
def BoxB_has_BoxC (B C : ℕ) : Prop := B = 6 * C

-- Define the proof problem
theorem BoxC_in_BoxA {A B C : ℕ} (h1 : BoxA_has_BoxB A B) (h2 : BoxB_has_BoxC B C) : A = 24 * C :=
by
  sorry

end BoxC_in_BoxA_l106_106146


namespace total_population_of_Springfield_and_Greenville_l106_106003

theorem total_population_of_Springfield_and_Greenville :
  let Springfield := 482653
  let diff := 119666
  let Greenville := Springfield - diff
  Springfield + Greenville = 845640 := by
  sorry

end total_population_of_Springfield_and_Greenville_l106_106003


namespace total_cost_is_correct_l106_106700

-- Definitions of the conditions given
def price_iphone12 : ℝ := 800
def price_iwatch : ℝ := 300
def discount_iphone12 : ℝ := 0.15
def discount_iwatch : ℝ := 0.1
def cashback_discount : ℝ := 0.02

-- The final total cost after applying all discounts and cashback
def total_cost_after_discounts_and_cashback : ℝ :=
  let discount_amount_iphone12 := price_iphone12 * discount_iphone12
  let new_price_iphone12 := price_iphone12 - discount_amount_iphone12
  let discount_amount_iwatch := price_iwatch * discount_iwatch
  let new_price_iwatch := price_iwatch - discount_amount_iwatch
  let initial_total_cost := new_price_iphone12 + new_price_iwatch
  let cashback_amount := initial_total_cost * cashback_discount
  initial_total_cost - cashback_amount

-- Statement to be proved
theorem total_cost_is_correct :
  total_cost_after_discounts_and_cashback = 931 := by
  sorry

end total_cost_is_correct_l106_106700


namespace directrix_of_parabola_l106_106486

-- Define the given condition: the equation of the parabola
def given_parabola (x : ℝ) : ℝ := 4 * x ^ 2

-- State the theorem to be proven
theorem directrix_of_parabola : 
  (∀ x : ℝ, given_parabola x = 4 * x ^ 2) → 
  (y = -1 / 16) :=
sorry

end directrix_of_parabola_l106_106486


namespace positive_integer_solutions_eq_17_l106_106396

theorem positive_integer_solutions_eq_17 :
  {x : ℕ // x > 0} × {y : ℕ // y > 0} → 5 * x + 10 * y = 100 ->
  ∃ (n : ℕ), n = 17 := sorry

end positive_integer_solutions_eq_17_l106_106396


namespace eval_expression_l106_106539

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l106_106539


namespace fixed_point_and_max_distance_eqn_l106_106879

-- Define line l1
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y - 8 = 0

-- Define line l2 parallel to l1 passing through origin
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y = 0

-- Define line y = x
def line_y_eq_x (x y : ℝ) : Prop :=
  y = x

-- Define line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop :=
  x + y = 0

theorem fixed_point_and_max_distance_eqn :
  (∀ m : ℝ, l1 m 2 2) ∧ (∀ m : ℝ, (l2 m 2 2 → false)) →
  (∃ x y : ℝ, l2 m x y ∧ line_x_plus_y_eq_0 x y) :=
by sorry

end fixed_point_and_max_distance_eqn_l106_106879


namespace count_mod_6_mod_11_lt_1000_l106_106038

theorem count_mod_6_mod_11_lt_1000 : ∃ n : ℕ, (∀ x : ℕ, (x < n + 1) ∧ ((6 + 11 * x) < 1000) ∧ (6 + 11 * x) % 11 = 6) ∧ (n + 1 = 91) :=
by
  sorry

end count_mod_6_mod_11_lt_1000_l106_106038


namespace derivative_of_odd_function_is_even_l106_106765

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the main theorem
theorem derivative_of_odd_function_is_even (f g : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : ∀ x, g x = deriv f x) :
  ∀ x, g (-x) = g x :=
by
  sorry

end derivative_of_odd_function_is_even_l106_106765


namespace raman_salary_loss_l106_106221

theorem raman_salary_loss : 
  ∀ (S : ℝ), S > 0 →
  let decreased_salary := S - (0.5 * S) 
  let final_salary := decreased_salary + (0.5 * decreased_salary) 
  let loss := S - final_salary 
  let percentage_loss := (loss / S) * 100
  percentage_loss = 25 := 
by
  intros S hS
  let decreased_salary := S - (0.5 * S)
  let final_salary := decreased_salary + (0.5 * decreased_salary)
  let loss := S - final_salary
  let percentage_loss := (loss / S) * 100
  have h1 : decreased_salary = 0.5 * S := by sorry
  have h2 : final_salary = 0.75 * S := by sorry
  have h3 : loss = 0.25 * S := by sorry
  have h4 : percentage_loss = 25 := by sorry
  exact h4

end raman_salary_loss_l106_106221


namespace quadratic_root_value_m_l106_106728

theorem quadratic_root_value_m (m : ℝ) : ∃ x, x = 1 ∧ x^2 + x - m = 0 → m = 2 := by
  sorry

end quadratic_root_value_m_l106_106728


namespace john_got_rolls_l106_106204

def cost_per_dozen : ℕ := 5
def money_spent : ℕ := 15
def rolls_per_dozen : ℕ := 12

theorem john_got_rolls : (money_spent / cost_per_dozen) * rolls_per_dozen = 36 :=
by sorry

end john_got_rolls_l106_106204


namespace sum_S19_is_190_l106_106917

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
∀ n m, a n + a m = a (n+1) + a (m-1)

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

-- Main theorem
theorem sum_S19_is_190 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_def : sum_of_first_n_terms a S)
  (h_condition : a 6 + a 14 = 20) :
  S 19 = 190 :=
sorry

end sum_S19_is_190_l106_106917


namespace value_of_a_l106_106296

theorem value_of_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -3) (h3 : a * x - y = 1) : a = -2 :=
by
  -- Placeholder for the proof
  sorry

end value_of_a_l106_106296


namespace highest_value_of_a_l106_106158

theorem highest_value_of_a (a : ℕ) (h : 0 ≤ a ∧ a ≤ 9) : (365 * 10 ^ 3 + a * 10 ^ 2 + 16) % 8 = 0 → a = 8 := by
  sorry

end highest_value_of_a_l106_106158


namespace original_price_of_cycle_l106_106135

variable (P : ℝ)

theorem original_price_of_cycle (h1 : 0.75 * P = 1050) : P = 1400 :=
sorry

end original_price_of_cycle_l106_106135


namespace rows_seating_l106_106420

theorem rows_seating (x y : ℕ) (h : 7 * x + 6 * y = 52) : x = 4 :=
by
  sorry

end rows_seating_l106_106420


namespace abs_inequality_l106_106783

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l106_106783


namespace p2_div_q2_eq_4_l106_106695

theorem p2_div_q2_eq_4 
  (p q : ℝ → ℝ)
  (h1 : ∀ x, p x = 12 * x)
  (h2 : ∀ x, q x = (x + 4) * (x - 1))
  (h3 : p 0 = 0)
  (h4 : p (-1) / q (-1) = -2) :
  (p 2 / q 2 = 4) :=
by {
  sorry
}

end p2_div_q2_eq_4_l106_106695


namespace volume_of_rect_box_l106_106256

open Real

/-- Proof of the volume of a rectangular box given its face areas -/
theorem volume_of_rect_box (l w h : ℝ) 
  (A1 : l * w = 40) 
  (A2 : w * h = 10) 
  (A3 : l * h = 8) : 
  l * w * h = 40 * sqrt 2 :=
by
  sorry

end volume_of_rect_box_l106_106256


namespace combined_afternoon_burning_rate_l106_106637

theorem combined_afternoon_burning_rate 
  (morning_period_hours : ℕ)
  (afternoon_period_hours : ℕ)
  (rate_A_morning : ℕ)
  (rate_B_morning : ℕ)
  (total_morning_burn : ℕ)
  (initial_wood : ℕ)
  (remaining_wood : ℕ) :
  morning_period_hours = 4 →
  afternoon_period_hours = 4 →
  rate_A_morning = 2 →
  rate_B_morning = 1 →
  total_morning_burn = 12 →
  initial_wood = 50 →
  remaining_wood = 6 →
  ((initial_wood - remaining_wood - total_morning_burn) / afternoon_period_hours) = 8 := 
by
  intros
  -- We would continue with a proof here
  sorry

end combined_afternoon_burning_rate_l106_106637


namespace odd_lattice_points_on_BC_l106_106748

theorem odd_lattice_points_on_BC
  (A B C : ℤ × ℤ)
  (odd_lattice_points_AB : Odd ((B.1 - A.1) * (B.2 - A.2)))
  (odd_lattice_points_AC : Odd ((C.1 - A.1) * (C.2 - A.2))) :
  Odd ((C.1 - B.1) * (C.2 - B.2)) :=
sorry

end odd_lattice_points_on_BC_l106_106748


namespace domain_of_tan_l106_106711

theorem domain_of_tan :
    ∀ k : ℤ, ∀ x : ℝ,
    (x > (k * π / 2 - π / 8) ∧ x < (k * π / 2 + 3 * π / 8)) ↔
    2 * x - π / 4 ≠ k * π + π / 2 :=
by
  intro k x
  sorry

end domain_of_tan_l106_106711


namespace ratio_triangle_areas_l106_106283

-- Define the parallelogram and conditions
variables {A B C D E F : Type} [plane A] [plane B] [plane C] [plane D] [plane E] [plane F]
variables {AB CD : Segment A} [parallelogram AB CD]
variables {AE ED BF FC : Segment E} [ratio_AE_ED : ratio AE ED = 9 / 5] [ratio_BF_FC : ratio BF FC = 7 / 4]

theorem ratio_triangle_areas (h1 : parallelogram AB CD) (h2 : ratio_AE_ED) (h3 : ratio_BF_FC) :
  area (triangle A C E) / area (triangle B D F) = 99 / 98 :=
sorry

end ratio_triangle_areas_l106_106283


namespace matrices_are_inverses_l106_106439

-- Define variables
variables {a b c d s : ℚ}

-- Define matrices
def matrix1 : Matrix (Fin 2) (Fin 2) ℚ := ![![a, -1], ![3, b]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℚ := ![![4, c], ![d, -2]]

-- Define the identity matrix
def identity_matrix : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

-- Prove that matrix1 and matrix2 are inverses
theorem matrices_are_inverses (h1 : matrix1 * matrix2 = identity_matrix) :
  a + b + c + d = s :=
by
  sorry

end matrices_are_inverses_l106_106439


namespace value_of_number_l106_106456

theorem value_of_number (number y : ℝ) 
  (h1 : (number + 5) * (y - 5) = 0) 
  (h2 : ∀ n m : ℝ, (n + 5) * (m - 5) = 0 → n^2 + m^2 ≥ 25) 
  (h3 : number^2 + y^2 = 25) : number = -5 :=
sorry

end value_of_number_l106_106456


namespace real_solutions_quadratic_l106_106010

theorem real_solutions_quadratic (d : ℝ) (h : 0 < d) :
  ∃ x : ℝ, x^2 - 8 * x + d < 0 ↔ 0 < d ∧ d < 16 :=
by
  sorry

end real_solutions_quadratic_l106_106010


namespace ratio_of_a_to_c_l106_106634

variable {a b c d : ℚ}

theorem ratio_of_a_to_c (h₁ : a / b = 5 / 4) (h₂ : c / d = 4 / 3) (h₃ : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
sorry

end ratio_of_a_to_c_l106_106634


namespace minimum_f_value_l106_106016

noncomputable def f (x : ℝ) : ℝ :=
   Real.sqrt (2 * x ^ 2 - 4 * x + 4) + 
   Real.sqrt (2 * x ^ 2 - 16 * x + (Real.log x / Real.log 2) ^ 2 - 2 * x * (Real.log x / Real.log 2) + 
              2 * (Real.log x / Real.log 2) + 50)

theorem minimum_f_value : ∀ x : ℝ, x > 0 → f x ≥ 7 ∧ f 2 = 7 :=
by
  sorry

end minimum_f_value_l106_106016


namespace arcsin_sqrt_three_over_two_l106_106689

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l106_106689


namespace emily_spending_l106_106155

theorem emily_spending : ∀ {x : ℝ}, (x + 2 * x + 3 * x = 120) → (x = 20) :=
by
  intros x h
  sorry

end emily_spending_l106_106155


namespace triangle_base_length_l106_106798

theorem triangle_base_length :
  ∀ (base height area : ℕ), height = 4 → area = 16 → area = (base * height) / 2 → base = 8 :=
by
  intros base height area h_height h_area h_formula
  sorry

end triangle_base_length_l106_106798


namespace combined_distance_l106_106640

theorem combined_distance (second_lady_distance : ℕ) (first_lady_distance : ℕ) 
  (h1 : second_lady_distance = 4) 
  (h2 : first_lady_distance = 2 * second_lady_distance) : 
  first_lady_distance + second_lady_distance = 12 :=
by 
  sorry

end combined_distance_l106_106640


namespace smallest_positive_e_l106_106098

-- Define the polynomial and roots condition
def polynomial (a b c d e : ℤ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

def has_integer_roots (p : ℝ → ℝ) (roots : List ℝ) : Prop :=
  ∀ r ∈ roots, p r = 0

def polynomial_with_given_roots (a b c d e : ℤ) : Prop :=
  has_integer_roots (polynomial a b c d e) [-3, 4, 11, -(1/4)]

-- Main theorem to prove the smallest positive integer e
theorem smallest_positive_e (a b c d : ℤ) :
  ∃ e : ℤ, e > 0 ∧ polynomial_with_given_roots a b c d e ∧
            (∀ e' : ℤ, e' > 0 ∧ polynomial_with_given_roots a b c d e' → e ≤ e') :=
  sorry

end smallest_positive_e_l106_106098


namespace smallest_positive_n_l106_106560

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l106_106560


namespace first_digit_of_sum_l106_106080

theorem first_digit_of_sum (n : ℕ) (a : ℕ) (hs : 9 * a = n)
  (h_sum : n = 43040102 - (10^7 * d - 10^7 * 4)) : 
  (10^7 * d - 10^7 * 4) / 10^7 = 8 :=
by
  sorry

end first_digit_of_sum_l106_106080


namespace probability_no_adjacent_equal_l106_106019

open Finset
open BigOperators

def no_adjacent_equal_prob : ℚ :=
  -- Number of valid arrangements
  let num_valid_arrangements := 8 * 7^4 - 8 * 6 * 7^3 in
  -- Total possible arrangements
  let total_arrangements := 8^5 in
  -- The probability
  (637 : ℚ) / (2048 : ℚ)

theorem probability_no_adjacent_equal :
  no_adjacent_equal_prob = (637 : ℚ) / (2048 : ℚ) :=
  by
    sorry

end probability_no_adjacent_equal_l106_106019


namespace absolute_sum_value_l106_106797

theorem absolute_sum_value (x1 x2 x3 x4 x5 : ℝ) 
(h : x1 + 1 = x2 + 2 ∧ x2 + 2 = x3 + 3 ∧ x3 + 3 = x4 + 4 ∧ x4 + 4 = x5 + 5 ∧ x5 + 5 = x1 + x2 + x3 + x4 + x5 + 6) :
  |(x1 + x2 + x3 + x4 + x5)| = 3.75 := 
by
  sorry

end absolute_sum_value_l106_106797


namespace at_least_one_success_l106_106141

-- Define probabilities for A, B, and C
def pA : ℚ := 1 / 2
def pB : ℚ := 2 / 3
def pC : ℚ := 4 / 5

-- Define the probability that none succeed
def pNone : ℚ := (1 - pA) * (1 - pB) * (1 - pC)

-- Define the probability that at least one of them succeeds
def pAtLeastOne : ℚ := 1 - pNone

theorem at_least_one_success : pAtLeastOne = 29 / 30 := 
by sorry

end at_least_one_success_l106_106141


namespace sprinkler_days_needed_l106_106131

-- Definitions based on the conditions
def morning_water : ℕ := 4
def evening_water : ℕ := 6
def daily_water : ℕ := morning_water + evening_water
def total_water_needed : ℕ := 50

-- The proof statement
theorem sprinkler_days_needed : total_water_needed / daily_water = 5 := by
  sorry

end sprinkler_days_needed_l106_106131


namespace THH_before_HHH_l106_106957

-- Fair coin flip probabilities
def fair_coin := (1/2 : ℚ)

-- Definitions for sequences
def sequence_THH := [tt, ff, ff]
def sequence_HHH := [ff, ff, ff]

-- Event that checks for THH before HHH
noncomputable def Prob_THH_before_HHH : ℚ :=
  1 - (fair_coin ^ 3)

theorem THH_before_HHH : Prob_THH_before_HHH = 7 / 8 := by
  sorry

end THH_before_HHH_l106_106957


namespace hyperbola_eq_l106_106731

theorem hyperbola_eq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (hyp_eq : ∀ x y, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1)
  (asymptote : b / a = Real.sqrt 3)
  (focus_parabola : c = 4) : 
  a^2 = 4 ∧ b^2 = 12 := by
sorry

end hyperbola_eq_l106_106731


namespace necessary_and_sufficient_condition_l106_106021

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (|a + b| / (|a| + |b|) ≤ 1) ↔ (a^2 + b^2 ≠ 0) :=
sorry

end necessary_and_sufficient_condition_l106_106021


namespace TripleApplicationOfF_l106_106149

def f (N : ℝ) : ℝ := 0.7 * N + 2

theorem TripleApplicationOfF :
  f (f (f 40)) = 18.1 :=
  sorry

end TripleApplicationOfF_l106_106149


namespace polynomial_sum_l106_106632

noncomputable def p : ℝ → ℝ :=
  λ x, 2

noncomputable def q : ℝ → ℝ :=
  λ x, -3 * x^2 + 18 * x - 24

theorem polynomial_sum :
  p 1 = 2 ∧ q 3 = 3 ∧ (∀ x, (x = 2 ∨ x = 4) → q x = 0) →
  ∀ x, p x + q x = -3 * x^2 + 18 * x - 22 :=
begin
  sorry
end

end polynomial_sum_l106_106632


namespace counting_numbers_remainder_7_l106_106328

theorem counting_numbers_remainder_7 :
  {n : ℕ | 7 < n ∧ ∃ (k : ℕ), 52 = k * n}.to_finset.card = 3 :=
sorry

end counting_numbers_remainder_7_l106_106328


namespace Enid_made_8_sweaters_l106_106857

def scarves : ℕ := 10
def sweaters_Aaron : ℕ := 5
def wool_per_scarf : ℕ := 3
def wool_per_sweater : ℕ := 4
def total_wool_used : ℕ := 82
def Enid_sweaters : ℕ := 8

theorem Enid_made_8_sweaters
  (scarves : ℕ)
  (sweaters_Aaron : ℕ)
  (wool_per_scarf : ℕ)
  (wool_per_sweater : ℕ)
  (total_wool_used : ℕ)
  (Enid_sweaters : ℕ)
  : Enid_sweaters = 8 :=
by
  sorry

end Enid_made_8_sweaters_l106_106857


namespace min_value_geometric_sequence_l106_106907

noncomputable def geometric_min_value (b1 b2 b3 : ℝ) (s : ℝ) : ℝ :=
  3 * b2 + 4 * b3

theorem min_value_geometric_sequence (s : ℝ) :
  ∃ s : ℝ, 2 = b1 ∧ b2 = 2 * s ∧ b3 = 2 * s^2 ∧ 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end min_value_geometric_sequence_l106_106907


namespace point_c_in_second_quadrant_l106_106894

-- Definitions for the points
def PointA : ℝ × ℝ := (1, 2)
def PointB : ℝ × ℝ := (-1, -2)
def PointC : ℝ × ℝ := (-1, 2)
def PointD : ℝ × ℝ := (1, -2)

-- Definition of the second quadrant condition
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

theorem point_c_in_second_quadrant : in_second_quadrant PointC :=
sorry

end point_c_in_second_quadrant_l106_106894


namespace students_present_in_class_l106_106055

noncomputable def num_students : ℕ := 100
noncomputable def percent_boys : ℝ := 0.55
noncomputable def percent_girls : ℝ := 0.45
noncomputable def absent_boys_percent : ℝ := 0.16
noncomputable def absent_girls_percent : ℝ := 0.12

theorem students_present_in_class :
  let num_boys := percent_boys * num_students
  let num_girls := percent_girls * num_students
  let absent_boys := absent_boys_percent * num_boys
  let absent_girls := absent_girls_percent * num_girls
  let present_boys := num_boys - absent_boys
  let present_girls := num_girls - absent_girls
  present_boys + present_girls = 86 :=
by
  sorry

end students_present_in_class_l106_106055


namespace problem1_problem2_l106_106028

-- Define the first problem: For positive real numbers a and b,
-- with the condition a + b = 2, show that the minimum value of 
-- (1 / (1 + a) + 4 / (1 + b)) is 9/4.
theorem problem1 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  1 / (1 + a) + 4 / (1 + b) ≥ 9 / 4 :=
sorry

-- Define the second problem: For any positive real numbers a and b,
-- prove that a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1).
theorem problem2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end problem1_problem2_l106_106028


namespace sufficient_but_not_necessary_condition_for_hyperbola_l106_106180

theorem sufficient_but_not_necessary_condition_for_hyperbola (k : ℝ) :
  (∃ k : ℝ, k > 3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) ∧ 
  (∃ k : ℝ, k < -3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) :=
    sorry

end sufficient_but_not_necessary_condition_for_hyperbola_l106_106180


namespace correct_arrangements_l106_106273

open Finset Nat

-- Definitions for combinations and powers
def comb (n k : ℕ) : ℕ := choose n k

-- The number of computer rooms
def num_computer_rooms : ℕ := 6

-- The number of arrangements to open at least 2 out of 6 computer rooms
def arrangement_count1 : ℕ := 2^num_computer_rooms - (comb num_computer_rooms 0 + comb num_computer_rooms 1)

-- Another calculation for the number of arrangements to open at least 2 out of 6 computer rooms
def arrangement_count2 : ℕ := comb num_computer_rooms 2 + comb num_computer_rooms 3 + comb num_computer_rooms 4 + 
                               comb num_computer_rooms 5 + comb num_computer_rooms 6

theorem correct_arrangements :
  arrangement_count1 = arrangement_count2 := 
  sorry

end correct_arrangements_l106_106273


namespace average_sale_l106_106134

-- Defining the monthly sales as constants
def sale_month1 : ℝ := 6435
def sale_month2 : ℝ := 6927
def sale_month3 : ℝ := 6855
def sale_month4 : ℝ := 7230
def sale_month5 : ℝ := 6562
def sale_month6 : ℝ := 7391

-- The final theorem statement to prove the average sale
theorem average_sale : (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6900 := 
by 
  sorry

end average_sale_l106_106134


namespace angle_rotation_l106_106489

theorem angle_rotation (initial_angle : ℝ) (rotation : ℝ) :
  initial_angle = 30 → rotation = 450 → 
  ∃ (new_angle : ℝ), new_angle = 60 :=
by
  sorry

end angle_rotation_l106_106489


namespace inequality_am_gm_l106_106614

variable (a b c d : ℝ)

theorem inequality_am_gm (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9  :=
by    
  sorry

end inequality_am_gm_l106_106614


namespace max_value_quadratic_function_l106_106493

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 6*x - 7

theorem max_value_quadratic_function (t : ℝ) :
  (∃ x, t ≤ x ∧ x ≤ t + 2 ∧ ∀ x' ∈ set.Icc t (t + 2), quadratic_function x' ≤ quadratic_function x) ↔ t ≥ 3 :=
by
  sorry

end max_value_quadratic_function_l106_106493


namespace smallest_x_l106_106507

theorem smallest_x 
  (x : ℝ)
  (h : ( ( (5 * x - 20) / (4 * x - 5) ) ^ 2 + ( (5 * x - 20) / (4 * x - 5) ) ) = 6 ) :
  x = -10 / 3 := sorry

end smallest_x_l106_106507


namespace ratio_initial_to_doubled_l106_106977

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 5) = 105) : x / (2 * x) = 1 / 2 :=
by
  sorry

end ratio_initial_to_doubled_l106_106977


namespace solve_inequality_l106_106789

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l106_106789


namespace part1_part2_l106_106069

-- Part 1: Number of k-tuples of ordered subsets with empty intersection
theorem part1 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (∃ (f : Fin (n) → Fin (2^k - 1)), true) :=
sorry

-- Part 2: Number of k-tuples of subsets with chain condition
theorem part2 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (S.card = (k + 1)^n) :=
sorry

end part1_part2_l106_106069


namespace factorable_polynomial_l106_106084

theorem factorable_polynomial (a b : ℝ) :
  (∀ x y : ℝ, ∃ u v p q : ℝ, (x + uy + v) * (x + py + q) = x * (x + 4) + a * (y^2 - 1) + 2 * b * y) ↔
  (a + 2)^2 + b^2 = 4 :=
  sorry

end factorable_polynomial_l106_106084


namespace recording_incorrect_l106_106596

-- Definitions for given conditions
def qualifying_standard : ℝ := 1.5
def xiao_ming_jump : ℝ := 1.95
def xiao_liang_jump : ℝ := 1.23
def xiao_ming_recording : ℝ := 0.45
def xiao_liang_recording : ℝ := -0.23

-- The proof statement to verify the correctness of the recordings
theorem recording_incorrect :
  (xiao_ming_jump - qualifying_standard = xiao_ming_recording) ∧ 
  (xiao_liang_jump - qualifying_standard ≠ xiao_liang_recording) :=
by
  sorry

end recording_incorrect_l106_106596


namespace find_angle_l106_106750

theorem find_angle {x : ℝ} (h1 : ∀ i, 1 ≤ i ∧ i ≤ 9 → ∃ x, x > 0) (h2 : 9 * x = 900) : x = 100 :=
  sorry

end find_angle_l106_106750


namespace find_a8_l106_106573

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n ≥ 2, (2 * a n - 3) / (a n - 1) = 2) (h2 : a 2 = 1) : a 8 = 16 := 
sorry

end find_a8_l106_106573


namespace width_of_door_l106_106377

theorem width_of_door 
  (L W H : ℕ) 
  (cost_per_sq_ft : ℕ) 
  (door_height window_height window_width : ℕ) 
  (num_windows total_cost : ℕ) 
  (door_width : ℕ) 
  (total_wall_area area_door area_windows area_to_whitewash : ℕ)
  (raw_area_door raw_area_windows total_walls_to_paint : ℕ) 
  (cost_per_sq_ft_eq : cost_per_sq_ft = 9)
  (total_cost_eq : total_cost = 8154)
  (room_dimensions_eq : L = 25 ∧ W = 15 ∧ H = 12)
  (door_dimensions_eq : door_height = 6)
  (window_dimensions_eq : window_height = 3 ∧ window_width = 4)
  (num_windows_eq : num_windows = 3)
  (total_wall_area_eq : total_wall_area = 2 * (L * H) + 2 * (W * H))
  (raw_area_door_eq : raw_area_door = door_height * door_width)
  (raw_area_windows_eq : raw_area_windows = num_windows * (window_width * window_height))
  (total_walls_to_paint_eq : total_walls_to_paint = total_wall_area - raw_area_door - raw_area_windows)
  (area_to_whitewash_eq : area_to_whitewash = 924 - 6 * door_width)
  (total_cost_eq_calc : total_cost = area_to_whitewash * cost_per_sq_ft) :
  door_width = 3 := sorry

end width_of_door_l106_106377


namespace hypotenuse_length_l106_106241

def triangle_hypotenuse := ∃ (a b c : ℚ) (x : ℚ), 
  a = 9 ∧ b = 3 * x + 6 ∧ c = x + 15 ∧ 
  a + b + c = 45 ∧ 
  a^2 + b^2 = c^2 ∧ 
  x = 15 / 4 ∧ 
  c = 75 / 4

theorem hypotenuse_length : triangle_hypotenuse :=
sorry

end hypotenuse_length_l106_106241


namespace arcsin_sqrt_three_over_two_l106_106690

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l106_106690


namespace infinite_series_sum_l106_106873

noncomputable def S : ℝ :=
∑' n, (if n % 3 == 0 then 1 / (3 ^ (n / 3)) else if n % 3 == 1 then -1 / (3 ^ (n / 3 + 1)) else -1 / (3 ^ (n / 3 + 2)))

theorem infinite_series_sum : S = 15 / 26 := by
  sorry

end infinite_series_sum_l106_106873


namespace distance_after_3_minutes_l106_106838

/-- Let truck_speed be 65 km/h and car_speed be 85 km/h.
    Let time_in_minutes be 3 and converted to hours it is 0.05 hours.
    The goal is to prove that the distance between the truck and the car
    after 3 minutes is 1 kilometer. -/
def truck_speed : ℝ := 65 -- speed in km/h
def car_speed : ℝ := 85 -- speed in km/h
def time_in_minutes : ℝ := 3 -- time in minutes
def time_in_hours : ℝ := time_in_minutes / 60 -- converted time in hours
def distance_truck := truck_speed * time_in_hours
def distance_car := car_speed * time_in_hours
def distance_between : ℝ := distance_car - distance_truck

theorem distance_after_3_minutes : distance_between = 1 := by
  -- Proof steps would go here
  sorry

end distance_after_3_minutes_l106_106838


namespace incorrect_statement_l106_106955

theorem incorrect_statement : ¬ (∀ x : ℝ, x ≠ 0 → (1 / x = 1 ∨ 1 / x = -1)) :=
by
  -- Proof goes here
  sorry

end incorrect_statement_l106_106955


namespace strawberry_growth_rate_l106_106618

theorem strawberry_growth_rate
  (initial_plants : ℕ)
  (months : ℕ)
  (plants_given_away : ℕ)
  (total_plants_after : ℕ)
  (growth_rate : ℕ)
  (h_initial : initial_plants = 3)
  (h_months : months = 3)
  (h_given_away : plants_given_away = 4)
  (h_total_after : total_plants_after = 20)
  (h_equation : initial_plants + growth_rate * months - plants_given_away = total_plants_after) :
  growth_rate = 7 :=
sorry

end strawberry_growth_rate_l106_106618


namespace surface_area_of_circumscribed_sphere_l106_106427

theorem surface_area_of_circumscribed_sphere :
  let a := 2
  let AD := Real.sqrt (a^2 - (a/2)^2)
  let r := Real.sqrt (1 + 1 + AD^2) / 2
  4 * Real.pi * r^2 = 5 * Real.pi := by
  sorry

end surface_area_of_circumscribed_sphere_l106_106427


namespace geometric_sequence_first_term_l106_106500

-- Define factorial values for convenience
def fact (n : ℕ) : ℕ := Nat.factorial n
#eval fact 6 -- This should give us 720
#eval fact 7 -- This should give us 5040

-- State the hypotheses and the goal
theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^2 = 720)
  (h2 : a * r^5 = 5040) :
  a = 720 / (7^(2/3 : ℝ)) :=
by
  sorry

end geometric_sequence_first_term_l106_106500


namespace area_of_trapezoid_l106_106895

noncomputable def triangle_XYZ_is_isosceles : Prop := 
  ∃ (X Y Z : Type) (XY XZ : ℝ), XY = XZ

noncomputable def identical_smaller_triangles (area : ℝ) (num : ℕ) : Prop := 
  num = 9 ∧ area = 3

noncomputable def total_area_large_triangle (total_area : ℝ) : Prop := 
  total_area = 135

noncomputable def trapezoid_contains_smaller_triangles (contained : ℕ) : Prop :=
  contained = 4

theorem area_of_trapezoid (XYZ_area smaller_triangle_area : ℝ) 
    (num_smaller_triangles contained_smaller_triangles : ℕ) : 
    triangle_XYZ_is_isosceles → 
    identical_smaller_triangles smaller_triangle_area num_smaller_triangles →
    total_area_large_triangle XYZ_area →
    trapezoid_contains_smaller_triangles contained_smaller_triangles →
    (XYZ_area - contained_smaller_triangles * smaller_triangle_area) = 123 :=
by
  intros iso smaller_triangles total_area contained
  sorry

end area_of_trapezoid_l106_106895


namespace rhombus_side_length_l106_106592

-- Define the statement of the problem in Lean
theorem rhombus_side_length (a b m : ℝ) (h_eq1 : a + b = 10) (h_eq2 : a * b = 22) (h_area : 1 / 2 * a * b = 11) :
  let side_length := (1 / 2 * Real.sqrt (a^2 + b^2)) in
  side_length = Real.sqrt 14 :=
by
  -- Proof omitted
  sorry

end rhombus_side_length_l106_106592


namespace older_brother_has_17_stamps_l106_106097

def stamps_problem (y : ℕ) : Prop := y + (2 * y + 1) = 25

theorem older_brother_has_17_stamps (y : ℕ) (h : stamps_problem y) : 2 * y + 1 = 17 :=
by
  sorry

end older_brother_has_17_stamps_l106_106097


namespace sub_numbers_correct_l106_106627

theorem sub_numbers_correct : 
  (500.50 - 123.45 - 55 : ℝ) = 322.05 := by 
-- The proof can be filled in here
sorry

end sub_numbers_correct_l106_106627


namespace factory_dolls_per_day_l106_106986

-- Define the number of normal dolls made per day
def N : ℝ := 4800

-- Define the total number of dolls made per day as 1.33 times the number of normal dolls
def T : ℝ := 1.33 * N

-- The theorem statement to prove the factory makes 6384 dolls per day
theorem factory_dolls_per_day : T = 6384 :=
by
  -- Proof here
  sorry

end factory_dolls_per_day_l106_106986


namespace value_of_c_l106_106886

theorem value_of_c (a b c d w x y z : ℕ) (primes : ∀ p ∈ [w, x, y, z], Prime p)
  (h1 : w < x) (h2 : x < y) (h3 : y < z) 
  (h4 : (w^a) * (x^b) * (y^c) * (z^d) = 660) 
  (h5 : (a + b) - (c + d) = 1) : c = 1 :=
by {
  sorry
}

end value_of_c_l106_106886


namespace arithmetic_sequence_problem_l106_106194

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 + a 6 + a 8 + a 10 + a 12 = 60)
  (h2 : ∀ n, a (n + 1) = a n + d) :
  a 7 - (1 / 3) * a 5 = 8 :=
by
  sorry

end arithmetic_sequence_problem_l106_106194


namespace number_of_games_l106_106362

theorem number_of_games (total_points points_per_game : ℕ) (h1 : total_points = 21) (h2 : points_per_game = 7) : total_points / points_per_game = 3 := by
  sorry

end number_of_games_l106_106362


namespace determine_C_l106_106349
noncomputable def A : ℕ := sorry
noncomputable def B : ℕ := sorry
noncomputable def C : ℕ := sorry

-- Conditions
axiom cond1 : A + B + 1 = C + 10
axiom cond2 : B = A + 2

-- Proof statement
theorem determine_C : C = 1 :=
by {
  -- using the given conditions, deduce that C must equal 1
  sorry
}

end determine_C_l106_106349


namespace find_g_5_l106_106934

variable (g : ℝ → ℝ)

axiom func_eqn : ∀ x y : ℝ, x * g y = y * g x
axiom g_10 : g 10 = 15

theorem find_g_5 : g 5 = 7.5 :=
by
  sorry

end find_g_5_l106_106934


namespace area_of_triangle_with_medians_l106_106551

theorem area_of_triangle_with_medians (m1 m2 m3 : ℝ) (h_m1 : m1 = 12) (h_m2 : m2 = 15) (h_m3 : m3 = 21) :
  ∃(A : ℝ), A = 48 * Real.sqrt 6 :=
by
  sorry

end area_of_triangle_with_medians_l106_106551


namespace parabola_equation_l106_106830

theorem parabola_equation (A B : ℝ × ℝ) (x₁ x₂ y₁ y₂ p : ℝ) :
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  x₁ + x₂ = (p + 8) / 2 →
  x₁ * x₂ = 4 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 45 →
  (y₁ = 2 * x₁ - 4) →
  (y₂ = 2 * x₂ - 4) →
  ((y₁^2 = 2 * p * x₁) ∧ (y₂^2 = 2 * p * x₂)) →
  (y₁^2 = 4 * x₁ ∨ y₂^2 = -36 * x₂) := 
by {
  sorry
}

end parabola_equation_l106_106830


namespace hoodies_ownership_l106_106018

-- Step a): Defining conditions
variables (Fiona_casey_hoodies_total: ℕ) (Casey_difference: ℕ) (Alex_hoodies: ℕ)

-- Functions representing the constraints
def hoodies_owned_by_Fiona (F : ℕ) : Prop :=
  (F + (F + 2) + 3 = 15)

-- Step c): Prove the correct number of hoodies owned by each
theorem hoodies_ownership (F : ℕ) (H1 : hoodies_owned_by_Fiona F) : 
  F = 5 ∧ (F + 2 = 7) ∧ (3 = 3) :=
by {
  -- Skipping proof details
  sorry
}

end hoodies_ownership_l106_106018


namespace fernanda_total_time_to_finish_l106_106860

noncomputable def fernanda_days_to_finish_audiobooks
  (num_audiobooks : ℕ) (hours_per_audiobook : ℕ) (hours_listened_per_day : ℕ) : ℕ :=
num_audiobooks * hours_per_audiobook / hours_listened_per_day

-- Definitions based on the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Statement to prove
theorem fernanda_total_time_to_finish :
  fernanda_days_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 := 
sorry

end fernanda_total_time_to_finish_l106_106860


namespace jimmy_bread_packs_needed_l106_106465

theorem jimmy_bread_packs_needed 
  (sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (initial_bread_slices : ℕ)
  (slices_per_pack : ℕ)
  (H1 : sandwiches = 8)
  (H2 : slices_per_sandwich = 2)
  (H3 : initial_bread_slices = 0)
  (H4 : slices_per_pack = 4) : 
  (8 * 2) / 4 = 4 := 
sorry

end jimmy_bread_packs_needed_l106_106465


namespace marbles_in_larger_bottle_l106_106408

theorem marbles_in_larger_bottle 
  (small_bottle_volume : ℕ := 20)
  (small_bottle_marbles : ℕ := 40)
  (larger_bottle_volume : ℕ := 60) :
  (small_bottle_marbles / small_bottle_volume) * larger_bottle_volume = 120 := 
by
  sorry

end marbles_in_larger_bottle_l106_106408


namespace find_value_l106_106182

theorem find_value (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x - 3 = 12 :=
by
  sorry

end find_value_l106_106182


namespace barrel_to_cask_ratio_l106_106900

theorem barrel_to_cask_ratio
  (k : ℕ) -- k is the multiple
  (B C : ℕ) -- B is the amount a barrel can store, C is the amount a cask can store
  (h1 : C = 20) -- C stores 20 gallons
  (h2 : B = k * C + 3) -- A barrel stores 3 gallons more than k times the amount a cask stores
  (h3 : 4 * B + C = 172) -- The total storage capacity is 172 gallons
  : B / C = 19 / 10 :=
sorry

end barrel_to_cask_ratio_l106_106900


namespace find_natural_n_l106_106708

theorem find_natural_n (n x y k : ℕ) (h_rel_prime : Nat.gcd x y = 1) (h_k_gt_one : k > 1) (h_eq : 3^n = x^k + y^k) :
  n = 2 := by
  sorry

end find_natural_n_l106_106708


namespace find_ordered_triples_l106_106291

-- Define the problem conditions using Lean structures.
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_ordered_triples (a b c : ℕ) :
  (is_perfect_square (a^2 + 2 * b + c) ∧
   is_perfect_square (b^2 + 2 * c + a) ∧
   is_perfect_square (c^2 + 2 * a + b))
  ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 43 ∧ b = 127 ∧ c = 106) :=
by sorry

end find_ordered_triples_l106_106291


namespace minimize_cost_l106_106974

noncomputable def cost_function (x : ℝ) : ℝ :=
  (1 / 2) * (x + 5)^2 + 1000 / (x + 5)

theorem minimize_cost :
  (∀ x, 2 ≤ x ∧ x ≤ 8 → cost_function x ≥ 150) ∧ cost_function 5 = 150 :=
by
  sorry

end minimize_cost_l106_106974


namespace probability_of_even_sum_is_two_thirds_l106_106719

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def choose_4_without_2 : ℕ := (Nat.factorial 11) / ((Nat.factorial 4) * (Nat.factorial 7))

noncomputable def choose_4_from_12 : ℕ := (Nat.factorial 12) / ((Nat.factorial 4) * (Nat.factorial 8))

noncomputable def probability_even_sum : ℚ := (choose_4_without_2 : ℚ) / (choose_4_from_12 : ℚ)

theorem probability_of_even_sum_is_two_thirds :
  probability_even_sum = (2 / 3 : ℚ) :=
sorry

end probability_of_even_sum_is_two_thirds_l106_106719


namespace min_increase_velocity_correct_l106_106252

noncomputable def min_increase_velocity (V_A V_B V_C V_D : ℝ) (dist_AC dist_CD : ℝ) : ℝ :=
  let t_AC := dist_AC / (V_A + V_C)
  let t_AB := 30 / (V_A - V_B)
  let t_AD := (dist_AC + dist_CD) / (V_A + V_D)
  let new_velocity_A := (dist_AC + dist_CD) / t_AC - V_D
  new_velocity_A - V_A

theorem min_increase_velocity_correct :
  min_increase_velocity 80 50 70 60 300 400 = 210 :=
by
  sorry

end min_increase_velocity_correct_l106_106252


namespace quadratic_function_properties_l106_106022

-- Definitions based on given conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def pointCondition (a b c : ℝ) : Prop := quadraticFunction a b c (-2) = 0
def inequalityCondition (a b c : ℝ) : Prop := ∀ x : ℝ, 2 * x ≤ quadraticFunction a b c x ∧ quadraticFunction a b c x ≤ (1 / 2) * x^2 + 2
def strengthenCondition (f : ℝ → ℝ) (t : ℝ) : Prop := ∀ x, -1 ≤ x ∧ x ≤ 1 → f (x + t) < f (x / 3)

-- Our primary statement to prove
theorem quadratic_function_properties :
  ∃ a b c, pointCondition a b c ∧ inequalityCondition a b c ∧
           (a = 1 / 4 ∧ b = 1 ∧ c = 1) ∧
           (∀ t, (-8 / 3 < t ∧ t < -2 / 3) ↔ strengthenCondition (quadraticFunction (1 / 4) 1 1) t) :=
by sorry 

end quadratic_function_properties_l106_106022


namespace smallest_positive_n_l106_106561

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3 / 2], [Real.sqrt 3 / 2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], [0, 1]]

theorem smallest_positive_n :
  ∃ n : ℕ, 0 < n ∧ rotation_matrix ^ n = identity_matrix ∧ ∀ m : ℕ, 0 < m ∧ rotation_matrix ^ m = identity_matrix → n ≤ m :=
by
  sorry

end smallest_positive_n_l106_106561


namespace jessie_final_position_l106_106079

theorem jessie_final_position :
  ∃ y : ℕ,
  (0 + 6 * 4 = 24) ∧
  (y = 24) :=
by
  sorry

end jessie_final_position_l106_106079


namespace find_angle_C_range_of_a_plus_b_l106_106052

variables {A B C a b c : ℝ}

-- Define the conditions
def conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a + c) * (Real.sin A - Real.sin C) = Real.sin B * (a - b)

-- Proof problem 1: show angle C is π/3
theorem find_angle_C (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b c A B C) : 
  C = π / 3 :=
sorry

-- Proof problem 2: if c = 2, then show the range of a + b
theorem range_of_a_plus_b (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b 2 A B C) :
  2 < a + b ∧ a + b ≤ 4 :=
sorry

end find_angle_C_range_of_a_plus_b_l106_106052


namespace probability_age_20_to_40_l106_106350

theorem probability_age_20_to_40 
    (total_people : ℕ) (aged_20_to_30 : ℕ) (aged_30_to_40 : ℕ) 
    (h_total : total_people = 350) 
    (h_aged_20_to_30 : aged_20_to_30 = 105) 
    (h_aged_30_to_40 : aged_30_to_40 = 85) : 
    (190 / 350 : ℚ) = 19 / 35 := 
by 
  sorry

end probability_age_20_to_40_l106_106350


namespace prove_AP_BP_CP_product_l106_106024

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

end prove_AP_BP_CP_product_l106_106024


namespace factorizations_of_2079_l106_106039

theorem factorizations_of_2079 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 2079 ∧ (a, b) = (21, 99) ∨ (a, b) = (33, 63) :=
sorry

end factorizations_of_2079_l106_106039


namespace find_k_l106_106292

theorem find_k (a k : ℝ) (h : a ≠ 0) (h1 : 3 * a + a = -12)
  (h2 : (3 * a) * a = k) : k = 27 :=
by
  sorry

end find_k_l106_106292


namespace p_suff_but_not_nec_q_l106_106585

variable (p q : Prop)

-- Given conditions: ¬p is a necessary but not sufficient condition for ¬q.
def neg_p_nec_but_not_suff_neg_q : Prop :=
  (¬q → ¬p) ∧ ¬(¬p → ¬q)

-- Concluding statement: p is a sufficient but not necessary condition for q.
theorem p_suff_but_not_nec_q 
  (h : neg_p_nec_but_not_suff_neg_q p q) : (p → q) ∧ ¬(q → p) := 
sorry

end p_suff_but_not_nec_q_l106_106585


namespace bucket_full_weight_l106_106257

variable (c d : ℝ)

def total_weight_definition (x y : ℝ) := x + y

theorem bucket_full_weight (x y : ℝ) 
  (h₁ : x + 3/4 * y = c) 
  (h₂ : x + 1/3 * y = d) : 
  total_weight_definition x y = (8 * c - 3 * d) / 5 :=
sorry

end bucket_full_weight_l106_106257


namespace speed_of_stream_l106_106950

theorem speed_of_stream (v : ℝ) (canoe_speed : ℝ) 
  (upstream_speed_condition : canoe_speed - v = 3) 
  (downstream_speed_condition : canoe_speed + v = 12) :
  v = 4.5 := 
by 
  sorry

end speed_of_stream_l106_106950


namespace alpha_value_l106_106929

-- Define the conditions in Lean
variables (α β γ k : ℝ)

-- Mathematically equivalent problem statements translated to Lean
theorem alpha_value :
  (∀ β γ, α = (k * γ) / β) → -- proportionality condition
  (α = 4) →
  (β = 27) →
  (γ = 3) →
  (∀ β γ, β = -81 → γ = 9 → α = -4) :=
by
  sorry

end alpha_value_l106_106929


namespace system_of_equations_correct_l106_106822

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end system_of_equations_correct_l106_106822


namespace chromium_percentage_in_second_alloy_l106_106747

theorem chromium_percentage_in_second_alloy (x : ℝ) :
  (15 * 0.12) + (35 * (x / 100)) = 50 * 0.106 → x = 10 :=
by
  sorry

end chromium_percentage_in_second_alloy_l106_106747


namespace max_peaceful_clients_kept_l106_106251

-- Defining the types for knights, liars, and troublemakers
def Person : Type := ℕ

noncomputable def isKnight : Person → Prop := sorry
noncomputable def isLiar : Person → Prop := sorry
noncomputable def isTroublemaker : Person → Prop := sorry

-- Total number of people in the bar
def totalPeople : ℕ := 30

-- Number of knights, liars, and troublemakers
def numberKnights : ℕ := 10
def numberLiars : ℕ := 10
def numberTroublemakers : ℕ := 10

-- The bartender's goal: get rid of all troublemakers and keep as many peaceful clients as possible
def maxPeacefulClients (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ): ℕ :=
  total - troublemakers

-- Statement to be proved
theorem max_peaceful_clients_kept (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ)
  (h_total : total = 30)
  (h_knights : knights = 10)
  (h_liars : liars = 10)
  (h_troublemakers : troublemakers = 10) :
  maxPeacefulClients total knights liars troublemakers = 19 :=
by
  -- Proof steps go here
  sorry

end max_peaceful_clients_kept_l106_106251


namespace smallest_positive_integer_n_l106_106564

open Matrix

def is_rotation_matrix_240_degrees (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A = ![![1 / 2, - (Real.sqrt 3) / 2], ![(Real.sqrt 3) / 2, 1 / 2]]

noncomputable def I_2 : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem smallest_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧
  is_rotation_matrix_240_degrees (A \^ n) ∧
  (A^n = I_2) → n = 3 :=
sorry

end smallest_positive_integer_n_l106_106564


namespace mul_112_54_l106_106518

theorem mul_112_54 : 112 * 54 = 6048 :=
by
  sorry

end mul_112_54_l106_106518


namespace beau_age_today_l106_106846

-- Definitions based on conditions
def sons_are_triplets : Prop := ∀ (i j : Nat), i ≠ j → i = 0 ∨ i = 1 ∨ i = 2 → j = 0 ∨ j = 1 ∨ j = 2
def sons_age_today : Nat := 16
def sum_of_ages_equals_beau_age_3_years_ago (beau_age_3_years_ago : Nat) : Prop :=
  beau_age_3_years_ago = 3 * (sons_age_today - 3)

-- Proposition to prove
theorem beau_age_today (beau_age_3_years_ago : Nat) (h_triplets : sons_are_triplets) 
  (h_ages_sum : sum_of_ages_equals_beau_age_3_years_ago beau_age_3_years_ago) : 
  beau_age_3_years_ago + 3 = 42 := 
by
  sorry

end beau_age_today_l106_106846


namespace basketball_lineups_l106_106287

noncomputable def num_starting_lineups (total_players : ℕ) (fixed_players : ℕ) (chosen_players : ℕ) : ℕ :=
  Nat.choose (total_players - fixed_players) (chosen_players - fixed_players)

theorem basketball_lineups :
  num_starting_lineups 15 2 6 = 715 := by
  sorry

end basketball_lineups_l106_106287


namespace speed_in_still_water_l106_106395

theorem speed_in_still_water (upstream_speed : ℝ) (downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 45) (h_downstream : downstream_speed = 55) : 
  (upstream_speed + downstream_speed) / 2 = 50 := 
by
  rw [h_upstream, h_downstream] 
  norm_num  -- simplifies the numeric expression
  done

end speed_in_still_water_l106_106395


namespace task_D_cannot_be_sampled_l106_106651

def task_A := "Measuring the range of a batch of shells"
def task_B := "Determining the content of a certain microorganism in ocean waters"
def task_C := "Calculating the difficulty of each question on the math test after the college entrance examination"
def task_D := "Checking the height and weight of all sophomore students in a school"

def sampling_method (description: String) : Prop :=
  description = task_A ∨ description = task_B ∨ description = task_C

theorem task_D_cannot_be_sampled : ¬ sampling_method task_D :=
sorry

end task_D_cannot_be_sampled_l106_106651


namespace quadratic_ineq_real_solutions_l106_106009

theorem quadratic_ineq_real_solutions (d : ℝ) (h₀ : 0 < d) :
  (∀ x : ℝ, x^2 - 8 * x + d < 0 → 0 < d ∧ d < 16) :=
by
  sorry

end quadratic_ineq_real_solutions_l106_106009


namespace telescope_visual_range_increase_l106_106659

theorem telescope_visual_range_increase (original_range : ℝ) (increase_percent : ℝ) 
(h1 : original_range = 100) (h2 : increase_percent = 0.50) : 
original_range + (increase_percent * original_range) = 150 := 
sorry

end telescope_visual_range_increase_l106_106659


namespace cyclist_return_trip_average_speed_l106_106965

theorem cyclist_return_trip_average_speed :
  let first_leg_distance := 12
  let second_leg_distance := 24
  let first_leg_speed := 8
  let second_leg_speed := 12
  let round_trip_time := 7.5
  let distance_to_destination := first_leg_distance + second_leg_distance
  let time_to_destination := (first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)
  let return_trip_time := round_trip_time - time_to_destination
  let return_trip_distance := distance_to_destination
  (return_trip_distance / return_trip_time) = 9 := 
by
  sorry

end cyclist_return_trip_average_speed_l106_106965


namespace metal_waste_l106_106519

theorem metal_waste
  (length : ℝ) (width : ℝ) (diameter : ℝ) (radius : ℝ)
  (rect_area : ℝ) (circle_area : ℝ) (inscribed_rect_area : ℝ)
  (h_length : length = 20)
  (h_width : width = 10)
  (h_diameter : diameter = 10)
  (h_radius : radius = diameter / 2)
  (h_rect_area : rect_area = length * width)
  (h_circle_area : circle_area = Real.pi * radius^2)
  (h_inscribed_rect_area : inscribed_rect_area = (width * length) / 5):

  rect_area - circle_area + circle_area - inscribed_rect_area = 160 := 
sorry

end metal_waste_l106_106519


namespace evaluate_expression_l106_106535

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l106_106535


namespace license_plate_difference_l106_106889

theorem license_plate_difference :
  (26^4 * 10^3 - 26^5 * 10^2 = -731161600) :=
sorry

end license_plate_difference_l106_106889


namespace value_of_a_l106_106297

theorem value_of_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -3) (h3 : a * x - y = 1) : a = -2 :=
by
  -- Placeholder for the proof
  sorry

end value_of_a_l106_106297


namespace relative_complement_correct_l106_106437

noncomputable def M : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}
def complement_M_N : Set ℤ := {x ∈ M | x ∉ N}

theorem relative_complement_correct : complement_M_N = {-1, 0, 3} := 
by
  sorry

end relative_complement_correct_l106_106437


namespace find_m_over_n_l106_106933

noncomputable
def ellipse_intersection_midpoint (m n : ℝ) (P : ℝ × ℝ) : Prop :=
  let M := (P.1, 1 - P.1)
  let N := (1 - P.2, P.2)
  let midpoint_MN := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  P = midpoint_MN

noncomputable
def ellipse_condition (m n : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

noncomputable
def line_condition (x y : ℝ) : Prop :=
  x + y = 1

noncomputable
def slope_OP_condition (P : ℝ × ℝ) : Prop :=
  P.2 / P.1 = (Real.sqrt 2 / 2)

theorem find_m_over_n
  (m n : ℝ)
  (P : ℝ × ℝ)
  (h1 : ellipse_condition m n P.1 P.2)
  (h2 : line_condition P.1 P.2)
  (h3 : slope_OP_condition P)
  (h4 : ellipse_intersection_midpoint m n P) :
  (m / n = 1) :=
sorry

end find_m_over_n_l106_106933


namespace bricks_in_top_half_l106_106361

theorem bricks_in_top_half (total_rows bottom_rows top_rows bricks_per_bottom_row total_bricks bricks_per_top_row: ℕ) 
  (h_total_rows : total_rows = 10)
  (h_bottom_rows : bottom_rows = 5)
  (h_top_rows : top_rows = 5)
  (h_bricks_per_bottom_row : bricks_per_bottom_row = 12)
  (h_total_bricks : total_bricks = 100)
  (h_bricks_per_top_row : bricks_per_top_row = (total_bricks - bottom_rows * bricks_per_bottom_row) / top_rows) : 
  bricks_per_top_row = 8 := 
by 
  sorry

end bricks_in_top_half_l106_106361


namespace recurring_decimal_to_fraction_l106_106290

theorem recurring_decimal_to_fraction :
  ∃ (frac : ℚ), frac = 1045 / 1998 ∧ 0.5 + (23 / 999) = frac :=
by
  sorry

end recurring_decimal_to_fraction_l106_106290


namespace find_incomes_l106_106994

theorem find_incomes (M N O P Q : ℝ) 
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (O + P) / 2 = 6800)
  (h4 : (P + Q) / 2 = 7500)
  (h5 : (M + O + Q) / 3 = 6000) :
  M = 300 ∧ N = 9800 ∧ O = 2700 ∧ P = 10900 ∧ Q = 4100 :=
by
  sorry


end find_incomes_l106_106994


namespace jennifer_boxes_l106_106603

theorem jennifer_boxes (kim_sold : ℕ) (h₁ : kim_sold = 54) (h₂ : ∃ jennifer_sold, jennifer_sold = kim_sold + 17) : ∃ jennifer_sold, jennifer_sold = 71 := by
  sorry

end jennifer_boxes_l106_106603


namespace evaluate_expression_l106_106536

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l106_106536


namespace total_employee_costs_in_February_l106_106892

def weekly_earnings (hours_per_week : ℕ) (rate_per_hour : ℕ) : ℕ :=
  hours_per_week * rate_per_hour

def monthly_earnings 
  (hours_per_week : ℕ) 
  (rate_per_hour : ℕ) 
  (weeks_worked : ℕ) 
  (bonus_deduction : ℕ := 0) 
  : ℕ :=
  weeks_worked * weekly_earnings hours_per_week rate_per_hour + bonus_deduction

theorem total_employee_costs_in_February 
  (hours_Fiona : ℕ := 40) (rate_Fiona : ℕ := 20) (weeks_worked_Fiona : ℕ := 3)
  (hours_John : ℕ := 30) (rate_John : ℕ := 22) (overtime_hours_John : ℕ := 10)
  (hours_Jeremy : ℕ := 25) (rate_Jeremy : ℕ := 18) (bonus_Jeremy : ℕ := 200)
  (hours_Katie : ℕ := 35) (rate_Katie : ℕ := 21) (deduction_Katie : ℕ := 150)
  (hours_Matt : ℕ := 28) (rate_Matt : ℕ := 19) : 
  monthly_earnings hours_Fiona rate_Fiona weeks_worked_Fiona 
  + monthly_earnings hours_John rate_John 4 
    + overtime_hours_John * (rate_John * 3 / 2)
  + monthly_earnings hours_Jeremy rate_Jeremy 4 bonus_Jeremy
  + monthly_earnings hours_Katie rate_Katie 4 - deduction_Katie
  + monthly_earnings hours_Matt rate_Matt 4 = 13278 := 
by sorry

end total_employee_costs_in_February_l106_106892


namespace sector_angle_measure_l106_106165

theorem sector_angle_measure
  (r l : ℝ)
  (h1 : 2 * r + l = 4)
  (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 :=
sorry

end sector_angle_measure_l106_106165


namespace part_a_l106_106913

def system_of_equations (x y z a : ℝ) := 
  (x - a * y = y * z) ∧ (y - a * z = z * x) ∧ (z - a * x = x * y)

theorem part_a (x y z : ℝ) : 
  system_of_equations x y z 0 ↔ (x = 0 ∧ y = 0 ∧ z = 0) 
  ∨ (∃ x, y = x ∧ z = 1) 
  ∨ (∃ x, y = -x ∧ z = -1) := 
  sorry

end part_a_l106_106913


namespace salary_increase_l106_106803

theorem salary_increase (x : ℝ) (y : ℝ) :
  (1000 : ℝ) * 80 + 50 = y → y - (50 + 80 * x) = 80 :=
by
  intros h
  sorry

end salary_increase_l106_106803


namespace remaining_fruits_correct_l106_106673

-- The definitions for the number of fruits in terms of the number of plums
def apples := 180
def plums := apples / 3
def pears := 2 * plums
def cherries := 4 * apples

-- Damien's portion of each type of fruit picked
def apples_picked := (3/5) * apples
def plums_picked := (2/3) * plums
def pears_picked := (3/4) * pears
def cherries_picked := (7/10) * cherries

-- The remaining number of fruits
def apples_remaining := apples - apples_picked
def plums_remaining := plums - plums_picked
def pears_remaining := pears - pears_picked
def cherries_remaining := cherries - cherries_picked

-- The total remaining number of fruits
def total_remaining_fruits := apples_remaining + plums_remaining + pears_remaining + cherries_remaining

theorem remaining_fruits_correct :
  total_remaining_fruits = 338 :=
by {
  -- The conditions ensure that the imported libraries are broad
  sorry
}

end remaining_fruits_correct_l106_106673


namespace least_possible_value_a2008_l106_106359

theorem least_possible_value_a2008 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, a n < a (n + 1)) 
  (h2 : ∀ i j k l, 1 ≤ i → i < j → j ≤ k → k < l → i + l = j + k → a i + a l > a j + a k)
  : a 2008 ≥ 2015029 :=
sorry

end least_possible_value_a2008_l106_106359


namespace intersection_of_A_and_B_l106_106475

open Set

noncomputable def A : Set ℤ := {1, 3, 5, 7}
noncomputable def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by
  sorry

end intersection_of_A_and_B_l106_106475


namespace train_speed_l106_106811

theorem train_speed
  (length_of_train : ℝ) 
  (time_to_cross : ℝ) 
  (train_length_is_140 : length_of_train = 140)
  (time_is_6 : time_to_cross = 6) :
  (length_of_train / time_to_cross) = 23.33 :=
sorry

end train_speed_l106_106811


namespace geometric_series_sum_l106_106851

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (last_term : ℝ) 
  (h_a : a = 1) (h_r : r = -3) 
  (h_last_term : last_term = 6561) 
  (h_last_term_eq : a * r^n = last_term) : 
  a * (r^n - 1) / (r - 1) = 4921.25 :=
by
  -- Proof goes here
  sorry

end geometric_series_sum_l106_106851


namespace breadth_of_plot_l106_106229

theorem breadth_of_plot (b l : ℝ) (h1 : l * b = 18 * b) (h2 : l - b = 10) : b = 8 :=
by
  sorry

end breadth_of_plot_l106_106229


namespace pyramid_angles_sum_pi_over_four_l106_106458

theorem pyramid_angles_sum_pi_over_four :
  ∃ (α β : ℝ), 
    α + β = Real.pi / 4 ∧ 
    α = Real.arctan ((Real.sqrt 17 - 3) / 4) ∧ 
    β = Real.pi / 4 - Real.arctan ((Real.sqrt 17 - 3) / 4) :=
by
  sorry

end pyramid_angles_sum_pi_over_four_l106_106458


namespace sum_of_center_coordinates_l106_106418

theorem sum_of_center_coordinates : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 = 6*x - 10*y + 24) -> 
  (∃ (cx cy : ℝ), (x^2 - 6*x + y^2 + 10*y = (cx - 3)^2 + (cy + 5)^2 + 58) ∧ (cx + cy = -2)) :=
  sorry

end sum_of_center_coordinates_l106_106418


namespace find_g_at_7_l106_106213

noncomputable def g (x : ℝ) (a b c : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 4

theorem find_g_at_7 (a b c : ℝ) (h_symm : ∀ x : ℝ, g x a b c + g (-x) a b c = -8) (h_neg7: g (-7) a b c = 12) :
  g 7 a b c = -20 :=
by
  sorry

end find_g_at_7_l106_106213


namespace option_b_correct_l106_106954

theorem option_b_correct (a : ℝ) : (a ^ 3) * (a ^ 2) = a ^ 5 := 
by
  sorry

end option_b_correct_l106_106954


namespace arcsin_sqrt3_div_2_l106_106685

theorem arcsin_sqrt3_div_2 :
  ∃ θ : ℝ, θ ∈ Icc (-(Real.pi / 2)) (Real.pi / 2) ∧ Real.sin θ = (Real.sqrt 3) / 2 ∧ Real.arcsin ((Real.sqrt 3) / 2) = θ ∧ θ = (Real.pi / 3) :=
by
  sorry

end arcsin_sqrt3_div_2_l106_106685


namespace solution_set_inequalities_l106_106804

theorem solution_set_inequalities (x : ℝ) :
  (2 * x + 3 ≥ -1) ∧ (7 - 3 * x > 1) ↔ (-2 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_inequalities_l106_106804


namespace initial_marbles_count_l106_106400

theorem initial_marbles_count (g y : ℕ) 
  (h1 : (g + 3) * 4 = g + y + 3) 
  (h2 : 3 * g = g + y + 4) : 
  g + y = 8 := 
by 
  -- The proof will go here
  sorry

end initial_marbles_count_l106_106400


namespace convinced_of_twelve_models_vitya_review_58_offers_l106_106114

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l106_106114


namespace bus_dispatch_interval_l106_106479

-- Variables representing the speeds of Xiao Nan and the bus
variable (V_1 V_2 : ℝ)
-- The interval between the dispatch of two buses
variable (interval : ℝ)

-- Stating the conditions in Lean

-- Xiao Nan notices a bus catches up with him every 10 minutes
def cond1 : Prop := ∃ s, s = 10 * (V_1 - V_2)

-- Xiao Yu notices he encounters a bus every 5 minutes
def cond2 : Prop := ∃ s, s = 5 * (V_1 + 3 * V_2)

-- Proof statement
theorem bus_dispatch_interval (h1 : cond1 V_1 V_2) (h2 : cond2 V_1 V_2) : interval = 8 := by
  -- Proof would be provided here
  sorry

end bus_dispatch_interval_l106_106479


namespace sin_double_angle_l106_106308

theorem sin_double_angle (k α : ℝ) (h : Real.cos (π / 4 - α) = k) : Real.sin (2 * α) = 2 * k^2 - 1 := 
by
  sorry

end sin_double_angle_l106_106308


namespace z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l106_106877

variables (m : ℝ)

def z_re (m : ℝ) : ℝ := 2 * m^2 - 3 * m - 2
def z_im (m : ℝ) : ℝ := m^2 - 3 * m + 2

-- Part (Ⅰ) Question 1
theorem z_real_iff_m_1_or_2 (m : ℝ) :
  z_im m = 0 ↔ (m = 1 ∨ m = 2) :=
sorry

-- Part (Ⅰ) Question 2
theorem z_complex_iff_not_m_1_and_2 (m : ℝ) :
  ¬ (m = 1 ∨ m = 2) ↔ (m ≠ 1 ∧ m ≠ 2) :=
sorry

-- Part (Ⅰ) Question 3
theorem z_pure_imaginary_iff_m_neg_half (m : ℝ) :
  z_re m = 0 ∧ z_im m ≠ 0 ↔ (m = -1/2) :=
sorry

-- Part (Ⅱ) Question
theorem z_in_second_quadrant (m : ℝ) :
  z_re m < 0 ∧ z_im m > 0 ↔ -1/2 < m ∧ m < 1 :=
sorry

end z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l106_106877


namespace ratio_cher_to_gab_l106_106189

-- Definitions based on conditions
def sammy_score : ℕ := 20
def gab_score : ℕ := 2 * sammy_score
def opponent_score : ℕ := 85
def total_points : ℕ := opponent_score + 55
def cher_score : ℕ := total_points - (sammy_score + gab_score)

-- Theorem to prove the ratio of Cher's score to Gab's score
theorem ratio_cher_to_gab : cher_score / gab_score = 2 := by
  sorry

end ratio_cher_to_gab_l106_106189


namespace fraction_equality_l106_106344

theorem fraction_equality (x y a b : ℝ) (hx : x / y = 3) (h : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end fraction_equality_l106_106344


namespace total_high_sulfur_samples_l106_106832

-- Define the conditions as given in the problem
def total_samples : ℕ := 143
def heavy_oil_freq : ℚ := 2 / 11
def light_low_sulfur_freq : ℚ := 7 / 13
def no_low_sulfur_in_heavy_oil : Prop := ∀ (x : ℕ), (x / total_samples = heavy_oil_freq) → false

-- Define total high-sulfur samples
def num_heavy_oil := heavy_oil_freq * total_samples
def num_light_oil := total_samples - num_heavy_oil
def num_light_low_sulfur_oil := light_low_sulfur_freq * num_light_oil
def num_light_high_sulfur_oil := num_light_oil - num_light_low_sulfur_oil

-- Now state that we need to prove the total number of high-sulfur samples
theorem total_high_sulfur_samples : num_light_high_sulfur_oil + num_heavy_oil = 80 :=
by
  sorry

end total_high_sulfur_samples_l106_106832


namespace count_valid_subsets_l106_106448

open Finset

-- Define the set A excluding the primes
def A : Finset ℕ := {1, 4, 6, 8, 9, 10, 12}

-- Property definitions
def no_two_consecutive (S : Finset ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ y + 1 ∧ y ≠ x + 1

def element_condition (S : Finset ℕ) : Prop :=
  ∀ k (Hk : S.card = k), ∀ (x ∈ S), x ≤ 2 * k

-- Main theorem statement
theorem count_valid_subsets : (A.filter (λ S, no_two_consecutive S ∧ element_condition S)).card = 19 := by
  sorry

end count_valid_subsets_l106_106448


namespace fernanda_total_time_eq_90_days_l106_106861

-- Define the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Define the total time calculation
def total_time_to_finish_audiobooks (a h r : ℕ) : ℕ :=
  (h / r) * a

-- The assertion we need to prove
theorem fernanda_total_time_eq_90_days :
  total_time_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 :=
by sorry

end fernanda_total_time_eq_90_days_l106_106861


namespace romanov_family_savings_l106_106515

theorem romanov_family_savings :
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption := monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  savings = 3824 :=
by
  let meter_cost := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let monthly_night_consumption := 230
  let monthly_day_consumption :=monthly_consumption - monthly_night_consumption
  let night_rate := 3.4
  let day_rate := 5.2 
  let standard_rate := 4.6
  let years := 3

  let monthly_cost_multi_tariff := monthly_night_consumption * night_rate + monthly_day_consumption * day_rate
  let annual_cost_multi_tariff := monthly_cost_multi_tariff * 12
  let total_cost_multi_tariff := (annual_cost_multi_tariff * years) + meter_cost + installation_cost

  let monthly_cost_standard := monthly_consumption * standard_rate
  let annual_cost_standard := monthly_cost_standard * 12
  let total_cost_standard := annual_cost_standard * years

  let savings := total_cost_standard - total_cost_multi_tariff

  show savings = 3824 
  sorry

end romanov_family_savings_l106_106515


namespace find_tangent_lines_l106_106867

noncomputable def tangent_lines (x y : ℝ) : Prop :=
  (x = 2 ∨ 3 * x - 4 * y + 10 = 0)

theorem find_tangent_lines :
  ∃ (x y : ℝ), tangent_lines x y ∧ (x^2 + y^2 = 4) ∧ ((x, y) ≠ (2, 4)) :=
by
  sorry

end find_tangent_lines_l106_106867


namespace smallest_n_for_identity_matrix_l106_106558

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l106_106558


namespace find_angle_D_l106_106431

theorem find_angle_D (A B C D E F : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 40) 
  (triangle_sum1 : A + B + C + E + F = 180) (triangle_sum2 : D + E + F = 180) : 
  D = 125 :=
by
  -- Only adding a comment, proof omitted for the purpose of this task
  sorry

end find_angle_D_l106_106431


namespace domain_of_sqrt_sum_l106_106710

theorem domain_of_sqrt_sum (x : ℝ) (h1 : 3 + x ≥ 0) (h2 : 1 - x ≥ 0) : -3 ≤ x ∧ x ≤ 1 := by
  sorry

end domain_of_sqrt_sum_l106_106710


namespace zachary_seventh_day_cans_l106_106958

-- Define the number of cans found by Zachary every day.
def cans_found_on (day : ℕ) : ℕ :=
  if day = 1 then 4
  else if day = 2 then 9
  else if day = 3 then 14
  else 5 * (day - 1) - 1

-- The theorem to prove the number of cans found on the seventh day.
theorem zachary_seventh_day_cans : cans_found_on 7 = 34 :=
by 
  sorry

end zachary_seventh_day_cans_l106_106958


namespace exponential_decreasing_l106_106046

theorem exponential_decreasing (a : ℝ) : (∀ x y : ℝ, x < y → (2 * a - 1)^y < (2 * a - 1)^x) ↔ (1 / 2 < a ∧ a < 1) := 
by
    sorry

end exponential_decreasing_l106_106046


namespace maximum_temperature_difference_l106_106228

theorem maximum_temperature_difference
  (highest_temp : ℝ) (lowest_temp : ℝ)
  (h_highest : highest_temp = 58)
  (h_lowest : lowest_temp = -34) :
  highest_temp - lowest_temp = 92 :=
by sorry

end maximum_temperature_difference_l106_106228


namespace expand_product_l106_106999

noncomputable def question_expression (x : ℝ) := -3 * (2 * x + 4) * (x - 7)
noncomputable def correct_answer (x : ℝ) := -6 * x^2 + 30 * x + 84

theorem expand_product (x : ℝ) : question_expression x = correct_answer x := 
by sorry

end expand_product_l106_106999


namespace sum_of_fourth_powers_l106_106100

theorem sum_of_fourth_powers (n : ℤ) (h1 : n > 0) (h2 : (n - 1)^2 + n^2 + (n + 1)^2 = 9458) :
  (n - 1)^4 + n^4 + (n + 1)^4 = 30212622 :=
by sorry

end sum_of_fourth_powers_l106_106100


namespace sum_of_series_l106_106992

noncomputable def sum_term (k : ℕ) : ℝ :=
  (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

theorem sum_of_series : (∑' k : ℕ, sum_term (k + 1)) = 7 / 4 := by
  sorry

end sum_of_series_l106_106992


namespace sum_solution_equation_l106_106888

theorem sum_solution_equation (n : ℚ) : (∃ x : ℚ, (n / x = 3 - n) ∧ (x = 1 / (n + (3 - n)))) → n = 3 / 4 := by
  intros h
  sorry

end sum_solution_equation_l106_106888


namespace product_xyz_is_minus_one_l106_106452

-- Definitions of the variables and equations
variables (x y z : ℝ)

-- Assumptions based on the given conditions
def condition1 : Prop := x + (1 / y) = 2
def condition2 : Prop := y + (1 / z) = 2
def condition3 : Prop := z + (1 / x) = 2

-- The theorem stating the conclusion to be proved
theorem product_xyz_is_minus_one (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z x) : x * y * z = -1 :=
by sorry

end product_xyz_is_minus_one_l106_106452


namespace three_digit_avg_permutations_l106_106013

theorem three_digit_avg_permutations (a b c: ℕ) (A: ℕ) (h₀: 1 ≤ a ∧ a ≤ 9) (h₁: 0 ≤ b ∧ b ≤ 9) (h₂: 0 ≤ c ∧ c ≤ 9) (h₃: A = 100 * a + 10 * b + c):
  ((100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)) / 6 = A ↔ 7 * a = 3 * b + 4 * c := by
  sorry

end three_digit_avg_permutations_l106_106013


namespace median_of_first_ten_positive_integers_l106_106393

def first_ten_positive_integers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem median_of_first_ten_positive_integers : 
  ∃ median : ℝ, median = 5.5 := by
  sorry

end median_of_first_ten_positive_integers_l106_106393


namespace total_amount_shared_l106_106345

noncomputable def z : ℝ := 300
noncomputable def y : ℝ := 1.2 * z
noncomputable def x : ℝ := 1.25 * y

theorem total_amount_shared (z y x : ℝ) (hz : z = 300) (hy : y = 1.2 * z) (hx : x = 1.25 * y) :
  x + y + z = 1110 :=
by
  simp [hx, hy, hz]
  -- Add intermediate steps here if necessary
  sorry

end total_amount_shared_l106_106345


namespace isha_original_length_l106_106462

variable (current_length sharpened_off : ℕ)

-- Condition 1: Isha's pencil is now 14 inches long
def isha_current_length : current_length = 14 := sorry

-- Condition 2: She sharpened off 17 inches of her pencil
def isha_sharpened_off : sharpened_off = 17 := sorry

-- Statement to prove:
theorem isha_original_length (current_length sharpened_off : ℕ) 
  (h1 : current_length = 14) (h2 : sharpened_off = 17) :
  current_length + sharpened_off = 31 :=
by
  sorry

end isha_original_length_l106_106462


namespace find_first_term_l106_106732

open Int

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_first_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 2 = 1)
  (h_a4_a10 : a 3 + a 9 = 18) :
  a 0 = -3 :=
by
  sorry

end find_first_term_l106_106732


namespace median_of_first_ten_positive_integers_l106_106392

theorem median_of_first_ten_positive_integers : 
  let nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
  ∃ median : ℝ, median = 5.5 ∧ 
  median = (nums.nth 4).getOrElse 0 + (nums.nth 5).getOrElse 0 / 2 := 
by 
  sorry

end median_of_first_ten_positive_integers_l106_106392


namespace spending_on_gifts_l106_106327

-- Defining the conditions as Lean statements
def num_sons_teachers : ℕ := 3
def num_daughters_teachers : ℕ := 4
def cost_per_gift : ℕ := 10

-- The total number of teachers
def total_teachers : ℕ := num_sons_teachers + num_daughters_teachers

-- Proving that the total spending on gifts is $70
theorem spending_on_gifts : total_teachers * cost_per_gift = 70 :=
by
  -- proof goes here
  sorry

end spending_on_gifts_l106_106327


namespace measure_one_kg_grain_l106_106850

/-- Proving the possibility of measuring exactly 1 kg of grain
    using a balance scale, one 3 kg weight, and three weighings. -/
theorem measure_one_kg_grain :
  ∃ (weighings : ℕ) (balance_scale : ℕ → ℤ) (weight_3kg : ℤ → Prop),
  weighings = 3 ∧
  (∀ w, weight_3kg w ↔ w = 3) ∧
  ∀ n m, balance_scale n = 0 ∧ balance_scale m = 1 → true :=
sorry

end measure_one_kg_grain_l106_106850


namespace smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l106_106170

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * (Real.sin x)^2 - (1 + 2)

theorem smallest_period_of_f : ∀ x : ℝ, f (x + π) = f x := 
by sorry

theorem center_of_symmetry_of_f : ∀ k : ℤ, ∃ c : ℝ, ∀ x : ℝ, f (c - x) = f (c + x) := 
by sorry

theorem range_of_f_on_interval : 
  ∃ a b, (∀ x ∈ Set.Icc (-π / 4) (π / 4), f x ∈ Set.Icc a b) ∧ 
          (∀ y, y ∈ Set.Icc 3 5 → ∃ x ∈ Set.Icc (-π / 4) (π / 4), y = f x) := 
by sorry

end smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l106_106170


namespace beau_age_today_l106_106845

-- Definitions based on conditions
def sons_are_triplets : Prop := ∀ (i j : Nat), i ≠ j → i = 0 ∨ i = 1 ∨ i = 2 → j = 0 ∨ j = 1 ∨ j = 2
def sons_age_today : Nat := 16
def sum_of_ages_equals_beau_age_3_years_ago (beau_age_3_years_ago : Nat) : Prop :=
  beau_age_3_years_ago = 3 * (sons_age_today - 3)

-- Proposition to prove
theorem beau_age_today (beau_age_3_years_ago : Nat) (h_triplets : sons_are_triplets) 
  (h_ages_sum : sum_of_ages_equals_beau_age_3_years_ago beau_age_3_years_ago) : 
  beau_age_3_years_ago + 3 = 42 := 
by
  sorry

end beau_age_today_l106_106845


namespace analyze_quadratic_function_l106_106842

variable (x : ℝ)

def quadratic_function : ℝ → ℝ := λ x => x^2 - 4 * x + 6

theorem analyze_quadratic_function :
  (∃ y : ℝ, quadratic_function y = (x - 2)^2 + 2) ∧
  (∃ x0 : ℝ, quadratic_function x0 = (x0 - 2)^2 + 2 ∧ x0 = 2 ∧ (∀ y : ℝ, quadratic_function y ≥ 2)) :=
by
  sorry

end analyze_quadratic_function_l106_106842


namespace weekly_goal_cans_l106_106260

theorem weekly_goal_cans (c₁ c₂ c₃ c₄ c₅ : ℕ) (h₁ : c₁ = 20) (h₂ : c₂ = c₁ + 5) (h₃ : c₃ = c₂ + 5) 
  (h₄ : c₄ = c₃ + 5) (h₅ : c₅ = c₄ + 5) : 
  c₁ + c₂ + c₃ + c₄ + c₅ = 150 :=
by
  sorry

end weekly_goal_cans_l106_106260


namespace sufficient_not_necessary_ellipse_l106_106309

theorem sufficient_not_necessary_ellipse (m n : ℝ) (h : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → m > 0 ∧ n > 0 ∧ m ≠ n) ∧
  ¬(∀ x y : ℝ, mx^2 + ny^2 = 1 → m > 0 ∧ n > 0 ∧ m > n ∧ n > 0) :=
sorry

end sufficient_not_necessary_ellipse_l106_106309


namespace range_of_expression_l106_106729

theorem range_of_expression (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 ≤ β ∧ β ≤ π / 2) :
    -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
by
  sorry

end range_of_expression_l106_106729


namespace complement_of_A_in_U_l106_106323

variable {U : Set ℤ}
variable {A : Set ℤ}

theorem complement_of_A_in_U (hU : U = {-1, 0, 1}) (hA : A = {0, 1}) : U \ A = {-1} := by
  sorry

end complement_of_A_in_U_l106_106323


namespace grazing_area_of_goat_l106_106147

/-- 
Consider a circular park with a diameter of 50 feet, and a square monument with 10 feet on each side.
Sally ties her goat on one corner of the monument with a 20-foot rope. Calculate the total grazing area
around the monument considering the space limited by the park's boundary.
-/
theorem grazing_area_of_goat : 
  let park_radius := 25
  let monument_side := 10
  let rope_length := 20
  let monument_radius := monument_side / 2 
  let grazing_quarter_circle := (1 / 4) * Real.pi * rope_length^2
  let ungrazable_area := (1 / 4) * Real.pi * monument_radius^2
  grazing_quarter_circle - ungrazable_area = 93.75 * Real.pi :=
by
  sorry

end grazing_area_of_goat_l106_106147


namespace equal_diagonals_implies_quad_or_pent_l106_106717

-- Define a convex polygon with n edges and equal diagonals
structure ConvexPolygon (n : ℕ) :=
(edges : ℕ)
(convex : Prop)
(diagonalsEqualLength : Prop)

-- State the theorem to prove
theorem equal_diagonals_implies_quad_or_pent (n : ℕ) (poly : ConvexPolygon n) 
    (h1 : poly.convex) 
    (h2 : poly.diagonalsEqualLength) :
    (n = 4) ∨ (n = 5) :=
sorry

end equal_diagonals_implies_quad_or_pent_l106_106717


namespace sufficient_not_necessary_condition_l106_106567

noncomputable def sequence_increasing_condition (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) > |a n|

noncomputable def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n < a (n + 1)

theorem sufficient_not_necessary_condition (a : ℕ → ℝ) :
  sequence_increasing_condition a → is_increasing_sequence a ∧ ¬(∀ b : ℕ → ℝ, is_increasing_sequence b → sequence_increasing_condition b) :=
sorry

end sufficient_not_necessary_condition_l106_106567


namespace arc_length_of_sector_l106_106337

theorem arc_length_of_sector (theta : ℝ) (r : ℝ) (h_theta : theta = 90) (h_r : r = 6) : 
  (theta / 360) * 2 * Real.pi * r = 3 * Real.pi :=
by
  sorry

end arc_length_of_sector_l106_106337


namespace vitya_needs_58_offers_l106_106113

noncomputable def smallest_integer_k (P : ℝ → ℝ) : ℝ :=
  if H : ∃ k, k > P (100), then classical.some H else 0

theorem vitya_needs_58_offers :
  ∀ n : ℕ, n ≥ 13 → 
  (12:ℝ/13:ℝ) ^ smallest_integer_k (fun x => Real.log x / (Real.log 13 - Real.log 12)) < 0.01 :=
begin
  assume n h,
  rw smallest_integer_k,
  split_ifs,
  { sorry }, -- proof would go here
  { exfalso, exact sorry }, -- no proof steps provided
end

end vitya_needs_58_offers_l106_106113


namespace brother_books_total_l106_106988

theorem brother_books_total (pb_sarah hb_sarah : ℕ) (h_pb_sarah : pb_sarah = 6) (h_hb_sarah : hb_sarah = 4) : 
  let pb_brother := pb_sarah / 3 in
  let hb_brother := 2 * hb_sarah in
  pb_brother + hb_brother = 10 :=
by
  have h_pb_brother : pb_brother = 2 := by rw [h_pb_sarah] ; exact Nat.div_eq_of_lt (by decide) -- 6 / 3 = 2
  have h_hb_brother : hb_brother = 8 := by rw [h_hb_sarah] ; exact by norm_num -- 4 * 2 = 8
  rw [h_pb_brother, h_hb_brother]
  norm_num  -- 2 + 8 = 10
  sorry

end brother_books_total_l106_106988


namespace solution_to_abs_eq_l106_106547

theorem solution_to_abs_eq :
  ∀ x : ℤ, abs ((-5) + x) = 11 → (x = 16 ∨ x = -6) :=
by sorry

end solution_to_abs_eq_l106_106547


namespace find_first_term_of_arithmetic_sequence_l106_106169

theorem find_first_term_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 2)
  (h_d : d = -1/2) : a 1 = 3 :=
sorry

end find_first_term_of_arithmetic_sequence_l106_106169


namespace total_animals_in_jacobs_flock_l106_106922

-- Define the conditions of the problem
def one_third_of_animals_are_goats (total goats : ℕ) : Prop := 
  3 * goats = total

def twelve_more_sheep_than_goats (goats sheep : ℕ) : Prop :=
  sheep = goats + 12

-- Define the main theorem to prove
theorem total_animals_in_jacobs_flock : 
  ∃ total goats sheep : ℕ, one_third_of_animals_are_goats total goats ∧ 
                           twelve_more_sheep_than_goats goats sheep ∧ 
                           total = 36 := 
by
  sorry

end total_animals_in_jacobs_flock_l106_106922


namespace abs_inequality_l106_106782

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l106_106782


namespace sandra_age_l106_106626

theorem sandra_age (S : ℕ) (h1 : ∀ x : ℕ, x = 14) (h2 : S - 3 = 3 * (14 - 3)) : S = 36 :=
by sorry

end sandra_age_l106_106626


namespace problem_solution_exists_l106_106864

theorem problem_solution_exists (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)]
  (h : a > 0 ∧ b > 0 ∧ n > 0 ∧ a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ n = 2013 * k + 1 ∧ p = 2 := by
  sorry

end problem_solution_exists_l106_106864


namespace mean_of_combined_sets_l106_106380

theorem mean_of_combined_sets (mean_set1 : ℝ) (mean_set2 : ℝ) (n1 : ℕ) (n2 : ℕ)
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 27) (h3 : n1 = 7) (h4 : n2 = 8) :
  (mean_set1 * n1 + mean_set2 * n2) / (n1 + n2) = 21.4 := 
sorry

end mean_of_combined_sets_l106_106380


namespace solve_inequality_l106_106786

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l106_106786


namespace find_c_l106_106959

theorem find_c (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : C = 10 :=
by
  sorry

end find_c_l106_106959


namespace no_positive_integers_satisfy_equation_l106_106968

theorem no_positive_integers_satisfy_equation :
  ¬ ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a^2 = b^11 + 23 :=
by
  sorry

end no_positive_integers_satisfy_equation_l106_106968


namespace points_on_same_side_of_line_l106_106856

theorem points_on_same_side_of_line (m : ℝ) :
  (2 * 0 + 0 + m > 0 ∧ 2 * -1 + 1 + m > 0) ∨ 
  (2 * 0 + 0 + m < 0 ∧ 2 * -1 + 1 + m < 0) ↔ 
  (m < 0 ∨ m > 1) :=
by
  sorry

end points_on_same_side_of_line_l106_106856


namespace tom_age_ratio_l106_106254

-- Define the constants T and N with the given conditions
variables (T N : ℕ)
-- Tom's age T years, sum of three children's ages is also T
-- N years ago, Tom's age was three times the sum of children's ages then

-- We need to prove that T / N = 4 under these conditions
theorem tom_age_ratio (h1 : T = 3 * T - 8 * N) : T / N = 4 :=
sorry

end tom_age_ratio_l106_106254


namespace profit_at_original_price_l106_106270

theorem profit_at_original_price (x : ℝ) (h : 0.8 * x = 1.2) : x - 1 = 0.5 :=
by
  sorry

end profit_at_original_price_l106_106270


namespace solution_l106_106550

/-- Definition of the number with 2023 ones. -/
def x_2023 : ℕ := (10^2023 - 1) / 9

/-- Definition of the polynomial equation. -/
def polynomial_eq (x : ℕ) : ℤ :=
  567 * x^3 + 171 * x^2 + 15 * x - (7 * x + 5 * 10^2023 + 3 * 10^(2*2023))

/-- The solution x_2023 satisfies the polynomial equation. -/
theorem solution : polynomial_eq x_2023 = 0 := sorry

end solution_l106_106550


namespace intersection_of_P_and_Q_l106_106322

def P (x : ℝ) : Prop := 2 ≤ x ∧ x < 4
def Q (x : ℝ) : Prop := 3 * x - 7 ≥ 8 - 2 * x

theorem intersection_of_P_and_Q :
  ∀ x, P x ∧ Q x ↔ 3 ≤ x ∧ x < 4 :=
by
  sorry

end intersection_of_P_and_Q_l106_106322


namespace find_triangle_altitude_l106_106231

variable (A b h : ℝ)

theorem find_triangle_altitude (h_eq_40 :  A = 800 ∧ b = 40) : h = 40 :=
sorry

end find_triangle_altitude_l106_106231


namespace snickers_bars_needed_l106_106199

-- Definitions for the problem conditions
def total_required_points : ℕ := 2000
def bunnies_sold : ℕ := 8
def bunny_points : ℕ := 100
def snickers_points : ℕ := 25
def points_from_bunnies : ℕ := bunnies_sold * bunny_points
def remaining_points_needed : ℕ := total_required_points - points_from_bunnies

-- Define the problem statement to prove
theorem snickers_bars_needed : remaining_points_needed / snickers_points = 48 :=
by
  -- Skipping the proof steps
  sorry

end snickers_bars_needed_l106_106199


namespace cookies_per_bag_l106_106372

-- Definitions based on given conditions
def total_cookies : ℕ := 75
def number_of_bags : ℕ := 25

-- The statement of the problem
theorem cookies_per_bag : total_cookies / number_of_bags = 3 := by
  sorry

end cookies_per_bag_l106_106372


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l106_106692

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l106_106692


namespace shorter_piece_length_l106_106263

theorem shorter_piece_length (x : ℕ) (h1 : ∃ l : ℕ, x + l = 120 ∧ l = 2 * x + 15) : x = 35 :=
sorry

end shorter_piece_length_l106_106263


namespace notebook_cost_l106_106976

theorem notebook_cost (n c : ℝ) (h1 : n + c = 2.50) (h2 : n = c + 2) : n = 2.25 :=
by
  sorry

end notebook_cost_l106_106976


namespace find_remainder_l106_106472

theorem find_remainder (x : ℤ) (h : 0 < x ∧ 7 * x % 26 = 1) : (13 + 3 * x) % 26 = 6 :=
sorry

end find_remainder_l106_106472


namespace infinitely_many_triples_l106_106613

theorem infinitely_many_triples (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : ∀ k : ℕ, 
  ∃ (x y z : ℕ), 
    x = 2^(k * m * n + 1) ∧ 
    y = 2^(n + n * k * (m * n + 1)) ∧ 
    z = 2^(m + m * k * (m * n + 1)) ∧ 
    x^(m * n + 1) = y^m + z^n := 
by 
  intros k
  use 2^(k * m * n + 1), 2^(n + n * k * (m * n + 1)), 2^(m + m * k * (m * n + 1))
  simp
  sorry

end infinitely_many_triples_l106_106613


namespace magnolia_trees_below_threshold_l106_106998

-- Define the initial number of trees and the function describing the decrease
def initial_tree_count (N₀ : ℕ) (t : ℕ) : ℝ := N₀ * (0.8 ^ t)

-- Define the year when the number of trees is less than 25% of initial trees
theorem magnolia_trees_below_threshold (N₀ : ℕ) : (t : ℕ) -> initial_tree_count N₀ t < 0.25 * N₀ -> t > 14 := 
-- Provide the required statement but omit the actual proof with "sorry"
by sorry

end magnolia_trees_below_threshold_l106_106998


namespace three_xy_eq_24_l106_106586

variable {x y : ℝ}

theorem three_xy_eq_24 (h : x * (x + 3 * y) = x^2 + 24) : 3 * x * y = 24 :=
sorry

end three_xy_eq_24_l106_106586


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l106_106693

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l106_106693


namespace sum_of_cubes_l106_106049

-- Definitions based on the conditions
variables (a b : ℝ)
variables (h1 : a + b = 2) (h2 : a * b = -3)

-- The Lean statement to prove the sum of their cubes is 26
theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
by
  sorry

end sum_of_cubes_l106_106049


namespace average_of_second_class_l106_106093

variable (average1 : ℝ) (average2 : ℝ) (combined_average : ℝ) (n1 : ℕ) (n2 : ℕ)

theorem average_of_second_class
  (h1 : n1 = 25) 
  (h2 : average1 = 40) 
  (h3 : n2 = 30) 
  (h4 : combined_average = 50.90909090909091) 
  (h5 : n1 + n2 = 55) 
  (h6 : n2 * average2 = 55 * combined_average - n1 * average1) :
  average2 = 60 := by
  sorry

end average_of_second_class_l106_106093


namespace deepak_present_age_l106_106966

def present_age_rahul (x : ℕ) : ℕ := 4 * x
def present_age_deepak (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age : ∀ (x : ℕ), 
  (present_age_rahul x + 22 = 26) →
  present_age_deepak x = 3 := 
by
  intros x h
  sorry

end deepak_present_age_l106_106966


namespace ordered_pairs_count_l106_106583

theorem ordered_pairs_count : 
    ∃ (s : Finset (ℝ × ℝ)), 
        (∀ (x y : ℝ), (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1 ↔ (x, y) ∈ s)) ∧ 
        s.card = 3 :=
    by
    sorry

end ordered_pairs_count_l106_106583


namespace small_possible_value_l106_106307

theorem small_possible_value (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : 2^12 * 3^3 = a^b) : a + b = 110593 := by
  sorry

end small_possible_value_l106_106307


namespace total_squares_l106_106881

theorem total_squares (num_groups : ℕ) (squares_per_group : ℕ) (total : ℕ) 
  (h1 : num_groups = 5) (h2 : squares_per_group = 5) (h3 : total = num_groups * squares_per_group) : 
  total = 25 :=
by
  rw [h1, h2] at h3
  exact h3

end total_squares_l106_106881


namespace least_number_of_cans_l106_106265

theorem least_number_of_cans 
  (Maaza_volume : ℕ) (Pepsi_volume : ℕ) (Sprite_volume : ℕ) 
  (h1 : Maaza_volume = 80) (h2 : Pepsi_volume = 144) (h3 : Sprite_volume = 368) :
  (Maaza_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) +
  (Pepsi_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) +
  (Sprite_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) = 37 := by
  sorry

end least_number_of_cans_l106_106265


namespace initial_num_nuts_l106_106104

theorem initial_num_nuts (total_nuts : ℕ) (h1 : 1/6 * total_nuts = 5) : total_nuts = 30 := 
sorry

end initial_num_nuts_l106_106104


namespace find_number_l106_106335

theorem find_number (N p q : ℝ) 
  (h1 : N / p = 6) 
  (h2 : N / q = 18) 
  (h3 : p - q = 1 / 3) : 
  N = 3 := 
by 
  sorry

end find_number_l106_106335


namespace cost_of_pencil_and_pen_l106_106095

variable (p q : ℝ)

axiom condition1 : 4 * p + 3 * q = 4.20
axiom condition2 : 3 * p + 4 * q = 4.55

theorem cost_of_pencil_and_pen : p + q = 1.25 :=
by
  sorry

end cost_of_pencil_and_pen_l106_106095


namespace braxton_total_earnings_l106_106928

-- Definitions of the given problem conditions
def students_ashwood : ℕ := 9
def days_ashwood : ℕ := 4
def students_braxton : ℕ := 6
def days_braxton : ℕ := 7
def students_cedar : ℕ := 8
def days_cedar : ℕ := 6

def total_payment : ℕ := 1080
def daily_wage_per_student : ℚ := total_payment / ((students_ashwood * days_ashwood) + 
                                                   (students_braxton * days_braxton) + 
                                                   (students_cedar * days_cedar))

-- The statement to be proven
theorem braxton_total_earnings :
  (students_braxton * days_braxton * daily_wage_per_student) = 360 := 
by
  sorry -- proof goes here

end braxton_total_earnings_l106_106928


namespace aluminum_foil_thickness_l106_106983

-- Define the variables and constants
variables (d l m w t : ℝ)

-- Define the conditions
def density_condition : Prop := d = m / (l * w * t)
def volume_formula : Prop := t = m / (d * l * w)

-- The theorem to prove
theorem aluminum_foil_thickness (h1 : density_condition d l m w t) : volume_formula d l m w t :=
sorry

end aluminum_foil_thickness_l106_106983


namespace empty_boxes_count_l106_106348

-- Definitions based on conditions:
def large_box_contains (B : Type) : ℕ := 1
def initial_small_boxes (B : Type) : ℕ := 10
def non_empty_boxes (B : Type) : ℕ := 6
def additional_smaller_boxes_in_non_empty (B : Type) (b : B) : ℕ := 10
def non_empty_small_boxes := 5

-- Proving that the number of empty boxes is 55 given the conditions:
theorem empty_boxes_count (B : Type) : 
  large_box_contains B = 1 ∧
  initial_small_boxes B = 10 ∧
  non_empty_boxes B = 6 ∧
  (∃ b : B, additional_smaller_boxes_in_non_empty B b = 10) →
  (initial_small_boxes B - non_empty_small_boxes + non_empty_small_boxes * additional_smaller_boxes_in_non_empty B) = 55 :=
by 
  sorry

end empty_boxes_count_l106_106348


namespace non_arithmetic_sequence_l106_106099

theorem non_arithmetic_sequence (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) :
    (∀ n, S_n n = n^2 + 2 * n - 1) →
    (∀ n, a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1)) →
    ¬(∀ d, ∀ n, a_n (n+1) = a_n n + d) :=
by
  intros hS ha
  sorry

end non_arithmetic_sequence_l106_106099


namespace sum_gcd_lcm_is_159_l106_106951

-- Definitions for GCD and LCM for specific values
def gcd_45_75 := Int.gcd 45 75
def lcm_48_18 := Int.lcm 48 18

-- The proof problem statement
theorem sum_gcd_lcm_is_159 : gcd_45_75 + lcm_48_18 = 159 := by
  sorry

end sum_gcd_lcm_is_159_l106_106951


namespace find_m_value_l106_106744

def quadratic_inequality_solution_set (a b c : ℝ) (m : ℝ) := {x : ℝ | 0 < x ∧ x < 2}

theorem find_m_value (a b c : ℝ) (m : ℝ) 
  (h1 : a = -1/2) 
  (h2 : b = 2) 
  (h3 : c = m) 
  (h4 : quadratic_inequality_solution_set a b c m = {x : ℝ | 0 < x ∧ x < 2}) : 
  m = 1 := 
sorry

end find_m_value_l106_106744


namespace billboards_and_road_length_l106_106599

theorem billboards_and_road_length :
  ∃ (x y : ℕ), 5 * (x + 21 - 1) = y ∧ (55 * (x - 1)) / 10 = y ∧ x = 200 ∧ y = 1100 :=
sorry

end billboards_and_road_length_l106_106599


namespace speed_of_mother_minimum_running_time_l106_106921

namespace XiaotongTravel

def distance_to_binjiang : ℝ := 4320
def time_diff : ℝ := 12
def speed_rate : ℝ := 1.2

theorem speed_of_mother : 
  ∃ (x : ℝ), (distance_to_binjiang / x - distance_to_binjiang / (speed_rate * x) = time_diff) → (speed_rate * x = 72) :=
sorry

def distance_to_company : ℝ := 2940
def running_speed : ℝ := 150
def total_time : ℝ := 30

theorem minimum_running_time :
  ∃ (y : ℝ), ((distance_to_company - running_speed * y) / 72 + y ≤ total_time) → (y ≥ 10) :=
sorry

end XiaotongTravel

end speed_of_mother_minimum_running_time_l106_106921


namespace coordinate_sum_l106_106168

theorem coordinate_sum (f : ℝ → ℝ) (x y : ℝ) (h₁ : f 9 = 7) (h₂ : 3 * y = f (3 * x) / 3 + 3) (h₃ : x = 3) : 
  x + y = 43 / 9 :=
by
  -- Proof goes here
  sorry

end coordinate_sum_l106_106168


namespace katie_travel_distance_l106_106755

theorem katie_travel_distance (d_train d_bus d_bike d_car d_total d1 d2 d3 : ℕ)
  (h1 : d_train = 162)
  (h2 : d_bus = 124)
  (h3 : d_bike = 88)
  (h4 : d_car = 224)
  (h_total : d_total = d_train + d_bus + d_bike + d_car)
  (h1_distance : d1 = 96)
  (h2_distance : d2 = 108)
  (h3_distance : d3 = 130)
  (h1_prob : 30 = 30)
  (h2_prob : 50 = 50)
  (h3_prob : 20 = 20) :
  (d_total + d1 = 694) ∧
  (d_total + d2 = 706) ∧
  (d_total + d3 = 728) :=
sorry

end katie_travel_distance_l106_106755


namespace number_of_terms_in_sequence_l106_106331

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem number_of_terms_in_sequence : 
  ∃ n : ℕ, arithmetic_sequence (-3) 4 n = 53 ∧ n = 15 :=
by
  use 15
  constructor
  · unfold arithmetic_sequence
    norm_num
  · norm_num

end number_of_terms_in_sequence_l106_106331


namespace player_c_wins_l106_106106

theorem player_c_wins :
  ∀ (A_wins A_losses B_wins B_losses C_losses C_wins : ℕ),
  A_wins = 4 →
  A_losses = 2 →
  B_wins = 3 →
  B_losses = 3 →
  C_losses = 3 →
  A_wins + B_wins + C_wins = A_losses + B_losses + C_losses →
  C_wins = 2 :=
by
  intros A_wins A_losses B_wins B_losses C_losses C_wins
  sorry

end player_c_wins_l106_106106


namespace sin_cos_product_l106_106720

theorem sin_cos_product (ϕ : ℝ) (h : Real.tan (ϕ + Real.pi / 4) = 5) : 
  1 / (Real.sin ϕ * Real.cos ϕ) = 13 / 6 :=
by
  sorry

end sin_cos_product_l106_106720


namespace Sandy_change_l106_106224

theorem Sandy_change (pants shirt sweater shoes total paid change : ℝ)
  (h1 : pants = 13.58) (h2 : shirt = 10.29) (h3 : sweater = 24.97) (h4 : shoes = 39.99) (h5 : total = pants + shirt + sweater + shoes) (h6 : paid = 100) (h7 : change = paid - total) :
  change = 11.17 := 
sorry

end Sandy_change_l106_106224


namespace solve_inequality_l106_106792

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ x ∈ set.Ioo (4 / 3 : ℝ) 2 ∪ set.Ioi 2 :=
by
  sorry

end solve_inequality_l106_106792


namespace prob_rain_both_days_correct_l106_106243

-- Definitions according to the conditions
def prob_rain_Saturday : ℝ := 0.4
def prob_rain_Sunday : ℝ := 0.3
def cond_prob_rain_Sunday_given_Saturday : ℝ := 0.5

-- Target probability to prove
def prob_rain_both_days : ℝ := prob_rain_Saturday * cond_prob_rain_Sunday_given_Saturday

-- Theorem statement
theorem prob_rain_both_days_correct : prob_rain_both_days = 0.2 :=
by
  sorry

end prob_rain_both_days_correct_l106_106243


namespace evaluate_expression_to_zero_l106_106925

-- Assuming 'm' is an integer with specific constraints and providing a proof that the expression evaluates to 0 when m = -1
theorem evaluate_expression_to_zero (m : ℤ) (h1 : -2 ≤ m) (h2 : m ≤ 2) (h3 : m ≠ 0) (h4 : m ≠ 1) (h5 : m ≠ 2) (h6 : m ≠ -2) : 
  (m = -1) → ((m / (m - 2) - 4 / (m ^ 2 - 2 * m)) / (m + 2) / (m ^ 2 - m)) = 0 := 
by
  intro hm_eq_neg1
  sorry

end evaluate_expression_to_zero_l106_106925


namespace determine_s_value_l106_106874

def f (x : ℚ) : ℚ := abs (x - 1) - abs x

def u : ℚ := f (5 / 16)
def v : ℚ := f u
def s : ℚ := f v

theorem determine_s_value : s = 1 / 2 :=
by
  -- Proof needed here
  sorry

end determine_s_value_l106_106874


namespace times_older_l106_106705

-- Conditions
variables (H S : ℕ)
axiom hold_age : H = 36
axiom hold_son_relation : H = 3 * S

-- Statement of the problem
theorem times_older (H S : ℕ) (h1 : H = 36) (h2 : H = 3 * S) : (H - 8) / (S - 8) = 7 :=
by
  -- Proof will be provided here
  sorry

end times_older_l106_106705


namespace no_solution_ineq_positive_exponents_l106_106970

theorem no_solution_ineq (m : ℝ) (h : m < 6) : ¬∃ x : ℝ, |x + 1| + |x - 5| ≤ m := 
sorry

theorem positive_exponents (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) : a^a * b^b - a^b * b^a > 0 := 
sorry

end no_solution_ineq_positive_exponents_l106_106970


namespace number_with_20_multiples_l106_106806

theorem number_with_20_multiples : ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k) → (k ≤ 100) → (n ∣ k) → (k / n ≤ 20) ) ∧ n = 5 := 
  sorry

end number_with_20_multiples_l106_106806


namespace geometric_sequence_a_11_l106_106059

-- Define the geometric sequence with given terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

axiom a_5 : a 5 = -16
axiom a_8 : a 8 = 8

-- Question to prove
theorem geometric_sequence_a_11 (h : is_geometric_sequence a q) : a 11 = -4 := 
sorry

end geometric_sequence_a_11_l106_106059


namespace subsequence_sum_q_l106_106610

theorem subsequence_sum_q (S : Fin 1995 → ℕ) (m : ℕ) (hS_pos : ∀ i : Fin 1995, 0 < S i)
  (hS_sum : (Finset.univ : Finset (Fin 1995)).sum S = m) (h_m_lt : m < 3990) :
  ∀ q : ℕ, 1 ≤ q → q ≤ m → ∃ (I : Finset (Fin 1995)), I.sum S = q := 
sorry

end subsequence_sum_q_l106_106610


namespace stars_total_is_correct_l106_106103

-- Define the given conditions
def number_of_stars_per_student : ℕ := 6
def number_of_students : ℕ := 210

-- Define total number of stars calculation
def total_number_of_stars : ℕ := number_of_stars_per_student * number_of_students

-- Proof statement that the total number of stars is correct
theorem stars_total_is_correct : total_number_of_stars = 1260 := by
  sorry

end stars_total_is_correct_l106_106103


namespace magical_stack_130_cards_l106_106373

theorem magical_stack_130_cards (n : ℕ) (h1 : 2 * n > 0) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ 2 * (n - k + 1) = 131 ∨ 
                                   (n + 1) ≤ k ∧ k ≤ 2 * n ∧ 2 * k - 1 = 131) : 2 * n = 130 :=
by
  sorry

end magical_stack_130_cards_l106_106373


namespace maximum_x_plus_2y_l106_106030

theorem maximum_x_plus_2y 
  (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^2 + 8 * y^2 + x * y = 2) :
  x + 2 * y ≤ 4 / 3 :=
sorry

end maximum_x_plus_2y_l106_106030


namespace f_1_geq_25_l106_106378

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- State that f is increasing on the interval [-2, +∞)
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m

-- Prove that given the function is increasing on [-2, +∞),
-- then f(1) is at least 25.
theorem f_1_geq_25 (m : ℝ) (h : is_increasing_on_interval m) : f 1 m ≥ 25 :=
  sorry

end f_1_geq_25_l106_106378


namespace paco_ate_more_sweet_than_salty_l106_106219

theorem paco_ate_more_sweet_than_salty (s t : ℕ) (h_s : s = 5) (h_t : t = 2) : s - t = 3 :=
by
  sorry

end paco_ate_more_sweet_than_salty_l106_106219


namespace jim_gas_tank_capacity_l106_106201

/-- Jim has 2/3 of a tank left after a round-trip of 20 miles where he gets 5 miles per gallon.
    Prove that the capacity of Jim's gas tank is 12 gallons. --/
theorem jim_gas_tank_capacity
    (remaining_fraction : ℚ)
    (round_trip_distance : ℚ)
    (fuel_efficiency : ℚ)
    (used_fraction : ℚ)
    (used_gallons : ℚ)
    (total_capacity : ℚ)
    (h1 : remaining_fraction = 2/3)
    (h2 : round_trip_distance = 20)
    (h3 : fuel_efficiency = 5)
    (h4 : used_fraction = 1 - remaining_fraction)
    (h5 : used_gallons = round_trip_distance / fuel_efficiency)
    (h6 : used_gallons = used_fraction * total_capacity) :
  total_capacity = 12 :=
sorry

end jim_gas_tank_capacity_l106_106201


namespace find_g_two_l106_106630

variable (g : ℝ → ℝ)

-- Condition 1: Functional equation
axiom g_eq : ∀ x y : ℝ, g (x - y) = g x * g y

-- Condition 2: Non-zero property
axiom g_ne_zero : ∀ x : ℝ, g x ≠ 0

-- Proof statement
theorem find_g_two : g 2 = 1 := 
by sorry

end find_g_two_l106_106630


namespace amy_bob_games_count_l106_106852

def crestwood_three_square (players : Finset ℕ) (Amy Bob : ℕ) : Prop :=
  ∃ (game : Finset (Finset ℕ)), 
    game.card = 3 ∧ 
    (∀ (p q r : ℕ), {p, q, r} ∈ game → p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧
    (∀ player ∈ players, ∃! g ∈ game, player ∈ g) ∧ 
    (Amy ∈ players ∧ Bob ∈ players)

theorem amy_bob_games_count
  (players : Finset ℕ) (Amy Bob : ℕ) (semester_games : Finset (Finset ℕ))
  (h9 : players.card = 9) 
  (h_game_split : ∀ (day_games : Finset (Finset ℕ)), day_games.card = 3 ∧ 
    (∀ (game : Finset ℕ), game ∈ day_games → game.card = 3) ∧ 
    (∀ player ∈ players, ∃! game ∈ day_games, player ∈ game))
  (h_once : ∀ (g : Finset ℕ), g.card = 3 → g ∈ semester_games) : 
  (∃! (game : Finset ℕ), Amy ∈ game ∧ Bob ∈ game) → 
  semester_games.filter (λ g, Amy ∈ g ∧ Bob ∈ g).card = 7 := by
sorry

end amy_bob_games_count_l106_106852


namespace find_integer_m_l106_106910

theorem find_integer_m 
  (m : ℤ) (h_pos : m > 0) 
  (h_intersect : ∃ (x y : ℤ), 17 * x + 7 * y = 1000 ∧ y = m * x + 2) : 
  m = 68 :=
by
  sorry

end find_integer_m_l106_106910


namespace trisect_angle_l106_106082

noncomputable def can_trisect_with_ruler_and_compasses (n : ℕ) : Prop :=
  ¬(3 ∣ n) → ∃ a b : ℤ, 3 * a + n * b = 1

theorem trisect_angle (n : ℕ) (h : ¬(3 ∣ n)) :
  can_trisect_with_ruler_and_compasses n :=
sorry

end trisect_angle_l106_106082


namespace race_distance_l106_106942

theorem race_distance (a b c : ℝ) (s_A s_B s_C : ℝ) :
  s_A * a = 100 → 
  s_B * a = 95 → 
  s_C * a = 90 → 
  s_B = s_A - 5 → 
  s_C = s_A - 10 → 
  s_C * (s_B / s_A) = 100 → 
  (100 - s_C) = 5 * (5 / 19) :=
sorry

end race_distance_l106_106942


namespace expression_square_l106_106923

theorem expression_square (a b c d : ℝ) :
  (2*a + b + 2*c - d)^2 - (3*a + 2*b + 3*c - 2*d)^2 - (4*a + 3*b + 4*c - 3*d)^2 + (5*a + 4*b + 5*c - 4*d)^2 =
  (2*(a + b + c - d))^2 := 
sorry

end expression_square_l106_106923


namespace perfect_squares_count_l106_106739

theorem perfect_squares_count : 
  let n_min := Nat.ceil (sqrt 50)
  let n_max := Nat.floor (sqrt 1000)
  n_min = 8 ∧ n_max = 31 → (n_max - n_min + 1 = 24) :=
begin
  intros,
  -- step a, nonnegative integer sqrt
  have h_n_min := Nat.ceil_spec (sqrt 50),
  have h_n_max := Nat.floor_spec (sqrt 1000),
  -- step b, we can prove the floors by direct calculation
  -- n_min = 8 and n_max = 31 must be true
  have : n_min = 8 := by linarith only [Nat.ceil (sqrt 50)],
  have : n_max = 31 := by linarith only [Nat.floor (sqrt 1000)],
  exact sorry -- Proof of main statement, assuming correct bounds give 24
end

end perfect_squares_count_l106_106739


namespace solution_set_of_quadratic_l106_106734

theorem solution_set_of_quadratic (a b x : ℝ) (h1 : a = 5) (h2 : b = -6) :
  (2 ≤ x ∧ x ≤ 3) → (bx^2 - ax - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end solution_set_of_quadratic_l106_106734


namespace quadratic_has_single_solution_l106_106423

theorem quadratic_has_single_solution (q : ℚ) (h : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 16 * x + 9 = 0 → q = 64 / 9) := by
  sorry

end quadratic_has_single_solution_l106_106423


namespace solve_inequality_l106_106787

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l106_106787


namespace jon_toaster_total_cost_l106_106525

def total_cost_toaster (MSRP : ℝ) (std_ins_pct : ℝ) (premium_upgrade_cost : ℝ) (state_tax_pct : ℝ) (environmental_fee : ℝ) : ℝ :=
  let std_ins_cost := std_ins_pct * MSRP
  let premium_ins_cost := std_ins_cost + premium_upgrade_cost
  let subtotal_before_tax := MSRP + premium_ins_cost
  let state_tax := state_tax_pct * subtotal_before_tax
  let total_before_env_fee := subtotal_before_tax + state_tax
  total_before_env_fee + environmental_fee

theorem jon_toaster_total_cost :
  total_cost_toaster 30 0.2 7 0.5 5 = 69.5 :=
by
  sorry

end jon_toaster_total_cost_l106_106525


namespace rat_to_chihuahua_ratio_is_six_to_one_l106_106144

noncomputable def chihuahuas_thought_to_be : ℕ := 70
noncomputable def actual_rats : ℕ := 60

theorem rat_to_chihuahua_ratio_is_six_to_one
    (h : chihuahuas_thought_to_be - actual_rats = 10) :
    actual_rats / (chihuahuas_thought_to_be - actual_rats) = 6 :=
by
  sorry

end rat_to_chihuahua_ratio_is_six_to_one_l106_106144


namespace find_a_in_triangle_l106_106890

theorem find_a_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : c = 3)
  (h2 : C = Real.pi / 3)
  (h3 : Real.sin B = 2 * Real.sin A)
  (h4 : a = 3) :
  a = Real.sqrt 3 := by
  sorry

end find_a_in_triangle_l106_106890


namespace trig_identity_l106_106590

open Real

theorem trig_identity (α : ℝ) (hα : α > -π ∧ α < -π/2) :
  (sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α))) = - 2 / tan α :=
by
  sorry

end trig_identity_l106_106590


namespace sam_remaining_money_l106_106222

def cost_of_candy_bars (num_candies cost_per_candy: nat) : nat := num_candies * cost_per_candy
def remaining_dimes (initial_dimes cost_in_dimes: nat) : nat := initial_dimes - cost_in_dimes
def remaining_quarters (initial_quarters cost_in_quarters: nat) : nat := initial_quarters - cost_in_quarters
def total_money_in_cents (dimes quarters: nat) : nat := (dimes * 10) + (quarters * 25)

theorem sam_remaining_money : 
  let initial_dimes := 19 in
  let initial_quarters := 6 in
  let num_candy_bars := 4 in
  let cost_per_candy := 3 in
  let cost_of_lollipop := 1 in
  let dimes_left := remaining_dimes initial_dimes (cost_of_candy_bars num_candy_bars cost_per_candy) in
  let quarters_left := remaining_quarters initial_quarters cost_of_lollipop in
  total_money_in_cents dimes_left quarters_left = 195 :=
by
  sorry

end sam_remaining_money_l106_106222


namespace min_ab_is_2sqrt6_l106_106721

noncomputable def min_ab (a b : ℝ) : ℝ :=
  if h : (a > 0) ∧ (b > 0) ∧ ((2 / a) + (3 / b) = Real.sqrt (a * b)) then
      2 * Real.sqrt 6
  else
      0 -- or any other value, since this case should not occur in the context

theorem min_ab_is_2sqrt6 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : (2 / a) + (3 / b) = Real.sqrt (a * b)) :
  min_ab a b = 2 * Real.sqrt 6 := 
by
  sorry

end min_ab_is_2sqrt6_l106_106721


namespace problem_lean_l106_106436

theorem problem_lean (x y : ℝ) (h₁ : (|x + 2| ≥ 0) ∧ (|y - 4| ≥ 0)) : 
  (|x + 2| = 0 ∧ |y - 4| = 0) → x + y - 3 = -1 :=
by sorry

end problem_lean_l106_106436


namespace brother_books_total_l106_106987

-- Define the conditions
def sarah_paperbacks : ℕ := 6
def sarah_hardbacks : ℕ := 4
def brother_paperbacks : ℕ := sarah_paperbacks / 3
def brother_hardbacks : ℕ := 2 * sarah_hardbacks

-- Define the statement to be proven
theorem brother_books_total : brother_paperbacks + brother_hardbacks = 10 :=
by
  -- Proof will be added here
  sorry

end brother_books_total_l106_106987


namespace original_team_size_l106_106504

theorem original_team_size (n : ℕ) (W : ℕ) :
  (W = n * 94) →
  ((W + 110 + 60) / (n + 2) = 92) →
  n = 7 :=
by
  intro hW_avg hnew_avg
  -- The proof steps would go here
  sorry

end original_team_size_l106_106504


namespace items_sold_increase_by_20_percent_l106_106972

-- Assume initial variables P (price per item without discount) and N (number of items sold without discount)
variables (P N : ℝ)

-- Define the conditions and the final proof goal
theorem items_sold_increase_by_20_percent 
  (h1 : ∀ (P N : ℝ), P > 0 → N > 0 → (P * N > 0))
  (h2 : ∀ (P : ℝ), P' = P * 0.90)
  (h3 : ∀ (P' N' : ℝ), P' * N' = P * N * 1.08)
  : (N' - N) / N * 100 = 20 := 
sorry

end items_sold_increase_by_20_percent_l106_106972


namespace part1_l106_106120

theorem part1 (a b c : ℤ) (h : a + b + c = 0) : a^3 + a^2 * c - a * b * c + b^2 * c + b^3 = 0 := 
sorry

end part1_l106_106120


namespace concyclic_iff_angle_equality_l106_106302

open EuclideanGeometry

-- Definitions based on conditions
variables {A B C D E P Q : Point}
variables (ABC : Triangle A B C)
variable (M : Point) (hM_mid : M = midpoint A B)
variable (hP_in_tri : P ∈ interior ABC)
variable (hQ : Q = reflection P M)
variable (D_int : D = line_intersect (line_through A P) (line_through B C))
variable (E_int : E = line_intersect (line_through B P) (line_through A C))

-- The theorem statement
theorem concyclic_iff_angle_equality :
  CyclicQuadrilateral A B D E ↔ ∠ A C P = ∠ Q C B :=
sorry

end concyclic_iff_angle_equality_l106_106302


namespace difference_two_numbers_l106_106247

theorem difference_two_numbers (a b : ℕ) (h₁ : a + b = 20250) (h₂ : b % 15 = 0) (h₃ : a = b / 3) : b - a = 10130 :=
by 
  sorry

end difference_two_numbers_l106_106247


namespace find_whole_number_l106_106863

theorem find_whole_number (N : ℕ) : 9.25 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 9.75 → N = 38 := by
  intros h
  have hN : 37 < (N : ℝ) ∧ (N : ℝ) < 39 := by
    -- This part follows directly from multiplying the inequality by 4.
    sorry

  -- Convert to integer comparison
  have h1 : 38 ≤ N := by
    -- Since 37 < N, N must be at least 38 as N is an integer.
    sorry
    
  have h2 : N < 39 := by
    sorry

  -- Conclude that N = 38 as it is the single whole number within the range.
  sorry

end find_whole_number_l106_106863


namespace domain_f₁_range_f₂_l106_106926

noncomputable def f₁ (x : ℝ) : ℝ := (x - 2)^0 / Real.sqrt (x + 1)
noncomputable def f₂ (x : ℝ) : ℝ := 2 * x - Real.sqrt (x - 1)

theorem domain_f₁ : ∀ x : ℝ, x > -1 ∧ x ≠ 2 → ∃ y : ℝ, y = f₁ x :=
by
  sorry

theorem range_f₂ : ∀ y : ℝ, y ≥ 15 / 8 → ∃ x : ℝ, y = f₂ x :=
by
  sorry

end domain_f₁_range_f₂_l106_106926


namespace find_power_of_7_l106_106716

theorem find_power_of_7 (x : ℕ) :
  ∀ (total_prime_factors : ℕ),
    total_prime_factors = 29 →
    x = total_prime_factors - (22 + 2) →
    x = 5 :=
by
  intros total_prime_factors total_pf_eq x_eq
  have pf_4_11 := 22  -- Number of prime factors from (4)^{11} = (2^2)^{11} = 2^{22}
  have pf_11_2 := 2  -- Number of prime factors from (11)^2 = 11^2
  have pf_total := pf_4_11 + pf_11_2 -- Total prime factors of (4)^{11} and (11)^2
  rw total_pf_eq at x_eq
  calc
    29 - pf_total = 29 - 24 := by sorry
    29 - 24 = 5 := by sorry

end find_power_of_7_l106_106716


namespace how_many_both_books_l106_106365

-- Definitions based on the conditions
def total_workers : ℕ := 40
def saramago_workers : ℕ := total_workers / 4
def kureishi_workers : ℕ := (total_workers * 5) / 8
def both_books (B : ℕ) : Prop :=
  B + (saramago_workers - B) + (kureishi_workers - B) + (9 - B) = total_workers

theorem how_many_both_books : ∃ B : ℕ, both_books B ∧ B = 4 := by
  use 4
  -- Proof goes here, skipped by using sorry
  sorry

end how_many_both_books_l106_106365


namespace total_money_received_l106_106638

-- Define the given prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def adult_tickets_sold : ℕ := 90
def child_tickets_sold : ℕ := 40

-- Define the theorem to prove the total amount received
theorem total_money_received :
  (adult_ticket_price * adult_tickets_sold + child_ticket_price * child_tickets_sold) = 1240 :=
by
  -- Proof goes here
  sorry

end total_money_received_l106_106638


namespace blue_ball_weight_l106_106625

variable (b t x : ℝ)
variable (c1 : b = 3.12)
variable (c2 : t = 9.12)
variable (c3 : t = b + x)

theorem blue_ball_weight : x = 6 :=
by
  sorry

end blue_ball_weight_l106_106625


namespace max_fourth_term_l106_106939

open Nat

/-- Constants representing the properties of the arithmetic sequence -/
axiom a : ℕ
axiom d : ℕ
axiom pos1 : a > 0
axiom pos2 : a + d > 0
axiom pos3 : a + 2 * d > 0
axiom pos4 : a + 3 * d > 0
axiom pos5 : a + 4 * d > 0
axiom sum_condition : 5 * a + 10 * d = 75

/-- Theorem stating the maximum fourth term of the arithmetic sequence -/
theorem max_fourth_term : a + 3 * d = 22 := sorry

end max_fourth_term_l106_106939


namespace max_n_for_factorable_poly_l106_106697

/-- 
  Let p(x) = 6x^2 + n * x + 48 be a quadratic polynomial.
  We want to find the maximum value of n such that p(x) can be factored into
  the product of two linear factors with integer coefficients.
-/
theorem max_n_for_factorable_poly :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * B + A = n → A * B = 48) ∧ n = 289 := 
by
  sorry

end max_n_for_factorable_poly_l106_106697


namespace length_of_segment_NB_l106_106597

variable (L W x : ℝ)
variable (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W))

theorem length_of_segment_NB (L W x : ℝ) (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W)) : 
  x = 0.8 * L :=
by
  sorry

end length_of_segment_NB_l106_106597


namespace round_robin_tournament_l106_106805

theorem round_robin_tournament (n k : ℕ) (h : (n-2) * (n-3) = 2 * 3^k): n = 5 :=
sorry

end round_robin_tournament_l106_106805


namespace equation_line_AB_length_median_AM_equation_altitude_A_l106_106438

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

noncomputable def line (A B : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let k := slope A B
  (k, -1, k * (A.1 - B.1))

noncomputable def altitude (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let k := slope B C
  let perp_k := -1 / k
  (perp_k, -1, A.2 - perp_k * A.1)

/- Question 1 -/
theorem equation_line_AB :
  line (-1, 5) (-2, -1) = (6, -1, 11) := 
sorry

/- Question 2 -/
theorem length_median_AM :
  length (-1, 5) (midpoint (-2, -1) (4, 3)) = 2 * real.sqrt 5 := 
sorry

/- Question 3 -/
theorem equation_altitude_A :
  altitude (-1, 5) (-2, -1) (4, 3) = (1, 6, -22) := 
sorry

end equation_line_AB_length_median_AM_equation_altitude_A_l106_106438


namespace smallest_n_for_identity_l106_106562

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l106_106562


namespace triangle_angle_problem_l106_106598

open Real

-- Define degrees to radians conversion (if necessary)
noncomputable def degrees (d : ℝ) : ℝ := d * π / 180

-- Define the problem conditions and goal
theorem triangle_angle_problem
  (x y : ℝ)
  (h1 : degrees 3 * x + degrees y = degrees 90) :
  x = 18 ∧ y = 36 := by
  sorry

end triangle_angle_problem_l106_106598


namespace value_of_expression_l106_106962

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by
  sorry

end value_of_expression_l106_106962


namespace find_probabilities_l106_106660

theorem find_probabilities (p_1 p_3 : ℝ)
  (h1 : p_1 + 0.15 + p_3 + 0.25 + 0.35 = 1)
  (h2 : p_3 = 4 * p_1) :
  p_1 = 0.05 ∧ p_3 = 0.20 :=
by
  sorry

end find_probabilities_l106_106660


namespace new_tv_cost_l106_106216

/-
Mark bought his first TV which was 24 inches wide and 16 inches tall. It cost $672.
His new TV is 48 inches wide and 32 inches tall.
The first TV was $1 more expensive per square inch compared to his newest TV.
Prove that the cost of his new TV is $1152.
-/

theorem new_tv_cost :
  let width_first_tv := 24
  let height_first_tv := 16
  let cost_first_tv := 672
  let width_new_tv := 48
  let height_new_tv := 32
  let discount_per_square_inch := 1
  let area_first_tv := width_first_tv * height_first_tv
  let cost_per_square_inch_first_tv := cost_first_tv / area_first_tv
  let cost_per_square_inch_new_tv := cost_per_square_inch_first_tv - discount_per_square_inch
  let area_new_tv := width_new_tv * height_new_tv
  let cost_new_tv := cost_per_square_inch_new_tv * area_new_tv
  cost_new_tv = 1152 := by
  sorry

end new_tv_cost_l106_106216


namespace gray_region_area_l106_106991

noncomputable def area_gray_region : ℝ :=
  let area_rectangle := (12 - 4) * (12 - 4)
  let radius_c := 4
  let radius_d := 4
  let area_quarter_circle_c := 1/4 * Real.pi * radius_c^2
  let area_quarter_circle_d := 1/4 * Real.pi * radius_d^2
  let overlap_area := area_quarter_circle_c + area_quarter_circle_d
  area_rectangle - overlap_area

theorem gray_region_area :
  area_gray_region = 64 - 8 * Real.pi := by
  sorry

end gray_region_area_l106_106991


namespace proportional_division_middle_part_l106_106742

theorem proportional_division_middle_part : 
  ∃ x : ℕ, x = 8 ∧ 5 * x = 40 ∧ 3 * x + 5 * x + 7 * x = 120 := 
by
  sorry

end proportional_division_middle_part_l106_106742


namespace machines_finish_job_in_24_over_11_hours_l106_106918

theorem machines_finish_job_in_24_over_11_hours :
    let work_rate_A := 1 / 4
    let work_rate_B := 1 / 12
    let work_rate_C := 1 / 8
    let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
    (1 : ℝ) / combined_work_rate = 24 / 11 :=
by
  sorry

end machines_finish_job_in_24_over_11_hours_l106_106918


namespace minimum_x2_y2_z2_l106_106153

theorem minimum_x2_y2_z2 :
  ∀ x y z : ℝ, (x^3 + y^3 + z^3 - 3 * x * y * z = 1) → (∃ a b c : ℝ, a = x ∨ a = y ∨ a = z ∧ b = x ∨ b = y ∨ b = z ∧ c = x ∨ c = y ∨ a ≤ b ∨ a ≤ c ∧ b ≤ c) → (x^2 + y^2 + z^2 ≥ 1) :=
by
  sorry

end minimum_x2_y2_z2_l106_106153


namespace find_g_5_l106_106238

theorem find_g_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1) : g 5 = 8 :=
sorry

end find_g_5_l106_106238


namespace angle_B_eq_pi_div_3_l106_106303

variables {A B C : ℝ} {a b c : ℝ}

/-- Given an acute triangle ABC, where sides a, b, c are opposite the angles A, B, and C respectively, 
    and given the condition b cos C + sqrt 3 * b sin C = a + c, prove that B = π / 3. -/
theorem angle_B_eq_pi_div_3 
  (h : ∀ (A B C : ℝ), 0 < A ∧ A < π / 2  ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (cond : b * Real.cos C + Real.sqrt 3 * b * Real.sin C = a + c) :
  B = π / 3 := 
sorry

end angle_B_eq_pi_div_3_l106_106303


namespace batsman_average_after_25th_innings_l106_106125

theorem batsman_average_after_25th_innings :
  ∃ A : ℝ, 
    (∀ s : ℝ, s = 25 * A + 62.5 → 24 * A + 95 = s) →
    A + 2.5 = 35 :=
by
  sorry

end batsman_average_after_25th_innings_l106_106125


namespace jimmy_needs_4_packs_of_bread_l106_106466

theorem jimmy_needs_4_packs_of_bread
  (num_sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (slices_per_pack : ℕ)
  (initial_slices : ℕ)
  (h1 : num_sandwiches = 8)
  (h2 : slices_per_sandwich = 2)
  (h3 : slices_per_pack = 4)
  (h4 : initial_slices = 0) :
  (num_sandwiches * slices_per_sandwich) / slices_per_pack = 4 := by
  sorry

end jimmy_needs_4_packs_of_bread_l106_106466


namespace sum_of_coefficients_l106_106042

/-- If (2x - 1)^4 = a₄x^4 + a₃x^3 + a₂x^2 + a₁x + a₀, then the sum of the coefficients a₀ + a₁ + a₂ + a₃ + a₄ is 1. -/
theorem sum_of_coefficients :
  ∃ a₄ a₃ a₂ a₁ a₀ : ℝ, (2 * x - 1) ^ 4 = a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀ → 
  a₀ + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  sorry

end sum_of_coefficients_l106_106042


namespace even_numbers_average_18_l106_106094

variable (n : ℕ)
variable (avg : ℕ)

theorem even_numbers_average_18 (h : avg = 18) : n = 17 := 
    sorry

end even_numbers_average_18_l106_106094


namespace car_average_speed_l106_106126

def average_speed (speed1 speed2 : ℕ) (time1 time2 : ℕ) : ℕ := 
  (speed1 * time1 + speed2 * time2) / (time1 + time2)

theorem car_average_speed :
  average_speed 60 90 (1/3) (2/3) = 80 := 
by 
  sorry

end car_average_speed_l106_106126


namespace sin2alpha_cos2beta_l106_106433

variable (α β : ℝ)

-- Conditions
def tan_add_eq : Prop := Real.tan (α + β) = -3
def tan_sub_eq : Prop := Real.tan (α - β) = 2

-- Question
theorem sin2alpha_cos2beta (h1 : tan_add_eq α β) (h2 : tan_sub_eq α β) : 
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = -1 / 7 := 
  sorry

end sin2alpha_cos2beta_l106_106433


namespace second_discount_correct_l106_106495

noncomputable def second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : ℝ :=
  let first_discount_amount := first_discount / 100 * original_price
  let price_after_first_discount := original_price - first_discount_amount
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

theorem second_discount_correct :
  second_discount_percentage 510 12 381.48 = 15 :=
by
  sorry

end second_discount_correct_l106_106495


namespace eval_expression_l106_106542

theorem eval_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  show 2^3 * 2^4 = 128
  calc
    2^3 * 2^4 = 2^(3 + 4) : by rw [pow_add]
    ...      = 2^7       : by rfl
    ...      = 128       : by norm_num

end eval_expression_l106_106542


namespace prize_calculations_l106_106490

-- Definitions for the conditions
def total_prizes := 50
def first_prize_unit_price := 20
def second_prize_unit_price := 14
def third_prize_unit_price := 8
def num_second_prize (x : ℕ) := 3 * x - 2
def num_third_prize (x : ℕ) := total_prizes - x - num_second_prize x
def total_cost (x : ℕ) := first_prize_unit_price * x + second_prize_unit_price * num_second_prize x + third_prize_unit_price * num_third_prize x

-- Proof problem statement
theorem prize_calculations (x : ℕ) (h : num_second_prize x = 22) : 
  num_second_prize x = 3 * x - 2 ∧ 
  num_third_prize x = 52 - 4 * x ∧ 
  total_cost x = 30 * x + 388 ∧ 
  total_cost 8 = 628 :=
by
  sorry

end prize_calculations_l106_106490


namespace angle_difference_parallelogram_l106_106054

theorem angle_difference_parallelogram (A B : ℝ) (hA : A = 55) (h1 : A + B = 180) :
  B - A = 70 := 
by
  sorry

end angle_difference_parallelogram_l106_106054


namespace number_system_base_l106_106076

theorem number_system_base (a : ℕ) (h : 2 * a^2 + 5 * a + 3 = 136) : a = 7 := 
sorry

end number_system_base_l106_106076


namespace smallest_n_exists_l106_106608

theorem smallest_n_exists (n : ℤ) (r : ℝ) : 
  (∃ m : ℤ, m = (↑n + r) ^ 3 ∧ r > 0 ∧ r < 1 / 1000) ∧ n > 0 → n = 19 := 
by sorry

end smallest_n_exists_l106_106608


namespace roots_polynomial_value_l106_106471

theorem roots_polynomial_value (r s t : ℝ) (h₁ : r + s + t = 15) (h₂ : r * s + s * t + t * r = 25) (h₃ : r * s * t = 10) :
  (1 + r) * (1 + s) * (1 + t) = 51 :=
by
  sorry

end roots_polynomial_value_l106_106471


namespace find_a_l106_106210

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
    (h3 : 13 ∣ 53^2016 + a) : a = 12 := 
by 
  -- proof would be written here
  sorry

end find_a_l106_106210


namespace chord_length_l106_106240

theorem chord_length
  (x y : ℝ)
  (h_circle : (x-1)^2 + (y-2)^2 = 2)
  (h_line : 3*x - 4*y = 0) :
  ∃ L : ℝ, L = 2 :=
sorry

end chord_length_l106_106240


namespace ensure_two_of_each_kind_l106_106401

def tablets_A := 10
def tablets_B := 14
def least_number_of_tablets_to_ensure_two_of_each := 12

theorem ensure_two_of_each_kind 
  (total_A : ℕ) 
  (total_B : ℕ) 
  (extracted : ℕ) 
  (hA : total_A = tablets_A) 
  (hB : total_B = tablets_B)
  (hExtract : extracted = least_number_of_tablets_to_ensure_two_of_each) : 
  ∃ (extracted : ℕ), extracted = least_number_of_tablets_to_ensure_two_of_each ∧ extracted ≥ tablets_A + 2 := 
sorry

end ensure_two_of_each_kind_l106_106401


namespace rectangle_perimeter_l106_106829

theorem rectangle_perimeter :
  ∃ (a b : ℤ), a ≠ b ∧ a * b = 2 * (2 * a + 2 * b) ∧ 2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_l106_106829


namespace solve_for_a_l106_106299

variable (x y a : ℤ)
variable (hx : x = 1)
variable (hy : y = -3)
variable (eq : a * x - y = 1)
 
theorem solve_for_a : a = -2 := by
  -- Placeholder to satisfy the lean prover, no actual proof steps
  sorry

end solve_for_a_l106_106299


namespace red_blue_pencil_difference_l106_106385

theorem red_blue_pencil_difference :
  let total_pencils := 36
  let red_fraction := 5 / 9
  let blue_fraction := 5 / 12
  let red_pencils := red_fraction * total_pencils
  let blue_pencils := blue_fraction * total_pencils
  red_pencils - blue_pencils = 5 :=
by
  -- placeholder proof
  sorry

end red_blue_pencil_difference_l106_106385


namespace find_triangle_altitude_l106_106232

variable (A b h : ℝ)

theorem find_triangle_altitude (h_eq_40 :  A = 800 ∧ b = 40) : h = 40 :=
sorry

end find_triangle_altitude_l106_106232


namespace number_of_ways_to_choose_one_person_l106_106975

-- Definitions for the conditions
def people_using_first_method : ℕ := 3
def people_using_second_method : ℕ := 5

-- Definition of the total number of ways to choose one person
def total_ways_to_choose_one_person : ℕ :=
  people_using_first_method + people_using_second_method

-- Statement of the theorem to be proved
theorem number_of_ways_to_choose_one_person :
  total_ways_to_choose_one_person = 8 :=
by 
  sorry

end number_of_ways_to_choose_one_person_l106_106975


namespace range_of_a_div_b_l106_106432

theorem range_of_a_div_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : 2 < b ∧ b < 8) : 
  1 / 8 < a / b ∧ a / b < 2 :=
sorry

end range_of_a_div_b_l106_106432


namespace tax_diminished_percentage_l106_106248

theorem tax_diminished_percentage (T C : ℝ) (x : ℝ) (h : (T * (1 - x / 100)) * (C * 1.10) = T * C * 0.88) :
  x = 20 :=
sorry

end tax_diminished_percentage_l106_106248


namespace evaluate_expression_l106_106544

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l106_106544


namespace factorization_correct_l106_106809

theorem factorization_correct (m : ℤ) : m^2 - 1 = (m - 1) * (m + 1) :=
by {
  -- sorry, this is a place-holder for the proof.
  sorry
}

end factorization_correct_l106_106809


namespace find_coefficients_l106_106530

def polynomial (a b : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 3 - 3 * x ^ 2 + b * x - 7

theorem find_coefficients (a b : ℝ) :
  polynomial a b 2 = -17 ∧ polynomial a b (-1) = -11 → a = 0 ∧ b = -1 :=
by
  sorry

end find_coefficients_l106_106530


namespace customer_count_l106_106666

theorem customer_count :
  let initial_customers := 13
  let customers_after_first_leave := initial_customers - 5
  let customers_after_new_arrival := customers_after_first_leave + 4
  let customers_after_group_join := customers_after_new_arrival + 8
  let final_customers := customers_after_group_join - 6
  final_customers = 14 :=
by
  sorry

end customer_count_l106_106666


namespace sum_of_numbers_l106_106122

def a : ℝ := 217
def b : ℝ := 2.017
def c : ℝ := 0.217
def d : ℝ := 2.0017

theorem sum_of_numbers :
  a + b + c + d = 221.2357 :=
by
  sorry

end sum_of_numbers_l106_106122


namespace find_x_l106_106036

theorem find_x (U : Set ℕ) (A B : Set ℕ) (x : ℕ) 
  (hU : U = Set.univ)
  (hA : A = {1, 4, x})
  (hB : B = {1, x ^ 2})
  (h : compl A ⊂ compl B) :
  x = 0 ∨ x = 2 := 
by 
  sorry

end find_x_l106_106036


namespace fraction_simplification_l106_106810

theorem fraction_simplification :
    1 + (1 / (1 + (1 / (2 + (1 / 3))))) = 17 / 10 := by
  sorry

end fraction_simplification_l106_106810


namespace board_train_immediately_probability_l106_106730

-- Define conditions
def total_time : ℝ := 10
def favorable_time : ℝ := 1

-- Define the probability P(A) as favorable_time / total_time
noncomputable def probability_A : ℝ := favorable_time / total_time

-- State the proposition to prove that the probability is 1/10
theorem board_train_immediately_probability : probability_A = 1 / 10 :=
by sorry

end board_train_immediately_probability_l106_106730


namespace abs_fraction_inequality_solution_l106_106773

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l106_106773


namespace functional_eq_zero_function_l106_106612

theorem functional_eq_zero_function (f : ℝ → ℝ) (k : ℝ) (h : ∀ x y : ℝ, f (f x + f y + k * x * y) = x * f y + y * f x) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_function_l106_106612


namespace find_f_neg_two_l106_106735

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg_two (h : ∀ x : ℝ, x ≠ 0 → f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
by
  sorry

end find_f_neg_two_l106_106735


namespace primes_sum_eq_2001_l106_106334

/-- If a and b are prime numbers such that a^2 + b = 2003, then a + b = 2001. -/
theorem primes_sum_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) :
    a + b = 2001 := 
  sorry

end primes_sum_eq_2001_l106_106334


namespace zeros_distance_l106_106876

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3*x^2 + a

theorem zeros_distance (a x1 x2 : ℝ) 
  (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) (h_order: x1 < x2) : 
  x2 - x1 = 3 := 
sorry

end zeros_distance_l106_106876


namespace evaluate_expression_l106_106531

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l106_106531


namespace abs_fraction_inequality_solution_l106_106775

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l106_106775


namespace max_view_angle_exists_l106_106478

-- Definitions based on problem conditions
variables (O : Point) (A B : Point)
variable (α : Angle) -- representing the given acute angle with vertex O
variable (OC : Length) -- representing the distance from O to C

-- The final theorem statement
theorem max_view_angle_exists :
  ∃ C : Point, (C ≠ A ∧ C ≠ B ∧ ∠ACB = max_angle AB) ∧ (C lies_on_other_side O α) :=
sorry

end max_view_angle_exists_l106_106478


namespace average_of_five_digits_l106_106799

theorem average_of_five_digits 
  (S : ℝ)
  (S3 : ℝ)
  (h_avg8 : S / 8 = 20)
  (h_avg3 : S3 / 3 = 33.333333333333336) :
  (S - S3) / 5 = 12 := 
by
  sorry

end average_of_five_digits_l106_106799


namespace horner_operations_count_l106_106107

def polynomial (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

def horner_polynomial (x : ℝ) := (((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1)

theorem horner_operations_count (x : ℝ) : 
    (polynomial x = horner_polynomial x) → 
    (x = 2) → 
    (mul_ops : ℕ) = 5 → 
    (add_ops : ℕ) = 5 := 
by 
  sorry

end horner_operations_count_l106_106107


namespace ab_le_one_l106_106726

theorem ab_le_one {a b : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 2) : ab ≤ 1 :=
by
  sorry

end ab_le_one_l106_106726


namespace least_number_of_roots_l106_106661

variable (g : ℝ → ℝ) -- Declare the function g with domain ℝ and codomain ℝ

-- Define the conditions as assumptions.
variable (h1 : ∀ x : ℝ, g (3 + x) = g (3 - x))
variable (h2 : ∀ x : ℝ, g (8 + x) = g (8 - x))
variable (h3 : g 0 = 0)

-- State the theorem to prove the necessary number of roots.
theorem least_number_of_roots : ∀ a b : ℝ, a ≤ -2000 ∧ b ≥ 2000 → ∃ n ≥ 668, ∃ x : ℝ, g x = 0 ∧ a ≤ x ∧ x ≤ b :=
by
  -- To be filled in with the logic to prove the theorem.
  sorry

end least_number_of_roots_l106_106661


namespace purchasing_options_count_l106_106398

theorem purchasing_options_count : ∃ (s : Finset (ℕ × ℕ)), s.card = 4 ∧
  ∀ (a : ℕ × ℕ), a ∈ s ↔ 
    (80 * a.1 + 120 * a.2 = 1000) 
    ∧ (a.1 > 0) ∧ (a.2 > 0) :=
by
  sorry

end purchasing_options_count_l106_106398


namespace Jame_tears_30_cards_at_a_time_l106_106063

theorem Jame_tears_30_cards_at_a_time
    (cards_per_deck : ℕ)
    (times_per_week : ℕ)
    (decks : ℕ)
    (weeks : ℕ)
    (total_cards : ℕ := decks * cards_per_deck)
    (total_times : ℕ := weeks * times_per_week)
    (cards_at_a_time : ℕ := total_cards / total_times)
    (h1 : cards_per_deck = 55)
    (h2 : times_per_week = 3)
    (h3 : decks = 18)
    (h4 : weeks = 11) :
    cards_at_a_time = 30 := by
  -- Proof can be added here
  sorry

end Jame_tears_30_cards_at_a_time_l106_106063


namespace part1_expression_for_f_part2_three_solutions_l106_106430

noncomputable def f1 (x : ℝ) := x^2

noncomputable def f2 (x : ℝ) := 8 / x

noncomputable def f (x : ℝ) := f1 x + f2 x

theorem part1_expression_for_f : ∀ x:ℝ, f x = x^2 + 8 / x := by
  sorry  -- This is where the proof would go

theorem part2_three_solutions (a : ℝ) (h : a > 3) : 
  ∃ x1 x2 x3 : ℝ, f x1 = f a ∧ f x2 = f a ∧ f x3 = f a ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 := by
  sorry  -- This is where the proof would go

end part1_expression_for_f_part2_three_solutions_l106_106430


namespace find_C_when_F_10_l106_106450

theorem find_C_when_F_10 : (∃ C : ℚ, ∀ F : ℚ, F = 10 → F = (9 / 5 : ℚ) * C + 32 → C = -110 / 9) :=
by
  sorry

end find_C_when_F_10_l106_106450


namespace find_number_of_pencils_l106_106382

-- Define the conditions
def number_of_people : Nat := 6
def notebooks_per_person : Nat := 9
def number_of_notebooks : Nat := number_of_people * notebooks_per_person
def pencils_multiplier : Nat := 6
def number_of_pencils : Nat := pencils_multiplier * number_of_notebooks

-- Prove the main statement
theorem find_number_of_pencils : number_of_pencils = 324 :=
by
  sorry

end find_number_of_pencils_l106_106382


namespace product_xyz_l106_106745

theorem product_xyz (x y z : ℝ) (h1 : x = y) (h2 : x = 2 * z) (h3 : x = 7.999999999999999) :
    x * y * z = 255.9999999999998 := by
  sorry

end product_xyz_l106_106745


namespace older_brother_catches_up_l106_106058

-- Define the initial conditions and required functions
def younger_brother_steps_before_chase : ℕ := 10
def time_per_3_steps_older := 1  -- in seconds
def time_per_4_steps_younger := 1  -- in seconds 
def dist_older_in_5_steps : ℕ := 7  -- 7d_younger / 5
def dist_younger_in_7_steps : ℕ := 5
def speed_older : ℕ := 3 * dist_older_in_5_steps / 5  -- steps/second 
def speed_younger : ℕ := 4 * dist_younger_in_7_steps / 7  -- steps/second

theorem older_brother_catches_up : ∃ n : ℕ, n = 150 :=
by sorry  -- final theorem statement with proof omitted

end older_brother_catches_up_l106_106058


namespace calculate_expression_l106_106849

theorem calculate_expression :
  500 * 996 * 0.0996 * 20 + 5000 = 997016 :=
by
  sorry

end calculate_expression_l106_106849


namespace distance_between_truck_and_car_l106_106839

def truck_speed : ℝ := 65 -- km/h
def car_speed : ℝ := 85 -- km/h
def time_minutes : ℝ := 3 -- minutes
def time_hours : ℝ := time_minutes / 60 -- converting minutes to hours

def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem distance_between_truck_and_car :
  let truck_distance := distance_traveled truck_speed time_hours
  let car_distance := distance_traveled car_speed time_hours
  truck_distance - car_distance = -1 := -- the distance is 1 km but negative when subtracting truck from car
by {
  sorry
}

end distance_between_truck_and_car_l106_106839


namespace value_of_N_l106_106499

theorem value_of_N (a b c N : ℚ) 
  (h1 : a + b + c = 120)
  (h2 : a + 8 = N)
  (h3 : 8 * b = N)
  (h4 : c / 8 = N) :
  N = 960 / 73 :=
by
  sorry

end value_of_N_l106_106499


namespace auditorium_shared_days_l106_106801

theorem auditorium_shared_days :
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  Nat.lcm (Nat.lcm drama_club_days choir_days) debate_team_days = 105 :=
by
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  sorry

end auditorium_shared_days_l106_106801


namespace groom_age_proof_l106_106102

theorem groom_age_proof (G B : ℕ) (h1 : B = G + 19) (h2 : G + B = 185) : G = 83 :=
by
  sorry

end groom_age_proof_l106_106102


namespace roots_of_quadratic_l106_106318

theorem roots_of_quadratic :
  ∃ m n : ℝ, (∀ x : ℝ, x^2 - 4 * x - 1 = 0 → (x = m ∨ x = n)) ∧
            (m + n = 4) ∧
            (m * n = -1) ∧
            (m + n - m * n = 5) :=
by
  sorry

end roots_of_quadratic_l106_106318


namespace exists_quadratic_open_upwards_passing_through_origin_l106_106509

-- Define the general form of a quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Constants for the conditions
constants (a b c : ℝ)

-- Conditions
axiom a_pos : a > 0
axiom passes_through_origin : quadratic_function a b c 0 = 1

-- Goal: Prove that under the given conditions, the quadratic function exists (and hence provide an example of such function).
theorem exists_quadratic_open_upwards_passing_through_origin : 
  ∃ (a b c : ℝ), a > 0 ∧ quadratic_function a b c 0 = 1 :=
by
  use 1, 0, 1
  split
  { exact zero_lt_one }
  { simp [quadratic_function] }

end exists_quadratic_open_upwards_passing_through_origin_l106_106509


namespace nests_count_l106_106388

theorem nests_count (birds nests : ℕ) (h1 : birds = 6) (h2 : birds - nests = 3) : nests = 3 := by
  sorry

end nests_count_l106_106388


namespace dave_coins_l106_106288

theorem dave_coins :
  ∃ n : ℕ, n ≡ 2 [MOD 7] ∧ n ≡ 3 [MOD 5] ∧ n ≡ 1 [MOD 3] ∧ n = 58 :=
sorry

end dave_coins_l106_106288


namespace distance_after_3_minutes_l106_106837

-- Define the given speeds and time interval
def speed_truck : ℝ := 65 -- in km/h
def speed_car : ℝ := 85 -- in km/h
def time_minutes : ℝ := 3 -- in minutes

-- The equivalent time in hours
def time_hours : ℝ := time_minutes / 60

-- Calculate the distances travelled by the truck and the car
def distance_truck : ℝ := speed_truck * time_hours
def distance_car : ℝ := speed_car * time_hours

-- Define the distance between the truck and the car
def distance_between : ℝ := distance_car - distance_truck

-- Theorem: The distance between the truck and car after 3 minutes is 1 km.
theorem distance_after_3_minutes : distance_between = 1 := by
  sorry

end distance_after_3_minutes_l106_106837


namespace find_c_plus_d_l106_106379

theorem find_c_plus_d (c d : ℝ) :
  (∀ x y, (x = (1 / 3) * y + c) → (y = (1 / 3) * x + d) → (x, y) = (3, 3)) → 
  c + d = 4 :=
by
  -- ahead declaration to meet the context requirements in Lean 4
  intros h
  -- Proof steps would go here, but they are omitted
  sorry

end find_c_plus_d_l106_106379


namespace pseudo_symmetry_abscissa_l106_106215

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4 * Real.log x

theorem pseudo_symmetry_abscissa :
  ∃ x0 : ℝ, x0 = Real.sqrt 2 ∧
    (∀ x : ℝ, x ≠ x0 → (f x - ((2*x0 + 4/x0 - 6)*(x - x0) + x0^2 - 6*x0 + 4*Real.log x0)) / (x - x0) > 0) :=
sorry

end pseudo_symmetry_abscissa_l106_106215


namespace penny_dime_halfdollar_same_probability_l106_106932

def probability_same_penny_dime_halfdollar : ℚ :=
  let total_outcomes := 2 ^ 5
  let successful_outcomes := 2 * 2 * 2
  successful_outcomes / total_outcomes

theorem penny_dime_halfdollar_same_probability :
  probability_same_penny_dime_halfdollar = 1 / 4 :=
by 
  sorry

end penny_dime_halfdollar_same_probability_l106_106932


namespace harmonic_mean_closest_to_2_l106_106996

theorem harmonic_mean_closest_to_2 (a : ℝ) (b : ℝ) (h₁ : a = 1) (h₂ : b = 4032) : 
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  rw [h₁, h₂]
  -- The rest of the proof follows from here, skipped with sorry
  sorry

end harmonic_mean_closest_to_2_l106_106996


namespace area_ratio_l106_106061

-- Definitions for the conditions in the problem
variables (PQ QR RP : ℝ) (p q r : ℝ)

-- Conditions
def pq_condition := PQ = 18
def qr_condition := QR = 24
def rp_condition := RP = 30
def pqr_sum := p + q + r = 3 / 4
def pqr_squaresum := p^2 + q^2 + r^2 = 1 / 2

-- Goal statement that the area ratio of triangles XYZ to PQR is 23/32
theorem area_ratio (h1 : PQ = 18) (h2 : QR = 24) (h3 : RP = 30) 
  (h4 : p + q + r = 3 / 4) (h5 : p^2 + q^2 + r^2 = 1 / 2) : 
  ∃ (m n : ℕ), (m + n = 55) ∧ (m / n = 23 / 32) := 
sorry

end area_ratio_l106_106061


namespace xiaoming_bus_time_l106_106118

-- Definitions derived from the conditions:
def total_time : ℕ := 40
def transfer_time : ℕ := 6
def subway_time : ℕ := 30
def bus_time : ℕ := 50

-- Theorem statement to prove the bus travel time equals 10 minutes
theorem xiaoming_bus_time : (total_time - transfer_time = 34) ∧ (subway_time = 30 ∧ bus_time = 50) → 
  ∃ (T_bus : ℕ), T_bus = 10 := by
  sorry

end xiaoming_bus_time_l106_106118


namespace triangle_altitude_l106_106234

theorem triangle_altitude (b : ℕ) (h : ℕ) (area : ℕ) (h_area : area = 800) (h_base : b = 40) (h_formula : area = (1 / 2) * b * h) : h = 40 :=
by
  sorry

end triangle_altitude_l106_106234


namespace min_questions_any_three_cards_min_questions_consecutive_three_cards_l106_106621

-- Definitions for numbers on cards and necessary questions
variables (n : ℕ) (h_n : n > 3)
  (cards : Fin n → ℤ)
  (h_cards_range : ∀ i, cards i = 1 ∨ cards i = -1)

-- Case (a): Product of any three cards
theorem min_questions_any_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (∃ (k : ℕ), n = 3 * k + 1 ∧ p = k + 1) ∨
  (∃ (k : ℕ), n = 3 * k + 2 ∧ p = k + 2) :=
sorry
  
-- Case (b): Product of any three consecutive cards
theorem min_questions_consecutive_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (¬(∃ (k : ℕ), n = 3 * k) ∧ p = n) :=
sorry

end min_questions_any_three_cards_min_questions_consecutive_three_cards_l106_106621


namespace largest_4digit_congruent_17_mod_28_l106_106646

theorem largest_4digit_congruent_17_mod_28 :
  ∃ n, n < 10000 ∧ n % 28 = 17 ∧ ∀ m, m < 10000 ∧ m % 28 = 17 → m ≤ 9982 :=
by
  sorry

end largest_4digit_congruent_17_mod_28_l106_106646


namespace probability_two_heads_in_succession_in_10_tosses_l106_106403

theorem probability_two_heads_in_succession_in_10_tosses : 
  let g : ℕ → ℕ := λ n, (Nat.fib (n + 2)) in
  (prob := 1 - (g 10) / 1024) ∧ prob = 55 / 64 :=
by
  let g := λ n, Nat.fib (n + 2)
  have g_10 : g 10 = 144 := by sorry
  have total_sequences : 2^10 = 1024 := by norm_num
  have prob := 1 - (g 10 : ℚ) / total_sequences
  have h : prob = 55 / 64 := by sorry
  exact prob, h

end probability_two_heads_in_succession_in_10_tosses_l106_106403


namespace scatter_plot_exists_l106_106295

theorem scatter_plot_exists (sample_data : List (ℝ × ℝ)) :
  ∃ plot : List (ℝ × ℝ), plot = sample_data :=
by
  sorry

end scatter_plot_exists_l106_106295


namespace distance_traveled_by_car_l106_106835

theorem distance_traveled_by_car :
  let total_distance := 90
  let distance_by_foot := (1 / 5 : ℝ) * total_distance
  let distance_by_bus := (2 / 3 : ℝ) * total_distance
  let distance_by_car := total_distance - (distance_by_foot + distance_by_bus)
  distance_by_car = 12 :=
by
  sorry

end distance_traveled_by_car_l106_106835


namespace problem_solution_l106_106090

variables (p q : Prop)

theorem problem_solution (h1 : ¬ (p ∧ q)) (h2 : p ∨ q) : ¬ p ∨ ¬ q := by
  sorry

end problem_solution_l106_106090


namespace find_larger_number_l106_106236

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end find_larger_number_l106_106236


namespace ratio_EG_FH_l106_106480

theorem ratio_EG_FH (EF FG EH : ℝ) (hEF : EF = 3) (hFG : FG = 7) (hEH : EH = 20) :
  (EF + FG) / (EH - EF) = 10 / 17 :=
by
  sorry

end ratio_EG_FH_l106_106480


namespace minimize_d_and_distance_l106_106914

-- Define point and geometric shapes
structure Point :=
  (x : ℝ)
  (y : ℝ)

def Parabola (P : Point) : Prop := P.x^2 = 4 * P.y
def Circle (P1 : Point) : Prop := (P1.x - 2)^2 + (P1.y + 1)^2 = 1

-- Define the point P and point P1
variable (P : Point)
variable (P1 : Point)

-- Condition: P is on the parabola
axiom on_parabola : Parabola P

-- Condition: P1 is on the circle
axiom on_circle : Circle P1

-- Theorem: coordinates of P when the function d + distance(P, P1) is minimized
theorem minimize_d_and_distance :
  P = { x := 2 * Real.sqrt 2 - 2, y := 3 - 2 * Real.sqrt 2 } :=
sorry

end minimize_d_and_distance_l106_106914


namespace industrial_park_investment_l106_106841

noncomputable def investment_in_projects : Prop :=
  ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500

theorem industrial_park_investment :
  investment_in_projects :=
by
  have h : ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500 := 
    sorry
  exact h

end industrial_park_investment_l106_106841


namespace arithmetic_mean_squares_l106_106485

theorem arithmetic_mean_squares (n : ℕ) (h : 0 < n) :
  let S_n2 := (n * (n + 1) * (2 * n + 1)) / 6 
  let A_n2 := S_n2 / n
  A_n2 = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end arithmetic_mean_squares_l106_106485


namespace sequence_eventually_periodic_modulo_l106_106696

noncomputable def a_n (n : ℕ) : ℕ :=
  n ^ n + (n - 1) ^ (n + 1)

theorem sequence_eventually_periodic_modulo (m : ℕ) (hm : m > 0) : ∃ K s : ℕ, ∀ k : ℕ, (K ≤ k → a_n (k) % m = a_n (k + s) % m) :=
sorry

end sequence_eventually_periodic_modulo_l106_106696


namespace sum_of_squares_l106_106767

theorem sum_of_squares (n : Nat) (h : n = 2005^2) : 
  ∃ a1 b1 a2 b2 a3 b3 a4 b4 : Int, 
    (n = a1^2 + b1^2 ∧ a1 ≠ 0 ∧ b1 ≠ 0) ∧ 
    (n = a2^2 + b2^2 ∧ a2 ≠ 0 ∧ b2 ≠ 0) ∧ 
    (n = a3^2 + b3^2 ∧ a3 ≠ 0 ∧ b3 ≠ 0) ∧ 
    (n = a4^2 + b4^2 ∧ a4 ≠ 0 ∧ b4 ≠ 0) ∧ 
    (a1, b1) ≠ (a2, b2) ∧ 
    (a1, b1) ≠ (a3, b3) ∧ 
    (a1, b1) ≠ (a4, b4) ∧ 
    (a2, b2) ≠ (a3, b3) ∧ 
    (a2, b2) ≠ (a4, b4) ∧ 
    (a3, b3) ≠ (a4, b4) :=
by
  sorry

end sum_of_squares_l106_106767


namespace total_handshakes_five_people_l106_106258

theorem total_handshakes_five_people : 
  let n := 5
  let total_handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2
  total_handshakes 5 = 10 :=
by sorry

end total_handshakes_five_people_l106_106258


namespace rashmi_late_time_is_10_l106_106924

open Real

noncomputable def rashmi_late_time : ℝ :=
  let d : ℝ := 9.999999999999993
  let v1 : ℝ := 5 / 60 -- km per minute
  let v2 : ℝ := 6 / 60 -- km per minute
  let time1 := d / v1 -- time taken at 5 kmph
  let time2 := d / v2 -- time taken at 6 kmph
  let difference := time1 - time2
  let T := difference / 2 -- The time she was late or early
  T

theorem rashmi_late_time_is_10 : rashmi_late_time = 10 := by
  simp [rashmi_late_time]
  sorry

end rashmi_late_time_is_10_l106_106924


namespace circle_center_eq_l106_106034

theorem circle_center_eq (x y : ℝ) :
    (x^2 + y^2 - 2*x + y + 1/4 = 0) → (x = 1 ∧ y = -1/2) :=
by
  sorry

end circle_center_eq_l106_106034


namespace cubicroots_expression_l106_106071

theorem cubicroots_expression (a b c : ℝ)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 11)
  (h₃ : a * b * c = 6) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 251 / 216 :=
by sorry

end cubicroots_expression_l106_106071


namespace minimize_sum_dist_l106_106026

noncomputable section

variables {Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ}

-- Conditions
def clusters (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) :=
  Q3 <= Q1 + Q2 + Q4 / 3 ∧ Q3 = (Q1 + 2 * Q2 + 2 * Q4) / 5 ∧
  Q7 <= Q5 + Q6 + Q8 / 3 ∧ Q7 = (Q5 + 2 * Q6 + 2 * Q8) / 5

-- Sum of distances function
def sum_dist (Q : ℝ) (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) : ℝ :=
  abs (Q - Q1) + abs (Q - Q2) + abs (Q - Q3) + abs (Q - Q4) +
  abs (Q - Q5) + abs (Q - Q6) + abs (Q - Q7) + abs (Q - Q8) + abs (Q - Q9)

-- Theorem
theorem minimize_sum_dist (h : clusters Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) :
  ∃ Q : ℝ, (∀ Q' : ℝ, sum_dist Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 ≤ sum_dist Q' Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) → Q = Q5 :=
sorry

end minimize_sum_dist_l106_106026


namespace quadratic_floor_eq_solutions_count_l106_106237

theorem quadratic_floor_eq_solutions_count : 
  ∃ s : Finset ℝ, (∀ x : ℝ, x^2 - 4 * ⌊x⌋ + 3 = 0 → x ∈ s) ∧ s.card = 3 :=
by 
  sorry

end quadratic_floor_eq_solutions_count_l106_106237


namespace area_of_square_field_l106_106230

-- Definitions
def cost_per_meter : ℝ := 1.40
def total_cost : ℝ := 932.40
def gate_width : ℝ := 1.0

-- Problem Statement
theorem area_of_square_field (s : ℝ) (A : ℝ) 
  (h1 : (4 * s - 2 * gate_width) * cost_per_meter = total_cost)
  (h2 : A = s^2) : A = 27889 := 
  sorry

end area_of_square_field_l106_106230


namespace find_a_l106_106067

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x / (3 * x + 4)

theorem find_a (a : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : (f a) (f a x) = x → a = -2 := by
  unfold f
  -- Remaining proof steps skipped
  sorry

end find_a_l106_106067


namespace circle_radius_l106_106963

theorem circle_radius (d : ℝ) (h : d = 10) : d / 2 = 5 :=
by
  sorry

end circle_radius_l106_106963


namespace smallest_b_for_quadratic_factors_l106_106553

theorem smallest_b_for_quadratic_factors :
  ∃ b : ℕ, (∀ r s : ℤ, (r * s = 1764 → r + s = b) → b = 84) :=
sorry

end smallest_b_for_quadratic_factors_l106_106553


namespace vanessa_phone_pictures_l106_106946

theorem vanessa_phone_pictures
  (C : ℕ) (P : ℕ) (hC : C = 7)
  (hAlbums : 5 * 6 = 30)
  (hTotal : 30 = P + C) :
  P = 23 := by
  sorry

end vanessa_phone_pictures_l106_106946


namespace find_divisor_l106_106341

theorem find_divisor {x y : ℤ} (h1 : (x - 5) / y = 7) (h2 : (x - 24) / 10 = 3) : y = 7 :=
by
  sorry

end find_divisor_l106_106341


namespace solution_set_a_eq_1_no_positive_a_for_all_x_l106_106878

-- Define the original inequality for a given a.
def inequality (a x : ℝ) : Prop := |a * x - 1| + |a * x - a| ≥ 2

-- Part 1: For a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | inequality 1 x } = {x : ℝ | x ≤ 0 ∨ x ≥ 2} :=
sorry

-- Part 2: There is no positive a such that the inequality holds for all x ∈ ℝ
theorem no_positive_a_for_all_x :
  ¬ ∃ a > 0, ∀ x : ℝ, inequality a x :=
sorry

end solution_set_a_eq_1_no_positive_a_for_all_x_l106_106878


namespace tim_score_l106_106390

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

theorem tim_score :
  (first_seven_primes.sum = 58) :=
by
  sorry

end tim_score_l106_106390


namespace f_3_equals_1000_l106_106435

-- Define the function property f(lg x) = x
axiom f : ℝ → ℝ
axiom lg : ℝ → ℝ -- log function
axiom f_property : ∀ x : ℝ, f (lg x) = x

-- Prove that f(3) = 10^3
theorem f_3_equals_1000 : f 3 = 10^3 :=
by 
  -- Sorry to skip the proof
  sorry

end f_3_equals_1000_l106_106435


namespace number_of_trees_planted_l106_106503

def current_trees : ℕ := 34
def final_trees : ℕ := 83
def planted_trees : ℕ := final_trees - current_trees

theorem number_of_trees_planted : planted_trees = 49 :=
by
  -- proof goes here, but it is skipped for now
  sorry

end number_of_trees_planted_l106_106503


namespace more_seventh_graders_than_sixth_graders_l106_106225

theorem more_seventh_graders_than_sixth_graders 
  (n m : ℕ)
  (H1 : ∀ x : ℕ, x = n → 7 * n = 6 * m) : 
  m > n := 
by
  -- Proof is not required and will be skipped with sorry.
  sorry

end more_seventh_graders_than_sixth_graders_l106_106225


namespace find_lines_and_intersections_l106_106580

-- Define the intersection point conditions
def intersection_point (m n : ℝ) : Prop :=
  (2 * m - n + 7 = 0) ∧ (m + n - 1 = 0)

-- Define the perpendicular line to l1 passing through (-2, 3)
def perpendicular_line_through_A (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

-- Define the parallel line to l passing through (-2, 3)
def parallel_line_through_A (x y : ℝ) : Prop :=
  2 * x - 3 * y + 13 = 0

-- main theorem
theorem find_lines_and_intersections :
  ∃ m n : ℝ, intersection_point m n ∧ m = -2 ∧ n = 3 ∧
  ∃ l3 : ℝ → ℝ → Prop, l3 = perpendicular_line_through_A ∧
  ∃ l4 : ℝ → ℝ → Prop, l4 = parallel_line_through_A :=
sorry

end find_lines_and_intersections_l106_106580


namespace parabola_point_distance_l106_106235

open Real

noncomputable def parabola_coords (y z: ℝ) : Prop :=
  y^2 = 12 * z

noncomputable def distance (x1 y1 x2 y2: ℝ) : ℝ :=
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem parabola_point_distance (x y: ℝ) :
  parabola_coords y x ∧ distance x y 3 0 = 9 ↔ ( x = 6 ∧ (y = 6 * sqrt 2 ∨ y = -6 * sqrt 2)) :=
by
  sorry

end parabola_point_distance_l106_106235


namespace evaluate_expression_l106_106858

theorem evaluate_expression : (2^(2 + 1) - 4 * (2 - 1)^2)^2 = 16 :=
by
  sorry

end evaluate_expression_l106_106858


namespace each_baby_worms_per_day_l106_106074

variable (babies : Nat) (worms_papa : Nat) (worms_mama_caught : Nat) (worms_mama_stolen : Nat) (worms_needed : Nat)
variable (days : Nat)

theorem each_baby_worms_per_day 
  (h1 : babies = 6) 
  (h2 : worms_papa = 9) 
  (h3 : worms_mama_caught = 13) 
  (h4 : worms_mama_stolen = 2)
  (h5 : worms_needed = 34) 
  (h6 : days = 3) :
  (worms_papa + (worms_mama_caught - worms_mama_stolen) + worms_needed) / babies / days = 3 :=
by
  sorry

end each_baby_worms_per_day_l106_106074


namespace carson_total_distance_l106_106286

def perimeter (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def total_distance (length : ℕ) (width : ℕ) (rounds : ℕ) (breaks : ℕ) (break_distance : ℕ) : ℕ :=
  let P := perimeter length width
  let distance_rounds := rounds * P
  let distance_breaks := breaks * break_distance
  distance_rounds + distance_breaks

theorem carson_total_distance :
  total_distance 600 400 8 4 100 = 16400 :=
by
  sorry

end carson_total_distance_l106_106286


namespace find_x_minus_y_l106_106884

variables (x y : ℚ)

theorem find_x_minus_y
  (h1 : 3 * x - 4 * y = 17)
  (h2 : x + 3 * y = 1) :
  x - y = 69 / 13 := 
sorry

end find_x_minus_y_l106_106884


namespace alok_total_payment_l106_106667

theorem alok_total_payment :
  let chapatis_cost := 16 * 6
  let rice_cost := 5 * 45
  let mixed_vegetable_cost := 7 * 70
  chapatis_cost + rice_cost + mixed_vegetable_cost = 811 :=
by
  sorry

end alok_total_payment_l106_106667


namespace boys_girls_ratio_l106_106656

theorem boys_girls_ratio (T G : ℕ) (h : (1/2 : ℚ) * G = (1/6 : ℚ) * T) :
  ((T - G) : ℚ) / G = 2 :=
by 
  sorry

end boys_girls_ratio_l106_106656


namespace evaluate_expression_l106_106533

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l106_106533


namespace find_n_l106_106714

theorem find_n :
  (∃ n : ℕ, arctan (1 / 3 : ℝ) + arctan (1 / 4) + arctan (1 / 6) + arctan (1 / n) = π / 4) →
  ∃ (n : ℕ), n = 113 :=
by
  sorry

end find_n_l106_106714


namespace discount_percentage_l106_106674

theorem discount_percentage (cost_price marked_price : ℝ) (profit_percentage : ℝ) 
  (h_cost_price : cost_price = 47.50)
  (h_marked_price : marked_price = 65)
  (h_profit_percentage : profit_percentage = 0.30) :
  ((marked_price - (cost_price + (profit_percentage * cost_price))) / marked_price) * 100 = 5 :=
by
  sorry

end discount_percentage_l106_106674


namespace flowerbed_width_l106_106980

theorem flowerbed_width (w : ℝ) (h₁ : 22 = 2 * (2 * w - 1) + 2 * w) : w = 4 :=
sorry

end flowerbed_width_l106_106980


namespace find_x_l106_106017

open Real

noncomputable def log_base (b x : ℝ) : ℝ := log x / log b

theorem find_x :
  ∃ x : ℝ, 0 < x ∧
  log_base 5 (x - 1) + log_base (sqrt 5) (x^2 - 1) + log_base (1/5) (x - 1) = 3 ∧
  x = sqrt (5 * sqrt 5 + 1) :=
by
  sorry

end find_x_l106_106017


namespace unique_representation_l106_106911

theorem unique_representation {p x y : ℕ} 
  (hp : p > 2 ∧ Prime p) 
  (h : 2 * y = p * (x + y)) 
  (hx : x ≠ y) : 
  ∃ x y : ℕ, (1/x + 1/y = 2/p) ∧ x ≠ y := 
sorry

end unique_representation_l106_106911


namespace volume_of_truncated_cone_l106_106031

noncomputable def surface_area_top : ℝ := 3 * Real.pi
noncomputable def surface_area_bottom : ℝ := 12 * Real.pi
noncomputable def slant_height : ℝ := 2
noncomputable def volume_cone : ℝ := 7 * Real.pi

theorem volume_of_truncated_cone :
  ∃ V : ℝ, V = volume_cone :=
sorry

end volume_of_truncated_cone_l106_106031


namespace days_until_see_grandma_l106_106903

def hours_in_a_day : ℕ := 24
def hours_until_see_grandma : ℕ := 48

theorem days_until_see_grandma : hours_until_see_grandma / hours_in_a_day = 2 := by
  sorry

end days_until_see_grandma_l106_106903


namespace books_jerry_added_l106_106901

def initial_action_figures : ℕ := 7
def initial_books : ℕ := 2

theorem books_jerry_added (B : ℕ) (h : initial_action_figures = initial_books + B + 1) : B = 4 :=
by
  sorry

end books_jerry_added_l106_106901


namespace no_integral_roots_l106_106484

theorem no_integral_roots :
  ¬(∃ (x : ℤ), 5 * x^2 + 3 = 40) ∧
  ¬(∃ (x : ℤ), (3 * x - 2)^3 = (x - 2)^3 - 27) ∧
  ¬(∃ (x : ℤ), x^2 - 4 = 3 * x - 4) :=
by sorry

end no_integral_roots_l106_106484


namespace john_bought_slurpees_l106_106754

noncomputable def slurpees_bought (total_money paid change slurpee_cost : ℕ) : ℕ :=
  (paid - change) / slurpee_cost

theorem john_bought_slurpees :
  let total_money := 20
  let slurpee_cost := 2
  let change := 8
  slurpees_bought total_money total_money change slurpee_cost = 6 :=
by
  sorry

end john_bought_slurpees_l106_106754


namespace remainder_q_div_x_plus_2_l106_106953

-- Define the polynomial q(x)
def q (x : ℝ) := 2 * x^4 - 3 * x^2 - 13 * x + 6

-- The main theorem we want to prove
theorem remainder_q_div_x_plus_2 :
  q 2 = 6 → (q (-2) = 52) :=
by
  intros h
  have q_2 : q 2 = 6 := h
  have q_neg2 : q (-2) = 2 * (-2)^4 - 3 * (-2)^2 - 13 * (-2) + 6 := by rfl
  rw [q_neg2]
  linarith
  sorry -- The actual proof steps would go here

end remainder_q_div_x_plus_2_l106_106953


namespace relay_team_average_time_l106_106854

theorem relay_team_average_time :
  let d1 := 200
  let t1 := 38
  let d2 := 300
  let t2 := 56
  let d3 := 250
  let t3 := 47
  let d4 := 400
  let t4 := 80
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  let average_time_per_meter := total_time / total_distance
  average_time_per_meter = 0.1922 := by
  sorry

end relay_team_average_time_l106_106854


namespace painting_frame_ratio_l106_106272

theorem painting_frame_ratio (x l : ℝ) (h1 : x > 0) (h2 : l > 0) 
  (h3 : (2 / 3) * x * x = (x + 2 * l) * ((3 / 2) * x + 2 * l) - x * (3 / 2) * x) :
  (x + 2 * l) / ((3 / 2) * x + 2 * l) = 3 / 4 :=
by
  sorry

end painting_frame_ratio_l106_106272


namespace price_per_kilo_of_bananas_l106_106766

def initial_money : ℕ := 500
def potatoes_cost : ℕ := 6 * 2
def tomatoes_cost : ℕ := 9 * 3
def cucumbers_cost : ℕ := 5 * 4
def bananas_weight : ℕ := 3
def remaining_money : ℕ := 426

-- Defining total cost of all items
def total_item_cost : ℕ := initial_money - remaining_money

-- Defining the total cost of bananas
def cost_bananas : ℕ := total_item_cost - (potatoes_cost + tomatoes_cost + cucumbers_cost)

-- Final question: Prove that the price per kilo of bananas is $5
theorem price_per_kilo_of_bananas : cost_bananas / bananas_weight = 5 :=
by
  sorry

end price_per_kilo_of_bananas_l106_106766


namespace smallest_n_for_identity_l106_106563

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1/2, - (Real.sqrt 3) / 2],
  ![(Real.sqrt 3) / 2, 1/2]
]

theorem smallest_n_for_identity : ∃ (n : ℕ), n > 0 ∧ A ^ n = 1 ∧ ∀ m : ℕ, m > 0 → A ^ m = 1 → n ≤ m :=
by
  sorry

end smallest_n_for_identity_l106_106563


namespace regular_polygon_sides_l106_106185

theorem regular_polygon_sides (exterior_angle : ℕ) (h : exterior_angle = 30) : (360 / exterior_angle) = 12 := by
  sorry

end regular_polygon_sides_l106_106185


namespace cost_of_one_dozen_pens_l106_106800

variable (x : ℝ)

-- Conditions 1 and 2 as assumptions
def pen_cost := 5 * x
def pencil_cost := x

axiom cost_equation  : 3 * pen_cost + 5 * pencil_cost = 200
axiom cost_ratio     : pen_cost / pencil_cost = 5 / 1 -- ratio is given

-- Question and target statement
theorem cost_of_one_dozen_pens : 12 * pen_cost = 600 :=
by
  sorry

end cost_of_one_dozen_pens_l106_106800


namespace polynomial_simplification_l106_106086

theorem polynomial_simplification (p : ℤ) :
  (5 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 2) + (-3 * p^4 + 4 * p^3 + 8 * p^2 - 2 * p + 6) = 
  2 * p^4 + 6 * p^3 + p^2 + p + 4 :=
by
  sorry

end polynomial_simplification_l106_106086


namespace equation_I_consecutive_integers_equation_II_consecutive_even_integers_l106_106528

theorem equation_I_consecutive_integers :
  ∃ (x y z : ℕ), x + y + z = 48 ∧ (x = y - 1) ∧ (z = y + 1) := sorry

theorem equation_II_consecutive_even_integers :
  ∃ (x y z w : ℕ), x + y + z + w = 52 ∧ (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) := sorry

end equation_I_consecutive_integers_equation_II_consecutive_even_integers_l106_106528


namespace find_other_number_l106_106492

def a : ℝ := 0.5
def d : ℝ := 0.16666666666666669
def b : ℝ := 0.3333333333333333

theorem find_other_number : a - d = b := by
  sorry

end find_other_number_l106_106492


namespace photograph_perimeter_l106_106653

theorem photograph_perimeter (w l m : ℕ) 
  (h1 : (w + 4) * (l + 4) = m)
  (h2 : (w + 8) * (l + 8) = m + 94) :
  2 * (w + l) = 23 := 
by
  sorry

end photograph_perimeter_l106_106653


namespace sqrt_fraction_simplified_l106_106157

theorem sqrt_fraction_simplified :
  Real.sqrt (4 / 3) = 2 * Real.sqrt 3 / 3 :=
by sorry

end sqrt_fraction_simplified_l106_106157


namespace exists_x_y_with_specific_difference_l106_106183

theorem exists_x_y_with_specific_difference :
  ∃ x y : ℤ, (2 * x^2 + 8 * y = 26) ∧ (x - y = 26) := 
sorry

end exists_x_y_with_specific_difference_l106_106183


namespace total_population_l106_106005

-- Defining the populations of Springfield and the difference in population
def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

-- The definition of Greenville's population in terms of Springfield's population
def greenville_population : ℕ := springfield_population - population_difference

-- The statement that we want to prove: the total population of Springfield and Greenville
theorem total_population :
  springfield_population + greenville_population = 845640 := by
  sorry

end total_population_l106_106005


namespace johnny_marbles_l106_106602

def num_ways_to_choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem johnny_marbles :
  num_ways_to_choose_marbles 7 3 = 35 :=
by
  sorry

end johnny_marbles_l106_106602


namespace max_value_on_ellipse_l106_106572

theorem max_value_on_ellipse (b : ℝ) (hb : b > 0) :
  ∃ (M : ℝ), 
    (∀ (x y : ℝ), (x^2 / 4 + y^2 / b^2 = 1) → x^2 + 2 * y ≤ M) ∧
    ((b ≤ 4 → M = b^2 / 4 + 4) ∧ (b > 4 → M = 2 * b)) :=
  sorry

end max_value_on_ellipse_l106_106572


namespace inequality_nonneg_ab_l106_106366

theorem inequality_nonneg_ab (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) :
  (1 + a)^4 * (1 + b)^4 ≥ 64 * a * b * (a + b)^2 :=
by
  sorry

end inequality_nonneg_ab_l106_106366


namespace equation_1_solution_set_equation_2_solution_set_l106_106088

open Real

theorem equation_1_solution_set (x : ℝ) : x^2 - 4 * x - 8 = 0 ↔ (x = 2 * sqrt 3 + 2 ∨ x = -2 * sqrt 3 + 2) :=
by sorry

theorem equation_2_solution_set (x : ℝ) : 3 * x - 6 = x * (x - 2) ↔ (x = 2 ∨ x = 3) :=
by sorry

end equation_1_solution_set_equation_2_solution_set_l106_106088


namespace weighted_average_yield_l106_106007

-- Define the conditions
def face_value_A : ℝ := 1000
def market_price_A : ℝ := 1200
def yield_A : ℝ := 0.18

def face_value_B : ℝ := 1000
def market_price_B : ℝ := 800
def yield_B : ℝ := 0.22

def face_value_C : ℝ := 1000
def market_price_C : ℝ := 1000
def yield_C : ℝ := 0.15

def investment_A : ℝ := 5000
def investment_B : ℝ := 3000
def investment_C : ℝ := 2000

-- Prove the weighted average yield
theorem weighted_average_yield :
  (investment_A + investment_B + investment_C) = 10000 →
  ((investment_A / (investment_A + investment_B + investment_C)) * yield_A +
   (investment_B / (investment_A + investment_B + investment_C)) * yield_B +
   (investment_C / (investment_A + investment_B + investment_C)) * yield_C) = 0.186 :=
by
  sorry

end weighted_average_yield_l106_106007


namespace multiply_seven_l106_106188

variable (x : ℕ)

theorem multiply_seven (h : 8 * x = 64) : 7 * x = 56 := by
  sorry


end multiply_seven_l106_106188


namespace driver_license_advantage_l106_106217

def AdvantageousReasonsForEarlyLicenseObtaining 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) : Prop :=
  ∀ age1 age2 : ℕ, (eligible age1 ∧ eligible age2 ∧ age1 < age2) →
  (effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1) →
  effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1

theorem driver_license_advantage 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) :
  AdvantageousReasonsForEarlyLicenseObtaining eligible effectiveInsurance rentalCarFlexibility employmentOpportunity :=
by
  sorry

end driver_license_advantage_l106_106217


namespace domain_of_function_l106_106855

theorem domain_of_function :
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ↔ (1 - x ≥ 0 ∧ x ≠ 0) :=
by
  sorry

end domain_of_function_l106_106855


namespace total_cost_after_discounts_l106_106704

theorem total_cost_after_discounts 
    (price_iphone : ℝ)
    (discount_iphone : ℝ)
    (price_iwatch : ℝ)
    (discount_iwatch : ℝ)
    (cashback_percentage : ℝ) :
    (price_iphone = 800) →
    (discount_iphone = 0.15) →
    (price_iwatch = 300) →
    (discount_iwatch = 0.10) →
    (cashback_percentage = 0.02) →
    let discounted_iphone := price_iphone * (1 - discount_iphone),
        discounted_iwatch := price_iwatch * (1 - discount_iwatch),
        total_discounted := discounted_iphone + discounted_iwatch,
        cashback := total_discounted * cashback_percentage 
    in total_discounted - cashback = 931 :=
by {
  intros,
  sorry
}

end total_cost_after_discounts_l106_106704


namespace average_weight_of_students_l106_106893

theorem average_weight_of_students (b_avg_weight g_avg_weight : ℝ) (num_boys num_girls : ℕ)
  (hb : b_avg_weight = 155) (hg : g_avg_weight = 125) (hb_num : num_boys = 8) (hg_num : num_girls = 5) :
  (num_boys * b_avg_weight + num_girls * g_avg_weight) / (num_boys + num_girls) = 143 :=
by sorry

end average_weight_of_students_l106_106893


namespace rubber_boat_fall_time_l106_106274

variable {a b x : ℝ}

theorem rubber_boat_fall_time
  (h1 : 5 - x = (a - b) / (a + b))
  (h2 : 6 - x = b / (a + b)) :
  x = 4 := by
  sorry

end rubber_boat_fall_time_l106_106274


namespace laura_charges_for_truck_l106_106904

theorem laura_charges_for_truck : 
  ∀ (car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars : ℕ),
  car_wash = 5 →
  suv_wash = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  num_cars = 7 →
  total_amount = 100 →
  car_wash * num_cars + suv_wash * num_suvs + truck_wash * num_trucks = total_amount →
  truck_wash = 6 :=
by
  intros car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars h1 h2 h3 h4 h5 h6 h7
  sorry

end laura_charges_for_truck_l106_106904


namespace problem1_subproblem1_subproblem2_l106_106277

-- Problem 1: Prove that a² + b² = 40 given ab = 30 and a + b = 10
theorem problem1 (a b : ℝ) (h1 : a * b = 30) (h2 : a + b = 10) : a^2 + b^2 = 40 := 
sorry

-- Problem 2: Subproblem 1 - Prove that (40 - x)² + (x - 20)² = 420 given (40 - x)(x - 20) = -10
theorem subproblem1 (x : ℝ) (h : (40 - x) * (x - 20) = -10) : (40 - x)^2 + (x - 20)^2 = 420 := 
sorry

-- Problem 2: Subproblem 2 - Prove that (30 + x)² + (20 + x)² = 120 given (30 + x)(20 + x) = 10
theorem subproblem2 (x : ℝ) (h : (30 + x) * (20 + x) = 10) : (30 + x)^2 + (20 + x)^2 = 120 :=
sorry

end problem1_subproblem1_subproblem2_l106_106277


namespace six_digit_pair_divisibility_l106_106293

theorem six_digit_pair_divisibility (a b : ℕ) (ha : 100000 ≤ a ∧ a < 1000000) (hb : 100000 ≤ b ∧ b < 1000000) :
  ((1000000 * a + b) % (a * b) = 0) ↔ (a = 166667 ∧ b = 333334) ∨ (a = 500001 ∧ b = 500001) :=
by sorry

end six_digit_pair_divisibility_l106_106293


namespace domain_ln_l106_106152

theorem domain_ln (x : ℝ) : (1 - 2 * x > 0) ↔ x < (1 / 2) :=
by
  sorry

end domain_ln_l106_106152


namespace simplify_expression_l106_106371

theorem simplify_expression :
  (3^4 + 3^2) / (3^3 - 3) = 15 / 4 :=
by {
  sorry
}

end simplify_expression_l106_106371


namespace james_profit_l106_106358

def cattle_profit (num_cattle : ℕ) (purchase_price total_feed_increase : ℝ)
    (weight_per_cattle : ℝ) (selling_price_per_pound : ℝ) : ℝ :=
  let feed_cost := purchase_price * (1 + total_feed_increase)
  let total_cost := purchase_price + feed_cost
  let revenue_per_cattle := weight_per_cattle * selling_price_per_pound
  let total_revenue := revenue_per_cattle * num_cattle
  total_revenue - total_cost

theorem james_profit : cattle_profit 100 40000 0.20 1000 2 = 112000 := by
  sorry

end james_profit_l106_106358


namespace range_of_a_l106_106736

noncomputable def f (x : ℝ) : ℝ := Real.log (2 + 3 * x) - (3 / 2) * x^2
noncomputable def f' (x : ℝ) : ℝ := (3 / (2 + 3 * x)) - 3 * x
noncomputable def valid_range (a : ℝ) : Prop := 
∀ x : ℝ, (1 / 6) ≤ x ∧ x ≤ (1 / 3) → |a - Real.log x| + Real.log (f' x + 3 * x) > 0

theorem range_of_a : { a : ℝ | valid_range a } = { a : ℝ | a ≠ Real.log (1 / 3) } := 
sorry

end range_of_a_l106_106736


namespace percent_of_male_literate_l106_106056

noncomputable def female_percentage : ℝ := 0.6
noncomputable def total_employees : ℕ := 1500
noncomputable def literate_percentage : ℝ := 0.62
noncomputable def literate_female_employees : ℕ := 630

theorem percent_of_male_literate :
  let total_females := (female_percentage * total_employees)
  let total_males := total_employees - total_females
  let total_literate := literate_percentage * total_employees
  let literate_male_employees := total_literate - literate_female_employees
  let male_literate_percentage := (literate_male_employees / total_males) * 100
  male_literate_percentage = 50 := by
  sorry

end percent_of_male_literate_l106_106056


namespace ratio_x_2y_l106_106869

theorem ratio_x_2y (x y : ℤ) (h : (7 * x + 8 * y) / (x - 2 * y) = 29) : x / (2 * y) = 3 / 2 :=
sorry

end ratio_x_2y_l106_106869


namespace quad_eq_pos_neg_root_l106_106491

theorem quad_eq_pos_neg_root (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 2 ∧ x₁ * x₂ = a + 1) ↔ a < -1 :=
by sorry

end quad_eq_pos_neg_root_l106_106491


namespace recycling_weight_l106_106762

theorem recycling_weight :
  let marcus_milk_bottles := 25
  let john_milk_bottles := 20
  let sophia_milk_bottles := 15
  let marcus_cans := 30
  let john_cans := 25
  let sophia_cans := 35
  let milk_bottle_weight := 0.5
  let can_weight := 0.025

  let total_milk_bottles_weight := (marcus_milk_bottles + john_milk_bottles + sophia_milk_bottles) * milk_bottle_weight
  let total_cans_weight := (marcus_cans + john_cans + sophia_cans) * can_weight
  let combined_weight := total_milk_bottles_weight + total_cans_weight

  combined_weight = 32.25 :=
by
  sorry

end recycling_weight_l106_106762


namespace count_valid_N_l106_106178

theorem count_valid_N : ∃ (N : ℕ), N = 1174 ∧ ∀ (n : ℕ), (1 ≤ n ∧ n < 2000) → ∃ (x : ℝ), x ^ (⌊x⌋ + 1) = n :=
by
  sorry

end count_valid_N_l106_106178


namespace advantage_18_vs_30_l106_106218

-- Definitions for advantages
def insurance_cost_effectiveness (age : ℕ) : Prop :=
  age = 18 → true

def rental_car_flexibility (age : ℕ) : Prop :=
  age = 18 → true

def employment_opportunities (age : ℕ) : Prop :=
  age = 18 → true

-- Aggregated definition of advantages
def advantage (age : ℕ) : ℕ :=
  if age = 18 then 3 else 0 -- Simplistic model for advantages count

-- Proof statement
theorem advantage_18_vs_30 : advantage 18 > advantage 30 :=
by { unfold advantage, norm_num }

end advantage_18_vs_30_l106_106218


namespace distance_between_house_and_school_l106_106261

theorem distance_between_house_and_school (T D : ℕ) 
    (h1 : D = 10 * (T + 2)) 
    (h2 : D = 20 * (T - 1)) : 
    D = 60 := by
  sorry

end distance_between_house_and_school_l106_106261


namespace smallest_next_divisor_l106_106753

theorem smallest_next_divisor (m : ℕ) (h_digit : 10000 ≤ m ∧ m < 100000) (h_odd : m % 2 = 1) (h_div : 437 ∣ m) :
  ∃ d : ℕ, 437 < d ∧ d ∣ m ∧ (∀ e : ℕ, 437 < e ∧ e < d → ¬ e ∣ m) ∧ d = 475 := 
sorry

end smallest_next_divisor_l106_106753


namespace cubic_expression_l106_106044

theorem cubic_expression {x : ℝ} (h : x + (1/x) = 5) : x^3 + (1/x^3) = 110 := 
by
  sorry

end cubic_expression_l106_106044


namespace cost_of_fencing_per_meter_l106_106633

theorem cost_of_fencing_per_meter (length breadth : ℕ) (total_cost : ℚ) 
    (h_length : length = 61) 
    (h_rule : length = breadth + 22) 
    (h_total_cost : total_cost = 5300) :
    total_cost / (2 * length + 2 * breadth) = 26.5 := 
by 
  sorry

end cost_of_fencing_per_meter_l106_106633


namespace noah_class_size_l106_106190

theorem noah_class_size :
  ∀ n : ℕ, (n = 39 + 39 + 1) → n = 79 :=
by
  intro n
  intro h
  exact h

end noah_class_size_l106_106190


namespace find_a_plus_b_l106_106579

theorem find_a_plus_b {f : ℝ → ℝ} (a b : ℝ) :
  (∀ x, f x = x^3 + 3*x^2 + 6*x + 14) →
  f a = 1 →
  f b = 19 →
  a + b = -2 :=
by
  sorry

end find_a_plus_b_l106_106579


namespace real_solutions_quadratic_l106_106011

theorem real_solutions_quadratic (d : ℝ) (h : 0 < d) :
  ∃ x : ℝ, x^2 - 8 * x + d < 0 ↔ 0 < d ∧ d < 16 :=
by
  sorry

end real_solutions_quadratic_l106_106011


namespace fraction_still_missing_l106_106826

theorem fraction_still_missing (x : ℕ) (hx : x > 0) :
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = (1/9 : ℚ) :=
by
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  have h_fraction_still_missing : (x - remaining) / x = (1/9 : ℚ) := sorry
  exact h_fraction_still_missing

end fraction_still_missing_l106_106826


namespace shanghai_team_score_l106_106053

variables (S B : ℕ)

-- Conditions
def yao_ming_points : ℕ := 30
def point_margin : ℕ := 10
def total_points_minus_10 : ℕ := 5 * yao_ming_points - 10
def combined_total_points : ℕ := total_points_minus_10

-- The system of equations as conditions
axiom condition1 : S - B = point_margin
axiom condition2 : S + B = combined_total_points

-- The proof statement
theorem shanghai_team_score : S = 75 :=
by
  sorry

end shanghai_team_score_l106_106053


namespace greater_chance_without_replacement_l106_106282

theorem greater_chance_without_replacement :
  let P1 := 3 / 8
  let P2 := 5 / 12
  P2 > P1 :=
by
  have h1 : P1 = 3 / 8 := rfl
  have h2 : P2 = 5 / 12 := rfl
  have h3 : (5 : ℚ) / 12 > 3 / 8 :=
    by norm_num
  exact h3

end greater_chance_without_replacement_l106_106282


namespace relation_between_a_b_c_l106_106360

theorem relation_between_a_b_c :
  let a := (3/7 : ℝ) ^ (2/7)
  let b := (2/7 : ℝ) ^ (3/7)
  let c := (2/7 : ℝ) ^ (2/7)
  a > c ∧ c > b :=
by {
  let a := (3/7 : ℝ) ^ (2/7)
  let b := (2/7 : ℝ) ^ (3/7)
  let c := (2/7 : ℝ) ^ (2/7)
  sorry
}

end relation_between_a_b_c_l106_106360


namespace white_socks_cost_proof_l106_106644

-- Define the cost of a single brown sock in cents
def brown_sock_cost (B : ℕ) : Prop :=
  15 * B = 300

-- Define the cost of two white socks in cents
def white_socks_cost (B : ℕ) (W : ℕ) : Prop :=
  W = B + 25

-- Statement of the problem
theorem white_socks_cost_proof : 
  ∃ B W : ℕ, brown_sock_cost B ∧ white_socks_cost B W ∧ W = 45 :=
by
  sorry

end white_socks_cost_proof_l106_106644


namespace fraction_of_tank_used_l106_106844

theorem fraction_of_tank_used (speed : ℝ) (fuel_efficiency : ℝ) (initial_fuel : ℝ) (time_traveled : ℝ)
  (h_speed : speed = 40) (h_fuel_eff : fuel_efficiency = 1 / 40) (h_initial_fuel : initial_fuel = 12) 
  (h_time : time_traveled = 5) : 
  (speed * time_traveled * fuel_efficiency) / initial_fuel = 5 / 12 :=
by
  -- Here the proof would go, but we add sorry to indicate it's incomplete.
  sorry

end fraction_of_tank_used_l106_106844


namespace num_whole_numbers_between_sqrt_50_and_sqrt_200_l106_106179

theorem num_whole_numbers_between_sqrt_50_and_sqrt_200 :
  let lower := Nat.ceil (Real.sqrt 50)
  let upper := Nat.floor (Real.sqrt 200)
  lower <= upper ∧ (upper - lower + 1) = 7 :=
by
  sorry

end num_whole_numbers_between_sqrt_50_and_sqrt_200_l106_106179


namespace petya_recover_x_y_l106_106506

theorem petya_recover_x_y (x y a b c d : ℝ)
    (hx_pos : x > 0) (hy_pos : y > 0)
    (ha : a = x + y) (hb : b = x - y) (hc : c = x / y) (hd : d = x * y) :
    ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ a = x' + y' ∧ b = x' - y' ∧ c = x' / y' ∧ d = x' * y' :=
sorry

end petya_recover_x_y_l106_106506


namespace john_buys_36_rolls_l106_106203

-- Definitions of the conditions
def cost_per_dozen := 5
def total_money_spent := 15
def rolls_per_dozen := 12

-- Theorem statement: John bought 36 rolls
theorem john_buys_36_rolls :
  let dozens_bought := total_money_spent / cost_per_dozen in
  let total_rolls := dozens_bought * rolls_per_dozen in
  total_rolls = 36 :=
by
  -- Proof steps would go here
  sorry

end john_buys_36_rolls_l106_106203


namespace snickers_bars_needed_l106_106200

-- Definitions for the problem conditions
def total_required_points : ℕ := 2000
def bunnies_sold : ℕ := 8
def bunny_points : ℕ := 100
def snickers_points : ℕ := 25
def points_from_bunnies : ℕ := bunnies_sold * bunny_points
def remaining_points_needed : ℕ := total_required_points - points_from_bunnies

-- Define the problem statement to prove
theorem snickers_bars_needed : remaining_points_needed / snickers_points = 48 :=
by
  -- Skipping the proof steps
  sorry

end snickers_bars_needed_l106_106200


namespace alok_total_payment_l106_106668

theorem alok_total_payment :
  let chapatis_cost := 16 * 6
  let rice_cost := 5 * 45
  let mixed_vegetable_cost := 7 * 70
  chapatis_cost + rice_cost + mixed_vegetable_cost = 811 :=
by
  sorry

end alok_total_payment_l106_106668


namespace snickers_bars_needed_l106_106198

-- Definitions of the conditions
def points_needed : ℕ := 2000
def chocolate_bunny_points : ℕ := 100
def number_of_chocolate_bunnies : ℕ := 8
def snickers_points : ℕ := 25

-- Derived conditions
def points_from_bunnies : ℕ := number_of_chocolate_bunnies * chocolate_bunny_points
def remaining_points : ℕ := points_needed - points_from_bunnies

-- Statement to prove
theorem snickers_bars_needed : ∀ (n : ℕ), n = remaining_points / snickers_points → n = 48 :=
by 
  sorry

end snickers_bars_needed_l106_106198


namespace Gunther_typing_correct_l106_106192

def GuntherTypingProblem : Prop :=
  let first_phase := (160 * (120 / 3))
  let second_phase := (200 * (180 / 3))
  let third_phase := (50 * 60)
  let fourth_phase := (140 * (90 / 3))
  let total_words := first_phase + second_phase + third_phase + fourth_phase
  total_words = 26200

theorem Gunther_typing_correct : GuntherTypingProblem := by
  sorry

end Gunther_typing_correct_l106_106192


namespace solve_inequality_l106_106312

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := (x - 2) * (a * x + 2 * a)

-- Theorem Statement
theorem solve_inequality (f_even : ∀ x a, f x a = f (-x) a) (f_inc : ∀ x y a, 0 < x → x < y → f x a ≤ f y a) :
    ∀ a > 0, { x : ℝ | f (2 - x) a > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  -- Sorry to skip the proof
  sorry

end solve_inequality_l106_106312


namespace count_common_divisors_l106_106151

theorem count_common_divisors : 
  (Nat.divisors 60 ∩ Nat.divisors 90 ∩ Nat.divisors 30).card = 8 :=
by
  sorry

end count_common_divisors_l106_106151


namespace abs_fraction_inequality_solution_l106_106774

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l106_106774


namespace range_of_m_l106_106587

theorem range_of_m {x : ℝ} (m : ℝ) :
  (∀ x, |x - 1| + |x - 2| + |x - 3| ≥ m) ↔ m ≤ 2 :=
by
  sorry

end range_of_m_l106_106587


namespace ratio_alcohol_to_water_l106_106813

-- Definitions of volume fractions for alcohol and water
def alcohol_volume_fraction : ℚ := 1 / 7
def water_volume_fraction : ℚ := 2 / 7

-- The theorem stating the ratio of alcohol to water volumes
theorem ratio_alcohol_to_water : (alcohol_volume_fraction / water_volume_fraction) = 1 / 2 :=
by sorry

end ratio_alcohol_to_water_l106_106813


namespace jimmy_bread_packs_needed_l106_106464

theorem jimmy_bread_packs_needed 
  (sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (initial_bread_slices : ℕ)
  (slices_per_pack : ℕ)
  (H1 : sandwiches = 8)
  (H2 : slices_per_sandwich = 2)
  (H3 : initial_bread_slices = 0)
  (H4 : slices_per_pack = 4) : 
  (8 * 2) / 4 = 4 := 
sorry

end jimmy_bread_packs_needed_l106_106464


namespace part1_part2_l106_106033

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 / (3 ^ x + 1) + a

theorem part1 (h : ∀ x : ℝ, f (-x) a = -f x a) : a = -1 :=
by sorry

noncomputable def f' (x : ℝ) : ℝ := 2 / (3 ^ x + 1) - 1

theorem part2 : ∀ t : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f' x + 1 = t ↔ 1 / 2 ≤ t ∧ t ≤ 1 :=
by sorry

end part1_part2_l106_106033


namespace solve_for_y_l106_106012

theorem solve_for_y :
  ∃ (y : ℝ), 
    (∑' n : ℕ, (4 * (n + 1) - 2) * y^n) = 100 ∧ |y| < 1 ∧ y = 0.6036 :=
sorry

end solve_for_y_l106_106012
