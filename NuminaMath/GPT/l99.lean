import Mathlib

namespace convert_2e_15pi_i4_to_rectangular_form_l99_99883

noncomputable def convert_to_rectangular_form (z : ℂ) : ℂ :=
  let θ := (15 * Real.pi) / 4
  let θ' := θ - 2 * Real.pi
  2 * Complex.exp (θ' * Complex.I)

theorem convert_2e_15pi_i4_to_rectangular_form :
  convert_to_rectangular_form (2 * Complex.exp ((15 * Real.pi) / 4 * Complex.I)) = (Real.sqrt 2 - Complex.I * Real.sqrt 2) :=
  sorry

end convert_2e_15pi_i4_to_rectangular_form_l99_99883


namespace find_principal_amount_l99_99314

def interest_rate_first_year : ℝ := 0.10
def compounding_periods_first_year : ℕ := 2
def interest_rate_second_year : ℝ := 0.12
def compounding_periods_second_year : ℕ := 4
def diff_interest : ℝ := 12

theorem find_principal_amount (P : ℝ)
  (h1_first : interest_rate_first_year / (compounding_periods_first_year : ℝ) = 0.05)
  (h1_second : interest_rate_second_year / (compounding_periods_second_year : ℝ) = 0.03)
  (compounded_amount : ℝ := P * (1 + 0.05)^(compounding_periods_first_year) * (1 + 0.03)^compounding_periods_second_year)
  (simple_interest : ℝ := P * (interest_rate_first_year + interest_rate_second_year) / 2 * 2)
  (h_diff : compounded_amount - P - simple_interest = diff_interest) : P = 597.01 :=
sorry

end find_principal_amount_l99_99314


namespace arith_seq_sum_geom_mean_proof_l99_99671

theorem arith_seq_sum_geom_mean_proof (a_1 : ℝ) (a_n : ℕ → ℝ)
(common_difference : ℝ) (s_n : ℕ → ℝ)
(h_sequence : ∀ n, a_n n = a_1 + (n - 1) * common_difference)
(h_sum : ∀ n, s_n n = n / 2 * (2 * a_1 + (n - 1) * common_difference))
(h_geom_mean : (s_n 2) ^ 2 = s_n 1 * s_n 4)
(h_common_diff : common_difference = -1) :
a_1 = -1 / 2 :=
sorry

end arith_seq_sum_geom_mean_proof_l99_99671


namespace find_x_y_sum_l99_99882

variable {x y : ℝ}

theorem find_x_y_sum (h₁ : (x-1)^3 + 1997 * (x-1) = -1) (h₂ : (y-1)^3 + 1997 * (y-1) = 1) : 
  x + y = 2 := 
by
  sorry

end find_x_y_sum_l99_99882


namespace determine_c_l99_99530

theorem determine_c (c : ℝ) 
  (h : ∃ a : ℝ, (∀ x : ℝ, x^2 + 200 * x + c = (x + a)^2)) : c = 10000 :=
sorry

end determine_c_l99_99530


namespace capital_of_z_l99_99370

theorem capital_of_z (x y z : ℕ) (annual_profit z_share : ℕ) (months_x months_y months_z : ℕ) 
    (rx ry : ℕ) (r : ℚ) :
  x = 20000 →
  y = 25000 →
  z_share = 14000 →
  annual_profit = 50000 →
  rx = 240000 →
  ry = 300000 →
  months_x = 12 →
  months_y = 12 →
  months_z = 7 →
  r = 7 / 25 →
  z * months_z * r = z_share / (rx + ry + z * months_z) →
  z = 30000 := 
by intros; sorry

end capital_of_z_l99_99370


namespace change_is_correct_l99_99251

def regular_ticket_cost : ℕ := 109
def child_discount : ℕ := 5
def payment_given : ℕ := 500

-- Prices for different people in the family
def child_ticket_cost (age : ℕ) : ℕ :=
  if age < 12 then regular_ticket_cost - child_discount else regular_ticket_cost

def parent_ticket_cost : ℕ := regular_ticket_cost
def family_ticket_cost : ℕ :=
  (child_ticket_cost 6) + (child_ticket_cost 10) + parent_ticket_cost + parent_ticket_cost

def change_received : ℕ := payment_given - family_ticket_cost

-- Prove that the change received is 74
theorem change_is_correct : change_received = 74 :=
by sorry

end change_is_correct_l99_99251


namespace randy_initial_blocks_l99_99110

theorem randy_initial_blocks (used_blocks left_blocks total_blocks : ℕ) (h1 : used_blocks = 19) (h2 : left_blocks = 59) : total_blocks = used_blocks + left_blocks → total_blocks = 78 :=
by 
  intros
  sorry

end randy_initial_blocks_l99_99110


namespace employees_excluding_manager_l99_99069

theorem employees_excluding_manager (average_salary average_increase manager_salary n : ℕ)
  (h_avg_salary : average_salary = 2400)
  (h_avg_increase : average_increase = 100)
  (h_manager_salary : manager_salary = 4900)
  (h_new_avg_salary : average_salary + average_increase = 2500)
  (h_total_salary : (n + 1) * (average_salary + average_increase) = n * average_salary + manager_salary) :
  n = 24 :=
by
  sorry

end employees_excluding_manager_l99_99069


namespace triangle_inequality_l99_99548

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality :
  ∃ (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 1 ∧ b = 2 ∧ c = 3) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 2 ∧ b = 3 ∧ c = 6) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 3 ∧ b = 3 ∧ c = 6) ∧ can_form_triangle a b c) :=
by
  sorry

end triangle_inequality_l99_99548


namespace apples_not_ripe_l99_99653

theorem apples_not_ripe (total_apples good_apples : ℕ) (h1 : total_apples = 14) (h2 : good_apples = 8) : total_apples - good_apples = 6 :=
by {
  sorry
}

end apples_not_ripe_l99_99653


namespace no_positive_integers_solution_l99_99241

theorem no_positive_integers_solution (m n : ℕ) (hm : m > 0) (hn : n > 0) : 4 * m * (m + 1) ≠ n * (n + 1) := 
by
  sorry

end no_positive_integers_solution_l99_99241


namespace opposite_of_neg_two_l99_99608

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l99_99608


namespace range_of_a_l99_99628

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → x^2 - 2*x + a < 0) ↔ a ≤ 0 :=
by sorry

end range_of_a_l99_99628


namespace solution_set_of_inequality_l99_99319

theorem solution_set_of_inequality (x : ℝ) : -2 * x - 1 < 3 ↔ x > -2 := 
by 
  sorry

end solution_set_of_inequality_l99_99319


namespace evaluate_expression_l99_99283

theorem evaluate_expression :
  (∃ (a b c : ℕ), a = 18 ∧ b = 3 ∧ c = 54 ∧ c = a * b ∧ (18^36 / 54^18) = (6^18)) :=
sorry

end evaluate_expression_l99_99283


namespace ratio_of_ages_l99_99942

variables (R J K : ℕ)

axiom h1 : R = J + 8
axiom h2 : R + 4 = 2 * (J + 4)
axiom h3 : (R + 4) * (K + 4) = 192

theorem ratio_of_ages : (R - J) / (R - K) = 2 :=
by sorry

end ratio_of_ages_l99_99942


namespace solution_set_of_inequality_l99_99052

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_deriv_neg : ∀ x : ℝ, 0 < x → (x^2 + 1) * deriv f x + 2 * x * f x < 0)
  (h_f_neg1_zero : f (-1) = 0) :
  { x : ℝ | f x > 0 } = { x | x < -1 } ∪ { x | 0 < x ∧ x < 1 } := by
  sorry

end solution_set_of_inequality_l99_99052


namespace x_cubed_plus_y_cubed_l99_99338

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l99_99338


namespace sin_sum_arcsin_arctan_l99_99965

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_sum_arcsin_arctan_l99_99965


namespace cone_fits_in_cube_l99_99866

noncomputable def height_cone : ℝ := 15
noncomputable def diameter_cone_base : ℝ := 8
noncomputable def side_length_cube : ℝ := 15
noncomputable def volume_cube : ℝ := side_length_cube ^ 3

theorem cone_fits_in_cube :
  (height_cone = 15) →
  (diameter_cone_base = 8) →
  (height_cone ≤ side_length_cube ∧ diameter_cone_base ≤ side_length_cube) →
  volume_cube = 3375 := by
  intros h_cone d_base fits
  sorry

end cone_fits_in_cube_l99_99866


namespace trajectory_equation_l99_99260

theorem trajectory_equation (x y a : ℝ) (h : x^2 + y^2 = a^2) :
  (x - y)^2 + 2*x*y = a^2 :=
by
  sorry

end trajectory_equation_l99_99260


namespace custom_op_example_l99_99068

def custom_op (a b : ℕ) : ℕ := (a + 1) / b

theorem custom_op_example : custom_op 2 (custom_op 3 4) = 3 := 
by
  sorry

end custom_op_example_l99_99068


namespace pascal_triangle_10_to_30_l99_99347

-- Definitions
def pascal_row_numbers (n : ℕ) : ℕ := n + 1

def total_numbers_up_to (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

-- Proof Statement
theorem pascal_triangle_10_to_30 :
  total_numbers_up_to 29 - total_numbers_up_to 9 = 400 := by
  sorry

end pascal_triangle_10_to_30_l99_99347


namespace boats_distance_one_minute_before_collision_l99_99504

noncomputable def distance_between_boats_one_minute_before_collision
  (speed_boat1 : ℝ) (speed_boat2 : ℝ) (initial_distance : ℝ) : ℝ :=
  let relative_speed := speed_boat1 + speed_boat2
  let relative_speed_per_minute := relative_speed / 60
  let time_to_collide := initial_distance / relative_speed_per_minute
  let distance_one_minute_before := initial_distance - (relative_speed_per_minute * (time_to_collide - 1))
  distance_one_minute_before

theorem boats_distance_one_minute_before_collision :
  distance_between_boats_one_minute_before_collision 5 21 20 = 0.4333 :=
by
  -- Proof skipped
  sorry

end boats_distance_one_minute_before_collision_l99_99504


namespace sufficient_not_necessary_of_and_false_or_true_l99_99212

variables (p q : Prop)

theorem sufficient_not_necessary_of_and_false_or_true :
  (¬(p ∧ q) → (p ∨ q)) ∧ ((p ∨ q) → ¬(¬(p ∧ q))) :=
sorry

end sufficient_not_necessary_of_and_false_or_true_l99_99212


namespace constant_expression_l99_99417

-- Suppose x is a real number
variable {x : ℝ}

-- Define the expression sum
def expr_sum (x : ℝ) : ℝ :=
|3 * x - 1| + |4 * x - 1| + |5 * x - 1| + |6 * x - 1| + 
|7 * x - 1| + |8 * x - 1| + |9 * x - 1| + |10 * x - 1| + 
|11 * x - 1| + |12 * x - 1| + |13 * x - 1| + |14 * x - 1| + 
|15 * x - 1| + |16 * x - 1| + |17 * x - 1|

-- The Lean statement of the problem to be proven
theorem constant_expression : (∃ x : ℝ, expr_sum x = 5) :=
sorry

end constant_expression_l99_99417


namespace exist_N_for_fn_eq_n_l99_99785

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_condition1 (m n : ℕ+) : (f m, f n) ≤ (m, n) ^ 2014
axiom f_condition2 (n : ℕ+) : n ≤ f n ∧ f n ≤ n + 2014

theorem exist_N_for_fn_eq_n :
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → f n = n := sorry

end exist_N_for_fn_eq_n_l99_99785


namespace value_of_a_l99_99551

theorem value_of_a (a : ℝ) :
  (∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) ↔ (6 ≤ a ∧ a ≤ 9) :=
by
  sorry

end value_of_a_l99_99551


namespace geometric_sum_equals_fraction_l99_99165

theorem geometric_sum_equals_fraction (n : ℕ) (a r : ℝ) 
  (h_a : a = 1) (h_r : r = 1 / 2) 
  (h_sum : a * (1 - r^n) / (1 - r) = 511 / 512) : 
  n = 9 := 
by 
  sorry

end geometric_sum_equals_fraction_l99_99165


namespace fraction_multiplication_l99_99329

theorem fraction_multiplication :
  ((3 : ℚ) / 4) ^ 3 * ((2 : ℚ) / 5) ^ 3 = (27 : ℚ) / 1000 := sorry

end fraction_multiplication_l99_99329


namespace estimate_less_Exact_l99_99371

variables (a b c d : ℕ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

def round_up (x : ℕ) : ℕ := x + 1
def round_down (x : ℕ) : ℕ := x - 1

theorem estimate_less_Exact
  (h₁ : round_down a = a - 1)
  (h₂ : round_down b = b - 1)
  (h₃ : round_down c = c - 1)
  (h₄ : round_up d = d + 1) :
  (round_down a + round_down b) / round_down c - round_up d < 
  (a + b) / c - d :=
sorry

end estimate_less_Exact_l99_99371


namespace principal_amount_l99_99529

-- Define the conditions and required result
theorem principal_amount
  (P R T : ℝ)
  (hR : R = 0.5)
  (h_diff : (P * R * (T + 4) / 100) - (P * R * T / 100) = 40) :
  P = 2000 :=
  sorry

end principal_amount_l99_99529


namespace shadow_length_of_flagpole_l99_99508

theorem shadow_length_of_flagpole :
  ∀ (S : ℝ), (18 : ℝ) / S = (22 : ℝ) / 55 → S = 45 :=
by
  intro S h
  sorry

end shadow_length_of_flagpole_l99_99508


namespace cost_price_of_article_l99_99662

theorem cost_price_of_article (SP : ℝ) (profit_percentage : ℝ) (profit_fraction : ℝ) (CP : ℝ) : 
  SP = 120 → profit_percentage = 25 → profit_fraction = profit_percentage / 100 → 
  SP = CP + profit_fraction * CP → CP = 96 :=
by intros hSP hprofit_percentage hprofit_fraction heq
   sorry

end cost_price_of_article_l99_99662


namespace sector_triangle_radii_l99_99957

theorem sector_triangle_radii 
  (r : ℝ) (theta : ℝ) (radius : ℝ) 
  (h_theta_eq: theta = 60)
  (h_radius_eq: radius = 10) :
  let R := (radius * Real.sqrt 3) / 3
  let r_in := (radius * Real.sqrt 3) / 6
  R = 10 * (Real.sqrt 3) / 3 ∧ r_in = 10 * (Real.sqrt 3) / 6 := 
by
  sorry

end sector_triangle_radii_l99_99957


namespace jello_cost_calculation_l99_99565

-- Conditions as definitions
def jello_per_pound : ℝ := 1.5
def tub_volume_cubic_feet : ℝ := 6
def cubic_foot_to_gallons : ℝ := 7.5
def gallon_weight_pounds : ℝ := 8
def cost_per_tablespoon_jello : ℝ := 0.5

-- Tub total water calculation
def tub_water_gallons (volume_cubic_feet : ℝ) (cubic_foot_to_gallons : ℝ) : ℝ :=
  volume_cubic_feet * cubic_foot_to_gallons

-- Water weight calculation
def water_weight_pounds (water_gallons : ℝ) (gallon_weight_pounds : ℝ) : ℝ :=
  water_gallons * gallon_weight_pounds

-- Jello mix required calculation
def jello_mix_tablespoons (water_pounds : ℝ) (jello_per_pound : ℝ) : ℝ :=
  water_pounds * jello_per_pound

-- Total cost calculation
def total_cost (jello_mix_tablespoons : ℝ) (cost_per_tablespoon_jello : ℝ) : ℝ :=
  jello_mix_tablespoons * cost_per_tablespoon_jello

-- Theorem statement
theorem jello_cost_calculation :
  total_cost (jello_mix_tablespoons (water_weight_pounds (tub_water_gallons tub_volume_cubic_feet cubic_foot_to_gallons) gallon_weight_pounds) jello_per_pound) cost_per_tablespoon_jello = 270 := 
by sorry

end jello_cost_calculation_l99_99565


namespace ellipse_foci_distance_sum_l99_99967

theorem ellipse_foci_distance_sum
    (x y : ℝ)
    (PF1 PF2 : ℝ)
    (a : ℝ)
    (h_ellipse : (x^2 / 36) + (y^2 / 16) = 1)
    (h_foci : ∀F1 F2, ∃e > 0, F1 = (e, 0) ∧ F2 = (-e, 0))
    (h_point_on_ellipse : ∀x y, (x^2 / 36) + (y^2 / 16) = 1 → (x, y) = (PF1, PF2))
    (h_semi_major_axis : a = 6):
    |PF1| + |PF2| = 12 := 
by
  sorry

end ellipse_foci_distance_sum_l99_99967


namespace find_c_l99_99819

-- Define the two points as given in the problem
def pointA : ℝ × ℝ := (-6, 1)
def pointB : ℝ × ℝ := (-3, 4)

-- Define the direction vector as subtraction of the two points
def directionVector : ℝ × ℝ := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Define the target direction vector format with unknown c
def targetDirectionVector (c : ℝ) : ℝ × ℝ := (3, c)

-- The theorem stating that c must be 3
theorem find_c : ∃ c : ℝ, directionVector = targetDirectionVector c ∧ c = 3 := 
by
  -- Prove the statement or show it is derivable
  sorry

end find_c_l99_99819


namespace triangle_ABC_area_l99_99572

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1, 2)
def C : point := (2, 0)

def triangle_area (A B C : point) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))|

theorem triangle_ABC_area :
  triangle_area A B C = 2 :=
by
  sorry

end triangle_ABC_area_l99_99572


namespace sum_a2012_a2013_l99_99249

-- Define the geometric sequence and its conditions
def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n * q

-- Parameters for the problem
variable (a : ℕ → ℚ)
variable (q : ℚ)
variable (h_seq : geometric_sequence a q)
variable (h_q : 1 < q)
variable (h_eq : ∀ x : ℚ, 4 * x^2 - 8 * x + 3 = 0 → x = a 2010 ∨ x = a 2011)

-- Statement to prove
theorem sum_a2012_a2013 : a 2012 + a 2013 = 18 :=
by
  sorry

end sum_a2012_a2013_l99_99249


namespace math_problem_l99_99090

variable (a : ℝ)
noncomputable def problem := a = Real.sqrt 11 - 1
noncomputable def target := a^2 + 2 * a + 1 = 11

theorem math_problem (h : problem a) : target a :=
  sorry

end math_problem_l99_99090


namespace correct_option_l99_99082

-- Definition of the conditions
def conditionA : Prop := (Real.sqrt ((-1 : ℝ)^2) = 1)
def conditionB : Prop := (Real.sqrt ((-1 : ℝ)^2) = -1)
def conditionC : Prop := (Real.sqrt (-(1^2) : ℝ) = 1)
def conditionD : Prop := (Real.sqrt (-(1^2) : ℝ) = -1)

-- Proving the correct condition
theorem correct_option : conditionA := by
  sorry

end correct_option_l99_99082


namespace accounting_majors_l99_99620

theorem accounting_majors (p q r s t u : ℕ) 
  (hpqt : (p * q * r * s * t * u = 51030)) 
  (hineq : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) :
  p = 2 :=
sorry

end accounting_majors_l99_99620


namespace round_table_vip_arrangements_l99_99878

-- Define the conditions
def number_of_people : ℕ := 10
def vip_seats : ℕ := 2

noncomputable def number_of_arrangements : ℕ :=
  let total_arrangements := Nat.factorial number_of_people
  let vip_choices := Nat.choose number_of_people vip_seats
  let remaining_arrangements := Nat.factorial (number_of_people - vip_seats)
  vip_choices * remaining_arrangements

-- Theorem stating the result
theorem round_table_vip_arrangements : number_of_arrangements = 1814400 := by
  sorry

end round_table_vip_arrangements_l99_99878


namespace line_through_points_l99_99815

theorem line_through_points :
  ∀ x y : ℝ, (∃ t : ℝ, (x, y) = (2 * t, -3 * (1 - t))) ↔ (x / 2) - (y / 3) = 1 :=
by
  sorry

end line_through_points_l99_99815


namespace fraction_of_percent_l99_99363

theorem fraction_of_percent (h : (1 / 8 * (1 / 100)) * 800 = 1) : true :=
by
  trivial

end fraction_of_percent_l99_99363


namespace circle_area_isosceles_triangle_l99_99162

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end circle_area_isosceles_triangle_l99_99162


namespace expression_eval_l99_99250

theorem expression_eval :
  (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) = 125 :=
by
  sorry

end expression_eval_l99_99250


namespace flour_vs_sugar_difference_l99_99185

-- Definitions based on the conditions
def flour_needed : ℕ := 10
def flour_added : ℕ := 7
def sugar_needed : ℕ := 2

-- Define the mathematical statement to prove
theorem flour_vs_sugar_difference :
  (flour_needed - flour_added) - sugar_needed = 1 :=
by
  sorry

end flour_vs_sugar_difference_l99_99185


namespace geometric_quadratic_root_l99_99332

theorem geometric_quadratic_root (a b c : ℝ) (h1 : a > 0) (h2 : b = a * (1 / 4)) (h3 : c = a * (1 / 16)) (h4 : a * a * (1 / 4)^2 = 4 * a * a * (1 / 16)) : 
    -b / (2 * a) = -1 / 8 :=
by 
    sorry

end geometric_quadratic_root_l99_99332


namespace compare_negative_fractions_l99_99496

theorem compare_negative_fractions : (- (1 / 3 : ℝ)) < (- (1 / 4 : ℝ)) :=
sorry

end compare_negative_fractions_l99_99496


namespace books_received_l99_99528

theorem books_received (students : ℕ) (books_per_student : ℕ) (books_fewer : ℕ) (expected_books : ℕ) (received_books : ℕ) :
  students = 20 →
  books_per_student = 15 →
  books_fewer = 6 →
  expected_books = students * books_per_student →
  received_books = expected_books - books_fewer →
  received_books = 294 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end books_received_l99_99528


namespace find_q_l99_99959

variable {m n q : ℝ}

theorem find_q (h1 : m = 3 * n + 5) (h2 : m + 2 = 3 * (n + q) + 5) : q = 2 / 3 := by
  sorry

end find_q_l99_99959


namespace katie_earnings_l99_99947

theorem katie_earnings :
  4 * 3 + 3 * 7 + 2 * 5 + 5 * 2 = 53 := 
by 
  sorry

end katie_earnings_l99_99947


namespace total_drink_ounces_l99_99710

def total_ounces_entire_drink (coke_parts sprite_parts md_parts coke_ounces : ℕ) : ℕ :=
  let total_parts := coke_parts + sprite_parts + md_parts
  let ounces_per_part := coke_ounces / coke_parts
  total_parts * ounces_per_part

theorem total_drink_ounces (coke_parts sprite_parts md_parts coke_ounces : ℕ) (coke_cond : coke_ounces = 8) (parts_cond : coke_parts = 4 ∧ sprite_parts = 2 ∧ md_parts = 5) : 
  total_ounces_entire_drink coke_parts sprite_parts md_parts coke_ounces = 22 :=
by
  sorry

end total_drink_ounces_l99_99710


namespace smallest_N_l99_99478

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n
  
noncomputable def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

noncomputable def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

theorem smallest_N :
  ∃ N : ℕ, is_square (N / 2) ∧ is_cube (N / 3) ∧ is_fifth_power (N / 5) ∧
  N = 2^15 * 3^10 * 5^6 :=
by
  exists 2^15 * 3^10 * 5^6
  sorry

end smallest_N_l99_99478


namespace sum_of_perimeters_of_squares_l99_99045

theorem sum_of_perimeters_of_squares (x y : ℕ)
  (h1 : x^2 - y^2 = 19) : 4 * x + 4 * y = 76 := 
by
  sorry

end sum_of_perimeters_of_squares_l99_99045


namespace ratio_of_P_to_Q_l99_99397

theorem ratio_of_P_to_Q (p q r s : ℕ) (h1 : p + q + r + s = 1000)
    (h2 : s = 4 * r) (h3 : q = r) (h4 : s - p = 250) : 
    p = 2 * q :=
by
  -- Proof omitted
  sorry

end ratio_of_P_to_Q_l99_99397


namespace vector_problem_l99_99042

noncomputable def t_value : ℝ :=
  (-5 - Real.sqrt 13) / 2

theorem vector_problem 
  (t : ℝ)
  (a : ℝ × ℝ := (1, 1))
  (b : ℝ × ℝ := (2, t))
  (h : Real.sqrt ((1 - 2)^2 + (1 - t)^2) = (1 * 2 + 1 * t)) :
  t = t_value := 
sorry

end vector_problem_l99_99042


namespace total_sum_of_money_l99_99547

theorem total_sum_of_money (x : ℝ) (A B C : ℝ) 
  (hA : A = x) 
  (hB : B = 0.65 * x) 
  (hC : C = 0.40 * x) 
  (hC_share : C = 32) :
  A + B + C = 164 := 
  sorry

end total_sum_of_money_l99_99547


namespace calculation_result_l99_99376

theorem calculation_result : 7 * (9 + 2 / 5) + 3 = 68.8 :=
by
  sorry

end calculation_result_l99_99376


namespace jan_clean_car_water_l99_99645

def jan_water_problem
  (initial_water : ℕ)
  (car_water : ℕ)
  (plant_additional : ℕ)
  (plate_clothes_water : ℕ)
  (remaining_water : ℕ)
  (used_water : ℕ)
  (car_cleaning_water : ℕ) : Prop :=
  initial_water = 65 ∧
  plate_clothes_water = 24 ∧
  plant_additional = 11 ∧
  remaining_water = 2 * plate_clothes_water ∧
  used_water = initial_water - remaining_water ∧
  car_water = used_water + plant_additional ∧
  car_cleaning_water = car_water / 4

theorem jan_clean_car_water : jan_water_problem 65 17 11 24 48 17 7 :=
by {
  sorry
}

end jan_clean_car_water_l99_99645


namespace find_weight_of_second_square_l99_99555

-- Define given conditions
def side_length1 : ℝ := 4
def weight1 : ℝ := 16
def side_length2 : ℝ := 6

-- Define the uniform density and thickness condition
def uniform_density (a₁ a₂ : ℝ) (w₁ w₂ : ℝ) : Prop :=
  (a₁ * w₂ = a₂ * w₁)

-- Problem statement:
theorem find_weight_of_second_square : 
  uniform_density (side_length1 ^ 2) (side_length2 ^ 2) weight1 w₂ → 
  w₂ = 36 :=
by
  sorry

end find_weight_of_second_square_l99_99555


namespace volume_of_cone_l99_99075

theorem volume_of_cone (d h : ℝ) (d_eq : d = 12) (h_eq : h = 9) : 
  (1 / 3) * π * (d / 2)^2 * h = 108 * π := 
by 
  rw [d_eq, h_eq] 
  sorry

end volume_of_cone_l99_99075


namespace water_usage_l99_99790

def fee (x : ℕ) : ℕ :=
  if x ≤ 8 then 2 * x else 4 * x - 16

theorem water_usage (h : fee 9 = 20) : fee 9 = 20 := by
  sorry

end water_usage_l99_99790


namespace range_of_a_l99_99585

noncomputable def has_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 * a * 9^(Real.sin x) + 4 * a * 3^(Real.sin x) + a - 8 = 0

theorem range_of_a : ∀ a : ℝ,
  (has_solutions a ↔ (8 / 31 <= a ∧ a <= 72 / 23)) := sorry

end range_of_a_l99_99585


namespace rate_of_discount_l99_99583

theorem rate_of_discount (marked_price : ℝ) (selling_price : ℝ) (rate : ℝ)
  (h_marked : marked_price = 125) (h_selling : selling_price = 120)
  (h_rate : rate = ((marked_price - selling_price) / marked_price) * 100) :
  rate = 4 :=
by
  subst h_marked
  subst h_selling
  subst h_rate
  sorry

end rate_of_discount_l99_99583


namespace value_of_x_squared_minus_y_squared_l99_99617

theorem value_of_x_squared_minus_y_squared 
  (x y : ℚ)
  (h1 : x + y = 5 / 8) 
  (h2 : x - y = 3 / 8) :
  x^2 - y^2 = 15 / 64 :=
by 
  sorry

end value_of_x_squared_minus_y_squared_l99_99617


namespace parallel_lines_m_value_l99_99321

theorem parallel_lines_m_value (x y m : ℝ) (h₁ : 2 * x + m * y - 7 = 0) (h₂ : m * x + 8 * y - 14 = 0) (parallel : (2 / m = m / 8)) : m = -4 := 
sorry

end parallel_lines_m_value_l99_99321


namespace triangle_inequality_for_n6_l99_99353

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l99_99353


namespace opposite_sqrt_4_l99_99265

theorem opposite_sqrt_4 : - (Real.sqrt 4) = -2 := sorry

end opposite_sqrt_4_l99_99265


namespace Sam_total_books_l99_99510

/-- Sam's book purchases -/
def Sam_bought_books : Real := 
  let used_adventure_books := 13.0
  let used_mystery_books := 17.0
  let new_crime_books := 15.0
  used_adventure_books + used_mystery_books + new_crime_books

theorem Sam_total_books : Sam_bought_books = 45.0 :=
by
  -- The proof will show that Sam indeed bought 45 books in total
  sorry

end Sam_total_books_l99_99510


namespace min_players_team_l99_99820

theorem min_players_team : Nat.lcm (Nat.lcm (Nat.lcm 8 9) 10) 11 = 7920 := 
by 
  -- The proof will be filled here.
  sorry

end min_players_team_l99_99820


namespace functionG_has_inverse_l99_99261

noncomputable def functionG : ℝ → ℝ := -- function G described in the problem.
sorry

-- Define the horizontal line test
def horizontal_line_test (f : ℝ → ℝ) : Prop :=
∀ y : ℝ, ∃! x : ℝ, f x = y

theorem functionG_has_inverse : horizontal_line_test functionG :=
sorry

end functionG_has_inverse_l99_99261


namespace smallest_rel_prime_to_180_l99_99597

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ (∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → y ≥ x) ∧ x = 7 :=
  sorry

end smallest_rel_prime_to_180_l99_99597


namespace find_c_l99_99738

variable (c : ℝ)

theorem find_c (h : c * (1 + 1/2 + 1/3 + 1/4) = 1) : c = 12 / 25 :=
by 
  sorry

end find_c_l99_99738


namespace intersection_point_of_lines_PQ_RS_l99_99014

def point := ℝ × ℝ × ℝ

def P : point := (4, -3, 6)
def Q : point := (1, 10, 11)
def R : point := (3, -4, 2)
def S : point := (-1, 5, 16)

theorem intersection_point_of_lines_PQ_RS :
  let line_PQ (u : ℝ) := (4 - 3 * u, -3 + 13 * u, 6 + 5 * u)
  let line_RS (v : ℝ) := (3 - 4 * v, -4 + 9 * v, 2 + 14 * v)
  ∃ u v : ℝ,
    line_PQ u = line_RS v →
    line_PQ u = (19 / 5, 44 / 3, 23 / 3) :=
by
  sorry

end intersection_point_of_lines_PQ_RS_l99_99014


namespace tax_free_amount_l99_99006

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) 
    (tax_rate : ℝ) (exceeds_value : ℝ) :
    total_value = 1720 → 
    tax_rate = 0.11 → 
    tax_paid = 123.2 → 
    total_value - X = exceeds_value → 
    tax_paid = tax_rate * exceeds_value → 
    X = 600 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end tax_free_amount_l99_99006


namespace cube_removal_minimum_l99_99956

theorem cube_removal_minimum (l w h : ℕ) (hu : l = 4) (hv : w = 5) (hw : h = 6) :
  ∃ num_cubes_removed : ℕ, 
    (l * w * h - num_cubes_removed = 4 * 4 * 4) ∧ 
    num_cubes_removed = 56 := 
by
  sorry

end cube_removal_minimum_l99_99956


namespace greatest_possible_value_of_a_l99_99186

theorem greatest_possible_value_of_a :
  ∃ a : ℕ, (∀ x : ℤ, x * (x + a) = -12) → a = 13 := by
  sorry

end greatest_possible_value_of_a_l99_99186


namespace minimum_value_fraction_l99_99515

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

/-- Given that the function f(x) = log_a(4x-3) + 1 (where a > 0 and a ≠ 1) has a fixed point A(m, n), 
if for any positive numbers x and y, mx + ny = 3, 
then the minimum value of 1/(x+1) + 1/y is 1. -/
theorem minimum_value_fraction (a x y : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (hx : x + y = 3) : 
  (1 / (x + 1) + 1 / y) = 1 := 
sorry

end minimum_value_fraction_l99_99515


namespace pump_fill_time_without_leak_l99_99103

theorem pump_fill_time_without_leak
    (P : ℝ)
    (h1 : 2 + 1/7 = (15:ℝ)/7)
    (h2 : 1 / P - 1 / 30 = 7 / 15) :
  P = 2 := by
  sorry

end pump_fill_time_without_leak_l99_99103


namespace min_value_expression_l99_99841

theorem min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π / 2 → 
    (3 * Real.sin θ' + 4 / Real.cos θ' + 2 * Real.sqrt 3 * Real.tan θ') ≥ 9 * Real.sqrt 3) ∧ 
    (3 * Real.sin θ + 4 / Real.cos θ + 2 * Real.sqrt 3 * Real.tan θ = 9 * Real.sqrt 3) :=
by
  sorry

end min_value_expression_l99_99841


namespace loss_percentage_on_first_book_l99_99716

theorem loss_percentage_on_first_book 
    (C1 C2 SP : ℝ) 
    (H1 : C1 = 210) 
    (H2 : C1 + C2 = 360) 
    (H3 : SP = 1.19 * C2) 
    (H4 : SP = 178.5) :
    ((C1 - SP) / C1) * 100 = 15 :=
by
  sorry

end loss_percentage_on_first_book_l99_99716


namespace diagonals_of_polygon_l99_99481

theorem diagonals_of_polygon (f : ℕ → ℕ) (k : ℕ) (h_k : k ≥ 3) : f (k + 1) = f k + (k - 1) :=
sorry

end diagonals_of_polygon_l99_99481


namespace contrapositive_of_squared_sum_eq_zero_l99_99410

theorem contrapositive_of_squared_sum_eq_zero (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by
  sorry

end contrapositive_of_squared_sum_eq_zero_l99_99410


namespace largest_angle_of_trapezoid_arithmetic_sequence_l99_99390

variables (a d : ℝ)

-- Given Conditions
def smallest_angle : Prop := a = 45
def trapezoid_property : Prop := a + 3 * d = 135

theorem largest_angle_of_trapezoid_arithmetic_sequence 
  (ha : smallest_angle a) (ht : a + (a + 3 * d) = 180) : 
  a + 3 * d = 135 :=
by
  sorry

end largest_angle_of_trapezoid_arithmetic_sequence_l99_99390


namespace total_surfers_calculation_l99_99909

def surfers_on_malibu_beach (m_sm : ℕ) (s_sm : ℕ) : ℕ := 2 * s_sm

def total_surfers (m_sm s_sm : ℕ) : ℕ := m_sm + s_sm

theorem total_surfers_calculation : total_surfers (surfers_on_malibu_beach 20 20) 20 = 60 := by
  sorry

end total_surfers_calculation_l99_99909


namespace parabola_equation_l99_99157

theorem parabola_equation (h k a : ℝ) (same_shape : ∀ x, -2 * x^2 + 2 = a * x^2 + k) (vertex : h = 4 ∧ k = -2) :
  ∀ x, -2 * (x - 4)^2 - 2 = a * (x - h)^2 + k :=
by
  -- This is where the actual proof would go
  simp
  sorry

end parabola_equation_l99_99157


namespace no_solution_intervals_l99_99167

theorem no_solution_intervals (a : ℝ) :
  (a < -13 ∨ a > 0) → ¬ ∃ x : ℝ, 6 * abs (x - 4 * a) + abs (x - a^2) + 5 * x - 3 * a = 0 :=
by sorry

end no_solution_intervals_l99_99167


namespace current_women_count_l99_99401

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

end current_women_count_l99_99401


namespace intersection_of_A_and_B_l99_99201

def A := {x : ℝ | |x - 2| ≤ 1}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}
def C := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l99_99201


namespace find_k_intersecting_lines_l99_99214

theorem find_k_intersecting_lines : 
  ∃ (k : ℚ), (∃ (x y : ℚ), y = 6 * x + 4 ∧ y = -3 * x - 30 ∧ y = 4 * x + k) ∧ k = -32 / 9 :=
by
  sorry

end find_k_intersecting_lines_l99_99214


namespace fg_eval_at_3_l99_99232

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_eval_at_3 : f (g 3) = 99 := by
  sorry

end fg_eval_at_3_l99_99232


namespace select_eight_genuine_dinars_l99_99126

theorem select_eight_genuine_dinars (coins : Fin 11 → ℝ) :
  (∃ (fake_coin : Option (Fin 11)), 
    ((∀ i j : Fin 11, i ≠ j → coins i = coins j) ∨
    (∀ (genuine_coins impostor_coins : Finset (Fin 11)), 
      genuine_coins ∪ impostor_coins = Finset.univ →
      impostor_coins.card = 1 →
      (∃ difference : ℝ, ∀ i ∈ genuine_coins, coins i = difference) ∧
      (∃ i ∈ impostor_coins, coins i ≠ difference)))) →
  (∃ (selected_coins : Finset (Fin 11)), selected_coins.card = 8 ∧
   (∀ i j : Fin 11, i ∈ selected_coins → j ∈ selected_coins → coins i = coins j)) :=
sorry

end select_eight_genuine_dinars_l99_99126


namespace jasmine_percentage_after_adding_l99_99173

def initial_solution_volume : ℕ := 80
def initial_jasmine_percentage : ℝ := 0.10
def additional_jasmine_volume : ℕ := 5
def additional_water_volume : ℕ := 15

theorem jasmine_percentage_after_adding :
  let initial_jasmine_volume := initial_jasmine_percentage * initial_solution_volume
  let total_jasmine_volume := initial_jasmine_volume + additional_jasmine_volume
  let total_solution_volume := initial_solution_volume + additional_jasmine_volume + additional_water_volume
  let final_jasmine_percentage := (total_jasmine_volume / total_solution_volume) * 100
  final_jasmine_percentage = 13 := by
  sorry

end jasmine_percentage_after_adding_l99_99173


namespace directrix_parabola_l99_99300

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end directrix_parabola_l99_99300


namespace find_f_2009_l99_99520

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom cond2 : f 1 = 2

theorem find_f_2009 : f 2009 = 2 := by
  sorry

end find_f_2009_l99_99520


namespace expand_product_l99_99181

theorem expand_product (x : ℝ) : (x + 5) * (x + 9) = x^2 + 14 * x + 45 :=
by
  sorry

end expand_product_l99_99181


namespace gift_contributors_l99_99918

theorem gift_contributors :
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ 20 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → (9 : ℕ) ≤ 20) →
  (∃ (n : ℕ), n = 12) :=
by
  sorry

end gift_contributors_l99_99918


namespace functional_equation_identity_l99_99885

def f : ℝ → ℝ := sorry

theorem functional_equation_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y + y) = f (f (x * y)) + y) : 
  ∀ y : ℝ, f y = y :=
sorry

end functional_equation_identity_l99_99885


namespace age_problem_l99_99708

theorem age_problem :
  ∃ (x y z : ℕ), 
    x - y = 3 ∧
    z = 2 * x + 2 * y - 3 ∧
    z = x + y + 20 ∧
    x = 13 ∧
    y = 10 ∧
    z = 43 :=
by 
  sorry

end age_problem_l99_99708


namespace ratio_of_A_to_B_l99_99700

theorem ratio_of_A_to_B (total_weight compound_A_weight compound_B_weight : ℝ)
  (h1 : total_weight = 108)
  (h2 : compound_B_weight = 90)
  (h3 : compound_A_weight = total_weight - compound_B_weight) :
  compound_A_weight / compound_B_weight = 1 / 5 :=
by
  sorry

end ratio_of_A_to_B_l99_99700


namespace tom_total_payment_l99_99168

variable (apples_kg : ℕ := 8)
variable (apples_rate : ℕ := 70)
variable (mangoes_kg : ℕ := 9)
variable (mangoes_rate : ℕ := 65)
variable (oranges_kg : ℕ := 5)
variable (oranges_rate : ℕ := 50)
variable (bananas_kg : ℕ := 3)
variable (bananas_rate : ℕ := 30)
variable (discount_apples : ℝ := 0.10)
variable (discount_oranges : ℝ := 0.15)

def total_cost_apple : ℝ := apples_kg * apples_rate
def total_cost_mango : ℝ := mangoes_kg * mangoes_rate
def total_cost_orange : ℝ := oranges_kg * oranges_rate
def total_cost_banana : ℝ := bananas_kg * bananas_rate
def discount_apples_amount : ℝ := discount_apples * total_cost_apple
def discount_oranges_amount : ℝ := discount_oranges * total_cost_orange
def apples_after_discount : ℝ := total_cost_apple - discount_apples_amount
def oranges_after_discount : ℝ := total_cost_orange - discount_oranges_amount

theorem tom_total_payment :
  apples_after_discount + total_cost_mango + oranges_after_discount + total_cost_banana = 1391.5 := by
  sorry

end tom_total_payment_l99_99168


namespace midpoint_of_hyperbola_l99_99516

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l99_99516


namespace tea_maker_capacity_l99_99032

theorem tea_maker_capacity (x : ℝ) (h : 0.45 * x = 54) : x = 120 :=
by
  sorry

end tea_maker_capacity_l99_99032


namespace multiples_7_not_14_l99_99894

theorem multiples_7_not_14 (n : ℕ) : (n < 500) → (n % 7 = 0) → (n % 14 ≠ 0) → ∃ k, (k = 36) :=
by {
  sorry
}

end multiples_7_not_14_l99_99894


namespace solve_system_l99_99627

-- Define the system of equations
def eq1 (x y : ℝ) : Prop := 2 * x - y = 8
def eq2 (x y : ℝ) : Prop := 3 * x + 2 * y = 5

-- State the theorem to be proved
theorem solve_system : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 3 ∧ y = -2 := 
by 
  exists 3
  exists -2
  -- Proof steps would go here, but we're using sorry to indicate it's incomplete
  sorry

end solve_system_l99_99627


namespace intersection_eq_l99_99449

open Set

def A : Set ℕ := {0, 2, 4, 6}
def B : Set ℕ := {x | 3 < x ∧ x < 7}

theorem intersection_eq : A ∩ B = {4, 6} := 
by 
  sorry

end intersection_eq_l99_99449


namespace largest_multiple_of_8_less_than_neg_63_l99_99017

theorem largest_multiple_of_8_less_than_neg_63 : 
  ∃ n : ℤ, (n < -63) ∧ (∃ k : ℤ, n = 8 * k) ∧ (∀ m : ℤ, (m < -63) ∧ (∃ l : ℤ, m = 8 * l) → m ≤ n) :=
sorry

end largest_multiple_of_8_less_than_neg_63_l99_99017


namespace bill_salary_increase_l99_99795

theorem bill_salary_increase (S P : ℝ) 
  (h1 : S + 0.16 * S = 812) 
  (h2 : S + P * S = 770.0000000000001) : 
  P = 0.1 :=
by {
  sorry
}

end bill_salary_increase_l99_99795


namespace main_theorem_l99_99403

-- defining the conditions
def cost_ratio_pen_pencil (x : ℕ) : Prop :=
  ∀ (pen pencil : ℕ), pen = 5 * pencil ∧ x = pencil

def cost_3_pens_pencils (pen pencil total_cost : ℕ) : Prop :=
  total_cost = 3 * pen + 7 * pencil  -- assuming "some pencils" translates to 7 pencils for this demonstration

def total_cost_dozen_pens (pen total_cost : ℕ) : Prop :=
  total_cost = 12 * pen

-- proving the main statement from conditions
theorem main_theorem (pen pencil total_cost : ℕ) (x : ℕ) 
  (h1 : cost_ratio_pen_pencil x)
  (h2 : cost_3_pens_pencils (5 * x) x 100)
  (h3 : total_cost_dozen_pens (5 * x) 300) :
  total_cost = 300 :=
by
  sorry

end main_theorem_l99_99403


namespace number_of_students_taking_statistics_l99_99683

theorem number_of_students_taking_statistics
  (total_students : ℕ)
  (history_students : ℕ)
  (history_or_statistics : ℕ)
  (history_only : ℕ)
  (history_and_statistics : ℕ := history_students - history_only)
  (statistics_only : ℕ := history_or_statistics - history_and_statistics - history_only)
  (statistics_students : ℕ := history_and_statistics + statistics_only) :
  total_students = 90 → history_students = 36 → history_or_statistics = 59 → history_only = 29 →
    statistics_students = 30 :=
by
  intros
  -- Proof goes here but is omitted.
  sorry

end number_of_students_taking_statistics_l99_99683


namespace sandy_correct_sums_l99_99218

theorem sandy_correct_sums
  (c i : ℕ)
  (h1 : c + i = 30)
  (h2 : 3 * c - 2 * i = 45) :
  c = 21 :=
by
  sorry

end sandy_correct_sums_l99_99218


namespace min_value_l99_99949

theorem min_value (m n : ℝ) (h1 : 2 * m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, (2 * m + n = 1 → m > 0 → n > 0 → y = (1 / m) + (1 / n) → y ≥ x)) :=
by
  sorry

end min_value_l99_99949


namespace solve_for_x_l99_99208

theorem solve_for_x :
  ∀ (x y : ℚ), (3 * x - 4 * y = 8) → (2 * x + 3 * y = 1) → x = 28 / 17 :=
by
  intros x y h1 h2
  sorry

end solve_for_x_l99_99208


namespace number_of_correct_judgments_is_zero_l99_99718

theorem number_of_correct_judgments_is_zero :
  (¬ ∀ (x : ℚ), -x ≠ |x|) ∧
  (¬ ∀ (x y : ℚ), -x = y → y = 1 / x) ∧
  (¬ ∀ (x y : ℚ), |x| = |y| → x = y) →
  0 = 0 :=
by
  intros h
  exact rfl

end number_of_correct_judgments_is_zero_l99_99718


namespace arbitrarily_large_ratios_l99_99749

open Nat

theorem arbitrarily_large_ratios (a : ℕ → ℕ) (h_distinct: ∀ m n, m ≠ n → a m ≠ a n)
  (h_no_100_ones: ∀ n, ¬ (∃ k, a n / 10^k % 10^100 = 10^100 - 1)):
  ∀ M : ℕ, ∃ n : ℕ, a n / n ≥ M :=
by
  sorry

end arbitrarily_large_ratios_l99_99749


namespace smallest_integer_y_l99_99728

theorem smallest_integer_y (y : ℤ) (h : 7 - 5 * y < 22) : y ≥ -2 :=
by sorry

end smallest_integer_y_l99_99728


namespace part1_part2_l99_99893

open Real

noncomputable def a_value := 2 * sqrt 2

noncomputable def line_cartesian_eqn (x y : ℝ) : Prop :=
  x + y - 4 = 0

noncomputable def point_on_line (ρ θ : ℝ) :=
  ρ * cos (θ - π / 4) = a_value

noncomputable def curve_param_eqns (θ : ℝ) : (ℝ × ℝ) :=
  (sqrt 3 * cos θ, sin θ)

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 4) / sqrt 2

theorem part1 (P : ℝ × ℝ) (ρ θ : ℝ) : 
  P = (4, π / 2) ∧ point_on_line ρ θ → 
  a_value = 2 * sqrt 2 ∧ line_cartesian_eqn 4 (4 * tan (π / 4)) :=
sorry

theorem part2 :
  (∀ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) ≤ 3 * sqrt 2) ∧
  (∃ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) = 3 * sqrt 2) :=
sorry

end part1_part2_l99_99893


namespace perfect_square_transformation_l99_99476

theorem perfect_square_transformation (a : ℤ) :
  (∃ x y : ℤ, x^2 + a = y^2) ↔ 
  ∃ α β : ℤ, α * β = a ∧ (α % 2 = β % 2) ∧ 
  ∃ x y : ℤ, x = (β - α) / 2 ∧ y = (β + α) / 2 :=
by
  sorry

end perfect_square_transformation_l99_99476


namespace compute_expression_l99_99797

theorem compute_expression : 20 * (150 / 3 + 36 / 4 + 4 / 25 + 2) = 1223 + 1/5 :=
by
  sorry

end compute_expression_l99_99797


namespace common_difference_l99_99396

variable (a : ℕ → ℝ)

def arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ a1, ∀ n, a n = a1 + (n - 1) * d

def geometric_sequence (a1 a2 a5 : ℝ) : Prop :=
  a1 * (a1 + 4 * (a2 - a1)) = (a2 - a1)^2

theorem common_difference {d : ℝ} (hd : d ≠ 0)
  (h_arith : arithmetic a d)
  (h_sum : a 1 + a 2 + a 5 = 13)
  (h_geom : geometric_sequence (a 1) (a 2) (a 5)) :
  d = 2 :=
sorry

end common_difference_l99_99396


namespace decimal_digits_of_fraction_l99_99304

noncomputable def fraction : ℚ := 987654321 / (2 ^ 30 * 5 ^ 2)

theorem decimal_digits_of_fraction :
  ∃ n ≥ 30, fraction = (987654321 / 10^2) / 2^28 := sorry

end decimal_digits_of_fraction_l99_99304


namespace infinite_series_sum_l99_99566

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * (n + 1) - 3) / 3 ^ (n + 1)) = 13 / 8 :=
by sorry

end infinite_series_sum_l99_99566


namespace verification_equation_3_conjecture_general_equation_l99_99689

theorem verification_equation_3 : 
  4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15)) :=
sorry

theorem conjecture :
  Real.sqrt (5 * (5 / 24)) = 5 * Real.sqrt (5 / 24) :=
sorry

theorem general_equation (n : ℕ) (h : 2 ≤ n) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) :=
sorry

end verification_equation_3_conjecture_general_equation_l99_99689


namespace interval_monotonically_increasing_range_g_l99_99003

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sqrt 3) * Real.sin (x + (Real.pi / 4)) * Real.cos (x + (Real.pi / 4)) + Real.sin (2 * x) - 1

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + (2 * Real.pi / 3)) - 1

theorem interval_monotonically_increasing :
  ∃ (k : ℤ), ∀ (x : ℝ), (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) → 0 ≤ deriv f x :=
sorry

theorem range_g (m : ℝ) : 
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → g x = m ↔ -3 ≤ m ∧ m ≤ Real.sqrt 3 - 1 :=
sorry

end interval_monotonically_increasing_range_g_l99_99003


namespace xyz_inequality_l99_99093

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := by
  sorry

end xyz_inequality_l99_99093


namespace intersect_complement_eq_l99_99488

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}
def comp_B : Set ℕ := U \ B

theorem intersect_complement_eq :
  A ∩ comp_B = {4, 5} := by
  sorry

end intersect_complement_eq_l99_99488


namespace fraction_uninterested_students_interested_l99_99366

theorem fraction_uninterested_students_interested 
  (students : Nat)
  (interest_ratio : ℚ)
  (express_interest_ratio_if_interested : ℚ)
  (express_disinterest_ratio_if_not_interested : ℚ) 
  (h1 : students > 0)
  (h2 : interest_ratio = 0.70)
  (h3 : express_interest_ratio_if_interested = 0.75)
  (h4 : express_disinterest_ratio_if_not_interested = 0.85) :
  let interested_students := students * interest_ratio
  let not_interested_students := students * (1 - interest_ratio)
  let express_interest_and_interested := interested_students * express_interest_ratio_if_interested
  let not_express_interest_and_interested := interested_students * (1 - express_interest_ratio_if_interested)
  let express_disinterest_and_not_interested := not_interested_students * express_disinterest_ratio_if_not_interested
  let express_interest_and_not_interested := not_interested_students * (1 - express_disinterest_ratio_if_not_interested)
  let not_express_interest_total := not_express_interest_and_interested + express_disinterest_and_not_interested
  let fraction := not_express_interest_and_interested / not_express_interest_total
  fraction = 0.407 := 
by
  sorry

end fraction_uninterested_students_interested_l99_99366


namespace calc_value_l99_99856

def f (x : ℤ) : ℤ := x^2 + 5 * x + 4
def g (x : ℤ) : ℤ := 2 * x - 3

theorem calc_value :
  f (g (-3)) - 2 * g (f 2) = -26 := by
  sorry

end calc_value_l99_99856


namespace intersection_sum_l99_99879

noncomputable def f (x : ℝ) : ℝ := 5 - (x - 1) ^ 2 / 3

theorem intersection_sum :
  ∃ a b : ℝ, f a = f (a - 4) ∧ b = f a ∧ a + b = 16 / 3 :=
sorry

end intersection_sum_l99_99879


namespace ArianaBoughtTulips_l99_99589

theorem ArianaBoughtTulips (total_flowers : ℕ) (fraction_roses : ℚ) (carnations : ℕ) 
    (h_total : total_flowers = 40) (h_fraction : fraction_roses = 2/5) (h_carnations : carnations = 14) : 
    total_flowers - (total_flowers * fraction_roses + carnations) = 10 := by
  sorry

end ArianaBoughtTulips_l99_99589


namespace dog_food_consumption_per_meal_l99_99172

theorem dog_food_consumption_per_meal
  (dogs : ℕ) (meals_per_day : ℕ) (total_food_kg : ℕ) (days : ℕ)
  (h_dogs : dogs = 4) (h_meals_per_day : meals_per_day = 2)
  (h_total_food_kg : total_food_kg = 100) (h_days : days = 50) :
  (total_food_kg * 1000 / days / meals_per_day / dogs) = 250 :=
by
  sorry

end dog_food_consumption_per_meal_l99_99172


namespace second_daily_rate_l99_99373

noncomputable def daily_rate_sunshine : ℝ := 17.99
noncomputable def mileage_cost_sunshine : ℝ := 0.18
noncomputable def mileage_cost_second : ℝ := 0.16
noncomputable def distance : ℝ := 48.0

theorem second_daily_rate (daily_rate_second : ℝ) : 
  daily_rate_sunshine + (mileage_cost_sunshine * distance) = 
  daily_rate_second + (mileage_cost_second * distance) → 
  daily_rate_second = 18.95 :=
by 
  sorry

end second_daily_rate_l99_99373


namespace factory_employees_l99_99960

def num_employees (n12 n14 n17 : ℕ) : ℕ := n12 + n14 + n17

def total_cost (n12 n14 n17 : ℕ) : ℕ := 
    (200 * 12 * 8) + (40 * 14 * 8) + (n17 * 17 * 8)

theorem factory_employees (n17 : ℕ) 
    (h_cost : total_cost 200 40 n17 = 31840) : 
    num_employees 200 40 n17 = 300 := 
by 
    sorry

end factory_employees_l99_99960


namespace range_of_a_l99_99184

theorem range_of_a (a : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ (a^2 * x - 2 * a + 1 = 0)) ↔ (a > 1/2 ∧ a ≠ 1) :=
by
  sorry

end range_of_a_l99_99184


namespace find_n_l99_99828

def alpha (n : ℕ) : ℚ := ((n - 2) * 180) / n
def alpha_plus_3 (n : ℕ) : ℚ := ((n + 1) * 180) / (n + 3)
def alpha_minus_2 (n : ℕ) : ℚ := ((n - 4) * 180) / (n - 2)

theorem find_n (n : ℕ) (h : alpha_plus_3 n - alpha n = alpha n - alpha_minus_2 n) : n = 12 :=
by
  -- The proof will be added here
  sorry

end find_n_l99_99828


namespace number_of_girls_l99_99130

variable (G B : ℕ)

theorem number_of_girls (h1 : G + B = 2000)
    (h2 : 0.28 * (B : ℝ) + 0.32 * (G : ℝ) = 596) : 
    G = 900 := 
sorry

end number_of_girls_l99_99130


namespace intersection_point_l99_99415

theorem intersection_point :
  ∃ (x y : ℝ), (y = 2 * x) ∧ (x + y = 3) ∧ (x = 1) ∧ (y = 2) := 
by
  sorry

end intersection_point_l99_99415


namespace sphere_volume_increase_factor_l99_99563

theorem sphere_volume_increase_factor (r : Real) : 
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  V_increased / V_original = 8 :=
by
  -- Definitions of volumes
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  -- Volume ratio
  have h : V_increased / V_original = 8 := sorry
  exact h

end sphere_volume_increase_factor_l99_99563


namespace sum_mod_17_l99_99744

theorem sum_mod_17 : (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 17 = 2 :=
by
  sorry

end sum_mod_17_l99_99744


namespace system_solution_ratio_l99_99936

theorem system_solution_ratio (x y z : ℝ) (h_xyz_nonzero: x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h1 : x + (95/9)*y + 4*z = 0) (h2 : 4*x + (95/9)*y - 3*z = 0) (h3 : 3*x + 5*y - 4*z = 0) :
  (x * z) / (y ^ 2) = 175 / 81 := 
by sorry

end system_solution_ratio_l99_99936


namespace exponentiation_comparison_l99_99623

theorem exponentiation_comparison :
  1.7 ^ 0.3 > 0.9 ^ 0.3 :=
by sorry

end exponentiation_comparison_l99_99623


namespace pie_shop_revenue_l99_99219

noncomputable def revenue_day1 := 5 * 6 * 12 + 6 * 6 * 8 + 7 * 6 * 10
noncomputable def revenue_day2 := 6 * 6 * 15 + 7 * 6 * 10 + 8 * 6 * 14
noncomputable def revenue_day3 := 4 * 6 * 18 + 7 * 6 * 7 + 9 * 6 * 13
noncomputable def total_revenue := revenue_day1 + revenue_day2 + revenue_day3

theorem pie_shop_revenue : total_revenue = 4128 := by
  sorry

end pie_shop_revenue_l99_99219


namespace scientific_notation_932700_l99_99048

theorem scientific_notation_932700 : 932700 = 9.327 * 10^5 :=
sorry

end scientific_notation_932700_l99_99048


namespace percentage_of_exceedance_l99_99884

theorem percentage_of_exceedance (x p : ℝ) (h : x = (p / 100) * x + 52.8) (hx : x = 60) : p = 12 :=
by 
  sorry

end percentage_of_exceedance_l99_99884


namespace total_pencils_correct_l99_99685

def pencils_per_child := 4
def num_children := 8
def total_pencils := pencils_per_child * num_children

theorem total_pencils_correct : total_pencils = 32 := by
  sorry

end total_pencils_correct_l99_99685


namespace parabola_expression_correct_area_triangle_ABM_correct_l99_99512

-- Given conditions
def pointA : ℝ × ℝ := (-1, 0)
def pointB : ℝ × ℝ := (3, 0)
def pointC : ℝ × ℝ := (0, 3)

-- Analytical expression of the parabola as y = -x^2 + 2x + 3
def parabola_eqn (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Definition of the vertex M of the parabola (derived from calculations)
def vertexM : ℝ × ℝ := (1, 4)

-- Calculation of distance AB
def distance_AB : ℝ := 4

-- Calculation of area of triangle ABM
def triangle_area_ABM : ℝ := 8

theorem parabola_expression_correct :
  (∀ x y, (y = parabola_eqn x ↔ (parabola_eqn x = y))) ∧
  (parabola_eqn pointC.1 = pointC.2) :=
by
  sorry

theorem area_triangle_ABM_correct :
  (1 / 2 * distance_AB * vertexM.2 = 8) :=
by
  sorry

end parabola_expression_correct_area_triangle_ABM_correct_l99_99512


namespace total_weight_of_balls_l99_99838

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  weight_blue + weight_brown = 9.12 :=
by
  sorry

end total_weight_of_balls_l99_99838


namespace milk_price_increase_day_l99_99830

theorem milk_price_increase_day (total_cost : ℕ) (old_price : ℕ) (new_price : ℕ) (days : ℕ) (x : ℕ)
    (h1 : old_price = 1500)
    (h2 : new_price = 1600)
    (h3 : days = 30)
    (h4 : total_cost = 46200)
    (h5 : (x - 1) * old_price + (days + 1 - x) * new_price = total_cost) :
  x = 19 :=
by
  sorry

end milk_price_increase_day_l99_99830


namespace delivery_truck_speed_l99_99952

theorem delivery_truck_speed :
  ∀ d t₁ t₂: ℝ,
    (t₁ = 15 / 60) ∧ (t₂ = -15 / 60) ∧ 
    (t₁ = d / 20 - 1 / 4) ∧ (t₂ = d / 60 + 1 / 4) →
    (d = 15) →
    (t = 1 / 2) →
    ( ∃ v: ℝ, t = d / v ∧ v = 30 ) :=
by sorry

end delivery_truck_speed_l99_99952


namespace has_four_digits_l99_99649

def least_number_divisible (n: ℕ) : Prop := 
  n = 9600 ∧ 
  (∃ k1 k2 k3 k4: ℕ, n = 15 * k1 ∧ n = 25 * k2 ∧ n = 40 * k3 ∧ n = 75 * k4)

theorem has_four_digits : ∀ n: ℕ, least_number_divisible n → (Nat.digits 10 n).length = 4 :=
by
  intros n h
  sorry

end has_four_digits_l99_99649


namespace random_event_sum_gt_six_l99_99411

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def selection (s : List ℕ) := s.length = 3 ∧ s ⊆ numbers

def sum_is_greater_than_six (s : List ℕ) : Prop := s.sum > 6

theorem random_event_sum_gt_six :
  ∀ (s : List ℕ), selection s → (sum_is_greater_than_six s ∨ ¬ sum_is_greater_than_six s) := 
by
  intros s h
  -- Proof omitted
  sorry

end random_event_sum_gt_six_l99_99411


namespace cos_even_function_l99_99462

theorem cos_even_function : ∀ x : ℝ, Real.cos (-x) = Real.cos x := 
by 
  sorry

end cos_even_function_l99_99462


namespace population_time_interval_l99_99175

theorem population_time_interval (T : ℕ) 
  (birth_rate : ℕ) (death_rate : ℕ) (net_increase_day : ℕ) (seconds_in_day : ℕ)
  (h_birth_rate : birth_rate = 8) 
  (h_death_rate : death_rate = 6) 
  (h_net_increase_day : net_increase_day = 86400)
  (h_seconds_in_day : seconds_in_day = 86400) : 
  T = 2 := sorry

end population_time_interval_l99_99175


namespace combination_value_l99_99737

theorem combination_value (m : ℕ) (h : (1 / (Nat.choose 5 m) - 1 / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m))) : 
    Nat.choose 8 m = 28 := 
sorry

end combination_value_l99_99737


namespace ms_tom_investment_l99_99714

def invested_amounts (X Y : ℝ) : Prop :=
  X + Y = 100000 ∧ 0.17 * Y = 0.23 * X + 200 

theorem ms_tom_investment (X Y : ℝ) (h : invested_amounts X Y) : X = 42000 :=
by
  sorry

end ms_tom_investment_l99_99714


namespace hyperbola_eccentricity_l99_99928

def isHyperbolaWithEccentricity (e : ℝ) : Prop :=
  ∃ (a b : ℝ), a = 4 * b ∧ e = (Real.sqrt (a^2 + b^2)) / a

theorem hyperbola_eccentricity : isHyperbolaWithEccentricity (Real.sqrt 17 / 4) :=
sorry

end hyperbola_eccentricity_l99_99928


namespace boxes_difference_l99_99084

theorem boxes_difference (white_balls red_balls balls_per_box : ℕ)
  (h_white : white_balls = 30)
  (h_red : red_balls = 18)
  (h_box : balls_per_box = 6) :
  (white_balls / balls_per_box) - (red_balls / balls_per_box) = 2 :=
by 
  sorry

end boxes_difference_l99_99084


namespace line_circle_intersection_l99_99810

theorem line_circle_intersection (k : ℝ) :
  ∃ x y : ℝ, y = k * (x + 1 / 2) ∧ x^2 + y^2 = 1 :=
sorry

end line_circle_intersection_l99_99810


namespace inequality_range_l99_99701

theorem inequality_range (k : ℝ) : (∀ x : ℝ, abs (x + 1) - abs (x - 2) > k) → k < -3 :=
by
  sorry

end inequality_range_l99_99701


namespace three_digit_number_is_473_l99_99259

theorem three_digit_number_is_473 (x y z : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) (h5 : 0 ≤ z) (h6 : z ≤ 9)
  (h7 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 99)
  (h8 : x + y + z = 14)
  (h9 : x + z = y) : 100 * x + 10 * y + z = 473 :=
by
  sorry

end three_digit_number_is_473_l99_99259


namespace elise_initial_money_l99_99030

theorem elise_initial_money :
  ∃ (X : ℤ), X + 13 - 2 - 18 = 1 ∧ X = 8 :=
by
  sorry

end elise_initial_money_l99_99030


namespace expand_and_simplify_l99_99022

theorem expand_and_simplify (x : ℝ) : 
  (2 * x + 6) * (x + 10) = 2 * x^2 + 26 * x + 60 :=
sorry

end expand_and_simplify_l99_99022


namespace volunteer_comprehensive_score_l99_99473

theorem volunteer_comprehensive_score :
  let written_score := 90
  let trial_score := 94
  let interview_score := 92
  let written_weight := 0.30
  let trial_weight := 0.50
  let interview_weight := 0.20
  (written_score * written_weight + trial_score * trial_weight + interview_score * interview_weight = 92.4) := by
  sorry

end volunteer_comprehensive_score_l99_99473


namespace min_chord_length_eq_l99_99800

-- Define the Circle C with center (1, 2) and radius 5
def isCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

-- Define the Line l parameterized by m
def isLine (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that the minimal chord length intercepted by the circle occurs when the line l is 2x - y - 5 = 0
theorem min_chord_length_eq (x y : ℝ) : 
  (∀ m, isLine m x y → isCircle x y) → isLine 0 x y :=
sorry

end min_chord_length_eq_l99_99800


namespace todd_ratio_boss_l99_99036

theorem todd_ratio_boss
  (total_cost : ℕ)
  (boss_contribution : ℕ)
  (employees_contribution : ℕ)
  (num_employees : ℕ)
  (each_employee_pay : ℕ) 
  (total_contributed : ℕ)
  (todd_contribution : ℕ) :
  total_cost = 100 →
  boss_contribution = 15 →
  num_employees = 5 →
  each_employee_pay = 11 →
  total_contributed = num_employees * each_employee_pay + boss_contribution →
  todd_contribution = total_cost - total_contributed →
  (todd_contribution : ℚ) / (boss_contribution : ℚ) = 2 := by
  sorry

end todd_ratio_boss_l99_99036


namespace cans_needed_eq_l99_99198

axiom Paula_initial_rooms : ℕ
axiom Paula_lost_cans : ℕ
axiom Paula_after_loss_rooms : ℕ
axiom cans_for_25_rooms : ℕ

theorem cans_needed_eq :
  Paula_initial_rooms = 30 →
  Paula_lost_cans = 3 →
  Paula_after_loss_rooms = 25 →
  cans_for_25_rooms = 15 :=
by
  intros
  sorry

end cans_needed_eq_l99_99198


namespace middle_number_is_10_l99_99818

theorem middle_number_is_10 (x y z : ℤ) (hx : x < y) (hy : y < z) 
    (h1 : x + y = 18) (h2 : x + z = 25) (h3 : y + z = 27) : y = 10 :=
by 
  sorry

end middle_number_is_10_l99_99818


namespace does_not_pass_first_quadrant_l99_99470

def linear_function (x : ℝ) : ℝ := -3 * x - 2

def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem does_not_pass_first_quadrant : ∀ (x : ℝ), ¬ in_first_quadrant x (linear_function x) := 
sorry

end does_not_pass_first_quadrant_l99_99470


namespace decompose_96_l99_99846

theorem decompose_96 (a b : ℤ) (h1 : a * b = 96) (h2 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) ∨ (a = -8 ∧ b = -12) ∨ (a = -12 ∧ b = -8) :=
by
  sorry

end decompose_96_l99_99846


namespace fraction_lt_sqrt2_bound_l99_99121

theorem fraction_lt_sqrt2_bound (m n : ℕ) (h : (m : ℝ) / n < Real.sqrt 2) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * (n * n))) :=
sorry

end fraction_lt_sqrt2_bound_l99_99121


namespace ratio_of_men_to_women_l99_99074

/-- Define the number of men and women on a co-ed softball team. -/
def number_of_men : ℕ := 8
def number_of_women : ℕ := 12

/--
  Given:
  1. There are 4 more women than men.
  2. The total number of players is 20.
  Prove that the ratio of men to women is 2 : 3.
-/
theorem ratio_of_men_to_women 
  (h1 : number_of_women = number_of_men + 4)
  (h2 : number_of_men + number_of_women = 20) :
  (number_of_men * 3) = (number_of_women * 2) :=
by
  have h3 : number_of_men = 8 := by sorry
  have h4 : number_of_women = 12 := by sorry
  sorry

end ratio_of_men_to_women_l99_99074


namespace find_square_subtraction_l99_99999

theorem find_square_subtraction (x y : ℝ) (h1 : x = Real.sqrt 5) (h2 : y = Real.sqrt 2) : (x - y)^2 = 7 - 2 * Real.sqrt 10 :=
by
  sorry

end find_square_subtraction_l99_99999


namespace exponent_equality_l99_99176

theorem exponent_equality (n : ℕ) : (4^8 = 4^n) → (n = 8) := by
  intro h
  sorry

end exponent_equality_l99_99176


namespace brother_books_total_l99_99210

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

end brother_books_total_l99_99210


namespace sum_of_ages_l99_99150

variable {P M Mo : ℕ}

-- Conditions
axiom ratio1 : 3 * M = 5 * P
axiom ratio2 : 3 * Mo = 5 * M
axiom age_difference : Mo - P = 80

-- Statement that needs to be proved
theorem sum_of_ages : P + M + Mo = 245 := by
  sorry

end sum_of_ages_l99_99150


namespace sequence_elements_are_prime_l99_99028

variable {a : ℕ → ℕ} {p : ℕ → ℕ}

def increasing_seq (f : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → f i < f j

def divisible_by_prime (a p : ℕ → ℕ) : Prop :=
  ∀ n, Prime (p n) ∧ p n ∣ a n

def satisfies_condition (a p : ℕ → ℕ) : Prop :=
  ∀ n k, a n - a k = p n - p k

theorem sequence_elements_are_prime (h1 : increasing_seq a) 
    (h2 : divisible_by_prime a p) 
    (h3 : satisfies_condition a p) :
    ∀ n, Prime (a n) :=
by 
  sorry

end sequence_elements_are_prime_l99_99028


namespace ratio_of_bases_l99_99598

theorem ratio_of_bases 
(AB CD : ℝ) 
(h_trapezoid : AB < CD) 
(h_AC : ∃ k : ℝ, k = 2 * CD ∧ k = AC) 
(h_altitude : AB = (D - foot)) : 
AB / CD = 3 := 
sorry

end ratio_of_bases_l99_99598


namespace brokerage_percentage_correct_l99_99767

noncomputable def brokerage_percentage (market_value : ℝ) (income : ℝ) (investment : ℝ) (nominal_rate : ℝ) : ℝ :=
  let face_value := (income * 100) / nominal_rate
  let market_price := (face_value * market_value) / 100
  let brokerage_amount := investment - market_price
  (brokerage_amount / investment) * 100

theorem brokerage_percentage_correct :
  brokerage_percentage 110.86111111111111 756 8000 10.5 = 0.225 :=
by
  sorry

end brokerage_percentage_correct_l99_99767


namespace parabola_opens_upward_l99_99992

theorem parabola_opens_upward (a : ℝ) (b : ℝ) (h : a > 0) : ∀ x : ℝ, 3*x^2 + 2 = a*x^2 + b → a = 3 ∧ b = 2 → ∀ x : ℝ, 3 * x^2 + 2 ≤ a * x^2 + b := 
by
  sorry

end parabola_opens_upward_l99_99992


namespace function_properties_l99_99285

noncomputable def f (x : ℝ) : ℝ := Real.sin (x * Real.cos x)

theorem function_properties :
  (f x = -f (-x)) ∧
  (∀ x, 0 < x ∧ x < Real.pi / 2 → 0 < f x) ∧
  ¬(∃ T, ∀ x, f (x + T) = f x) ∧
  (∀ n : ℤ, f (n * Real.pi) = 0) := 
by
  sorry

end function_properties_l99_99285


namespace PQR_positive_iff_P_Q_R_positive_l99_99870

noncomputable def P (a b c : ℝ) : ℝ := a + b - c
noncomputable def Q (a b c : ℝ) : ℝ := b + c - a
noncomputable def R (a b c : ℝ) : ℝ := c + a - b

theorem PQR_positive_iff_P_Q_R_positive (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c * Q a b c * R a b c > 0) ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end PQR_positive_iff_P_Q_R_positive_l99_99870


namespace time_to_paint_one_room_l99_99037

theorem time_to_paint_one_room (total_rooms : ℕ) (rooms_painted : ℕ) (time_remaining : ℕ) (rooms_left : ℕ) :
  total_rooms = 9 ∧ rooms_painted = 5 ∧ time_remaining = 32 ∧ rooms_left = total_rooms - rooms_painted → time_remaining / rooms_left = 8 :=
by
  intros h
  sorry

end time_to_paint_one_room_l99_99037


namespace two_digit_number_is_24_l99_99455

-- Definitions from the problem conditions
def is_two_digit_number (n : ℕ) := n ≥ 10 ∧ n < 100

def tens_digit (n : ℕ) := n / 10

def ones_digit (n : ℕ) := n % 10

def condition_2 (n : ℕ) := tens_digit n = ones_digit n - 2

def condition_3 (n : ℕ) := 3 * tens_digit n * ones_digit n = n

-- The proof problem statement
theorem two_digit_number_is_24 (n : ℕ) (h1 : is_two_digit_number n)
  (h2 : condition_2 n) (h3 : condition_3 n) : n = 24 := by
  sorry

end two_digit_number_is_24_l99_99455


namespace area_enclosed_by_equation_l99_99652

theorem area_enclosed_by_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 10 * y = -20) → (∃ r : ℝ, r^2 = 9 ∧ ∃ c : ℝ × ℝ, (∃ a b, (x - a)^2 + (y - b)^2 = r^2)) :=
by
  sorry

end area_enclosed_by_equation_l99_99652


namespace units_digit_of_expression_l99_99522

theorem units_digit_of_expression :
  (3 * 19 * 1981 - 3^4) % 10 = 6 :=
sorry

end units_digit_of_expression_l99_99522


namespace divisibility_by_2k_l99_99280

-- Define the sequence according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ ∀ n, 2 ≤ n → a n = 2 * a (n - 1) + a (n - 2)

-- The theorem to be proved
theorem divisibility_by_2k (a : ℕ → ℤ) (k : ℕ) (n : ℕ)
  (h : seq a) :
  2^k ∣ a n ↔ 2^k ∣ n :=
sorry

end divisibility_by_2k_l99_99280


namespace exists_projectile_time_l99_99874

noncomputable def projectile_time := 
  ∃ t1 t2 : ℝ, (-4.9 * t1^2 + 31 * t1 - 40 = 0) ∧ ((abs (t1 - 1.8051) < 0.001) ∨ (abs (t2 - 4.5319) < 0.001))

theorem exists_projectile_time : projectile_time := 
sorry

end exists_projectile_time_l99_99874


namespace max_fraction_l99_99159

theorem max_fraction (a b : ℕ) (h1 : a + b = 101) (h2 : (a : ℚ) / b ≤ 1 / 3) : (a, b) = (25, 76) :=
sorry

end max_fraction_l99_99159


namespace difference_of_squares_l99_99521

theorem difference_of_squares (x : ℤ) (h : x^2 = 1521) : (x + 1) * (x - 1) = 1520 := by
  sorry

end difference_of_squares_l99_99521


namespace parallelogram_sides_are_parallel_l99_99116

theorem parallelogram_sides_are_parallel 
  {a b c : ℤ} (h_area : c * (a^2 + b^2) = 2011 * b) : 
  (∃ k : ℤ, a = 2011 * k ∧ (b = 2011 ∨ b = -2011)) :=
by
  sorry

end parallelogram_sides_are_parallel_l99_99116


namespace red_ball_probability_l99_99695

theorem red_ball_probability 
  (red_balls : ℕ)
  (black_balls : ℕ)
  (total_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : black_balls = 9)
  (h3 : total_balls = red_balls + black_balls) :
  (red_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end red_ball_probability_l99_99695


namespace present_age_of_son_l99_99113

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 25) (h2 : F + 2 = 2 * (S + 2)) : S = 23 :=
by
  sorry

end present_age_of_son_l99_99113


namespace base_7_perfect_square_ab2c_l99_99490

-- Define the necessary conditions
def is_base_7_representation_of (n : ℕ) (a b c : ℕ) : Prop :=
  n = a * 7^3 + b * 7^2 + 2 * 7 + c

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Lean statement for the problem
theorem base_7_perfect_square_ab2c (n a b c : ℕ) (h1 : a ≠ 0) (h2 : is_base_7_representation_of n a b c) (h3 : is_perfect_square n) :
  c = 2 ∨ c = 3 ∨ c = 6 :=
  sorry

end base_7_perfect_square_ab2c_l99_99490


namespace ratio_of_sleep_l99_99907

theorem ratio_of_sleep (connor_sleep : ℝ) (luke_extra : ℝ) (puppy_sleep : ℝ) 
    (h1 : connor_sleep = 6)
    (h2 : luke_extra = 2)
    (h3 : puppy_sleep = 16) :
    puppy_sleep / (connor_sleep + luke_extra) = 2 := 
by 
  sorry

end ratio_of_sleep_l99_99907


namespace find_y_values_l99_99267

open Real

-- Problem statement as a Lean statement.
theorem find_y_values (x : ℝ) (hx : x^2 + 2 * (x / (x - 1)) ^ 2 = 20) :
  ∃ y : ℝ, (y = ((x - 1) ^ 3 * (x + 2)) / (2 * x - 1)) ∧ (y = 14 ∨ y = -56 / 3) := 
sorry

end find_y_values_l99_99267


namespace solve_for_x_l99_99639

theorem solve_for_x : ∃ x : ℚ, 6 * (2 * x + 3) - 4 = -3 * (2 - 5 * x) + 3 * x ∧ x = 10 / 3 := by
  sorry

end solve_for_x_l99_99639


namespace find_c_l99_99863

theorem find_c :
  ∃ c : ℝ, 0 < c ∧ ∀ line : ℝ, (∃ x y : ℝ, (x = 1 ∧ y = c) ∧ (x*x + y*y - 2*x - 2*y - 7 = 0)) ∧ (line = 1*x + 0 + y*c - 0) :=
sorry

end find_c_l99_99863


namespace minimum_value_of_f_l99_99122

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem minimum_value_of_f : ∃ x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end minimum_value_of_f_l99_99122


namespace cos_2alpha_zero_l99_99976

theorem cos_2alpha_zero (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
(h : Real.sin (2 * α) = Real.cos (Real.pi / 4 - α)) : 
  Real.cos (2 * α) = 0 :=
by
  sorry

end cos_2alpha_zero_l99_99976


namespace polygon_sides_l99_99494

theorem polygon_sides (n : ℕ) (h : 3 * n * (n * (n - 3)) = 300) : n = 10 :=
sorry

end polygon_sides_l99_99494


namespace component_unqualified_l99_99221

theorem component_unqualified :
  ∀ (φ : ℝ), (19.98 ≤ φ ∧ φ ≤ 20.02) → ¬(φ = 19.9) → True :=
by
  intro φ
  intro h
  intro h'
  -- skip proof
  sorry

end component_unqualified_l99_99221


namespace solution_set_inequality_l99_99320

theorem solution_set_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x, f x + (deriv^[2] f) x < 1) (h_f0 : f 0 = 2018) :
  ∀ x, x > 0 → f x < 2017 * Real.exp (-x) + 1 :=
by
  sorry

end solution_set_inequality_l99_99320


namespace polynomial_divisibility_l99_99969

theorem polynomial_divisibility (n : ℕ) (h : n > 2) : 
    (∀ k : ℕ, n = 3 * k + 1) ↔ ∃ (k : ℕ), n = 3 * k + 1 := 
sorry

end polynomial_divisibility_l99_99969


namespace fishbowl_count_l99_99029

def number_of_fishbowls (total_fish : ℕ) (fish_per_bowl : ℕ) : ℕ :=
  total_fish / fish_per_bowl

theorem fishbowl_count (h1 : 23 > 0) (h2 : 6003 % 23 = 0) :
  number_of_fishbowls 6003 23 = 261 :=
by
  -- proof goes here
  sorry

end fishbowl_count_l99_99029


namespace map_length_l99_99326

theorem map_length 
  (width : ℝ) (area : ℝ) 
  (h_width : width = 10) (h_area : area = 20) : 
  ∃ length : ℝ, area = width * length ∧ length = 2 :=
by 
  sorry

end map_length_l99_99326


namespace value_of_m_l99_99860

variable (a m : ℝ)
variable (h1 : a > 0)
variable (h2 : -a*m^2 + 2*a*m + 3 = 3)
variable (h3 : m ≠ 0)

theorem value_of_m : m = 2 :=
by
  sorry

end value_of_m_l99_99860


namespace probability_of_specific_combination_l99_99631

def total_shirts : ℕ := 3
def total_shorts : ℕ := 7
def total_socks : ℕ := 4
def total_clothes : ℕ := total_shirts + total_shorts + total_socks
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def favorable_outcomes : ℕ := (choose total_shirts 2) * (choose total_shorts 1) * (choose total_socks 1)
def total_outcomes : ℕ := choose total_clothes 4

theorem probability_of_specific_combination :
  favorable_outcomes / total_outcomes = 84 / 1001 :=
by
  -- Proof omitted
  sorry

end probability_of_specific_combination_l99_99631


namespace a_2016_is_neg1_l99_99010

noncomputable def a : ℕ → ℤ
| 0     => 0 -- Arbitrary value for n = 0 since sequences generally start from 1 in Lean
| 1     => 1
| 2     => 2
| n + 1 => a n - a (n - 1)

theorem a_2016_is_neg1 : a 2016 = -1 := sorry

end a_2016_is_neg1_l99_99010


namespace sprinter_speed_l99_99873

theorem sprinter_speed
  (distance : ℝ)
  (time : ℝ)
  (H1 : distance = 100)
  (H2 : time = 10) :
    (distance / time = 10) ∧
    ((distance / time) * 60 = 600) ∧
    (((distance / time) * 60 * 60) / 1000 = 36) :=
by
  sorry

end sprinter_speed_l99_99873


namespace max_value_fraction_l99_99065

theorem max_value_fraction (a b x y : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : a^x = 3) (h4 : b^y = 3) (h5 : a + b = 2 * Real.sqrt 3) :
  1/x + 1/y ≤ 1 :=
sorry

end max_value_fraction_l99_99065


namespace original_mixture_volume_l99_99158

theorem original_mixture_volume (x : ℝ) (h1 : 0.20 * x / (x + 3) = 1 / 6) : x = 15 :=
  sorry

end original_mixture_volume_l99_99158


namespace gear_q_revolutions_per_minute_l99_99400

noncomputable def gear_p_revolutions_per_minute : ℕ := 10

noncomputable def additional_revolutions : ℕ := 15

noncomputable def calculate_q_revolutions_per_minute
  (p_rev_per_min : ℕ) (additional_rev : ℕ) : ℕ :=
  2 * (p_rev_per_min / 2 + additional_rev)

theorem gear_q_revolutions_per_minute :
  calculate_q_revolutions_per_minute gear_p_revolutions_per_minute additional_revolutions = 40 :=
by
  sorry

end gear_q_revolutions_per_minute_l99_99400


namespace Davante_boys_count_l99_99492

def days_in_week := 7
def friends (days : Nat) := days * 2
def girls := 3
def boys (total_friends girls : Nat) := total_friends - girls

theorem Davante_boys_count :
  boys (friends days_in_week) girls = 11 :=
  by
    sorry

end Davante_boys_count_l99_99492


namespace simplify_problem_l99_99315

noncomputable def simplify_expression : ℝ :=
  let numer := (Real.sqrt 3 - 1) ^ (1 - Real.sqrt 2)
  let denom := (Real.sqrt 3 + 1) ^ (1 + Real.sqrt 2)
  numer / denom

theorem simplify_problem :
  simplify_expression = 2 ^ (1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3) :=
by
  sorry

end simplify_problem_l99_99315


namespace axis_of_symmetry_of_f_l99_99755

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x + 1)

theorem axis_of_symmetry_of_f : (axis_of_symmetry : ℝ) = -1 :=
by
  sorry

end axis_of_symmetry_of_f_l99_99755


namespace a_squared_plus_b_squared_eq_sqrt_11_l99_99486

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_condition : a * b * (a - b) = 1

theorem a_squared_plus_b_squared_eq_sqrt_11 : a^2 + b^2 = Real.sqrt 11 := by
  sorry

end a_squared_plus_b_squared_eq_sqrt_11_l99_99486


namespace g_54_l99_99310

def g : ℕ → ℤ := sorry

axiom g_multiplicative (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_6 : g 6 = 10
axiom g_18 : g 18 = 14

theorem g_54 : g 54 = 18 := by
  sorry

end g_54_l99_99310


namespace total_votes_cast_l99_99500

theorem total_votes_cast (b_votes c_votes total_votes : ℕ)
  (h1 : b_votes = 48)
  (h2 : c_votes = 35)
  (h3 : b_votes = (4 * total_votes) / 15) :
  total_votes = 180 :=
by
  sorry

end total_votes_cast_l99_99500


namespace total_gas_cost_l99_99794

def car_city_mpg : ℝ := 30
def car_highway_mpg : ℝ := 40
def city_miles : ℝ := 60 + 40 + 25
def highway_miles : ℝ := 200 + 150 + 180
def gas_price_per_gallon : ℝ := 3.00

theorem total_gas_cost : 
  (city_miles / car_city_mpg + highway_miles / car_highway_mpg) * gas_price_per_gallon = 52.25 := 
by
  sorry

end total_gas_cost_l99_99794


namespace abc_over_ab_bc_ca_l99_99183

variable {a b c : ℝ}

theorem abc_over_ab_bc_ca (h1 : ab / (a + b) = 2)
                          (h2 : bc / (b + c) = 5)
                          (h3 : ca / (c + a) = 7) :
        abc / (ab + bc + ca) = 35 / 44 :=
by
  -- The proof would go here.
  sorry

end abc_over_ab_bc_ca_l99_99183


namespace geometric_sequence_b_value_l99_99469

noncomputable def b_value (b : ℝ) : Prop :=
  ∃ s : ℝ, 180 * s = b ∧ b * s = 75 / 32 ∧ b > 0

theorem geometric_sequence_b_value (b : ℝ) : b_value b → b = 20.542 :=
by
  sorry

end geometric_sequence_b_value_l99_99469


namespace ken_change_l99_99789

theorem ken_change (cost_per_pound : ℕ) (quantity : ℕ) (amount_paid : ℕ) (total_cost : ℕ) (change : ℕ) 
(h1 : cost_per_pound = 7)
(h2 : quantity = 2)
(h3 : amount_paid = 20)
(h4 : total_cost = cost_per_pound * quantity)
(h5 : change = amount_paid - total_cost) : change = 6 :=
by 
  sorry

end ken_change_l99_99789


namespace inequality_of_products_l99_99190

theorem inequality_of_products
  (a b c d : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hd : 0 < d)
  (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_of_products_l99_99190


namespace zephyr_island_population_capacity_reach_l99_99278

-- Definitions for conditions
def acres := 30000
def acres_per_person := 2
def initial_year := 2023
def initial_population := 500
def population_growth_rate := 4
def growth_period := 20

-- Maximum population supported by the island
def max_population := acres / acres_per_person

-- Function to calculate population after a given number of years
def population (years : ℕ) : ℕ := initial_population * (population_growth_rate ^ (years / growth_period))

-- The Lean statement to prove that the population will reach or exceed max_capacity in 60 years
theorem zephyr_island_population_capacity_reach : ∃ t : ℕ, t ≤ 60 ∧ population t ≥ max_population :=
by
  sorry

end zephyr_island_population_capacity_reach_l99_99278


namespace probability_of_no_adjacent_standing_is_123_over_1024_l99_99092

def total_outcomes : ℕ := 2 ^ 10

 -- Define the recursive sequence a_n
def a : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => a (n + 1) + a n

lemma a_10_val : a 10 = 123 := by
  sorry

def probability_no_adjacent_standing (n : ℕ): ℚ :=
  a n / total_outcomes

theorem probability_of_no_adjacent_standing_is_123_over_1024 :
  probability_no_adjacent_standing 10 = 123 / 1024 := by
  rw [probability_no_adjacent_standing, total_outcomes, a_10_val]
  norm_num

end probability_of_no_adjacent_standing_is_123_over_1024_l99_99092


namespace stratified_sampling_females_l99_99691

theorem stratified_sampling_females :
  let total_employees := 200
  let male_employees := 120
  let female_employees := 80
  let sample_size := 20
  number_of_female_in_sample = (female_employees / total_employees) * sample_size := by
  sorry

end stratified_sampling_females_l99_99691


namespace Ruby_math_homework_l99_99324

theorem Ruby_math_homework : 
  ∃ M : ℕ, ∃ R : ℕ, R = 2 ∧ 5 * M + 9 * R = 48 ∧ M = 6 := by
  sorry

end Ruby_math_homework_l99_99324


namespace capture_probability_correct_l99_99626

structure ProblemConditions where
  rachel_speed : ℕ -- seconds per lap
  robert_speed : ℕ -- seconds per lap
  rachel_direction : Bool -- true if counterclockwise, false if clockwise
  robert_direction : Bool -- true if counterclockwise, false if clockwise
  start_time : ℕ -- 0 seconds
  end_time_start : ℕ -- 900 seconds
  end_time_end : ℕ -- 1200 seconds
  photo_coverage_fraction : ℚ -- fraction of the track covered by the photo

noncomputable def probability_capture_in_photo (pc : ProblemConditions) : ℚ :=
  sorry -- define and prove the exact probability

-- Given the conditions in the problem
def problem_instance : ProblemConditions :=
{
  rachel_speed := 120,
  robert_speed := 100,
  rachel_direction := true,
  robert_direction := false,
  start_time := 0,
  end_time_start := 900,
  end_time_end := 1200,
  photo_coverage_fraction := 1/3
}

-- The theorem statement we are asked to prove
theorem capture_probability_correct :
  probability_capture_in_photo problem_instance = 1/9 :=
sorry

end capture_probability_correct_l99_99626


namespace avg_age_when_youngest_born_l99_99392

theorem avg_age_when_youngest_born
  (num_people : ℕ) (avg_age_now : ℝ) (youngest_age_now : ℝ) (sum_ages_others_then : ℝ) 
  (h1 : num_people = 7) 
  (h2 : avg_age_now = 30) 
  (h3 : youngest_age_now = 6) 
  (h4 : sum_ages_others_then = 150) :
  (sum_ages_others_then / num_people) = 21.43 :=
by
  sorry

end avg_age_when_youngest_born_l99_99392


namespace x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l99_99505

theorem x_less_than_2_necessary_not_sufficient (x : ℝ) :
  (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) := sorry

theorem x_less_than_2_is_necessary_not_sufficient : 
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧ 
  (¬ ∀ x : ℝ, x < 2 → x^2 - 3*x + 2 < 0) := sorry

end x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l99_99505


namespace exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l99_99393

variable (m : ℝ)
def f (x : ℝ) : ℝ := x^2 + m*x + 1

theorem exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2 :
  (∃ x0 : ℝ, x0 > 0 ∧ f m x0 < 0) → m < -2 := by
  sorry

end exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l99_99393


namespace remainder_sum_l99_99669

theorem remainder_sum (n : ℤ) (h : n % 21 = 13) : (n % 3 + n % 7) = 7 := by
  sorry

end remainder_sum_l99_99669


namespace combined_average_score_girls_l99_99763

open BigOperators

variable (A a B b C c : ℕ) -- number of boys and girls at each school
variable (x : ℕ) -- common value for number of boys and girls

axiom Adams_HS : 74 * (A : ℤ) + 81 * (a : ℤ) = 77 * (A + a)
axiom Baker_HS : 83 * (B : ℤ) + 92 * (b : ℤ) = 86 * (B + b)
axiom Carter_HS : 78 * (C : ℤ) + 85 * (c : ℤ) = 80 * (C + c)

theorem combined_average_score_girls :
  (A = a ∧ B = b ∧ C = c) →
  (A = B ∧ B = C) →
  (81 * (A : ℤ) + 92 * (B : ℤ) + 85 * (C : ℤ)) / (A + B + C) = 86 := 
by
  intro h1 h2
  sorry

end combined_average_score_girls_l99_99763


namespace stock_decrease_required_l99_99033

theorem stock_decrease_required (x : ℝ) (h : x > 0) : 
  (∃ (p : ℝ), (1 - p) * 1.40 * x = x ∧ p * 100 = 28.57) :=
sorry

end stock_decrease_required_l99_99033


namespace problem_1_problem_2_l99_99562

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ :=
( sin θ, cos θ - 2 * sin θ )

def vec_b : ℝ × ℝ :=
( 1, 2 )

theorem problem_1 (θ : ℝ) (h : (cos θ - 2 * sin θ) / sin θ = 2) : tan θ = 1 / 4 :=
by {
  sorry
}

theorem problem_2 (θ : ℝ) (h1 : sin θ ^ 2 + (cos θ - 2 * sin θ) ^ 2 = 5) (h2 : 0 < θ) (h3 : θ < π) : θ = π / 2 ∨ θ = 3 * π / 4 :=
by {
  sorry
}

end problem_1_problem_2_l99_99562


namespace commodities_price_difference_l99_99667

theorem commodities_price_difference : 
  ∀ (C1 C2 : ℕ), 
    C1 = 477 → 
    C1 + C2 = 827 → 
    C1 - C2 = 127 :=
by
  intros C1 C2 h1 h2
  sorry

end commodities_price_difference_l99_99667


namespace oxygen_atoms_in_compound_l99_99255

-- Define given conditions as parameters in the problem.
def number_of_oxygen_atoms (molecular_weight : ℕ) (weight_Al : ℕ) (weight_H : ℕ) (weight_O : ℕ) (atoms_Al : ℕ) (atoms_H : ℕ) (weight : ℕ) : ℕ := 
  (weight - (atoms_Al * weight_Al + atoms_H * weight_H)) / weight_O

-- Define the actual problem using the defined conditions.
theorem oxygen_atoms_in_compound
  (molecular_weight : ℕ := 78) 
  (weight_Al : ℕ := 27) 
  (weight_H : ℕ := 1) 
  (weight_O : ℕ := 16) 
  (atoms_Al : ℕ := 1) 
  (atoms_H : ℕ := 3) : 
  number_of_oxygen_atoms molecular_weight weight_Al weight_H weight_O atoms_Al atoms_H molecular_weight = 3 := 
sorry

end oxygen_atoms_in_compound_l99_99255


namespace goat_can_circle_around_tree_l99_99675

/-- 
  Given a goat tied with a rope of length 4.7 meters (L) near an old tree with a cylindrical trunk of radius 0.5 meters (R), 
  with the shortest distance from the stake to the surface of the tree being 1 meter (d), 
  prove that the minimal required rope length to encircle the tree and return to the stake is less than 
  or equal to the given rope length of 4.7 meters (L).
-/ 
theorem goat_can_circle_around_tree (L R d : ℝ) (hR : R = 0.5) (hd : d = 1) (hL : L = 4.7) : 
  ∃ L_min, L_min ≤ L := 
by
  -- Detailed proof steps omitted.
  sorry

end goat_can_circle_around_tree_l99_99675


namespace system1_solution_system2_solution_l99_99305

theorem system1_solution (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := sorry

theorem system2_solution (x y : ℝ) (h1 : 3 * x - 5 * y = 9) (h2 : 2 * x + 3 * y = -6) : 
  x = -3 / 19 ∧ y = -36 / 19 := sorry

end system1_solution_system2_solution_l99_99305


namespace min_lamps_l99_99844

theorem min_lamps (n p : ℕ) (h1: p > 0) (h_total_profit : 3 * (3 * p / 4 / n) + (n - 3) * (p / n + 10) - p = 100) : n = 13 :=
by
  sorry

end min_lamps_l99_99844


namespace determine_a_and_b_l99_99539

variable (a b : ℕ)
theorem determine_a_and_b 
  (h1: 0 ≤ a ∧ a ≤ 9) 
  (h2: 0 ≤ b ∧ b ≤ 9)
  (h3: (a + b + 45) % 9 = 0)
  (h4: (b - a) % 11 = 3) : 
  a = 3 ∧ b = 6 :=
sorry

end determine_a_and_b_l99_99539


namespace worth_of_entire_lot_l99_99663

theorem worth_of_entire_lot (half_share : ℝ) (amount_per_tenth : ℝ) (total_amount : ℝ) :
  half_share = 0.5 →
  amount_per_tenth = 460 →
  total_amount = (amount_per_tenth * 10) →
  (total_amount * 2) = 9200 :=
by
  intros h1 h2 h3
  sorry

end worth_of_entire_lot_l99_99663


namespace visitors_on_that_day_l99_99813

theorem visitors_on_that_day 
  (prev_visitors : ℕ) 
  (additional_visitors : ℕ) 
  (h1 : prev_visitors = 100)
  (h2 : additional_visitors = 566)
  : prev_visitors + additional_visitors = 666 := by
  sorry

end visitors_on_that_day_l99_99813


namespace elastic_collision_inelastic_collision_l99_99668

-- Given conditions for Case A and Case B
variables (L V : ℝ) (m : ℝ) -- L is length of the rods, V is the speed, m is mass of each sphere

-- Prove Case A: The dumbbells separate maintaining their initial velocities
theorem elastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly elastic collision, the dumbbells separate maintaining their initial velocities
  true := sorry

-- Prove Case B: The dumbbells start rotating around the collision point with angular velocity V / (2 * L)
theorem inelastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly inelastic collision, the dumbbells start rotating around the collision point with angular velocity V / (2 * L)
  true := sorry

end elastic_collision_inelastic_collision_l99_99668


namespace exists_m_inequality_l99_99635

theorem exists_m_inequality (a b : ℝ) (h : a > b) : ∃ m : ℝ, m < 0 ∧ a * m < b * m :=
by
  sorry

end exists_m_inequality_l99_99635


namespace graph_passes_through_0_1_l99_99295

theorem graph_passes_through_0_1 (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^x) } :=
sorry

end graph_passes_through_0_1_l99_99295


namespace ellipse_minimum_distance_point_l99_99731

theorem ellipse_minimum_distance_point :
  ∃ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1) ∧ (∀ p, x - 2 * y - 12 = 0 → dist (x, y) p ≥ dist (2, -3) p) :=
sorry

end ellipse_minimum_distance_point_l99_99731


namespace intersection_line_through_circles_l99_99112

def circle1_equation (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2_equation (x y : ℝ) : Prop := x^2 + y^2 + 2 * x + 2 * y - 14 = 0

theorem intersection_line_through_circles : 
  (∀ x y : ℝ, circle1_equation x y → circle2_equation x y → x + y - 2 = 0) :=
by
  intros x y h1 h2
  sorry

end intersection_line_through_circles_l99_99112


namespace decimal_111_to_base_5_l99_99944

def decimal_to_base_5 (n : ℕ) : ℕ :=
  let rec loop (n : ℕ) (acc : ℕ) (place : ℕ) :=
    if n = 0 then acc
    else 
      let rem := n % 5
      let q := n / 5
      loop q (acc + rem * place) (place * 10)
  loop n 0 1

theorem decimal_111_to_base_5 : decimal_to_base_5 111 = 421 :=
  sorry

end decimal_111_to_base_5_l99_99944


namespace forecast_interpretation_l99_99404

-- Define the conditions
def condition (precipitation_probability : ℕ) : Prop :=
  precipitation_probability = 78

-- Define the interpretation question as a proof
theorem forecast_interpretation (precipitation_probability: ℕ) (cond : condition precipitation_probability) :
  precipitation_probability = 78 :=
by
  sorry

end forecast_interpretation_l99_99404


namespace more_white_animals_than_cats_l99_99369

theorem more_white_animals_than_cats (C W WC : ℕ) 
  (h1 : WC = C / 3) 
  (h2 : WC = W / 6) : W = 2 * C :=
by {
  sorry
}

end more_white_animals_than_cats_l99_99369


namespace part_a_part_b_l99_99592

-- Definitions of the basic tiles, colorings, and the proposition

inductive Color
| black : Color
| white : Color

structure Tile :=
(c00 c01 c10 c11 : Color)

-- Ali's forbidden tiles (6 types for part (a))
def forbiddenTiles_6 : List Tile := 
[ Tile.mk Color.black Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white,
  Tile.mk Color.black Color.white Color.white Color.black,
  Tile.mk Color.black Color.white Color.black Color.black,
  Tile.mk Color.black Color.black Color.black Color.black,
  Tile.mk Color.white Color.white Color.white Color.white
]

-- Ali's forbidden tiles (7 types for part (b))
def forbiddenTiles_7 : List Tile := 
[ Tile.mk Color.black Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white,
  Tile.mk Color.black Color.white Color.white Color.black,
  Tile.mk Color.black Color.white Color.black Color.black,
  Tile.mk Color.black Color.black Color.black Color.black,
  Tile.mk Color.white Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white
]

-- Propositions to be proved

-- Part (a): Mohammad can color the infinite table with no forbidden tiles present
theorem part_a :
  ∃f : ℕ × ℕ → Color, ∀ t ∈ forbiddenTiles_6, ∃ x y : ℕ, ¬(f (x, y) = t.c00 ∧ f (x, y+1) = t.c01 ∧ 
  f (x+1, y) = t.c10 ∧ f (x+1, y+1) = t.c11) := 
sorry

-- Part (b): Ali can present 7 forbidden tiles such that Mohammad cannot achieve his goal
theorem part_b :
  ∀ f : ℕ × ℕ → Color, ∃ t ∈ forbiddenTiles_7, ∃ x y : ℕ, (f (x, y) = t.c00 ∧ f (x, y+1) = t.c01 ∧ 
  f (x+1, y) = t.c10 ∧ f (x+1, y+1) = t.c11) := 
sorry

end part_a_part_b_l99_99592


namespace number_of_schools_l99_99097

-- Define the conditions as parameters and assumptions
structure CityContest (n : ℕ) :=
  (students_per_school : ℕ := 4)
  (total_students : ℕ := students_per_school * n)
  (andrea_percentile : ℕ := 75)
  (andrea_highest_team : Prop)
  (beth_rank : ℕ := 20)
  (carla_rank : ℕ := 47)
  (david_rank : ℕ := 78)
  (andrea_position : ℕ)
  (h3 : andrea_position = (3 * total_students + 1) / 4)
  (h4 : 3 * n > 78)

-- Define the main theorem statement
theorem number_of_schools (n : ℕ) (contest : CityContest n) (h5 : contest.andrea_highest_team) : n = 20 :=
  by {
    -- You would insert the detailed proof of the theorem based on the conditions here.
    sorry
  }

end number_of_schools_l99_99097


namespace equal_sum_sequence_a18_l99_99132

theorem equal_sum_sequence_a18
    (a : ℕ → ℕ)
    (h1 : a 1 = 2)
    (h2 : ∀ n, a n + a (n + 1) = 5) :
    a 18 = 3 :=
sorry

end equal_sum_sequence_a18_l99_99132


namespace value_of_a4_l99_99402

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := n^2 - 3 * n - 4

-- State the main proof problem.
theorem value_of_a4 : a_n 4 = 0 := by
  sorry

end value_of_a4_l99_99402


namespace kyle_gas_and_maintenance_expense_l99_99001

def monthly_income : ℝ := 3200
def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous_expenses : ℝ := 200
def car_payment : ℝ := 350

def total_bills : ℝ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous_expenses

theorem kyle_gas_and_maintenance_expense :
  monthly_income - total_bills - car_payment = 350 :=
by
  sorry

end kyle_gas_and_maintenance_expense_l99_99001


namespace train_passing_pole_l99_99303

variables (v L t_platform D_platform t_pole : ℝ)
variables (H1 : L = 500)
variables (H2 : t_platform = 100)
variables (H3 : D_platform = L + 500)
variables (H4 : t_platform = D_platform / v)

theorem train_passing_pole :
  t_pole = L / v := 
sorry

end train_passing_pole_l99_99303


namespace quadratic_inequality_solution_l99_99024

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + a > 0 ↔ x ≠ -1/a) → a = 1 :=
by
  sorry

end quadratic_inequality_solution_l99_99024


namespace intersection_of_M_and_N_l99_99152

def M := {x : ℝ | 3 * x - x^2 > 0}
def N := {x : ℝ | x^2 - 4 * x + 3 > 0}
def I := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = I :=
by
  sorry

end intersection_of_M_and_N_l99_99152


namespace average_tree_height_l99_99633

theorem average_tree_height : 
  ∀ (T₁ T₂ T₃ T₄ T₅ T₆ : ℕ),
  T₂ = 27 ->
  ((T₁ = 3 * T₂) ∨ (T₁ = T₂ / 3)) ->
  ((T₃ = 3 * T₂) ∨ (T₃ = T₂ / 3)) ->
  ((T₄ = 3 * T₃) ∨ (T₄ = T₃ / 3)) ->
  ((T₅ = 3 * T₄) ∨ (T₅ = T₄ / 3)) ->
  ((T₆ = 3 * T₅) ∨ (T₆ = T₅ / 3)) ->
  (T₁ + T₂ + T₃ + T₄ + T₅ + T₆) / 6 = 22 := 
by 
  intros T₁ T₂ T₃ T₄ T₅ T₆ hT2 hT1 hT3 hT4 hT5 hT6
  sorry

end average_tree_height_l99_99633


namespace smallest_k_for_no_real_roots_l99_99557

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem smallest_k_for_no_real_roots :
  ∃ (k : ℤ), (∀ (x : ℝ), (x * x + 6 * x + 2 * k : ℝ) ≠ 0 ∧ k ≥ 5) :=
by
  sorry

end smallest_k_for_no_real_roots_l99_99557


namespace average_sleep_is_8_l99_99709

-- Define the hours of sleep for each day
def mondaySleep : ℕ := 8
def tuesdaySleep : ℕ := 7
def wednesdaySleep : ℕ := 8
def thursdaySleep : ℕ := 10
def fridaySleep : ℕ := 7

-- Calculate the total hours of sleep over the week
def totalSleep : ℕ := mondaySleep + tuesdaySleep + wednesdaySleep + thursdaySleep + fridaySleep
-- Define the total number of days
def totalDays : ℕ := 5

-- Calculate the average sleep per night
def averageSleepPerNight : ℕ := totalSleep / totalDays

-- Prove the statement
theorem average_sleep_is_8 : averageSleepPerNight = 8 := 
by
  -- All conditions are automatically taken into account as definitions
  -- Add a placeholder to skip the actual proof
  sorry

end average_sleep_is_8_l99_99709


namespace ratio_roots_l99_99430

theorem ratio_roots (p q r s : ℤ)
    (h1 : p ≠ 0)
    (h_roots : ∀ x : ℤ, (x = -1 ∨ x = 3 ∨ x = 4) → (p*x^3 + q*x^2 + r*x + s = 0)) : 
    (r : ℚ) / s = -5 / 12 :=
by sorry

end ratio_roots_l99_99430


namespace money_distribution_l99_99776

theorem money_distribution (a b c : ℝ) (h1 : 4 * (a - b - c) = 16)
                           (h2 : 6 * b - 2 * a - 2 * c = 16)
                           (h3 : 7 * c - a - b = 16) :
  a = 29 := 
by 
  sorry

end money_distribution_l99_99776


namespace feeding_sequences_count_l99_99498

def num_feeding_sequences (num_pairs : ℕ) : ℕ :=
  num_pairs * num_pairs.pred * num_pairs.pred * num_pairs.pred.pred *
  num_pairs.pred.pred * num_pairs.pred.pred.pred * num_pairs.pred.pred.pred *
  1 * 1

theorem feeding_sequences_count (num_pairs : ℕ) (h : num_pairs = 5) :
  num_feeding_sequences num_pairs = 5760 := 
by
  rw [h]
  unfold num_feeding_sequences
  norm_num
  sorry

end feeding_sequences_count_l99_99498


namespace range_of_m_l99_99541

def sufficient_condition (x m : ℝ) : Prop :=
  m - 1 < x ∧ x < m + 1

def inequality (x : ℝ) : Prop :=
  x^2 - 2 * x - 3 > 0

theorem range_of_m (m : ℝ) :
  (∀ x, sufficient_condition x m → inequality x) ↔ (m ≤ -2 ∨ m ≥ 4) :=
by 
  sorry

end range_of_m_l99_99541


namespace number_put_in_machine_l99_99939

theorem number_put_in_machine (x : ℕ) (y : ℕ) (h1 : y = x + 15 - 6) (h2 : y = 77) : x = 68 :=
by
  sorry

end number_put_in_machine_l99_99939


namespace choir_grouping_l99_99062

theorem choir_grouping (sopranos altos tenors basses : ℕ)
  (h_sopranos : sopranos = 10)
  (h_altos : altos = 15)
  (h_tenors : tenors = 12)
  (h_basses : basses = 18)
  (ratio : ℕ) :
  ratio = 1 →
  ∃ G : ℕ, G ≤ 10 ∧ G ≤ 15 ∧ G ≤ 12 ∧ 2 * G ≤ 18 ∧ G = 9 :=
by sorry

end choir_grouping_l99_99062


namespace lime_bottom_means_magenta_top_l99_99228

-- Define the colors as an enumeration for clarity
inductive Color
| Purple : Color
| Cyan : Color
| Magenta : Color
| Lime : Color
| Silver : Color
| Black : Color

open Color

-- Define the function representing the question
def opposite_top_face_given_bottom (bottom : Color) : Color :=
  match bottom with
  | Lime => Magenta
  | _ => Lime  -- For simplicity, we're only handling the Lime case as specified

-- State the theorem
theorem lime_bottom_means_magenta_top : 
  opposite_top_face_given_bottom Lime = Magenta :=
by
  -- This theorem states exactly what we need: if Lime is the bottom face, then Magenta is the top face.
  sorry

end lime_bottom_means_magenta_top_l99_99228


namespace domain_of_sqrt_function_l99_99463

theorem domain_of_sqrt_function : {x : ℝ | 0 ≤ x ∧ x ≤ 1} = {x : ℝ | 1 - x ≥ 0 ∧ x - Real.sqrt (1 - x) ≥ 0} :=
by
  sorry

end domain_of_sqrt_function_l99_99463


namespace karthik_weight_average_l99_99900

noncomputable def average_probable_weight_of_karthik (weight : ℝ) : Prop :=
  (55 < weight ∧ weight < 62) ∧
  (50 < weight ∧ weight < 60) ∧
  (weight ≤ 58) →
  weight = 56.5

theorem karthik_weight_average :
  ∀ weight : ℝ, average_probable_weight_of_karthik weight :=
by
  sorry

end karthik_weight_average_l99_99900


namespace unique_m_for_prime_condition_l99_99302

theorem unique_m_for_prime_condition :
  ∃ (m : ℕ), m > 0 ∧ (∀ (p : ℕ), Prime p → (∀ (n : ℕ), ¬ p ∣ (n^m - m))) ↔ m = 1 :=
sorry

end unique_m_for_prime_condition_l99_99302


namespace rectangle_width_l99_99514

theorem rectangle_width (width : ℝ) : 
  ∃ w, w = 14 ∧
  (∀ length : ℝ, length = 10 →
  (2 * (length + width) = 3 * 16)) → 
  width = w :=
by
  sorry

end rectangle_width_l99_99514


namespace car_count_l99_99445

theorem car_count (x y : ℕ) (h1 : x + y = 36) (h2 : 6 * x + 4 * y = 176) :
  x = 16 ∧ y = 20 :=
by
  sorry

end car_count_l99_99445


namespace ellipse_condition_necessary_not_sufficient_l99_99298

theorem ellipse_condition_necessary_not_sufficient {a b : ℝ} (h : a * b > 0):
  (∀ x y : ℝ, a * x^2 + b * y^2 = 1 → a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0) ∧ 
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) → a * b > 0) :=
sorry

end ellipse_condition_necessary_not_sufficient_l99_99298


namespace Bridget_skittles_after_giving_l99_99237

-- Given conditions
def Bridget_initial_skittles : ℕ := 4
def Henry_skittles : ℕ := 4
def Henry_gives_all_to_Bridget : Prop := True

-- Prove that Bridget will have 8 Skittles in total after Henry gives all of his Skittles to her.
theorem Bridget_skittles_after_giving (h : Henry_gives_all_to_Bridget) :
  Bridget_initial_skittles + Henry_skittles = 8 :=
by
  sorry

end Bridget_skittles_after_giving_l99_99237


namespace simplify_expr1_simplify_expr2_l99_99707

theorem simplify_expr1 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
by
  sorry

end simplify_expr1_simplify_expr2_l99_99707


namespace arrangement_count_l99_99552

def arrangements_with_conditions 
  (boys girls : Nat) 
  (cannot_be_next_to_each_other : Bool) : Nat :=
if cannot_be_next_to_each_other then
  sorry -- The proof will go here
else
  sorry

theorem arrangement_count :
  arrangements_with_conditions 3 2 true = 72 :=
sorry

end arrangement_count_l99_99552


namespace polly_to_sandy_ratio_l99_99637

variable {W P S : ℝ}
variable (h1 : S = (5/2) * W) (h2 : P = 2 * W)

theorem polly_to_sandy_ratio : P = (4/5) * S := by
  sorry

end polly_to_sandy_ratio_l99_99637


namespace total_work_completed_in_18_days_l99_99268

theorem total_work_completed_in_18_days :
  let amit_work_rate := 1/10
  let ananthu_work_rate := 1/20
  let amit_days := 2
  let amit_work_done := amit_days * amit_work_rate
  let remaining_work := 1 - amit_work_done
  let ananthu_days := remaining_work / ananthu_work_rate
  amit_days + ananthu_days = 18 := 
by
  sorry

end total_work_completed_in_18_days_l99_99268


namespace bacteria_growth_rate_l99_99715

theorem bacteria_growth_rate (P : ℝ) (r : ℝ) : 
  (P * r ^ 25 = 2 * (P * r ^ 24) ) → r = 2 :=
by sorry

end bacteria_growth_rate_l99_99715


namespace mutually_exclusive_necessary_not_sufficient_complementary_l99_99919

variables {Ω : Type} {A1 A2 : Set Ω}

/-- Definition of mutually exclusive events -/
def mutually_exclusive (A1 A2 : Set Ω) : Prop :=
  A1 ∩ A2 = ∅

/-- Definition of complementary events -/
def complementary (A1 A2 : Set Ω) : Prop :=
  A1 ∪ A2 = Set.univ ∧ mutually_exclusive A1 A2

/-- The proposition that mutually exclusive events are necessary but not sufficient for being complementary -/
theorem mutually_exclusive_necessary_not_sufficient_complementary :
  (mutually_exclusive A1 A2 → complementary A1 A2) = false 
  ∧ (complementary A1 A2 → mutually_exclusive A1 A2) = true :=
sorry

end mutually_exclusive_necessary_not_sufficient_complementary_l99_99919


namespace cone_and_sphere_volume_l99_99625

theorem cone_and_sphere_volume (π : ℝ) (r h : ℝ) (V_cylinder : ℝ) (V_cone V_sphere V_total : ℝ) 
  (h_cylinder : V_cylinder = 54 * π) 
  (h_radius : h = 3 * r)
  (h_cone : V_cone = (1 / 3) * π * r^2 * h) 
  (h_sphere : V_sphere = (4 / 3) * π * r^3) :
  V_total = 42 * π := 
by
  sorry

end cone_and_sphere_volume_l99_99625


namespace algebraic_expression_correct_l99_99895

theorem algebraic_expression_correct (x y : ℝ) :
  (x - y)^2 - (x^2 - y^2) = (x - y)^2 - (x^2 - y^2) :=
by
  sorry

end algebraic_expression_correct_l99_99895


namespace correct_statement_l99_99128

-- Definition of quadrants
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def is_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_third_quadrant (θ : ℝ) : Prop := -180 < θ ∧ θ < -90
def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement of the problem
theorem correct_statement : is_obtuse_angle θ → is_second_quadrant θ :=
by sorry

end correct_statement_l99_99128


namespace range_of_a_l99_99537

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ a < -4 ∨ a > 4 :=
by
  sorry

end range_of_a_l99_99537


namespace ones_digit_of_3_pow_52_l99_99425

theorem ones_digit_of_3_pow_52 : (3 ^ 52 % 10) = 1 := 
by sorry

end ones_digit_of_3_pow_52_l99_99425


namespace car_selection_proportion_l99_99619

def production_volume_emgrand : ℕ := 1600
def production_volume_king_kong : ℕ := 6000
def production_volume_freedom_ship : ℕ := 2000
def total_selected_cars : ℕ := 48

theorem car_selection_proportion :
  (8, 30, 10) = (
    total_selected_cars * production_volume_emgrand /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_king_kong /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_freedom_ship /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship)
  ) :=
by sorry

end car_selection_proportion_l99_99619


namespace average_age_union_l99_99239

theorem average_age_union (students_A students_B students_C : ℕ)
  (sumA sumB sumC : ℕ) (avgA avgB avgC avgAB avgAC avgBC : ℚ)
  (hA : avgA = (sumA : ℚ) / students_A)
  (hB : avgB = (sumB : ℚ) / students_B)
  (hC : avgC = (sumC : ℚ) / students_C)
  (hAB : avgAB = (sumA + sumB) / (students_A + students_B))
  (hAC : avgAC = (sumA + sumC) / (students_A + students_C))
  (hBC : avgBC = (sumB + sumC) / (students_B + students_C))
  (h_avgA: avgA = 34)
  (h_avgB: avgB = 25)
  (h_avgC: avgC = 45)
  (h_avgAB: avgAB = 30)
  (h_avgAC: avgAC = 42)
  (h_avgBC: avgBC = 36) :
  (sumA + sumB + sumC : ℚ) / (students_A + students_B + students_C) = 33 := 
  sorry

end average_age_union_l99_99239


namespace mixture_percentage_l99_99941

variable (P : ℝ)
variable (x_ryegrass_percent : ℝ := 0.40)
variable (y_ryegrass_percent : ℝ := 0.25)
variable (final_mixture_ryegrass_percent : ℝ := 0.32)

theorem mixture_percentage (h : 0.40 * P + 0.25 * (1 - P) = 0.32) : P = 0.07 / 0.15 := by
  sorry

end mixture_percentage_l99_99941


namespace div_expression_l99_99657

variable {α : Type*} [Field α]

theorem div_expression (a b c : α) : 4 * a^2 * b^2 * c / (-2 * a * b^2) = -2 * a * c := by
  sorry

end div_expression_l99_99657


namespace parallelogram_base_length_l99_99091

theorem parallelogram_base_length (area height : ℝ) (h_area : area = 108) (h_height : height = 9) : 
  ∃ base : ℝ, base = area / height ∧ base = 12 := 
  by sorry

end parallelogram_base_length_l99_99091


namespace determine_A_l99_99933

theorem determine_A (A M C : ℕ) (h1 : A < 10) (h2 : M < 10) (h3 : C < 10) 
(h4 : (100 * A + 10 * M + C) * (A + M + C) = 2244) : A = 3 :=
sorry

end determine_A_l99_99933


namespace inequality_am_gm_l99_99923

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (y * z) + y / (z * x) + z / (x * y)) ≥ (1 / x + 1 / y + 1 / z) := 
by
  sorry

end inequality_am_gm_l99_99923


namespace acute_angle_at_9_35_is_77_5_degrees_l99_99365

def degrees_in_acute_angle_formed_by_hands_of_clock_9_35 : ℝ := 77.5

theorem acute_angle_at_9_35_is_77_5_degrees 
  (hour_angle : ℝ := 270 + (35/60 * 30))
  (minute_angle : ℝ := 35/60 * 360) : 
  |hour_angle - minute_angle| < 180 → |hour_angle - minute_angle| = degrees_in_acute_angle_formed_by_hands_of_clock_9_35 := 
by 
  sorry

end acute_angle_at_9_35_is_77_5_degrees_l99_99365


namespace corporate_event_handshakes_l99_99187

def GroupHandshakes (A B C : Nat) (knows_all_A : Nat) (knows_none : Nat) (C_knows_none : Nat) : Nat :=
  -- Handshakes between Group A and Group B
  let handshakes_AB := knows_none * A
  -- Handshakes within Group B
  let handshakes_B := (knows_none * (knows_none - 1)) / 2
  -- Handshakes between Group B and Group C
  let handshakes_BC := B * C_knows_none
  -- Total handshakes
  handshakes_AB + handshakes_B + handshakes_BC

theorem corporate_event_handshakes : GroupHandshakes 15 20 5 5 15 = 430 :=
by
  sorry

end corporate_event_handshakes_l99_99187


namespace hypotenuse_length_l99_99018

theorem hypotenuse_length
  (x : ℝ) 
  (h_leg_relation : 3 * x - 3 > 0) -- to ensure the legs are positive
  (hypotenuse : ℝ)
  (area_eq : 1 / 2 * x * (3 * x - 3) = 84)
  (pythagorean : hypotenuse^2 = x^2 + (3 * x - 3)^2) :
  hypotenuse = Real.sqrt 505 :=
by 
  sorry

end hypotenuse_length_l99_99018


namespace multiple_of_totient_l99_99340

theorem multiple_of_totient (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a : ℕ), ∀ (i : ℕ), 0 ≤ i ∧ i ≤ n → m ∣ Nat.totient (a + i) :=
by
sorry

end multiple_of_totient_l99_99340


namespace ammonium_chloride_reaction_l99_99344

/-- 
  Given the reaction NH4Cl + H2O → NH4OH + HCl, 
  if 1 mole of NH4Cl reacts with 1 mole of H2O to produce 1 mole of NH4OH, 
  then 1 mole of HCl is formed.
-/
theorem ammonium_chloride_reaction :
  (∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl = 1 ∧ H2O = 1 ∧ NH4OH = 1 → HCl = 1) :=
by
  sorry

end ammonium_chloride_reaction_l99_99344


namespace two_digit_product_GCD_l99_99273

-- We define the condition for two-digit integer numbers
def two_digit_num (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Lean statement capturing the conditions
theorem two_digit_product_GCD :
  ∃ (a b : ℕ), two_digit_num a ∧ two_digit_num b ∧ a * b = 1728 ∧ Nat.gcd a b = 12 := 
by {
  sorry -- The proof steps would go here
}

end two_digit_product_GCD_l99_99273


namespace inequality_solution_l99_99474

variable {α : Type*} [LinearOrderedField α]
variable (a b x : α)

theorem inequality_solution (h1 : a < 0) (h2 : b = -a) :
  0 < x ∧ x < 1 ↔ ax^2 + bx > 0 :=
by sorry

end inequality_solution_l99_99474


namespace solve_exponential_diophantine_equation_l99_99730

theorem solve_exponential_diophantine_equation :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 → (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by {
  sorry
}

end solve_exponential_diophantine_equation_l99_99730


namespace solve_for_n_l99_99240

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end solve_for_n_l99_99240


namespace final_coordinates_l99_99456

-- Definitions for the given conditions
def initial_point : ℝ × ℝ := (-2, 6)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

-- The final proof statement
theorem final_coordinates :
  let S_reflected := reflect_x_axis initial_point
  let S_translated := translate_up S_reflected 10
  S_translated = (-2, 4) :=
by
  sorry

end final_coordinates_l99_99456


namespace value_of_y_l99_99891

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 8) (h2 : x = 2) : y = 3 / 2 :=
by
  sorry

end value_of_y_l99_99891


namespace prism_surface_area_is_8pi_l99_99579

noncomputable def prismSphereSurfaceArea : ℝ :=
  let AB := 2
  let AC := 1
  let BAC := Real.pi / 3 -- angle 60 degrees in radians
  let volume := Real.sqrt 3
  let AA1 := 2
  let radius := Real.sqrt 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area

theorem prism_surface_area_is_8pi : prismSphereSurfaceArea = 8 * Real.pi :=
  by
    sorry

end prism_surface_area_is_8pi_l99_99579


namespace tetrahedron_volume_le_one_eight_l99_99964

theorem tetrahedron_volume_le_one_eight {A B C D : Type} 
  (e₁_AB e₂_AC e₃_AD e₄_BC e₅_BD : ℝ) (h₁ : e₁_AB ≤ 1) (h₂ : e₂_AC ≤ 1) (h₃ : e₃_AD ≤ 1)
  (h₄ : e₄_BC ≤ 1) (h₅ : e₅_BD ≤ 1) : 
  ∃ (vol : ℝ), vol ≤ 1 / 8 :=
sorry

end tetrahedron_volume_le_one_eight_l99_99964


namespace coordinates_of_M_l99_99070

theorem coordinates_of_M :
  -- Given the function f(x) = 2x^2 + 1
  let f : Real → Real := λ x => 2 * x^2 + 1
  -- And its derivative
  let f' : Real → Real := λ x => 4 * x
  -- The coordinates of point M where the instantaneous rate of change is -8 are (-2, 9)
  (∃ x0 : Real, f' x0 = -8 ∧ f x0 = y0 ∧ x0 = -2 ∧ y0 = 9) := by
    sorry

end coordinates_of_M_l99_99070


namespace min_total_balls_l99_99452

theorem min_total_balls (R G B : Nat) (hG : G = 12) (hRG : R + G < 24) : 23 ≤ R + G + B :=
by {
  sorry
}

end min_total_balls_l99_99452


namespace geraldine_more_than_jazmin_l99_99816

def geraldine_dolls : ℝ := 2186.0
def jazmin_dolls : ℝ := 1209.0
def difference_dolls : ℝ := 977.0

theorem geraldine_more_than_jazmin : geraldine_dolls - jazmin_dolls = difference_dolls :=
by sorry

end geraldine_more_than_jazmin_l99_99816


namespace sara_initial_black_marbles_l99_99545

-- Define the given conditions
def red_marbles (sara_has : Nat) : Prop := sara_has = 122
def black_marbles_taken_by_fred (fred_took : Nat) : Prop := fred_took = 233
def black_marbles_now (sara_has_now : Nat) : Prop := sara_has_now = 559

-- The proof problem statement
theorem sara_initial_black_marbles
  (sara_has_red : ∀ n : Nat, red_marbles n)
  (fred_took_marbles : ∀ f : Nat, black_marbles_taken_by_fred f)
  (sara_has_now_black : ∀ b : Nat, black_marbles_now b) :
  ∃ b, b = 559 + 233 :=
by
  sorry

end sara_initial_black_marbles_l99_99545


namespace time_per_employee_updating_payroll_records_l99_99788

-- Define the conditions
def minutes_making_coffee : ℕ := 5
def minutes_per_employee_status_update : ℕ := 2
def num_employees : ℕ := 9
def total_morning_routine_minutes : ℕ := 50

-- Define the proof statement encapsulating the problem
theorem time_per_employee_updating_payroll_records :
  (total_morning_routine_minutes - (minutes_making_coffee + minutes_per_employee_status_update * num_employees)) / num_employees = 3 := by
  sorry

end time_per_employee_updating_payroll_records_l99_99788


namespace HCF_a_b_LCM_a_b_l99_99748

-- Given the HCF condition
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Given numbers
def a : ℕ := 210
def b : ℕ := 286

-- Given HCF condition
theorem HCF_a_b : HCF a b = 26 := by
  sorry

-- LCM definition based on the product and HCF
def LCM (a b : ℕ) : ℕ := (a * b) / HCF a b

-- Theorem to prove
theorem LCM_a_b : LCM a b = 2310 := by
  sorry

end HCF_a_b_LCM_a_b_l99_99748


namespace postman_pete_mileage_l99_99930

theorem postman_pete_mileage :
  let initial_steps := 30000
  let resets := 72
  let final_steps := 45000
  let steps_per_mile := 1500
  let steps_per_full_cycle := 99999 + 1
  let total_steps := initial_steps + resets * steps_per_full_cycle + final_steps
  total_steps / steps_per_mile = 4850 := 
by 
  sorry

end postman_pete_mileage_l99_99930


namespace operation_8_to_cube_root_16_l99_99004

theorem operation_8_to_cube_root_16 : ∃ (x : ℕ), x = 8 ∧ (x * x = (Nat.sqrt 16)^3) :=
by
  sorry

end operation_8_to_cube_root_16_l99_99004


namespace polygon_sides_count_l99_99331

theorem polygon_sides_count :
    ∀ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 3 ∧ n2 = 4 ∧ n3 = 5 ∧ n4 = 6 ∧ n5 = 7 ∧ n6 = 8 →
    (n1 - 2) + (n2 - 2) + (n3 - 2) + (n4 - 2) + (n5 - 2) + (n6 - 1) + 3 = 24 :=
by
  intros n1 n2 n3 n4 n5 n6 h
  sorry

end polygon_sides_count_l99_99331


namespace greatest_three_digit_multiple_of_17_l99_99935

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l99_99935


namespace invalid_prob_distribution_D_l99_99586

noncomputable def sum_of_probs_A : ℚ :=
  0 + 1/2 + 0 + 0 + 1/2

noncomputable def sum_of_probs_B : ℚ :=
  0.1 + 0.2 + 0.3 + 0.4

noncomputable def sum_of_probs_C (p : ℚ) (hp : 0 ≤ p ∧ p ≤ 1) : ℚ :=
  p + (1 - p)

noncomputable def sum_of_probs_D : ℚ :=
  (1/1*2) + (1/2*3) + (1/3*4) + (1/4*5) + (1/5*6) + (1/6*7) + (1/7*8)

theorem invalid_prob_distribution_D :
  sum_of_probs_D ≠ 1 := sorry

end invalid_prob_distribution_D_l99_99586


namespace new_average_is_21_l99_99005

def initial_number_of_students : ℕ := 30
def late_students : ℕ := 4
def initial_jumping_students : ℕ := initial_number_of_students - late_students
def initial_average_score : ℕ := 20
def late_student_scores : List ℕ := [26, 27, 28, 29]
def total_jumps_initial_students : ℕ := initial_jumping_students * initial_average_score
def total_jumps_late_students : ℕ := late_student_scores.sum
def total_jumps_all_students : ℕ := total_jumps_initial_students + total_jumps_late_students
def new_average_score : ℕ := total_jumps_all_students / initial_number_of_students

theorem new_average_is_21 :
  new_average_score = 21 :=
sorry

end new_average_is_21_l99_99005


namespace first_mission_days_l99_99872

-- Definitions
variable (x : ℝ) (extended_first_mission : ℝ) (second_mission : ℝ) (total_mission_time : ℝ)

axiom h1 : extended_first_mission = 1.60 * x
axiom h2 : second_mission = 3
axiom h3 : total_mission_time = 11
axiom h4 : extended_first_mission + second_mission = total_mission_time

-- Theorem to prove
theorem first_mission_days : x = 5 :=
by
  sorry

end first_mission_days_l99_99872


namespace train_length_l99_99479

theorem train_length (L S : ℝ) 
  (h1 : L = S * 15) 
  (h2 : L + 100 = S * 25) : 
  L = 150 :=
by
  sorry

end train_length_l99_99479


namespace greatest_k_dividing_n_l99_99372

theorem greatest_k_dividing_n (n : ℕ) 
  (h1 : Nat.totient n = 72) 
  (h2 : Nat.totient (3 * n) = 96) : ∃ k : ℕ, 3^k ∣ n ∧ ∀ j : ℕ, 3^j ∣ n → j ≤ 2 := 
by {
  sorry
}

end greatest_k_dividing_n_l99_99372


namespace pentagon_area_l99_99903

noncomputable def square_area (side_length : ℤ) : ℤ :=
  side_length * side_length

theorem pentagon_area (CF : ℤ) (a b : ℤ) (CE : ℤ) (ED : ℤ) (EF : ℤ) :
  (CF = 5) →
  (a = CE + ED) →
  (b = EF) →
  (CE < ED) →
  CF * CF = CE * CE + EF * EF →
  square_area a + square_area b - (CE * EF / 2) = 71 :=
by
  intros hCF ha hb hCE_lt_ED hPythagorean
  sorry

end pentagon_area_l99_99903


namespace negate_universal_proposition_l99_99293

open Classical

def P (x : ℝ) : Prop := x^3 - 3*x > 0

theorem negate_universal_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

end negate_universal_proposition_l99_99293


namespace problem_1_problem_2_l99_99650

theorem problem_1 (x y : ℝ) (h1 : x - y = 3) (h2 : 3*x - 8*y = 14) : x = 2 ∧ y = -1 :=
sorry

theorem problem_2 (x y : ℝ) (h1 : 3*x + 4*y = 16) (h2 : 5*x - 6*y = 33) : x = 6 ∧ y = -1/2 :=
sorry

end problem_1_problem_2_l99_99650


namespace average_of_remaining_two_numbers_l99_99290

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 8) 
  (h2 : (a + b + c + d) / 4 = 5) : 
  (e + f) / 2 = 14 := 
by  
  sorry

end average_of_remaining_two_numbers_l99_99290


namespace find_unknown_number_l99_99379

theorem find_unknown_number (x : ℝ) (h : (2 / 3) * x + 6 = 10) : x = 6 :=
  sorry

end find_unknown_number_l99_99379


namespace total_cost_of_books_and_pencils_l99_99987

variable (a b : ℕ)

theorem total_cost_of_books_and_pencils (a b : ℕ) : 5 * a + 2 * b = 5 * a + 2 * b := by
  sorry

end total_cost_of_books_and_pencils_l99_99987


namespace seventh_observation_l99_99135

theorem seventh_observation (avg6 : ℕ) (new_avg7 : ℕ) (old_avg : ℕ) (new_avg_diff : ℕ) (n : ℕ) (m : ℕ) (h1 : avg6 = 12) (h2 : new_avg_diff = 1) (h3 : n = 6) (h4 : m = 7) :
  ((n * old_avg = avg6 * old_avg) ∧ (m * new_avg7 = avg6 * old_avg + m - n)) →
  m * new_avg7 = 77 →
  avg6 * old_avg = 72 →
  77 - 72 = 5 :=
by
  sorry

end seventh_observation_l99_99135


namespace hyperbola_asymptotes_l99_99742

theorem hyperbola_asymptotes:
  ∀ (x y : ℝ),
  ( ∀ y, y = (1 + (4 / 5) * x) ∨ y = (1 - (4 / 5) * x) ) →
  (y-1)^2 / 16 - x^2 / 25 = 1 →
  (∃ m b: ℝ, m > 0 ∧ m = 4/5 ∧ b = 1) := by
  sorry

end hyperbola_asymptotes_l99_99742


namespace graph_empty_l99_99258

theorem graph_empty (x y : ℝ) : 
  x^2 + 3 * y^2 - 4 * x - 6 * y + 9 ≠ 0 :=
by
  -- Proof omitted
  sorry

end graph_empty_l99_99258


namespace xyz_stock_final_price_l99_99606

theorem xyz_stock_final_price :
  let s0 := 120
  let s1 := s0 + s0 * 1.5
  let s2 := s1 - s1 * 0.3
  let s3 := s2 + s2 * 0.2
  s3 = 252 := by
  sorry

end xyz_stock_final_price_l99_99606


namespace f_f_2_l99_99316

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 2 then 2 * Real.exp (x - 1) else Real.log (2^x - 1) / Real.log 3

theorem f_f_2 : f (f 2) = 2 :=
by
  sorry

end f_f_2_l99_99316


namespace susan_vacation_pay_missed_l99_99224

noncomputable def susan_weekly_pay (hours_worked : ℕ) : ℕ :=
  let regular_hours := min 40 hours_worked
  let overtime_hours := max (hours_worked - 40) 0
  15 * regular_hours + 20 * overtime_hours

noncomputable def susan_sunday_pay (num_sundays : ℕ) (hours_per_sunday : ℕ) : ℕ :=
  25 * num_sundays * hours_per_sunday

noncomputable def pay_without_sundays : ℕ :=
  susan_weekly_pay 48
    
noncomputable def total_three_week_pay : ℕ :=
  let weeks_normal_pay := 3 * pay_without_sundays
  let sunday_hours_1 := 1 * 8
  let sunday_hours_2 := 2 * 8
  let sunday_hours_3 := 0 * 8
  let sundays_total_pay := susan_sunday_pay 1 8 + susan_sunday_pay 2 8 + susan_sunday_pay 0 8
  weeks_normal_pay + sundays_total_pay
  
noncomputable def paid_vacation_pay : ℕ :=
  let paid_days := 6
  let paid_weeks_pay := susan_weekly_pay 40 + susan_weekly_pay (paid_days % 5 * 8)
  paid_weeks_pay

theorem susan_vacation_pay_missed :
  let missed_pay := total_three_week_pay - paid_vacation_pay
  missed_pay = 2160 := sorry

end susan_vacation_pay_missed_l99_99224


namespace real_solutions_l99_99533

theorem real_solutions (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 5) :
  ( (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1) ) / 
  ( (x - 2) * (x - 4) * (x - 5) * (x - 2) ) = 1 
  ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by sorry

end real_solutions_l99_99533


namespace twentieth_number_l99_99253

-- Defining the conditions and goal
theorem twentieth_number :
  ∃ x : ℕ, x % 8 = 5 ∧ x % 3 = 2 ∧ (∃ n : ℕ, x = 5 + 24 * n) ∧ x = 461 := 
sorry

end twentieth_number_l99_99253


namespace instrument_failure_probability_l99_99908

noncomputable def probability_of_instrument_not_working (m : ℕ) (P : ℝ) : ℝ :=
  1 - (1 - P)^m

theorem instrument_failure_probability (m : ℕ) (P : ℝ) :
  0 ≤ P → P ≤ 1 → probability_of_instrument_not_working m P = 1 - (1 - P)^m :=
by
  intros _ _
  sorry

end instrument_failure_probability_l99_99908


namespace cos_of_three_pi_div_two_l99_99630

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end cos_of_three_pi_div_two_l99_99630


namespace price_difference_l99_99007

-- Define the prices of commodity X and Y in the year 2001 + n.
def P_X (n : ℕ) (a : ℝ) : ℝ := 4.20 + 0.45 * n + a * n
def P_Y (n : ℕ) (b : ℝ) : ℝ := 6.30 + 0.20 * n + b * n

-- Define the main theorem to prove
theorem price_difference (n : ℕ) (a b : ℝ) :
  P_X n a = P_Y n b + 0.65 ↔ (0.25 + a - b) * n = 2.75 :=
by
  sorry

end price_difference_l99_99007


namespace bug_converges_to_final_position_l99_99696

noncomputable def bug_final_position : ℝ × ℝ := 
  let horizontal_sum := ∑' n, if n % 4 = 0 then (1 / 4) ^ (n / 4) else 0
  let vertical_sum := ∑' n, if n % 4 = 1 then (1 / 4) ^ (n / 4) else 0
  (horizontal_sum, vertical_sum)

theorem bug_converges_to_final_position : bug_final_position = (4 / 5, 2 / 5) := 
  sorry

end bug_converges_to_final_position_l99_99696


namespace total_passengers_landed_l99_99323

theorem total_passengers_landed 
  (passengers_on_time : ℕ) 
  (passengers_late : ℕ) 
  (passengers_connecting : ℕ) 
  (passengers_changed_plans : ℕ)
  (H1 : passengers_on_time = 14507)
  (H2 : passengers_late = 213)
  (H3 : passengers_connecting = 320)
  (H4 : passengers_changed_plans = 95) : 
  passengers_on_time + passengers_late + passengers_connecting = 15040 :=
by 
  sorry

end total_passengers_landed_l99_99323


namespace small_bottles_initial_l99_99106

theorem small_bottles_initial
  (S : ℤ)
  (big_bottles_initial : ℤ := 15000)
  (sold_small_bottles_percentage : ℚ := 0.11)
  (sold_big_bottles_percentage : ℚ := 0.12)
  (remaining_bottles_in_storage : ℤ := 18540)
  (remaining_small_bottles : ℚ := 0.89 * S)
  (remaining_big_bottles : ℚ := 0.88 * big_bottles_initial)
  (h : remaining_small_bottles + remaining_big_bottles = remaining_bottles_in_storage)
  : S = 6000 :=
by
  sorry

end small_bottles_initial_l99_99106


namespace necessary_but_not_sufficient_l99_99890

theorem necessary_but_not_sufficient (x : Real)
  (p : Prop := x < 1) 
  (q : Prop := x^2 + x - 2 < 0) 
  : p -> (q <-> x > -2 ∧ x < 1) ∧ (q -> p) → ¬ (p -> q) ∧ (x > -2 -> p) :=
by
  sorry

end necessary_but_not_sufficient_l99_99890


namespace pet_store_cages_l99_99044

theorem pet_store_cages (init_puppies sold_puppies puppies_per_cage : ℕ)
  (h1 : init_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (init_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l99_99044


namespace find_ratio_l99_99095

theorem find_ratio (x y c d : ℝ) (h₁ : 4 * x - 2 * y = c) (h₂ : 5 * y - 10 * x = d) (h₃ : d ≠ 0) : c / d = 0 :=
sorry

end find_ratio_l99_99095


namespace number_of_divisors_of_square_l99_99531

theorem number_of_divisors_of_square {n : ℕ} (h : ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ n = p * q) : Nat.totient (n^2) = 9 :=
sorry

end number_of_divisors_of_square_l99_99531


namespace bus_problem_l99_99717

theorem bus_problem
  (initial_children : ℕ := 18)
  (final_total_children : ℕ := 25) :
  final_total_children - initial_children = 7 :=
by
  sorry

end bus_problem_l99_99717


namespace ratio_of_volumes_l99_99367

-- Define the edge lengths
def edge_length_cube1 : ℝ := 9
def edge_length_cube2 : ℝ := 24

-- Theorem stating the ratio of the volumes
theorem ratio_of_volumes :
  (edge_length_cube1 / edge_length_cube2) ^ 3 = 27 / 512 :=
by
  sorry

end ratio_of_volumes_l99_99367


namespace find_a1_l99_99388

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (s : ℕ → ℝ) :=
∀ n : ℕ, s n = (n * (a 1 + a n)) / 2

theorem find_a1 
  (a : ℕ → ℝ) (s : ℕ → ℝ)
  (d : ℝ)
  (h_seq : arithmetic_sequence a d)
  (h_sum : sum_first_n_terms a s)
  (h_S10_eq_S11 : s 10 = s 11) : 
  a 1 = 20 := 
sorry

end find_a1_l99_99388


namespace find_counterfeit_coin_l99_99603

def is_counterfeit (coins : Fin 9 → ℝ) (i : Fin 9) : Prop :=
  ∀ j : Fin 9, j ≠ i → coins j = coins 0 ∧ coins i < coins 0

def algorithm_exists (coins : Fin 9 → ℝ) : Prop :=
  ∃ f : (Fin 9 → ℝ) → Fin 9, is_counterfeit coins (f coins)

theorem find_counterfeit_coin (coins : Fin 9 → ℝ) (h : ∃ i : Fin 9, is_counterfeit coins i) : algorithm_exists coins :=
by sorry

end find_counterfeit_coin_l99_99603


namespace symmetric_points_coords_l99_99615

theorem symmetric_points_coords (a b : ℝ) :
    let N := (a, -b)
    let P := (-a, -b)
    let Q := (b, a)
    N = (a, -b) ∧ P = (-a, -b) ∧ Q = (b, a) →
    Q = (b, a) :=
by
  intro h
  sorry

end symmetric_points_coords_l99_99615


namespace find_number_l99_99954

theorem find_number 
  (x : ℝ) 
  (h1 : 3 * (2 * x + 9) = 69) : x = 7 := by
  sorry

end find_number_l99_99954


namespace max_snowmen_l99_99832

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l99_99832


namespace sum_cubes_coeffs_l99_99869

theorem sum_cubes_coeffs :
  ∃ a b c d e : ℤ, 
  (1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
  (a + b + c + d + e = 92) :=
sorry

end sum_cubes_coeffs_l99_99869


namespace percentage_salt_solution_l99_99880

-- Definitions
def P : ℝ := 60
def ounces_added := 40
def initial_solution_ounces := 40
def initial_solution_percentage := 0.20
def final_solution_percentage := 0.40
def final_solution_ounces := 80

-- Lean Statement
theorem percentage_salt_solution (P : ℝ) :
  (8 + 0.01 * P * ounces_added) = 0.40 * final_solution_ounces → P = 60 := 
by
  sorry

end percentage_salt_solution_l99_99880


namespace cos_theta_value_l99_99720

theorem cos_theta_value (θ : ℝ) (h_tan : Real.tan θ = -4/3) (h_range : 0 < θ ∧ θ < π) : Real.cos θ = -3/5 :=
by
  sorry

end cos_theta_value_l99_99720


namespace more_blue_marbles_l99_99725

theorem more_blue_marbles (r_boxes b_boxes marbles_per_box : ℕ) 
    (red_total_eq : r_boxes * marbles_per_box = 70) 
    (blue_total_eq : b_boxes * marbles_per_box = 126) 
    (r_boxes_eq : r_boxes = 5) 
    (b_boxes_eq : b_boxes = 9) 
    (marbles_per_box_eq : marbles_per_box = 14) : 
    126 - 70 = 56 := 
by 
  sorry

end more_blue_marbles_l99_99725


namespace solve_for_x_l99_99581

-- Define the given condition
def condition (x : ℝ) : Prop := (x - 5) ^ 3 = -((1 / 27)⁻¹)

-- State the problem as a Lean theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 2 := by
  sorry

end solve_for_x_l99_99581


namespace cannot_determine_letters_afternoon_l99_99493

theorem cannot_determine_letters_afternoon
  (emails_morning : ℕ) (letters_morning : ℕ)
  (emails_afternoon : ℕ) (letters_afternoon : ℕ)
  (h1 : emails_morning = 10)
  (h2 : letters_morning = 12)
  (h3 : emails_afternoon = 3)
  (h4 : emails_morning = emails_afternoon + 7) :
  ¬∃ (letters_afternoon : ℕ), true := 
sorry

end cannot_determine_letters_afternoon_l99_99493


namespace real_polynomial_has_exactly_one_real_solution_l99_99713

theorem real_polynomial_has_exactly_one_real_solution:
  ∀ a : ℝ, ∃! x : ℝ, x^3 - a * x^2 - 3 * a * x + a^2 - 1 = 0 := 
by
  sorry

end real_polynomial_has_exactly_one_real_solution_l99_99713


namespace cadastral_value_of_land_l99_99263

theorem cadastral_value_of_land (tax_amount : ℝ) (tax_rate : ℝ) (V : ℝ)
    (h1 : tax_amount = 4500)
    (h2 : tax_rate = 0.003) :
    V = 1500000 :=
by
  sorry

end cadastral_value_of_land_l99_99263


namespace x_plus_y_equals_six_l99_99182

theorem x_plus_y_equals_six (x y : ℝ) (h₁ : y - x = 1) (h₂ : y^2 = x^2 + 6) : x + y = 6 :=
by
  sorry

end x_plus_y_equals_six_l99_99182


namespace range_of_t_l99_99254

variable {f : ℝ → ℝ}

theorem range_of_t (h₁ : ∀ x y : ℝ, x < y → f x ≥ f y) (h₂ : ∀ t : ℝ, f (t^2) < f t) : 
  ∀ t : ℝ, f (t^2) < f t ↔ (t < 0 ∨ t > 1) := 
by 
  sorry

end range_of_t_l99_99254


namespace solve_for_diamond_l99_99757

theorem solve_for_diamond (d : ℤ) (h : d * 9 + 5 = d * 10 + 2) : d = 3 :=
by
  sorry

end solve_for_diamond_l99_99757


namespace travel_speed_l99_99833

theorem travel_speed (distance : ℕ) (time : ℕ) (h_distance : distance = 160) (h_time : time = 8) :
  ∃ speed : ℕ, speed = distance / time ∧ speed = 20 :=
by
  sorry

end travel_speed_l99_99833


namespace volume_inside_sphere_outside_cylinder_l99_99466

noncomputable def sphere_radius := 6
noncomputable def cylinder_diameter := 8
noncomputable def sphere_volume := 4/3 * Real.pi * (sphere_radius ^ 3)
noncomputable def cylinder_height := Real.sqrt ((sphere_radius * 2) ^ 2 - (cylinder_diameter) ^ 2)
noncomputable def cylinder_volume := Real.pi * ((cylinder_diameter / 2) ^ 2) * cylinder_height
noncomputable def volume_difference := sphere_volume - cylinder_volume

theorem volume_inside_sphere_outside_cylinder:
  volume_difference = (288 - 64 * Real.sqrt 5) * Real.pi :=
sorry

end volume_inside_sphere_outside_cylinder_l99_99466


namespace negation_of_p_l99_99636

theorem negation_of_p (p : Prop) : (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ (∀ x : ℝ, x > 0 → ¬ ((x + 1) * Real.exp x > 1)) :=
by
  sorry

end negation_of_p_l99_99636


namespace joan_missed_games_l99_99387

-- Define the number of total games and games attended as constants
def total_games : ℕ := 864
def games_attended : ℕ := 395

-- The theorem statement: the number of missed games is equal to 469
theorem joan_missed_games : total_games - games_attended = 469 :=
by
  -- Proof goes here
  sorry

end joan_missed_games_l99_99387


namespace pyramid_base_side_length_l99_99155

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l99_99155


namespace tan_eleven_pi_over_three_l99_99111

theorem tan_eleven_pi_over_three : Real.tan (11 * Real.pi / 3) = -Real.sqrt 3 := 
    sorry

end tan_eleven_pi_over_three_l99_99111


namespace quadratic_trinomial_constant_l99_99125

theorem quadratic_trinomial_constant (m : ℝ) (h : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
sorry

end quadratic_trinomial_constant_l99_99125


namespace percentage_increase_l99_99989

def originalPrice : ℝ := 300
def newPrice : ℝ := 390

theorem percentage_increase :
  ((newPrice - originalPrice) / originalPrice) * 100 = 30 := by
  sorry

end percentage_increase_l99_99989


namespace pyramid_volume_l99_99096

noncomputable def volume_of_pyramid 
  (EFGH_rect : ℝ × ℝ) 
  (EF_len : EFGH_rect.1 = 15 * Real.sqrt 2) 
  (FG_len : EFGH_rect.2 = 14 * Real.sqrt 2)
  (isosceles_pyramid : Prop) : ℝ :=
  sorry

theorem pyramid_volume 
  (EFGH_rect : ℝ × ℝ) 
  (EF_len : EFGH_rect.1 = 15 * Real.sqrt 2) 
  (FG_len : EFGH_rect.2 = 14 * Real.sqrt 2) 
  (isosceles_pyramid : Prop) : 
  volume_of_pyramid EFGH_rect EF_len FG_len isosceles_pyramid = 735 := 
sorry

end pyramid_volume_l99_99096


namespace ratio_singers_joined_second_to_remaining_first_l99_99998

-- Conditions
def total_singers : ℕ := 30
def singers_first_verse : ℕ := total_singers / 2
def remaining_after_first : ℕ := total_singers - singers_first_verse
def singers_joined_third_verse : ℕ := 10
def all_singing : ℕ := total_singers

-- Definition for singers who joined in the second verse
def singers_joined_second_verse : ℕ := all_singing - singers_joined_third_verse - singers_first_verse

-- The target proof
theorem ratio_singers_joined_second_to_remaining_first :
  (singers_joined_second_verse : ℚ) / remaining_after_first = 1 / 3 :=
by
  sorry

end ratio_singers_joined_second_to_remaining_first_l99_99998


namespace find_n_l99_99867

theorem find_n (n : ℕ) (h : (n + 1) * n.factorial = 5040) : n = 6 := 
by sorry

end find_n_l99_99867


namespace part_I_part_II_l99_99659

-- Definition of functions
def f (x a : ℝ) := |3 * x - a|
def g (x : ℝ) := |x + 1|

-- Part (I): Solution set for f(x) < 3 when a = 4
theorem part_I (x : ℝ) : f x 4 < 3 ↔ (1 / 3 < x ∧ x < 7 / 3) :=
by 
  sorry

-- Part (II): Range of a such that f(x) + g(x) > 1 for all x in ℝ
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x > 1) ↔ (a < -6 ∨ a > 0) :=
by 
  sorry

end part_I_part_II_l99_99659


namespace remainder_n_plus_2023_l99_99721

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 5 = 2) : (n + 2023) % 5 = 0 :=
sorry

end remainder_n_plus_2023_l99_99721


namespace mike_total_cost_self_correct_l99_99312

-- Definition of the given conditions
def cost_per_rose_bush : ℕ := 75
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def cost_per_tiger_tooth_aloes : ℕ := 100
def total_tiger_tooth_aloes : ℕ := 2

-- Calculate the total cost for Mike's plants
def total_cost_mike_self: ℕ := 
  (total_rose_bushes - friend_rose_bushes) * cost_per_rose_bush + total_tiger_tooth_aloes * cost_per_tiger_tooth_aloes

-- The main proposition to be proved
theorem mike_total_cost_self_correct : total_cost_mike_self = 500 := by
  sorry

end mike_total_cost_self_correct_l99_99312


namespace distinct_prime_divisors_l99_99809

theorem distinct_prime_divisors (a : ℤ) (n : ℕ) (h₁ : a > 3) (h₂ : Odd a) (h₃ : n > 0) : 
  ∃ (p : Finset ℤ), p.card ≥ n + 1 ∧ ∀ q ∈ p, Prime q ∧ q ∣ (a ^ (2 ^ n) - 1) :=
sorry

end distinct_prime_divisors_l99_99809


namespace sum_of_solutions_l99_99526

-- Define the system of equations as lean functions
def equation1 (x y : ℝ) : Prop := |x - 4| = |y - 10|
def equation2 (x y : ℝ) : Prop := |x - 10| = 3 * |y - 4|

-- Statement of the theorem
theorem sum_of_solutions : 
  ∃ (solutions : List (ℝ × ℝ)), 
    (∀ (sol : ℝ × ℝ), sol ∈ solutions → equation1 sol.1 sol.2 ∧ equation2 sol.1 sol.2) ∧ 
    (List.sum (solutions.map (fun sol => sol.1 + sol.2)) = 24) :=
  sorry

end sum_of_solutions_l99_99526


namespace find_k_l99_99977

theorem find_k (k : ℝ) (h : (-3 : ℝ)^2 + (-3 : ℝ) - k = 0) : k = 6 :=
by
  sorry

end find_k_l99_99977


namespace area_of_triangle_l99_99549

theorem area_of_triangle (a c : ℝ) (A : ℝ) (h_a : a = 2) (h_c : c = 2 * Real.sqrt 3) (h_A : A = Real.pi / 6) :
  ∃ (area : ℝ), area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l99_99549


namespace joanie_loan_difference_l99_99995

theorem joanie_loan_difference:
  let P := 6000
  let r := 0.12
  let t := 4
  let n_quarterly := 4
  let n_annually := 1
  let A_quarterly := P * (1 + r / n_quarterly)^(n_quarterly * t)
  let A_annually := P * (1 + r / n_annually)^t
  A_quarterly - A_annually = 187.12 := sorry

end joanie_loan_difference_l99_99995


namespace power_mod_8_l99_99154

theorem power_mod_8 (n : ℕ) (h : n % 2 = 0) : 3^n % 8 = 1 :=
by sorry

end power_mod_8_l99_99154


namespace fraction_of_total_l99_99286

def total_amount : ℝ := 5000
def r_amount : ℝ := 2000.0000000000002

theorem fraction_of_total
  (h1 : r_amount = 2000.0000000000002)
  (h2 : total_amount = 5000) :
  r_amount / total_amount = 0.40000000000000004 :=
by
  -- The proof is skipped
  sorry

end fraction_of_total_l99_99286


namespace bert_earns_more_l99_99739

def bert_toy_phones : ℕ := 8
def bert_price_per_phone : ℕ := 18
def tory_toy_guns : ℕ := 7
def tory_price_per_gun : ℕ := 20

theorem bert_earns_more : (bert_toy_phones * bert_price_per_phone) - (tory_toy_guns * tory_price_per_gun) = 4 := by
  sorry

end bert_earns_more_l99_99739


namespace divisible_sum_l99_99624

theorem divisible_sum (k : ℕ) (n : ℕ) (h : n = 2^(k-1)) : 
  ∀ (S : Finset ℕ), S.card = 2*n - 1 → ∃ T ⊆ S, T.card = n ∧ T.sum id % n = 0 :=
by
  sorry

end divisible_sum_l99_99624


namespace negate_proposition_l99_99405

variable (x : ℝ)

theorem negate_proposition :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ + 1/4 ≤ 0)) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 :=
by
  sorry

end negate_proposition_l99_99405


namespace order_of_variables_l99_99433

variable (a b c d : ℝ)

theorem order_of_variables (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) : c > a ∧ a > b ∧ b > d :=
by
  sorry

end order_of_variables_l99_99433


namespace find_sports_package_channels_l99_99736

-- Defining the conditions
def initial_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def reduce_package_by : ℕ := 10
def supreme_sports_package : ℕ := 7
def final_channels : ℕ := 147

-- Defining the situation before the final step
def channels_after_reduction := initial_channels - channels_taken_away + channels_replaced - reduce_package_by
def channels_after_supreme := channels_after_reduction + supreme_sports_package

-- Prove the original sports package added 8 channels
theorem find_sports_package_channels : ∀ sports_package_added : ℕ,
  sports_package_added + channels_after_supreme = final_channels → sports_package_added = 8 :=
by
  intro sports_package_added
  intro h
  sorry

end find_sports_package_channels_l99_99736


namespace cost_effective_combination_l99_99693

/--
Jackson wants to impress his girlfriend by filling her hot tub with champagne.
The hot tub holds 400 liters of liquid. He has three types of champagne bottles:
1. Small bottle: Holds 0.75 liters with a price of $70 per bottle.
2. Medium bottle: Holds 1.5 liters with a price of $120 per bottle.
3. Large bottle: Holds 3 liters with a price of $220 per bottle.

If he purchases more than 50 bottles of any type, he will get a 10% discount on 
that type. If he purchases over 100 bottles of any type, he will get 20% off 
on that type of bottles. 

Prove that the most cost-effective combination of bottles for 
Jackson to purchase is 134 large bottles for a total cost of $23,584 after the discount.
-/
theorem cost_effective_combination :
  let volume := 400
  let small_bottle_volume := 0.75
  let small_bottle_cost := 70
  let medium_bottle_volume := 1.5
  let medium_bottle_cost := 120
  let large_bottle_volume := 3
  let large_bottle_cost := 220
  let discount_50 := 0.10
  let discount_100 := 0.20
  let cost_134_large_bottles := (134 * large_bottle_cost) * (1 - discount_100)
  cost_134_large_bottles = 23584 :=
sorry

end cost_effective_combination_l99_99693


namespace transformed_ellipse_equation_l99_99427

namespace EllipseTransformation

open Real

def original_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 = 1

def transformation (x' y' x y : ℝ) : Prop :=
  x' = 1 / 2 * x ∧ y' = 2 * y

theorem transformed_ellipse_equation (x y x' y' : ℝ) 
  (h : original_ellipse x y) (tr : transformation x' y' x y) :
  2 * x'^2 / 3 + y'^2 / 4 = 1 :=
by 
  sorry

end EllipseTransformation

end transformed_ellipse_equation_l99_99427


namespace customers_sampling_candy_l99_99180

theorem customers_sampling_candy (total_customers caught fined not_caught : ℝ) 
    (h1 : total_customers = 100) 
    (h2 : caught = 0.22 * total_customers) 
    (h3 : not_caught / (caught / 0.9) = 0.1) :
    (not_caught + caught) / total_customers = 0.2444 := 
by sorry

end customers_sampling_candy_l99_99180


namespace probability_is_1_over_90_l99_99067

/-- Probability Calculation -/
noncomputable def probability_of_COLD :=
  (1 / (Nat.choose 5 3)) * (2 / 3) * (1 / (Nat.choose 4 2))

theorem probability_is_1_over_90 :
  probability_of_COLD = (1 / 90) :=
by
  sorry

end probability_is_1_over_90_l99_99067


namespace evaluate_ratio_l99_99865

theorem evaluate_ratio : (2^2003 * 3^2002) / (6^2002) = 2 := 
by {
  sorry
}

end evaluate_ratio_l99_99865


namespace amusing_permutations_formula_l99_99359

-- Definition of amusing permutations and their count
def amusing_permutations_count (n : ℕ) : ℕ :=
  2^(n-1)

-- Theorem statement: The number of amusing permutations of the set {1, 2, ..., n} is 2^(n-1)
theorem amusing_permutations_formula (n : ℕ) : 
  -- The number of amusing permutations should be equal to 2^(n-1)
  amusing_permutations_count n = 2^(n-1) :=
by
  sorry

end amusing_permutations_formula_l99_99359


namespace frustum_small_cone_height_is_correct_l99_99705

noncomputable def frustum_small_cone_height (altitude : ℝ) 
                                             (lower_base_area : ℝ) 
                                             (upper_base_area : ℝ) : ℝ :=
  let r1 := Real.sqrt (lower_base_area / Real.pi)
  let r2 := Real.sqrt (upper_base_area / Real.pi)
  let H := 2 * altitude
  altitude

theorem frustum_small_cone_height_is_correct 
  (altitude : ℝ)
  (lower_base_area : ℝ)
  (upper_base_area : ℝ)
  (h1 : altitude = 16)
  (h2 : lower_base_area = 196 * Real.pi)
  (h3 : upper_base_area = 49 * Real.pi ) : 
  frustum_small_cone_height altitude lower_base_area upper_base_area = 16 := by
  sorry

end frustum_small_cone_height_is_correct_l99_99705


namespace probability_mask_with_ear_loops_l99_99499

-- Definitions from the conditions
def production_ratio_regular : ℝ := 0.8
def production_ratio_surgical : ℝ := 0.2
def proportion_ear_loops_regular : ℝ := 0.1
def proportion_ear_loops_surgical : ℝ := 0.2

-- Theorem statement based on the translated proof problem
theorem probability_mask_with_ear_loops :
  production_ratio_regular * proportion_ear_loops_regular +
  production_ratio_surgical * proportion_ear_loops_surgical = 0.12 :=
by
  -- Proof omitted
  sorry

end probability_mask_with_ear_loops_l99_99499


namespace larger_interior_angle_trapezoid_pavilion_l99_99780

theorem larger_interior_angle_trapezoid_pavilion :
  let n := 12
  let central_angle := 360 / n
  let smaller_angle := 180 - (central_angle / 2)
  let larger_angle := 180 - smaller_angle
  larger_angle = 97.5 :=
by
  sorry

end larger_interior_angle_trapezoid_pavilion_l99_99780


namespace additional_oil_needed_l99_99525

variable (oil_per_cylinder : ℕ) (number_of_cylinders : ℕ) (oil_already_added : ℕ)

theorem additional_oil_needed (h1 : oil_per_cylinder = 8) (h2 : number_of_cylinders = 6) (h3 : oil_already_added = 16) :
  oil_per_cylinder * number_of_cylinders - oil_already_added = 32 :=
by
  -- proof here
  sorry

end additional_oil_needed_l99_99525


namespace expenditure_representation_l99_99672

theorem expenditure_representation (income expenditure : ℤ)
  (h_income : income = 60)
  (h_expenditure : expenditure = 40) :
  -expenditure = -40 :=
by {
  sorry
}

end expenditure_representation_l99_99672


namespace identical_functions_l99_99289

def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := (x^2 - 1)^3^(1/3)

theorem identical_functions : ∀ x : ℝ, f x = g x :=
by
  intro x
  -- Proof to be completed
  sorry

end identical_functions_l99_99289


namespace average_age_of_boys_l99_99245

def boys_age_proportions := (3, 5, 7)
def eldest_boy_age := 21

theorem average_age_of_boys : 
  ∃ (x : ℕ), 7 * x = eldest_boy_age ∧ (3 * x + 5 * x + 7 * x) / 3 = 15 :=
by
  sorry

end average_age_of_boys_l99_99245


namespace merchant_marked_price_l99_99940

-- Given conditions: 30% discount on list price, 10% discount on marked price, 25% profit on selling price
variable (L : ℝ) -- List price
variable (C : ℝ) -- Cost price after discount: C = 0.7 * L
variable (M : ℝ) -- Marked price
variable (S : ℝ) -- Selling price after discount on marked price: S = 0.9 * M

noncomputable def proof_problem : Prop :=
  C = 0.7 * L ∧
  C = 0.75 * S ∧
  S = 0.9 * M ∧
  M = 103.7 / 100 * L

theorem merchant_marked_price (L : ℝ) (C : ℝ) (S : ℝ) (M : ℝ) :
  (C = 0.7 * L) → 
  (C = 0.75 * S) → 
  (S = 0.9 * M) → 
  M = 103.7 / 100 * L :=
by
  sorry

end merchant_marked_price_l99_99940


namespace find_original_polynomial_calculate_correct_result_l99_99647

variable {P : Polynomial ℝ}
variable (Q : Polynomial ℝ := 2 * X ^ 2 + X - 5)
variable (R : Polynomial ℝ := X ^ 2 + 3 * X - 1)

theorem find_original_polynomial (h : P - Q = R) : P = 3 * X ^ 2 + 4 * X - 6 :=
by
  sorry

theorem calculate_correct_result (h : P = 3 * X ^ 2 + 4 * X - 6) : P - Q = X ^ 2 + X + 9 :=
by
  sorry

end find_original_polynomial_calculate_correct_result_l99_99647


namespace ratio_equivalence_to_minutes_l99_99439

-- Define conditions and equivalence
theorem ratio_equivalence_to_minutes :
  ∀ (x : ℝ), (8 / 4 = 8 / x) → x = 4 / 60 :=
by
  intro x
  sorry

end ratio_equivalence_to_minutes_l99_99439


namespace tan_sub_eq_minus_2sqrt3_l99_99862

theorem tan_sub_eq_minus_2sqrt3 
  (h1 : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3)
  (h2 : Real.tan (5 * Real.pi / 12) = 2 + Real.sqrt 3) : 
  Real.tan (Real.pi / 12) - Real.tan (5 * Real.pi / 12) = -2 * Real.sqrt 3 :=
by
  sorry

end tan_sub_eq_minus_2sqrt3_l99_99862


namespace percentage_alcohol_final_l99_99706

-- Let's define the given conditions
variable (A B totalVolume : ℝ)
variable (percentAlcoholA percentAlcoholB : ℝ)
variable (approxA : ℝ)

-- Assume the conditions
axiom condition1 : percentAlcoholA = 0.20
axiom condition2 : percentAlcoholB = 0.50
axiom condition3 : totalVolume = 15
axiom condition4 : approxA = 10
axiom condition5 : A = approxA
axiom condition6 : B = totalVolume - A

-- The proof statement
theorem percentage_alcohol_final : 
  (0.20 * A + 0.50 * B) / 15 * 100 = 30 :=
by 
  -- Introduce enough structure for Lean to handle the problem.
  sorry

end percentage_alcohol_final_l99_99706


namespace total_exercise_hours_l99_99287

-- Define the conditions
def Natasha_minutes_per_day : ℕ := 30
def Natasha_days : ℕ := 7
def Esteban_minutes_per_day : ℕ := 10
def Esteban_days : ℕ := 9
def Charlotte_monday_minutes : ℕ := 20
def Charlotte_wednesday_minutes : ℕ := 45
def Charlotte_thursday_minutes : ℕ := 30
def Charlotte_sunday_minutes : ℕ := 60

-- Sum up the minutes for each individual
def Natasha_total_minutes : ℕ := Natasha_minutes_per_day * Natasha_days
def Esteban_total_minutes : ℕ := Esteban_minutes_per_day * Esteban_days
def Charlotte_total_minutes : ℕ := Charlotte_monday_minutes + Charlotte_wednesday_minutes + Charlotte_thursday_minutes + Charlotte_sunday_minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Calculation of hours for each individual
noncomputable def Natasha_total_hours : ℚ := minutes_to_hours Natasha_total_minutes
noncomputable def Esteban_total_hours : ℚ := minutes_to_hours Esteban_total_minutes
noncomputable def Charlotte_total_hours : ℚ := minutes_to_hours Charlotte_total_minutes

-- Prove total hours of exercise for all three individuals
theorem total_exercise_hours : Natasha_total_hours + Esteban_total_hours + Charlotte_total_hours = 7.5833 := by
  sorry

end total_exercise_hours_l99_99287


namespace cos_3theta_value_l99_99840

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l99_99840


namespace sum_of_variables_l99_99946

theorem sum_of_variables (a b c d : ℕ) (h1 : ac + bd + ad + bc = 1997) : a + b + c + d = 1998 :=
sorry

end sum_of_variables_l99_99946


namespace fodder_lasting_days_l99_99793

theorem fodder_lasting_days (buffalo_fodder_rate cow_fodder_rate ox_fodder_rate : ℕ)
  (initial_buffaloes initial_cows initial_oxen added_buffaloes added_cows initial_days : ℕ)
  (h1 : 3 * buffalo_fodder_rate = 4 * cow_fodder_rate)
  (h2 : 3 * buffalo_fodder_rate = 2 * ox_fodder_rate)
  (h3 : initial_days * (initial_buffaloes * buffalo_fodder_rate + initial_cows * cow_fodder_rate + initial_oxen * ox_fodder_rate) = 4320) :
  (4320 / ((initial_buffaloes + added_buffaloes) * buffalo_fodder_rate + (initial_cows + added_cows) * cow_fodder_rate + initial_oxen * ox_fodder_rate)) = 9 :=
by 
  sorry

end fodder_lasting_days_l99_99793


namespace circle_equation_through_points_l99_99203

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l99_99203


namespace annual_income_before_tax_l99_99050

variable (I : ℝ) -- Define I as the annual income before tax

-- Conditions
def original_tax (I : ℝ) : ℝ := 0.42 * I
def new_tax (I : ℝ) : ℝ := 0.32 * I
def differential_savings (I : ℝ) : ℝ := original_tax I - new_tax I

-- Theorem: Given the conditions, the taxpayer's annual income before tax is $42,400
theorem annual_income_before_tax : differential_savings I = 4240 → I = 42400 := by
  sorry

end annual_income_before_tax_l99_99050


namespace min_value_of_x_plus_y_l99_99594

theorem min_value_of_x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end min_value_of_x_plus_y_l99_99594


namespace problem_statement_l99_99352

open Classical

variable (a_n : ℕ → ℝ) (a1 d : ℝ)

-- Condition: Arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ (n : ℕ), a_n (n + 1) = a1 + n * d 

-- Condition: Geometric relationship between a1, a3, and a9
def geometric_relation (a1 a3 a9 : ℝ) : Prop :=
  a3 / a1 = a9 / a3

-- Given conditions for the arithmetic sequence and geometric relation
axiom arith : arithmetic_sequence a_n a1 d
axiom geom : geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)

theorem problem_statement : d ≠ 0 → (∃ (a1 d : ℝ), d ≠ 0 ∧ arithmetic_sequence a_n a1 d ∧ geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)) → (a1 + 2 * d) / a1 = 3 := by
  sorry

end problem_statement_l99_99352


namespace find_first_number_l99_99943

theorem find_first_number (HCF LCM num2 num1 : ℕ) (hcf_cond : HCF = 20) (lcm_cond : LCM = 396) (num2_cond : num2 = 220) 
    (relation_cond : HCF * LCM = num1 * num2) : num1 = 36 :=
by
  sorry

end find_first_number_l99_99943


namespace negation_example_l99_99970

theorem negation_example : (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negation_example_l99_99970


namespace problem_1_problem_2_l99_99244

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

-- Define proposition q
def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the range of values for a in proposition p
def range_p (a : ℝ) : Prop :=
  a ≤ 1

-- Define set A and set B
def set_A (a : ℝ) : Prop := a ≤ 1
def set_B (a : ℝ) : Prop := a ≥ 1 ∨ a ≤ -2

theorem problem_1 (a : ℝ) (h : proposition_p a) : range_p a := 
sorry

theorem problem_2 (a : ℝ) : 
  (∃ h1 : proposition_p a, set_A a) ∧ (∃ h2 : proposition_q a, set_B a)
  ↔ ¬ ((∃ h1 : proposition_p a, set_B a) ∧ (∃ h2 : proposition_q a, set_A a)) :=
sorry

end problem_1_problem_2_l99_99244


namespace rectangular_prism_inequality_l99_99857

variable {a b c l : ℝ}

theorem rectangular_prism_inequality (h_diag : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
sorry

end rectangular_prism_inequality_l99_99857


namespace prove_fraction_identity_l99_99016

theorem prove_fraction_identity (x y : ℂ) (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) : 
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 41 / 20 := 
by 
  sorry

end prove_fraction_identity_l99_99016


namespace hospital_staff_l99_99558

-- Define the conditions
variables (d n : ℕ) -- d: number of doctors, n: number of nurses
variables (x : ℕ) -- common multiplier

theorem hospital_staff (h1 : d + n = 456) (h2 : 8 * x = d) (h3 : 11 * x = n) : n = 264 :=
by
  -- noncomputable def only when necessary, skipping the proof with sorry
  sorry

end hospital_staff_l99_99558


namespace dogwood_trees_after_planting_l99_99288

-- Define the number of current dogwood trees and the number to be planted.
def current_dogwood_trees : ℕ := 34
def trees_to_be_planted : ℕ := 49

-- Problem statement to prove the total number of dogwood trees after planting.
theorem dogwood_trees_after_planting : current_dogwood_trees + trees_to_be_planted = 83 := by
  -- A placeholder for proof
  sorry

end dogwood_trees_after_planting_l99_99288


namespace kelly_games_giveaway_l99_99000

theorem kelly_games_giveaway (n m g : ℕ) (h_current: n = 50) (h_left: m = 35) : g = n - m :=
by
  sorry

end kelly_games_giveaway_l99_99000


namespace min_value_geq_4_plus_2sqrt2_l99_99026

theorem min_value_geq_4_plus_2sqrt2
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 1)
  (h4: a + b = 1) :
  ( ( (a^2 + 1) / (a * b) - 2 ) * c + (Real.sqrt 2) / (c - 1) ) ≥ (4 + 2 * (Real.sqrt 2)) :=
sorry

end min_value_geq_4_plus_2sqrt2_l99_99026


namespace min_q_of_abs_poly_eq_three_l99_99292

theorem min_q_of_abs_poly_eq_three (p q : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (|x1^2 + p * x1 + q| = 3) ∧ (|x2^2 + p * x2 + q| = 3) ∧ (|x3^2 + p * x3 + q| = 3)) →
  q = -3 :=
sorry

end min_q_of_abs_poly_eq_three_l99_99292


namespace find_present_worth_l99_99532

noncomputable def present_worth (BG : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
(BG * 100) / (R * ((1 + R/100)^T - 1) - R * T)

theorem find_present_worth : present_worth 36 10 3 = 1161.29 :=
by
  sorry

end find_present_worth_l99_99532


namespace daily_pre_promotion_hours_l99_99511

-- Defining conditions
def weekly_additional_hours := 6
def hours_driven_in_two_weeks_after_promotion := 40
def days_in_two_weeks := 14
def hours_added_in_two_weeks := 2 * weekly_additional_hours

-- Math proof problem statement
theorem daily_pre_promotion_hours :
  (hours_driven_in_two_weeks_after_promotion - hours_added_in_two_weeks) / days_in_two_weeks = 2 :=
by
  sorry

end daily_pre_promotion_hours_l99_99511


namespace moles_of_NaOH_l99_99191

-- Statement of the problem conditions and desired conclusion
theorem moles_of_NaOH (moles_H2SO4 moles_NaHSO4 : ℕ) (h : moles_H2SO4 = 3) (h_eq : moles_H2SO4 = moles_NaHSO4) : moles_NaHSO4 = 3 := by
  sorry

end moles_of_NaOH_l99_99191


namespace game_ends_in_36_rounds_l99_99101

theorem game_ends_in_36_rounds 
    (tokens_A : ℕ := 17) (tokens_B : ℕ := 16) (tokens_C : ℕ := 15)
    (rounds : ℕ) 
    (game_rule : (tokens_A tokens_B tokens_C round_num : ℕ) → Prop) 
    (extra_discard_rule : (tokens_A tokens_B tokens_C round_num : ℕ) → Prop)  
    (game_ends_when_token_zero : (tokens_A tokens_B tokens_C : ℕ) → Prop) :
    game_rule tokens_A tokens_B tokens_C rounds ∧
    extra_discard_rule tokens_A tokens_B tokens_C rounds ∧
    game_ends_when_token_zero tokens_A tokens_B tokens_C → 
    rounds = 36 := by
    sorry

end game_ends_in_36_rounds_l99_99101


namespace lathes_equal_parts_processed_15_minutes_l99_99868

variable (efficiencyA efficiencyB efficiencyC : ℝ)
variable (timeA timeB timeC : ℕ)

/-- Lathe A starts 10 minutes before lathe C -/
def start_time_relation_1 : Prop := timeA + 10 = timeC

/-- Lathe C starts 5 minutes before lathe B -/
def start_time_relation_2 : Prop := timeC + 5 = timeB

/-- After lathe B has been working for 10 minutes, B and C process the same number of parts -/
def parts_processed_relation_1 (efficiencyB efficiencyC : ℝ) : Prop :=
  10 * efficiencyB = (10 + 5) * efficiencyC

/-- After lathe C has been working for 30 minutes, A and C process the same number of parts -/
def parts_processed_relation_2 (efficiencyA efficiencyC : ℝ) : Prop :=
  (30 + 10) * efficiencyA = 30 * efficiencyC

/-- How many minutes after lathe B starts will it have processed the same number of standard parts as lathe A? -/
theorem lathes_equal_parts_processed_15_minutes
  (h₁ : start_time_relation_1 timeA timeC)
  (h₂ : start_time_relation_2 timeC timeB)
  (h₃ : parts_processed_relation_1 efficiencyB efficiencyC)
  (h₄ : parts_processed_relation_2 efficiencyA efficiencyC) :
  ∃ t : ℕ, (t = 15) ∧ ( (timeB + t) * efficiencyB = (timeA + (timeB + t - timeA)) * efficiencyA ) := sorry

end lathes_equal_parts_processed_15_minutes_l99_99868


namespace additional_payment_each_friend_l99_99997

theorem additional_payment_each_friend (initial_cost : ℕ) (earned_amount : ℕ) (total_friends : ℕ) (final_friends : ℕ) 
(h_initial_cost : initial_cost = 1700) (h_earned_amount : earned_amount = 500) 
(h_total_friends : total_friends = 6) (h_final_friends : final_friends = 5) : 
  ((initial_cost - earned_amount) / total_friends) / final_friends = 40 :=
sorry

end additional_payment_each_friend_l99_99997


namespace calculate_highest_score_l99_99812

noncomputable def highest_score (avg_60 : ℕ) (delta_HL : ℕ) (avg_58 : ℕ) : ℕ :=
  let total_60 := 60 * avg_60
  let total_58 := 58 * avg_58
  let sum_HL := total_60 - total_58
  let L := (sum_HL - delta_HL) / 2
  let H := L + delta_HL
  H

theorem calculate_highest_score :
  highest_score 55 200 52 = 242 :=
by
  sorry

end calculate_highest_score_l99_99812


namespace expected_adjacent_black_pairs_60_cards_l99_99021

noncomputable def expected_adjacent_black_pairs 
(deck_size : ℕ) (black_cards : ℕ) (red_cards : ℕ) : ℚ :=
  if h : deck_size = black_cards + red_cards 
  then (black_cards:ℚ) * (black_cards - 1) / (deck_size - 1) 
  else 0

theorem expected_adjacent_black_pairs_60_cards :
  expected_adjacent_black_pairs 60 36 24 = 1260 / 59 := by
  sorry

end expected_adjacent_black_pairs_60_cards_l99_99021


namespace right_triangle_area_l99_99491

theorem right_triangle_area (a b : ℝ) (ha : a = 3) (hb : b = 5) : 
  (1 / 2) * a * b = 7.5 := 
by
  rw [ha, hb]
  sorry

end right_triangle_area_l99_99491


namespace paul_crayons_left_l99_99217

theorem paul_crayons_left (initial_crayons lost_crayons : ℕ) 
  (h_initial : initial_crayons = 253) 
  (h_lost : lost_crayons = 70) : (initial_crayons - lost_crayons) = 183 := 
by
  sorry

end paul_crayons_left_l99_99217


namespace range_of_f_l99_99247

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  Set.range f = Set.Icc (Real.pi / 2 + Real.arctan (-2)) (Real.pi / 2 + Real.arctan 2) :=
sorry

end range_of_f_l99_99247


namespace remainder_of_N_mod_37_l99_99595

theorem remainder_of_N_mod_37 (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_of_N_mod_37_l99_99595


namespace p_sufficient_for_q_iff_l99_99431

-- Definitions based on conditions
def p (x : ℝ) : Prop := x^2 - 2 * x - 8 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) ≤ 0
def m_condition (m : ℝ) : Prop := m < 0

-- The statement to prove
theorem p_sufficient_for_q_iff (m : ℝ) :
  (∀ x, p x → q x m) ↔ m <= -3 :=
by
  sorry

-- noncomputable theory is not necessary here since all required functions are computable.

end p_sufficient_for_q_iff_l99_99431


namespace prime_sum_is_prime_l99_99702

def prime : ℕ → Prop := sorry 

theorem prime_sum_is_prime (A B : ℕ) (hA : prime A) (hB : prime B) (hAB : prime (A - B)) (hABB : prime (A - B - B)) : prime (A + B + (A - B) + (A - B - B)) :=
sorry

end prime_sum_is_prime_l99_99702


namespace find_xyz_l99_99242

open Complex

theorem find_xyz (a b c x y z : ℂ)
(h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0)
(h7 : a = (b + c) / (x - 3)) (h8 : b = (a + c) / (y - 3)) (h9 : c = (a + b) / (z - 3))
(h10 : x * y + x * z + y * z = 10) (h11 : x + y + z = 6) : 
(x * y * z = 15) :=
by
  sorry

end find_xyz_l99_99242


namespace candy_bars_weeks_l99_99487

theorem candy_bars_weeks (buy_per_week : ℕ) (eat_per_4_weeks : ℕ) (saved_candies : ℕ) (weeks_passed : ℕ) :
  (buy_per_week = 2) →
  (eat_per_4_weeks = 1) →
  (saved_candies = 28) →
  (weeks_passed = 4 * (saved_candies / (4 * buy_per_week - eat_per_4_weeks))) →
  weeks_passed = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end candy_bars_weeks_l99_99487


namespace range_of_a_l99_99835

noncomputable def f (a x : ℝ) := (a - Real.sin x) / Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (π / 6 < x) → (x < π / 3) → (f a x) ≤ (f a (x + ε))) → 2 ≤ a :=
by
  sorry

end range_of_a_l99_99835


namespace find_f_neg2016_l99_99408

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg2016 (a b k : ℝ) (h : f a b 2016 = k) (h_ab : a * b ≠ 0) : f a b (-2016) = 2 - k :=
by
  sorry

end find_f_neg2016_l99_99408


namespace ben_bonus_leftover_l99_99246

theorem ben_bonus_leftover (b : ℝ) (k h c : ℝ) (bk : k = 1/22 * b) (bh : h = 1/4 * b) (bc : c = 1/8 * b) :
  b - (k + h + c) = 867 :=
by
  sorry

end ben_bonus_leftover_l99_99246


namespace area_of_square_STUV_l99_99443

-- Defining the conditions
variable (C L : ℝ)
variable (h1 : 2 * (C + L) = 40)

-- The goal is to prove the area of the square STUV
theorem area_of_square_STUV : (C + L) * (C + L) = 400 :=
by
  sorry

end area_of_square_STUV_l99_99443


namespace sum_of_smallest_and_largest_even_l99_99423

theorem sum_of_smallest_and_largest_even (n : ℤ) (h : n + (n + 2) + (n + 4) = 1194) : n + (n + 4) = 796 :=
by
  sorry

end sum_of_smallest_and_largest_even_l99_99423


namespace value_of_expression_l99_99291

variables {a b c : ℝ}

theorem value_of_expression (h1 : a * b * c = 10) (h2 : a + b + c = 15) (h3 : a * b + b * c + c * a = 25) :
  (2 + a) * (2 + b) * (2 + c) = 128 := 
sorry

end value_of_expression_l99_99291


namespace minimum_cost_l99_99656

theorem minimum_cost (
    x y m w : ℝ) 
    (h1 : 4 * x + 2 * y = 400)
    (h2 : 2 * x + 4 * y = 320)
    (h3 : m ≥ 16)
    (h4 : m + (80 - m) = 80)
    (h5 : w = 80 * m + 40 * (80 - m)) :
    x = 80 ∧ y = 40 ∧ w = 3840 :=
by 
  sorry

end minimum_cost_l99_99656


namespace total_amount_paid_l99_99886

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_tshirts : ℕ := 6

theorem total_amount_paid : 
  (number_of_tshirts : ℝ) * (original_price * discount_rate) = 60 := by
  sorry

end total_amount_paid_l99_99886


namespace hexagon_area_l99_99446

-- Definitions of the conditions
def DEF_perimeter := 42
def circumcircle_radius := 10
def area_of_hexagon_DE'F'D'E'F := 210

-- The theorem statement
theorem hexagon_area (DEF_perimeter : ℕ) (circumcircle_radius : ℕ) : Prop :=
  DEF_perimeter = 42 → circumcircle_radius = 10 → 
  area_of_hexagon_DE'F'D'E'F = 210

-- Example invocation of the theorem, proof omitted.
example : hexagon_area DEF_perimeter circumcircle_radius :=
by {
  sorry
}

end hexagon_area_l99_99446


namespace solve_equation_l99_99664

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end solve_equation_l99_99664


namespace cost_price_of_article_l99_99920

-- Definitions based on the conditions
def sellingPrice : ℝ := 800
def profitPercentage : ℝ := 25

-- Statement to prove the cost price
theorem cost_price_of_article :
  ∃ cp : ℝ, profitPercentage = ((sellingPrice - cp) / cp) * 100 ∧ cp = 640 :=
by
  sorry

end cost_price_of_article_l99_99920


namespace hot_dogs_total_l99_99593

theorem hot_dogs_total (D : ℕ)
  (h1 : 9 = 2 * D + D + 3) :
  (2 * D + 9 + D = 15) :=
by sorry

end hot_dogs_total_l99_99593


namespace problem1_problem2_l99_99282

-- Proof of Problem 1
theorem problem1 (x y : ℤ) (h1 : x = -2) (h2 : y = -3) : (6 * x - 5 * y + 3 * y - 2 * x) = -2 :=
by
  sorry

-- Proof of Problem 2
theorem problem2 (a : ℚ) (h : a = -1 / 2) : (1 / 4 * (-4 * a^2 + 2 * a - 8) - (1 / 2 * a - 2)) = -1 / 4 :=
by
  sorry

end problem1_problem2_l99_99282


namespace first_variety_cost_l99_99450

noncomputable def cost_of_second_variety : ℝ := 8.75
noncomputable def ratio_of_first_variety : ℚ := 5 / 6
noncomputable def ratio_of_second_variety : ℚ := 1 - ratio_of_first_variety
noncomputable def cost_of_mixture : ℝ := 7.50

theorem first_variety_cost :
  ∃ x : ℝ, x * (ratio_of_first_variety : ℝ) + cost_of_second_variety * (ratio_of_second_variety : ℝ) = cost_of_mixture * (ratio_of_first_variety + ratio_of_second_variety : ℝ) 
    ∧ x = 7.25 :=
sorry

end first_variety_cost_l99_99450


namespace smallest_positive_integer_exists_l99_99489

theorem smallest_positive_integer_exists
    (x : ℕ) :
    (x % 7 = 2) ∧
    (x % 4 = 3) ∧
    (x % 6 = 1) →
    x = 135 :=
by
    sorry

end smallest_positive_integer_exists_l99_99489


namespace find_yellow_shells_l99_99134

-- Define the conditions
def total_shells : ℕ := 65
def purple_shells : ℕ := 13
def pink_shells : ℕ := 8
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

-- Define the result as the proof goal
theorem find_yellow_shells (total_shells purple_shells pink_shells blue_shells orange_shells : ℕ) : 
  total_shells = 65 →
  purple_shells = 13 →
  pink_shells = 8 →
  blue_shells = 12 →
  orange_shells = 14 →
  65 - (13 + 8 + 12 + 14) = 18 :=
by
  intros
  sorry

end find_yellow_shells_l99_99134


namespace correct_statement_c_l99_99188

-- Definitions
variables {Point : Type*} {Line Plane : Type*}
variables (l m : Line) (α β : Plane)

-- Conditions
def parallel_planes (α β : Plane) : Prop := sorry  -- α ∥ β
def perpendicular_line_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊥ α
def line_in_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊂ α
def line_perpendicular (l m : Line) : Prop := sorry  -- l ⊥ m

-- Theorem to be proven
theorem correct_statement_c 
  (α β : Plane) (l : Line)
  (h_parallel : parallel_planes α β)
  (h_perpendicular : perpendicular_line_plane l α) :
  ∀ (m : Line), line_in_plane m β → line_perpendicular m l := 
sorry

end correct_statement_c_l99_99188


namespace roommate_payment_l99_99497

theorem roommate_payment :
  (1100 + 114 + 300) / 2 = 757 := 
by
  sorry

end roommate_payment_l99_99497


namespace f_7_minus_a_eq_neg_7_over_4_l99_99784

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2 else -Real.logb 3 x

variable (a : ℝ)

-- Given conditions
axiom h1 : f a = -2

-- The proof of the required condition
theorem f_7_minus_a_eq_neg_7_over_4 (h1 : f a = -2) : f (7 - a) = -7 / 4 := sorry

end f_7_minus_a_eq_neg_7_over_4_l99_99784


namespace students_in_front_l99_99202

theorem students_in_front (total_students : ℕ) (students_behind : ℕ) (students_total : total_students = 25) (behind_Yuna : students_behind = 9) :
  (total_students - (students_behind + 1)) = 15 :=
by
  sorry

end students_in_front_l99_99202


namespace find_f2_l99_99081

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y + 1)
  (H2 : f 8 = 15) :
  f 2 = 3 := 
sorry

end find_f2_l99_99081


namespace minimum_value_problem_l99_99143

open Real

theorem minimum_value_problem (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 6) :
  9 / x + 16 / y + 25 / z ≥ 24 :=
by
  sorry

end minimum_value_problem_l99_99143


namespace no_integer_solution_for_150_l99_99230

theorem no_integer_solution_for_150 : ∀ (x : ℤ), x - Int.sqrt x ≠ 150 := 
sorry

end no_integer_solution_for_150_l99_99230


namespace equation_solution_l99_99009

noncomputable def solve_equation : Prop :=
∃ (x : ℝ), x^6 + (3 - x)^6 = 730 ∧ (x = 1.5 + Real.sqrt 5 ∨ x = 1.5 - Real.sqrt 5)

theorem equation_solution : solve_equation :=
sorry

end equation_solution_l99_99009


namespace angle_invariant_under_magnification_l99_99374

theorem angle_invariant_under_magnification :
  ∀ (angle magnification : ℝ), angle = 10 → magnification = 5 → angle = 10 := by
  intros angle magnification h_angle h_magnification
  exact h_angle

end angle_invariant_under_magnification_l99_99374


namespace log_inequality_l99_99461

theorem log_inequality (a x y : ℝ) (ha : 0 < a) (ha_lt_1 : a < 1) 
(h : x^2 + y = 0) : 
  Real.log (a^x + a^y) / Real.log a ≤ Real.log 2 / Real.log a + 1 / 8 :=
sorry

end log_inequality_l99_99461


namespace initial_pretzels_in_bowl_l99_99544

-- Definitions and conditions
def John_pretzels := 28
def Alan_pretzels := John_pretzels - 9
def Marcus_pretzels := John_pretzels + 12
def Marcus_pretzels_actual := 40

-- The main theorem stating the initial number of pretzels in the bowl
theorem initial_pretzels_in_bowl : 
  Marcus_pretzels = Marcus_pretzels_actual → 
  John_pretzels + Alan_pretzels + Marcus_pretzels = 87 :=
by
  intro h
  sorry -- proof to be filled in

end initial_pretzels_in_bowl_l99_99544


namespace trigonometric_identity_l99_99953

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end trigonometric_identity_l99_99953


namespace homes_distance_is_65_l99_99614

noncomputable def distance_between_homes
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (maxwell_distance : ℕ)
  (time : ℕ) : ℕ :=
  maxwell_distance + brad_speed * time

theorem homes_distance_is_65
  (maxwell_speed : ℕ := 2)
  (brad_speed : ℕ := 3)
  (maxwell_distance : ℕ := 26)
  (time : ℕ := maxwell_distance / maxwell_speed) :
  distance_between_homes maxwell_speed brad_speed maxwell_distance time = 65 :=
by 
  sorry

end homes_distance_is_65_l99_99614


namespace trigonometric_inequality_l99_99799

-- Let \( f(x) \) be defined as \( cos \, x \)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Given a, b, c are the sides of triangle ∆ABC opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}

-- Condition: \( 3a^2 + 3b^2 - c^2 = 4ab \)
variable (h : 3 * a^2 + 3 * b^2 - c^2 = 4 * a * b)

-- Goal: Prove that \( f(\cos A) \leq f(\sin B) \)
theorem trigonometric_inequality (h1 : A + B + C = π) (h2 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) : 
  f (Real.cos A) ≤ f (Real.sin B) :=
by
  sorry

end trigonometric_inequality_l99_99799


namespace trig_identity_sum_l99_99758

-- Define the trigonometric functions and their properties
def sin_210_eq : Real.sin (210 * Real.pi / 180) = - Real.sin (30 * Real.pi / 180) := by
  sorry

def cos_60_eq : Real.cos (60 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) := by
  sorry

-- The goal is to prove that the sum of these specific trigonometric values is 0
theorem trig_identity_sum : Real.sin (210 * Real.pi / 180) + Real.cos (60 * Real.pi / 180) = 0 := by
  rw [sin_210_eq, cos_60_eq]
  sorry

end trig_identity_sum_l99_99758


namespace evaluate_expression_l99_99437

theorem evaluate_expression (x y z : ℝ) : 
  (x + (y + z)) - ((-x + y) + z) = 2 * x := 
by
  sorry

end evaluate_expression_l99_99437


namespace function_identity_l99_99330

theorem function_identity (f : ℕ → ℕ) 
  (h_pos : f 1 > 0) 
  (h_property : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l99_99330


namespace find_D_l99_99854

theorem find_D (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : D = 1 :=
by
  sorry

end find_D_l99_99854


namespace initial_pens_count_l99_99099

theorem initial_pens_count (P : ℕ) (h : 2 * (P + 22) - 19 = 75) : P = 25 :=
by
  sorry

end initial_pens_count_l99_99099


namespace find_numbers_l99_99419

theorem find_numbers (x y : ℕ) :
  x + y = 1244 →
  10 * x + 3 = (y - 2) / 10 →
  x = 12 ∧ y = 1232 :=
by
  intro h_sum h_trans
  -- We'll use sorry here to state that the proof is omitted.
  sorry

end find_numbers_l99_99419


namespace hazel_additional_days_l99_99234

theorem hazel_additional_days (school_year_days : ℕ) (miss_percent : ℝ) (already_missed : ℕ)
  (h1 : school_year_days = 180)
  (h2 : miss_percent = 0.05)
  (h3 : already_missed = 6) :
  (⌊miss_percent * school_year_days⌋ - already_missed) = 3 :=
by
  sorry

end hazel_additional_days_l99_99234


namespace cars_overtake_distance_l99_99582

def speed_red_car : ℝ := 30
def speed_black_car : ℝ := 50
def time_to_overtake : ℝ := 1
def distance_between_cars : ℝ := 20

theorem cars_overtake_distance :
  (speed_black_car - speed_red_car) * time_to_overtake = distance_between_cars :=
by sorry

end cars_overtake_distance_l99_99582


namespace solve_for_x_l99_99570

/-- Given condition that 0.75 : x :: 5 : 9 -/
def ratio_condition (x : ℝ) : Prop := 0.75 / x = 5 / 9

theorem solve_for_x (x : ℝ) (h : ratio_condition x) : x = 1.35 := by
  sorry

end solve_for_x_l99_99570


namespace find_x_y_z_sum_l99_99934

theorem find_x_y_z_sum :
  ∃ (x y z : ℝ), 
    x^2 + 27 = -8 * y + 10 * z ∧
    y^2 + 196 = 18 * z + 13 * x ∧
    z^2 + 119 = -3 * x + 30 * y ∧
    x + 3 * y + 5 * z = 127.5 :=
sorry

end find_x_y_z_sum_l99_99934


namespace correct_option_B_l99_99418

def linear_function (x : ℝ) : ℝ := -x + 2

theorem correct_option_B :
  ∃ x : ℝ, linear_function x = 0 ∧ x = 2 :=
by
  sorry

end correct_option_B_l99_99418


namespace distinct_ways_to_divide_books_l99_99834

theorem distinct_ways_to_divide_books : 
  ∃ (ways : ℕ), ways = 5 := sorry

end distinct_ways_to_divide_books_l99_99834


namespace g_nested_result_l99_99768

def g (n : ℕ) : ℕ :=
if n < 5 then
  n^2 + 1
else
  2 * n + 3

theorem g_nested_result : g (g (g 3)) = 49 := by
sorry

end g_nested_result_l99_99768


namespace solve_equation_l99_99972

theorem solve_equation (x : ℝ) :
  (2 * x - 1)^2 - 25 = 0 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end solve_equation_l99_99972


namespace calculate_leakage_rate_l99_99802

variable (B : ℕ) (T : ℕ) (R : ℝ)

-- B represents the bucket's capacity in ounces, T represents time in hours, R represents the rate of leakage per hour in ounces per hour.

def leakage_rate (B : ℕ) (T : ℕ) (R : ℝ) : Prop :=
  (B = 36) ∧ (T = 12) ∧ (B / 2 = T * R)

theorem calculate_leakage_rate : leakage_rate 36 12 1.5 :=
by 
  simp [leakage_rate]
  sorry

end calculate_leakage_rate_l99_99802


namespace iris_jackets_l99_99453

theorem iris_jackets (J : ℕ) (h : 10 * J + 12 + 48 = 90) : J = 3 :=
by
  sorry

end iris_jackets_l99_99453


namespace max_volume_of_sphere_in_cube_l99_99399

theorem max_volume_of_sphere_in_cube (a : ℝ) (h : a = 1) : 
  ∃ V, V = π / 6 ∧ 
        ∀ (r : ℝ), r = a / 2 →
        V = (4 / 3) * π * r^3 :=
by
  sorry

end max_volume_of_sphere_in_cube_l99_99399


namespace number_of_dogs_is_112_l99_99648

-- Definitions based on the given conditions.
def ratio_dogs_to_cats_to_bunnies (D C B : ℕ) : Prop := 4 * C = 7 * D ∧ 9 * C = 7 * B
def total_dogs_and_bunnies (D B : ℕ) (total : ℕ) : Prop := D + B = total

-- The hypothesis and conclusion of the problem.
theorem number_of_dogs_is_112 (D C B : ℕ) (x : ℕ) (h1: ratio_dogs_to_cats_to_bunnies D C B) (h2: total_dogs_and_bunnies D B 364) : D = 112 :=
by 
  sorry

end number_of_dogs_is_112_l99_99648


namespace tickets_spent_on_hat_l99_99441

def tickets_won_whack_a_mole := 32
def tickets_won_skee_ball := 25
def tickets_left := 50
def total_tickets := tickets_won_whack_a_mole + tickets_won_skee_ball

theorem tickets_spent_on_hat : 
  total_tickets - tickets_left = 7 :=
by
  sorry

end tickets_spent_on_hat_l99_99441


namespace reading_time_difference_l99_99535

theorem reading_time_difference 
  (xanthia_reading_speed : ℕ) 
  (molly_reading_speed : ℕ) 
  (book_pages : ℕ) 
  (time_conversion_factor : ℕ)
  (hx : xanthia_reading_speed = 150)
  (hm : molly_reading_speed = 75)
  (hp : book_pages = 300)
  (ht : time_conversion_factor = 60) :
  ((book_pages / molly_reading_speed - book_pages / xanthia_reading_speed) * time_conversion_factor = 120) := 
by
  sorry

end reading_time_difference_l99_99535


namespace line_canonical_form_l99_99921

theorem line_canonical_form :
  ∃ (x y z : ℝ),
  x + y + z - 2 = 0 ∧
  x - y - 2 * z + 2 = 0 →
  ∃ (k : ℝ),
  x / k = -1 ∧
  (y - 2) / (3 * k) = 1 ∧
  z / (-2 * k) = 1 :=
sorry

end line_canonical_form_l99_99921


namespace find_the_number_l99_99509

theorem find_the_number : ∃ x : ℝ, (10 + x + 50) / 3 = (20 + 40 + 6) / 3 + 8 ∧ x = 30 := 
by
  sorry

end find_the_number_l99_99509


namespace sum_of_squares_not_divisible_by_17_l99_99145

theorem sum_of_squares_not_divisible_by_17
  (x y z : ℤ)
  (h_sum_div : 17 ∣ (x + y + z))
  (h_prod_div : 17 ∣ (x * y * z))
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_coprime_zx : Int.gcd z x = 1) :
  ¬ (17 ∣ (x^2 + y^2 + z^2)) := 
sorry

end sum_of_squares_not_divisible_by_17_l99_99145


namespace cylinder_height_to_diameter_ratio_l99_99629

theorem cylinder_height_to_diameter_ratio
  (r h : ℝ)
  (inscribed_sphere : h = 2 * r)
  (cylinder_volume : π * r^2 * h = 3 * (4/3) * π * r^3) :
  (h / (2 * r)) = 2 :=
by
  sorry

end cylinder_height_to_diameter_ratio_l99_99629


namespace sum_of_ages_l99_99564

-- Given conditions and definitions
variables (M J : ℝ)

def condition1 : Prop := M = J + 8
def condition2 : Prop := M + 6 = 3 * (J - 3)

-- Proof goal
theorem sum_of_ages (h1 : condition1 M J) (h2 : condition2 M J) : M + J = 31 := 
by sorry

end sum_of_ages_l99_99564


namespace value_of_a_b_l99_99342

theorem value_of_a_b (a b : ℕ) (ha : 2 * 100 + a * 10 + 3 + 326 = 5 * 100 + b * 10 + 9) (hb : (5 + b + 9) % 9 = 0): 
  a + b = 6 := 
sorry

end value_of_a_b_l99_99342


namespace find_four_digit_number_abcd_exists_l99_99807

theorem find_four_digit_number_abcd_exists (M : ℕ) (H1 : M > 0) (H2 : M % 10 ≠ 0) 
    (H3 : M % 100000 = M^2 % 100000) : ∃ abcd : ℕ, abcd = 2502 :=
by
  -- Proof is omitted
  sorry

end find_four_digit_number_abcd_exists_l99_99807


namespace product_of_consecutive_even_numbers_divisible_by_8_l99_99888

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 8 ∣ (2 * n * (2 * n + 2)) :=
by sorry

end product_of_consecutive_even_numbers_divisible_by_8_l99_99888


namespace find_rate_percent_l99_99041

theorem find_rate_percent 
  (P : ℝ) 
  (r : ℝ) 
  (h1 : 2420 = P * (1 + r / 100)^2) 
  (h2 : 3025 = P * (1 + r / 100)^3) : 
  r = 25 :=
by
  sorry

end find_rate_percent_l99_99041


namespace savings_after_one_year_l99_99651

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem savings_after_one_year :
  compound_interest 1000 0.10 2 1 = 1102.50 :=
by
  sorry

end savings_after_one_year_l99_99651


namespace smallest_positive_x_l99_99722

theorem smallest_positive_x
  (x : ℕ)
  (h1 : x % 3 = 2)
  (h2 : x % 7 = 6)
  (h3 : x % 8 = 7) : x = 167 :=
by
  sorry

end smallest_positive_x_l99_99722


namespace product_mod_7_zero_l99_99083

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l99_99083


namespace total_handshakes_l99_99877

def people := 40
def groupA := 25
def groupB := 15
def knownByGroupB (x : ℕ) : ℕ := 5
def interactionsWithinGroupB : ℕ := 105
def interactionsBetweenGroups : ℕ := 75

theorem total_handshakes : (groupB * knownByGroupB 0) + interactionsWithinGroupB = 180 :=
by
  sorry

end total_handshakes_l99_99877


namespace solution_set_of_inequality_system_l99_99226

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end solution_set_of_inequality_system_l99_99226


namespace michael_truck_meetings_l99_99482

theorem michael_truck_meetings :
  let michael_speed := 6
  let truck_speed := 12
  let pail_distance := 200
  let truck_stop_time := 20
  let initial_distance := pail_distance
  ∃ (meetings : ℕ), 
  (michael_speed, truck_speed, pail_distance, truck_stop_time, initial_distance, meetings) = 
  (6, 12, 200, 20, 200, 10) :=
sorry

end michael_truck_meetings_l99_99482


namespace power_sum_identity_l99_99808

theorem power_sum_identity (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) : 
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 := 
by
  sorry

end power_sum_identity_l99_99808


namespace range_of_t_l99_99756

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (a t : ℝ) := 2 * a * t - t^2

theorem range_of_t (t : ℝ) (a : ℝ) (x : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x)
                   (h₂ : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 1 → f x₁ ≤ f x₂)
                   (h₃ : f (-1) = -1) (h₄ : -1 ≤ x ∧ x ≤ 1 → f x ≤ t^2 - 2 * a * t + 1)
                   (h₅ : -1 ≤ a ∧ a ≤ 1) :
  t ≥ 2 ∨ t = 0 ∨ t ≤ -2 := sorry

end range_of_t_l99_99756


namespace min_number_knights_l99_99587

theorem min_number_knights (h1 : ∃ n : ℕ, n = 7) (h2 : ∃ s : ℕ, s = 42) (h3 : ∃ l : ℕ, l = 24) :
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 7 ∧ k * (7 - k) = 12 ∧ k = 3 :=
by
  sorry

end min_number_knights_l99_99587


namespace find_larger_number_l99_99484

theorem find_larger_number (x y : ℤ) (h1 : 4 * y = 3 * x) (h2 : y - x = 12) : y = -36 := 
by sorry

end find_larger_number_l99_99484


namespace slope_undefined_iff_vertical_l99_99670

theorem slope_undefined_iff_vertical (m : ℝ) :
  let M := (2 * m + 3, m)
  let N := (m - 2, 1)
  (2 * m + 3 - (m - 2) = 0 ∧ m - 1 ≠ 0) ↔ m = -5 :=
by
  sorry

end slope_undefined_iff_vertical_l99_99670


namespace total_amount_shared_l99_99209

theorem total_amount_shared
  (A B C : ℕ)
  (h_ratio : A / 2 = B / 3 ∧ B / 3 = C / 8)
  (h_Ben_share : B = 30) : A + B + C = 130 :=
by
  -- Add placeholder for the proof.
  sorry

end total_amount_shared_l99_99209


namespace runners_meet_again_l99_99207

theorem runners_meet_again 
  (v1 v2 v3 v4 v5 : ℕ)
  (h1 : v1 = 32) 
  (h2 : v2 = 40) 
  (h3 : v3 = 48) 
  (h4 : v4 = 56) 
  (h5 : v5 = 64) 
  (h6 : 400 % (v2 - v1) = 0)
  (h7 : 400 % (v3 - v2) = 0)
  (h8 : 400 % (v4 - v3) = 0)
  (h9 : 400 % (v5 - v4) = 0) :
  ∃ t : ℕ, t = 500 :=
by sorry

end runners_meet_again_l99_99207


namespace number_of_valid_m_values_l99_99002

noncomputable def polynomial (m : ℤ) (x : ℤ) : ℤ := 
  2 * (m - 1) * x ^ 2 - (m ^ 2 - m + 12) * x + 6 * m

noncomputable def discriminant (m : ℤ) : ℤ :=
  (m ^ 2 - m + 12) ^ 2 - 4 * 2 * (m - 1) * 6 * m

def is_perfect_square (n : ℤ) : Prop :=
  ∃ (k : ℤ), k * k = n

def has_integral_roots (m : ℤ) : Prop :=
  ∃ (r1 r2 : ℤ), polynomial m r1 = 0 ∧ polynomial m r2 = 0

def valid_m_values (m : ℤ) : Prop :=
  (discriminant m) > 0 ∧ is_perfect_square (discriminant m) ∧ has_integral_roots m

theorem number_of_valid_m_values : 
  (∃ M : List ℤ, (∀ m ∈ M, valid_m_values m) ∧ M.length = 4) :=
  sorry

end number_of_valid_m_values_l99_99002


namespace Megatech_budget_allocation_l99_99796

theorem Megatech_budget_allocation :
  let total_degrees := 360
  let degrees_astrophysics := 90
  let home_electronics := 19
  let food_additives := 10
  let genetically_modified_microorganisms := 24
  let industrial_lubricants := 8

  let percentage_astrophysics := (degrees_astrophysics / total_degrees) * 100
  let known_percentages_sum := home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + percentage_astrophysics
  let percentage_microphotonics := 100 - known_percentages_sum

  percentage_microphotonics = 14 :=
by
  sorry

end Megatech_budget_allocation_l99_99796


namespace problem1_solution_set_problem2_range_of_m_l99_99640

def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

theorem problem1_solution_set :
  {x : ℝ | f x ≤ 2} = {x : ℝ | -4 ≤ x ∧ x ≤ 10} := 
sorry

theorem problem2_range_of_m (m : ℝ) (h : ∃ x : ℝ, f x - g x ≥ m - 3) :
  m ≤ 5 :=
sorry

end problem1_solution_set_problem2_range_of_m_l99_99640


namespace trader_goal_l99_99602

theorem trader_goal 
  (profit : ℕ)
  (half_profit : ℕ)
  (donation : ℕ)
  (total_funds : ℕ)
  (made_above_goal : ℕ)
  (goal : ℕ)
  (h1 : profit = 960)
  (h2 : half_profit = profit / 2)
  (h3 : donation = 310)
  (h4 : total_funds = half_profit + donation)
  (h5 : made_above_goal = 180)
  (h6 : goal = total_funds - made_above_goal) :
  goal = 610 :=
by 
  sorry

end trader_goal_l99_99602


namespace cost_of_calf_l99_99348

theorem cost_of_calf (C : ℝ) (total_cost : ℝ) (cow_to_calf_ratio : ℝ) :
  total_cost = 990 ∧ cow_to_calf_ratio = 8 ∧ total_cost = C + 8 * C → C = 110 := by
  sorry

end cost_of_calf_l99_99348


namespace find_x_for_f_of_one_fourth_l99_99753

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
if h : x < 1 then 2^(-x) else Real.log x / Real.log 4 

-- Define the proof problem
theorem find_x_for_f_of_one_fourth : 
  ∃ x : ℝ, (f x = 1 / 4) ∧ (x = Real.sqrt 2)  :=
sorry

end find_x_for_f_of_one_fourth_l99_99753


namespace value_of_f_12_l99_99459

theorem value_of_f_12 (f : ℕ → ℤ) 
  (h1 : f 2 = 5)
  (h2 : f 3 = 7)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → f m + f n = f (m * n)) :
  f 12 = 17 :=
by
  sorry

end value_of_f_12_l99_99459


namespace point_transformation_correct_l99_99847

-- Define the rectangular coordinate system O-xyz
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the point in the original coordinate system
def originalPoint : Point3D := { x := 1, y := -2, z := 3 }

-- Define the transformation function for the yOz plane
def transformToYOzPlane (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

-- Define the expected transformed point
def transformedPoint : Point3D := { x := -1, y := -2, z := 3 }

-- State the theorem to be proved
theorem point_transformation_correct :
  transformToYOzPlane originalPoint = transformedPoint :=
by
  sorry

end point_transformation_correct_l99_99847


namespace range_of_k_l99_99199

theorem range_of_k (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → y^2 = 2 * x → (∃! (x₀ y₀ : ℝ), y₀ = k * x₀ + 1 ∧ y₀^2 = 2 * x₀)) ↔ 
  (k = 0 ∨ k ≥ 1/2) :=
sorry

end range_of_k_l99_99199


namespace find_x_from_percents_l99_99438

theorem find_x_from_percents (x : ℝ) (h : 0.65 * x = 0.20 * 487.50) : x = 150 :=
by
  -- Distilled condition from problem
  have h1 : 0.65 * x = 0.20 * 487.50 := h
  -- Start actual logic here
  sorry

end find_x_from_percents_l99_99438


namespace balloons_total_l99_99632

theorem balloons_total (a b : ℕ) (h1 : a = 47) (h2 : b = 13) : a + b = 60 := 
by
  -- Since h1 and h2 provide values for a and b respectively,
  -- the result can be proved using these values.
  sorry

end balloons_total_l99_99632


namespace rectangle_width_l99_99123

theorem rectangle_width (L W : ℝ) 
  (h1 : L * W = 750) 
  (h2 : 2 * L + 2 * W = 110) : 
  W = 25 :=
sorry

end rectangle_width_l99_99123


namespace compound_interest_correct_l99_99426

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem compound_interest_correct (SI R T : ℝ) (hSI : SI = 58) (hR : R = 5) (hT : T = 2) : 
  compound_interest (SI * 100 / (R * T)) R T = 59.45 :=
by
  sorry

end compound_interest_correct_l99_99426


namespace autumn_pencils_l99_99889

-- Define the conditions of the problem.
def initial_pencils := 20
def misplaced_pencils := 7
def broken_pencils := 3
def found_pencils := 4
def bought_pencils := 2

-- Define the number of pencils lost and gained.
def pencils_lost := misplaced_pencils + broken_pencils
def pencils_gained := found_pencils + bought_pencils

-- Define the final number of pencils.
def final_pencils := initial_pencils - pencils_lost + pencils_gained

-- The theorem we want to prove.
theorem autumn_pencils : final_pencils = 16 := by
  sorry

end autumn_pencils_l99_99889


namespace range_of_k_for_distinct_real_roots_l99_99727

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ k*x1^2 - 2*x1 - 1 = 0 ∧ k*x2^2 - 2*x2 - 1 = 0) → k > -1 ∧ k ≠ 0 :=
by
  sorry

end range_of_k_for_distinct_real_roots_l99_99727


namespace total_time_required_l99_99034

noncomputable def walking_speed_flat : ℝ := 4
noncomputable def walking_speed_uphill : ℝ := walking_speed_flat * 0.8

noncomputable def running_speed_flat : ℝ := 8
noncomputable def running_speed_uphill : ℝ := running_speed_flat * 0.7

noncomputable def distance_walked_uphill : ℝ := 2
noncomputable def distance_run_uphill : ℝ := 1
noncomputable def distance_run_flat : ℝ := 1

noncomputable def time_walk_uphill := distance_walked_uphill / walking_speed_uphill
noncomputable def time_run_uphill := distance_run_uphill / running_speed_uphill
noncomputable def time_run_flat := distance_run_flat / running_speed_flat

noncomputable def total_time := time_walk_uphill + time_run_uphill + time_run_flat

theorem total_time_required :
  total_time = 0.9286 := by
  sorry

end total_time_required_l99_99034


namespace fabric_ratio_l99_99556

theorem fabric_ratio
  (d_m : ℕ) (d_t : ℕ) (d_w : ℕ) (cost : ℕ) (total_revenue : ℕ) (revenue_monday : ℕ) (revenue_tuesday : ℕ) (revenue_wednesday : ℕ)
  (h_d_m : d_m = 20)
  (h_cost : cost = 2)
  (h_d_w : d_w = d_t / 4)
  (h_total_revenue : total_revenue = 140)
  (h_revenue : revenue_monday + revenue_tuesday + revenue_wednesday = total_revenue)
  (h_r_m : revenue_monday = d_m * cost)
  (h_r_t : revenue_tuesday = d_t * cost) 
  (h_r_w : revenue_wednesday = d_w * cost) :
  (d_t / d_m = 1) :=
by
  sorry

end fabric_ratio_l99_99556


namespace teal_more_blue_proof_l99_99814

theorem teal_more_blue_proof (P G B N : ℕ) (hP : P = 150) (hG : G = 90) (hB : B = 40) (hN : N = 25) : 
  (∃ (x : ℕ), x = 75) :=
by
  sorry

end teal_more_blue_proof_l99_99814


namespace value_of_a_minus_b_l99_99518

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = - (a + b)) :
  a - b = -2 ∨ a - b = -6 := sorry

end value_of_a_minus_b_l99_99518


namespace smallest_positive_m_l99_99301

theorem smallest_positive_m (m : ℕ) (h : ∀ (n : ℕ), n % 2 = 1 → (529^n + m * 132^n) % 262417 = 0) : m = 1 :=
sorry

end smallest_positive_m_l99_99301


namespace problem_statement_l99_99839

theorem problem_statement (a b : ℝ) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) (h_b_gt_1 : b > 1)
  (h1 : a = 1) (h2 : 1/2 * (b - 1)^2 + 1 = b) : a + b = 4 :=
sorry

end problem_statement_l99_99839


namespace radar_placement_and_coverage_area_l99_99177

noncomputable def max_distance (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (15 : ℝ) / Real.sin (Real.pi / n)

noncomputable def coverage_area (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (480 : ℝ) * Real.pi / Real.tan (Real.pi / n)

theorem radar_placement_and_coverage_area 
  (n : ℕ) (r w : ℝ) (hn : n = 8) (hr : r = 17) (hw : w = 16) :
  max_distance n r w = (15 : ℝ) / Real.sin (Real.pi / 8) ∧
  coverage_area n r w = (480 : ℝ) * Real.pi / Real.tan (Real.pi / 8) :=
by
  sorry

end radar_placement_and_coverage_area_l99_99177


namespace lawnmower_percentage_drop_l99_99687

theorem lawnmower_percentage_drop :
  ∀ (initial_value value_after_one_year value_after_six_months : ℝ)
    (percentage_drop_in_year : ℝ),
  initial_value = 100 →
  value_after_one_year = 60 →
  percentage_drop_in_year = 20 →
  value_after_one_year = (1 - percentage_drop_in_year / 100) * value_after_six_months →
  (initial_value - value_after_six_months) / initial_value * 100 = 25 :=
by
  intros initial_value value_after_one_year value_after_six_months percentage_drop_in_year
  intros h_initial h_value_after_one_year h_percentage_drop_in_year h_value_equation
  sorry

end lawnmower_percentage_drop_l99_99687


namespace total_money_l99_99723

theorem total_money (a b c : ℕ) (h_ratio : (a / 2) / (b / 3) / (c / 4) = 1) (h_c : c = 306) : 
  a + b + c = 782 := 
by sorry

end total_money_l99_99723


namespace minimize_q_l99_99457

noncomputable def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

theorem minimize_q : ∃ x : ℝ, q x = 2 :=
by
  sorry

end minimize_q_l99_99457


namespace one_eq_one_of_ab_l99_99986

variable {a b : ℝ}

theorem one_eq_one_of_ab (h : a * b = a^2 - a * b + b^2) : 1 = 1 := by
  sorry

end one_eq_one_of_ab_l99_99986


namespace no_same_distribution_of_silver_as_gold_l99_99336

theorem no_same_distribution_of_silver_as_gold (n m : ℕ) 
  (hn : n ≡ 5 [MOD 10]) 
  (hm : m = 2 * n) 
  : ∀ (f : Fin 10 → ℕ), (∀ i j : Fin 10, i ≠ j → ¬ (f i - f j ≡ 0 [MOD 10])) 
  → ∀ (g : Fin 10 → ℕ), ¬ (∀ i j : Fin 10, i ≠ j → ¬ (g i - g j ≡ 0 [MOD 10])) :=
sorry

end no_same_distribution_of_silver_as_gold_l99_99336


namespace salary_decrease_increase_l99_99243

theorem salary_decrease_increase (S : ℝ) (x : ℝ) (h : (S * (1 - x / 100) * (1 + x / 100) = 0.51 * S)) : x = 70 := 
by sorry

end salary_decrease_increase_l99_99243


namespace trip_time_difference_l99_99196

theorem trip_time_difference
  (avg_speed : ℝ)
  (dist1 dist2 : ℝ)
  (h_avg_speed : avg_speed = 60)
  (h_dist1 : dist1 = 540)
  (h_dist2 : dist2 = 570) :
  ((dist2 - dist1) / avg_speed) * 60 = 30 := by
  sorry

end trip_time_difference_l99_99196


namespace evaluate_g_g_g_25_l99_99845

def g (x : ℤ) : ℤ :=
  if x < 10 then x^2 - 9 else x - 20

theorem evaluate_g_g_g_25 : g (g (g 25)) = -4 := by
  sorry

end evaluate_g_g_g_25_l99_99845


namespace necessary_but_not_sufficient_l99_99747

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x > 2 → |x| ≥ 1) ∧ (∃ x : ℝ, |x| ≥ 1 ∧ ¬ (x > 2)) :=
by
  sorry

end necessary_but_not_sufficient_l99_99747


namespace setA_times_setB_equals_desired_l99_99391

def setA : Set ℝ := { x | abs (x - 1/2) < 1 }
def setB : Set ℝ := { x | 1/x ≥ 1 }
def setAB : Set ℝ := { x | (x ∈ setA ∪ setB) ∧ (x ∉ setA ∩ setB) }

theorem setA_times_setB_equals_desired :
  setAB = { x | (-1/2 < x ∧ x ≤ 0) ∨ (1 < x ∧ x < 3/2) } :=
by
  sorry

end setA_times_setB_equals_desired_l99_99391


namespace sum_faces_edges_vertices_triangular_prism_l99_99471

-- Given conditions for triangular prism:
def triangular_prism_faces : Nat := 2 + 3  -- 2 triangular faces and 3 rectangular faces
def triangular_prism_edges : Nat := 3 + 3 + 3  -- 3 top edges, 3 bottom edges, 3 connecting edges
def triangular_prism_vertices : Nat := 3 + 3  -- 3 vertices on the top base, 3 on the bottom base

-- Proof statement for the sum of the faces, edges, and vertices of a triangular prism
theorem sum_faces_edges_vertices_triangular_prism : 
  triangular_prism_faces + triangular_prism_edges + triangular_prism_vertices = 20 := by
  sorry

end sum_faces_edges_vertices_triangular_prism_l99_99471


namespace geometric_sequence_sufficient_and_necessary_l99_99604

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_sufficient_and_necessary (a : ℕ → ℝ) (h1 : a 0 > 0) :
  (a 0 < a 1) ↔ (is_geometric_sequence a ∧ is_increasing_sequence a) :=
sorry

end geometric_sequence_sufficient_and_necessary_l99_99604


namespace bob_total_calories_l99_99538

def total_calories (slices_300 : ℕ) (calories_300 : ℕ) (slices_400 : ℕ) (calories_400 : ℕ) : ℕ :=
  slices_300 * calories_300 + slices_400 * calories_400

theorem bob_total_calories 
  (slices_300 : ℕ := 3)
  (calories_300 : ℕ := 300)
  (slices_400 : ℕ := 4)
  (calories_400 : ℕ := 400) :
  total_calories slices_300 calories_300 slices_400 calories_400 = 2500 := 
by 
  sorry

end bob_total_calories_l99_99538


namespace quadratic_inequality_solution_l99_99773

open Real

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 15 < 0) : 3 < x ∧ x < 5 :=
sorry

end quadratic_inequality_solution_l99_99773


namespace erica_duration_is_correct_l99_99824

-- Define the durations for Dave, Chuck, and Erica
def dave_duration : ℝ := 10
def chuck_duration : ℝ := 5 * dave_duration
def erica_duration : ℝ := chuck_duration + 0.30 * chuck_duration

-- State the theorem
theorem erica_duration_is_correct : erica_duration = 65 := by
  sorry

end erica_duration_is_correct_l99_99824


namespace raviraj_distance_home_l99_99502

theorem raviraj_distance_home :
  let origin := (0, 0)
  let after_south := (0, -20)
  let after_west := (-10, -20)
  let after_north := (-10, 0)
  let final_pos := (-30, 0)
  Real.sqrt ((final_pos.1 - origin.1)^2 + (final_pos.2 - origin.2)^2) = 30 :=
by
  sorry

end raviraj_distance_home_l99_99502


namespace Ted_age_48_l99_99031

/-- Given ages problem:
 - t is Ted's age
 - s is Sally's age
 - a is Alex's age 
 - The following conditions hold:
   1. t = 2s + 17 
   2. a = s / 2
   3. t + s + a = 72
 - Prove that Ted's age (t) is 48.
-/ 
theorem Ted_age_48 {t s a : ℕ} (h1 : t = 2 * s + 17) (h2 : a = s / 2) (h3 : t + s + a = 72) : t = 48 := by
  sorry

end Ted_age_48_l99_99031


namespace weight_conversion_l99_99559

theorem weight_conversion (a b : ℝ) (conversion_rate : ℝ) : a = 3600 → b = 600 → conversion_rate = 1000 → (a - b) / conversion_rate = 3 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end weight_conversion_l99_99559


namespace conveyor_belt_efficiencies_and_min_cost_l99_99759

theorem conveyor_belt_efficiencies_and_min_cost :
  ∃ (efficiency_B efficiency_A : ℝ),
    efficiency_A = 1.5 * efficiency_B ∧
    18000 / efficiency_B - 18000 / efficiency_A = 10 ∧
    efficiency_B = 600 ∧
    efficiency_A = 900 ∧
    ∃ (cost_A cost_B : ℝ),
      cost_A = 8 * 20 ∧
      cost_B = 6 * 30 ∧
      cost_A = 160 ∧
      cost_B = 180 ∧
      cost_A < cost_B :=
by
  sorry

end conveyor_belt_efficiencies_and_min_cost_l99_99759


namespace quadratic_condition_l99_99550

variables {c y1 y2 y3 : ℝ}

/-- Points P1(-1, y1), P2(3, y2), P3(5, y3) are all on the graph of the quadratic function y = -x^2 + 2x + c. --/
def points_on_parabola (y1 y2 y3 c : ℝ) : Prop :=
  y1 = -(-1)^2 + 2*(-1) + c ∧
  y2 = -(3)^2 + 2*(3) + c ∧
  y3 = -(5)^2 + 2*(5) + c

/-- The quadratic function y = -x^2 + 2x + c has an axis of symmetry at x = 1 and opens downwards. --/
theorem quadratic_condition (h : points_on_parabola y1 y2 y3 c) : 
  y1 = y2 ∧ y2 > y3 :=
sorry

end quadratic_condition_l99_99550


namespace find_m_l99_99317

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x + α / x + Real.log x

theorem find_m (α : ℝ) (m : ℝ) (l e : ℝ) (hα_range : α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 2))
(h1 : f 1 α < m) (he : f (Real.exp 1) α < m) :
m > 1 + 2 * Real.exp 2 := by
  sorry

end find_m_l99_99317


namespace Andy_collects_16_balls_l99_99013

-- Define the number of balls collected by Andy, Roger, and Maria.
variables (x : ℝ) (r : ℝ) (m : ℝ)

-- Define the conditions
def Andy_twice_as_many_as_Roger : Prop := r = x / 2
def Andy_five_more_than_Maria : Prop := m = x - 5
def Total_balls : Prop := x + r + m = 35

-- Define the main theorem to prove Andy's number of balls
theorem Andy_collects_16_balls (h1 : Andy_twice_as_many_as_Roger x r) 
                               (h2 : Andy_five_more_than_Maria x m) 
                               (h3 : Total_balls x r m) : 
                               x = 16 := 
by 
  sorry

end Andy_collects_16_balls_l99_99013


namespace meaningful_expression_range_l99_99931

theorem meaningful_expression_range (x : ℝ) : 
  (x - 1 ≥ 0) ∧ (x ≠ 3) ↔ (x ≥ 1 ∧ x ≠ 3) := 
by
  sorry

end meaningful_expression_range_l99_99931


namespace difference_in_money_in_cents_l99_99131

theorem difference_in_money_in_cents (p : ℤ) (h₁ : ℤ) (h₂ : ℤ) 
  (h₁ : Linda_nickels = 7 * p - 2) (h₂ : Carol_nickels = 3 * p + 4) :
  5 * (Linda_nickels - Carol_nickels) = 20 * p - 30 := 
by sorry

end difference_in_money_in_cents_l99_99131


namespace sum_every_third_odd_integer_l99_99600

theorem sum_every_third_odd_integer (a₁ d n : ℕ) (S : ℕ) 
  (h₁ : a₁ = 201) 
  (h₂ : d = 6) 
  (h₃ : n = 50) 
  (h₄ : S = (n * (2 * a₁ + (n - 1) * d)) / 2) 
  (h₅ : a₁ + (n - 1) * d = 495) 
  : S = 17400 := 
  by sorry

end sum_every_third_odd_integer_l99_99600


namespace rate_of_interest_per_annum_l99_99256

theorem rate_of_interest_per_annum (SI P : ℝ) (T : ℕ) (hSI : SI = 4016.25) (hP : P = 10040.625) (hT : T = 5) :
  (SI * 100) / (P * T) = 8 :=
by 
  -- Given simple interest formula
  -- SI = P * R * T / 100, solving for R we get R = (SI * 100) / (P * T)
  -- Substitute SI = 4016.25, P = 10040.625, and T = 5
  -- (4016.25 * 100) / (10040.625 * 5) = 8
  sorry

end rate_of_interest_per_annum_l99_99256


namespace simplify_and_evaluate_l99_99335

theorem simplify_and_evaluate (a : ℤ) (h : a = 0) : 
  ((a / (a - 1) : ℚ) + ((a + 1) / (a^2 - 1) : ℚ)) = (-1 : ℚ) := by
  have ha_ne1 : a ≠ 1 := by norm_num [h]
  have ha_ne_neg1 : a ≠ -1 := by norm_num [h]
  have h1 : (a^2 - 1) ≠ 0 := by
    rw [sub_ne_zero]
    norm_num [h]
  sorry

end simplify_and_evaluate_l99_99335


namespace infinite_primes_4k1_l99_99272

theorem infinite_primes_4k1 : ∀ (P : List ℕ), (∀ (p : ℕ), p ∈ P → Nat.Prime p ∧ ∃ k, p = 4 * k + 1) → 
  ∃ q, Nat.Prime q ∧ ∃ k, q = 4 * k + 1 ∧ q ∉ P :=
sorry

end infinite_primes_4k1_l99_99272


namespace max_students_per_class_l99_99569

theorem max_students_per_class (num_students : ℕ) (seats_per_bus : ℕ) (num_buses : ℕ) (k : ℕ) 
  (h_num_students : num_students = 920) 
  (h_seats_per_bus : seats_per_bus = 71) 
  (h_num_buses : num_buses = 16) 
  (h_class_size_bound : ∀ c, c ≤ k) : 
  k = 17 :=
sorry

end max_students_per_class_l99_99569


namespace min_value_b_minus_a_l99_99035

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

theorem min_value_b_minus_a :
  ∀ (a : ℝ), ∃ (b : ℝ), b > 0 ∧ f a = g b ∧ ∀ (y : ℝ), b - a = 2 * Real.exp (y - 1 / 2) - Real.log y → y = 1 / 2 → b - a = 2 + Real.log 2 := by
  sorry

end min_value_b_minus_a_l99_99035


namespace probability_different_colors_l99_99929

-- Define the total number of blue and yellow chips
def blue_chips : ℕ := 5
def yellow_chips : ℕ := 7
def total_chips : ℕ := blue_chips + yellow_chips

-- Define the probability of drawing a blue chip and a yellow chip
def prob_blue : ℚ := blue_chips / total_chips
def prob_yellow : ℚ := yellow_chips / total_chips

-- Define the probability of drawing two chips of different colors
def prob_different_colors := 2 * (prob_blue * prob_yellow)

theorem probability_different_colors :
  prob_different_colors = (35 / 72) := by
  sorry

end probability_different_colors_l99_99929


namespace ratio_of_c_to_d_l99_99694

theorem ratio_of_c_to_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
    (h1 : 9 * x - 6 * y = c) (h2 : 15 * x - 10 * y = d) :
    c / d = -2 / 5 :=
by
  sorry

end ratio_of_c_to_d_l99_99694


namespace arithmetic_sequence_terms_l99_99144

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h2 : a 1 + a 2 + a 3 = 34)
  (h3 : a n + a (n-1) + a (n-2) = 146)
  (h4 : S n = 390)
  (h5 : ∀ i j, a i + a j = a (i+1) + a (j-1)) :
  n = 13 :=
sorry

end arithmetic_sequence_terms_l99_99144


namespace bridget_apples_l99_99079

theorem bridget_apples (x : ℕ) (h1 : x - 2 ≥ 0) (h2 : (x - 2) / 3 = 0 → false)
    (h3 : (2 * (x - 2) / 3) - 5 = 6) : x = 20 :=
by
  sorry

end bridget_apples_l99_99079


namespace gcd_cubed_and_sum_l99_99901

theorem gcd_cubed_and_sum (n : ℕ) (h_pos : 0 < n) (h_gt_square : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := 
sorry

end gcd_cubed_and_sum_l99_99901


namespace necessary_but_not_sufficient_l99_99729

theorem necessary_but_not_sufficient (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≠ b) : ab > 0 :=
  sorry

end necessary_but_not_sufficient_l99_99729


namespace area_of_triangle_l99_99334

theorem area_of_triangle (base : ℝ) (height : ℝ) (h_base : base = 3.6) (h_height : height = 2.5 * base) : 
  (base * height) / 2 = 16.2 :=
by {
  sorry
}

end area_of_triangle_l99_99334


namespace find_s_when_t_eq_5_l99_99904

theorem find_s_when_t_eq_5 (s : ℝ) (h : 5 = 8 * s^2 + 2 * s) :
  s = (-1 + Real.sqrt 41) / 8 ∨ s = (-1 - Real.sqrt 41) / 8 :=
by sorry

end find_s_when_t_eq_5_l99_99904


namespace ratio_abc_xyz_l99_99791

theorem ratio_abc_xyz
  (a b c x y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7 / 8 := 
sorry

end ratio_abc_xyz_l99_99791


namespace frog_jump_paths_l99_99114

noncomputable def φ : ℕ × ℕ → ℕ
| (0, 0) => 1
| (x, y) =>
  let φ_x1 := if x > 1 then φ (x - 1, y) else 0
  let φ_x2 := if x > 1 then φ (x - 2, y) else 0
  let φ_y1 := if y > 1 then φ (x, y - 1) else 0
  let φ_y2 := if y > 1 then φ (x, y - 2) else 0
  φ_x1 + φ_x2 + φ_y1 + φ_y2

theorem frog_jump_paths : φ (4, 4) = 556 := sorry

end frog_jump_paths_l99_99114


namespace solve_equation_l99_99472

-- Define the conditions of the problem.
def equation (x : ℝ) : Prop := (5 - x / 3)^(1/3) = -2

-- Define the main theorem to prove that x = 39 is the solution to the equation.
theorem solve_equation : ∃ x : ℝ, equation x ∧ x = 39 :=
by
  existsi 39
  intros
  simp [equation]
  sorry

end solve_equation_l99_99472


namespace distance_covered_at_40_kmph_l99_99480

theorem distance_covered_at_40_kmph (x : ℝ) : 
  (x / 40 + (250 - x) / 60 = 5.4) → (x = 148) :=
by
  intro h
  sorry

end distance_covered_at_40_kmph_l99_99480


namespace reams_for_haley_correct_l99_99085

-- Definitions: 
-- total reams = 5
-- reams for sister = 3
-- reams for Haley = ?

def total_reams : Nat := 5
def reams_for_sister : Nat := 3
def reams_for_haley : Nat := total_reams - reams_for_sister

-- The proof problem: prove reams_for_haley = 2 given the conditions.
theorem reams_for_haley_correct : reams_for_haley = 2 := by 
  sorry

end reams_for_haley_correct_l99_99085


namespace gcd_48_180_l99_99743

theorem gcd_48_180 : Nat.gcd 48 180 = 12 := by
  have f1 : 48 = 2^4 * 3 := by norm_num
  have f2 : 180 = 2^2 * 3^2 * 5 := by norm_num
  sorry

end gcd_48_180_l99_99743


namespace at_least_one_fraction_lt_two_l99_99341

theorem at_least_one_fraction_lt_two 
  (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : 2 < x + y) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_fraction_lt_two_l99_99341


namespace min_value_a2_plus_b2_l99_99983

theorem min_value_a2_plus_b2 (a b : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 2 * b = 0 -> x = -2) : (∃ a b, a = 1 ∧ b = -1 ∧ ∀ a' b', a^2 + b^2 ≥ a'^2 + b'^2) := 
by {
  sorry
}

end min_value_a2_plus_b2_l99_99983


namespace poly_square_of_binomial_l99_99754

theorem poly_square_of_binomial (x y : ℝ) : (x + y) * (x - y) = x^2 - y^2 := 
by 
  sorry

end poly_square_of_binomial_l99_99754


namespace circle_equation_through_points_l99_99231

theorem circle_equation_through_points 
  (M N : ℝ × ℝ)
  (hM : M = (5, 2))
  (hN : N = (3, 2))
  (hk : ∃ k : ℝ, (M.1 + N.1) / 2 = k ∧ (M.2 + N.2) / 2 = (2 * k - 3))
  : (∃ h : ℝ, ∀ x y: ℝ, (x - 4) ^ 2 + (y - 5) ^ 2 = h) ∧ (∃ r : ℝ, r = 10) := 
sorry

end circle_equation_through_points_l99_99231


namespace range_m_l99_99339

theorem range_m (m : ℝ) :
  (∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ abs (x - m) < 1) →
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  intro h
  sorry

end range_m_l99_99339


namespace rectangles_in_5x5_grid_l99_99881

theorem rectangles_in_5x5_grid : 
  ∃ n : ℕ, n = 100 ∧ (∀ (grid : Fin 6 → Fin 6 → Prop), 
  (∃ (vlines hlines : Finset (Fin 6)),
   (vlines.card = 2 ∧ hlines.card = 2) ∧
   n = (vlines.card.choose 2) * (hlines.card.choose 2))) :=
by
  sorry

end rectangles_in_5x5_grid_l99_99881


namespace good_arrangement_iff_coprime_l99_99195

-- Definitions for the concepts used
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_good_arrangement (n m : ℕ) : Prop :=
  ∃ k₀, ∀ i, (n * k₀ * i) % (m + n) = (i % (m + n))

theorem good_arrangement_iff_coprime (n m : ℕ) : is_good_arrangement n m ↔ is_coprime n m := 
sorry

end good_arrangement_iff_coprime_l99_99195


namespace identity_completion_factorize_polynomial_equilateral_triangle_l99_99043

-- Statement 1: Prove that a^3 - b^3 + a^2 b - ab^2 = (a - b)(a + b)^2 
theorem identity_completion (a b : ℝ) : a^3 - b^3 + a^2 * b - a * b^2 = (a - b) * (a + b)^2 :=
sorry

-- Statement 2: Prove that 4x^2 - 2x - y^2 - y = (2x + y)(2x - y - 1)
theorem factorize_polynomial (x y : ℝ) : 4 * x^2 - 2 * x - y^2 - y = (2 * x + y) * (2 * x - y - 1) :=
sorry

-- Statement 3: Given a^2 + b^2 + 2c^2 - 2ac - 2bc = 0, Prove that triangle ABC is equilateral
theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + 2 * c^2 - 2 * a * c - 2 * b * c = 0) : a = b ∧ b = c :=
sorry

end identity_completion_factorize_polynomial_equilateral_triangle_l99_99043


namespace gabi_final_prices_l99_99848

theorem gabi_final_prices (x y : ℝ) (hx : 0.8 * x = 1.2 * y) (hl : (x - 0.8 * x) + (y - 1.2 * y) = 10) :
  x = 30 ∧ y = 20 := sorry

end gabi_final_prices_l99_99848


namespace k_value_range_l99_99351

-- Definitions
def f (x : ℝ) (k : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- The theorem we are interested in
theorem k_value_range (k : ℝ) (h : ∀ x₁ x₂ : ℝ, (x₁ > 5 → x₂ > 5 → f x₁ k ≤ f x₂ k) ∨ (x₁ > 5 → x₂ > 5 → f x₁ k ≥ f x₂ k)) :
  k ≥ 40 :=
sorry

end k_value_range_l99_99351


namespace two_pipes_fill_time_l99_99613

theorem two_pipes_fill_time (R : ℝ) (h : 3 * R = 1 / 8) : 2 * R = 1 / 12 := 
by sorry

end two_pipes_fill_time_l99_99613


namespace solution_set_of_inequality_l99_99465

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 2*x + 3 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l99_99465


namespace solution_set_inequality_l99_99778

theorem solution_set_inequality (x : ℝ) :
  (|x + 3| - |x - 3| > 3) ↔ (x > 3 / 2) := 
sorry

end solution_set_inequality_l99_99778


namespace max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l99_99501

noncomputable def max_xy (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then x * y else 0

noncomputable def min_y_over_x_plus_4_over_y (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then y / x + 4 / y else 0

theorem max_xy_is_2 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → max_xy x y = 2 :=
by
  intros x y hx hy hxy
  sorry

theorem min_y_over_x_plus_4_over_y_is_4 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → min_y_over_x_plus_4_over_y x y = 4 :=
by
  intros x y hx hy hxy
  sorry

end max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l99_99501


namespace cleaning_time_is_100_l99_99094

def time_hosing : ℕ := 10
def time_shampoo_per : ℕ := 15
def num_shampoos : ℕ := 3
def time_drying : ℕ := 20
def time_brushing : ℕ := 25

def total_time : ℕ :=
  time_hosing + (num_shampoos * time_shampoo_per) + time_drying + time_brushing

theorem cleaning_time_is_100 :
  total_time = 100 :=
by
  sorry

end cleaning_time_is_100_l99_99094


namespace maria_cookies_l99_99786

theorem maria_cookies :
  let c_initial := 19
  let c1 := c_initial - 5
  let c2 := c1 / 2
  let c_final := c2 - 2
  c_final = 5 :=
by
  sorry

end maria_cookies_l99_99786


namespace magic_shop_change_l99_99322

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l99_99322


namespace other_number_more_than_42_l99_99554

theorem other_number_more_than_42 (a b : ℕ) (h1 : a + b = 96) (h2 : a = 42) : b - a = 12 := by
  sorry

end other_number_more_than_42_l99_99554


namespace kenneth_distance_past_finish_l99_99864

noncomputable def distance_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) : ℕ :=
  let biff_time := race_distance / biff_speed
  let kenneth_distance := kenneth_speed * biff_time
  kenneth_distance - race_distance

theorem kenneth_distance_past_finish (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (finish_line_distance : ℕ) : 
  race_distance = 500 ->
  biff_speed = 50 -> 
  kenneth_speed = 51 ->
  finish_line_distance = 10 ->
  distance_past_finish_line race_distance biff_speed kenneth_speed = finish_line_distance := by
  sorry

end kenneth_distance_past_finish_l99_99864


namespace find_b_l99_99434

-- Definitions based on the given conditions
def good_point (a b : ℝ) (φ : ℝ) : Prop :=
  a + (b - a) * φ = 2.382 ∨ b - (b - a) * φ = 2.382

theorem find_b (b : ℝ) (φ : ℝ := 0.618) :
  good_point 2 b φ → b = 2.618 ∨ b = 3 :=
by
  sorry

end find_b_l99_99434


namespace intersection_of_M_and_N_l99_99858

open Set

noncomputable def M := {x : ℝ | ∃ y:ℝ, y = Real.log (2 - x)}
noncomputable def N := {x : ℝ | x^2 - 3*x - 4 ≤ 0 }
noncomputable def I := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = I := 
  sorry

end intersection_of_M_and_N_l99_99858


namespace problem_conditions_l99_99740

def G (m : ℕ) : ℕ := m % 10

theorem problem_conditions (a b c : ℕ) (non_neg_m : ∀ m : ℕ, 0 ≤ m) :
  -- Condition ①
  ¬ (G (a - b) = G a - G b) ∧
  -- Condition ②
  (a - b = 10 * c → G a = G b) ∧
  -- Condition ③
  (G (a * b * c) = G (G a * G b * G c)) ∧
  -- Condition ④
  ¬ (G (3^2015) = 9) :=
by sorry

end problem_conditions_l99_99740


namespace ab_operation_l99_99350

theorem ab_operation (a b : ℤ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h1 : a + b = 10) (h2 : a * b = 24) : 
  (1 / a + 1 / b) = 5 / 12 :=
by
  sorry

end ab_operation_l99_99350


namespace ratio_diamond_brace_ring_l99_99086

theorem ratio_diamond_brace_ring
  (cost_ring : ℤ) (cost_car : ℤ) (total_worth : ℤ) (cost_diamond_brace : ℤ)
  (h1 : cost_ring = 4000) (h2 : cost_car = 2000) (h3 : total_worth = 14000)
  (h4 : cost_diamond_brace = total_worth - (cost_ring + cost_car)) :
  cost_diamond_brace / cost_ring = 2 :=
by
  sorry

end ratio_diamond_brace_ring_l99_99086


namespace student_2005_says_1_l99_99777

def pattern : List ℕ := [1, 2, 3, 4, 3, 2]

def nth_number_in_pattern (n : ℕ) : ℕ :=
  List.nthLe pattern (n % 6) sorry  -- The index is (n-1) % 6 because Lean indices start at 0

theorem student_2005_says_1 : nth_number_in_pattern 2005 = 1 := 
  by
  -- The proof goes here
  sorry

end student_2005_says_1_l99_99777


namespace find_m_value_l99_99343

def symmetric_inverse (g : ℝ → ℝ) (h : ℝ → ℝ) :=
  ∀ x, g (h x) = x ∧ h (g x) = x

def symmetric_y_axis (f : ℝ → ℝ) (g : ℝ → ℝ) :=
  ∀ x, f x = g (-x)

theorem find_m_value :
  (∀ g, symmetric_inverse g (Real.exp) → (∀ f, symmetric_y_axis f g → (∀ m, f m = -1 → m = - (1 / Real.exp 1)))) := by
  sorry

end find_m_value_l99_99343


namespace polygon_sides_l99_99054

theorem polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by {
  sorry
}

end polygon_sides_l99_99054


namespace identical_prob_of_painted_cubes_l99_99692

/-
  Given:
  - Each face of a cube can be painted in one of 3 colors.
  - Each cube has 6 faces.
  - The total possible ways to paint both cubes is 531441.
  - The total ways to paint them such that they are identical after rotation is 66.

  Prove:
  - The probability of two painted cubes being identical after rotation is 2/16101.
-/
theorem identical_prob_of_painted_cubes :
  let total_ways := 531441
  let identical_ways := 66
  (identical_ways : ℚ) / total_ways = 2 / 16101 := by
  sorry

end identical_prob_of_painted_cubes_l99_99692


namespace inequality_problem_l99_99642

theorem inequality_problem (x : ℝ) (hx : 0 < x) : 
  1 + x ^ 2018 ≥ (2 * x) ^ 2017 / (1 + x) ^ 2016 := 
by
  sorry

end inequality_problem_l99_99642


namespace joan_books_l99_99166

theorem joan_books : 
  (33 - 26 = 7) :=
by
  sorry

end joan_books_l99_99166


namespace find_angle_l99_99680

variable (x : ℝ)

theorem find_angle (h1 : x + (180 - x) = 180) (h2 : x + (90 - x) = 90) (h3 : 180 - x = 3 * (90 - x)) : x = 45 := 
by
  sorry

end find_angle_l99_99680


namespace half_angle_in_quadrant_l99_99107

theorem half_angle_in_quadrant (α : ℝ) (k : ℤ) (h : k * 360 + 90 < α ∧ α < k * 360 + 180) :
  ∃ n : ℤ, (n * 360 + 45 < α / 2 ∧ α / 2 < n * 360 + 90) ∨ (n * 360 + 225 < α / 2 ∧ α / 2 < n * 360 + 270) :=
by sorry

end half_angle_in_quadrant_l99_99107


namespace price_of_basic_computer_l99_99270

-- Definitions for the prices
variables (C_b P M K C_e : ℝ)

-- Conditions
axiom h1 : C_b + P + M + K = 2500
axiom h2 : C_e + P + M + K = 3100
axiom h3 : P = (3100 / 6)
axiom h4 : M = (3100 / 5)
axiom h5 : K = (3100 / 8)
axiom h6 : C_e = C_b + 600

-- Theorem stating the price of the basic computer
theorem price_of_basic_computer : C_b = 975.83 :=
by {
  sorry
}

end price_of_basic_computer_l99_99270


namespace identity_1_identity_2_identity_3_l99_99311

-- Variables and assumptions
variables (a b c : ℝ)
variables (h_different : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0)

-- Part 1
theorem identity_1 : 
  (1 / ((a - b) * (a - c))) + (1 / ((b - c) * (b - a))) + (1 / ((c - a) * (c - b))) = 0 := 
by sorry

-- Part 2
theorem identity_2 :
  (a / ((a - b) * (a - c))) + (b / ((b - c) * (b - a))) + (c / ((c - a) * (c - b))) = 0 :=
by sorry

-- Part 3
theorem identity_3 :
  (a^2 / ((a - b) * (a - c))) + (b^2 / ((b - c) * (b - a))) + (c^2 / ((c - a) * (c - b))) = 1 :=
by sorry

end identity_1_identity_2_identity_3_l99_99311


namespace max_value_of_a_l99_99059

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) → a ≤ 5 :=
by sorry

end max_value_of_a_l99_99059


namespace trigonometric_identity_l99_99477

variable {α : Real}
variable (h : Real.cos α = -2 / 3)

theorem trigonometric_identity : 
  (Real.cos α = -2 / 3) → 
  (Real.cos (4 * Real.pi - α) * Real.sin (-α) / 
  (Real.sin (Real.pi / 2 + α) * Real.tan (Real.pi - α)) = Real.cos α) :=
by
  intro h
  sorry

end trigonometric_identity_l99_99477


namespace circle_properties_l99_99046

theorem circle_properties :
  ∃ (c d s : ℝ), (∀ x y : ℝ, x^2 - 4 * y - 25 = -y^2 + 10 * x + 49 → (x - 5)^2 + (y - 2)^2 = s^2) ∧
  c = 5 ∧ d = 2 ∧ s = Real.sqrt 103 ∧ c + d + s = 7 + Real.sqrt 103 :=
by
  sorry

end circle_properties_l99_99046


namespace larger_cross_section_distance_l99_99681

theorem larger_cross_section_distance
  (h_area1 : ℝ)
  (h_area2 : ℝ)
  (dist_planes : ℝ)
  (h_area1_val : h_area1 = 256 * Real.sqrt 2)
  (h_area2_val : h_area2 = 576 * Real.sqrt 2)
  (dist_planes_val : dist_planes = 10) :
  ∃ h : ℝ, h = 30 :=
by
  sorry

end larger_cross_section_distance_l99_99681


namespace johnson_potatoes_l99_99008

/-- Given that Johnson has a sack of 300 potatoes, 
    gives some to Gina, twice that amount to Tom, and 
    one-third of the amount given to Tom to Anne,
    and has 47 potatoes left, we prove that 
    Johnson gave Gina 69 potatoes. -/
theorem johnson_potatoes : 
  ∃ G : ℕ, 
  ∀ (Gina Tom Anne total : ℕ), 
    total = 300 ∧ 
    total - (Gina + Tom + Anne) = 47 ∧ 
    Tom = 2 * Gina ∧ 
    Anne = (1 / 3 : ℚ) * Tom ∧ 
    (Gina + Tom + (Anne : ℕ)) = (11 / 3 : ℚ) * Gina ∧ 
    (Gina + Tom + Anne) = 253 
    ∧ total = Gina + Tom + Anne + 47 
    → Gina = 69 := sorry


end johnson_potatoes_l99_99008


namespace number_of_8_tuples_l99_99098

-- Define the constraints for a_k
def valid_a (a : ℕ) (k : ℕ) : Prop := 0 ≤ a ∧ a ≤ k

-- Define the condition for the 8-tuple
def valid_8_tuple (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ) : Prop :=
  valid_a a1 1 ∧ valid_a a2 2 ∧ valid_a a3 3 ∧ valid_a a4 4 ∧ 
  (a1 + a2 + a3 + a4 + 2 * b1 + 3 * b2 + 4 * b3 + 5 * b4 = 19)

theorem number_of_8_tuples : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ), valid_8_tuple a1 a2 a3 a4 b1 b2 b3 b4 := 
sorry

end number_of_8_tuples_l99_99098


namespace difference_in_balances_l99_99313

/-- Define the parameters for Angela's and Bob's accounts --/
def P_A : ℕ := 5000  -- Angela's principal
def r_A : ℚ := 0.05  -- Angela's annual interest rate
def n_A : ℕ := 2  -- Compounding frequency for Angela
def t : ℕ := 15  -- Time in years

def P_B : ℕ := 7000  -- Bob's principal
def r_B : ℚ := 0.04  -- Bob's annual interest rate

/-- Computing the final amounts for Angela and Bob after 15 years --/
noncomputable def A_A : ℚ := P_A * ((1 + (r_A / n_A)) ^ (n_A * t))  -- Angela's final amount
noncomputable def A_B : ℚ := P_B * (1 + r_B * t)  -- Bob's final amount

/-- Proof statement: The difference in account balances to the nearest dollar --/
theorem difference_in_balances : abs (A_A - A_B) = 726 := by
  sorry

end difference_in_balances_l99_99313


namespace ethanol_percentage_in_fuel_A_l99_99354

variable {capacity_A fuel_A : ℝ}
variable (ethanol_A ethanol_B total_ethanol : ℝ)
variable (E : ℝ)

def fuelTank (capacity_A fuel_A ethanol_A ethanol_B total_ethanol : ℝ) (E : ℝ) : Prop := 
  (ethanol_A / fuel_A = E) ∧
  (capacity_A - fuel_A = 200 - 99.99999999999999) ∧
  (ethanol_B = 0.16 * (200 - 99.99999999999999)) ∧
  (total_ethanol = ethanol_A + ethanol_B) ∧
  (total_ethanol = 28)

theorem ethanol_percentage_in_fuel_A : 
  ∃ E, fuelTank 99.99999999999999 99.99999999999999 ethanol_A ethanol_B 28 E ∧ E = 0.12 := 
sorry

end ethanol_percentage_in_fuel_A_l99_99354


namespace quadratic_rewrite_as_square_of_binomial_plus_integer_l99_99100

theorem quadratic_rewrite_as_square_of_binomial_plus_integer :
    ∃ a b, ∀ x, x^2 + 16 * x + 72 = (x + a)^2 + b ∧ b = 8 :=
by
  sorry

end quadratic_rewrite_as_square_of_binomial_plus_integer_l99_99100


namespace product_pqr_l99_99849

/-- Mathematical problem statement -/
theorem product_pqr (p q r : ℤ) (hp: p ≠ 0) (hq: q ≠ 0) (hr: r ≠ 0)
  (h1 : p + q + r = 36)
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 540 / (p * q * r) = 1) :
  p * q * r = 864 :=
sorry

end product_pqr_l99_99849


namespace allocation_first_grade_places_l99_99561

theorem allocation_first_grade_places (total_students : ℕ)
                                      (ratio_1 : ℕ)
                                      (ratio_2 : ℕ)
                                      (ratio_3 : ℕ)
                                      (total_places : ℕ) :
  total_students = 160 →
  ratio_1 = 6 →
  ratio_2 = 5 →
  ratio_3 = 5 →
  total_places = 160 →
  (total_places * ratio_1) / (ratio_1 + ratio_2 + ratio_3) = 60 :=
sorry

end allocation_first_grade_places_l99_99561


namespace david_still_has_l99_99384

variable (P L S R : ℝ)

def initial_amount : ℝ := 1800
def post_spending_condition (S : ℝ) : ℝ := S - 800
def remaining_money (P S : ℝ) : ℝ := P - S

theorem david_still_has :
  ∀ (S : ℝ),
    initial_amount = P →
    post_spending_condition S = L →
    remaining_money P S = R →
    R = L →
    R = 500 :=
by
  intros S hP hL hR hCl
  sorry

end david_still_has_l99_99384


namespace find_x_l99_99040

theorem find_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - y^2 - 4.5 = 0) → x = 1.5 := 
by 
  sorry

end find_x_l99_99040


namespace isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l99_99057

-- Part 1 proof
theorem isosceles_triangle_sides_part1 (x : ℝ) (h1 : x + 2 * x + 2 * x = 20) : 
  x = 4 ∧ 2 * x = 8 :=
by
  sorry

-- Part 2 proof
theorem isosceles_triangle_sides_part2 (a b : ℝ) (h2 : a = 5) (h3 : 2 * b + a = 20) :
  b = 7.5 :=
by
  sorry

end isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l99_99057


namespace john_candies_l99_99752

theorem john_candies (mark_candies : ℕ) (peter_candies : ℕ) (total_candies : ℕ) (equal_share : ℕ) (h1 : mark_candies = 30) (h2 : peter_candies = 25) (h3 : total_candies = 90) (h4 : equal_share * 3 = total_candies) : 
  (total_candies - mark_candies - peter_candies = 35) :=
by
  sorry

end john_candies_l99_99752


namespace term_61_is_201_l99_99842

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (a5 : ℤ)

-- Define the general formula for the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ :=
  a5 + (n - 5) * d

-- Given variables and conditions:
axiom h1 : a5 = 33
axiom h2 : d = 3

theorem term_61_is_201 :
  arithmetic_sequence a5 d 61 = 201 :=
by
  -- proof here
  sorry

end term_61_is_201_l99_99842


namespace task_completion_days_l99_99644

theorem task_completion_days (a b c: ℕ) :
  (b = a + 6) → (c = b + 3) → 
  (3 / a + 4 / b = 9 / c) →
  a = 18 ∧ b = 24 ∧ c = 27 :=
  by
  sorry

end task_completion_days_l99_99644


namespace number_of_chairs_borrowed_l99_99994

-- Define the conditions
def red_chairs := 4
def yellow_chairs := 2 * red_chairs
def blue_chairs := yellow_chairs - 2
def total_initial_chairs : Nat := red_chairs + yellow_chairs + blue_chairs
def chairs_left_in_the_afternoon := 15

-- Define the question
def chairs_borrowed_by_Lisa : Nat := total_initial_chairs - chairs_left_in_the_afternoon

-- The theorem to state the proof problem
theorem number_of_chairs_borrowed : chairs_borrowed_by_Lisa = 3 := by
  -- Proof to be added
  sorry

end number_of_chairs_borrowed_l99_99994


namespace sticker_price_l99_99406

theorem sticker_price (x : ℝ) (h1 : 0.8 * x - 100 = 0.7 * x - 25) : x = 750 :=
by
  sorry

end sticker_price_l99_99406


namespace find_correct_speed_l99_99072

-- Definitions for given conditions
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Given conditions as definitions
def condition1 (d t : ℝ) : Prop := distance_traveled 35 (t + (5 / 60)) = d
def condition2 (d t : ℝ) : Prop := distance_traveled 55 (t - (5 / 60)) = d

-- Statement to prove
theorem find_correct_speed (d t r : ℝ) (h1 : condition1 d t) (h2 : condition2 d t) :
  r = (d / t) ∧ r = 42.78 :=
by sorry

end find_correct_speed_l99_99072


namespace combined_salaries_l99_99896

-- Define the variables and constants corresponding to the conditions
variable (A B D E C : ℝ)
variable (avg_salary : ℝ)
variable (num_individuals : ℕ)

-- Given conditions translated into Lean definitions 
def salary_C : ℝ := 15000
def average_salary : ℝ := 8800
def number_of_individuals : ℕ := 5

-- Define the statement to prove
theorem combined_salaries (h1 : C = salary_C) (h2 : avg_salary = average_salary) (h3 : num_individuals = number_of_individuals) : 
  A + B + D + E = avg_salary * num_individuals - salary_C := 
by 
  -- Here the proof would involve calculating the total salary and subtracting C's salary
  sorry

end combined_salaries_l99_99896


namespace pascal_triangle_fifth_number_twentieth_row_l99_99712

theorem pascal_triangle_fifth_number_twentieth_row : 
  (Nat.choose 20 4) = 4845 :=
by
  sorry

end pascal_triangle_fifth_number_twentieth_row_l99_99712


namespace find_distinct_prime_triples_l99_99368

noncomputable def areDistinctPrimes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r

def satisfiesConditions (p q r : ℕ) : Prop :=
  p ∣ (q + r) ∧ q ∣ (r + 2 * p) ∧ r ∣ (p + 3 * q)

theorem find_distinct_prime_triples :
  { (p, q, r) : ℕ × ℕ × ℕ | areDistinctPrimes p q r ∧ satisfiesConditions p q r } =
  { (5, 3, 2), (2, 11, 7), (2, 3, 11) } :=
by
  sorry

end find_distinct_prime_triples_l99_99368


namespace age_of_eldest_child_l99_99053

theorem age_of_eldest_child (age_sum : ∀ (x : ℕ), x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 40) :
  ∃ x, x + 8 = 12 :=
by {
  sorry
}

end age_of_eldest_child_l99_99053


namespace square_side_length_l99_99151

theorem square_side_length 
  (A B C D E : Type) 
  (AB AC hypotenuse square_side_length : ℝ) 
  (h1: AB = 9) 
  (h2: AC = 12) 
  (h3: hypotenuse = Real.sqrt (9^2 + 12^2)) 
  (h4: square_side_length = 300 / 41) 
  : square_side_length = 300 / 41 := 
by 
  sorry

end square_side_length_l99_99151


namespace decimal_to_binary_49_l99_99771

theorem decimal_to_binary_49 : ((49:ℕ) = 6 * 2^4 + 3 * 2^3 + 0 * 2^2 + 0 * 2^1 + 0 * 2^0 + 1) ↔ (110001 = 110001) :=
by
  sorry

end decimal_to_binary_49_l99_99771


namespace number_of_good_students_is_5_or_7_l99_99213

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l99_99213


namespace find_x_values_l99_99364

theorem find_x_values (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 ∧ x2 = 3 / 5 ∧ x3 = 2 / 5 ∧ x4 = 1 / 5 :=
by
  sorry

end find_x_values_l99_99364


namespace number_of_zeros_of_f_l99_99386

noncomputable def f : ℝ → ℝ
| x => if x >= 0 then x^3 - 3*x + 1 else x^2 - 2*x - 4

theorem number_of_zeros_of_f : ∃ z, z = 3 := by
  sorry

end number_of_zeros_of_f_l99_99386


namespace find_first_number_in_sequence_l99_99066

theorem find_first_number_in_sequence :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℚ),
    (a3 = a2 * a1) ∧ 
    (a4 = a3 * a2) ∧ 
    (a5 = a4 * a3) ∧ 
    (a6 = a5 * a4) ∧ 
    (a7 = a6 * a5) ∧ 
    (a8 = a7 * a6) ∧ 
    (a9 = a8 * a7) ∧ 
    (a10 = a9 * a8) ∧ 
    (a8 = 36) ∧ 
    (a9 = 324) ∧ 
    (a10 = 11664) ∧ 
    (a1 = 59049 / 65536) := 
sorry

end find_first_number_in_sequence_l99_99066


namespace maggie_total_income_l99_99307

def total_income (h_tractor : ℕ) (r_office r_tractor : ℕ) :=
  let h_office := 2 * h_tractor
  (h_tractor * r_tractor) + (h_office * r_office)

theorem maggie_total_income :
  total_income 13 10 12 = 416 := 
  sorry

end maggie_total_income_l99_99307


namespace factory_produces_6400_toys_per_week_l99_99981

-- Definition of worker productivity per day
def toys_per_day : ℝ := 2133.3333333333335

-- Definition of workdays per week
def workdays_per_week : ℕ := 3

-- Definition of total toys produced per week
def toys_per_week : ℝ := toys_per_day * workdays_per_week

-- Theorem stating the total number of toys produced per week
theorem factory_produces_6400_toys_per_week : toys_per_week = 6400 :=
by
  sorry

end factory_produces_6400_toys_per_week_l99_99981


namespace correct_idiom_l99_99064

-- Define the conditions given in the problem
def context := "The vast majority of office clerks read a significant amount of materials"
def idiom_usage := "to say _ of additional materials"

-- Define the proof problem
theorem correct_idiom (context: String) (idiom_usage: String) : idiom_usage.replace "_ of additional materials" "nothing of newspapers and magazines" = "to say nothing of newspapers and magazines" :=
sorry

end correct_idiom_l99_99064


namespace coral_third_week_pages_l99_99266

theorem coral_third_week_pages :
  let total_pages := 600
  let week1_read := total_pages / 2
  let remaining_after_week1 := total_pages - week1_read
  let week2_read := remaining_after_week1 * 0.30
  let remaining_after_week2 := remaining_after_week1 - week2_read
  remaining_after_week2 = 210 :=
by
  sorry

end coral_third_week_pages_l99_99266


namespace sqrt_sum_simplify_l99_99938

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l99_99938


namespace sum_polynomials_l99_99189

def p (x : ℝ) : ℝ := 4 * x^2 - 2 * x + 1
def q (x : ℝ) : ℝ := -3 * x^2 + x - 5
def r (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem sum_polynomials (x : ℝ) : p x + q x + r x = 3 * x^2 - 5 * x - 1 :=
by
  sorry

end sum_polynomials_l99_99189


namespace intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l99_99389

variable (a b : ℝ)

-- Define the equations given in the problem
def line1 := ∀ y : ℝ, 3 = (1/3) * y + a
def line2 := ∀ x : ℝ, 3 = (1/3) * x + b

-- The Lean statement for the proof
theorem intersecting_lines_at_3_3_implies_a_plus_b_eq_4 :
  (line1 3) ∧ (line2 3) → a + b = 4 :=
by 
  sorry

end intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l99_99389


namespace p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l99_99377

-- Define conditions
def p (x : ℝ) : Prop := -x^2 + 2 * x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def s (x : ℝ) : Prop := -x^2 + 8 * x + 20 ≥ 0

variable {x m : ℝ}

-- Question 1
theorem p_sufficient_not_necessary_for_q (hp : ∀ x, p x → q x m) : m ≥ 3 :=
sorry

-- Defining negation of s and q
def neg_s (x : ℝ) : Prop := ¬s x
def neg_q (x m : ℝ) : Prop := ¬q x m

-- Question 2
theorem neg_s_sufficient_not_necessary_for_neg_q (hp : ∀ x, neg_s x → neg_q x m) : false :=
sorry

end p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l99_99377


namespace quadratic_solution_interval_l99_99436

noncomputable def quadratic_inequality (z : ℝ) : Prop :=
  z^2 - 56*z + 360 ≤ 0

theorem quadratic_solution_interval :
  {z : ℝ // quadratic_inequality z} = {z : ℝ // 8 ≤ z ∧ z ≤ 45} :=
by
  sorry

end quadratic_solution_interval_l99_99436


namespace coby_travel_time_l99_99349

theorem coby_travel_time :
  let d1 := 640
  let d2 := 400
  let d3 := 250
  let d4 := 380
  let s1 := 80
  let s2 := 65
  let s3 := 75
  let s4 := 50
  let time1 := d1 / s1
  let time2 := d2 / s2
  let time3 := d3 / s3
  let time4 := d4 / s4
  let total_time := time1 + time2 + time3 + time4
  total_time = 25.08 :=
by
  sorry

end coby_travel_time_l99_99349


namespace larger_number_is_23_l99_99534

theorem larger_number_is_23 (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 6) : a = 23 := 
by
  sorry

end larger_number_is_23_l99_99534


namespace find_fourth_term_geometric_progression_l99_99968

theorem find_fourth_term_geometric_progression (x : ℝ) (a1 a2 a3 : ℝ) (r : ℝ)
  (h1 : a1 = x)
  (h2 : a2 = 3 * x + 6)
  (h3 : a3 = 7 * x + 21)
  (h4 : ∃ r, a2 / a1 = r ∧ a3 / a2 = r)
  (hx : x = 3 / 2) :
  7 * (7 * x + 21) = 220.5 :=
by
  sorry

end find_fourth_term_geometric_progression_l99_99968


namespace compare_exponents_l99_99345

noncomputable def exp_of_log (a : ℝ) (b : ℝ) : ℝ :=
  Real.exp ((1 / b) * Real.log a)

theorem compare_exponents :
  let a := exp_of_log 4 4
  let b := exp_of_log 5 5
  let c := exp_of_log 16 16
  let d := exp_of_log 25 25
  a = max a (max b (max c d)) ∧
  b = max (min a (max b (max c d))) (max (min b (max c d)) (max (min c d) (min d (min a b))))
  :=
  by
    sorry

end compare_exponents_l99_99345


namespace polynomial_sum_l99_99690

variable {R : Type*} [CommRing R] {x y : R}

/-- Given that the sum of a polynomial P and x^2 - y^2 is x^2 + y^2, we want to prove that P is 2y^2. -/
theorem polynomial_sum (P : R) (h : P + (x^2 - y^2) = x^2 + y^2) : P = 2 * y^2 :=
by
  sorry

end polynomial_sum_l99_99690


namespace investment_schemes_correct_l99_99677

-- Define the parameters of the problem
def num_projects : Nat := 3
def num_districts : Nat := 4

-- Function to count the number of valid investment schemes
def count_investment_schemes (num_projects num_districts : Nat) : Nat :=
  let total_schemes := num_districts ^ num_projects
  let invalid_schemes := num_districts
  total_schemes - invalid_schemes

-- Theorem statement
theorem investment_schemes_correct :
  count_investment_schemes num_projects num_districts = 60 := by
  sorry

end investment_schemes_correct_l99_99677


namespace bond_face_value_l99_99643

theorem bond_face_value
  (F : ℝ)
  (S : ℝ)
  (hS : S = 3846.153846153846)
  (hI1 : I = 0.05 * F)
  (hI2 : I = 0.065 * S) :
  F = 5000 :=
by
  sorry

end bond_face_value_l99_99643


namespace wall_building_problem_l99_99011

theorem wall_building_problem 
    (num_workers_1 : ℕ) (length_wall_1 : ℕ) (days_1 : ℕ)
    (num_workers_2 : ℕ) (length_wall_2 : ℕ) (days_2 : ℕ) :
    num_workers_1 = 8 → length_wall_1 = 140 → days_1 = 42 →
    num_workers_2 = 30 → length_wall_2 = 100 →
    (work_done : ℕ → ℕ → ℕ) → 
    (work_done length_wall_1 days_1 = num_workers_1 * days_1 * length_wall_1) →
    (work_done length_wall_2 days_2 = num_workers_2 * days_2 * length_wall_2) →
    (days_2 = 8) :=
by
  intros h1 h2 h3 h4 h5 wf wlen1 wlen2
  sorry

end wall_building_problem_l99_99011


namespace expected_heads_of_fair_coin_l99_99513

noncomputable def expected_heads (n : ℕ) (p : ℝ) : ℝ := n * p

theorem expected_heads_of_fair_coin :
  expected_heads 5 0.5 = 2.5 :=
by
  sorry

end expected_heads_of_fair_coin_l99_99513


namespace population_growth_l99_99527

theorem population_growth (P : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : P = 5.48) 
  (h₂ : y = P * (1 + x / 100)^8) : 
  y = 5.48 * (1 + x / 100)^8 := 
by
  sorry

end population_growth_l99_99527


namespace toys_in_stock_l99_99855

theorem toys_in_stock (sold_first_week sold_second_week toys_left toys_initial: ℕ) :
  sold_first_week = 38 → 
  sold_second_week = 26 → 
  toys_left = 19 → 
  toys_initial = sold_first_week + sold_second_week + toys_left → 
  toys_initial = 83 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end toys_in_stock_l99_99855


namespace draw_four_balls_in_order_l99_99762

theorem draw_four_balls_in_order :
  let total_balls := 15
  let color_sequence_length := 4
  let colors_sequence := ["Red", "Green", "Blue", "Yellow"]
  total_balls * (total_balls - 1) * (total_balls - 2) * (total_balls - 3) = 32760 :=
by 
  sorry

end draw_four_balls_in_order_l99_99762


namespace find_ordered_pair_l99_99039

theorem find_ordered_pair : ∃ (x y : ℚ), 
  3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ 
  x = 57 / 31 ∧ y = 195 / 62 :=
by {
  sorry
}

end find_ordered_pair_l99_99039


namespace find_alpha_l99_99750

theorem find_alpha (α : Real) (hα : 0 < α ∧ α < π) :
  (∃ x : Real, (|2 * x - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * x| = Real.sin α) ∧ 
  ∀ y : Real, (|2 * y - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * y| = Real.sin α) → y = x) →
  α = π / 12 ∨ α = 11 * π / 12 :=
by
  sorry

end find_alpha_l99_99750


namespace range_of_m_l99_99460

variables (f : ℝ → ℝ) (m : ℝ)

-- Assume f is a decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x > f y

-- Assume f is an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem stating the main condition and the implication
theorem range_of_m (h_decreasing : is_decreasing f) (h_odd : is_odd f) (h_condition : f (m - 1) + f (2 * m - 1) > 0) : m > 2 / 3 :=
sorry

end range_of_m_l99_99460


namespace lyle_payment_l99_99836

def pen_cost : ℝ := 1.50

def notebook_cost : ℝ := 3 * pen_cost

def cost_for_4_notebooks : ℝ := 4 * notebook_cost

theorem lyle_payment : cost_for_4_notebooks = 18.00 :=
by
  sorry

end lyle_payment_l99_99836


namespace abs_eq_case_solution_l99_99238

theorem abs_eq_case_solution :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| := sorry

end abs_eq_case_solution_l99_99238


namespace total_batteries_produced_l99_99524

def time_to_gather_materials : ℕ := 6 -- in minutes
def time_to_create_battery : ℕ := 9   -- in minutes
def num_robots : ℕ := 10
def total_time : ℕ := 5 * 60 -- in minutes (5 hours * 60 minutes/hour)

theorem total_batteries_produced :
  total_time / (time_to_gather_materials + time_to_create_battery) * num_robots = 200 :=
by
  -- Placeholder for the proof steps
  sorry

end total_batteries_produced_l99_99524


namespace min_sum_x_y_l99_99047

theorem min_sum_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y ≥ 9 :=
by sorry

example (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y = 9 ↔ (x = 3 ∧ y = 6) :=
by sorry

end min_sum_x_y_l99_99047


namespace correct_propositions_l99_99850

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * (Real.sin x + Real.cos x)

-- Proposition 2: Symmetry about the line x = -3π/4
def proposition_2 : Prop := ∀ x, f (x + 3 * Real.pi / 4) = f (-x)

-- Proposition 3: There exists φ ∈ ℝ, such that the graph of the function f(x + φ) is centrally symmetric about the origin
def proposition_3 : Prop := ∃ φ : ℝ, ∀ x, f (x + φ) = -f (-x)

theorem correct_propositions :
  (proposition_2 ∧ proposition_3) := by
  sorry

end correct_propositions_l99_99850


namespace factorize_expression_l99_99590

variable {R : Type*} [CommRing R] (a b : R)

theorem factorize_expression : 2 * a^2 * b - 4 * a * b + 2 * b = 2 * b * (a - 1)^2 :=
by
  sorry

end factorize_expression_l99_99590


namespace Pam_has_740_fruits_l99_99485

/-
Define the given conditions.
-/
def Gerald_apple_bags : ℕ := 5
def apples_per_Gerald_bag : ℕ := 30
def Gerald_orange_bags : ℕ := 4
def oranges_per_Gerald_bag : ℕ := 25

def Pam_apple_bags : ℕ := 6
def apples_per_Pam_bag : ℕ := 3 * apples_per_Gerald_bag
def Pam_orange_bags : ℕ := 4
def oranges_per_Pam_bag : ℕ := 2 * oranges_per_Gerald_bag

/-
Proving the total number of apples and oranges Pam has.
-/
def total_fruits_Pam : ℕ :=
    Pam_apple_bags * apples_per_Pam_bag + Pam_orange_bags * oranges_per_Pam_bag

theorem Pam_has_740_fruits : total_fruits_Pam = 740 := by
  sorry

end Pam_has_740_fruits_l99_99485


namespace intersection_of_sets_l99_99913

open Set

theorem intersection_of_sets : 
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  M ∩ N = {0, 4, 8} := 
by
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  sorry

end intersection_of_sets_l99_99913


namespace total_visitors_over_two_days_l99_99464

-- Conditions given in the problem statement
def first_day_visitors : ℕ := 583
def second_day_visitors : ℕ := 246

-- The main problem: proving the total number of visitors over the two days
theorem total_visitors_over_two_days : first_day_visitors + second_day_visitors = 829 := by
  -- Proof is omitted
  sorry

end total_visitors_over_two_days_l99_99464


namespace intersection_complement_correct_l99_99916

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {1, 4}
def C_I (s : Set ℕ) := I \ s  -- set complement

theorem intersection_complement_correct: A ∩ C_I B = {3, 5} := by
  -- proof steps go here
  sorry

end intersection_complement_correct_l99_99916


namespace Carolyn_wants_to_embroider_l99_99200

theorem Carolyn_wants_to_embroider (s : ℕ) (f : ℕ) (u : ℕ) (g : ℕ) (n_f : ℕ) (t : ℕ) (number_of_unicorns : ℕ) :
  s = 4 ∧ f = 60 ∧ u = 180 ∧ g = 800 ∧ n_f = 50 ∧ t = 1085 ∧ 
  (t * s - (n_f * f) - g) / u = number_of_unicorns ↔ number_of_unicorns = 3 :=
by 
  sorry

end Carolyn_wants_to_embroider_l99_99200


namespace not_possible_to_color_l99_99765

theorem not_possible_to_color (f : ℕ → ℕ) (c1 c2 c3 : ℕ) :
  ∃ (x : ℕ), 1 < x ∧ f 2 = c1 ∧ f 4 = c1 ∧ 
  ∀ (a b : ℕ), 1 < a → 1 < b → f a ≠ f b → (f (a * b) ≠ f a ∧ f (a * b) ≠ f b) → 
  false :=
sorry

end not_possible_to_color_l99_99765


namespace number_of_boys_is_12500_l99_99361

-- Define the number of boys and girls in the school
def numberOfBoys (B : ℕ) : ℕ := B
def numberOfGirls : ℕ := 5000

-- Define the total attendance
def totalAttendance (B : ℕ) : ℕ := B + numberOfGirls

-- Define the condition for the percentage increase from boys to total attendance
def percentageIncreaseCondition (B : ℕ) : Prop :=
  totalAttendance B = B + Int.ofNat numberOfGirls

-- Statement to prove
theorem number_of_boys_is_12500 (B : ℕ) (h : totalAttendance B = B + numberOfGirls) : B = 12500 :=
sorry

end number_of_boys_is_12500_l99_99361


namespace x_eq_1_sufficient_but_not_necessary_l99_99038

theorem x_eq_1_sufficient_but_not_necessary (x : ℝ) : x^2 - 3 * x + 2 = 0 → (x = 1 ↔ true) ∧ (x ≠ 1 → ∃ y : ℝ, y ≠ x ∧ y^2 - 3 * y + 2 = 0) :=
by
  sorry

end x_eq_1_sufficient_but_not_necessary_l99_99038


namespace michael_truck_meet_once_l99_99912

/-- Michael walks at 6 feet per second -/
def michael_speed := 6

/-- Trash pails are located every 300 feet along the path -/
def pail_distance := 300

/-- A garbage truck travels at 15 feet per second -/
def truck_speed := 15

/-- The garbage truck stops for 45 seconds at each pail -/
def truck_stop_time := 45

/-- Michael passes a pail just as the truck leaves the next pail -/
def initial_distance := 300

/-- Prove that Michael and the truck meet exactly 1 time -/
theorem michael_truck_meet_once :
  ∀ (meeting_times : ℕ), meeting_times = 1 := by
  sorry

end michael_truck_meet_once_l99_99912


namespace john_back_squat_increase_l99_99395

-- Definitions based on conditions
def back_squat_initial : ℝ := 200
def k : ℝ := 0.8
def j : ℝ := 0.9
def total_weight_moved : ℝ := 540

-- The variable representing the increase in back squat
variable (x : ℝ)

-- The Lean statement to prove
theorem john_back_squat_increase :
  3 * (j * k * (back_squat_initial + x)) = total_weight_moved → x = 50 := by
  sorry

end john_back_squat_increase_l99_99395


namespace parabola_distance_l99_99432

theorem parabola_distance (x y : ℝ) (h_parabola : y^2 = 8 * x)
  (h_distance_focus : ∀ x y, (x - 2)^2 + y^2 = 6^2) :
  abs x = 4 :=
by sorry

end parabola_distance_l99_99432


namespace find_starting_number_l99_99108

theorem find_starting_number (n : ℤ) (h1 : ∀ k : ℤ, n ≤ k ∧ k ≤ 38 → k % 4 = 0) (h2 : (n + 38) / 2 = 22) : n = 8 :=
sorry

end find_starting_number_l99_99108


namespace clever_value_points_l99_99060

def clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = (deriv f) x₀

theorem clever_value_points :
  (clever_value_point (fun x : ℝ => x^2)) ∧
  (clever_value_point (fun x : ℝ => Real.log x)) ∧
  (clever_value_point (fun x : ℝ => x + (1 / x))) :=
by
  -- Proof omitted
  sorry

end clever_value_points_l99_99060


namespace problem_solution_l99_99325

noncomputable def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 - p * x + q

theorem problem_solution
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : p > 0)
  (h3 : q > 0)
  (h4 : f a p q = 0)
  (h5 : f b p q = 0)
  (h6 : ∃ k : ℝ, (a = -2 + k ∧ b = -2 - k) ∨ (a = -2 - k ∧ b = -2 + k))
  (h7 : ∃ l : ℝ, (a = -2 * l ∧ b = 4 * l) ∨ (a = 4 * l ∧ b = -2 * l))
  : p + q = 9 :=
sorry

end problem_solution_l99_99325


namespace balloon_permutations_l99_99447

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end balloon_permutations_l99_99447


namespace remaining_funds_correct_l99_99076

def david_initial_funds : ℝ := 1800
def emma_initial_funds : ℝ := 2400
def john_initial_funds : ℝ := 1200

def david_spent_percentage : ℝ := 0.60
def emma_spent_percentage : ℝ := 0.75
def john_spent_percentage : ℝ := 0.50

def david_remaining_funds : ℝ := david_initial_funds * (1 - david_spent_percentage)
def emma_spent : ℝ := emma_initial_funds * emma_spent_percentage
def emma_remaining_funds : ℝ := emma_spent - 800
def john_remaining_funds : ℝ := john_initial_funds * (1 - john_spent_percentage)

theorem remaining_funds_correct :
  david_remaining_funds = 720 ∧
  emma_remaining_funds = 1400 ∧
  john_remaining_funds = 600 :=
by
  sorry

end remaining_funds_correct_l99_99076


namespace unique_ab_not_determined_l99_99932

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - Real.sqrt 2

theorem unique_ab_not_determined :
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  f a b (f a b (Real.sqrt 2)) = 1 → False := 
by
  sorry

end unique_ab_not_determined_l99_99932


namespace batsman_average_after_17th_inning_l99_99686

theorem batsman_average_after_17th_inning (A : ℝ) :
  let total_runs_after_17_innings := 16 * A + 87
  let new_average := total_runs_after_17_innings / 17
  new_average = A + 3 → 
  (A + 3) = 39 :=
by
  sorry

end batsman_average_after_17th_inning_l99_99686


namespace count_triangles_with_center_inside_l99_99826

theorem count_triangles_with_center_inside :
  let n := 201
  let num_triangles_with_center_inside (n : ℕ) : ℕ := 
    let half := n / 2
    let group_count := half * (half + 1) / 2
    group_count * n / 3
  num_triangles_with_center_inside n = 338350 :=
by
  sorry

end count_triangles_with_center_inside_l99_99826


namespace balance_problem_l99_99622

variable {G B Y W : ℝ}

theorem balance_problem
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 7.5 * B)
  (h3 : 5 * B = 3.5 * W) :
  5 * G + 4 * Y + 3 * W = (170 / 7) * B := by
  sorry

end balance_problem_l99_99622


namespace trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l99_99277

open Real

theorem trig_cos2_minus_sin2_eq_neg_sqrt5_div3 (α : ℝ) (hα1 : 0 < α ∧ α < π) (hα2 : sin α + cos α = sqrt 3 / 3) :
  cos α ^ 2 - sin α ^ 2 = - sqrt 5 / 3 := 
  sorry

end trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l99_99277


namespace frac_two_over_x_values_l99_99978

theorem frac_two_over_x_values (x : ℝ) (h : 1 - 9 / x + 20 / (x ^ 2) = 0) :
  (2 / x = 1 / 2 ∨ 2 / x = 0.4) :=
sorry

end frac_two_over_x_values_l99_99978


namespace proposition_1_proposition_2_proposition_3_proposition_4_l99_99409

theorem proposition_1 : ∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0 := sorry

theorem proposition_2 : ¬ (∀ x ∈ ({-1, 0, 1} : Set ℤ), 2 * x + 1 > 0) := sorry

theorem proposition_3 : ∃ x : ℕ, x^2 ≤ x := sorry

theorem proposition_4 : ∃ x : ℕ, x ∣ 29 := sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l99_99409


namespace min_cube_edge_division_l99_99141

theorem min_cube_edge_division (n : ℕ) (h : n^3 ≥ 1996) : n = 13 :=
by {
  sorry
}

end min_cube_edge_division_l99_99141


namespace water_breaks_frequency_l99_99948

theorem water_breaks_frequency :
  ∃ W : ℕ, (240 / 120 + 10) = 240 / W :=
by
  existsi (20 : ℕ)
  sorry

end water_breaks_frequency_l99_99948


namespace temperature_decrease_l99_99136

theorem temperature_decrease (rise_1_degC : ℝ) (decrease_2_degC : ℝ) 
  (h : rise_1_degC = 1) : decrease_2_degC = -2 :=
by 
  -- This is the statement with the condition and problem to be proven:
  sorry

end temperature_decrease_l99_99136


namespace set_intersection_complement_l99_99746

open Set

variable (U P Q: Set ℕ)

theorem set_intersection_complement (hU: U = {1, 2, 3, 4}) (hP: P = {1, 2}) (hQ: Q = {2, 3}) :
  P ∩ (U \ Q) = {1} :=
by
  sorry

end set_intersection_complement_l99_99746


namespace solution_set_of_inequality_l99_99088

theorem solution_set_of_inequality (x : ℝ) : 
  (|x| * (1 - 2 * x) > 0) ↔ (x ∈ ((Set.Iio 0) ∪ (Set.Ioo 0 (1/2)))) :=
by
  sorry

end solution_set_of_inequality_l99_99088


namespace three_digit_number_is_504_l99_99139

theorem three_digit_number_is_504 (x : ℕ) [Decidable (x = 504)] :
  100 ≤ x ∧ x ≤ 999 →
  (x - 7) % 7 = 0 ∧
  (x - 8) % 8 = 0 ∧
  (x - 9) % 9 = 0 →
  x = 504 :=
by
  sorry

end three_digit_number_is_504_l99_99139


namespace interval_intersection_l99_99233

theorem interval_intersection (x : ℝ) : 
  (1 < 4 * x ∧ 4 * x < 3) ∧ (2 < 6 * x ∧ 6 * x < 4) ↔ (1 / 3 < x ∧ x < 2 / 3) := 
by 
  sorry

end interval_intersection_l99_99233


namespace quadratic_has_two_distinct_real_roots_l99_99174

-- Define the quadratic equation and its coefficients
def a := 1
def b := -4
def c := -3

-- Define the discriminant function for a quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- State the problem in Lean: Prove that the quadratic equation x^2 - 4x - 3 = 0 has a positive discriminant.
theorem quadratic_has_two_distinct_real_roots : discriminant a b c > 0 :=
by
  sorry -- This is where the proof would go

end quadratic_has_two_distinct_real_roots_l99_99174


namespace proposition_p_l99_99822

variable (x : ℝ)

-- Define condition
def negation_of_p : Prop := ∃ x, x < 1 ∧ x^2 < 1

-- Define proposition p
def p : Prop := ∀ x, x < 1 → x^2 ≥ 1

-- Theorem statement
theorem proposition_p (h : negation_of_p) : (p) :=
sorry

end proposition_p_l99_99822


namespace additional_people_required_l99_99056

-- Given condition: Four people can mow a lawn in 6 hours
def work_rate: ℕ := 4 * 6

-- New condition: Number of people needed to mow the lawn in 3 hours
def people_required_in_3_hours: ℕ := work_rate / 3

-- Statement: Number of additional people required
theorem additional_people_required : people_required_in_3_hours - 4 = 4 :=
by
  -- Proof would go here
  sorry

end additional_people_required_l99_99056


namespace fraction_addition_l99_99638

theorem fraction_addition : (3 / 8) + (9 / 12) = 9 / 8 := sorry

end fraction_addition_l99_99638


namespace calculate_total_students_l99_99897

-- Define the conditions and state the theorem
theorem calculate_total_students (perc_bio : ℝ) (num_not_bio : ℝ) (perc_not_bio : ℝ) (T : ℝ) :
  perc_bio = 0.475 →
  num_not_bio = 462 →
  perc_not_bio = 1 - perc_bio →
  perc_not_bio * T = num_not_bio →
  T = 880 :=
by
  intros
  -- proof will be here
  sorry

end calculate_total_students_l99_99897


namespace pow_evaluation_l99_99779

theorem pow_evaluation : 81^(5/4) = 243 := 
by sorry

end pow_evaluation_l99_99779


namespace vector_odot_not_symmetric_l99_99704

-- Define the vector operation ⊛
def vector_odot (a b : ℝ × ℝ) : ℝ :=
  let (m, n) := a
  let (p, q) := b
  m * q - n * p

-- Statement: Prove that the operation is not symmetric
theorem vector_odot_not_symmetric (a b : ℝ × ℝ) : vector_odot a b ≠ vector_odot b a := by
  sorry

end vector_odot_not_symmetric_l99_99704


namespace math_proof_l99_99124

open Real

noncomputable def function (a b x : ℝ): ℝ := a * x^3 + b * x^2

theorem math_proof (a b : ℝ) :
  (function a b 1 = 3) ∧
  (deriv (function a b) 1 = 0) ∧
  (∃ (a b : ℝ), a = -6 ∧ b = 9 ∧ 
    function a b = -6 * (x^3) + 9 * (x^2)) ∧
  (∀ x, (0 < x ∧ x < 1) → deriv (function a b) x > 0) ∧
  (∀ x, (x < 0 ∨ x > 1) → deriv (function a b) x < 0) ∧
  (min (function a b (-2)) (function a b 2) = (-12)) ∧
  (max (function a b (-2)) (function a b 2) = 84) :=
by
  sorry

end math_proof_l99_99124


namespace geometric_sequence_properties_l99_99902

noncomputable def geometric_sequence_sum (r a1 : ℝ) : Prop :=
  a1 * (r^3 + r^4) = 27 ∨ a1 * (r^3 + r^4) = -27

theorem geometric_sequence_properties (a1 r : ℝ) (h1 : a1 + a1 * r = 1) (h2 : a1 * r^2 + a1 * r^3 = 9) :
  geometric_sequence_sum r a1 :=
sorry

end geometric_sequence_properties_l99_99902


namespace negation_example_l99_99355

theorem negation_example :
  (¬ (∀ x: ℝ, x > 0 → x^2 + x + 1 > 0)) ↔ (∃ x: ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end negation_example_l99_99355


namespace max_cables_cut_l99_99821

def initial_cameras : ℕ := 200
def initial_cables : ℕ := 345
def resulting_clusters : ℕ := 8

theorem max_cables_cut :
  ∃ (cables_cut : ℕ), resulting_clusters = 8 ∧ initial_cables - cables_cut = (initial_cables - cables_cut) - (resulting_clusters - 1) ∧ cables_cut = 153 :=
by
  sorry

end max_cables_cut_l99_99821


namespace find_s2_side_length_l99_99616

-- Define the variables involved
variables (r s : ℕ)

-- Conditions based on problem statement
def height_eq : Prop := 2 * r + s = 2160
def width_eq : Prop := 2 * r + 3 * s + 110 = 4020

-- The theorem stating that s = 875 given the conditions
theorem find_s2_side_length (h1 : height_eq r s) (h2 : width_eq r s) : s = 875 :=
by {
  sorry
}

end find_s2_side_length_l99_99616


namespace inequality_solution_set_l99_99225

theorem inequality_solution_set (x : ℝ) :
  ∀ x, 
  (x^2 * (x + 1) / (-x^2 - 5 * x + 6) <= 0) ↔ (-6 < x ∧ x <= -1) ∨ (x = 0) ∨ (1 < x) :=
by
  sorry

end inequality_solution_set_l99_99225


namespace joan_seashells_total_l99_99761

-- Definitions
def original_seashells : ℕ := 70
def additional_seashells : ℕ := 27
def total_seashells : ℕ := original_seashells + additional_seashells

-- Proof Statement
theorem joan_seashells_total : total_seashells = 97 := by
  sorry

end joan_seashells_total_l99_99761


namespace correct_multiplication_l99_99153

theorem correct_multiplication (n : ℕ) (h₁ : 15 * n = 45) : 5 * n = 15 :=
by
  -- skipping the proof
  sorry

end correct_multiplication_l99_99153


namespace sum_of_squares_of_roots_l99_99801

/-- If r, s, and t are the roots of the cubic equation x³ - ax² + bx - c = 0, then r² + s² + t² = a² - 2b. -/
theorem sum_of_squares_of_roots (r s t a b c : ℝ) (h1 : r + s + t = a) (h2 : r * s + r * t + s * t = b) (h3 : r * s * t = c) :
    r ^ 2 + s ^ 2 + t ^ 2 = a ^ 2 - 2 * b := 
by 
  sorry

end sum_of_squares_of_roots_l99_99801


namespace part1_part2_l99_99571

open Set

-- Define the sets M and N based on given conditions
def M (a : ℝ) : Set ℝ := { x | (x + a) * (x - 1) ≤ 0 }
def N : Set ℝ := { x | 4 * x^2 - 4 * x - 3 < 0 }

-- Part (1): Prove that if M ∪ N = { x | -2 ≤ x < 3 / 2 }, then a = 2
theorem part1 (a : ℝ) (h : a > 0)
  (h_union : M a ∪ N = { x | -2 ≤ x ∧ x < 3 / 2 }) : a = 2 := by
  sorry

-- Part (2): Prove that if N ∪ (compl (M a)) = univ, then 0 < a ≤ 1/2
theorem part2 (a : ℝ) (h : a > 0)
  (h_union : N ∪ compl (M a) = univ) : 0 < a ∧ a ≤ 1 / 2 := by
  sorry

end part1_part2_l99_99571


namespace number_of_unit_distance_pairs_lt_bound_l99_99362

/-- Given n distinct points in the plane, the number of pairs of points with a unit distance between them is less than n / 4 + (1 / sqrt 2) * n^(3 / 2). -/
theorem number_of_unit_distance_pairs_lt_bound (n : ℕ) (hn : 0 < n) :
  ∃ E : ℕ, E < n / 4 + (1 / Real.sqrt 2) * n^(3 / 2) :=
by
  sorry

end number_of_unit_distance_pairs_lt_bound_l99_99362


namespace age_proof_l99_99641

   variable (x : ℝ)
   
   theorem age_proof (h : 3 * (x + 5) - 3 * (x - 5) = x) : x = 30 :=
   by
     sorry
   
end age_proof_l99_99641


namespace exist_rectangle_same_color_l99_99577

-- Define the colors.
inductive Color
| red
| green
| blue

open Color

-- Define the point and the plane.
structure Point :=
(x : ℝ) (y : ℝ)

-- Assume a coloring function that assigns colors to points on the plane.
def coloring : Point → Color := sorry

-- The theorem stating the existence of a rectangle with vertices of the same color.
theorem exist_rectangle_same_color :
  ∃ (A B C D : Point), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  coloring A = coloring B ∧ coloring B = coloring C ∧ coloring C = coloring D :=
sorry

end exist_rectangle_same_color_l99_99577


namespace apples_per_person_l99_99634

-- Define conditions
def total_apples : ℝ := 45
def number_of_people : ℝ := 3.0

-- Theorem statement: Calculate how many apples each person received.
theorem apples_per_person : 
  (total_apples / number_of_people) = 15 := 
by
  sorry

end apples_per_person_l99_99634


namespace proof_sin_315_eq_neg_sqrt_2_div_2_l99_99149

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end proof_sin_315_eq_neg_sqrt_2_div_2_l99_99149


namespace mixed_numbers_sum_l99_99383

-- Declare the mixed numbers as fraction equivalents
def mixed1 : ℚ := 2 + 1/10
def mixed2 : ℚ := 3 + 11/100
def mixed3 : ℚ := 4 + 111/1000

-- Assert that the sum of mixed1, mixed2, and mixed3 is equal to 9.321
theorem mixed_numbers_sum : mixed1 + mixed2 + mixed3 = 9321 / 1000 := by
  sorry

end mixed_numbers_sum_l99_99383


namespace volume_P3_correct_m_plus_n_l99_99605

noncomputable def P_0_volume : ℚ := 1

noncomputable def tet_volume (v : ℚ) : ℚ := (1/27) * v

noncomputable def volume_P3 : ℚ := 
  let ΔP1 := 4 * tet_volume P_0_volume
  let ΔP2 := (2/9) * ΔP1
  let ΔP3 := (2/9) * ΔP2
  P_0_volume + ΔP1 + ΔP2 + ΔP3

theorem volume_P3_correct : volume_P3 = 22615 / 6561 := 
by {
  sorry
}

theorem m_plus_n : 22615 + 6561 = 29176 := 
by {
  sorry
}

end volume_P3_correct_m_plus_n_l99_99605


namespace moe_share_of_pie_l99_99055

-- Definitions based on conditions
def leftover_pie : ℚ := 8 / 9
def num_people : ℚ := 3

-- Theorem to prove the amount of pie Moe took home
theorem moe_share_of_pie : (leftover_pie / num_people) = 8 / 27 := by
  sorry

end moe_share_of_pie_l99_99055


namespace order_of_t_t2_neg_t_l99_99061

theorem order_of_t_t2_neg_t (t : ℝ) (h : t^2 + t < 0) : t < t^2 ∧ t^2 < -t :=
by
  sorry

end order_of_t_t2_neg_t_l99_99061


namespace puzzle_pieces_l99_99252

theorem puzzle_pieces (x : ℝ) (h : x + 2 * 1.5 * x = 4000) : x = 1000 :=
  sorry

end puzzle_pieces_l99_99252


namespace restaurant_total_earnings_l99_99859

noncomputable def restaurant_earnings (weekdays weekends : ℕ) (weekday_earnings : ℝ) 
    (weekend_min_earnings weekend_max_earnings discount special_event_earnings : ℝ) : ℝ :=
  let num_mondays := weekdays / 5 
  let weekday_earnings_with_discount := weekday_earnings - (weekday_earnings * discount)
  let earnings_mondays := num_mondays * weekday_earnings_with_discount
  let earnings_other_weekdays := (weekdays - num_mondays) * weekday_earnings
  let average_weekend_earnings := (weekend_min_earnings + weekend_max_earnings) / 2
  let total_weekday_earnings := earnings_mondays + earnings_other_weekdays
  let total_weekend_earnings := 2 * weekends * average_weekend_earnings
  total_weekday_earnings + total_weekend_earnings + special_event_earnings

theorem restaurant_total_earnings 
  (weekdays weekends : ℕ)
  (weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings total_earnings : ℝ)
  (h_weekdays : weekdays = 22)
  (h_weekends : weekends = 8)
  (h_weekday_earnings : weekday_earnings = 600)
  (h_weekend_min_earnings : weekend_min_earnings = 1000)
  (h_weekend_max_earnings : weekend_max_earnings = 1500)
  (h_discount : discount = 0.1)
  (h_special_event_earnings : special_event_earnings = 500)
  (h_total_earnings : total_earnings = 33460) :
  restaurant_earnings weekdays weekends weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings = total_earnings := 
by
  sorry

end restaurant_total_earnings_l99_99859


namespace problem_statement_l99_99596

noncomputable def α : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def β : ℝ := 3 - 2 * Real.sqrt 2
noncomputable def x : ℝ := α ^ 50
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end problem_statement_l99_99596


namespace f_zero_unique_l99_99118

theorem f_zero_unique (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x + f (xy)) : f 0 = 0 :=
by {
  -- proof goes here
  sorry
}

end f_zero_unique_l99_99118


namespace min_value_of_sum_l99_99804

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + 2 * b = 1) : 
  (∃ x, x = (3 / a + 2 / b) ∧ x = 25) :=
sorry

end min_value_of_sum_l99_99804


namespace min_sum_equals_nine_l99_99413

theorem min_sum_equals_nine (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 4 * a + b - a * b = 0) : a + b = 9 :=
by
  sorry

end min_sum_equals_nine_l99_99413


namespace tallest_is_first_l99_99398

variable (P : Type) -- representing people
variable (line : Fin 9 → P) -- original line order (0 = shortest, 8 = tallest)
variable (Hoseok : P) -- Hoseok

-- Conditions
axiom tallest_person : line 8 = Hoseok

-- Theorem
theorem tallest_is_first :
  ∃ line' : Fin 9 → P, (∀ i : Fin 9, line' i = line (8 - i)) → line' 0 = Hoseok :=
  by
  sorry

end tallest_is_first_l99_99398


namespace fill_half_jar_in_18_days_l99_99451

-- Define the doubling condition and the days required to fill half the jar
variable (area : ℕ → ℕ)
variable (doubling : ∀ t, area (t + 1) = 2 * area t)
variable (full_jar : area 19 = 2^19)
variable (half_jar : area 18 = 2^18)

theorem fill_half_jar_in_18_days :
  ∃ n, n = 18 ∧ area n = 2^18 :=
by {
  -- The proof is omitted, but we state the goal
  sorry
}

end fill_half_jar_in_18_days_l99_99451


namespace sum_coordinates_point_C_l99_99679

/-
Let point A = (0,0), point B is on the line y = 6, and the slope of AB is 3/4.
Point C lies on the y-axis with a slope of 1/2 from B to C.
We need to prove that the sum of the coordinates of point C is 2.
-/
theorem sum_coordinates_point_C : 
  ∃ (A B C : ℝ × ℝ), 
  A = (0, 0) ∧ 
  B.2 = 6 ∧ 
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 ∧ 
  C.1 = 0 ∧ 
  (C.2 - B.2) / (C.1 - B.1) = 1 / 2 ∧ 
  C.1 + C.2 = 2 :=
by
  sorry

end sum_coordinates_point_C_l99_99679


namespace men_absent_l99_99678

theorem men_absent (n : ℕ) (d1 d2 : ℕ) (x : ℕ) 
  (h1 : n = 22) 
  (h2 : d1 = 20) 
  (h3 : d2 = 22) 
  (hc : n * d1 = (n - x) * d2) : 
  x = 2 := 
by {
  sorry
}

end men_absent_l99_99678


namespace solve_for_x_l99_99507

theorem solve_for_x (x : ℝ) (h : (9 + 1/x)^(1/3) = -2) : x = -1/17 :=
by
  sorry

end solve_for_x_l99_99507


namespace probability_odd_sum_probability_even_product_l99_99975
open Classical

noncomputable def number_of_possible_outcomes : ℕ := 36
noncomputable def number_of_odd_sum_outcomes : ℕ := 18
noncomputable def number_of_even_product_outcomes : ℕ := 27

theorem probability_odd_sum (n : ℕ) (m_1 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_1 = number_of_odd_sum_outcomes) : (m_1 : ℝ) / n = 1 / 2 :=
by
  sorry

theorem probability_even_product (n : ℕ) (m_2 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_2 = number_of_even_product_outcomes) : (m_2 : ℝ) / n = 3 / 4 :=
by
  sorry

end probability_odd_sum_probability_even_product_l99_99975


namespace inverse_function_l99_99911

theorem inverse_function (x : ℝ) (hx : x > 1) : ∃ y : ℝ, x = 2^y + 1 ∧ y = Real.logb 2 (x - 1) :=
sorry

end inverse_function_l99_99911


namespace geometric_sequence_increasing_condition_l99_99063

theorem geometric_sequence_increasing_condition (a₁ a₂ a₄ : ℝ) (q : ℝ) (n : ℕ) (a : ℕ → ℝ):
  (∀ n, a n = a₁ * q^n) →
  (a₁ < a₂ ∧ a₂ < a₄) → 
  ¬ (∀ n, a n < a (n + 1)) → 
  (a₁ < a₂ ∧ a₂ < a₄) ∧ ¬ (∀ n, a n < a (n + 1)) :=
sorry

end geometric_sequence_increasing_condition_l99_99063


namespace expand_product_l99_99646

theorem expand_product (x : ℤ) : 
  (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 :=
by
  sorry

end expand_product_l99_99646


namespace find_reduced_price_l99_99811

noncomputable def reduced_price_per_kg 
  (total_spent : ℝ) (original_quantity : ℝ) (additional_quantity : ℝ) (price_reduction_rate : ℝ) : ℝ :=
  let original_price := total_spent / original_quantity
  let reduced_price := original_price * (1 - price_reduction_rate)
  reduced_price

theorem find_reduced_price 
  (total_spent : ℝ := 800)
  (original_quantity : ℝ := 20)
  (additional_quantity : ℝ := 5)
  (price_reduction_rate : ℝ := 0.15) :
  reduced_price_per_kg total_spent original_quantity additional_quantity price_reduction_rate = 34 :=
by
  sorry

end find_reduced_price_l99_99811


namespace vic_max_marks_l99_99837

theorem vic_max_marks (M : ℝ) (h : 0.92 * M = 368) : M = 400 := 
sorry

end vic_max_marks_l99_99837


namespace add_base3_numbers_l99_99020

-- Definitions to represent the numbers in base 3
def base3_num1 := (2 : ℕ) -- 2_3
def base3_num2 := (2 * 3 + 2 : ℕ) -- 22_3
def base3_num3 := (2 * 3^2 + 0 * 3 + 2 : ℕ) -- 202_3
def base3_num4 := (2 * 3^3 + 0 * 3^2 + 2 * 3 + 2 : ℕ) -- 2022_3

-- Summing the numbers in base 10 first
def sum_base10 := base3_num1 + base3_num2 + base3_num3 + base3_num4

-- Expected result in base 10 for 21010_3
def result_base10 := 2 * 3^4 + 1 * 3^3 + 0 * 3^2 + 1 * 3 + 0

-- Proof statement
theorem add_base3_numbers : sum_base10 = result_base10 :=
by {
  -- Proof not required, so we skip it using sorry
  sorry
}

end add_base3_numbers_l99_99020


namespace sufficient_but_not_necessary_condition_l99_99803

theorem sufficient_but_not_necessary_condition 
  (a b : ℝ) (h : a > b ∧ b > 0) : (a^2 > b^2) ∧ (¬ ∀ (a' b' : ℝ), a'^2 > b'^2 → a' > b' ∧ b' > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l99_99803


namespace function_identity_l99_99138

theorem function_identity
    (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x ≤ x)
    (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
    ∀ x : ℝ, f x = x :=
by
    sorry

end function_identity_l99_99138


namespace find_a_subtract_two_l99_99276

theorem find_a_subtract_two (a b : ℤ) 
    (h1 : 2 + a = 5 - b) 
    (h2 : 5 + b = 8 + a) : 
    2 - a = 2 := 
by
  sorry

end find_a_subtract_two_l99_99276


namespace root_of_quadratic_expression_l99_99576

theorem root_of_quadratic_expression (n : ℝ) (h : n^2 - 5 * n + 4 = 0) : n^2 - 5 * n = -4 :=
by
  sorry

end root_of_quadratic_expression_l99_99576


namespace least_number_to_add_l99_99719

theorem least_number_to_add (n : ℕ) (H : n = 433124) : ∃ k, k = 15 ∧ (n + k) % 17 = 0 := by
  sorry

end least_number_to_add_l99_99719


namespace min_value_sin_cos_l99_99927

open Real

theorem min_value_sin_cos : ∀ x : ℝ, 
  ∃ (y : ℝ), (∀ x, y ≤ sin x ^ 6 + (5 / 3) * cos x ^ 6) ∧ y = 5 / 8 :=
by
  sorry

end min_value_sin_cos_l99_99927


namespace minimize_f_l99_99429

noncomputable def f : ℝ → ℝ := λ x => (3/2) * x^2 - 9 * x + 7

theorem minimize_f : ∀ x, f x ≥ f 3 :=
by 
  intro x
  sorry

end minimize_f_l99_99429


namespace inequality_proof_l99_99766

theorem inequality_proof
  (a b c d e f : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (he : 0 < e)
  (hf : 0 < f)
  (h_condition : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := 
sorry

end inequality_proof_l99_99766


namespace Maggie_earnings_l99_99378

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end Maggie_earnings_l99_99378


namespace find_three_xsq_ysq_l99_99356

theorem find_three_xsq_ysq (x y : ℤ) (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 3*x^2*y^2 = 588 :=
sorry

end find_three_xsq_ysq_l99_99356


namespace integer_roots_p_l99_99073

theorem integer_roots_p (p x1 x2 : ℤ) (h1 : x1 * x2 = p + 4) (h2 : x1 + x2 = -p) : p = 8 ∨ p = -4 := 
sorry

end integer_roots_p_l99_99073


namespace probability_age_between_30_and_40_l99_99655

-- Assume total number of people in the group is 200
def total_people : ℕ := 200

-- Assume 80 people have an age of more than 40 years
def people_age_more_than_40 : ℕ := 80

-- Assume 70 people have an age between 30 and 40 years
def people_age_between_30_and_40 : ℕ := 70

-- Assume 30 people have an age between 20 and 30 years
def people_age_between_20_and_30 : ℕ := 30

-- Assume 20 people have an age of less than 20 years
def people_age_less_than_20 : ℕ := 20

-- The proof problem statement
theorem probability_age_between_30_and_40 :
  (people_age_between_30_and_40 : ℚ) / (total_people : ℚ) = 7 / 20 :=
by
  sorry

end probability_age_between_30_and_40_l99_99655


namespace teacher_A_realizes_fish_l99_99665

variable (Teacher : Type) (has_fish : Teacher → Prop) (is_laughing : Teacher → Prop)
variables (A B C : Teacher)

-- Initial assumptions
axiom all_laughing : is_laughing A ∧ is_laughing B ∧ is_laughing C
axiom each_thinks_others_have_fish : (¬has_fish A ∧ has_fish B ∧ has_fish C) 
                                      ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
                                      ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C)

-- The logical conclusion
theorem teacher_A_realizes_fish : (∃ A B C : Teacher, 
  is_laughing A ∧ is_laughing B ∧ is_laughing C ∧
  ((¬has_fish A ∧ has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C))) →
  (has_fish A ∧ is_laughing B ∧ is_laughing C) :=
sorry -- proof not required.

end teacher_A_realizes_fish_l99_99665


namespace compare_numbers_l99_99990

theorem compare_numbers :
  2^27 < 10^9 ∧ 10^9 < 5^13 :=
by {
  sorry
}

end compare_numbers_l99_99990


namespace place_value_ratio_l99_99852

def number : ℝ := 90347.6208
def place_value_0 : ℝ := 10000 -- tens of thousands
def place_value_6 : ℝ := 0.1 -- tenths

theorem place_value_ratio : 
  place_value_0 / place_value_6 = 100000 := by 
    sorry

end place_value_ratio_l99_99852


namespace abs_val_of_5_minus_e_l99_99148

theorem abs_val_of_5_minus_e : ∀ (e : ℝ), e = 2.718 → |5 - e| = 2.282 :=
by
  intros e he
  sorry

end abs_val_of_5_minus_e_l99_99148


namespace no_valid_prime_angles_l99_99077

def is_prime (n : ℕ) : Prop := Prime n

theorem no_valid_prime_angles :
  ∀ (x : ℕ), (x < 30) ∧ is_prime x ∧ is_prime (3 * x) → False :=
by sorry

end no_valid_prime_angles_l99_99077


namespace option_D_correct_l99_99871

-- Defining the types for lines and planes
variables {Line Plane : Type}

-- Defining what's needed for perpendicularity and parallelism
variables (perp : Line → Plane → Prop)
variables (subset : Line → Plane → Prop)
variables (parallel : Line → Line → Prop)
variables (perp_planes : Plane → Plane → Prop)

-- Main theorem statement
theorem option_D_correct (a b : Line) (α β : Plane) :
  perp a α → subset b β → parallel a b → perp_planes α β :=
by
  sorry

end option_D_correct_l99_99871


namespace quadratic_inequality_range_l99_99966

variable (x : ℝ)

-- Statement of the mathematical problem
theorem quadratic_inequality_range (h : ¬ (x^2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end quadratic_inequality_range_l99_99966


namespace attendees_received_all_items_l99_99206

theorem attendees_received_all_items {n : ℕ} (h1 : ∀ k, k ∣ 45 → n % k = 0) (h2 : ∀ k, k ∣ 75 → n % k = 0) (h3 : ∀ k, k ∣ 100 → n % k = 0) (h4 : n = 4500) :
  (4500 / Nat.lcm (Nat.lcm 45 75) 100) = 5 :=
by
  sorry

end attendees_received_all_items_l99_99206


namespace determine_k_l99_99467

theorem determine_k (k : ℝ) : 
  (∀ x : ℝ, (x^2 = 2 * x + k) → (∃ x0 : ℝ, ∀ x : ℝ, (x - x0)^2 = 0)) ↔ k = -1 :=
by 
  sorry

end determine_k_l99_99467


namespace tank_salt_solution_l99_99760

theorem tank_salt_solution (x : ℝ) (hx1 : 0.20 * x / (3 / 4 * x + 30) = 1 / 3) : x = 200 :=
by sorry

end tank_salt_solution_l99_99760


namespace factorable_iff_some_even_b_l99_99517

open Int

theorem factorable_iff_some_even_b (b : ℤ) :
  (∃ m n p q : ℤ,
    (35 : ℤ) = m * p ∧
    (35 : ℤ) = n * q ∧
    b = m * q + n * p) →
  (∃ k : ℤ, b = 2 * k) :=
by
  sorry

end factorable_iff_some_even_b_l99_99517


namespace cistern_empty_time_l99_99147

theorem cistern_empty_time
  (fill_time_without_leak : ℝ := 4)
  (additional_time_due_to_leak : ℝ := 2) :
  (1 / (fill_time_without_leak + additional_time_due_to_leak - fill_time_without_leak / fill_time_without_leak)) = 12 :=
by
  sorry

end cistern_empty_time_l99_99147


namespace symmetric_line_equation_l99_99542

theorem symmetric_line_equation :
  (∃ l : ℝ × ℝ × ℝ, (∀ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ + x₂ = -4 → y₁ + y₂ = 2 → 
    ∃ a b c : ℝ, l = (a, b, c) ∧ x₁ * a + y₁ * b + c = 0 ∧ x₂ * a + y₂ * b + c = 0) → 
  l = (2, -1, 5)) :=
sorry

end symmetric_line_equation_l99_99542


namespace part1_part2_l99_99958

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l99_99958


namespace fraction_is_5_div_9_l99_99308

-- Define the conditions t = f * (k - 32), t = 35, and k = 95
theorem fraction_is_5_div_9 {f k t : ℚ} (h1 : t = f * (k - 32)) (h2 : t = 35) (h3 : k = 95) : f = 5 / 9 :=
by
  sorry

end fraction_is_5_div_9_l99_99308


namespace incorrect_statement_about_GIS_l99_99853

def statement_A := "GIS can provide information for geographic decision-making"
def statement_B := "GIS are computer systems specifically designed to process geographic spatial data"
def statement_C := "Urban management is one of the earliest and most effective fields of GIS application"
def statement_D := "GIS's main functions include data collection, data analysis, decision-making applications, etc."

def correct_answer := statement_B

theorem incorrect_statement_about_GIS:
  correct_answer = statement_B := 
sorry

end incorrect_statement_about_GIS_l99_99853


namespace base_of_exponent_l99_99787

theorem base_of_exponent (x : ℤ) (m : ℕ) (h₁ : (-2 : ℤ)^(2 * m) = x^(12 - m)) (h₂ : m = 4) : x = -2 :=
by 
  sorry

end base_of_exponent_l99_99787


namespace find_value_of_expression_l99_99127

theorem find_value_of_expression (m n : ℝ) (h : |m - n - 5| + (2 * m + n - 4)^2 = 0) : 3 * m + n = 7 := 
sorry

end find_value_of_expression_l99_99127


namespace set_union_example_l99_99440

theorem set_union_example (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2, 3}) : M ∪ N = {1, 2, 3} := 
by
  sorry

end set_union_example_l99_99440


namespace find_n_l99_99102

theorem find_n (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 40)
  (h1 : b 1 = 70)
  (h2 : b n = 0)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → b (k + 1) = b (k - 1) - 2 / b k) :
  n = 1401 :=
sorry

end find_n_l99_99102


namespace train_speed_is_144_kmph_l99_99982

noncomputable def length_of_train : ℝ := 130 -- in meters
noncomputable def time_to_cross_pole : ℝ := 3.249740020798336 -- in seconds
noncomputable def speed_m_per_s : ℝ := length_of_train / time_to_cross_pole -- in m/s
noncomputable def conversion_factor : ℝ := 3.6 -- 1 m/s = 3.6 km/hr

theorem train_speed_is_144_kmph : speed_m_per_s * conversion_factor = 144 :=
by
  sorry

end train_speed_is_144_kmph_l99_99982


namespace intersection_A_B_l99_99089

def setA : Set ℝ := {x | 0 < x}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}
def intersectionAB : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = intersectionAB := by
  sorry

end intersection_A_B_l99_99089


namespace blake_spent_60_on_mangoes_l99_99674

def spent_on_oranges : ℕ := 40
def spent_on_apples : ℕ := 50
def initial_amount : ℕ := 300
def change : ℕ := 150
def total_spent := initial_amount - change
def total_spent_on_fruits := spent_on_oranges + spent_on_apples
def spending_on_mangoes := total_spent - total_spent_on_fruits

theorem blake_spent_60_on_mangoes : spending_on_mangoes = 60 := 
by
  -- The proof will go here
  sorry

end blake_spent_60_on_mangoes_l99_99674


namespace paint_coverage_l99_99751

theorem paint_coverage 
  (width height cost_per_quart money_spent area : ℕ)
  (cover : ℕ → ℕ → ℕ)
  (num_sides quarts_purchased : ℕ)
  (total_area num_quarts : ℕ)
  (sqfeet_per_quart : ℕ) :
  width = 5 
  → height = 4 
  → cost_per_quart = 2 
  → money_spent = 20 
  → num_sides = 2
  → cover width height = area
  → area * num_sides = total_area
  → money_spent / cost_per_quart = quarts_purchased
  → total_area / quarts_purchased = sqfeet_per_quart
  → total_area = 40 
  → quarts_purchased = 10 
  → sqfeet_per_quart = 4 :=
by 
  intros
  sorry

end paint_coverage_l99_99751


namespace roots_are_integers_l99_99601

theorem roots_are_integers (a b : ℤ) (h_discriminant : ∃ (q r : ℚ), r ≠ 0 ∧ a^2 - 4 * b = (q/r)^2) : 
  ∃ x y : ℤ, x^2 - a * x + b = 0 ∧ y^2 - a * y + b = 0 := 
sorry

end roots_are_integers_l99_99601


namespace opposite_of_neg_five_l99_99961

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l99_99961


namespace minimum_value_of_tan_sum_l99_99019

open Real

theorem minimum_value_of_tan_sum :
  ∀ {A B C : ℝ}, 
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π ∧ 
  2 * sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2 ->
  ( ∃ t : ℝ, ( t = 1 / tan A + 1 / tan B + 1 / tan C ) ∧ t = sqrt 13 / 2 ) := 
sorry

end minimum_value_of_tan_sum_l99_99019


namespace percentage_basketball_l99_99222

theorem percentage_basketball (total_students : ℕ) (chess_percentage : ℝ) (students_like_chess_basketball : ℕ) 
  (percentage_conversion : ∀ p : ℝ, 0 ≤ p → p / 100 = p) 
  (h_total : total_students = 250) 
  (h_chess : chess_percentage = 10) 
  (h_chess_basketball : students_like_chess_basketball = 125) :
  ∃ (basketball_percentage : ℝ), basketball_percentage = 40 := by
  sorry

end percentage_basketball_l99_99222


namespace correct_operation_l99_99204

variable (a : ℝ)

theorem correct_operation :
  (2 * a^2 * a = 2 * a^3) ∧
  ((a + 1)^2 ≠ a^2 + 1) ∧
  ((a^2 / (2 * a)) ≠ 2 * a) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) :=
by
  { sorry }

end correct_operation_l99_99204


namespace car_cost_difference_l99_99915

-- Definitions based on the problem's conditions
def car_cost_ratio (C A : ℝ) := C / A = 3 / 2
def ac_cost := 1500

-- Theorem statement that needs proving
theorem car_cost_difference (C A : ℝ) (h1 : car_cost_ratio C A) (h2 : A = ac_cost) : C - A = 750 := 
by sorry

end car_cost_difference_l99_99915


namespace quadratic_trinomial_neg_values_l99_99899

theorem quadratic_trinomial_neg_values (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 7 * x + 4 * a < 0) ↔ a < -7/4 := by
sorry

end quadratic_trinomial_neg_values_l99_99899


namespace cyclic_quadrilateral_AC_plus_BD_l99_99327

theorem cyclic_quadrilateral_AC_plus_BD (AB BC CD DA : ℝ) (AC BD : ℝ) (h1 : AB = 5) (h2 : BC = 10) (h3 : CD = 11) (h4 : DA = 14)
  (h5 : AC = Real.sqrt 221) (h6 : BD = 195 / Real.sqrt 221) :
  AC + BD = 416 / Real.sqrt (13 * 17) ∧ (AC = Real.sqrt 221 ∧ BD = 195 / Real.sqrt 221) →
  (AC + BD = 416 / Real.sqrt (13 * 17)) ∧ (AC + BD = 446) :=
by
  sorry

end cyclic_quadrilateral_AC_plus_BD_l99_99327


namespace sum_first_six_terms_geometric_sequence_l99_99381

theorem sum_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * ((1 - r^n) / (1 - r))
  S_n = 455 / 1365 := by
  sorry

end sum_first_six_terms_geometric_sequence_l99_99381


namespace rectangular_coords_of_neg_theta_l99_99023

theorem rectangular_coords_of_neg_theta 
  (x y z : ℝ) 
  (rho theta phi : ℝ)
  (hx : x = 8)
  (hy : y = 6)
  (hz : z = -3)
  (h_rho : rho = Real.sqrt (x^2 + y^2 + z^2))
  (h_cos_phi : Real.cos phi = z / rho)
  (h_sin_phi : Real.sin phi = Real.sqrt (1 - (Real.cos phi)^2))
  (h_tan_theta : Real.tan theta = y / x) :
  (rho * Real.sin phi * Real.cos (-theta), rho * Real.sin phi * Real.sin (-theta), rho * Real.cos phi) = (8, -6, -3) := 
  sorry

end rectangular_coords_of_neg_theta_l99_99023


namespace find_number_l99_99328

theorem find_number (x : ℝ) : 
  10 * ((2 * (x * x + 2) + 3) / 5) = 50 → x = 3 := 
by
  sorry

end find_number_l99_99328


namespace no_non_similar_triangles_with_geometric_angles_l99_99741

theorem no_non_similar_triangles_with_geometric_angles :
  ¬ ∃ (a r : ℤ), 0 < a ∧ 0 < r ∧ a ≠ ar ∧ a ≠ ar^2 ∧ ar ≠ ar^2 ∧
  a + ar + ar^2 = 180 :=
sorry

end no_non_similar_triangles_with_geometric_angles_l99_99741


namespace second_gym_signup_fee_covers_4_months_l99_99724

-- Define constants
def cheap_gym_monthly_fee : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def total_spent_first_year : ℕ := 650

-- Define the monthly fee of the second gym
def second_gym_monthly_fee : ℕ := 3 * cheap_gym_monthly_fee

-- Calculate the amount spent on the second gym
def spent_on_second_gym : ℕ := total_spent_first_year - (12 * cheap_gym_monthly_fee + cheap_gym_signup_fee)

-- Define the number of months the sign-up fee covers
def months_covered_by_signup_fee : ℕ := spent_on_second_gym / second_gym_monthly_fee

theorem second_gym_signup_fee_covers_4_months :
  months_covered_by_signup_fee = 4 :=
by
  sorry

end second_gym_signup_fee_covers_4_months_l99_99724


namespace total_cantaloupes_l99_99407

def Fred_grew_38 : ℕ := 38
def Tim_grew_44 : ℕ := 44

theorem total_cantaloupes : Fred_grew_38 + Tim_grew_44 = 82 := by
  sorry

end total_cantaloupes_l99_99407


namespace twenty_five_percent_less_than_80_one_fourth_more_l99_99435

theorem twenty_five_percent_less_than_80_one_fourth_more (n : ℕ) (h : (5 / 4 : ℝ) * n = 60) : n = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_one_fourth_more_l99_99435


namespace algebraic_expression_value_l99_99543

theorem algebraic_expression_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : m ^ 2 = 25) :
  m^2 - 100*a - 99*b - b*c*d + |c*d - 2| = -74 :=
by
  sorry

end algebraic_expression_value_l99_99543


namespace inequality_proof_l99_99458

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) : (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

end inequality_proof_l99_99458


namespace coeff_of_x_square_l99_99271

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem coeff_of_x_square :
  (binom 8 3 = 56) ∧ (8 - 2 * 3 = 2) :=
sorry

end coeff_of_x_square_l99_99271


namespace billy_distance_l99_99115

-- Definitions
def distance_billy_spit (b : ℝ) : ℝ := b
def distance_madison_spit (m : ℝ) (b : ℝ) : Prop := m = 1.20 * b
def distance_ryan_spit (r : ℝ) (m : ℝ) : Prop := r = 0.50 * m

-- Conditions
variables (m : ℝ) (b : ℝ) (r : ℝ)
axiom madison_farther: distance_madison_spit m b
axiom ryan_shorter: distance_ryan_spit r m
axiom ryan_distance: r = 18

-- Proof problem
theorem billy_distance : b = 30 := by
  sorry

end billy_distance_l99_99115


namespace sophie_saves_money_l99_99973

variable (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ)
variable (given_on_birthday : Bool)

noncomputable def money_saved_per_year (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ) : ℝ :=
  (loads_per_week * dryer_sheets_per_load * weeks_per_year / sheets_per_box) * cost_per_box

theorem sophie_saves_money (h_loads_per_week : loads_per_week = 4) (h_dryer_sheets_per_load : dryer_sheets_per_load = 1)
                           (h_weeks_per_year : weeks_per_year = 52) (h_cost_per_box : cost_per_box = 5.50)
                           (h_sheets_per_box : sheets_per_box = 104) (h_given_on_birthday : given_on_birthday = true) :
  money_saved_per_year 4 1 52 5.50 104 = 11 :=
by
  have h1 : loads_per_week = 4 := h_loads_per_week
  have h2 : dryer_sheets_per_load = 1 := h_dryer_sheets_per_load
  have h3 : weeks_per_year = 52 := h_weeks_per_year
  have h4 : cost_per_box = 5.50 := h_cost_per_box
  have h5 : sheets_per_box = 104 := h_sheets_per_box
  have h6 : given_on_birthday = true := h_given_on_birthday
  sorry

end sophie_saves_money_l99_99973


namespace perpendicular_tangents_add_l99_99806

open Real

noncomputable def f1 (x : ℝ): ℝ := x^2 - 2 * x + 2
noncomputable def f2 (x : ℝ) (a : ℝ) (b : ℝ): ℝ := -x^2 + a * x + b

-- Definitions of derivatives for the given functions
noncomputable def f1' (x : ℝ): ℝ := 2 * x - 2
noncomputable def f2' (x : ℝ) (a : ℝ): ℝ := -2 * x + a

theorem perpendicular_tangents_add (x0 y0 a b : ℝ)
  (h1 : y0 = f1 x0)
  (h2 : y0 = f2 x0 a b)
  (h3 : f1' x0 * f2' x0 a = -1) :
  a + b = 5 / 2 := sorry

end perpendicular_tangents_add_l99_99806


namespace leila_spending_l99_99774

theorem leila_spending (sweater jewelry total money_left : ℕ) (h1 : sweater = 40) (h2 : sweater * 4 = total) (h3 : money_left = 20) (h4 : total - sweater - jewelry = money_left) : jewelry - sweater = 60 :=
by
  sorry

end leila_spending_l99_99774


namespace B_subscription_difference_l99_99414

noncomputable def subscription_difference (A B C P : ℕ) (delta : ℕ) (comb_sub: A + B + C = 50000) (c_profit: 8400 = 35000 * C / 50000) :=
  B - C

theorem B_subscription_difference (A B C : ℕ) (z: ℕ) 
  (h1 : A + B + C = 50000) 
  (h2 : A = B + 4000) 
  (h3 : (B - C) = z)
  (h4 :  8400 = 35000 * C / 50000):
  B - C = 10000 :=
by {
  sorry
}

end B_subscription_difference_l99_99414


namespace solve_equation_l99_99703

variable (x : ℝ)

def equation := (x / (2 * x - 3)) + (5 / (3 - 2 * x)) = 4
def condition := x ≠ 3 / 2

theorem solve_equation : equation x ∧ condition x → x = 1 :=
by
  sorry

end solve_equation_l99_99703


namespace balance_proof_l99_99876

variables (a b c : ℝ)

theorem balance_proof (h1 : 4 * a + 2 * b = 12 * c) (h2 : 2 * a = b + 3 * c) : 3 * b = 4.5 * c :=
sorry

end balance_proof_l99_99876


namespace ratio_of_areas_of_triangle_and_trapezoid_l99_99105

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

theorem ratio_of_areas_of_triangle_and_trapezoid :
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  (a_small / a_trapezoid) = (1 / 3) :=
by
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  have h : (a_small / a_trapezoid) = (1 / 3) := 
    by sorry  -- Here would be the proof steps, but we're skipping
  exact h

end ratio_of_areas_of_triangle_and_trapezoid_l99_99105


namespace f_plus_g_eq_l99_99892

variables {R : Type*} [CommRing R]

-- Define the odd function f
def f (x : R) : R := sorry

-- Define the even function g
def g (x : R) : R := sorry

-- Define that f is odd and g is even
axiom f_odd (x : R) : f (-x) = -f x
axiom g_even (x : R) : g (-x) = g x

-- Define the given equation
axiom f_minus_g_eq (x : R) : f x - g x = x ^ 2 + 9 * x + 12

-- Statement of the goal
theorem f_plus_g_eq (x : R) : f x + g x = -x ^ 2 + 9 * x - 12 := by
  sorry

end f_plus_g_eq_l99_99892


namespace sandy_took_310_dollars_l99_99661

theorem sandy_took_310_dollars (X : ℝ) (h70percent : 0.70 * X = 217) : X = 310 := by
  sorry

end sandy_took_310_dollars_l99_99661


namespace alex_blueberry_pies_l99_99974

-- Definitions based on given conditions:
def total_pies : ℕ := 30
def ratio (a b c : ℕ) : Prop := (a : ℚ) / b = 2 / 3 ∧ (b : ℚ) / c = 3 / 5

-- Statement to prove the number of blueberry pies
theorem alex_blueberry_pies :
  ∃ (a b c : ℕ), ratio a b c ∧ a + b + c = total_pies ∧ b = 9 :=
by
  sorry

end alex_blueberry_pies_l99_99974


namespace dice_probability_five_or_six_l99_99420

theorem dice_probability_five_or_six :
  let outcomes := 36
  let favorable := 18
  let probability := favorable / outcomes
  probability = 1 / 2 :=
by
  sorry

end dice_probability_five_or_six_l99_99420


namespace sin_half_alpha_l99_99591

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l99_99591


namespace allocate_25_rubles_in_4_weighings_l99_99575

theorem allocate_25_rubles_in_4_weighings :
  ∃ (coins : ℕ) (coins5 : ℕ → ℕ), 
    (coins = 1600) ∧ 
    (coins5 0 = 800 ∧ coins5 1 = 800) ∧
    (coins5 2 = 400 ∧ coins5 3 = 400) ∧
    (coins5 4 = 200 ∧ coins5 5 = 200) ∧
    (coins5 6 = 100 ∧ coins5 7 = 100) ∧
    (
      25 = 20 + 5 ∧ 
      (∃ i j k l m n, coins5 i = 400 ∧ coins5 j = 400 ∧ coins5 k = 200 ∧
        coins5 l = 200 ∧ coins5 m = 100 ∧ coins5 n = 100)
    )
  := 
sorry

end allocate_25_rubles_in_4_weighings_l99_99575


namespace factorization_ce_sum_eq_25_l99_99654

theorem factorization_ce_sum_eq_25 {C E : ℤ} (h : (C * x - 13) * (E * x - 7) = 20 * x^2 - 87 * x + 91) : 
  C * E + C = 25 :=
sorry

end factorization_ce_sum_eq_25_l99_99654


namespace ring_toss_total_l99_99887

theorem ring_toss_total (money_per_day : ℕ) (days : ℕ) (total_money : ℕ) 
(h1 : money_per_day = 140) (h2 : days = 3) : total_money = 420 :=
by
  sorry

end ring_toss_total_l99_99887


namespace price_reduction_l99_99951

theorem price_reduction (x : ℝ) (h : 560 * (1 - x) * (1 - x) = 315) : 
  560 * (1 - x)^2 = 315 := 
by
  sorry

end price_reduction_l99_99951


namespace find_a_value_l99_99770

theorem find_a_value (a : ℝ) (A B : Set ℝ) (hA : A = {3, 5}) (hB : B = {x | a * x - 1 = 0}) :
  B ⊆ A → a = 0 ∨ a = 1/3 ∨ a = 1/5 :=
by sorry

end find_a_value_l99_99770


namespace percentage_increase_in_radius_l99_99257

theorem percentage_increase_in_radius (r R : ℝ) (h : π * R^2 = π * r^2 + 1.25 * (π * r^2)) :
  R = 1.5 * r :=
by
  -- Proof goes here
  sorry

end percentage_increase_in_radius_l99_99257


namespace is_isosceles_of_x_eq_one_root_is_right_angled_of_equal_roots_l99_99688

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Given that a, b, c are the sides of the triangle
axiom lengths_of_triangle : a > 0 ∧ b > 0 ∧ c > 0

-- Problem 1: Prove that triangle is isosceles if x=1 is a root
theorem is_isosceles_of_x_eq_one_root  : ((a - c) * (1:ℝ)^2 - 2 * b * (1:ℝ) + (a + c) = 0) → a = b ∧ a ≠ c := 
by
  intros h
  sorry

-- Problem 2: Prove that triangle is right-angled if the equation has two equal real roots
theorem is_right_angled_of_equal_roots : (b^2 = a^2 - c^2) → (a^2 = b^2 + c^2) := 
by 
  intros h
  sorry

end is_isosceles_of_x_eq_one_root_is_right_angled_of_equal_roots_l99_99688


namespace line_intercepts_l99_99495

theorem line_intercepts (x y : ℝ) (P : ℝ × ℝ) (h1 : P = (1, 4)) (h2 : ∃ k : ℝ, (x + y = k ∨ 4 * x - y = 0) ∧ 
  ∃ intercepts_p : ℝ × ℝ, intercepts_p = (k / 2, k / 2)) :
  ∃ k : ℝ, (x + y - k = 0 ∧ k = 5) ∨ (4 * x - y = 0) :=
sorry

end line_intercepts_l99_99495


namespace maggie_goldfish_fraction_l99_99220

theorem maggie_goldfish_fraction :
  ∀ (x : ℕ), 3*x / 5 + 20 = x → (x / 100 : ℚ) = 1 / 2 :=
by
  sorry

end maggie_goldfish_fraction_l99_99220


namespace number_of_ways_to_sum_to_4_l99_99937

-- Definitions deriving from conditions
def cards : List ℕ := [0, 1, 2, 3, 4]

-- Goal to prove
theorem number_of_ways_to_sum_to_4 : 
  let pairs := List.product cards cards
  let valid_pairs := pairs.filter (λ (x, y) => x + y = 4)
  List.length valid_pairs = 5 := 
by
  sorry

end number_of_ways_to_sum_to_4_l99_99937


namespace length_of_first_leg_of_triangle_l99_99380

theorem length_of_first_leg_of_triangle 
  (a b c : ℝ) 
  (h1 : b = 8) 
  (h2 : c = 10) 
  (h3 : c^2 = a^2 + b^2) : 
  a = 6 :=
by
  sorry

end length_of_first_leg_of_triangle_l99_99380


namespace probability_of_same_color_balls_l99_99211

-- Definitions of the problem
def total_balls_bag_A := 8 + 4
def total_balls_bag_B := 6 + 6
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

def P (event: Nat -> Bool) (total: Nat) : Nat :=
  let favorable := (List.range total).filter event |>.length
  favorable / total

-- Probability of drawing a white ball from bag A
def P_A := P (λ n => n < white_balls_bag_A) total_balls_bag_A

-- Probability of drawing a red ball from bag A
def P_not_A := P (λ n => n >= white_balls_bag_A && n < total_balls_bag_A) total_balls_bag_A

-- Probability of drawing a white ball from bag B
def P_B := P (λ n => n < white_balls_bag_B) total_balls_bag_B

-- Probability of drawing a red ball from bag B
def P_not_B := P (λ n => n >= white_balls_bag_B && n < total_balls_bag_B) total_balls_bag_B

-- Independence assumption (product rule for independent events)
noncomputable def P_same_color := P_A * P_B + P_not_A * P_not_B

-- Final theorem to prove
theorem probability_of_same_color_balls :
  P_same_color = 1 / 2 := by
    sorry

end probability_of_same_color_balls_l99_99211


namespace correlation_implies_slope_positive_l99_99416

-- Definition of the regression line
def regression_line (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Given conditions
variables (x y : ℝ)
variables (b a r : ℝ)

-- The statement of the proof problem
theorem correlation_implies_slope_positive (h1 : r > 0) (h2 : regression_line x y b a) : b > 0 :=
sorry

end correlation_implies_slope_positive_l99_99416


namespace find_xz_over_y_squared_l99_99025

variable {x y z : ℝ}

noncomputable def k : ℝ := 7

theorem find_xz_over_y_squared
    (h1 : x + k * y + 4 * z = 0)
    (h2 : 4 * x + k * y - 3 * z = 0)
    (h3 : x + 3 * y - 2 * z = 0)
    (h_nz : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
    (x * z) / (y ^ 2) = 26 / 9 :=
by sorry

end find_xz_over_y_squared_l99_99025


namespace spinner_probabilities_l99_99297

noncomputable def prob_A : ℚ := 1 / 3
noncomputable def prob_B : ℚ := 1 / 4
noncomputable def prob_C : ℚ := 5 / 18
noncomputable def prob_D : ℚ := 5 / 36

theorem spinner_probabilities :
  prob_A + prob_B + prob_C + prob_D = 1 ∧
  prob_C = 2 * prob_D :=
by {
  -- The statement of the theorem matches the given conditions and the correct answers.
  -- Proof will be provided later.
  sorry
}

end spinner_probabilities_l99_99297


namespace cookies_in_second_type_l99_99588

theorem cookies_in_second_type (x : ℕ) (h1 : 50 * 12 + 80 * x + 70 * 16 = 3320) : x = 20 :=
by sorry

end cookies_in_second_type_l99_99588


namespace pythagorean_relationship_l99_99745

theorem pythagorean_relationship (a b c : ℝ) (h : c^2 = a^2 + b^2) : c^2 = a^2 + b^2 :=
by
  sorry

end pythagorean_relationship_l99_99745


namespace sum_of_coefficients_eq_minus_36_l99_99682

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem sum_of_coefficients_eq_minus_36 
  (a b c : ℝ)
  (h_min : ∀ x, quadratic a b c x ≥ -36)
  (h_points : quadratic a b c (-3) = 0 ∧ quadratic a b c 5 = 0)
  : a + b + c = -36 :=
sorry

end sum_of_coefficients_eq_minus_36_l99_99682


namespace zoo_gorillas_sent_6_l99_99483

theorem zoo_gorillas_sent_6 (G : ℕ) : 
  let initial_animals := 68
  let after_sending_gorillas := initial_animals - G
  let after_adopting_hippopotamus := after_sending_gorillas + 1
  let after_taking_rhinos := after_adopting_hippopotamus + 3
  let after_birth_lion_cubs := after_taking_rhinos + 8
  let after_adding_meerkats := after_birth_lion_cubs + (2 * 8)
  let final_animals := 90
  after_adding_meerkats = final_animals → G = 6 := 
by
  intros
  let initial_animals := 68
  let after_sending_gorillas := initial_animals - G
  let after_adopting_hippopotamus := after_sending_gorillas + 1
  let after_taking_rhinos := after_adopting_hippopotamus + 3
  let after_birth_lion_cubs := after_taking_rhinos + 8
  let after_adding_meerkats := after_birth_lion_cubs + (2 * 8)
  let final_animals := 90
  sorry

end zoo_gorillas_sent_6_l99_99483


namespace weeks_to_fill_moneybox_l99_99560

-- Monica saves $15 every week
def savings_per_week : ℕ := 15

-- Number of cycles Monica repeats
def cycles : ℕ := 5

-- Total amount taken to the bank
def total_savings : ℕ := 4500

-- Prove that the number of weeks it takes for the moneybox to get full is 60
theorem weeks_to_fill_moneybox : ∃ W : ℕ, (cycles * savings_per_week * W = total_savings) ∧ W = 60 := 
by 
  sorry

end weeks_to_fill_moneybox_l99_99560


namespace rachel_age_is_24_5_l99_99610

/-- Rachel is 4 years older than Leah -/
def rachel_age_eq_leah_plus_4 (R L : ℝ) : Prop := R = L + 4

/-- Together, Rachel and Leah are twice as old as Sam -/
def rachel_and_leah_eq_twice_sam (R L S : ℝ) : Prop := R + L = 2 * S

/-- Alex is twice as old as Rachel -/
def alex_eq_twice_rachel (A R : ℝ) : Prop := A = 2 * R

/-- The sum of all four friends' ages is 92 -/
def sum_ages_eq_92 (R L S A : ℝ) : Prop := R + L + S + A = 92

theorem rachel_age_is_24_5 (R L S A : ℝ) :
  rachel_age_eq_leah_plus_4 R L →
  rachel_and_leah_eq_twice_sam R L S →
  alex_eq_twice_rachel A R →
  sum_ages_eq_92 R L S A →
  R = 24.5 := 
by 
  sorry

end rachel_age_is_24_5_l99_99610


namespace geometric_sequence_common_ratio_l99_99984

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℤ) 
  (q : ℤ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 1 / 4) : 
  q = 1 / 2 :=
  sorry

end geometric_sequence_common_ratio_l99_99984


namespace employee_payments_l99_99666

theorem employee_payments :
  ∃ (A B C : ℤ), A = 900 ∧ B = 600 ∧ C = 500 ∧
    A + B + C = 2000 ∧
    A = 3 * B / 2 ∧
    C = 400 + 100 := 
by
  sorry

end employee_payments_l99_99666


namespace white_roses_per_table_decoration_l99_99142

theorem white_roses_per_table_decoration (x : ℕ) :
  let bouquets := 5
  let table_decorations := 7
  let roses_per_bouquet := 5
  let total_roses := 109
  5 * roses_per_bouquet + 7 * x = total_roses → x = 12 :=
by
  intros
  sorry

end white_roses_per_table_decoration_l99_99142


namespace increase_by_percentage_proof_l99_99194

def initial_number : ℕ := 150
def percentage_increase : ℝ := 0.4
def final_number : ℕ := 210

theorem increase_by_percentage_proof :
  initial_number + (percentage_increase * initial_number) = final_number :=
by
  sorry

end increase_by_percentage_proof_l99_99194


namespace intersection_is_isosceles_right_angled_l99_99926

def is_isosceles_triangle (x : Type) : Prop := sorry -- Definition of isosceles triangle
def is_right_angled_triangle (x : Type) : Prop := sorry -- Definition of right-angled triangle

def M : Set Type := {x | is_isosceles_triangle x}
def N : Set Type := {x | is_right_angled_triangle x}

theorem intersection_is_isosceles_right_angled :
  (M ∩ N) = {x | is_isosceles_triangle x ∧ is_right_angled_triangle x} := by
  sorry

end intersection_is_isosceles_right_angled_l99_99926


namespace placement_proof_l99_99309

def claimed_first_place (p: String) : Prop := 
  p = "Olya" ∨ p = "Oleg" ∨ p = "Pasha"

def odd_places_boys (positions: ℕ → String) : Prop := 
  (positions 1 = "Oleg" ∨ positions 1 = "Pasha") ∧ (positions 3 = "Oleg" ∨ positions 3 = "Pasha")

def olya_wrong (positions : ℕ → String) : Prop := 
  ¬odd_places_boys positions

def always_truthful_or_lying (Olya_st: Prop) (Oleg_st: Prop) (Pasha_st: Prop) : Prop := 
  Olya_st = Oleg_st ∧ Oleg_st = Pasha_st

def competition_placement : Prop :=
  ∃ (positions: ℕ → String),
    claimed_first_place (positions 1) ∧
    claimed_first_place (positions 2) ∧
    claimed_first_place (positions 3) ∧
    (positions 1 = "Oleg") ∧
    (positions 2 = "Pasha") ∧
    (positions 3 = "Olya") ∧
    olya_wrong positions ∧
    always_truthful_or_lying
      ((claimed_first_place "Olya" ∧ odd_places_boys positions))
      ((claimed_first_place "Oleg" ∧ olya_wrong positions))
      (claimed_first_place "Pasha")

theorem placement_proof : competition_placement :=
  sorry

end placement_proof_l99_99309


namespace find_weights_l99_99792

def item_weights (a b c d e f g h : ℕ) : Prop :=
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧ 1 ≤ e ∧ 1 ≤ f ∧ 1 ≤ g ∧ 1 ≤ h ∧
  a > b ∧ b > c ∧ c > d ∧ d > e ∧ e > f ∧ f > g ∧ g > h ∧
  a ≤ 15 ∧ b ≤ 15 ∧ c ≤ 15 ∧ d ≤ 15 ∧ e ≤ 15 ∧ f ≤ 15 ∧ g ≤ 15 ∧ h ≤ 15

theorem find_weights (a b c d e f g h : ℕ) (hw : item_weights a b c d e f g h) 
    (h1 : d + e + f + g > a + b + c + h) 
    (h2 : e + f > d + g) 
    (h3 : e > f) : e = 11 ∧ g = 5 := sorry

end find_weights_l99_99792


namespace find_number_l99_99296

-- Define the condition given in the problem
def condition (x : ℕ) : Prop :=
  x / 5 + 6 = 65

-- Prove that the solution satisfies the condition
theorem find_number : ∃ x : ℕ, condition x ∧ x = 295 :=
by
  -- Skip the actual proof steps
  sorry

end find_number_l99_99296


namespace average_pages_per_book_l99_99178

theorem average_pages_per_book :
  let pages := [120, 150, 180, 210, 240]
  let num_books := 5
  let total_pages := pages.sum
  total_pages / num_books = 180 := by
  sorry

end average_pages_per_book_l99_99178


namespace part_I_part_II_l99_99783

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + 4 * a * x - 3

-- Part (I)
theorem part_I (a : ℝ) (h_a : a > 0) (h_roots: ∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : 
  0 < a ∧ a < 2 / 5 :=
sorry

-- Part (II)
theorem part_II (a : ℝ) (h_max : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f a x ≤ f a 2) : 
  a ≥ -1 / 3 :=
sorry

end part_I_part_II_l99_99783


namespace fg_of_3_eq_79_l99_99051

def g (x : ℤ) : ℤ := x ^ 3
def f (x : ℤ) : ℤ := 3 * x - 2

theorem fg_of_3_eq_79 : f (g 3) = 79 := by
  sorry

end fg_of_3_eq_79_l99_99051


namespace rod_cut_l99_99262

theorem rod_cut (x : ℕ) (h : 3 * x + 5 * x + 7 * x = 120) : 3 * x = 24 :=
by
  sorry

end rod_cut_l99_99262


namespace measure_of_two_equal_angles_l99_99448

noncomputable def measure_of_obtuse_angle (θ : ℝ) : ℝ := θ + (0.6 * θ)

-- Given conditions
def is_obtuse_isosceles_triangle (θ : ℝ) : Prop :=
  θ = 90 ∧ measure_of_obtuse_angle 90 = 144 ∧ 180 - 144 = 36

-- The main theorem
theorem measure_of_two_equal_angles :
  ∀ θ, is_obtuse_isosceles_triangle θ → 36 / 2 = 18 :=
by
  intros θ h
  sorry

end measure_of_two_equal_angles_l99_99448


namespace wendy_distance_difference_l99_99164

-- Defining the distances ran and walked by Wendy
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- The theorem to prove the difference in distance
theorem wendy_distance_difference : distance_ran - distance_walked = 10.66 := by
  -- Proof goes here
  sorry

end wendy_distance_difference_l99_99164


namespace find_k_l99_99697

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

theorem find_k (k : ℝ) :
  let sum_vector := (vector_a.1 + 2 * (vector_b k).1, vector_a.2 + 2 * (vector_b k).2)
  let diff_vector := (2 * vector_a.1 - (vector_b k).1, 2 * vector_a.2 - (vector_b k).2)
  sum_vector.1 * diff_vector.2 = sum_vector.2 * diff_vector.1
  → k = 6 :=
by
  sorry

end find_k_l99_99697


namespace shirts_sewn_on_tuesday_l99_99823

theorem shirts_sewn_on_tuesday 
  (shirts_monday : ℕ) 
  (shirts_wednesday : ℕ) 
  (total_buttons : ℕ) 
  (buttons_per_shirt : ℕ) 
  (shirts_tuesday : ℕ) 
  (h1: shirts_monday = 4) 
  (h2: shirts_wednesday = 2) 
  (h3: total_buttons = 45) 
  (h4: buttons_per_shirt = 5) 
  (h5: shirts_tuesday * buttons_per_shirt + shirts_monday * buttons_per_shirt + shirts_wednesday * buttons_per_shirt = total_buttons) : 
  shirts_tuesday = 3 :=
by 
  sorry

end shirts_sewn_on_tuesday_l99_99823


namespace y_plus_z_value_l99_99442

theorem y_plus_z_value (v w x y z S : ℕ) 
  (h1 : 196 + x + y = S)
  (h2 : 269 + z + 123 = S)
  (h3 : 50 + x + z = S) : 
  y + z = 196 := 
sorry

end y_plus_z_value_l99_99442


namespace closest_integer_to_cube_root_of_150_l99_99861

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l99_99861


namespace find_m_l99_99546

theorem find_m (m : ℝ) :
  (∃ m : ℝ, ∀ x y : ℝ, x + y - m = 0 ∧ x + (3 - 2 * m) * y = 0 → 
     (m = 1)) := 
sorry

end find_m_l99_99546


namespace quadrant_of_tan_and_cos_l99_99179

theorem quadrant_of_tan_and_cos (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  ∃ Q, (Q = 2) :=
by
  sorry


end quadrant_of_tan_and_cos_l99_99179


namespace factor_sum_l99_99129

theorem factor_sum : 
  (∃ d e, x^2 + 9 * x + 20 = (x + d) * (x + e)) ∧ 
  (∃ e f, x^2 - x - 56 = (x + e) * (x - f)) → 
  ∃ d e f, d + e + f = 19 :=
by
  sorry

end factor_sum_l99_99129


namespace sum_divisible_by_3_l99_99805

theorem sum_divisible_by_3 (a : ℤ) : 3 ∣ (a^3 + 2 * a) :=
sorry

end sum_divisible_by_3_l99_99805


namespace missing_dimension_of_carton_l99_99506

theorem missing_dimension_of_carton (x : ℕ) 
  (h1 : 0 < x)
  (h2 : 0 < 48)
  (h3 : 0 < 60)
  (h4 : 0 < 8)
  (h5 : 0 < 6)
  (h6 : 0 < 5)
  (h7 : (x * 48 * 60) / (8 * 6 * 5) = 300) : 
  x = 25 :=
by
  sorry

end missing_dimension_of_carton_l99_99506


namespace max_area_of_sector_l99_99223

variable (r l S : ℝ)

theorem max_area_of_sector (h_circumference : 2 * r + l = 8) (h_area : S = (1 / 2) * l * r) : 
  S ≤ 4 :=
sorry

end max_area_of_sector_l99_99223


namespace find_m_range_l99_99197

noncomputable def p (m : ℝ) : Prop :=
  m < 1 / 3

noncomputable def q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem find_m_range (m : ℝ) :
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (1 / 3 ≤ m ∧ m < 15) :=
by
  sorry

end find_m_range_l99_99197


namespace find_x_l99_99015

theorem find_x (x : ℝ) (h : (20 + 30 + 40 + x) / 4 = 35) : x = 50 := by
  sorry

end find_x_l99_99015


namespace find_X_l99_99698

def star (a b : ℤ) : ℤ := 5 * a - 3 * b

theorem find_X (X : ℤ) (h1 : star X (star 3 2) = 18) : X = 9 :=
by
  sorry

end find_X_l99_99698


namespace find_interior_angles_l99_99444

theorem find_interior_angles (A B C : ℝ) (h1 : B = A + 10) (h2 : C = B + 10) (h3 : A + B + C = 180) : 
  A = 50 ∧ B = 60 ∧ C = 70 := by
  sorry

end find_interior_angles_l99_99444


namespace train_probability_correct_l99_99996

/-- Define the necessary parameters and conditions --/
noncomputable def train_arrival_prob (train_start train_wait max_time_Alex max_time_train : ℝ) : ℝ :=
  let total_possible_area := max_time_Alex * max_time_train
  let overlap_area := (max_time_train - train_wait) * train_wait + (train_wait) * max_time_train / 2
  overlap_area / total_possible_area

/-- Main theorem stating that the probability is 3/10 --/
theorem train_probability_correct :
  train_arrival_prob 0 15 75 60 = 3 / 10 :=
by sorry

end train_probability_correct_l99_99996


namespace sequence_remainder_zero_l99_99146

theorem sequence_remainder_zero :
  let a := 3
  let d := 8
  let n := 32
  let aₙ := a + (n - 1) * d
  let Sₙ := n * (a + aₙ) / 2
  aₙ = 251 → Sₙ % 8 = 0 :=
by
  intros
  sorry

end sequence_remainder_zero_l99_99146


namespace problem_solution_l99_99104

theorem problem_solution (a b c d x : ℚ) 
  (h1 : 2 * a + 2 = x) 
  (h2 : 3 * b + 3 = x) 
  (h3 : 4 * c + 4 = x) 
  (h4 : 5 * d + 5 = x) 
  (h5 : 2 * a + 3 * b + 4 * c + 5 * d + 6 = x) 
  : 2 * a + 3 * b + 4 * c + 5 * d = -10 / 3 := 
by 
  sorry

end problem_solution_l99_99104


namespace janice_items_l99_99468

theorem janice_items : 
  ∃ a b c : ℕ, 
    a + b + c = 60 ∧ 
    15 * a + 400 * b + 500 * c = 6000 ∧ 
    a = 50 := 
by 
  sorry

end janice_items_l99_99468


namespace planted_area_ratio_l99_99333

noncomputable def ratio_of_planted_area_to_total_area : ℚ := 145 / 147

theorem planted_area_ratio (h : ∃ (S : ℚ), 
  (∃ (x y : ℚ), x * x + y * y ≤ S * S) ∧
  (∃ (a b : ℚ), 3 * a + 4 * b = 12 ∧ (3 * x + 4 * y - 12) / 5 = 2)) :
  ratio_of_planted_area_to_total_area = 145 / 147 :=
sorry

end planted_area_ratio_l99_99333


namespace height_is_geometric_mean_of_bases_l99_99227

-- Given conditions
variables (a c m : ℝ)
-- we declare the condition that the given trapezoid is symmetric and tangential
variables (isSymmetricTangentialTrapezoid : Prop)

-- The theorem to be proven
theorem height_is_geometric_mean_of_bases 
(isSymmetricTangentialTrapezoid: isSymmetricTangentialTrapezoid) 
: m = Real.sqrt (a * c) :=
sorry

end height_is_geometric_mean_of_bases_l99_99227


namespace length_of_arc_l99_99540

theorem length_of_arc (angle_SIT : ℝ) (radius_OS : ℝ) (h1 : angle_SIT = 45) (h2 : radius_OS = 15) :
  arc_length_SIT = 7.5 * Real.pi :=
by
  sorry

end length_of_arc_l99_99540


namespace max_total_weight_l99_99621

-- Definitions
def A_max_weight := 5
def E_max_weight := 2 * A_max_weight
def total_swallows := 90
def A_to_E_ratio := 2

-- Main theorem statement
theorem max_total_weight :
  ∃ A E, (A = A_to_E_ratio * E) ∧ (A + E = total_swallows) ∧ ((A * A_max_weight + E * E_max_weight) = 600) :=
  sorry

end max_total_weight_l99_99621


namespace time_to_fill_pool_l99_99831

def LindasPoolCapacity : ℕ := 30000
def CurrentVolume : ℕ := 6000
def NumberOfHoses : ℕ := 6
def RatePerHosePerMinute : ℕ := 3
def GallonsNeeded : ℕ := LindasPoolCapacity - CurrentVolume
def RatePerHosePerHour : ℕ := RatePerHosePerMinute * 60
def TotalHourlyRate : ℕ := NumberOfHoses * RatePerHosePerHour

theorem time_to_fill_pool : (GallonsNeeded / TotalHourlyRate) = 22 :=
by
  sorry

end time_to_fill_pool_l99_99831


namespace sin_double_angle_identity_l99_99843

theorem sin_double_angle_identity 
  (α : ℝ) 
  (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h₂ : Real.sin α = 1 / 5) : 
  Real.sin (2 * α) = - (4 * Real.sqrt 6) / 25 :=
by
  sorry

end sin_double_angle_identity_l99_99843


namespace part_a_part_b_l99_99769

-- Part (a)
theorem part_a {x y n : ℕ} (h : x^3 + 2^n * y = y^3 + 2^n * x) : x = y :=
sorry

-- Part (b)
theorem part_b {x y : ℤ} {n : ℕ} (h : x ≠ 0 ∧ y ≠ 0 ∧ x^3 + 2^n * y = y^3 + 2^n * x) : |x| = |y| :=
sorry

end part_a_part_b_l99_99769


namespace correct_proposition_is_D_l99_99193

-- Define the propositions
def propositionA : Prop :=
  (∀ x : ℝ, x^2 = 4 → x = 2 ∨ x = -2) → (∀ x : ℝ, (x ≠ 2 ∨ x ≠ -2) → x^2 ≠ 4)

def propositionB (p : Prop) : Prop :=
  (p → (∀ x : ℝ, x^2 - 2*x + 3 > 0)) → (¬p → (∃ x : ℝ, x^2 - 2*x + 3 < 0))

def propositionC : Prop :=
  ∀ (a b : ℝ) (n : ℕ), a > b → n > 0 → a^n > b^n

def p : Prop := ∀ x : ℝ, x^3 ≥ 0
def q : Prop := ∀ e : ℝ, e > 0 → e < 1
def propositionD := p ∧ q

-- The proof problem
theorem correct_proposition_is_D : propositionD :=
  sorry

end correct_proposition_is_D_l99_99193


namespace remainder_division_l99_99235

theorem remainder_division (G Q1 R1 Q2 : ℕ) (hG : G = 88)
  (h1 : 3815 = G * Q1 + R1) (h2 : 4521 = G * Q2 + 33) : R1 = 31 :=
sorry

end remainder_division_l99_99235


namespace katie_has_more_games_l99_99584

   -- Conditions
   def katie_games : Nat := 81
   def friends_games : Nat := 59

   -- Problem statement
   theorem katie_has_more_games : (katie_games - friends_games) = 22 :=
   by
     -- Proof to be provided
     sorry
   
end katie_has_more_games_l99_99584


namespace sunil_total_amount_proof_l99_99775

theorem sunil_total_amount_proof
  (CI : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) (P : ℝ) (A : ℝ)
  (h1 : CI = 492)
  (h2 : r = 0.05)
  (h3 : n = 1)
  (h4 : t = 2)
  (h5 : CI = P * ((1 + r / n) ^ (n * t) - 1))
  (h6 : A = P + CI) :
  A = 5292 :=
by
  -- Skip the proof.
  sorry

end sunil_total_amount_proof_l99_99775


namespace Josephine_sold_10_liters_l99_99306

def milk_sold (n1 n2 n3 : ℕ) (v1 v2 v3 : ℝ) : ℝ :=
  (v1 * n1) + (v2 * n2) + (v3 * n3)

theorem Josephine_sold_10_liters :
  milk_sold 3 2 5 2 0.75 0.5 = 10 :=
by
  sorry

end Josephine_sold_10_liters_l99_99306


namespace is_incorrect_B_l99_99914

variable {a b c : ℝ}

theorem is_incorrect_B :
  ¬ ((a > b ∧ b > c) → (1 / (b - c)) < (1 / (a - c))) :=
sorry

end is_incorrect_B_l99_99914


namespace ratio_of_x_and_y_l99_99917

theorem ratio_of_x_and_y {x y a b : ℝ} (h1 : (2 * a - x) / (3 * b - y) = 3) (h2 : a / b = 4.5) : x / y = 3 :=
sorry

end ratio_of_x_and_y_l99_99917


namespace simplify_and_evaluate_expression_l99_99119

theorem simplify_and_evaluate_expression (x : ℤ) (hx : x = 3) : 
  (1 - (x / (x + 1))) / ((x^2 - 2 * x + 1) / (x^2 - 1)) = 1 / 2 := by
  rw [hx]
  -- Here we perform the necessary rewrites and simplifications as shown in the steps
  sorry

end simplify_and_evaluate_expression_l99_99119


namespace dependent_variable_is_temperature_l99_99027

-- Define the variables involved in the problem
variables (intensity_of_sunlight : ℝ)
variables (temperature_of_water : ℝ)
variables (duration_of_exposure : ℝ)
variables (capacity_of_heater : ℝ)

-- Define the conditions
def changes_with_duration (temp: ℝ) (duration: ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∀ d, temp = f d) ∧ ∀ d₁ d₂, d₁ ≠ d₂ → f d₁ ≠ f d₂

-- The theorem we need to prove
theorem dependent_variable_is_temperature :
  changes_with_duration temperature_of_water duration_of_exposure → 
  (∀ t, ∃ d, temperature_of_water = t → duration_of_exposure = d) :=
sorry

end dependent_variable_is_temperature_l99_99027


namespace train_length_is_correct_l99_99421

noncomputable def train_speed_kmh : ℝ := 40
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
noncomputable def cross_time : ℝ := 25.2
noncomputable def train_length : ℝ := train_speed_ms * cross_time

theorem train_length_is_correct : train_length = 280.392 := by
  sorry

end train_length_is_correct_l99_99421


namespace problem_statement_l99_99573

theorem problem_statement (y : ℝ) (h : 8 / y^3 = y / 32) : y = 4 :=
by
  sorry

end problem_statement_l99_99573


namespace find_p_over_q_l99_99523

variables (x y p q : ℚ)

theorem find_p_over_q (h1 : (7 * x + 6 * y) / (x - 2 * y) = 27)
                      (h2 : x / (2 * y) = p / q) :
                      p / q = 3 / 2 :=
sorry

end find_p_over_q_l99_99523


namespace lockers_number_l99_99825

theorem lockers_number (total_cost : ℝ) (cost_per_digit : ℝ) (total_lockers : ℕ) 
  (locker_numbered_from_one : ∀ n : ℕ, n >= 1) :
  total_cost = 248.43 → cost_per_digit = 0.03 → total_lockers = 2347 :=
by
  intros h_total_cost h_cost_per_digit
  sorry

end lockers_number_l99_99825


namespace solution_set_of_inequality_l99_99568

theorem solution_set_of_inequality :
  { x : ℝ | (x - 1) / x ≥ 2 } = { x : ℝ | -1 ≤ x ∧ x < 0 } :=
by
  sorry

end solution_set_of_inequality_l99_99568


namespace largest_integral_x_l99_99711

theorem largest_integral_x (x : ℤ) : (2 / 7 : ℝ) < (x / 6) ∧ (x / 6) < (7 / 9) → x = 4 :=
by
  sorry

end largest_integral_x_l99_99711


namespace ratio_of_interior_to_exterior_angle_in_regular_octagon_l99_99782

theorem ratio_of_interior_to_exterior_angle_in_regular_octagon
  (n : ℕ) (regular_polygon : n = 8) : 
  let interior_angle := ((n - 2) * 180) / n
  let exterior_angle := 360 / n
  (interior_angle / exterior_angle) = 3 :=
by
  sorry

end ratio_of_interior_to_exterior_angle_in_regular_octagon_l99_99782


namespace minimum_surface_area_of_circumscribed_sphere_of_prism_l99_99382

theorem minimum_surface_area_of_circumscribed_sphere_of_prism :
  ∃ S : ℝ, 
    (∀ h r, r^2 * h = 4 → r^2 + (h^2 / 4) = R → 4 * π * R^2 = S) ∧ 
    (∀ S', S' ≤ S) ∧ 
    S = 12 * π :=
sorry

end minimum_surface_area_of_circumscribed_sphere_of_prism_l99_99382


namespace difference_of_cubes_not_div_by_twice_diff_l99_99117

theorem difference_of_cubes_not_div_by_twice_diff (a b : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_neq : a ≠ b) :
  ¬ (2 * (a - b)) ∣ ((a^3) - (b^3)) := 
sorry

end difference_of_cubes_not_div_by_twice_diff_l99_99117


namespace dice_number_divisible_by_7_l99_99726

theorem dice_number_divisible_by_7 :
  ∃ a b c : ℕ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) 
               ∧ (1001 * (100 * a + 10 * b + c)) % 7 = 0 :=
by
  sorry

end dice_number_divisible_by_7_l99_99726


namespace marvin_solved_yesterday_l99_99612

variables (M : ℕ)

def Marvin_yesterday := M
def Marvin_today := 3 * M
def Arvin_yesterday := 2 * M
def Arvin_today := 6 * M
def total_problems := Marvin_yesterday + Marvin_today + Arvin_yesterday + Arvin_today

theorem marvin_solved_yesterday :
  total_problems M = 480 → M = 40 :=
sorry

end marvin_solved_yesterday_l99_99612


namespace age_of_youngest_child_l99_99140

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by {
  sorry
}

end age_of_youngest_child_l99_99140


namespace damian_serena_passing_times_l99_99963

/-- 
  Damian and Serena are running on a circular track for 40 minutes.
  Damian runs clockwise at 220 m/min on the inner lane with a radius of 45 meters.
  Serena runs counterclockwise at 260 m/min on the outer lane with a radius of 55 meters.
  They start on the same radial line.
  Prove that they pass each other exactly 184 times in 40 minutes. 
-/
theorem damian_serena_passing_times
  (time_run : ℕ)
  (damian_speed : ℕ)
  (serena_speed : ℕ)
  (damian_radius : ℝ)
  (serena_radius : ℝ)
  (start_same_line : Prop) :
  time_run = 40 →
  damian_speed = 220 →
  serena_speed = 260 →
  damian_radius = 45 →
  serena_radius = 55 →
  start_same_line →
  ∃ n : ℕ, n = 184 :=
by
  sorry

end damian_serena_passing_times_l99_99963


namespace left_handed_like_jazz_l99_99519

theorem left_handed_like_jazz (total_people left_handed like_jazz right_handed_dislike_jazz : ℕ)
    (h1 : total_people = 30)
    (h2 : left_handed = 12)
    (h3 : like_jazz = 20)
    (h4 : right_handed_dislike_jazz = 3)
    (h5 : ∀ p, p = total_people - left_handed ∧ p = total_people - (left_handed + right_handed_dislike_jazz)) :
    ∃ x, x = 5 := by
  sorry

end left_handed_like_jazz_l99_99519


namespace david_account_amount_l99_99160

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem david_account_amount : compound_interest 5000 0.06 2 1 = 5304.50 := by
  sorry

end david_account_amount_l99_99160


namespace toothpicks_in_arithmetic_sequence_l99_99071

theorem toothpicks_in_arithmetic_sequence :
  let a1 := 5
  let d := 3
  let n := 15
  let a_n n := a1 + (n - 1) * d
  let sum_to_n n := n * (2 * a1 + (n - 1) * d) / 2
  sum_to_n n = 390 := by
  sorry

end toothpicks_in_arithmetic_sequence_l99_99071


namespace statements_evaluation_l99_99991

-- Define the statements A, B, C, D, E as propositions
def A : Prop := ∀ (A B C D E : Prop), (A → ¬B ∧ ¬C ∧ ¬D ∧ ¬E)
def B : Prop := sorry  -- Assume we have some way to read the statement B under special conditions
def C : Prop := ∀ (A B C D E : Prop), (A ∧ B ∧ C ∧ D ∧ E)
def D : Prop := sorry  -- Assume we have some way to read the statement D under special conditions
def E : Prop := A

-- Prove the conditions
theorem statements_evaluation : ¬ A ∧ ¬ C ∧ ¬ E ∧ B ∧ D :=
by
  sorry

end statements_evaluation_l99_99991


namespace capacity_of_initial_20_buckets_l99_99829

theorem capacity_of_initial_20_buckets (x : ℝ) (h : 20 * x = 270) : x = 13.5 :=
by 
  sorry

end capacity_of_initial_20_buckets_l99_99829


namespace pages_used_l99_99906

variable (n o c : ℕ)

theorem pages_used (h_n : n = 3) (h_o : o = 13) (h_c : c = 8) :
  (n + o) / c = 2 :=
  by
    sorry

end pages_used_l99_99906


namespace range_of_a_l99_99985

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (3 / 2)^x = (2 + 3 * a) / (5 - a)) ↔ a ∈ Set.Ioo (-2 / 3) (3 / 4) :=
by
  sorry

end range_of_a_l99_99985


namespace parabola_vertex_l99_99578

theorem parabola_vertex (y x : ℝ) (h : y = x^2 - 6 * x + 1) : 
  ∃ v_x v_y, (v_x, v_y) = (3, -8) :=
by 
  sorry

end parabola_vertex_l99_99578


namespace correct_analytical_method_l99_99607

-- Definitions of the different reasoning methods
def reasoning_from_cause_to_effect : Prop := ∀ (cause effect : Prop), cause → effect
def reasoning_from_effect_to_cause : Prop := ∀ (cause effect : Prop), effect → cause
def distinguishing_and_mutually_inferring : Prop := ∀ (cause effect : Prop), (cause ↔ effect)
def proving_converse_statement : Prop := ∀ (P Q : Prop), (P → Q) → (Q → P)

-- Definition of the analytical method
def analytical_method : Prop := reasoning_from_effect_to_cause

-- Theorem stating that the analytical method is the method of reasoning from effect to cause
theorem correct_analytical_method : analytical_method = reasoning_from_effect_to_cause := 
by 
  -- Complete this proof with refined arguments
  sorry

end correct_analytical_method_l99_99607


namespace stay_nights_l99_99284

theorem stay_nights (cost_per_night : ℕ) (num_people : ℕ) (total_cost : ℕ) (n : ℕ) 
    (h1 : cost_per_night = 40) (h2 : num_people = 3) (h3 : total_cost = 360) (h4 : cost_per_night * num_people * n = total_cost) :
    n = 3 :=
sorry

end stay_nights_l99_99284


namespace t_range_l99_99962

noncomputable def exists_nonneg_real_numbers_satisfying_conditions (t : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
  (3 * x^2 + 3 * z * x + z^2 = 1) ∧ 
  (3 * y^2 + 3 * y * z + z^2 = 4) ∧ 
  (x^2 - x * y + y^2 = t)

theorem t_range : ∀ t : ℝ, exists_nonneg_real_numbers_satisfying_conditions t → 
  (t ≥ (3 - Real.sqrt 5) / 2 ∧ t ≤ 1) :=
sorry

end t_range_l99_99962


namespace condition_neither_sufficient_nor_necessary_l99_99910

theorem condition_neither_sufficient_nor_necessary (p q : Prop) :
  (¬ (p ∧ q)) → (p ∨ q) → False :=
by sorry

end condition_neither_sufficient_nor_necessary_l99_99910


namespace trig_identity_proof_l99_99156

theorem trig_identity_proof :
  let sin240 := - (Real.sin (120 * Real.pi / 180))
  let tan240 := Real.tan (240 * Real.pi / 180)
  Real.sin (600 * Real.pi / 180) + tan240 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_proof_l99_99156


namespace percentage_of_boys_and_additional_boys_l99_99599

theorem percentage_of_boys_and_additional_boys (total_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ)
  (total_students_eq : total_students = 42) (ratio_condition : boys_ratio = 3 ∧ girls_ratio = 4) :
  let total_groups := total_students / (boys_ratio + girls_ratio)
  let total_boys := boys_ratio * total_groups
  (total_boys * 100 / total_students = 300 / 7) ∧ (21 - total_boys = 3) :=
by {
  sorry
}

end percentage_of_boys_and_additional_boys_l99_99599


namespace percentage_no_job_diploma_l99_99294

def percentage_with_university_diploma {total_population : ℕ} (has_diploma : ℕ) : ℕ :=
  (has_diploma / total_population) * 100

variables {total_population : ℕ} (p_no_diploma_and_job : ℕ) (p_with_job : ℕ) (p_diploma : ℕ)

axiom percentage_no_diploma_job :
  p_no_diploma_and_job = 10

axiom percentage_with_job :
  p_with_job = 40

axiom percentage_diploma :
  p_diploma = 39

theorem percentage_no_job_diploma :
  ∃ p : ℕ, p = (9 / 60) * 100 := sorry

end percentage_no_job_diploma_l99_99294


namespace expansion_three_times_expansion_six_times_l99_99058

-- Definition for the rule of expansion
def expand (a b : Nat) : Nat := a * b + a + b

-- Problem 1: Expansion with a = 1, b = 3 for 3 times results in 255.
theorem expansion_three_times : expand (expand (expand 1 3) 7) 31 = 255 := sorry

-- Problem 2: After 6 operations, the expanded number matches the given pattern.
theorem expansion_six_times (p q : ℕ) (hp : p > q) (hq : q > 0) : 
  ∃ m n, m = 8 ∧ n = 13 ∧ (expand (expand (expand (expand (expand (expand q (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) = (q + 1) ^ m * (p + 1) ^ n - 1 :=
sorry

end expansion_three_times_expansion_six_times_l99_99058


namespace individual_max_food_l99_99475

/-- Given a minimum number of guests and a total amount of food consumed,
    we want to find the maximum amount of food an individual guest could have consumed. -/
def total_food : ℝ := 319
def min_guests : ℝ := 160
def max_food_per_guest : ℝ := 1.99

theorem individual_max_food :
  total_food / min_guests <= max_food_per_guest := by
  sorry

end individual_max_food_l99_99475


namespace part_cost_l99_99087

theorem part_cost (hours : ℕ) (hourly_rate total_paid : ℕ) 
  (h1 : hours = 2)
  (h2 : hourly_rate = 75)
  (h3 : total_paid = 300) : 
  total_paid - (hours * hourly_rate) = 150 := 
by
  sorry

end part_cost_l99_99087


namespace thirtieth_term_of_arithmetic_seq_l99_99611

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l99_99611


namespace solve_quadratic_eq_l99_99988

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2 * x = 1) : x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end solve_quadratic_eq_l99_99988


namespace leoCurrentWeight_l99_99161

def currentWeightProblem (L K : Real) : Prop :=
  (L + 15 = 1.75 * K) ∧ (L + K = 250)

theorem leoCurrentWeight (L K : Real) (h : currentWeightProblem L K) : L = 154 :=
by
  sorry

end leoCurrentWeight_l99_99161


namespace range_of_k_intersecting_AB_l99_99676

theorem range_of_k_intersecting_AB 
  (A B : ℝ × ℝ) 
  (hA : A = (2, 7)) 
  (hB : B = (9, 6)) 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (H : ∃ x : ℝ, A.2 = k * A.1 ∧ B.2 = k * B.1):
  (2 / 3) ≤ k ∧ k ≤ 7 / 2 :=
by sorry

end range_of_k_intersecting_AB_l99_99676


namespace partOneCorrectProbability_partTwoCorrectProbability_l99_99980

noncomputable def teachers_same_gender_probability (mA fA mB fB : ℕ) : ℚ :=
  let total_outcomes := mA * mB + mA * fB + fA * mB + fA * fB
  let same_gender := mA * mB + fA * fB
  same_gender / total_outcomes

noncomputable def teachers_same_school_probability (SA SB : ℕ) : ℚ :=
  let total_teachers := SA + SB
  let total_outcomes := (total_teachers * (total_teachers - 1)) / 2
  let same_school := (SA * (SA - 1)) / 2 + (SB * (SB - 1)) / 2
  same_school / total_outcomes

theorem partOneCorrectProbability : teachers_same_gender_probability 2 1 1 2 = 4 / 9 := by
  sorry

theorem partTwoCorrectProbability : teachers_same_school_probability 3 3 = 2 / 5 := by
  sorry

end partOneCorrectProbability_partTwoCorrectProbability_l99_99980


namespace complex_power_sum_l99_99229

open Complex

theorem complex_power_sum (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 + Complex.I * Real.sqrt 3 ∨ z^99 + z^100 + z^101 + z^102 + z^103 = 2 - Complex.I * Real.sqrt 3 :=
sorry

end complex_power_sum_l99_99229


namespace ratio_of_age_difference_l99_99945

-- Define the ages of the scrolls and the ratio R
variables (S1 S2 S3 S4 S5 : ℕ)
variables (R : ℚ)

-- Conditions
axiom h1 : S1 = 4080
axiom h5 : S5 = 20655
axiom h2 : S2 - S1 = R * S5
axiom h3 : S3 - S2 = R * S5
axiom h4 : S4 - S3 = R * S5
axiom h6 : S5 - S4 = R * S5

-- The theorem to prove
theorem ratio_of_age_difference : R = 16575 / 82620 :=
by 
  sorry

end ratio_of_age_difference_l99_99945


namespace sum_of_distances_l99_99078

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_1 = 1 / 9 * d_2) (h2 : d_1 + d_2 = 6) : d_1 + d_2 + 6 = 20 :=
by
  sorry

end sum_of_distances_l99_99078


namespace derivative_at_one_l99_99080

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_at_one : deriv f 1 = -1 := sorry

end derivative_at_one_l99_99080


namespace f_even_l99_99851

-- Define E_x^n as specified
def E_x (n : ℕ) (x : ℝ) : ℝ := List.prod (List.map (λ i => x + i) (List.range n))

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * E_x 5 (x - 2)

-- Define the statement to prove f(x) is even
theorem f_even (x : ℝ) : f x = f (-x) := by
  sorry

end f_even_l99_99851


namespace trajectory_eq_range_of_k_l99_99375

-- definitions based on the conditions:
def fixed_circle (x y : ℝ) := (x + 1)^2 + y^2 = 16
def moving_circle_passing_through_B (M : ℝ × ℝ) (B : ℝ × ℝ) := 
    B = (1, 0) ∧ M.1^2 / 4 + M.2^2 / 3 = 1 -- the ellipse trajectory equation

-- question 1: prove the equation of the ellipse
theorem trajectory_eq :
    ∀ M : ℝ × ℝ, (∃ B : ℝ × ℝ, moving_circle_passing_through_B M B)
    → (M.1^2 / 4 + M.2^2 / 3 = 1) :=
sorry

-- question 2: find the range of k which satisfies given area condition
theorem range_of_k (k : ℝ) :
    (∃ M : ℝ × ℝ, ∃ B : ℝ × ℝ, moving_circle_passing_through_B M B) → 
    (0 < k) → (¬ (k = 0)) →
    ((∃ m : ℝ, (4 * k^2 + 3 - m^2 > 0) ∧ 
    (1 / 2) * (|k| * m^2 / (4 * k^2 + 3)^2) = 1 / 14) → (3 / 4 < k ∧ k < 1) 
    ∨ (-1 < k ∧ k < -3 / 4)) :=
sorry

end trajectory_eq_range_of_k_l99_99375


namespace fraction_eval_l99_99699

theorem fraction_eval : 
    (1 / (3 - (1 / (3 - (1 / (3 - (1 / 4))))))) = (11 / 29) := 
by
  sorry

end fraction_eval_l99_99699


namespace tan_135_eq_neg1_l99_99875

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg1_l99_99875


namespace incorrect_statement_c_l99_99971

-- Definitions based on conditions
variable (p q : Prop)

-- Lean 4 statement to check the logical proposition
theorem incorrect_statement_c (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
  sorry

end incorrect_statement_c_l99_99971


namespace eq_of_line_through_points_l99_99422

noncomputable def line_eqn (x y : ℝ) : Prop :=
  x - y + 3 = 0

theorem eq_of_line_through_points :
  ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = -1 → y1 = 2 → x2 = 2 → y2 = 5 → 
    line_eqn (x1 + y1 - x2) (y2 - y1) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2
  rw [hx1, hy1, hx2, hy2]
  sorry -- Proof steps would go here.

end eq_of_line_through_points_l99_99422


namespace solve_equation_in_integers_l99_99660
-- Import the necessary library for Lean

-- Define the main theorem to solve the equation in integers
theorem solve_equation_in_integers :
  ∃ (xs : List (ℕ × ℕ)), (∀ x y, (3^x - 2^y = 1 → (x, y) ∈ xs)) ∧ xs = [(1, 1), (2, 3)] :=
by
  sorry

end solve_equation_in_integers_l99_99660


namespace partition_555_weights_l99_99734

theorem partition_555_weights :
  ∃ A B C : Finset ℕ, 
  (∀ x ∈ A, x ∈ Finset.range (555 + 1)) ∧ 
  (∀ y ∈ B, y ∈ Finset.range (555 + 1)) ∧ 
  (∀ z ∈ C, z ∈ Finset.range (555 + 1)) ∧ 
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
  A ∪ B ∪ C = Finset.range (555 + 1) ∧ 
  A.sum id = 51430 ∧ B.sum id = 51430 ∧ C.sum id = 51430 := sorry

end partition_555_weights_l99_99734


namespace line_condition_l99_99428

/-- Given a line l1 passing through points A(-2, m) and B(m, 4),
    a line l2 given by the equation 2x + y - 1 = 0,
    and a line l3 given by the equation x + ny + 1 = 0,
    if l1 is parallel to l2 and l2 is perpendicular to l3,
    then the value of m + n is -10. -/
theorem line_condition (m n : ℝ) (h1 : (4 - m) / (m + 2) = -2)
  (h2 : (2 * -1) * (-1 / n) = -1) : m + n = -10 := 
sorry

end line_condition_l99_99428


namespace speed_of_man_in_still_water_l99_99735

variable (V_m V_s : ℝ)

/-- The speed of a man in still water -/
theorem speed_of_man_in_still_water (h_downstream : 18 = (V_m + V_s) * 3)
                                     (h_upstream : 12 = (V_m - V_s) * 3) :
    V_m = 5 := 
sorry

end speed_of_man_in_still_water_l99_99735


namespace discount_percentage_is_25_l99_99955

-- Define the conditions
def cost_of_coffee : ℕ := 6
def cost_of_cheesecake : ℕ := 10
def final_price_with_discount : ℕ := 12

-- Define the total cost without discount
def total_cost_without_discount : ℕ := cost_of_coffee + cost_of_cheesecake

-- Define the discount amount
def discount_amount : ℕ := total_cost_without_discount - final_price_with_discount

-- Define the percentage discount
def percentage_discount : ℕ := (discount_amount * 100) / total_cost_without_discount

-- Proof Statement
theorem discount_percentage_is_25 : percentage_discount = 25 := by
  sorry

end discount_percentage_is_25_l99_99955


namespace tetrahedron_volume_is_zero_l99_99109

noncomputable def volume_of_tetrahedron (p q r : ℝ) : ℝ :=
  (1 / 6) * p * q * r

theorem tetrahedron_volume_is_zero (p q r : ℝ)
  (hpq : p^2 + q^2 = 36)
  (hqr : q^2 + r^2 = 64)
  (hrp : r^2 + p^2 = 100) :
  volume_of_tetrahedron p q r = 0 := by
  sorry

end tetrahedron_volume_is_zero_l99_99109


namespace second_solution_sugar_percentage_l99_99205

theorem second_solution_sugar_percentage
  (initial_solution_pct : ℝ)
  (second_solution_pct : ℝ)
  (initial_solution_amount : ℝ)
  (final_solution_pct : ℝ)
  (replaced_fraction : ℝ)
  (final_amount : ℝ) :
  initial_solution_pct = 0.1 →
  final_solution_pct = 0.17 →
  replaced_fraction = 1/4 →
  initial_solution_amount = 100 →
  final_amount = 100 →
  second_solution_pct = 0.38 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end second_solution_sugar_percentage_l99_99205


namespace circle_center_sum_l99_99171

theorem circle_center_sum (h k : ℝ) :
  (∃ h k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = 6 * x + 8 * y - 15) → (h, k) = (3, 4)) →
  h + k = 7 :=
by
  sorry

end circle_center_sum_l99_99171


namespace picture_area_l99_99169

theorem picture_area (x y : ℤ) (hx : 1 < x) (hy : 1 < y) (h : (x + 2) * (y + 4) = 45) : x * y = 15 := by
  sorry

end picture_area_l99_99169


namespace evelyn_lost_bottle_caps_l99_99925

-- Definitions from the conditions
def initial_amount : ℝ := 63.0
def final_amount : ℝ := 45.0
def lost_amount : ℝ := 18.0

-- Statement to be proved
theorem evelyn_lost_bottle_caps : initial_amount - final_amount = lost_amount := 
by 
  sorry

end evelyn_lost_bottle_caps_l99_99925


namespace isaiah_types_more_words_than_micah_l99_99905

theorem isaiah_types_more_words_than_micah :
  let micah_speed := 20   -- Micah's typing speed in words per minute
  let isaiah_speed := 40  -- Isaiah's typing speed in words per minute
  let minutes_in_hour := 60  -- Number of minutes in an hour
  (isaiah_speed * minutes_in_hour) - (micah_speed * minutes_in_hour) = 1200 :=
by
  sorry

end isaiah_types_more_words_than_micah_l99_99905


namespace simplify_expression_l99_99357

variable (m : ℝ)

theorem simplify_expression (h₁ : m ≠ 2) (h₂ : m ≠ 3) :
  (m - (4 * m - 9) / (m - 2)) / ((m ^ 2 - 9) / (m - 2)) = (m - 3) / (m + 3) := 
sorry

end simplify_expression_l99_99357


namespace max_elves_without_caps_proof_max_elves_with_caps_proof_l99_99798

-- Defining the conditions and the problem statement
open Nat

-- We model the problem with the following:
axiom truth_teller : Type
axiom liar_with_caps : Type
axiom dwarf_with_caps : Type
axiom dwarf_without_caps : Type

noncomputable def max_elves_without_caps : ℕ :=
  59

noncomputable def max_elves_with_caps : ℕ :=
  30

-- Part (a): Given the conditions, we show that the maximum number of elves without caps is 59
theorem max_elves_without_caps_proof : max_elves_without_caps = 59 :=
by
  sorry

-- Part (b): Given the conditions, we show that the maximum number of elves with caps is 30
theorem max_elves_with_caps_proof : max_elves_with_caps = 30 :=
by
  sorry

end max_elves_without_caps_proof_max_elves_with_caps_proof_l99_99798


namespace share_of_A_correct_l99_99950

theorem share_of_A_correct :
  let investment_A1 := 20000
  let investment_A2 := 15000
  let investment_B1 := 20000
  let investment_B2 := 16000
  let investment_C1 := 20000
  let investment_C2 := 26000
  let total_months1 := 5
  let total_months2 := 7
  let total_profit := 69900

  let total_investment_A := (investment_A1 * total_months1) + (investment_A2 * total_months2)
  let total_investment_B := (investment_B1 * total_months1) + (investment_B2 * total_months2)
  let total_investment_C := (investment_C1 * total_months1) + (investment_C2 * total_months2)
  let total_investment := total_investment_A + total_investment_B + total_investment_C

  let share_A := (total_investment_A : ℝ) / (total_investment : ℝ)
  let profit_A := share_A * (total_profit : ℝ)

  profit_A = 20500.99 :=
by
  sorry

end share_of_A_correct_l99_99950


namespace option_d_may_not_hold_l99_99924

theorem option_d_may_not_hold (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m^2 * a > m^2 * b) :=
sorry

end option_d_may_not_hold_l99_99924


namespace necessary_and_sufficient_condition_l99_99412

noncomputable def f (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem necessary_and_sufficient_condition
  {a b c : ℝ}
  (ha_pos : a > 0) :
  ( (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, f a b c x = y } → ∃! x : ℝ, f a b c x = y) ∧ 
    (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, y = f a b c x } → ∃! x : ℝ, f a b c x = y)
  ) ↔
  f a b c (f a b c (-b / (2 * a))) < 0 :=
sorry

end necessary_and_sufficient_condition_l99_99412


namespace x_is_integer_l99_99764

theorem x_is_integer
  (x : ℝ)
  (h_pos : 0 < x)
  (h1 : ∃ k1 : ℤ, x^2012 = x^2001 + k1)
  (h2 : ∃ k2 : ℤ, x^2001 = x^1990 + k2) : 
  ∃ n : ℤ, x = n :=
sorry

end x_is_integer_l99_99764


namespace question1_question2_question3_l99_99133

-- Question 1
theorem question1 (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5)^2) :
  a = m^2 + 5 * n^2 ∧ b = 2 * m * n :=
sorry

-- Question 2
theorem question2 (x m n: ℕ) (h : x + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) :
  (m = 1 ∧ n = 2 ∧ x = 13) ∨ (m = 2 ∧ n = 1 ∧ x = 7) :=
sorry

-- Question 3
theorem question3 : Real.sqrt (5 + 2 * Real.sqrt 6) = Real.sqrt 2 + Real.sqrt 3 :=
sorry

end question1_question2_question3_l99_99133


namespace exist_end_2015_l99_99337

def in_sequence (n : Nat) : Nat :=
  90 * n + 75

theorem exist_end_2015 :
  ∃ n : Nat, in_sequence n % 10000 = 2015 :=
by
  sorry

end exist_end_2015_l99_99337


namespace clock_angle_3_to_7_l99_99658

theorem clock_angle_3_to_7 : 
  let number_of_rays := 12
  let total_degrees := 360
  let degree_per_ray := total_degrees / number_of_rays
  let angle_3_to_7 := 4 * degree_per_ray
  angle_3_to_7 = 120 :=
by
  sorry

end clock_angle_3_to_7_l99_99658


namespace count_multiples_of_15_l99_99574

theorem count_multiples_of_15 (a b n : ℕ) (h_gte : 25 ≤ a) (h_lte : b ≤ 205) (h15 : n = 15) : 
  (∃ (k : ℕ), a ≤ k * n ∧ k * n ≤ b ∧ 1 ≤ k - 1 ∧ k - 1 ≤ 12) :=
sorry

end count_multiples_of_15_l99_99574


namespace person_A_takes_12_more_minutes_l99_99163

-- Define distances, speeds, times
variables (S : ℝ) (v_A v_B : ℝ) (t : ℝ)

-- Define conditions as hypotheses
def conditions (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t) : Prop :=
  (v_A * (t + 4/5) = 2/3 * S) ∧ (v_B * t = 2/3 * S) ∧ (v_A * (t + 4/5 + 1/2 * t + 1/10) + 1/10 * v_B = S)

-- The proof problem statement
theorem person_A_takes_12_more_minutes
  (S : ℝ) (v_A v_B : ℝ) (t : ℝ)
  (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t)
  (h4 : conditions S v_A v_B t h1 h2 h3) : (t + 4/5) + 6/5 = 96 / 60 + 12 / 60 :=
sorry

end person_A_takes_12_more_minutes_l99_99163


namespace gcd_stamps_pages_l99_99732

def num_stamps_book1 : ℕ := 924
def num_stamps_book2 : ℕ := 1200

theorem gcd_stamps_pages : Nat.gcd num_stamps_book1 num_stamps_book2 = 12 := by
  sorry

end gcd_stamps_pages_l99_99732


namespace evaluate_expression_l99_99922

theorem evaluate_expression : 1234562 - (12 * 3 * (2 + 7)) = 1234238 :=
by 
  sorry

end evaluate_expression_l99_99922


namespace largest_value_of_a_l99_99817

theorem largest_value_of_a
  (a b c d e : ℕ)
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : e = d - 10)
  (h5 : e < 105) :
  a ≤ 6824 :=
by {
  -- Proof omitted
  sorry
}

end largest_value_of_a_l99_99817


namespace value_of_a_l99_99360

theorem value_of_a (a b c : ℕ) (h1 : a + b = 12) (h2 : b + c = 16) (h3 : c = 7) : a = 3 := by
  sorry

end value_of_a_l99_99360


namespace exists_a_satisfying_inequality_l99_99137

theorem exists_a_satisfying_inequality (x : ℝ) : 
  x < -2 ∨ (0 < x ∧ x < 1) ∨ 1 < x → 
  ∃ a ∈ Set.Icc (-1 : ℝ) 2, (2 - a) * x^3 + (1 - 2 * a) * x^2 - 6 * x + 5 + 4 * a - a^2 < 0 := 
by 
  intros h
  sorry

end exists_a_satisfying_inequality_l99_99137


namespace last_integer_in_sequence_div3_l99_99993

theorem last_integer_in_sequence_div3 (a0 : ℤ) (sequence : ℕ → ℤ)
  (h0 : a0 = 1000000000)
  (h_seq : ∀ n, sequence n = a0 / (3^n)) :
  ∃ k, sequence k = 2 ∧ ∀ m, sequence m < 2 → sequence m < 1 := 
sorry

end last_integer_in_sequence_div3_l99_99993


namespace dice_sum_surface_l99_99170

theorem dice_sum_surface (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 6) : 
  ∃ Y : ℕ, Y = 28175 + 2 * X ∧ (Y = 28177 ∨ Y = 28179 ∨ Y = 28181 ∨ Y = 28183 ∨ 
  Y = 28185 ∨ Y = 28187) :=
by
  sorry

end dice_sum_surface_l99_99170


namespace g_g_g_g_2_eq_1406_l99_99385

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 1

theorem g_g_g_g_2_eq_1406 : g (g (g (g 2))) = 1406 := by
  sorry

end g_g_g_g_2_eq_1406_l99_99385


namespace even_square_even_square_even_even_l99_99684

-- Definition for a natural number being even
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Statement 1: If p is even, then p^2 is even
theorem even_square_even (p : ℕ) (hp : is_even p) : is_even (p * p) :=
sorry

-- Statement 2: If p^2 is even, then p is even
theorem square_even_even (p : ℕ) (hp_squared : is_even (p * p)) : is_even p :=
sorry

end even_square_even_square_even_even_l99_99684


namespace average_marks_l99_99553

theorem average_marks
  (M P C : ℕ)
  (h1 : M + P = 70)
  (h2 : C = P + 20) :
  (M + C) / 2 = 45 :=
sorry

end average_marks_l99_99553


namespace eiffel_tower_model_ratio_l99_99424

/-- Define the conditions of the problem as a structure -/
structure ModelCondition where
  eiffelTowerHeight : ℝ := 984 -- in feet
  modelHeight : ℝ := 6        -- in inches

/-- The main theorem statement -/
theorem eiffel_tower_model_ratio (cond : ModelCondition) : cond.eiffelTowerHeight / cond.modelHeight = 164 := 
by
  -- We can leave the proof out with 'sorry' for now.
  sorry

end eiffel_tower_model_ratio_l99_99424


namespace number_of_walls_l99_99269

theorem number_of_walls (bricks_per_row rows_per_wall total_bricks : Nat) :
  bricks_per_row = 30 → 
  rows_per_wall = 50 → 
  total_bricks = 3000 → 
  total_bricks / (bricks_per_row * rows_per_wall) = 2 := 
by
  intros h1 h2 h3
  sorry

end number_of_walls_l99_99269


namespace probability_is_pi_over_12_l99_99274

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let radius := 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := 6 * 8
  circle_area / rectangle_area

theorem probability_is_pi_over_12 :
  probability_within_two_units_of_origin = Real.pi / 12 :=
by
  sorry

end probability_is_pi_over_12_l99_99274


namespace M_intersect_N_l99_99454

def M : Set ℝ := {x | 1 + x > 0}
def N : Set ℝ := {x | x < 1}

theorem M_intersect_N : M ∩ N = {x | -1 < x ∧ x < 1} := 
by
  sorry

end M_intersect_N_l99_99454


namespace shrimp_appetizer_cost_l99_99781

-- Define the conditions
def shrimp_per_guest : ℕ := 5
def number_of_guests : ℕ := 40
def cost_per_pound : ℕ := 17
def shrimp_per_pound : ℕ := 20

-- Define the proof statement
theorem shrimp_appetizer_cost : 
  (shrimp_per_guest * number_of_guests / shrimp_per_pound) * cost_per_pound = 170 := 
by
  sorry

end shrimp_appetizer_cost_l99_99781


namespace problem1_problem2_l99_99248

theorem problem1 : (1 * (-5) - (-6) + (-7)) = -6 :=
by
  sorry

theorem problem2 : (-1)^2021 + (-18) * abs (-2 / 9) - 4 / (-2) = -3 :=
by
  sorry

end problem1_problem2_l99_99248


namespace lift_ratio_l99_99299

theorem lift_ratio (total_weight first_lift second_lift : ℕ) (h1 : total_weight = 1500)
(h2 : first_lift = 600) (h3 : first_lift = 2 * (second_lift - 300)) : first_lift / second_lift = 1 := 
by
  sorry

end lift_ratio_l99_99299


namespace johns_trip_distance_is_160_l99_99264

noncomputable def total_distance (y : ℕ) : Prop :=
  y / 2 + 40 + y / 4 = y

theorem johns_trip_distance_is_160 : ∃ y : ℕ, total_distance y ∧ y = 160 :=
by
  use 160
  unfold total_distance
  sorry

end johns_trip_distance_is_160_l99_99264


namespace maximize_h_at_1_l99_99236

-- Definitions and conditions
def f (x : ℝ) : ℝ := -2 * x + 2
def g (x : ℝ) : ℝ := -3 * x + 6
def h (x : ℝ) : ℝ := f x * g x

-- The theorem to prove
theorem maximize_h_at_1 : (∀ x : ℝ, h x <= h 1) :=
sorry

end maximize_h_at_1_l99_99236


namespace proof_Bill_age_is_24_l99_99215

noncomputable def Bill_is_24 (C : ℝ) (Bill_age : ℝ) (Daniel_age : ℝ) :=
  (Bill_age = 2 * C - 1) ∧ 
  (Daniel_age = C - 4) ∧ 
  (C + Bill_age + Daniel_age = 45) → 
  (Bill_age = 24)

theorem proof_Bill_age_is_24 (C Bill_age Daniel_age : ℝ) : 
  Bill_is_24 C Bill_age Daniel_age :=
by
  sorry

end proof_Bill_age_is_24_l99_99215


namespace square_assembly_possible_l99_99979

theorem square_assembly_possible (Area1 Area2 Area3 : ℕ) (h1 : Area1 = 29) (h2 : Area2 = 18) (h3 : Area3 = 10) (h_total : Area1 + Area2 + Area3 = 57) : 
  ∃ s : ℝ, s^2 = 57 ∧ true :=
by
  sorry

end square_assembly_possible_l99_99979


namespace arithmetic_seq_a8_l99_99536

def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h_arith : is_arith_seq a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 6) :
  a 8 = 14 := sorry

end arithmetic_seq_a8_l99_99536


namespace maria_needs_green_beans_l99_99673

theorem maria_needs_green_beans :
  ∀ (potatoes carrots onions green_beans : ℕ), 
  (carrots = 6 * potatoes) →
  (onions = 2 * carrots) →
  (green_beans = onions / 3) →
  (potatoes = 2) →
  green_beans = 8 :=
by
  intros potatoes carrots onions green_beans h1 h2 h3 h4
  rw [h4, Nat.mul_comm 6 2] at h1
  rw [h1, Nat.mul_comm 2 12] at h2
  rw [h2] at h3
  sorry

end maria_needs_green_beans_l99_99673


namespace bugs_eat_total_flowers_l99_99733

def num_bugs : ℝ := 2.0
def flowers_per_bug : ℝ := 1.5
def total_flowers_eaten : ℝ := 3.0

theorem bugs_eat_total_flowers : 
  (num_bugs * flowers_per_bug) = total_flowers_eaten := 
  by 
    sorry

end bugs_eat_total_flowers_l99_99733


namespace usual_time_of_train_l99_99216

theorem usual_time_of_train (S T : ℝ) (h_speed : S ≠ 0) 
(h_speed_ratio : ∀ (T' : ℝ), T' = T + 3/4 → S * T = (4/5) * S * T' → T = 3) : Prop :=
  T = 3

end usual_time_of_train_l99_99216


namespace harmonic_mean_closest_to_six_l99_99192

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_closest_to_six : 
     |harmonic_mean 3 2023 - 6| < 1 :=
sorry

end harmonic_mean_closest_to_six_l99_99192


namespace critics_voted_same_actor_actress_l99_99275

theorem critics_voted_same_actor_actress :
  ∃ (critic1 critic2 : ℕ) 
  (actor_vote1 actor_vote2 actress_vote1 actress_vote2 : ℕ),
  1 ≤ critic1 ∧ critic1 ≤ 3366 ∧
  1 ≤ critic2 ∧ critic2 ≤ 3366 ∧
  (critic1 ≠ critic2) ∧
  ∃ (vote_count : Fin 100 → ℕ) 
  (actor actress : Fin 3366 → Fin 100),
  (∀ n : Fin 100, ∃ act : Fin 100, vote_count act = n + 1) ∧
  actor critic1 = actor_vote1 ∧ actress critic1 = actress_vote1 ∧
  actor critic2 = actor_vote2 ∧ actress critic2 = actress_vote2 ∧
  actor_vote1 = actor_vote2 ∧ actress_vote1 = actress_vote2 :=
by
  -- Proof omitted
  sorry

end critics_voted_same_actor_actress_l99_99275


namespace solve_for_x_l99_99567

theorem solve_for_x (h : 125 = 5 ^ 3) : ∃ x : ℕ, 125 ^ 4 = 5 ^ x ∧ x = 12 := by
  sorry

end solve_for_x_l99_99567


namespace assignment_schemes_with_at_least_one_girl_l99_99580

theorem assignment_schemes_with_at_least_one_girl
  (boys girls : ℕ)
  (tasks : ℕ)
  (hb : boys = 4)
  (hg : girls = 3)
  (ht : tasks = 3)
  (total_choices : ℕ := (boys + girls).choose tasks * tasks.factorial)
  (all_boys : ℕ := boys.choose tasks * tasks.factorial) :
  total_choices - all_boys = 186 :=
by
  sorry

end assignment_schemes_with_at_least_one_girl_l99_99580


namespace probability_two_boys_l99_99012

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def total_pairs : ℕ := Nat.choose number_of_students 2
def boys_pairs : ℕ := Nat.choose number_of_boys 2

theorem probability_two_boys :
  number_of_students = 5 →
  number_of_boys = 2 →
  number_of_girls = 3 →
  (boys_pairs : ℝ) / (total_pairs : ℝ) = 1 / 10 :=
by
  sorry

end probability_two_boys_l99_99012


namespace probability_same_tribe_l99_99772

def totalPeople : ℕ := 18
def peoplePerTribe : ℕ := 6
def tribes : ℕ := 3
def totalQuitters : ℕ := 2

def totalWaysToChooseQuitters := Nat.choose totalPeople totalQuitters
def waysToChooseFromTribe := Nat.choose peoplePerTribe totalQuitters
def totalWaysFromSameTribe := tribes * waysToChooseFromTribe

theorem probability_same_tribe (h1 : totalPeople = 18) (h2 : peoplePerTribe = 6) (h3 : tribes = 3) (h4 : totalQuitters = 2)
    (h5 : totalWaysToChooseQuitters = 153) (h6 : totalWaysFromSameTribe = 45) :
    (totalWaysFromSameTribe : ℚ) / totalWaysToChooseQuitters = 5 / 17 := by
  sorry

end probability_same_tribe_l99_99772


namespace children_count_l99_99609

variable (M W C : ℕ)

theorem children_count (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : M + W + C = 300) : C = 30 := by
  sorry

end children_count_l99_99609


namespace housewife_spending_l99_99318

theorem housewife_spending (P R A : ℝ) (h1 : R = 34.2) (h2 : R = 0.8 * P) (h3 : A / R - A / P = 4) :
  A = 683.45 :=
by
  sorry

end housewife_spending_l99_99318


namespace knocks_to_knicks_l99_99898

variable (knicks knacks knocks : ℝ)

def knicks_eq_knacks : Prop := 
  8 * knicks = 3 * knacks

def knacks_eq_knocks : Prop := 
  4 * knacks = 5 * knocks

theorem knocks_to_knicks
  (h1 : knicks_eq_knacks knicks knacks)
  (h2 : knacks_eq_knocks knacks knocks) :
  20 * knocks = 320 / 15 * knicks :=
  sorry

end knocks_to_knicks_l99_99898


namespace largest_angle_in_isosceles_triangle_l99_99503

-- Definitions of the conditions from the problem
def isosceles_triangle (A B C : ℕ) : Prop :=
  A = B ∨ B = C ∨ A = C

def angle_opposite_equal_side (θ : ℕ) : Prop :=
  θ = 50

-- The proof problem statement
theorem largest_angle_in_isosceles_triangle (A B C : ℕ) (θ : ℕ)
  : isosceles_triangle A B C → angle_opposite_equal_side θ → ∃ γ, γ = 80 :=
by
  sorry

end largest_angle_in_isosceles_triangle_l99_99503


namespace coral_three_night_total_pages_l99_99279

-- Definitions based on conditions in the problem
def night1_pages : ℕ := 30
def night2_pages : ℕ := 2 * night1_pages - 2
def night3_pages : ℕ := night1_pages + night2_pages + 3
def total_pages : ℕ := night1_pages + night2_pages + night3_pages

-- The statement we want to prove
theorem coral_three_night_total_pages : total_pages = 179 := by
  sorry

end coral_three_night_total_pages_l99_99279


namespace pig_farm_fence_l99_99358

theorem pig_farm_fence (fenced_side : ℝ) (area : ℝ) 
  (h1 : fenced_side * 2 * fenced_side = area) 
  (h2 : area = 1250) :
  4 * fenced_side = 100 :=
by {
  sorry
}

end pig_farm_fence_l99_99358


namespace ratio_b4_b3_a2_a1_l99_99346

variables {x y d d' : ℝ}
variables {a1 a2 a3 b1 b2 b3 b4 : ℝ}
-- Conditions
variables (h1 : x ≠ y)
variables (h2 : a1 = x + d)
variables (h3 : a2 = x + 2 * d)
variables (h4 : a3 = x + 3 * d)
variables (h5 : y = x + 4 * d)
variables (h6 : b2 = x + d')
variables (h7 : b3 = x + 2 * d')
variables (h8 : y = x + 3 * d')
variables (h9 : b4 = x + 4 * d')

theorem ratio_b4_b3_a2_a1 :
  (b4 - b3) / (a2 - a1) = 8 / 3 :=
by sorry

end ratio_b4_b3_a2_a1_l99_99346


namespace factorize_1_factorize_2_l99_99281

-- Proof problem 1: Prove x² - 6x + 9 = (x - 3)²
theorem factorize_1 (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by sorry

-- Proof problem 2: Prove x²(y - 2) - 4(y - 2) = (y - 2)(x + 2)(x - 2)
theorem factorize_2 (x y : ℝ) : x^2 * (y - 2) - 4 * (y - 2) = (y - 2) * (x + 2) * (x - 2) :=
by sorry

end factorize_1_factorize_2_l99_99281


namespace cost_price_is_correct_l99_99827

-- Define the conditions
def purchasing_clocks : ℕ := 150
def gain_60_clocks : ℝ := 0.12
def gain_90_clocks : ℝ := 0.18
def uniform_profit : ℝ := 0.16
def difference_in_profit : ℝ := 75

-- Define the cost price of each clock
noncomputable def C : ℝ := 125

-- Define and state the theorem
theorem cost_price_is_correct (C : ℝ) :
  (60 * C * (1 + gain_60_clocks) + 90 * C * (1 + gain_90_clocks)) - (150 * C * (1 + uniform_profit)) = difference_in_profit :=
sorry

end cost_price_is_correct_l99_99827


namespace smith_gave_randy_l99_99618

theorem smith_gave_randy :
  ∀ (s amount_given amount_left : ℕ), amount_given = 1200 → amount_left = 2000 → s = amount_given + amount_left → s = 3200 :=
by
  intros s amount_given amount_left h_given h_left h_total
  rw [h_given, h_left] at h_total
  exact h_total

end smith_gave_randy_l99_99618


namespace meters_examined_l99_99049

theorem meters_examined (x : ℝ) (h1 : 0.07 / 100 * x = 2) : x = 2857 :=
by
  -- using the given setup and simplification
  sorry

end meters_examined_l99_99049


namespace posters_total_l99_99120

-- Definitions based on conditions
def Mario_posters : Nat := 18
def Samantha_posters : Nat := Mario_posters + 15

-- Statement to prove: They made 51 posters altogether
theorem posters_total : Mario_posters + Samantha_posters = 51 := 
by sorry

end posters_total_l99_99120


namespace calculate_spadesuit_l99_99394

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem calculate_spadesuit : spadesuit 3 (spadesuit 5 6) = -112 := by
  sorry

end calculate_spadesuit_l99_99394
