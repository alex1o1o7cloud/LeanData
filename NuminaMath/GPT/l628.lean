import Mathlib

namespace final_people_amount_l628_62837

def initial_people : ℕ := 250
def people_left1 : ℕ := 35
def people_joined1 : ℕ := 20
def percentage_left : ℕ := 10
def groups_joined : ℕ := 4
def group_size : ℕ := 15

theorem final_people_amount :
  let intermediate_people1 := initial_people - people_left1;
  let intermediate_people2 := intermediate_people1 + people_joined1;
  let people_left2 := (intermediate_people2 * percentage_left) / 100;
  let rounded_people_left2 := people_left2;
  let intermediate_people3 := intermediate_people2 - rounded_people_left2;
  let total_new_join := groups_joined * group_size;
  let final_people := intermediate_people3 + total_new_join;
  final_people = 272 :=
by sorry

end final_people_amount_l628_62837


namespace probability_exceeds_175_l628_62816

theorem probability_exceeds_175 (P_lt_160 : ℝ) (P_160_to_175 : ℝ) (h : ℝ) :
  P_lt_160 = 0.2 → P_160_to_175 = 0.5 → 1 - P_lt_160 - P_160_to_175 = 0.3 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end probability_exceeds_175_l628_62816


namespace combined_length_of_legs_is_ten_l628_62867

-- Define the conditions given in the problem.
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * Real.sqrt 2

def hypotenuse_length (c : ℝ) : Prop :=
  c = 7.0710678118654755

def perimeter_condition (a b c perimeter : ℝ) : Prop :=
  perimeter = a + b + c ∧ perimeter = 10 + c

-- Prove the combined length of the two legs is 10.
theorem combined_length_of_legs_is_ten :
  ∃ (a b c : ℝ), is_isosceles_right_triangle a b c →
  hypotenuse_length c →
  ∀ perimeter : ℝ, perimeter_condition a b c perimeter →
  2 * a = 10 :=
by
  sorry

end combined_length_of_legs_is_ten_l628_62867


namespace remainder_calculation_l628_62879

theorem remainder_calculation : 
  ∀ (dividend divisor quotient remainder : ℕ), 
  dividend = 158 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 :=
by
  intros dividend divisor quotient remainder hdividend hdivisor hquotient heq
  sorry

end remainder_calculation_l628_62879


namespace gasoline_tank_capacity_l628_62878

-- Given conditions
def initial_fraction_full := 5 / 6
def used_gallons := 15
def final_fraction_full := 2 / 3

-- Mathematical problem statement in Lean 4
theorem gasoline_tank_capacity (x : ℝ)
  (initial_full : initial_fraction_full * x = 5 / 6 * x)
  (final_full : initial_fraction_full * x - used_gallons = final_fraction_full * x) :
  x = 90 := by
  sorry

end gasoline_tank_capacity_l628_62878


namespace find_x_l628_62882

theorem find_x (x y: ℤ) (h1: x + 2 * y = 12) (h2: y = 3) : x = 6 := by
  sorry

end find_x_l628_62882


namespace problem_inequality_l628_62812

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

axiom f_pos : ∀ x : ℝ, x > 0 → f x > 0

axiom f_increasing : ∀ x y : ℝ, x > 0 → y > 0 → x ≤ y → (f x / x) ≤ (f y / y)

theorem problem_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * ((f a + f b) / (a + b) + (f b + f c) / (b + c) + (f c + f a) / (c + a)) ≥ 
    3 * (f a + f b + f c) / (a + b + c) + (f a / a + f b / b + f c / c) :=
sorry

end problem_inequality_l628_62812


namespace polynomial_solution_l628_62868

theorem polynomial_solution (p : ℝ → ℝ) (h : ∀ x, p (p x) = x * (p x) ^ 2 + x ^ 3) : 
  p = id :=
by {
    sorry
}

end polynomial_solution_l628_62868


namespace positive_divisors_840_multiple_of_4_l628_62871

theorem positive_divisors_840_multiple_of_4 :
  let n := 840
  let prime_factors := (2^3 * 3^1 * 5^1 * 7^1)
  (∀ k : ℕ, k ∣ n → k % 4 = 0 → ∀ a b c d : ℕ, 2 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 →
  k = 2^a * 3^b * 5^c * 7^d) → 
  (∃ count, count = 16) :=
by {
  sorry
}

end positive_divisors_840_multiple_of_4_l628_62871


namespace sum_of_areas_of_triangles_in_cube_l628_62815

theorem sum_of_areas_of_triangles_in_cube : 
  let m := 48
  let n := 4608
  let p := 576
  m + n + p = 5232 := by 
    sorry

end sum_of_areas_of_triangles_in_cube_l628_62815


namespace power_function_increasing_l628_62839

   theorem power_function_increasing (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x < y → x^a < y^a) : 0 < a :=
   by
   sorry
   
end power_function_increasing_l628_62839


namespace ratio_of_w_to_y_l628_62862

variables (w x y z : ℚ)

theorem ratio_of_w_to_y:
  (w / x = 5 / 4) →
  (y / z = 5 / 3) →
  (z / x = 1 / 5) →
  (w / y = 15 / 4) :=
by
  intros hwx hyz hzx
  sorry

end ratio_of_w_to_y_l628_62862


namespace original_number_of_members_l628_62892

-- Define the initial conditions
variables (x y : ℕ)

-- First condition: if five 9-year-old members leave
def condition1 : Prop := x * y - 45 = (y + 1) * (x - 5)

-- Second condition: if five 17-year-old members join
def condition2 : Prop := x * y + 85 = (y + 1) * (x + 5)

-- The theorem to be proven
theorem original_number_of_members (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 :=
by sorry

end original_number_of_members_l628_62892


namespace product_of_coordinates_of_D_l628_62895

theorem product_of_coordinates_of_D (Mx My Cx Cy Dx Dy : ℝ) (M : (Mx, My) = (4, 8)) (C : (Cx, Cy) = (5, 4)) 
  (midpoint : (Mx, My) = ((Cx + Dx) / 2, (Cy + Dy) / 2)) : (Dx * Dy) = 36 := 
by
  sorry

end product_of_coordinates_of_D_l628_62895


namespace youseff_blocks_l628_62857

theorem youseff_blocks (x : ℕ) 
  (H1 : (1 : ℚ) * x = (1/3 : ℚ) * x + 8) : 
  x = 12 := 
sorry

end youseff_blocks_l628_62857


namespace largest_fraction_l628_62830

theorem largest_fraction :
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  frac3 > frac1 ∧ frac3 > frac2 ∧ frac3 > frac4 ∧ frac3 > frac5 :=
by
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  sorry

end largest_fraction_l628_62830


namespace meetings_percentage_l628_62820

/-- Define the total work day in hours -/
def total_work_day_hours : ℕ := 10

/-- Define the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 60 -- 1 hour = 60 minutes

/-- Define the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 75

/-- Define the break duration in minutes -/
def break_minutes : ℕ := 30

/-- Define the effective work minutes -/
def effective_work_minutes : ℕ := (total_work_day_hours * 60) - break_minutes

/-- Define the total meeting minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- The percentage of the effective work day spent in meetings -/
def percent_meetings : ℕ := (total_meeting_minutes * 100) / effective_work_minutes

theorem meetings_percentage : percent_meetings = 24 := by
  sorry

end meetings_percentage_l628_62820


namespace problem_statement_l628_62811

def oper (x : ℕ) (w : ℕ) := (2^x) / (2^w)

theorem problem_statement : ∃ n : ℕ, oper (oper 4 2) n = 2 ↔ n = 3 :=
by sorry

end problem_statement_l628_62811


namespace solutions_of_quadratic_eq_l628_62836

theorem solutions_of_quadratic_eq : 
    {x : ℝ | x^2 - 3 * x = 0} = {0, 3} :=
sorry

end solutions_of_quadratic_eq_l628_62836


namespace pencils_to_sell_for_desired_profit_l628_62805

/-- Definitions based on the conditions provided in the problem. -/
def total_pencils : ℕ := 2000
def cost_per_pencil : ℝ := 0.20
def sell_price_per_pencil : ℝ := 0.40
def desired_profit : ℝ := 160
def total_cost : ℝ := total_pencils * cost_per_pencil

/-- The theorem considers all the conditions and asks to prove the number of pencils to sell -/
theorem pencils_to_sell_for_desired_profit : 
  (desired_profit + total_cost) / sell_price_per_pencil = 1400 :=
by 
  sorry

end pencils_to_sell_for_desired_profit_l628_62805


namespace number_of_distinct_intersections_l628_62865

theorem number_of_distinct_intersections :
  (∃ x y : ℝ, 9 * x^2 + 16 * y^2 = 16 ∧ 16 * x^2 + 9 * y^2 = 9) →
  (∀ x y₁ y₂ : ℝ, 9 * x^2 + 16 * y₁^2 = 16 ∧ 16 * x^2 + 9 * y₁^2 = 9 ∧
    9 * x^2 + 16 * y₂^2 = 16 ∧ 16 * x^2 + 9 * y₂^2 = 9 → y₁ = y₂) →
  (∃! p : ℝ × ℝ, 9 * p.1^2 + 16 * p.2^2 = 16 ∧ 16 * p.1^2 + 9 * p.2^2 = 9) :=
by
  sorry

end number_of_distinct_intersections_l628_62865


namespace stu_books_count_l628_62843

theorem stu_books_count (S : ℕ) (h1 : S + 4 * S = 45) : S = 9 := 
by
  sorry

end stu_books_count_l628_62843


namespace min_dot_product_on_hyperbola_l628_62833

theorem min_dot_product_on_hyperbola (x1 y1 x2 y2 : ℝ) 
  (hA : x1^2 - y1^2 = 2) 
  (hB : x2^2 - y2^2 = 2)
  (h_x1 : x1 > 0) 
  (h_x2 : x2 > 0) : 
  x1 * x2 + y1 * y2 ≥ 2 :=
sorry

end min_dot_product_on_hyperbola_l628_62833


namespace find_fx_when_x_positive_l628_62847

def isOddFunction {α : Type} [AddGroup α] [Neg α] (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)
variable (h_odd : isOddFunction f)
variable (h_neg : ∀ x : ℝ, x < 0 → f x = -x^2 + x)

theorem find_fx_when_x_positive : ∀ x : ℝ, x > 0 → f x = x^2 + x :=
by
  sorry

end find_fx_when_x_positive_l628_62847


namespace bowling_ball_weight_l628_62840

def weight_of_canoe : ℕ := 32
def weight_of_canoes (n : ℕ) := n * weight_of_canoe
def weight_of_bowling_balls (n : ℕ) := 128

theorem bowling_ball_weight :
  (128 / 5 : ℚ) = (weight_of_bowling_balls 5 / 5 : ℚ) :=
by
  -- Theorems and calculations would typically be carried out here
  sorry

end bowling_ball_weight_l628_62840


namespace yield_percentage_of_stock_l628_62869

noncomputable def annual_dividend (par_value : ℝ) : ℝ := 0.21 * par_value
noncomputable def market_price : ℝ := 210
noncomputable def yield_percentage (annual_dividend : ℝ) (market_price : ℝ) : ℝ :=
  (annual_dividend / market_price) * 100

theorem yield_percentage_of_stock (par_value : ℝ)
  (h_par_value : par_value = 100) :
  yield_percentage (annual_dividend par_value) market_price = 10 :=
by
  sorry

end yield_percentage_of_stock_l628_62869


namespace negation_of_prop_p_l628_62887

open Classical

theorem negation_of_prop_p:
  (¬ ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≤ 1 / 2) ↔ ∃ x : ℕ, x > 0 ∧ (1 / 2) ^ x > 1 / 2 := 
by
  sorry

end negation_of_prop_p_l628_62887


namespace price_increase_decrease_l628_62818

theorem price_increase_decrease (P : ℝ) (x : ℝ) (h : P > 0) :
  (P * (1 + x / 100) * (1 - x / 100) = 0.64 * P) → (x = 60) :=
by
  sorry

end price_increase_decrease_l628_62818


namespace common_ratio_of_infinite_geometric_series_l628_62854

theorem common_ratio_of_infinite_geometric_series 
  (a b : ℚ) 
  (h1 : a = 8 / 10) 
  (h2 : b = -6 / 15) 
  (h3 : b = a * r) : 
  r = -1 / 2 :=
by
  -- The proof goes here
  sorry

end common_ratio_of_infinite_geometric_series_l628_62854


namespace union_of_sets_l628_62873

def setA : Set ℝ := { x | -5 ≤ x ∧ x < 1 }
def setB : Set ℝ := { x | x ≤ 2 }

theorem union_of_sets : setA ∪ setB = { x | x ≤ 2 } :=
by sorry

end union_of_sets_l628_62873


namespace abs_neg_two_l628_62897

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l628_62897


namespace percentage_of_students_owning_cats_l628_62807

theorem percentage_of_students_owning_cats (dogs cats total : ℕ) (h_dogs : dogs = 45) (h_cats : cats = 75) (h_total : total = 500) : 
  (cats / total) * 100 = 15 :=
by
  sorry

end percentage_of_students_owning_cats_l628_62807


namespace smallest_value_a2_b2_c2_l628_62883

theorem smallest_value_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 4 * c = 120) : 
  a^2 + b^2 + c^2 ≥ 14400 / 29 :=
by sorry

end smallest_value_a2_b2_c2_l628_62883


namespace truck_gas_consumption_l628_62858

theorem truck_gas_consumption :
  ∀ (initial_gasoline total_distance remaining_gasoline : ℝ),
    initial_gasoline = 12 →
    total_distance = (2 * 5 + 2 + 2 * 2 + 6) →
    remaining_gasoline = 2 →
    (initial_gasoline - remaining_gasoline) ≠ 0 →
    (total_distance / (initial_gasoline - remaining_gasoline)) = 2.2 :=
by
  intros initial_gasoline total_distance remaining_gasoline
  intro h_initial_gas h_total_distance h_remaining_gas h_non_zero
  sorry

end truck_gas_consumption_l628_62858


namespace solve_oranges_problem_find_plans_and_max_profit_l628_62814

theorem solve_oranges_problem :
  ∃ (a b : ℕ), 15 * a + 20 * b = 430 ∧ 10 * a + 8 * b = 212 ∧ a = 10 ∧ b = 14 := by
    sorry

theorem find_plans_and_max_profit (a b : ℕ) (h₁ : 15 * a + 20 * b = 430) (h₂ : 10 * a + 8 * b = 212) (ha : a = 10) (hb : b = 14) :
  ∃ (x : ℕ), 58 ≤ x ∧ x ≤ 60 ∧ (10 * x + 14 * (100 - x) ≥ 1160) ∧ (10 * x + 14 * (100 - x) ≤ 1168) ∧ (1000 - 4 * x = 768) :=
    sorry

end solve_oranges_problem_find_plans_and_max_profit_l628_62814


namespace eval_polynomial_at_3_l628_62896

theorem eval_polynomial_at_3 : (3 : ℤ) ^ 3 + (3 : ℤ) ^ 2 + 3 + 1 = 40 := by
  sorry

end eval_polynomial_at_3_l628_62896


namespace log_change_of_base_log_change_of_base_with_b_l628_62802

variable {a b x : ℝ}
variable (h₁ : 0 < a ∧ a ≠ 1)
variable (h₂ : 0 < b ∧ b ≠ 1)
variable (h₃ : 0 < x)

theorem log_change_of_base (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) (h₃ : 0 < x) : 
  Real.log x / Real.log a = Real.log x / Real.log b := by
  sorry

theorem log_change_of_base_with_b (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) : 
  Real.log b / Real.log a = 1 / Real.log a := by
  sorry

end log_change_of_base_log_change_of_base_with_b_l628_62802


namespace find_d_l628_62866

theorem find_d (d : ℝ) (h : 4 * (3.6 * 0.48 * 2.50) / (d * 0.09 * 0.5) = 3200.0000000000005) : d = 0.3 :=
by
  sorry

end find_d_l628_62866


namespace hyperbola_equiv_l628_62849

-- The existing hyperbola
def hyperbola1 (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- The new hyperbola with same asymptotes passing through (2, 2) should have this form
def hyperbola2 (x y : ℝ) : Prop := (x^2 / 3 - y^2 / 12 = 1)

theorem hyperbola_equiv (x y : ℝ) :
  (hyperbola1 2 2) →
  (y^2 / 4 - x^2 / 4 = -3) →
  (hyperbola2 x y) :=
by
  intros h1 h2
  sorry

end hyperbola_equiv_l628_62849


namespace spiral_wire_length_l628_62825

noncomputable def wire_length (turns : ℕ) (height : ℝ) (circumference : ℝ) : ℝ :=
  Real.sqrt (height^2 + (turns * circumference)^2)

theorem spiral_wire_length
  (turns : ℕ) (height : ℝ) (circumference : ℝ)
  (turns_eq : turns = 10)
  (height_eq : height = 9)
  (circumference_eq : circumference = 4) :
  wire_length turns height circumference = 41 := 
by
  rw [turns_eq, height_eq, circumference_eq]
  simp [wire_length]
  norm_num
  rw [Real.sqrt_eq_rpow]
  norm_num
  sorry

end spiral_wire_length_l628_62825


namespace intersection_of_A_and_B_l628_62875

-- Define the set A as the solutions to the equation x^2 - 4 = 0
def A : Set ℝ := { x | x^2 - 4 = 0 }

-- Define the set B as the explicit set {1, 2}
def B : Set ℝ := {1, 2}

-- Prove that the intersection of sets A and B is {2}
theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  unfold A B
  sorry

end intersection_of_A_and_B_l628_62875


namespace roots_quadratic_sum_l628_62848

theorem roots_quadratic_sum (a b : ℝ) (h1 : (-2) + (-(1/4)) = -b/a)
  (h2 : -2 * (-(1/4)) = -2/a) : a + b = -13 := by
  sorry

end roots_quadratic_sum_l628_62848


namespace recommended_apps_l628_62832

namespace RogerPhone

-- Let's define the conditions.
def optimalApps : ℕ := 50
def currentApps (R : ℕ) : ℕ := 2 * R
def appsToDelete : ℕ := 20

-- Defining the problem as a theorem.
theorem recommended_apps (R : ℕ) (h1 : 2 * R = optimalApps + appsToDelete) : R = 35 := by
  sorry

end RogerPhone

end recommended_apps_l628_62832


namespace probability_dmitry_before_anatoly_l628_62891

theorem probability_dmitry_before_anatoly (m : ℝ) (non_neg_m : 0 < m) :
  let volume_prism := (m^3) / 2
  let volume_tetrahedron := (m^3) / 3
  let probability := volume_tetrahedron / volume_prism
  probability = (2 : ℝ) / 3 :=
by
  sorry

end probability_dmitry_before_anatoly_l628_62891


namespace correct_time_after_2011_minutes_l628_62841

def time_2011_minutes_after_midnight : String :=
  "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM"

theorem correct_time_after_2011_minutes :
  time_2011_minutes_after_midnight = "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM" :=
sorry

end correct_time_after_2011_minutes_l628_62841


namespace power_difference_expression_l628_62880

theorem power_difference_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * (30^1001) :=
by
  sorry

end power_difference_expression_l628_62880


namespace find_x_coordinate_l628_62831

theorem find_x_coordinate (m b x y : ℝ) (h1: m = 4) (h2: b = 100) (h3: y = 300) (line_eq: y = m * x + b) : x = 50 :=
by {
  sorry
}

end find_x_coordinate_l628_62831


namespace ratio_major_minor_is_15_4_l628_62835

-- Define the given conditions
def main_characters : ℕ := 5
def minor_characters : ℕ := 4
def minor_character_pay : ℕ := 15000
def total_payment : ℕ := 285000

-- Define the total pay to minor characters
def minor_total_pay : ℕ := minor_characters * minor_character_pay

-- Define the total pay to major characters
def major_total_pay : ℕ := total_payment - minor_total_pay

-- Define the ratio computation
def ratio_major_minor : ℕ × ℕ := (major_total_pay / 15000, minor_total_pay / 15000)

-- State the theorem
theorem ratio_major_minor_is_15_4 : ratio_major_minor = (15, 4) :=
by
  -- Proof goes here
  sorry

end ratio_major_minor_is_15_4_l628_62835


namespace seven_pow_l628_62813

theorem seven_pow (k : ℕ) (h : 7 ^ k = 2) : 7 ^ (4 * k + 2) = 784 :=
by 
  sorry

end seven_pow_l628_62813


namespace maximal_length_sequence_l628_62826

theorem maximal_length_sequence :
  ∃ (a : ℕ → ℤ) (n : ℕ), (∀ i, 1 ≤ i → i + 6 ≤ n → (a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) + a (i + 6) > 0)) ∧ 
                          (∀ j, 1 ≤ j → j + 10 ≤ n → (a j + a (j + 1) + a (j + 2) + a (j + 3) + a (j + 4) + a (j + 5) + a (j + 6) + a (j + 7) + a (j + 8) + a (j + 9) + a (j + 10) < 0)) ∧ 
                          n = 16 :=
sorry

end maximal_length_sequence_l628_62826


namespace ice_cream_eaten_l628_62842

variables (f : ℝ)

theorem ice_cream_eaten (h : f + 0.25 = 3.5) : f = 3.25 :=
sorry

end ice_cream_eaten_l628_62842


namespace range_of_a_l628_62823

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * Real.cos (Real.pi / 2 - x)

theorem range_of_a (a : ℝ) (h_condition : f (2 * a ^ 2) + f (a - 3) + f 0 < 0) : -3/2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l628_62823


namespace shuxue_count_l628_62828

theorem shuxue_count : 
  (∃ (count : ℕ), count = (List.length (List.filter (λ n => (30 * n.1 + 3 * n.2 < 100) 
    ∧ (30 * n.1 + 3 * n.2 > 9)) 
      (List.product 
        (List.range' 1 3) -- Possible values for "a" are 1 to 3
        (List.range' 1 9)) -- Possible values for "b" are 1 to 9
    ))) ∧ count = 9 :=
  sorry

end shuxue_count_l628_62828


namespace sequence_a2017_l628_62861

theorem sequence_a2017 (a : ℕ → ℚ) (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n / (3 * a n + 2)) :
  a 2017 = 1 / 3026 :=
sorry

end sequence_a2017_l628_62861


namespace cone_base_radius_l628_62864

theorem cone_base_radius (slant_height : ℝ) (central_angle_deg : ℝ) (r : ℝ) 
  (h1 : slant_height = 6) 
  (h2 : central_angle_deg = 120) 
  (h3 : 2 * π * slant_height * (central_angle_deg / 360) = 4 * π) 
  : r = 2 := by
  sorry

end cone_base_radius_l628_62864


namespace solution_of_equation_l628_62822

theorem solution_of_equation (a : ℝ) : (∃ x : ℝ, x = 4 ∧ (a * x - 3 = 4 * x + 1)) → a = 5 :=
by
  sorry

end solution_of_equation_l628_62822


namespace area_ratio_of_squares_l628_62824

theorem area_ratio_of_squares (s t : ℝ) (h : 4 * s = 4 * (4 * t)) : (s ^ 2) / (t ^ 2) = 16 :=
by
  sorry

end area_ratio_of_squares_l628_62824


namespace max_inscribed_circle_area_of_triangle_l628_62817

theorem max_inscribed_circle_area_of_triangle
  (a b : ℝ)
  (ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (f1 f2 : ℝ × ℝ)
  (F1_coords : f1 = (-1, 0))
  (F2_coords : f2 = (1, 0))
  (P Q : ℝ × ℝ)
  (line_through_F2 : ∀ y : ℝ, x = 1 → y^2 = 9 / 4)
  (P_coords : P = (1, 3/2))
  (Q_coords : Q = (1, -3/2))
  : (π * (3 / 4)^2 = 9 * π / 16) :=
  sorry

end max_inscribed_circle_area_of_triangle_l628_62817


namespace bacon_suggestion_l628_62846

theorem bacon_suggestion (x y : ℕ) (h1 : x = 479) (h2 : y = x + 10) : y = 489 := 
by {
  sorry
}

end bacon_suggestion_l628_62846


namespace find_angle_B_l628_62898

theorem find_angle_B
  (a : ℝ) (c : ℝ) (A B C : ℝ)
  (h1 : a = 5 * Real.sqrt 2)
  (h2 : c = 10)
  (h3 : A = π / 6) -- 30 degrees in radians
  (h4 : A + B + C = π) -- sum of angles in a triangle
  : B = 7 * π / 12 ∨ B = π / 12 := -- 105 degrees or 15 degrees in radians
sorry

end find_angle_B_l628_62898


namespace combined_vacations_and_classes_l628_62870

-- Define the conditions
def Kelvin_classes : ℕ := 90
def Grant_vacations : ℕ := 4 * Kelvin_classes

-- The Lean 4 statement proving the combined total of vacations and classes
theorem combined_vacations_and_classes : Kelvin_classes + Grant_vacations = 450 := by
  sorry

end combined_vacations_and_classes_l628_62870


namespace max_value_of_f_l628_62863

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_value_of_f (a : ℝ) (h : -2 < a ∧ a ≤ 0) : 
  ∀ x ∈ (Set.Icc 0 (a + 2)), f x ≤ 3 :=
sorry

end max_value_of_f_l628_62863


namespace final_pen_count_l628_62888

theorem final_pen_count
  (initial_pens : ℕ := 7) 
  (mike_given_pens : ℕ := 22) 
  (doubled_pens : ℕ := 2)
  (sharon_given_pens : ℕ := 19) :
  let total_after_mike := initial_pens + mike_given_pens
  let total_after_cindy := total_after_mike * doubled_pens
  let final_count := total_after_cindy - sharon_given_pens
  final_count = 39 :=
by
  sorry

end final_pen_count_l628_62888


namespace pure_alcohol_added_l628_62894

theorem pure_alcohol_added (x : ℝ) (h1 : 6 * 0.40 = 2.4)
    (h2 : (2.4 + x) / (6 + x) = 0.50) : x = 1.2 :=
by
  sorry

end pure_alcohol_added_l628_62894


namespace mowing_field_l628_62827

theorem mowing_field (x : ℝ) 
  (h1 : 1 / 84 + 1 / x = 1 / 21) : 
  x = 28 := 
sorry

end mowing_field_l628_62827


namespace range_of_a_given_quadratic_condition_l628_62885

theorem range_of_a_given_quadratic_condition:
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - 3 * a * x + 9 ≥ 0) → (-2 ≤ a ∧ a ≤ 2)) :=
by
  sorry

end range_of_a_given_quadratic_condition_l628_62885


namespace solve_system_l628_62881

theorem solve_system (x y : ℝ) (h1 : x^2 + y^2 + x + y = 50) (h2 : x * y = 20) :
  (x = 5 ∧ y = 4) ∨ (x = 4 ∧ y = 5) ∨ (x = -5 + Real.sqrt 5 ∧ y = -5 - Real.sqrt 5) ∨ (x = -5 - Real.sqrt 5 ∧ y = -5 + Real.sqrt 5) :=
by
  sorry

end solve_system_l628_62881


namespace solve_ordered_pair_l628_62834

theorem solve_ordered_pair : ∃ (x y : ℚ), 3*x - 24*y = 3 ∧ x - 3*y = 4 ∧ x = 29/5 ∧ y = 3/5 := by
  sorry

end solve_ordered_pair_l628_62834


namespace train_speed_before_accident_l628_62877

theorem train_speed_before_accident (d v : ℝ) (hv_pos : v > 0) (hd_pos : d > 0) :
  (d / ((3/4) * v) - d / v = 35 / 60) ∧
  (d - 24) / ((3/4) * v) - (d - 24) / v = 25 / 60 → 
  v = 64 :=
by
  sorry

end train_speed_before_accident_l628_62877


namespace doubled_marks_new_average_l628_62860

theorem doubled_marks_new_average (avg_marks : ℝ) (num_students : ℕ) (h_avg : avg_marks = 36) (h_num : num_students = 12) : 2 * avg_marks = 72 :=
by
  sorry

end doubled_marks_new_average_l628_62860


namespace prove_positive_a_l628_62801

variable (a b c n : ℤ)
variable (p : ℤ → ℤ)

-- Conditions given in the problem
def quadratic_polynomial (x : ℤ) : ℤ := a*x^2 + b*x + c

def condition_1 : Prop := a ≠ 0
def condition_2 : Prop := n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))

-- Proof goal
theorem prove_positive_a (h1 : a ≠ 0) (h2 : n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))) :
  0 < a :=
by
  sorry

end prove_positive_a_l628_62801


namespace cos_C_in_triangle_l628_62874

theorem cos_C_in_triangle 
  (A B C : ℝ) 
  (h_triangle : A + B + C = 180)
  (sin_A : Real.sin A = 4 / 5) 
  (cos_B : Real.cos B = 12 / 13) : 
  Real.cos C = -16 / 65 :=
by
  sorry

end cos_C_in_triangle_l628_62874


namespace exist_pairwise_distinct_gcd_l628_62800

theorem exist_pairwise_distinct_gcd (S : Set ℕ) (h_inf : S.Infinite) 
  (h_gcd : ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ gcd a b ≠ gcd c d) :
  ∃ x y z : ℕ, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x := 
by sorry

end exist_pairwise_distinct_gcd_l628_62800


namespace lines_intersect_first_quadrant_l628_62803

theorem lines_intersect_first_quadrant (k : ℝ) :
  (∃ (x y : ℝ), 2 * x + 7 * y = 14 ∧ k * x - y = k + 1 ∧ x > 0 ∧ y > 0) ↔ k > 0 :=
by
  sorry

end lines_intersect_first_quadrant_l628_62803


namespace packs_of_beef_l628_62821

noncomputable def pounds_per_pack : ℝ := 4
noncomputable def price_per_pound : ℝ := 5.50
noncomputable def total_paid : ℝ := 110
noncomputable def price_per_pack : ℝ := price_per_pound * pounds_per_pack

theorem packs_of_beef (n : ℝ) (h : n = total_paid / price_per_pack) : n = 5 := 
by
  sorry

end packs_of_beef_l628_62821


namespace line_intersects_circle_l628_62850

variable (x0 y0 R : ℝ)

theorem line_intersects_circle (h : x0^2 + y0^2 > R^2) :
  ∃ (x y : ℝ), (x^2 + y^2 = R^2) ∧ (x0 * x + y0 * y = R^2) :=
sorry

end line_intersects_circle_l628_62850


namespace digit_after_decimal_l628_62855

theorem digit_after_decimal (n : ℕ) : (n = 123) → (123 % 12 ≠ 0) → (123 % 12 = 3) → (∃ d : ℕ, d = 1 ∧ (43 / 740 : ℚ)^123 = 0 + d / 10^(123)) := 
by
    intros h₁ h₂ h₃
    sorry

end digit_after_decimal_l628_62855


namespace slant_height_of_cone_l628_62890

theorem slant_height_of_cone
  (r : ℝ) (CSA : ℝ) (l : ℝ)
  (hr : r = 14)
  (hCSA : CSA = 1539.3804002589986) :
  CSA = Real.pi * r * l → l = 35 := 
sorry

end slant_height_of_cone_l628_62890


namespace coefficient_of_determination_l628_62876

-- Define the observations and conditions for the problem
def observations (n : ℕ) := 
  {x : ℕ → ℝ // ∃ b a : ℝ, ∀ i : ℕ, i < n → ∃ y_i : ℝ, y_i = b * x i + a}

/-- 
  Given a set of observations (x_1, y_1), (x_2, y_2), ..., (x_n, y_n) 
  that satisfies the equation y_i = bx_i + a for i = 1, 2, ..., n, 
  prove that the coefficient of determination R² is 1.
-/
theorem coefficient_of_determination (n : ℕ) (obs : observations n) : 
  ∃ R_squared : ℝ, R_squared = 1 :=
sorry

end coefficient_of_determination_l628_62876


namespace candy_problem_l628_62806

variable (total_pieces_eaten : ℕ) (pieces_from_sister : ℕ) (pieces_from_neighbors : ℕ)

theorem candy_problem
  (h1 : total_pieces_eaten = 18)
  (h2 : pieces_from_sister = 13)
  (h3 : total_pieces_eaten = pieces_from_sister + pieces_from_neighbors) :
  pieces_from_neighbors = 5 := by
  -- Add proof here
  sorry

end candy_problem_l628_62806


namespace correct_answers_count_l628_62859

-- Define the conditions from the problem
def total_questions : ℕ := 25
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def total_score : ℤ := 85

-- State the theorem
theorem correct_answers_count :
  ∃ x : ℕ, (x ≤ total_questions) ∧ 
           (total_questions - x : ℕ) ≥ 0 ∧ 
           (correct_points * x + incorrect_points * (total_questions - x) = total_score) :=
sorry

end correct_answers_count_l628_62859


namespace root_increases_implies_m_neg7_l628_62853

theorem root_increases_implies_m_neg7 
  (m : ℝ) 
  (h : ∃ x : ℝ, x ≠ 3 ∧ x = -m - 4 → x = 3) 
  : m = -7 := by
  sorry

end root_increases_implies_m_neg7_l628_62853


namespace ExpandedOHaraTripleValue_l628_62872

/-- Define an Expanded O'Hara triple -/
def isExpandedOHaraTriple (a b x : ℕ) : Prop :=
  2 * (Nat.sqrt a + Nat.sqrt b) = x

/-- Prove that for given a=64 and b=49, x is equal to 30 if (a, b, x) is an Expanded O'Hara triple -/
theorem ExpandedOHaraTripleValue (a b x : ℕ) (ha : a = 64) (hb : b = 49) (h : isExpandedOHaraTriple a b x) : x = 30 :=
by
  sorry

end ExpandedOHaraTripleValue_l628_62872


namespace replaced_person_weight_l628_62819

theorem replaced_person_weight (W : ℝ) (increase : ℝ) (new_weight : ℝ) (average_increase : ℝ) (number_of_persons : ℕ) :
  average_increase = 2.5 →
  new_weight = 70 →
  number_of_persons = 8 →
  increase = number_of_persons * average_increase →
  W + increase = W - replaced_weight + new_weight →
  replaced_weight = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end replaced_person_weight_l628_62819


namespace divisible_by_five_solution_exists_l628_62844

theorem divisible_by_five_solution_exists
  (a b c d : ℤ)
  (h₀ : ∃ k : ℤ, d = 5 * k + d % 5 ∧ d % 5 ≠ 0)
  (h₁ : ∃ n : ℤ, (a * n^3 + b * n^2 + c * n + d) % 5 = 0) :
  ∃ m : ℤ, (a + b * m + c * m^2 + d * m^3) % 5 = 0 := 
sorry

end divisible_by_five_solution_exists_l628_62844


namespace nonneg_for_all_x_iff_a_in_range_l628_62804

def f (x a : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem nonneg_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end nonneg_for_all_x_iff_a_in_range_l628_62804


namespace new_boxes_of_markers_l628_62884

theorem new_boxes_of_markers (initial_markers new_markers_per_box total_markers : ℕ) 
    (h_initial : initial_markers = 32) 
    (h_new_markers_per_box : new_markers_per_box = 9)
    (h_total : total_markers = 86) :
  (total_markers - initial_markers) / new_markers_per_box = 6 := 
by
  sorry

end new_boxes_of_markers_l628_62884


namespace slope_probability_l628_62845

def line_equation (a x y : ℝ) : Prop := a * x + 2 * y - 3 = 0

def in_interval (a : ℝ) : Prop := -5 ≤ a ∧ a ≤ 4

def slope_not_less_than_1 (a : ℝ) : Prop := - a / 2 ≥ 1

noncomputable def probability_slope_not_less_than_1 : ℝ :=
  (2 - (-5)) / (4 - (-5))

theorem slope_probability :
  ∀ (a : ℝ), in_interval a → slope_not_less_than_1 a → probability_slope_not_less_than_1 = 1 / 3 :=
by
  intros a h_in h_slope
  sorry

end slope_probability_l628_62845


namespace mia_has_110_l628_62838

def darwin_money : ℕ := 45
def mia_money (darwin_money : ℕ) : ℕ := 2 * darwin_money + 20

theorem mia_has_110 :
  mia_money darwin_money = 110 := sorry

end mia_has_110_l628_62838


namespace area_of_square_plot_l628_62808

theorem area_of_square_plot (price_per_foot : ℕ) (total_cost : ℕ) (h_price : price_per_foot = 58) (h_cost : total_cost = 2088) :
  ∃ s : ℕ, s^2 = 81 := by
  sorry

end area_of_square_plot_l628_62808


namespace opposite_sides_line_l628_62886

theorem opposite_sides_line (m : ℝ) : 
  (2 * 1 + 3 + m) * (2 * -4 + -2 + m) < 0 ↔ -5 < m ∧ m < 10 :=
by sorry

end opposite_sides_line_l628_62886


namespace sufficient_not_necessary_condition_l628_62810

theorem sufficient_not_necessary_condition (x : ℝ) : (x ≥ 3 → (x - 2) ≥ 0) ∧ ((x - 2) ≥ 0 → x ≥ 3) = false :=
by
  sorry

end sufficient_not_necessary_condition_l628_62810


namespace matrix_power_difference_l628_62852

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 4;
     0, 1]

theorem matrix_power_difference :
  B^30 - 3 * B^29 = !![-2, 0;
                       0,  2] := 
by
  sorry

end matrix_power_difference_l628_62852


namespace smallest_integer_y_l628_62829

theorem smallest_integer_y (y : ℤ) : (5 : ℝ) / 8 < (y : ℝ) / 17 → y = 11 := by
  sorry

end smallest_integer_y_l628_62829


namespace opposite_face_of_lime_is_black_l628_62809

-- Define the colors
inductive Color
| P | C | M | S | K | L

-- Define the problem conditions
def face_opposite (c : Color) : Color := sorry

-- Theorem statement
theorem opposite_face_of_lime_is_black : face_opposite Color.L = Color.K := sorry

end opposite_face_of_lime_is_black_l628_62809


namespace probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l628_62856

-- Definitions and conditions from the problem
def success_probability_each_segment : ℚ := 3 / 4
def num_segments : ℕ := 4

-- Correct answers from the solution
def prob_both_success : ℚ := 9 / 16
def expected_successful_segments : ℚ := 3
def cond_prob_given_3_successful : ℚ := 3 / 4

theorem probability_both_segments_successful :
  (success_probability_each_segment * success_probability_each_segment) = prob_both_success :=
by
  sorry

theorem expected_number_of_successful_segments :
  (num_segments * success_probability_each_segment) = expected_successful_segments :=
by
  sorry

theorem probability_given_3_successful :
  let prob_M := 4 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  let prob_NM := 3 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  (prob_NM / prob_M) = cond_prob_given_3_successful :=
by
  sorry

end probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l628_62856


namespace parallel_lines_m_values_l628_62889

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (m-2) * x - y - 1 = 0) ∧ (∀ x y : ℝ, 3 * x - m * y = 0) → 
  (m = -1 ∨ m = 3) :=
by
  sorry

end parallel_lines_m_values_l628_62889


namespace Cagney_and_Lacey_Cupcakes_l628_62899

-- Conditions
def CagneyRate := 1 / 25 -- cupcakes per second
def LaceyRate := 1 / 35 -- cupcakes per second
def TotalTimeInSeconds := 10 * 60 -- total time in seconds
def LaceyPrepTimeInSeconds := 1 * 60 -- Lacey's preparation time in seconds
def EffectiveWorkTimeInSeconds := TotalTimeInSeconds - LaceyPrepTimeInSeconds -- effective working time

-- Calculate combined rate
def CombinedRate := 1 / (1 / CagneyRate + 1 / LaceyRate) -- combined rate in cupcakes per second

-- Calculate the total number of cupcakes frosted
def TotalCupcakesFrosted := EffectiveWorkTimeInSeconds * CombinedRate -- total cupcakes frosted

-- We state the theorem that corresponds to our proof problem
theorem Cagney_and_Lacey_Cupcakes : TotalCupcakesFrosted = 37 := by
  sorry

end Cagney_and_Lacey_Cupcakes_l628_62899


namespace system1_solution_exists_system2_solution_exists_l628_62851

-- System (1)
theorem system1_solution_exists (x y : ℝ) (h1 : y = 2 * x - 5) (h2 : 3 * x + 4 * y = 2) : 
  x = 2 ∧ y = -1 :=
by
  sorry

-- System (2)
theorem system2_solution_exists (x y : ℝ) (h1 : 3 * x - y = 8) (h2 : (y - 1) / 3 = (x + 5) / 5) : 
  x = 5 ∧ y = 7 :=
by
  sorry

end system1_solution_exists_system2_solution_exists_l628_62851


namespace sales_ratio_l628_62893

def large_price : ℕ := 60
def small_price : ℕ := 30
def last_month_large_paintings : ℕ := 8
def last_month_small_paintings : ℕ := 4
def this_month_sales : ℕ := 1200

theorem sales_ratio :
  (this_month_sales : ℕ) = 2 * (last_month_large_paintings * large_price + last_month_small_paintings * small_price) :=
by
  -- We will just state the proof steps as sorry.
  sorry

end sales_ratio_l628_62893
