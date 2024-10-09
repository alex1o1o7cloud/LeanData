import Mathlib

namespace minimum_b_value_l1598_159849

theorem minimum_b_value (k : ℕ) (x y z b : ℕ) (h1 : x = 3 * k) (h2 : y = 4 * k)
  (h3 : z = 7 * k) (h4 : y = 15 * b - 5) (h5 : ∀ n : ℕ, n = 4 * k + 5 → n % 15 = 0) : 
  b = 3 :=
by
  sorry

end minimum_b_value_l1598_159849


namespace arithmetic_sequence_a5_value_l1598_159806

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_cond : (a 5)^2 - a 3 - a 7 = 0) 
  : a 5 = 2 := 
sorry

end arithmetic_sequence_a5_value_l1598_159806


namespace tank_filling_time_l1598_159883

-- Define the rates at which pipes fill or drain the tank
def capacity : ℕ := 1200
def rate_A : ℕ := 50
def rate_B : ℕ := 35
def rate_C : ℕ := 20
def rate_D : ℕ := 40

-- Define the times each pipe is open
def time_A : ℕ := 2
def time_B : ℕ := 4
def time_C : ℕ := 3
def time_D : ℕ := 5

-- Calculate the total time for one cycle
def cycle_time : ℕ := time_A + time_B + time_C + time_D

-- Calculate the net amount of water added in one cycle
def net_amount_per_cycle : ℕ := (rate_A * time_A) + (rate_B * time_B) + (rate_C * time_C) - (rate_D * time_D)

-- Calculate the number of cycles needed to fill the tank
def num_cycles : ℕ := capacity / net_amount_per_cycle

-- Calculate the total time to fill the tank
def total_time : ℕ := num_cycles * cycle_time

-- Prove that the total time to fill the tank is 168 minutes
theorem tank_filling_time : total_time = 168 := by
  sorry

end tank_filling_time_l1598_159883


namespace min_value_expression_l1598_159836

theorem min_value_expression {x y : ℝ} : 
  2 * x + y - 3 = 0 →
  x + 2 * y - 1 = 0 →
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2) = 5 / 32 :=
by
  sorry

end min_value_expression_l1598_159836


namespace find_a_and_a100_l1598_159854

def seq (a : ℝ) (n : ℕ) : ℝ := (-1)^n * n + a

theorem find_a_and_a100 :
  ∃ a : ℝ, (seq a 1 + seq a 4 = 3 * seq a 2) ∧ (seq a 100 = 97) :=
by
  sorry

end find_a_and_a100_l1598_159854


namespace MrSlinkums_total_count_l1598_159803

variable (T : ℕ)

-- Defining the conditions as given in the problem
def placed_on_shelves (T : ℕ) : ℕ := (20 * T) / 100
def storage (T : ℕ) : ℕ := (80 * T) / 100

-- Stating the main theorem to prove
theorem MrSlinkums_total_count 
    (h : storage T = 120) : 
    T = 150 :=
sorry

end MrSlinkums_total_count_l1598_159803


namespace johns_contribution_l1598_159873

theorem johns_contribution (A : ℝ) (J : ℝ) : 
  (1.7 * A = 85) ∧ ((5 * A + J) / 6 = 85) → J = 260 := 
by
  sorry

end johns_contribution_l1598_159873


namespace range_of_a_l1598_159825

-- Define the function f and its derivative f'
def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * (a + 2)

-- We are given that for f to have both maximum and minimum values, f' must have two distinct roots
-- Thus we translate the mathematical condition to the discriminant of f' being greater than 0
def discriminant_greater_than_zero (a : ℝ) : Prop :=
  (6 * a)^2 - 4 * 3 * 3 * (a + 2) > 0

-- Finally, we want to prove that this simplifies to a condition on a
theorem range_of_a (a : ℝ) : discriminant_greater_than_zero a ↔ (a > 2 ∨ a < -1) :=
by
  -- Write the proof here
  sorry

end range_of_a_l1598_159825


namespace find_totally_damaged_cartons_l1598_159811

def jarsPerCarton : ℕ := 20
def initialCartons : ℕ := 50
def reducedCartons : ℕ := 30
def damagedJarsPerCarton : ℕ := 3
def damagedCartons : ℕ := 5
def totalGoodJars : ℕ := 565

theorem find_totally_damaged_cartons :
  (initialCartons * jarsPerCarton - ((initialCartons - reducedCartons) * jarsPerCarton + damagedJarsPerCarton * damagedCartons - totalGoodJars)) / jarsPerCarton = 1 := by
  sorry

end find_totally_damaged_cartons_l1598_159811


namespace cos_theta_is_correct_l1598_159851

def vector_1 : ℝ × ℝ := (4, 5)
def vector_2 : ℝ × ℝ := (2, 7)

noncomputable def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1 * v1.1 + v1.2 * v1.2) * Real.sqrt (v2.1 * v2.1 + v2.2 * v2.2))

theorem cos_theta_is_correct :
  cos_theta vector_1 vector_2 = 43 / (Real.sqrt 41 * Real.sqrt 53) :=
by
  -- proof goes here
  sorry

end cos_theta_is_correct_l1598_159851


namespace net_income_difference_l1598_159885

theorem net_income_difference
    (terry_daily_income : ℝ := 24) (terry_daily_hours : ℝ := 6) (terry_days : ℕ := 7)
    (jordan_daily_income : ℝ := 30) (jordan_daily_hours : ℝ := 8) (jordan_days : ℕ := 6)
    (standard_week_hours : ℝ := 40) (overtime_rate_multiplier : ℝ := 1.5)
    (terry_tax_rate : ℝ := 0.12) (jordan_tax_rate : ℝ := 0.15) :
    jordan_daily_income * jordan_days - jordan_daily_income * jordan_days * jordan_tax_rate 
      + jordan_daily_income * jordan_days * jordan_daily_hours * (overtime_rate_multiplier - 1) * jordan_tax_rate
    - (terry_daily_income * terry_days - terry_daily_income * terry_days * terry_tax_rate 
      + terry_daily_income * terry_days * terry_daily_hours * (overtime_rate_multiplier - 1) * terry_tax_rate) 
      = 32.85 := 
sorry

end net_income_difference_l1598_159885


namespace steps_to_get_down_empire_state_building_l1598_159823

theorem steps_to_get_down_empire_state_building (total_steps : ℕ) (steps_building_to_garden : ℕ) (steps_to_madison_square : ℕ) :
  total_steps = 991 -> steps_building_to_garden = 315 -> steps_to_madison_square = total_steps - steps_building_to_garden -> steps_to_madison_square = 676 :=
by
  intros
  subst_vars
  sorry

end steps_to_get_down_empire_state_building_l1598_159823


namespace puppy_weight_l1598_159881

variable (p s l r : ℝ)

theorem puppy_weight :
  p + s + l + r = 40 ∧ 
  p^2 + l^2 = 4 * s ∧ 
  p^2 + s^2 = l^2 → 
  p = Real.sqrt 2 :=
sorry

end puppy_weight_l1598_159881


namespace inequality_sum_l1598_159888

variable {a b c d : ℝ}

theorem inequality_sum (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by {
  sorry
}

end inequality_sum_l1598_159888


namespace correlation_comparison_l1598_159875

/-- The data for variables x and y are (1, 3), (2, 5.3), (3, 6.9), (4, 9.1), and (5, 10.8) -/
def xy_data : List (Int × Float) := [(1, 3), (2, 5.3), (3, 6.9), (4, 9.1), (5, 10.8)]

/-- The data for variables U and V are (1, 12.7), (2, 10.2), (3, 7), (4, 3.6), and (5, 1) -/
def UV_data : List (Int × Float) := [(1, 12.7), (2, 10.2), (3, 7), (4, 3.6), (5, 1)]

/-- r1 is the linear correlation coefficient between y and x -/
noncomputable def r1 : Float := sorry

/-- r2 is the linear correlation coefficient between V and U -/
noncomputable def r2 : Float := sorry

/-- The problem is to prove that r2 < 0 < r1 given the data conditions -/
theorem correlation_comparison : r2 < 0 ∧ 0 < r1 := 
by 
  sorry

end correlation_comparison_l1598_159875


namespace milo_running_distance_l1598_159898

theorem milo_running_distance
  (run_speed skateboard_speed cory_speed : ℕ)
  (h1 : skateboard_speed = 2 * run_speed)
  (h2 : cory_speed = 2 * skateboard_speed)
  (h3 : cory_speed = 12) :
  run_speed * 2 = 6 :=
by
  sorry

end milo_running_distance_l1598_159898


namespace mark_exceeded_sugar_intake_by_100_percent_l1598_159877

-- Definitions of the conditions
def softDrinkCalories : ℕ := 2500
def sugarPercentage : ℝ := 0.05
def caloriesPerCandy : ℕ := 25
def numCandyBars : ℕ := 7
def recommendedSugarIntake : ℕ := 150

-- Calculating the amount of added sugar in the soft drink
def addedSugarSoftDrink : ℝ := sugarPercentage * softDrinkCalories

-- Calculating the total added sugar from the candy bars
def addedSugarCandyBars : ℕ := numCandyBars * caloriesPerCandy

-- Summing the added sugar from the soft drink and the candy bars
def totalAddedSugar : ℝ := addedSugarSoftDrink + (addedSugarCandyBars : ℝ)

-- Calculate the excess intake of added sugar over the recommended amount
def excessSugarIntake : ℝ := totalAddedSugar - (recommendedSugarIntake : ℝ)

-- Prove that the percentage by which Mark exceeded the recommended intake of added sugar is 100%
theorem mark_exceeded_sugar_intake_by_100_percent :
  (excessSugarIntake / (recommendedSugarIntake : ℝ)) * 100 = 100 :=
by
  sorry

end mark_exceeded_sugar_intake_by_100_percent_l1598_159877


namespace evaluate_f_l1598_159810

def f (x : ℝ) : ℝ := sorry  -- Placeholder function definition

theorem evaluate_f :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + 5/2) = -1 / f x) ∧
  (∀ x : ℝ, x ∈ [-5/2, 0] → f x = x * (x + 5/2))
  → f 2016 = 3/2 :=
by
  sorry

end evaluate_f_l1598_159810


namespace max_dot_and_area_of_triangle_l1598_159820

noncomputable def triangle_data (A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (m = (2, 2 * (Real.cos ((B + C) / 2))^2 - 1)) ∧
  (n = (Real.sin (A / 2), -1))

noncomputable def is_max_dot_product (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = (if A = Real.pi / 3 then 3 / 2 else 0)

noncomputable def max_area (A B C : ℝ) : ℝ :=
  let a : ℝ := 2
  let b : ℝ := 2
  let c : ℝ := 2
  if A = Real.pi / 3 then (Real.sqrt 3) else 0

theorem max_dot_and_area_of_triangle {A B C : ℝ} {m n : ℝ × ℝ}
  (h_triangle : triangle_data A B C m n) :
  is_max_dot_product (Real.pi / 3) m n ∧ max_area A B C = Real.sqrt 3 := by sorry

end max_dot_and_area_of_triangle_l1598_159820


namespace original_number_is_7_l1598_159879

theorem original_number_is_7 (x : ℤ) (h : (((3 * (x + 3) + 3) - 3) / 3) = 10) : x = 7 :=
sorry

end original_number_is_7_l1598_159879


namespace sum_first_four_terms_geo_seq_l1598_159805

theorem sum_first_four_terms_geo_seq (q : ℝ) (a_1 : ℝ)
  (h1 : q ≠ 1) 
  (h2 : a_1 * (a_1 * q) * (a_1 * q^2) = -1/8)
  (h3 : 2 * (a_1 * q^3) = (a_1 * q) + (a_1 * q^2)) :
  (a_1 + (a_1 * q) + (a_1 * q^2) + (a_1 * q^3)) = 5 / 8 :=
  sorry

end sum_first_four_terms_geo_seq_l1598_159805


namespace find_number_eq_l1598_159880

theorem find_number_eq (x : ℝ) (h : (35 / 100) * x = (20 / 100) * 40) : x = 160 / 7 :=
by
  sorry

end find_number_eq_l1598_159880


namespace combined_gold_cost_l1598_159863

def gary_gold_weight : ℕ := 30
def gary_gold_cost_per_gram : ℕ := 15
def anna_gold_weight : ℕ := 50
def anna_gold_cost_per_gram : ℕ := 20

theorem combined_gold_cost : (gary_gold_weight * gary_gold_cost_per_gram) + (anna_gold_weight * anna_gold_cost_per_gram) = 1450 :=
by {
  sorry -- Proof goes here
}

end combined_gold_cost_l1598_159863


namespace find_g4_l1598_159819

noncomputable def g : ℝ → ℝ := sorry

theorem find_g4 (h : ∀ x y : ℝ, x * g y = 2 * y * g x) (h₁ : g 10 = 5) : g 4 = 4 :=
sorry

end find_g4_l1598_159819


namespace initial_ratio_l1598_159891

variables {p q : ℝ}

theorem initial_ratio (h₁ : p + q = 20) (h₂ : p / (q + 1) = 4 / 3) : p / q = 3 / 2 :=
sorry

end initial_ratio_l1598_159891


namespace prize_distribution_l1598_159845

theorem prize_distribution (x y z : ℕ) (h₁ : 15000 * x + 10000 * y + 5000 * z = 1000000) (h₂ : 93 ≤ z - x) (h₃ : z - x < 96) :
  x + y + z = 147 :=
sorry

end prize_distribution_l1598_159845


namespace length_of_bridge_is_230_l1598_159841

noncomputable def train_length : ℚ := 145
noncomputable def train_speed_kmh : ℚ := 45
noncomputable def time_to_cross_bridge : ℚ := 30
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 1000) / 3600
noncomputable def bridge_length : ℚ := (train_speed_ms * time_to_cross_bridge) - train_length

theorem length_of_bridge_is_230 :
  bridge_length = 230 :=
sorry

end length_of_bridge_is_230_l1598_159841


namespace parabola_tangent_to_hyperbola_l1598_159893

theorem parabola_tangent_to_hyperbola (m : ℝ) :
  (∀ x y : ℝ, y = x^2 + 4 → y^2 - m * x^2 = 4) ↔ m = 8 := 
sorry

end parabola_tangent_to_hyperbola_l1598_159893


namespace remainder_of_x_plus_2_power_2008_l1598_159887

-- Given: x^3 ≡ 1 (mod x^2 + x + 1)
def given_condition : Prop := ∀ x : ℤ, (x^3 - 1) % (x^2 + x + 1) = 0

-- To prove: The remainder when (x + 2)^2008 is divided by x^2 + x + 1 is 1
theorem remainder_of_x_plus_2_power_2008 (x : ℤ) (h : given_condition) :
  ((x + 2) ^ 2008) % (x^2 + x + 1) = 1 := by
  sorry

end remainder_of_x_plus_2_power_2008_l1598_159887


namespace trigonometric_equation_solution_l1598_159842

theorem trigonometric_equation_solution (n : ℕ) (h_pos : 0 < n) (x : ℝ) (hx1 : ∀ k : ℤ, x ≠ k * π / 2) :
  (1 / (Real.sin x)^(2 * n) + 1 / (Real.cos x)^(2 * n) = 2^(n + 1)) ↔ ∃ k : ℤ, x = (2 * k + 1) * π / 4 :=
by sorry

end trigonometric_equation_solution_l1598_159842


namespace sequence_5th_term_l1598_159868

theorem sequence_5th_term (a b c : ℚ) (h1 : a = 1 / 4 * (4 + b)) (h2 : b = 1 / 4 * (a + 40)) (h3 : 40 = 1 / 4 * (b + c)) : 
  c = 2236 / 15 := 
by 
  sorry

end sequence_5th_term_l1598_159868


namespace correct_formula_l1598_159896

def table : List (ℕ × ℕ) :=
    [(1, 3), (2, 8), (3, 15), (4, 24), (5, 35)]

theorem correct_formula : ∀ x y, (x, y) ∈ table → y = x^2 + 4 * x + 3 :=
by
  intros x y H
  sorry

end correct_formula_l1598_159896


namespace proof_problem_l1598_159832

-- Define the propositions as Lean terms
def prop1 : Prop := ∀ (l1 l2 : ℝ) (h1 : l1 ≠ 0 ∧ l2 ≠ 0), (l1 * l2 = -1) → (l1 ≠ l2)  -- Two perpendicular lines must intersect (incorrect definition)
def prop2 : Prop := ∀ (l : ℝ), ∃! (m : ℝ), (l * m = -1)  -- There is only one perpendicular line (incorrect definition)
def prop3 : Prop := (∀ (α β γ : ℝ), α = β → γ = 90 → α + γ = β + γ)  -- Equal corresponding angles when intersecting a third (incorrect definition)
def prop4 : Prop := ∀ (A B C : ℝ), (A = B ∧ B = C) → (A = C)  -- Transitive property of parallel lines

-- The statement that only one of these propositions is true, and it is the fourth one
theorem proof_problem (h1 : ¬ prop1) (h2 : ¬ prop2) (h3 : ¬ prop3) (h4 : prop4) : 
  ∃! (i : ℕ), i = 4 := 
by
  sorry

end proof_problem_l1598_159832


namespace extended_pattern_ratio_l1598_159817

noncomputable def original_black_tiles : ℕ := 12
noncomputable def original_white_tiles : ℕ := 24
noncomputable def original_total_tiles : ℕ := 36
noncomputable def extended_total_tiles : ℕ := 64
noncomputable def border_black_tiles : ℕ := 24 /- The new border adds 24 black tiles -/
noncomputable def extended_black_tiles : ℕ := 36
noncomputable def extended_white_tiles := original_white_tiles

theorem extended_pattern_ratio :
  (extended_black_tiles : ℚ) / extended_white_tiles = 3 / 2 :=
by
  sorry

end extended_pattern_ratio_l1598_159817


namespace find_a_l1598_159878

noncomputable def lines_perpendicular (a : ℝ) (l1: ℝ × ℝ × ℝ) (l2: ℝ × ℝ × ℝ) : Prop :=
  let (A1, B1, C1) := l1
  let (A2, B2, C2) := l2
  (B1 ≠ 0) ∧ (B2 ≠ 0) ∧ (-A1 / B1) * (-A2 / B2) = -1

theorem find_a (a : ℝ) :
  lines_perpendicular a (a, 1, 1) (2*a, a - 3, 1) → a = 1 ∨ a = -3/2 :=
by
  sorry

end find_a_l1598_159878


namespace polynomial_integer_root_l1598_159808

theorem polynomial_integer_root (b : ℤ) :
  (∃ x : ℤ, x^3 + 5 * x^2 + b * x + 9 = 0) ↔ b = -127 ∨ b = -74 ∨ b = -27 ∨ b = -24 ∨ b = -15 ∨ b = -13 :=
by
  sorry

end polynomial_integer_root_l1598_159808


namespace calculate_value_l1598_159846

-- Definition of the given values
def val1 : ℕ := 444
def val2 : ℕ := 44
def val3 : ℕ := 4

-- Theorem statement proving the value of the expression
theorem calculate_value : (val1 - val2 - val3) = 396 := 
by 
  sorry

end calculate_value_l1598_159846


namespace pocket_knife_worth_40_l1598_159814

def value_of_pocket_knife (x : ℕ) (p : ℕ) (R : ℕ) : Prop :=
  p = 10 * x ∧
  R = 10 * x^2 ∧
  (∃ num_100_bills : ℕ, 2 * num_100_bills * 100 + 40 = R)

theorem pocket_knife_worth_40 (x : ℕ) (p : ℕ) (R : ℕ) :
  value_of_pocket_knife x p R → (∃ knife_value : ℕ, knife_value = 40) :=
by
  sorry

end pocket_knife_worth_40_l1598_159814


namespace elderly_people_pears_l1598_159890

theorem elderly_people_pears (x y : ℕ) :
  (y = x + 1) ∧ (2 * x = y + 2) ↔
  (x = y - 1) ∧ (2 * x = y + 2) := by
  sorry

end elderly_people_pears_l1598_159890


namespace thirty_six_forty_five_nine_eighteen_l1598_159859

theorem thirty_six_forty_five_nine_eighteen :
  18 * 36 + 45 * 18 - 9 * 18 = 1296 :=
by
  sorry

end thirty_six_forty_five_nine_eighteen_l1598_159859


namespace gray_region_area_l1598_159821

noncomputable def area_of_gray_region (length width : ℝ) (angle_deg : ℝ) : ℝ :=
  if (length = 55 ∧ width = 44 ∧ angle_deg = 45) then 10 else 0

theorem gray_region_area :
  area_of_gray_region 55 44 45 = 10 :=
by sorry

end gray_region_area_l1598_159821


namespace find_sum_of_roots_l1598_159889

open Real

theorem find_sum_of_roots (p q r s : ℝ): 
  r + s = 12 * p →
  r * s = 13 * q →
  p + q = 12 * r →
  p * q = 13 * s →
  p ≠ r →
  p + q + r + s = 2028 := by
  intros
  sorry

end find_sum_of_roots_l1598_159889


namespace balloons_given_by_mom_l1598_159828

-- Definitions of the initial and total number of balloons
def initial_balloons := 26
def total_balloons := 60

-- Theorem: Proving the number of balloons Tommy's mom gave him
theorem balloons_given_by_mom : total_balloons - initial_balloons = 34 :=
by
  -- This proof is obvious from the setup, so we write sorry to skip the proof.
  sorry

end balloons_given_by_mom_l1598_159828


namespace quadratic_inequality_solution_l1598_159815

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 7*x + 6 < 0 } = { x : ℝ | -6 < x ∧ x < -1 } :=
by sorry

end quadratic_inequality_solution_l1598_159815


namespace tim_youth_comparison_l1598_159872

theorem tim_youth_comparison :
  let Tim_age : ℕ := 5
  let Rommel_age : ℕ := 3 * Tim_age
  let Jenny_age : ℕ := Rommel_age + 2
  Jenny_age - Tim_age = 12 := 
by 
  sorry

end tim_youth_comparison_l1598_159872


namespace max_perimeter_convex_quadrilateral_l1598_159844

theorem max_perimeter_convex_quadrilateral :
  ∃ (AB BC AD CD AC BD : ℝ), 
    AB = 1 ∧ BC = 1 ∧
    AD ≤ 1 ∧ CD ≤ 1 ∧ AC ≤ 1 ∧ BD ≤ 1 ∧
    2 + 4 * Real.sin (Real.pi / 12) = 
      AB + BC + AD + CD :=
sorry

end max_perimeter_convex_quadrilateral_l1598_159844


namespace no_such_real_x_exists_l1598_159861

theorem no_such_real_x_exists :
  ¬ ∃ (x : ℝ), ⌊ x ⌋ + ⌊ 2 * x ⌋ + ⌊ 4 * x ⌋ + ⌊ 8 * x ⌋ + ⌊ 16 * x ⌋ + ⌊ 32 * x ⌋ = 12345 := 
sorry

end no_such_real_x_exists_l1598_159861


namespace find_ending_number_divisible_by_eleven_l1598_159813

theorem find_ending_number_divisible_by_eleven (start n end_num : ℕ) (h1 : start = 29) (h2 : n = 5) (h3 : ∀ k : ℕ, ∃ m : ℕ, m = start + k * 11) : end_num = 77 :=
sorry

end find_ending_number_divisible_by_eleven_l1598_159813


namespace rainfall_second_week_value_l1598_159865

-- Define the conditions
variables (rainfall_first_week rainfall_second_week : ℝ)
axiom condition1 : rainfall_first_week + rainfall_second_week = 30
axiom condition2 : rainfall_second_week = 1.5 * rainfall_first_week

-- Define the theorem we want to prove
theorem rainfall_second_week_value : rainfall_second_week = 18 := by
  sorry

end rainfall_second_week_value_l1598_159865


namespace area_of_inscribed_square_l1598_159837

theorem area_of_inscribed_square (a : ℝ) : 
    ∃ S : ℝ, S = 3 * a^2 / (7 - 4 * Real.sqrt 3) :=
by
  sorry

end area_of_inscribed_square_l1598_159837


namespace parabola_directrix_l1598_159894

theorem parabola_directrix (y x : ℝ) (h : y = x^2) : 4 * y + 1 = 0 :=
sorry

end parabola_directrix_l1598_159894


namespace problem_1_problem_2_l1598_159850

-- Define the sets M and N as conditions and include a > 0 condition.
def M (a : ℝ) : Set ℝ := {x : ℝ | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x : ℝ | 4 * x ^ 2 - 4 * x - 3 < 0}

-- Problem 1: Prove that a = 2 given the set conditions.
theorem problem_1 (a : ℝ) (h_pos : a > 0) :
  M a ∪ N = {x : ℝ | -2 ≤ x ∧ x < 3 / 2} → a = 2 :=
sorry

-- Problem 2: Prove the range of a is 0 < a ≤ 1 / 2 given the set conditions.
theorem problem_2 (a : ℝ) (h_pos : a > 0) :
  N ∪ (compl (M a)) = Set.univ → 0 < a ∧ a ≤ 1 / 2 :=
sorry

end problem_1_problem_2_l1598_159850


namespace parabola_line_intersect_solutions_count_l1598_159862

theorem parabola_line_intersect_solutions_count :
  ∃ b1 b2 : ℝ, (b1 ≠ b2 ∧ (b1^2 - b1 - 3 = 0) ∧ (b2^2 - b2 - 3 = 0)) :=
by
  sorry

end parabola_line_intersect_solutions_count_l1598_159862


namespace proof_problem_l1598_159816

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem proof_problem (x1 x2 : ℝ) (h₁ : x1 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₂ : x2 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₃ : f x1 + f x2 > 0) : 
  x1 + x2 > 0 :=
sorry

end proof_problem_l1598_159816


namespace calculation_result_l1598_159800

theorem calculation_result :
  (2 : ℝ)⁻¹ - (1 / 2 : ℝ)^0 + (2 : ℝ)^2023 * (-0.5 : ℝ)^2023 = -3 / 2 := sorry

end calculation_result_l1598_159800


namespace simplify_expression_l1598_159871

theorem simplify_expression :
  (18 / 17) * (13 / 24) * (68 / 39) = 1 := 
by
  sorry

end simplify_expression_l1598_159871


namespace range_AD_dot_BC_l1598_159843

noncomputable def vector_dot_product_range (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : ℝ :=
  let ab := 2
  let ac := 1
  let bc := ac - ab
  let ad := x * ac + (1 - x) * ab
  ad * bc

theorem range_AD_dot_BC : 
  ∃ (a b : ℝ), vector_dot_product_range x h1 h2 = a ∧ ∀ (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1), a ≤ vector_dot_product_range x h1 h2 ∧ vector_dot_product_range x h1 h2 ≤ b :=
sorry

end range_AD_dot_BC_l1598_159843


namespace initial_range_without_telescope_l1598_159886

variable (V : ℝ)

def telescope_increases_range (V : ℝ) : Prop :=
  V + 0.875 * V = 150

theorem initial_range_without_telescope (V : ℝ) (h : telescope_increases_range V) : V = 80 :=
by
  sorry

end initial_range_without_telescope_l1598_159886


namespace billy_can_play_l1598_159857

-- Define the conditions
def total_songs : ℕ := 52
def songs_to_learn : ℕ := 28

-- Define the statement to be proved
theorem billy_can_play : total_songs - songs_to_learn = 24 := by
  -- Proof goes here
  sorry

end billy_can_play_l1598_159857


namespace union_M_N_l1598_159858

-- Define the set M
def M : Set ℤ := {x | x^2 - x = 0}

-- Define the set N
def N : Set ℤ := {y | y^2 + y = 0}

-- Prove that the union of M and N is {-1, 0, 1}
theorem union_M_N :
  M ∪ N = {-1, 0, 1} :=
by
  sorry

end union_M_N_l1598_159858


namespace trig_identity_evaluation_l1598_159884

theorem trig_identity_evaluation :
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end trig_identity_evaluation_l1598_159884


namespace total_pages_l1598_159838

-- Conditions
variables (B1 B2 : ℕ)
variable (h1 : (2 / 3 : ℚ) * B1 - (1 / 3 : ℚ) * B1 = 90)
variable (h2 : (3 / 4 : ℚ) * B2 - (1 / 4 : ℚ) * B2 = 120)

-- Theorem statement
theorem total_pages (B1 B2 : ℕ) (h1 : (2 / 3 : ℚ) * B1 - (1 / 3 : ℚ) * B1 = 90) (h2 : (3 / 4 : ℚ) * B2 - (1 / 4 : ℚ) * B2 = 120) :
  B1 + B2 = 510 :=
sorry

end total_pages_l1598_159838


namespace perfect_square_a_i_l1598_159855

theorem perfect_square_a_i (a : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n + 2) = 18 * a (n + 1) - a n) :
  ∀ i, ∃ k, 5 * (a i) ^ 2 - 1 = k ^ 2 :=
by
  -- The proof is missing the skipped definitions from the problem and solution context
  sorry

end perfect_square_a_i_l1598_159855


namespace solve_problem_l1598_159830

theorem solve_problem :
  ∃ (x y : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y = 5 :=
by
  sorry

end solve_problem_l1598_159830


namespace triangle_fraction_correct_l1598_159833

def point : Type := ℤ × ℤ

def area_triangle (A B C : point) : ℚ :=
  (1 / 2 : ℚ) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℚ))

def area_grid (length width : ℚ) : ℚ :=
  length * width

noncomputable def fraction_covered (A B C : point) (grid_length grid_width : ℚ) : ℚ :=
  area_triangle A B C / area_grid grid_length grid_width

theorem triangle_fraction_correct :
  fraction_covered (-2, 3) (2, -2) (3, 5) 8 6 = 11 / 32 :=
by
  sorry

end triangle_fraction_correct_l1598_159833


namespace negation_of_exists_proposition_l1598_159831

theorem negation_of_exists_proposition :
  ¬ (∃ x₀ : ℝ, x₀^2 - 1 < 0) ↔ ∀ x : ℝ, x^2 - 1 ≥ 0 :=
by
  sorry

end negation_of_exists_proposition_l1598_159831


namespace votes_cast_l1598_159869

theorem votes_cast (V : ℝ) (h1 : 0.35 * V + 2250 = 0.65 * V) : V = 7500 := 
by
  sorry

end votes_cast_l1598_159869


namespace find_b_l1598_159899

noncomputable def h (x : ℝ) : ℝ := x^2 + 9
noncomputable def j (x : ℝ) : ℝ := x^2 + 1

theorem find_b (b : ℝ) (hjb : h (j b) = 15) (b_pos : b > 0) : b = Real.sqrt (Real.sqrt 6 - 1) := by
  sorry

end find_b_l1598_159899


namespace min_value_expr_l1598_159834

theorem min_value_expr (x y : ℝ) : ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + x * y + y^2 ≥ m) ∧ m = 0 :=
by
  sorry

end min_value_expr_l1598_159834


namespace tree_original_height_l1598_159882

theorem tree_original_height (current_height_in: ℝ) (growth_percentage: ℝ)
  (h1: current_height_in = 180) (h2: growth_percentage = 0.50) :
  ∃ (original_height_ft: ℝ), original_height_ft = 10 :=
by
  have original_height_in := current_height_in / (1 + growth_percentage)
  have original_height_ft := original_height_in / 12
  use original_height_ft
  sorry

end tree_original_height_l1598_159882


namespace digit_sum_2001_not_perfect_square_l1598_159840

theorem digit_sum_2001_not_perfect_square (n : ℕ) (h : (n.digits 10).sum = 2001) : ¬ ∃ k : ℕ, n = k * k := 
sorry

end digit_sum_2001_not_perfect_square_l1598_159840


namespace total_games_played_l1598_159835

theorem total_games_played (won_games : ℕ) (won_ratio : ℕ) (lost_ratio : ℕ) (tied_ratio : ℕ) (total_games : ℕ) :
  won_games = 42 →
  won_ratio = 7 →
  lost_ratio = 4 →
  tied_ratio = 5 →
  total_games = won_games + lost_ratio * (won_games / won_ratio) + tied_ratio * (won_games / won_ratio) →
  total_games = 96 :=
by
  intros h_won h_won_ratio h_lost_ratio h_tied_ratio h_total
  sorry

end total_games_played_l1598_159835


namespace volume_of_intersection_of_two_perpendicular_cylinders_l1598_159822

theorem volume_of_intersection_of_two_perpendicular_cylinders (R : ℝ) : 
  ∃ V : ℝ, V = (16 / 3) * R^3 := 
sorry

end volume_of_intersection_of_two_perpendicular_cylinders_l1598_159822


namespace probability_red_white_green_probability_any_order_l1598_159826

-- Definitions based on the conditions
def total_balls := 28
def red_balls := 15
def white_balls := 9
def green_balls := 4

-- Part (a): Probability of first red, second white, third green
theorem probability_red_white_green : 
  (red_balls / total_balls) * (white_balls / (total_balls - 1)) * (green_balls / (total_balls - 2)) = 5 / 182 :=
by 
  sorry

-- Part (b): Probability of red, white, and green in any order
theorem probability_any_order :
  6 * ((red_balls / total_balls) * (white_balls / (total_balls - 1)) * (green_balls / (total_balls - 2))) = 15 / 91 :=
by
  sorry

end probability_red_white_green_probability_any_order_l1598_159826


namespace find_x_l1598_159809

variable (x y : ℚ)

-- Condition
def condition : Prop :=
  (x / (x - 2)) = ((y^3 + 3 * y - 2) / (y^3 + 3 * y - 5))

-- Assertion to prove
theorem find_x (h : condition x y) : x = ((2 * y^3 + 6 * y - 4) / 3) :=
sorry

end find_x_l1598_159809


namespace min_value_of_expression_l1598_159856

theorem min_value_of_expression (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
    (h1 : ∀ n, S_n n = (4/3) * (a_n n - 1)) :
  ∃ (n : ℕ), (4^(n - 2) + 1) * (16 / a_n n + 1) = 4 :=
by
  sorry

end min_value_of_expression_l1598_159856


namespace total_books_l1598_159876

-- Conditions
def TimsBooks : Nat := 44
def SamsBooks : Nat := 52
def AlexsBooks : Nat := 65
def KatiesBooks : Nat := 37

-- Theorem Statement
theorem total_books :
  TimsBooks + SamsBooks + AlexsBooks + KatiesBooks = 198 :=
by
  sorry

end total_books_l1598_159876


namespace ratio_b_to_c_l1598_159874

variables (a b c d e f : ℝ)

theorem ratio_b_to_c 
  (h1 : a / b = 1 / 3)
  (h2 : c / d = 1 / 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 10)
  (h5 : a * b * c / (d * e * f) = 0.15) :
  b / c = 9 := 
sorry

end ratio_b_to_c_l1598_159874


namespace greatest_second_term_l1598_159802

-- Definitions and Conditions
def is_arithmetic_sequence (a d : ℕ) : Bool := (a > 0) && (d > 0)
def sum_four_terms (a d : ℕ) : Bool := (4 * a + 6 * d = 80)
def integer_d (a d : ℕ) : Bool := ((40 - 2 * a) % 3 = 0)

-- Theorem statement to prove
theorem greatest_second_term : ∃ a d : ℕ, is_arithmetic_sequence a d ∧ sum_four_terms a d ∧ integer_d a d ∧ (a + d = 19) :=
sorry

end greatest_second_term_l1598_159802


namespace original_wattage_l1598_159853

theorem original_wattage (W : ℝ) (new_W : ℝ) (h1 : new_W = 1.25 * W) (h2 : new_W = 100) : W = 80 :=
by
  sorry

end original_wattage_l1598_159853


namespace martha_total_butterflies_l1598_159827

variable (Yellow Blue Black : ℕ)

def butterfly_equations (Yellow Blue Black : ℕ) : Prop :=
  (Blue = 2 * Yellow) ∧ (Blue = 6) ∧ (Black = 10)

theorem martha_total_butterflies 
  (h : butterfly_equations Yellow Blue Black) : 
  (Yellow + Blue + Black = 19) :=
by
  sorry

end martha_total_butterflies_l1598_159827


namespace eccentricity_of_ellipse_l1598_159866

noncomputable def ellipse_eccentricity (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse:
  ∀ (a b : ℝ) (c : ℝ), 
    0 < b ∧ b < a ∧ a = 3 * c → 
    ellipse_eccentricity a b c = 1/3 := by
  intros a b c h
  let e := ellipse_eccentricity a b c
  have h1 : 0 < b := h.1
  have h2 : b < a := h.2.left
  have h3 : a = 3 * c := h.2.right
  simp [ellipse_eccentricity, h3]
  sorry

end eccentricity_of_ellipse_l1598_159866


namespace max_cars_with_ac_but_not_rs_l1598_159829

namespace CarProblem

variables (total_cars : ℕ) 
          (cars_without_ac : ℕ)
          (cars_with_rs : ℕ)
          (cars_with_ac : ℕ := total_cars - cars_without_ac)
          (cars_with_ac_and_rs : ℕ)
          (cars_with_ac_but_not_rs : ℕ := cars_with_ac - cars_with_ac_and_rs)

theorem max_cars_with_ac_but_not_rs 
        (h1 : total_cars = 100)
        (h2 : cars_without_ac = 37)
        (h3 : cars_with_rs ≥ 51)
        (h4 : cars_with_ac_and_rs = min cars_with_rs cars_with_ac) :
        cars_with_ac_but_not_rs = 12 := by
    sorry

end CarProblem

end max_cars_with_ac_but_not_rs_l1598_159829


namespace base7_sum_correct_l1598_159897

theorem base7_sum_correct : 
  ∃ (A B C : ℕ), 
  A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (A = 2 ∨ A = 3 ∨ A = 5) ∧
  (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A ∧
  A + B + C = 16 :=
by
  sorry

end base7_sum_correct_l1598_159897


namespace binomial_identity_l1598_159852

-- Given:
variables {k n : ℕ}

-- Conditions:
axiom h₁ : 1 < k
axiom h₂ : 1 < n

-- Statement:
theorem binomial_identity (h₁ : 1 < k) (h₂ : 1 < n) : 
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := 
sorry

end binomial_identity_l1598_159852


namespace pandemic_cut_percentage_l1598_159895

-- Define the conditions
def initial_planned_production : ℕ := 200
def decrease_due_to_metal_shortage : ℕ := 50
def doors_per_car : ℕ := 5
def total_doors_produced : ℕ := 375

-- Define the quantities after metal shortage and before the pandemic
def production_after_metal_shortage : ℕ := initial_planned_production - decrease_due_to_metal_shortage
def doors_after_metal_shortage : ℕ := production_after_metal_shortage * doors_per_car
def cars_after_pandemic : ℕ := total_doors_produced / doors_per_car
def reduction_in_production : ℕ := production_after_metal_shortage - cars_after_pandemic

-- Define the expected percentage cut
def expected_percentage_cut : ℕ := 50

-- Prove that the percentage of production cut due to the pandemic is as required
theorem pandemic_cut_percentage : (reduction_in_production * 100 / production_after_metal_shortage) = expected_percentage_cut := by
  sorry

end pandemic_cut_percentage_l1598_159895


namespace chessboard_no_single_black_square_l1598_159867

theorem chessboard_no_single_black_square :
  (∀ (repaint : (Fin 8) × Bool → (Fin 8) × Bool), False) :=
by 
  sorry

end chessboard_no_single_black_square_l1598_159867


namespace find_value_of_x_l1598_159801

theorem find_value_of_x (a b c d e f x : ℕ) (h1 : a ≠ 1 ∧ a ≠ 6 ∧ b ≠ 1 ∧ b ≠ 6 ∧ c ≠ 1 ∧ c ≠ 6 ∧ d ≠ 1 ∧ d ≠ 6 ∧ e ≠ 1 ∧ e ≠ 6 ∧ f ≠ 1 ∧ f ≠ 6 ∧ x ≠ 1 ∧ x ≠ 6)
  (h2 : a + x + d = 18)
  (h3 : b + x + f = 18)
  (h4 : c + x + 6 = 18)
  (h5 : a + b + c + d + e + f + x + 6 + 1 = 45) :
  x = 7 :=
sorry

end find_value_of_x_l1598_159801


namespace negation_is_false_l1598_159812

-- Define even numbers
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the original proposition P
def P (a b : ℕ) : Prop := even a ∧ even b → even (a + b)

-- The negation of the proposition P
def notP (a b : ℕ) : Prop := ¬(even a ∧ even b → even (a + b))

-- The theorem to prove
theorem negation_is_false : ∀ a b : ℕ, ¬notP a b :=
by
  sorry

end negation_is_false_l1598_159812


namespace length_of_intersection_segment_l1598_159870

-- Define the polar coordinates conditions
def curve_1 (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def curve_2 (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

-- Convert polar equations to Cartesian coordinates
def curve_1_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4 * y
def curve_2_cartesian (x y : ℝ) : Prop := x = 1

-- Define the intersection points and the segment length function
def segment_length (y1 y2 : ℝ) : ℝ := abs (y1 - y2)

-- The statement to prove
theorem length_of_intersection_segment :
  (curve_1_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_1_cartesian 1 (2 - Real.sqrt 3)) →
  (curve_2_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_2_cartesian 1 (2 - Real.sqrt 3)) →
  segment_length (2 + Real.sqrt 3) (2 - Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end length_of_intersection_segment_l1598_159870


namespace curve_is_line_l1598_159892

noncomputable def curve_representation (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1) * (-1) = 0

theorem curve_is_line (x y : ℝ) (h : curve_representation x y) : 2 * x + 3 * y - 1 = 0 :=
by
  sorry

end curve_is_line_l1598_159892


namespace intersection_sets_l1598_159860

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l1598_159860


namespace inscribed_regular_polygon_sides_l1598_159847

theorem inscribed_regular_polygon_sides (n : ℕ) (h_central_angle : 360 / n = 72) : n = 5 :=
by
  sorry

end inscribed_regular_polygon_sides_l1598_159847


namespace xy_gt_1_necessary_but_not_sufficient_l1598_159807

-- To define the conditions and prove the necessary and sufficient conditions.

variable (x y : ℝ)

-- The main statement to prove once conditions are defined.
theorem xy_gt_1_necessary_but_not_sufficient : 
  (x > 1 ∧ y > 1 → x * y > 1) ∧ ¬ (x * y > 1 → x > 1 ∧ y > 1) := 
by 
  sorry

end xy_gt_1_necessary_but_not_sufficient_l1598_159807


namespace simplify_and_evaluate_l1598_159818

-- Problem statement with conditions translated into Lean
theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5 + 1) :
  (a / (a^2 - 2*a + 1)) / (1 + 1 / (a - 1)) = Real.sqrt 5 / 5 := sorry

end simplify_and_evaluate_l1598_159818


namespace distance_between_cities_l1598_159824

noncomputable def speed_a : ℝ := 1 / 10
noncomputable def speed_b : ℝ := 1 / 15
noncomputable def time_to_meet : ℝ := 6
noncomputable def distance_diff : ℝ := 12

theorem distance_between_cities : 
  (time_to_meet * (speed_a + speed_b) = 60) →
  time_to_meet * speed_a - time_to_meet * speed_b = distance_diff →
  time_to_meet * (speed_a + speed_b) = 60 :=
by
  intros h1 h2
  sorry

end distance_between_cities_l1598_159824


namespace min_people_wearing_both_l1598_159804

theorem min_people_wearing_both (n : ℕ) (h1 : n % 3 = 0)
  (h_gloves : ∃ g, g = n / 3 ∧ g = 1) (h_hats : ∃ h, h = (2 * n) / 3 ∧ h = 2) :
  ∃ x, x = 0 := by
  sorry

end min_people_wearing_both_l1598_159804


namespace cost_of_apples_l1598_159864

theorem cost_of_apples (price_per_six_pounds : ℕ) (pounds_to_buy : ℕ) (expected_cost : ℕ) :
  price_per_six_pounds = 5 → pounds_to_buy = 18 → (expected_cost = 15) → 
  (price_per_six_pounds / 6) * pounds_to_buy = expected_cost :=
by
  intro price_per_six_pounds_eq pounds_to_buy_eq expected_cost_eq
  rw [price_per_six_pounds_eq, pounds_to_buy_eq, expected_cost_eq]
  -- the actual proof would follow, using math steps similar to the solution but skipped here
  sorry

end cost_of_apples_l1598_159864


namespace product_of_real_roots_of_equation_l1598_159839

theorem product_of_real_roots_of_equation : 
  ∀ x : ℝ, (x^4 + (x - 4)^4 = 32) → x = 2 :=
sorry

end product_of_real_roots_of_equation_l1598_159839


namespace evaluate_expression_l1598_159848

noncomputable def expression_equal : Prop :=
  let a := (11: ℝ)
  let b := (11 : ℝ)^((1 : ℝ) / 6)
  let c := (11 : ℝ)^((1 : ℝ) / 5)
  (b / c = a^(-((1 : ℝ) / 30)))

theorem evaluate_expression :
  expression_equal :=
sorry

end evaluate_expression_l1598_159848
