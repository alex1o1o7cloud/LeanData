import Mathlib

namespace max_value_of_k_l1177_117799

theorem max_value_of_k (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + 2 * y) / (x * y) ≥ k / (2 * x + y)) :
  k ≤ 9 :=
by
  sorry

end max_value_of_k_l1177_117799


namespace intersection_A_B_l1177_117774

def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x^2) }
def B : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

end intersection_A_B_l1177_117774


namespace find_greater_solution_of_quadratic_l1177_117737

theorem find_greater_solution_of_quadratic:
  (x^2 + 14 * x - 88 = 0 → x = -22 ∨ x = 4) → (∀ x₁ x₂, (x₁ = -22 ∨ x₁ = 4) ∧ (x₂ = -22 ∨ x₂ = 4) → max x₁ x₂ = 4) :=
by
  intros h x₁ x₂ hx1x2
  -- proof omitted
  sorry

end find_greater_solution_of_quadratic_l1177_117737


namespace perimeter_square_C_l1177_117770

theorem perimeter_square_C 
  (a b c : ℝ) 
  (ha : 4 * a = 16) 
  (hb : 4 * b = 28) 
  (hc : c = |a - b|) : 
  4 * c = 12 := 
sorry

end perimeter_square_C_l1177_117770


namespace range_of_m_value_of_m_l1177_117712

-- Defining the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- The condition for having real roots
def has_real_roots (a b c : ℝ) := b^2 - 4 * a * c ≥ 0

-- First part: Range of values for m
theorem range_of_m (m : ℝ) : has_real_roots 1 (-2) (m - 1) ↔ m ≤ 2 := sorry

-- Second part: Finding m when x₁² + x₂² = 6x₁x₂
theorem value_of_m 
  (x₁ x₂ m : ℝ) (h₁ : quadratic_eq 1 (-2) (m - 1) x₁) (h₂ : quadratic_eq 1 (-2) (m - 1) x₂) 
  (h_sum : x₁ + x₂ = 2) (h_prod : x₁ * x₂ = m - 1) (h_condition : x₁^2 + x₂^2 = 6 * (x₁ * x₂)) : 
  m = 3 / 2 := sorry

end range_of_m_value_of_m_l1177_117712


namespace root_expression_value_l1177_117754

theorem root_expression_value
  (r s : ℝ)
  (h1 : 3 * r^2 - 4 * r - 8 = 0)
  (h2 : 3 * s^2 - 4 * s - 8 = 0) :
  (9 * r^3 - 9 * s^3) * (r - s)⁻¹ = 40 := 
sorry

end root_expression_value_l1177_117754


namespace geometric_sequence_expression_l1177_117796

theorem geometric_sequence_expression (a : ℝ) (a_n: ℕ → ℝ)
  (h1 : a_n 1 = a - 1)
  (h2 : a_n 2 = a + 1)
  (h3 : a_n 3 = a + 4)
  (hn : ∀ n, a_n (n + 1) = a_n n * (a_n 2 / a_n 1)) :
  a_n n = 4 * (3/2)^(n-1) :=
sorry

end geometric_sequence_expression_l1177_117796


namespace inclination_angle_of_line_l1177_117768

theorem inclination_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 2 * x - y + 1 = 0 → m = 2) → θ = Real.arctan 2 :=
by
  sorry

end inclination_angle_of_line_l1177_117768


namespace symmetric_points_x_axis_l1177_117706

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ := (a, 1)) (Q : ℝ × ℝ := (-4, b)) :
  (Q.1 = -P.1 ∧ Q.2 = -P.2) → (a = -4 ∧ b = -1) :=
by {
  sorry
}

end symmetric_points_x_axis_l1177_117706


namespace problem_inequality_l1177_117714

def f (x : ℝ) : ℝ := abs (x - 1)

def A := {x : ℝ | -1 < x ∧ x < 1}

theorem problem_inequality (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) : 
  f (a * b) > f a - f b := by
  sorry

end problem_inequality_l1177_117714


namespace Evelyn_bottle_caps_l1177_117779

theorem Evelyn_bottle_caps (initial_caps found_caps total_caps : ℕ)
  (h1 : initial_caps = 18)
  (h2 : found_caps = 63) :
  total_caps = 81 :=
by
  sorry

end Evelyn_bottle_caps_l1177_117779


namespace largest_multiple_of_8_less_than_100_l1177_117718

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l1177_117718


namespace certain_number_any_number_l1177_117778

theorem certain_number_any_number (k : ℕ) (n : ℕ) (h1 : 5^k - k^5 = 1) (h2 : 15^k ∣ n) : true :=
by
  sorry

end certain_number_any_number_l1177_117778


namespace cone_volume_increase_l1177_117748

theorem cone_volume_increase (r h : ℝ) (k : ℝ) :
  let V := (1/3) * π * r^2 * h
  let h' := 2.60 * h
  let r' := r * (1 + k / 100)
  let V' := (1/3) * π * (r')^2 * h'
  let percentage_increase := ((V' / V) - 1) * 100
  percentage_increase = ((1 + k / 100)^2 * 2.60 - 1) * 100 :=
by
  sorry

end cone_volume_increase_l1177_117748


namespace garden_furniture_costs_l1177_117745

theorem garden_furniture_costs (B T U : ℝ)
    (h1 : T + B + U = 765)
    (h2 : T = 2 * B)
    (h3 : U = 3 * B) :
    B = 127.5 ∧ T = 255 ∧ U = 382.5 :=
by
  sorry

end garden_furniture_costs_l1177_117745


namespace diet_soda_bottles_l1177_117756

theorem diet_soda_bottles (r d l t : Nat) (h1 : r = 49) (h2 : l = 6) (h3 : t = 89) (h4 : t = r + d) : d = 40 :=
by
  sorry

end diet_soda_bottles_l1177_117756


namespace find_ratio_AF_FB_l1177_117750

-- Define the vector space over reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions of points A, B, C, D, F, P
variables (a b c d f p : V)

-- Given conditions as hypotheses
variables (h1 : (p = 2 / 5 • a + 3 / 5 • d))
variables (h2 : (p = 5 / 7 • f + 2 / 7 • c))
variables (hd : (d = 1 / 3 • b + 2 / 3 • c))
variables (hf : (f = 1 / 4 • a + 3 / 4 • b))

-- Theorem statement
theorem find_ratio_AF_FB : (41 : ℝ) / 15 = (41 : ℝ) / 15 := 
by sorry

end find_ratio_AF_FB_l1177_117750


namespace George_says_365_l1177_117738

-- Definitions based on conditions
def skips_Alice (n : Nat) : Prop :=
  ∃ k, n = 3 * k - 1

def skips_Barbara (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * k - 1) - 1
  
def skips_Candice (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * k - 1) - 1) - 1

def skips_Debbie (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1

def skips_Eliza (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1

def skips_Fatima (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1) - 1

def numbers_said_by_students (n : Nat) : Prop :=
  skips_Alice n ∨ skips_Barbara n ∨ skips_Candice n ∨ skips_Debbie n ∨ skips_Eliza n ∨ skips_Fatima n

-- The proof statement
theorem George_says_365 : ¬numbers_said_by_students 365 :=
sorry

end George_says_365_l1177_117738


namespace find_positive_integer_l1177_117732

theorem find_positive_integer (x : ℕ) (h1 : (10 * x + 4) % (x + 4) = 0) (h2 : (10 * x + 4) / (x + 4) = x - 23) : x = 32 :=
by
  sorry

end find_positive_integer_l1177_117732


namespace sector_area_l1177_117780

theorem sector_area (r : ℝ) : (2 * r + 2 * r = 16) → (1/2 * r^2 * 2 = 16) :=
by
  intro h1
  sorry

end sector_area_l1177_117780


namespace cary_earnings_l1177_117767

variable (shoe_cost : ℕ) (saved_amount : ℕ)
variable (lawns_per_weekend : ℕ) (weeks_needed : ℕ)
variable (total_cost_needed : ℕ) (total_lawns : ℕ) (earn_per_lawn : ℕ)
variable (h1 : shoe_cost = 120)
variable (h2 : saved_amount = 30)
variable (h3 : lawns_per_weekend = 3)
variable (h4 : weeks_needed = 6)
variable (h5 : total_cost_needed = shoe_cost - saved_amount)
variable (h6 : total_lawns = lawns_per_weekend * weeks_needed)
variable (h7 : earn_per_lawn = total_cost_needed / total_lawns)

theorem cary_earnings :
  earn_per_lawn = 5 :=
by 
  sorry

end cary_earnings_l1177_117767


namespace total_fish_l1177_117741

theorem total_fish (fish_Lilly fish_Rosy : ℕ) (hL : fish_Lilly = 10) (hR : fish_Rosy = 8) : fish_Lilly + fish_Rosy = 18 := 
by 
  sorry

end total_fish_l1177_117741


namespace rooster_stamps_eq_two_l1177_117781

variable (r d : ℕ) -- r is the number of rooster stamps, d is the number of daffodil stamps

theorem rooster_stamps_eq_two (h1 : d = 2) (h2 : r - d = 0) : r = 2 := by
  sorry

end rooster_stamps_eq_two_l1177_117781


namespace generatrix_length_of_cone_l1177_117763

theorem generatrix_length_of_cone (r : ℝ) (l : ℝ) (h1 : r = 4) (h2 : (2 * Real.pi * r) = (Real.pi / 2) * l) : l = 16 := 
by
  sorry

end generatrix_length_of_cone_l1177_117763


namespace max_marks_paper_I_l1177_117792

-- Definitions based on the problem conditions
def percent_to_pass : ℝ := 0.35
def secured_marks : ℝ := 42
def failed_by : ℝ := 23

-- The calculated passing marks
def passing_marks : ℝ := secured_marks + failed_by

-- The theorem statement that needs to be proved
theorem max_marks_paper_I : ∀ (M : ℝ), (percent_to_pass * M = passing_marks) → M = 186 :=
by
  intros M h
  have h1 : M = passing_marks / percent_to_pass := by sorry
  have h2 : M = 186 := by sorry
  exact h2

end max_marks_paper_I_l1177_117792


namespace algebra_correct_option_B_l1177_117753

theorem algebra_correct_option_B (a b c : ℝ) (h : b * (c^2 + 1) ≠ 0) : 
  (a * (c^2 + 1)) / (b * (c^2 + 1)) = a / b := 
by
  -- Skipping the proof to focus on the statement
  sorry

end algebra_correct_option_B_l1177_117753


namespace prism_edges_l1177_117721

theorem prism_edges (n : ℕ) (h1 : n > 310) (h2 : n < 320) (h3 : n % 2 = 1) : n = 315 := by
  sorry

end prism_edges_l1177_117721


namespace students_accommodated_l1177_117746

theorem students_accommodated 
  (total_students : ℕ)
  (total_workstations : ℕ)
  (workstations_accommodating_x_students : ℕ)
  (x : ℕ)
  (workstations_accommodating_3_students : ℕ)
  (workstation_capacity_10 : ℕ)
  (workstation_capacity_6 : ℕ) :
  total_students = 38 → 
  total_workstations = 16 → 
  workstations_accommodating_x_students = 10 → 
  workstations_accommodating_3_students = 6 → 
  workstation_capacity_10 = 10 * x → 
  workstation_capacity_6 = 6 * 3 → 
  10 * x + 18 = 38 → 
  10 * 2 = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end students_accommodated_l1177_117746


namespace smaller_acute_angle_l1177_117701

theorem smaller_acute_angle (x : ℝ) (h : 5 * x + 4 * x = 90) : 4 * x = 40 :=
by 
  -- proof steps can be added here, but are omitted as per the instructions
  sorry

end smaller_acute_angle_l1177_117701


namespace quadratic_function_expr_value_of_b_minimum_value_of_m_l1177_117758

-- Problem 1: Proving the quadratic function expression
theorem quadratic_function_expr (x : ℝ) (b c : ℝ)
  (h1 : (0:ℝ) = x^2 + b * 0 + c)
  (h2 : -b / 2 = (1:ℝ)) :
  x^2 - 2 * x + 4 = x^2 + b * x + c := sorry

-- Problem 2: Proving specific values of b
theorem value_of_b (b c : ℝ)
  (h1 : b^2 - c = 0)
  (h2 : ∀ x : ℝ, (b - 3 ≤ x ∧ x ≤ b → (x^2 + b * x + c ≥ 21))) :
  b = -Real.sqrt 7 ∨ b = 4 := sorry

-- Problem 3: Proving the minimum value of m
theorem minimum_value_of_m (x : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x^2 + x + m ≥ x^2 - 2 * x + 4) :
  m = 4 := sorry

end quadratic_function_expr_value_of_b_minimum_value_of_m_l1177_117758


namespace jerry_needs_money_l1177_117724

theorem jerry_needs_money
  (jerry_has : ℕ := 7)
  (total_needed : ℕ := 16)
  (cost_per_figure : ℕ := 8) :
  (total_needed - jerry_has) * cost_per_figure = 72 :=
by
  sorry

end jerry_needs_money_l1177_117724


namespace problem1_problem2_problem3_problem4_l1177_117782

theorem problem1 : (-23 + 13 - 12) = -22 := 
by sorry

theorem problem2 : ((-2)^3 / 4 + 3 * (-5)) = -17 := 
by sorry

theorem problem3 : (-24 * (1/2 - 3/4 - 1/8)) = 9 := 
by sorry

theorem problem4 : ((2 - 7) / 5^2 + (-1)^2023 * (1/10)) = -3/10 := 
by sorry

end problem1_problem2_problem3_problem4_l1177_117782


namespace sin_symmetry_value_l1177_117744

theorem sin_symmetry_value (ϕ : ℝ) (hϕ₀ : 0 < ϕ) (hϕ₁ : ϕ < π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end sin_symmetry_value_l1177_117744


namespace find_y_intercept_l1177_117769

-- Conditions
def line_equation (x y : ℝ) : Prop := 4 * x + 7 * y - 3 * x * y = 28

-- Statement (Proof Problem)
theorem find_y_intercept : ∃ y : ℝ, line_equation 0 y ∧ (0, y) = (0, 4) := by
  sorry

end find_y_intercept_l1177_117769


namespace log_sum_range_l1177_117773

theorem log_sum_range {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h : Real.log (x + y) / Real.log 2 = Real.log x / Real.log 2 + Real.log y / Real.log 2) :
  4 ≤ x + y :=
by
  sorry

end log_sum_range_l1177_117773


namespace jerry_liters_of_mustard_oil_l1177_117708

-- Definitions
def cost_per_liter_mustard_oil : ℕ := 13
def cost_per_pound_penne_pasta : ℕ := 4
def cost_per_pound_pasta_sauce : ℕ := 5
def total_money_jerry_had : ℕ := 50
def money_left_with_jerry : ℕ := 7
def pounds_of_penne_pasta : ℕ := 3
def pounds_of_pasta_sauce : ℕ := 1

-- Our goal is to calculate how many liters of mustard oil Jerry bought
theorem jerry_liters_of_mustard_oil : ℕ :=
  let cost_of_penne_pasta := pounds_of_penne_pasta * cost_per_pound_penne_pasta
  let cost_of_pasta_sauce := pounds_of_pasta_sauce * cost_per_pound_pasta_sauce
  let total_spent := total_money_jerry_had - money_left_with_jerry
  let spent_on_pasta_and_sauce := cost_of_penne_pasta + cost_of_pasta_sauce
  let spent_on_mustard_oil := total_spent - spent_on_pasta_and_sauce
  spent_on_mustard_oil / cost_per_liter_mustard_oil

example : jerry_liters_of_mustard_oil = 2 := by
  unfold jerry_liters_of_mustard_oil
  simp
  sorry

end jerry_liters_of_mustard_oil_l1177_117708


namespace gcd_1248_1001_l1177_117765

theorem gcd_1248_1001 : Nat.gcd 1248 1001 = 13 := by
  sorry

end gcd_1248_1001_l1177_117765


namespace james_carrot_sticks_l1177_117710

theorem james_carrot_sticks (carrots_before : ℕ) (carrots_after : ℕ) 
(h_before : carrots_before = 22) (h_after : carrots_after = 15) : 
carrots_before + carrots_after = 37 := 
by 
  -- Placeholder for proof
  sorry

end james_carrot_sticks_l1177_117710


namespace sandra_remaining_money_l1177_117739

def sandra_savings : ℝ := 10
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def num_candies : ℝ := 14
def num_jelly_beans : ℝ := 20

theorem sandra_remaining_money : (sandra_savings + mother_contribution + father_contribution) - (num_candies * candy_cost + num_jelly_beans * jelly_bean_cost) = 11 :=
by
  sorry

end sandra_remaining_money_l1177_117739


namespace contrapositive_true_l1177_117700

theorem contrapositive_true (x : ℝ) : (x^2 - 2*x - 8 ≤ 0 → x ≥ -3) :=
by
  -- Proof omitted
  sorry

end contrapositive_true_l1177_117700


namespace evaluate_expression_l1177_117762

theorem evaluate_expression : (36 + 12) / (6 - (2 + 1)) = 16 := by
  sorry

end evaluate_expression_l1177_117762


namespace tom_has_1_dollar_left_l1177_117766

/-- Tom has $19 and each folder costs $2. After buying as many folders as possible,
Tom will have $1 left. -/
theorem tom_has_1_dollar_left (initial_money : ℕ) (folder_cost : ℕ) (folders_bought : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 19)
  (h2 : folder_cost = 2)
  (h3 : folders_bought = initial_money / folder_cost)
  (h4 : money_left = initial_money - folders_bought * folder_cost) :
  money_left = 1 :=
by
  -- proof will be provided here
  sorry

end tom_has_1_dollar_left_l1177_117766


namespace not_partition_1985_1987_partition_1987_1989_l1177_117749

-- Define the number of squares in an L-shape
def squares_in_lshape : ℕ := 3

-- Question 1: Can 1985 x 1987 be partitioned into L-shapes?
def partition_1985_1987 (m n : ℕ) (L_shape_size : ℕ) : Prop :=
  ∃ k : ℕ, m * n = k * L_shape_size ∧ (m % L_shape_size = 0 ∨ n % L_shape_size = 0)

theorem not_partition_1985_1987 :
  ¬ partition_1985_1987 1985 1987 squares_in_lshape :=
by {
  -- Proof omitted
  sorry
}

-- Question 2: Can 1987 x 1989 be partitioned into L-shapes?
theorem partition_1987_1989 :
  partition_1985_1987 1987 1989 squares_in_lshape :=
by {
  -- Proof omitted
  sorry
}

end not_partition_1985_1987_partition_1987_1989_l1177_117749


namespace tomatoes_left_l1177_117729

theorem tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) (eaters : ℕ) (final_tomatoes : ℕ)  
  (h_initial : initial_tomatoes = 21)
  (h_fraction : fraction_eaten = 1 / 3)
  (h_eaters : eaters = 2)
  (h_final : final_tomatoes = initial_tomatoes - initial_tomatoes * fraction_eaten) :
  final_tomatoes = 14 := by
  sorry

end tomatoes_left_l1177_117729


namespace truck_distance_l1177_117760

theorem truck_distance (V_t : ℝ) (D : ℝ) (h1 : D = V_t * 8) (h2 : D = (V_t + 18) * 5) : D = 240 :=
by
  sorry

end truck_distance_l1177_117760


namespace length_FJ_is_35_l1177_117716

noncomputable def length_of_FJ (h : ℝ) : ℝ :=
  let FG := 50
  let HI := 20
  let trapezium_area := (1 / 2) * (FG + HI) * h
  let half_trapezium_area := trapezium_area / 2
  let JI_area := (1 / 2) * 35 * h
  35

theorem length_FJ_is_35 (h : ℝ) : length_of_FJ h = 35 :=
  sorry

end length_FJ_is_35_l1177_117716


namespace logical_equivalence_l1177_117713

variable (R S T : Prop)

theorem logical_equivalence :
  (R → ¬S ∧ ¬T) ↔ ((S ∨ T) → ¬R) :=
by
  sorry

end logical_equivalence_l1177_117713


namespace total_hours_played_l1177_117734

-- Definitions based on conditions
def Nathan_hours_per_day : ℕ := 3
def Nathan_weeks : ℕ := 2
def days_per_week : ℕ := 7

def Tobias_hours_per_day : ℕ := 5
def Tobias_weeks : ℕ := 1

-- Calculating total hours
def Nathan_total_hours := Nathan_hours_per_day * days_per_week * Nathan_weeks
def Tobias_total_hours := Tobias_hours_per_day * days_per_week * Tobias_weeks

-- Theorem statement
theorem total_hours_played : Nathan_total_hours + Tobias_total_hours = 77 := by
  -- Proof would go here
  sorry

end total_hours_played_l1177_117734


namespace ramu_paid_for_old_car_l1177_117783

theorem ramu_paid_for_old_car (repairs : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (P : ℝ) :
    repairs = 12000 ∧ selling_price = 64900 ∧ profit_percent = 20.185185185185187 → 
    selling_price = P + repairs + (P + repairs) * (profit_percent / 100) → 
    P = 42000 :=
by
  intros h1 h2
  sorry

end ramu_paid_for_old_car_l1177_117783


namespace number_of_zeros_of_g_l1177_117715

noncomputable def f (x a : ℝ) := Real.exp x * (x + a)

noncomputable def g (x a : ℝ) := f (x - a) a - x^2

theorem number_of_zeros_of_g (a : ℝ) :
  (if a < 1 then ∃! x, g x a = 0
   else if a = 1 then ∃! x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0
   else ∃! x1 x2 x3, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0) := sorry

end number_of_zeros_of_g_l1177_117715


namespace quadratic_transform_l1177_117703

theorem quadratic_transform (x : ℝ) : x^2 - 6 * x - 5 = 0 → (x - 3)^2 = 14 :=
by
  intro h
  sorry

end quadratic_transform_l1177_117703


namespace smallest_prime_after_seven_non_primes_l1177_117791

-- Define the property of being non-prime
def non_prime (n : ℕ) : Prop :=
¬Nat.Prime n

-- Statement of the proof problem
theorem smallest_prime_after_seven_non_primes :
  ∃ m : ℕ, (∀ i : ℕ, (m - 7 ≤ i ∧ i < m) → non_prime i) ∧ Nat.Prime m ∧
  (∀ p : ℕ, (∀ i : ℕ, (p - 7 ≤ i ∧ i < p) → non_prime i) → Nat.Prime p → m ≤ p) :=
sorry

end smallest_prime_after_seven_non_primes_l1177_117791


namespace percentage_markup_l1177_117736

theorem percentage_markup (selling_price cost_price : ℚ)
  (h_selling_price : selling_price = 8325)
  (h_cost_price : cost_price = 7239.13) :
  ((selling_price - cost_price) / cost_price) * 100 = 15 := 
sorry

end percentage_markup_l1177_117736


namespace volume_pyramid_l1177_117702

theorem volume_pyramid (V : ℝ) : 
  ∃ V_P : ℝ, V_P = V / 6 :=
by
  sorry

end volume_pyramid_l1177_117702


namespace number_of_jars_good_for_sale_l1177_117722

def numberOfGoodJars (initialCartons : Nat) (cartonsNotDelivered : Nat) (jarsPerCarton : Nat)
  (damagedJarsPerCarton : Nat) (numberOfDamagedCartons : Nat) (oneTotallyDamagedCarton : Nat) : Nat := 
  let deliveredCartons := initialCartons - cartonsNotDelivered
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := (damagedJarsPerCarton * numberOfDamagedCartons) + oneTotallyDamagedCarton
  totalJars - damagedJars

theorem number_of_jars_good_for_sale : 
  numberOfGoodJars 50 20 20 3 5 20 = 565 :=
by
  sorry

end number_of_jars_good_for_sale_l1177_117722


namespace value_fraction_eq_three_l1177_117752

namespace Problem

variable {R : Type} [Field R]

theorem value_fraction_eq_three (a b c : R) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b + c) / (2 * a + b - c) = 3 := by
  sorry

end Problem

end value_fraction_eq_three_l1177_117752


namespace distance_center_to_line_l1177_117776

noncomputable def circle_center : ℝ × ℝ :=
  let b := 2
  let c := -4
  (1, -2)

noncomputable def distance_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / (Real.sqrt (a^2 + b^2))

theorem distance_center_to_line : distance_point_to_line circle_center 3 4 5 = 0 :=
by
  sorry

end distance_center_to_line_l1177_117776


namespace gcf_75_90_l1177_117757

theorem gcf_75_90 : Nat.gcd 75 90 = 15 :=
by
  sorry

end gcf_75_90_l1177_117757


namespace smallest_number_of_oranges_l1177_117764

theorem smallest_number_of_oranges (n : ℕ) (total_oranges : ℕ) :
  (total_oranges > 200) ∧ total_oranges = 15 * n - 6 ∧ n ≥ 14 → total_oranges = 204 :=
by
  sorry

end smallest_number_of_oranges_l1177_117764


namespace new_class_average_l1177_117743

theorem new_class_average (total_students : ℕ) (students_group1 : ℕ) (avg1 : ℝ) (students_group2 : ℕ) (avg2 : ℝ) : 
  total_students = 40 → students_group1 = 28 → avg1 = 68 → students_group2 = 12 → avg2 = 77 → 
  ((students_group1 * avg1 + students_group2 * avg2) / total_students) = 70.7 :=
by
  sorry

end new_class_average_l1177_117743


namespace action_figures_more_than_books_proof_l1177_117711

-- Definitions for the conditions
def books := 3
def action_figures_initial := 4
def action_figures_added := 2

-- Definition for the total action figures
def action_figures_total := action_figures_initial + action_figures_added

-- Definition for the number difference
def action_figures_more_than_books := action_figures_total - books

-- Proof statement
theorem action_figures_more_than_books_proof : action_figures_more_than_books = 3 :=
by
  sorry

end action_figures_more_than_books_proof_l1177_117711


namespace find_number_l1177_117789

theorem find_number (number : ℝ) (h1 : 213 * number = 3408) (h2 : 0.16 * 2.13 = 0.3408) : number = 16 :=
by
  sorry

end find_number_l1177_117789


namespace total_wolves_l1177_117704

theorem total_wolves (x y : ℕ) :
  (x + 2 * y = 20) →
  (4 * x + 3 * y = 55) →
  (x + y = 15) :=
by
  intro h1 h2
  sorry

end total_wolves_l1177_117704


namespace find_B_investment_l1177_117787

def A_investment : ℝ := 24000
def C_investment : ℝ := 36000
def C_profit : ℝ := 36000
def total_profit : ℝ := 92000
def B_investment := 32000

theorem find_B_investment (B_investment_unknown : ℝ) :
  (C_investment / C_profit) = ((A_investment + B_investment_unknown + C_investment) / total_profit) →
  B_investment_unknown = B_investment := 
by 
  -- Mathematical equivalence to the given problem
  -- Proof omitted since only the statement is required
  sorry

end find_B_investment_l1177_117787


namespace find_sin_2alpha_l1177_117798

theorem find_sin_2alpha (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) 
    (h2 : 3 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π / 4 - α)) : 
  Real.sin (2 * α) = -8 / 9 := 
sorry

end find_sin_2alpha_l1177_117798


namespace matthew_more_strawberries_than_betty_l1177_117726

noncomputable def B : ℕ := 16

theorem matthew_more_strawberries_than_betty (M N : ℕ) 
  (h1 : M > B)
  (h2 : M = 2 * N) 
  (h3 : B + M + N = 70) : M - B = 20 :=
by
  sorry

end matthew_more_strawberries_than_betty_l1177_117726


namespace smallest_base_l1177_117720

theorem smallest_base (b : ℕ) (h1 : b^2 ≤ 125) (h2 : 125 < b^3) : b = 6 := by
  sorry

end smallest_base_l1177_117720


namespace find_alpha_after_five_operations_l1177_117717

def returns_to_starting_point_after_operations (α : Real) (n : Nat) : Prop :=
  (n * α) % 360 = 0

theorem find_alpha_after_five_operations (α : Real) 
  (hα1 : 0 < α)
  (hα2 : α < 180)
  (h_return : returns_to_starting_point_after_operations α 5) :
  α = 72 ∨ α = 144 :=
sorry

end find_alpha_after_five_operations_l1177_117717


namespace linear_inequality_solution_l1177_117733

theorem linear_inequality_solution (a b : ℝ)
  (h₁ : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -3 ∨ x > 1)) :
  ∀ x : ℝ, a * x + b < 0 ↔ x < 3 / 2 :=
by
  sorry

end linear_inequality_solution_l1177_117733


namespace jesse_started_with_l1177_117727

-- Define the conditions
variables (g e : ℕ)

-- Theorem stating that given the conditions, Jesse started with 78 pencils
theorem jesse_started_with (g e : ℕ) (h1 : g = 44) (h2 : e = 34) : e + g = 78 :=
by sorry

end jesse_started_with_l1177_117727


namespace find_f2_l1177_117759

noncomputable def f (x : ℝ) : ℝ := (4*x + 2/x + 3) / 3

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (1 / x) = 2 * x + 1) : f 2 = 4 :=
  by
  sorry

end find_f2_l1177_117759


namespace martha_points_calculation_l1177_117784

theorem martha_points_calculation :
  let beef_cost := 3 * 11
  let beef_discount := 0.10 * beef_cost
  let total_beef_cost := beef_cost - beef_discount

  let fv_cost := 8 * 4
  let fv_discount := 0.05 * fv_cost
  let total_fv_cost := fv_cost - fv_discount

  let spices_cost := 2 * 6

  let other_groceries_cost := 37 - 3

  let total_cost := total_beef_cost + total_fv_cost + spices_cost + other_groceries_cost

  let spending_points := (total_cost / 10).floor * 50

  let bonus_points_over_100 := if total_cost > 100 then 250 else 0

  let loyalty_points := 100
  
  spending_points + bonus_points_over_100 + loyalty_points = 850 := by
    sorry

end martha_points_calculation_l1177_117784


namespace clothing_store_earnings_l1177_117728

-- Definitions for the given conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def cost_per_shirt : ℕ := 10
def cost_per_jeans : ℕ := 2 * cost_per_shirt

-- Theorem statement
theorem clothing_store_earnings : 
  (num_shirts * cost_per_shirt + num_jeans * cost_per_jeans = 400) := 
sorry

end clothing_store_earnings_l1177_117728


namespace quadratic_has_two_distinct_real_roots_l1177_117771

theorem quadratic_has_two_distinct_real_roots (k : ℝ) (hk1 : k ≠ 0) (hk2 : k < 0) : (5 - 4 * k) > 0 :=
sorry

end quadratic_has_two_distinct_real_roots_l1177_117771


namespace product_of_numbers_l1177_117755

theorem product_of_numbers (a b : ℝ) 
  (h1 : a + b = 5 * (a - b))
  (h2 : a * b = 18 * (a - b)) : 
  a * b = 54 :=
by
  sorry

end product_of_numbers_l1177_117755


namespace sum_ef_l1177_117793

variables (a b c d e f : ℝ)

-- Definitions based on conditions
def avg_ab : Prop := (a + b) / 2 = 5.2
def avg_cd : Prop := (c + d) / 2 = 5.8
def overall_avg : Prop := (a + b + c + d + e + f) / 6 = 5.4

-- Main theorem to prove
theorem sum_ef (h1 : avg_ab a b) (h2 : avg_cd c d) (h3 : overall_avg a b c d e f) : e + f = 10.4 :=
sorry

end sum_ef_l1177_117793


namespace square_garden_perimeter_l1177_117742

theorem square_garden_perimeter (q p : ℝ) (h : q = 2 * p + 20) : p = 40 :=
sorry

end square_garden_perimeter_l1177_117742


namespace sequence_an_sum_sequence_Tn_l1177_117751

theorem sequence_an (k c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = k * c ^ n - k) (ha2 : a 2 = 4) (ha6 : a 6 = 8 * a 3) :
  ∀ n, a n = 2 ^ n :=
by
  -- Proof is assumed to be given
  sorry

theorem sum_sequence_Tn (a : ℕ → ℝ) (T : ℕ → ℝ)
  (ha : ∀ n, a n = 2 ^ n) :
  ∀ n, T n = (n - 1) * 2 ^ (n + 1) + 2 :=
by
  -- Proof is assumed to be given
  sorry

end sequence_an_sum_sequence_Tn_l1177_117751


namespace distance_A_B_l1177_117794

theorem distance_A_B (d : ℝ)
  (speed_A : ℝ := 100) (speed_B : ℝ := 90) (speed_C : ℝ := 75)
  (location_A location_B : point) (is_at_A : location_A = point_A) (is_at_B : location_B = point_B)
  (t_meet_AB : ℝ := d / (speed_A + speed_B))
  (t_meet_AC : ℝ := t_meet_AB + 3)
  (distance_AC : ℝ := speed_A * 3)
  (distance_C : ℝ := speed_C * t_meet_AC) :
  d = 650 :=
by {
  sorry
}

end distance_A_B_l1177_117794


namespace easter_eggs_problem_l1177_117719

noncomputable def mia_rate : ℕ := 24
noncomputable def billy_rate : ℕ := 10
noncomputable def total_hours : ℕ := 5
noncomputable def total_eggs : ℕ := 170

theorem easter_eggs_problem :
  (mia_rate + billy_rate) * total_hours = total_eggs :=
by
  sorry

end easter_eggs_problem_l1177_117719


namespace ratio_of_areas_l1177_117777

theorem ratio_of_areas (s : ℝ) : (s^2) / ((3 * s)^2) = 1 / 9 := 
by
  sorry

end ratio_of_areas_l1177_117777


namespace tangent_line_parallel_points_l1177_117707

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Prove the points where the derivative equals 4
theorem tangent_line_parallel_points :
  ∃ (P0 : ℝ × ℝ), P0 = (1, 0) ∨ P0 = (-1, -4) ∧ (f' P0.fst = 4) :=
by
  sorry

end tangent_line_parallel_points_l1177_117707


namespace daria_needs_to_earn_more_money_l1177_117723

noncomputable def moneyNeeded (ticket_cost : ℕ) (discount : ℕ) (gift_card : ℕ) 
  (transport_cost : ℕ) (parking_cost : ℕ) (tshirt_cost : ℕ) (current_money : ℕ) (tickets : ℕ) : ℕ :=
  let discounted_ticket_price := ticket_cost - (ticket_cost * discount / 100)
  let total_ticket_cost := discounted_ticket_price * tickets
  let ticket_cost_after_gift_card := total_ticket_cost - gift_card
  let total_cost := ticket_cost_after_gift_card + transport_cost + parking_cost + tshirt_cost
  total_cost - current_money

theorem daria_needs_to_earn_more_money :
  moneyNeeded 90 10 50 20 10 25 189 6 = 302 :=
by
  sorry

end daria_needs_to_earn_more_money_l1177_117723


namespace find_n_l1177_117735

open Nat

theorem find_n (d : ℕ → ℕ) (n : ℕ) (h1 : ∀ j, d (j + 1) > d j) (h2 : n = d 13 + d 14 + d 15) (h3 : (d 5 + 1)^3 = d 15 + 1) : 
  n = 1998 :=
by
  sorry

end find_n_l1177_117735


namespace find_a8_l1177_117790

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n ≥ 2, (2 * a n - 3) / (a n - 1) = 2) (h2 : a 2 = 1) : a 8 = 16 := 
sorry

end find_a8_l1177_117790


namespace age_of_15th_student_l1177_117731

theorem age_of_15th_student (avg_age_15_students avg_age_5_students avg_age_9_students : ℕ)
  (total_students total_age_15_students total_age_5_students total_age_9_students : ℕ)
  (h1 : total_students = 15)
  (h2 : avg_age_15_students = 15)
  (h3 : avg_age_5_students = 14)
  (h4 : avg_age_9_students = 16)
  (h5 : total_age_15_students = total_students * avg_age_15_students)
  (h6 : total_age_5_students = 5 * avg_age_5_students)
  (h7 : total_age_9_students = 9 * avg_age_9_students):
  total_age_15_students = total_age_5_students + total_age_9_students + 11 :=
by
  sorry

end age_of_15th_student_l1177_117731


namespace prudence_nap_is_4_hours_l1177_117740

def prudence_nap_length (total_sleep : ℕ) (weekdays_sleep : ℕ) (weekend_sleep : ℕ) (weeks : ℕ) (total_weeks : ℕ) : ℕ :=
  (total_sleep - (weekdays_sleep + weekend_sleep) * total_weeks) / (2 * total_weeks)

theorem prudence_nap_is_4_hours
  (total_sleep weekdays_sleep weekend_sleep total_weeks : ℕ) :
  total_sleep = 200 ∧ weekdays_sleep = 5 * 6 ∧ weekend_sleep = 2 * 9 ∧ total_weeks = 4 →
  prudence_nap_length total_sleep weekdays_sleep weekend_sleep total_weeks total_weeks = 4 :=
by
  intros
  sorry

end prudence_nap_is_4_hours_l1177_117740


namespace new_light_wattage_is_143_l1177_117747

-- Define the original wattage and the percentage increase
def original_wattage : ℕ := 110
def percentage_increase : ℕ := 30

-- Compute the increase in wattage
noncomputable def increase : ℕ := (percentage_increase * original_wattage) / 100

-- The new wattage should be the original wattage plus the increase
noncomputable def new_wattage : ℕ := original_wattage + increase

-- State the theorem that proves the new wattage is 143 watts
theorem new_light_wattage_is_143 : new_wattage = 143 := by
  unfold new_wattage
  unfold increase
  sorry

end new_light_wattage_is_143_l1177_117747


namespace range_of_a_l1177_117785

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
sorry

end range_of_a_l1177_117785


namespace rectangles_in_grid_squares_in_grid_l1177_117761

theorem rectangles_in_grid (h_lines : ℕ) (v_lines : ℕ) : h_lines = 31 → v_lines = 31 → 
  (∃ rect_count : ℕ, rect_count = 216225) :=
by
  intros h_lines_eq v_lines_eq
  sorry

theorem squares_in_grid (n : ℕ) : n = 31 → (∃ square_count : ℕ, square_count = 6975) :=
by
  intros n_eq
  sorry

end rectangles_in_grid_squares_in_grid_l1177_117761


namespace expected_value_decagonal_die_l1177_117786

-- Given conditions
def decagonal_die_faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def probability (n : ℕ) : ℚ := 1 / 10

-- The mathematical proof problem (statement only, no proof required)
theorem expected_value_decagonal_die : 
  (List.sum decagonal_die_faces : ℚ) / List.length decagonal_die_faces = 5.5 := by
  sorry

end expected_value_decagonal_die_l1177_117786


namespace max_value_of_abs_asinx_plus_b_l1177_117795

theorem max_value_of_abs_asinx_plus_b 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) : 
  ∃ M, M = 2 ∧ ∀ x : ℝ, |a * Real.sin x + b| ≤ M :=
by
  use 2
  sorry

end max_value_of_abs_asinx_plus_b_l1177_117795


namespace lisa_minimum_fifth_term_score_l1177_117775

theorem lisa_minimum_fifth_term_score :
  ∀ (score1 score2 score3 score4 average_needed total_terms : ℕ),
  score1 = 84 →
  score2 = 80 →
  score3 = 82 →
  score4 = 87 →
  average_needed = 85 →
  total_terms = 5 →
  (∃ (score5 : ℕ), 
     (score1 + score2 + score3 + score4 + score5) / total_terms ≥ average_needed ∧ 
     score5 = 92) :=
by
  sorry

end lisa_minimum_fifth_term_score_l1177_117775


namespace greatest_possible_perimeter_l1177_117788

theorem greatest_possible_perimeter (a b c : ℕ) 
    (h₁ : a = 4 * b ∨ b = 4 * a ∨ c = 4 * a ∨ c = 4 * b)
    (h₂ : a = 18 ∨ b = 18 ∨ c = 18)
    (triangle_ineq : a + b > c ∧ a + c > b ∧ b + c > a) :
    a + b + c = 43 :=
by {
  sorry
}

end greatest_possible_perimeter_l1177_117788


namespace joe_list_possibilities_l1177_117730

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l1177_117730


namespace xy_identity_l1177_117725

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end xy_identity_l1177_117725


namespace original_price_l1177_117709

variable (a : ℝ)

theorem original_price (h : 0.6 * x = a) : x = (5 / 3) * a :=
sorry

end original_price_l1177_117709


namespace cost_scheme_1_cost_scheme_2_cost_comparison_scheme_more_cost_effective_combined_plan_l1177_117772

variable (x : ℕ) (x_ge_4 : x ≥ 4)

-- Total cost under scheme ①
def scheme_1_cost (x : ℕ) : ℕ := 5 * x + 60

-- Total cost under scheme ②
def scheme_2_cost (x : ℕ) : ℕ := 9 * (80 + 5 * x) / 10

theorem cost_scheme_1 (x : ℕ) (x_ge_4 : x ≥ 4) : 
  scheme_1_cost x = 5 * x + 60 :=  
sorry

theorem cost_scheme_2 (x : ℕ) (x_ge_4 : x ≥ 4) : 
  scheme_2_cost x = (80 + 5 * x) * 9 / 10 := 
sorry

-- When x = 30, compare which scheme is more cost-effective
variable (x_eq_30 : x = 30)
theorem cost_comparison_scheme (x_eq_30 : x = 30) : 
  scheme_1_cost 30 > scheme_2_cost 30 := 
sorry

-- When x = 30, a more cost-effective combined purchasing plan
def combined_scheme_cost : ℕ := scheme_1_cost 4 + scheme_2_cost (30 - 4)

theorem more_cost_effective_combined_plan (x_eq_30 : x = 30) : 
  combined_scheme_cost < scheme_1_cost 30 ∧ combined_scheme_cost < scheme_2_cost 30 := 
sorry

end cost_scheme_1_cost_scheme_2_cost_comparison_scheme_more_cost_effective_combined_plan_l1177_117772


namespace total_pay_of_two_employees_l1177_117705

theorem total_pay_of_two_employees
  (Y_pay : ℝ)
  (X_pay : ℝ)
  (h1 : Y_pay = 280)
  (h2 : X_pay = 1.2 * Y_pay) :
  X_pay + Y_pay = 616 :=
by
  sorry

end total_pay_of_two_employees_l1177_117705


namespace calc_expression_l1177_117797

theorem calc_expression : (4 + 6 + 10) / 3 - 2 / 3 = 6 := by
  sorry

end calc_expression_l1177_117797
