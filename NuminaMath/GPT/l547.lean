import Mathlib

namespace polynomial_expansion_sum_is_21_l547_54775

theorem polynomial_expansion_sum_is_21 :
  ∃ (A B C D : ℤ), (∀ (x : ℤ), (x + 2) * (3 * x^2 - x + 5) = A * x^3 + B * x^2 + C * x + D) ∧
  A + B + C + D = 21 :=
by
  sorry

end polynomial_expansion_sum_is_21_l547_54775


namespace calculate_angles_and_side_l547_54791

theorem calculate_angles_and_side (a b B : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 2) (h_B : B = 45) :
  ∃ A C c, (A = 60 ∧ C = 75 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨ (A = 120 ∧ C = 15 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2) :=
by sorry

end calculate_angles_and_side_l547_54791


namespace range_of_a_l547_54708

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a - 1 / Real.exp x)

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ deriv (f a) x₁ = 0 ∧ deriv (f a) x₂ = 0) ↔ -1 / Real.exp 2 < a ∧ a < 0 := 
sorry

end range_of_a_l547_54708


namespace other_group_land_l547_54717

def total_land : ℕ := 900
def remaining_land : ℕ := 385
def lizzies_group_land : ℕ := 250

theorem other_group_land :
  total_land - remaining_land - lizzies_group_land = 265 :=
by
  sorry

end other_group_land_l547_54717


namespace cos_sin_sequence_rational_l547_54747

variable (α : ℝ) (h₁ : ∃ r : ℚ, r = (Real.sin α + Real.cos α))

theorem cos_sin_sequence_rational :
    (∀ n : ℕ, n > 0 → ∃ r : ℚ, r = (Real.cos α)^n + (Real.sin α)^n) :=
by
  sorry

end cos_sin_sequence_rational_l547_54747


namespace largest_lcm_l547_54702

theorem largest_lcm :
  max (max (max (max (Nat.lcm 18 4) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 14)) (Nat.lcm 18 18) = 126 :=
by
  sorry

end largest_lcm_l547_54702


namespace find_c_l547_54706

theorem find_c (c : ℝ) (h : ∀ x : ℝ, ∃ a : ℝ, (x + a)^2 = x^2 + 200 * x + c) : c = 10000 :=
sorry

end find_c_l547_54706


namespace range_of_a_l547_54752

-- Define the propositions p and q
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the main theorem which combines both propositions and infers the range of a
theorem range_of_a (a : ℝ) : prop_p a ∧ prop_q a → a ≤ -2 := sorry

end range_of_a_l547_54752


namespace gross_profit_percentage_l547_54787

theorem gross_profit_percentage (sales_price gross_profit : ℝ) (h_sales_price : sales_price = 91) (h_gross_profit : gross_profit = 56) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 160 :=
by
  sorry

end gross_profit_percentage_l547_54787


namespace worker_weekly_pay_l547_54788

variable (regular_rate : ℕ) -- Regular rate of Rs. 10 per survey
variable (total_surveys : ℕ) -- Worker completes 100 surveys per week
variable (cellphone_surveys : ℕ) -- 60 surveys involve the use of cellphone
variable (increased_rate : ℕ) -- Increased rate 30% higher than regular rate

-- Defining given values
def reg_rate : ℕ := 10
def total_survey_count : ℕ := 100
def cellphone_survey_count : ℕ := 60
def inc_rate : ℕ := reg_rate + 3

-- Calculating payments
def regular_survey_count : ℕ := total_survey_count - cellphone_survey_count
def regular_pay : ℕ := regular_survey_count * reg_rate
def cellphone_pay : ℕ := cellphone_survey_count * inc_rate

-- Total pay calculation
def total_pay : ℕ := regular_pay + cellphone_pay

-- Theorem to be proved
theorem worker_weekly_pay : total_pay = 1180 := 
by
  -- instantiate variables
  let regular_rate := reg_rate
  let total_surveys := total_survey_count
  let cellphone_surveys := cellphone_survey_count
  let increased_rate := inc_rate
  
  -- skip proof
  sorry

end worker_weekly_pay_l547_54788


namespace base_85_solution_l547_54714

theorem base_85_solution (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 16) :
  (352936524 - b) % 17 = 0 ↔ b = 4 :=
by
  sorry

end base_85_solution_l547_54714


namespace lychee_ratio_l547_54734

theorem lychee_ratio (total_lychees : ℕ) (sold_lychees : ℕ) (remaining_home : ℕ) (remaining_after_eat : ℕ) 
    (h1: total_lychees = 500) 
    (h2: sold_lychees = total_lychees / 2) 
    (h3: remaining_home = total_lychees - sold_lychees) 
    (h4: remaining_after_eat = 100)
    (h5: remaining_after_eat + (remaining_home - remaining_after_eat) = remaining_home) : 
    (remaining_home - remaining_after_eat) / remaining_home = 3 / 5 :=
by
    -- Proof is omitted
    sorry

end lychee_ratio_l547_54734


namespace find_c_l547_54720

   noncomputable def c_value (c : ℝ) : Prop :=
     ∃ (x y : ℝ), (x^2 - 8*x + y^2 + 10*y + c = 0) ∧ (x - 4)^2 + (y + 5)^2 = 25

   theorem find_c (c : ℝ) : c_value c → c = 16 := by
     sorry
   
end find_c_l547_54720


namespace tan_x_y_l547_54704

theorem tan_x_y (x y : ℝ) (h : Real.sin (2 * x + y) = 5 * Real.sin y) :
  Real.tan (x + y) = (3 / 2) * Real.tan x :=
sorry

end tan_x_y_l547_54704


namespace minimum_value_expression_l547_54776

theorem minimum_value_expression : ∃ x : ℝ, (3 * x^2 - 18 * x + 2023) = 1996 := sorry

end minimum_value_expression_l547_54776


namespace polar_to_rectangular_4sqrt2_pi_over_4_l547_54710

theorem polar_to_rectangular_4sqrt2_pi_over_4 :
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (4, 4) :=
by
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  sorry

end polar_to_rectangular_4sqrt2_pi_over_4_l547_54710


namespace remainder_of_five_consecutive_odds_mod_12_l547_54705

/-- Let x be an odd integer. Prove that (x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) % 12 = 9 
    when x ≡ 5 (mod 12). -/
theorem remainder_of_five_consecutive_odds_mod_12 {x : ℤ} (h : x % 12 = 5) :
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) % 12 = 9 :=
sorry

end remainder_of_five_consecutive_odds_mod_12_l547_54705


namespace max_ab_plus_2bc_l547_54765

theorem max_ab_plus_2bc (A B C : ℝ) (AB AC BC : ℝ) (hB : B = 60) (hAC : AC = Real.sqrt 3) :
  (AB + 2 * BC) ≤ 2 * Real.sqrt 7 :=
sorry

end max_ab_plus_2bc_l547_54765


namespace increasing_function_range_l547_54783

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x y : ℝ, x < y → f a x ≤ f a y) : 
  1.5 ≤ a ∧ a < 2 :=
sorry

end increasing_function_range_l547_54783


namespace box_cubes_no_green_face_l547_54722

theorem box_cubes_no_green_face (a b c : ℕ) (h_a2 : a > 2) (h_b2 : b > 2) (h_c2 : c > 2)
  (h_no_green_face : (a-2)*(b-2)*(c-2) = (a*b*c) / 3) :
  (a, b, c) = (7, 30, 4) ∨ (a, b, c) = (8, 18, 4) ∨ (a, b, c) = (9, 14, 4) ∨
  (a, b, c) = (10, 12, 4) ∨ (a, b, c) = (5, 27, 5) ∨ (a, b, c) = (6, 12, 5) ∨
  (a, b, c) = (7, 9, 5) ∨ (a, b, c) = (6, 8, 6) :=
sorry

end box_cubes_no_green_face_l547_54722


namespace drone_altitude_l547_54731

theorem drone_altitude (h c d : ℝ) (HC HD CD : ℝ)
  (HCO_eq : h^2 + c^2 = HC^2)
  (HDO_eq : h^2 + d^2 = HD^2)
  (CD_eq : c^2 + d^2 = CD^2) 
  (HC_val : HC = 170)
  (HD_val : HD = 160)
  (CD_val : CD = 200) :
  h = 50 * Real.sqrt 29 :=
by
  sorry

end drone_altitude_l547_54731


namespace B_days_to_complete_work_l547_54764

theorem B_days_to_complete_work 
  (W : ℝ) -- Define the amount of work
  (A_rate : ℝ := W / 15) -- A can complete the work in 15 days
  (B_days : ℝ) -- B can complete the work in B_days days
  (B_rate : ℝ := W / B_days) -- B's rate of work
  (total_days : ℝ := 12) -- Total days to complete the work
  (A_days_after_B_leaves : ℝ := 10) -- Days A works alone after B leaves
  (work_done_together : ℝ := 2 * (A_rate + B_rate)) -- Work done together in 2 days
  (work_done_by_A : ℝ := 10 * A_rate) -- Work done by A alone in 10 days
  (total_work_done : ℝ := work_done_together + work_done_by_A) -- Total work done
  (h_total_work_done : total_work_done = W) -- Total work equals W
  : B_days = 10 :=
sorry

end B_days_to_complete_work_l547_54764


namespace sum_of_adjacent_cells_multiple_of_4_l547_54779

theorem sum_of_adjacent_cells_multiple_of_4 :
  ∃ (i j : ℕ) (a b : ℕ) (H₁ : i < 22) (H₂ : j < 22),
    let grid (i j : ℕ) : ℕ := -- define the function for grid indexing
      ((i * 22) + j + 1 : ℕ)
    ∃ (i1 j1 : ℕ) (H₁₁ : i1 = i ∨ i1 = i + 1 ∨ i1 = i - 1)
                   (H₁₂ : j1 = j ∨ j1 = j + 1 ∨ j1 = j - 1),
      a = grid i j ∧ b = grid i1 j1 ∧ (a + b) % 4 = 0 := sorry

end sum_of_adjacent_cells_multiple_of_4_l547_54779


namespace calculate_f_f_f_one_l547_54744

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem calculate_f_f_f_one : f (f (f 1)) = 9184 :=
by
  sorry

end calculate_f_f_f_one_l547_54744


namespace Alfred_repair_cost_l547_54797

noncomputable def scooter_price : ℕ := 4700
noncomputable def sale_price : ℕ := 5800
noncomputable def gain_percent : ℚ := 9.433962264150944
noncomputable def gain_value (repair_cost : ℚ) : ℚ := sale_price - (scooter_price + repair_cost)

theorem Alfred_repair_cost : ∃ R : ℚ, gain_percent = (gain_value R / (scooter_price + R)) * 100 ∧ R = 600 :=
by
  sorry

end Alfred_repair_cost_l547_54797


namespace inequality_solution_l547_54725

noncomputable def condition (x : ℝ) : Prop :=
  2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))
  ∧ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2

theorem inequality_solution (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) (h₁ : condition x) :
  Real.cos x ≤ Real.sqrt (2:ℝ) / 2 ∧ x ∈ [Real.pi/4, 7 * Real.pi/4] := sorry

end inequality_solution_l547_54725


namespace mr_smith_spends_l547_54727

def buffet_price 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (senior_discount : ℕ) 
  (num_full_price_adults : ℕ) 
  (num_children : ℕ) 
  (num_seniors : ℕ) : ℕ :=
  num_full_price_adults * adult_price + num_children * child_price + num_seniors * (adult_price - (adult_price * senior_discount / 100))

theorem mr_smith_spends (adult_price : ℕ) (child_price : ℕ) (senior_discount : ℕ) (num_full_price_adults : ℕ) (num_children : ℕ) (num_seniors : ℕ) : 
  adult_price = 30 → 
  child_price = 15 → 
  senior_discount = 10 → 
  num_full_price_adults = 3 → 
  num_children = 3 → 
  num_seniors = 1 → 
  buffet_price adult_price child_price senior_discount num_full_price_adults num_children num_seniors = 162 :=
by 
  intros h_adult_price h_child_price h_senior_discount h_num_full_price_adults h_num_children h_num_seniors
  rw [h_adult_price, h_child_price, h_senior_discount, h_num_full_price_adults, h_num_children, h_num_seniors]
  sorry

end mr_smith_spends_l547_54727


namespace point_on_graph_of_inverse_proportion_l547_54715

theorem point_on_graph_of_inverse_proportion :
  ∃ x y : ℝ, (x = 2 ∧ y = 4) ∧ y = 8 / x :=
by
  sorry

end point_on_graph_of_inverse_proportion_l547_54715


namespace expr_undefined_iff_l547_54701

theorem expr_undefined_iff (b : ℝ) : ¬ ∃ y : ℝ, y = (b - 1) / (b^2 - 9) ↔ b = -3 ∨ b = 3 :=
by 
  sorry

end expr_undefined_iff_l547_54701


namespace maximize_quadratic_function_l547_54735

theorem maximize_quadratic_function (x : ℝ) :
  (∀ x, -2 * x ^ 2 - 8 * x + 18 ≤ 26) ∧ (-2 * (-2) ^ 2 - 8 * (-2) + 18 = 26) :=
by (
  sorry
)

end maximize_quadratic_function_l547_54735


namespace shaded_area_is_correct_l547_54728

noncomputable def total_shaded_area : ℝ :=
  let s := 10
  let R := s / (2 * Real.sin (Real.pi / 8))
  let A := (1 / 2) * R^2 * Real.sin (2 * Real.pi / 8)
  4 * A

theorem shaded_area_is_correct :
  total_shaded_area = 200 * Real.sqrt 2 / Real.sin (Real.pi / 8)^2 := 
sorry

end shaded_area_is_correct_l547_54728


namespace midpoint_trajectory_l547_54790

-- Define the data for the problem
variable (P : (ℝ × ℝ)) (Q : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable (x y : ℝ)
variable (hQ : Q = (2*x - 2, 2*y)) -- Definition of point Q based on midpoint M
variable (hC : (Q.1)^2 + (Q.2)^2 = 1) -- Q moves on the circle x^2 + y^2 = 1

-- Define the proof problem
theorem midpoint_trajectory (P : (ℝ × ℝ)) (hP : P = (2, 0)) (M : ℝ × ℝ) (hQ : Q = (2*M.1 - 2, 2*M.2))
  (hC : (Q.1)^2 + (Q.2)^2 = 1) : 4*(M.1 - 1)^2 + 4*(M.2)^2 = 1 := by
  sorry

end midpoint_trajectory_l547_54790


namespace prob_heads_even_correct_l547_54770

noncomputable def prob_heads_even (n : Nat) : ℝ :=
  if n = 0 then 1
  else (2 / 3) - (1 / 3) * prob_heads_even (n - 1)

theorem prob_heads_even_correct : 
  prob_heads_even 50 = (1 / 2) * (1 + (1 / 3 ^ 50)) :=
sorry

end prob_heads_even_correct_l547_54770


namespace unique_zero_of_quadratic_l547_54751

theorem unique_zero_of_quadratic {m : ℝ} (h : ∃ x : ℝ, x^2 + 2*x + m = 0 ∧ (∀ y : ℝ, y^2 + 2*y + m = 0 → y = x)) : m = 1 :=
sorry

end unique_zero_of_quadratic_l547_54751


namespace convert_base10_to_base7_l547_54754

-- Definitions for powers and conditions
def n1 : ℕ := 7
def n2 : ℕ := n1 * n1
def n3 : ℕ := n2 * n1
def n4 : ℕ := n3 * n1

theorem convert_base10_to_base7 (n : ℕ) (h₁ : n = 395) : 
  ∃ a b c d : ℕ, 
    a * n3 + b * n2 + c * n1 + d = 395 ∧
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 3 :=
by { sorry }

end convert_base10_to_base7_l547_54754


namespace largest_divisor_of_n4_minus_n_l547_54741

theorem largest_divisor_of_n4_minus_n (n : ℤ) (h : ∃ k : ℤ, n = 4 * k) : 4 ∣ (n^4 - n) :=
by sorry

end largest_divisor_of_n4_minus_n_l547_54741


namespace length_minus_width_l547_54755

theorem length_minus_width 
  (area length diff width : ℝ)
  (h_area : area = 171)
  (h_length : length = 19.13)
  (h_diff : diff = length - width)
  (h_area_eq : area = length * width) :
  diff = 10.19 := 
by {
  sorry
}

end length_minus_width_l547_54755


namespace proof_problem_l547_54700

-- Definitions of points and vectors
def C : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (3, 4)
def N : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (0, 1)

-- Definition of vector operations
def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2)

-- Vectors needed
def AC : ℝ × ℝ := vector_sub C A
def AM : ℝ × ℝ := vector_sub M A
def AN : ℝ × ℝ := vector_sub N A

-- The Lean proof statement
theorem proof_problem :
  (∃ (x y : ℝ), AC = (x * AM.1 + y * AN.1, x * AM.2 + y * AN.2) ∧
     (x, y) = (2 / 3, 1 / 2)) ∧
  (9 * (2 / 3:ℝ) ^ 2 + 16 * (1 / 2:ℝ) ^ 2 = 8) :=
by
  sorry

end proof_problem_l547_54700


namespace complete_square_solution_l547_54745

theorem complete_square_solution (x : ℝ) :
  (x^2 + 6 * x - 4 = 0) → ((x + 3)^2 = 13) :=
by
  sorry

end complete_square_solution_l547_54745


namespace loss_per_meter_is_five_l547_54743

def cost_price_per_meter : ℝ := 50
def total_meters_sold : ℝ := 400
def selling_price : ℝ := 18000

noncomputable def total_cost_price : ℝ := cost_price_per_meter * total_meters_sold
noncomputable def total_loss : ℝ := total_cost_price - selling_price
noncomputable def loss_per_meter : ℝ := total_loss / total_meters_sold

theorem loss_per_meter_is_five : loss_per_meter = 5 :=
by sorry

end loss_per_meter_is_five_l547_54743


namespace part1_part2_part3_l547_54737

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2
noncomputable def h (x : ℝ) : ℝ := f (x + 1) - g' x

theorem part1 : ∃ x : ℝ, (h x ≤ 2) := sorry

theorem part2 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) : 
  f (a + b) - f (2 * a) < (b - a) / (2 * a) := sorry

theorem part3 (k : ℤ) : (∀ x : ℝ, x > 1 → k * (x - 1) < x * f x + 3 * g' x + 4) ↔ k ≤ 5 := sorry

end part1_part2_part3_l547_54737


namespace great_white_shark_teeth_is_420_l547_54703

-- Define the number of teeth in a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Define the number of teeth in a hammerhead shark based on the tiger shark's teeth
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Define the number of teeth in a great white shark based on the sum of tiger and hammerhead shark's teeth
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- The theorem statement that we need to prove
theorem great_white_shark_teeth_is_420 : great_white_shark_teeth = 420 :=
by
  -- Provide space for the proof
  sorry

end great_white_shark_teeth_is_420_l547_54703


namespace power_quotient_example_l547_54774

theorem power_quotient_example (a : ℕ) (m n : ℕ) (h : 23^11 / 23^8 = 23^(11 - 8)) : 23^3 = 12167 := by
  sorry

end power_quotient_example_l547_54774


namespace point_above_line_range_l547_54792

theorem point_above_line_range (a : ℝ) :
  (2 * a - (-1) + 1 < 0) ↔ a < -1 :=
by
  sorry

end point_above_line_range_l547_54792


namespace parity_of_f_monotonicity_of_f_9_l547_54753

-- Condition: f(x) = x + k / x with k ≠ 0
variable (k : ℝ) (hkn0 : k ≠ 0)
noncomputable def f (x : ℝ) : ℝ := x + k / x

-- 1. Prove the parity of the function is odd
theorem parity_of_f : ∀ x : ℝ, f k (-x) = -f k x := by
  sorry

-- Given condition: f(3) = 6, we derive k = 9
def k_9 : ℝ := 9
noncomputable def f_9 (x : ℝ) : ℝ := x + k_9 / x

-- 2. Prove the monotonicity of the function y = f(x) in the interval (-∞, -3]
theorem monotonicity_of_f_9 : ∀ (x1 x2 : ℝ), x1 < x2 → x1 ≤ -3 → x2 ≤ -3 → f_9 x1 < f_9 x2 := by
  sorry

end parity_of_f_monotonicity_of_f_9_l547_54753


namespace inequality_proof_l547_54769

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (1 + a)^2) + (1 / (1 + b)^2) + (1 / (1 + c)^2) + (1 / (1 + d)^2) ≥ 1 :=
by
  sorry

end inequality_proof_l547_54769


namespace problem_statement_l547_54786

theorem problem_statement (a b c : ℝ)
  (h : a * b * c = ( Real.sqrt ( (a + 2) * (b + 3) ) ) / (c + 1)) :
  6 * 15 * 7 = 1.5 :=
sorry

end problem_statement_l547_54786


namespace div30k_929260_l547_54781

theorem div30k_929260 (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 1 := by
  sorry

end div30k_929260_l547_54781


namespace value_of_x_squared_plus_one_over_x_squared_l547_54758

noncomputable def x: ℝ := sorry

theorem value_of_x_squared_plus_one_over_x_squared (h : 20 = x^6 + 1 / x^6) : x^2 + 1 / x^2 = 23 :=
sorry

end value_of_x_squared_plus_one_over_x_squared_l547_54758


namespace Berry_temperature_on_Sunday_l547_54789

theorem Berry_temperature_on_Sunday :
  let avg_temp := 99.0
  let days_in_week := 7
  let temp_day1 := 98.2
  let temp_day2 := 98.7
  let temp_day3 := 99.3
  let temp_day4 := 99.8
  let temp_day5 := 99.0
  let temp_day6 := 98.9
  let total_temp_week := avg_temp * days_in_week
  let total_temp_six_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5 + temp_day6
  let temp_on_sunday := total_temp_week - total_temp_six_days
  temp_on_sunday = 98.1 :=
by
  -- Proof of the statement goes here
  sorry

end Berry_temperature_on_Sunday_l547_54789


namespace hiking_supplies_l547_54771

theorem hiking_supplies (hours_per_day : ℕ) (days : ℕ) (rate_mph : ℝ) 
    (supply_per_mile : ℝ) (resupply_rate : ℝ)
    (initial_pack_weight : ℝ) : 
    hours_per_day = 8 → days = 5 → rate_mph = 2.5 → 
    supply_per_mile = 0.5 → resupply_rate = 0.25 → 
    initial_pack_weight = (40 : ℝ) :=
by
  intros hpd hd rm spm rr
  sorry

end hiking_supplies_l547_54771


namespace range_of_m_l547_54740

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x - 3| ≤ 2 → 1 ≤ x ∧ x ≤ 5) → 
  (∀ x : ℝ, (x - m + 1) * (x - m - 1) ≤ 0 → m - 1 ≤ x ∧ x ≤ m + 1) → 
  (∀ x : ℝ, x < 1 ∨ x > 5 → x < m - 1 ∨ x > m + 1) → 
  2 ≤ m ∧ m ≤ 4 := 
by
  sorry

end range_of_m_l547_54740


namespace satisfy_inequality_l547_54746

theorem satisfy_inequality (x : ℤ) : 
  (3 * x - 5 ≤ 10 - 2 * x) ↔ (x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
sorry

end satisfy_inequality_l547_54746


namespace binary_addition_and_subtraction_correct_l547_54711

def add_binary_and_subtract : ℕ :=
  let n1 := 0b1101  -- binary for 1101_2
  let n2 := 0b0010  -- binary for 10_2
  let n3 := 0b0101  -- binary for 101_2
  let n4 := 0b1011  -- expected result 1011_2
  n1 + n2 + n3 - 0b0011  -- subtract binary for 11_2

theorem binary_addition_and_subtraction_correct : add_binary_and_subtract = 0b1011 := 
by 
  sorry

end binary_addition_and_subtraction_correct_l547_54711


namespace nina_total_money_l547_54773

def original_cost_widget (C : ℝ) : ℝ := C
def num_widgets_nina_can_buy_original (C : ℝ) : ℝ := 6
def num_widgets_nina_can_buy_reduced (C : ℝ) : ℝ := 8
def cost_reduction : ℝ := 1.5

theorem nina_total_money (C : ℝ) (hc : 6 * C = 8 * (C - cost_reduction)) : 
  6 * C = 36 :=
by
  sorry

end nina_total_money_l547_54773


namespace sufficient_but_not_necessary_condition_l547_54721

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3
def g (k x : ℝ) : ℝ := k * x - 1

theorem sufficient_but_not_necessary_condition (k : ℝ) :
  (∀ x : ℝ, f x ≥ g k x) ↔ (-6 ≤ k ∧ k ≤ 2) :=
sorry

end sufficient_but_not_necessary_condition_l547_54721


namespace simplify_trig_expression_l547_54796

open Real

/-- 
Given that θ is in the interval (π/2, π), simplify the expression 
( sin θ / sqrt (1 - sin^2 θ) ) + ( sqrt (1 - cos^2 θ) / cos θ ) to 0.
-/
theorem simplify_trig_expression (θ : ℝ) (hθ1 : π / 2 < θ) (hθ2 : θ < π) :
  (sin θ / sqrt (1 - sin θ ^ 2)) + (sqrt (1 - cos θ ^ 2) / cos θ) = 0 :=
by 
  sorry

end simplify_trig_expression_l547_54796


namespace number_of_tables_l547_54793

theorem number_of_tables (last_year_distance : ℕ) (factor : ℕ) 
  (distance_between_table_1_and_3 : ℕ) (number_of_tables : ℕ) :
  (last_year_distance = 300) ∧ 
  (factor = 4) ∧ 
  (distance_between_table_1_and_3 = 400) ∧
  (number_of_tables = ((factor * last_year_distance) / (distance_between_table_1_and_3 / 2)) + 1) 
  → number_of_tables = 7 :=
by
  intros
  sorry

end number_of_tables_l547_54793


namespace amc_inequality_l547_54756

theorem amc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := 
by 
  sorry

end amc_inequality_l547_54756


namespace m_gt_p_l547_54729

theorem m_gt_p (p m n : ℕ) (prime_p : Nat.Prime p) (pos_m : 0 < m) (pos_n : 0 < n) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end m_gt_p_l547_54729


namespace terminal_side_quadrant_l547_54762

theorem terminal_side_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) :
  ∃ k : ℤ, (k % 2 = 0 ∧ (k * Real.pi + Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + Real.pi / 2)) ∨
           (k % 2 = 1 ∧ (k * Real.pi + 3 * Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + 5 * Real.pi / 4)) := sorry

end terminal_side_quadrant_l547_54762


namespace smallest_d_for_inverse_l547_54799

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse :
  ∃ d, (∀ x₁ x₂, d ≤ x₁ ∧ d ≤ x₂ ∧ g x₁ = g x₂ → x₁ = x₂) ∧ (∀ e, (∀ x₁ x₂, e ≤ x₁ ∧ e ≤ x₂ ∧ g x₁ = g x₂ → x₁ = x₂) → d ≤ e) ∧ d = 3 :=
by
  sorry

end smallest_d_for_inverse_l547_54799


namespace jerry_age_l547_54761

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J + 10) (h2 : M = 30) : J = 5 := by
  sorry

end jerry_age_l547_54761


namespace sum_a1_a5_l547_54798

def sequence_sum (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 1

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h_sum : sequence_sum S)
  (h_a1 : a 1 = S 1)
  (h_a5 : a 5 = S 5 - S 4) :
  a 1 + a 5 = 11 := by
  sorry

end sum_a1_a5_l547_54798


namespace trig_identity_l547_54749

theorem trig_identity :
  (Real.cos (80 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) + 
   Real.sin (80 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  sorry

end trig_identity_l547_54749


namespace root_conditions_l547_54780

-- Given conditions and definitions:
def quadratic_eq (m x : ℝ) : ℝ := x^2 + (m - 3) * x + m

-- The proof problem statement
theorem root_conditions (m : ℝ) (h1 : ∃ x y : ℝ, quadratic_eq m x = 0 ∧ quadratic_eq m y = 0 ∧ x > 1 ∧ y < 1) : m < 1 :=
sorry

end root_conditions_l547_54780


namespace personBCatchesPersonAAtB_l547_54760

-- Definitions based on the given problem's conditions
def personADepartsTime : ℕ := 8 * 60  -- Person A departs at 8:00 AM, given in minutes
def personBDepartsTime : ℕ := 9 * 60  -- Person B departs at 9:00 AM, given in minutes
def catchUpTime : ℕ := 11 * 60        -- Persons meet at 11:00 AM, given in minutes
def returnMultiplier : ℕ := 2         -- Person B returns at double the speed
def chaseMultiplier : ℕ := 2          -- After returning, Person B doubles their speed again

-- Exact question we want to prove
def meetAtBTime : ℕ := 12 * 60 + 48   -- Time when Person B catches up with Person A at point B

-- Statement to be proven
theorem personBCatchesPersonAAtB :
  ∀ (VA VB : ℕ) (x : ℕ),
    VA = 2 * x ∧ VB = 3 * x →
    ∃ t : ℕ, t = meetAtBTime := by
  sorry

end personBCatchesPersonAAtB_l547_54760


namespace chris_money_before_birthday_l547_54707

-- Define the given amounts of money from each source
def money_from_grandmother : ℕ := 25
def money_from_aunt_and_uncle : ℕ := 20
def money_from_parents : ℕ := 75
def total_money_now : ℕ := 279

-- Calculate the total birthday money
def total_birthday_money := money_from_grandmother + money_from_aunt_and_uncle + money_from_parents

-- Define the amount of money Chris had before his birthday
def money_before_birthday := total_money_now - total_birthday_money

-- The proof statement
theorem chris_money_before_birthday : money_before_birthday = 159 :=
by
  sorry

end chris_money_before_birthday_l547_54707


namespace factor_3x2_minus_3y2_l547_54763

theorem factor_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factor_3x2_minus_3y2_l547_54763


namespace fermat_prime_divisibility_l547_54723

def F (k : ℕ) : ℕ := 2 ^ 2 ^ k + 1

theorem fermat_prime_divisibility {m n : ℕ} (hmn : m > n) : F n ∣ (F m - 2) :=
sorry

end fermat_prime_divisibility_l547_54723


namespace nina_money_l547_54719

-- Definitions based on the problem's conditions
def original_widgets := 15
def reduced_widgets := 25
def price_reduction := 5

-- The statement
theorem nina_money : 
  ∃ (W : ℝ), 15 * W = 25 * (W - 5) ∧ 15 * W = 187.5 :=
by
  sorry

end nina_money_l547_54719


namespace domain_width_of_g_l547_54738

theorem domain_width_of_g (h : ℝ → ℝ) (domain_h : ∀ x, -8 ≤ x ∧ x ≤ 8 → h x = h x) :
  let g (x : ℝ) := h (x / 2)
  ∃ a b, (∀ x, a ≤ x ∧ x ≤ b → ∃ y, g x = y) ∧ (b - a = 32) := 
sorry

end domain_width_of_g_l547_54738


namespace child_wants_to_buy_3_toys_l547_54785

/- 
  Problem Statement:
  There are 10 toys, and the number of ways to select a certain number 
  of those toys in any order is 120. We need to find out how many toys 
  were selected.
-/

def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem child_wants_to_buy_3_toys :
  ∃ r : ℕ, r ≤ 10 ∧ comb 10 r = 120 :=
by
  use 3
  -- Here you would write the proof
  sorry

end child_wants_to_buy_3_toys_l547_54785


namespace find_nat_numbers_l547_54739

theorem find_nat_numbers (a b : ℕ) (h : 1 / (a - b) = 3 * (1 / (a * b))) : a = 6 ∧ b = 2 :=
sorry

end find_nat_numbers_l547_54739


namespace nails_painted_purple_l547_54718

variable (P S : ℕ)

theorem nails_painted_purple :
  (P + 8 + S = 20) ∧ ((8 / 20 : ℚ) * 100 - (S / 20 : ℚ) * 100 = 10) → P = 6 :=
by
  sorry

end nails_painted_purple_l547_54718


namespace team_b_can_serve_on_submarine_l547_54712

   def can_serve_on_submarine (height : ℝ) : Prop := height ≤ 168

   def average_height_condition (avg_height : ℝ) : Prop := avg_height = 166

   def median_height_condition (median_height : ℝ) : Prop := median_height = 167

   def tallest_height_condition (max_height : ℝ) : Prop := max_height = 169

   def mode_height_condition (mode_height : ℝ) : Prop := mode_height = 167

   theorem team_b_can_serve_on_submarine (H : median_height_condition 167) :
     ∀ (h : ℝ), can_serve_on_submarine h :=
   sorry
   
end team_b_can_serve_on_submarine_l547_54712


namespace digit_D_eq_9_l547_54730

-- Define digits and the basic operations on 2-digit numbers
def is_digit (n : ℕ) : Prop := n < 10
def tens (n : ℕ) : ℕ := n / 10
def units (n : ℕ) : ℕ := n % 10
def two_digit (a b : ℕ) : ℕ := 10 * a + b

theorem digit_D_eq_9 (A B C D : ℕ):
  is_digit A → is_digit B → is_digit C → is_digit D →
  (two_digit A B) + (two_digit C B) = two_digit D A →
  (two_digit A B) - (two_digit C B) = A →
  D = 9 :=
by sorry

end digit_D_eq_9_l547_54730


namespace simplify_expression_l547_54716

variable (x y : ℝ)

theorem simplify_expression :
  3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + 2 * y) = 9 * x^2 + 6 * x - 2 * y - 3 :=
by
  sorry

end simplify_expression_l547_54716


namespace overall_percentage_decrease_l547_54709

-- Define the initial pay cut percentages as given in the conditions.
def first_pay_cut := 5.25 / 100
def second_pay_cut := 9.75 / 100
def third_pay_cut := 14.6 / 100
def fourth_pay_cut := 12.8 / 100

-- Define the single shot percentage decrease we want to prove.
def single_shot_decrease := 36.73 / 100

-- Calculate the cumulative multiplier from individual pay cuts.
def cumulative_multiplier := 
  (1 - first_pay_cut) * (1 - second_pay_cut) * (1 - third_pay_cut) * (1 - fourth_pay_cut)

-- Statement: Prove the overall percentage decrease using cumulative multiplier is equal to single shot decrease.
theorem overall_percentage_decrease :
  1 - cumulative_multiplier = single_shot_decrease :=
by sorry

end overall_percentage_decrease_l547_54709


namespace sum_of_factorials_is_perfect_square_l547_54726

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).map factorial |>.sum

theorem sum_of_factorials_is_perfect_square (n : ℕ) (h : n > 0) :
  (∃ m : ℕ, m * m = sum_of_factorials n) ↔ (n = 1 ∨ n = 3) := 
sorry

end sum_of_factorials_is_perfect_square_l547_54726


namespace hyperbola_equations_l547_54757

def eq1 (x y : ℝ) : Prop := x^2 - 4 * y^2 = (5 + Real.sqrt 6)^2
def eq2 (x y : ℝ) : Prop := 4 * y^2 - x^2 = 4

theorem hyperbola_equations 
  (x y : ℝ)
  (hx1 : x - 2 * y = 0)
  (hx2 : x + 2 * y = 0)
  (dist : Real.sqrt ((x - 5)^2 + y^2) = Real.sqrt 6) :
  eq1 x y ∧ eq2 x y := 
sorry

end hyperbola_equations_l547_54757


namespace initial_alarm_time_was_l547_54768

def faster_watch_gain (rate : ℝ) (hours : ℝ) : ℝ := hours * rate

def absolute_time_difference (faster_time : ℝ) (correct_time : ℝ) : ℝ := faster_time - correct_time

theorem initial_alarm_time_was :
  ∀ (rate minutes time_difference : ℝ),
  rate = 2 →
  minutes = 12 →
  time_difference = minutes / rate →
  abs (4 - (4 - time_difference)) = 6 →
  (24 - 6) = 22 :=
by
  intros rate minutes time_difference hrate hminutes htime_diff htime
  sorry

end initial_alarm_time_was_l547_54768


namespace ratio_of_girls_to_boys_l547_54750

variable (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : g + b = 36)
                               (h₂ : g = b + 6) : g / b = 7 / 5 :=
by sorry

end ratio_of_girls_to_boys_l547_54750


namespace charity_event_assignment_l547_54795

theorem charity_event_assignment (students : Finset ℕ) (h_students : students.card = 5) :
  ∃ (num_ways : ℕ), num_ways = 60 :=
by
  let select_two_for_friday := Nat.choose 5 2
  let remaining_students_after_friday := 5 - 2
  let select_one_for_saturday := Nat.choose remaining_students_after_friday 1
  let remaining_students_after_saturday := remaining_students_after_friday - 1
  let select_one_for_sunday := Nat.choose remaining_students_after_saturday 1
  let total_ways := select_two_for_friday * select_one_for_saturday * select_one_for_sunday
  use total_ways
  sorry

end charity_event_assignment_l547_54795


namespace remainder_when_divided_by_five_l547_54767

theorem remainder_when_divided_by_five :
  let E := 1250 * 1625 * 1830 * 2075 + 245
  E % 5 = 0 := by
  sorry

end remainder_when_divided_by_five_l547_54767


namespace transistors_in_2002_transistors_in_2010_l547_54736

-- Definitions based on the conditions
def mooresLawDoubling (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

-- Conditions
def initial_transistors := 2000000
def year_1992 := 1992
def year_2002 := 2002
def year_2010 := 2010

-- Questions translated into proof targets
theorem transistors_in_2002 : mooresLawDoubling initial_transistors (year_2002 - year_1992) = 64000000 := by
  sorry

theorem transistors_in_2010 : mooresLawDoubling (mooresLawDoubling initial_transistors (year_2002 - year_1992)) (year_2010 - year_2002) = 1024000000 := by
  sorry

end transistors_in_2002_transistors_in_2010_l547_54736


namespace expression_eval_l547_54778

noncomputable def a : ℕ := 2001
noncomputable def b : ℕ := 2003

theorem expression_eval : 
  b^3 - a * b^2 - a^2 * b + a^3 = 8 :=
by sorry

end expression_eval_l547_54778


namespace sequence_general_term_l547_54759

theorem sequence_general_term (n : ℕ) (hn : 0 < n) : 
  ∃ (a_n : ℕ), a_n = 2 * Int.floor (Real.sqrt (n - 1)) + 1 :=
by
  sorry

end sequence_general_term_l547_54759


namespace cows_total_l547_54713

theorem cows_total (A M R : ℕ) (h1 : A = 4 * M) (h2 : M = 60) (h3 : A + M = R + 30) : 
  A + M + R = 570 := by
  sorry

end cows_total_l547_54713


namespace rational_numbers_sum_reciprocal_integer_l547_54782

theorem rational_numbers_sum_reciprocal_integer (p1 q1 p2 q2 : ℤ) (k m : ℤ)
  (h1 : Int.gcd p1 q1 = 1)
  (h2 : Int.gcd p2 q2 = 1)
  (h3 : p1 * q2 + p2 * q1 = k * q1 * q2)
  (h4 : q1 * p2 + q2 * p1 = m * p1 * p2) :
  (p1, q1, p2, q2) = (x, y, -x, y) ∨
  (p1, q1, p2, q2) = (2, 1, 2, 1) ∨
  (p1, q1, p2, q2) = (-2, 1, -2, 1) ∨
  (p1, q1, p2, q2) = (1, 1, 1, 1) ∨
  (p1, q1, p2, q2) = (-1, 1, -1, 1) ∨
  (p1, q1, p2, q2) = (1, 2, 1, 2) ∨
  (p1, q1, p2, q2) = (-1, 2, -1, 2) :=
sorry

end rational_numbers_sum_reciprocal_integer_l547_54782


namespace axis_of_symmetry_of_parabola_l547_54733

-- Definitions (from conditions):
def quadratic_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def is_root_of_quadratic (a b c x : ℝ) : Prop := quadratic_equation a b c x = 0

-- Given conditions
variables {a b c : ℝ}
variable (h_a_nonzero : a ≠ 0)
variable (h_root1 : is_root_of_quadratic a b c 1)
variable (h_root2 : is_root_of_quadratic a b c 5)

-- Problem statement
theorem axis_of_symmetry_of_parabola : (3 : ℝ) = (1 + 5) / 2 :=
by
  -- proof omitted
  sorry

end axis_of_symmetry_of_parabola_l547_54733


namespace hyperbola_asymptote_slope_l547_54742

theorem hyperbola_asymptote_slope (m : ℝ) :
  (∀ x y : ℝ, mx^2 + y^2 = 1) →
  (∀ x y : ℝ, y = 2 * x) →
  m = -4 :=
by
  sorry

end hyperbola_asymptote_slope_l547_54742


namespace min_time_adult_worms_l547_54748

noncomputable def f : ℕ → ℝ
| 1 => 0
| n => (1 - 1 / (2 ^ (n - 1)))

theorem min_time_adult_worms (n : ℕ) (h : n ≥ 1) : 
  ∃ min_time : ℝ, 
  (min_time = 1 - 1 / (2 ^ (n - 1))) ∧ 
  (∀ t : ℝ, (t = 1 - 1 / (2 ^ (n - 1)))) := 
sorry

end min_time_adult_worms_l547_54748


namespace jerry_removed_old_figures_l547_54732

-- Let's declare the conditions
variables (initial_count added_count current_count removed_count : ℕ)
variables (h1 : initial_count = 7)
variables (h2 : added_count = 11)
variables (h3 : current_count = 8)

-- The statement to prove
theorem jerry_removed_old_figures : removed_count = initial_count + added_count - current_count :=
by
  -- The proof will go here, but we'll use sorry to skip it
  sorry

end jerry_removed_old_figures_l547_54732


namespace dylan_ice_cubes_l547_54766

-- Definitions based on conditions
def trays := 2
def spaces_per_tray := 12
def total_tray_ice := trays * spaces_per_tray
def pitcher_multiplier := 2

-- The statement to be proven
theorem dylan_ice_cubes (x : ℕ) : x + pitcher_multiplier * x = total_tray_ice → x = 8 :=
by {
  sorry
}

end dylan_ice_cubes_l547_54766


namespace product_of_D_coordinates_l547_54724

theorem product_of_D_coordinates 
  (M D : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hC : C = (5, 3))
  (hM : M = (3, 7))
  (h_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
  D.1 * D.2 = 11 :=
by
  sorry

end product_of_D_coordinates_l547_54724


namespace molecular_weight_calculation_l547_54794

/-- Define the molecular weight of the compound as 972 grams per mole. -/
def molecular_weight : ℕ := 972

/-- Define the number of moles as 9 moles. -/
def number_of_moles : ℕ := 9

/-- Define the total weight of the compound for the given number of moles. -/
def total_weight : ℕ := number_of_moles * molecular_weight

/-- Prove the total weight is 8748 grams. -/
theorem molecular_weight_calculation : total_weight = 8748 := by
  sorry

end molecular_weight_calculation_l547_54794


namespace original_manufacturing_cost_l547_54777

variable (SP OC : ℝ)
variable (ManuCost : ℝ) -- Declaring manufacturing cost

-- Current conditions
axiom profit_percentage_constant : ∀ SP, 0.5 * SP = SP - 50

-- Problem Statement
theorem original_manufacturing_cost : (∃ OC, 0.5 * SP - OC = 0.5 * SP) ∧ ManuCost = 50 → OC = 50 := by
  sorry

end original_manufacturing_cost_l547_54777


namespace room_width_correct_l547_54772

noncomputable def length_of_room : ℝ := 5
noncomputable def total_cost_of_paving : ℝ := 21375
noncomputable def cost_per_square_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem room_width_correct :
  (total_cost_of_paving / cost_per_square_meter) = (length_of_room * width_of_room) :=
by
  sorry

end room_width_correct_l547_54772


namespace elizabeth_net_profit_l547_54784

noncomputable section

def net_profit : ℝ :=
  let cost_bag_1 := 2.5
  let cost_bag_2 := 3.5
  let total_cost := 10 * cost_bag_1 + 10 * cost_bag_2
  let selling_price := 6.0
  let sold_bags_1_no_discount := 7 * selling_price
  let sold_bags_2_no_discount := 8 * selling_price
  let discount_1 := 0.2
  let discount_2 := 0.3
  let discounted_price_1 := selling_price * (1 - discount_1)
  let discounted_price_2 := selling_price * (1 - discount_2)
  let sold_bags_1_with_discount := 3 * discounted_price_1
  let sold_bags_2_with_discount := 2 * discounted_price_2
  let total_revenue := sold_bags_1_no_discount + sold_bags_2_no_discount + sold_bags_1_with_discount + sold_bags_2_with_discount
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.8 := by
  sorry

end elizabeth_net_profit_l547_54784
