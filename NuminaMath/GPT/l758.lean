import Mathlib

namespace circle_tangent_area_l758_75850

noncomputable def circle_tangent_area_problem 
  (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) : ℝ :=
  if (radiusA = 1 ∧ radiusB = 1 ∧ radiusC = 2 ∧ tangent_midpoint) then 
    (4 * Real.pi) - (2 * Real.pi) 
  else 0

theorem circle_tangent_area (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) :
  radiusA = 1 → radiusB = 1 → radiusC = 2 → tangent_midpoint = true → 
  circle_tangent_area_problem radiusA radiusB radiusC tangent_midpoint = 2 * Real.pi :=
by
  intros
  simp [circle_tangent_area_problem]
  split_ifs
  · sorry
  · sorry

end circle_tangent_area_l758_75850


namespace leah_coins_value_l758_75865

theorem leah_coins_value :
  ∃ (p n d : ℕ), 
    p + n + d = 20 ∧
    p = n ∧
    p = d + 4 ∧
    1 * p + 5 * n + 10 * d = 88 :=
by
  sorry

end leah_coins_value_l758_75865


namespace profit_maximization_l758_75888

-- Define the conditions 
variable (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5)

-- Expression for yield ω
noncomputable def yield (x : ℝ) : ℝ := 4 - (3 / (x + 1))

-- Expression for profit function L(x)
noncomputable def profit (x : ℝ) : ℝ := 16 * yield x - x - 2 * x

-- Theorem stating the profit function expression and its maximum
theorem profit_maximization (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5) :
  profit x = 64 - 48 / (x + 1) - 3 * x ∧ 
  (∀ x₀, 0 ≤ x₀ ∧ x₀ ≤ 5 → profit x₀ ≤ profit 3) :=
sorry

end profit_maximization_l758_75888


namespace smallest_x_for_div_by9_l758_75846

-- Define the digit sum of the number 761*829 with a placeholder * for x
def digit_sum_with_x (x : Nat) : Nat :=
  7 + 6 + 1 + x + 8 + 2 + 9

-- State the theorem to prove the smallest value of x makes the sum divisible by 9
theorem smallest_x_for_div_by9 : ∃ x : Nat, digit_sum_with_x x % 9 = 0 ∧ (∀ y : Nat, y < x → digit_sum_with_x y % 9 ≠ 0) :=
sorry

end smallest_x_for_div_by9_l758_75846


namespace combined_capacity_is_40_l758_75817

/-- Define the bus capacity as 1/6 the train capacity -/
def bus_capacity (train_capacity : ℕ) := train_capacity / 6

/-- There are two buses in the problem -/
def number_of_buses := 2

/-- The train capacity given in the problem is 120 people -/
def train_capacity := 120

/-- The combined capacity of the two buses is -/
def combined_bus_capacity := number_of_buses * bus_capacity train_capacity

/-- Proof that the combined capacity of the two buses is 40 people -/
theorem combined_capacity_is_40 : combined_bus_capacity = 40 := by
  -- Proof will be filled in here
  sorry

end combined_capacity_is_40_l758_75817


namespace final_state_probability_l758_75863

-- Define the initial state and conditions of the problem
structure GameState where
  raashan : ℕ
  sylvia : ℕ
  ted : ℕ
  uma : ℕ

-- Conditions: each player starts with $2, and the game evolves over 500 rounds
def initial_state : GameState :=
  { raashan := 2, sylvia := 2, ted := 2, uma := 2 }

def valid_statements (state : GameState) : Prop :=
  state.raashan = 2 ∧ state.sylvia = 2 ∧ state.ted = 2 ∧ state.uma = 2

-- Final theorem statement
theorem final_state_probability :
  let states := 500 -- representing the number of rounds
  -- proof outline implies that after the games have properly transitioned and bank interactions, the probability is calculated
  -- state after the transitions
  ∃ (prob : ℚ), prob = 1/4 ∧ valid_statements initial_state :=
  sorry

end final_state_probability_l758_75863


namespace find_constants_l758_75855

open Matrix 

noncomputable def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem find_constants :
  let s := (-10 : ℤ)
  let t := (-8 : ℤ)
  let u := (-36 : ℤ)
  B^3 + s • (B^2) + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := sorry

end find_constants_l758_75855


namespace sam_quarters_mowing_lawns_l758_75868

-- Definitions based on the given conditions
def pennies : ℕ := 9
def total_amount_dollars : ℝ := 1.84
def penny_value_dollars : ℝ := 0.01
def quarter_value_dollars : ℝ := 0.25

-- Theorem statement that Sam got 7 quarters given the conditions
theorem sam_quarters_mowing_lawns : 
  (total_amount_dollars - pennies * penny_value_dollars) / quarter_value_dollars = 7 := by
  sorry

end sam_quarters_mowing_lawns_l758_75868


namespace ball_travel_distance_five_hits_l758_75882

def total_distance_traveled (h₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  let descents := List.range (n + 1) |>.map (λ i => h₀ * r ^ i)
  let ascents := List.range n |>.map (λ i => h₀ * r ^ (i + 1))
  (descents.sum + ascents.sum)

theorem ball_travel_distance_five_hits :
  total_distance_traveled 120 (3 / 4) 5 = 612.1875 :=
by
  sorry

end ball_travel_distance_five_hits_l758_75882


namespace seashells_total_correct_l758_75839

def total_seashells (red_shells green_shells other_shells : ℕ) : ℕ :=
  red_shells + green_shells + other_shells

theorem seashells_total_correct :
  total_seashells 76 49 166 = 291 :=
by
  sorry

end seashells_total_correct_l758_75839


namespace solution_set_l758_75835

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn (a b : ℝ) : f (a + b) = f a + f b - 1
axiom monotonic (x y : ℝ) : x ≤ y → f x ≤ f y
axiom initial_condition : f 4 = 5

theorem solution_set : {m : ℝ | f (3 * m^2 - m - 2) < 3} = {m : ℝ | -4/3 < m ∧ m < 1} :=
by
  sorry

end solution_set_l758_75835


namespace iron_aluminum_weight_difference_l758_75859

theorem iron_aluminum_weight_difference :
  let iron_weight := 11.17
  let aluminum_weight := 0.83
  iron_weight - aluminum_weight = 10.34 :=
by
  sorry

end iron_aluminum_weight_difference_l758_75859


namespace potion_kits_needed_l758_75870

-- Definitions
def num_spellbooks := 5
def cost_spellbook_gold := 5
def cost_potion_kit_silver := 20
def num_owls := 1
def cost_owl_gold := 28
def silver_per_gold := 9
def total_silver := 537

-- Prove that Harry needs to buy 3 potion kits.
def Harry_needs_to_buy : Prop :=
  let cost_spellbooks_silver := num_spellbooks * cost_spellbook_gold * silver_per_gold
  let cost_owl_silver := num_owls * cost_owl_gold * silver_per_gold
  let total_cost_silver := cost_spellbooks_silver + cost_owl_silver
  let remaining_silver := total_silver - total_cost_silver
  let num_potion_kits := remaining_silver / cost_potion_kit_silver
  num_potion_kits = 3

theorem potion_kits_needed : Harry_needs_to_buy :=
  sorry

end potion_kits_needed_l758_75870


namespace ratio_flowers_l758_75883

theorem ratio_flowers (flowers_monday flowers_tuesday flowers_week total_flowers flowers_friday : ℕ)
    (h_monday : flowers_monday = 4)
    (h_tuesday : flowers_tuesday = 8)
    (h_total : total_flowers = 20)
    (h_week : total_flowers = flowers_monday + flowers_tuesday + flowers_friday) :
    flowers_friday / flowers_monday = 2 :=
by
  sorry

end ratio_flowers_l758_75883


namespace y_not_directly_nor_inversely_proportional_l758_75854

theorem y_not_directly_nor_inversely_proportional (x y : ℝ) :
  (∃ k : ℝ, x + y = 0 ∧ y = k * x) ∨
  (∃ k : ℝ, 3 * x * y = 10 ∧ x * y = k) ∨
  (∃ k : ℝ, x = 5 * y ∧ x = k * y) ∨
  (∃ k : ℝ, (y = 10 - x^2 - 3 * x) ∧ y ≠ k * x ∧ y * x ≠ k) ∨
  (∃ k : ℝ, x / y = Real.sqrt 3 ∧ x = k * y)
  → (∃ k : ℝ, y = 10 - x^2 - 3 * x ∧ y ≠ k * x ∧ y * x ≠ k) :=
by
  sorry

end y_not_directly_nor_inversely_proportional_l758_75854


namespace tank_capacity_l758_75889

variable (C : ℝ)

noncomputable def leak_rate := C / 6 -- litres per hour
noncomputable def inlet_rate := 6 * 60 -- litres per hour
noncomputable def net_emptying_rate := C / 12 -- litres per hour

theorem tank_capacity : 
  (360 - leak_rate C = net_emptying_rate C) → 
  C = 1440 :=
by 
  sorry

end tank_capacity_l758_75889


namespace arithmetic_sqrt_of_25_l758_75875

theorem arithmetic_sqrt_of_25 : ∃ (x : ℝ), x^2 = 25 ∧ x = 5 :=
by 
  sorry

end arithmetic_sqrt_of_25_l758_75875


namespace solve_for_A_l758_75860

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^2
def g (B x : ℝ) : ℝ := B * x^2

-- A Lean theorem that formalizes the given math problem.
theorem solve_for_A (A B : ℝ) (h₁ : B ≠ 0) (h₂ : f A B (g B 1) = 0) : A = 3 :=
by {
  sorry
}

end solve_for_A_l758_75860


namespace sum_of_f_greater_than_zero_l758_75864

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem sum_of_f_greater_than_zero 
  (a b c : ℝ) 
  (h1 : a + b > 0) 
  (h2 : b + c > 0) 
  (h3 : c + a > 0) : 
  f a + f b + f c > 0 := 
by 
  sorry

end sum_of_f_greater_than_zero_l758_75864


namespace cos_diff_l758_75826

theorem cos_diff (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.tan α = 2) : 
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end cos_diff_l758_75826


namespace solve_first_system_solve_second_system_solve_third_system_l758_75807

-- First system of equations
theorem solve_first_system (x y : ℝ) 
  (h1 : 2*x + 3*y = 16)
  (h2 : x + 4*y = 13) : 
  x = 5 ∧ y = 2 := 
sorry

-- Second system of equations
theorem solve_second_system (x y : ℝ) 
  (h1 : 0.3*x - y = 1)
  (h2 : 0.2*x - 0.5*y = 19) : 
  x = 370 ∧ y = 110 := 
sorry

-- Third system of equations
theorem solve_third_system (x y : ℝ) 
  (h1 : 3 * (x - 1) = y + 5)
  (h2 : (x + 2) / 2 = ((y - 1) / 3) + 1) : 
  x = 6 ∧ y = 10 := 
sorry

end solve_first_system_solve_second_system_solve_third_system_l758_75807


namespace b_completes_work_in_48_days_l758_75828

noncomputable def work_rate (days : ℕ) : ℚ := 1 / days

theorem b_completes_work_in_48_days (a b c : ℕ) 
  (h1 : work_rate (a + b) = work_rate 16)
  (h2 : work_rate a = work_rate 24)
  (h3 : work_rate c = work_rate 48) :
  work_rate b = work_rate 48 :=
by
  sorry

end b_completes_work_in_48_days_l758_75828


namespace complementary_angle_beta_l758_75836

theorem complementary_angle_beta (α β : ℝ) (h_compl : α + β = 90) (h_alpha : α = 40) : β = 50 :=
by
  -- Skipping the proof, which initial assumption should be defined.
  sorry

end complementary_angle_beta_l758_75836


namespace find_a9_l758_75833

variable (a : ℕ → ℝ)

theorem find_a9 (h1 : a 4 - a 2 = -2) (h2 : a 7 = -3) : a 9 = -5 :=
sorry

end find_a9_l758_75833


namespace greatest_value_of_x_for_equation_l758_75893

theorem greatest_value_of_x_for_equation :
  ∃ x : ℝ, (4 * x - 5) ≠ 0 ∧ ((5 * x - 20) / (4 * x - 5)) ^ 2 + ((5 * x - 20) / (4 * x - 5)) = 18 ∧ x = 50 / 29 :=
sorry

end greatest_value_of_x_for_equation_l758_75893


namespace cost_of_adult_ticket_l758_75853

def cost_of_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50
def adult_tickets : ℕ := 5

theorem cost_of_adult_ticket
  (A : ℝ)
  (h : 5 * A + 16 * cost_of_child_ticket = total_cost) :
  A = 5.50 :=
by
  sorry

end cost_of_adult_ticket_l758_75853


namespace hyperbola_eccentricity_l758_75879

theorem hyperbola_eccentricity (h : ∀ x y m : ℝ, x^2 - y^2 / m = 1 → m > 0 → (Real.sqrt (1 + m) = Real.sqrt 3)) : ∃ m : ℝ, m = 2 := sorry

end hyperbola_eccentricity_l758_75879


namespace area_triangle_AMC_l758_75841

noncomputable def area_of_triangle_AMC (AB AD AM : ℝ) : ℝ :=
  if AB = 10 ∧ AD = 12 ∧ AM = 9 then
    (1 / 2) * AM * AB
  else 0

theorem area_triangle_AMC :
  ∀ (AB AD AM : ℝ), AB = 10 → AD = 12 → AM = 9 → area_of_triangle_AMC AB AD AM = 45 := by
  intros AB AD AM hAB hAD hAM
  simp [area_of_triangle_AMC, hAB, hAD, hAM]
  sorry

end area_triangle_AMC_l758_75841


namespace solve_system_equation_152_l758_75886

theorem solve_system_equation_152 (x y z a b c : ℝ)
  (h1 : x * y - 2 * y - 3 * x = 0)
  (h2 : y * z - 3 * z - 5 * y = 0)
  (h3 : x * z - 5 * x - 2 * z = 0)
  (h4 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h5 : x = a)
  (h6 : y = b)
  (h7 : z = c) :
  a^2 + b^2 + c^2 = 152 := by
  sorry

end solve_system_equation_152_l758_75886


namespace cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l758_75867

noncomputable def p1 (x : ℝ) : ℝ :=
  if x < -1 ∨ x > 1 then 0 else 0.5

noncomputable def p2 (y : ℝ) : ℝ :=
  if y < 0 ∨ y > 2 then 0 else 0.5

noncomputable def F1 (x : ℝ) : ℝ :=
  if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1

noncomputable def F2 (y : ℝ) : ℝ :=
  if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1

noncomputable def p (x : ℝ) (y : ℝ) : ℝ :=
  if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25

noncomputable def F (x : ℝ) (y : ℝ) : ℝ :=
  if x ≤ -1 ∨ y ≤ 0 then 0
  else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y 
  else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
  else if x > 1 ∧ y ≤ 2 then 0.5 * y
  else 1

theorem cumulative_distribution_F1 (x : ℝ) : 
  F1 x = if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1 := by sorry

theorem cumulative_distribution_F2 (y : ℝ) : 
  F2 y = if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1 := by sorry

theorem joint_density (x : ℝ) (y : ℝ) : 
  p x y = if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25 := by sorry

theorem joint_cumulative_distribution (x : ℝ) (y : ℝ) : 
  F x y = if x ≤ -1 ∨ y ≤ 0 then 0
          else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y
          else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
          else if x > 1 ∧ y ≤ 2 then 0.5 * y
          else 1 := by sorry

end cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l758_75867


namespace binomial_coefficient_19_13_l758_75884

theorem binomial_coefficient_19_13 
  (h1 : Nat.choose 20 13 = 77520) 
  (h2 : Nat.choose 20 14 = 38760) 
  (h3 : Nat.choose 18 13 = 18564) :
  Nat.choose 19 13 = 37128 := 
sorry

end binomial_coefficient_19_13_l758_75884


namespace greatest_b_for_no_minus_nine_in_range_l758_75814

theorem greatest_b_for_no_minus_nine_in_range :
  ∃ b_max : ℤ, (b_max = 16) ∧ (∀ b : ℤ, (b^2 < 288) ↔ (b ≤ 16)) :=
by
  sorry

end greatest_b_for_no_minus_nine_in_range_l758_75814


namespace correct_multiplication_l758_75877

theorem correct_multiplication (n : ℕ) (wrong_answer correct_answer : ℕ) 
    (h1 : wrong_answer = 559981)
    (h2 : correct_answer = 987 * n)
    (h3 : ∃ (x y : ℕ), correct_answer = 500000 + x + 901 + y ∧ x ≠ 98 ∧ y ≠ 98 ∧ (wrong_answer - correct_answer) % 10 = 0) :
    correct_answer = 559989 :=
by
  sorry

end correct_multiplication_l758_75877


namespace at_least_one_term_le_one_l758_75847

theorem at_least_one_term_le_one
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) :
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
  sorry

end at_least_one_term_le_one_l758_75847


namespace thomas_total_drawings_l758_75802

theorem thomas_total_drawings :
  let colored_pencil_drawings := 14
  let blending_marker_drawings := 7
  let charcoal_drawings := 4
  colored_pencil_drawings + blending_marker_drawings + charcoal_drawings = 25 := 
by
  sorry

end thomas_total_drawings_l758_75802


namespace certain_number_approx_l758_75878

theorem certain_number_approx (x : ℝ) : 213 * 16 = 3408 → x * 2.13 = 0.3408 → x = 0.1600 :=
by
  intro h1 h2
  sorry

end certain_number_approx_l758_75878


namespace cost_of_pencils_and_notebooks_l758_75856

variable (p n : ℝ)

theorem cost_of_pencils_and_notebooks 
  (h1 : 9 * p + 10 * n = 5.06) 
  (h2 : 6 * p + 4 * n = 2.42) :
  20 * p + 14 * n = 8.31 :=
by
  sorry

end cost_of_pencils_and_notebooks_l758_75856


namespace second_rate_of_return_l758_75806

namespace Investment

def total_investment : ℝ := 33000
def interest_total : ℝ := 970
def investment_4_percent : ℝ := 13000
def interest_rate_4_percent : ℝ := 0.04

def amount_second_investment : ℝ := total_investment - investment_4_percent
def interest_from_first_part : ℝ := interest_rate_4_percent * investment_4_percent
def interest_from_second_part (R : ℝ) : ℝ := R * amount_second_investment

theorem second_rate_of_return : (∃ R : ℝ, interest_from_first_part + interest_from_second_part R = interest_total) → 
  R = 0.0225 :=
by
  intro h
  sorry

end Investment

end second_rate_of_return_l758_75806


namespace part1_part2_l758_75816

open Complex

noncomputable def z0 : ℂ := 3 + 4 * Complex.I

theorem part1 (z1 : ℂ) (h : z1 * z0 = 3 * z1 + z0) : z1.im = -3/4 := by
  sorry

theorem part2 (x : ℝ) 
    (z : ℂ := (x^2 - 4 * x) + (x + 2) * Complex.I) 
    (z0_conj : ℂ := 3 - 4 * Complex.I) 
    (h : (z + z0_conj).re < 0 ∧ (z + z0_conj).im > 0) : 
    2 < x ∧ x < 3 :=
  by 
  sorry

end part1_part2_l758_75816


namespace symmetric_points_x_axis_l758_75849

theorem symmetric_points_x_axis (m n : ℤ) :
  (-4, m - 3) = (2 * n, -1) → (m = 2 ∧ n = -2) :=
by
  sorry

end symmetric_points_x_axis_l758_75849


namespace triangle_side_length_l758_75871

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 20)
  (h2 : (1 / 2) * b * c * (Real.sin (Real.pi / 3)) = 10 * Real.sqrt 3) : a = 7 :=
sorry

end triangle_side_length_l758_75871


namespace algebraic_inequality_l758_75872

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  |a| > 1 ∧ |b| > 1 ∧ |c| > 1 ∧ |d| > 1 ∧
  a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0 →
  (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) + (1 / (d - 1)) > 0

theorem algebraic_inequality (a b c d : ℝ) :
  problem_statement a b c d :=
by
  sorry

end algebraic_inequality_l758_75872


namespace positive_y_percentage_l758_75890

theorem positive_y_percentage (y : ℝ) (hy_pos : 0 < y) (h : 0.01 * y * y = 9) : y = 30 := by
  sorry

end positive_y_percentage_l758_75890


namespace fully_factor_expression_l758_75894

theorem fully_factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by
  -- pending proof, represented by sorry
  sorry

end fully_factor_expression_l758_75894


namespace root_equation_solution_l758_75873

theorem root_equation_solution (a : ℝ) (h : 3 * a^2 - 5 * a - 2 = 0) : 6 * a^2 - 10 * a = 4 :=
by 
  sorry

end root_equation_solution_l758_75873


namespace valid_grid_iff_divisible_by_9_l758_75825

-- Definitions for the letters used in the grid
inductive Letter
| I
| M
| O

-- Function that captures the condition that each row and column must contain exactly one-third of each letter
def valid_row_col (n : ℕ) (grid : ℕ -> ℕ -> Letter) : Prop :=
  ∀ row, (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = n / 3 ∧ count_M = n / 3 ∧ count_O = n / 3 ∧
    (∀ col, grid row col ∈ [Letter.I, Letter.M, Letter.O])) ∧
  ∀ col, (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = n / 3 ∧ count_M = n / 3 ∧ count_O = n / 3 ∧
    (∀ row, grid row col ∈ [Letter.I, Letter.M, Letter.O]))

-- Function that captures the condition that each diagonal must contain exactly one-third of each letter when the length is a multiple of 3
def valid_diagonals (n : ℕ) (grid : ℕ -> ℕ -> Letter) : Prop :=
  ∀ k, (3 ∣ k → (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = k / 3 ∧ count_M = k / 3 ∧ count_O = k / 3 ∧
    ((∀ (i j : ℕ), (i + j = k) → grid i j ∈ [Letter.I, Letter.M, Letter.O]) ∨
     (∀ (i j : ℕ), (i - j = k) → grid i j ∈ [Letter.I, Letter.M, Letter.O]))))

-- The main theorem stating that if we can fill the grid according to the rules, then n must be a multiple of 9
theorem valid_grid_iff_divisible_by_9 (n : ℕ) :
  (∃ grid : ℕ → ℕ → Letter, valid_row_col n grid ∧ valid_diagonals n grid) ↔ 9 ∣ n :=
by
  sorry

end valid_grid_iff_divisible_by_9_l758_75825


namespace coeff_x5_term_l758_75815

-- We define the binomial coefficient function C(n, k)
def C (n k : ℕ) : ℕ := Nat.choose n k

-- We define the expression in question
noncomputable def expr (x : ℝ) : ℝ := (1/x + 2*x)^7

-- The coefficient of x^5 term in the expansion
theorem coeff_x5_term : 
  let general_term (r : ℕ) (x : ℝ) := (2:ℝ)^r * C 7 r * x^(2 * r - 7)
  -- r is chosen such that the power of x is 5
  let r := 6
  -- The coefficient for r=6
  general_term r 1 = 448 := 
by sorry

end coeff_x5_term_l758_75815


namespace singer_worked_10_hours_per_day_l758_75899

noncomputable def hours_per_day_worked_on_one_song (total_songs : ℕ) (days_per_song : ℕ) (total_hours : ℕ) : ℕ :=
  total_hours / (total_songs * days_per_song)

theorem singer_worked_10_hours_per_day :
  hours_per_day_worked_on_one_song 3 10 300 = 10 := 
by
  sorry

end singer_worked_10_hours_per_day_l758_75899


namespace calculate_value_l758_75823

theorem calculate_value :
  12 * ( (1 / 3 : ℝ) + (1 / 4) + (1 / 6) )⁻¹ = 16 :=
sorry

end calculate_value_l758_75823


namespace quadratic_root_exists_l758_75898

theorem quadratic_root_exists (a b c : ℝ) : 
  ∃ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ∨ (b * x^2 + 2 * c * x + a = 0) ∨ (c * x^2 + 2 * a * x + b = 0) :=
by sorry

end quadratic_root_exists_l758_75898


namespace remainder_when_M_divided_by_32_l758_75818

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l758_75818


namespace number_solution_l758_75827

theorem number_solution (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
by
  -- The proof is omitted according to the instructions
  sorry

end number_solution_l758_75827


namespace min_expression_value_l758_75845

theorem min_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (z : ℝ) (h3 : x^2 + y^2 = z) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) = -2040200 :=
  sorry

end min_expression_value_l758_75845


namespace problem1_problem2_l758_75820

-- Problem 1: Proving the given equation under specified conditions
theorem problem1 (x y : ℝ) (h : x + y ≠ 0) : ((2 * x + 3 * y) / (x + y)) - ((x + 2 * y) / (x + y)) = 1 :=
sorry

-- Problem 2: Proving the given equation under specified conditions
theorem problem2 (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ 1) : ((a^2 - 1) / (a^2 - 4 * a + 4)) / ((a - 1) / (a - 2)) = (a + 1) / (a - 2) :=
sorry

end problem1_problem2_l758_75820


namespace gain_percent_of_50C_eq_25S_l758_75809

variable {C S : ℝ}

theorem gain_percent_of_50C_eq_25S (h : 50 * C = 25 * S) : 
  ((S - C) / C) * 100 = 100 :=
by
  sorry

end gain_percent_of_50C_eq_25S_l758_75809


namespace tan_angle_identity_l758_75895

open Real

theorem tan_angle_identity (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : sin β / cos β = (1 + cos (2 * α)) / (2 * cos α + sin (2 * α))) :
  tan (α + 2 * β + π / 4) = -1 := 
sorry

end tan_angle_identity_l758_75895


namespace additional_charge_per_segment_l758_75810

variable (initial_fee : ℝ := 2.35)
variable (total_charge : ℝ := 5.5)
variable (distance : ℝ := 3.6)
variable (segment_length : ℝ := (2/5 : ℝ))

theorem additional_charge_per_segment :
  let number_of_segments := distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  let additional_charge_per_segment := charge_for_distance / number_of_segments
  additional_charge_per_segment = 0.35 :=
by
  sorry

end additional_charge_per_segment_l758_75810


namespace total_voters_in_districts_l758_75866

theorem total_voters_in_districts :
  let D1 := 322
  let D2 := (D1 / 2) - 19
  let D3 := 2 * D1
  let D4 := D2 + 45
  let D5 := (3 * D3) - 150
  let D6 := (D1 + D4) + (1 / 5) * (D1 + D4)
  let D7 := D2 + (D5 - D2) / 2
  D1 + D2 + D3 + D4 + D5 + D6 + D7 = 4650 := 
by
  sorry

end total_voters_in_districts_l758_75866


namespace number_of_houses_l758_75838

theorem number_of_houses (total_mail_per_block : ℕ) (mail_per_house : ℕ) (h1 : total_mail_per_block = 24) (h2 : mail_per_house = 4) : total_mail_per_block / mail_per_house = 6 :=
by
  sorry

end number_of_houses_l758_75838


namespace find_p_from_conditions_l758_75852

variable (p : ℝ) (y x : ℝ)

noncomputable def parabola_eq : Prop := y^2 = 2 * p * x

noncomputable def p_positive : Prop := p > 0

noncomputable def point_on_parabola : Prop := parabola_eq p 1 (p / 4)

theorem find_p_from_conditions (hp : p_positive p) (hpp : point_on_parabola p) : p = Real.sqrt 2 :=
by 
  -- The actual proof goes here
  sorry

end find_p_from_conditions_l758_75852


namespace divisible_by_condition_a_l758_75861

theorem divisible_by_condition_a (a b c k : ℤ) 
  (h : ∃ k : ℤ, a - b * c = (10 * c + 1) * k) : 
  ∃ k : ℤ, 10 * a + b = (10 * c + 1) * k :=
by
  sorry

end divisible_by_condition_a_l758_75861


namespace exist_matrices_with_dets_l758_75804

noncomputable section

open Matrix BigOperators

variables {α : Type} [Field α] [DecidableEq α]

theorem exist_matrices_with_dets (m n : ℕ) (h₁ : 1 < m) (h₂ : 1 < n)
  (αs : Fin m → α) (β : α) :
  ∃ (A : Fin m → Matrix (Fin n) (Fin n) α), (∀ i, det (A i) = αs i) ∧ det (∑ i, A i) = β :=
sorry

end exist_matrices_with_dets_l758_75804


namespace inequality_solution_set_l758_75844

theorem inequality_solution_set (x : ℝ) : (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := sorry

end inequality_solution_set_l758_75844


namespace total_number_of_outfits_l758_75824

-- Definitions of the conditions as functions/values
def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_ties_options : Nat := 4 + 1  -- 4 ties + 1 option for no tie
def num_belts_options : Nat := 2 + 1  -- 2 belts + 1 option for no belt

-- Lean statement to formulate the proof problem
theorem total_number_of_outfits : 
  num_shirts * num_pants * num_ties_options * num_belts_options = 600 := by
  sorry

end total_number_of_outfits_l758_75824


namespace replace_question_with_division_l758_75831

theorem replace_question_with_division :
  ∃ op: (ℤ → ℤ → ℤ), (op 8 2) + 5 - (3 - 2) = 8 ∧ 
  (∀ a b, op = Int.div ∧ ((op a b) = a / b)) :=
by
  sorry

end replace_question_with_division_l758_75831


namespace purple_ring_weight_l758_75885

def orange_ring_weight : ℝ := 0.08
def white_ring_weight : ℝ := 0.42
def total_weight : ℝ := 0.83

theorem purple_ring_weight : 
  ∃ (purple_ring_weight : ℝ), purple_ring_weight = total_weight - (orange_ring_weight + white_ring_weight) := 
  by
  use 0.33
  sorry

end purple_ring_weight_l758_75885


namespace three_segments_form_triangle_l758_75842

theorem three_segments_form_triangle
    (lengths : Fin 10 → ℕ)
    (h1 : lengths 0 = 1)
    (h2 : lengths 1 = 1)
    (h3 : lengths 9 = 50) :
    ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    lengths i + lengths j > lengths k ∧ 
    lengths i + lengths k > lengths j ∧ 
    lengths j + lengths k > lengths i := 
sorry

end three_segments_form_triangle_l758_75842


namespace range_of_m_l758_75896

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 / x) + (1 / y) = 1) (h2 : x + 2 * y > m^2 + 2 * m) : -4 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l758_75896


namespace derivative_f_l758_75851

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.exp (Real.sin x))

theorem derivative_f (x : ℝ) : deriv f x = ((Real.cos x)^2 - Real.sin x) * (Real.exp (Real.sin x)) :=
by
  sorry

end derivative_f_l758_75851


namespace range_of_m_in_third_quadrant_l758_75891

theorem range_of_m_in_third_quadrant (m : ℝ) : (1 - (1/3) * m < 0) ∧ (m - 5 < 0) ↔ (3 < m ∧ m < 5) := 
by 
  intros
  sorry

end range_of_m_in_third_quadrant_l758_75891


namespace no_cubic_term_l758_75811

noncomputable def p1 (a b k : ℝ) : ℝ := -2 * a * b + (1 / 3) * k * a^2 * b + 5 * b^2
noncomputable def p2 (a b : ℝ) : ℝ := b^2 + 3 * a^2 * b - 5 * a * b + 1
noncomputable def diff (a b k : ℝ) : ℝ := p1 a b k - p2 a b
noncomputable def cubic_term_coeff (a b k : ℝ) : ℝ := (1 / 3) * k - 3

theorem no_cubic_term (a b : ℝ) : ∀ k, (cubic_term_coeff a b k = 0) → k = 9 :=
by
  intro k h
  sorry

end no_cubic_term_l758_75811


namespace inequality_solution_l758_75857

theorem inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | ax ^ 2 - (a + 1) * x + 1 < 0} =
    if a = 1 then ∅
    else if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
    else if a > 1 then {x : ℝ | 1 / a < x ∧ x < 1} 
    else ∅ := sorry

end inequality_solution_l758_75857


namespace cheesecake_needs_more_eggs_l758_75801

def chocolate_eggs_per_cake := 3
def cheesecake_eggs_per_cake := 8
def num_chocolate_cakes := 5
def num_cheesecakes := 9

theorem cheesecake_needs_more_eggs :
  cheesecake_eggs_per_cake * num_cheesecakes - chocolate_eggs_per_cake * num_chocolate_cakes = 57 :=
by
  sorry

end cheesecake_needs_more_eggs_l758_75801


namespace number_of_convex_quadrilaterals_with_parallel_sides_l758_75840

-- Define a regular 20-sided polygon
def regular_20_sided_polygon : Type := 
  { p : ℕ // 0 < p ∧ p ≤ 20 }

-- The main theorem statement
theorem number_of_convex_quadrilaterals_with_parallel_sides : 
  ∃ (n : ℕ), n = 765 :=
sorry

end number_of_convex_quadrilaterals_with_parallel_sides_l758_75840


namespace max_min_value_f_l758_75876

theorem max_min_value_f (x m : ℝ) : ∃ m : ℝ, (∀ x : ℝ, x^2 - 2*m*x + 8*m + 4 ≥ -m^2 + 8*m + 4) ∧ (∀ n : ℝ, -n^2 + 8*n + 4 ≤ 20) :=
  sorry

end max_min_value_f_l758_75876


namespace shifted_parabola_relationship_l758_75848

-- Step a) and conditions
def original_function (x : ℝ) : ℝ := -2 * x ^ 2 + 4

def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x => f (x + a)
def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := fun x => f x + b

-- Step c) encoding the proof problem
theorem shifted_parabola_relationship :
  (shift_up (shift_left original_function 2) 3 = fun x => -2 * (x + 2) ^ 2 + 7) :=
by
  sorry

end shifted_parabola_relationship_l758_75848


namespace trig_identity_l758_75897

open Real

theorem trig_identity (α : ℝ) (h : tan α = 2) :
  2 * cos (2 * α) + 3 * sin (2 * α) - sin (α) ^ 2 = 2 / 5 :=
by sorry

end trig_identity_l758_75897


namespace Rohan_earning_after_6_months_l758_75813

def farm_area : ℕ := 20
def trees_per_sqm : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval : ℕ := 3
def sale_price : ℝ := 0.50
def total_months : ℕ := 6

theorem Rohan_earning_after_6_months :
  farm_area * trees_per_sqm * coconuts_per_tree * (total_months / harvest_interval) * sale_price 
    = 240 := by
  sorry

end Rohan_earning_after_6_months_l758_75813


namespace trigonometric_identity_l758_75874

noncomputable def cos190 := Real.cos (190 * Real.pi / 180)
noncomputable def sin290 := Real.sin (290 * Real.pi / 180)
noncomputable def cos40 := Real.cos (40 * Real.pi / 180)
noncomputable def tan10 := Real.tan (10 * Real.pi / 180)

theorem trigonometric_identity :
  (cos190 * (1 + Real.sqrt 3 * tan10)) / (sin290 * Real.sqrt (1 - cos40)) = 2 * Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_l758_75874


namespace steel_strength_value_l758_75881

theorem steel_strength_value 
  (s : ℝ) 
  (condition: s = 4.6 * 10^8) : 
  s = 460000000 := 
by sorry

end steel_strength_value_l758_75881


namespace constant_subsequence_exists_l758_75869

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem constant_subsequence_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (f : ℕ → ℕ) (c : ℕ), (∀ n m, n < m → f n < f m) ∧ (∀ n, sum_of_digits (⌊a * ↑(f n) + b⌋₊) = c) :=
sorry

end constant_subsequence_exists_l758_75869


namespace george_choices_l758_75858

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- State the theorem to prove the number of ways to choose 3 out of 9 colors is 84
theorem george_choices : binomial 9 3 = 84 := by
  sorry

end george_choices_l758_75858


namespace max_truck_speed_l758_75834

theorem max_truck_speed (D : ℝ) (C : ℝ) (F : ℝ) (L : ℝ → ℝ) (T : ℝ) (x : ℝ) : 
  D = 125 ∧ C = 30 ∧ F = 1000 ∧ (∀ s, L s = 2 * s) ∧ (∃ s, D / s * C + F + L s ≤ T) → x ≤ 75 :=
by
  sorry

end max_truck_speed_l758_75834


namespace probability_of_nonzero_product_probability_of_valid_dice_values_l758_75832

def dice_values := {x : ℕ | 1 ≤ x ∧ x ≤ 6}

def valid_dice_values := {x : ℕ | 2 ≤ x ∧ x ≤ 6}

noncomputable def probability_no_one : ℚ := 625 / 1296

theorem probability_of_nonzero_product (a b c d : ℕ) 
  (ha : a ∈ dice_values) (hb : b ∈ dice_values) 
  (hc : c ∈ dice_values) (hd : d ∈ dice_values) : 
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  (a ∈ valid_dice_values ∧ b ∈ valid_dice_values ∧ 
   c ∈ valid_dice_values ∧ d ∈ valid_dice_values) :=
sorry

theorem probability_of_valid_dice_values : 
  probability_no_one = (5 / 6) ^ 4 :=
sorry

end probability_of_nonzero_product_probability_of_valid_dice_values_l758_75832


namespace simplify_and_evaluate_expression_l758_75805

theorem simplify_and_evaluate_expression (a : ℤ) (ha : a = -2) : 
  (1 + 1 / (a - 1)) / ((2 * a) / (a ^ 2 - 1)) = -1 / 2 := by
  sorry

end simplify_and_evaluate_expression_l758_75805


namespace continuous_at_1_l758_75837

theorem continuous_at_1 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x, |x - 1| < δ → |(-4 * x^2 - 6) - (-10)| < ε :=
by
  sorry

end continuous_at_1_l758_75837


namespace gumball_difference_l758_75812

theorem gumball_difference :
  let c := 17
  let l := 12
  let a := 24
  let t := 8
  let n := c + l + a + t
  let low := 14
  let high := 32
  ∃ x : ℕ, (low ≤ (n + x) / 7 ∧ (n + x) / 7 ≤ high) →
  (∃ x_min x_max, x_min ≤ x ∧ x ≤ x_max ∧ x_max - x_min = 126) :=
by
  sorry

end gumball_difference_l758_75812


namespace find_m_l758_75819

theorem find_m (x y m : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x + m * y = 5) : m = 3 := 
by
  sorry

end find_m_l758_75819


namespace range_of_a_l758_75829

open Real

theorem range_of_a (x y z a : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1)
  (heq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
by
  sorry

end range_of_a_l758_75829


namespace prime_factor_count_l758_75800

theorem prime_factor_count (n : ℕ) (H : 22 + n + 2 = 29) : n = 5 := 
  sorry

end prime_factor_count_l758_75800


namespace extremum_points_of_f_l758_75808

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x + 1)^3 * Real.exp (x + 1) - Real.exp 1
  else -((if -x < 0 then (-x + 1)^3 * Real.exp (-x + 1) - Real.exp 1 else 0))

theorem extremum_points_of_f : ∃! (a b : ℝ), 
  (∀ x < 0, f x = (x + 1)^3 * Real.exp (x + 1) - Real.exp 1) ∧ (f a = f b) ∧ a ≠ b :=
sorry

end extremum_points_of_f_l758_75808


namespace find_some_value_l758_75887

theorem find_some_value (m n : ℝ) (some_value : ℝ) (p : ℝ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + some_value) / 6 - 2 / 5)
  (h3 : p = 3)
  : some_value = -12 / 5 :=
by
  sorry

end find_some_value_l758_75887


namespace B_finishes_remaining_work_in_3_days_l758_75821

theorem B_finishes_remaining_work_in_3_days
  (A_works_in : ℕ)
  (B_works_in : ℕ)
  (work_days_together : ℕ)
  (A_leaves : A_works_in = 4)
  (B_leaves : B_works_in = 10)
  (work_days : work_days_together = 2) :
  ∃ days_remaining : ℕ, days_remaining = 3 :=
by
  sorry

end B_finishes_remaining_work_in_3_days_l758_75821


namespace smaller_angle_at_3_pm_l758_75880

-- Define the condition for minute hand position at 3:00 p.m.
def minute_hand_position_at_3_pm_deg : ℝ := 0

-- Define the condition for hour hand position at 3:00 p.m.
def hour_hand_position_at_3_pm_deg : ℝ := 90

-- Define the angle between the minute hand and hour hand
def angle_between_hands (minute_deg hour_deg : ℝ) : ℝ :=
  abs (hour_deg - minute_deg)

-- The main theorem we need to prove
theorem smaller_angle_at_3_pm :
  angle_between_hands minute_hand_position_at_3_pm_deg hour_hand_position_at_3_pm_deg = 90 :=
by
  sorry

end smaller_angle_at_3_pm_l758_75880


namespace find_angle_C_find_triangle_area_l758_75830

theorem find_angle_C (A B C : ℝ) (a b c : ℝ) 
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : B + C + A = Real.pi) :
  C = Real.pi / 12 :=
by
  sorry

theorem find_triangle_area (A B C : ℝ) (a b c : ℝ)
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : b^2 + c^2 = a - b * c + 2) 
  (h4 : B + C + A = Real.pi) 
  (h5 : a^2 = b^2 + c^2 + b * c) :
  (1/2) * a * b * Real.sin C = 1 - Real.sqrt 3 / 3 :=
by
  sorry

end find_angle_C_find_triangle_area_l758_75830


namespace largest_number_obtained_l758_75892

theorem largest_number_obtained : 
  ∃ n : ℤ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m, 10 ≤ m ∧ m ≤ 99 → (250 - 3 * m)^2 ≤ (250 - 3 * n)^2) ∧ (250 - 3 * n)^2 = 4 :=
sorry

end largest_number_obtained_l758_75892


namespace min_radius_for_area_l758_75803

theorem min_radius_for_area (A : ℝ) (hA : A = 500) : ∃ r : ℝ, r = 13 ∧ π * r^2 ≥ A :=
by
  sorry

end min_radius_for_area_l758_75803


namespace inequality_not_less_than_four_by_at_least_one_l758_75843

-- Definitions based on the conditions
def not_less_than_by_at_least (y : ℝ) (a b : ℝ) : Prop := y - a ≥ b

-- Problem statement (theorem) based on the given question and correct answer
theorem inequality_not_less_than_four_by_at_least_one (y : ℝ) :
  not_less_than_by_at_least y 4 1 → y ≥ 5 :=
by
  sorry

end inequality_not_less_than_four_by_at_least_one_l758_75843


namespace circumscribed_quadrilateral_identity_l758_75822

variables 
  (α β γ θ : ℝ)
  (h_angle_sum : α + β + γ + θ = 180)
  (OA OB OC OD AB BC CD DA : ℝ)
  (h_OA : OA = 1 / Real.sin α)
  (h_OB : OB = 1 / Real.sin β)
  (h_OC : OC = 1 / Real.sin γ)
  (h_OD : OD = 1 / Real.sin θ)
  (h_AB : AB = Real.sin (α + β) / (Real.sin α * Real.sin β))
  (h_BC : BC = Real.sin (β + γ) / (Real.sin β * Real.sin γ))
  (h_CD : CD = Real.sin (γ + θ) / (Real.sin γ * Real.sin θ))
  (h_DA : DA = Real.sin (θ + α) / (Real.sin θ * Real.sin α))

theorem circumscribed_quadrilateral_identity :
  OA * OC + OB * OD = Real.sqrt (AB * BC * CD * DA) := 
sorry

end circumscribed_quadrilateral_identity_l758_75822


namespace isosceles_triangle_perimeter_l758_75862

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4) (h2 : b = 6) : 
  ∃ p, (p = 14 ∨ p = 16) :=
by
  sorry

end isosceles_triangle_perimeter_l758_75862
