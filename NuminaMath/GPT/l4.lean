import Mathlib

namespace tan_half_angle_third_quadrant_l4_4305

theorem tan_half_angle_third_quadrant (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h : Real.sin α = -24/25) :
  Real.tan (α / 2) = -4/3 := 
by 
  sorry

end tan_half_angle_third_quadrant_l4_4305


namespace initial_cats_l4_4369

-- Define the conditions as hypotheses
variables (total_cats now : ℕ) (cats_given : ℕ)

-- State the main theorem
theorem initial_cats:
  total_cats = 31 → cats_given = 14 → (total_cats - cats_given) = 17 :=
by sorry

end initial_cats_l4_4369


namespace arithmetic_sequence_a10_l4_4858

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 7 = 9) (h2 : a 13 = -3) 
  (ha : ∀ n, a n = a1 + (n - 1) * d) :
  a 10 = 3 :=
by sorry

end arithmetic_sequence_a10_l4_4858


namespace factorial_mod_5_l4_4244

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_5 :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
   factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) % 5 = 3 :=
by
  sorry

end factorial_mod_5_l4_4244


namespace triangle_formation_l4_4494

theorem triangle_formation (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c^2 = a^2 + b^2 + a * b) : 
  a + b > c ∧ a + c > b ∧ c + (a + b) > a :=
by
  sorry

end triangle_formation_l4_4494


namespace minimize_f_sin_65_sin_40_l4_4314

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := m^2 * x^2 + (n + 1) * x + 1

theorem minimize_f_sin_65_sin_40 (m n : ℝ) (h₁ : m = Real.sin (65 * Real.pi / 180))
  (h₂ : n = Real.sin (40 * Real.pi / 180)) : 
  ∃ x, x = -1 ∧ (∀ y, f y m n ≥ f (-1) m n) :=
by
  -- Proof to be completed
  sorry

end minimize_f_sin_65_sin_40_l4_4314


namespace unique_solution_l4_4522

theorem unique_solution (x y z : ℝ) 
  (h : x^2 + 2*x + y^2 + 4*y + z^2 + 6*z = -14) : 
  x = -1 ∧ y = -2 ∧ z = -3 :=
by
  -- entering main proof section
  sorry

end unique_solution_l4_4522


namespace solve_for_y_l4_4292

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end solve_for_y_l4_4292


namespace product_mod_32_is_15_l4_4891

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l4_4891


namespace round_trip_time_l4_4194

theorem round_trip_time (current_speed : ℝ) (boat_speed_still : ℝ) (distance_upstream : ℝ) (total_time : ℝ) :
  current_speed = 4 → 
  boat_speed_still = 18 → 
  distance_upstream = 85.56 →
  total_time = 10 :=
by
  intros h_current h_boat h_distance
  sorry

end round_trip_time_l4_4194


namespace new_planet_volume_eq_l4_4586

noncomputable def volume_of_new_planet (V_earth : ℝ) (scaling_factor : ℝ) : ℝ :=
  V_earth * (scaling_factor^3)

theorem new_planet_volume_eq 
  (V_earth : ℝ)
  (scaling_factor : ℝ)
  (hV_earth : V_earth = 1.08 * 10^12)
  (h_scaling_factor : scaling_factor = 10^4) :
  volume_of_new_planet V_earth scaling_factor = 1.08 * 10^24 :=
by
  sorry

end new_planet_volume_eq_l4_4586


namespace min_colors_required_l4_4417

-- Define predicate for the conditions
def conditions (n : ℕ) (m : ℕ) (k : ℕ)(Paint : ℕ → Set ℕ) : Prop := 
  (∀ S : Finset ℕ, S.card = n → (∃ c ∈ ⋃ p ∈ S, Paint p, c ∈ S)) ∧ 
  (∀ c, ¬ (∀ i ∈ (Finset.range m).1, c ∈ Paint i))

-- The main theorem statement
theorem min_colors_required :
  ∀ (Paint : ℕ → Set ℕ), conditions 20 100 21 Paint → 
  ∃ k, conditions 20 100 k Paint ∧ k = 21 :=
sorry

end min_colors_required_l4_4417


namespace find_m_max_value_l4_4443

noncomputable def f (x : ℝ) := |x - 1|

theorem find_m (m : ℝ) :
  (∀ x, f (x + 5) ≤ 3 * m) ∧ m > 0 ∧ (∀ x, -7 ≤ x ∧ x ≤ -1 → f (x + 5) ≤ 3 * m) →
  m = 1 :=
by
  sorry

theorem max_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h2 : 2 * a ^ 2 + b ^ 2 = 3) :
  ∃ x, (∀ a b, 2 * a * Real.sqrt (1 + b ^ 2) ≤ x) ∧ x = 2 * Real.sqrt 2 :=
by
  sorry

end find_m_max_value_l4_4443


namespace cos2_a_plus_sin2_b_eq_one_l4_4449

variable {a b c : ℝ}

theorem cos2_a_plus_sin2_b_eq_one
  (h1 : Real.sin a = Real.cos b)
  (h2 : Real.sin b = Real.cos c)
  (h3 : Real.sin c = Real.cos a) :
  Real.cos a ^ 2 + Real.sin b ^ 2 = 1 := 
  sorry

end cos2_a_plus_sin2_b_eq_one_l4_4449


namespace soccer_ball_problem_l4_4045

-- Definitions of conditions
def price_eqs (x y : ℕ) : Prop :=
  x + 2 * y = 800 ∧ 3 * x + 2 * y = 1200

def total_cost_constraint (m : ℕ) : Prop :=
  200 * m + 300 * (20 - m) ≤ 5000 ∧ 1 ≤ m ∧ m ≤ 19

def store_discounts (x y : ℕ) (m : ℕ) : Prop :=
  200 * m + (3 / 5) * 300 * (20 - m) = (200 * m + (3 / 5) * 300 * (20 - m))

-- Main problem statement
theorem soccer_ball_problem :
  ∃ (x y m : ℕ), price_eqs x y ∧ total_cost_constraint m ∧ store_discounts x y m :=
sorry

end soccer_ball_problem_l4_4045


namespace smallest_degree_of_f_l4_4642

theorem smallest_degree_of_f (p : Polynomial ℂ) (hp_deg : p.degree < 1992)
  (hp0 : p.eval 0 ≠ 0) (hp1 : p.eval 1 ≠ 0) (hp_1 : p.eval (-1) ≠ 0) :
  ∃ f g : Polynomial ℂ, 
    (Polynomial.derivative^[1992] (p / (X^3 - X))) = f / g ∧ f.degree = 3984 := 
sorry

end smallest_degree_of_f_l4_4642


namespace S9_is_45_l4_4505

-- Define the required sequence and conditions
variable {a : ℕ → ℝ} -- a function that gives us the arithmetic sequence
variable {S : ℕ → ℝ} -- a function that gives us the sum of the first n terms of the sequence

-- Define the condition that a_2 + a_8 = 10
axiom a2_a8_condition : a 2 + a 8 = 10

-- Define the arithmetic property of the sequence
axiom arithmetic_property (n m : ℕ) : a (n + m) = a n + a m

-- Define the sum formula for the first n terms of an arithmetic sequence
axiom sum_formula (n : ℕ) : S n = (n / 2) * (a 1 + a n)

-- The main theorem to prove
theorem S9_is_45 : S 9 = 45 :=
by
  -- Here would go the proof, but it is omitted
  sorry

end S9_is_45_l4_4505


namespace upstream_speed_l4_4989

theorem upstream_speed (Vm Vdownstream Vupstream Vs : ℝ) 
  (h1 : Vm = 50) 
  (h2 : Vdownstream = 55) 
  (h3 : Vdownstream = Vm + Vs) 
  (h4 : Vupstream = Vm - Vs) : 
  Vupstream = 45 :=
by
  sorry

end upstream_speed_l4_4989


namespace tg_plus_ctg_l4_4148

theorem tg_plus_ctg (x : ℝ) (h : 1 / Real.cos x - 1 / Real.sin x = Real.sqrt 15) :
  Real.tan x + (1 / Real.tan x) = -3 ∨ Real.tan x + (1 / Real.tan x) = 5 :=
sorry

end tg_plus_ctg_l4_4148


namespace difference_xy_l4_4281

theorem difference_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^2 - y^2 = 27) : x - y = 3 := sorry

end difference_xy_l4_4281


namespace volume_of_given_sphere_l4_4830

noncomputable def volume_of_sphere (A d : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (Real.sqrt (d^2 + A / Real.pi))^3

theorem volume_of_given_sphere
  (hA : 2 * Real.pi = 2 * Real.pi)
  (hd : 1 = 1):
  volume_of_sphere (2 * Real.pi) 1 = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end volume_of_given_sphere_l4_4830


namespace small_drinking_glasses_count_l4_4955

theorem small_drinking_glasses_count :
  ∀ (large_jelly_beans_per_large_glass small_jelly_beans_per_small_glass total_jelly_beans : ℕ),
  (large_jelly_beans_per_large_glass = 50) →
  (small_jelly_beans_per_small_glass = large_jelly_beans_per_large_glass / 2) →
  (5 * large_jelly_beans_per_large_glass + n * small_jelly_beans_per_small_glass = total_jelly_beans) →
  (total_jelly_beans = 325) →
  n = 3 := by
  sorry

end small_drinking_glasses_count_l4_4955


namespace ship_length_in_emilys_steps_l4_4252

variable (L E S : ℝ)

-- Conditions from the problem:
variable (cond1 : 240 * E = L + 240 * S)
variable (cond2 : 60 * E = L - 60 * S)

-- Theorem to prove:
theorem ship_length_in_emilys_steps (cond1 : 240 * E = L + 240 * S) (cond2 : 60 * E = L - 60 * S) : 
  L = 96 * E := 
sorry

end ship_length_in_emilys_steps_l4_4252


namespace complex_multiplication_quadrant_l4_4448

-- Given conditions
def complex_mul (z1 z2 : ℂ) : ℂ := z1 * z2

-- Proving point is in the fourth quadrant
theorem complex_multiplication_quadrant
  (a b : ℝ) (z : ℂ)
  (h1 : z = a + b * Complex.I)
  (h2 : z = complex_mul (1 + Complex.I) (3 - Complex.I)) :
  b < 0 ∧ a > 0 :=
by
  sorry

end complex_multiplication_quadrant_l4_4448


namespace find_c_k_l4_4652

-- Definitions of the arithmetic and geometric sequences
def a (n d : ℕ) := 1 + (n - 1) * d
def b (n r : ℕ) := r ^ (n - 1)
def c (n d r : ℕ) := a n d + b n r

-- Conditions for the specific problem
theorem find_c_k (k d r : ℕ) (h1 : 1 + (k - 2) * d + r ^ (k - 2) = 150) (h2 : 1 + k * d + r ^ k = 1500) : c k d r = 314 :=
by
  sorry

end find_c_k_l4_4652


namespace min_xy_sum_is_7_l4_4071

noncomputable def min_xy_sum (x y : ℝ) : ℝ := 
x + y

theorem min_xy_sum_is_7 (x y : ℝ) (h1 : x > 1) (h2 : y > 2) (h3 : (x - 1) * (y - 2) = 4) : 
  min_xy_sum x y = 7 := by 
  sorry

end min_xy_sum_is_7_l4_4071


namespace evaluate_expression_l4_4413

theorem evaluate_expression :
  (2 ^ (-1 : ℤ) + 2 ^ (-2 : ℤ))⁻¹ = (4 / 3 : ℚ) := by
    sorry

end evaluate_expression_l4_4413


namespace blue_pill_cost_l4_4526

theorem blue_pill_cost :
  ∃ (y : ℝ), (∀ (d : ℝ), d = 45) ∧
  (∀ (b : ℝ) (r : ℝ), b = y ∧ r = y - 2) ∧
  ((21 : ℝ) * 45 = 945) ∧
  (b + r = 45) ∧
  y = 23.5 := 
by
  sorry

end blue_pill_cost_l4_4526


namespace molecular_weight_of_compound_l4_4129

theorem molecular_weight_of_compound :
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  molecular_weight = 156.22615 :=
by
  -- conditions
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  -- prove statement
  have h1 : average_atomic_weight_c = 12.05015 := by sorry
  have h2 : molecular_weight = 156.22615 := by sorry
  exact h2

end molecular_weight_of_compound_l4_4129


namespace original_days_l4_4819

-- Definitions based on the given problem conditions
def totalLaborers : ℝ := 17.5
def absentLaborers : ℝ := 7
def workingLaborers : ℝ := totalLaborers - absentLaborers
def workDaysByWorkingLaborers : ℝ := 10
def totalLaborDays : ℝ := workingLaborers * workDaysByWorkingLaborers

theorem original_days (D : ℝ) (h : totalLaborers * D = totalLaborDays) : D = 6 := sorry

end original_days_l4_4819


namespace repeating_decimal_multiplication_l4_4433

theorem repeating_decimal_multiplication :
  (0.0808080808 : ℝ) * (0.3333333333 : ℝ) = (8 / 297) := by
  sorry

end repeating_decimal_multiplication_l4_4433


namespace triangle_square_side_length_ratio_l4_4833

theorem triangle_square_side_length_ratio (t s : ℝ) (ht : 3 * t = 12) (hs : 4 * s = 12) : 
  t / s = 4 / 3 :=
by
  sorry

end triangle_square_side_length_ratio_l4_4833


namespace find_t_eq_l4_4368

variable (a V V_0 S t : ℝ)

theorem find_t_eq (h1 : V = a * t + V_0) (h2 : S = (1/3) * a * t^3 + V_0 * t) : t = (V - V_0) / a :=
sorry

end find_t_eq_l4_4368


namespace find_all_quartets_l4_4214

def is_valid_quartet (a b c d : ℕ) : Prop :=
  a + b = c * d ∧
  a * b = c + d ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d

theorem find_all_quartets :
  ∀ (a b c d : ℕ),
  is_valid_quartet a b c d ↔
  (a, b, c, d) = (1, 5, 3, 2) ∨ 
  (a, b, c, d) = (1, 5, 2, 3) ∨ 
  (a, b, c, d) = (5, 1, 3, 2) ∨
  (a, b, c, d) = (5, 1, 2, 3) ∨ 
  (a, b, c, d) = (2, 3, 1, 5) ∨ 
  (a, b, c, d) = (3, 2, 1, 5) ∨ 
  (a, b, c, d) = (2, 3, 5, 1) ∨ 
  (a, b, c, d) = (3, 2, 5, 1) := by
  sorry

end find_all_quartets_l4_4214


namespace number_of_hikers_in_the_morning_l4_4924

theorem number_of_hikers_in_the_morning (H : ℕ) :
  41 + 26 + H = 71 → H = 4 :=
by
  intros h_eq
  sorry

end number_of_hikers_in_the_morning_l4_4924


namespace deliveries_conditions_l4_4548

variables (M P D : ℕ)
variables (MeMa MeBr MeQu MeBx: ℕ)

def distribution := (MeMa = 3 * MeBr) ∧ (MeBr = MeBr) ∧ (MeQu = MeBr) ∧ (MeBx = MeBr)

theorem deliveries_conditions 
  (h1 : P = 8 * M) 
  (h2 : D = 4 * M) 
  (h3 : M + P + D = 75) 
  (h4 : MeMa + MeBr + MeQu + MeBx = M)
  (h5 : distribution MeMa MeBr MeQu MeBx) :
  M = 5 ∧ MeMa = 2 ∧ MeBr = 1 ∧ MeQu = 1 ∧ MeBx = 1 :=
    sorry 

end deliveries_conditions_l4_4548


namespace solve_system_l4_4812

theorem solve_system (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : 2 * b - 3 * a = 4) : b = 2 :=
by {
  -- Given the conditions, we need to show that b = 2
  sorry
}

end solve_system_l4_4812


namespace time_for_first_half_is_15_l4_4435

-- Definitions of the conditions in Lean
def floors := 20
def time_per_floor_next_5 := 5
def time_per_floor_final_5 := 16
def total_time := 120

-- Theorem statement
theorem time_for_first_half_is_15 :
  ∃ T, (T + (5 * time_per_floor_next_5) + (5 * time_per_floor_final_5) = total_time) ∧ (T = 15) :=
by
  sorry

end time_for_first_half_is_15_l4_4435


namespace operation_value_l4_4200

variable (a b : ℤ)

theorem operation_value (h : (21 - 1) * (9 - 1) = 160) : a = 21 :=
by
  sorry

end operation_value_l4_4200


namespace expenses_each_month_l4_4615
noncomputable def total_expenses (worked_hours1 worked_hours2 worked_hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (total_left : ℕ) : ℕ :=
  (worked_hours1 * rate1) + (worked_hours2 * rate2) + (worked_hours3 * rate3) - total_left

theorem expenses_each_month (hours1 : ℕ)
  (hours2 : ℕ)
  (hours3 : ℕ)
  (rate1 : ℕ)
  (rate2 : ℕ)
  (rate3 : ℕ)
  (left_over : ℕ) :
  hours1 = 20 → 
  rate1 = 10 →
  hours2 = 30 →
  rate2 = 20 →
  hours3 = 5 →
  rate3 = 40 →
  left_over = 500 → 
  total_expenses hours1 hours2 hours3 rate1 rate2 rate3 left_over = 500 := by
  intros h1 r1 h2 r2 h3 r3 l
  sorry

end expenses_each_month_l4_4615


namespace ellipse_properties_l4_4177

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def slopes_condition (x1 y1 x2 y2 : ℝ) (k_ab k_oa k_ob : ℝ) : Prop :=
  (k_ab^2 = k_oa * k_ob)

variables {x y : ℝ}

theorem ellipse_properties :
  (ellipse x y 2 1) ∧ -- Given ellipse equation
  (∃ (x1 y1 x2 y2 k_ab k_oa k_ob : ℝ), slopes_condition x1 y1 x2 y2 k_ab k_oa k_ob) →
  (∃ (OA OB : ℝ), OA^2 + OB^2 = 5) ∧ -- Prove sum of squares is constant
  (∃ (m : ℝ), (m = 1 → ∃ (line_eq : ℝ → ℝ), ∀ x, line_eq x = (1 / 2) * x + m)) -- Maximum area of triangle AOB

:= sorry

end ellipse_properties_l4_4177


namespace fraction_subtraction_l4_4843

theorem fraction_subtraction :
  (9 / 19) - (5 / 57) - (2 / 38) = 1 / 3 := by
sorry

end fraction_subtraction_l4_4843


namespace adam_earning_per_lawn_l4_4331

theorem adam_earning_per_lawn 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 12) 
  (h2 : forgotten_lawns = 8) 
  (h3 : total_earnings = 36) : 
  total_earnings / (total_lawns - forgotten_lawns) = 9 :=
by
  sorry

end adam_earning_per_lawn_l4_4331


namespace inclination_angle_range_l4_4055

theorem inclination_angle_range (k : ℝ) (h : |k| ≤ 1) :
    ∃ α : ℝ, (k = Real.tan α) ∧ (0 ≤ α ∧ α ≤ Real.pi / 4 ∨ 3 * Real.pi / 4 ≤ α ∧ α < Real.pi) :=
by
  sorry

end inclination_angle_range_l4_4055


namespace lollipops_left_l4_4201

def problem_conditions : Prop :=
  ∃ (lollipops_bought lollipops_eaten lollipops_left : ℕ),
    lollipops_bought = 12 ∧
    lollipops_eaten = 5 ∧
    lollipops_left = lollipops_bought - lollipops_eaten

theorem lollipops_left (lollipops_bought lollipops_eaten lollipops_left : ℕ) 
  (hb : lollipops_bought = 12) (he : lollipops_eaten = 5) (hl : lollipops_left = lollipops_bought - lollipops_eaten) : 
  lollipops_left = 7 := 
by 
  sorry

end lollipops_left_l4_4201


namespace sum_of_numbers_l4_4075

theorem sum_of_numbers (a b c : ℕ) (h_order: a ≤ b ∧ b ≤ c) (h_median: b = 10) 
    (h_mean_least: (a + b + c) / 3 = a + 15) (h_mean_greatest: (a + b + c) / 3 = c - 20) :
    a + b + c = 45 :=
  by
  sorry

end sum_of_numbers_l4_4075


namespace four_digit_number_exists_l4_4708

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), A = B / 3 ∧ C = A + B ∧ D = 3 * B ∧
  A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1349) :=
by
  sorry

end four_digit_number_exists_l4_4708


namespace initial_markers_l4_4455

variable (markers_given : ℕ) (total_markers : ℕ)

theorem initial_markers (h_given : markers_given = 109) (h_total : total_markers = 326) :
  total_markers - markers_given = 217 :=
by
  sorry

end initial_markers_l4_4455


namespace parabola_coefficients_l4_4739

theorem parabola_coefficients :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, (a * (x - 3)^2 + 2 = 0 → (x = 1) ∧ (a * (1 - 3)^2 + 2 = 0))
    ∧ (a = -1/2 ∧ b = 3 ∧ c = -5/2)) 
    ∧ (∀ x : ℝ, a * x^2 + b * x + c = - 1 / 2 * x^2 + 3 * x - 5 / 2) :=
sorry

end parabola_coefficients_l4_4739


namespace largest_prime_divisor_36_squared_plus_81_squared_l4_4390

-- Definitions of the key components in the problem
def a := 36
def b := 81
def expr := a^2 + b^2
def largest_prime_divisor (n : ℕ) : ℕ := sorry -- Assume this function can compute the largest prime divisor

-- Theorem stating the problem
theorem largest_prime_divisor_36_squared_plus_81_squared : largest_prime_divisor (36^2 + 81^2) = 53 := 
  sorry

end largest_prime_divisor_36_squared_plus_81_squared_l4_4390


namespace necessary_but_not_sufficient_condition_l4_4585

def p (x : ℝ) : Prop := |4 * x - 3| ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≤ 0

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, p x → q x a) ∧ ¬ (∀ x : ℝ, q x a → p x) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end necessary_but_not_sufficient_condition_l4_4585


namespace find_d_l4_4653

theorem find_d (y d : ℝ) (hy : y > 0) (h : (8 * y) / 20 + (3 * y) / d = 0.7 * y) : d = 10 :=
by
  sorry

end find_d_l4_4653


namespace find_a_for_parabola_l4_4853

theorem find_a_for_parabola (a : ℝ) :
  (∃ y : ℝ, y = a * (-1 / 2)^2) → a = 1 / 2 :=
by
  sorry

end find_a_for_parabola_l4_4853


namespace smallest_common_students_l4_4859

theorem smallest_common_students 
    (z : ℕ) (k : ℕ) (j : ℕ) 
    (hz : z = k ∧ k = j) 
    (hz_ratio : ∃ x : ℕ, z = 3 * x ∧ k = 2 * x ∧ j = 5 * x)
    (hz_group : ∃ y : ℕ, z = 14 * y) 
    (hk_group : ∃ w : ℕ, k = 10 * w) 
    (hj_group : ∃ v : ℕ, j = 15 * v) : 
    z = 630 ∧ k = 420 ∧ j = 1050 :=
    sorry

end smallest_common_students_l4_4859


namespace worksheets_already_graded_eq_5_l4_4082

def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 9
def remaining_problems : ℕ := 16

def total_problems := total_worksheets * problems_per_worksheet
def graded_problems := total_problems - remaining_problems
def graded_worksheets := graded_problems / problems_per_worksheet

theorem worksheets_already_graded_eq_5 :
  graded_worksheets = 5 :=
by 
  sorry

end worksheets_already_graded_eq_5_l4_4082


namespace ethan_arianna_apart_l4_4209

def ethan_distance := 1000 -- the distance Ethan ran
def arianna_distance := 184 -- the distance Arianna ran

theorem ethan_arianna_apart : ethan_distance - arianna_distance = 816 := by
  sorry

end ethan_arianna_apart_l4_4209


namespace f_neg_one_value_l4_4866

theorem f_neg_one_value (f : ℝ → ℝ) (b : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x → f x = 2^x + 2 * x + b) :
  f (-1) = -3 := by
sorry

end f_neg_one_value_l4_4866


namespace neither_sufficient_nor_necessary_l4_4278

theorem neither_sufficient_nor_necessary (x : ℝ) : 
  ¬(-1 < x ∧ x < 2 → |x - 2| < 1) ∧ ¬(|x - 2| < 1 → -1 < x ∧ x < 2) :=
by
  sorry

end neither_sufficient_nor_necessary_l4_4278


namespace granger_total_amount_l4_4688

-- Define the constants for the problem
def cost_spam := 3
def cost_peanut_butter := 5
def cost_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Define the total cost calculation
def total_cost := (quantity_spam * cost_spam) + (quantity_peanut_butter * cost_peanut_butter) + (quantity_bread * cost_bread)

-- The theorem we need to prove
theorem granger_total_amount : total_cost = 59 := by
  sorry

end granger_total_amount_l4_4688


namespace kanul_total_amount_l4_4857

def kanul_spent : ℝ := 3000 + 1000
def kanul_spent_percentage (T : ℝ) : ℝ := 0.30 * T

theorem kanul_total_amount (T : ℝ) (h : T = kanul_spent + kanul_spent_percentage T) :
  T = 5714.29 := sorry

end kanul_total_amount_l4_4857


namespace intersection_of_M_and_N_l4_4904

def set_M : Set ℝ := {x | 0 ≤ x ∧ x < 2}
def set_N : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def intersection_M_N : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := 
by sorry

end intersection_of_M_and_N_l4_4904


namespace outliers_in_data_set_l4_4040

-- Define the data set
def dataSet : List ℕ := [6, 19, 33, 33, 39, 41, 41, 43, 51, 57]

-- Define the given quartiles
def Q1 : ℕ := 33
def Q3 : ℕ := 43

-- Define the interquartile range
def IQR : ℕ := Q3 - Q1

-- Define the outlier thresholds
def lowerOutlierThreshold : ℕ := Q1 - 3 / 2 * IQR
def upperOutlierThreshold : ℕ := Q3 + 3 / 2 * IQR

-- Define what it means to be an outlier
def isOutlier (x : ℕ) : Bool :=
  x < lowerOutlierThreshold ∨ x > upperOutlierThreshold

-- Count the number of outliers in the data set
def countOutliers (data : List ℕ) : ℕ :=
  (data.filter isOutlier).length

theorem outliers_in_data_set :
  countOutliers dataSet = 1 :=
by
  sorry

end outliers_in_data_set_l4_4040


namespace answer_is_correct_l4_4986

-- We define the prime checking function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

-- We define the set of candidates satisfying initial prime condition
def candidates : Set ℕ := {A | is_prime A ∧ A < 100 
                                   ∧ is_prime (A + 10) 
                                   ∧ is_prime (A - 20)
                                   ∧ is_prime (A + 30) 
                                   ∧ is_prime (A + 60) 
                                   ∧ is_prime (A + 70)}

-- The explicit set of valid answers
def valid_answers : Set ℕ := {37, 43, 79}

-- The statement that we need to prove
theorem answer_is_correct : candidates = valid_answers := 
sorry

end answer_is_correct_l4_4986


namespace translated_parabola_eq_new_equation_l4_4052

-- Definitions following directly from the condition
def original_parabola (x : ℝ) : ℝ := 2 * x^2
def new_vertex : (ℝ × ℝ) := (-2, -2)
def new_parabola (x : ℝ) : ℝ := 2 * (x + 2)^2 - 2

-- Statement to prove the equivalency of the translated parabola equation
theorem translated_parabola_eq_new_equation :
  (∀ (x : ℝ), (original_parabola x = new_parabola (x - 2))) :=
by
  sorry

end translated_parabola_eq_new_equation_l4_4052


namespace statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l4_4407

-- Statement A: Proving the solution set of the inequality
theorem statement_A_solution_set (x : ℝ) : 
  (x + 2) / (2 * x + 1) > 1 ↔ (-1 / 2) < x ∧ x < 1 :=
sorry

-- Statement B: "ab > 1" is not a sufficient condition for "a > 1, b > 1"
theorem statement_B_insufficient_condition (a b : ℝ) :
  (a * b > 1) → ¬(a > 1 ∧ b > 1) :=
sorry

-- Statement C: The negation of p: ∀ x ∈ ℝ, x² > 0 is true
theorem statement_C_negation (x0 : ℝ) : 
  (∀ x : ℝ, x^2 > 0) → ¬ (∃ x0 : ℝ, x0^2 ≤ 0) :=
sorry

-- Statement D: "a < 2" is not a necessary condition for "a < 6"
theorem statement_D_not_necessary_condition (a : ℝ) :
  (a < 2) → ¬(a < 6) :=
sorry

end statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l4_4407


namespace parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l4_4797

theorem parabola_right_shift_unique_intersection (p : ℚ) :
  let y := 2 * (x - p)^2;
  (x * x - 4) = 0 →
  p = 31 / 8 := sorry

theorem parabola_down_shift_unique_intersection (q : ℚ) :
  let y := 2 * x^2 - q;
  (x * x - 4) = 0 →
  q = 31 / 8 := sorry

end parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l4_4797


namespace abs_diff_ps_pds_eq_31_100_l4_4512

-- Defining the conditions
def num_red : ℕ := 500
def num_black : ℕ := 700
def num_blue : ℕ := 800
def total_marbles : ℕ := num_red + num_black + num_blue
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculating P_s and P_d
def ways_same_color : ℕ := choose num_red 2 + choose num_black 2 + choose num_blue 2
def total_ways : ℕ := choose total_marbles 2
def P_s : ℚ := ways_same_color / total_ways

def ways_different_color : ℕ := num_red * num_black + num_red * num_blue + num_black * num_blue
def P_d : ℚ := ways_different_color / total_ways

-- Proving the statement
theorem abs_diff_ps_pds_eq_31_100 : |P_s - P_d| = (31 : ℚ) / 100 := by
  sorry

end abs_diff_ps_pds_eq_31_100_l4_4512


namespace remainder_when_sum_div_by_3_l4_4002

theorem remainder_when_sum_div_by_3 
  (m n p q : ℕ)
  (a : ℕ := 6 * m + 4)
  (b : ℕ := 6 * n + 4)
  (c : ℕ := 6 * p + 4)
  (d : ℕ := 6 * q + 4)
  : (a + b + c + d) % 3 = 1 :=
by
  sorry

end remainder_when_sum_div_by_3_l4_4002


namespace arithmetic_sequence_50th_term_l4_4563

theorem arithmetic_sequence_50th_term :
  let a1 := 3
  let d := 2
  let n := 50
  let a_n := a1 + (n - 1) * d
  a_n = 101 :=
by
  sorry

end arithmetic_sequence_50th_term_l4_4563


namespace claudia_fills_4ounce_glasses_l4_4584

theorem claudia_fills_4ounce_glasses :
  ∀ (total_water : ℕ) (five_ounce_glasses : ℕ) (eight_ounce_glasses : ℕ) 
    (four_ounce_glass_volume : ℕ),
  total_water = 122 →
  five_ounce_glasses = 6 →
  eight_ounce_glasses = 4 →
  four_ounce_glass_volume = 4 →
  (total_water - (five_ounce_glasses * 5 + eight_ounce_glasses * 8)) / four_ounce_glass_volume = 15 :=
by
  intros _ _ _ _ _ _ _ _ 
  sorry

end claudia_fills_4ounce_glasses_l4_4584


namespace line_BC_eq_l4_4580

def altitude1 (x y : ℝ) : Prop := x + y = 0
def altitude2 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def point_A : ℝ × ℝ := (1, 2)

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem line_BC_eq (x y : ℝ) :
  (∃ b c : ℝ × ℝ, altitude1 b.1 b.2 ∧ altitude2 c.1 c.2 ∧
                   line_eq 2 3 7 b.1 b.2 ∧ line_eq 2 3 7 c.1 c.2 ∧
                   b ≠ c) → 
    line_eq 2 3 7 x y :=
by sorry

end line_BC_eq_l4_4580


namespace tennis_tournament_rounds_needed_l4_4446

theorem tennis_tournament_rounds_needed (n : ℕ) (total_participants : ℕ) (win_points loss_points : ℕ) (get_point_no_pair : ℕ) (elimination_loss : ℕ) :
  total_participants = 1152 →
  win_points = 1 →
  loss_points = 0 →
  get_point_no_pair = 1 →
  elimination_loss = 2 →
  n = 14 :=
by
  sorry

end tennis_tournament_rounds_needed_l4_4446


namespace evaluate_expression_l4_4601

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 5) :
  (1 / (y : ℚ) / (1 / (x : ℚ)) + 2) = 14 / 5 :=
by
  rw [hx, hy]
  simp
  sorry

end evaluate_expression_l4_4601


namespace enlarged_sticker_height_l4_4236

theorem enlarged_sticker_height (original_width original_height new_width : ℕ) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 12) : (new_width / original_width) * original_height = 8 := 
by 
  -- Prove the height of the enlarged sticker is 8 inches
  sorry

end enlarged_sticker_height_l4_4236


namespace range_of_a_l4_4645

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4*x + a

theorem range_of_a 
  (f : ℝ → ℝ → ℝ)
  (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x a ≥ 0) : 
  3 ≤ a :=
sorry

end range_of_a_l4_4645


namespace range_of_magnitudes_l4_4471

theorem range_of_magnitudes (AB AC BC : ℝ) (hAB : AB = 8) (hAC : AC = 5) :
  3 ≤ BC ∧ BC ≤ 13 :=
by
  sorry

end range_of_magnitudes_l4_4471


namespace f_2015_2016_l4_4994

noncomputable def f : ℤ → ℤ := sorry

theorem f_2015_2016 (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (x + 2) = -f x) (h3 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end f_2015_2016_l4_4994


namespace tony_walking_speed_l4_4411

-- Define the conditions as hypotheses
def walking_speed_on_weekend (W : ℝ) : Prop := 
  let store_distance := 4 
  let run_speed := 10
  let day1_time := store_distance / W
  let day2_time := store_distance / run_speed
  let day3_time := store_distance / run_speed
  let avg_time := (day1_time + day2_time + day3_time) / 3
  avg_time = 56 / 60

-- State the theorem
theorem tony_walking_speed : ∃ W : ℝ, walking_speed_on_weekend W ∧ W = 2 := 
sorry

end tony_walking_speed_l4_4411


namespace proportional_increase_l4_4425

theorem proportional_increase (x y : ℝ) (h : 3 * x - 2 * y = 7) : y = (3 / 2) * x - 7 / 2 :=
by
  sorry

end proportional_increase_l4_4425


namespace closest_ratio_adults_children_l4_4298

theorem closest_ratio_adults_children 
  (a c : ℕ) 
  (H1 : 30 * a + 15 * c = 2550) 
  (H2 : a > 0) 
  (H3 : c > 0) : 
  (a = 57 ∧ c = 56) ∨ (a = 56 ∧ c = 58) :=
by
  sorry

end closest_ratio_adults_children_l4_4298


namespace no_two_perfect_cubes_l4_4366

theorem no_two_perfect_cubes (n : ℕ) : ¬ (∃ a b : ℕ, a^3 = n + 2 ∧ b^3 = n^2 + n + 1) := by
  sorry

end no_two_perfect_cubes_l4_4366


namespace termites_count_l4_4036

theorem termites_count (total_workers monkeys : ℕ) (h1 : total_workers = 861) (h2 : monkeys = 239) : total_workers - monkeys = 622 :=
by
  -- The proof steps will go here
  sorry

end termites_count_l4_4036


namespace inequality_for_positive_reals_l4_4636

theorem inequality_for_positive_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
    ≥ (Real.sqrt (3 / 2) * Real.sqrt (x + y + z)) := 
sorry

end inequality_for_positive_reals_l4_4636


namespace more_sightings_than_triple_cape_may_l4_4820

def daytona_shark_sightings := 26
def cape_may_shark_sightings := 7

theorem more_sightings_than_triple_cape_may :
  daytona_shark_sightings - 3 * cape_may_shark_sightings = 5 :=
by
  sorry

end more_sightings_than_triple_cape_may_l4_4820


namespace quadratic_roots_p_eq_l4_4087

theorem quadratic_roots_p_eq (b c p q r s : ℝ)
  (h1 : r + s = -b)
  (h2 : r * s = c)
  (h3 : r^2 + s^2 = -p)
  (h4 : r^2 * s^2 = q):
  p = 2 * c - b^2 :=
by sorry

end quadratic_roots_p_eq_l4_4087


namespace sqrt_fraction_eq_half_l4_4608

-- Define the problem statement in a Lean 4 theorem:
theorem sqrt_fraction_eq_half : Real.sqrt ((25 / 36 : ℚ) - (4 / 9 : ℚ)) = 1 / 2 := by
  sorry

end sqrt_fraction_eq_half_l4_4608


namespace hcf_lcm_fraction_l4_4524

theorem hcf_lcm_fraction (m n : ℕ) (HCF : Nat.gcd m n = 6) (LCM : Nat.lcm m n = 210) (sum_mn : m + n = 72) : 
  (1 / m : ℚ) + (1 / n : ℚ) = 2 / 35 :=
by
  sorry

end hcf_lcm_fraction_l4_4524


namespace shadow_problem_l4_4926

-- Define the conditions
def cube_edge_length : ℝ := 2
def shadow_area_outside : ℝ := 147
def total_shadow_area : ℝ := shadow_area_outside + cube_edge_length^2

-- The main statement to prove
theorem shadow_problem :
  let x := 4 / (Real.sqrt total_shadow_area - cube_edge_length)
  (⌊1000 * x⌋ : ℤ) = 481 :=
by
  let x := 4 / (Real.sqrt total_shadow_area - cube_edge_length)
  have h : (⌊1000 * x⌋ : ℤ) = 481 := sorry
  exact h

end shadow_problem_l4_4926


namespace parallelogram_base_length_l4_4354

theorem parallelogram_base_length (A H : ℝ) (base : ℝ) 
    (hA : A = 72) (hH : H = 6) (h_area : A = base * H) : base = 12 := 
by 
  sorry

end parallelogram_base_length_l4_4354


namespace evaluate_f_at_2_l4_4531

def f (x : ℝ) : ℝ := x^2 - x

theorem evaluate_f_at_2 : f 2 = 2 := by
  sorry

end evaluate_f_at_2_l4_4531


namespace joan_remaining_balloons_l4_4480

def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2
def remaining_balloons : ℕ := initial_balloons - lost_balloons

theorem joan_remaining_balloons : remaining_balloons = 7 := by
  sorry

end joan_remaining_balloons_l4_4480


namespace reflex_angle_at_T_l4_4460

-- Assume points P, Q, R, and S are aligned
def aligned (P Q R S : ℝ × ℝ) : Prop :=
  ∃ a b, ∀ x, x = 0 * a + b + (P.1, Q.1, R.1, S.1)

-- Angles given in the problem
def PQT_angle : ℝ := 150
def RTS_angle : ℝ := 70

-- definition of the reflex angle at T
def reflex_angle (angle : ℝ) : ℝ := 360 - angle

theorem reflex_angle_at_T (P Q R S T : ℝ × ℝ) :
  aligned P Q R S → PQT_angle = 150 → RTS_angle = 70 →
  reflex_angle 40 = 320 :=
by
  sorry

end reflex_angle_at_T_l4_4460


namespace number_of_students_scoring_above_90_l4_4362

theorem number_of_students_scoring_above_90
  (total_students : ℕ)
  (mean : ℝ)
  (variance : ℝ)
  (students_scoring_at_least_60 : ℕ)
  (h1 : total_students = 1200)
  (h2 : mean = 75)
  (h3 : ∃ (σ : ℝ), variance = σ^2)
  (h4 : students_scoring_at_least_60 = 960)
  : ∃ n, n = total_students - students_scoring_at_least_60 ∧ n = 240 :=
by {
  sorry
}

end number_of_students_scoring_above_90_l4_4362


namespace train_speed_l4_4680

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor : ℝ)
  (h1 : length = 500) (h2 : time = 5) (h3 : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 360 :=
by
  sorry

end train_speed_l4_4680


namespace intersecting_lines_at_point_find_b_plus_m_l4_4520

theorem intersecting_lines_at_point_find_b_plus_m :
  ∀ (m b : ℝ),
  (12 = m * 4 + 2) →
  (12 = -2 * 4 + b) →
  (b + m = 22.5) :=
by
  intros m b h1 h2
  sorry

end intersecting_lines_at_point_find_b_plus_m_l4_4520


namespace solve_system_l4_4783

theorem solve_system (x y : ℚ) (h1 : 6 * x = -9 - 3 * y) (h2 : 4 * x = 5 * y - 34) : x = 1/2 ∧ y = -4 :=
by
  sorry

end solve_system_l4_4783


namespace g_800_eq_768_l4_4610

noncomputable def g : ℕ → ℕ := sorry

axiom g_condition1 (n : ℕ) : g (g n) = 2 * n
axiom g_condition2 (n : ℕ) : g (4 * n + 3) = 4 * n + 1

theorem g_800_eq_768 : g 800 = 768 := by
  sorry

end g_800_eq_768_l4_4610


namespace candy_selection_l4_4012

theorem candy_selection (m n : ℕ) (h1 : Nat.gcd m n = 1) (h2 : m = 1) (h3 : n = 5) :
  m + n = 6 := by
  sorry

end candy_selection_l4_4012


namespace no_prime_divisible_by_42_l4_4774

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → ¬ (42 ∣ p) :=
by sorry

end no_prime_divisible_by_42_l4_4774


namespace no_solution_range_has_solution_range_l4_4361

open Real

theorem no_solution_range (a : ℝ) : (∀ x, ¬ (|x - 4| + |3 - x| < a)) ↔ a ≤ 1 := 
sorry

theorem has_solution_range (a : ℝ) : (∃ x, |x - 4| + |3 - x| < a) ↔ 1 < a :=
sorry

end no_solution_range_has_solution_range_l4_4361


namespace fraction_dropped_l4_4951

theorem fraction_dropped (f : ℝ) 
  (h1 : 0 ≤ f ∧ f ≤ 1) 
  (initial_passengers : ℝ) 
  (final_passenger_count : ℝ)
  (first_pickup : ℝ)
  (second_pickup : ℝ) 
  (first_drop_factor : ℝ)
  (second_drop_factor : ℕ):
  initial_passengers = 270 →
  final_passenger_count = 242 →
  first_pickup = 280 →
  second_pickup = 12 →
  first_drop_factor = f →
  second_drop_factor = 2 →
  ((initial_passengers - initial_passengers * first_drop_factor) + first_pickup) / second_drop_factor + second_pickup = final_passenger_count →
  f = 1 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end fraction_dropped_l4_4951


namespace calories_per_slice_l4_4093

theorem calories_per_slice
  (total_calories : ℕ)
  (portion_eaten : ℕ)
  (percentage_eaten : ℝ)
  (slices_in_cheesecake : ℕ)
  (calories_in_slice : ℕ) :
  total_calories = 2800 →
  percentage_eaten = 0.25 →
  portion_eaten = 2 →
  portion_eaten = percentage_eaten * slices_in_cheesecake →
  calories_in_slice = total_calories / slices_in_cheesecake →
  calories_in_slice = 350 :=
by
  intros
  sorry

end calories_per_slice_l4_4093


namespace exists_equilateral_triangle_same_color_l4_4704

-- Define a type for colors
inductive Color
| red : Color
| blue : Color

-- Define our statement
-- Given each point in the plane is colored either red or blue,
-- there exists an equilateral triangle with vertices of the same color.
theorem exists_equilateral_triangle_same_color (coloring : ℝ × ℝ → Color) : 
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ 
    dist p₁ p₂ = dist p₂ p₃ ∧ dist p₂ p₃ = dist p₃ p₁ ∧ 
    (coloring p₁ = coloring p₂ ∧ coloring p₂ = coloring p₃) :=
by
  sorry

end exists_equilateral_triangle_same_color_l4_4704


namespace melanie_missed_games_l4_4204

-- Define the total number of games and the number of games attended by Melanie
def total_games : ℕ := 7
def games_attended : ℕ := 3

-- Define the number of games missed as total games minus games attended
def games_missed : ℕ := total_games - games_attended

-- Theorem stating the number of games missed by Melanie
theorem melanie_missed_games : games_missed = 4 := by
  -- The proof is omitted
  sorry

end melanie_missed_games_l4_4204


namespace stable_state_exists_l4_4327

-- Definition of the problem
theorem stable_state_exists 
(N : ℕ) (N_ge_3 : N ≥ 3) (letters : Fin N → Fin 3) 
(perform_operation : ∀ (letters : Fin N → Fin 3), Fin N → Fin 3)
(stable : ∀ (letters : Fin N → Fin 3), Prop)
(initial_state : Fin N → Fin 3):
  ∃ (state : Fin N → Fin 3), (∀ i, perform_operation state i = state i) ∧ stable state :=
sorry

end stable_state_exists_l4_4327


namespace scheme_choice_l4_4457

variable (x y₁ y₂ : ℕ)

def cost_scheme_1 (x : ℕ) : ℕ := 12 * x + 40

def cost_scheme_2 (x : ℕ) : ℕ := 16 * x

theorem scheme_choice :
  ∀ (x : ℕ), 5 ≤ x → x ≤ 20 →
  (if x < 10 then cost_scheme_2 x < cost_scheme_1 x else
   if x = 10 then cost_scheme_2 x = cost_scheme_1 x else
   cost_scheme_1 x < cost_scheme_2 x) :=
by
  sorry

end scheme_choice_l4_4457


namespace cos_300_eq_half_l4_4551

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 := 
by sorry

end cos_300_eq_half_l4_4551


namespace amber_worked_hours_l4_4535

-- Define the variables and conditions
variables (A : ℝ) (Armand_hours : ℝ) (Ella_hours : ℝ)
variables (h1 : Armand_hours = A / 3) (h2 : Ella_hours = 2 * A)
variables (h3 : A + Armand_hours + Ella_hours = 40)

-- Prove the statement
theorem amber_worked_hours : A = 12 :=
by
  sorry

end amber_worked_hours_l4_4535


namespace probability_of_losing_weight_l4_4323

theorem probability_of_losing_weight (total_volunteers lost_weight : ℕ) (h_total : total_volunteers = 1000) (h_lost : lost_weight = 241) : 
    (lost_weight : ℚ) / total_volunteers = 0.24 := by
  sorry

end probability_of_losing_weight_l4_4323


namespace number_of_children_at_reunion_l4_4675

theorem number_of_children_at_reunion (A C : ℕ) 
    (h1 : 3 * A = C)
    (h2 : 2 * A / 3 = 10) : 
  C = 45 :=
by
  sorry

end number_of_children_at_reunion_l4_4675


namespace ratio_of_areas_l4_4637

noncomputable def side_length_S : ℝ := sorry
noncomputable def side_length_longer_R : ℝ := 1.2 * side_length_S
noncomputable def side_length_shorter_R : ℝ := 0.8 * side_length_S
noncomputable def area_S : ℝ := side_length_S ^ 2
noncomputable def area_R : ℝ := side_length_longer_R * side_length_shorter_R

theorem ratio_of_areas :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l4_4637


namespace total_ladybugs_l4_4110

theorem total_ladybugs (leaves : Nat) (ladybugs_per_leaf : Nat) (total_ladybugs : Nat) : 
  leaves = 84 → 
  ladybugs_per_leaf = 139 → 
  total_ladybugs = leaves * ladybugs_per_leaf → 
  total_ladybugs = 11676 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end total_ladybugs_l4_4110


namespace monomial_sum_l4_4381

theorem monomial_sum (m n : ℤ) (h1 : n - 1 = 4) (h2 : m - 1 = 2) : m - 2 * n = -7 := by
  sorry

end monomial_sum_l4_4381


namespace speed_conversion_l4_4571

noncomputable def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_conversion (h : 1 = 3.6) : mps_to_kmph 12.7788 = 45.96 :=
  by
    sorry

end speed_conversion_l4_4571


namespace sequence_a4_value_l4_4561

theorem sequence_a4_value :
  ∀ {a : ℕ → ℚ}, (a 1 = 3) → ((∀ n, a (n + 1) = 3 * a n / (a n + 3))) → (a 4 = 3 / 4) :=
by
  intros a h1 hRec
  sorry

end sequence_a4_value_l4_4561


namespace average_water_drunk_l4_4125

theorem average_water_drunk (d1 d2 d3 : ℕ) (h1 : d1 = 215) (h2 : d2 = d1 + 76) (h3 : d3 = d2 - 53) :
  (d1 + d2 + d3) / 3 = 248 :=
by
  -- placeholder for actual proof
  sorry

end average_water_drunk_l4_4125


namespace parallelogram_not_symmetrical_l4_4518

def is_symmetrical (shape : String) : Prop :=
  shape = "Circle" ∨ shape = "Rectangle" ∨ shape = "Isosceles Trapezoid"

theorem parallelogram_not_symmetrical : ¬ is_symmetrical "Parallelogram" :=
by
  sorry

end parallelogram_not_symmetrical_l4_4518


namespace problem_1_problem_2_problem_3_l4_4547

def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def notU (s : Set ℝ) : Set ℝ := { x | x ∉ s ∧ x ∈ U }

theorem problem_1 : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
sorry

theorem problem_2 : notU A ∪ B = { x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7) } :=
sorry

theorem problem_3 : A ∩ notU B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end problem_1_problem_2_problem_3_l4_4547


namespace minimum_apples_l4_4655

theorem minimum_apples (n : ℕ) (A : ℕ) (h1 : A = 25 * n + 24) (h2 : A > 300) : A = 324 :=
sorry

end minimum_apples_l4_4655


namespace last_digit_of_1_div_3_pow_9_is_7_l4_4371

noncomputable def decimal_expansion_last_digit (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem last_digit_of_1_div_3_pow_9_is_7 :
  decimal_expansion_last_digit 1 (3^9) = 7 :=
by
  sorry

end last_digit_of_1_div_3_pow_9_is_7_l4_4371


namespace Andrew_spent_1395_dollars_l4_4077

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

end Andrew_spent_1395_dollars_l4_4077


namespace binom_n_plus_one_n_l4_4355

theorem binom_n_plus_one_n (n : ℕ) (h : 0 < n) : Nat.choose (n + 1) n = n + 1 := 
sorry

end binom_n_plus_one_n_l4_4355


namespace multiply_decimals_l4_4138

theorem multiply_decimals :
  0.25 * 0.08 = 0.02 :=
sorry

end multiply_decimals_l4_4138


namespace increasing_sum_sequence_l4_4063

theorem increasing_sum_sequence (a : ℕ → ℝ) (Sn : ℕ → ℝ)
  (ha : ∀ n : ℕ, 0 < a (n + 1))
  (hSn : ∀ n : ℕ, Sn (n + 1) = Sn n + a (n + 1)) :
  (∀ n : ℕ, Sn (n + 1) > Sn n)
  ∧ ¬ (∀ n : ℕ, Sn (n + 1) > Sn n → 0 < a (n + 1)) :=
sorry

end increasing_sum_sequence_l4_4063


namespace field_length_l4_4246

-- Definitions of the conditions
def pond_area : ℝ := 25  -- area of the square pond
def width_to_length_ratio (w l : ℝ) : Prop := l = 2 * w  -- length is double the width
def pond_to_field_ratio (pond_area field_area : ℝ) : Prop := pond_area = (1/8) * field_area  -- pond area is 1/8 of field area

-- Statement to prove
theorem field_length (w l : ℝ) (h1 : width_to_length_ratio w l) (h2 : pond_to_field_ratio pond_area (l * w)) : l = 20 :=
by sorry

end field_length_l4_4246


namespace correct_calculation_l4_4359

theorem correct_calculation (a b : ℝ) :
  ((ab)^3 = a^3 * b^3) ∧ 
  ¬(a + 2 * a^2 = 3 * a^3) ∧ 
  ¬(a * (-a)^4 = -a^5) ∧ 
  ¬((a^3)^2 = a^5) :=
  by
  sorry

end correct_calculation_l4_4359


namespace remainder_when_divided_by_x_minus_2_l4_4221

-- Define the polynomial
def f (x : ℕ) : ℕ := x^3 - x^2 + 4 * x - 1

-- Statement of the problem: Prove f(2) = 11 using the Remainder Theorem
theorem remainder_when_divided_by_x_minus_2 : f 2 = 11 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l4_4221


namespace ages_of_siblings_l4_4650

-- Define the variables representing the ages of the siblings
variables (R D S E : ℕ)

-- Define the conditions
def conditions := 
  R = D + 6 ∧ 
  D = S + 8 ∧ 
  E = R - 5 ∧ 
  R + 8 = 2 * (S + 8)

-- Define the statement to be proved
theorem ages_of_siblings (h : conditions R D S E) : 
  R = 20 ∧ D = 14 ∧ S = 6 ∧ E = 15 :=
sorry

end ages_of_siblings_l4_4650


namespace hockey_league_num_games_l4_4090

theorem hockey_league_num_games :
  ∃ (num_teams : ℕ) (num_times : ℕ), 
    num_teams = 16 ∧ num_times = 10 ∧ 
    (num_teams * (num_teams - 1) / 2) * num_times = 2400 := by
  sorry

end hockey_league_num_games_l4_4090


namespace carmen_burning_candles_l4_4004

theorem carmen_burning_candles (candle_hours_per_night: ℕ) (nights_per_candle: ℕ) (candles_used: ℕ) (total_nights: ℕ) : 
  candle_hours_per_night = 2 →
  nights_per_candle = 8 / candle_hours_per_night →
  candles_used = 6 →
  total_nights = candles_used * (nights_per_candle / candle_hours_per_night) →
  total_nights = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end carmen_burning_candles_l4_4004


namespace ellipse_foci_distance_l4_4871

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end ellipse_foci_distance_l4_4871


namespace brother_pays_correct_amount_l4_4761

-- Definition of constants and variables
def friend_per_day := 5
def cousin_per_day := 4
def total_amount_collected := 119
def days := 7
def brother_per_day := 8

-- Statement of the theorem to be proven
theorem brother_pays_correct_amount :
  friend_per_day * days + cousin_per_day * days + brother_per_day * days = total_amount_collected :=
by {
  sorry
}

end brother_pays_correct_amount_l4_4761


namespace no_real_solution_l4_4408

theorem no_real_solution (n : ℝ) : (∀ x : ℝ, (x+6)*(x-3) = n + 4*x → false) ↔ n < -73/4 := by
  sorry

end no_real_solution_l4_4408


namespace tan_sum_simplification_l4_4598

open Real

theorem tan_sum_simplification :
  tan 70 + tan 50 - sqrt 3 * tan 70 * tan 50 = -sqrt 3 := by
  sorry

end tan_sum_simplification_l4_4598


namespace general_term_formula_l4_4326

theorem general_term_formula (n : ℕ) : 
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n > 1, a n - a (n-1) = 2^(n-1)) → (a n = 2^n - 1) :=
  by 
  intros a h1 hdif
  sorry

end general_term_formula_l4_4326


namespace cosine_periodicity_l4_4766

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l4_4766


namespace problem_proof_l4_4150

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def y := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem problem_proof :
  is_multiple_of 5 y ∧
  is_multiple_of 10 y ∧
  is_multiple_of 20 y ∧
  is_multiple_of 40 y := 
by
  sorry

end problem_proof_l4_4150


namespace monotonic_intervals_l4_4860

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_intervals :
  (∀ x (h : 0 < x ∧ x < Real.exp 1), 0 < f x) ∧
  (∀ x (h : Real.exp 1 < x), f x < 0) :=
by
  sorry

end monotonic_intervals_l4_4860


namespace julieta_total_spent_l4_4215

def original_price_backpack : ℕ := 50
def original_price_ring_binder : ℕ := 20
def quantity_ring_binders : ℕ := 3
def price_increase_backpack : ℕ := 5
def price_decrease_ring_binder : ℕ := 2

def total_spent (original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder : ℕ) : ℕ :=
  let new_price_backpack := original_price_backpack + price_increase_backpack
  let new_price_ring_binder := original_price_ring_binder - price_decrease_ring_binder
  new_price_backpack + (new_price_ring_binder * quantity_ring_binders)

theorem julieta_total_spent :
  total_spent original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder = 109 :=
by 
  -- Proof steps are omitted intentionally
  sorry

end julieta_total_spent_l4_4215


namespace wicket_keeper_older_than_captain_l4_4339

-- Define the team and various ages
def captain_age : ℕ := 28
def average_age_team : ℕ := 25
def number_of_players : ℕ := 11
def number_of_remaining_players : ℕ := number_of_players - 2
def average_age_remaining_players : ℕ := average_age_team - 1

theorem wicket_keeper_older_than_captain :
  ∃ (W : ℕ), W = captain_age + 3 ∧
  275 = number_of_players * average_age_team ∧
  216 = number_of_remaining_players * average_age_remaining_players ∧
  59 = 275 - 216 ∧
  W = 59 - captain_age :=
by
  sorry

end wicket_keeper_older_than_captain_l4_4339


namespace oates_reunion_attendees_l4_4594

noncomputable def total_guests : ℕ := 100
noncomputable def hall_attendees : ℕ := 70
noncomputable def both_reunions_attendees : ℕ := 10

theorem oates_reunion_attendees :
  ∃ O : ℕ, total_guests = O + hall_attendees - both_reunions_attendees ∧ O = 40 :=
by
  sorry

end oates_reunion_attendees_l4_4594


namespace largest_prime_factor_always_37_l4_4228

-- We define the cyclic sequence conditions
def cyclic_shift (seq : List ℕ) : Prop :=
  ∀ i, seq.get! (i % seq.length) % 10 = seq.get! ((i + 1) % seq.length) / 100 ∧
       (seq.get! ((i + 1) % seq.length) / 10 % 10 = seq.get! ((i + 2) % seq.length) % 10) ∧
       (seq.get! ((i + 2) % seq.length) / 10 % 10 = seq.get! ((i + 3) % seq.length) / 100)

-- Summing all elements of a list
def sum (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Prove that 37 is always a factor of the sum T
theorem largest_prime_factor_always_37 (seq : List ℕ) (h : cyclic_shift seq) : 
  37 ∣ sum seq := 
sorry

end largest_prime_factor_always_37_l4_4228


namespace solution_set_ineq_l4_4841

theorem solution_set_ineq (m : ℝ) (hm : m > 1) :
  {x : ℝ | x^2 + (m-1) * x - m >= 0} = {x : ℝ | x <= -m ∨ x >= 1} :=
sorry

end solution_set_ineq_l4_4841


namespace find_original_cost_price_l4_4763

variable (C : ℝ)

-- Conditions
def first_discount (C : ℝ) : ℝ := 0.95 * C
def second_discount (C : ℝ) : ℝ := 0.9215 * C
def loss_price (C : ℝ) : ℝ := 0.90 * C
def gain_price_before_tax (C : ℝ) : ℝ := 1.08 * C
def gain_price_after_tax (C : ℝ) : ℝ := 1.20 * C

-- Prove that original cost price is 1800
theorem find_original_cost_price 
  (h1 : first_discount C = loss_price C)
  (h2 : gain_price_after_tax C - loss_price C = 540) : 
  C = 1800 := 
sorry

end find_original_cost_price_l4_4763


namespace intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l4_4738

def U := ℝ
def A : Set ℝ := {x | 0 ≤ x ∧ x < 5}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def C_U_B : Set ℝ := {x | x < -2 ∨ x ≥ 4}

theorem intersection_A_B_eq : A ∩ B = {x | 0 ≤ x ∧ x < 4} := by
  sorry

theorem union_A_B_eq : A ∪ B = {x | -2 ≤ x ∧ x < 5} := by
  sorry

theorem intersection_A_C_U_B_eq : A ∩ C_U_B = {x | 4 ≤ x ∧ x < 5} := by
  sorry

end intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l4_4738


namespace root_of_equation_in_interval_l4_4497

theorem root_of_equation_in_interval :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 2^x = 2 - x := 
sorry

end root_of_equation_in_interval_l4_4497


namespace percentage_conveyance_l4_4523

def percentage_on_food := 40 / 100
def percentage_on_rent := 20 / 100
def percentage_on_entertainment := 10 / 100
def salary := 12500
def savings := 2500

def total_percentage_spent := percentage_on_food + percentage_on_rent + percentage_on_entertainment
def total_spent := salary - savings
def amount_spent_on_conveyance := total_spent - (salary * total_percentage_spent)
def percentage_spent_on_conveyance := (amount_spent_on_conveyance / salary) * 100

theorem percentage_conveyance : percentage_spent_on_conveyance = 10 :=
by sorry

end percentage_conveyance_l4_4523


namespace polynomial_sum_of_squares_l4_4146

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ (Q R : Polynomial ℝ), P = Q^2 + R^2 :=
sorry

end polynomial_sum_of_squares_l4_4146


namespace rhombus_side_length_l4_4560

theorem rhombus_side_length (total_length : ℕ) (num_sides : ℕ) (h1 : total_length = 32) (h2 : num_sides = 4) :
    total_length / num_sides = 8 :=
by
  -- Proof will be provided here
  sorry

end rhombus_side_length_l4_4560


namespace cost_of_peaches_eq_2_per_pound_l4_4723

def initial_money : ℕ := 20
def after_buying_peaches : ℕ := 14
def pounds_of_peaches : ℕ := 3
def cost_per_pound : ℕ := 2

theorem cost_of_peaches_eq_2_per_pound (h: initial_money - after_buying_peaches = pounds_of_peaches * cost_per_pound) :
  cost_per_pound = 2 := by
  sorry

end cost_of_peaches_eq_2_per_pound_l4_4723


namespace lcm_24_36_42_l4_4117

-- Definitions of the numbers involved
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 42

-- Statement for the lowest common multiple
theorem lcm_24_36_42 : Nat.lcm (Nat.lcm a b) c = 504 :=
by
  -- The proof will be filled in here
  sorry

end lcm_24_36_42_l4_4117


namespace smallest_sum_of_exterior_angles_l4_4798

open Real

theorem smallest_sum_of_exterior_angles 
  (p q r : ℕ) 
  (hp : p > 2) 
  (hq : q > 2) 
  (hr : r > 2) 
  (hpq : p ≠ q) 
  (hqr : q ≠ r) 
  (hrp : r ≠ p) 
  : (360 / p + 360 / q + 360 / r) ≥ 282 ∧ 
    (360 / p + 360 / q + 360 / r) = 282 → 
    360 / p = 120 ∧ 360 / q = 90 ∧ 360 / r = 72 := 
sorry

end smallest_sum_of_exterior_angles_l4_4798


namespace stormi_needs_more_money_to_afford_bicycle_l4_4679

-- Definitions from conditions
def money_washed_cars : ℕ := 3 * 10
def money_mowed_lawns : ℕ := 2 * 13
def bicycle_cost : ℕ := 80
def total_earnings : ℕ := money_washed_cars + money_mowed_lawns

-- The goal to prove 
theorem stormi_needs_more_money_to_afford_bicycle :
  (bicycle_cost - total_earnings) = 24 := by
  sorry

end stormi_needs_more_money_to_afford_bicycle_l4_4679


namespace xyz_value_l4_4230

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + 2 * x * y * z = 20) 
  : x * y * z = 20 := 
by
  sorry

end xyz_value_l4_4230


namespace isosceles_perimeter_l4_4865

theorem isosceles_perimeter (peri_eqt : ℕ) (side_eqt : ℕ) (base_iso : ℕ) (side_iso : ℕ)
    (h1 : peri_eqt = 60)
    (h2 : side_eqt = peri_eqt / 3)
    (h3 : side_iso = side_eqt)
    (h4 : base_iso = 25) :
  2 * side_iso + base_iso = 65 :=
by
  sorry

end isosceles_perimeter_l4_4865


namespace trajectory_of_P_below_x_axis_l4_4185

theorem trajectory_of_P_below_x_axis (x y : ℝ) (P_below_x_axis : y < 0)
    (tangent_to_parabola : ∃ A B: ℝ × ℝ, A.1^2 = 2 * A.2 ∧ B.1^2 = 2 * B.2 ∧ (x^2 + y^2 = 1))
    (AB_tangent_to_circle : ∀ (x0 y0 : ℝ), x0^2 + y0^2 = 1 → x0 * x + y0 * y = 1) :
    y^2 - x^2 = 1 :=
sorry

end trajectory_of_P_below_x_axis_l4_4185


namespace certain_number_value_l4_4306

theorem certain_number_value
  (t b c x : ℝ)
  (h1 : (t + b + c + x + 15) / 5 = 12)
  (h2 : (t + b + c + 29) / 4 = 15) :
  x = 14 :=
by 
  sorry

end certain_number_value_l4_4306


namespace hannah_money_left_l4_4475

variable (initial_amount : ℕ) (amount_spent_rides : ℕ) (amount_spent_dessert : ℕ)
  (remaining_after_rides : ℕ) (remaining_money : ℕ)

theorem hannah_money_left :
  initial_amount = 30 →
  amount_spent_rides = initial_amount / 2 →
  remaining_after_rides = initial_amount - amount_spent_rides →
  amount_spent_dessert = 5 →
  remaining_money = remaining_after_rides - amount_spent_dessert →
  remaining_money = 10 := by
  sorry

end hannah_money_left_l4_4475


namespace ratio_shorter_to_longer_l4_4508

theorem ratio_shorter_to_longer (x y : ℝ) (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = y / 3) : x / y = 5 / 12 :=
sorry

end ratio_shorter_to_longer_l4_4508


namespace edward_initial_money_l4_4124

theorem edward_initial_money (initial_cost_books : ℝ) (discount_percent : ℝ) (num_pens : ℕ) 
  (cost_per_pen : ℝ) (money_left : ℝ) : 
  initial_cost_books = 40 → discount_percent = 0.25 → num_pens = 3 → cost_per_pen = 2 → money_left = 6 → 
  (initial_cost_books * (1 - discount_percent) + num_pens * cost_per_pen + money_left) = 42 :=
by
  sorry

end edward_initial_money_l4_4124


namespace inverse_proportional_l4_4674

-- Define the variables and the condition
variables {R : Type*} [CommRing R] {x y k : R}
-- Assuming x and y are non-zero
variables (hx : x ≠ 0) (hy : y ≠ 0)

-- Define the constant product relationship
def product_constant (x y k : R) : Prop := x * y = k

-- The main statement that needs to be proved
theorem inverse_proportional (h : product_constant x y k) : 
  ∃ k, x * y = k :=
by sorry

end inverse_proportional_l4_4674


namespace compute_expression_l4_4437

theorem compute_expression : 12 * (1 / 26) * 52 * 4 = 96 :=
by
  sorry

end compute_expression_l4_4437


namespace arithmetic_sequence_sum_l4_4373

theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℕ) (h1 : S n = 54) (h2 : S (2 * n) = 72) :
  S (3 * n) = 78 :=
sorry

end arithmetic_sequence_sum_l4_4373


namespace skee_ball_tickets_l4_4734

-- Represent the given conditions as Lean definitions
def whack_a_mole_tickets : ℕ := 33
def candy_cost_per_piece : ℕ := 6
def candies_bought : ℕ := 7
def total_candy_tickets : ℕ := candies_bought * candy_cost_per_piece

-- Goal: Prove the number of tickets won playing 'skee ball'
theorem skee_ball_tickets (h : 42 = total_candy_tickets): whack_a_mole_tickets + 9 = total_candy_tickets :=
by {
  sorry
}

end skee_ball_tickets_l4_4734


namespace greatest_constant_triangle_l4_4923

theorem greatest_constant_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  ∃ N : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → c + a > b → (a^2 + b^2 + a * b) / c^2 > N) ∧ N = 3 / 4 :=
  sorry

end greatest_constant_triangle_l4_4923


namespace box_weight_l4_4886

theorem box_weight (W : ℝ) (h : 7 * (W - 20) = 3 * W) : W = 35 := by
  sorry

end box_weight_l4_4886


namespace find_difference_l4_4265

theorem find_difference (P : ℝ) (hP : P > 150) :
  let q := P - 150
  let A := 0.2 * P
  let B := 40
  let C := 0.3 * q
  ∃ w z, (0.2 * (150 + 50) >= B) ∧ (30 + 0.2 * q >= 0.3 * q) ∧ 150 + 50 = w ∧ 150 + 300 = z ∧ z - w = 250 :=
by
  sorry

end find_difference_l4_4265


namespace triangle_area_correct_l4_4276

/-- Define the points of the triangle -/
def x1 : ℝ := -4
def y1 : ℝ := 2
def x2 : ℝ := 2
def y2 : ℝ := 8
def x3 : ℝ := -2
def y3 : ℝ := -2

/-- Define the area calculation function -/
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Define the area of the given triangle -/
def given_triangle_area : ℝ :=
  triangle_area x1 y1 x2 y2 x3 y3

/-- The goal is to prove that the area of the given triangle is 22 square units -/
theorem triangle_area_correct : given_triangle_area = 22 := by
  sorry

end triangle_area_correct_l4_4276


namespace average_beef_sales_l4_4769

def ground_beef_sales.Thur : ℕ := 210
def ground_beef_sales.Fri : ℕ := 2 * ground_beef_sales.Thur
def ground_beef_sales.Sat : ℕ := 150
def ground_beef_sales.total : ℕ := ground_beef_sales.Thur + ground_beef_sales.Fri + ground_beef_sales.Sat
def ground_beef_sales.days : ℕ := 3
def ground_beef_sales.average : ℕ := ground_beef_sales.total / ground_beef_sales.days

theorem average_beef_sales (thur : ℕ) (fri : ℕ) (sat : ℕ) (days : ℕ) (total : ℕ) (avg : ℕ) :
  thur = 210 → 
  fri = 2 * thur → 
  sat = 150 → 
  total = thur + fri + sat → 
  days = 3 → 
  avg = total / days → 
  avg = 260 := by
    sorry

end average_beef_sales_l4_4769


namespace total_rocks_is_300_l4_4795

-- Definitions of rock types in Cliff's collection
variables (I S M : ℕ) -- I: number of igneous rocks, S: number of sedimentary rocks, M: number of metamorphic rocks
variables (shinyI shinyS shinyM : ℕ) -- shinyI: shiny igneous rocks, shinyS: shiny sedimentary rocks, shinyM: shiny metamorphic rocks

-- Given conditions
def igneous_one_third_shiny (I shinyI : ℕ) := 2 * shinyI = 3 * I
def sedimentary_two_ig_as_sed (S I : ℕ) := S = 2 * I
def metamorphic_twice_as_ig (M I : ℕ) := M = 2 * I
def shiny_igneous_is_40 (shinyI : ℕ) := shinyI = 40
def one_fifth_sed_shiny (S shinyS : ℕ) := 5 * shinyS = S
def three_quarters_met_shiny (M shinyM : ℕ) := 4 * shinyM = 3 * M

-- Theorem statement
theorem total_rocks_is_300 (I S M shinyI shinyS shinyM : ℕ)
  (h1 : igneous_one_third_shiny I shinyI)
  (h2 : sedimentary_two_ig_as_sed S I)
  (h3 : metamorphic_twice_as_ig M I)
  (h4 : shiny_igneous_is_40 shinyI)
  (h5 : one_fifth_sed_shiny S shinyS)
  (h6 : three_quarters_met_shiny M shinyM) :
  (I + S + M) = 300 :=
sorry -- Proof to be completed

end total_rocks_is_300_l4_4795


namespace cab_drivers_income_on_third_day_l4_4983

theorem cab_drivers_income_on_third_day
  (day1 day2 day4 day5 avg_income n_days : ℝ)
  (h_day1 : day1 = 600)
  (h_day2 : day2 = 250)
  (h_day4 : day4 = 400)
  (h_day5 : day5 = 800)
  (h_avg_income : avg_income = 500)
  (h_n_days : n_days = 5) :
  ∃ day3 : ℝ, (day1 + day2 + day3 + day4 + day5) / n_days = avg_income ∧ day3 = 450 :=
by
  sorry

end cab_drivers_income_on_third_day_l4_4983


namespace combined_rate_of_three_cars_l4_4488

theorem combined_rate_of_three_cars
  (m : ℕ)
  (ray_avg : ℕ)
  (tom_avg : ℕ)
  (alice_avg : ℕ)
  (h1 : ray_avg = 30)
  (h2 : tom_avg = 15)
  (h3 : alice_avg = 20) :
  let total_distance := 3 * m
  let total_gasoline := m / ray_avg + m / tom_avg + m / alice_avg
  (total_distance / total_gasoline) = 20 := 
by
  sorry

end combined_rate_of_three_cars_l4_4488


namespace general_form_line_eq_line_passes_fixed_point_l4_4558

-- (Ⅰ) Prove that if m = 1/2 and point P (1/2, 2), the general form equation of line l is 2x - y + 1 = 0
theorem general_form_line_eq (m n : ℝ) (h1 : m = 1/2) (h2 : n = 1 / (1 - m)) (h3 : n = 2) (P : (ℝ × ℝ)) (hP : P = (1/2, 2)) :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 1 := sorry

-- (Ⅱ) Prove that if point P(m,n) is on the line l0, then the line mx + (n-1)y + n + 5 = 0 passes through a fixed point, coordinates (1,1)
theorem line_passes_fixed_point (m n : ℝ) (h1 : m + 2 * n + 4 = 0) :
  ∀ (x y : ℝ), (m * x + (n - 1) * y + n + 5 = 0) ↔ (x = 1) ∧ (y = 1) := sorry

end general_form_line_eq_line_passes_fixed_point_l4_4558


namespace parallel_line_with_y_intercept_l4_4340

theorem parallel_line_with_y_intercept (x y : ℝ) (m : ℝ) : 
  ((x + y + 4 = 0) → (x + y + m = 0)) ∧ (m = 1)
 := by sorry

end parallel_line_with_y_intercept_l4_4340


namespace smallest_value_of_3a_plus_2_l4_4885

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 3 * a + 2 = -1 :=
by
  sorry

end smallest_value_of_3a_plus_2_l4_4885


namespace root_difference_l4_4593

theorem root_difference (p : ℝ) (r s : ℝ) 
  (h₁ : r + s = 2 * p) 
  (h₂ : r * s = (p^2 - 4) / 3) : 
  r - s = 2 * (Real.sqrt 3) / 3 :=
by
  sorry

end root_difference_l4_4593


namespace unobstructed_sight_l4_4959

-- Define the curve C as y = 2x^2
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define point A and point B
def pointA : ℝ × ℝ := (0, -2)
def pointB (a : ℝ) : ℝ × ℝ := (3, a)

-- Statement of the problem
theorem unobstructed_sight {a : ℝ} (h : ∀ x : ℝ, 0 ≤ x → x ≤ 3 → 4 * x - 2 ≥ 2 * x^2) : a < 10 :=
sorry

end unobstructed_sight_l4_4959


namespace distinct_values_of_products_l4_4058

theorem distinct_values_of_products (n : ℤ) (h : 1 ≤ n) :
  ¬ ∃ a b c d : ℤ, n^2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2 ∧ ad = bc :=
sorry

end distinct_values_of_products_l4_4058


namespace distinct_schedules_l4_4328

-- Define the problem setting and assumptions
def subjects := ["Chinese", "Mathematics", "Politics", "English", "Physical Education", "Art"]

-- Given conditions
def math_in_first_three_periods (schedule : List String) : Prop :=
  ∃ k, (k < 3) ∧ (schedule.get! k = "Mathematics")

def english_not_in_sixth_period (schedule : List String) : Prop :=
  schedule.get! 5 ≠ "English"

-- Define the proof problem
theorem distinct_schedules : 
  ∃! (schedules : List (List String)), 
  (∀ schedule ∈ schedules, 
    math_in_first_three_periods schedule ∧ 
    english_not_in_sixth_period schedule) ∧
  schedules.length = 288 :=
by
  sorry

end distinct_schedules_l4_4328


namespace molecular_weight_of_NH4Br_l4_4068

def atomic_weight (element : String) : Real :=
  match element with
  | "N" => 14.01
  | "H" => 1.01
  | "Br" => 79.90
  | _ => 0.0

def molecular_weight (composition : List (String × Nat)) : Real :=
  composition.foldl (λ acc (elem, count) => acc + count * atomic_weight elem) 0

theorem molecular_weight_of_NH4Br :
  molecular_weight [("N", 1), ("H", 4), ("Br", 1)] = 97.95 :=
by
  sorry

end molecular_weight_of_NH4Br_l4_4068


namespace students_neither_class_l4_4007

theorem students_neither_class : 
  let total_students := 1500
  let music_students := 300
  let art_students := 200
  let dance_students := 100
  let theater_students := 50
  let music_art_students := 80
  let music_dance_students := 40
  let music_theater_students := 30
  let art_dance_students := 25
  let art_theater_students := 20
  let dance_theater_students := 10
  let music_art_dance_students := 50
  let music_art_theater_students := 30
  let art_dance_theater_students := 20
  let music_dance_theater_students := 10
  let all_four_students := 5
  total_students - 
    (music_students + 
     art_students + 
     dance_students + 
     theater_students - 
     (music_art_students + 
      music_dance_students + 
      music_theater_students + 
      art_dance_students + 
      art_theater_students + 
      dance_theater_students) + 
     (music_art_dance_students + 
      music_art_theater_students + 
      art_dance_theater_students + 
      music_dance_theater_students) - 
     all_four_students) = 950 :=
sorry

end students_neither_class_l4_4007


namespace cos_double_angle_l4_4631

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_double_angle_l4_4631


namespace quadratic_real_roots_k_le_one_fourth_l4_4275

theorem quadratic_real_roots_k_le_one_fourth (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - (4 * k - 2) * x + k^2 = 0) ↔ k ≤ 1/4 :=
sorry

end quadratic_real_roots_k_le_one_fourth_l4_4275


namespace file_organization_ratio_l4_4575

variable (X : ℕ) -- The number of files organized in the morning
variable (total_files morning_files afternoon_files missing_files : ℕ)

-- Conditions
def condition1 : total_files = 60 := by sorry
def condition2 : afternoon_files = 15 := by sorry
def condition3 : missing_files = 15 := by sorry
def condition4 : morning_files = X := by sorry
def condition5 : morning_files + afternoon_files + missing_files = total_files := by sorry

-- Question
def ratio_morning_to_total : Prop :=
  let organized_files := total_files - afternoon_files - missing_files
  (organized_files / total_files : ℚ) = 1 / 2

-- Proof statement
theorem file_organization_ratio : 
  ∀ (X total_files morning_files afternoon_files missing_files : ℕ), 
    total_files = 60 → 
    afternoon_files = 15 → 
    missing_files = 15 → 
    morning_files = X → 
    morning_files + afternoon_files + missing_files = total_files → 
    (X / 60 : ℚ) = 1 / 2 := by 
  sorry

end file_organization_ratio_l4_4575


namespace purely_imaginary_complex_number_l4_4229

theorem purely_imaginary_complex_number (a : ℝ) 
  (h1 : (a^2 - 4 * a + 3 = 0))
  (h2 : a ≠ 1) 
  : a = 3 := 
sorry

end purely_imaginary_complex_number_l4_4229


namespace min_value_abs_x1_x2_l4_4816

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem min_value_abs_x1_x2 
  (a : ℝ) (x1 x2 : ℝ)
  (h_symm : ∃ k : ℤ, -π / 6 - (Real.arctan (Real.sqrt 3 / a)) = (k * π + π / 2))
  (h_diff : f a x1 - f a x2 = -4) :
  |x1 + x2| = (2 * π) / 3 := 
sorry

end min_value_abs_x1_x2_l4_4816


namespace roots_identity_l4_4123

theorem roots_identity (p q r : ℝ) (h₁ : p + q + r = 15) (h₂ : p * q + q * r + r * p = 25) (h₃ : p * q * r = 10) :
  (1 + p) * (1 + q) * (1 + r) = 51 :=
by sorry

end roots_identity_l4_4123


namespace triangle_area_l4_4119

variables {A B C a b c : ℝ}

/-- In triangle ABC, the sides opposite to angles A, B, and C are denoted as a, b, and c, respectively.
It is given that b * sin C + c * sin B = 4 * a * sin B * sin C and b^2 + c^2 - a^2 = 8.
Prove that the area of triangle ABC is 4 * sqrt 3 / 3. -/
theorem triangle_area (h1 : b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C)
  (h2 : b^2 + c^2 - a^2 = 8) :
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 / 3 :=
sorry

end triangle_area_l4_4119


namespace find_a3_l4_4664

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : geometric_sequence a r)
  (h2 : a 0 * a 1 * a 2 * a 3 * a 4 = 32):
  a 2 = 2 :=
sorry

end find_a3_l4_4664


namespace vector_orthogonality_l4_4539

variables (x : ℝ)

def vec_a := (x - 1, 2)
def vec_b := (1, x)

theorem vector_orthogonality :
  (vec_a x).fst * (vec_b x).fst + (vec_a x).snd * (vec_b x).snd = 0 ↔ x = 1 / 3 := by
  sorry

end vector_orthogonality_l4_4539


namespace row_even_col_odd_contradiction_row_odd_col_even_contradiction_l4_4385

theorem row_even_col_odd_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∃ i : Fin 15, M r i = 2) ∧ 
      (∀ c : Fin 15, ∀ j : Fin 20, M j c = 5)) := 
sorry

theorem row_odd_col_even_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∀ i : Fin 15, M r i = 5) ∧ 
      (∀ c : Fin 15, ∃ j : Fin 20, M j c = 2)) := 
sorry

end row_even_col_odd_contradiction_row_odd_col_even_contradiction_l4_4385


namespace digits_arithmetic_l4_4851

theorem digits_arithmetic :
  (12 / 3 / 4) * (56 / 7 / 8) = 1 :=
by
  sorry

end digits_arithmetic_l4_4851


namespace ratio_of_triangle_BFD_to_square_ABCE_l4_4751

def is_square (ABCE : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a b c e : ℝ, ABCE a b c e → a = b ∧ b = c ∧ c = e

def ratio_of_areas (AF FE CD DE : ℝ) (ratio : ℝ) : Prop :=
  AF = 3 * FE ∧ CD = 3 * DE ∧ ratio = 1 / 2

theorem ratio_of_triangle_BFD_to_square_ABCE (AF FE CD DE ratio : ℝ) (ABCE : ℝ → ℝ → ℝ → ℝ → Prop)
  (h1 : is_square ABCE)
  (h2 : AF = 3 * FE) (h3 : CD = 3 * DE) : ratio_of_areas AF FE CD DE (1 / 2) :=
by
  sorry

end ratio_of_triangle_BFD_to_square_ABCE_l4_4751


namespace cubic_polynomials_common_roots_c_d_l4_4627

theorem cubic_polynomials_common_roots_c_d (c d : ℝ) :
  (∀ (r s : ℝ), r ≠ s ∧
     (r^3 + c*r^2 + 12*r + 7 = 0) ∧ (s^3 + c*s^2 + 12*s + 7 = 0) ∧
     (r^3 + d*r^2 + 15*r + 9 = 0) ∧ (s^3 + d*s^2 + 15*s + 9 = 0)) →
  (c = -5 ∧ d = -6) := 
by
  sorry

end cubic_polynomials_common_roots_c_d_l4_4627


namespace count_integers_satisfying_inequality_l4_4257

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), S.card = 8 ∧ ∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 :=
by
  sorry

end count_integers_satisfying_inequality_l4_4257


namespace tank_capacity_l4_4039

theorem tank_capacity :
  let rateA := 40  -- Pipe A fills at 40 liters per minute
  let rateB := 30  -- Pipe B fills at 30 liters per minute
  let rateC := -20  -- Pipe C (drains) at 20 liters per minute, thus negative contribution
  let cycle_duration := 3  -- The cycle duration is 3 minutes
  let total_duration := 51  -- The tank gets full in 51 minutes
  let net_per_cycle := rateA + rateB + rateC  -- Net fill per cycle of 3 minutes
  let num_cycles := total_duration / cycle_duration  -- Number of complete cycles
  let tank_capacity := net_per_cycle * num_cycles  -- Tank capacity in liters
  tank_capacity = 850  -- Assertion that needs to be proven
:= by
  let rateA := 40
  let rateB := 30
  let rateC := -20
  let cycle_duration := 3
  let total_duration := 51
  let net_per_cycle := rateA + rateB + rateC
  let num_cycles := total_duration / cycle_duration
  let tank_capacity := net_per_cycle * num_cycles
  have : tank_capacity = 850 := by
    sorry
  assumption

end tank_capacity_l4_4039


namespace balloon_arrangement_count_l4_4233

theorem balloon_arrangement_count :
  let n := 7
  let l := 2
  let o := 2
  n.factorial / (l.factorial * o.factorial) = 1260 :=
by
  sorry

end balloon_arrangement_count_l4_4233


namespace multiplicative_inverse_modulo_2799_l4_4016

theorem multiplicative_inverse_modulo_2799 :
  ∃ n : ℤ, 0 ≤ n ∧ n < 2799 ∧ (225 * n) % 2799 = 1 :=
by {
  -- conditions are expressed directly in the theorem assumption
  sorry
}

end multiplicative_inverse_modulo_2799_l4_4016


namespace cos_double_angle_l4_4111

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2 * n) = 12) : Real.cos (2 * θ) = 5 / 6 := 
sorry

end cos_double_angle_l4_4111


namespace floor_eq_correct_l4_4698

theorem floor_eq_correct (y : ℝ) (h : ⌊y⌋ + y = 17 / 4) : y = 9 / 4 :=
sorry

end floor_eq_correct_l4_4698


namespace sum_of_xyz_l4_4050

theorem sum_of_xyz (x y z : ℝ) (h : (x - 5)^2 + (y - 3)^2 + (z - 1)^2 = 0) : x + y + z = 9 :=
by {
  sorry
}

end sum_of_xyz_l4_4050


namespace evaporation_rate_is_200_ml_per_hour_l4_4241

-- Definitions based on the given conditions
def faucet_drip_rate : ℕ := 40 -- ml per minute
def running_time : ℕ := 9 -- hours
def dumped_water : ℕ := 12000 -- ml (converted from liters)
def water_left : ℕ := 7800 -- ml

-- Alias for total water dripped in running_time
noncomputable def total_dripped_water : ℕ := faucet_drip_rate * 60 * running_time

-- Total water that should have been in the bathtub without evaporation
noncomputable def total_without_evaporation : ℕ := total_dripped_water - dumped_water

-- Water evaporated
noncomputable def evaporated_water : ℕ := total_without_evaporation - water_left

-- Evaporation rate in ml/hour
noncomputable def evaporation_rate : ℕ := evaporated_water / running_time

-- The goal theorem statement
theorem evaporation_rate_is_200_ml_per_hour : evaporation_rate = 200 := by
  -- proof here
  sorry

end evaporation_rate_is_200_ml_per_hour_l4_4241


namespace bowling_ball_weight_l4_4718

theorem bowling_ball_weight (b k : ℝ)  (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 := by
  sorry

end bowling_ball_weight_l4_4718


namespace problem_statement_l4_4697

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B C D : V)
  (BC CD : V)
  (AC AB AD : V)

theorem problem_statement
  (h1 : BC = 2 • CD)
  (h2 : BC = AC - AB) :
  AD = (3 / 2 : ℝ) • AC - (1 / 2 : ℝ) • AB :=
sorry

end problem_statement_l4_4697


namespace arithmetic_sequence_general_term_l4_4972

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_increasing : d > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = a 2 ^ 2 - 4) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l4_4972


namespace alice_age_l4_4207

theorem alice_age (a m : ℕ) (h1 : a = m - 18) (h2 : a + m = 50) : a = 16 := by
  sorry

end alice_age_l4_4207


namespace change_in_us_volume_correct_l4_4917

-- Definition: Change in the total import and export volume of goods in a given year
def change_in_volume (country : String) : Float :=
  if country = "China" then 7.5
  else if country = "United States" then -6.4
  else 0

-- Theorem: The change in the total import and export volume of goods in the United States is correctly represented.
theorem change_in_us_volume_correct :
  change_in_volume "United States" = -6.4 := by
  sorry

end change_in_us_volume_correct_l4_4917


namespace tax_rate_correct_l4_4735

def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 89.6

def taxable_amount : ℝ := total_value - non_taxable_amount

theorem tax_rate_correct : (tax_paid / taxable_amount) * 100 = 8 := by
  sorry

end tax_rate_correct_l4_4735


namespace endangered_species_count_l4_4949

section BirdsSanctuary

-- Define the given conditions
def pairs_per_species : ℕ := 7
def total_pairs : ℕ := 203

-- Define the result to be proved
theorem endangered_species_count : total_pairs / pairs_per_species = 29 := by
  sorry

end BirdsSanctuary

end endangered_species_count_l4_4949


namespace math_problem_l4_4180

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c + 2 * (a + b + c) = 672 :=
by
  sorry

end math_problem_l4_4180


namespace scientific_notation_l4_4313

variables (n : ℕ) (h : n = 505000)

theorem scientific_notation : n = 505000 → "5.05 * 10^5" = "scientific notation of 505000" :=
by
  intro h
  sorry

end scientific_notation_l4_4313


namespace integer_solutions_l4_4164

-- Define the problem statement in Lean
theorem integer_solutions :
  {p : ℤ × ℤ | ∃ x y : ℤ, p = (x, y) ∧ x^2 + x = y^4 + y^3 + y^2 + y} =
  {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)} :=
by
  sorry

end integer_solutions_l4_4164


namespace mixed_number_expression_l4_4271

open Real

-- Definitions of the given mixed numbers
def mixed_number1 : ℚ := (37 / 7)
def mixed_number2 : ℚ := (18 / 5)
def mixed_number3 : ℚ := (19 / 6)
def mixed_number4 : ℚ := (9 / 4)

-- Main theorem statement
theorem mixed_number_expression :
  25 * (mixed_number1 - mixed_number2) / (mixed_number3 + mixed_number4) = 7 + 49 / 91 :=
by
  sorry

end mixed_number_expression_l4_4271


namespace sqrt_fraction_expression_l4_4202

theorem sqrt_fraction_expression : 
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + (Real.sqrt (9 / 4) + Real.sqrt (4 / 9))^2) = (199 / 36) := 
by
  sorry

end sqrt_fraction_expression_l4_4202


namespace rice_mixture_ratio_l4_4337

theorem rice_mixture_ratio
  (cost_variety1 : ℝ := 5) 
  (cost_variety2 : ℝ := 8.75) 
  (desired_cost_mixture : ℝ := 7.50) 
  (x y : ℝ) :
  5 * x + 8.75 * y = 7.50 * (x + y) → 
  y / x = 2 :=
by
  intro h
  sorry

end rice_mixture_ratio_l4_4337


namespace area_of_triangle_tangent_line_l4_4931

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def tangent_line_at_1 (y x : ℝ) : Prop := y = x - 1

theorem area_of_triangle_tangent_line :
  let tangent_intercept_x : ℝ := 1
  let tangent_intercept_y : ℝ := -1
  let area_of_triangle : ℝ := 1 / 2 * tangent_intercept_x * -tangent_intercept_y
  area_of_triangle = 1 / 2 :=
by
  sorry

end area_of_triangle_tangent_line_l4_4931


namespace price_per_kilo_of_bananas_l4_4546

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

end price_per_kilo_of_bananas_l4_4546


namespace existence_of_nonnegative_value_l4_4009

theorem existence_of_nonnegative_value :
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 12 ≥ 0 := 
by
  sorry

end existence_of_nonnegative_value_l4_4009


namespace min_value_of_g_function_l4_4247

noncomputable def g (x : Real) := x + (x + 1) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem min_value_of_g_function : ∀ x : ℝ, x > 0 → g x ≥ 3 := sorry

end min_value_of_g_function_l4_4247


namespace total_time_taken_l4_4573

theorem total_time_taken (b km : ℝ) : 
  (b / 50 + km / 80) = (8 * b + 5 * km) / 400 := 
sorry

end total_time_taken_l4_4573


namespace coefficient_x4_in_expansion_sum_l4_4533

theorem coefficient_x4_in_expansion_sum :
  (Nat.choose 5 4 + Nat.choose 6 4 + Nat.choose 7 4 = 55) :=
by
  sorry

end coefficient_x4_in_expansion_sum_l4_4533


namespace not_p_and_p_or_q_implies_q_l4_4249

theorem not_p_and_p_or_q_implies_q (p q : Prop) (h1 : ¬ p) (h2 : p ∨ q) : q :=
by
  have h3 : p := sorry
  have h4 : false := sorry
  exact sorry

end not_p_and_p_or_q_implies_q_l4_4249


namespace problem_l4_4758

noncomputable def a : Real := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : Real := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : Real := Real.sqrt 6 / 2

theorem problem :
  a < c ∧ c < b := by
  sorry

end problem_l4_4758


namespace find_radius_of_circle_l4_4023

variable (AB BC AC R : ℝ)

-- Conditions
def is_right_triangle (ABC : Type) (AB BC : ℝ) (AC : outParam ℝ) : Prop :=
  AC = Real.sqrt (AB^2 + BC^2)

def is_tangent (O : Type) (AB BC AC R : ℝ) : Prop :=
  ∃ (P Q : ℝ), P = R ∧ Q = R ∧ P < AC ∧ Q < AC

theorem find_radius_of_circle (h1 : is_right_triangle ABC 21 28 AC) (h2 : is_tangent O 21 28 AC R) : R = 12 :=
sorry

end find_radius_of_circle_l4_4023


namespace thabo_books_l4_4970

/-- Thabo's book count puzzle -/
theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 200) : H = 35 :=
by
  -- sorry is used to skip the proof, only state the theorem.
  sorry

end thabo_books_l4_4970


namespace second_integer_is_66_l4_4358

-- Define the conditions
def are_two_units_apart (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = a + 4

def sum_of_first_and_third_is_132 (a b c : ℤ) : Prop :=
  a + c = 132

-- State the theorem
theorem second_integer_is_66 (a b c : ℤ) 
  (H1 : are_two_units_apart a b c) 
  (H2 : sum_of_first_and_third_is_132 a b c) : b = 66 :=
by
  sorry -- Proof omitted

end second_integer_is_66_l4_4358


namespace mean_of_set_eq_10point6_l4_4172

open Real -- For real number operations

theorem mean_of_set_eq_10point6 (n : ℝ)
  (h : n + 7 = 11) :
  (4 + 7 + 11 + 13 + 18) / 5 = 10.6 :=
by
  have h1 : n = 4 := by linarith
  sorry -- skip the proof part

end mean_of_set_eq_10point6_l4_4172


namespace number_division_l4_4392

theorem number_division (n q r d : ℕ) (h1 : d = 18) (h2 : q = 11) (h3 : r = 1) (h4 : n = (d * q) + r) : n = 199 := 
by 
  sorry

end number_division_l4_4392


namespace algebraic_expression_value_l4_4099

theorem algebraic_expression_value (m n : ℝ) 
  (h1 : m * n = 3) 
  (h2 : n = m + 1) : 
  (m - n) ^ 2 * ((1 / n) - (1 / m)) = -1 / 3 :=
by sorry

end algebraic_expression_value_l4_4099


namespace cylinder_surface_area_l4_4038

theorem cylinder_surface_area
  (r : ℝ) (V : ℝ) (h_radius : r = 1) (h_volume : V = 4 * Real.pi) :
  ∃ S : ℝ, S = 10 * Real.pi :=
by
  let l := V / (Real.pi * r^2)
  have h_l : l = 4 := sorry
  let S := 2 * Real.pi * r^2 + 2 * Real.pi * r * l
  have h_S : S = 10 * Real.pi := sorry
  exact ⟨S, h_S⟩

end cylinder_surface_area_l4_4038


namespace emma_list_count_l4_4059

theorem emma_list_count : 
  let m1 := 900
  let m2 := 27000
  let d := 30
  (m1 / d <= m2 / d) → (m2 / d - m1 / d + 1 = 871) :=
by
  intros m1 m2 d h
  have h1 : m1 / d ≤ m2 / d := h
  have h2 : m2 / d - m1 / d + 1 = 871 := by sorry
  exact h2

end emma_list_count_l4_4059


namespace number_of_cows_l4_4900

variable (C H : ℕ)

section
-- Condition 1: Cows have 4 legs each
def cows_legs := C * 4

-- Condition 2: Chickens have 2 legs each
def chickens_legs := H * 2

-- Condition 3: The number of legs was 10 more than twice the number of heads
def total_legs := cows_legs C + chickens_legs H = 2 * (C + H) + 10

theorem number_of_cows : total_legs C H → C = 5 :=
by
  intros h
  sorry

end

end number_of_cows_l4_4900


namespace sufficient_but_not_necessary_l4_4515

theorem sufficient_but_not_necessary (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) :
    (x + y - 1 = 0) ∧ ¬ ∀ x y, (x + y - 1 = 0) → (x = 2 ∧ y = -1) :=
  by
  sorry

end sufficient_but_not_necessary_l4_4515


namespace tank_filling_time_with_leaks_l4_4155

theorem tank_filling_time_with_leaks (pump_time : ℝ) (leak1_time : ℝ) (leak2_time : ℝ) (leak3_time : ℝ) (fill_time : ℝ)
  (h1 : pump_time = 2)
  (h2 : fill_time = 3)
  (h3 : leak1_time = 6)
  (h4 : leak2_time = 8)
  (h5 : leak3_time = 12) :
  fill_time = 8 := 
sorry

end tank_filling_time_with_leaks_l4_4155


namespace frank_peanuts_average_l4_4398

theorem frank_peanuts_average :
  let one_dollar := 7 * 1
  let five_dollar := 4 * 5
  let ten_dollar := 2 * 10
  let twenty_dollar := 1 * 20
  let total_money := one_dollar + five_dollar + ten_dollar + twenty_dollar
  let change := 4
  let money_spent := total_money - change
  let cost_per_pound := 3
  let total_pounds := money_spent / cost_per_pound
  let days := 7
  let average_per_day := total_pounds / days
  average_per_day = 3 :=
by
  sorry

end frank_peanuts_average_l4_4398


namespace find_triplets_satisfying_equation_l4_4577

theorem find_triplets_satisfying_equation :
  ∃ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧ (x, y, z) = (2, 251, 252) :=
by
  sorry

end find_triplets_satisfying_equation_l4_4577


namespace find_f7_l4_4091

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f (x)

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = 2 * x^2

theorem find_f7 (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : periodic_function f 4)
  (h3 : specific_values f) :
  f 7 = -2 :=
by
  sorry

end find_f7_l4_4091


namespace inequalities_always_true_l4_4074

theorem inequalities_always_true (x y a b : ℝ) (hx : 0 < x) (hy : 0 < y) (ha : 0 < a) (hb : 0 < b) 
  (hxa : x ≤ a) (hyb : y ≤ b) : 
  (x + y ≤ a + b) ∧ (x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b) := by
  sorry

end inequalities_always_true_l4_4074


namespace tim_coins_value_l4_4377

variable (d q : ℕ)

-- Given Conditions
def total_coins (d q : ℕ) : Prop := d + q = 18
def quarter_to_dime_relation (d q : ℕ) : Prop := q = d + 2

-- Prove the value of the coins
theorem tim_coins_value (d q : ℕ) (h1 : total_coins d q) (h2 : quarter_to_dime_relation d q) : 10 * d + 25 * q = 330 := by
  sorry

end tim_coins_value_l4_4377


namespace shale_mix_per_pound_is_5_l4_4395

noncomputable def cost_of_shale_mix_per_pound 
  (cost_limestone : ℝ) (cost_compound : ℝ) (weight_limestone : ℝ) (total_weight : ℝ) : ℝ :=
  let total_cost_limestone := weight_limestone * cost_limestone 
  let weight_shale := total_weight - weight_limestone
  let total_cost := total_weight * cost_compound
  let total_cost_shale := total_cost - total_cost_limestone
  total_cost_shale / weight_shale

theorem shale_mix_per_pound_is_5 :
  cost_of_shale_mix_per_pound 3 4.25 37.5 100 = 5 := 
by 
  sorry

end shale_mix_per_pound_is_5_l4_4395


namespace find_p_q_of_divisible_polynomial_l4_4847

theorem find_p_q_of_divisible_polynomial :
  ∃ p q : ℤ, (p, q) = (-7, -12) ∧
    (∀ x : ℤ, (x^5 - x^4 + x^3 - p*x^2 + q*x + 4 = 0) → (x = -2 ∨ x = 1)) :=
by
  sorry

end find_p_q_of_divisible_polynomial_l4_4847


namespace find_number_l4_4892

theorem find_number (x : ℕ) (h : x + 56 = 110) : x = 54 :=
sorry

end find_number_l4_4892


namespace max_value_sqrt_expr_l4_4806

open Real

theorem max_value_sqrt_expr (x y z : ℝ)
  (h1 : x + y + z = 1)
  (h2 : x ≥ -1/3)
  (h3 : y ≥ -1)
  (h4 : z ≥ -5/3) :
  (sqrt (3 * x + 1) + sqrt (3 * y + 3) + sqrt (3 * z + 5)) ≤ 6 :=
  sorry

end max_value_sqrt_expr_l4_4806


namespace prove_ordered_pair_l4_4980

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

theorem prove_ordered_pair (h1 : p 0 = -24) (h2 : q 0 = 30) (h3 : ∀ x : ℝ, p (q x) = q (p x)) : (p 3, q 6) = (3, -24) := 
sorry

end prove_ordered_pair_l4_4980


namespace range_of_q_l4_4302

variable (x : ℝ)

def q (x : ℝ) := (3 * x^2 + 1)^2

theorem range_of_q : ∀ y, (∃ x : ℝ, x ≥ 0 ∧ y = q x) ↔ y ≥ 1 := by
  sorry

end range_of_q_l4_4302


namespace part_one_part_two_l4_4482

def universal_set : Set ℝ := Set.univ
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

noncomputable def C_R_A : Set ℝ := { x | x < 1 ∨ x ≥ 7 }
noncomputable def C_R_A_union_B : Set ℝ := C_R_A ∪ B

theorem part_one : C_R_A_union_B = { x | x < 1 ∨ x > 2 } :=
sorry

theorem part_two (a : ℝ) (h : A ⊆ C a) : a ≥ 7 :=
sorry

end part_one_part_two_l4_4482


namespace production_volume_l4_4031

/-- 
A certain school's factory produces 200 units of a certain product this year.
It is planned to increase the production volume by the same percentage \( x \)
over the next two years such that the total production volume over three years is 1400 units.
The goal is to prove that the correct equation for this scenario is:
200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400.
-/
theorem production_volume (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 := 
sorry

end production_volume_l4_4031


namespace exists_consecutive_with_square_factors_l4_4773

theorem exists_consecutive_with_square_factors (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ m : ℕ, m^2 ∣ (k + i) ∧ m > 1 :=
by {
  sorry
}

end exists_consecutive_with_square_factors_l4_4773


namespace hyperbola_m_range_l4_4485

-- Given conditions
def is_hyperbola_equation (m : ℝ) : Prop :=
  ∃ x y : ℝ, (4 - m) ≠ 0 ∧ (2 + m) ≠ 0 ∧ x^2 / (4 - m) - y^2 / (2 + m) = 1

-- Prove the range of m is -2 < m < 4
theorem hyperbola_m_range (m : ℝ) : is_hyperbola_equation m → (-2 < m ∧ m < 4) :=
by
  sorry

end hyperbola_m_range_l4_4485


namespace perpendicular_exists_l4_4216

-- Definitions for geometric entities involved

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  p1 : Point
  p2 : Point

structure Circle where
  center : Point
  radius : ℝ

-- Definitions for conditions in the problem

-- Condition 1: Point C is not on the circle
def point_not_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 ≠ circle.radius^2

-- Condition 2: Point C is on the circle
def point_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 = circle.radius^2

-- Definitions for lines and perpendicularity
def is_perpendicular (line1 : Line) (line2 : Line) : Prop :=
  (line1.p1.x - line1.p2.x) * (line2.p1.x - line2.p2.x) +
  (line1.p1.y - line1.p2.y) * (line2.p1.y - line2.p2.y) = 0

noncomputable def perpendicular_from_point_to_line (C : Point) (line : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : Line := 
  sorry

-- The Lean statement for part (a) and (b) combined into one proof.
theorem perpendicular_exists (C : Point) (lineAB : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : 
  ∃ (line_perpendicular : Line), is_perpendicular line_perpendicular lineAB ∧ 
  (line_perpendicular.p1 = C ∨ line_perpendicular.p2 = C) :=
  sorry

end perpendicular_exists_l4_4216


namespace pastries_and_cost_correct_l4_4796

def num_pastries_lola := 13 + 10 + 8 + 6
def cost_lola := 13 * 0.50 + 10 * 1.00 + 8 * 3.00 + 6 * 2.00

def num_pastries_lulu := 16 + 12 + 14 + 9
def cost_lulu := 16 * 0.50 + 12 * 1.00 + 14 * 3.00 + 9 * 2.00

def num_pastries_lila := 22 + 15 + 10 + 12
def cost_lila := 22 * 0.50 + 15 * 1.00 + 10 * 3.00 + 12 * 2.00

def num_pastries_luka := 18 + 20 + 7 + 14 + 25
def cost_luka := 18 * 0.50 + 20 * 1.00 + 7 * 3.00 + 14 * 2.00 + 25 * 1.50

def total_pastries := num_pastries_lola + num_pastries_lulu + num_pastries_lila + num_pastries_luka
def total_cost := cost_lola + cost_lulu + cost_lila + cost_luka

theorem pastries_and_cost_correct :
  total_pastries = 231 ∧ total_cost = 328.00 :=
by
  sorry

end pastries_and_cost_correct_l4_4796


namespace maximum_bags_of_milk_l4_4691

theorem maximum_bags_of_milk (bag_cost : ℚ) (promotion : ℕ → ℕ) (total_money : ℚ) 
  (h1 : bag_cost = 2.5) 
  (h2 : promotion 2 = 3) 
  (h3 : total_money = 30) : 
  ∃ n, n = 18 ∧ (total_money >= n * bag_cost - (n / 3) * bag_cost) :=
by
  sorry

end maximum_bags_of_milk_l4_4691


namespace correct_option_l4_4760

theorem correct_option 
  (A_false : ¬ (-6 - (-9)) = -3)
  (B_false : ¬ (-2 * (-5)) = -7)
  (C_false : ¬ (-x^2 + 3 * x^2) = 2)
  (D_true : (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b) :
  (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b :=
by sorry

end correct_option_l4_4760


namespace leo_peeled_potatoes_l4_4498

noncomputable def lucy_rate : ℝ := 4
noncomputable def leo_rate : ℝ := 6
noncomputable def total_potatoes : ℝ := 60
noncomputable def lucy_time_alone : ℝ := 6
noncomputable def total_potatoes_left : ℝ := total_potatoes - lucy_rate * lucy_time_alone
noncomputable def combined_rate : ℝ := lucy_rate + leo_rate
noncomputable def combined_time : ℝ := total_potatoes_left / combined_rate
noncomputable def leo_potatoes : ℝ := combined_time * leo_rate

theorem leo_peeled_potatoes :
  leo_potatoes = 22 :=
by
  sorry

end leo_peeled_potatoes_l4_4498


namespace cubic_roots_number_l4_4788

noncomputable def determinant_cubic (a b c d : ℝ) (x : ℝ) : ℝ :=
  x * (x^2 + a^2) + c * (b * x + a * b) - b * (c * a - b * x)

theorem cubic_roots_number (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ roots : ℕ, (roots = 1 ∨ roots = 3) :=
  sorry

end cubic_roots_number_l4_4788


namespace third_box_weight_l4_4496

def box1_height := 1 -- inches
def box1_width := 2 -- inches
def box1_length := 4 -- inches
def box1_weight := 30 -- grams

def box2_height := 3 * box1_height
def box2_width := 2 * box1_width
def box2_length := box1_length

def box3_height := box2_height
def box3_width := box2_width / 2
def box3_length := box2_length

def volume (height : ℕ) (width : ℕ) (length : ℕ) : ℕ := height * width * length

def weight (box1_weight : ℕ) (box1_volume : ℕ) (box3_volume : ℕ) : ℕ := 
  box3_volume / box1_volume * box1_weight

theorem third_box_weight :
  weight box1_weight (volume box1_height box1_width box1_length) 
  (volume box3_height box3_width box3_length) = 90 :=
by
  sorry

end third_box_weight_l4_4496


namespace juan_european_stamps_total_cost_l4_4365

/-- Define the cost of European stamps collection for Juan -/
def total_cost_juan_stamps : ℝ := 
  -- Costs of stamps from the 1980s
  (15 * 0.07) + (11 * 0.06) + (14 * 0.08) +
  -- Costs of stamps from the 1990s
  (14 * 0.07) + (10 * 0.06) + (12 * 0.08)

/-- Prove that the total cost for European stamps from the 80s and 90s is $5.37 -/
theorem juan_european_stamps_total_cost : total_cost_juan_stamps = 5.37 :=
  by sorry

end juan_european_stamps_total_cost_l4_4365


namespace globe_division_l4_4581

theorem globe_division (parallels meridians : ℕ)
  (h_parallels : parallels = 17)
  (h_meridians : meridians = 24) :
  let slices_per_sector := parallels + 1
  let sectors := meridians
  let total_parts := slices_per_sector * sectors
  total_parts = 432 := by
  sorry

end globe_division_l4_4581


namespace total_days_in_month_eq_l4_4911

-- Definition of the conditions
def took_capsules_days : ℕ := 27
def forgot_capsules_days : ℕ := 4

-- The statement to be proved
theorem total_days_in_month_eq : took_capsules_days + forgot_capsules_days = 31 := by
  sorry

end total_days_in_month_eq_l4_4911


namespace ramu_profit_percent_l4_4318

theorem ramu_profit_percent
  (cost_of_car : ℕ)
  (cost_of_repairs : ℕ)
  (selling_price : ℕ)
  (total_cost : ℕ := cost_of_car + cost_of_repairs)
  (profit : ℕ := selling_price - total_cost)
  (profit_percent : ℚ := ((profit : ℚ) / total_cost) * 100)
  (h1 : cost_of_car = 42000)
  (h2 : cost_of_repairs = 15000)
  (h3 : selling_price = 64900) :
  profit_percent = 13.86 :=
by
  sorry

end ramu_profit_percent_l4_4318


namespace n_minus_m_eq_200_l4_4870

-- Define the parameters
variable (m n x : ℝ)

-- State the conditions
def condition1 : Prop := m ≤ 8 * x - 1 ∧ 8 * x - 1 ≤ n 
def condition2 : Prop := (n + 1)/8 - (m + 1)/8 = 25

-- State the theorem to prove
theorem n_minus_m_eq_200 (h1 : condition1 m n x) (h2 : condition2 m n) : n - m = 200 := 
by 
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end n_minus_m_eq_200_l4_4870


namespace find_f_minus3_and_f_2009_l4_4707

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Conditions
axiom h1 : is_odd f
axiom h2 : f 1 = 2
axiom h3 : ∀ x : ℝ, f (x + 6) = f x + f 3

-- Questions
theorem find_f_minus3_and_f_2009 : f (-3) = 0 ∧ f 2009 = -2 :=
by 
  sorry

end find_f_minus3_and_f_2009_l4_4707


namespace lucas_change_l4_4254

-- Define the costs of items and the initial amount.
def initial_amount : ℝ := 20.00
def cost_avocados : ℝ := 1.50 + 2.25 + 3.00
def cost_water : ℝ := 2 * 1.75
def cost_apples : ℝ := 4 * 0.75

-- Define the total cost.
def total_cost : ℝ := cost_avocados + cost_water + cost_apples

-- Define the expected change.
def expected_change : ℝ := initial_amount - total_cost

-- The proposition (statement) we want to prove.
theorem lucas_change : expected_change = 6.75 :=
by
  sorry -- Proof to be completed.

end lucas_change_l4_4254


namespace July_husband_age_l4_4918

namespace AgeProof

variable (HannahAge JulyAge HusbandAge : ℕ)

def double_age_condition (hannah_age : ℕ) (july_age : ℕ) : Prop :=
  hannah_age = 2 * july_age

def twenty_years_later (current_age : ℕ) : ℕ :=
  current_age + 20

def two_years_older (age : ℕ) : ℕ :=
  age + 2

theorem July_husband_age :
  ∃ (hannah_age july_age : ℕ), double_age_condition hannah_age july_age ∧
    twenty_years_later hannah_age = 26 ∧
    twenty_years_later july_age = 23 ∧
    two_years_older (twenty_years_later july_age) = 25 :=
by
  sorry
end AgeProof

end July_husband_age_l4_4918


namespace compute_expression_l4_4710

theorem compute_expression : (46 + 15)^2 - (46 - 15)^2 = 2760 :=
by
  sorry

end compute_expression_l4_4710


namespace least_common_multiple_xyz_l4_4897

theorem least_common_multiple_xyz (x y z : ℕ) 
  (h1 : Nat.lcm x y = 18) 
  (h2 : Nat.lcm y z = 20) : 
  Nat.lcm x z = 90 := 
sorry

end least_common_multiple_xyz_l4_4897


namespace min_gennadies_l4_4908

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l4_4908


namespace area_new_rectangle_l4_4709

-- Define the given rectangle's dimensions
def a : ℕ := 3
def b : ℕ := 4

-- Define the diagonal of the given rectangle
def d : ℕ := Nat.sqrt (a^2 + b^2)

-- Define the new rectangle's dimensions
def length_new : ℕ := d + a
def breadth_new : ℕ := d - b

-- The target area of the new rectangle
def area_new : ℕ := length_new * breadth_new

-- Prove that the area of the new rectangle is 8 square units
theorem area_new_rectangle (h : d = 5) : area_new = 8 := by
  -- Indicate that proof steps are not provided
  sorry

end area_new_rectangle_l4_4709


namespace find_pq_cube_l4_4502

theorem find_pq_cube (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 3) : (p + q) ^ 3 = 125 := 
by
  -- This is where the proof would go
  sorry

end find_pq_cube_l4_4502


namespace parity_of_expression_l4_4453

theorem parity_of_expression {a b c : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : 0 < c) :
  ∃ k : ℕ, 3 ^ a + (b - 1) ^ 2 * c = 2 * k + 1 :=
by
  sorry

end parity_of_expression_l4_4453


namespace draw_probability_l4_4089

theorem draw_probability (P_A_win : ℝ) (P_A_not_lose : ℝ) (h1 : P_A_win = 0.3) (h2 : P_A_not_lose = 0.8) : 
  ∃ P_draw : ℝ, P_draw = 0.5 := 
by
  sorry

end draw_probability_l4_4089


namespace first_pump_rate_is_180_l4_4227

-- Define the known conditions
variables (R : ℕ) -- The rate of the first pump in gallons per hour
def second_pump_rate : ℕ := 250 -- The rate of the second pump in gallons per hour
def second_pump_time : ℕ := 35 / 10 -- 3.5 hours represented as a fraction
def total_pump_time : ℕ := 60 / 10 -- 6 hours represented as a fraction
def total_volume : ℕ := 1325 -- Total volume pumped by both pumps in gallons

-- Define derived conditions from the problem
def second_pump_volume : ℕ := second_pump_rate * second_pump_time -- Volume pumped by the second pump
def first_pump_volume : ℕ := total_volume - second_pump_volume -- Volume pumped by the first pump
def first_pump_time : ℕ := total_pump_time - second_pump_time -- Time the first pump was used

-- The main theorem to prove that the rate of the first pump is 180 gallons per hour
theorem first_pump_rate_is_180 : R = 180 :=
by
  -- The proof would go here
  sorry

end first_pump_rate_is_180_l4_4227


namespace time_for_A_to_complete_race_l4_4232

theorem time_for_A_to_complete_race
  (V_A V_B : ℝ) (T_A : ℝ)
  (h1 : V_B = 975 / T_A) (h2 : V_B = 2.5) :
  T_A = 390 :=
by
  sorry

end time_for_A_to_complete_race_l4_4232


namespace time_reduced_fraction_l4_4182

theorem time_reduced_fraction
  (T : ℝ)
  (V : ℝ)
  (hV : V = 42)
  (D : ℝ)
  (hD_1 : D = V * T)
  (V' : ℝ)
  (hV' : V' = V + 21)
  (T' : ℝ)
  (hD_2 : D = V' * T') :
  (T - T') / T = 1 / 3 :=
by
  -- Proof omitted
  sorry

end time_reduced_fraction_l4_4182


namespace angle_sum_155_l4_4335

theorem angle_sum_155
  (AB AC DE DF : ℝ)
  (h1 : AB = AC)
  (h2 : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h3 : angle_BAC = 20)
  (h4 : angle_EDF = 30) :
  ∃ (angle_DAC angle_ADE : ℝ), angle_DAC + angle_ADE = 155 :=
by
  sorry

end angle_sum_155_l4_4335


namespace arithmetic_sequence_y_value_l4_4759

theorem arithmetic_sequence_y_value (y : ℚ) :
  ∃ y : ℚ, 
    (y - 2) - (2/3) = (4 * y - 1) - (y - 2) → 
    y = 11/6 := by
  sorry

end arithmetic_sequence_y_value_l4_4759


namespace find_f_prime_at_two_l4_4477

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l4_4477


namespace temperature_on_tuesday_l4_4511

variable (T W Th F : ℕ)

-- Conditions
def cond1 : Prop := (T + W + Th) / 3 = 32
def cond2 : Prop := (W + Th + F) / 3 = 34
def cond3 : Prop := F = 44

-- Theorem statement
theorem temperature_on_tuesday : cond1 T W Th → cond2 W Th F → cond3 F → T = 38 :=
by
  sorry

end temperature_on_tuesday_l4_4511


namespace megan_carrots_second_day_l4_4349

theorem megan_carrots_second_day : 
  ∀ (initial : ℕ) (thrown : ℕ) (total : ℕ) (second_day : ℕ),
  initial = 19 →
  thrown = 4 →
  total = 61 →
  second_day = (total - (initial - thrown)) →
  second_day = 46 :=
by
  intros initial thrown total second_day h_initial h_thrown h_total h_second_day
  rw [h_initial, h_thrown, h_total] at h_second_day
  sorry

end megan_carrots_second_day_l4_4349


namespace fixed_point_for_all_k_l4_4304

theorem fixed_point_for_all_k (k : ℝ) : (5, 225) ∈ { p : ℝ × ℝ | ∃ k : ℝ, p.snd = 9 * p.fst^2 + k * p.fst - 5 * k } :=
by
  sorry

end fixed_point_for_all_k_l4_4304


namespace exists_ordering_no_arithmetic_progression_l4_4440

theorem exists_ordering_no_arithmetic_progression (m : ℕ) (hm : 0 < m) :
  ∃ (a : Fin (2^m) → ℕ), (∀ i j k : Fin (2^m), i < j → j < k → a j - a i ≠ a k - a j) := sorry

end exists_ordering_no_arithmetic_progression_l4_4440


namespace increasing_sequences_count_with_modulo_l4_4630

theorem increasing_sequences_count_with_modulo : 
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sequences_count % mod_value = k :=
by
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sorry

end increasing_sequences_count_with_modulo_l4_4630


namespace sum_of_arithmetic_sequence_l4_4803

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : S n = n * (a 1 + a n) / 2)
  (h3 : a 2 + a 5 + a 11 = 6) :
  S 11 = 22 :=
sorry

end sum_of_arithmetic_sequence_l4_4803


namespace simplest_fraction_sum_l4_4714

theorem simplest_fraction_sum (c d : ℕ) (h1 : 0.325 = (c:ℚ)/d) (h2 : Int.gcd c d = 1) : c + d = 53 :=
by sorry

end simplest_fraction_sum_l4_4714


namespace lexi_laps_l4_4384

theorem lexi_laps (total_distance lap_distance : ℝ) (h1 : total_distance = 3.25) (h2 : lap_distance = 0.25) :
  total_distance / lap_distance = 13 :=
by
  sorry

end lexi_laps_l4_4384


namespace wait_time_probability_l4_4409

theorem wait_time_probability
  (P_B1_8_00 : ℚ)
  (P_B1_8_20 : ℚ)
  (P_B1_8_40 : ℚ)
  (P_B2_9_00 : ℚ)
  (P_B2_9_20 : ℚ)
  (P_B2_9_40 : ℚ)
  (h_independent : true)
  (h_employee_arrival : true)
  (h_P_B1 : P_B1_8_00 = 1/4 ∧ P_B1_8_20 = 1/2 ∧ P_B1_8_40 = 1/4)
  (h_P_B2 : P_B2_9_00 = 1/4 ∧ P_B2_9_20 = 1/2 ∧ P_B2_9_40 = 1/4) :
  (P_B1_8_00 * P_B2_9_20 + P_B1_8_00 * P_B2_9_40 = 3/16) :=
sorry

end wait_time_probability_l4_4409


namespace permutations_five_three_eq_sixty_l4_4320

theorem permutations_five_three_eq_sixty : (Nat.factorial 5) / (Nat.factorial (5 - 3)) = 60 := 
by
  sorry

end permutations_five_three_eq_sixty_l4_4320


namespace series_sum_eq_l4_4837

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * ↑n + 3) / ((4 * ↑n + 1)^2 * (4 * ↑n + 5)^2)

theorem series_sum_eq :
  (∑' n, series_term n) = 1 / 800 :=
by
  sorry

end series_sum_eq_l4_4837


namespace tan_theta_point_l4_4974

open Real

theorem tan_theta_point :
  ∀ θ : ℝ,
  ∃ (x y : ℝ), x = -sqrt 3 / 2 ∧ y = 1 / 2 ∧ (tan θ) = y / x → (tan θ) = -sqrt 3 / 3 :=
by
  sorry

end tan_theta_point_l4_4974


namespace sum_of_not_visible_faces_l4_4943

-- Define the sum of the numbers on the faces of one die
def die_sum : ℕ := 21

-- List of visible numbers on the dice
def visible_faces_sum : ℕ := 4 + 3 + 2 + 5 + 1 + 3 + 1

-- Define the total sum of the numbers on the faces of three dice
def total_sum : ℕ := die_sum * 3

-- Statement to prove the sum of not-visible faces equals 44
theorem sum_of_not_visible_faces : 
  total_sum - visible_faces_sum = 44 :=
sorry

end sum_of_not_visible_faces_l4_4943


namespace larger_number_of_ratio_and_lcm_l4_4579

theorem larger_number_of_ratio_and_lcm (x : ℕ) (h1 : (2 * x) % (5 * x) = 160) : (5 * x) = 160 := by
  sorry

end larger_number_of_ratio_and_lcm_l4_4579


namespace initial_shed_bales_zero_l4_4300

def bales_in_barn_initial : ℕ := 47
def bales_added_by_benny : ℕ := 35
def bales_in_barn_total : ℕ := 82

theorem initial_shed_bales_zero (b_shed : ℕ) :
  bales_in_barn_initial + bales_added_by_benny = bales_in_barn_total → b_shed = 0 :=
by
  intro h
  sorry

end initial_shed_bales_zero_l4_4300


namespace uphill_distance_is_100_l4_4626

def speed_uphill := 30  -- km/hr
def speed_downhill := 60  -- km/hr
def distance_downhill := 50  -- km
def avg_speed := 36  -- km/hr

-- Let d be the distance traveled uphill
variable (d : ℕ)

-- total distance is d + 50 km
def total_distance := d + distance_downhill

-- total time is (time uphill) + (time downhill)
def total_time := (d / speed_uphill) + (distance_downhill / speed_downhill)

theorem uphill_distance_is_100 (d : ℕ) (h : avg_speed = total_distance / total_time) : d = 100 :=
by
  sorry  -- proof is omitted

end uphill_distance_is_100_l4_4626


namespace part1_part2_l4_4921

-- Define the initial conditions and the given inequality.
def condition1 (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def condition2 (m : ℝ) (x : ℝ) : Prop := x = (1/2)^(m - 1) ∧ 1 < m ∧ m < 2

-- Definitions of the correct ranges
def range_x (x : ℝ) : Prop := 1/2 < x ∧ x < 3/4
def range_a (a : ℝ) : Prop := 1/3 ≤ a ∧ a ≤ 1/2

-- Mathematical equivalent proof problem
theorem part1 {x : ℝ} (h1 : condition1 x (1/4)) (h2 : ∃ (m : ℝ), condition2 m x) : range_x x :=
sorry

theorem part2 {a : ℝ} (h : ∀ x : ℝ, (1/2 < x ∧ x < 1) → condition1 x a) : range_a a :=
sorry

end part1_part2_l4_4921


namespace compute_infinite_series_l4_4670

noncomputable def infinite_series (c d : ℝ) (hcd : c > d) : ℝ :=
  ∑' n, 1 / (((n - 1 : ℝ) * c - (n - 2 : ℝ) * d) * (n * c - (n - 1 : ℝ) * d))

theorem compute_infinite_series (c d : ℝ) (hcd : c > d) :
  infinite_series c d hcd = 1 / ((c - d) * d) :=
by
  sorry

end compute_infinite_series_l4_4670


namespace remainder_of_polynomial_l4_4663

theorem remainder_of_polynomial (x : ℕ) :
  (x + 1) ^ 2021 % (x ^ 2 + x + 1) = 1 + x ^ 2 := 
by
  sorry

end remainder_of_polynomial_l4_4663


namespace sum_of_A_H_l4_4378

theorem sum_of_A_H (A B C D E F G H : ℝ) (h1 : C = 10) 
  (h2 : A + B + C = 40) (h3 : B + C + D = 40) (h4 : C + D + E = 40) 
  (h5 : D + E + F = 40) (h6 : E + F + G = 40) (h7 : F + G + H = 40) :
  A + H = 30 := 
sorry

end sum_of_A_H_l4_4378


namespace number_of_connections_l4_4492

theorem number_of_connections (n m : ℕ) (h1 : n = 30) (h2 : m = 4) :
    (n * m) / 2 = 60 := by
  -- Since each switch is connected to 4 others,
  -- and each connection is counted twice, 
  -- the number of unique connections is 60.
  sorry

end number_of_connections_l4_4492


namespace inequality_areas_l4_4725

theorem inequality_areas (a b c α β γ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  a / α + b / β + c / γ ≥ 3 / 2 :=
by
  -- Insert the AM-GM inequality application and simplifications
  sorry

end inequality_areas_l4_4725


namespace minoxidil_percentage_l4_4621

-- Define the conditions
variable (x : ℝ) -- percentage of Minoxidil in the solution to add
def pharmacist_scenario (x : ℝ) : Prop :=
  let amt_2_percent_solution := 70 -- 70 ml of 2% solution
  let percent_in_2_percent := 0.02
  let amt_of_2_percent := percent_in_2_percent * amt_2_percent_solution
  let amt_added_solution := 35 -- 35 ml of solution to add
  let total_volume := amt_2_percent_solution + amt_added_solution -- 105 ml in total
  let desired_percent := 0.03
  let desired_amt := desired_percent * total_volume
  amt_of_2_percent + (x / 100) * amt_added_solution = desired_amt

-- Define the proof problem statement
theorem minoxidil_percentage : pharmacist_scenario 5 := by
  -- Proof goes here
  sorry

end minoxidil_percentage_l4_4621


namespace smallest_N_is_14_l4_4267

-- Definition of depicted number and cyclic arrangement
def depicted_number : Type := List (Fin 2) -- Depicted numbers are lists of digits (0 corresponds to 1, 1 corresponds to 2)

-- A condition representing the function that checks if a list contains all possible four-digit combinations
def contains_all_four_digit_combinations (arr: List (Fin 2)) : Prop :=
  ∀ (seq: List (Fin 2)), seq.length = 4 → seq ⊆ arr

-- The problem statement: find the smallest N where an arrangement contains all four-digit combinations
def smallest_N (N: Nat) (arr: List (Fin 2)) : Prop :=
  N = arr.length ∧ contains_all_four_digit_combinations arr

theorem smallest_N_is_14 : ∃ (N : Nat) (arr: List (Fin 2)), smallest_N N arr ∧ N = 14 :=
by
  -- Placeholder for the proof
  sorry

end smallest_N_is_14_l4_4267


namespace multiple_of_Mel_weight_l4_4937

/-- Given that Brenda weighs 10 pounds more than a certain multiple of Mel's weight,
    and given that Brenda weighs 220 pounds and Mel's weight is 70 pounds,
    show that the multiple is 3. -/
theorem multiple_of_Mel_weight 
    (Brenda_weight Mel_weight certain_multiple : ℝ) 
    (h1 : Brenda_weight = Mel_weight * certain_multiple + 10)
    (h2 : Brenda_weight = 220)
    (h3 : Mel_weight = 70) :
  certain_multiple = 3 :=
by 
  sorry

end multiple_of_Mel_weight_l4_4937


namespace third_number_is_32_l4_4517

theorem third_number_is_32 (A B C : ℕ) 
  (hA : A = 24) (hB : B = 36) 
  (hHCF : Nat.gcd (Nat.gcd A B) C = 32) 
  (hLCM : Nat.lcm (Nat.lcm A B) C = 1248) : 
  C = 32 := 
sorry

end third_number_is_32_l4_4517


namespace find_coefficient_of_x_l4_4122

theorem find_coefficient_of_x :
  ∃ a : ℚ, ∀ (x y : ℚ),
  (x + y = 19) ∧ (x + 3 * y = 1) ∧ (2 * x + y = 5) →
  (a * x + y = 19) ∧ (a = 7) :=
by
  sorry

end find_coefficient_of_x_l4_4122


namespace largest_angle_of_scalene_triangle_l4_4779

-- Define the problem statement in Lean
theorem largest_angle_of_scalene_triangle (x : ℝ) (hx : x = 30) : 3 * x = 90 :=
by {
  -- Given that the smallest angle is x and x = 30 degrees
  sorry
}

end largest_angle_of_scalene_triangle_l4_4779


namespace sin_2alpha_pos_if_tan_alpha_pos_l4_4567

theorem sin_2alpha_pos_if_tan_alpha_pos (α : ℝ) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end sin_2alpha_pos_if_tan_alpha_pos_l4_4567


namespace number_of_ways_to_form_divisible_number_l4_4011

def valid_digits : List ℕ := [0, 2, 4, 7, 8, 9]

def is_divisible_by_4 (d1 d2 : ℕ) : Prop :=
  (d1 * 10 + d2) % 4 = 0

def is_divisible_by_3 (sum_of_digits : ℕ) : Prop :=
  sum_of_digits % 3 = 0

def replace_asterisks_to_form_divisible_number : Prop :=
  ∃ (a1 a2 a3 a4 a5 l : ℕ), a1 ∈ valid_digits ∧ a2 ∈ valid_digits ∧ a3 ∈ valid_digits ∧ a4 ∈ valid_digits ∧ a5 ∈ valid_digits ∧
  l ∈ [0, 2, 4, 8] ∧
  is_divisible_by_4 0 l ∧
  is_divisible_by_3 (11 + a1 + a2 + a3 + a4 + a5) ∧
  (4 * 324 = 1296)

theorem number_of_ways_to_form_divisible_number :
  replace_asterisks_to_form_divisible_number :=
  sorry

end number_of_ways_to_form_divisible_number_l4_4011


namespace find_other_endpoint_of_diameter_l4_4850

noncomputable def circle_center : (ℝ × ℝ) := (4, -2)
noncomputable def one_endpoint_of_diameter : (ℝ × ℝ) := (7, 5)
noncomputable def other_endpoint_of_diameter : (ℝ × ℝ) := (1, -9)

theorem find_other_endpoint_of_diameter :
  let (cx, cy) := circle_center
  let (x1, y1) := one_endpoint_of_diameter
  let (x2, y2) := other_endpoint_of_diameter
  (x2, y2) = (2 * cx - x1, 2 * cy - y1) :=
by
  sorry

end find_other_endpoint_of_diameter_l4_4850


namespace amare_needs_more_fabric_l4_4403

theorem amare_needs_more_fabric :
  let first_two_dresses_in_feet := 2 * 5.5 * 3
  let next_two_dresses_in_feet := 2 * 6 * 3
  let last_two_dresses_in_feet := 2 * 6.5 * 3
  let total_fabric_needed := first_two_dresses_in_feet + next_two_dresses_in_feet + last_two_dresses_in_feet
  let fabric_amare_has := 10
  total_fabric_needed - fabric_amare_has = 98 :=
by {
  sorry
}

end amare_needs_more_fabric_l4_4403


namespace wire_length_l4_4473

theorem wire_length (r_sphere r_cylinder : ℝ) (V_sphere_eq_V_cylinder : (4/3) * π * r_sphere^3 = π * r_cylinder^2 * 144) :
  r_sphere = 12 → r_cylinder = 4 → 144 = 144 := sorry

end wire_length_l4_4473


namespace proof_problem_l4_4308

variable {a b c d : ℝ}
variable {x1 y1 x2 y2 x3 y3 x4 y4 : ℝ}

-- Assume the conditions
variable (habcd_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
variable (unity_circle : x1^2 + y1^2 = 1 ∧ x2^2 + y2^2 = 1 ∧ x3^2 + y3^2 = 1 ∧ x4^2 + y4^2 = 1)
variable (unit_sum : a * b + c * d = 1)

-- Statement to prove
theorem proof_problem :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2
    ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
  sorry

end proof_problem_l4_4308


namespace remainder_4873_div_29_l4_4248

theorem remainder_4873_div_29 : 4873 % 29 = 1 := 
by sorry

end remainder_4873_div_29_l4_4248


namespace probability_of_all_red_is_correct_l4_4481

noncomputable def probability_of_all_red_drawn : ℚ :=
  let total_ways := (Nat.choose 10 5)   -- Total ways to choose 5 balls from 10
  let red_ways := (Nat.choose 5 5)      -- Ways to choose all 5 red balls
  red_ways / total_ways

theorem probability_of_all_red_is_correct :
  probability_of_all_red_drawn = 1 / 252 := by
  sorry

end probability_of_all_red_is_correct_l4_4481


namespace insurance_compensation_correct_l4_4427

def actual_damage : ℝ := 300000
def deductible_percent : ℝ := 0.01
def deductible_amount : ℝ := deductible_percent * actual_damage
def insurance_compensation : ℝ := actual_damage - deductible_amount

theorem insurance_compensation_correct : insurance_compensation = 297000 :=
by
  -- To be proved
  sorry

end insurance_compensation_correct_l4_4427


namespace eggs_volume_correct_l4_4649

def raw_spinach_volume : ℕ := 40
def cooking_reduction_ratio : ℚ := 0.20
def cream_cheese_volume : ℕ := 6
def total_quiche_volume : ℕ := 18
def cooked_spinach_volume := (raw_spinach_volume : ℚ) * cooking_reduction_ratio
def combined_spinach_and_cream_cheese_volume := cooked_spinach_volume + (cream_cheese_volume : ℚ)
def eggs_volume := (total_quiche_volume : ℚ) - combined_spinach_and_cream_cheese_volume

theorem eggs_volume_correct : eggs_volume = 4 := by
  sorry

end eggs_volume_correct_l4_4649


namespace harmonic_mean_closest_to_2_l4_4950

theorem harmonic_mean_closest_to_2 (a : ℝ) (b : ℝ) (h₁ : a = 1) (h₂ : b = 4032) : 
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  rw [h₁, h₂]
  -- The rest of the proof follows from here, skipped with sorry
  sorry

end harmonic_mean_closest_to_2_l4_4950


namespace observations_number_l4_4967

theorem observations_number 
  (mean : ℚ)
  (wrong_obs corrected_obs : ℚ)
  (new_mean : ℚ)
  (n : ℚ)
  (initial_mean : mean = 36)
  (wrong_obs_taken : wrong_obs = 23)
  (corrected_obs_value : corrected_obs = 34)
  (corrected_mean : new_mean = 36.5) :
  (n * mean + (corrected_obs - wrong_obs) = n * new_mean) → 
  n = 22 :=
by
  sorry

end observations_number_l4_4967


namespace bryce_received_raisins_l4_4996

theorem bryce_received_raisins
  (C B : ℕ)
  (h1 : B = C + 8)
  (h2 : C = B / 3) :
  B = 12 :=
by sorry

end bryce_received_raisins_l4_4996


namespace value_of_power_l4_4966

theorem value_of_power (a : ℝ) (m n k : ℕ) (h1 : a ^ m = 2) (h2 : a ^ n = 4) (h3 : a ^ k = 32) : 
  a ^ (3 * m + 2 * n - k) = 4 := 
by sorry

end value_of_power_l4_4966


namespace largest_n_for_factorable_poly_l4_4755

theorem largest_n_for_factorable_poly :
  ∃ n : ℤ, (∀ A B : ℤ, (3 * B + A = n) ∧ (A * B = 72) → (A = 1 ∧ B = 72 ∧ n = 217)) ∧
           (∀ A B : ℤ, A * B = 72 → 3 * B + A ≤ 217) :=
by
  sorry

end largest_n_for_factorable_poly_l4_4755


namespace depth_of_sand_l4_4454

theorem depth_of_sand (h : ℝ) (fraction_above_sand : ℝ) :
  h = 9000 → fraction_above_sand = 1/9 → depth = 342 :=
by
  -- height of the pyramid
  let height := 9000
  -- ratio of submerged height to the total height
  let ratio := (8 / 9)^(1 / 3)
  -- height of the submerged part
  let submerged_height := height * ratio
  -- depth of the sand
  let depth := height - submerged_height
  sorry

end depth_of_sand_l4_4454


namespace rectangle_area_ratio_l4_4310

-- Define points in complex plane or as tuples (for 2D geometry)
structure Point where
  x : ℝ
  y : ℝ

-- Rectangle vertices
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}
def C : Point := {x := 1, y := 2}
def D : Point := {x := 0, y := 2}

-- Centroid of triangle BCD
def E : Point := {x := 1.0, y := 1.333}

-- Point F such that DF = 1/4 * DA
def F : Point := {x := 1.5, y := 0}

-- Calculate areas of triangles and quadrilateral
noncomputable def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

noncomputable def area_rectangle : ℝ :=
  2.0  -- Area of rectangle ABCD (1 * 2)

noncomputable def problem_statement : Prop :=
  let area_DFE := area_triangle D F E
  let area_ABEF := area_rectangle - area_triangle A B F - area_triangle D A F
  area_DFE / area_ABEF = 1 / 10.5

theorem rectangle_area_ratio :
  problem_statement :=
by
  sorry

end rectangle_area_ratio_l4_4310


namespace symmetric_point_origin_l4_4836

theorem symmetric_point_origin (A : ℝ × ℝ) (A_sym : ℝ × ℝ) (h : A = (3, -2)) (h_sym : A_sym = (-A.1, -A.2)) : A_sym = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l4_4836


namespace complement_intersection_l4_4447

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x ∈ A | x ∈ B}

-- Define the complement of A_inter_B in U
def complement_U_A_inter_B : Set ℕ := {x ∈ U | x ∉ A_inter_B}

-- Prove that the complement of the intersection of A and B in U is {1, 4, 5}
theorem complement_intersection :
  complement_U_A_inter_B = {1, 4, 5} :=
by
  sorry

end complement_intersection_l4_4447


namespace find_a₈_l4_4620

noncomputable def a₃ : ℝ := -11 / 6
noncomputable def a₅ : ℝ := -13 / 7

theorem find_a₈ (h : ∃ d : ℝ, ∀ n : ℕ, (1 / (a₃ + 2)) + (n-2) * d = (1 / (a_n + 2)))
  : a_n = -32 / 17 := sorry

end find_a₈_l4_4620


namespace factorization_of_expression_l4_4187

theorem factorization_of_expression
  (a b c : ℝ)
  (expansion : (b+c)*(c+a)*(a+b) + abc = (a+b+c)*(ab+ac+bc)) : 
  ∃ (m l : ℝ), (m = 0 ∧ l = a + b + c ∧ 
  (b+c)*(c+a)*(a+b) + abc = m*(a^2 + b^2 + c^2) + l*(ab + ac + bc)) :=
by
  sorry

end factorization_of_expression_l4_4187


namespace mike_earnings_l4_4303

theorem mike_earnings (total_games non_working_games price_per_game : ℕ) 
  (h1 : total_games = 15) (h2 : non_working_games = 9) (h3 : price_per_game = 5) : 
  total_games - non_working_games * price_per_game = 30 :=
by
  rw [h1, h2, h3]
  show 15 - 9 * 5 = 30
  sorry

end mike_earnings_l4_4303


namespace max_sum_marks_l4_4107

theorem max_sum_marks (a b c : ℕ) (h1 : a + b + c = 2019) (h2 : a ≤ c + 2) : 
  2 * a + b ≤ 2021 :=
by {
  -- We'll skip the proof but formulate the statement following conditions strictly.
  sorry
}

end max_sum_marks_l4_4107


namespace trig_identity_l4_4468

theorem trig_identity :
  (4 * (1 / 2) - Real.sqrt 2 * (Real.sqrt 2 / 2) - Real.sqrt 3 * (Real.sqrt 3 / 3) + 2 * (Real.sqrt 3 / 2)) = Real.sqrt 3 :=
by sorry

end trig_identity_l4_4468


namespace boat_license_combinations_l4_4000

theorem boat_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let digit_positions := 5
  (letter_choices * (digit_choices ^ digit_positions)) = 300000 :=
  sorry

end boat_license_combinations_l4_4000


namespace simplify_and_evaluate_expr_find_ab_l4_4500

theorem simplify_and_evaluate_expr (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5 * y) * (-x - 5 * y) - (-x + 5 * y)^2 = -5.5 :=
by
  rw [hx, hy]
  sorry

theorem find_ab (a b : ℝ) (h : a^2 - 2 * a + b^2 + 4 * b + 5 = 0) :
  (a + b) ^ 2013 = -1 :=
by
  sorry

end simplify_and_evaluate_expr_find_ab_l4_4500


namespace Bridget_weight_is_correct_l4_4606

-- Definitions based on conditions
def Martha_weight : ℕ := 2
def weight_difference : ℕ := 37

-- Bridget's weight based on the conditions
def Bridget_weight : ℕ := Martha_weight + weight_difference

-- Proof problem: Prove that Bridget's weight is 39
theorem Bridget_weight_is_correct : Bridget_weight = 39 := by
  -- Proof goes here
  sorry

end Bridget_weight_is_correct_l4_4606


namespace cuboid_surface_area_l4_4976

-- Define the given conditions
def cuboid (a b c : ℝ) := 2 * (a + b + c)

-- Given areas of distinct sides
def area_face_1 : ℝ := 4
def area_face_2 : ℝ := 3
def area_face_3 : ℝ := 6

-- Prove the total surface area of the cuboid
theorem cuboid_surface_area : cuboid area_face_1 area_face_2 area_face_3 = 26 :=
by
  sorry

end cuboid_surface_area_l4_4976


namespace stream_speed_l4_4534

theorem stream_speed (v : ℝ) : (24 + v) = 168 / 6 → v = 4 :=
by
  intro h
  sorry

end stream_speed_l4_4534


namespace polynomial_divisibility_l4_4100

theorem polynomial_divisibility (n : ℕ) : (¬ n % 3 = 0) → (x ^ (2 * n) + x ^ n + 1) % (x ^ 2 + x + 1) = 0 :=
by
  sorry

end polynomial_divisibility_l4_4100


namespace ending_number_of_second_range_l4_4462

theorem ending_number_of_second_range :
  let avg100_400 := (100 + 400) / 2
  let avg_50_n := (50 + n) / 2
  avg100_400 = avg_50_n + 100 → n = 250 :=
by
  sorry

end ending_number_of_second_range_l4_4462


namespace blue_more_than_white_l4_4424

theorem blue_more_than_white :
  ∃ (B R : ℕ), (B > 16) ∧ (R = 2 * B) ∧ (B + R + 16 = 100) ∧ (B - 16 = 12) :=
sorry

end blue_more_than_white_l4_4424


namespace successive_product_4160_l4_4316

theorem successive_product_4160 (n : ℕ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end successive_product_4160_l4_4316


namespace concentration_after_dilution_l4_4656

-- Definitions and conditions
def initial_volume : ℝ := 5
def initial_concentration : ℝ := 0.06
def poured_out_volume : ℝ := 1
def added_water_volume : ℝ := 2

-- Theorem statement
theorem concentration_after_dilution : 
  (initial_volume * initial_concentration - poured_out_volume * initial_concentration) / 
  (initial_volume - poured_out_volume + added_water_volume) = 0.04 :=
by 
  sorry

end concentration_after_dilution_l4_4656


namespace nat_divides_power_difference_l4_4341

theorem nat_divides_power_difference (n : ℕ) : n ∣ 2 ^ (2 * n.factorial) - 2 ^ n.factorial := by
  sorry

end nat_divides_power_difference_l4_4341


namespace positive_when_x_negative_l4_4948

theorem positive_when_x_negative (x : ℝ) (h : x < 0) : (x / |x|)^2 > 0 := by
  sorry

end positive_when_x_negative_l4_4948


namespace smallest_possible_N_l4_4374

theorem smallest_possible_N (p q r s t : ℕ) (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0)
(h_sum : p + q + r + s + t = 2022) :
    ∃ N : ℕ, N = 506 ∧ N = max (p + q) (max (q + r) (max (r + s) (s + t))) :=
by
    sorry

end smallest_possible_N_l4_4374


namespace hypotenuse_length_l4_4603

theorem hypotenuse_length {a b c : ℝ} (h1 : a = 3) (h2 : b = 4) (h3 : c ^ 2 = a ^ 2 + b ^ 2) : c = 5 :=
by
  sorry

end hypotenuse_length_l4_4603


namespace front_view_heights_l4_4047

-- Define conditions
def column1 := [4, 2]
def column2 := [3, 0, 3]
def column3 := [1, 5]

-- Define a function to get the max height in each column
def max_height (col : List Nat) : Nat :=
  col.foldr Nat.max 0

-- Define the statement to prove the frontal view heights
theorem front_view_heights : 
  max_height column1 = 4 ∧ 
  max_height column2 = 3 ∧ 
  max_height column3 = 5 :=
by 
  sorry

end front_view_heights_l4_4047


namespace gcd_3_1200_1_3_1210_1_l4_4529

theorem gcd_3_1200_1_3_1210_1 : 
  Int.gcd (3^1200 - 1) (3^1210 - 1) = 59048 := 
by 
  sorry

end gcd_3_1200_1_3_1210_1_l4_4529


namespace ball_radius_l4_4762

theorem ball_radius (x r : ℝ) 
  (h1 : (15 : ℝ) ^ 2 + x ^ 2 = r ^ 2) 
  (h2 : x + 12 = r) : 
  r = 15.375 := 
sorry

end ball_radius_l4_4762


namespace area_of_garden_l4_4363

-- Define the garden properties
variables {l w : ℕ}

-- Calculate length from the condition of walking length 30 times
def length_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Calculate perimeter from the condition of walking perimeter 12 times
def perimeter_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Define the proof statement
theorem area_of_garden (total_distance : ℕ) (times_length_walk : ℕ) (times_perimeter_walk : ℕ)
  (h1 : length_of_garden total_distance times_length_walk = l)
  (h2 : perimeter_of_garden total_distance (2 * times_perimeter_walk) = 2 * (l + w)) :
  l * w = 400 := 
sorry

end area_of_garden_l4_4363


namespace price_of_first_metal_l4_4590

theorem price_of_first_metal (x : ℝ) 
  (h1 : (x + 96) / 2 = 82) : 
  x = 68 :=
by sorry

end price_of_first_metal_l4_4590


namespace max_tickets_l4_4566

theorem max_tickets (n : ℕ) (H : 15 * n ≤ 120) : n ≤ 8 :=
by sorry

end max_tickets_l4_4566


namespace point_in_second_quadrant_l4_4855

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end point_in_second_quadrant_l4_4855


namespace increasing_interval_l4_4309

-- Define the function f(x) = x^2 + 2*(a - 1)*x
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*(a - 1)*x

-- Define the condition for f(x) being increasing on [4, +∞)
def is_increasing_on_interval (a : ℝ) : Prop := 
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → 
    f x a ≤ f y a

-- Define the main theorem that we need to prove
theorem increasing_interval (a : ℝ) (h : is_increasing_on_interval a) : -3 ≤ a :=
by 
  sorry -- proof is required, but omitted as per the instruction.

end increasing_interval_l4_4309


namespace largest_number_among_options_l4_4728

theorem largest_number_among_options :
  let A := 8.12366
  let B := 8.1236666666666 -- Repeating decimal 8.123\overline{6}
  let C := 8.1236363636363 -- Repeating decimal 8.12\overline{36}
  let D := 8.1236236236236 -- Repeating decimal 8.1\overline{236}
  let E := 8.1236123612361 -- Repeating decimal 8.\overline{1236}
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  let A := 8.12366
  let B := 8.12366666666666
  let C := 8.12363636363636
  let D := 8.12362362362362
  let E := 8.12361236123612
  sorry

end largest_number_among_options_l4_4728


namespace common_difference_arith_seq_l4_4578

theorem common_difference_arith_seq (a : ℕ → ℝ) (d : ℝ)
    (h₀ : a 1 + a 5 = 10)
    (h₁ : a 4 = 7)
    (h₂ : ∀ n, a (n + 1) = a n + d) : 
    d = 2 := by
  sorry

end common_difference_arith_seq_l4_4578


namespace sausage_more_than_pepperoni_l4_4810

noncomputable def pieces_of_meat_per_slice : ℕ := 22
noncomputable def slices : ℕ := 6
noncomputable def total_pieces_of_meat : ℕ := pieces_of_meat_per_slice * slices

noncomputable def pieces_of_pepperoni : ℕ := 30
noncomputable def pieces_of_ham : ℕ := 2 * pieces_of_pepperoni

noncomputable def total_pieces_of_meat_without_sausage : ℕ := pieces_of_pepperoni + pieces_of_ham
noncomputable def pieces_of_sausage : ℕ := total_pieces_of_meat - total_pieces_of_meat_without_sausage

theorem sausage_more_than_pepperoni : (pieces_of_sausage - pieces_of_pepperoni) = 12 := by
  sorry

end sausage_more_than_pepperoni_l4_4810


namespace tan_sum_to_expression_l4_4831

theorem tan_sum_to_expression (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 :=
by 
  sorry

end tan_sum_to_expression_l4_4831


namespace equation_pattern_l4_4684

theorem equation_pattern (n : ℕ) (h : n = 999999) : n^2 = (n + 1) * (n - 1) + 1 :=
by
  sorry

end equation_pattern_l4_4684


namespace rectangle_width_l4_4329

theorem rectangle_width (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 + y^2 = 25) : y = 3 := 
by 
  sorry

end rectangle_width_l4_4329


namespace puzzle_piece_total_l4_4006

theorem puzzle_piece_total :
  let p1 := 1000
  let p2 := p1 + 0.30 * p1
  let p3 := 2 * p2
  let p4 := (p1 + p3) + 0.50 * (p1 + p3)
  let p5 := 3 * p4
  let p6 := p1 + p2 + p3 + p4 + p5
  p1 + p2 + p3 + p4 + p5 + p6 = 55000
:= sorry

end puzzle_piece_total_l4_4006


namespace distance_between_5th_and_23rd_red_light_l4_4051

theorem distance_between_5th_and_23rd_red_light :
  let inch_to_feet (inches : ℕ) : ℝ := inches / 12.0
  let distance_in_inches := 40 * 8
  inch_to_feet distance_in_inches = 26.67 :=
by
  sorry

end distance_between_5th_and_23rd_red_light_l4_4051


namespace point_K_outside_hexagon_and_length_KC_l4_4334

theorem point_K_outside_hexagon_and_length_KC :
    ∀ (A B C K : ℝ × ℝ),
    A = (0, 0) →
    B = (3, 0) →
    C = (3 / 2, (3 * Real.sqrt 3) / 2) →
    K = (15 / 2, - (3 * Real.sqrt 3) / 2) →
    (¬ (0 ≤ K.1 ∧ K.1 ≤ 3 ∧ 0 ≤ K.2 ∧ K.2 ≤ 3 * Real.sqrt 3)) ∧
    Real.sqrt ((K.1 - C.1) ^ 2 + (K.2 - C.2) ^ 2) = 3 * Real.sqrt 7 :=
by
  intros A B C K hA hB hC hK
  sorry

end point_K_outside_hexagon_and_length_KC_l4_4334


namespace modulus_of_z_l4_4476

open Complex -- Open the Complex number namespace

-- Define the given condition as a hypothesis
def condition (z : ℂ) : Prop := (1 + I) * z = 3 + I

-- Statement of the theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = Real.sqrt 5 :=
sorry

end modulus_of_z_l4_4476


namespace find_k_inverse_proportion_l4_4785

theorem find_k_inverse_proportion :
  ∃ k : ℝ, (∀ x y : ℝ, y = (k + 1) / x → (x = 1 ∧ y = -2) → k = -3) :=
by
  sorry

end find_k_inverse_proportion_l4_4785


namespace Mina_has_2_25_cent_coins_l4_4144

def MinaCoinProblem : Prop :=
  ∃ (x y z : ℕ), -- number of 5-cent, 10-cent, and 25-cent coins
  x + y + z = 15 ∧
  (74 - 4 * x - 3 * y = 30) ∧ -- corresponds to 30 different values can be obtained
  z = 2

theorem Mina_has_2_25_cent_coins : MinaCoinProblem :=
by 
  sorry

end Mina_has_2_25_cent_coins_l4_4144


namespace average_age_in_club_l4_4287

theorem average_age_in_club :
  let women_avg_age := 32
  let men_avg_age := 38
  let children_avg_age := 10
  let women_count := 12
  let men_count := 18
  let children_count := 10
  let total_ages := (women_avg_age * women_count) + (men_avg_age * men_count) + (children_avg_age * children_count)
  let total_people := women_count + men_count + children_count
  let overall_avg_age := (total_ages : ℝ) / (total_people : ℝ)
  overall_avg_age = 29.2 := by
  sorry

end average_age_in_club_l4_4287


namespace sum_of_cubes_l4_4343

theorem sum_of_cubes (x y : ℝ) (h₁ : x + y = -1) (h₂ : x * y = -1) : x^3 + y^3 = -4 := by
  sorry

end sum_of_cubes_l4_4343


namespace puppies_adopted_per_day_l4_4487

theorem puppies_adopted_per_day 
    (initial_puppies : ℕ) 
    (additional_puppies : ℕ) 
    (total_days : ℕ) 
    (total_puppies : ℕ)
    (H1 : initial_puppies = 5) 
    (H2 : additional_puppies = 35) 
    (H3 : total_days = 5) 
    (H4 : total_puppies = initial_puppies + additional_puppies) : 
    total_puppies / total_days = 8 := by
  sorry

end puppies_adopted_per_day_l4_4487


namespace molly_takes_180_minutes_more_l4_4041

noncomputable def xanthia_speed : ℕ := 120
noncomputable def molly_speed : ℕ := 60
noncomputable def first_book_pages : ℕ := 360

-- Time taken by Xanthia to read the first book in hours
noncomputable def xanthia_time_first_book : ℕ := first_book_pages / xanthia_speed

-- Time taken by Molly to read the first book in hours
noncomputable def molly_time_first_book : ℕ := first_book_pages / molly_speed

-- Difference in time taken to read the first book in minutes
noncomputable def time_diff_minutes : ℕ := (molly_time_first_book - xanthia_time_first_book) * 60

theorem molly_takes_180_minutes_more : time_diff_minutes = 180 := by
  sorry

end molly_takes_180_minutes_more_l4_4041


namespace g_of_g_of_g_of_20_l4_4467

def g (x : ℕ) : ℕ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_of_g_of_g_of_20 : g (g (g 20)) = 1 := by
  -- Proof steps would go here
  sorry

end g_of_g_of_g_of_20_l4_4467


namespace loaf_slices_l4_4756

theorem loaf_slices (S : ℕ) (T : ℕ) : 
  (S - 7 = 2 * T + 3) ∧ (S ≥ 20) → S = 20 :=
by
  sorry

end loaf_slices_l4_4756


namespace initial_cupcakes_baked_l4_4410

variable (toddAte := 21)       -- Todd ate 21 cupcakes.
variable (packages := 6)       -- She could make 6 packages.
variable (cupcakesPerPackage := 3) -- Each package contains 3 cupcakes.
variable (cupcakesLeft := packages * cupcakesPerPackage) -- Cupcakes left after Todd ate some.

theorem initial_cupcakes_baked : cupcakesLeft + toddAte = 39 :=
by
  -- Proof placeholder
  sorry

end initial_cupcakes_baked_l4_4410


namespace negation_of_exists_implies_forall_l4_4953

theorem negation_of_exists_implies_forall :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry

end negation_of_exists_implies_forall_l4_4953


namespace simple_interest_rate_l4_4259

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ) : 
  T = 6 → I = (7/6) * P - P → I = P * R * T / 100 → R = 100 / 36 :=
by
  intros T_eq I_eq simple_interest_eq
  sorry

end simple_interest_rate_l4_4259


namespace find_value_of_sum_of_squares_l4_4975

theorem find_value_of_sum_of_squares
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 6 / 5 := by
  sorry

end find_value_of_sum_of_squares_l4_4975


namespace notebook_pen_cost_correct_l4_4510

noncomputable def notebook_pen_cost : Prop :=
  ∃ (x y : ℝ), 
  3 * x + 2 * y = 7.40 ∧ 
  2 * x + 5 * y = 9.75 ∧ 
  (x + 3 * y) = 5.53

theorem notebook_pen_cost_correct : notebook_pen_cost :=
sorry

end notebook_pen_cost_correct_l4_4510


namespace A_wins_if_perfect_square_or_prime_l4_4856

theorem A_wins_if_perfect_square_or_prime (n : ℕ) (h_pos : 0 < n) : 
  (∃ A_wins : Bool, A_wins = true ↔ (∃ k : ℕ, n = k^2) ∨ (∃ p : ℕ, Nat.Prime p ∧ n = p)) :=
by
  sorry

end A_wins_if_perfect_square_or_prime_l4_4856


namespace max_value_x_1_minus_3x_is_1_over_12_l4_4469

open Real

noncomputable def max_value_of_x_1_minus_3x (x : ℝ) : ℝ :=
  x * (1 - 3 * x)

theorem max_value_x_1_minus_3x_is_1_over_12 :
  ∀ x : ℝ, 0 < x ∧ x < 1 / 3 → max_value_of_x_1_minus_3x x ≤ 1 / 12 :=
by
  intros x h
  sorry

end max_value_x_1_minus_3x_is_1_over_12_l4_4469


namespace balloons_floated_away_l4_4423

theorem balloons_floated_away (starting_balloons given_away grabbed_balloons final_balloons flattened_balloons : ℕ)
  (h1 : starting_balloons = 50)
  (h2 : given_away = 10)
  (h3 : grabbed_balloons = 11)
  (h4 : final_balloons = 39)
  : flattened_balloons = starting_balloons - given_away + grabbed_balloons - final_balloons → flattened_balloons = 12 :=
by
  sorry

end balloons_floated_away_l4_4423


namespace vanessa_recycled_correct_l4_4555

-- Define conditions as separate hypotheses
variable (weight_per_point : ℕ := 9)
variable (points_earned : ℕ := 4)
variable (friends_recycled : ℕ := 16)

-- Define the total weight recycled as points earned times the weight per point
def total_weight_recycled (points_earned weight_per_point : ℕ) : ℕ := points_earned * weight_per_point

-- Define the weight recycled by Vanessa
def vanessa_recycled (total_recycled friends_recycled : ℕ) : ℕ := total_recycled - friends_recycled

-- Main theorem statement
theorem vanessa_recycled_correct (weight_per_point points_earned friends_recycled : ℕ) 
    (hw : weight_per_point = 9) (hp : points_earned = 4) (hf : friends_recycled = 16) : 
    vanessa_recycled (total_weight_recycled points_earned weight_per_point) friends_recycled = 20 := 
by 
  sorry

end vanessa_recycled_correct_l4_4555


namespace number_of_people_is_ten_l4_4988

-- Define the total number of Skittles and the number of Skittles per person.
def total_skittles : ℕ := 20
def skittles_per_person : ℕ := 2

-- Define the number of people as the total Skittles divided by the Skittles per person.
def number_of_people : ℕ := total_skittles / skittles_per_person

-- Theorem stating that the number of people is 10.
theorem number_of_people_is_ten : number_of_people = 10 := sorry

end number_of_people_is_ten_l4_4988


namespace john_learns_vowels_in_fifteen_days_l4_4008

def days_to_learn_vowels (days_per_vowel : ℕ) (num_vowels : ℕ) : ℕ :=
  days_per_vowel * num_vowels

theorem john_learns_vowels_in_fifteen_days :
  days_to_learn_vowels 3 5 = 15 :=
by
  -- Proof goes here
  sorry

end john_learns_vowels_in_fifteen_days_l4_4008


namespace geometric_sequence_general_term_l4_4770

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ q : ℝ, q > 0 ∧ (∀ n, a (n + 1) = q * a n)

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_S3 : a 1 * (1 + (a 2 / a 1) + (a 3 / a 1)) = 21) 
  (h_condition : 2 * a 2 = a 3) :
  ∃ c : ℝ, c = 3 ∧ ∀ n, a n = 3 * 2^(n - 1) := sorry

end geometric_sequence_general_term_l4_4770


namespace boxes_given_to_brother_l4_4190

-- Definitions
def total_boxes : ℝ := 14.0
def pieces_per_box : ℝ := 6.0
def pieces_remaining : ℝ := 42.0

-- Theorem stating the problem
theorem boxes_given_to_brother : 
  (total_boxes * pieces_per_box - pieces_remaining) / pieces_per_box = 7.0 := 
by
  sorry

end boxes_given_to_brother_l4_4190


namespace f_3_equals_1000_l4_4324

-- Define the function property f(lg x) = x
axiom f : ℝ → ℝ
axiom lg : ℝ → ℝ -- log function
axiom f_property : ∀ x : ℝ, f (lg x) = x

-- Prove that f(3) = 10^3
theorem f_3_equals_1000 : f 3 = 10^3 :=
by 
  -- Sorry to skip the proof
  sorry

end f_3_equals_1000_l4_4324


namespace quadratic_coefficients_l4_4613

theorem quadratic_coefficients (b c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 3)) → 
  b = -4 ∧ c = -6 :=
by
  intro h
  -- The proof would go here, but we'll skip it.
  sorry

end quadratic_coefficients_l4_4613


namespace parabola_sum_l4_4619

-- Define the quadratic equation
noncomputable def quadratic_eq (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Given conditions
variables (a b c : ℝ)
variables (h1 : (∀ x y : ℝ, y = quadratic_eq a b c x → y = a * (x - 6)^2 - 2))
variables (h2 : quadratic_eq a b c 3 = 0)

-- Prove the sum a + b + c
theorem parabola_sum :
  a + b + c = 14 / 9 :=
sorry

end parabola_sum_l4_4619


namespace combined_cost_of_items_is_221_l4_4521

def wallet_cost : ℕ := 22
def purse_cost : ℕ := 4 * wallet_cost - 3
def shoes_cost : ℕ := wallet_cost + purse_cost + 7
def combined_cost : ℕ := wallet_cost + purse_cost + shoes_cost

theorem combined_cost_of_items_is_221 : combined_cost = 221 := by
  sorry

end combined_cost_of_items_is_221_l4_4521


namespace temperature_on_fourth_day_l4_4556

theorem temperature_on_fourth_day
  (t₁ t₂ t₃ : ℤ) 
  (avg : ℤ)
  (h₁ : t₁ = -36) 
  (h₂ : t₂ = 13) 
  (h₃ : t₃ = -10) 
  (h₄ : avg = -12) 
  : ∃ t₄ : ℤ, t₄ = -15 :=
by
  sorry

end temperature_on_fourth_day_l4_4556


namespace girls_exceed_boys_by_402_l4_4835

theorem girls_exceed_boys_by_402 : 
  let girls := 739
  let boys := 337
  girls - boys = 402 :=
by
  sorry

end girls_exceed_boys_by_402_l4_4835


namespace expand_product_l4_4544

theorem expand_product (x : ℝ) : (x + 2) * (x^2 + 3 * x + 4) = x^3 + 5 * x^2 + 10 * x + 8 := 
by
  sorry

end expand_product_l4_4544


namespace wendy_baked_29_cookies_l4_4802

variables (cupcakes : ℕ) (pastries_taken_home : ℕ) (pastries_sold : ℕ)

def total_initial_pastries (cupcakes pastries_taken_home pastries_sold : ℕ) : ℕ :=
  pastries_taken_home + pastries_sold

def cookies_baked (total_initial_pastries cupcakes : ℕ) : ℕ :=
  total_initial_pastries - cupcakes

theorem wendy_baked_29_cookies :
  cupcakes = 4 →
  pastries_taken_home = 24 →
  pastries_sold = 9 →
  cookies_baked (total_initial_pastries cupcakes pastries_taken_home pastries_sold) cupcakes = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end wendy_baked_29_cookies_l4_4802


namespace distinct_roots_of_quadratic_l4_4660

variable {a b : ℝ}
-- condition: a and b are distinct
variable (h_distinct: a ≠ b)

theorem distinct_roots_of_quadratic (a b : ℝ) (h_distinct : a ≠ b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x + a)*(x + b) = 2*x + a + b :=
by
  sorry

end distinct_roots_of_quadratic_l4_4660


namespace sin_cos_power_sum_l4_4416

theorem sin_cos_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 61 / 64 := 
by
  sorry

end sin_cos_power_sum_l4_4416


namespace overall_average_marks_l4_4640

theorem overall_average_marks
  (avg_A : ℝ) (n_A : ℕ) (avg_B : ℝ) (n_B : ℕ) (avg_C : ℝ) (n_C : ℕ)
  (h_avg_A : avg_A = 40) (h_n_A : n_A = 12)
  (h_avg_B : avg_B = 60) (h_n_B : n_B = 28)
  (h_avg_C : avg_C = 55) (h_n_C : n_C = 15) :
  ((n_A * avg_A) + (n_B * avg_B) + (n_C * avg_C)) / (n_A + n_B + n_C) = 54.27 := by
  sorry

end overall_average_marks_l4_4640


namespace nonagon_blue_quadrilateral_l4_4282

theorem nonagon_blue_quadrilateral :
  ∀ (vertices : Finset ℕ) (red blue : ℕ → ℕ → Prop),
    (vertices.card = 9) →
    (∀ a b, red a b ∨ blue a b) →
    (∀ a b c, (red a b ∧ red b c ∧ red c a) → False) →
    (∃ A B C D, blue A B ∧ blue B C ∧ blue C D ∧ blue D A ∧ blue A C ∧ blue B D) := 
by
  -- Proof goes here
  sorry

end nonagon_blue_quadrilateral_l4_4282


namespace temperature_on_Monday_l4_4919

theorem temperature_on_Monday 
  (M T W Th F : ℝ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : F = 36) : 
  M = 44 := 
by 
  -- Proof omitted
  sorry

end temperature_on_Monday_l4_4919


namespace bear_problem_l4_4376

-- Definitions of the variables
variables (W B Br : ℕ)

-- Given conditions
def condition1 : B = 2 * W := sorry
def condition2 : B = 60 := sorry
def condition3 : W + B + Br = 190 := sorry

-- The proof statement
theorem bear_problem : Br - B = 40 :=
by
  -- we would use the given conditions to prove this statement
  sorry

end bear_problem_l4_4376


namespace small_cubes_with_two_faces_painted_l4_4854

theorem small_cubes_with_two_faces_painted :
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  12 * (n - 2) = 36 :=
by
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  exact sorry

end small_cubes_with_two_faces_painted_l4_4854


namespace complementary_event_l4_4019

-- Definitions based on the conditions
def EventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≥ 2

def complementEventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≤ 1

-- Theorem based on the question and correct answer
theorem complementary_event (products : List Bool) :
  complementEventA products ↔ ¬ EventA products :=
by sorry

end complementary_event_l4_4019


namespace novice_experienced_parts_l4_4444

variables (x y : ℕ)

theorem novice_experienced_parts :
  (y - x = 30) ∧ (x + 2 * y = 180) :=
sorry

end novice_experienced_parts_l4_4444


namespace floor_sum_proof_l4_4979

noncomputable def floor_sum (x y z w : ℝ) : ℝ :=
  x + y + z + w

theorem floor_sum_proof
  (x y z w : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hw_pos : 0 < w)
  (h1 : x^2 + y^2 = 2010)
  (h2 : z^2 + w^2 = 2010)
  (h3 : x * z = 1008)
  (h4 : y * w = 1008) :
  ⌊floor_sum x y z w⌋ = 126 :=
by
  sorry

end floor_sum_proof_l4_4979


namespace additional_cost_tv_ad_l4_4722

theorem additional_cost_tv_ad (in_store_price : ℝ) (payment : ℝ) (shipping : ℝ) :
  in_store_price = 129.95 → payment = 29.99 → shipping = 14.95 → 
  (4 * payment + shipping - in_store_price) * 100 = 496 :=
by
  intros h1 h2 h3
  sorry

end additional_cost_tv_ad_l4_4722


namespace expression_value_l4_4927

theorem expression_value 
  (x : ℝ)
  (h : x = 1/5) :
  (x^2 - 4) / (x^2 - 2 * x) = 11 :=
  by
  rw [h]
  sorry

end expression_value_l4_4927


namespace worth_of_presents_l4_4301

def ring_value := 4000
def car_value := 2000
def bracelet_value := 2 * ring_value

def total_worth := ring_value + car_value + bracelet_value

theorem worth_of_presents : total_worth = 14000 := by
  sorry

end worth_of_presents_l4_4301


namespace best_fitting_model_is_model_2_l4_4842

-- Variables representing the correlation coefficients of the four models
def R2_model_1 : ℝ := 0.86
def R2_model_2 : ℝ := 0.96
def R2_model_3 : ℝ := 0.73
def R2_model_4 : ℝ := 0.66

-- Statement asserting that Model 2 has the best fitting effect
theorem best_fitting_model_is_model_2 :
  R2_model_2 = 0.96 ∧ R2_model_2 > R2_model_1 ∧ R2_model_2 > R2_model_3 ∧ R2_model_2 > R2_model_4 :=
by {
  sorry
}

end best_fitting_model_is_model_2_l4_4842


namespace find_x2_plus_y2_l4_4624

noncomputable def xy : ℝ := 12
noncomputable def eq2 (x y : ℝ) : Prop := x^2 * y + x * y^2 + x + y = 120

theorem find_x2_plus_y2 (x y : ℝ) (h1 : xy = 12) (h2 : eq2 x y) : 
  x^2 + y^2 = 10344 / 169 :=
sorry

end find_x2_plus_y2_l4_4624


namespace probability_of_choosing_gulongzhong_l4_4736

def num_attractions : Nat := 4
def num_ways_gulongzhong : Nat := 1
def probability_gulongzhong : ℚ := num_ways_gulongzhong / num_attractions

theorem probability_of_choosing_gulongzhong : probability_gulongzhong = 1 / 4 := 
by 
  sorry

end probability_of_choosing_gulongzhong_l4_4736


namespace cylinder_volume_transformation_l4_4792

variable (r h : ℝ)
variable (V_original : ℝ)
variable (V_new : ℝ)

noncomputable def original_volume : ℝ := Real.pi * r^2 * h

noncomputable def new_volume : ℝ := Real.pi * (3 * r)^2 * (2 * h)

theorem cylinder_volume_transformation 
  (h_original : original_volume r h = 15) :
  new_volume r h = 270 :=
by
  unfold original_volume at h_original
  unfold new_volume
  sorry

end cylinder_volume_transformation_l4_4792


namespace average_math_score_l4_4212

theorem average_math_score (scores : Fin 4 → ℕ) (other_avg : ℕ) (num_students : ℕ) (num_other_students : ℕ)
  (h1 : scores 0 = 90) (h2 : scores 1 = 85) (h3 : scores 2 = 88) (h4 : scores 3 = 80)
  (h5 : other_avg = 82) (h6 : num_students = 30) (h7 : num_other_students = 26) :
  (90 + 85 + 88 + 80 + 26 * 82) / 30 = 82.5 :=
by
  sorry

end average_math_score_l4_4212


namespace jack_total_cost_l4_4078

def cost_of_tires (n : ℕ) (price_per_tire : ℕ) : ℕ := n * price_per_tire
def cost_of_window (price_per_window : ℕ) : ℕ := price_per_window

theorem jack_total_cost :
  cost_of_tires 3 250 + cost_of_window 700 = 1450 :=
by
  sorry

end jack_total_cost_l4_4078


namespace range_of_e_l4_4105

theorem range_of_e (a b c d e : ℝ) (h₁ : a + b + c + d + e = 8) (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end range_of_e_l4_4105


namespace fiona_probability_correct_l4_4459

def probability_to_reach_pad14 :=
  (1 / 27) + (1 / 3) = 13 / 27 ∧
  (13 / 27) * (1 / 3) = 13 / 81 ∧
  (13 / 81) * (1 / 3) = 13 / 243 ∧
  (13 / 243) * (1 / 3) = 13 / 729 ∧
  (1 / 81) + (1 / 27) + (1 / 27) = 4 / 81 ∧
  (13 / 729) * (4 / 81) = 52 / 59049

theorem fiona_probability_correct :
  (probability_to_reach_pad14 : Prop) := by
  sorry

end fiona_probability_correct_l4_4459


namespace solve_inequality_l4_4591

def numerator (x : ℝ) : ℝ := x ^ 2 - 4 * x + 3
def denominator (x : ℝ) : ℝ := (x - 2) ^ 2

theorem solve_inequality : { x : ℝ | numerator x / denominator x < 0 } = { x : ℝ | 1 < x ∧ x < 3 } :=
by
  sorry

end solve_inequality_l4_4591


namespace range_of_x_for_y1_gt_y2_l4_4872

noncomputable def y1 (x : ℝ) : ℝ := x - 3
noncomputable def y2 (x : ℝ) : ℝ := 4 / x

theorem range_of_x_for_y1_gt_y2 :
  ∀ x : ℝ, (y1 x > y2 x) ↔ ((-1 < x ∧ x < 0) ∨ (x > 4)) := by
  sorry

end range_of_x_for_y1_gt_y2_l4_4872


namespace particle_probability_l4_4599

theorem particle_probability 
  (P : ℕ → ℝ) (n : ℕ)
  (h1 : P 0 = 1)
  (h2 : P 1 = 2 / 3)
  (h3 : ∀ n ≥ 3, P n = 2 / 3 * P (n-1) + 1 / 3 * P (n-2)) :
  P n = 2 / 3 + 1 / 12 * (1 - (-1 / 3)^(n-1)) := 
sorry

end particle_probability_l4_4599


namespace intersection_of_A_and_B_l4_4628

noncomputable def A : Set ℕ := {x | 2 ≤ x ∧ x ≤ 4}
def B : Set ℕ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l4_4628


namespace initial_overs_l4_4299

theorem initial_overs (x : ℝ) (r1 : ℝ) (r2 : ℝ) (target : ℝ) (overs_remaining : ℝ) :
  r1 = 3.2 ∧ overs_remaining = 22 ∧ r2 = 11.363636363636363 ∧ target = 282 ∧
  (r1 * x + r2 * overs_remaining = target) → x = 10 :=
by
  intro h
  obtain ⟨hr1, ho, hr2, ht, heq⟩ := h
  sorry

end initial_overs_l4_4299


namespace solve_diophantine_eq_l4_4881

theorem solve_diophantine_eq (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^2 = b * (b + 7) ↔ (a = 12 ∧ b = 9) ∨ (a = 0 ∧ b = 0) := 
by 
  sorry

end solve_diophantine_eq_l4_4881


namespace area_of_triangle_l4_4893

theorem area_of_triangle (p : ℝ) (h_p : 0 < p ∧ p < 10) : 
    let C := (0, p)
    let O := (0, 0)
    let B := (10, 0)
    (1/2) * 10 * p = 5 * p := 
by
  sorry

end area_of_triangle_l4_4893


namespace find_C_l4_4730

theorem find_C (A B C : ℕ) (h1 : (8 + 4 + A + 7 + 3 + B + 2) % 3 = 0)
  (h2 : (5 + 2 + 9 + A + B + 4 + C) % 3 = 0) : C = 2 :=
by
  sorry

end find_C_l4_4730


namespace problem_solution_l4_4519

theorem problem_solution (x : ℝ) : 
  (x < -2 ∨ (-2 < x ∧ x ≤ 0) ∨ (0 < x ∧ x < 2) ∨ (2 ≤ x ∧ x < (15 - Real.sqrt 257) / 8) ∨ ((15 + Real.sqrt 257) / 8 < x)) ↔ 
  (x^2 - 1) / (x + 2) ≥ 3 / (x - 2) + 7 / 4 := sorry

end problem_solution_l4_4519


namespace simplify_and_evaluate_expression_l4_4715

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 6) :
  (1 + (2 / (x + 1))) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end simplify_and_evaluate_expression_l4_4715


namespace suresh_wifes_speed_l4_4623

-- Define conditions
def circumference_of_track : ℝ := 0.726 -- track circumference in kilometers
def suresh_speed : ℝ := 4.5 -- Suresh's speed in km/hr
def meeting_time_in_hours : ℝ := 0.088 -- time till they meet in hours

-- Define the question and expected answer
theorem suresh_wifes_speed : ∃ (V : ℝ), V = 3.75 :=
  by
    -- Let Distance_covered_by_both = circumference_of_track
    let Distance_covered_by_suresh : ℝ := suresh_speed * meeting_time_in_hours
    let Distance_covered_by_suresh_wife : ℝ := circumference_of_track - Distance_covered_by_suresh
    let suresh_wifes_speed : ℝ := Distance_covered_by_suresh_wife / meeting_time_in_hours
    -- Expected answer
    existsi suresh_wifes_speed
    sorry

end suresh_wifes_speed_l4_4623


namespace proof_inequalities_equivalence_max_f_value_l4_4616

-- Definitions for the conditions
def inequality1 (x: ℝ) := |x - 2| > 1
def inequality2 (x: ℝ) := x^2 - 4 * x + 3 > 0

-- The main statements to prove
theorem proof_inequalities_equivalence : 
  {x : ℝ | inequality1 x} = {x : ℝ | inequality2 x} := 
sorry

noncomputable def f (x: ℝ) := 4 * Real.sqrt (x - 3) + 3 * Real.sqrt (5 - x)

theorem max_f_value : 
  ∃ x : ℝ, (3 ≤ x ∧ x ≤ 5) ∧ (f x = 5 * Real.sqrt 2) ∧ ∀ y : ℝ, ((3 ≤ y ∧ y ≤ 5) → f y ≤ 5 * Real.sqrt 2) :=
sorry

end proof_inequalities_equivalence_max_f_value_l4_4616


namespace total_cost_of_fruits_l4_4939

theorem total_cost_of_fruits (h_orange_weight : 12 * 2 = 24)
                             (h_apple_weight : 8 * 3.75 = 30)
                             (price_orange : ℝ := 1.5)
                             (price_apple : ℝ := 2.0) :
  (5 * 2 * price_orange + 4 * 3.75 * price_apple) = 45 :=
by
  sorry

end total_cost_of_fruits_l4_4939


namespace max_tickets_l4_4838


theorem max_tickets (cost_regular : ℕ) (cost_discounted : ℕ) (threshold : ℕ) (total_money : ℕ) 
  (h1 : cost_regular = 15) 
  (h2 : cost_discounted = 12) 
  (h3 : threshold = 5)
  (h4 : total_money = 150) 
  : (total_money / cost_regular ≤ 10) ∧ 
    ((total_money - threshold * cost_regular) / cost_discounted + threshold = 11) :=
by
  sorry

end max_tickets_l4_4838


namespace original_price_petrol_in_euros_l4_4428

theorem original_price_petrol_in_euros
  (P : ℝ) -- The original price of petrol in USD per gallon
  (h1 : 0.865 * P * 7.25 + 0.135 * 325 = 325) -- Condition derived from price reduction and additional gallons
  (h2 : P > 0) -- Ensure original price is positive
  (exchange_rate : ℝ) (h3 : exchange_rate = 1.15) : 
  P / exchange_rate = 38.98 :=
by 
  let price_in_euros := P / exchange_rate 
  have h4 : price_in_euros = 38.98 := sorry
  exact h4

end original_price_petrol_in_euros_l4_4428


namespace trains_distance_apart_l4_4971

-- Define the initial conditions
def cattle_train_speed : ℝ := 56
def diesel_train_speed : ℝ := cattle_train_speed - 33
def cattle_train_time : ℝ := 6 + 12
def diesel_train_time : ℝ := 12

-- Calculate distances
def cattle_train_distance : ℝ := cattle_train_speed * cattle_train_time
def diesel_train_distance : ℝ := diesel_train_speed * diesel_train_time

-- Define total distance apart
def distance_apart : ℝ := cattle_train_distance + diesel_train_distance

-- The theorem to prove
theorem trains_distance_apart :
  distance_apart = 1284 :=
by
  -- Skip the proof
  sorry

end trains_distance_apart_l4_4971


namespace xy_power_l4_4429

def x : ℚ := 3/4
def y : ℚ := 4/3

theorem xy_power : x^7 * y^8 = 4/3 := by
  sorry

end xy_power_l4_4429


namespace randolph_age_l4_4749

theorem randolph_age (R Sy S : ℕ) 
  (h1 : R = Sy + 5) 
  (h2 : Sy = 2 * S) 
  (h3 : S = 25) : 
  R = 55 :=
by 
  sorry

end randolph_age_l4_4749


namespace people_counted_l4_4906

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end people_counted_l4_4906


namespace car_speed_in_mph_l4_4189

-- Defining the given conditions
def fuel_efficiency : ℚ := 56 -- kilometers per liter
def gallons_to_liters : ℚ := 3.8 -- liters per gallon
def kilometers_to_miles : ℚ := 1 / 1.6 -- miles per kilometer
def fuel_decrease_gallons : ℚ := 3.9 -- gallons
def time_hours : ℚ := 5.7 -- hours

-- Using definitions to compute the speed
theorem car_speed_in_mph :
  (fuel_decrease_gallons * gallons_to_liters * fuel_efficiency * kilometers_to_miles) / time_hours = 91 :=
sorry

end car_speed_in_mph_l4_4189


namespace gcd_polynomials_l4_4350

-- Define a as a multiple of 1836
def is_multiple_of (a b : ℤ) : Prop := ∃ k : ℤ, a = k * b

-- Problem statement: gcd of the polynomial expressions given the condition
theorem gcd_polynomials (a : ℤ) (h : is_multiple_of a 1836) : Int.gcd (2 * a^2 + 11 * a + 40) (a + 4) = 4 :=
by
  sorry

end gcd_polynomials_l4_4350


namespace wage_difference_l4_4554

variable (P Q : ℝ)
variable (h : ℝ)
axiom wage_relation : P = 1.5 * Q
axiom time_relation : 360 = P * h
axiom time_relation_q : 360 = Q * (h + 10)

theorem wage_difference : P - Q = 6 :=
  by
  sorry

end wage_difference_l4_4554


namespace total_cost_of_cable_l4_4701

-- Defining the conditions as constants
def east_west_streets := 18
def east_west_length := 2
def north_south_streets := 10
def north_south_length := 4
def cable_per_mile_street := 5
def cost_per_mile_cable := 2000

-- The theorem contains the problem statement and asserts the answer
theorem total_cost_of_cable :
  (east_west_streets * east_west_length + north_south_streets * north_south_length) * cable_per_mile_street * cost_per_mile_cable = 760000 := 
  sorry

end total_cost_of_cable_l4_4701


namespace factorize_expression_l4_4178

variable (a : ℝ)

theorem factorize_expression : a^3 - 2 * a^2 = a^2 * (a - 2) :=
by
  sorry

end factorize_expression_l4_4178


namespace minimum_value_of_x_plus_2y_l4_4283

open Real

theorem minimum_value_of_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 := by
  sorry

end minimum_value_of_x_plus_2y_l4_4283


namespace annual_income_of_A_l4_4583

theorem annual_income_of_A 
  (ratio_AB : ℕ → ℕ → Prop)
  (income_C : ℕ)
  (income_B_more_C : ℕ → ℕ → Prop)
  (income_B_from_ratio : ℕ → ℕ → Prop)
  (income_C_value : income_C = 16000)
  (income_B_condition : ∀ c, income_B_more_C 17920 c)
  (income_A_condition : ∀ b, ratio_AB 5 (b/2))
  : ∃ a, a = 537600 :=
by
  sorry

end annual_income_of_A_l4_4583


namespace sequence_tuple_l4_4961

/-- Prove the unique solution to the system of equations derived from the sequence pattern. -/
theorem sequence_tuple (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 7) : (x, y) = (8, 1) :=
by
  sorry

end sequence_tuple_l4_4961


namespace first_term_of_geometric_sequence_l4_4015

theorem first_term_of_geometric_sequence :
  ∀ (a b c : ℝ), 
    (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 16 = b * r ∧ c = 16 * r ∧ 128 = c * r) →
    a = 1 / 4 :=
by
  intros a b c
  rintro ⟨r, hr0, hbr, h16r, hcr, h128r⟩
  sorry

end first_term_of_geometric_sequence_l4_4015


namespace number_of_buses_l4_4489

theorem number_of_buses (total_people : ℕ) (bus_capacity : ℕ) (h1 : total_people = 1230) (h2 : bus_capacity = 48) : 
  Nat.ceil (total_people / bus_capacity : ℝ) = 26 := 
by 
  unfold Nat.ceil 
  sorry

end number_of_buses_l4_4489


namespace parker_total_weight_l4_4266

def twenty_pound := 20
def thirty_pound := 30
def forty_pound := 40

def first_set_weight := (2 * twenty_pound) + (1 * thirty_pound) + (1 * forty_pound)
def second_set_weight := (1 * twenty_pound) + (2 * thirty_pound) + (2 * forty_pound)
def third_set_weight := (3 * thirty_pound) + (3 * forty_pound)

def total_weight := first_set_weight + second_set_weight + third_set_weight

theorem parker_total_weight :
  total_weight = 480 := by
  sorry

end parker_total_weight_l4_4266


namespace polynomial_identity_l4_4605

theorem polynomial_identity (a b c d e f : ℤ)
  (h_eq : ∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 770 := by
  sorry

end polynomial_identity_l4_4605


namespace eq_condition_implies_inequality_l4_4490

theorem eq_condition_implies_inequality (a : ℝ) (h_neg_root : 2 * a - 4 < 0) : (a - 3) * (a - 4) > 0 :=
by {
  sorry
}

end eq_condition_implies_inequality_l4_4490


namespace count_valid_arrays_l4_4987

-- Define the integer array condition
def valid_array (x1 x2 x3 x4 : ℕ) : Prop :=
  0 < x1 ∧ x1 ≤ x2 ∧ x2 < x3 ∧ x3 ≤ x4 ∧ x4 < 7

-- State the theorem that proves the number of valid arrays is 70
theorem count_valid_arrays : ∃ (n : ℕ), n = 70 ∧ 
    ∀ (x1 x2 x3 x4 : ℕ), valid_array x1 x2 x3 x4 -> ∃ (n : ℕ), n = 70 :=
by
  -- The proof can be filled in later
  sorry

end count_valid_arrays_l4_4987


namespace train_cross_pole_in_time_l4_4978

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_pole_in_time :
  time_to_cross_pole 100 126 = 100 / (126 * (1000 / 3600)) :=
by
  -- this will unfold the calculation step-by-step
  unfold time_to_cross_pole
  sorry

end train_cross_pole_in_time_l4_4978


namespace mrs_sheridan_total_cats_l4_4451

-- Definitions from the conditions
def original_cats : Nat := 17
def additional_cats : Nat := 14

-- The total number of cats is the sum of the original and additional cats
def total_cats : Nat := original_cats + additional_cats

-- Statement to prove
theorem mrs_sheridan_total_cats : total_cats = 31 := by
  sorry

end mrs_sheridan_total_cats_l4_4451


namespace find_c_plus_inv_b_l4_4969

theorem find_c_plus_inv_b (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 :=
by
  sorry

end find_c_plus_inv_b_l4_4969


namespace cartesian_equation_of_circle_c2_positional_relationship_between_circles_l4_4727
noncomputable def circle_c1 := {p : ℝ × ℝ | (p.1)^2 - 2*p.1 + (p.2)^2 = 0}
noncomputable def circle_c2_polar (theta : ℝ) : ℝ × ℝ := (2 * Real.sin theta * Real.cos theta, 2 * Real.sin theta * Real.sin theta)
noncomputable def circle_c2_cartesian := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 1)^2 = 1}

theorem cartesian_equation_of_circle_c2 :
  ∀ p : ℝ × ℝ, (∃ θ : ℝ, p = circle_c2_polar θ) ↔ p ∈ circle_c2_cartesian :=
by
  sorry

theorem positional_relationship_between_circles :
  ∃ p : ℝ × ℝ, p ∈ circle_c1 ∧ p ∈ circle_c2_cartesian :=
by
  sorry

end cartesian_equation_of_circle_c2_positional_relationship_between_circles_l4_4727


namespace paint_cost_is_correct_l4_4721

-- Definition of known conditions
def costPerKg : ℕ := 50
def coveragePerKg : ℕ := 20
def sideOfCube : ℕ := 20

-- Definition of correct answer
def totalCost : ℕ := 6000

-- Theorem statement
theorem paint_cost_is_correct : (6 * (sideOfCube * sideOfCube) / coveragePerKg) * costPerKg = totalCost :=
by
  sorry

end paint_cost_is_correct_l4_4721


namespace max_digit_d_of_form_7d733e_multiple_of_33_l4_4470

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end max_digit_d_of_form_7d733e_multiple_of_33_l4_4470


namespace find_number_of_olives_l4_4289

theorem find_number_of_olives (O : ℕ)
  (lettuce_choices : 2 = 2)
  (tomato_choices : 3 = 3)
  (soup_choices : 2 = 2)
  (total_combos : 2 * 3 * O * 2 = 48) :
  O = 4 :=
by
  sorry

end find_number_of_olives_l4_4289


namespace count_even_integers_between_l4_4825

theorem count_even_integers_between : 
    let lower := 18 / 5
    let upper := 45 / 2
    ∃ (count : ℕ), (∀ n : ℕ, lower < n ∧ n < upper → n % 2 = 0 → n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 10 ∨ n = 12 ∨ n = 14 ∨ n = 16 ∨ n = 18 ∨ n = 20 ∨ n = 22) ∧ count = 10 :=
by
  sorry

end count_even_integers_between_l4_4825


namespace minimum_x_y_l4_4226

theorem minimum_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
by sorry

end minimum_x_y_l4_4226


namespace calculate_selling_price_l4_4741

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 20
noncomputable def profit_percent : ℝ := 22.448979591836732

noncomputable def total_cost : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := (profit_percent / 100) * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem calculate_selling_price : selling_price = 300 := by
  sorry

end calculate_selling_price_l4_4741


namespace arithmetic_geometric_sequence_ab_l4_4671

theorem arithmetic_geometric_sequence_ab :
  ∀ (a l m b n : ℤ), 
    (b < 0) → 
    (2 * a = -10) → 
    (b^2 = 9) → 
    ab = 15 :=
by
  intros a l m b n hb ha hb_eq
  sorry

end arithmetic_geometric_sequence_ab_l4_4671


namespace find_C_plus_D_l4_4543

noncomputable def polynomial_divisible (x : ℝ) (C : ℝ) (D : ℝ) : Prop := 
  ∃ (ω : ℝ), ω^2 + ω + 1 = 0 ∧ ω^104 + C*ω + D = 0

theorem find_C_plus_D (C D : ℝ) : 
  (∃ x : ℝ, polynomial_divisible x C D) → C + D = 2 :=
by
  sorry

end find_C_plus_D_l4_4543


namespace second_monkey_took_20_peaches_l4_4748

theorem second_monkey_took_20_peaches (total_peaches : ℕ) 
  (h1 : total_peaches > 0)
  (eldest_share : ℕ)
  (middle_share : ℕ)
  (youngest_share : ℕ)
  (h3 : total_peaches = eldest_share + middle_share + youngest_share)
  (h4 : eldest_share = (total_peaches * 5) / 9)
  (second_total : ℕ := total_peaches - eldest_share)
  (h5 : middle_share = (second_total * 5) / 9)
  (h6 : youngest_share = second_total - middle_share)
  (h7 : eldest_share - youngest_share = 29) :
  middle_share = 20 :=
by
  sorry

end second_monkey_took_20_peaches_l4_4748


namespace carrots_thrown_out_l4_4702

variable (x : ℕ)

theorem carrots_thrown_out :
  let initial_carrots := 23
  let picked_later := 47
  let total_carrots := 60
  initial_carrots - x + picked_later = total_carrots → x = 10 :=
by
  intros
  sorry

end carrots_thrown_out_l4_4702


namespace intersection_of_A_and_B_l4_4771

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end intersection_of_A_and_B_l4_4771


namespace find_m_l4_4184

open Real

noncomputable def curve_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

noncomputable def line_equation (m t x y : ℝ) : Prop :=
  x = (sqrt 3 / 2) * t + m ∧ y = (1 / 2) * t

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_m (m : ℝ) (h_nonneg : 0 ≤ m) :
  (∀ (t1 t2 : ℝ), (∀ x y, line_equation m t1 x y → curve_equation x y) → 
                   (∀ x y, line_equation m t2 x y → curve_equation x y) →
                   (dist m 0 x1 y1) * (dist m 0 x2 y2) = 1) →
  m = 1 ∨ m = 1 + sqrt 2 :=
sorry

end find_m_l4_4184


namespace question_statement_l4_4076

def line := Type
def plane := Type

-- Definitions for line lying in plane and planes being parallel 
def isIn (a : line) (α : plane) : Prop := sorry
def isParallel (α β : plane) : Prop := sorry
def isParallelLinePlane (a : line) (β : plane) : Prop := sorry

-- Conditions 
variables (a b : line) (α β : plane) 
variable (distinct_lines : a ≠ b)
variable (distinct_planes : α ≠ β)

-- Main statement to prove
theorem question_statement (h_parallel_planes : isParallel α β) (h_line_in_plane : isIn a α) : isParallelLinePlane a β := 
sorry

end question_statement_l4_4076


namespace positive_difference_of_two_numbers_l4_4203

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l4_4203


namespace max_m_value_l4_4694

variables {x y m : ℝ}

theorem max_m_value (h1 : 4 * x + 3 * y = 4 * m + 5)
                     (h2 : 3 * x - y = m - 1)
                     (h3 : x + 4 * y ≤ 3) :
                     m ≤ -1 :=
sorry

end max_m_value_l4_4694


namespace foci_distance_of_hyperbola_l4_4491

theorem foci_distance_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 32 - y^2 / 8 = 1) → 2 * (Real.sqrt (32 + 8)) = 4 * Real.sqrt 10 :=
by
  intros x y h
  sorry

end foci_distance_of_hyperbola_l4_4491


namespace cylindrical_to_rectangular_l4_4925

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 6) (hθ : θ = π / 3) (hz : z = -3) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, -3) :=
by
  sorry

end cylindrical_to_rectangular_l4_4925


namespace range_x_plus_y_l4_4849

theorem range_x_plus_y (x y : ℝ) (h : x^3 + y^3 = 2) : 0 < x + y ∧ x + y ≤ 2 :=
by {
  sorry
}

end range_x_plus_y_l4_4849


namespace binomial_odd_sum_l4_4234

theorem binomial_odd_sum (n : ℕ) (h : (2:ℕ)^(n - 1) = 64) : n = 7 :=
by
  sorry

end binomial_odd_sum_l4_4234


namespace find_bottle_price_l4_4062

theorem find_bottle_price 
  (x : ℝ) 
  (promotion_free_bottles : ℝ := 3)
  (discount_per_bottle : ℝ := 0.6)
  (box_price : ℝ := 26)
  (box_bottles : ℝ := 4) :
  ∃ x : ℝ, (box_price / (x - discount_per_bottle)) - (box_price / x) = promotion_free_bottles :=
sorry

end find_bottle_price_l4_4062


namespace student_answers_all_correctly_l4_4463

/-- 
The exam tickets have 2 theoretical questions and 1 problem each. There are 28 tickets. 
A student is prepared for 50 theoretical questions out of 56 and 22 problems out of 28.
The probability that by drawing a ticket at random, and the student answers all questions 
correctly is 0.625.
-/
theorem student_answers_all_correctly :
  let total_theoretical := 56
  let total_problems := 28
  let prepared_theoretical := 50
  let prepared_problems := 22
  let p_correct_theoretical := (prepared_theoretical * (prepared_theoretical - 1)) / (total_theoretical * (total_theoretical - 1))
  let p_correct_problem := prepared_problems / total_problems
  let combined_probability := p_correct_theoretical * p_correct_problem
  combined_probability = 0.625 :=
  sorry

end student_answers_all_correctly_l4_4463


namespace average_of_remaining_five_l4_4418

open Nat Real

theorem average_of_remaining_five (avg9 avg4 : ℝ) (S S4 : ℝ) 
(h1 : avg9 = 18) (h2 : avg4 = 8) 
(h_sum9 : S = avg9 * 9) 
(h_sum4 : S4 = avg4 * 4) :
(S - S4) / 5 = 26 := by
  sorry

end average_of_remaining_five_l4_4418


namespace sufficient_but_not_necessary_condition_l4_4868

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 0 → x^2 > 0) ∧ ¬(x^2 > 0 → x > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l4_4868


namespace p_is_necessary_but_not_sufficient_for_q_l4_4592

def p (x : ℝ) : Prop := |2 * x - 3| < 1
def q (x : ℝ) : Prop := x * (x - 3) < 0

theorem p_is_necessary_but_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ ¬(∀ x : ℝ, p x → q x) :=
by sorry

end p_is_necessary_but_not_sufficient_for_q_l4_4592


namespace right_triangle_area_and_hypotenuse_l4_4612

-- Definitions based on given conditions
def a : ℕ := 24
def b : ℕ := 2 * a + 10

-- Statements based on the questions and correct answers
theorem right_triangle_area_and_hypotenuse :
  (1 / 2 : ℝ) * (a : ℝ) * (b : ℝ) = 696 ∧ (Real.sqrt ((a : ℝ)^2 + (b : ℝ)^2) = Real.sqrt 3940) := by
  sorry

end right_triangle_area_and_hypotenuse_l4_4612


namespace fraction_subtraction_l4_4220

theorem fraction_subtraction :
  ((2 + 4 + 6 : ℚ) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12) :=
by
  sorry

end fraction_subtraction_l4_4220


namespace sum_of_reciprocals_factors_12_l4_4817

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l4_4817


namespace sufficient_not_necessary_condition_l4_4217

theorem sufficient_not_necessary_condition {x : ℝ} (h : 1 < x ∧ x < 2) : x < 2 ∧ ¬(∀ x, x < 2 → (1 < x ∧ x < 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l4_4217


namespace photos_on_last_page_l4_4775

noncomputable def total_photos : ℕ := 10 * 35 * 4
noncomputable def photos_per_page_after_reorganization : ℕ := 8
noncomputable def total_pages_needed : ℕ := (total_photos + photos_per_page_after_reorganization - 1) / photos_per_page_after_reorganization
noncomputable def pages_filled_in_first_6_albums : ℕ := 6 * 35
noncomputable def last_page_photos : ℕ := if total_pages_needed ≤ pages_filled_in_first_6_albums then 0 else total_photos % photos_per_page_after_reorganization

theorem photos_on_last_page : last_page_photos = 0 :=
by
  sorry

end photos_on_last_page_l4_4775


namespace shaded_area_l4_4793

theorem shaded_area (R : ℝ) (π : ℝ) (h1 : π * (R / 2)^2 * 2 = 1) : 
  (π * R^2 - (π * (R / 2)^2 * 2)) = 1 := 
by
  sorry

end shaded_area_l4_4793


namespace shifted_parabola_correct_l4_4379

-- Define original equation of parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2 - 1

-- Define shifted equation of parabola
def shifted_parabola (x : ℝ) : ℝ := 2 * (x + 1)^2 - 1

-- Proof statement: the expression of the new parabola after shifting 1 unit to the left
theorem shifted_parabola_correct :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) :=
by
  -- Proof is omitted, sorry
  sorry

end shifted_parabola_correct_l4_4379


namespace determine_students_and_benches_l4_4442

theorem determine_students_and_benches (a b s : ℕ) :
  (s = a * b + 5) ∧ (s = 8 * b - 4) →
  ((a = 7 ∧ b = 9 ∧ s = 68) ∨ (a = 5 ∧ b = 3 ∧ s = 20)) :=
by
  sorry

end determine_students_and_benches_l4_4442


namespace Bill_original_profit_percentage_l4_4999

theorem Bill_original_profit_percentage 
  (S : ℝ) 
  (h_S : S = 879.9999999999993) 
  (h_cond : ∀ (P : ℝ), 1.17 * P = S + 56) :
  ∃ (profit_percentage : ℝ), profit_percentage = 10 := 
by
  sorry

end Bill_original_profit_percentage_l4_4999


namespace quadratic_inequality_solution_l4_4136

theorem quadratic_inequality_solution:
  (∃ p : ℝ, ∀ x : ℝ, x^2 + p * x - 6 < 0 ↔ -3 < x ∧ x < 2) → ∃ p : ℝ, p = 1 :=
by
  intro h
  sorry

end quadratic_inequality_solution_l4_4136


namespace flat_rate_first_night_l4_4686

-- Definitions of conditions
def total_cost_sarah (f n : ℕ) := f + 3 * n = 210
def total_cost_mark (f n : ℕ) := f + 7 * n = 450

-- Main theorem to be proven
theorem flat_rate_first_night : 
  ∃ f n : ℕ, total_cost_sarah f n ∧ total_cost_mark f n ∧ f = 30 :=
by
  sorry

end flat_rate_first_night_l4_4686


namespace ab_plus_2_l4_4391

theorem ab_plus_2 (a b : ℝ) (h : ∀ x : ℝ, (x - 3) * (3 * x + 7) = x^2 - 12 * x + 27 → x = a ∨ x = b) (ha : a ≠ b) :
  (a + 2) * (b + 2) = -30 :=
sorry

end ab_plus_2_l4_4391


namespace sacks_after_days_l4_4294

-- Define the number of sacks harvested per day
def harvest_per_day : ℕ := 74

-- Define the number of sacks discarded per day
def discard_per_day : ℕ := 71

-- Define the days of harvest
def days_of_harvest : ℕ := 51

-- Define the number of sacks that are not discarded per day
def net_sacks_per_day : ℕ := harvest_per_day - discard_per_day

-- Define the total number of sacks after the specified days of harvest
def total_sacks : ℕ := days_of_harvest * net_sacks_per_day

theorem sacks_after_days :
  total_sacks = 153 := by
  sorry

end sacks_after_days_l4_4294


namespace lending_rate_is_8_percent_l4_4696

-- Define all given conditions.
def principal₁ : ℝ := 5000
def time₁ : ℝ := 2
def rate₁ : ℝ := 4  -- in percentage
def gain_per_year : ℝ := 200

-- Prove that the interest rate for lending is 8%
theorem lending_rate_is_8_percent :
  ∃ (rate₂ : ℝ), rate₂ = 8 :=
by
  let interest₁ := principal₁ * rate₁ * time₁ / 100
  let interest_per_year₁ := interest₁ / time₁
  let total_interest_received_per_year := gain_per_year + interest_per_year₁
  let rate₂ := (total_interest_received_per_year * 100) / principal₁
  use rate₂
  sorry

end lending_rate_is_8_percent_l4_4696


namespace cos_B_value_l4_4336

theorem cos_B_value (A B C a b c : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * a * Real.cos B) :
  Real.cos B = Real.sqrt 3 / 3 := by
  sorry

end cos_B_value_l4_4336


namespace monotonic_iff_a_range_l4_4161

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem monotonic_iff_a_range (a : ℝ) : 
  (∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 ∨ f a x1 ≥ f a x2) ↔ (-3 < a ∧ a < 6) :=
by 
  sorry

end monotonic_iff_a_range_l4_4161


namespace change_in_spiders_l4_4947

theorem change_in_spiders 
  (x a y b : ℤ) 
  (h1 : x + a = 20) 
  (h2 : y + b = 23) 
  (h3 : x - b = 5) :
  y - a = 8 := 
by
  sorry

end change_in_spiders_l4_4947


namespace distance_from_point_to_line_l4_4072

-- Definition of the conditions
def point := (3, 0)
def line_y := 1

-- Problem statement: Prove that the distance between the point (3,0) and the line y=1 is 1.
theorem distance_from_point_to_line (point : ℝ × ℝ) (line_y : ℝ) : abs (point.snd - line_y) = 1 :=
by
  -- insert proof here
  sorry

end distance_from_point_to_line_l4_4072


namespace problem1_problem2_problem3_problem4_l4_4088

-- Problem 1: Prove (1 * -6) + -13 = -19
theorem problem1 : (1 * -6) + -13 = -19 := by 
  sorry

-- Problem 2: Prove (3/5) + (-3/4) = -3/20
theorem problem2 : (3/5 : ℚ) + (-3/4) = -3/20 := by 
  sorry

-- Problem 3: Prove 4.7 + (-0.8) + 5.3 + (-8.2) = 1
theorem problem3 : (4.7 + (-0.8) + 5.3 + (-8.2) : ℝ) = 1 := by 
  sorry

-- Problem 4: Prove (-1/6) + (1/3) + (-1/12) = 1/12
theorem problem4 : (-1/6 : ℚ) + (1/3) + (-1/12) = 1/12 := by 
  sorry

end problem1_problem2_problem3_problem4_l4_4088


namespace integer_solutions_system_ineq_l4_4375

theorem integer_solutions_system_ineq (x : ℤ) :
  (3 * x + 6 > x + 8 ∧ (x : ℚ) / 4 ≥ (x - 1) / 3) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by
  sorry

end integer_solutions_system_ineq_l4_4375


namespace johns_father_age_l4_4166

variable {Age : Type} [OrderedRing Age]
variables (J M F : Age)

def john_age := J
def mother_age := M
def father_age := F

def john_younger_than_father (F J : Age) : Prop := F = 2 * J
def father_older_than_mother (F M : Age) : Prop := F = M + 4
def age_difference_between_john_and_mother (M J : Age) : Prop := M = J + 16

-- The question to be proved in Lean:
theorem johns_father_age :
  john_younger_than_father F J →
  father_older_than_mother F M →
  age_difference_between_john_and_mother M J →
  F = 40 := 
by
  intros h1 h2 h3
  sorry

end johns_father_age_l4_4166


namespace min_packs_126_l4_4809

-- Define the sizes of soda packs
def pack_sizes : List ℕ := [6, 12, 24, 48]

-- Define the total number of cans required
def total_cans : ℕ := 126

-- Define a function to calculate the minimum number of packs required
noncomputable def min_packs_to_reach_target (target : ℕ) (sizes : List ℕ) : ℕ :=
sorry -- Implementation will be complex dynamic programming or greedy algorithm

-- The main theorem statement to prove
theorem min_packs_126 (P : ℕ) (h1 : (min_packs_to_reach_target total_cans pack_sizes) = P) : P = 4 :=
sorry -- Proof not required

end min_packs_126_l4_4809


namespace inequality1_solution_inequality2_solution_l4_4171

variables (x a : ℝ)

theorem inequality1_solution : (∀ x : ℝ, (2 * x) / (x + 1) < 1 ↔ -1 < x ∧ x < 1) :=
by
  sorry

theorem inequality2_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 + (2 - a) * x - 2 * a ≥ 0 ↔ 
    (a = -2 → true) ∧ 
    (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧ 
    (a < -2 → (x ≤ a ∨ x ≥ -2))) :=
by
  sorry

end inequality1_solution_inequality2_solution_l4_4171


namespace division_of_negatives_l4_4352

theorem division_of_negatives : (-500 : ℤ) / (-50 : ℤ) = 10 := by
  sorry

end division_of_negatives_l4_4352


namespace number_of_possible_monograms_l4_4832

-- Define the set of letters before 'M'
def letters_before_M : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'}

-- Define the set of letters after 'M'
def letters_after_M : Finset Char := {'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

-- State the theorem 
theorem number_of_possible_monograms : 
  (letters_before_M.card * letters_after_M.card) = 156 :=
by
  sorry

end number_of_possible_monograms_l4_4832


namespace area_code_length_l4_4061

theorem area_code_length (n : ℕ) (h : 224^n - 222^n = 888) : n = 2 :=
sorry

end area_code_length_l4_4061


namespace bilion_wins_1000000_dollars_l4_4095

theorem bilion_wins_1000000_dollars :
  ∃ (p : ℕ), (p = 1000000) ∧ (p % 3 = 1) → p = 1000000 :=
by
  sorry

end bilion_wins_1000000_dollars_l4_4095


namespace total_value_of_goods_l4_4587

theorem total_value_of_goods (V : ℝ) (tax_paid : ℝ) (tax_exemption : ℝ) (tax_rate : ℝ) :
  tax_exemption = 600 → tax_rate = 0.11 → tax_paid = 123.2 → 0.11 * (V - 600) = tax_paid → V = 1720 :=
by
  sorry

end total_value_of_goods_l4_4587


namespace technician_percent_round_trip_l4_4944

noncomputable def round_trip_percentage_completed (D : ℝ) : ℝ :=
  let total_round_trip := 2 * D
  let distance_completed := D + 0.10 * D
  (distance_completed / total_round_trip) * 100

theorem technician_percent_round_trip (D : ℝ) (h : D > 0) : 
  round_trip_percentage_completed D = 55 := 
by 
  sorry

end technician_percent_round_trip_l4_4944


namespace six_inch_cube_value_eq_844_l4_4689

-- Definition of the value of a cube in lean
noncomputable def cube_value (s₁ s₂ : ℕ) (value₁ : ℕ) : ℕ :=
  let volume₁ := s₁ ^ 3
  let volume₂ := s₂ ^ 3
  (value₁ * volume₂) / volume₁

-- Theorem stating the equivalence between the volumes and values.
theorem six_inch_cube_value_eq_844 :
  cube_value 4 6 250 = 844 :=
by
  sorry

end six_inch_cube_value_eq_844_l4_4689


namespace lower_limit_of_b_l4_4876

theorem lower_limit_of_b (a : ℤ) (b : ℤ) (h₁ : 8 < a ∧ a < 15) (h₂ : ∃ x, x < b ∧ b < 21) (h₃ : (14 : ℚ) / b - (9 : ℚ) / b = 1.55) : b = 4 :=
by
  sorry

end lower_limit_of_b_l4_4876


namespace father_son_fish_problem_l4_4102

variables {F S x : ℕ}

theorem father_son_fish_problem (h1 : F - x = S + x) (h2 : F + x = 2 * (S - x)) : 
  (F - S) / S = 2 / 5 :=
by sorry

end father_son_fish_problem_l4_4102


namespace becky_packs_lunch_days_l4_4799

-- Definitions of conditions
def school_days := 180
def aliyah_packing_fraction := 1 / 2
def becky_relative_fraction := 1 / 2

-- Derived quantities from conditions
def aliyah_pack_days := school_days * aliyah_packing_fraction
def becky_pack_days := aliyah_pack_days * becky_relative_fraction

-- Statement to prove
theorem becky_packs_lunch_days : becky_pack_days = 45 := by
  sorry

end becky_packs_lunch_days_l4_4799


namespace max_value_2x_minus_y_l4_4168

theorem max_value_2x_minus_y (x y : ℝ) (h₁ : x + y - 1 < 0) (h₂ : x - y ≤ 0) (h₃ : 0 ≤ x) :
  ∃ z, (z = 2 * x - y) ∧ (z ≤ (1 / 2)) :=
sorry

end max_value_2x_minus_y_l4_4168


namespace range_of_a_no_fixed_points_l4_4658

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1

theorem range_of_a_no_fixed_points : 
  ∀ a : ℝ, ¬∃ x : ℝ, f x a = x ↔ -1 < a ∧ a < 3 :=
by sorry

end range_of_a_no_fixed_points_l4_4658


namespace average_score_l4_4342

theorem average_score (a_males : ℕ) (a_females : ℕ) (n_males : ℕ) (n_females : ℕ)
  (h_males : a_males = 85) (h_females : a_females = 92) (h_n_males : n_males = 8) (h_n_females : n_females = 20) :
  (a_males * n_males + a_females * n_females) / (n_males + n_females) = 90 :=
by
  sorry

end average_score_l4_4342


namespace largest_c_l4_4151

theorem largest_c (c : ℝ) : (∃ x : ℝ, x^2 + 4 * x + c = -3) → c ≤ 1 :=
by
  sorry

end largest_c_l4_4151


namespace Collin_total_petals_l4_4644

-- Definitions of the conditions
def initial_flowers_Collin : ℕ := 25
def flowers_Ingrid : ℕ := 33
def petals_per_flower : ℕ := 4
def third_of_flowers_Ingrid : ℕ := flowers_Ingrid / 3

-- Total number of flowers Collin has after receiving from Ingrid
def total_flowers_Collin : ℕ := initial_flowers_Collin + third_of_flowers_Ingrid

-- Total number of petals Collin has
def total_petals_Collin : ℕ := total_flowers_Collin * petals_per_flower

-- The theorem to be proved
theorem Collin_total_petals : total_petals_Collin = 144 := by
  -- Proof goes here
  sorry

end Collin_total_petals_l4_4644


namespace relationship_among_abc_l4_4818

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 0.4
noncomputable def c : ℝ := 0.4 ^ 3

theorem relationship_among_abc : a > c ∧ c > b := by
  sorry

end relationship_among_abc_l4_4818


namespace inequality_solution_empty_set_l4_4559

theorem inequality_solution_empty_set : ∀ x : ℝ, ¬ (x * (2 - x) > 3) :=
by
  -- Translate the condition and show that there are no x satisfying the inequality
  sorry

end inequality_solution_empty_set_l4_4559


namespace mr_william_land_percentage_l4_4754

def total_tax_collected : ℝ := 3840
def mr_william_tax_paid : ℝ := 480
def expected_percentage : ℝ := 12.5

theorem mr_william_land_percentage :
  (mr_william_tax_paid / total_tax_collected) * 100 = expected_percentage := 
sorry

end mr_william_land_percentage_l4_4754


namespace book_cost_l4_4693

theorem book_cost (x y : ℝ) (h₁ : 2 * y = x) (h₂ : 100 + y = x - 100) : x = 200 := by
  sorry

end book_cost_l4_4693


namespace infinite_solutions_l4_4274

theorem infinite_solutions (x y : ℕ) (h : x ≥ 1 ∧ y ≥ 1) : ∃ (x y : ℕ), x^2 + y^2 = x^3 :=
by {
  sorry 
}

end infinite_solutions_l4_4274


namespace problem_1_problem_2_l4_4834

def setP (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def setS (x : ℝ) (m : ℝ) : Prop := |x - 1| ≤ m

theorem problem_1 (m : ℝ) : (m ∈ Set.Iic (3)) → ∀ x, (setP x ∨ setS x m) → setP x := sorry

theorem problem_2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (setP x ↔ setS x m) := sorry

end problem_1_problem_2_l4_4834


namespace number_of_cows_l4_4804

theorem number_of_cows (x y : ℕ) 
  (h1 : 4 * x + 2 * y = 14 + 2 * (x + y)) : 
  x = 7 :=
by
  sorry

end number_of_cows_l4_4804


namespace weight_of_dried_grapes_l4_4353

def fresh_grapes_initial_weight : ℝ := 25
def fresh_grapes_water_percentage : ℝ := 0.90
def dried_grapes_water_percentage : ℝ := 0.20

theorem weight_of_dried_grapes :
  (fresh_grapes_initial_weight * (1 - fresh_grapes_water_percentage)) /
  (1 - dried_grapes_water_percentage) = 3.125 := by
  -- Proof omitted
  sorry

end weight_of_dried_grapes_l4_4353


namespace value_of_a_l4_4869

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (5-x)/(x-2) ≥ 0 ↔ -3 < x ∧ x < a) → a > 5 :=
by
  intro h
  sorry

end value_of_a_l4_4869


namespace combinations_of_coins_l4_4027

noncomputable def count_combinations (target : ℕ) : ℕ :=
  (30 - 0*0) -- As it just returns 45 combinations

theorem combinations_of_coins : count_combinations 30 = 45 :=
  sorry

end combinations_of_coins_l4_4027


namespace monotonicity_and_extrema_l4_4882

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem monotonicity_and_extrema :
  (∀ x, -3 / 2 < x ∧ x < -1 → f x < f (x + 0.0001)) ∧
  (∀ x, -1 < x ∧ x < -1 / 2 → f x > f (x + 0.0001)) ∧
  (∀ x, -1 / 2 < x ∧ x < (Real.exp 2 - 3) / 2 → f x < f (x + 0.0001)) ∧
  ∀ x, x ∈ Set.Icc (-1 : ℝ) ((Real.exp 2 - 3) / 2) →
     (f (x) ≥ Real.log 2 + 1 / 4 → x = -1 / 2) ∧
     (f (x) ≤ 2 + (Real.exp 2 - 3)^2 / 4 → x = (Real.exp 2 - 3) / 2) :=
sorry

end monotonicity_and_extrema_l4_4882


namespace solve_inequality_l4_4954

theorem solve_inequality (k : ℝ) :
  (∀ (x : ℝ), (k + 2) * x > k + 2 → x < 1) → k = -3 :=
  by
  sorry

end solve_inequality_l4_4954


namespace money_last_weeks_l4_4957

-- Conditions
def money_from_mowing : ℕ := 14
def money_from_weeding : ℕ := 31
def weekly_spending : ℕ := 5

-- Total money made
def total_money : ℕ := money_from_mowing + money_from_weeding

-- Expected result
def expected_weeks : ℕ := 9

-- Prove the number of weeks the money will last Jerry
theorem money_last_weeks : (total_money / weekly_spending) = expected_weeks :=
by
  sorry

end money_last_weeks_l4_4957


namespace circle_radius_zero_l4_4159

theorem circle_radius_zero (x y : ℝ) :
  4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → ∃ c : ℝ × ℝ, ∃ r : ℝ, (x - c.1)^2 + (y - c.2)^2 = r^2 ∧ r = 0 :=
by
  sorry

end circle_radius_zero_l4_4159


namespace total_pieces_of_candy_l4_4438

-- Define the given conditions
def students : ℕ := 43
def pieces_per_student : ℕ := 8

-- Define the goal, which is proving the total number of pieces of candy is 344
theorem total_pieces_of_candy : students * pieces_per_student = 344 :=
by
  sorry

end total_pieces_of_candy_l4_4438


namespace min_m_n_sum_l4_4823

theorem min_m_n_sum (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_eq : 45 * m = n^3) : m + n = 90 :=
sorry

end min_m_n_sum_l4_4823


namespace tan_945_equals_1_l4_4279

noncomputable def tan_circular (x : ℝ) : ℝ := Real.tan x

theorem tan_945_equals_1 :
  tan_circular 945 = 1 := 
by
  sorry

end tan_945_equals_1_l4_4279


namespace find_number_l4_4720

theorem find_number (x q : ℕ) (h1 : x = 3 * q) (h2 : q + x + 3 = 63) : x = 45 :=
sorry

end find_number_l4_4720


namespace math_problem_l4_4683

-- Define constants and conversions from decimal/mixed numbers to fractions
def thirteen_and_three_quarters : ℚ := 55 / 4
def nine_and_sixth : ℚ := 55 / 6
def one_point_two : ℚ := 1.2
def ten_point_three : ℚ := 103 / 10
def eight_and_half : ℚ := 17 / 2
def six_point_eight : ℚ := 34 / 5
def three_and_three_fifths : ℚ := 18 / 5
def five_and_five_sixths : ℚ := 35 / 6
def three_and_two_thirds : ℚ := 11 / 3
def three_and_one_sixth : ℚ := 19 / 6
def fifty_six : ℚ := 56
def twenty_seven_and_sixth : ℚ := 163 / 6

def E : ℚ := 
  ((thirteen_and_three_quarters + nine_and_sixth) * one_point_two) / ((ten_point_three - eight_and_half) * (5 / 9)) + 
  ((six_point_eight - three_and_three_fifths) * five_and_five_sixths) / ((three_and_two_thirds - three_and_one_sixth) * fifty_six) - 
  twenty_seven_and_sixth

theorem math_problem : E = 29 / 3 := by
  sorry

end math_problem_l4_4683


namespace total_pages_read_l4_4861

def pages_read_yesterday : ℕ := 21
def pages_read_today : ℕ := 17

theorem total_pages_read : pages_read_yesterday + pages_read_today = 38 :=
by
  sorry

end total_pages_read_l4_4861


namespace max_product_l4_4070

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l4_4070


namespace yang_tricks_modulo_l4_4086

noncomputable def number_of_tricks_result : Nat :=
  let N := 20000
  let modulo := 100000
  N % modulo

theorem yang_tricks_modulo :
  number_of_tricks_result = 20000 :=
by
  sorry

end yang_tricks_modulo_l4_4086


namespace prob_four_of_a_kind_after_re_roll_l4_4253

noncomputable def probability_of_four_of_a_kind : ℚ :=
sorry

theorem prob_four_of_a_kind_after_re_roll :
  (probability_of_four_of_a_kind =
    (1 : ℚ) / 6) :=
sorry

end prob_four_of_a_kind_after_re_roll_l4_4253


namespace carol_first_round_points_l4_4452

theorem carol_first_round_points (P : ℤ) (h1 : P + 6 - 16 = 7) : P = 17 :=
by
  sorry

end carol_first_round_points_l4_4452


namespace total_jokes_sum_l4_4097

theorem total_jokes_sum :
  let jessy_week1 := 11
  let alan_week1 := 7
  let tom_week1 := 5
  let emily_week1 := 3
  let jessy_week4 := 11 * 3 ^ 3
  let alan_week4 := 7 * 2 ^ 3
  let tom_week4 := 5 * 4 ^ 3
  let emily_week4 := 3 * 4 ^ 3
  let jessy_total := 11 + 11 * 3 + 11 * 3 ^ 2 + jessy_week4
  let alan_total := 7 + 7 * 2 + 7 * 2 ^ 2 + alan_week4
  let tom_total := 5 + 5 * 4 + 5 * 4 ^ 2 + tom_week4
  let emily_total := 3 + 3 * 4 + 3 * 4 ^ 2 + emily_week4
  jessy_total + alan_total + tom_total + emily_total = 1225 :=
by 
  sorry

end total_jokes_sum_l4_4097


namespace diff_present_students_l4_4196

theorem diff_present_students (T A1 A2 A3 P1 P2 : ℕ) 
  (hT : T = 280)
  (h_total_absent : A1 + A2 + A3 = 240)
  (h_absent_ratio : A2 = 2 * A3)
  (h_absent_third_day : A3 = 280 / 7) 
  (hP1 : P1 = T - A1)
  (hP2 : P2 = T - A2) :
  P2 - P1 = 40 :=
sorry

end diff_present_students_l4_4196


namespace geometric_seq_relation_l4_4432

variables {α : Type*} [Field α]

-- Conditions for the arithmetic sequence (for reference)
def arithmetic_seq_sum (S : ℕ → α) (d : α) : Prop :=
∀ m n : ℕ, S (m + n) = S m + S n + (m * n) * d

-- Conditions for the geometric sequence
def geometric_seq_prod (T : ℕ → α) (q : α) : Prop :=
∀ m n : ℕ, T (m + n) = T m * T n * (q ^ (m * n))

-- Proving the desired relationship
theorem geometric_seq_relation {T : ℕ → α} {q : α} (h : geometric_seq_prod T q) (m n : ℕ) :
  T (m + n) = T m * T n * (q ^ (m * n)) :=
by
  apply h m n

end geometric_seq_relation_l4_4432


namespace tan_ratio_l4_4048

theorem tan_ratio (x y : ℝ)
  (h1 : Real.sin (x + y) = 5 / 8)
  (h2 : Real.sin (x - y) = 1 / 4) :
  (Real.tan x) / (Real.tan y) = 2 := sorry

end tan_ratio_l4_4048


namespace vasya_made_a_mistake_l4_4946

theorem vasya_made_a_mistake (A B V G D E : ℕ)
  (h1 : A ≠ B)
  (h2 : V ≠ G)
  (h3 : (10 * A + B) * (10 * V + G) = 1000 * D + 100 * D + 10 * E + E)
  (h4 : ∀ {X Y : ℕ}, X ≠ Y → D ≠ E) :
  False :=
by
  -- Proof goes here (skipped)
  sorry

end vasya_made_a_mistake_l4_4946


namespace find_two_digit_number_l4_4199

def is_positive (n : ℕ) := n > 0
def is_even (n : ℕ) := n % 2 = 0
def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def product_of_digits_is_square (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  ∃ k : ℕ, (tens * units) = k * k

theorem find_two_digit_number (N : ℕ) 
  (h_pos : is_positive N) 
  (h_ev : is_even N) 
  (h_mult_9 : is_multiple_of_9 N)
  (h_prod_square : product_of_digits_is_square N) 
: N = 90 := by 
  sorry

end find_two_digit_number_l4_4199


namespace alexandra_brianna_meeting_probability_l4_4383

noncomputable def probability_meeting (A B : ℕ × ℕ) : ℚ :=
if A = (0,0) ∧ B = (5,7) then 347 / 768 else 0

theorem alexandra_brianna_meeting_probability :
  probability_meeting (0,0) (5,7) = 347 / 768 := 
by sorry

end alexandra_brianna_meeting_probability_l4_4383


namespace infinite_rational_points_on_circle_l4_4968

noncomputable def exists_infinitely_many_rational_points_on_circle : Prop :=
  ∃ f : ℚ → ℚ × ℚ, (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                   (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m)

theorem infinite_rational_points_on_circle :
  ∃ (f : ℚ → ℚ × ℚ), (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                     (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m) := sorry

end infinite_rational_points_on_circle_l4_4968


namespace lcm_18_20_25_l4_4141

-- Lean 4 statement to prove the smallest positive integer divisible by 18, 20, and 25 is 900
theorem lcm_18_20_25 : Nat.lcm (Nat.lcm 18 20) 25 = 900 :=
by
  sorry

end lcm_18_20_25_l4_4141


namespace average_points_per_player_l4_4945

theorem average_points_per_player (Lefty_points Righty_points OtherTeammate_points : ℕ)
  (hL : Lefty_points = 20)
  (hR : Righty_points = Lefty_points / 2)
  (hO : OtherTeammate_points = 6 * Righty_points) :
  (Lefty_points + Righty_points + OtherTeammate_points) / 3 = 30 :=
by
  sorry

end average_points_per_player_l4_4945


namespace quadratic_single_intersection_l4_4405

theorem quadratic_single_intersection (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + m = 0 → x^2 - 2 * x + m = (x-1)^2) :=
sorry

end quadratic_single_intersection_l4_4405


namespace angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l4_4370

-- Conditions for (1): In ΔABC, A = 60°, a = 4√3, b = 4√2, prove B = 45°.
theorem angle_B_in_triangle_ABC
  (A : Real)
  (a b : Real)
  (hA : A = 60)
  (ha : a = 4 * Real.sqrt 3)
  (hb : b = 4 * Real.sqrt 2) :
  ∃ B : Real, B = 45 := by
  sorry

-- Conditions for (2): In ΔABC, a = 3√3, c = 2, B = 150°, prove b = 7.
theorem side_b_in_triangle_ABC
  (a c B : Real)
  (ha : a = 3 * Real.sqrt 3)
  (hc : c = 2)
  (hB : B = 150) :
  ∃ b : Real, b = 7 := by
  sorry

end angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l4_4370


namespace sin_150_eq_half_l4_4291

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end sin_150_eq_half_l4_4291


namespace minimum_value_of_y_at_l4_4024

def y (x : ℝ) : ℝ := |x + 1| + |x + 2| + |x + 3|

theorem minimum_value_of_y_at (x : ℝ) :
  (∀ x : ℝ, y x ≥ 2) ∧ (y (-2) = 2) :=
by 
  sorry

end minimum_value_of_y_at_l4_4024


namespace correct_answer_A_correct_answer_C_correct_answer_D_l4_4080

variable (f g : ℝ → ℝ)

namespace ProofProblem

-- Assume the given conditions
axiom f_eq : ∀ x, f x = 6 - deriv g x
axiom f_compl : ∀ x, f (1 - x) = 6 + deriv g (1 + x)
axiom g_odd : ∀ x, g x - 2 = -(g (-x) - 2)

-- Proving the correct answers
theorem correct_answer_A : g 0 = 2 :=
sorry

theorem correct_answer_C : ∀ x, g (x + 4) = g x :=
sorry

theorem correct_answer_D : f 1 * g 1 + f 3 * g 3 = 24 :=
sorry

end ProofProblem

end correct_answer_A_correct_answer_C_correct_answer_D_l4_4080


namespace abs_of_neg_one_third_l4_4003

theorem abs_of_neg_one_third : abs (- (1 / 3)) = (1 / 3) := by
  sorry

end abs_of_neg_one_third_l4_4003


namespace time_difference_correct_l4_4625

-- Definitions based on conditions
def malcolm_speed : ℝ := 5 -- Malcolm's speed in minutes per mile
def joshua_speed : ℝ := 7 -- Joshua's speed in minutes per mile
def race_length : ℝ := 12 -- Length of the race in miles

-- Calculate times based on speeds and race length
def malcolm_time : ℝ := malcolm_speed * race_length
def joshua_time : ℝ := joshua_speed * race_length

-- The statement that the difference in finish times is 24 minutes
theorem time_difference_correct : joshua_time - malcolm_time = 24 :=
by
  -- Proof goes here
  sorry

end time_difference_correct_l4_4625


namespace min_value_expression_l4_4641

variable {m n : ℝ}

theorem min_value_expression (hm : m > 0) (hn : n > 0) (hperp : m + n = 1) :
  ∃ (m n : ℝ), (1 / m + 2 / n = 3 + 2 * Real.sqrt 2) :=
by 
  sorry

end min_value_expression_l4_4641


namespace problem1_problem2_l4_4557

-- Problem1
theorem problem1 (a : ℤ) (h : a = -2) :
    ( (a^2 + a) / (a^2 - 3 * a) / (a^2 - 1) / (a - 3) - 1 / (a + 1) = 2 / 3) :=
by 
  sorry

-- Problem2
theorem problem2 (x : ℤ) :
    ( (x^2 - 1) / (x - 4) / (x + 1) / (4 - x) = 1 - x) :=
by 
  sorry

end problem1_problem2_l4_4557


namespace intersection_of_A_and_B_l4_4527

-- Definitions from conditions
def A : Set ℤ := {x | x - 1 ≥ 0}
def B : Set ℤ := {0, 1, 2}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l4_4527


namespace third_candidate_votes_l4_4879

-- Definition of the problem's conditions
variables (total_votes winning_votes candidate2_votes : ℕ)
variables (winning_percentage : ℚ)

-- Conditions given in the problem
def conditions : Prop :=
  winning_votes = 11628 ∧
  winning_percentage = 0.4969230769230769 ∧
  (total_votes : ℚ) = winning_votes / winning_percentage ∧
  candidate2_votes = 7636

-- The theorem we need to prove
theorem third_candidate_votes (total_votes winning_votes candidate2_votes : ℕ)
    (winning_percentage : ℚ)
    (h : conditions total_votes winning_votes candidate2_votes winning_percentage) :
    total_votes - (winning_votes + candidate2_votes) = 4136 := 
  sorry

end third_candidate_votes_l4_4879


namespace geom_mean_does_not_exist_l4_4426

theorem geom_mean_does_not_exist (a b : Real) (h1 : a = 2) (h2 : b = -2) : ¬ ∃ g : Real, g^2 = a * b := 
by
  sorry

end geom_mean_does_not_exist_l4_4426


namespace football_count_white_patches_count_l4_4617

theorem football_count (x : ℕ) (footballs : ℕ) (students : ℕ) (h1 : students - 9 = footballs + 9) (h2 : students = 2 * footballs + 9) : footballs = 27 :=
sorry

theorem white_patches_count (white_patches : ℕ) (h : 2 * 12 * 5 = 6 * white_patches) : white_patches = 20 :=
sorry

end football_count_white_patches_count_l4_4617


namespace Diane_age_l4_4084

variable (C D E : ℝ)

def Carla_age_is_four_times_Diane_age : Prop := C = 4 * D
def Emma_is_eight_years_older_than_Diane : Prop := E = D + 8
def Carla_and_Emma_are_twins : Prop := C = E

theorem Diane_age : Carla_age_is_four_times_Diane_age C D → 
                    Emma_is_eight_years_older_than_Diane D E → 
                    Carla_and_Emma_are_twins C E → 
                    D = 8 / 3 :=
by
  intros hC hE hTwins
  have h1 : C = 4 * D := hC
  have h2 : E = D + 8 := hE
  have h3 : C = E := hTwins
  sorry

end Diane_age_l4_4084


namespace problem_solution_l4_4213

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Complement within U
def complement_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- The proof goal
theorem problem_solution : (complement_U A) ∪ B = {2, 3, 4, 5} := by
  sorry

end problem_solution_l4_4213


namespace value_of_b7b9_l4_4916

-- Define arithmetic sequence and geometric sequence with given conditions
variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- The given conditions in the problem
def a_seq_arithmetic (a : ℕ → ℝ) := ∀ n, a n = a 1 + (n - 1) • (a 2 - a 1)
def b_seq_geometric (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n, b (n + 1) = r * b n
def given_condition (a : ℕ → ℝ) := 2 * a 5 - (a 8)^2 + 2 * a 11 = 0
def b8_eq_a8 (a b : ℕ → ℝ) := b 8 = a 8

-- The statement to prove
theorem value_of_b7b9 : a_seq_arithmetic a → b_seq_geometric b → given_condition a → b8_eq_a8 a b → b 7 * b 9 = 4 := by
  intros a_arith b_geom cond b8a8
  sorry

end value_of_b7b9_l4_4916


namespace shari_total_distance_l4_4965

theorem shari_total_distance (speed : ℝ) (time_1 : ℝ) (rest : ℝ) (time_2 : ℝ) (distance : ℝ) :
  speed = 4 ∧ time_1 = 2 ∧ rest = 0.5 ∧ time_2 = 1 ∧ distance = speed * time_1 + speed * time_2 → distance = 12 :=
by
  sorry

end shari_total_distance_l4_4965


namespace area_of_quadrilateral_l4_4805

theorem area_of_quadrilateral 
  (area_ΔBDF : ℝ) (area_ΔBFE : ℝ) (area_ΔEFC : ℝ) (area_ΔCDF : ℝ) (h₁ : area_ΔBDF = 5)
  (h₂ : area_ΔBFE = 10) (h₃ : area_ΔEFC = 10) (h₄ : area_ΔCDF = 15) :
  (80 - (area_ΔBDF + area_ΔBFE + area_ΔEFC + area_ΔCDF)) = 40 := 
  by sorry

end area_of_quadrilateral_l4_4805


namespace problem_one_l4_4920

def S_n (n : Nat) : Nat := 
  List.foldl (fun acc x => acc * 10 + 2) 0 (List.replicate n 2)

theorem problem_one : ∃ n ∈ Finset.range 2011, S_n n % 2011 = 0 := 
  sorry

end problem_one_l4_4920


namespace charity_distribution_l4_4154

theorem charity_distribution 
  (X : ℝ) (Y : ℝ) (Z : ℝ) (W : ℝ) (A : ℝ)
  (h1 : X > 0) (h2 : Y > 0) (h3 : Y < 100) (h4 : Z > 0) (h5 : W > 0) (h6 : A > 0)
  (h7 : W * A = X * (100 - Y) / 100) :
  (Y * X) / (100 * Z) = A * W * Y / (100 * Z) :=
by 
  sorry

end charity_distribution_l4_4154


namespace range_of_z_l4_4235

theorem range_of_z (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : -5 ≤ x + y ∧ x + y ≤ 5 :=
sorry

end range_of_z_l4_4235


namespace cannot_determine_E1_l4_4163

variable (a b c d : ℝ)

theorem cannot_determine_E1 (h1 : a + b - c - d = 5) (h2 : (b - d)^2 = 16) : 
  ¬ ∃ e : ℝ, e = a - b - c + d :=
by
  sorry

end cannot_determine_E1_l4_4163


namespace john_initial_payment_l4_4364

-- Definitions based on the conditions from step a)
def cost_per_soda : ℕ := 2
def num_sodas : ℕ := 3
def change_received : ℕ := 14

-- Problem Statement: Prove that the total amount of money John paid initially is $20
theorem john_initial_payment :
  cost_per_soda * num_sodas + change_received = 20 := 
by
  sorry -- Proof steps are omitted as per instructions

end john_initial_payment_l4_4364


namespace algebraic_identity_l4_4692

theorem algebraic_identity (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

example : (2011 : ℝ)^2 - (2010 : ℝ)^2 = 4021 := 
by
  have h := algebraic_identity 2011 2010
  rw [h]
  norm_num

end algebraic_identity_l4_4692


namespace combination_15_5_l4_4747

theorem combination_15_5 : 
  ∀ (n r : ℕ), n = 15 → r = 5 → n.choose r = 3003 :=
by
  intro n r h1 h2
  rw [h1, h2]
  exact Nat.choose_eq_factorial_div_factorial (by norm_num)

end combination_15_5_l4_4747


namespace cost_of_each_fish_is_four_l4_4648

-- Definitions according to the conditions
def number_of_fish_given_to_dog := 40
def number_of_fish_given_to_cat := number_of_fish_given_to_dog / 2
def total_fish := number_of_fish_given_to_dog + number_of_fish_given_to_cat
def total_cost := 240
def cost_per_fish := total_cost / total_fish

-- The main statement / theorem that needs to be proved
theorem cost_of_each_fish_is_four :
  cost_per_fish = 4 :=
by
  sorry

end cost_of_each_fish_is_four_l4_4648


namespace range_of_a_l4_4380

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, (2 * a + 1) * x + a - 2 > (2 * a + 1) * 0 + a - 2)
  (h2 : a - 2 < 0) : -1 / 2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l4_4380


namespace mildred_weight_is_correct_l4_4589

noncomputable def carol_weight := 9
noncomputable def mildred_weight := carol_weight + 50

theorem mildred_weight_is_correct : mildred_weight = 59 :=
by 
  -- the proof is omitted
  sorry

end mildred_weight_is_correct_l4_4589


namespace seconds_in_3_hours_45_minutes_l4_4822

theorem seconds_in_3_hours_45_minutes :
  let hours := 3
  let minutes := 45
  let minutes_in_hour := 60
  let seconds_in_minute := 60
  (hours * minutes_in_hour + minutes) * seconds_in_minute = 13500 := by
  sorry

end seconds_in_3_hours_45_minutes_l4_4822


namespace ensure_two_different_colors_ensure_two_yellow_balls_l4_4690

-- First statement: Ensuring two balls of different colors
theorem ensure_two_different_colors (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 11 ∧ 
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, draws i ≠ draws j := 
sorry

-- Second statement: Ensuring two yellow balls
theorem ensure_two_yellow_balls (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 22 ∧
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, (draws i).val - balls_red - balls_white < balls_yellow ∧ 
              (draws j).val - balls_red - balls_white < balls_yellow ∧
              draws i = draws j := 
sorry

end ensure_two_different_colors_ensure_two_yellow_balls_l4_4690


namespace fraction_of_journey_by_rail_l4_4600

theorem fraction_of_journey_by_rail :
  ∀ (x : ℝ), x * 130 + (17 / 20) * 130 + 6.5 = 130 → x = 1 / 10 :=
by
  -- proof
  sorry

end fraction_of_journey_by_rail_l4_4600


namespace evaluate_N_l4_4896

theorem evaluate_N (N : ℕ) :
    988 + 990 + 992 + 994 + 996 = 5000 - N → N = 40 :=
by
  sorry

end evaluate_N_l4_4896


namespace number_of_graphing_calculators_in_class_l4_4351

-- Define a structure for the problem
structure ClassData where
  num_boys : ℕ
  num_girls : ℕ
  num_scientific_calculators : ℕ
  num_girls_with_calculators : ℕ
  num_graphing_calculators : ℕ
  no_overlap : Prop

-- Instantiate the problem using given conditions
def mrs_anderson_class : ClassData :=
{
  num_boys := 20,
  num_girls := 18,
  num_scientific_calculators := 30,
  num_girls_with_calculators := 15,
  num_graphing_calculators := 10,
  no_overlap := true
}

-- Lean statement for the proof problem
theorem number_of_graphing_calculators_in_class (data : ClassData) :
  data.num_graphing_calculators = 10 :=
by
  sorry

end number_of_graphing_calculators_in_class_l4_4351


namespace total_slices_l4_4399

theorem total_slices (pizzas : ℕ) (slices1 slices2 slices3 slices4 : ℕ)
  (h1 : pizzas = 4)
  (h2 : slices1 = 8)
  (h3 : slices2 = 8)
  (h4 : slices3 = 10)
  (h5 : slices4 = 12) :
  slices1 + slices2 + slices3 + slices4 = 38 := by
  sorry

end total_slices_l4_4399


namespace range_of_x_l4_4251

noncomputable def T (x : ℝ) : ℝ := |(2 * x - 1)|

theorem range_of_x (x : ℝ) (h : ∀ a : ℝ, T x ≥ |1 + a| - |2 - a|) : 
  x ≤ -1 ∨ 2 ≤ x :=
by
  sorry

end range_of_x_l4_4251


namespace power_function_passes_through_fixed_point_l4_4162

theorem power_function_passes_through_fixed_point 
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
  (P : ℝ × ℝ) (hP : P = (4, 2))
  (f : ℝ → ℝ) (hf : f x = x ^ a) : ∀ x, f x = x ^ (1 / 2) :=
by
  sorry

end power_function_passes_through_fixed_point_l4_4162


namespace proof_problem_l4_4863

variable {x y : ℝ}

def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y = 1

theorem proof_problem (h : conditions x y) :
  x + y - 4 * x * y ≥ 0 ∧ (1 / x) + 4 / (1 + y) ≥ 9 / 2 :=
by
  sorry

end proof_problem_l4_4863


namespace Liliane_more_soda_than_Alice_l4_4499

variable (J : ℝ) -- Represents the amount of soda Jacqueline has

-- Conditions: Representing the amounts for Benjamin, Liliane, and Alice
def B := 1.75 * J
def L := 1.60 * J
def A := 1.30 * J

-- Question: Proving the relationship in percentage terms between the amounts Liliane and Alice have
theorem Liliane_more_soda_than_Alice :
  (L - A) / A * 100 = 23 := 
by sorry

end Liliane_more_soda_than_Alice_l4_4499


namespace Annabelle_saved_12_dollars_l4_4165

def weekly_allowance : ℕ := 30
def spent_on_junk_food : ℕ := weekly_allowance / 3
def spent_on_sweets : ℕ := 8
def total_spent : ℕ := spent_on_junk_food + spent_on_sweets
def saved_amount : ℕ := weekly_allowance - total_spent

theorem Annabelle_saved_12_dollars : saved_amount = 12 := by
  -- proof goes here
  sorry

end Annabelle_saved_12_dollars_l4_4165


namespace correct_average_l4_4115

theorem correct_average (avg: ℕ) (n: ℕ) (incorrect: ℕ) (correct: ℕ) 
  (h_avg : avg = 16) (h_n : n = 10) (h_incorrect : incorrect = 25) (h_correct : correct = 35) :
  (avg * n + (correct - incorrect)) / n = 17 := 
by
  sorry

end correct_average_l4_4115


namespace RS_plus_ST_l4_4661

theorem RS_plus_ST {a b c d e : ℕ} 
  (h1 : a = 68) 
  (h2 : b = 10) 
  (h3 : c = 7) 
  (h4 : d = 6) 
  : e = 3 :=
sorry

end RS_plus_ST_l4_4661


namespace parabola_vertex_range_l4_4777

def parabola_vertex_in_first_quadrant (m : ℝ) : Prop :=
  ∃ v : ℝ × ℝ, v = (m, m - 1) ∧ 0 < m ∧ 0 < (m - 1)

theorem parabola_vertex_range (m : ℝ) (h_vertex : parabola_vertex_in_first_quadrant m) :
  1 < m :=
by
  sorry

end parabola_vertex_range_l4_4777


namespace max_value_of_e_l4_4905

theorem max_value_of_e (a b c d e : ℝ) 
  (h₁ : a + b + c + d + e = 8) 
  (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ 16 / 5 :=
sorry

end max_value_of_e_l4_4905


namespace arithmetic_sequence_common_difference_l4_4504

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l4_4504


namespace tangent_line_through_origin_l4_4319

noncomputable def curve (x : ℝ) : ℝ := Real.exp (x - 1) + x

theorem tangent_line_through_origin :
  ∃ k : ℝ, k = 2 ∧ ∀ x y : ℝ, (y = k * x) ↔ (∃ m : ℝ, curve m = m + Real.exp (m - 1) ∧ (curve m) = (m + Real.exp (m - 1)) ∧ k = (Real.exp (m - 1) + 1) ∧ y = k * x ∧ y = 2*x) :=
by 
  sorry

end tangent_line_through_origin_l4_4319


namespace find_a_from_inequality_solution_set_l4_4632

theorem find_a_from_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (x^2 - a*x + 4 < 0) ↔ (1 < x ∧ x < 4)) -> a = 5 :=
by
  intro h
  sorry

end find_a_from_inequality_solution_set_l4_4632


namespace general_formula_minimum_n_l4_4431

-- Definitions based on given conditions
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d
def sum_arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions of the problem
def a2 : ℤ := -5
def S5 : ℤ := -20

-- Proving the general formula of the sequence
theorem general_formula :
  ∃ a₁ d, arith_seq a₁ d 2 = a2 ∧ sum_arith_seq a₁ d 5 = S5 ∧ (∀ n, arith_seq a₁ d n = n - 7) :=
by
  sorry

-- Proving the minimum value of n for which Sn > an
theorem minimum_n :
  ∃ n : ℕ, (n > 14) ∧ sum_arith_seq (-6) 1 n > arith_seq (-6) 1 n :=
by
  sorry

end general_formula_minimum_n_l4_4431


namespace solve_dfrac_eq_l4_4060

theorem solve_dfrac_eq (x : ℝ) (h : (x / 5) / 3 = 3 / (x / 5)) : x = 15 ∨ x = -15 := by
  sorry

end solve_dfrac_eq_l4_4060


namespace final_result_l4_4839

/-- A student chose a number, multiplied it by 5, then subtracted 138 
from the result. The number he chose was 48. What was the final result 
after subtracting 138? -/
theorem final_result (x : ℕ) (h1 : x = 48) : (x * 5) - 138 = 102 := by
  sorry

end final_result_l4_4839


namespace PR_length_right_triangle_l4_4745

theorem PR_length_right_triangle
  (P Q R : Type)
  (cos_R : ℝ)
  (PQ PR : ℝ)
  (h1 : cos_R = 5 * Real.sqrt 34 / 34)
  (h2 : PQ = Real.sqrt 34)
  (h3 : cos_R = PR / PQ) : PR = 5 := by
  sorry

end PR_length_right_triangle_l4_4745


namespace figure_total_area_l4_4622

theorem figure_total_area :
  let height_left_rect := 6
  let width_base_left_rect := 5
  let height_top_left_rect := 3
  let width_top_left_rect := 5
  let height_top_center_rect := 3
  let width_sum_center_rect := 10
  let height_top_right_rect := 8
  let width_top_right_rect := 2
  let area_total := (height_left_rect * width_base_left_rect) + (height_top_left_rect * width_top_left_rect) + (height_top_center_rect * width_sum_center_rect) + (height_top_right_rect * width_top_right_rect)
  area_total = 91
:= sorry

end figure_total_area_l4_4622


namespace radius_of_circle_l4_4677

theorem radius_of_circle (P : ℝ) (PQ QR : ℝ) (distance_center_P : ℝ) (r : ℝ) :
  P = 17 ∧ PQ = 12 ∧ QR = 8 ∧ (PQ * (PQ + QR) = (distance_center_P - r) * (distance_center_P + r)) → r = 7 :=
by
  sorry

end radius_of_circle_l4_4677


namespace intersection_dist_general_l4_4782

theorem intersection_dist_general {a b : ℝ} 
  (h1 : (a^2 + 1) * (a^2 + 4 * (b + 1)) = 34)
  (h2 : (a^2 + 1) * (a^2 + 4 * (b + 2)) = 42) : 
  ∀ x1 x2 : ℝ, 
  x1 ≠ x2 → 
  (x1 * x1 = a * x1 + b - 1 ∧ x2 * x2 = a * x2 + b - 1) → 
  |x2 - x1| = 3 * Real.sqrt 2 :=
by
  sorry

end intersection_dist_general_l4_4782


namespace line_passes_fixed_point_max_distance_eqn_l4_4643

-- Definition of the line equation
def line_eq (a b x y : ℝ) : Prop :=
  (2 * a + b) * x + (a + b) * y + a - b = 0

-- Point P
def point_P : ℝ × ℝ :=
  (3, 4)

-- Fixed point that the line passes through
def fixed_point : ℝ × ℝ :=
  (-2, 3)

-- Statement that the line passes through the fixed point
theorem line_passes_fixed_point (a b : ℝ) :
  line_eq a b (-2) 3 :=
sorry

-- Equation of the line when distance from point P to line is maximized
def line_max_distance (a b : ℝ) : Prop :=
  5 * 3 + 4 + 7 = 0

-- Statement that the equation of the line is as given when distance is maximized
theorem max_distance_eqn (a b : ℝ) :
  line_max_distance a b :=
sorry

end line_passes_fixed_point_max_distance_eqn_l4_4643


namespace tree_initial_leaves_l4_4325

theorem tree_initial_leaves (L : ℝ) (h1 : ∀ n : ℤ, 1 ≤ n ∧ n ≤ 4 → ∃ k : ℝ, L = k * (9/10)^n + k / 10^n)
                            (h2 : L * (9/10)^4 = 204) :
  L = 311 :=
by
  sorry

end tree_initial_leaves_l4_4325


namespace prob1_prob2_prob3_prob4_l4_4776

theorem prob1 : (3^3)^2 = 3^6 := by
  sorry

theorem prob2 : (-4 * x * y^3) * (-2 * x^2) = 8 * x^3 * y^3 := by
  sorry

theorem prob3 : 2 * x * (3 * y - x^2) + 2 * x * x^2 = 6 * x * y := by
  sorry

theorem prob4 : (20 * x^3 * y^5 - 10 * x^4 * y^4 - 20 * x^3 * y^2) / (-5 * x^3 * y^2) = -4 * y^3 + 2 * x * y^2 + 4 := by
  sorry

end prob1_prob2_prob3_prob4_l4_4776


namespace sequence_exists_l4_4242

theorem sequence_exists
  {a_0 b_0 c_0 a b c : ℤ}
  (gcd1 : Int.gcd (Int.gcd a_0 b_0) c_0 = 1)
  (gcd2 : Int.gcd (Int.gcd a b) c = 1) :
  ∃ (n : ℕ) (a_seq b_seq c_seq : Fin (n + 1) → ℤ),
    a_seq 0 = a_0 ∧ b_seq 0 = b_0 ∧ c_seq 0 = c_0 ∧ 
    a_seq n = a ∧ b_seq n = b ∧ c_seq n = c ∧
    ∀ (i : Fin n), (a_seq i) * (a_seq i.succ) + (b_seq i) * (b_seq i.succ) + (c_seq i) * (c_seq i.succ) = 1 :=
sorry

end sequence_exists_l4_4242


namespace root_interval_sum_l4_4173

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_interval_sum (a b : ℤ) (h1 : b - a = 1) (h2 : ∃! x : ℝ, a < x ∧ x < b ∧ f x = 0) : a + b = -3 :=
by
  sorry

end root_interval_sum_l4_4173


namespace geometric_sequence_a6_l4_4699

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 4) (h2 : a 4 = 2) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 6 = 4 :=
sorry

end geometric_sequence_a6_l4_4699


namespace fraction_evaluation_l4_4167

def h (x : ℤ) : ℤ := 3 * x + 4
def k (x : ℤ) : ℤ := 4 * x - 3

theorem fraction_evaluation :
  (h (k (h 3))) / (k (h (k 3))) = 151 / 121 :=
by sorry

end fraction_evaluation_l4_4167


namespace factorization_correct_l4_4224

theorem factorization_correct (x : ℝ) : 
  x^4 - 5*x^2 - 36 = (x^2 + 4)*(x + 3)*(x - 3) :=
sorry

end factorization_correct_l4_4224


namespace hyperbola_focal_length_l4_4264

theorem hyperbola_focal_length (x y : ℝ) : 
  (∃ h : x^2 / 9 - y^2 / 4 = 1, 
   ∀ a b : ℝ, a^2 = 9 → b^2 = 4 → 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 13) :=
by sorry

end hyperbola_focal_length_l4_4264


namespace total_apples_correct_l4_4255

def cost_per_apple : ℤ := 2
def money_emmy_has : ℤ := 200
def money_gerry_has : ℤ := 100
def total_money : ℤ := money_emmy_has + money_gerry_has
def total_apples_bought : ℤ := total_money / cost_per_apple

theorem total_apples_correct :
    total_apples_bought = 150 := by
    sorry

end total_apples_correct_l4_4255


namespace discount_percentage_l4_4422

theorem discount_percentage (shirts : ℕ) (total_cost : ℕ) (price_after_discount : ℕ) 
  (h1 : shirts = 3) (h2 : total_cost = 60) (h3 : price_after_discount = 12) : 
  ∃ discount_percentage : ℕ, discount_percentage = 40 := 
by 
  sorry

end discount_percentage_l4_4422


namespace three_digit_oddfactors_count_is_22_l4_4568

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l4_4568


namespace arithmetic_geometric_sequence_solution_l4_4397

theorem arithmetic_geometric_sequence_solution (u v : ℕ → ℝ) (a b u₀ : ℝ) :
  (∀ n, u (n + 1) = a * u n + b) ∧ (∀ n, v (n + 1) = a * v n + b) →
  u 0 = u₀ →
  v 0 = b / (1 - a) →
  ∀ n, u n = a ^ n * (u₀ - b / (1 - a)) + b / (1 - a) :=
by
  intros
  sorry

end arithmetic_geometric_sequence_solution_l4_4397


namespace winning_strategy_l4_4607

noncomputable def winning_player (n : ℕ) (h : n ≥ 2) : String :=
if n = 2 ∨ n = 4 ∨ n = 8 then "Ariane" else "Bérénice"

theorem winning_strategy (n : ℕ) (h : n ≥ 2) :
  (winning_player n h = "Ariane" ↔ (n = 2 ∨ n = 4 ∨ n = 8)) ∧
  (winning_player n h = "Bérénice" ↔ ¬ (n = 2 ∨ n = 4 ∨ n = 8)) :=
sorry

end winning_strategy_l4_4607


namespace fraction_numerator_less_denominator_l4_4312

theorem fraction_numerator_less_denominator (x : ℝ) (h : -3 ≤ x ∧ x ≤ 3) :
  (8 * x - 3 < 9 + 5 * x) ↔ (-3 ≤ x ∧ x < 3) :=
by sorry

end fraction_numerator_less_denominator_l4_4312


namespace average_score_first_2_matches_l4_4332

theorem average_score_first_2_matches (A : ℝ) 
  (h1 : 3 * 40 = 120) 
  (h2 : 5 * 36 = 180) 
  (h3 : 2 * A + 120 = 180) : 
  A = 30 := 
by 
  have hA : 2 * A = 60 := by linarith [h3]
  have hA2 : A = 30 := by linarith [hA]
  exact hA2

end average_score_first_2_matches_l4_4332


namespace van_distance_covered_l4_4360

noncomputable def distance_covered (V : ℝ) := 
  let D := V * 6
  D

theorem van_distance_covered : ∃ (D : ℝ), ∀ (V : ℝ), 
  (D = 288) ∧ (D = distance_covered V) ∧ (D = 32 * 9) :=
by
  sorry

end van_distance_covered_l4_4360


namespace find_a4_plus_b4_l4_4742

-- Variables representing the given conditions
variables {a b : ℝ}

-- The theorem statement to prove
theorem find_a4_plus_b4 (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 56 :=
sorry

end find_a4_plus_b4_l4_4742


namespace tony_belinda_combined_age_l4_4018

/-- Tony and Belinda have a combined age. Belinda is 8 more than twice Tony's age. 
Tony is 16 years old and Belinda is 40 years old. What is their combined age? -/
theorem tony_belinda_combined_age 
  (tonys_age : ℕ)
  (belindas_age : ℕ)
  (h1 : tonys_age = 16)
  (h2 : belindas_age = 40)
  (h3 : belindas_age = 2 * tonys_age + 8) :
  tonys_age + belindas_age = 56 :=
  by sorry

end tony_belinda_combined_age_l4_4018


namespace exists_x_divisible_by_3n_not_by_3np1_l4_4028

noncomputable def f (x : ℕ) : ℕ := x ^ 3 + 17

theorem exists_x_divisible_by_3n_not_by_3np1 (n : ℕ) (hn : 2 ≤ n) : 
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬ (3^(n+1) ∣ f x) :=
sorry

end exists_x_divisible_by_3n_not_by_3np1_l4_4028


namespace fraction_equality_l4_4731

theorem fraction_equality (a b c : ℝ) (h1 : b + c + a ≠ 0) (h2 : b + c ≠ a) : 
  (b^2 + a^2 - c^2 + 2*b*c) / (b^2 + c^2 - a^2 + 2*b*c) = 1 := 
by 
  sorry

end fraction_equality_l4_4731


namespace job_completion_days_l4_4719

theorem job_completion_days :
  let days_total := 150
  let workers_initial := 25
  let workers_less_efficient := 15
  let workers_more_efficient := 10
  let days_elapsed := 40
  let efficiency_less := 1
  let efficiency_more := 1.5
  let work_fraction_completed := 1/3
  let workers_fired_less := 4
  let workers_fired_more := 3
  let units_per_day_initial := (workers_less_efficient * efficiency_less) + (workers_more_efficient * efficiency_more)
  let work_completed := units_per_day_initial * days_elapsed
  let total_work := work_completed / work_fraction_completed
  let workers_remaining_less := workers_less_efficient - workers_fired_less
  let workers_remaining_more := workers_more_efficient - workers_fired_more
  let units_per_day_new := (workers_remaining_less * efficiency_less) + (workers_remaining_more * efficiency_more)
  let work_remaining := total_work * (2/3)
  let remaining_days := work_remaining / units_per_day_new
  remaining_days.ceil = 112 :=
by
  sorry

end job_completion_days_l4_4719


namespace special_pair_example_1_special_pair_example_2_special_pair_negation_l4_4285

-- Definition of "special rational number pair"
def is_special_rational_pair (a b : ℚ) : Prop := a + b = a * b - 1

-- Problem (1)
theorem special_pair_example_1 : is_special_rational_pair 5 (3 / 2) :=
  by sorry

-- Problem (2)
theorem special_pair_example_2 (a : ℚ) : is_special_rational_pair a 3 → a = 2 :=
  by sorry

-- Problem (3)
theorem special_pair_negation (m n : ℚ) : is_special_rational_pair m n → ¬ is_special_rational_pair (-n) (-m) :=
  by sorry

end special_pair_example_1_special_pair_example_2_special_pair_negation_l4_4285


namespace cardinality_bound_l4_4659

theorem cardinality_bound {m n : ℕ} (hm : m > 1) (hn : n > 1)
  (S : Finset ℕ) (hS : S.card = n)
  (A : Fin m → Finset ℕ)
  (h : ∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ∃ i, (x ∈ A i ∧ y ∉ (A i)) ∨ (x ∉ (A i) ∧ y ∈ A i)) :
  n ≤ 2^m :=
sorry

end cardinality_bound_l4_4659


namespace ship_speed_in_still_water_l4_4956

theorem ship_speed_in_still_water 
  (distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (x : ℝ) 
  (h1 : distance = 36)
  (h2 : time = 6)
  (h3 : current_speed = 3) 
  (h4 : (18 / (x + 3) + 18 / (x - 3) = 6)) 
  : x = 3 + 3 * Real.sqrt 2 :=
sorry

end ship_speed_in_still_water_l4_4956


namespace true_propositions_count_l4_4372

theorem true_propositions_count
  (a b c : ℝ)
  (h : a > b) :
  ( (a > b → a * c^2 > b * c^2) ∧
    (a * c^2 > b * c^2 → a > b) ∧
    (a ≤ b → a * c^2 ≤ b * c^2) ∧
    (a * c^2 ≤ b * c^2 → a ≤ b) 
  ) ∧ 
  (¬(a > b → a * c^2 > b * c^2) ∧
   ¬(a * c^2 ≤ b * c^2 → a ≤ b)) →
  (a * c^2 > b * c^2 → a > b) ∧
  (a ≤ b → a * c^2 ≤ b * c^2) ∨
  (a > b → a * c^2 > b * c^2) ∨
  (a * c^2 ≤ b * c^2 → a ≤ b) :=
sorry

end true_propositions_count_l4_4372


namespace median_of_circumscribed_trapezoid_l4_4120

theorem median_of_circumscribed_trapezoid (a b c d : ℝ) (h1 : a + b + c + d = 12) (h2 : a + b = c + d) : (a + b) / 2 = 3 :=
by
  sorry

end median_of_circumscribed_trapezoid_l4_4120


namespace john_mean_score_l4_4914

-- Define John's quiz scores as a list
def johnQuizScores := [95, 88, 90, 92, 94, 89]

-- Define the function to calculate the mean of a list of integers
def mean_scores (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

-- Prove that the mean of John's quiz scores is 91.3333 
theorem john_mean_score :
  mean_scores johnQuizScores = 91.3333 := by
  -- sorry is a placeholder for the missing proof
  sorry

end john_mean_score_l4_4914


namespace toms_initial_investment_l4_4667

theorem toms_initial_investment (t j k : ℕ) (hj_neq_ht : t ≠ j) (hk_neq_ht : t ≠ k) (hj_neq_hk : j ≠ k) 
  (h1 : t + j + k = 1200) 
  (h2 : t - 150 + 3 * j + 3 * k = 1800) : 
  t = 825 := 
sorry

end toms_initial_investment_l4_4667


namespace value_ranges_l4_4296

theorem value_ranges 
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_eq1 : 3 * a + 2 * b + c = 5)
  (h_eq2 : 2 * a + b - 3 * c = 1) :
  (3 / 7 ≤ c ∧ c ≤ 7 / 11) ∧ 
  (-5 / 7 ≤ (3 * a + b - 7 * c) ∧ (3 * a + b - 7 * c) ≤ -1 / 11) :=
by 
  sorry

end value_ranges_l4_4296


namespace perfect_cube_prime_l4_4033

theorem perfect_cube_prime (p : ℕ) (h_prime : Nat.Prime p) (h_cube : ∃ x : ℕ, 2 * p + 1 = x^3) : 
  2 * p + 1 = 27 ∧ p = 13 :=
by
  sorry

end perfect_cube_prime_l4_4033


namespace alpha_value_l4_4750

noncomputable def alpha (x : ℝ) := Real.arccos x

theorem alpha_value (h1 : Real.cos α = -1/6) (h2 : 0 < α ∧ α < Real.pi) : 
  α = Real.pi - alpha (1/6) :=
by
  sorry

end alpha_value_l4_4750


namespace quadratic_completion_l4_4272

theorem quadratic_completion (x d e f : ℤ) (h1 : 100*x^2 + 80*x - 144 = 0) (hd : d > 0) 
  (hde : (d * x + e)^2 = f) : d + e + f = 174 :=
sorry

end quadratic_completion_l4_4272


namespace find_number_l4_4183

theorem find_number (x : ℤ) (h : (((55 + x) / 7 + 40) * 5 = 555)) : x = 442 :=
sorry

end find_number_l4_4183


namespace f_at_one_f_increasing_f_range_for_ineq_l4_4195

-- Define the function f with its properties
noncomputable def f : ℝ → ℝ := sorry

-- Properties of f
axiom f_domain : ∀ x, 0 < x → f x ≠ 0 
axiom f_property_additive : ∀ x y, f (x * y) = f x + f y
axiom f_property_positive : ∀ x, (1 < x) → (0 < f x)
axiom f_property_fract : f (1/3) = -1

-- Proofs to be completed
theorem f_at_one : f 1 = 0 :=
sorry

theorem f_increasing : ∀ (x₁ x₂ : ℝ), (0 < x₁) → (0 < x₂) → (x₁ < x₂) → (f x₁ < f x₂) :=
sorry

theorem f_range_for_ineq : {x : ℝ | 2 < x ∧ x ≤ 9/4} = {x : ℝ | f x - f (x - 2) ≥ 2} :=
sorry

end f_at_one_f_increasing_f_range_for_ineq_l4_4195


namespace Connie_needs_more_money_l4_4013

-- Definitions based on the given conditions
def Money_saved : ℝ := 39
def Cost_of_watch : ℝ := 55
def Cost_of_watch_strap : ℝ := 15
def Tax_rate : ℝ := 0.08

-- Lean 4 statement to prove the required amount of money
theorem Connie_needs_more_money : 
  let total_cost_before_tax := Cost_of_watch + Cost_of_watch_strap
  let tax_amount := total_cost_before_tax * Tax_rate
  let total_cost_including_tax := total_cost_before_tax + tax_amount
  Money_saved < total_cost_including_tax →
  total_cost_including_tax - Money_saved = 36.60 :=
by
  sorry

end Connie_needs_more_money_l4_4013


namespace find_a5_l4_4895

variable {a : ℕ → ℝ}

-- Condition 1: {a_n} is an arithmetic sequence
def arithmetic_sequence (a: ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition 2: a1 + a9 = 10
axiom a1_a9_sum : a 1 + a 9 = 10

theorem find_a5 (h_arith : arithmetic_sequence a) : a 5 = 5 :=
by {
  sorry
}

end find_a5_l4_4895


namespace number_of_trees_planted_l4_4635

def current_trees : ℕ := 34
def final_trees : ℕ := 83
def planted_trees : ℕ := final_trees - current_trees

theorem number_of_trees_planted : planted_trees = 49 :=
by
  -- proof goes here, but it is skipped for now
  sorry

end number_of_trees_planted_l4_4635


namespace large_bucket_capacity_l4_4466

variable (S L : ℕ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
sorry

end large_bucket_capacity_l4_4466


namespace find_a200_l4_4685

def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + 2 * a n / n

theorem find_a200 (a : ℕ → ℕ) (h : seq a) : a 200 = 20100 :=
sorry

end find_a200_l4_4685


namespace compare_negatives_l4_4222

theorem compare_negatives : -2 < -3 / 2 :=
by sorry

end compare_negatives_l4_4222


namespace margarita_jumps_farther_l4_4984

-- Definitions for conditions
def RiccianaTotalDistance := 24
def RiccianaRunDistance := 20
def RiccianaJumpDistance := 4

def MargaritaRunDistance := 18
def MargaritaJumpDistance := 2 * RiccianaJumpDistance - 1

-- Theorem statement to prove the question
theorem margarita_jumps_farther :
  (MargaritaRunDistance + MargaritaJumpDistance) - RiccianaTotalDistance = 1 :=
by
  -- Proof will be written here
  sorry

end margarita_jumps_farther_l4_4984


namespace tenly_more_stuffed_animals_than_kenley_l4_4400

def mckenna_stuffed_animals := 34
def kenley_stuffed_animals := 2 * mckenna_stuffed_animals
def total_stuffed_animals_all := 175
def total_stuffed_animals_mckenna_kenley := mckenna_stuffed_animals + kenley_stuffed_animals
def tenly_stuffed_animals := total_stuffed_animals_all - total_stuffed_animals_mckenna_kenley
def stuffed_animals_difference := tenly_stuffed_animals - kenley_stuffed_animals

theorem tenly_more_stuffed_animals_than_kenley :
  stuffed_animals_difference = 5 := by
  sorry

end tenly_more_stuffed_animals_than_kenley_l4_4400


namespace total_travel_time_l4_4268

theorem total_travel_time (subway_time : ℕ) (train_multiplier : ℕ) (bike_time : ℕ) 
  (h_subway : subway_time = 10) 
  (h_train_multiplier : train_multiplier = 2) 
  (h_bike : bike_time = 8) : 
  subway_time + train_multiplier * subway_time + bike_time = 38 :=
by
  sorry

end total_travel_time_l4_4268


namespace simplify_expression_l4_4703

theorem simplify_expression (n : ℕ) (hn : 0 < n) :
  (3^(n+5) - 3 * 3^n) / (3 * 3^(n+4) - 6) = 80 / 81 :=
by
  sorry

end simplify_expression_l4_4703


namespace range_of_a_l4_4126

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 5 * a) ↔ (4 ≤ a ∨ a ≤ 1) :=
by
  sorry

end range_of_a_l4_4126


namespace basketball_starting_lineups_l4_4588

theorem basketball_starting_lineups (n_players n_guards n_forwards n_centers : ℕ)
  (h_players : n_players = 12)
  (h_guards : n_guards = 2)
  (h_forwards : n_forwards = 2)
  (h_centers : n_centers = 1) :
  (Nat.choose n_players n_guards) * (Nat.choose (n_players - n_guards) n_forwards) * (Nat.choose (n_players - n_guards - n_forwards) n_centers) = 23760 := by
  sorry

end basketball_starting_lineups_l4_4588


namespace train_length_is_correct_l4_4419

noncomputable def speed_kmph_to_mps (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def distance_crossed (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def train_length (speed_kmph crossing_time bridge_length : ℝ) : ℝ :=
  distance_crossed (speed_kmph_to_mps speed_kmph) crossing_time - bridge_length

theorem train_length_is_correct :
  ∀ (crossing_time bridge_length speed_kmph : ℝ),
    crossing_time = 26.997840172786177 →
    bridge_length = 150 →
    speed_kmph = 36 →
    train_length speed_kmph crossing_time bridge_length = 119.97840172786177 :=
by
  intros crossing_time bridge_length speed_kmph h1 h2 h3
  rw [h1, h2, h3]
  simp only [speed_kmph_to_mps, distance_crossed, train_length]
  sorry

end train_length_is_correct_l4_4419


namespace arithmetic_sequence_ratio_l4_4852

-- Definitions and conditions from the problem
variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
variable (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
variable (h3 : ∀ n, S n / T n = (3 * n - 1) / (n + 3))

-- The theorem that will give us the required answer
theorem arithmetic_sequence_ratio : 
  (a 8) / (b 5 + b 11) = 11 / 9 := by 
  have h4 := h3 15
  sorry

end arithmetic_sequence_ratio_l4_4852


namespace final_laptop_price_l4_4465

theorem final_laptop_price :
  let original_price := 1000.00
  let first_discounted_price := original_price * (1 - 0.10)
  let second_discounted_price := first_discounted_price * (1 - 0.25)
  let recycling_fee := second_discounted_price * 0.05
  let final_price := second_discounted_price + recycling_fee
  final_price = 708.75 :=
by
  sorry

end final_laptop_price_l4_4465


namespace Sasha_can_write_2011_l4_4862

theorem Sasha_can_write_2011 (N : ℕ) (hN : N > 1) : 
    ∃ (s : ℕ → ℕ), (s 0 = N) ∧ (∃ n, s n = 2011) ∧ 
    (∀ k, ∃ d, d > 1 ∧ (s (k + 1) = s k + d ∨ s (k + 1) = s k - d)) :=
sorry

end Sasha_can_write_2011_l4_4862


namespace find_added_number_l4_4662

def S₁₅ := 15 * 17
def S₁₆ := 16 * 20
def added_number := S₁₆ - S₁₅

theorem find_added_number : added_number = 65 :=
by
  sorry

end find_added_number_l4_4662


namespace solve_system_l4_4478

theorem solve_system :
  ∃ x y : ℚ, (4 * x - 7 * y = -20) ∧ (9 * x + 3 * y = -21) ∧ (x = -69 / 25) ∧ (y = 32 / 25) := by
  sorry

end solve_system_l4_4478


namespace min_value_fraction_l4_4614

theorem min_value_fraction (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (h_sum : a + 3 * b = 1) :
  (∀ x y : ℝ, (0 < x) → (0 < y) → x + 3 * y = 1 → 16 ≤ 1 / x + 3 / y) :=
sorry

end min_value_fraction_l4_4614


namespace min_cards_to_guarantee_four_same_suit_l4_4942

theorem min_cards_to_guarantee_four_same_suit (n : ℕ) (suits : Fin n) (cards_per_suit : ℕ) (total_cards : ℕ)
  (h1 : n = 4) (h2 : cards_per_suit = 13) : total_cards ≥ 13 :=
by
  sorry

end min_cards_to_guarantee_four_same_suit_l4_4942


namespace minimum_yellow_balls_l4_4676

theorem minimum_yellow_balls (g o y : ℕ) :
  (o ≥ (1/3:ℝ) * g) ∧ (o ≤ (1/4:ℝ) * y) ∧ (g + o ≥ 75) → y ≥ 76 :=
sorry

end minimum_yellow_balls_l4_4676


namespace solve_xyz_l4_4673

theorem solve_xyz (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : z > 0) (h4 : x^2 = y * 2^z + 1) :
  (z ≥ 4 ∧ x = 2^(z-1) + 1 ∧ y = 2^(z-2) + 1) ∨
  (z ≥ 5 ∧ x = 2^(z-1) - 1 ∧ y = 2^(z-2) - 1) ∨
  (z ≥ 3 ∧ x = 2^z - 1 ∧ y = 2^z - 2) :=
sorry

end solve_xyz_l4_4673


namespace quadratic_inequality_solution_l4_4243

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5 * x + 6 < 0 ↔ 2 < x ∧ x < 3 :=
by
  sorry

end quadratic_inequality_solution_l4_4243


namespace necessary_but_not_sufficient_condition_l4_4666

variable (a b : ℝ)

theorem necessary_but_not_sufficient_condition : (a > b) → ((a > b) ↔ ((a - b) * b^2 > 0)) :=
sorry

end necessary_but_not_sufficient_condition_l4_4666


namespace correct_calculation_l4_4991

def calculation_is_correct (a b x y : ℝ) : Prop :=
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y)

theorem correct_calculation :
  ∀ (a b x y : ℝ), calculation_is_correct a b x y :=
by
  intros a b x y
  sorry

end correct_calculation_l4_4991


namespace therapists_next_meeting_day_l4_4501

theorem therapists_next_meeting_day : Nat.lcm (Nat.lcm 5 2) (Nat.lcm 9 3) = 90 := by
  -- Given that Alex works every 5 days,
  -- Brice works every 2 days,
  -- Emma works every 9 days,
  -- and Fiona works every 3 days, we need to show that the LCM of these numbers is 90.
  sorry

end therapists_next_meeting_day_l4_4501


namespace sin_330_deg_l4_4726

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l4_4726


namespace selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l4_4733

-- Problem conditions
def cost_price : ℕ := 70
def max_price : ℕ := 99
def initial_price : ℕ := 110
def initial_sales : ℕ := 20
def price_drop_rate : ℕ := 1
def sales_increase_rate : ℕ := 2
def sales_increase_per_yuan : ℕ := 2
def profit_target : ℕ := 1200

-- Selling price for given sales volume
def selling_price_for_sales_volume (sales_volume : ℕ) : ℕ :=
  initial_price - (sales_volume - initial_sales) / sales_increase_per_yuan

-- Functional relationship between sales volume (y) and price (x)
def sales_volume_function (x : ℕ) : ℕ :=
  initial_sales + sales_increase_rate * (initial_price - x)

-- Profit for given price and resulting sales volume
def daily_profit (x : ℕ) : ℤ :=
  (x - cost_price) * (sales_volume_function x)

-- Part 1: Selling price for 30 items sold
theorem selling_price_30_items : selling_price_for_sales_volume 30 = 105 :=
by
  sorry

-- Part 2: Functional relationship between sales volume and selling price
theorem sales_volume_functional_relationship (x : ℕ) (hx : 70 ≤ x ∧ x ≤ 99) :
  sales_volume_function x = 240 - 2 * x :=
by
  sorry

-- Part 3: Selling price for a daily profit of 1200 yuan
theorem selling_price_for_1200_profit {x : ℕ} (hx : 70 ≤ x ∧ x ≤ 99) :
  daily_profit x = 1200 → x = 90 :=
by
  sorry

end selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l4_4733


namespace fraction_shaded_area_l4_4814

theorem fraction_shaded_area (PX XQ : ℝ) (PA PR PQ : ℝ) (h1 : PX = 1) (h2 : 3 * XQ = PX) (h3 : PQ = PR) (h4 : PA = 1) (h5 : PA + AR = PR) (h6 : PR = 4):
  (3 / 16 : ℝ) = 0.375 :=
by
  -- proof here
  sorry

end fraction_shaded_area_l4_4814


namespace smallest_number_of_students_l4_4406

theorem smallest_number_of_students
  (tenth_graders eighth_graders ninth_graders : ℕ)
  (ratio1 : 7 * eighth_graders = 4 * tenth_graders)
  (ratio2 : 9 * ninth_graders = 5 * tenth_graders) :
  (∀ n, (∃ a b c, a = 7 * b ∧ b = 4 * n ∧ a = 9 * c ∧ c = 5 * n) → n = 134) :=
by {
  -- We currently just assume the result for Lean to be syntactically correct
  sorry
}

end smallest_number_of_students_l4_4406


namespace age_of_person_l4_4420

theorem age_of_person (x : ℕ) (h : 3 * (x + 3) - 3 * (x - 3) = x) : x = 18 :=
  sorry

end age_of_person_l4_4420


namespace horses_eat_oats_twice_a_day_l4_4700

-- Define the main constants and assumptions
def number_of_horses : ℕ := 4
def oats_per_meal : ℕ := 4
def grain_per_day : ℕ := 3
def total_food : ℕ := 132
def duration_in_days : ℕ := 3

-- Main theorem statement
theorem horses_eat_oats_twice_a_day (x : ℕ) (h : duration_in_days * number_of_horses * (oats_per_meal * x + grain_per_day) = total_food) : x = 2 := 
sorry

end horses_eat_oats_twice_a_day_l4_4700


namespace checkerboard_black_squares_count_l4_4899

namespace Checkerboard

def is_black (n : ℕ) : Bool :=
  -- Define the alternating pattern of the checkerboard
  (n % 2 = 0)

def black_square_count (n : ℕ) : ℕ :=
  -- Calculate the number of black squares in a checkerboard of size n x n
  if n % 2 = 0 then n * n / 2 else n * n / 2 + n / 2 + 1

def additional_black_squares (n : ℕ) : ℕ :=
  -- Calculate the additional black squares due to modification of every 33rd square in every third row
  ((n - 1) / 3 + 1)

def total_black_squares (n : ℕ) : ℕ :=
  -- Calculate the total black squares considering the modified hypothesis
  black_square_count n + additional_black_squares n

theorem checkerboard_black_squares_count : total_black_squares 33 = 555 := 
  by sorry

end Checkerboard

end checkerboard_black_squares_count_l4_4899


namespace tan_sum_identity_l4_4963

theorem tan_sum_identity : (1 + Real.tan (Real.pi / 180)) * (1 + Real.tan (44 * Real.pi / 180)) = 2 := 
by sorry

end tan_sum_identity_l4_4963


namespace people_landed_in_virginia_l4_4867

def initial_passengers : ℕ := 124
def texas_out : ℕ := 58
def texas_in : ℕ := 24
def north_carolina_out : ℕ := 47
def north_carolina_in : ℕ := 14
def crew_members : ℕ := 10

def final_passengers := initial_passengers - texas_out + texas_in - north_carolina_out + north_carolina_in
def total_people_landed := final_passengers + crew_members

theorem people_landed_in_virginia : total_people_landed = 67 :=
by
  sorry

end people_landed_in_virginia_l4_4867


namespace value_of_expression_l4_4158

theorem value_of_expression : 8 * (6 - 4) + 2 = 18 := by
  sorry

end value_of_expression_l4_4158


namespace find_value_l4_4936

-- Define the theorem with the given conditions and the expected result
theorem find_value (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + a * c^2 + a + b + c = 2 * (a * b + b * c + a * c)) :
  c^2017 / (a^2016 + b^2018) = 1 / 2 :=
sorry

end find_value_l4_4936


namespace perimeter_of_square_l4_4269

theorem perimeter_of_square (s : ℝ) (h : s^2 = 588) : (4 * s) = 56 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_l4_4269


namespace proof_problem_l4_4295

-- Definitions
variable (T : Type) (Sam : T)
variable (solves_all : T → Prop) (passes : T → Prop)

-- Given condition (Dr. Evans's statement)
axiom dr_evans_statement : ∀ x : T, solves_all x → passes x

-- Statement to be proven
theorem proof_problem : ¬ (passes Sam) → ¬ (solves_all Sam) :=
  by sorry

end proof_problem_l4_4295


namespace square_perimeter_ratio_l4_4724

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l4_4724


namespace compare_fractions_l4_4907

theorem compare_fractions : (31 : ℚ) / 11 > (17 : ℚ) / 14 := 
by
  sorry

end compare_fractions_l4_4907


namespace total_feed_amount_l4_4877

theorem total_feed_amount (x : ℝ) : 
  (17 * 0.18) + (x * 0.53) = (17 + x) * 0.36 → 17 + x = 35 :=
by
  intros h
  sorry

end total_feed_amount_l4_4877


namespace total_selling_price_correct_l4_4929

-- Defining the given conditions
def profit_per_meter : ℕ := 5
def cost_price_per_meter : ℕ := 100
def total_meters_sold : ℕ := 85

-- Using the conditions to define the total selling price
def total_selling_price := total_meters_sold * (cost_price_per_meter + profit_per_meter)

-- Stating the theorem without the proof
theorem total_selling_price_correct : total_selling_price = 8925 := by
  sorry

end total_selling_price_correct_l4_4929


namespace quadratic_always_positive_l4_4570

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - k + 4 > 0) ↔ -2 * Real.sqrt 3 < k ∧ k < 2 * Real.sqrt 3 := by
  sorry

end quadratic_always_positive_l4_4570


namespace real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l4_4098

-- Define z as a complex number with components dependent on m
def z (m : ℝ) : ℂ := ⟨m^2 - m, m - 1⟩

-- Statement 1: z is real iff m = 1
theorem real_z_iff (m : ℝ) : (∃ r : ℝ, z m = ⟨r, 0⟩) ↔ m = 1 := 
    sorry

-- Statement 2: z is purely imaginary iff m = 0
theorem imaginary_z_iff (m : ℝ) : (∃ i : ℝ, z m = ⟨0, i⟩ ∧ i ≠ 0) ↔ m = 0 := 
    sorry

-- Statement 3: z is in the first quadrant iff m > 1
theorem first_quadrant_z_iff (m : ℝ) : (z m).re > 0 ∧ (z m).im > 0 ↔ m > 1 := 
    sorry

end real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l4_4098


namespace percentage_shaded_l4_4889

theorem percentage_shaded (total_squares shaded_squares : ℕ) (h1 : total_squares = 5 * 5) (h2 : shaded_squares = 9) :
  (shaded_squares:ℚ) / total_squares * 100 = 36 :=
by
  sorry

end percentage_shaded_l4_4889


namespace complement_intersection_l4_4716

-- Definitions
def U : Set ℕ := {x | x ≤ 4 ∧ 0 < x}
def A : Set ℕ := {1, 4}
def B : Set ℕ := {2, 4}
def complement (s : Set ℕ) := {x | x ∈ U ∧ x ∉ s}

-- The theorem to prove
theorem complement_intersection :
  complement (A ∩ B) = {1, 2, 3} :=
by
  sorry

end complement_intersection_l4_4716


namespace range_of_a_l4_4545

def f (x a : ℝ) : ℝ := x^2 + a * x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ∧ (∃ x : ℝ, f (f x a) a = 0) → (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l4_4545


namespace gcd_40_56_l4_4609

theorem gcd_40_56 : Nat.gcd 40 56 = 8 := 
by 
  sorry

end gcd_40_56_l4_4609


namespace anne_speed_l4_4175

-- Definition of distance and time
def distance : ℝ := 6
def time : ℝ := 3

-- Statement to prove
theorem anne_speed : distance / time = 2 := by
  sorry

end anne_speed_l4_4175


namespace find_second_term_l4_4284

theorem find_second_term (A B : ℕ) (h1 : A / B = 3 / 4) (h2 : (A + 10) / (B + 10) = 4 / 5) : B = 40 :=
sorry

end find_second_term_l4_4284


namespace cone_radius_l4_4147

noncomputable def radius_of_cone (V : ℝ) (h : ℝ) : ℝ := 
  3 / Real.sqrt (Real.pi)

theorem cone_radius :
  ∀ (V h : ℝ), V = 12 → h = 4 → radius_of_cone V h = 3 / Real.sqrt (Real.pi) :=
by
  intros V h hV hv
  sorry

end cone_radius_l4_4147


namespace simplify_expr_l4_4026

open Real

theorem simplify_expr (x : ℝ) (hx : 1 ≤ x) :
  sqrt (x + 2 * sqrt (x - 1)) + sqrt (x - 2 * sqrt (x - 1)) = 
  if x ≤ 2 then 2 else 2 * sqrt (x - 1) :=
by sorry

end simplify_expr_l4_4026


namespace total_snakes_l4_4069

def People (n : ℕ) : Prop := n = 59
def OnlyDogs (n : ℕ) : Prop := n = 15
def OnlyCats (n : ℕ) : Prop := n = 10
def OnlyCatsAndDogs (n : ℕ) : Prop := n = 5
def CatsDogsSnakes (n : ℕ) : Prop := n = 3

theorem total_snakes (n_people n_dogs n_cats n_catsdogs n_catdogsnsnakes : ℕ)
  (h_people : People n_people) 
  (h_onlyDogs : OnlyDogs n_dogs)
  (h_onlyCats : OnlyCats n_cats)
  (h_onlyCatsAndDogs : OnlyCatsAndDogs n_catsdogs)
  (h_catsDogsSnakes : CatsDogsSnakes n_catdogsnsnakes) :
  n_catdogsnsnakes >= 3 :=
by
  -- Proof goes here
  sorry

end total_snakes_l4_4069


namespace vectors_parallel_x_eq_four_l4_4288

theorem vectors_parallel_x_eq_four (x : ℝ) :
  (x > 0) →
  (∃ k : ℝ, (8 + 1/2 * x, x) = k • (x + 1, 2)) →
  x = 4 :=
by
  intro h1 h2
  sorry

end vectors_parallel_x_eq_four_l4_4288


namespace mod_equiv_1043_36_mod_equiv_1_10_l4_4509

open Int

-- Define the integers involved
def a : ℤ := -1043
def m1 : ℕ := 36
def m2 : ℕ := 10

-- Theorems to prove modulo equivalence
theorem mod_equiv_1043_36 : a % m1 = 1 := by
  sorry

theorem mod_equiv_1_10 : 1 % m2 = 1 := by
  sorry

end mod_equiv_1043_36_mod_equiv_1_10_l4_4509


namespace ratio_a_to_b_l4_4237

theorem ratio_a_to_b (a b : ℝ) (h : (a - 3 * b) / (2 * a - b) = 0.14285714285714285) : a / b = 4 :=
by 
  -- The proof goes here
  sorry

end ratio_a_to_b_l4_4237


namespace complement_intersection_l4_4231

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {1, 6} :=
by
  sorry

end complement_intersection_l4_4231


namespace cows_relationship_l4_4618

theorem cows_relationship (H : ℕ) (W : ℕ) (T : ℕ) (hcows : W = 17) (tcows : T = 70) (together : H + W = T) : H = 53 :=
by
  rw [hcows, tcows] at together
  linarith
  -- sorry

end cows_relationship_l4_4618


namespace count_powers_of_2_not_4_under_2000000_l4_4827

theorem count_powers_of_2_not_4_under_2000000 :
  ∃ n, ∀ x, x < 2000000 → (∃ k, x = 2 ^ k ∧ (∀ m, x ≠ 4 ^ m)) ↔ x > 0 ∧ x < 2 ^ (n + 1) := by
  sorry

end count_powers_of_2_not_4_under_2000000_l4_4827


namespace second_and_third_finish_job_together_in_8_days_l4_4681

theorem second_and_third_finish_job_together_in_8_days
  (x y : ℕ)
  (h1 : 1/24 + 1/x + 1/y = 1/6) :
  1/x + 1/y = 1/8 :=
by sorry

end second_and_third_finish_job_together_in_8_days_l4_4681


namespace solve_for_x_l4_4678

theorem solve_for_x (x : ℝ) :
  let area_square1 := (2 * x) ^ 2
  let area_square2 := (5 * x) ^ 2
  let area_triangle := 0.5 * (2 * x) * (5 * x)
  (area_square1 + area_square2 + area_triangle = 850) → x = 5 := by
  sorry

end solve_for_x_l4_4678


namespace num_valid_pairs_l4_4114

theorem num_valid_pairs (a b : ℕ) (hb : b > a) (h_unpainted_area : ab = 3 * (a - 4) * (b - 4)) :
  (∃ (a b : ℕ), b > a ∧ ab = 3 * (a-4) * (b-4) ∧ (a-6) * (b-6) = 12 ∧ ((a, b) = (7, 18) ∨ (a, b) = (8, 12))) ∧
  (2 = 2) :=
by sorry

end num_valid_pairs_l4_4114


namespace percent_is_250_l4_4998

def part : ℕ := 150
def whole : ℕ := 60
def percent := (part : ℚ) / (whole : ℚ) * 100

theorem percent_is_250 : percent = 250 := 
by 
  sorry

end percent_is_250_l4_4998


namespace symmetric_point_proof_l4_4169

def symmetric_point (P : ℝ × ℝ) (line : ℝ → ℝ) : ℝ × ℝ := sorry

theorem symmetric_point_proof :
  symmetric_point (2, 5) (λ x => 1 - x) = (-4, -1) := sorry

end symmetric_point_proof_l4_4169


namespace marble_probability_correct_l4_4800

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end marble_probability_correct_l4_4800


namespace max_value_A_l4_4737

theorem max_value_A (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( ( (x - y) * Real.sqrt (x^2 + y^2) + (y - z) * Real.sqrt (y^2 + z^2) + (z - x) * Real.sqrt (z^2 + x^2) + Real.sqrt 2 ) / 
    ( (x - y)^2 + (y - z)^2 + (z - x)^2 + 2 ) ) ≤ 1 / Real.sqrt 2 :=
sorry

end max_value_A_l4_4737


namespace partI_inequality_solution_partII_minimum_value_l4_4711

-- Part (I)
theorem partI_inequality_solution (x : ℝ) : 
  (abs (x + 1) + abs (2 * x - 1) ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 1) :=
sorry

-- Part (II)
theorem partII_minimum_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) :
  (∀ a b c : ℝ, a + b + c = 2 ->  a > 0 -> b > 0 -> c > 0 -> 
    (1 / a + 1 / b + 1 / c) = (9 / 2)) :=
sorry

end partI_inequality_solution_partII_minimum_value_l4_4711


namespace find_z_value_l4_4029

noncomputable def y_varies_inversely_with_z (y z : ℝ) (k : ℝ) : Prop :=
  (y^4 * z^(1/4) = k)

theorem find_z_value (y z : ℝ) (k : ℝ) : 
  y_varies_inversely_with_z y z k → 
  y_varies_inversely_with_z 3 16 162 → 
  k = 162 →
  y = 6 → 
  z = 1 / 4096 := 
by 
  sorry

end find_z_value_l4_4029


namespace number_of_sides_of_polygon_l4_4669

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 40) : 
  (360 / exterior_angle) = 9 :=
by
  sorry

end number_of_sides_of_polygon_l4_4669


namespace compute_F_2_f_3_l4_4176

def f (a : ℝ) : ℝ := a^2 - 3 * a + 2
def F (a b : ℝ) : ℝ := b + a^3

theorem compute_F_2_f_3 : F 2 (f 3) = 10 :=
by
  sorry

end compute_F_2_f_3_l4_4176


namespace gcd_of_expression_l4_4472

theorem gcd_of_expression 
  (a b c d : ℕ) :
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (a - b) (c - d)) (a - c)) (b - d)) (a - d)) (b - c) = 12 :=
sorry

end gcd_of_expression_l4_4472


namespace tamara_has_30_crackers_l4_4066

theorem tamara_has_30_crackers :
  ∀ (Tamara Nicholas Marcus Mona : ℕ),
    Tamara = 2 * Nicholas →
    Marcus = 3 * Mona →
    Nicholas = Mona + 6 →
    Marcus = 27 →
    Tamara = 30 :=
by
  intros Tamara Nicholas Marcus Mona h1 h2 h3 h4
  sorry

end tamara_has_30_crackers_l4_4066


namespace hiker_distance_l4_4096

noncomputable def distance_from_start (north south east west : ℕ) : ℝ :=
  let north_south := north - south
  let east_west := east - west
  Real.sqrt (north_south ^ 2 + east_west ^ 2)

theorem hiker_distance :
  distance_from_start 24 8 15 9 = 2 * Real.sqrt 73 := by
  sorry

end hiker_distance_l4_4096


namespace least_number_of_marbles_l4_4997

def divisible_by (n : ℕ) (d : ℕ) : Prop := n % d = 0

theorem least_number_of_marbles 
  (n : ℕ)
  (h3 : divisible_by n 3)
  (h4 : divisible_by n 4)
  (h5 : divisible_by n 5)
  (h7 : divisible_by n 7)
  (h8 : divisible_by n 8) :
  n = 840 :=
sorry

end least_number_of_marbles_l4_4997


namespace interior_angle_regular_octagon_l4_4513

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l4_4513


namespace total_savings_in_2_months_l4_4576

def students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_in_month : ℕ := 4
def months : ℕ := 2

def total_contribution_per_week : ℕ := students * contribution_per_student_per_week
def total_weeks : ℕ := months * weeks_in_month
def total_savings : ℕ := total_contribution_per_week * total_weeks

theorem total_savings_in_2_months : total_savings = 480 := by
  -- Proof goes here
  sorry

end total_savings_in_2_months_l4_4576


namespace average_mileage_correct_l4_4211

def total_distance : ℕ := 150 * 2
def sedan_mileage : ℕ := 25
def hybrid_mileage : ℕ := 50
def sedan_gas_used : ℕ := 150 / sedan_mileage
def hybrid_gas_used : ℕ := 150 / hybrid_mileage
def total_gas_used : ℕ := sedan_gas_used + hybrid_gas_used
def average_gas_mileage : ℚ := total_distance / total_gas_used

theorem average_mileage_correct :
  average_gas_mileage = 33 + 1 / 3 :=
by
  sorry

end average_mileage_correct_l4_4211


namespace find_x_squared_add_y_squared_l4_4829

noncomputable def x_squared_add_y_squared (x y : ℝ) : ℝ :=
  x^2 + y^2

theorem find_x_squared_add_y_squared (x y : ℝ) 
  (h1 : x + y = 48)
  (h2 : x * y = 168) :
  x_squared_add_y_squared x y = 1968 :=
by
  sorry

end find_x_squared_add_y_squared_l4_4829


namespace gcd_sum_product_pairwise_coprime_l4_4938

theorem gcd_sum_product_pairwise_coprime 
  (a b c : ℤ) 
  (h1 : Int.gcd a b = 1)
  (h2 : Int.gcd b c = 1)
  (h3 : Int.gcd a c = 1) : 
  Int.gcd (a * b + b * c + a * c) (a * b * c) = 1 := 
sorry

end gcd_sum_product_pairwise_coprime_l4_4938


namespace range_of_a_l4_4273

open Real

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → -1 ≤ a ∧ a ≤ 3 :=
by
  intro h
  -- insert the actual proof here
  sorry

end range_of_a_l4_4273


namespace exists_m_such_that_m_poly_is_zero_mod_p_l4_4902

theorem exists_m_such_that_m_poly_is_zero_mod_p (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 7 = 1) :
  ∃ m : ℕ, m > 0 ∧ (m^3 + m^2 - 2*m - 1) % p = 0 := 
sorry

end exists_m_such_that_m_poly_is_zero_mod_p_l4_4902


namespace number_divided_by_005_l4_4765

theorem number_divided_by_005 (number : ℝ) (h : number / 0.05 = 1500) : number = 75 :=
sorry

end number_divided_by_005_l4_4765


namespace abs_add_opposite_signs_l4_4044

theorem abs_add_opposite_signs (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a * b < 0) : |a + b| = 1 := 
sorry

end abs_add_opposite_signs_l4_4044


namespace no_solutions_to_cubic_sum_l4_4647

theorem no_solutions_to_cubic_sum (x y z : ℤ) : 
    ¬ (x^3 + y^3 = z^3 + 4) :=
by 
  sorry

end no_solutions_to_cubic_sum_l4_4647


namespace count_integers_congruent_to_7_mod_13_l4_4346

theorem count_integers_congruent_to_7_mod_13 : 
  (∃ (n : ℕ), ∀ x, (1 ≤ x ∧ x < 500 ∧ x % 13 = 7) → x = 7 + 13 * n ∧ n < 38) :=
sorry

end count_integers_congruent_to_7_mod_13_l4_4346


namespace minimum_value_is_1_l4_4503

def minimum_value_expression (x y : ℝ) : ℝ :=
  x^2 + y^2 - 8*x + 6*y + 26

theorem minimum_value_is_1 (x y : ℝ) (h : x ≥ 4) : 
  minimum_value_expression x y ≥ 1 :=
by {
  sorry
}

end minimum_value_is_1_l4_4503


namespace total_amount_l4_4898

theorem total_amount (T_pq r : ℝ) (h1 : r = 2/3 * T_pq) (h2 : r = 1600) : T_pq + r = 4000 :=
by
  -- proof skipped
  sorry

end total_amount_l4_4898


namespace range_of_a_l4_4789

theorem range_of_a 
  (x y a : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy : x + y + 3 = x * y) 
  (h_a : ∀ x y : ℝ, (x + y)^2 - a * (x + y) + 1 ≥ 0) :
  a ≤ 37 / 6 := 
sorry

end range_of_a_l4_4789


namespace find_height_of_tank_A_l4_4250

noncomputable def height_of_tank_A (C_A C_B h_B ratio V_ratio : ℝ) : ℝ :=
  let r_A := C_A / (2 * Real.pi)
  let r_B := C_B / (2 * Real.pi)
  let V_A := Real.pi * (r_A ^ 2) * ratio
  let V_B := Real.pi * (r_B ^ 2) * h_B
  (V_ratio * V_B) / (Real.pi * (r_A ^ 2))

theorem find_height_of_tank_A :
  height_of_tank_A 8 10 8 10 0.8000000000000001 = 10 :=
by
  sorry

end find_height_of_tank_A_l4_4250


namespace second_term_is_neg_12_l4_4022

-- Define the problem conditions
variables {a d : ℤ}
axiom tenth_term : a + 9 * d = 20
axiom eleventh_term : a + 10 * d = 24

-- Define the second term calculation
def second_term (a d : ℤ) := a + d

-- The problem statement: Prove that the second term is -12 given the conditions
theorem second_term_is_neg_12 : second_term a d = -12 :=
by sorry

end second_term_is_neg_12_l4_4022


namespace composite_sum_l4_4333

theorem composite_sum (m n : ℕ) (h : 88 * m = 81 * n) : ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ (m + n) = p * q :=
by sorry

end composite_sum_l4_4333


namespace find_ordered_pair_l4_4569

theorem find_ordered_pair :
  ∃ (x y : ℚ), 7 * x - 3 * y = 6 ∧ 4 * x + 5 * y = 23 ∧ 
               x = 99 / 47 ∧ y = 137 / 47 :=
by
  sorry

end find_ordered_pair_l4_4569


namespace base_7_is_good_number_l4_4778

def is_good_number (m: ℕ) : Prop :=
  ∃ (p: ℕ) (n: ℕ), Prime p ∧ n ≥ 2 ∧ m = p^n

theorem base_7_is_good_number : 
  ∀ b: ℕ, (is_good_number (b^2 - (2 * b + 3))) → b = 7 :=
by
  intro b h
  sorry

end base_7_is_good_number_l4_4778


namespace root_iff_coeff_sum_zero_l4_4934

theorem root_iff_coeff_sum_zero (a b c : ℝ) :
    (a * 1^2 + b * 1 + c = 0) ↔ (a + b + c = 0) := sorry

end root_iff_coeff_sum_zero_l4_4934


namespace problem_solution_l4_4657

def otimes (a b : ℚ) : ℚ := (a ^ 3) / (b ^ 2)

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (-2016) / 729 := by
  sorry

end problem_solution_l4_4657


namespace probability_sum_15_l4_4101

/-- If three standard 6-faced dice are rolled, the probability that the sum of the face-up integers is 15 is 5/72. -/
theorem probability_sum_15 : (1 / 6 : ℚ) ^ 3 * 3 + (1 / 6 : ℚ) ^ 3 * 6 = 5 / 72 := by 
  sorry

end probability_sum_15_l4_4101


namespace range_of_a_l4_4744

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) →
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l4_4744


namespace beads_bracelet_rotational_symmetry_l4_4672

theorem beads_bracelet_rotational_symmetry :
  let n := 8
  let factorial := Nat.factorial
  (factorial n / n = 5040) := by
  sorry

end beads_bracelet_rotational_symmetry_l4_4672


namespace proof_problem_l4_4121

noncomputable def aₙ (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d
noncomputable def Sₙ (a₁ d : ℝ) (n : ℕ) := n * a₁ + (n * (n - 1) / 2) * d

def given_conditions (Sₙ : ℕ → ℝ) : Prop :=
  Sₙ 10 = 0 ∧ Sₙ 15 = 25

theorem proof_problem (a₁ d : ℝ) (Sₙ : ℕ → ℝ)
  (h₁ : Sₙ 10 = 0) (h₂ : Sₙ 15 = 25) :
  (aₙ a₁ d 5 = -1/3) ∧
  (∀ n, Sₙ n = (1 / 3) * (n ^ 2 - 10 * n) → n = 5) ∧
  (∀ n, n * Sₙ n = (n ^ 3 / 3) - (10 * n ^ 2 / 3) → min (n * Sₙ n) = -49) ∧
  (¬ ∃ n, (Sₙ n / n) > 0) :=
sorry

end proof_problem_l4_4121


namespace train_pass_time_l4_4790

-- Assuming conversion factor, length of the train, and speed in km/hr
def conversion_factor := 1000 / 3600
def train_length := 280
def speed_km_hr := 36

-- Defining speed in m/s
def speed_m_s := speed_km_hr * conversion_factor

-- Defining the time to pass a tree
def time_to_pass_tree := train_length / speed_m_s

-- Theorem statement
theorem train_pass_time : time_to_pass_tree = 28 := by
  sorry

end train_pass_time_l4_4790


namespace Xiaoyong_age_solution_l4_4085

theorem Xiaoyong_age_solution :
  ∃ (x y : ℕ), 1 ≤ y ∧ y < x ∧ x < 20 ∧ 2 * x + 5 * y = 97 ∧ x = 16 ∧ y = 13 :=
by
  -- You should provide a suitable proof here
  sorry

end Xiaoyong_age_solution_l4_4085


namespace largest_share_of_partner_l4_4149

theorem largest_share_of_partner 
    (ratios : List ℕ := [2, 3, 4, 4, 6])
    (total_profit : ℕ := 38000) :
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    largest_share = 12000 :=
by
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    have h1 : total_parts = 19 := by
        sorry
    have h2 : part_value = 2000 := by
        sorry
    have h3 : List.maximum ratios = 6 := by
        sorry
    have h4 : largest_share = 12000 := by
        sorry
    exact h4


end largest_share_of_partner_l4_4149


namespace hyperbola_parabola_focus_l4_4239

open Classical

theorem hyperbola_parabola_focus :
  ∃ a : ℝ, (a > 0) ∧ (∃ c > 0, (c = 2) ∧ (a^2 + 3 = c^2)) → a = 1 :=
sorry

end hyperbola_parabola_focus_l4_4239


namespace school_anniversary_problem_l4_4258

theorem school_anniversary_problem
    (total_cost : ℕ)
    (cost_commemorative_albums cost_bone_china_cups : ℕ)
    (num_commemorative_albums num_bone_china_cups : ℕ)
    (price_commemorative_album price_bone_china_cup : ℕ)
    (H1 : total_cost = 312000)
    (H2 : cost_commemorative_albums + cost_bone_china_cups = total_cost)
    (H3 : cost_commemorative_albums = 3 * cost_bone_china_cups)
    (H4 : price_commemorative_album = 3 / 2 * price_bone_china_cup)
    (H5 : num_bone_china_cups = 4 * num_commemorative_albums + 1600) :
    (cost_commemorative_albums = 72000 ∧ cost_bone_china_cups = 240000) ∧
    (price_commemorative_album = 45 ∧ price_bone_china_cup = 30) :=
by
  sorry

end school_anniversary_problem_l4_4258


namespace sample_size_correct_l4_4883

-- Define the total number of students in a certain grade.
def total_students : ℕ := 500

-- Define the number of students selected for statistical analysis.
def selected_students : ℕ := 30

-- State the theorem to prove the selected students represent the sample size.
theorem sample_size_correct : selected_students = 30 := by
  -- The proof would go here, but we use sorry to indicate it is skipped.
  sorry

end sample_size_correct_l4_4883


namespace negation_example_l4_4717

theorem negation_example : (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end negation_example_l4_4717


namespace tan_of_alpha_l4_4153

noncomputable def point_P : ℝ × ℝ := (1, -2)

theorem tan_of_alpha (α : ℝ) (h : ∃ (P : ℝ × ℝ), P = point_P ∧ P.2 / P.1 = -2) :
  Real.tan α = -2 :=
sorry

end tan_of_alpha_l4_4153


namespace savings_on_cheapest_flight_l4_4192

theorem savings_on_cheapest_flight :
  let delta_price := 850
  let delta_discount := 0.20
  let united_price := 1100
  let united_discount := 0.30
  let delta_final_price := delta_price - delta_price * delta_discount
  let united_final_price := united_price - united_price * united_discount
  delta_final_price < united_final_price →
  united_final_price - delta_final_price = 90 :=
by
  sorry

end savings_on_cheapest_flight_l4_4192


namespace percent_of_volume_filled_by_cubes_l4_4186

theorem percent_of_volume_filled_by_cubes :
  let box_width := 8
  let box_height := 6
  let box_length := 12
  let cube_size := 2
  let box_volume := box_width * box_height * box_length
  let cube_volume := cube_size ^ 3
  let num_cubes := (box_width / cube_size) * (box_height / cube_size) * (box_length / cube_size)
  let cubes_volume := num_cubes * cube_volume
  (cubes_volume / box_volume : ℝ) * 100 = 100 := by
  sorry

end percent_of_volume_filled_by_cubes_l4_4186


namespace points_on_same_side_of_line_l4_4132

theorem points_on_same_side_of_line (m : ℝ) :
  (2 * 0 + 0 + m > 0 ∧ 2 * -1 + 1 + m > 0) ∨ 
  (2 * 0 + 0 + m < 0 ∧ 2 * -1 + 1 + m < 0) ↔ 
  (m < 0 ∨ m > 1) :=
by
  sorry

end points_on_same_side_of_line_l4_4132


namespace probability_at_least_one_passes_l4_4604

theorem probability_at_least_one_passes (prob_pass : ℚ) (prob_fail : ℚ) (p_all_fail: ℚ):
  (prob_pass = 1/3) →
  (prob_fail = 1 - prob_pass) →
  (p_all_fail = prob_fail ^ 3) →
  (1 - p_all_fail = 19/27) :=
by
  intros hpp hpf hpaf
  sorry

end probability_at_least_one_passes_l4_4604


namespace a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l4_4864

noncomputable def a_0 (n : ℕ) : ℕ := 2^n
noncomputable def S_n (n : ℕ) : ℕ := 3^n - 2^n
noncomputable def T_n (n : ℕ) : ℕ := (n - 2) * 2^n + 2 * n^2

theorem a_0_eq_2_pow_n (n : ℕ) (h : n > 0) : a_0 n = 2^n := sorry

theorem S_n_eq_3_pow_n_minus_2_pow_n (n : ℕ) (h : n > 0) : S_n n = 3^n - 2^n := sorry

theorem S_n_magnitude_comparison : 
  ∀ (n : ℕ), 
    (n = 1 → S_n n > T_n n) ∧
    (n = 2 ∨ n = 3 → S_n n < T_n n) ∧
    (n ≥ 4 → S_n n > T_n n) := sorry

end a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l4_4864


namespace breakfast_problem_probability_l4_4357

def are_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

theorem breakfast_problem_probability : 
  ∃ m n : ℕ, are_relatively_prime m n ∧ 
  (1 / 1 * 9 / 11 * 6 / 10 * 1 / 3) * 1 = 9 / 55 ∧ m + n = 64 :=
by
  sorry

end breakfast_problem_probability_l4_4357


namespace number_of_oranges_l4_4910

def bananas : ℕ := 7
def apples : ℕ := 2 * bananas
def pears : ℕ := 4
def grapes : ℕ := apples / 2
def total_fruits : ℕ := 40

theorem number_of_oranges : total_fruits - (bananas + apples + pears + grapes) = 8 :=
by sorry

end number_of_oranges_l4_4910


namespace floor_x_plus_x_eq_13_div_3_l4_4109

-- Statement representing the mathematical problem
theorem floor_x_plus_x_eq_13_div_3 (x : ℚ) (h : ⌊x⌋ + x = 13/3) : x = 7/3 := 
sorry

end floor_x_plus_x_eq_13_div_3_l4_4109


namespace total_distance_covered_l4_4263

theorem total_distance_covered :
  let t1 := 30 / 60 -- time in hours for first walking session
  let s1 := 3       -- speed in mph for first walking session
  let t2 := 20 / 60 -- time in hours for running session
  let s2 := 8       -- speed in mph for running session
  let t3 := 10 / 60 -- time in hours for second walking session
  let s3 := 2       -- speed in mph for second walking session
  let d1 := s1 * t1 -- distance for first walking session
  let d2 := s2 * t2 -- distance for running session
  let d3 := s3 * t3 -- distance for second walking session
  d1 + d2 + d3 = 4.5 :=
by
  sorry

end total_distance_covered_l4_4263


namespace calculate_chord_length_l4_4179

noncomputable def chord_length_of_tangent (r1 r2 : ℝ) (c : ℝ) : Prop :=
  r1^2 - r2^2 = 18 ∧ (c / 2)^2 = 18

theorem calculate_chord_length (r1 r2 : ℝ) (h : chord_length_of_tangent r1 r2 (6 * Real.sqrt 2)) :
  (6 * Real.sqrt 2) = 6 * Real.sqrt 2 :=
by
  sorry

end calculate_chord_length_l4_4179


namespace product_of_coordinates_of_D_l4_4553

theorem product_of_coordinates_of_D 
  (x y : ℝ)
  (midpoint_x : (5 + x) / 2 = 4)
  (midpoint_y : (3 + y) / 2 = 7) : 
  x * y = 33 := 
by 
  sorry

end product_of_coordinates_of_D_l4_4553


namespace distance_between_points_on_line_l4_4528

theorem distance_between_points_on_line 
  (p q r s : ℝ)
  (line_eq : q = 2 * p + 3) 
  (s_eq : s = 2 * r + 6) :
  Real.sqrt ((r - p)^2 + (s - q)^2) = Real.sqrt (5 * (r - p)^2 + 12 * (r - p) + 9) :=
sorry

end distance_between_points_on_line_l4_4528


namespace machine_a_sprockets_per_hour_l4_4280

theorem machine_a_sprockets_per_hour (s h : ℝ)
    (H1 : 1.1 * s * h = 550)
    (H2 : s * (h + 10) = 550) : s = 5 := by
  sorry

end machine_a_sprockets_per_hour_l4_4280


namespace not_perfect_square_l4_4639

theorem not_perfect_square (a b : ℤ) (h1 : a > b) (h2 : Int.gcd (ab - 1) (a + b) = 1) (h3 : Int.gcd (ab + 1) (a - b) = 1) :
  ¬ ∃ c : ℤ, (a + b)^2 + (ab - 1)^2 = c^2 := 
  sorry

end not_perfect_square_l4_4639


namespace fraction_calculation_l4_4875

theorem fraction_calculation :
  let a := (1 / 2) + (1 / 3)
  let b := (2 / 7) + (1 / 4)
  ((a / b) * (3 / 5)) = (14 / 15) :=
by
  sorry

end fraction_calculation_l4_4875


namespace find_n_for_divisibility_l4_4786

def digit_sum_odd_positions := 8 + 4 + 5 + 6 -- The sum of the digits in odd positions
def digit_sum_even_positions (n : ℕ) := 5 + n + 2 -- The sum of the digits in even positions

def is_divisible_by_11 (n : ℕ) := (digit_sum_odd_positions - digit_sum_even_positions n) % 11 = 0

theorem find_n_for_divisibility : is_divisible_by_11 5 :=
by
  -- Proof would go here (but according to the instructions, we'll insert a placeholder)
  sorry

end find_n_for_divisibility_l4_4786


namespace periodic_even_l4_4781

noncomputable def f : ℝ → ℝ := sorry  -- We assume the existence of such a function.

variables {α β : ℝ}  -- acute angles of a right triangle

-- Function properties
theorem periodic_even (h_periodic: ∀ x: ℝ, f (x + 2) = f x)
  (h_even: ∀ x: ℝ, f (-x) = f x)
  (h_decreasing: ∀ x y: ℝ, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f x > f y)
  (h_inc_interval_0_1: ∀ x y: ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y)
  (ha: 0 < α ∧ α < π / 2)
  (hb: 0 < β ∧ β < π / 2)
  (h_sum_right_triangle: α + β = π / 2): f (Real.sin α) > f (Real.cos β) :=
sorry

end periodic_even_l4_4781


namespace diagonal_intersection_probability_decagon_l4_4445

noncomputable def probability_diagonal_intersection_in_decagon : ℚ :=
  let vertices := 10
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let total_pairs_of_diagonals := total_diagonals * (total_diagonals - 1) / 2
  let total_intersecting_pairs := (vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24
  total_intersecting_pairs / total_pairs_of_diagonals

theorem diagonal_intersection_probability_decagon (h : probability_diagonal_intersection_in_decagon = 42 / 119) : 
  probability_diagonal_intersection_in_decagon = 42 / 119 :=
sorry

end diagonal_intersection_probability_decagon_l4_4445


namespace length_of_bridge_l4_4729

theorem length_of_bridge
  (walking_speed_km_hr : ℝ) (time_minutes : ℝ) (length_bridge : ℝ) 
  (h1 : walking_speed_km_hr = 5) 
  (h2 : time_minutes = 15) 
  (h3 : length_bridge = 1250) : 
  length_bridge = (walking_speed_km_hr * 1000 / 60) * time_minutes := 
by 
  sorry

end length_of_bridge_l4_4729


namespace geometric_sum_S15_l4_4135

noncomputable def S (n : ℕ) : ℝ := sorry  -- Assume S is defined for the sequence sum

theorem geometric_sum_S15 (S_5 S_10 : ℝ) (h1 : S_5 = 5) (h2 : S_10 = 30) : 
    S 15 = 155 := 
by 
  -- Placeholder for geometric sequence proof
  sorry

end geometric_sum_S15_l4_4135


namespace find_positive_integers_n_l4_4940

open Real Int

noncomputable def satisfies_conditions (x y z : ℝ) (n : ℕ) : Prop :=
  sqrt x + sqrt y + sqrt z = 1 ∧ 
  (∃ m : ℤ, sqrt (x + n) + sqrt (y + n) + sqrt (z + n) = m)

theorem find_positive_integers_n (n : ℕ) :
  (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ satisfies_conditions x y z n) ↔
  (∃ k : ℤ, k ≥ 1 ∧ (k % 9 = 1 ∨ k % 9 = 8) ∧ n = (k^2 - 1) / 9) :=
by
  sorry

end find_positive_integers_n_l4_4940


namespace cubic_representation_l4_4345

variable (a b : ℝ) (x : ℝ)
variable (v u w : ℝ)

axiom h1 : 6.266 * x^3 - 3 * a * x^2 + (3 * a^2 - b) * x - (a^3 - a * b) = 0
axiom h2 : b ≥ 0

theorem cubic_representation : v = a ∧ u = a ∧ w^2 = b → 
  6.266 * x^3 - 3 * v * x^2 + (3 * u^2 - w^2) * x - (u^3 - u * w^2) = 0 :=
by
  sorry

end cubic_representation_l4_4345


namespace compound_interest_calculation_l4_4874

theorem compound_interest_calculation :
  let P_SI := 1750.0000000000018
  let r_SI := 0.08
  let t_SI := 3
  let r_CI := 0.10
  let t_CI := 2
  let SI := P_SI * r_SI * t_SI
  let CI (P_CI : ℝ) := P_CI * ((1 + r_CI) ^ t_CI - 1)
  (SI = 420.0000000000004) →
  (SI = (1 / 2) * CI P_CI) →
  P_CI = 4000.000000000004 :=
by
  intros P_SI r_SI t_SI r_CI t_CI SI CI h1 h2
  sorry

end compound_interest_calculation_l4_4874


namespace t_of_polynomial_has_factor_l4_4010

theorem t_of_polynomial_has_factor (t : ℤ) :
  (∃ a b : ℤ, x ^ 3 - x ^ 2 - 7 * x + t = (x + 1) * (x ^ 2 + a * x + b)) → t = -5 :=
by
  sorry

end t_of_polynomial_has_factor_l4_4010


namespace fixed_point_and_max_distance_eqn_l4_4458

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

end fixed_point_and_max_distance_eqn_l4_4458


namespace Jirina_number_l4_4542

theorem Jirina_number (a b c d : ℕ) (h_abcd : 1000 * a + 100 * b + 10 * c + d = 1468) :
  (1000 * a + 100 * b + 10 * c + d) +
  (1000 * a + 100 * d + 10 * c + b) = 3332 ∧ 
  (1000 * a + 100 * b + 10 * c + d)+
  (1000 * c + 100 * b + 10 * a + d) = 7886 :=
by
  sorry

end Jirina_number_l4_4542


namespace negation_of_p_l4_4450

def p := ∀ x : ℝ, Real.sin x ≤ 1

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, Real.sin x > 1 := 
by 
  sorry

end negation_of_p_l4_4450


namespace find_y_of_series_eq_92_l4_4732

theorem find_y_of_series_eq_92 (y : ℝ) (h : (∑' n, (2 + 5 * n) * y^n) = 92) (converge : abs y < 1) : y = 18 / 23 :=
sorry

end find_y_of_series_eq_92_l4_4732


namespace union_A_B_eq_neg2_neg1_0_l4_4143

def setA : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}
def setB : Set ℤ := {-2, -1}

theorem union_A_B_eq_neg2_neg1_0 : (setA ∪ setB) = ({-2, -1, 0} : Set ℤ) :=
by
  sorry

end union_A_B_eq_neg2_neg1_0_l4_4143


namespace units_digit_sum_is_9_l4_4434

-- Define the units function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def x := 42 ^ 2
def y := 25 ^ 3

-- Define variables for the units digits of x and y
def units_digit_x := units_digit x
def units_digit_y := units_digit y

-- Define the problem statement to be proven
theorem units_digit_sum_is_9 : units_digit (x + y) = 9 :=
by sorry

end units_digit_sum_is_9_l4_4434


namespace smallest_value_l4_4602

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end smallest_value_l4_4602


namespace pumpkins_eaten_l4_4389

theorem pumpkins_eaten (initial: ℕ) (left: ℕ) (eaten: ℕ) (h1 : initial = 43) (h2 : left = 20) : eaten = 23 :=
by {
  -- We are skipping the proof as per the requirement
  sorry
}

end pumpkins_eaten_l4_4389


namespace smallest_total_squares_l4_4928

theorem smallest_total_squares (n : ℕ) (h : 4 * n - 4 = 2 * n) : n^2 = 4 :=
by
  sorry

end smallest_total_squares_l4_4928


namespace problem_1_problem_2_l4_4884

-- Problem (1): Proving the solutions for \( x^2 - 3x = 0 \)
theorem problem_1 : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ (x = 0 ∨ x = 3) :=
by
  intro x
  sorry

-- Problem (2): Proving the solutions for \( 5x + 2 = 3x^2 \)
theorem problem_2 : ∀ x : ℝ, 5 * x + 2 = 3 * x^2 ↔ (x = -1/3 ∨ x = 2) :=
by
  intro x
  sorry

end problem_1_problem_2_l4_4884


namespace original_phone_number_eq_l4_4712

theorem original_phone_number_eq :
  ∃ (a b c d e f : ℕ), 
    (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 282500) ∧
    (1000000 * 2 + 100000 * a + 10000 * 8 + 1000 * b + 100 * c + 10 * d + e = 81 * (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)) ∧
    (0 ≤ a ∧ a ≤ 9) ∧
    (0 ≤ b ∧ b ≤ 9) ∧
    (0 ≤ c ∧ c ≤ 9) ∧
    (0 ≤ d ∧ d ≤ 9) ∧
    (0 ≤ e ∧ e ≤ 9) ∧
    (0 ≤ f ∧ f ≤ 9) :=
sorry

end original_phone_number_eq_l4_4712


namespace total_songs_time_l4_4338

-- Definitions of durations for each radio show
def duration_show1 : ℕ := 180
def duration_show2 : ℕ := 240
def duration_show3 : ℕ := 120

-- Definitions of talking segments for each show
def talking_segments_show1 : ℕ := 3 * 10  -- 3 segments, 10 minutes each
def talking_segments_show2 : ℕ := 4 * 15  -- 4 segments, 15 minutes each
def talking_segments_show3 : ℕ := 2 * 8   -- 2 segments, 8 minutes each

-- Definitions of ad breaks for each show
def ad_breaks_show1 : ℕ := 5 * 5  -- 5 breaks, 5 minutes each
def ad_breaks_show2 : ℕ := 6 * 4  -- 6 breaks, 4 minutes each
def ad_breaks_show3 : ℕ := 3 * 6  -- 3 breaks, 6 minutes each

-- Function to calculate time spent on songs for a given show
def time_spent_on_songs (duration talking ad_breaks : ℕ) : ℕ :=
  duration - talking - ad_breaks

-- Total time spent on songs for all three shows
def total_time_spent_on_songs : ℕ :=
  time_spent_on_songs duration_show1 talking_segments_show1 ad_breaks_show1 +
  time_spent_on_songs duration_show2 talking_segments_show2 ad_breaks_show2 +
  time_spent_on_songs duration_show3 talking_segments_show3 ad_breaks_show3

-- The theorem we want to prove
theorem total_songs_time : total_time_spent_on_songs = 367 := 
  sorry

end total_songs_time_l4_4338


namespace island_inhabitants_even_l4_4081

theorem island_inhabitants_even 
  (total : ℕ) 
  (knights liars : ℕ)
  (H : total = knights + liars)
  (H1 : ∃ (knk : Prop), (knk → (knights % 2 = 0)) ∧ (¬knk → (knights % 2 = 1)))
  (H2 : ∃ (lkr : Prop), (lkr → (liars % 2 = 1)) ∧ (¬lkr → (liars % 2 = 0)))
  : (total % 2 = 0) := sorry

end island_inhabitants_even_l4_4081


namespace num_factors_of_90_multiple_of_6_l4_4536

def is_factor (m n : ℕ) : Prop := n % m = 0
def is_multiple_of (m n : ℕ) : Prop := n % m = 0

theorem num_factors_of_90_multiple_of_6 : 
  ∃ (count : ℕ), count = 4 ∧ ∀ x, is_factor x 90 → is_multiple_of 6 x → x > 0 :=
sorry

end num_factors_of_90_multiple_of_6_l4_4536


namespace f_of_8_l4_4270

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom function_property : ∀ x : ℝ, f (x + 2) = -1 / f (x)

-- Statement to prove
theorem f_of_8 : f 8 = 0 :=
sorry

end f_of_8_l4_4270


namespace travel_distance_correct_l4_4057

noncomputable def traveler_distance : ℝ :=
  let x1 : ℝ := -4
  let y1 : ℝ := 0
  let x2 : ℝ := x1 + 5 * Real.cos (-(Real.pi / 3))
  let y2 : ℝ := y1 + 5 * Real.sin (-(Real.pi / 3))
  let x3 : ℝ := x2 + 2
  let y3 : ℝ := y2
  Real.sqrt (x3^2 + y3^2)

theorem travel_distance_correct : traveler_distance = Real.sqrt 19 := by
  sorry

end travel_distance_correct_l4_4057


namespace main_world_population_transition_l4_4046

noncomputable def world_population_reproduction_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) : Prop :=
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional"

theorem main_world_population_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) :
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional" :=
by
  sorry

end main_world_population_transition_l4_4046


namespace find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l4_4130

-- Define the coordinate functions for point P
def coord_x (m : ℚ) : ℚ := 3 * m + 6
def coord_y (m : ℚ) : ℚ := m - 3

-- Definitions for each condition
def condition1 (m : ℚ) : Prop := coord_x m = coord_y m
def condition2 (m : ℚ) : Prop := coord_y m = coord_x m + 5
def condition3 (m : ℚ) : Prop := coord_x m = 3

-- Proof statements for the coordinates based on each condition
theorem find_coordinates_condition1 : 
  ∃ m, condition1 m ∧ coord_x m = -7.5 ∧ coord_y m = -7.5 :=
by sorry

theorem find_coordinates_condition2 : 
  ∃ m, condition2 m ∧ coord_x m = -15 ∧ coord_y m = -10 :=
by sorry

theorem find_coordinates_condition3 : 
  ∃ m, condition3 m ∧ coord_x m = 3 ∧ coord_y m = -4 :=
by sorry

end find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l4_4130


namespace modulo_remainder_product_l4_4930

theorem modulo_remainder_product :
  let a := 2022
  let b := 2023
  let c := 2024
  let d := 2025
  let n := 17
  (a * b * c * d) % n = 0 :=
by
  sorry

end modulo_remainder_product_l4_4930


namespace inscribed_squares_equilateral_triangle_l4_4845

theorem inscribed_squares_equilateral_triangle (a b c h_a h_b h_c : ℝ) 
  (h1 : a * h_a / (a + h_a) = b * h_b / (b + h_b))
  (h2 : b * h_b / (b + h_b) = c * h_c / (c + h_c)) :
  a = b ∧ b = c ∧ h_a = h_b ∧ h_b = h_c :=
sorry

end inscribed_squares_equilateral_triangle_l4_4845


namespace min_value_f_l4_4262

theorem min_value_f (x : ℝ) (h : 0 < x) : 
  ∃ c: ℝ, c = 2.5 ∧ (∀ x, 0 < x → x^2 + 1 / x^2 + 1 / (x^2 + 1 / x^2) ≥ c) :=
by sorry

end min_value_f_l4_4262


namespace quadratic_inequality_ab_l4_4507

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set -1 < x < 1/3,
    prove that ab = 6. -/
theorem quadratic_inequality_ab (a b : ℝ) (h1 : ∀ x, -1 < x ∧ x < 1 / 3 → a * x ^ 2 + b * x + 1 > 0):
  a * b = 6 := 
sorry

end quadratic_inequality_ab_l4_4507


namespace not_recurring_decimal_l4_4746

-- Definitions based on the provided conditions
def is_recurring_decimal (x : ℝ) : Prop :=
  ∃ d m n : ℕ, d ≠ 0 ∧ (x * d) % 10 ^ n = m

-- Condition: 0.89898989
def number_0_89898989 : ℝ := 0.89898989

-- Proof statement to show 0.89898989 is not a recurring decimal
theorem not_recurring_decimal : ¬ is_recurring_decimal number_0_89898989 :=
sorry

end not_recurring_decimal_l4_4746


namespace mike_total_payment_l4_4152

def camera_initial_cost : ℝ := 4000
def camera_increase_rate : ℝ := 0.30
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200
def sales_tax_rate : ℝ := 0.08

def new_camera_cost := camera_initial_cost * (1 + camera_increase_rate)
def discounted_lens_cost := lens_initial_cost - lens_discount
def total_purchase_before_tax := new_camera_cost + discounted_lens_cost
def sales_tax := total_purchase_before_tax * sales_tax_rate
def total_purchase_with_tax := total_purchase_before_tax + sales_tax

theorem mike_total_payment : total_purchase_with_tax = 5832 := by
  sorry

end mike_total_payment_l4_4152


namespace two_fruits_probability_l4_4191

noncomputable def prob_exactly_two_fruits : ℚ := 10 / 9

theorem two_fruits_probability :
  (∀ (f : ℕ → ℝ), (f 0 = 1/3) ∧ (f 1 = 1/3) ∧ (f 2 = 1/3) ∧
   (∃ f1 f2 f3, f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    (f1 + f2 + f3 = prob_exactly_two_fruits))) :=
sorry

end two_fruits_probability_l4_4191


namespace percent_voters_for_A_l4_4479

-- Definitions from conditions
def total_voters : ℕ := 100
def percent_democrats : ℝ := 0.70
def percent_republicans : ℝ := 0.30
def percent_dems_for_A : ℝ := 0.80
def percent_reps_for_A : ℝ := 0.30

-- Calculations based on definitions
def num_democrats := total_voters * percent_democrats
def num_republicans := total_voters * percent_republicans
def dems_for_A := num_democrats * percent_dems_for_A
def reps_for_A := num_republicans * percent_reps_for_A
def total_for_A := dems_for_A + reps_for_A

-- Proof problem statement
theorem percent_voters_for_A : (total_for_A / total_voters) * 100 = 65 :=
by
  sorry

end percent_voters_for_A_l4_4479


namespace numerical_form_463001_l4_4079

theorem numerical_form_463001 : 463001 = 463001 := by
  rfl

end numerical_form_463001_l4_4079


namespace calculate_number_of_girls_l4_4941

-- Definitions based on the conditions provided
def ratio_girls_to_boys : ℕ := 3
def ratio_boys_to_girls : ℕ := 4
def total_students : ℕ := 35

-- The proof statement
theorem calculate_number_of_girls (k : ℕ) (hk : ratio_girls_to_boys * k + ratio_boys_to_girls * k = total_students) :
  ratio_girls_to_boys * k = 15 :=
by sorry

end calculate_number_of_girls_l4_4941


namespace positive_reals_condition_l4_4888

theorem positive_reals_condition (a : ℝ) (h_pos : 0 < a) : a < 2 :=
by
  -- Problem conditions:
  -- There exists a positive integer n and n pairwise disjoint infinite sets A_i
  -- such that A_1 ∪ ... ∪ A_n = ℕ* and for any two numbers b > c in each A_i,
  -- b - c ≥ a^i.

  sorry

end positive_reals_condition_l4_4888


namespace remembers_umbrella_prob_l4_4208

theorem remembers_umbrella_prob 
    (P_forgets : ℚ) 
    (h_forgets : P_forgets = 5 / 8) : 
    ∃ P_remembers : ℚ, P_remembers = 3 / 8 := 
by
    sorry

end remembers_umbrella_prob_l4_4208


namespace solution_set_of_inequality_l4_4595

theorem solution_set_of_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (a - x) * (x - 1 / a) > 0} = {x : ℝ | a < x ∧ x < 1 / a} :=
sorry

end solution_set_of_inequality_l4_4595


namespace checkered_triangle_division_l4_4757

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l4_4757


namespace original_bill_l4_4572

theorem original_bill (n : ℕ) (d : ℝ) (p : ℝ) (B : ℝ) (h1 : n = 5) (h2 : d = 0.06) (h3 : p = 18.8)
  (h4 : 0.94 * B = n * p) :
  B = 100 :=
sorry

end original_bill_l4_4572


namespace notebook_cost_l4_4065

theorem notebook_cost (n c : ℝ) (h1 : n + c = 2.50) (h2 : n = c + 2) : n = 2.25 :=
by
  sorry

end notebook_cost_l4_4065


namespace loss_per_metre_l4_4840

theorem loss_per_metre
  (total_metres : ℕ)
  (selling_price : ℕ)
  (cost_price_per_m: ℕ)
  (selling_price_total : selling_price = 18000)
  (cost_price_per_m_def : cost_price_per_m = 95)
  (total_metres_def : total_metres = 200) :
  ((cost_price_per_m * total_metres - selling_price) / total_metres) = 5 :=
by
  sorry

end loss_per_metre_l4_4840


namespace hermans_breakfast_cost_l4_4801

-- Define the conditions
def meals_per_day : Nat := 4
def days_per_week : Nat := 5
def cost_per_meal : Nat := 4
def total_weeks : Nat := 16

-- Define the statement to prove
theorem hermans_breakfast_cost :
  (meals_per_day * days_per_week * cost_per_meal * total_weeks) = 1280 := by
  sorry

end hermans_breakfast_cost_l4_4801


namespace addition_example_l4_4772

theorem addition_example : 300 + 2020 + 10001 = 12321 := 
by 
  sorry

end addition_example_l4_4772


namespace divisor_of_first_division_l4_4474

theorem divisor_of_first_division (n d : ℕ) (hn_pos : 0 < n)
  (h₁ : (n + 1) % d = 4) (h₂ : n % 2 = 1) : 
  d = 6 :=
sorry

end divisor_of_first_division_l4_4474


namespace percentage_of_mixture_X_is_13_333_l4_4014

variable (X Y : ℝ) (P : ℝ)

-- Conditions
def mixture_X_contains_40_percent_ryegrass : Prop := X = 0.40
def mixture_Y_contains_25_percent_ryegrass : Prop := Y = 0.25
def final_mixture_contains_27_percent_ryegrass : Prop := 0.4 * P + 0.25 * (100 - P) = 27

-- The goal
theorem percentage_of_mixture_X_is_13_333
    (h1 : mixture_X_contains_40_percent_ryegrass X)
    (h2 : mixture_Y_contains_25_percent_ryegrass Y)
    (h3 : final_mixture_contains_27_percent_ryegrass P) :
  P = 200 / 15 := by
  sorry

end percentage_of_mixture_X_is_13_333_l4_4014


namespace total_value_is_correct_l4_4753

-- We will define functions that convert base 7 numbers to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let digits := (n.digits 7)
  digits.enum.foldr (λ ⟨i, d⟩ acc => acc + d * 7^i) 0

-- Define the specific numbers in base 7
def silver_value_base7 : Nat := 5326
def gemstone_value_base7 : Nat := 3461
def spice_value_base7 : Nat := 656

-- Define the combined total in base 10
def total_value_base10 : Nat := base7_to_base10 silver_value_base7 + base7_to_base10 gemstone_value_base7 + base7_to_base10 spice_value_base7

theorem total_value_is_correct :
  total_value_base10 = 3485 :=
by
  sorry

end total_value_is_correct_l4_4753


namespace square_difference_l4_4393

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x - 2) * (x + 2) = 9797 :=
by 
  have diff_squares : (x - 2) * (x + 2) = x^2 - 4 := by ring
  rw [diff_squares, h]
  norm_num

end square_difference_l4_4393


namespace minimum_shift_value_l4_4993

theorem minimum_shift_value
    (m : ℝ) 
    (h1 : m > 0) :
    (∃ (k : ℤ), m = k * π - π / 3 ∧ k > 0) → (m = (2 * π) / 3) :=
sorry

end minimum_shift_value_l4_4993


namespace probability_of_selecting_boy_given_girl_A_selected_l4_4880

-- Define the total number of girls and boys
def total_girls : ℕ := 5
def total_boys : ℕ := 2

-- Define the group size to be selected
def group_size : ℕ := 3

-- Define the probability of selecting at least one boy given girl A is selected
def probability_at_least_one_boy_given_girl_A : ℚ := 3 / 5

-- Math problem reformulated as a Lean theorem
theorem probability_of_selecting_boy_given_girl_A_selected : 
  (total_girls = 5) → (total_boys = 2) → (group_size = 3) → 
  (probability_at_least_one_boy_given_girl_A = 3 / 5) :=
by sorry

end probability_of_selecting_boy_given_girl_A_selected_l4_4880


namespace total_income_percentage_l4_4634

-- Define the base income of Juan
def juan_base_income (J : ℝ) := J

-- Define Tim's base income
def tim_base_income (J : ℝ) := 0.70 * J

-- Define Mary's total income
def mary_total_income (J : ℝ) := 1.232 * J

-- Define Lisa's total income
def lisa_total_income (J : ℝ) := 0.6489 * J

-- Define Nina's total income
def nina_total_income (J : ℝ) := 1.3375 * J

-- Define the sum of the total incomes of Mary, Lisa, and Nina
def sum_income (J : ℝ) := mary_total_income J + lisa_total_income J + nina_total_income J

-- Define the statement we need to prove: the percentage of Juan's total income
theorem total_income_percentage (J : ℝ) (hJ : J ≠ 0) :
  ((sum_income J / juan_base_income J) * 100) = 321.84 :=
by
  unfold juan_base_income sum_income mary_total_income lisa_total_income nina_total_income
  sorry

end total_income_percentage_l4_4634


namespace youseff_lives_6_blocks_from_office_l4_4037

-- Definitions
def blocks_youseff_lives_from_office (x : ℕ) : Prop :=
  ∃ t_walk t_bike : ℕ,
    t_walk = x ∧
    t_bike = (20 * x) / 60 ∧
    t_walk = t_bike + 4

-- Main theorem
theorem youseff_lives_6_blocks_from_office (x : ℕ) (h : blocks_youseff_lives_from_office x) : x = 6 :=
  sorry

end youseff_lives_6_blocks_from_office_l4_4037


namespace inequality_proof_l4_4654

theorem inequality_proof 
  (a b c d : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (sum_eq : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a * b * c * d)^2 := 
by 
  sorry

end inequality_proof_l4_4654


namespace tyrone_gives_non_integer_marbles_to_eric_l4_4225

theorem tyrone_gives_non_integer_marbles_to_eric
  (T_init : ℕ) (E_init : ℕ) (x : ℚ)
  (hT : T_init = 120) (hE : E_init = 18)
  (h_eq : T_init - x = 3 * (E_init + x)) :
  ¬ (∃ n : ℕ, x = n) :=
by
  sorry

end tyrone_gives_non_integer_marbles_to_eric_l4_4225


namespace max_b_in_box_l4_4981

theorem max_b_in_box (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 := 
by
  sorry

end max_b_in_box_l4_4981


namespace salary_May_l4_4245

theorem salary_May
  (J F M A M' : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + M') / 4 = 8400)
  (h3 : J = 4900) :
  M' = 6500 :=
  by
  sorry

end salary_May_l4_4245


namespace sequence_max_length_l4_4181

theorem sequence_max_length (x : ℕ) :
  (2000 - 2 * x > 0) ∧ (3 * x - 2000 > 0) ∧ (4000 - 5 * x > 0) ∧ 
  (8 * x - 6000 > 0) ∧ (10000 - 13 * x > 0) ∧ (21 * x - 16000 > 0) → x = 762 :=
by
  sorry

end sequence_max_length_l4_4181


namespace translate_right_one_unit_l4_4705

theorem translate_right_one_unit (x y : ℤ) (hx : x = 4) (hy : y = -3) : (x + 1, y) = (5, -3) :=
by
  -- The proof would go here
  sorry

end translate_right_one_unit_l4_4705


namespace solve_system_l4_4386

theorem solve_system :
  ∀ (x y z : ℝ),
  (x^2 - 23 * y - 25 * z = -681) →
  (y^2 - 21 * x - 21 * z = -419) →
  (z^2 - 19 * x - 21 * y = -313) →
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by
  intros x y z h1 h2 h3
  sorry

end solve_system_l4_4386


namespace cos_double_angle_of_tangent_is_2_l4_4094

theorem cos_double_angle_of_tangent_is_2
  (θ : ℝ)
  (h_tan : Real.tan θ = 2) :
  Real.cos (2 * θ) = -3 / 5 := 
by
  sorry

end cos_double_angle_of_tangent_is_2_l4_4094


namespace karen_cases_picked_up_l4_4791

theorem karen_cases_picked_up (total_boxes : ℤ) (boxes_per_case : ℤ) (h1 : total_boxes = 36) (h2 : boxes_per_case = 12) : (total_boxes / boxes_per_case) = 3 := by
  sorry

end karen_cases_picked_up_l4_4791


namespace largest_perimeter_l4_4890

noncomputable def triangle_largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : ℕ :=
7 + 8 + y

theorem largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : triangle_largest_perimeter y h1 h2 = 29 :=
sorry

end largest_perimeter_l4_4890


namespace hyperbola_parameters_sum_l4_4815

theorem hyperbola_parameters_sum :
  ∃ (h k a b : ℝ), 
    (h = 2 ∧ k = 0 ∧ a = 3 ∧ b = 3 * Real.sqrt 3) ∧
    h + k + a + b = 3 * Real.sqrt 3 + 5 := by
  sorry

end hyperbola_parameters_sum_l4_4815


namespace xiao_ming_second_half_time_l4_4990

theorem xiao_ming_second_half_time :
  ∀ (total_distance : ℕ) (speed1 : ℕ) (speed2 : ℕ), 
    total_distance = 360 →
    speed1 = 5 →
    speed2 = 4 →
    let t_total := total_distance / (speed1 + speed2) * 2
    let half_distance := total_distance / 2
    let t2 := half_distance / speed2
    half_distance / speed2 + (half_distance / speed1) = 44 :=
sorry

end xiao_ming_second_half_time_l4_4990


namespace total_steps_eliana_walked_l4_4565

-- Define the conditions of the problem.
def first_day_exercise_steps : Nat := 200
def first_day_additional_steps : Nat := 300
def second_day_multiplier : Nat := 2
def third_day_additional_steps : Nat := 100

-- Define the steps calculation for each day.
def first_day_total_steps : Nat := first_day_exercise_steps + first_day_additional_steps
def second_day_total_steps : Nat := second_day_multiplier * first_day_total_steps
def third_day_total_steps : Nat := second_day_total_steps + third_day_additional_steps

-- Prove that the total number of steps Eliana walked during these three days is 1600.
theorem total_steps_eliana_walked :
  first_day_total_steps + second_day_total_steps + third_day_additional_steps = 1600 :=
by
  -- Conditional values are constants. We can use Lean's deterministic evaluator here.
  -- Hence, there's no need to write out full proof for now. Using sorry to bypass actual proof.
  sorry

end total_steps_eliana_walked_l4_4565


namespace range_of_a_l4_4290

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f x a ≥ a) ↔ -2 ≤ a ∧ a ≤ (-1 + Real.sqrt 5) / 2 := 
sorry

end range_of_a_l4_4290


namespace HephaestusCharges_l4_4651

variable (x : ℕ)

theorem HephaestusCharges :
  3 * x + 6 * (12 - x) = 54 -> x = 6 :=
by
  intros h
  sorry

end HephaestusCharges_l4_4651


namespace proof_problem_l4_4913

-- Define the problem conditions
variables (x y : ℝ)

-- State the theorem
theorem proof_problem (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 :=
by sorry

end proof_problem_l4_4913


namespace space_diagonal_of_prism_l4_4964

theorem space_diagonal_of_prism (l w h : ℝ) (hl : l = 2) (hw : w = 3) (hh : h = 4) :
  (l ^ 2 + w ^ 2 + h ^ 2).sqrt = Real.sqrt 29 :=
by
  rw [hl, hw, hh]
  sorry

end space_diagonal_of_prism_l4_4964


namespace vertices_after_removal_l4_4873

theorem vertices_after_removal (a b : ℕ) (h₁ : a = 5) (h₂ : b = 2) : 
  let initial_vertices := 8
  let removed_vertices := initial_vertices
  let new_vertices := 8 * 9
  let final_vertices := new_vertices - removed_vertices
  final_vertices = 64 :=
by
  sorry

end vertices_after_removal_l4_4873


namespace major_premise_is_wrong_l4_4752

-- Definitions of the conditions
def line_parallel_to_plane (l : Type) (p : Type) : Prop := sorry
def line_contained_in_plane (l : Type) (p : Type) : Prop := sorry

-- Stating the main problem: the major premise is wrong
theorem major_premise_is_wrong :
  ∀ (a b : Type) (α : Type), line_contained_in_plane a α → line_parallel_to_plane b α → ¬ (line_parallel_to_plane b a) := 
by 
  intros a b α h1 h2
  sorry

end major_premise_is_wrong_l4_4752


namespace express_308_million_in_scientific_notation_l4_4549

theorem express_308_million_in_scientific_notation :
    (308000000 : ℝ) = 3.08 * (10 ^ 8) :=
by
  sorry

end express_308_million_in_scientific_notation_l4_4549


namespace ratio_a_to_c_l4_4049

theorem ratio_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 7) : 
  a / c = 105 / 16 :=
by sorry

end ratio_a_to_c_l4_4049


namespace even_and_odd_functions_satisfying_equation_l4_4256

theorem even_and_odd_functions_satisfying_equation :
  ∀ (f g : ℝ → ℝ),
    (∀ x : ℝ, f (-x) = f x) →                      -- condition 1: f is even
    (∀ x : ℝ, g (-x) = -g x) →                    -- condition 2: g is odd
    (∀ x : ℝ, f x - g x = x^3 + x^2 + 1) →        -- condition 3: f(x) - g(x) = x^3 + x^2 + 1
    f 1 + g 1 = 1 :=                              -- question: proof of f(1) + g(1) = 1
by
  intros f g h_even h_odd h_eqn
  sorry

end even_and_odd_functions_satisfying_equation_l4_4256


namespace length_of_train_l4_4020

variable (d_train d_bridge v t : ℝ)

theorem length_of_train
  (h1 : v = 12.5) 
  (h2 : t = 30) 
  (h3 : d_bridge = 255) 
  (h4 : v * t = d_train + d_bridge) : 
  d_train = 120 := 
by {
  -- We should infer from here that d_train = 120
  sorry
}

end length_of_train_l4_4020


namespace smallest_int_ends_in_3_div_by_11_l4_4922

theorem smallest_int_ends_in_3_div_by_11 :
  ∃ k : ℕ, k > 0 ∧ k % 10 = 3 ∧ k % 11 = 0 ∧ k = 33 :=
by {
  sorry
}

end smallest_int_ends_in_3_div_by_11_l4_4922


namespace totalBooksOnShelves_l4_4322

-- Define the conditions
def numShelves : Nat := 150
def booksPerShelf : Nat := 15

-- Define the statement to be proved
theorem totalBooksOnShelves : numShelves * booksPerShelf = 2250 :=
by
  -- Skipping the proof
  sorry

end totalBooksOnShelves_l4_4322


namespace inequality_proof_l4_4740

theorem inequality_proof {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  (1 / Real.sqrt (1 + x^2)) + (1 / Real.sqrt (1 + y^2)) ≤ (2 / Real.sqrt (1 + x * y)) :=
by
  sorry

end inequality_proof_l4_4740


namespace find_formula_and_range_l4_4633

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := 4^x + a * 2^x + b

theorem find_formula_and_range
  (a b : ℝ)
  (h₀ : f 0 a b = 1)
  (h₁ : f (-1) a b = -5 / 4) :
  f x (-3) 3 = 4^x - 3 * 2^x + 3 ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ 2 → 1 ≤ f x (-3) 3 ∧ f x (-3) 3 ≤ 25) :=
by
  sorry

end find_formula_and_range_l4_4633


namespace problem_solution_l4_4170

-- Define the given circle equation C
def circle_C_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 3 = 0

-- Define the line of symmetry
def line_symmetry_eq (x y : ℝ) : Prop := y = -x - 4

-- Define the symmetric circle equation
def sym_circle_eq (x y : ℝ) : Prop := (x + 4)^2 + (y + 6)^2 = 1

theorem problem_solution (x y : ℝ)
  (H1 : circle_C_eq x y)
  (H2 : line_symmetry_eq x y) :
  sym_circle_eq x y :=
sorry

end problem_solution_l4_4170


namespace gcd_of_128_144_480_450_l4_4404

theorem gcd_of_128_144_480_450 : Nat.gcd (Nat.gcd 128 144) (Nat.gcd 480 450) = 6 := 
by
  sorry

end gcd_of_128_144_480_450_l4_4404


namespace plywood_problem_exists_squares_l4_4128

theorem plywood_problem_exists_squares :
  ∃ (a b : ℕ), a^2 + b^2 = 625 ∧ a ≠ 20 ∧ b ≠ 20 ∧ a ≠ 15 ∧ b ≠ 15 := by
  sorry

end plywood_problem_exists_squares_l4_4128


namespace find_valid_N_l4_4848

def is_divisible_by_10_consec (N : ℕ) : Prop :=
  ∀ m : ℕ, (N % (List.prod (List.range' m 10)) = 0)

def is_not_divisible_by_11_consec (N : ℕ) : Prop :=
  ∀ m : ℕ, ¬ (N % (List.prod (List.range' m 11)) = 0)

theorem find_valid_N (N : ℕ) :
  (is_divisible_by_10_consec N ∧ is_not_divisible_by_11_consec N) ↔
  (∃ k : ℕ, (k > 0) ∧ ¬ (k % 11 = 0) ∧ N = k * Nat.factorial 10) :=
sorry

end find_valid_N_l4_4848


namespace condition_sufficient_not_necessary_l4_4562

theorem condition_sufficient_not_necessary
  (A B C D : Prop)
  (h1 : A → B)
  (h2 : B ↔ C)
  (h3 : C → D) :
  (A → D) ∧ ¬(D → A) :=
by
  sorry

end condition_sufficient_not_necessary_l4_4562


namespace reach_any_composite_from_4_l4_4140

/-- 
Prove that starting from the number \( 4 \), it is possible to reach any given composite number 
through repeatedly adding one of its divisors, different from itself and one. 
-/
theorem reach_any_composite_from_4:
  ∀ n : ℕ, Prime (n) → n ≥ 4 → (∃ k d : ℕ, d ∣ k ∧ k = k + d ∧ k = n) := 
by 
  sorry


end reach_any_composite_from_4_l4_4140


namespace difference_max_min_is_7_l4_4456

-- Define the number of times Kale mowed his lawn during each season
def timesSpring : ℕ := 8
def timesSummer : ℕ := 5
def timesFall : ℕ := 12

-- Statement to prove
theorem difference_max_min_is_7 : 
  (max timesSpring (max timesSummer timesFall)) - (min timesSpring (min timesSummer timesFall)) = 7 :=
by
  -- Proof would go here
  sorry

end difference_max_min_is_7_l4_4456


namespace pairs_of_integers_solution_l4_4083

-- Define the main theorem
theorem pairs_of_integers_solution :
  ∃ (x y : ℤ), 9 * x * y - x^2 - 8 * y^2 = 2005 ∧ 
               ((x = 63 ∧ y = 58) ∨
               (x = -63 ∧ y = -58) ∨
               (x = 459 ∧ y = 58) ∨
               (x = -459 ∧ y = -58)) :=
by
  sorry

end pairs_of_integers_solution_l4_4083


namespace volume_ratio_l4_4106

-- Define the edge lengths
def edge_length_cube1 : ℝ := 4 -- in inches
def edge_length_cube2 : ℝ := 2 * 12 -- 2 feet converted to inches

-- Define the volumes
def volume_cube (a : ℝ) : ℝ := a ^ 3

-- Statement asserting the ratio of the volumes is 1/216
theorem volume_ratio : volume_cube edge_length_cube1 / volume_cube edge_length_cube2 = 1 / 216 :=
by
  -- This is the placeholder to skip the proof
  sorry

end volume_ratio_l4_4106


namespace points_do_not_exist_l4_4824

/-- 
  If \( A, B, C, D \) are four points in space and 
  \( AB = 8 \) cm, 
  \( CD = 8 \) cm, 
  \( AC = 10 \) cm, 
  \( BD = 10 \) cm, 
  \( AD = 13 \) cm, 
  \( BC = 13 \) cm, 
  then such points \( A, B, C, D \) cannot exist.
-/
theorem points_do_not_exist 
  (A B C D : Type)
  (AB CD AC BD AD BC : ℝ) 
  (h1 : AB = 8) 
  (h2 : CD = 8) 
  (h3 : AC = 10)
  (h4 : BD = 10)
  (h5 : AD = 13)
  (h6 : BC = 13) : 
  false :=
sorry

end points_do_not_exist_l4_4824


namespace correct_inequality_relation_l4_4495

theorem correct_inequality_relation :
  ¬(∀ (a b c : ℝ), a > b ↔ a * (c^2) > b * (c^2)) ∧
  ¬(∀ (a b : ℝ), a > b → (1/a) < (1/b)) ∧
  ¬(∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d → a/d > b/c) ∧
  (∀ (a b c : ℝ), a > b ∧ b > 1 ∧ c < 0 → a^c < b^c) := sorry

end correct_inequality_relation_l4_4495


namespace football_sampling_l4_4516

theorem football_sampling :
  ∀ (total_members football_members basketball_members volleyball_members total_sample : ℕ),
  total_members = 120 →
  football_members = 40 →
  basketball_members = 60 →
  volleyball_members = 20 →
  total_sample = 24 →
  (total_sample * football_members / (football_members + basketball_members + volleyball_members) = 8) :=
by 
  intros total_members football_members basketball_members volleyball_members total_sample h_total_members h_football_members h_basketball_members h_volleyball_members h_total_sample
  sorry

end football_sampling_l4_4516


namespace second_range_is_18_l4_4421

variable (range1 range2 range3 : ℕ)

theorem second_range_is_18
  (h1 : range1 = 30)
  (h2 : range2 = 18)
  (h3 : range3 = 32) :
  range2 = 18 := by
  sorry

end second_range_is_18_l4_4421


namespace simplify_expression_l4_4188

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l4_4188


namespace find_a_of_cool_frog_meeting_l4_4134

-- Question and conditions
def frogs : ℕ := 16
def friend_probability : ℚ := 1 / 2
def cool_condition (f: ℕ → ℕ) : Prop := ∀ i, f i % 4 = 0

-- Example theorem where we need to find 'a'
theorem find_a_of_cool_frog_meeting :
  let a := 1167
  let b := 2 ^ 41
  ∀ (f: ℕ → ℕ), ∀ (p: ℚ) (h: p = friend_probability),
    (cool_condition f) →
    (∃ a b, a / b = p ∧ a % gcd a b = 0 ∧ gcd a b = 1) ∧ a = 1167 :=
by
  sorry

end find_a_of_cool_frog_meeting_l4_4134


namespace range_of_m_l4_4348

theorem range_of_m (m : ℝ) (h1 : 0 < m) (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → |m * x^3 - Real.log x| ≥ 1) : m ≥ (1 / 3) * Real.exp 2 :=
sorry

end range_of_m_l4_4348


namespace div_eq_four_l4_4054

theorem div_eq_four (x : ℝ) (h : 64 / x = 4) : x = 16 :=
sorry

end div_eq_four_l4_4054


namespace range_of_a_plus_3b_l4_4113

theorem range_of_a_plus_3b :
  ∀ (a b : ℝ),
    -1 ≤ a + b ∧ a + b ≤ 1 ∧ 1 ≤ a - 2 * b ∧ a - 2 * b ≤ 3 →
    -11 / 3 ≤ a + 3 * b ∧ a + 3 * b ≤ 7 / 3 :=
by
  sorry

end range_of_a_plus_3b_l4_4113


namespace elvins_fixed_monthly_charge_l4_4321

-- Definition of the conditions
def january_bill (F C_J : ℝ) : Prop := F + C_J = 48
def february_bill (F C_J : ℝ) : Prop := F + 2 * C_J = 90

theorem elvins_fixed_monthly_charge (F C_J : ℝ) (h_jan : january_bill F C_J) (h_feb : february_bill F C_J) : F = 6 :=
by
  sorry

end elvins_fixed_monthly_charge_l4_4321


namespace peter_savings_l4_4412

noncomputable def calc_discounted_price (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    original_price * (1 - discount_percentage / 100)

noncomputable def calc_savings (original_price : ℝ) (external_price : ℝ) : ℝ :=
    original_price - external_price

noncomputable def total_savings : ℝ :=
    let math_original := 45.0
    let math_discount := 20.0
    let science_original := 60.0
    let science_discount := 25.0
    let literature_original := 35.0
    let literature_discount := 15.0
    let math_external := calc_discounted_price math_original math_discount
    let science_external := calc_discounted_price science_original science_discount
    let literature_external := calc_discounted_price literature_original literature_discount
    let math_savings := calc_savings math_original math_external
    let science_savings := calc_savings science_original science_external
    let literature_savings := calc_savings literature_original literature_external
    math_savings + science_savings + literature_savings

theorem peter_savings :
  total_savings = 29.25 :=
by
    sorry

end peter_savings_l4_4412


namespace Marilyn_end_caps_l4_4682

def starting_caps := 51
def shared_caps := 36
def ending_caps := starting_caps - shared_caps

theorem Marilyn_end_caps : ending_caps = 15 := by
  -- proof omitted
  sorry

end Marilyn_end_caps_l4_4682


namespace part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l4_4382

-- Part (1)
theorem part1_inequality (m : ℝ) : (∀ x : ℝ, (m^2 + 1)*x^2 - (2*m - 1)*x + 1 > 0) ↔ m > -3/4 := sorry

-- Part (2)
theorem part2_inequality_case1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ x < 1 ∨ x > 1/a) := sorry

theorem part2_inequality_case2 : 
  (∀ x : ℝ, (x - 1)*(0*x - 1) > 0 ↔ x < 1) := sorry

theorem part2_inequality_case3 (a : ℝ) (h : a < 0) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ 1/a < x ∧ x < 1) := sorry

end part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l4_4382


namespace parallel_vectors_xy_l4_4108

theorem parallel_vectors_xy {x y : ℝ} (h : ∃ k : ℝ, (1, y, -3) = (k * x, k * (-2), k * 5)) : x * y = -2 :=
by sorry

end parallel_vectors_xy_l4_4108


namespace polar_to_rectangular_l4_4260

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 4) :
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x = -5 * Real.sqrt 2 / 2 ∧ y = -5 * Real.sqrt 2 / 2 :=
by
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l4_4260


namespace train_pass_time_l4_4293

def train_length : ℕ := 250
def train_speed_kmph : ℕ := 36
def station_length : ℕ := 200

def total_distance : ℕ := train_length + station_length

noncomputable def train_speed_mps : ℚ := (train_speed_kmph : ℚ) * 1000 / 3600

noncomputable def time_to_pass_station : ℚ := total_distance / train_speed_mps

theorem train_pass_time : time_to_pass_station = 45 := by
  sorry

end train_pass_time_l4_4293


namespace no_a_for_empty_intersection_a_in_range_for_subset_union_l4_4053

open Set

def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}
def B : Set ℝ := {x | x^2 - 4 * x - 5 > 0}

-- Problem 1: There is no a such that A ∩ B = ∅
theorem no_a_for_empty_intersection : ∀ a : ℝ, A a ∩ B = ∅ → False := by
  sorry

-- Problem 2: If A ∪ B = B, then a ∈ (-∞, -4) ∪ (5, ∞)
theorem a_in_range_for_subset_union (a : ℝ) : A a ∪ B = B → a ∈ Iio (-4) ∪ Ioi 5 := by
  sorry

end no_a_for_empty_intersection_a_in_range_for_subset_union_l4_4053


namespace find_n_l4_4131

noncomputable def e : ℝ := Real.exp 1

-- lean cannot compute non-trivial transcendental solutions, this would need numerical methods
theorem find_n (n : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3) (h2 : y = 27) :
  Real.log n ^ (n / (2 * Real.sqrt (Real.pi + x))) = y :=
by
  rw [h1, h2]
  sorry

end find_n_l4_4131


namespace angle_A_size_max_area_triangle_l4_4985

open Real

variable {A B C a b c : ℝ}

-- Part 1: Prove the size of angle A given the conditions
theorem angle_A_size (h1 : (2 * c - b) / a = cos B / cos A) :
  A = π / 3 :=
sorry

-- Part 2: Prove the maximum area of triangle ABC
theorem max_area_triangle (h2 : a = 2 * sqrt 5) :
  ∃ (S : ℝ), S = 5 * sqrt 3 ∧ ∀ (b c : ℝ), S ≤ 1/2 * b * c * sin (π / 3) :=
sorry

end angle_A_size_max_area_triangle_l4_4985


namespace paul_initial_crayons_l4_4912

-- Define the variables for the crayons given away, lost, and left
def crayons_given_away : ℕ := 563
def crayons_lost : ℕ := 558
def crayons_left : ℕ := 332

-- Define the total number of crayons Paul got for his birthday
def initial_crayons : ℕ := 1453

-- The proof statement
theorem paul_initial_crayons :
  initial_crayons = crayons_given_away + crayons_lost + crayons_left :=
sorry

end paul_initial_crayons_l4_4912


namespace find_a_l4_4160

variables {a b c : ℤ}

theorem find_a (h1 : a + b = c) (h2 : b + c = 7) (h3 : c = 5) : a = 3 :=
by
  sorry

end find_a_l4_4160


namespace sum_lent_is_1500_l4_4483

/--
A person lent a certain sum of money at 4% per annum at simple interest.
In 4 years, the interest amounted to Rs. 1260 less than the sum lent.
Prove that the sum lent was Rs. 1500.
-/
theorem sum_lent_is_1500
  (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)
  (h1 : r = 4) (h2 : t = 4)
  (h3 : I = P - 1260)
  (h4 : I = P * r * t / 100):
  P = 1500 :=
by
  sorry

end sum_lent_is_1500_l4_4483


namespace Joshua_share_correct_l4_4223

noncomputable def Joshua_share (J : ℝ) : ℝ :=
  3 * J

noncomputable def Jasmine_share (J : ℝ) : ℝ :=
  J / 2

theorem Joshua_share_correct (J : ℝ) (h : J + 3 * J + J / 2 = 120) :
  Joshua_share J = 80.01 := by
  sorry

end Joshua_share_correct_l4_4223


namespace area_of_nonagon_on_other_cathetus_l4_4034

theorem area_of_nonagon_on_other_cathetus 
    (A₁ A₂ A₃ : ℝ) 
    (h1 : A₁ = 2019) 
    (h2 : A₂ = 1602) 
    (h3 : A₁ = A₂ + A₃) : 
    A₃ = 417 :=
by
  rw [h1, h2] at h3
  linarith

end area_of_nonagon_on_other_cathetus_l4_4034


namespace projection_of_orthogonal_vectors_l4_4330

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (scale * v.1, scale * v.2)

theorem projection_of_orthogonal_vectors
  (a b : ℝ × ℝ)
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj : proj (4, -2) a = (4 / 5, 8 / 5)) :
  proj (4, -2) b = (16 / 5, -18 / 5) :=
sorry

end projection_of_orthogonal_vectors_l4_4330


namespace part1_part2_l4_4764

theorem part1 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h_imag : z.re = 0) : a = 1 :=
sorry

theorem part2 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h4thQuad : z.re > 0 ∧ z.im < 0) : -1 < a ∧ a < 1 :=
sorry

end part1_part2_l4_4764


namespace solve_system_of_equations_l4_4828

theorem solve_system_of_equations (a b c x y z : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  (a * y + b * x = c) ∧ (c * x + a * z = b) ∧ (b * z + c * y = a) →
  (x = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
  (y = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
  (z = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by
  sorry

end solve_system_of_equations_l4_4828


namespace no_integer_solutions_l4_4574

theorem no_integer_solutions (x y : ℤ) : 19 * x^3 - 84 * y^2 ≠ 1984 :=
by
  sorry

end no_integer_solutions_l4_4574


namespace solve_x_l4_4307

theorem solve_x (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) := 
by 
  sorry

end solve_x_l4_4307


namespace no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l4_4261

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l4_4261


namespace plan1_more_cost_effective_than_plan2_l4_4394

variable (x : ℝ)

def plan1_cost (x : ℝ) : ℝ :=
  36 + 0.1 * x

def plan2_cost (x : ℝ) : ℝ :=
  0.6 * x

theorem plan1_more_cost_effective_than_plan2 (h : x > 72) : 
  plan1_cost x < plan2_cost x :=
by
  sorry

end plan1_more_cost_effective_than_plan2_l4_4394


namespace minute_hand_position_l4_4787

theorem minute_hand_position (t : ℕ) (h_start : t = 2022) :
  let cycle_minutes := 8
  let net_movement_per_cycle := 2
  let full_cycles := t / cycle_minutes
  let remaining_minutes := t % cycle_minutes
  let full_cycles_movement := full_cycles * net_movement_per_cycle
  let extra_movement := if remaining_minutes <= 5 then remaining_minutes else 5 - (remaining_minutes - 5)
  let total_movement := full_cycles_movement + extra_movement
  (total_movement % 60) = 28 :=
by {
  sorry
}

end minute_hand_position_l4_4787


namespace initial_men_count_l4_4205

theorem initial_men_count (M : ℕ) (P : ℕ) :
  P = M * 20 →
  P = (M + 650) * 109 / 9 →
  M = 1000 :=
by
  sorry

end initial_men_count_l4_4205


namespace kenny_pieces_used_l4_4104

-- Definitions based on conditions
def mushrooms_cut := 22
def pieces_per_mushroom := 4
def karla_pieces := 42
def remaining_pieces := 8
def total_pieces := mushrooms_cut * pieces_per_mushroom

-- Theorem to be proved
theorem kenny_pieces_used :
  total_pieces - (karla_pieces + remaining_pieces) = 38 := 
by 
  sorry

end kenny_pieces_used_l4_4104


namespace geometric_sequence_common_ratio_l4_4611

theorem geometric_sequence_common_ratio (a₁ : ℚ) (q : ℚ) 
  (S : ℕ → ℚ) (hS : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h : 8 * S 6 = 7 * S 3) : 
  q = -1/2 :=
sorry

end geometric_sequence_common_ratio_l4_4611


namespace binary_divisible_by_136_l4_4297

theorem binary_divisible_by_136 :
  let N := 2^139 + 2^105 + 2^15 + 2^13
  N % 136 = 0 :=
by {
  let N := 2^139 + 2^105 + 2^15 + 2^13;
  sorry
}

end binary_divisible_by_136_l4_4297


namespace g_of_3_over_8_l4_4043

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end g_of_3_over_8_l4_4043


namespace staircase_problem_l4_4436

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem staircase_problem (total_steps required_steps : ℕ) (num_two_steps : ℕ) :
  total_steps = 11 ∧ required_steps = 7 ∧ num_two_steps = 4 →
  C 7 4 = 35 :=
by
  intro h
  sorry

end staircase_problem_l4_4436


namespace race_problem_l4_4784

theorem race_problem
  (total_distance : ℕ)
  (A_time : ℕ)
  (B_extra_time : ℕ)
  (A_speed B_speed : ℕ)
  (A_distance B_distance : ℕ)
  (H1 : total_distance = 120)
  (H2 : A_time = 8)
  (H3 : B_extra_time = 7)
  (H4 : A_speed = total_distance / A_time)
  (H5 : B_speed = total_distance / (A_time + B_extra_time))
  (H6 : A_distance = total_distance)
  (H7 : B_distance = B_speed * A_time) :
  A_distance - B_distance = 56 := 
sorry

end race_problem_l4_4784


namespace micah_total_envelopes_l4_4286

-- Define the conditions as hypotheses
def weight_threshold := 5
def stamps_for_heavy := 5
def stamps_for_light := 2
def total_stamps := 52
def light_envelopes := 6

-- Noncomputable because we are using abstract reasoning rather than computational functions
noncomputable def total_envelopes : ℕ :=
  light_envelopes + (total_stamps - light_envelopes * stamps_for_light) / stamps_for_heavy

-- The theorem to prove
theorem micah_total_envelopes : total_envelopes = 14 := by
  sorry

end micah_total_envelopes_l4_4286


namespace double_sum_evaluation_l4_4240

theorem double_sum_evaluation :
  ∑' m:ℕ, ∑' n:ℕ, (if m > 0 ∧ n > 0 then 1 / (m * n * (m + n + 2)) else 0) = -Real.pi^2 / 6 :=
sorry

end double_sum_evaluation_l4_4240


namespace find_constant_term_of_polynomial_with_negative_integer_roots_l4_4582

theorem find_constant_term_of_polynomial_with_negative_integer_roots
  (p q r s : ℝ) (t1 t2 t3 t4 : ℝ)
  (h_roots : ∀ {x : ℝ}, x^4 + p*x^3 + q*x^2 + r*x + s = (x + t1)*(x + t2)*(x + t3)*(x + t4))
  (h_neg_int_roots : ∀ {i : ℕ}, i < 4 → t1 = i ∨ t2 = i ∨ t3 = i ∨ t4 = i)
  (h_sum_coeffs : p + q + r + s = 168) :
  s = 144 :=
by
  sorry

end find_constant_term_of_polynomial_with_negative_integer_roots_l4_4582


namespace sufficient_but_not_necessary_condition_l4_4846

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1) → (|x + 1| + |x - 1| = 2 * |x|) ∧ ¬((x ≥ 1) ↔ (|x + 1| + |x - 1| = 2 * |x|)) := by
  sorry

end sufficient_but_not_necessary_condition_l4_4846


namespace solve_equation_l4_4903

theorem solve_equation :
  ∀ x : ℝ, (1 + 2 * x ^ (1/2) - x ^ (1/3) - 2 * x ^ (1/6) = 0) ↔ (x = 1 ∨ x = 1 / 64) :=
by
  sorry

end solve_equation_l4_4903


namespace intersection_M_N_l4_4493

open Set

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | (x + 2) * (x - 1) < 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end intersection_M_N_l4_4493


namespace solve_for_y_l4_4530

theorem solve_for_y (x y : ℝ) (h1 : x * y = 25) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 5 / 6 := 
by
  sorry

end solve_for_y_l4_4530


namespace geometric_series_first_term_l4_4311

theorem geometric_series_first_term
  (a r : ℚ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 150) :
  a = 60 / 7 :=
by
  sorry

end geometric_series_first_term_l4_4311


namespace sum_of_fractions_l4_4116

theorem sum_of_fractions : 
  (2 / 5 : ℚ) + (4 / 50 : ℚ) + (3 / 500 : ℚ) + (8 / 5000 : ℚ) = 4876 / 10000 :=
by
  -- The proof can be completed by converting fractions and summing them accurately.
  sorry

end sum_of_fractions_l4_4116


namespace geometric_sequence_not_sufficient_nor_necessary_l4_4780

theorem geometric_sequence_not_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) → 
  (¬ (q > 1 → ∀ n : ℕ, a n < a (n + 1))) ∧ (¬ (∀ n : ℕ, a n < a (n + 1) → q > 1)) :=
by
  sorry

end geometric_sequence_not_sufficient_nor_necessary_l4_4780


namespace quadratic_eq_solutions_l4_4992

theorem quadratic_eq_solutions : ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ x = 3 / 2 ∨ x = 1 :=
by
  sorry

end quadratic_eq_solutions_l4_4992


namespace sam_letters_on_wednesday_l4_4092

/-- Sam's average letters per day. -/
def average_letters_per_day : ℕ := 5

/-- Number of days Sam wrote letters. -/
def number_of_days : ℕ := 2

/-- Letters Sam wrote on Tuesday. -/
def letters_on_tuesday : ℕ := 7

/-- Total letters Sam wrote in two days. -/
def total_letters : ℕ := average_letters_per_day * number_of_days

/-- Letters Sam wrote on Wednesday. -/
def letters_on_wednesday : ℕ := total_letters - letters_on_tuesday

theorem sam_letters_on_wednesday : letters_on_wednesday = 3 :=
by
  -- placeholder proof
  sorry

end sam_letters_on_wednesday_l4_4092


namespace largest_n_satisfying_inequality_l4_4901

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, (1/4 + n/6 < 3/2) ∧ (∀ m : ℕ, m > n → 1/4 + m/6 ≥ 3/2) :=
by
  sorry

end largest_n_satisfying_inequality_l4_4901


namespace total_legs_of_collection_l4_4142

theorem total_legs_of_collection (spiders ants : ℕ) (legs_per_spider legs_per_ant : ℕ)
  (h_spiders : spiders = 8) (h_ants : ants = 12)
  (h_legs_per_spider : legs_per_spider = 8) (h_legs_per_ant : legs_per_ant = 6) :
  (spiders * legs_per_spider + ants * legs_per_ant) = 136 :=
by
  sorry

end total_legs_of_collection_l4_4142


namespace negation_proposition_l4_4552

theorem negation_proposition : ¬ (∀ x : ℝ, (1 < x) → x^3 > x^(1/3)) ↔ ∃ x : ℝ, (1 < x) ∧ x^3 ≤ x^(1/3) := by
  sorry

end negation_proposition_l4_4552


namespace largest_divisor_for_consecutive_seven_odds_l4_4932

theorem largest_divisor_for_consecutive_seven_odds (n : ℤ) (h_even : 2 ∣ n) (h_pos : 0 < n) : 
  105 ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) :=
sorry

end largest_divisor_for_consecutive_seven_odds_l4_4932


namespace carson_can_ride_giant_slide_exactly_twice_l4_4157

noncomputable def Carson_Carnival : Prop := 
  let total_time_available := 240
  let roller_coaster_time := 30
  let tilt_a_whirl_time := 60
  let giant_slide_time := 15
  let vortex_time := 45
  let bumper_cars_time := 25
  let roller_coaster_rides := 4
  let tilt_a_whirl_rides := 2
  let vortex_rides := 1
  let bumper_cars_rides := 3

  let total_time_spent := 
    roller_coaster_time * roller_coaster_rides +
    tilt_a_whirl_time * tilt_a_whirl_rides +
    vortex_time * vortex_rides +
    bumper_cars_time * bumper_cars_rides

  total_time_available - (total_time_spent + giant_slide_time * 2) = 0

theorem carson_can_ride_giant_slide_exactly_twice : Carson_Carnival :=
by
  unfold Carson_Carnival
  sorry -- proof will be provided here

end carson_can_ride_giant_slide_exactly_twice_l4_4157


namespace find_a_for_cubic_sum_l4_4935

theorem find_a_for_cubic_sum (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - a * x1 + a + 2 = 0 ∧ 
    x2^2 - a * x2 + a + 2 = 0 ∧
    x1 + x2 = a ∧
    x1 * x2 = a + 2 ∧
    x1^3 + x2^3 = -8) ↔ a = -2 := 
by
  sorry

end find_a_for_cubic_sum_l4_4935


namespace parabola_intersects_x_axis_l4_4977

theorem parabola_intersects_x_axis 
  (a c : ℝ) 
  (h : ∃ x : ℝ, x = 1 ∧ (a * x^2 + x + c = 0)) : 
  a + c = -1 :=
sorry

end parabola_intersects_x_axis_l4_4977


namespace gamma_bank_min_savings_l4_4743

def total_airfare_cost : ℕ := 10200 * 2 * 3
def total_hotel_cost : ℕ := 6500 * 12
def total_food_cost : ℕ := 1000 * 14 * 3
def total_excursion_cost : ℕ := 20000
def total_expenses : ℕ := total_airfare_cost + total_hotel_cost + total_food_cost + total_excursion_cost
def initial_amount_available : ℕ := 150000

def annual_rate_rebs : ℝ := 0.036
def annual_rate_gamma : ℝ := 0.045
def annual_rate_tisi : ℝ := 0.0312
def monthly_rate_btv : ℝ := 0.0025

noncomputable def compounded_amount_rebs : ℝ :=
  initial_amount_available * (1 + annual_rate_rebs / 12) ^ 6

noncomputable def compounded_amount_gamma : ℝ :=
  initial_amount_available * (1 + annual_rate_gamma / 2)

noncomputable def compounded_amount_tisi : ℝ :=
  initial_amount_available * (1 + annual_rate_tisi / 4) ^ 2

noncomputable def compounded_amount_btv : ℝ :=
  initial_amount_available * (1 + monthly_rate_btv) ^ 6

noncomputable def interest_rebs : ℝ := compounded_amount_rebs - initial_amount_available
noncomputable def interest_gamma : ℝ := compounded_amount_gamma - initial_amount_available
noncomputable def interest_tisi : ℝ := compounded_amount_tisi - initial_amount_available
noncomputable def interest_btv : ℝ := compounded_amount_btv - initial_amount_available

noncomputable def amount_needed_from_salary_rebs : ℝ :=
  total_expenses - initial_amount_available - interest_rebs

noncomputable def amount_needed_from_salary_gamma : ℝ :=
  total_expenses - initial_amount_available - interest_gamma

noncomputable def amount_needed_from_salary_tisi : ℝ :=
  total_expenses - initial_amount_available - interest_tisi

noncomputable def amount_needed_from_salary_btv : ℝ :=
  total_expenses - initial_amount_available - interest_btv

theorem gamma_bank_min_savings :
  amount_needed_from_salary_gamma = 47825.00 ∧
  amount_needed_from_salary_rebs = 48479.67 ∧
  amount_needed_from_salary_tisi = 48850.87 ∧
  amount_needed_from_salary_btv = 48935.89 :=
by sorry

end gamma_bank_min_savings_l4_4743


namespace intersection_S_T_l4_4794

def S : Set ℤ := {-4, -3, 6, 7}
def T : Set ℤ := {x | x^2 > 4 * x}

theorem intersection_S_T : S ∩ T = {-4, -3, 6, 7} :=
by
  sorry

end intersection_S_T_l4_4794


namespace solve_inequality_l4_4056

variable (x : ℝ)

theorem solve_inequality : 3 * (x + 2) - 1 ≥ 5 - 2 * (x - 2) → x ≥ 4 / 5 :=
by
  sorry

end solve_inequality_l4_4056


namespace marge_final_plant_count_l4_4356

/-- Define the initial conditions of the garden -/
def initial_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 8
def lavender_seeds : ℕ := 5
def seeds_without_growth : ℕ := 5

/-- Growth rates for each type of plant -/
def marigold_growth_rate : ℕ := 4
def sunflower_growth_rate : ℕ := 4
def lavender_growth_rate : ℕ := 3

/-- Impact of animals -/
def marigold_eaten_by_squirrels : ℕ := 2
def sunflower_eaten_by_rabbits : ℕ := 1

/-- Impact of pest control -/
def marigold_pest_control_reduction : ℕ := 0
def sunflower_pest_control_reduction : ℕ := 0
def lavender_pest_control_protected : ℕ := 2

/-- Impact of weeds -/
def weeds_strangled_plants : ℕ := 2

/-- Weeds left as plants -/
def weeds_kept_as_plants : ℕ := 1

/-- Marge's final number of plants -/
def survived_plants :=
  (marigold_growth_rate - marigold_eaten_by_squirrels - marigold_pest_control_reduction) +
  (sunflower_growth_rate - sunflower_eaten_by_rabbits - sunflower_pest_control_reduction) +
  (lavender_growth_rate - (lavender_growth_rate - lavender_pest_control_protected)) - weeds_strangled_plants

theorem marge_final_plant_count :
  survived_plants + weeds_kept_as_plants = 6 :=
by
  sorry

end marge_final_plant_count_l4_4356


namespace jeremy_age_l4_4137

theorem jeremy_age (A J C : ℕ) (h1 : A + J + C = 132) (h2 : A = J / 3) (h3 : C = 2 * A) : J = 66 :=
by
  sorry

end jeremy_age_l4_4137


namespace quadratic_distinct_roots_l4_4540

theorem quadratic_distinct_roots (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
sorry

end quadratic_distinct_roots_l4_4540


namespace max_area_circle_between_parallel_lines_l4_4238

theorem max_area_circle_between_parallel_lines : 
  ∀ (l₁ l₂ : ℝ → ℝ → Prop), 
    (∀ x y, l₁ x y ↔ 3*x - 4*y = 0) → 
    (∀ x y, l₂ x y ↔ 3*x - 4*y - 20 = 0) → 
  ∃ A, A = 4 * Real.pi :=
by 
  sorry

end max_area_circle_between_parallel_lines_l4_4238


namespace find_congruence_l4_4952

theorem find_congruence (x : ℤ) (h : 4 * x + 9 ≡ 3 [ZMOD 17]) : 3 * x + 12 ≡ 16 [ZMOD 17] :=
sorry

end find_congruence_l4_4952


namespace not_simplifiable_by_difference_of_squares_l4_4030

theorem not_simplifiable_by_difference_of_squares :
  ¬(∃ a b : ℝ, (-x + y) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (-x - y) * (-x + y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y + x) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y - x) * (x + y) = a^2 - b^2) :=
sorry

end not_simplifiable_by_difference_of_squares_l4_4030


namespace rainfall_hydroville_2012_l4_4695

-- Define the average monthly rainfall for each year
def avg_rainfall_2010 : ℝ := 37.2
def avg_rainfall_2011 : ℝ := avg_rainfall_2010 + 3.5
def avg_rainfall_2012 : ℝ := avg_rainfall_2011 - 1.2

-- Define the total rainfall for 2012
def total_rainfall_2012 : ℝ := 12 * avg_rainfall_2012

-- The theorem to be proved
theorem rainfall_hydroville_2012 : total_rainfall_2012 = 474 := by
  sorry

end rainfall_hydroville_2012_l4_4695


namespace interest_rate_of_first_investment_l4_4388

theorem interest_rate_of_first_investment (x y : ℝ) (h1 : x + y = 2000) (h2 : y = 650) (h3 : 0.10 * x - 0.08 * y = 83) : (0.10 * x) / x = 0.10 := by
  sorry

end interest_rate_of_first_investment_l4_4388


namespace rectangular_prism_faces_edges_vertices_sum_l4_4317

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l4_4317


namespace mean_score_calculation_l4_4958

noncomputable def class_mean_score (total_students students_1 mean_score_1 students_2 mean_score_2 : ℕ) : ℚ :=
  ((students_1 * mean_score_1 + students_2 * mean_score_2) : ℚ) / total_students

theorem mean_score_calculation :
  class_mean_score 60 54 76 6 82 = 76.6 := 
sorry

end mean_score_calculation_l4_4958


namespace minimum_Q_l4_4127

def is_special (m : ℕ) : Prop :=
  let d1 := m / 10 
  let d2 := m % 10
  d1 ≠ d2 ∧ d1 ≠ 0 ∧ d2 ≠ 0

def F (m : ℕ) : ℤ :=
  let d1 := m / 10
  let d2 := m % 10
  (d1 * 100 + d2 * 10 + d1) - (d2 * 100 + d1 * 10 + d2) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s) / s

variables (a b x y : ℕ)
variables (h1 : 1 ≤ b ∧ b < a ∧ a ≤ 7)
variables (h2 : 1 ≤ x ∧ x ≤ 8)
variables (h3 : 1 ≤ y ∧ y ≤ 8)
variables (hs_is_special : is_special (10 * a + b))
variables (ht_is_special : is_special (10 * x + y))
variables (s := 10 * a + b)
variables (t := 10 * x + y)
variables (h4 : (F s % 5) = 1)
variables (h5 : F t - F s + 18 * x = 36)

theorem minimum_Q : Q s t = -42 / 73 := sorry

end minimum_Q_l4_4127


namespace no_positive_alpha_exists_l4_4139

theorem no_positive_alpha_exists :
  ¬ ∃ α > 0, ∀ x : ℝ, |Real.cos x| + |Real.cos (α * x)| > Real.sin x + Real.sin (α * x) :=
by
  sorry

end no_positive_alpha_exists_l4_4139


namespace oranges_ratio_l4_4206

theorem oranges_ratio (T : ℕ) (h1 : 100 + T + 70 = 470) : T / 100 = 3 := by
  -- The solution steps are omitted.
  sorry

end oranges_ratio_l4_4206


namespace total_fish_bought_l4_4525

theorem total_fish_bought (gold_fish blue_fish : Nat) (h1 : gold_fish = 15) (h2 : blue_fish = 7) : gold_fish + blue_fish = 22 := by
  sorry

end total_fish_bought_l4_4525


namespace worker_bees_in_hive_l4_4826

variable (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ)

def finalWorkerBees (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ) : ℕ :=
  initialWorkerBees - leavingWorkerBees + returningWorkerBees

theorem worker_bees_in_hive
  (initialWorkerBees : ℕ := 400)
  (leavingWorkerBees : ℕ := 28)
  (returningWorkerBees : ℕ := 15) :
  finalWorkerBees initialWorkerBees leavingWorkerBees returningWorkerBees = 387 := by
  sorry

end worker_bees_in_hive_l4_4826


namespace polygon_of_T_has_4_sides_l4_4894

def T (b : ℝ) (x y : ℝ) : Prop :=
  b ≤ x ∧ x ≤ 4 * b ∧
  b ≤ y ∧ y ≤ 4 * b ∧
  x + y ≥ 3 * b ∧
  x + 2 * b ≥ 2 * y ∧
  2 * y ≥ x + b

noncomputable def sides_of_T (b : ℝ) : ℕ :=
  if b > 0 then 4 else 0

theorem polygon_of_T_has_4_sides (b : ℝ) (hb : b > 0) : sides_of_T b = 4 := by
  sorry

end polygon_of_T_has_4_sides_l4_4894


namespace fixed_amount_at_least_190_l4_4665

variable (F S : ℝ)

theorem fixed_amount_at_least_190
  (h1 : S = 7750)
  (h2 : F + 0.04 * S ≥ 500) :
  F ≥ 190 := by
  sorry

end fixed_amount_at_least_190_l4_4665


namespace cities_with_fewer_than_200000_residents_l4_4032

def percentage_of_cities_with_fewer_than_50000 : ℕ := 20
def percentage_of_cities_with_50000_to_199999 : ℕ := 65

theorem cities_with_fewer_than_200000_residents :
  percentage_of_cities_with_fewer_than_50000 + percentage_of_cities_with_50000_to_199999 = 85 :=
by
  sorry

end cities_with_fewer_than_200000_residents_l4_4032


namespace lateral_surface_area_of_cylinder_l4_4401

theorem lateral_surface_area_of_cylinder :
  let r := 1
  let h := 2
  2 * Real.pi * r * h = 4 * Real.pi :=
by
  sorry

end lateral_surface_area_of_cylinder_l4_4401


namespace Helen_raisins_l4_4414

/-- Given that Helen baked 19 chocolate chip cookies yesterday, baked some raisin cookies and 237 chocolate chip cookies this morning,
    and baked 25 more chocolate chip cookies than raisin cookies in total,
    prove that the number of raisin cookies (R) she baked is 231. -/
theorem Helen_raisins (R : ℕ) (h1 : 25 + R = 256) : R = 231 :=
by
  sorry

end Helen_raisins_l4_4414


namespace find_d_l4_4464

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)
  (h1 : α = c)
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : c = 36) :
  d = 42 := 
sorry

end find_d_l4_4464


namespace projection_areas_are_correct_l4_4706

noncomputable def S1 := 1/2 * 2 * 2
noncomputable def S2 := 1/2 * 2 * Real.sqrt 2
noncomputable def S3 := 1/2 * 2 * Real.sqrt 2

theorem projection_areas_are_correct :
  S3 = S2 ∧ S3 ≠ S1 :=
by
  sorry

end projection_areas_are_correct_l4_4706


namespace black_balls_count_l4_4347

theorem black_balls_count :
  ∀ (r k : ℕ), r = 10 -> (2 : ℚ) / 7 = r / (r + k : ℚ) -> k = 25 := by
  intros r k hr hprob
  sorry

end black_balls_count_l4_4347


namespace find_f_2_l4_4962

-- Condition: f(x + 1) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Statement to prove
theorem find_f_2 : f 2 = -1 := by
  sorry

end find_f_2_l4_4962


namespace trains_length_difference_eq_zero_l4_4807

theorem trains_length_difference_eq_zero
  (T1_pole_time : ℕ) (T1_platform_time : ℕ) (T2_pole_time : ℕ) (T2_platform_time : ℕ) (platform_length : ℕ)
  (h1 : T1_pole_time = 11)
  (h2 : T1_platform_time = 22)
  (h3 : T2_pole_time = 15)
  (h4 : T2_platform_time = 30)
  (h5 : platform_length = 120) :
  let L1 := T1_pole_time * platform_length / (T1_platform_time - T1_pole_time)
  let L2 := T2_pole_time * platform_length / (T2_platform_time - T2_pole_time)
  L1 = L2 :=
by
  sorry

end trains_length_difference_eq_zero_l4_4807


namespace interval_solution_l4_4811

theorem interval_solution (x : ℝ) : 
  (1 < 5 * x ∧ 5 * x < 3) ∧ (2 < 8 * x ∧ 8 * x < 4) ↔ (1/4 < x ∧ x < 1/2) := 
by
  sorry

end interval_solution_l4_4811


namespace ratio_of_average_speeds_l4_4982

-- Define the conditions as constants
def distance_ab : ℕ := 510
def distance_ac : ℕ := 300
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4

-- Define the speeds
def speed_eddy := distance_ab / time_eddy
def speed_freddy := distance_ac / time_freddy

-- The ratio calculation and verification function
def speed_ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- Define the main theorem to be proved
theorem ratio_of_average_speeds : speed_ratio speed_eddy speed_freddy = (34, 15) := by
  sorry

end ratio_of_average_speeds_l4_4982


namespace sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l4_4768

noncomputable def compare_sq_roots_sum : Prop := 
  (Real.sqrt 11 + Real.sqrt 3) < (Real.sqrt 9 + Real.sqrt 5)

theorem sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5 :
  compare_sq_roots_sum :=
sorry

end sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l4_4768


namespace find_values_l4_4001

theorem find_values (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 = 4 * a * b) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end find_values_l4_4001


namespace find_a6_l4_4439

variable {a : ℕ → ℤ} -- Assume we have a sequence of integers
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Conditions
axiom h1 : a 3 = 7
axiom h2 : a 5 = a 2 + 6

-- Define arithmetic sequence property
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n + d

-- Theorem to prove
theorem find_a6 (h1 : a 3 = 7) (h2 : a 5 = a 2 + 6) (h3 : arithmetic_seq a d) : a 6 = 13 :=
by
  sorry

end find_a6_l4_4439


namespace aunt_gave_each_20_l4_4005

theorem aunt_gave_each_20
  (jade_initial : ℕ)
  (julia_initial : ℕ)
  (total_after_aunt : ℕ)
  (equal_amount_from_aunt : ℕ)
  (h1 : jade_initial = 38)
  (h2 : julia_initial = jade_initial / 2)
  (h3 : total_after_aunt = 97)
  (h4 : jade_initial + julia_initial + 2 * equal_amount_from_aunt = total_after_aunt) :
  equal_amount_from_aunt = 20 := 
sorry

end aunt_gave_each_20_l4_4005


namespace arithmetic_sequence_150th_term_l4_4219

theorem arithmetic_sequence_150th_term :
  let a1 := 3
  let d := 5
  let n := 150
  a1 + (n - 1) * d = 748 :=
by
  sorry

end arithmetic_sequence_150th_term_l4_4219


namespace triangle_altitude_length_l4_4668

variable (AB AC BC BA1 AA1 : ℝ)
variable (eq1 : AB = 8)
variable (eq2 : AC = 10)
variable (eq3 : BC = 12)

theorem triangle_altitude_length (h : ∃ AA1, AA1 * AA1 + BA1 * BA1 = 64 ∧ 
                                AA1 * AA1 + (BC - BA1) * (BC - BA1) = 100) :
    BA1 = 4.5 := by
  sorry 

end triangle_altitude_length_l4_4668


namespace most_frequent_digit_100000_l4_4646

/- Define the digital root function -/
def digital_root (n : ℕ) : ℕ :=
  if n == 0 then 0 else if n % 9 == 0 then 9 else n % 9

/- Define the problem statement -/
theorem most_frequent_digit_100000 : 
  ∃ digit : ℕ, 
  digit = 1 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → ∃ k : ℕ, k = digital_root n ∧ k = digit) →
  digit = 1 :=
sorry

end most_frequent_digit_100000_l4_4646


namespace new_average_increased_by_40_percent_l4_4550

theorem new_average_increased_by_40_percent 
  (n : ℕ) (initial_avg : ℝ) (initial_marks : ℝ) (new_marks : ℝ) (new_avg : ℝ)
  (h1 : n = 37)
  (h2 : initial_avg = 73)
  (h3 : initial_marks = (initial_avg * n))
  (h4 : new_marks = (initial_marks * 1.40))
  (h5 : new_avg = (new_marks / n)) :
  new_avg = 102.2 :=
sorry

end new_average_increased_by_40_percent_l4_4550


namespace average_age_is_27_l4_4484

variables (a b c : ℕ)

def average_age_of_a_and_c (a c : ℕ) := (a + c) / 2

def age_of_b := 23

def average_age_of_a_b_and_c (a b c : ℕ) := (a + b + c) / 3

theorem average_age_is_27 (h1 : average_age_of_a_and_c a c = 29) (h2 : b = age_of_b) :
  average_age_of_a_b_and_c a b c = 27 := by
  sorry

end average_age_is_27_l4_4484


namespace P_necessary_but_not_sufficient_for_q_l4_4541

def M : Set ℝ := {x : ℝ | (x - 1) * (x - 2) > 0}
def N : Set ℝ := {x : ℝ | x^2 + x < 0}

theorem P_necessary_but_not_sufficient_for_q :
  (∀ x, x ∈ N → x ∈ M) ∧ (∃ x, x ∈ M ∧ x ∉ N) :=
by
  sorry

end P_necessary_but_not_sufficient_for_q_l4_4541


namespace sin_beta_equals_sqrt3_div_2_l4_4103

noncomputable def angles_acute (α β : ℝ) : Prop :=
0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2

theorem sin_beta_equals_sqrt3_div_2 
  (α β : ℝ) 
  (h_acute: angles_acute α β) 
  (h_sin_alpha: Real.sin α = (4/7) * Real.sqrt 3) 
  (h_cos_alpha_plus_beta: Real.cos (α + β) = -(11/14)) 
  : Real.sin β = (Real.sqrt 3) / 2 :=
sorry

end sin_beta_equals_sqrt3_div_2_l4_4103


namespace topological_sort_possible_l4_4506
-- Import the necessary library

-- Definition of simple, directed, and acyclic graph (DAG)
structure SimpleDirectedAcyclicGraph (V : Type*) :=
  (E : V → V → Prop)
  (acyclic : ∀ v : V, ¬(E v v)) -- no loops
  (simple : ∀ (u v : V), (E u v) → ¬(E v u)) -- no bidirectional edges
  (directional : ∀ (u v w : V), E u v → E v w → E u w) -- directional transitivity

-- Existence of topological sort definition
def topological_sort_exists {V : Type*} (G : SimpleDirectedAcyclicGraph V) : Prop :=
  ∃ (numbering : V → ℕ), ∀ (u v : V), (G.E u v) → (numbering u > numbering v)

-- Theorem statement
theorem topological_sort_possible (V : Type*) (G : SimpleDirectedAcyclicGraph V) : topological_sort_exists G :=
  sorry

end topological_sort_possible_l4_4506


namespace trig_identity_l4_4933

theorem trig_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l4_4933


namespace units_digit_of_x4_plus_inv_x4_l4_4564

theorem units_digit_of_x4_plus_inv_x4 (x : ℝ) (hx : x^2 - 13 * x + 1 = 0) : 
  (x^4 + x⁻¹ ^ 4) % 10 = 7 := sorry

end units_digit_of_x4_plus_inv_x4_l4_4564


namespace solution_set_M_abs_ineq_l4_4277

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | 1 < x ∧ x < 4}

-- The first statement to prove the solution set M for the inequality
theorem solution_set_M : ∀ x, f x < 3 ↔ x ∈ M :=
by sorry

-- The second statement to prove the inequality when a, b ∈ M
theorem abs_ineq (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + ab| :=
by sorry

end solution_set_M_abs_ineq_l4_4277


namespace line_through_midpoint_l4_4887

theorem line_through_midpoint (x y : ℝ) (P : x = 2 ∧ y = -1) :
  (∃ l : ℝ, ∀ t : ℝ, 
  (1 + 5 * Real.cos t = x) ∧ (5 * Real.sin t = y) →
  (x - y = 3)) :=
by
  sorry

end line_through_midpoint_l4_4887


namespace one_fourth_in_one_eighth_l4_4878

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end one_fourth_in_one_eighth_l4_4878


namespace distribution_ways_5_to_3_l4_4112

noncomputable def num_ways (n m : ℕ) : ℕ :=
  m ^ n

theorem distribution_ways_5_to_3 : num_ways 5 3 = 243 := by
  sorry

end distribution_ways_5_to_3_l4_4112


namespace sqrt_20_19_18_17_plus_1_eq_341_l4_4145

theorem sqrt_20_19_18_17_plus_1_eq_341 :
  Real.sqrt ((20: ℝ) * 19 * 18 * 17 + 1) = 341 := by
sorry

end sqrt_20_19_18_17_plus_1_eq_341_l4_4145


namespace solution_proof_l4_4821

noncomputable def problem_statement : Prop :=
  ((16^(1/4) * 32^(1/5)) + 64^(1/6)) = 6

theorem solution_proof : problem_statement :=
by
  sorry

end solution_proof_l4_4821


namespace student_community_arrangement_l4_4430

theorem student_community_arrangement :
  let students := 4
  let communities := 3
  (students.choose 2) * (communities.factorial / (communities - (students - 1)).factorial) = 36 :=
by
  have students := 4
  have communities := 3
  sorry

end student_community_arrangement_l4_4430


namespace determine_m_value_l4_4067

-- Define the condition that the roots of the quadratic are given
def quadratic_equation_has_given_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, (8 * x^2 + 4 * x + m = 0) → (x = (-2 + (Complex.I * Real.sqrt 88)) / 8) ∨ (x = (-2 - (Complex.I * Real.sqrt 88)) / 8)

-- The main statement to be proven
theorem determine_m_value (m : ℝ) (h : quadratic_equation_has_given_roots m) : m = 13 / 4 :=
sorry

end determine_m_value_l4_4067


namespace diane_age_proof_l4_4156

noncomputable def diane_age (A Al D : ℕ) : Prop :=
  ((A + (30 - D) = 60) ∧ (Al + (30 - D) = 15) ∧ (A + Al = 47)) → (D = 16)

theorem diane_age_proof : ∃ (D : ℕ), ∃ (A Al : ℕ), diane_age A Al D :=
by {
  sorry
}

end diane_age_proof_l4_4156


namespace arithmetic_seq_sum_l4_4367

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_sum : a 0 + a 1 + a 2 + a 3 = 30) : a 1 + a 2 = 15 :=
by
  sorry

end arithmetic_seq_sum_l4_4367


namespace value_of_y_l4_4197

theorem value_of_y (x y : ℤ) (h1 : x - y = 6) (h2 : x + y = 12) : y = 3 := 
by
  sorry

end value_of_y_l4_4197


namespace mul_72519_9999_eq_725117481_l4_4486

theorem mul_72519_9999_eq_725117481 : 72519 * 9999 = 725117481 := by
  sorry

end mul_72519_9999_eq_725117481_l4_4486


namespace felix_trees_per_sharpening_l4_4538

theorem felix_trees_per_sharpening (dollars_spent : ℕ) (cost_per_sharpen : ℕ) (trees_chopped : ℕ) 
  (h1 : dollars_spent = 35) (h2 : cost_per_sharpen = 5) (h3 : trees_chopped ≥ 91) :
  (91 / (35 / 5)) = 13 := 
by 
  sorry

end felix_trees_per_sharpening_l4_4538


namespace num_distinct_pos_factors_81_l4_4198

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end num_distinct_pos_factors_81_l4_4198


namespace power_six_tens_digit_l4_4596

def tens_digit (x : ℕ) : ℕ := (x / 10) % 10

theorem power_six_tens_digit (n : ℕ) (hn : tens_digit (6^n) = 1) : n = 3 :=
sorry

end power_six_tens_digit_l4_4596


namespace travel_time_in_minutes_l4_4042

def bird_speed : ℝ := 8 -- Speed of the bird in miles per hour
def distance_to_travel : ℝ := 3 -- Distance to be traveled in miles

theorem travel_time_in_minutes : (distance_to_travel / bird_speed) * 60 = 22.5 :=
by
  sorry

end travel_time_in_minutes_l4_4042


namespace reeya_fifth_score_l4_4532

theorem reeya_fifth_score
  (s1 s2 s3 s4 avg: ℝ)
  (h1: s1 = 65)
  (h2: s2 = 67)
  (h3: s3 = 76)
  (h4: s4 = 82)
  (h_avg: avg = 75) :
  ∃ s5, s1 + s2 + s3 + s4 + s5 = 5 * avg ∧ s5 = 85 :=
by
  use 85
  sorry

end reeya_fifth_score_l4_4532


namespace no_solution_xn_yn_zn_l4_4021

theorem no_solution_xn_yn_zn (x y z n : ℕ) (h : n ≥ z) : ¬ (x^n + y^n = z^n) :=
sorry

end no_solution_xn_yn_zn_l4_4021


namespace find_m_l4_4514

noncomputable def ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / 25) + (p.2 ^ 2 / 16) = 1}
noncomputable def hyperbola (m : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / m) - (p.2 ^ 2 / 5) = 1}

theorem find_m (m : ℝ) (h1 : ∃ f : ℝ × ℝ, f ∈ ellipse ∧ f ∈ hyperbola m) : m = 4 := by
  sorry

end find_m_l4_4514


namespace value_of_v3_at_2_l4_4218

def f (x : ℝ) : ℝ := x^5 - 2 * x^4 + 3 * x^3 - 7 * x^2 + 6 * x - 3

def v3 (x : ℝ) := (x - 2) * x + 3 
def v3_eval_at_2 : ℝ := (2 - 2) * 2 + 3

theorem value_of_v3_at_2 : v3 2 - 7 = -1 := by
    sorry

end value_of_v3_at_2_l4_4218


namespace trig_identity_l4_4035

theorem trig_identity :
  (Real.cos (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (48 * Real.pi / 180) * Real.sin (18 * Real.pi / 180)) = 1 / 2 := 
by sorry

end trig_identity_l4_4035


namespace cos_sin_identity_l4_4344

theorem cos_sin_identity : 
  (Real.cos (14 * Real.pi / 180) * Real.cos (59 * Real.pi / 180) + 
   Real.sin (14 * Real.pi / 180) * Real.sin (121 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end cos_sin_identity_l4_4344


namespace train_length_l4_4210

theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) (total_distance : ℝ) (train_length : ℝ) 
  (h1 : speed = 48) (h2 : time = 45) (h3 : bridge_length = 300)
  (h4 : total_distance = speed * time) (h5 : train_length = total_distance - bridge_length) : 
  train_length = 1860 :=
sorry

end train_length_l4_4210


namespace sum_of_cubes_minus_tripled_product_l4_4064

theorem sum_of_cubes_minus_tripled_product (a b c d : ℝ) 
  (h1 : a + b + c + d = 15)
  (h2 : ab + ac + ad + bc + bd + cd = 40) :
  a^3 + b^3 + c^3 + d^3 - 3 * a * b * c * d = 1695 :=
by
  sorry

end sum_of_cubes_minus_tripled_product_l4_4064


namespace quadratic_solution_downward_solution_minimum_solution_l4_4017

def is_quadratic (m : ℝ) : Prop :=
  m^2 + 3 * m - 2 = 2

def opens_downwards (m : ℝ) : Prop :=
  m + 3 < 0

def has_minimum (m : ℝ) : Prop :=
  m + 3 > 0

theorem quadratic_solution (m : ℝ) :
  is_quadratic m → (m = -4 ∨ m = 1) :=
sorry

theorem downward_solution (m : ℝ) :
  is_quadratic m → opens_downwards m → m = -4 :=
sorry

theorem minimum_solution (m : ℝ) :
  is_quadratic m → has_minimum m → m = 1 :=
sorry

end quadratic_solution_downward_solution_minimum_solution_l4_4017


namespace count_valid_four_digit_numbers_l4_4315

theorem count_valid_four_digit_numbers : 
  let valid_first_digits := (4*5 + 4*4)
  let valid_last_digits := (5*5 + 4*4)
  valid_first_digits * valid_last_digits = 1476 :=
by
  sorry

end count_valid_four_digit_numbers_l4_4315


namespace Kvi_wins_race_l4_4133

/-- Define the frogs and their properties --/
structure Frog :=
  (name : String)
  (jump_distance_in_dm : ℕ) /-- jump distance in decimeters --/
  (jumps_per_cycle : ℕ) /-- number of jumps per cycle (unit time of reference) --/

def FrogKva : Frog := ⟨"Kva", 6, 2⟩
def FrogKvi : Frog := ⟨"Kvi", 4, 3⟩

/-- Define the conditions for the race --/
def total_distance_in_m : ℕ := 40
def total_distance_in_dm := total_distance_in_m * 10

/-- Racing function to determine winner --/
def race_winner (f1 f2 : Frog) (total_distance : ℕ) : String :=
  if (total_distance % (f1.jump_distance_in_dm * f1.jumps_per_cycle) < total_distance % (f2.jump_distance_in_dm * f2.jumps_per_cycle))
  then f1.name
  else f2.name

/-- Proving Kvi wins under the given conditions --/
theorem Kvi_wins_race :
  race_winner FrogKva FrogKvi total_distance_in_dm = "Kvi" :=
by
  sorry

end Kvi_wins_race_l4_4133


namespace find_e_of_x_l4_4597

noncomputable def x_plus_inv_x_eq_five (x : ℝ) : Prop :=
  x + (1 / x) = 5

theorem find_e_of_x (x : ℝ) (h : x_plus_inv_x_eq_five x) : 
  x^2 + (1 / x)^2 = 23 := sorry

end find_e_of_x_l4_4597


namespace probability_of_exactly_9_correct_matches_is_zero_l4_4174

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l4_4174


namespace solve_quartic_eqn_l4_4415

noncomputable def solutionSet : Set ℂ :=
  {x | x^2 = 6 ∨ x^2 = -6}

theorem solve_quartic_eqn (x : ℂ) : (x^4 - 36 = 0) ↔ (x ∈ solutionSet) := 
sorry

end solve_quartic_eqn_l4_4415


namespace relationship_of_ys_l4_4537

variable (f : ℝ → ℝ)

def inverse_proportion := ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k / x

theorem relationship_of_ys
  (h_inv_prop : inverse_proportion f)
  (h_pts1 : f (-2) = 3)
  (h_pts2 : f (-3) = y₁)
  (h_pts3 : f 1 = y₂)
  (h_pts4 : f 2 = y₃) :
  y₂ < y₃ ∧ y₃ < y₁ :=
sorry

end relationship_of_ys_l4_4537


namespace domain_of_f1_x2_l4_4973

theorem domain_of_f1_x2 (f : ℝ → ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → ∃ y, y = f x) → 
  (∀ x, -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → ∃ y, y = f (1 - x^2)) :=
by
  sorry

end domain_of_f1_x2_l4_4973


namespace proof_statement_l4_4441

-- Assume 5 * 3^x = 243
def condition (x : ℝ) : Prop := 5 * (3:ℝ)^x = 243

-- Define the log base 3 for use in the statement
noncomputable def log_base_3 (y : ℝ) : ℝ := Real.log y / Real.log 3

-- State that if the condition holds, then (x + 2)(x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2
theorem proof_statement (x : ℝ) (h : condition x) : (x + 2) * (x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2 := sorry

end proof_statement_l4_4441


namespace simplify_expression_l4_4387

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l4_4387


namespace value_of_ab_over_cd_l4_4193

theorem value_of_ab_over_cd (a b c d : ℚ) (h₁ : a / b = 2 / 3) (h₂ : c / b = 1 / 5) (h₃ : c / d = 7 / 15) : (a * b) / (c * d) = 140 / 9 :=
by
  sorry

end value_of_ab_over_cd_l4_4193


namespace original_price_of_second_pair_l4_4915

variable (P : ℝ) -- original price of the second pair of shoes
variable (discounted_price : ℝ := P / 2)
variable (total_before_discount : ℝ := 40 + discounted_price)
variable (final_payment : ℝ := (3 / 4) * total_before_discount)
variable (payment : ℝ := 60)

theorem original_price_of_second_pair (h : final_payment = payment) : P = 80 :=
by
  -- Skipping the proof with sorry.
  sorry

end original_price_of_second_pair_l4_4915


namespace difference_ne_1998_l4_4025

-- Define the function f(n) = n^2 + 4n
def f (n : ℕ) : ℕ := n^2 + 4 * n

-- Statement: For all natural numbers n and m, the difference f(n) - f(m) is not 1998
theorem difference_ne_1998 (n m : ℕ) : f n - f m ≠ 1998 := 
by {
  sorry
}

end difference_ne_1998_l4_4025


namespace unique_sequence_l4_4844

theorem unique_sequence (a : ℕ → ℝ) 
  (h1 : a 0 = 1) 
  (h2 : ∀ n : ℕ, a n > 0) 
  (h3 : ∀ n : ℕ, a n - a (n + 1) = a (n + 2)) : 
  ∀ n : ℕ, a n = ( (-1 + Real.sqrt 5) / 2)^n := 
sorry

end unique_sequence_l4_4844


namespace James_beat_old_record_by_296_points_l4_4629

def touchdowns_per_game := 4
def points_per_touchdown := 6
def number_of_games := 15
def two_point_conversions := 6
def points_per_two_point_conversion := 2
def field_goals := 8
def points_per_field_goal := 3
def extra_point_attempts := 20
def points_per_extra_point := 1
def consecutive_touchdowns := 3
def games_with_consecutive_touchdowns := 5
def bonus_multiplier := 2
def old_record := 300

def James_points : ℕ :=
  (touchdowns_per_game * number_of_games * points_per_touchdown) + 
  ((consecutive_touchdowns * games_with_consecutive_touchdowns) * points_per_touchdown * bonus_multiplier) +
  (two_point_conversions * points_per_two_point_conversion) +
  (field_goals * points_per_field_goal) +
  (extra_point_attempts * points_per_extra_point)

def points_above_old_record := James_points - old_record

theorem James_beat_old_record_by_296_points : points_above_old_record = 296 := by
  -- here would be the proof
  sorry

end James_beat_old_record_by_296_points_l4_4629


namespace power_function_properties_l4_4960

theorem power_function_properties (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ a) (h2 : f 2 = Real.sqrt 2) : 
  a = 1 / 2 ∧ ∀ x, 0 ≤ x → f x ≤ f (x + 1) :=
by
  sorry

end power_function_properties_l4_4960


namespace sin_double_angle_l4_4909

theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 1 / 3) (h2 : (π / 2) < α ∧ α < π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := sorry

end sin_double_angle_l4_4909


namespace smallest_number_l4_4808

-- Define the numbers
def A := 5.67823
def B := 5.67833333333 -- repeating decimal
def C := 5.67838383838 -- repeating decimal
def D := 5.67837837837 -- repeating decimal
def E := 5.6783678367  -- repeating decimal

-- The Lean statement to prove that E is the smallest
theorem smallest_number : E < A ∧ E < B ∧ E < C ∧ E < D :=
by
  sorry

end smallest_number_l4_4808


namespace fourth_number_value_l4_4402

variable (A B C D E F : ℝ)

theorem fourth_number_value 
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) : 
  D = 25 := 
by 
  sorry

end fourth_number_value_l4_4402


namespace sum_of_consecutive_page_numbers_l4_4638

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20250) : n + (n + 1) = 285 := 
sorry

end sum_of_consecutive_page_numbers_l4_4638


namespace binary_calculation_l4_4813

-- Binary arithmetic definition
def binary_mul (a b : Nat) : Nat := a * b
def binary_div (a b : Nat) : Nat := a / b

-- Binary numbers in Nat (representing binary literals by their decimal equivalent)
def b110010 := 50   -- 110010_2 in decimal
def b101000 := 40   -- 101000_2 in decimal
def b100 := 4       -- 100_2 in decimal
def b10 := 2        -- 10_2 in decimal
def b10111000 := 184-- 10111000_2 in decimal

theorem binary_calculation :
  binary_div (binary_div (binary_mul b110010 b101000) b100) b10 = b10111000 :=
by
  sorry

end binary_calculation_l4_4813


namespace student_rank_from_left_l4_4687

theorem student_rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h1 : total_students = 21) 
  (h2 : rank_from_right = 16) 
  (h3 : total_students = rank_from_right + rank_from_left - 1) 
  : rank_from_left = 6 := 
by 
  sorry

end student_rank_from_left_l4_4687


namespace driving_scenario_l4_4713

theorem driving_scenario (x : ℝ) (h1 : x > 0) :
  (240 / x) - (240 / (1.5 * x)) = 1 :=
by
  sorry

end driving_scenario_l4_4713


namespace roots_modulus_less_than_one_l4_4767

theorem roots_modulus_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ x, x^2 + A * x + B = 0 → |x| < 1)
  (h2 : ∀ x, x^2 + C * x + D = 0 → |x| < 1) :
  ∀ x, x^2 + (A + C) / 2 * x + (B + D) / 2 = 0 → |x| < 1 :=
by
  sorry

end roots_modulus_less_than_one_l4_4767


namespace find_a_l4_4461

theorem find_a (a x_0 : ℝ) (h_tangent: (ax_0^3 + 1 = x_0) ∧ (3 * a * x_0^2 = 1)) : a = 4 / 27 :=
sorry

end find_a_l4_4461


namespace solve_numRedBalls_l4_4118

-- Condition (1): There are a total of 10 balls in the bag
def totalBalls : ℕ := 10

-- Condition (2): The probability of drawing a black ball is 2/5
-- This means the number of black balls is 4
def numBlackBalls : ℕ := 4

-- Condition (3): The probability of drawing at least 1 white ball when drawing 2 balls is 7/9
def probAtLeastOneWhiteBall : ℚ := 7 / 9

-- The number of red balls in the bag is calculated based on the given conditions
def numRedBalls (totalBalls numBlackBalls : ℕ) (probAtLeastOneWhiteBall : ℚ) : ℕ := 
  let totalWhiteAndRedBalls := totalBalls - numBlackBalls
  let probTwoNonWhiteBalls := 1 - probAtLeastOneWhiteBall
  let comb (n k : ℕ) := Nat.choose n k
  let equation := comb totalWhiteAndRedBalls 2 * comb (totalBalls - 2) 0 / comb totalBalls 2
  if equation = probTwoNonWhiteBalls then totalWhiteAndRedBalls else 0

theorem solve_numRedBalls : numRedBalls totalBalls numBlackBalls probAtLeastOneWhiteBall = 1 := by
  sorry

end solve_numRedBalls_l4_4118


namespace part_I_part_II_l4_4995

open Set

variable (a b : ℝ)

theorem part_I (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hB_sub_A : B ⊆ A) : a = -1 :=
  sorry

theorem part_II (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hA_inter_B_nonempty : A ∩ B ≠ ∅) : a^2 - b^2 + 2 * a = -1 :=
  sorry

end part_I_part_II_l4_4995


namespace egg_sales_l4_4396

/-- Two vendors together sell 110 eggs and both have equal revenues.
    Given the conditions about changing the number of eggs and corresponding revenues,
    the first vendor sells 60 eggs and the second vendor sells 50 eggs. -/
theorem egg_sales (x y : ℝ) (h1 : x + (110 - x) = 110) (h2 : 110 * (y / x) = 5) (h3 : 110 * (y / (110 - x)) = 7.2) :
  x = 60 ∧ (110 - x) = 50 :=
by sorry

end egg_sales_l4_4396


namespace circle_equation_a_value_l4_4073

theorem circle_equation_a_value (a : ℝ) : (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
sorry

end circle_equation_a_value_l4_4073
