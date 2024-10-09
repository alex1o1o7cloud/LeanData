import Mathlib

namespace find_AC_find_angle_A_l1449_144931

noncomputable def triangle_AC (AB BC : ℝ) (sinC_over_sinB : ℝ) : ℝ :=
  if h : sinC_over_sinB = 3 / 5 ∧ AB = 3 ∧ BC = 7 then 5 else 0

noncomputable def triangle_angle_A (AB AC BC : ℝ) : ℝ :=
  if h : AB = 3 ∧ AC = 5 ∧ BC = 7 then 120 else 0

theorem find_AC (BC AB : ℝ) (sinC_over_sinB : ℝ) (h : BC = 7 ∧ AB = 3 ∧ sinC_over_sinB = 3 / 5) : 
  triangle_AC AB BC sinC_over_sinB = 5 := by
  sorry

theorem find_angle_A (BC AB AC : ℝ) (h : BC = 7 ∧ AB = 3 ∧ AC = 5) : 
  triangle_angle_A AB AC BC = 120 := by
  sorry

end find_AC_find_angle_A_l1449_144931


namespace total_amount_proof_l1449_144943

-- Definitions of the base 8 numbers
def silks_base8 := 5267
def stones_base8 := 6712
def spices_base8 := 327

-- Conversion function from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ := sorry -- Assume this function converts a base 8 number to base 10

-- Converted values
def silks_base10 := base8_to_base10 silks_base8
def stones_base10 := base8_to_base10 stones_base8
def spices_base10 := base8_to_base10 spices_base8

-- Total amount calculation in base 10
def total_amount_base10 := silks_base10 + stones_base10 + spices_base10

-- The theorem that we want to prove
theorem total_amount_proof : total_amount_base10 = 6488 :=
by
  -- The proof is omitted here.
  sorry

end total_amount_proof_l1449_144943


namespace total_tourists_proof_l1449_144928

noncomputable def calculate_total_tourists : ℕ :=
  let start_time := 8  
  let end_time := 17   -- 5 PM in 24-hour format
  let initial_tourists := 120
  let increment := 2
  let number_of_trips := end_time - start_time  -- total number of trips including both start and end
  let first_term := initial_tourists
  let last_term := initial_tourists + increment * (number_of_trips - 1)
  (number_of_trips * (first_term + last_term)) / 2

theorem total_tourists_proof : calculate_total_tourists = 1290 := by
  sorry

end total_tourists_proof_l1449_144928


namespace systematic_sampling_probability_l1449_144947

/-- Given a population of 1002 individuals, if we remove 2 randomly and then pick 50 out of the remaining 1000, then the probability of picking each individual is 50/1002. 
This is because the process involves two independent steps: not being removed initially and then being chosen in the sample of size 50. --/
theorem systematic_sampling_probability :
  let population_size := 1002
  let removal_count := 2
  let sample_size := 50
  ∀ p : ℕ, p = 50 / (1002 : ℚ) := sorry

end systematic_sampling_probability_l1449_144947


namespace ratio_of_a_b_l1449_144937

variable (x y a b : ℝ)

theorem ratio_of_a_b (h₁ : 4 * x - 2 * y = a)
                     (h₂ : 6 * y - 12 * x = b)
                     (hb : b ≠ 0)
                     (ha_solution : ∃ x y, 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b) :
                     a / b = 1 / 3 :=
by sorry

end ratio_of_a_b_l1449_144937


namespace mod_sum_l1449_144906

theorem mod_sum : 
  (5432 + 5433 + 5434 + 5435) % 7 = 2 := 
by
  sorry

end mod_sum_l1449_144906


namespace possible_values_of_a_l1449_144981

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - a*x + 5 else a / x

theorem possible_values_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) ↔ (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end possible_values_of_a_l1449_144981


namespace g_six_g_seven_l1449_144994

noncomputable def g : ℝ → ℝ :=
sorry

axiom additivity : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_three : g 3 = 4

theorem g_six : g 6 = 8 :=
by {
  -- proof steps to be added by the prover
  sorry
}

theorem g_seven : g 7 = 28 / 3 :=
by {
  -- proof steps to be added by the prover
  sorry
}

end g_six_g_seven_l1449_144994


namespace isosceles_triangle_perimeter_l1449_144970

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), 
  (a = 3 ∧ b = 6 ∧ (c = 6 ∨ c = 3)) ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
  (a + b + c = 15) :=
sorry

end isosceles_triangle_perimeter_l1449_144970


namespace dots_not_visible_l1449_144913

-- Define the sum of numbers on a single die
def sum_die_faces : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Define the sum of numbers on four dice
def total_dots_on_four_dice : ℕ := 4 * sum_die_faces

-- List the visible numbers
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 4, 5, 5, 6]

-- Calculate the sum of visible numbers
def sum_visible_numbers : ℕ := (visible_numbers.sum)

-- Define the math proof problem
theorem dots_not_visible : total_dots_on_four_dice - sum_visible_numbers = 53 := by
  sorry

end dots_not_visible_l1449_144913


namespace avg_b_c_is_45_l1449_144968

-- Define the weights of a, b, and c
variables (a b c : ℝ)

-- Conditions given in the problem
def avg_a_b_c (a b c : ℝ) := (a + b + c) / 3 = 45
def avg_a_b (a b : ℝ) := (a + b) / 2 = 40
def weight_b (b : ℝ) := b = 35

-- Theorem statement
theorem avg_b_c_is_45 (a b c : ℝ) (h1 : avg_a_b_c a b c) (h2 : avg_a_b a b) (h3 : weight_b b) :
  (b + c) / 2 = 45 := by
  -- Proof omitted for brevity
  sorry

end avg_b_c_is_45_l1449_144968


namespace solve_inequality_l1449_144938

theorem solve_inequality (a x : ℝ) :
  (a - x) * (x - 1) < 0 ↔
  (a > 1 ∧ (x < 1 ∨ x > a)) ∨
  (a < 1 ∧ (x < a ∨ x > 1)) ∨
  (a = 1 ∧ x ≠ 1) :=
by
  sorry

end solve_inequality_l1449_144938


namespace sum_of_n_values_l1449_144935

theorem sum_of_n_values (sum_n : ℕ) : (∀ n : ℕ, 0 < n ∧ 24 % (2 * n - 1) = 0) → sum_n = 3 :=
by
  sorry

end sum_of_n_values_l1449_144935


namespace calculate_expression_l1449_144932

theorem calculate_expression :
  |(-Real.sqrt 3)| - (1/3)^(-1/2 : ℝ) + 2 / (Real.sqrt 3 - 1) - 12^(1/2 : ℝ) = 1 - Real.sqrt 3 :=
by
  sorry

end calculate_expression_l1449_144932


namespace spherical_to_rectangular_coordinates_l1449_144942

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_coordinates :
  sphericalToRectangular 10 (5 * Real.pi / 4) (Real.pi / 4) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l1449_144942


namespace max_b_value_l1449_144952

theorem max_b_value (a b c : ℕ) (h_volume : a * b * c = 360) (h_conditions : 1 < c ∧ c < b ∧ b < a) : b = 12 :=
  sorry

end max_b_value_l1449_144952


namespace function_form_l1449_144951

noncomputable def f : ℕ → ℕ := sorry

theorem function_form (c d a : ℕ) (h1 : c > 1) (h2 : a - c > 1)
  (hf : ∀ n : ℕ, f n + f (n + 1) = f (n + 2) + f (n + 3) - 168) :
  (∀ n : ℕ, f (2 * n) = c + n * d) ∧ (∀ n : ℕ, f (2 * n + 1) = (168 - d) * n + a - c) :=
sorry

end function_form_l1449_144951


namespace harry_morning_routine_l1449_144988

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end harry_morning_routine_l1449_144988


namespace two_pi_irrational_l1449_144975

-- Assuming \(\pi\) is irrational as is commonly accepted
def irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

theorem two_pi_irrational : irrational (2 * Real.pi) := 
by 
  sorry

end two_pi_irrational_l1449_144975


namespace sum_of_other_endpoint_coordinates_l1449_144945

theorem sum_of_other_endpoint_coordinates (x y : ℝ) (hx : (x + 5) / 2 = 3) (hy : (y - 2) / 2 = 4) :
  x + y = 11 :=
sorry

end sum_of_other_endpoint_coordinates_l1449_144945


namespace num_triangles_l1449_144929

def vertices := 10
def chosen_vertices := 3

theorem num_triangles : (Nat.choose vertices chosen_vertices) = 120 := by
  sorry

end num_triangles_l1449_144929


namespace arithmetic_mean_of_a_and_b_is_sqrt3_l1449_144965

theorem arithmetic_mean_of_a_and_b_is_sqrt3 :
  let a := (Real.sqrt 3 + Real.sqrt 2)
  let b := (Real.sqrt 3 - Real.sqrt 2)
  (a + b) / 2 = Real.sqrt 3 := 
by
  sorry

end arithmetic_mean_of_a_and_b_is_sqrt3_l1449_144965


namespace power_sum_l1449_144956

theorem power_sum : 1^234 + 4^6 / 4^4 = 17 :=
by
  sorry

end power_sum_l1449_144956


namespace geometric_series_common_ratio_l1449_144971

theorem geometric_series_common_ratio (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) 
  (h₃ : S = a / (1 - r)) : r = 21 / 25 := 
sorry

end geometric_series_common_ratio_l1449_144971


namespace fountain_pen_price_l1449_144940

theorem fountain_pen_price
  (n_fpens : ℕ) (n_mpens : ℕ) (total_cost : ℕ) (avg_cost_mpens : ℝ)
  (hpens : n_fpens = 450) (mpens : n_mpens = 3750) 
  (htotal : total_cost = 11250) (havg_mpens : avg_cost_mpens = 2.25) : 
  (total_cost - n_mpens * avg_cost_mpens) / n_fpens = 6.25 :=
by
  sorry

end fountain_pen_price_l1449_144940


namespace purchase_price_of_grinder_l1449_144991

theorem purchase_price_of_grinder (G : ℝ) (H : 0.95 * G + 8800 - (G + 8000) = 50) : G = 15000 := 
sorry

end purchase_price_of_grinder_l1449_144991


namespace number_is_minus_72_l1449_144909

noncomputable def find_number (x : ℝ) : Prop :=
  0.833 * x = -60

theorem number_is_minus_72 : ∃ x : ℝ, find_number x ∧ x = -72 :=
by
  sorry

end number_is_minus_72_l1449_144909


namespace Angie_necessities_amount_l1449_144979

noncomputable def Angie_salary : ℕ := 80
noncomputable def Angie_left_over : ℕ := 18
noncomputable def Angie_taxes : ℕ := 20
noncomputable def Angie_expenses : ℕ := Angie_salary - Angie_left_over
noncomputable def Angie_necessities : ℕ := Angie_expenses - Angie_taxes

theorem Angie_necessities_amount :
  Angie_necessities = 42 :=
by
  unfold Angie_necessities
  unfold Angie_expenses
  sorry

end Angie_necessities_amount_l1449_144979


namespace quadratic_function_fixed_points_range_l1449_144953

def has_two_distinct_fixed_points (c : ℝ) : Prop := 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
               (x1 = x1^2 - x1 + c) ∧ 
               (x2 = x2^2 - x2 + c) ∧ 
               x1 < 2 ∧ 2 < x2

theorem quadratic_function_fixed_points_range (c : ℝ) :
  has_two_distinct_fixed_points c ↔ c < 0 :=
sorry

end quadratic_function_fixed_points_range_l1449_144953


namespace statement_B_statement_C_statement_D_l1449_144902

-- Statement B
theorem statement_B (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a^3 * c < b^3 * c :=
sorry

-- Statement C
theorem statement_C (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) : (a / (c - a)) > (b / (c - b)) :=
sorry

-- Statement D
theorem statement_D (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ b < 0 :=
sorry

end statement_B_statement_C_statement_D_l1449_144902


namespace sequence_sum_l1449_144946

-- Definitions for the sequences
def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

-- The theorem we need to prove
theorem sequence_sum : a (b 1) + a (b 2) + a (b 3) + a (b 4) = 19 := by
  sorry

end sequence_sum_l1449_144946


namespace average_balance_correct_l1449_144917

-- Define the monthly balances
def january_balance : ℕ := 120
def february_balance : ℕ := 240
def march_balance : ℕ := 180
def april_balance : ℕ := 180
def may_balance : ℕ := 210
def june_balance : ℕ := 300

-- List of all balances
def balances : List ℕ := [january_balance, february_balance, march_balance, april_balance, may_balance, june_balance]

-- Define the function to calculate the average balance
def average_balance (balances : List ℕ) : ℕ :=
  (balances.sum / balances.length)

-- Define the target average balance
def target_average_balance : ℕ := 205

-- The theorem we need to prove
theorem average_balance_correct :
  average_balance balances = target_average_balance :=
by
  sorry

end average_balance_correct_l1449_144917


namespace algebraic_expression_value_l1449_144926

theorem algebraic_expression_value (x : ℝ) :
  let a := 2003 * x + 2001
  let b := 2003 * x + 2002
  let c := 2003 * x + 2003
  a^2 + b^2 + c^2 - a * b - a * c - b * c = 3 :=
by
  sorry

end algebraic_expression_value_l1449_144926


namespace number_of_members_after_four_years_l1449_144963

theorem number_of_members_after_four_years (b : ℕ → ℕ) (initial_condition : b 0 = 21) 
    (yearly_update : ∀ k, b (k + 1) = 4 * b k - 9) : 
    b 4 = 4611 :=
    sorry

end number_of_members_after_four_years_l1449_144963


namespace combined_weight_loss_l1449_144910

theorem combined_weight_loss (a_weekly_loss : ℝ) (a_weeks : ℕ) (x_weekly_loss : ℝ) (x_weeks : ℕ)
  (h1 : a_weekly_loss = 1.5) (h2 : a_weeks = 10) (h3 : x_weekly_loss = 2.5) (h4 : x_weeks = 8) :
  a_weekly_loss * a_weeks + x_weekly_loss * x_weeks = 35 := 
by
  -- We will not provide the proof body; the goal is to ensure the statement compiles.
  sorry

end combined_weight_loss_l1449_144910


namespace number_of_people_in_family_l1449_144990

-- Define the conditions
def planned_spending : ℝ := 15
def savings_percentage : ℝ := 0.40
def cost_per_orange : ℝ := 1.5

-- Define the proof target: the number of people in the family
theorem number_of_people_in_family : 
  planned_spending * savings_percentage / cost_per_orange = 4 := 
by
  -- sorry to skip the proof; this is for statement only
  sorry

end number_of_people_in_family_l1449_144990


namespace burglar_goods_value_l1449_144978

theorem burglar_goods_value (V : ℝ) (S : ℝ) (S_increased : ℝ) (S_total : ℝ) (h1 : S = V / 5000) (h2 : S_increased = 1.25 * S) (h3 : S_total = S_increased + 2) (h4 : S_total = 12) : V = 40000 := by
  sorry

end burglar_goods_value_l1449_144978


namespace train_speed_l1449_144980

theorem train_speed (distance_meters : ℕ) (time_seconds : ℕ) 
  (h_distance : distance_meters = 150) (h_time : time_seconds = 20) : 
  distance_meters / 1000 / (time_seconds / 3600) = 27 :=
by 
  have h1 : distance_meters = 150 := h_distance
  have h2 : time_seconds = 20 := h_time
  -- other intermediate steps would go here, but are omitted
  -- for now, we assume the final calculation is:
  sorry

end train_speed_l1449_144980


namespace no_square_from_vertices_of_equilateral_triangles_l1449_144985

-- Definitions
def equilateral_triangle_grid (p : ℝ × ℝ) : Prop := 
  ∃ k l : ℤ, p.1 = k * (1 / 2) ∧ p.2 = l * (Real.sqrt 3 / 2)

def form_square_by_vertices (A B C D : ℝ × ℝ) : Prop := 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 ∧ 
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = (D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2 ∧ 
  (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2
  
-- Problem Statement
theorem no_square_from_vertices_of_equilateral_triangles :
  ¬ ∃ (A B C D : ℝ × ℝ), 
    equilateral_triangle_grid A ∧ 
    equilateral_triangle_grid B ∧ 
    equilateral_triangle_grid C ∧ 
    equilateral_triangle_grid D ∧ 
    form_square_by_vertices A B C D :=
by
  sorry

end no_square_from_vertices_of_equilateral_triangles_l1449_144985


namespace average_age_of_team_l1449_144922

theorem average_age_of_team
    (A : ℝ)
    (captain_age : ℝ)
    (wicket_keeper_age : ℝ)
    (bowlers_count : ℝ)
    (batsmen_count : ℝ)
    (team_members_count : ℝ)
    (avg_bowlers_age : ℝ)
    (avg_batsmen_age : ℝ)
    (total_age_team : ℝ) :
    captain_age = 28 →
    wicket_keeper_age = 31 →
    bowlers_count = 5 →
    batsmen_count = 4 →
    avg_bowlers_age = A - 2 →
    avg_batsmen_age = A + 3 →
    total_age_team = 28 + 31 + 5 * (A - 2) + 4 * (A + 3) →
    team_members_count * A = total_age_team →
    team_members_count = 11 →
    A = 30.5 :=
by
  intros
  sorry

end average_age_of_team_l1449_144922


namespace total_flour_l1449_144939

theorem total_flour (original_flour extra_flour : Real) (h_orig : original_flour = 7.0) (h_extra : extra_flour = 2.0) : original_flour + extra_flour = 9.0 :=
sorry

end total_flour_l1449_144939


namespace repeating_decimal_fraction_form_l1449_144987

noncomputable def repeating_decimal_rational := 2.71717171

theorem repeating_decimal_fraction_form : 
  repeating_decimal_rational = 269 / 99 ∧ (269 + 99 = 368) := 
by 
  sorry

end repeating_decimal_fraction_form_l1449_144987


namespace pieces_per_box_l1449_144921

theorem pieces_per_box 
  (a : ℕ) -- Adam bought 13 boxes of chocolate candy 
  (g : ℕ) -- Adam gave 7 boxes to his little brother 
  (p : ℕ) -- Adam still has 36 pieces 
  (n : ℕ) (b : ℕ) 
  (h₁ : a = 13) 
  (h₂ : g = 7) 
  (h₃ : p = 36) 
  (h₄ : n = a - g) 
  (h₅ : p = n * b) 
  : b = 6 :=
by 
  sorry

end pieces_per_box_l1449_144921


namespace find_real_parts_l1449_144915

theorem find_real_parts (a b : ℝ) (i : ℂ) (hi : i*i = -1) 
(h : a + b*i = (1 - i) * i) : a = 1 ∧ b = -1 :=
sorry

end find_real_parts_l1449_144915


namespace optimal_hospital_location_l1449_144907

-- Define the coordinates for points A, B, and C
def A : ℝ × ℝ := (0, 12)
def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

-- Define the distance function
def dist_sq (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the statement to be proved: minimizing sum of squares of distances
theorem optimal_hospital_location : ∃ y : ℝ, 
  (∀ (P : ℝ × ℝ), P = (0, y) → (dist_sq P A + dist_sq P B + dist_sq P C) = 146) ∧ y = 4 :=
by sorry

end optimal_hospital_location_l1449_144907


namespace trapezoid_median_l1449_144977

theorem trapezoid_median 
  (h : ℝ)
  (triangle_base : ℝ := 24)
  (trapezoid_base1 : ℝ := 15)
  (trapezoid_base2 : ℝ := 33)
  (triangle_area_eq_trapezoid_area : (1 / 2) * triangle_base * h = ((trapezoid_base1 + trapezoid_base2) / 2) * h)
  : (trapezoid_base1 + trapezoid_base2) / 2 = 24 :=
by
  sorry

end trapezoid_median_l1449_144977


namespace dealership_sales_l1449_144986

theorem dealership_sales :
  (∀ (n : ℕ), 3 * n ≤ 36 → 5 * n ≤ x) →
  (36 / 3) * 5 = 60 :=
by
  sorry

end dealership_sales_l1449_144986


namespace sum_of_three_numbers_l1449_144911

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h : a + (b * c) = (a + b) * (a + c)) : a + b + c = 1 :=
by
  sorry

end sum_of_three_numbers_l1449_144911


namespace bleaching_takes_3_hours_l1449_144974

-- Define the total time and the relationship between dyeing and bleaching.
def total_time : ℕ := 9
def dyeing_takes_twice (H : ℕ) : Prop := 2 * H + H = total_time

-- Prove that bleaching takes 3 hours.
theorem bleaching_takes_3_hours : ∃ H : ℕ, dyeing_takes_twice H ∧ H = 3 := 
by 
  sorry

end bleaching_takes_3_hours_l1449_144974


namespace range_of_a_l1449_144996

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → 0 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ x₀, -2 ≤ x₀ ∧ x₀ ≤ 2 ∧ (a * x₀ - 1 = f x)) →
  a ∈ Set.Iic (-5/2) ∪ Set.Ici (5/2) :=
sorry

end range_of_a_l1449_144996


namespace find_number_l1449_144944

noncomputable def S (x : ℝ) : ℝ :=
  -- Assuming S(x) is a non-trivial function that sums the digits
  sorry

theorem find_number (x : ℝ) (hx_nonzero : x ≠ 0) (h_cond : x = (S x) / 5) : x = 1.8 :=
by
  sorry

end find_number_l1449_144944


namespace arithmetic_geometric_means_l1449_144908

theorem arithmetic_geometric_means (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 110) : 
  a^2 + b^2 = 1380 :=
sorry

end arithmetic_geometric_means_l1449_144908


namespace calc_f_18_48_l1449_144901

def f (x y : ℕ) : ℕ := sorry

axiom f_self (x : ℕ) : f x x = x
axiom f_symm (x y : ℕ) : f x y = f y x
axiom f_third_cond (x y : ℕ) : (x + y) * f x y = x * f x (x + y)

theorem calc_f_18_48 : f 18 48 = 48 := sorry

end calc_f_18_48_l1449_144901


namespace viewers_watching_program_A_l1449_144930

theorem viewers_watching_program_A (T : ℕ) (hT : T = 560) (x : ℕ)
  (h_ratio : 1 * x + (2 * x - x) + (3 * x - x) = T) : 2 * x = 280 :=
by
  -- by solving the given equation, we find x = 140
  -- substituting x = 140 in 2 * x gives 2 * x = 280
  sorry

end viewers_watching_program_A_l1449_144930


namespace geometric_sequence_arithmetic_l1449_144957

theorem geometric_sequence_arithmetic (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h2 : 2 * S 6 = S 3 + S 9) : 
  q^3 = -1 := 
sorry

end geometric_sequence_arithmetic_l1449_144957


namespace algebraic_expression_value_l1449_144912

namespace MathProof

variables {α β : ℝ} 

-- Given conditions
def is_root (a : ℝ) : Prop := a^2 - a - 1 = 0
def roots_of_quadratic (α β : ℝ) : Prop := is_root α ∧ is_root β

-- The proof problem statement
theorem algebraic_expression_value (h : roots_of_quadratic α β) : α^2 + α * (β^2 - 2) = 0 := 
by sorry

end MathProof

end algebraic_expression_value_l1449_144912


namespace triangle_condition_proof_l1449_144903

variables {A B C D M K : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace K]
variables (AB AC AD : ℝ)

-- Definitions based on the conditions
def is_isosceles (A B C : Type*) (AB AC : ℝ) : Prop :=
  AB = AC

def is_altitude (A D B C : Type*) : Prop :=
  true -- Ideally, this condition is more complex and involves perpendicular projection

def is_midpoint (M A D : Type*) : Prop :=
  true -- Ideally, this condition is more specific and involves equality of segments

def extends_to (C M A B K : Type*) : Prop :=
  true -- Represents the extension relationship

-- The theorem to be proved
theorem triangle_condition_proof (A B C D M K : Type*)
  (h_iso : is_isosceles A B C AB AC)
  (h_alt : is_altitude A D B C)
  (h_mid : is_midpoint M A D)
  (h_ext : extends_to C M A B K)
  : AB = 3 * AK :=
  sorry

end triangle_condition_proof_l1449_144903


namespace hayden_ironing_weeks_l1449_144933

variable (total_daily_minutes : Nat := 5 + 3)
variable (days_per_week : Nat := 5)
variable (total_minutes : Nat := 160)

def calculate_weeks (total_daily_minutes : Nat) (days_per_week : Nat) (total_minutes : Nat) : Nat :=
  total_minutes / (total_daily_minutes * days_per_week)

theorem hayden_ironing_weeks :
  calculate_weeks (5 + 3) 5 160 = 4 := 
by
  sorry

end hayden_ironing_weeks_l1449_144933


namespace scatter_plot_variable_placement_l1449_144918

theorem scatter_plot_variable_placement
  (forecast explanatory : Type)
  (scatter_plot : explanatory → forecast → Prop) : 
  ∀ (x : explanatory) (y : forecast), scatter_plot x y → (True -> True) := 
by
  intros x y h
  sorry

end scatter_plot_variable_placement_l1449_144918


namespace correct_equation_solves_time_l1449_144976

noncomputable def solve_time_before_stop (t : ℝ) : Prop :=
  let total_trip_time := 4 -- total trip time in hours including stop
  let stop_time := 0.5 -- stop time in hours
  let total_distance := 180 -- total distance in km
  let speed_before_stop := 60 -- speed before stop in km/h
  let speed_after_stop := 80 -- speed after stop in km/h
  let time_after_stop := total_trip_time - stop_time - t -- time after the stop in hours
  speed_before_stop * t + speed_after_stop * time_after_stop = total_distance -- distance equation

-- The theorem states that the equation is valid for solving t
theorem correct_equation_solves_time :
  solve_time_before_stop t = (60 * t + 80 * (7/2 - t) = 180) :=
sorry -- proof not required

end correct_equation_solves_time_l1449_144976


namespace pieces_cut_from_rod_l1449_144941

theorem pieces_cut_from_rod (rod_length_m : ℝ) (piece_length_cm : ℝ) (rod_length_cm_eq : rod_length_m * 100 = 4250) (piece_length_eq : piece_length_cm = 85) :
  (4250 / 85) = 50 :=
by sorry

end pieces_cut_from_rod_l1449_144941


namespace number_of_boys_l1449_144966

theorem number_of_boys (total_students girls : ℕ) (h1 : total_students = 13) (h2 : girls = 6) :
  total_students - girls = 7 :=
by 
  -- We'll skip the proof as instructed
  sorry

end number_of_boys_l1449_144966


namespace length_of_BC_l1449_144992

noncomputable def perimeter (a b c : ℝ) := a + b + c
noncomputable def area (b c : ℝ) (A : ℝ) := 0.5 * b * c * (Real.sin A)

theorem length_of_BC
  (a b c : ℝ)
  (h_perimeter : perimeter a b c = 20)
  (h_area : area b c (Real.pi / 3) = 10 * Real.sqrt 3) :
  a = 7 :=
by
  sorry

end length_of_BC_l1449_144992


namespace option_C_incorrect_l1449_144958

def p (x y : ℝ) : ℝ := x^3 - 3 * x^2 * y + 3 * x * y^2 - y^3

theorem option_C_incorrect (x y : ℝ) : 
  ((x^3 - 3 * x^2 * y) - (3 * x * y^2 + y^3)) ≠ p x y := by
  sorry

end option_C_incorrect_l1449_144958


namespace remaining_unit_area_l1449_144961

theorem remaining_unit_area
    (total_units : ℕ)
    (total_area : ℕ)
    (num_12x6_units : ℕ)
    (length_12x6_unit : ℕ)
    (width_12x6_unit : ℕ)
    (remaining_units_area : ℕ)
    (num_remaining_units : ℕ)
    (remaining_unit_area : ℕ) :
  total_units = 72 →
  total_area = 8640 →
  num_12x6_units = 30 →
  length_12x6_unit = 12 →
  width_12x6_unit = 6 →
  remaining_units_area = total_area - (num_12x6_units * length_12x6_unit * width_12x6_unit) →
  num_remaining_units = total_units - num_12x6_units →
  remaining_unit_area = remaining_units_area / num_remaining_units →
  remaining_unit_area = 154 :=
by
  intros h_total_units h_total_area h_num_12x6_units h_length_12x6_unit h_width_12x6_unit h_remaining_units_area h_num_remaining_units h_remaining_unit_area
  sorry

end remaining_unit_area_l1449_144961


namespace victor_score_l1449_144967

-- Definitions based on the conditions
def max_marks : ℕ := 300
def percentage : ℕ := 80

-- Statement to be proved
theorem victor_score : (percentage * max_marks) / 100 = 240 := by
  sorry

end victor_score_l1449_144967


namespace common_difference_value_l1449_144950

-- Define the arithmetic sequence and the sum of the first n terms
def sum_of_arithmetic_sequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

-- Define the given condition in terms of the arithmetic sequence
def given_condition (a1 d : ℚ) : Prop :=
  (sum_of_arithmetic_sequence a1 d 2017) / 2017 - (sum_of_arithmetic_sequence a1 d 17) / 17 = 100

-- Prove the common difference d is 1/10 given the condition
theorem common_difference_value (a1 d : ℚ) :
  given_condition a1 d → d = 1/10 :=
by
  sorry

end common_difference_value_l1449_144950


namespace divisor_greater_2016_l1449_144984

theorem divisor_greater_2016 (d : ℕ) (h : 2016 / d = 0) : d > 2016 :=
sorry

end divisor_greater_2016_l1449_144984


namespace olivia_wallet_final_amount_l1449_144972

variable (initial_money : ℕ) (money_added : ℕ) (money_spent : ℕ)

theorem olivia_wallet_final_amount
  (h1 : initial_money = 100)
  (h2 : money_added = 148)
  (h3 : money_spent = 89) :
  initial_money + money_added - money_spent = 159 :=
  by 
    sorry

end olivia_wallet_final_amount_l1449_144972


namespace greatest_prime_factor_of_341_l1449_144948

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l1449_144948


namespace necklace_cost_l1449_144936

theorem necklace_cost (N : ℕ) (h1 : N + (N + 5) = 73) : N = 34 := by
  sorry

end necklace_cost_l1449_144936


namespace can_measure_all_weights_l1449_144923

def weights : List ℕ := [1, 3, 9, 27]

theorem can_measure_all_weights :
  (∀ n, 1 ≤ n ∧ n ≤ 40 → ∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = n) ∧ 
  (∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = 40) :=
by
  sorry

end can_measure_all_weights_l1449_144923


namespace RSA_next_challenge_digits_l1449_144927

theorem RSA_next_challenge_digits (previous_digits : ℕ) (prize_increase : ℕ) :
  previous_digits = 193 ∧ prize_increase > 10000 → ∃ N : ℕ, N = 212 :=
by {
  sorry -- Proof is omitted
}

end RSA_next_challenge_digits_l1449_144927


namespace algebraic_expression_value_l1449_144905

variable (a b : ℝ)
axiom h1 : a = 3
axiom h2 : a - b = 1

theorem algebraic_expression_value :
  a^2 - a * b = 3 :=
by
  sorry

end algebraic_expression_value_l1449_144905


namespace exists_num_with_digit_sum_div_by_11_l1449_144920

-- Helper function to sum the digits of a natural number
def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem statement
theorem exists_num_with_digit_sum_div_by_11 (N : ℕ) :
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k)) % 11 = 0 :=
sorry

end exists_num_with_digit_sum_div_by_11_l1449_144920


namespace delta_value_l1449_144997

theorem delta_value (Δ : ℤ) : 5 * (-3) = Δ - 3 → Δ = -12 :=
by
  sorry

end delta_value_l1449_144997


namespace LaurynCompanyEmployees_l1449_144993

noncomputable def LaurynTotalEmployees (men women total : ℕ) : Prop :=
  men = 80 ∧ women = men + 20 ∧ total = men + women

theorem LaurynCompanyEmployees : ∃ total, ∀ men women, LaurynTotalEmployees men women total → total = 180 :=
by 
  sorry

end LaurynCompanyEmployees_l1449_144993


namespace smallest_x_value_l1449_144959

open Real

theorem smallest_x_value (x : ℝ) 
  (h : x * abs x = 3 * x + 2) : 
  x = -2 ∨ (∀ y, y * abs y = 3 * y + 2 → y ≥ -2) := sorry

end smallest_x_value_l1449_144959


namespace gcd_12012_18018_l1449_144955

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l1449_144955


namespace smallest_initial_number_sum_of_digits_l1449_144989

theorem smallest_initial_number_sum_of_digits : ∃ (N : ℕ), 
  (0 ≤ N ∧ N < 1000) ∧ 
  ∃ (k : ℕ), 16 * N + 700 + 50 * k < 1000 ∧ 
  (N = 16) ∧ 
  (Nat.digits 10 N).sum = 7 := 
by
  sorry

end smallest_initial_number_sum_of_digits_l1449_144989


namespace polynomial_simplification_l1449_144919

variable (x : ℝ)

theorem polynomial_simplification : 
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 - 4 * x ^ 9 + x ^ 8)) = 
  (15 * x ^ 13 - x ^ 12 - 6 * x ^ 11 - 12 * x ^ 10 + 11 * x ^ 9 - 2 * x ^ 8) := by
  sorry

end polynomial_simplification_l1449_144919


namespace difference_of_squares_l1449_144900

theorem difference_of_squares (x y : ℕ) (h₁ : x + y = 22) (h₂ : x * y = 120) (h₃ : x > y) : 
  x^2 - y^2 = 44 :=
sorry

end difference_of_squares_l1449_144900


namespace joe_average_score_l1449_144982

theorem joe_average_score (A B C : ℕ) (lowest_score : ℕ) (final_average : ℕ) :
  lowest_score = 45 ∧ final_average = 65 ∧ (A + B + C) / 3 = final_average →
  (A + B + C + lowest_score) / 4 = 60 := by
  sorry

end joe_average_score_l1449_144982


namespace sum_of_xyz_l1449_144914

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : 1/x + y + z = 3) 
  (h2 : x + 1/y + z = 3) 
  (h3 : x + y + 1/z = 3) : 
  ∃ m n : ℕ, m = 9 ∧ n = 2 ∧ Nat.gcd m n = 1 ∧ 100 * m + n = 902 := 
sorry

end sum_of_xyz_l1449_144914


namespace height_of_tree_l1449_144964

-- Definitions based on conditions
def net_gain (hop: ℕ) (slip: ℕ) : ℕ := hop - slip

def total_distance (hours: ℕ) (net_gain: ℕ) (final_hop: ℕ) : ℕ :=
  hours * net_gain + final_hop

-- Conditions
def hop : ℕ := 3
def slip : ℕ := 2
def time : ℕ := 20

-- Deriving the net gain per hour
#eval net_gain hop slip  -- Evaluates to 1

-- Final height proof problem
theorem height_of_tree : total_distance 19 (net_gain hop slip) hop = 22 := by
  sorry  -- Proof to be filled in

end height_of_tree_l1449_144964


namespace find_a_l1449_144916

theorem find_a (x a a1 a2 a3 a4 : ℝ) :
  (x + a) ^ 4 = x ^ 4 + a1 * x ^ 3 + a2 * x ^ 2 + a3 * x + a4 → 
  a1 + a2 + a3 = 64 → a = 2 :=
by
  sorry

end find_a_l1449_144916


namespace number_of_juniors_l1449_144983

variables (J S x : ℕ)

theorem number_of_juniors (h1 : (2 / 5 : ℚ) * J = x)
                          (h2 : (1 / 4 : ℚ) * S = x)
                          (h3 : J + S = 30) :
  J = 11 :=
sorry

end number_of_juniors_l1449_144983


namespace max_selection_no_five_times_l1449_144962

theorem max_selection_no_five_times (S : Finset ℕ) (hS : S = Finset.Icc 1 2014) :
  ∃ n, n = 1665 ∧ 
  ∀ (a b : ℕ), a ∈ S → b ∈ S → (a = 5 * b ∨ b = 5 * a) → false :=
sorry

end max_selection_no_five_times_l1449_144962


namespace truck_transportation_l1449_144949

theorem truck_transportation
  (x y t : ℕ) 
  (h1 : xt - yt = 60)
  (h2 : (x - 4) * (t + 10) = xt)
  (h3 : (y - 3) * (t + 10) = yt)
  (h4 : xt = x * t)
  (h5 : yt = y * t) : 
  x - 4 = 8 ∧ y - 3 = 6 ∧ t + 10 = 30 := 
by
  sorry

end truck_transportation_l1449_144949


namespace total_money_spent_l1449_144973

theorem total_money_spent (emma_spent : ℤ) (elsa_spent : ℤ) (elizabeth_spent : ℤ) 
(emma_condition : emma_spent = 58) 
(elsa_condition : elsa_spent = 2 * emma_spent) 
(elizabeth_condition : elizabeth_spent = 4 * elsa_spent) 
:
emma_spent + elsa_spent + elizabeth_spent = 638 :=
by
  rw [emma_condition, elsa_condition, elizabeth_condition]
  norm_num
  sorry

end total_money_spent_l1449_144973


namespace intersection_points_rectangular_coords_l1449_144934

theorem intersection_points_rectangular_coords :
  ∃ (x y : ℝ),
    (∃ (ρ θ : ℝ), ρ = 2 * Real.cos θ ∧ ρ^2 * (Real.cos θ)^2 - 4 * ρ^2 * (Real.sin θ)^2 = 4 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
    (x = (1 + Real.sqrt 13) / 3 ∧ y = 0) := 
sorry

end intersection_points_rectangular_coords_l1449_144934


namespace intersection_points_l1449_144925

open Real

def parabola1 (x : ℝ) : ℝ := x^2 - 3 * x + 2
def parabola2 (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem intersection_points : 
  ∃ x y : ℝ, 
  (parabola1 x = y ∧ parabola2 x = y) ∧ 
  ((x = 1/2 ∧ y = 3/4) ∨ (x = -3 ∧ y = 20)) :=
by sorry

end intersection_points_l1449_144925


namespace min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l1449_144998

variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (h : 4 * a + b = a * b)

theorem min_ab : 16 ≤ a * b :=
sorry

theorem min_a_b : 9 ≤ a + b :=
sorry

theorem max_two_a_one_b : 2 > (2 / a + 1 / b) :=
sorry

theorem min_one_a_sq_four_b_sq : 1 / 5 ≤ (1 / a^2 + 4 / b^2) :=
sorry

end min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l1449_144998


namespace remainder_mod_8_l1449_144904

theorem remainder_mod_8 (x : ℤ) (h : x % 63 = 25) : x % 8 = 1 := 
sorry

end remainder_mod_8_l1449_144904


namespace pictures_at_dolphin_show_l1449_144969

def taken_before : Int := 28
def total_pictures_taken : Int := 44

theorem pictures_at_dolphin_show : total_pictures_taken - taken_before = 16 := by
  -- solution proof goes here
  sorry

end pictures_at_dolphin_show_l1449_144969


namespace Vanya_two_digit_number_l1449_144960

-- Define the conditions as a mathematical property
theorem Vanya_two_digit_number:
  ∃ (m n : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ 0 ≤ n ∧ n ≤ 9 ∧ (10 * n + m) ^ 2 = 4 * (10 * m + n) ∧ (10 * m + n) = 81 :=
by
  -- Remember to replace the proof with 'sorry'
  sorry

end Vanya_two_digit_number_l1449_144960


namespace set_of_a_l1449_144999

theorem set_of_a (a : ℝ) :
  (∃ x : ℝ, a * x ^ 2 + a * x + 1 = 0) → -- Set A contains elements
  (a ≠ 0 ∧ a ^ 2 - 4 * a = 0) →           -- Conditions a ≠ 0 and Δ = 0
  a = 4 := 
sorry

end set_of_a_l1449_144999


namespace triangle_side_range_l1449_144954

theorem triangle_side_range (x : ℝ) (hx1 : 8 + 10 > x) (hx2 : 10 + x > 8) (hx3 : x + 8 > 10) : 2 < x ∧ x < 18 :=
by
  sorry

end triangle_side_range_l1449_144954


namespace product_decrease_increase_fifteenfold_l1449_144924

theorem product_decrease_increase_fifteenfold (a1 a2 a3 a4 a5 : ℕ) :
  ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * a1 * a2 * a3 * a4 * a5) → true :=
by
  sorry

end product_decrease_increase_fifteenfold_l1449_144924


namespace necessary_and_sufficient_condition_l1449_144995

theorem necessary_and_sufficient_condition {a b : ℝ} :
  (a > b) ↔ (a^3 > b^3) := sorry

end necessary_and_sufficient_condition_l1449_144995
