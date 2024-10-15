import Mathlib

namespace NUMINAMATH_GPT_coefficient_ratio_is_4_l705_70567

noncomputable def coefficient_x3 := 
  let a := 60 -- Coefficient of x^3 in the expansion
  let b := Nat.choose 6 2 -- Binomial coefficient \binom{6}{2}
  a / b

theorem coefficient_ratio_is_4 : coefficient_x3 = 4 := by
  sorry

end NUMINAMATH_GPT_coefficient_ratio_is_4_l705_70567


namespace NUMINAMATH_GPT_cocktail_cost_per_litre_l705_70576

theorem cocktail_cost_per_litre :
  let mixed_fruit_cost := 262.85
  let acai_berry_cost := 3104.35
  let mixed_fruit_volume := 37
  let acai_berry_volume := 24.666666666666668
  let total_cost := mixed_fruit_volume * mixed_fruit_cost + acai_berry_volume * acai_berry_cost
  let total_volume := mixed_fruit_volume + acai_berry_volume
  total_cost / total_volume = 1400 :=
by
  sorry

end NUMINAMATH_GPT_cocktail_cost_per_litre_l705_70576


namespace NUMINAMATH_GPT_find_y_l705_70516

theorem find_y : ∃ y : ℕ, y > 0 ∧ (y + 3050) % 15 = 1234 % 15 ∧ y = 14 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l705_70516


namespace NUMINAMATH_GPT_rectangle_area_l705_70522

theorem rectangle_area (side_length width length : ℝ) (h_square_area : side_length^2 = 36)
  (h_width : width = side_length) (h_length : length = 2.5 * width) :
  width * length = 90 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l705_70522


namespace NUMINAMATH_GPT_bouquets_ratio_l705_70598

theorem bouquets_ratio (monday tuesday wednesday : ℕ) 
  (h1 : monday = 12) 
  (h2 : tuesday = 3 * monday) 
  (h3 : monday + tuesday + wednesday = 60) :
  wednesday / tuesday = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_bouquets_ratio_l705_70598


namespace NUMINAMATH_GPT_total_admission_cost_l705_70515

-- Define the standard admission cost
def standard_admission_cost : ℕ := 8

-- Define the discount if watched before 6 P.M.
def discount : ℕ := 3

-- Define the number of people in Kath's group
def number_of_people : ℕ := 1 + 2 + 3

-- Define the movie starting time (not directly used in calculation but part of the conditions)
def movie_start_time : ℕ := 16

-- Define the discounted admission cost
def discounted_admission_cost : ℕ := standard_admission_cost - discount

-- Prove that the total cost Kath will pay is $30
theorem total_admission_cost : number_of_people * discounted_admission_cost = 30 := by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_total_admission_cost_l705_70515


namespace NUMINAMATH_GPT_max_value_of_M_l705_70566

theorem max_value_of_M (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (2 * x + y) + y / (2 * y + z) + z / (2 * z + x)) ≤ 1 :=
sorry -- Proof placeholder

end NUMINAMATH_GPT_max_value_of_M_l705_70566


namespace NUMINAMATH_GPT_contest_end_time_l705_70559

def start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 765 -- duration of the contest in minutes

theorem contest_end_time : start_time + duration = 3 * 60 + 45 := by
  -- start_time is 15 * 60 (3:00 p.m. in minutes)
  -- duration is 765 minutes
  -- end_time should be 3:45 a.m. which is 3 * 60 + 45 minutes from midnight
  sorry

end NUMINAMATH_GPT_contest_end_time_l705_70559


namespace NUMINAMATH_GPT_rockham_soccer_league_members_count_l705_70583

def cost_per_pair_of_socks : Nat := 4
def additional_cost_per_tshirt : Nat := 5
def cost_per_tshirt : Nat := cost_per_pair_of_socks + additional_cost_per_tshirt

def pairs_of_socks_per_member : Nat := 2
def tshirts_per_member : Nat := 2

def total_cost_per_member : Nat :=
  pairs_of_socks_per_member * cost_per_pair_of_socks + tshirts_per_member * cost_per_tshirt

def total_cost_all_members : Nat := 2366
def total_members : Nat := total_cost_all_members / total_cost_per_member

theorem rockham_soccer_league_members_count : total_members = 91 :=
by
  -- Given steps in the solution, verify each condition and calculation.
  sorry

end NUMINAMATH_GPT_rockham_soccer_league_members_count_l705_70583


namespace NUMINAMATH_GPT_no_positive_integers_abc_l705_70552

theorem no_positive_integers_abc :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (a * b + b * c = a * c) ∧ (a * b * c = Nat.factorial 10) :=
sorry

end NUMINAMATH_GPT_no_positive_integers_abc_l705_70552


namespace NUMINAMATH_GPT_geometric_sequence_eleventh_term_l705_70505

theorem geometric_sequence_eleventh_term (a₁ : ℚ) (r : ℚ) (n : ℕ) (hₐ : a₁ = 5) (hᵣ : r = 2 / 3) (hₙ : n = 11) :
  (a₁ * r^(n - 1) = 5120 / 59049) :=
by
  -- conditions of the problem
  rw [hₐ, hᵣ, hₙ]
  sorry

end NUMINAMATH_GPT_geometric_sequence_eleventh_term_l705_70505


namespace NUMINAMATH_GPT_compute_expression_l705_70524

theorem compute_expression :
  (-9 * 5) - (-7 * -2) + (11 * -4) = -103 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l705_70524


namespace NUMINAMATH_GPT_absolute_value_half_l705_70519

theorem absolute_value_half (a : ℝ) (h : |a| = 1/2) : a = 1/2 ∨ a = -1/2 :=
sorry

end NUMINAMATH_GPT_absolute_value_half_l705_70519


namespace NUMINAMATH_GPT_sum_first_100_terms_is_l705_70569

open Nat

noncomputable def seq (a_n : ℕ → ℤ) : Prop :=
  a_n 2 = 2 ∧ ∀ n : ℕ, n > 0 → a_n (n + 2) + (-1)^(n + 1) * a_n n = 1 + (-1)^n

def sum_seq (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum f

theorem sum_first_100_terms_is :
  ∃ (a_n : ℕ → ℤ), seq a_n ∧ sum_seq a_n 100 = 2550 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_100_terms_is_l705_70569


namespace NUMINAMATH_GPT_slope_of_line_eq_slope_of_line_l705_70531

theorem slope_of_line_eq (x y : ℝ) (h : 4 * x + 6 * y = 24) : (6 * y = -4 * x + 24) → (y = - (2 : ℝ) / 3 * x + 4) :=
by
  intro h1
  sorry

theorem slope_of_line (x y m : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = - (2 : ℝ) / 3 * x + 4) : m = - (2 : ℝ) / 3 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_eq_slope_of_line_l705_70531


namespace NUMINAMATH_GPT_work_ratio_of_man_to_boy_l705_70507

theorem work_ratio_of_man_to_boy 
  (M B : ℝ) 
  (work : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = work)
  (h2 : (13 * M + 24 * B) * 4 = work) :
  M / B = 2 :=
by 
  sorry

end NUMINAMATH_GPT_work_ratio_of_man_to_boy_l705_70507


namespace NUMINAMATH_GPT_solve_inequalities_l705_70520

/-- Solve the inequality system and find all non-negative integer solutions. -/
theorem solve_inequalities :
  { x : ℤ | 0 ≤ x ∧ 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l705_70520


namespace NUMINAMATH_GPT_find_x_l705_70589
-- The first priority is to ensure the generated Lean code can be built successfully.

theorem find_x (x : ℤ) (h : 9823 + x = 13200) : x = 3377 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l705_70589


namespace NUMINAMATH_GPT_sector_area_max_sector_area_l705_70513

-- Definitions based on the given conditions
def perimeter : ℝ := 8
def central_angle (α : ℝ) : Prop := α = 2

-- Question 1: Find the area of the sector given the central angle is 2 rad
theorem sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) (h2 : l = 2 * r) : 
  (1/2) * r * l = 4 := 
by sorry

-- Question 2: Find the maximum area of the sector and the corresponding central angle
theorem max_sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) : 
  ∃ r, 0 < r ∧ r < 4 ∧ l = 8 - 2 * r ∧ 
  (1/2) * r * l = 4 ∧ l = 2 * r := 
by sorry

end NUMINAMATH_GPT_sector_area_max_sector_area_l705_70513


namespace NUMINAMATH_GPT_least_months_exceed_tripled_borrowed_l705_70500

theorem least_months_exceed_tripled_borrowed :
  ∃ t : ℕ, (1.03 : ℝ)^t > 3 ∧ ∀ n < t, (1.03 : ℝ)^n ≤ 3 :=
sorry

end NUMINAMATH_GPT_least_months_exceed_tripled_borrowed_l705_70500


namespace NUMINAMATH_GPT_solution_set_of_abs_inequality_l705_70502

theorem solution_set_of_abs_inequality (x : ℝ) : 
  (x < 5 ↔ |x - 8| - |x - 4| > 2) :=
sorry

end NUMINAMATH_GPT_solution_set_of_abs_inequality_l705_70502


namespace NUMINAMATH_GPT_sqrt_inequality_sum_inverse_ge_9_l705_70541

-- (1) Prove that \(\sqrt{3} + \sqrt{8} < 2 + \sqrt{7}\)
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := sorry

-- (2) Prove that given \(a > 0, b > 0, c > 0\) and \(a + b + c = 1\), \(\frac{1}{a} + \frac{1}{b} + \frac{1}{c} \geq 9\)
theorem sum_inverse_ge_9 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) : 
    1 / a + 1 / b + 1 / c ≥ 9 := sorry

end NUMINAMATH_GPT_sqrt_inequality_sum_inverse_ge_9_l705_70541


namespace NUMINAMATH_GPT_sale_in_first_month_is_5000_l705_70534

def sales : List ℕ := [6524, 5689, 7230, 6000, 12557]
def avg_sales : ℕ := 7000
def total_months : ℕ := 6

theorem sale_in_first_month_is_5000 :
  (avg_sales * total_months) - sales.sum = 5000 :=
by sorry

end NUMINAMATH_GPT_sale_in_first_month_is_5000_l705_70534


namespace NUMINAMATH_GPT_toms_total_score_l705_70582

def points_per_enemy : ℕ := 10
def enemies_killed : ℕ := 175

def base_score (enemies : ℕ) : ℝ := enemies * points_per_enemy

def bonus_percentage (enemies : ℕ) : ℝ :=
  if 100 ≤ enemies ∧ enemies < 150 then 0.50
  else if 150 ≤ enemies ∧ enemies < 200 then 0.75
  else if enemies ≥ 200 then 1.00
  else 0.0

def total_score (enemies : ℕ) : ℝ :=
  let base := base_score enemies
  let bonus := base * bonus_percentage enemies
  base + bonus

theorem toms_total_score :
  total_score enemies_killed = 3063 :=
by
  -- The proof will show the computed total score
  -- matches the expected value
  sorry

end NUMINAMATH_GPT_toms_total_score_l705_70582


namespace NUMINAMATH_GPT_simplest_common_denominator_fraction_exist_l705_70563

variable (x y : ℝ)

theorem simplest_common_denominator_fraction_exist :
  let d1 := x + y
  let d2 := x - y
  let d3 := x^2 - y^2
  (d3 = d1 * d2) → 
    ∀ n, (n = d1 * d2) → 
      (∃ m, (d1 * m = n) ∧ (d2 * m = n) ∧ (d3 * m = n)) :=
by
  sorry

end NUMINAMATH_GPT_simplest_common_denominator_fraction_exist_l705_70563


namespace NUMINAMATH_GPT_fill_tank_time_l705_70590

/-- 
If pipe A fills a tank in 30 minutes, pipe B fills the same tank in 20 minutes, 
and pipe C empties it in 40 minutes, then the time it takes to fill the tank 
when all three pipes are working together is 120/7 minutes.
-/
theorem fill_tank_time 
  (rate_A : ℝ) (rate_B : ℝ) (rate_C : ℝ) (combined_rate : ℝ) (T : ℝ) :
  rate_A = 1/30 ∧ rate_B = 1/20 ∧ rate_C = -1/40 ∧ combined_rate = rate_A + rate_B + rate_C
  → T = 1 / combined_rate
  → T = 120 / 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fill_tank_time_l705_70590


namespace NUMINAMATH_GPT_sum_T_19_34_51_l705_70509

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2 : ℕ) else (n + 1) / 2

def T (n : ℕ) : ℤ :=
  2 + S n

theorem sum_T_19_34_51 : T 19 + T 34 + T 51 = 25 := 
by
  -- Add the steps here
  sorry

end NUMINAMATH_GPT_sum_T_19_34_51_l705_70509


namespace NUMINAMATH_GPT_speed_of_current_l705_70587

theorem speed_of_current (m c : ℝ) (h1 : m + c = 20) (h2 : m - c = 18) : c = 1 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_current_l705_70587


namespace NUMINAMATH_GPT_moores_law_l705_70588

theorem moores_law (initial_transistors : ℕ) (doubling_period : ℕ) (t1 t2 : ℕ) 
  (initial_year : t1 = 1985) (final_year : t2 = 2010) (transistors_in_1985 : initial_transistors = 300000) 
  (doubles_every_two_years : doubling_period = 2) : 
  (initial_transistors * 2 ^ ((t2 - t1) / doubling_period) = 1228800000) := 
by
  sorry

end NUMINAMATH_GPT_moores_law_l705_70588


namespace NUMINAMATH_GPT_sum_gcd_lcm_60_429_l705_70535

theorem sum_gcd_lcm_60_429 : 
  let a := 60
  let b := 429
  gcd a b + lcm a b = 8583 :=
by
  -- Definitions of a and b
  let a := 60
  let b := 429
  
  -- The GCD and LCM calculations would go here
  
  -- Proof body (skipped with 'sorry')
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_60_429_l705_70535


namespace NUMINAMATH_GPT_no_such_divisor_l705_70511

theorem no_such_divisor (n : ℕ) : 
  (n ∣ (823435 : ℕ)^15) ∧ (n^5 - n^n = 1) → false := 
by sorry

end NUMINAMATH_GPT_no_such_divisor_l705_70511


namespace NUMINAMATH_GPT_form_of_reasoning_is_incorrect_l705_70578

-- Definitions from the conditions
def some_rational_numbers_are_fractions : Prop := 
  ∃ q : ℚ, ∃ f : ℚ, q = f / 1

def integers_are_rational_numbers : Prop :=
  ∀ z : ℤ, ∃ q : ℚ, q = z

-- The proposition to be proved
theorem form_of_reasoning_is_incorrect (h1 : some_rational_numbers_are_fractions) (h2 : integers_are_rational_numbers) : 
  ¬ ∀ z : ℤ, ∃ f : ℚ, f = z  := sorry

end NUMINAMATH_GPT_form_of_reasoning_is_incorrect_l705_70578


namespace NUMINAMATH_GPT_xyz_positive_and_distinct_l705_70508

theorem xyz_positive_and_distinct (a b x y z : ℝ)
  (h₁ : x + y + z = a)
  (h₂ : x^2 + y^2 + z^2 = b^2)
  (h₃ : x * y = z^2)
  (ha_pos : a > 0)
  (hb_condition : b^2 < a^2 ∧ a^2 < 3*b^2) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by
  sorry

end NUMINAMATH_GPT_xyz_positive_and_distinct_l705_70508


namespace NUMINAMATH_GPT_cos_two_sum_l705_70551

theorem cos_two_sum {α β : ℝ} 
  (h1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h2 : 3 * (Real.sin α + Real.cos α) ^ 2 - 2 * (Real.sin β + Real.cos β) ^ 2 = 1) :
  Real.cos (2 * (α + β)) = -1 / 3 :=
sorry

end NUMINAMATH_GPT_cos_two_sum_l705_70551


namespace NUMINAMATH_GPT_grace_crayon_selection_l705_70554

def crayons := {i // 1 ≤ i ∧ i ≤ 15}
def red_crayons := {i // 1 ≤ i ∧ i ≤ 3}

def total_ways := Nat.choose 15 5
def non_favorable := Nat.choose 12 5

theorem grace_crayon_selection : total_ways - non_favorable = 2211 :=
by
  sorry

end NUMINAMATH_GPT_grace_crayon_selection_l705_70554


namespace NUMINAMATH_GPT_marley_total_fruits_l705_70538

theorem marley_total_fruits (louis_oranges : ℕ) (louis_apples : ℕ) 
                            (samantha_oranges : ℕ) (samantha_apples : ℕ)
                            (marley_oranges : ℕ) (marley_apples : ℕ) : 
  (louis_oranges = 5) → (louis_apples = 3) → 
  (samantha_oranges = 8) → (samantha_apples = 7) → 
  (marley_oranges = 2 * louis_oranges) → (marley_apples = 3 * samantha_apples) → 
  (marley_oranges + marley_apples = 31) :=
by
  intros
  sorry

end NUMINAMATH_GPT_marley_total_fruits_l705_70538


namespace NUMINAMATH_GPT_cos_identity_15_30_degrees_l705_70514

theorem cos_identity_15_30_degrees (a b : ℝ) (h : b = 2 * a^2 - 1) : 2 * a^2 - b = 1 :=
by
  sorry

end NUMINAMATH_GPT_cos_identity_15_30_degrees_l705_70514


namespace NUMINAMATH_GPT_alice_no_guarantee_win_when_N_is_18_l705_70532

noncomputable def alice_cannot_guarantee_win : Prop :=
  ∀ (B : ℝ × ℝ) (P : ℕ → ℝ × ℝ),
    (∀ k, 0 ≤ k → k ≤ 18 → 
         dist (P (k + 1)) B < dist (P k) B ∨ dist (P (k + 1)) B ≥ dist (P k) B) →
    ∀ A : ℝ × ℝ, dist A B > 1 / 2020

theorem alice_no_guarantee_win_when_N_is_18 : alice_cannot_guarantee_win :=
sorry

end NUMINAMATH_GPT_alice_no_guarantee_win_when_N_is_18_l705_70532


namespace NUMINAMATH_GPT_max_value_of_abc_expression_l705_70555

noncomputable def max_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) : ℝ :=
  a^3 * b^2 * c^2

theorem max_value_of_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  max_abc_expression a b c h1 h2 h3 h4 ≤ 432 / 7^7 :=
sorry

end NUMINAMATH_GPT_max_value_of_abc_expression_l705_70555


namespace NUMINAMATH_GPT_jenny_grade_l705_70599

theorem jenny_grade (J A B : ℤ) 
  (hA : A = J - 25) 
  (hB : B = A / 2) 
  (hB_val : B = 35) : 
  J = 95 :=
by
  sorry

end NUMINAMATH_GPT_jenny_grade_l705_70599


namespace NUMINAMATH_GPT_petrol_expense_l705_70528

theorem petrol_expense 
  (rent milk groceries education misc savings petrol total_salary : ℝ)
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : education = 2500)
  (H5 : misc = 6100)
  (H6 : savings = 2400)
  (H7 : total_salary = savings / 0.10)
  (H8 : total_salary = rent + milk + groceries + education + misc + petrol + savings) :
  petrol = 2000 :=
by
  sorry

end NUMINAMATH_GPT_petrol_expense_l705_70528


namespace NUMINAMATH_GPT_tangent_line_equation_l705_70550

open Real

noncomputable def circle_center : ℝ × ℝ := (2, 1)
noncomputable def tangent_point : ℝ × ℝ := (4, 3)

def circle_equation (x y : ℝ) : Prop := (x - circle_center.1)^2 + (y - circle_center.2)^2 = 1

theorem tangent_line_equation :
  ∀ (x y : ℝ), ( (x = 4 ∧ y = 3) ∨ circle_equation x y ) → 2 * x + 2 * y - 7 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_l705_70550


namespace NUMINAMATH_GPT_girl_walking_speed_l705_70571

-- Definitions of the conditions
def distance := 30 -- in kilometers
def time := 6 -- in hours

-- Definition of the walking speed function
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- The theorem we want to prove
theorem girl_walking_speed : speed distance time = 5 := by
  sorry

end NUMINAMATH_GPT_girl_walking_speed_l705_70571


namespace NUMINAMATH_GPT_find_double_pieces_l705_70593

theorem find_double_pieces (x : ℕ) 
  (h1 : 100 + 2 * x + 150 + 660 = 1000) : x = 45 :=
by sorry

end NUMINAMATH_GPT_find_double_pieces_l705_70593


namespace NUMINAMATH_GPT_correct_option_is_B_l705_70501

-- Define the conditions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (m : ℝ) : Prop := (-2 * m^2)^3 = -8 * m^6
def optionC (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2
def optionD (a b : ℝ) : Prop := 2 * a * b + 3 * a^2 * b = 5 * a^3 * b^2

-- The proof problem: which option is correct
theorem correct_option_is_B (m : ℝ) : optionB m := by
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l705_70501


namespace NUMINAMATH_GPT_circle_radius_five_d_value_l705_70575

theorem circle_radius_five_d_value :
  ∀ (d : ℝ), (∃ (x y : ℝ), (x - 4)^2 + (y + 5)^2 = 41 - d) → d = 16 :=
by
  intros d h
  sorry

end NUMINAMATH_GPT_circle_radius_five_d_value_l705_70575


namespace NUMINAMATH_GPT_decreased_cost_proof_l705_70529

def original_cost : ℝ := 200
def percentage_decrease : ℝ := 0.5
def decreased_cost (original_cost : ℝ) (percentage_decrease : ℝ) : ℝ := 
  original_cost - (percentage_decrease * original_cost)

theorem decreased_cost_proof : decreased_cost original_cost percentage_decrease = 100 := 
by { 
  sorry -- Proof is not required
}

end NUMINAMATH_GPT_decreased_cost_proof_l705_70529


namespace NUMINAMATH_GPT_carnations_count_l705_70560

theorem carnations_count (total_flowers : ℕ) (fract_rose : ℚ) (num_tulips : ℕ) (h1 : total_flowers = 40) (h2 : fract_rose = 2 / 5) (h3 : num_tulips = 10) :
  total_flowers - ((fract_rose * total_flowers) + num_tulips) = 14 := 
by
  sorry

end NUMINAMATH_GPT_carnations_count_l705_70560


namespace NUMINAMATH_GPT_tromino_covering_l705_70540

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def chessboard_black_squares (n : ℕ) : ℕ := (n^2 + 1) / 2

def minimum_trominos (n : ℕ) : ℕ := (n^2 + 1) / 6

theorem tromino_covering (n : ℕ) (h_odd : is_odd n) (h_ge7 : n ≥ 7) :
  ∃ k : ℕ, chessboard_black_squares n = 3 * k ∧ (k = minimum_trominos n) :=
sorry

end NUMINAMATH_GPT_tromino_covering_l705_70540


namespace NUMINAMATH_GPT_parabola_and_line_sum_l705_70510

theorem parabola_and_line_sum (A B F : ℝ × ℝ)
  (h_parabola : ∀ x y : ℝ, (y^2 = 4 * x) ↔ (x, y) = A ∨ (x, y) = B)
  (h_line : ∀ x y : ℝ, (2 * x + y - 4 = 0) ↔ (x, y) = A ∨ (x, y) = B)
  (h_focus : F = (1, 0))
  : |F - A| + |F - B| = 7 := 
sorry

end NUMINAMATH_GPT_parabola_and_line_sum_l705_70510


namespace NUMINAMATH_GPT_change_received_l705_70584

variable (a : ℕ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a := 
by 
  sorry

end NUMINAMATH_GPT_change_received_l705_70584


namespace NUMINAMATH_GPT_sum_of_inscribed_sphere_volumes_l705_70523

theorem sum_of_inscribed_sphere_volumes :
  let height := 3
  let angle := Real.pi / 3
  let r₁ := height / 3 -- Radius of the first inscribed sphere
  let geometric_ratio := 1 / 3
  let volume (r : ℝ) := (4 / 3) * Real.pi * r^3
  let volumes : ℕ → ℝ := λ n => volume (r₁ * geometric_ratio^(n - 1))
  let total_volume := ∑' n, volumes n
  total_volume = (18 * Real.pi) / 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_inscribed_sphere_volumes_l705_70523


namespace NUMINAMATH_GPT_total_points_is_400_l705_70542

-- Define the conditions as definitions in Lean 4 
def pointsPerEnemy : ℕ := 15
def bonusPoints : ℕ := 50
def totalEnemies : ℕ := 25
def enemiesLeftUndestroyed : ℕ := 5
def bonusesEarned : ℕ := 2

-- Calculate the total number of enemies defeated
def enemiesDefeated : ℕ := totalEnemies - enemiesLeftUndestroyed

-- Calculate the points from defeating enemies
def pointsFromEnemies := enemiesDefeated * pointsPerEnemy

-- Calculate the total bonus points
def totalBonusPoints := bonusesEarned * bonusPoints

-- The total points earned is the sum of points from enemies and bonus points
def totalPointsEarned := pointsFromEnemies + totalBonusPoints

-- Prove that the total points earned is equal to 400
theorem total_points_is_400 : totalPointsEarned = 400 := by
    sorry

end NUMINAMATH_GPT_total_points_is_400_l705_70542


namespace NUMINAMATH_GPT_pow_div_pow_eq_result_l705_70597

theorem pow_div_pow_eq_result : 13^8 / 13^5 = 2197 := by
  sorry

end NUMINAMATH_GPT_pow_div_pow_eq_result_l705_70597


namespace NUMINAMATH_GPT_arctan_sum_in_right_triangle_l705_70553

theorem arctan_sum_in_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : 
  (Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_arctan_sum_in_right_triangle_l705_70553


namespace NUMINAMATH_GPT_sedrach_divides_each_pie_l705_70565

theorem sedrach_divides_each_pie (P : ℕ) :
  (13 * P * 5 = 130) → P = 2 :=
by
  sorry

end NUMINAMATH_GPT_sedrach_divides_each_pie_l705_70565


namespace NUMINAMATH_GPT_geometric_sequence_S8_l705_70586

theorem geometric_sequence_S8 (S : ℕ → ℝ) (hs2 : S 2 = 4) (hs4 : S 4 = 16) : 
  S 8 = 160 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_S8_l705_70586


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l705_70521

theorem inverse_proportion_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∃ x y : ℝ, x = -2 ∧ y = 3 ∧ y = k / x) →
  (∀ x : ℝ, (x < 0 → k / x > 0) ∧ (x > 0 → k / x < 0)) :=
sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l705_70521


namespace NUMINAMATH_GPT_monotonic_iff_m_ge_one_third_l705_70504

-- Define the function f(x) = x^3 + x^2 + mx + 1
def f (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

-- Define the derivative of the function f w.r.t x
def f' (x m : ℝ) : ℝ := 3 * x^2 + 2 * x + m

-- State the main theorem: f is monotonic on ℝ if and only if m ≥ 1/3
theorem monotonic_iff_m_ge_one_third (m : ℝ) :
  (∀ x y : ℝ, x < y → f x m ≤ f y m) ↔ (m ≥ 1 / 3) :=
sorry

end NUMINAMATH_GPT_monotonic_iff_m_ge_one_third_l705_70504


namespace NUMINAMATH_GPT_checker_arrangements_five_digit_palindromes_l705_70570

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem checker_arrangements :
  comb 32 12 * comb 20 12 = Nat.choose 32 12 * Nat.choose 20 12 := by
  sorry

theorem five_digit_palindromes :
  9 * 10 * 10 = 900 := by
  sorry

end NUMINAMATH_GPT_checker_arrangements_five_digit_palindromes_l705_70570


namespace NUMINAMATH_GPT_possible_measures_of_angle_X_l705_70506

theorem possible_measures_of_angle_X :
  ∃ (n : ℕ), n = 17 ∧ ∀ (X Y : ℕ), 
    (X > 0) → 
    (Y > 0) → 
    (∃ k : ℕ, k ≥ 1 ∧ X = k * Y) → 
    X + Y = 180 → 
    ∃ d : ℕ, d ∈ {d | d ∣ 180 } ∧ d ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_possible_measures_of_angle_X_l705_70506


namespace NUMINAMATH_GPT_monthly_income_P_l705_70580

theorem monthly_income_P (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : (P + R) / 2 = 5200) :
  P = 4000 := 
sorry

end NUMINAMATH_GPT_monthly_income_P_l705_70580


namespace NUMINAMATH_GPT_least_number_of_stamps_l705_70574

theorem least_number_of_stamps : ∃ c f : ℕ, 3 * c + 4 * f = 50 ∧ c + f = 13 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_stamps_l705_70574


namespace NUMINAMATH_GPT_vectors_opposite_direction_l705_70517

noncomputable def a : ℝ × ℝ := (-2, 4)
noncomputable def b : ℝ × ℝ := (1, -2)

theorem vectors_opposite_direction : a = (-2 : ℝ) • b :=
by
  sorry

end NUMINAMATH_GPT_vectors_opposite_direction_l705_70517


namespace NUMINAMATH_GPT_least_number_to_subtract_l705_70503

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h : n = 427398) (k : d = 13) (r_val : r = 2) : 
  ∃ x : ℕ, (n - x) % d = 0 ∧ r = x :=
by sorry

end NUMINAMATH_GPT_least_number_to_subtract_l705_70503


namespace NUMINAMATH_GPT_ratio_five_to_one_l705_70527

theorem ratio_five_to_one (x : ℕ) : (5 : ℕ) * 13 = 1 * x → x = 65 := 
by 
  intro h
  linarith

end NUMINAMATH_GPT_ratio_five_to_one_l705_70527


namespace NUMINAMATH_GPT_value_of_a2_l705_70518

variable {R : Type*} [Ring R] (x a_0 a_1 a_2 a_3 : R)

theorem value_of_a2 
  (h : ∀ x : R, x^3 = a_0 + a_1 * (x - 2) + a_2 * (x - 2)^2 + a_3 * (x - 2)^3) :
  a_2 = 6 :=
sorry

end NUMINAMATH_GPT_value_of_a2_l705_70518


namespace NUMINAMATH_GPT_range_of_p_l705_70579

def sequence_sum (n : ℕ) : ℚ := (-1) ^ (n + 1) * (1 / 2 ^ n)

def a_n (n : ℕ) : ℚ :=
  if h : n = 0 then sequence_sum 1 else
  sequence_sum n - sequence_sum (n - 1)

theorem range_of_p (p : ℚ) : 
  (∃ n : ℕ, 0 < n ∧ (p - a_n n) * (p - a_n (n + 1)) < 0) ↔ 
  - 3 / 4 < p ∧ p < 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_p_l705_70579


namespace NUMINAMATH_GPT_proportion_exists_x_l705_70537

theorem proportion_exists_x : ∃ x : ℕ, 1 * x = 3 * 4 :=
by
  sorry

end NUMINAMATH_GPT_proportion_exists_x_l705_70537


namespace NUMINAMATH_GPT_solve_for_x_l705_70533

theorem solve_for_x (x : ℝ) (h1 : 3 * x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l705_70533


namespace NUMINAMATH_GPT_number_of_cars_l705_70577

theorem number_of_cars 
  (num_bikes : ℕ) (num_wheels_total : ℕ) (wheels_per_bike : ℕ) (wheels_per_car : ℕ)
  (h1 : num_bikes = 10) (h2 : num_wheels_total = 76) (h3 : wheels_per_bike = 2) (h4 : wheels_per_car = 4) :
  ∃ (C : ℕ), C = 14 := 
by
  sorry

end NUMINAMATH_GPT_number_of_cars_l705_70577


namespace NUMINAMATH_GPT_jane_played_8_rounds_l705_70539

variable (points_per_round : ℕ)
variable (end_points : ℕ)
variable (lost_points : ℕ)
variable (total_points : ℕ)
variable (rounds_played : ℕ)

theorem jane_played_8_rounds 
  (h1 : points_per_round = 10) 
  (h2 : end_points = 60) 
  (h3 : lost_points = 20)
  (h4 : total_points = end_points + lost_points)
  (h5 : total_points = points_per_round * rounds_played) : 
  rounds_played = 8 := 
by 
  sorry

end NUMINAMATH_GPT_jane_played_8_rounds_l705_70539


namespace NUMINAMATH_GPT_total_geese_l705_70530

/-- Definition of the number of geese that remain flying after each lake, 
    based on the given conditions. -/
def geese_after_lake (G : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then G else 2^(n : ℕ) - 1

/-- Main theorem stating the total number of geese in the flock. -/
theorem total_geese (n : ℕ) : ∃ (G : ℕ), geese_after_lake G n = 2^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_total_geese_l705_70530


namespace NUMINAMATH_GPT_simplify_expression_l705_70585

theorem simplify_expression : (2^4 * 2^4 * 2^4) = 2^12 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l705_70585


namespace NUMINAMATH_GPT_greatest_difference_l705_70512

-- Definitions: Number of marbles in each basket
def basketA_red : Nat := 4
def basketA_yellow : Nat := 2
def basketB_green : Nat := 6
def basketB_yellow : Nat := 1
def basketC_white : Nat := 3
def basketC_yellow : Nat := 9

-- Define the differences
def diff_basketA : Nat := basketA_red - basketA_yellow
def diff_basketB : Nat := basketB_green - basketB_yellow
def diff_basketC : Nat := basketC_yellow - basketC_white

-- The goal is to prove that 6 is the greatest difference
theorem greatest_difference : max (max diff_basketA diff_basketB) diff_basketC = 6 :=
by 
  -- The proof is not provided
  sorry

end NUMINAMATH_GPT_greatest_difference_l705_70512


namespace NUMINAMATH_GPT_tan_5pi_over_4_l705_70547

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_5pi_over_4_l705_70547


namespace NUMINAMATH_GPT_cannot_be_correct_average_l705_70543

theorem cannot_be_correct_average (a : ℝ) (h_pos : a > 0) (h_median : a ≤ 12) : 
  ∀ avg, avg = (12 + a + 8 + 15 + 23) / 5 → avg ≠ 71 / 5 := 
by
  intro avg h_avg
  sorry

end NUMINAMATH_GPT_cannot_be_correct_average_l705_70543


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l705_70568

-- Given conditions
def repeating_decimal : ℚ := 7 + 832 / 999

-- Goal: Prove that the repeating decimal 7.\overline{832} equals 70/9
theorem repeating_decimal_as_fraction : repeating_decimal = 70 / 9 := by
  unfold repeating_decimal
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l705_70568


namespace NUMINAMATH_GPT_train_crosses_pole_in_12_seconds_l705_70573

noncomputable def time_to_cross_pole (speed train_length : ℕ) : ℕ := 
  train_length / speed

theorem train_crosses_pole_in_12_seconds 
  (speed : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) (train_crossing_time : ℕ)
  (h_speed : speed = 10) 
  (h_platform_length : platform_length = 320) 
  (h_time_to_cross_platform : time_to_cross_platform = 44) 
  (h_train_crossing_time : train_crossing_time = 12) :
  time_to_cross_pole speed 120 = train_crossing_time := 
by 
  sorry

end NUMINAMATH_GPT_train_crosses_pole_in_12_seconds_l705_70573


namespace NUMINAMATH_GPT_total_kids_played_l705_70562

theorem total_kids_played (kids_monday : ℕ) (kids_tuesday : ℕ) (h_monday : kids_monday = 4) (h_tuesday : kids_tuesday = 14) : 
  kids_monday + kids_tuesday = 18 := 
by
  -- proof steps here (for now, use sorry to skip the proof)
  sorry

end NUMINAMATH_GPT_total_kids_played_l705_70562


namespace NUMINAMATH_GPT_sum_of_nonneg_real_numbers_inequality_l705_70548

open BigOperators

variables {α : Type*} [LinearOrderedField α]

theorem sum_of_nonneg_real_numbers_inequality 
  (a : ℕ → α) (n : ℕ)
  (h_nonneg : ∀ i : ℕ, 0 ≤ a i) : 
  (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j) * (∑ j in Finset.Icc i (n - 1), a j ^ 2))) 
  ≤ (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j)) ^ 2) :=
sorry

end NUMINAMATH_GPT_sum_of_nonneg_real_numbers_inequality_l705_70548


namespace NUMINAMATH_GPT_solve_for_x_l705_70581

theorem solve_for_x (x y : ℝ) (h₁ : y = (x^2 - 9) / (x - 3)) (h₂ : y = 3 * x - 4) : x = 7 / 2 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l705_70581


namespace NUMINAMATH_GPT_symmetric_circle_equation_l705_70572

-- Define original circle equation
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Define symmetric circle equation
def symmetric_circle (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

theorem symmetric_circle_equation (x y : ℝ) : 
  symmetric_circle x y ↔ original_circle (-x) y :=
by sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l705_70572


namespace NUMINAMATH_GPT_find_b_l705_70594

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (527816429 - b) % 17 = 0 ∧ b = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_l705_70594


namespace NUMINAMATH_GPT_second_smallest_packs_of_hot_dogs_l705_70592

theorem second_smallest_packs_of_hot_dogs
    (n : ℤ) 
    (h1 : ∃ m : ℤ, 12 * n = 8 * m + 6) :
    ∃ k : ℤ, n = 4 * k + 7 :=
sorry

end NUMINAMATH_GPT_second_smallest_packs_of_hot_dogs_l705_70592


namespace NUMINAMATH_GPT_max_radius_of_circle_l705_70596

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_max_radius_of_circle_l705_70596


namespace NUMINAMATH_GPT_expression_at_x_equals_2_l705_70525

theorem expression_at_x_equals_2 (a b : ℝ) (h : 2 * a - b = -1) : (2 * b - 4 * a) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_at_x_equals_2_l705_70525


namespace NUMINAMATH_GPT_rebus_system_solution_l705_70544

theorem rebus_system_solution :
  ∃ (M A H P h : ℕ), 
  (M > 0) ∧ (P > 0) ∧ 
  (M ≠ A) ∧ (M ≠ H) ∧ (M ≠ P) ∧ (M ≠ h) ∧
  (A ≠ H) ∧ (A ≠ P) ∧ (A ≠ h) ∧ 
  (H ≠ P) ∧ (H ≠ h) ∧ (P ≠ h) ∧
  ((M * 10 + A) * (M * 10 + A) = M * 100 + H * 10 + P) ∧ 
  ((A * 10 + M) * (A * 10 + M) = P * 100 + h * 10 + M) ∧ 
  (((M = 1) ∧ (A = 3) ∧ (H = 6) ∧ (P = 9) ∧ (h = 6)) ∨
   ((M = 3) ∧ (A = 1) ∧ (H = 9) ∧ (P = 6) ∧ (h = 9))) :=
by
  sorry

end NUMINAMATH_GPT_rebus_system_solution_l705_70544


namespace NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l705_70536

theorem ratio_of_spinsters_to_cats (S C : ℕ) (hS : S = 12) (hC : C = S + 42) : S / gcd S C = 2 ∧ C / gcd S C = 9 :=
by
  -- skip proof (use sorry)
  sorry

end NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l705_70536


namespace NUMINAMATH_GPT_total_material_ordered_l705_70595

theorem total_material_ordered (c b s : ℝ) (hc : c = 0.17) (hb : b = 0.17) (hs : s = 0.5) :
  c + b + s = 0.84 :=
by sorry

end NUMINAMATH_GPT_total_material_ordered_l705_70595


namespace NUMINAMATH_GPT_trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l705_70545

theorem trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13 :
  (Real.cos (58 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) +
   Real.sin (58 * Real.pi / 180) * Real.sin (13 * Real.pi / 180) =
   Real.cos (45 * Real.pi / 180)) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l705_70545


namespace NUMINAMATH_GPT_common_root_implies_remaining_roots_l705_70557

variables {R : Type*} [LinearOrderedField R]

theorem common_root_implies_remaining_roots
  (a b c x1 x2 x3 : R) 
  (h_non_zero_a : a ≠ 0)
  (h_non_zero_b : b ≠ 0)
  (h_non_zero_c : c ≠ 0)
  (h_a_ne_b : a ≠ b)
  (h_common_root1 : x1^2 + a*x1 + b*c = 0)
  (h_common_root2 : x1^2 + b*x1 + c*a = 0)
  (h_root2_eq : x2^2 + a*x2 + b*c = 0)
  (h_root3_eq : x3^2 + b*x3 + c*a = 0)
  : x2^2 + c*x2 + a*b = 0 ∧ x3^2 + c*x3 + a*b = 0 :=
sorry

end NUMINAMATH_GPT_common_root_implies_remaining_roots_l705_70557


namespace NUMINAMATH_GPT_f_2015_eq_neg_2014_l705_70549

variable {f : ℝ → ℝ}

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f1_value : f 1 = 2014 := sorry

-- Theorem to prove
theorem f_2015_eq_neg_2014 :
  isOddFunction f → isPeriodic f 3 → (f 1 = 2014) → f 2015 = -2014 :=
by
  intros hOdd hPeriodic hF1
  sorry

end NUMINAMATH_GPT_f_2015_eq_neg_2014_l705_70549


namespace NUMINAMATH_GPT_quadrilateral_trapezoid_or_parallelogram_l705_70591

theorem quadrilateral_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ)
  (hs : s1^2 = s2 * s4) :
  (exists (is_trapezoid : Prop), is_trapezoid) ∨ (exists (is_parallelogram : Prop), is_parallelogram) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_trapezoid_or_parallelogram_l705_70591


namespace NUMINAMATH_GPT_variance_transformation_example_l705_70556

def variance (X : List ℝ) : ℝ := sorry -- Assuming some definition of variance

theorem variance_transformation_example {n : ℕ} (X : List ℝ) (h_len : X.length = 2021) (h_var : variance X = 3) :
  variance (X.map (fun x => 3 * (x - 2))) = 27 := 
sorry

end NUMINAMATH_GPT_variance_transformation_example_l705_70556


namespace NUMINAMATH_GPT_smallest_n_divisible_l705_70564

open Nat

theorem smallest_n_divisible (n : ℕ) : (∃ (n : ℕ), n > 0 ∧ 45 ∣ n^2 ∧ 720 ∣ n^3) → n = 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_l705_70564


namespace NUMINAMATH_GPT_find_y_parallel_l705_70558

-- Definitions
def a : ℝ × ℝ := (2, 3)
def b (y : ℝ) : ℝ × ℝ := (4, -1 + y)

-- Parallel condition implies proportional components
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- The proof problem
theorem find_y_parallel : ∀ y : ℝ, parallel_vectors a (b y) → y = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_y_parallel_l705_70558


namespace NUMINAMATH_GPT_triangle_inequality_sqrt_sum_three_l705_70526

theorem triangle_inequality_sqrt_sum_three
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  3 ≤ (Real.sqrt (a / (-a + b + c)) + 
       Real.sqrt (b / (a - b + c)) + 
       Real.sqrt (c / (a + b - c))) := 
sorry

end NUMINAMATH_GPT_triangle_inequality_sqrt_sum_three_l705_70526


namespace NUMINAMATH_GPT_henry_age_l705_70561

theorem henry_age (H J : ℕ) 
  (sum_ages : H + J = 40) 
  (age_relation : H - 11 = 2 * (J - 11)) : 
  H = 23 := 
sorry

end NUMINAMATH_GPT_henry_age_l705_70561


namespace NUMINAMATH_GPT_find_n_from_equation_l705_70546

theorem find_n_from_equation (n m : ℕ) (h1 : (1^m / 5^m) * (1^n / 4^n) = 1 / (2 * 10^31)) (h2 : m = 31) : n = 16 := 
by
  sorry

end NUMINAMATH_GPT_find_n_from_equation_l705_70546
