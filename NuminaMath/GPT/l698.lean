import Mathlib

namespace a9_proof_l698_69871

variable {a : ℕ → ℝ}

-- Conditions
axiom a1 : a 1 = 1
axiom an_recurrence : ∀ n > 1, a n = (a (n - 1)) * 2^(n - 1)

-- Goal
theorem a9_proof : a 9 = 2^36 := 
by 
  sorry

end a9_proof_l698_69871


namespace third_pipe_empty_time_l698_69828

theorem third_pipe_empty_time (x : ℝ) :
  (1 / 60 : ℝ) + (1 / 120) - (1 / x) = (1 / 60) →
  x = 120 :=
by
  intros h
  sorry

end third_pipe_empty_time_l698_69828


namespace prove_composite_k_l698_69880

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := ∃ p q, p > 1 ∧ q > 1 ∧ n = p * q

def problem_statement (a b c d : ℕ) (h : a * b = c * d) : Prop :=
  is_composite (a^1984 + b^1984 + c^1984 + d^1984)

-- The theorem to prove
theorem prove_composite_k (a b c d : ℕ) (h : a * b = c * d) : 
  problem_statement a b c d h := sorry

end prove_composite_k_l698_69880


namespace sphere_surface_area_l698_69842

-- Define the conditions
def points_on_sphere (A B C : Type) := 
  ∃ (AB BC AC : Real), AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define the distance condition
def distance_condition (R : Real) := 
  ∃ (d : Real), d = R / 2

-- Define the main theorem
theorem sphere_surface_area 
  (A B C : Type) 
  (h_points : points_on_sphere A B C) 
  (h_distance : ∃ R : Real, distance_condition R) : 
  4 * Real.pi * (10 / 3 * Real.sqrt 3) ^ 2 = 400 / 3 * Real.pi := 
by 
  sorry

end sphere_surface_area_l698_69842


namespace min_value_expression_l698_69873

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (∃ c : ℝ, c = (1 / (2 * x) + x / (y + 1)) ∧ c = 5 / 4) :=
sorry

end min_value_expression_l698_69873


namespace volume_of_smaller_cube_l698_69875

noncomputable def volume_of_larger_cube : ℝ := 343
noncomputable def number_of_smaller_cubes : ℝ := 343
noncomputable def surface_area_difference : ℝ := 1764

theorem volume_of_smaller_cube (v_lc : ℝ) (n_sc : ℝ) (sa_diff : ℝ) :
  v_lc = volume_of_larger_cube →
  n_sc = number_of_smaller_cubes →
  sa_diff = surface_area_difference →
  ∃ (v_sc : ℝ), v_sc = 1 :=
by sorry

end volume_of_smaller_cube_l698_69875


namespace theater_cost_per_square_foot_l698_69866

theorem theater_cost_per_square_foot
    (n_seats : ℕ)
    (space_per_seat : ℕ)
    (cost_ratio : ℕ)
    (partner_coverage : ℕ)
    (tom_expense : ℕ)
    (total_seats := 500)
    (square_footage := total_seats * space_per_seat)
    (construction_cost := cost_ratio * land_cost)
    (total_cost := land_cost + construction_cost)
    (partner_expense := total_cost * partner_coverage / 100)
    (tom_expense_ratio := 100 - partner_coverage)
    (cost_equation := tom_expense = total_cost * tom_expense_ratio / 100)
    (land_cost := 30000) :
    tom_expense = 54000 → 
    space_per_seat = 12 → 
    cost_ratio = 2 →
    partner_coverage = 40 → 
    tom_expense_ratio = 60 → 
    total_cost = 90000 → 
    total_cost / 3 = land_cost →
    land_cost / square_footage = 5 :=
    sorry

end theater_cost_per_square_foot_l698_69866


namespace As_annual_income_l698_69894

theorem As_annual_income :
  let Cm := 14000
  let Bm := Cm + 0.12 * Cm
  let Am := (5 / 2) * Bm
  Am * 12 = 470400 := by
  sorry

end As_annual_income_l698_69894


namespace inequality_xy_yz_zx_l698_69888

theorem inequality_xy_yz_zx {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) <= 1 / 4 * (Real.sqrt 33 + 1) :=
sorry

end inequality_xy_yz_zx_l698_69888


namespace paula_bought_two_shirts_l698_69856

-- Define the conditions
def total_money : Int := 109
def shirt_cost : Int := 11
def pants_cost : Int := 13
def remaining_money : Int := 74

-- Calculate the expenditure on shirts and pants
def expenditure : Int := total_money - remaining_money

-- Define the number of shirts bought
def number_of_shirts (S : Int) : Prop := expenditure = shirt_cost * S + pants_cost

-- The theorem stating that Paula bought 2 shirts
theorem paula_bought_two_shirts : number_of_shirts 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end paula_bought_two_shirts_l698_69856


namespace hyperbola_equation_l698_69814

theorem hyperbola_equation :
  ∃ (b : ℝ), (∀ (x y : ℝ), ((x = 2) ∧ (y = 2)) →
    ((x^2 / 5) - (y^2 / b^2) = 1)) ∧
    (∀ x y, (y = (2 / Real.sqrt 5) * x) ∨ (y = -(2 / Real.sqrt 5) * x) → 
    (∀ (a b : ℝ), (a = 2) → (b = 2) →
      (b^2 = 4) → ((5 * y^2 / 4) - x^2 = 1))) :=
sorry

end hyperbola_equation_l698_69814


namespace candy_distribution_l698_69850

theorem candy_distribution (n k : ℕ) (h1 : 3 < n) (h2 : n < 15) (h3 : 195 - n * k = 8) : k = 17 :=
  by
    sorry

end candy_distribution_l698_69850


namespace systematic_sample_contains_18_l698_69887

theorem systematic_sample_contains_18 (employees : Finset ℕ) (sample : Finset ℕ)
    (h1 : employees = Finset.range 52)
    (h2 : sample.card = 4)
    (h3 : ∀ n ∈ sample, n ∈ employees)
    (h4 : 5 ∈ sample)
    (h5 : 31 ∈ sample)
    (h6 : 44 ∈ sample) :
  18 ∈ sample :=
sorry

end systematic_sample_contains_18_l698_69887


namespace value_of_M_l698_69822

-- Define M as given in the conditions
def M : ℤ :=
  (150^2 + 2) + (149^2 - 2) - (148^2 + 2) - (147^2 - 2) + (146^2 + 2) +
  (145^2 - 2) - (144^2 + 2) - (143^2 - 2) + (142^2 + 2) + (141^2 - 2) -
  (140^2 + 2) - (139^2 - 2) + (138^2 + 2) + (137^2 - 2) - (136^2 + 2) -
  (135^2 - 2) + (134^2 + 2) + (133^2 - 2) - (132^2 + 2) - (131^2 - 2) +
  (130^2 + 2) + (129^2 - 2) - (128^2 + 2) - (127^2 - 2) + (126^2 + 2) +
  (125^2 - 2) - (124^2 + 2) - (123^2 - 2) + (122^2 + 2) + (121^2 - 2) -
  (120^2 + 2) - (119^2 - 2) + (118^2 + 2) + (117^2 - 2) - (116^2 + 2) -
  (115^2 - 2) + (114^2 + 2) + (113^2 - 2) - (112^2 + 2) - (111^2 - 2) +
  (110^2 + 2) + (109^2 - 2) - (108^2 + 2) - (107^2 - 2) + (106^2 + 2) +
  (105^2 - 2) - (104^2 + 2) - (103^2 - 2) + (102^2 + 2) + (101^2 - 2) -
  (100^2 + 2) - (99^2 - 2) + (98^2 + 2) + (97^2 - 2) - (96^2 + 2) -
  (95^2 - 2) + (94^2 + 2) + (93^2 - 2) - (92^2 + 2) - (91^2 - 2) +
  (90^2 + 2) + (89^2 - 2) - (88^2 + 2) - (87^2 - 2) + (86^2 + 2) +
  (85^2 - 2) - (84^2 + 2) - (83^2 - 2) + (82^2 + 2) + (81^2 - 2) -
  (80^2 + 2) - (79^2 - 2) + (78^2 + 2) + (77^2 - 2) - (76^2 + 2) -
  (75^2 - 2) + (74^2 + 2) + (73^2 - 2) - (72^2 + 2) - (71^2 - 2) +
  (70^2 + 2) + (69^2 - 2) - (68^2 + 2) - (67^2 - 2) + (66^2 + 2) +
  (65^2 - 2) - (64^2 + 2) - (63^2 - 2) + (62^2 + 2) + (61^2 - 2) -
  (60^2 + 2) - (59^2 - 2) + (58^2 + 2) + (57^2 - 2) - (56^2 + 2) -
  (55^2 - 2) + (54^2 + 2) + (53^2 - 2) - (52^2 + 2) - (51^2 - 2) +
  (50^2 + 2) + (49^2 - 2) - (48^2 + 2) - (47^2 - 2) + (46^2 + 2) +
  (45^2 - 2) - (44^2 + 2) - (43^2 - 2) + (42^2 + 2) + (41^2 - 2) -
  (40^2 + 2) - (39^2 - 2) + (38^2 + 2) + (37^2 - 2) - (36^2 + 2) -
  (35^2 - 2) + (34^2 + 2) + (33^2 - 2) - (32^2 + 2) - (31^2 - 2) +
  (30^2 + 2) + (29^2 - 2) - (28^2 + 2) - (27^2 - 2) + (26^2 + 2) +
  (25^2 - 2) - (24^2 + 2) - (23^2 - 2) + (22^2 + 2) + (21^2 - 2) -
  (20^2 + 2) - (19^2 - 2) + (18^2 + 2) + (17^2 - 2) - (16^2 + 2) -
  (15^2 - 2) + (14^2 + 2) + (13^2 - 2) - (12^2 + 2) - (11^2 - 2) +
  (10^2 + 2) + (9^2 - 2) - (8^2 + 2) - (7^2 - 2) + (6^2 + 2) +
  (5^2 - 2) - (4^2 + 2) - (3^2 - 2) + (2^2 + 2) + (1^2 - 2)

-- Statement to prove that the value of M is 22700
theorem value_of_M : M = 22700 :=
  by sorry

end value_of_M_l698_69822


namespace polynomial_q_correct_l698_69879

noncomputable def polynomial_q (x : ℝ) : ℝ :=
  -x^6 + 12*x^5 + 9*x^4 + 14*x^3 - 5*x^2 + 17*x + 1

noncomputable def polynomial_rhs (x : ℝ) : ℝ :=
  x^6 + 12*x^5 + 13*x^4 + 14*x^3 + 17*x + 3

noncomputable def polynomial_2 (x : ℝ) : ℝ :=
  2*x^6 + 4*x^4 + 5*x^2 + 2

theorem polynomial_q_correct (x : ℝ) : 
  polynomial_q x = polynomial_rhs x - polynomial_2 x := 
by
  sorry

end polynomial_q_correct_l698_69879


namespace log_inequality_l698_69837

theorem log_inequality
  (a : ℝ := Real.log 4 / Real.log 5)
  (b : ℝ := (Real.log 3 / Real.log 5)^2)
  (c : ℝ := Real.log 5 / Real.log 4) :
  b < a ∧ a < c :=
by
  sorry

end log_inequality_l698_69837


namespace root_of_quadratic_eq_l698_69800

open Complex

theorem root_of_quadratic_eq :
  ∃ z1 z2 : ℂ, (z1 = 3.5 - I) ∧ (z2 = -2.5 + I) ∧ (∀ z : ℂ, z^2 - z = 6 - 6 * I → (z = z1 ∨ z = z2)) := 
sorry

end root_of_quadratic_eq_l698_69800


namespace monthly_salary_equals_l698_69893

-- Define the base salary
def base_salary : ℝ := 1600

-- Define the commission rate
def commission_rate : ℝ := 0.04

-- Define the sales amount for which the salaries are equal
def sales_amount : ℝ := 5000

-- Define the total earnings with a base salary and commission for 5000 worth of sales
def total_earnings : ℝ := base_salary + (commission_rate * sales_amount)

-- Define the monthly salary from Furniture by Design
def monthly_salary : ℝ := 1800

-- Prove that the monthly salary S is equal to 1800
theorem monthly_salary_equals :
  total_earnings = monthly_salary :=
by
  -- The proof is skipped with sorry.
  sorry

end monthly_salary_equals_l698_69893


namespace time_for_A_l698_69849

noncomputable def work_days (A B C D E : ℝ) : Prop :=
  (1/A + 1/B + 1/C + 1/D = 1/8) ∧
  (1/B + 1/C + 1/D + 1/E = 1/6) ∧
  (1/A + 1/E = 1/12)

theorem time_for_A (A B C D E : ℝ) (h : work_days A B C D E) : A = 48 :=
  by
    sorry

end time_for_A_l698_69849


namespace coin_flips_heads_l698_69886

theorem coin_flips_heads (H T : ℕ) (flip_condition : H + T = 211) (tail_condition : T = H + 81) :
    H = 65 :=
by
  sorry

end coin_flips_heads_l698_69886


namespace number_of_marked_points_l698_69801

theorem number_of_marked_points
  (a1 a2 b1 b2 : ℕ)
  (hA : a1 * a2 = 50)
  (hB : b1 * b2 = 56)
  (h_sum : a1 + a2 = b1 + b2) :
  a1 + a2 + 1 = 16 :=
sorry

end number_of_marked_points_l698_69801


namespace lana_needs_to_sell_more_muffins_l698_69810

/--
Lana aims to sell 20 muffins at the bake sale.
She sells 12 muffins in the morning.
She sells another 4 in the afternoon.
How many more muffins does Lana need to sell to hit her goal?
-/
theorem lana_needs_to_sell_more_muffins (goal morningSales afternoonSales : ℕ)
  (h_goal : goal = 20) (h_morning : morningSales = 12) (h_afternoon : afternoonSales = 4) :
  goal - (morningSales + afternoonSales) = 4 :=
by
  sorry

end lana_needs_to_sell_more_muffins_l698_69810


namespace sum_of_abc_is_12_l698_69890

theorem sum_of_abc_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
by
  sorry

end sum_of_abc_is_12_l698_69890


namespace sum_underlined_numbers_non_negative_l698_69839

def sum_underlined_numbers (seq : Fin 100 → Int) : Bool :=
  let underlined_indices : List (Fin 100) :=
    List.range 100 |>.filter (λ i =>
      seq i > 0 ∨ (i < 99 ∧ seq i + seq (i + 1) > 0) ∨ (i < 98 ∧ seq i + seq (i + 1) + seq (i + 2) > 0))
  let underlined_sum : Int := underlined_indices.map (λ i => seq i) |>.sum
  underlined_sum ≤ 0

theorem sum_underlined_numbers_non_negative {seq : Fin 100 → Int} :
  ¬ sum_underlined_numbers seq :=
sorry

end sum_underlined_numbers_non_negative_l698_69839


namespace factorization_correct_l698_69874

theorem factorization_correct (x : ℝ) :
  16 * x ^ 2 + 8 * x - 24 = 8 * (2 * x ^ 2 + x - 3) ∧ (2 * x ^ 2 + x - 3) = (2 * x + 3) * (x - 1) :=
by
  sorry

end factorization_correct_l698_69874


namespace percent_of_z_l698_69897

variable {x y z : ℝ}

theorem percent_of_z (h₁ : x = 1.20 * y) (h₂ : y = 0.50 * z) : x = 0.60 * z :=
by
  sorry

end percent_of_z_l698_69897


namespace total_distance_yards_remaining_yards_l698_69854

structure Distance where
  miles : Nat
  yards : Nat

def marathon_distance : Distance :=
  { miles := 26, yards := 385 }

def miles_to_yards (miles : Nat) : Nat :=
  miles * 1760

def total_yards_in_marathon (d : Distance) : Nat :=
  miles_to_yards d.miles + d.yards

def total_distance_in_yards (d : Distance) (n : Nat) : Nat :=
  n * total_yards_in_marathon d

def remaining_yards (total_yards : Nat) (yards_in_mile : Nat) : Nat :=
  total_yards % yards_in_mile

theorem total_distance_yards_remaining_yards :
    let total_yards := total_distance_in_yards marathon_distance 15
    remaining_yards total_yards 1760 = 495 :=
by
  sorry

end total_distance_yards_remaining_yards_l698_69854


namespace average_score_of_entire_class_l698_69896

theorem average_score_of_entire_class :
  ∀ (num_students num_boys : ℕ) (avg_score_girls avg_score_boys : ℝ),
  num_students = 50 →
  num_boys = 20 →
  avg_score_girls = 85 →
  avg_score_boys = 80 →
  (avg_score_boys * num_boys + avg_score_girls * (num_students - num_boys)) / num_students = 83 :=
by
  intros num_students num_boys avg_score_girls avg_score_boys
  sorry

end average_score_of_entire_class_l698_69896


namespace david_and_maria_ages_l698_69816

theorem david_and_maria_ages 
  (D Y M : ℕ)
  (h1 : Y = D + 7)
  (h2 : Y = 2 * D)
  (h3 : M = D + 4)
  (h4 : M = Y / 2)
  : D = 7 ∧ M = 11 := by
  sorry

end david_and_maria_ages_l698_69816


namespace part1_l698_69852

theorem part1 (m n p : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : p > 0) : 
  (n / m) < (n + p) / (m + p) := 
sorry

end part1_l698_69852


namespace train_cross_bridge_time_l698_69809

open Nat

-- Defining conditions as per the problem
def train_length : ℕ := 200
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 5 / 18
def total_distance : ℕ := train_length + bridge_length
def time_to_cross : ℕ := total_distance / speed_mps

-- Stating the theorem
theorem train_cross_bridge_time : time_to_cross = 35 := by
  sorry

end train_cross_bridge_time_l698_69809


namespace sum_of_subsets_l698_69878

theorem sum_of_subsets (a1 a2 a3 : ℝ) (h : (a1 + a2 + a3) + (a1 + a2 + a1 + a3 + a2 + a3) = 12) : 
  a1 + a2 + a3 = 4 := 
by 
  sorry

end sum_of_subsets_l698_69878


namespace part1_part2_l698_69836

variables {a m n : ℝ}

theorem part1 (h1 : a^m = 2) (h2 : a^n = 3) : a^(4*m + 3*n) = 432 :=
by sorry

theorem part2 (h1 : a^m = 2) (h2 : a^n = 3) : a^(5*m - 2*n) = 32 / 9 :=
by sorry

end part1_part2_l698_69836


namespace line_properties_l698_69825

theorem line_properties : 
  ∃ (m b : ℝ), 
  (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 7) → y = m * x + b) ∧
  m + b = 3 ∧
  (∀ x : ℝ, ∀ y : ℝ, (x = 0 ∧ y = 1) → y = m * x + b) :=
sorry

end line_properties_l698_69825


namespace joan_balloons_l698_69872

theorem joan_balloons (m t j : ℕ) (h1 : m = 41) (h2 : t = 81) : j = t - m → j = 40 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end joan_balloons_l698_69872


namespace parabola_problem_l698_69860

noncomputable def p_value_satisfy_all_conditions (p : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A B : ℝ × ℝ),
    F = (p / 2, 0) ∧
    (A.2 = A.1 - p / 2 ∧ (A.2)^2 = 2 * p * A.1) ∧
    (B.2 = B.1 - p / 2 ∧ (B.2)^2 = 2 * p * B.1) ∧
    (A.1 + B.1) / 2 = 3 * p / 2 ∧
    (A.2 + B.2) / 2 = p ∧
    (p - 2 = -3 * p / 2)

theorem parabola_problem : ∃ (p : ℝ), p_value_satisfy_all_conditions p ∧ p = 4 / 5 :=
by
  sorry

end parabola_problem_l698_69860


namespace original_cookie_price_l698_69808

theorem original_cookie_price (C : ℝ) (h1 : 1.5 * 16 + (C / 2) * 8 = 32) : C = 2 :=
by
  -- Proof omitted
  sorry

end original_cookie_price_l698_69808


namespace jill_basket_total_weight_l698_69869

def jill_basket_capacity : ℕ := 24
def type_a_weight : ℕ := 150
def type_b_weight : ℕ := 170
def jill_basket_type_a_count : ℕ := 12
def jill_basket_type_b_count : ℕ := 12

theorem jill_basket_total_weight :
  (jill_basket_type_a_count * type_a_weight + jill_basket_type_b_count * type_b_weight) = 3840 :=
by
  -- We provide the calculations for clarification; not essential to the theorem statement
  -- (12 * 150) + (12 * 170) = 1800 + 2040 = 3840
  -- Started proof to provide context; actual proof steps are omitted
  sorry

end jill_basket_total_weight_l698_69869


namespace frequency_calculation_l698_69811

-- Define the given conditions
def sample_capacity : ℕ := 20
def group_frequency : ℚ := 0.25

-- The main theorem statement
theorem frequency_calculation :
  sample_capacity * group_frequency = 5 :=
by sorry

end frequency_calculation_l698_69811


namespace more_people_needed_to_paint_fence_l698_69858

theorem more_people_needed_to_paint_fence :
  ∀ (n t m t' : ℕ), n = 8 → t = 3 → t' = 2 → (n * t = m * t') → m - n = 4 :=
by
  intros n t m t'
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end more_people_needed_to_paint_fence_l698_69858


namespace find_point_B_l698_69823

structure Point where
  x : Int
  y : Int

def translation (p : Point) (dx dy : Int) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem find_point_B :
  let A := Point.mk (-2) 3
  let A' := Point.mk 3 2
  let B' := Point.mk 4 0
  let dx := 5
  let dy := -1
  (translation A dx dy = A') →
  ∃ B : Point, translation B dx dy = B' ∧ B = Point.mk (-1) (-1) :=
by
  intros
  use Point.mk (-1) (-1)
  constructor
  sorry
  rfl

end find_point_B_l698_69823


namespace paityn_red_hats_l698_69817

theorem paityn_red_hats (R : ℕ) : 
  (R + 24 + (4 / 5) * ↑R + 48 = 108) → R = 20 :=
by
  intro h
  sorry


end paityn_red_hats_l698_69817


namespace part1_part2_l698_69889

-- Define the constants based on given conditions
def cost_price : ℕ := 5
def initial_selling_price : ℕ := 9
def initial_sales_volume : ℕ := 32
def price_increment : ℕ := 2
def sales_decrement : ℕ := 8

-- Part 1: Define the elements 
def selling_price_part1 : ℕ := 11
def profit_per_item_part1 : ℕ := 6
def daily_sales_volume_part1 : ℕ := 24

theorem part1 :
  (selling_price_part1 - cost_price = profit_per_item_part1) ∧ 
  (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price_part1 - initial_selling_price) = daily_sales_volume_part1) := 
by
  sorry

-- Part 2: Define the elements 
def target_daily_profit : ℕ := 140
def selling_price1_part2 : ℕ := 12
def selling_price2_part2 : ℕ := 10

theorem part2 :
  (((selling_price1_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price1_part2 - initial_selling_price)) = target_daily_profit) ∨
  ((selling_price2_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price2_part2 - initial_selling_price)) = target_daily_profit)) :=
by
  sorry

end part1_part2_l698_69889


namespace length_of_platform_l698_69845

-- Definitions for the given conditions
def speed_of_train_kmph : ℕ := 54
def speed_of_train_mps : ℕ := 15
def time_to_pass_platform : ℕ := 16
def time_to_pass_man : ℕ := 10

-- Main statement of the problem
theorem length_of_platform (v_kmph : ℕ) (v_mps : ℕ) (t_p : ℕ) (t_m : ℕ) 
    (h1 : v_kmph = 54) 
    (h2 : v_mps = 15) 
    (h3 : t_p = 16) 
    (h4 : t_m = 10) : 
    v_mps * t_p - v_mps * t_m = 90 := 
sorry

end length_of_platform_l698_69845


namespace boys_without_notebooks_l698_69891

theorem boys_without_notebooks
  (total_boys : ℕ) (students_with_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24) (h2 : students_with_notebooks = 30) (h3 : girls_with_notebooks = 17) :
  total_boys - (students_with_notebooks - girls_with_notebooks) = 11 :=
by
  sorry

end boys_without_notebooks_l698_69891


namespace naomi_drives_to_parlor_l698_69830

theorem naomi_drives_to_parlor (d v t t_back : ℝ)
  (ht : t = d / v)
  (ht_back : t_back = 2 * d / v)
  (h_total : 2 * (t + t_back) = 6) : 
  t = 1 :=
by sorry

end naomi_drives_to_parlor_l698_69830


namespace nat_games_volunteer_allocation_l698_69821

theorem nat_games_volunteer_allocation 
  (volunteers : Fin 6 → Type) 
  (venues : Fin 3 → Type)
  (A B : volunteers 0)
  (remaining : Fin 4 → Type) 
  (assigned_pairings : Π (v : Fin 3), Fin 2 → volunteers 0) :
  (∀ v, assigned_pairings v 0 = A ∨ assigned_pairings v 1 = B) →
  (3 * 6 = 18) := 
by
  sorry

end nat_games_volunteer_allocation_l698_69821


namespace complete_the_square_l698_69805

theorem complete_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  intro h
  sorry

end complete_the_square_l698_69805


namespace relay_race_order_count_l698_69864

-- Definitions based on the given conditions
def team_members : List String := ["Sam", "Priya", "Jordan", "Luis"]
def first_runner := "Sam"
def last_runner := "Jordan"

-- Theorem stating the number of different possible orders
theorem relay_race_order_count {team_members first_runner last_runner} :
  (team_members = ["Sam", "Priya", "Jordan", "Luis"]) →
  (first_runner = "Sam") →
  (last_runner = "Jordan") →
  (2 = 2) :=
by
  intros _ _ _
  sorry

end relay_race_order_count_l698_69864


namespace greatest_divisor_of_620_and_180_l698_69824

/-- This theorem asserts that the greatest divisor of 620 that 
    is smaller than 100 and also a factor of 180 is 20. -/
theorem greatest_divisor_of_620_and_180 (d : ℕ) (h1 : d ∣ 620) (h2 : d ∣ 180) (h3 : d < 100) : d ≤ 20 :=
by
  sorry

end greatest_divisor_of_620_and_180_l698_69824


namespace inequality_solution_l698_69885

theorem inequality_solution (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) : a < -1 :=
sorry

end inequality_solution_l698_69885


namespace solution_x_chemical_b_l698_69857

theorem solution_x_chemical_b (percentage_x_a percentage_y_a percentage_y_b : ℝ) :
  percentage_x_a = 0.3 →
  percentage_y_a = 0.4 →
  percentage_y_b = 0.6 →
  (0.8 * percentage_x_a + 0.2 * percentage_y_a = 0.32) →
  (100 * (1 - percentage_x_a) = 70) :=
by {
  sorry
}

end solution_x_chemical_b_l698_69857


namespace pair_comparison_l698_69831

theorem pair_comparison :
  (∀ (a b : ℤ), (a, b) = (-2^4, (-2)^4) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (5^3, 3^5) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (-(-3), -|-3|) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = ((-1)^2, (-1)^2008) → a = b) :=
by
  sorry

end pair_comparison_l698_69831


namespace solve_fraction_equation_l698_69806

theorem solve_fraction_equation (x : ℝ) (hx1 : 0 < x) (hx2 : (x - 6) / 12 = 6 / (x - 12)) : x = 18 := 
sorry

end solve_fraction_equation_l698_69806


namespace complement_M_l698_69851

noncomputable def U : Set ℝ := Set.univ

def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M : U \ M = { x | x < -2 ∨ x > 2 } :=
by 
  sorry

end complement_M_l698_69851


namespace committee_problem_solution_l698_69859

def committee_problem : Prop :=
  let total_committees := Nat.choose 15 5
  let zero_profs_committees := Nat.choose 8 5
  let one_prof_committees := (Nat.choose 7 1) * (Nat.choose 8 4)
  let undesirable_committees := zero_profs_committees + one_prof_committees
  let desired_committees := total_committees - undesirable_committees
  desired_committees = 2457

theorem committee_problem_solution : committee_problem :=
by
  sorry

end committee_problem_solution_l698_69859


namespace joe_paint_left_after_third_week_l698_69863

def initial_paint : ℕ := 360

def paint_used_first_week (initial_paint : ℕ) : ℕ := initial_paint / 4

def paint_left_after_first_week (initial_paint : ℕ) : ℕ := initial_paint - paint_used_first_week initial_paint

def paint_used_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week / 2

def paint_left_after_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week - paint_used_second_week paint_left_after_first_week

def paint_used_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week * 2 / 3

def paint_left_after_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week - paint_used_third_week paint_left_after_second_week

theorem joe_paint_left_after_third_week : 
  paint_left_after_third_week (paint_left_after_second_week (paint_left_after_first_week initial_paint)) = 45 :=
by 
  sorry

end joe_paint_left_after_third_week_l698_69863


namespace sum_of_coefficients_l698_69803

def u (n : ℕ) : ℕ := 
  match n with
  | 0 => 6 -- Assume the sequence starts at u_0 for easier indexing
  | n + 1 => u n + 5 + 2 * n

theorem sum_of_coefficients (u : ℕ → ℕ) : 
  (∀ n, u (n + 1) = u n + 5 + 2 * n) ∧ u 1 = 6 → 
  (∃ a b c : ℕ, (∀ n, u n = a * n^2 + b * n + c) ∧ a + b + c = 6) := 
by
  sorry

end sum_of_coefficients_l698_69803


namespace trigonometric_identity_l698_69861

theorem trigonometric_identity : 
  let sin := Real.sin
  let cos := Real.cos
  sin 18 * cos 63 - sin 72 * sin 117 = - (Real.sqrt 2 / 2) :=
by
  -- The proof would go here
  sorry

end trigonometric_identity_l698_69861


namespace gcd_45_75_l698_69840

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l698_69840


namespace four_numbers_are_perfect_squares_l698_69882

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem four_numbers_are_perfect_squares (a b c d : ℕ) (h1 : is_perfect_square (a * b * c))
                                                      (h2 : is_perfect_square (a * c * d))
                                                      (h3 : is_perfect_square (b * c * d))
                                                      (h4 : is_perfect_square (a * b * d)) : 
                                                      is_perfect_square a ∧
                                                      is_perfect_square b ∧
                                                      is_perfect_square c ∧
                                                      is_perfect_square d :=
by
  sorry

end four_numbers_are_perfect_squares_l698_69882


namespace tank_capacity_correctness_l698_69892

noncomputable def tankCapacity : ℝ := 77.65

theorem tank_capacity_correctness (T : ℝ) 
  (h_initial: T * (5 / 8) + 11 = T * (23 / 30)) : 
  T = tankCapacity := 
by
  sorry

end tank_capacity_correctness_l698_69892


namespace triangle_angle_bisector_l698_69848

theorem triangle_angle_bisector 
  (a b l : ℝ) (h1: a > 0) (h2: b > 0) (h3: l > 0) :
  ∃ α : ℝ, α = 2 * Real.arccos (l * (a + b) / (2 * a * b)) :=
by
  sorry

end triangle_angle_bisector_l698_69848


namespace rational_operation_example_l698_69841

def rational_operation (a b : ℚ) : ℚ := a^3 - 2 * a * b + 4

theorem rational_operation_example : rational_operation 4 (-9) = 140 := 
by
  sorry

end rational_operation_example_l698_69841


namespace sequence_general_term_l698_69813

open Nat

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  (∀ n : ℕ, 0 < n → a (n + 2) ≤ a n + 3 * 2^n) ∧
  (∀ n : ℕ, 0 < n → a (n + 1) ≥ 2 * a n + 1)

theorem sequence_general_term (a : ℕ → ℕ) (h : sequence_a a) :
  ∀ n : ℕ, 0 < n → a n = 2^n - 1 :=
by
  sorry

end sequence_general_term_l698_69813


namespace triangle_angle_and_side_l698_69865

theorem triangle_angle_and_side (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
  (h2 : a + b = 6)
  (h3 : 1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3)
  : C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := by
  -- proof omitted
  sorry

end triangle_angle_and_side_l698_69865


namespace art_group_students_count_l698_69802

theorem art_group_students_count (x : ℕ) (h1 : x * (1 / 60) + 2 * (x + 15) * (1 / 60) = 1) : x = 10 :=
by {
  sorry
}

end art_group_students_count_l698_69802


namespace range_of_a_analytical_expression_l698_69812

variables {f : ℝ → ℝ}

-- Problem 1
theorem range_of_a (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ x y, x < y → f x ≥ f y)
  {a : ℝ} (h_ineq : f (1 - a) + f (1 - 2 * a) < 0) :
  0 < a ∧ a ≤ 2 / 3 :=
sorry

-- Problem 2
theorem analytical_expression 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, 0 < x ∧ x < 1 → f x = x^2 + x + 1)
  (h_zero : f 0 = 0) :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f x = 
    if x > 0 then x^2 + x + 1
    else if x = 0 then 0
    else -x^2 + x - 1 :=
sorry

end range_of_a_analytical_expression_l698_69812


namespace canal_depth_l698_69867

theorem canal_depth (A : ℝ) (W_top : ℝ) (W_bottom : ℝ) (d : ℝ) (h: ℝ)
  (h₁ : A = 840) 
  (h₂ : W_top = 12) 
  (h₃ : W_bottom = 8)
  (h₄ : A = (1/2) * (W_top + W_bottom) * d) : 
  d = 84 :=
by 
  sorry

end canal_depth_l698_69867


namespace TomTotalWeight_l698_69884

def TomWeight : ℝ := 150
def HandWeight (personWeight: ℝ) : ℝ := 1.5 * personWeight
def VestWeight (personWeight: ℝ) : ℝ := 0.5 * personWeight
def TotalHandWeight (handWeight: ℝ) : ℝ := 2 * handWeight
def TotalWeight (totalHandWeight vestWeight: ℝ) : ℝ := totalHandWeight + vestWeight

theorem TomTotalWeight : TotalWeight (TotalHandWeight (HandWeight TomWeight)) (VestWeight TomWeight) = 525 := 
by
  sorry

end TomTotalWeight_l698_69884


namespace max_xy_l698_69843

theorem max_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 2 * x + 3 * y = 6) : 
  xy ≤ (3/2) :=
sorry

end max_xy_l698_69843


namespace vasya_can_interfere_with_petya_goal_l698_69835

theorem vasya_can_interfere_with_petya_goal :
  ∃ (evens odds : ℕ), evens + odds = 50 ∧ (evens + odds) % 2 = 1 :=
sorry

end vasya_can_interfere_with_petya_goal_l698_69835


namespace find_tricksters_in_16_questions_l698_69832

-- Definitions
def Inhabitant := {i : Nat // i < 65}
def isKnight (i : Inhabitant) : Prop := sorry  -- placeholder for actual definition
def isTrickster (i : Inhabitant) : Prop := sorry -- placeholder for actual definition

-- Conditions
def condition1 : ∀ i : Inhabitant, isKnight i ∨ isTrickster i := sorry
def condition2 : ∃ t1 t2 : Inhabitant, t1 ≠ t2 ∧ isTrickster t1 ∧ isTrickster t2 :=
  sorry
def condition3 : ∀ t1 t2 : Inhabitant, t1 ≠ t2 → ¬(isTrickster t1 ∧ isTrickster t2) → isKnight t1 ∧ isKnight t2 := 
  sorry 
def question (i : Inhabitant) (group : List Inhabitant) : Prop :=
  ∀ j ∈ group, isKnight j

-- Theorem statement
theorem find_tricksters_in_16_questions : ∃ (knight : Inhabitant) (knaves : (Inhabitant × Inhabitant)), 
  isKnight knight ∧ isTrickster knaves.fst ∧ isTrickster knaves.snd ∧ knaves.fst ≠ knaves.snd ∧
  (∀ questionsAsked ≤ 30, sorry) :=
  sorry

end find_tricksters_in_16_questions_l698_69832


namespace probability_A_will_receive_2_awards_l698_69846

def classes := Fin 4
def awards := 8

-- The number of ways to distribute 4 remaining awards to 4 classes
noncomputable def total_distributions : ℕ :=
  Nat.choose (awards - 4 + 4 - 1) (4 - 1)

-- The number of ways when class A receives exactly 2 awards
noncomputable def favorable_distributions : ℕ :=
  Nat.choose (2 + 3 - 1) (4 - 1)

-- The probability that class A receives exactly 2 out of 8 awards
noncomputable def probability_A_receives_2_awards : ℚ :=
  favorable_distributions / total_distributions

theorem probability_A_will_receive_2_awards :
  probability_A_receives_2_awards = 2 / 7 := by
  sorry

end probability_A_will_receive_2_awards_l698_69846


namespace Shekar_science_marks_l698_69807

-- Define Shekar's known marks
def math_marks : ℕ := 76
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 47
def biology_marks : ℕ := 85

-- Define the average mark and the number of subjects
def average_mark : ℕ := 71
def number_of_subjects : ℕ := 5

-- Define Shekar's unknown mark in Science
def science_marks : ℕ := sorry  -- We expect to prove science_marks = 65

-- State the theorem to be proved
theorem Shekar_science_marks :
  average_mark * number_of_subjects = math_marks + science_marks + social_studies_marks + english_marks + biology_marks →
  science_marks = 65 :=
by sorry

end Shekar_science_marks_l698_69807


namespace expression_simplifies_to_neg_seven_l698_69895

theorem expression_simplifies_to_neg_seven (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
(h₃ : a + b + c = 0) (h₄ : ab + ac + bc ≠ 0) : 
    (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
by
  sorry

end expression_simplifies_to_neg_seven_l698_69895


namespace usual_time_to_school_l698_69870

theorem usual_time_to_school (R T : ℕ) (h : 7 * R * (T - 4) = 6 * R * T) : T = 28 :=
sorry

end usual_time_to_school_l698_69870


namespace kim_boxes_sold_on_tuesday_l698_69868

theorem kim_boxes_sold_on_tuesday :
  ∀ (T W Th F : ℕ),
  (T = 3 * W) →
  (W = 2 * Th) →
  (Th = 3 / 2 * F) →
  (F = 600) →
  T = 5400 :=
by
  intros T W Th F h1 h2 h3 h4
  sorry

end kim_boxes_sold_on_tuesday_l698_69868


namespace product_of_sum_positive_and_quotient_negative_l698_69827

-- Definitions based on conditions in the problem
def sum_positive (a b : ℝ) : Prop := a + b > 0
def quotient_negative (a b : ℝ) : Prop := a / b < 0

-- Problem statement as a theorem
theorem product_of_sum_positive_and_quotient_negative (a b : ℝ)
  (h1 : sum_positive a b)
  (h2 : quotient_negative a b) :
  a * b < 0 := by
  sorry

end product_of_sum_positive_and_quotient_negative_l698_69827


namespace find_max_problems_l698_69844

def max_problems_in_7_days (P : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, i ∈ Finset.range 7 → P i ≤ 10) ∧
  (∀ i : ℕ, i ∈ Finset.range 5 → (P i > 7) → (P (i + 1) ≤ 5 ∧ P (i + 2) ≤ 5))

theorem find_max_problems : ∃ P : ℕ → ℕ, max_problems_in_7_days P ∧ (Finset.range 7).sum P = 50 :=
by
  sorry

end find_max_problems_l698_69844


namespace sum_of_cube_edges_l698_69815

theorem sum_of_cube_edges (edge_len : ℝ) (num_edges : ℕ) (lengths : ℝ) (h1 : edge_len = 15) (h2 : num_edges = 12) : lengths = num_edges * edge_len :=
by
  sorry

end sum_of_cube_edges_l698_69815


namespace spherical_coordinates_cone_l698_69804

open Real

-- Define spherical coordinates and the equation φ = c
def spherical_coordinates (ρ θ φ : ℝ) : Prop := 
  ∃ (c : ℝ), φ = c

-- Prove that φ = c describes a cone
theorem spherical_coordinates_cone (ρ θ : ℝ) (c : ℝ) :
  spherical_coordinates ρ θ c → ∃ ρ' θ', spherical_coordinates ρ' θ' c :=
by
  sorry

end spherical_coordinates_cone_l698_69804


namespace primes_solution_l698_69862

theorem primes_solution (p q : ℕ) (m n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m ≥ 2) (hn : n ≥ 2) :
    p^n = q^m + 1 ∨ p^n = q^m - 1 → (p = 2 ∧ n = 3 ∧ q = 3 ∧ m = 2) :=
by
  sorry

end primes_solution_l698_69862


namespace problem_statement_l698_69838

noncomputable def a : ℝ := 13 / 2
noncomputable def b : ℝ := -4

theorem problem_statement :
  ∀ k : ℝ, ∃ x : ℝ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6 ↔ x = 1 :=
by
  sorry

end problem_statement_l698_69838


namespace perpendicular_bisector_of_circles_l698_69818

theorem perpendicular_bisector_of_circles
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6 * x = 0) :
  ∃ x y : ℝ, (3 * x - y - 9 = 0) :=
by
  sorry

end perpendicular_bisector_of_circles_l698_69818


namespace find_OH_squared_l698_69847

theorem find_OH_squared (R a b c : ℝ) (hR : R = 10) (hsum : a^2 + b^2 + c^2 = 50) : 
  9 * R^2 - (a^2 + b^2 + c^2) = 850 :=
by
  sorry

end find_OH_squared_l698_69847


namespace cost_price_of_article_l698_69819

theorem cost_price_of_article
  (C SP1 SP2 : ℝ)
  (h1 : SP1 = 0.8 * C)
  (h2 : SP2 = 1.05 * C)
  (h3 : SP2 = SP1 + 100) : 
  C = 400 := 
sorry

end cost_price_of_article_l698_69819


namespace toys_produced_each_day_l698_69833

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) (h₁ : weekly_production = 4340) (h₂ : days_worked = 2) : weekly_production / days_worked = 2170 :=
by {
  -- Proof can be filled in here
  sorry
}

end toys_produced_each_day_l698_69833


namespace subtract_square_l698_69820

theorem subtract_square (n : ℝ) (h : n = 68.70953354520753) : (n^2 - 20^2) = 4321.000000000001 := by
  sorry

end subtract_square_l698_69820


namespace geometric_series_sum_eq_l698_69853

theorem geometric_series_sum_eq :
  let a := (5 : ℚ)
  let r := (-1/2 : ℚ)
  (∑' n : ℕ, a * r^n) = (10 / 3 : ℚ) :=
by
  sorry

end geometric_series_sum_eq_l698_69853


namespace cost_per_mile_l698_69855

theorem cost_per_mile (m x : ℝ) (h_cost_eq : 2.50 + x * m = 2.50 + 5.00 + x * 14) : 
  x = 5 / 14 :=
by
  sorry

end cost_per_mile_l698_69855


namespace find_angle_B_find_triangle_area_l698_69876

open Real

theorem find_angle_B (B : ℝ) (h : sqrt 3 * sin (2 * B) = 1 - cos (2 * B)) : B = π / 3 :=
sorry

theorem find_triangle_area (BC A B : ℝ) (hBC : BC = 2) (hA : A = π / 4) (hB : B = π / 3) :
  let AC := BC * (sin B / sin A)
  let C := π - A - B
  let area := (1 / 2) * AC * BC * sin C
  area = (3 + sqrt 3) / 2 :=
sorry


end find_angle_B_find_triangle_area_l698_69876


namespace num_divisors_360_l698_69826

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end num_divisors_360_l698_69826


namespace find_cost_price_l698_69883

variable (CP : ℝ)

def selling_price (CP : ℝ) := CP * 1.40

theorem find_cost_price (h : selling_price CP = 1680) : CP = 1200 :=
by
  sorry

end find_cost_price_l698_69883


namespace B_finish_work_alone_in_12_days_l698_69834

theorem B_finish_work_alone_in_12_days (A_days B_days both_days : ℕ) :
  A_days = 6 →
  both_days = 4 →
  (1 / A_days + 1 / B_days = 1 / both_days) →
  B_days = 12 :=
by
  intros hA hBoth hRate
  sorry

end B_finish_work_alone_in_12_days_l698_69834


namespace part_1_part_2_l698_69829

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (1 + 2^x)

theorem part_1 (a : ℝ) (h1 : f a 1 + f a (-1) = 0) : a = 1 / 2 :=
by sorry

theorem part_2 : ∃ a : ℝ, ∀ x : ℝ, f a (-x) + f a x = 0 :=
by sorry

end part_1_part_2_l698_69829


namespace relationship_between_A_and_B_l698_69881

theorem relationship_between_A_and_B (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let A := a^2
  let B := 2 * a - 1
  A > B :=
by
  let A := a^2
  let B := 2 * a - 1
  sorry

end relationship_between_A_and_B_l698_69881


namespace divides_floor_factorial_div_l698_69898

theorem divides_floor_factorial_div {m n : ℕ} (h1 : 1 < m) (h2 : m < n + 2) (h3 : 3 < n) :
  (m - 1) ∣ (n! / m) :=
sorry

end divides_floor_factorial_div_l698_69898


namespace loss_percent_l698_69877

theorem loss_percent (C S : ℝ) (h : 100 * S = 40 * C) : ((C - S) / C) * 100 = 60 :=
by
  sorry

end loss_percent_l698_69877


namespace carl_marbles_l698_69899

-- Define initial conditions
def initial_marbles : ℕ := 12
def lost_marbles : ℕ := initial_marbles / 2
def remaining_marbles : ℕ := initial_marbles - lost_marbles
def additional_marbles : ℕ := 10
def new_marbles_from_mother : ℕ := 25

-- Define the final number of marbles Carl will put back in the jar
def total_marbles_put_back : ℕ := remaining_marbles + additional_marbles + new_marbles_from_mother

-- Statement to be proven
theorem carl_marbles : total_marbles_put_back = 41 :=
by
  sorry

end carl_marbles_l698_69899
