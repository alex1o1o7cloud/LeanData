import Mathlib

namespace lines_are_skew_l1379_137965

def line1 (a t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 1 + 4 * t, a + 5 * t)
  
def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 3 + 3 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) : (∀ t u : ℝ, line1 a t ≠ line2 u) ↔ a ≠ -4/5 :=
sorry

end lines_are_skew_l1379_137965


namespace sam_total_distance_l1379_137991

-- Definitions based on conditions
def first_half_distance : ℕ := 120
def first_half_time : ℕ := 3
def second_half_distance : ℕ := 80
def second_half_time : ℕ := 2
def sam_time : ℚ := 5.5

-- Marguerite's overall average speed
def marguerite_average_speed : ℚ := (first_half_distance + second_half_distance) / (first_half_time + second_half_time)

-- Theorem statement: Sam's total distance driven
theorem sam_total_distance : ∀ (d : ℚ), d = (marguerite_average_speed * sam_time) ↔ d = 220 := by
  intro d
  sorry

end sam_total_distance_l1379_137991


namespace standard_deviation_is_one_l1379_137941

noncomputable def standard_deviation (μ : ℝ) (σ : ℝ) : Prop :=
  ∀ x : ℝ, (0.68 * μ ≤ x ∧ x ≤ 1.32 * μ) → σ = 1

theorem standard_deviation_is_one (a : ℝ) (σ : ℝ) :
  (0.68 * a ≤ a + σ ∧ a + σ ≤ 1.32 * a) → σ = 1 :=
by
  -- Proof omitted.
  sorry

end standard_deviation_is_one_l1379_137941


namespace pure_imaginary_solution_l1379_137901

-- Defining the main problem as a theorem in Lean 4

theorem pure_imaginary_solution (m : ℝ) : 
  (∃ a b : ℝ, (m^2 - m = a ∧ a = 0) ∧ (m^2 - 3 * m + 2 = b ∧ b ≠ 0)) → 
  m = 0 :=
sorry -- Proof is omitted as per the instructions

end pure_imaginary_solution_l1379_137901


namespace initial_roses_in_vase_l1379_137903

/-- 
There were some roses in a vase. Mary cut roses from her flower garden 
and put 16 more roses in the vase. There are now 22 roses in the vase.
Prove that the initial number of roses in the vase was 6. 
-/
theorem initial_roses_in_vase (initial_roses added_roses current_roses : ℕ) 
  (h_add : added_roses = 16) 
  (h_current : current_roses = 22) 
  (h_current_eq : current_roses = initial_roses + added_roses) : 
  initial_roses = 6 := 
by
  subst h_add
  subst h_current
  linarith

end initial_roses_in_vase_l1379_137903


namespace part1_part2_l1379_137999

open Real

def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |3 * x + a|

theorem part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} := by
  sorry

theorem part2 (h : ∃ x_0 : ℝ, f x_0 (a := a) + 2 * |x_0 - 2| < 3) : -9 < a ∧ a < -3 := by
  sorry

end part1_part2_l1379_137999


namespace closest_multiple_of_18_l1379_137984

def is_multiple_of_2 (n : ℤ) : Prop := n % 2 = 0
def is_multiple_of_9 (n : ℤ) : Prop := n % 9 = 0
def is_multiple_of_18 (n : ℤ) : Prop := is_multiple_of_2 n ∧ is_multiple_of_9 n

theorem closest_multiple_of_18 (n : ℤ) (h : n = 2509) : 
  ∃ k : ℤ, is_multiple_of_18 k ∧ (abs (2509 - k) = 7) :=
sorry

end closest_multiple_of_18_l1379_137984


namespace bernardo_probability_is_correct_l1379_137902

noncomputable def bernardo_larger_probability : ℚ :=
  let total_bernardo_combinations := (Nat.choose 10 3 : ℚ)
  let total_silvia_combinations := (Nat.choose 8 3 : ℚ)
  let bernardo_has_10 := (Nat.choose 8 2 : ℚ) / total_bernardo_combinations
  let bernardo_not_has_10 := ((total_silvia_combinations - 1) / total_silvia_combinations) / 2
  bernardo_has_10 * 1 + (1 - bernardo_has_10) * bernardo_not_has_10

theorem bernardo_probability_is_correct :
  bernardo_larger_probability = 19 / 28 := by
  sorry

end bernardo_probability_is_correct_l1379_137902


namespace smallest_positive_value_is_A_l1379_137944

noncomputable def expr_A : ℝ := 12 - 4 * Real.sqrt 8
noncomputable def expr_B : ℝ := 4 * Real.sqrt 8 - 12
noncomputable def expr_C : ℝ := 20 - 6 * Real.sqrt 10
noncomputable def expr_D : ℝ := 60 - 15 * Real.sqrt 16
noncomputable def expr_E : ℝ := 15 * Real.sqrt 16 - 60

theorem smallest_positive_value_is_A :
  expr_A = 12 - 4 * Real.sqrt 8 ∧ 
  expr_B = 4 * Real.sqrt 8 - 12 ∧ 
  expr_C = 20 - 6 * Real.sqrt 10 ∧ 
  expr_D = 60 - 15 * Real.sqrt 16 ∧ 
  expr_E = 15 * Real.sqrt 16 - 60 ∧ 
  expr_A > 0 ∧ 
  expr_A < expr_C := 
sorry

end smallest_positive_value_is_A_l1379_137944


namespace expected_waiting_time_approx_l1379_137918

noncomputable def expectedWaitingTime : ℚ :=
  (10 * (1/2) + 30 * (1/3) + 50 * (1/36) + 70 * (1/12) + 90 * (1/18))

theorem expected_waiting_time_approx :
  abs (expectedWaitingTime - 27.22) < 1 :=
by
  sorry

end expected_waiting_time_approx_l1379_137918


namespace chapatis_ordered_l1379_137925

theorem chapatis_ordered (C : ℕ) 
  (chapati_cost : ℕ) (plates_rice : ℕ) (rice_cost : ℕ)
  (plates_mixed_veg : ℕ) (mixed_veg_cost : ℕ)
  (ice_cream_cups : ℕ) (ice_cream_cost : ℕ)
  (total_amount_paid : ℕ)
  (cost_eq : chapati_cost = 6)
  (plates_rice_eq : plates_rice = 5)
  (rice_cost_eq : rice_cost = 45)
  (plates_mixed_veg_eq : plates_mixed_veg = 7)
  (mixed_veg_cost_eq : mixed_veg_cost = 70)
  (ice_cream_cups_eq : ice_cream_cups = 6)
  (ice_cream_cost_eq : ice_cream_cost = 40)
  (total_paid_eq : total_amount_paid = 1051) :
  6 * C + 5 * 45 + 7 * 70 + 6 * 40 = 1051 → C = 16 :=
by
  intro h
  sorry

end chapatis_ordered_l1379_137925


namespace sqrt_a_minus_b_squared_eq_one_l1379_137943

noncomputable def PointInThirdQuadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b < 0

noncomputable def DistanceToYAxis (a : ℝ) : Prop :=
  abs a = 5

noncomputable def BCondition (b : ℝ) : Prop :=
  abs (b + 1) = 3

theorem sqrt_a_minus_b_squared_eq_one
    (a b : ℝ)
    (h1 : PointInThirdQuadrant a b)
    (h2 : DistanceToYAxis a)
    (h3 : BCondition b) :
    Real.sqrt ((a - b) ^ 2) = 1 := 
  sorry

end sqrt_a_minus_b_squared_eq_one_l1379_137943


namespace finish_fourth_task_l1379_137953

noncomputable def time_task_starts : ℕ := 12 -- Time in hours (12:00 PM)
noncomputable def time_task_ends : ℕ := 15 -- Time in hours (3:00 PM)
noncomputable def total_tasks : ℕ := 4 -- Total number of tasks
noncomputable def tasks_time (tasks: ℕ) := (time_task_ends - time_task_starts) * 60 / (total_tasks - 1) -- Time in minutes for each task

theorem finish_fourth_task : tasks_time 1 + ((total_tasks - 1) * tasks_time 1) = 240 := -- 4:00 PM expressed as 240 minutes from 12:00 PM
by
  sorry

end finish_fourth_task_l1379_137953


namespace multiply_fractions_l1379_137949

theorem multiply_fractions :
  (2 / 9) * (5 / 14) = 5 / 63 :=
by
  sorry

end multiply_fractions_l1379_137949


namespace remainder_of_power_division_l1379_137980

-- Define the main entities
def power : ℕ := 3
def exponent : ℕ := 19
def divisor : ℕ := 10

-- Define the proof problem
theorem remainder_of_power_division :
  (power ^ exponent) % divisor = 7 := 
  by 
    sorry

end remainder_of_power_division_l1379_137980


namespace broken_perfect_spiral_shells_difference_l1379_137971

theorem broken_perfect_spiral_shells_difference :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let not_spiral_perfect_shells := 12
  let spiral_perfect_shells := perfect_shells - not_spiral_perfect_shells
  broken_spiral_shells - spiral_perfect_shells = 21 := by
  sorry

end broken_perfect_spiral_shells_difference_l1379_137971


namespace triangle_abs_simplification_l1379_137909

theorem triangle_abs_simplification
  (x y z : ℝ)
  (h1 : x + y > z)
  (h2 : y + z > x)
  (h3 : x + z > y) :
  |x + y - z| - 2 * |y - x - z| = -x + 3 * y - 3 * z :=
by
  sorry

end triangle_abs_simplification_l1379_137909


namespace hyperbola_equation_center_origin_asymptote_l1379_137945

theorem hyperbola_equation_center_origin_asymptote
  (center_origin : ∀ x y : ℝ, x = 0 ∧ y = 0)
  (focus_parabola : ∃ x : ℝ, 4 * x^2 = 8 * x)
  (asymptote : ∀ x y : ℝ, x + y = 0):
  ∃ a b : ℝ, a^2 = 2 ∧ b^2 = 2 ∧ (x^2 / 2) - (y^2 / 2) = 1 := 
sorry

end hyperbola_equation_center_origin_asymptote_l1379_137945


namespace tire_price_l1379_137987

theorem tire_price (x : ℕ) (h : 4 * x + 5 = 485) : x = 120 :=
by
  sorry

end tire_price_l1379_137987


namespace range_of_m_l1379_137988

theorem range_of_m (m : ℝ) (h : (8 - m) / (m - 5) > 1) : 5 < m ∧ m < 13 / 2 :=
by
  sorry

end range_of_m_l1379_137988


namespace pool_capacity_l1379_137937

theorem pool_capacity:
  (∃ (V1 V2 : ℝ) (t : ℝ), 
    (V1 = t / 120) ∧ 
    (V2 = V1 + 50) ∧ 
    (V1 + V2 = t / 48) ∧ 
    t = 12000) := 
by 
  sorry

end pool_capacity_l1379_137937


namespace expected_rolls_to_2010_l1379_137979

noncomputable def expected_rolls_for_sum (n : ℕ) : ℝ :=
  if n = 2010 then 574.5238095 else sorry

theorem expected_rolls_to_2010 : expected_rolls_for_sum 2010 = 574.5238095 := sorry

end expected_rolls_to_2010_l1379_137979


namespace find_intended_number_l1379_137904

theorem find_intended_number (x : ℕ) 
    (condition : 3 * x = (10 * 3 * x + 2) / 19 + 7) : 
    x = 5 :=
sorry

end find_intended_number_l1379_137904


namespace arithmetic_seq_S10_l1379_137910

open BigOperators

variables (a : ℕ → ℚ) (d : ℚ)

-- Definitions based on the conditions
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) := ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 5 = 1
axiom h2 : a 1 + a 7 + a 10 = a 4 + a 6

-- We aim to prove the sum of the first 10 terms
def S (n : ℕ) :=
  ∑ i in Finset.range n, a (i + 1)

theorem arithmetic_seq_S10 : arithmetic_seq a d → S a 10 = 25 / 3 :=
by
  sorry

end arithmetic_seq_S10_l1379_137910


namespace triangle_side_length_range_l1379_137921

theorem triangle_side_length_range (x : ℝ) : 
  (1 < x) ∧ (x < 9) → ¬ (x = 10) :=
by
  sorry

end triangle_side_length_range_l1379_137921


namespace min_performances_l1379_137905

theorem min_performances (total_singers : ℕ) (m : ℕ) (n_pairs : ℕ := 28) (pairs_performance : ℕ := 6)
  (condition : total_singers = 108) 
  (const_pairs : ∀ (r : ℕ), (n_pairs * r = pairs_performance * m)) : m ≥ 14 :=
by
  sorry

end min_performances_l1379_137905


namespace cone_volume_proof_l1379_137946

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_proof :
  (cone_volume 1 (Real.sqrt 3)) = (Real.sqrt 3 / 3) * Real.pi :=
by
  sorry

end cone_volume_proof_l1379_137946


namespace squirrels_and_nuts_l1379_137927

theorem squirrels_and_nuts (number_of_squirrels number_of_nuts : ℕ) 
    (h1 : number_of_squirrels = 4) 
    (h2 : number_of_squirrels = number_of_nuts + 2) : 
    number_of_nuts = 2 :=
by
  sorry

end squirrels_and_nuts_l1379_137927


namespace cos_A_eq_find_a_l1379_137956

variable {A B C a b c : ℝ}

-- Proposition 1: If in triangle ABC, b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then cos A = sqrt 6 / 4
theorem cos_A_eq (h : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) : Real.cos A = Real.sqrt 6 / 4 :=
sorry

-- Proposition 2: Given b = sqrt 6, B = 2 * A, and b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then a = 2
theorem find_a (h1 : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) (h2 : B = 2 * A) (h3 : b = Real.sqrt 6) : a = 2 :=
sorry

end cos_A_eq_find_a_l1379_137956


namespace range_of_a_l1379_137952

open Real

noncomputable def f (x a : ℝ) : ℝ := (exp x / 2) - (a / exp x)

def condition (x₁ x₂ a : ℝ) : Prop :=
  x₁ ≠ x₂ ∧ 1 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1 ≤ x₂ ∧ x₂ ≤ 2 ∧ ((abs (f x₁ a) - abs (f x₂ a)) * (x₁ - x₂) > 0)

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), condition x₁ x₂ a) ↔ (- (exp 2) / 2 ≤ a ∧ a ≤ (exp 2) / 2) :=
by
  sorry

end range_of_a_l1379_137952


namespace max_roses_purchase_l1379_137968

/--
Given three purchasing options for roses:
1. Individual roses cost $5.30 each.
2. One dozen (12) roses cost $36.
3. Two dozen (24) roses cost $50.
Given a total budget of $680, prove that the maximum number of roses that can be purchased is 317.
-/
noncomputable def max_roses : ℝ := 317

/--
Prove that given the purchasing options and the budget, the maximum number of roses that can be purchased is 317.
-/
theorem max_roses_purchase (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 5.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  max_roses = 317 := 
sorry

end max_roses_purchase_l1379_137968


namespace evaluate_expression_l1379_137995

variable {c d : ℝ}

theorem evaluate_expression (h : c ≠ d ∧ c ≠ -d) :
  (c^4 - d^4) / (2 * (c^2 - d^2)) = (c^2 + d^2) / 2 :=
by sorry

end evaluate_expression_l1379_137995


namespace number_of_chain_links_l1379_137985

noncomputable def length_of_chain (number_of_links : ℕ) : ℝ :=
  (number_of_links * (7 / 3)) + 1

theorem number_of_chain_links (n m : ℕ) (d : ℝ) (thickness : ℝ) (max_length min_length : ℕ) 
  (h1 : d = 2 + 1 / 3)
  (h2 : thickness = 0.5)
  (h3 : max_length = 36)
  (h4 : min_length = 22)
  (h5 : m = n + 6)
  : length_of_chain n = 22 ∧ length_of_chain m = 36 
  :=
  sorry

end number_of_chain_links_l1379_137985


namespace time_to_groom_rottweiler_l1379_137997

theorem time_to_groom_rottweiler
  (R : ℕ)  -- Time to groom a rottweiler
  (B : ℕ)  -- Time to groom a border collie
  (C : ℕ)  -- Time to groom a chihuahua
  (total_time_6R_9B_1C : 6 * R + 9 * B + C = 255)  -- Total time for grooming 6 rottweilers, 9 border collies, and 1 chihuahua
  (time_to_groom_border_collie : B = 10)  -- Time to groom a border collie is 10 minutes
  (time_to_groom_chihuahua : C = 45) :  -- Time to groom a chihuahua is 45 minutes
  R = 20 :=  -- Prove that it takes 20 minutes to groom a rottweiler
by
  sorry

end time_to_groom_rottweiler_l1379_137997


namespace lewis_found_20_items_l1379_137942

noncomputable def tanya_items : ℕ := 4

noncomputable def samantha_items : ℕ := 4 * tanya_items

noncomputable def lewis_items : ℕ := samantha_items + 4

theorem lewis_found_20_items : lewis_items = 20 := by
  sorry

end lewis_found_20_items_l1379_137942


namespace child_support_amount_l1379_137954

-- Definitions
def base_salary_1_3 := 30000
def base_salary_4_7 := 36000
def bonus_1 := 2000
def bonus_2 := 3000
def bonus_3 := 4000
def bonus_4 := 5000
def bonus_5 := 6000
def bonus_6 := 7000
def bonus_7 := 8000
def child_support_1_5 := 30 / 100
def child_support_6_7 := 25 / 100
def paid_total := 1200

-- Total Income per year
def income_year_1 := base_salary_1_3 + bonus_1
def income_year_2 := base_salary_1_3 + bonus_2
def income_year_3 := base_salary_1_3 + bonus_3
def income_year_4 := base_salary_4_7 + bonus_4
def income_year_5 := base_salary_4_7 + bonus_5
def income_year_6 := base_salary_4_7 + bonus_6
def income_year_7 := base_salary_4_7 + bonus_7

-- Child Support per year
def support_year_1 := child_support_1_5 * income_year_1
def support_year_2 := child_support_1_5 * income_year_2
def support_year_3 := child_support_1_5 * income_year_3
def support_year_4 := child_support_1_5 * income_year_4
def support_year_5 := child_support_1_5 * income_year_5
def support_year_6 := child_support_6_7 * income_year_6
def support_year_7 := child_support_6_7 * income_year_7

-- Total Support calculation
def total_owed := support_year_1 + support_year_2 + support_year_3 + 
                  support_year_4 + support_year_5 +
                  support_year_6 + support_year_7

-- Final amount owed
def amount_owed := total_owed - paid_total

-- Theorem statement
theorem child_support_amount :
  amount_owed = 75150 :=
sorry

end child_support_amount_l1379_137954


namespace expression_equality_l1379_137911

-- Define the conditions
variables {a b x : ℝ}
variable (h1 : x = a / b)
variable (h2 : a ≠ 2 * b)
variable (h3 : b ≠ 0)

-- Define and state the theorem
theorem expression_equality : (2 * a + b) / (a + 2 * b) = (2 * x + 1) / (x + 2) :=
by 
  intros
  sorry

end expression_equality_l1379_137911


namespace sam_grew_3_carrots_l1379_137930

-- Let Sandy's carrots and the total number of carrots be defined
def sandy_carrots : ℕ := 6
def total_carrots : ℕ := 9

-- Define the number of carrots grown by Sam
def sam_carrots : ℕ := total_carrots - sandy_carrots

-- The theorem to prove
theorem sam_grew_3_carrots : sam_carrots = 3 := by
  sorry

end sam_grew_3_carrots_l1379_137930


namespace maximal_sum_of_xy_l1379_137957

theorem maximal_sum_of_xy (x y : ℤ) (h : x^2 + y^2 = 100) : ∃ (s : ℤ), s = 14 ∧ ∀ (u v : ℤ), u^2 + v^2 = 100 → u + v ≤ s :=
by sorry

end maximal_sum_of_xy_l1379_137957


namespace smallest_single_discount_l1379_137916

noncomputable def discount1 : ℝ := (1 - 0.20) * (1 - 0.20)
noncomputable def discount2 : ℝ := (1 - 0.10) * (1 - 0.15)
noncomputable def discount3 : ℝ := (1 - 0.08) * (1 - 0.08) * (1 - 0.08)

theorem smallest_single_discount : ∃ n : ℕ, (1 - n / 100) < discount1 ∧ (1 - n / 100) < discount2 ∧ (1 - n / 100) < discount3 ∧ n = 37 := sorry

end smallest_single_discount_l1379_137916


namespace fraction_addition_l1379_137982

theorem fraction_addition :
  (1 / 3 * 2 / 5) + 1 / 4 = 23 / 60 := 
  sorry

end fraction_addition_l1379_137982


namespace a_range_of_proposition_l1379_137970

theorem a_range_of_proposition (a : ℝ) : (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + 5 <= a * x) ↔ a ∈ Set.Ici (2 * Real.sqrt 5) := by
  sorry

end a_range_of_proposition_l1379_137970


namespace axis_of_symmetry_parabola_l1379_137975

theorem axis_of_symmetry_parabola (x y : ℝ) : 
  (∃ k : ℝ, (y^2 = -8 * k) → (y^2 = -8 * x) → x = -1) :=
by
  sorry

end axis_of_symmetry_parabola_l1379_137975


namespace six_times_expression_l1379_137912

theorem six_times_expression {x y Q : ℝ} (h : 3 * (4 * x + 5 * y) = Q) : 
  6 * (8 * x + 10 * y) = 4 * Q :=
by
  sorry

end six_times_expression_l1379_137912


namespace gcd_455_299_eq_13_l1379_137919

theorem gcd_455_299_eq_13 : Nat.gcd 455 299 = 13 := by
  sorry

end gcd_455_299_eq_13_l1379_137919


namespace marbles_problem_l1379_137933

theorem marbles_problem (a : ℚ) (h1: 34 * a = 156) : a = 78 / 17 := 
by
  sorry

end marbles_problem_l1379_137933


namespace find_integer_pairs_l1379_137934

theorem find_integer_pairs (m n : ℤ) (h1 : m * n ≥ 0) (h2 : m^3 + n^3 + 99 * m * n = 33^3) :
  (m = -33 ∧ n = -33) ∨ ∃ k : ℕ, k ≤ 33 ∧ m = k ∧ n = 33 - k ∨ m = 33 - k ∧ n = k :=
by
  sorry

end find_integer_pairs_l1379_137934


namespace initial_water_amount_l1379_137960

theorem initial_water_amount (E D R F I : ℕ) 
  (hE : E = 2000) 
  (hD : D = 3500) 
  (hR : R = 350 * (30 / 10))
  (hF : F = 1550) 
  (h : I - (E + D) + R = F) : 
  I = 6000 :=
by
  sorry

end initial_water_amount_l1379_137960


namespace bridge_length_l1379_137977

variable (speed : ℝ) (time_minutes : ℝ)
variable (time_hours : ℝ := time_minutes / 60)

theorem bridge_length (h1 : speed = 5) (h2 : time_minutes = 15) : 
  speed * time_hours = 1.25 := by
  sorry

end bridge_length_l1379_137977


namespace mrs_white_expected_yield_l1379_137913

noncomputable def orchard_yield : ℝ :=
  let length_in_feet : ℝ := 10 * 3
  let width_in_feet : ℝ := 30 * 3
  let total_area : ℝ := length_in_feet * width_in_feet
  let half_area : ℝ := total_area / 2
  let tomato_yield : ℝ := half_area * 0.75
  let cucumber_yield : ℝ := half_area * 0.4
  tomato_yield + cucumber_yield

theorem mrs_white_expected_yield :
  orchard_yield = 1552.5 := sorry

end mrs_white_expected_yield_l1379_137913


namespace B_speaks_truth_60_l1379_137962

variable (P_A P_B P_A_and_B : ℝ)

-- Given conditions
def A_speaks_truth_85 : Prop := P_A = 0.85
def both_speak_truth_051 : Prop := P_A_and_B = 0.51

-- Solution condition
noncomputable def B_speaks_truth_percentage : ℝ := P_A_and_B / P_A

-- Statement to prove
theorem B_speaks_truth_60 (hA : A_speaks_truth_85 P_A) (hAB : both_speak_truth_051 P_A_and_B) : B_speaks_truth_percentage P_A_and_B P_A = 0.6 :=
by
  rw [A_speaks_truth_85] at hA
  rw [both_speak_truth_051] at hAB
  unfold B_speaks_truth_percentage
  sorry

end B_speaks_truth_60_l1379_137962


namespace validate_model_and_profit_range_l1379_137935

noncomputable def is_exponential_model_valid (x y : ℝ) : Prop :=
  ∃ T a : ℝ, T > 0 ∧ a > 1 ∧ y = T * a^x

noncomputable def is_profitable_for_at_least_one_billion (x : ℝ) : Prop :=
  (∃ T a : ℝ, T > 0 ∧ a > 1 ∧ 1/5 * (Real.sqrt 2)^x ≥ 10 ∧ 0 < x ∧ x ≤ 12) ∨
  (-0.2 * (x - 12) * (x - 17) + 12.8 ≥ 10 ∧ x > 12)

theorem validate_model_and_profit_range :
  (is_exponential_model_valid 2 0.4) ∧
  (is_exponential_model_valid 4 0.8) ∧
  (is_exponential_model_valid 12 12.8) ∧
  is_profitable_for_at_least_one_billion 11.3 ∧
  is_profitable_for_at_least_one_billion 19 :=
by
  sorry

end validate_model_and_profit_range_l1379_137935


namespace charlie_has_largest_final_answer_l1379_137963

theorem charlie_has_largest_final_answer :
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  charlie > alice ∧ charlie > bob :=
by
  -- Definitions of intermediate variables
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  -- Comparison assertions
  sorry

end charlie_has_largest_final_answer_l1379_137963


namespace shortest_time_to_camp_l1379_137914

/-- 
Given:
- The width of the river is 1 km.
- The camp is 1 km away from the point directly across the river.
- Swimming speed is 2 km/hr.
- Walking speed is 3 km/hr.

Prove the shortest time required to reach the camp is (2 + √5) / 6 hours.
--/
theorem shortest_time_to_camp :
  ∃ t : ℝ, t = (2 + Real.sqrt 5) / 6 := 
sorry

end shortest_time_to_camp_l1379_137914


namespace total_spent_on_toys_l1379_137915

-- Definitions for costs
def cost_car : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_truck : ℝ := 5.86

-- The statement to prove
theorem total_spent_on_toys : cost_car + cost_skateboard + cost_truck = 25.62 := by
  sorry

end total_spent_on_toys_l1379_137915


namespace sequence_periodic_from_some_term_l1379_137976

def is_bounded (s : ℕ → ℤ) (M : ℤ) : Prop :=
  ∀ n, |s n| ≤ M

def is_periodic_from (s : ℕ → ℤ) (N : ℕ) (p : ℕ) : Prop :=
  ∀ n, s (N + n) = s (N + n + p)

theorem sequence_periodic_from_some_term (s : ℕ → ℤ) (M : ℤ) (h_bounded : is_bounded s M)
    (h_recurrence : ∀ n, s (n + 5) = (5 * s (n + 4) ^ 3 + s (n + 3) - 3 * s (n + 2) + s n) / (2 * s (n + 2) + s (n + 1) ^ 2 + s (n + 1) * s n)) :
    ∃ N p, is_periodic_from s N p := by
  sorry

end sequence_periodic_from_some_term_l1379_137976


namespace probability_X_eq_Y_l1379_137928

-- Define the conditions as functions or predicates.
def is_valid_pair (x y : ℝ) : Prop :=
  -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi ∧ -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi ∧ Real.cos (Real.cos x) = Real.cos (Real.cos y)

-- Final statement asserting the required probability.
theorem probability_X_eq_Y :
  ∃ (prob : ℝ), prob = 1 / 11 ∧ ∀ (x y : ℝ), is_valid_pair x y → (x = y ∨ x ≠ y ∧ prob = 1/11) :=
  sorry

end probability_X_eq_Y_l1379_137928


namespace coins_dimes_count_l1379_137948

theorem coins_dimes_count :
  ∃ (p n d q : ℕ), 
    p + n + d + q = 10 ∧ 
    p + 5 * n + 10 * d + 25 * q = 110 ∧ 
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 2 ∧ d = 5 :=
by {
    sorry
}

end coins_dimes_count_l1379_137948


namespace find_x_l1379_137992

theorem find_x (x : ℤ) (h : (2008 + x)^2 = x^2) : x = -1004 :=
sorry

end find_x_l1379_137992


namespace complex_in_third_quadrant_l1379_137939

open Complex

noncomputable def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else 4

theorem complex_in_third_quadrant (z : ℂ) (h : (2 + I) * z = -I) : quadrant z = 3 := by
  sorry

end complex_in_third_quadrant_l1379_137939


namespace externally_tangent_circles_m_l1379_137955

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2_eqn (x y m : ℝ) : Prop := x^2 + y^2 - 2 * m * x + m^2 - 1 = 0

theorem externally_tangent_circles_m (m : ℝ) :
  (∀ x y : ℝ, circle1_eqn x y) →
  (∀ x y : ℝ, circle2_eqn x y m) →
  m = 3 ∨ m = -3 :=
by sorry

end externally_tangent_circles_m_l1379_137955


namespace OHara_triple_example_l1379_137973

def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a + Real.sqrt b = x)

theorem OHara_triple_example : is_OHara_triple 36 25 11 :=
by {
  sorry
}

end OHara_triple_example_l1379_137973


namespace range_of_independent_variable_l1379_137967

theorem range_of_independent_variable (x : ℝ) (hx : 1 - 2 * x ≥ 0) : x ≤ 0.5 :=
sorry

end range_of_independent_variable_l1379_137967


namespace proportion_equation_correct_l1379_137917

theorem proportion_equation_correct (x y : ℝ) (h1 : 2 * x = 3 * y) (h2 : x ≠ 0) (h3 : y ≠ 0) : 
  x / 3 = y / 2 := 
  sorry

end proportion_equation_correct_l1379_137917


namespace ball_returns_to_bella_after_13_throws_l1379_137908

theorem ball_returns_to_bella_after_13_throws:
  ∀ (girls : Fin 13) (n : ℕ), (∃ k, k > 0 ∧ (1 + k * 5) % 13 = 1) → (n = 13) :=
by
  sorry

end ball_returns_to_bella_after_13_throws_l1379_137908


namespace c_divides_n_l1379_137961

theorem c_divides_n (a b c n : ℤ) (h : a * n^2 + b * n + c = 0) : c ∣ n :=
sorry

end c_divides_n_l1379_137961


namespace simplify_expression_l1379_137964

theorem simplify_expression (a b : ℤ) : 4 * a + 5 * b - a - 7 * b = 3 * a - 2 * b :=
by
  sorry

end simplify_expression_l1379_137964


namespace prime_and_n_eq_m_minus_1_l1379_137969

theorem prime_and_n_eq_m_minus_1 (n m : ℕ) (h1 : n ≥ 2) (h2 : m ≥ 2)
  (h3 : ∀ k : ℕ, k ∈ Finset.range n.succ → k^n % m = 1) : Nat.Prime m ∧ n = m - 1 := 
sorry

end prime_and_n_eq_m_minus_1_l1379_137969


namespace volume_of_prism_l1379_137900

variables (a b c : ℝ)
variables (ab_prod : a * b = 36) (ac_prod : a * c = 48) (bc_prod : b * c = 72)

theorem volume_of_prism : a * b * c = 352.8 :=
by
  sorry

end volume_of_prism_l1379_137900


namespace focal_radii_l1379_137990

theorem focal_radii (a e x y : ℝ) (h1 : x + y = 2 * a) (h2 : x - y = 2 * e) : x = a + e ∧ y = a - e :=
by
  -- We will add here the actual proof, but for now, we leave it as a placeholder.
  sorry

end focal_radii_l1379_137990


namespace least_number_to_divisible_l1379_137951

theorem least_number_to_divisible (x : ℕ) : 
  (∃ x, (1049 + x) % 25 = 0) ∧ (∀ y, y < x → (1049 + y) % 25 ≠ 0) ↔ x = 1 :=
by
  sorry

end least_number_to_divisible_l1379_137951


namespace number_line_move_l1379_137929

theorem number_line_move (A B: ℤ):  A = -3 → B = A + 4 → B = 1 := by
  intros hA hB
  rw [hA] at hB
  rw [hB]
  sorry

end number_line_move_l1379_137929


namespace margaret_time_correct_l1379_137938

def margaret_time : ℕ :=
  let n := 7
  let r := 15
  (Nat.factorial n) / r

theorem margaret_time_correct : margaret_time = 336 := by
  sorry

end margaret_time_correct_l1379_137938


namespace find_b_l1379_137931

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 4) : b = 4 :=
sorry

end find_b_l1379_137931


namespace problem_solution_l1379_137906

theorem problem_solution (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 2) (h3 : a^3 + b^3 + c^3 = 3) :
  (a * b * c = 1 / 6) ∧ (a^4 + b^4 + c^4 = 25 / 6) :=
by {
  sorry
}

end problem_solution_l1379_137906


namespace tan_alpha_plus_pi_div_4_l1379_137936

noncomputable def tan_plus_pi_div_4 (α : ℝ) : ℝ := Real.tan (α + Real.pi / 4)

theorem tan_alpha_plus_pi_div_4 (α : ℝ) 
  (h1 : α > Real.pi / 2) 
  (h2 : α < Real.pi) 
  (h3 : (Real.cos α, Real.sin α) • (Real.cos α ^ 2, Real.sin α - 1) = 1 / 5)
  : tan_plus_pi_div_4 α = -1 / 7 := sorry

end tan_alpha_plus_pi_div_4_l1379_137936


namespace calc_perm_product_l1379_137922

-- Define the permutation function
def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Lean statement to prove the given problem
theorem calc_perm_product : permutation 6 2 * permutation 4 2 = 360 := 
by
  -- Test the calculations if necessary, otherwise use sorry
  sorry

end calc_perm_product_l1379_137922


namespace quadratic_equation_solution_unique_l1379_137940

theorem quadratic_equation_solution_unique (b : ℝ) (hb : b ≠ 0) (h1_sol : ∀ x1 x2 : ℝ, 2*b*x1^2 + 16*x1 + 5 = 0 → 2*b*x2^2 + 16*x2 + 5 = 0 → x1 = x2) :
  ∃ x : ℝ, x = -5/8 ∧ 2*b*x^2 + 16*x + 5 = 0 :=
by
  sorry

end quadratic_equation_solution_unique_l1379_137940


namespace trader_sold_pens_l1379_137932

theorem trader_sold_pens (C : ℝ) (N : ℕ) (hC : C > 0) (h_gain : N * (2 / 5) = 40) : N = 100 :=
by
  sorry

end trader_sold_pens_l1379_137932


namespace max_value_l1379_137974

open Real

theorem max_value (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha1 : a ≤ 1) (hb1 : b ≤ 1) (hc1 : c ≤ 1/2) :
  sqrt (a * b * c) + sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ (1 / sqrt 2) + (1 / 2) :=
sorry

end max_value_l1379_137974


namespace find_cost_10_pound_bag_l1379_137986

def cost_5_pound_bag : ℝ := 13.82
def cost_25_pound_bag : ℝ := 32.25
def minimum_required_weight : ℝ := 65
def maximum_required_weight : ℝ := 80
def least_possible_cost : ℝ := 98.75
def cost_10_pound_bag (cost : ℝ) : Prop :=
  ∃ n m l, 
    (n * 5 + m * 10 + l * 25 ≥ minimum_required_weight) ∧
    (n * 5 + m * 10 + l * 25 ≤ maximum_required_weight) ∧
    (n * cost_5_pound_bag + m * cost + l * cost_25_pound_bag = least_possible_cost)

theorem find_cost_10_pound_bag : cost_10_pound_bag 2 := 
by
  sorry

end find_cost_10_pound_bag_l1379_137986


namespace lizard_eyes_l1379_137947

theorem lizard_eyes (E W S : Nat) 
  (h1 : W = 3 * E) 
  (h2 : S = 7 * W) 
  (h3 : E = S + W - 69) : 
  E = 3 := 
by
  sorry

end lizard_eyes_l1379_137947


namespace range_of_x_plus_2y_l1379_137994

theorem range_of_x_plus_2y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) : x + 2 * y ≥ 9 :=
sorry

end range_of_x_plus_2y_l1379_137994


namespace students_between_jimin_yuna_l1379_137907

theorem students_between_jimin_yuna 
  (total_students : ℕ) 
  (jimin_position : ℕ) 
  (yuna_position : ℕ) 
  (h1 : total_students = 32) 
  (h2 : jimin_position = 27) 
  (h3 : yuna_position = 11) 
  : (jimin_position - yuna_position - 1) = 15 := 
by
  sorry

end students_between_jimin_yuna_l1379_137907


namespace spring_work_done_l1379_137959

theorem spring_work_done (F : ℝ) (l : ℝ) (stretched_length : ℝ) (k : ℝ) (W : ℝ) 
  (hF : F = 10) (hl : l = 0.1) (hk : k = F / l) (h_stretched_length : stretched_length = 0.06) : 
  W = 0.18 :=
by
  sorry

end spring_work_done_l1379_137959


namespace probability_blue_or_purple_is_correct_l1379_137923

def total_jelly_beans : ℕ := 7 + 8 + 9 + 10 + 4

def blue_jelly_beans : ℕ := 10

def purple_jelly_beans : ℕ := 4

def blue_or_purple_jelly_beans : ℕ := blue_jelly_beans + purple_jelly_beans

def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_correct :
  probability_blue_or_purple = 7 / 19 :=
by
  sorry

end probability_blue_or_purple_is_correct_l1379_137923


namespace small_pos_int_n_l1379_137920

theorem small_pos_int_n (a : ℕ → ℕ) (n : ℕ) (a1_val : a 1 = 7)
  (recurrence: ∀ n, a (n + 1) = a n * (a n + 2)) :
  ∃ n : ℕ, a n > 2 ^ 4036 ∧ ∀ m : ℕ, (m < n) → a m ≤ 2 ^ 4036 :=
by
  sorry

end small_pos_int_n_l1379_137920


namespace solve_fraction_eq_l1379_137978

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 3) :
  (x = 0 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) ∨ 
  (x = 2 / 3 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) :=
sorry

end solve_fraction_eq_l1379_137978


namespace license_plate_count_l1379_137966

-- Define the number of letters and digits
def num_letters := 26
def num_digits := 10
def num_odd_digits := 5  -- (1, 3, 5, 7, 9)
def num_even_digits := 5  -- (0, 2, 4, 6, 8)

-- Calculate the number of possible license plates
theorem license_plate_count : 
  (num_letters ^ 3) * ((num_even_digits * num_odd_digits * num_digits) * 3) = 13182000 :=
by sorry

end license_plate_count_l1379_137966


namespace find_positive_real_unique_solution_l1379_137998

theorem find_positive_real_unique_solution (x : ℝ) (h : 0 < x ∧ (x - 6) / 16 = 6 / (x - 16)) : x = 22 :=
sorry

end find_positive_real_unique_solution_l1379_137998


namespace circle_tangent_to_parabola_and_x_axis_eqn_l1379_137924

theorem circle_tangent_to_parabola_and_x_axis_eqn :
  (∃ (h k : ℝ), k^2 = 2 * h ∧ (x - h)^2 + (y - k)^2 = 2 * h ∧ k > 0) →
    (∀ (x y : ℝ), x^2 + y^2 - x - 2 * y + 1 / 4 = 0) := by
  sorry

end circle_tangent_to_parabola_and_x_axis_eqn_l1379_137924


namespace mul_99_101_square_98_l1379_137983

theorem mul_99_101 : 99 * 101 = 9999 := sorry

theorem square_98 : 98^2 = 9604 := sorry

end mul_99_101_square_98_l1379_137983


namespace locust_population_doubling_time_l1379_137981

theorem locust_population_doubling_time 
  (h: ℕ)
  (initial_population : ℕ := 1000)
  (time_past : ℕ := 4)
  (future_time: ℕ := 10)
  (population_limit: ℕ := 128000) :
  1000 * 2 ^ ((10 + 4) / h) > 128000 → h = 2 :=
by
  sorry

end locust_population_doubling_time_l1379_137981


namespace negation_of_proposition_l1379_137989

open Real

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > sin x) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end negation_of_proposition_l1379_137989


namespace new_circle_equation_l1379_137950

-- Define the initial conditions
def initial_circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0
def radius_of_new_circle : ℝ := 2

-- Define the target equation of the circle
def target_circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- The theorem statement
theorem new_circle_equation (x y : ℝ) :
  initial_circle_equation x y → target_circle_equation x y :=
sorry

end new_circle_equation_l1379_137950


namespace absent_laborers_l1379_137996

theorem absent_laborers (L : ℝ) (A : ℝ) (hL : L = 17.5) (h_work_done : (L - A) / 10 = L / 6) : A = 14 :=
by
  sorry

end absent_laborers_l1379_137996


namespace unique_solution_of_quadratics_l1379_137972

theorem unique_solution_of_quadratics (y : ℚ) 
    (h1 : 9 * y^2 + 8 * y - 3 = 0) 
    (h2 : 27 * y^2 + 35 * y - 12 = 0) : 
    y = 1 / 3 :=
sorry

end unique_solution_of_quadratics_l1379_137972


namespace father_twice_as_old_in_years_l1379_137993

-- Conditions
def father_age : ℕ := 42
def son_age : ℕ := 14
def years : ℕ := 14

-- Proof statement
theorem father_twice_as_old_in_years : (father_age + years) = 2 * (son_age + years) :=
by
  -- Proof content is omitted as per the instruction.
  sorry

end father_twice_as_old_in_years_l1379_137993


namespace pictures_left_l1379_137926

def zoo_pics : ℕ := 802
def museum_pics : ℕ := 526
def beach_pics : ℕ := 391
def amusement_park_pics : ℕ := 868
def duplicates_deleted : ℕ := 1395

theorem pictures_left : 
  (zoo_pics + museum_pics + beach_pics + amusement_park_pics - duplicates_deleted) = 1192 := 
by
  sorry

end pictures_left_l1379_137926


namespace square_of_chord_length_l1379_137958

/--
Given two circles with radii 10 and 7, and centers 15 units apart, if they intersect at a point P such that the chords QP and PR are of equal lengths, then the square of the length of chord QP is 289.
-/
theorem square_of_chord_length :
  ∀ (r1 r2 d x : ℝ), r1 = 10 → r2 = 7 → d = 15 →
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  cos_theta1 = cos_theta2 →
  x^2 = 289 := 
by
  intros r1 r2 d x h1 h2 h3
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  intro h4
  sorry

end square_of_chord_length_l1379_137958
