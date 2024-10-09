import Mathlib

namespace locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l175_17566

/-- Given a circle with center at point P passes through point A (1,0) 
    and is tangent to the line x = -1, the locus of point P is the parabola C. -/
theorem locus_of_P_is_parabola (P A : ℝ × ℝ) (x y : ℝ):
  (A = (1, 0)) → (P.1 + 1)^2 + P.2^2 = 0 → y^2 = 4 * x := 
sorry

/-- If the line passing through point H(4, 0) intersects the parabola 
    C (denoted by y^2 = 4x) at points M and N, and T is any point on 
    the line x = -4, then the slopes of lines TM, TH, and TN form an 
    arithmetic sequence. -/
theorem slopes_form_arithmetic_sequence (H M N T : ℝ × ℝ) (m n k : ℝ): 
  (H = (4, 0)) → (T.1 = -4) → 
  (M.1, M.2) = (k^2, 4*k) ∧ (N.1, N.2) = (m^2, 4*m) → 
  ((T.2 - M.2) / (T.1 - M.1) + (T.2 - N.2) / (T.1 - N.1)) = 
  2 * (T.2 / -8) := 
sorry

end locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l175_17566


namespace actual_distance_traveled_l175_17532

theorem actual_distance_traveled (D : ℕ) (h : (D:ℚ) / 12 = (D + 20) / 16) : D = 60 :=
sorry

end actual_distance_traveled_l175_17532


namespace remainder_4_power_100_div_9_l175_17555

theorem remainder_4_power_100_div_9 : (4^100) % 9 = 4 :=
by
  sorry

end remainder_4_power_100_div_9_l175_17555


namespace part_I_part_II_l175_17571

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

theorem part_I (α : ℝ) (hα : ∃ (P : ℝ × ℝ), P = (Real.sqrt 3, -1) ∧
  (Real.tan α = -1 / Real.sqrt 3 ∨ Real.tan α = - (Real.sqrt 3) / 3)) :
  f α = -3 := by
  sorry

theorem part_II (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  -2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end part_I_part_II_l175_17571


namespace duration_of_resulting_video_l175_17558

theorem duration_of_resulting_video 
    (vasya_walk_time : ℕ) (petya_walk_time : ℕ) 
    (sync_meet_point : ℕ) :
    vasya_walk_time = 8 → petya_walk_time = 5 → sync_meet_point = sync_meet_point → 
    (vasya_walk_time - sync_meet_point + petya_walk_time) = 5 :=
by
  intros
  sorry

end duration_of_resulting_video_l175_17558


namespace problem_1_problem_2_l175_17591

noncomputable def f (x : ℝ) : ℝ :=
  (Real.logb 3 (x / 27)) * (Real.logb 3 (3 * x))

theorem problem_1 (h₁ : 1 / 27 ≤ x)
(h₂ : x ≤ 1 / 9) :
    (∀ x, f x ≤ 12) ∧ (∃ x, f x = 5) := 
sorry

theorem problem_2
(m α β : ℝ)
(h₁ : f α + m = 0)
(h₂ : f β + m = 0) :
    α * β = 9 :=
sorry

end problem_1_problem_2_l175_17591


namespace value_of_expression_l175_17503

variable (x1 x2 : ℝ)

def sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 3
def product_roots (x1 x2 : ℝ) : Prop := x1 * x2 = -4

theorem value_of_expression (h1 : sum_roots x1 x2) (h2 : product_roots x1 x2) : 
  x1^2 - 4*x1 - x2 + 2*x1*x2 = -7 :=
by sorry

end value_of_expression_l175_17503


namespace remainder_77_pow_77_minus_15_mod_19_l175_17529

theorem remainder_77_pow_77_minus_15_mod_19 : (77^77 - 15) % 19 = 5 := by
  sorry

end remainder_77_pow_77_minus_15_mod_19_l175_17529


namespace cost_of_hiring_actors_l175_17544

theorem cost_of_hiring_actors
  (A : ℕ)
  (CostOfFood : ℕ := 150)
  (EquipmentRental : ℕ := 300 + 2 * A)
  (TotalCost : ℕ := 3 * A + 450)
  (SellingPrice : ℕ := 10000)
  (Profit : ℕ := 5950) :
  TotalCost = SellingPrice - Profit → A = 1200 :=
by
  intro h
  sorry

end cost_of_hiring_actors_l175_17544


namespace Ming_initial_ladybugs_l175_17572

-- Define the conditions
def Sami_spiders : Nat := 3
def Hunter_ants : Nat := 12
def insects_remaining : Nat := 21
def ladybugs_flew_away : Nat := 2

-- Formalize the proof problem
theorem Ming_initial_ladybugs : Sami_spiders + Hunter_ants + (insects_remaining + ladybugs_flew_away) - (Sami_spiders + Hunter_ants) = 8 := by
  sorry

end Ming_initial_ladybugs_l175_17572


namespace problem_statement_l175_17543

theorem problem_statement (p : ℕ) (hprime : Prime p) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ p = m^2 + n^2 ∧ p ∣ (m^3 + n^3 + 8 * m * n)) → p = 5 :=
by
  sorry

end problem_statement_l175_17543


namespace greatest_fourth_term_arith_seq_sum_90_l175_17523

theorem greatest_fourth_term_arith_seq_sum_90 :
  ∃ a d : ℕ, 6 * a + 15 * d = 90 ∧ (∀ n : ℕ, n < 6 → a + n * d > 0) ∧ (a + 3 * d = 17) :=
by
  sorry

end greatest_fourth_term_arith_seq_sum_90_l175_17523


namespace race_length_l175_17514

theorem race_length
  (B_s : ℕ := 50) -- Biff's speed in yards per minute
  (K_s : ℕ := 51) -- Kenneth's speed in yards per minute
  (D_above_finish : ℕ := 10) -- distance Kenneth is past the finish line when Biff finishes
  : {L : ℕ // L = 500} := -- the length of the race is 500 yards.
  sorry

end race_length_l175_17514


namespace interest_difference_l175_17561

def principal : ℝ := 3600
def rate : ℝ := 0.25
def time : ℕ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

theorem interest_difference :
  let SI := simple_interest principal rate time;
  let CI := compound_interest principal rate time;
  CI - SI = 225 :=
by
  sorry

end interest_difference_l175_17561


namespace tax_rate_correct_l175_17581

/-- The tax rate in dollars per $100.00 is $82.00, given that the tax rate as a percent is 82%. -/
theorem tax_rate_correct (x : ℝ) (h : x = 82) : (x / 100) * 100 = 82 :=
by
  rw [h]
  sorry

end tax_rate_correct_l175_17581


namespace money_collected_is_correct_l175_17521

-- Define the conditions as constants and definitions in Lean
def ticket_price_adult : ℝ := 0.60
def ticket_price_child : ℝ := 0.25
def total_persons : ℕ := 280
def children_attended : ℕ := 80

-- Define the number of adults
def adults_attended : ℕ := total_persons - children_attended

-- Define the total money collected
def total_money_collected : ℝ :=
  (adults_attended * ticket_price_adult) + (children_attended * ticket_price_child)

-- Statement to prove
theorem money_collected_is_correct :
  total_money_collected = 140 := by
  sorry

end money_collected_is_correct_l175_17521


namespace value_of_N_l175_17573

theorem value_of_N (a b c N : ℚ) (h1 : a + b + c = 120) (h2 : a - 10 = N) (h3 : 10 * b = N) (h4 : c - 10 = N) : N = 1100 / 21 := 
sorry

end value_of_N_l175_17573


namespace quadratic_properties_l175_17539

noncomputable def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ)
  (root_neg1 : quadratic a b c (-1) = 0)
  (ineq_condition : ∀ x : ℝ, (quadratic a b c x - x) * (quadratic a b c x - (x^2 + 1) / 2) ≤ 0) :
  quadratic a b c 1 = 1 ∧ ∀ x : ℝ, quadratic a b c x = (1 / 4) * x^2 + (1 / 2) * x + (1 / 4) :=
by
  sorry

end quadratic_properties_l175_17539


namespace annual_interest_rate_is_approx_14_87_percent_l175_17593

-- Let P be the principal amount, r the annual interest rate, and n the number of years
-- Given: A = P(1 + r)^n, where A is the amount of money after n years
-- In this problem: A = 2P, n = 5

theorem annual_interest_rate_is_approx_14_87_percent
    (P : Real) (r : Real) (n : Real) (A : Real) (condition1 : n = 5)
    (condition2 : A = 2 * P)
    (condition3 : A = P * (1 + r)^n) :
  r = 2^(1/5) - 1 := 
  sorry

end annual_interest_rate_is_approx_14_87_percent_l175_17593


namespace profit_ratio_l175_17547

theorem profit_ratio (I_P I_Q : ℝ) (t_P t_Q : ℕ) 
  (h1 : I_P / I_Q = 7 / 5)
  (h2 : t_P = 5)
  (h3 : t_Q = 14) : 
  (I_P * t_P) / (I_Q * t_Q) = 1 / 2 :=
by
  sorry

end profit_ratio_l175_17547


namespace arnold_danny_age_l175_17588

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 17 → x = 8 :=
by
  sorry

end arnold_danny_age_l175_17588


namespace silver_value_percentage_l175_17541

theorem silver_value_percentage
  (side_length : ℝ) (weight_per_cubic_inch : ℝ) (price_per_ounce : ℝ) 
  (selling_price : ℝ) (volume : ℝ) (weight : ℝ) (silver_value : ℝ) 
  (percentage_sold : ℝ ) 
  (h1 : side_length = 3) 
  (h2 : weight_per_cubic_inch = 6) 
  (h3 : price_per_ounce = 25)
  (h4 : selling_price = 4455)
  (h5 : volume = side_length^3)
  (h6 : weight = volume * weight_per_cubic_inch)
  (h7 : silver_value = weight * price_per_ounce)
  (h8 : percentage_sold = (selling_price / silver_value) * 100) :
  percentage_sold = 110 :=
by
  sorry

end silver_value_percentage_l175_17541


namespace miranda_savings_l175_17565

theorem miranda_savings:
  ∀ (months : ℕ) (sister_contribution price shipping total paid_per_month : ℝ),
    months = 3 →
    sister_contribution = 50 →
    price = 210 →
    shipping = 20 →
    total = 230 →
    total - sister_contribution = price + shipping →
    paid_per_month = (total - sister_contribution) / months →
    paid_per_month = 60 :=
by
  intros months sister_contribution price shipping total paid_per_month h1 h2 h3 h4 h5 h6 h7
  sorry

end miranda_savings_l175_17565


namespace solve_system_l175_17512

theorem solve_system :
  ∃ x y : ℝ, (x + 2*y = 1 ∧ 3*x - 2*y = 7) → (x = 2 ∧ y = -1/2) :=
by
  sorry

end solve_system_l175_17512


namespace impossibility_of_sum_sixteen_l175_17596

open Nat

def max_roll_value : ℕ := 6
def sum_of_two_rolls (a b : ℕ) : ℕ := a + b

theorem impossibility_of_sum_sixteen :
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ max_roll_value) ∧ (1 ≤ b ∧ b ≤ max_roll_value) → sum_of_two_rolls a b ≠ 16 :=
by
  intros a b h
  sorry

end impossibility_of_sum_sixteen_l175_17596


namespace ratio_is_one_to_two_l175_17519

def valentina_share_to_whole_ratio (valentina_share : ℕ) (whole_burger : ℕ) : ℕ × ℕ :=
  (valentina_share / (Nat.gcd valentina_share whole_burger), 
   whole_burger / (Nat.gcd valentina_share whole_burger))

theorem ratio_is_one_to_two : valentina_share_to_whole_ratio 6 12 = (1, 2) := 
  by
  sorry

end ratio_is_one_to_two_l175_17519


namespace digit_sum_equality_l175_17584

-- Definitions for the conditions
def is_permutation_of_digits (a b : ℕ) : Prop :=
  -- Assume implementation that checks if b is a permutation of the digits of a
  sorry

def sum_of_digits (n : ℕ) : ℕ :=
  -- Assume implementation that computes the sum of digits of n
  sorry

-- The theorem statement
theorem digit_sum_equality (a b : ℕ)
  (h : is_permutation_of_digits a b) :
  sum_of_digits (5 * a) = sum_of_digits (5 * b) :=
sorry

end digit_sum_equality_l175_17584


namespace train_speed_in_m_per_s_l175_17525

-- Define the given train speed in kmph
def train_speed_kmph : ℕ := 72

-- Define the conversion factor from kmph to m/s
def km_per_hour_to_m_per_second (speed_in_kmph : ℕ) : ℕ := (speed_in_kmph * 1000) / 3600

-- State the theorem
theorem train_speed_in_m_per_s (h : train_speed_kmph = 72) : km_per_hour_to_m_per_second train_speed_kmph = 20 := by
  sorry

end train_speed_in_m_per_s_l175_17525


namespace more_trees_died_than_survived_l175_17594

def haley_trees : ℕ := 14
def died_in_typhoon : ℕ := 9
def survived_trees := haley_trees - died_in_typhoon

theorem more_trees_died_than_survived : (died_in_typhoon - survived_trees) = 4 := by
  -- proof goes here
  sorry

end more_trees_died_than_survived_l175_17594


namespace michael_completes_in_50_days_l175_17520

theorem michael_completes_in_50_days :
  ∀ {M A W : ℝ},
    (W / M + W / A = W / 20) →
    (14 * W / 20 + 10 * W / A = W) →
    M = 50 :=
by
  sorry

end michael_completes_in_50_days_l175_17520


namespace inner_tetrahedron_volume_l175_17538

def volume_of_inner_tetrahedron(cube_side : ℕ) : ℚ :=
  let base_area := (cube_side * cube_side) / 2
  let height := cube_side
  let original_tetra_volume := (1 / 3) * base_area * height
  let inner_tetra_volume := original_tetra_volume / 8
  inner_tetra_volume

theorem inner_tetrahedron_volume {cube_side : ℕ} (h : cube_side = 2) : 
  volume_of_inner_tetrahedron cube_side = 1 / 6 := 
by
  rw [h]
  unfold volume_of_inner_tetrahedron 
  norm_num
  sorry

end inner_tetrahedron_volume_l175_17538


namespace statement_B_is_algorithm_l175_17599

def is_algorithm (statement : String) : Prop := 
  statement = "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."

def condition_A : String := "At home, it is generally the mother who cooks."
def condition_B : String := "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."
def condition_C : String := "Cooking outdoors is called a picnic."
def condition_D : String := "Rice is necessary for cooking."

theorem statement_B_is_algorithm : is_algorithm condition_B :=
by
  sorry

end statement_B_is_algorithm_l175_17599


namespace investment_period_l175_17598

theorem investment_period (x t : ℕ) (p_investment q_investment q_time : ℕ) (profit_ratio : ℚ):
  q_investment = 5 * x →
  p_investment = 7 * x →
  q_time = 16 →
  profit_ratio = 7 / 10 →
  7 * x * t = profit_ratio * 5 * x * q_time →
  t = 8 := sorry

end investment_period_l175_17598


namespace problem_statement_l175_17549

noncomputable def a := Real.sqrt 3 + Real.sqrt 2
noncomputable def b := Real.sqrt 3 - Real.sqrt 2
noncomputable def expression := a^(2 * Real.log (Real.sqrt 5) / Real.log b)

theorem problem_statement : expression = 1 / 5 := by
  sorry

end problem_statement_l175_17549


namespace geometric_sequence_product_proof_l175_17531

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_product_proof (a : ℕ → ℝ) (q : ℝ)
  (h_geo : geometric_sequence a q) 
  (h1 : a 2010 * a 2011 * a 2012 = 3)
  (h2 : a 2013 * a 2014 * a 2015 = 24) :
  a 2016 * a 2017 * a 2018 = 192 :=
sorry

end geometric_sequence_product_proof_l175_17531


namespace radius_of_circle_l175_17501

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l175_17501


namespace cylinder_lateral_area_l175_17552

-- Define the cylindrical lateral area calculation
noncomputable def lateral_area_of_cylinder (d h : ℝ) : ℝ := (2 * Real.pi * (d / 2)) * h

-- The statement of the problem in Lean 4.
theorem cylinder_lateral_area : lateral_area_of_cylinder 4 4 = 16 * Real.pi := by
  sorry

end cylinder_lateral_area_l175_17552


namespace gcd_4536_8721_l175_17524

theorem gcd_4536_8721 : Nat.gcd 4536 8721 = 3 := by
  sorry

end gcd_4536_8721_l175_17524


namespace coris_aunt_age_today_l175_17518

variable (Cori_age_now : ℕ) (age_diff : ℕ)

theorem coris_aunt_age_today (H1 : Cori_age_now = 3) (H2 : ∀ (Cori_age5 Aunt_age5 : ℕ), Cori_age5 = Cori_age_now + 5 → Aunt_age5 = 3 * Cori_age5 → Aunt_age5 - 5 = age_diff) :
  age_diff = 19 := 
by
  intros
  sorry

end coris_aunt_age_today_l175_17518


namespace percentage_of_total_l175_17554

theorem percentage_of_total (total part : ℕ) (h₁ : total = 100) (h₂ : part = 30):
  (part / total) * 100 = 30 := by
  sorry

end percentage_of_total_l175_17554


namespace find_k_and_angle_l175_17533

def vector := ℝ × ℝ

def dot_product (u v: vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def orthogonal (u v: vector) : Prop :=
  dot_product u v = 0

theorem find_k_and_angle (k : ℝ) :
  let a : vector := (3, -1)
  let b : vector := (1, k)
  orthogonal a b →
  (k = 3 ∧ dot_product (3+1, -1+3) (3-1, -1-3) = 0) :=
by
  intros
  sorry

end find_k_and_angle_l175_17533


namespace base_conversion_min_sum_l175_17522

theorem base_conversion_min_sum (c d : ℕ) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 := by
  sorry

end base_conversion_min_sum_l175_17522


namespace groupB_is_basis_l175_17530

section
variables (eA1 eA2 : ℝ × ℝ) (eB1 eB2 : ℝ × ℝ) (eC1 eC2 : ℝ × ℝ) (eD1 eD2 : ℝ × ℝ)

def is_collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k • w) ∨ w = (k • v)

-- Define each vector group
def groupA := eA1 = (0, 0) ∧ eA2 = (1, -2)
def groupB := eB1 = (-1, 2) ∧ eB2 = (5, 7)
def groupC := eC1 = (3, 5) ∧ eC2 = (6, 10)
def groupD := eD1 = (2, -3) ∧ eD2 = (1/2, -3/4)

-- The goal is to prove that group B vectors can serve as a basis
theorem groupB_is_basis : ¬ is_collinear eB1 eB2 :=
sorry
end

end groupB_is_basis_l175_17530


namespace harry_walks_9_dogs_on_thursday_l175_17575

-- Define the number of dogs Harry walks on specific days
def dogs_monday : Nat := 7
def dogs_wednesday : Nat := 7
def dogs_friday : Nat := 7
def dogs_tuesday : Nat := 12

-- Define the payment per dog
def payment_per_dog : Nat := 5

-- Define total weekly earnings
def total_weekly_earnings : Nat := 210

-- Define the number of dogs Harry walks on Thursday
def dogs_thursday : Nat := 9

-- Define the total earnings for Monday, Wednesday, Friday, and Tuesday
def earnings_first_four_days : Nat := (dogs_monday + dogs_wednesday + dogs_friday + dogs_tuesday) * payment_per_dog

-- Now we state the theorem that we need to prove
theorem harry_walks_9_dogs_on_thursday :
  (total_weekly_earnings - earnings_first_four_days) / payment_per_dog = dogs_thursday :=
by
  -- Proof omitted
  sorry

end harry_walks_9_dogs_on_thursday_l175_17575


namespace distinct_m_count_l175_17526

noncomputable def countDistinctMValues : Nat :=
  let pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), 
                (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  let ms := pairs.map (λ p => p.1 + p.2)
  ms.eraseDups.length

theorem distinct_m_count :
  countDistinctMValues = 10 := sorry

end distinct_m_count_l175_17526


namespace intersection_of_sets_l175_17568

theorem intersection_of_sets :
  let A := {1, 2}
  let B := {x : ℝ | x^2 - 3 * x + 2 = 0}
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_sets_l175_17568


namespace box_triple_count_l175_17507

theorem box_triple_count (a b c : ℕ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a * b * c = 2 * (a * b + b * c + c * a)) :
  (a = 2 ∧ b = 8 ∧ c = 8) ∨ (a = 3 ∧ b = 6 ∧ c = 6) ∨ (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = 5 ∧ b = 5 ∧ c = 5) ∨ (a = 6 ∧ b = 6 ∧ c = 6) :=
sorry

end box_triple_count_l175_17507


namespace total_number_of_students_l175_17597

theorem total_number_of_students 
    (group1 : Nat) (group2 : Nat) (group3 : Nat) (group4 : Nat) 
    (h1 : group1 = 5) (h2 : group2 = 8) (h3 : group3 = 7) (h4 : group4 = 4) : 
    group1 + group2 + group3 + group4 = 24 := 
by
  sorry

end total_number_of_students_l175_17597


namespace sqrt_defined_range_l175_17567

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 2)) → (x ≥ 2) := by
  sorry

end sqrt_defined_range_l175_17567


namespace fraction_absent_l175_17535

theorem fraction_absent (p : ℕ) (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : p * 1 = (1 - x) * p * 1.5) : x = 1 / 3 :=
by
  sorry

end fraction_absent_l175_17535


namespace two_card_draw_probability_l175_17582

open ProbabilityTheory

def card_values (card : ℕ) : ℕ :=
  if card = 1 ∨ card = 11 ∨ card = 12 ∨ card = 13 then 10 else card

def deck_size := 52

def total_prob : ℚ :=
  let cards := (1, deck_size)
  let case_1 := (card_values 6 * card_values 9 / (deck_size * (deck_size - 1))) + 
                (card_values 7 * card_values 8 / (deck_size * (deck_size - 1)))
  let case_2 := (3 * 4 / (deck_size * (deck_size - 1))) + 
                (4 * 3 / (deck_size * (deck_size - 1)))
  case_1 + case_2

theorem two_card_draw_probability :
  total_prob = 16 / 331 :=
by
  sorry

end two_card_draw_probability_l175_17582


namespace area_excluding_hole_l175_17576

open Polynomial

theorem area_excluding_hole (x : ℝ) : 
  ((x^2 + 7) * (x^2 + 5)) - ((2 * x^2 - 3) * (x^2 - 2)) = -x^4 + 19 * x^2 + 29 :=
by
  sorry

end area_excluding_hole_l175_17576


namespace total_young_fish_l175_17595

-- Define conditions
def tanks : ℕ := 3
def fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20

-- Define the main proof statement
theorem total_young_fish : tanks * fish_per_tank * young_per_fish = 240 := by
  sorry

end total_young_fish_l175_17595


namespace train_speed_l175_17504

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 300) (h_time : time = 15) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end train_speed_l175_17504


namespace jane_spent_75_days_reading_l175_17587

def pages : ℕ := 500
def speed_first_half : ℕ := 10
def speed_second_half : ℕ := 5

def book_reading_days (p s1 s2 : ℕ) : ℕ :=
  let half_pages := p / 2
  let days_first_half := half_pages / s1
  let days_second_half := half_pages / s2
  days_first_half + days_second_half

theorem jane_spent_75_days_reading :
  book_reading_days pages speed_first_half speed_second_half = 75 :=
by
  sorry

end jane_spent_75_days_reading_l175_17587


namespace maximum_value_of_w_l175_17515

variables (x y : ℝ)

def condition : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

def w (x y : ℝ) := 4 * x + 3 * y

theorem maximum_value_of_w : ∃ x y, condition x y ∧ w x y = 74 :=
sorry

end maximum_value_of_w_l175_17515


namespace solve_problem_l175_17505

theorem solve_problem (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  7^m - 3 * 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) := sorry

end solve_problem_l175_17505


namespace problem_solution_l175_17577

theorem problem_solution (k x1 x2 y1 y2 : ℝ) 
  (h₁ : k ≠ 0) 
  (h₂ : y1 = k * x1) 
  (h₃ : y1 = -5 / x1) 
  (h₄ : y2 = k * x2) 
  (h₅ : y2 = -5 / x2) 
  (h₆ : x1 = -x2) 
  (h₇ : y1 = -y2) : 
  x1 * y2 - 3 * x2 * y1 = 10 := 
sorry

end problem_solution_l175_17577


namespace number_of_days_woman_weaves_l175_17590

theorem number_of_days_woman_weaves
  (a_1 : ℝ) (a_n : ℝ) (S_n : ℝ) (n : ℝ)
  (h1 : a_1 = 5)
  (h2 : a_n = 1)
  (h3 : S_n = 90)
  (h4 : S_n = n * (a_1 + a_n) / 2) :
  n = 30 :=
by
  rw [h1, h2, h3] at h4
  sorry

end number_of_days_woman_weaves_l175_17590


namespace sum_of_r_s_l175_17559

theorem sum_of_r_s (m : ℝ) (x : ℝ) (y : ℝ) (r s : ℝ) 
  (parabola_eqn : y = x^2 + 4) 
  (point_Q : (x, y) = (10, 5)) 
  (roots_rs : ∀ (m : ℝ), m^2 - 40*m + 4 = 0 → r < m → m < s)
  : r + s = 40 := 
sorry

end sum_of_r_s_l175_17559


namespace pencils_given_out_l175_17537

theorem pencils_given_out
  (num_children : ℕ)
  (pencils_per_student : ℕ)
  (dozen : ℕ)
  (children : num_children = 46)
  (dozen_def : dozen = 12)
  (pencils_def : pencils_per_student = 4 * dozen) :
  num_children * pencils_per_student = 2208 :=
by {
  sorry
}

end pencils_given_out_l175_17537


namespace solve_equation_l175_17508

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_equation:
    (7.331 * ((log_base 3 x - 1) / (log_base 3 (x / 3))) - 
    2 * (log_base 3 (Real.sqrt x)) + (log_base 3 x)^2 = 3) → 
    (x = 1 / 3 ∨ x = 9) := by
  sorry

end solve_equation_l175_17508


namespace triangle_median_inequality_l175_17510

-- Defining the parameters and the inequality theorem.
theorem triangle_median_inequality
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (Δ : ℝ)
  (median_medians : ∀ {a b c : ℝ}, ma ≤ mb ∧ mb ≤ mc ∧ a ≥ b ∧ b ≥ c)  :
  a * (-ma + mb + mc) + b * (ma - mb + mc) + c * (ma + mb - mc) ≥ 6 * Δ := 
sorry

end triangle_median_inequality_l175_17510


namespace sin_neg_pi_div_two_l175_17574

theorem sin_neg_pi_div_two : Real.sin (-π / 2) = -1 := by
  -- Define the necessary conditions
  let π_in_deg : ℝ := 180 -- π radians equals 180 degrees
  have sin_neg_angle : ∀ θ : ℝ, Real.sin (-θ) = -Real.sin θ := sorry -- sin(-θ) = -sin(θ) for any θ
  have sin_90_deg : Real.sin (π_in_deg / 2) = 1 := sorry -- sin(90 degrees) = 1

  -- The main statement to prove
  sorry

end sin_neg_pi_div_two_l175_17574


namespace rectangle_area_coefficient_l175_17536

theorem rectangle_area_coefficient (length width d k : ℝ) 
(h1 : length / width = 5 / 2) 
(h2 : d^2 = length^2 + width^2) 
(h3 : k = 10 / 29) :
  (length * width = k * d^2) :=
by
  sorry

end rectangle_area_coefficient_l175_17536


namespace infinite_series_eq_15_l175_17570

theorem infinite_series_eq_15 (x : ℝ) :
  (∑' (n : ℕ), (5 + n * x) / 3^n) = 15 ↔ x = 10 :=
by
  sorry

end infinite_series_eq_15_l175_17570


namespace find_m_plus_c_l175_17546

-- We need to define the conditions first
variable {A : ℝ × ℝ} {B : ℝ × ℝ} {c : ℝ} {m : ℝ}

-- Given conditions from part a)
def A_def : Prop := A = (1, 3)
def B_def : Prop := B = (m, -1)
def centers_line : Prop := ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)

-- Define the theorem for the proof problem
theorem find_m_plus_c (A_def : A = (1, 3)) (B_def : B = (m, -1)) (centers_line : ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)) : m + c = 3 :=
sorry

end find_m_plus_c_l175_17546


namespace same_terminal_side_l175_17509

theorem same_terminal_side : ∃ k : ℤ, k * 360 - 60 = 300 := by
  sorry

end same_terminal_side_l175_17509


namespace volume_of_soil_removal_l175_17583

theorem volume_of_soil_removal {a b m c d : ℝ} :
  (∃ (K : ℝ), K = (m / 6) * (2 * a * c + 2 * b * d + a * d + b * c)) :=
sorry

end volume_of_soil_removal_l175_17583


namespace raisin_cost_fraction_l175_17548

theorem raisin_cost_fraction
  (R : ℝ)                -- cost of a pound of raisins
  (cost_nuts : ℝ := 2 * R)  -- cost of a pound of nuts
  (cost_raisins : ℝ := 3 * R)  -- cost of 3 pounds of raisins
  (cost_nuts_total : ℝ := 4 * cost_nuts)  -- cost of 4 pounds of nuts
  (total_cost : ℝ := cost_raisins + cost_nuts_total)  -- total cost of the mixture
  (fraction_of_raisins : ℝ := cost_raisins / total_cost)  -- fraction of cost of raisins
  : fraction_of_raisins = 3 / 11 := 
by
  sorry

end raisin_cost_fraction_l175_17548


namespace friends_travelled_distance_l175_17579

theorem friends_travelled_distance :
  let lionel_distance : ℝ := 4 * 5280
  let esther_distance : ℝ := 975 * 3
  let niklaus_distance : ℝ := 1287
  let isabella_distance : ℝ := 18 * 1000 * 3.28084
  let sebastian_distance : ℝ := 2400 * 3.28084
  let total_distance := lionel_distance + esther_distance + niklaus_distance + isabella_distance + sebastian_distance
  total_distance = 91261.136 := 
by
  sorry

end friends_travelled_distance_l175_17579


namespace find_smallest_natural_number_l175_17527

theorem find_smallest_natural_number :
  ∃ x : ℕ, (2 * x = b^2 ∧ 3 * x = c^3) ∧ (∀ y : ℕ, (2 * y = d^2 ∧ 3 * y = e^3) → x ≤ y) := by
  sorry

end find_smallest_natural_number_l175_17527


namespace amount_each_girl_receives_l175_17534

theorem amount_each_girl_receives (total_amount : ℕ) (total_children : ℕ) (amount_per_boy : ℕ) (num_boys : ℕ) (remaining_amount : ℕ) (num_girls : ℕ) (amount_per_girl : ℕ) 
  (h1 : total_amount = 460) 
  (h2 : total_children = 41)
  (h3 : amount_per_boy = 12)
  (h4 : num_boys = 33)
  (h5 : remaining_amount = total_amount - num_boys * amount_per_boy)
  (h6 : num_girls = total_children - num_boys)
  (h7 : amount_per_girl = remaining_amount / num_girls) :
  amount_per_girl = 8 := 
sorry

end amount_each_girl_receives_l175_17534


namespace am_gm_inequality_l175_17569

theorem am_gm_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - (Real.sqrt (a * b)) ∧ 
  (a + b) / 2 - (Real.sqrt (a * b)) < (a - b)^2 / (8 * b) := 
sorry

end am_gm_inequality_l175_17569


namespace intersection_eq_l175_17550

open Set

-- Define the sets A and B according to the given conditions
def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | (x - 1) * (x + 2) < 0}

-- Define the intended intersection result
def C : Set ℤ := {-1, 0}

-- The theorem to prove
theorem intersection_eq : A ∩ {x | (x - 1) * (x + 2) < 0} = C := by
  sorry

end intersection_eq_l175_17550


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l175_17528

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l175_17528


namespace min_packs_needed_l175_17500

-- Define pack sizes
def pack_sizes : List ℕ := [6, 12, 24, 30]

-- Define the total number of cans needed
def total_cans : ℕ := 150

-- Define the minimum number of packs needed to buy exactly 150 cans of soda
theorem min_packs_needed : ∃ packs : List ℕ, (∀ p ∈ packs, p ∈ pack_sizes) ∧ List.sum packs = total_cans ∧ packs.length = 5 := by
  sorry

end min_packs_needed_l175_17500


namespace largest_n_satisfying_conditions_l175_17540

theorem largest_n_satisfying_conditions :
  ∃ n : ℤ, n = 181 ∧
    (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧
    ∃ k : ℤ, 2 * n + 79 = k^2 :=
by
  sorry

end largest_n_satisfying_conditions_l175_17540


namespace possible_triplets_l175_17556

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem possible_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (is_power_of_two (a * b - c) ∧ is_power_of_two (b * c - a) ∧ is_power_of_two (c * a - b)) ↔ 
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) :=
by
  sorry

end possible_triplets_l175_17556


namespace set_C_cannot_form_right_triangle_l175_17516

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem set_C_cannot_form_right_triangle :
  ¬ is_right_triangle 7 8 9 :=
by
  sorry

end set_C_cannot_form_right_triangle_l175_17516


namespace rope_length_l175_17553

theorem rope_length (x : ℝ) 
  (h : 10^2 + (x - 4)^2 = x^2) : 
  x = 14.5 :=
sorry

end rope_length_l175_17553


namespace not_divisible_by_n_l175_17592

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬ (n ∣ (2^n - 1)) :=
by
  sorry

end not_divisible_by_n_l175_17592


namespace part1_part2_l175_17502

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : 2 ≤ a ↔ ∀ (x : ℝ), f x a + g x ≥ 3 := by
  sorry

end part1_part2_l175_17502


namespace problem_statement_l175_17545

theorem problem_statement (n k : ℕ) (h1 : n = 2^2007 * k + 1) (h2 : k % 2 = 1) : ¬ n ∣ 2^(n-1) + 1 := by
  sorry

end problem_statement_l175_17545


namespace area_of_triangle_ABC_l175_17589

noncomputable def distance (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem area_of_triangle_ABC (A B C O : ℝ × ℝ)
  (h_isosceles_right : ∃ d: ℝ, distance A B = d ∧ distance A C = d ∧ distance B C = Real.sqrt (2 * d^2))
  (h_A_right : A = (0, 0))
  (h_OA : distance O A = 5)
  (h_OB : distance O B = 7)
  (h_OC : distance O C = 3) :
  ∃ S : ℝ, S = (29 / 2) + (5 / 2) * Real.sqrt 17 :=
sorry

end area_of_triangle_ABC_l175_17589


namespace converse_prop_inverse_prop_contrapositive_prop_l175_17557

-- Given condition: the original proposition is true
axiom original_prop : ∀ (x y : ℝ), x * y = 0 → x = 0 ∨ y = 0

-- Converse: If x=0 or y=0, then xy=0 - prove this is true
theorem converse_prop (x y : ℝ) : (x = 0 ∨ y = 0) → x * y = 0 :=
by
  sorry

-- Inverse: If xy ≠ 0, then x ≠ 0 and y ≠ 0 - prove this is true
theorem inverse_prop (x y : ℝ) : x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0 :=
by
  sorry

-- Contrapositive: If x ≠ 0 and y ≠ 0, then xy ≠ 0 - prove this is true
theorem contrapositive_prop (x y : ℝ) : (x ≠ 0 ∧ y ≠ 0) → x * y ≠ 0 :=
by
  sorry

end converse_prop_inverse_prop_contrapositive_prop_l175_17557


namespace quadratic_no_real_solution_l175_17562

theorem quadratic_no_real_solution (a : ℝ) : (∀ x : ℝ, x^2 - x + a ≠ 0) → a > 1 / 4 :=
by
  intro h
  sorry

end quadratic_no_real_solution_l175_17562


namespace candy_mixture_solution_l175_17585

theorem candy_mixture_solution :
  ∃ x y : ℝ, 18 * x + 10 * y = 1500 ∧ x + y = 100 ∧ x = 62.5 ∧ y = 37.5 := by
  sorry

end candy_mixture_solution_l175_17585


namespace students_in_both_band_and_chorus_l175_17542

-- Definitions of conditions
def total_students := 250
def band_students := 90
def chorus_students := 120
def band_or_chorus_students := 180

-- Theorem statement to prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : 
  (band_students + chorus_students - band_or_chorus_students) = 30 := 
by sorry

end students_in_both_band_and_chorus_l175_17542


namespace sin_alpha_cos_half_beta_minus_alpha_l175_17551

open Real

noncomputable def problem_condition (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  sin (π / 3 - α) = 3 / 5 ∧
  cos (β / 2 - π / 3) = 2 * sqrt 5 / 5

theorem sin_alpha (α β : ℝ) (h : problem_condition α β) : 
  sin α = (4 * sqrt 3 - 3) / 10 := sorry

theorem cos_half_beta_minus_alpha (α β : ℝ) (h : problem_condition α β) :
  cos (β / 2 - α) = 11 * sqrt 5 / 25 := sorry

end sin_alpha_cos_half_beta_minus_alpha_l175_17551


namespace find_N_l175_17506

-- Definition of the conditions
def is_largest_divisor_smaller_than (m N : ℕ) : Prop := m < N ∧ Nat.gcd m N = m

def produces_power_of_ten (N m : ℕ) : Prop := ∃ k : ℕ, k > 0 ∧ N + m = 10^k

-- Final statement to prove
theorem find_N (N : ℕ) : (∃ m : ℕ, is_largest_divisor_smaller_than m N ∧ produces_power_of_ten N m) → N = 75 :=
by
  sorry

end find_N_l175_17506


namespace find_side_b_in_triangle_l175_17560

noncomputable def triangle_side_b (a A : ℝ) (cosB : ℝ) : ℝ :=
  let sinB := Real.sqrt (1 - cosB^2)
  let sinA := Real.sin A
  (a * sinB) / sinA

theorem find_side_b_in_triangle :
  triangle_side_b 5 (Real.pi / 4) (3 / 5) = 4 * Real.sqrt 2 :=
by
  sorry

end find_side_b_in_triangle_l175_17560


namespace average_sweater_less_by_21_after_discount_l175_17563

theorem average_sweater_less_by_21_after_discount
  (shirt_count sweater_count jeans_count : ℕ)
  (total_shirt_price total_sweater_price total_jeans_price : ℕ)
  (shirt_discount sweater_discount jeans_discount : ℕ)
  (shirt_avg_before_discount sweater_avg_before_discount jeans_avg_before_discount 
   shirt_avg_after_discount sweater_avg_after_discount jeans_avg_after_discount : ℕ) :
  shirt_count = 20 →
  sweater_count = 45 →
  jeans_count = 30 →
  total_shirt_price = 360 →
  total_sweater_price = 900 →
  total_jeans_price = 1200 →
  shirt_discount = 2 →
  sweater_discount = 4 →
  jeans_discount = 3 →
  shirt_avg_before_discount = total_shirt_price / shirt_count →
  sweater_avg_before_discount = total_sweater_price / sweater_count →
  jeans_avg_before_discount = total_jeans_price / jeans_count →
  shirt_avg_after_discount = shirt_avg_before_discount - shirt_discount →
  sweater_avg_after_discount = sweater_avg_before_discount - sweater_discount →
  jeans_avg_after_discount = jeans_avg_before_discount - jeans_discount →
  sweater_avg_after_discount = shirt_avg_after_discount →
  jeans_avg_after_discount - sweater_avg_after_discount = 21 :=
by
  intros
  sorry

end average_sweater_less_by_21_after_discount_l175_17563


namespace cone_base_radius_l175_17586

noncomputable def sector_radius : ℝ := 9
noncomputable def central_angle_deg : ℝ := 240

theorem cone_base_radius :
  let arc_length := (central_angle_deg * Real.pi * sector_radius) / 180
  let base_circumference := arc_length
  let base_radius := base_circumference / (2 * Real.pi)
  base_radius = 6 :=
by
  sorry

end cone_base_radius_l175_17586


namespace sin_double_angle_of_tan_l175_17578

-- Given condition: tan(alpha) = 2
-- To prove: sin(2 * alpha) = 4/5
theorem sin_double_angle_of_tan (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
  sorry

end sin_double_angle_of_tan_l175_17578


namespace quadratic_passes_through_l175_17580

def quadratic_value_at_point (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_passes_through (a b c : ℝ) :
  quadratic_value_at_point a b c 1 = 5 ∧ 
  quadratic_value_at_point a b c 3 = n ∧ 
  a * (-2)^2 + b * (-2) + c = -8 ∧ 
  (-4*a + b = 0) → 
  n = 253/9 := 
sorry

end quadratic_passes_through_l175_17580


namespace lcm_multiplied_by_2_is_72x_l175_17513

-- Define the denominators
def denom1 (x : ℕ) := 4 * x
def denom2 (x : ℕ) := 6 * x
def denom3 (x : ℕ) := 9 * x

-- Define the least common multiple of three natural numbers
def lcm_three (a b c : ℕ) := Nat.lcm a (Nat.lcm b c)

-- Define the multiplication by 2
def multiply_by_2 (n : ℕ) := 2 * n

-- Define the final result
def final_result (x : ℕ) := 72 * x

-- The proof statement
theorem lcm_multiplied_by_2_is_72x (x : ℕ): 
  multiply_by_2 (lcm_three (denom1 x) (denom2 x) (denom3 x)) = final_result x := 
by
  sorry

end lcm_multiplied_by_2_is_72x_l175_17513


namespace arithmetic_sequence_problem_l175_17517

-- Define the arithmetic sequence and related sum functions
def a_n (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

def S (a1 d : ℤ) (n : ℕ) : ℤ :=
  (a1 + a_n a1 d n) * n / 2

-- Problem statement: proving a_5 = -1 given the conditions
theorem arithmetic_sequence_problem :
  (∃ (a1 d : ℕ), S a1 d 2 = S a1 d 6 ∧ a_n a1 d 4 = 1) → a_n a1 d 5 = -1 :=
by
  -- Assume the statement and then skip the proof
  sorry

end arithmetic_sequence_problem_l175_17517


namespace diagonals_in_octagon_l175_17511

/-- The formula to calculate the number of diagonals in a polygon -/
def number_of_diagonals (n : Nat) : Nat :=
  (n * (n - 3)) / 2

/-- The number of sides in an octagon -/
def sides_of_octagon : Nat := 8

/-- The number of diagonals in an octagon is 20. -/
theorem diagonals_in_octagon : number_of_diagonals sides_of_octagon = 20 :=
by
  sorry

end diagonals_in_octagon_l175_17511


namespace hyperbola_eccentricity_l175_17564

theorem hyperbola_eccentricity (a b c : ℝ) (h : (c^2 - a^2 = 5 * a^2)) (hb : a / b = 2) :
  (c / a = Real.sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_l175_17564
