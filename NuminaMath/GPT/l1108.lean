import Mathlib

namespace maximize_profit_l1108_110827

noncomputable section

-- Definitions of parameters
def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 200
def daily_cost : ℝ := 450
def price_min : ℝ := 30
def price_max : ℝ := 60

-- Function for daily profit
def daily_profit (x : ℝ) : ℝ := (x - 30) * daily_sales_volume x - daily_cost

-- Theorem statement
theorem maximize_profit :
  let max_profit_price := 60
  let max_profit_value := 1950
  30 ≤ max_profit_price ∧ max_profit_price ≤ 60 ∧
  daily_profit max_profit_price = max_profit_value :=
by
  sorry

end maximize_profit_l1108_110827


namespace highest_number_paper_l1108_110879

theorem highest_number_paper
  (n : ℕ)
  (P : ℝ)
  (hP : P = 0.010309278350515464)
  (hP_formula : 1 / n = P) :
  n = 97 :=
by
  -- Placeholder for proof
  sorry

end highest_number_paper_l1108_110879


namespace m_n_solution_l1108_110878

theorem m_n_solution (m n : ℝ) (h1 : m - n = -5) (h2 : m^2 + n^2 = 13) : m^4 + n^4 = 97 :=
by
  sorry

end m_n_solution_l1108_110878


namespace rose_bushes_in_park_l1108_110887

theorem rose_bushes_in_park (current_rose_bushes total_new_rose_bushes total_rose_bushes : ℕ) 
(h1 : total_new_rose_bushes = 4)
(h2 : total_rose_bushes = 6) :
current_rose_bushes + total_new_rose_bushes = total_rose_bushes → current_rose_bushes = 2 := 
by 
  sorry

end rose_bushes_in_park_l1108_110887


namespace value_of_m_l1108_110884

noncomputable def TV_sales_volume_function (x : ℕ) : ℚ :=
  10 * x + 540

theorem value_of_m : ∀ (m : ℚ),
  (3200 * (1 + m / 100) * 9 / 10) * (600 * (1 - 2 * m / 100) + 220) = 3200 * 600 * (1 + 15.5 / 100) →
  m = 10 :=
by sorry

end value_of_m_l1108_110884


namespace k_gonal_number_proof_l1108_110809

-- Definitions for specific k-gonal numbers based on given conditions.
def triangular_number (n : ℕ) := (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
def square_number (n : ℕ) := n^2
def pentagonal_number (n : ℕ) := (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
def hexagonal_number (n : ℕ) := 2 * n^2 - n

-- General definition for the k-gonal number
def k_gonal_number (n k : ℕ) : ℚ := ((k - 2) / 2) * n^2 + ((4 - k) / 2) * n

-- Corresponding Lean statement for the proof problem
theorem k_gonal_number_proof (n k : ℕ) (hk : k ≥ 3) :
    (k = 3 -> triangular_number n = k_gonal_number n k) ∧
    (k = 4 -> square_number n = k_gonal_number n k) ∧
    (k = 5 -> pentagonal_number n = k_gonal_number n k) ∧
    (k = 6 -> hexagonal_number n = k_gonal_number n k) ∧
    (n = 10 ∧ k = 24 -> k_gonal_number n k = 1000) :=
by
  intros
  sorry

end k_gonal_number_proof_l1108_110809


namespace probability_is_two_thirds_l1108_110810

noncomputable def probabilityOfEvent : ℚ :=
  let Ω := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 }
  let A := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 ∧ 2 * p.1 - p.2 + 2 ≥ 0 }
  let area_Ω := (2 - 0) * (6 - 0)
  let area_A := area_Ω - (1 / 2) * 2 * 4
  (area_A / area_Ω : ℚ)

theorem probability_is_two_thirds : probabilityOfEvent = (2 / 3 : ℚ) :=
  sorry

end probability_is_two_thirds_l1108_110810


namespace drawing_red_ball_is_certain_l1108_110853

def certain_event (balls : List String) : Prop :=
  ∀ ball ∈ balls, ball = "red"

theorem drawing_red_ball_is_certain:
  certain_event ["red", "red", "red", "red", "red"] :=
by
  sorry

end drawing_red_ball_is_certain_l1108_110853


namespace inequality_x2_y2_l1108_110820

theorem inequality_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  |x^2 + y^2| / (x + y) < |x^2 - y^2| / (x - y) :=
sorry

end inequality_x2_y2_l1108_110820


namespace probability_alpha_in_interval_l1108_110864

def vector_of_die_rolls_angle_probability : ℚ := 
  let total_outcomes := 36
  let favorable_pairs := 15
  favorable_pairs / total_outcomes

theorem probability_alpha_in_interval (m n : ℕ)
  (hm : 1 ≤ m ∧ m ≤ 6) (hn : 1 ≤ n ∧ n ≤ 6) :
  (vector_of_die_rolls_angle_probability = 5 / 12) := by
  sorry

end probability_alpha_in_interval_l1108_110864


namespace john_initial_clean_jerk_weight_l1108_110861

def initial_snatch_weight : ℝ := 50
def increase_rate : ℝ := 1.8
def total_new_lifting_capacity : ℝ := 250

theorem john_initial_clean_jerk_weight :
  ∃ (C : ℝ), 2 * C + (increase_rate * initial_snatch_weight) = total_new_lifting_capacity ∧ C = 80 := by
  sorry

end john_initial_clean_jerk_weight_l1108_110861


namespace inequality1_inequality2_l1108_110858

theorem inequality1 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + a * b * c ≥ 2 * Real.sqrt 3 :=
by
  sorry

theorem inequality2 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  (1 / A) + (1 / B) + (1 / C) ≥ 9 / Real.pi :=
by
  sorry

end inequality1_inequality2_l1108_110858


namespace amy_red_balloons_l1108_110874

theorem amy_red_balloons (total_balloons green_balloons blue_balloons : ℕ) (h₁ : total_balloons = 67) (h₂: green_balloons = 17) (h₃ : blue_balloons = 21) : (total_balloons - (green_balloons + blue_balloons)) = 29 :=
by
  sorry

end amy_red_balloons_l1108_110874


namespace train_departure_time_l1108_110805

-- Conditions
def arrival_time : ℕ := 1000  -- Representing 10:00 as 1000 (in minutes since midnight)
def travel_time : ℕ := 15  -- 15 minutes

-- Definition of time subtraction
def time_sub (arrival : ℕ) (travel : ℕ) : ℕ :=
arrival - travel

-- Proof that the train left at 9:45
theorem train_departure_time : time_sub arrival_time travel_time = 945 := by
  sorry

end train_departure_time_l1108_110805


namespace radius_of_ball_l1108_110870

theorem radius_of_ball (diameter depth : ℝ) (h₁ : diameter = 30) (h₂ : depth = 10) : 
  ∃ r : ℝ, r = 25 :=
by
  sorry

end radius_of_ball_l1108_110870


namespace fly_in_box_maximum_path_length_l1108_110819

theorem fly_in_box_maximum_path_length :
  let side1 := 1
  let side2 := Real.sqrt 2
  let side3 := Real.sqrt 3
  let space_diagonal := Real.sqrt (side1^2 + side2^2 + side3^2)
  let face_diagonal1 := Real.sqrt (side1^2 + side2^2)
  let face_diagonal2 := Real.sqrt (side1^2 + side3^2)
  let face_diagonal3 := Real.sqrt (side2^2 + side3^2)
  (4 * space_diagonal + 2 * face_diagonal3) = 4 * Real.sqrt 6 + 2 * Real.sqrt 5 :=
by
  sorry

end fly_in_box_maximum_path_length_l1108_110819


namespace graveling_cost_l1108_110847

def lawn_length : ℝ := 110
def lawn_breadth: ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 3

def road_1_area : ℝ := lawn_length * road_width
def intersecting_length : ℝ := lawn_breadth - road_width
def road_2_area : ℝ := intersecting_length * road_width
def total_area : ℝ := road_1_area + road_2_area
def total_cost : ℝ := total_area * cost_per_sq_meter

theorem graveling_cost :
  total_cost = 4800 := 
  by
    sorry

end graveling_cost_l1108_110847


namespace reflected_line_equation_l1108_110899

def line_reflection_about_x_axis (x y : ℝ) : Prop :=
  x - y + 1 = 0 → y = -x - 1

theorem reflected_line_equation :
  ∀ (x y : ℝ), x - y + 1 = 0 → x + y + 1 = 0 :=
by
  intros x y h
  suffices y = -x - 1 by
    linarith
  sorry

end reflected_line_equation_l1108_110899


namespace b_20_value_l1108_110859

noncomputable def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => b (n+1) * b n

theorem b_20_value : b 19 = 2^4181 :=
sorry

end b_20_value_l1108_110859


namespace trigonometric_identity_l1108_110882

theorem trigonometric_identity :
  Real.tan (70 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) * (Real.sqrt 3 * Real.tan (20 * Real.pi / 180) - 1) = -1 :=
by
  sorry

end trigonometric_identity_l1108_110882


namespace quadratic_has_two_distinct_real_roots_iff_l1108_110845

theorem quadratic_has_two_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6 * x - m = 0 ∧ y^2 - 6 * y - m = 0) ↔ m > -9 :=
by 
  sorry

end quadratic_has_two_distinct_real_roots_iff_l1108_110845


namespace find_angle_A_l1108_110855

noncomputable def exists_angle_A (A B C : ℝ) (a b : ℝ) : Prop :=
  C = (A + B) / 2 ∧ 
  A + B + C = 180 ∧ 
  (a + b) / 2 = Real.sqrt 3 + 1 ∧ 
  C = 2 * Real.sqrt 2

theorem find_angle_A : ∃ A B C a b, 
  exists_angle_A A B C a b ∧ (A = 75 ∨ A = 45) :=
by
  -- This is where the detailed proof would go
  sorry

end find_angle_A_l1108_110855


namespace james_total_earnings_l1108_110801

def january_earnings : ℕ := 4000
def february_earnings : ℕ := january_earnings + (50 * january_earnings / 100)
def march_earnings : ℕ := february_earnings - (20 * february_earnings / 100)
def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings :
  total_earnings = 14800 :=
by
  -- skip the proof
  sorry

end james_total_earnings_l1108_110801


namespace david_pushups_more_than_zachary_l1108_110869

theorem david_pushups_more_than_zachary :
  ∀ (Z D J : ℕ), Z = 51 → J = 69 → J = D - 4 → D = Z + 22 :=
by
  intros Z D J hZ hJ hJD
  sorry

end david_pushups_more_than_zachary_l1108_110869


namespace solve_for_x_l1108_110802

theorem solve_for_x : ∃ x : ℚ, 5 * (2 * x - 3) = 3 * (3 - 4 * x) + 15 ∧ x = (39 : ℚ) / 22 :=
by
  use (39 : ℚ) / 22
  sorry

end solve_for_x_l1108_110802


namespace emily_beads_l1108_110836

theorem emily_beads (n : ℕ) (b : ℕ) (total_beads : ℕ) (h1 : n = 26) (h2 : b = 2) (h3 : total_beads = n * b) : total_beads = 52 :=
by
  sorry

end emily_beads_l1108_110836


namespace yolanda_walking_rate_l1108_110860

-- Definitions for the conditions given in the problem
variables (X Y : ℝ) -- Points X and Y
def distance_X_to_Y := 52 -- Distance between X and Y in miles
def Bob_rate := 4 -- Bob's walking rate in miles per hour
def Bob_distance_walked := 28 -- The distance Bob walked in miles
def start_time_diff := 1 -- The time difference (in hours) between Yolanda and Bob starting

-- The statement to prove
theorem yolanda_walking_rate : 
  ∃ (y : ℝ), (distance_X_to_Y = y * (Bob_distance_walked / Bob_rate + start_time_diff) + Bob_distance_walked) ∧ y = 3 := by 
  sorry

end yolanda_walking_rate_l1108_110860


namespace domain_of_f_l1108_110818

def domain_of_log_func := Set ℝ

def is_valid (x : ℝ) : Prop := x - 1 > 0

def func_domain (f : ℝ → ℝ) : domain_of_log_func := {x : ℝ | is_valid x}

theorem domain_of_f :
  func_domain (λ x => Real.log (x - 1)) = {x : ℝ | 1 < x} := by
  sorry

end domain_of_f_l1108_110818


namespace part1_solution_part2_solution_l1108_110865

-- Definitions of the lines
def l1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a + 1, a + 2, 3)

def l2 (a : ℝ) : ℝ × ℝ × ℝ :=
  (a - 1, -2, 2)

-- Parallel lines condition
def parallel_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 / B1) = (A2 / B2)

-- Perpendicular lines condition
def perpendicular_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 * A2 + B1 * B2 = 0)

-- Statement for part 1
theorem part1_solution (a : ℝ) : parallel_lines a ↔ a = 0 :=
  sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) : perpendicular_lines a ↔ (a = -1 ∨ a = 5 / 2) :=
  sorry


end part1_solution_part2_solution_l1108_110865


namespace angle_between_vectors_45_degrees_l1108_110890

open Real

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_mag (v : ℝ × ℝ) : ℝ := sqrt (vec_dot v v)

noncomputable def vec_angle (v w : ℝ × ℝ) : ℝ := arccos (vec_dot v w / (vec_mag v * vec_mag w))

theorem angle_between_vectors_45_degrees 
  (e1 e2 : ℝ × ℝ)
  (h1 : vec_mag e1 = 1)
  (h2 : vec_mag e2 = 1)
  (h3 : vec_dot e1 e2 = 0)
  (a : ℝ × ℝ := (3, 0) - (0, 1))  -- (3 * e1 - e2) is represented in a direct vector form (3, -1)
  (b : ℝ × ℝ := (2, 0) + (0, 1)): -- (2 * e1 + e2) is represented in a direct vector form (2, 1)
  vec_angle a b = π / 4 :=  -- π / 4 radians is equivalent to 45 degrees
sorry

end angle_between_vectors_45_degrees_l1108_110890


namespace x_squared_minus_y_squared_l1108_110822

theorem x_squared_minus_y_squared
    (x y : ℚ) 
    (h1 : x + y = 3 / 8) 
    (h2 : x - y = 1 / 4) : x^2 - y^2 = 3 / 32 := 
by 
    sorry

end x_squared_minus_y_squared_l1108_110822


namespace remainder_of_3024_l1108_110846

theorem remainder_of_3024 (M : ℤ) (hM1 : M = 3024) (h_condition : ∃ k : ℤ, M = 24 * k + 13) :
  M % 1821 = 1203 :=
by
  sorry

end remainder_of_3024_l1108_110846


namespace intersection_is_correct_l1108_110808

noncomputable def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x < 4}

theorem intersection_is_correct : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} :=
sorry

end intersection_is_correct_l1108_110808


namespace find_measure_angle_AOD_l1108_110876

-- Definitions of angles in the problem
def angle_COA := 150
def angle_BOD := 120

-- Definition of the relationship between angles
def angle_AOD_eq_four_times_angle_BOC (x : ℝ) : Prop :=
  4 * x = 360

-- Proof Problem Lean Statement
theorem find_measure_angle_AOD (x : ℝ) (h1 : 180 - 30 = angle_COA) (h2 : 180 - 60 = angle_BOD) (h3 : angle_AOD_eq_four_times_angle_BOC x) : 
  4 * x = 360 :=
  by 
  -- Insert necessary steps here
  sorry

end find_measure_angle_AOD_l1108_110876


namespace domain_of_f_l1108_110800

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 12))

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ (x ≠ 15 / 2) :=
by
  sorry

end domain_of_f_l1108_110800


namespace complement_of_P_union_Q_in_Z_is_M_l1108_110852

-- Definitions of the sets M, P, Q
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

-- Theorem statement
theorem complement_of_P_union_Q_in_Z_is_M : (Set.univ \ (P ∪ Q)) = M :=
by 
  sorry

end complement_of_P_union_Q_in_Z_is_M_l1108_110852


namespace correct_statement_is_D_l1108_110866

/-
Given the following statements and their conditions:
A: Conducting a comprehensive survey is not an accurate approach to understand the sleep situation of middle school students in Changsha.
B: The mode of the dataset \(-1\), \(2\), \(5\), \(5\), \(7\), \(7\), \(4\) is not \(7\) only, because both \(5\) and \(7\) are modes.
C: A probability of precipitation of \(90\%\) does not guarantee it will rain tomorrow.
D: If two datasets, A and B, have the same mean, and the variances \(s_{A}^{2} = 0.3\) and \(s_{B}^{2} = 0.02\), then set B with a lower variance \(s_{B}^{2}\) is more stable.

Prove that the correct statement based on these conditions is D.
-/
theorem correct_statement_is_D
  (dataset_A dataset_B : Type)
  (mean_A mean_B : ℝ)
  (sA2 sB2 : ℝ)
  (h_same_mean: mean_A = mean_B)
  (h_variances: sA2 = 0.3 ∧ sB2 = 0.02)
  (h_stability: sA2 > sB2) :
  (if sA2 = 0.3 ∧ sB2 = 0.02 ∧ sA2 > sB2 then "D" else "not D") = "D" := by
  sorry

end correct_statement_is_D_l1108_110866


namespace find_volume_of_pure_alcohol_l1108_110834

variable (V1 Vf V2 : ℝ)
variable (P1 Pf : ℝ)

theorem find_volume_of_pure_alcohol
  (h : V2 = Vf * Pf / 100 - V1 * P1 / 100) : 
  V2 = Vf * (Pf / 100) - V1 * (P1 / 100) :=
by
  -- This is the theorem statement. The proof is omitted.
  sorry

end find_volume_of_pure_alcohol_l1108_110834


namespace cylinder_surface_area_l1108_110832

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (r_radius : r = 4) : 
  2 * π * r * (r + h) = 128 * π :=
by
  -- providing the proof steps is beyond the scope of this task
  sorry

end cylinder_surface_area_l1108_110832


namespace value_of_m_l1108_110806

-- Problem Statement
theorem value_of_m (m : ℝ) : (∃ x : ℝ, (m-2)*x^(|m|-1) + 16 = 0 ∧ |m| - 1 = 1) → m = -2 :=
by
  sorry

end value_of_m_l1108_110806


namespace find_d_l1108_110875

theorem find_d (
  x : ℝ
) (
  h1 : 3 * x + 8 = 5
) (
  d : ℝ
) (
  h2 : d * x - 15 = -7
) : d = -8 :=
by
  sorry

end find_d_l1108_110875


namespace log7_18_l1108_110877

theorem log7_18 (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) : 
  Real.log 18 / Real.log 7 = (a + 2 * b) / (1 - a) :=
by
  -- proof to be completed
  sorry

end log7_18_l1108_110877


namespace sum_first_60_natural_numbers_l1108_110898

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end sum_first_60_natural_numbers_l1108_110898


namespace function_eq_l1108_110862

noncomputable def f (x : ℝ) : ℝ := x^4 - 2

theorem function_eq (f : ℝ → ℝ) (h1 : ∀ x : ℝ, deriv f x = 4 * x^3) (h2 : f 1 = -1) : 
  ∀ x : ℝ, f x = x^4 - 2 :=
by
  intro x
  -- Proof omitted
  sorry

end function_eq_l1108_110862


namespace unique_integral_root_l1108_110823

theorem unique_integral_root {x : ℤ} :
  x - 12 / (x - 3) = 5 - 12 / (x - 3) ↔ x = 5 :=
by
  sorry

end unique_integral_root_l1108_110823


namespace decreasing_omega_range_l1108_110871

open Real

theorem decreasing_omega_range {ω : ℝ} (h1 : 1 < ω) :
  (∀ x y : ℝ, π ≤ x ∧ x ≤ y ∧ y ≤ (5 * π) / 4 → 
    (|sin (ω * y + π / 3)| ≤ |sin (ω * x + π / 3)|)) → 
  (7 / 6 ≤ ω ∧ ω ≤ 4 / 3) :=
by
  sorry

end decreasing_omega_range_l1108_110871


namespace maximize_sum_l1108_110854

def a_n (n : ℕ): ℤ := 11 - 2 * (n - 1)

theorem maximize_sum (n : ℕ) (S : ℕ → ℤ → Prop) :
  (∀ n, S n (a_n n)) → (a_n n ≥ 0) → n = 6 :=
by
  sorry

end maximize_sum_l1108_110854


namespace find_x_value_l1108_110883

theorem find_x_value (x : ℝ) (h : (55 + 113 / x) * x = 4403) : x = 78 :=
sorry

end find_x_value_l1108_110883


namespace runway_show_time_correct_l1108_110838

def runwayShowTime (bathing_suit_sets evening_wear_sets formal_wear_sets models trip_time_in_minutes : ℕ) : ℕ :=
  let trips_per_model := bathing_suit_sets + evening_wear_sets + formal_wear_sets
  let total_trips := models * trips_per_model
  total_trips * trip_time_in_minutes

theorem runway_show_time_correct :
  runwayShowTime 3 4 2 10 3 = 270 :=
by
  sorry

end runway_show_time_correct_l1108_110838


namespace t_n_closed_form_t_2022_last_digit_l1108_110835

noncomputable def t_n (n : ℕ) : ℕ :=
  (4^n - 3 * 3^n + 3 * 2^n - 1) / 6

theorem t_n_closed_form (n : ℕ) (hn : 0 < n) :
  t_n n = (4^n - 3 * 3^n + 3 * 2^n - 1) / 6 :=
by
  sorry

theorem t_2022_last_digit :
  (t_n 2022) % 10 = 1 :=
by
  sorry

end t_n_closed_form_t_2022_last_digit_l1108_110835


namespace min_value_abs_sum_pqr_inequality_l1108_110885

theorem min_value_abs_sum (x : ℝ) : |x + 1| + |x - 2| ≥ 3 :=
by
  sorry

theorem pqr_inequality (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := 
by
  have f_min : ∀ x, |x + 1| + |x - 2| ≥ 3 := min_value_abs_sum
  sorry

end min_value_abs_sum_pqr_inequality_l1108_110885


namespace calc_expression_solve_equation_l1108_110863

-- Problem 1: Calculation

theorem calc_expression : 
  |Real.sqrt 3 - 2| + Real.sqrt 12 - 6 * Real.sin (Real.pi / 6) + (-1/2 : Real)⁻¹ = Real.sqrt 3 - 3 := 
by {
  sorry
}

-- Problem 2: Solve the Equation

theorem solve_equation (x : Real) : 
  x * (x + 6) = -5 ↔ (x = -5 ∨ x = -1) := 
by {
  sorry
}

end calc_expression_solve_equation_l1108_110863


namespace truth_probability_of_A_l1108_110840

theorem truth_probability_of_A (P_B : ℝ) (P_AB : ℝ) (h : P_AB = 0.45 ∧ P_B = 0.60 ∧ ∀ (P_A : ℝ), P_AB = P_A * P_B) : 
  ∃ (P_A : ℝ), P_A = 0.75 :=
by
  sorry

end truth_probability_of_A_l1108_110840


namespace percentage_increase_correct_l1108_110826

def highest_price : ℕ := 24
def lowest_price : ℕ := 16

theorem percentage_increase_correct :
  ((highest_price - lowest_price) * 100 / lowest_price) = 50 :=
by
  sorry

end percentage_increase_correct_l1108_110826


namespace jordan_width_l1108_110881

-- Definitions based on conditions
def area_of_carols_rectangle : ℝ := 15 * 20
def jordan_length_feet : ℝ := 6
def feet_to_inches (feet: ℝ) : ℝ := feet * 12
def jordan_length_inches : ℝ := feet_to_inches jordan_length_feet

-- Main statement
theorem jordan_width :
  ∃ w : ℝ, w = 300 / 72 :=
sorry

end jordan_width_l1108_110881


namespace natalie_bushes_to_zucchinis_l1108_110824

/-- Each of Natalie's blueberry bushes yields ten containers of blueberries,
    and she trades six containers of blueberries for three zucchinis.
    Given this setup, prove that the number of bushes Natalie needs to pick
    in order to get sixty zucchinis is twelve. --/
theorem natalie_bushes_to_zucchinis :
  (∀ (bush_yield containers_needed : ℕ), bush_yield = 10 ∧ containers_needed = 60 * (6 / 3)) →
  (∀ (containers_total bushes_needed : ℕ), containers_total = 60 * (6 / 3) ∧ bushes_needed = containers_total * (1 / bush_yield)) →
  bushes_needed = 12 :=
by
  sorry

end natalie_bushes_to_zucchinis_l1108_110824


namespace percentage_vets_recommend_puppy_kibble_l1108_110841

theorem percentage_vets_recommend_puppy_kibble :
  ∀ (P : ℝ), (30 / 100 * 1000 = 300) → (1000 * P / 100 + 100 = 300) → P = 20 :=
by
  intros P h1 h2
  sorry

end percentage_vets_recommend_puppy_kibble_l1108_110841


namespace find_first_offset_l1108_110830

theorem find_first_offset (d b : ℝ) (Area : ℝ) :
  d = 22 → b = 6 → Area = 165 → (first_offset : ℝ) → 22 * (first_offset + 6) / 2 = 165 → first_offset = 9 :=
by
  intros hd hb hArea first_offset heq
  sorry

end find_first_offset_l1108_110830


namespace no_positive_integer_solutions_l1108_110851

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^5 ≠ y^2 + 4 := 
by sorry

end no_positive_integer_solutions_l1108_110851


namespace number_of_children_is_4_l1108_110816

-- Define the conditions from the problem
def youngest_child_age : ℝ := 1.5
def sum_of_ages : ℝ := 12
def common_difference : ℝ := 1

-- Define the number of children
def n : ℕ := 4

-- Prove that the number of children is 4 given the conditions
theorem number_of_children_is_4 :
  (∃ n : ℕ, (n / 2) * (2 * youngest_child_age + (n - 1) * common_difference) = sum_of_ages) ↔ n = 4 :=
by sorry

end number_of_children_is_4_l1108_110816


namespace problem_domains_equal_l1108_110839

/-- Proof problem:
    Prove that the domain of the function y = (x - 1)^(-1/2) is equal to the domain of the function y = ln(x - 1).
--/
theorem problem_domains_equal :
  {x : ℝ | x > 1} = {x : ℝ | x > 1} :=
by
  sorry

end problem_domains_equal_l1108_110839


namespace annie_hamburgers_l1108_110837

theorem annie_hamburgers (H : ℕ) (h₁ : 4 * H + 6 * 5 = 132 - 70) : H = 8 := by
  sorry

end annie_hamburgers_l1108_110837


namespace kate_bought_wands_l1108_110873

theorem kate_bought_wands (price_per_wand : ℕ)
                           (additional_cost : ℕ)
                           (total_money_collected : ℕ)
                           (number_of_wands_sold : ℕ)
                           (total_wands_bought : ℕ) :
  price_per_wand = 60 → additional_cost = 5 → total_money_collected = 130 → 
  number_of_wands_sold = total_money_collected / (price_per_wand + additional_cost) →
  total_wands_bought = number_of_wands_sold + 1 →
  total_wands_bought = 3 := by
  sorry

end kate_bought_wands_l1108_110873


namespace impossible_arrangement_l1108_110825

theorem impossible_arrangement : 
  ∀ (a : Fin 111 → ℕ), (∀ i, a i ≤ 500) → (∀ i j, i ≠ j → a i ≠ a j) → 
  ¬ ∀ i : Fin 111, (a i % 10 = ((Finset.univ.sum (λ j => if j = i then 0 else a j)) % 10)) :=
by 
  sorry

end impossible_arrangement_l1108_110825


namespace ab_range_l1108_110811

theorem ab_range (a b : ℝ) (h : a * b = a + b + 3) : a * b ≤ 1 ∨ a * b ≥ 9 := by
  sorry

end ab_range_l1108_110811


namespace boxes_of_chocolates_l1108_110894

theorem boxes_of_chocolates (total_pieces : ℕ) (pieces_per_box : ℕ) (h_total : total_pieces = 3000) (h_each : pieces_per_box = 500) : total_pieces / pieces_per_box = 6 :=
by
  sorry

end boxes_of_chocolates_l1108_110894


namespace find_coefficients_l1108_110803

theorem find_coefficients (k b : ℝ) :
    (∀ x y : ℝ, (y = k * x) → ((x-2)^2 + y^2 = 1) → (2*x + y + b = 0)) →
    ((k = 1/2) ∧ (b = -4)) :=
by
  sorry

end find_coefficients_l1108_110803


namespace average_number_of_carnations_l1108_110856

-- Define the conditions in Lean
def number_of_bouquet_1 : ℕ := 9
def number_of_bouquet_2 : ℕ := 14
def number_of_bouquet_3 : ℕ := 13
def total_bouquets : ℕ := 3

-- The main statement to be proved
theorem average_number_of_carnations : 
  (number_of_bouquet_1 + number_of_bouquet_2 + number_of_bouquet_3) / total_bouquets = 12 := 
by
  sorry

end average_number_of_carnations_l1108_110856


namespace max_profit_max_profit_price_l1108_110849

-- Definitions based on the conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 120
def initial_sales : ℕ := 20
def extra_sales_per_unit_decrease : ℕ := 2
def cost_price_constraint (x : ℝ) : Prop := 0 < x ∧ x ≤ 40

-- Expression for the profit function
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

-- Prove the maximum profit given the conditions
theorem max_profit : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 :=
by
  sorry

-- Proving that the selling price for max profit is 105 yuan
theorem max_profit_price : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 ∧ (initial_selling_price - x) = 105 :=
by
  sorry

end max_profit_max_profit_price_l1108_110849


namespace divisible_by_1989_l1108_110857

theorem divisible_by_1989 (n : ℕ) : 
  1989 ∣ (13 * (-50)^n + 17 * 40^n - 30) :=
by
  sorry

end divisible_by_1989_l1108_110857


namespace balloon_rearrangements_l1108_110842

-- Define the letters involved: vowels and consonants
def vowels := ['A', 'O', 'O', 'O']
def consonants := ['B', 'L', 'L', 'N']

-- State the problem in Lean 4:
theorem balloon_rearrangements : 
  ∃ n : ℕ, 
  (∀ (vowels := ['A', 'O', 'O', 'O']) 
     (consonants := ['B', 'L', 'L', 'N']), 
     n = 32) := sorry  -- we state that the number of rearrangements is 32 but do not provide the proof itself.

end balloon_rearrangements_l1108_110842


namespace chess_pieces_missing_l1108_110868

theorem chess_pieces_missing (total_pieces present_pieces missing_pieces : ℕ) 
  (h1 : total_pieces = 32)
  (h2 : present_pieces = 22)
  (h3 : missing_pieces = total_pieces - present_pieces) :
  missing_pieces = 10 :=
by
  sorry

end chess_pieces_missing_l1108_110868


namespace max_distinct_rectangles_l1108_110812

theorem max_distinct_rectangles : 
  ∃ (rectangles : Finset ℕ), (∀ n ∈ rectangles, n > 0) ∧ rectangles.sum id = 100 ∧ rectangles.card = 14 :=
by 
  sorry

end max_distinct_rectangles_l1108_110812


namespace probability_correct_l1108_110804

-- Definitions of given conditions
def P_AB := 2 / 3
def P_BC := 1 / 2

-- Probability that at least one road is at least 5 miles long
def probability_at_least_one_road_is_5_miles_long : ℚ :=
  1 - (1 - P_AB) * (1 - P_BC)

theorem probability_correct :
  probability_at_least_one_road_is_5_miles_long = 5 / 6 :=
by
  -- Proof goes here
  sorry

end probability_correct_l1108_110804


namespace original_number_of_people_l1108_110892

-- Define the conditions as Lean definitions
def two_thirds_left (x : ℕ) : ℕ := (2 * x) / 3
def one_fourth_dancing_left (x : ℕ) : ℕ := ((x / 3) - (x / 12))

-- The problem statement as Lean theorem
theorem original_number_of_people (x : ℕ) (h : x / 4 = 15) : x = 60 :=
by sorry

end original_number_of_people_l1108_110892


namespace dot_product_of_a_and_b_l1108_110880

-- Define the vectors
def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (3, 7)

-- Define the dot product function
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- State the theorem
theorem dot_product_of_a_and_b : dot_product a b = -18 := by
  sorry

end dot_product_of_a_and_b_l1108_110880


namespace min_value_l1108_110821

open Real

noncomputable def func (x y z : ℝ) : ℝ := 1 / x + 1 / y + 1 / z

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  func x y z ≥ 4.5 :=
by
  sorry

end min_value_l1108_110821


namespace circle_reflection_l1108_110829

-- Definition of the original center of the circle
def original_center : ℝ × ℝ := (8, -3)

-- Definition of the reflection transformation over the line y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Theorem stating that reflecting the original center over the line y = x results in a specific point
theorem circle_reflection : reflect original_center = (-3, 8) :=
  by
  -- skipping the proof part
  sorry

end circle_reflection_l1108_110829


namespace cos_thirteen_pi_over_three_l1108_110831

theorem cos_thirteen_pi_over_three : Real.cos (13 * Real.pi / 3) = 1 / 2 := 
by
  sorry

end cos_thirteen_pi_over_three_l1108_110831


namespace determine_n_l1108_110893

theorem determine_n (n : ℕ) (h : 17^(4 * n) = (1 / 17)^(n - 30)) : n = 6 :=
by {
  sorry
}

end determine_n_l1108_110893


namespace remaining_quantities_count_l1108_110897

theorem remaining_quantities_count 
  (S : ℕ) (S3 : ℕ) (S2 : ℕ) (n : ℕ) 
  (h1 : S / 5 = 10) 
  (h2 : S3 / 3 = 4) 
  (h3 : S = 50) 
  (h4 : S3 = 12) 
  (h5 : S2 = S - S3) 
  (h6 : S2 / n = 19) 
  : n = 2 := 
by 
  sorry

end remaining_quantities_count_l1108_110897


namespace green_balls_in_bag_l1108_110848

theorem green_balls_in_bag (b : ℕ) (P_blue : ℚ) (g : ℕ) (h1 : b = 8) (h2 : P_blue = 1 / 3) (h3 : P_blue = (b : ℚ) / (b + g)) :
  g = 16 :=
by
  sorry

end green_balls_in_bag_l1108_110848


namespace measure_of_angle_C_l1108_110886

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 12 * D) : C = 2160 / 13 :=
by {
  sorry
}

end measure_of_angle_C_l1108_110886


namespace percentage_of_adult_men_l1108_110872

theorem percentage_of_adult_men (total_members : ℕ) (children : ℕ) (p : ℕ) :
  total_members = 2000 → children = 200 → 
  (∀ adult_men_percentage : ℕ, adult_women_percentage = 2 * adult_men_percentage) → 
  (100 - p) = 3 * (p - 10) →  p = 30 :=
by sorry

end percentage_of_adult_men_l1108_110872


namespace initial_bottles_proof_l1108_110850

-- Define the conditions as variables and statements
def initial_bottles (X : ℕ) : Prop :=
X - 8 + 45 = 51

-- Theorem stating the proof problem
theorem initial_bottles_proof : initial_bottles 14 :=
by
  -- We need to prove the following:
  -- 14 - 8 + 45 = 51
  sorry

end initial_bottles_proof_l1108_110850


namespace condition_for_positive_expression_l1108_110815

theorem condition_for_positive_expression (a b c : ℝ) :
  (∀ x y : ℝ, x^2 + x * y + y^2 + a * x + b * y + c > 0) ↔ a^2 - a * b + b^2 < 3 * c :=
by
  -- Proof should be provided here
  sorry

end condition_for_positive_expression_l1108_110815


namespace smallest_positive_integer_l1108_110889

theorem smallest_positive_integer (n : ℕ) (h : 721 * n % 30 = 1137 * n % 30) :
  ∃ k : ℕ, k > 0 ∧ n = 2 * k :=
by
  sorry

end smallest_positive_integer_l1108_110889


namespace solve_equation1_solve_equation2_l1108_110844

theorem solve_equation1 (x : ℝ) (h1 : 3 * x^3 - 15 = 9) : x = 2 :=
sorry

theorem solve_equation2 (x : ℝ) (h2 : 2 * (x - 1)^2 = 72) : x = 7 ∨ x = -5 :=
sorry

end solve_equation1_solve_equation2_l1108_110844


namespace women_left_l1108_110895

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l1108_110895


namespace units_digit_17_pow_53_l1108_110833

theorem units_digit_17_pow_53 : (17^53) % 10 = 7 := 
by sorry

end units_digit_17_pow_53_l1108_110833


namespace trivia_team_original_members_l1108_110828

theorem trivia_team_original_members (x : ℕ) (h1 : 6 * (x - 2) = 18) : x = 5 :=
by
  sorry

end trivia_team_original_members_l1108_110828


namespace true_statement_count_l1108_110888

def n_star (n : ℕ) : ℚ := 1 / n

theorem true_statement_count :
  let s1 := (n_star 4 + n_star 8 = n_star 12)
  let s2 := (n_star 9 - n_star 1 = n_star 8)
  let s3 := (n_star 5 * n_star 3 = n_star 15)
  let s4 := (n_star 16 - n_star 4 = n_star 12)
  (if s1 then 1 else 0) +
  (if s2 then 1 else 0) +
  (if s3 then 1 else 0) +
  (if s4 then 1 else 0) = 1 :=
by
  -- Proof goes here
  sorry

end true_statement_count_l1108_110888


namespace volume_second_cube_l1108_110891

open Real

-- Define the ratio of the edges of the cubes
def edge_ratio (a b : ℝ) := a / b = 3 / 1

-- Define the volume of the first cube
def volume_first_cube (a : ℝ) := a^3 = 27

-- Define the edge of the second cube based on the edge of the first cube
def edge_second_cube (a b : ℝ) := a / 3 = b

-- Statement of the problem in Lean 4
theorem volume_second_cube 
  (a b : ℝ) 
  (h_edge_ratio : edge_ratio a b) 
  (h_volume_first : volume_first_cube a) 
  (h_edge_second : edge_second_cube a b) : 
  b^3 = 1 := 
sorry

end volume_second_cube_l1108_110891


namespace number_of_three_cell_shapes_l1108_110896

theorem number_of_three_cell_shapes (x y : ℕ) (h : 3 * x + 4 * y = 22) : x = 6 :=
sorry

end number_of_three_cell_shapes_l1108_110896


namespace calculate_f_at_2_l1108_110817

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_at_2 :
  (∀ x : ℝ, 25 * f (x / 1580) + (3 - Real.sqrt 34) * f (1580 / x) = 2017 * x) →
  f 2 = 265572 :=
by
  intro h
  sorry

end calculate_f_at_2_l1108_110817


namespace max_value_trig_expression_l1108_110867

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end max_value_trig_expression_l1108_110867


namespace range_of_k_l1108_110843

theorem range_of_k (a b c d k : ℝ) (hA : b = k * a - 2 * a - 1) (hB : d = k * c - 2 * c - 1) (h_diff : a ≠ c) (h_lt : (c - a) * (d - b) < 0) : k < 2 := 
sorry

end range_of_k_l1108_110843


namespace negation_of_universal_statement_l1108_110807

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end negation_of_universal_statement_l1108_110807


namespace james_total_cost_l1108_110813

def subscription_cost (base_cost : ℕ) (free_hours : ℕ) (extra_hour_cost : ℕ) (movie_rental_cost : ℝ) (streamed_hours : ℕ) (rented_movies : ℕ) : ℝ :=
  let extra_hours := max (streamed_hours - free_hours) 0
  base_cost + extra_hours * extra_hour_cost + rented_movies * movie_rental_cost

theorem james_total_cost 
  (base_cost : ℕ)
  (free_hours : ℕ)
  (extra_hour_cost : ℕ)
  (movie_rental_cost : ℝ)
  (streamed_hours : ℕ)
  (rented_movies : ℕ)
  (h_base_cost : base_cost = 15)
  (h_free_hours : free_hours = 50)
  (h_extra_hour_cost : extra_hour_cost = 2)
  (h_movie_rental_cost : movie_rental_cost = 0.10)
  (h_streamed_hours : streamed_hours = 53)
  (h_rented_movies : rented_movies = 30) :
  subscription_cost base_cost free_hours extra_hour_cost movie_rental_cost streamed_hours rented_movies = 24 := 
by {
  sorry
}

end james_total_cost_l1108_110813


namespace butterfat_mixture_l1108_110814

/-
  Given:
  - 8 gallons of milk with 40% butterfat
  - x gallons of milk with 10% butterfat
  - Resulting mixture with 20% butterfat

  Prove:
  - x = 16 gallons
-/

theorem butterfat_mixture (x : ℝ) : 
  (0.40 * 8 + 0.10 * x) / (8 + x) = 0.20 → x = 16 := 
by
  sorry

end butterfat_mixture_l1108_110814
