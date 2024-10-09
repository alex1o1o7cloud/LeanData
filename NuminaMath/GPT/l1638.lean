import Mathlib

namespace new_roots_quadratic_l1638_163807

variable {p q : ℝ}

theorem new_roots_quadratic :
  (∀ (r₁ r₂ : ℝ), r₁ + r₂ = -p ∧ r₁ * r₂ = q → 
  (x : ℝ) → x^2 + ((p^2 - 2 * q)^2 - 2 * q^2) * x + q^4 = 0) :=
by 
  intros r₁ r₂ h x
  have : r₁ + r₂ = -p := h.1
  have : r₁ * r₂ = q := h.2
  sorry

end new_roots_quadratic_l1638_163807


namespace find_solutions_l1638_163823

theorem find_solutions (x y z : ℝ) :
  (x = 5 / 3 ∧ y = -4 / 3 ∧ z = -4 / 3) ∨
  (x = 4 / 3 ∧ y = 4 / 3 ∧ z = -5 / 3) →
  (x^2 - y * z = abs (y - z) + 1) ∧ 
  (y^2 - z * x = abs (z - x) + 1) ∧ 
  (z^2 - x * y = abs (x - y) + 1) :=
by
  sorry

end find_solutions_l1638_163823


namespace largest_number_among_four_l1638_163851

theorem largest_number_among_four :
  let a := 0.965
  let b := 0.9687
  let c := 0.9618
  let d := 0.955
  max a (max b (max c d)) = b := 
sorry

end largest_number_among_four_l1638_163851


namespace intersection_eq_l1638_163809

def M : Set (ℝ × ℝ) := { p | ∃ x, p.2 = x^2 }
def N : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 2 }
def Intersect : Set (ℝ × ℝ) := { p | (M p) ∧ (N p)}

theorem intersection_eq : Intersect = { p : ℝ × ℝ | p = (1,1) ∨ p = (-1, 1) } :=
  sorry

end intersection_eq_l1638_163809


namespace least_product_of_distinct_primes_gt_30_l1638_163863

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l1638_163863


namespace bus_speed_excluding_stoppages_l1638_163852

theorem bus_speed_excluding_stoppages 
  (V : ℝ) -- Denote the average speed excluding stoppages as V
  (h1 : 30 / 1 = 30) -- condition 1: average speed including stoppages is 30 km/hr
  (h2 : 1 / 2 = 0.5) -- condition 2: The bus is moving for 0.5 hours per hour due to 30 min stoppage
  (h3 : V = 2 * 30) -- from the condition that the bus must cover the distance in half the time
  : V = 60 :=
by {
  sorry -- proof is not required
}

end bus_speed_excluding_stoppages_l1638_163852


namespace find_other_number_l1638_163834

theorem find_other_number (A : ℕ) (hcf_cond : Nat.gcd A 48 = 12) (lcm_cond : Nat.lcm A 48 = 396) : A = 99 := by
    sorry

end find_other_number_l1638_163834


namespace exist_one_common_ball_l1638_163883

theorem exist_one_common_ball (n : ℕ) (h_n : 5 ≤ n) (A : Fin (n+1) → Finset (Fin n))
  (hA_card : ∀ i, (A i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
sorry

end exist_one_common_ball_l1638_163883


namespace solution_set_absolute_value_inequality_l1638_163843

theorem solution_set_absolute_value_inequality (x : ℝ) :
  (|x-3| + |x-5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end solution_set_absolute_value_inequality_l1638_163843


namespace apples_and_oranges_l1638_163824

theorem apples_and_oranges :
  ∃ x y : ℝ, 2 * x + 3 * y = 6 ∧ 4 * x + 7 * y = 13 ∧ (16 * x + 23 * y = 47) :=
by
  sorry

end apples_and_oranges_l1638_163824


namespace carmen_parsley_left_l1638_163811

theorem carmen_parsley_left (plates_whole_sprig : ℕ) (plates_half_sprig : ℕ) (initial_sprigs : ℕ) :
  plates_whole_sprig = 8 →
  plates_half_sprig = 12 →
  initial_sprigs = 25 →
  initial_sprigs - (plates_whole_sprig + plates_half_sprig / 2) = 11 := by
  intros
  sorry

end carmen_parsley_left_l1638_163811


namespace train_speed_l1638_163816

theorem train_speed
  (train_length : ℝ) (platform_length : ℝ) (time_seconds : ℝ)
  (h_train_length : train_length = 450)
  (h_platform_length : platform_length = 300.06)
  (h_time : time_seconds = 25) :
  (train_length + platform_length) / time_seconds * 3.6 = 108.01 :=
by
  -- skipping the proof with sorry
  sorry

end train_speed_l1638_163816


namespace final_price_jacket_l1638_163887

-- Defining the conditions as per the problem
def original_price : ℚ := 250
def first_discount_rate : ℚ := 0.40
def second_discount_rate : ℚ := 0.15
def tax_rate : ℚ := 0.05

-- Defining the calculation steps
def first_discounted_price : ℚ := original_price * (1 - first_discount_rate)
def second_discounted_price : ℚ := first_discounted_price * (1 - second_discount_rate)
def final_price_inclusive_tax : ℚ := second_discounted_price * (1 + tax_rate)

-- The proof problem statement
theorem final_price_jacket : final_price_inclusive_tax = 133.88 := sorry

end final_price_jacket_l1638_163887


namespace high_fever_temperature_l1638_163819

theorem high_fever_temperature (T t : ℝ) (h1 : T = 36) (h2 : t > 13 / 12 * T) : t > 39 :=
by
  sorry

end high_fever_temperature_l1638_163819


namespace expression_evaluation_l1638_163825

theorem expression_evaluation (a b : ℤ) (h : a - 2 * b = 4) : 3 - a + 2 * b = -1 :=
by
  sorry

end expression_evaluation_l1638_163825


namespace find_divisor_l1638_163854

-- Defining the conditions
def dividend : ℕ := 181
def quotient : ℕ := 9
def remainder : ℕ := 1

-- The statement to prove
theorem find_divisor : ∃ (d : ℕ), dividend = (d * quotient) + remainder ∧ d = 20 := by
  sorry

end find_divisor_l1638_163854


namespace find_x_l1638_163896

-- Defining the conditions
def angle_PQR : ℝ := 180
def angle_PQS : ℝ := 125
def angle_QSR (x : ℝ) : ℝ := x
def SQ_eq_SR : Prop := true -- Assuming an isosceles triangle where SQ = SR.

-- The theorem to be proved
theorem find_x (x : ℝ) :
  angle_PQR = 180 → angle_PQS = 125 → SQ_eq_SR → angle_QSR x = 70 :=
by
  intros _ _ _
  sorry

end find_x_l1638_163896


namespace trajectory_of_A_l1638_163814

theorem trajectory_of_A (A B C : (ℝ × ℝ)) (x y : ℝ) : 
  B = (-2, 0) ∧ C = (2, 0) ∧ (dist A (0, 0) = 3) → 
  (x, y) = A → 
  x^2 + y^2 = 9 ∧ y ≠ 0 := 
sorry

end trajectory_of_A_l1638_163814


namespace value_of_k_parallel_vectors_l1638_163876

theorem value_of_k_parallel_vectors :
  (a : ℝ × ℝ) → (b : ℝ × ℝ) → (k : ℝ) →
  a = (2, 1) → b = (-1, k) → 
  (a.1 * b.2 - a.2 * b.1 = 0) →
  k = -(1/2) :=
by
  intros a b k ha hb hab_det
  sorry

end value_of_k_parallel_vectors_l1638_163876


namespace find_y_of_arithmetic_mean_l1638_163810

theorem find_y_of_arithmetic_mean (y : ℝ) (h: (7 + 12 + 19 + 8 + 10 + y) / 6 = 15) : y = 34 :=
by {
  -- Skipping the proof
  sorry
}

end find_y_of_arithmetic_mean_l1638_163810


namespace part_I_part_II_l1638_163874

-- Part I
theorem part_I :
  ∀ (x_0 y_0 : ℝ),
  (x_0 ^ 2 + y_0 ^ 2 = 8) ∧
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) →
  ∃ a b : ℝ, (a = 2 ∧ b = 2) →
  (∀ x y : ℝ, (x - 2) ^ 2 + (y - 2) ^ 2 = 8) :=
by 
sorry

-- Part II
theorem part_II :
  ¬ ∃ (x_0 y_0 k_1 k_2 : ℝ),
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) ∧
  (k_1k_2 = (y_0^2 - 4) / (x_0^2 - 4)) ∧
  (k_1 + k_2 = 2 * x_0 * y_0 / (x_0^2 - 4)) ∧
  (k_1k_2 - (k_1 + k_2) / (x_0 * y_0) + 1 = 0) :=
by 
sorry

end part_I_part_II_l1638_163874


namespace percent_non_unionized_women_is_80_l1638_163827

noncomputable def employeeStatistics :=
  let total_employees := 100
  let percent_men := 50
  let percent_unionized := 60
  let percent_unionized_men := 70
  let men := (percent_men / 100) * total_employees
  let unionized := (percent_unionized / 100) * total_employees
  let unionized_men := (percent_unionized_men / 100) * unionized
  let non_unionized_men := men - unionized_men
  let non_unionized := total_employees - unionized
  let non_unionized_women := non_unionized - non_unionized_men
  let percent_non_unionized_women := (non_unionized_women / non_unionized) * 100
  percent_non_unionized_women

theorem percent_non_unionized_women_is_80 :
  employeeStatistics = 80 :=
by
  sorry

end percent_non_unionized_women_is_80_l1638_163827


namespace ball_beyond_hole_l1638_163800

theorem ball_beyond_hole
  (first_turn_distance : ℕ)
  (second_turn_distance : ℕ)
  (total_distance_to_hole : ℕ) :
  first_turn_distance = 180 →
  second_turn_distance = first_turn_distance / 2 →
  total_distance_to_hole = 250 →
  second_turn_distance - (total_distance_to_hole - first_turn_distance) = 20 :=
by
  intros
  -- Proof omitted
  sorry

end ball_beyond_hole_l1638_163800


namespace boy_and_girl_roles_l1638_163806

-- Definitions of the conditions
def Sasha_says_boy : Prop := True
def Zhenya_says_girl : Prop := True
def at_least_one_lying (sasha_boy zhenya_girl : Prop) : Prop := 
  (sasha_boy = False) ∨ (zhenya_girl = False)

-- Theorem statement
theorem boy_and_girl_roles (sasha_boy : Prop) (zhenya_girl : Prop) 
  (H1 : Sasha_says_boy) (H2 : Zhenya_says_girl) (H3 : at_least_one_lying sasha_boy zhenya_girl) :
  sasha_boy = False ∧ zhenya_girl = True :=
sorry

end boy_and_girl_roles_l1638_163806


namespace total_amount_l1638_163826

theorem total_amount (a b c total first : ℕ)
  (h1 : a = 1 / 2) (h2 : b = 2 / 3) (h3 : c = 3 / 4)
  (h4 : first = 204)
  (ratio_sum : a * 12 + b * 12 + c * 12 = 23)
  (first_ratio : a * 12 = 6) :
  total = 23 * (first / 6) → total = 782 :=
by 
  sorry

end total_amount_l1638_163826


namespace slow_population_growth_before_ir_l1638_163877

-- Define the conditions
def low_level_social_productivity_before_ir : Prop := sorry
def high_birth_rate_before_ir : Prop := sorry
def high_mortality_rate_before_ir : Prop := sorry

-- The correct answer
def low_natural_population_growth_rate_before_ir : Prop := sorry

-- The theorem to prove
theorem slow_population_growth_before_ir 
  (h1 : low_level_social_productivity_before_ir) 
  (h2 : high_birth_rate_before_ir) 
  (h3 : high_mortality_rate_before_ir) : low_natural_population_growth_rate_before_ir := 
sorry

end slow_population_growth_before_ir_l1638_163877


namespace compute_fraction_l1638_163837

theorem compute_fraction (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end compute_fraction_l1638_163837


namespace proof_l1638_163875

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x ≥ 1
def q : Prop := ∀ x : ℝ, 0 < x → Real.exp x > Real.log x

-- The theorem statement
theorem proof : p ∧ q := by sorry

end proof_l1638_163875


namespace derivative_at_1_l1638_163831

def f (x : ℝ) : ℝ := (1 - 2 * x^3) ^ 10

theorem derivative_at_1 : deriv f 1 = 60 :=
by
  sorry

end derivative_at_1_l1638_163831


namespace units_cost_l1638_163805

theorem units_cost (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15)
  (h2 : 4 * x + 10 * y + z = 4.20) : 
  x + y + z = 1.05 :=
by 
  sorry

end units_cost_l1638_163805


namespace tom_hours_per_week_l1638_163897

-- Define the conditions
def summer_hours_per_week := 40
def summer_weeks := 8
def summer_total_earnings := 3200
def semester_weeks := 24
def semester_total_earnings := 2400
def hourly_wage := summer_total_earnings / (summer_hours_per_week * summer_weeks)
def total_hours_needed := semester_total_earnings / hourly_wage

-- Define the theorem to prove
theorem tom_hours_per_week :
  (total_hours_needed / semester_weeks) = 10 :=
sorry

end tom_hours_per_week_l1638_163897


namespace average_marks_l1638_163849

noncomputable def TatuyaScore (IvannaScore : ℝ) : ℝ :=
2 * IvannaScore

noncomputable def IvannaScore (DorothyScore : ℝ) : ℝ :=
(3/5) * DorothyScore

noncomputable def DorothyScore : ℝ := 90

noncomputable def XanderScore (TatuyaScore IvannaScore DorothyScore : ℝ) : ℝ :=
((TatuyaScore + IvannaScore + DorothyScore) / 3) + 10

noncomputable def SamScore (IvannaScore : ℝ) : ℝ :=
(3.8 * IvannaScore) + 5.5

noncomputable def OliviaScore (SamScore : ℝ) : ℝ :=
(3/2) * SamScore

theorem average_marks :
  let I := IvannaScore DorothyScore
  let T := TatuyaScore I
  let S := SamScore I
  let O := OliviaScore S
  let X := XanderScore T I DorothyScore
  let total_marks := T + I + DorothyScore + X + O + S
  (total_marks / 6) = 145.458333 := by sorry

end average_marks_l1638_163849


namespace inequality_proof_l1638_163855

theorem inequality_proof
  (x1 x2 x3 x4 x5 : ℝ)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5) :
  x1^2 + x2^2 + x3^2 + x4^2 + x5^2 ≥ x1 * (x2 + x3 + x4 + x5) :=
by
  sorry

end inequality_proof_l1638_163855


namespace digit_7_occurrences_in_range_1_to_2017_l1638_163869

-- Define the predicate that checks if a digit appears in a number
def digit_occurrences (d n : Nat) : Nat :=
  Nat.digits 10 n |>.count d

-- Define the range of numbers we are interested in
def range := (List.range' 1 2017)

-- Sum up the occurrences of digit 7 in the defined range
def total_occurrences (d : Nat) (range : List Nat) : Nat :=
  range.foldr (λ n acc => digit_occurrences d n + acc) 0

-- The main theorem to prove
theorem digit_7_occurrences_in_range_1_to_2017 : total_occurrences 7 range = 602 := by
  -- The proof should go here, but we only need to define the statement.
  sorry

end digit_7_occurrences_in_range_1_to_2017_l1638_163869


namespace no_positive_integer_solutions_l1638_163836

theorem no_positive_integer_solutions:
    ∀ x y : ℕ, x > 0 → y > 0 → x^2 + 2 * y^2 = 2 * x^3 - x → false :=
by
  sorry

end no_positive_integer_solutions_l1638_163836


namespace circle_center_l1638_163878

theorem circle_center : 
  ∃ (h k : ℝ), (h, k) = (1, -2) ∧ 
    ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - h)^2 + (y - k)^2 = 9 :=
by
  sorry

end circle_center_l1638_163878


namespace simplify_T_l1638_163881

noncomputable def T (x : ℝ) : ℝ :=
  (x+1)^4 - 4*(x+1)^3 + 6*(x+1)^2 - 4*(x+1) + 1

theorem simplify_T (x : ℝ) : T x = x^4 :=
  sorry

end simplify_T_l1638_163881


namespace solve_congruence_l1638_163829

-- Define the initial condition of the problem
def condition (x : ℤ) : Prop := (15 * x + 3) % 21 = 9 % 21

-- The statement that we want to prove
theorem solve_congruence : ∃ (a m : ℤ), condition a ∧ a % m = 6 % 7 ∧ a < m ∧ a + m = 13 :=
by {
    sorry
}

end solve_congruence_l1638_163829


namespace scientific_notation_of_120000_l1638_163850

theorem scientific_notation_of_120000 : 
  (120000 : ℝ) = 1.2 * 10^5 := 
by 
  sorry

end scientific_notation_of_120000_l1638_163850


namespace find_original_mean_l1638_163818

noncomputable def original_mean (M : ℝ) : Prop :=
  let num_observations := 50
  let decrement := 47
  let updated_mean := 153
  M * num_observations - (num_observations * decrement) = updated_mean * num_observations

theorem find_original_mean : original_mean 200 :=
by
  unfold original_mean
  simp [*, mul_sub_left_distrib] at *
  sorry

end find_original_mean_l1638_163818


namespace pole_intersection_height_l1638_163822

theorem pole_intersection_height :
  ∀ (d h1 h2 : ℝ), d = 120 ∧ h1 = 30 ∧ h2 = 90 → 
  ∃ y : ℝ, y = 18 :=
by
  sorry

end pole_intersection_height_l1638_163822


namespace friend_P_distance_l1638_163872

theorem friend_P_distance (v t : ℝ) (hv : v > 0)
  (distance_trail : 22 = (1.20 * v * t) + (v * t))
  (h_t : t = 22 / (2.20 * v)) : 
  (1.20 * v * t = 12) :=
by
  sorry

end friend_P_distance_l1638_163872


namespace chicken_problem_l1638_163815

theorem chicken_problem (x y z : ℕ) :
  x + y + z = 100 ∧ 5 * x + 3 * y + z / 3 = 100 → 
  (x = 0 ∧ y = 25 ∧ z = 75) ∨ 
  (x = 12 ∧ y = 4 ∧ z = 84) ∨ 
  (x = 8 ∧ y = 11 ∧ z = 81) ∨ 
  (x = 4 ∧ y = 18 ∧ z = 78) := 
sorry

end chicken_problem_l1638_163815


namespace determine_q_l1638_163890

-- Lean 4 statement
theorem determine_q (a : ℝ) (q : ℝ → ℝ) :
  (∀ x, q x = a * (x + 2) * (x - 3)) ∧ q 1 = 8 →
  q x = - (4 / 3) * x ^ 2 + (4 / 3) * x + 8 := 
sorry

end determine_q_l1638_163890


namespace find_speed_of_second_car_l1638_163864

noncomputable def problem : Prop := 
  let s1 := 1600 -- meters
  let s2 := 800 -- meters
  let v1 := 72 / 3.6 -- converting to meters per second for convenience; 72 km/h = 20 m/s
  let s := 200 -- meters
  let t1 := s1 / v1 -- time taken by the first car to reach the intersection
  let l1 := s2 - s -- scenario 1: second car travels 600 meters
  let l2 := s2 + s -- scenario 2: second car travels 1000 meters
  let v2_1 := l1 / t1 -- speed calculation for scenario 1
  let v2_2 := l2 / t1 -- speed calculation for scenario 2
  v2_1 = 7.5 ∧ v2_2 = 12.5 -- expected speeds in both scenarios

theorem find_speed_of_second_car : problem := sorry

end find_speed_of_second_car_l1638_163864


namespace different_quantifiers_not_equiv_l1638_163833

theorem different_quantifiers_not_equiv {x₀ : ℝ} :
  (∃ x₀ : ℝ, x₀^2 > 3) ↔ ¬ (∀ x₀ : ℝ, x₀^2 > 3) :=
by
  sorry

end different_quantifiers_not_equiv_l1638_163833


namespace expression_evaluation_l1638_163865

theorem expression_evaluation :
  (4 * 6 / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) - 1 = 0) :=
by sorry

end expression_evaluation_l1638_163865


namespace problem_solution_l1638_163861

theorem problem_solution (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 6 = 6 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x ^ Real.sqrt 3 + y ^ Real.sqrt 3 = 189 :=
sorry

end problem_solution_l1638_163861


namespace find_a_from_log_condition_l1638_163886

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_a_from_log_condition (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1)
  (h₂ : f a 9 = 2) : a = 3 :=
by
  sorry

end find_a_from_log_condition_l1638_163886


namespace allocation_schemes_for_5_teachers_to_3_buses_l1638_163802

noncomputable def number_of_allocation_schemes (teachers : ℕ) (buses : ℕ) : ℕ :=
  if buses = 3 ∧ teachers = 5 then 150 else 0

theorem allocation_schemes_for_5_teachers_to_3_buses : 
  number_of_allocation_schemes 5 3 = 150 := 
by
  sorry

end allocation_schemes_for_5_teachers_to_3_buses_l1638_163802


namespace golden_section_MP_length_l1638_163895

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem golden_section_MP_length (MN : ℝ) (hMN : MN = 2) (P : ℝ) 
  (hP : P > 0 ∧ P < MN ∧ P / (MN - P) = (MN - P) / P)
  (hMP_NP : MN - P < P) :
  P = Real.sqrt 5 - 1 :=
by
  sorry

end golden_section_MP_length_l1638_163895


namespace correct_operation_l1638_163898

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^3 ≠ 2 * a^5) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  (a^3 * a^5 ≠ a^15) ∧
  ((ab^2)^2 = a^2 * b^4) :=
by
  sorry

end correct_operation_l1638_163898


namespace circle_area_increase_l1638_163840

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  percentage_increase = 125 :=
by
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  sorry

end circle_area_increase_l1638_163840


namespace A_alone_finishes_work_in_30_days_l1638_163889

noncomputable def work_rate_A (B : ℝ) : ℝ := 2 * B

noncomputable def total_work (B : ℝ) : ℝ := 60 * B

theorem A_alone_finishes_work_in_30_days (B : ℝ) : (total_work B) / (work_rate_A B) = 30 := by
  sorry

end A_alone_finishes_work_in_30_days_l1638_163889


namespace eraser_crayon_difference_l1638_163894

def initial_crayons : Nat := 601
def initial_erasers : Nat := 406
def final_crayons : Nat := 336
def final_erasers : Nat := initial_erasers

theorem eraser_crayon_difference :
  final_erasers - final_crayons = 70 :=
by
  sorry

end eraser_crayon_difference_l1638_163894


namespace triangle_area_MEQF_l1638_163871

theorem triangle_area_MEQF
  (radius_P : ℝ)
  (chord_EF : ℝ)
  (par_EF_MN : Prop)
  (MQ : ℝ)
  (collinear_MQPN : Prop)
  (P MEF : ℝ × ℝ)
  (segment_P_Q : ℝ)
  (EF_length : ℝ)
  (radius_value : radius_P = 10)
  (EF_value : chord_EF = 12)
  (MQ_value : MQ = 20)
  (MN_parallel : par_EF_MN)
  (collinear : collinear_MQPN) :
  ∃ (area : ℝ), area = 48 := 
sorry

end triangle_area_MEQF_l1638_163871


namespace Malou_third_quiz_score_l1638_163853

theorem Malou_third_quiz_score (q1 q2 q3 : ℕ) (avg_score : ℕ) (total_quizzes : ℕ) : 
  q1 = 91 ∧ q2 = 90 ∧ avg_score = 91 ∧ total_quizzes = 3 → q3 = 92 :=
by
  intro h
  sorry

end Malou_third_quiz_score_l1638_163853


namespace c_share_l1638_163859

theorem c_share (x : ℕ) (a b c d : ℕ) 
  (h1: a = 5 * x)
  (h2: b = 3 * x)
  (h3: c = 2 * x)
  (h4: d = 3 * x)
  (h5: a = b + 1000): 
  c = 1000 := 
by 
  sorry

end c_share_l1638_163859


namespace exists_zero_in_interval_l1638_163845

open Set Real

theorem exists_zero_in_interval (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Icc a b)) 
  (h_pos : f a * f b > 0) : ∃ c ∈ Ioo a b, f c = 0 := sorry

end exists_zero_in_interval_l1638_163845


namespace infinite_series_sum_l1638_163842

theorem infinite_series_sum :
  (∑' n : ℕ, (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l1638_163842


namespace percent_equivalence_l1638_163848

theorem percent_equivalence (y : ℝ) : 0.30 * (0.60 * y) = 0.18 * y :=
by sorry

end percent_equivalence_l1638_163848


namespace domain_of_function_l1638_163870

theorem domain_of_function :
  {x : ℝ | x > 4 ∧ x ≠ 5} = (Set.Ioo 4 5 ∪ Set.Ioi 5) :=
by
  sorry

end domain_of_function_l1638_163870


namespace gcd_of_840_and_1764_l1638_163879

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_of_840_and_1764_l1638_163879


namespace evaluate_expression_l1638_163856

theorem evaluate_expression : (3200 - 3131) ^ 2 / 121 = 36 :=
by
  sorry

end evaluate_expression_l1638_163856


namespace negation_of_existence_l1638_163857

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem negation_of_existence:
  (∃ x : ℝ, log_base 3 x ≤ 0) ↔ ∀ x : ℝ, log_base 3 x < 0 :=
by
  sorry

end negation_of_existence_l1638_163857


namespace arithmetic_sequence_general_term_l1638_163891

theorem arithmetic_sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = 5 * n^2 + 3 * n)
  (hS₁ : a 1 = S 1)
  (hS₂ : ∀ n, a (n + 1) = S (n + 1) - S n) :
  ∀ n, a n = 10 * n - 2 :=
by
  sorry

end arithmetic_sequence_general_term_l1638_163891


namespace number_of_piles_l1638_163839

theorem number_of_piles (n : ℕ) (h₁ : 1000 < n) (h₂ : n < 2000)
  (h3 : n % 2 = 1) (h4 : n % 3 = 1) (h5 : n % 4 = 1) 
  (h6 : n % 5 = 1) (h7 : n % 6 = 1) (h8 : n % 7 = 1) (h9 : n % 8 = 1) : 
  ∃ p, p ≠ 1 ∧ p ≠ n ∧ (n % p = 0) ∧ p = 41 :=
by
  sorry

end number_of_piles_l1638_163839


namespace value_of_f_at_4_l1638_163838

noncomputable def f (α : ℝ) (x : ℝ) := x^α

theorem value_of_f_at_4 : 
  (∃ α : ℝ, f α 2 = (Real.sqrt 2) / 2) → f (-1 / 2) 4 = 1 / 2 :=
by
  intros h
  sorry

end value_of_f_at_4_l1638_163838


namespace largest_possible_markers_in_package_l1638_163828

theorem largest_possible_markers_in_package (alex_markers jordan_markers : ℕ) 
  (h1 : alex_markers = 56)
  (h2 : jordan_markers = 42) :
  Nat.gcd alex_markers jordan_markers = 14 :=
by
  sorry

end largest_possible_markers_in_package_l1638_163828


namespace cora_cookies_per_day_l1638_163821

theorem cora_cookies_per_day :
  (∀ (day : ℕ), day ∈ (Finset.range 30) →
    ∃ cookies_per_day : ℕ,
    cookies_per_day * 30 = 1620 / 18) →
  cookies_per_day = 3 := by
  sorry

end cora_cookies_per_day_l1638_163821


namespace solve_for_x_l1638_163804

theorem solve_for_x (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
sorry

end solve_for_x_l1638_163804


namespace tan_double_angle_l1638_163830

theorem tan_double_angle (α : ℝ) 
  (h : Real.tan α = 1 / 2) : Real.tan (2 * α) = 4 / 3 := 
by
  sorry

end tan_double_angle_l1638_163830


namespace father_age_l1638_163835

theorem father_age : 
  ∀ (S F : ℕ), (S - 5 = 11) ∧ (F - S = S) → F = 32 := 
by
  intros S F h
  -- Use the conditions to derive further equations and steps
  sorry

end father_age_l1638_163835


namespace product_and_sum_of_roots_l1638_163803

theorem product_and_sum_of_roots :
  let a := 24
  let b := 60
  let c := -600
  (c / a = -25) ∧ (-b / a = -2.5) := 
by
  sorry

end product_and_sum_of_roots_l1638_163803


namespace fresh_fruit_sold_l1638_163884

-- Define the conditions
def total_fruit_sold : ℕ := 9792
def frozen_fruit_sold : ℕ := 3513

-- Define what we need to prove
theorem fresh_fruit_sold : (total_fruit_sold - frozen_fruit_sold = 6279) := by
  sorry

end fresh_fruit_sold_l1638_163884


namespace find_B_squared_l1638_163817

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt 23 + 105 / x

theorem find_B_squared :
  ∃ B : ℝ, (B = (Real.sqrt 443)) ∧ (B^2 = 443) :=
by
  sorry

end find_B_squared_l1638_163817


namespace div_of_power_diff_div_l1638_163893

theorem div_of_power_diff_div (a b n : ℕ) (h : a ≠ b) (h₀ : n ∣ (a^n - b^n)) : n ∣ (a^n - b^n) / (a - b) :=
  sorry

end div_of_power_diff_div_l1638_163893


namespace percentage_change_area_right_triangle_l1638_163892

theorem percentage_change_area_right_triangle
  (b h : ℝ)
  (hb : b = 0.5 * h)
  (A_original A_new : ℝ)
  (H_original : A_original = (1 / 2) * b * h)
  (H_new : A_new = (1 / 2) * (1.10 * b) * (1.10 * h)) :
  ((A_new - A_original) / A_original) * 100 = 21 := by
  sorry

end percentage_change_area_right_triangle_l1638_163892


namespace hexagon_cookie_cutters_count_l1638_163812

-- Definitions for the conditions
def triangle_side_count := 3
def triangles := 6
def square_side_count := 4
def squares := 4
def total_sides := 46

-- Given conditions translated to Lean 4
def sides_from_triangles := triangles * triangle_side_count
def sides_from_squares := squares * square_side_count
def sides_from_triangles_and_squares := sides_from_triangles + sides_from_squares
def sides_from_hexagons := total_sides - sides_from_triangles_and_squares
def hexagon_side_count := 6

-- Statement to prove that there are 2 hexagon-shaped cookie cutters
theorem hexagon_cookie_cutters_count : sides_from_hexagons / hexagon_side_count = 2 := by
  sorry

end hexagon_cookie_cutters_count_l1638_163812


namespace determinant_scaled_matrix_l1638_163820

-- Definitions based on the conditions given in the problem.
def determinant2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

variable (a b c d : ℝ)
variable (h : determinant2x2 a b c d = 5)

-- The proof statement to be filled, proving the correct answer.
theorem determinant_scaled_matrix :
  determinant2x2 (2 * a) (2 * b) (2 * c) (2 * d) = 20 :=
by
  sorry

end determinant_scaled_matrix_l1638_163820


namespace geom_sequence_a4_times_a7_l1638_163801

theorem geom_sequence_a4_times_a7 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_q : q = 2) 
  (h_a2_a5 : a 2 * a 5 = 32) : 
  a 4 * a 7 = 512 :=
by 
  sorry

end geom_sequence_a4_times_a7_l1638_163801


namespace ellie_loan_difference_l1638_163847

noncomputable def principal : ℝ := 8000
noncomputable def simple_rate : ℝ := 0.10
noncomputable def compound_rate : ℝ := 0.08
noncomputable def time : ℝ := 5
noncomputable def compounding_periods : ℝ := 1

noncomputable def simple_interest_total (P r t : ℝ) : ℝ :=
  P + (P * r * t)

noncomputable def compound_interest_total (P r t n : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem ellie_loan_difference :
  (compound_interest_total principal compound_rate time compounding_periods) -
  (simple_interest_total principal simple_rate time) = -245.36 := 
  by sorry

end ellie_loan_difference_l1638_163847


namespace math_problem_l1638_163873

theorem math_problem 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (h_def : ∀ x, f x = 2 * Real.sin (2 * x + phi) + 1)
  (h_point : f 0 = 0)
  (h_phi_range : -Real.pi / 2 < phi ∧ phi < 0) : 
  (phi = -Real.pi / 6) ∧ (∃ k : ℤ, ∀ x, f x = 3 ↔ x = k * Real.pi + 2 * Real.pi / 3) :=
sorry

end math_problem_l1638_163873


namespace calculate_expression_l1638_163882
open Complex

-- Define the given values for a and b
def a := 3 + 2 * Complex.I
def b := 2 - 3 * Complex.I

-- Define the target expression
def target := 3 * a + 4 * b

-- The statement asserts that the target expression equals the expected result
theorem calculate_expression : target = 17 - 6 * Complex.I := by
  sorry

end calculate_expression_l1638_163882


namespace problem1_solve_eq_l1638_163832

theorem problem1_solve_eq (x : ℝ) : x * (x - 5) = 3 * x - 15 ↔ (x = 5 ∨ x = 3) := by
  sorry

end problem1_solve_eq_l1638_163832


namespace remainder_of_2_pow_30_plus_3_mod_7_l1638_163867

theorem remainder_of_2_pow_30_plus_3_mod_7 :
  (2^30 + 3) % 7 = 4 := 
sorry

end remainder_of_2_pow_30_plus_3_mod_7_l1638_163867


namespace solution_set_inequality_l1638_163846

variable (a x : ℝ)

-- Conditions
theorem solution_set_inequality (h₀ : 0 < a) (h₁ : a < 1) :
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) := 
by 
  sorry

end solution_set_inequality_l1638_163846


namespace train_length_correct_l1638_163899

noncomputable def length_of_train (train_speed_kmh : ℝ) (cross_time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * cross_time_s
  total_distance - bridge_length_m

theorem train_length_correct :
  length_of_train 45 30 205 = 170 :=
by
  sorry

end train_length_correct_l1638_163899


namespace lada_vs_elevator_l1638_163868

def Lada_speed_ratio (V U : ℝ) (S : ℝ) : Prop :=
  (∃ t_wait t_wait' : ℝ,
  ((t_wait = 3*S/U - 3*S/V) ∧ (t_wait' = 7*S/(2*U) - 7*S/V)) ∧
   (t_wait' = 3 * t_wait)) →
  U = 11/4 * V

theorem lada_vs_elevator (V U : ℝ) (S : ℝ) : Lada_speed_ratio V U S :=
sorry

end lada_vs_elevator_l1638_163868


namespace sector_arc_length_circumference_ratio_l1638_163860

theorem sector_arc_length_circumference_ratio
  {r : ℝ}
  (h_radius : ∀ (sector_radius : ℝ), sector_radius = 2/3 * r)
  (h_area : ∀ (sector_area circle_area : ℝ), sector_area / circle_area = 5/27) :
  ∀ (l C : ℝ), l / C = 5 / 18 :=
by
  -- Prove the theorem using the given hypothesis.
  -- Construction of the detailed proof will go here.
  sorry

end sector_arc_length_circumference_ratio_l1638_163860


namespace percentage_born_in_september_l1638_163862

theorem percentage_born_in_september (total famous : ℕ) (born_in_september : ℕ) (h1 : total = 150) (h2 : born_in_september = 12) :
  (born_in_september * 100 / total) = 8 :=
by
  sorry

end percentage_born_in_september_l1638_163862


namespace g_triple_of_10_l1638_163880

def g (x : Int) : Int :=
  if x < 4 then x^2 - 9 else x + 7

theorem g_triple_of_10 : g (g (g 10)) = 31 := by
  sorry

end g_triple_of_10_l1638_163880


namespace emily_lives_l1638_163844

theorem emily_lives :
  ∃ (lives_gained : ℕ), 
    let initial_lives := 42
    let lives_lost := 25
    let lives_after_loss := initial_lives - lives_lost
    let final_lives := 41
    lives_after_loss + lives_gained = final_lives :=
sorry

end emily_lives_l1638_163844


namespace identity_proof_l1638_163813

theorem identity_proof (a b c x y z : ℝ) : 
  (a * x + b * y + c * z) ^ 2 + (b * x + c * y + a * z) ^ 2 + (c * x + a * y + b * z) ^ 2 = 
  (c * x + b * y + a * z) ^ 2 + (b * x + a * y + c * z) ^ 2 + (a * x + c * y + b * z) ^ 2 := 
by
  sorry

end identity_proof_l1638_163813


namespace Ram_Gohul_days_work_together_l1638_163866

-- Define the conditions
def Ram_days := 10
def Gohul_days := 15

-- Define the work rates
def Ram_rate := 1 / Ram_days
def Gohul_rate := 1 / Gohul_days

-- Define the combined work rate
def Combined_rate := Ram_rate + Gohul_rate

-- Define the number of days to complete the job together
def Together_days := 1 / Combined_rate

-- State the proof problem
theorem Ram_Gohul_days_work_together : Together_days = 6 := by
  sorry

end Ram_Gohul_days_work_together_l1638_163866


namespace second_player_wins_for_n_11_l1638_163808

theorem second_player_wins_for_n_11 (N : ℕ) (h1 : N = 11) :
  ∃ (list : List ℕ), (∀ x ∈ list, x > 0 ∧ x ≤ 25) ∧
     list.sum ≥ 200 ∧
     (∃ sublist : List ℕ, sublist.sum ≥ 200 - N ∧ sublist.sum ≤ 200 + N) :=
by
  let N := 11
  sorry

end second_player_wins_for_n_11_l1638_163808


namespace harmonic_mean_of_1_3_1_div_2_l1638_163858

noncomputable def harmonicMean (a b c : ℝ) : ℝ :=
  let reciprocals := [1 / a, 1 / b, 1 / c]
  (reciprocals.sum) / reciprocals.length

theorem harmonic_mean_of_1_3_1_div_2 : harmonicMean 1 3 (1 / 2) = 9 / 10 :=
  sorry

end harmonic_mean_of_1_3_1_div_2_l1638_163858


namespace shorts_more_than_checkered_l1638_163888

noncomputable def total_students : ℕ := 81

noncomputable def striped_shirts : ℕ := (2 * total_students) / 3

noncomputable def checkered_shirts : ℕ := total_students - striped_shirts

noncomputable def shorts : ℕ := striped_shirts - 8

theorem shorts_more_than_checkered :
  shorts - checkered_shirts = 19 :=
by
  sorry

end shorts_more_than_checkered_l1638_163888


namespace total_height_of_pipes_l1638_163885

theorem total_height_of_pipes 
  (diameter : ℝ) (radius : ℝ) (total_pipes : ℕ) (first_row_pipes : ℕ) (second_row_pipes : ℕ) 
  (h : ℝ) 
  (h_diam : diameter = 10)
  (h_radius : radius = 5)
  (h_total_pipes : total_pipes = 5)
  (h_first_row : first_row_pipes = 2)
  (h_second_row : second_row_pipes = 3) :
  h = 10 + 5 * Real.sqrt 3 := 
sorry

end total_height_of_pipes_l1638_163885


namespace metric_regression_equation_l1638_163841

noncomputable def predicted_weight_imperial (height : ℝ) : ℝ :=
  4 * height - 130

def inch_to_cm (inch : ℝ) : ℝ := 2.54 * inch
def pound_to_kg (pound : ℝ) : ℝ := 0.45 * pound

theorem metric_regression_equation (height_cm : ℝ) :
  (0.72 * height_cm - 58.5) = 
  (pound_to_kg (predicted_weight_imperial (height_cm / 2.54))) :=
by
  sorry

end metric_regression_equation_l1638_163841
