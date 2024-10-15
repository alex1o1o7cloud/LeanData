import Mathlib

namespace NUMINAMATH_GPT_crows_and_trees_l238_23820

variable (x y : ℕ)

theorem crows_and_trees (h1 : x = 3 * y + 5) (h2 : x = 5 * (y - 1)) : 
  (x - 5) / 3 = y ∧ x / 5 = y - 1 :=
by
  sorry

end NUMINAMATH_GPT_crows_and_trees_l238_23820


namespace NUMINAMATH_GPT_profit_percentage_l238_23863

theorem profit_percentage (SP CP : ℕ) (h₁ : SP = 800) (h₂ : CP = 640) : (SP - CP) / CP * 100 = 25 :=
by 
  sorry

end NUMINAMATH_GPT_profit_percentage_l238_23863


namespace NUMINAMATH_GPT_problem_statement_l238_23845

theorem problem_statement (m : ℝ) (h : m^2 - m - 2 = 0) : m^2 - m + 2023 = 2025 :=
sorry

end NUMINAMATH_GPT_problem_statement_l238_23845


namespace NUMINAMATH_GPT_range_of_m_l238_23829

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + m^2 - 1 = 0) → (-2 < x)) ↔ m > -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l238_23829


namespace NUMINAMATH_GPT_jacob_has_5_times_more_l238_23856

variable (A J D : ℕ)
variable (hA : A = 75)
variable (hAJ : A = J / 2)
variable (hD : D = 30)

theorem jacob_has_5_times_more (hA : A = 75) (hAJ : A = J / 2) (hD : D = 30) : J / D = 5 :=
sorry

end NUMINAMATH_GPT_jacob_has_5_times_more_l238_23856


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_l238_23870

variable (a : ℕ → ℝ)
variable (a2_eq : a 2 = 4)
variable (a6_eq : a 6 = 2)

theorem arithmetic_sequence_a8 :
  a 8 = 1 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_l238_23870


namespace NUMINAMATH_GPT_marble_box_l238_23876

theorem marble_box (T: ℕ) 
  (h_white: (1 / 6) * T = T / 6)
  (h_green: (1 / 5) * T = T / 5)
  (h_red_blue: (19 / 30) * T = 19 * T / 30)
  (h_sum: (T / 6) + (T / 5) + (19 * T / 30) = T): 
  ∃ k : ℕ, T = 30 * k ∧ k ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_marble_box_l238_23876


namespace NUMINAMATH_GPT_Ken_bought_2_pounds_of_steak_l238_23860

theorem Ken_bought_2_pounds_of_steak (pound_cost total_paid change: ℝ) 
    (h1 : pound_cost = 7) 
    (h2 : total_paid = 20) 
    (h3 : change = 6) : 
    (total_paid - change) / pound_cost = 2 :=
by
  sorry

end NUMINAMATH_GPT_Ken_bought_2_pounds_of_steak_l238_23860


namespace NUMINAMATH_GPT_car_speed_l238_23880

theorem car_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 495) (h_time : time = 5) : 
  distance / time = 99 :=
by
  rw [h_distance, h_time]
  norm_num

end NUMINAMATH_GPT_car_speed_l238_23880


namespace NUMINAMATH_GPT_inequality_sqrt_sum_leq_one_plus_sqrt_l238_23858

theorem inequality_sqrt_sum_leq_one_plus_sqrt (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  Real.sqrt (a * (1 - b) * (1 - c)) + Real.sqrt (b * (1 - a) * (1 - c)) + Real.sqrt (c * (1 - a) * (1 - b)) 
  ≤ 1 + Real.sqrt (a * b * c) :=
sorry

end NUMINAMATH_GPT_inequality_sqrt_sum_leq_one_plus_sqrt_l238_23858


namespace NUMINAMATH_GPT_largest_alpha_exists_l238_23869

theorem largest_alpha_exists : 
  ∃ α, (∀ m n : ℕ, 0 < m → 0 < n → (m:ℝ) / (n:ℝ) < Real.sqrt 7 → α / (n^2:ℝ) ≤ 7 - (m^2:ℝ) / (n^2:ℝ)) ∧ α = 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_alpha_exists_l238_23869


namespace NUMINAMATH_GPT_grover_total_profit_is_15_l238_23847

theorem grover_total_profit_is_15 
  (boxes : ℕ) 
  (masks_per_box : ℕ) 
  (price_per_mask : ℝ) 
  (cost_of_boxes : ℝ) 
  (total_profit : ℝ)
  (hb : boxes = 3)
  (hm : masks_per_box = 20)
  (hp : price_per_mask = 0.5)
  (hc : cost_of_boxes = 15)
  (htotal : total_profit = (boxes * masks_per_box) * price_per_mask - cost_of_boxes) :
  total_profit = 15 :=
sorry

end NUMINAMATH_GPT_grover_total_profit_is_15_l238_23847


namespace NUMINAMATH_GPT_triangle_at_most_one_obtuse_angle_l238_23836

theorem triangle_at_most_one_obtuse_angle :
  (∀ (α β γ : ℝ), α + β + γ = 180 → α ≤ 90 ∨ β ≤ 90 ∨ γ ≤ 90) ↔
  ¬ (∃ (α β γ : ℝ), α + β + γ = 180 ∧ α > 90 ∧ β > 90) :=
by
  sorry

end NUMINAMATH_GPT_triangle_at_most_one_obtuse_angle_l238_23836


namespace NUMINAMATH_GPT_john_pays_12_dollars_l238_23886

/-- Define the conditions -/
def number_of_toys : ℕ := 5
def cost_per_toy : ℝ := 3
def discount_rate : ℝ := 0.2

/-- Define the total cost before discount -/
def total_cost_before_discount := number_of_toys * cost_per_toy

/-- Define the discount amount -/
def discount_amount := total_cost_before_discount * discount_rate

/-- Define the final amount John pays -/
def final_amount := total_cost_before_discount - discount_amount

/-- The theorem to be proven -/
theorem john_pays_12_dollars : final_amount = 12 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_john_pays_12_dollars_l238_23886


namespace NUMINAMATH_GPT_evaluations_total_l238_23889

theorem evaluations_total :
    let class_A_students := 30
    let class_A_mc := 12
    let class_A_essay := 3
    let class_A_presentation := 1

    let class_B_students := 25
    let class_B_mc := 15
    let class_B_short_answer := 5
    let class_B_essay := 2

    let class_C_students := 35
    let class_C_mc := 10
    let class_C_essay := 3
    let class_C_presentation_groups := class_C_students / 5 -- groups of 5

    let class_D_students := 40
    let class_D_mc := 11
    let class_D_short_answer := 4
    let class_D_essay := 3

    let class_E_students := 20
    let class_E_mc := 14
    let class_E_short_answer := 5
    let class_E_essay := 2

    let total_mc := (class_A_students * class_A_mc) +
                    (class_B_students * class_B_mc) +
                    (class_C_students * class_C_mc) +
                    (class_D_students * class_D_mc) +
                    (class_E_students * class_E_mc)

    let total_short_answer := (class_B_students * class_B_short_answer) +
                              (class_D_students * class_D_short_answer) +
                              (class_E_students * class_E_short_answer)

    let total_essay := (class_A_students * class_A_essay) +
                       (class_B_students * class_B_essay) +
                       (class_C_students * class_C_essay) +
                       (class_D_students * class_D_essay) +
                       (class_E_students * class_E_essay)

    let total_presentation := (class_A_students * class_A_presentation) +
                              class_C_presentation_groups

    total_mc + total_short_answer + total_essay + total_presentation = 2632 := by
    sorry

end NUMINAMATH_GPT_evaluations_total_l238_23889


namespace NUMINAMATH_GPT_sum_integers_minus15_to_6_l238_23862

def sum_range (a b : ℤ) : ℤ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_minus15_to_6 : sum_range (-15) (6) = -99 :=
  by
  -- Skipping the proof details
  sorry

end NUMINAMATH_GPT_sum_integers_minus15_to_6_l238_23862


namespace NUMINAMATH_GPT_triangle_proof_l238_23888

noncomputable def length_DC (AB DA BC DB : ℝ) : ℝ :=
  Real.sqrt (BC^2 - DB^2)

theorem triangle_proof :
  ∀ (AB DA BC DB : ℝ), AB = 30 → DA = 24 → BC = 22.5 → DB = 18 →
  length_DC AB DA BC DB = 13.5 :=
by
  intros AB DA BC DB hAB hDA hBC hDB
  rw [length_DC]
  sorry

end NUMINAMATH_GPT_triangle_proof_l238_23888


namespace NUMINAMATH_GPT_min_expression_value_l238_23868

theorem min_expression_value (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ m : ℝ, (∀ x y : ℝ, x > 2 → y > 2 → (x^3 / (y - 2) + y^3 / (x - 2)) ≥ m) ∧ 
          (m = 64) :=
by
  sorry

end NUMINAMATH_GPT_min_expression_value_l238_23868


namespace NUMINAMATH_GPT_Amy_work_hours_l238_23881

theorem Amy_work_hours (summer_weeks: ℕ) (summer_hours_per_week: ℕ) (summer_total_earnings: ℕ)
                       (school_weeks: ℕ) (school_total_earnings: ℕ) (hourly_wage: ℕ) 
                       (school_hours_per_week: ℕ):
    summer_weeks = 8 →
    summer_hours_per_week = 40 →
    summer_total_earnings = 3200 →
    school_weeks = 32 →
    school_total_earnings = 4800 →
    hourly_wage = summer_total_earnings / (summer_weeks * summer_hours_per_week) →
    school_hours_per_week = school_total_earnings / (hourly_wage * school_weeks) →
    school_hours_per_week = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Amy_work_hours_l238_23881


namespace NUMINAMATH_GPT_find_principal_l238_23875

variable (R P : ℝ)
variable (h1 : ∀ (R P : ℝ), (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400)

theorem find_principal (h1 : ∀ (R P : ℝ), (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400) :
  P = 800 := 
sorry

end NUMINAMATH_GPT_find_principal_l238_23875


namespace NUMINAMATH_GPT_range_of_m_l238_23844

theorem range_of_m (m : ℝ) (y_P : ℝ) (h1 : -3 ≤ y_P) (h2 : y_P ≤ 0) :
  m = (2 + y_P) / 2 → -1 / 2 ≤ m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l238_23844


namespace NUMINAMATH_GPT_problem_l238_23814

def g (x : ℕ) : ℕ := x^2 + 1
def f (x : ℕ) : ℕ := 3 * x - 2

theorem problem : f (g 3) = 28 := by
  sorry

end NUMINAMATH_GPT_problem_l238_23814


namespace NUMINAMATH_GPT_negation_of_proposition_l238_23854

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ ∃ x : ℝ, Real.exp x ≤ x^2 :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l238_23854


namespace NUMINAMATH_GPT_waiter_earnings_l238_23843

theorem waiter_earnings (total_customers : ℕ) (no_tip_customers : ℕ) (tip_per_customer : ℕ)
  (h1 : total_customers = 10)
  (h2 : no_tip_customers = 5)
  (h3 : tip_per_customer = 3) :
  (total_customers - no_tip_customers) * tip_per_customer = 15 :=
by sorry

end NUMINAMATH_GPT_waiter_earnings_l238_23843


namespace NUMINAMATH_GPT_percentage_small_bottles_sold_l238_23896

theorem percentage_small_bottles_sold :
  ∀ (x : ℕ), (6000 - (x * 60)) + 8500 = 13780 → x = 12 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_percentage_small_bottles_sold_l238_23896


namespace NUMINAMATH_GPT_find_am_2n_l238_23812

-- Definition of the conditions
variables {a : ℝ} {m n : ℝ}
axiom am_eq_5 : a ^ m = 5
axiom an_eq_2 : a ^ n = 2

-- The statement we want to prove
theorem find_am_2n : a ^ (m - 2 * n) = 5 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_am_2n_l238_23812


namespace NUMINAMATH_GPT_technicians_in_workshop_l238_23857

theorem technicians_in_workshop 
  (total_workers : ℕ) 
  (avg_salary_all : ℕ) 
  (avg_salary_tech : ℕ) 
  (avg_salary_rest : ℕ) 
  (total_salary : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (h1 : total_workers = 14) 
  (h2 : avg_salary_all = 8000) 
  (h3 : avg_salary_tech = 10000) 
  (h4 : avg_salary_rest = 6000) 
  (h5 : total_salary = total_workers * avg_salary_all) 
  (h6 : T + R = 14)
  (h7 : total_salary = 112000) 
  (h8 : total_salary = avg_salary_tech * T + avg_salary_rest * R) :
  T = 7 := 
by {
  -- Proof goes here
  sorry
} 

end NUMINAMATH_GPT_technicians_in_workshop_l238_23857


namespace NUMINAMATH_GPT_parabola_min_value_roots_l238_23864

-- Lean definition encapsulating the problem conditions and conclusion
theorem parabola_min_value_roots (a b c : ℝ) 
  (h1 : ∀ x, (a * x^2 + b * x + c) ≥ 36)
  (hvc : (b^2 - 4 * a * c) = 0)
  (hx1 : (a * (-3)^2 + b * (-3) + c) = 0)
  (hx2 : (a * (5)^2 + b * 5 + c) = 0)
  : a + b + c = 36 := by
  sorry

end NUMINAMATH_GPT_parabola_min_value_roots_l238_23864


namespace NUMINAMATH_GPT_money_together_l238_23802

variable (Billy Sam : ℕ)

theorem money_together (h1 : Billy = 2 * Sam - 25) (h2 : Sam = 75) : Billy + Sam = 200 := by
  sorry

end NUMINAMATH_GPT_money_together_l238_23802


namespace NUMINAMATH_GPT_find_pairs_l238_23838

noncomputable def diamond (a b : ℝ) : ℝ :=
  a^2 * b^2 - a^3 * b - a * b^3

theorem find_pairs (x y : ℝ) :
  diamond x y = diamond y x ↔
  x = 0 ∨ y = 0 ∨ x = y ∨ x = -y :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l238_23838


namespace NUMINAMATH_GPT_shorter_base_length_l238_23859

-- Let AB be the longer base of the trapezoid with length 24 cm
def AB : ℝ := 24

-- Let KT be the distance between midpoints of the diagonals with length 4 cm
def KT : ℝ := 4

-- Let CD be the shorter base of the trapezoid
variable (CD : ℝ)

-- The given condition is that KT is equal to half the difference of the lengths of the bases
axiom KT_eq : KT = (AB - CD) / 2

theorem shorter_base_length : CD = 16 := by
  sorry

end NUMINAMATH_GPT_shorter_base_length_l238_23859


namespace NUMINAMATH_GPT_angle_ACB_is_25_l238_23841

theorem angle_ACB_is_25 (angle_ABD angle_BAC : ℝ) (is_supplementary : angle_ABD + (180 - angle_BAC) = 180) (angle_ABC_eq : angle_BAC = 95) (angle_ABD_eq : angle_ABD = 120) :
  180 - (angle_BAC + (180 - angle_ABD)) = 25 :=
by
  sorry

end NUMINAMATH_GPT_angle_ACB_is_25_l238_23841


namespace NUMINAMATH_GPT_max_of_2xy_l238_23837

theorem max_of_2xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 2 * x * y ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_max_of_2xy_l238_23837


namespace NUMINAMATH_GPT_sum_of_coordinates_l238_23890

theorem sum_of_coordinates (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 20) : x + y = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_l238_23890


namespace NUMINAMATH_GPT_gears_can_look_complete_l238_23832

theorem gears_can_look_complete (n : ℕ) (h1 : n = 14)
                                 (h2 : ∀ k, k = 4)
                                 (h3 : ∀ i, 0 ≤ i ∧ i < n) :
  ∃ j, 1 ≤ j ∧ j < n ∧ (∀ m1 m2, m1 ≠ m2 → ((m1 + j) % n) ≠ ((m2 + j) % n)) := 
sorry

end NUMINAMATH_GPT_gears_can_look_complete_l238_23832


namespace NUMINAMATH_GPT_find_common_ratio_l238_23848

theorem find_common_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : 3 * S 3 = a 4 - 2)
  (h4 : 3 * S 2 = a 3 - 2)
  (h5 : ∀ n : ℕ, a (n+1) = q * a n) : q = 4 := sorry

end NUMINAMATH_GPT_find_common_ratio_l238_23848


namespace NUMINAMATH_GPT_cylinder_volume_triple_quadruple_l238_23803

theorem cylinder_volume_triple_quadruple (r h : ℝ) (V : ℝ) (π : ℝ) (original_volume : V = π * r^2 * h) 
                                         (original_volume_value : V = 8):
  ∃ V', V' = π * (3 * r)^2 * (4 * h) ∧ V' = 288 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_triple_quadruple_l238_23803


namespace NUMINAMATH_GPT_find_ordered_pairs_l238_23822

theorem find_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a - b) ^ (a * b) = a ^ b * b ^ a) :
  (a, b) = (4, 2) := by
  sorry

end NUMINAMATH_GPT_find_ordered_pairs_l238_23822


namespace NUMINAMATH_GPT_ratio_proof_l238_23899

variable (a b c d : ℚ)

theorem ratio_proof 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7) :
  d / a = 2 / 35 := by
  sorry

end NUMINAMATH_GPT_ratio_proof_l238_23899


namespace NUMINAMATH_GPT_store_cost_comparison_l238_23816

noncomputable def store_A_cost (x : ℕ) : ℝ := 1760 + 40 * x
noncomputable def store_B_cost (x : ℕ) : ℝ := 1920 + 32 * x

theorem store_cost_comparison (x : ℕ) (h : x > 16) :
  (x > 20 → store_B_cost x < store_A_cost x) ∧ (x < 20 → store_A_cost x < store_B_cost x) :=
by
  sorry

end NUMINAMATH_GPT_store_cost_comparison_l238_23816


namespace NUMINAMATH_GPT_problem_statement_l238_23809

open Real

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, sin θ)

theorem problem_statement (A B : ℝ × ℝ) 
  (θA θB : ℝ) 
  (hA : A = curve_C θA) 
  (hB : B = curve_C θB) 
  (h_perpendicular : θB = θA + π / 2) :
  (1 / (A.1 ^ 2 + A.2 ^ 2)) + (1 / (B.1 ^ 2 + B.2 ^ 2)) = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l238_23809


namespace NUMINAMATH_GPT_find_m_of_quadratic_fn_l238_23883

theorem find_m_of_quadratic_fn (m : ℚ) (h : 2 * m - 1 = 2) : m = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_of_quadratic_fn_l238_23883


namespace NUMINAMATH_GPT_total_count_not_47_l238_23891

theorem total_count_not_47 (h c : ℕ) : 11 * h + 6 * c ≠ 47 := by
  sorry

end NUMINAMATH_GPT_total_count_not_47_l238_23891


namespace NUMINAMATH_GPT_num_O_atoms_correct_l238_23823

-- Conditions
def atomic_weight_H : ℕ := 1
def atomic_weight_Cr : ℕ := 52
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_Cr_atoms : ℕ := 1
def molecular_weight : ℕ := 118

-- Calculations
def weight_H : ℕ := num_H_atoms * atomic_weight_H
def weight_Cr : ℕ := num_Cr_atoms * atomic_weight_Cr
def total_weight_H_Cr : ℕ := weight_H + weight_Cr
def weight_O : ℕ := molecular_weight - total_weight_H_Cr
def num_O_atoms : ℕ := weight_O / atomic_weight_O

-- Theorem to prove the number of Oxygen atoms is 4
theorem num_O_atoms_correct : num_O_atoms = 4 :=
by {
  sorry -- Proof not provided.
}

end NUMINAMATH_GPT_num_O_atoms_correct_l238_23823


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l238_23806

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = 1) (h₂ : b = -2) :
  (2 * a + b)^2 - 3 * a * (2 * a - b) = -12 :=
by
  rw [h₁, h₂]
  -- Now the expression to prove transforms to:
  -- (2 * 1 + (-2))^2 - 3 * 1 * (2 * 1 - (-2)) = -12
  -- Subsequent proof steps would follow simplification directly.
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l238_23806


namespace NUMINAMATH_GPT_solve_fish_tank_problem_l238_23872

def fish_tank_problem : Prop :=
  ∃ (first_tank_fish second_tank_fish third_tank_fish : ℕ),
  first_tank_fish = 7 + 8 ∧
  second_tank_fish = 2 * first_tank_fish ∧
  third_tank_fish = 10 ∧
  (third_tank_fish : ℚ) / second_tank_fish = 1 / 3

theorem solve_fish_tank_problem : fish_tank_problem :=
by
  sorry

end NUMINAMATH_GPT_solve_fish_tank_problem_l238_23872


namespace NUMINAMATH_GPT_f_11_5_equals_neg_1_l238_23887

-- Define the function f with the given properties
axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom periodic_function (f : ℝ → ℝ) : ∀ x, f (x + 2) = f x
axiom f_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x

-- State the theorem to be proved
theorem f_11_5_equals_neg_1 (f : ℝ → ℝ) 
  (odd_f : ∀ x, f (-x) = -f x)
  (periodic_f : ∀ x, f (x + 2) = f x)
  (f_int : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (11.5) = -1 :=
sorry

end NUMINAMATH_GPT_f_11_5_equals_neg_1_l238_23887


namespace NUMINAMATH_GPT_minimum_value_expression_l238_23861

noncomputable def minimum_value (a b : ℝ) := (1 / (2 * |a|)) + (|a| / b)

theorem minimum_value_expression
  (a : ℝ) (b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  ∃ (min_val : ℝ), min_val = 3 / 4 ∧ ∀ (a b : ℝ), a + b = 2 → b > 0 → minimum_value a b ≥ min_val :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l238_23861


namespace NUMINAMATH_GPT_find_MorkTaxRate_l238_23867

noncomputable def MorkIncome : ℝ := sorry
noncomputable def MorkTaxRate : ℝ := sorry 
noncomputable def MindyTaxRate : ℝ := 0.30 
noncomputable def MindyIncome : ℝ := 4 * MorkIncome 
noncomputable def combinedTaxRate : ℝ := 0.32 

theorem find_MorkTaxRate :
  (MorkTaxRate * MorkIncome + MindyTaxRate * MindyIncome) / (MorkIncome + MindyIncome) = combinedTaxRate →
  MorkTaxRate = 0.40 := sorry

end NUMINAMATH_GPT_find_MorkTaxRate_l238_23867


namespace NUMINAMATH_GPT_time_passed_since_midnight_l238_23804

theorem time_passed_since_midnight (h : ℝ) :
  h = (12 - h) + (2/5) * h → h = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_time_passed_since_midnight_l238_23804


namespace NUMINAMATH_GPT_traffic_flow_solution_l238_23810

noncomputable def traffic_flow_second_ring : ℕ := 10000
noncomputable def traffic_flow_third_ring (x : ℕ) : Prop := 3 * x - (x + 2000) = 2 * traffic_flow_second_ring

theorem traffic_flow_solution :
  ∃ (x : ℕ), traffic_flow_third_ring x ∧ (x = 11000) ∧ (x + 2000 = 13000) :=
by
  sorry

end NUMINAMATH_GPT_traffic_flow_solution_l238_23810


namespace NUMINAMATH_GPT_number_of_routes_4x3_grid_l238_23831

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem number_of_routes_4x3_grid : binomial_coefficient 7 4 = 35 := by
  sorry

end NUMINAMATH_GPT_number_of_routes_4x3_grid_l238_23831


namespace NUMINAMATH_GPT_parallel_lines_equal_slopes_l238_23898

theorem parallel_lines_equal_slopes (a : ℝ) :
  (∀ x y, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = -7 + a) →
  a = 3 := sorry

end NUMINAMATH_GPT_parallel_lines_equal_slopes_l238_23898


namespace NUMINAMATH_GPT_min_cards_needed_l238_23865

/-- 
On a table, there are five types of number cards: 1, 3, 5, 7, and 9, with 30 cards of each type. 
Prove that the minimum number of cards required to ensure that the sum of the drawn card numbers 
can represent all integers from 1 to 200 is 26.
-/
theorem min_cards_needed : ∀ (cards_1 cards_3 cards_5 cards_7 cards_9 : ℕ), 
  cards_1 = 30 → cards_3 = 30 → cards_5 = 30 → cards_7 = 30 → cards_9 = 30 → 
  ∃ n, (n = 26) ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ 200 → 
    ∃ a b c d e, 
      a ≤ cards_1 ∧ b ≤ cards_3 ∧ c ≤ cards_5 ∧ d ≤ cards_7 ∧ e ≤ cards_9 ∧ 
      k = a * 1 + b * 3 + c * 5 + d * 7 + e * 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_cards_needed_l238_23865


namespace NUMINAMATH_GPT_divisible_by_6_l238_23894

theorem divisible_by_6 (n : ℕ) : 6 ∣ ((n - 1) * n * (n^3 + 1)) := sorry

end NUMINAMATH_GPT_divisible_by_6_l238_23894


namespace NUMINAMATH_GPT_number_of_schools_l238_23826

def yellow_balloons := 3414
def additional_black_balloons := 1762
def balloons_per_school := 859

def black_balloons := yellow_balloons + additional_black_balloons
def total_balloons := yellow_balloons + black_balloons

theorem number_of_schools : total_balloons / balloons_per_school = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_schools_l238_23826


namespace NUMINAMATH_GPT_tan_equals_three_l238_23892

variable (α : ℝ)

theorem tan_equals_three : 
  (Real.tan α = 3) → (1 / (Real.sin α * Real.sin α + 2 * Real.sin α * Real.cos α) = 2 / 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tan_equals_three_l238_23892


namespace NUMINAMATH_GPT_find_x_l238_23849

variables {x y z d e f : ℝ}
variables (h1 : xy / (x + 2 * y) = d)
variables (h2 : xz / (2 * x + z) = e)
variables (h3 : yz / (y + 2 * z) = f)

theorem find_x :
  x = 3 * d * e * f / (d * e - 2 * d * f + e * f) :=
sorry

end NUMINAMATH_GPT_find_x_l238_23849


namespace NUMINAMATH_GPT_range_of_m_l238_23807

/-- The range of the real number m such that the equation x^2/m + y^2/(2m - 1) = 1 represents an ellipse with foci on the x-axis is (1/2, 1). -/
theorem range_of_m (m : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, x^2 / m + y^2 / (2 * m - 1) = 1 → x^2 / a^2 + y^2 / b^2 = 1 ∧ b^2 < a^2))
  ↔ 1 / 2 < m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l238_23807


namespace NUMINAMATH_GPT_parts_per_day_system_l238_23834

variable (x y : ℕ)

def personA_parts_per_day (x : ℕ) : ℕ := x
def personB_parts_per_day (y : ℕ) : ℕ := y

-- First condition
def condition1 (x y : ℕ) : Prop :=
  6 * x = 5 * y

-- Second condition
def condition2 (x y : ℕ) : Prop :=
  30 + 4 * x = 4 * y - 10

theorem parts_per_day_system (x y : ℕ) :
  condition1 x y ∧ condition2 x y :=
sorry

end NUMINAMATH_GPT_parts_per_day_system_l238_23834


namespace NUMINAMATH_GPT_square_lawn_side_length_l238_23813

theorem square_lawn_side_length (length width : ℕ) (h_length : length = 18) (h_width : width = 8) : 
  ∃ x : ℕ, x * x = length * width ∧ x = 12 := by
  -- Assume the necessary definitions and theorems to build the proof
  sorry

end NUMINAMATH_GPT_square_lawn_side_length_l238_23813


namespace NUMINAMATH_GPT_ways_to_append_digit_divisible_by_3_l238_23825

-- Define a function that takes a digit and checks if it can make the number divisible by 3
def is_divisible_by_3 (n : ℕ) (d : ℕ) : Bool :=
  (n * 10 + d) % 3 == 0

-- Theorem stating that there are 4 ways to append a digit to make the number divisible by 3
theorem ways_to_append_digit_divisible_by_3 
  (n : ℕ) 
  (divisible_by_9_conditions : (n * 10 + 0) % 9 = 0 ∧ (n * 10 + 9) % 9 = 0) : 
  ∃ (ds : Finset ℕ), ds.card = 4 ∧ ∀ d ∈ ds, is_divisible_by_3 n d :=
  sorry

end NUMINAMATH_GPT_ways_to_append_digit_divisible_by_3_l238_23825


namespace NUMINAMATH_GPT_core_temperature_calculation_l238_23884

-- Define the core temperature of the Sun, given in degrees Celsius
def T_Sun : ℝ := 19200000

-- Define the multiple factor
def factor : ℝ := 312.5

-- The expected result in scientific notation
def expected_temperature : ℝ := 6.0 * (10 ^ 9)

-- Prove that the calculated temperature is equal to the expected temperature
theorem core_temperature_calculation : (factor * T_Sun) = expected_temperature := by
  sorry

end NUMINAMATH_GPT_core_temperature_calculation_l238_23884


namespace NUMINAMATH_GPT_base7_65432_to_dec_is_16340_l238_23833

def base7_to_dec (n : ℕ) : ℕ :=
  6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0

theorem base7_65432_to_dec_is_16340 : base7_to_dec 65432 = 16340 :=
by
  sorry

end NUMINAMATH_GPT_base7_65432_to_dec_is_16340_l238_23833


namespace NUMINAMATH_GPT_rectangle_diagonals_equiv_positive_even_prime_equiv_l238_23808

-- Definitions based on problem statement (1)
def is_rectangle (q : Quadrilateral) : Prop := sorry -- "q is a rectangle"
def diagonals_equal_and_bisect (q : Quadrilateral) : Prop := sorry -- "the diagonals of q are equal and bisect each other"

-- Problem statement (1)
theorem rectangle_diagonals_equiv (q : Quadrilateral) :
  (is_rectangle q → diagonals_equal_and_bisect q) ∧
  (diagonals_equal_and_bisect q → is_rectangle q) ∧
  (¬ is_rectangle q → ¬ diagonals_equal_and_bisect q) ∧
  (¬ diagonals_equal_and_bisect q → ¬ is_rectangle q) :=
sorry

-- Definitions based on problem statement (2)
def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0
def is_prime (n : ℕ) : Prop := sorry -- "n is a prime number"

-- Problem statement (2)
theorem positive_even_prime_equiv (n : ℕ) :
  (is_positive_even n → ¬ is_prime n) ∧
  ((¬ is_prime n → is_positive_even n) = False) ∧
  ((¬ is_positive_even n → is_prime n) = False) ∧
  ((is_prime n → ¬ is_positive_even n) = False) :=
sorry

end NUMINAMATH_GPT_rectangle_diagonals_equiv_positive_even_prime_equiv_l238_23808


namespace NUMINAMATH_GPT_bowling_ball_weight_l238_23824

theorem bowling_ball_weight (b k : ℕ) (h1 : 8 * b = 4 * k) (h2 : 3 * k = 84) : b = 14 := by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l238_23824


namespace NUMINAMATH_GPT_valid_arrangement_after_removal_l238_23827

theorem valid_arrangement_after_removal (n : ℕ) (m : ℕ → ℕ) :
  (∀ i j, i ≠ j → m i ≠ m j → ¬ (i < n ∧ j < n))
  → (∀ i, i < n → m i ≥ m (i + 1))
  → ∃ (m' : ℕ → ℕ), (∀ i, i < n.pred → m' i = m (i + 1) - 1 ∨ m' i = m (i + 1))
    ∧ (∀ i, m' i ≥ m' (i + 1))
    ∧ (∀ i j, i ≠ j → i < n.pred → j < n.pred → ¬ (m' i = m' j ∧ m' i = m (i + 1))) := sorry

end NUMINAMATH_GPT_valid_arrangement_after_removal_l238_23827


namespace NUMINAMATH_GPT_major_premise_wrong_l238_23895

-- Definition of the problem conditions and the proof goal
theorem major_premise_wrong :
  (∀ a : ℝ, |a| > 0) ↔ false :=
by {
  sorry  -- the proof goes here but is omitted as per the instructions
}

end NUMINAMATH_GPT_major_premise_wrong_l238_23895


namespace NUMINAMATH_GPT_time_to_cross_pole_is_correct_l238_23821

-- Define the conversion factor to convert km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : ℕ) : ℕ := speed_km_per_hr * 1000 / 3600

-- Define the speed of the train in m/s
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s 216

-- Define the length of the train
def train_length_m : ℕ := 480

-- Define the time to cross an electric pole
def time_to_cross_pole : ℕ := train_length_m / train_speed_m_per_s

-- Theorem stating that the computed time to cross the pole is 8 seconds
theorem time_to_cross_pole_is_correct :
  time_to_cross_pole = 8 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_pole_is_correct_l238_23821


namespace NUMINAMATH_GPT_meat_per_slice_is_22_l238_23835

noncomputable def piecesOfMeatPerSlice : ℕ :=
  let pepperoni := 30
  let ham := 2 * pepperoni
  let sausage := pepperoni + 12
  let totalMeat := pepperoni + ham + sausage
  let slices := 6
  totalMeat / slices

theorem meat_per_slice_is_22 : piecesOfMeatPerSlice = 22 :=
by
  -- Here would be the proof (not required in the task)
  sorry

end NUMINAMATH_GPT_meat_per_slice_is_22_l238_23835


namespace NUMINAMATH_GPT_convert_decimal_to_fraction_l238_23828

theorem convert_decimal_to_fraction : (0.38 : ℚ) = 19 / 50 :=
by
  sorry

end NUMINAMATH_GPT_convert_decimal_to_fraction_l238_23828


namespace NUMINAMATH_GPT_MeganMarkers_l238_23846

def initialMarkers : Nat := 217
def additionalMarkers : Nat := 109
def totalMarkers : Nat := initialMarkers + additionalMarkers

theorem MeganMarkers : totalMarkers = 326 := by
    sorry

end NUMINAMATH_GPT_MeganMarkers_l238_23846


namespace NUMINAMATH_GPT_joel_strawberries_area_l238_23830

-- Define the conditions
def garden_area : ℕ := 64
def fruit_fraction : ℚ := 1 / 2
def strawberry_fraction : ℚ := 1 / 4

-- Define the desired conclusion
def strawberries_area : ℕ := 8

-- State the theorem
theorem joel_strawberries_area 
  (H1 : garden_area = 64) 
  (H2 : fruit_fraction = 1 / 2) 
  (H3 : strawberry_fraction = 1 / 4)
  : garden_area * fruit_fraction * strawberry_fraction = strawberries_area := 
sorry

end NUMINAMATH_GPT_joel_strawberries_area_l238_23830


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l238_23839

theorem problem_part1
  (A : ℤ → ℤ → ℤ)
  (B : ℤ → ℤ → ℤ)
  (x y : ℤ)
  (hA : A x y = 2 * x ^ 2 + 4 * x * y - 2 * x - 3)
  (hB : B x y = -x^2 + x*y + 2) :
  3 * A x y - 2 * (A x y + 2 * B x y) = 6 * x ^ 2 - 2 * x - 11 := by
  sorry

theorem problem_part2
  (A : ℤ → ℤ → ℤ)
  (B : ℤ → ℤ → ℤ)
  (y : ℤ)
  (H : ∀ x, B x y + (1 / 2) * A x y = C) :
  y = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l238_23839


namespace NUMINAMATH_GPT_subset_of_primes_is_all_primes_l238_23882

theorem subset_of_primes_is_all_primes
  (P : Set ℕ)
  (M : Set ℕ)
  (hP : ∀ n, n ∈ P ↔ Nat.Prime n)
  (hM : ∀ S : Finset ℕ, (∀ p ∈ S, p ∈ M) → ∀ p, p ∣ (Finset.prod S id + 1) → p ∈ M) :
  M = P :=
sorry

end NUMINAMATH_GPT_subset_of_primes_is_all_primes_l238_23882


namespace NUMINAMATH_GPT_upper_bound_expression_l238_23874

theorem upper_bound_expression (n : ℤ) (U : ℤ) :
  (∀ n, 4 * n + 7 > 1 ∧ 4 * n + 7 < U → ∃ k : ℤ, k = 50) →
  U = 204 :=
by
  sorry

end NUMINAMATH_GPT_upper_bound_expression_l238_23874


namespace NUMINAMATH_GPT_max_value_fraction_diff_l238_23818

noncomputable def max_fraction_diff (a b : ℝ) : ℝ :=
  1 / a - 1 / b

theorem max_value_fraction_diff (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : 4 * a - b ≥ 2) :
  max_fraction_diff a b ≤ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_diff_l238_23818


namespace NUMINAMATH_GPT_find_f3_l238_23840

theorem find_f3 (f : ℚ → ℚ)
  (h : ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x) / x = x^3) :
  f 3 = 7753 / 729 :=
sorry

end NUMINAMATH_GPT_find_f3_l238_23840


namespace NUMINAMATH_GPT_family_eggs_count_l238_23885

theorem family_eggs_count : 
  ∀ (initial_eggs parent_use child_use : ℝ) (chicken1 chicken2 chicken3 chicken4 : ℝ), 
    initial_eggs = 25 →
    parent_use = 7.5 + 2.5 →
    chicken1 = 2.5 →
    chicken2 = 3 →
    chicken3 = 4.5 →
    chicken4 = 1 →
    child_use = 1.5 + 0.5 →
    (initial_eggs - parent_use + (chicken1 + chicken2 + chicken3 + chicken4) - child_use) = 24 :=
by
  intros initial_eggs parent_use child_use chicken1 chicken2 chicken3 chicken4 
         h_initial_eggs h_parent_use h_chicken1 h_chicken2 h_chicken3 h_chicken4 h_child_use
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_family_eggs_count_l238_23885


namespace NUMINAMATH_GPT_simplify_expression_l238_23842

theorem simplify_expression : 
  (2 * Real.sqrt 7) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 6) = 
  (2 * Real.sqrt 14 + 8 * Real.sqrt 21 + 2 * Real.sqrt 42 + 8 * Real.sqrt 63) / 23 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l238_23842


namespace NUMINAMATH_GPT_incorrect_statement_for_function_l238_23852

theorem incorrect_statement_for_function (x : ℝ) (h : x > 0) : 
  ¬(∀ x₁ x₂ : ℝ, (x₁ > 0) → (x₂ > 0) → (x₁ < x₂) → (6 / x₁ < 6 / x₂)) := 
sorry

end NUMINAMATH_GPT_incorrect_statement_for_function_l238_23852


namespace NUMINAMATH_GPT_pats_password_length_l238_23871

-- Definitions based on conditions
def num_lowercase_letters := 8
def num_uppercase_numbers := num_lowercase_letters / 2
def num_symbols := 2

-- Translate the math proof problem to Lean 4 statement
theorem pats_password_length : 
  num_lowercase_letters + num_uppercase_numbers + num_symbols = 14 := by
  sorry

end NUMINAMATH_GPT_pats_password_length_l238_23871


namespace NUMINAMATH_GPT_sharon_trip_distance_l238_23850

noncomputable section

variable (x : ℝ)

def sharon_original_speed (x : ℝ) := x / 200

def sharon_reduced_speed (x : ℝ) := (x / 200) - 1 / 2

def time_before_traffic (x : ℝ) := (x / 2) / (sharon_original_speed x)

def time_after_traffic (x : ℝ) := (x / 2) / (sharon_reduced_speed x)

theorem sharon_trip_distance : 
  (time_before_traffic x) + (time_after_traffic x) = 300 → x = 200 := 
by
  sorry

end NUMINAMATH_GPT_sharon_trip_distance_l238_23850


namespace NUMINAMATH_GPT_shape_is_spiral_l238_23801

-- Assume cylindrical coordinates and constants.
variables (c : ℝ)
-- Define cylindrical coordinate properties.
variables (r θ z : ℝ)

-- Define the equation rθ = c.
def cylindrical_equation : Prop := r * θ = c

theorem shape_is_spiral (h : cylindrical_equation c r θ):
  ∃ f : ℝ → ℝ, ∀ θ > 0, r = f θ ∧ (∀ θ₁ θ₂, θ₁ < θ₂ ↔ f θ₁ > f θ₂) :=
sorry

end NUMINAMATH_GPT_shape_is_spiral_l238_23801


namespace NUMINAMATH_GPT_gcd_factorial_8_6_squared_l238_23877

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_GPT_gcd_factorial_8_6_squared_l238_23877


namespace NUMINAMATH_GPT_student_average_less_than_actual_average_l238_23805

variable {a b c : ℝ}

theorem student_average_less_than_actual_average (h : a < b) (h2 : b < c) :
  (a + (b + c) / 2) / 2 < (a + b + c) / 3 :=
by
  sorry

end NUMINAMATH_GPT_student_average_less_than_actual_average_l238_23805


namespace NUMINAMATH_GPT_f_neg_m_equals_neg_8_l238_23855

def f (x : ℝ) : ℝ := x^5 + x^3 + 1

theorem f_neg_m_equals_neg_8 (m : ℝ) (h : f m = 10) : f (-m) = -8 :=
by
  sorry

end NUMINAMATH_GPT_f_neg_m_equals_neg_8_l238_23855


namespace NUMINAMATH_GPT_f_100_eq_11_l238_23878
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def f (n : ℕ) : ℕ := sum_of_digits (n^2 + 1)

def f_iter : ℕ → ℕ → ℕ
| 0,    n => f n
| k+1,  n => f (f_iter k n)

theorem f_100_eq_11 (n : ℕ) (h : n = 1990) : f_iter 100 n = 11 := by
  sorry

end NUMINAMATH_GPT_f_100_eq_11_l238_23878


namespace NUMINAMATH_GPT_range_of_b_l238_23853

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a x b : ℝ) (ha : -1 ≤ a) (ha' : a < 0) (hx : 0 < x) (hx' : x ≤ 1) 
  (h : f x a < b) : -3 / 2 < b := 
sorry

end NUMINAMATH_GPT_range_of_b_l238_23853


namespace NUMINAMATH_GPT_regular_polygon_sides_l238_23873

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 12) : n = 30 := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l238_23873


namespace NUMINAMATH_GPT_expected_final_set_size_l238_23811

noncomputable def final_expected_set_size : ℚ :=
  let n := 8
  let initial_size := 255
  let steps := initial_size - 1
  n * (2^7 / initial_size)

theorem expected_final_set_size :
  final_expected_set_size = 1024 / 255 :=
by
  sorry

end NUMINAMATH_GPT_expected_final_set_size_l238_23811


namespace NUMINAMATH_GPT_boys_play_football_l238_23819

theorem boys_play_football (total_boys basketball_players neither_players both_players : ℕ)
    (h_total : total_boys = 22)
    (h_basketball : basketball_players = 13)
    (h_neither : neither_players = 3)
    (h_both : both_players = 18) : total_boys - neither_players - both_players + (both_players - basketball_players) = 19 :=
by
  sorry

end NUMINAMATH_GPT_boys_play_football_l238_23819


namespace NUMINAMATH_GPT_Kiarra_age_l238_23800

variable (Kiarra Bea Job Figaro Harry : ℕ)

theorem Kiarra_age 
  (h1 : Kiarra = 2 * Bea)
  (h2 : Job = 3 * Bea)
  (h3 : Figaro = Job + 7)
  (h4 : Harry = Figaro / 2)
  (h5 : Harry = 26) : 
  Kiarra = 30 := sorry

end NUMINAMATH_GPT_Kiarra_age_l238_23800


namespace NUMINAMATH_GPT_units_digit_fraction_l238_23897

-- Given conditions
def numerator : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 1500
def simplified_fraction : ℕ := 2^5 * 3 * 31 * 33 * 17 * 7

-- Statement of the proof goal
theorem units_digit_fraction :
  (simplified_fraction) % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_fraction_l238_23897


namespace NUMINAMATH_GPT_servings_made_l238_23866

noncomputable def chickpeas_per_can := 16 -- ounces in one can
noncomputable def ounces_per_serving := 6 -- ounces needed per serving
noncomputable def total_cans := 8 -- total cans Thomas buys

theorem servings_made : (total_cans * chickpeas_per_can) / ounces_per_serving = 21 :=
by
  sorry

end NUMINAMATH_GPT_servings_made_l238_23866


namespace NUMINAMATH_GPT_part1_part2_l238_23893

def partsProcessedA : ℕ → ℕ
| 0 => 10
| (n + 1) => if n = 0 then 8 else partsProcessedA n - 2

def partsProcessedB : ℕ → ℕ
| 0 => 8
| (n + 1) => if n = 0 then 7 else partsProcessedB n - 1

def partsProcessedLineB_A (n : ℕ) := 7 * n
def partsProcessedLineB_B (n : ℕ) := 8 * n

def maxSetsIn14Days : ℕ := 
  let aLineA := 2 * (10 + 8 + 6) + (10 + 8)
  let aLineB := 2 * (8 + 7 + 6) + (8 + 8)
  min aLineA aLineB

theorem part1 :
  partsProcessedA 0 + partsProcessedA 1 + partsProcessedA 2 = 24 := 
by sorry

theorem part2 :
  maxSetsIn14Days = 106 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l238_23893


namespace NUMINAMATH_GPT_find_y_l238_23879

theorem find_y (a b c x : ℝ) (p q r y: ℝ) (hx : x ≠ 1) 
  (h₁ : (Real.log a) / p = Real.log x) 
  (h₂ : (Real.log b) / q = Real.log x) 
  (h₃ : (Real.log c) / r = Real.log x)
  (h₄ : (b^3) / (a^2 * c) = x^y) : 
  y = 3 * q - 2 * p - r := 
by {
  sorry
}

end NUMINAMATH_GPT_find_y_l238_23879


namespace NUMINAMATH_GPT_intersection_eq_0_l238_23817

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 3, 4}

theorem intersection_eq_0 : M ∩ N = {0} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_0_l238_23817


namespace NUMINAMATH_GPT_num_of_integers_l238_23851

theorem num_of_integers (n : ℤ) (h : -1000 ≤ n ∧ n ≤ 1000) (h1 : 1 < 4 * n + 7) (h2 : 4 * n + 7 < 150) : 
  (∃ N : ℕ, N = 37) :=
by
  sorry

end NUMINAMATH_GPT_num_of_integers_l238_23851


namespace NUMINAMATH_GPT_train_speed_l238_23815

theorem train_speed (len_train len_bridge time : ℝ)
  (h1 : len_train = 100)
  (h2 : len_bridge = 180)
  (h3 : time = 27.997760179185665) :
  (len_train + len_bridge) / time * 3.6 = 36 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l238_23815
