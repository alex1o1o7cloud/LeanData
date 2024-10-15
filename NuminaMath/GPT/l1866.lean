import Mathlib

namespace NUMINAMATH_GPT_max_integer_value_l1866_186629

theorem max_integer_value (x : ℝ) : ∃ (m : ℤ), m = 53 ∧ ∀ y : ℝ, (1 + 13 / (3 * y^2 + 9 * y + 7) ≤ m) := 
sorry

end NUMINAMATH_GPT_max_integer_value_l1866_186629


namespace NUMINAMATH_GPT_tammy_speed_second_day_l1866_186675

theorem tammy_speed_second_day :
  ∀ (v1 t1 v2 t2 : ℝ), 
    t1 + t2 = 14 →
    t2 = t1 - 2 →
    v2 = v1 + 0.5 →
    v1 * t1 + v2 * t2 = 52 →
    v2 = 4 :=
by
  intros v1 t1 v2 t2 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_tammy_speed_second_day_l1866_186675


namespace NUMINAMATH_GPT_algebra_problem_l1866_186687

variable (a : ℝ)

-- Condition: Given (a + 1/a)^3 = 4
def condition : Prop := (a + 1/a)^3 = 4

-- Statement: Prove a^4 + 1/a^4 = -158/81
theorem algebra_problem (h : condition a) : a^4 + 1/a^4 = -158/81 := 
sorry

end NUMINAMATH_GPT_algebra_problem_l1866_186687


namespace NUMINAMATH_GPT_find_N_l1866_186660

variable (N : ℚ)
variable (p : ℚ)

def ball_probability_same_color 
  (green1 : ℚ) (total1 : ℚ) 
  (green2 : ℚ) (blue2 : ℚ) 
  (p : ℚ) : Prop :=
  (green1/total1) * (green2 / (green2 + blue2)) + 
  ((total1 - green1) / total1) * (blue2 / (green2 + blue2)) = p

theorem find_N :
  p = 0.65 → 
  ball_probability_same_color 5 12 20 N p → 
  N = 280 / 311 := 
by
  sorry

end NUMINAMATH_GPT_find_N_l1866_186660


namespace NUMINAMATH_GPT_expected_sample_size_l1866_186611

noncomputable def highSchoolTotalStudents (f s j : ℕ) : ℕ :=
  f + s + j

noncomputable def expectedSampleSize (total : ℕ) (p : ℝ) : ℝ :=
  total * p

theorem expected_sample_size :
  let f := 400
  let s := 320
  let j := 280
  let p := 0.2
  let total := highSchoolTotalStudents f s j
  expectedSampleSize total p = 200 :=
by
  sorry

end NUMINAMATH_GPT_expected_sample_size_l1866_186611


namespace NUMINAMATH_GPT_bins_of_soup_l1866_186630

theorem bins_of_soup (total_bins : ℝ) (bins_of_vegetables : ℝ) (bins_of_pasta : ℝ) 
(h1 : total_bins = 0.75) (h2 : bins_of_vegetables = 0.125) (h3 : bins_of_pasta = 0.5) :
  total_bins - (bins_of_vegetables + bins_of_pasta) = 0.125 := by
  -- proof
  sorry

end NUMINAMATH_GPT_bins_of_soup_l1866_186630


namespace NUMINAMATH_GPT_distance_to_place_l1866_186666

theorem distance_to_place 
  (row_speed_still_water : ℝ) 
  (current_speed : ℝ) 
  (headwind_speed : ℝ) 
  (tailwind_speed : ℝ) 
  (total_trip_time : ℝ) 
  (htotal_trip_time : total_trip_time = 15) 
  (hrow_speed_still_water : row_speed_still_water = 10) 
  (hcurrent_speed : current_speed = 2) 
  (hheadwind_speed : headwind_speed = 4) 
  (htailwind_speed : tailwind_speed = 4) :
  ∃ (D : ℝ), D = 48 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_place_l1866_186666


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l1866_186654

theorem geometric_sequence_fourth_term
  (a₁ a₅ : ℕ)
  (r : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₅ = 2187)
  (h₃ : a₅ = a₁ * r ^ 4) :
  a₁ * r ^ 3 = 2187 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l1866_186654


namespace NUMINAMATH_GPT_wire_pieces_difference_l1866_186678

theorem wire_pieces_difference (L1 L2 : ℝ) (H1 : L1 = 14) (H2 : L2 = 16) : L2 - L1 = 2 :=
by
  rw [H1, H2]
  norm_num

end NUMINAMATH_GPT_wire_pieces_difference_l1866_186678


namespace NUMINAMATH_GPT_cyclic_identity_l1866_186696

theorem cyclic_identity (a b c : ℝ) :
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) =
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) ∧
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) =
  c * (c - b)^2 + a * (a - b)^2 - (c - b) * (a - b) * (c + a - b) := by
sorry

end NUMINAMATH_GPT_cyclic_identity_l1866_186696


namespace NUMINAMATH_GPT_experimental_fertilizer_height_is_correct_l1866_186655

/-- Define the static heights and percentages for each plant's growth conditions. -/
def control_plant_height : ℝ := 36
def bone_meal_multiplier : ℝ := 1.25
def cow_manure_multiplier : ℝ := 2
def experimental_fertilizer_multiplier : ℝ := 1.5

/-- Define each plant's height based on the given multipliers and conditions. -/
def bone_meal_plant_height : ℝ := bone_meal_multiplier * control_plant_height
def cow_manure_plant_height : ℝ := cow_manure_multiplier * bone_meal_plant_height
def experimental_fertilizer_plant_height : ℝ := experimental_fertilizer_multiplier * cow_manure_plant_height

/-- Proof that the height of the experimental fertilizer plant is 135 inches. -/
theorem experimental_fertilizer_height_is_correct :
  experimental_fertilizer_plant_height = 135 := by
    sorry

end NUMINAMATH_GPT_experimental_fertilizer_height_is_correct_l1866_186655


namespace NUMINAMATH_GPT_two_talents_students_l1866_186644

-- Definitions and conditions
def total_students : ℕ := 120
def cannot_sing : ℕ := 50
def cannot_dance : ℕ := 75
def cannot_act : ℕ := 35

-- Definitions based on conditions
def can_sing : ℕ := total_students - cannot_sing
def can_dance : ℕ := total_students - cannot_dance
def can_act : ℕ := total_students - cannot_act

-- The main theorem statement
theorem two_talents_students : can_sing + can_dance + can_act - total_students = 80 :=
by
  -- substituting actual numbers to prove directly
  have h_can_sing : can_sing = 70 := rfl
  have h_can_dance : can_dance = 45 := rfl
  have h_can_act : can_act = 85 := rfl
  sorry

end NUMINAMATH_GPT_two_talents_students_l1866_186644


namespace NUMINAMATH_GPT_solve_equation_l1866_186606

theorem solve_equation (x : ℝ) : (x + 2) * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 1) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1866_186606


namespace NUMINAMATH_GPT_fill_blank_1_fill_blank_2_l1866_186676

theorem fill_blank_1 (x : ℤ) (h : 1 + x = -10) : x = -11 := sorry

theorem fill_blank_2 (y : ℝ) (h : y - 4.5 = -4.5) : y = 0 := sorry

end NUMINAMATH_GPT_fill_blank_1_fill_blank_2_l1866_186676


namespace NUMINAMATH_GPT_find_ab_from_conditions_l1866_186620

theorem find_ab_from_conditions (a b : ℝ) (h1 : a^2 + b^2 = 5) (h2 : a + b = 3) : a * b = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_ab_from_conditions_l1866_186620


namespace NUMINAMATH_GPT_analysis_hours_l1866_186635

theorem analysis_hours (n t : ℕ) (h1 : n = 206) (h2 : t = 1) : n * t = 206 := by
  sorry

end NUMINAMATH_GPT_analysis_hours_l1866_186635


namespace NUMINAMATH_GPT_rectangle_length_l1866_186607

theorem rectangle_length {width length : ℝ} (h1 : (3 : ℝ) * 3 = 9) (h2 : width = 3) (h3 : width * length = 9) : 
  length = 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l1866_186607


namespace NUMINAMATH_GPT_purchase_price_l1866_186664

theorem purchase_price (marked_price : ℝ) (discount_rate profit_rate x : ℝ)
  (h1 : marked_price = 126)
  (h2 : discount_rate = 0.05)
  (h3 : profit_rate = 0.05)
  (h4 : marked_price * (1 - discount_rate) - x = x * profit_rate) : 
  x = 114 :=
by 
  sorry

end NUMINAMATH_GPT_purchase_price_l1866_186664


namespace NUMINAMATH_GPT_range_of_a_l1866_186632

noncomputable def isNotPurelyImaginary (a : ℝ) : Prop :=
  let re := a^2 - a - 2
  re ≠ 0

theorem range_of_a (a : ℝ) (h : isNotPurelyImaginary a) : a ≠ -1 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1866_186632


namespace NUMINAMATH_GPT_Bobby_paycheck_final_amount_l1866_186616

theorem Bobby_paycheck_final_amount :
  let salary := 450
  let federal_tax := (1 / 3 : ℚ) * salary
  let state_tax := 0.08 * salary
  let health_insurance := 50
  let life_insurance := 20
  let city_fee := 10
  let total_deductions := federal_tax + state_tax + health_insurance + life_insurance + city_fee
  salary - total_deductions = 184 :=
by
  -- We put sorry here to skip the proof step
  sorry

end NUMINAMATH_GPT_Bobby_paycheck_final_amount_l1866_186616


namespace NUMINAMATH_GPT_quadratic_y1_gt_y2_l1866_186609

theorem quadratic_y1_gt_y2 (a b c y1 y2 : ℝ) (h_a_pos : a > 0) (h_sym : ∀ x, a * (x - 1)^2 + c = a * (1 - x)^2 + c) (h1 : y1 = a * (-1)^2 + b * (-1) + c) (h2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
sorry

end NUMINAMATH_GPT_quadratic_y1_gt_y2_l1866_186609


namespace NUMINAMATH_GPT_solve_for_n_l1866_186680

theorem solve_for_n (n : ℤ) (h : (5/4 : ℚ) * n + (5/4 : ℚ) = n) : n = -5 := by
    sorry

end NUMINAMATH_GPT_solve_for_n_l1866_186680


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1866_186688

theorem sum_of_x_and_y (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hprod : x * y = 555) : x + y = 52 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1866_186688


namespace NUMINAMATH_GPT_duty_person_C_l1866_186653

/-- Given amounts of money held by three persons and a total custom duty,
    prove that the duty person C should pay is 17 when payments are proportional. -/
theorem duty_person_C (money_A money_B money_C total_duty : ℕ) (total_money : ℕ)
  (hA : money_A = 560) (hB : money_B = 350) (hC : money_C = 180) (hD : total_duty = 100)
  (hT : total_money = money_A + money_B + money_C) :
  total_duty * money_C / total_money = 17 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_duty_person_C_l1866_186653


namespace NUMINAMATH_GPT_function_properties_l1866_186602

theorem function_properties
  (f : ℝ → ℝ)
  (h1 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : ∀ x, f (x - t) = f (x + t)) 
  (h3_even : ∀ x, f (-x) = f x)
  (h3_decreasing : ∀ x1 x2, x1 < x2 ∧ x2 < 0 → f x1 > f x2)
  (h3_at_neg2 : f (-2) = 0)
  (h4_odd : ∀ x, f (-x) = -f x) : 
  ((∀ x1 x2, x1 < x2 → f x1 > f x2) ∧
   (¬∀ x, (f x > 0) ↔ (-2 < x ∧ x < 2)) ∧
   (∀ x, f (x) * f (|x|) = - f (-x) * f |x|) ∧
   (¬∀ x, f (x) = f (x + 2 * t))) :=
by 
  sorry

end NUMINAMATH_GPT_function_properties_l1866_186602


namespace NUMINAMATH_GPT_volume_parallelepiped_l1866_186603

open Real

theorem volume_parallelepiped :
  ∃ (a h : ℝ), 
    let S_base := (4 : ℝ)
    let AB := a
    let AD := 2 * a
    let lateral_face1 := (6 : ℝ)
    let lateral_face2 := (12 : ℝ)
    (AB * h = lateral_face1) ∧
    (AD * h = lateral_face2) ∧
    (1 / 2 * AD * S_base = AB * (1 / 2 * AD)) ∧ 
    (AB^2 + AD^2 - 2 * AB * AD * (cos (π / 6)) = S_base) ∧
    (a = 2) ∧
    (h = 3) ∧ 
    (S_base * h = 12) :=
sorry

end NUMINAMATH_GPT_volume_parallelepiped_l1866_186603


namespace NUMINAMATH_GPT_problem_condition_problem_statement_l1866_186604

noncomputable def a : ℕ → ℕ 
| 0     => 2
| (n+1) => 3 * a n

noncomputable def S : ℕ → ℕ
| 0     => 0
| (n+1) => S n + a n

theorem problem_condition : ∀ n, 3 * a n - 2 * S n = 2 :=
by
  sorry

theorem problem_statement (n : ℕ) (h : ∀ n, 3 * a n - 2 * S n = 2) :
  (S (n+1))^2 - (S n) * (S (n+2)) = 4 * 3^n :=
by
  sorry

end NUMINAMATH_GPT_problem_condition_problem_statement_l1866_186604


namespace NUMINAMATH_GPT_ratio_sum_ineq_l1866_186633

theorem ratio_sum_ineq 
  (a b α β : ℝ) 
  (hαβ : 0 < α ∧ 0 < β) 
  (h_range : α ≤ a ∧ a ≤ β ∧ α ≤ b ∧ b ≤ β) : 
  (b / a + a / b ≤ β / α + α / β) ∧ 
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β ∨ a = β ∧ b = α)) :=
by
  sorry

end NUMINAMATH_GPT_ratio_sum_ineq_l1866_186633


namespace NUMINAMATH_GPT_quadratic_function_min_value_l1866_186669

theorem quadratic_function_min_value (x : ℝ) (y : ℝ) :
  (y = x^2 - 2 * x + 6) →
  (∃ x_min, x_min = 1 ∧ y = (1 : ℝ)^2 - 2 * (1 : ℝ) + 6 ∧ (∀ x, y ≥ x^2 - 2 * x + 6)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_min_value_l1866_186669


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1866_186601

theorem common_ratio_of_geometric_sequence (a_1 a_2 a_3 a_4 q : ℝ)
  (h1 : a_1 * a_2 * a_3 = 27)
  (h2 : a_2 + a_4 = 30)
  (geometric_sequence : a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3) :
  q = 3 ∨ q = -3 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1866_186601


namespace NUMINAMATH_GPT_chocolate_cost_l1866_186626

theorem chocolate_cost (Ccb Cc : ℝ) (h1 : Ccb = 6) (h2 : Ccb = Cc + 3) : Cc = 3 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_cost_l1866_186626


namespace NUMINAMATH_GPT_complement_of_A_is_correct_l1866_186628

-- Define the universal set U and the set A.
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U.
def A_complement : Set ℕ := {x ∈ U | x ∉ A}

-- The theorem statement that the complement of A in U is {2, 4}.
theorem complement_of_A_is_correct : A_complement = {2, 4} :=
sorry

end NUMINAMATH_GPT_complement_of_A_is_correct_l1866_186628


namespace NUMINAMATH_GPT_B_pow_five_l1866_186683

def B : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![2, 3], ![4, 6]]
  
theorem B_pow_five : 
  B^5 = (4096 : ℝ) • B + (0 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end NUMINAMATH_GPT_B_pow_five_l1866_186683


namespace NUMINAMATH_GPT_points_on_hyperbola_l1866_186615

theorem points_on_hyperbola {s : ℝ} :
  let x := Real.exp s - Real.exp (-s)
  let y := 5 * (Real.exp s + Real.exp (-s))
  (y^2 / 100 - x^2 / 4 = 1) :=
by
  sorry

end NUMINAMATH_GPT_points_on_hyperbola_l1866_186615


namespace NUMINAMATH_GPT_radius_range_l1866_186651

-- Conditions:
-- r1 is the radius of circle O1
-- r2 is the radius of circle O2
-- d is the distance between centers of circles O1 and O2
-- PO1 is the distance from a point P on circle O2 to the center of circle O1

variables (r1 r2 d PO1 : ℝ)

-- Given r1 = 1, d = 5, PO1 = 2
axiom r1_def : r1 = 1
axiom d_def : d = 5
axiom PO1_def : PO1 = 2

-- To prove: 3 ≤ r2 ≤ 7
theorem radius_range (r2 : ℝ) (h : d = 5 ∧ r1 = 1 ∧ PO1 = 2 ∧ (∃ P : ℝ, P = r2)) : 3 ≤ r2 ∧ r2 ≤ 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_radius_range_l1866_186651


namespace NUMINAMATH_GPT_cost_of_fencing_theorem_l1866_186698

noncomputable def cost_of_fencing (area : ℝ) (ratio_length_width : ℝ) (cost_per_meter_paise : ℝ) : ℝ :=
  let width := (area / (ratio_length_width * 2 * ratio_length_width * 3)).sqrt
  let length := ratio_length_width * 3 * width
  let perimeter := 2 * (length + width)
  let cost_per_meter_rupees := cost_per_meter_paise / 100
  perimeter * cost_per_meter_rupees

theorem cost_of_fencing_theorem :
  cost_of_fencing 3750 3 50 = 125 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_theorem_l1866_186698


namespace NUMINAMATH_GPT_max_min_M_l1866_186614

noncomputable def M (x y : ℝ) : ℝ :=
  abs (x + y) + abs (y + 1) + abs (2 * y - x - 4)

theorem max_min_M (x y : ℝ) (hx : abs x ≤ 1) (hy : abs y ≤ 1) :
  3 ≤ M x y ∧ M x y ≤ 7 :=
sorry

end NUMINAMATH_GPT_max_min_M_l1866_186614


namespace NUMINAMATH_GPT_loan_amount_calculation_l1866_186617

theorem loan_amount_calculation
  (annual_interest : ℝ) (interest_rate : ℝ) (time : ℝ) (loan_amount : ℝ)
  (h1 : annual_interest = 810)
  (h2 : interest_rate = 0.09)
  (h3 : time = 1)
  (h4 : loan_amount = annual_interest / (interest_rate * time)) :
  loan_amount = 9000 := by
sorry

end NUMINAMATH_GPT_loan_amount_calculation_l1866_186617


namespace NUMINAMATH_GPT_second_workshop_production_l1866_186619

theorem second_workshop_production (a b c : ℕ) (h₁ : a + b + c = 3600) (h₂ : a + c = 2 * b) : b * 3 = 3600 := 
by 
  sorry

end NUMINAMATH_GPT_second_workshop_production_l1866_186619


namespace NUMINAMATH_GPT_average_income_B_and_C_l1866_186624

variables (A_income B_income C_income : ℝ)

noncomputable def average_monthly_income_B_and_C (A_income : ℝ) :=
  (B_income + C_income) / 2

theorem average_income_B_and_C
  (h1 : (A_income + B_income) / 2 = 5050)
  (h2 : (A_income + C_income) / 2 = 5200)
  (h3 : A_income = 4000) :
  average_monthly_income_B_and_C 4000 = 6250 :=
by
  sorry

end NUMINAMATH_GPT_average_income_B_and_C_l1866_186624


namespace NUMINAMATH_GPT_percentage_of_children_who_speak_only_english_l1866_186681

theorem percentage_of_children_who_speak_only_english :
  (∃ (total_children both_languages hindi_speaking only_english : ℝ),
    total_children = 60 ∧
    both_languages = 0.20 * total_children ∧
    hindi_speaking = 42 ∧
    only_english = total_children - (hindi_speaking - both_languages + both_languages) ∧
    (only_english / total_children) * 100 = 30) :=
  sorry

end NUMINAMATH_GPT_percentage_of_children_who_speak_only_english_l1866_186681


namespace NUMINAMATH_GPT_gcd_884_1071_l1866_186642

theorem gcd_884_1071 : Nat.gcd 884 1071 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_884_1071_l1866_186642


namespace NUMINAMATH_GPT_T_shaped_area_l1866_186647

theorem T_shaped_area (a b c d : ℕ) (side1 side2 side3 large_side : ℕ)
  (h_side1: side1 = 2)
  (h_side2: side2 = 2)
  (h_side3: side3 = 4)
  (h_large_side: large_side = 6)
  (h_area_large_square : a = large_side * large_side)
  (h_area_square1 : b = side1 * side1)
  (h_area_square2 : c = side2 * side2)
  (h_area_square3 : d = side3 * side3) :
  a - (b + c + d) = 12 := by
  sorry

end NUMINAMATH_GPT_T_shaped_area_l1866_186647


namespace NUMINAMATH_GPT_coin_flips_probability_l1866_186685

section 

-- Definition for the probability of heads in a single flip
def prob_heads : ℚ := 1 / 2

-- Definition for flipping the coin 5 times and getting heads on the first 4 flips and tails on the last flip
def prob_specific_sequence (n : ℕ) (k : ℕ) : ℚ := (prob_heads) ^ k * (prob_heads) ^ (n - k)

-- The main theorem which states the probability of the desired outcome
theorem coin_flips_probability : 
  prob_specific_sequence 5 4 = 1 / 32 :=
sorry

end

end NUMINAMATH_GPT_coin_flips_probability_l1866_186685


namespace NUMINAMATH_GPT_harry_worked_32_hours_l1866_186686

variable (x y : ℝ)
variable (harry_pay james_pay : ℝ)

-- Definitions based on conditions
def harry_weekly_pay (h : ℝ) := 30*x + (h - 30)*y
def james_weekly_pay := 40*x + 1*y

-- Condition: Harry and James were paid the same last week
axiom harry_james_same_pay : ∀ (h : ℝ), harry_weekly_pay x y h = james_weekly_pay x y

-- Prove: Harry worked 32 hours
theorem harry_worked_32_hours : ∃ h : ℝ, h = 32 ∧ harry_weekly_pay x y h = james_weekly_pay x y := by
  sorry

end NUMINAMATH_GPT_harry_worked_32_hours_l1866_186686


namespace NUMINAMATH_GPT_find_initial_mangoes_l1866_186674

-- Define the initial conditions
def initial_apples : Nat := 7
def initial_oranges : Nat := 8
def apples_taken : Nat := 2
def oranges_taken : Nat := 2 * apples_taken
def remaining_fruits : Nat := 14
def mangoes_remaining (M : Nat) : Nat := M / 3

-- Define the problem statement
theorem find_initial_mangoes (M : Nat) (hM : 7 - apples_taken + 8 - oranges_taken + mangoes_remaining M = remaining_fruits) : M = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_mangoes_l1866_186674


namespace NUMINAMATH_GPT_math_problem_l1866_186668

theorem math_problem 
  (m n : ℕ) 
  (h1 : (m^2 - n) ∣ (m + n^2))
  (h2 : (n^2 - m) ∣ (m^2 + n)) : 
  (m, n) = (2, 2) ∨ (m, n) = (3, 3) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 1) ∨ (m, n) = (2, 3) ∨ (m, n) = (3, 2) := 
sorry

end NUMINAMATH_GPT_math_problem_l1866_186668


namespace NUMINAMATH_GPT_evaluate_expression_l1866_186690

theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2 * x + 2) / x) * ((y^2 + 2 * y + 2) / y) + ((x^2 - 3 * x + 2) / y) * ((y^2 - 3 * y + 2) / x) 
  = 2 * x * y - (x / y) - (y / x) + 13 + 10 / x + 4 / y + 8 / (x * y) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1866_186690


namespace NUMINAMATH_GPT_no_such_class_exists_l1866_186695

theorem no_such_class_exists : ¬ ∃ (b g : ℕ), (3 * b = 5 * g) ∧ (32 < b + g) ∧ (b + g < 40) :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_no_such_class_exists_l1866_186695


namespace NUMINAMATH_GPT_total_chewing_gums_l1866_186636

-- Definitions for the conditions
def mary_gums : Nat := 5
def sam_gums : Nat := 10
def sue_gums : Nat := 15

-- Lean 4 Theorem statement to prove the total chewing gums
theorem total_chewing_gums : mary_gums + sam_gums + sue_gums = 30 := by
  sorry

end NUMINAMATH_GPT_total_chewing_gums_l1866_186636


namespace NUMINAMATH_GPT_pens_count_l1866_186663

theorem pens_count (N P : ℕ) (h1 : N = 40) (h2 : P / N = 5 / 4) : P = 50 :=
by
  sorry

end NUMINAMATH_GPT_pens_count_l1866_186663


namespace NUMINAMATH_GPT_complex_number_solution_l1866_186659

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i * z = 1) : z = -i :=
by sorry

end NUMINAMATH_GPT_complex_number_solution_l1866_186659


namespace NUMINAMATH_GPT_geometric_N_digit_not_20_l1866_186612

-- Variables and definitions
variables (a b c : ℕ)

-- Given conditions
def geometric_progression (a b c : ℕ) : Prop :=
  ∃ q : ℚ, (b = q * a) ∧ (c = q * b)

def ends_with_20 (N : ℕ) : Prop := N % 100 = 20

-- Prove the main theorem
theorem geometric_N_digit_not_20 (h1 : geometric_progression a b c) (h2 : ends_with_20 (a^3 + b^3 + c^3 - 3 * a * b * c)) :
  False :=
sorry

end NUMINAMATH_GPT_geometric_N_digit_not_20_l1866_186612


namespace NUMINAMATH_GPT_elevator_travel_time_l1866_186622

noncomputable def total_time_in_hours (floors : ℕ) (time_first_half : ℕ) (time_next_floors_per_floor : ℕ) (next_floors : ℕ) (time_final_floors_per_floor : ℕ) (final_floors : ℕ) : ℕ :=
  let time_first_part := time_first_half
  let time_next_part := time_next_floors_per_floor * next_floors
  let time_final_part := time_final_floors_per_floor * final_floors
  (time_first_part + time_next_part + time_final_part) / 60

theorem elevator_travel_time :
  total_time_in_hours 20 15 5 5 16 5 = 2 := 
by
  sorry

end NUMINAMATH_GPT_elevator_travel_time_l1866_186622


namespace NUMINAMATH_GPT_negation_example_l1866_186677

theorem negation_example :
  (¬ ∀ x y : ℝ, |x + y| > 3) ↔ (∃ x y : ℝ, |x + y| ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l1866_186677


namespace NUMINAMATH_GPT_prob_green_second_given_first_green_l1866_186652

def total_balls : Nat := 14
def green_balls : Nat := 8
def red_balls : Nat := 6

def prob_green_first_draw : ℚ := green_balls / total_balls

theorem prob_green_second_given_first_green :
  prob_green_first_draw = (8 / 14) → (green_balls / total_balls) = (4 / 7) :=
by
  sorry

end NUMINAMATH_GPT_prob_green_second_given_first_green_l1866_186652


namespace NUMINAMATH_GPT_tangent_line_at_one_e_l1866_186699

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_one_e : ∀ (x y : ℝ), (x, y) = (1, Real.exp 1) → (y = 2 * Real.exp x * x - Real.exp 1) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_tangent_line_at_one_e_l1866_186699


namespace NUMINAMATH_GPT_max_value_2ab_2bc_root_3_l1866_186610

theorem max_value_2ab_2bc_root_3 (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a^2 + b^2 + c^2 = 3) :
  2 * a * b + 2 * b * c * Real.sqrt 3 ≤ 6 := by
sorry

end NUMINAMATH_GPT_max_value_2ab_2bc_root_3_l1866_186610


namespace NUMINAMATH_GPT_part1_proof_part2_proof_part3_proof_l1866_186671

-- Definitions and conditions for part 1
def P (a : ℤ) : ℤ × ℤ := (-3 * a - 4, 2 + a)
def part1_condition (a : ℤ) : Prop := (2 + a = 0)
def part1_answer : ℤ × ℤ := (2, 0)

-- Definitions and conditions for part 2
def Q : ℤ × ℤ := (5, 8)
def part2_condition (a : ℤ) : Prop := (-3 * a - 4 = 5)
def part2_answer : ℤ × ℤ := (5, -1)

-- Definitions and conditions for part 3
def part3_condition (a : ℤ) : Prop := 
  (-3 * a - 4 + 2 + a = 0) ∧ (-3 * a - 4 < 0 ∧ 2 + a > 0) -- Second quadrant
def part3_answer (a : ℤ) : ℤ := (a ^ 2023 + 2023)

-- Lean statements for proofs

theorem part1_proof (a : ℤ) (h : part1_condition a) : P a = part1_answer :=
by sorry

theorem part2_proof (a : ℤ) (h : part2_condition a) : P a = part2_answer :=
by sorry

theorem part3_proof (a : ℤ) (h : part3_condition a) : part3_answer a = 2022 :=
by sorry

end NUMINAMATH_GPT_part1_proof_part2_proof_part3_proof_l1866_186671


namespace NUMINAMATH_GPT_bc_possible_values_l1866_186613

theorem bc_possible_values (a b c : ℝ) 
  (h1 : a + b + c = 100) 
  (h2 : ab + bc + ca = 20) 
  (h3 : (a + b) * (a + c) = 24) : 
  bc = -176 ∨ bc = 224 :=
by
  sorry

end NUMINAMATH_GPT_bc_possible_values_l1866_186613


namespace NUMINAMATH_GPT_track_width_l1866_186645

theorem track_width (r1 r2 : ℝ) (h : 2 * π * r1 - 2 * π * r2 = 10 * π) : r1 - r2 = 5 :=
sorry

end NUMINAMATH_GPT_track_width_l1866_186645


namespace NUMINAMATH_GPT_train_stop_time_l1866_186658

theorem train_stop_time : 
  let speed_exc_stoppages := 45.0
  let speed_inc_stoppages := 31.0
  let speed_diff := speed_exc_stoppages - speed_inc_stoppages
  let km_per_minute := speed_exc_stoppages / 60.0
  let stop_time := speed_diff / km_per_minute
  stop_time = 18.67 :=
  by
    sorry

end NUMINAMATH_GPT_train_stop_time_l1866_186658


namespace NUMINAMATH_GPT_main_problem_proof_l1866_186640

def main_problem : Prop :=
  (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2

theorem main_problem_proof : main_problem :=
by {
  sorry
}

end NUMINAMATH_GPT_main_problem_proof_l1866_186640


namespace NUMINAMATH_GPT_simplify_expression_l1866_186648

theorem simplify_expression (x y : ℝ) : 3 * x + 2 * y + 4 * x + 5 * y + 7 = 7 * x + 7 * y + 7 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l1866_186648


namespace NUMINAMATH_GPT_larger_number_of_two_integers_l1866_186694

theorem larger_number_of_two_integers (x y : ℤ) (h1 : x * y = 30) (h2 : x + y = 13) : (max x y = 10) :=
by
  sorry

end NUMINAMATH_GPT_larger_number_of_two_integers_l1866_186694


namespace NUMINAMATH_GPT_tangency_condition_l1866_186639

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 3)^2 = 4

theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m → x^2 = 9 - 9 * y^2 ∧ x^2 = 4 + m * (y + 3)^2 → ((m - 9) * y^2 + 6 * m * y + (9 * m - 5) = 0 → 36 * m^2 - 4 * (m - 9) * (9 * m - 5) = 0 ) ) → 
  m = 5 / 54 :=
by
  sorry

end NUMINAMATH_GPT_tangency_condition_l1866_186639


namespace NUMINAMATH_GPT_M_inter_N_empty_l1866_186646

def M : Set ℝ := {a : ℝ | (1 / 2 < a ∧ a < 1) ∨ (1 < a)}
def N : Set ℝ := {a : ℝ | 0 < a ∧ a ≤ 1 / 2}

theorem M_inter_N_empty : M ∩ N = ∅ :=
sorry

end NUMINAMATH_GPT_M_inter_N_empty_l1866_186646


namespace NUMINAMATH_GPT_compacted_space_of_all_cans_l1866_186661

def compacted_space_per_can (original_space: ℕ) (compaction_rate: ℕ) : ℕ :=
  original_space * compaction_rate / 100

def total_compacted_space (num_cans: ℕ) (compacted_space: ℕ) : ℕ :=
  num_cans * compacted_space

theorem compacted_space_of_all_cans :
  ∀ (num_cans original_space compaction_rate : ℕ),
  num_cans = 100 →
  original_space = 30 →
  compaction_rate = 35 →
  total_compacted_space num_cans (compacted_space_per_can original_space compaction_rate) = 1050 :=
by
  intros num_cans original_space compaction_rate h1 h2 h3
  rw [h1, h2, h3]
  dsimp [compacted_space_per_can, total_compacted_space]
  norm_num
  sorry

end NUMINAMATH_GPT_compacted_space_of_all_cans_l1866_186661


namespace NUMINAMATH_GPT_total_amount_received_l1866_186665

theorem total_amount_received (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ) (A : ℝ) 
  (hCI : CI = P * ((1 + r / n) ^ (n * t) - 1))
  (hCI_value : CI = 370.80)
  (hr : r = 0.06)
  (hn : n = 1)
  (ht : t = 2)
  (hP : P = 3000)
  (hP_value : P = CI / 0.1236) :
  A = P + CI := 
by 
sorry

end NUMINAMATH_GPT_total_amount_received_l1866_186665


namespace NUMINAMATH_GPT_cosine_of_negative_three_pi_over_two_l1866_186679

theorem cosine_of_negative_three_pi_over_two : 
  Real.cos (-3 * Real.pi / 2) = 0 := 
by sorry

end NUMINAMATH_GPT_cosine_of_negative_three_pi_over_two_l1866_186679


namespace NUMINAMATH_GPT_converse_l1866_186638

theorem converse (x y : ℝ) (h : x + y ≥ 5) : x ≥ 2 ∧ y ≥ 3 := 
sorry

end NUMINAMATH_GPT_converse_l1866_186638


namespace NUMINAMATH_GPT_ratio_of_gilled_to_spotted_l1866_186608

theorem ratio_of_gilled_to_spotted (total_mushrooms gilled_mushrooms spotted_mushrooms : ℕ) 
  (h1 : total_mushrooms = 30) 
  (h2 : gilled_mushrooms = 3) 
  (h3 : spotted_mushrooms = total_mushrooms - gilled_mushrooms) :
  gilled_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 1 ∧ 
  spotted_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 9 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_gilled_to_spotted_l1866_186608


namespace NUMINAMATH_GPT_distance_difference_l1866_186618

-- Given conditions
def speed_train1 : ℕ := 20
def speed_train2 : ℕ := 25
def total_distance : ℕ := 675

-- Define the problem statement
theorem distance_difference : ∃ t : ℝ, (speed_train2 * t - speed_train1 * t) = 75 ∧ (speed_train1 * t + speed_train2 * t) = total_distance := by 
  sorry

end NUMINAMATH_GPT_distance_difference_l1866_186618


namespace NUMINAMATH_GPT_binom_8_3_eq_56_and_2_pow_56_l1866_186627

theorem binom_8_3_eq_56_and_2_pow_56 :
  (Nat.choose 8 3 = 56) ∧ (2 ^ (Nat.choose 8 3) = 2 ^ 56) :=
by
  sorry

end NUMINAMATH_GPT_binom_8_3_eq_56_and_2_pow_56_l1866_186627


namespace NUMINAMATH_GPT_Kevin_lost_cards_l1866_186641

theorem Kevin_lost_cards (initial_cards final_cards : ℝ) (h1 : initial_cards = 47.0) (h2 : final_cards = 40) :
  initial_cards - final_cards = 7 :=
by
  sorry

end NUMINAMATH_GPT_Kevin_lost_cards_l1866_186641


namespace NUMINAMATH_GPT_water_temp_increase_per_minute_l1866_186691

theorem water_temp_increase_per_minute :
  ∀ (initial_temp final_temp total_time pasta_time mixing_ratio : ℝ),
    initial_temp = 41 →
    final_temp = 212 →
    total_time = 73 →
    pasta_time = 12 →
    mixing_ratio = (1 / 3) →
    ((final_temp - initial_temp) / (total_time - pasta_time - (mixing_ratio * pasta_time)) = 3) :=
by
  intros initial_temp final_temp total_time pasta_time mixing_ratio
  sorry

end NUMINAMATH_GPT_water_temp_increase_per_minute_l1866_186691


namespace NUMINAMATH_GPT_consecutive_integer_sets_l1866_186656

theorem consecutive_integer_sets (S : ℕ) (hS : S = 180) : 
  ∃ n_values : Finset ℕ, 
  (∀ n ∈ n_values, (∃ a : ℕ, n * (2 * a + n - 1) = 2 * S) ∧ n >= 2) ∧ 
  n_values.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integer_sets_l1866_186656


namespace NUMINAMATH_GPT_polynomial_roots_l1866_186697

theorem polynomial_roots :
  ∀ (x : ℝ), (x^3 - x^2 - 6 * x + 8 = 0) ↔ (x = 2 ∨ x = (-1 + Real.sqrt 17) / 2 ∨ x = (-1 - Real.sqrt 17) / 2) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l1866_186697


namespace NUMINAMATH_GPT_journey_total_distance_l1866_186634

def miles_driven : ℕ := 923
def miles_to_go : ℕ := 277
def total_distance : ℕ := 1200

theorem journey_total_distance : miles_driven + miles_to_go = total_distance := by
  sorry

end NUMINAMATH_GPT_journey_total_distance_l1866_186634


namespace NUMINAMATH_GPT_samantha_exam_score_l1866_186693

theorem samantha_exam_score :
  ∀ (q1 q2 q3 : ℕ) (s1 s2 s3 : ℚ),
  q1 = 30 → q2 = 50 → q3 = 20 →
  s1 = 0.75 → s2 = 0.8 → s3 = 0.65 →
  (22.5 + 40 + 2 * (0.65 * 20)) / (30 + 50 + 2 * 20) = 0.7375 :=
by
  intros q1 q2 q3 s1 s2 s3 hq1 hq2 hq3 hs1 hs2 hs3
  sorry

end NUMINAMATH_GPT_samantha_exam_score_l1866_186693


namespace NUMINAMATH_GPT_find_B_l1866_186670

def A (a : ℝ) : Set ℝ := {3, Real.log a / Real.log 2}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem find_B (a b : ℝ) (hA : A a = {3, 2}) (hB : B a b = {a, b}) (h : (A a) ∩ (B a b) = {2}) :
  B a b = {2, 4} :=
sorry

end NUMINAMATH_GPT_find_B_l1866_186670


namespace NUMINAMATH_GPT_intersection_A_B_l1866_186650

/-- Definitions for the sets A and B --/
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4, 5}

-- Theorem statement regarding the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {1} :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1866_186650


namespace NUMINAMATH_GPT_find_y_l1866_186692

theorem find_y (y : ℚ) (h : ⌊y⌋ + y = 5) : y = 7 / 3 :=
sorry

end NUMINAMATH_GPT_find_y_l1866_186692


namespace NUMINAMATH_GPT_expression_equals_36_l1866_186623

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_36_l1866_186623


namespace NUMINAMATH_GPT_no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l1866_186621
-- Bringing in the entirety of Mathlib

-- Problem (a): There are no non-zero integers that increase by 7 or 9 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_7_or_9 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ (10 * X + d = 7 * n ∨ 10 * X + d = 9 * n)) :=
by sorry

-- Problem (b): There are no non-zero integers that increase by 4 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_4 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ 10 * X + d = 4 * n) :=
by sorry

end NUMINAMATH_GPT_no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l1866_186621


namespace NUMINAMATH_GPT_cost_price_of_one_toy_l1866_186649

theorem cost_price_of_one_toy (C : ℝ) (h : 21 * C = 21000) : C = 1000 :=
by sorry

end NUMINAMATH_GPT_cost_price_of_one_toy_l1866_186649


namespace NUMINAMATH_GPT_first_number_remainder_one_l1866_186682

theorem first_number_remainder_one (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 2023) :
  (∀ (a b c : ℕ), a < b ∧ b < c ∧ b = a + 1 ∧ c = a + 2 → (a % 3 ≠ b % 3 ∧ a % 3 ≠ c % 3 ∧ b % 3 ≠ c % 3))
  → (n % 3 = 1) :=
sorry

end NUMINAMATH_GPT_first_number_remainder_one_l1866_186682


namespace NUMINAMATH_GPT_probability_and_relationship_l1866_186684

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end NUMINAMATH_GPT_probability_and_relationship_l1866_186684


namespace NUMINAMATH_GPT_alexander_eq_alice_l1866_186643

-- Definitions and conditions
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.07

-- Calculation functions for Alexander and Alice
def alexander_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let taxed_price := price * (1 + tax)
  let discounted_price := taxed_price * (1 - discount)
  discounted_price

def alice_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

-- Proof that the difference between Alexander's and Alice's total is 0
theorem alexander_eq_alice : 
  alexander_total original_price discount_rate sales_tax_rate = 
  alice_total original_price discount_rate sales_tax_rate :=
by
  sorry

end NUMINAMATH_GPT_alexander_eq_alice_l1866_186643


namespace NUMINAMATH_GPT_flight_duration_l1866_186667

theorem flight_duration (takeoff landing : ℕ) (h : ℕ) (m : ℕ)
  (h0 : takeoff = 11 * 60 + 7)
  (h1 : landing = 2 * 60 + 49 + 12 * 60)
  (h2 : 0 < m) (h3 : m < 60) :
  h + m = 45 := 
sorry

end NUMINAMATH_GPT_flight_duration_l1866_186667


namespace NUMINAMATH_GPT_count_colorings_l1866_186662

-- Define the number of disks
def num_disks : ℕ := 6

-- Define colorings with constraints: 2 black, 2 white, 2 blue considering rotations and reflections as equivalent
def valid_colorings : ℕ :=
  18  -- This is the result obtained using Burnside's Lemma as shown in the solution

theorem count_colorings : valid_colorings = 18 := by
  sorry

end NUMINAMATH_GPT_count_colorings_l1866_186662


namespace NUMINAMATH_GPT_maize_donation_amount_l1866_186631

-- Definitions and Conditions
def monthly_storage : ℕ := 1
def months_in_year : ℕ := 12
def years : ℕ := 2
def stolen_tonnes : ℕ := 5
def total_tonnes_at_end : ℕ := 27

-- Theorem statement
theorem maize_donation_amount :
  let total_stored := monthly_storage * (months_in_year * years)
  let remaining_after_theft := total_stored - stolen_tonnes
  total_tonnes_at_end - remaining_after_theft = 8 :=
by
  -- This part is just the statement, hence we use sorry to omit the proof.
  sorry

end NUMINAMATH_GPT_maize_donation_amount_l1866_186631


namespace NUMINAMATH_GPT_tangent_of_inclination_of_OP_l1866_186689

noncomputable def point_P_x (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def point_P_y (φ : ℝ) : ℝ := 2 * Real.sin φ

theorem tangent_of_inclination_of_OP (φ : ℝ) (h: φ = Real.pi / 6) :
  (point_P_y φ / point_P_x φ) = 2 * Real.sqrt 3 / 9 :=
by
  have h1 : point_P_x φ = 3 * (Real.sqrt 3 / 2) := by sorry
  have h2 : point_P_y φ = 1 := by sorry
  sorry

end NUMINAMATH_GPT_tangent_of_inclination_of_OP_l1866_186689


namespace NUMINAMATH_GPT_speed_ratio_l1866_186672

-- Definitions of the conditions in the problem
variables (v_A v_B : ℝ) -- speeds of A and B

-- Condition 1: positions after 3 minutes are equidistant from O
def equidistant_3min : Prop := 3 * v_A = |(-300 + 3 * v_B)|

-- Condition 2: positions after 12 minutes are equidistant from O
def equidistant_12min : Prop := 12 * v_A = |(-300 + 12 * v_B)|

-- Statement to prove
theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_12min v_A v_B) :
  v_A / v_B = 4 / 5 := sorry

end NUMINAMATH_GPT_speed_ratio_l1866_186672


namespace NUMINAMATH_GPT_largest_number_is_870_l1866_186600

-- Define the set of digits {8, 7, 0}
def digits : Set ℕ := {8, 7, 0}

-- Define the largest number that can be made by arranging these digits
def largest_number (s : Set ℕ) : ℕ := 870

-- Statement to prove
theorem largest_number_is_870 : largest_number digits = 870 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_largest_number_is_870_l1866_186600


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1866_186673

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set A
def A : Set ℤ := {x | x ∈ Set.univ ∧ x^2 + x - 2 < 0}

-- State the theorem about the complement of A in U
theorem complement_of_A_in_U :
  (U \ A) = {-2, 1, 2} :=
sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1866_186673


namespace NUMINAMATH_GPT_total_members_in_club_l1866_186625

theorem total_members_in_club (females : ℕ) (males : ℕ) (total : ℕ) : 
  (females = 12) ∧ (females = 2 * males) ∧ (total = females + males) → total = 18 := 
by
  sorry

end NUMINAMATH_GPT_total_members_in_club_l1866_186625


namespace NUMINAMATH_GPT_find_s_l1866_186657

theorem find_s (s : ℝ) (m : ℤ) (d : ℝ) (h_floor : ⌊s⌋ = m) (h_decompose : s = m + d) (h_fractional : 0 ≤ d ∧ d < 1) (h_equation : ⌊s⌋ - s = -10.3) : s = -9.7 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l1866_186657


namespace NUMINAMATH_GPT_find_k_l1866_186637

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n + 5 else (n + 1) / 2

theorem find_k (k : ℤ) (h1 : k % 2 = 0) (h2 : g (g (g k)) = 61) : k = 236 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1866_186637


namespace NUMINAMATH_GPT_vertex_of_parabola_l1866_186605

theorem vertex_of_parabola : 
  ∀ x, (3 * (x - 1)^2 + 2) = ((x - 1)^2 * 3 + 2) := 
by {
  -- The proof steps would go here
  sorry -- Placeholder to signify the proof steps are omitted
}

end NUMINAMATH_GPT_vertex_of_parabola_l1866_186605
