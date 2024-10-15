import Mathlib

namespace NUMINAMATH_GPT_seq_inequality_l1378_137887

variable (a : ℕ → ℝ)
variable (n m : ℕ)

-- Conditions
axiom pos_seq (k : ℕ) : a k ≥ 0
axiom add_condition (i j : ℕ) : a (i + j) ≤ a i + a j

-- Statement to prove
theorem seq_inequality (n m : ℕ) (h : m > 0) (h' : n ≥ m) : 
  a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := sorry

end NUMINAMATH_GPT_seq_inequality_l1378_137887


namespace NUMINAMATH_GPT_tom_pie_share_l1378_137800

theorem tom_pie_share :
  (∃ (x : ℚ), 4 * x = (5 / 8) ∧ x = 5 / 32) :=
by
  sorry

end NUMINAMATH_GPT_tom_pie_share_l1378_137800


namespace NUMINAMATH_GPT_rachel_bought_3_tables_l1378_137852

-- Definitions from conditions
def chairs := 7
def minutes_per_furniture := 4
def total_minutes := 40

-- Define the number of tables Rachel bought
def number_of_tables (chairs : ℕ) (minutes_per_furniture : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes - (chairs * minutes_per_furniture)) / minutes_per_furniture

-- Lean theorem stating the proof problem
theorem rachel_bought_3_tables : number_of_tables chairs minutes_per_furniture total_minutes = 3 :=
by
  sorry

end NUMINAMATH_GPT_rachel_bought_3_tables_l1378_137852


namespace NUMINAMATH_GPT_water_displaced_volume_square_l1378_137833

-- Given conditions:
def radius : ℝ := 5
def height : ℝ := 10
def cube_side : ℝ := 6

-- Theorem statement for the problem
theorem water_displaced_volume_square (r h s : ℝ) (w : ℝ) 
  (hr : r = 5) 
  (hh : h = 10) 
  (hs : s = 6) : 
  (w * w) = 13141.855 :=
by 
  sorry

end NUMINAMATH_GPT_water_displaced_volume_square_l1378_137833


namespace NUMINAMATH_GPT_bah_rah_yah_equiv_l1378_137834

-- We define the initial equivalences given in the problem statement.
theorem bah_rah_yah_equiv (bahs rahs yahs : ℕ) :
  (18 * bahs = 30 * rahs) ∧
  (12 * rahs = 20 * yahs) →
  (1200 * yahs = 432 * bahs) :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_bah_rah_yah_equiv_l1378_137834


namespace NUMINAMATH_GPT_find_common_difference_l1378_137878

variable (a₁ d : ℝ)

theorem find_common_difference
  (h1 : a₁ + (a₁ + 6 * d) = 22)
  (h2 : (a₁ + 3 * d) + (a₁ + 9 * d) = 40) :
  d = 3 := by
  sorry

end NUMINAMATH_GPT_find_common_difference_l1378_137878


namespace NUMINAMATH_GPT_length_of_bridge_correct_l1378_137861

noncomputable def L_train : ℝ := 180
noncomputable def v_km_per_hr : ℝ := 60  -- speed in km/hr
noncomputable def t : ℝ := 25

-- Convert speed from km/hr to m/s
noncomputable def km_per_hr_to_m_per_s (v: ℝ) : ℝ := v * (1000 / 3600)
noncomputable def v : ℝ := km_per_hr_to_m_per_s v_km_per_hr

-- Distance covered by the train while crossing the bridge
noncomputable def d : ℝ := v * t

-- Length of the bridge
noncomputable def L_bridge : ℝ := d - L_train

theorem length_of_bridge_correct :
  L_bridge = 236.75 :=
  by
    sorry

end NUMINAMATH_GPT_length_of_bridge_correct_l1378_137861


namespace NUMINAMATH_GPT_additional_income_needed_to_meet_goal_l1378_137801

def monthly_current_income : ℤ := 4000
def annual_goal : ℤ := 60000
def additional_amount_per_month (monthly_current_income annual_goal : ℤ) : ℤ :=
  (annual_goal - (monthly_current_income * 12)) / 12

theorem additional_income_needed_to_meet_goal :
  additional_amount_per_month monthly_current_income annual_goal = 1000 :=
by
  sorry

end NUMINAMATH_GPT_additional_income_needed_to_meet_goal_l1378_137801


namespace NUMINAMATH_GPT_average_salary_correct_l1378_137870

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 15000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 9000 := by
  -- proof is skipped
  sorry

end NUMINAMATH_GPT_average_salary_correct_l1378_137870


namespace NUMINAMATH_GPT_fish_total_after_transfer_l1378_137820

-- Definitions of the initial conditions
def lilly_initial : ℕ := 10
def rosy_initial : ℕ := 9
def jack_initial : ℕ := 15
def fish_transferred : ℕ := 2

-- Total fish after Lilly transfers 2 fish to Jack
theorem fish_total_after_transfer : (lilly_initial - fish_transferred) + rosy_initial + (jack_initial + fish_transferred) = 34 := by
  sorry

end NUMINAMATH_GPT_fish_total_after_transfer_l1378_137820


namespace NUMINAMATH_GPT_product_of_three_numbers_l1378_137837

theorem product_of_three_numbers:
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ 
    a = 2 * (b + c) ∧ 
    b = 5 * c ∧ 
    a * b * c = 2500 / 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_three_numbers_l1378_137837


namespace NUMINAMATH_GPT_y_increase_by_30_when_x_increases_by_12_l1378_137888

theorem y_increase_by_30_when_x_increases_by_12
  (h : ∀ x y : ℝ, x = 4 → y = 10)
  (x_increase : ℝ := 12) :
  ∃ y_increase : ℝ, y_increase = 30 :=
by
  -- Here we assume the condition h and x_increase
  let ratio := 10 / 4  -- Establish the ratio of increase
  let expected_y_increase := x_increase * ratio
  exact ⟨expected_y_increase, sorry⟩  -- Prove it is 30

end NUMINAMATH_GPT_y_increase_by_30_when_x_increases_by_12_l1378_137888


namespace NUMINAMATH_GPT_geometric_common_ratio_l1378_137835

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_common_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : q > 0) 
  (h2 : geometric_seq a q) (h3 : a 3 * a 7 = 4 * (a 4)^2) : q = 2 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_common_ratio_l1378_137835


namespace NUMINAMATH_GPT_even_function_cos_sin_l1378_137886

theorem even_function_cos_sin {f : ℝ → ℝ}
  (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = Real.cos (3 * x) + Real.sin (2 * x)) :
  ∀ x, x > 0 → f x = Real.cos (3 * x) - Real.sin (2 * x) := by
  sorry

end NUMINAMATH_GPT_even_function_cos_sin_l1378_137886


namespace NUMINAMATH_GPT_area_reflected_arcs_l1378_137880

theorem area_reflected_arcs (s : ℝ) (h : s = 2) : 
  ∃ A, A = 2 * Real.pi * Real.sqrt 2 - 8 :=
by
  -- constants
  let r := Real.sqrt (2 * Real.sqrt 2)
  let sector_area := Real.pi * r^2 / 8
  let triangle_area := 1 -- Equilateral triangle properties
  let reflected_arc_area := sector_area - triangle_area
  let total_area := 8 * reflected_arc_area
  use total_area
  sorry

end NUMINAMATH_GPT_area_reflected_arcs_l1378_137880


namespace NUMINAMATH_GPT_ratio_of_girls_to_boys_l1378_137882

theorem ratio_of_girls_to_boys (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : x = 16 ∧ y = 12 ∧ x / y = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_girls_to_boys_l1378_137882


namespace NUMINAMATH_GPT_single_elimination_games_l1378_137821

theorem single_elimination_games (n : ℕ) (h : n = 512) : (n - 1) = 511 :=
by
  sorry

end NUMINAMATH_GPT_single_elimination_games_l1378_137821


namespace NUMINAMATH_GPT_valid_outfits_count_l1378_137869

-- Definitions based on problem conditions
def shirts : Nat := 5
def pants : Nat := 6
def invalid_combination : Nat := 1

-- Problem statement
theorem valid_outfits_count : shirts * pants - invalid_combination = 29 := by 
  sorry

end NUMINAMATH_GPT_valid_outfits_count_l1378_137869


namespace NUMINAMATH_GPT_smallest_solution_of_equation_l1378_137830

theorem smallest_solution_of_equation : 
    ∃ x : ℝ, x*|x| = 3 * x - 2 ∧ 
            ∀ y : ℝ, y*|y| = 3 * y - 2 → x ≤ y :=
sorry

end NUMINAMATH_GPT_smallest_solution_of_equation_l1378_137830


namespace NUMINAMATH_GPT_percentage_is_4_l1378_137805

-- Define the problem conditions
def percentage_condition (p : ℝ) : Prop := p * 50 = 200

-- State the theorem with the given conditions and the correct answer
theorem percentage_is_4 (p : ℝ) (h : percentage_condition p) : p = 4 := sorry

end NUMINAMATH_GPT_percentage_is_4_l1378_137805


namespace NUMINAMATH_GPT_correct_calculation_l1378_137862

variable (a b : ℝ)

theorem correct_calculation : (-a^3)^2 = a^6 := 
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_l1378_137862


namespace NUMINAMATH_GPT_largest_binomial_coeff_and_rational_terms_l1378_137817

theorem largest_binomial_coeff_and_rational_terms 
  (n : ℕ) 
  (h_sum_coeffs : 4^n - 2^n = 992) 
  (T : ℕ → ℝ → ℝ)
  (x : ℝ) :
  (∃ (r1 r2 : ℕ), T r1 x = 270 * x^(22/3) ∧ T r2 x = 90 * x^6)
  ∧
  (∃ (r3 r4 : ℕ), T r3 x = 243 * x^10 ∧ T r4 x = 90 * x^6)
:= 
  
sorry

end NUMINAMATH_GPT_largest_binomial_coeff_and_rational_terms_l1378_137817


namespace NUMINAMATH_GPT_exponentiation_and_division_l1378_137822

theorem exponentiation_and_division (a b c : ℕ) (h : a = 6) (h₂ : b = 3) (h₃ : c = 15) :
  9^a * 3^b / 3^c = 1 := by
  sorry

end NUMINAMATH_GPT_exponentiation_and_division_l1378_137822


namespace NUMINAMATH_GPT_triangle_inequality_check_l1378_137865

theorem triangle_inequality_check :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔
    ((a = 6 ∧ b = 9 ∧ c = 14) ∨ (a = 9 ∧ b = 6 ∧ c = 14) ∨ (a = 6 ∧ b = 14 ∧ c = 9) ∨
     (a = 14 ∧ b = 6 ∧ c = 9) ∨ (a = 9 ∧ b = 14 ∧ c = 6) ∨ (a = 14 ∧ b = 9 ∧ c = 6)) := sorry

end NUMINAMATH_GPT_triangle_inequality_check_l1378_137865


namespace NUMINAMATH_GPT_quadratic_is_perfect_square_l1378_137804

theorem quadratic_is_perfect_square (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ d e : ℤ, a*x^2 + b*x + c = (d*x + e)^2) : 
  ∃ d e : ℤ, a = d^2 ∧ b = 2*d*e ∧ c = e^2 :=
sorry

end NUMINAMATH_GPT_quadratic_is_perfect_square_l1378_137804


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l1378_137806

variable (a d : ℤ)
variable (n : ℕ)

/-- Given the following conditions:
1. The sum of the first three terms of an arithmetic sequence is -3.
2. The product of the first three terms is 8,
This theorem proves that:
1. The general term formula of the sequence is 3 * n - 7.
2. The sum of the first n terms is (3 / 2) * n ^ 2 - (11 / 2) * n.
-/
theorem arithmetic_sequence_solution
  (h1 : (a - d) + a + (a + d) = -3)
  (h2 : (a - d) * a * (a + d) = 8) :
  (∃ a d : ℤ, (∀ n : ℕ, (n ≥ 1) → (3 * n - 7 = a + (n - 1) * d) ∧ (∃ S : ℕ → ℤ, S n = (3 / 2) * n ^ 2 - (11 / 2) * n))) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l1378_137806


namespace NUMINAMATH_GPT_vasya_faster_than_petya_l1378_137874

theorem vasya_faster_than_petya 
  (L : ℝ) (v : ℝ) (x : ℝ) (t : ℝ) 
  (meeting_condition : (v + x * v) * t = L)
  (petya_lap : v * t = L)
  (vasya_meet_petya_after_lap : x * v * t = 2 * L) :
  x = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_vasya_faster_than_petya_l1378_137874


namespace NUMINAMATH_GPT_number_of_strawberries_stolen_l1378_137858

-- Define the conditions
def daily_harvest := 5
def days_in_april := 30
def strawberries_given_away := 20
def strawberries_left_by_end := 100

-- Calculate total harvested strawberries
def total_harvest := daily_harvest * days_in_april
-- Calculate strawberries after giving away
def remaining_after_giveaway := total_harvest - strawberries_given_away

-- Prove the number of strawberries stolen
theorem number_of_strawberries_stolen : remaining_after_giveaway - strawberries_left_by_end = 30 := by
  sorry

end NUMINAMATH_GPT_number_of_strawberries_stolen_l1378_137858


namespace NUMINAMATH_GPT_a_parallel_b_l1378_137809

variable {Line : Type} (a b c : Line)

-- Definition of parallel lines
def parallel (x y : Line) : Prop := sorry

-- Conditions
axiom a_parallel_c : parallel a c
axiom b_parallel_c : parallel b c

-- Theorem to prove a is parallel to b given the conditions
theorem a_parallel_b : parallel a b :=
by
  sorry

end NUMINAMATH_GPT_a_parallel_b_l1378_137809


namespace NUMINAMATH_GPT_probability_scoring_80_or_above_probability_failing_exam_l1378_137845

theorem probability_scoring_80_or_above (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C) = 0.69 :=
by
  intros hB hC hD hE
  sorry

theorem probability_failing_exam (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C ∪ D ∪ E) = 0.93 →
  1 - P (B ∪ C ∪ D ∪ E) = 0.07 :=
by
  intros hB hC hD hE hBCDE
  sorry

end NUMINAMATH_GPT_probability_scoring_80_or_above_probability_failing_exam_l1378_137845


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l1378_137823

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l1378_137823


namespace NUMINAMATH_GPT_common_ratio_q_l1378_137816

noncomputable def Sn (n : ℕ) (a1 q : ℝ) := a1 * (1 - q^n) / (1 - q)

theorem common_ratio_q (a1 : ℝ) (q : ℝ) (h : q ≠ 1) (h1 : 6 * Sn 4 a1 q = Sn 5 a1 q + 5 * Sn 6 a1 q) : q = -6/5 := by
  sorry

end NUMINAMATH_GPT_common_ratio_q_l1378_137816


namespace NUMINAMATH_GPT_intersection_range_of_b_l1378_137853

theorem intersection_range_of_b (b : ℝ) :
  (∀ (m : ℝ), ∃ (x y : ℝ), x^2 + 2 * y^2 = 3 ∧ y = m * x + b) ↔ 
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := 
sorry

end NUMINAMATH_GPT_intersection_range_of_b_l1378_137853


namespace NUMINAMATH_GPT_probability_neither_cake_nor_muffin_l1378_137854

noncomputable def probability_of_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  (total - (cake + muffin - both)) / total

theorem probability_neither_cake_nor_muffin
  (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) (h_total : total = 100)
  (h_cake : cake = 50) (h_muffin : muffin = 40) (h_both : both = 18) :
  probability_of_neither total cake muffin both = 0.28 :=
by
  rw [h_total, h_cake, h_muffin, h_both]
  norm_num
  sorry

end NUMINAMATH_GPT_probability_neither_cake_nor_muffin_l1378_137854


namespace NUMINAMATH_GPT_part1_part2_l1378_137842

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (m n : ℝ) : f (m + n) = f m * f n
axiom positive_property (x : ℝ) (h : x > 0) : 0 < f x ∧ f x < 1

theorem part1 (x : ℝ) : f 0 = 1 ∧ (x < 0 → f x > 1) := by
  sorry

theorem part2 (x : ℝ) : 
  f (2 * x^2 - 4 * x - 1) < 1 ∧ f (x - 1) < 1 → x < -1/2 ∨ x > 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1378_137842


namespace NUMINAMATH_GPT_max_equilateral_triangle_area_l1378_137860

theorem max_equilateral_triangle_area (length width : ℝ) (h_len : length = 15) (h_width : width = 12) 
: ∃ (area : ℝ), area = 200.25 * Real.sqrt 3 - 450 := by
  sorry

end NUMINAMATH_GPT_max_equilateral_triangle_area_l1378_137860


namespace NUMINAMATH_GPT_find_percentage_second_alloy_l1378_137884

open Real

def percentage_copper_second_alloy (percentage_alloy1: ℝ) (ounces_alloy1: ℝ) (percentage_desired_alloy: ℝ) (total_ounces: ℝ) (percentage_second_alloy: ℝ) : Prop :=
  let copper_ounces_alloy1 := percentage_alloy1 * ounces_alloy1 / 100
  let desired_copper_ounces := percentage_desired_alloy * total_ounces / 100
  let needed_copper_ounces := desired_copper_ounces - copper_ounces_alloy1
  let ounces_alloy2 := total_ounces - ounces_alloy1
  (needed_copper_ounces / ounces_alloy2) * 100 = percentage_second_alloy

theorem find_percentage_second_alloy :
  percentage_copper_second_alloy 18 45 19.75 108 21 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_second_alloy_l1378_137884


namespace NUMINAMATH_GPT_solve_sqrt_equation_l1378_137896

theorem solve_sqrt_equation :
  ∀ (x : ℝ), (3 * Real.sqrt x + 3 * x⁻¹/2 = 7) →
  (x = (49 + 14 * Real.sqrt 13 + 13) / 36 ∨ x = (49 - 14 * Real.sqrt 13 + 13) / 36) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_solve_sqrt_equation_l1378_137896


namespace NUMINAMATH_GPT_num_ways_first_to_fourth_floor_l1378_137849

theorem num_ways_first_to_fourth_floor (floors : ℕ) (staircases_per_floor : ℕ) 
  (H_floors : floors = 4) (H_staircases : staircases_per_floor = 2) : 
  (staircases_per_floor) ^ (floors - 1) = 2^3 := 
by 
  sorry

end NUMINAMATH_GPT_num_ways_first_to_fourth_floor_l1378_137849


namespace NUMINAMATH_GPT_inequality_proof_l1378_137877

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 1) :
  ((1 / x^2 - x) * (1 / y^2 - y) * (1 / z^2 - z) ≥ (26 / 3)^3) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1378_137877


namespace NUMINAMATH_GPT_maintain_order_time_l1378_137829

theorem maintain_order_time :
  ∀ (x : ℕ), 
  (let ppl_per_min_norm := 9
   let ppl_per_min_cong := 3
   let total_people := 36 
   let teacher_time_saved := 6

   let time_without_order := total_people / ppl_per_min_cong
   let time_with_order := time_without_order - teacher_time_saved

   let ppl_passed_while_order := ppl_per_min_cong * x
   let ppl_passed_norm_order := ppl_per_min_norm * (time_with_order - x)

   ppl_passed_while_order + ppl_passed_norm_order = total_people) → 
  x = 3 :=
sorry

end NUMINAMATH_GPT_maintain_order_time_l1378_137829


namespace NUMINAMATH_GPT_capital_after_18_years_l1378_137807

noncomputable def initial_investment : ℝ := 2000
def rate_of_increase : ℝ := 0.50
def period : ℕ := 3
def total_time : ℕ := 18

theorem capital_after_18_years :
  (initial_investment * (1 + rate_of_increase) ^ (total_time / period)) = 22781.25 :=
by
  sorry

end NUMINAMATH_GPT_capital_after_18_years_l1378_137807


namespace NUMINAMATH_GPT_no_ingredient_pies_max_l1378_137819

theorem no_ingredient_pies_max :
  ∃ (total apple blueberry cream chocolate no_ingredient : ℕ),
    total = 48 ∧
    apple = 24 ∧
    blueberry = 16 ∧
    cream = 18 ∧
    chocolate = 12 ∧
    no_ingredient = total - (apple + blueberry + chocolate - min apple blueberry - min apple chocolate - min blueberry chocolate) - cream ∧
    no_ingredient = 10 := sorry

end NUMINAMATH_GPT_no_ingredient_pies_max_l1378_137819


namespace NUMINAMATH_GPT_parametrize_line_l1378_137856

theorem parametrize_line (s h : ℝ) :
    s = -5/2 ∧ h = 20 → ∀ t : ℝ, ∃ x y : ℝ, 4 * x + 7 = y ∧ 
    (x = s + 5 * t ∧ y = -3 + h * t) :=
by
  sorry

end NUMINAMATH_GPT_parametrize_line_l1378_137856


namespace NUMINAMATH_GPT_unit_circle_solution_l1378_137863

noncomputable def unit_circle_point_x (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) : ℝ :=
  1 / 26

theorem unit_circle_solution (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) :
  unit_circle_point_x α hα hcos = 1 / 26 :=
by
  sorry

end NUMINAMATH_GPT_unit_circle_solution_l1378_137863


namespace NUMINAMATH_GPT_equal_share_candy_l1378_137855

theorem equal_share_candy :
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  total_candy / number_of_people = 7 :=
by
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  show total_candy / number_of_people = 7
  sorry

end NUMINAMATH_GPT_equal_share_candy_l1378_137855


namespace NUMINAMATH_GPT_length_AB_of_parallelogram_l1378_137815

theorem length_AB_of_parallelogram
  (AD BC : ℝ) (AB CD : ℝ) 
  (h1 : AD = 5) 
  (h2 : BC = 5) 
  (h3 : AB = CD)
  (h4 : AD + BC + AB + CD = 14) : 
  AB = 2 :=
by
  sorry

end NUMINAMATH_GPT_length_AB_of_parallelogram_l1378_137815


namespace NUMINAMATH_GPT_max_sum_of_first_n_terms_l1378_137851

variable {a : ℕ → ℝ} -- Define sequence a with index ℕ and real values
variable {d : ℝ}      -- Common difference for the arithmetic sequence

-- Conditions and question are formulated into the theorem statement
theorem max_sum_of_first_n_terms (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_diff_neg : d < 0)
  (h_a4_eq_a12 : (a 4)^2 = (a 12)^2) :
  n = 7 ∨ n = 8 := 
sorry

end NUMINAMATH_GPT_max_sum_of_first_n_terms_l1378_137851


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1378_137812

theorem other_root_of_quadratic (m : ℝ) :
  (∃ t : ℝ, (x^2 + m * x - 20 = 0) ∧ (x = -4 ∨ x = t)) → (t = 5) :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1378_137812


namespace NUMINAMATH_GPT_parallel_line_distance_equation_l1378_137899

theorem parallel_line_distance_equation :
  ∃ m : ℝ, (m = -20 ∨ m = 32) ∧
  ∀ x y : ℝ, (5 * x - 12 * y + 6 = 0) → 
            (5 * x - 12 * y + m = 0) :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_distance_equation_l1378_137899


namespace NUMINAMATH_GPT_question_statement_l1378_137876

-- Definitions based on conditions
def all_cards : List ℕ := [8, 3, 6, 5, 0, 7]
def A : ℕ := 876  -- The largest number from the given cards.
def B : ℕ := 305  -- The smallest number from the given cards with non-zero hundreds place.

-- The proof problem statement
theorem question_statement :
  (A - B) * 6 = 3426 := by
  sorry

end NUMINAMATH_GPT_question_statement_l1378_137876


namespace NUMINAMATH_GPT_total_seeds_l1378_137843

-- Define the conditions given in the problem
def morningMikeTomato := 50
def morningMikePepper := 30

def morningTedTomato := 2 * morningMikeTomato
def morningTedPepper := morningMikePepper / 2

def morningSarahTomato := morningMikeTomato + 30
def morningSarahPepper := morningMikePepper + 30

def afternoonMikeTomato := 60
def afternoonMikePepper := 40

def afternoonTedTomato := afternoonMikeTomato - 20
def afternoonTedPepper := afternoonMikePepper

def afternoonSarahTomato := morningSarahTomato + 20
def afternoonSarahPepper := morningSarahPepper + 10

-- Prove that the total number of seeds planted is 685
theorem total_seeds (total: Nat) : 
    total = (
        (morningMikeTomato + afternoonMikeTomato) + 
        (morningTedTomato + afternoonTedTomato) + 
        (morningSarahTomato + afternoonSarahTomato) +
        (morningMikePepper + afternoonMikePepper) + 
        (morningTedPepper + afternoonTedPepper) + 
        (morningSarahPepper + afternoonSarahPepper)
    ) := 
    by 
        have tomato_seeds := (
            morningMikeTomato + afternoonMikeTomato +
            morningTedTomato + afternoonTedTomato + 
            morningSarahTomato + afternoonSarahTomato
        )
        have pepper_seeds := (
            morningMikePepper + afternoonMikePepper +
            morningTedPepper + afternoonTedPepper + 
            morningSarahPepper + afternoonSarahPepper
        )
        have total_seeds := tomato_seeds + pepper_seeds
        sorry

end NUMINAMATH_GPT_total_seeds_l1378_137843


namespace NUMINAMATH_GPT_equivalent_solution_l1378_137841

theorem equivalent_solution (c x : ℤ) 
    (h1 : 3 * x + 9 = 6)
    (h2 : c * x - 15 = -5)
    (hx : x = -1) :
    c = -10 :=
sorry

end NUMINAMATH_GPT_equivalent_solution_l1378_137841


namespace NUMINAMATH_GPT_trace_ellipse_l1378_137818

open Complex

theorem trace_ellipse (z : ℂ) (θ : ℝ) (h₁ : z = 3 * exp (θ * I))
  (h₂ : abs z = 3) : ∃ a b : ℝ, ∀ θ, z + 1/z = a * Real.cos θ + b * (I * Real.sin θ) :=
sorry

end NUMINAMATH_GPT_trace_ellipse_l1378_137818


namespace NUMINAMATH_GPT_male_red_ants_percentage_l1378_137883

noncomputable def percentage_of_total_ant_population_that_are_red_females (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    (percent_red_are_females / 100) * red_population_pct

noncomputable def percentage_of_total_ant_population_that_are_red_males (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    red_population_pct - percentage_of_total_ant_population_that_are_red_females total_population_pct red_population_pct percent_red_are_females

theorem male_red_ants_percentage (total_population_pct red_population_pct percent_red_are_females male_red_ants_pct : ℝ) :
    red_population_pct = 85 → percent_red_are_females = 45 → male_red_ants_pct = 46.75 →
    percentage_of_total_ant_population_that_are_red_males total_population_pct red_population_pct percent_red_are_females = male_red_ants_pct :=
by
sorry

end NUMINAMATH_GPT_male_red_ants_percentage_l1378_137883


namespace NUMINAMATH_GPT_total_germs_calculation_l1378_137879

def number_of_dishes : ℕ := 10800
def germs_per_dish : ℕ := 500
def total_germs : ℕ := 5400000

theorem total_germs_calculation : germs_per_ddish * number_of_idshessh = total_germs :=
by sorry

end NUMINAMATH_GPT_total_germs_calculation_l1378_137879


namespace NUMINAMATH_GPT_number_of_clients_l1378_137826

theorem number_of_clients (cars_clients_selects : ℕ)
                          (cars_selected_per_client : ℕ)
                          (each_car_selected_times : ℕ)
                          (total_cars : ℕ)
                          (h1 : total_cars = 18)
                          (h2 : cars_clients_selects = total_cars * each_car_selected_times)
                          (h3 : each_car_selected_times = 3)
                          (h4 : cars_selected_per_client = 3)
                          : total_cars * each_car_selected_times / cars_selected_per_client = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_clients_l1378_137826


namespace NUMINAMATH_GPT_ratio_length_to_width_is_3_l1378_137866

-- Define the conditions given in the problem
def area_of_garden : ℕ := 768
def width_of_garden : ℕ := 16

-- Define the length calculated from the area and width
def length_of_garden := area_of_garden / width_of_garden

-- Define the ratio to be proven
def ratio_of_length_to_width := length_of_garden / width_of_garden

-- Prove that the ratio is 3:1
theorem ratio_length_to_width_is_3 :
  ratio_of_length_to_width = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_length_to_width_is_3_l1378_137866


namespace NUMINAMATH_GPT_sector_area_is_2_l1378_137825

-- Definition of the sector's properties
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := r * θ + 2 * r

def sector_area (r : ℝ) (θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Theorem stating that the area of the sector is 2 cm² given the conditions
theorem sector_area_is_2 (r θ : ℝ) (h1 : sector_perimeter r θ = 6) (h2 : θ = 1) : sector_area r θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_is_2_l1378_137825


namespace NUMINAMATH_GPT_lynne_total_spending_l1378_137811

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end NUMINAMATH_GPT_lynne_total_spending_l1378_137811


namespace NUMINAMATH_GPT_find_S12_l1378_137827

theorem find_S12 (S : ℕ → ℕ) (h1 : S 3 = 6) (h2 : S 9 = 15) : S 12 = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_S12_l1378_137827


namespace NUMINAMATH_GPT_scientific_notation_of_million_l1378_137813

theorem scientific_notation_of_million : 1000000 = 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_million_l1378_137813


namespace NUMINAMATH_GPT_addition_result_l1378_137898

theorem addition_result (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end NUMINAMATH_GPT_addition_result_l1378_137898


namespace NUMINAMATH_GPT_arithmetic_sequence_second_term_l1378_137873

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_second_term_l1378_137873


namespace NUMINAMATH_GPT_percentage_of_sikh_boys_l1378_137848

-- Define the conditions
def total_boys : ℕ := 850
def percentage_muslim_boys : ℝ := 0.46
def percentage_hindu_boys : ℝ := 0.28
def boys_other_communities : ℕ := 136

-- Theorem to prove the percentage of Sikh boys is 10%
theorem percentage_of_sikh_boys : 
  (((total_boys - 
      (percentage_muslim_boys * total_boys + 
       percentage_hindu_boys * total_boys + 
       boys_other_communities))
    / total_boys) * 100 = 10) :=
by
  -- sorry prevents the need to provide proof here
  sorry

end NUMINAMATH_GPT_percentage_of_sikh_boys_l1378_137848


namespace NUMINAMATH_GPT_find_k_value_l1378_137875

theorem find_k_value :
  (∃ p q : ℝ → ℝ,
    (∀ x, p x = 3 * x + 5) ∧
    (∃ k : ℝ, (∀ x, q x = k * x + 3) ∧
      (p (-4) = -7) ∧ (q (-4) = -7) ∧ k = 2.5)) :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l1378_137875


namespace NUMINAMATH_GPT_derek_lowest_score_l1378_137814

theorem derek_lowest_score:
  ∀ (score1 score2 max_points target_avg min_score tests_needed last_test1 last_test2 : ℕ),
  score1 = 85 →
  score2 = 78 →
  max_points = 100 →
  target_avg = 84 →
  min_score = 60 →
  tests_needed = 4 →
  last_test1 >= min_score →
  last_test2 >= min_score →
  last_test1 <= max_points →
  last_test2 <= max_points →
  (score1 + score2 + last_test1 + last_test2) = target_avg * tests_needed →
  min last_test1 last_test2 = 73 :=
by
  sorry

end NUMINAMATH_GPT_derek_lowest_score_l1378_137814


namespace NUMINAMATH_GPT_mean_equality_l1378_137895

theorem mean_equality (z : ℚ) :
  ((8 + 7 + 28) / 3 : ℚ) = (14 + z) / 2 → z = 44 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_equality_l1378_137895


namespace NUMINAMATH_GPT_supplementary_angles_difference_l1378_137846
-- Import necessary libraries

-- Define the conditions
def are_supplementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 180

def ratio_7_2 (θ₁ θ₂ : ℝ) : Prop := θ₁ / θ₂ = 7 / 2

-- State the theorem
theorem supplementary_angles_difference (θ₁ θ₂ : ℝ) 
  (h_supp : are_supplementary θ₁ θ₂) 
  (h_ratio : ratio_7_2 θ₁ θ₂) :
  |θ₁ - θ₂| = 100 :=
by
  sorry

end NUMINAMATH_GPT_supplementary_angles_difference_l1378_137846


namespace NUMINAMATH_GPT_equation_for_number_l1378_137890

variable (a : ℤ)

theorem equation_for_number : 3 * a + 5 = 9 :=
sorry

end NUMINAMATH_GPT_equation_for_number_l1378_137890


namespace NUMINAMATH_GPT_remaining_money_correct_l1378_137891

open Nat

def initial_money : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def remaining_money : ℕ := initial_money - total_spent

theorem remaining_money_correct : remaining_money = 78 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remaining_money_correct_l1378_137891


namespace NUMINAMATH_GPT_time_to_install_rest_of_windows_l1378_137838

-- Definition of the given conditions:
def num_windows_needed : ℕ := 10
def num_windows_installed : ℕ := 6
def install_time_per_window : ℕ := 5

-- Statement that we aim to prove:
theorem time_to_install_rest_of_windows :
  install_time_per_window * (num_windows_needed - num_windows_installed) = 20 := by
  sorry

end NUMINAMATH_GPT_time_to_install_rest_of_windows_l1378_137838


namespace NUMINAMATH_GPT_ab_cd_eq_neg_37_over_9_l1378_137840

theorem ab_cd_eq_neg_37_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a + b + d = 2)
  (h3 : a + c + d = 3)
  (h4 : b + c + d = -3) :
  a * b + c * d = -37 / 9 := by
  sorry

end NUMINAMATH_GPT_ab_cd_eq_neg_37_over_9_l1378_137840


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1378_137857

theorem necessary_but_not_sufficient (a : ℝ) : (a - 1 < 0 ↔ a < 1) ∧ (|a| < 1 → a < 1) ∧ ¬ (a < 1 → |a| < 1) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1378_137857


namespace NUMINAMATH_GPT_cosine_between_vectors_l1378_137802

noncomputable def vector_cos_angle (a b : ℝ × ℝ) := 
  let dot_product := (a.1 * b.1) + (a.2 * b.2)
  let norm_a := Real.sqrt (a.1 * a.1 + a.2 * a.2)
  let norm_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  dot_product / (norm_a * norm_b)

theorem cosine_between_vectors (t : ℝ) 
  (ht : let a := (1, t); let b := (-1, 2 * t);
        (3 * a.1 - b.1) * b.1 + (3 * a.2 - b.2) * b.2 = 0) :
  vector_cos_angle (1, t) (-1, 2 * t) = Real.sqrt 3 / 3 := 
by
  sorry

end NUMINAMATH_GPT_cosine_between_vectors_l1378_137802


namespace NUMINAMATH_GPT_number_of_marked_points_l1378_137893

theorem number_of_marked_points (S S' : ℤ) (n : ℤ) 
  (h1 : S = 25) 
  (h2 : S' = S - 5 * n) 
  (h3 : S' = -35) : 
  n = 12 := 
  sorry

end NUMINAMATH_GPT_number_of_marked_points_l1378_137893


namespace NUMINAMATH_GPT_length_BC_fraction_of_AD_l1378_137831

-- Define variables and conditions
variables (x y : ℝ)
variable (h1 : 4 * x = 8 * y) -- given: length of AD from both sides
variable (h2 : 3 * x) -- AB = 3 * BD
variable (h3 : 7 * y) -- AC = 7 * CD

-- State the goal to prove
theorem length_BC_fraction_of_AD (x y : ℝ) (h1 : 4 * x = 8 * y) :
  (y / (4 * x)) = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_length_BC_fraction_of_AD_l1378_137831


namespace NUMINAMATH_GPT_find_m_l1378_137897

-- Definitions based on conditions
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def are_roots_of_quadratic (b c m : ℝ) : Prop :=
  b * c = 6 - m ∧ b + c = -(m + 2)

-- The theorem statement
theorem find_m {a b c m : ℝ} (h₁ : a = 5) (h₂ : is_isosceles_triangle a b c) (h₃ : are_roots_of_quadratic b c m) : m = -10 :=
sorry

end NUMINAMATH_GPT_find_m_l1378_137897


namespace NUMINAMATH_GPT_car_travel_distance_20_minutes_l1378_137889

noncomputable def train_speed_in_mph : ℝ := 80
noncomputable def car_speed_ratio : ℝ := 3/4
noncomputable def car_speed_in_mph : ℝ := car_speed_ratio * train_speed_in_mph
noncomputable def travel_time_in_hours : ℝ := 20 / 60
noncomputable def distance_travelled_by_car : ℝ := car_speed_in_mph * travel_time_in_hours

theorem car_travel_distance_20_minutes : distance_travelled_by_car = 20 := 
by 
  sorry

end NUMINAMATH_GPT_car_travel_distance_20_minutes_l1378_137889


namespace NUMINAMATH_GPT_sum_series_eq_one_third_l1378_137844

theorem sum_series_eq_one_third :
  ∑' n : ℕ, (if h : n > 0 then (2^n / (1 + 2^n + 2^(n + 1) + 2^(2 * n + 1))) else 0) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_eq_one_third_l1378_137844


namespace NUMINAMATH_GPT_quantity_of_milk_in_original_mixture_l1378_137832

variable (M W : ℕ)

-- Conditions
def ratio_original : Prop := M = 2 * W
def ratio_after_adding_water : Prop := M * 5 = 6 * (W + 10)

theorem quantity_of_milk_in_original_mixture
  (h1 : ratio_original M W)
  (h2 : ratio_after_adding_water M W) :
  M = 30 := by
  sorry

end NUMINAMATH_GPT_quantity_of_milk_in_original_mixture_l1378_137832


namespace NUMINAMATH_GPT_find_n_divisibility_l1378_137847

theorem find_n_divisibility :
  ∃ n : ℕ, n < 10 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 11 = 0 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 5 = 0 :=
by
  use 3
  sorry

end NUMINAMATH_GPT_find_n_divisibility_l1378_137847


namespace NUMINAMATH_GPT_ratio_of_money_l1378_137859

-- Conditions
def amount_given := 14
def cost_of_gift := 28

-- Theorem statement to prove
theorem ratio_of_money (h1 : amount_given = 14) (h2 : cost_of_gift = 28) :
  amount_given / cost_of_gift = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_money_l1378_137859


namespace NUMINAMATH_GPT_gcd_possible_values_l1378_137881

theorem gcd_possible_values (a b : ℕ) (hab : a * b = 288) : 
  ∃ S : Finset ℕ, (∀ g : ℕ, g ∈ S ↔ ∃ p q r s : ℕ, p + r = 5 ∧ q + s = 2 ∧ g = 2^min p r * 3^min q s) 
  ∧ S.card = 14 := 
sorry

end NUMINAMATH_GPT_gcd_possible_values_l1378_137881


namespace NUMINAMATH_GPT_lcm_is_only_function_l1378_137867

noncomputable def f (x y : ℕ) : ℕ := Nat.lcm x y

theorem lcm_is_only_function 
    (f : ℕ → ℕ → ℕ)
    (h1 : ∀ x : ℕ, f x x = x) 
    (h2 : ∀ x y : ℕ, f x y = f y x) 
    (h3 : ∀ x y : ℕ, (x + y) * f x y = y * f x (x + y)) : 
  ∀ x y : ℕ, f x y = Nat.lcm x y := 
sorry

end NUMINAMATH_GPT_lcm_is_only_function_l1378_137867


namespace NUMINAMATH_GPT_f_monotonic_intervals_g_not_below_f_inequality_holds_l1378_137836

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem f_monotonic_intervals :
  ∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < 1 / 2 → f x < f (x + 1)) ∧ 
    (1 / 2 < x ∧ x < 1 → f x > f (x + 1)) ∧ 
    (1 < x → f x < f (x + 1)) :=
sorry

theorem g_not_below_f :
  ∀ x : ℝ, 0 < x → f x < g x :=
sorry

theorem inequality_holds (n : ℕ) : (2 * n + 1)^2 > 4 * Real.log (Nat.factorial n) :=
sorry

end NUMINAMATH_GPT_f_monotonic_intervals_g_not_below_f_inequality_holds_l1378_137836


namespace NUMINAMATH_GPT_sarah_reads_40_words_per_minute_l1378_137872

-- Define the conditions as constants
def words_per_page := 100
def pages_per_book := 80
def reading_hours := 20
def number_of_books := 6

-- Convert hours to minutes
def total_reading_time := reading_hours * 60

-- Calculate the total number of words in one book
def words_per_book := words_per_page * pages_per_book

-- Calculate the total number of words in all books
def total_words := words_per_book * number_of_books

-- Define the words read per minute
def words_per_minute := total_words / total_reading_time

-- Theorem statement: Sarah reads 40 words per minute
theorem sarah_reads_40_words_per_minute : words_per_minute = 40 :=
by
  sorry

end NUMINAMATH_GPT_sarah_reads_40_words_per_minute_l1378_137872


namespace NUMINAMATH_GPT_prob_draw_l1378_137871

-- Define the probabilities as constants
def prob_A_winning : ℝ := 0.4
def prob_A_not_losing : ℝ := 0.9

-- Prove that the probability of a draw is 0.5
theorem prob_draw : prob_A_not_losing - prob_A_winning = 0.5 :=
by sorry

end NUMINAMATH_GPT_prob_draw_l1378_137871


namespace NUMINAMATH_GPT_solve_inequality_l1378_137824

open Set

variable {f : ℝ → ℝ}
open Function

theorem solve_inequality (h_inc : ∀ x y, 0 < x → 0 < y → x < y → f x < f y)
  (h_func_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y)
  (h_f3 : f 3 = 1)
  (x : ℝ) (hx_pos : 0 < x)
  (hx_ge : x > 5)
  (h_ineq : f x - f (1 / (x - 5)) ≥ 2) :
  x ≥ (5 + Real.sqrt 61) / 2 := sorry

end NUMINAMATH_GPT_solve_inequality_l1378_137824


namespace NUMINAMATH_GPT_lines_not_intersecting_may_be_parallel_or_skew_l1378_137828

theorem lines_not_intersecting_may_be_parallel_or_skew (a b : ℝ × ℝ → Prop) 
  (h : ∀ x, ¬ (a x ∧ b x)) : 
  (∃ c d : ℝ × ℝ → Prop, a = c ∧ b = d) := 
sorry

end NUMINAMATH_GPT_lines_not_intersecting_may_be_parallel_or_skew_l1378_137828


namespace NUMINAMATH_GPT_tailoring_cost_is_200_l1378_137850

variables 
  (cost_first_suit : ℕ := 300)
  (total_paid : ℕ := 1400)

def cost_of_second_suit (tailoring_cost : ℕ) := 3 * cost_first_suit + tailoring_cost

theorem tailoring_cost_is_200 (T : ℕ) (h1 : cost_first_suit = 300) (h2 : total_paid = 1400) 
  (h3 : total_paid = cost_first_suit + cost_of_second_suit T) : 
  T = 200 := 
by 
  sorry

end NUMINAMATH_GPT_tailoring_cost_is_200_l1378_137850


namespace NUMINAMATH_GPT_correct_operation_l1378_137839

/-- Proving that among the given mathematical operations, only the second option is correct. -/
theorem correct_operation (m : ℝ) : ¬ (m^3 - m^2 = m) ∧ (3 * m^2 * 2 * m^3 = 6 * m^5) ∧ ¬ (3 * m^2 + 2 * m^3 = 5 * m^5) ∧ ¬ ((2 * m^2)^3 = 8 * m^5) :=
by
  -- These are the conditions, proof is omitted using sorry
  sorry

end NUMINAMATH_GPT_correct_operation_l1378_137839


namespace NUMINAMATH_GPT_range_of_x_inequality_l1378_137892

theorem range_of_x_inequality (a : ℝ) (x : ℝ)
  (h : -1 ≤ a ∧ a ≤ 1) : 
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_inequality_l1378_137892


namespace NUMINAMATH_GPT_tom_age_ratio_l1378_137868

theorem tom_age_ratio (T N : ℕ)
  (sum_children : T = T) 
  (age_condition : T - N = 3 * (T - 4 * N)) :
  T / N = 11 / 2 := 
sorry

end NUMINAMATH_GPT_tom_age_ratio_l1378_137868


namespace NUMINAMATH_GPT_smallest_percent_increase_l1378_137885

-- Define the values of each question
def question_values : List ℕ :=
  [150, 250, 400, 600, 1100, 2300, 4700, 9500, 19000, 38000, 76000, 150000, 300000, 600000, 1200000]

-- Define a function to calculate the percent increase between two questions
def percent_increase (v1 v2 : ℕ) : Float :=
  ((v2 - v1).toFloat / v1.toFloat) * 100

-- Define the specific question transitions and their percent increases
def percent_increase_1_to_4 : Float := percent_increase question_values[0] question_values[3]  -- Question 1 to 4
def percent_increase_2_to_6 : Float := percent_increase question_values[1] question_values[5]  -- Question 2 to 6
def percent_increase_5_to_10 : Float := percent_increase question_values[4] question_values[9]  -- Question 5 to 10
def percent_increase_9_to_15 : Float := percent_increase question_values[8] question_values[14] -- Question 9 to 15

-- Prove that the smallest percent increase is from Question 1 to 4
theorem smallest_percent_increase :
  percent_increase_1_to_4 < percent_increase_2_to_6 ∧
  percent_increase_1_to_4 < percent_increase_5_to_10 ∧
  percent_increase_1_to_4 < percent_increase_9_to_15 :=
by
  sorry

end NUMINAMATH_GPT_smallest_percent_increase_l1378_137885


namespace NUMINAMATH_GPT_positive_irrational_less_than_one_l1378_137864

theorem positive_irrational_less_than_one : 
  ∃! (x : ℝ), 
    (x = (Real.sqrt 6) / 3 ∧ Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = -(Real.sqrt 3) / 3 ∧ Irrational x ∧ x < 0) ∨ 
    (x = 1 / 3 ∧ ¬Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = Real.pi / 3 ∧ Irrational x ∧ x > 1) :=
by
  sorry

end NUMINAMATH_GPT_positive_irrational_less_than_one_l1378_137864


namespace NUMINAMATH_GPT_minimum_value_of_xy_l1378_137810

noncomputable def minimum_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : ℝ :=
  if hmin : 4 * x + y + 12 = x * y then 36 else sorry

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : 
  minimum_value_xy x y hx hy h = 36 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_xy_l1378_137810


namespace NUMINAMATH_GPT_least_pebbles_2021_l1378_137803

noncomputable def least_pebbles (n : ℕ) : ℕ :=
  n + n / 2

theorem least_pebbles_2021 :
  least_pebbles 2021 = 3031 :=
by
  sorry

end NUMINAMATH_GPT_least_pebbles_2021_l1378_137803


namespace NUMINAMATH_GPT_probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l1378_137808

-- Definitions for balls and initial conditions
def totalBalls : ℕ := 10
def redBalls : ℕ := 2
def whiteBalls : ℕ := 3
def yellowBalls : ℕ := 5

-- Drawing without replacement
noncomputable def probability_second_ball_red : ℚ :=
  (2/10) * (1/9) + (8/10) * (2/9)

-- Probabilities for each case
noncomputable def probability_first_prize : ℚ := 
  (redBalls.choose 1 * whiteBalls.choose 1) / (totalBalls.choose 2)

noncomputable def probability_second_prize : ℚ := 
  (redBalls.choose 2) / (totalBalls.choose 2)

noncomputable def probability_third_prize : ℚ := 
  (whiteBalls.choose 2) / (totalBalls.choose 2)

-- Probability of at least one yellow ball (no prize)
noncomputable def probability_no_prize : ℚ := 
  1 - probability_first_prize - probability_second_prize - probability_third_prize

-- Probability distribution and expectation for number of winners X
noncomputable def winning_probability : ℚ := probability_first_prize + probability_second_prize + probability_third_prize

noncomputable def P_X (n : ℕ) : ℚ :=
  if n = 0 then (7/9)^3
  else if n = 1 then 3 * (2/9) * (7/9)^2
  else if n = 2 then 3 * (2/9)^2 * (7/9)
  else if n = 3 then (2/9)^3
  else 0

noncomputable def expectation_X : ℚ := 
  3 * winning_probability

-- Lean statements
theorem probability_of_second_ball_red_is_correct :
  probability_second_ball_red = 1 / 5 := by
  sorry

theorem probabilities_of_winning_prizes :
  probability_first_prize = 2 / 15 ∧
  probability_second_prize = 1 / 45 ∧
  probability_third_prize = 1 / 15 := by
  sorry

theorem distribution_and_expectation_of_X :
  P_X 0 = 343 / 729 ∧
  P_X 1 = 294 / 729 ∧
  P_X 2 = 84 / 729 ∧
  P_X 3 = 8 / 729 ∧
  expectation_X = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l1378_137808


namespace NUMINAMATH_GPT_vehicle_speed_l1378_137894

theorem vehicle_speed (distance : ℝ) (time : ℝ) (h_dist : distance = 150) (h_time : time = 0.75) : distance / time = 200 :=
  by
    sorry

end NUMINAMATH_GPT_vehicle_speed_l1378_137894
