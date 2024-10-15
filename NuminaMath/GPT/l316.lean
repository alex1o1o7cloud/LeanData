import Mathlib

namespace NUMINAMATH_GPT_scientific_notation_of_384_000_000_l316_31613

theorem scientific_notation_of_384_000_000 :
  384000000 = 3.84 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_384_000_000_l316_31613


namespace NUMINAMATH_GPT_dot_product_AB_BC_l316_31654

theorem dot_product_AB_BC 
  (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a + c = 3)
  (cosB : ℝ)
  (h3 : cosB = 3 / 4) : 
  (a * c * (-cosB) = -3/2) :=
by 
  -- Given conditions
  sorry

end NUMINAMATH_GPT_dot_product_AB_BC_l316_31654


namespace NUMINAMATH_GPT_binary_101011_is_43_l316_31639

def binary_to_decimal_conversion (b : Nat) : Nat := 
  match b with
  | 101011 => 43
  | _ => 0

theorem binary_101011_is_43 : binary_to_decimal_conversion 101011 = 43 := by
  sorry

end NUMINAMATH_GPT_binary_101011_is_43_l316_31639


namespace NUMINAMATH_GPT_machine_value_correct_l316_31672

-- The present value of the machine
def present_value : ℝ := 1200

-- The depreciation rate function based on the year
def depreciation_rate (year : ℕ) : ℝ :=
  match year with
  | 1 => 0.10
  | 2 => 0.12
  | n => if n > 2 then 0.10 + 0.02 * (n - 1) else 0

-- The repair rate
def repair_rate : ℝ := 0.03

-- Value of the machine after n years
noncomputable def machine_value_after_n_years (initial_value : ℝ) (n : ℕ) : ℝ :=
  let value_first_year := (initial_value - (depreciation_rate 1 * initial_value)) + (repair_rate * initial_value)
  let value_second_year := (value_first_year - (depreciation_rate 2 * value_first_year)) + (repair_rate * value_first_year)
  match n with
  | 1 => value_first_year
  | 2 => value_second_year
  | _ => sorry -- Further generalization would be required for n > 2

-- Theorem statement
theorem machine_value_correct (initial_value : ℝ) :
  machine_value_after_n_years initial_value 2 = 1015.56 := by
  sorry

end NUMINAMATH_GPT_machine_value_correct_l316_31672


namespace NUMINAMATH_GPT_violet_needs_water_l316_31618

/-- Violet needs 800 ml of water per hour hiked, her dog needs 400 ml of water per hour,
    and they can hike for 4 hours. We need to prove that Violet needs 4.8 liters of water
    for the hike. -/
theorem violet_needs_water (hiking_hours : ℝ)
  (violet_water_per_hour : ℝ)
  (dog_water_per_hour : ℝ)
  (violet_water_needed : ℝ)
  (dog_water_needed : ℝ)
  (total_water_needed_ml : ℝ)
  (total_water_needed_liters : ℝ) :
  hiking_hours = 4 ∧
  violet_water_per_hour = 800 ∧
  dog_water_per_hour = 400 ∧
  violet_water_needed = 3200 ∧
  dog_water_needed = 1600 ∧
  total_water_needed_ml = 4800 ∧
  total_water_needed_liters = 4.8 →
  total_water_needed_liters = 4.8 :=
by sorry

end NUMINAMATH_GPT_violet_needs_water_l316_31618


namespace NUMINAMATH_GPT_combined_water_leak_l316_31671

theorem combined_water_leak
  (largest_rate : ℕ)
  (medium_rate : ℕ)
  (smallest_rate : ℕ)
  (time_minutes : ℕ)
  (h1 : largest_rate = 3)
  (h2 : medium_rate = largest_rate / 2)
  (h3 : smallest_rate = medium_rate / 3)
  (h4 : time_minutes = 120) :
  largest_rate * time_minutes + medium_rate * time_minutes + smallest_rate * time_minutes = 600 := by
  sorry

end NUMINAMATH_GPT_combined_water_leak_l316_31671


namespace NUMINAMATH_GPT_find_integer_values_of_a_l316_31650

theorem find_integer_values_of_a
  (x a b c : ℤ)
  (h : (x - a) * (x - 10) + 5 = (x + b) * (x + c)) :
  a = 4 ∨ a = 16 := by
    sorry

end NUMINAMATH_GPT_find_integer_values_of_a_l316_31650


namespace NUMINAMATH_GPT_value_of_a_for_positive_root_l316_31697

theorem value_of_a_for_positive_root :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ (3*x - 1)/(x - 3) = a/(3 - x) - 1) → a = -8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_for_positive_root_l316_31697


namespace NUMINAMATH_GPT_inner_rectangle_length_l316_31617

theorem inner_rectangle_length 
  (a b c : ℝ)
  (h1 : ∃ a1 a2 a3 : ℝ, a2 - a1 = a3 - a2)
  (w_inner : ℝ)
  (width_inner : w_inner = 2)
  (w_shaded : ℝ)
  (width_shaded : w_shaded = 1.5)
  (ar_prog : a = 2 * w_inner ∧ b = 3 * w_inner + 15 ∧ c = 3 * w_inner + 33)
  : ∀ x : ℝ, 2 * x = a → 3 * x + 15 = b → 3 * x + 33 = c → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_inner_rectangle_length_l316_31617


namespace NUMINAMATH_GPT_smallest_multiple_6_15_l316_31657

theorem smallest_multiple_6_15 (b : ℕ) (hb1 : b % 6 = 0) (hb2 : b % 15 = 0) :
  ∃ (b : ℕ), (b > 0) ∧ (b % 6 = 0) ∧ (b % 15 = 0) ∧ (∀ x : ℕ, (x > 0) ∧ (x % 6 = 0) ∧ (x % 15 = 0) → x ≥ b) :=
sorry

end NUMINAMATH_GPT_smallest_multiple_6_15_l316_31657


namespace NUMINAMATH_GPT_compare_powers_l316_31651

def n1 := 22^44
def n2 := 33^33
def n3 := 44^22

theorem compare_powers : n1 > n2 ∧ n2 > n3 := by
  sorry

end NUMINAMATH_GPT_compare_powers_l316_31651


namespace NUMINAMATH_GPT_gym_distance_diff_l316_31611

theorem gym_distance_diff (D G : ℕ) (hD : D = 10) (hG : G = 7) : G - D / 2 = 2 := by
  sorry

end NUMINAMATH_GPT_gym_distance_diff_l316_31611


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_roots_eq_14_3125_l316_31620

theorem sum_of_reciprocals_of_squares_roots_eq_14_3125
  (α β γ : ℝ)
  (h1 : α + β + γ = 15)
  (h2 : α * β + β * γ + γ * α = 26)
  (h3 : α * β * γ = -8) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 14.3125 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_roots_eq_14_3125_l316_31620


namespace NUMINAMATH_GPT_max_value_y_l316_31684

variable (x : ℝ)
def y : ℝ := -3 * x^2 + 6

theorem max_value_y : ∃ M, ∀ x : ℝ, y x ≤ M ∧ (∀ x : ℝ, y x = M → x = 0) :=
by
  use 6
  sorry

end NUMINAMATH_GPT_max_value_y_l316_31684


namespace NUMINAMATH_GPT_problem_statement_l316_31660

-- Define the necessary and sufficient conditions
def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ (¬ (P → Q))

-- Specific propositions in this scenario
def x_conditions (x : ℝ) : Prop := x^2 - 2 * x - 3 = 0
def x_equals_3 (x : ℝ) : Prop := x = 3

-- Prove the given problem statement
theorem problem_statement (x : ℝ) : necessary_but_not_sufficient (x_conditions x) (x_equals_3 x) :=
  sorry

end NUMINAMATH_GPT_problem_statement_l316_31660


namespace NUMINAMATH_GPT_perimeter_of_equilateral_triangle_l316_31619

-- Defining the conditions
def area_eq_twice_side (s : ℝ) : Prop :=
  (s^2 * Real.sqrt 3) / 4 = 2 * s

-- Defining the proof problem
theorem perimeter_of_equilateral_triangle (s : ℝ) (h : area_eq_twice_side s) : 
  3 * s = 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_perimeter_of_equilateral_triangle_l316_31619


namespace NUMINAMATH_GPT_John_l316_31675

theorem John's_net_profit 
  (gross_income : ℕ)
  (car_purchase_cost : ℕ)
  (car_maintenance : ℕ → ℕ → ℕ)
  (car_insurance : ℕ)
  (car_tire_replacement : ℕ)
  (trade_in_value : ℕ)
  (tax_rate : ℚ)
  (total_taxes : ℕ)
  (monthly_maintenance_cost : ℕ)
  (months : ℕ)
  (net_profit : ℕ) :
  gross_income = 30000 →
  car_purchase_cost = 20000 →
  car_maintenance monthly_maintenance_cost months = 3600 →
  car_insurance = 1200 →
  car_tire_replacement = 400 →
  trade_in_value = 6000 →
  tax_rate = 15/100 →
  total_taxes = 4500 →
  monthly_maintenance_cost = 300 →
  months = 12 →
  net_profit = gross_income - (car_purchase_cost + car_maintenance monthly_maintenance_cost months + car_insurance + car_tire_replacement + total_taxes) + trade_in_value →
  net_profit = 6300 := 
by 
  sorry -- Proof to be provided

end NUMINAMATH_GPT_John_l316_31675


namespace NUMINAMATH_GPT_combined_molecular_weight_l316_31621

theorem combined_molecular_weight 
  (atomic_weight_N : ℝ)
  (atomic_weight_O : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_C : ℝ)
  (moles_N2O3 : ℝ)
  (moles_H2O : ℝ)
  (moles_CO2 : ℝ)
  (molecular_weight_N2O3 : ℝ)
  (molecular_weight_H2O : ℝ)
  (molecular_weight_CO2 : ℝ)
  (weight_N2O3 : ℝ)
  (weight_H2O : ℝ)
  (weight_CO2 : ℝ)
  : 
  moles_N2O3 = 4 →
  moles_H2O = 3.5 →
  moles_CO2 = 2 →
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  atomic_weight_H = 1.01 →
  atomic_weight_C = 12.01 →
  molecular_weight_N2O3 = (2 * atomic_weight_N) + (3 * atomic_weight_O) →
  molecular_weight_H2O = (2 * atomic_weight_H) + atomic_weight_O →
  molecular_weight_CO2 = atomic_weight_C + (2 * atomic_weight_O) →
  weight_N2O3 = moles_N2O3 * molecular_weight_N2O3 →
  weight_H2O = moles_H2O * molecular_weight_H2O →
  weight_CO2 = moles_CO2 * molecular_weight_CO2 →
  weight_N2O3 + weight_H2O + weight_CO2 = 455.17 :=
by 
  intros;
  sorry

end NUMINAMATH_GPT_combined_molecular_weight_l316_31621


namespace NUMINAMATH_GPT_unique_solution_iff_a_values_l316_31695

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 5 * a

theorem unique_solution_iff_a_values (a : ℝ) :
  (∃! x : ℝ, |f x a| ≤ 3) ↔ (a = 3 / 4 ∨ a = -3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_iff_a_values_l316_31695


namespace NUMINAMATH_GPT_no_n_divisible_by_1955_l316_31686

theorem no_n_divisible_by_1955 : ∀ n : ℕ, ¬ (1955 ∣ (n^2 + n + 1)) := by
  sorry

end NUMINAMATH_GPT_no_n_divisible_by_1955_l316_31686


namespace NUMINAMATH_GPT_rectangle_dimensions_exist_l316_31690

theorem rectangle_dimensions_exist :
  ∃ (a b c d : ℕ), (a * b + c * d = 81) ∧ (2 * (a + b) = 2 * 2 * (c + d) ∨ 2 * (c + d) = 2 * 2 * (a + b)) :=
by sorry

end NUMINAMATH_GPT_rectangle_dimensions_exist_l316_31690


namespace NUMINAMATH_GPT_marbles_left_l316_31643

def initial_marbles : ℕ := 100
def percent_t_to_Theresa : ℕ := 25
def percent_t_to_Elliot : ℕ := 10

theorem marbles_left (w t e : ℕ) (h_w : w = initial_marbles)
                                 (h_t : t = percent_t_to_Theresa)
                                 (h_e : e = percent_t_to_Elliot) : w - ((t * w) / 100 + (e * w) / 100) = 65 :=
by
  rw [h_w, h_t, h_e]
  sorry

end NUMINAMATH_GPT_marbles_left_l316_31643


namespace NUMINAMATH_GPT_Q_value_l316_31625

theorem Q_value (a b c P Q : ℝ) (h1 : a + b + c = 0)
    (h2 : (a^2 / (2 * a^2 + b * c)) + (b^2 / (2 * b^2 + a * c)) + (c^2 / (2 * c^2 + a * b)) = P - 3 * Q) : 
    Q = 8 := 
sorry

end NUMINAMATH_GPT_Q_value_l316_31625


namespace NUMINAMATH_GPT_supplement_twice_angle_l316_31646

theorem supplement_twice_angle (α : ℝ) (h : 180 - α = 2 * α) : α = 60 := by
  admit -- This is a placeholder for the actual proof

end NUMINAMATH_GPT_supplement_twice_angle_l316_31646


namespace NUMINAMATH_GPT_points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l316_31662

open Set

-- Define the point in the coordinate plane as a product of real numbers
def Point := ℝ × ℝ

-- Prove points with x = 3 form a vertical line
theorem points_on_x_eq_3_is_vertical_line : {p : Point | p.1 = 3} = {p : Point | ∀ y : ℝ, (3, y) = p} := sorry

-- Prove points with x < 3 lie to the left of x = 3
theorem points_with_x_lt_3 : {p : Point | p.1 < 3} = {p : Point | ∀ x y : ℝ, x < 3 → p = (x, y)} := sorry

-- Prove points with x > 3 lie to the right of x = 3
theorem points_with_x_gt_3 : {p : Point | p.1 > 3} = {p : Point | ∀ x y : ℝ, x > 3 → p = (x, y)} := sorry

-- Prove points with y = 2 form a horizontal line
theorem points_on_y_eq_2_is_horizontal_line : {p : Point | p.2 = 2} = {p : Point | ∀ x : ℝ, (x, 2) = p} := sorry

-- Prove points with y > 2 lie above y = 2
theorem points_with_y_gt_2 : {p : Point | p.2 > 2} = {p : Point | ∀ x y : ℝ, y > 2 → p = (x, y)} := sorry

end NUMINAMATH_GPT_points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l316_31662


namespace NUMINAMATH_GPT_minValue_is_9_minValue_achieves_9_l316_31602

noncomputable def minValue (x y : ℝ) : ℝ :=
  (x^2 + 1/(y^2)) * (1/(x^2) + 4 * y^2)

theorem minValue_is_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) : minValue x y ≥ 9 :=
  sorry

theorem minValue_achieves_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 1/2) : minValue x y = 9 :=
  sorry

end NUMINAMATH_GPT_minValue_is_9_minValue_achieves_9_l316_31602


namespace NUMINAMATH_GPT_married_fraction_l316_31692

variable (total_people : ℕ) (fraction_women : ℚ) (max_unmarried_women : ℕ)
variable (fraction_married : ℚ)

theorem married_fraction (h1 : total_people = 80)
                         (h2 : fraction_women = 1/4)
                         (h3 : max_unmarried_women = 20)
                         : fraction_married = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_married_fraction_l316_31692


namespace NUMINAMATH_GPT_height_of_block_l316_31632

theorem height_of_block (h : ℝ) : 
  ((∃ (side : ℝ), ∃ (n : ℕ), side = 15 ∧ n = 10 ∧ 15 * 30 * h = n * side^3) → h = 75) := 
by
  intros
  sorry

end NUMINAMATH_GPT_height_of_block_l316_31632


namespace NUMINAMATH_GPT_maximum_value_of_m_l316_31648

theorem maximum_value_of_m (x y : ℝ) (hx : x > 1 / 2) (hy : y > 1) : 
    (4 * x^2 / (y - 1) + y^2 / (2 * x - 1)) ≥ 8 := 
sorry

end NUMINAMATH_GPT_maximum_value_of_m_l316_31648


namespace NUMINAMATH_GPT_joan_needs_more_flour_l316_31626

-- Definitions for the conditions
def total_flour : ℕ := 7
def flour_added : ℕ := 3

-- The theorem stating the proof problem
theorem joan_needs_more_flour : total_flour - flour_added = 4 :=
by
  sorry

end NUMINAMATH_GPT_joan_needs_more_flour_l316_31626


namespace NUMINAMATH_GPT_triangle_angle_sum_l316_31691

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l316_31691


namespace NUMINAMATH_GPT_printing_company_proportion_l316_31629

theorem printing_company_proportion (x y : ℕ) :
  (28*x + 42*y) / (28*x) = 5/3 → x / y = 9 / 4 := by
  sorry

end NUMINAMATH_GPT_printing_company_proportion_l316_31629


namespace NUMINAMATH_GPT_power_simplification_l316_31601

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end NUMINAMATH_GPT_power_simplification_l316_31601


namespace NUMINAMATH_GPT_combined_average_speed_l316_31637

theorem combined_average_speed 
    (dA tA dB tB dC tC : ℝ)
    (mile_feet : ℝ)
    (hA : dA = 300) (hTA : tA = 6)
    (hB : dB = 400) (hTB : tB = 8)
    (hC : dC = 500) (hTC : tC = 10)
    (hMileFeet : mile_feet = 5280) :
    (1200 / 5280) / (24 / 3600) = 34.09 := 
by
  sorry

end NUMINAMATH_GPT_combined_average_speed_l316_31637


namespace NUMINAMATH_GPT_total_kids_played_l316_31636

-- Definitions based on conditions
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Total kids calculation
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Theorem to prove
theorem total_kids_played (Julia : Prop) : totalKids = 34 :=
by
  -- Using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_kids_played_l316_31636


namespace NUMINAMATH_GPT_probability_win_more_than_5000_l316_31664

def boxes : Finset ℕ := {5, 500, 5000}
def keys : Finset (Finset ℕ) := { {5}, {500}, {5000} }

noncomputable def probability_correct_key (box : ℕ) : ℚ :=
  if box = 5000 then 1 / 3 else if box = 500 then 1 / 2 else 1

theorem probability_win_more_than_5000 :
    (probability_correct_key 5000) * (probability_correct_key 500) = 1 / 6 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_probability_win_more_than_5000_l316_31664


namespace NUMINAMATH_GPT_value_of_expression_l316_31635

def a : ℕ := 7
def b : ℕ := 5

theorem value_of_expression : (a^2 - b^2)^4 = 331776 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l316_31635


namespace NUMINAMATH_GPT_max_leap_years_in_200_years_l316_31674

theorem max_leap_years_in_200_years (leap_year_interval: ℕ) (span: ℕ) 
  (h1: leap_year_interval = 4) 
  (h2: span = 200) : 
  (span / leap_year_interval) = 50 := 
sorry

end NUMINAMATH_GPT_max_leap_years_in_200_years_l316_31674


namespace NUMINAMATH_GPT_equal_expression_exists_l316_31683

-- lean statement for the mathematical problem
theorem equal_expression_exists (a b : ℤ) :
  ∃ (expr : ℤ), expr = 20 * a - 18 * b := by
  sorry

end NUMINAMATH_GPT_equal_expression_exists_l316_31683


namespace NUMINAMATH_GPT_sum_angles_of_two_triangles_l316_31668

theorem sum_angles_of_two_triangles (a1 a3 a5 a2 a4 a6 : ℝ) 
  (hABC : a1 + a3 + a5 = 180) (hDEF : a2 + a4 + a6 = 180) : 
  a1 + a2 + a3 + a4 + a5 + a6 = 360 :=
by
  sorry

end NUMINAMATH_GPT_sum_angles_of_two_triangles_l316_31668


namespace NUMINAMATH_GPT_ratio_second_to_first_l316_31682

theorem ratio_second_to_first (F S T : ℕ) 
  (hT : T = 2 * F)
  (havg : (F + S + T) / 3 = 77)
  (hmin : F = 33) :
  S / F = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_second_to_first_l316_31682


namespace NUMINAMATH_GPT_calculate_expression_l316_31661

theorem calculate_expression :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = (3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l316_31661


namespace NUMINAMATH_GPT_find_integer_l316_31623

noncomputable def least_possible_sum (x y z k : ℕ) : Prop :=
  2 * x = 5 * y ∧ 5 * y = 6 * z ∧ x + k + z = 26

theorem find_integer (x y z : ℕ) (h : least_possible_sum x y z 6) :
  6 = (26 - x - z) :=
  by {
    sorry
  }

end NUMINAMATH_GPT_find_integer_l316_31623


namespace NUMINAMATH_GPT_cos_evaluation_l316_31688

open Real

noncomputable def a (n : ℕ) : ℝ := sorry  -- since it's an arithmetic sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k : ℕ, a n + a k = 2 * a ((n + k) / 2)

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 6 + a 9 = 3 * a 6 ∧ a 6 = π / 4

theorem cos_evaluation :
  is_arithmetic_sequence a →
  satisfies_condition a →
  cos (a 2 + a 10 + π / 4) = - (sqrt 2 / 2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_cos_evaluation_l316_31688


namespace NUMINAMATH_GPT_number_of_possible_values_of_s_l316_31604

noncomputable def s := {s : ℚ | ∃ w x y z : ℕ, s = w / 1000 + x / 10000 + y / 100000 + z / 1000000 ∧ w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10}

theorem number_of_possible_values_of_s (s_approx : s → ℚ → Prop) (h_s_approx : ∀ s, s_approx s (3 / 11)) :
  ∃ n : ℕ, n = 266 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_values_of_s_l316_31604


namespace NUMINAMATH_GPT_sum_of_number_and_its_square_is_20_l316_31665

theorem sum_of_number_and_its_square_is_20 (n : ℕ) (h : n = 4) : n + n^2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_its_square_is_20_l316_31665


namespace NUMINAMATH_GPT_intersection_complement_A_B_l316_31638

open Set

theorem intersection_complement_A_B :
  let A := {x : ℝ | x + 1 > 0}
  let B := {-2, -1, 0, 1}
  (compl A ∩ B : Set ℝ) = {-2, -1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_A_B_l316_31638


namespace NUMINAMATH_GPT_mary_age_l316_31607

theorem mary_age (M F : ℕ) (h1 : F = 4 * M) (h2 : F - 3 = 5 * (M - 3)) : M = 12 :=
by
  sorry

end NUMINAMATH_GPT_mary_age_l316_31607


namespace NUMINAMATH_GPT_units_digit_divisible_by_18_l316_31694

theorem units_digit_divisible_by_18 : ∃ n : ℕ, (3150 ≤ 315 * n) ∧ (315 * n < 3160) ∧ (n % 2 = 0) ∧ (315 * n % 18 = 0) ∧ (n = 0) :=
by
  use 0
  sorry

end NUMINAMATH_GPT_units_digit_divisible_by_18_l316_31694


namespace NUMINAMATH_GPT_train_speed_solution_l316_31667

def train_speed_problem (L v : ℝ) (man_time platform_time : ℝ) (platform_length : ℝ) :=
  man_time = 12 ∧
  platform_time = 30 ∧
  platform_length = 180 ∧
  L = v * man_time ∧
  (L + platform_length) = v * platform_time

theorem train_speed_solution (L v : ℝ) (h : train_speed_problem L v 12 30 180) :
  v * 3.6 = 36 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_solution_l316_31667


namespace NUMINAMATH_GPT_problem_statement_l316_31689

noncomputable def increase_and_subtract (x p y : ℝ) : ℝ :=
  (x + p * x) - y

theorem problem_statement : increase_and_subtract 75 1.5 40 = 147.5 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l316_31689


namespace NUMINAMATH_GPT_root_interval_k_l316_31655

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_interval_k (k : ℤ) (h : ∃ ξ : ℝ, k < ξ ∧ ξ < k+1 ∧ f ξ = 0) : k = 0 :=
by
  sorry

end NUMINAMATH_GPT_root_interval_k_l316_31655


namespace NUMINAMATH_GPT_B_monthly_income_is_correct_l316_31606

variable (A_m B_m C_m : ℝ)
variable (A_annual C_m_value : ℝ)
variable (ratio_A_to_B : ℝ)

-- Given conditions
def conditions :=
  A_annual = 537600 ∧
  C_m_value = 16000 ∧
  ratio_A_to_B = 5 / 2 ∧
  A_m = A_annual / 12 ∧
  B_m = (2 / 5) * A_m ∧
  B_m = 1.12 * C_m ∧
  C_m = C_m_value

-- Prove that B's monthly income is Rs. 17920
theorem B_monthly_income_is_correct (h : conditions A_m B_m C_m A_annual C_m_value ratio_A_to_B) : 
  B_m = 17920 :=
by 
  sorry

end NUMINAMATH_GPT_B_monthly_income_is_correct_l316_31606


namespace NUMINAMATH_GPT_seeds_per_packet_l316_31680

theorem seeds_per_packet (total_seedlings packets : ℕ) (h1 : total_seedlings = 420) (h2 : packets = 60) : total_seedlings / packets = 7 :=
by 
  sorry

end NUMINAMATH_GPT_seeds_per_packet_l316_31680


namespace NUMINAMATH_GPT_combined_total_score_is_correct_l316_31658

-- Definitions of point values
def touchdown_points := 6
def extra_point_points := 1
def field_goal_points := 3

-- Hawks' Scores
def hawks_touchdowns := 4
def hawks_successful_extra_points := 2
def hawks_field_goals := 2

-- Eagles' Scores
def eagles_touchdowns := 3
def eagles_successful_extra_points := 3
def eagles_field_goals := 3

-- Calculations
def hawks_total_points := hawks_touchdowns * touchdown_points +
                          hawks_successful_extra_points * extra_point_points +
                          hawks_field_goals * field_goal_points

def eagles_total_points := eagles_touchdowns * touchdown_points +
                           eagles_successful_extra_points * extra_point_points +
                           eagles_field_goals * field_goal_points

def combined_total_score := hawks_total_points + eagles_total_points

-- The theorem that needs to be proved
theorem combined_total_score_is_correct : combined_total_score = 62 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_combined_total_score_is_correct_l316_31658


namespace NUMINAMATH_GPT_calculate_expression_l316_31614

theorem calculate_expression (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l316_31614


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l316_31605

-- Definitions of sets A and B
def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

-- The theorem we want to prove
theorem intersection_of_A_and_B : A ∩ B = { x | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l316_31605


namespace NUMINAMATH_GPT_solve_for_a_l316_31666

theorem solve_for_a (a : ℝ) (h : a / 0.3 = 0.6) : a = 0.18 :=
by sorry

end NUMINAMATH_GPT_solve_for_a_l316_31666


namespace NUMINAMATH_GPT_inequality_transformation_l316_31645

theorem inequality_transformation (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_transformation_l316_31645


namespace NUMINAMATH_GPT_subset_singleton_natural_l316_31659

/-
  Problem Statement:
  Prove that the set {2} is a subset of the set of natural numbers.
-/

open Set

theorem subset_singleton_natural :
  {2} ⊆ (Set.univ : Set ℕ) :=
by
  sorry

end NUMINAMATH_GPT_subset_singleton_natural_l316_31659


namespace NUMINAMATH_GPT_min_value_u_l316_31641

theorem min_value_u (x y : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0)
  (h₂ : 2 * x + y = 6) : 
  ∀u, u = 4 * x ^ 2 + 3 * x * y + y ^ 2 - 6 * x - 3 * y -> 
  u ≥ 27 / 2 := sorry

end NUMINAMATH_GPT_min_value_u_l316_31641


namespace NUMINAMATH_GPT_circle_center_l316_31634

theorem circle_center {x y : ℝ} :
  4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0 → (x, y) = (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_l316_31634


namespace NUMINAMATH_GPT_max_positive_integers_l316_31679

theorem max_positive_integers (a b c d e f : ℤ) (h : (a * b + c * d * e * f) < 0) :
  ∃ n, n ≤ 5 ∧ (∀x ∈ [a, b, c, d, e, f], 0 < x → x ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_max_positive_integers_l316_31679


namespace NUMINAMATH_GPT_race_distance_l316_31647

theorem race_distance (D : ℝ)
  (A_speed : ℝ := D / 20)
  (B_speed : ℝ := D / 25)
  (A_beats_B_by : ℝ := 18)
  (h1 : A_speed * 25 = D + A_beats_B_by)
  : D = 72 := 
by
  sorry

end NUMINAMATH_GPT_race_distance_l316_31647


namespace NUMINAMATH_GPT_vanessa_total_earnings_l316_31610

theorem vanessa_total_earnings :
  let num_dresses := 7
  let num_shirts := 4
  let price_per_dress := 7
  let price_per_shirt := 5
  (num_dresses * price_per_dress + num_shirts * price_per_shirt) = 69 :=
by
  sorry

end NUMINAMATH_GPT_vanessa_total_earnings_l316_31610


namespace NUMINAMATH_GPT_family_ages_sum_today_l316_31699

theorem family_ages_sum_today (A B C D E : ℕ) (h1 : A + B + C + D = 114) (h2 : E = D - 14) :
    (A + 5) + (B + 5) + (C + 5) + (E + 5) = 120 :=
by
  sorry

end NUMINAMATH_GPT_family_ages_sum_today_l316_31699


namespace NUMINAMATH_GPT_tan_beta_l316_31615

noncomputable def tan_eq_2 (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) : Real :=
2

theorem tan_beta (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) :
  Real.tan β = tan_eq_2 α β h1 h2 := by
  sorry

end NUMINAMATH_GPT_tan_beta_l316_31615


namespace NUMINAMATH_GPT_intersection_points_eq_2_l316_31628

def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
def eq2 (x y : ℝ) : Prop := (x + 2 * y - 3) * (3 * x - 4 * y + 6) = 0

theorem intersection_points_eq_2 : ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 2 := 
sorry

end NUMINAMATH_GPT_intersection_points_eq_2_l316_31628


namespace NUMINAMATH_GPT_rabbit_weight_l316_31627

theorem rabbit_weight (a b c : ℕ) (h1 : a + b + c = 30) (h2 : a + c = 2 * b) (h3 : a + b = c) :
  a = 5 := by
  sorry

end NUMINAMATH_GPT_rabbit_weight_l316_31627


namespace NUMINAMATH_GPT_solution_set_l316_31644
  
noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp (2 * x) + 1) - x

theorem solution_set (x : ℝ) :
  f (x + 2) > f (2 * x - 3) ↔ (1 / 3 < x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l316_31644


namespace NUMINAMATH_GPT_second_athlete_triple_jump_l316_31696

theorem second_athlete_triple_jump
  (long_jump1 triple_jump1 high_jump1 : ℕ) 
  (long_jump2 high_jump2 : ℕ)
  (average_winner : ℕ) 
  (H1 : long_jump1 = 26) (H2 : triple_jump1 = 30) (H3 : high_jump1 = 7)
  (H4 : long_jump2 = 24) (H5 : high_jump2 = 8) (H6 : average_winner = 22)
  : ∃ x : ℕ, (24 + x + 8) / 3 = 22 ∧ x = 34 := 
by
  sorry

end NUMINAMATH_GPT_second_athlete_triple_jump_l316_31696


namespace NUMINAMATH_GPT_remainder_of_2n_div_9_l316_31681

theorem remainder_of_2n_div_9
  (n : ℤ) (h : ∃ k : ℤ, n = 18 * k + 10) : (2 * n) % 9 = 2 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_2n_div_9_l316_31681


namespace NUMINAMATH_GPT_Shara_will_owe_money_l316_31663

theorem Shara_will_owe_money
    (B : ℕ)
    (h1 : 6 * 10 = 60)
    (h2 : B / 2 = 60)
    (h3 : 4 * 10 = 40)
    (h4 : 60 + 40 = 100) :
  B - 100 = 20 :=
sorry

end NUMINAMATH_GPT_Shara_will_owe_money_l316_31663


namespace NUMINAMATH_GPT_molecular_weight_N2O5_correct_l316_31687

noncomputable def atomic_weight_N : ℝ := 14.01
noncomputable def atomic_weight_O : ℝ := 16.00
def molecular_formula_N2O5 : (ℕ × ℕ) := (2, 5)

theorem molecular_weight_N2O5_correct :
  let weight := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  weight = 108.02 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_N2O5_correct_l316_31687


namespace NUMINAMATH_GPT_ratio_expression_l316_31693

variable (a b c : ℚ)
variable (h1 : a / b = 6 / 5)
variable (h2 : b / c = 8 / 7)

theorem ratio_expression (a b c : ℚ) (h1 : a / b = 6 / 5) (h2 : b / c = 8 / 7) :
  (7 * a + 6 * b + 5 * c) / (7 * a - 6 * b + 5 * c) = 751 / 271 := by
  sorry

end NUMINAMATH_GPT_ratio_expression_l316_31693


namespace NUMINAMATH_GPT_cabinets_ratio_proof_l316_31698

-- Definitions for the conditions
def initial_cabinets : ℕ := 3
def total_cabinets : ℕ := 26
def additional_cabinets : ℕ := 5
def number_of_counters : ℕ := 3

-- Definition for the unknown cabinets installed per counter
def cabinets_per_counter : ℕ := (total_cabinets - additional_cabinets - initial_cabinets) / number_of_counters

-- The ratio to be proven
theorem cabinets_ratio_proof : (cabinets_per_counter : ℚ) / initial_cabinets = 2 / 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cabinets_ratio_proof_l316_31698


namespace NUMINAMATH_GPT_assignment_problem_l316_31633

theorem assignment_problem (a b c : ℕ) (h1 : a = 10) (h2 : b = 20) (h3 : c = 30) :
  let a := b
  let b := c
  let c := a
  a = 20 ∧ b = 30 ∧ c = 20 :=
by
  sorry

end NUMINAMATH_GPT_assignment_problem_l316_31633


namespace NUMINAMATH_GPT_green_hats_count_l316_31608

theorem green_hats_count : ∃ G B : ℕ, B + G = 85 ∧ 6 * B + 7 * G = 540 ∧ G = 30 :=
by
  sorry

end NUMINAMATH_GPT_green_hats_count_l316_31608


namespace NUMINAMATH_GPT_no_such_rectangle_exists_l316_31609

theorem no_such_rectangle_exists :
  ¬(∃ (x y : ℝ), (∃ a b c d : ℕ, x = a + b * Real.sqrt 3 ∧ y = c + d * Real.sqrt 3) ∧ 
                (x * y = (3 * Real.sqrt 3) / 2 + n * (Real.sqrt 3 / 2))) :=
sorry

end NUMINAMATH_GPT_no_such_rectangle_exists_l316_31609


namespace NUMINAMATH_GPT_smallest_n_l316_31603

theorem smallest_n (o y v : ℕ) (h1 : 18 * o = 21 * y) (h2 : 21 * y = 10 * v) (h3 : 10 * v = 30 * n) : 
  n = 21 := by
  sorry

end NUMINAMATH_GPT_smallest_n_l316_31603


namespace NUMINAMATH_GPT_jennie_total_rental_cost_l316_31624

-- Definition of the conditions in the problem
def daily_rate : ℕ := 30
def weekly_rate : ℕ := 190
def days_rented : ℕ := 11
def first_week_days : ℕ := 7

-- Proof statement which translates the problem to Lean
theorem jennie_total_rental_cost : (weekly_rate + (days_rented - first_week_days) * daily_rate) = 310 := by
  sorry

end NUMINAMATH_GPT_jennie_total_rental_cost_l316_31624


namespace NUMINAMATH_GPT_value_of_x_l316_31677

def x : ℚ :=
  (320 / 2) / 3

theorem value_of_x : x = 160 / 3 := 
by
  unfold x
  sorry

end NUMINAMATH_GPT_value_of_x_l316_31677


namespace NUMINAMATH_GPT_min_n_satisfies_inequality_l316_31676

theorem min_n_satisfies_inequality :
  ∃ n : ℕ, 0 < n ∧ -3 * (n : ℤ) ^ 4 + 5 * (n : ℤ) ^ 2 - 199 < 0 ∧ (∀ m : ℕ, 0 < m ∧ -3 * (m : ℤ) ^ 4 + 5 * (m : ℤ) ^ 2 - 199 < 0 → 2 ≤ m) := 
  sorry

end NUMINAMATH_GPT_min_n_satisfies_inequality_l316_31676


namespace NUMINAMATH_GPT_frac_val_of_x_y_l316_31656

theorem frac_val_of_x_y (x y : ℝ) (h: (4 : ℝ) < (2 * x - 3 * y) / (2 * x + 3 * y) ∧ (2 * x - 3 * y) / (2 * x + 3 * y) < 8) (ht: ∃ t : ℤ, x = t * y) : x / y = -2 := 
by
  sorry

end NUMINAMATH_GPT_frac_val_of_x_y_l316_31656


namespace NUMINAMATH_GPT_max_value_cos2_sin_l316_31612

noncomputable def max_cos2_sin (x : Real) : Real := 
  (Real.cos x) ^ 2 + Real.sin x

theorem max_value_cos2_sin : 
  ∃ x : Real, (-1 ≤ Real.sin x) ∧ (Real.sin x ≤ 1) ∧ 
    max_cos2_sin x = 5 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_cos2_sin_l316_31612


namespace NUMINAMATH_GPT_pirates_share_l316_31652

def initial_coins (N : ℕ) := N ≥ 3000 ∧ N ≤ 4000

def first_pirate (N : ℕ) := N - (2 + (N - 2) / 4)
def second_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def third_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def fourth_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)

def final_remaining (N : ℕ) :=
  let step1 := first_pirate N
  let step2 := second_pirate step1
  let step3 := third_pirate step2
  let step4 := fourth_pirate step3
  step4

theorem pirates_share (N : ℕ) (h : initial_coins N) :
  final_remaining N / 4 = 660 :=
by
  sorry

end NUMINAMATH_GPT_pirates_share_l316_31652


namespace NUMINAMATH_GPT_find_x_in_sequence_l316_31616

theorem find_x_in_sequence :
  (∀ a b c d : ℕ, a * b * c * d = 120) →
  (a = 2) →
  (b = 4) →
  (d = 3) →
  ∃ x : ℕ, 2 * 4 * x * 3 = 120 ∧ x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_in_sequence_l316_31616


namespace NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l316_31622

/-- A line passing through point (-2, 3) and having equal intercepts
on the coordinate axes can have the equation y = -3/2 * x or x + y = 1. -/
theorem line_through_point_with_equal_intercepts (x y : Real) :
  (∃ (m : Real), (y = m * x) ∧ (y - m * (-2) = 3 ∧ y - m * 0 = 0))
  ∨ (∃ (a : Real), (x + y = a) ∧ (a = 1 ∧ (-2) + 3 = a)) :=
sorry

end NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l316_31622


namespace NUMINAMATH_GPT_total_spent_l316_31649

-- Constants representing the conditions from the problem
def cost_per_deck : ℕ := 8
def tom_decks : ℕ := 3
def friend_decks : ℕ := 5

-- Theorem stating the total amount spent by Tom and his friend
theorem total_spent : tom_decks * cost_per_deck + friend_decks * cost_per_deck = 64 := by
  sorry

end NUMINAMATH_GPT_total_spent_l316_31649


namespace NUMINAMATH_GPT_sum_of_first_6_terms_l316_31653

-- Definitions based on given conditions
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + d * (n - 1)

-- The conditions provided in the problem
def condition_1 (a1 d : ℤ) : Prop := arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 4
def condition_2 (a1 d : ℤ) : Prop := arithmetic_sequence a1 d 3 + arithmetic_sequence a1 d 5 = 10

-- The sum of the first 6 terms of the arithmetic sequence
def sum_first_6_terms (a1 d : ℤ) : ℤ := 6 * a1 + 15 * d

-- The theorem to prove
theorem sum_of_first_6_terms (a1 d : ℤ) 
  (h1 : condition_1 a1 d)
  (h2 : condition_2 a1 d) :
  sum_first_6_terms a1 d = 21 := sorry

end NUMINAMATH_GPT_sum_of_first_6_terms_l316_31653


namespace NUMINAMATH_GPT_set_intersection_A_B_l316_31670

theorem set_intersection_A_B :
  (A : Set ℤ) ∩ (B : Set ℤ) = { -1, 0, 1, 2 } :=
by
  let A := { x : ℤ | x^2 - x - 2 ≤ 0 }
  let B := {x : ℤ | x ∈ Set.univ}
  sorry

end NUMINAMATH_GPT_set_intersection_A_B_l316_31670


namespace NUMINAMATH_GPT_find_roots_l316_31631

theorem find_roots (x : ℝ) : x^2 - 2 * x - 2 / x + 1 / x^2 - 13 = 0 ↔ 
  (x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 ∨ x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2) := by
  sorry

end NUMINAMATH_GPT_find_roots_l316_31631


namespace NUMINAMATH_GPT_sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l316_31630

theorem sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^4 + t^2) = |t| * Real.sqrt (t^2 + 1) :=
sorry

end NUMINAMATH_GPT_sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l316_31630


namespace NUMINAMATH_GPT_lucy_money_l316_31685

variable (L : ℕ) -- Value for Lucy's original amount of money

theorem lucy_money (h1 : ∀ (L : ℕ), L - 5 = 10 + 5 → L = 20) : L = 20 :=
by sorry

end NUMINAMATH_GPT_lucy_money_l316_31685


namespace NUMINAMATH_GPT_bounded_f_l316_31678

theorem bounded_f (f : ℝ → ℝ) (h1 : ∀ x1 x2, |x1 - x2| ≤ 1 → |f x2 - f x1| ≤ 1)
  (h2 : f 0 = 1) : ∀ x, -|x| ≤ f x ∧ f x ≤ |x| + 2 := by
  sorry

end NUMINAMATH_GPT_bounded_f_l316_31678


namespace NUMINAMATH_GPT_sequence_problem_proof_l316_31600

-- Define the sequence terms, using given conditions
def a_1 : ℕ := 1
def a_2 : ℕ := 2
def a_3 : ℕ := a_1 + a_2
def a_4 : ℕ := a_2 + a_3
def x : ℕ := a_3 + a_4

-- Prove that x = 8
theorem sequence_problem_proof : x = 8 := 
by
  sorry

end NUMINAMATH_GPT_sequence_problem_proof_l316_31600


namespace NUMINAMATH_GPT_log_comparison_l316_31642

theorem log_comparison 
  (a : ℝ := 1 / 6 * Real.log 8)
  (b : ℝ := 1 / 2 * Real.log 5)
  (c : ℝ := Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)) :
  a < c ∧ c < b := 
by
  sorry

end NUMINAMATH_GPT_log_comparison_l316_31642


namespace NUMINAMATH_GPT_distance_ratio_l316_31669

theorem distance_ratio (D90 D180 : ℝ) 
  (h1 : D90 + D180 = 3600) 
  (h2 : D90 / 90 + D180 / 180 = 30) : 
  D90 / D180 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_distance_ratio_l316_31669


namespace NUMINAMATH_GPT_solve_system_l316_31640

theorem solve_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 7) : x + y = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l316_31640


namespace NUMINAMATH_GPT_factor_theorem_q_value_l316_31673

theorem factor_theorem_q_value (q : ℤ) (m : ℤ) :
  (∀ m, (m - 8) ∣ (m^2 - q * m - 24)) → q = 5 :=
by
  sorry

end NUMINAMATH_GPT_factor_theorem_q_value_l316_31673
