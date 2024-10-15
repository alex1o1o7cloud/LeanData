import Mathlib

namespace NUMINAMATH_GPT_cody_ate_dumplings_l1548_154804

theorem cody_ate_dumplings (initial_dumplings remaining_dumplings : ℕ) (h1 : initial_dumplings = 14) (h2 : remaining_dumplings = 7) : initial_dumplings - remaining_dumplings = 7 :=
by
  sorry

end NUMINAMATH_GPT_cody_ate_dumplings_l1548_154804


namespace NUMINAMATH_GPT_distance_on_dirt_section_distance_on_muddy_section_l1548_154890

section RaceProblem

variables {v_h v_d v_m : ℕ} (initial_gap : ℕ)

-- Problem conditions
def highway_speed := 150 -- km/h
def dirt_road_speed := 60 -- km/h
def muddy_section_speed := 18 -- km/h
def initial_gap_start := 300 -- meters

-- Convert km/h to m/s
def to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Speeds in m/s
def highway_speed_mps := to_m_per_s highway_speed
def dirt_road_speed_mps := to_m_per_s dirt_road_speed
def muddy_section_speed_mps := to_m_per_s muddy_section_speed

-- Questions
theorem distance_on_dirt_section :
  ∃ (d : ℕ), (d = 120) :=
sorry

theorem distance_on_muddy_section :
  ∃ (d : ℕ), (d = 36) :=
sorry

end RaceProblem

end NUMINAMATH_GPT_distance_on_dirt_section_distance_on_muddy_section_l1548_154890


namespace NUMINAMATH_GPT_books_jerry_added_l1548_154882

def initial_action_figures : ℕ := 7
def initial_books : ℕ := 2

theorem books_jerry_added (B : ℕ) (h : initial_action_figures = initial_books + B + 1) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_books_jerry_added_l1548_154882


namespace NUMINAMATH_GPT_stephanie_bills_l1548_154895

theorem stephanie_bills :
  let electricity_bill := 120
  let electricity_paid := 0.80 * electricity_bill
  let gas_bill := 80
  let gas_paid := (3 / 4) * gas_bill
  let additional_gas_payment := 10
  let water_bill := 60
  let water_paid := 0.65 * water_bill
  let internet_bill := 50
  let internet_paid := 6 * 5
  let internet_remaining_before_discount := internet_bill - internet_paid
  let internet_discount := 0.10 * internet_remaining_before_discount
  let phone_bill := 45
  let phone_paid := 0.20 * phone_bill
  let remaining_electricity := electricity_bill - electricity_paid
  let remaining_gas := gas_bill - (gas_paid + additional_gas_payment)
  let remaining_water := water_bill - water_paid
  let remaining_internet := internet_remaining_before_discount - internet_discount
  let remaining_phone := phone_bill - phone_paid
  (remaining_electricity + remaining_gas + remaining_water + remaining_internet + remaining_phone) = 109 :=
by
  sorry

end NUMINAMATH_GPT_stephanie_bills_l1548_154895


namespace NUMINAMATH_GPT_people_with_uncool_parents_l1548_154829

theorem people_with_uncool_parents :
  ∀ (total cool_dads cool_moms cool_both : ℕ),
    total = 50 →
    cool_dads = 25 →
    cool_moms = 30 →
    cool_both = 15 →
    (total - (cool_dads + cool_moms - cool_both)) = 10 := 
by
  intros total cool_dads cool_moms cool_both h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_people_with_uncool_parents_l1548_154829


namespace NUMINAMATH_GPT_second_difference_is_quadratic_l1548_154840

theorem second_difference_is_quadratic (f : ℕ → ℝ) 
  (h : ∀ n : ℕ, (f (n + 2) - 2 * f (n + 1) + f n) = 2) :
  ∃ (a b : ℝ), ∀ (n : ℕ), f n = n^2 + a * n + b :=
by
  sorry

end NUMINAMATH_GPT_second_difference_is_quadratic_l1548_154840


namespace NUMINAMATH_GPT_expenses_of_5_yuan_l1548_154810

-- Define the given condition: income of 5 yuan is +5 yuan
def income (x : Int) : Int := x

-- Define the opposite relationship between income and expenses
def expenses (x : Int) : Int := -income x

-- Proof statement to show that expenses of 5 yuan are -5 yuan, given the above definitions
theorem expenses_of_5_yuan : expenses 5 = -5 := by
  -- The proof is not provided here, so we use sorry to indicate its place
  sorry

end NUMINAMATH_GPT_expenses_of_5_yuan_l1548_154810


namespace NUMINAMATH_GPT_allan_balloons_l1548_154806

theorem allan_balloons (x : ℕ) : 
  (2 + x) + 1 = 6 → x = 3 :=
by
  intro h
  linarith

end NUMINAMATH_GPT_allan_balloons_l1548_154806


namespace NUMINAMATH_GPT_union_M_N_equals_set_x_ge_1_l1548_154889

-- Definitions of M and N based on the conditions from step a)
def M : Set ℝ := { x | x - 2 > 0 }

def N : Set ℝ := { y | ∃ x : ℝ, y = Real.sqrt (x^2 + 1) }

-- Statement of the theorem
theorem union_M_N_equals_set_x_ge_1 : (M ∪ N) = { x : ℝ | x ≥ 1 } := 
sorry

end NUMINAMATH_GPT_union_M_N_equals_set_x_ge_1_l1548_154889


namespace NUMINAMATH_GPT_exponential_comparison_l1548_154819

theorem exponential_comparison
  (a : ℕ := 3^55)
  (b : ℕ := 4^44)
  (c : ℕ := 5^33) :
  c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_exponential_comparison_l1548_154819


namespace NUMINAMATH_GPT_intersection_M_N_l1548_154872

open Set

variable (x : ℝ)
def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def N : Set ℝ := {-2, 0, 2}

theorem intersection_M_N : M ∩ N = {0, 2} := sorry

end NUMINAMATH_GPT_intersection_M_N_l1548_154872


namespace NUMINAMATH_GPT_roots_sum_of_squares_l1548_154885

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h_roots : ∀ x, (3 * x^3 - 4 * x^2 + 3 * x + 7 = 0) → (x = p ∨ x = q ∨ x = r))
  (h_sum : p + q + r = 4 / 3)
  (h_prod_sum : p * q + q * r + r * p = 1)
  (h_prod : p * q * r = -7 / 3) :
  p^2 + q^2 + r^2 = -2 / 9 := 
sorry

end NUMINAMATH_GPT_roots_sum_of_squares_l1548_154885


namespace NUMINAMATH_GPT_total_legs_l1548_154820

-- Define the number of octopuses
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- The total number of legs should be num_octopuses * legs_per_octopus
theorem total_legs : num_octopuses * legs_per_octopus = 40 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_total_legs_l1548_154820


namespace NUMINAMATH_GPT_no_negative_roots_l1548_154803

theorem no_negative_roots (x : ℝ) : 4 * x^4 - 7 * x^3 - 20 * x^2 - 13 * x + 25 ≠ 0 ∨ x ≥ 0 := 
sorry

end NUMINAMATH_GPT_no_negative_roots_l1548_154803


namespace NUMINAMATH_GPT_intersection_correct_l1548_154846

def A (x : ℝ) : Prop := |x| > 4
def B (x : ℝ) : Prop := -2 < x ∧ x ≤ 6
def intersection (x : ℝ) : Prop := B x ∧ A x

theorem intersection_correct :
  ∀ x : ℝ, intersection x ↔ 4 < x ∧ x ≤ 6 := 
by
  sorry

end NUMINAMATH_GPT_intersection_correct_l1548_154846


namespace NUMINAMATH_GPT_max_value_of_f_l1548_154894

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x - 1/2

theorem max_value_of_f : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ (∀ y, (0 ≤ y ∧ y ≤ 2) → f y ≤ f x) ∧ f x = -3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1548_154894


namespace NUMINAMATH_GPT_one_third_of_7_times_9_l1548_154854

theorem one_third_of_7_times_9 : (1 / 3) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_GPT_one_third_of_7_times_9_l1548_154854


namespace NUMINAMATH_GPT_impossible_digit_placement_l1548_154832

-- Define the main variables and assumptions
variable (A B C : ℕ)
variable (h_sum : A + B = 45)
variable (h_segmentSum : 3 * A + B = 6 * C)

-- Define the impossible placement problem
theorem impossible_digit_placement :
  ¬(∃ A B C, A + B = 45 ∧ 3 * A + B = 6 * C ∧ 2 * A = 6 * C - 45) :=
by
  sorry

end NUMINAMATH_GPT_impossible_digit_placement_l1548_154832


namespace NUMINAMATH_GPT_find_other_number_l1548_154860

open BigOperators

noncomputable def other_number (n : ℕ) : Prop := n = 12

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 8 n = 24) (h_hcf : Nat.gcd 8 n = 4) : other_number n := 
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1548_154860


namespace NUMINAMATH_GPT_business_hours_correct_l1548_154813

-- Define the business hours
def start_time : ℕ := 8 * 60 + 30   -- 8:30 in minutes
def end_time : ℕ := 22 * 60 + 30    -- 22:30 in minutes

-- Calculate total business hours in minutes and convert it to hours
def total_business_hours : ℕ := (end_time - start_time) / 60

-- State the business hour condition (which says the total business hour is 15 hours).
def business_hour_claim : ℕ := 15

-- Formulate the statement to prove: the claim that the total business hours are 15 hours is false.
theorem business_hours_correct : total_business_hours ≠ business_hour_claim := by
  sorry

end NUMINAMATH_GPT_business_hours_correct_l1548_154813


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1548_154884

-- Define the complex number z
def z : ℂ :=
  3 - 2 * Complex.I

-- Lean theorem statement to prove the imaginary part of z is -2
theorem imaginary_part_of_z :
  Complex.im z = -2 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1548_154884


namespace NUMINAMATH_GPT_smallest_N_l1548_154835

theorem smallest_N (l m n N : ℕ) (hl : l > 1) (hm : m > 1) (hn : n > 1) :
  (l - 1) * (m - 1) * (n - 1) = 231 → l * m * n = N → N = 384 :=
sorry

end NUMINAMATH_GPT_smallest_N_l1548_154835


namespace NUMINAMATH_GPT_inequality_example_l1548_154892

theorem inequality_example (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) (h4 : b < 0) : a + b < b + c := 
by sorry

end NUMINAMATH_GPT_inequality_example_l1548_154892


namespace NUMINAMATH_GPT_set_representation_equiv_l1548_154800

open Nat

theorem set_representation_equiv :
  {x : ℕ | (0 < x) ∧ (x - 3 < 2)} = {1, 2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_set_representation_equiv_l1548_154800


namespace NUMINAMATH_GPT_point_not_on_line_l1548_154874

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) : ¬(0 = 2500 * a + c) := by
  sorry

end NUMINAMATH_GPT_point_not_on_line_l1548_154874


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1548_154822

theorem other_root_of_quadratic (p x : ℝ) (h : 7 * x^2 + p * x - 9 = 0) (root1 : x = -3) : 
  x = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1548_154822


namespace NUMINAMATH_GPT_hoot_difference_l1548_154855

def owl_hoot_rate : ℕ := 5
def heard_hoots_per_min : ℕ := 20
def owls_count : ℕ := 3

theorem hoot_difference :
  heard_hoots_per_min - (owls_count * owl_hoot_rate) = 5 := by
  sorry

end NUMINAMATH_GPT_hoot_difference_l1548_154855


namespace NUMINAMATH_GPT_kim_driving_speed_l1548_154827

open Nat
open Real

noncomputable def driving_speed (distance there distance_back time_spent traveling_time total_time: ℝ) : ℝ :=
  (distance + distance_back) / traveling_time

theorem kim_driving_speed:
  ∀ (distance there distance_back time_spent traveling_time total_time: ℝ),
  distance = 30 →
  distance_back = 30 * 1.20 →
  total_time = 2 →
  time_spent = 0.5 →
  traveling_time = total_time - time_spent →
  driving_speed distance there distance_back time_spent traveling_time total_time = 44 :=
by
  intros
  simp only [driving_speed]
  sorry

end NUMINAMATH_GPT_kim_driving_speed_l1548_154827


namespace NUMINAMATH_GPT_not_possible_100_odd_sequence_l1548_154870

def is_square_mod_8 (n : ℤ) : Prop :=
  n % 8 = 0 ∨ n % 8 = 1 ∨ n % 8 = 4

def sum_consecutive_is_square_mod_8 (seq : List ℤ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i + k ≤ seq.length →
  is_square_mod_8 (seq.drop i |>.take k |>.sum)

def valid_odd_sequence (seq : List ℤ) : Prop :=
  seq.length = 100 ∧
  (∀ n ∈ seq, n % 2 = 1) ∧
  sum_consecutive_is_square_mod_8 seq 5 ∧
  sum_consecutive_is_square_mod_8 seq 9

theorem not_possible_100_odd_sequence :
  ¬∃ seq : List ℤ, valid_odd_sequence seq :=
by
  sorry

end NUMINAMATH_GPT_not_possible_100_odd_sequence_l1548_154870


namespace NUMINAMATH_GPT_incorrect_option_D_l1548_154851

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_incorrect_option_D_l1548_154851


namespace NUMINAMATH_GPT_megatek_employees_in_manufacturing_l1548_154824

theorem megatek_employees_in_manufacturing :
  let total_degrees := 360
  let manufacturing_degrees := 108
  (manufacturing_degrees / total_degrees.toFloat) * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_megatek_employees_in_manufacturing_l1548_154824


namespace NUMINAMATH_GPT_fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l1548_154876

def problem1_seq : List ℕ := [102, 101, 100, 99, 98, 97, 96]
def problem2_seq : List ℕ := [190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
def problem3_seq : List ℕ := [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

theorem fill_in_blanks_problem1 :
  ∃ (a b c d : ℕ), [102, a, 100, b, c, 97, d] = [102, 101, 100, 99, 98, 97, 96] :=
by
  exact ⟨101, 99, 98, 96, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem2 :
  ∃ (a b c d e f g : ℕ), [190, a, b, 160, c, d, e, 120, f, g] = [190, 180, 170, 160, 150, 140, 130, 120, 110, 100] :=
by
  exact ⟨180, 170, 150, 140, 130, 110, 100, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem3 :
  ∃ (a b c d e f : ℕ), [5000, a, 6000, b, 7000, c, d, e, f, 9500] = [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500] :=
by
  exact ⟨5500, 6500, 7500, 8000, 8500, 9000, rfl⟩ -- Proof omitted with exact values

end NUMINAMATH_GPT_fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l1548_154876


namespace NUMINAMATH_GPT_solve_for_percentage_l1548_154886

-- Define the constants and variables
variables (P : ℝ)

-- Define the given conditions
def condition : Prop := (P / 100 * 1600 = P / 100 * 650 + 190)

-- Formalize the conjecture: if the conditions hold, then P = 20
theorem solve_for_percentage (h : condition P) : P = 20 :=
sorry

end NUMINAMATH_GPT_solve_for_percentage_l1548_154886


namespace NUMINAMATH_GPT_consecutive_odds_base_eqn_l1548_154833

-- Given conditions
def isOdd (n : ℕ) : Prop := n % 2 = 1

variables {C D : ℕ}

theorem consecutive_odds_base_eqn (C_odd : isOdd C) (D_odd : isOdd D) (consec : D = C + 2)
    (base_eqn : 2 * C^2 + 4 * C + 3 + 6 * D + 5 = 10 * (C + D) + 7) :
    C + D = 16 :=
sorry

end NUMINAMATH_GPT_consecutive_odds_base_eqn_l1548_154833


namespace NUMINAMATH_GPT_investment_c_is_correct_l1548_154878

-- Define the investments of a and b
def investment_a : ℕ := 45000
def investment_b : ℕ := 63000
def profit_c : ℕ := 24000
def total_profit : ℕ := 60000

-- Define the equation to find the investment of c
def proportional_share (x y total : ℕ) : Prop :=
  2 * (x + y + total) = 5 * total

-- The theorem to prove c's investment given the conditions
theorem investment_c_is_correct (c : ℕ) (h_proportional: proportional_share investment_a investment_b c) :
  c = 72000 :=
by
  sorry

end NUMINAMATH_GPT_investment_c_is_correct_l1548_154878


namespace NUMINAMATH_GPT_total_weight_of_load_l1548_154896

def weight_of_crate : ℕ := 4
def weight_of_carton : ℕ := 3
def number_of_crates : ℕ := 12
def number_of_cartons : ℕ := 16

theorem total_weight_of_load :
  number_of_crates * weight_of_crate + number_of_cartons * weight_of_carton = 96 :=
by sorry

end NUMINAMATH_GPT_total_weight_of_load_l1548_154896


namespace NUMINAMATH_GPT_problem_equivalent_statement_l1548_154898

-- Conditions as Lean definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def periodic_property (f : ℝ → ℝ) := ∀ x, x ≥ 0 → f (x + 2) = -f x
def specific_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 8

-- The main theorem
theorem problem_equivalent_statement (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_periodic : periodic_property f) 
  (hf_specific : specific_interval f) :
  f (-2013) + f 2014 = 1 / 3 := 
sorry

end NUMINAMATH_GPT_problem_equivalent_statement_l1548_154898


namespace NUMINAMATH_GPT_treadmill_discount_percentage_l1548_154850

theorem treadmill_discount_percentage
  (p_t : ℝ) -- original price of the treadmill
  (t_p : ℝ) -- total amount paid for treadmill and plates
  (p_plate : ℝ) -- price of each plate
  (n_plate : ℕ) -- number of plates
  (h_t : p_t = 1350)
  (h_tp : t_p = 1045)
  (h_p_plate : p_plate = 50)
  (h_n_plate : n_plate = 2) :
  ((p_t - (t_p - n_plate * p_plate)) / p_t) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_treadmill_discount_percentage_l1548_154850


namespace NUMINAMATH_GPT_total_stamps_l1548_154842

def c : ℕ := 578833
def bw : ℕ := 523776
def total : ℕ := 1102609

theorem total_stamps : c + bw = total := 
by 
  sorry

end NUMINAMATH_GPT_total_stamps_l1548_154842


namespace NUMINAMATH_GPT_intersection_points_l1548_154807

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x^2 + 6 * x + 2

theorem intersection_points :
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} = 
  {(-5/3, 17), (0, 2)} :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l1548_154807


namespace NUMINAMATH_GPT_equilateral_triangle_stack_impossible_l1548_154866

theorem equilateral_triangle_stack_impossible :
  ¬ ∃ n : ℕ, 3 * 55 = 6 * n :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_stack_impossible_l1548_154866


namespace NUMINAMATH_GPT_furthest_distance_l1548_154838

-- Definitions of point distances as given conditions
def PQ : ℝ := 13
def QR : ℝ := 11
def RS : ℝ := 14
def SP : ℝ := 12

-- Statement of the problem in Lean
theorem furthest_distance :
  ∃ (P Q R S : ℝ),
    |P - Q| = PQ ∧
    |Q - R| = QR ∧
    |R - S| = RS ∧
    |S - P| = SP ∧
    ∀ (a b : ℝ), a ≠ b →
      |a - b| ≤ 25 :=
sorry

end NUMINAMATH_GPT_furthest_distance_l1548_154838


namespace NUMINAMATH_GPT_count_squares_within_region_l1548_154811

noncomputable def countSquares : Nat := sorry

theorem count_squares_within_region :
  countSquares = 45 :=
sorry

end NUMINAMATH_GPT_count_squares_within_region_l1548_154811


namespace NUMINAMATH_GPT_first_class_product_probability_l1548_154862

theorem first_class_product_probability
  (defective_rate : ℝ) (first_class_rate_qualified : ℝ)
  (H_def_rate : defective_rate = 0.04)
  (H_first_class_rate_qualified : first_class_rate_qualified = 0.75) :
  (1 - defective_rate) * first_class_rate_qualified = 0.72 :=
by
  sorry

end NUMINAMATH_GPT_first_class_product_probability_l1548_154862


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1548_154859

noncomputable def f (x : ℝ) := 
  5 / (Real.sqrt (x - 9) - 8) - 
  2 / (Real.sqrt (x - 9) - 5) + 
  6 / (Real.sqrt (x - 9) + 5) - 
  9 / (Real.sqrt (x - 9) + 8)

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x ≥ 9 → f x = 0 → 
  x = 19.2917 ∨ x = 8.9167 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1548_154859


namespace NUMINAMATH_GPT_fishing_boat_should_go_out_to_sea_l1548_154839

def good_weather_profit : ℤ := 6000
def bad_weather_loss : ℤ := -8000
def stay_at_port_loss : ℤ := -1000

def prob_good_weather : ℚ := 0.6
def prob_bad_weather : ℚ := 0.4

def expected_profit_going : ℚ :=  prob_good_weather * good_weather_profit + prob_bad_weather * bad_weather_loss
def expected_profit_staying : ℚ := stay_at_port_loss

theorem fishing_boat_should_go_out_to_sea : 
  expected_profit_going > expected_profit_staying :=
  sorry

end NUMINAMATH_GPT_fishing_boat_should_go_out_to_sea_l1548_154839


namespace NUMINAMATH_GPT_find_special_numbers_l1548_154875

theorem find_special_numbers (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) :=
by
  sorry

end NUMINAMATH_GPT_find_special_numbers_l1548_154875


namespace NUMINAMATH_GPT_find_number_of_raccoons_squirrels_opossums_l1548_154812

theorem find_number_of_raccoons_squirrels_opossums
  (R : ℕ)
  (total_animals : ℕ)
  (number_of_squirrels : ℕ := 6 * R)
  (number_of_opossums : ℕ := 2 * R)
  (total : ℕ := R + number_of_squirrels + number_of_opossums) 
  (condition : total_animals = 168)
  (correct_total : total = total_animals) :
  ∃ R : ℕ, R + 6 * R + 2 * R = total_animals :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_raccoons_squirrels_opossums_l1548_154812


namespace NUMINAMATH_GPT_one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l1548_154888

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem one_zero_implies_a_eq_pm2 (a : ℝ) : (∃! x, f a x = 0) → (a = 2 ∨ a = -2) := by
  sorry

theorem zero_in_interval_implies_a_in_open_interval (a : ℝ) : (∃ x, f a x = 0 ∧ 0 < x ∧ x < 1) → 2 < a := by
  sorry

end NUMINAMATH_GPT_one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l1548_154888


namespace NUMINAMATH_GPT_set_equivalence_l1548_154818

-- Define the given set using the condition.
def given_set : Set ℕ := {x | x ∈ {x | 0 < x} ∧ x - 3 < 2}

-- Define the enumerated set.
def enumerated_set : Set ℕ := {1, 2, 3, 4}

-- Statement of the proof problem.
theorem set_equivalence : given_set = enumerated_set :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_set_equivalence_l1548_154818


namespace NUMINAMATH_GPT_high_temp_three_years_same_l1548_154825

theorem high_temp_three_years_same
  (T : ℝ)                               -- The high temperature for the three years with the same temperature
  (temp2017 : ℝ := 79)                   -- The high temperature for 2017
  (temp2016 : ℝ := 71)                   -- The high temperature for 2016
  (average_temp : ℝ := 84)               -- The average high temperature for 5 years
  (num_years : ℕ := 5)                   -- The number of years to consider
  (years_with_same_temp : ℕ := 3)        -- The number of years with the same high temperature
  (total_temp : ℝ := average_temp * num_years) -- The sum of the high temperatures for the 5 years
  (total_known_temp : ℝ := temp2017 + temp2016) -- The known high temperatures for 2016 and 2017
  (total_for_three_years : ℝ := total_temp - total_known_temp) -- Total high temperatures for the three years
  (high_temp_per_year : ℝ := total_for_three_years / years_with_same_temp) -- High temperature per year for three years
  :
  T = 90 :=
sorry

end NUMINAMATH_GPT_high_temp_three_years_same_l1548_154825


namespace NUMINAMATH_GPT_find_smallest_nat_with_remainder_2_l1548_154805

noncomputable def smallest_nat_with_remainder_2 : Nat :=
    let x := 26
    if x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] then x
    else 0

theorem find_smallest_nat_with_remainder_2 :
    ∃ x : Nat, x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] ∧ x = smallest_nat_with_remainder_2 :=
    sorry

end NUMINAMATH_GPT_find_smallest_nat_with_remainder_2_l1548_154805


namespace NUMINAMATH_GPT_vector_subtraction_l1548_154856

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end NUMINAMATH_GPT_vector_subtraction_l1548_154856


namespace NUMINAMATH_GPT_inequality_proof_l1548_154815

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9 * x * y * z) :
    x / Real.sqrt (x^2 + 2 * y * z + 2) + y / Real.sqrt (y^2 + 2 * z * x + 2) + z / Real.sqrt (z^2 + 2 * x * y + 2) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1548_154815


namespace NUMINAMATH_GPT_value_of_f_2_plus_g_3_l1548_154877

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 - 1

theorem value_of_f_2_plus_g_3 : f (2 + g 3) = 26 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_2_plus_g_3_l1548_154877


namespace NUMINAMATH_GPT_find_constant_c_l1548_154880

theorem find_constant_c : ∃ (c : ℝ), (∀ n : ℤ, c * (n:ℝ)^2 ≤ 3600) ∧ (∀ n : ℤ, n ≤ 5) ∧ (c = 144) :=
by
  sorry

end NUMINAMATH_GPT_find_constant_c_l1548_154880


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l1548_154861

theorem geometric_sequence_b_value (a b c : ℝ) (h : 1 * a = a * b ∧ a * b = b * c ∧ b * c = c * 5) : b = Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l1548_154861


namespace NUMINAMATH_GPT_scientific_notation_of_population_l1548_154836

theorem scientific_notation_of_population (population : Real) (h_pop : population = 6.8e6) :
    ∃ a n, (1 ≤ |a| ∧ |a| < 10) ∧ (population = a * 10^n) ∧ (a = 6.8) ∧ (n = 6) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_population_l1548_154836


namespace NUMINAMATH_GPT_area_percentage_change_is_neg_4_percent_l1548_154814

noncomputable def percent_change_area (L W : ℝ) : ℝ :=
  let A_initial := L * W
  let A_new := (1.20 * L) * (0.80 * W)
  ((A_new - A_initial) / A_initial) * 100

theorem area_percentage_change_is_neg_4_percent (L W : ℝ) :
  percent_change_area L W = -4 :=
by
  sorry

end NUMINAMATH_GPT_area_percentage_change_is_neg_4_percent_l1548_154814


namespace NUMINAMATH_GPT_min_x_y_sum_l1548_154849

theorem min_x_y_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/(x+1) + 1/y = 1/2) : x + y ≥ 7 := 
by 
  sorry

end NUMINAMATH_GPT_min_x_y_sum_l1548_154849


namespace NUMINAMATH_GPT_find_c2_given_d4_l1548_154897

theorem find_c2_given_d4 (c d k : ℝ) (h : c^2 * d^4 = k) (hc8 : c = 8) (hd2 : d = 2) (hd4 : d = 4):
  c^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_c2_given_d4_l1548_154897


namespace NUMINAMATH_GPT_cost_equal_at_60_l1548_154821

variable (x : ℝ)

def PlanA_cost (x : ℝ) : ℝ := 0.25 * x + 9
def PlanB_cost (x : ℝ) : ℝ := 0.40 * x

theorem cost_equal_at_60 : PlanA_cost x = PlanB_cost x → x = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_equal_at_60_l1548_154821


namespace NUMINAMATH_GPT_remainder_of_266_div_33_and_8_is_2_l1548_154826

theorem remainder_of_266_div_33_and_8_is_2 :
  (266 % 33 = 2) ∧ (266 % 8 = 2) := by
  sorry

end NUMINAMATH_GPT_remainder_of_266_div_33_and_8_is_2_l1548_154826


namespace NUMINAMATH_GPT_min_value_of_y_l1548_154808

noncomputable def y (x : ℝ) : ℝ :=
  2 * Real.sin (Real.pi / 3 - x) - Real.cos (Real.pi / 6 + x)

theorem min_value_of_y : ∃ x : ℝ, y x = -1 := by
  sorry

end NUMINAMATH_GPT_min_value_of_y_l1548_154808


namespace NUMINAMATH_GPT_gain_percentage_l1548_154834

theorem gain_percentage (MP CP : ℝ) (h1 : 0.90 * MP = 1.17 * CP) :
  (((MP - CP) / CP) * 100) = 30 := 
by
  sorry

end NUMINAMATH_GPT_gain_percentage_l1548_154834


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l1548_154830

/-- If a parabola passes through points A(-2,0) and B(4,0), then the axis of symmetry of the parabola is the line x = 1. -/
theorem axis_of_symmetry_parabola (x : ℝ → ℝ) (hA : x (-2) = 0) (hB : x 4 = 0) : 
  ∃ c : ℝ, c = 1 ∧ ∀ y : ℝ, x y = x (2 * c - y) :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l1548_154830


namespace NUMINAMATH_GPT_directrix_of_parabola_l1548_154869

theorem directrix_of_parabola (p : ℝ) (hp : 2 * p = 4) : 
  (∃ x : ℝ, x = -1) :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1548_154869


namespace NUMINAMATH_GPT_sculpture_and_base_height_l1548_154843

def height_sculpture : ℕ := 2 * 12 + 10
def height_base : ℕ := 8
def total_height : ℕ := 42

theorem sculpture_and_base_height :
  height_sculpture + height_base = total_height :=
by
  -- provide the necessary proof steps here
  sorry

end NUMINAMATH_GPT_sculpture_and_base_height_l1548_154843


namespace NUMINAMATH_GPT_randy_trips_l1548_154891

def trips_per_month
  (initial : ℕ) -- Randy initially had $200 in his piggy bank
  (final : ℕ)   -- Randy had $104 left in his piggy bank after a year
  (spend_per_trip : ℕ) -- Randy spends $2 every time he goes to the store
  (months_in_year : ℕ) -- Number of months in a year, which is 12
  (total_trips_per_year : ℕ) -- Total trips he makes in a year
  (trips_per_month : ℕ) -- Trips to the store every month
  : Prop :=
  initial = 200 ∧ final = 104 ∧ spend_per_trip = 2 ∧ months_in_year = 12 ∧
  total_trips_per_year = (initial - final) / spend_per_trip ∧ 
  trips_per_month = total_trips_per_year / months_in_year ∧
  trips_per_month = 4

theorem randy_trips :
  trips_per_month 200 104 2 12 ((200 - 104) / 2) (48 / 12) :=
by 
  sorry

end NUMINAMATH_GPT_randy_trips_l1548_154891


namespace NUMINAMATH_GPT_oranges_in_bin_l1548_154817

theorem oranges_in_bin (initial_oranges : ℕ) (oranges_thrown_away : ℕ) (oranges_added : ℕ) 
  (h1 : initial_oranges = 50) (h2 : oranges_thrown_away = 40) (h3 : oranges_added = 24) 
  : initial_oranges - oranges_thrown_away + oranges_added = 34 := 
by
  -- Simplification and calculation here
  sorry

end NUMINAMATH_GPT_oranges_in_bin_l1548_154817


namespace NUMINAMATH_GPT_ratio_planes_bisect_volume_l1548_154883

-- Definitions
def n : ℕ := 6
def m : ℕ := 20

-- Statement to prove
theorem ratio_planes_bisect_volume : (n / m : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_ratio_planes_bisect_volume_l1548_154883


namespace NUMINAMATH_GPT_max_odd_integers_l1548_154809

theorem max_odd_integers (a1 a2 a3 a4 a5 a6 a7 : ℕ) (hpos : ∀ i, i ∈ [a1, a2, a3, a4, a5, a6, a7] → i > 0) 
  (hprod : a1 * a2 * a3 * a4 * a5 * a6 * a7 % 2 = 0) : 
  ∃ l : List ℕ, l.length = 6 ∧ (∀ i, i ∈ l → i % 2 = 1) ∧ ∃ e : ℕ, e % 2 = 0 ∧ e ∈ [a1, a2, a3, a4, a5, a6, a7] :=
by
  sorry

end NUMINAMATH_GPT_max_odd_integers_l1548_154809


namespace NUMINAMATH_GPT_fraction_eggs_given_to_Sofia_l1548_154879

variables (m : ℕ) -- Number of eggs Mia has
def Sofia_eggs := 3 * m
def Pablo_eggs := 4 * Sofia_eggs
def Lucas_eggs := 0

theorem fraction_eggs_given_to_Sofia (h1 : Pablo_eggs = 12 * m) :
  (1 : ℚ) / (12 : ℚ) = 1 / 12 := by sorry

end NUMINAMATH_GPT_fraction_eggs_given_to_Sofia_l1548_154879


namespace NUMINAMATH_GPT_integer_solution_l1548_154863

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n > -9) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_l1548_154863


namespace NUMINAMATH_GPT_smallest_integer_form_l1548_154873

theorem smallest_integer_form (m n : ℤ) : ∃ (a : ℤ), a = 2011 * m + 55555 * n ∧ a > 0 → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_form_l1548_154873


namespace NUMINAMATH_GPT_sum_series_equals_half_l1548_154857

theorem sum_series_equals_half :
  ∑' n, 1 / (n * (n+1) * (n+2)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_sum_series_equals_half_l1548_154857


namespace NUMINAMATH_GPT_find_correct_average_of_numbers_l1548_154867

variable (nums : List ℝ)
variable (n : ℕ) (avg_wrong avg_correct : ℝ) (wrong_val correct_val : ℝ)

noncomputable def correct_average (nums : List ℝ) (wrong_val correct_val : ℝ) : ℝ :=
  let correct_sum := nums.sum - wrong_val + correct_val
  correct_sum / nums.length

theorem find_correct_average_of_numbers
  (h₀ : n = 10)
  (h₁ : avg_wrong = 15)
  (h₂ : wrong_val = 26)
  (h₃ : correct_val = 36)
  (h₄ : avg_correct = 16)
  (nums : List ℝ) :
  avg_wrong * n - wrong_val + correct_val = avg_correct * n := 
sorry

end NUMINAMATH_GPT_find_correct_average_of_numbers_l1548_154867


namespace NUMINAMATH_GPT_option_one_cost_option_two_cost_cost_effectiveness_l1548_154858

-- Definition of costs based on conditions
def price_of_suit : ℕ := 500
def price_of_tie : ℕ := 60
def discount_option_one (x : ℕ) : ℕ := 60 * x + 8800
def discount_option_two (x : ℕ) : ℕ := 54 * x + 9000

-- Theorem statements
theorem option_one_cost (x : ℕ) (hx : x > 20) : discount_option_one x = 60 * x + 8800 :=
by sorry

theorem option_two_cost (x : ℕ) (hx : x > 20) : discount_option_two x = 54 * x + 9000 :=
by sorry

theorem cost_effectiveness (x : ℕ) (hx : x = 30) : discount_option_one x < discount_option_two x :=
by sorry

end NUMINAMATH_GPT_option_one_cost_option_two_cost_cost_effectiveness_l1548_154858


namespace NUMINAMATH_GPT_at_least_one_does_not_land_l1548_154801

/-- Proposition stating "A lands within the designated area". -/
def p : Prop := sorry

/-- Proposition stating "B lands within the designated area". -/
def q : Prop := sorry

/-- Negation of proposition p, stating "A does not land within the designated area". -/
def not_p : Prop := ¬p

/-- Negation of proposition q, stating "B does not land within the designated area". -/
def not_q : Prop := ¬q

/-- The proposition "At least one trainee does not land within the designated area" can be expressed as (¬p) ∨ (¬q). -/
theorem at_least_one_does_not_land : (¬p ∨ ¬q) := sorry

end NUMINAMATH_GPT_at_least_one_does_not_land_l1548_154801


namespace NUMINAMATH_GPT_triangle_constructibility_l1548_154871

noncomputable def constructible_triangle (a b w_c : ℝ) : Prop :=
  (2 * a * b) / (a + b) > w_c

theorem triangle_constructibility {a b w_c : ℝ} (h : (a > 0) ∧ (b > 0) ∧ (w_c > 0)) :
  constructible_triangle a b w_c ↔ True :=
by
  sorry

end NUMINAMATH_GPT_triangle_constructibility_l1548_154871


namespace NUMINAMATH_GPT_quadratic_eq_m_neg1_l1548_154828

theorem quadratic_eq_m_neg1 (m : ℝ) (h1 : (m - 3) ≠ 0) (h2 : m^2 - 2*m - 3 = 0) : m = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_eq_m_neg1_l1548_154828


namespace NUMINAMATH_GPT_roots_of_equation_l1548_154802

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l1548_154802


namespace NUMINAMATH_GPT_Sara_has_3194_quarters_in_the_end_l1548_154868

theorem Sara_has_3194_quarters_in_the_end
  (initial_quarters : ℕ)
  (borrowed_quarters : ℕ)
  (initial_quarters_eq : initial_quarters = 4937)
  (borrowed_quarters_eq : borrowed_quarters = 1743) :
  initial_quarters - borrowed_quarters = 3194 := by
  sorry

end NUMINAMATH_GPT_Sara_has_3194_quarters_in_the_end_l1548_154868


namespace NUMINAMATH_GPT_radius_of_cone_l1548_154899

theorem radius_of_cone (A : ℝ) (g : ℝ) (R : ℝ) (hA : A = 15 * Real.pi) (hg : g = 5) : R = 3 :=
sorry

end NUMINAMATH_GPT_radius_of_cone_l1548_154899


namespace NUMINAMATH_GPT_remaining_budget_l1548_154844

def charge_cost : ℝ := 3.5
def num_charges : ℝ := 4
def total_budget : ℝ := 20

theorem remaining_budget : total_budget - (num_charges * charge_cost) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_budget_l1548_154844


namespace NUMINAMATH_GPT_counterexample_exists_l1548_154847

theorem counterexample_exists : ∃ n : ℕ, n ≥ 2 ∧ ¬ ∃ k : ℕ, 2 ^ 2 ^ n % (2 ^ n - 1) = 4 ^ k := 
by
  sorry

end NUMINAMATH_GPT_counterexample_exists_l1548_154847


namespace NUMINAMATH_GPT_solve_for_x_l1548_154823

theorem solve_for_x (x : ℝ) (h : 2 * x - 5 = 15) : x = 10 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1548_154823


namespace NUMINAMATH_GPT_total_snowfall_l1548_154865

theorem total_snowfall (morning_snowfall : ℝ) (afternoon_snowfall : ℝ) (h_morning : morning_snowfall = 0.125) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.625 :=
by 
  sorry

end NUMINAMATH_GPT_total_snowfall_l1548_154865


namespace NUMINAMATH_GPT_sum_of_roots_l1548_154881

theorem sum_of_roots (m n : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : ∀ x : ℝ, x^2 + m * x + n = 0 → (x = m ∨ x = n)) :
  m + n = -1 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1548_154881


namespace NUMINAMATH_GPT_grade_on_second_test_l1548_154887

variable (first_test_grade second_test_average : ℕ)
#check first_test_grade
#check second_test_average

theorem grade_on_second_test :
  first_test_grade = 78 →
  second_test_average = 81 →
  (first_test_grade + (second_test_average * 2 - first_test_grade)) / 2 = second_test_average →
  second_test_grade = 84 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_grade_on_second_test_l1548_154887


namespace NUMINAMATH_GPT_solve_inequality_l1548_154816

theorem solve_inequality (a : ℝ) (x : ℝ) :
  (a = 0 → x > 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (0 < a → a < 2 → 1 < x → x < 2 / a → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a = 2 → False → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a > 2 → 2 / a < x → x < 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a < 0 → ((x < 2 / a ∨ x > 1) → (ax^2 - (a + 2) * x + 2 < 0))) := sorry

end NUMINAMATH_GPT_solve_inequality_l1548_154816


namespace NUMINAMATH_GPT_max_discount_rate_l1548_154841

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end NUMINAMATH_GPT_max_discount_rate_l1548_154841


namespace NUMINAMATH_GPT_gemstones_needed_for_sets_l1548_154845

-- Define the number of magnets per earring
def magnets_per_earring : ℕ := 2

-- Define the number of buttons per earring as half the number of magnets
def buttons_per_earring (magnets : ℕ) : ℕ := magnets / 2

-- Define the number of gemstones per earring as three times the number of buttons
def gemstones_per_earring (buttons : ℕ) : ℕ := 3 * buttons

-- Define the number of earrings per set
def earrings_per_set : ℕ := 2

-- Define the number of sets
def sets : ℕ := 4

-- Prove that Rebecca needs 24 gemstones for 4 sets of earrings given the conditions
theorem gemstones_needed_for_sets :
  gemstones_per_earring (buttons_per_earring magnets_per_earring) * earrings_per_set * sets = 24 :=
by
  sorry

end NUMINAMATH_GPT_gemstones_needed_for_sets_l1548_154845


namespace NUMINAMATH_GPT_non_juniors_play_instrument_l1548_154831

theorem non_juniors_play_instrument (total_students juniors non_juniors play_instrument_juniors play_instrument_non_juniors total_do_not_play : ℝ) :
  total_students = 600 →
  play_instrument_juniors = 0.3 * juniors →
  play_instrument_non_juniors = 0.65 * non_juniors →
  total_do_not_play = 0.4 * total_students →
  0.7 * juniors + 0.35 * non_juniors = total_do_not_play →
  juniors + non_juniors = total_students →
  non_juniors * 0.65 = 334 :=
by
  sorry

end NUMINAMATH_GPT_non_juniors_play_instrument_l1548_154831


namespace NUMINAMATH_GPT_river_length_l1548_154853

theorem river_length :
  let still_water_speed := 10 -- Karen's paddling speed on still water in miles per hour
  let current_speed      := 4  -- River's current speed in miles per hour
  let time               := 2  -- Time it takes Karen to paddle up the river in hours
  let effective_speed    := still_water_speed - current_speed -- Karen's effective speed against the current
  effective_speed * time = 12 -- Length of the river in miles
:= by
  sorry

end NUMINAMATH_GPT_river_length_l1548_154853


namespace NUMINAMATH_GPT_circle_radius_l1548_154893

theorem circle_radius (D : ℝ) (h : D = 14) : (D / 2) = 7 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1548_154893


namespace NUMINAMATH_GPT_parallelogram_proof_l1548_154848

noncomputable def parallelogram_ratio (AP AB AQ AD AC AT : ℝ) (hP : AP / AB = 61 / 2022) (hQ : AQ / AD = 61 / 2065) (h_intersect : true) : ℕ :=
if h : AC / AT = 4087 / 61 then 67 else 0

theorem parallelogram_proof :
  ∀ (ABCD : Type) (P : Type) (Q : Type) (T : Type) 
     (AP AB AQ AD AC AT : ℝ) 
     (hP : AP / AB = 61 / 2022) 
     (hQ : AQ / AD = 61 / 2065)
     (h_intersect : true),
  parallelogram_ratio AP AB AQ AD AC AT hP hQ h_intersect = 67 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_proof_l1548_154848


namespace NUMINAMATH_GPT_find_initial_mean_l1548_154837

/-- 
  The mean of 50 observations is M.
  One observation was wrongly taken as 23 but should have been 30.
  The corrected mean is 36.5.
  Prove that the initial mean M was 36.36.
-/
theorem find_initial_mean (M : ℝ) (h : 50 * 36.36 + 7 = 50 * 36.5) : 
  (500 * 36.36 - 7) = 1818 :=
sorry

end NUMINAMATH_GPT_find_initial_mean_l1548_154837


namespace NUMINAMATH_GPT_true_false_question_count_l1548_154852

theorem true_false_question_count (n : ℕ) (h : (1 / 3) * (1 / 2)^n = 1 / 12) : n = 2 := by
  sorry

end NUMINAMATH_GPT_true_false_question_count_l1548_154852


namespace NUMINAMATH_GPT_product_of_roots_l1548_154864

theorem product_of_roots (p q r : ℝ)
  (h1 : ∀ x : ℝ, (3 * x^3 - 9 * x^2 + 5 * x - 15 = 0) → (x = p ∨ x = q ∨ x = r)) :
  p * q * r = 5 := by
  sorry

end NUMINAMATH_GPT_product_of_roots_l1548_154864
