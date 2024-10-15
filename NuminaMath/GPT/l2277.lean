import Mathlib

namespace NUMINAMATH_GPT_inheritance_amount_l2277_227775

theorem inheritance_amount (x : ℝ) (total_taxes_paid : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
  (federal_tax_paid : ℝ) (state_tax_base : ℝ) (state_tax_paid : ℝ) 
  (federal_tax_eq : federal_tax_paid = federal_tax_rate * x)
  (state_tax_base_eq : state_tax_base = x - federal_tax_paid)
  (state_tax_eq : state_tax_paid = state_tax_rate * state_tax_base)
  (total_taxes_eq : total_taxes_paid = federal_tax_paid + state_tax_paid) 
  (total_taxes_val : total_taxes_paid = 18000)
  (federal_tax_rate_val : federal_tax_rate = 0.25)
  (state_tax_rate_val : state_tax_rate = 0.15)
  : x = 50000 :=
sorry

end NUMINAMATH_GPT_inheritance_amount_l2277_227775


namespace NUMINAMATH_GPT_pine_cone_weight_on_roof_l2277_227757

theorem pine_cone_weight_on_roof
  (num_trees : ℕ) (cones_per_tree : ℕ) (percentage_on_roof : ℝ) (weight_per_cone : ℕ)
  (H1 : num_trees = 8)
  (H2 : cones_per_tree = 200)
  (H3 : percentage_on_roof = 0.30)
  (H4 : weight_per_cone = 4) :
  num_trees * cones_per_tree * percentage_on_roof * weight_per_cone = 1920 := by
  sorry

end NUMINAMATH_GPT_pine_cone_weight_on_roof_l2277_227757


namespace NUMINAMATH_GPT_present_age_of_son_l2277_227725

variable (S M : ℝ)

-- Conditions
def condition1 : Prop := M = S + 35
def condition2 : Prop := M + 5 = 3 * (S + 5)

-- Proof Problem
theorem present_age_of_son
  (h1 : condition1 S M)
  (h2 : condition2 S M) :
  S = 12.5 :=
sorry

end NUMINAMATH_GPT_present_age_of_son_l2277_227725


namespace NUMINAMATH_GPT_area_when_other_side_shortened_l2277_227763

def original_width := 5
def original_length := 8
def target_area := 24
def shortened_amount := 2

theorem area_when_other_side_shortened :
  (original_width - shortened_amount) * original_length = target_area →
  original_width * (original_length - shortened_amount) = 30 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_area_when_other_side_shortened_l2277_227763


namespace NUMINAMATH_GPT_tea_sales_revenue_l2277_227717

theorem tea_sales_revenue (x : ℝ) (price_last_year price_this_year : ℝ) (yield_last_year yield_this_year : ℝ) (revenue_last_year revenue_this_year : ℝ) :
  price_this_year = 10 * price_last_year →
  yield_this_year = 198.6 →
  yield_last_year = 198.6 + 87.4 →
  revenue_this_year = 198.6 * price_this_year →
  revenue_last_year = yield_last_year * price_last_year →
  revenue_this_year = revenue_last_year + 8500 →
  revenue_this_year = 9930 := 
by
  sorry

end NUMINAMATH_GPT_tea_sales_revenue_l2277_227717


namespace NUMINAMATH_GPT_cuboid_edge_length_l2277_227753

theorem cuboid_edge_length
  (x : ℝ)
  (h_surface_area : 2 * (4 * x + 24 + 6 * x) = 148) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_edge_length_l2277_227753


namespace NUMINAMATH_GPT_initial_amount_of_money_l2277_227759

-- Define the conditions
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20
def num_friends : ℕ := 2
def amount_left : ℝ := 114.85

-- Define the calculated amount given to friends
def total_given_to_friends : ℝ := given_to_each_friend * num_friends

-- State the theorem to prove the initial amount of money
theorem initial_amount_of_money :
  spent_on_sweets + total_given_to_friends + amount_left = 200.50 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_initial_amount_of_money_l2277_227759


namespace NUMINAMATH_GPT_max_constant_term_l2277_227743

theorem max_constant_term (c : ℝ) : 
  (∀ x : ℝ, (x^2 - 6 * x + c = 0 → (x^2 - 6 * x + c ≥ 0))) → c ≤ 9 :=
by sorry

end NUMINAMATH_GPT_max_constant_term_l2277_227743


namespace NUMINAMATH_GPT_nancy_total_spending_l2277_227772

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end NUMINAMATH_GPT_nancy_total_spending_l2277_227772


namespace NUMINAMATH_GPT_difference_of_squares_multiple_of_20_l2277_227774

theorem difference_of_squares_multiple_of_20 (a b : ℕ) (h1 : a > b) (h2 : a + b = 10) (hb : b = 10 - a) : 
  ∃ k : ℕ, (9 * a + 10)^2 - (100 - 9 * a)^2 = 20 * k :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_multiple_of_20_l2277_227774


namespace NUMINAMATH_GPT_possible_value_of_n_l2277_227731

open Nat

def coefficient_is_rational (n r : ℕ) : Prop :=
  (n - r) % 2 = 0 ∧ r % 3 = 0

theorem possible_value_of_n :
  ∃ n : ℕ, n > 0 ∧ (∀ r : ℕ, r ≤ n → coefficient_is_rational n r) ↔ n = 9 :=
sorry

end NUMINAMATH_GPT_possible_value_of_n_l2277_227731


namespace NUMINAMATH_GPT_time_to_fill_tank_with_two_pipes_simultaneously_l2277_227783

def PipeA : ℝ := 30
def PipeB : ℝ := 45

theorem time_to_fill_tank_with_two_pipes_simultaneously :
  let A := 1 / PipeA
  let B := 1 / PipeB
  let combined_rate := A + B
  let time_to_fill_tank := 1 / combined_rate
  time_to_fill_tank = 18 := 
by
  sorry

end NUMINAMATH_GPT_time_to_fill_tank_with_two_pipes_simultaneously_l2277_227783


namespace NUMINAMATH_GPT_find_n_l2277_227765

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 10) : n = 2100 :=
by
sorry

end NUMINAMATH_GPT_find_n_l2277_227765


namespace NUMINAMATH_GPT_highest_score_runs_l2277_227778

theorem highest_score_runs 
  (avg : ℕ) (innings : ℕ) (total_runs : ℕ) (H L : ℕ)
  (diff_HL : ℕ) (excl_avg : ℕ) (excl_innings : ℕ) (excl_total_runs : ℕ) :
  avg = 60 → innings = 46 → total_runs = avg * innings →
  diff_HL = 180 → excl_avg = 58 → excl_innings = 44 → 
  excl_total_runs = excl_avg * excl_innings →
  H - L = diff_HL →
  total_runs = excl_total_runs + H + L →
  H = 194 :=
by
  intros h_avg h_innings h_total_runs h_diff_HL h_excl_avg h_excl_innings h_excl_total_runs h_H_minus_L h_total_eq
  sorry

end NUMINAMATH_GPT_highest_score_runs_l2277_227778


namespace NUMINAMATH_GPT_total_tickets_needed_l2277_227726

-- Definitions representing the conditions
def rides_go_karts : ℕ := 1
def cost_per_go_kart_ride : ℕ := 4
def rides_bumper_cars : ℕ := 4
def cost_per_bumper_car_ride : ℕ := 5

-- Calculate the total tickets needed
def total_tickets : ℕ := rides_go_karts * cost_per_go_kart_ride + rides_bumper_cars * cost_per_bumper_car_ride

-- The theorem stating the main proof problem
theorem total_tickets_needed : total_tickets = 24 := by
  -- Proof steps should go here, but we use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_tickets_needed_l2277_227726


namespace NUMINAMATH_GPT_carson_pumps_needed_l2277_227745

theorem carson_pumps_needed 
  (full_tire_capacity : ℕ) (flat_tires_count : ℕ) 
  (full_percentage_tire_1 : ℚ) (full_percentage_tire_2 : ℚ)
  (air_per_pump : ℕ) : 
  flat_tires_count = 2 →
  full_tire_capacity = 500 →
  full_percentage_tire_1 = 0.40 →
  full_percentage_tire_2 = 0.70 →
  air_per_pump = 50 →
  let needed_air_flat_tires := flat_tires_count * full_tire_capacity
  let needed_air_tire_1 := (1 - full_percentage_tire_1) * full_tire_capacity
  let needed_air_tire_2 := (1 - full_percentage_tire_2) * full_tire_capacity
  let total_needed_air := needed_air_flat_tires + needed_air_tire_1 + needed_air_tire_2
  let pumps_needed := total_needed_air / air_per_pump
  pumps_needed = 29 := 
by
  intros
  sorry

end NUMINAMATH_GPT_carson_pumps_needed_l2277_227745


namespace NUMINAMATH_GPT_set_cannot_be_divided_l2277_227735

theorem set_cannot_be_divided
  (p : ℕ) (prime_p : Nat.Prime p) (p_eq_3_mod_4 : p % 4 = 3)
  (S : Finset ℕ) (hS : S.card = p - 1) :
  ¬∃ A B : Finset ℕ, A ∪ B = S ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
by {
  sorry
}

end NUMINAMATH_GPT_set_cannot_be_divided_l2277_227735


namespace NUMINAMATH_GPT_problem_statement_l2277_227747

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Nontrivial α]

def is_monotone_increasing (f : α → α) (s : Set α) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

theorem problem_statement (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : ∀ x, f (-x) = -f (x + 4))
  (h2 : is_monotone_increasing f {x | x > 2})
  (hx1 : x1 < 2) (hx2 : 2 < x2) (h_sum : x1 + x2 < 4) :
  f (x1) + f (x2) < 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2277_227747


namespace NUMINAMATH_GPT_problem_l2277_227727

theorem problem
    (a b c d : ℕ)
    (h1 : a = b + 7)
    (h2 : b = c + 15)
    (h3 : c = d + 25)
    (h4 : d = 90) :
  a = 137 := by
  sorry

end NUMINAMATH_GPT_problem_l2277_227727


namespace NUMINAMATH_GPT_probability_two_win_one_lose_l2277_227793

noncomputable def p_A : ℚ := 1 / 5
noncomputable def p_B : ℚ := 3 / 8
noncomputable def p_C : ℚ := 2 / 7

noncomputable def P_two_win_one_lose : ℚ :=
  p_A * p_B * (1 - p_C) +
  p_A * p_C * (1 - p_B) +
  p_B * p_C * (1 - p_A)

theorem probability_two_win_one_lose :
  P_two_win_one_lose = 49 / 280 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_win_one_lose_l2277_227793


namespace NUMINAMATH_GPT_question_1_question_2_l2277_227722

variable (m x : ℝ)
def f (x : ℝ) := |x + m|

theorem question_1 (h : f 1 + f (-2) ≥ 5) : 
  m ≤ -2 ∨ m ≥ 3 := sorry

theorem question_2 (hx : x ≠ 0) : 
  f (1 / x) + f (-x) ≥ 2 := sorry

end NUMINAMATH_GPT_question_1_question_2_l2277_227722


namespace NUMINAMATH_GPT_find_m_value_l2277_227771

variable (m : ℝ)
noncomputable def a : ℝ × ℝ := (2 * Real.sqrt 2, 2)
noncomputable def b : ℝ × ℝ := (0, 2)
noncomputable def c (m : ℝ) : ℝ × ℝ := (m, Real.sqrt 2)

theorem find_m_value (h : (a.1 + 2 * b.1) * (m) + (a.2 + 2 * b.2) * (Real.sqrt 2) = 0) : m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l2277_227771


namespace NUMINAMATH_GPT_unit_square_divisible_l2277_227719

theorem unit_square_divisible (n : ℕ) (h: n ≥ 6) : ∃ squares : ℕ, squares = n :=
by
  sorry

end NUMINAMATH_GPT_unit_square_divisible_l2277_227719


namespace NUMINAMATH_GPT_t_f_3_equals_sqrt_44_l2277_227769

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 4)
noncomputable def f (x : ℝ) : ℝ := 6 + t x

theorem t_f_3_equals_sqrt_44 : t (f 3) = Real.sqrt 44 := by
  sorry

end NUMINAMATH_GPT_t_f_3_equals_sqrt_44_l2277_227769


namespace NUMINAMATH_GPT_negate_prop_l2277_227797

theorem negate_prop (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) :
  ¬ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) ↔ ∃ x_0 : ℝ, 0 ≤ x_0 ∧ x_0 ≤ 2 * Real.pi ∧ |Real.sin x_0| > 1 :=
by sorry

end NUMINAMATH_GPT_negate_prop_l2277_227797


namespace NUMINAMATH_GPT_expenditure_recording_l2277_227780

theorem expenditure_recording (income expense : ℤ) (h1 : income = 100) (h2 : expense = -100)
  (h3 : income = -expense) : expense = -100 :=
by
  sorry

end NUMINAMATH_GPT_expenditure_recording_l2277_227780


namespace NUMINAMATH_GPT_stack_height_difference_l2277_227730

theorem stack_height_difference :
  ∃ S : ℕ,
    (7 + S + (S - 6) + (S + 4) + 2 * S = 55) ∧ (S - 7 = 3) := 
by 
  sorry

end NUMINAMATH_GPT_stack_height_difference_l2277_227730


namespace NUMINAMATH_GPT_notebook_and_pencil_cost_l2277_227782

theorem notebook_and_pencil_cost :
  ∃ (x y : ℝ), 6 * x + 4 * y = 9.2 ∧ 3 * x + y = 3.8 ∧ x + y = 1.8 :=
by
  sorry

end NUMINAMATH_GPT_notebook_and_pencil_cost_l2277_227782


namespace NUMINAMATH_GPT_prime_of_the_form_4x4_plus_1_l2277_227733

theorem prime_of_the_form_4x4_plus_1 (x : ℤ) (p : ℤ) (h : 4 * x ^ 4 + 1 = p) (hp : Prime p) : p = 5 :=
sorry

end NUMINAMATH_GPT_prime_of_the_form_4x4_plus_1_l2277_227733


namespace NUMINAMATH_GPT_option_D_functions_same_l2277_227787

theorem option_D_functions_same (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by 
  sorry

end NUMINAMATH_GPT_option_D_functions_same_l2277_227787


namespace NUMINAMATH_GPT_largest_hole_leakage_rate_l2277_227792

theorem largest_hole_leakage_rate (L : ℝ) (h1 : 600 = (L + L / 2 + L / 6) * 120) : 
  L = 3 :=
sorry

end NUMINAMATH_GPT_largest_hole_leakage_rate_l2277_227792


namespace NUMINAMATH_GPT_range_of_a_l2277_227755

theorem range_of_a (x y z a : ℝ) 
    (h1 : x > 0) 
    (h2 : y > 0) 
    (h3 : z > 0) 
    (h4 : x + y + z = 1) 
    (h5 : a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) : 
    0 < a ∧ a ≤ 7 / 27 := 
  sorry

end NUMINAMATH_GPT_range_of_a_l2277_227755


namespace NUMINAMATH_GPT_XT_value_l2277_227739

noncomputable def AB := 15
noncomputable def BC := 20
noncomputable def height_P := 30
noncomputable def volume_ratio := 9

theorem XT_value 
  (AB BC height_P : ℕ)
  (volume_ratio : ℕ)
  (h1 : AB = 15)
  (h2 : BC = 20)
  (h3 : height_P = 30)
  (h4 : volume_ratio = 9) : 
  ∃ (m n : ℕ), m + n = 97 ∧ m.gcd n = 1 :=
by sorry

end NUMINAMATH_GPT_XT_value_l2277_227739


namespace NUMINAMATH_GPT_arithmetic_sequence_odd_function_always_positive_l2277_227750

theorem arithmetic_sequence_odd_function_always_positive
    (f : ℝ → ℝ) (a : ℕ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_monotone_geq_0 : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
    (h_arith_seq : ∀ n, a (n + 1) = a n + (a 2 - a 1))
    (h_a3_neg : a 3 < 0) :
    f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 := by
    sorry

end NUMINAMATH_GPT_arithmetic_sequence_odd_function_always_positive_l2277_227750


namespace NUMINAMATH_GPT_fewest_coach_handshakes_l2277_227777

theorem fewest_coach_handshakes (n m1 m2 : ℕ) 
  (handshakes_total : (n * (n - 1)) / 2 + m1 + m2 = 465) 
  (m1_m2_eq_n : m1 + m2 = n) : 
  n * (n - 1) / 2 = 465 → m1 + m2 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_fewest_coach_handshakes_l2277_227777


namespace NUMINAMATH_GPT_hamburgers_served_l2277_227767

-- Definitions for the conditions
def hamburgers_made : ℕ := 9
def hamburgers_left_over : ℕ := 6

-- The main statement to prove
theorem hamburgers_served : hamburgers_made - hamburgers_left_over = 3 := by
  sorry

end NUMINAMATH_GPT_hamburgers_served_l2277_227767


namespace NUMINAMATH_GPT_good_numbers_identification_l2277_227720

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), 
    (∀ k : Fin n, ∃ m : ℕ, k.val + a k = m * m)

theorem good_numbers_identification : 
  { n : ℕ | ¬is_good_number n } = {1, 2, 4, 6, 7, 9, 11} :=
  sorry

end NUMINAMATH_GPT_good_numbers_identification_l2277_227720


namespace NUMINAMATH_GPT_gold_problem_proof_l2277_227776

noncomputable def solve_gold_problem : Prop :=
  ∃ (a : ℕ → ℝ), 
  (a 1) + (a 2) + (a 3) = 4 ∧ 
  (a 8) + (a 9) + (a 10) = 3 ∧
  (a 5) + (a 6) = 7 / 3

theorem gold_problem_proof : solve_gold_problem := 
  sorry

end NUMINAMATH_GPT_gold_problem_proof_l2277_227776


namespace NUMINAMATH_GPT_sphere_views_identical_l2277_227707

-- Define the geometric shape as a type
inductive GeometricShape
| sphere
| cube
| other (name : String)

-- Define a function to get the view of a sphere
def view (s : GeometricShape) (direction : String) : String :=
  match s with
  | GeometricShape.sphere => "circle"
  | GeometricShape.cube => "square"
  | GeometricShape.other _ => "unknown"

-- The theorem to prove that a sphere has identical front, top, and side views
theorem sphere_views_identical :
  ∀ (direction1 direction2 : String), view GeometricShape.sphere direction1 = view GeometricShape.sphere direction2 :=
by
  intros direction1 direction2
  sorry

end NUMINAMATH_GPT_sphere_views_identical_l2277_227707


namespace NUMINAMATH_GPT_trajectory_point_M_l2277_227704

theorem trajectory_point_M (x y : ℝ) : 
  (∃ (m n : ℝ), x^2 + y^2 = 9 ∧ (m = x) ∧ (n = 3 * y)) → 
  (x^2 / 9 + y^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_point_M_l2277_227704


namespace NUMINAMATH_GPT_number_of_hens_l2277_227714

variables (H C : ℕ)

def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 144

theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 24 :=
sorry

end NUMINAMATH_GPT_number_of_hens_l2277_227714


namespace NUMINAMATH_GPT_find_divisor_l2277_227766

theorem find_divisor (d : ℕ) : (55 / d) + 10 = 21 → d = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_divisor_l2277_227766


namespace NUMINAMATH_GPT_price_per_kg_of_fruits_l2277_227712

theorem price_per_kg_of_fruits (mangoes apples oranges : ℕ) (total_amount : ℕ)
  (h1 : mangoes = 400)
  (h2 : apples = 2 * mangoes)
  (h3 : oranges = mangoes + 200)
  (h4 : total_amount = 90000) :
  (total_amount / (mangoes + apples + oranges) = 50) :=
by
  sorry

end NUMINAMATH_GPT_price_per_kg_of_fruits_l2277_227712


namespace NUMINAMATH_GPT_day_crew_fraction_l2277_227798

-- Definitions of number of boxes per worker for day crew, and workers for day crew
variables (D : ℕ) (W : ℕ)

-- Definitions of night crew loading rate and worker ratio based on given conditions
def night_boxes_per_worker := (3 / 4 : ℚ) * D
def night_workers := (2 / 3 : ℚ) * W

-- Definition of total boxes loaded by each crew
def day_crew_total := D * W
def night_crew_total := night_boxes_per_worker D * night_workers W

-- The proof problem shows fraction loaded by day crew equals 2/3
theorem day_crew_fraction : (day_crew_total D W) / (day_crew_total D W + night_crew_total D W) = (2 / 3 : ℚ) := by
  sorry

end NUMINAMATH_GPT_day_crew_fraction_l2277_227798


namespace NUMINAMATH_GPT_lcm_15_48_eq_240_l2277_227705

def is_least_common_multiple (n a b : Nat) : Prop :=
  n % a = 0 ∧ n % b = 0 ∧ ∀ m, (m % a = 0 ∧ m % b = 0) → n ≤ m

theorem lcm_15_48_eq_240 : is_least_common_multiple 240 15 48 :=
by
  sorry

end NUMINAMATH_GPT_lcm_15_48_eq_240_l2277_227705


namespace NUMINAMATH_GPT_difference_of_two_numbers_l2277_227715

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 15) (h2 : x^2 - y^2 = 150) : x - y = 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l2277_227715


namespace NUMINAMATH_GPT_age_problem_l2277_227710

theorem age_problem 
  (A : ℕ) 
  (x : ℕ) 
  (h1 : 3 * (A + x) - 3 * (A - 3) = A) 
  (h2 : A = 18) : 
  x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_age_problem_l2277_227710


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2277_227784

theorem solution_set_of_inequality :
  {x : ℝ | (x - 3) / x ≥ 0} = {x : ℝ | x < 0 ∨ x ≥ 3} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2277_227784


namespace NUMINAMATH_GPT_largest_possible_green_cards_l2277_227751

-- Definitions of conditions
variables (g y t : ℕ)

-- Defining the total number of cards t
def total_cards := g + y

-- Condition on maximum number of cards
def max_total_cards := total_cards g y ≤ 2209

-- Probability condition for drawing 3 same-color cards
def probability_condition := 
  g * (g - 1) * (g - 2) + y * (y - 1) * (y - 2) 
  = (1 : ℚ) / 3 * t * (t - 1) * (t - 2)

-- Proving the largest possible number of green cards
theorem largest_possible_green_cards
  (h1 : total_cards g y = t)
  (h2 : max_total_cards g y)
  (h3 : probability_condition g y t) :
  g ≤ 1092 :=
sorry

end NUMINAMATH_GPT_largest_possible_green_cards_l2277_227751


namespace NUMINAMATH_GPT_number_of_games_l2277_227789

-- Definitions based on the conditions
def initial_money : ℕ := 104
def cost_of_blades : ℕ := 41
def cost_per_game : ℕ := 9

-- Lean 4 statement asserting the number of games Will can buy is 7
theorem number_of_games : (initial_money - cost_of_blades) / cost_per_game = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_games_l2277_227789


namespace NUMINAMATH_GPT_ratio_S7_S3_l2277_227799

variable {a_n : ℕ → ℕ} -- Arithmetic sequence {a_n}
variable (S_n : ℕ → ℕ) -- Sum of the first n terms of the arithmetic sequence

-- Conditions
def ratio_a2_a4 (a_2 a_4 : ℕ) : Prop := a_2 = 7 * (a_4 / 6)
def sum_formula (n a_1 d : ℕ) : ℕ := n * (2 * a_1 + (n - 1) * d) / 2

-- Proof goal
theorem ratio_S7_S3 (a_1 d : ℕ) (h : ratio_a2_a4 (a_1 + d) (a_1 + 3 * d)): 
  (S_n 7 = sum_formula 7 a_1 d) ∧ (S_n 3 = sum_formula 3 a_1 d) →
  (S_n 7 / S_n 3 = 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_S7_S3_l2277_227799


namespace NUMINAMATH_GPT_unpaintedRegionArea_l2277_227721

def boardWidth1 : ℝ := 5
def boardWidth2 : ℝ := 7
def angle : ℝ := 45

theorem unpaintedRegionArea
  (bw1 bw2 angle : ℝ)
  (h1 : bw1 = boardWidth1)
  (h2 : bw2 = boardWidth2)
  (h3 : angle = 45) :
  let base := bw2 * Real.sqrt 2
  let height := bw1
  let area := base * height
  area = 35 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_unpaintedRegionArea_l2277_227721


namespace NUMINAMATH_GPT_circleII_area_l2277_227779

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

theorem circleII_area (r₁ : ℝ) (h₁ : area_of_circle r₁ = 9) (h₂ : r₂ = 3 * 2 * r₁) : 
  area_of_circle r₂ = 324 :=
by
  sorry

end NUMINAMATH_GPT_circleII_area_l2277_227779


namespace NUMINAMATH_GPT_dilation_image_l2277_227724

theorem dilation_image 
  (z z₀ : ℂ) (k : ℝ) 
  (hz : z = -2 + i) 
  (hz₀ : z₀ = 1 - 3 * I) 
  (hk : k = 3) : 
  (k * (z - z₀) + z₀) = (-8 + 9 * I) := 
by 
  rw [hz, hz₀, hk]
  -- Sorry means here we didn't write the complete proof, we assume it is correct.
  sorry

end NUMINAMATH_GPT_dilation_image_l2277_227724


namespace NUMINAMATH_GPT_triangle_angle_sum_l2277_227737

theorem triangle_angle_sum (x : ℝ) :
    let angle1 : ℝ := 40
    let angle2 : ℝ := 4 * x
    let angle3 : ℝ := 3 * x
    angle1 + angle2 + angle3 = 180 -> x = 20 := 
sorry

end NUMINAMATH_GPT_triangle_angle_sum_l2277_227737


namespace NUMINAMATH_GPT_num_students_basketball_l2277_227749

-- Definitions for conditions
def num_students_cricket : ℕ := 8
def num_students_both : ℕ := 5
def num_students_either : ℕ := 10

-- statement to be proven
theorem num_students_basketball : ∃ B : ℕ, B = 7 ∧ (num_students_either = B + num_students_cricket - num_students_both) := sorry

end NUMINAMATH_GPT_num_students_basketball_l2277_227749


namespace NUMINAMATH_GPT_find_expression_l2277_227713

theorem find_expression (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 3 * x + 2) : 
  ∀ x : ℤ, f x = 3 * x - 1 :=
sorry

end NUMINAMATH_GPT_find_expression_l2277_227713


namespace NUMINAMATH_GPT_f_is_odd_function_f_is_increasing_f_max_min_in_interval_l2277_227711

variable {f : ℝ → ℝ}

-- The conditions:
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom positive_for_positive : ∀ x : ℝ, x > 0 → f x > 0
axiom f_one_is_two : f 1 = 2

-- The proof tasks:
theorem f_is_odd_function : ∀ x : ℝ, f (-x) = -f x := 
sorry

theorem f_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := 
sorry

theorem f_max_min_in_interval : 
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≤ 6) ∧ (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -6) :=
sorry

end NUMINAMATH_GPT_f_is_odd_function_f_is_increasing_f_max_min_in_interval_l2277_227711


namespace NUMINAMATH_GPT_geometric_series_seventh_term_l2277_227795

theorem geometric_series_seventh_term (a₁ a₁₀ : ℝ) (n : ℝ) (r : ℝ) :
  a₁ = 4 →
  a₁₀ = 93312 →
  n = 10 →
  a₁₀ = a₁ * r^(n-1) →
  (∃ (r : ℝ), r = 6) →
  4 * 6^(7-1) = 186624 := by
  intros a1_eq a10_eq n_eq an_eq exists_r
  sorry

end NUMINAMATH_GPT_geometric_series_seventh_term_l2277_227795


namespace NUMINAMATH_GPT_triangle_problems_l2277_227794

open Real

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

def triangle_sides_and_angles (a b c : ℝ) (A B C : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

def perpendicular (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

noncomputable def area_of_triangle (a b c : ℝ) (A : ℝ) : ℝ :=
  1 / 2 * b * c * sin A

theorem triangle_problems
  (h1 : triangle_sides_and_angles a b c A B C)
  (h2 : m = (1, 1))
  (h3 : n = (sqrt 3 / 2 - sin B * sin C, cos B * cos C))
  (h4 : perpendicular m n)
  (h5 : a = 1)
  (h6 : b = sqrt 3 * c) :
  A = π / 6 ∧ area_of_triangle a b c A = sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_problems_l2277_227794


namespace NUMINAMATH_GPT_percentage_of_water_in_first_liquid_l2277_227732

theorem percentage_of_water_in_first_liquid (x : ℝ) 
  (h1 : 0 < x ∧ x ≤ 1)
  (h2 : 0.35 = 0.35)
  (h3 : 10 = 10)
  (h4 : 4 = 4)
  (h5 : 0.24285714285714285 = 0.24285714285714285) :
  ((10 * x + 4 * 0.35) / (10 + 4) = 0.24285714285714285) → (x = 0.2) :=
sorry

end NUMINAMATH_GPT_percentage_of_water_in_first_liquid_l2277_227732


namespace NUMINAMATH_GPT_coordinates_of_point_with_respect_to_origin_l2277_227738

theorem coordinates_of_point_with_respect_to_origin (P : ℝ × ℝ) (h : P = (-2, 4)) : P = (-2, 4) := 
by 
  exact h

end NUMINAMATH_GPT_coordinates_of_point_with_respect_to_origin_l2277_227738


namespace NUMINAMATH_GPT_rearrange_marked_squares_l2277_227758

theorem rearrange_marked_squares (n k : ℕ) (h : n > 1) (h' : k ≤ n + 1) :
  ∃ (f g : Fin n → Fin n), true := sorry

end NUMINAMATH_GPT_rearrange_marked_squares_l2277_227758


namespace NUMINAMATH_GPT_triangular_pyramid_height_l2277_227740

noncomputable def pyramid_height (a b c h : ℝ) : Prop :=
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2

theorem triangular_pyramid_height {a b c h : ℝ} (h_gt_0 : h > 0) (a_gt_0 : a > 0) (b_gt_0 : b > 0) (c_gt_0 : c > 0) :
  pyramid_height a b c h := by
  sorry

end NUMINAMATH_GPT_triangular_pyramid_height_l2277_227740


namespace NUMINAMATH_GPT_lumberjack_trees_l2277_227702

theorem lumberjack_trees (trees logs firewood : ℕ) 
  (h1 : ∀ t, logs = t * 4)
  (h2 : ∀ l, firewood = l * 5)
  (h3 : firewood = 500)
  : trees = 25 :=
by
  sorry

end NUMINAMATH_GPT_lumberjack_trees_l2277_227702


namespace NUMINAMATH_GPT_part1_l2277_227761

def p (m x : ℝ) := x^2 - 3*m*x + 2*m^2 ≤ 0
def q (x : ℝ) := (x + 2)^2 < 1

theorem part1 (x : ℝ) (m : ℝ) (hm : m = -2) : p m x ∧ q x ↔ -3 < x ∧ x ≤ -2 :=
by
  unfold p q
  sorry

end NUMINAMATH_GPT_part1_l2277_227761


namespace NUMINAMATH_GPT_whale_sixth_hour_consumption_l2277_227788

-- Definitions based on the given conditions
def consumption (x : ℕ) (hour : ℕ) : ℕ := x + 3 * (hour - 1)

def total_consumption (x : ℕ) : ℕ := 
  (consumption x 1) + (consumption x 2) + (consumption x 3) +
  (consumption x 4) + (consumption x 5) + (consumption x 6) + 
  (consumption x 7) + (consumption x 8) + (consumption x 9)

-- Given problem translated to Lean
theorem whale_sixth_hour_consumption (x : ℕ) (h1 : total_consumption x = 270) :
  consumption x 6 = 33 :=
sorry

end NUMINAMATH_GPT_whale_sixth_hour_consumption_l2277_227788


namespace NUMINAMATH_GPT_proof_x_y_3_l2277_227716

noncomputable def prime (n : ℤ) : Prop := 2 <= n ∧ ∀ m : ℤ, 1 ≤ m → m < n → n % m ≠ 0

theorem proof_x_y_3 (x y : ℝ) (p q r : ℤ) (h1 : x - y = p) (hp : prime p) 
  (h2 : x^2 - y^2 = q) (hq : prime q)
  (h3 : x^3 - y^3 = r) (hr : prime r) : p = 3 :=
sorry

end NUMINAMATH_GPT_proof_x_y_3_l2277_227716


namespace NUMINAMATH_GPT_range_of_a_for_extreme_points_l2277_227762

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * Real.exp (2 * x)

theorem range_of_a_for_extreme_points :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    ∀ a : ℝ, 0 < a ∧ a < (1 / 2) →
    (Real.exp x₁ * (x₁ + 1 - 2 * a * Real.exp x₁) = 0) ∧ 
    (Real.exp x₂ * (x₂ + 1 - 2 * a * Real.exp x₂) = 0)) ↔ 
  ∀ a : ℝ, 0 < a ∧ a < (1 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_for_extreme_points_l2277_227762


namespace NUMINAMATH_GPT_largest_spherical_ball_radius_in_torus_l2277_227785

theorem largest_spherical_ball_radius_in_torus 
    (inner_radius outer_radius : ℝ) 
    (circle_center : ℝ × ℝ × ℝ) 
    (circle_radius : ℝ) 
    (r : ℝ)
    (h0 : inner_radius = 2)
    (h1 : outer_radius = 4)
    (h2 : circle_center = (3, 0, 1))
    (h3 : circle_radius = 1)
    (h4 : 3^2 + (r - 1)^2 = (r + 1)^2) :
    r = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_spherical_ball_radius_in_torus_l2277_227785


namespace NUMINAMATH_GPT_warehouse_capacity_l2277_227781

theorem warehouse_capacity (total_bins num_20_ton_bins cap_20_ton_bin cap_15_ton_bin : Nat) 
  (h1 : total_bins = 30) 
  (h2 : num_20_ton_bins = 12) 
  (h3 : cap_20_ton_bin = 20) 
  (h4 : cap_15_ton_bin = 15) : 
  total_bins * cap_20_ton_bin + (total_bins - num_20_ton_bins) * cap_15_ton_bin = 510 := 
by
  sorry

end NUMINAMATH_GPT_warehouse_capacity_l2277_227781


namespace NUMINAMATH_GPT_triangle_perimeter_l2277_227768

theorem triangle_perimeter (r : ℝ) (A B C P Q R S T : ℝ)
  (triangle_isosceles : A = C)
  (circle_tangent : P = A ∧ Q = B ∧ R = B ∧ S = C ∧ T = C)
  (center_dist : P + Q = 2 ∧ Q + R = 2 ∧ R + S = 2 ∧ S + T = 2) :
  2 * (A + B + C) = 6 := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2277_227768


namespace NUMINAMATH_GPT_original_decimal_number_l2277_227701

theorem original_decimal_number (x : ℝ) (h : 10 * x - x / 10 = 23.76) : x = 2.4 :=
sorry

end NUMINAMATH_GPT_original_decimal_number_l2277_227701


namespace NUMINAMATH_GPT_students_without_favorite_subject_l2277_227791

theorem students_without_favorite_subject
  (total_students : ℕ)
  (students_like_math : ℕ)
  (students_like_english : ℕ)
  (remaining_students : ℕ)
  (students_like_science : ℕ)
  (students_without_favorite : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_math = total_students * (1 / 5))
  (h3 : students_like_english = total_students * (1 / 3))
  (h4 : remaining_students = total_students - (students_like_math + students_like_english))
  (h5 : students_like_science = remaining_students * (1 / 7))
  (h6 : students_without_favorite = remaining_students - students_like_science) :
  students_without_favorite = 12 := by
  sorry

end NUMINAMATH_GPT_students_without_favorite_subject_l2277_227791


namespace NUMINAMATH_GPT_sin_identity_l2277_227729

theorem sin_identity (α : ℝ) (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 :=
sorry

end NUMINAMATH_GPT_sin_identity_l2277_227729


namespace NUMINAMATH_GPT_circle_radius_l2277_227718

theorem circle_radius (r M N : ℝ) (hM : M = π * r^2) (hN : N = 2 * π * r) (hRatio : M / N = 20) : r = 40 := 
by
  sorry

end NUMINAMATH_GPT_circle_radius_l2277_227718


namespace NUMINAMATH_GPT_x_sq_sub_y_sq_l2277_227709

theorem x_sq_sub_y_sq (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_x_sq_sub_y_sq_l2277_227709


namespace NUMINAMATH_GPT_factorize_a3_minus_4ab2_l2277_227706

theorem factorize_a3_minus_4ab2 (a b : ℝ) : a^3 - 4 * a * b^2 = a * (a + 2 * b) * (a - 2 * b) :=
by
  -- Proof is omitted; write 'sorry' as a placeholder
  sorry

end NUMINAMATH_GPT_factorize_a3_minus_4ab2_l2277_227706


namespace NUMINAMATH_GPT_can_capacity_l2277_227734

/-- Given a can with a mixture of milk and water in the ratio 4:3, and adding 10 liters of milk
results in the can being full and changes the ratio to 5:2, prove that the capacity of the can is 30 liters. -/
theorem can_capacity (x : ℚ)
  (h1 : 4 * x + 3 * x + 10 = 30)
  (h2 : (4 * x + 10) / (3 * x) = 5 / 2) :
  4 * x + 3 * x + 10 = 30 := 
by sorry

end NUMINAMATH_GPT_can_capacity_l2277_227734


namespace NUMINAMATH_GPT_correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l2277_227770

theorem correct_removal_of_parentheses_C (a : ℝ) :
    -(2 * a - 1) = -2 * a + 1 :=
by sorry

theorem incorrect_removal_of_parentheses_A (a : ℝ) :
    -(7 * a - 5) ≠ -7 * a - 5 :=
by sorry

theorem incorrect_removal_of_parentheses_B (a : ℝ) :
    -(-1 / 2 * a + 2) ≠ -1 / 2 * a - 2 :=
by sorry

theorem incorrect_removal_of_parentheses_D (a : ℝ) :
    -(-3 * a + 2) ≠ 3 * a + 2 :=
by sorry

end NUMINAMATH_GPT_correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l2277_227770


namespace NUMINAMATH_GPT_find_fractions_l2277_227703

open Function

-- Define the set and the condition that all numbers must be used precisely once
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define what it means for fractions to multiply to 1 within the set
def fractions_mul_to_one (a b c d e f : ℕ) : Prop :=
  (a * c * e) = (b * d * f)

-- Define irreducibility condition for a fraction a/b
def irreducible_fraction (a b : ℕ) := 
  Nat.gcd a b = 1

-- Final main problem statement
theorem find_fractions :
  ∃ (a b c d e f : ℕ) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : c ∈ S) (h₄ : d ∈ S) (h₅ : e ∈ S) (h₆ : f ∈ S),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  irreducible_fraction a b ∧ irreducible_fraction c d ∧ irreducible_fraction e f ∧
  fractions_mul_to_one a b c d e f := 
sorry

end NUMINAMATH_GPT_find_fractions_l2277_227703


namespace NUMINAMATH_GPT_smallest_base_l2277_227741

theorem smallest_base (b : ℕ) : (b^2 ≤ 80 ∧ 80 < b^3) → b = 5 := by
  sorry

end NUMINAMATH_GPT_smallest_base_l2277_227741


namespace NUMINAMATH_GPT_find_number_l2277_227773

theorem find_number 
  (x y n : ℝ)
  (h1 : n * x = 0.04 * y)
  (h2 : (y - x) / (y + x) = 0.948051948051948) :
  n = 37.5 :=
sorry  -- proof omitted

end NUMINAMATH_GPT_find_number_l2277_227773


namespace NUMINAMATH_GPT_unique_integer_solution_l2277_227708

def is_point_in_circle (x y cx cy radius : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 ≤ radius^2

theorem unique_integer_solution : ∃! (x : ℤ), is_point_in_circle (2 * x) (-x) 4 6 8 := by
  sorry

end NUMINAMATH_GPT_unique_integer_solution_l2277_227708


namespace NUMINAMATH_GPT_sarah_amount_l2277_227760

theorem sarah_amount:
  ∀ (X : ℕ), (X + (X + 50) = 300) → X = 125 := by
  sorry

end NUMINAMATH_GPT_sarah_amount_l2277_227760


namespace NUMINAMATH_GPT_problem_l2277_227790

noncomputable def p (k : ℝ) (x : ℝ) := k * (x - 5) * (x - 2)
noncomputable def q (x : ℝ) := (x - 5) * (x + 3)

theorem problem {p q : ℝ → ℝ} (k : ℝ) :
  (∀ x, q x = (x - 5) * (x + 3)) →
  (∀ x, p x = k * (x - 5) * (x - 2)) →
  (∀ x ≠ 5, (p x) / (q x) = (3 * (x - 2)) / (x + 3)) →
  p 3 / q 3 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2277_227790


namespace NUMINAMATH_GPT_find_x_l2277_227756

-- Let's define the constants and the condition
def a : ℝ := 2.12
def b : ℝ := 0.345
def c : ℝ := 2.4690000000000003

-- We need to prove that there exists a number x such that
def x : ℝ := 0.0040000000000003

-- Formal statement
theorem find_x : a + b + x = c :=
by
  -- Proof skipped
  sorry
 
end NUMINAMATH_GPT_find_x_l2277_227756


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2277_227786

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2277_227786


namespace NUMINAMATH_GPT_is_minimum_value_l2277_227742

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem is_minimum_value (h : ∀ x > 0, f x ≥ 0) : ∃ (a : ℝ) (h : a > 0), f a = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_is_minimum_value_l2277_227742


namespace NUMINAMATH_GPT_exists_k_lt_ak_by_2001_fac_l2277_227744

theorem exists_k_lt_ak_by_2001_fac (a : ℕ → ℝ) (H0 : a 0 = 1)
(Hn : ∀ n : ℕ, n > 0 → a n = a (⌊(7 * n / 9)⌋₊) + a (⌊(n / 9)⌋₊)) :
  ∃ k : ℕ, k > 0 ∧ a k < k / ↑(Nat.factorial 2001) := by
  sorry

end NUMINAMATH_GPT_exists_k_lt_ak_by_2001_fac_l2277_227744


namespace NUMINAMATH_GPT_cos_triple_angle_l2277_227764

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l2277_227764


namespace NUMINAMATH_GPT_number_of_cities_sampled_from_group_B_l2277_227754

variable (N_total : ℕ) (N_A : ℕ) (N_B : ℕ) (N_C : ℕ) (S : ℕ)

theorem number_of_cities_sampled_from_group_B :
    N_total = 48 → 
    N_A = 10 → 
    N_B = 18 → 
    N_C = 20 → 
    S = 16 → 
    (N_B * S) / N_total = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cities_sampled_from_group_B_l2277_227754


namespace NUMINAMATH_GPT_area_enclosed_by_line_and_curve_l2277_227796

theorem area_enclosed_by_line_and_curve :
  ∃ area, ∀ (x : ℝ), x^2 = 4 * (x - 4/2) → 
    area = ∫ (t : ℝ) in Set.Icc (-1 : ℝ) 2, (1/4 * t + 1/2 - 1/4 * t^2) :=
sorry

end NUMINAMATH_GPT_area_enclosed_by_line_and_curve_l2277_227796


namespace NUMINAMATH_GPT_nine_chapters_coins_l2277_227700

theorem nine_chapters_coins (a d : ℚ)
  (h1 : (a - 2 * d) + (a - d) = a + (a + d) + (a + 2 * d))
  (h2 : (a - 2 * d) + (a - d) + a + (a + d) + (a + 2 * d) = 5) :
  a - d = 7 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_nine_chapters_coins_l2277_227700


namespace NUMINAMATH_GPT_corn_growth_first_week_l2277_227723

theorem corn_growth_first_week (x : ℝ) (h1 : x + 2*x + 8*x = 22) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_corn_growth_first_week_l2277_227723


namespace NUMINAMATH_GPT_chromosome_stability_due_to_meiosis_and_fertilization_l2277_227748

-- Definitions for conditions
def chrom_replicate_distribute_evenly : Prop := true
def central_cell_membrane_invagination : Prop := true
def mitosis : Prop := true
def meiosis_and_fertilization : Prop := true

-- Main theorem statement to be proved
theorem chromosome_stability_due_to_meiosis_and_fertilization :
  meiosis_and_fertilization :=
sorry

end NUMINAMATH_GPT_chromosome_stability_due_to_meiosis_and_fertilization_l2277_227748


namespace NUMINAMATH_GPT_probability_log3_N_integer_l2277_227728
noncomputable def probability_log3_integer : ℚ :=
  let count := 2
  let total := 900
  count / total

theorem probability_log3_N_integer :
  probability_log3_integer = 1 / 450 :=
sorry

end NUMINAMATH_GPT_probability_log3_N_integer_l2277_227728


namespace NUMINAMATH_GPT_julia_age_correct_l2277_227736

def julia_age_proof : Prop :=
  ∃ (j : ℚ) (m : ℚ), m = 15 * j ∧ m - j = 40 ∧ j = 20 / 7

theorem julia_age_correct : julia_age_proof :=
by
  sorry

end NUMINAMATH_GPT_julia_age_correct_l2277_227736


namespace NUMINAMATH_GPT_bruce_money_left_l2277_227746

theorem bruce_money_left :
  let initial_amount := 71
  let cost_per_shirt := 5
  let number_of_shirts := 5
  let cost_of_pants := 26
  let total_cost := number_of_shirts * cost_per_shirt + cost_of_pants
  let money_left := initial_amount - total_cost
  money_left = 20 :=
by
  sorry

end NUMINAMATH_GPT_bruce_money_left_l2277_227746


namespace NUMINAMATH_GPT_seulgi_second_round_score_l2277_227752

theorem seulgi_second_round_score
    (h_score1 : Nat) (h_score2 : Nat)
    (hj_score1 : Nat) (hj_score2 : Nat)
    (s_score1 : Nat) (required_second_score : Nat) :
    h_score1 = 23 →
    h_score2 = 28 →
    hj_score1 = 32 →
    hj_score2 = 17 →
    s_score1 = 27 →
    required_second_score = 25 →
    s_score1 + required_second_score > h_score1 + h_score2 ∧ 
    s_score1 + required_second_score > hj_score1 + hj_score2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_seulgi_second_round_score_l2277_227752
