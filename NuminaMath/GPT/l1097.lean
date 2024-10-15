import Mathlib

namespace NUMINAMATH_GPT_find_chord_line_eq_l1097_109717

theorem find_chord_line_eq (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ)
    (hP : P = (1, 1)) (hC : C = (3, 0)) (hr : r = 3)
    (circle_eq : ∀ (x y : ℝ), (x - 3)^2 + y^2 = r^2) :
    ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = -1 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := by
  sorry

end NUMINAMATH_GPT_find_chord_line_eq_l1097_109717


namespace NUMINAMATH_GPT_all_numbers_divisible_by_5_l1097_109786

variable {a b c d e f g : ℕ}

-- Seven natural numbers and the condition that the sum of any six is divisible by 5
axiom cond_a : (a + b + c + d + e + f) % 5 = 0
axiom cond_b : (b + c + d + e + f + g) % 5 = 0
axiom cond_c : (a + c + d + e + f + g) % 5 = 0
axiom cond_d : (a + b + c + e + f + g) % 5 = 0
axiom cond_e : (a + b + c + d + f + g) % 5 = 0
axiom cond_f : (a + b + c + d + e + g) % 5 = 0
axiom cond_g : (a + b + c + d + e + f) % 5 = 0

theorem all_numbers_divisible_by_5 :
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0 ∧ f % 5 = 0 ∧ g % 5 = 0 :=
sorry

end NUMINAMATH_GPT_all_numbers_divisible_by_5_l1097_109786


namespace NUMINAMATH_GPT_compute_expression_l1097_109710

theorem compute_expression (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 6) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1097_109710


namespace NUMINAMATH_GPT_min_value_inequality_l1097_109759

theorem min_value_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  ∃ n : ℝ, n = 9 / 4 ∧ (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 2 → (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ n) :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l1097_109759


namespace NUMINAMATH_GPT_remainder_4x_mod_7_l1097_109753

theorem remainder_4x_mod_7 (x : ℤ) (k : ℤ) (h : x = 7 * k + 5) : (4 * x) % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_4x_mod_7_l1097_109753


namespace NUMINAMATH_GPT_student_A_selection_probability_l1097_109772

def probability_student_A_selected (total_students : ℕ) (students_removed : ℕ) (representatives : ℕ) : ℚ :=
  representatives / (total_students : ℚ)

theorem student_A_selection_probability :
  probability_student_A_selected 752 2 5 = 5 / 752 :=
by
  sorry

end NUMINAMATH_GPT_student_A_selection_probability_l1097_109772


namespace NUMINAMATH_GPT_tory_toys_sold_is_7_l1097_109722

-- Define the conditions as Lean definitions
def bert_toy_phones_sold : Nat := 8
def price_per_toy_phone : Nat := 18
def bert_earnings : Nat := bert_toy_phones_sold * price_per_toy_phone
def tory_earnings : Nat := bert_earnings - 4
def price_per_toy_gun : Nat := 20
def tory_toys_sold := tory_earnings / price_per_toy_gun

-- Prove that the number of toy guns Tory sold is 7
theorem tory_toys_sold_is_7 : tory_toys_sold = 7 :=
by
  sorry

end NUMINAMATH_GPT_tory_toys_sold_is_7_l1097_109722


namespace NUMINAMATH_GPT_at_least_one_genuine_l1097_109758

theorem at_least_one_genuine :
  ∀ (total_products genuine_products defective_products selected_products : ℕ),
  total_products = 12 →
  genuine_products = 10 →
  defective_products = 2 →
  selected_products = 3 →
  (∃ g d : ℕ, g + d = selected_products ∧ g = 0 ∧ d = selected_products) = false :=
by
  intros total_products genuine_products defective_products selected_products
  intros H_total H_gen H_def H_sel
  sorry

end NUMINAMATH_GPT_at_least_one_genuine_l1097_109758


namespace NUMINAMATH_GPT_probability_of_urn_contains_nine_red_and_four_blue_after_operations_l1097_109767

-- Definition of the initial urn state
def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1

-- Definition of the number of operations
def num_operations : ℕ := 5

-- Definition of the final state
def final_red_balls : ℕ := 9
def final_blue_balls : ℕ := 4

-- Definition of total number of balls after five operations
def total_balls_after_operations : ℕ := 13

-- The probability we aim to prove
def target_probability : ℚ := 1920 / 10395

noncomputable def george_experiment_probability_theorem 
  (initial_red_balls initial_blue_balls num_operations final_red_balls final_blue_balls : ℕ)
  (total_balls_after_operations : ℕ) : ℚ :=
if initial_red_balls = 2 ∧ initial_blue_balls = 1 ∧ num_operations = 5 ∧ final_red_balls = 9 ∧ final_blue_balls = 4 ∧ total_balls_after_operations = 13 then
  target_probability
else
  0

-- The theorem statement, no proof provided (using sorry).
theorem probability_of_urn_contains_nine_red_and_four_blue_after_operations :
  george_experiment_probability_theorem 2 1 5 9 4 13 = target_probability := sorry

end NUMINAMATH_GPT_probability_of_urn_contains_nine_red_and_four_blue_after_operations_l1097_109767


namespace NUMINAMATH_GPT_trig_relationship_l1097_109737

noncomputable def a := Real.cos 1
noncomputable def b := Real.cos 2
noncomputable def c := Real.sin 2

theorem trig_relationship : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_trig_relationship_l1097_109737


namespace NUMINAMATH_GPT_roots_eq_squares_l1097_109790

theorem roots_eq_squares (p q : ℝ) (h1 : p^2 - 5 * p + 6 = 0) (h2 : q^2 - 5 * q + 6 = 0) :
  p^2 + q^2 = 13 :=
sorry

end NUMINAMATH_GPT_roots_eq_squares_l1097_109790


namespace NUMINAMATH_GPT_inequality_absolute_value_l1097_109754

theorem inequality_absolute_value (a b : ℝ) (h1 : a < b) (h2 : b < 0) : |a| > -b :=
sorry

end NUMINAMATH_GPT_inequality_absolute_value_l1097_109754


namespace NUMINAMATH_GPT_right_angle_locus_l1097_109796

noncomputable def P (x y : ℝ) : Prop :=
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16

theorem right_angle_locus (x y : ℝ) : P x y → x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2 :=
by
  sorry

end NUMINAMATH_GPT_right_angle_locus_l1097_109796


namespace NUMINAMATH_GPT_area_of_park_l1097_109794

noncomputable def length (x : ℕ) : ℕ := 3 * x
noncomputable def width (x : ℕ) : ℕ := 2 * x
noncomputable def area (x : ℕ) : ℕ := length x * width x
noncomputable def cost_per_meter : ℕ := 80
noncomputable def total_cost : ℕ := 200
noncomputable def perimeter (x : ℕ) : ℕ := 2 * (length x + width x)

theorem area_of_park : ∃ x : ℕ, area x = 3750 ∧ total_cost = (perimeter x) * cost_per_meter / 100 := by
  sorry

end NUMINAMATH_GPT_area_of_park_l1097_109794


namespace NUMINAMATH_GPT_total_peanuts_in_box_l1097_109783

def initial_peanuts := 4
def peanuts_taken_out := 3
def peanuts_added := 12

theorem total_peanuts_in_box : initial_peanuts - peanuts_taken_out + peanuts_added = 13 :=
by
sorry

end NUMINAMATH_GPT_total_peanuts_in_box_l1097_109783


namespace NUMINAMATH_GPT_average_speed_l1097_109730

-- Definitions of conditions
def speed_first_hour : ℝ := 120
def speed_second_hour : ℝ := 60
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := 2

-- Theorem stating the equivalent proof problem
theorem average_speed : total_distance / total_time = 90 := by
  sorry

end NUMINAMATH_GPT_average_speed_l1097_109730


namespace NUMINAMATH_GPT_eighth_term_is_79_l1097_109740

variable (a d : ℤ)

def fourth_term_condition : Prop := a + 3 * d = 23
def sixth_term_condition : Prop := a + 5 * d = 51

theorem eighth_term_is_79 (h₁ : fourth_term_condition a d) (h₂ : sixth_term_condition a d) : a + 7 * d = 79 :=
sorry

end NUMINAMATH_GPT_eighth_term_is_79_l1097_109740


namespace NUMINAMATH_GPT_intersection_of_sets_l1097_109721

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 ≥ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x < 2

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1097_109721


namespace NUMINAMATH_GPT_point_A_final_position_supplement_of_beta_l1097_109771

-- Define the initial and final position of point A on the number line
def initial_position := -5
def moved_position_right := initial_position + 4
def final_position := moved_position_right - 1

theorem point_A_final_position : final_position = -2 := 
by 
-- Proof can be added here
sorry

-- Define the angles and the relationship between them
def alpha := 40
def beta := 90 - alpha
def supplement_beta := 180 - beta

theorem supplement_of_beta : supplement_beta = 130 := 
by 
-- Proof can be added here
sorry

end NUMINAMATH_GPT_point_A_final_position_supplement_of_beta_l1097_109771


namespace NUMINAMATH_GPT_probability_matching_shoes_l1097_109751

theorem probability_matching_shoes :
  let total_shoes := 24;
  let total_pairs := 12;
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2;
  let matching_pairs := total_pairs;
  let probability := matching_pairs / total_combinations;
  probability = 1 / 23 :=
by
  let total_shoes := 24
  let total_pairs := 12
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := total_pairs
  let probability := matching_pairs / total_combinations
  have : total_combinations = 276 := by norm_num
  have : matching_pairs = 12 := by norm_num
  have : probability = 1 / 23 := by norm_num
  exact this

end NUMINAMATH_GPT_probability_matching_shoes_l1097_109751


namespace NUMINAMATH_GPT_necessary_condition_for_inequality_l1097_109793

theorem necessary_condition_for_inequality 
  (m : ℝ) : (∀ x : ℝ, x^2 - 2 * x + m > 0) → m > 0 :=
by 
  sorry

end NUMINAMATH_GPT_necessary_condition_for_inequality_l1097_109793


namespace NUMINAMATH_GPT_susana_chocolate_chips_l1097_109736

theorem susana_chocolate_chips :
  ∃ (S_c : ℕ), 
  (∃ (V_c V_v S_v : ℕ), 
    V_c = S_c + 5 ∧
    S_v = (3 * V_v) / 4 ∧
    V_v = 20 ∧
    V_c + S_c + V_v + S_v = 90) ∧
  S_c = 25 :=
by
  existsi 25
  sorry

end NUMINAMATH_GPT_susana_chocolate_chips_l1097_109736


namespace NUMINAMATH_GPT_correct_operation_l1097_109708

theorem correct_operation (a b : ℝ) : 
  (2 * a) * (3 * a) = 6 * a^2 :=
by
  -- The proof would be here; using "sorry" to skip the actual proof steps.
  sorry

end NUMINAMATH_GPT_correct_operation_l1097_109708


namespace NUMINAMATH_GPT_find_f_m_l1097_109760

noncomputable def f (x : ℝ) := x^5 + Real.tan x - 3

theorem find_f_m (m : ℝ) (h : f (-m) = -2) : f m = -4 :=
sorry

end NUMINAMATH_GPT_find_f_m_l1097_109760


namespace NUMINAMATH_GPT_fuel_cost_equation_l1097_109707

theorem fuel_cost_equation (x : ℝ) (h : (x / 4) - (x / 6) = 8) : x = 96 :=
sorry

end NUMINAMATH_GPT_fuel_cost_equation_l1097_109707


namespace NUMINAMATH_GPT_total_cost_of_4_sandwiches_and_6_sodas_is_37_4_l1097_109768

def sandwiches_cost (s: ℕ) : ℝ := 4 * s
def sodas_cost (d: ℕ) : ℝ := 3 * d
def total_cost_before_tax (s: ℕ) (d: ℕ) : ℝ := sandwiches_cost s + sodas_cost d
def tax (amount: ℝ) : ℝ := 0.10 * amount
def total_cost (s: ℕ) (d: ℕ) : ℝ := total_cost_before_tax s d + tax (total_cost_before_tax s d)

theorem total_cost_of_4_sandwiches_and_6_sodas_is_37_4 :
    total_cost 4 6 = 37.4 :=
sorry

end NUMINAMATH_GPT_total_cost_of_4_sandwiches_and_6_sodas_is_37_4_l1097_109768


namespace NUMINAMATH_GPT_trig_functions_symmetry_l1097_109706

theorem trig_functions_symmetry :
  ∀ k₁ k₂ : ℤ,
  (∃ x, x = k₁ * π / 2 + π / 3 ∧ x = k₂ * π + π / 3) ∧
  (¬ ∃ x, (x, 0) = (k₁ * π / 2 + π / 12, 0) ∧ (x, 0) = (k₂ * π + 5 * π / 6, 0)) :=
by
  sorry

end NUMINAMATH_GPT_trig_functions_symmetry_l1097_109706


namespace NUMINAMATH_GPT_matthew_initial_crackers_l1097_109776

theorem matthew_initial_crackers :
  ∃ C : ℕ,
  (∀ (crackers_per_friend cakes_per_friend : ℕ), cakes_per_friend * 4 = 98 → crackers_per_friend = cakes_per_friend → crackers_per_friend * 4 + 8 * 4 = C) ∧ C = 128 :=
sorry

end NUMINAMATH_GPT_matthew_initial_crackers_l1097_109776


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1097_109755

theorem sufficient_but_not_necessary (a b : ℝ) :
  ((a - b) ^ 3 * b ^ 2 > 0 → a > b) ∧ ¬(a > b → (a - b) ^ 3 * b ^ 2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1097_109755


namespace NUMINAMATH_GPT_trigonometric_identity_l1097_109780

theorem trigonometric_identity (α : ℝ) :
  (2 * (Real.cos (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) / 
  (2 * (Real.sin (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) = 
  Real.sin (4 * α + Real.pi / 6) / Real.sin (4 * α - Real.pi / 6) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1097_109780


namespace NUMINAMATH_GPT_total_decorations_l1097_109728

-- Define the conditions
def decorations_per_box := 4 + 1 + 5
def total_boxes := 11 + 1

-- Statement of the problem: Prove that the total number of decorations handed out is 120
theorem total_decorations : total_boxes * decorations_per_box = 120 := by
  sorry

end NUMINAMATH_GPT_total_decorations_l1097_109728


namespace NUMINAMATH_GPT_residue_mod_13_l1097_109752

theorem residue_mod_13 :
  (250 ≡ 3 [MOD 13]) → 
  (20 ≡ 7 [MOD 13]) → 
  (5^2 ≡ 12 [MOD 13]) → 
  ((250 * 11 - 20 * 6 + 5^2) % 13 = 3) :=
by 
  sorry

end NUMINAMATH_GPT_residue_mod_13_l1097_109752


namespace NUMINAMATH_GPT_new_person_weight_l1097_109700

theorem new_person_weight (N : ℝ) (h : N - 65 = 22.5) : N = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l1097_109700


namespace NUMINAMATH_GPT_calculate_value_l1097_109775

-- Given conditions
def n : ℝ := 2.25

-- Lean statement to express the proof problem
theorem calculate_value : (n / 3) * 12 = 9 := by
  -- Proof will be supplied here
  sorry

end NUMINAMATH_GPT_calculate_value_l1097_109775


namespace NUMINAMATH_GPT_distance_to_big_rock_l1097_109713

variables (D : ℝ) (stillWaterSpeed : ℝ) (currentSpeed : ℝ) (totalTime : ℝ)

-- Define the conditions as constraints
def conditions := 
  stillWaterSpeed = 6 ∧
  currentSpeed = 1 ∧
  totalTime = 1 ∧
  (D / (stillWaterSpeed - currentSpeed) + D / (stillWaterSpeed + currentSpeed) = totalTime)

-- The theorem to prove the distance to Big Rock
theorem distance_to_big_rock (h : conditions D 6 1 1) : D = 35 / 12 :=
sorry

end NUMINAMATH_GPT_distance_to_big_rock_l1097_109713


namespace NUMINAMATH_GPT_value_of_x_l1097_109727

theorem value_of_x (w : ℝ) (hw : w = 90) (z : ℝ) (hz : z = 2 / 3 * w) (y : ℝ) (hy : y = 1 / 4 * z) (x : ℝ) (hx : x = 1 / 2 * y) : x = 7.5 :=
by
  -- Proof skipped; conclusion derived from conditions
  sorry

end NUMINAMATH_GPT_value_of_x_l1097_109727


namespace NUMINAMATH_GPT_only_composite_positive_integer_with_divisors_form_l1097_109704

theorem only_composite_positive_integer_with_divisors_form (n : ℕ) (composite : ¬Nat.Prime n ∧ 1 < n)
  (H : ∀ d ∈ Nat.divisors n, ∃ (a r : ℕ), a ≥ 0 ∧ r ≥ 2 ∧ d = a^r + 1) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_only_composite_positive_integer_with_divisors_form_l1097_109704


namespace NUMINAMATH_GPT_maximum_distance_l1097_109798

-- Given conditions for the problem.
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline : ℝ := 23

-- Problem statement: prove the maximum distance on highway mileage.
theorem maximum_distance : highway_mpg * gasoline = 280.6 :=
sorry

end NUMINAMATH_GPT_maximum_distance_l1097_109798


namespace NUMINAMATH_GPT_diagonals_from_one_vertex_l1097_109734

theorem diagonals_from_one_vertex (x : ℕ) (h : (x - 2) * 180 = 1800) : (x - 3) = 9 :=
  by
  sorry

end NUMINAMATH_GPT_diagonals_from_one_vertex_l1097_109734


namespace NUMINAMATH_GPT_solve_exponents_l1097_109719

theorem solve_exponents (x y z : ℕ) (hx : x < y) (hy : y < z) 
  (h : 3^x + 3^y + 3^z = 179415) : x = 4 ∧ y = 7 ∧ z = 11 :=
by sorry

end NUMINAMATH_GPT_solve_exponents_l1097_109719


namespace NUMINAMATH_GPT_yarn_for_second_ball_l1097_109784

variable (first_ball second_ball third_ball : ℝ) (yarn_used : ℝ)

-- Conditions
variable (h1 : first_ball = second_ball / 2)
variable (h2 : third_ball = 3 * first_ball)
variable (h3 : third_ball = 27)

-- Question: Prove that the second ball used 18 feet of yarn.
theorem yarn_for_second_ball (h1 : first_ball = second_ball / 2) (h2 : third_ball = 3 * first_ball) (h3 : third_ball = 27) :
  second_ball = 18 := by
  sorry

end NUMINAMATH_GPT_yarn_for_second_ball_l1097_109784


namespace NUMINAMATH_GPT_find_t_l1097_109746

variables (t : ℝ)

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, t)
def a_plus_b : ℝ × ℝ := (2, 1 + t)

def are_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_t (t : ℝ) :
  are_parallel (3, t) (2, 1 + t) ↔ t = -3 :=
sorry

end NUMINAMATH_GPT_find_t_l1097_109746


namespace NUMINAMATH_GPT_arithmetic_geometric_progression_l1097_109731

theorem arithmetic_geometric_progression (a b : ℝ) :
  (b = 2 - a) ∧ (b = 1 / a ∨ b = -1 / a) →
  (a = 1 ∧ b = 1) ∨
  (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
  (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_progression_l1097_109731


namespace NUMINAMATH_GPT_earl_envelope_rate_l1097_109709

theorem earl_envelope_rate:
  ∀ (E L : ℝ),
  L = (2/3) * E ∧
  (E + L = 60) →
  E = 36 :=
by
  intros E L h
  sorry

end NUMINAMATH_GPT_earl_envelope_rate_l1097_109709


namespace NUMINAMATH_GPT_driver_net_rate_of_pay_is_30_33_l1097_109774

noncomputable def driver_net_rate_of_pay : ℝ :=
  let hours := 3
  let speed_mph := 65
  let miles_per_gallon := 30
  let pay_per_mile := 0.55
  let cost_per_gallon := 2.50
  let total_distance := speed_mph * hours
  let gallons_used := total_distance / miles_per_gallon
  let gross_earnings := total_distance * pay_per_mile
  let fuel_cost := gallons_used * cost_per_gallon
  let net_earnings := gross_earnings - fuel_cost
  let net_rate_per_hour := net_earnings / hours
  net_rate_per_hour

theorem driver_net_rate_of_pay_is_30_33 :
  driver_net_rate_of_pay = 30.33 :=
by
  sorry

end NUMINAMATH_GPT_driver_net_rate_of_pay_is_30_33_l1097_109774


namespace NUMINAMATH_GPT_projectile_reaches_24_meters_l1097_109781

theorem projectile_reaches_24_meters (h : ℝ) (t : ℝ) (v₀ : ℝ) :
  (h = -4.9 * t^2 + 19.6 * t) ∧ (h = 24) → t = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_projectile_reaches_24_meters_l1097_109781


namespace NUMINAMATH_GPT_range_of_a_l1097_109738

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) ≤ 0)) ↔ (1 ≤ a ∧ a ≤ 3 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1097_109738


namespace NUMINAMATH_GPT_product_of_squares_is_perfect_square_l1097_109792

theorem product_of_squares_is_perfect_square (a b c : ℤ) (h : a * b + b * c + c * a = 1) :
    ∃ k : ℤ, (1 + a^2) * (1 + b^2) * (1 + c^2) = k^2 :=
sorry

end NUMINAMATH_GPT_product_of_squares_is_perfect_square_l1097_109792


namespace NUMINAMATH_GPT_citizen_income_l1097_109739

theorem citizen_income (I : ℝ) (h1 : ∀ I ≤ 40000, 0.15 * I = 8000) 
  (h2 : ∀ I > 40000, (0.15 * 40000 + 0.20 * (I - 40000)) = 8000) : 
  I = 50000 :=
by
  sorry

end NUMINAMATH_GPT_citizen_income_l1097_109739


namespace NUMINAMATH_GPT_rectangular_solid_width_l1097_109725

theorem rectangular_solid_width 
  (l : ℝ) (w : ℝ) (h : ℝ) (S : ℝ)
  (hl : l = 5)
  (hh : h = 1)
  (hs : S = 58) :
  2 * l * w + 2 * l * h + 2 * w * h = S → w = 4 := 
by
  intros h_surface_area 
  sorry

end NUMINAMATH_GPT_rectangular_solid_width_l1097_109725


namespace NUMINAMATH_GPT_find_term_number_l1097_109766

theorem find_term_number :
  ∃ n : ℕ, (2 * (5 : ℝ)^(1/2) = (3 * (n : ℝ) - 1)^(1/2)) ∧ n = 7 :=
sorry

end NUMINAMATH_GPT_find_term_number_l1097_109766


namespace NUMINAMATH_GPT_maximum_area_of_region_l1097_109716

/-- Given four circles with radii 2, 4, 6, and 8, tangent to the same point B 
on a line ℓ, with the two largest circles (radii 6 and 8) on the same side of ℓ,
prove that the maximum possible area of the region consisting of points lying
inside exactly one of these circles is 120π. -/
theorem maximum_area_of_region 
  (radius1 : ℝ) (radius2 : ℝ) (radius3 : ℝ) (radius4 : ℝ)
  (line : ℝ → Prop) (B : ℝ)
  (tangent1 : ∀ x, line x → dist x B = radius1) 
  (tangent2 : ∀ x, line x → dist x B = radius2)
  (tangent3 : ∀ x, line x → dist x B = radius3)
  (tangent4 : ∀ x, line x → dist x B = radius4)
  (side1 : ℕ)
  (side2 : ℕ)
  (equal_side : side1 = side2)
  (r1 : ℝ := 2) 
  (r2 : ℝ := 4)
  (r3 : ℝ := 6) 
  (r4 : ℝ := 8) :
  (π * (radius1 * radius1) + π * (radius2 * radius2) + π * (radius3 * radius3) + π * (radius4 * radius4)) = 120 * π := 
sorry

end NUMINAMATH_GPT_maximum_area_of_region_l1097_109716


namespace NUMINAMATH_GPT_choir_group_students_l1097_109745

theorem choir_group_students : ∃ n : ℕ, (n % 5 = 0) ∧ (n % 9 = 0) ∧ (n % 12 = 0) ∧ (∃ m : ℕ, n = m * m) ∧ n ≥ 360 := 
sorry

end NUMINAMATH_GPT_choir_group_students_l1097_109745


namespace NUMINAMATH_GPT_tank_capacities_l1097_109750

theorem tank_capacities (x y z : ℕ) 
  (h1 : x + y + z = 1620)
  (h2 : z = x + y / 5) 
  (h3 : z = y + x / 3) :
  x = 540 ∧ y = 450 ∧ z = 630 := 
by 
  sorry

end NUMINAMATH_GPT_tank_capacities_l1097_109750


namespace NUMINAMATH_GPT_sqrt_3x_eq_5x_largest_value_l1097_109720

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_3x_eq_5x_largest_value_l1097_109720


namespace NUMINAMATH_GPT_binomial_constant_term_l1097_109741

theorem binomial_constant_term (n : ℕ) (h : n > 0) :
  (∃ r : ℕ, n = 2 * r) ↔ (n = 6) :=
by
  sorry

end NUMINAMATH_GPT_binomial_constant_term_l1097_109741


namespace NUMINAMATH_GPT_percentage_of_students_enrolled_is_40_l1097_109762

def total_students : ℕ := 880
def not_enrolled_in_biology : ℕ := 528
def enrolled_in_biology : ℕ := total_students - not_enrolled_in_biology
def percentage_enrolled : ℕ := (enrolled_in_biology * 100) / total_students

theorem percentage_of_students_enrolled_is_40 : percentage_enrolled = 40 := by
  -- Beginning of the proof
  sorry

end NUMINAMATH_GPT_percentage_of_students_enrolled_is_40_l1097_109762


namespace NUMINAMATH_GPT_complement_union_A_B_l1097_109778

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_l1097_109778


namespace NUMINAMATH_GPT_find_m_l1097_109701

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -11.6 :=
sorry

end NUMINAMATH_GPT_find_m_l1097_109701


namespace NUMINAMATH_GPT_average_p_q_l1097_109747

theorem average_p_q (p q : ℝ) 
  (h1 : (4 + 6 + 8 + 2 * p + 2 * q) / 7 = 20) : 
  (p + q) / 2 = 30.5 :=
by
  sorry

end NUMINAMATH_GPT_average_p_q_l1097_109747


namespace NUMINAMATH_GPT_m_squared_n_minus_1_l1097_109742

theorem m_squared_n_minus_1 (a b m n : ℝ)
  (h1 : a * m^2001 + b * n^2001 = 3)
  (h2 : a * m^2002 + b * n^2002 = 7)
  (h3 : a * m^2003 + b * n^2003 = 24)
  (h4 : a * m^2004 + b * n^2004 = 102) :
  m^2 * (n - 1) = 6 := by
  sorry

end NUMINAMATH_GPT_m_squared_n_minus_1_l1097_109742


namespace NUMINAMATH_GPT_harper_jack_distance_apart_l1097_109763

def total_distance : ℕ := 1000
def distance_jack_run : ℕ := 152
def distance_apart (total_distance : ℕ) (distance_jack_run : ℕ) : ℕ :=
  total_distance - distance_jack_run 

theorem harper_jack_distance_apart :
  distance_apart total_distance distance_jack_run = 848 :=
by
  unfold distance_apart
  sorry

end NUMINAMATH_GPT_harper_jack_distance_apart_l1097_109763


namespace NUMINAMATH_GPT_divisibility_of_poly_l1097_109726

theorem divisibility_of_poly (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x):
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * (y-z) * (z-x) * (x-y) * k :=
by
  sorry

end NUMINAMATH_GPT_divisibility_of_poly_l1097_109726


namespace NUMINAMATH_GPT_ratio_of_inradii_l1097_109712

-- Given triangle XYZ with sides XZ=5, YZ=12, XY=13
-- Let W be on XY such that ZW bisects ∠ YZX
-- The inscribed circles of triangles ZWX and ZWY have radii r_x and r_y respectively
-- Prove the ratio r_x / r_y = 1/6

theorem ratio_of_inradii
  (XZ YZ XY : ℝ)
  (W : ℝ)
  (r_x r_y : ℝ)
  (h1 : XZ = 5)
  (h2 : YZ = 12)
  (h3 : XY = 13)
  (h4 : r_x / r_y = 1/6) :
  r_x / r_y = 1/6 :=
by sorry

end NUMINAMATH_GPT_ratio_of_inradii_l1097_109712


namespace NUMINAMATH_GPT_organization_population_after_six_years_l1097_109748

theorem organization_population_after_six_years :
  ∀ (b : ℕ → ℕ),
  (b 0 = 20) →
  (∀ k, b (k + 1) = 3 * (b k - 5) + 5) →
  b 6 = 10895 :=
by
  intros b h0 hr
  sorry

end NUMINAMATH_GPT_organization_population_after_six_years_l1097_109748


namespace NUMINAMATH_GPT_solve_problem_l1097_109788

-- Definitions based on conditions
def salty_cookies_eaten : ℕ := 28
def sweet_cookies_eaten : ℕ := 15

-- Problem statement
theorem solve_problem : salty_cookies_eaten - sweet_cookies_eaten = 13 := by
  sorry

end NUMINAMATH_GPT_solve_problem_l1097_109788


namespace NUMINAMATH_GPT_abs_diff_eq_l1097_109795

theorem abs_diff_eq (a b c d : ℤ) (h1 : a = 13) (h2 : b = 3) (h3 : c = 4) (h4 : d = 10) : 
  |a - b| - |c - d| = 4 := 
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_abs_diff_eq_l1097_109795


namespace NUMINAMATH_GPT_smallest_range_between_allocations_l1097_109715

-- Problem statement in Lean
theorem smallest_range_between_allocations :
  ∀ (A B C D E : ℕ), 
  (A = 30000) →
  (B < 18000 ∨ B > 42000) →
  (C < 18000 ∨ C > 42000) →
  (D < 58802 ∨ D > 82323) →
  (E < 58802 ∨ E > 82323) →
  min B (min C (min D E)) = 17999 →
  max B (max C (max D E)) = 82323 →
  82323 - 17999 = 64324 :=
by
  intros A B C D E hA hB hC hD hE hmin hmax
  sorry

end NUMINAMATH_GPT_smallest_range_between_allocations_l1097_109715


namespace NUMINAMATH_GPT_find_m_of_perpendicular_vectors_l1097_109761

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_of_perpendicular_vectors_l1097_109761


namespace NUMINAMATH_GPT_minimize_squared_distances_l1097_109703

variable {P : ℝ}

/-- Points A, B, C, D, E are collinear with distances AB = 3, BC = 3, CD = 5, and DE = 7 -/
def collinear_points : Prop :=
  ∀ (A B C D E : ℝ), B = A + 3 ∧ C = B + 3 ∧ D = C + 5 ∧ E = D + 7

/-- Define the squared distance function -/
def squared_distances (P A B C D E : ℝ) : ℝ :=
  (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2

/-- Statement of the proof problem -/
theorem minimize_squared_distances :
  collinear_points →
  ∀ (A B C D E P : ℝ), 
    squared_distances P A B C D E ≥ 181.2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_squared_distances_l1097_109703


namespace NUMINAMATH_GPT_rectangular_field_area_l1097_109782

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangular_field_area_l1097_109782


namespace NUMINAMATH_GPT_james_calories_per_minute_l1097_109756

-- Define the conditions
def bags : Nat := 3
def ounces_per_bag : Nat := 2
def calories_per_ounce : Nat := 150
def excess_calories : Nat := 420
def run_minutes : Nat := 40

-- Calculate the total consumed calories
def consumed_calories : Nat := (bags * ounces_per_bag) * calories_per_ounce

-- Calculate the calories burned during the run
def run_calories : Nat := consumed_calories - excess_calories

-- Calculate the calories burned per minute
def calories_per_minute : Nat := run_calories / run_minutes

-- The proof problem statement
theorem james_calories_per_minute : calories_per_minute = 12 := by
  -- Due to the proof not required, we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_james_calories_per_minute_l1097_109756


namespace NUMINAMATH_GPT_painting_problem_l1097_109735

theorem painting_problem
    (H_rate : ℝ := 1 / 60)
    (T_rate : ℝ := 1 / 90)
    (combined_rate : ℝ := H_rate + T_rate)
    (time_worked : ℝ := 15)
    (wall_painted : ℝ := time_worked * combined_rate):
  wall_painted = 5 / 12 := 
by
  sorry

end NUMINAMATH_GPT_painting_problem_l1097_109735


namespace NUMINAMATH_GPT_age_difference_l1097_109789

theorem age_difference (sum_ages : ℕ) (eldest_age : ℕ) (age_diff : ℕ) 
(h1 : sum_ages = 50) (h2 : eldest_age = 14) :
  14 + (14 - age_diff) + (14 - 2 * age_diff) + (14 - 3 * age_diff) + (14 - 4 * age_diff) = 50 → age_diff = 2 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_age_difference_l1097_109789


namespace NUMINAMATH_GPT_teresa_science_marks_l1097_109777

-- Definitions for the conditions
def music_marks : ℕ := 80
def social_studies_marks : ℕ := 85
def physics_marks : ℕ := music_marks / 2
def total_marks : ℕ := 275

-- Statement to prove
theorem teresa_science_marks : ∃ S : ℕ, 
  S + music_marks + social_studies_marks + physics_marks = total_marks ∧ S = 70 :=
sorry

end NUMINAMATH_GPT_teresa_science_marks_l1097_109777


namespace NUMINAMATH_GPT_mirror_area_l1097_109797

/-- The outer dimensions of the frame are given as 100 cm by 140 cm,
and the frame width is 15 cm. We aim to prove that the area of the mirror
inside the frame is 7700 cm². -/
theorem mirror_area (W H F: ℕ) (hW : W = 100) (hH : H = 140) (hF : F = 15) :
  (W - 2 * F) * (H - 2 * F) = 7700 :=
by
  sorry

end NUMINAMATH_GPT_mirror_area_l1097_109797


namespace NUMINAMATH_GPT_scalene_triangle_third_side_l1097_109779

theorem scalene_triangle_third_side (a b c : ℕ) (h : (a - 3)^2 + (b - 2)^2 = 0) : 
  a = 3 ∧ b = 2 → c = 2 ∨ c = 3 ∨ c = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_scalene_triangle_third_side_l1097_109779


namespace NUMINAMATH_GPT_value_at_2013_l1097_109743

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f x = -f (-x)
axiom periodic_5 : ∀ x : ℝ, f (x + 5) ≥ f x
axiom periodic_1 : ∀ x : ℝ, f (x + 1) ≤ f x

theorem value_at_2013 : f 2013 = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_at_2013_l1097_109743


namespace NUMINAMATH_GPT_range_of_m_l1097_109724

open Real

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - m * x + m > 0) ↔ (0 < m ∧ m < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1097_109724


namespace NUMINAMATH_GPT_total_items_bought_l1097_109702

def total_money : ℝ := 40
def sandwich_cost : ℝ := 5
def chip_cost : ℝ := 2
def soft_drink_cost : ℝ := 1.5

/-- Ike and Mike spend their total money on sandwiches, chips, and soft drinks.
  We want to prove that the total number of items bought (sandwiches, chips, and soft drinks)
  is equal to 8. -/
theorem total_items_bought :
  ∃ (s c d : ℝ), (sandwich_cost * s + chip_cost * c + soft_drink_cost * d ≤ total_money) ∧
  (∀x : ℝ, sandwich_cost * s ≤ total_money) ∧ ((s + c + d) = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_total_items_bought_l1097_109702


namespace NUMINAMATH_GPT_cocktail_cans_l1097_109765

theorem cocktail_cans (prev_apple_ratio : ℝ) (prev_grape_ratio : ℝ) 
  (new_apple_cans : ℝ) : ∃ new_grape_cans : ℝ, new_grape_cans = 15 :=
by
  let prev_apple_per_can := 1 / 6
  let prev_grape_per_can := 1 / 10
  let prev_total_per_can := (1 / 6) + (1 / 10)
  let new_apple_per_can := 1 / 5
  let new_grape_per_can := prev_total_per_can - new_apple_per_can
  let result := 1 / new_grape_per_can
  use result
  sorry

end NUMINAMATH_GPT_cocktail_cans_l1097_109765


namespace NUMINAMATH_GPT_distinct_roots_condition_l1097_109791

theorem distinct_roots_condition (a : ℝ) : 
  (∃ (x1 x2 x3 x4 : ℝ), (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧ 
  (|x1^2 - 4| = a * x1 + 6) ∧ (|x2^2 - 4| = a * x2 + 6) ∧ (|x3^2 - 4| = a * x3 + 6) ∧ (|x4^2 - 4| = a * x4 + 6)) ↔ 
  ((-3 < a ∧ a < -2 * Real.sqrt 2) ∨ (2 * Real.sqrt 2 < a ∧ a < 3)) := sorry

end NUMINAMATH_GPT_distinct_roots_condition_l1097_109791


namespace NUMINAMATH_GPT_find_abc_l1097_109799

def rearrangements (a b c : ℕ) : List ℕ :=
  [100 * a + 10 * b + c, 100 * a + 10 * c + b, 100 * b + 10 * a + c,
   100 * b + 10 * c + a, 100 * c + 10 * a + b, 100 * c + 10 * b + a]

theorem find_abc (a b c : ℕ) (habc : ℕ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (rearrangements a b c).sum = 2017 + habc →
  habc = 425 :=
by
  sorry

end NUMINAMATH_GPT_find_abc_l1097_109799


namespace NUMINAMATH_GPT_max_rock_value_l1097_109723

def rock_value (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  14 * weight_5 + 11 * weight_4 + 2 * weight_1

def total_weight (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  5 * weight_5 + 4 * weight_4 + 1 * weight_1

theorem max_rock_value : ∃ (weight_5 weight_4 weight_1 : Nat), 
  total_weight weight_5 weight_4 weight_1 ≤ 18 ∧ 
  rock_value weight_5 weight_4 weight_1 = 50 :=
by
  -- We need to find suitable weight_5, weight_4, and weight_1.
  use 2, 2, 0 -- Example values
  apply And.intro
  -- Prove the total weight condition
  show total_weight 2 2 0 ≤ 18
  sorry
  -- Prove the value condition
  show rock_value 2 2 0 = 50
  sorry

end NUMINAMATH_GPT_max_rock_value_l1097_109723


namespace NUMINAMATH_GPT_investment_in_stocks_l1097_109764

theorem investment_in_stocks (T b s : ℝ) (h1 : T = 200000) (h2 : s = 5 * b) (h3 : T = b + s) :
  s = 166666.65 :=
by sorry

end NUMINAMATH_GPT_investment_in_stocks_l1097_109764


namespace NUMINAMATH_GPT_ratio_of_areas_l1097_109749

theorem ratio_of_areas (aC aD : ℕ) (hC : aC = 48) (hD : aD = 60) : 
  (aC^2 : ℚ) / (aD^2 : ℚ) = (16 : ℚ) / (25 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1097_109749


namespace NUMINAMATH_GPT_compute_complex_power_l1097_109769

noncomputable def complex_number := Complex.exp (Complex.I * 125 * Real.pi / 180)

theorem compute_complex_power :
  (complex_number ^ 28) = Complex.ofReal (-Real.cos (40 * Real.pi / 180)) + Complex.I * Real.sin (40 * Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_compute_complex_power_l1097_109769


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1097_109770

theorem repeating_decimal_sum (c d : ℕ) (h : 7 / 19 = (c * 10 + d) / 99) : c + d = 9 :=
sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1097_109770


namespace NUMINAMATH_GPT_tan_405_eq_1_l1097_109744

theorem tan_405_eq_1 : Real.tan (405 * Real.pi / 180) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_tan_405_eq_1_l1097_109744


namespace NUMINAMATH_GPT_max_k_value_l1097_109785

noncomputable def max_k (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : ℝ :=
  let k := (-1 + Real.sqrt 7) / 2
  k

theorem max_k_value (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end NUMINAMATH_GPT_max_k_value_l1097_109785


namespace NUMINAMATH_GPT_circuit_analysis_l1097_109718

/-
There are 3 conducting branches connected between points A and B.
First branch: a 2 Volt EMF and a 2 Ohm resistor connected in series.
Second branch: a 2 Volt EMF and a 1 Ohm resistor.
Third branch: a conductor with a resistance of 1 Ohm.
Prove the currents and voltage drop are as follows:
- Current in first branch: i1 = 0.4 A
- Current in second branch: i2 = 0.8 A
- Current in third branch: i3 = 1.2 A
- Voltage between A and B: E_AB = 1.2 Volts
-/
theorem circuit_analysis :
  ∃ (i1 i2 i3 : ℝ) (E_AB : ℝ),
    (i1 = 0.4) ∧
    (i2 = 0.8) ∧
    (i3 = 1.2) ∧
    (E_AB = 1.2) ∧
    (2 = 2 * i1 + i3) ∧
    (2 = i2 + i3) ∧
    (i3 = i1 + i2) ∧
    (E_AB = i3 * 1) := sorry

end NUMINAMATH_GPT_circuit_analysis_l1097_109718


namespace NUMINAMATH_GPT_find_fourth_number_l1097_109787

theorem find_fourth_number (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)) 
  (h2 : a 6 = 42) 
  (h3 : a 8 = 110) : 
  a 3 = 10 := 
sorry

end NUMINAMATH_GPT_find_fourth_number_l1097_109787


namespace NUMINAMATH_GPT_union_eq_universal_set_l1097_109773

-- Define the sets U, M, and N
def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 6}

-- The theorem stating the desired equality
theorem union_eq_universal_set : M ∪ N = U := 
sorry

end NUMINAMATH_GPT_union_eq_universal_set_l1097_109773


namespace NUMINAMATH_GPT_minimum_value_l1097_109733

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2 * y = 3) :
  (1 / x + 1 / y) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1097_109733


namespace NUMINAMATH_GPT_domain_of_f_l1097_109714

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (4 * x - 3))

theorem domain_of_f :
  {x : ℝ | 4 * x - 3 > 0 ∧ Real.log (4 * x - 3) ≠ 0} = 
  {x : ℝ | x ∈ Set.Ioo (3 / 4) 1 ∪ Set.Ioi 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1097_109714


namespace NUMINAMATH_GPT_mike_last_5_shots_l1097_109711

theorem mike_last_5_shots :
  let initial_shots := 30
  let initial_percentage := 40 / 100
  let additional_shots_1 := 10
  let new_percentage_1 := 45 / 100
  let additional_shots_2 := 5
  let new_percentage_2 := 46 / 100
  
  let initial_makes := initial_shots * initial_percentage
  let total_shots_after_1 := initial_shots + additional_shots_1
  let makes_after_1 := total_shots_after_1 * new_percentage_1 - initial_makes
  let total_makes_after_1 := initial_makes + makes_after_1
  let total_shots_after_2 := total_shots_after_1 + additional_shots_2
  let final_makes := total_shots_after_2 * new_percentage_2
  let makes_in_last_5 := final_makes - total_makes_after_1
  
  makes_in_last_5 = 2
:=
by
  sorry

end NUMINAMATH_GPT_mike_last_5_shots_l1097_109711


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1097_109757

theorem simplify_and_evaluate
  (m : ℝ) (hm : m = 2 + Real.sqrt 2) :
  (1 - (m / (m + 2))) / ((m^2 - 4*m + 4) / (m^2 - 4)) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1097_109757


namespace NUMINAMATH_GPT_least_multiple_of_15_greater_than_520_l1097_109705

theorem least_multiple_of_15_greater_than_520 : ∃ n : ℕ, n > 520 ∧ n % 15 = 0 ∧ (∀ m : ℕ, m > 520 ∧ m % 15 = 0 → n ≤ m) ∧ n = 525 := 
by
  sorry

end NUMINAMATH_GPT_least_multiple_of_15_greater_than_520_l1097_109705


namespace NUMINAMATH_GPT_parallel_vectors_y_value_l1097_109732

theorem parallel_vectors_y_value 
  (y : ℝ) 
  (a : ℝ × ℝ := (6, 2)) 
  (b : ℝ × ℝ := (y, 3)) 
  (h : ∃ k : ℝ, b = k • a) : y = 9 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_y_value_l1097_109732


namespace NUMINAMATH_GPT_exists_irreducible_fractions_l1097_109729

theorem exists_irreducible_fractions:
  ∃ (f : Fin 2018 → ℚ), 
    (∀ i j : Fin 2018, i ≠ j → (f i).den ≠ (f j).den) ∧ 
    (∀ i j : Fin 2018, i ≠ j → ∀ d : ℚ, d = f i - f j → d ≠ 0 → d.den < (f i).den ∧ d.den < (f j).den) :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_exists_irreducible_fractions_l1097_109729
