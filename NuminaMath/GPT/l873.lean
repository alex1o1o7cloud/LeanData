import Mathlib

namespace largest_num_of_hcf_and_lcm_factors_l873_87379

theorem largest_num_of_hcf_and_lcm_factors (hcf : ℕ) (f1 f2 : ℕ) (hcf_eq : hcf = 23) (f1_eq : f1 = 13) (f2_eq : f2 = 14) : 
    hcf * max f1 f2 = 322 :=
by
  -- use the conditions to find the largest number
  rw [hcf_eq, f1_eq, f2_eq]
  sorry

end largest_num_of_hcf_and_lcm_factors_l873_87379


namespace sufficient_but_not_necessary_condition_l873_87337

-- The conditions of the problem
variables (a b : ℝ)

-- The proposition to be proved
theorem sufficient_but_not_necessary_condition (h : a + b = 1) : 4 * a * b ≤ 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l873_87337


namespace sweet_tray_GCD_l873_87398

/-!
Tim has a bag of 36 orange-flavoured sweets and Peter has a bag of 44 grape-flavoured sweets.
They have to divide up the sweets into small trays with equal number of sweets;
each tray containing either orange-flavoured or grape-flavoured sweets only.
The largest possible number of sweets in each tray without any remainder is 4.
-/

theorem sweet_tray_GCD :
  Nat.gcd 36 44 = 4 :=
by
  sorry

end sweet_tray_GCD_l873_87398


namespace exists_two_integers_with_difference_divisible_by_2022_l873_87369

theorem exists_two_integers_with_difference_divisible_by_2022 (a : Fin 2023 → ℤ) : 
  ∃ i j : Fin 2023, i ≠ j ∧ (a i - a j) % 2022 = 0 := by
  sorry

end exists_two_integers_with_difference_divisible_by_2022_l873_87369


namespace calc_expression_l873_87319

theorem calc_expression (x y z : ℚ) (h1 : x = 1 / 3) (h2 : y = 2 / 3) (h3 : z = x * y) :
  3 * x^2 * y^5 * z^3 = 768 / 1594323 :=
by
  sorry

end calc_expression_l873_87319


namespace exists_positive_integer_special_N_l873_87314

theorem exists_positive_integer_special_N : 
  ∃ (N : ℕ), 
    (∃ (m : ℕ), N = 1990 * (m + 995)) ∧ 
    (∀ (n : ℕ), (∃ (m : ℕ), 2 * N = (n + 1) * (2 * m + n)) ↔ (3980 = 2 * 1990)) := by
  sorry

end exists_positive_integer_special_N_l873_87314


namespace bridesmaids_count_l873_87376

theorem bridesmaids_count
  (hours_per_dress : ℕ)
  (hours_per_week : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (dresses : ℕ) :
  hours_per_dress = 12 →
  hours_per_week = 4 →
  weeks = 15 →
  total_hours = hours_per_week * weeks →
  dresses = total_hours / hours_per_dress →
  dresses = 5 := by
  sorry

end bridesmaids_count_l873_87376


namespace total_doctors_and_nurses_l873_87375

theorem total_doctors_and_nurses
    (ratio_doctors_nurses : ℕ -> ℕ -> Prop)
    (num_nurses : ℕ)
    (h₁ : ratio_doctors_nurses 2 3)
    (h₂ : num_nurses = 150) :
    ∃ num_doctors total_doctors_nurses, 
    (total_doctors_nurses = num_doctors + num_nurses) 
    ∧ (num_doctors / num_nurses = 2 / 3) 
    ∧ total_doctors_nurses = 250 := 
by
  sorry

end total_doctors_and_nurses_l873_87375


namespace value_of_x_l873_87311

theorem value_of_x : ∀ x : ℝ, (x^2 - 4) / (x - 2) = 0 → x ≠ 2 → x = -2 := by
  intros x h1 h2
  sorry

end value_of_x_l873_87311


namespace inscribed_sphere_volume_l873_87385

theorem inscribed_sphere_volume (edge_length : ℝ) (h_edge : edge_length = 12) : 
  ∃ (V : ℝ), V = 288 * Real.pi :=
by
  sorry

end inscribed_sphere_volume_l873_87385


namespace paper_string_area_l873_87346

theorem paper_string_area (side len overlap : ℝ) (n : ℕ) (h_side : side = 30) 
                          (h_len : len = 30) (h_overlap : overlap = 7) (h_n : n = 6) :
  let area_one_sheet := side * len
  let effective_len := side - overlap
  let total_length := len + effective_len * (n - 1)
  let width := side
  let area := total_length * width
  area = 4350 := 
by
  sorry

end paper_string_area_l873_87346


namespace hyperbola_equation_l873_87332

theorem hyperbola_equation (a b c : ℝ)
  (ha : a > 0) (hb : b > 0)
  (eccentricity : c = 2 * a)
  (distance_foci_asymptote : b = 1)
  (hyperbola_eq : c^2 = a^2 + b^2) :
  (3 * x^2 - y^2 = 1) :=
by
  sorry

end hyperbola_equation_l873_87332


namespace find_rate_percent_l873_87354

def P : ℝ := 800
def SI : ℝ := 200
def T : ℝ := 4

theorem find_rate_percent (R : ℝ) :
  SI = P * R * T / 100 → R = 6.25 :=
by
  sorry

end find_rate_percent_l873_87354


namespace resulting_solution_percentage_l873_87372

theorem resulting_solution_percentage :
  ∀ (C_init R C_replace : ℚ), 
  C_init = 0.85 → 
  R = 0.6923076923076923 → 
  C_replace = 0.2 → 
  (C_init * (1 - R) + C_replace * R) = 0.4 :=
by
  intros C_init R C_replace hC_init hR hC_replace
  -- Omitted proof here
  sorry

end resulting_solution_percentage_l873_87372


namespace Rose_has_20_crystal_beads_l873_87306

noncomputable def num_crystal_beads (metal_beads_Nancy : ℕ) (pearl_beads_more_than_metal : ℕ) (beads_per_bracelet : ℕ)
    (total_bracelets : ℕ) (stone_to_crystal_ratio : ℕ) : ℕ :=
  let pearl_beads_Nancy := metal_beads_Nancy + pearl_beads_more_than_metal
  let total_beads_Nancy := metal_beads_Nancy + pearl_beads_Nancy
  let beads_needed := beads_per_bracelet * total_bracelets
  let beads_Rose := beads_needed - total_beads_Nancy
  beads_Rose / stone_to_crystal_ratio.succ

theorem Rose_has_20_crystal_beads :
  num_crystal_beads 40 20 8 20 2 = 20 :=
by
  sorry

end Rose_has_20_crystal_beads_l873_87306


namespace planes_divide_space_l873_87387

-- Definition of a triangular prism
def triangular_prism (V : Type) (P : Set (Set V)) : Prop :=
  ∃ (A B C D E F : V),
    P = {{A, B, C}, {D, E, F}, {A, B, D, E}, {B, C, E, F}, {C, A, F, D}}

-- The condition: planes containing the faces of a triangular prism
def planes_containing_faces (V : Type) (P : Set (Set V)) : Prop :=
  triangular_prism V P

-- Proof statement: The planes containing the faces of a triangular prism divide the space into 21 parts
theorem planes_divide_space (V : Type) (P : Set (Set V))
  (h : planes_containing_faces V P) :
  ∃ parts : ℕ, parts = 21 := by
  sorry

end planes_divide_space_l873_87387


namespace jade_driving_hours_per_day_l873_87382

variable (Jade Krista : ℕ)
variable (days driving_hours total_hours : ℕ)

theorem jade_driving_hours_per_day :
  (days = 3) →
  (Krista = 6) →
  (total_hours = 42) →
  (total_hours = days * Jade + days * Krista) →
  Jade = 8 :=
by
  intros h_days h_krista h_total_hours h_equation
  sorry

end jade_driving_hours_per_day_l873_87382


namespace mutually_exclusive_not_complementary_l873_87352

-- Define the people
inductive Person
| A 
| B 
| C

open Person

-- Define the colors
inductive Color
| Red
| Yellow
| Blue

open Color

-- Event A: Person A gets the Red card
def event_a (assignment: Person → Color) : Prop := assignment A = Red

-- Event B: Person B gets the Red card
def event_b (assignment: Person → Color) : Prop := assignment B = Red

-- Definition of mutually exclusive events
def mutually_exclusive (P Q: Prop): Prop := P → ¬Q

-- Definition of complementary events
def complementary (P Q: Prop): Prop := P ↔ ¬Q

theorem mutually_exclusive_not_complementary :
  ∀ (assignment: Person → Color),
  mutually_exclusive (event_a assignment) (event_b assignment) ∧ ¬complementary (event_a assignment) (event_b assignment) :=
by
  sorry

end mutually_exclusive_not_complementary_l873_87352


namespace monthly_rent_l873_87378

theorem monthly_rent (cost : ℕ) (maintenance_percentage : ℚ) (annual_taxes : ℕ) (desired_return_rate : ℚ) (monthly_rent : ℚ) :
  cost = 20000 ∧
  maintenance_percentage = 0.10 ∧
  annual_taxes = 460 ∧
  desired_return_rate = 0.06 →
  monthly_rent = 153.70 := 
sorry

end monthly_rent_l873_87378


namespace min_value_expr_l873_87324

open Real

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 :=
by
  sorry

end min_value_expr_l873_87324


namespace determine_remaining_sides_l873_87349

variables (A B C D E : Type)

def cyclic_quadrilateral (A B C D : Type) : Prop := sorry

def known_sides (AB CD : ℝ) : Prop := AB > 0 ∧ CD > 0

def known_ratio (m n : ℝ) : Prop := m > 0 ∧ n > 0

theorem determine_remaining_sides
  {A B C D : Type}
  (h_cyclic : cyclic_quadrilateral A B C D)
  (AB CD : ℝ) (h_sides : known_sides AB CD)
  (m n : ℝ) (h_ratio : known_ratio m n) :
  ∃ (BC AD : ℝ), BC / AD = m / n ∧ BC > 0 ∧ AD > 0 :=
sorry

end determine_remaining_sides_l873_87349


namespace find_first_term_of_sequence_l873_87393

theorem find_first_term_of_sequence (a : ℕ → ℝ)
  (h_rec : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h_a8 : a 8 = 2) :
  a 1 = 1 / 2 :=
sorry

end find_first_term_of_sequence_l873_87393


namespace McKenna_stuffed_animals_count_l873_87310

def stuffed_animals (M K T : ℕ) : Prop :=
  M + K + T = 175 ∧ K = 2 * M ∧ T = K + 5

theorem McKenna_stuffed_animals_count (M K T : ℕ) (h : stuffed_animals M K T) : M = 34 :=
by
  sorry

end McKenna_stuffed_animals_count_l873_87310


namespace find_m_l873_87394

theorem find_m (m : ℕ) (h : m * (m - 1) * (m - 2) * (m - 3) * (m - 4) = 2 * m * (m - 1) * (m - 2)) : m = 5 :=
sorry

end find_m_l873_87394


namespace find_AC_l873_87320

noncomputable def isTriangle (A B C : Type) : Type := sorry

-- Define angles and lengths.
variables (A B C : Type)
variables (angle_A angle_B : ℝ)
variables (BC AC : ℝ)

-- Assume the given conditions.
axiom angle_A_60 : angle_A = 60 * Real.pi / 180
axiom angle_B_45 : angle_B = 45 * Real.pi / 180
axiom BC_12 : BC = 12

-- Statement to prove.
theorem find_AC 
  (h_triangle : isTriangle A B C)
  (h_angle_A : angle_A = 60 * Real.pi / 180)
  (h_angle_B : angle_B = 45 * Real.pi / 180)
  (h_BC : BC = 12) :
  ∃ AC : ℝ, AC = 8 * Real.sqrt 3 / 3 :=
sorry

end find_AC_l873_87320


namespace c_left_before_completion_l873_87350

def a_one_day_work : ℚ := 1 / 24
def b_one_day_work : ℚ := 1 / 30
def c_one_day_work : ℚ := 1 / 40
def total_work_completed (days : ℚ) : Prop := days = 11

theorem c_left_before_completion (days_left : ℚ) (h : total_work_completed 11) :
  (11 - days_left) * (a_one_day_work + b_one_day_work + c_one_day_work) +
  (days_left * (a_one_day_work + b_one_day_work)) = 1 :=
sorry

end c_left_before_completion_l873_87350


namespace f_n_2_l873_87351

def f (m n : ℕ) : ℝ :=
if h : m = 1 ∧ n = 1 then 1 else
if h : n > m then 0 else 
sorry -- This would be calculated based on the recursive definition

lemma f_2_2 : f 2 2 = 2 :=
sorry

theorem f_n_2 (n : ℕ) (hn : n ≥ 1) : f n 2 = 2^(n - 1) :=
sorry

end f_n_2_l873_87351


namespace jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l873_87333

theorem jia_can_formulate_quadratic :
  ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem yi_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem bing_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem ding_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

end jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l873_87333


namespace ludek_unique_stamps_l873_87344

theorem ludek_unique_stamps (K M L : ℕ) (k_m_shared k_l_shared m_l_shared : ℕ)
  (hk : K + M = 101)
  (hl : K + L = 115)
  (hm : M + L = 110)
  (k_m_shared := 5)
  (k_l_shared := 12)
  (m_l_shared := 7) :
  L - k_l_shared - m_l_shared = 43 :=
by
  sorry

end ludek_unique_stamps_l873_87344


namespace order_of_abc_l873_87338

noncomputable def a : ℝ := 2017^0
noncomputable def b : ℝ := 2015 * 2017 - 2016^2
noncomputable def c : ℝ := ((-2/3)^2016) * ((3/2)^2017)

theorem order_of_abc : b < a ∧ a < c := by
  -- proof omitted
  sorry

end order_of_abc_l873_87338


namespace four_digit_numbers_count_l873_87340

open Nat

def is_valid_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def four_diff_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

def leading_digit_not_zero (a : ℕ) : Prop :=
  a ≠ 0

def largest_digit_seven (a b c d : ℕ) : Prop :=
  a = 7 ∨ b = 7 ∨ c = 7 ∨ d = 7

theorem four_digit_numbers_count :
  ∃ n, n = 45 ∧
  ∀ (a b c d : ℕ),
    four_diff_digits a b c d ∧
    leading_digit_not_zero a ∧
    is_multiple_of_5 (a * 1000 + b * 100 + c * 10 + d) ∧
    is_multiple_of_3 (a * 1000 + b * 100 + c * 10 + d) ∧
    largest_digit_seven a b c d →
    n = 45 :=
sorry

end four_digit_numbers_count_l873_87340


namespace symmetrical_character_l873_87308

def symmetrical (char : String) : Prop :=
  -- Define a predicate symmetrical which checks if a given character
  -- is a symmetrical figure somehow. This needs to be implemented
  -- properly based on the graphical property of the character.
  sorry 

theorem symmetrical_character :
  ∀ (c : String), (c = "幸" → symmetrical c) ∧ 
                  (c = "福" → ¬ symmetrical c) ∧ 
                  (c = "惠" → ¬ symmetrical c) ∧ 
                  (c = "州" → ¬ symmetrical c) :=
by
  sorry

end symmetrical_character_l873_87308


namespace fewer_popsicle_sticks_l873_87366

theorem fewer_popsicle_sticks :
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  boys_total - girls_total = 6 := 
by
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  show boys_total - girls_total = 6
  sorry

end fewer_popsicle_sticks_l873_87366


namespace smallest_perimeter_of_triangle_with_area_sqrt3_l873_87315

open Real

-- Define an equilateral triangle with given area
def equilateral_triangle (a : ℝ) : Prop :=
  ∃ s: ℝ, s > 0 ∧ a = (sqrt 3 / 4) * s^2

-- Problem statement: Prove the smallest perimeter of such a triangle is 6.
theorem smallest_perimeter_of_triangle_with_area_sqrt3 : 
  equilateral_triangle (sqrt 3) → ∃ s: ℝ, s > 0 ∧ 3 * s = 6 :=
by 
  sorry

end smallest_perimeter_of_triangle_with_area_sqrt3_l873_87315


namespace paper_cut_count_incorrect_l873_87391

theorem paper_cut_count_incorrect (n : ℕ) (h : n = 1961) : 
  ∀ i, (∃ k, i = 7 ∨ i = 7 + 6 * k) → i % 6 = 1 → n ≠ i :=
by
  sorry

end paper_cut_count_incorrect_l873_87391


namespace probability_none_hit_l873_87355

theorem probability_none_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (1 - p)^5 = (1 - p) * (1 - p) * (1 - p) * (1 - p) * (1 - p) :=
by sorry

end probability_none_hit_l873_87355


namespace cost_of_small_bonsai_l873_87336

variable (cost_small_bonsai cost_big_bonsai : ℝ)

theorem cost_of_small_bonsai : 
  cost_big_bonsai = 20 → 
  3 * cost_small_bonsai + 5 * cost_big_bonsai = 190 → 
  cost_small_bonsai = 30 := 
by
  intros h1 h2 
  sorry

end cost_of_small_bonsai_l873_87336


namespace exists_infinite_subset_with_gcd_l873_87304

/-- A set of natural numbers where each number is a product of at most 1987 primes -/
def is_bounded_product_set (A : Set ℕ) (k : ℕ) : Prop :=
  ∀ a ∈ A, ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ a = S.prod id ∧ S.card ≤ k

/-- Prove the existence of an infinite subset and a common gcd for any pair of its elements -/
theorem exists_infinite_subset_with_gcd (A : Set ℕ) (k : ℕ) (hk : k = 1987)
  (hA : is_bounded_product_set A k) (h_inf : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Subset B A ∧ Set.Infinite B ∧ ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = b := 
sorry

end exists_infinite_subset_with_gcd_l873_87304


namespace simplify_fraction_l873_87325

variable (y b : ℚ)

theorem simplify_fraction : 
  (y+2) / 4 + (5 - 4*y + b) / 3 = (-13*y + 4*b + 26) / 12 := 
by
  sorry

end simplify_fraction_l873_87325


namespace number_of_students_from_second_department_is_17_l873_87358

noncomputable def students_selected_from_second_department 
  (total_students : ℕ)
  (num_departments : ℕ)
  (students_per_department : List (ℕ × ℕ))
  (sample_size : ℕ)
  (starting_number : ℕ) : ℕ :=
-- This function will compute the number of students selected from the second department.
sorry

theorem number_of_students_from_second_department_is_17 : 
  students_selected_from_second_department 600 3 
    [(1, 300), (301, 495), (496, 600)] 50 3 = 17 :=
-- Proof is left as an exercise.
sorry

end number_of_students_from_second_department_is_17_l873_87358


namespace math_problem_l873_87342

theorem math_problem :
  let numerator := (15^4 + 400) * (30^4 + 400) * (45^4 + 400) * (60^4 + 400) * (75^4 + 400)
  let denominator := (5^4 + 400) * (20^4 + 400) * (35^4 + 400) * (50^4 + 400) * (65^4 + 400)
  numerator / denominator = 301 :=
by 
  sorry

end math_problem_l873_87342


namespace part1_part2_l873_87318

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (2 - a) * x + a

-- Part 1
theorem part1 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem part2 (a x : ℝ) : 
  (f a x < a + 2) ↔ 
    (a = 0 ∧ x < 1) ∨ 
    (a > 0 ∧ -2 / a < x ∧ x < 1) ∨ 
    (-2 < a ∧ a < 0 ∧ (x < 1 ∨ x > -2 / a)) ∨ 
    (a = -2) ∨ 
    (a < -2 ∧ (x < -2 / a ∨ x > 1)) := sorry

end part1_part2_l873_87318


namespace female_managers_count_l873_87392

-- Definitions for the problem statement

def total_female_employees : ℕ := 500
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Problem parameters
variable (E M FM : ℕ) -- E: total employees, M: male employees, FM: female managers

-- Conditions
def total_employees_eq : Prop := E = M + total_female_employees
def total_managers_eq : Prop := fraction_of_managers * E = fraction_of_male_managers * M + FM

-- The statement we want to prove
theorem female_managers_count (h1 : total_employees_eq E M) (h2 : total_managers_eq E M FM) : FM = 200 :=
by
  -- to be proven
  sorry

end female_managers_count_l873_87392


namespace quadratic_function_expression_rational_function_expression_l873_87356

-- Problem 1:
theorem quadratic_function_expression (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 3 * x) ∧ (f 0 = 1) → (∀ x, f x = (3 / 2) * x^2 - (3 / 2) * x + 1) :=
by
  sorry

-- Problem 2:
theorem rational_function_expression (f : ℝ → ℝ) : 
  (∀ x, x ≠ 0 → 3 * f (1 / x) + f x = x) → 
  (∀ x, x ≠ 0 → f x = 3 / (8 * x) - x / 8) :=
by
  sorry

end quadratic_function_expression_rational_function_expression_l873_87356


namespace real_life_distance_between_cities_l873_87302

variable (map_distance : ℕ)
variable (scale : ℕ)

theorem real_life_distance_between_cities (h1 : map_distance = 45) (h2 : scale = 10) :
  map_distance * scale = 450 :=
sorry

end real_life_distance_between_cities_l873_87302


namespace sum_of_squares_correct_l873_87348

-- Define the three incorrect entries
def incorrect_entry_1 : Nat := 52
def incorrect_entry_2 : Nat := 81
def incorrect_entry_3 : Nat := 111

-- Define the sum of the squares of these entries
def sum_of_squares : Nat := incorrect_entry_1 ^ 2 + incorrect_entry_2 ^ 2 + incorrect_entry_3 ^ 2

-- State that this sum of squares equals 21586
theorem sum_of_squares_correct : sum_of_squares = 21586 := by
  sorry

end sum_of_squares_correct_l873_87348


namespace max_value_x_minus_y_l873_87399

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l873_87399


namespace incorrect_operation_l873_87301

variable (a : ℕ)

-- Conditions
def condition1 := 4 * a ^ 2 - a ^ 2 = 3 * a ^ 2
def condition2 := a ^ 3 * a ^ 6 = a ^ 9
def condition3 := (a ^ 2) ^ 3 = a ^ 5
def condition4 := (2 * a ^ 2) ^ 2 = 4 * a ^ 4

-- Theorem to prove
theorem incorrect_operation : (a ^ 2) ^ 3 ≠ a ^ 5 := 
by
  sorry

end incorrect_operation_l873_87301


namespace perp_lines_value_of_m_parallel_lines_value_of_m_l873_87343

theorem perp_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) * ((m - 2) / 3) = -1)) → 
  m = 1 / 2 := 
sorry

theorem parallel_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) = ((m - 2) / 3))) → 
  m = -1 := 
sorry

end perp_lines_value_of_m_parallel_lines_value_of_m_l873_87343


namespace only_solutions_mod_n_l873_87345

theorem only_solutions_mod_n (n : ℕ) : (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % (n : ℤ) = 0) ↔ (∃ k : ℕ, n = 3 ^ k) := 
sorry

end only_solutions_mod_n_l873_87345


namespace blue_faces_ratio_l873_87362

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l873_87362


namespace high_card_point_value_l873_87322

theorem high_card_point_value :
  ∀ (H L : ℕ), 
  (L = 1) →
  ∀ (high low total_points : ℕ), 
  (total_points = 5) →
  (high + (L + L + L) = total_points) →
  high = 2 :=
by
  intros
  sorry

end high_card_point_value_l873_87322


namespace calculate_result_l873_87371

theorem calculate_result : (-3 : ℝ)^(2022) * (1 / 3 : ℝ)^(2023) = 1 / 3 := 
by sorry

end calculate_result_l873_87371


namespace area_of_EFGH_l873_87316

def shorter_side := 6
def ratio := 2
def longer_side := shorter_side * ratio
def width := 2 * longer_side
def length := shorter_side

theorem area_of_EFGH : length * width = 144 := by
  sorry

end area_of_EFGH_l873_87316


namespace unique_n_l873_87317

theorem unique_n (n : ℕ) (h_pos : 0 < n) :
  (∀ x y : ℕ, (xy + 1) % n = 0 → (x + y) % n = 0) ↔ n = 2 :=
by
  sorry

end unique_n_l873_87317


namespace eliminate_x3_term_l873_87339

noncomputable def polynomial (n : ℝ) : Polynomial ℝ :=
  (Polynomial.X ^ 2 + Polynomial.C n * Polynomial.X + Polynomial.C 3) *
  (Polynomial.X ^ 2 - Polynomial.C 3 * Polynomial.X)

theorem eliminate_x3_term (n : ℝ) : (polynomial n).coeff 3 = 0 ↔ n = 3 :=
by
  -- sorry to skip the proof for now as it's not required
  sorry

end eliminate_x3_term_l873_87339


namespace solve_for_x_l873_87313

theorem solve_for_x (x : ℕ) (h1 : x > 0) (h2 : x % 6 = 0) (h3 : x^2 > 144) (h4 : x < 30) : x = 18 ∨ x = 24 :=
by
  sorry

end solve_for_x_l873_87313


namespace convert_to_polar_coordinates_l873_87386

open Real

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * π - arctan (abs y / abs x) else arctan (abs y / abs x)
  (r, θ)

theorem convert_to_polar_coordinates : 
  polar_coordinates 3 (-3) = (3 * sqrt 2, 7 * π / 4) :=
by
  sorry

end convert_to_polar_coordinates_l873_87386


namespace collinear_vectors_l873_87341

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (OA OB OP : V) (m n : ℝ)

-- Given conditions
def non_collinear (OA OB : V) : Prop :=
  ∀ (t : ℝ), OA ≠ t • OB

def collinear_points (P A B : V) : Prop :=
  ∃ (t : ℝ), P - A = t • (B - A)

def linear_combination (OP OA OB : V) (m n : ℝ) : Prop :=
  OP = m • OA + n • OB

-- The theorem statement
theorem collinear_vectors (noncol : non_collinear OA OB)
  (collinearPAB : collinear_points OP OA OB)
  (lin_comb : linear_combination OP OA OB m n) :
  m = 2 ∧ n = -1 := by
sorry

end collinear_vectors_l873_87341


namespace evaluate_three_star_twostar_one_l873_87365

def operator_star (a b : ℕ) : ℕ :=
  a^b - b^a

theorem evaluate_three_star_twostar_one : operator_star 3 (operator_star 2 1) = 2 := 
  by
    sorry

end evaluate_three_star_twostar_one_l873_87365


namespace ratio_b_a_l873_87326

theorem ratio_b_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a ≠ b) (h4 : a + b > 2 * a) (h5 : 2 * a > a) 
  (h6 : a + b > b) (h7 : a + 2 * a = b) : 
  b = a * Real.sqrt 2 :=
by
  sorry

end ratio_b_a_l873_87326


namespace parallel_lines_m_l873_87330

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 3 * m * x + (m + 2) * y + 1 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + (m + 2) * y + 2 = 0) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (3 * m) / (m + 2) = (m - 2) / (m + 2)) →
  (m = -1 ∨ m = -2) :=
sorry

end parallel_lines_m_l873_87330


namespace simplify_and_rationalize_l873_87300

theorem simplify_and_rationalize : (1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5) :=
by sorry

end simplify_and_rationalize_l873_87300


namespace wire_length_before_cut_l873_87347

theorem wire_length_before_cut (S : ℝ) (L : ℝ) (h1 : S = 4) (h2 : S = (2/5) * L) : S + L = 14 :=
by 
  sorry

end wire_length_before_cut_l873_87347


namespace final_withdrawal_amount_july_2005_l873_87321

-- Define the conditions given in the problem
variables (a r : ℝ) (n : ℕ)

-- Define the recursive formula for deposits
def deposit_amount (n : ℕ) : ℝ :=
  if n = 0 then a else (deposit_amount (n - 1)) * (1 + r) + a

-- The problem statement translated to Lean
theorem final_withdrawal_amount_july_2005 :
  deposit_amount a r 5 = a / r * ((1 + r) ^ 6 - (1 + r)) :=
sorry

end final_withdrawal_amount_july_2005_l873_87321


namespace necessary_but_not_sufficient_condition_l873_87380

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ( (2*x - 1)*x = 0 → x = 0 ) ∧ ( x = 0 → (2*x - 1)*x = 0 ) :=
by
  sorry

end necessary_but_not_sufficient_condition_l873_87380


namespace domain_of_f_l873_87390

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (Real.sqrt (x - 7))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = x} = Set.Ioi 7 := by
  sorry

end domain_of_f_l873_87390


namespace quotient_is_20_l873_87397

theorem quotient_is_20 (D d r Q : ℕ) (hD : D = 725) (hd : d = 36) (hr : r = 5) (h : D = d * Q + r) :
  Q = 20 :=
by sorry

end quotient_is_20_l873_87397


namespace symmetric_angle_of_inclination_l873_87331

theorem symmetric_angle_of_inclination (α₁ : ℝ) (h : 0 ≤ α₁ ∧ α₁ < π) : 
  (∃ β₁ : ℝ, (α₁ = 0 ∧ β₁ = 0) ∨ (0 < α₁ ∧ α₁ < π ∧ β₁ = π - α₁)) :=
by
  sorry

end symmetric_angle_of_inclination_l873_87331


namespace abs_function_le_two_l873_87312

theorem abs_function_le_two {x : ℝ} (h : |x| ≤ 2) : |3 * x - x^3| ≤ 2 :=
sorry

end abs_function_le_two_l873_87312


namespace minimal_functions_l873_87323

open Int

theorem minimal_functions (f : ℤ → ℤ) (c : ℤ) :
  (∀ x, f (x + 2017) = f x) ∧
  (∀ x y, (f (f x + f y + 1) - f (f x + f y)) % 2017 = c) →
  (c = 1 ∨ c = 2016 ∨ c = 1008 ∨ c = 1009) :=
by
  sorry

end minimal_functions_l873_87323


namespace cube_surface_area_l873_87303

theorem cube_surface_area (V : ℝ) (hV : V = 64) : ∃ S : ℝ, S = 96 := 
by
  sorry

end cube_surface_area_l873_87303


namespace cuboid_surface_area_l873_87309

noncomputable def total_surface_area (x y z : ℝ) : ℝ :=
  2 * (x * y + y * z + z * x)

theorem cuboid_surface_area (x y z : ℝ) (h1 : x + y + z = 40) (h2 : x^2 + y^2 + z^2 = 625) :
  total_surface_area x y z = 975 :=
sorry

end cuboid_surface_area_l873_87309


namespace total_time_l873_87374

def time_to_eat_cereal (rate1 rate2 rate3 : ℚ) (amount : ℚ) : ℚ :=
  let combined_rate := rate1 + rate2 + rate3
  amount / combined_rate

theorem total_time (rate1 rate2 rate3 : ℚ) (amount : ℚ) 
  (h1 : rate1 = 1 / 15)
  (h2 : rate2 = 1 / 20)
  (h3 : rate3 = 1 / 30)
  (h4 : amount = 4) : 
  time_to_eat_cereal rate1 rate2 rate3 amount = 80 / 3 := 
by 
  rw [time_to_eat_cereal, h1, h2, h3, h4]
  sorry

end total_time_l873_87374


namespace sum_of_integers_from_neg15_to_5_l873_87335

-- defining the conditions
def first_term : ℤ := -15
def last_term : ℤ := 5

-- sum of integers from first_term to last_term
def sum_arithmetic_series (a l : ℤ) : ℤ :=
  let n := l - a + 1
  (n * (a + l)) / 2

-- the statement we need to prove
theorem sum_of_integers_from_neg15_to_5 : sum_arithmetic_series first_term last_term = -105 := by
  sorry

end sum_of_integers_from_neg15_to_5_l873_87335


namespace find_printer_price_l873_87334

variable (C P M : ℝ)

theorem find_printer_price
  (h1 : C + P + M = 3000)
  (h2 : P = (1/4) * (C + P + M + 800)) :
  P = 950 :=
sorry

end find_printer_price_l873_87334


namespace Carly_fourth_week_running_distance_l873_87370

theorem Carly_fourth_week_running_distance :
  let week1_distance_per_day := 2
  let week2_distance_per_day := (week1_distance_per_day * 2) + 3
  let week3_distance_per_day := week2_distance_per_day * (9 / 7)
  let week4_intended_distance_per_day := week3_distance_per_day * 0.9
  let week4_actual_distance_per_day := week4_intended_distance_per_day * 0.5
  let week4_days_run := 5 -- due to 2 rest days
  (week4_actual_distance_per_day * week4_days_run) = 20.25 := 
by 
    -- We use sorry here to skip the proof
    sorry

end Carly_fourth_week_running_distance_l873_87370


namespace complement_union_complement_intersection_complementA_intersect_B_l873_87396

def setA (x : ℝ) : Prop := 3 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 2 < x ∧ x < 10

theorem complement_union (x : ℝ) : ¬(setA x ∨ setB x) ↔ x ≤ 2 ∨ x ≥ 10 := sorry

theorem complement_intersection (x : ℝ) : ¬(setA x ∧ setB x) ↔ x < 3 ∨ x ≥ 7 := sorry

theorem complementA_intersect_B (x : ℝ) : (¬setA x ∧ setB x) ↔ (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) := sorry

end complement_union_complement_intersection_complementA_intersect_B_l873_87396


namespace solution_of_az_eq_b_l873_87307

theorem solution_of_az_eq_b (a b z x y : ℝ) :
  (∃! x, 4 + 3 * a * x = 2 * a - 7) →
  (¬ ∃ y, 2 + y = (b + 1) * y) →
  az = b →
  z = 0 :=
by
  intros h1 h2 h3
  -- proof starts here
  sorry

end solution_of_az_eq_b_l873_87307


namespace sum_of_first_six_primes_mod_seventh_prime_l873_87364

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l873_87364


namespace min_value_one_over_x_plus_one_over_y_l873_87373

theorem min_value_one_over_x_plus_one_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) : 
  (1 / x + 1 / y) ≥ 1 :=
by
  sorry -- Proof goes here

end min_value_one_over_x_plus_one_over_y_l873_87373


namespace fraction_value_l873_87383

theorem fraction_value (m n : ℤ) (h : (m - 8) * (m - 8) + abs (n + 6) = 0) : n / m = -(3 / 4) :=
by sorry

end fraction_value_l873_87383


namespace min_value_expression_l873_87368

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b - a - 2 * b = 0) :
  ∃ p : ℝ, p = (a^2/4 - 2/a + b^2 - 1/b) ∧ p = 7 :=
by sorry

end min_value_expression_l873_87368


namespace uncle_jerry_total_tomatoes_l873_87388

def day1_tomatoes : ℕ := 120
def day2_tomatoes : ℕ := day1_tomatoes + 50
def day3_tomatoes : ℕ := 2 * day2_tomatoes
def total_tomatoes : ℕ := day1_tomatoes + day2_tomatoes + day3_tomatoes

theorem uncle_jerry_total_tomatoes : total_tomatoes = 630 := by
  sorry

end uncle_jerry_total_tomatoes_l873_87388


namespace mrs_smith_class_boys_girls_ratio_l873_87389

theorem mrs_smith_class_boys_girls_ratio (total_students boys girls : ℕ) (h1 : boys / girls = 3 / 4) (h2 : boys + girls = 42) : girls = boys + 6 :=
by
  sorry

end mrs_smith_class_boys_girls_ratio_l873_87389


namespace min_quadratic_expr_l873_87353

noncomputable def quadratic_expr (x : ℝ) := x^2 + 10 * x + 3

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = -22 :=
by
  use -5
  simp [quadratic_expr]
  sorry

end min_quadratic_expr_l873_87353


namespace obtuse_triangle_contradiction_l873_87357

theorem obtuse_triangle_contradiction (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) : 
  (A > 90 ∧ B > 90) → false :=
by
  sorry

end obtuse_triangle_contradiction_l873_87357


namespace least_faces_combined_l873_87327

theorem least_faces_combined (a b : ℕ) (h1 : a ≥ 6) (h2 : b ≥ 6)
  (h3 : (∃ k : ℕ, k * a * b = 20) → (∃ m : ℕ, 2 * m = 10 * (k + 10))) 
  (h4 : (∃ n : ℕ, n = (a * b) / 10)) (h5 : ∃ l : ℕ, l = 5) : a + b = 20 :=
by
  sorry

end least_faces_combined_l873_87327


namespace std_deviation_calc_l873_87329

theorem std_deviation_calc 
  (μ : ℝ) (σ : ℝ) (V : ℝ) (k : ℝ)
  (hμ : μ = 14.0)
  (hσ : σ = 1.5)
  (hV : V = 11)
  (hk : k = (μ - V) / σ) :
  k = 2 := by
  sorry

end std_deviation_calc_l873_87329


namespace compare_M_N_l873_87305

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 7
def N : ℝ := (a - 2) * (a - 3)

theorem compare_M_N : M a > N a :=
by
  sorry

end compare_M_N_l873_87305


namespace petya_friends_l873_87381

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l873_87381


namespace alexis_suit_coat_expense_l873_87384

theorem alexis_suit_coat_expense :
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  budget - leftover - other_expenses = 38 := 
by
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  sorry

end alexis_suit_coat_expense_l873_87384


namespace files_remaining_on_flash_drive_l873_87377

def initial_music_files : ℕ := 32
def initial_video_files : ℕ := 96
def deleted_files : ℕ := 60

def total_initial_files : ℕ := initial_music_files + initial_video_files

theorem files_remaining_on_flash_drive 
  (h : total_initial_files = 128) : (total_initial_files - deleted_files) = 68 := by
  sorry

end files_remaining_on_flash_drive_l873_87377


namespace total_spent_l873_87360

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end total_spent_l873_87360


namespace max_integer_value_l873_87363

theorem max_integer_value (x : ℝ) : 
  ∃ M : ℤ, ∀ y : ℝ, (M = ⌊ 1 + 10 / (4 * y^2 + 12 * y + 9) ⌋ ∧ M ≤ 11) := 
sorry

end max_integer_value_l873_87363


namespace find_subtracted_number_l873_87395

variable (initial_number : Real)
variable (sum : Real := initial_number + 5)
variable (product : Real := sum * 7)
variable (quotient : Real := product / 5)
variable (remainder : Real := 33)

theorem find_subtracted_number 
  (initial_number_eq : initial_number = 22.142857142857142)
  : quotient - remainder = 5 := by
  sorry

end find_subtracted_number_l873_87395


namespace emily_necklaces_l873_87361

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (necklaces_made : ℕ) 
  (h1 : total_beads = 52)
  (h2 : beads_per_necklace = 2)
  (h3 : necklaces_made = total_beads / beads_per_necklace) :
  necklaces_made = 26 :=
by
  rw [h1, h2] at h3
  exact h3

end emily_necklaces_l873_87361


namespace fixed_point_of_function_l873_87367

theorem fixed_point_of_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ x y : ℝ, y = a^(x-1) + 1 ∧ (x, y) = (1, 2) :=
by 
  sorry

end fixed_point_of_function_l873_87367


namespace find_multiple_l873_87359

-- Definitions based on the problem's conditions
def n_drunk_drivers : ℕ := 6
def total_students : ℕ := 45
def num_speeders (M : ℕ) : ℕ := M * n_drunk_drivers - 3

-- The theorem that we need to prove
theorem find_multiple (M : ℕ) (h1: total_students = n_drunk_drivers + num_speeders M) : M = 7 :=
by
  sorry

end find_multiple_l873_87359


namespace hypotenuse_length_l873_87328

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l873_87328
