import Mathlib

namespace NUMINAMATH_GPT_average_of_ratios_l1943_194369

theorem average_of_ratios (a b c : ℕ) (h1 : 2 * b = 3 * a) (h2 : 3 * c = 4 * a) (h3 : a = 28) : (a + b + c) / 3 = 42 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_average_of_ratios_l1943_194369


namespace NUMINAMATH_GPT_gcd_problem_l1943_194322

def a : ℕ := 101^5 + 1
def b : ℕ := 101^5 + 101^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_problem_l1943_194322


namespace NUMINAMATH_GPT_complex_expression_l1943_194338

theorem complex_expression (i : ℂ) (h₁ : i^2 = -1) (h₂ : i^4 = 1) :
  (i + i^3)^100 + (i + i^2 + i^3 + i^4 + i^5)^120 = 1 := by
  sorry

end NUMINAMATH_GPT_complex_expression_l1943_194338


namespace NUMINAMATH_GPT_unit_prices_max_colored_tiles_l1943_194337

-- Define the given conditions
def condition1 (x y : ℝ) := 40 * x + 60 * y = 5600
def condition2 (x y : ℝ) := 50 * x + 50 * y = 6000

-- Prove the solution for part 1
theorem unit_prices (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 40 := 
sorry

-- Define the condition for the kitchen tiles
def condition3 (a : ℝ) := 80 * a + 40 * (60 - a) ≤ 3400

-- Prove the maximum number of colored tiles for the kitchen
theorem max_colored_tiles (a : ℝ) (h3 : condition3 a) :
  a ≤ 25 := 
sorry

end NUMINAMATH_GPT_unit_prices_max_colored_tiles_l1943_194337


namespace NUMINAMATH_GPT_max_electronic_thermometers_l1943_194377

theorem max_electronic_thermometers :
  ∀ (x : ℕ), 10 * x + 3 * (53 - x) ≤ 300 → x ≤ 20 :=
by
  sorry

end NUMINAMATH_GPT_max_electronic_thermometers_l1943_194377


namespace NUMINAMATH_GPT_number_of_people_in_group_l1943_194343

-- Definitions and conditions
def total_cost : ℕ := 94
def mango_juice_cost : ℕ := 5
def pineapple_juice_cost : ℕ := 6
def pineapple_cost_total : ℕ := 54

-- Theorem statement to prove
theorem number_of_people_in_group : 
  ∃ M P : ℕ, 
    mango_juice_cost * M + pineapple_juice_cost * P = total_cost ∧ 
    pineapple_juice_cost * P = pineapple_cost_total ∧ 
    M + P = 17 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_people_in_group_l1943_194343


namespace NUMINAMATH_GPT_combined_average_pieces_lost_l1943_194345

theorem combined_average_pieces_lost
  (audrey_losses : List ℕ) (thomas_losses : List ℕ)
  (h_audrey : audrey_losses = [6, 8, 4, 7, 10])
  (h_thomas : thomas_losses = [5, 6, 3, 7, 11]) :
  (audrey_losses.sum + thomas_losses.sum : ℚ) / 5 = 13.4 := by 
  sorry

end NUMINAMATH_GPT_combined_average_pieces_lost_l1943_194345


namespace NUMINAMATH_GPT_sequence_term_divisible_by_n_l1943_194359

theorem sequence_term_divisible_by_n (n : ℕ) (hn1 : 1 < n) (hn_odd : n % 2 = 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ n ∣ (2^k - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_divisible_by_n_l1943_194359


namespace NUMINAMATH_GPT_tan_theta_value_l1943_194340

open Real

theorem tan_theta_value
  (theta : ℝ)
  (h_quad : 3 * pi / 2 < theta ∧ theta < 2 * pi)
  (h_sin : sin theta = -sqrt 6 / 3) :
  tan theta = -sqrt 2 := by
  sorry

end NUMINAMATH_GPT_tan_theta_value_l1943_194340


namespace NUMINAMATH_GPT_solve_equation_l1943_194339

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1943_194339


namespace NUMINAMATH_GPT_intersection_of_sets_l1943_194366

def set_A (x : ℝ) := x + 1 ≤ 3
def set_B (x : ℝ) := 4 - x^2 ≤ 0

theorem intersection_of_sets : {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | x ≤ -2} ∪ {2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1943_194366


namespace NUMINAMATH_GPT_arvin_first_day_km_l1943_194372

theorem arvin_first_day_km :
  ∀ (x : ℕ), (∀ i : ℕ, (i < 5 → (i + x) < 6) → (x + 4 = 6)) → x = 2 :=
by sorry

end NUMINAMATH_GPT_arvin_first_day_km_l1943_194372


namespace NUMINAMATH_GPT_circle_tangent_x_axis_at_origin_l1943_194349

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0) ↔ (D = 0 ∧ F = 0 ∧ E ≠ 0) :=
sorry

end NUMINAMATH_GPT_circle_tangent_x_axis_at_origin_l1943_194349


namespace NUMINAMATH_GPT_ratio_of_areas_l1943_194390

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1943_194390


namespace NUMINAMATH_GPT_range_of_a_l1943_194379

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) (q : 0 < 2 * a - 1 ∧ 2 * a - 1 < 1) : 
  (1 / 2) < a ∧ a ≤ (2 / 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1943_194379


namespace NUMINAMATH_GPT_spongebob_earnings_l1943_194358

-- Define the conditions as variables and constants
def burgers_sold : ℕ := 30
def price_per_burger : ℝ := 2
def fries_sold : ℕ := 12
def price_per_fries : ℝ := 1.5

-- Define total earnings calculation
def earnings_from_burgers := burgers_sold * price_per_burger
def earnings_from_fries := fries_sold * price_per_fries
def total_earnings := earnings_from_burgers + earnings_from_fries

-- State the theorem we need to prove
theorem spongebob_earnings :
  total_earnings = 78 := by
    sorry

end NUMINAMATH_GPT_spongebob_earnings_l1943_194358


namespace NUMINAMATH_GPT_max_f_l1943_194395

noncomputable def S_n (n : ℕ) : ℚ :=
  n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℚ :=
  S_n n / ((n + 32) * S_n (n + 1))

theorem max_f (n : ℕ) : f n ≤ 1 / 50 := sorry

-- Verify the bound is achieved for n = 8
example : f 8 = 1 / 50 := by
  unfold f S_n
  norm_num

end NUMINAMATH_GPT_max_f_l1943_194395


namespace NUMINAMATH_GPT_number_of_children_l1943_194357

-- Define conditions
variable (A C : ℕ) (h1 : A + C = 280) (h2 : 60 * A + 25 * C = 14000)

-- Lean statement to prove the number of children
theorem number_of_children : C = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l1943_194357


namespace NUMINAMATH_GPT_g_f_g_1_equals_82_l1943_194364

def f (x : ℤ) : ℤ := 2 * x + 2
def g (x : ℤ) : ℤ := 5 * x + 2
def x : ℤ := 1

theorem g_f_g_1_equals_82 : g (f (g x)) = 82 := by
  sorry

end NUMINAMATH_GPT_g_f_g_1_equals_82_l1943_194364


namespace NUMINAMATH_GPT_valid_license_plates_count_l1943_194330

def num_valid_license_plates := (26 ^ 3) * (10 ^ 4)

theorem valid_license_plates_count : num_valid_license_plates = 175760000 :=
by
  sorry

end NUMINAMATH_GPT_valid_license_plates_count_l1943_194330


namespace NUMINAMATH_GPT_mike_washed_cars_l1943_194355

theorem mike_washed_cars 
    (total_work_time : ℕ := 4 * 60) 
    (wash_time : ℕ := 10)
    (oil_change_time : ℕ := 15) 
    (tire_change_time : ℕ := 30) 
    (num_oil_changes : ℕ := 6) 
    (num_tire_changes : ℕ := 2) 
    (remaining_time : ℕ := total_work_time - (num_oil_changes * oil_change_time + num_tire_changes * tire_change_time))
    (num_cars_washed : ℕ := remaining_time / wash_time) :
    num_cars_washed = 9 := by
  sorry

end NUMINAMATH_GPT_mike_washed_cars_l1943_194355


namespace NUMINAMATH_GPT_find_n_values_l1943_194319

theorem find_n_values : {n : ℕ | n ≥ 1 ∧ n ≤ 6 ∧ ∃ a b c : ℤ, a^n + b^n = c^n + n} = {1, 2, 3} :=
by sorry

end NUMINAMATH_GPT_find_n_values_l1943_194319


namespace NUMINAMATH_GPT_addition_result_l1943_194365

theorem addition_result : 148 + 32 + 18 + 2 = 200 :=
by
  sorry

end NUMINAMATH_GPT_addition_result_l1943_194365


namespace NUMINAMATH_GPT_ratio_pat_mark_l1943_194382

-- Conditions (as definitions)
variables (K P M : ℕ)
variables (h1 : P = 2 * K)  -- Pat charged twice as much time as Kate
variables (h2 : M = K + 80) -- Mark charged 80 more hours than Kate
variables (h3 : K + P + M = 144) -- Total hours charged is 144

theorem ratio_pat_mark (h1 : P = 2 * K) (h2 : M = K + 80) (h3 : K + P + M = 144) : 
  P / M = 1 / 3 :=
by
  sorry -- to be proved

end NUMINAMATH_GPT_ratio_pat_mark_l1943_194382


namespace NUMINAMATH_GPT_polyhedron_volume_l1943_194317

/-- Each 12 cm × 12 cm square is cut into two right-angled isosceles triangles by joining the midpoints of two adjacent sides. 
    These six triangles are attached to a regular hexagon to form a polyhedron.
    Prove that the volume of the resulting polyhedron is 864 cubic cm. -/
theorem polyhedron_volume :
  let s : ℝ := 12
  let volume_of_cube := s^3
  let volume_of_polyhedron := volume_of_cube / 2
  volume_of_polyhedron = 864 := 
by
  sorry

end NUMINAMATH_GPT_polyhedron_volume_l1943_194317


namespace NUMINAMATH_GPT_temperature_conversion_correct_l1943_194360

noncomputable def f_to_c (T : ℝ) : ℝ := (T - 32) * (5 / 9)

theorem temperature_conversion_correct :
  f_to_c 104 = 40 :=
by
  sorry

end NUMINAMATH_GPT_temperature_conversion_correct_l1943_194360


namespace NUMINAMATH_GPT_div_difference_l1943_194310

theorem div_difference {a b n : ℕ} (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) (h : n ∣ a^n - b^n) :
  n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end NUMINAMATH_GPT_div_difference_l1943_194310


namespace NUMINAMATH_GPT_president_and_committee_combination_l1943_194347

theorem president_and_committee_combination (n : ℕ) (k : ℕ) (total : ℕ) :
  n = 10 ∧ k = 3 ∧ total = (10 * Nat.choose 9 3) → total = 840 :=
by
  intros
  sorry

end NUMINAMATH_GPT_president_and_committee_combination_l1943_194347


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1943_194380

theorem necessary_but_not_sufficient (a b : ℝ) : (a > b) → (a + 1 > b - 2) :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1943_194380


namespace NUMINAMATH_GPT_john_height_in_feet_l1943_194353

theorem john_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (months : ℕ) (inches_per_foot : ℕ) :
  initial_height = 66 → growth_rate = 2 → months = 3 → inches_per_foot = 12 → 
  (initial_height + growth_rate * months) / inches_per_foot = 6 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_john_height_in_feet_l1943_194353


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1943_194350

variable {α : Type*} (A B : Set α)

theorem sufficient_but_not_necessary_condition (h₁ : A ∩ B = A) (h₂ : A ≠ B) :
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1943_194350


namespace NUMINAMATH_GPT_circle_radius_integer_l1943_194370

theorem circle_radius_integer (r : ℤ)
  (center : ℝ × ℝ)
  (inside_point : ℝ × ℝ)
  (outside_point : ℝ × ℝ)
  (h1 : center = (-2, -3))
  (h2 : inside_point = (-2, 2))
  (h3 : outside_point = (5, -3))
  (h4 : (dist center inside_point : ℝ) < r)
  (h5 : (dist center outside_point : ℝ) > r) 
  : r = 6 :=
sorry

end NUMINAMATH_GPT_circle_radius_integer_l1943_194370


namespace NUMINAMATH_GPT_rabbit_weight_l1943_194399

variable (k r p : ℝ)

theorem rabbit_weight :
  k + r + p = 39 →
  r + p = 3 * k →
  r + k = 1.5 * p →
  r = 13.65 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_rabbit_weight_l1943_194399


namespace NUMINAMATH_GPT_total_amount_paid_l1943_194323

theorem total_amount_paid :
  let pizzas := 3
  let cost_per_pizza := 8
  let total_cost := pizzas * cost_per_pizza
  total_cost = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1943_194323


namespace NUMINAMATH_GPT_no_intersection_of_graphs_l1943_194312

theorem no_intersection_of_graphs :
  ∃ x y : ℝ, y = |3 * x + 6| ∧ y = -|4 * x - 3| → false := by
  sorry

end NUMINAMATH_GPT_no_intersection_of_graphs_l1943_194312


namespace NUMINAMATH_GPT_trebled_principal_after_5_years_l1943_194328

theorem trebled_principal_after_5_years 
(P R : ℝ) (T total_interest : ℝ) (n : ℝ) 
(h1 : T = 10) 
(h2 : total_interest = 800) 
(h3 : (P * R * 10) / 100 = 400) 
(h4 : (P * R * n) / 100 + (3 * P * R * (10 - n)) / 100 = 800) :
n = 5 :=
by
-- The Lean proof will go here
sorry

end NUMINAMATH_GPT_trebled_principal_after_5_years_l1943_194328


namespace NUMINAMATH_GPT_part_II_l1943_194346

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 + (a - 1) * x - Real.log x

theorem part_II (a : ℝ) (h : a > 0) :
  ∀ x > 0, f a x ≥ 2 - (3 / (2 * a)) :=
sorry

end NUMINAMATH_GPT_part_II_l1943_194346


namespace NUMINAMATH_GPT_turnover_threshold_l1943_194397

-- Definitions based on the problem conditions
def valid_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def daily_turnover (x : ℝ) : ℝ := 20 * (10 - x) * (50 + 8 * x)

-- Lean 4 statement equivalent to mathematical proof problem
theorem turnover_threshold (x : ℝ) (hx : valid_domain x) (h_turnover : daily_turnover x ≥ 10260) :
  x ≥ 1 / 2 ∧ x ≤ 2 :=
sorry

end NUMINAMATH_GPT_turnover_threshold_l1943_194397


namespace NUMINAMATH_GPT_discount_rate_for_1000_min_price_for_1_3_discount_l1943_194341

def discounted_price (original_price : ℕ) : ℕ := 
  original_price * 80 / 100

def voucher_amount (discounted_price : ℕ) : ℕ :=
  if discounted_price < 400 then 30
  else if discounted_price < 500 then 60
  else if discounted_price < 700 then 100
  else if discounted_price < 900 then 130
  else 0 -- Can extend the rule as needed

def discount_rate (original_price : ℕ) : ℚ := 
  let total_discount := original_price * 20 / 100 + voucher_amount (discounted_price original_price)
  (total_discount : ℚ) / (original_price : ℚ)

theorem discount_rate_for_1000 : 
  discount_rate 1000 = 0.33 := 
by
  sorry

theorem min_price_for_1_3_discount :
  ∀ (x : ℕ), 500 ≤ x ∧ x ≤ 800 → 0.33 ≤ discount_rate x ↔ (625 ≤ x ∧ x ≤ 750) :=
by
  sorry

end NUMINAMATH_GPT_discount_rate_for_1000_min_price_for_1_3_discount_l1943_194341


namespace NUMINAMATH_GPT_total_treats_l1943_194307

theorem total_treats (children : ℕ) (hours : ℕ) (houses_per_hour : ℕ) (treats_per_house_per_kid : ℕ) :
  children = 3 → hours = 4 → houses_per_hour = 5 → treats_per_house_per_kid = 3 → 
  (children * hours * houses_per_hour * treats_per_house_per_kid) = 180 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_treats_l1943_194307


namespace NUMINAMATH_GPT_fraction_satisfactory_is_two_thirds_l1943_194371

-- Total number of students with satisfactory grades
def satisfactory_grades : ℕ := 3 + 7 + 4 + 2

-- Total number of students with unsatisfactory grades
def unsatisfactory_grades : ℕ := 4

-- Total number of students
def total_students : ℕ := satisfactory_grades + unsatisfactory_grades

-- Fraction of satisfactory grades
def fraction_satisfactory : ℚ := satisfactory_grades / total_students

theorem fraction_satisfactory_is_two_thirds :
  fraction_satisfactory = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_satisfactory_is_two_thirds_l1943_194371


namespace NUMINAMATH_GPT_B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l1943_194332

-- Define the sets A and B
def setA : Set ℝ := {x : ℝ | x < 1 ∨ x > 2}
def setB (m : ℝ) : Set ℝ := 
  if m = 0 then {x : ℝ | x > 1} 
  else if m < 0 then {x : ℝ | x > 1 ∨ x < (2/m)}
  else if 0 < m ∧ m < 2 then {x : ℝ | 1 < x ∧ x < (2/m)}
  else if m = 2 then ∅
  else {x : ℝ | (2/m) < x ∧ x < 1}

-- Complement of set A
def complementA : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}

-- Proposition: if B subset of complement of A
theorem B_subset_complementA (m : ℝ) : setB m ⊆ complementA ↔ 1 ≤ m ∧ m ≤ 2 := by
  sorry

-- Similarly, we can define the other two propositions
theorem A_intersection_B_nonempty (m : ℝ) : (setA ∩ setB m).Nonempty ↔ m < 1 ∨ m > 2 := by
  sorry

theorem A_union_B_eq_A (m : ℝ) : setA ∪ setB m = setA ↔ m ≥ 2 := by
  sorry

end NUMINAMATH_GPT_B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l1943_194332


namespace NUMINAMATH_GPT_vasya_did_not_buy_anything_days_l1943_194398

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end NUMINAMATH_GPT_vasya_did_not_buy_anything_days_l1943_194398


namespace NUMINAMATH_GPT_minimum_value_shifted_function_l1943_194388

def f (x a : ℝ) : ℝ := x^2 + 4 * x + 7 - a

theorem minimum_value_shifted_function (a : ℝ) (h : ∃ x, f x a = 2) :
  ∃ y, (∃ x, y = f (x - 2015) a) ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_shifted_function_l1943_194388


namespace NUMINAMATH_GPT_complementary_angles_ratio_4_to_1_smaller_angle_l1943_194384

theorem complementary_angles_ratio_4_to_1_smaller_angle :
  ∃ (θ : ℝ), (4 * θ + θ = 90) ∧ (θ = 18) :=
by
  sorry

end NUMINAMATH_GPT_complementary_angles_ratio_4_to_1_smaller_angle_l1943_194384


namespace NUMINAMATH_GPT_last_digit_of_large_prime_l1943_194300

theorem last_digit_of_large_prime :
  let n := 2^859433 - 1
  let last_digit := n % 10
  last_digit = 1 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_large_prime_l1943_194300


namespace NUMINAMATH_GPT_value_range_f_l1943_194354

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 2 * Real.cos x - Real.sin (2 * x) + 1

theorem value_range_f :
  ∀ x ∈ Set.Ico (-(5 * Real.pi) / 12) (Real.pi / 3), 
  f x ∈ Set.Icc ((3 : ℝ) / 2 - Real.sqrt 2) 3 :=
by
  sorry

end NUMINAMATH_GPT_value_range_f_l1943_194354


namespace NUMINAMATH_GPT_january_roses_l1943_194385

theorem january_roses (r_october r_november r_december r_february r_january : ℕ)
  (h_october_november : r_november = r_october + 12)
  (h_november_december : r_december = r_november + 12)
  (h_december_january : r_january = r_december + 12)
  (h_january_february : r_february = r_january + 12) :
  r_january = 144 :=
by {
  -- The proof would go here.
  sorry
}

end NUMINAMATH_GPT_january_roses_l1943_194385


namespace NUMINAMATH_GPT_pencils_multiple_of_40_l1943_194333

theorem pencils_multiple_of_40 :
  ∃ n : ℕ, 640 % n = 0 ∧ n ≤ 40 → ∃ m : ℕ, 40 * m = 40 * n :=
by
  sorry

end NUMINAMATH_GPT_pencils_multiple_of_40_l1943_194333


namespace NUMINAMATH_GPT_krish_remaining_money_l1943_194351

variable (initial_amount sweets stickers friends each_friend charity : ℝ)

theorem krish_remaining_money :
  initial_amount = 200.50 →
  sweets = 35.25 →
  stickers = 10.75 →
  friends = 4 →
  each_friend = 25.20 →
  charity = 15.30 →
  initial_amount - (sweets + stickers + friends * each_friend + charity) = 38.40 :=
by
  intros h_initial h_sweets h_stickers h_friends h_each_friend h_charity
  sorry

end NUMINAMATH_GPT_krish_remaining_money_l1943_194351


namespace NUMINAMATH_GPT_max_roads_no_intersections_l1943_194316

theorem max_roads_no_intersections (V : ℕ) (hV : V = 100) : 
  ∃ E : ℕ, E ≤ 3 * V - 6 ∧ E = 294 := 
by 
  sorry

end NUMINAMATH_GPT_max_roads_no_intersections_l1943_194316


namespace NUMINAMATH_GPT_number_of_pictures_l1943_194314

theorem number_of_pictures (x : ℕ) (h : x - (x / 2 - 1) = 25) : x = 48 :=
sorry

end NUMINAMATH_GPT_number_of_pictures_l1943_194314


namespace NUMINAMATH_GPT_cost_of_one_basketball_deck_l1943_194301

theorem cost_of_one_basketball_deck (total_money_spent : ℕ) 
  (mary_sunglasses_cost : ℕ) (mary_jeans_cost : ℕ) 
  (rose_shoes_cost : ℕ) (rose_decks_count : ℕ) 
  (mary_total_cost : total_money_spent = 2 * mary_sunglasses_cost + mary_jeans_cost)
  (rose_total_cost : total_money_spent = rose_shoes_cost + 2 * (total_money_spent - rose_shoes_cost) / rose_decks_count) :
  (total_money_spent - rose_shoes_cost) / rose_decks_count = 25 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_one_basketball_deck_l1943_194301


namespace NUMINAMATH_GPT_find_exponent_l1943_194392

theorem find_exponent (x : ℕ) (h : 2^x + 2^x + 2^x + 2^x + 2^x = 2048) : x = 9 :=
sorry

end NUMINAMATH_GPT_find_exponent_l1943_194392


namespace NUMINAMATH_GPT_total_people_at_zoo_l1943_194393

theorem total_people_at_zoo (A K : ℕ) (ticket_price_adult : ℕ := 28) (ticket_price_kid : ℕ := 12) (total_sales : ℕ := 3864) (number_of_kids : ℕ := 203) :
  (ticket_price_adult * A + ticket_price_kid * number_of_kids = total_sales) → 
  (A + number_of_kids = 254) :=
by
  sorry

end NUMINAMATH_GPT_total_people_at_zoo_l1943_194393


namespace NUMINAMATH_GPT_line_intersects_ellipse_slopes_l1943_194396

theorem line_intersects_ellipse_slopes (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (1/5)) ∨ m ∈ Set.Ici (Real.sqrt (1/5)) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_ellipse_slopes_l1943_194396


namespace NUMINAMATH_GPT_correct_model_l1943_194375

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end NUMINAMATH_GPT_correct_model_l1943_194375


namespace NUMINAMATH_GPT_factor_of_quadratic_implies_m_value_l1943_194334

theorem factor_of_quadratic_implies_m_value (m : ℤ) : (∀ x : ℤ, (x + 6) ∣ (x^2 - m * x - 42)) → m = 1 := by
  sorry

end NUMINAMATH_GPT_factor_of_quadratic_implies_m_value_l1943_194334


namespace NUMINAMATH_GPT_original_cost_proof_l1943_194374

/-!
# Prove that the original cost of the yearly subscription to professional magazines is $940.
# Given conditions:
# 1. The company must make a 20% cut in the magazine budget.
# 2. After the cut, the company will spend $752.
-/

theorem original_cost_proof (x : ℝ)
  (h1 : 0.80 * x = 752) :
  x = 940 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_proof_l1943_194374


namespace NUMINAMATH_GPT_area_ratio_of_isosceles_triangle_l1943_194331

variable (x : ℝ)
variable (hx : 0 < x)

def isosceles_triangle (AB AC : ℝ) (BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 * x ∧ BC = x

def extend_side (B_length AB_length : ℝ) : Prop :=
  B_length = 2 * AB_length

def ratio_of_areas (area_AB'B'C' area_ABC : ℝ) : Prop :=
  area_AB'B'C' / area_ABC = 9

theorem area_ratio_of_isosceles_triangle
  (AB AC BC : ℝ) (BB' B'C' area_ABC area_AB'B'C' : ℝ)
  (h_isosceles : isosceles_triangle x AB AC BC)
  (h_extend_A : extend_side BB' AB)
  (h_extend_C : extend_side B'C' AC) :
  ratio_of_areas area_AB'B'C' area_ABC := by
  sorry

end NUMINAMATH_GPT_area_ratio_of_isosceles_triangle_l1943_194331


namespace NUMINAMATH_GPT_sum_of_areas_of_six_rectangles_eq_572_l1943_194321

theorem sum_of_areas_of_six_rectangles_eq_572 :
  let lengths := [1, 3, 5, 7, 9, 11]
  let areas := lengths.map (λ x => 2 * x^2)
  areas.sum = 572 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_six_rectangles_eq_572_l1943_194321


namespace NUMINAMATH_GPT_sequence_term_1000_l1943_194303

theorem sequence_term_1000 :
  ∃ (a : ℕ → ℤ), a 1 = 2007 ∧ a 2 = 2008 ∧ (∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) ∧ a 1000 = 2340 := 
by
  sorry

end NUMINAMATH_GPT_sequence_term_1000_l1943_194303


namespace NUMINAMATH_GPT_cyclists_equal_distance_l1943_194383

theorem cyclists_equal_distance (v1 v2 v3 : ℝ) (t1 t2 t3 : ℝ) (d : ℝ)
  (h_v1 : v1 = 12) (h_v2 : v2 = 16) (h_v3 : v3 = 24)
  (h_one_riding : t1 + t2 + t3 = 3) 
  (h_dist_equal : v1 * t1 = v2 * t2 ∧ v2 * t2 = v3 * t3 ∧ v1 * t1 = d) :
  d = 16 :=
by
  sorry

end NUMINAMATH_GPT_cyclists_equal_distance_l1943_194383


namespace NUMINAMATH_GPT_range_of_x_l1943_194302

variable (a x : ℝ)

theorem range_of_x :
  (∃ a ∈ Set.Icc 2 4, a * x ^ 2 + (a - 3) * x - 3 > 0) →
  x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1943_194302


namespace NUMINAMATH_GPT_speed_of_stream_l1943_194348

theorem speed_of_stream 
  (b s : ℝ) 
  (h1 : 78 = (b + s) * 2) 
  (h2 : 50 = (b - s) * 2) 
  : s = 7 := 
sorry

end NUMINAMATH_GPT_speed_of_stream_l1943_194348


namespace NUMINAMATH_GPT_power_equation_l1943_194304

theorem power_equation (m : ℤ) (h : 16 = 2 ^ 4) : (16 : ℝ) ^ (3 / 4) = (2 : ℝ) ^ (m : ℝ) → m = 3 := by
  intros
  sorry

end NUMINAMATH_GPT_power_equation_l1943_194304


namespace NUMINAMATH_GPT_max_distance_from_origin_to_line_l1943_194367

variable (k : ℝ)

def line (x y : ℝ) : Prop := k * x + y + 1 = 0

theorem max_distance_from_origin_to_line :
  ∃ k : ℝ, ∀ x y : ℝ, line k x y -> dist (0, 0) (x, y) ≤ 1 := 
sorry

end NUMINAMATH_GPT_max_distance_from_origin_to_line_l1943_194367


namespace NUMINAMATH_GPT_wall_area_l1943_194315

-- Definition of the width and length of the wall
def width : ℝ := 5.4
def length : ℝ := 2.5

-- Statement of the theorem
theorem wall_area : (width * length) = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_wall_area_l1943_194315


namespace NUMINAMATH_GPT_expected_value_of_win_l1943_194352

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_win_l1943_194352


namespace NUMINAMATH_GPT_min_number_of_girls_l1943_194389

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_number_of_girls_l1943_194389


namespace NUMINAMATH_GPT_Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l1943_194309

/-- Definitions for phone plans A and B and phone call durations -/
def fixed_cost_A : ℕ := 18
def free_minutes_A : ℕ := 1500
def price_per_minute_A : ℕ → ℚ := λ t => 0.1 * t

def fixed_cost_B : ℕ := 38
def free_minutes_B : ℕ := 4000
def price_per_minute_B : ℕ → ℚ := λ t => 0.07 * t

def call_duration_October : ℕ := 2600
def total_bill_November_December : ℚ := 176
def total_call_duration_November_December : ℕ := 5200

/-- Problem statements to be proven -/

theorem Phone_Bill_October : 
  fixed_cost_A + price_per_minute_A (call_duration_October - free_minutes_A) = 128 :=
  sorry

theorem Phone_Bill_November_December (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ total_call_duration_November_December) : 
  let bill_November := fixed_cost_A + price_per_minute_A (x - free_minutes_A)
  let bill_December := fixed_cost_B + price_per_minute_B (total_call_duration_November_December - x - free_minutes_B)
  bill_November + bill_December = total_bill_November_December :=
  sorry
  
theorem Extra_Cost_November_December :
  let actual_cost := 138 + 38
  let hypothetical_cost := fixed_cost_A + price_per_minute_A (total_call_duration_November_December - free_minutes_A)
  hypothetical_cost - actual_cost = 80 :=
  sorry

end NUMINAMATH_GPT_Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l1943_194309


namespace NUMINAMATH_GPT_square_of_binomial_l1943_194376

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_square_of_binomial_l1943_194376


namespace NUMINAMATH_GPT_triangle_area_l1943_194318

theorem triangle_area (c b : ℝ) (c_eq : c = 15) (b_eq : b = 9) :
  ∃ a : ℝ, a^2 = c^2 - b^2 ∧ (b * a) / 2 = 54 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1943_194318


namespace NUMINAMATH_GPT_Emma_age_ratio_l1943_194325

theorem Emma_age_ratio (E M : ℕ) (h1 : E = E) (h2 : E = E) 
(h3 : E - M = 3 * (E - 4 * M)) : E / M = 11 / 2 :=
sorry

end NUMINAMATH_GPT_Emma_age_ratio_l1943_194325


namespace NUMINAMATH_GPT_trigonometric_identity_l1943_194320

open Real

theorem trigonometric_identity (α : ℝ) (h1 : tan α = 4/3) (h2 : 0 < α ∧ α < π / 2) :
  sin (π + α) + cos (π - α) = -7/5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1943_194320


namespace NUMINAMATH_GPT_compute_expression_l1943_194391

theorem compute_expression (p q r : ℝ) 
  (h1 : p + q + r = 6) 
  (h2 : pq + qr + rp = 11) 
  (h3 : pqr = 12) : 
  (pq / r) + (qr / p) + (rp / q) = -23 / 12 := 
sorry

end NUMINAMATH_GPT_compute_expression_l1943_194391


namespace NUMINAMATH_GPT_comb_sum_C8_2_C8_3_l1943_194386

open Nat

theorem comb_sum_C8_2_C8_3 : (Nat.choose 8 2) + (Nat.choose 8 3) = 84 :=
by
  sorry

end NUMINAMATH_GPT_comb_sum_C8_2_C8_3_l1943_194386


namespace NUMINAMATH_GPT_find_natural_number_A_l1943_194356

theorem find_natural_number_A (A : ℕ) : 
  (A * 1000 ≤ (A * (A + 1)) / 2 ∧ (A * (A + 1)) / 2 ≤ A * 1000 + 999) → A = 1999 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_number_A_l1943_194356


namespace NUMINAMATH_GPT_four_digit_number_count_l1943_194336

theorem four_digit_number_count (A : ℕ → ℕ → ℕ)
  (odd_digits even_digits : Finset ℕ)
  (odds : ∀ x ∈ odd_digits, x % 2 = 1)
  (evens : ∀ x ∈ even_digits, x % 2 = 0) :
  odd_digits = {1, 3, 5, 7, 9} ∧ 
  even_digits = {2, 4, 6, 8} →
  A 5 2 * A 7 2 = 840 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_four_digit_number_count_l1943_194336


namespace NUMINAMATH_GPT_exponent_is_23_l1943_194381

theorem exponent_is_23 (k : ℝ) : (1/2: ℝ) ^ 23 * (1/81: ℝ) ^ k = (1/18: ℝ) ^ 23 → 23 = 23 := by
  intro h
  sorry

end NUMINAMATH_GPT_exponent_is_23_l1943_194381


namespace NUMINAMATH_GPT_opening_price_calculation_l1943_194335

variable (Closing_Price : ℝ)
variable (Percent_Increase : ℝ)
variable (Opening_Price : ℝ)

theorem opening_price_calculation
    (H1 : Closing_Price = 28)
    (H2 : Percent_Increase = 0.1200000000000001) :
    Opening_Price = Closing_Price / (1 + Percent_Increase) := by
  sorry

end NUMINAMATH_GPT_opening_price_calculation_l1943_194335


namespace NUMINAMATH_GPT_triangle_ineq_sqrt_triangle_l1943_194306

open Real

theorem triangle_ineq_sqrt_triangle (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a):
  (∃ u v w : ℝ, u > 0 ∧ v > 0 ∧ w > 0 ∧ a = v + w ∧ b = u + w ∧ c = u + v) ∧ 
  (sqrt (a * b) + sqrt (b * c) + sqrt (c * a) ≤ a + b + c ∧ a + b + c ≤ 2 * sqrt (a * b) + 2 * sqrt (b * c) + 2 * sqrt (c * a)) :=
  sorry

end NUMINAMATH_GPT_triangle_ineq_sqrt_triangle_l1943_194306


namespace NUMINAMATH_GPT_number_of_female_officers_is_382_l1943_194305

noncomputable def F : ℝ := 
  let total_on_duty := 210
  let ratio_male_female := 3 / 2
  let percent_female_on_duty := 22 / 100
  let female_on_duty := total_on_duty * (2 / (3 + 2))
  let total_females := female_on_duty / percent_female_on_duty
  total_females

theorem number_of_female_officers_is_382 : F = 382 := 
by
  sorry

end NUMINAMATH_GPT_number_of_female_officers_is_382_l1943_194305


namespace NUMINAMATH_GPT_bake_four_pans_l1943_194329

-- Define the conditions
def bake_time_one_pan : ℕ := 7
def total_bake_time (n : ℕ) : ℕ := 28

-- Define the theorem statement
theorem bake_four_pans : total_bake_time 4 = 28 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_bake_four_pans_l1943_194329


namespace NUMINAMATH_GPT_Y_subset_X_l1943_194324

def X : Set ℕ := {n | ∃ m : ℕ, n = 4 * m + 2}

def Y : Set ℕ := {t | ∃ k : ℕ, t = (2 * k - 1)^2 + 1}

theorem Y_subset_X : Y ⊆ X := by
  sorry

end NUMINAMATH_GPT_Y_subset_X_l1943_194324


namespace NUMINAMATH_GPT_remainder_n_plus_2023_mod_7_l1943_194368

theorem remainder_n_plus_2023_mod_7 (n : ℤ) (h : n % 7 = 2) : (n + 2023) % 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_n_plus_2023_mod_7_l1943_194368


namespace NUMINAMATH_GPT_min_x2_y2_z2_l1943_194387

open Real

theorem min_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_x2_y2_z2_l1943_194387


namespace NUMINAMATH_GPT_find_b_of_perpendicular_bisector_l1943_194362

theorem find_b_of_perpendicular_bisector :
  (∃ b : ℝ, (∀ x y : ℝ, x + y = b → x + y = 4 + 6)) → b = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_b_of_perpendicular_bisector_l1943_194362


namespace NUMINAMATH_GPT_greatest_possible_n_l1943_194394

theorem greatest_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 8100) : n ≤ 8 :=
by
  -- Intentionally left uncommented.
  sorry

end NUMINAMATH_GPT_greatest_possible_n_l1943_194394


namespace NUMINAMATH_GPT_solve_fractional_equation_l1943_194378

theorem solve_fractional_equation {x : ℝ} (h1 : x ≠ -1) (h2 : x ≠ 0) :
  6 / (x + 1) = (x + 5) / (x * (x + 1)) ↔ x = 1 :=
by
  -- This proof is left as an exercise.
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1943_194378


namespace NUMINAMATH_GPT_michael_average_speed_l1943_194342

-- Definitions of conditions
def motorcycle_speed := 20 -- mph
def motorcycle_time := 40 / 60 -- hours
def jogging_speed := 5 -- mph
def jogging_time := 60 / 60 -- hours

-- Define the total distance
def motorcycle_distance := motorcycle_speed * motorcycle_time
def jogging_distance := jogging_speed * jogging_time
def total_distance := motorcycle_distance + jogging_distance

-- Define the total time
def total_time := motorcycle_time + jogging_time

-- The proof statement to be proven
theorem michael_average_speed :
  total_distance / total_time = 11 := 
sorry

end NUMINAMATH_GPT_michael_average_speed_l1943_194342


namespace NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l1943_194311

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l1943_194311


namespace NUMINAMATH_GPT_remainder_of_xyz_l1943_194313

theorem remainder_of_xyz {x y z : ℕ} (hx: x < 9) (hy: y < 9) (hz: z < 9)
  (h1: (x + 3*y + 2*z) % 9 = 0)
  (h2: (2*x + 2*y + z) % 9 = 7)
  (h3: (x + 2*y + 3*z) % 9 = 5) :
  (x * y * z) % 9 = 5 :=
sorry

end NUMINAMATH_GPT_remainder_of_xyz_l1943_194313


namespace NUMINAMATH_GPT_garden_to_land_area_ratio_l1943_194326

variables (l_ter w_ter l_gard w_gard : ℝ)

-- Condition 1: Width of the land rectangle is 3/5 of its length
def land_conditions : Prop := w_ter = (3 / 5) * l_ter

-- Condition 2: Width of the garden rectangle is 3/5 of its length
def garden_conditions : Prop := w_gard = (3 / 5) * l_gard

-- Problem: Ratio of the area of the garden to the area of the land is 36%.
theorem garden_to_land_area_ratio
  (h_land : land_conditions l_ter w_ter)
  (h_garden : garden_conditions l_gard w_gard) :
  (l_gard * w_gard) / (l_ter * w_ter) = 0.36 := sorry

end NUMINAMATH_GPT_garden_to_land_area_ratio_l1943_194326


namespace NUMINAMATH_GPT_matts_weight_l1943_194308

theorem matts_weight (protein_per_powder_rate : ℝ)
                     (weekly_intake_powder : ℝ)
                     (daily_protein_required_per_kg : ℝ)
                     (days_in_week : ℝ)
                     (expected_weight : ℝ)
    (h1 : protein_per_powder_rate = 0.8)
    (h2 : weekly_intake_powder = 1400)
    (h3 : daily_protein_required_per_kg = 2)
    (h4 : days_in_week = 7)
    (h5 : expected_weight = 80) :
    (weekly_intake_powder / days_in_week) * protein_per_powder_rate / daily_protein_required_per_kg = expected_weight := by
  sorry

end NUMINAMATH_GPT_matts_weight_l1943_194308


namespace NUMINAMATH_GPT_distance_between_lines_l1943_194344

-- Definitions from conditions in (a)
def l1 (x y : ℝ) := 3 * x + 4 * y - 7 = 0
def l2 (x y : ℝ) := 6 * x + 8 * y + 1 = 0

-- The proof goal from (c)
theorem distance_between_lines : 
  ∀ (x y : ℝ),
    (l1 x y) → 
    (l2 x y) →
      -- Distance between the lines is 3/2
      ( (|(-14) - 1| : ℝ) / (Real.sqrt (6^2 + 8^2)) ) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_lines_l1943_194344


namespace NUMINAMATH_GPT_pastries_sold_l1943_194373

def initial_pastries : ℕ := 148
def pastries_left : ℕ := 45

theorem pastries_sold : initial_pastries - pastries_left = 103 := by
  sorry

end NUMINAMATH_GPT_pastries_sold_l1943_194373


namespace NUMINAMATH_GPT_feasibility_orderings_l1943_194363

theorem feasibility_orderings (a : ℝ) :
  (a ≠ 0) →
  (∀ a > 0, a < 2 * a ∧ 2 * a < 3 * a + 1) ∧
  ¬∃ a, a < 3 * a + 1 ∧ 3 * a + 1 < 2 * a ∧ 2 * a < 3 * a + 1 ∧ a ≠ 0 ∧ a > 0 ∧ a < -1 / 2 ∧ a < 0 ∧ a < -1 ∧ a < -1 / 2 ∧ a < -1 / 2 ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_feasibility_orderings_l1943_194363


namespace NUMINAMATH_GPT_gcd_f_x_l1943_194327

-- Define that x is a multiple of 23478
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Define the function f(x)
noncomputable def f (x : ℕ) : ℕ := (2 * x + 3) * (7 * x + 2) * (13 * x + 7) * (x + 13)

-- Assert the proof problem
theorem gcd_f_x (x : ℕ) (h : is_multiple_of x 23478) : Nat.gcd (f x) x = 546 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_f_x_l1943_194327


namespace NUMINAMATH_GPT_prove_nat_number_l1943_194361

theorem prove_nat_number (p : ℕ) (hp : Nat.Prime p) (n : ℕ) :
  n^2 = p^2 + 3*p + 9 → n = 7 :=
sorry

end NUMINAMATH_GPT_prove_nat_number_l1943_194361
