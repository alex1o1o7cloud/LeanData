import Mathlib

namespace NUMINAMATH_GPT_light_travel_distance_in_km_l1540_154032

-- Define the conditions
def speed_of_light_miles_per_sec : ℝ := 186282
def conversion_factor_mile_to_km : ℝ := 1.609
def time_seconds : ℕ := 500
def expected_distance_km : ℝ := 1.498 * 10^8

-- The theorem we need to prove
theorem light_travel_distance_in_km :
  (speed_of_light_miles_per_sec * time_seconds * conversion_factor_mile_to_km) = expected_distance_km :=
  sorry

end NUMINAMATH_GPT_light_travel_distance_in_km_l1540_154032


namespace NUMINAMATH_GPT_find_k_l1540_154063

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x = -2 ∧ x^2 - k * x + 2 = 0) : k = -3 := by
  sorry

end NUMINAMATH_GPT_find_k_l1540_154063


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1540_154005

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ)
  (hq_pos : 0 < q)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_arith : 2 * (1/2) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 10 + a 12) / (a 7 + a 9) = 27 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1540_154005


namespace NUMINAMATH_GPT_min_degree_g_l1540_154057

open Polynomial

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

-- Conditions
axiom cond1 : 5 • f + 7 • g = h
axiom cond2 : natDegree f = 10
axiom cond3 : natDegree h = 12

-- Question: Minimum degree of g
theorem min_degree_g : natDegree g = 12 :=
sorry

end NUMINAMATH_GPT_min_degree_g_l1540_154057


namespace NUMINAMATH_GPT_square_area_EFGH_l1540_154009

theorem square_area_EFGH (AB BP : ℝ) (h1 : AB = Real.sqrt 72) (h2 : BP = 2) (x : ℝ)
  (h3 : AB + BP = 2 * x + 2) : x^2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_square_area_EFGH_l1540_154009


namespace NUMINAMATH_GPT_cost_per_steak_knife_l1540_154096

theorem cost_per_steak_knife
  (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ)
  (h1 : sets = 2) (h2 : knives_per_set = 4) (h3 : cost_per_set = 80) :
  (cost_per_set * sets) / (sets * knives_per_set) = 20 := by
  sorry

end NUMINAMATH_GPT_cost_per_steak_knife_l1540_154096


namespace NUMINAMATH_GPT_find_x_solution_l1540_154064

noncomputable def find_x (x y : ℝ) (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : Prop := 
  x = (3 + Real.sqrt 17) / 2

theorem find_x_solution (x y : ℝ) 
(h1 : x - y^2 = 3) 
(h2 : x^2 + y^4 = 13) 
(hx_pos : 0 < x) 
(hy_pos : 0 < y) : 
  find_x x y h1 h2 :=
sorry

end NUMINAMATH_GPT_find_x_solution_l1540_154064


namespace NUMINAMATH_GPT_multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l1540_154076

variable (a b : ℤ)
variable (h1 : a % 4 = 0) 
variable (h2 : b % 8 = 0)

theorem multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : b % 4 = 0 := by
  sorry

theorem diff_multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 4 = 0 := by
  sorry

theorem diff_multiple_of_two (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 2 = 0 := by
  sorry

end NUMINAMATH_GPT_multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l1540_154076


namespace NUMINAMATH_GPT_abs_five_minus_e_l1540_154045

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_e : |5 - e| = 2.282 := 
by 
    -- Proof is omitted 
    sorry

end NUMINAMATH_GPT_abs_five_minus_e_l1540_154045


namespace NUMINAMATH_GPT_circle_table_acquaintance_impossible_l1540_154030

theorem circle_table_acquaintance_impossible (P : Finset ℕ) (hP : P.card = 40) :
  ¬ (∀ (a b : ℕ), (a ∈ P) → (b ∈ P) → (∃ k, 2 * k ≠ 0) → (∃ c, c ∈ P) ∧ (a ≠ b) ∧ (c = a ∨ c = b)
       ↔ ¬(∃ k, 2 * k + 1 ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_circle_table_acquaintance_impossible_l1540_154030


namespace NUMINAMATH_GPT_POTOP_correct_l1540_154055

def POTOP : Nat := 51715

theorem POTOP_correct :
  (99999 * POTOP) % 1000 = 285 := by
  sorry

end NUMINAMATH_GPT_POTOP_correct_l1540_154055


namespace NUMINAMATH_GPT_min_value_tan_product_l1540_154068

theorem min_value_tan_product (A B C : ℝ) (h : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (sin_eq : Real.sin A = 3 * Real.sin B * Real.sin C) :
  ∃ t : ℝ, t = Real.tan A * Real.tan B * Real.tan C ∧ t = 12 :=
sorry

end NUMINAMATH_GPT_min_value_tan_product_l1540_154068


namespace NUMINAMATH_GPT_bird_counts_l1540_154014

theorem bird_counts :
  ∀ (num_cages_1 num_cages_2 num_cages_empty parrot_per_cage parakeet_per_cage canary_per_cage cockatiel_per_cage lovebird_per_cage finch_per_cage total_cages : ℕ),
    num_cages_1 = 7 →
    num_cages_2 = 6 →
    num_cages_empty = 2 →
    parrot_per_cage = 3 →
    parakeet_per_cage = 5 →
    canary_per_cage = 4 →
    cockatiel_per_cage = 2 →
    lovebird_per_cage = 3 →
    finch_per_cage = 1 →
    total_cages = 15 →
    (num_cages_1 * parrot_per_cage = 21) ∧
    (num_cages_1 * parakeet_per_cage = 35) ∧
    (num_cages_1 * canary_per_cage = 28) ∧
    (num_cages_2 * cockatiel_per_cage = 12) ∧
    (num_cages_2 * lovebird_per_cage = 18) ∧
    (num_cages_2 * finch_per_cage = 6) :=
by
  intros
  sorry

end NUMINAMATH_GPT_bird_counts_l1540_154014


namespace NUMINAMATH_GPT_greatest_possible_perimeter_l1540_154070

theorem greatest_possible_perimeter (x : ℤ) (hx1 : 3 * x > 17) (hx2 : 17 > x) : 
  (3 * x + 17 ≤ 65) :=
by
  have Hx : x ≤ 16 := sorry -- Derived from inequalities hx1 and hx2
  have Hx_ge_6 : x ≥ 6 := sorry -- Derived from integer constraint and hx1, hx2
  sorry -- Show 3 * x + 17 has maximum value 65 when x = 16

end NUMINAMATH_GPT_greatest_possible_perimeter_l1540_154070


namespace NUMINAMATH_GPT_ratio_of_volumes_l1540_154047

noncomputable def volumeSphere (p : ℝ) : ℝ := (4/3) * Real.pi * (p^3)

noncomputable def volumeHemisphere (p : ℝ) : ℝ := (1/2) * (4/3) * Real.pi * (3*p)^3

theorem ratio_of_volumes (p : ℝ) (hp : p > 0) : volumeSphere p / volumeHemisphere p = 2 / 27 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l1540_154047


namespace NUMINAMATH_GPT_residue_of_neg_1237_mod_37_l1540_154008

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by
  sorry

end NUMINAMATH_GPT_residue_of_neg_1237_mod_37_l1540_154008


namespace NUMINAMATH_GPT_fraction_of_women_married_l1540_154066

theorem fraction_of_women_married (total : ℕ) (women men married: ℕ) (h1 : total = women + men)
(h2 : women = 76 * total / 100) (h3 : married = 60 * total / 100) (h4 : 2 * (men - married) = 3 * men):
 (married - (total - women - married) * 1 / 3) = 13 * women / 19 :=
sorry

end NUMINAMATH_GPT_fraction_of_women_married_l1540_154066


namespace NUMINAMATH_GPT_form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l1540_154025

theorem form_eleven : 22 - (2 + (2 / 2)) = 11 := by
  sorry

theorem form_twelve : (2 * 2 * 2) - 2 / 2 = 12 := by
  sorry

theorem form_thirteen : (22 + 2 + 2) / 2 = 13 := by
  sorry

theorem form_fourteen : 2 * 2 * 2 * 2 - 2 = 14 := by
  sorry

theorem form_fifteen : (2 * 2)^2 - 2 / 2 = 15 := by
  sorry

theorem form_sixteen : (2 * 2)^2 * (2 / 2) = 16 := by
  sorry

theorem form_seventeen : (2 * 2)^2 + 2 / 2 = 17 := by
  sorry

theorem form_eighteen : 2 * 2 * 2 * 2 + 2 = 18 := by
  sorry

theorem form_nineteen : 22 - 2 - 2 / 2 = 19 := by
  sorry

theorem form_twenty : (22 - 2) * (2 / 2) = 20 := by
  sorry

end NUMINAMATH_GPT_form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l1540_154025


namespace NUMINAMATH_GPT_cost_of_whitewashing_l1540_154072

-- Definitions of the dimensions
def length_room : ℝ := 25.0
def width_room : ℝ := 15.0
def height_room : ℝ := 12.0

def dimensions_door : (ℝ × ℝ) := (6.0, 3.0)
def dimensions_window : (ℝ × ℝ) := (4.0, 3.0)
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 6.0

-- Definition of areas and costs
def area_wall (a b : ℝ) : ℝ := 2 * (a * b)
def area_door : ℝ := (dimensions_door.1 * dimensions_door.2)
def area_window : ℝ := (dimensions_window.1 * dimensions_window.2) * (num_windows)
def total_area_walls : ℝ := (area_wall length_room height_room) + (area_wall width_room height_room)
def area_to_paint : ℝ := total_area_walls - (area_door + area_window)
def total_cost : ℝ := area_to_paint * cost_per_sqft

-- Proof statement
theorem cost_of_whitewashing : total_cost = 5436 := by
  sorry

end NUMINAMATH_GPT_cost_of_whitewashing_l1540_154072


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1540_154085

theorem sale_in_fifth_month (s1 s2 s3 s4 s5 s6 : ℤ) (avg_sale : ℤ) (h1 : s1 = 6435) (h2 : s2 = 6927)
  (h3 : s3 = 6855) (h4 : s4 = 7230) (h6 : s6 = 7391) (h_avg_sale : avg_sale = 6900) :
    (s1 + s2 + s3 + s4 + s5 + s6) / 6 = avg_sale → s5 = 6562 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l1540_154085


namespace NUMINAMATH_GPT_emily_selects_green_apples_l1540_154043

theorem emily_selects_green_apples :
  let total_apples := 10
  let red_apples := 6
  let green_apples := 4
  let selected_apples := 3
  let total_combinations := Nat.choose total_apples selected_apples
  let green_combinations := Nat.choose green_apples selected_apples
  (green_combinations / total_combinations : ℚ) = 1 / 30 :=
by
  sorry

end NUMINAMATH_GPT_emily_selects_green_apples_l1540_154043


namespace NUMINAMATH_GPT_isabel_money_left_l1540_154054

theorem isabel_money_left (initial_amount : ℕ) (half_toy_expense half_book_expense money_left : ℕ) :
  initial_amount = 204 →
  half_toy_expense = initial_amount / 2 →
  half_book_expense = (initial_amount - half_toy_expense) / 2 →
  money_left = initial_amount - half_toy_expense - half_book_expense →
  money_left = 51 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_isabel_money_left_l1540_154054


namespace NUMINAMATH_GPT_circle_area_solution_l1540_154001

def circle_area_problem : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 6 * x - 8 * y - 12 = 0 -> ∃ (A : ℝ), A = 37 * Real.pi

theorem circle_area_solution : circle_area_problem :=
by
  sorry

end NUMINAMATH_GPT_circle_area_solution_l1540_154001


namespace NUMINAMATH_GPT_find_other_number_l1540_154095

theorem find_other_number (y : ℕ) : Nat.lcm 240 y = 5040 ∧ Nat.gcd 240 y = 24 → y = 504 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1540_154095


namespace NUMINAMATH_GPT_WalterWorksDaysAWeek_l1540_154081

theorem WalterWorksDaysAWeek (hourlyEarning : ℕ) (hoursPerDay : ℕ) (schoolAllocationFraction : ℚ) (schoolAllocation : ℕ) 
  (dailyEarning : ℕ) (weeklyEarning : ℕ) (daysWorked : ℕ) :
  hourlyEarning = 5 →
  hoursPerDay = 4 →
  schoolAllocationFraction = 3 / 4 →
  schoolAllocation = 75 →
  dailyEarning = hourlyEarning * hoursPerDay →
  weeklyEarning = (schoolAllocation : ℚ) / schoolAllocationFraction →
  daysWorked = weeklyEarning / dailyEarning →
  daysWorked = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_WalterWorksDaysAWeek_l1540_154081


namespace NUMINAMATH_GPT_find_y_l1540_154017

variables (x y : ℝ)

theorem find_y (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1540_154017


namespace NUMINAMATH_GPT_exists_solution_in_interval_l1540_154028

noncomputable def f (x : ℝ) : ℝ := x^3 - 2^x

theorem exists_solution_in_interval : ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f x = 0 :=
by {
  -- Use the Intermediate Value Theorem, given f is continuous on [1, 2]
  sorry
}

end NUMINAMATH_GPT_exists_solution_in_interval_l1540_154028


namespace NUMINAMATH_GPT_degrees_to_radians_l1540_154003

theorem degrees_to_radians (deg : ℝ) (rad : ℝ) (h1 : 1 = π / 180) (h2 : deg = 60) : rad = deg * (π / 180) :=
by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_l1540_154003


namespace NUMINAMATH_GPT_complex_number_value_l1540_154021

open Complex

theorem complex_number_value (a : ℝ) 
  (h1 : z = (2 + a * I) / (1 + I)) 
  (h2 : (z.re, z.im) ∈ { p : ℝ × ℝ | p.2 = -p.1 }) : 
  a = 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_value_l1540_154021


namespace NUMINAMATH_GPT_neg_mod_eq_1998_l1540_154091

theorem neg_mod_eq_1998 {a : ℤ} (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end NUMINAMATH_GPT_neg_mod_eq_1998_l1540_154091


namespace NUMINAMATH_GPT_fewest_cookies_by_ben_l1540_154039

noncomputable def cookie_problem : Prop :=
  let ana_area := 4 * Real.pi
  let ben_area := 9
  let carol_area := Real.sqrt (5 * (5 + 2 * Real.sqrt 5))
  let dave_area := 3.375 * Real.sqrt 3
  let dough := ana_area * 10
  let ana_cookies := dough / ana_area
  let ben_cookies := dough / ben_area
  let carol_cookies := dough / carol_area
  let dave_cookies := dough / dave_area
  ben_cookies < ana_cookies ∧ ben_cookies < carol_cookies ∧ ben_cookies < dave_cookies

theorem fewest_cookies_by_ben : cookie_problem := by
  sorry

end NUMINAMATH_GPT_fewest_cookies_by_ben_l1540_154039


namespace NUMINAMATH_GPT_correct_answer_l1540_154089

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + m * x - 1

theorem correct_answer (m : ℝ) : 
  (∀ x₁ x₂, 1 < x₁ → 1 < x₂ → (f x₁ m - f x₂ m) / (x₁ - x₂) > 0) → m ≥ -4 :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_l1540_154089


namespace NUMINAMATH_GPT_proof_problem_l1540_154090

theorem proof_problem (a b c : ℝ) (h1 : 4 * a - 2 * b + c > 0) (h2 : a + b + c < 0) : b^2 > a * c :=
sorry

end NUMINAMATH_GPT_proof_problem_l1540_154090


namespace NUMINAMATH_GPT_unique_element_a_values_set_l1540_154010

open Set

theorem unique_element_a_values_set :
  {a : ℝ | ∃! x : ℝ, a * x^2 + 2 * x - a = 0} = {0} :=
by
  sorry

end NUMINAMATH_GPT_unique_element_a_values_set_l1540_154010


namespace NUMINAMATH_GPT_imaginary_unit_sum_l1540_154023

theorem imaginary_unit_sum (i : ℂ) (H : i^4 = 1) : i^1234 + i^1235 + i^1236 + i^1237 = 0 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_unit_sum_l1540_154023


namespace NUMINAMATH_GPT_range_quadratic_function_l1540_154036

theorem range_quadratic_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = x^2 - 2 * x + 5 ↔ y ∈ Set.Ici 4 :=
by 
  sorry

end NUMINAMATH_GPT_range_quadratic_function_l1540_154036


namespace NUMINAMATH_GPT_angle_A_in_triangle_l1540_154079

theorem angle_A_in_triangle :
  ∀ (A B C : ℝ) (a b c : ℝ),
  a = 2 * Real.sqrt 3 → b = 2 * Real.sqrt 2 → B = π / 4 → 
  (A = π / 3 ∨ A = 2 * π / 3) :=
by
  intros A B C a b c ha hb hB
  sorry

end NUMINAMATH_GPT_angle_A_in_triangle_l1540_154079


namespace NUMINAMATH_GPT_total_bulbs_is_118_l1540_154097

-- Define the number of medium lights
def medium_lights : Nat := 12

-- Define the number of large and small lights based on the given conditions
def large_lights : Nat := 2 * medium_lights
def small_lights : Nat := medium_lights + 10

-- Define the number of bulbs required for each type of light
def bulbs_needed_for_medium : Nat := 2 * medium_lights
def bulbs_needed_for_large : Nat := 3 * large_lights
def bulbs_needed_for_small : Nat := 1 * small_lights

-- Define the total number of bulbs needed
def total_bulbs_needed : Nat := bulbs_needed_for_medium + bulbs_needed_for_large + bulbs_needed_for_small

-- The theorem that represents the proof problem
theorem total_bulbs_is_118 : total_bulbs_needed = 118 := by 
  sorry

end NUMINAMATH_GPT_total_bulbs_is_118_l1540_154097


namespace NUMINAMATH_GPT_smallest_d_factors_l1540_154034

theorem smallest_d_factors (d : ℕ) (h₁ : ∃ p q : ℤ, p * q = 2050 ∧ p + q = d ∧ p > 0 ∧ q > 0) :
    d = 107 :=
by
  sorry

end NUMINAMATH_GPT_smallest_d_factors_l1540_154034


namespace NUMINAMATH_GPT_mary_daily_tasks_l1540_154087

theorem mary_daily_tasks :
  ∃ (x y : ℕ), (x + y = 15) ∧ (4 * x + 7 * y = 85) ∧ (y = 8) :=
by
  sorry

end NUMINAMATH_GPT_mary_daily_tasks_l1540_154087


namespace NUMINAMATH_GPT_intersection_with_y_axis_l1540_154099

theorem intersection_with_y_axis (x y : ℝ) : (x + y - 3 = 0 ∧ x = 0) → (x = 0 ∧ y = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_with_y_axis_l1540_154099


namespace NUMINAMATH_GPT_inequality_for_a_ne_1_l1540_154098

theorem inequality_for_a_ne_1 (a : ℝ) (h : a ≠ 1) : (1 + a + a^2)^2 < 3 * (1 + a^2 + a^4) :=
sorry

end NUMINAMATH_GPT_inequality_for_a_ne_1_l1540_154098


namespace NUMINAMATH_GPT_probability_bypass_kth_intersection_l1540_154050

variable (n k : ℕ)

def P (n k : ℕ) : ℚ := (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_bypass_kth_intersection :
  P n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 :=
by
  sorry

end NUMINAMATH_GPT_probability_bypass_kth_intersection_l1540_154050


namespace NUMINAMATH_GPT_find_starting_point_of_a_l1540_154011

def point := ℝ × ℝ
def vector := ℝ × ℝ

def B : point := (1, 0)

def b : vector := (-3, -4)
def c : vector := (1, 1)

def a : vector := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)

theorem find_starting_point_of_a (hb : b = (-3, -4)) (hc : c = (1, 1)) (hB : B = (1, 0)) :
    let a := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)
    let start_A := (B.1 - a.1, B.2 - a.2)
    start_A = (12, 14) :=
by
  rw [hb, hc, hB]
  let a := (3 * (-3) - 2 * (1), 3 * (-4) - 2 * (1))
  let start_A := (1 - a.1, 0 - a.2)
  simp [a]
  sorry

end NUMINAMATH_GPT_find_starting_point_of_a_l1540_154011


namespace NUMINAMATH_GPT_Sandwiches_count_l1540_154029

-- Define the number of toppings and the number of choices for the patty
def num_toppings : Nat := 10
def num_choices_per_topping : Nat := 2
def num_patties : Nat := 3

-- Define the theorem to prove the total number of sandwiches
theorem Sandwiches_count : (num_choices_per_topping ^ num_toppings) * num_patties = 3072 :=
by
  sorry

end NUMINAMATH_GPT_Sandwiches_count_l1540_154029


namespace NUMINAMATH_GPT_integer_solutions_to_abs_equation_l1540_154019

theorem integer_solutions_to_abs_equation :
  {p : ℤ × ℤ | abs (p.1 - 2) + abs (p.2 - 1) = 1} =
  {(3, 1), (1, 1), (2, 2), (2, 0)} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_to_abs_equation_l1540_154019


namespace NUMINAMATH_GPT_triangle_area_is_60_l1540_154093

noncomputable def triangle_area (P r : ℝ) : ℝ :=
  (r * P) / 2

theorem triangle_area_is_60 (hP : 48 = 48) (hr : 2.5 = 2.5) : triangle_area 48 2.5 = 60 := by
  sorry

end NUMINAMATH_GPT_triangle_area_is_60_l1540_154093


namespace NUMINAMATH_GPT_second_part_of_ratio_l1540_154044

theorem second_part_of_ratio (first_part : ℝ) (whole second_part : ℝ) (h1 : first_part = 5) (h2 : first_part / whole = 25 / 100) : second_part = 15 :=
by
  sorry

end NUMINAMATH_GPT_second_part_of_ratio_l1540_154044


namespace NUMINAMATH_GPT_simplify_fraction_l1540_154075

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : (2 + 4 * i) / (1 - 5 * i) = (-9 / 13) + (7 / 13) * i :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1540_154075


namespace NUMINAMATH_GPT_triangle_angle_A_triangle_bc_range_l1540_154042

theorem triangle_angle_A (a b c A B C : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (ha : a = b * Real.sin C + c * Real.sin B)
  (hb : b = c * Real.sin A + a * Real.sin C)
  (hc : c = a * Real.sin B + b * Real.sin A)
  (h_eq : (Real.sqrt 3) * a * Real.sin C + a * Real.cos C = c + b)
  (h_angles_sum : A + B + C = π) :
    A = π/3 := -- π/3 radians equals 60 degrees
sorry

theorem triangle_bc_range (a b c : ℝ) (h : a = Real.sqrt 3) :
  Real.sqrt 3 < b + c ∧ b + c ≤ 2 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_triangle_angle_A_triangle_bc_range_l1540_154042


namespace NUMINAMATH_GPT_ratio_Binkie_Frankie_eq_4_l1540_154016

-- Definitions based on given conditions
def SpaatzGems : ℕ := 1
def BinkieGems : ℕ := 24

-- Assume the number of gemstones on Frankie's collar
variable (FrankieGems : ℕ)

-- Given condition about the gemstones on Spaatz's collar
axiom SpaatzCondition : SpaatzGems = (FrankieGems / 2) - 2

-- The theorem to be proved
theorem ratio_Binkie_Frankie_eq_4 
    (FrankieGems : ℕ) 
    (SpaatzCondition : SpaatzGems = (FrankieGems / 2) - 2) 
    (BinkieGems_eq : BinkieGems = 24) 
    (SpaatzGems_eq : SpaatzGems = 1) 
    (f_nonzero : FrankieGems ≠ 0) :
    BinkieGems / FrankieGems = 4 :=
by
  sorry  -- We're only writing the statement, not the proof.

end NUMINAMATH_GPT_ratio_Binkie_Frankie_eq_4_l1540_154016


namespace NUMINAMATH_GPT_mary_screws_sections_l1540_154061

def number_of_sections (initial_screws : Nat) (multiplier : Nat) (screws_per_section : Nat) : Nat :=
  let additional_screws := initial_screws * multiplier
  let total_screws := initial_screws + additional_screws
  total_screws / screws_per_section

theorem mary_screws_sections :
  number_of_sections 8 2 6 = 4 := by
  sorry

end NUMINAMATH_GPT_mary_screws_sections_l1540_154061


namespace NUMINAMATH_GPT_length_of_AP_in_right_triangle_l1540_154031

theorem length_of_AP_in_right_triangle 
  (A B C : ℝ × ℝ)
  (hA : A = (0, 2))
  (hB : B = (0, 0))
  (hC : C = (2, 0))
  (M : ℝ × ℝ)
  (hM : M.1 = 0 ∧ M.2 = 0)
  (inc : ℝ × ℝ)
  (hinc : inc = (1, 1)) :
  ∃ P : ℝ × ℝ, (P.1 = 0 ∧ P.2 = 1) ∧ dist A P = 1 := by
  sorry

end NUMINAMATH_GPT_length_of_AP_in_right_triangle_l1540_154031


namespace NUMINAMATH_GPT_f_1992_eq_1992_l1540_154038

def f (x : ℕ) : ℤ := sorry

theorem f_1992_eq_1992 (f : ℕ → ℤ) 
  (h1 : ∀ x : ℕ, 0 < x -> f x = f (x - 1) + f (x + 1))
  (h2 : f 0 = 1992) :
  f 1992 = 1992 := 
sorry

end NUMINAMATH_GPT_f_1992_eq_1992_l1540_154038


namespace NUMINAMATH_GPT_cats_kittentotal_l1540_154026

def kittens_given_away : ℕ := 2
def kittens_now : ℕ := 6
def kittens_original : ℕ := 8

theorem cats_kittentotal : kittens_now + kittens_given_away = kittens_original := 
by 
  sorry

end NUMINAMATH_GPT_cats_kittentotal_l1540_154026


namespace NUMINAMATH_GPT_f_is_constant_l1540_154020

noncomputable def f (x θ : ℝ) : ℝ :=
  (Real.cos (x - θ))^2 + (Real.cos x)^2 - 2 * Real.cos θ * Real.cos (x - θ) * Real.cos x

theorem f_is_constant (θ : ℝ) : ∀ x, f x θ = (Real.sin θ)^2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_f_is_constant_l1540_154020


namespace NUMINAMATH_GPT_joan_missed_games_l1540_154071

variable (total_games : ℕ) (night_games : ℕ) (attended_games : ℕ)

theorem joan_missed_games (h1 : total_games = 864) (h2 : night_games = 128) (h3 : attended_games = 395) : 
  total_games - attended_games = 469 :=
  by
    sorry

end NUMINAMATH_GPT_joan_missed_games_l1540_154071


namespace NUMINAMATH_GPT_total_payment_correct_l1540_154094

-- Define the conditions for each singer
def firstSingerPayment : ℝ := 2 * 25
def secondSingerPayment : ℝ := 3 * 35
def thirdSingerPayment : ℝ := 4 * 20
def fourthSingerPayment : ℝ := 2.5 * 30

def firstSingerTip : ℝ := 0.15 * firstSingerPayment
def secondSingerTip : ℝ := 0.20 * secondSingerPayment
def thirdSingerTip : ℝ := 0.25 * thirdSingerPayment
def fourthSingerTip : ℝ := 0.18 * fourthSingerPayment

def firstSingerTotal : ℝ := firstSingerPayment + firstSingerTip
def secondSingerTotal : ℝ := secondSingerPayment + secondSingerTip
def thirdSingerTotal : ℝ := thirdSingerPayment + thirdSingerTip
def fourthSingerTotal : ℝ := fourthSingerPayment + fourthSingerTip

-- Define the total amount paid
def totalPayment : ℝ := firstSingerTotal + secondSingerTotal + thirdSingerTotal + fourthSingerTotal

-- The proof problem: Prove the total amount paid
theorem total_payment_correct : totalPayment = 372 := by
  sorry

end NUMINAMATH_GPT_total_payment_correct_l1540_154094


namespace NUMINAMATH_GPT_cooperative_payment_divisibility_l1540_154082

theorem cooperative_payment_divisibility (T_old : ℕ) (N : ℕ) 
  (hN : N = 99 * T_old / 100) : 99 ∣ N :=
by
  sorry

end NUMINAMATH_GPT_cooperative_payment_divisibility_l1540_154082


namespace NUMINAMATH_GPT_best_fitting_model_l1540_154051

/-- Four models with different coefficients of determination -/
def model1_R2 : ℝ := 0.98
def model2_R2 : ℝ := 0.80
def model3_R2 : ℝ := 0.50
def model4_R2 : ℝ := 0.25

/-- Prove that Model 1 has the best fitting effect among the given models -/
theorem best_fitting_model :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by {sorry}

end NUMINAMATH_GPT_best_fitting_model_l1540_154051


namespace NUMINAMATH_GPT_problem1_l1540_154035

theorem problem1 {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end NUMINAMATH_GPT_problem1_l1540_154035


namespace NUMINAMATH_GPT_cylinder_inscribed_in_sphere_l1540_154007

noncomputable def sphere_volume (r : ℝ) : ℝ := 
  (4 / 3) * Real.pi * r^3

theorem cylinder_inscribed_in_sphere 
  (r_cylinder : ℝ)
  (h₁ : r_cylinder > 0)
  (height_cylinder : ℝ)
  (radius_sphere : ℝ)
  (h₂ : radius_sphere = r_cylinder + 2)
  (h₃ : height_cylinder = r_cylinder + 1)
  (h₄ : 2 * radius_sphere = Real.sqrt ((2 * r_cylinder)^2 + (height_cylinder)^2))
  : sphere_volume 17 = 6550 * 2 / 3 * Real.pi :=
by
  -- solution steps and proof go here
  sorry

end NUMINAMATH_GPT_cylinder_inscribed_in_sphere_l1540_154007


namespace NUMINAMATH_GPT_Jackson_missed_one_wednesday_l1540_154086

theorem Jackson_missed_one_wednesday (weeks total_sandwiches missed_fridays sandwiches_eaten : ℕ) 
  (h1 : weeks = 36)
  (h2 : total_sandwiches = 2 * weeks)
  (h3 : missed_fridays = 2)
  (h4 : sandwiches_eaten = 69) :
  (total_sandwiches - missed_fridays - sandwiches_eaten) / 2 = 1 :=
by
  -- sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_Jackson_missed_one_wednesday_l1540_154086


namespace NUMINAMATH_GPT_sequence_periodicity_l1540_154012

theorem sequence_periodicity (a : ℕ → ℚ) (h1 : a 1 = 6 / 7)
  (h_rec : ∀ n, 0 ≤ a n ∧ a n < 1 → a (n+1) = if a n ≤ 1/2 then 2 * a n else 2 * a n - 1) :
  a 2017 = 6 / 7 :=
  sorry

end NUMINAMATH_GPT_sequence_periodicity_l1540_154012


namespace NUMINAMATH_GPT_problem_l1540_154077

theorem problem (x y : ℝ) (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5) : 5 * x ^ 2 + 8 * x * y + 5 * y ^ 2 = 41 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l1540_154077


namespace NUMINAMATH_GPT_ratio_of_boys_to_total_l1540_154060

theorem ratio_of_boys_to_total (b : ℝ) (h1 : b = 3 / 4 * (1 - b)) : b = 3 / 7 :=
by
  {
    -- The given condition (we use it to prove the target statement)
    sorry
  }

end NUMINAMATH_GPT_ratio_of_boys_to_total_l1540_154060


namespace NUMINAMATH_GPT_range_x_minus_q_l1540_154078

theorem range_x_minus_q (x q : ℝ) (h1 : |x - 3| > q) (h2 : x < 3) : x - q < 3 - 2*q :=
by
  sorry

end NUMINAMATH_GPT_range_x_minus_q_l1540_154078


namespace NUMINAMATH_GPT_remainder_of_3_pow_19_div_10_l1540_154088

def w : ℕ := 3 ^ 19

theorem remainder_of_3_pow_19_div_10 : w % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_19_div_10_l1540_154088


namespace NUMINAMATH_GPT_simplify_expression_l1540_154067

theorem simplify_expression :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l1540_154067


namespace NUMINAMATH_GPT_cos_of_angle_in_third_quadrant_l1540_154002

theorem cos_of_angle_in_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = -5 / 13) : Real.cos B = -12 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_cos_of_angle_in_third_quadrant_l1540_154002


namespace NUMINAMATH_GPT_minimum_guests_economical_option_l1540_154080

theorem minimum_guests_economical_option :
  ∀ (x : ℕ), (150 + 20 * x > 300 + 15 * x) → x > 30 :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_minimum_guests_economical_option_l1540_154080


namespace NUMINAMATH_GPT_linear_coefficient_l1540_154084

theorem linear_coefficient (a b c : ℤ) (h : a = 1 ∧ b = -2 ∧ c = -1) :
    b = -2 := 
by
  -- Use the given hypothesis directly
  exact h.2.1

end NUMINAMATH_GPT_linear_coefficient_l1540_154084


namespace NUMINAMATH_GPT_find_salary_month_l1540_154027

variable (J F M A May : ℝ)

def condition_1 : Prop := (J + F + M + A) / 4 = 8000
def condition_2 : Prop := (F + M + A + May) / 4 = 8450
def condition_3 : Prop := J = 4700
def condition_4 (X : ℝ) : Prop := X = 6500

theorem find_salary_month (J F M A May : ℝ) 
  (h1 : condition_1 J F M A) 
  (h2 : condition_2 F M A May) 
  (h3 : condition_3 J) 
  : ∃ M : ℝ, condition_4 May :=
by sorry

end NUMINAMATH_GPT_find_salary_month_l1540_154027


namespace NUMINAMATH_GPT_number_of_adults_in_family_l1540_154040

-- Conditions as definitions
def total_apples : ℕ := 1200
def number_of_children : ℕ := 45
def apples_per_child : ℕ := 15
def apples_per_adult : ℕ := 5

-- Calculations based on conditions
def apples_eaten_by_children : ℕ := number_of_children * apples_per_child
def remaining_apples : ℕ := total_apples - apples_eaten_by_children
def number_of_adults : ℕ := remaining_apples / apples_per_adult

-- Proof target: number of adults in Bob's family equals 105
theorem number_of_adults_in_family : number_of_adults = 105 := by
  sorry

end NUMINAMATH_GPT_number_of_adults_in_family_l1540_154040


namespace NUMINAMATH_GPT_algebraic_expression_simplification_l1540_154049

theorem algebraic_expression_simplification :
  0.25 * (-1 / 2) ^ (-4 : ℝ) - 4 / (Real.sqrt 5 - 1) ^ (0 : ℝ) - (1 / 16) ^ (-1 / 2 : ℝ) = -4 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_simplification_l1540_154049


namespace NUMINAMATH_GPT_same_oxidation_state_HNO3_N2O5_l1540_154062

def oxidation_state_HNO3 (H O: Int) : Int := 1 + 1 + (3 * (-2))
def oxidation_state_N2O5 (H O: Int) : Int := (2 * 1) + (5 * (-2))
def oxidation_state_substances_equal : Prop :=
  oxidation_state_HNO3 1 (-2) = oxidation_state_N2O5 1 (-2)

theorem same_oxidation_state_HNO3_N2O5 : oxidation_state_substances_equal :=
  by
  sorry

end NUMINAMATH_GPT_same_oxidation_state_HNO3_N2O5_l1540_154062


namespace NUMINAMATH_GPT_min_sum_of_factors_240_l1540_154033

theorem min_sum_of_factors_240 :
  ∃ a b : ℕ, a * b = 240 ∧ (∀ a' b' : ℕ, a' * b' = 240 → a + b ≤ a' + b') ∧ a + b = 31 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_240_l1540_154033


namespace NUMINAMATH_GPT_isabella_hair_length_l1540_154037

-- Define conditions: original length and doubled length
variable (original_length : ℕ)
variable (doubled_length : ℕ := 36)

-- Theorem: Prove that if the original length doubled equals 36, then the original length is 18.
theorem isabella_hair_length (h : 2 * original_length = doubled_length) : original_length = 18 := by
  sorry

end NUMINAMATH_GPT_isabella_hair_length_l1540_154037


namespace NUMINAMATH_GPT_sum_eq_sqrt_122_l1540_154041

theorem sum_eq_sqrt_122 
  (a b c : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h1 : a^2 + b^2 + c^2 = 58) 
  (h2 : a * b + b * c + c * a = 32) :
  a + b + c = Real.sqrt 122 := 
by
  sorry

end NUMINAMATH_GPT_sum_eq_sqrt_122_l1540_154041


namespace NUMINAMATH_GPT_mean_volume_of_cubes_l1540_154058

theorem mean_volume_of_cubes (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) :
  ((a^3 + b^3 + c^3) / 3) = 135 :=
by
  -- known cube volumes and given edge lengths conditions
  sorry

end NUMINAMATH_GPT_mean_volume_of_cubes_l1540_154058


namespace NUMINAMATH_GPT_number_of_possible_values_of_k_l1540_154006

-- Define the primary conditions and question
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def quadratic_roots_prime (p q k : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 72 ∧ p * q = k

theorem number_of_possible_values_of_k :
  ¬ ∃ k : ℕ, ∃ p q : ℕ, quadratic_roots_prime p q k :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_values_of_k_l1540_154006


namespace NUMINAMATH_GPT_goods_train_speed_l1540_154073

noncomputable def passenger_train_speed := 64 -- in km/h
noncomputable def passing_time := 18 -- in seconds
noncomputable def goods_train_length := 420 -- in meters
noncomputable def relative_speed_kmh := 84 -- in km/h (derived from solution)

theorem goods_train_speed :
  (∃ V_g, relative_speed_kmh = V_g + passenger_train_speed) →
  (goods_train_length / (passing_time / 3600): ℝ) = relative_speed_kmh →
  V_g = 20 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_goods_train_speed_l1540_154073


namespace NUMINAMATH_GPT_binom_10_3_l1540_154065

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end NUMINAMATH_GPT_binom_10_3_l1540_154065


namespace NUMINAMATH_GPT_smallest_int_with_18_divisors_l1540_154092

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end NUMINAMATH_GPT_smallest_int_with_18_divisors_l1540_154092


namespace NUMINAMATH_GPT_soda_cost_l1540_154024

theorem soda_cost (x : ℝ) : 
    (1.5 * 35 + x * (87 - 35) = 78.5) → 
    x = 0.5 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_soda_cost_l1540_154024


namespace NUMINAMATH_GPT_find_f8_l1540_154015

theorem find_f8 (f : ℕ → ℕ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : f 8 = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_f8_l1540_154015


namespace NUMINAMATH_GPT_quadratic_roots_properties_l1540_154046

-- Given the quadratic equation x^2 - 7x + 12 = 0
-- Prove that the absolute value of the difference of the roots is 1
-- Prove that the maximum value of the roots is 4

theorem quadratic_roots_properties :
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → abs (r1 - r2) = 1) ∧ 
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → max r1 r2 = 4) :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_properties_l1540_154046


namespace NUMINAMATH_GPT_units_digit_37_pow_37_l1540_154018

theorem units_digit_37_pow_37 : (37 ^ 37) % 10 = 7 := by
  -- The proof is omitted as per instructions.
  sorry

end NUMINAMATH_GPT_units_digit_37_pow_37_l1540_154018


namespace NUMINAMATH_GPT_trips_Jean_l1540_154013

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end NUMINAMATH_GPT_trips_Jean_l1540_154013


namespace NUMINAMATH_GPT_eldorado_license_plates_count_l1540_154056

theorem eldorado_license_plates_count:
  let letters := 26
  let digits := 10
  let total := (letters ^ 3) * (digits ^ 4)
  total = 175760000 :=
by
  sorry

end NUMINAMATH_GPT_eldorado_license_plates_count_l1540_154056


namespace NUMINAMATH_GPT_gcd_lcm_product_eq_l1540_154059

theorem gcd_lcm_product_eq (a b : ℕ) : gcd a b * lcm a b = a * b := by
  sorry

example : ∃ (a b : ℕ), a = 30 ∧ b = 75 ∧ gcd a b * lcm a b = a * b :=
  ⟨30, 75, rfl, rfl, gcd_lcm_product_eq 30 75⟩

end NUMINAMATH_GPT_gcd_lcm_product_eq_l1540_154059


namespace NUMINAMATH_GPT_sum_of_c_n_l1540_154022

-- Define the sequence {b_n}
def b : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * b n + 3

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := (a n) / (b n + 3)

-- Define the sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i => c i)

-- Theorem to prove
theorem sum_of_c_n : ∀ (n : ℕ), T n = (3 / 2 : ℚ) - ((2 * n + 3) / 2^(n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_c_n_l1540_154022


namespace NUMINAMATH_GPT_transformed_curve_eq_l1540_154069

/-- Given the initial curve equation and the scaling transformation,
    prove that the resulting curve has the transformed equation. -/
theorem transformed_curve_eq 
  (x y x' y' : ℝ)
  (h_curve : x^2 + 9*y^2 = 9)
  (h_transform_x : x' = x)
  (h_transform_y : y' = 3*y) :
  (x')^2 + y'^2 = 9 := 
sorry

end NUMINAMATH_GPT_transformed_curve_eq_l1540_154069


namespace NUMINAMATH_GPT_max_f_l1540_154083

theorem max_f (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x : ℝ, (-1 < x) →  ∀ y : ℝ, (y > -1) → ((1 + y)^a - a*y ≤ 1) :=
sorry

end NUMINAMATH_GPT_max_f_l1540_154083


namespace NUMINAMATH_GPT_option_B_option_D_l1540_154052

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- The maximum value of (y + 1) / (x + 1) is 2 + sqrt(6)
theorem option_B (x y : ℝ) (h : curve_C x y) :
  ∃ k, (y + 1) / (x + 1) = k ∧ k = 2 + Real.sqrt 6 :=
sorry

-- A tangent line through the point (0, √2) on curve C has the equation x - √2 * y + 2 = 0
theorem option_D (h : curve_C 0 (Real.sqrt 2)) :
  ∃ a b c, a * 0 + b * Real.sqrt 2 + c = 0 ∧ c = 2 ∧ a = 1 ∧ b = - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_option_B_option_D_l1540_154052


namespace NUMINAMATH_GPT_original_price_before_discounts_l1540_154000

theorem original_price_before_discounts (P : ℝ) 
  (h : 0.75 * (0.75 * P) = 18) : P = 32 :=
by
  sorry

end NUMINAMATH_GPT_original_price_before_discounts_l1540_154000


namespace NUMINAMATH_GPT_pow_add_div_eq_l1540_154053

   theorem pow_add_div_eq (a b c d e : ℕ) (h1 : b = 2) (h2 : c = 345) (h3 : d = 9) (h4 : e = 8 - 5) : 
     a = b^c + d^e -> a = 2^345 + 729 := 
   by 
     intros 
     sorry
   
end NUMINAMATH_GPT_pow_add_div_eq_l1540_154053


namespace NUMINAMATH_GPT_minimum_connected_components_l1540_154074

/-- We start with two points A, B on a 6*7 lattice grid. We say two points 
  X, Y are connected if one can reflect several times with respect to points A, B 
  and reach from X to Y. Prove that the minimum number of connected components 
  over all choices of A, B is 8. -/
theorem minimum_connected_components (A B : ℕ × ℕ) 
  (hA : A.1 < 6 ∧ A.2 < 7) (hB : B.1 < 6 ∧ B.2 < 7) :
  ∃ k, k = 8 :=
sorry

end NUMINAMATH_GPT_minimum_connected_components_l1540_154074


namespace NUMINAMATH_GPT_perimeter_difference_l1540_154004

-- Definitions for the conditions
def num_stakes_sheep : ℕ := 96
def interval_sheep : ℕ := 10
def num_stakes_horse : ℕ := 82
def interval_horse : ℕ := 20

-- Definition for the perimeters
def perimeter_sheep : ℕ := num_stakes_sheep * interval_sheep
def perimeter_horse : ℕ := num_stakes_horse * interval_horse

-- Definition for the target difference
def target_difference : ℕ := 680

-- The theorem stating the proof problem
theorem perimeter_difference : perimeter_horse - perimeter_sheep = target_difference := by
  sorry

end NUMINAMATH_GPT_perimeter_difference_l1540_154004


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1540_154048

-- Define sets A and B
def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The proof statement
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1540_154048
