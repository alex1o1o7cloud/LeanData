import Mathlib

namespace NUMINAMATH_GPT_product_eq_sum_l2337_233749

variables {x y : ℝ}

theorem product_eq_sum (h : x * y = x + y) (h_ne : y ≠ 1) : x = y / (y - 1) :=
sorry

end NUMINAMATH_GPT_product_eq_sum_l2337_233749


namespace NUMINAMATH_GPT_total_students_in_Lansing_l2337_233721

theorem total_students_in_Lansing:
  (number_of_schools : Nat) → 
  (students_per_school : Nat) → 
  (total_students : Nat) →
  number_of_schools = 25 → 
  students_per_school = 247 → 
  total_students = number_of_schools * students_per_school → 
  total_students = 6175 :=
by
  intros number_of_schools students_per_school total_students h_schools h_students h_total
  rw [h_schools, h_students] at h_total
  exact h_total

end NUMINAMATH_GPT_total_students_in_Lansing_l2337_233721


namespace NUMINAMATH_GPT_average_price_of_fruit_l2337_233773

theorem average_price_of_fruit :
  ∃ (A O : ℕ), A + O = 10 ∧ (40 * A + 60 * (O - 4)) / (A + O - 4) = 50 → 
  (40 * A + 60 * O) / 10 = 54 :=
by
  sorry

end NUMINAMATH_GPT_average_price_of_fruit_l2337_233773


namespace NUMINAMATH_GPT_initial_overs_l2337_233792

theorem initial_overs (initial_run_rate remaining_run_rate target runs initially remaining_overs : ℝ)
    (h_target : target = 282)
    (h_remaining_overs : remaining_overs = 40)
    (h_initial_run_rate : initial_run_rate = 3.6)
    (h_remaining_run_rate : remaining_run_rate = 6.15)
    (h_target_eq : initial_run_rate * initially + remaining_run_rate * remaining_overs = target) :
    initially = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_overs_l2337_233792


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l2337_233701

noncomputable def count_valid_ordered_pairs (a b: ℝ) : Prop :=
  ∃ (x y : ℤ), a * (x : ℝ) + b * (y : ℝ) = 2 ∧ x^2 + y^2 = 65

theorem number_of_ordered_pairs : ∃ s : Finset (ℝ × ℝ), s.card = 128 ∧ ∀ (p : ℝ × ℝ), p ∈ s ↔ count_valid_ordered_pairs p.1 p.2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l2337_233701


namespace NUMINAMATH_GPT_barrel_to_cask_ratio_l2337_233735

theorem barrel_to_cask_ratio
  (k : ℕ) -- k is the multiple
  (B C : ℕ) -- B is the amount a barrel can store, C is the amount a cask can store
  (h1 : C = 20) -- C stores 20 gallons
  (h2 : B = k * C + 3) -- A barrel stores 3 gallons more than k times the amount a cask stores
  (h3 : 4 * B + C = 172) -- The total storage capacity is 172 gallons
  : B / C = 19 / 10 :=
sorry

end NUMINAMATH_GPT_barrel_to_cask_ratio_l2337_233735


namespace NUMINAMATH_GPT_min_tan_of_acute_angle_l2337_233744

def is_ocular_ray (u : ℚ) (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ u = x / y

def acute_angle_tangent (u v : ℚ) : ℚ :=
  |(u - v) / (1 + u * v)|

theorem min_tan_of_acute_angle :
  ∃ θ : ℚ, (∀ u v : ℚ, (∃ x1 y1 x2 y2 : ℕ, is_ocular_ray u x1 y1 ∧ is_ocular_ray v x2 y2 ∧ u ≠ v) 
  → acute_angle_tangent u v ≥ θ) ∧ θ = 1 / 722 :=
sorry

end NUMINAMATH_GPT_min_tan_of_acute_angle_l2337_233744


namespace NUMINAMATH_GPT_mina_numbers_l2337_233762

theorem mina_numbers (a b : ℤ) (h1 : 3 * a + 4 * b = 140) (h2 : a = 20 ∨ b = 20) : a = 20 ∧ b = 20 :=
by
  sorry

end NUMINAMATH_GPT_mina_numbers_l2337_233762


namespace NUMINAMATH_GPT_shaded_area_l2337_233718

noncomputable def area_of_shaded_region (AB : ℝ) (pi_approx : ℝ) : ℝ :=
  let R := AB / 2
  let r := R / 2
  let A_large := (1/2) * pi_approx * R^2
  let A_small := (1/2) * pi_approx * r^2
  2 * A_large - 4 * A_small

theorem shaded_area (h : area_of_shaded_region 40 3.14 = 628) : true :=
  sorry

end NUMINAMATH_GPT_shaded_area_l2337_233718


namespace NUMINAMATH_GPT_airsickness_related_to_gender_l2337_233717

def a : ℕ := 28
def b : ℕ := 28
def c : ℕ := 28
def d : ℕ := 56
def n : ℕ := 140

def contingency_relation (a b c d n K2 : ℕ) : Prop := 
  let numerator := n * (a * d - b * c)^2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  K2 > 3841 / 1000

-- Goal statement for the proof
theorem airsickness_related_to_gender :
  contingency_relation a b c d n 3888 :=
  sorry

end NUMINAMATH_GPT_airsickness_related_to_gender_l2337_233717


namespace NUMINAMATH_GPT_seq_a_eval_a4_l2337_233764

theorem seq_a_eval_a4 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) : a 4 = 15 :=
sorry

end NUMINAMATH_GPT_seq_a_eval_a4_l2337_233764


namespace NUMINAMATH_GPT_ratio_of_Steve_speeds_l2337_233783

noncomputable def Steve_speeds_ratio : Nat := 
  let d := 40 -- distance in km
  let T := 6  -- total time in hours
  let v2 := 20 -- speed on the way back in km/h
  let t2 := d / v2 -- time taken on the way back in hours
  let t1 := T - t2 -- time taken on the way to work in hours
  let v1 := d / t1 -- speed on the way to work in km/h
  v2 / v1

theorem ratio_of_Steve_speeds :
  Steve_speeds_ratio = 2 := 
  by sorry

end NUMINAMATH_GPT_ratio_of_Steve_speeds_l2337_233783


namespace NUMINAMATH_GPT_problem_intersecting_lines_l2337_233787

theorem problem_intersecting_lines (c d : ℝ) :
  (3 : ℝ) = (1 / 3 : ℝ) * (6 : ℝ) + c ∧ (6 : ℝ) = (1 / 3 : ℝ) * (3 : ℝ) + d → c + d = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_problem_intersecting_lines_l2337_233787


namespace NUMINAMATH_GPT_shoes_difference_l2337_233777

theorem shoes_difference : 
  ∀ (Scott_shoes Anthony_shoes Jim_shoes : ℕ), 
  Scott_shoes = 7 → 
  Anthony_shoes = 3 * Scott_shoes → 
  Jim_shoes = Anthony_shoes - 2 → 
  Anthony_shoes - Jim_shoes = 2 :=
by
  intros Scott_shoes Anthony_shoes Jim_shoes 
  intros h1 h2 h3 
  sorry

end NUMINAMATH_GPT_shoes_difference_l2337_233777


namespace NUMINAMATH_GPT_minimize_expression_l2337_233736

theorem minimize_expression :
  ∀ n : ℕ, 0 < n → (n = 6 ↔ ∀ m : ℕ, 0 < m → (n ≤ (2 * (m + 9))/(m))) := 
by
  sorry

end NUMINAMATH_GPT_minimize_expression_l2337_233736


namespace NUMINAMATH_GPT_k_zero_only_solution_l2337_233710

noncomputable def polynomial_factorable (k : ℤ) : Prop :=
  ∃ (A B C D E F : ℤ), (A * D = 1) ∧ (B * E = 4) ∧ (A * E + B * D = k) ∧ (A * F + C * D = 1) ∧ (C * F = -k)

theorem k_zero_only_solution : ∀ k : ℤ, polynomial_factorable k ↔ k = 0 :=
by 
  sorry

end NUMINAMATH_GPT_k_zero_only_solution_l2337_233710


namespace NUMINAMATH_GPT_sum_is_1716_l2337_233732

-- Given conditions:
variables (a b c d : ℤ)
variable (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h_roots1 : ∀ t, t * t - 12 * a * t - 13 * b = 0 ↔ t = c ∨ t = d)
variable (h_roots2 : ∀ t, t * t - 12 * c * t - 13 * d = 0 ↔ t = a ∨ t = b)

-- Prove the desired sum of the constants:
theorem sum_is_1716 : a + b + c + d = 1716 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_1716_l2337_233732


namespace NUMINAMATH_GPT_cindy_correct_method_l2337_233739

theorem cindy_correct_method (x : ℝ) (h : (x - 7) / 5 = 15) : (x - 5) / 7 = 11 := 
by
  sorry

end NUMINAMATH_GPT_cindy_correct_method_l2337_233739


namespace NUMINAMATH_GPT_cost_of_dvd_player_l2337_233798

/-- The ratio of the cost of a DVD player to the cost of a movie is 9:2.
    A DVD player costs $63 more than a movie.
    Prove that the cost of the DVD player is $81. -/
theorem cost_of_dvd_player 
(D M : ℝ)
(h1 : D = (9 / 2) * M)
(h2 : D = M + 63) : 
D = 81 := 
sorry

end NUMINAMATH_GPT_cost_of_dvd_player_l2337_233798


namespace NUMINAMATH_GPT_smallest_ab_41503_539_l2337_233754

noncomputable def find_smallest_ab : (ℕ × ℕ) :=
  let a := 41503
  let b := 539
  (a, b)

theorem smallest_ab_41503_539 (a b : ℕ) (h : 7 * a^3 = 11 * b^5) (ha : a > 0) (hb : b > 0) :
  (a = 41503 ∧ b = 539) :=
  by
    -- Add sorry to skip the proof
    sorry

end NUMINAMATH_GPT_smallest_ab_41503_539_l2337_233754


namespace NUMINAMATH_GPT_power_identity_l2337_233704

theorem power_identity (a b : ℕ) (R S : ℕ) (hR : R = 2^a) (hS : S = 5^b) : 
    20^(a * b) = R^(2 * b) * S^a := 
by 
    -- Insert the proof here
    sorry

end NUMINAMATH_GPT_power_identity_l2337_233704


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_still_holds_when_not_positive_l2337_233703

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a > 0 ∧ b > 0) → (b / a + a / b ≥ 2) :=
by 
  sorry

theorem still_holds_when_not_positive (a b : ℝ) (h1 : a ≤ 0 ∨ b ≤ 0) :
  (b / a + a / b ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_still_holds_when_not_positive_l2337_233703


namespace NUMINAMATH_GPT_cost_of_one_of_the_shirts_l2337_233796

theorem cost_of_one_of_the_shirts
    (total_cost : ℕ) 
    (cost_two_shirts : ℕ) 
    (num_equal_shirts : ℕ) 
    (cost_of_shirt : ℕ) :
    total_cost = 85 → 
    cost_two_shirts = 20 → 
    num_equal_shirts = 3 → 
    cost_of_shirt = (total_cost - 2 * cost_two_shirts) / num_equal_shirts → 
    cost_of_shirt = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_of_one_of_the_shirts_l2337_233796


namespace NUMINAMATH_GPT_max_volume_cylinder_l2337_233725

theorem max_volume_cylinder (x : ℝ) (h1 : x > 0) (h2 : x < 10) : 
  (∀ x, 0 < x ∧ x < 10 → ∃ max_v, max_v = (4 * (10^3) * Real.pi) / 27) ∧ 
  ∃ x, x = 20/3 := 
by
  sorry

end NUMINAMATH_GPT_max_volume_cylinder_l2337_233725


namespace NUMINAMATH_GPT_minimum_discount_l2337_233712

variable (C P : ℝ) (r x : ℝ)

def microwave_conditions := 
  C = 1000 ∧ 
  P = 1500 ∧ 
  r = 0.02 ∧ 
  P * (x / 10) ≥ C * (1 + r)

theorem minimum_discount : ∃ x, microwave_conditions C P r x ∧ x ≥ 6.8 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_discount_l2337_233712


namespace NUMINAMATH_GPT_inequality_solution_l2337_233740

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) ↔
  (1 < x ∧ x < 2 ∨ 3 < x ∧ x < 6) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2337_233740


namespace NUMINAMATH_GPT_parsnip_box_fullness_l2337_233768

theorem parsnip_box_fullness (capacity : ℕ) (fraction_full : ℚ) (avg_boxes : ℕ) (avg_parsnips : ℕ) :
  capacity = 20 →
  fraction_full = 3 / 4 →
  avg_boxes = 20 →
  avg_parsnips = 350 →
  ∃ (full_boxes : ℕ) (non_full_boxes : ℕ) (parsnips_in_full_boxes : ℕ) (parsnips_in_non_full_boxes : ℕ)
    (avg_fullness_non_full_boxes : ℕ),
    full_boxes = fraction_full * avg_boxes ∧
    non_full_boxes = avg_boxes - full_boxes ∧
    parsnips_in_full_boxes = full_boxes * capacity ∧
    parsnips_in_non_full_boxes = avg_parsnips - parsnips_in_full_boxes ∧
    avg_fullness_non_full_boxes = parsnips_in_non_full_boxes / non_full_boxes ∧
    avg_fullness_non_full_boxes = 10 :=
by
  sorry

end NUMINAMATH_GPT_parsnip_box_fullness_l2337_233768


namespace NUMINAMATH_GPT_possible_values_of_k_l2337_233788

-- Definition of the proposition
def proposition (k : ℝ) : Prop :=
  ∃ x : ℝ, (k^2 - 1) * x^2 + 4 * (1 - k) * x + 3 ≤ 0

-- The main statement to prove in Lean 4
theorem possible_values_of_k (k : ℝ) : ¬ proposition k ↔ (k = 1 ∨ (1 < k ∧ k < 7)) :=
by 
  sorry

end NUMINAMATH_GPT_possible_values_of_k_l2337_233788


namespace NUMINAMATH_GPT_tan_value_l2337_233784

open Real

theorem tan_value (α : ℝ) (h : sin (5 * π / 6 - α) = sqrt 3 * cos (α + π / 6)) : 
  tan (α + π / 6) = sqrt 3 := 
  sorry

end NUMINAMATH_GPT_tan_value_l2337_233784


namespace NUMINAMATH_GPT_no_real_solutions_for_equation_l2337_233727

theorem no_real_solutions_for_equation : ¬ (∃ x : ℝ, x + Real.sqrt (2 * x - 6) = 5) :=
sorry

end NUMINAMATH_GPT_no_real_solutions_for_equation_l2337_233727


namespace NUMINAMATH_GPT_point_on_line_l2337_233724

theorem point_on_line (s : ℝ) : 
  (∃ b : ℝ, ∀ x y : ℝ, (y = 3 * x + b) → 
    ((2 = x ∧ y = 8) ∨ (4 = x ∧ y = 14) ∨ (6 = x ∧ y = 20) ∨ (35 = x ∧ y = s))) → s = 107 :=
by
  sorry

end NUMINAMATH_GPT_point_on_line_l2337_233724


namespace NUMINAMATH_GPT_equal_books_for_students_l2337_233789

-- Define the conditions
def num_girls : ℕ := 15
def num_boys : ℕ := 10
def total_books : ℕ := 375
def books_for_girls : ℕ := 225
def books_for_boys : ℕ := total_books - books_for_girls -- Calculate books for boys

-- Define the theorem
theorem equal_books_for_students :
  books_for_girls / num_girls = 15 ∧ books_for_boys / num_boys = 15 :=
by
  sorry

end NUMINAMATH_GPT_equal_books_for_students_l2337_233789


namespace NUMINAMATH_GPT_find_x0_l2337_233746

-- Defining the function f
def f (a c x : ℝ) : ℝ := a * x^2 + c

-- Defining the integral condition
def integral_condition (a c x0 : ℝ) : Prop :=
  (∫ x in (0 : ℝ)..(1 : ℝ), f a c x) = f a c x0

-- Proving the main statement
theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (h_range : 0 ≤ x0 ∧ x0 ≤ 1) (h_integral : integral_condition a c x0) :
  x0 = Real.sqrt (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_x0_l2337_233746


namespace NUMINAMATH_GPT_total_spent_l2337_233741

theorem total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ)
  (h1 : deck_price = 8)
  (h2 : victor_decks = 6)
  (h3 : friend_decks = 2) :
  deck_price * victor_decks + deck_price * friend_decks = 64 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_l2337_233741


namespace NUMINAMATH_GPT_evaluate_expression_l2337_233734

theorem evaluate_expression : 
  (2 ^ 2003 * 3 ^ 2002 * 5) / (6 ^ 2003) = (5 / 3) :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2337_233734


namespace NUMINAMATH_GPT_jason_hours_saturday_l2337_233761

def hours_after_school (x : ℝ) : ℝ := 4 * x
def hours_saturday (y : ℝ) : ℝ := 6 * y

theorem jason_hours_saturday 
  (x y : ℝ) 
  (total_hours : x + y = 18) 
  (total_earnings : 4 * x + 6 * y = 88) : 
  y = 8 :=
by 
  sorry

end NUMINAMATH_GPT_jason_hours_saturday_l2337_233761


namespace NUMINAMATH_GPT_original_price_of_sweater_l2337_233752

theorem original_price_of_sweater (sold_price : ℝ) (discount : ℝ) (original_price : ℝ) 
    (h1 : sold_price = 120) (h2 : discount = 0.40) (h3: (1 - discount) * original_price = sold_price) : 
    original_price = 200 := by 
  sorry

end NUMINAMATH_GPT_original_price_of_sweater_l2337_233752


namespace NUMINAMATH_GPT_speed_in_still_water_l2337_233759

-- Define variables for speed of the boy in still water and speed of the stream.
variables (v s : ℝ)

-- Define the conditions as Lean statements
def downstream_condition (v s : ℝ) : Prop := (v + s) * 7 = 91
def upstream_condition (v s : ℝ) : Prop := (v - s) * 7 = 21

-- The theorem to prove that the speed of the boy in still water is 8 km/h given the conditions
theorem speed_in_still_water
  (h1 : downstream_condition v s)
  (h2 : upstream_condition v s) :
  v = 8 := 
sorry

end NUMINAMATH_GPT_speed_in_still_water_l2337_233759


namespace NUMINAMATH_GPT_no_integer_roots_l2337_233782
open Polynomial

theorem no_integer_roots (p : Polynomial ℤ) (c1 c2 c3 : ℤ) (h1 : p.eval c1 = 1) (h2 : p.eval c2 = 1) (h3 : p.eval c3 = 1) (h_distinct : c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3) : ¬ ∃ a : ℤ, p.eval a = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_l2337_233782


namespace NUMINAMATH_GPT_YooSeung_has_108_marbles_l2337_233769

def YoungSoo_marble_count : ℕ := 21
def HanSol_marble_count : ℕ := YoungSoo_marble_count + 15
def YooSeung_marble_count : ℕ := 3 * HanSol_marble_count
def total_marble_count : ℕ := YoungSoo_marble_count + HanSol_marble_count + YooSeung_marble_count

theorem YooSeung_has_108_marbles 
  (h1 : YooSeung_marble_count = 3 * (YoungSoo_marble_count + 15))
  (h2 : HanSol_marble_count = YoungSoo_marble_count + 15)
  (h3 : total_marble_count = 165) :
  YooSeung_marble_count = 108 :=
by sorry

end NUMINAMATH_GPT_YooSeung_has_108_marbles_l2337_233769


namespace NUMINAMATH_GPT_smallest_whole_number_for_inequality_l2337_233790

theorem smallest_whole_number_for_inequality:
  ∃ (x : ℕ), (2 : ℝ) / 5 + (x : ℝ) / 9 > 1 ∧ ∀ (y : ℕ), (2 : ℝ) / 5 + (y : ℝ) / 9 > 1 → x ≤ y :=
by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_for_inequality_l2337_233790


namespace NUMINAMATH_GPT_units_digit_7_pow_2023_l2337_233771

theorem units_digit_7_pow_2023 :
  (7 ^ 2023) % 10 = 3 := 
sorry

end NUMINAMATH_GPT_units_digit_7_pow_2023_l2337_233771


namespace NUMINAMATH_GPT_linda_fraction_savings_l2337_233743

theorem linda_fraction_savings (savings tv_cost : ℝ) (f : ℝ) 
  (h1 : savings = 800) 
  (h2 : tv_cost = 200) 
  (h3 : f * savings + tv_cost = savings) : 
  f = 3 / 4 := 
sorry

end NUMINAMATH_GPT_linda_fraction_savings_l2337_233743


namespace NUMINAMATH_GPT_percentage_difference_l2337_233767

theorem percentage_difference :
  ((75 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25) = 10 := 
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l2337_233767


namespace NUMINAMATH_GPT_number_of_functions_satisfying_conditions_l2337_233797

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_conditions (f : ℕ → ℕ) : Prop :=
  (∀ s ∈ S, f (f (f s)) = s) ∧ (∀ s ∈ S, (f s - s) % 3 ≠ 0)

theorem number_of_functions_satisfying_conditions :
  (∃ (f : ℕ → ℕ), f_conditions f) ∧ (∃! (n : ℕ), n = 288) :=
by
  sorry

end NUMINAMATH_GPT_number_of_functions_satisfying_conditions_l2337_233797


namespace NUMINAMATH_GPT_find_natural_number_l2337_233760

theorem find_natural_number (n : ℕ) (h1 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n ∨ d = 3 ∨ d = 5 ∨ d = 9 ∨ d = 15)
  (h2 : 1 + 3 + 5 + 9 + 15 + n = 78) : n = 45 := sorry

end NUMINAMATH_GPT_find_natural_number_l2337_233760


namespace NUMINAMATH_GPT_opening_night_customers_l2337_233772

theorem opening_night_customers
  (matinee_tickets : ℝ := 5)
  (evening_tickets : ℝ := 7)
  (opening_night_tickets : ℝ := 10)
  (popcorn_cost : ℝ := 10)
  (num_matinee_customers : ℝ := 32)
  (num_evening_customers : ℝ := 40)
  (total_revenue : ℝ := 1670) :
  ∃ x : ℝ, 
    (matinee_tickets * num_matinee_customers + 
    evening_tickets * num_evening_customers + 
    opening_night_tickets * x + 
    popcorn_cost * (num_matinee_customers + num_evening_customers + x) / 2 = total_revenue) 
    ∧ x = 58 := 
by
  use 58
  sorry

end NUMINAMATH_GPT_opening_night_customers_l2337_233772


namespace NUMINAMATH_GPT_calculate_value_l2337_233799

theorem calculate_value : (3^2 * 5^4 * 7^2) / 7 = 39375 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l2337_233799


namespace NUMINAMATH_GPT_sum_x_coordinates_common_points_l2337_233747

-- Definition of the equivalence relation modulo 9
def equiv_mod (a b n : ℤ) : Prop := ∃ k : ℤ, a = b + n * k

-- Definitions of the given conditions
def graph1 (x y : ℤ) : Prop := equiv_mod y (3 * x + 6) 9
def graph2 (x y : ℤ) : Prop := equiv_mod y (7 * x + 3) 9

-- Definition of when two graphs intersect
def points_in_common (x y : ℤ) : Prop := graph1 x y ∧ graph2 x y

-- Proof that the sum of the x-coordinates of the points in common is 3
theorem sum_x_coordinates_common_points : 
  ∃ x y, points_in_common x y ∧ (x = 3) := 
sorry

end NUMINAMATH_GPT_sum_x_coordinates_common_points_l2337_233747


namespace NUMINAMATH_GPT_find_average_of_xyz_l2337_233794

variable (x y z k : ℝ)

def system_of_equations : Prop :=
  (2 * x + y - z = 26) ∧
  (x + 2 * y + z = 10) ∧
  (x - y + z = k)

theorem find_average_of_xyz (h : system_of_equations x y z k) : 
  (x + y + z) / 3 = (36 + k) / 6 :=
by sorry

end NUMINAMATH_GPT_find_average_of_xyz_l2337_233794


namespace NUMINAMATH_GPT_correct_operation_c_l2337_233753

theorem correct_operation_c (a b : ℝ) :
  ¬ (a^2 + a^2 = 2 * a^4)
  ∧ ¬ ((-3 * a * b^2)^2 = -6 * a^2 * b^4)
  ∧ a^6 / (-a)^2 = a^4
  ∧ ¬ ((a - b)^2 = a^2 - b^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_c_l2337_233753


namespace NUMINAMATH_GPT_pizza_fraction_eaten_l2337_233791

theorem pizza_fraction_eaten :
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  (a * (1 - r ^ n) / (1 - r)) = 63 / 128 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  sorry

end NUMINAMATH_GPT_pizza_fraction_eaten_l2337_233791


namespace NUMINAMATH_GPT_no_absolute_winner_prob_l2337_233726

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end NUMINAMATH_GPT_no_absolute_winner_prob_l2337_233726


namespace NUMINAMATH_GPT_earliest_year_exceeds_target_l2337_233723

/-- Define the initial deposit and annual interest rate -/
def initial_deposit : ℝ := 100000
def annual_interest_rate : ℝ := 0.10

/-- Define the amount in the account after n years -/
def amount_after_years (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

/-- Define the target amount to exceed -/
def target_amount : ℝ := 150100

/-- Define the year the initial deposit is made -/
def initial_year : ℕ := 2021

/-- Prove that the earliest year the amount exceeds the target is 2026 -/
theorem earliest_year_exceeds_target :
  ∃ n : ℕ, n > 0 ∧ amount_after_years initial_deposit annual_interest_rate n > target_amount ∧ (initial_year + n) = 2026 :=
by
  sorry

end NUMINAMATH_GPT_earliest_year_exceeds_target_l2337_233723


namespace NUMINAMATH_GPT_ratio_female_to_male_l2337_233705

variable (m f : ℕ)

-- Average ages given in the conditions
def avg_female_age : ℕ := 35
def avg_male_age : ℕ := 45
def avg_total_age : ℕ := 40

-- Total ages based on number of members
def total_female_age (f : ℕ) : ℕ := avg_female_age * f
def total_male_age (m : ℕ) : ℕ := avg_male_age * m
def total_age (f m : ℕ) : ℕ := total_female_age f + total_male_age m

-- Equation based on average age of all members
def avg_age_eq (f m : ℕ) : Prop :=
  total_age f m / (f + m) = avg_total_age

theorem ratio_female_to_male : avg_age_eq f m → f = m :=
by
  sorry

end NUMINAMATH_GPT_ratio_female_to_male_l2337_233705


namespace NUMINAMATH_GPT_prove_side_c_prove_sin_B_prove_area_circumcircle_l2337_233785

-- Define the given conditions
def triangle_ABC (a b A : ℝ) : Prop :=
  a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3

-- Prove that side 'c' is equal to 3
theorem prove_side_c (h : triangle_ABC a b A) : c = 3 := by
  sorry

-- Prove that sin B is equal to \frac{\sqrt{21}}{7}
theorem prove_sin_B (h : triangle_ABC a b A) : Real.sin B = Real.sqrt 21 / 7 := by
  sorry

-- Prove that the area of the circumcircle is \frac{7\pi}{3}
theorem prove_area_circumcircle (h : triangle_ABC a b A) (R : ℝ) : 
  let circumcircle_area := Real.pi * R^2
  circumcircle_area = 7 * Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_prove_side_c_prove_sin_B_prove_area_circumcircle_l2337_233785


namespace NUMINAMATH_GPT_initial_action_figures_l2337_233793

theorem initial_action_figures (x : ℕ) (h1 : x + 2 = 10) : x = 8 := 
by sorry

end NUMINAMATH_GPT_initial_action_figures_l2337_233793


namespace NUMINAMATH_GPT_karthik_weight_average_l2337_233765

theorem karthik_weight_average
  (weight : ℝ)
  (hKarthik: 55 < weight )
  (hBrother: weight < 58 )
  (hFather : 56 < weight )
  (hSister: 54 < weight ∧ weight < 57) :
  (56 < weight ∧ weight < 57) → (weight = 56.5) :=
by 
  sorry

end NUMINAMATH_GPT_karthik_weight_average_l2337_233765


namespace NUMINAMATH_GPT_solve_inequality_l2337_233702

open Real

theorem solve_inequality (a : ℝ) :
  ((a < 0 ∨ a > 1) → (∀ x, a < x ∧ x < a^2 ↔ (x - a) * (x - a^2) < 0)) ∧
  ((0 < a ∧ a < 1) → (∀ x, a^2 < x ∧ x < a ↔ (x - a) * (x - a^2) < 0)) ∧
  ((a = 0 ∨ a = 1) → (∀ x, ¬((x - a) * (x - a^2) < 0))) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2337_233702


namespace NUMINAMATH_GPT_total_profit_equals_254000_l2337_233700

-- Definitions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 6000
def investment_D : ℕ := 10000

def time_A : ℕ := 12
def time_B : ℕ := 8
def time_C : ℕ := 6
def time_D : ℕ := 9

def capital_months (investment : ℕ) (time : ℕ) : ℕ := investment * time

-- Given conditions
def A_capital_months := capital_months investment_A time_A
def B_capital_months := capital_months investment_B time_B
def C_capital_months := capital_months investment_C time_C
def D_capital_months := capital_months investment_D time_D

def total_capital_months : ℕ := A_capital_months + B_capital_months + C_capital_months + D_capital_months

def C_profit : ℕ := 36000

-- Proportion equation
def total_profit (C_capital_months : ℕ) (total_capital_months : ℕ) (C_profit : ℕ) : ℕ :=
  (C_profit * total_capital_months) / C_capital_months

-- Theorem statement
theorem total_profit_equals_254000 : total_profit C_capital_months total_capital_months C_profit = 254000 := by
  sorry

end NUMINAMATH_GPT_total_profit_equals_254000_l2337_233700


namespace NUMINAMATH_GPT_smallest_sum_of_consecutive_primes_divisible_by_5_l2337_233780

-- Define a predicate for consecutive prime numbers
def is_consecutive_primes (a b c d : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧
  (b = a + 1 ∨ b = a + 2) ∧
  (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2)

-- Define the main problem statement
theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (a b c d : ℕ), is_consecutive_primes a b c d ∧ (a + b + c + d) % 5 = 0 ∧ ∀ (w x y z : ℕ), is_consecutive_primes w x y z ∧ (w + x + y + z) % 5 = 0 → a + b + c + d ≤ w + x + y + z :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_consecutive_primes_divisible_by_5_l2337_233780


namespace NUMINAMATH_GPT_triple_angle_l2337_233778

theorem triple_angle (α : ℝ) : 3 * α = α + α + α := 
by sorry

end NUMINAMATH_GPT_triple_angle_l2337_233778


namespace NUMINAMATH_GPT_at_least_one_not_less_than_2_l2337_233722

theorem at_least_one_not_less_than_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_2_l2337_233722


namespace NUMINAMATH_GPT_calculation_is_correct_l2337_233751

theorem calculation_is_correct : 450 / (6 * 5 - 10 / 2) = 18 :=
by {
  -- Let me provide an outline for solving this problem
  -- (6 * 5 - 10 / 2) must be determined first
  -- After that substituted into the fraction
  sorry
}

end NUMINAMATH_GPT_calculation_is_correct_l2337_233751


namespace NUMINAMATH_GPT_find_seating_capacity_l2337_233719

noncomputable def seating_capacity (buses : ℕ) (students_left : ℤ) : ℤ :=
  buses * 40 + students_left

theorem find_seating_capacity :
  (seating_capacity 4 30) = (seating_capacity 5 (-10)) :=
by
  -- Proof is not required, hence omitted.
  sorry

end NUMINAMATH_GPT_find_seating_capacity_l2337_233719


namespace NUMINAMATH_GPT_gcf_of_294_and_108_l2337_233729

theorem gcf_of_294_and_108 : Nat.gcd 294 108 = 6 :=
by
  -- We are given numbers 294 and 108
  -- Their prime factorizations are 294 = 2 * 3 * 7^2 and 108 = 2^2 * 3^3
  -- The minimum power of the common prime factors are 2^1 and 3^1
  -- Thus, the GCF by multiplying these factors is 2^1 * 3^1 = 6
  sorry

end NUMINAMATH_GPT_gcf_of_294_and_108_l2337_233729


namespace NUMINAMATH_GPT_percentage_taxed_l2337_233713

theorem percentage_taxed (T : ℝ) (H1 : 3840 = T * (P : ℝ)) (H2 : 480 = 0.25 * T * (P : ℝ)) : P = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_percentage_taxed_l2337_233713


namespace NUMINAMATH_GPT_find_y_l2337_233745

def operation (x y : ℝ) : ℝ := 5 * x - 4 * y + 3 * x * y

theorem find_y : ∃ y : ℝ, operation 4 y = 21 ∧ y = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_find_y_l2337_233745


namespace NUMINAMATH_GPT_seventeen_divides_l2337_233731

theorem seventeen_divides (a b : ℤ) (h : 17 ∣ (2 * a + 3 * b)) : 17 ∣ (9 * a + 5 * b) :=
sorry

end NUMINAMATH_GPT_seventeen_divides_l2337_233731


namespace NUMINAMATH_GPT_find_CD_l2337_233709

theorem find_CD (C D : ℚ) :
  (∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 → (7 * x - 4) / (x ^ 2 - 9 * x - 36) = C / (x - 12) + D / (x + 3))
  → C = 16 / 3 ∧ D = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_CD_l2337_233709


namespace NUMINAMATH_GPT_correct_option_is_d_l2337_233714

theorem correct_option_is_d (x : ℚ) : -x^3 = (-x)^3 :=
sorry

end NUMINAMATH_GPT_correct_option_is_d_l2337_233714


namespace NUMINAMATH_GPT_tan_alpha_sin_cos_half_alpha_l2337_233781

variable (α : ℝ)

-- Conditions given in the problem
def cond1 : Real.sin α = 1 / 3 := sorry
def cond2 : 0 < α ∧ α < Real.pi := sorry

-- Lean proof that given the conditions, the solutions are as follows:
theorem tan_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = Real.sqrt 2 / 4 ∨ Real.tan α = - Real.sqrt 2 / 4 := sorry

theorem sin_cos_half_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.sin (α / 2) + Real.cos (α / 2) = 2 * Real.sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_tan_alpha_sin_cos_half_alpha_l2337_233781


namespace NUMINAMATH_GPT_average_bowling_score_l2337_233716

theorem average_bowling_score 
    (gretchen_score : ℕ) (mitzi_score : ℕ) (beth_score : ℕ)
    (gretchen_eq : gretchen_score = 120)
    (mitzi_eq : mitzi_score = 113)
    (beth_eq : beth_score = 85) :
    (gretchen_score + mitzi_score + beth_score) / 3 = 106 := 
by
  sorry

end NUMINAMATH_GPT_average_bowling_score_l2337_233716


namespace NUMINAMATH_GPT_same_terminal_side_l2337_233755

theorem same_terminal_side (k : ℤ): ∃ k : ℤ, 1303 = k * 360 - 137 := by
  -- Proof left as an exercise.
  sorry

end NUMINAMATH_GPT_same_terminal_side_l2337_233755


namespace NUMINAMATH_GPT_find_digits_sum_l2337_233706

theorem find_digits_sum (A B : ℕ) (h1 : A < 10) (h2 : B < 10) 
  (h3 : (A = 6) ∧ (B = 6))
  (h4 : (100 * A + 44610 + B) % 72 = 0) : A + B = 12 := 
by
  sorry

end NUMINAMATH_GPT_find_digits_sum_l2337_233706


namespace NUMINAMATH_GPT_infinite_series_sum_l2337_233757

theorem infinite_series_sum :
  ∑' n : ℕ, (1 / (n.succ * (n.succ + 2))) = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_infinite_series_sum_l2337_233757


namespace NUMINAMATH_GPT_inequality_solution_l2337_233756

theorem inequality_solution :
  {x : ℝ | (3 * x - 9) * (x - 4) / (x - 1) ≥ 0} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 3} ∪ {x : ℝ | x ≥ 4} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2337_233756


namespace NUMINAMATH_GPT_relationship_among_abc_l2337_233728

theorem relationship_among_abc (e1 e2 : ℝ) (h1 : 0 ≤ e1) (h2 : e1 < 1) (h3 : e2 > 1) :
  let a := 3 ^ e1
  let b := 2 ^ (-e2)
  let c := Real.sqrt 5
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l2337_233728


namespace NUMINAMATH_GPT_ravi_overall_profit_l2337_233738

-- Define the cost price of the refrigerator and the mobile phone
def cost_price_refrigerator : ℝ := 15000
def cost_price_mobile_phone : ℝ := 8000

-- Define the loss percentage for the refrigerator and the profit percentage for the mobile phone
def loss_percentage_refrigerator : ℝ := 0.05
def profit_percentage_mobile_phone : ℝ := 0.10

-- Calculate the loss amount and the selling price of the refrigerator
def loss_amount_refrigerator : ℝ := loss_percentage_refrigerator * cost_price_refrigerator
def selling_price_refrigerator : ℝ := cost_price_refrigerator - loss_amount_refrigerator

-- Calculate the profit amount and the selling price of the mobile phone
def profit_amount_mobile_phone : ℝ := profit_percentage_mobile_phone * cost_price_mobile_phone
def selling_price_mobile_phone : ℝ := cost_price_mobile_phone + profit_amount_mobile_phone

-- Calculate the total cost price and the total selling price
def total_cost_price : ℝ := cost_price_refrigerator + cost_price_mobile_phone
def total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone

-- Calculate the overall profit or loss
def overall_profit_or_loss : ℝ := total_selling_price - total_cost_price

theorem ravi_overall_profit : overall_profit_or_loss = 50 := 
by
  sorry

end NUMINAMATH_GPT_ravi_overall_profit_l2337_233738


namespace NUMINAMATH_GPT_length_side_AB_is_4_l2337_233715

-- Defining a triangle ABC with area 6
variables {A B C K L Q : Type*}
variables {side_AB : Float} {ratio_K : Float} {ratio_L : Float} {dist_Q : Float}
variables (area_ABC : ℝ := 6) (ratio_AK_BK : ℝ := 2 / 3) (ratio_AL_LC : ℝ := 5 / 3)
variables (dist_Q_to_AB : ℝ := 1.5)

theorem length_side_AB_is_4 : 
  side_AB = 4 → 
  (area_ABC = 6 ∧ ratio_AK_BK = 2 / 3 ∧ ratio_AL_LC = 5 / 3 ∧ dist_Q_to_AB = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_length_side_AB_is_4_l2337_233715


namespace NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l2337_233779

namespace SetTheory

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def C_UA : Set ℕ := {3, 4, 5}

theorem complement_of_A_with_respect_to_U :
  (U \ A) = C_UA := by
  sorry

end SetTheory

end NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l2337_233779


namespace NUMINAMATH_GPT_probability_left_oar_works_l2337_233786

structure Oars where
  P_L : ℝ -- Probability that the left oar works
  P_R : ℝ -- Probability that the right oar works
  
def independent_prob (o : Oars) : Prop :=
  o.P_L = o.P_R ∧ (1 - o.P_L) * (1 - o.P_R) = 0.16

theorem probability_left_oar_works (o : Oars) (h1 : independent_prob o) (h2 : 1 - (1 - o.P_L) * (1 - o.P_R) = 0.84) : o.P_L = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_probability_left_oar_works_l2337_233786


namespace NUMINAMATH_GPT_YaoMing_stride_impossible_l2337_233730

-- Defining the conditions as Lean definitions.
def XiaoMing_14_years_old (current_year : ℕ) : Prop := current_year = 14
def sum_of_triangle_angles (angles : ℕ) : Prop := angles = 180
def CCTV5_broadcasting_basketball_game : Prop := ∃ t : ℕ, true -- Random event placeholder
def YaoMing_stride (stride_length : ℕ) : Prop := stride_length = 10

-- The main statement: Prove that Yao Ming cannot step 10 meters in one stride.
theorem YaoMing_stride_impossible (h1: ∃ y : ℕ, XiaoMing_14_years_old y) 
                                  (h2: ∃ a : ℕ, sum_of_triangle_angles a) 
                                  (h3: CCTV5_broadcasting_basketball_game) 
: ¬ ∃ s : ℕ, YaoMing_stride s := sorry

end NUMINAMATH_GPT_YaoMing_stride_impossible_l2337_233730


namespace NUMINAMATH_GPT_smallest_sum_of_digits_l2337_233775

theorem smallest_sum_of_digits :
  ∃ (a b S : ℕ), 
    (100 ≤ a ∧ a < 1000) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (∃ (d1 d2 d3 d4 d5 : ℕ), 
      (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ 
      (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ 
      (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ 
      (d4 ≠ d5) ∧ 
      S = a + b ∧ 100 ≤ S ∧ S < 1000 ∧ 
      (∃ (s : ℕ), 
        s = (S / 100) + ((S % 100) / 10) + (S % 10) ∧ 
        s = 3)) :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_digits_l2337_233775


namespace NUMINAMATH_GPT_seymour_fertilizer_requirement_l2337_233707

theorem seymour_fertilizer_requirement :
  let flats_petunias := 4
  let petunias_per_flat := 8
  let flats_roses := 3
  let roses_per_flat := 6
  let venus_flytraps := 2
  let fert_per_petunia := 8
  let fert_per_rose := 3
  let fert_per_venus_flytrap := 2

  let total_petunias := flats_petunias * petunias_per_flat
  let total_roses := flats_roses * roses_per_flat
  let fert_petunias := total_petunias * fert_per_petunia
  let fert_roses := total_roses * fert_per_rose
  let fert_venus_flytraps := venus_flytraps * fert_per_venus_flytrap

  let total_fertilizer := fert_petunias + fert_roses + fert_venus_flytraps
  total_fertilizer = 314 := sorry

end NUMINAMATH_GPT_seymour_fertilizer_requirement_l2337_233707


namespace NUMINAMATH_GPT_constant_term_in_binomial_expansion_is_40_l2337_233742

-- Define the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression for the binomial expansion of (x^2 + 2/x^3)^5
def term (r : ℕ) : ℕ := binom 5 r * 2^r

theorem constant_term_in_binomial_expansion_is_40 
  (x : ℝ) (h : x ≠ 0) : 
  (∃ r : ℕ, 10 - 5 * r = 0) ∧ term 2 = 40 :=
by 
  sorry

end NUMINAMATH_GPT_constant_term_in_binomial_expansion_is_40_l2337_233742


namespace NUMINAMATH_GPT_A_share_of_profit_l2337_233748

-- Define necessary financial terms and operations
def initial_investment_A := 3000
def initial_investment_B := 4000

def withdrawal_A := 1000
def advanced_B := 1000

def duration_initial := 8
def duration_remaining := 4

def total_profit := 630

-- Calculate the equivalent investment duration for A and B
def investment_months_A_first := initial_investment_A * duration_initial
def investment_months_A_remaining := (initial_investment_A - withdrawal_A) * duration_remaining
def investment_months_A := investment_months_A_first + investment_months_A_remaining

def investment_months_B_first := initial_investment_B * duration_initial
def investment_months_B_remaining := (initial_investment_B + advanced_B) * duration_remaining
def investment_months_B := investment_months_B_first + investment_months_B_remaining

-- Prove that A's share of the profit is Rs. 240
theorem A_share_of_profit : 
  let ratio_A : ℚ := 4
  let ratio_B : ℚ := 6.5
  let total_ratio : ℚ := ratio_A + ratio_B
  let a_share : ℚ := (total_profit * ratio_A) / total_ratio
  a_share = 240 := 
by
  sorry

end NUMINAMATH_GPT_A_share_of_profit_l2337_233748


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2337_233766

theorem boat_speed_in_still_water (x y : ℝ) :
  (80 / (x + y) + 48 / (x - y) = 9) ∧ 
  (64 / (x + y) + 96 / (x - y) = 12) → 
  x = 12 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l2337_233766


namespace NUMINAMATH_GPT_reciprocal_roots_condition_l2337_233711

theorem reciprocal_roots_condition (a b c : ℝ) (h : a ≠ 0) (roots_reciprocal : ∃ r s : ℝ, r * s = 1 ∧ r + s = -b/a ∧ r * s = c/a) : c = a :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_roots_condition_l2337_233711


namespace NUMINAMATH_GPT_intersection_points_product_l2337_233795

theorem intersection_points_product (x y : ℝ) :
  (x^2 - 2 * x + y^2 - 6 * y + 9 = 0) ∧ (x^2 - 8 * x + y^2 - 6 * y + 28 = 0) → x * y = 6 :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_product_l2337_233795


namespace NUMINAMATH_GPT_rectangle_area_increase_l2337_233733

variable (L B : ℝ)

theorem rectangle_area_increase :
  let L_new := 1.30 * L
  let B_new := 1.45 * B
  let A_original := L * B
  let A_new := L_new * B_new
  let A_increase := A_new - A_original
  let percentage_increase := (A_increase / A_original) * 100
  percentage_increase = 88.5 := by
    sorry

end NUMINAMATH_GPT_rectangle_area_increase_l2337_233733


namespace NUMINAMATH_GPT_distance_to_campground_l2337_233737

-- definitions for speeds and times
def speed1 : ℤ := 50
def time1 : ℤ := 3
def speed2 : ℤ := 60
def time2 : ℤ := 2
def speed3 : ℤ := 55
def time3 : ℤ := 1
def speed4 : ℤ := 65
def time4 : ℤ := 2

-- definitions for calculating the distances
def distance1 : ℤ := speed1 * time1
def distance2 : ℤ := speed2 * time2
def distance3 : ℤ := speed3 * time3
def distance4 : ℤ := speed4 * time4

-- definition for the total distance
def total_distance : ℤ := distance1 + distance2 + distance3 + distance4

-- proof statement
theorem distance_to_campground : total_distance = 455 := by
  sorry -- proof omitted

end NUMINAMATH_GPT_distance_to_campground_l2337_233737


namespace NUMINAMATH_GPT_part1_part2_l2337_233758

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
  sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → h a x ≤ if a ≥ 0 then 3*a + 3 else if -3 ≤ a then a + 3 else 0) :=
  sorry

end NUMINAMATH_GPT_part1_part2_l2337_233758


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l2337_233720

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom a2 : a 2 = 6
axiom S5 : S 5 = 40

-- Prove the general formulas
theorem arithmetic_sequence_general_formula (n : ℕ)
  (h1 : ∃ d a1, ∀ n, a n = a1 + (n - 1) * d)
  (h2 : ∃ d a1, ∀ n, S n = n * ((2 * a1) + (n - 1) * d) / 2) :
  (a n = 2 * n + 2) ∧ (S n = n * (n + 3)) := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l2337_233720


namespace NUMINAMATH_GPT_divisibility_by_5_l2337_233774

theorem divisibility_by_5 (B : ℕ) (hB : B < 10) : (476 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := 
by
  sorry

end NUMINAMATH_GPT_divisibility_by_5_l2337_233774


namespace NUMINAMATH_GPT_union_complement_eq_l2337_233770

open Set

-- Condition definitions
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem statement (what we want to prove)
theorem union_complement_eq :
  A ∪ compl B = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
by
  sorry

end NUMINAMATH_GPT_union_complement_eq_l2337_233770


namespace NUMINAMATH_GPT_smallest_common_multiple_l2337_233750

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end NUMINAMATH_GPT_smallest_common_multiple_l2337_233750


namespace NUMINAMATH_GPT_problem_sol_l2337_233708

-- Assume g is an invertible function
variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
variable (h_invertible : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y)

-- Define p and q such that g(p) = 3 and g(q) = 5
variable (p q : ℝ)
variable (h1 : g p = 3) (h2 : g q = 5)

-- Goal to prove that p - q = 2
theorem problem_sol : p - q = 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_sol_l2337_233708


namespace NUMINAMATH_GPT_rectangle_area_l2337_233763

theorem rectangle_area (r l b : ℝ) (h1: r = 30) (h2: l = (2 / 5) * r) (h3: b = 10) : 
  l * b = 120 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2337_233763


namespace NUMINAMATH_GPT_photograph_goal_reach_l2337_233776

-- Define the initial number of photographs
def initial_photos : ℕ := 250

-- Define the percentage splits initially
def beth_pct_init : ℝ := 0.40
def my_pct_init : ℝ := 0.35
def julia_pct_init : ℝ := 0.25

-- Define the photographs taken initially by each person
def beth_photos_init : ℕ := 100
def my_photos_init : ℕ := 88
def julia_photos_init : ℕ := 63

-- Confirm initial photographs sum
example (h : beth_photos_init + my_photos_init + julia_photos_init = 251) : true := 
by trivial

-- Define today's decreased productivity percentages
def beth_decrease_pct : ℝ := 0.35
def my_decrease_pct : ℝ := 0.45
def julia_decrease_pct : ℝ := 0.25

-- Define the photographs taken today by each person after decreases
def beth_photos_today : ℕ := 65
def my_photos_today : ℕ := 48
def julia_photos_today : ℕ := 47

-- Sum of photographs taken today
def total_photos_today : ℕ := 160

-- Define the initial plus today's needed photographs to reach goal
def goal_photos : ℕ := 650

-- Define the additional number of photographs needed
def additional_photos_needed : ℕ := 399 - total_photos_today

-- Final proof statement
theorem photograph_goal_reach : 
  (beth_photos_init + my_photos_init + julia_photos_init) + (beth_photos_today + my_photos_today + julia_photos_today) + additional_photos_needed = goal_photos := 
by sorry

end NUMINAMATH_GPT_photograph_goal_reach_l2337_233776
