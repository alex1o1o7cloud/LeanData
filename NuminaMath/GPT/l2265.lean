import Mathlib

namespace NUMINAMATH_GPT_expected_value_of_10_sided_die_l2265_226558

-- Definition of the conditions
def num_faces : ℕ := 10
def face_values : List ℕ := List.range' 2 num_faces

-- Theorem statement: The expected value of a roll of this die is 6.5
theorem expected_value_of_10_sided_die : 
  (List.sum face_values : ℚ) / num_faces = 6.5 := 
sorry

end NUMINAMATH_GPT_expected_value_of_10_sided_die_l2265_226558


namespace NUMINAMATH_GPT_parameter_range_exists_solution_l2265_226591

theorem parameter_range_exists_solution :
  (∃ b : ℝ, -14 < b ∧ b < 9 ∧ ∃ a : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * b * (b + x + y) = 81 ∧ y = 5 / ((x - a)^2 + 1)) :=
sorry

end NUMINAMATH_GPT_parameter_range_exists_solution_l2265_226591


namespace NUMINAMATH_GPT_number_of_pairs_l2265_226506

open Nat

theorem number_of_pairs :
  ∃ n, n = 9 ∧
    (∃ x y : ℕ,
      x > 0 ∧ y > 0 ∧
      x + y = 150 ∧
      x % 3 = 0 ∧
      y % 5 = 0 ∧
      (∃! (x y : ℕ), x + y = 150 ∧ x % 3 = 0 ∧ y % 5 = 0 ∧ x > 0 ∧ y > 0)) := sorry

end NUMINAMATH_GPT_number_of_pairs_l2265_226506


namespace NUMINAMATH_GPT_exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l2265_226567

def omega (n : Nat) : Nat :=
  if n = 1 then 0 else n.factors.toFinset.card

theorem exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2 :
  ∃ᶠ n in atTop, ∃ k : Nat, n = 2^k ∧
    omega n < omega (n + 1) ∧
    omega (n + 1) < omega (n + 2) :=
sorry

end NUMINAMATH_GPT_exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l2265_226567


namespace NUMINAMATH_GPT_crayons_lost_or_given_away_correct_l2265_226545

def initial_crayons : ℕ := 606
def remaining_crayons : ℕ := 291
def crayons_lost_or_given_away : ℕ := initial_crayons - remaining_crayons

theorem crayons_lost_or_given_away_correct :
  crayons_lost_or_given_away = 315 :=
by
  sorry

end NUMINAMATH_GPT_crayons_lost_or_given_away_correct_l2265_226545


namespace NUMINAMATH_GPT_cannot_form_right_triangle_l2265_226514

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem cannot_form_right_triangle : ¬ is_right_triangle 40 50 60 := 
by
  sorry

end NUMINAMATH_GPT_cannot_form_right_triangle_l2265_226514


namespace NUMINAMATH_GPT_find_z_value_l2265_226534

variables {BD FC GC FE : Prop}
variables {a b c d e f g z : ℝ}

-- Assume all given conditions
axiom BD_is_straight : BD
axiom FC_is_straight : FC
axiom GC_is_straight : GC
axiom FE_is_straight : FE
axiom sum_is_z : z = a + b + c + d + e + f + g

-- Goal to prove
theorem find_z_value : z = 540 :=
by
  sorry

end NUMINAMATH_GPT_find_z_value_l2265_226534


namespace NUMINAMATH_GPT_exponent_problem_l2265_226544

theorem exponent_problem : (5 ^ 6 * 5 ^ 9 * 5) / 5 ^ 3 = 5 ^ 13 := 
by
  sorry

end NUMINAMATH_GPT_exponent_problem_l2265_226544


namespace NUMINAMATH_GPT_points_of_third_l2265_226526

noncomputable def points_of_first : ℕ := 11
noncomputable def points_of_second : ℕ := 7
noncomputable def points_of_fourth : ℕ := 2
noncomputable def johns_total_points : ℕ := 38500

theorem points_of_third :
  ∃ x : ℕ, (points_of_first * points_of_second * x * points_of_fourth ∣ johns_total_points) ∧
    (johns_total_points / (points_of_first * points_of_second * points_of_fourth)) = x := 
sorry

end NUMINAMATH_GPT_points_of_third_l2265_226526


namespace NUMINAMATH_GPT_maximum_sum_of_digits_difference_l2265_226576

-- Definition of the sum of the digits of a number
-- For the purpose of this statement, we'll assume the existence of a function sum_of_digits

def sum_of_digits (n : ℕ) : ℕ :=
  sorry -- Assume the function is defined elsewhere

-- Statement of the problem
theorem maximum_sum_of_digits_difference :
  ∃ x : ℕ, sum_of_digits (x + 2019) - sum_of_digits x = 12 :=
sorry

end NUMINAMATH_GPT_maximum_sum_of_digits_difference_l2265_226576


namespace NUMINAMATH_GPT_guacamole_serving_and_cost_l2265_226502

theorem guacamole_serving_and_cost 
  (initial_avocados : ℕ) 
  (additional_avocados : ℕ) 
  (avocados_per_serving : ℕ) 
  (x : ℝ) 
  (h_initial : initial_avocados = 5) 
  (h_additional : additional_avocados = 4) 
  (h_serving : avocados_per_serving = 3) :
  (initial_avocados + additional_avocados) / avocados_per_serving = 3 
  ∧ additional_avocados * x = 4 * x := by
  sorry

end NUMINAMATH_GPT_guacamole_serving_and_cost_l2265_226502


namespace NUMINAMATH_GPT_hyperbola_asymptote_l2265_226507

theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, (y^2 / 16 - x^2 / 9 = 1) ↔ (y = m * x ∨ y = -m * x)) → 
  (m = 4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l2265_226507


namespace NUMINAMATH_GPT_operation_difference_l2265_226571

def operation (x y : ℕ) : ℕ := x * y - 3 * x + y

theorem operation_difference : operation 5 9 - operation 9 5 = 16 :=
by
  sorry

end NUMINAMATH_GPT_operation_difference_l2265_226571


namespace NUMINAMATH_GPT_find_n_l2265_226529

variable (a r : ℚ) (n : ℕ)

def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Given conditions
axiom seq_first_term : a = 1 / 3
axiom seq_common_ratio : r = 1 / 3
axiom sum_of_first_n_terms_eq : geom_sum a r n = 80 / 243

-- Prove that n = 5
theorem find_n : n = 5 := by
  sorry

end NUMINAMATH_GPT_find_n_l2265_226529


namespace NUMINAMATH_GPT_smallest_integer_for_polynomial_div_l2265_226596

theorem smallest_integer_for_polynomial_div (x : ℤ) : 
  (∃ k : ℤ, x = 6) ↔ ∃ y, y * (x - 5) = x^2 + 4 * x + 7 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_integer_for_polynomial_div_l2265_226596


namespace NUMINAMATH_GPT_taco_truck_profit_l2265_226597

-- Definitions and conditions
def pounds_of_beef : ℕ := 100
def beef_per_taco : ℝ := 0.25
def price_per_taco : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Desired profit result
def expected_profit : ℝ := 200

-- The proof statement (to be completed)
theorem taco_truck_profit :
  let tacos := pounds_of_beef / beef_per_taco;
  let revenue := tacos * price_per_taco;
  let cost := tacos * cost_per_taco;
  let profit := revenue - cost;
  profit = expected_profit :=
by
  sorry

end NUMINAMATH_GPT_taco_truck_profit_l2265_226597


namespace NUMINAMATH_GPT_find_angle_A_l2265_226594

def triangle_ABC_angle_A (a b : ℝ) (B A : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute) : Prop :=
  A = Real.pi / 3

theorem find_angle_A 
  (a b A B : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute)
  (h_conditions : triangle_ABC_angle_A a b B A acute ha hb hB hacute) : 
  A = Real.pi / 3 := 
sorry

end NUMINAMATH_GPT_find_angle_A_l2265_226594


namespace NUMINAMATH_GPT_no_possible_path_l2265_226563

theorem no_possible_path (n : ℕ) (h1 : n > 0) :
  ¬ ∃ (path : ℕ × ℕ → ℕ × ℕ), 
    (∀ (i : ℕ × ℕ), path i = if (i.1 < n - 1 ∧ i.2 < n - 1) then (i.1 + 1, i.2) else if i.2 < n - 1 then (i.1, i.2 + 1) else (i.1 - 1, i.2 - 1)) ∧
    (∀ (i j : ℕ × ℕ), i ≠ j → path i ≠ path j) ∧
    path (0, 0) = (0, 1) ∧
    path (n-1, n-1) = (n-1, 0) :=
sorry

end NUMINAMATH_GPT_no_possible_path_l2265_226563


namespace NUMINAMATH_GPT_distance_from_tee_to_hole_l2265_226527

-- Define the constants based on the problem conditions
def s1 : ℕ := 180
def s2 : ℕ := (1 / 2 * s1 + 20 - 20)

-- Define the total distance calculation
def total_distance := s1 + s2

-- State the ultimate theorem that needs to be proved
theorem distance_from_tee_to_hole : total_distance = 270 := by
  sorry

end NUMINAMATH_GPT_distance_from_tee_to_hole_l2265_226527


namespace NUMINAMATH_GPT_aquafaba_needed_for_cakes_l2265_226592

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end NUMINAMATH_GPT_aquafaba_needed_for_cakes_l2265_226592


namespace NUMINAMATH_GPT_total_swordfish_caught_l2265_226555

theorem total_swordfish_caught (fishing_trips : ℕ) (shelly_each_trip : ℕ) (sam_each_trip : ℕ) : 
  shelly_each_trip = 3 → 
  sam_each_trip = 2 → 
  fishing_trips = 5 → 
  (shelly_each_trip + sam_each_trip) * fishing_trips = 25 :=
by
  sorry

end NUMINAMATH_GPT_total_swordfish_caught_l2265_226555


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_consecutive_even_l2265_226538

theorem right_triangle_hypotenuse_consecutive_even (x : ℕ) (h : x ≠ 0) :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ ((a, b, c) = (x - 2, x, x + 2) ∨ (a, b, c) = (x, x - 2, x + 2) ∨ (a, b, c) = (x + 2, x, x - 2)) ∧ c = 10 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_consecutive_even_l2265_226538


namespace NUMINAMATH_GPT_arun_weight_lower_limit_l2265_226512

variable {W B : ℝ}

theorem arun_weight_lower_limit
  (h1 : 64 < W ∧ W < 72)
  (h2 : B < W ∧ W < 70)
  (h3 : W ≤ 67)
  (h4 : (64 + 67) / 2 = 66) :
  64 < B :=
by sorry

end NUMINAMATH_GPT_arun_weight_lower_limit_l2265_226512


namespace NUMINAMATH_GPT_prove_county_growth_condition_l2265_226508

variable (x : ℝ)
variable (investment2014 : ℝ) (investment2016 : ℝ)

def county_growth_condition
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : Prop :=
  investment2014 * (1 + x)^2 = investment2016

theorem prove_county_growth_condition
  (x : ℝ)
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : county_growth_condition x investment2014 investment2016 h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_prove_county_growth_condition_l2265_226508


namespace NUMINAMATH_GPT_simplify_fraction_l2265_226557

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2265_226557


namespace NUMINAMATH_GPT_calculate_expression_l2265_226578

theorem calculate_expression :
  427 / 2.68 * 16 * 26.8 / 42.7 * 16 = 25600 :=
sorry

end NUMINAMATH_GPT_calculate_expression_l2265_226578


namespace NUMINAMATH_GPT_initial_weasels_count_l2265_226541

theorem initial_weasels_count (initial_rabbits : ℕ) (foxes : ℕ) (weasels_per_fox : ℕ) (rabbits_per_fox : ℕ) 
                              (weeks : ℕ) (remaining_rabbits_weasels : ℕ) (initial_weasels : ℕ) 
                              (total_rabbits_weasels : ℕ) : 
    initial_rabbits = 50 → foxes = 3 → weasels_per_fox = 4 → rabbits_per_fox = 2 → weeks = 3 → 
    remaining_rabbits_weasels = 96 → total_rabbits_weasels = initial_rabbits + initial_weasels → initial_weasels = 100 :=
by
  sorry

end NUMINAMATH_GPT_initial_weasels_count_l2265_226541


namespace NUMINAMATH_GPT_find_f2_l2265_226590

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 - a * x^3 + b * x - 6

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -22 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l2265_226590


namespace NUMINAMATH_GPT_negation_of_implication_l2265_226574

theorem negation_of_implication (x : ℝ) : x^2 + x - 6 < 0 → x ≤ 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_negation_of_implication_l2265_226574


namespace NUMINAMATH_GPT_birds_initially_l2265_226503

-- Definitions of the conditions
def initial_birds (B : Nat) := B
def initial_storks := 4
def additional_storks := 6
def total := 13

-- The theorem we need to prove
theorem birds_initially (B : Nat) (h : initial_birds B + initial_storks + additional_storks = total) : initial_birds B = 3 :=
by
  -- The proof can go here
  sorry

end NUMINAMATH_GPT_birds_initially_l2265_226503


namespace NUMINAMATH_GPT_total_commission_l2265_226522

-- Define the commission rate
def commission_rate : ℝ := 0.02

-- Define the sale prices of the three houses
def sale_price1 : ℝ := 157000
def sale_price2 : ℝ := 499000
def sale_price3 : ℝ := 125000

-- Total commission calculation
theorem total_commission :
  (commission_rate * sale_price1 + commission_rate * sale_price2 + commission_rate * sale_price3) = 15620 := 
by
  sorry

end NUMINAMATH_GPT_total_commission_l2265_226522


namespace NUMINAMATH_GPT_gravitational_force_at_384000km_l2265_226540

theorem gravitational_force_at_384000km
  (d1 d2 : ℝ)
  (f1 f2 : ℝ)
  (k : ℝ)
  (h1 : d1 = 6400)
  (h2 : d2 = 384000)
  (h3 : f1 = 800)
  (h4 : f1 * d1^2 = k)
  (h5 : f2 * d2^2 = k) :
  f2 = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_gravitational_force_at_384000km_l2265_226540


namespace NUMINAMATH_GPT_quadratic_y1_gt_y2_l2265_226559

theorem quadratic_y1_gt_y2 {a b c y1 y2 : ℝ} (ha : a > 0) (hy1 : y1 = a * (-1)^2 + b * (-1) + c) (hy2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
  sorry

end NUMINAMATH_GPT_quadratic_y1_gt_y2_l2265_226559


namespace NUMINAMATH_GPT_time_difference_l2265_226535

theorem time_difference (speed_Xanthia speed_Molly book_pages : ℕ) (minutes_in_hour : ℕ) :
  speed_Xanthia = 120 ∧ speed_Molly = 40 ∧ book_pages = 360 ∧ minutes_in_hour = 60 →
  (book_pages / speed_Molly - book_pages / speed_Xanthia) * minutes_in_hour = 360 := by
  sorry

end NUMINAMATH_GPT_time_difference_l2265_226535


namespace NUMINAMATH_GPT_input_statement_is_INPUT_l2265_226542

namespace ProgrammingStatements

-- Definitions of each type of statement
def PRINT_is_output : Prop := True
def INPUT_is_input : Prop := True
def THEN_is_conditional : Prop := True
def END_is_termination : Prop := True

-- The proof problem
theorem input_statement_is_INPUT :
  INPUT_is_input := by
  sorry

end ProgrammingStatements

end NUMINAMATH_GPT_input_statement_is_INPUT_l2265_226542


namespace NUMINAMATH_GPT_divisor_of_635_l2265_226520

theorem divisor_of_635 (p : ℕ) (h1 : Nat.Prime p) (k : ℕ) (h2 : 635 = 7 * k * p + 11) : p = 89 :=
sorry

end NUMINAMATH_GPT_divisor_of_635_l2265_226520


namespace NUMINAMATH_GPT_neither_necessary_nor_sufficient_l2265_226582

def set_M : Set ℝ := {x | x > 2}
def set_P : Set ℝ := {x | x < 3}

theorem neither_necessary_nor_sufficient (x : ℝ) :
  (x ∈ set_M ∨ x ∈ set_P) ↔ (x ∉ set_M ∩ set_P) :=
sorry

end NUMINAMATH_GPT_neither_necessary_nor_sufficient_l2265_226582


namespace NUMINAMATH_GPT_distance_against_current_l2265_226539

theorem distance_against_current (V_b V_c : ℝ) (h1 : V_b + V_c = 2) (h2 : V_b = 1.5) : 
  (V_b - V_c) * 3 = 3 := by
  sorry

end NUMINAMATH_GPT_distance_against_current_l2265_226539


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_S6_l2265_226524

variables (S : ℕ → ℕ)

-- Definitions of conditions from a)
def S2 := S 2 = 3
def S4 := S 4 = 15

-- Main proof statement
theorem arithmetic_geometric_sequence_S6 (S : ℕ → ℕ) (h1 : S 2 = 3) (h2 : S 4 = 15) :
  S 6 = 63 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_S6_l2265_226524


namespace NUMINAMATH_GPT_pool_width_l2265_226599

-- Define the given conditions
def hose_rate : ℝ := 60 -- cubic feet per minute
def drain_time : ℝ := 2000 -- minutes
def pool_length : ℝ := 150 -- feet
def pool_depth : ℝ := 10 -- feet

-- Calculate the total volume drained
def total_volume := hose_rate * drain_time -- cubic feet

-- Define a variable for the pool width
variable (W : ℝ)

-- The statement to prove
theorem pool_width :
  (total_volume = pool_length * W * pool_depth) → W = 80 :=
by
  sorry

end NUMINAMATH_GPT_pool_width_l2265_226599


namespace NUMINAMATH_GPT_solution_l2265_226528

def p (x : ℝ) : Prop := x^2 + 2 * x - 3 < 0
def q (x : ℝ) : Prop := x ∈ Set.univ

theorem solution (x : ℝ) (hx : p x ∧ q x) : x = -2 ∨ x = -1 ∨ x = 0 := 
by
  sorry

end NUMINAMATH_GPT_solution_l2265_226528


namespace NUMINAMATH_GPT_solve_inequality_l2265_226561

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_inequality_l2265_226561


namespace NUMINAMATH_GPT_geometric_sequence_nec_not_suff_l2265_226530

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≠ 0 → (a (n + 1) / a n) = (a (n + 2) / a (n + 1))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 3) = a (n + 1) * a (n + 2)

theorem geometric_sequence_nec_not_suff (a : ℕ → ℝ) (hn : ∀ n : ℕ, a n ≠ 0) : 
  (is_geometric_sequence a → satisfies_condition a) ∧ ¬(satisfies_condition a → is_geometric_sequence a) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_nec_not_suff_l2265_226530


namespace NUMINAMATH_GPT_candy_bar_reduction_l2265_226547

variable (W P x : ℝ)
noncomputable def percent_reduction := (x / W) * 100

theorem candy_bar_reduction (h_weight_reduced : W > 0) 
                            (h_price_same : P > 0) 
                            (h_price_increase : P / (W - x) = (5 / 3) * (P / W)) :
    percent_reduction W x = 40 := 
sorry

end NUMINAMATH_GPT_candy_bar_reduction_l2265_226547


namespace NUMINAMATH_GPT_cubes_not_arithmetic_progression_l2265_226562

theorem cubes_not_arithmetic_progression (x y z : ℤ) (h1 : y = (x + z) / 2) (h2 : x ≠ y) (h3 : y ≠ z) : x^3 + z^3 ≠ 2 * y^3 :=
by
  sorry

end NUMINAMATH_GPT_cubes_not_arithmetic_progression_l2265_226562


namespace NUMINAMATH_GPT_travel_time_equation_l2265_226551

theorem travel_time_equation
 (d : ℝ) (x t_saved factor : ℝ) 
 (h : d = 202) 
 (h1 : t_saved = 1.8) 
 (h2 : factor = 1.6)
 : (d / x) * factor = d / (x - t_saved) := sorry

end NUMINAMATH_GPT_travel_time_equation_l2265_226551


namespace NUMINAMATH_GPT_proof1_proof2_l2265_226531

open Real

noncomputable def problem1 (a b c : ℝ) (A : ℝ) (S : ℝ) :=
  ∃ (a b : ℝ), A = π / 3 ∧ c = 2 ∧ S = sqrt 3 / 2 ∧ S = 1/2 * b * 2 * sin (π / 3) ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos (π / 3) ∧ b = 1 ∧ a = sqrt 3

noncomputable def problem2 (a b c : ℝ) (A B : ℝ) :=
  c = a * cos B ∧ (a + b + c) * (a + b - c) = (2 + sqrt 2) * a * b ∧ 
  B = π / 4 ∧ A = π / 2 → 
  ∃ C, C = π / 4 ∧ C = B

theorem proof1 : problem1 (sqrt 3) 1 2 (π / 3) (sqrt 3 / 2) :=
by
  sorry

theorem proof2 : problem2 (sqrt 3) 1 2 (π / 2) (π / 4) :=
by
  sorry

end NUMINAMATH_GPT_proof1_proof2_l2265_226531


namespace NUMINAMATH_GPT_samantha_lost_pieces_l2265_226505

theorem samantha_lost_pieces (total_pieces_on_board : ℕ) (arianna_lost : ℕ) (initial_pieces_per_player : ℕ) :
  total_pieces_on_board = 20 →
  arianna_lost = 3 →
  initial_pieces_per_player = 16 →
  (initial_pieces_per_player - (total_pieces_on_board - (initial_pieces_per_player - arianna_lost))) = 9 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_samantha_lost_pieces_l2265_226505


namespace NUMINAMATH_GPT_custom_op_eval_l2265_226537

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b - a^2 * b

theorem custom_op_eval :
  custom_op 3 4 = -4 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_eval_l2265_226537


namespace NUMINAMATH_GPT_same_asymptotes_hyperbolas_l2265_226533

theorem same_asymptotes_hyperbolas (M : ℝ) :
  (∀ x y : ℝ, ((x^2 / 9) - (y^2 / 16) = 1) ↔ ((y^2 / 32) - (x^2 / M) = 1)) →
  M = 18 :=
by
  sorry

end NUMINAMATH_GPT_same_asymptotes_hyperbolas_l2265_226533


namespace NUMINAMATH_GPT_average_halfway_l2265_226513

theorem average_halfway (a b : ℚ) (h_a : a = 1/8) (h_b : b = 1/3) : (a + b) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_GPT_average_halfway_l2265_226513


namespace NUMINAMATH_GPT_C_share_of_profit_l2265_226516

def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def C_investment : ℕ := 20000
def total_profit : ℕ := 86400

theorem C_share_of_profit: 
  (C_investment / (A_investment + B_investment + C_investment) * total_profit) = 36000 :=
by
  sorry

end NUMINAMATH_GPT_C_share_of_profit_l2265_226516


namespace NUMINAMATH_GPT_largest_number_not_sum_of_two_composites_l2265_226564

-- Definitions related to composite numbers and natural numbers
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)

-- Lean statement of the problem
theorem largest_number_not_sum_of_two_composites : ∀ n : ℕ,
  (∀ m : ℕ, m > n → cannot_be_sum_of_two_composites m → false) → n = 11 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_not_sum_of_two_composites_l2265_226564


namespace NUMINAMATH_GPT_parabola_x_intercept_y_intercept_point_l2265_226509

theorem parabola_x_intercept_y_intercept_point (a b w : ℝ) 
  (h1 : a = -1) 
  (h2 : b = 4) 
  (h3 : ∀ x : ℝ, x = 0 → w = 8): 
  ∃ (w : ℝ), w = 8 := 
by
  sorry

end NUMINAMATH_GPT_parabola_x_intercept_y_intercept_point_l2265_226509


namespace NUMINAMATH_GPT_Mike_owes_Laura_l2265_226588

theorem Mike_owes_Laura :
  let rate_per_room := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let total_amount := (104 : ℚ) / 15
  rate_per_room * rooms_cleaned = total_amount :=
by
  sorry

end NUMINAMATH_GPT_Mike_owes_Laura_l2265_226588


namespace NUMINAMATH_GPT_average_percentage_increase_l2265_226560

def initial_income_A : ℝ := 60
def new_income_A : ℝ := 80
def initial_income_B : ℝ := 100
def new_income_B : ℝ := 130
def hours_worked_C : ℝ := 20
def initial_rate_C : ℝ := 8
def new_rate_C : ℝ := 10

theorem average_percentage_increase :
  let initial_weekly_income_C := hours_worked_C * initial_rate_C
  let new_weekly_income_C := hours_worked_C * new_rate_C
  let percentage_increase_A := (new_income_A - initial_income_A) / initial_income_A * 100
  let percentage_increase_B := (new_income_B - initial_income_B) / initial_income_B * 100
  let percentage_increase_C := (new_weekly_income_C - initial_weekly_income_C) / initial_weekly_income_C * 100
  let average_percentage_increase := (percentage_increase_A + percentage_increase_B + percentage_increase_C) / 3
  average_percentage_increase = 29.44 :=
by sorry

end NUMINAMATH_GPT_average_percentage_increase_l2265_226560


namespace NUMINAMATH_GPT_count_sum_or_diff_squares_at_least_1500_l2265_226518

theorem count_sum_or_diff_squares_at_least_1500 : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 2000 ∧ (∃ (x y : ℕ), n = x^2 + y^2 ∨ n = x^2 - y^2)) → 
  1500 ≤ 2000 :=
by
  sorry

end NUMINAMATH_GPT_count_sum_or_diff_squares_at_least_1500_l2265_226518


namespace NUMINAMATH_GPT_initial_marbles_count_l2265_226523

-- Leo's initial conditions and quantities
def initial_packs := 40
def marbles_per_pack := 10
def given_Manny (P: ℕ) := P / 4
def given_Neil (P: ℕ) := P / 8
def kept_by_Leo := 25

-- The equivalent proof problem stated in Lean
theorem initial_marbles_count (P: ℕ) (Manny_packs: ℕ) (Neil_packs: ℕ) (kept_packs: ℕ) :
  Manny_packs = given_Manny P → Neil_packs = given_Neil P → kept_packs = kept_by_Leo → P = initial_packs → P * marbles_per_pack = 400 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_marbles_count_l2265_226523


namespace NUMINAMATH_GPT_worker_cellphone_surveys_l2265_226579

theorem worker_cellphone_surveys 
  (regular_rate : ℕ) 
  (num_surveys : ℕ) 
  (higher_rate : ℕ)
  (total_earnings : ℕ) 
  (earned : ℕ → ℕ → ℕ)
  (higher_earned : ℕ → ℕ → ℕ) 
  (h1 : regular_rate = 10) 
  (h2 : num_surveys = 50) 
  (h3 : higher_rate = 13) 
  (h4 : total_earnings = 605) 
  (h5 : ∀ x, earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings)
  : (∃ x, x = 35 ∧ earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings) :=
sorry

end NUMINAMATH_GPT_worker_cellphone_surveys_l2265_226579


namespace NUMINAMATH_GPT_find_c_for_square_of_binomial_l2265_226504

theorem find_c_for_square_of_binomial (c : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 50 * x + c = (x + b)^2) → c = 625 :=
by
  intro h
  obtain ⟨b, h⟩ := h
  sorry

end NUMINAMATH_GPT_find_c_for_square_of_binomial_l2265_226504


namespace NUMINAMATH_GPT_swimming_lane_length_l2265_226550

theorem swimming_lane_length (round_trips : ℕ) (total_distance : ℕ) (lane_length : ℕ) 
  (h1 : round_trips = 4) (h2 : total_distance = 800) 
  (h3 : total_distance = lane_length * (round_trips * 2)) : 
  lane_length = 100 := 
by
  sorry

end NUMINAMATH_GPT_swimming_lane_length_l2265_226550


namespace NUMINAMATH_GPT_find_g5_l2265_226553

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end NUMINAMATH_GPT_find_g5_l2265_226553


namespace NUMINAMATH_GPT_extra_people_got_on_the_train_l2265_226536

-- Definitions corresponding to the conditions
def initial_people_on_train : ℕ := 78
def people_got_off : ℕ := 27
def current_people_on_train : ℕ := 63

-- The mathematical equivalent proof problem
theorem extra_people_got_on_the_train :
  (initial_people_on_train - people_got_off + extra_people = current_people_on_train) → (extra_people = 12) :=
by
  sorry

end NUMINAMATH_GPT_extra_people_got_on_the_train_l2265_226536


namespace NUMINAMATH_GPT_roots_quadratic_eq_a2_b2_l2265_226587

theorem roots_quadratic_eq_a2_b2 (a b : ℝ) (h1 : a^2 - 5 * a + 5 = 0) (h2 : b^2 - 5 * b + 5 = 0) : a^2 + b^2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_eq_a2_b2_l2265_226587


namespace NUMINAMATH_GPT_money_distribution_l2265_226573

theorem money_distribution (a : ℕ) (h1 : 5 * a = 1500) : 7 * a - 3 * a = 1200 := by
  sorry

end NUMINAMATH_GPT_money_distribution_l2265_226573


namespace NUMINAMATH_GPT_roots_diff_eq_4_l2265_226575

theorem roots_diff_eq_4 {r s : ℝ} (h₁ : r ≠ s) (h₂ : r > s) (h₃ : r^2 - 10 * r + 21 = 0) (h₄ : s^2 - 10 * s + 21 = 0) : r - s = 4 := 
by
  sorry

end NUMINAMATH_GPT_roots_diff_eq_4_l2265_226575


namespace NUMINAMATH_GPT_total_ages_l2265_226515

def Kate_age : ℕ := 19
def Maggie_age : ℕ := 17
def Sue_age : ℕ := 12

theorem total_ages : Kate_age + Maggie_age + Sue_age = 48 := sorry

end NUMINAMATH_GPT_total_ages_l2265_226515


namespace NUMINAMATH_GPT_max_value_a_l2265_226572

def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1 / x|

theorem max_value_a : ∃ (a : ℝ), condition a ∧ (∀ b : ℝ, condition b → b ≤ 4) :=
  sorry

end NUMINAMATH_GPT_max_value_a_l2265_226572


namespace NUMINAMATH_GPT_ruler_cost_l2265_226577

variable {s c r : ℕ}

theorem ruler_cost (h1 : s > 18) (h2 : r > 1) (h3 : c > r) (h4 : s * c * r = 1729) : c = 13 :=
by
  sorry

end NUMINAMATH_GPT_ruler_cost_l2265_226577


namespace NUMINAMATH_GPT_general_form_identity_expression_simplification_l2265_226595

section
variable (a b x y : ℝ)

theorem general_form_identity : (a + b) * (a^2 - a * b + b^2) = a^3 + b^3 :=
by
  sorry

theorem expression_simplification : (x + y) * (x^2 - x * y + y^2) - (x - y) * (x^2 + x * y + y^2) = 2 * y^3 :=
by
  sorry
end

end NUMINAMATH_GPT_general_form_identity_expression_simplification_l2265_226595


namespace NUMINAMATH_GPT_width_of_park_l2265_226583

theorem width_of_park (L : ℕ) (A_lawn : ℕ) (w_road : ℕ) (W : ℚ) :
  L = 60 → A_lawn = 2109 → w_road = 3 →
  60 * W - 2 * 60 * 3 = 2109 →
  W = 41.15 :=
by
  intros hL hA_lawn hw_road hEq
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_width_of_park_l2265_226583


namespace NUMINAMATH_GPT_triangle_side_length_l2265_226532

variables {BC AC : ℝ} {α β γ : ℝ}

theorem triangle_side_length :
  α = 45 ∧ β = 75 ∧ AC = 6 ∧ α + β + γ = 180 →
  BC = 6 * (Real.sqrt 3 - 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_triangle_side_length_l2265_226532


namespace NUMINAMATH_GPT_train_cross_time_l2265_226548

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := 20
noncomputable def platform_length : ℝ := 320
noncomputable def time_cross_platform : ℝ := 34
noncomputable def train_length : ℝ := 360

theorem train_cross_time (v_kmph : ℝ) (v_mps : ℝ) (p_len : ℝ) (t_cross : ℝ) (t_len : ℝ) :
  v_kmph = 72 ∧ v_mps = 20 ∧ p_len = 320 ∧ t_cross = 34 ∧ t_len = 360 →
  (t_len / v_mps) = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_cross_time_l2265_226548


namespace NUMINAMATH_GPT_g_f_neg2_l2265_226586

def f (x : ℤ) : ℤ := x^3 + 3

def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg2 : g (f (-2)) = 41 :=
by {
  -- proof steps skipped
  sorry
}

end NUMINAMATH_GPT_g_f_neg2_l2265_226586


namespace NUMINAMATH_GPT_repeating_decimal_eq_fraction_l2265_226584

theorem repeating_decimal_eq_fraction :
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  (∑' n : ℕ, a * (r ^ n)) = 85 / 99 := by
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  exact sorry

end NUMINAMATH_GPT_repeating_decimal_eq_fraction_l2265_226584


namespace NUMINAMATH_GPT_express_in_scientific_notation_l2265_226581

theorem express_in_scientific_notation :
  102200 = 1.022 * 10^5 :=
sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l2265_226581


namespace NUMINAMATH_GPT_deposit_percentage_is_10_l2265_226554

-- Define the deposit and remaining amount
def deposit := 120
def remaining := 1080

-- Define total cost
def total_cost := deposit + remaining

-- Define deposit percentage calculation
def deposit_percentage := (deposit / total_cost) * 100

-- Theorem to prove the deposit percentage is 10%
theorem deposit_percentage_is_10 : deposit_percentage = 10 := by
  -- Since deposit, remaining and total_cost are defined explicitly,
  -- the proof verification of final result is straightforward.
  sorry

end NUMINAMATH_GPT_deposit_percentage_is_10_l2265_226554


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l2265_226556

theorem quadratic_has_distinct_real_roots (m : ℝ) (hm : m ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (m * x1^2 - 2 * x1 + 3 = 0) ∧ (m * x2^2 - 2 * x2 + 3 = 0) ↔ 0 < m ∧ m < (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l2265_226556


namespace NUMINAMATH_GPT_no_valid_sequence_of_integers_from_1_to_2004_l2265_226521

theorem no_valid_sequence_of_integers_from_1_to_2004 :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 1 ≤ a i ∧ a i ≤ 2004) ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ k, 1 ≤ k ∧ k + 9 ≤ 2004 → 
      (a k + a (k + 1) + a (k + 2) + a (k + 3) + a (k + 4) + a (k + 5) + 
       a (k + 6) + a (k + 7) + a (k + 8) + a (k + 9)) % 10 = 0) :=
  sorry

end NUMINAMATH_GPT_no_valid_sequence_of_integers_from_1_to_2004_l2265_226521


namespace NUMINAMATH_GPT_rectangle_area_l2265_226566

noncomputable def side_of_square : ℝ := Real.sqrt 625

noncomputable def radius_of_circle : ℝ := side_of_square

noncomputable def length_of_rectangle : ℝ := (2 / 5) * radius_of_circle

def breadth_of_rectangle : ℝ := 10

theorem rectangle_area :
  length_of_rectangle * breadth_of_rectangle = 100 := 
by
  simp [length_of_rectangle, breadth_of_rectangle, radius_of_circle, side_of_square]
  sorry

end NUMINAMATH_GPT_rectangle_area_l2265_226566


namespace NUMINAMATH_GPT_ratio_x_y_z_l2265_226585

theorem ratio_x_y_z (x y z : ℝ) (h1 : 0.10 * x = 0.20 * y) (h2 : 0.30 * y = 0.40 * z) :
  ∃ k : ℝ, x = 8 * k ∧ y = 4 * k ∧ z = 3 * k :=
by                         
  sorry

end NUMINAMATH_GPT_ratio_x_y_z_l2265_226585


namespace NUMINAMATH_GPT_min_a_for_monotonic_increase_l2265_226580

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x ^ 3 + 2 * a * x ^ 2 + 2

theorem min_a_for_monotonic_increase :
  ∀ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → x^2 + 4 * a * x ≥ 0) ↔ a ≥ -1/4 := sorry

end NUMINAMATH_GPT_min_a_for_monotonic_increase_l2265_226580


namespace NUMINAMATH_GPT_actual_time_before_storm_l2265_226589

-- Define valid hour digit ranges before the storm
def valid_first_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3
def valid_second_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define valid minute digit ranges before the storm
def valid_third_digit (d : ℕ) : Prop := d = 4 ∨ d = 5 ∨ d = 6
def valid_fourth_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define a valid time in HH:MM format
def valid_time (hh mm : ℕ) : Prop :=
  hh < 24 ∧ mm < 60

-- The proof problem
theorem actual_time_before_storm (hh hh' mm mm' : ℕ) 
  (h1 : valid_first_digit hh) (h2 : valid_second_digit hh') 
  (h3 : valid_third_digit mm) (h4 : valid_fourth_digit mm') 
  (h_valid : valid_time (hh * 10 + hh') (mm * 10 + mm')) 
  (h_display : (hh + 1) * 10 + (hh' - 1) = 20 ∧ (mm + 1) * 10 + (mm' - 1) = 50) :
  hh * 10 + hh' = 19 ∧ mm * 10 + mm' = 49 :=
by
  sorry

end NUMINAMATH_GPT_actual_time_before_storm_l2265_226589


namespace NUMINAMATH_GPT_angles_equal_l2265_226552

theorem angles_equal {α β γ α1 β1 γ1 : ℝ} (h1 : α + β + γ = 180) (h2 : α1 + β1 + γ1 = 180) 
  (h_eq_or_sum_to_180 : (α = α1 ∨ α + α1 = 180) ∧ (β = β1 ∨ β + β1 = 180) ∧ (γ = γ1 ∨ γ + γ1 = 180)) :
  α = α1 ∧ β = β1 ∧ γ = γ1 := 
by 
  sorry

end NUMINAMATH_GPT_angles_equal_l2265_226552


namespace NUMINAMATH_GPT_ordered_pair_solution_l2265_226569

theorem ordered_pair_solution :
  ∃ (x y : ℤ), 
    (x + y = (7 - x) + (7 - y)) ∧ 
    (x - y = (x - 2) + (y - 2)) ∧ 
    (x = 5 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_solution_l2265_226569


namespace NUMINAMATH_GPT_A_inter_B_domain_l2265_226543

def A_domain : Set ℝ := {x : ℝ | x^2 + x - 2 >= 0}
def B_domain : Set ℝ := {x : ℝ | (2*x + 6)/(3 - x) >= 0 ∧ x ≠ -2}

theorem A_inter_B_domain :
  (A_domain ∩ B_domain) = {x : ℝ | (1 <= x ∧ x < 3) ∨ (-3 <= x ∧ x < -2)} :=
by
  sorry

end NUMINAMATH_GPT_A_inter_B_domain_l2265_226543


namespace NUMINAMATH_GPT_complex_number_location_second_quadrant_l2265_226519

theorem complex_number_location_second_quadrant (z : ℂ) (h : z / (1 + I) = I) : z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_GPT_complex_number_location_second_quadrant_l2265_226519


namespace NUMINAMATH_GPT_tan_theta_solution_l2265_226517

theorem tan_theta_solution (θ : ℝ)
  (h : 2 * Real.sin (θ + Real.pi / 3) = 3 * Real.sin (Real.pi / 3 - θ)) :
  Real.tan θ = Real.sqrt 3 / 5 := sorry

end NUMINAMATH_GPT_tan_theta_solution_l2265_226517


namespace NUMINAMATH_GPT_min_price_floppy_cd_l2265_226593

theorem min_price_floppy_cd (x y : ℝ) (h1 : 4 * x + 5 * y ≥ 20) (h2 : 6 * x + 3 * y ≤ 24) : 3 * x + 9 * y ≥ 22 :=
by
  -- The proof is not provided as per the instructions.
  sorry

end NUMINAMATH_GPT_min_price_floppy_cd_l2265_226593


namespace NUMINAMATH_GPT_sid_spent_on_computer_accessories_l2265_226501

def initial_money : ℕ := 48
def snacks_cost : ℕ := 8
def remaining_money_more_than_half : ℕ := 4

theorem sid_spent_on_computer_accessories : 
  ∀ (m s r : ℕ), m = initial_money → s = snacks_cost → r = remaining_money_more_than_half →
  m - (r + m / 2 + s) = 12 :=
by
  intros m s r h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_sid_spent_on_computer_accessories_l2265_226501


namespace NUMINAMATH_GPT_largest_fraction_l2265_226598

theorem largest_fraction (x y z w : ℝ) (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzw : z < w) :
  max (max (max (max ((x + y) / (z + w)) ((x + w) / (y + z))) ((y + z) / (x + w))) ((y + w) / (x + z))) ((z + w) / (x + y)) = (z + w) / (x + y) :=
by sorry

end NUMINAMATH_GPT_largest_fraction_l2265_226598


namespace NUMINAMATH_GPT_probability_of_matching_pair_l2265_226549

/-!
# Probability of Selecting a Matching Pair of Shoes

Given:
- 12 pairs of sneakers, each with a 4% probability of being chosen.
- 15 pairs of boots, each with a 3% probability of being chosen.
- 18 pairs of dress shoes, each with a 2% probability of being chosen.

If two shoes are selected from the warehouse without replacement, prove that the probability 
of selecting a matching pair of shoes is 52.26%.
-/

namespace ShoeWarehouse

def prob_sneakers_first : ℝ := 0.48
def prob_sneakers_second : ℝ := 0.44
def prob_boots_first : ℝ := 0.45
def prob_boots_second : ℝ := 0.42
def prob_dress_first : ℝ := 0.36
def prob_dress_second : ℝ := 0.34

theorem probability_of_matching_pair :
  (prob_sneakers_first * prob_sneakers_second) +
  (prob_boots_first * prob_boots_second) +
  (prob_dress_first * prob_dress_second) = 0.5226 :=
sorry

end ShoeWarehouse

end NUMINAMATH_GPT_probability_of_matching_pair_l2265_226549


namespace NUMINAMATH_GPT_incorrect_statement_b_l2265_226510

-- Defining the equation of the circle
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 25

-- Defining the point not on the circle
def is_not_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 ≠ 25

-- The proposition to be proved
theorem incorrect_statement_b : ¬ ∀ p : ℝ × ℝ, is_not_on_circle p.1 p.2 → ¬ is_on_circle p.1 p.2 :=
by
  -- Here we should provide the proof, but this is not required based on the instructions.
  sorry

end NUMINAMATH_GPT_incorrect_statement_b_l2265_226510


namespace NUMINAMATH_GPT_f_at_three_bounds_l2265_226511

theorem f_at_three_bounds (a c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^2 - c)
  (h2 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h3 : -1 ≤ f 2 ∧ f 2 ≤ 5) : -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end NUMINAMATH_GPT_f_at_three_bounds_l2265_226511


namespace NUMINAMATH_GPT_Danny_finishes_first_l2265_226546

-- Definitions based on the conditions
variables (E D F : ℝ)    -- Garden areas for Emily, Danny, Fiona
variables (e d f : ℝ)    -- Mowing rates for Emily, Danny, Fiona
variables (start_time : ℝ)

-- Condition definitions
def emily_garden_size := E = 3 * D
def emily_garden_size_fiona := E = 5 * F
def fiona_mower_speed_danny := f = (1/4) * d
def fiona_mower_speed_emily := f = (1/5) * e

-- Prove Danny finishes first
theorem Danny_finishes_first 
  (h1 : emily_garden_size E D)
  (h2 : emily_garden_size_fiona E F)
  (h3 : fiona_mower_speed_danny f d)
  (h4 : fiona_mower_speed_emily f e) : 
  (start_time ≤ (5/12) * (start_time + E/d) ∧ start_time ≤ (E/f)) -> (start_time + E/d < start_time + E/e) -> 
  true := 
sorry -- proof is omitted

end NUMINAMATH_GPT_Danny_finishes_first_l2265_226546


namespace NUMINAMATH_GPT_pie_crusts_flour_l2265_226570

theorem pie_crusts_flour (initial_crusts : ℕ)
  (initial_flour_per_crust : ℚ)
  (new_crusts : ℕ)
  (total_flour : ℚ)
  (h1 : initial_crusts = 40)
  (h2 : initial_flour_per_crust = 1/8)
  (h3 : new_crusts = 25)
  (h4 : total_flour = initial_crusts * initial_flour_per_crust) :
  (new_crusts * (total_flour / new_crusts) = total_flour) :=
by
  sorry

end NUMINAMATH_GPT_pie_crusts_flour_l2265_226570


namespace NUMINAMATH_GPT_notebook_cost_3_dollars_l2265_226525

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end NUMINAMATH_GPT_notebook_cost_3_dollars_l2265_226525


namespace NUMINAMATH_GPT_percentage_goods_lost_eq_l2265_226565

-- Define the initial conditions
def initial_value : ℝ := 100
def profit_margin : ℝ := 0.10 * initial_value
def selling_price : ℝ := initial_value + profit_margin
def loss_percentage : ℝ := 0.12

-- Define the correct answer as a constant
def correct_percentage_loss : ℝ := 13.2

-- Define the target theorem
theorem percentage_goods_lost_eq : (0.12 * selling_price / initial_value * 100) = correct_percentage_loss := 
by
  -- sorry is used to skip the proof part as per instructions
  sorry

end NUMINAMATH_GPT_percentage_goods_lost_eq_l2265_226565


namespace NUMINAMATH_GPT_average_of_numbers_in_range_l2265_226568

-- Define the set of numbers we are considering
def numbers_in_range : List ℕ := [10, 15, 20, 25, 30]

-- Define the sum of these numbers
def sum_in_range : ℕ := 10 + 15 + 20 + 25 + 30

-- Define the number of elements in our range
def count_in_range : ℕ := 5

-- Prove that the average of numbers in the range is 20
theorem average_of_numbers_in_range : (sum_in_range / count_in_range) = 20 := by
  -- TODO: Proof to be written, for now we use sorry as a placeholder
  sorry

end NUMINAMATH_GPT_average_of_numbers_in_range_l2265_226568


namespace NUMINAMATH_GPT_log_product_eq_two_l2265_226500

open Real

theorem log_product_eq_two
  : log 5 / log 3 * log 6 / log 5 * log 9 / log 6 = 2 := by
  sorry

end NUMINAMATH_GPT_log_product_eq_two_l2265_226500
