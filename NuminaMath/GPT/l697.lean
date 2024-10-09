import Mathlib

namespace total_weight_of_lifts_l697_69729

theorem total_weight_of_lifts
  (F S : ℕ)
  (h1 : F = 600)
  (h2 : 2 * F = S + 300) :
  F + S = 1500 := by
  sorry

end total_weight_of_lifts_l697_69729


namespace other_asymptote_l697_69721

-- Define the conditions
def C1 := ∀ x y, y = -2 * x
def C2 := ∀ x, x = -3

-- Formulate the problem
theorem other_asymptote :
  (∃ y m b, y = m * x + b ∧ m = 2 ∧ b = 12) :=
by
  sorry

end other_asymptote_l697_69721


namespace solve_floor_equation_l697_69780

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem solve_floor_equation :
  (∃ x : ℝ, (floor ((x - 1) / 2))^2 + 2 * x + 2 = 0) → x = -3 :=
by
  sorry

end solve_floor_equation_l697_69780


namespace pond_water_amount_l697_69799

theorem pond_water_amount : 
  let initial_water := 500 
  let evaporation_rate := 4
  let rain_amount := 2
  let days := 40
  initial_water - days * (evaporation_rate - rain_amount) = 420 :=
by
  sorry

end pond_water_amount_l697_69799


namespace sum_of_sins_is_zero_l697_69782

variable {x y z : ℝ}

theorem sum_of_sins_is_zero
  (h1 : Real.sin x = Real.tan y)
  (h2 : Real.sin y = Real.tan z)
  (h3 : Real.sin z = Real.tan x) :
  Real.sin x + Real.sin y + Real.sin z = 0 :=
sorry

end sum_of_sins_is_zero_l697_69782


namespace crumble_topping_correct_amount_l697_69730

noncomputable def crumble_topping_total_mass (flour butter sugar : ℕ) (factor : ℚ) : ℚ :=
  factor * (flour + butter + sugar) / 1000  -- convert grams to kilograms

theorem crumble_topping_correct_amount {flour butter sugar : ℕ} (factor : ℚ) (h_flour : flour = 100) (h_butter : butter = 50) (h_sugar : sugar = 50) (h_factor : factor = 2.5) :
  crumble_topping_total_mass flour butter sugar factor = 0.5 :=
by
  sorry

end crumble_topping_correct_amount_l697_69730


namespace jessica_milk_problem_l697_69704

theorem jessica_milk_problem (gallons_owned : ℝ) (gallons_given : ℝ) : gallons_owned = 5 → gallons_given = 16 / 3 → gallons_owned - gallons_given = -(1 / 3) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  -- sorry

end jessica_milk_problem_l697_69704


namespace part_a_part_b_l697_69760

-- Step d: Lean statements for the proof problems
theorem part_a (p : ℕ) (hp : Nat.Prime p) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 :=
by {
  sorry
}

theorem part_b (p : ℕ) (hp : Nat.Prime p) : (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 ∧ a % p ≠ 0 ∧ b % p ≠ 0) ↔ p ≠ 3 :=
by {
  sorry
}

end part_a_part_b_l697_69760


namespace factorize_x2y_minus_4y_l697_69703

variable {x y : ℝ}

theorem factorize_x2y_minus_4y : x^2 * y - 4 * y = y * (x + 2) * (x - 2) :=
sorry

end factorize_x2y_minus_4y_l697_69703


namespace rectangle_within_l697_69724

theorem rectangle_within (a b c d : ℝ) (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
by
  sorry

end rectangle_within_l697_69724


namespace find_m_l697_69755

-- Define the function with given conditions
def f (m : ℕ) (n : ℕ) : ℕ := 
if n > m^2 then n - m + 14 else sorry

-- Define the main problem
theorem find_m (m : ℕ) (hyp : m ≥ 14) : f m 1995 = 1995 ↔ m = 14 ∨ m = 45 :=
by
  sorry

end find_m_l697_69755


namespace simplify_expression_l697_69716

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l697_69716


namespace john_horizontal_distance_l697_69774

theorem john_horizontal_distance
  (vertical_distance_ratio horizontal_distance_ratio : ℕ)
  (initial_elevation final_elevation : ℕ)
  (h_ratio : vertical_distance_ratio = 1)
  (h_dist_ratio : horizontal_distance_ratio = 3)
  (h_initial : initial_elevation = 500)
  (h_final : final_elevation = 3450) :
  (final_elevation - initial_elevation) * horizontal_distance_ratio = 8850 := 
by {
  sorry
}

end john_horizontal_distance_l697_69774


namespace braden_total_money_after_winning_bet_l697_69756

def initial_amount : ℕ := 400
def factor : ℕ := 2
def winnings (initial_amount : ℕ) (factor : ℕ) : ℕ := factor * initial_amount

theorem braden_total_money_after_winning_bet : 
  winnings initial_amount factor + initial_amount = 1200 := 
by
  sorry

end braden_total_money_after_winning_bet_l697_69756


namespace store_A_has_highest_capacity_l697_69797

noncomputable def total_capacity_A : ℕ := 5 * 6 * 9
noncomputable def total_capacity_B : ℕ := 8 * 4 * 7
noncomputable def total_capacity_C : ℕ := 10 * 3 * 8

theorem store_A_has_highest_capacity : total_capacity_A = 270 ∧ total_capacity_A > total_capacity_B ∧ total_capacity_A > total_capacity_C := 
by 
  -- Proof skipped with a placeholder
  sorry

end store_A_has_highest_capacity_l697_69797


namespace euler_criterion_l697_69765

theorem euler_criterion (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (hp_gt_two : p > 2) (ha : 1 ≤ a ∧ a ≤ p - 1) : 
  (∃ b : ℕ, b^2 % p = a % p) ↔ a^((p - 1) / 2) % p = 1 :=
sorry

end euler_criterion_l697_69765


namespace domain_of_f_l697_69719

theorem domain_of_f (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end domain_of_f_l697_69719


namespace relationship_abc_l697_69750

open Real

variable {x : ℝ}
variable (a b c : ℝ)
variable (h1 : 0 < x ∧ x ≤ 1)
variable (h2 : a = (sin x / x) ^ 2)
variable (h3 : b = sin x / x)
variable (h4 : c = sin (x^2) / x^2)

theorem relationship_abc (h1 : 0 < x ∧ x ≤ 1) (h2 : a = (sin x / x) ^ 2) (h3 : b = sin x / x) (h4 : c = sin (x^2) / x^2) :
  a < b ∧ b ≤ c :=
sorry

end relationship_abc_l697_69750


namespace increasing_on_positive_reals_l697_69790

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem increasing_on_positive_reals : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end increasing_on_positive_reals_l697_69790


namespace globe_surface_area_l697_69772

theorem globe_surface_area (d : ℚ) (h : d = 9) : 
  4 * Real.pi * (d / 2) ^ 2 = 81 * Real.pi := 
by 
  sorry

end globe_surface_area_l697_69772


namespace vertical_angles_congruent_l697_69777

-- Define what it means for two angles to be vertical angles
def areVerticalAngles (a b : ℝ) : Prop := -- placeholder definition
  sorry

-- Define what it means for two angles to be congruent
def areCongruentAngles (a b : ℝ) : Prop := a = b

-- State the problem in the form of a theorem
theorem vertical_angles_congruent (a b : ℝ) :
  areVerticalAngles a b → areCongruentAngles a b := by
  sorry

end vertical_angles_congruent_l697_69777


namespace age_problem_l697_69744

theorem age_problem (my_age mother_age : ℕ) 
  (h1 : mother_age = 3 * my_age) 
  (h2 : my_age + mother_age = 40)
  : my_age = 10 :=
by 
  sorry

end age_problem_l697_69744


namespace symmetric_points_x_axis_l697_69722

theorem symmetric_points_x_axis (a b : ℤ) 
  (h1 : a - 1 = 2) (h2 : 5 = -(b - 1)) : (a + b) ^ 2023 = -1 := 
by
  -- The proof steps will go here.
  sorry

end symmetric_points_x_axis_l697_69722


namespace initial_quantity_of_milk_in_container_A_l697_69734

variables {CA MB MC : ℝ}

theorem initial_quantity_of_milk_in_container_A (h1 : MB = 0.375 * CA)
    (h2 : MC = 0.625 * CA)
    (h_eq : MB + 156 = MC - 156) :
    CA = 1248 :=
by
  sorry

end initial_quantity_of_milk_in_container_A_l697_69734


namespace max_squares_covered_by_card_l697_69740

theorem max_squares_covered_by_card (side_len : ℕ) (card_side : ℕ) : 
  side_len = 1 → card_side = 2 → n ≤ 12 :=
by
  sorry

end max_squares_covered_by_card_l697_69740


namespace andrea_sod_rectangles_l697_69795

def section_1_length : ℕ := 35
def section_1_width : ℕ := 42
def section_2_length : ℕ := 55
def section_2_width : ℕ := 86
def section_3_length : ℕ := 20
def section_3_width : ℕ := 50
def section_4_length : ℕ := 48
def section_4_width : ℕ := 66

def sod_length : ℕ := 3
def sod_width : ℕ := 4

def area (length width : ℕ) : ℕ := length * width
def sod_area : ℕ := area sod_length sod_width

def rectangles_needed (section_length section_width sod_area : ℕ) : ℕ :=
  (area section_length section_width + sod_area - 1) / sod_area

def total_rectangles_needed : ℕ :=
  rectangles_needed section_1_length section_1_width sod_area +
  rectangles_needed section_2_length section_2_width sod_area +
  rectangles_needed section_3_length section_3_width sod_area +
  rectangles_needed section_4_length section_4_width sod_area

theorem andrea_sod_rectangles : total_rectangles_needed = 866 := by
  sorry

end andrea_sod_rectangles_l697_69795


namespace average_age_of_family_l697_69717

theorem average_age_of_family :
  let num_grandparents := 2
  let num_parents := 2
  let num_grandchildren := 3
  let avg_age_grandparents := 64
  let avg_age_parents := 39
  let avg_age_grandchildren := 6
  let total_age_grandparents := avg_age_grandparents * num_grandparents
  let total_age_parents := avg_age_parents * num_parents
  let total_age_grandchildren := avg_age_grandchildren * num_grandchildren
  let total_age_family := total_age_grandparents + total_age_parents + total_age_grandchildren
  let num_family_members := num_grandparents + num_parents + num_grandchildren
  let avg_age_family := total_age_family / num_family_members
  avg_age_family = 32 := 
  by 
  repeat { sorry }

end average_age_of_family_l697_69717


namespace find_minimal_N_l697_69754

theorem find_minimal_N (N : ℕ) (l m n : ℕ) (h1 : (l - 1) * (m - 1) * (n - 1) = 252)
  (h2 : l ≥ 5 ∨ m ≥ 5 ∨ n ≥ 5) : N = l * m * n → N = 280 :=
by
  sorry

end find_minimal_N_l697_69754


namespace num_of_poly_sci_majors_l697_69752

-- Define the total number of applicants
def total_applicants : ℕ := 40

-- Define the number of applicants with GPA > 3.0
def gpa_higher_than_3_point_0 : ℕ := 20

-- Define the number of applicants who did not major in political science and had GPA ≤ 3.0
def non_poly_sci_and_low_gpa : ℕ := 10

-- Define the number of political science majors with GPA > 3.0
def poly_sci_with_high_gpa : ℕ := 5

-- Prove the number of political science majors
theorem num_of_poly_sci_majors : ∀ (P : ℕ),
  P = poly_sci_with_high_gpa + 
      (total_applicants - non_poly_sci_and_low_gpa - 
       (gpa_higher_than_3_point_0 - poly_sci_with_high_gpa)) → 
  P = 20 :=
by
  intros P h
  sorry

end num_of_poly_sci_majors_l697_69752


namespace triangular_number_30_l697_69743

theorem triangular_number_30 : (30 * (30 + 1)) / 2 = 465 :=
by
  sorry

end triangular_number_30_l697_69743


namespace similar_triangles_area_ratio_l697_69733

theorem similar_triangles_area_ratio (r : ℚ) (h : r = 1/3) : (r^2) = 1/9 :=
by
  sorry

end similar_triangles_area_ratio_l697_69733


namespace simplify_expression_l697_69773
open Real

theorem simplify_expression (x y : ℝ) : -x + y - 2 * x - 3 * y = -3 * x - 2 * y :=
by
  sorry

end simplify_expression_l697_69773


namespace part1_part2_l697_69784

-- Define the linear function
def linear_function (m x : ℝ) : ℝ := (m - 2) * x + 6

-- Prove part 1: If y increases as x increases, then m > 2
theorem part1 (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → linear_function m x1 < linear_function m x2) → m > 2 :=
sorry

-- Prove part 2: When -2 ≤ x ≤ 4, and y ≤ 10, the range of m is (2, 3] or [0, 2)
theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 4 → linear_function m x ≤ 10) →
  (2 < m ∧ m ≤ 3) ∨ (0 ≤ m ∧ m < 2) :=
sorry

end part1_part2_l697_69784


namespace intersection_A_B_l697_69739

-- Defining the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 3 - x > 0}

-- Stating the theorem that A ∩ B equals (1, 2)
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l697_69739


namespace part_a_l697_69706

theorem part_a (a b c : ℕ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 := 
by sorry

end part_a_l697_69706


namespace domain_sqrt_frac_l697_69731

theorem domain_sqrt_frac (x : ℝ) :
  (x^2 + 4*x + 3 ≠ 0) ∧ (x + 3 ≥ 0) ↔ ((x ∈ Set.Ioc (-3) (-1)) ∨ (x ∈ Set.Ioi (-1))) :=
by
  sorry

end domain_sqrt_frac_l697_69731


namespace joan_change_received_l697_69776

/-- Definition of the cat toy cost -/
def cat_toy_cost : ℝ := 8.77

/-- Definition of the cage cost -/
def cage_cost : ℝ := 10.97

/-- Definition of the total cost -/
def total_cost : ℝ := cat_toy_cost + cage_cost

/-- Definition of the payment amount -/
def payment : ℝ := 20.00

/-- Definition of the change received -/
def change_received : ℝ := payment - total_cost

/-- Statement proving that Joan received $0.26 in change -/
theorem joan_change_received : change_received = 0.26 := by
  sorry

end joan_change_received_l697_69776


namespace total_number_of_workers_l697_69775

theorem total_number_of_workers (W : ℕ) (R : ℕ) 
  (h1 : (7 + R) * 8000 = 7 * 18000 + R * 6000) 
  (h2 : W = 7 + R) : W = 42 :=
by
  -- Proof steps will go here
  sorry

end total_number_of_workers_l697_69775


namespace part1_solution_set_part2_range_of_a_l697_69710

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end part1_solution_set_part2_range_of_a_l697_69710


namespace fat_caterpillars_left_l697_69737

-- Define the initial and the newly hatched caterpillars
def initial_caterpillars : ℕ := 14
def hatched_caterpillars : ℕ := 4

-- Define the caterpillars left on the tree now
def current_caterpillars : ℕ := 10

-- Define the total caterpillars before any left
def total_caterpillars : ℕ := initial_caterpillars + hatched_caterpillars
-- Define the caterpillars leaving the tree
def caterpillars_left : ℕ := total_caterpillars - current_caterpillars

-- The theorem to be proven
theorem fat_caterpillars_left : caterpillars_left = 8 :=
by
  sorry

end fat_caterpillars_left_l697_69737


namespace zeros_in_decimal_representation_l697_69786

def term_decimal_zeros (x : ℚ) : ℕ := sorry  -- Function to calculate the number of zeros in the terminating decimal representation.

theorem zeros_in_decimal_representation :
  term_decimal_zeros (1 / (2^7 * 5^9)) = 8 :=
sorry

end zeros_in_decimal_representation_l697_69786


namespace harbor_distance_l697_69763

-- Definitions from conditions
variable (d : ℝ)

-- Define the assumptions
def condition_dave := d < 10
def condition_elena := d > 9

-- The proof statement that the interval for d is (9, 10)
theorem harbor_distance (hd : condition_dave d) (he : condition_elena d) : d ∈ Set.Ioo 9 10 :=
sorry

end harbor_distance_l697_69763


namespace sum_of_first_six_terms_l697_69785

theorem sum_of_first_six_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (hS : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 2 = 2 → S 4 = 10 → S 6 = 24 := 
by
  intros h1 h2
  sorry

end sum_of_first_six_terms_l697_69785


namespace problem_statement_l697_69705

variable (x P : ℝ)

theorem problem_statement
  (h1 : x^2 - 5 * x + 6 < 0)
  (h2 : P = x^2 + 5 * x + 6) :
  (20 < P) ∧ (P < 30) :=
sorry

end problem_statement_l697_69705


namespace correct_statements_l697_69771

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

def satisfies_condition (f : ℝ → ℝ) : Prop := 
  ∀ x, f (1 - x) + f (1 + x) = 0

def is_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (a + x)

theorem correct_statements (f : ℝ → ℝ) :
  is_even f →
  is_monotonically_increasing f (-1) 0 →
  satisfies_condition f →
  (f (-3) = 0 ∧
   is_monotonically_increasing f 1 2 ∧
   is_symmetric_about_line f 1) :=
by
  intros h_even h_mono h_cond
  sorry

end correct_statements_l697_69771


namespace linear_relationship_increase_in_y_l697_69764

theorem linear_relationship_increase_in_y (x y : ℝ) (hx : x = 12) (hy : y = 10 / 4 * x) : y = 30 := by
  sorry

end linear_relationship_increase_in_y_l697_69764


namespace shaded_area_l697_69769

theorem shaded_area 
  (R r : ℝ) 
  (h_area_larger_circle : π * R ^ 2 = 100 * π) 
  (h_shaded_larger_fraction : 2 / 3 = (area_shaded_larger / (π * R ^ 2))) 
  (h_relationship_radius : r = R / 2) 
  (h_area_smaller_circle : π * r ^ 2 = 25 * π)
  (h_shaded_smaller_fraction : 1 / 3 = (area_shaded_smaller / (π * r ^ 2))) : 
  (area_shaded_larger + area_shaded_smaller = 75 * π) := 
sorry

end shaded_area_l697_69769


namespace g_of_3_equals_5_l697_69723

def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

theorem g_of_3_equals_5 :
  g 3 = 5 :=
by
  sorry

end g_of_3_equals_5_l697_69723


namespace borya_number_l697_69757

theorem borya_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) 
  (h3 : (n * 2 + 5) * 5 = 715) : n = 69 :=
sorry

end borya_number_l697_69757


namespace angle_B_in_triangle_l697_69728

theorem angle_B_in_triangle (A B C : ℝ) (hA : A = 60) (hB : B = 2 * C) (hSum : A + B + C = 180) : B = 80 :=
by sorry

end angle_B_in_triangle_l697_69728


namespace problem_statement_l697_69735

variable (a : ℝ)

theorem problem_statement (h : a^2 + 3*a - 4 = 0) : 2*a^2 + 6*a - 3 = 5 := 
by sorry

end problem_statement_l697_69735


namespace circle_positional_relationship_l697_69712

noncomputable def r1 : ℝ := 2
noncomputable def r2 : ℝ := 3
noncomputable def d : ℝ := 5

theorem circle_positional_relationship :
  d = r1 + r2 → "externally tangent" = "externally tangent" := by
  intro h
  exact rfl

end circle_positional_relationship_l697_69712


namespace fair_coin_three_flips_probability_l697_69787

theorem fair_coin_three_flips_probability :
  ∀ (prob : ℕ → ℚ) (independent : ∀ n, prob n = 1 / 2),
    prob 0 * prob 1 * prob 2 = 1 / 8 := 
by
  intros prob independent
  sorry

end fair_coin_three_flips_probability_l697_69787


namespace ratio_of_m1_m2_l697_69788

open Real

theorem ratio_of_m1_m2 :
  ∀ (m : ℝ) (p q : ℝ), p ≠ 0 ∧ q ≠ 0 ∧ m ≠ 0 ∧
    (p + q = -((3 - 2 * m) / m)) ∧ 
    (p * q = 4 / m) ∧ 
    (p / q + q / p = 2) → 
   ∃ (m1 m2 : ℝ), 
    (4 * m1^2 - 28 * m1 + 9 = 0) ∧
    (4 * m2^2 - 28 * m2 + 9 = 0) ∧ 
    (m1 ≠ m2) ∧ 
    (m1 + m2 = 7) ∧ 
    (m1 * m2 = 9 / 4) ∧ 
    (m1 / m2 + m2 / m1 = 178 / 9) :=
by sorry

end ratio_of_m1_m2_l697_69788


namespace sheep_transaction_gain_l697_69726

noncomputable def percent_gain (cost_per_sheep total_sheep sold_sheep remaining_sheep : ℕ) : ℚ :=
let total_cost := (cost_per_sheep : ℚ) * total_sheep
let initial_revenue := total_cost
let price_per_sheep := initial_revenue / sold_sheep
let remaining_revenue := remaining_sheep * price_per_sheep
let total_revenue := initial_revenue + remaining_revenue
let profit := total_revenue - total_cost
(profit / total_cost) * 100

theorem sheep_transaction_gain :
  percent_gain 1 1000 950 50 = -47.37 := sorry

end sheep_transaction_gain_l697_69726


namespace initial_amount_calc_l697_69738

theorem initial_amount_calc 
  (M : ℝ)
  (H1 : M * 0.3675 = 350) :
  M = 952.38 :=
by
  sorry

end initial_amount_calc_l697_69738


namespace probability_of_seeing_red_light_l697_69718

def red_light_duration : ℝ := 30
def yellow_light_duration : ℝ := 5
def green_light_duration : ℝ := 40

def total_cycle_duration : ℝ := red_light_duration + yellow_light_duration + green_light_duration

theorem probability_of_seeing_red_light :
  (red_light_duration / total_cycle_duration) = 30 / 75 := by
  sorry

end probability_of_seeing_red_light_l697_69718


namespace peaches_sold_to_friends_l697_69725

theorem peaches_sold_to_friends (x : ℕ) (total_peaches : ℕ) (peaches_to_relatives : ℕ) (peach_price_friend : ℕ) (peach_price_relative : ℝ) (total_earnings : ℝ) (peaches_left : ℕ) (total_peaches_sold : ℕ) 
  (h1 : total_peaches = 15) 
  (h2 : peaches_to_relatives = 4) 
  (h3 : peach_price_relative = 1.25) 
  (h4 : total_earnings = 25) 
  (h5 : peaches_left = 1)
  (h6 : total_peaches_sold = 14)
  (h7 : total_earnings = peach_price_friend * x + peach_price_relative * peaches_to_relatives)
  (h8 : total_peaches_sold = total_peaches - peaches_left) :
  x = 10 := 
sorry

end peaches_sold_to_friends_l697_69725


namespace average_speed_last_segment_l697_69748

theorem average_speed_last_segment (D : ℝ) (T_mins : ℝ) (S1 S2 : ℝ) (t : ℝ) (S_last : ℝ) :
  D = 150 ∧ T_mins = 135 ∧ S1 = 50 ∧ S2 = 60 ∧ t = 45 →
  S_last = 90 :=
by
    sorry

end average_speed_last_segment_l697_69748


namespace rachel_total_clothing_l697_69732

def box_1_scarves : ℕ := 2
def box_1_mittens : ℕ := 3
def box_1_hats : ℕ := 1
def box_2_scarves : ℕ := 4
def box_2_mittens : ℕ := 2
def box_2_hats : ℕ := 2
def box_3_scarves : ℕ := 1
def box_3_mittens : ℕ := 5
def box_3_hats : ℕ := 3
def box_4_scarves : ℕ := 3
def box_4_mittens : ℕ := 4
def box_4_hats : ℕ := 1
def box_5_scarves : ℕ := 5
def box_5_mittens : ℕ := 3
def box_5_hats : ℕ := 2
def box_6_scarves : ℕ := 2
def box_6_mittens : ℕ := 6
def box_6_hats : ℕ := 0
def box_7_scarves : ℕ := 4
def box_7_mittens : ℕ := 1
def box_7_hats : ℕ := 3
def box_8_scarves : ℕ := 3
def box_8_mittens : ℕ := 2
def box_8_hats : ℕ := 4
def box_9_scarves : ℕ := 1
def box_9_mittens : ℕ := 4
def box_9_hats : ℕ := 5

def total_clothing : ℕ := 
  box_1_scarves + box_1_mittens + box_1_hats +
  box_2_scarves + box_2_mittens + box_2_hats +
  box_3_scarves + box_3_mittens + box_3_hats +
  box_4_scarves + box_4_mittens + box_4_hats +
  box_5_scarves + box_5_mittens + box_5_hats +
  box_6_scarves + box_6_mittens + box_6_hats +
  box_7_scarves + box_7_mittens + box_7_hats +
  box_8_scarves + box_8_mittens + box_8_hats +
  box_9_scarves + box_9_mittens + box_9_hats

theorem rachel_total_clothing : total_clothing = 76 :=
by
  sorry

end rachel_total_clothing_l697_69732


namespace sector_angle_measure_l697_69759

theorem sector_angle_measure (r α : ℝ) 
  (h1 : 2 * r + α * r = 6)
  (h2 : (1 / 2) * α * r^2 = 2) :
  α = 1 ∨ α = 4 := 
sorry

end sector_angle_measure_l697_69759


namespace problem_statement_l697_69758

variables (x y : ℝ)

theorem problem_statement
  (h1 : abs x = 4)
  (h2 : abs y = 2)
  (h3 : abs (x + y) = x + y) : 
  x - y = 2 ∨ x - y = 6 :=
sorry

end problem_statement_l697_69758


namespace sqrt_abc_sum_eq_54_sqrt_5_l697_69742

theorem sqrt_abc_sum_eq_54_sqrt_5 
  (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 18) 
  (h3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 54 * Real.sqrt 5 := 
by 
  sorry

end sqrt_abc_sum_eq_54_sqrt_5_l697_69742


namespace quadratic_roots_form_l697_69746

theorem quadratic_roots_form {a b c : ℤ} (h : a = 3 ∧ b = -7 ∧ c = 1) :
  ∃ (m n p : ℤ), (∀ x, 3*x^2 - 7*x + 1 = 0 ↔ x = (m + Real.sqrt n)/p ∨ x = (m - Real.sqrt n)/p)
  ∧ Int.gcd m (Int.gcd n p) = 1 ∧ n = 37 :=
by
  sorry

end quadratic_roots_form_l697_69746


namespace subset_implication_l697_69798

noncomputable def M (x : ℝ) : Prop := -2 * x + 1 ≥ 0
noncomputable def N (a x : ℝ) : Prop := x < a

theorem subset_implication (a : ℝ) :
  (∀ x, M x → N a x) → a > 1 / 2 :=
by
  sorry

end subset_implication_l697_69798


namespace algebraic_expression_value_l697_69701

variables {m n : ℝ}

theorem algebraic_expression_value (h : n = 3 - 5 * m) : 10 * m + 2 * n - 3 = 3 :=
by sorry

end algebraic_expression_value_l697_69701


namespace solve_a_solve_inequality_solution_set_l697_69783

theorem solve_a (a : ℝ) :
  (∀ x : ℝ, (1 / 2 < x ∧ x < 2) ↔ ax^2 + 5 * x - 2 > 0) →
  a = -2 :=
by
  sorry

theorem solve_inequality_solution_set (x : ℝ) :
  (a = -2) →
  (2 * x^2 + 5 * x - 3 < 0) ↔
  (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end solve_a_solve_inequality_solution_set_l697_69783


namespace sum_of_three_squares_l697_69736

theorem sum_of_three_squares (a b c : ℤ) (h1 : 2 * a + 2 * b + c = 27) (h2 : a + 3 * b + c = 25) : 3 * c = 33 :=
  sorry

end sum_of_three_squares_l697_69736


namespace probability_of_X_eq_2_l697_69761

-- Define the random variable distribution condition
def random_variable_distribution (a : ℝ) (P : ℝ → ℝ) : Prop :=
  P 1 = 1 / (2 * a) ∧ P 2 = 2 / (2 * a) ∧ P 3 = 3 / (2 * a) ∧
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) = 1)

-- State the theorem given the conditions and the result
theorem probability_of_X_eq_2 (a : ℝ) (P : ℝ → ℝ) (h : random_variable_distribution a P) : 
  P 2 = 1 / 3 :=
sorry

end probability_of_X_eq_2_l697_69761


namespace domain_correct_l697_69768

def domain_of_function (x : ℝ) : Prop :=
  (x > 2) ∧ (x ≠ 5)

theorem domain_correct : {x : ℝ | domain_of_function x} = {x : ℝ | x > 2 ∧ x ≠ 5} :=
by
  sorry

end domain_correct_l697_69768


namespace total_cost_of_books_l697_69700

theorem total_cost_of_books (total_children : ℕ) (n : ℕ) (extra_payment_per_child : ℕ) (cost : ℕ) :
  total_children = 12 →
  n = 2 →
  extra_payment_per_child = 10 →
  (total_children - n) * extra_payment_per_child = 100 →
  cost = 600 :=
by
  intros h1 h2 h3 h4
  sorry

end total_cost_of_books_l697_69700


namespace soccer_team_wins_l697_69794

theorem soccer_team_wins :
  ∃ W D : ℕ, 
    (W + 2 + D = 20) ∧  -- total games
    (3 * W + D = 46) ∧  -- total points
    (W = 14) :=         -- correct answer
by
  sorry

end soccer_team_wins_l697_69794


namespace unique_identity_element_l697_69762

variable {G : Type*} [Group G]

theorem unique_identity_element (e e' : G) (h1 : ∀ g : G, e * g = g ∧ g * e = g) (h2 : ∀ g : G, e' * g = g ∧ g * e' = g) : e = e' :=
by 
sorry

end unique_identity_element_l697_69762


namespace integer_solutions_l697_69789

theorem integer_solutions (x : ℤ) : 
  (⌊(x : ℚ) / 2⌋ * ⌊(x : ℚ) / 3⌋ * ⌊(x : ℚ) / 4⌋ = x^2) ↔ (x = 0 ∨ x = 24) := 
sorry

end integer_solutions_l697_69789


namespace problem_divisible_by_64_l697_69778

theorem problem_divisible_by_64 (n : ℕ) (hn : n > 0) : (3 ^ (2 * n + 2) - 8 * n - 9) % 64 = 0 := 
by
  sorry

end problem_divisible_by_64_l697_69778


namespace smallest_period_sin_cos_l697_69741

theorem smallest_period_sin_cos (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x + Real.cos x) :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 2 * Real.pi :=
sorry

end smallest_period_sin_cos_l697_69741


namespace juice_oranges_l697_69770

theorem juice_oranges (oranges_per_glass : ℕ) (glasses : ℕ) (total_oranges : ℕ)
  (h1 : oranges_per_glass = 3)
  (h2 : glasses = 10)
  (h3 : total_oranges = oranges_per_glass * glasses) :
  total_oranges = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end juice_oranges_l697_69770


namespace point_B_value_l697_69709

theorem point_B_value :
  ∃ B : ℝ, (|B + 1| = 4) ∧ (B = 3 ∨ B = -5) := 
by
  sorry

end point_B_value_l697_69709


namespace total_social_media_hours_in_a_week_l697_69745

variable (daily_social_media_hours : ℕ) (days_in_week : ℕ)

theorem total_social_media_hours_in_a_week
(h1 : daily_social_media_hours = 3)
(h2 : days_in_week = 7) :
daily_social_media_hours * days_in_week = 21 := by
  sorry

end total_social_media_hours_in_a_week_l697_69745


namespace gcd_8Tn_nplus1_eq_4_l697_69727

noncomputable def T_n (n : ℕ) : ℕ :=
(n * (n + 1)) / 2

theorem gcd_8Tn_nplus1_eq_4 (n : ℕ) (hn: 0 < n) : gcd (8 * T_n n) (n + 1) = 4 :=
sorry

end gcd_8Tn_nplus1_eq_4_l697_69727


namespace completing_square_to_simplify_eq_l697_69791

theorem completing_square_to_simplify_eq : 
  ∃ (c : ℝ), (∀ x : ℝ, x^2 - 6 * x + 4 = 0 ↔ (x - 3)^2 = c) :=
by
  use 5
  intro x
  constructor
  { intro h
    -- proof conversion process (skipped)
    sorry }
  { intro h
    -- reverse proof process (skipped)
    sorry }

end completing_square_to_simplify_eq_l697_69791


namespace second_smallest_is_3_probability_l697_69796

noncomputable def probability_of_second_smallest_is_3 : ℚ := 
  let total_ways := Nat.choose 10 6
  let favorable_ways := 2 * Nat.choose 7 4
  favorable_ways / total_ways

theorem second_smallest_is_3_probability : probability_of_second_smallest_is_3 = 1 / 3 := sorry

end second_smallest_is_3_probability_l697_69796


namespace tan_ineq_solution_l697_69793

theorem tan_ineq_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x, x = a * Real.pi → ¬ (Real.tan x = a * Real.pi)) :
    {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2}
    = {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2} := sorry

end tan_ineq_solution_l697_69793


namespace initial_music_files_eq_sixteen_l697_69714

theorem initial_music_files_eq_sixteen (M : ℕ) :
  (M + 48 - 30 = 34) → (M = 16) :=
by
  sorry

end initial_music_files_eq_sixteen_l697_69714


namespace product_a2_a3_a4_l697_69749

open Classical

noncomputable def geometric_sequence (a : ℕ → ℚ) (a1 : ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n - 1)

theorem product_a2_a3_a4 (a : ℕ → ℚ) (q : ℚ) 
  (h_seq : geometric_sequence a 1 q)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 1 / 9) :
  a 2 * a 3 * a 4 = 1 / 27 :=
sorry

end product_a2_a3_a4_l697_69749


namespace problem1_l697_69767

   theorem problem1 : (Real.sqrt (9 / 4) + |2 - Real.sqrt 3| - (64 : ℝ) ^ (1 / 3) + 2⁻¹) = -Real.sqrt 3 :=
   by
     sorry
   
end problem1_l697_69767


namespace total_bears_l697_69766

-- Definitions based on given conditions
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27

-- Theorem to prove the total number of bears
theorem total_bears : brown_bears + white_bears + black_bears = 66 := by
  sorry

end total_bears_l697_69766


namespace man_speed_is_correct_l697_69747

noncomputable def train_length : ℝ := 275
noncomputable def train_speed_kmh : ℝ := 60
noncomputable def time_seconds : ℝ := 14.998800095992323

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
noncomputable def relative_speed_ms : ℝ := train_length / time_seconds
noncomputable def man_speed_ms : ℝ := relative_speed_ms - train_speed_ms
noncomputable def man_speed_kmh : ℝ := man_speed_ms * (3600 / 1000)
noncomputable def expected_man_speed_kmh : ℝ := 6.006

theorem man_speed_is_correct : abs (man_speed_kmh - expected_man_speed_kmh) < 0.001 :=
by
  -- proof goes here
  sorry

end man_speed_is_correct_l697_69747


namespace Mandy_older_than_Jackson_l697_69713

variable (M J A : ℕ)

-- Given conditions
variables (h1 : J = 20)
variables (h2 : A = (3 * J) / 4)
variables (h3 : (M + 10) + (J + 10) + (A + 10) = 95)

-- Prove that Mandy is 10 years older than Jackson
theorem Mandy_older_than_Jackson : M - J = 10 :=
by
  sorry

end Mandy_older_than_Jackson_l697_69713


namespace travel_cost_AB_l697_69702

theorem travel_cost_AB
  (distance_AB : ℕ)
  (booking_fee : ℕ)
  (cost_per_km_flight : ℝ)
  (correct_total_cost : ℝ)
  (h1 : distance_AB = 4000)
  (h2 : booking_fee = 150)
  (h3 : cost_per_km_flight = 0.12) :
  correct_total_cost = 630 :=
by
  sorry

end travel_cost_AB_l697_69702


namespace find_slope_l697_69707

theorem find_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : ∃ m : ℝ, m = -4/7 ∧ (∀ x, y = m * x + 4) := 
by
  sorry

end find_slope_l697_69707


namespace smallest_M_l697_69781

def Q (M : ℕ) := (2 * M / 3 + 1) / (M + 1)

theorem smallest_M (M : ℕ) (h : M % 6 = 0) (h_pos : 0 < M) : 
  (∃ k, M = 6 * k ∧ Q M < 3 / 4) ↔ M = 6 := 
by 
  sorry

end smallest_M_l697_69781


namespace average_of_four_given_conditions_l697_69753

noncomputable def average_of_four_integers : ℕ × ℕ × ℕ × ℕ → ℚ :=
  λ ⟨a, b, c, d⟩ => (a + b + c + d : ℚ) / 4

theorem average_of_four_given_conditions :
  ∀ (A B C D : ℕ), 
    (A + B) / 2 = 35 → 
    C = 130 → 
    D = 1 → 
    average_of_four_integers (A, B, C, D) = 50.25 := 
by
  intros A B C D hAB hC hD
  unfold average_of_four_integers
  sorry

end average_of_four_given_conditions_l697_69753


namespace bigger_part_of_dividing_56_l697_69751

theorem bigger_part_of_dividing_56 (x y : ℕ) (h₁ : x + y = 56) (h₂ : 10 * x + 22 * y = 780) : max x y = 38 :=
by
  sorry

end bigger_part_of_dividing_56_l697_69751


namespace complement_union_l697_69779

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l697_69779


namespace compare_a_b_l697_69711

theorem compare_a_b (a b : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) : a < b :=
by {
  sorry -- We'll leave the proof as a placeholder.
}

end compare_a_b_l697_69711


namespace total_profit_l697_69708

variable (A_s B_s C_s : ℝ)
variable (A_p : ℝ := 14700)
variable (P : ℝ)

theorem total_profit
  (h1 : A_s + B_s + C_s = 50000)
  (h2 : A_s = B_s + 4000)
  (h3 : B_s = C_s + 5000)
  (h4 : A_p = 14700) :
  P = 35000 :=
sorry

end total_profit_l697_69708


namespace frog_jumps_further_l697_69720

-- Given conditions
def grasshopper_jump : ℕ := 9 -- The grasshopper jumped 9 inches
def frog_jump : ℕ := 12 -- The frog jumped 12 inches

-- Proof statement
theorem frog_jumps_further : frog_jump - grasshopper_jump = 3 := by
  sorry

end frog_jumps_further_l697_69720


namespace original_length_equals_13_l697_69792

-- Definitions based on conditions
def original_width := 18
def increased_length (x : ℕ) := x + 2
def increased_width := 20

-- Total area condition
def total_area (x : ℕ) := 
  4 * ((increased_length x) * increased_width) + 2 * ((increased_length x) * increased_width)

theorem original_length_equals_13 (x : ℕ) (h : total_area x = 1800) : x = 13 := 
by
  sorry

end original_length_equals_13_l697_69792


namespace tim_prank_combinations_l697_69715

def number_of_combinations (monday_choices : ℕ) (tuesday_choices : ℕ) (wednesday_choices : ℕ) (thursday_choices : ℕ) (friday_choices : ℕ) : ℕ :=
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations 2 3 0 6 1 = 0 :=
by
  -- Calculation yields 2 * 3 * 0 * 6 * 1 = 0
  sorry

end tim_prank_combinations_l697_69715
