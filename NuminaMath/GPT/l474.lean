import Mathlib

namespace arithmetic_sequence_a7_value_l474_47455

variable (a : ℕ → ℝ) (a1 a13 a7 : ℝ)

theorem arithmetic_sequence_a7_value
  (h1 : a 1 = a1)
  (h13 : a 13 = a13)
  (h_sum : a1 + a13 = 12)
  (h_arith : 2 * a7 = a1 + a13) :
  a7 = 6 :=
by
  sorry

end arithmetic_sequence_a7_value_l474_47455


namespace exists_root_in_interval_l474_47429

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x - 3

theorem exists_root_in_interval : ∃ c ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f c = 0 :=
by
  sorry

end exists_root_in_interval_l474_47429


namespace vector_x_solution_l474_47479

theorem vector_x_solution (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (-2,0))
  (hb : b = (2,1))
  (hc : c = (x,1))
  (collinear : ∃ k : ℝ, 3 • a + b = k • c) :
  x = -4 :=
by
  sorry

end vector_x_solution_l474_47479


namespace log_expression_simplify_l474_47498

variable (x y : ℝ)

theorem log_expression_simplify (hx : 0 < x) (hx' : x ≠ 1) (hy : 0 < y) (hy' : y ≠ 1) :
  (Real.log x^2 / Real.log y^4) * 
  (Real.log y^3 / Real.log x^3) * 
  (Real.log x^4 / Real.log y^5) * 
  (Real.log y^5 / Real.log x^2) * 
  (Real.log x^3 / Real.log y^3) = (1 / 3) * Real.log x / Real.log y := 
sorry

end log_expression_simplify_l474_47498


namespace find_value_of_x_plus_5_l474_47430

-- Define a variable x
variable (x : ℕ)

-- Define the condition given in the problem
def condition := x - 10 = 15

-- The statement we need to prove
theorem find_value_of_x_plus_5 (h : x - 10 = 15) : x + 5 = 30 := 
by sorry

end find_value_of_x_plus_5_l474_47430


namespace largest_c_value_l474_47493

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, x^2 + 5 * x + c = -3) → c ≤ 13 / 4 :=
sorry

end largest_c_value_l474_47493


namespace max_matching_pairs_l474_47400

theorem max_matching_pairs 
  (total_pairs : ℕ := 23) 
  (total_colors : ℕ := 6) 
  (total_sizes : ℕ := 3) 
  (lost_shoes : ℕ := 9)
  (shoes_per_pair : ℕ := 2) 
  (total_shoes := total_pairs * shoes_per_pair) 
  (remaining_shoes := total_shoes - lost_shoes) :
  ∃ max_pairs : ℕ, max_pairs = total_pairs - lost_shoes / shoes_per_pair :=
sorry

end max_matching_pairs_l474_47400


namespace remainder_when_13_plus_x_divided_by_26_l474_47490

theorem remainder_when_13_plus_x_divided_by_26 (x : ℕ) (h1 : 9 * x % 26 = 1) : (13 + x) % 26 = 16 := 
by sorry

end remainder_when_13_plus_x_divided_by_26_l474_47490


namespace striped_octopus_has_8_legs_l474_47465

-- Definitions for Octopus and Statements
structure Octopus :=
  (legs : ℕ)
  (tellsTruth : Prop)

-- Given conditions translations
def tellsTruthCondition (o : Octopus) : Prop :=
  if o.legs % 2 = 0 then o.tellsTruth else ¬o.tellsTruth

def green_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def dark_blue_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def violet_octopus : Octopus :=
  { legs := 9, tellsTruth := sorry }  -- Placeholder truth value

def striped_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

-- Octopus statements (simplified for output purposes)
def green_statement := (green_octopus.legs = 8) ∧ (dark_blue_octopus.legs = 6)
def dark_blue_statement := (dark_blue_octopus.legs = 8) ∧ (green_octopus.legs = 7)
def violet_statement := (dark_blue_octopus.legs = 8) ∧ (violet_octopus.legs = 9)
def striped_statement := ¬(green_octopus.legs = 8 ∨ dark_blue_octopus.legs = 8 ∨ violet_octopus.legs = 8) ∧ (striped_octopus.legs = 8)

-- The goal to prove that the striped octopus has exactly 8 legs
theorem striped_octopus_has_8_legs : striped_octopus.legs = 8 :=
sorry

end striped_octopus_has_8_legs_l474_47465


namespace identity_holds_l474_47473

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l474_47473


namespace domain_of_f_equals_l474_47480

noncomputable def domain_of_function := {x : ℝ | x > -1 ∧ -(x+4) * (x-1) > 0}

theorem domain_of_f_equals : domain_of_function = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end domain_of_f_equals_l474_47480


namespace axis_of_symmetry_range_l474_47405

theorem axis_of_symmetry_range (a : ℝ) : (-(a + 2) / (3 - 4 * a) > 0) ↔ (a < -2 ∨ a > 3 / 4) :=
by
  sorry

end axis_of_symmetry_range_l474_47405


namespace fraction_multiplication_division_l474_47435

theorem fraction_multiplication_division :
  ((3 / 4) * (5 / 6)) / (7 / 8) = 5 / 7 :=
by
  sorry

end fraction_multiplication_division_l474_47435


namespace base7_divisibility_rules_2_base7_divisibility_rules_3_l474_47403

def divisible_by_2 (d : Nat) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4

def divisible_by_3 (d : Nat) : Prop :=
  d = 0 ∨ d = 3

def last_digit_base7 (n : Nat) : Nat :=
  n % 7

theorem base7_divisibility_rules_2 (n : Nat) :
  (∃ k, n = 2 * k) ↔ divisible_by_2 (last_digit_base7 n) :=
by
  sorry

theorem base7_divisibility_rules_3 (n : Nat) :
  (∃ k, n = 3 * k) ↔ divisible_by_3 (last_digit_base7 n) :=
by
  sorry

end base7_divisibility_rules_2_base7_divisibility_rules_3_l474_47403


namespace additional_houses_built_by_october_l474_47410

def total_houses : ℕ := 2000
def fraction_built_first_half : ℚ := 3 / 5
def houses_needed_by_october : ℕ := 500

def houses_built_first_half : ℚ := fraction_built_first_half * total_houses
def houses_built_by_october : ℕ := total_houses - houses_needed_by_october

theorem additional_houses_built_by_october :
  (houses_built_by_october - houses_built_first_half) = 300 := by
  sorry

end additional_houses_built_by_october_l474_47410


namespace youngest_child_age_l474_47477

theorem youngest_child_age {x : ℝ} (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by
  sorry

end youngest_child_age_l474_47477


namespace commensurable_iff_rat_l474_47425

def commensurable (A B : ℝ) : Prop :=
  ∃ d : ℝ, ∃ m n : ℤ, A = m * d ∧ B = n * d

theorem commensurable_iff_rat (A B : ℝ) :
  commensurable A B ↔ ∃ (m n : ℤ) (h : n ≠ 0), A / B = m / n :=
by
  sorry

end commensurable_iff_rat_l474_47425


namespace willy_crayons_difference_l474_47420

def willy : Int := 5092
def lucy : Int := 3971
def jake : Int := 2435

theorem willy_crayons_difference : willy - (lucy + jake) = -1314 := by
  sorry

end willy_crayons_difference_l474_47420


namespace value_of_a_minus_b_l474_47412

theorem value_of_a_minus_b (a b : ℝ)
  (h1 : ∃ (x : ℝ), x = 3 ∧ (ax / (x - 1)) = 1)
  (h2 : ∀ (x : ℝ), (ax / (x - 1)) < 1 ↔ (x < b ∨ x > 3)) :
  a - b = -1 / 3 :=
by
  sorry

end value_of_a_minus_b_l474_47412


namespace cheryl_material_left_l474_47496

def square_yards_left (bought1 bought2 used : ℚ) : ℚ :=
  bought1 + bought2 - used

theorem cheryl_material_left :
  square_yards_left (4/19) (2/13) (0.21052631578947367 : ℚ) = (0.15384615384615385 : ℚ) :=
by
  sorry

end cheryl_material_left_l474_47496


namespace product_of_solutions_of_quadratic_l474_47453

theorem product_of_solutions_of_quadratic :
  ∀ (x p q : ℝ), 36 - 9 * x - x^2 = 0 ∧ (x = p ∨ x = q) → p * q = -36 :=
by sorry

end product_of_solutions_of_quadratic_l474_47453


namespace evaluate_expression_l474_47426

theorem evaluate_expression : 
  let a := 2
  let b := 1 / 2
  2 * (a^2 - 2 * a * b) - 3 * (a^2 - a * b - 4 * b^2) = -2 :=
by
  let a := 2
  let b := 1 / 2
  sorry

end evaluate_expression_l474_47426


namespace systematic_sampling_student_selection_l474_47421

theorem systematic_sampling_student_selection
    (total_students : ℕ)
    (num_groups : ℕ)
    (students_per_group : ℕ)
    (third_group_selected : ℕ)
    (third_group_num : ℕ)
    (eighth_group_num : ℕ)
    (h1 : total_students = 50)
    (h2 : num_groups = 10)
    (h3 : students_per_group = total_students / num_groups)
    (h4 : students_per_group = 5)
    (h5 : 11 ≤ third_group_selected ∧ third_group_selected ≤ 15)
    (h6 : third_group_selected = 12)
    (h7 : third_group_num = 3)
    (h8 : eighth_group_num = 8) :
  eighth_group_selected = 37 :=
by
  sorry

end systematic_sampling_student_selection_l474_47421


namespace paper_area_difference_l474_47483

def area (length width : ℕ) : ℕ := length * width

def combined_area (length width : ℕ) : ℕ := 2 * (area length width)

def sq_inch_to_sq_ft (sq_inch : ℕ) : ℕ := sq_inch / 144

theorem paper_area_difference :
  sq_inch_to_sq_ft (combined_area 15 24 - combined_area 12 18) = 2 :=
by
  sorry

end paper_area_difference_l474_47483


namespace price_of_brand_Y_pen_l474_47433

theorem price_of_brand_Y_pen (cost_X : ℝ) (num_X : ℕ) (total_pens : ℕ) (total_cost : ℝ) :
  cost_X = 4 ∧ num_X = 6 ∧ total_pens = 12 ∧ total_cost = 42 →
  (∃ (price_Y : ℝ), price_Y = 3) :=
by
  sorry

end price_of_brand_Y_pen_l474_47433


namespace gcd_of_product_diff_is_12_l474_47424

theorem gcd_of_product_diff_is_12
  (a b c d : ℤ) : ∃ (D : ℤ), D = 12 ∧
  ∀ (a b c d : ℤ), D ∣ (b - a) * (c - b) * (d - c) * (d - a) * (c - a) * (d - b) :=
by
  use 12
  sorry

end gcd_of_product_diff_is_12_l474_47424


namespace lcm_of_two_numbers_l474_47441

theorem lcm_of_two_numbers (A B : ℕ) 
  (h_prod : A * B = 987153000) 
  (h_hcf : Int.gcd A B = 440) : 
  Nat.lcm A B = 2243525 :=
by
  sorry

end lcm_of_two_numbers_l474_47441


namespace lines_perpendicular_l474_47485

theorem lines_perpendicular
  (k₁ k₂ : ℝ)
  (h₁ : k₁^2 - 3*k₁ - 1 = 0)
  (h₂ : k₂^2 - 3*k₂ - 1 = 0) :
  k₁ * k₂ = -1 → 
  (∃ l₁ l₂: ℝ → ℝ, 
    ∀ x, l₁ x = k₁ * x ∧ l₂ x = k₂ * x → 
    ∃ m, m = -1) := 
sorry

end lines_perpendicular_l474_47485


namespace max_enclosed_area_perimeter_160_length_twice_width_l474_47408

theorem max_enclosed_area_perimeter_160_length_twice_width 
  (W L : ℕ) 
  (h1 : 2 * (L + W) = 160) 
  (h2 : L = 2 * W) : 
  L * W = 1352 := 
sorry

end max_enclosed_area_perimeter_160_length_twice_width_l474_47408


namespace tank_capacity_l474_47463

theorem tank_capacity :
  (∃ c: ℝ, (∃ w: ℝ, w / c = 1/6 ∧ (w + 5) / c = 1/3) → c = 30) :=
by
  sorry

end tank_capacity_l474_47463


namespace find_sin_θ_find_cos_2θ_find_cos_φ_l474_47460

noncomputable def θ : ℝ := sorry
noncomputable def φ : ℝ := sorry

-- Conditions
axiom cos_eq : Real.cos θ = Real.sqrt 5 / 5
axiom θ_in_quadrant_I : 0 < θ ∧ θ < Real.pi / 2
axiom sin_diff_eq : Real.sin (θ - φ) = Real.sqrt 10 / 10
axiom φ_in_quadrant_I : 0 < φ ∧ φ < Real.pi / 2

-- Goals
-- Part (I) Prove the value of sin θ
theorem find_sin_θ : Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by
  sorry

-- Part (II) Prove the value of cos 2θ
theorem find_cos_2θ : Real.cos (2 * θ) = -3 / 5 :=
by
  sorry

-- Part (III) Prove the value of cos φ
theorem find_cos_φ : Real.cos φ = Real.sqrt 2 / 2 :=
by
  sorry

end find_sin_θ_find_cos_2θ_find_cos_φ_l474_47460


namespace certain_number_exists_l474_47474

theorem certain_number_exists (a b : ℝ) (C : ℝ) (h1 : a ≠ b) (h2 : a + b = 4) (h3 : a * (a - 4) = C) (h4 : b * (b - 4) = C) : 
  C = -3 := 
sorry

end certain_number_exists_l474_47474


namespace place_signs_correct_l474_47440

theorem place_signs_correct :
  1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99 :=
by
  sorry

end place_signs_correct_l474_47440


namespace A_share_value_l474_47470

-- Define the shares using the common multiplier x
variable (x : ℝ)

-- Define the shares in terms of x
def A_share := 5 * x
def B_share := 2 * x
def C_share := 4 * x
def D_share := 3 * x

-- Given condition that C gets Rs. 500 more than D
def condition := C_share - D_share = 500

-- State the theorem to determine A's share given the conditions
theorem A_share_value (h : condition) : A_share = 2500 := by 
  sorry

end A_share_value_l474_47470


namespace isosceles_triangle_angle_measure_l474_47427

theorem isosceles_triangle_angle_measure:
  ∀ (α β : ℝ), (α = 112.5) → (2 * β + α = 180) → β = 33.75 :=
by
  intros α β hα h_sum
  sorry

end isosceles_triangle_angle_measure_l474_47427


namespace max_S_at_n_four_l474_47482

-- Define the sequence sum S_n
def S (n : ℕ) : ℤ := -(n^2 : ℤ) + (8 * n : ℤ)

-- Prove that S_n attains its maximum value at n = 4
theorem max_S_at_n_four : ∀ n : ℕ, S n ≤ S 4 :=
by
  sorry

end max_S_at_n_four_l474_47482


namespace geometric_sequence_sum_l474_47434

variable {α : Type*} [NormedField α] [CompleteSpace α]

def geometric_sum (a r : α) (n : ℕ) : α :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (S : ℕ → α) (a r : α) (hS : ∀ n, S n = geometric_sum a r n) :
  S 2 = 6 → S 4 = 30 → S 6 = 126 :=
by
  sorry

end geometric_sequence_sum_l474_47434


namespace minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l474_47464

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  sorry

theorem minimum_value_achieved : ∃ x : ℝ, f x = 3 := by
  sorry

theorem sum_of_squares_ge_three (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := by
  sorry

end minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l474_47464


namespace total_time_last_two_videos_l474_47432

theorem total_time_last_two_videos
  (first_video_length : ℕ := 2 * 60)
  (second_video_length : ℕ := 4 * 60 + 30)
  (total_time : ℕ := 510) :
  ∃ t1 t2 : ℕ, t1 ≠ t2 ∧ t1 + t2 = total_time - first_video_length - second_video_length := by
  sorry

end total_time_last_two_videos_l474_47432


namespace alternating_sum_of_coefficients_l474_47404

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (2 * x + 1)^5

theorem alternating_sum_of_coefficients :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), polynomial_expansion x = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = -1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h
  sorry

end alternating_sum_of_coefficients_l474_47404


namespace product_of_base8_digits_of_5432_l474_47472

open Nat

def base8_digits (n : ℕ) : List ℕ :=
  let rec digits_helper (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc
    else digits_helper (n / 8) ((n % 8) :: acc)
  digits_helper n []

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_base8_digits_of_5432 : 
    product_of_digits (base8_digits 5432) = 0 :=
by
  sorry

end product_of_base8_digits_of_5432_l474_47472


namespace inequality_proof_l474_47487

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a^3 + b^3 + c^3 = 3) :
  (1 / (a^4 + 3) + 1 / (b^4 + 3) + 1 / (c^4 + 3) >= 3 / 4) :=
by
  sorry

end inequality_proof_l474_47487


namespace quadratic_inequality_solution_l474_47418

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 4 * x > 45 ↔ x < -9 ∨ x > 5 := 
  sorry

end quadratic_inequality_solution_l474_47418


namespace find_a_l474_47401

open Complex

theorem find_a (a : ℝ) (i : ℂ := Complex.I) (h : (a - i) ^ 2 = 2 * i) : a = -1 :=
sorry

end find_a_l474_47401


namespace minimum_containers_needed_l474_47423

-- Definition of the problem conditions
def container_sizes := [5, 10, 20]
def target_units := 85

-- Proposition stating the minimum number of containers required
theorem minimum_containers_needed : 
  ∃ (x y z : ℕ), 
    5 * x + 10 * y + 20 * z = target_units ∧ 
    x + y + z = 5 :=
sorry

end minimum_containers_needed_l474_47423


namespace bertha_descendants_without_daughters_l474_47459

-- Definitions based on conditions
def num_daughters : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30
def daughters_with_daughters := (total_daughters_and_granddaughters - num_daughters) / 6

-- The number of Bertha's daughters who have no daughters:
def daughters_without_daughters := num_daughters - daughters_with_daughters
-- The number of Bertha's granddaughters:
def num_granddaughters := total_daughters_and_granddaughters - num_daughters
-- All granddaughters have no daughters:
def granddaughters_without_daughters := num_granddaughters

-- The total number of daughters and granddaughters without daughters
def total_without_daughters := daughters_without_daughters + granddaughters_without_daughters

-- Main theorem statement
theorem bertha_descendants_without_daughters :
  total_without_daughters = 26 :=
by
  sorry

end bertha_descendants_without_daughters_l474_47459


namespace evaluate_x2_plus_y2_plus_z2_l474_47415

theorem evaluate_x2_plus_y2_plus_z2 (x y z : ℤ) 
  (h1 : x^2 * y + y^2 * z + z^2 * x = 2186)
  (h2 : x * y^2 + y * z^2 + z * x^2 = 2188) 
  : x^2 + y^2 + z^2 = 245 := 
sorry

end evaluate_x2_plus_y2_plus_z2_l474_47415


namespace daves_apps_count_l474_47442

theorem daves_apps_count (x : ℕ) : 
  let initial_apps : ℕ := 21
  let added_apps : ℕ := 89
  let total_apps : ℕ := initial_apps + added_apps
  let deleted_apps : ℕ := x
  let more_added_apps : ℕ := x + 3
  total_apps - deleted_apps + more_added_apps = 113 :=
by
  sorry

end daves_apps_count_l474_47442


namespace rain_forest_animals_l474_47439

theorem rain_forest_animals (R : ℕ) 
  (h1 : 16 = 3 * R - 5) : R = 7 := 
  by sorry

end rain_forest_animals_l474_47439


namespace total_sections_l474_47456

theorem total_sections (boys girls max_students per_section boys_ratio girls_ratio : ℕ)
  (hb : boys = 408) (hg : girls = 240) (hm : max_students = 24) 
  (br : boys_ratio = 3) (gr : girls_ratio = 2)
  (hboy_sec : (boys + max_students - 1) / max_students = 17)
  (hgirl_sec : (girls + max_students - 1) / max_students = 10) 
  : (3 * (((boys + max_students - 1) / max_students) + 2 * ((girls + max_students - 1) / max_students))) / 5 = 30 :=
by
  sorry

end total_sections_l474_47456


namespace jennifer_boxes_l474_47413

theorem jennifer_boxes (kim_sold : ℕ) (h₁ : kim_sold = 54) (h₂ : ∃ jennifer_sold, jennifer_sold = kim_sold + 17) : ∃ jennifer_sold, jennifer_sold = 71 := by
  sorry

end jennifer_boxes_l474_47413


namespace area_convex_quadrilateral_l474_47497

theorem area_convex_quadrilateral (x y : ℝ) :
  (x^2 + y^2 = 73 ∧ x * y = 24) →
  -- You can place a formal statement specifying the four vertices here if needed
  ∃ a b c d : ℝ × ℝ,
  a.1^2 + a.2^2 = 73 ∧
  a.1 * a.2 = 24 ∧
  b.1^2 + b.2^2 = 73 ∧
  b.1 * b.2 = 24 ∧
  c.1^2 + c.2^2 = 73 ∧
  c.1 * c.2 = 24 ∧
  d.1^2 + d.2^2 = 73 ∧
  d.1 * d.2 = 24 ∧
  -- Ensure the quadrilateral forms a rectangle (additional conditions here)
  -- Compute the side lengths and area
  -- Specify finally the area and prove it equals 110
  True :=
sorry

end area_convex_quadrilateral_l474_47497


namespace employed_females_percentage_l474_47469

variable (P : ℝ) -- Total population of town X
variable (E_P : ℝ) -- Percentage of the population that is employed
variable (M_E_P : ℝ) -- Percentage of the population that are employed males

-- Conditions
axiom h1 : E_P = 0.64
axiom h2 : M_E_P = 0.55

-- Target: Prove the percentage of employed people in town X that are females
theorem employed_females_percentage (h : P > 0) : 
  (E_P * P - M_E_P * P) / (E_P * P) * 100 = 14.06 := by
sorry

end employed_females_percentage_l474_47469


namespace dot_product_of_vectors_l474_47447

theorem dot_product_of_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  a.1 * b.1 + a.2 * b.2 = -4 :=
by
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  sorry

end dot_product_of_vectors_l474_47447


namespace cube_vertex_plane_distance_l474_47446

theorem cube_vertex_plane_distance
  (d : ℝ)
  (h_dist : d = 9 - Real.sqrt 186)
  (h7 : ∀ (a b c  : ℝ), a^2 + b^2 + c^2 = 1 → 64 * (a^2 + b^2 + c^2) = 64)
  (h8 : ∀ (d : ℝ), 3 * d^2 - 54 * d + 181 = 0) :
  ∃ (p q r : ℕ), 
    p = 27 ∧ q = 186 ∧ r = 3 ∧ (p + q + r < 1000) ∧ (d = (p - Real.sqrt q) / r) := 
  by
    sorry

end cube_vertex_plane_distance_l474_47446


namespace ryan_lost_initially_l474_47454

-- Define the number of leaves initially collected
def initial_leaves : ℤ := 89

-- Define the number of leaves broken afterwards
def broken_leaves : ℤ := 43

-- Define the number of leaves left in the collection
def remaining_leaves : ℤ := 22

-- Define the lost leaves
def lost_leaves (L : ℤ) : Prop :=
  initial_leaves - L - broken_leaves = remaining_leaves

theorem ryan_lost_initially : ∃ L : ℤ, lost_leaves L ∧ L = 24 :=
by
  sorry

end ryan_lost_initially_l474_47454


namespace additional_license_plates_l474_47414

def original_license_plates : ℕ := 5 * 3 * 5
def new_license_plates : ℕ := 6 * 4 * 5

theorem additional_license_plates : new_license_plates - original_license_plates = 45 := by
  sorry

end additional_license_plates_l474_47414


namespace closest_point_on_plane_exists_l474_47417

def point_on_plane : Type := {P : ℝ × ℝ × ℝ // ∃ (x y z : ℝ), P = (x, y, z) ∧ 2 * x - 3 * y + 4 * z = 20}

def point_A : ℝ × ℝ × ℝ := (0, 1, -1)

theorem closest_point_on_plane_exists (P : point_on_plane) :
  ∃ (x y z : ℝ), (x, y, z) = (54 / 29, -80 / 29, 83 / 29) := sorry

end closest_point_on_plane_exists_l474_47417


namespace investment_inequality_l474_47489

-- Defining the initial investment
def initial_investment : ℝ := 200

-- Year 1 changes
def alpha_year1 := initial_investment * 1.30
def beta_year1 := initial_investment * 0.80
def gamma_year1 := initial_investment * 1.10
def delta_year1 := initial_investment * 0.90

-- Year 2 changes
def alpha_final := alpha_year1 * 0.85
def beta_final := beta_year1 * 1.30
def gamma_final := gamma_year1 * 0.95
def delta_final := delta_year1 * 1.20

-- Prove the final inequality
theorem investment_inequality : beta_final < gamma_final ∧ gamma_final < delta_final ∧ delta_final < alpha_final :=
by {
  sorry
}

end investment_inequality_l474_47489


namespace angle_x_value_l474_47495

theorem angle_x_value 
  (AB CD : Prop) -- AB and CD are straight lines
  (angle_AXB angle_AXZ angle_BXY angle_CYX : ℝ) -- Given angles in the problem
  (h1 : AB) (h2 : CD)
  (h3 : angle_AXB = 180)
  (h4 : angle_AXZ = 60)
  (h5 : angle_BXY = 50)
  (h6 : angle_CYX = 120) : 
  ∃ x : ℝ, x = 50 := by
sorry

end angle_x_value_l474_47495


namespace a_plus_b_minus_c_in_S_l474_47411

-- Define the sets P, Q, and S
def P := {x : ℤ | ∃ k : ℤ, x = 3 * k}
def Q := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def S := {x : ℤ | ∃ k : ℤ, x = 3 * k - 1}

-- Define the elements a, b, and c as members of sets P, Q, and S respectively
variables (a b c : ℤ)
variable (ha : a ∈ P) -- a ∈ P
variable (hb : b ∈ Q) -- b ∈ Q
variable (hc : c ∈ S) -- c ∈ S

-- Theorem statement proving the question
theorem a_plus_b_minus_c_in_S : a + b - c ∈ S := sorry

end a_plus_b_minus_c_in_S_l474_47411


namespace g_25_eq_zero_l474_47494

noncomputable def g : ℝ → ℝ := sorry

axiom g_def (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x^2 * g y - y^2 * g x = g (x^2 / y^2)

theorem g_25_eq_zero : g 25 = 0 := by
  sorry

end g_25_eq_zero_l474_47494


namespace factor_theorem_for_Q_l474_47491

variable (d : ℝ) -- d is a real number

def Q (x : ℝ) : ℝ := x^3 + 3 * x^2 + d * x + 20

theorem factor_theorem_for_Q :
  (x : ℝ) → (Q x = 0) → (x = 4) → d = -33 :=
by
  intro x Q4 hx
  sorry

end factor_theorem_for_Q_l474_47491


namespace Yuna_boarding_place_l474_47461

-- Conditions
def Eunji_place : ℕ := 10
def people_after_Eunji : ℕ := 11

-- Proof Problem: Yuna's boarding place calculation
theorem Yuna_boarding_place :
  Eunji_place + people_after_Eunji + 1 = 22 :=
by
  sorry

end Yuna_boarding_place_l474_47461


namespace labor_day_to_national_day_l474_47444

theorem labor_day_to_national_day :
  let labor_day := 1 -- Monday is represented as 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  (labor_day + total_days % 7) % 7 = 0 := -- Since 0 corresponds to Sunday modulo 7
by
  let labor_day := 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  have h1 : (labor_day + total_days % 7) % 7 = ((1 + (31 * 3 + 30 * 2) % 7) % 7) := by rfl
  sorry

end labor_day_to_national_day_l474_47444


namespace impossible_coins_l474_47458

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end impossible_coins_l474_47458


namespace volume_of_cone_l474_47416

theorem volume_of_cone
  (r h l : ℝ) -- declaring variables
  (base_area : ℝ) (lateral_surface_is_semicircle : ℝ) 
  (h_eq : h = Real.sqrt (l^2 - r^2))
  (base_area_eq : π * r^2 = π)
  (lateral_surface_eq : π * l = 2 * π * r) : 
  (∀ (V : ℝ), V = (1 / 3) * π * r^2 * h → V = (Real.sqrt 3) / 3 * π) :=
by
  sorry

end volume_of_cone_l474_47416


namespace largest_multiple_of_18_with_digits_9_or_0_l474_47499

theorem largest_multiple_of_18_with_digits_9_or_0 :
  ∃ (n : ℕ), (n = 9990) ∧ (n % 18 = 0) ∧ (∀ d ∈ (n.digits 10), d = 9 ∨ d = 0) ∧ (n / 18 = 555) :=
by
  sorry

end largest_multiple_of_18_with_digits_9_or_0_l474_47499


namespace perimeter_reduction_percentage_l474_47468

-- Given initial dimensions x and y
-- Initial Perimeter
def initial_perimeter (x y : ℝ) : ℝ := 2 * (x + y)

-- First reduction
def first_reduction_length (x : ℝ) : ℝ := 0.9 * x
def first_reduction_width (y : ℝ) : ℝ := 0.8 * y

-- New perimeter after first reduction
def new_perimeter_first (x y : ℝ) : ℝ := 2 * (first_reduction_length x + first_reduction_width y)

-- Condition: new perimeter is 88% of the initial perimeter
def perimeter_condition (x y : ℝ) : Prop := new_perimeter_first x y = 0.88 * initial_perimeter x y

-- Solve for x in terms of y
def solve_for_x (y : ℝ) : ℝ := 4 * y

-- Second reduction
def second_reduction_length (x : ℝ) : ℝ := 0.8 * x
def second_reduction_width (y : ℝ) : ℝ := 0.9 * y

-- New perimeter after second reduction
def new_perimeter_second (x y : ℝ) : ℝ := 2 * (second_reduction_length x + second_reduction_width y)

-- Proof statement
theorem perimeter_reduction_percentage (x y : ℝ) (h : perimeter_condition x y) : 
  new_perimeter_second x y = 0.82 * initial_perimeter x y :=
by
  sorry

end perimeter_reduction_percentage_l474_47468


namespace smallest_sum_l474_47450

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l474_47450


namespace inequality_not_less_than_l474_47409

theorem inequality_not_less_than (y : ℝ) : 2 * y + 8 ≥ -3 := 
sorry

end inequality_not_less_than_l474_47409


namespace number_of_real_roots_l474_47475

theorem number_of_real_roots (a : ℝ) :
  (|a| < (2 * Real.sqrt 3 / 9) → ∃ x y z : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ z^3 - z - a = 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  (|a| = (2 * Real.sqrt 3 / 9) → ∃ x y : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ x = y) ∧
  (|a| > (2 * Real.sqrt 3 / 9) → ∃ x : ℝ, x^3 - x - a = 0 ∧ ∀ y : ℝ, y^3 - y - a ≠ 0 ∨ y = x) :=
sorry

end number_of_real_roots_l474_47475


namespace initial_savings_amount_l474_47437

theorem initial_savings_amount (A : ℝ) (P : ℝ) (r1 r2 t1 t2 : ℝ) (hA : A = 2247.50) (hr1 : r1 = 0.08) (hr2 : r2 = 0.04) (ht1 : t1 = 0.25) (ht2 : t2 = 0.25) :
  P = 2181 :=
by
  sorry

end initial_savings_amount_l474_47437


namespace number_of_ways_to_choose_one_book_is_correct_l474_47428

-- Definitions of the given problem conditions
def number_of_chinese_books : Nat := 10
def number_of_english_books : Nat := 7
def number_of_math_books : Nat := 5

-- Theorem stating the proof problem
theorem number_of_ways_to_choose_one_book_is_correct : 
  number_of_chinese_books + number_of_english_books + number_of_math_books = 22 := by
  -- This proof is left as an exercise.
  sorry

end number_of_ways_to_choose_one_book_is_correct_l474_47428


namespace purchase_price_l474_47443

noncomputable def cost_price_after_discount (P : ℝ) : ℝ :=
  0.8 * P + 375

theorem purchase_price {P : ℝ} (h : 1.15 * P = 18400) : cost_price_after_discount P = 13175 := by
  sorry

end purchase_price_l474_47443


namespace find_number_l474_47449

theorem find_number (x : ℤ) (h : x = 5 * (x - 4)) : x = 5 :=
by {
  sorry
}

end find_number_l474_47449


namespace salary_increase_gt_90_percent_l474_47419

theorem salary_increase_gt_90_percent (S : ℝ) : 
  (S * (1.12^6) - S) / S > 0.90 :=
by
  -- Here we skip the proof with sorry
  sorry

end salary_increase_gt_90_percent_l474_47419


namespace probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l474_47436

noncomputable section

-- Problem 1: Probability of drawing a white ball on the third draw without replacement is 1/3.
theorem probability_third_white_no_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let totalWaysToDraw3 := Nat.choose totalBalls 3
  let favorableWays := Nat.choose (totalBalls - 1) 2 * Nat.choose white 1
  let probability := favorableWays / totalWaysToDraw3
  probability = 1 / 3 :=
by
  sorry

-- Problem 2: Probability of drawing red balls no more than 4 times in 6 draws with replacement is 441/729.
theorem probability_red_no_more_than_4_in_6_draws_with_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let p_red := red / totalBalls
  let p_X5 := Nat.choose 6 5 * p_red^5 * (1 - p_red)
  let p_X6 := Nat.choose 6 6 * p_red^6
  let probability := 1 - p_X5 - p_X6
  probability = 441 / 729 :=
by
  sorry

end probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l474_47436


namespace keith_remaining_cards_l474_47451

-- Definitions and conditions
def initial_cards := 0
def new_cards := 8
def total_cards_after_purchase := initial_cards + new_cards
def remaining_cards := total_cards_after_purchase / 2

-- Proof statement (in Lean, the following would be a theorem)
theorem keith_remaining_cards : remaining_cards = 4 := sorry

end keith_remaining_cards_l474_47451


namespace megan_earnings_l474_47406

-- Define the given conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- Define the total number of necklaces
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

-- Define the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings are 90 dollars
theorem megan_earnings : total_earnings = 90 := by
  sorry

end megan_earnings_l474_47406


namespace fractions_sum_correct_l474_47467

noncomputable def fractions_sum : ℝ := (3 / 20) + (5 / 200) + (7 / 2000) + 5

theorem fractions_sum_correct : fractions_sum = 5.1785 :=
by
  sorry

end fractions_sum_correct_l474_47467


namespace quadrant_of_complex_number_l474_47402

theorem quadrant_of_complex_number
  (h : ∀ x : ℝ, 0 < x → (a^2 + a + 2)/x < 1/x^2 + 1) :
  ∃ a : ℝ, -1 < a ∧ a < 0 ∧ i^27 = -i :=
sorry

end quadrant_of_complex_number_l474_47402


namespace Petya_receives_last_wrapper_l474_47484

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem Petya_receives_last_wrapper
  (h1 : discriminant a b c ≥ 0)
  (h2 : discriminant a c b ≥ 0)
  (h3 : discriminant b a c ≥ 0)
  (h4 : discriminant c a b < 0)
  (h5 : discriminant b c a < 0) :
  discriminant c b a ≥ 0 :=
sorry

end Petya_receives_last_wrapper_l474_47484


namespace height_of_cylinder_l474_47457

theorem height_of_cylinder (r_hemisphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) :
  r_hemisphere = 7 → r_cylinder = 3 → h_cylinder = 2 * Real.sqrt 10 :=
by
  intro r_hemisphere_eq r_cylinder_eq
  sorry

end height_of_cylinder_l474_47457


namespace find_s_at_1_l474_47486

variable (t s : ℝ → ℝ)
variable (x : ℝ)

-- Define conditions
def t_def : t x = 4 * x - 9 := by sorry

def s_def : s (t x) = x^2 + 4 * x - 5 := by sorry

-- Prove the question
theorem find_s_at_1 : s 1 = 11.25 := by
  -- Proof goes here
  sorry

end find_s_at_1_l474_47486


namespace seven_a_plus_seven_b_l474_47438

noncomputable def g (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem seven_a_plus_seven_b (a b : ℝ) (h₁ : ∀ x, g x = f_inv x - 2) (h₂ : ∀ x, f_inv (f x a b) = x) :
  7 * a + 7 * b = 5 :=
by
  sorry

end seven_a_plus_seven_b_l474_47438


namespace totalTilesUsed_l474_47476

-- Define the dining room dimensions
def diningRoomLength : ℕ := 18
def diningRoomWidth : ℕ := 15

-- Define the border width
def borderWidth : ℕ := 2

-- Define tile dimensions
def tile1x1 : ℕ := 1
def tile2x2 : ℕ := 2

-- Calculate the number of tiles used along the length and width for the border
def borderTileCountLength : ℕ := 2 * 2 * (diningRoomLength - 2 * borderWidth)
def borderTileCountWidth : ℕ := 2 * 2 * (diningRoomWidth - 2 * borderWidth)

-- Total number of one-foot by one-foot tiles for the border
def totalBorderTileCount : ℕ := borderTileCountLength + borderTileCountWidth

-- Calculate the inner area dimensions
def innerLength : ℕ := diningRoomLength - 2 * borderWidth
def innerWidth : ℕ := diningRoomWidth - 2 * borderWidth
def innerArea : ℕ := innerLength * innerWidth

-- Number of two-foot by two-foot tiles needed
def tile2x2Count : ℕ := (innerArea + tile2x2 * tile2x2 - 1) / (tile2x2 * tile2x2) -- Ensures rounding up without floating point arithmetic

-- Prove that the total number of tiles used is 139
theorem totalTilesUsed : totalBorderTileCount + tile2x2Count = 139 := by
  sorry

end totalTilesUsed_l474_47476


namespace problem_1_2_a_problem_1_2_b_l474_47481

theorem problem_1_2_a (x : ℝ) : x * (1 - x) ≤ 1 / 4 := sorry

theorem problem_1_2_b (x a : ℝ) : x * (a - x) ≤ a^2 / 4 := sorry

end problem_1_2_a_problem_1_2_b_l474_47481


namespace sufficient_not_necessary_condition_l474_47422

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (x - 1)^2 < 9 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 → ¬ (x - 1)^2 < 9) →
  a < -4 :=
by
  sorry

end sufficient_not_necessary_condition_l474_47422


namespace relationship_among_y_values_l474_47462

theorem relationship_among_y_values (c y1 y2 y3 : ℝ) :
  (-1)^2 - 2 * (-1) + c = y1 →
  (3)^2 - 2 * 3 + c = y2 →
  (5)^2 - 2 * 5 + c = y3 →
  y1 = y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  sorry

end relationship_among_y_values_l474_47462


namespace magnitude_of_resultant_vector_is_sqrt_5_l474_47407

-- We denote the vectors a and b
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (-2, y)

-- We encode the condition that vectors are parallel
def parallel_vectors (y : ℝ) : Prop := 1 * y = (-2) * (-2)

-- We calculate the resultant vector and its magnitude
def resultant_vector (y : ℝ) : ℝ × ℝ :=
  ((3 * 1 + 2 * -2), (3 * -2 + 2 * y))

def magnitude_square (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

-- The target statement
theorem magnitude_of_resultant_vector_is_sqrt_5 (y : ℝ) (hy : parallel_vectors y) :
  magnitude_square (resultant_vector y) = 5 := by
  sorry

end magnitude_of_resultant_vector_is_sqrt_5_l474_47407


namespace real_y_values_for_given_x_l474_47471

theorem real_y_values_for_given_x (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 4 = 0) ↔ (x ≤ -2 / 3 ∨ x ≥ 4) :=
by
  sorry

end real_y_values_for_given_x_l474_47471


namespace range_of_sum_l474_47466

variable {x y t : ℝ}

theorem range_of_sum :
  (1 = x^2 + 4*y^2 - 2*x*y) ∧ (x < 0) ∧ (y < 0) →
  -2 <= x + 2*y ∧ x + 2*y < 0 :=
by {
  sorry
}

end range_of_sum_l474_47466


namespace five_student_committee_l474_47452

theorem five_student_committee : ∀ (students : Finset ℕ) (alice bob : ℕ), 
  alice ∈ students → bob ∈ students → students.card = 8 → ∃ (committees : Finset (Finset ℕ)),
  (∀ committee ∈ committees, alice ∈ committee ∧ bob ∈ committee) ∧
  ∀ committee ∈ committees, committee.card = 5 ∧ committees.card = 20 :=
by
  sorry

end five_student_committee_l474_47452


namespace work_time_l474_47448

-- Definitions and conditions
variables (A B C D h : ℝ)
variable (h_def : ℝ := 1 / (1 / A + 1 / B + 1 / D))

-- Conditions
axiom cond1 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (A - 8)
axiom cond2 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (B - 2)
axiom cond3 : 1 / A + 1 / B + 1 / C + 1 / D = 3 / C
axiom cond4 : 1 / A + 1 / B + 1 / D = 2 / C

-- The statement to prove
theorem work_time : h_def = 16 / 11 := by
  sorry

end work_time_l474_47448


namespace range_of_a_range_of_m_l474_47445

-- Definition of proposition p: Equation has real roots
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a * x + a + 3 = 0

-- Definition of proposition q: m - 1 <= a <= m + 1
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 1

-- Part (I): Range of a when ¬p is true
theorem range_of_a (a : ℝ) (hp : ¬ p a) : -2 < a ∧ a < 6 :=
sorry

-- Part (II): Range of m when p is a necessary but not sufficient condition for q
theorem range_of_m (m : ℝ) (hnp : ∀ a, q m a → p a) (hns : ∃ a, q m a ∧ ¬p a) : m ≤ -3 ∨ m ≥ 7 :=
sorry

end range_of_a_range_of_m_l474_47445


namespace lindas_nickels_l474_47492

theorem lindas_nickels
  (N : ℕ)
  (initial_dimes : ℕ := 2)
  (initial_quarters : ℕ := 6)
  (initial_nickels : ℕ := N)
  (additional_dimes : ℕ := 2)
  (additional_quarters : ℕ := 10)
  (additional_nickels : ℕ := 2 * N)
  (total_coins : ℕ := 35)
  (h : initial_dimes + initial_quarters + initial_nickels + additional_dimes + additional_quarters + additional_nickels = total_coins) :
  N = 5 := by
  sorry

end lindas_nickels_l474_47492


namespace find_x_from_arithmetic_mean_l474_47431

theorem find_x_from_arithmetic_mean (x : ℝ) 
  (h : (x + 10 + 18 + 3 * x + 16 + (x + 5) + (3 * x + 6)) / 6 = 25) : 
  x = 95 / 8 := by
  sorry

end find_x_from_arithmetic_mean_l474_47431


namespace three_digit_multiples_of_3_and_11_l474_47488

theorem three_digit_multiples_of_3_and_11 : 
  ∃ n, n = 27 ∧ ∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 33 = 0 ↔ ∃ k, x = 33 * k ∧ 4 ≤ k ∧ k ≤ 30 :=
by
  sorry

end three_digit_multiples_of_3_and_11_l474_47488


namespace will_initial_money_l474_47478

theorem will_initial_money (spent_game : ℕ) (number_of_toys : ℕ) (cost_per_toy : ℕ) (initial_money : ℕ) :
  spent_game = 27 →
  number_of_toys = 5 →
  cost_per_toy = 6 →
  initial_money = spent_game + number_of_toys * cost_per_toy →
  initial_money = 57 :=
by
  intros
  sorry

end will_initial_money_l474_47478
