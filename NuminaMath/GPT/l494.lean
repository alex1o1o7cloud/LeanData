import Mathlib

namespace exists_nat_sum_of_squares_two_ways_l494_49462

theorem exists_nat_sum_of_squares_two_ways :
  ∃ n : ℕ, n < 100 ∧ ∃ a b c d : ℕ, a ≠ b ∧ c ≠ d ∧ n = a^2 + b^2 ∧ n = c^2 + d^2 :=
by {
  sorry
}

end exists_nat_sum_of_squares_two_ways_l494_49462


namespace product_has_correct_sign_and_units_digit_l494_49446

noncomputable def product_negative_integers_divisible_by_3_less_than_198 : ℤ :=
  sorry

theorem product_has_correct_sign_and_units_digit :
  product_negative_integers_divisible_by_3_less_than_198 < 0 ∧
  product_negative_integers_divisible_by_3_less_than_198 % 10 = 6 :=
by
  sorry

end product_has_correct_sign_and_units_digit_l494_49446


namespace coats_collected_in_total_l494_49407

def high_school_coats : Nat := 6922
def elementary_school_coats : Nat := 2515
def total_coats : Nat := 9437

theorem coats_collected_in_total : 
  high_school_coats + elementary_school_coats = total_coats := 
  by
  sorry

end coats_collected_in_total_l494_49407


namespace jerry_cut_pine_trees_l494_49421

theorem jerry_cut_pine_trees (P : ℕ)
  (h1 : 3 * 60 = 180)
  (h2 : 4 * 100 = 400)
  (h3 : 80 * P + 180 + 400 = 1220) :
  P = 8 :=
by {
  sorry -- Proof not required as per the instructions
}

end jerry_cut_pine_trees_l494_49421


namespace find_number_l494_49419

theorem find_number (N : ℕ) (k : ℕ) (Q : ℕ)
  (h1 : N = 9 * k)
  (h2 : Q = 25 * 9 + 7)
  (h3 : N / 9 = Q) :
  N = 2088 :=
by
  sorry

end find_number_l494_49419


namespace fraction_transformation_l494_49481

theorem fraction_transformation (a b x: ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2 * b) / (a - 2 * b) = (x + 2) / (x - 2) :=
by sorry

end fraction_transformation_l494_49481


namespace number_of_blue_crayons_given_to_Becky_l494_49427

-- Definitions based on the conditions
def initial_green_crayons : ℕ := 5
def initial_blue_crayons : ℕ := 8
def given_out_green_crayons : ℕ := 3
def total_crayons_left : ℕ := 9

-- Statement of the problem and expected proof
theorem number_of_blue_crayons_given_to_Becky (initial_green_crayons initial_blue_crayons given_out_green_crayons total_crayons_left : ℕ) : 
  initial_green_crayons = 5 →
  initial_blue_crayons = 8 →
  given_out_green_crayons = 3 →
  total_crayons_left = 9 →
  ∃ num_blue_crayons_given_to_Becky, num_blue_crayons_given_to_Becky = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_blue_crayons_given_to_Becky_l494_49427


namespace sqrt_cosine_identity_l494_49455

theorem sqrt_cosine_identity :
  Real.sqrt ((3 - Real.cos (Real.pi / 8)^2) * (3 - Real.cos (3 * Real.pi / 8)^2)) = (3 * Real.sqrt 5) / 4 :=
by
  sorry

end sqrt_cosine_identity_l494_49455


namespace red_peaches_count_l494_49468

/-- Math problem statement:
There are some red peaches and 16 green peaches in the basket.
There is 1 more red peach than green peaches in the basket.
Prove that the number of red peaches in the basket is 17.
--/

-- Let G be the number of green peaches and R be the number of red peaches.
def G : ℕ := 16
def R : ℕ := G + 1

theorem red_peaches_count : R = 17 := by
  sorry

end red_peaches_count_l494_49468


namespace minimize_distance_l494_49445

-- Definitions of points and distances
structure Point where
  x : ℝ
  y : ℝ

def distanceSquared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition points A, B, and C
def A := Point.mk 7 3
def B := Point.mk 3 0

-- Mathematical problem: Find the value of k that minimizes the sum of distances squared
theorem minimize_distance : ∃ k : ℝ, ∀ k', 
  (distanceSquared A (Point.mk 0 k) + distanceSquared B (Point.mk 0 k) ≤ 
   distanceSquared A (Point.mk 0 k') + distanceSquared B (Point.mk 0 k')) → 
  k = 3 / 2 :=
by
  sorry

end minimize_distance_l494_49445


namespace points_satisfy_l494_49406

theorem points_satisfy (x y : ℝ) : 
  (y^2 - y = x^2 - x) ↔ (y = x ∨ y = 1 - x) :=
by sorry

end points_satisfy_l494_49406


namespace number_of_packages_needed_l494_49443

-- Define the problem constants and constraints
def students_per_class := 30
def number_of_classes := 4
def buns_per_student := 2
def buns_per_package := 8

-- Calculate the total number of students
def total_students := number_of_classes * students_per_class

-- Calculate the total number of buns needed
def total_buns := total_students * buns_per_student

-- Calculate the required number of packages
def required_packages := total_buns / buns_per_package

-- Prove that the required number of packages is 30
theorem number_of_packages_needed : required_packages = 30 := by
  -- The proof would be here, but for now we assume it is correct
  sorry

end number_of_packages_needed_l494_49443


namespace similar_triangle_leg_length_l494_49411

theorem similar_triangle_leg_length (a b c : ℝ) (h0 : a = 12) (h1 : b = 9) (h2 : c = 7.5) :
  ∃ y : ℝ, ((12 / 7.5) = (9 / y) → y = 5.625) :=
by
  use 5.625
  intro h
  linarith

end similar_triangle_leg_length_l494_49411


namespace fraction_equality_l494_49403

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 3 / 5 :=
by
  sorry

end fraction_equality_l494_49403


namespace triangle_area_l494_49466

theorem triangle_area (A B C : ℝ) (AB AC : ℝ) (A_angle : ℝ) (h1 : A_angle = π / 6)
  (h2 : AB * AC * Real.cos A_angle = Real.tan A_angle) :
  1 / 2 * AB * AC * Real.sin A_angle = 1 / 6 :=
by
  sorry

end triangle_area_l494_49466


namespace seafood_regular_price_l494_49486

theorem seafood_regular_price (y : ℝ) (h : y / 4 = 4) : 2 * y = 32 := by
  sorry

end seafood_regular_price_l494_49486


namespace total_students_in_lunchroom_l494_49485

theorem total_students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end total_students_in_lunchroom_l494_49485


namespace identify_A_B_l494_49417

variable {Person : Type}
variable (isTruthful isLiar : Person → Prop)
variable (isBoy isGirl : Person → Prop)

variables (A B : Person)

-- Conditions
axiom truthful_or_liar : ∀ x : Person, isTruthful x ∨ isLiar x
axiom boy_or_girl : ∀ x : Person, isBoy x ∨ isGirl x
axiom not_both_truthful_and_liar : ∀ x : Person, ¬(isTruthful x ∧ isLiar x)
axiom not_both_boy_and_girl : ∀ x : Person, ¬(isBoy x ∧ isGirl x)

-- Statements made by A and B
axiom A_statement : isTruthful A → isLiar B 
axiom B_statement : isBoy B → isGirl A 

-- Goal: prove the identities of A and B
theorem identify_A_B : isTruthful A ∧ isBoy A ∧ isLiar B ∧ isBoy B :=
by {
  sorry
}

end identify_A_B_l494_49417


namespace edge_length_of_cube_l494_49425

theorem edge_length_of_cube (V : ℝ) (e : ℝ) (h1 : V = 2744) (h2 : V = e^3) : e = 14 := 
by 
  sorry

end edge_length_of_cube_l494_49425


namespace suitable_bases_for_346_l494_49447

theorem suitable_bases_for_346 (b : ℕ) (hb : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0) : b = 6 ∨ b = 7 :=
sorry

end suitable_bases_for_346_l494_49447


namespace find_fourth_number_l494_49418

theorem find_fourth_number : 
  ∃ (x : ℝ), (217 + 2.017 + 0.217 + x = 221.2357) ∧ (x = 2.0017) :=
by
  sorry

end find_fourth_number_l494_49418


namespace max_sum_of_arithmetic_sequence_l494_49448

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → 4 * a (n + 1) = 4 * a n - 7) →
  a 1 = 25 →
  (∀ n : ℕ, S n = (n * (50 - (7/4 : ℚ) * (n - 1))) / 2) →
  ∃ n : ℕ, n = 15 ∧ S n = 765 / 4 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l494_49448


namespace total_musicians_is_98_l494_49490

-- Define the number of males and females in the orchestra
def males_in_orchestra : ℕ := 11
def females_in_orchestra : ℕ := 12

-- Define the total number of musicians in the orchestra
def total_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

-- Define the number of musicians in the band as twice the number in the orchestra
def total_in_band : ℕ := 2 * total_in_orchestra

-- Define the number of males and females in the choir
def males_in_choir : ℕ := 12
def females_in_choir : ℕ := 17

-- Define the total number of musicians in the choir
def total_in_choir : ℕ := males_in_choir + females_in_choir

-- Prove that the total number of musicians in the orchestra, band, and choir is 98
theorem total_musicians_is_98 : total_in_orchestra + total_in_band + total_in_choir = 98 :=
by {
  -- Adding placeholders for the proof steps
  sorry
}

end total_musicians_is_98_l494_49490


namespace largest_divisor_prime_cube_diff_l494_49489

theorem largest_divisor_prime_cube_diff (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge5 : p ≥ 5) : 
  ∃ k, k = 12 ∧ ∀ n, n ∣ (p^3 - p) ↔ n ∣ 12 :=
by
  sorry

end largest_divisor_prime_cube_diff_l494_49489


namespace problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l494_49460

-- Problem I2.1
theorem problem_I2_1 (a : ℕ) (h₁ : a > 0) (h₂ : a^2 - 1 = 123 * 125) : a = 124 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.2
theorem problem_I2_2 (b : ℕ) (h₁ : b = (2^3 - 16*2^2 - 9*2 + 124)) : b = 50 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.3
theorem problem_I2_3 (n : ℕ) (h₁ : (n * (n - 3)) / 2 = 54) : n = 12 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2_4
theorem problem_I2_4 (d : ℤ) (n : ℤ) (h₁ : n = 12) 
  (h₂ : (d - 1) * 2 = (1 - n) * 2) : d = -10 :=
by {
  -- This proof needs to be filled in
  sorry
}

end problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l494_49460


namespace wheel_radius_increase_l494_49474

theorem wheel_radius_increase 
  (d₁ d₂ : ℝ) -- distances according to the odometer (600 and 580 miles)
  (r₀ : ℝ)   -- original radius (17 inches)
  (C₁: d₁ = 600)
  (C₂: d₂ = 580)
  (C₃: r₀ = 17) :
  ∃ Δr : ℝ, Δr = 0.57 :=
by
  sorry

end wheel_radius_increase_l494_49474


namespace fraction_of_p_l494_49423

theorem fraction_of_p (p q r f : ℝ) (hp : p = 49) (hqr : p = (2 * f * 49) + 35) : f = 1/7 :=
sorry

end fraction_of_p_l494_49423


namespace AC_amount_l494_49451

variable (A B C : ℝ)

theorem AC_amount
  (h1 : A + B + C = 400)
  (h2 : B + C = 150)
  (h3 : C = 50) :
  A + C = 300 := by
  sorry

end AC_amount_l494_49451


namespace valid_operation_l494_49408

theorem valid_operation :
  ∀ x : ℝ, x^2 + x^3 ≠ x^5 ∧
  ∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2 ∧
  ∀ m : ℝ, (|m| = m ↔ m ≥ 0) :=
by
  sorry

end valid_operation_l494_49408


namespace men_count_in_first_group_is_20_l494_49404

noncomputable def men_needed_to_build_fountain (work1 : ℝ) (days1 : ℕ) (length1 : ℝ) (workers2 : ℕ) (days2 : ℕ) (length2 : ℝ) (work_per_man_per_day2 : ℝ) : ℕ :=
  let work_per_day2 := length2 / days2
  let work_per_man_per_day2 := work_per_day2 / workers2
  let total_work1 := length1 / days1
  Nat.floor (total_work1 / work_per_man_per_day2)

theorem men_count_in_first_group_is_20 :
  men_needed_to_build_fountain 56 6 56 35 3 49 (49 / (35 * 3)) = 20 :=
by
  sorry

end men_count_in_first_group_is_20_l494_49404


namespace pythagorean_triple_square_l494_49482

theorem pythagorean_triple_square (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pythagorean : a^2 + b^2 = c^2) : ∃ k : ℤ, k^2 = (c - a) * (c - b) / 2 := 
sorry

end pythagorean_triple_square_l494_49482


namespace isosceles_triangle_base_angle_l494_49405

theorem isosceles_triangle_base_angle (a b h θ : ℝ)
  (h1 : a^2 = 4 * b^2 * h)
  (h_b : b = 2 * a * Real.cos θ)
  (h_h : h = a * Real.sin θ) :
  θ = Real.arccos (1/4) :=
by
  sorry

end isosceles_triangle_base_angle_l494_49405


namespace sum_of_number_and_reverse_l494_49457

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
  (h5 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 99 := 
sorry

end sum_of_number_and_reverse_l494_49457


namespace find_first_number_l494_49479

theorem find_first_number (x : ℕ) (h : x + 15 = 20) : x = 5 :=
by
  sorry

end find_first_number_l494_49479


namespace study_group_number_l494_49487

theorem study_group_number (b : ℤ) :
  (¬ (b % 2 = 0) ∧ (b + b^3 < 8000) ∧ ¬ (∃ r : ℚ, r^2 = 13) ∧ (b % 7 = 0)
  ∧ (∃ r : ℚ, r = b) ∧ ¬ (b % 14 = 0)) →
  b = 7 :=
by
  sorry

end study_group_number_l494_49487


namespace ratio_perimeters_of_squares_l494_49499

theorem ratio_perimeters_of_squares 
  (s₁ s₂ : ℝ)
  (h : (s₁ ^ 2) / (s₂ ^ 2) = 25 / 36) :
  (4 * s₁) / (4 * s₂) = 5 / 6 :=
by
  sorry

end ratio_perimeters_of_squares_l494_49499


namespace simplify_sqrt_l494_49484

theorem simplify_sqrt (a : ℝ) (h : a < 2) : Real.sqrt ((a - 2)^2) = 2 - a :=
by
  sorry

end simplify_sqrt_l494_49484


namespace ratio_of_siblings_l494_49430

/-- Let's define the sibling relationships and prove the ratio of Janet's to Masud's siblings is 3 to 1. -/
theorem ratio_of_siblings (masud_siblings : ℕ) (carlos_siblings janet_siblings : ℕ)
  (h1 : masud_siblings = 60)
  (h2 : carlos_siblings = 3 * masud_siblings / 4)
  (h3 : janet_siblings = carlos_siblings + 135) 
  (h4 : janet_siblings < some_mul * masud_siblings) : 
  janet_siblings / masud_siblings = 3 :=
by
  sorry

end ratio_of_siblings_l494_49430


namespace sum_of_units_digits_eq_0_l494_49414

-- Units digit function definition
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement in Lean 
theorem sum_of_units_digits_eq_0 :
  units_digit (units_digit (17 * 34) + units_digit (19 * 28)) = 0 :=
by
  sorry

end sum_of_units_digits_eq_0_l494_49414


namespace largest_divisor_of_n_l494_49495

theorem largest_divisor_of_n 
  (n : ℕ) (h_pos : n > 0) (h_div : 72 ∣ n^2) : 
  ∃ v : ℕ, v = 12 ∧ v ∣ n :=
by
  use 12
  sorry

end largest_divisor_of_n_l494_49495


namespace floor_area_l494_49434

theorem floor_area (length_feet : ℝ) (width_feet : ℝ) (feet_to_meters : ℝ) 
  (h_length : length_feet = 15) (h_width : width_feet = 10) (h_conversion : feet_to_meters = 0.3048) :
  let length_meters := length_feet * feet_to_meters
  let width_meters := width_feet * feet_to_meters
  let area_meters := length_meters * width_meters
  area_meters = 13.93 := 
by
  sorry

end floor_area_l494_49434


namespace addition_of_two_negatives_l494_49458

theorem addition_of_two_negatives (a b : ℤ) (ha : a < 0) (hb : b < 0) : a + b < a ∧ a + b < b :=
by
  sorry

end addition_of_two_negatives_l494_49458


namespace find_t_l494_49470

variable (s t : ℚ) -- Using the rational numbers since the correct answer involves a fraction

theorem find_t (h1 : 8 * s + 7 * t = 145) (h2 : s = t + 3) : t = 121 / 15 :=
by 
  sorry

end find_t_l494_49470


namespace reaction_requires_two_moles_of_HNO3_l494_49437

def nitric_acid_reaction (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) 
  (reaction : HNO3 + NaHCO3 = NaNO3 + CO2 + H2O)
  (n_NaHCO3 : ℕ) : ℕ :=
  if n_NaHCO3 = 2 then 2 else sorry

theorem reaction_requires_two_moles_of_HNO3
  (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) 
  (reaction : HNO3 + NaHCO3 = NaNO3 + CO2 + H2O)
  (n_NaHCO3 : ℕ) :
  n_NaHCO3 = 2 → nitric_acid_reaction HNO3 NaHCO3 NaNO3 CO2 H2O reaction n_NaHCO3 = 2 :=
by sorry

end reaction_requires_two_moles_of_HNO3_l494_49437


namespace annual_interest_rate_l494_49469

-- Define the conditions as given in the problem
def principal : ℝ := 5000
def maturity_amount : ℝ := 5080
def interest_tax_rate : ℝ := 0.2

-- Define the annual interest rate x
variable (x : ℝ)

-- Statement to be proved: the annual interest rate x is 0.02
theorem annual_interest_rate :
  principal + principal * x - interest_tax_rate * (principal * x) = maturity_amount → x = 0.02 :=
by
  sorry

end annual_interest_rate_l494_49469


namespace max_value_of_z_l494_49424

theorem max_value_of_z 
    (x y : ℝ) 
    (h1 : |2 * x + y + 1| ≤ |x + 2 * y + 2|)
    (h2 : -1 ≤ y ∧ y ≤ 1) : 
    2 * x + y ≤ 5 := 
sorry

end max_value_of_z_l494_49424


namespace greatest_x_l494_49456

-- Define x as a positive multiple of 4.
def is_positive_multiple_of_four (x : ℕ) : Prop :=
  x > 0 ∧ ∃ k : ℕ, x = 4 * k

-- Statement of the equivalent proof problem
theorem greatest_x (x : ℕ) (h1: is_positive_multiple_of_four x) (h2: x^3 < 4096) : x ≤ 12 :=
by {
  sorry
}

end greatest_x_l494_49456


namespace gcd_36_n_eq_12_l494_49428

theorem gcd_36_n_eq_12 (n : ℕ) (h1 : 80 ≤ n) (h2 : n ≤ 100) (h3 : Int.gcd 36 n = 12) : n = 84 ∨ n = 96 :=
by
  sorry

end gcd_36_n_eq_12_l494_49428


namespace tax_rate_for_remaining_l494_49422

variable (total_earnings deductions first_tax_rate total_tax taxed_amount remaining_taxable_income rem_tax_rate : ℝ)

def taxable_income (total_earnings deductions : ℝ) := total_earnings - deductions

def tax_on_first_portion (portion tax_rate : ℝ) := portion * tax_rate

def remaining_taxable (total_taxable first_portion : ℝ) := total_taxable - first_portion

def total_tax_payable (tax_first tax_remaining : ℝ) := tax_first + tax_remaining

theorem tax_rate_for_remaining :
  total_earnings = 100000 ∧ 
  deductions = 30000 ∧ 
  first_tax_rate = 0.10 ∧
  total_tax = 12000 ∧
  tax_on_first_portion 20000 first_tax_rate = 2000 ∧
  taxed_amount = 2000 ∧
  remaining_taxable_income = taxable_income total_earnings deductions - 20000 ∧
  total_tax_payable taxed_amount (remaining_taxable_income * rem_tax_rate) = total_tax →
  rem_tax_rate = 0.20 := 
sorry

end tax_rate_for_remaining_l494_49422


namespace checkerboard_inequivalent_color_schemes_l494_49465

/-- 
  We consider a 7x7 checkerboard where two squares are painted yellow, and the remaining 
  are painted green. Two color schemes are equivalent if one can be obtained from 
  the other by rotations of 0°, 90°, 180°, or 270°. We aim to prove that the 
  number of inequivalent color schemes is 312. 
-/
theorem checkerboard_inequivalent_color_schemes : 
  let n := 7
  let total_squares := n * n
  let total_pairs := total_squares.choose 2
  let symmetric_pairs := 24
  let nonsymmetric_pairs := total_pairs - symmetric_pairs
  let unique_symmetric_pairs := symmetric_pairs 
  let unique_nonsymmetric_pairs := nonsymmetric_pairs / 4
  unique_symmetric_pairs + unique_nonsymmetric_pairs = 312 :=
by sorry

end checkerboard_inequivalent_color_schemes_l494_49465


namespace total_songs_l494_49415

-- Define the number of albums Faye bought and the number of songs per album
def country_albums : ℕ := 2
def pop_albums : ℕ := 3
def songs_per_album : ℕ := 6

-- Define the total number of albums Faye bought
def total_albums : ℕ := country_albums + pop_albums

-- Prove that the total number of songs Faye bought is 30
theorem total_songs : total_albums * songs_per_album = 30 := by
  sorry

end total_songs_l494_49415


namespace area_of_triangle_PQR_l494_49442

theorem area_of_triangle_PQR 
  (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 8)
  (P_is_center : ∃ P : ℝ, True) -- Simplified assumption that P exists
  (bases_on_same_line : True) -- Assumed true, as touching condition implies it
  : ∃ area : ℝ, area = 20 := 
by
  sorry

end area_of_triangle_PQR_l494_49442


namespace problem_l494_49441

-- Definitions of the function g and its values at specific points
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

-- Conditions given in the problem
theorem problem (d e f : ℝ)
  (h0 : g d e f 0 = 8)
  (h1 : g d e f 1 = 5) :
  d + e + 2 * f = 13 :=
by
  sorry

end problem_l494_49441


namespace incorrect_statement_D_l494_49464

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + x
else -(x^2 + x)

theorem incorrect_statement_D : ¬(∀ x : ℝ, x ≤ 0 → f x = x^2 + x) :=
by
  sorry

end incorrect_statement_D_l494_49464


namespace product_of_roots_l494_49498

theorem product_of_roots (r1 r2 r3 : ℝ) : 
  (∀ x : ℝ, 2 * x^3 - 24 * x^2 + 96 * x + 56 = 0 → x = r1 ∨ x = r2 ∨ x = r3) →
  r1 * r2 * r3 = -28 :=
by
  sorry

end product_of_roots_l494_49498


namespace min_visible_sum_of_prism_faces_l494_49420

theorem min_visible_sum_of_prism_faces :
  let corners := 8
  let edges := 8
  let face_centers := 12
  let min_corner_sum := 6 -- Each corner dice can show 1, 2, and 3
  let min_edge_sum := 3    -- Each edge dice can show 1 and 2
  let min_face_center_sum := 1 -- Each face center dice can show 1
  let total_sum := corners * min_corner_sum + edges * min_edge_sum + face_centers * min_face_center_sum
  total_sum = 84 := 
by
  -- The proof is omitted
  sorry

end min_visible_sum_of_prism_faces_l494_49420


namespace model_tower_height_l494_49493

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_real : real_height = 60) (v_real : real_volume = 200000) (v_model : model_volume = 0.2) :
  real_height / (real_volume / model_volume)^(1/3) = 0.6 :=
by
  rw [h_real, v_real, v_model]
  norm_num
  sorry

end model_tower_height_l494_49493


namespace x_lt_1_nec_not_suff_l494_49433

theorem x_lt_1_nec_not_suff (x : ℝ) : (x < 1 → x^2 < 1) ∧ (¬(x < 1) → x^2 < 1) := 
by {
  sorry
}

end x_lt_1_nec_not_suff_l494_49433


namespace shot_put_distance_l494_49440

theorem shot_put_distance :
  (∃ x : ℝ, (y = - 1 / 12 * x^2 + 2 / 3 * x + 5 / 3) ∧ y = 0) ↔ x = 10 := 
by
  sorry

end shot_put_distance_l494_49440


namespace min_value_nS_n_l494_49429

theorem min_value_nS_n (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) 
  (h2 : m ≥ 2)
  (h3 : S (m - 1) = -2)
  (h4 : S m = 0)
  (h5 : S (m + 1) = 3) :
  ∃ n : ℕ, n * S n = -9 :=
sorry

end min_value_nS_n_l494_49429


namespace count_solutions_congruence_l494_49496

theorem count_solutions_congruence : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, x + 20 ≡ 75 [MOD 45] ∧ x < 150 :=
sorry

end count_solutions_congruence_l494_49496


namespace second_fish_length_l494_49449

-- Defining the conditions
def first_fish_length : ℝ := 0.3
def length_difference : ℝ := 0.1

-- Proof statement
theorem second_fish_length : ∀ (second_fish : ℝ), first_fish_length = second_fish + length_difference → second_fish = 0.2 :=
by 
  intro second_fish
  intro h
  sorry

end second_fish_length_l494_49449


namespace constant_term_correct_l494_49409

variable (x : ℝ)

noncomputable def constant_term_expansion : ℝ :=
  let term := λ (r : ℕ) => (Nat.choose 9 r) * (-2)^r * x^((9 - 9 * r) / 2)
  term 1

theorem constant_term_correct : 
  constant_term_expansion x = -18 :=
sorry

end constant_term_correct_l494_49409


namespace linear_dependence_condition_l494_49467

theorem linear_dependence_condition (k : ℝ) :
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 1 + b * 4 = 0) ∧ (a * 2 + b * k = 0) ∧ (a * 1 + b * 2 = 0)) ↔ k = 8 := 
by sorry

end linear_dependence_condition_l494_49467


namespace fraction_of_juan_chocolates_given_to_tito_l494_49497

variable (n : ℕ)
variable (Juan Angela Tito : ℕ)
variable (f : ℝ)

-- Conditions
def chocolates_Angela_Tito : Angela = 3 * Tito := 
by sorry

def chocolates_Juan_Angela : Juan = 4 * Angela := 
by sorry

def equal_distribution : (Juan + Angela + Tito) = 16 * n := 
by sorry

-- Theorem to prove
theorem fraction_of_juan_chocolates_given_to_tito (n : ℕ) 
  (H1 : Angela = 3 * Tito)
  (H2 : Juan = 4 * Angela)
  (H3 : Juan + Angela + Tito = 16 * n) :
  f = 13 / 36 :=
by sorry

end fraction_of_juan_chocolates_given_to_tito_l494_49497


namespace winner_more_votes_l494_49436

variable (totalStudents : ℕ) (votingPercentage : ℤ) (winnerPercentage : ℤ) (loserPercentage : ℤ)

theorem winner_more_votes
    (h1 : totalStudents = 2000)
    (h2 : votingPercentage = 25)
    (h3 : winnerPercentage = 55)
    (h4 : loserPercentage = 100 - winnerPercentage)
    (h5 : votingStudents = votingPercentage * totalStudents / 100)
    (h6 : winnerVotes = winnerPercentage * votingStudents / 100)
    (h7 : loserVotes = loserPercentage * votingStudents / 100)
    : winnerVotes - loserVotes = 50 := by
  sorry

end winner_more_votes_l494_49436


namespace max_arith_seq_20_terms_l494_49492

noncomputable def max_arithmetic_sequences :
  Nat :=
  180

theorem max_arith_seq_20_terms (a : Nat → Nat) :
  (∀ (k : Nat), k ≥ 1 ∧ k ≤ 20 → ∃ d : Nat, a (k + 1) = a k + d) →
  (P : Nat) = max_arithmetic_sequences :=
  by
  -- here's where the proof would go
  sorry

end max_arith_seq_20_terms_l494_49492


namespace triangle_area_is_correct_l494_49491

-- Defining the points
structure Point where
  x : ℝ
  y : ℝ

-- Defining vertices A, B, C
def A : Point := { x := 2, y := -3 }
def B : Point := { x := 0, y := 4 }
def C : Point := { x := 3, y := -1 }

-- Vector from C to A
def v : Point := { x := A.x - C.x, y := A.y - C.y }

-- Vector from C to B
def w : Point := { x := B.x - C.x, y := B.y - C.y }

-- Cross product of vectors v and w in 2D
noncomputable def cross_product (v w : Point) : ℝ :=
  v.x * w.y - v.y * w.x

-- Absolute value of the cross product
noncomputable def abs_cross_product (v w : Point) : ℝ :=
  |cross_product v w|

-- Area of the triangle
noncomputable def area_of_triangle (v w : Point) : ℝ :=
  (1 / 2) * abs_cross_product v w

-- Prove the area of the triangle is 5.5
theorem triangle_area_is_correct : area_of_triangle v w = 5.5 :=
  sorry

end triangle_area_is_correct_l494_49491


namespace portia_high_school_students_l494_49432

variables (P L M : ℕ)
axiom h1 : P = 4 * L
axiom h2 : P = 2 * M
axiom h3 : P + L + M = 4800

theorem portia_high_school_students : P = 2740 :=
by sorry

end portia_high_school_students_l494_49432


namespace find_roots_l494_49401

theorem find_roots : 
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ (x = -2 ∨ x = 2 ∨ x = 3) := by
  sorry

end find_roots_l494_49401


namespace inequality_holds_l494_49431

theorem inequality_holds (c : ℝ) : (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x + c > 0) → c > 5 := by sorry

end inequality_holds_l494_49431


namespace abs_iff_sq_gt_l494_49477

theorem abs_iff_sq_gt (x y : ℝ) : (|x| > |y|) ↔ (x^2 > y^2) :=
by sorry

end abs_iff_sq_gt_l494_49477


namespace degree_at_least_three_l494_49444

noncomputable def p : Polynomial ℤ := sorry
noncomputable def q : Polynomial ℤ := sorry

theorem degree_at_least_three (h1 : p.degree ≥ 1)
                              (h2 : q.degree ≥ 1)
                              (h3 : (∃ xs : Fin 33 → ℤ, ∀ i, p.eval (xs i) * q.eval (xs i) - 2015 = 0)) :
  p.degree ≥ 3 ∧ q.degree ≥ 3 := 
sorry

end degree_at_least_three_l494_49444


namespace discount_rate_on_pony_jeans_l494_49453

theorem discount_rate_on_pony_jeans (F P : ℝ) 
  (h1 : F + P = 25)
  (h2 : 45 * F + 36 * P = 900) :
  P = 25 :=
by
  sorry

end discount_rate_on_pony_jeans_l494_49453


namespace inscribed_sphere_radius_eq_l494_49413

-- Define the parameters for the right cone
structure RightCone where
  base_radius : ℝ
  height : ℝ

-- Given the right cone conditions
def givenCone : RightCone := { base_radius := 15, height := 40 }

-- Define the properties for inscribed sphere
def inscribedSphereRadius (c : RightCone) : ℝ := sorry

-- The theorem statement for the radius of the inscribed sphere
theorem inscribed_sphere_radius_eq (c : RightCone) : ∃ (b d : ℝ), 
  inscribedSphereRadius c = b * Real.sqrt d - b ∧ (b + d = 14) :=
by
  use 5, 9
  sorry

end inscribed_sphere_radius_eq_l494_49413


namespace fixed_monthly_fee_l494_49471

/-
  We want to prove that given two conditions:
  1. x + y = 12.48
  2. x + 2y = 17.54
  The fixed monthly fee (x) is 7.42.
-/

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + y = 12.48) 
  (h2 : x + 2 * y = 17.54) : 
  x = 7.42 := 
sorry

end fixed_monthly_fee_l494_49471


namespace white_bread_served_l494_49476

theorem white_bread_served (total_bread : ℝ) (wheat_bread : ℝ) (white_bread : ℝ) 
  (h1 : total_bread = 0.9) (h2 : wheat_bread = 0.5) : white_bread = 0.4 :=
by
  sorry

end white_bread_served_l494_49476


namespace determine_g_2023_l494_49435

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_pos (x : ℕ) (hx : x > 0) : g x > 0

axiom g_property (x y : ℕ) (h1 : x > 2 * y) (h2 : 0 < y) : 
  g (x - y) = Real.sqrt (g (x / y) + 3)

theorem determine_g_2023 : g 2023 = (1 + Real.sqrt 13) / 2 :=
by
  sorry

end determine_g_2023_l494_49435


namespace triangle_side_length_l494_49472

/-
  Given a triangle ABC with sides |AB| = c, |AC| = b, and centroid G, incenter I,
  if GI is perpendicular to BC, then we need to prove that |BC| = (b+c)/2.
-/
variable {A B C G I : Type}
variable {AB AC BC : ℝ} -- Lengths of the sides
variable {b c : ℝ} -- Given lengths
variable {G_centroid : IsCentroid A B C G} -- G is the centroid of triangle ABC
variable {I_incenter : IsIncenter A B C I} -- I is the incenter of triangle ABC
variable {G_perp_BC : IsPerpendicular G I BC} -- G I ⊥ BC

theorem triangle_side_length (h1 : |AB| = c) (h2 : |AC| = b) :
  |BC| = (b + c) / 2 := 
sorry

end triangle_side_length_l494_49472


namespace Isabel_initial_flowers_l494_49478

-- Constants for conditions
def b := 7  -- Number of bouquets after wilting
def fw := 10  -- Number of wilted flowers
def n := 8  -- Number of flowers in each bouquet

-- Theorem statement
theorem Isabel_initial_flowers (h1 : b = 7) (h2 : fw = 10) (h3 : n = 8) : 
  (b * n + fw = 66) := by
  sorry

end Isabel_initial_flowers_l494_49478


namespace maximum_time_for_3_digit_combination_lock_l494_49459

def max_time_to_open_briefcase : ℕ :=
  let num_combinations := 9 * 9 * 9
  let time_per_trial := 3
  num_combinations * time_per_trial

theorem maximum_time_for_3_digit_combination_lock :
  max_time_to_open_briefcase = 2187 :=
by
  sorry

end maximum_time_for_3_digit_combination_lock_l494_49459


namespace phone_price_increase_is_40_percent_l494_49483

-- Definitions based on the conditions
def initial_price_tv := 500
def increased_fraction_tv := 2 / 5
def initial_price_phone := 400
def total_amount_received := 1260

-- The price increase of the TV
def final_price_tv := initial_price_tv * (1 + increased_fraction_tv)

-- The final price of the phone
def final_price_phone := total_amount_received - final_price_tv

-- The percentage increase in the phone's price
def percentage_increase_phone := ((final_price_phone - initial_price_phone) / initial_price_phone) * 100

-- The theorem to prove
theorem phone_price_increase_is_40_percent :
  percentage_increase_phone = 40 := by
  sorry

end phone_price_increase_is_40_percent_l494_49483


namespace julia_height_in_cm_l494_49438

def height_in_feet : ℕ := 5
def height_in_inches : ℕ := 4
def feet_to_inches : ℕ := 12
def inch_to_cm : ℝ := 2.54

theorem julia_height_in_cm : (height_in_feet * feet_to_inches + height_in_inches) * inch_to_cm = 162.6 :=
sorry

end julia_height_in_cm_l494_49438


namespace jung_kook_blue_balls_l494_49461

def num_boxes := 2
def blue_balls_per_box := 5
def total_blue_balls := num_boxes * blue_balls_per_box

theorem jung_kook_blue_balls : total_blue_balls = 10 :=
by
  sorry

end jung_kook_blue_balls_l494_49461


namespace vertical_distance_rotated_square_l494_49480

-- Lean 4 statement for the mathematically equivalent proof problem
theorem vertical_distance_rotated_square
  (side_length : ℝ)
  (n : ℕ)
  (rot_angle : ℝ)
  (orig_line_height before_rotation : ℝ)
  (diagonal_length : ℝ)
  (lowered_distance : ℝ)
  (highest_point_drop : ℝ)
  : side_length = 2 →
    n = 4 →
    rot_angle = 45 →
    orig_line_height = 1 →
    diagonal_length = side_length * (2:ℝ)^(1/2) →
    lowered_distance = (diagonal_length / 2) - orig_line_height →
    highest_point_drop = lowered_distance →
    2 = 2 :=
    sorry

end vertical_distance_rotated_square_l494_49480


namespace cost_of_mens_t_shirt_l494_49494

-- Definitions based on conditions
def womens_price : ℕ := 18
def womens_interval : ℕ := 30
def mens_interval : ℕ := 40
def shop_open_hours_per_day : ℕ := 12
def total_earnings_per_week : ℕ := 4914

-- Auxiliary definitions based on conditions
def t_shirts_sold_per_hour (interval : ℕ) : ℕ := 60 / interval
def t_shirts_sold_per_day (interval : ℕ) : ℕ := shop_open_hours_per_day * t_shirts_sold_per_hour interval
def t_shirts_sold_per_week (interval : ℕ) : ℕ := t_shirts_sold_per_day interval * 7

def weekly_earnings_womens : ℕ := womens_price * t_shirts_sold_per_week womens_interval
def weekly_earnings_mens : ℕ := total_earnings_per_week - weekly_earnings_womens
def mens_price : ℚ := weekly_earnings_mens / t_shirts_sold_per_week mens_interval

-- The statement to be proved
theorem cost_of_mens_t_shirt : mens_price = 15 := by
  sorry

end cost_of_mens_t_shirt_l494_49494


namespace triangle_DEF_area_l494_49426

noncomputable def point := (ℝ × ℝ)

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_DEF_area : area_of_triangle D E F = 30 := by
  sorry

end triangle_DEF_area_l494_49426


namespace day_before_yesterday_l494_49488

theorem day_before_yesterday (day_after_tomorrow_is_monday : String) : String :=
by
  have tomorrow := "Sunday"
  have today := "Saturday"
  exact today

end day_before_yesterday_l494_49488


namespace rectangular_plot_area_l494_49450

/-- The ratio between the length and the breadth of a rectangular plot is 7 : 5.
    If the perimeter of the plot is 288 meters, then the area of the plot is 5040 square meters.
-/
theorem rectangular_plot_area
    (L B : ℝ)
    (h1 : L / B = 7 / 5)
    (h2 : 2 * (L + B) = 288) :
    L * B = 5040 :=
by
  sorry

end rectangular_plot_area_l494_49450


namespace minimum_a_l494_49412

theorem minimum_a (a : ℝ) (h : a > 0) :
  (∀ (N : ℝ × ℝ), (N.1 - a)^2 + (N.2 + a - 3)^2 = 1 → 
   dist (N.1, N.2) (0, 0) ≥ 2) → a ≥ 3 :=
by
  sorry

end minimum_a_l494_49412


namespace min_elements_l494_49475

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l494_49475


namespace parabola_translation_correct_l494_49454

noncomputable def translate_parabola (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) : Prop :=
  let x' := x - 1
  let y' := y + 3
  y' = -2 * x'^2 - 1

theorem parabola_translation_correct (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) :
  translate_parabola x y h :=
sorry

end parabola_translation_correct_l494_49454


namespace min_rows_needed_l494_49402

-- Define the basic conditions
def total_students := 2016
def seats_per_row := 168
def max_students_per_school := 40

-- Define the minimum number of rows required to accommodate all conditions
noncomputable def min_required_rows (students : ℕ) (seats : ℕ) (max_per_school : ℕ) : ℕ := 15

-- Lean theorem asserting the truth of the above definition under given conditions
theorem min_rows_needed : min_required_rows total_students seats_per_row max_students_per_school = 15 :=
by
  -- Proof omitted
  sorry

end min_rows_needed_l494_49402


namespace max_marks_exam_l494_49463

theorem max_marks_exam (M : ℝ) 
  (h1 : 0.80 * M = 400) :
  M = 500 := 
by
  sorry

end max_marks_exam_l494_49463


namespace find_smaller_number_l494_49410

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number_l494_49410


namespace cube_face_sum_l494_49452

theorem cube_face_sum (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) :
  (a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1287) →
  (a + d + b + e + c + f = 33) :=
by
  sorry

end cube_face_sum_l494_49452


namespace inclination_angle_of_y_axis_l494_49439

theorem inclination_angle_of_y_axis : 
  ∀ (l : ℝ), l = 90 :=
sorry

end inclination_angle_of_y_axis_l494_49439


namespace james_owns_145_l494_49416

theorem james_owns_145 (total : ℝ) (diff : ℝ) (james_and_ali : total = 250) (james_more_than_ali : diff = 40):
  ∃ (james ali : ℝ), ali + diff = james ∧ ali + james = total ∧ james = 145 :=
by
  sorry

end james_owns_145_l494_49416


namespace mangoes_in_basket_B_l494_49473

theorem mangoes_in_basket_B :
  ∀ (A C D E B : ℕ), 
    (A = 15) →
    (C = 20) →
    (D = 25) →
    (E = 35) →
    (5 * 25 = A + C + D + E + B) →
    (B = 30) :=
by
  intros A C D E B hA hC hD hE hSum
  sorry

end mangoes_in_basket_B_l494_49473


namespace coneCannotBeQuadrilateral_l494_49400

-- Define types for our geometric solids
inductive Solid
| Cylinder
| Cone
| FrustumCone
| Prism

-- Define a predicate for whether the cross-section can be a quadrilateral
def canBeQuadrilateral (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => true
  | Solid.Cone => false
  | Solid.FrustumCone => true
  | Solid.Prism => true

-- The theorem we need to prove
theorem coneCannotBeQuadrilateral : canBeQuadrilateral Solid.Cone = false := by
  sorry

end coneCannotBeQuadrilateral_l494_49400
