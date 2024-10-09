import Mathlib

namespace parabola_directrix_l883_88336

variable (a : ℝ)

theorem parabola_directrix (h1 : ∀ x : ℝ, y = a * x^2) (h2 : y = -1/4) : a = 1 :=
sorry

end parabola_directrix_l883_88336


namespace smallest_hiding_number_l883_88397

/-- Define the concept of "hides" -/
def hides (A B : ℕ) : Prop :=
  ∃ (remove : ℕ → ℕ), remove A = B

/-- The smallest natural number that hides all numbers from 2000 to 2021 is 20012013456789 -/
theorem smallest_hiding_number : hides 20012013456789 2000 ∧ hides 20012013456789 2001 ∧ hides 20012013456789 2002 ∧
    hides 20012013456789 2003 ∧ hides 20012013456789 2004 ∧ hides 20012013456789 2005 ∧ hides 20012013456789 2006 ∧
    hides 20012013456789 2007 ∧ hides 20012013456789 2008 ∧ hides 20012013456789 2009 ∧ hides 20012013456789 2010 ∧
    hides 20012013456789 2011 ∧ hides 20012013456789 2012 ∧ hides 20012013456789 2013 ∧ hides 20012013456789 2014 ∧
    hides 20012013456789 2015 ∧ hides 20012013456789 2016 ∧ hides 20012013456789 2017 ∧ hides 20012013456789 2018 ∧
    hides 20012013456789 2019 ∧ hides 20012013456789 2020 ∧ hides 20012013456789 2021 :=
by
  sorry

end smallest_hiding_number_l883_88397


namespace factorization_identity_l883_88389

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 2 * x^2 - 2

-- Define the factorized form
def factorized_expr (x : ℝ) : ℝ := 2 * (x + 1) * (x - 1)

-- The theorem stating the equality
theorem factorization_identity (x : ℝ) : initial_expr x = factorized_expr x := 
by sorry

end factorization_identity_l883_88389


namespace upstream_distance_calc_l883_88350

noncomputable def speed_in_still_water : ℝ := 10.5
noncomputable def downstream_distance : ℝ := 45
noncomputable def downstream_time : ℝ := 3
noncomputable def upstream_time : ℝ := 3

theorem upstream_distance_calc : 
  ∃ (d v : ℝ), (10.5 + v) * downstream_time = downstream_distance ∧ 
               v = 4.5 ∧ 
               d = (10.5 - v) * upstream_time ∧ 
               d = 18 :=
by
  sorry

end upstream_distance_calc_l883_88350


namespace dice_probability_l883_88306

def prob_at_least_one_one : ℚ :=
  let total_outcomes := 36
  let no_1_outcomes := 25
  let favorable_outcomes := total_outcomes - no_1_outcomes
  let probability := favorable_outcomes / total_outcomes
  probability

theorem dice_probability :
  prob_at_least_one_one = 11 / 36 :=
by
  sorry

end dice_probability_l883_88306


namespace michael_lap_time_l883_88343

theorem michael_lap_time (T : ℝ) :
  (∀ (lap_time_donovan : ℝ), lap_time_donovan = 45 → (9 * T) / lap_time_donovan + 1 = 9 → T = 40) :=
by
  intro lap_time_donovan
  intro h1
  intro h2
  sorry

end michael_lap_time_l883_88343


namespace find_a_in_triangle_l883_88386

theorem find_a_in_triangle (C : ℝ) (b c : ℝ) (hC : C = 60) (hb : b = 1) (hc : c = Real.sqrt 3) :
  ∃ (a : ℝ), a = 2 := 
by
  sorry

end find_a_in_triangle_l883_88386


namespace rectangle_perimeter_l883_88360

-- Conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_of_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def perimeter_of_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

-- Given conditions from the problem
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15
def width_of_rectangle : ℕ := 6

-- Main theorem
theorem rectangle_perimeter :
  is_right_triangle a b c →
  area_of_triangle a b = area_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle →
  perimeter_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle = 30 :=
by
  sorry

end rectangle_perimeter_l883_88360


namespace rational_product_sum_l883_88302

theorem rational_product_sum (x y : ℚ) 
  (h1 : x * y < 0) 
  (h2 : x + y < 0) : 
  |y| < |x| ∧ y < 0 ∧ x > 0 ∨ |x| < |y| ∧ x < 0 ∧ y > 0 :=
by
  sorry

end rational_product_sum_l883_88302


namespace larger_number_hcf_lcm_l883_88378

theorem larger_number_hcf_lcm (a b : ℕ) (h1 : Nat.gcd a b = 84) (h2 : Nat.lcm a b = 21) (h3 : a = b / 4) : max a b = 84 :=
by
  sorry

end larger_number_hcf_lcm_l883_88378


namespace eliza_ironing_hours_l883_88325

theorem eliza_ironing_hours (h : ℕ) 
  (blouse_minutes : ℕ := 15) 
  (dress_minutes : ℕ := 20) 
  (hours_ironing_blouses : ℕ := h)
  (hours_ironing_dresses : ℕ := 3)
  (total_clothes : ℕ := 17) :
  ((60 / blouse_minutes) * hours_ironing_blouses) + ((60 / dress_minutes) * hours_ironing_dresses) = total_clothes →
  hours_ironing_blouses = 2 := 
sorry

end eliza_ironing_hours_l883_88325


namespace stamp_collection_cost_l883_88333

def cost_brazil_per_stamp : ℝ := 0.08
def cost_peru_per_stamp : ℝ := 0.05
def num_brazil_stamps_60s : ℕ := 7
def num_peru_stamps_60s : ℕ := 4
def num_brazil_stamps_70s : ℕ := 12
def num_peru_stamps_70s : ℕ := 6

theorem stamp_collection_cost :
  num_brazil_stamps_60s * cost_brazil_per_stamp +
  num_peru_stamps_60s * cost_peru_per_stamp +
  num_brazil_stamps_70s * cost_brazil_per_stamp +
  num_peru_stamps_70s * cost_peru_per_stamp =
  2.02 :=
by
  -- Skipping proof steps.
  sorry

end stamp_collection_cost_l883_88333


namespace question_condition_l883_88376

def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
  (1 - 2 * x) * (x + 1) < 0 → x > 1 / 2 ∨ x < -1

theorem question_condition
(x : ℝ) : sufficient_but_not_necessary_condition x := sorry

end question_condition_l883_88376


namespace ways_to_divide_day_l883_88317

theorem ways_to_divide_day (n m : ℕ+) : n * m = 86400 → 96 = 96 :=
by
  sorry

end ways_to_divide_day_l883_88317


namespace value_of_fraction_l883_88387

theorem value_of_fraction (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 :=
by
  sorry

end value_of_fraction_l883_88387


namespace ambika_candles_count_l883_88372

-- Definitions
def Aniyah_candles (A : ℕ) : ℕ := 6 * A
def combined_candles (A : ℕ) : ℕ := A + Aniyah_candles A

-- Problem Statement:
theorem ambika_candles_count : ∃ A : ℕ, combined_candles A = 28 ∧ A = 4 :=
by
  sorry

end ambika_candles_count_l883_88372


namespace company_pays_each_man_per_hour_l883_88326

theorem company_pays_each_man_per_hour
  (men : ℕ) (hours_per_job : ℕ) (jobs : ℕ) (total_pay : ℕ)
  (completion_time : men * hours_per_job = 1)
  (total_jobs_time : jobs * hours_per_job = 5)
  (total_earning : total_pay = 150) :
  (total_pay / (jobs * men * hours_per_job)) = 10 :=
sorry

end company_pays_each_man_per_hour_l883_88326


namespace part1_part2_l883_88340

def z1 (a : ℝ) : Complex := Complex.mk 2 a
def z2 : Complex := Complex.mk 3 (-4)

-- Part 1: Prove that the product of z1 and z2 equals 10 - 5i when a = 1.
theorem part1 : z1 1 * z2 = Complex.mk 10 (-5) :=
by
  -- proof to be filled in
  sorry

-- Part 2: Prove that a = 4 when z1 + z2 is a real number.
theorem part2 (a : ℝ) (h : (z1 a + z2).im = 0) : a = 4 :=
by
  -- proof to be filled in
  sorry

end part1_part2_l883_88340


namespace A_pow_five_eq_rA_add_sI_l883_88361

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 1; 4, 3]

def I : Matrix (Fin 2) (Fin 2) ℚ :=
  1

theorem A_pow_five_eq_rA_add_sI :
  ∃ (r s : ℚ), (A^5) = r • A + s • I :=
sorry

end A_pow_five_eq_rA_add_sI_l883_88361


namespace single_reduction_equivalent_l883_88327

theorem single_reduction_equivalent (P : ℝ) (h1 : P > 0) :
  let final_price := 0.75 * P - 0.7 * (0.75 * P)
  let single_reduction := (P - final_price) / P
  single_reduction * 100 = 77.5 := 
by
  sorry

end single_reduction_equivalent_l883_88327


namespace remainder_of_prime_when_divided_by_240_l883_88354

theorem remainder_of_prime_when_divided_by_240 (n : ℕ) (hn : n > 0) (hp : Nat.Prime (2^n + 1)) : (2^n + 1) % 240 = 17 := 
sorry

end remainder_of_prime_when_divided_by_240_l883_88354


namespace intersection_A_B_eq_B_l883_88337

variable (a : ℝ) (A : Set ℝ) (B : Set ℝ)

def satisfies_quadratic (a : ℝ) (x : ℝ) : Prop := x^2 - a*x + 1 = 0

def set_A : Set ℝ := {1, 2, 3}

def set_B (a : ℝ) : Set ℝ := {x | satisfies_quadratic a x}

theorem intersection_A_B_eq_B (a : ℝ) (h : a ∈ set_A) : 
  (∀ x, x ∈ set_B a → x ∈ set_A) → (∃ x, x ∈ set_A ∧ satisfies_quadratic a x) →
  a = 2 :=
sorry

end intersection_A_B_eq_B_l883_88337


namespace polygon_has_twelve_sides_l883_88309

theorem polygon_has_twelve_sides
  (sum_exterior_angles : ℝ)
  (sum_interior_angles : ℝ → ℝ)
  (n : ℝ)
  (h1 : sum_exterior_angles = 360)
  (h2 : ∀ n, sum_interior_angles n = 180 * (n - 2))
  (h3 : ∀ n, sum_interior_angles n = 5 * sum_exterior_angles) :
  n = 12 :=
by
  sorry

end polygon_has_twelve_sides_l883_88309


namespace milford_age_in_3_years_l883_88315

theorem milford_age_in_3_years (current_age_eustace : ℕ) (current_age_milford : ℕ) :
  (current_age_eustace = 2 * current_age_milford) → 
  (current_age_eustace + 3 = 39) → 
  current_age_milford + 3 = 21 :=
by
  intros h1 h2
  sorry

end milford_age_in_3_years_l883_88315


namespace platform_length_l883_88382

noncomputable def length_of_platform (L : ℝ) : Prop :=
  ∃ (a : ℝ), 
    -- Train starts from rest
    (0 : ℝ) * 24 + (1/2) * a * 24^2 = 300 ∧
    -- Train crosses a platform in 39 seconds
    (0 : ℝ) * 39 + (1/2) * a * 39^2 = 300 + L ∧
    -- Constant acceleration found
    a = (25 : ℝ) / 24

-- Claim that length of platform should be 492.19 meters
theorem platform_length : length_of_platform 492.19 :=
sorry

end platform_length_l883_88382


namespace boxcar_capacity_ratio_l883_88381

theorem boxcar_capacity_ratio :
  ∀ (total_capacity : ℕ)
    (num_red num_blue num_black : ℕ)
    (black_capacity blue_capacity : ℕ)
    (red_capacity : ℕ),
    num_red = 3 →
    num_blue = 4 →
    num_black = 7 →
    black_capacity = 4000 →
    blue_capacity = 2 * black_capacity →
    total_capacity = 132000 →
    total_capacity = num_red * red_capacity + num_blue * blue_capacity + num_black * black_capacity →
    (red_capacity / blue_capacity = 3) :=
by
  intros total_capacity num_red num_blue num_black black_capacity blue_capacity red_capacity
         h_num_red h_num_blue h_num_black h_black_capacity h_blue_capacity h_total_capacity h_combined_capacity
  sorry

end boxcar_capacity_ratio_l883_88381


namespace find_constant_a_l883_88322

theorem find_constant_a (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = (1/2) * 3^(n+1) - a) →
  a = 3/2 :=
sorry

end find_constant_a_l883_88322


namespace ratio_playground_landscape_l883_88362

-- Defining the conditions
def breadth := 420
def length := breadth / 6
def playground_area := 4200
def landscape_area := length * breadth

-- Stating the theorem to prove the ratio is 1:7
theorem ratio_playground_landscape :
  (playground_area.toFloat / landscape_area.toFloat) = (1.0 / 7.0) :=
by
  sorry

end ratio_playground_landscape_l883_88362


namespace animath_workshop_lists_l883_88339

/-- The 79 trainees of the Animath workshop each choose an activity for the free afternoon 
among 5 offered activities. It is known that:
- The swimming pool was at least as popular as soccer.
- The students went shopping in groups of 5.
- No more than 4 students played cards.
- At most one student stayed in their room.
We write down the number of students who participated in each activity.
How many different lists could we have written? --/
theorem animath_workshop_lists :
  ∃ (l : ℕ), l = Nat.choose 81 2 := 
sorry

end animath_workshop_lists_l883_88339


namespace apples_in_box_ratio_mixed_fruits_to_total_l883_88385

variable (total_fruits : Nat) (oranges : Nat) (peaches : Nat) (apples : Nat) (mixed_fruits : Nat)
variable (one_fourth_of_box_contains_oranges : oranges = total_fruits / 4)
variable (half_as_many_peaches_as_oranges : peaches = oranges / 2)
variable (five_times_as_many_apples_as_peaches : apples = 5 * peaches)
variable (mixed_fruits_double_peaches : mixed_fruits = 2 * peaches)
variable (total_fruits_56 : total_fruits = 56)

theorem apples_in_box : apples = 35 := by
  sorry

theorem ratio_mixed_fruits_to_total : mixed_fruits / total_fruits = 1 / 4 := by
  sorry

end apples_in_box_ratio_mixed_fruits_to_total_l883_88385


namespace inscribed_sphere_radius_base_height_l883_88318

noncomputable def radius_of_inscribed_sphere (r base_radius height : ℝ) := 
  r = (30 / (Real.sqrt 5 + 1)) * (Real.sqrt 5 - 1) 

theorem inscribed_sphere_radius_base_height (r : ℝ) (b d : ℝ) (base_radius height : ℝ) 
  (h_base: base_radius = 15) (h_height: height = 30) 
  (h_radius: radius_of_inscribed_sphere r base_radius height) 
  (h_expr: r = b * (Real.sqrt d) - b) : 
  b + d = 12.5 :=
sorry

end inscribed_sphere_radius_base_height_l883_88318


namespace solution_set_inequality_l883_88375

theorem solution_set_inequality (a x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) :=
by sorry

end solution_set_inequality_l883_88375


namespace cos_fourth_minus_sin_fourth_l883_88374

theorem cos_fourth_minus_sin_fourth (α : ℝ) (h : Real.sin α = (Real.sqrt 5) / 5) :
  Real.cos α ^ 4 - Real.sin α ^ 4 = 3 / 5 := 
sorry

end cos_fourth_minus_sin_fourth_l883_88374


namespace invertible_elements_mod_8_l883_88352

theorem invertible_elements_mod_8 :
  {x : ℤ | (x * x) % 8 = 1} = {1, 3, 5, 7} :=
by
  sorry

end invertible_elements_mod_8_l883_88352


namespace arthur_bought_2_hamburgers_on_second_day_l883_88390

theorem arthur_bought_2_hamburgers_on_second_day
  (H D X: ℕ)
  (h1: 3 * H + 4 * D = 10)
  (h2: D = 1)
  (h3: 2 * X + 3 * D = 7):
  X = 2 :=
by
  sorry

end arthur_bought_2_hamburgers_on_second_day_l883_88390


namespace magnitude_of_complex_l883_88380

variable (z : ℂ)
variable (h : Complex.I * z = 3 - 4 * Complex.I)

theorem magnitude_of_complex :
  Complex.abs z = 5 :=
by
  sorry

end magnitude_of_complex_l883_88380


namespace cos_alpha_minus_pi_over_4_l883_88345

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tan : Real.tan α = 2) :
  Real.cos (α - π / 4) = (3 * Real.sqrt 10) / 10 := 
  sorry

end cos_alpha_minus_pi_over_4_l883_88345


namespace solve_system_l883_88392

theorem solve_system (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (eq1 : a * y + b * x = c)
  (eq2 : c * x + a * z = b)
  (eq3 : b * z + c * y = a) :
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧ 
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧ 
  z = (a^2 + b^2 - c^2) / (2 * a * b) :=
sorry

end solve_system_l883_88392


namespace largest_integer_x_l883_88371

theorem largest_integer_x (x : ℤ) : (x / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) → x ≤ 7 ∧ (7 / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) :=
by
  sorry

end largest_integer_x_l883_88371


namespace number_of_valid_sequences_l883_88373

-- Definitions for conditions
def digit := Fin 10 -- Digit can be any number from 0 to 9
def is_odd (n : digit) : Prop := n.val % 2 = 1
def is_even (n : digit) : Prop := n.val % 2 = 0

def valid_sequence (s : Fin 8 → digit) : Prop :=
  ∀ i : Fin 7, (is_odd (s i) ↔ is_even (s (i+1)))

-- Theorem statement
theorem number_of_valid_sequences : 
  ∃ n, n = 781250 ∧ 
    ∃ s : (Fin 8 → digit), valid_sequence s :=
sorry -- Proof is not required

end number_of_valid_sequences_l883_88373


namespace find_m_no_solution_l883_88359

-- Define the condition that the equation has no solution
def no_solution (m : ℤ) : Prop :=
  ∀ x : ℤ, (x + m)/(4 - x^2) + x / (x - 2) ≠ 1

-- State the proof problem in Lean 4
theorem find_m_no_solution : ∀ m : ℤ, no_solution m → (m = 2 ∨ m = 6) :=
by
  sorry

end find_m_no_solution_l883_88359


namespace part1_part2_l883_88334

noncomputable def f (a x : ℝ) : ℝ := (Real.exp x) - a * x ^ 2 - x

theorem part1 {a : ℝ} : (∀ x y: ℝ, x < y → f a x ≤ f a y) ↔ (a = 1 / 2) :=
sorry

theorem part2 {a : ℝ} (h1 : a > 1 / 2):
  ∃ (x1 x2 : ℝ), (x1 < x2) ∧ (f a x2 < 1 + (Real.sin x2 - x2) / 2) :=
sorry

end part1_part2_l883_88334


namespace no_pre_period_decimal_representation_l883_88396

theorem no_pre_period_decimal_representation (m : ℕ) (h : Nat.gcd m 10 = 1) : ¬∃ k : ℕ, ∃ a : ℕ, 0 < a ∧ 10^a < m ∧ (10^a - 1) % m = k ∧ k ≠ 0 :=
sorry

end no_pre_period_decimal_representation_l883_88396


namespace geoff_total_spending_l883_88351

def price_day1 : ℕ := 60
def pairs_day1 : ℕ := 2
def price_per_pair_day1 : ℕ := price_day1 / pairs_day1

def multiplier_day2 : ℕ := 3
def price_per_pair_day2 : ℕ := price_per_pair_day1 * 3 / 2
def discount_day2 : Real := 0.10
def cost_before_discount_day2 : ℕ := multiplier_day2 * price_per_pair_day2
def cost_after_discount_day2 : Real := cost_before_discount_day2 * (1 - discount_day2)

def multiplier_day3 : ℕ := 5
def price_per_pair_day3 : ℕ := price_per_pair_day1 * 2
def sales_tax_day3 : Real := 0.08
def cost_before_tax_day3 : ℕ := multiplier_day3 * price_per_pair_day3
def cost_after_tax_day3 : Real := cost_before_tax_day3 * (1 + sales_tax_day3)

def total_cost : Real := price_day1 + cost_after_discount_day2 + cost_after_tax_day3

theorem geoff_total_spending : total_cost = 505.50 := by
  sorry

end geoff_total_spending_l883_88351


namespace parametric_to_ellipse_parametric_to_line_l883_88379

-- Define the conditions and the corresponding parametric equations
variable (φ t : ℝ) (x y : ℝ)

-- The first parametric equation converted to the ordinary form
theorem parametric_to_ellipse (h1 : x = 5 * Real.cos φ) (h2 : y = 4 * Real.sin φ) :
  (x ^ 2 / 25) + (y ^ 2 / 16) = 1 := sorry

-- The second parametric equation converted to the ordinary form
theorem parametric_to_line (h3 : x = 1 - 3 * t) (h4 : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := sorry

end parametric_to_ellipse_parametric_to_line_l883_88379


namespace subset_problem_l883_88370

theorem subset_problem (a : ℝ) (P S : Set ℝ) :
  P = { x | x^2 - 2 * x - 3 = 0 } →
  S = { x | a * x + 2 = 0 } →
  (S ⊆ P) →
  (a = 0 ∨ a = 2 ∨ a = -2 / 3) :=
by
  intro hP hS hSubset
  sorry

end subset_problem_l883_88370


namespace exp_function_not_increasing_l883_88366

open Real

theorem exp_function_not_increasing (a : ℝ) (x : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : a < 1) :
  ¬(∀ x₁ x₂ : ℝ, x₁ < x₂ → a^x₁ < a^x₂) := by
  sorry

end exp_function_not_increasing_l883_88366


namespace taco_truck_revenue_l883_88391

-- Conditions
def price_of_soft_taco : ℕ := 2
def price_of_hard_taco : ℕ := 5
def family_soft_tacos : ℕ := 3
def family_hard_tacos : ℕ := 4
def other_customers : ℕ := 10
def soft_tacos_per_other_customer : ℕ := 2

-- Calculation
def total_soft_tacos : ℕ := family_soft_tacos + other_customers * soft_tacos_per_other_customer
def revenue_from_soft_tacos : ℕ := total_soft_tacos * price_of_soft_taco
def revenue_from_hard_tacos : ℕ := family_hard_tacos * price_of_hard_taco
def total_revenue : ℕ := revenue_from_soft_tacos + revenue_from_hard_tacos

-- The proof problem
theorem taco_truck_revenue : total_revenue = 66 := 
by 
-- The proof should go here
sorry

end taco_truck_revenue_l883_88391


namespace additional_pass_combinations_l883_88335

def original_combinations : ℕ := 4 * 2 * 3 * 3
def new_combinations : ℕ := 6 * 2 * 4 * 3
def additional_combinations : ℕ := new_combinations - original_combinations

theorem additional_pass_combinations : additional_combinations = 72 := by
  sorry

end additional_pass_combinations_l883_88335


namespace rectangle_area_formula_l883_88395

-- Define the given conditions: perimeter is 20, one side length is x
def rectangle_perimeter (P x : ℝ) (w : ℝ) : Prop := P = 2 * (x + w)
def rectangle_area (x w : ℝ) : ℝ := x * w

-- The theorem to prove
theorem rectangle_area_formula (x : ℝ) (h_perimeter : rectangle_perimeter 20 x (10 - x)) : 
  rectangle_area x (10 - x) = x * (10 - x) := 
by 
  sorry

end rectangle_area_formula_l883_88395


namespace expIConjugate_l883_88342

open Complex

-- Define the given condition
def expICondition (θ φ : ℝ) : Prop :=
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I

-- The theorem we want to prove
theorem expIConjugate (θ φ : ℝ) (h : expICondition θ φ) : 
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I :=
sorry

end expIConjugate_l883_88342


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l883_88328

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l883_88328


namespace simple_interest_rate_l883_88313

theorem simple_interest_rate (P A : ℝ) (T : ℕ) (R : ℝ) 
  (P_pos : P = 800) (A_pos : A = 950) (T_pos : T = 5) :
  R = 3.75 :=
by
  sorry

end simple_interest_rate_l883_88313


namespace product_has_no_linear_term_l883_88364

theorem product_has_no_linear_term (m : ℝ) (h : ((x : ℝ) → (x - m) * (x - 3) = x^2 + 3 * m)) : m = -3 := 
by
  sorry

end product_has_no_linear_term_l883_88364


namespace line_pass_through_point_l883_88363

theorem line_pass_through_point (k b : ℝ) (x1 x2 : ℝ) (h1: b ≠ 0) (h2: x1^2 - k*x1 - b = 0) (h3: x2^2 - k*x2 - b = 0)
(h4: x1 + x2 = k) (h5: x1 * x2 = -b) 
(h6: (k^2 * (-b) + k * b * k + b^2 = b^2) = true) : 
  ∃ (x y : ℝ), (y = k * x + 1) ∧ (x, y) = (0, 1) :=
by
  sorry

end line_pass_through_point_l883_88363


namespace ratio_of_professionals_l883_88331

-- Define the variables and conditions as stated in the problem.
variables (e d l : ℕ)

-- The condition about the average ages leading to the given equation.
def avg_age_condition : Prop := (40 * e + 50 * d + 60 * l) / (e + d + l) = 45

-- The statement to prove that given the average age condition, the ratio is 1:1:3.
theorem ratio_of_professionals (h : avg_age_condition e d l) : e = d + 3 * l :=
sorry

end ratio_of_professionals_l883_88331


namespace vasya_can_construct_polyhedron_l883_88305

-- Definition of a polyhedron using given set of shapes
-- where the original set of shapes can form a polyhedron
def original_set_can_form_polyhedron (squares triangles : ℕ) : Prop :=
  squares = 1 ∧ triangles = 4

-- Transformation condition: replacing 2 triangles with 2 squares
def replacement_condition (initial_squares initial_triangles replaced_squares replaced_triangles : ℕ) : Prop :=
  initial_squares + 2 = replaced_squares ∧ initial_triangles - 2 = replaced_triangles

-- Proving that new set of shapes can form a polyhedron
theorem vasya_can_construct_polyhedron :
  ∃ (new_squares new_triangles : ℕ),
    (original_set_can_form_polyhedron 1 4)
    ∧ (replacement_condition 1 4 new_squares new_triangles)
    ∧ (new_squares = 3 ∧ new_triangles = 2) :=
by
  sorry

end vasya_can_construct_polyhedron_l883_88305


namespace complex_number_quadrant_l883_88311

open Complex

theorem complex_number_quadrant (z : ℂ) (h : (1 + 2 * Complex.I) / z = Complex.I) : 
  (0 < z.re) ∧ (0 < z.im) :=
by
  -- sorry to skip the actual proof
  sorry

end complex_number_quadrant_l883_88311


namespace class_B_more_uniform_than_class_A_l883_88384

-- Definitions based on the given problem
def class_height_variance (class_name : String) : ℝ :=
  if class_name = "A" then 3.24 else if class_name = "B" then 1.63 else 0

-- The theorem statement proving that Class B has more uniform heights (smaller variance)
theorem class_B_more_uniform_than_class_A :
  class_height_variance "B" < class_height_variance "A" :=
by
  sorry

end class_B_more_uniform_than_class_A_l883_88384


namespace no_strategy_for_vasya_tolya_l883_88308

-- This definition encapsulates the conditions and question
def players_game (coins : ℕ) : Prop :=
  ∀ p v t : ℕ, 
    (1 ≤ p ∧ p ≤ 4) ∧ (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
    (∃ (n : ℕ), coins = 5 * n)

-- Theorem formalizing the problem's conclusion
theorem no_strategy_for_vasya_tolya (n : ℕ) (h : n = 300) : 
  ¬ ∀ (v t : ℕ), 
     (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
     players_game (n - v - t) :=
by
  intro h
  sorry -- Skip the proof, as it is not required

end no_strategy_for_vasya_tolya_l883_88308


namespace shanghai_expo_visitors_l883_88347

theorem shanghai_expo_visitors :
  505000 = 5.05 * 10^5 :=
by
  sorry

end shanghai_expo_visitors_l883_88347


namespace three_point_sixty_eight_as_fraction_l883_88368

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l883_88368


namespace intersection_complement_l883_88358

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_complement :
  A ∩ ({x | x < -1 ∨ x > 3} : Set ℝ) = {x | 3 < x ∧ x < 4} :=
by
  sorry

end intersection_complement_l883_88358


namespace ned_initial_lives_l883_88300

variable (lost_lives : ℕ) (current_lives : ℕ) 
variable (initial_lives : ℕ)

theorem ned_initial_lives (h_lost: lost_lives = 13) (h_current: current_lives = 70) :
  initial_lives = current_lives + lost_lives := by
  sorry

end ned_initial_lives_l883_88300


namespace fraction_is_one_fifth_l883_88304

theorem fraction_is_one_fifth (f : ℚ) (h1 : f * 50 - 4 = 6) : f = 1 / 5 :=
by
  sorry

end fraction_is_one_fifth_l883_88304


namespace centroid_value_l883_88301

-- Define the points P, Q, R
def P : ℝ × ℝ := (4, 3)
def Q : ℝ × ℝ := (-1, 6)
def R : ℝ × ℝ := (7, -2)

-- Define the coordinates of the centroid S
noncomputable def S : ℝ × ℝ := 
  ( (4 + (-1) + 7) / 3, (3 + 6 + (-2)) / 3 )

-- Statement to prove
theorem centroid_value : 
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  8 * x + 3 * y = 101 / 3 :=
by
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  have h: 8 * x + 3 * y = 101 / 3 := sorry
  exact h

end centroid_value_l883_88301


namespace opposite_of_neg_2023_l883_88338

theorem opposite_of_neg_2023 : -( -2023 ) = 2023 := by
  sorry

end opposite_of_neg_2023_l883_88338


namespace gina_expenditure_l883_88312

noncomputable def gina_total_cost : ℝ :=
  let regular_classes_cost := 12 * 450
  let lab_classes_cost := 6 * 550
  let textbooks_cost := 3 * 150
  let online_resources_cost := 4 * 95
  let facilities_fee := 200
  let lab_fee := 6 * 75
  let total_cost := regular_classes_cost + lab_classes_cost + textbooks_cost + online_resources_cost + facilities_fee + lab_fee
  let scholarship_amount := 0.5 * regular_classes_cost
  let discount_amount := 0.25 * lab_classes_cost
  let adjusted_cost := total_cost - scholarship_amount - discount_amount
  let interest := 0.04 * adjusted_cost
  adjusted_cost + interest

theorem gina_expenditure : gina_total_cost = 5881.20 :=
by
  sorry

end gina_expenditure_l883_88312


namespace each_cut_piece_weight_l883_88303

theorem each_cut_piece_weight (L : ℕ) (W : ℕ) (c : ℕ) 
  (hL : L = 20) (hW : W = 150) (hc : c = 2) : (L / c) * W = 1500 := by
  sorry

end each_cut_piece_weight_l883_88303


namespace new_person_weight_l883_88324

theorem new_person_weight (W : ℝ) (old_weight : ℝ) (increase_per_person : ℝ) (num_persons : ℕ)
  (h1 : old_weight = 68)
  (h2 : increase_per_person = 5.5)
  (h3 : num_persons = 5)
  (h4 : W = old_weight + increase_per_person * num_persons) :
  W = 95.5 :=
by
  sorry

end new_person_weight_l883_88324


namespace alpha_epsilon_time_difference_l883_88353

def B := 100
def M := 120
def A := B - 10

theorem alpha_epsilon_time_difference : M - A = 30 := by
  sorry

end alpha_epsilon_time_difference_l883_88353


namespace intersection_of_curves_l883_88319

theorem intersection_of_curves (x : ℝ) (y : ℝ) (h₁ : y = 9 / (x^2 + 3)) (h₂ : x + y = 3) : x = 0 :=
sorry

end intersection_of_curves_l883_88319


namespace find_coefficients_l883_88316

-- Define the polynomial
def poly (a b : ℤ) (x : ℚ) : ℚ := a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8

-- Define the factor
def factor (x : ℚ) : ℚ := 3 * x^2 - 2 * x + 2

-- States that for a given polynomial and factor, the resulting (a, b) pair is (-51, 25)
theorem find_coefficients :
  ∃ a b c d : ℤ, 
  (∀ x, poly a b x = (factor x) * (c * x^2 + d * x + 4)) ∧ 
  a = -51 ∧ 
  b = 25 :=
by sorry

end find_coefficients_l883_88316


namespace molecular_weight_of_one_mole_l883_88307

-- Definitions as Conditions
def total_molecular_weight := 960
def number_of_moles := 5

-- The theorem statement
theorem molecular_weight_of_one_mole :
  total_molecular_weight / number_of_moles = 192 :=
by
  sorry

end molecular_weight_of_one_mole_l883_88307


namespace minimum_time_to_replace_shades_l883_88348

theorem minimum_time_to_replace_shades :
  ∀ (C : ℕ) (S : ℕ) (T : ℕ) (E : ℕ),
  ((C = 60) ∧ (S = 4) ∧ (T = 5) ∧ (E = 48)) →
  ((C * S * T) / E = 25) :=
by
  intros C S T E h
  rcases h with ⟨hC, hS, hT, hE⟩
  sorry

end minimum_time_to_replace_shades_l883_88348


namespace carla_marble_purchase_l883_88357

variable (started_with : ℕ) (now_has : ℕ) (bought : ℕ)

theorem carla_marble_purchase (h1 : started_with = 53) (h2 : now_has = 187) : bought = 134 := by
  sorry

end carla_marble_purchase_l883_88357


namespace actual_average_height_l883_88329

theorem actual_average_height 
  (incorrect_avg_height : ℝ)
  (num_students : ℕ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_avg_height : ℝ) :
  incorrect_avg_height = 175 →
  num_students = 20 →
  incorrect_height = 151 →
  correct_height = 111 →
  actual_avg_height = 173 :=
by
  sorry

end actual_average_height_l883_88329


namespace parallelogram_base_l883_88356

theorem parallelogram_base (A h b : ℝ) (hA : A = 375) (hh : h = 15) : b = 25 :=
by
  sorry

end parallelogram_base_l883_88356


namespace cameron_list_count_l883_88346

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l883_88346


namespace possible_values_y_l883_88332

theorem possible_values_y (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y : ℝ, (y = 0 ∨ y = 41 ∨ y = 144) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end possible_values_y_l883_88332


namespace kaleb_first_load_pieces_l883_88367

-- Definitions of given conditions
def total_pieces : ℕ := 39
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- Definition for calculation of pieces in equal loads
def pieces_in_equal_loads : ℕ := num_equal_loads * pieces_per_load

-- Definition for pieces in the first load
def pieces_in_first_load : ℕ := total_pieces - pieces_in_equal_loads

-- Statement to prove that the pieces in the first load is 19
theorem kaleb_first_load_pieces : pieces_in_first_load = 19 := 
by
  -- The proof is skipped
  sorry

end kaleb_first_load_pieces_l883_88367


namespace yellow_marbles_in_C_l883_88369

theorem yellow_marbles_in_C 
  (Y : ℕ)
  (conditionA : 4 - 2 ≠ 6)
  (conditionB : 6 - 1 ≠ 6)
  (conditionC1 : 3 > Y → 3 - Y = 6)
  (conditionC2 : Y > 3 → Y - 3 = 6) :
  Y = 9 :=
by
  sorry

end yellow_marbles_in_C_l883_88369


namespace cubic_sum_identity_l883_88383

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 40) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 637 :=
by
  sorry

end cubic_sum_identity_l883_88383


namespace base7_to_base10_div_l883_88344

theorem base7_to_base10_div (x y : ℕ) (h : 546 = x * 10^2 + y * 10 + 9) : (x + y + 9) / 21 = 6 / 7 :=
by {
  sorry
}

end base7_to_base10_div_l883_88344


namespace net_income_difference_l883_88341

-- Define Terry's and Jordan's daily income and working days
def terryDailyIncome : ℝ := 24
def terryWorkDays : ℝ := 7
def jordanDailyIncome : ℝ := 30
def jordanWorkDays : ℝ := 6

-- Define the tax rate
def taxRate : ℝ := 0.10

-- Calculate weekly gross incomes
def terryGrossWeeklyIncome : ℝ := terryDailyIncome * terryWorkDays
def jordanGrossWeeklyIncome : ℝ := jordanDailyIncome * jordanWorkDays

-- Calculate tax deductions
def terryTaxDeduction : ℝ := taxRate * terryGrossWeeklyIncome
def jordanTaxDeduction : ℝ := taxRate * jordanGrossWeeklyIncome

-- Calculate net weekly incomes
def terryNetWeeklyIncome : ℝ := terryGrossWeeklyIncome - terryTaxDeduction
def jordanNetWeeklyIncome : ℝ := jordanGrossWeeklyIncome - jordanTaxDeduction

-- Calculate the difference
def incomeDifference : ℝ := jordanNetWeeklyIncome - terryNetWeeklyIncome

-- The theorem to be proven
theorem net_income_difference :
  incomeDifference = 10.80 :=
by
  sorry

end net_income_difference_l883_88341


namespace cage_cost_correct_l883_88349

def cost_of_cat_toy : Real := 10.22
def total_cost_of_purchases : Real := 21.95
def cost_of_cage : Real := total_cost_of_purchases - cost_of_cat_toy

theorem cage_cost_correct : cost_of_cage = 11.73 := by
  sorry

end cage_cost_correct_l883_88349


namespace power_division_identity_l883_88310

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l883_88310


namespace arithmetic_sequence_sum_properties_l883_88398

theorem arithmetic_sequence_sum_properties {S : ℕ → ℝ} {a : ℕ → ℝ} (d : ℝ)
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  let a6 := (S 6 - S 5)
  let a7 := (S 7 - S 6)
  (d = a7 - a6) →
  d < 0 ∧ S 12 > 0 ∧ ¬(∀ n, S n = S 11) ∧ abs a6 > abs a7 :=
by
  sorry

end arithmetic_sequence_sum_properties_l883_88398


namespace correct_operations_l883_88399

theorem correct_operations : 6 * 3 + 4 + 2 = 24 := by
  -- Proof goes here
  sorry

end correct_operations_l883_88399


namespace starWars_earnings_correct_l883_88365

-- Define the given conditions
def lionKing_cost : ℕ := 10
def lionKing_earnings : ℕ := 200
def starWars_cost : ℕ := 25
def lionKing_profit : ℕ := lionKing_earnings - lionKing_cost
def starWars_profit : ℕ := lionKing_profit * 2
def starWars_earnings : ℕ := starWars_profit + starWars_cost

-- The theorem which states that the Star Wars earnings are indeed 405 million
theorem starWars_earnings_correct : starWars_earnings = 405 := by
  -- proof goes here
  sorry

end starWars_earnings_correct_l883_88365


namespace solve_3x_plus_5_squared_l883_88394

theorem solve_3x_plus_5_squared (x : ℝ) (h : 5 * x - 6 = 15 * x + 21) : 
  3 * (x + 5) ^ 2 = 2523 / 100 :=
by
  sorry

end solve_3x_plus_5_squared_l883_88394


namespace area_quadrilateral_is_60_l883_88388

-- Definitions of the lengths of the quadrilateral sides and the ratio condition
def AB : ℝ := 8
def BC : ℝ := 5
def CD : ℝ := 17
def DA : ℝ := 10

-- Function representing the area of the quadrilateral ABCD
def area_ABCD (AB BC CD DA : ℝ) (ratio: ℝ) : ℝ :=
  -- Here we define the function to calculate the area, incorporating the given ratio
  sorry

-- The theorem to show that the area of quadrilateral ABCD is 60
theorem area_quadrilateral_is_60 : 
  area_ABCD AB BC CD DA (1/2) = 60 :=
by
  sorry

end area_quadrilateral_is_60_l883_88388


namespace students_walk_home_fraction_l883_88330

theorem students_walk_home_fraction :
  (1 - (3 / 8 + 2 / 5 + 1 / 8 + 5 / 100)) = (1 / 20) :=
by 
  -- The detailed proof is complex and would require converting these fractions to a common denominator,
  -- performing the arithmetic operations carefully and using Lean's rational number properties. Thus,
  -- the full detailed proof can be written with further steps, but here we insert 'sorry' to focus on the statement.
  sorry

end students_walk_home_fraction_l883_88330


namespace sum_reciprocal_of_roots_l883_88393

variables {m n : ℝ}

-- Conditions: m and n are real roots of the quadratic equation x^2 + 4x - 1 = 0
def is_root (a : ℝ) : Prop := a^2 + 4 * a - 1 = 0

theorem sum_reciprocal_of_roots (hm : is_root m) (hn : is_root n) : 
  (1 / m) + (1 / n) = 4 :=
by sorry

end sum_reciprocal_of_roots_l883_88393


namespace expression_divisible_by_1897_l883_88321

theorem expression_divisible_by_1897 (n : ℕ) :
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end expression_divisible_by_1897_l883_88321


namespace triangle_side_a_l883_88320

theorem triangle_side_a (a : ℝ) : 2 < a ∧ a < 8 → a = 7 :=
by
  sorry

end triangle_side_a_l883_88320


namespace find_point_B_l883_88355

structure Point where
  x : ℝ
  y : ℝ

def vec_scalar_mult (c : ℝ) (v : Point) : Point :=
  ⟨c * v.x, c * v.y⟩

def vec_add (p : Point) (v : Point) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

theorem find_point_B :
  let A := Point.mk 1 (-3)
  let a := Point.mk 3 4
  let B := vec_add A (vec_scalar_mult 2 a)
  B = Point.mk 7 5 :=
by {
  sorry
}

end find_point_B_l883_88355


namespace min_value_abs_expression_l883_88314

theorem min_value_abs_expression {p x : ℝ} (hp1 : 0 < p) (hp2 : p < 15) (hx1 : p ≤ x) (hx2 : x ≤ 15) :
  |x - p| + |x - 15| + |x - p - 15| = 15 :=
sorry

end min_value_abs_expression_l883_88314


namespace sally_took_home_pens_l883_88323

theorem sally_took_home_pens
    (initial_pens : ℕ)
    (students : ℕ)
    (pens_per_student : ℕ)
    (locker_fraction : ℕ)
    (total_pens_given : ℕ)
    (remainder : ℕ)
    (locker_pens : ℕ)
    (home_pens : ℕ) :
    initial_pens = 5230 →
    students = 89 →
    pens_per_student = 58 →
    locker_fraction = 2 →
    total_pens_given = students * pens_per_student →
    remainder = initial_pens - total_pens_given →
    locker_pens = remainder / locker_fraction →
    home_pens = locker_pens →
    home_pens = 34 :=
by {
  sorry
}

end sally_took_home_pens_l883_88323


namespace target_expression_l883_88377

variable (a b : ℤ)

-- Definitions based on problem conditions
def op1 (x y : ℤ) : ℤ := x + y  -- "!" could be addition
def op2 (x y : ℤ) : ℤ := x - y  -- "?" could be subtraction in one order

-- Using these operations to create expressions
def exp1 (a b : ℤ) := op1 (op2 a b) (op2 b a)

def exp2 (x y : ℤ) := op2 (op2 x 0) (op2 0 y)

-- The final expression we need to check
def final_exp (a b : ℤ) := exp1 (20 * a) (18 * b)

-- Theorem proving the final expression equals target
theorem target_expression : final_exp a b = 20 * a - 18 * b :=
sorry

end target_expression_l883_88377
