import Mathlib

namespace sum_of_odd_integers_from_13_to_53_l77_7770

-- Definition of the arithmetic series summing from 13 to 53 with common difference 2
def sum_of_arithmetic_series (a l d : ℕ) (n : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Main theorem
theorem sum_of_odd_integers_from_13_to_53 :
  sum_of_arithmetic_series 13 53 2 21 = 693 := 
sorry

end sum_of_odd_integers_from_13_to_53_l77_7770


namespace six_dice_not_same_probability_l77_7792

theorem six_dice_not_same_probability :
  let total_outcomes := 6^6
  let all_same := 6
  let probability_all_same := all_same / total_outcomes
  let probability_not_all_same := 1 - probability_all_same
  probability_not_all_same = 7775 / 7776 :=
by
  sorry

end six_dice_not_same_probability_l77_7792


namespace mother_age_when_harry_born_l77_7769

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end mother_age_when_harry_born_l77_7769


namespace problem1_eval_problem2_eval_l77_7778

-- Problem 1 equivalent proof problem
theorem problem1_eval : |(-2 + 1/4)| - (-3/4) + 1 - |(1 - 1/2)| = 3 + 1/2 := 
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2_eval : -3^2 - (8 / (-2)^3 - 1) + 3 / 2 * (1 / 2) = -6 + 1/4 :=
by
  sorry

end problem1_eval_problem2_eval_l77_7778


namespace am_gm_inequality_l77_7720

variable {x y z : ℝ}

theorem am_gm_inequality (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y + z) / 3 ≥ Real.sqrt (Real.sqrt (x * y) * Real.sqrt z) :=
by
  sorry

end am_gm_inequality_l77_7720


namespace negation_of_sum_of_squares_l77_7700

variables (a b : ℝ)

theorem negation_of_sum_of_squares:
  ¬(a^2 + b^2 = 0) → (a ≠ 0 ∨ b ≠ 0) := 
by
  sorry

end negation_of_sum_of_squares_l77_7700


namespace question1_question2_question3_l77_7749

-- Define the scores and relevant statistics for seventh and eighth grades
def seventh_grade_scores : List ℕ := [96, 86, 96, 86, 99, 96, 90, 100, 89, 82]
def eighth_grade_C_scores : List ℕ := [94, 90, 92]
def total_eighth_grade_students : ℕ := 800

def a := 40
def b := 93
def c := 96

-- Define given statistics from the table
def seventh_grade_mean := 92
def seventh_grade_variance := 34.6
def eighth_grade_mean := 91
def eighth_grade_median := 93
def eighth_grade_mode := 100
def eighth_grade_variance := 50.4

-- Proof for question 1
theorem question1 : (a = 40) ∧ (b = 93) ∧ (c = 96) :=
by sorry

-- Proof for question 2 (stability comparison)
theorem question2 : seventh_grade_variance < eighth_grade_variance :=
by sorry

-- Proof for question 3 (estimating number of excellent students)
theorem question3 : (7 / 10 : ℝ) * total_eighth_grade_students = 560 :=
by sorry

end question1_question2_question3_l77_7749


namespace prime_n_if_power_of_prime_l77_7761

theorem prime_n_if_power_of_prime (n : ℕ) (h1 : n ≥ 2) (b : ℕ) (h2 : b > 0) (p : ℕ) (k : ℕ) 
  (hk : k > 0) (hb : (b^n - 1) / (b - 1) = p^k) : Nat.Prime n :=
sorry

end prime_n_if_power_of_prime_l77_7761


namespace animal_products_sampled_l77_7746

theorem animal_products_sampled
  (grains : ℕ)
  (oils : ℕ)
  (animal_products : ℕ)
  (fruits_vegetables : ℕ)
  (total_sample : ℕ)
  (total_food_types : grains + oils + animal_products + fruits_vegetables = 100)
  (sample_size : total_sample = 20)
  : (animal_products * total_sample / 100) = 6 := by
  sorry

end animal_products_sampled_l77_7746


namespace circle_radius_l77_7734

theorem circle_radius {r : ℤ} (center: ℝ × ℝ) (inside_pt: ℝ × ℝ) (outside_pt: ℝ × ℝ)
  (h_center: center = (2, 1))
  (h_inside: dist center inside_pt < r)
  (h_outside: dist center outside_pt > r)
  (h_inside_pt: inside_pt = (-2, 1))
  (h_outside_pt: outside_pt = (2, -5))
  (h_integer: r > 0) :
  r = 5 :=
by
  sorry

end circle_radius_l77_7734


namespace min_disks_required_for_files_l77_7753

theorem min_disks_required_for_files :
  ∀ (number_of_files : ℕ)
    (files_0_9MB : ℕ)
    (files_0_6MB : ℕ)
    (disk_capacity_MB : ℝ)
    (file_size_0_9MB : ℝ)
    (file_size_0_6MB : ℝ)
    (file_size_0_45MB : ℝ),
  number_of_files = 40 →
  files_0_9MB = 5 →
  files_0_6MB = 15 →
  disk_capacity_MB = 1.44 →
  file_size_0_9MB = 0.9 →
  file_size_0_6MB = 0.6 →
  file_size_0_45MB = 0.45 →
  ∃ (min_disks : ℕ), min_disks = 16 :=
by
  sorry

end min_disks_required_for_files_l77_7753


namespace red_peaches_each_basket_l77_7788

variable (TotalGreenPeachesInABasket : Nat) (TotalPeachesInABasket : Nat)

theorem red_peaches_each_basket (h1 : TotalPeachesInABasket = 10) (h2 : TotalGreenPeachesInABasket = 3) :
  (TotalPeachesInABasket - TotalGreenPeachesInABasket) = 7 := by
  sorry

end red_peaches_each_basket_l77_7788


namespace k1_k2_ratio_l77_7775

theorem k1_k2_ratio (a b k k1 k2 : ℝ)
  (h1 : a^2 * k - (k - 1) * a + 5 = 0)
  (h2 : b^2 * k - (k - 1) * b + 5 = 0)
  (h3 : (a / b) + (b / a) = 4/5)
  (h4 : k1^2 - 16 * k1 + 1 = 0)
  (h5 : k2^2 - 16 * k2 + 1 = 0) :
  (k1 / k2) + (k2 / k1) = 254 := by
  sorry

end k1_k2_ratio_l77_7775


namespace range_of_a_for_inequality_solutions_to_equation_l77_7733

noncomputable def f (x a : ℝ) := x^2 + 2 * a * x + 1
noncomputable def f_prime (x a : ℝ) := 2 * x + 2 * a

theorem range_of_a_for_inequality :
  (∀ x, -2 ≤ x ∧ x ≤ -1 → f x a ≤ f_prime x a) → a ≥ 3 / 2 :=
sorry

theorem solutions_to_equation (a : ℝ) (x : ℝ) :
  f x a = |f_prime x a| ↔ 
  (if a < -1 then x = -1 ∨ x = 1 - 2 * a 
  else if -1 ≤ a ∧ a ≤ 1 then x = 1 ∨ x = -1 ∨ x = 1 - 2 * a ∨ x = -(1 + 2 * a)
  else x = 1 ∨ x = -(1 + 2 * a)) :=
sorry

end range_of_a_for_inequality_solutions_to_equation_l77_7733


namespace f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l77_7732

open Real

noncomputable def f : ℝ → ℝ :=
sorry

axiom func_prop : ∀ x y : ℝ, f (x + y) = f x + f y - 1
axiom pos_x_gt_1 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_1 : f 1 = 2

-- Prove that f(0) = 1
theorem f_0_eq_1 : f 0 = 1 :=
sorry

-- Prove that f(-1) ≠ 1 (and direct derivation showing f(-1) = 0)
theorem f_neg_1_ne_1 : f (-1) ≠ 1 ∧ f (-1) = 0 :=
sorry

-- Prove that f(x) is increasing
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ > f x₁ :=
sorry

-- Prove minimum value of f on [-3, 3] is -2
theorem min_f_neg3_3 : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -2 :=
sorry

end f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l77_7732


namespace original_selling_price_l77_7776

theorem original_selling_price:
  ∀ (P : ℝ), (1.17 * P - 1.10 * P = 56) → (P > 0) → 1.10 * P = 880 :=
by
  intro P h₁ h₂
  sorry

end original_selling_price_l77_7776


namespace vector_condition_l77_7738

open Real

def acute_angle (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.1 + a.2 * b.2) > 0

def not_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 ≠ 0

theorem vector_condition (x : ℝ) :
  acute_angle (2, x + 1) (x + 2, 6) ∧ not_collinear (2, x + 1) (x + 2, 6) ↔ x > -5/4 ∧ x ≠ 2 :=
by
  sorry

end vector_condition_l77_7738


namespace edge_length_of_cube_l77_7748

noncomputable def cost_per_quart : ℝ := 3.20
noncomputable def coverage_per_quart : ℕ := 120
noncomputable def total_cost : ℝ := 16
noncomputable def total_coverage : ℕ := 600 -- From 5 quarts * 120 square feet per quart
noncomputable def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)

theorem edge_length_of_cube :
  (∃ edge_length : ℝ, surface_area edge_length = total_coverage) → 
  ∃ edge_length : ℝ, edge_length = 10 :=
by
  sorry

end edge_length_of_cube_l77_7748


namespace brick_wall_l77_7726

theorem brick_wall (x : ℕ) 
  (h1 : x / 9 * 9 = x)
  (h2 : x / 10 * 10 = x)
  (h3 : 5 * (x / 9 + x / 10 - 10) = x) :
  x = 900 := 
sorry

end brick_wall_l77_7726


namespace necessary_but_not_sufficient_condition_l77_7707

theorem necessary_but_not_sufficient_condition :
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬ (x > 2)) :=
by {
  sorry
}

end necessary_but_not_sufficient_condition_l77_7707


namespace range_of_a_l77_7793

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * x^2 + 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x > (2 / 3), (deriv (f a)) x > 0) → a > -(1 / 9) :=
by
  sorry

end range_of_a_l77_7793


namespace partial_fraction_decomposition_l77_7713

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ), 
  (∀ x : ℝ, x^4 - 3 * x^3 - 7 * x^2 + 15 * x - 10 ≠ 0 →
    (x^2 - 23) /
    (x^4 - 3 * x^3 - 7 * x^2 + 15 * x - 10) = 
    A / (x - 1) + B / (x + 2) + C / (x - 2)) →
  (A = 44 / 21 ∧ B = -5 / 2 ∧ C = -5 / 6 → A * B * C = 275 / 63)
  := by
  intros A B C h₁ h₂
  sorry

end partial_fraction_decomposition_l77_7713


namespace diagonal_rectangle_l77_7757

theorem diagonal_rectangle (l w : ℝ) (hl : l = 20 * Real.sqrt 5) (hw : w = 10 * Real.sqrt 3) :
    Real.sqrt (l^2 + w^2) = 10 * Real.sqrt 23 :=
by
  sorry

end diagonal_rectangle_l77_7757


namespace jigsaw_puzzle_completion_l77_7774

theorem jigsaw_puzzle_completion (p : ℝ) :
  let total_pieces := 1000
  let pieces_first_day := total_pieces * 0.10
  let remaining_after_first_day := total_pieces - pieces_first_day

  let pieces_second_day := remaining_after_first_day * (p / 100)
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day

  let pieces_third_day := remaining_after_second_day * 0.30
  let remaining_after_third_day := remaining_after_second_day - pieces_third_day

  remaining_after_third_day = 504 ↔ p = 20 := 
by {
    sorry
}

end jigsaw_puzzle_completion_l77_7774


namespace vertex_property_l77_7768

theorem vertex_property (a b c m k : ℝ) (h : a ≠ 0)
  (vertex_eq : k = a * m^2 + b * m + c)
  (point_eq : m = a * k^2 + b * k + c) : a * (m - k) > 0 :=
sorry

end vertex_property_l77_7768


namespace inverse_proportion_quadrants_l77_7708

theorem inverse_proportion_quadrants (k b : ℝ) (h1 : b > 0) (h2 : k < 0) :
  ∀ x : ℝ, (x > 0 → (y = kb / x) → y < 0) ∧ (x < 0 → (y = kb / x) → y > 0) :=
by
  sorry

end inverse_proportion_quadrants_l77_7708


namespace candies_bought_friday_l77_7789

-- Definitions based on the given conditions
def candies_bought_tuesday : ℕ := 3
def candies_bought_thursday : ℕ := 5
def candies_left (c : ℕ) : Prop := c = 4
def candies_eaten (c : ℕ) : Prop := c = 6

-- Theorem to prove the number of candies bought on Friday
theorem candies_bought_friday (c_left c_eaten : ℕ) (h_left : candies_left c_left) (h_eaten : candies_eaten c_eaten) : 
  (10 - (candies_bought_tuesday + candies_bought_thursday) = 2) :=
  by
    sorry

end candies_bought_friday_l77_7789


namespace bricks_needed_for_wall_l77_7719

noncomputable def brick_volume (length : ℝ) (height : ℝ) (thickness : ℝ) : ℝ :=
  length * height * thickness

noncomputable def wall_volume (length : ℝ) (height : ℝ) (average_thickness : ℝ) : ℝ :=
  length * height * average_thickness

noncomputable def number_of_bricks (wall_vol : ℝ) (brick_vol : ℝ) : ℝ :=
  wall_vol / brick_vol

theorem bricks_needed_for_wall : 
  let length_wall := 800
  let height_wall := 660
  let avg_thickness_wall := (25 + 22.5) / 2 -- in cm
  let length_brick := 25
  let height_brick := 11.25
  let thickness_brick := 6
  let mortar_thickness := 1

  let adjusted_length_brick := length_brick + mortar_thickness
  let adjusted_height_brick := height_brick + mortar_thickness

  let volume_wall := wall_volume length_wall height_wall avg_thickness_wall
  let volume_brick_with_mortar := brick_volume adjusted_length_brick adjusted_height_brick thickness_brick

  number_of_bricks volume_wall volume_brick_with_mortar = 6565 :=
by
  sorry

end bricks_needed_for_wall_l77_7719


namespace parallelogram_properties_l77_7750

variable {b h : ℕ}

theorem parallelogram_properties
  (hb : b = 20)
  (hh : h = 4) :
  (b * h = 80) ∧ ((b^2 + h^2) = 416) :=
by
  sorry

end parallelogram_properties_l77_7750


namespace stamp_problem_solution_l77_7718

theorem stamp_problem_solution : ∃ n : ℕ, n > 1 ∧ (∀ m : ℕ, m ≥ 2 * n + 2 → ∃ a b : ℕ, m = n * a + (n + 2) * b) ∧ ∀ x : ℕ, 1 < x ∧ (∀ m : ℕ, m ≥ 2 * x + 2 → ∃ a b : ℕ, m = x * a + (x + 2) * b) → x ≥ 3 :=
by
  sorry

end stamp_problem_solution_l77_7718


namespace ratio_of_80_pencils_l77_7751

theorem ratio_of_80_pencils (C S : ℝ)
  (CP : ℝ := 80 * C)
  (L : ℝ := 30 * S)
  (SP : ℝ := 80 * S)
  (h : CP = SP + L) :
  CP / SP = 11 / 8 :=
by
  -- Start the proof
  sorry

end ratio_of_80_pencils_l77_7751


namespace min_value_expression_l77_7791

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) ≥ 18 :=
by
  sorry

end min_value_expression_l77_7791


namespace emails_in_afternoon_l77_7790

variable (e_m e_t e_a : Nat)
variable (h1 : e_m = 3)
variable (h2 : e_t = 8)

theorem emails_in_afternoon : e_a = 5 :=
by
  -- (Proof steps would go here)
  sorry

end emails_in_afternoon_l77_7790


namespace total_students_surveyed_l77_7758

variable (F E S FE FS ES FES N T : ℕ)

def only_one_language := 230
def exactly_two_languages := 190
def all_three_languages := 40
def no_language := 60

-- Summing up all categories
def total_students := only_one_language + exactly_two_languages + all_three_languages + no_language

theorem total_students_surveyed (h1 : F + E + S = only_one_language) 
    (h2 : FE + FS + ES = exactly_two_languages) 
    (h3 : FES = all_three_languages) 
    (h4 : N = no_language) 
    (h5 : T = F + E + S + FE + FS + ES + FES + N) : 
    T = total_students :=
by
  rw [total_students, only_one_language, exactly_two_languages, all_three_languages, no_language]
  sorry

end total_students_surveyed_l77_7758


namespace bill_buys_125_bouquets_to_make_1000_l77_7728

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end bill_buys_125_bouquets_to_make_1000_l77_7728


namespace chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l77_7763

-- Part a: Prove that with 40 chips, exactly one chip cannot remain after both players have made two moves.
theorem chips_removal_even_initial_40 
  (initial_chips : Nat)
  (num_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 40 → 
  num_moves = 4 → 
  remaining_chips = 1 → 
  False :=
by
  sorry

-- Part b: Prove that with 1000 chips, the minimum number of moves to reduce to one chip is 8.
theorem chips_removal_minimum_moves_1000
  (initial_chips : Nat)
  (min_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 1000 → 
  remaining_chips = 1 → 
  min_moves = 8 :=
by
  sorry

end chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l77_7763


namespace least_value_of_x_l77_7735

theorem least_value_of_x (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) (h3 : x = 11 * p * 2) : x = 44 := 
by
  sorry

end least_value_of_x_l77_7735


namespace composite_numbers_equal_l77_7730

-- Define composite natural number
def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

-- Define principal divisors
def principal_divisors (n : ℕ) (principal1 principal2 : ℕ) : Prop :=
  is_composite n ∧ 
  (1 < principal1 ∧ principal1 < n) ∧ 
  (1 < principal2 ∧ principal2 < n) ∧
  principal1 * principal2 = n

-- Problem statement to prove
theorem composite_numbers_equal (a b p1 p2 : ℕ) :
  is_composite a → is_composite b →
  principal_divisors a p1 p2 → principal_divisors b p1 p2 →
  a = b :=
by
  sorry

end composite_numbers_equal_l77_7730


namespace nine_a_plus_a_plus_nine_l77_7780

theorem nine_a_plus_a_plus_nine (A : Nat) (hA : 0 < A) : 
  10 * A + 9 = 9 * A + (A + 9) := 
by 
  sorry

end nine_a_plus_a_plus_nine_l77_7780


namespace product_expansion_l77_7714

theorem product_expansion (x : ℝ) : 2 * (x + 3) * (x + 4) = 2 * x^2 + 14 * x + 24 := 
by
  sorry

end product_expansion_l77_7714


namespace price_reduction_l77_7786

theorem price_reduction (p0 p1 p2 : ℝ) (H0 : p0 = 1) (H1 : p1 = 1.25 * p0) (H2 : p2 = 1.1 * p0) :
  ∃ x : ℝ, p2 = p1 * (1 - x / 100) ∧ x = 12 :=
  sorry

end price_reduction_l77_7786


namespace coefficients_of_quadratic_function_l77_7784

-- Define the quadratic function.
def quadratic_function (x : ℝ) : ℝ :=
  2 * (x - 3) ^ 2 + 2

-- Define the expected expanded form.
def expanded_form (x : ℝ) : ℝ :=
  2 * x ^ 2 - 12 * x + 20

-- State the proof problem.
theorem coefficients_of_quadratic_function :
  ∀ (x : ℝ), quadratic_function x = expanded_form x := by
  sorry

end coefficients_of_quadratic_function_l77_7784


namespace vacation_cost_l77_7710

theorem vacation_cost (C : ℝ) (h : C / 6 - C / 8 = 120) : C = 2880 :=
by
  sorry

end vacation_cost_l77_7710


namespace trig_identity_example_l77_7785

theorem trig_identity_example :
  (Real.sin (43 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - Real.sin (13 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_example_l77_7785


namespace sin_C_value_l77_7709

theorem sin_C_value (A B C : ℝ) (a b c : ℝ) 
  (h_a : a = 1) 
  (h_b : b = 1/2) 
  (h_cos_A : Real.cos A = (Real.sqrt 3) / 2) 
  (h_angles : A + B + C = Real.pi) 
  (h_sides : Real.sin A / a = Real.sin B / b) :
  Real.sin C = (Real.sqrt 15 + Real.sqrt 3) / 8 :=
by 
  sorry

end sin_C_value_l77_7709


namespace coefficient_x6_in_expansion_l77_7781

theorem coefficient_x6_in_expansion :
  (∃ c : ℕ, c = 81648 ∧ (3 : ℝ) ^ 6 * c * 2 ^ 2  = c * (3 : ℝ) ^ 6 * 4) :=
sorry

end coefficient_x6_in_expansion_l77_7781


namespace Jenny_wants_to_read_three_books_l77_7723

noncomputable def books : Nat := 3

-- Definitions based on provided conditions
def reading_speed : Nat := 100 -- words per hour
def book1_words : Nat := 200 
def book2_words : Nat := 400
def book3_words : Nat := 300
def daily_reading_minutes : Nat := 54 
def days : Nat := 10

-- Derived definitions for the proof
def total_words : Nat := book1_words + book2_words + book3_words
def total_hours_needed : ℚ := total_words / reading_speed
def daily_reading_hours : ℚ := daily_reading_minutes / 60
def total_reading_hours : ℚ := daily_reading_hours * days

theorem Jenny_wants_to_read_three_books :
  total_reading_hours = total_hours_needed → books = 3 :=
by
  -- Proof goes here
  sorry

end Jenny_wants_to_read_three_books_l77_7723


namespace joan_paid_amount_l77_7725

theorem joan_paid_amount (J K : ℕ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end joan_paid_amount_l77_7725


namespace decrease_angle_equilateral_l77_7794

theorem decrease_angle_equilateral (D E F : ℝ) (h : D = 60) (h_equilateral : D = E ∧ E = F) (h_decrease : D' = D - 20) :
  ∃ max_angle : ℝ, max_angle = 70 :=
by
  sorry

end decrease_angle_equilateral_l77_7794


namespace chickens_and_rabbits_l77_7760

-- Let x be the number of chickens and y be the number of rabbits
variables (x y : ℕ)

-- Conditions: There are 35 heads and 94 feet in total
def heads_eq : Prop := x + y = 35
def feet_eq : Prop := 2 * x + 4 * y = 94

-- Proof statement (no proof is required, so we use sorry)
theorem chickens_and_rabbits :
  (heads_eq x y) ∧ (feet_eq x y) ↔ (x + y = 35 ∧ 2 * x + 4 * y = 94) :=
by
  sorry

end chickens_and_rabbits_l77_7760


namespace find_actual_balance_l77_7762

-- Define the given conditions
def current_balance : ℝ := 90000
def rate : ℝ := 0.10

-- Define the target
def actual_balance_before_deduction (X : ℝ) : Prop :=
  (X * (1 - rate) = current_balance)

-- Statement of the theorem
theorem find_actual_balance : ∃ X : ℝ, actual_balance_before_deduction X :=
  sorry

end find_actual_balance_l77_7762


namespace bird_probability_l77_7712

def uniform_probability (segment_count bird_count : ℕ) : ℚ :=
  if bird_count = segment_count then
    1 / (segment_count ^ bird_count)
  else
    0

theorem bird_probability :
  let wire_length := 10
  let birds := 10
  let distance := 1
  let segments := wire_length / distance
  segments = birds ->
  uniform_probability segments birds = 1 / (10 ^ 10) := by
  intros
  sorry

end bird_probability_l77_7712


namespace nelly_earns_per_night_l77_7797

/-- 
  Nelly wants to buy pizza for herself and her 14 friends. Each pizza costs $12 and can feed 3 
  people. Nelly has to babysit for 15 nights to afford the pizza. We need to prove that Nelly earns 
  $4 per night babysitting.
--/
theorem nelly_earns_per_night 
  (total_people : ℕ) (people_per_pizza : ℕ) 
  (cost_per_pizza : ℕ) (total_nights : ℕ) (total_cost : ℕ) 
  (total_pizzas : ℕ) (cost_per_night : ℕ)
  (h1 : total_people = 15)
  (h2 : people_per_pizza = 3)
  (h3 : cost_per_pizza = 12)
  (h4 : total_nights = 15)
  (h5 : total_pizzas = total_people / people_per_pizza)
  (h6 : total_cost = total_pizzas * cost_per_pizza)
  (h7 : cost_per_night = total_cost / total_nights) :
  cost_per_night = 4 := sorry

end nelly_earns_per_night_l77_7797


namespace find_a_l77_7796

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a = 0 ↔ 3 * x^4 - 48 = 0) → a = 4 :=
  by
    intros h
    sorry

end find_a_l77_7796


namespace trader_profit_l77_7772

noncomputable def profit_percentage (P : ℝ) : ℝ :=
  let purchased_price := 0.72 * P
  let market_increase := 1.05 * purchased_price
  let expenses := 0.08 * market_increase
  let net_price := market_increase - expenses
  let first_sale_price := 1.50 * net_price
  let final_sale_price := 1.25 * first_sale_price
  let profit := final_sale_price - P
  (profit / P) * 100

theorem trader_profit
  (P : ℝ) 
  (hP : 0 < P) :
  profit_percentage P = 30.41 :=
by
  sorry

end trader_profit_l77_7772


namespace square_of_1005_l77_7798

theorem square_of_1005 : (1005 : ℕ)^2 = 1010025 := 
  sorry

end square_of_1005_l77_7798


namespace B_subset_A_implies_range_m_l77_7706

variable {x m : ℝ}

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | -m < x ∧ x < m}

theorem B_subset_A_implies_range_m (m : ℝ) (h : B m ⊆ A) : m ≤ 1 := by
  sorry

end B_subset_A_implies_range_m_l77_7706


namespace square_side_length_properties_l77_7787

theorem square_side_length_properties (a: ℝ) (h: a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by
  sorry

end square_side_length_properties_l77_7787


namespace domain_of_f_l77_7724

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 1 - x > 0
def condition2 (x : ℝ) : Prop := 3 * x + 1 > 0

-- Define the domain interval
def domain (x : ℝ) : Prop := -1 / 3 < x ∧ x < 1

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 / (Real.sqrt (1 - x)) + Real.log (3 * x + 1)

-- The main theorem to prove
theorem domain_of_f : 
  (∀ x : ℝ, condition1 x ∧ condition2 x ↔ domain x) :=
by {
  sorry
}

end domain_of_f_l77_7724


namespace constant_term_in_binomial_expansion_l77_7711

theorem constant_term_in_binomial_expansion 
  (a b : ℕ) (n : ℕ)
  (sum_of_coefficients : (1 + 1)^n = 4)
  (A B : ℕ)
  (sum_A_B : A + B = 72) 
  (A_value : A = 4) :
  (b^2 = 9) :=
by sorry

end constant_term_in_binomial_expansion_l77_7711


namespace count_japanese_stamps_l77_7729

theorem count_japanese_stamps (total_stamps : ℕ) (perc_chinese perc_us : ℕ) (h1 : total_stamps = 100) 
  (h2 : perc_chinese = 35) (h3 : perc_us = 20): 
  total_stamps - ((perc_chinese * total_stamps / 100) + (perc_us * total_stamps / 100)) = 45 :=
by
  sorry

end count_japanese_stamps_l77_7729


namespace circle_center_l77_7755

theorem circle_center (x y : ℝ) (h : x^2 - 4 * x + y^2 - 6 * y - 12 = 0) : (x, y) = (2, 3) :=
sorry

end circle_center_l77_7755


namespace range_of_m_l77_7795

theorem range_of_m (m : ℝ) (x : ℝ) : (∀ x, (1 - m) * x = 2 - 3 * x → x > 0) ↔ m < 4 :=
by
  sorry

end range_of_m_l77_7795


namespace factor_difference_of_squares_l77_7759
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l77_7759


namespace combined_points_kjm_l77_7743

theorem combined_points_kjm {P B K J M H C E: ℕ} 
  (total_points : P + B + K + J + M = 81)
  (paige_points : P = 21)
  (brian_points : B = 20)
  (karen_jennifer_michael_sum : K + J + M = 40)
  (karen_scores : ∀ p, K = 2 * p + 5 * (H - p))
  (jennifer_scores : ∀ p, J = 2 * p + 5 * (C - p))
  (michael_scores : ∀ p, M = 2 * p + 5 * (E - p)) :
  K + J + M = 40 :=
by sorry

end combined_points_kjm_l77_7743


namespace arithmetic_sequence_sum_l77_7773

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h1 : a 1 + a 3 + a 5 = 9) (h2 : a 2 + a 4 + a 6 = 15) : a 3 + a 4 = 8 := 
by 
  sorry

end arithmetic_sequence_sum_l77_7773


namespace year_when_mother_age_is_twice_jack_age_l77_7704

noncomputable def jack_age_2010 := 12
noncomputable def mother_age_2010 := 3 * jack_age_2010

theorem year_when_mother_age_is_twice_jack_age :
  ∃ x : ℕ, mother_age_2010 + x = 2 * (jack_age_2010 + x) ∧ (2010 + x = 2022) :=
by
  sorry

end year_when_mother_age_is_twice_jack_age_l77_7704


namespace area_outside_small_squares_l77_7752

theorem area_outside_small_squares (a b : ℕ) (ha : a = 10) (hb : b = 4) (n : ℕ) (hn: n = 2) :
  a^2 - n * b^2 = 68 :=
by
  rw [ha, hb, hn]
  sorry

end area_outside_small_squares_l77_7752


namespace multiplication_integer_multiple_l77_7771

theorem multiplication_integer_multiple (a b n : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
(h_eq : 10000 * a + b = n * (a * b)) : n = 73 := 
sorry

end multiplication_integer_multiple_l77_7771


namespace y_decreases_as_x_increases_l77_7747

-- Define the function y = 7 - x
def my_function (x : ℝ) : ℝ := 7 - x

-- Prove that y decreases as x increases
theorem y_decreases_as_x_increases : ∀ x1 x2 : ℝ, x1 < x2 → my_function x1 > my_function x2 := by
  intro x1 x2 h
  unfold my_function
  sorry

end y_decreases_as_x_increases_l77_7747


namespace product_of_ab_l77_7766

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_7766


namespace exists_positive_integer_m_l77_7765

noncomputable def d (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r - 1)
noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d
noncomputable def g_n (n : ℕ) (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r ^ (n - 1))

theorem exists_positive_integer_m (a1 g1 : ℝ) (r : ℝ) (h0 : g1 ≠ 0) (h1 : a1 = g1) (h2 : a2 = g2)
(h3 : a_n 10 a1 (d g1 r) = g_n 3 g1 r) :
  ∀ (p : ℕ), ∃ (m : ℕ), g_n p g1 r = a_n m a1 (d g1 r) := by
  sorry

end exists_positive_integer_m_l77_7765


namespace find_other_cat_weight_l77_7727

variable (cat1 cat2 dog : ℕ)

def weight_of_other_cat (cat1 cat2 dog : ℕ) : Prop :=
  cat1 = 7 ∧
  dog = 34 ∧
  dog = 2 * (cat1 + cat2) ∧
  cat2 = 10

theorem find_other_cat_weight (cat1 : ℕ) (cat2 : ℕ) (dog : ℕ) :
  weight_of_other_cat cat1 cat2 dog := by
  sorry

end find_other_cat_weight_l77_7727


namespace professor_D_error_l77_7783

noncomputable def polynomial_calculation_error (n : ℕ) : Prop :=
  ∃ (f : ℝ → ℝ), (∀ i : ℕ, i ≤ n+1 → f i = 2^i) ∧ f (n+2) ≠ 2^(n+2) - n - 3

theorem professor_D_error (n : ℕ) : polynomial_calculation_error n :=
  sorry

end professor_D_error_l77_7783


namespace hyperbola_center_is_correct_l77_7782

theorem hyperbola_center_is_correct :
  ∃ h k : ℝ, (∀ x y : ℝ, ((4 * y + 8)^2 / 16^2) - ((5 * x - 15)^2 / 9^2) = 1 → x - h = 0 ∧ y + k = 0) ∧ h = 3 ∧ k = -2 :=
sorry

end hyperbola_center_is_correct_l77_7782


namespace graphs_intersect_at_one_point_l77_7767

theorem graphs_intersect_at_one_point (m : ℝ) (e := Real.exp 1) :
  (∀ f g : ℝ → ℝ,
    (∀ x, f x = x + Real.log x - 2 / e) ∧ (∀ x, g x = m / x) →
    ∃! x, f x = g x) ↔ (m ≥ 0 ∨ m = - (e + 1) / (e ^ 2)) :=
by sorry

end graphs_intersect_at_one_point_l77_7767


namespace exists_a_bc_l77_7739

-- Definitions & Conditions
def satisfies_conditions (a b c : ℤ) : Prop :=
  - (b + c) - 10 = a ∧ (b + 10) * (c + 10) = 1

-- Theorem Statement
theorem exists_a_bc : ∃ (a b c : ℤ), satisfies_conditions a b c := by
  -- Substitute the correct proof below
  sorry

end exists_a_bc_l77_7739


namespace Brandon_can_still_apply_l77_7715

-- Definitions based on the given conditions
def total_businesses : ℕ := 72
def fired_businesses : ℕ := total_businesses / 2
def quit_businesses : ℕ := total_businesses / 3
def businesses_restricted : ℕ := fired_businesses + quit_businesses

-- The final proof statement
theorem Brandon_can_still_apply : total_businesses - businesses_restricted = 12 :=
by
  -- Note: Proof is omitted; replace sorry with detailed proof in practice.
  sorry

end Brandon_can_still_apply_l77_7715


namespace find_y_l77_7741

theorem find_y (x y : ℝ) (h1 : x^2 - 4 * x = y + 5) (h2 : x = 7) : y = 16 := by
  sorry

end find_y_l77_7741


namespace no_nat_n_for_9_pow_n_minus_7_is_product_l77_7717

theorem no_nat_n_for_9_pow_n_minus_7_is_product :
  ¬ ∃ (n k : ℕ), 9 ^ n - 7 = k * (k + 1) :=
by
  sorry

end no_nat_n_for_9_pow_n_minus_7_is_product_l77_7717


namespace total_short_trees_after_planting_l77_7701

def current_short_oak_trees := 3
def current_short_pine_trees := 4
def current_short_maple_trees := 5
def new_short_oak_trees := 9
def new_short_pine_trees := 6
def new_short_maple_trees := 4

theorem total_short_trees_after_planting :
  current_short_oak_trees + current_short_pine_trees + current_short_maple_trees +
  new_short_oak_trees + new_short_pine_trees + new_short_maple_trees = 31 := by
  sorry

end total_short_trees_after_planting_l77_7701


namespace amount_of_bill_is_1575_l77_7754

noncomputable def time_in_years := (9 : ℝ) / 12

noncomputable def true_discount := 189
noncomputable def rate := 16

noncomputable def face_value (TD : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (TD * 100) / (R * T)

theorem amount_of_bill_is_1575 :
  face_value true_discount rate time_in_years = 1575 := by
  sorry

end amount_of_bill_is_1575_l77_7754


namespace sculpture_cost_in_cny_l77_7779

-- Define the equivalence rates
def usd_to_nad : ℝ := 8
def usd_to_cny : ℝ := 8

-- Define the cost of the sculpture in Namibian dollars
def sculpture_cost_nad : ℝ := 160

-- Theorem: Given the conversion rates, the sculpture cost in Chinese yuan is 160
theorem sculpture_cost_in_cny : (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 160 :=
by sorry

end sculpture_cost_in_cny_l77_7779


namespace cos_relation_l77_7721

theorem cos_relation 
  (a b c A B C : ℝ)
  (h1 : a = b * Real.cos C + c * Real.cos B)
  (h2 : b = c * Real.cos A + a * Real.cos C)
  (h3 : c = a * Real.cos B + b * Real.cos A)
  (h_abc_nonzero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 + 2 * Real.cos A * Real.cos B * Real.cos C = 1 :=
sorry

end cos_relation_l77_7721


namespace circle_equation_tangent_to_line_l77_7731

theorem circle_equation_tangent_to_line
  (h k : ℝ) (A B C : ℝ)
  (hxk : h = 2) (hyk : k = -1) 
  (hA : A = 3) (hB : B = -4) (hC : C = 5)
  (r_squared : ℝ := (|A * h + B * k + C| / Real.sqrt (A^2 + B^2))^2)
  (h_radius : r_squared = 9) :
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r_squared := 
by
  sorry

end circle_equation_tangent_to_line_l77_7731


namespace intersection_points_l77_7716

theorem intersection_points (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  (∃ x1 x2, 0 ≤ x1 ∧ x1 ≤ 2 * Real.pi ∧ 
   0 ≤ x2 ∧ x2 ≤ 2 * Real.pi ∧ 
   x1 ≠ x2 ∧ 
   1 + Real.sin x1 = 3 / 2 ∧ 
   1 + Real.sin x2 = 3 / 2 ) ∧ 
  (∀ x, (0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 1 + Real.sin x = 3 / 2) → 
   (x = x1 ∨ x = x2)) :=
sorry

end intersection_points_l77_7716


namespace original_savings_l77_7737

theorem original_savings (tv_cost : ℝ) (furniture_fraction : ℝ) (total_fraction : ℝ) (original_savings : ℝ) :
  tv_cost = 300 → furniture_fraction = 3 / 4 → total_fraction = 1 → 
  (total_fraction - furniture_fraction) * original_savings = tv_cost →
  original_savings = 1200 :=
by 
  intros htv hfurniture htotal hsavings_eq
  sorry

end original_savings_l77_7737


namespace terminative_decimal_of_45_div_72_l77_7702

theorem terminative_decimal_of_45_div_72 :
  (45 / 72 : ℚ) = 0.625 :=
sorry

end terminative_decimal_of_45_div_72_l77_7702


namespace train_time_to_pass_platform_l77_7764

-- Definitions as per the conditions
def length_of_train : ℕ := 720 -- Length of train in meters
def speed_of_train_kmh : ℕ := 72 -- Speed of train in km/hr
def length_of_platform : ℕ := 280 -- Length of platform in meters

-- Conversion factor and utility functions
def kmh_to_ms (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

def time_to_pass (distance speed_ms : ℕ) : ℕ :=
  distance / speed_ms

-- Main statement to be proven
theorem train_time_to_pass_platform :
  time_to_pass (total_distance length_of_train length_of_platform) (kmh_to_ms speed_of_train_kmh) = 50 :=
by
  sorry

end train_time_to_pass_platform_l77_7764


namespace range_of_m_l77_7742

theorem range_of_m (f : ℝ → ℝ) 
  (Hmono : ∀ x y, -2 ≤ x → x ≤ 2 → -2 ≤ y → y ≤ 2 → x ≤ y → f x ≤ f y)
  (Hineq : ∀ m, f (Real.log m / Real.log 2) < f (Real.log (m + 2) / Real.log 4))
  : ∀ m, (1 / 4 : ℝ) ≤ m ∧ m < 2 :=
sorry

end range_of_m_l77_7742


namespace increased_consumption_5_percent_l77_7740

theorem increased_consumption_5_percent (T C : ℕ) (h1 : ¬ (T = 0)) (h2 : ¬ (C = 0)) :
  (0.80 * (1 + x/100) = 0.84) → (x = 5) :=
by
  sorry

end increased_consumption_5_percent_l77_7740


namespace percentage_problem_l77_7777

theorem percentage_problem (P : ℕ) : (P / 100 * 400 = 20 / 100 * 700) → P = 35 :=
by
  intro h
  sorry

end percentage_problem_l77_7777


namespace floor_plus_x_eq_17_over_4_l77_7744

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l77_7744


namespace min_a_n_l77_7736

def a_n (n : ℕ) : ℤ := n^2 - 8 * n + 5

theorem min_a_n : ∃ n : ℕ, ∀ m : ℕ, a_n n ≤ a_n m ∧ a_n n = -11 :=
by
  sorry

end min_a_n_l77_7736


namespace rain_probability_tel_aviv_l77_7756

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l77_7756


namespace triangle_area_range_l77_7705

theorem triangle_area_range (x₁ x₂ : ℝ) (h₀ : 0 < x₁) (h₁ : x₁ < 1) (h₂ : 1 < x₂) (h₃ : x₁ * x₂ = 1) :
  0 < (2 / (x₁ + 1 / x₁)) ∧ (2 / (x₁ + 1 / x₁)) < 1 :=
by
  sorry

end triangle_area_range_l77_7705


namespace garrett_bought_peanut_granola_bars_l77_7799

def garrett_granola_bars (t o : ℕ) (h_t : t = 14) (h_o : o = 6) : ℕ :=
  t - o

theorem garrett_bought_peanut_granola_bars : garrett_granola_bars 14 6 rfl rfl = 8 :=
  by
    unfold garrett_granola_bars
    rw [Nat.sub_eq_of_eq_add]
    sorry

end garrett_bought_peanut_granola_bars_l77_7799


namespace exponent_property_l77_7745

variable {a : ℝ} {m n : ℕ}

theorem exponent_property (h1 : a^m = 2) (h2 : a^n = 3) : a^(2*m + n) = 12 :=
sorry

end exponent_property_l77_7745


namespace tom_already_has_4_pounds_of_noodles_l77_7722

-- Define the conditions
def beef : ℕ := 10
def noodle_multiplier : ℕ := 2
def packages : ℕ := 8
def weight_per_package : ℕ := 2

-- Define the total noodles needed
def total_noodles_needed : ℕ := noodle_multiplier * beef

-- Define the total noodles bought
def total_noodles_bought : ℕ := packages * weight_per_package

-- Define the already owned noodles
def already_owned_noodles : ℕ := total_noodles_needed - total_noodles_bought

-- State the theorem to prove
theorem tom_already_has_4_pounds_of_noodles :
  already_owned_noodles = 4 :=
  sorry

end tom_already_has_4_pounds_of_noodles_l77_7722


namespace correct_choice_D_l77_7703

theorem correct_choice_D (a : ℝ) :
  (2 * a ^ 2) ^ 3 = 8 * a ^ 6 ∧ 
  (a ^ 10 * a ^ 2 ≠ a ^ 20) ∧ 
  (a ^ 10 / a ^ 2 ≠ a ^ 5) ∧ 
  ((Real.pi - 3) ^ 0 ≠ 0) :=
by {
  sorry
}

end correct_choice_D_l77_7703
