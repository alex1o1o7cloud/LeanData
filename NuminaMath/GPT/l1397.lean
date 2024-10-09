import Mathlib

namespace Lucas_identity_l1397_139750

def Lucas (L : ℕ → ℤ) (F : ℕ → ℤ) : Prop :=
  ∀ n, L n = F (n + 1) + F (n - 1)

def Fib_identity1 (F : ℕ → ℤ) : Prop :=
  ∀ n, F (2 * n + 1) = F (n + 1) ^ 2 + F n ^ 2

def Fib_identity2 (F : ℕ → ℤ) : Prop :=
  ∀ n, F n ^ 2 = F (n + 1) * F (n - 1) - (-1) ^ n

theorem Lucas_identity {L F : ℕ → ℤ} (hL : Lucas L F) (hF1 : Fib_identity1 F) (hF2 : Fib_identity2 F) :
  ∀ n, L (2 * n) = L n ^ 2 - 2 * (-1) ^ n := 
sorry

end Lucas_identity_l1397_139750


namespace find_y_l1397_139736

theorem find_y (y : ℝ) (h : (y - 8) / (5 - (-3)) = -5 / 4) : y = -2 :=
by sorry

end find_y_l1397_139736


namespace find_C_and_D_l1397_139709

theorem find_C_and_D :
  (∀ x, x^2 - 3 * x - 10 ≠ 0 → (4 * x - 3) / (x^2 - 3 * x - 10) = (17 / 7) / (x - 5) + (11 / 7) / (x + 2)) :=
by
  sorry

end find_C_and_D_l1397_139709


namespace arithmetic_series_product_l1397_139793

theorem arithmetic_series_product (a b c : ℝ) (h1 : a = b - d) (h2 : c = b + d) (h3 : a * b * c = 125) (h4 : 0 < a) (h5 : 0 < b) (h6 : 0 < c) : b ≥ 5 :=
sorry

end arithmetic_series_product_l1397_139793


namespace num_positive_k_for_solution_to_kx_minus_18_eq_3k_l1397_139728

theorem num_positive_k_for_solution_to_kx_minus_18_eq_3k : 
  ∃ (k_vals : Finset ℕ), 
  (∀ k ∈ k_vals, ∃ x : ℤ, k * x - 18 = 3 * k) ∧ 
  k_vals.card = 6 :=
by
  sorry

end num_positive_k_for_solution_to_kx_minus_18_eq_3k_l1397_139728


namespace gcd_lcm_ordering_l1397_139726

theorem gcd_lcm_ordering (a b p q : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_a_gt_b : a > b) 
    (h_p_gcd : p = Nat.gcd a b) (h_q_lcm : q = Nat.lcm a b) : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end gcd_lcm_ordering_l1397_139726


namespace speed_conversion_l1397_139789

theorem speed_conversion (v : ℚ) (h : v = 9/36) : v * 3.6 = 0.9 := by
  sorry

end speed_conversion_l1397_139789


namespace Bobby_candy_l1397_139701

theorem Bobby_candy (initial_candy remaining_candy1 remaining_candy2 : ℕ)
  (H1 : initial_candy = 21)
  (H2 : remaining_candy1 = initial_candy - 5)
  (H3 : remaining_candy2 = remaining_candy1 - 9):
  remaining_candy2 = 7 :=
by
  sorry

end Bobby_candy_l1397_139701


namespace quadratic_roots_l1397_139721

theorem quadratic_roots (a b : ℝ) (h : a^2 - 4*a*b + 5*b^2 - 2*b + 1 = 0) :
  ∃ (p q : ℝ), (∀ (x : ℝ), x^2 - p*x + q = 0 ↔ (x = a ∨ x = b)) ∧
               p = 3 ∧ q = 2 :=
by {
  sorry
}

end quadratic_roots_l1397_139721


namespace initial_walking_rate_proof_l1397_139773

noncomputable def initial_walking_rate (d : ℝ) (v_miss : ℝ) (t_miss : ℝ) (v_early : ℝ) (t_early : ℝ) : ℝ :=
  d / ((d / v_early) + t_early - t_miss)

theorem initial_walking_rate_proof :
  initial_walking_rate 6 5 (7/60) 6 (5/60) = 5 := by
  sorry

end initial_walking_rate_proof_l1397_139773


namespace value_of_a_l1397_139730

theorem value_of_a (a : ℝ) (A : ℝ × ℝ) (h : A = (1, 0)) : (a * A.1 + 3 * A.2 - 2 = 0) → a = 2 :=
by
  intro h1
  rw [h] at h1
  sorry

end value_of_a_l1397_139730


namespace focus_of_parabola_l1397_139700

theorem focus_of_parabola :
  (∃ f : ℝ, ∀ y : ℝ, (x = -1 / 4 * y^2) = (x = (y^2 / 4 + f)) -> f = -1) :=
by
  sorry

end focus_of_parabola_l1397_139700


namespace sum_even_odd_diff_l1397_139772

theorem sum_even_odd_diff (n : ℕ) (h : n = 1500) : 
  let S_odd := n / 2 * (1 + (1 + (n - 1) * 2))
  let S_even := n / 2 * (2 + (2 + (n - 1) * 2))
  (S_even - S_odd) = n :=
by
  sorry

end sum_even_odd_diff_l1397_139772


namespace least_integer_value_l1397_139792

theorem least_integer_value (x : ℤ) : 3 * abs x + 4 < 19 → x = -4 :=
by
  intro h
  sorry

end least_integer_value_l1397_139792


namespace cylinder_radius_inscribed_box_l1397_139763

theorem cylinder_radius_inscribed_box :
  ∀ (x y z r : ℝ),
    4 * (x + y + z) = 160 →
    2 * (x * y + y * z + x * z) = 600 →
    z = 40 - x - y →
    r = (1/2) * Real.sqrt (x^2 + y^2) →
    r = (15 * Real.sqrt 2) / 2 :=
by
  sorry

end cylinder_radius_inscribed_box_l1397_139763


namespace percent_increase_calculation_l1397_139760

variable (x y : ℝ) -- Declare x and y as real numbers representing the original salary and increment

-- The statement that the percent increase z follows from the given conditions
theorem percent_increase_calculation (h : y + x = x + y) : (y / x) * 100 = ((y / x) * 100) := by
  sorry

end percent_increase_calculation_l1397_139760


namespace spherical_to_rectangular_conversion_l1397_139717

/-- Convert a point in spherical coordinates to rectangular coordinates given specific angles and distance -/
theorem spherical_to_rectangular_conversion :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ), 
  ρ = 15 → θ = 225 * (Real.pi / 180) → φ = 45 * (Real.pi / 180) →
  x = ρ * Real.sin φ * Real.cos θ → y = ρ * Real.sin φ * Real.sin θ → z = ρ * Real.cos φ →
  x = -15 / 2 ∧ y = -15 / 2 ∧ z = 15 * Real.sqrt 2 / 2 := by
  sorry

end spherical_to_rectangular_conversion_l1397_139717


namespace and_or_distrib_left_or_and_distrib_right_l1397_139734

theorem and_or_distrib_left (A B C : Prop) : A ∧ (B ∨ C) ↔ (A ∧ B) ∨ (A ∧ C) :=
sorry

theorem or_and_distrib_right (A B C : Prop) : A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C) :=
sorry

end and_or_distrib_left_or_and_distrib_right_l1397_139734


namespace final_selling_price_l1397_139735

-- Define the conditions as constants
def CP := 750
def loss_percentage := 20 / 100
def sales_tax_percentage := 10 / 100

-- Define the final selling price after loss and adding sales tax
theorem final_selling_price 
  (CP : ℝ) 
  (loss_percentage : ℝ)
  (sales_tax_percentage : ℝ) 
  : 750 = CP ∧ 20 / 100 = loss_percentage ∧ 10 / 100 = sales_tax_percentage → 
    (CP - (loss_percentage * CP) + (sales_tax_percentage * CP) = 675) := 
by
  intros
  sorry

end final_selling_price_l1397_139735


namespace M_is_correct_l1397_139781

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x > 2}

def M := {x | x ∈ A ∧ x ∉ B}

theorem M_is_correct : M = {1, 2} := by
  -- Proof needed here
  sorry

end M_is_correct_l1397_139781


namespace max_min_of_f_on_interval_l1397_139766

-- Conditions
def f (x : ℝ) : ℝ := x^3 - 3 * x + 1
def interval : Set ℝ := Set.Icc (-3) 0

-- Problem statement
theorem max_min_of_f_on_interval : 
  ∃ (max min : ℝ), max = 1 ∧ min = -17 ∧ 
  (∀ x ∈ interval, f x ≤ max) ∧ 
  (∀ x ∈ interval, f x ≥ min) := 
sorry

end max_min_of_f_on_interval_l1397_139766


namespace find_number_l1397_139775

theorem find_number
  (a b c : ℕ)
  (h_a1 : a ≤ 3)
  (h_b1 : b ≤ 3)
  (h_c1 : c ≤ 3)
  (h_a2 : a ≠ 3)
  (h_b_condition1 : b ≠ 1 → 2 * a * b < 10)
  (h_b_condition2 : b ≠ 2 → 2 * a * b < 10)
  (h_c3 : c = 3)
  : a = 2 ∧ b = 3 ∧ c = 3 :=
by
  sorry

end find_number_l1397_139775


namespace weight_7_moles_AlI3_l1397_139712

-- Definitions from the conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_I : ℝ := 126.90
def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ := moles * molecular_weight

-- Theorem stating the weight of 7 moles of AlI3
theorem weight_7_moles_AlI3 : 
  weight_of_compound 7 molecular_weight_AlI3 = 2853.76 :=
by
  -- Proof will be added here
  sorry

end weight_7_moles_AlI3_l1397_139712


namespace bacteria_growth_rate_l1397_139718

theorem bacteria_growth_rate (r : ℝ) 
  (h1 : ∀ n : ℕ, n = 22 → ∃ c : ℝ, c * r^n = c) 
  (h2 : ∀ n : ℕ, n = 21 → ∃ c : ℝ, 2 * c * r^n = c) : 
  r = 2 := 
by
  sorry

end bacteria_growth_rate_l1397_139718


namespace production_line_improvement_better_than_financial_investment_l1397_139759

noncomputable def improved_mean_rating (initial_mean : ℝ) := initial_mean + 0.05

noncomputable def combined_mean_rating (mean_unimproved : ℝ) (mean_improved : ℝ) : ℝ :=
  (mean_unimproved * 200 + mean_improved * 200) / 400

noncomputable def combined_variance (variance : ℝ) (combined_mean : ℝ) : ℝ :=
  (2 * variance) - combined_mean ^ 2

noncomputable def increased_returns (grade_a_price : ℝ) (grade_b_price : ℝ) 
  (proportion_upgraded : ℝ) (units_per_day : ℕ) (days_per_year : ℕ) : ℝ :=
  (grade_a_price - grade_b_price) * proportion_upgraded * units_per_day * days_per_year - 200000000

noncomputable def financial_returns (initial_investment : ℝ) (annual_return_rate : ℝ) : ℝ :=
  initial_investment * (1 + annual_return_rate) - initial_investment

theorem production_line_improvement_better_than_financial_investment 
  (initial_mean : ℝ := 9.98) 
  (initial_variance : ℝ := 0.045) 
  (grade_a_price : ℝ := 2000) 
  (grade_b_price : ℝ := 1200) 
  (proportion_upgraded : ℝ := 3 / 8) 
  (units_per_day : ℕ := 200) 
  (days_per_year : ℕ := 365) 
  (initial_investment : ℝ := 200000000) 
  (annual_return_rate : ℝ := 0.082) : 
  combined_mean_rating initial_mean (improved_mean_rating initial_mean) = 10.005 ∧ 
  combined_variance initial_variance (combined_mean_rating initial_mean (improved_mean_rating initial_mean)) = 0.045625 ∧ 
  increased_returns grade_a_price grade_b_price proportion_upgraded units_per_day days_per_year > financial_returns initial_investment annual_return_rate := 
by {
  sorry
}

end production_line_improvement_better_than_financial_investment_l1397_139759


namespace painting_faces_not_sum_to_nine_l1397_139785

def eight_sided_die_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def pairs_that_sum_to_nine : List (ℕ × ℕ) := [(1, 8), (2, 7), (3, 6), (4, 5)]

theorem painting_faces_not_sum_to_nine :
  let total_pairs := (eight_sided_die_numbers.length * (eight_sided_die_numbers.length - 1)) / 2
  let invalid_pairs := pairs_that_sum_to_nine.length
  total_pairs - invalid_pairs = 24 :=
by
  sorry

end painting_faces_not_sum_to_nine_l1397_139785


namespace negation_of_universal_proposition_l1397_139707

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x+1) * exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x+1) * exp x ≤ 1 :=
by sorry

end negation_of_universal_proposition_l1397_139707


namespace tangent_line_at_point_l1397_139791

noncomputable def curve (x : ℝ) : ℝ := Real.exp x + x

theorem tangent_line_at_point :
  (∃ k b : ℝ, (∀ x : ℝ, curve x = k * x + b) ∧ k = 2 ∧ b = 1) :=
by
  sorry

end tangent_line_at_point_l1397_139791


namespace sin_mul_cos_eq_quarter_l1397_139744

open Real

theorem sin_mul_cos_eq_quarter (α : ℝ) (h : sin α - cos α = sqrt 2 / 2) : sin α * cos α = 1 / 4 :=
by
  sorry

end sin_mul_cos_eq_quarter_l1397_139744


namespace positive_expression_l1397_139761

theorem positive_expression (x y : ℝ) : (x^2 - 4 * x + y^2 + 13) > 0 := by
  sorry

end positive_expression_l1397_139761


namespace rectangle_width_l1397_139745

theorem rectangle_width
  (l w : ℕ)
  (h1 : l * w = 1638)
  (h2 : 10 * l = 390) :
  w = 42 :=
by
  sorry

end rectangle_width_l1397_139745


namespace coloring_ways_of_circle_l1397_139796

noncomputable def num_ways_to_color_circle (n : ℕ) (k : ℕ) : ℕ :=
  if h : n % 2 = 1 then -- There are 13 parts; n must be odd (since adjacent matching impossible in even n)
    (k * (k - 1)^(n - 1) : ℕ)
  else
    0

theorem coloring_ways_of_circle :
  num_ways_to_color_circle 13 3 = 6 :=
by
  sorry

end coloring_ways_of_circle_l1397_139796


namespace find_k_l1397_139755

theorem find_k (k : ℚ) :
  (∃ (x y : ℚ), y = 4 * x + 5 ∧ y = -3 * x + 10 ∧ y = 2 * x + k) →
  k = 45 / 7 :=
by
  sorry

end find_k_l1397_139755


namespace pet_store_animals_left_l1397_139757

def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5
def initial_spiders : Nat := 15

def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

def birds_left : Nat := initial_birds - birds_sold
def puppies_left : Nat := initial_puppies - puppies_adopted
def cats_left : Nat := initial_cats
def spiders_left : Nat := initial_spiders - spiders_loose

def total_animals_left : Nat := birds_left + puppies_left + cats_left + spiders_left

theorem pet_store_animals_left : total_animals_left = 25 :=
by
  sorry

end pet_store_animals_left_l1397_139757


namespace bond_selling_price_l1397_139794

theorem bond_selling_price
    (face_value : ℝ)
    (interest_rate_face : ℝ)
    (interest_rate_selling : ℝ)
    (interest : ℝ)
    (selling_price : ℝ)
    (h1 : face_value = 5000)
    (h2 : interest_rate_face = 0.07)
    (h3 : interest_rate_selling = 0.065)
    (h4 : interest = face_value * interest_rate_face)
    (h5 : interest = selling_price * interest_rate_selling) :
  selling_price = 5384.62 :=
sorry

end bond_selling_price_l1397_139794


namespace positive_difference_is_30_l1397_139711

-- Define the absolute value equation condition
def abs_condition (x : ℝ) : Prop := abs (x - 3) = 15

-- Define the solutions to the absolute value equation
def solution1 : ℝ := 18
def solution2 : ℝ := -12

-- Define the positive difference of the solutions
def positive_difference : ℝ := abs (solution1 - solution2)

-- Theorem statement: the positive difference is 30
theorem positive_difference_is_30 : positive_difference = 30 :=
by
  sorry

end positive_difference_is_30_l1397_139711


namespace parametric_eq_to_ordinary_l1397_139703

theorem parametric_eq_to_ordinary (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
    let x := abs (Real.sin (θ / 2) + Real.cos (θ / 2))
    let y := 1 + Real.sin θ
    x ^ 2 = y := by sorry

end parametric_eq_to_ordinary_l1397_139703


namespace final_movie_ticket_price_l1397_139797

variable (initial_price : ℝ) (price_year1 price_year2 price_year3 price_year4 price_year5 : ℝ)

def price_after_years (initial_price : ℝ) : ℝ :=
  let price_year1 := initial_price * 1.12
  let price_year2 := price_year1 * 0.95
  let price_year3 := price_year2 * 1.08
  let price_year4 := price_year3 * 0.96
  let price_year5 := price_year4 * 1.06
  price_year5

theorem final_movie_ticket_price :
  price_after_years 100 = 116.9344512 :=
by
  sorry

end final_movie_ticket_price_l1397_139797


namespace sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l1397_139786

noncomputable def b : ℕ → ℚ
| 0     => 2
| 1     => 3
| (n+2) => 2 * b (n+1) + 3 * b n

theorem sum_bn_over_3_pow_n_plus_1_eq_2_over_5 :
  (∑' n : ℕ, (b n) / (3 ^ (n + 1))) = (2 / 5) :=
by
  sorry

end sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l1397_139786


namespace shortest_third_stick_length_l1397_139746

-- Definitions of the stick lengths
def length1 := 6
def length2 := 9

-- Statement: The shortest length of the third stick that forms a triangle with lengths 6 and 9 should be 4
theorem shortest_third_stick_length : ∃ length3, length3 = 4 ∧
  (length1 + length2 > length3) ∧ (length1 + length3 > length2) ∧ (length2 + length3 > length1) :=
sorry

end shortest_third_stick_length_l1397_139746


namespace combined_mean_score_l1397_139708

-- Definitions based on the conditions
def mean_score_class1 : ℕ := 90
def mean_score_class2 : ℕ := 80
def ratio_students (n1 n2 : ℕ) : Prop := n1 / n2 = 2 / 3

-- Proof statement
theorem combined_mean_score (n1 n2 : ℕ) 
  (h1 : ratio_students n1 n2) 
  (h2 : mean_score_class1 = 90) 
  (h3 : mean_score_class2 = 80) : 
  ((mean_score_class1 * n1) + (mean_score_class2 * n2)) / (n1 + n2) = 84 := 
by
  sorry

end combined_mean_score_l1397_139708


namespace distance_from_origin_is_correct_l1397_139753

noncomputable def is_distance_8_from_x_axis (x y : ℝ) := y = 8
noncomputable def is_distance_12_from_point (x y : ℝ) := (x - 1)^2 + (y - 6)^2 = 144
noncomputable def x_greater_than_1 (x : ℝ) := x > 1
noncomputable def distance_from_origin (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin_is_correct (x y : ℝ)
  (h1 : is_distance_8_from_x_axis x y)
  (h2 : is_distance_12_from_point x y)
  (h3 : x_greater_than_1 x) :
  distance_from_origin x y = Real.sqrt (205 + 2 * Real.sqrt 140) :=
by
  sorry

end distance_from_origin_is_correct_l1397_139753


namespace least_values_3198_l1397_139771

theorem least_values_3198 (x y : ℕ) (hX : ∃ n : ℕ, 3198 + n * 9 = 27)
                         (hY : ∃ m : ℕ, 3198 + m * 11 = 11) :
  x = 6 ∧ y = 8 :=
by
  sorry

end least_values_3198_l1397_139771


namespace sin_theta_plus_pi_over_six_l1397_139778

open Real

theorem sin_theta_plus_pi_over_six (theta : ℝ) (h : sin θ + sin (θ + π / 3) = sqrt 3) :
  sin (θ + π / 6) = 1 := 
sorry

end sin_theta_plus_pi_over_six_l1397_139778


namespace greatest_three_digit_divisible_by_3_6_5_l1397_139780

/-- Define a three-digit number and conditions for divisibility by 3, 6, and 5 -/
def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_divisible_by (n : ℕ) (d : ℕ) : Prop := d ∣ n

/-- Greatest three-digit number divisible by 3, 6, and 5 is 990 -/
theorem greatest_three_digit_divisible_by_3_6_5 : ∃ n : ℕ, is_three_digit n ∧ is_divisible_by n 3 ∧ is_divisible_by n 6 ∧ is_divisible_by n 5 ∧ n = 990 :=
sorry

end greatest_three_digit_divisible_by_3_6_5_l1397_139780


namespace nth_term_formula_l1397_139727

theorem nth_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * n^2 + n)
  (h2 : a 1 = S 1)
  (h3 : ∀ n ≥ 2, a n = S n - S (n - 1))
  : ∀ n, a n = 4 * n - 1 := by
  sorry

end nth_term_formula_l1397_139727


namespace solve_for_pairs_l1397_139752
-- Import necessary libraries

-- Define the operation
def diamond (a b c d : ℤ) : ℤ × ℤ :=
  (a * c - b * d, a * d + b * c)

theorem solve_for_pairs : ∃! (x y : ℤ), diamond x 3 x y = (6, 0) ∧ (x, y) = (0, -2) := by
  sorry

end solve_for_pairs_l1397_139752


namespace eight_sided_dice_theorem_l1397_139751
open Nat

noncomputable def eight_sided_dice_probability : ℚ :=
  let total_outcomes := 8^8
  let favorable_outcomes := 8!
  let probability_all_different := favorable_outcomes / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same

theorem eight_sided_dice_theorem :
  eight_sided_dice_probability = 16736996 / 16777216 := by
    sorry

end eight_sided_dice_theorem_l1397_139751


namespace repeat_decimal_to_fraction_l1397_139737

theorem repeat_decimal_to_fraction : 0.36666 = 11 / 30 :=
by {
    sorry
}

end repeat_decimal_to_fraction_l1397_139737


namespace crayons_left_l1397_139776

theorem crayons_left (initial_crayons : ℕ) (crayons_taken : ℕ) : initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 :=
by
  sorry

end crayons_left_l1397_139776


namespace convert_536_oct_to_base7_l1397_139748

def octal_to_decimal (n : ℕ) : ℕ :=
  n % 10 + (n / 10 % 10) * 8 + (n / 100 % 10) * 64

def decimal_to_base7 (n : ℕ) : ℕ :=
  n % 7 + (n / 7 % 7) * 10 + (n / 49 % 7) * 100 + (n / 343 % 7) * 1000

theorem convert_536_oct_to_base7 : 
  decimal_to_base7 (octal_to_decimal 536) = 1010 :=
by
  sorry

end convert_536_oct_to_base7_l1397_139748


namespace postage_cost_correct_l1397_139710

-- Conditions
def base_rate : ℕ := 35
def additional_rate_per_ounce : ℕ := 25
def weight_in_ounces : ℚ := 5.25
def first_ounce : ℚ := 1
def fraction_weight : ℚ := weight_in_ounces - first_ounce
def num_additional_charges : ℕ := Nat.ceil (fraction_weight)

-- Question and correct answer
def total_postage_cost : ℕ := base_rate + (num_additional_charges * additional_rate_per_ounce)
def answer_in_cents : ℕ := 160

theorem postage_cost_correct : total_postage_cost = answer_in_cents := by sorry

end postage_cost_correct_l1397_139710


namespace find_a22_l1397_139738

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l1397_139738


namespace union_A_B_l1397_139720

open Set

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : A ∪ B = {x | x < 2} := 
by sorry

end union_A_B_l1397_139720


namespace find_h_l1397_139702

theorem find_h (h : ℝ) (j k : ℝ) 
  (y_eq1 : ∀ x : ℝ, (4 * (x - h)^2 + j) = 2030)
  (y_eq2 : ∀ x : ℝ, (5 * (x - h)^2 + k) = 2040)
  (int_xint1 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (4 * x1 * x2 = 2032) )
  (int_xint2 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (5 * x1 * x2 = 2040) ) :
  h = 20.5 :=
by
  sorry

end find_h_l1397_139702


namespace find_parcera_triples_l1397_139739

noncomputable def is_prime (n : ℕ) : Prop := sorry
noncomputable def parcera_triple (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧
  p ∣ q^2 - 4 ∧ q ∣ r^2 - 4 ∧ r ∣ p^2 - 4

theorem find_parcera_triples : 
  {t : ℕ × ℕ × ℕ | parcera_triple t.1 t.2.1 t.2.2} = 
  {(2, 2, 2), (5, 3, 7), (7, 5, 3), (3, 7, 5)} :=
sorry

end find_parcera_triples_l1397_139739


namespace consecutive_integers_divisible_product_l1397_139716

theorem consecutive_integers_divisible_product (m n : ℕ) (h : m < n) :
  ∀ k : ℕ, ∃ i j : ℕ, i ≠ j ∧ k + i < k + n ∧ k + j < k + n ∧ (k + i) * (k + j) % (m * n) = 0 :=
by sorry

end consecutive_integers_divisible_product_l1397_139716


namespace necessary_and_sufficient_condition_l1397_139754

theorem necessary_and_sufficient_condition (x : ℝ) : (x > 0) ↔ (1 / x > 0) :=
by
  sorry

end necessary_and_sufficient_condition_l1397_139754


namespace ark5_ensures_metabolic_energy_l1397_139722

-- Define conditions
def inhibits_ark5_activity (inhibits: Bool) (balance: Bool): Prop :=
  if inhibits then ¬balance else balance

def cancer_cells_proliferate_without_energy (proliferate: Bool) (die_due_to_insufficient_energy: Bool) : Prop :=
  proliferate → die_due_to_insufficient_energy

-- Define the hypothesis based on conditions
def hypothesis (inhibits: Bool) (balance: Bool) (proliferate: Bool) (die_due_to_insufficient_energy: Bool): Prop :=
  inhibits_ark5_activity inhibits balance ∧ cancer_cells_proliferate_without_energy proliferate die_due_to_insufficient_energy

-- Define the theorem to be proved
theorem ark5_ensures_metabolic_energy
  (inhibits : Bool)
  (balance : Bool)
  (proliferate : Bool)
  (die_due_to_insufficient_energy : Bool)
  (h : hypothesis inhibits balance proliferate die_due_to_insufficient_energy) :
  ensures_metabolic_energy :=
  sorry

end ark5_ensures_metabolic_energy_l1397_139722


namespace linda_spent_total_l1397_139715

noncomputable def total_spent (notebooks_price_euro : ℝ) (notebooks_count : ℕ) 
    (pencils_price_pound : ℝ) (pencils_gift_card_pound : ℝ)
    (pens_price_yen : ℝ) (pens_points : ℝ) 
    (markers_price_dollar : ℝ) (calculator_price_dollar : ℝ)
    (marker_discount : ℝ) (coupon_discount : ℝ) (sales_tax : ℝ)
    (euro_to_dollar : ℝ) (pound_to_dollar : ℝ) (yen_to_dollar : ℝ) : ℝ :=
  let notebooks_cost := (notebooks_price_euro * notebooks_count) * euro_to_dollar
  let pencils_cost := 0
  let pens_cost := 0
  let marked_price := markers_price_dollar * (1 - marker_discount)
  let us_total_before_tax := (marked_price + calculator_price_dollar) * (1 - coupon_discount)
  let us_total_after_tax := us_total_before_tax * (1 + sales_tax)
  notebooks_cost + pencils_cost + pens_cost + us_total_after_tax

theorem linda_spent_total : 
  total_spent 1.2 3 1.5 5 170 200 2.8 12.5 0.15 0.10 0.05 1.1 1.25 0.009 = 18.0216 := 
  by
  sorry

end linda_spent_total_l1397_139715


namespace quadratic_function_graph_opens_downwards_l1397_139777

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

-- The problem statement to prove
theorem quadratic_function_graph_opens_downwards :
  (∀ x : ℝ, (quadratic_function (x + 1) - quadratic_function x) < (quadratic_function x - quadratic_function (x - 1))) :=
by
  -- Proof omitted
  sorry

end quadratic_function_graph_opens_downwards_l1397_139777


namespace contractor_absent_days_l1397_139764

variable (x y : ℕ)  -- Number of days worked and absent, both are natural numbers

-- Conditions from the problem
def total_days (x y : ℕ) : Prop := x + y = 30
def total_payment (x y : ℕ) : Prop := 25 * x - 75 * y / 10 = 360

-- Main statement
theorem contractor_absent_days (h1 : total_days x y) (h2 : total_payment x y) : y = 12 :=
by
  sorry

end contractor_absent_days_l1397_139764


namespace completing_square_l1397_139749

theorem completing_square (x : ℝ) (h : x^2 - 6 * x - 7 = 0) : (x - 3)^2 = 16 := 
sorry

end completing_square_l1397_139749


namespace true_proposition_l1397_139729

-- Definitions based on the conditions
def p (x : ℝ) := x * (x - 1) ≠ 0 → x ≠ 0 ∧ x ≠ 1
def q (a b c : ℝ) := a > b → c > 0 → a * c > b * c

-- The theorem based on the question and the conditions
theorem true_proposition (x a b c : ℝ) (hp : p x) (hq_false : ¬ q a b c) : p x ∨ q a b c :=
by
  sorry

end true_proposition_l1397_139729


namespace divisibility_of_polynomial_l1397_139782

theorem divisibility_of_polynomial (n : ℕ) (h : n ≥ 1) : 
  ∃ primes : Finset ℕ, primes.card = n ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ (2^(2^n) + 2^(2^(n-1)) + 1) :=
sorry

end divisibility_of_polynomial_l1397_139782


namespace product_modulo_l1397_139732

theorem product_modulo (n : ℕ) (h : 93 * 68 * 105 ≡ n [MOD 20]) (h_range : 0 ≤ n ∧ n < 20) : n = 0 := 
by
  sorry

end product_modulo_l1397_139732


namespace function_properties_l1397_139788

noncomputable def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

theorem function_properties 
  (b c : ℝ) :
  ((c = 0 → (∀ x : ℝ, f (-x) b 0 = -f x b 0)) ∧
   (b = 0 → (∀ x₁ x₂ : ℝ, (x₁ ≤ x₂ → f x₁ 0 c ≤ f x₂ 0 c))) ∧
   (∃ (c : ℝ), ∀ (x : ℝ), f (x + c) b c = f (x - c) b c) ∧
   (¬ ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0))) := 
by
  sorry

end function_properties_l1397_139788


namespace exists_xy_l1397_139765

-- Given conditions from the problem
variables (m x0 y0 : ℕ)
-- Integers x0 and y0 are relatively prime
variables (rel_prim : Nat.gcd x0 y0 = 1)
-- y0 divides x0^2 + m
variables (div_y0 : y0 ∣ x0^2 + m)
-- x0 divides y0^2 + m
variables (div_x0 : x0 ∣ y0^2 + m)

-- Main theorem statement
theorem exists_xy 
  (hm : m > 0) 
  (hx0 : x0 > 0) 
  (hy0 : y0 > 0) 
  (rel_prim : Nat.gcd x0 y0 = 1) 
  (div_y0 : y0 ∣ x0^2 + m) 
  (div_x0 : x0 ∣ y0^2 + m) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ Nat.gcd x y = 1 ∧ y ∣ x^2 + m ∧ x ∣ y^2 + m ∧ x + y ≤ m + 1 := 
sorry

end exists_xy_l1397_139765


namespace people_counted_on_second_day_l1397_139724

theorem people_counted_on_second_day (x : ℕ) (H1 : 2 * x + x = 1500) : x = 500 :=
by {
  sorry -- Proof goes here
}

end people_counted_on_second_day_l1397_139724


namespace inequality_to_prove_l1397_139741

variable {r r1 r2 r3 m : ℝ}
variable {A B C : ℝ}

-- Conditions
-- r is the radius of an inscribed circle in a triangle
-- r1, r2, r3 are radii of circles each touching two sides of the triangle and the inscribed circle
-- m is a real number such that m >= 1/2

axiom r_radii_condition : r > 0
axiom r1_radii_condition : r1 > 0
axiom r2_radii_condition : r2 > 0
axiom r3_radii_condition : r3 > 0
axiom m_condition : m ≥ 1/2

-- Inequality to prove
theorem inequality_to_prove : 
  (r1 * r2) ^ m + (r2 * r3) ^ m + (r3 * r1) ^ m ≥ 3 * (r / 3) ^ (2 * m) := 
sorry

end inequality_to_prove_l1397_139741


namespace total_digits_2500_is_9449_l1397_139795

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end total_digits_2500_is_9449_l1397_139795


namespace total_pencils_l1397_139769

theorem total_pencils (reeta_pencils anika_pencils kamal_pencils : ℕ) :
  reeta_pencils = 30 →
  anika_pencils = 2 * reeta_pencils + 4 →
  kamal_pencils = 3 * reeta_pencils - 2 →
  reeta_pencils + anika_pencils + kamal_pencils = 182 :=
by
  intros h_reeta h_anika h_kamal
  sorry

end total_pencils_l1397_139769


namespace find_sum_of_a_and_d_l1397_139770

theorem find_sum_of_a_and_d 
  {a b c d : ℝ} 
  (h1 : ab + ac + bd + cd = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 :=
sorry

end find_sum_of_a_and_d_l1397_139770


namespace part_I_part_II_l1397_139762

noncomputable def seq_a : ℕ → ℝ 
| 0       => 1   -- Normally, we start with n = 1, so we set a_0 to some default value.
| (n+1)   => (1 + 1 / (n^2 + n)) * seq_a n + 1 / (2^n)

theorem part_I (n : ℕ) (h: n ≥ 2) : seq_a n ≥ 2 :=
sorry

theorem part_II (n : ℕ) : seq_a n < Real.exp 2 :=
sorry

-- Assumption: ln(1 + x) < x for all x > 0
axiom ln_ineq (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x

end part_I_part_II_l1397_139762


namespace minimum_n_of_colored_balls_l1397_139768

theorem minimum_n_of_colored_balls (n : ℕ) (h1 : n ≥ 3)
  (h2 : (n * (n + 1)) / 2 % 10 = 0) : n = 24 :=
sorry

end minimum_n_of_colored_balls_l1397_139768


namespace selection_methods_including_both_boys_and_girls_l1397_139713

def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def select : ℕ := 4

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_methods_including_both_boys_and_girls :
  combination 7 4 - combination boys 4 = 34 :=
by
  sorry

end selection_methods_including_both_boys_and_girls_l1397_139713


namespace evaluate_expression_l1397_139774

theorem evaluate_expression : 
  (3^2015 + 3^2013 + 3^2012) / (3^2015 - 3^2013 + 3^2012) = 31 / 25 :=
by
  sorry

end evaluate_expression_l1397_139774


namespace least_n_prime_condition_l1397_139787

theorem least_n_prime_condition : ∃ n : ℕ, (∀ p : ℕ, Prime p → ¬ Prime (p^2 + n)) ∧ (∀ m : ℕ, 
 (m > 0 ∧ ∀ p : ℕ, Prime p → ¬ Prime (p^2 + m)) → m ≥ 5) ∧ n = 5 := by
  sorry

end least_n_prime_condition_l1397_139787


namespace possible_values_of_m_plus_n_l1397_139740

theorem possible_values_of_m_plus_n (m n : ℕ) (hmn_pos : 0 < m ∧ 0 < n) 
  (cond : Nat.lcm m n - Nat.gcd m n = 103) : m + n = 21 ∨ m + n = 105 ∨ m + n = 309 := by
  sorry

end possible_values_of_m_plus_n_l1397_139740


namespace base_six_equals_base_b_l1397_139747

theorem base_six_equals_base_b (b : ℕ) (h1 : 3 * 6 ^ 1 + 4 * 6 ^ 0 = 22)
  (h2 : b ^ 2 + 2 * b + 1 = 22) : b = 3 :=
sorry

end base_six_equals_base_b_l1397_139747


namespace flagpole_height_l1397_139725

theorem flagpole_height
  (AB : ℝ) (AD : ℝ) (BC : ℝ)
  (h1 : AB = 10)
  (h2 : BC = 3)
  (h3 : 2 * AD^2 = AB^2 + BC^2) :
  AD = Real.sqrt 54.5 :=
by 
  -- Proof omitted
  sorry

end flagpole_height_l1397_139725


namespace distribution_of_earnings_l1397_139784

theorem distribution_of_earnings :
  let payments := [10, 15, 20, 25, 30, 50]
  let total_earnings := payments.sum 
  let equal_share := total_earnings / 6
  50 - equal_share = 25 := by
  sorry

end distribution_of_earnings_l1397_139784


namespace meadowbrook_total_not_74_l1397_139706

theorem meadowbrook_total_not_74 (h c : ℕ) : 
  21 * h + 6 * c ≠ 74 := sorry

end meadowbrook_total_not_74_l1397_139706


namespace union_of_A_and_B_l1397_139758

open Set

variable {α : Type}

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 4} := 
by
  sorry

end union_of_A_and_B_l1397_139758


namespace square_field_side_length_l1397_139733

theorem square_field_side_length (time_sec : ℕ) (speed_kmh : ℕ) (perimeter : ℕ) (side_length : ℕ)
  (h1 : time_sec = 96)
  (h2 : speed_kmh = 9)
  (h3 : perimeter = (9 * 1000 / 3600 : ℕ) * 96)
  (h4 : perimeter = 4 * side_length) :
  side_length = 60 :=
by
  sorry

end square_field_side_length_l1397_139733


namespace find_m_n_sum_l1397_139783

noncomputable def q : ℚ := 2 / 11

theorem find_m_n_sum {m n : ℕ} (hq : q = m / n) (coprime_mn : Nat.gcd m n = 1) : m + n = 13 := by
  sorry

end find_m_n_sum_l1397_139783


namespace solve_for_A_in_terms_of_B_l1397_139731

noncomputable def f (A B x : ℝ) := A * x - 2 * B^2
noncomputable def g (B x : ℝ) := B * x

theorem solve_for_A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 1) = 0) : A = 2 * B := by
  sorry

end solve_for_A_in_terms_of_B_l1397_139731


namespace apple_count_l1397_139743

-- Definitions of initial conditions and calculations.
def B_0 : Int := 5  -- initial number of blue apples
def R_0 : Int := 3  -- initial number of red apples
def Y : Int := 2 * B_0  -- number of yellow apples given by neighbor
def R : Int := R_0 - 2  -- number of red apples after giving away to a friend
def B : Int := B_0 - 3  -- number of blue apples after 3 rot
def G : Int := (B + Y) / 3  -- number of green apples received
def Y' : Int := Y - 2  -- number of yellow apples after eating 2
def R' : Int := R - 1  -- number of red apples after eating 1

-- Lean theorem statement
theorem apple_count (B_0 R_0 Y R B G Y' R' : ℤ)
  (h1 : B_0 = 5)
  (h2 : R_0 = 3)
  (h3 : Y = 2 * B_0)
  (h4 : R = R_0 - 2)
  (h5 : B = B_0 - 3)
  (h6 : G = (B + Y) / 3)
  (h7 : Y' = Y - 2)
  (h8 : R' = R - 1)
  : B + Y' + G + R' = 14 := 
by
  sorry

end apple_count_l1397_139743


namespace problem_f_2019_l1397_139779

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f1 : f 1 = 1/4
axiom f2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2019 : f 2019 = -1/2 :=
by
  sorry

end problem_f_2019_l1397_139779


namespace adam_and_simon_50_miles_apart_l1397_139704

noncomputable def time_when_50_miles_apart (x : ℝ) : Prop :=
  let adam_distance := 10 * x
  let simon_distance := 8 * x
  (adam_distance^2 + simon_distance^2 = 50^2) 

theorem adam_and_simon_50_miles_apart : 
  ∃ x : ℝ, time_when_50_miles_apart x ∧ x = 50 / 12.8 := 
sorry

end adam_and_simon_50_miles_apart_l1397_139704


namespace sum_of_cubes_l1397_139723

variable {R : Type} [OrderedRing R] [Field R] [DecidableEq R]

theorem sum_of_cubes (a b c : R) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
    (h₄ : (a^3 + 12) / a = (b^3 + 12) / b) (h₅ : (b^3 + 12) / b = (c^3 + 12) / c) :
    a^3 + b^3 + c^3 = -36 := by
  sorry

end sum_of_cubes_l1397_139723


namespace evaluate_f_2x_l1397_139742

def f (x : ℝ) : ℝ := x^2 - 1

theorem evaluate_f_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end evaluate_f_2x_l1397_139742


namespace line_intersects_x_axis_at_point_l1397_139705

-- Define the conditions and required proof
theorem line_intersects_x_axis_at_point :
  (∃ x : ℝ, ∃ y : ℝ, 5 * y - 7 * x = 35 ∧ y = 0 ∧ (x, y) = (-5, 0)) :=
by
  -- The proof is omitted according to the steps
  sorry

end line_intersects_x_axis_at_point_l1397_139705


namespace sylvia_buttons_l1397_139719

theorem sylvia_buttons (n : ℕ) (h₁: n % 10 = 0) (h₂: n ≥ 80):
  (∃ w : ℕ, w = (n - (n / 2) - (n / 5) - 8)) ∧ (n - (n / 2) - (n / 5) - 8 = 1) :=
by
  sorry

end sylvia_buttons_l1397_139719


namespace total_food_amount_l1397_139799

-- Define constants for the given problem
def chicken : ℕ := 16
def hamburgers : ℕ := chicken / 2
def hot_dogs : ℕ := hamburgers + 2
def sides : ℕ := hot_dogs / 2

-- Prove the total amount of food Peter will buy is 39 pounds
theorem total_food_amount : chicken + hamburgers + hot_dogs + sides = 39 := by
  sorry

end total_food_amount_l1397_139799


namespace polygon_sides_with_diagonals_44_l1397_139767

theorem polygon_sides_with_diagonals_44 (n : ℕ) (hD : 44 = n * (n - 3) / 2) : n = 11 :=
by
  sorry

end polygon_sides_with_diagonals_44_l1397_139767


namespace find_r_over_s_at_2_l1397_139714

noncomputable def r (x : ℝ) := 6 * x
noncomputable def s (x : ℝ) := (x + 4) * (x - 1)

theorem find_r_over_s_at_2 :
  r 2 / s 2 = 2 :=
by
  -- The corresponding steps to show this theorem.
  sorry

end find_r_over_s_at_2_l1397_139714


namespace train_length_l1397_139790

theorem train_length (v_train : ℝ) (v_man : ℝ) (t : ℝ) (length_train : ℝ)
  (h1 : v_train = 55) (h2 : v_man = 7) (h3 : t = 10.45077684107852) :
  length_train = 180 :=
by
  sorry

end train_length_l1397_139790


namespace evaluate_expression_l1397_139756

theorem evaluate_expression :
  (- (3 / 4 : ℚ)) / 3 * (- (2 / 5 : ℚ)) = 1 / 10 := 
by
  -- Here is where the proof would go
  sorry

end evaluate_expression_l1397_139756


namespace extreme_point_at_1_l1397_139798

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 + (2 * a^3 - a^2) * Real.log x - (a^2 + 2 * a - 1) * x

theorem extreme_point_at_1 (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ ∀ x > 0, deriv (f a) x = 0 →
  a = -1) := sorry

end extreme_point_at_1_l1397_139798
