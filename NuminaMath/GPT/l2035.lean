import Mathlib

namespace cost_per_page_of_notebooks_l2035_203543

-- Define the conditions
def notebooks : Nat := 2
def pages_per_notebook : Nat := 50
def cost_in_dollars : Nat := 5

-- Define the conversion constants
def dollars_to_cents : Nat := 100

-- Define the correct answer
def expected_cost_per_page := 5

-- State the theorem to prove the cost per page
theorem cost_per_page_of_notebooks :
  let total_pages := notebooks * pages_per_notebook
  let total_cost_in_cents := cost_in_dollars * dollars_to_cents
  let cost_per_page := total_cost_in_cents / total_pages
  cost_per_page = expected_cost_per_page :=
by
  -- Skip the proof with sorry
  sorry

end cost_per_page_of_notebooks_l2035_203543


namespace dolls_total_l2035_203569

theorem dolls_total (V S A : ℕ) 
  (hV : V = 20) 
  (hS : S = 2 * V)
  (hA : A = 2 * S) 
  : A + S + V = 140 := 
by 
  sorry

end dolls_total_l2035_203569


namespace sum_alternating_sequence_l2035_203500

theorem sum_alternating_sequence : (Finset.range 2012).sum (λ k => (-1 : ℤ)^(k + 1)) = 0 :=
by
  sorry

end sum_alternating_sequence_l2035_203500


namespace find_x_l2035_203566

def vec_a : ℝ × ℝ × ℝ := (-2, 1, 3)
def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-2) * 1 + 1 * x + 3 * (-1) = 0) : x = 5 :=
by
  sorry

end find_x_l2035_203566


namespace min_value_fraction_l2035_203537

theorem min_value_fraction (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∃c, (c = 8) ∧ (∀z w : ℝ, z > 1 → w > 1 → ((z^3 / (w - 1) + w^3 / (z - 1)) ≥ c))) :=
by 
  sorry

end min_value_fraction_l2035_203537


namespace inequality_proof_l2035_203548

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : -b > 0) (h3 : a > -b) (h4 : c < 0) : 
  a * (1 - c) > b * (c - 1) :=
sorry

end inequality_proof_l2035_203548


namespace company_spends_less_l2035_203528

noncomputable def total_spending_reduction_in_dollars : ℝ :=
  let magazine_initial_cost := 840.00
  let online_resources_initial_cost_gbp := 960.00
  let exchange_rate := 1.40
  let mag_cut_percentage := 0.30
  let online_cut_percentage := 0.20

  let magazine_cost_cut := magazine_initial_cost * mag_cut_percentage
  let online_resource_cost_cut_gbp := online_resources_initial_cost_gbp * online_cut_percentage
  
  let new_magazine_cost := magazine_initial_cost - magazine_cost_cut
  let new_online_resource_cost_gbp := online_resources_initial_cost_gbp - online_resource_cost_cut_gbp

  let online_resources_initial_cost := online_resources_initial_cost_gbp * exchange_rate
  let new_online_resource_cost := new_online_resource_cost_gbp * exchange_rate

  let mag_cut_amount := magazine_initial_cost - new_magazine_cost
  let online_cut_amount := online_resources_initial_cost - new_online_resource_cost
  
  mag_cut_amount + online_cut_amount

theorem company_spends_less : total_spending_reduction_in_dollars = 520.80 :=
by
  sorry

end company_spends_less_l2035_203528


namespace S6_values_l2035_203584

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

axiom geo_seq (q : ℝ) :
  ∀ n : ℕ, a n = a 0 * q ^ n

variable (a3_eq_4 : a 2 = 4) 
variable (S3_eq_7 : S 3 = 7)

theorem S6_values : S 6 = 63 ∨ S 6 = 133 / 27 := sorry

end S6_values_l2035_203584


namespace spending_required_for_free_shipping_l2035_203518

def shampoo_cost : ℕ := 10
def conditioner_cost : ℕ := 10
def lotion_cost : ℕ := 6
def shampoo_count : ℕ := 1
def conditioner_count : ℕ := 1
def lotion_count : ℕ := 3
def additional_spending_needed : ℕ := 12
def current_spending : ℕ := (shampoo_cost * shampoo_count) + (conditioner_cost * conditioner_count) + (lotion_cost * lotion_count)

theorem spending_required_for_free_shipping : current_spending + additional_spending_needed = 50 := by
  sorry

end spending_required_for_free_shipping_l2035_203518


namespace largest_divisor_of_expression_l2035_203591

theorem largest_divisor_of_expression :
  ∃ k : ℕ, (∀ m : ℕ, (m > k → m ∣ (1991 ^ k * 1990 ^ (1991 ^ 1992) + 1992 ^ (1991 ^ 1990)) = false))
  ∧ k = 1991 := by
sorry

end largest_divisor_of_expression_l2035_203591


namespace find_f_3_l2035_203531

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l2035_203531


namespace age_problem_l2035_203596

-- Defining the conditions and the proof problem
variables (B A : ℕ) -- B and A are natural numbers

-- Given conditions
def B_age : ℕ := 38
def A_age (B : ℕ) : ℕ := B + 8
def age_in_10_years (A : ℕ) : ℕ := A + 10
def years_ago (B : ℕ) (X : ℕ) : ℕ := B - X

-- Lean statement of the problem
theorem age_problem (X : ℕ) (hB : B = B_age) (hA : A = A_age B):
  age_in_10_years A = 2 * (years_ago B X) → X = 10 :=
by
  sorry

end age_problem_l2035_203596


namespace line_equation_of_intersection_points_l2035_203556

theorem line_equation_of_intersection_points (x y : ℝ) :
  (x^2 + y^2 - 6*x - 7 = 0) ∧ (x^2 + y^2 - 6*y - 27 = 0) → (3*x - 3*y = 10) :=
by
  sorry

end line_equation_of_intersection_points_l2035_203556


namespace ratio_jacob_edward_l2035_203590

-- Definitions and conditions
def brian_shoes : ℕ := 22
def edward_shoes : ℕ := 3 * brian_shoes
def total_shoes : ℕ := 121
def jacob_shoes : ℕ := total_shoes - brian_shoes - edward_shoes

-- Statement of the problem
theorem ratio_jacob_edward (h_brian : brian_shoes = 22)
                          (h_edward : edward_shoes = 3 * brian_shoes)
                          (h_total : total_shoes = 121)
                          (h_jacob : jacob_shoes = total_shoes - brian_shoes - edward_shoes) :
                          jacob_shoes / edward_shoes = 1 / 2 :=
by sorry

end ratio_jacob_edward_l2035_203590


namespace arithmetic_sequence_term_l2035_203553

theorem arithmetic_sequence_term (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 3 = 4) : a 10 = 18 :=
by
  sorry

end arithmetic_sequence_term_l2035_203553


namespace find_values_l2035_203524

theorem find_values (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  3 * Real.sin x + 2 * Real.cos x = ( -21 + 13 * Real.sqrt 145 ) / 58 ∨
  3 * Real.sin x + 2 * Real.cos x = ( -21 - 13 * Real.sqrt 145 ) / 58 := sorry

end find_values_l2035_203524


namespace find_g_function_l2035_203581

noncomputable def g : ℝ → ℝ :=
  sorry

theorem find_g_function (x y : ℝ) (h1 : g 1 = 2) (h2 : ∀ (x y : ℝ), g (x + y) = 5^y * g x + 3^x * g y) :
  g x = 5^x - 3^x :=
by
  sorry

end find_g_function_l2035_203581


namespace line_intersects_ellipse_slopes_l2035_203588

theorem line_intersects_ellipse_slopes :
  {m : ℝ | ∃ x, 4 * x^2 + 25 * (m * x + 8)^2 = 100} = 
  {m : ℝ | m ≤ -Real.sqrt 2.4 ∨ Real.sqrt 2.4 ≤ m} := 
by
  sorry

end line_intersects_ellipse_slopes_l2035_203588


namespace symmetric_point_l2035_203585

-- Define the given conditions
def pointP : (ℤ × ℤ) := (3, -2)
def symmetry_line (y : ℤ) := (y = 1)

-- Prove the assertion that point Q is (3, 4)
theorem symmetric_point (x y1 y2 : ℤ) (hx: x = 3) (hy1: y1 = -2) (hy : symmetry_line 1) :
  (x, 2 * 1 - y1) = (3, 4) :=
by
  sorry

end symmetric_point_l2035_203585


namespace B_pow_48_l2035_203559

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 0],
  ![0, 0, 1],
  ![0, -1, 0]
]

theorem B_pow_48 :
  B^48 = ![
    ![0, 0, 0],
    ![0, 1, 0],
    ![0, 0, 1]
  ] := by sorry

end B_pow_48_l2035_203559


namespace max_students_l2035_203564

open BigOperators

def seats_in_row (i : ℕ) : ℕ := 8 + 2 * i

def max_students_in_row (i : ℕ) : ℕ := 4 + i

def total_max_students : ℕ := ∑ i in Finset.range 15, max_students_in_row (i + 1)

theorem max_students (condition1 : true) : total_max_students = 180 :=
by
  sorry

end max_students_l2035_203564


namespace forum_members_l2035_203527

theorem forum_members (M : ℕ)
  (h1 : ∀ q a, a = 3 * q)
  (h2 : ∀ h d, q = 3 * h * d)
  (h3 : 24 * (M * 3 * (24 + 3 * 72)) = 57600) : M = 200 :=
by
  sorry

end forum_members_l2035_203527


namespace snowfall_rate_in_Hamilton_l2035_203562

theorem snowfall_rate_in_Hamilton 
  (initial_depth_Kingston : ℝ := 12.1)
  (rate_Kingston : ℝ := 2.6)
  (initial_depth_Hamilton : ℝ := 18.6)
  (duration : ℕ := 13)
  (final_depth_equal : initial_depth_Kingston + rate_Kingston * duration = initial_depth_Hamilton + duration * x)
  (x : ℝ) :
  x = 2.1 :=
sorry

end snowfall_rate_in_Hamilton_l2035_203562


namespace contradiction_example_l2035_203571

theorem contradiction_example 
  (a b c : ℝ) 
  (h : (a - 1) * (b - 1) * (c - 1) > 0) : 
  (1 < a) ∨ (1 < b) ∨ (1 < c) :=
by
  sorry

end contradiction_example_l2035_203571


namespace oblique_asymptote_l2035_203511

theorem oblique_asymptote :
  ∀ x : ℝ, (∃ δ > 0, ∀ y > x, (abs (3 * y^2 + 8 * y + 12) / (3 * y + 4) - (y + 4 / 3)) < δ) :=
sorry

end oblique_asymptote_l2035_203511


namespace cos_double_angle_l2035_203558

theorem cos_double_angle (theta : ℝ) (h : Real.sin (Real.pi - theta) = 1 / 3) : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end cos_double_angle_l2035_203558


namespace fraction_inequality_solution_l2035_203504

open Set

theorem fraction_inequality_solution :
  {x : ℝ | 7 * x - 3 ≥ x^2 - x - 12 ∧ x ≠ 3 ∧ x ≠ -4} = Icc (-1 : ℝ) 3 ∪ Ioo (3 : ℝ) 4 ∪ Icc 4 9 :=
by
  sorry

end fraction_inequality_solution_l2035_203504


namespace div_polynomials_l2035_203503

variable (a b : ℝ)

theorem div_polynomials :
  10 * a^3 * b^2 / (-5 * a^2 * b) = -2 * a * b := 
by sorry

end div_polynomials_l2035_203503


namespace ab_product_l2035_203583

theorem ab_product (a b : ℝ) (h_sol : ∀ x, -1 < x ∧ x < 4 → x^2 + a * x + b < 0) 
  (h_roots : ∀ x, x^2 + a * x + b = 0 ↔ x = -1 ∨ x = 4) : 
  a * b = 12 :=
sorry

end ab_product_l2035_203583


namespace find_percentage_l2035_203538

theorem find_percentage (P : ℕ) (h: (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 18) : P = 15 :=
sorry

end find_percentage_l2035_203538


namespace mono_increasing_m_value_l2035_203532

theorem mono_increasing_m_value (m : ℝ) :
  (∀ x : ℝ, 0 ≤ 3 * x ^ 2 + 4 * x + m) → (m ≥ 4 / 3) :=
by
  intro h
  sorry

end mono_increasing_m_value_l2035_203532


namespace geometric_progression_common_ratio_l2035_203554

/--
If \( a_1, a_2, a_3 \) are terms of an arithmetic progression with common difference \( d \neq 0 \),
and the products \( a_1 a_2, a_2 a_3, a_3 a_1 \) form a geometric progression,
then the common ratio of this geometric progression is \(-2\).
-/
theorem geometric_progression_common_ratio (a₁ a₂ a₃ d : ℝ) (h₀ : d ≠ 0) (h₁ : a₂ = a₁ + d)
  (h₂ : a₃ = a₁ + 2 * d) (h₃ : (a₂ * a₃) / (a₁ * a₂) = (a₃ * a₁) / (a₂ * a₃)) :
  (a₂ * a₃) / (a₁ * a₂) = -2 :=
by
  sorry

end geometric_progression_common_ratio_l2035_203554


namespace neg_one_exponent_difference_l2035_203516

theorem neg_one_exponent_difference : (-1 : ℤ) ^ 2004 - (-1 : ℤ) ^ 2003 = 2 := by
  sorry

end neg_one_exponent_difference_l2035_203516


namespace min_max_solution_A_l2035_203508

theorem min_max_solution_A (x y z : ℕ) (h₁ : x + y + z = 100) (h₂ : 5 * x + 8 * y + 9 * z = 700) 
                           (h₃ : 0 ≤ x ∧ x ≤ 60) (h₄ : 0 ≤ y ∧ y ≤ 60) (h₅ : 0 ≤ z ∧ z ≤ 47) :
    35 ≤ x ∧ x ≤ 49 :=
by
  sorry

end min_max_solution_A_l2035_203508


namespace rectangle_area_with_circles_touching_l2035_203570

theorem rectangle_area_with_circles_touching
  (r : ℝ)
  (radius_pos : r = 3)
  (short_side : ℝ)
  (long_side : ℝ)
  (dim_rect : short_side = 2 * r ∧ long_side = 4 * r) :
  short_side * long_side = 72 :=
by
  sorry

end rectangle_area_with_circles_touching_l2035_203570


namespace parking_spaces_in_the_back_l2035_203545

theorem parking_spaces_in_the_back
  (front_spaces : ℕ)
  (cars_parked : ℕ)
  (half_back_filled : ℕ → ℚ)
  (spaces_available : ℕ)
  (B : ℕ)
  (h1 : front_spaces = 52)
  (h2 : cars_parked = 39)
  (h3 : half_back_filled B = B / 2)
  (h4 : spaces_available = 32) :
  B = 38 :=
by
  -- Here you can provide the proof steps.
  sorry

end parking_spaces_in_the_back_l2035_203545


namespace chord_segments_division_l2035_203597

theorem chord_segments_division (O : Point) (r r0 : ℝ) (h : r0 < r) : 
  3 * r0 ≥ r :=
sorry

end chord_segments_division_l2035_203597


namespace intercept_sum_l2035_203521

theorem intercept_sum {x y : ℝ} 
  (h : y - 3 = -3 * (x - 5)) 
  (hx : x = 6) 
  (hy : y = 18) 
  (intercept_sum_eq : x + y = 24) : 
  x + y = 24 :=
by
  sorry

end intercept_sum_l2035_203521


namespace arithmetic_sequence_nth_term_l2035_203547

theorem arithmetic_sequence_nth_term (x n : ℕ) (a1 a2 a3 : ℚ) (a_n : ℕ) :
  a1 = 3 * x - 5 ∧ a2 = 7 * x - 17 ∧ a3 = 4 * x + 3 ∧ a_n = 4033 →
  n = 641 :=
by sorry

end arithmetic_sequence_nth_term_l2035_203547


namespace algebraic_expression_value_l2035_203510

theorem algebraic_expression_value (p q : ℤ) 
  (h : 8 * p + 2 * q = -2023) : 
  (p * (-2) ^ 3 + q * (-2) + 1 = 2024) :=
by
  sorry

end algebraic_expression_value_l2035_203510


namespace bowling_ball_weight_l2035_203523

-- Define the weights of the bowling balls and canoes
variables (b c : ℝ)

-- Conditions provided by the problem statement
axiom eq1 : 8 * b = 4 * c
axiom eq2 : 3 * c = 108

-- Prove that one bowling ball weighs 18 pounds
theorem bowling_ball_weight : b = 18 :=
by
  sorry

end bowling_ball_weight_l2035_203523


namespace domain_of_f_l2035_203555

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ -3) ↔ ((x < -3) ∨ (-3 < x ∧ x < 3) ∨ (x > 3)) :=
by
  sorry

end domain_of_f_l2035_203555


namespace geom_seq_sum_is_15_l2035_203541

theorem geom_seq_sum_is_15 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) (hq : q = -2) (h_geom : ∀ n, a (n + 1) = a n * q) :
  a 1 + |a 2| + a 3 + |a 4| = 15 :=
by
  sorry

end geom_seq_sum_is_15_l2035_203541


namespace rationalize_denominator_l2035_203552

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end rationalize_denominator_l2035_203552


namespace min_value_proof_l2035_203530

noncomputable def min_value (x y : ℝ) : ℝ := 1 / x + 1 / (2 * y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) :
  min_value x y = 4 :=
sorry

end min_value_proof_l2035_203530


namespace find_annual_interest_rate_l2035_203522

theorem find_annual_interest_rate 
  (TD : ℝ) (FV : ℝ) (T : ℝ) (expected_R: ℝ)
  (hTD : TD = 189)
  (hFV : FV = 1764)
  (hT : T = 9 / 12)
  (hExpected : expected_R = 16) : 
  ∃ R : ℝ, 
  (TD = (FV - (FV - TD)) * R * T / 100) ∧ 
  R = expected_R := 
by 
  sorry

end find_annual_interest_rate_l2035_203522


namespace parabola_shift_units_l2035_203514

theorem parabola_shift_units (h : ℝ) :
  (∃ h, (0 + 3 - h)^2 - 1 = 0) ↔ (h = 2 ∨ h = 4) :=
by 
  sorry

end parabola_shift_units_l2035_203514


namespace necessity_of_A_for_B_l2035_203525

variables {a b h : ℝ}

def PropA (a b h : ℝ) : Prop := |a - b| < 2 * h
def PropB (a b h : ℝ) : Prop := |a - 1| < h ∧ |b - 1| < h

theorem necessity_of_A_for_B (h_pos : 0 < h) : 
  (∀ a b, PropB a b h → PropA a b h) ∧ ¬ (∀ a b, PropA a b h → PropB a b h) :=
by sorry

end necessity_of_A_for_B_l2035_203525


namespace yasmine_chocolate_beverage_l2035_203544

theorem yasmine_chocolate_beverage :
  ∃ (m s : ℕ), (∀ k : ℕ, k > 0 → (∃ n : ℕ, 4 * n = 7 * k) → (m, s) = (7 * k, 4 * k)) ∧
  (2 * 7 * 1 + 1.4 * 4 * 1) = 19.6 := by
sorry

end yasmine_chocolate_beverage_l2035_203544


namespace equidistant_point_l2035_203563

theorem equidistant_point (x y : ℝ) :
  (abs x = abs y) → (abs x = abs (x + y - 3) / (Real.sqrt 2)) → x = 1.5 :=
by {
  -- proof omitted
  sorry
}

end equidistant_point_l2035_203563


namespace problem_I_problem_II_l2035_203568

-- Problem (I)
theorem problem_I (a b : ℝ) (h1 : a = 1) (h2 : b = 1) :
  { x : ℝ | |2*x + a| + |2*x - 2*b| + 3 > 8 } = 
  { x : ℝ | x < -1 ∨ x > 1.5 } := by
  sorry

-- Problem (II)
theorem problem_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∀ x : ℝ, |2*x + a| + |2*x - 2*b| + 3 ≥ 5) :
  (1 / a + 1 / b) = (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end problem_I_problem_II_l2035_203568


namespace stock_price_end_of_second_year_l2035_203513

def initial_price : ℝ := 80
def first_year_increase_rate : ℝ := 1.2
def second_year_decrease_rate : ℝ := 0.3

theorem stock_price_end_of_second_year : 
  initial_price * (1 + first_year_increase_rate) * (1 - second_year_decrease_rate) = 123.2 := 
by sorry

end stock_price_end_of_second_year_l2035_203513


namespace evaluate_F_2_f_3_l2035_203587

def f (a : ℕ) : ℕ := a^2 - 2*a
def F (a b : ℕ) : ℕ := b^2 + a*b

theorem evaluate_F_2_f_3 : F 2 (f 3) = 15 := by
  sorry

end evaluate_F_2_f_3_l2035_203587


namespace tea_bags_number_l2035_203574

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l2035_203574


namespace tennis_tournament_matches_l2035_203549

theorem tennis_tournament_matches (num_players : ℕ) (total_days : ℕ) (rest_days : ℕ)
  (num_matches_per_day : ℕ) (matches_per_player : ℕ)
  (h1 : num_players = 10)
  (h2 : total_days = 9)
  (h3 : rest_days = 1)
  (h4 : num_matches_per_day = 5)
  (h5 : matches_per_player = 1)
  : (num_players * (num_players - 1) / 2) - (num_matches_per_day * (total_days - rest_days)) = 40 :=
by
  sorry

end tennis_tournament_matches_l2035_203549


namespace problem_statement_l2035_203573

structure Pricing :=
  (price_per_unit_1 : ℕ) (threshold_1 : ℕ)
  (price_per_unit_2 : ℕ) (threshold_2 : ℕ)
  (price_per_unit_3 : ℕ)

def cost (units : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if units ≤ t1 then units * p1
  else if units ≤ t2 then t1 * p1 + (units - t1) * p2
  else t1 * p1 + (t2 - t1) * p2 + (units - t2) * p3 

def units_given_cost (c : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if c ≤ t1 * p1 then c / p1
  else if c ≤ t1 * p1 + (t2 - t1) * p2 then t1 + (c - t1 * p1) / p2
  else t2 + (c - t1 * p1 - (t2 - t1) * p2) / p3

def double_eleven_case (total_units total_cost : ℕ) (x_units : ℕ) (pricing : Pricing) : ℕ :=
  let y_units := total_units - x_units
  let case1_cost := cost x_units pricing + cost y_units pricing
  if case1_cost = total_cost then (x_units, y_units).fst
  else sorry

theorem problem_statement (pricing : Pricing):
  (cost 120 pricing = 420) ∧ 
  (cost 260 pricing = 868) ∧
  (units_given_cost 740 pricing = 220) ∧
  (double_eleven_case 400 1349 290 pricing = 290)
  := sorry

end problem_statement_l2035_203573


namespace pair_not_equal_to_64_l2035_203580

theorem pair_not_equal_to_64 :
  ¬(4 * (9 / 2) = 64) := by
  sorry

end pair_not_equal_to_64_l2035_203580


namespace ab_gt_c_l2035_203550

theorem ab_gt_c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 4 / b = 1) (hc : c < 9) : a + b > c :=
sorry

end ab_gt_c_l2035_203550


namespace min_additional_matchsticks_needed_l2035_203561

-- Define the number of matchsticks in a 3x7 grid
def matchsticks_in_3x7_grid : Nat := 4 * 7 + 3 * 8

-- Define the number of matchsticks in a 5x5 grid
def matchsticks_in_5x5_grid : Nat := 6 * 5 + 6 * 5

-- Define the minimum number of additional matchsticks required
def additional_matchsticks (matchsticks_in_3x7_grid matchsticks_in_5x5_grid : Nat) : Nat :=
  matchsticks_in_5x5_grid - matchsticks_in_3x7_grid

theorem min_additional_matchsticks_needed :
  additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid = 8 :=
by 
  unfold additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid
  sorry

end min_additional_matchsticks_needed_l2035_203561


namespace find_divisor_l2035_203535

theorem find_divisor : exists d : ℕ, 
  (∀ x : ℕ, x ≥ 10 ∧ x ≤ 1000000 → x % d = 0) ∧ 
  (10 + 999990 * d/111110 = 1000000) ∧
  d = 9 := by
  sorry

end find_divisor_l2035_203535


namespace area_change_l2035_203512

variable (L B : ℝ)

def initial_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.20 * L

def new_breadth (B : ℝ) : ℝ := 0.95 * B

def new_area (L B : ℝ) : ℝ := (new_length L) * (new_breadth B)

theorem area_change (L B : ℝ) : new_area L B = 1.14 * (initial_area L B) := by
  -- Proof goes here
  sorry

end area_change_l2035_203512


namespace simplify_expression_l2035_203539

theorem simplify_expression (a c b : ℝ) (h1 : a > c) (h2 : c ≥ 0) (h3 : b > 0) :
  (a * b^2 * (1 / (a + c)^2 + 1 / (a - c)^2) = a - b) → (2 * a * b = a^2 - c^2) :=
by
  sorry

end simplify_expression_l2035_203539


namespace speed_including_stoppages_l2035_203593

theorem speed_including_stoppages : 
  ∀ (speed_excluding_stoppages : ℝ) (stoppage_minutes_per_hour : ℝ), 
  speed_excluding_stoppages = 65 → 
  stoppage_minutes_per_hour = 15.69 → 
  (speed_excluding_stoppages * (1 - stoppage_minutes_per_hour / 60)) = 47.9025 := 
by intros speed_excluding_stoppages stoppage_minutes_per_hour h1 h2
   sorry

end speed_including_stoppages_l2035_203593


namespace fraction_increase_by_50_percent_l2035_203598

variable (x y : ℝ)
variable (h1 : 0 < y)

theorem fraction_increase_by_50_percent (h2 : 0.6 * x / 0.4 * y = 1.5 * x / y) : 
  1.5 * (x / y) = 1.5 * (x / y) :=
by
  sorry

end fraction_increase_by_50_percent_l2035_203598


namespace rosa_peaches_more_than_apples_l2035_203505

def steven_peaches : ℕ := 17
def steven_apples  : ℕ := 16
def jake_peaches : ℕ := steven_peaches - 6
def jake_apples  : ℕ := steven_apples + 8
def rosa_peaches : ℕ := 3 * jake_peaches
def rosa_apples  : ℕ := steven_apples / 2

theorem rosa_peaches_more_than_apples : rosa_peaches - rosa_apples = 25 := by
  sorry

end rosa_peaches_more_than_apples_l2035_203505


namespace cost_price_toy_l2035_203509

theorem cost_price_toy (selling_price_total : ℝ) (total_toys : ℕ) (gain_toys : ℕ) (sp_per_toy : ℝ) (general_cost : ℝ) :
  selling_price_total = 27300 →
  total_toys = 18 →
  gain_toys = 3 →
  sp_per_toy = selling_price_total / total_toys →
  general_cost = sp_per_toy * total_toys - (sp_per_toy * gain_toys / total_toys) →
    general_cost = 1300 := 
by 
  sorry

end cost_price_toy_l2035_203509


namespace total_earnings_l2035_203536

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def num_phone_repairs : ℕ := 5
def num_laptop_repairs : ℕ := 2
def num_computer_repairs : ℕ := 2

theorem total_earnings :
  phone_repair_cost * num_phone_repairs
  + laptop_repair_cost * num_laptop_repairs
  + computer_repair_cost * num_computer_repairs = 121 := by
  sorry

end total_earnings_l2035_203536


namespace bicycle_cost_l2035_203592

theorem bicycle_cost (CP_A SP_B SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 225) : CP_A = 150 :=
by
  sorry

end bicycle_cost_l2035_203592


namespace initial_population_l2035_203517

/--
Suppose 5% of people in a village died by bombardment,
15% of the remaining population left the village due to fear,
and the population is now reduced to 3294.
Prove that the initial population was 4080.
-/
theorem initial_population (P : ℝ) 
  (H1 : 0.05 * P + 0.15 * (1 - 0.05) * P + 3294 = P) : P = 4080 :=
sorry

end initial_population_l2035_203517


namespace ratio_of_segments_of_hypotenuse_l2035_203576

theorem ratio_of_segments_of_hypotenuse (k : Real) :
  let AB := 3 * k
  let BC := 2 * k
  let AC := Real.sqrt (AB^2 + BC^2)
  ∃ D : Real, 
    let BD := (2 / 3) * D
    let AD := (4 / 9) * D
    let CD := D
    ∀ AD CD, AD / CD = 4 / 9 :=
by
  sorry

end ratio_of_segments_of_hypotenuse_l2035_203576


namespace simplify_and_evaluate_l2035_203501

theorem simplify_and_evaluate (a : ℝ) (h : a = -3 / 2) : 
  (a - 2) * (a + 2) - (a + 2)^2 = -2 := 
by 
  sorry

end simplify_and_evaluate_l2035_203501


namespace not_car_probability_l2035_203546

-- Defining the probabilities of taking different modes of transportation.
def P_train : ℝ := 0.5
def P_car : ℝ := 0.2
def P_plane : ℝ := 0.3

-- Defining the event that these probabilities are for mutually exclusive events
axiom mutually_exclusive_events : P_train + P_car + P_plane = 1

-- Statement of the theorem to prove
theorem not_car_probability : P_train + P_plane = 0.8 := 
by 
  -- Use the definitions and axiom provided
  sorry

end not_car_probability_l2035_203546


namespace unit_digit_2_pow_15_l2035_203594

theorem unit_digit_2_pow_15 : (2^15) % 10 = 8 := by
  sorry

end unit_digit_2_pow_15_l2035_203594


namespace find_possible_values_l2035_203534

noncomputable def complex_values (x y : ℂ) : Prop :=
  (x^2 + y^2) / (x + y) = 4 ∧ (x^4 + y^4) / (x^3 + y^3) = 2

theorem find_possible_values (x y : ℂ) (h : complex_values x y) :
  ∃ z : ℂ, z = (x^6 + y^6) / (x^5 + y^5) ∧ (z = 10 + 2 * Real.sqrt 17 ∨ z = 10 - 2 * Real.sqrt 17) :=
sorry

end find_possible_values_l2035_203534


namespace intersection_eq_l2035_203526

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 - x ≤ 0}

theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end intersection_eq_l2035_203526


namespace num_divisors_of_factorial_9_multiple_3_l2035_203586

-- Define the prime factorization of 9!
def factorial_9 := 2^7 * 3^4 * 5 * 7

-- Define the conditions for the exponents a, b, c, d
def valid_exponents (a b c d : ℕ) : Prop :=
  (0 ≤ a ∧ a ≤ 7) ∧ (1 ≤ b ∧ b ≤ 4) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1)

-- Define the number of valid exponent combinations
def num_valid_combinations : ℕ :=
  8 * 4 * 2 * 2

-- Theorem stating that the number of divisors of 9! that are multiples of 3 is 128
theorem num_divisors_of_factorial_9_multiple_3 : num_valid_combinations = 128 := by
  sorry

end num_divisors_of_factorial_9_multiple_3_l2035_203586


namespace discount_percentage_l2035_203502

theorem discount_percentage (marked_price sale_price cost_price : ℝ) (gain1 gain2 : ℝ)
  (h1 : gain1 = 0.35)
  (h2 : gain2 = 0.215)
  (h3 : sale_price = 30)
  (h4 : cost_price = marked_price / (1 + gain1))
  (h5 : marked_price = cost_price * (1 + gain2)) :
  ((sale_price - marked_price) / sale_price) * 100 = 10.009 :=
sorry

end discount_percentage_l2035_203502


namespace func_passes_through_1_2_l2035_203529

-- Given conditions
variable (a : ℝ) (x : ℝ) (y : ℝ)
variable (h1 : 0 < a) (h2 : a ≠ 1)

-- Definition of the function
noncomputable def func (x : ℝ) : ℝ := a^(x-1) + 1

-- Proof statement
theorem func_passes_through_1_2 : func a 1 = 2 :=
by
  -- proof goes here
  sorry

end func_passes_through_1_2_l2035_203529


namespace num_black_balls_l2035_203542

theorem num_black_balls 
  (R W B : ℕ) 
  (R_eq : R = 30) 
  (prob_white : (W : ℝ) / 100 = 0.47) 
  (total_balls : R + W + B = 100) : B = 23 := 
by 
  sorry

end num_black_balls_l2035_203542


namespace sufficient_but_not_necessary_l2035_203507

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1 → x^2 > 1) ∧ ¬(x^2 > 1 → x < -1) :=
by
  sorry

end sufficient_but_not_necessary_l2035_203507


namespace range_of_m_l2035_203579

def f (x : ℝ) := |x - 3|
def g (x : ℝ) (m : ℝ) := -|x - 7| + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 4 :=
by
  sorry

end range_of_m_l2035_203579


namespace min_weights_needed_l2035_203578

theorem min_weights_needed :
  ∃ (weights : List ℕ), (∀ m : ℕ, 1 ≤ m ∧ m ≤ 100 → ∃ (left right : List ℕ), m = (left.sum - right.sum)) ∧ weights.length = 5 :=
sorry

end min_weights_needed_l2035_203578


namespace property_related_only_to_temperature_l2035_203567

-- The conditions given in the problem
def solubility_of_ammonia_gas (T P : Prop) : Prop := T ∧ P
def ion_product_of_water (T : Prop) : Prop := T
def oxidizing_property_of_pp (T C A : Prop) : Prop := T ∧ C ∧ A
def degree_of_ionization_of_acetic_acid (T C : Prop) : Prop := T ∧ C

-- The statement to prove
theorem property_related_only_to_temperature
  (T P C A : Prop)
  (H1 : solubility_of_ammonia_gas T P)
  (H2 : ion_product_of_water T)
  (H3 : oxidizing_property_of_pp T C A)
  (H4 : degree_of_ionization_of_acetic_acid T C) :
  ∃ T, ion_product_of_water T ∧
        ¬solubility_of_ammonia_gas T P ∧
        ¬oxidizing_property_of_pp T C A ∧
        ¬degree_of_ionization_of_acetic_acid T C :=
by
  sorry

end property_related_only_to_temperature_l2035_203567


namespace hanoi_moves_correct_l2035_203577

def hanoi_moves (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * hanoi_moves (n - 1) + 1

theorem hanoi_moves_correct (n : ℕ) : hanoi_moves n = 2^n - 1 := by
  sorry

end hanoi_moves_correct_l2035_203577


namespace area_of_triangle_PQR_l2035_203506

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := -4, y := 2 }
def Q : Point := { x := 8, y := 2 }
def R : Point := { x := 6, y := -4 }

noncomputable def triangle_area (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangle_area P Q R = 36 := by
  sorry

end area_of_triangle_PQR_l2035_203506


namespace monthly_income_of_P_l2035_203595

theorem monthly_income_of_P (P Q R : ℝ) 
    (h1 : (P + Q) / 2 = 2050) 
    (h2 : (Q + R) / 2 = 5250) 
    (h3 : (P + R) / 2 = 6200) : 
    P = 3000 :=
by
  sorry

end monthly_income_of_P_l2035_203595


namespace find_m_direct_proportion_l2035_203515

theorem find_m_direct_proportion (m : ℝ) (h1 : m^2 - 3 = 1) (h2 : m ≠ 2) : m = -2 :=
by {
  -- here would be the proof, but it's omitted as per instructions
  sorry
}

end find_m_direct_proportion_l2035_203515


namespace income_exceeds_previous_l2035_203589

noncomputable def a_n (a b : ℝ) (n : ℕ) : ℝ :=
if n = 1 then a
else a * (2 / 3)^(n - 1) + b * (3 / 2)^(n - 2)

theorem income_exceeds_previous (a b : ℝ) (h : b ≥ 3 * a / 8) (n : ℕ) (hn : n ≥ 2) : 
  a_n a b n ≥ a :=
sorry

end income_exceeds_previous_l2035_203589


namespace A_finishes_work_in_8_days_l2035_203533

theorem A_finishes_work_in_8_days 
  (A_work B_work W : ℝ) 
  (h1 : 4 * A_work + 6 * B_work = W)
  (h2 : (A_work + B_work) * 4.8 = W) :
  A_work = W / 8 :=
by
  -- We should provide the proof here, but we will use "sorry" for now.
  sorry

end A_finishes_work_in_8_days_l2035_203533


namespace women_fraction_l2035_203557

/-- In a room with 100 people, 1/4 of whom are married, the maximum number of unmarried women is 40.
    We need to prove that the fraction of women in the room is 2/5. -/
theorem women_fraction (total_people : ℕ) (married_fraction : ℚ) (unmarried_women : ℕ) (W : ℚ) 
  (h1 : total_people = 100) 
  (h2 : married_fraction = 1 / 4) 
  (h3 : unmarried_women = 40) 
  (hW : W = 2 / 5) : 
  W = 2 / 5 := 
by
  sorry

end women_fraction_l2035_203557


namespace factorial_division_l2035_203519

theorem factorial_division :
  (Nat.factorial 4) / (Nat.factorial (4 - 3)) = 24 :=
by
  sorry

end factorial_division_l2035_203519


namespace smallest_number_l2035_203582

-- Define the conditions
def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def conditions (n : ℕ) : Prop := 
  (n > 12) ∧ 
  is_divisible_by (n - 12) 12 ∧ 
  is_divisible_by (n - 12) 24 ∧
  is_divisible_by (n - 12) 36 ∧
  is_divisible_by (n - 12) 48 ∧
  is_divisible_by (n - 12) 56

-- State the theorem
theorem smallest_number : ∃ n : ℕ, conditions n ∧ n = 1020 :=
by
  sorry

end smallest_number_l2035_203582


namespace isabella_hair_length_l2035_203599

theorem isabella_hair_length (h : ℕ) (g : ℕ) (future_length : ℕ) (hg : g = 4) (future_length_eq : future_length = 22) :
  h = future_length - g :=
by
  rw [future_length_eq, hg]
  exact sorry

end isabella_hair_length_l2035_203599


namespace count_integers_P_leq_0_l2035_203560

def P(x : ℤ) : ℤ := 
  (x - 1^3) * (x - 2^3) * (x - 3^3) * (x - 4^3) * (x - 5^3) *
  (x - 6^3) * (x - 7^3) * (x - 8^3) * (x - 9^3) * (x - 10^3) *
  (x - 11^3) * (x - 12^3) * (x - 13^3) * (x - 14^3) * (x - 15^3) *
  (x - 16^3) * (x - 17^3) * (x - 18^3) * (x - 19^3) * (x - 20^3) *
  (x - 21^3) * (x - 22^3) * (x - 23^3) * (x - 24^3) * (x - 25^3) *
  (x - 26^3) * (x - 27^3) * (x - 28^3) * (x - 29^3) * (x - 30^3) *
  (x - 31^3) * (x - 32^3) * (x - 33^3) * (x - 34^3) * (x - 35^3) *
  (x - 36^3) * (x - 37^3) * (x - 38^3) * (x - 39^3) * (x - 40^3) *
  (x - 41^3) * (x - 42^3) * (x - 43^3) * (x - 44^3) * (x - 45^3) *
  (x - 46^3) * (x - 47^3) * (x - 48^3) * (x - 49^3) * (x - 50^3)

theorem count_integers_P_leq_0 : 
  ∃ n : ℕ, n = 15650 ∧ ∀ k : ℤ, (P k ≤ 0) → (n = 15650) :=
by sorry

end count_integers_P_leq_0_l2035_203560


namespace james_pays_660_for_bed_and_frame_l2035_203540

theorem james_pays_660_for_bed_and_frame :
  let bed_frame_price := 75
  let bed_price := 10 * bed_frame_price
  let total_price_before_discount := bed_frame_price + bed_price
  let discount := 0.20 * total_price_before_discount
  let final_price := total_price_before_discount - discount
  final_price = 660 := 
by
  sorry

end james_pays_660_for_bed_and_frame_l2035_203540


namespace discount_difference_is_correct_l2035_203551

-- Define the successive discounts in percentage
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10

-- Define the store's claimed discount
def claimed_discount : ℝ := 0.45

-- Calculate the true discount
def true_discount : ℝ := 1 - ((1 - discount1) * (1 - discount2) * (1 - discount3))

-- Calculate the difference between the true discount and the claimed discount
def discount_difference : ℝ := claimed_discount - true_discount

-- State the theorem to be proved
theorem discount_difference_is_correct : discount_difference = 2.375 / 100 := by
  sorry

end discount_difference_is_correct_l2035_203551


namespace part_1_property_part_2_property_part_3_geometric_l2035_203520

-- Defining properties
def prop1 (a : ℕ → ℕ) (i j m: ℕ) : Prop := i > j ∧ (a i)^2 / (a j) = a m
def prop2 (a : ℕ → ℕ) (n k l: ℕ) : Prop := n ≥ 3 ∧ k > l ∧ (a n) = (a k)^2 / (a l)

-- Part I: Sequence {a_n = n} check for property 1
theorem part_1_property (a : ℕ → ℕ) (h : ∀ n, a n = n) : ¬∃ i j m, prop1 a i j m := by
  sorry

-- Part II: Sequence {a_n = 2^(n-1)} check for property 1 and 2
theorem part_2_property (a : ℕ → ℕ) (h : ∀ n, a n = 2^(n-1)) : 
  (∀ i j, ∃ m, prop1 a i j m) ∧ (∀ n k l, prop2 a n k l) := by
  sorry

-- Part III: Increasing sequence that satisfies both properties is a geometric sequence
theorem part_3_geometric (a : ℕ → ℕ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_prop1 : ∀ i j, i > j → ∃ m, prop1 a i j m)
  (h_prop2 : ∀ n, n ≥ 3 → ∃ k l, k > l ∧ (a n) = (a k)^2 / (a l)) : 
  ∃ r, ∀ n, a (n + 1) = r * a n := by
  sorry

end part_1_property_part_2_property_part_3_geometric_l2035_203520


namespace sixth_inequality_l2035_203572

theorem sixth_inequality :
  (1 + 1/2^2 + 1/3^2 + 1/4^2 + 1/5^2 + 1/6^2 + 1/7^2) < 13/7 :=
  sorry

end sixth_inequality_l2035_203572


namespace max_value_of_h_l2035_203575

noncomputable def f (x : ℝ) : ℝ := -x + 3
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := min (f x) (g x)

theorem max_value_of_h : ∃ x : ℝ, h x = 1 :=
by
  sorry

end max_value_of_h_l2035_203575


namespace triangle_area_difference_l2035_203565

-- Definitions based on given lengths and right angles.
def GH : ℝ := 5
def HI : ℝ := 7
def FG : ℝ := 9

-- Note: Right angles are implicitly used in the area calculations and do not need to be represented directly in Lean.
-- Define areas for triangles involved.
def area_FGH : ℝ := 0.5 * FG * GH
def area_GHI : ℝ := 0.5 * GH * HI
def area_FHI : ℝ := 0.5 * FG * HI

-- Define areas of the triangles FGJ and HJI using variables.
variable (x y z : ℝ)
axiom area_FGJ : x = area_FHI - z
axiom area_HJI : y = area_GHI - z

-- The main proof statement involving the difference.
theorem triangle_area_difference : (x - y) = 14 := by
  sorry

end triangle_area_difference_l2035_203565
