import Mathlib

namespace set_P_equality_l1246_124605

open Set

variable {U : Set ℝ} (P : Set ℝ)
variable (h_univ : U = univ) (h_def : P = {x | abs (x - 2) ≥ 1})

theorem set_P_equality : P = {x | x ≥ 3 ∨ x ≤ 1} :=
by
  sorry

end set_P_equality_l1246_124605


namespace distance_between_B_and_C_l1246_124618

theorem distance_between_B_and_C
  (A B C : Type)
  (AB : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (h_AB : AB = 10)
  (h_angle_A : angle_A = 60)
  (h_angle_B : angle_B = 75) :
  ∃ BC : ℝ, BC = 5 * Real.sqrt 6 :=
by
  sorry

end distance_between_B_and_C_l1246_124618


namespace find_certain_number_l1246_124696

theorem find_certain_number (x : ℝ) (h : 0.7 * x = 28) : x = 40 := 
by
  sorry

end find_certain_number_l1246_124696


namespace prime_triples_l1246_124699

open Nat

theorem prime_triples (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) :
    (p ∣ q^r + 1) → (q ∣ r^p + 1) → (r ∣ p^q + 1) → (p, q, r) = (2, 5, 3) ∨ (p, q, r) = (3, 2, 5) ∨ (p, q, r) = (5, 3, 2) :=
  by
  sorry

end prime_triples_l1246_124699


namespace triangle_side_lengths_condition_l1246_124692

noncomputable def f (x k : ℝ) : ℝ := (x^2 + k*x + 1) / (x^2 + x + 1)

theorem triangle_side_lengths_condition (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, x1 > 0 → x2 > 0 → x3 > 0 →
    (f x1 k) + (f x2 k) > (f x3 k) ∧ (f x2 k) + (f x3 k) > (f x1 k) ∧ (f x3 k) + (f x1 k) > (f x2 k))
  ↔ (-1/2 ≤ k ∧ k ≤ 4) :=
by
  sorry

end triangle_side_lengths_condition_l1246_124692


namespace problem_statement_l1246_124650

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom f_pos (x : ℝ) : x > 0 → f x > 0
axiom f'_less_f (x : ℝ) : f' x < f x
axiom f_has_deriv_at : ∀ x, HasDerivAt f (f' x) x

def a : ℝ := sorry
axiom a_in_range : 0 < a ∧ a < 1

theorem problem_statement : 3 * f 0 > f a ∧ f a > a * f 1 :=
  sorry

end problem_statement_l1246_124650


namespace intersection_A_B_l1246_124617

-- Define set A and its condition
def A : Set ℝ := { y | ∃ (x : ℝ), y = x^2 }

-- Define set B and its condition
def B : Set ℝ := { x | ∃ (y : ℝ), y = Real.sqrt (1 - x^2) }

-- Define the set intersection A ∩ B
def A_intersect_B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- The theorem statement
theorem intersection_A_B :
  A ∩ B = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end intersection_A_B_l1246_124617


namespace product_expression_l1246_124669

theorem product_expression :
  (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) * (7^4 - 1) / (7^4 + 1) = 880 / 91 := by
sorry

end product_expression_l1246_124669


namespace clown_balloon_count_l1246_124665

theorem clown_balloon_count (b1 b2 : ℕ) (h1 : b1 = 47) (h2 : b2 = 13) : b1 + b2 = 60 := by
  sorry

end clown_balloon_count_l1246_124665


namespace find_constants_l1246_124626

variables {A B C x : ℝ}

theorem find_constants (h : (A = 6) ∧ (B = -5) ∧ (C = 5)) :
  (x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1) :=
by sorry

end find_constants_l1246_124626


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l1246_124654

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0) → n ≤ m :=
sorry

end smallest_positive_integer_ends_in_3_divisible_by_11_l1246_124654


namespace tangent_inclination_point_l1246_124623

theorem tangent_inclination_point :
  ∃ a : ℝ, (2 * a = 1) ∧ ((a, a^2) = (1 / 2, 1 / 4)) :=
by
  sorry

end tangent_inclination_point_l1246_124623


namespace one_point_one_billion_in_scientific_notation_l1246_124612

noncomputable def one_point_one_billion : ℝ := 1.1 * 10^9

theorem one_point_one_billion_in_scientific_notation :
  1.1 * 10^9 = 1100000000 :=
by
  sorry

end one_point_one_billion_in_scientific_notation_l1246_124612


namespace min_value_4x_plus_inv_l1246_124637

noncomputable def min_value_function (x : ℝ) := 4 * x + 1 / (4 * x - 5)

theorem min_value_4x_plus_inv (x : ℝ) (h : x > 5 / 4) : min_value_function x = 7 :=
sorry

end min_value_4x_plus_inv_l1246_124637


namespace three_seventy_five_as_fraction_l1246_124664

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end three_seventy_five_as_fraction_l1246_124664


namespace arithmetic_expression_evaluation_l1246_124633

theorem arithmetic_expression_evaluation : 
  (5 * 7 - (3 * 2 + 5 * 4) / 2) = 22 := 
by
  sorry

end arithmetic_expression_evaluation_l1246_124633


namespace discount_allowed_l1246_124698

-- Define the conditions
def CP : ℝ := 100 -- Cost Price (CP) is $100 for simplicity
def MP : ℝ := CP + 0.12 * CP -- Selling price marked 12% above cost price
def Loss : ℝ := 0.01 * CP -- Trader suffers a loss of 1% on CP
def SP : ℝ := CP - Loss -- Selling price after suffering the loss

-- State the equivalent proof problem in Lean
theorem discount_allowed : MP - SP = 13 := by
  sorry

end discount_allowed_l1246_124698


namespace avg_page_count_per_essay_l1246_124677

-- Definitions based on conditions
def students : ℕ := 15
def first_group_students : ℕ := 5
def second_group_students : ℕ := 5
def third_group_students : ℕ := 5
def pages_first_group : ℕ := 2
def pages_second_group : ℕ := 3
def pages_third_group : ℕ := 1
def total_pages := first_group_students * pages_first_group +
                   second_group_students * pages_second_group +
                   third_group_students * pages_third_group

-- Statement to prove
theorem avg_page_count_per_essay :
  total_pages / students = 2 :=
by
  sorry

end avg_page_count_per_essay_l1246_124677


namespace jackson_volume_discount_l1246_124662

-- Given conditions as parameters
def hotTubVolume := 40 -- gallons
def quartsPerGallon := 4 -- quarts per gallon
def bottleVolume := 1 -- quart per bottle
def bottleCost := 50 -- dollars per bottle
def totalSpent := 6400 -- dollars spent by Jackson

-- Calculation related definitions
def totalQuarts := hotTubVolume * quartsPerGallon
def totalBottles := totalQuarts / bottleVolume
def costWithoutDiscount := totalBottles * bottleCost
def discountAmount := costWithoutDiscount - totalSpent
def discountPercentage := (discountAmount / costWithoutDiscount) * 100

-- The proof problem
theorem jackson_volume_discount : discountPercentage = 20 :=
by
  sorry

end jackson_volume_discount_l1246_124662


namespace find_principal_amount_l1246_124656

variables (P R : ℝ)

theorem find_principal_amount (h : (4 * P * (R + 2) / 100) - (4 * P * R / 100) = 56) : P = 700 :=
sorry

end find_principal_amount_l1246_124656


namespace theta_in_second_quadrant_l1246_124691

theorem theta_in_second_quadrant
  (θ : ℝ)
  (h1 : Real.sin θ > 0)
  (h2 : Real.tan θ < 0) :
  (π / 2 < θ) ∧ (θ < π) :=
by
  sorry

end theta_in_second_quadrant_l1246_124691


namespace probability_diff_colors_l1246_124697

theorem probability_diff_colors (total_balls red_balls white_balls selected_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_red : red_balls = 2)
  (h_white : white_balls = 2)
  (h_selected : selected_balls = 2) :
  (∃ P : ℚ, P = (red_balls.choose (selected_balls / 2) * white_balls.choose (selected_balls / 2)) / total_balls.choose selected_balls ∧ P = 2 / 3) :=
by 
  sorry

end probability_diff_colors_l1246_124697


namespace rollo_guinea_pigs_food_l1246_124683

theorem rollo_guinea_pigs_food :
  let first_food := 2
  let second_food := 2 * first_food
  let third_food := second_food + 3
  first_food + second_food + third_food = 13 :=
by
  sorry

end rollo_guinea_pigs_food_l1246_124683


namespace distance_between_foci_of_ellipse_l1246_124667

theorem distance_between_foci_of_ellipse :
  let F1 := (4, -3)
  let F2 := (-6, 9)
  let distance := Real.sqrt ( ((4 - (-6))^2) + ((-3 - 9)^2) )
  distance = 2 * Real.sqrt 61 :=
by
  let F1 := (4, -3)
  let F2 := (-6, 9)
  let distance := Real.sqrt ( ((4 - (-6))^2) + ((-3 - 9)^2) )
  sorry

end distance_between_foci_of_ellipse_l1246_124667


namespace sandwiches_prepared_l1246_124651

-- Define the conditions as given in the problem.
def ruth_ate_sandwiches : ℕ := 1
def brother_ate_sandwiches : ℕ := 2
def first_cousin_ate_sandwiches : ℕ := 2
def each_other_cousin_ate_sandwiches : ℕ := 1
def number_of_other_cousins : ℕ := 2
def sandwiches_left : ℕ := 3

-- Define the total number of sandwiches eaten.
def total_sandwiches_eaten : ℕ := ruth_ate_sandwiches 
                                  + brother_ate_sandwiches
                                  + first_cousin_ate_sandwiches 
                                  + (each_other_cousin_ate_sandwiches * number_of_other_cousins)

-- Define the number of sandwiches prepared by Ruth.
def sandwiches_prepared_by_ruth : ℕ := total_sandwiches_eaten + sandwiches_left

-- Formulate the theorem to prove.
theorem sandwiches_prepared : sandwiches_prepared_by_ruth = 10 :=
by
  -- Use the solution steps to prove the theorem (proof omitted here).
  sorry

end sandwiches_prepared_l1246_124651


namespace solution_set_of_inequality_l1246_124640

theorem solution_set_of_inequality (x : ℝ) : (|x + 1| - |x - 3| ≥ 0) ↔ (1 ≤ x) := 
sorry

end solution_set_of_inequality_l1246_124640


namespace fifth_house_number_is_13_l1246_124643

theorem fifth_house_number_is_13 (n : ℕ) (a₁ : ℕ) (h₀ : n ≥ 5) (h₁ : (a₁ + n - 1) * n = 117) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n -> (a₁ + 2 * (i - 1)) = 2*(i-1) + a₁) : 
  (a₁ + 2 * (5 - 1)) = 13 :=
by
  sorry

end fifth_house_number_is_13_l1246_124643


namespace second_character_more_lines_l1246_124642

theorem second_character_more_lines
  (C1 : ℕ) (S : ℕ) (T : ℕ) (X : ℕ)
  (h1 : C1 = 20)
  (h2 : C1 = S + 8)
  (h3 : T = 2)
  (h4 : S = 3 * T + X) :
  X = 6 :=
by
  -- proof can be filled in here
  sorry

end second_character_more_lines_l1246_124642


namespace initial_amount_celine_had_l1246_124621

-- Define the costs and quantities
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def change_received : ℕ := 200

-- Calculate costs and total amount
def cost_laptops : ℕ := num_laptops * laptop_cost
def cost_smartphones : ℕ := num_smartphones * smartphone_cost
def total_cost : ℕ := cost_laptops + cost_smartphones
def initial_amount : ℕ := total_cost + change_received

-- The statement to prove
theorem initial_amount_celine_had : initial_amount = 3000 := by
  sorry

end initial_amount_celine_had_l1246_124621


namespace exists_four_functions_l1246_124629

theorem exists_four_functions 
  (f : ℝ → ℝ)
  (h_periodic : ∀ x, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
sorry

end exists_four_functions_l1246_124629


namespace find_number_of_friends_l1246_124657

def dante_balloons : Prop :=
  ∃ F : ℕ, (F > 0 ∧ (250 / F) - 11 = 39) ∧ F = 5

theorem find_number_of_friends : dante_balloons :=
by
  sorry

end find_number_of_friends_l1246_124657


namespace fraction_inequality_l1246_124619

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  b / (a - c) < a / (b - d) :=
sorry

end fraction_inequality_l1246_124619


namespace remainder_when_divided_l1246_124661

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 + x^3 + 1

-- The statement to be proved
theorem remainder_when_divided (x : ℝ) : (p 2) = 25 :=
by
  sorry

end remainder_when_divided_l1246_124661


namespace M_inter_N_eq_l1246_124615

open Set

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {3, 4, 5, 6}

theorem M_inter_N_eq : M ∩ N = {3, 4} := 
by 
  sorry

end M_inter_N_eq_l1246_124615


namespace rectangular_prism_cut_corners_edges_l1246_124604

def original_edges : Nat := 12
def corners : Nat := 8
def new_edges_per_corner : Nat := 3
def total_new_edges : Nat := corners * new_edges_per_corner

theorem rectangular_prism_cut_corners_edges :
  original_edges + total_new_edges = 36 := sorry

end rectangular_prism_cut_corners_edges_l1246_124604


namespace xy_div_eq_one_third_l1246_124614

theorem xy_div_eq_one_third (x y z : ℝ) 
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y / z = 6) : 
  x / y = 1 / 3 :=
by
  sorry

end xy_div_eq_one_third_l1246_124614


namespace sum_of_numbers_l1246_124686

theorem sum_of_numbers (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : ab + bc + ca = 100) :
  a + b + c = 21 :=
sorry

end sum_of_numbers_l1246_124686


namespace at_least_one_less_than_two_l1246_124607

theorem at_least_one_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 2) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
sorry

end at_least_one_less_than_two_l1246_124607


namespace soccer_league_fraction_female_proof_l1246_124687

variable (m f : ℝ)

def soccer_league_fraction_female : Prop :=
  let males_last_year := m
  let females_last_year := f
  let males_this_year := 1.05 * m
  let females_this_year := 1.2 * f
  let total_this_year := 1.1 * (m + f)
  (1.05 * m + 1.2 * f = 1.1 * (m + f)) → ((0.6 * m) / (1.65 * m) = 4 / 11)

theorem soccer_league_fraction_female_proof (m f : ℝ) : soccer_league_fraction_female m f :=
by {
  sorry
}

end soccer_league_fraction_female_proof_l1246_124687


namespace polynomial_roots_product_l1246_124653

theorem polynomial_roots_product (a b : ℤ)
  (h1 : ∀ (r : ℝ), r^2 - r - 2 = 0 → r^3 - a * r - b = 0) : a * b = 6 := sorry

end polynomial_roots_product_l1246_124653


namespace find_M_l1246_124675

theorem find_M : ∀ M : ℕ, (10 + 11 + 12 : ℕ) / 3 = (2024 + 2025 + 2026 : ℕ) / M → M = 552 :=
by
  intro M
  sorry

end find_M_l1246_124675


namespace percentage_increase_efficiency_l1246_124638

-- Defining the times taken by Sakshi and Tanya
def sakshi_time : ℕ := 12
def tanya_time : ℕ := 10

-- Defining the efficiency in terms of work per day for Sakshi and Tanya
def sakshi_efficiency : ℚ := 1 / sakshi_time
def tanya_efficiency : ℚ := 1 / tanya_time

-- The statement of the proof: percentage increase
theorem percentage_increase_efficiency : 
  100 * ((tanya_efficiency - sakshi_efficiency) / sakshi_efficiency) = 20 := 
by
  -- The actual proof will go here
  sorry

end percentage_increase_efficiency_l1246_124638


namespace value_of_g_at_five_l1246_124682

def g (x : ℕ) : ℕ := x^2 - 2 * x

theorem value_of_g_at_five : g 5 = 15 := by
  sorry

end value_of_g_at_five_l1246_124682


namespace hyperbola_standard_equations_l1246_124631

-- Definitions derived from conditions
def focal_distance (c : ℝ) : Prop := c = 8
def eccentricity (e : ℝ) : Prop := e = 4 / 3
def equilateral_focus (c : ℝ) : Prop := c^2 = 36

-- Theorem stating the standard equations given the conditions
noncomputable def hyperbola_equation1 (y2 : ℝ) (x2 : ℝ) : Prop :=
y2 / 36 - x2 / 28 = 1

noncomputable def hyperbola_equation2 (x2 : ℝ) (y2 : ℝ) : Prop :=
x2 / 18 - y2 / 18 = 1

theorem hyperbola_standard_equations
  (c y2 x2 : ℝ)
  (c_focus : focal_distance c)
  (e_value : eccentricity (4 / 3))
  (equi_focus : equilateral_focus c) :
  hyperbola_equation1 y2 x2 ∧ hyperbola_equation2 x2 y2 :=
by
  sorry

end hyperbola_standard_equations_l1246_124631


namespace point_P_in_third_quadrant_l1246_124620

def point_in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_P_in_third_quadrant :
  point_in_third_quadrant (-3) (-2) :=
by
  sorry -- Proof of the statement, as per the steps given.

end point_P_in_third_quadrant_l1246_124620


namespace geometric_arithmetic_sequence_l1246_124647

theorem geometric_arithmetic_sequence 
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (q : ℝ) 
  (h0 : 0 < q) (h1 : q ≠ 1)
  (h2 : ∀ n, a_n n = a_n 1 * q ^ (n - 1)) -- a_n is a geometric sequence
  (h3 : 2 * a_n 3 * a_n 5 = a_n 4 * (a_n 3 + a_n 5)) -- a3, a5, a4 form an arithmetic sequence
  (h4 : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) -- S_n is the sum of the first n terms
  : S 6 / S 3 = 9 / 8 :=
by
  sorry

end geometric_arithmetic_sequence_l1246_124647


namespace ratio_surface_area_l1246_124671

noncomputable def side_length (a : ℝ) := a
noncomputable def radius (R : ℝ) := R

theorem ratio_surface_area (a R : ℝ) (h : a^3 = (4/3) * Real.pi * R^3) : 
  (6 * a^2) / (4 * Real.pi * R^2) = (3 * (6 / Real.pi)) :=
by sorry

end ratio_surface_area_l1246_124671


namespace Joe_total_time_correct_l1246_124695

theorem Joe_total_time_correct :
  ∀ (distance : ℝ) (walk_rate : ℝ) (bike_rate : ℝ) (walk_time bike_time : ℝ),
    (walk_time = 9) →
    (bike_rate = 5 * walk_rate) →
    (walk_rate * walk_time = distance / 3) →
    (bike_rate * bike_time = 2 * distance / 3) →
    (walk_time + bike_time = 12.6) := 
by
  intros distance walk_rate bike_rate walk_time bike_time
  intro walk_time_cond
  intro bike_rate_cond
  intro walk_distance_cond
  intro bike_distance_cond
  sorry

end Joe_total_time_correct_l1246_124695


namespace remainder_of_3_to_40_plus_5_mod_5_l1246_124634

theorem remainder_of_3_to_40_plus_5_mod_5 : (3^40 + 5) % 5 = 1 :=
by
  sorry

end remainder_of_3_to_40_plus_5_mod_5_l1246_124634


namespace min_possible_value_box_l1246_124609

theorem min_possible_value_box (a b : ℤ) (h_ab : a * b = 35) : a^2 + b^2 ≥ 74 := sorry

end min_possible_value_box_l1246_124609


namespace billy_total_problems_solved_l1246_124680

theorem billy_total_problems_solved :
  ∃ (Q : ℕ), (3 * Q = 132) ∧ ((Q) + (2 * Q) + (3 * Q) = 264) :=
by
  sorry

end billy_total_problems_solved_l1246_124680


namespace intersection_point_in_polar_coordinates_l1246_124606

theorem intersection_point_in_polar_coordinates (theta : ℝ) (rho : ℝ) (h₁ : theta = π / 3) (h₂ : rho = 2 * Real.cos theta) (h₃ : rho > 0) : rho = 1 :=
by
  -- Proof skipped
  sorry

end intersection_point_in_polar_coordinates_l1246_124606


namespace miles_tankful_highway_l1246_124679

variable (miles_tankful_city : ℕ)
variable (mpg_city : ℕ)
variable (mpg_highway : ℕ)

-- Relationship between miles per gallon in city and highway
axiom h_mpg_relation : mpg_highway = mpg_city + 18

-- Given the car travels 336 miles per tankful of gasoline in the city
axiom h_miles_tankful_city : miles_tankful_city = 336

-- Given the car travels 48 miles per gallon in the city
axiom h_mpg_city : mpg_city = 48

-- Prove the car travels 462 miles per tankful of gasoline on the highway
theorem miles_tankful_highway : ∃ (miles_tankful_highway : ℕ), miles_tankful_highway = (mpg_highway * (miles_tankful_city / mpg_city)) := 
by 
  exists (66 * (336 / 48)) -- Since 48 + 18 = 66 and 336 / 48 = 7, 66 * 7 = 462
  sorry

end miles_tankful_highway_l1246_124679


namespace math_contest_students_l1246_124668

theorem math_contest_students (n : ℝ) (h : n / 3 + n / 4 + n / 5 + 26 = n) : n = 120 :=
by {
    sorry
}

end math_contest_students_l1246_124668


namespace largest_n_exists_l1246_124611

theorem largest_n_exists :
  ∃ (n : ℕ), (∃ (x y z : ℕ), n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) ∧
    ∀ (m : ℕ), (∃ (x y z : ℕ), m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) →
    n ≥ m :=
  sorry

end largest_n_exists_l1246_124611


namespace jessica_carrots_l1246_124602

theorem jessica_carrots
  (joan_carrots : ℕ)
  (total_carrots : ℕ)
  (jessica_carrots : ℕ) :
  joan_carrots = 29 →
  total_carrots = 40 →
  jessica_carrots = total_carrots - joan_carrots →
  jessica_carrots = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jessica_carrots_l1246_124602


namespace palm_trees_in_forest_l1246_124676

variable (F D : ℕ)

theorem palm_trees_in_forest 
  (h1 : D = 2 * F / 5)
  (h2 : D + F = 7000) :
  F = 5000 := by
  sorry

end palm_trees_in_forest_l1246_124676


namespace combined_weight_of_Alexa_and_Katerina_l1246_124628

variable (total_weight: ℝ) (alexas_weight: ℝ) (michaels_weight: ℝ)

theorem combined_weight_of_Alexa_and_Katerina
  (h1: total_weight = 154)
  (h2: alexas_weight = 46)
  (h3: michaels_weight = 62) :
  total_weight - michaels_weight = 92 :=
by 
  sorry

end combined_weight_of_Alexa_and_Katerina_l1246_124628


namespace cost_of_each_soda_l1246_124635

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l1246_124635


namespace cos_identity_l1246_124610

theorem cos_identity (α : ℝ) : 
  3.4028 * (Real.cos α)^4 + 4 * (Real.cos α)^3 - 8 * (Real.cos α)^2 - 3 * (Real.cos α) + 1 = 
  2 * (Real.cos (7 * α / 2)) * (Real.cos (α / 2)) := 
by sorry

end cos_identity_l1246_124610


namespace rectangle_to_square_l1246_124663

theorem rectangle_to_square (a b : ℝ) (h1 : b / 2 < a) (h2 : a < b) :
  ∃ (r : ℝ), r = Real.sqrt (a * b) ∧ 
    (∃ (cut1 cut2 : ℝ × ℝ), 
      cut1.1 = 0 ∧ cut1.2 = a ∧
      cut2.1 = b - r ∧ cut2.2 = r - a ∧
      ∀ t, t = (a * b) - (r ^ 2)) := sorry

end rectangle_to_square_l1246_124663


namespace least_possible_value_l1246_124685

theorem least_possible_value (x : ℚ) (h1 : x > 5 / 3) (h2 : x < 9 / 2) : 
  (9 / 2 - 5 / 3 : ℚ) = 17 / 6 :=
by sorry

end least_possible_value_l1246_124685


namespace boat_speed_in_still_water_l1246_124616

def speed_of_boat (V_b : ℝ) : Prop :=
  let stream_speed := 4  -- speed of the stream in km/hr
  let downstream_distance := 168  -- distance traveled downstream in km
  let time := 6  -- time taken to travel downstream in hours
  (downstream_distance = (V_b + stream_speed) * time)

theorem boat_speed_in_still_water : ∃ V_b, speed_of_boat V_b ∧ V_b = 24 := 
by
  exists 24
  unfold speed_of_boat
  simp
  sorry

end boat_speed_in_still_water_l1246_124616


namespace necessary_but_not_sufficient_condition_l1246_124622

theorem necessary_but_not_sufficient_condition
  (a : ℝ)
  (h : ∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) :
  (a < 2 ∧ a < 3) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1246_124622


namespace original_length_before_final_cut_l1246_124648

-- Defining the initial length of the board
def initial_length : ℕ := 143

-- Defining the length after the first cut
def length_after_first_cut : ℕ := initial_length - 25

-- Defining the length after the final cut
def length_after_final_cut : ℕ := length_after_first_cut - 7

-- Stating the theorem to prove that the original length of the board before cutting the final 7 cm is 125 cm
theorem original_length_before_final_cut : initial_length - 25 + 7 = 125 :=
sorry

end original_length_before_final_cut_l1246_124648


namespace garden_area_l1246_124649

theorem garden_area (P b l: ℕ) (hP: P = 900) (hb: b = 190) (hl: l = P / 2 - b):
  l * b = 49400 := 
by
  sorry

end garden_area_l1246_124649


namespace range_of_a_l1246_124624

theorem range_of_a (x a : ℝ) (hp : x^2 + 2 * x - 3 > 0) (hq : x > a)
  (h_suff : x^2 + 2 * x - 3 > 0 → ¬ (x > a)):
  a ≥ 1 := 
by
  sorry

end range_of_a_l1246_124624


namespace find_a_if_circle_l1246_124608

noncomputable def curve_eq (a x y : ℝ) : ℝ :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a

def is_circle_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, curve_eq a x y = 0 → (∃ k : ℝ, curve_eq a x y = k * (x^2 + y^2))

theorem find_a_if_circle :
  (∀ a : ℝ, is_circle_condition a → a = -1) :=
by
  sorry

end find_a_if_circle_l1246_124608


namespace plus_signs_count_l1246_124678

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l1246_124678


namespace problem_one_problem_two_l1246_124603

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x + 4

-- Problem (I)
theorem problem_one (m : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → f x m < 0) ↔ m ≤ -5 :=
sorry

-- Problem (II)
theorem problem_two (m : ℝ) :
  (∀ x, (x = 1 ∨ x = 2) → abs ((f x m - x^2) / m) < 1) ↔ (-4 < m ∧ m ≤ -2) :=
sorry

end problem_one_problem_two_l1246_124603


namespace part1_part2_l1246_124690

open Real

noncomputable def f (x : ℝ) : ℝ := abs ((2 / 3) * x + 1)

theorem part1 (a : ℝ) : (∀ x, f x ≥ -abs x + a) → a ≤ 1 :=
sorry

theorem part2 (x y : ℝ) (h1 : abs (x + y + 1) ≤ 1 / 3) (h2 : abs (y - 1 / 3) ≤ 2 / 3) : 
  f x ≤ 7 / 9 :=
sorry

end part1_part2_l1246_124690


namespace number_of_geese_l1246_124627

theorem number_of_geese (A x n k : ℝ) 
  (h1 : A = k * x * n)
  (h2 : A = (k + 20) * x * (n - 75))
  (h3 : A = (k - 15) * x * (n + 100)) 
  : n = 300 :=
sorry

end number_of_geese_l1246_124627


namespace rectangle_width_decrease_l1246_124689

theorem rectangle_width_decrease {L W : ℝ} (A : ℝ) (hA : A = L * W) (h_new_length : A = 1.25 * L * (W * y)) : y = 0.8 :=
by sorry

end rectangle_width_decrease_l1246_124689


namespace find_m_l1246_124613

theorem find_m {m : ℝ} :
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end find_m_l1246_124613


namespace aliyah_more_phones_l1246_124641

theorem aliyah_more_phones (vivi_phones : ℕ) (phone_price : ℕ) (total_money : ℕ) (aliyah_more : ℕ) : 
  vivi_phones = 40 → 
  phone_price = 400 → 
  total_money = 36000 → 
  40 + 40 + aliyah_more = total_money / phone_price → 
  aliyah_more = 10 :=
sorry

end aliyah_more_phones_l1246_124641


namespace apple_pies_l1246_124600

theorem apple_pies (total_apples not_ripe_apples apples_per_pie : ℕ) 
    (h1 : total_apples = 34) 
    (h2 : not_ripe_apples = 6) 
    (h3 : apples_per_pie = 4) : 
    (total_apples - not_ripe_apples) / apples_per_pie = 7 :=
by 
    sorry

end apple_pies_l1246_124600


namespace table_area_l1246_124660

theorem table_area (A : ℝ) 
  (combined_area : ℝ)
  (coverage_percentage : ℝ)
  (area_two_layers : ℝ)
  (area_three_layers : ℝ)
  (combined_area_eq : combined_area = 220)
  (coverage_percentage_eq : coverage_percentage = 0.80 * A)
  (area_two_layers_eq : area_two_layers = 24)
  (area_three_layers_eq : area_three_layers = 28) :
  A = 275 :=
by
  -- Assumptions and derivations can be filled in.
  sorry

end table_area_l1246_124660


namespace non_chocolate_candy_count_l1246_124601

theorem non_chocolate_candy_count (total_candy : ℕ) (total_bags : ℕ) 
  (chocolate_hearts_bags : ℕ) (chocolate_kisses_bags : ℕ) (each_bag_pieces : ℕ) 
  (non_chocolate_bags : ℕ) : 
  total_candy = 63 ∧ 
  total_bags = 9 ∧ 
  chocolate_hearts_bags = 2 ∧ 
  chocolate_kisses_bags = 3 ∧ 
  total_candy / total_bags = each_bag_pieces ∧ 
  total_bags - (chocolate_hearts_bags + chocolate_kisses_bags) = non_chocolate_bags ∧ 
  non_chocolate_bags * each_bag_pieces = 28 :=
by
  -- use "sorry" to skip the proof
  sorry

end non_chocolate_candy_count_l1246_124601


namespace trigonometric_identity_application_l1246_124666

theorem trigonometric_identity_application :
  2 * (Real.sin (35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) +
       Real.cos (35 * Real.pi / 180) * Real.cos (65 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end trigonometric_identity_application_l1246_124666


namespace UVWXY_perimeter_l1246_124652

theorem UVWXY_perimeter (U V W X Y Z : ℝ) 
  (hUV : UV = 5)
  (hVW : VW = 3)
  (hWY : WY = 5)
  (hYX : YX = 9)
  (hXU : XU = 7) :
  UV + VW + WY + YX + XU = 29 :=
by
  sorry

end UVWXY_perimeter_l1246_124652


namespace estimate_probability_l1246_124646

noncomputable def freq_20 : ℝ := 0.300
noncomputable def freq_50 : ℝ := 0.360
noncomputable def freq_100 : ℝ := 0.350
noncomputable def freq_300 : ℝ := 0.350
noncomputable def freq_500 : ℝ := 0.352
noncomputable def freq_1000 : ℝ := 0.351
noncomputable def freq_5000 : ℝ := 0.351

theorem estimate_probability : (|0.35 - ((freq_20 + freq_50 + freq_100 + freq_300 + freq_500 + freq_1000 + freq_5000) / 7)| < 0.01) :=
by sorry

end estimate_probability_l1246_124646


namespace find_pointA_coordinates_l1246_124681

-- Define point B
def pointB : ℝ × ℝ := (4, -1)

-- Define the symmetry condition with respect to the x-axis
def symmetricWithRespectToXAxis (p₁ p₂ : ℝ × ℝ) : Prop :=
  p₁.1 = p₂.1 ∧ p₁.2 = -p₂.2

-- Theorem statement: Prove the coordinates of point A given the conditions
theorem find_pointA_coordinates :
  ∃ A : ℝ × ℝ, symmetricWithRespectToXAxis pointB A ∧ A = (4, 1) :=
by
  sorry

end find_pointA_coordinates_l1246_124681


namespace baron_munchausen_not_lying_l1246_124670

def sum_of_digits (n : Nat) : Nat := sorry

theorem baron_munchausen_not_lying :
  ∃ a b : Nat, a ≠ b ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a < 10^10 ∧ 10^9 ≤ a) ∧ (b < 10^10 ∧ 10^9 ≤ b) ∧ 
  (a + sum_of_digits (a ^ 2) = b + sum_of_digits (b ^ 2)) :=
sorry

end baron_munchausen_not_lying_l1246_124670


namespace percentage_of_boys_currently_l1246_124636

variables (B G : ℕ)

theorem percentage_of_boys_currently
  (h1 : B + G = 50)
  (h2 : B + 50 = 95) :
  (B * 100) / 50 = 90 :=
by
  sorry

end percentage_of_boys_currently_l1246_124636


namespace percentage_reduction_l1246_124644

theorem percentage_reduction :
  let original := 243.75
  let reduced := 195
  let percentage := ((original - reduced) / original) * 100
  percentage = 20 :=
by
  sorry

end percentage_reduction_l1246_124644


namespace shifted_sine_monotonically_increasing_l1246_124632

noncomputable def shifted_sine_function (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x - (2 * Real.pi / 3))

theorem shifted_sine_monotonically_increasing :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) → (y ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) → x < y → shifted_sine_function x < shifted_sine_function y :=
by
  sorry

end shifted_sine_monotonically_increasing_l1246_124632


namespace popularity_order_l1246_124684

def chess_popularity := 5 / 16
def drama_popularity := 7 / 24
def music_popularity := 11 / 32
def art_popularity := 13 / 48

theorem popularity_order :
  (31 / 96 < 34 / 96) ∧ (34 / 96 < 35 / 96) ∧ (35 / 96 < 36 / 96) ∧ 
  (chess_popularity < music_popularity) ∧ 
  (drama_popularity < music_popularity) ∧ 
  (music_popularity > art_popularity) ∧ 
  (chess_popularity > drama_popularity) ∧ 
  (drama_popularity > art_popularity) := 
sorry

end popularity_order_l1246_124684


namespace repeating_decimal_sum_l1246_124659

theorem repeating_decimal_sum :
  (0.3333333333 : ℚ) + (0.0404040404 : ℚ) + (0.005005005 : ℚ) + (0.000600060006 : ℚ) = 3793 / 9999 := by
sorry

end repeating_decimal_sum_l1246_124659


namespace ronald_next_roll_l1246_124688

/-- Ronald's rolls -/
def rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

/-- Total number of rolls after the next roll -/
def total_rolls := rolls.length + 1

/-- The desired average of the rolls -/
def desired_average : ℕ := 3

/-- The sum Ronald needs to reach after the next roll to achieve the desired average -/
def required_sum : ℕ := desired_average * total_rolls

/-- Ronald's current sum of rolls -/
def current_sum : ℕ := List.sum rolls

/-- The next roll needed to achieve the desired average -/
def next_roll_needed : ℕ := required_sum - current_sum

theorem ronald_next_roll :
  next_roll_needed = 2 := by
  sorry

end ronald_next_roll_l1246_124688


namespace smallest_possible_value_of_other_integer_l1246_124630

theorem smallest_possible_value_of_other_integer 
  (n : ℕ) (hn_pos : 0 < n) (h_eq : (Nat.lcm 75 n) / (Nat.gcd 75 n) = 45) : n = 135 :=
by sorry

end smallest_possible_value_of_other_integer_l1246_124630


namespace scoops_per_carton_l1246_124655

-- Definitions for scoops required by everyone
def ethan_vanilla := 1
def ethan_chocolate := 1
def lucas_danny_connor_chocolate_each := 2
def lucas_danny_connor := 3
def olivia_vanilla := 1
def olivia_strawberry := 1
def shannon_vanilla := 2 * olivia_vanilla
def shannon_strawberry := 2 * olivia_strawberry

-- Definitions for total scoops taken
def total_vanilla_taken := ethan_vanilla + olivia_vanilla + shannon_vanilla
def total_chocolate_taken := ethan_chocolate + (lucas_danny_connor_chocolate_each * lucas_danny_connor)
def total_strawberry_taken := olivia_strawberry + shannon_strawberry
def total_scoops_taken := total_vanilla_taken + total_chocolate_taken + total_strawberry_taken

-- Definitions for remaining scoops and original total scoops
def remaining_scoops := 16
def original_scoops := total_scoops_taken + remaining_scoops

-- Definition for number of cartons
def total_cartons := 3

-- Proof goal: scoops per carton
theorem scoops_per_carton : original_scoops / total_cartons = 10 := 
by
  -- Add your proof steps here
  sorry

end scoops_per_carton_l1246_124655


namespace heather_total_distance_l1246_124672

-- Definitions for distances walked
def distance_car_to_entrance : ℝ := 0.33
def distance_entrance_to_rides : ℝ := 0.33
def distance_rides_to_car : ℝ := 0.08

-- Statement of the problem to be proven
theorem heather_total_distance :
  distance_car_to_entrance + distance_entrance_to_rides + distance_rides_to_car = 0.74 :=
by
  sorry

end heather_total_distance_l1246_124672


namespace Suzanna_bike_distance_l1246_124693

theorem Suzanna_bike_distance (ride_rate distance_time total_time : ℕ)
  (constant_rate : ride_rate = 3) (time_interval : distance_time = 10)
  (total_riding_time : total_time = 40) :
  (total_time / distance_time) * ride_rate = 12 :=
by
  -- Assuming the conditions:
  -- ride_rate = 3
  -- distance_time = 10
  -- total_time = 40
  sorry

end Suzanna_bike_distance_l1246_124693


namespace income_of_A_l1246_124673

theorem income_of_A (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050) 
  (h2 : (B + C) / 2 = 5250) 
  (h3 : (A + C) / 2 = 4200) : 
  A = 3000 :=
by
  sorry

end income_of_A_l1246_124673


namespace part1_part2_l1246_124658

theorem part1 (a : ℝ) (x : ℝ) (h : a ≠ 0) :
    (|x - a| + |x + a + (1 / a)|) ≥ 2 * Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) (h : a ≠ 0) (h₁ : |2 - a| + |2 + a + 1 / a| ≤ 3) :
    a ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ico (1/2 : ℝ) 2 :=
sorry

end part1_part2_l1246_124658


namespace solve_for_x_l1246_124694

theorem solve_for_x : 
  ∃ x₁ x₂ : ℝ, abs (x₁ - 0.175) < 1e-3 ∧ abs (x₂ - 18.325) < 1e-3 ∧
    (∀ x : ℝ, (8 * x ^ 2 + 120 * x + 7) / (3 * x + 10) = 4 * x + 2 → x = x₁ ∨ x = x₂) := 
by 
  sorry

end solve_for_x_l1246_124694


namespace distinct_symbols_count_l1246_124639

/-- A modified Morse code symbol is represented by a sequence of dots, dashes, and spaces, where spaces can only appear between dots and dashes but not at the beginning or end of the sequence. -/
def valid_sequence_length_1 := 2
def valid_sequence_length_2 := 2^2
def valid_sequence_length_3 := 2^3 + 3
def valid_sequence_length_4 := 2^4 + 3 * 2^4 + 3 * 2^4 
def valid_sequence_length_5 := 2^5 + 4 * 2^5 + 6 * 2^5 + 4 * 2^5

theorem distinct_symbols_count : 
  valid_sequence_length_1 + valid_sequence_length_2 + valid_sequence_length_3 + valid_sequence_length_4 + valid_sequence_length_5 = 609 := by
  sorry

end distinct_symbols_count_l1246_124639


namespace average_stickers_per_pack_l1246_124625

-- Define the conditions given in the problem
def pack1 := 5
def pack2 := 7
def pack3 := 7
def pack4 := 10
def pack5 := 11
def num_packs := 5
def total_stickers := pack1 + pack2 + pack3 + pack4 + pack5

-- Statement to prove the average number of stickers per pack
theorem average_stickers_per_pack :
  (total_stickers / num_packs) = 8 := by
  sorry

end average_stickers_per_pack_l1246_124625


namespace point_inside_circle_range_of_a_l1246_124674

/- 
  Define the circle and the point P. 
  We would show that ensuring the point lies inside the circle implies |a| < 1/13.
-/

theorem point_inside_circle_range_of_a (a : ℝ) : 
  ((5 * a + 1 - 1) ^ 2 + (12 * a) ^ 2 < 1) -> |a| < 1 / 13 := 
by 
  sorry

end point_inside_circle_range_of_a_l1246_124674


namespace function_parallel_l1246_124645

theorem function_parallel {x y : ℝ} (h : y = -2 * x + 1) : 
    ∀ {a : ℝ}, y = -2 * a + 3 -> y = -2 * x + 1 := by
    sorry

end function_parallel_l1246_124645
