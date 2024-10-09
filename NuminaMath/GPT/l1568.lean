import Mathlib

namespace function_domain_l1568_156866

theorem function_domain (x : ℝ) : x ≠ 3 → ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end function_domain_l1568_156866


namespace logarithmic_ratio_l1568_156886

theorem logarithmic_ratio (m n : ℝ) (h1 : Real.log 2 = m) (h2 : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2 * m + n) / (1 - m + n) := 
sorry

end logarithmic_ratio_l1568_156886


namespace not_prime_for_any_n_l1568_156861

theorem not_prime_for_any_n (k : ℕ) (hk : 1 < k) (n : ℕ) : 
  ¬ Prime (n^4 + 4 * k^4) :=
sorry

end not_prime_for_any_n_l1568_156861


namespace alpha_necessary_not_sufficient_for_beta_l1568_156852

def alpha (x : ℝ) : Prop := x^2 = 4
def beta (x : ℝ) : Prop := x = 2

theorem alpha_necessary_not_sufficient_for_beta :
  (∀ x : ℝ, beta x → alpha x) ∧ ¬(∀ x : ℝ, alpha x → beta x) :=
by
  sorry

end alpha_necessary_not_sufficient_for_beta_l1568_156852


namespace lefty_jazz_non_basketball_l1568_156853

-- Definitions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def jazz_loving_members : ℕ := 20
def right_handed_non_jazz_non_basketball : ℕ := 5
def basketball_players : ℕ := 10
def left_handed_jazz_loving_basketball_players : ℕ := 3

-- Problem Statement: Prove the number of lefty jazz lovers who do not play basketball.
theorem lefty_jazz_non_basketball (x : ℕ) :
  (x + left_handed_jazz_loving_basketball_players) + (left_handed_members - x - left_handed_jazz_loving_basketball_players) + 
  (jazz_loving_members - x - left_handed_jazz_loving_basketball_players) + 
  right_handed_non_jazz_non_basketball + left_handed_jazz_loving_basketball_players = 
  total_members → x = 4 :=
by
  sorry

end lefty_jazz_non_basketball_l1568_156853


namespace problem_solution_l1568_156877

def complex_expression : ℕ := 3 * (3 * (4 * (3 * (4 * (2 + 1) + 1) + 2) + 1) + 2) + 1

theorem problem_solution : complex_expression = 1492 := by
  sorry

end problem_solution_l1568_156877


namespace fred_initial_money_l1568_156875

def initial_money (book_count : ℕ) (average_cost : ℕ) (money_left : ℕ) : ℕ :=
  book_count * average_cost + money_left

theorem fred_initial_money :
  initial_money 6 37 14 = 236 :=
by
  sorry

end fred_initial_money_l1568_156875


namespace tan_add_pi_over_4_sin_over_expression_l1568_156840

variable (α : ℝ)

theorem tan_add_pi_over_4 (h : Real.tan α = 2) : 
  Real.tan (α + π / 4) = -3 := 
  sorry

theorem sin_over_expression (h : Real.tan α = 2) : 
  (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 1 := 
  sorry

end tan_add_pi_over_4_sin_over_expression_l1568_156840


namespace motorcycle_price_l1568_156837

variable (x : ℝ) -- selling price of each motorcycle
variable (car_cost material_car material_motorcycle : ℝ)

theorem motorcycle_price
  (h1 : car_cost = 100)
  (h2 : material_car = 4 * 50)
  (h3 : material_motorcycle = 250)
  (h4 : 8 * x - material_motorcycle = material_car - car_cost + 50)
  : x = 50 := 
sorry

end motorcycle_price_l1568_156837


namespace a_eq_zero_l1568_156884

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ x : ℤ, x^2 = 2^n * a + b) : a = 0 :=
sorry

end a_eq_zero_l1568_156884


namespace smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l1568_156825

theorem smallest_k_repr_19_pow_n_sub_5_pow_m_exists :
  ∃ (k n m : ℕ), k > 0 ∧ n > 0 ∧ m > 0 ∧ k = 19 ^ n - 5 ^ m ∧ k = 14 :=
by
  sorry

end smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l1568_156825


namespace find_number_l1568_156817

def digits_form_geometric_progression (x y z : ℕ) : Prop :=
  x * z = y * y

def swapped_hundreds_units (x y z : ℕ) : Prop :=
  100 * z + 10 * y + x = 100 * x + 10 * y + z - 594

def reversed_post_removal (x y z : ℕ) : Prop :=
  10 * z + y = 10 * y + z - 18

theorem find_number (x y z : ℕ) (h1 : digits_form_geometric_progression x y z) 
  (h2 : swapped_hundreds_units x y z) 
  (h3 : reversed_post_removal x y z) :
  100 * x + 10 * y + z = 842 := by
  sorry

end find_number_l1568_156817


namespace blue_string_length_is_320_l1568_156806

-- Define the lengths of the strings
def red_string_length := 8
def white_string_length := 5 * red_string_length
def blue_string_length := 8 * white_string_length

-- The main theorem to prove
theorem blue_string_length_is_320 : blue_string_length = 320 := by
  sorry

end blue_string_length_is_320_l1568_156806


namespace person_saves_2000_l1568_156868

variable (income expenditure savings : ℕ)
variable (h_ratio : income / expenditure = 7 / 6)
variable (h_income : income = 14000)

theorem person_saves_2000 (h_ratio : income / expenditure = 7 / 6) (h_income : income = 14000) :
  savings = income - (6 * (14000 / 7)) :=
by
  sorry

end person_saves_2000_l1568_156868


namespace problem_statement_l1568_156849

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vec_dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem problem_statement : vec_dot (vec_add a (vec_scalar_mul 2 b)) c = -3 := 
by
  sorry

end problem_statement_l1568_156849


namespace largest_common_term_in_sequences_l1568_156842

/-- An arithmetic sequence starts with 3 and has a common difference of 10. A second sequence starts
with 5 and has a common difference of 8. In the range of 1 to 150, the largest number common to 
both sequences is 133. -/
theorem largest_common_term_in_sequences : ∃ (b : ℕ), b < 150 ∧ (∃ (n m : ℤ), b = 3 + 10 * n ∧ b = 5 + 8 * m) ∧ (b = 133) := 
by
  sorry

end largest_common_term_in_sequences_l1568_156842


namespace zero_is_neither_positive_nor_negative_l1568_156801

theorem zero_is_neither_positive_nor_negative :
  ¬ (0 > 0) ∧ ¬ (0 < 0) :=
by
  sorry

end zero_is_neither_positive_nor_negative_l1568_156801


namespace minimize_expression_l1568_156869

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^3 * y^2 * z = 1) : 
  x + 2*y + 3*z ≥ 2 :=
sorry

end minimize_expression_l1568_156869


namespace downstream_speed_l1568_156895

-- Define the speed of the fish in still water
def V_s : ℝ := 45

-- Define the speed of the fish going upstream
def V_u : ℝ := 35

-- Define the speed of the stream
def V_r : ℝ := V_s - V_u

-- Define the speed of the fish going downstream
def V_d : ℝ := V_s + V_r

-- The theorem to be proved
theorem downstream_speed : V_d = 55 := by
  sorry

end downstream_speed_l1568_156895


namespace problem_part1_problem_part2_l1568_156828

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def f (x : ℝ) : ℝ :=
  dot_product (Real.cos x, Real.cos x) (Real.sqrt 3 * Real.cos x, Real.sin x)

theorem problem_part1 :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, (x ∈ Set.Icc (k * π + π / 12) (k * π + 7 * π / 12)) → MonotoneOn f (Set.Icc (k * π + π / 12) (k * π + 7 * π / 12))) :=
sorry

theorem problem_part2 (A : ℝ) (a b c : ℝ) (area : ℝ) :
  f (A / 2 - π / 6) = Real.sqrt 3 ∧ 
  c = 2 ∧ 
  area = 2 * Real.sqrt 3 →
  a = 2 * Real.sqrt 3 ∨ a = 2 * Real.sqrt 7 :=
sorry

end problem_part1_problem_part2_l1568_156828


namespace mary_fruits_left_l1568_156834

-- Conditions as definitions:
def mary_bought_apples : ℕ := 14
def mary_bought_oranges : ℕ := 9
def mary_bought_blueberries : ℕ := 6

def mary_ate_apples : ℕ := 1
def mary_ate_oranges : ℕ := 1
def mary_ate_blueberries : ℕ := 1

-- The problem statement:
theorem mary_fruits_left : 
  (mary_bought_apples - mary_ate_apples) + 
  (mary_bought_oranges - mary_ate_oranges) + 
  (mary_bought_blueberries - mary_ate_blueberries) = 26 := by
  sorry

end mary_fruits_left_l1568_156834


namespace original_prop_and_contrapositive_l1568_156814

theorem original_prop_and_contrapositive (m : ℝ) (h : m > 0) : 
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0 ∨ ∃ x y : ℝ, x^2 + x - m = 0 ∧ y^2 + y - m = 0) :=
by
  sorry

end original_prop_and_contrapositive_l1568_156814


namespace sqrt_sum_simplification_l1568_156800

theorem sqrt_sum_simplification : 
  Real.sqrt ((5 - 3 * Real.sqrt 2)^2) + Real.sqrt ((5 + 3 * Real.sqrt 2)^2) = 10 := by
  sorry

end sqrt_sum_simplification_l1568_156800


namespace connie_num_markers_l1568_156863

def num_red_markers (T : ℝ) := 0.41 * T
def num_total_markers (num_blue_markers : ℝ) (T : ℝ) := num_red_markers T + num_blue_markers

theorem connie_num_markers (T : ℝ) (h1 : num_total_markers 23 T = T) : T = 39 :=
by
sorry

end connie_num_markers_l1568_156863


namespace initial_goldfish_correct_l1568_156878

-- Define the constants related to the conditions
def weekly_die := 5
def weekly_purchase := 3
def final_goldfish := 4
def weeks := 7

-- Define the initial number of goldfish that we need to prove
def initial_goldfish := 18

-- The proof statement: initial_goldfish - weekly_change * weeks = final_goldfish
theorem initial_goldfish_correct (G : ℕ)
  (h : G - weeks * (weekly_purchase - weekly_die) = final_goldfish) :
  G = initial_goldfish := by
  sorry

end initial_goldfish_correct_l1568_156878


namespace conversion_points_worth_two_l1568_156857

theorem conversion_points_worth_two
  (touchdowns_per_game : ℕ := 4)
  (points_per_touchdown : ℕ := 6)
  (games_in_season : ℕ := 15)
  (total_touchdowns_scored : ℕ := touchdowns_per_game * games_in_season)
  (total_points_from_touchdowns : ℕ := total_touchdowns_scored * points_per_touchdown)
  (old_record_points : ℕ := 300)
  (points_above_record : ℕ := 72)
  (total_points_scored : ℕ := old_record_points + points_above_record)
  (conversions_scored : ℕ := 6)
  (total_points_from_conversions : ℕ := total_points_scored - total_points_from_touchdowns) :
  total_points_from_conversions / conversions_scored = 2 := by
sorry

end conversion_points_worth_two_l1568_156857


namespace tan_angle_sum_identity_l1568_156893

theorem tan_angle_sum_identity
  (θ : ℝ)
  (h1 : θ > π / 2 ∧ θ < π)
  (h2 : Real.cos θ = -3 / 5) :
  Real.tan (θ + π / 4) = -1 / 7 := by
  sorry

end tan_angle_sum_identity_l1568_156893


namespace catches_difference_is_sixteen_l1568_156855

noncomputable def joe_catches : ℕ := 23
noncomputable def derek_catches : ℕ := 2 * joe_catches - 4
noncomputable def tammy_catches : ℕ := 30
noncomputable def one_third_derek : ℕ := derek_catches / 3
noncomputable def difference : ℕ := tammy_catches - one_third_derek

theorem catches_difference_is_sixteen :
  difference = 16 := 
by
  sorry

end catches_difference_is_sixteen_l1568_156855


namespace divisibility_of_solutions_l1568_156860

theorem divisibility_of_solutions (p : ℕ) (k : ℕ) (x₀ y₀ z₀ t₀ : ℕ) 
  (hp_prime : Nat.Prime p)
  (hp_form : p = 4 * k + 3)
  (h_eq : x₀^(2*p) + y₀^(2*p) + z₀^(2*p) = t₀^(2*p)) : 
  p ∣ x₀ ∨ p ∣ y₀ ∨ p ∣ z₀ ∨ p ∣ t₀ :=
sorry

end divisibility_of_solutions_l1568_156860


namespace intersection_is_interval_l1568_156859

-- Let M be the set of numbers where the domain of the function y = log x is defined.
def M : Set ℝ := {x | 0 < x}

-- Let N be the set of numbers where x^2 - 4 > 0.
def N : Set ℝ := {x | x^2 - 4 > 0}

-- The complement of N in the real numbers ℝ.
def complement_N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- We need to prove that the intersection of M and the complement of N is the interval (0, 2].
theorem intersection_is_interval : (M ∩ complement_N) = {x | 0 < x ∧ x ≤ 2} := 
by 
  sorry

end intersection_is_interval_l1568_156859


namespace parallelogram_base_is_36_l1568_156873

def parallelogram_base (area height : ℕ) : ℕ :=
  area / height

theorem parallelogram_base_is_36 (h : parallelogram_base 864 24 = 36) : True :=
by
  trivial

end parallelogram_base_is_36_l1568_156873


namespace f_increasing_on_positive_l1568_156882

noncomputable def f (x : ℝ) : ℝ := - (1 / x) - 1

theorem f_increasing_on_positive (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 > x2) : f x1 > f x2 := by
  sorry

end f_increasing_on_positive_l1568_156882


namespace triangle_tan_inequality_l1568_156813

theorem triangle_tan_inequality (A B C : ℝ) (hA : A + B + C = π) :
    (Real.tan A)^2 + (Real.tan B)^2 + (Real.tan C)^2 ≥ (Real.tan A) * (Real.tan B) + (Real.tan B) * (Real.tan C) + (Real.tan C) * (Real.tan A) :=
by
  sorry

end triangle_tan_inequality_l1568_156813


namespace shorter_side_length_l1568_156850

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 50) (h2 : a * b = 126) : b = 9 :=
sorry

end shorter_side_length_l1568_156850


namespace circles_tangent_l1568_156843

/--
Two equal circles each with a radius of 5 are externally tangent to each other and both are internally tangent to a larger circle with a radius of 13. 
Let the points of tangency be A and B. Let AB = m/n where m and n are positive integers and gcd(m, n) = 1. 
We need to prove that m + n = 69.
-/
theorem circles_tangent (r1 r2 r3 : ℝ) (tangent_external : ℝ) (tangent_internal : ℝ) (AB : ℝ) (m n : ℕ) 
  (hmn_coprime : Nat.gcd m n = 1) (hr1 : r1 = 5) (hr2 : r2 = 5) (hr3 : r3 = 13) 
  (ht_external : tangent_external = r1 + r2) (ht_internal : tangent_internal = r3 - r1) 
  (hAB : AB = (130 / 8)): m + n = 69 :=
by
  sorry

end circles_tangent_l1568_156843


namespace correct_regression_equation_l1568_156867

variable (x y : ℝ)

-- Assume that y is negatively correlated with x
axiom negative_correlation : x * y ≤ 0

-- The candidate regression equations
def regression_A : ℝ := -2 * x - 100
def regression_B : ℝ := 2 * x - 100
def regression_C : ℝ := -2 * x + 100
def regression_D : ℝ := 2 * x + 100

-- Prove that the correct regression equation reflecting the negative correlation is regression_C
theorem correct_regression_equation : regression_C x = -2 * x + 100 := by
  sorry

end correct_regression_equation_l1568_156867


namespace min_value_3x_4y_l1568_156899

theorem min_value_3x_4y {x y : ℝ} (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) :
    3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_4y_l1568_156899


namespace semicircle_radius_l1568_156824

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (hP : P = 180) : radius_of_semicircle P = 180 / (Real.pi + 2) :=
by
  sorry

end semicircle_radius_l1568_156824


namespace line_parallel_through_M_line_perpendicular_through_M_l1568_156896

-- Define the lines L1 and L2
def L1 (x y: ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def L2 (x y: ℝ) : Prop := x - 3 * y + 8 = 0

-- Define the parallel and perpendicular lines
def parallel_to_line (x y: ℝ) : Prop := 2 * x + y + 5 = 0
def perpendicular_to_line (x y: ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the intersection points
def M : ℝ × ℝ := (-2, 2)

-- Define the lines that pass through point M and are parallel or perpendicular to the given line
def line_parallel (x y: ℝ) : Prop := 2 * x + y + 2 = 0
def line_perpendicular (x y: ℝ) : Prop := x - 2 * y + 6 = 0

-- The proof statements
theorem line_parallel_through_M : ∃ x y : ℝ, L1 x y ∧ L2 x y ∧ x = (-2) ∧ y = 2 -> line_parallel x y := by
  sorry

theorem line_perpendicular_through_M : ∃ x y : ℝ, L1 x y ∧ L2 x y ∧ x = (-2) ∧ y = 2 -> line_perpendicular x y := by
  sorry

end line_parallel_through_M_line_perpendicular_through_M_l1568_156896


namespace Ivan_uses_more_paint_l1568_156809

noncomputable def Ivan_section_area : ℝ := 10

noncomputable def Petr_section_area (α : ℝ) : ℝ := 10 * Real.sin α

theorem Ivan_uses_more_paint (α : ℝ) (hα : Real.sin α < 1) : 
  Ivan_section_area > Petr_section_area α := 
by 
  rw [Ivan_section_area, Petr_section_area]
  linarith [hα]

end Ivan_uses_more_paint_l1568_156809


namespace triangle_isosceles_of_sin_condition_l1568_156885

noncomputable def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

theorem triangle_isosceles_of_sin_condition {A B C : ℝ} (h : 2 * Real.sin A * Real.cos B = Real.sin C) : 
  isosceles_triangle A B C :=
by
  sorry

end triangle_isosceles_of_sin_condition_l1568_156885


namespace jennas_total_ticket_cost_l1568_156841

theorem jennas_total_ticket_cost :
  let normal_price := 50
  let tickets_from_website := 2 * normal_price
  let scalper_price := 2 * normal_price * 2.4 - 10
  let friend_discounted_ticket := normal_price * 0.6
  tickets_from_website + scalper_price + friend_discounted_ticket = 360 :=
by
  sorry

end jennas_total_ticket_cost_l1568_156841


namespace equilateral_triangle_perimeter_isosceles_triangle_leg_length_l1568_156889

-- Definitions for equilateral triangle problem
def side_length_equilateral : ℕ := 12
def perimeter_equilateral := side_length_equilateral * 3

-- Definitions for isosceles triangle problem
def perimeter_isosceles : ℕ := 72
def base_length_isosceles : ℕ := 28
def leg_length_isosceles := (perimeter_isosceles - base_length_isosceles) / 2

-- Theorem statement
theorem equilateral_triangle_perimeter : perimeter_equilateral = 36 := 
by
  sorry

theorem isosceles_triangle_leg_length : leg_length_isosceles = 22 := 
by
  sorry

end equilateral_triangle_perimeter_isosceles_triangle_leg_length_l1568_156889


namespace min_value_a_b_c_l1568_156864

theorem min_value_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : 9 * a + 4 * b = a * b * c) :
  a + b + c = 10 := sorry

end min_value_a_b_c_l1568_156864


namespace tunnel_length_l1568_156835

theorem tunnel_length (x : ℕ) (y : ℕ) 
  (h1 : 300 + x = 60 * y) 
  (h2 : x - 300 = 30 * y) : 
  x = 900 := 
by
  sorry

end tunnel_length_l1568_156835


namespace student_number_choice_l1568_156845

theorem student_number_choice (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 :=
sorry

end student_number_choice_l1568_156845


namespace rachel_steps_l1568_156836

theorem rachel_steps (x : ℕ) (h1 : x + 325 = 892) : x = 567 :=
sorry

end rachel_steps_l1568_156836


namespace sarah_toads_l1568_156832

theorem sarah_toads (tim_toads : ℕ) (jim_toads : ℕ) (sarah_toads : ℕ)
  (h1 : tim_toads = 30)
  (h2 : jim_toads = tim_toads + 20)
  (h3 : sarah_toads = 2 * jim_toads) :
  sarah_toads = 100 :=
by
  sorry

end sarah_toads_l1568_156832


namespace final_price_including_tax_l1568_156847

noncomputable def increasedPrice (originalPrice : ℝ) (increasePercentage : ℝ) : ℝ :=
  originalPrice + originalPrice * increasePercentage

noncomputable def discountedPrice (increasedPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  increasedPrice - increasedPrice * discountPercentage

noncomputable def finalPrice (discountedPrice : ℝ) (salesTax : ℝ) : ℝ :=
  discountedPrice + discountedPrice * salesTax

theorem final_price_including_tax :
  let originalPrice := 200
  let increasePercentage := 0.30
  let discountPercentage := 0.30
  let salesTax := 0.07
  let incPrice := increasedPrice originalPrice increasePercentage
  let disPrice := discountedPrice incPrice discountPercentage
  finalPrice disPrice salesTax = 194.74 :=
by
  simp [increasedPrice, discountedPrice, finalPrice]
  sorry

end final_price_including_tax_l1568_156847


namespace distance_from_P_to_AD_is_correct_l1568_156888

noncomputable def P_distance_to_AD : ℝ :=
  let A : ℝ × ℝ := (0, 6)
  let D : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, 0)
  let M : ℝ × ℝ := (3, 0)
  let radius1 : ℝ := 5
  let radius2 : ℝ := 6
  let circle1_eq := fun (x y : ℝ) => (x - 3)^2 + y^2 = 25
  let circle2_eq := fun (x y : ℝ) => x^2 + (y - 6)^2 = 36
  let P := (24/5, 18/5)
  let AD := fun x y : ℝ => x = 0
  abs ((P.fst : ℝ) - 0)

theorem distance_from_P_to_AD_is_correct :
  P_distance_to_AD = 24 / 5 := by
  sorry

end distance_from_P_to_AD_is_correct_l1568_156888


namespace problem1_problem2_l1568_156826

-- Define the function f
def f (x b : ℝ) := |2 * x + b|

-- First problem: prove if the solution set of |2x + b| <= 3 is {x | -1 ≤ x ≤ 2}, then b = -1.
theorem problem1 (b : ℝ) : (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 2 → |2 * x + b| ≤ 3)) → b = -1 :=
sorry

-- Second problem: given b = -1, prove that for all x ∈ ℝ, |2(x+3)-1| + |2(x+1)-1| ≥ -4.
theorem problem2 : (∀ x : ℝ, f (x + 3) (-1) + f (x + 1) (-1) ≥ -4) :=
sorry

end problem1_problem2_l1568_156826


namespace move_digit_produces_ratio_l1568_156821

theorem move_digit_produces_ratio
  (a b : ℕ)
  (h_original_eq : ∃ x : ℕ, x = 10 * a + b)
  (h_new_eq : ∀ (n : ℕ), 10^n * b + a = (3 * (10 * a + b)) / 2):
  285714 = 10 * a + b :=
by
  -- proof steps would go here
  sorry

end move_digit_produces_ratio_l1568_156821


namespace solution_set_of_inequality_l1568_156874

theorem solution_set_of_inequality (x : ℝ) : |x^2 - 2| < 2 ↔ ((-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2)) :=
by sorry

end solution_set_of_inequality_l1568_156874


namespace repeating_decimal_fraction_sum_l1568_156812

/-- The repeating decimal 3.171717... can be written as a fraction. When reduced to lowest
terms, the sum of the numerator and denominator of this fraction is 413. -/
theorem repeating_decimal_fraction_sum :
  let y := 3.17171717 -- The repeating decimal
  let frac_num := 314
  let frac_den := 99
  let sum := frac_num + frac_den
  y = frac_num / frac_den ∧ sum = 413 := by
  sorry

end repeating_decimal_fraction_sum_l1568_156812


namespace smallest_number_with_property_l1568_156829

theorem smallest_number_with_property: 
  ∃ (N : ℕ), N = 25 ∧ (∀ (x : ℕ) (h : N = x + (x / 5)), N ≤ x) := 
  sorry

end smallest_number_with_property_l1568_156829


namespace tan_theta_eq_two_implies_expression_l1568_156881

theorem tan_theta_eq_two_implies_expression (θ : ℝ) (h : Real.tan θ = 2) :
    (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
by
  -- Define trig identities and given condition
  have h_sin_cos : Real.sin θ = 2 / Real.sqrt 5 ∧ Real.cos θ = 1 / Real.sqrt 5 :=
    sorry -- This will be derived from the given condition h
  
  -- Main proof
  sorry

end tan_theta_eq_two_implies_expression_l1568_156881


namespace bananas_and_cantaloupe_cost_l1568_156858

noncomputable def prices (a b c d : ℕ) : Prop :=
  a + b + c + d = 40 ∧
  d = 3 * a ∧
  b = c - 2

theorem bananas_and_cantaloupe_cost (a b c d : ℕ) (h : prices a b c d) : b + c = 20 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  -- Using the given conditions:
  --     a + b + c + d = 40
  --     d = 3 * a
  --     b = c - 2
  -- We find that b + c = 20
  sorry

end bananas_and_cantaloupe_cost_l1568_156858


namespace magic_square_sum_l1568_156844

theorem magic_square_sum (a b c d e : ℕ) 
    (h1 : a + c + e = 55)
    (h2 : 30 + 10 + a = 55)
    (h3 : 30 + e + 15 = 55)
    (h4 : 10 + 30 + d = 55) :
    d + e = 25 := by
  sorry

end magic_square_sum_l1568_156844


namespace total_stamps_in_collection_l1568_156846

-- Definitions reflecting the problem conditions
def foreign_stamps : ℕ := 90
def old_stamps : ℕ := 60
def both_foreign_and_old_stamps : ℕ := 20
def neither_foreign_nor_old_stamps : ℕ := 70

-- The expected total number of stamps in the collection
def total_stamps : ℕ :=
  (foreign_stamps + old_stamps - both_foreign_and_old_stamps) + neither_foreign_nor_old_stamps

-- Statement to prove the total number of stamps is 200
theorem total_stamps_in_collection : total_stamps = 200 := by
  -- Proof omitted
  sorry

end total_stamps_in_collection_l1568_156846


namespace probability_no_adjacent_birch_trees_l1568_156871

open Nat

theorem probability_no_adjacent_birch_trees : 
    let m := 7
    let n := 990
    m + n = 106 := 
by
  sorry

end probability_no_adjacent_birch_trees_l1568_156871


namespace union_M_N_l1568_156815

def M := {x : ℝ | x^2 - 4*x + 3 ≤ 0}
def N := {x : ℝ | Real.log x / Real.log 2 ≤ 1}

theorem union_M_N :
  M ∪ N = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end union_M_N_l1568_156815


namespace ratio_problem_l1568_156862

theorem ratio_problem {q r s t : ℚ} (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 :=
sorry

end ratio_problem_l1568_156862


namespace angle_c_in_triangle_l1568_156851

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A/B = 1/3) (h3 : A/C = 1/5) : C = 100 :=
by
  sorry

end angle_c_in_triangle_l1568_156851


namespace initial_men_count_l1568_156883

theorem initial_men_count (M : ℕ) :
  let total_food := M * 22
  let food_after_2_days := total_food - 2 * M
  let remaining_food := 20 * M
  let new_total_men := M + 190
  let required_food_for_16_days := new_total_men * 16
  (remaining_food = required_food_for_16_days) → M = 760 :=
by
  intro h
  sorry

end initial_men_count_l1568_156883


namespace arithmetic_sequence_geometric_condition_l1568_156872

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 1) (h_d_nonzero : d ≠ 0)
  (h_geom : (1 + d) * (1 + d) = 1 * (1 + 4 * d)) : a 2013 = 4025 := by sorry

end arithmetic_sequence_geometric_condition_l1568_156872


namespace no_sum_of_squares_of_rationals_l1568_156892

theorem no_sum_of_squares_of_rationals (p q r s : ℕ) (hq : q ≠ 0) (hs : s ≠ 0)
    (hpq : Nat.gcd p q = 1) (hrs : Nat.gcd r s = 1) :
    (↑p / q : ℚ) ^ 2 + (↑r / s : ℚ) ^ 2 ≠ 168 := by 
    sorry

end no_sum_of_squares_of_rationals_l1568_156892


namespace max_slope_no_lattice_points_l1568_156894

theorem max_slope_no_lattice_points :
  (∃ b : ℚ, (∀ m : ℚ, 1 / 3 < m ∧ m < b → ∀ x : ℤ, 0 < x ∧ x ≤ 200 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ b = 68 / 203) := 
sorry

end max_slope_no_lattice_points_l1568_156894


namespace find_number_l1568_156838

theorem find_number (x : ℝ) : 14 * x + 15 * x + 18 * x + 11 = 152 → x = 3 := by
  sorry

end find_number_l1568_156838


namespace odometer_problem_l1568_156827

theorem odometer_problem (a b c : ℕ) (h₀ : a + b + c = 7) (h₁ : 1 ≤ a)
  (h₂ : a < 10) (h₃ : b < 10) (h₄ : c < 10) (h₅ : (c - a) % 20 = 0) : a^2 + b^2 + c^2 = 37 := 
  sorry

end odometer_problem_l1568_156827


namespace anthony_transactions_more_percentage_l1568_156897

def transactions (Mabel Anthony Cal Jade : ℕ) : Prop := 
  Mabel = 90 ∧ 
  Jade = 84 ∧ 
  Jade = Cal + 18 ∧ 
  Cal = (2 * Anthony) / 3 ∧ 
  Anthony = Mabel + (Mabel * 10 / 100)

theorem anthony_transactions_more_percentage (Mabel Anthony Cal Jade : ℕ) 
    (h : transactions Mabel Anthony Cal Jade) : 
  (Anthony = Mabel + (Mabel * 10 / 100)) :=
by 
  sorry

end anthony_transactions_more_percentage_l1568_156897


namespace team_a_builds_per_day_l1568_156876

theorem team_a_builds_per_day (x : ℝ) (h1 : (150 / x = 100 / (2 * x - 30))) : x = 22.5 := by
  sorry

end team_a_builds_per_day_l1568_156876


namespace solve_for_square_l1568_156854

theorem solve_for_square (x : ℤ) (s : ℤ) 
  (h1 : s + x = 80) 
  (h2 : 3 * (s + x) - 2 * x = 164) : 
  s = 42 :=
by 
  -- Include the implementation with sorry
  sorry

end solve_for_square_l1568_156854


namespace base_case_proof_l1568_156807

noncomputable def base_case_inequality := 1 + (1 / (2 ^ 3)) < 2 - (1 / 2)

theorem base_case_proof : base_case_inequality := by
  -- The proof would go here
  sorry

end base_case_proof_l1568_156807


namespace time_shortened_by_opening_both_pipes_l1568_156822

theorem time_shortened_by_opening_both_pipes 
  (a b p : ℝ) 
  (hp : a * p > 0) -- To ensure p > 0 and reservoir volume is positive
  (h1 : p = (a * p) / a) -- Given that pipe A alone takes p hours
  : p - (a * p) / (a + b) = (b * p) / (a + b) := 
sorry

end time_shortened_by_opening_both_pipes_l1568_156822


namespace f_sum_zero_l1568_156887

noncomputable def f : ℝ → ℝ := sorry

axiom f_property_1 : ∀ x : ℝ, f (x ^ 3) = (f x) ^ 3
axiom f_property_2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2

theorem f_sum_zero : f 0 + f (-1) + f 1 = 0 := by
  sorry

end f_sum_zero_l1568_156887


namespace larger_square_uncovered_area_l1568_156816

theorem larger_square_uncovered_area :
  let side_length_larger := 10
  let side_length_smaller := 4
  let area_larger := side_length_larger ^ 2
  let area_smaller := side_length_smaller ^ 2
  (area_larger - area_smaller) = 84 :=
by
  let side_length_larger := 10
  let side_length_smaller := 4
  let area_larger := side_length_larger ^ 2
  let area_smaller := side_length_smaller ^ 2
  sorry

end larger_square_uncovered_area_l1568_156816


namespace steve_book_earning_l1568_156810

theorem steve_book_earning
  (total_copies : ℕ)
  (advance_copies : ℕ)
  (total_kept : ℝ)
  (agent_cut_percentage : ℝ)
  (copies : ℕ)
  (money_kept : ℝ)
  (x : ℝ)
  (h1 : total_copies = 1000000)
  (h2 : advance_copies = 100000)
  (h3 : total_kept = 1620000)
  (h4 : agent_cut_percentage = 0.10)
  (h5 : copies = total_copies - advance_copies)
  (h6 : money_kept = copies * (1 - agent_cut_percentage) * x)
  (h7 : money_kept = total_kept) :
  x = 2 := 
by 
  sorry

end steve_book_earning_l1568_156810


namespace total_people_in_office_even_l1568_156880

theorem total_people_in_office_even (M W : ℕ) (h_even : M = W) (h_meeting_women : 6 = 20 / 100 * W) : 
  M + W = 60 :=
by
  sorry

end total_people_in_office_even_l1568_156880


namespace find_x_l1568_156804

theorem find_x (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 :=
sorry

end find_x_l1568_156804


namespace min_balls_to_draw_l1568_156856

theorem min_balls_to_draw (red blue green yellow white black : ℕ) (h_red : red = 35) (h_blue : blue = 25) (h_green : green = 22) (h_yellow : yellow = 18) (h_white : white = 14) (h_black : black = 12) : 
  ∃ n, n = 95 ∧ ∀ (r b g y w bl : ℕ), r ≤ red ∧ b ≤ blue ∧ g ≤ green ∧ y ≤ yellow ∧ w ≤ white ∧ bl ≤ black → (r + b + g + y + w + bl = 95 → r ≥ 18 ∨ b ≥ 18 ∨ g ≥ 18 ∨ y ≥ 18 ∨ w ≥ 18 ∨ bl ≥ 18) :=
by sorry

end min_balls_to_draw_l1568_156856


namespace relationship_l1568_156898

noncomputable def a : ℝ := 3^(-1/3 : ℝ)
noncomputable def b : ℝ := Real.log 3 / Real.log 2⁻¹
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship (a_def : a = 3^(-1/3 : ℝ)) 
                     (b_def : b = Real.log 3 / Real.log 2⁻¹) 
                     (c_def : c = Real.log 3 / Real.log 2) : 
  b < a ∧ a < c :=
  sorry

end relationship_l1568_156898


namespace volume_of_cube_in_pyramid_l1568_156870

theorem volume_of_cube_in_pyramid :
  (∃ (s : ℝ), 
    ( ∀ (b h l : ℝ),
      b = 2 ∧ 
      h = 3 ∧ 
      l = 2 * Real.sqrt 2 →
      s = 4 * Real.sqrt 2 - 3 ∧ 
      ((4 * Real.sqrt 2 - 3) ^ 3 = (4 * Real.sqrt 2 - 3) ^ 3))) :=
sorry

end volume_of_cube_in_pyramid_l1568_156870


namespace sufficient_but_not_necessary_condition_l1568_156848

theorem sufficient_but_not_necessary_condition (a1 d : ℝ) : 
  (2 * a1 + 11 * d > 0) → (2 * a1 + 11 * d ≥ 0) :=
by
  intro h
  apply le_of_lt
  exact h

end sufficient_but_not_necessary_condition_l1568_156848


namespace fraction_sum_eq_one_l1568_156890

variables {a b c x y z : ℝ}

-- Conditions
axiom h1 : 11 * x + b * y + c * z = 0
axiom h2 : a * x + 24 * y + c * z = 0
axiom h3 : a * x + b * y + 41 * z = 0
axiom h4 : a ≠ 11
axiom h5 : x ≠ 0

-- Theorem Statement
theorem fraction_sum_eq_one : 
  a/(a - 11) + b/(b - 24) + c/(c - 41) = 1 :=
by sorry

end fraction_sum_eq_one_l1568_156890


namespace hyperbola_eccentricity_sqrt_five_l1568_156879

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1 where a > 0 and b > 0,
and its focus lies symmetrically with respect to the asymptote lines and on the hyperbola,
proves that the eccentricity of the hyperbola is sqrt(5). -/
theorem hyperbola_eccentricity_sqrt_five 
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) 
  (c : ℝ) (h_focus : c^2 = 5 * a^2) : 
  (c / a = Real.sqrt 5) := sorry

end hyperbola_eccentricity_sqrt_five_l1568_156879


namespace tan_sum_identity_l1568_156803

theorem tan_sum_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.tan α + (1 / Real.tan α) = 3 :=
by
  sorry

end tan_sum_identity_l1568_156803


namespace first_quarter_days_2016_l1568_156823

theorem first_quarter_days_2016 : 
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  (jan_days + feb_days + mar_days) = 91 := 
by
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  have h_leap_year : leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) := by sorry
  have h_feb_days : feb_days = 29 := by sorry
  have h_first_quarter : jan_days + feb_days + mar_days = 31 + 29 + 31 := by sorry
  have h_sum : 31 + 29 + 31 = 91 := by norm_num
  exact h_sum

end first_quarter_days_2016_l1568_156823


namespace john_toy_store_fraction_l1568_156830

theorem john_toy_store_fraction :
  let allowance := 4.80
  let arcade_spent := 3 / 5 * allowance
  let remaining_after_arcade := allowance - arcade_spent
  let candy_store_spent := 1.28
  let toy_store_spent := remaining_after_arcade - candy_store_spent
  (toy_store_spent / remaining_after_arcade) = 1 / 3 := by
    sorry

end john_toy_store_fraction_l1568_156830


namespace percent_of_a_is_4b_l1568_156819

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b / a) * 100 = 333.33 := by
  sorry

end percent_of_a_is_4b_l1568_156819


namespace wyatt_headmaster_duration_l1568_156811

def Wyatt_start_month : Nat := 3 -- March
def Wyatt_break_start_month : Nat := 7 -- July
def Wyatt_break_end_month : Nat := 12 -- December
def Wyatt_end_year : Nat := 2011

def months_worked_before_break : Nat := Wyatt_break_start_month - Wyatt_start_month -- March to June (inclusive, hence -1)
def break_duration : Nat := 6
def months_worked_after_break : Nat := 12 -- January to December 2011

def total_months_worked : Nat := months_worked_before_break + months_worked_after_break
theorem wyatt_headmaster_duration : total_months_worked = 16 :=
by
  sorry

end wyatt_headmaster_duration_l1568_156811


namespace marks_in_social_studies_l1568_156891

def shekar_marks : ℕ := 82

theorem marks_in_social_studies 
  (marks_math : ℕ := 76)
  (marks_science : ℕ := 65)
  (marks_english : ℕ := 67)
  (marks_biology : ℕ := 55)
  (average_marks : ℕ := 69)
  (num_subjects : ℕ := 5) :
  marks_math + marks_science + marks_english + marks_biology + shekar_marks = average_marks * num_subjects :=
by
  sorry

end marks_in_social_studies_l1568_156891


namespace latitude_approx_l1568_156820

noncomputable def calculate_latitude (R h : ℝ) (θ : ℝ) : ℝ :=
  if h = 0 then θ else Real.arccos (1 / (2 * Real.pi))

theorem latitude_approx (R h θ : ℝ) (h_nonzero : h ≠ 0)
  (r1 : ℝ := R * Real.cos θ)
  (r2 : ℝ := (R + h) * Real.cos θ)
  (s : ℝ := 2 * Real.pi * h * Real.cos θ)
  (condition : s = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end latitude_approx_l1568_156820


namespace original_stone_counted_as_99_l1568_156839

theorem original_stone_counted_as_99 :
  (99 % 22) = 11 :=
by sorry

end original_stone_counted_as_99_l1568_156839


namespace square_of_integer_l1568_156818

theorem square_of_integer (n : ℕ) (h : ∃ l : ℤ, l^2 = 1 + 12 * (n^2 : ℤ)) :
  ∃ m : ℤ, 2 + 2 * Int.sqrt (1 + 12 * (n^2 : ℤ)) = m^2 := by
  sorry

end square_of_integer_l1568_156818


namespace problem1_problem2_l1568_156831

theorem problem1 (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = m - |x - 2|) 
  (h2 : ∀ x, f (x + 2) ≥ 0 → -1 ≤ x ∧ x ≤ 1) : 
  m = 1 := 
sorry

theorem problem2 (a b c : ℝ) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 := 
sorry

end problem1_problem2_l1568_156831


namespace parabola_equation_exists_line_m_equation_exists_l1568_156833

noncomputable def problem_1 : Prop :=
  ∃ (p : ℝ), p > 0 ∧ (∀ (x y : ℝ), x^2 = 2 * p * y → y = x^2 / (2 * p)) ∧ 
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 = 2 * p * y1 → x2^2 = 2 * p * y2 → 
    (y1 + y2 = 8 - p) ∧ ((y1 + y2) / 2 = 3) → p = 2)

noncomputable def problem_2 : Prop :=
  ∃ (k : ℝ), (k^2 = 1 / 4) ∧ (∀ (x : ℝ), (x^2 - 4 * k * x - 24 = 0) → 
    (∃ (x1 x2 : ℝ), x1 + x2 = 4 * k ∧ x1 * x2 = -24)) ∧
  (∀ (x1 x2 : ℝ), x1^2 = 4 * (k * x1 + 6) ∧ x2^2 = 4 * (k * x2 + 6) → 
    ∀ (x3 x4 : ℝ), (x1 * x2) ^ 2 - 4 * ((x1 + x2) ^ 2 - 2 * x1 * x2) + 16 + 16 * x1 * x2 = 0 → 
    (k = 1 / 2 ∨ k = -1 / 2))

theorem parabola_equation_exists : problem_1 :=
by {
  sorry
}

theorem line_m_equation_exists : problem_2 :=
by {
  sorry
}

end parabola_equation_exists_line_m_equation_exists_l1568_156833


namespace ratio_a_b_eq_neg_one_fifth_l1568_156808

theorem ratio_a_b_eq_neg_one_fifth (x y a b : ℝ) (hb_ne_zero : b ≠ 0) 
    (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) : a / b = -1 / 5 :=
by {
  sorry
}

end ratio_a_b_eq_neg_one_fifth_l1568_156808


namespace angie_age_l1568_156802

variables (A : ℕ)

theorem angie_age (h : 2 * A + 4 = 20) : A = 8 :=
by {
  -- Proof will be provided in actual usage or practice
  sorry
}

end angie_age_l1568_156802


namespace approx_change_in_y_l1568_156865

-- Definition of the function
def y (x : ℝ) : ℝ := x^3 - 7 * x^2 + 80

-- Derivative of the function, calculated manually
def y_prime (x : ℝ) : ℝ := 3 * x^2 - 14 * x

-- The change in x
def delta_x : ℝ := 0.01

-- The given value of x
def x_initial : ℝ := 5

-- To be proved: the approximate change in y
theorem approx_change_in_y : (y_prime x_initial) * delta_x = 0.05 :=
by
  -- Imported and recognized theorem verifications skipped
  sorry

end approx_change_in_y_l1568_156865


namespace domain_of_function_l1568_156805

theorem domain_of_function:
  {x : ℝ | x + 1 ≥ 0 ∧ 3 - x ≠ 0} = {x : ℝ | x ≥ -1 ∧ x ≠ 3} :=
by
  sorry

end domain_of_function_l1568_156805
