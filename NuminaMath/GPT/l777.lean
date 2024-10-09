import Mathlib

namespace all_cells_equal_l777_77750

-- Define the infinite grid
def Grid := ℕ → ℕ → ℕ

-- Define the condition on the grid values
def is_min_mean_grid (g : Grid) : Prop :=
  ∀ i j : ℕ, g i j ≥ (g (i-1) j + g (i+1) j + g i (j-1) + g i (j+1)) / 4

-- Main theorem
theorem all_cells_equal (g : Grid) (h : is_min_mean_grid g) : ∃ a : ℕ, ∀ i j : ℕ, g i j = a := 
sorry

end all_cells_equal_l777_77750


namespace not_hyperbola_condition_l777_77745

theorem not_hyperbola_condition (m : ℝ) (x y : ℝ) (h1 : 1 ≤ m) (h2 : m ≤ 3) :
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m) :=
sorry

end not_hyperbola_condition_l777_77745


namespace Yvettes_final_bill_l777_77794

namespace IceCreamShop

def sundae_price_Alicia : Real := 7.50
def sundae_price_Brant : Real := 10.00
def sundae_price_Josh : Real := 8.50
def sundae_price_Yvette : Real := 9.00
def tip_rate : Real := 0.20

theorem Yvettes_final_bill :
  let total_cost := sundae_price_Alicia + sundae_price_Brant + sundae_price_Josh + sundae_price_Yvette
  let tip := tip_rate * total_cost
  let final_bill := total_cost + tip
  final_bill = 42.00 :=
by
  -- calculations are skipped here
  sorry

end IceCreamShop

end Yvettes_final_bill_l777_77794


namespace inverse_function_coeff_ratio_l777_77728

noncomputable def f_inv_coeff_ratio : ℝ :=
  let f (x : ℝ) := (2 * x - 1) / (x + 5)
  let a := 5
  let b := 1
  let c := -1
  let d := 2
  a / c

theorem inverse_function_coeff_ratio :
  f_inv_coeff_ratio = -5 := 
by
  sorry

end inverse_function_coeff_ratio_l777_77728


namespace octahedron_has_constant_perimeter_cross_sections_l777_77703

structure Octahedron :=
(edge_length : ℝ)

def all_cross_sections_same_perimeter (oct : Octahedron) :=
  ∀ (face1 face2 : ℝ), (face1 = face2)

theorem octahedron_has_constant_perimeter_cross_sections (oct : Octahedron) :
  all_cross_sections_same_perimeter oct :=
  sorry

end octahedron_has_constant_perimeter_cross_sections_l777_77703


namespace find_b7_l777_77702

/-- We represent the situation with twelve people in a circle, each with an integer number. The
     average announced by a person is the average of their two immediate neighbors. Given the
     person who announced the average of 7, we aim to find the number they initially chose. --/
theorem find_b7 (b : ℕ → ℕ) (announced_avg : ℕ → ℕ) :
  (announced_avg 1 = (b 12 + b 2) / 2) ∧
  (announced_avg 2 = (b 1 + b 3) / 2) ∧
  (announced_avg 3 = (b 2 + b 4) / 2) ∧
  (announced_avg 4 = (b 3 + b 5) / 2) ∧
  (announced_avg 5 = (b 4 + b 6) / 2) ∧
  (announced_avg 6 = (b 5 + b 7) / 2) ∧
  (announced_avg 7 = (b 6 + b 8) / 2) ∧
  (announced_avg 8 = (b 7 + b 9) / 2) ∧
  (announced_avg 9 = (b 8 + b 10) / 2) ∧
  (announced_avg 10 = (b 9 + b 11) / 2) ∧
  (announced_avg 11 = (b 10 + b 12) / 2) ∧
  (announced_avg 12 = (b 11 + b 1) / 2) ∧
  (announced_avg 7 = 7) →
  b 7 = 12 := 
sorry

end find_b7_l777_77702


namespace layers_tall_l777_77766

def total_cards (n_d c_d : ℕ) : ℕ := n_d * c_d
def layers (total c_l : ℕ) : ℕ := total / c_l

theorem layers_tall (n_d c_d c_l : ℕ) (hn_d : n_d = 16) (hc_d : c_d = 52) (hc_l : c_l = 26) : 
  layers (total_cards n_d c_d) c_l = 32 := by
  sorry

end layers_tall_l777_77766


namespace color_block_prob_l777_77789

-- Definitions of the problem's conditions
def colors : List (List String) := [
    ["red", "blue", "yellow", "green"],
    ["red", "blue", "yellow", "white"]
]

-- The events in which at least one box receives 3 blocks of the same color
def event_prob : ℚ := 3 / 64

-- Tuple as a statement to prove in Lean
theorem color_block_prob (m n : ℕ) (h : m + n = 67) : 
  ∃ (m n : ℕ), (m / n : ℚ) = event_prob := 
by
  use 3
  use 64
  simp
  sorry

end color_block_prob_l777_77789


namespace quadratic_value_at_two_l777_77743

open Real

-- Define the conditions
variables (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 + a * x + b

-- State the proof problem
theorem quadratic_value_at_two (h₀ : f a b (f a b 0) = 0) (h₁ : f a b (f a b 1) = 0) (h₂ : f a b 0 ≠ f a b 1) :
  f a b 2 = 2 := 
sorry

end quadratic_value_at_two_l777_77743


namespace trapezium_area_l777_77748

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  sorry

end trapezium_area_l777_77748


namespace usual_time_catch_bus_l777_77756

variable (S T T' : ℝ)

theorem usual_time_catch_bus (h1 : T' = T + 6)
  (h2 : S * T = (4 / 5) * S * T') : T = 24 := by
  sorry

end usual_time_catch_bus_l777_77756


namespace find_a_plus_b_l777_77777

open Function

theorem find_a_plus_b (a b : ℝ) (f g h : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x - b)
  (h_g : ∀ x, g x = -4 * x - 1)
  (h_h : ∀ x, h x = f (g x))
  (h_h_inv : ∀ y, h⁻¹ y = y + 9) :
  a + b = -9 := 
by
  -- Proof goes here.
  sorry

end find_a_plus_b_l777_77777


namespace no_unsatisfactory_grades_l777_77706

theorem no_unsatisfactory_grades (total_students : ℕ)
  (top_marks : ℕ) (average_marks : ℕ) (good_marks : ℕ)
  (h1 : top_marks = total_students / 6)
  (h2 : average_marks = total_students / 3)
  (h3 : good_marks = total_students / 2) :
  total_students = top_marks + average_marks + good_marks := by
  sorry

end no_unsatisfactory_grades_l777_77706


namespace simplify_t_l777_77792

theorem simplify_t (t : ℝ) (cbrt3 : ℝ) (h : cbrt3 ^ 3 = 3) 
  (ht : t = 1 / (1 - cbrt3)) : 
  t = - (1 + cbrt3 + cbrt3 ^ 2) / 2 := 
sorry

end simplify_t_l777_77792


namespace total_weight_of_sections_l777_77788

theorem total_weight_of_sections :
  let doll_length := 5
  let doll_weight := 29 / 8
  let tree_length := 4
  let tree_weight := 2.8
  let section_length := 2
  let doll_weight_per_meter := doll_weight / doll_length
  let tree_weight_per_meter := tree_weight / tree_length
  let doll_section_weight := doll_weight_per_meter * section_length
  let tree_section_weight := tree_weight_per_meter * section_length
  doll_section_weight + tree_section_weight = 57 / 20 :=
sorry

end total_weight_of_sections_l777_77788


namespace age_ratio_l777_77776
open Nat

theorem age_ratio (B A x : ℕ) (h1 : B - 4 = 2 * (A - 4)) 
                                (h2 : B - 8 = 3 * (A - 8)) 
                                (h3 : (B + x) / (A + x) = 3 / 2) : 
                                x = 4 :=
by
  sorry

end age_ratio_l777_77776


namespace delivery_meals_l777_77708

theorem delivery_meals (M P : ℕ) 
  (h1 : P = 8 * M) 
  (h2 : M + P = 27) : 
  M = 3 := by
  sorry

end delivery_meals_l777_77708


namespace probability_all_choose_paper_l777_77774

-- Given conditions
def probability_choice_is_paper := 1 / 3

-- The theorem to be proved
theorem probability_all_choose_paper :
  probability_choice_is_paper ^ 3 = 1 / 27 :=
sorry

end probability_all_choose_paper_l777_77774


namespace intersection_A_B_union_A_B_complement_intersection_A_B_l777_77764

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def A_inter_B : Set ℝ := { x | 2 ≤ x ∧ x < 6 }
def A_union_B : Set ℝ := { x | 1 < x ∧ x ≤ 8 }
def A_compl_inter_B : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = A_inter_B := by
  sorry

theorem union_A_B :
  A ∪ B = A_union_B := by
  sorry

theorem complement_intersection_A_B :
  (Aᶜ ∩ B) = A_compl_inter_B := by
  sorry

end intersection_A_B_union_A_B_complement_intersection_A_B_l777_77764


namespace intersection_M_N_l777_77793

def set_M : Set ℝ := { x : ℝ | -3 ≤ x ∧ x < 4 }
def set_N : Set ℝ := { x : ℝ | x^2 - 2 * x - 8 ≤ 0 }

theorem intersection_M_N : (set_M ∩ set_N) = { x : ℝ | -2 ≤ x ∧ x < 4 } :=
sorry

end intersection_M_N_l777_77793


namespace sets_relationship_l777_77784

def M : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 3 * k - 2}
def P : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def S : Set ℤ := {x : ℤ | ∃ m : ℤ, x = 6 * m + 1}

theorem sets_relationship : S ⊆ P ∧ M = P := by
  sorry

end sets_relationship_l777_77784


namespace winning_ticket_probability_l777_77779

open BigOperators

-- Calculate n choose k
def choose (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def probability_PowerBall := (1 : ℚ) / 30
def probability_LuckyBalls := (1 : ℚ) / choose 49 6

-- Theorem to prove the result
theorem winning_ticket_probability :
  probability_PowerBall * probability_LuckyBalls = (1 : ℚ) / 419514480 := by
  sorry

end winning_ticket_probability_l777_77779


namespace domain_of_f_l777_77752

def domain_f := {x : ℝ | 2 * x - 3 > 0}

theorem domain_of_f : ∀ x : ℝ, x ∈ domain_f ↔ x > 3 / 2 := 
by
  intro x
  simp [domain_f]
  sorry

end domain_of_f_l777_77752


namespace slope_of_dividing_line_l777_77710

/--
Given a rectangle with vertices at (0,0), (0,4), (5,4), (5,2),
and a right triangle with vertices at (5,2), (7,2), (5,0),
prove that the slope of the line through the origin that divides the area
of this L-shaped region exactly in half is 16/11.
-/
theorem slope_of_dividing_line :
  let rectangle_area := 5 * 4
  let triangle_area := (1 / 2) * 2 * 2
  let total_area := rectangle_area + triangle_area
  let half_area := total_area / 2
  let x_division := half_area / 4
  let slope := 4 / x_division
  slope = 16 / 11 :=
by
  sorry

end slope_of_dividing_line_l777_77710


namespace flour_already_added_l777_77721

theorem flour_already_added (sugar flour salt additional_flour : ℕ) 
  (h1 : sugar = 9) 
  (h2 : flour = 14) 
  (h3 : salt = 40)
  (h4 : additional_flour = sugar + 1) : 
  flour - additional_flour = 4 :=
by
  sorry

end flour_already_added_l777_77721


namespace base_angle_of_isosceles_triangle_l777_77773

-- Definitions corresponding to the conditions
def isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a = b ∧ A + B + C = 180) ∧ A = 40 -- Isosceles and sum of angles is 180° with apex angle A = 40°

-- The theorem to be proven
theorem base_angle_of_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :
  isosceles_triangle a b c A B C → B = 70 :=
by
  intros h
  sorry

end base_angle_of_isosceles_triangle_l777_77773


namespace baseball_card_decrease_l777_77782

theorem baseball_card_decrease (V₀ : ℝ) (V₁ V₂ : ℝ)
  (h₁: V₁ = V₀ * (1 - 0.20))
  (h₂: V₂ = V₁ * (1 - 0.20)) :
  ((V₀ - V₂) / V₀) * 100 = 36 :=
by
  sorry

end baseball_card_decrease_l777_77782


namespace prove_s90_zero_l777_77731

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 0) + (n * (n - 1) * (a 1 - a 0)) / 2)

theorem prove_s90_zero (a : ℕ → ℕ) (h_arith : is_arithmetic_sequence a) (h : sum_of_first_n_terms a 30 = sum_of_first_n_terms a 60) :
  sum_of_first_n_terms a 90 = 0 :=
sorry

end prove_s90_zero_l777_77731


namespace function_range_of_roots_l777_77751

theorem function_range_of_roots (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : a > 1 := 
sorry

end function_range_of_roots_l777_77751


namespace product_of_values_of_t_squared_eq_49_l777_77707

theorem product_of_values_of_t_squared_eq_49 :
  (∀ t, t^2 = 49 → t = 7 ∨ t = -7) →
  (7 * -7 = -49) :=
by
  intros h
  sorry

end product_of_values_of_t_squared_eq_49_l777_77707


namespace min_value_expression_l777_77717

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) :=
by
  sorry

end min_value_expression_l777_77717


namespace percentage_dogs_movies_l777_77724

-- Definitions from conditions
def total_students : ℕ := 30
def students_preferring_dogs_videogames : ℕ := total_students / 2
def students_preferring_dogs : ℕ := 18
def students_preferring_dogs_movies : ℕ := students_preferring_dogs - students_preferring_dogs_videogames

-- Theorem statement
theorem percentage_dogs_movies : (students_preferring_dogs_movies * 100 / total_students) = 10 := by
  sorry

end percentage_dogs_movies_l777_77724


namespace find_daily_wage_c_l777_77733

noncomputable def daily_wage_c (total_earning : ℕ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (days_d : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) (ratio_d : ℕ) : ℝ :=
  let total_ratio := days_a * ratio_a + days_b * ratio_b + days_c * ratio_c + days_d * ratio_d
  let x := total_earning / total_ratio
  ratio_c * x

theorem find_daily_wage_c :
  daily_wage_c 3780 6 9 4 12 3 4 5 7 = 119.60 :=
by
  sorry

end find_daily_wage_c_l777_77733


namespace find_pairs_l777_77772

theorem find_pairs (p q : ℤ) (a b : ℤ) :
  (p^2 - 4 * q = a^2) ∧ (q^2 - 4 * p = b^2) ↔ 
    (p = 4 ∧ q = 4) ∨ (p = 9 ∧ q = 8) ∨ (p = 8 ∧ q = 9) :=
by
  sorry

end find_pairs_l777_77772


namespace razors_blades_equation_l777_77746

/-- Given the number of razors sold x,
each razor sold brings a profit of 30 yuan,
each blade sold incurs a loss of 0.5 yuan,
the number of blades sold is twice the number of razors sold,
and the total profit from these two products is 5800 yuan,
prove that the linear equation is -0.5 * 2 * x + 30 * x = 5800 -/
theorem razors_blades_equation (x : ℝ) :
  -0.5 * 2 * x + 30 * x = 5800 := 
sorry

end razors_blades_equation_l777_77746


namespace gift_bags_needed_l777_77718

/-
  Constants
  total_expected: \(\mathbb{N}\) := 90        -- 50 people who will show up + 40 more who may show up
  total_prepared: \(\mathbb{N}\) := 30        -- 10 extravagant gift bags + 20 average gift bags

  The property to be proved:
  prove that (total_expected - total_prepared = 60)
-/

def total_expected : ℕ := 50 + 40
def total_prepared : ℕ := 10 + 20
def additional_needed := total_expected - total_prepared

theorem gift_bags_needed : additional_needed = 60 := by
  sorry

end gift_bags_needed_l777_77718


namespace parallel_line_through_P_perpendicular_line_through_P_l777_77778

-- Define point P
def P := (-4, 2)

-- Define line l
def l (x y : ℝ) := 3 * x - 2 * y - 7 = 0

-- Define the equation of the line parallel to l that passes through P
def parallel_line (x y : ℝ) := 3 * x - 2 * y + 16 = 0

-- Define the equation of the line perpendicular to l that passes through P
def perpendicular_line (x y : ℝ) := 2 * x + 3 * y + 2 = 0

-- Theorem 1: Prove that parallel_line is the equation of the line passing through P and parallel to l
theorem parallel_line_through_P :
  ∀ (x y : ℝ), 
    (parallel_line x y → x = -4 ∧ y = 2) :=
sorry

-- Theorem 2: Prove that perpendicular_line is the equation of the line passing through P and perpendicular to l
theorem perpendicular_line_through_P :
  ∀ (x y : ℝ), 
    (perpendicular_line x y → x = -4 ∧ y = 2) :=
sorry

end parallel_line_through_P_perpendicular_line_through_P_l777_77778


namespace find_k_l777_77737

theorem find_k (k : ℝ) : 
  let a := 6
  let b := 25
  let root := (-25 - Real.sqrt 369) / 12
  6 * root^2 + 25 * root + k = 0 → k = 32 / 3 :=
sorry

end find_k_l777_77737


namespace ordered_sets_equal_l777_77709

theorem ordered_sets_equal
  (n : ℕ) 
  (h_gcd : gcd n 6 = 1) 
  (a b : ℕ → ℕ) 
  (h_order_a : ∀ {i j}, i < j → a i < a j)
  (h_order_b : ∀ {i j}, i < j → b i < b j) 
  (h_sum : ∀ {j k l : ℕ}, 1 ≤ j → j < k → k < l → l ≤ n → a j + a k + a l = b j + b k + b l) : 
  ∀ (j : ℕ), 1 ≤ j → j ≤ n → a j = b j := 
sorry

end ordered_sets_equal_l777_77709


namespace bacteria_eradication_time_l777_77725

noncomputable def infected_bacteria (n : ℕ) : ℕ := n

theorem bacteria_eradication_time (n : ℕ) : ∃ t : ℕ, t = n ∧ (∃ infect: ℕ → ℕ, ∀ t < n, infect t ≤ n ∧ infect n = n ∧ (∀ k < n, infect k = 2^(n-k))) :=
by sorry

end bacteria_eradication_time_l777_77725


namespace circle_center_radius_proof_l777_77722

noncomputable def circle_center_radius (x y : ℝ) :=
  x^2 + y^2 - 4*x + 2*y + 2 = 0

theorem circle_center_radius_proof :
  ∀ x y : ℝ, circle_center_radius x y ↔ ((x - 2)^2 + (y + 1)^2 = 3) :=
by
  sorry

end circle_center_radius_proof_l777_77722


namespace find_shares_l777_77796

def shareA (B : ℝ) : ℝ := 3 * B
def shareC (B : ℝ) : ℝ := B - 25
def shareD (A B : ℝ) : ℝ := A + B - 10
def total_share (A B C D : ℝ) : ℝ := A + B + C + D

theorem find_shares :
  ∃ (A B C D : ℝ),
  A = 744.99 ∧
  B = 248.33 ∧
  C = 223.33 ∧
  D = 983.32 ∧
  A = shareA B ∧
  C = shareC B ∧
  D = shareD A B ∧
  total_share A B C D = 2200 := 
sorry

end find_shares_l777_77796


namespace minimize_z_l777_77780

theorem minimize_z (x y : ℝ) (h1 : 2 * x - y ≥ 0) (h2 : y ≥ x) (h3 : y ≥ -x + 2) :
  ∃ (x y : ℝ), (z = 2 * x + y) ∧ z = 8 / 3 :=
by
  sorry

end minimize_z_l777_77780


namespace toys_sold_in_first_week_l777_77791

/-
  Problem statement:
  An online toy store stocked some toys. It sold some toys at the first week and 26 toys at the second week.
  If it had 19 toys left and there were 83 toys in stock at the beginning, how many toys were sold in the first week?
-/

theorem toys_sold_in_first_week (initial_stock toys_left toys_sold_second_week : ℕ) 
  (h_initial_stock : initial_stock = 83) 
  (h_toys_left : toys_left = 19) 
  (h_toys_sold_second_week : toys_sold_second_week = 26) : 
  (initial_stock - toys_left - toys_sold_second_week) = 38 :=
by
  -- Proof goes here
  sorry

end toys_sold_in_first_week_l777_77791


namespace problem1_solution_problem2_solution_l777_77723

-- Problem 1
theorem problem1_solution (x y : ℝ) : (2 * x - y = 3) ∧ (x + y = 3) ↔ (x = 2 ∧ y = 1) := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) : (x / 4 + y / 3 = 3) ∧ (3 * x - 2 * (y - 1) = 11) ↔ (x = 6 ∧ y = 9 / 2) := by
  sorry

end problem1_solution_problem2_solution_l777_77723


namespace sum_digits_10_pow_85_minus_85_l777_77758

-- Define the function that computes the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

-- Define the specific problem for n = 10^85 - 85
theorem sum_digits_10_pow_85_minus_85 : 
  sum_of_digits (10^85 - 85) = 753 :=
by
  sorry

end sum_digits_10_pow_85_minus_85_l777_77758


namespace man_swims_speed_l777_77701

theorem man_swims_speed (v_m v_s : ℝ) (h_downstream : 28 = (v_m + v_s) * 2) (h_upstream : 12 = (v_m - v_s) * 2) : v_m = 10 := 
by sorry

end man_swims_speed_l777_77701


namespace number_smaller_than_neg3_exists_l777_77730

def numbers := [0, -1, -5, -1/2]

theorem number_smaller_than_neg3_exists : ∃ x ∈ numbers, x < -3 :=
by
  let x := -5
  have h : x ∈ numbers := by simp [numbers]
  have h_lt : x < -3 := by norm_num
  exact ⟨x, h, h_lt⟩ -- show that -5 meets the criteria

end number_smaller_than_neg3_exists_l777_77730


namespace max_value_m_l777_77741

noncomputable def exists_triangle_with_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_value_m (a b c : ℝ) (m : ℝ) (h1 : 0 < m) (h2 : abc ≤ 1/4) (h3 : 1/(a^2) + 1/(b^2) + 1/(c^2) < m) :
  m ≤ 9 ↔ exists_triangle_with_sides a b c :=
sorry

end max_value_m_l777_77741


namespace sum_of_ages_l777_77749

-- Definitions based on conditions
def age_relation1 (a b c : ℕ) : Prop := a = 20 + b + c
def age_relation2 (a b c : ℕ) : Prop := a^2 = 2000 + (b + c)^2

-- The statement to be proven
theorem sum_of_ages (a b c : ℕ) (h1 : age_relation1 a b c) (h2 : age_relation2 a b c) : a + b + c = 80 :=
by
  sorry

end sum_of_ages_l777_77749


namespace quadratic_unique_solution_pair_l777_77765

theorem quadratic_unique_solution_pair (a c : ℝ) (h₁ : a + c = 12) (h₂ : a < c) (h₃ : a * c = 9) :
  (a, c) = (6 - 3 * Real.sqrt 3, 6 + 3 * Real.sqrt 3) :=
by
  sorry

end quadratic_unique_solution_pair_l777_77765


namespace intersection_A_B_l777_77799

def A := {x : ℝ | x^2 - ⌊x⌋ = 2}
def B := {x : ℝ | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, Real.sqrt 3} :=
sorry

end intersection_A_B_l777_77799


namespace find_k_parallel_lines_l777_77727

theorem find_k_parallel_lines (k : ℝ) : 
  (∀ x y, (k - 1) * x + y + 2 = 0 → 
            (8 * x + (k + 1) * y + k - 1 = 0 → False)) → 
  k = 3 :=
sorry

end find_k_parallel_lines_l777_77727


namespace seating_arrangements_l777_77740

-- Number of ways to arrange a block of n items
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Groups
def dodgers : ℕ := 4
def marlins : ℕ := 3
def phillies : ℕ := 2

-- Total number of players
def total_players : ℕ := dodgers + marlins + phillies

-- Number of ways to arrange the blocks
def blocks_arrangements : ℕ := factorial 3

-- Internal arrangements within each block
def dodgers_arrangements : ℕ := factorial dodgers
def marlins_arrangements : ℕ := factorial marlins
def phillies_arrangements : ℕ := factorial phillies

-- Total number of ways to seat the players
def total_arrangements : ℕ :=
  blocks_arrangements * dodgers_arrangements * marlins_arrangements * phillies_arrangements

-- Prove that the total arrangements is 1728
theorem seating_arrangements : total_arrangements = 1728 := by
  sorry

end seating_arrangements_l777_77740


namespace volume_of_regular_tetrahedron_with_edge_length_1_l777_77726

-- We define the concepts needed: regular tetrahedron, edge length, and volume.
open Real

noncomputable def volume_of_regular_tetrahedron (a : ℝ) : ℝ :=
  let base_area := (sqrt 3 / 4) * a^2
  let height := sqrt (a^2 - (a * (sqrt 3 / 3))^2)
  (1 / 3) * base_area * height

-- The problem statement and our goal to prove:
theorem volume_of_regular_tetrahedron_with_edge_length_1 :
  volume_of_regular_tetrahedron 1 = sqrt 2 / 12 := sorry

end volume_of_regular_tetrahedron_with_edge_length_1_l777_77726


namespace geometric_sequence_a4_a7_l777_77715

theorem geometric_sequence_a4_a7 (a : ℕ → ℝ) (h1 : ∃ a₁ a₁₀, a₁ * a₁₀ = -6 ∧ a 1 = a₁ ∧ a 10 = a₁₀) :
  a 4 * a 7 = -6 :=
sorry

end geometric_sequence_a4_a7_l777_77715


namespace triangle_side_lengths_l777_77759

variable {c z m : ℕ}

axiom condition1 : 3 * c + z + m = 43
axiom condition2 : c + z + 3 * m = 35
axiom condition3 : 2 * (c + z + m) = 46

theorem triangle_side_lengths : c = 10 ∧ z = 7 ∧ m = 6 := 
by 
  sorry

end triangle_side_lengths_l777_77759


namespace solve_rational_eq_l777_77787

theorem solve_rational_eq {x : ℝ} (h : 1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 4 * x - 5) + 1 / (x^2 - 15 * x - 12) = 0) :
  x = 3 ∨ x = -4 ∨ x = 1 ∨ x = -5 :=
by {
  sorry
}

end solve_rational_eq_l777_77787


namespace reduced_price_l777_77790

theorem reduced_price (P R : ℝ) (Q : ℝ) (h₁ : R = 0.80 * P) 
                      (h₂ : 800 = Q * P) 
                      (h₃ : 800 = (Q + 5) * R) 
                      : R = 32 :=
by
  -- Code that proves the theorem goes here.
  sorry

end reduced_price_l777_77790


namespace pow_mod_remainder_l777_77768

theorem pow_mod_remainder :
  (2^2013 % 11) = 8 :=
sorry

end pow_mod_remainder_l777_77768


namespace car_average_speed_l777_77713

theorem car_average_speed
  (d1 d2 t1 t2 : ℕ)
  (h1 : d1 = 85)
  (h2 : d2 = 45)
  (h3 : t1 = 1)
  (h4 : t2 = 1) :
  let total_distance := d1 + d2
  let total_time := t1 + t2
  (total_distance / total_time = 65) :=
by
  sorry

end car_average_speed_l777_77713


namespace geometric_mean_4_16_l777_77738

theorem geometric_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
sorry

end geometric_mean_4_16_l777_77738


namespace remaining_requests_after_7_days_l777_77712

-- Definitions based on the conditions
def dailyRequests : ℕ := 8
def dailyWork : ℕ := 4
def days : ℕ := 7

-- Theorem statement representing our final proof problem
theorem remaining_requests_after_7_days : 
  (dailyRequests * days - dailyWork * days) + dailyRequests * days = 84 := by
  sorry

end remaining_requests_after_7_days_l777_77712


namespace polygon_sides_l777_77747

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_sides_l777_77747


namespace boys_in_classroom_l777_77763

-- Definitions of the conditions
def total_children := 45
def girls_fraction := 1 / 3

-- The theorem we want to prove
theorem boys_in_classroom : (2 / 3) * total_children = 30 := by
  sorry

end boys_in_classroom_l777_77763


namespace towel_length_decrease_l777_77760

theorem towel_length_decrease (L B : ℝ) (HL1: L > 0) (HB1: B > 0)
  (length_percent_decr : ℝ) (breadth_decr : B' = 0.8 * B) 
  (area_decr : (L' * B') = 0.64 * (L * B)) :
  (L' = 0.8 * L) ∧ (length_percent_decrease = 20) := by
  sorry

end towel_length_decrease_l777_77760


namespace javier_initial_games_l777_77705

/--
Javier plays 2 baseball games a week. In each of his first some games, 
he averaged 2 hits. If he has 10 games left, he has to average 5 hits 
a game to bring his average for the season up to 3 hits a game. 
Prove that the number of games Javier initially played is 20.
-/
theorem javier_initial_games (x : ℕ) :
  (2 * x + 5 * 10) / (x + 10) = 3 → x = 20 :=
by
  sorry

end javier_initial_games_l777_77705


namespace bruce_bank_ratio_l777_77719

noncomputable def bruce_aunt : ℝ := 75
noncomputable def bruce_grandfather : ℝ := 150
noncomputable def bruce_bank : ℝ := 45
noncomputable def bruce_total : ℝ := bruce_aunt + bruce_grandfather
noncomputable def bruce_ratio : ℝ := bruce_bank / bruce_total

theorem bruce_bank_ratio :
  bruce_ratio = 1 / 5 :=
by
  -- proof goes here
  sorry

end bruce_bank_ratio_l777_77719


namespace loom_weaving_rate_l777_77735

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) 
    (h1 : total_cloth = 25) (h2 : total_time = 195.3125) : 
    total_cloth / total_time = 0.128 :=
sorry

end loom_weaving_rate_l777_77735


namespace triangle_side_lengths_l777_77798

theorem triangle_side_lengths 
  (r : ℝ) (CD : ℝ) (DB : ℝ) 
  (h_r : r = 4) 
  (h_CD : CD = 8) 
  (h_DB : DB = 10) :
  ∃ (AB AC : ℝ), AB = 14.5 ∧ AC = 12.5 :=
by
  sorry

end triangle_side_lengths_l777_77798


namespace inequality_l777_77755

theorem inequality (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
by
  sorry

end inequality_l777_77755


namespace factorization_correct_l777_77732

theorem factorization_correct (x : ℝ) :
    x^2 - 3 * x - 4 = (x + 1) * (x - 4) :=
  sorry

end factorization_correct_l777_77732


namespace simplify_expression_l777_77704

variables (y : ℝ)

theorem simplify_expression : 
  3 * y + 4 * y^2 - 2 - (8 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 10 :=
by sorry

end simplify_expression_l777_77704


namespace length_more_than_breadth_by_200_l777_77795

-- Definitions and conditions
def rectangular_floor_length := 23
def painting_cost := 529
def painting_rate := 3
def floor_area := painting_cost / painting_rate
def floor_breadth := floor_area / rectangular_floor_length

-- Prove that the length is more than the breadth by 200%
theorem length_more_than_breadth_by_200 : 
  rectangular_floor_length = floor_breadth * (1 + 200 / 100) :=
sorry

end length_more_than_breadth_by_200_l777_77795


namespace range_of_a_l777_77783

noncomputable def f (x : ℝ) := -Real.exp x - x
noncomputable def g (a x : ℝ) := a * x + Real.cos x

theorem range_of_a :
  (∀ x : ℝ, ∃ y : ℝ, (g a y - g a y) / (y - y) * ((f x - f x) / (x - x)) = -1) →
  (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l777_77783


namespace total_words_story_l777_77744

def words_per_line : ℕ := 10
def lines_per_page : ℕ := 20
def pages_filled : ℚ := 1.5
def words_left : ℕ := 100

theorem total_words_story : 
    words_per_line * lines_per_page * pages_filled + words_left = 400 := 
by
sorry

end total_words_story_l777_77744


namespace onion_pieces_per_student_l777_77771

theorem onion_pieces_per_student (total_pizzas : ℕ) (slices_per_pizza : ℕ)
  (cheese_pieces_leftover : ℕ) (onion_pieces_leftover : ℕ) (students : ℕ) (cheese_per_student : ℕ)
  (h1 : total_pizzas = 6) (h2 : slices_per_pizza = 18) (h3 : cheese_pieces_leftover = 8) (h4 : onion_pieces_leftover = 4)
  (h5 : students = 32) (h6 : cheese_per_student = 2) :
  ((total_pizzas * slices_per_pizza) - cheese_pieces_leftover - onion_pieces_leftover - (students * cheese_per_student)) / students = 1 := 
by
  sorry

end onion_pieces_per_student_l777_77771


namespace line_through_points_a_plus_b_l777_77720

theorem line_through_points_a_plus_b :
  ∃ a b : ℝ, (∀ x y : ℝ, (y = a * x + b) → ((x, y) = (6, 7)) ∨ ((x, y) = (10, 23))) ∧ (a + b = -13) :=
sorry

end line_through_points_a_plus_b_l777_77720


namespace tangency_of_abs_and_circle_l777_77700

theorem tangency_of_abs_and_circle (a : ℝ) (ha_pos : a > 0) (ha_ne_two : a ≠ 2) :
    (y = abs x ∧ ∀ x, y = abs x → x^2 + (y - a)^2 = 2 * (a - 2)^2)
    → (a = 4/3 ∨ a = 4) := sorry

end tangency_of_abs_and_circle_l777_77700


namespace length_of_tank_l777_77761

namespace TankProblem

def field_length : ℝ := 90
def field_breadth : ℝ := 50
def field_area : ℝ := field_length * field_breadth

def tank_breadth : ℝ := 20
def tank_depth : ℝ := 4

def earth_volume (L : ℝ) : ℝ := L * tank_breadth * tank_depth

def remaining_field_area (L : ℝ) : ℝ := field_area - L * tank_breadth

def height_increase : ℝ := 0.5

theorem length_of_tank (L : ℝ) :
  earth_volume L = remaining_field_area L * height_increase →
  L = 25 :=
by
  sorry

end TankProblem

end length_of_tank_l777_77761


namespace point_exists_if_square_or_rhombus_l777_77757

-- Definitions to state the problem
structure Point (α : Type*) := (x : α) (y : α)
structure Rectangle (α : Type*) := (A B C D : Point α)

-- Definition of equidistant property
def isEquidistant (α : Type*) [LinearOrderedField α] (P : Point α) (R : Rectangle α) : Prop :=
  let d1 := abs (P.y - R.A.y)
  let d2 := abs (P.y - R.C.y)
  let d3 := abs (P.x - R.A.x)
  let d4 := abs (P.x - R.B.x)
  d1 = d2 ∧ d2 = d3 ∧ d3 = d4

-- Theorem stating the problem
theorem point_exists_if_square_or_rhombus {α : Type*} [LinearOrderedField α]
  (R : Rectangle α) : 
  (∃ P : Point α, isEquidistant α P R) ↔ 
  (∃ (a b : α), (a ≠ b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b) ∨ 
                (a = b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b)) :=
sorry

end point_exists_if_square_or_rhombus_l777_77757


namespace golden_ratio_expression_l777_77716

variables (R : ℝ)
noncomputable def divide_segment (R : ℝ) := R^(R^(R^2 + 1/R) + 1/R) + 1/R

theorem golden_ratio_expression :
  (R = (1 / (1 + R))) →
  divide_segment R = 2 :=
by
  sorry

end golden_ratio_expression_l777_77716


namespace sin_sum_identity_l777_77753

theorem sin_sum_identity 
  (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) : 
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := 
by 
  sorry

end sin_sum_identity_l777_77753


namespace thickness_relation_l777_77775

noncomputable def a : ℝ := (1/3) * Real.sin (1/2)
noncomputable def b : ℝ := (1/2) * Real.sin (1/3)
noncomputable def c : ℝ := (1/3) * Real.cos (7/8)

theorem thickness_relation : c > b ∧ b > a := by
  sorry

end thickness_relation_l777_77775


namespace original_fund_was_830_l777_77797

/- Define the number of employees as a variable -/
variables (n : ℕ)

/- Define the conditions given in the problem -/
def initial_fund := 60 * n - 10
def new_fund_after_distributing_50 := initial_fund - 50 * n
def remaining_fund := 130

/- State the proof goal -/
theorem original_fund_was_830 :
  initial_fund = 830 :=
by sorry

end original_fund_was_830_l777_77797


namespace symmetric_line_eq_l777_77736

-- Define the original line equation
def original_line (x: ℝ) : ℝ := -2 * x - 3

-- Define the symmetric line with respect to y-axis
def symmetric_line (x: ℝ) : ℝ := 2 * x - 3

-- The theorem stating the symmetric line with respect to the y-axis
theorem symmetric_line_eq : (∀ x: ℝ, original_line (-x) = symmetric_line x) :=
by
  -- Proof goes here
  sorry

end symmetric_line_eq_l777_77736


namespace range_of_t_circle_largest_area_eq_point_P_inside_circle_l777_77769

open Real

-- Defining the given equation representing the trajectory of a point on a circle
def circle_eq (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16 * t^4 + 9 = 0

-- Problem 1: Proving the range of t
theorem range_of_t : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → -1/7 < t ∧ t < 1 :=
sorry

-- Problem 2: Proving the equation of the circle with the largest area
theorem circle_largest_area_eq : 
  ∃ t : ℝ, t = 3/7 ∧ (∀ x y : ℝ, circle_eq x y (3/7)) → 
  ∀ x y : ℝ, (x - 24/7)^2 + (y + 13/49)^2 = 16/7 :=
sorry

-- Problem 3: Proving the range of t for point P to be inside the circle
theorem point_P_inside_circle : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → 
  (0 < t ∧ t < 3/4) :=
sorry

end range_of_t_circle_largest_area_eq_point_P_inside_circle_l777_77769


namespace find_x_y_l777_77786

theorem find_x_y (x y : ℝ) (h1 : (10 + 25 + x + y) / 4 = 20) (h2 : x * y = 156) :
  (x = 12 ∧ y = 33) ∨ (x = 33 ∧ y = 12) :=
by
  sorry

end find_x_y_l777_77786


namespace smallest_candies_value_l777_77754

def smallest_valid_n := ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 9 = 2 ∧ n % 7 = 5 ∧ ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 9 = 2 ∧ m % 7 = 5 → n ≤ m

theorem smallest_candies_value : ∃ n : ℕ, smallest_valid_n ∧ n = 101 := 
by {
  sorry  
}

end smallest_candies_value_l777_77754


namespace find_m_value_l777_77781

def magic_box_output (a b : ℝ) : ℝ := a^2 + b - 1

theorem find_m_value :
  ∃ m : ℝ, (magic_box_output m (-2 * m) = 2) ↔ (m = 3 ∨ m = -1) :=
by
  sorry

end find_m_value_l777_77781


namespace decodeMINT_l777_77739

def charToDigit (c : Char) : Option Nat :=
  match c with
  | 'G' => some 0
  | 'R' => some 1
  | 'E' => some 2
  | 'A' => some 3
  | 'T' => some 4
  | 'M' => some 5
  | 'I' => some 6
  | 'N' => some 7
  | 'D' => some 8
  | 'S' => some 9
  | _   => none

def decodeWord (word : String) : Option Nat :=
  let digitsOption := word.toList.map charToDigit
  if digitsOption.all Option.isSome then
    let digits := digitsOption.map Option.get!
    some (digits.foldl (λ acc d => 10 * acc + d) 0)
  else
    none

theorem decodeMINT : decodeWord "MINT" = some 5674 := by
  sorry

end decodeMINT_l777_77739


namespace trigonometric_relationship_l777_77711

noncomputable def a : ℝ := Real.sin (393 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (50 * Real.pi / 180)

theorem trigonometric_relationship : a < b ∧ b < c := by
  sorry

end trigonometric_relationship_l777_77711


namespace 1_part1_2_part2_l777_77714

/-
Define M and N sets
-/
def M : Set ℝ := {x | x ≥ 1 / 2}
def N : Set ℝ := {y | y ≤ 1}

/-
Theorem 1: Difference set M - N
-/
theorem part1 : (M \ N) = {x | x > 1} := by
  sorry

/-
Define A and B sets and the condition A - B = ∅
-/
def A (a : ℝ) : Set ℝ := {x | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {y | -1 / 2 < y ∧ y ≤ 2}

/-
Theorem 2: Range of values for a
-/
theorem part2 (a : ℝ) (h : A a \ B = ∅) : a ∈ Set.Iio (-12) ∪ Set.Ici 3 := by
  sorry

end 1_part1_2_part2_l777_77714


namespace thyme_pots_count_l777_77734

theorem thyme_pots_count
  (basil_pots : ℕ := 3)
  (rosemary_pots : ℕ := 9)
  (leaves_per_basil_pot : ℕ := 4)
  (leaves_per_rosemary_pot : ℕ := 18)
  (leaves_per_thyme_pot : ℕ := 30)
  (total_leaves : ℕ := 354)
  : (total_leaves - (basil_pots * leaves_per_basil_pot + rosemary_pots * leaves_per_rosemary_pot)) / leaves_per_thyme_pot = 6 :=
by
  sorry

end thyme_pots_count_l777_77734


namespace geometric_common_ratio_l777_77785

theorem geometric_common_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (a₁ * (1 - q ^ 3)) / (1 - q) / ((a₁ * (1 - q ^ 2)) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  -- Proof omitted
  sorry

end geometric_common_ratio_l777_77785


namespace polynomial_simplification_l777_77767

theorem polynomial_simplification (y : ℤ) : 
  (2 * y - 1) * (4 * y ^ 10 + 2 * y ^ 9 + 4 * y ^ 8 + 2 * y ^ 7) = 8 * y ^ 11 + 6 * y ^ 9 - 2 * y ^ 7 :=
by 
  sorry

end polynomial_simplification_l777_77767


namespace length_of_DG_l777_77762

theorem length_of_DG {AB BC DG DF : ℝ} (h1 : AB = 8) (h2 : BC = 10) (h3 : DG = DF) 
  (h4 : 1/5 * (AB * BC) = 1/2 * DG^2) : DG = 4 * Real.sqrt 2 :=
by sorry

end length_of_DG_l777_77762


namespace smallest_number_increased_by_3_divisible_l777_77729

theorem smallest_number_increased_by_3_divisible (n : ℤ) 
    (h1 : (n + 3) % 18 = 0)
    (h2 : (n + 3) % 70 = 0)
    (h3 : (n + 3) % 25 = 0)
    (h4 : (n + 3) % 21 = 0) : 
    n = 3147 :=
by
  sorry

end smallest_number_increased_by_3_divisible_l777_77729


namespace complement_of_irreducible_proper_fraction_is_irreducible_l777_77742

theorem complement_of_irreducible_proper_fraction_is_irreducible 
  (a b : ℤ) (h0 : 0 < a) (h1 : a < b) (h2 : Int.gcd a b = 1) : Int.gcd (b - a) b = 1 :=
sorry

end complement_of_irreducible_proper_fraction_is_irreducible_l777_77742


namespace sequence_infinite_pos_neg_l777_77770

theorem sequence_infinite_pos_neg (a : ℕ → ℝ)
  (h : ∀ k : ℕ, a (k + 1) = (k * a k + 1) / (k - a k)) :
  ∃ (P N : ℕ → Prop), (∀ n, P n ↔ 0 < a n) ∧ (∀ n, N n ↔ a n < 0) ∧ 
  (∀ m, ∃ n, n > m ∧ P n) ∧ (∀ m, ∃ n, n > m ∧ N n) := 
sorry

end sequence_infinite_pos_neg_l777_77770
