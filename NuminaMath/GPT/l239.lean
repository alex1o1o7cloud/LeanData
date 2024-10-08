import Mathlib

namespace math_club_team_selection_l239_239786

open Nat

-- Lean statement of the problem
theorem math_club_team_selection : 
  (choose 7 3) * (choose 9 3) = 2940 :=
by 
  sorry

end math_club_team_selection_l239_239786


namespace sum_of_primes_no_solution_congruence_l239_239885

theorem sum_of_primes_no_solution_congruence :
  2 + 5 = 7 :=
by
  sorry

end sum_of_primes_no_solution_congruence_l239_239885


namespace minimum_score_to_win_l239_239465

namespace CompetitionPoints

-- Define points awarded for each position
def points_first : ℕ := 5
def points_second : ℕ := 3
def points_third : ℕ := 1

-- Define the number of competitions
def competitions : ℕ := 3

-- Total points in one competition
def total_points_one_competition : ℕ := points_first + points_second + points_third

-- Total points in all competitions
def total_points_all_competitions : ℕ := total_points_one_competition * competitions

theorem minimum_score_to_win : ∃ m : ℕ, m = 13 ∧ (∀ s : ℕ, s < 13 → ¬ ∃ c1 c2 c3 : ℕ, 
  c1 ≤ competitions ∧ c2 ≤ competitions ∧ c3 ≤ competitions ∧ 
  ((c1 * points_first) + (c2 * points_second) + (c3 * points_third)) = s) :=
by {
  sorry
}

end CompetitionPoints

end minimum_score_to_win_l239_239465


namespace remainder_3_pow_100_plus_5_mod_8_l239_239773

theorem remainder_3_pow_100_plus_5_mod_8 : (3^100 + 5) % 8 = 6 := by
  sorry

end remainder_3_pow_100_plus_5_mod_8_l239_239773


namespace c_share_l239_239854

theorem c_share (A B C : ℕ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : A + B + C = 406) : 
    C = 232 := by 
    sorry

end c_share_l239_239854


namespace seventh_graders_problems_l239_239673

theorem seventh_graders_problems (n : ℕ) (S : ℕ) (a : ℕ) (h1 : a > (S - a) / 5) (h2 : a < (S - a) / 3) : n = 5 :=
  sorry

end seventh_graders_problems_l239_239673


namespace number_of_students_who_bought_2_pencils_l239_239602

variable (a b c : ℕ)     -- a is the number of students buying 1 pencil, b is the number of students buying 2 pencils, c is the number of students buying 3 pencils.
variable (total_students total_pencils : ℕ) -- total_students is 36, total_pencils is 50
variable (students_condition1 students_condition2 : ℕ) -- conditions: students_condition1 for the sum of the students, students_condition2 for the sum of the pencils

theorem number_of_students_who_bought_2_pencils :
  total_students = 36 ∧
  total_pencils = 50 ∧
  total_students = a + b + c ∧
  total_pencils = a * 1 + b * 2 + c * 3 ∧
  a = 2 * (b + c) → 
  b = 10 :=
by sorry

end number_of_students_who_bought_2_pencils_l239_239602


namespace rectangle_area_l239_239883

/-- A figure is formed by a triangle and a rectangle, using 60 equal sticks.
Each side of the triangle uses 6 sticks, and each stick measures 5 cm in length.
Prove that the area of the rectangle is 2250 cm². -/
theorem rectangle_area (sticks_total : ℕ) (sticks_per_side_triangle : ℕ) (stick_length_cm : ℕ)
    (sticks_used_triangle : ℕ) (sticks_left_rectangle : ℕ) (sticks_per_width_rectangle : ℕ)
    (width_sticks_rectangle : ℕ) (length_sticks_rectangle : ℕ) (width_cm : ℕ) (length_cm : ℕ)
    (area_rectangle : ℕ) 
    (h_sticks_total : sticks_total = 60)
    (h_sticks_per_side_triangle : sticks_per_side_triangle = 6)
    (h_stick_length_cm : stick_length_cm = 5)
    (h_sticks_used_triangle  : sticks_used_triangle = sticks_per_side_triangle * 3)
    (h_sticks_left_rectangle : sticks_left_rectangle = sticks_total - sticks_used_triangle)
    (h_sticks_per_width_rectangle : sticks_per_width_rectangle = 6 * 2) 
    (h_width_sticks_rectangle : width_sticks_rectangle = 6)
    (h_length_sticks_rectangle : length_sticks_rectangle = (sticks_left_rectangle - sticks_per_width_rectangle) / 2)
    (h_width_cm : width_cm = width_sticks_rectangle * stick_length_cm)
    (h_length_cm : length_cm = length_sticks_rectangle * stick_length_cm)
    (h_area_rectangle : area_rectangle = width_cm * length_cm) :
    area_rectangle = 2250 := 
by sorry

end rectangle_area_l239_239883


namespace intersection_of_A_and_B_l239_239484

def setA : Set ℕ := {1, 2, 3}
def setB : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : setA ∩ setB = {2} :=
by
  sorry

end intersection_of_A_and_B_l239_239484


namespace smallest_positive_value_l239_239674

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℝ), k = 2 ∧ k = (↑(a - b) / ↑(a + b) + ↑(a + b) / ↑(a - b)) :=
sorry

end smallest_positive_value_l239_239674


namespace real_solution_count_l239_239886

theorem real_solution_count : 
  ∃ (n : ℕ), n = 1 ∧
    ∀ x : ℝ, 
      (3 * x / (x ^ 2 + 2 * x + 4) + 4 * x / (x ^ 2 - 4 * x + 4) = 1) ↔ (x = 2) :=
by
  sorry

end real_solution_count_l239_239886


namespace no_negative_roots_of_polynomial_l239_239145

def polynomial (x : ℝ) := x^4 - 5 * x^3 - 4 * x^2 - 7 * x + 4

theorem no_negative_roots_of_polynomial :
  ¬ ∃ (x : ℝ), x < 0 ∧ polynomial x = 0 :=
by
  sorry

end no_negative_roots_of_polynomial_l239_239145


namespace num_outfits_l239_239278

-- Define the number of trousers, shirts, and jackets available
def num_trousers : Nat := 5
def num_shirts : Nat := 6
def num_jackets : Nat := 4

-- Define the main theorem
theorem num_outfits (t : Nat) (s : Nat) (j : Nat) (ht : t = num_trousers) (hs : s = num_shirts) (hj : j = num_jackets) :
  t * s * j = 120 :=
by 
  rw [ht, hs, hj]
  exact rfl

end num_outfits_l239_239278


namespace original_ratio_l239_239972

theorem original_ratio (x y : ℤ) (h₁ : y = 72) (h₂ : (x + 6) / y = 1 / 3) : y / x = 4 := 
by
  sorry

end original_ratio_l239_239972


namespace sum_of_box_dimensions_l239_239189

theorem sum_of_box_dimensions (X Y Z : ℝ) (h1 : X * Y = 32) (h2 : X * Z = 50) (h3 : Y * Z = 80) :
  X + Y + Z = 25.5 * Real.sqrt 2 :=
by sorry

end sum_of_box_dimensions_l239_239189


namespace uma_fraction_part_l239_239113

theorem uma_fraction_part (r s t u : ℕ) 
  (hr : r = 6) 
  (hs : s = 5) 
  (ht : t = 7) 
  (hu : u = 8) 
  (shared_amount: ℕ)
  (hr_amount: shared_amount = r / 6)
  (hs_amount: shared_amount = s / 5)
  (ht_amount: shared_amount = t / 7)
  (hu_amount: shared_amount = u / 8) :
  ∃ total : ℕ, ∃ uma_total : ℕ, uma_total * 13 = 2 * total :=
sorry

end uma_fraction_part_l239_239113


namespace passed_candidates_l239_239386

theorem passed_candidates (P F : ℕ) (h1 : P + F = 120) (h2 : 39 * P + 15 * F = 35 * 120) : P = 100 :=
by
  sorry

end passed_candidates_l239_239386


namespace Tara_loss_point_l239_239460

theorem Tara_loss_point :
  ∀ (clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal) 
  (H1 : initial_savings = 10)
  (H2 : clarinet_cost = 90)
  (H3 : book_price = 5)
  (H4 : total_books_sold = 25)
  (H5 : books_sold_to_goal = (clarinet_cost - initial_savings) / book_price)
  (H6 : additional_books = total_books_sold - books_sold_to_goal),
  additional_books * book_price = 45 :=
by
  intros clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal
  intros H1 H2 H3 H4 H5 H6
  sorry

end Tara_loss_point_l239_239460


namespace dist_between_centers_l239_239142

noncomputable def dist_centers_tangent_circles : ℝ :=
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  Real.sqrt 2 * (a₁ - a₂)

theorem dist_between_centers :
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  let C₁ := (a₁, a₁)
  let C₂ := (a₂, a₂)
  dist_centers_tangent_circles = 8 :=
by
  sorry

end dist_between_centers_l239_239142


namespace r_cube_plus_inv_r_cube_eq_zero_l239_239613

theorem r_cube_plus_inv_r_cube_eq_zero {r : ℝ} (h : (r + 1/r)^2 = 3) : r^3 + 1/r^3 = 0 := 
sorry

end r_cube_plus_inv_r_cube_eq_zero_l239_239613


namespace find_percentage_l239_239877

theorem find_percentage (P : ℝ) : 
  (P / 100) * 100 - 40 = 30 → P = 70 :=
by
  intros h
  sorry

end find_percentage_l239_239877


namespace general_term_l239_239625

def S (n : ℕ) : ℕ := n^2 + 3 * n

def a (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = 2 * n + 2 :=
by {
  sorry
}

end general_term_l239_239625


namespace multiple_with_digits_l239_239538

theorem multiple_with_digits (n : ℕ) (h : n > 0) :
  ∃ (m : ℕ), (m % n = 0) ∧ (m < 10 ^ n) ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) :=
by
  sorry

end multiple_with_digits_l239_239538


namespace minimum_value_of_xy_l239_239513

theorem minimum_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x * y ≥ 8 :=
sorry

end minimum_value_of_xy_l239_239513


namespace necessary_not_sufficient_condition_l239_239170
-- Import the necessary libraries

-- Define the real number condition
def real_number (a : ℝ) : Prop := true

-- Define line l1
def line_l1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define line l2
def line_l2 (a y x: ℝ) : Prop := a * x + 4 * y + 6 = 0

-- Define the parallel condition
def parallel_lines (a : ℝ) : Prop :=
  (a = 2 ∨ a = -2) ∧ 
  ∀ x y : ℝ, line_l1 a x y ∧ line_l2 a x y → a * x + 4 * x + 6 = 3

-- State the main theorem to prove
theorem necessary_not_sufficient_condition (a : ℝ) : 
  real_number a → (a = 2 ∨ a = -2) ↔ (parallel_lines a) := 
by
  sorry

end necessary_not_sufficient_condition_l239_239170


namespace students_per_bus_correct_l239_239804

def total_students : ℝ := 28
def number_of_buses : ℝ := 2.0
def students_per_bus : ℝ := 14

theorem students_per_bus_correct :
  total_students / number_of_buses = students_per_bus := 
by
  -- Proof should go here
  sorry

end students_per_bus_correct_l239_239804


namespace sum_squares_divisible_by_7_implies_both_divisible_l239_239999

theorem sum_squares_divisible_by_7_implies_both_divisible (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 7 ∣ a ∧ 7 ∣ b :=
sorry

end sum_squares_divisible_by_7_implies_both_divisible_l239_239999


namespace remaining_movies_l239_239550

-- Definitions based on the problem's conditions
def total_movies : ℕ := 8
def watched_movies : ℕ := 4

-- Theorem statement to prove that you still have 4 movies left to watch
theorem remaining_movies : total_movies - watched_movies = 4 :=
by
  sorry

end remaining_movies_l239_239550


namespace jon_coffee_spending_in_april_l239_239527

def cost_per_coffee : ℕ := 2
def coffees_per_day : ℕ := 2
def days_in_april : ℕ := 30

theorem jon_coffee_spending_in_april :
  (coffees_per_day * cost_per_coffee) * days_in_april = 120 :=
by
  sorry

end jon_coffee_spending_in_april_l239_239527


namespace ratio_sub_add_l239_239776

theorem ratio_sub_add (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / (x + y) = 1 / 5 :=
sorry

end ratio_sub_add_l239_239776


namespace balls_in_boxes_l239_239914

-- Definition of the combinatorial function
def combinations (n k : ℕ) : ℕ :=
  n.choose k

-- Problem statement in Lean
theorem balls_in_boxes :
  combinations 7 2 = 21 :=
by
  -- Since the proof is not required here, we place sorry to skip the proof.
  sorry

end balls_in_boxes_l239_239914


namespace calculate_x_l239_239667

theorem calculate_x :
  529 + 2 * 23 * 11 + 121 = 1156 :=
by
  -- Begin the proof (which we won't complete here)
  -- The proof steps would go here
  sorry  -- placeholder for the actual proof steps

end calculate_x_l239_239667


namespace greatest_integer_value_l239_239375

theorem greatest_integer_value (x : ℤ) : 3 * |x - 2| + 9 ≤ 24 → x ≤ 7 :=
by sorry

end greatest_integer_value_l239_239375


namespace searchlight_probability_l239_239108

theorem searchlight_probability (revolutions_per_minute : ℕ) (D : ℝ) (prob : ℝ)
  (h1 : revolutions_per_minute = 4)
  (h2 : prob = 0.6666666666666667) :
  D = (2 / 3) * (60 / revolutions_per_minute) :=
by
  -- To complete the proof, we will use the conditions given.
  sorry

end searchlight_probability_l239_239108


namespace alice_bob_probability_l239_239238

noncomputable def probability_of_exactly_two_sunny_days : ℚ :=
  let p_sunny := 3 / 5
  let p_rain := 2 / 5
  3 * (p_sunny^2 * p_rain)

theorem alice_bob_probability :
  probability_of_exactly_two_sunny_days = 54 / 125 := 
sorry

end alice_bob_probability_l239_239238


namespace gain_percent_of_cost_selling_relation_l239_239109

theorem gain_percent_of_cost_selling_relation (C S : ℕ) (h : 50 * C = 45 * S) : 
  (S > C) ∧ ((S - C) / C * 100 = 100 / 9) :=
by
  sorry

end gain_percent_of_cost_selling_relation_l239_239109


namespace eval_expression_l239_239723

theorem eval_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end eval_expression_l239_239723


namespace find_number_l239_239973

theorem find_number (x : ℝ) (h : (168 / 100) * x / 6 = 354.2) : x = 1265 := 
by
  sorry

end find_number_l239_239973


namespace volume_small_pyramid_eq_27_60_l239_239149

noncomputable def volume_of_smaller_pyramid (base_edge : ℝ) (slant_edge : ℝ) (height_above_base : ℝ) : ℝ :=
  let total_height := Real.sqrt ((slant_edge ^ 2) - ((base_edge / (2 * Real.sqrt 2)) ^ 2))
  let smaller_pyramid_height := total_height - height_above_base
  let scale_factor := (smaller_pyramid_height / total_height)
  let new_base_edge := base_edge * scale_factor
  let new_base_area := (new_base_edge ^ 2) * 2
  (1 / 3) * new_base_area * smaller_pyramid_height

theorem volume_small_pyramid_eq_27_60 :
  volume_of_smaller_pyramid (10 * Real.sqrt 2) 12 4 = 27.6 :=
by
  sorry

end volume_small_pyramid_eq_27_60_l239_239149


namespace andrew_age_l239_239565

theorem andrew_age (a g : ℕ) (h1 : g = 10 * a) (h2 : g - a = 63) : a = 7 := by
  sorry

end andrew_age_l239_239565


namespace teacher_arrangements_l239_239875

theorem teacher_arrangements (T : Fin 30 → ℕ) (h1 : T 1 < T 2 ∧ T 2 < T 3 ∧ T 3 < T 4 ∧ T 4 < T 5)
  (h2 : ∀ i : Fin 4, T (i + 1) ≥ T i + 3)
  (h3 : 1 ≤ T 1)
  (h4 : T 5 ≤ 26) :
  ∃ n : ℕ, n = 26334 := by
  sorry

end teacher_arrangements_l239_239875


namespace correct_transformation_l239_239393

theorem correct_transformation :
  (∀ a b c : ℝ, c ≠ 0 → (a / c = b / c ↔ a = b)) ∧
  (∀ x : ℝ, ¬ (x / 4 + x / 3 = 1 ∧ 3 * x + 4 * x = 1)) ∧
  (∀ a b c : ℝ, ¬ (a * b = b * c ∧ a ≠ c)) ∧
  (∀ x a : ℝ, ¬ (4 * x = a ∧ x = 4 * a)) := sorry

end correct_transformation_l239_239393


namespace solve_equation_1_solve_equation_2_solve_equation_3_l239_239034

theorem solve_equation_1 : ∀ x : ℝ, (4 * (x + 3) = 25) ↔ (x = 13 / 4) :=
by
  sorry

theorem solve_equation_2 : ∀ x : ℝ, (5 * x^2 - 3 * x = x + 1) ↔ (x = -1 / 5 ∨ x = 1) :=
by
  sorry

theorem solve_equation_3 : ∀ x : ℝ, (2 * (x - 2)^2 - (x - 2) = 0) ↔ (x = 2 ∨ x = 5 / 2) :=
by
  sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l239_239034


namespace minimum_value_of_expression_l239_239283

noncomputable def monotonic_function_property
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0): Prop :=
    (1 : ℝ) / a + 8 / b = 25

theorem minimum_value_of_expression 
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0) :
    (1 : ℝ) / a + 8 / b = 25 := 
sorry

end minimum_value_of_expression_l239_239283


namespace reinforcement_1600_l239_239389

/-- A garrison of 2000 men has provisions for 54 days. After 18 days, a reinforcement arrives, and it is now found that the provisions will last only for 20 days more. We define the initial total provisions, remaining provisions after 18 days, and form equations to solve for the unknown reinforcement R.
We need to prove that R = 1600 given these conditions.
-/
theorem reinforcement_1600 (P : ℕ) (M1 M2 : ℕ) (D1 D2 : ℕ) (R : ℕ) :
  M1 = 2000 →
  D1 = 54 →
  D2 = 20 →
  M2 = 2000 + R →
  P = M1 * D1 →
  (M1 * (D1 - 18) = M2 * D2) →
  R = 1600 :=
by
  intros hM1 hD1 hD2 hM2 hP hEquiv
  sorry

end reinforcement_1600_l239_239389


namespace geo_seq_value_l239_239728

variable (a : ℕ → ℝ)
variable (a_2 : a 2 = 2) 
variable (a_4 : a 4 = 8)
variable (geo_prop : a 2 * a 6 = (a 4) ^ 2)

theorem geo_seq_value : a 6 = 32 := 
by 
  sorry

end geo_seq_value_l239_239728


namespace cos_5theta_l239_239159

theorem cos_5theta (theta : ℝ) (h : Real.cos theta = 2 / 5) : Real.cos (5 * theta) = 2762 / 3125 := 
sorry

end cos_5theta_l239_239159


namespace equation_solution_l239_239501

noncomputable def solve_equation (x : ℝ) : Prop :=
  (4 / (x - 1) + 1 / (1 - x) = 1) → x = 4

theorem equation_solution (x : ℝ) (h : 4 / (x - 1) + 1 / (1 - x) = 1) : x = 4 := by
  sorry

end equation_solution_l239_239501


namespace probability_A_and_B_same_county_l239_239366

/-
We have four experts and three counties. We need to assign the experts to the counties such 
that each county has at least one expert. We need to prove that the probability of experts 
A and B being dispatched to the same county is 1/6.
-/

def num_experts : Nat := 4
def num_counties : Nat := 3

def total_possible_events : Nat := 36
def favorable_events : Nat := 6

theorem probability_A_and_B_same_county :
  (favorable_events : ℚ) / total_possible_events = 1 / 6 := by sorry

end probability_A_and_B_same_county_l239_239366


namespace cloaks_always_short_l239_239104

-- Define the problem parameters
variables (Knights Cloaks : Type)
variables [Fintype Knights] [Fintype Cloaks]
variables (h_knights : Fintype.card Knights = 20) (h_cloaks : Fintype.card Cloaks = 20)

-- Assume every knight initially found their cloak too short
variable (too_short : Knights -> Prop)

-- Height order for knights
variable (height_order : LinearOrder Knights)
-- Length order for cloaks
variable (length_order : LinearOrder Cloaks)

-- Sorting function
noncomputable def sorted_cloaks (kn : Knights) : Cloaks := sorry

-- State that after redistribution, every knight's cloak is still too short
theorem cloaks_always_short : 
  ∀ (kn : Knights), too_short kn :=
by sorry

end cloaks_always_short_l239_239104


namespace range_of_a_l239_239620

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a ^ x

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) ↔ (3 / 2 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l239_239620


namespace at_least_one_not_less_than_one_l239_239002

open Real

theorem at_least_one_not_less_than_one (x : ℝ) :
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 :=
by
  -- Definitions of a, b, and c
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  -- Proof is omitted
  sorry

end at_least_one_not_less_than_one_l239_239002


namespace circumradius_relationship_l239_239211

theorem circumradius_relationship 
  (a b c a' b' c' R : ℝ)
  (S S' p p' : ℝ)
  (h₁ : R = (a * b * c) / (4 * S))
  (h₂ : R = (a' * b' * c') / (4 * S'))
  (h₃ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₄ : S' = Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')))
  (h₅ : p = (a + b + c) / 2)
  (h₆ : p' = (a' + b' + c') / 2) :
  (a * b * c) / Real.sqrt (p * (p - a) * (p - b) * (p - c)) = 
  (a' * b' * c') / Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')) :=
by 
  sorry

end circumradius_relationship_l239_239211


namespace inequality_min_m_l239_239661

theorem inequality_min_m (m : ℝ) (x : ℝ) (hx : 1 < x) : 
  x + m * Real.log x + 1 / Real.exp x ≥ Real.exp (m * Real.log x) :=
sorry

end inequality_min_m_l239_239661


namespace solution_in_quadrants_I_and_II_l239_239246

theorem solution_in_quadrants_I_and_II (x y : ℝ) :
  (y > 3 * x) ∧ (y > 6 - 2 * x) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
by
  sorry

end solution_in_quadrants_I_and_II_l239_239246


namespace maximum_fraction_sum_l239_239187

noncomputable def max_fraction_sum (n : ℕ) (a b c d : ℕ) : ℝ :=
  1 - (1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1)))

theorem maximum_fraction_sum (n a b c d : ℕ) (h₀ : n > 1) (h₁ : a + c ≤ n) (h₂ : (a : ℚ) / b + (c : ℚ) / d < 1) :
  ∃ m : ℝ, m = max_fraction_sum n a b c d := by
  sorry

end maximum_fraction_sum_l239_239187


namespace find_second_number_l239_239368

variable (A B : ℕ)

def is_LCM (a b lcm : ℕ) := Nat.lcm a b = lcm
def is_HCF (a b hcf : ℕ) := Nat.gcd a b = hcf

theorem find_second_number (h_lcm : is_LCM 330 B 2310) (h_hcf : is_HCF 330 B 30) : B = 210 := by
  sorry

end find_second_number_l239_239368


namespace christina_speed_l239_239792

theorem christina_speed
  (d v_j v_l t : ℝ)
  (D_l : ℝ)
  (h_d : d = 360)
  (h_v_j : v_j = 5)
  (h_v_l : v_l = 12)
  (h_D_l : D_l = 360)
  (h_t : t = D_l / v_l)
  (h_distance : d = v_j * t + c * t) :
  c = 7 :=
by
  sorry

end christina_speed_l239_239792


namespace Julia_played_with_kids_l239_239889

theorem Julia_played_with_kids :
  (∃ k : ℕ, k = 4) ∧ (∃ n : ℕ, n = 4 + 12) → (n = 16) :=
by
  sorry

end Julia_played_with_kids_l239_239889


namespace first_customer_bought_5_l239_239603

variables 
  (x : ℕ) -- Number of boxes the first customer bought
  (x2 : ℕ) -- Number of boxes the second customer bought
  (x3 : ℕ) -- Number of boxes the third customer bought
  (x4 : ℕ) -- Number of boxes the fourth customer bought
  (x5 : ℕ) -- Number of boxes the fifth customer bought

def goal : ℕ := 150
def remaining_boxes : ℕ := 75
def sold_boxes := x + x2 + x3 + x4 + x5

axiom second_customer (hx2 : x2 = 4 * x) : True
axiom third_customer (hx3 : x3 = (x2 / 2)) : True
axiom fourth_customer (hx4 : x4 = 3 * x3) : True
axiom fifth_customer (hx5 : x5 = 10) : True
axiom sales_goal (hgoal : sold_boxes = goal - remaining_boxes) : True

theorem first_customer_bought_5 (hx2 : x2 = 4 * x) 
                                (hx3 : x3 = (x2 / 2)) 
                                (hx4 : x4 = 3 * x3) 
                                (hx5 : x5 = 10) 
                                (hgoal : sold_boxes = goal - remaining_boxes) : 
                                x = 5 :=
by
  -- Here, we would perform the proof steps
  sorry

end first_customer_bought_5_l239_239603


namespace remainder_when_divided_by_x_minus_2_l239_239383

def f (x : ℝ) : ℝ := x^5 - 4 * x^4 + 6 * x^3 + 25 * x^2 - 20 * x - 24

theorem remainder_when_divided_by_x_minus_2 : f 2 = 52 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l239_239383


namespace compute_a_l239_239517

theorem compute_a (a b : ℚ) 
  (h_root1 : (-1:ℚ) - 5 * (Real.sqrt 3) = -1 - 5 * (Real.sqrt 3))
  (h_rational1 : (-1:ℚ) + 5 * (Real.sqrt 3) = -1 + 5 * (Real.sqrt 3))
  (h_poly : ∀ x, x^3 + a*x^2 + b*x + 48 = 0) :
  a = 50 / 37 :=
by
  sorry

end compute_a_l239_239517


namespace arthur_num_hamburgers_on_first_day_l239_239227

theorem arthur_num_hamburgers_on_first_day (H D : ℕ) (hamburgers_1 hamburgers_2 : ℕ) (hotdogs_1 hotdogs_2 : ℕ)
  (h1 : hamburgers_1 * H + hotdogs_1 * D = 10)
  (h2 : hamburgers_2 * H + hotdogs_2 * D = 7)
  (hprice : D = 1)
  (h1_hotdogs : hotdogs_1 = 4)
  (h2_hotdogs : hotdogs_2 = 3) : 
  hamburgers_1 = 1 := 
by
  sorry

end arthur_num_hamburgers_on_first_day_l239_239227


namespace linear_function_k_range_l239_239826

theorem linear_function_k_range (k b : ℝ) (h1 : k ≠ 0) (h2 : ∃ x : ℝ, (x = 2) ∧ (-3 = k * x + b)) (h3 : 0 < b ∧ b < 1) : -2 < k ∧ k < -3 / 2 :=
by
  sorry

end linear_function_k_range_l239_239826


namespace problem_solution_l239_239253

theorem problem_solution : 
  (∃ (N : ℕ), (1 + 2 + 3) / 6 = (1988 + 1989 + 1990) / N) → ∃ (N : ℕ), N = 5967 :=
by
  intro h
  sorry

end problem_solution_l239_239253


namespace inequality_proof_l239_239879

variable {a b c : ℝ}

theorem inequality_proof (h : a > b) : (a / (c^2 + 1)) > (b / (c^2 + 1)) := by
  sorry

end inequality_proof_l239_239879


namespace find_omega_and_range_l239_239576

noncomputable def f (ω : ℝ) (x : ℝ) := (Real.sin (ω * x))^2 + (Real.sqrt 3) * (Real.sin (ω * x)) * (Real.sin (ω * x + Real.pi / 2))

theorem find_omega_and_range :
  ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x = (Real.sin (2 * ω * x - Real.pi / 6) + 1/2)) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
      f 1 x ∈ Set.Icc ((1 - Real.sqrt 3) / 2) (3 / 2)) :=
by
  sorry

end find_omega_and_range_l239_239576


namespace quadratic_real_roots_range_l239_239764

-- Given conditions and definitions
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def equation_has_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c ≥ 0

-- Problem translated into a Lean statement
theorem quadratic_real_roots_range (m : ℝ) :
  equation_has_real_roots 1 (-2) (-m) ↔ m ≥ -1 :=
by
  sorry

end quadratic_real_roots_range_l239_239764


namespace joe_lifting_problem_l239_239650

theorem joe_lifting_problem (x y : ℝ) (h1 : x + y = 900) (h2 : 2 * x = y + 300) : x = 400 :=
sorry

end joe_lifting_problem_l239_239650


namespace banana_equivalence_l239_239396

theorem banana_equivalence :
  (3 / 4 : ℚ) * 12 = 9 → (1 / 3 : ℚ) * 6 = 2 :=
by
  intro h1
  linarith

end banana_equivalence_l239_239396


namespace eliminate_y_substitution_l239_239439

theorem eliminate_y_substitution (x y : ℝ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) : 3 * x - x + 5 = 8 := 
by
  sorry

end eliminate_y_substitution_l239_239439


namespace Tom_sold_games_for_240_l239_239709

-- Define the value of games and perform operations as per given conditions
def original_value : ℕ := 200
def tripled_value : ℕ := 3 * original_value
def sold_percentage : ℕ := 40
def sold_value : ℕ := (sold_percentage * tripled_value) / 100

-- Assert the proof problem
theorem Tom_sold_games_for_240 : sold_value = 240 := 
by
  sorry

end Tom_sold_games_for_240_l239_239709


namespace product_of_ratios_l239_239328

theorem product_of_ratios:
  ∀ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1^3 - 3 * x1 * y1^2 = 2023) ∧ (y1^3 - 3 * x1^2 * y1 = 2022) →
    (x2^3 - 3 * x2 * y2^2 = 2023) ∧ (y2^3 - 3 * x2^2 * y2 = 2022) →
    (x3^3 - 3 * x3 * y3^2 = 2023) ∧ (y3^3 - 3 * x3^2 * y3 = 2022) →
    (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1 / 2023 :=
by
  intros x1 y1 x2 y2 x3 y3
  sorry

end product_of_ratios_l239_239328


namespace square_pattern_1111111_l239_239511

theorem square_pattern_1111111 :
  11^2 = 121 ∧ 111^2 = 12321 ∧ 1111^2 = 1234321 → 1111111^2 = 1234567654321 :=
by
  sorry

end square_pattern_1111111_l239_239511


namespace room_length_difference_l239_239382

def width := 19
def length := 20
def difference := length - width

theorem room_length_difference : difference = 1 := by
  sorry

end room_length_difference_l239_239382


namespace expression_c_is_positive_l239_239310

def A : ℝ := 2.1
def B : ℝ := -0.5
def C : ℝ := -3.0
def D : ℝ := 4.2
def E : ℝ := 0.8

theorem expression_c_is_positive : |C| + |B| > 0 :=
by {
  sorry
}

end expression_c_is_positive_l239_239310


namespace find_solutions_l239_239724

noncomputable def equation (x : ℝ) : ℝ :=
  (1 / (x^2 + 11*x - 8)) + (1 / (x^2 + 2*x - 8)) + (1 / (x^2 - 13*x - 8))

theorem find_solutions : 
  {x : ℝ | equation x = 0} = {1, -8, 8, -1} := by
  sorry

end find_solutions_l239_239724


namespace log_inequality_l239_239702

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 2 / Real.log 5
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem log_inequality : c > a ∧ a > b := 
by
  sorry

end log_inequality_l239_239702


namespace problem1_problem2_l239_239388

-- Define the first problem
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 2 * x - 4 ↔ (x = 2 ∨ x = 4) := 
by 
  sorry

-- Define the second problem using completing the square method
theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) := 
by 
  sorry

end problem1_problem2_l239_239388


namespace geometric_sequence_eighth_term_is_correct_l239_239321

noncomputable def geometric_sequence_eighth_term : ℚ :=
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8

theorem geometric_sequence_eighth_term_is_correct :
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8 = 35651584 / 4782969 := by
    sorry

end geometric_sequence_eighth_term_is_correct_l239_239321


namespace find_ordered_triple_l239_239526

theorem find_ordered_triple (a b c : ℝ) (h₁ : 2 < a) (h₂ : 2 < b) (h₃ : 2 < c)
    (h_eq : (a + 1)^2 / (b + c - 1) + (b + 2)^2 / (c + a - 3) + (c + 3)^2 / (a + b - 5) = 32) :
    (a = 8 ∧ b = 6 ∧ c = 5) :=
sorry

end find_ordered_triple_l239_239526


namespace inverse_proportion_inequality_l239_239361

variable (x1 x2 k : ℝ)

theorem inverse_proportion_inequality (hA : 2 = k / x1) (hB : 4 = k / x2) (hk : 0 < k) : 
  x1 > x2 ∧ x1 > 0 ∧ x2 > 0 :=
sorry

end inverse_proportion_inequality_l239_239361


namespace count_positive_solutions_of_eq_l239_239359

theorem count_positive_solutions_of_eq : 
  (∃ x : ℝ, x^2 = -6 * x + 9 ∧ x > 0) ∧ (¬ ∃ y : ℝ, y^2 = -6 * y + 9 ∧ y > 0 ∧ y ≠ -3 + 3 * Real.sqrt 2) :=
sorry

end count_positive_solutions_of_eq_l239_239359


namespace triangle_perpendicular_bisector_properties_l239_239928

variables {A B C A1 A2 B1 B2 C1 C2 : Type} (triangle : triangle A B C)
  (A1_perpendicular : dropping_perpendicular_to_bisector A )
  (A2_perpendicular : dropping_perpendicular_to_bisector A )
  (B1_perpendicular : dropping_perpendicular_to_bisector B )
  (B2_perpendicular : dropping_perpendicular_to_bisector B )
  (C1_perpendicular : dropping_perpendicular_to_bisector C )
  (C2_perpendicular : dropping_perpendicular_to_bisector C )
  
-- Defining required structures
structure triangle (A B C : Type) :=
  (AB BC CA : ℝ)

structure dropping_perpendicular_to_bisector (v : Type) :=
  (perpendicular_to_bisector : ℝ)

namespace triangle_properties

theorem triangle_perpendicular_bisector_properties :
  2 * (A1_perpendicular.perpendicular_to_bisector + A2_perpendicular.perpendicular_to_bisector + 
       B1_perpendicular.perpendicular_to_bisector + B2_perpendicular.perpendicular_to_bisector + 
       C1_perpendicular.perpendicular_to_bisector + C2_perpendicular.perpendicular_to_bisector) = 
  (triangle.AB + triangle.BC + triangle.CA) :=
sorry

end triangle_properties

end triangle_perpendicular_bisector_properties_l239_239928


namespace perfect_square_expression_l239_239585

theorem perfect_square_expression (p : ℝ) : 
  (12.86^2 + 12.86 * p + 0.14^2) = (12.86 + 0.14)^2 → p = 0.28 :=
by
  sorry

end perfect_square_expression_l239_239585


namespace schoolchildren_initial_speed_l239_239331

theorem schoolchildren_initial_speed (v : ℝ) (t t_1 t_2 : ℝ) 
  (h1 : t_1 = (6 * v) / (v + 60) + (400 - 3 * v) / (v + 60)) 
  (h2 : t_2 = (400 - 3 * v) / v) 
  (h3 : t_1 = t_2) : v = 63.24 :=
by sorry

end schoolchildren_initial_speed_l239_239331


namespace binary_addition_to_decimal_l239_239843

theorem binary_addition_to_decimal : (0b111111111 + 0b1000001 = 576) :=
by {
  sorry
}

end binary_addition_to_decimal_l239_239843


namespace range_of_values_for_a_l239_239235

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * Real.sin x - (1 / 2) * Real.cos (2 * x) + a - (3 / a) + (1 / 2)

theorem range_of_values_for_a (a : ℝ) (ha : a ≠ 0) : 
  (∀ x : ℝ, f x a ≤ 0) ↔ (0 < a ∧ a ≤ 1) :=
by 
  let g (t : ℝ) : ℝ := t^2 + a * t + a - (3 / a)
  have h1 : g (-1) ≤ 0 := by sorry
  have h2 : g (1) ≤ 0 := by sorry
  sorry

end range_of_values_for_a_l239_239235


namespace six_digit_pair_divisibility_l239_239195

theorem six_digit_pair_divisibility (a b : ℕ) (ha : 100000 ≤ a ∧ a < 1000000) (hb : 100000 ≤ b ∧ b < 1000000) :
  ((1000000 * a + b) % (a * b) = 0) ↔ (a = 166667 ∧ b = 333334) ∨ (a = 500001 ∧ b = 500001) :=
by sorry

end six_digit_pair_divisibility_l239_239195


namespace unique_solution_of_functional_eqn_l239_239487

theorem unique_solution_of_functional_eqn (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1) → (∀ x : ℝ, f x = x) :=
by
  intros h
  sorry

end unique_solution_of_functional_eqn_l239_239487


namespace car_fuel_efficiency_l239_239303

theorem car_fuel_efficiency (distance gallons fuel_efficiency D : ℝ)
  (h₀ : fuel_efficiency = 40)
  (h₁ : gallons = 3.75)
  (h₂ : distance = 150)
  (h_eff : fuel_efficiency = distance / gallons) :
  fuel_efficiency = 40 ∧ (D / fuel_efficiency) = (D / 40) :=
by
  sorry

end car_fuel_efficiency_l239_239303


namespace find_x1_l239_239062

theorem find_x1 (x1 x2 x3 x4 : ℝ) (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : x1 = 4 / 5 := 
  sorry

end find_x1_l239_239062


namespace work_days_of_b_l239_239807

theorem work_days_of_b (d : ℕ) 
  (A B C : ℕ)
  (h_ratioA : A = (3 * 115) / 5)
  (h_ratioB : B = (4 * 115) / 5)
  (h_C : C = 115)
  (h_total_wages : 1702 = (A * 6) + (B * d) + (C * 4)) :
  d = 9 := 
sorry

end work_days_of_b_l239_239807


namespace problem1_problem2_l239_239497

theorem problem1 : ∃ (m : ℝ) (b : ℝ), ∀ (x y : ℝ),
  3 * x + 4 * y - 2 = 0 ∧ x - y + 4 = 0 →
  y = m * x + b ∧ (1 / m = -2) ∧ (y = - (2 * x + 2)) :=
sorry

theorem problem2 : ∀ (x y a : ℝ), (x = -1) ∧ (y = 3) → 
  (x + y = a) →
  a = 2 ∧ (x + y - 2 = 0) :=
sorry

end problem1_problem2_l239_239497


namespace find_V_D_l239_239360

noncomputable def V_A : ℚ := sorry
noncomputable def V_B : ℚ := sorry
noncomputable def V_C : ℚ := sorry
noncomputable def V_D : ℚ := sorry
noncomputable def V_E : ℚ := sorry

axiom condition1 : V_A + V_B + V_C + V_D + V_E = 1 / 7.5
axiom condition2 : V_A + V_C + V_E = 1 / 5
axiom condition3 : V_A + V_C + V_D = 1 / 6
axiom condition4 : V_B + V_D + V_E = 1 / 4

theorem find_V_D : V_D = 1 / 12 := 
  by
    sorry

end find_V_D_l239_239360


namespace multiple_of_A_share_l239_239239

theorem multiple_of_A_share (a b c : ℤ) (hC : c = 84) (hSum : a + b + c = 427)
  (hEquality1 : ∃ x : ℤ, x * a = 4 * b) (hEquality2 : 7 * c = 4 * b) : ∃ x : ℤ, x = 3 :=
by {
  sorry
}

end multiple_of_A_share_l239_239239


namespace probability_bob_wins_l239_239740

theorem probability_bob_wins (P_lose : ℝ) (P_tie : ℝ) (h1 : P_lose = 5/8) (h2 : P_tie = 1/8) :
  (1 - P_lose - P_tie) = 1/4 :=
by
  sorry

end probability_bob_wins_l239_239740


namespace third_person_fraction_removed_l239_239177

-- Define the number of teeth for each person and the fractions that are removed
def total_teeth := 32
def total_removed := 40

def first_person_removed := (1 / 4) * total_teeth
def second_person_removed := (3 / 8) * total_teeth
def fourth_person_removed := 4

-- Define the total teeth removed by the first, second, and fourth persons
def known_removed := first_person_removed + second_person_removed + fourth_person_removed

-- Define the total teeth removed by the third person
def third_person_removed := total_removed - known_removed

-- Prove that the third person had 1/2 of his teeth removed
theorem third_person_fraction_removed :
  third_person_removed / total_teeth = 1 / 2 :=
by
  sorry

end third_person_fraction_removed_l239_239177


namespace factorize_expression_l239_239134

theorem factorize_expression (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) :=
by sorry

end factorize_expression_l239_239134


namespace surjective_injective_eq_l239_239506

theorem surjective_injective_eq (f g : ℕ → ℕ) 
  (hf : Function.Surjective f) 
  (hg : Function.Injective g) 
  (h : ∀ n : ℕ, f n ≥ g n) : 
  ∀ n : ℕ, f n = g n := 
by
  sorry

end surjective_injective_eq_l239_239506


namespace typist_original_salary_l239_239946

theorem typist_original_salary (S : ℝ) :
  (1.10 * S * 0.95 * 1.07 * 0.97 = 2090) → (S = 2090 / (1.10 * 0.95 * 1.07 * 0.97)) :=
by
  intro h
  sorry

end typist_original_salary_l239_239946


namespace fraction_of_usual_speed_l239_239434

-- Definitions based on conditions
variable (S R : ℝ)
variable (h1 : S * 60 = R * 72)

-- Goal statement
theorem fraction_of_usual_speed (h1 : S * 60 = R * 72) : R / S = 5 / 6 :=
by
  sorry

end fraction_of_usual_speed_l239_239434


namespace lemons_required_for_new_recipe_l239_239752

noncomputable def lemons_needed_to_make_gallons (lemons_original : ℕ) (gallons_original : ℕ) (additional_lemons : ℕ) (additional_gallons : ℕ) (gallons_new : ℕ) : ℝ :=
  let lemons_per_gallon := (lemons_original : ℝ) / (gallons_original : ℝ)
  let additional_lemons_per_gallon := (additional_lemons : ℝ) / (additional_gallons : ℝ)
  let total_lemons_per_gallon := lemons_per_gallon + additional_lemons_per_gallon
  total_lemons_per_gallon * (gallons_new : ℝ)

theorem lemons_required_for_new_recipe : lemons_needed_to_make_gallons 36 48 2 6 18 = 19.5 :=
by
  sorry

end lemons_required_for_new_recipe_l239_239752


namespace find_f1_l239_239859

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f1
  (h1 : ∀ x : ℝ, |f x - x^2| ≤ 1/4)
  (h2 : ∀ x : ℝ, |f x + 1 - x^2| ≤ 3/4) :
  f 1 = 3/4 := 
sorry

end find_f1_l239_239859


namespace problem_m_n_l239_239600

theorem problem_m_n (m n : ℝ) (h1 : m * n = 1) (h2 : m^2 + n^2 = 3) (h3 : m^3 + n^3 = 44 + n^4) (h4 : m^5 + 5 = 11) : m^9 + n = -29 :=
sorry

end problem_m_n_l239_239600


namespace exists_small_area_triangle_l239_239270

structure Point :=
(x : ℝ)
(y : ℝ)

def is_valid_point (p : Point) : Prop :=
(|p.x| ≤ 2) ∧ (|p.y| ≤ 2)

def no_three_collinear (points : List Point) : Prop :=
  ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points →
  (p1 ≠ p2) → (p1 ≠ p3) → (p2 ≠ p3) →
  ((p1.y - p2.y) * (p1.x - p3.x) ≠ (p1.y - p3.y) * (p1.x - p2.x))

noncomputable def triangle_area (p1 p2 p3: Point) : ℝ :=
(abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))) / 2

theorem exists_small_area_triangle (points : List Point)
  (h_valid : ∀ p ∈ points, is_valid_point p)
  (h_no_collinear : no_three_collinear points)
  (h_len : points.length = 6) :
  ∃ (p1 p2 p3: Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
  triangle_area p1 p2 p3 ≤ 2 :=
sorry

end exists_small_area_triangle_l239_239270


namespace pipe_A_filling_time_l239_239633

theorem pipe_A_filling_time :
  ∃ (t : ℚ), 
  (∀ (t : ℚ), (t > 0) → (1 / t + 5 / t = 1 / 4.571428571428571) ↔ t = 27.42857142857143) := 
by
  -- definition of t and the corresponding conditions are directly derived from the problem
  sorry

end pipe_A_filling_time_l239_239633


namespace maximize_S_n_l239_239524

variable {a : ℕ → ℝ} -- Sequence term definition
variable {S : ℕ → ℝ} -- Sum of first n terms

-- Definitions based on conditions
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  n * a 1 + (n * (n - 1) / 2) * ((a 2) - (a 1))

axiom a1_positive (a1 : ℝ) : 0 < a1 -- given a1 > 0
axiom S3_eq_S16 (a1 d : ℝ) : sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16

-- Problem Statement
theorem maximize_S_n (a : ℕ → ℝ) (d : ℝ) : is_arithmetic_sequence a d →
  a 1 > 0 →
  sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16 →
  (∀ n, sum_of_first_n_terms a n = sum_of_first_n_terms a 9 ∨ sum_of_first_n_terms a n = sum_of_first_n_terms a 10) :=
by
  sorry

end maximize_S_n_l239_239524


namespace females_advanced_degrees_under_40_l239_239372

-- Definitions derived from conditions
def total_employees : ℕ := 280
def female_employees : ℕ := 160
def male_employees : ℕ := 120
def advanced_degree_holders : ℕ := 120
def college_degree_holders : ℕ := 100
def high_school_diploma_holders : ℕ := 60
def male_advanced_degree_holders : ℕ := 50
def male_college_degree_holders : ℕ := 35
def male_high_school_diploma_holders : ℕ := 35
def percentage_females_under_40 : ℝ := 0.75

-- The mathematically equivalent proof problem
theorem females_advanced_degrees_under_40 : 
  (advanced_degree_holders - male_advanced_degree_holders) * percentage_females_under_40 = 52 :=
by
  sorry -- Proof to be provided

end females_advanced_degrees_under_40_l239_239372


namespace smallest_possible_sum_l239_239938

theorem smallest_possible_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Nat.gcd (a + b) 330 = 1) (h4 : b ^ b ∣ a ^ a) (h5 : ¬ b ∣ a) :
  a + b = 147 :=
sorry

end smallest_possible_sum_l239_239938


namespace nitin_borrowed_amount_l239_239248

theorem nitin_borrowed_amount (P : ℝ) (I1 I2 I3 : ℝ) :
  (I1 = P * 0.06 * 3) ∧
  (I2 = P * 0.09 * 5) ∧
  (I3 = P * 0.13 * 3) ∧
  (I1 + I2 + I3 = 8160) →
  P = 8000 :=
by
  sorry

end nitin_borrowed_amount_l239_239248


namespace smallest_perfect_square_divisible_by_4_and_5_l239_239033

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ (n = 400) := 
by
  sorry

end smallest_perfect_square_divisible_by_4_and_5_l239_239033


namespace common_root_value_l239_239592

theorem common_root_value (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + 8 = 0 ∧ x^2 + x + a = 0) ↔ a = -6 :=
sorry

end common_root_value_l239_239592


namespace minimum_choir_members_l239_239191

theorem minimum_choir_members:
  ∃ n : ℕ, (n % 9 = 0) ∧ (n % 10 = 0) ∧ (n % 11 = 0) ∧ (∀ m : ℕ, (m % 9 = 0) ∧ (m % 10 = 0) ∧ (m % 11 = 0) → n ≤ m) → n = 990 :=
by
  sorry

end minimum_choir_members_l239_239191


namespace increasing_condition_l239_239820

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 2 * x - 2 * a

-- Prove that f is increasing on the interval [2, +∞) if and only if a ≤ 2
theorem increasing_condition (a : ℝ) : (∀ x ≥ 2, f' x a ≥ 0) ↔ (a ≤ 2) := 
sorry

end increasing_condition_l239_239820


namespace graph_of_function_does_not_pass_through_first_quadrant_l239_239406

theorem graph_of_function_does_not_pass_through_first_quadrant (k : ℝ) (h : k < 0) : 
  ¬(∃ x y : ℝ, y = k * (x - k) ∧ x > 0 ∧ y > 0) :=
sorry

end graph_of_function_does_not_pass_through_first_quadrant_l239_239406


namespace rachel_homework_total_l239_239067

-- Definitions based on conditions
def math_homework : Nat := 8
def biology_homework : Nat := 3

-- Theorem based on the problem statement
theorem rachel_homework_total : math_homework + biology_homework = 11 := by
  -- typically, here you would provide a proof, but we use sorry to skip it
  sorry

end rachel_homework_total_l239_239067


namespace price_increase_percentage_l239_239834

theorem price_increase_percentage (x : ℝ) :
  (0.9 * (1 + x / 100) * 0.9259259259259259 = 1) → x = 20 :=
by
  intros
  sorry

end price_increase_percentage_l239_239834


namespace part1_part2_l239_239853

noncomputable def f (x : ℝ) : ℝ := (Real.exp (-x) - Real.exp x) / 2

theorem part1 (h_odd : ∀ x, f (-x) = -f x) (g : ℝ → ℝ) (h_even : ∀ x, g (-x) = g x)
  (h_g_def : ∀ x, g x = f x + Real.exp x) :
  ∀ x, f x = (Real.exp (-x) - Real.exp x) / 2 := sorry

theorem part2 : {x : ℝ | f x ≥ 3 / 4} = {x | x ≤ -Real.log 2} := sorry

end part1_part2_l239_239853


namespace completing_square_result_l239_239214

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end completing_square_result_l239_239214


namespace total_people_can_ride_l239_239217

theorem total_people_can_ride (num_people_per_teacup : Nat) (num_teacups : Nat) (h1 : num_people_per_teacup = 9) (h2 : num_teacups = 7) : num_people_per_teacup * num_teacups = 63 := by
  sorry

end total_people_can_ride_l239_239217


namespace main_factor_is_D_l239_239679

-- Let A, B, C, and D be the factors where A is influenced by 1, B by 2, C by 3, and D by 4
def A := 1
def B := 2
def C := 3
def D := 4

-- Defining the main factor influenced by the plan
def main_factor_influenced_by_plan := D

-- The problem statement translated to a Lean theorem statement
theorem main_factor_is_D : main_factor_influenced_by_plan = D := 
by sorry

end main_factor_is_D_l239_239679


namespace parallel_vectors_m_l239_239488

theorem parallel_vectors_m (m : ℝ) :
  let a := (1, 2)
  let b := (m, m + 1)
  a.1 * b.2 = a.2 * b.1 → m = 1 :=
by
  intros a b h
  dsimp at *
  sorry

end parallel_vectors_m_l239_239488


namespace donuts_left_l239_239734

def initial_donuts : ℕ := 50
def after_bill_eats (initial : ℕ) : ℕ := initial - 2
def after_secretary_takes (remaining_after_bill : ℕ) : ℕ := remaining_after_bill - 4
def coworkers_take (remaining_after_secretary : ℕ) : ℕ := remaining_after_secretary / 2
def final_donuts (initial : ℕ) : ℕ :=
  let remaining_after_bill := after_bill_eats initial
  let remaining_after_secretary := after_secretary_takes remaining_after_bill
  remaining_after_secretary - coworkers_take remaining_after_secretary

theorem donuts_left : final_donuts 50 = 22 := by
  sorry

end donuts_left_l239_239734


namespace min_distance_PS_l239_239783

-- Definitions of the distances given in the problem
def PQ : ℝ := 12
def QR : ℝ := 7
def RS : ℝ := 5

-- Hypotheses for the problem
axiom h1 : PQ = 12
axiom h2 : QR = 7
axiom h3 : RS = 5

-- The goal is to prove that the minimum distance between P and S is 0.
theorem min_distance_PS : ∃ PS : ℝ, PS = 0 :=
by
  -- The proof is omitted
  sorry

end min_distance_PS_l239_239783


namespace p_or_q_is_false_implies_p_and_q_is_false_l239_239250

theorem p_or_q_is_false_implies_p_and_q_is_false (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ((¬ (p ∧ q) → (p ∨ q ∨ ¬ (p ∨ q)))) := sorry

end p_or_q_is_false_implies_p_and_q_is_false_l239_239250


namespace parallelogram_larger_angle_l239_239314

theorem parallelogram_larger_angle (a b : ℕ) (h₁ : b = a + 50) (h₂ : a = 65) : b = 115 := 
by
  -- Use the conditions h₁ and h₂ to prove the statement.
  sorry

end parallelogram_larger_angle_l239_239314


namespace quadratic_has_integer_solutions_l239_239444

theorem quadratic_has_integer_solutions : 
  ∃ (s : Finset ℕ), ∀ a : ℕ, a ∈ s ↔ (1 ≤ a ∧ a ≤ 50 ∧ ((∃ n : ℕ, 4 * a + 1 = n^2))) ∧ s.card = 6 := 
  sorry

end quadratic_has_integer_solutions_l239_239444


namespace exists_unique_t_exists_m_pos_l239_239890

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem exists_unique_t (m : ℝ) (h : m = 1) : 
  ∃! (t : ℝ), t ∈ Set.Ioc (1 / 2) 1 ∧ deriv (f 1) t = 0 := sorry

theorem exists_m_pos : ∃ (m : ℝ), 0 < m ∧ m < 1 ∧ ∀ (x : ℝ), 0 < x → f m x > 0 := sorry

end exists_unique_t_exists_m_pos_l239_239890


namespace triangle_split_points_l239_239209

noncomputable def smallest_n_for_split (AB BC CA : ℕ) : ℕ := 
  if AB = 13 ∧ BC = 14 ∧ CA = 15 then 27 else sorry

theorem triangle_split_points (AB BC CA : ℕ) (h : AB = 13 ∧ BC = 14 ∧ CA = 15) :
  smallest_n_for_split AB BC CA = 27 :=
by
  cases h with | intro h1 h23 => sorry

-- Assertions for the explicit values provided in the conditions
example : smallest_n_for_split 13 14 15 = 27 :=
  triangle_split_points 13 14 15 ⟨rfl, rfl, rfl⟩

end triangle_split_points_l239_239209


namespace positive_integers_satisfying_condition_l239_239083

theorem positive_integers_satisfying_condition :
  ∃! n : ℕ, 0 < n ∧ 24 - 6 * n > 12 :=
by
  sorry

end positive_integers_satisfying_condition_l239_239083


namespace minimum_ribbon_length_l239_239683

def side_length : ℚ := 13 / 12

def perimeter_of_equilateral_triangle (a : ℚ) : ℚ := 3 * a

theorem minimum_ribbon_length :
  perimeter_of_equilateral_triangle side_length = 3.25 := 
by
  sorry

end minimum_ribbon_length_l239_239683


namespace boat_speed_in_still_water_l239_239318

theorem boat_speed_in_still_water (V_s : ℝ) (D : ℝ) (t_down : ℝ) (t_up : ℝ) (V_b : ℝ) :
  V_s = 3 → t_down = 1 → t_up = 3 / 2 →
  (V_b + V_s) * t_down = D → (V_b - V_s) * t_up = D → V_b = 15 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end boat_speed_in_still_water_l239_239318


namespace Lisa_quiz_goal_l239_239722

theorem Lisa_quiz_goal (total_quizzes : ℕ) (required_percentage : ℝ) (a_scored : ℕ) (completed_quizzes : ℕ) : 
  total_quizzes = 60 → 
  required_percentage = 0.75 → 
  a_scored = 30 → 
  completed_quizzes = 40 → 
  ∃ lower_than_a_quizzes : ℕ, lower_than_a_quizzes = 5 :=
by
  intros total_quizzes_eq req_percent_eq a_scored_eq completed_quizzes_eq
  sorry

end Lisa_quiz_goal_l239_239722


namespace find_ages_l239_239267

theorem find_ages (P F M : ℕ) 
  (h1 : F - P = 31)
  (h2 : (F + 8) + (P + 8) = 69)
  (h3 : F - M = 4)
  (h4 : (P + 5) + (M + 5) = 65) :
  P = 11 ∧ F = 42 ∧ M = 38 :=
by
  sorry

end find_ages_l239_239267


namespace range_of_a_l239_239343

variable (f : ℝ → ℝ) (a : ℝ)

-- Definitions based on provided conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x ≤ f y

-- Main statement
theorem range_of_a
    (hf_even : is_even f)
    (hf_mono : is_monotonically_increasing f)
    (h_ineq : ∀ x : ℝ, f (Real.log (a) / Real.log 2) ≤ f (x^2 - 2 * x + 2)) :
  (1/2 : ℝ) ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l239_239343


namespace problem_1_problem_2_l239_239617

noncomputable def f (x : ℝ) : ℝ := (1 / (9 * (Real.sin x)^2)) + (4 / (9 * (Real.cos x)^2))

theorem problem_1 (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) : f x ≥ 1 := 
sorry

theorem problem_2 (x : ℝ) : x^2 + |x-2| + 1 ≥ 3 ↔ (x ≤ 0 ∨ x ≥ 1) :=
sorry

end problem_1_problem_2_l239_239617


namespace regular_polygon_sides_l239_239637

theorem regular_polygon_sides (n : ℕ) (h : 2 < n)
  (interior_angle : ∀ n, (n - 2) * 180 / n = 144) : n = 10 :=
sorry

end regular_polygon_sides_l239_239637


namespace girls_in_class_l239_239230

theorem girls_in_class (g b : ℕ) (h1 : g + b = 28) (h2 : g * 4 = b * 3) : g = 12 := by
  sorry

end girls_in_class_l239_239230


namespace perpendicular_lines_l239_239168

theorem perpendicular_lines (a : ℝ) :
  (∃ (m₁ m₂ : ℝ), ((a + 1) * m₁ + a * m₂ = 0) ∧ 
                  (a * m₁ + 2 * m₂ = 1) ∧ 
                  m₁ * m₂ = -1) ↔ (a = 0 ∨ a = -3) := 
sorry

end perpendicular_lines_l239_239168


namespace product_divisible_by_3_or_5_l239_239106

theorem product_divisible_by_3_or_5 {a b c d : ℕ} (h : Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = a + b + c + d) :
  (a * b * c * d) % 3 = 0 ∨ (a * b * c * d) % 5 = 0 :=
by
  sorry

end product_divisible_by_3_or_5_l239_239106


namespace trout_ratio_l239_239719

theorem trout_ratio (caleb_trouts dad_trouts : ℕ) (h_c : caleb_trouts = 2) (h_d : dad_trouts = caleb_trouts + 4) :
  dad_trouts / (Nat.gcd dad_trouts caleb_trouts) = 3 ∧ caleb_trouts / (Nat.gcd dad_trouts caleb_trouts) = 1 :=
by
  sorry

end trout_ratio_l239_239719


namespace fraction_of_positive_number_l239_239443

theorem fraction_of_positive_number (x : ℝ) (f : ℝ) (h : x = 0.4166666666666667 ∧ f * x = (25/216) * (1/x)) : f = 2/3 :=
sorry

end fraction_of_positive_number_l239_239443


namespace total_cost_l239_239798

/-- There are two types of discs, one costing 10.50 and another costing 8.50.
You bought a total of 10 discs, out of which 6 are priced at 8.50.
The task is to determine the total amount spent. -/
theorem total_cost (price1 price2 : ℝ) (num1 num2 : ℕ) 
  (h1 : price1 = 10.50) (h2 : price2 = 8.50) 
  (h3 : num1 = 6) (h4 : num2 = 10) 
  (h5 : num2 - num1 = 4) : 
  (num1 * price2 + (num2 - num1) * price1) = 93.00 := 
by
  sorry

end total_cost_l239_239798


namespace log_ratio_l239_239559

theorem log_ratio : (Real.logb 2 16) / (Real.logb 2 4) = 2 := sorry

end log_ratio_l239_239559


namespace avg_eq_pos_diff_l239_239648

theorem avg_eq_pos_diff (y : ℝ) (h : (35 + y) / 2 = 42) : |35 - y| = 14 := 
sorry

end avg_eq_pos_diff_l239_239648


namespace Pat_height_l239_239749

noncomputable def Pat_first_day_depth := 40 -- in cm
noncomputable def Mat_second_day_depth := 3 * Pat_first_day_depth -- Mat digs 3 times the depth on the second day
noncomputable def Pat_third_day_depth := Mat_second_day_depth - Pat_first_day_depth -- Pat digs the same amount on the third day
noncomputable def Total_depth_after_third_day := Mat_second_day_depth + Pat_third_day_depth -- Total depth after third day's digging
noncomputable def Depth_above_Pat_head := 50 -- The depth above Pat's head

theorem Pat_height : Total_depth_after_third_day - Depth_above_Pat_head = 150 := by
  sorry

end Pat_height_l239_239749


namespace numberOfBoys_playground_boys_count_l239_239968

-- Definitions and conditions
def numberOfGirls : ℕ := 28
def totalNumberOfChildren : ℕ := 63

-- Theorem statement
theorem numberOfBoys (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) : ℕ :=
  totalNumberOfChildren - numberOfGirls

-- Proof statement
theorem playground_boys_count (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) (boysOnPlayground : ℕ) : 
  numberOfGirls = 28 → 
  totalNumberOfChildren = 63 → 
  boysOnPlayground = totalNumberOfChildren - numberOfGirls →
  boysOnPlayground = 35 :=
by
  intros
  -- since no proof is required, we use sorry here
  exact sorry

end numberOfBoys_playground_boys_count_l239_239968


namespace polar_to_cartesian_max_and_min_x_plus_y_l239_239575

-- Define the given polar equation and convert it to Cartesian equations
def polar_equation (rho θ : ℝ) : Prop :=
  rho^2 - 4 * (Real.sqrt 2) * rho * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Cartesian equation derived from the polar equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0

-- Prove equivalence of the given polar equation and its equivalent Cartesian form for all ρ and \theta
theorem polar_to_cartesian (rho θ : ℝ) : 
  (∃ (x y : ℝ), polar_equation rho θ ∧ x = rho * Real.cos θ ∧ y = rho * Real.sin θ ∧ cartesian_equation x y) :=
by
  sorry

-- Property of points (x, y) on the circle defined by the Cartesian equation
def lies_on_circle (x y : ℝ) : Prop :=
  cartesian_equation x y

-- Given a point (x, y) on the circle defined by cartesian_equation, show bounds for x + y
theorem max_and_min_x_plus_y (x y : ℝ) (h : lies_on_circle x y) : 
  2 ≤ x + y ∧ x + y ≤ 6 :=
by
  sorry

end polar_to_cartesian_max_and_min_x_plus_y_l239_239575


namespace intersection_A_B_l239_239242

-- Define the sets A and B
def set_A : Set ℝ := { x | x^2 ≤ 1 }
def set_B : Set ℝ := { -2, -1, 0, 1, 2 }

-- The goal is to prove that the intersection of A and B is {-1, 0, 1}
theorem intersection_A_B : set_A ∩ set_B = ({-1, 0, 1} : Set ℝ) :=
by
  sorry

end intersection_A_B_l239_239242


namespace seedlings_total_l239_239199

theorem seedlings_total (seeds_per_packet : ℕ) (packets : ℕ) (total_seedlings : ℕ) 
  (h1 : seeds_per_packet = 7) (h2 : packets = 60) : total_seedlings = 420 :=
by {
  sorry
}

end seedlings_total_l239_239199


namespace angle_ABC_40_degrees_l239_239027

theorem angle_ABC_40_degrees (ABC ABD CBD : ℝ) 
    (h1 : CBD = 90) 
    (h2 : ABD = 60)
    (h3 : ABC + ABD + CBD = 190) : 
    ABC = 40 := 
by {
  sorry
}

end angle_ABC_40_degrees_l239_239027


namespace vertical_asymptote_at_5_l239_239274

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + 2*x + 10) / (x - 5)

theorem vertical_asymptote_at_5 : ∃ a : ℝ, (a = 5) ∧ ∀ δ > 0, ∃ ε > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < ε → |f x| > δ :=
by
  sorry

end vertical_asymptote_at_5_l239_239274


namespace parabola_vertex_coordinates_l239_239533

theorem parabola_vertex_coordinates {a b c : ℝ} (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 3)
  (h_root : a * 2^2 + b * 2 + c = 3) (h_symm : ∀ x : ℝ, a * (2 - x)^2 + b * (2 - x) + c = a * x^2 + b * x + c) :
  (2, 3) = (2, 3) :=
by
  sorry

end parabola_vertex_coordinates_l239_239533


namespace monthly_earnings_l239_239847

variable (e : ℕ) (s : ℕ) (p : ℕ) (t : ℕ)

-- conditions
def half_monthly_savings := s = e / 2
def car_price := p = 16000
def saving_months := t = 8
def total_saving := s * t = p

theorem monthly_earnings : ∀ (e s p t : ℕ), 
  half_monthly_savings e s → 
  car_price p → 
  saving_months t → 
  total_saving s t p → 
  e = 4000 :=
by
  intros e s p t h1 h2 h3 h4
  sorry

end monthly_earnings_l239_239847


namespace exponent_problem_l239_239632

theorem exponent_problem 
  (a : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : a > 0) 
  (h2 : a^x = 3) 
  (h3 : a^y = 5) : 
  a^(2*x + y/2) = 9 * Real.sqrt 5 :=
by
  sorry

end exponent_problem_l239_239632


namespace fraction_to_decimal_l239_239453

theorem fraction_to_decimal (numer: ℚ) (denom: ℕ) (h_denom: denom = 2^5 * 5^1) :
  numer.den = 160 → numer.num = 59 → numer == 0.36875 :=
by
  intros
  sorry  

end fraction_to_decimal_l239_239453


namespace points_among_transformations_within_square_l239_239512

def projection_side1 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, 2 - A.2)
def projection_side2 (A : ℝ × ℝ) : ℝ × ℝ := (-A.1, A.2)
def projection_side3 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, -A.2)
def projection_side4 (A : ℝ × ℝ) : ℝ × ℝ := (2 - A.1, A.2)

def within_square (A : ℝ × ℝ) : Prop := 
  0 ≤ A.1 ∧ A.1 ≤ 1 ∧ 0 ≤ A.2 ∧ A.2 ≤ 1

theorem points_among_transformations_within_square (A : ℝ × ℝ)
  (H1 : within_square A)
  (H2 : within_square (projection_side1 A))
  (H3 : within_square (projection_side2 (projection_side1 A)))
  (H4 : within_square (projection_side3 (projection_side2 (projection_side1 A))))
  (H5 : within_square (projection_side4 (projection_side3 (projection_side2 (projection_side1 A))))) :
  A = (1 / 3, 1 / 3) := sorry

end points_among_transformations_within_square_l239_239512


namespace sum_of_numbers_l239_239560

theorem sum_of_numbers :
  15.58 + 21.32 + 642.51 + 51.51 = 730.92 := 
  by
  sorry

end sum_of_numbers_l239_239560


namespace find_n_l239_239445

theorem find_n (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % 11 = 0) : n = 1 :=
by
  sorry

end find_n_l239_239445


namespace contrapositive_equivalence_l239_239666

variable (p q : Prop)

theorem contrapositive_equivalence : (p → ¬q) ↔ (q → ¬p) := by
  sorry

end contrapositive_equivalence_l239_239666


namespace arithmetic_sequence_S7_eq_28_l239_239188

/--
Given the arithmetic sequence \( \{a_n\} \) and the sum of its first \( n \) terms is \( S_n \),
if \( a_3 + a_4 + a_5 = 12 \), then prove \( S_7 = 28 \).
-/
theorem arithmetic_sequence_S7_eq_28
  (a : ℕ → ℤ) -- Sequence a_n
  (S : ℕ → ℤ) -- Sum sequence S_n
  (h1 : a 3 + a 4 + a 5 = 12)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) -- Sum formula
  : S 7 = 28 :=
sorry

end arithmetic_sequence_S7_eq_28_l239_239188


namespace correct_statements_l239_239644

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem correct_statements :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (f (Real.log 3 / Real.log 2) ≠ 2) ∧
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (|x|) ≥ 0 ∧ f 0 = 0) :=
by
  sorry

end correct_statements_l239_239644


namespace first_player_wins_l239_239044

noncomputable def game_win_guarantee : Prop :=
  ∃ (first_can_guarantee_win : Bool),
    first_can_guarantee_win = true

theorem first_player_wins :
  ∀ (nuts : ℕ) (players : (ℕ × ℕ)) (move : ℕ → ℕ) (end_condition : ℕ → Prop),
    nuts = 10 →
    players = (1, 2) →
    (∀ n, 0 < n ∧ n ≤ nuts → move n = n - 1) →
    (end_condition 3 = true) →
    (∀ x y z, x + y + z = 3 ↔ end_condition (x + y + z)) → 
    game_win_guarantee :=
by
  intros nuts players move end_condition H1 H2 H3 H4 H5
  sorry

end first_player_wins_l239_239044


namespace intersection_of_M_and_N_is_12_l239_239720

def M : Set ℤ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℤ := {1, 2, 3}

theorem intersection_of_M_and_N_is_12 : M ∩ N = {1, 2} :=
by
  sorry

end intersection_of_M_and_N_is_12_l239_239720


namespace sin_add_pi_over_three_l239_239930

theorem sin_add_pi_over_three (α : ℝ) (h : Real.sin (α - 2 * Real.pi / 3) = 1 / 4) : 
  Real.sin (α + Real.pi / 3) = -1 / 4 := by
  sorry

end sin_add_pi_over_three_l239_239930


namespace irreducible_fraction_denominator_l239_239415

theorem irreducible_fraction_denominator :
  let num := 201920192019
  let denom := 191719171917
  let gcd_num_denom := Int.gcd num denom
  let irreducible_denom := denom / gcd_num_denom
  irreducible_denom = 639 :=
by
  sorry

end irreducible_fraction_denominator_l239_239415


namespace total_number_of_coins_l239_239185

theorem total_number_of_coins {N B : ℕ} 
    (h1 : B - 2 = Nat.floor (N / 9))
    (h2 : N - 6 * (B - 3) = 3) 
    : N = 45 :=
by
  sorry

end total_number_of_coins_l239_239185


namespace sqrt_3_between_neg_1_and_2_l239_239971

theorem sqrt_3_between_neg_1_and_2 : -1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by
  sorry

end sqrt_3_between_neg_1_and_2_l239_239971


namespace solution_set_for_inequality_l239_239124

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

noncomputable def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

theorem solution_set_for_inequality
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_decreasing : decreasing_on f (Set.Iio 0))
  (h_f1 : f 1 = 0) :
  {x : ℝ | x^3 * f x > 0} = {x : ℝ | x > 1 ∨ x < -1} :=
by
  sorry

end solution_set_for_inequality_l239_239124


namespace buckets_required_l239_239874

theorem buckets_required (C : ℚ) (N : ℕ) (h : 250 * (4/5 : ℚ) * C = N * C) : N = 200 :=
by
  sorry

end buckets_required_l239_239874


namespace calculation_result_l239_239055

theorem calculation_result : (1955 - 1875)^2 / 64 = 100 := by
  sorry

end calculation_result_l239_239055


namespace geom_prog_235_l239_239962

theorem geom_prog_235 (q : ℝ) (k n : ℕ) (hk : 1 < k) (hn : k < n) : 
  ¬ (q > 0 ∧ q ≠ 1 ∧ 3 = 2 * q^(k - 1) ∧ 5 = 2 * q^(n - 1)) := 
by 
  sorry

end geom_prog_235_l239_239962


namespace no_solution_system_l239_239964

noncomputable def system_inconsistent : Prop :=
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 8 ∧ 6 * x - 8 * y = 12)

theorem no_solution_system : system_inconsistent :=
by
  sorry

end no_solution_system_l239_239964


namespace distinct_digit_sums_l239_239520

theorem distinct_digit_sums (A B C E D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ E ∧ A ≠ D ∧ B ≠ C ∧ B ≠ E ∧ B ≠ D ∧ C ≠ E ∧ C ≠ D ∧ E ≠ D)
 (h_ab : A + B = D) (h_ab_lt_10 : A + B < 10) (h_ce : C + E = D) :
  ∃ (x : ℕ), x = 8 := 
sorry

end distinct_digit_sums_l239_239520


namespace inequality_solution_l239_239941

theorem inequality_solution (x : ℝ) : 3 * x + 2 ≥ 5 ↔ x ≥ 1 :=
by sorry

end inequality_solution_l239_239941


namespace inequality_proof_l239_239554

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := by
  sorry

end inequality_proof_l239_239554


namespace remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l239_239649

def f (x : ℝ) : ℝ := x^15 + 1

theorem remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0 : f (-1) = 0 := by
  sorry

end remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l239_239649


namespace largest_sum_of_two_largest_angles_of_EFGH_l239_239224

theorem largest_sum_of_two_largest_angles_of_EFGH (x d : ℝ) (y z : ℝ) :
  (∃ a b : ℝ, a + 2 * b = x + 70 ∧ a + b = 70 ∧ 2 * a + 3 * b = 180) ∧
  (2 * x + 3 * d = 180) ∧ (x = 30) ∧ (y = 70) ∧ (z = 100) ∧ (z + 70 = x + d) ∧
  x + d + x + 2 * d + x + 3 * d + x = 360 →
  max (70 + y) (70 + z) + max (y + 70) (z + 70) = 210 := 
sorry

end largest_sum_of_two_largest_angles_of_EFGH_l239_239224


namespace length_of_segment_l239_239627

theorem length_of_segment (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi / 2)
  (h₁ : 6 * Real.cos x = 5 * Real.tan x) :
  ∃ P_1 P_2 : ℝ, P_1 = 0 ∧ P_2 = (1 / 2) * Real.sin x ∧ abs (P_2 - P_1) = 1 / 3 :=
by
  sorry

end length_of_segment_l239_239627


namespace combined_weight_l239_239353

theorem combined_weight (S R : ℝ) (h1 : S - 5 = 2 * R) (h2 : S = 75) : S + R = 110 :=
sorry

end combined_weight_l239_239353


namespace initial_bottle_count_l239_239936

variable (B: ℕ)

-- Conditions: Each bottle holds 15 stars, bought 3 more bottles, total 75 stars to fill
def bottle_capacity := 15
def additional_bottles := 3
def total_stars := 75

-- The main statement we want to prove
theorem initial_bottle_count (h : (B + additional_bottles) * bottle_capacity = total_stars) : 
    B = 2 :=
by sorry

end initial_bottle_count_l239_239936


namespace find_a_8_l239_239160

variable {α : Type*} [LinearOrderedField α]
variables (a : ℕ → α) (n : ℕ)

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) := ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition
def given_condition (a : ℕ → α) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- Main theorem to prove
theorem find_a_8 (h_arith : is_arithmetic_seq a) (h_cond : given_condition a) : a 8 = 24 :=
  sorry

end find_a_8_l239_239160


namespace total_worth_of_presents_l239_239855

-- Define the costs as given in the conditions
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def bracelet_cost : ℕ := 2 * ring_cost

-- Define the total worth of the presents
def total_worth : ℕ := ring_cost + car_cost + bracelet_cost

-- Statement: Prove the total worth is 14000
theorem total_worth_of_presents : total_worth = 14000 :=
by
  -- Here is the proof statement
  sorry

end total_worth_of_presents_l239_239855


namespace vincent_total_loads_l239_239689

def loads_wednesday : Nat := 2 + 1 + 3

def loads_thursday : Nat := 2 * loads_wednesday

def loads_friday : Nat := loads_thursday / 2

def loads_saturday : Nat := loads_wednesday / 3

def total_loads : Nat := loads_wednesday + loads_thursday + loads_friday + loads_saturday

theorem vincent_total_loads : total_loads = 20 := by
  -- Proof will be filled in here
  sorry

end vincent_total_loads_l239_239689


namespace lizzy_loan_amount_l239_239790

noncomputable def interest_rate : ℝ := 0.20
noncomputable def initial_amount : ℝ := 30
noncomputable def final_amount : ℝ := 33

theorem lizzy_loan_amount (X : ℝ) (h : initial_amount + (1 + interest_rate) * X = final_amount) : X = 2.5 := 
by
  sorry

end lizzy_loan_amount_l239_239790


namespace no_common_solution_general_case_l239_239588

-- Define the context: three linear equations in two variables
variables {a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℝ}

-- Statement of the theorem
theorem no_common_solution_general_case :
  (∃ (x y : ℝ), a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2 ∧ a3 * x + b3 * y = c3) →
  (a1 * b2 ≠ a2 * b1 ∧ a1 * b3 ≠ a3 * b1 ∧ a2 * b3 ≠ a3 * b2) →
  false := 
sorry

end no_common_solution_general_case_l239_239588


namespace find_a6_l239_239851

-- Define the geometric sequence conditions
noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the specific sequence with given initial conditions and sum of first three terms
theorem find_a6 : 
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (0 < q) ∧ (q ≠ 1) ∧ geom_seq a q ∧ 
    a 1 = 96 ∧ 
    (a 1 + a 2 + a 3 = 168) ∧
    a 6 = 3 := 
by
  sorry

end find_a6_l239_239851


namespace find_number_of_small_branches_each_branch_grows_l239_239410

theorem find_number_of_small_branches_each_branch_grows :
  ∃ x : ℕ, 1 + x + x^2 = 43 ∧ x = 6 :=
by {
  sorry
}

end find_number_of_small_branches_each_branch_grows_l239_239410


namespace mail_per_house_l239_239777

theorem mail_per_house (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : 
  total_mail / total_houses = 6 := 
by 
  sorry

end mail_per_house_l239_239777


namespace cos_of_angle_B_l239_239594

theorem cos_of_angle_B (A B C a b c : Real) (h₁ : A - C = Real.pi / 2) (h₂ : 2 * b = a + c) 
  (h₃ : 2 * a * Real.sin A = 2 * b * Real.sin B) (h₄ : 2 * c * Real.sin C = 2 * b * Real.sin B) :
  Real.cos B = 3 / 4 := by
  sorry

end cos_of_angle_B_l239_239594


namespace food_company_total_food_l239_239832

theorem food_company_total_food (boxes : ℕ) (kg_per_box : ℕ) (full_boxes : boxes = 388) (weight_per_box : kg_per_box = 2) :
  boxes * kg_per_box = 776 :=
by
  -- the proof would go here
  sorry

end food_company_total_food_l239_239832


namespace increasing_interval_l239_239435

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x^2

theorem increasing_interval :
  ∃ a b : ℝ, (0 < a) ∧ (a < b) ∧ (b = 1/2) ∧ (∀ x : ℝ, a < x ∧ x < b → (deriv f x > 0)) :=
by
  sorry

end increasing_interval_l239_239435


namespace mike_total_time_spent_l239_239400

theorem mike_total_time_spent : 
  let hours_watching_tv_per_day := 4
  let days_per_week := 7
  let days_playing_video_games := 3
  let hours_playing_video_games_per_day := hours_watching_tv_per_day / 2
  let total_hours_watching_tv := hours_watching_tv_per_day * days_per_week
  let total_hours_playing_video_games := hours_playing_video_games_per_day * days_playing_video_games
  let total_time_spent := total_hours_watching_tv + total_hours_playing_video_games
  total_time_spent = 34 :=
by
  sorry

end mike_total_time_spent_l239_239400


namespace probability_correct_digit_in_two_attempts_l239_239069

theorem probability_correct_digit_in_two_attempts :
  let total_digits := 10
  let probability_first_correct := 1 / total_digits
  let probability_first_incorrect := 9 / total_digits
  let probability_second_correct_if_first_incorrect := 1 / (total_digits - 1)
  (probability_first_correct + probability_first_incorrect * probability_second_correct_if_first_incorrect) = 1 / 5 := 
sorry

end probability_correct_digit_in_two_attempts_l239_239069


namespace tan_of_acute_angle_l239_239417

open Real

theorem tan_of_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 2 * sin (α - 15 * π / 180) - 1 = 0) : tan α = 1 :=
by
  sorry

end tan_of_acute_angle_l239_239417


namespace basket_weight_l239_239340

variables 
  (B : ℕ) -- Weight of the basket
  (L : ℕ) -- Lifting capacity of one balloon

-- Condition: One balloon can lift a basket with contents weighing not more than 80 kg
axiom one_balloon_lifts (h1 : B + L ≤ 80) : Prop

-- Condition: Two balloons can lift a basket with contents weighing not more than 180 kg
axiom two_balloons_lift (h2 : B + 2 * L ≤ 180) : Prop

-- The proof problem: Determine B under the given conditions
theorem basket_weight (B : ℕ) (L : ℕ) (h1 : B + L ≤ 80) (h2 : B + 2 * L ≤ 180) : B = 20 :=
  sorry

end basket_weight_l239_239340


namespace large_cube_surface_area_l239_239593

-- Define given conditions
def small_cube_volume := 512 -- volume in cm^3
def num_small_cubes := 8

-- Define side length of small cube
def small_cube_side_length := (small_cube_volume : ℝ)^(1/3)

-- Define side length of large cube
def large_cube_side_length := 2 * small_cube_side_length

-- Surface area formula for a cube
def surface_area (side_length : ℝ) := 6 * side_length^2

-- Theorem: The surface area of the large cube is 1536 cm^2
theorem large_cube_surface_area :
  surface_area large_cube_side_length = 1536 :=
sorry

end large_cube_surface_area_l239_239593


namespace quadrilateral_ABCD_r_plus_s_l239_239263

noncomputable def AB_is (AB : Real) (r s : Nat) : Prop :=
  AB = r + Real.sqrt s

theorem quadrilateral_ABCD_r_plus_s :
  ∀ (BC CD AD : Real) (mA mB : ℕ) (r s : ℕ), 
  BC = 7 → 
  CD = 10 → 
  AD = 8 → 
  mA = 60 → 
  mB = 60 → 
  AB_is AB r s →
  r + s = 99 :=
by intros BC CD AD mA mB r s hBC hCD hAD hMA hMB hAB_is
   sorry

end quadrilateral_ABCD_r_plus_s_l239_239263


namespace percent_blue_marbles_l239_239418

theorem percent_blue_marbles (total_items buttons red_marbles : ℝ) 
  (H1 : buttons = 0.30 * total_items)
  (H2 : red_marbles = 0.50 * (total_items - buttons)) :
  (total_items - buttons - red_marbles) / total_items = 0.35 :=
by 
  sorry

end percent_blue_marbles_l239_239418


namespace black_shirts_in_pack_l239_239534

-- defining the conditions
variables (B : ℕ) -- the number of black shirts in each pack
variable (total_shirts : ℕ := 21)
variable (yellow_shirts_per_pack : ℕ := 2)
variable (black_packs : ℕ := 3)
variable (yellow_packs : ℕ := 3)

-- ensuring the conditions are met, the total shirts equals 21
def total_black_shirts := black_packs * B
def total_yellow_shirts := yellow_packs * yellow_shirts_per_pack

-- the proof problem
theorem black_shirts_in_pack : total_black_shirts + total_yellow_shirts = total_shirts → B = 5 := by
  sorry

end black_shirts_in_pack_l239_239534


namespace quadratic_roots_m_eq_2_quadratic_discriminant_pos_l239_239586

theorem quadratic_roots_m_eq_2 (x : ℝ) (m : ℝ) (h1 : m = 2) : x^2 + 2 * x - 3 = 0 ↔ (x = -3 ∨ x = 1) :=
by sorry

theorem quadratic_discriminant_pos (m : ℝ) : m^2 + 12 > 0 :=
by sorry

end quadratic_roots_m_eq_2_quadratic_discriminant_pos_l239_239586


namespace point_probability_in_cone_l239_239887

noncomputable def volume_of_cone (S : ℝ) (h : ℝ) : ℝ :=
  (1/3) * S * h

theorem point_probability_in_cone (P M : ℝ) (S_ABC : ℝ) (h_P h_M : ℝ)
  (h_volume_condition : volume_of_cone S_ABC h_P ≤ volume_of_cone S_ABC h_M / 3) :
  (1 - (2 / 3) ^ 3) = 19 / 27 :=
by
  sorry

end point_probability_in_cone_l239_239887


namespace identify_correct_statement_l239_239249

-- Definitions based on conditions
def population (athletes : ℕ) : Prop := athletes = 1000
def is_individual (athlete : ℕ) : Prop := athlete ≤ 1000
def is_sample (sampled_athletes : ℕ) (sample_size : ℕ) : Prop := sampled_athletes = 100 ∧ sample_size = 100

-- Theorem statement based on the conclusion
theorem identify_correct_statement (athletes : ℕ) (sampled_athletes : ℕ) (sample_size : ℕ)
    (h1 : population athletes) (h2 : ∀ a, is_individual a) (h3 : is_sample sampled_athletes sample_size) : 
    (sampled_athletes = 100) ∧ (sample_size = 100) :=
by
  sorry

end identify_correct_statement_l239_239249


namespace total_shingles_needed_l239_239202

-- Defining the dimensions of the house and the porch
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_length : ℝ := 6
def porch_width : ℝ := 4.5

-- The goal is to prove that the total area of the shingles needed is 232 square feet
theorem total_shingles_needed :
  (house_length * house_width) + (porch_length * porch_width) = 232 := by
  sorry

end total_shingles_needed_l239_239202


namespace ceilings_left_to_paint_l239_239730

theorem ceilings_left_to_paint
    (floors : ℕ)
    (rooms_per_floor : ℕ)
    (ceilings_painted_this_week : ℕ)
    (hallways_per_floor : ℕ)
    (hallway_ceilings_per_hallway : ℕ)
    (ceilings_painted_ratio : ℚ)
    : floors = 4
    → rooms_per_floor = 7
    → ceilings_painted_this_week = 12
    → hallways_per_floor = 1
    → hallway_ceilings_per_hallway = 1
    → ceilings_painted_ratio = 1 / 4
    → (floors * rooms_per_floor + floors * hallways_per_floor * hallway_ceilings_per_hallway 
        - ceilings_painted_this_week 
        - (ceilings_painted_ratio * ceilings_painted_this_week + floors * hallway_ceilings_per_hallway) = 13) :=
by
  intros
  sorry

end ceilings_left_to_paint_l239_239730


namespace sqrt_x_eq_0_123_l239_239495

theorem sqrt_x_eq_0_123 (x : ℝ) (h1 : Real.sqrt 15129 = 123) (h2 : Real.sqrt x = 0.123) : x = 0.015129 := by
  -- proof goes here, but it is omitted
  sorry

end sqrt_x_eq_0_123_l239_239495


namespace leak_empties_tank_in_30_hours_l239_239006

-- Define the known rates based on the problem conditions
def rate_pipe_a : ℚ := 1 / 12
def combined_rate : ℚ := 1 / 20

-- Define the rate at which the leak empties the tank
def rate_leak : ℚ := rate_pipe_a - combined_rate

-- Define the time it takes for the leak to empty the tank
def time_to_empty_tank : ℚ := 1 / rate_leak

-- The theorem that needs to be proved
theorem leak_empties_tank_in_30_hours : time_to_empty_tank = 30 :=
sorry

end leak_empties_tank_in_30_hours_l239_239006


namespace placement_ways_l239_239842

theorem placement_ways (rows cols crosses : ℕ) (h1 : rows = 3) (h2 : cols = 4) (h3 : crosses = 4)
  (condition : ∀ r : Fin rows, ∃ c : Fin cols, r < rows ∧ c < cols) : 
  (∃ n, n = (3 * 6 * 2) → n = 36) :=
by 
  -- Proof placeholder
  sorry

end placement_ways_l239_239842


namespace circumscribed_circle_radius_l239_239105

noncomputable def radius_of_circumscribed_circle 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) : ℝ :=
2

theorem circumscribed_circle_radius 
  {a b c A B C : ℝ} 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) :
  radius_of_circumscribed_circle a b c A B C h1 h2 h3 = 2 :=
sorry

end circumscribed_circle_radius_l239_239105


namespace count_distinct_four_digit_numbers_ending_in_25_l239_239216

-- Define what it means for a number to be a four-digit number according to the conditions in (a).
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define what it means for a number to be divisible by 5 and end in 25 according to the conditions in (a).
def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the problem as a theorem in Lean
theorem count_distinct_four_digit_numbers_ending_in_25 : 
  ∃ (count : ℕ), count = 90 ∧ 
    (∀ n, is_four_digit_number n ∧ ends_in_25 n → n % 5 = 0) :=
by
  sorry

end count_distinct_four_digit_numbers_ending_in_25_l239_239216


namespace interest_second_month_l239_239544

theorem interest_second_month {P r n : ℝ} (hP : P = 200) (hr : r = 0.10) (hn : n = 12) :
  (P * (1 + r / n) ^ (n * (1/12)) - P) * r / n = 1.68 :=
by
  sorry

end interest_second_month_l239_239544


namespace quadrilateral_area_is_48_l239_239231

structure Quadrilateral :=
  (PQ QR RS SP : ℝ)
  (angle_QRS angle_SPQ : ℝ)

def quadrilateral_example : Quadrilateral :=
{ PQ := 11, QR := 7, RS := 9, SP := 3, angle_QRS := 90, angle_SPQ := 90 }

noncomputable def area_of_quadrilateral (Q : Quadrilateral) : ℝ :=
  (1/2 * Q.PQ * Q.SP) + (1/2 * Q.QR * Q.RS)

theorem quadrilateral_area_is_48 (Q : Quadrilateral) (h1 : Q.PQ = 11) (h2 : Q.QR = 7) (h3 : Q.RS = 9) (h4 : Q.SP = 3) (h5 : Q.angle_QRS = 90) (h6 : Q.angle_SPQ = 90) :
  area_of_quadrilateral Q = 48 :=
by
  -- Here would be the proof
  sorry

end quadrilateral_area_is_48_l239_239231


namespace find_c_l239_239658

-- Define the functions p and q as given in the conditions
def p (x : ℝ) : ℝ := 3 * x - 9
def q (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

-- State the main theorem with conditions and goal
theorem find_c (c : ℝ) (h : p (q 3 c) = 15) : c = 4 := by
  sorry -- Proof is not required

end find_c_l239_239658


namespace drivers_schedule_l239_239281

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l239_239281


namespace bob_sheep_and_ratio_l239_239635

-- Define the initial conditions
def mary_initial_sheep : ℕ := 300
def additional_sheep_bob_has : ℕ := 35
def sheep_mary_buys : ℕ := 266
def fewer_sheep_than_bob : ℕ := 69

-- Define the number of sheep Bob has
def bob_sheep (mary_initial_sheep : ℕ) (additional_sheep_bob_has : ℕ) : ℕ := 
  mary_initial_sheep + additional_sheep_bob_has

-- Define the number of sheep Mary has after buying more sheep
def mary_new_sheep (mary_initial_sheep : ℕ) (sheep_mary_buys : ℕ) : ℕ := 
  mary_initial_sheep + sheep_mary_buys

-- Define the relation between Mary's and Bob's sheep (after Mary buys sheep)
def mary_bob_relation (mary_new_sheep : ℕ) (fewer_sheep_than_bob : ℕ) : Prop :=
  mary_new_sheep + fewer_sheep_than_bob = bob_sheep mary_initial_sheep additional_sheep_bob_has

-- Define the proof problem
theorem bob_sheep_and_ratio : 
  bob_sheep mary_initial_sheep additional_sheep_bob_has = 635 ∧ 
  (bob_sheep mary_initial_sheep additional_sheep_bob_has) * 300 = 635 * mary_initial_sheep := 
by 
  sorry

end bob_sheep_and_ratio_l239_239635


namespace total_people_on_hike_l239_239578

theorem total_people_on_hike
  (cars : ℕ) (cars_people : ℕ)
  (taxis : ℕ) (taxis_people : ℕ)
  (vans : ℕ) (vans_people : ℕ)
  (buses : ℕ) (buses_people : ℕ)
  (minibuses : ℕ) (minibuses_people : ℕ)
  (h_cars : cars = 7) (h_cars_people : cars_people = 4)
  (h_taxis : taxis = 10) (h_taxis_people : taxis_people = 6)
  (h_vans : vans = 4) (h_vans_people : vans_people = 5)
  (h_buses : buses = 3) (h_buses_people : buses_people = 20)
  (h_minibuses : minibuses = 2) (h_minibuses_people : minibuses_people = 8) :
  cars * cars_people + taxis * taxis_people + vans * vans_people + buses * buses_people + minibuses * minibuses_people = 184 :=
by
  sorry

end total_people_on_hike_l239_239578


namespace abs_eq_zero_iff_l239_239479

theorem abs_eq_zero_iff {a : ℝ} (h : |a + 3| = 0) : a = -3 :=
sorry

end abs_eq_zero_iff_l239_239479


namespace people_with_fewer_than_7_cards_l239_239700

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l239_239700


namespace trapezoid_shorter_base_length_l239_239814

theorem trapezoid_shorter_base_length
  (L B : ℕ)
  (hL : L = 125)
  (hB : B = 5)
  (h : ∀ x, (L - x) / 2 = B → x = 115) :
  ∃ x, x = 115 := by
    sorry

end trapezoid_shorter_base_length_l239_239814


namespace o_hara_triple_example_l239_239641

-- definitions
def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a) + (Real.sqrt b) = x

-- conditions
def a : ℕ := 81
def b : ℕ := 49
def x : ℕ := 16

-- statement
theorem o_hara_triple_example : is_OHara_triple a b x :=
by
  sorry

end o_hara_triple_example_l239_239641


namespace ninety_percent_of_population_is_expected_number_l239_239816

/-- Define the total population of the village -/
def total_population : ℕ := 9000

/-- Define the percentage rate as a fraction -/
def percentage_rate : ℕ := 90

/-- Define the expected number of people representing 90% of the population -/
def expected_number : ℕ := 8100

/-- The proof problem: Prove that 90% of the total population is 8100 -/
theorem ninety_percent_of_population_is_expected_number :
  (percentage_rate * total_population / 100) = expected_number :=
by
  sorry

end ninety_percent_of_population_is_expected_number_l239_239816


namespace option_d_can_form_triangle_l239_239765

noncomputable def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem option_d_can_form_triangle : satisfies_triangle_inequality 2 3 4 :=
by {
  -- Using the triangle inequality theorem to check
  sorry
}

end option_d_can_form_triangle_l239_239765


namespace part1_positive_root_part2_negative_solution_l239_239135

theorem part1_positive_root (x k : ℝ) (hx1 : x > 0)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k = 6 ∨ k = -8 := 
sorry

theorem part2_negative_solution (x k : ℝ) (hx2 : x < 0)
  (hx_ne1 : x ≠ 1) (hx_ne_neg1 : x ≠ -1)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k < -1 ∧ k ≠ -8 := 
sorry

end part1_positive_root_part2_negative_solution_l239_239135


namespace polynomial_factor_l239_239201

def factorization_condition (p q : ℤ) : Prop :=
  ∃ r s : ℤ, 
    p = 4 * r ∧ 
    q = -3 * r + 4 * s ∧ 
    40 = 2 * r - 3 * s + 16 ∧ 
    -20 = s - 12

theorem polynomial_factor (p q : ℤ) (hpq : factorization_condition p q) : (p, q) = (0, -32) :=
by sorry

end polynomial_factor_l239_239201


namespace binomial_coeff_x5y3_in_expansion_eq_56_l239_239000

theorem binomial_coeff_x5y3_in_expansion_eq_56:
  let n := 8
  let k := 3
  let binom_coeff := Nat.choose n k
  binom_coeff = 56 := 
by sorry

end binomial_coeff_x5y3_in_expansion_eq_56_l239_239000


namespace find_n_l239_239662

theorem find_n (n a b : ℕ) 
  (h1 : a > 1)
  (h2 : a ∣ n)
  (h3 : b > a)
  (h4 : b ∣ n)
  (h5 : ∀ m, 1 < m ∧ m < a → ¬ m ∣ n)
  (h6 : ∀ m, a < m ∧ m < b → ¬ m ∣ n)
  (h7 : n = a^a + b^b)
  : n = 260 :=
by sorry

end find_n_l239_239662


namespace term_in_AP_is_zero_l239_239416

theorem term_in_AP_is_zero (a d : ℤ) 
  (h : (a + 4 * d) + (a + 20 * d) = (a + 7 * d) + (a + 14 * d) + (a + 12 * d)) :
  a + (-9) * d = 0 :=
by
  sorry

end term_in_AP_is_zero_l239_239416


namespace tens_digit_of_M_l239_239995

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

theorem tens_digit_of_M {M : ℕ} (h : 10 ≤ M ∧ M < 100) (h_eq : M = P M + S M + 6) :
  M / 10 = 1 ∨ M / 10 = 2 :=
sorry

end tens_digit_of_M_l239_239995


namespace simplify_problem_1_simplify_problem_2_l239_239436

-- Problem 1: Statement of Simplification Proof
theorem simplify_problem_1 :
  (- (99 + (71 / 72)) * 36 = - (3599 + 1 / 2)) :=
by sorry

-- Problem 2: Statement of Simplification Proof
theorem simplify_problem_2 :
  (-3 * (1 / 4) - 2.5 * (-2.45) + (7 / 2) * (1 / 4) = 6 + 1 / 4) :=
by sorry

end simplify_problem_1_simplify_problem_2_l239_239436


namespace trig_identity_l239_239194

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 3 * Real.sin x = 4) : 
  2 * Real.sin x + 3 * Real.cos x = 1 ∨ 2 * Real.sin x + 3 * Real.cos x = 3 :=
sorry

end trig_identity_l239_239194


namespace michelle_initial_crayons_l239_239112

variable (m j : Nat)

axiom janet_crayons : j = 2
axiom michelle_has_after_gift : m + j = 4

theorem michelle_initial_crayons : m = 2 :=
by
  sorry

end michelle_initial_crayons_l239_239112


namespace money_first_day_l239_239771

-- Define the total mushrooms
def total_mushrooms : ℕ := 65

-- Define the mushrooms picked on the second day
def mushrooms_day2 : ℕ := 12

-- Define the mushrooms picked on the third day
def mushrooms_day3 : ℕ := 2 * mushrooms_day2

-- Define the price per mushroom
def price_per_mushroom : ℕ := 2

-- Prove that the amount of money made on the first day is $58
theorem money_first_day : (total_mushrooms - mushrooms_day2 - mushrooms_day3) * price_per_mushroom = 58 := 
by
  -- Skip the proof
  sorry

end money_first_day_l239_239771


namespace problem_proof_l239_239846

-- Definitions based on conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Theorem to prove
theorem problem_proof (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end problem_proof_l239_239846


namespace line_bisects_circle_l239_239012

theorem line_bisects_circle (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, l x y ↔ x - y = 0) → 
  (∀ x y : ℝ, C x y ↔ x^2 + y^2 = 1) → 
  ∀ x y : ℝ, (x - y = 0) ∨ (x + y = 0) → l x y ∧ C x y → l x y = (x - y = 0) := by
  sorry

end line_bisects_circle_l239_239012


namespace trigonometric_identity_l239_239779

open Real

theorem trigonometric_identity (α : ℝ) : 
  sin α * sin α + cos (π / 6 + α) * cos (π / 6 + α) + sin α * cos (π / 6 + α) = 3 / 4 :=
sorry

end trigonometric_identity_l239_239779


namespace michelle_will_have_four_crayons_l239_239684

def michelle_crayons (m j : ℕ) : ℕ := m + j

theorem michelle_will_have_four_crayons (H₁ : michelle_crayons 2 2 = 4) : michelle_crayons 2 2 = 4 :=
by
  sorry

end michelle_will_have_four_crayons_l239_239684


namespace part1_solution_part2_solution_l239_239937

noncomputable def find_prices (price_peanuts price_tea : ℝ) : Prop :=
price_peanuts + 40 = price_tea ∧
50 * price_peanuts = 10 * price_tea

theorem part1_solution :
  ∃ (price_peanuts price_tea : ℝ), find_prices price_peanuts price_tea :=
by
  sorry

def cost_function (m : ℝ) : ℝ :=
6 * m + 36 * (60 - m)

def profit_function (m : ℝ) : ℝ :=
(10 - 6) * m + (50 - 36) * (60 - m)

noncomputable def max_profit := 540

theorem part2_solution :
  ∃ (m t : ℝ), 30 ≤ m ∧ m ≤ 40 ∧ cost_function m ≤ 1260 ∧ profit_function m = max_profit :=
by
  sorry

end part1_solution_part2_solution_l239_239937


namespace number_of_subsets_of_set_l239_239607

theorem number_of_subsets_of_set (x y : ℝ) 
  (z : ℂ) (hz : z = (2 - (1 : ℂ) * Complex.I) / (1 + (2 : ℂ) * Complex.I))
  (hx : z.re = x) (hy : z.im = y) : 
  (Finset.powerset ({x, 2^x, y} : Finset ℝ)).card = 8 :=
by
  sorry

end number_of_subsets_of_set_l239_239607


namespace sum_of_largest_three_l239_239825

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l239_239825


namespace problem_statement_l239_239464

theorem problem_statement (x y a : ℝ) (h1 : x + a < y + a) (h2 : a * x > a * y) : x < y ∧ a < 0 :=
sorry

end problem_statement_l239_239464


namespace perimeter_of_face_given_volume_l239_239190

-- Definitions based on conditions
def volume_of_cube (v : ℝ) := v = 512

def side_of_cube (s : ℝ) := s^3 = 512

def perimeter_of_face (p s : ℝ) := p = 4 * s

-- Lean 4 statement: prove that the perimeter of one face of the cube is 32 cm given the volume is 512 cm³.
theorem perimeter_of_face_given_volume :
  ∃ s : ℝ, volume_of_cube (s^3) ∧ perimeter_of_face 32 s :=
by sorry

end perimeter_of_face_given_volume_l239_239190


namespace determine_friends_l239_239924

inductive Grade
| first
| second
| third
| fourth

inductive Name
| Petya
| Kolya
| Alyosha
| Misha
| Dima
| Borya
| Vasya

inductive Surname
| Ivanov
| Krylov
| Petrov
| Orlov

structure Friend :=
  (name : Name)
  (surname : Surname)
  (grade : Grade)

def friends : List Friend :=
  [ {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first},
    {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second},
    {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third},
    {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ]

theorem determine_friends : ∃ l : List Friend, 
  {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first} ∈ l ∧
  {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second} ∈ l ∧
  {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third} ∈ l ∧
  {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ∈ l :=
by 
  use friends
  repeat { simp [friends] }


end determine_friends_l239_239924


namespace value_of_x_l239_239866

theorem value_of_x (x : ℝ) : 3 - 5 + 7 = 6 - x → x = 1 :=
by
  intro h
  sorry

end value_of_x_l239_239866


namespace solve_for_a_plus_b_l239_239681

theorem solve_for_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, a * (x + b) = 3 * x + 12) → a + b = 7 :=
by
  intros h
  sorry

end solve_for_a_plus_b_l239_239681


namespace Nero_speed_is_8_l239_239352

-- Defining the conditions
def Jerome_time := 6 -- in hours
def Nero_time := 3 -- in hours
def Jerome_speed := 4 -- in miles per hour

-- Calculation step
def Distance := Jerome_speed * Jerome_time

-- The theorem we need to prove (Nero's speed)
theorem Nero_speed_is_8 :
  (Distance / Nero_time) = 8 := by
  sorry

end Nero_speed_is_8_l239_239352


namespace trig_identity_l239_239171

variable (α : ℝ)
variable (h : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (h₁ : Real.sin α = 4 / 5)

theorem trig_identity : Real.sin (α + Real.pi / 4) + Real.cos (α + Real.pi / 4) = -3 * Real.sqrt 2 / 5 := 
by 
  sorry

end trig_identity_l239_239171


namespace inequality_proof_l239_239059

variable {a b c d : ℝ}

theorem inequality_proof
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_sum : a + b + c + d = 3) :
  1 / a^3 + 1 / b^3 + 1 / c^3 + 1 / d^3 ≤ 1 / (a^3 * b^3 * c^3 * d^3) :=
sorry

end inequality_proof_l239_239059


namespace length_to_width_ratio_is_three_l239_239629

def rectangle_ratio (x : ℝ) : Prop :=
  let side_length_large_square := 4 * x
  let length_rectangle := 4 * x
  let width_rectangle := x
  length_rectangle / width_rectangle = 3

-- We state the theorem to be proved
theorem length_to_width_ratio_is_three (x : ℝ) (h : 0 < x) :
  rectangle_ratio x :=
sorry

end length_to_width_ratio_is_three_l239_239629


namespace range_of_b_l239_239917

variable (a b c : ℝ)

theorem range_of_b (h1 : a * c = b^2) (h2 : a + b + c = 3) : -3 ≤ b ∧ b ≤ 1 :=
sorry

end range_of_b_l239_239917


namespace radius_of_circle_through_points_l239_239119

theorem radius_of_circle_through_points : 
  ∃ (x : ℝ), 
  (dist (x, 0) (2, 5) = dist (x, 0) (3, 4)) →
  (∃ (r : ℝ), r = dist (x, 0) (2, 5) ∧ r = 5) :=
by
  sorry

end radius_of_circle_through_points_l239_239119


namespace first_movie_series_seasons_l239_239713

theorem first_movie_series_seasons (S : ℕ) : 
  (∀ E : ℕ, E = 16) → 
  (∀ L : ℕ, L = 2) → 
  (∀ T : ℕ, T = 364) → 
  (∀ second_series_seasons : ℕ, second_series_seasons = 14) → 
  (∀ second_series_remaining : ℕ, second_series_remaining = second_series_seasons * (E - L)) → 
  (E - L = 14) → 
  (second_series_remaining = 196) → 
  (T - second_series_remaining = S * (E - L)) → 
  S = 12 :=
by 
  intros E_16 L_2 T_364 second_series_14 second_series_remaining_196 E_L second_series_total_episodes remaining_episodes
  sorry

end first_movie_series_seasons_l239_239713


namespace product_sum_of_roots_l239_239653

theorem product_sum_of_roots (p q r : ℂ)
  (h_eq : ∀ x : ℂ, (2 : ℂ) * x^3 + (1 : ℂ) * x^2 + (-7 : ℂ) * x + (2 : ℂ) = 0 → (x = p ∨ x = q ∨ x = r)) 
  : p * q + q * r + r * p = -7 / 2 := 
sorry

end product_sum_of_roots_l239_239653


namespace sally_credit_card_balance_l239_239152

theorem sally_credit_card_balance (G P : ℝ) (X : ℝ)  
  (h1 : P = 2 * G)  
  (h2 : XP = X * P)  
  (h3 : G / 3 + XP = (5 / 12) * P) : 
  X = 1 / 4 :=
by
  sorry

end sally_credit_card_balance_l239_239152


namespace min_weighings_to_determine_counterfeit_l239_239016

/-- 
  Given 2023 coins with two counterfeit coins and 2021 genuine coins, 
  and using a balance scale, determine whether the counterfeit coins 
  are heavier or lighter. Prove that the minimum number of weighings 
  required is 3. 
-/
theorem min_weighings_to_determine_counterfeit (n : ℕ) (k : ℕ) (l : ℕ) 
  (h : n = 2023) (h₁ : k = 2) (h₂ : l = 2021) 
  (w₁ w₂ : ℕ → ℝ) -- weights of coins
  (h_fake : ∀ i j, w₁ i = w₁ j) -- counterfeits have same weight
  (h_fake_diff : ∀ i j, i ≠ j → w₁ i ≠ w₂ j) -- fake different from genuine
  (h_genuine : ∀ i j, w₂ i = w₂ j) -- genuines have same weight
  (h_total : ∀ i, i ≤ l + k) -- total coins condition
  : ∃ min_weighings : ℕ, min_weighings = 3 :=
by
  sorry

end min_weighings_to_determine_counterfeit_l239_239016


namespace circle_tangent_radii_l239_239288

theorem circle_tangent_radii (a b c : ℝ) (A : ℝ) (p : ℝ)
  (r r_a r_b r_c : ℝ)
  (h1 : p = (a + b + c) / 2)
  (h2 : r = A / p)
  (h3 : r_a = A / (p - a))
  (h4 : r_b = A / (p - b))
  (h5 : r_c = A / (p - c))
  : 1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  sorry

end circle_tangent_radii_l239_239288


namespace work_completion_time_l239_239990

-- Define the rate of work done by a, b, and c.
def rate_a := 1 / 4
def rate_b := 1 / 12
def rate_c := 1 / 6

-- Define the time each person starts working and the cycle pattern.
def start_time : ℕ := 6 -- in hours
def cycle_pattern := [rate_a, rate_b, rate_c]

-- Calculate the total amount of work done in one cycle of 3 hours.
def work_per_cycle := (rate_a + rate_b + rate_c)

-- Calculate the total time to complete the work.
def total_time_to_complete_work := 2 * 3 -- number of cycles times 3 hours per cycle

-- Calculate the time of completion.
def completion_time := start_time + total_time_to_complete_work

-- Theorem to prove the work completion time.
theorem work_completion_time : completion_time = 12 := 
by
  -- Proof can be filled in here
  sorry

end work_completion_time_l239_239990


namespace slices_left_for_Era_l239_239493

def total_burgers : ℕ := 5
def slices_per_burger : ℕ := 8

def first_friend_slices : ℕ := 3
def second_friend_slices : ℕ := 8
def third_friend_slices : ℕ := 5
def fourth_friend_slices : ℕ := 11
def fifth_friend_slices : ℕ := 6

def total_slices : ℕ := total_burgers * slices_per_burger
def slices_given_to_friends : ℕ := first_friend_slices + second_friend_slices + third_friend_slices + fourth_friend_slices + fifth_friend_slices

theorem slices_left_for_Era : total_slices - slices_given_to_friends = 7 :=
by
  rw [total_slices, slices_given_to_friends]
  exact Eq.refl 7

#reduce slices_left_for_Era

end slices_left_for_Era_l239_239493


namespace find_number_l239_239608

-- Define the condition: a number exceeds by 40 from its 3/8 part.
def exceeds_by_40_from_its_fraction (x : ℝ) := x = (3/8) * x + 40

-- The theorem: prove that the number is 64 given the condition.
theorem find_number (x : ℝ) (h : exceeds_by_40_from_its_fraction x) : x = 64 := 
by
  sorry

end find_number_l239_239608


namespace distinct_solutions_diff_l239_239264

theorem distinct_solutions_diff (r s : ℝ) 
  (h1 : r ≠ s) 
  (h2 : (5*r - 15)/(r^2 + 3*r - 18) = r + 3) 
  (h3 : (5*s - 15)/(s^2 + 3*s - 18) = s + 3) 
  (h4 : r > s) : 
  r - s = 13 :=
sorry

end distinct_solutions_diff_l239_239264


namespace money_distribution_l239_239998

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 360) (h3 : C = 60) : A + B + C = 500 := by
  sorry

end money_distribution_l239_239998


namespace percentage_of_divisible_l239_239413

def count_divisible (n m : ℕ) : ℕ :=
(n / m)

def calculate_percentage (part total : ℕ) : ℚ :=
(part * 100 : ℚ) / (total : ℚ)

theorem percentage_of_divisible (n : ℕ) (k : ℕ) (h₁ : n = 150) (h₂ : k = 6) :
  calculate_percentage (count_divisible n k) n = 16.67 :=
by
  sorry

end percentage_of_divisible_l239_239413


namespace find_constant_l239_239922

noncomputable def expr (x C : ℝ) : ℝ :=
  (x - 1) * (x - 3) * (x - 4) * (x - 6) + C

theorem find_constant :
  (∀ x : ℝ, expr x (-0.5625) ≥ 1) → expr 3.5 (-0.5625) = 1 :=
by
  sorry

end find_constant_l239_239922


namespace max_pies_without_ingredients_l239_239377

theorem max_pies_without_ingredients (total_pies half_chocolate two_thirds_marshmallows three_fifths_cayenne one_eighth_peanuts : ℕ) 
  (h1 : total_pies = 48) 
  (h2 : half_chocolate = total_pies / 2)
  (h3 : two_thirds_marshmallows = 2 * total_pies / 3) 
  (h4 : three_fifths_cayenne = 3 * total_pies / 5)
  (h5 : one_eighth_peanuts = total_pies / 8) : 
  ∃ pies_without_any_ingredients, pies_without_any_ingredients = 16 :=
  by 
    sorry

end max_pies_without_ingredients_l239_239377


namespace human_height_weight_correlated_l239_239200

-- Define the relationships as types
def taxiFareDistanceRelated : Prop := ∀ x y : ℕ, x = y → True
def houseSizePriceRelated : Prop := ∀ x y : ℕ, x = y → True
def humanHeightWeightCorrelated : Prop := ∃ k : ℕ, ∀ x y : ℕ, x / k = y
def ironBlockMassRelated : Prop := ∀ x y : ℕ, x = y → True

-- Main theorem statement
theorem human_height_weight_correlated : humanHeightWeightCorrelated :=
  sorry

end human_height_weight_correlated_l239_239200


namespace original_price_of_article_l239_239856

theorem original_price_of_article :
  ∃ P : ℝ, (P * 0.55 * 0.85 = 920) ∧ P = 1968.04 :=
by
  sorry

end original_price_of_article_l239_239856


namespace quadratic_root_sum_eight_l239_239987

theorem quadratic_root_sum_eight (p r : ℝ) (hp : p > 0) (hr : r > 0) 
  (h : ∀ (x₁ x₂ : ℝ), (x₁ + x₂ = p) -> (x₁ * x₂ = r) -> (x₁ + x₂ = 8)) : r = 8 :=
sorry

end quadratic_root_sum_eight_l239_239987


namespace polynomial_problem_l239_239307

theorem polynomial_problem 
  (d_1 d_2 d_3 d_4 e_1 e_2 e_3 e_4 : ℝ)
  (h : ∀ (x : ℝ),
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + d_1 * x + e_1) * (x^2 + d_2 * x + e_2) * (x^2 + d_3 * x + e_3) * (x^2 + d_4 * x + e_4)) :
  d_1 * e_1 + d_2 * e_2 + d_3 * e_3 + d_4 * e_4 = -1 := 
by
  sorry

end polynomial_problem_l239_239307


namespace complete_squares_l239_239655

def valid_solutions (x y z : ℝ) : Prop :=
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = -2 ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = 6) ∨
  (x = 0 ∧ y = -2 ∧ z = 6) ∨
  (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 4 ∧ y = -2 ∧ z = 0) ∨
  (x = 4 ∧ y = 0 ∧ z = 6) ∨
  (x = 4 ∧ y = -2 ∧ z = 6)

theorem complete_squares (x y z : ℝ) : 
  (x - 2)^2 + (y + 1)^2 = 5 →
  (x - 2)^2 + (z - 3)^2 = 13 →
  (y + 1)^2 + (z - 3)^2 = 10 →
  valid_solutions x y z :=
by
  intros h1 h2 h3
  sorry

end complete_squares_l239_239655


namespace mean_greater_than_median_l239_239821

theorem mean_greater_than_median (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5 
  let median := x + 4 
  mean - median = 4 :=
by 
  sorry

end mean_greater_than_median_l239_239821


namespace buses_needed_for_trip_l239_239161

theorem buses_needed_for_trip :
  ∀ (total_students students_in_vans bus_capacity : ℕ),
  total_students = 500 →
  students_in_vans = 56 →
  bus_capacity = 45 →
  ⌈(total_students - students_in_vans : ℝ) / bus_capacity⌉ = 10 :=
by
  sorry

end buses_needed_for_trip_l239_239161


namespace sum_powers_div_5_iff_l239_239178

theorem sum_powers_div_5_iff (n : ℕ) (h : n > 0) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end sum_powers_div_5_iff_l239_239178


namespace length_of_side_b_l239_239531

theorem length_of_side_b (B C : ℝ) (c b : ℝ) (hB : B = 45 * Real.pi / 180) (hC : C = 60 * Real.pi / 180) (hc : c = 1) :
  b = Real.sqrt 6 / 3 :=
by
  sorry

end length_of_side_b_l239_239531


namespace num_common_points_of_three_lines_l239_239433

def three_planes {P : Type} [AddCommGroup P] (l1 l2 l3 : Set P) : Prop :=
  let p12 := Set.univ \ (l1 ∪ l2)
  let p13 := Set.univ \ (l1 ∪ l3)
  let p23 := Set.univ \ (l2 ∪ l3)
  ∃ (pl12 pl13 pl23 : Set P), 
    p12 = pl12 ∧ p13 = pl13 ∧ p23 = pl23

theorem num_common_points_of_three_lines (l1 l2 l3 : Set ℝ) 
  (h : three_planes l1 l2 l3) : ∃ n : ℕ, n = 0 ∨ n = 1 := by
  sorry

end num_common_points_of_three_lines_l239_239433


namespace solve_system_eq_l239_239705

theorem solve_system_eq (x y : ℚ) 
  (h1 : 3 * x - 7 * y = 31) 
  (h2 : 5 * x + 2 * y = -10) : 
  x = -336 / 205 := 
sorry

end solve_system_eq_l239_239705


namespace sin_2x_from_tan_pi_minus_x_l239_239287

theorem sin_2x_from_tan_pi_minus_x (x : ℝ) (h : Real.tan (Real.pi - x) = 3) : Real.sin (2 * x) = -3 / 5 := by
  sorry

end sin_2x_from_tan_pi_minus_x_l239_239287


namespace milkshake_hours_l239_239546

theorem milkshake_hours (h : ℕ) : 
  (3 * h + 7 * h = 80) → h = 8 := 
by
  intro h_milkshake_eq
  sorry

end milkshake_hours_l239_239546


namespace equation_of_line_l239_239615

theorem equation_of_line (A B : ℝ × ℝ) (M : ℝ × ℝ) (hM : M = (-1, 2)) (hA : A.2 = 0) (hB : B.1 = 0) (hMid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = 4) ∧ ∀ (x y : ℝ), y = a * x + b * y + c → 2 * x - y + 4 = 0 := 
  sorry

end equation_of_line_l239_239615


namespace sqrt_sqrt_16_eq_pm2_l239_239351

theorem sqrt_sqrt_16_eq_pm2 (h : Real.sqrt 16 = 4) : Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm2_l239_239351


namespace remainder_when_divided_by_DE_l239_239568

theorem remainder_when_divided_by_DE (P D E Q R M S C : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = E * M + S) :
  (∃ quotient : ℕ, P = quotient * (D * E) + (S * D + R + C)) :=
by {
  sorry
}

end remainder_when_divided_by_DE_l239_239568


namespace roots_condition_implies_m_range_l239_239411

theorem roots_condition_implies_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ (x₁^2 + (m-1)*x₁ + m^2 - 2 = 0) ∧ (x₂^2 + (m-1)*x₂ + m^2 - 2 = 0))
  → -2 < m ∧ m < 1 :=
by
  sorry

end roots_condition_implies_m_range_l239_239411


namespace kim_trip_time_l239_239828

-- Definitions
def distance_freeway : ℝ := 120
def distance_mountain : ℝ := 25
def speed_ratio : ℝ := 4
def time_mountain : ℝ := 75

-- The problem statement
theorem kim_trip_time : ∃ t_freeway t_total : ℝ,
  t_freeway = distance_freeway / (speed_ratio * (distance_mountain / time_mountain)) ∧
  t_total = time_mountain + t_freeway ∧
  t_total = 165 := by
  sorry

end kim_trip_time_l239_239828


namespace complement_of_angle_l239_239455

theorem complement_of_angle (x : ℝ) (h1 : 3 * x + 10 = 90 - x) : 3 * x + 10 = 70 :=
by
  sorry

end complement_of_angle_l239_239455


namespace equivalent_single_discount_l239_239114

theorem equivalent_single_discount (p : ℝ) :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  let final_price := (1 - discount1) * (1 - discount2) * (1 - discount3) * p
  (1 - final_price / p) = 0.27325 :=
by
  sorry

end equivalent_single_discount_l239_239114


namespace digit_to_make_multiple_of_5_l239_239902

theorem digit_to_make_multiple_of_5 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 9) 
  (N := 71360 + d) : (N % 5 = 0) → (d = 0 ∨ d = 5) :=
by
  sorry

end digit_to_make_multiple_of_5_l239_239902


namespace similar_pentagon_area_l239_239809

theorem similar_pentagon_area
  (K1 K2 : ℝ) (L1 L2 : ℝ)
  (h_similar : true)  -- simplifying the similarity condition as true for the purpose of this example
  (h_K1 : K1 = 18)
  (h_K2 : K2 = 24)
  (h_L1 : L1 = 8.4375) :
  L2 = 15 :=
by
  sorry

end similar_pentagon_area_l239_239809


namespace Caitlin_Sara_weight_l239_239323

variable (A C S : ℕ)

theorem Caitlin_Sara_weight 
  (h1 : A + C = 95) 
  (h2 : A = S + 8) : 
  C + S = 87 := by
  sorry

end Caitlin_Sara_weight_l239_239323


namespace no_consecutive_integers_square_difference_2000_l239_239660

theorem no_consecutive_integers_square_difference_2000 :
  ¬ ∃ a : ℤ, (a + 1) ^ 2 - a ^ 2 = 2000 :=
by {
  -- some detailed steps might go here in a full proof
  sorry
}

end no_consecutive_integers_square_difference_2000_l239_239660


namespace range_of_m_l239_239126

open Real Set

def P (m : ℝ) := |m + 1| ≤ 2
def Q (m : ℝ) := ∃ x : ℝ, x^2 - m*x + 1 = 0 ∧ (m^2 - 4 ≥ 0)

theorem range_of_m (m : ℝ) :
  (¬¬ P m ∧ ¬ (P m ∧ Q m)) → -2 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l239_239126


namespace range_of_a_l239_239864

-- Given definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 1 

-- Monotonicity condition on the interval [1, 2]
def is_monotonic (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (x ≤ y → f x a ≤ f y a) ∨ (x ≤ y → f x a ≥ f y a)

-- The proof objective
theorem range_of_a (a : ℝ) : is_monotonic a → (a ≤ -2 ∨ a ≥ -1) := 
sorry

end range_of_a_l239_239864


namespace y_coordinate_of_C_l239_239656

def Point : Type := (ℤ × ℤ)

def A : Point := (0, 0)
def B : Point := (0, 4)
def D : Point := (4, 4)
def E : Point := (4, 0)

def PentagonArea (C : Point) : ℚ :=
  let triangleArea : ℚ := (1/2 : ℚ) * 4 * ((C.2 : ℚ) - 4)
  let squareArea : ℚ := 4 * 4
  triangleArea + squareArea

theorem y_coordinate_of_C (h : ℤ) (C : Point := (2, h)) : PentagonArea C = 40 → C.2 = 16 :=
by
  sorry

end y_coordinate_of_C_l239_239656


namespace area_ratio_of_circles_l239_239380

theorem area_ratio_of_circles 
  (CX : ℝ)
  (CY : ℝ)
  (RX RY : ℝ)
  (hX : CX = 2 * π * RX)
  (hY : CY = 2 * π * RY)
  (arc_length_equality : (90 / 360) * CX = (60 / 360) * CY) :
  (π * RX^2) / (π * RY^2) = 9 / 4 :=
by
  sorry

end area_ratio_of_circles_l239_239380


namespace largest_real_number_condition_l239_239475

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l239_239475


namespace power_function_no_origin_l239_239296

theorem power_function_no_origin (m : ℝ) :
  (m = 1 ∨ m = 2) → 
  (m^2 - 3 * m + 3 ≠ 0 ∧ (m - 2) * (m + 1) ≤ 0) :=
by
  intro h
  cases h
  case inl =>
    -- m = 1 case will be processed here
    sorry
  case inr =>
    -- m = 2 case will be processed here
    sorry

end power_function_no_origin_l239_239296


namespace centroid_midpoint_triangle_eq_centroid_original_triangle_l239_239697

/-
Prove that the centroid of the triangle formed by the midpoints of the sides of another triangle
is the same as the centroid of the original triangle.
-/
theorem centroid_midpoint_triangle_eq_centroid_original_triangle
  (A B C M N P : ℝ × ℝ)
  (hM : M = (A + B) / 2)
  (hN : N = (A + C) / 2)
  (hP : P = (B + C) / 2) :
  (M.1 + N.1 + P.1) / 3 = (A.1 + B.1 + C.1) / 3 ∧
  (M.2 + N.2 + P.2) / 3 = (A.2 + B.2 + C.2) / 3 :=
by
  sorry

end centroid_midpoint_triangle_eq_centroid_original_triangle_l239_239697


namespace charge_per_trousers_l239_239206

-- Definitions
def pairs_of_trousers : ℕ := 10
def shirts : ℕ := 10
def bill : ℕ := 140
def charge_per_shirt : ℕ := 5

-- Theorem statement
theorem charge_per_trousers :
  ∃ (T : ℕ), (pairs_of_trousers * T + shirts * charge_per_shirt = bill) ∧ (T = 9) :=
by 
  sorry

end charge_per_trousers_l239_239206


namespace equilateral_triangle_not_centrally_symmetric_l239_239011

-- Definitions for the shapes
def is_centrally_symmetric (shape : Type) : Prop := sorry
def Parallelogram : Type := sorry
def LineSegment : Type := sorry
def EquilateralTriangle : Type := sorry
def Rhombus : Type := sorry

-- Main theorem statement
theorem equilateral_triangle_not_centrally_symmetric :
  ¬ is_centrally_symmetric EquilateralTriangle ∧
  is_centrally_symmetric Parallelogram ∧
  is_centrally_symmetric LineSegment ∧
  is_centrally_symmetric Rhombus :=
sorry

end equilateral_triangle_not_centrally_symmetric_l239_239011


namespace line_equation_through_point_and_area_l239_239467

theorem line_equation_through_point_and_area (b S x y : ℝ) 
  (h1 : ∀ y, (x, y) = (-2*b, 0) → True) 
  (h2 : ∀ p1 p2 p3 : ℝ × ℝ, p1 = (-2*b, 0) → p2 = (0, 0) → 
        ∃ k, p3 = (0, k) ∧ S = 1/2 * (2*b) * k) : 2*S*x - b^2*y + 4*b*S = 0 :=
sorry

end line_equation_through_point_and_area_l239_239467


namespace three_units_away_from_neg_one_l239_239241

def is_three_units_away (x : ℝ) (y : ℝ) : Prop := abs (x - y) = 3

theorem three_units_away_from_neg_one :
  { x : ℝ | is_three_units_away x (-1) } = {2, -4} := 
by
  sorry

end three_units_away_from_neg_one_l239_239241


namespace ufo_convention_attendees_l239_239378

theorem ufo_convention_attendees (f m total : ℕ) 
  (h1 : m = 62) 
  (h2 : m = f + 4) : 
  total = 120 :=
by
  sorry

end ufo_convention_attendees_l239_239378


namespace smallest_solution_proof_l239_239838

noncomputable def smallest_solution : ℝ :=
  let n := 11
  let a := 0.533
  n + a

theorem smallest_solution_proof :
  ∃ (x : ℝ), ⌊x^2⌋ - ⌊x⌋^2 = 21 ∧ x = smallest_solution :=
by
  use smallest_solution
  sorry

end smallest_solution_proof_l239_239838


namespace div_1_eq_17_div_2_eq_2_11_mul_1_eq_1_4_l239_239646

-- Define the values provided in the problem
def div_1 := (8 : ℚ) / (8 / 17 : ℚ)
def div_2 := (6 / 11 : ℚ) / 3
def mul_1 := (5 / 4 : ℚ) * (1 / 5 : ℚ)

-- Prove the equivalences
theorem div_1_eq_17 : div_1 = 17 := by
  sorry

theorem div_2_eq_2_11 : div_2 = 2 / 11 := by
  sorry

theorem mul_1_eq_1_4 : mul_1 = 1 / 4 := by
  sorry

end div_1_eq_17_div_2_eq_2_11_mul_1_eq_1_4_l239_239646


namespace square_division_l239_239989

theorem square_division (n k : ℕ) (m : ℕ) (h : n * k = m * m) :
  ∃ u v d : ℕ, (gcd u v = 1) ∧ (n = d * u * u) ∧ (k = d * v * v) ∧ (m = d * u * v) :=
by sorry

end square_division_l239_239989


namespace find_x_values_l239_239931

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l239_239931


namespace min_vertical_segment_length_l239_239598

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3
def L (x : ℝ) : ℝ := f x - g x

theorem min_vertical_segment_length : ∃ (x : ℝ), L x = 10 :=
by
  sorry

end min_vertical_segment_length_l239_239598


namespace find_a2_b2_c2_l239_239153

theorem find_a2_b2_c2 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 1) (h5 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 + 1) : 
  a^2 + b^2 + c^2 = 7 / 5 := 
sorry

end find_a2_b2_c2_l239_239153


namespace two_rel_prime_exists_l239_239868

theorem two_rel_prime_exists (A : Finset ℕ) (h1 : A.card = 2011) (h2 : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 4020) : 
  ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ Nat.gcd a b = 1 :=
by
  sorry

end two_rel_prime_exists_l239_239868


namespace max_distance_from_B_to_P_l239_239768

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -4, y := 1 }
def P : Point := { x := 3, y := -1 }

def line_l (m : ℝ) (pt : Point) : Prop :=
  (2 * m + 1) * pt.x - (m - 1) * pt.y - m - 5 = 0

theorem max_distance_from_B_to_P :
  ∃ B : Point, A = { x := -4, y := 1 } → 
               (∀ m : ℝ, line_l m B) →
               ∃ d, d = 5 + Real.sqrt 10 :=
sorry

end max_distance_from_B_to_P_l239_239768


namespace no_valid_placement_for_digits_on_45gon_l239_239908

theorem no_valid_placement_for_digits_on_45gon (f : Fin 45 → Fin 10) :
  ¬ ∀ (a b : Fin 10), a ≠ b → ∃ (i j : Fin 45), i ≠ j ∧ f i = a ∧ f j = b :=
by {
  sorry
}

end no_valid_placement_for_digits_on_45gon_l239_239908


namespace find_f_five_l239_239795

noncomputable def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

theorem find_f_five (y : ℝ) (h : f 2 y = 50) : f 5 y = 92 := by
  sorry

end find_f_five_l239_239795


namespace pure_imaginary_value_l239_239756

theorem pure_imaginary_value (a : ℝ) 
  (h1 : (a^2 - 3 * a + 2) = 0) 
  (h2 : (a - 2) ≠ 0) : a = 1 := sorry

end pure_imaginary_value_l239_239756


namespace average_of_a_b_l239_239858

theorem average_of_a_b (a b : ℚ) (h1 : b = 2 * a) (h2 : (4 + 6 + 8 + a + b) / 5 = 17) : (a + b) / 2 = 33.5 := 
by
  sorry

end average_of_a_b_l239_239858


namespace regular_polygon_sides_l239_239442

theorem regular_polygon_sides (ratio : ℕ) (interior exterior : ℕ) (sum_angles : ℕ) 
  (h1 : ratio = 5)
  (h2 : interior = 5 * exterior)
  (h3 : interior + exterior = sum_angles)
  (h4 : sum_angles = 180) : 

∃ (n : ℕ), n = 12 := 
by 
  sorry

end regular_polygon_sides_l239_239442


namespace range_of_m_l239_239572

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f y ≤ f x) (m : ℝ) (h : f (m-1) > f (2*m-1)) : 0 < m :=
by
  sorry

end range_of_m_l239_239572


namespace smallest_n_l239_239091

theorem smallest_n (n : ℕ) (hn : 0 < n) (h : 253 * n % 15 = 989 * n % 15) : n = 15 := by
  sorry

end smallest_n_l239_239091


namespace correct_factorization_l239_239591

theorem correct_factorization (x m n a : ℝ) : 
  (¬ (x^2 + 2 * x + 1 = x * (x + 2) + 1)) ∧
  (¬ (m^2 - 2 * m * n + n^2 = (m + n)^2)) ∧
  (¬ (-a^4 + 16 = -(a^2 + 4) * (a^2 - 4))) ∧
  (x^3 - 4 * x = x * (x + 2) * (x - 2)) :=
by
  sorry

end correct_factorization_l239_239591


namespace question1_effective_purification_16days_question2_min_mass_optimal_purification_l239_239904

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then x^2 / 16 + 2
else if x > 4 then (x + 14) / (2 * x - 2)
else 0

-- Effective Purification Conditions
def effective_purification (m : ℝ) (x : ℝ) : Prop := m * f x ≥ 4

-- Optimal Purification Conditions
def optimal_purification (m : ℝ) (x : ℝ) : Prop := 4 ≤ m * f x ∧ m * f x ≤ 10

-- Proof for Question 1
theorem question1_effective_purification_16days (x : ℝ) (hx : 0 < x ∧ x ≤ 16) :
  effective_purification 4 x :=
by sorry

-- Finding Minimum m for Optimal Purification within 7 days
theorem question2_min_mass_optimal_purification :
  ∃ m : ℝ, (16 / 7 ≤ m ∧ m ≤ 10 / 3) ∧ ∀ (x : ℝ), (0 < x ∧ x ≤ 7) → optimal_purification m x :=
by sorry

end question1_effective_purification_16days_question2_min_mass_optimal_purification_l239_239904


namespace element_in_set_l239_239399

open Set

theorem element_in_set : -7 ∈ ({1, -7} : Set ℤ) := by
  sorry

end element_in_set_l239_239399


namespace positive_integer_pair_solution_l239_239729

theorem positive_integer_pair_solution :
  ∃ a b : ℕ, (a > 0) ∧ (b > 0) ∧ 
    ¬ (7 ∣ (a * b * (a + b))) ∧ 
    (7^7 ∣ ((a + b)^7 - a^7 - b^7)) ∧ 
    (a, b) = (18, 1) :=
by {
  sorry
}

end positive_integer_pair_solution_l239_239729


namespace triangle_area_l239_239652

theorem triangle_area :
  ∀ (x y : ℝ), (3 * x + 2 * y = 12 ∧ x ≥ 0 ∧ y ≥ 0) →
  (1 / 2) * 4 * 6 = 12 := by
  sorry

end triangle_area_l239_239652


namespace markup_percentage_l239_239429

theorem markup_percentage 
  (CP : ℝ) (x : ℝ) (MP : ℝ) (SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (x / 100) * CP)
  (h3 : SP = MP - (10 / 100) * MP)
  (h4 : SP = CP + (35 / 100) * CP) :
  x = 50 :=
by sorry

end markup_percentage_l239_239429


namespace initial_average_runs_l239_239830

theorem initial_average_runs (A : ℝ) (h : 10 * A + 65 = 11 * (A + 3)) : A = 32 :=
  by sorry

end initial_average_runs_l239_239830


namespace parallelogram_area_example_l239_239732

noncomputable def area_parallelogram (A B C D : (ℝ × ℝ)) : ℝ := 
  0.5 * |(A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1)|

theorem parallelogram_area_example : 
  let A := (0, 0)
  let B := (20, 0)
  let C := (25, 7)
  let D := (5, 7)
  area_parallelogram A B C D = 140 := 
by
  sorry

end parallelogram_area_example_l239_239732


namespace integer_base10_from_bases_l239_239983

theorem integer_base10_from_bases (C D : ℕ) (hC : 0 ≤ C ∧ C ≤ 7) (hD : 0 ≤ D ∧ D ≤ 5)
    (h : 8 * C + D = 6 * D + C) : C = 0 ∧ D = 0 ∧ (8 * C + D = 0) := by
  sorry

end integer_base10_from_bases_l239_239983


namespace interest_rate_of_A_to_B_l239_239394

theorem interest_rate_of_A_to_B :
  ∀ (principal gain interest_B_to_C : ℝ), 
  principal = 3500 →
  gain = 525 →
  interest_B_to_C = 0.15 →
  (principal * interest_B_to_C * 3 - gain) = principal * (10 / 100) * 3 :=
by
  intros principal gain interest_B_to_C h_principal h_gain h_interest_B_to_C
  sorry

end interest_rate_of_A_to_B_l239_239394


namespace ratio_of_runs_l239_239516

theorem ratio_of_runs (A B C : ℕ) (h1 : B = C / 5) (h2 : A + B + C = 95) (h3 : C = 75) :
  A / B = 1 / 3 :=
by sorry

end ratio_of_runs_l239_239516


namespace min_tiles_needed_l239_239018

-- Definitions for the problem
def tile_width : ℕ := 3
def tile_height : ℕ := 4

def region_width_ft : ℕ := 2
def region_height_ft : ℕ := 5

def inches_in_foot : ℕ := 12

-- Conversion
def region_width_in := region_width_ft * inches_in_foot
def region_height_in := region_height_ft * inches_in_foot

-- Calculations
def region_area := region_width_in * region_height_in
def tile_area := tile_width * tile_height

-- Theorem statement
theorem min_tiles_needed : region_area / tile_area = 120 := 
  sorry

end min_tiles_needed_l239_239018


namespace pen_cost_difference_l239_239561

theorem pen_cost_difference :
  ∀ (P : ℕ), (P + 2 = 13) → (P - 2 = 9) :=
by
  intro P
  intro h
  sorry

end pen_cost_difference_l239_239561


namespace mod_17_residue_l239_239293

theorem mod_17_residue : (255 + 7 * 51 + 9 * 187 + 5 * 34) % 17 = 0 := 
  by sorry

end mod_17_residue_l239_239293


namespace range_of_a_for_three_distinct_zeros_l239_239750

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a_for_three_distinct_zeros : 
  ∀ a : ℝ, (∀ x y : ℝ, x ≠ y → f x a = 0 → f y a = 0 → (f (1:ℝ) a < 0 ∧ f (-1:ℝ) a > 0)) ↔ (-2 < a ∧ a < 2) := 
by
  sorry

end range_of_a_for_three_distinct_zeros_l239_239750


namespace volume_of_alcohol_correct_l239_239193

noncomputable def radius := 3 / 2 -- radius of the tank
noncomputable def total_height := 9 -- total height of the tank
noncomputable def full_solution_height := total_height / 3 -- height of the liquid when the tank is one-third full
noncomputable def volume := Real.pi * radius^2 * full_solution_height -- volume of liquid in the tank
noncomputable def alcohol_ratio := 1 / 6 -- ratio of alcohol to the total solution
noncomputable def volume_of_alcohol := volume * alcohol_ratio -- volume of alcohol in the tank

theorem volume_of_alcohol_correct : volume_of_alcohol = (9 / 8) * Real.pi :=
by
  -- Proof would go here
  sorry

end volume_of_alcohol_correct_l239_239193


namespace smallest_n_exists_l239_239567

theorem smallest_n_exists (n : ℕ) (h : n ≥ 4) :
  (∃ (S : Finset ℤ), S.card = n ∧
    (∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
        (a + b - c - d) % 20 = 0))
  ↔ n = 9 := sorry

end smallest_n_exists_l239_239567


namespace average_rainfall_correct_l239_239173

-- Definitions based on given conditions
def total_rainfall : ℚ := 420 -- inches
def days_in_august : ℕ := 31
def hours_in_a_day : ℕ := 24

-- Defining total hours in August
def total_hours_in_august : ℕ := days_in_august * hours_in_a_day

-- The average rainfall in inches per hour
def average_rainfall_per_hour : ℚ := total_rainfall / total_hours_in_august

-- The statement to prove
theorem average_rainfall_correct :
  average_rainfall_per_hour = 420 / 744 :=
by
  sorry

end average_rainfall_correct_l239_239173


namespace intersect_three_points_l239_239053

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * x

theorem intersect_three_points (a : ℝ) :
  (∃ (t1 t2 t3 : ℝ), t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    f t1 = g t1 a ∧ f t2 = g t2 a ∧ f t3 = g t3 a) ↔ 
  a ∈ Set.Ioo (2 / (7 * Real.pi)) (2 / (3 * Real.pi)) ∨ a = -2 / (5 * Real.pi) :=
sorry

end intersect_three_points_l239_239053


namespace max_possible_n_l239_239782

theorem max_possible_n :
  ∃ (n : ℕ), (n < 150) ∧ (∃ (k l : ℤ), n = 9 * k - 1 ∧ n = 6 * l - 5 ∧ n = 125) :=
by 
  sorry

end max_possible_n_l239_239782


namespace sales_tax_paid_l239_239010

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tax_free_cost : ℝ)

theorem sales_tax_paid (h_total : total_cost = 25) (h_rate : tax_rate = 0.10) (h_free : tax_free_cost = 21.7) :
  ∃ (X : ℝ), 21.7 + X + (0.10 * X) = 25 ∧ (0.10 * X = 0.3) := 
by
  sorry

end sales_tax_paid_l239_239010


namespace sum_of_three_squares_not_divisible_by_3_l239_239110

theorem sum_of_three_squares_not_divisible_by_3
    (N : ℕ) (n : ℕ) (a b c : ℤ) 
    (h1 : N = 9^n * (a^2 + b^2 + c^2))
    (h2 : ∃ (a1 b1 c1 : ℤ), a = 3 * a1 ∧ b = 3 * b1 ∧ c = 3 * c1) :
    ∃ (k m n : ℤ), N = k^2 + m^2 + n^2 ∧ (¬ (3 ∣ k ∧ 3 ∣ m ∧ 3 ∣ n)) :=
sorry

end sum_of_three_squares_not_divisible_by_3_l239_239110


namespace divisible_by_24_l239_239008

theorem divisible_by_24 (n : ℕ) (hn : n > 0) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) := 
by sorry

end divisible_by_24_l239_239008


namespace find_x_l239_239863

theorem find_x (x : ℝ) (h : x + 5 * 12 / (180 / 3) = 41) : x = 40 :=
sorry

end find_x_l239_239863


namespace gcd_lcm_product_24_60_l239_239845

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l239_239845


namespace one_less_than_neg_one_is_neg_two_l239_239988

theorem one_less_than_neg_one_is_neg_two : (-1 - 1 = -2) :=
by
  sorry

end one_less_than_neg_one_is_neg_two_l239_239988


namespace total_letters_in_all_names_l239_239025

theorem total_letters_in_all_names :
  let jonathan_first := 8
  let jonathan_surname := 10
  let younger_sister_first := 5
  let younger_sister_surname := 10
  let older_brother_first := 6
  let older_brother_surname := 10
  let youngest_sibling_first := 4
  let youngest_sibling_hyphenated_surname := 15
  jonathan_first + jonathan_surname + younger_sister_first + younger_sister_surname +
  older_brother_first + older_brother_surname + youngest_sibling_first + youngest_sibling_hyphenated_surname = 68 := by
  sorry

end total_letters_in_all_names_l239_239025


namespace range_of_a_l239_239405

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l239_239405


namespace number_of_slices_left_l239_239215

-- Conditions
def total_slices : ℕ := 8
def slices_given_to_joe_and_darcy : ℕ := total_slices / 2
def slices_given_to_carl : ℕ := total_slices / 4

-- Question: How many slices were left?
def slices_left : ℕ := total_slices - (slices_given_to_joe_and_darcy + slices_given_to_carl)

-- Proof statement to demonstrate that slices_left == 2
theorem number_of_slices_left : slices_left = 2 := by
  sorry

end number_of_slices_left_l239_239215


namespace cube_ratio_sum_l239_239082

theorem cube_ratio_sum (a b : ℝ) (h1 : |a| ≠ |b|) (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18 / 7 :=
by
  sorry

end cube_ratio_sum_l239_239082


namespace sin_double_angle_l239_239499

theorem sin_double_angle (x : ℝ) (h : Real.tan (π / 4 - x) = 2) : Real.sin (2 * x) = -3 / 5 :=
by
  sorry

end sin_double_angle_l239_239499


namespace inequality_and_equality_condition_l239_239370

variable (a b c t : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_and_equality_condition :
  abc * (a^t + b^t + c^t) ≥ a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ∧ 
  (abc * (a^t + b^t + c^t) = a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ↔ a = b ∧ b = c) :=
sorry

end inequality_and_equality_condition_l239_239370


namespace trees_to_plant_total_l239_239480

def trees_chopped_first_half := 200
def trees_chopped_second_half := 300
def trees_to_plant_per_tree_chopped := 3

theorem trees_to_plant_total : 
  (trees_chopped_first_half + trees_chopped_second_half) * trees_to_plant_per_tree_chopped = 1500 :=
by
  sorry

end trees_to_plant_total_l239_239480


namespace spongebob_price_l239_239898

variable (x : ℝ)

theorem spongebob_price (h : 30 * x + 12 * 1.5 = 78) : x = 2 :=
by
  -- Given condition: 30 * x + 12 * 1.5 = 78
  sorry

end spongebob_price_l239_239898


namespace flower_pots_on_path_count_l239_239569

theorem flower_pots_on_path_count (L d : ℕ) (hL : L = 15) (hd : d = 3) : 
  (L / d) + 1 = 6 :=
by
  sorry

end flower_pots_on_path_count_l239_239569


namespace solution_set_of_inequality_l239_239581

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end solution_set_of_inequality_l239_239581


namespace gcd_135_81_l239_239449

-- Define the numbers
def a : ℕ := 135
def b : ℕ := 81

-- State the goal: greatest common divisor of a and b is 27
theorem gcd_135_81 : Nat.gcd a b = 27 := by
  sorry

end gcd_135_81_l239_239449


namespace percent_women_surveryed_equal_40_l239_239337

theorem percent_women_surveryed_equal_40
  (W M : ℕ) 
  (h1 : W + M = 100)
  (h2 : (W / 100 * 1 / 10 : ℚ) + (M / 100 * 1 / 4 : ℚ) = (19 / 100 : ℚ))
  (h3 : (9 / 10 : ℚ) * (W / 100 : ℚ) + (3 / 4 : ℚ) * (M / 100 : ℚ) = (1 - 19 / 100 : ℚ)) :
  W = 40 := 
sorry

end percent_women_surveryed_equal_40_l239_239337


namespace min_value_when_a_equals_1_range_of_a_for_f_geq_a_l239_239175

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem min_value_when_a_equals_1 : 
  ∃ x, f x 1 = 1 :=
by
  sorry

theorem range_of_a_for_f_geq_a (a : ℝ) :
  (∀ x, x ≥ -1 → f x a ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end min_value_when_a_equals_1_range_of_a_for_f_geq_a_l239_239175


namespace nick_total_quarters_l239_239176

theorem nick_total_quarters (Q : ℕ)
  (h1 : 2 / 5 * Q = state_quarters)
  (h2 : 1 / 2 * state_quarters = PA_quarters)
  (h3 : PA_quarters = 7) :
  Q = 35 := by
  sorry

end nick_total_quarters_l239_239176


namespace black_and_white_films_l239_239341

theorem black_and_white_films (y x B : ℕ) 
  (h1 : ∀ B, B = 40 * x)
  (h2 : (4 * y : ℚ) / (((y / x : ℚ) * B / 100) + 4 * y) = 10 / 11) :
  B = 40 * x :=
by sorry

end black_and_white_films_l239_239341


namespace find_x_l239_239258

noncomputable def x : ℝ := 80 / 9

theorem find_x
  (hx_pos : 0 < x)
  (hx_condition : x * (⌊x⌋₊ : ℝ) = 80) :
  x = 80 / 9 :=
by
  sorry

end find_x_l239_239258


namespace multiple_of_people_l239_239780

-- Define the conditions
variable (P : ℕ) -- number of people who can do the work in 8 days

-- define a function that represents the work capacity of M * P people in days, 
-- we abstract away the solving steps into one declaration.

noncomputable def work_capacity (M P : ℕ) (days : ℕ) : ℚ :=
  M * (1/8) * days

-- Set up the problem to prove that the multiple of people is 2
theorem multiple_of_people (P : ℕ) : ∃ M : ℕ, work_capacity M P 2 = 1/2 :=
by
  use 2
  unfold work_capacity
  sorry

end multiple_of_people_l239_239780


namespace base5_division_l239_239867

theorem base5_division :
  ∀ (a b : ℕ), a = 1121 ∧ b = 12 → 
   ∃ (q r : ℕ), (a = b * q + r) ∧ (r < b) ∧ (q = 43) :=
by sorry

end base5_division_l239_239867


namespace prove_option_d_l239_239733

-- Definitions of conditions
variables (a b : ℝ)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (h_lt : a < b)

-- The theorem to be proved
theorem prove_option_d : a^3 < b^3 :=
sorry

end prove_option_d_l239_239733


namespace minimum_value_of_expression_l239_239276

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x * y / z + z * x / y + y * z / x) * (x / (y * z) + y / (z * x) + z / (x * y))

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  expression x y z ≥ 9 :=
sorry

end minimum_value_of_expression_l239_239276


namespace range_of_a_l239_239787

variable (a : ℝ)
def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a ∧ a > 0}

def p (a : ℝ) := 1 ∈ A a
def q (a : ℝ) := 2 ∈ A a

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : 1 < a ∧ a ≤ 2 := sorry

end range_of_a_l239_239787


namespace right_angled_triangle_other_angle_isosceles_triangle_base_angle_l239_239680

theorem right_angled_triangle_other_angle (a : ℝ) (h1 : 0 < a) (h2 : a < 90) (h3 : 40 = a) :
  50 = 90 - a :=
sorry

theorem isosceles_triangle_base_angle (v : ℝ) (h1 : 0 < v) (h2 : v < 180) (h3 : 80 = v) :
  50 = (180 - v) / 2 :=
sorry

end right_angled_triangle_other_angle_isosceles_triangle_base_angle_l239_239680


namespace range_of_b_l239_239721

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  - (1/2) * (x - 2)^2 + b * Real.log x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 1 < x → f x b ≤ f 1 b) → b ≤ -1 :=
by
  sorry

end range_of_b_l239_239721


namespace simplify_and_evaluate_l239_239063

theorem simplify_and_evaluate (x : ℝ) (h : x = 3 / 2) : 
  (2 + x) * (2 - x) + (x - 1) * (x + 5) = 5 := 
by
  sorry

end simplify_and_evaluate_l239_239063


namespace decimal_to_fraction_l239_239556

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l239_239556


namespace rainfall_on_tuesday_l239_239084

theorem rainfall_on_tuesday 
  (r_Mon r_Wed r_Total r_Tue : ℝ)
  (h_Mon : r_Mon = 0.16666666666666666)
  (h_Wed : r_Wed = 0.08333333333333333)
  (h_Total : r_Total = 0.6666666666666666)
  (h_Tue : r_Tue = r_Total - (r_Mon + r_Wed)) :
  r_Tue = 0.41666666666666663 := 
sorry

end rainfall_on_tuesday_l239_239084


namespace smallest_positive_integer_for_divisibility_l239_239685

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k, a = b * k

def smallest_n (n : ℕ) : Prop :=
  (is_divisible_by (n^2) 50) ∧ (is_divisible_by (n^3) 288) ∧ (∀ m : ℕ, m > 0 → m < n → ¬ (is_divisible_by (m^2) 50 ∧ is_divisible_by (m^3) 288))

theorem smallest_positive_integer_for_divisibility : smallest_n 60 :=
by
  sorry

end smallest_positive_integer_for_divisibility_l239_239685


namespace last_digit_7_powers_l239_239545

theorem last_digit_7_powers :
  (∃ n : ℕ, (∀ k < 4004, k.mod 2002 == n))
  := sorry

end last_digit_7_powers_l239_239545


namespace polynomial_system_solution_l239_239047

variable {x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ}

theorem polynomial_system_solution (
  h1 : x₁ + 3 * x₂ + 5 * x₃ + 7 * x₄ + 9 * x₅ + 11 * x₆ + 13 * x₇ = 3)
  (h2 : 3 * x₁ + 5 * x₂ + 7 * x₃ + 9 * x₄ + 11 * x₅ + 13 * x₆ + 15 * x₇ = 15)
  (h3 : 5 * x₁ + 7 * x₂ + 9 * x₃ + 11 * x₄ + 13 * x₅ + 15 * x₆ + 17 * x₇ = 85) :
  7 * x₁ + 9 * x₂ + 11 * x₃ + 13 * x₄ + 15 * x₅ + 17 * x₆ + 19 * x₇ = 213 :=
sorry

end polynomial_system_solution_l239_239047


namespace work_duration_l239_239876

theorem work_duration (p q r : ℕ) (Wp Wq Wr : ℕ) (t1 t2 : ℕ) (T : ℝ) :
  (Wp = 20) → (Wq = 12) → (Wr = 30) →
  (t1 = 4) → (t2 = 4) →
  (T = (t1 + t2 + (4/15 * Wr) / (1/(Wr) + 1/(Wq) + 1/(Wp)))) →
  T = 9.6 :=
by
  intros;
  sorry

end work_duration_l239_239876


namespace find_c_l239_239955

noncomputable def P (c : ℝ) (x : ℝ) : ℝ := x^3 - 3 * x^2 + c * x - 8

theorem find_c (c : ℝ) : (∀ x, P c (x + 2) = 0) → c = -14 :=
sorry

end find_c_l239_239955


namespace smallest_integer_greater_than_20_l239_239794

noncomputable def smallest_integer_greater_than_A : ℕ :=
  let a (n : ℕ) := 4 * n - 3
  let A := Real.sqrt (a 1580) - 1 / 4
  Nat.ceil A

theorem smallest_integer_greater_than_20 :
  smallest_integer_greater_than_A = 20 :=
sorry

end smallest_integer_greater_than_20_l239_239794


namespace chain_of_tangent_circles_iff_l239_239021

-- Define the circles, their centers, and the conditions
structure Circle := 
  (center : ℝ × ℝ) 
  (radius : ℝ)

structure TangentData :=
  (circle1 : Circle)
  (circle2 : Circle)
  (angle : ℝ)

-- Non-overlapping condition
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let dist := (x2 - x1)^2 + (y2 - y1)^2
  dist > (c1.radius + c2.radius)^2

-- Existence of tangent circles condition
def exists_chain_of_tangent_circles (c1 c2 : Circle) (n : ℕ) : Prop :=
  ∃ (tangent_circle : Circle), tangent_circle.radius = c1.radius ∨ tangent_circle.radius = c2.radius

-- Angle condition
def angle_condition (ang : ℝ) (n : ℕ) : Prop :=
  ∃ (k : ℤ), ang = k * (360 / n)

-- Final theorem to prove
theorem chain_of_tangent_circles_iff (c1 c2 : Circle) (t : TangentData) (n : ℕ) 
  (h1 : non_overlapping c1 c2) 
  (h2 : t.circle1 = c1 ∧ t.circle2 = c2) 
  : exists_chain_of_tangent_circles c1 c2 n ↔ angle_condition t.angle n := 
  sorry

end chain_of_tangent_circles_iff_l239_239021


namespace smallest_positive_cube_ends_in_112_l239_239324

theorem smallest_positive_cube_ends_in_112 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 112 ∧ (∀ m : ℕ, (m > 0 ∧ m^3 % 1000 = 112) → n ≤ m) :=
by
  sorry

end smallest_positive_cube_ends_in_112_l239_239324


namespace compute_expression_l239_239103

open Real

theorem compute_expression : 
  sqrt (1 / 4) * sqrt 16 - (sqrt (1 / 9))⁻¹ - sqrt 0 + sqrt (45 / 5) = 2 := 
by
  -- The proof details would go here, but they are omitted.
  sorry

end compute_expression_l239_239103


namespace mona_unique_players_l239_239630

-- Define the conditions
def groups (mona: String) : ℕ := 9
def players_per_group : ℕ := 4
def repeat_players_group1 : ℕ := 2
def repeat_players_group2 : ℕ := 1

-- Statement of the proof problem
theorem mona_unique_players
  (total_groups : ℕ := groups "Mona")
  (players_each_group : ℕ := players_per_group)
  (repeats_group1 : ℕ := repeat_players_group1)
  (repeats_group2 : ℕ := repeat_players_group2) :
  (total_groups * players_each_group) - (repeats_group1 + repeats_group2) = 33 := by
  sorry

end mona_unique_players_l239_239630


namespace find_prime_number_between_50_and_60_l239_239327

theorem find_prime_number_between_50_and_60 (n : ℕ) :
  (50 < n ∧ n < 60) ∧ Prime n ∧ n % 7 = 3 ↔ n = 59 :=
by
  sorry

end find_prime_number_between_50_and_60_l239_239327


namespace average_speed_calculation_l239_239419

def average_speed (s1 s2 t1 t2 : ℕ) : ℕ :=
  (s1 * t1 + s2 * t2) / (t1 + t2)

theorem average_speed_calculation :
  average_speed 40 60 1 3 = 55 :=
by
  -- skipping the proof
  sorry

end average_speed_calculation_l239_239419


namespace least_plates_to_ensure_matching_pair_l239_239884

theorem least_plates_to_ensure_matching_pair
  (white_plates : ℕ)
  (green_plates : ℕ)
  (red_plates : ℕ)
  (pink_plates : ℕ)
  (purple_plates : ℕ)
  (h_white : white_plates = 2)
  (h_green : green_plates = 6)
  (h_red : red_plates = 8)
  (h_pink : pink_plates = 4)
  (h_purple : purple_plates = 10) :
  ∃ n, n = 6 :=
by
  sorry

end least_plates_to_ensure_matching_pair_l239_239884


namespace sum_of_four_smallest_divisors_l239_239841

-- Define a natural number n and divisors d1, d2, d3, d4
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = k * d

-- Primary problem condition (sum of four divisors equals 2n)
def sum_of_divisors_eq (n d1 d2 d3 d4 : ℕ) : Prop := d1 + d2 + d3 + d4 = 2 * n

-- Assume the four divisors of n are distinct
def distinct (d1 d2 d3 d4 : ℕ) : Prop := d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

-- State the Lean proof problem
theorem sum_of_four_smallest_divisors (n d1 d2 d3 d4 : ℕ) (h1 : d1 < d2) (h2 : d2 < d3) (h3 : d3 < d4) 
    (h_div1 : is_divisor d1 n) (h_div2 : is_divisor d2 n) (h_div3 : is_divisor d3 n) (h_div4 : is_divisor d4 n)
    (h_sum : sum_of_divisors_eq n d1 d2 d3 d4) (h_distinct : distinct d1 d2 d3 d4) : 
    (d1 + d2 + d3 + d4 = 10 ∨ d1 + d2 + d3 + d4 = 11 ∨ d1 + d2 + d3 + d4 = 12) := 
sorry

end sum_of_four_smallest_divisors_l239_239841


namespace eccentricity_of_ellipse_l239_239850

theorem eccentricity_of_ellipse (k : ℝ) (h_k : k > 0)
  (focus : ∃ (x : ℝ), (x, 0) = ⟨3, 0⟩) :
  ∃ e : ℝ, e = (Real.sqrt 3 / 2) := 
sorry

end eccentricity_of_ellipse_l239_239850


namespace three_digit_number_div_by_11_l239_239772

theorem three_digit_number_div_by_11 (x y z n : ℕ) 
  (hx : 0 < x ∧ x < 10) 
  (hy : 0 ≤ y ∧ y < 10) 
  (hz : 0 ≤ z ∧ z < 10) 
  (hn : n = 100 * x + 10 * y + z) 
  (hq : (n / 11) = x + y + z) : 
  n = 198 :=
by
  sorry

end three_digit_number_div_by_11_l239_239772


namespace sum_of_digits_l239_239802

theorem sum_of_digits (A B C D : ℕ) (H1: A < B) (H2: B < C) (H3: C < D)
  (H4: A > 0) (H5: B > 0) (H6: C > 0) (H7: D > 0)
  (H8: 1000 * A + 100 * B + 10 * C + D + 1000 * D + 100 * C + 10 * B + A = 11990) : 
  (A, B, C, D) = (1, 9, 9, 9) :=
sorry

end sum_of_digits_l239_239802


namespace leo_total_points_l239_239121

theorem leo_total_points (x y : ℕ) (h1 : x + y = 50) :
  0.4 * (x : ℝ) * 3 + 0.5 * (y : ℝ) * 2 = 0.2 * (x : ℝ) + 50 :=
by sorry

end leo_total_points_l239_239121


namespace definite_integral_sin_cos_l239_239645

open Real

theorem definite_integral_sin_cos :
  ∫ x in - (π / 2)..(π / 2), (sin x + cos x) = 2 :=
sorry

end definite_integral_sin_cos_l239_239645


namespace smallest_number_is_C_l239_239640

def A : ℕ := 36
def B : ℕ := 27 + 5
def C : ℕ := 3 * 10
def D : ℕ := 40 - 3

theorem smallest_number_is_C :
  min (min A B) (min C D) = C :=
by
  -- Proof steps go here
  sorry

end smallest_number_is_C_l239_239640


namespace average_speed_of_car_l239_239766

noncomputable def averageSpeed : ℚ := 
  let speed1 := 45     -- kph
  let distance1 := 15  -- km
  let speed2 := 55     -- kph
  let distance2 := 30  -- km
  let speed3 := 65     -- kph
  let time3 := 35 / 60 -- hours
  let speed4 := 52     -- kph
  let time4 := 20 / 60 -- hours
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  let totalDistance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let totalTime := time1 + time2 + time3 + time4
  totalDistance / totalTime

theorem average_speed_of_car :
  abs (averageSpeed - 55.85) < 0.01 := 
  sorry

end average_speed_of_car_l239_239766


namespace problem_l239_239642

noncomputable def f (A B x : ℝ) : ℝ := A * x^2 + B
noncomputable def g (A B x : ℝ) : ℝ := B * x^2 + A

theorem problem (A B x : ℝ) (h : A ≠ B) 
  (h1 : f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = 0 := 
  sorry

end problem_l239_239642


namespace work_completion_days_l239_239257

theorem work_completion_days (A_time : ℝ) (A_efficiency : ℝ) (B_time : ℝ) (B_efficiency : ℝ) (C_time : ℝ) (C_efficiency : ℝ) :
  A_time = 60 → A_efficiency = 1.5 → B_time = 20 → B_efficiency = 1 → C_time = 30 → C_efficiency = 0.75 → 
  (1 / (A_efficiency / A_time + B_efficiency / B_time + C_efficiency / C_time)) = 10 := 
by
  intros A_time_eq A_efficiency_eq B_time_eq B_efficiency_eq C_time_eq C_efficiency_eq
  rw [A_time_eq, A_efficiency_eq, B_time_eq, B_efficiency_eq, C_time_eq, C_efficiency_eq]
  -- Proof omitted
  sorry

end work_completion_days_l239_239257


namespace ratio_of_larger_to_smaller_is_sqrt_six_l239_239087

def sum_of_squares_eq_seven_times_difference (a b : ℝ) : Prop := 
  a^2 + b^2 = 7 * (a - b)

theorem ratio_of_larger_to_smaller_is_sqrt_six {a b : ℝ} (h : sum_of_squares_eq_seven_times_difference a b) (h1 : a > b) : 
  a / b = Real.sqrt 6 :=
sorry

end ratio_of_larger_to_smaller_is_sqrt_six_l239_239087


namespace min_value_expression_l239_239115

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_expression : (1 + b / a) * (4 * a / b) ≥ 9 :=
sorry

end min_value_expression_l239_239115


namespace length_of_AE_l239_239342

variable (A B C D E : Type) [AddGroup A]
variable (AB CD AC AE EC : ℝ)
variable 
  (hAB : AB = 8)
  (hCD : CD = 18)
  (hAC : AC = 20)
  (hEqualAreas : ∀ (AED BEC : Type), (area AED = area BEC) → (AED = BEC))

theorem length_of_AE (hRatio : AE / EC = 4 / 9) (hSum : AC = AE + EC) : AE = 80 / 13 :=
by
  sorry

end length_of_AE_l239_239342


namespace problem_statement_l239_239557

theorem problem_statement (f : ℝ → ℝ) (hf_odd : ∀ x, f (-x) = - f x)
  (hf_deriv : ∀ x < 0, 2 * f x + x * deriv f x < 0) :
  f 1 < 2016 * f (Real.sqrt 2016) ∧ 2016 * f (Real.sqrt 2016) < 2017 * f (Real.sqrt 2017) := 
  sorry

end problem_statement_l239_239557


namespace invest_in_yourself_examples_l239_239459

theorem invest_in_yourself_examples (example1 example2 example3 : String)
  (benefit1 benefit2 benefit3 : String)
  (h1 : example1 = "Investment in Education")
  (h2 : benefit1 = "Spending money on education improves knowledge and skills, leading to better job opportunities and higher salaries. Education appreciates over time, providing financial stability.")
  (h3 : example2 = "Investment in Physical Health")
  (h4 : benefit2 = "Spending on sports activities, fitness programs, or healthcare prevents chronic diseases, saves future medical expenses, and enhances overall well-being.")
  (h5 : example3 = "Time Spent on Reading Books")
  (h6 : benefit3 = "Reading books expands knowledge, improves vocabulary and cognitive abilities, develops critical thinking and analytical skills, and fosters creativity and empathy."):
  "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." = "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." :=
by
  sorry

end invest_in_yourself_examples_l239_239459


namespace no_nonnegative_integral_solutions_l239_239317

theorem no_nonnegative_integral_solutions :
  ¬ ∃ (x y : ℕ), (x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0) ∧ (x + y = 10) :=
by
  sorry

end no_nonnegative_integral_solutions_l239_239317


namespace bogan_maggots_l239_239657

theorem bogan_maggots (x : ℕ) (total_maggots : ℕ) (eaten_first : ℕ) (eaten_second : ℕ) (thrown_out : ℕ) 
  (h1 : eaten_first = 1) (h2 : eaten_second = 3) (h3 : total_maggots = 20) (h4 : thrown_out = total_maggots - eaten_first - eaten_second) 
  (h5 : x + eaten_first = thrown_out) : x = 15 :=
by
  -- Use the given conditions
  sorry

end bogan_maggots_l239_239657


namespace hamza_bucket_problem_l239_239605

-- Definitions reflecting the problem conditions
def bucket_2_5_capacity : ℝ := 2.5
def bucket_3_0_capacity : ℝ := 3.0
def bucket_5_6_capacity : ℝ := 5.6
def bucket_6_5_capacity : ℝ := 6.5

def initial_fill_in_5_6 : ℝ := bucket_5_6_capacity
def pour_5_6_to_3_0_remaining : ℝ := 5.6 - 3.0
def remaining_in_5_6_after_second_fill : ℝ := bucket_5_6_capacity - 0.5

-- Main problem statement
theorem hamza_bucket_problem : (bucket_6_5_capacity - 2.6 = 3.9) :=
by sorry

end hamza_bucket_problem_l239_239605


namespace shaded_region_area_l239_239829

theorem shaded_region_area (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) (T : ℝ):
  d = 3 → L = 24 → L = n * d → n * 2 = 16 → r = d / 2 → 
  A = (1 / 2) * π * r ^ 2 → T = 16 * A → T = 18 * π :=
  by
  intros d_eq L_eq Ln_eq semicircle_count r_eq A_eq T_eq_total
  sorry

end shaded_region_area_l239_239829


namespace four_digit_integer_existence_l239_239810

theorem four_digit_integer_existence :
  ∃ (a b c d : ℕ), 
    (1000 * a + 100 * b + 10 * c + d = 4522) ∧
    (a + b + c + d = 16) ∧
    (b + c = 10) ∧
    (a - d = 3) ∧
    (1000 * a + 100 * b + 10 * c + d) % 9 = 0 :=
by sorry

end four_digit_integer_existence_l239_239810


namespace solve_inequality_l239_239086

theorem solve_inequality (x : ℝ) : x^3 - 9*x^2 - 16*x > 0 ↔ (x < -1 ∨ x > 16) := by
  sorry

end solve_inequality_l239_239086


namespace movement_of_hands_of_clock_involves_rotation_l239_239791

theorem movement_of_hands_of_clock_involves_rotation (A B C D : Prop) :
  (A ↔ (∃ p : ℝ, ∃ θ : ℝ, p ≠ θ)) → -- A condition: exists a fixed point and rotation around it
  (B ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- B condition: does not rotate around a fixed point
  (C ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- C condition: does not rotate around a fixed point
  (D ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- D condition: does not rotate around a fixed point
  A :=
by
  intros hA hB hC hD
  sorry

end movement_of_hands_of_clock_involves_rotation_l239_239791


namespace power_multiplication_equals_result_l239_239221

theorem power_multiplication_equals_result : 
  (-0.25)^11 * (-4)^12 = -4 := 
sorry

end power_multiplication_equals_result_l239_239221


namespace common_ratio_of_geometric_sequence_l239_239127

theorem common_ratio_of_geometric_sequence (a₁ : ℝ) (S : ℕ → ℝ) (q : ℝ) (h₁ : ∀ n, S (n + 1) = S n + a₁ * q ^ n) (h₂ : 2 * S n = S (n + 1) + S (n + 2)) :
  q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l239_239127


namespace seconds_hand_revolution_l239_239695

theorem seconds_hand_revolution (revTimeSeconds revTimeMinutes : ℕ) : 
  (revTimeSeconds = 60) ∧ (revTimeMinutes = 1) :=
sorry

end seconds_hand_revolution_l239_239695


namespace height_of_stack_correct_l239_239669

namespace PaperStack

-- Define the problem conditions
def sheets_per_package : ℕ := 500
def thickness_per_sheet_mm : ℝ := 0.1
def packages_per_stack : ℕ := 60
def mm_to_m : ℝ := 1000.0

-- Statement: the height of the stack of 60 paper packages
theorem height_of_stack_correct :
  (sheets_per_package * thickness_per_sheet_mm * packages_per_stack) / mm_to_m = 3 :=
sorry

end PaperStack

end height_of_stack_correct_l239_239669


namespace min_value_of_function_l239_239192

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x^2

theorem min_value_of_function :
  ∀ x > 0, f x ≥ 9 :=
by
  intro x hx_pos
  sorry

end min_value_of_function_l239_239192


namespace find_a6_l239_239982

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem find_a6 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h2 : a 2 = 4) (h4 : a 4 = 2) : a 6 = 0 :=
by sorry

end find_a6_l239_239982


namespace probability_three_fair_coins_l239_239129

noncomputable def probability_one_head_two_tails (n : ℕ) : ℚ :=
  if n = 3 then 3 / 8 else 0

theorem probability_three_fair_coins :
  probability_one_head_two_tails 3 = 3 / 8 :=
by
  sorry

end probability_three_fair_coins_l239_239129


namespace equal_division_of_balls_l239_239179

def total_balls : ℕ := 10
def num_boxes : ℕ := 5
def balls_per_box : ℕ := total_balls / num_boxes

theorem equal_division_of_balls :
  balls_per_box = 2 :=
by
  sorry

end equal_division_of_balls_l239_239179


namespace oliver_total_money_l239_239137

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end oliver_total_money_l239_239137


namespace consumption_increase_l239_239369

variable (T C C' : ℝ)
variable (h1 : 0.8 * T * C' = 0.92 * T * C)

theorem consumption_increase (T C C' : ℝ) (h1 : 0.8 * T * C' = 0.92 * T * C) : C' = 1.15 * C :=
by
  sorry

end consumption_increase_l239_239369


namespace type_2004_A_least_N_type_B_diff_2004_l239_239736

def game_type_A (N : ℕ) : Prop :=
  ∀ n, (1 ≤ n ∧ n ≤ N) → (n % 2 = 0 → false) 

def game_type_B (N : ℕ) : Prop :=
  ∃ n, (1 ≤ n ∧ n ≤ N) ∧ (n % 2 = 0 → true)


theorem type_2004_A : game_type_A 2004 :=
sorry

theorem least_N_type_B_diff_2004 : ∀ N, N > 2004 → game_type_B N → N = 2048 :=
sorry

end type_2004_A_least_N_type_B_diff_2004_l239_239736


namespace average_waiting_time_l239_239228

/-- 
A traffic light at a pedestrian crossing allows pedestrians to cross the street 
for one minute and prohibits crossing for two minutes. Prove that the average 
waiting time for a pedestrian who arrives at the intersection is 40 seconds.
-/ 
theorem average_waiting_time (pG : ℝ) (pR : ℝ) (eTG : ℝ) (eTR : ℝ) (cycle : ℝ) :
  pG = 1 / 3 ∧ pR = 2 / 3 ∧ eTG = 0 ∧ eTR = 1 ∧ cycle = 3 → 
  (eTG * pG + eTR * pR) * (60 / cycle) = 40 :=
by
  sorry

end average_waiting_time_l239_239228


namespace find_smallest_n_l239_239715

-- Definitions of the condition that m and n are relatively prime and that the fraction includes the digits 4, 5, and 6 consecutively
def is_coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

def has_digits_456 (m n : ℕ) : Prop := 
  ∃ k : ℕ, ∃ c : ℕ, 10^k * m % (10^k * n) = 456 * 10^c

-- The theorem to prove the smallest value of n
theorem find_smallest_n (m n : ℕ) (h1 : is_coprime m n) (h2 : m < n) (h3 : has_digits_456 m n) : n = 230 :=
sorry

end find_smallest_n_l239_239715


namespace sin2θ_value_l239_239014

theorem sin2θ_value (θ : Real) (h1 : Real.sin θ = 4/5) (h2 : Real.sin θ - Real.cos θ > 1) : Real.sin (2*θ) = -24/25 := 
by 
  sorry

end sin2θ_value_l239_239014


namespace find_numbers_l239_239682

theorem find_numbers (x y : ℤ) (h_sum : x + y = 40) (h_diff : x - y = 12) : x = 26 ∧ y = 14 :=
sorry

end find_numbers_l239_239682


namespace max_marks_mike_could_have_got_l239_239299

theorem max_marks_mike_could_have_got (p : ℝ) (m_s : ℝ) (d : ℝ) (M : ℝ) :
  p = 0.30 → m_s = 212 → d = 13 → 0.30 * M = (212 + 13) → M = 750 :=
by
  intros hp hms hd heq
  sorry

end max_marks_mike_could_have_got_l239_239299


namespace positive_polynomial_l239_239155

theorem positive_polynomial (x : ℝ) : 3 * x ^ 2 - 6 * x + 3.5 > 0 := 
by sorry

end positive_polynomial_l239_239155


namespace subcommittee_count_l239_239098

theorem subcommittee_count :
  let total_members := 12
  let total_teachers := 5
  let subcommittee_size := 5
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let non_teacher_subcommittees_with_0_teachers := Nat.choose (total_members - total_teachers) subcommittee_size
  let non_teacher_subcommittees_with_1_teacher :=
    Nat.choose total_teachers 1 * Nat.choose (total_members - total_teachers) (subcommittee_size - 1)
  (total_subcommittees
   - (non_teacher_subcommittees_with_0_teachers + non_teacher_subcommittees_with_1_teacher)) = 596 := 
by
  sorry

end subcommittee_count_l239_239098


namespace arithmetic_geometric_properties_l239_239269

noncomputable def arithmetic_seq (a₁ a₂ a₃ : ℝ) :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

noncomputable def geometric_seq (b₁ b₂ b₃ : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ b₂ = b₁ * q ∧ b₃ = b₂ * q

theorem arithmetic_geometric_properties (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) :
  arithmetic_seq a₁ a₂ a₃ →
  geometric_seq b₁ b₂ b₃ →
  ¬(a₁ < a₂ ∧ a₂ > a₃) ∧
  (b₁ < b₂ ∧ b₂ > b₃) ∧
  (a₁ + a₂ < 0 → ¬(a₂ + a₃ < 0)) ∧
  (b₁ * b₂ < 0 → b₂ * b₃ < 0) :=
by
  sorry

end arithmetic_geometric_properties_l239_239269


namespace Lilith_caps_collection_l239_239562

theorem Lilith_caps_collection
  (caps_per_month_first_year : ℕ)
  (caps_per_month_after_first_year : ℕ)
  (caps_received_each_christmas : ℕ)
  (caps_lost_per_year : ℕ)
  (total_caps_collected : ℕ)
  (first_year_caps : ℕ := caps_per_month_first_year * 12)
  (years_after_first_year : ℕ)
  (total_years : ℕ := years_after_first_year + 1)
  (caps_collected_after_first_year : ℕ := caps_per_month_after_first_year * 12 * years_after_first_year)
  (caps_received_total : ℕ := caps_received_each_christmas * total_years)
  (caps_lost_total : ℕ := caps_lost_per_year * total_years)
  (total_calculated_caps : ℕ := first_year_caps + caps_collected_after_first_year + caps_received_total - caps_lost_total) :
  total_caps_collected = 401 → total_years = 5 :=
by
  sorry

end Lilith_caps_collection_l239_239562


namespace telephone_charge_l239_239426

theorem telephone_charge (x : ℝ) (h1 : ∀ t : ℝ, t = 18.70 → x + 39 * 0.40 = t) : x = 3.10 :=
by
  sorry

end telephone_charge_l239_239426


namespace probability_N_lt_L_is_zero_l239_239570

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l239_239570


namespace find_quotient_l239_239117

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![4, 5]]

noncomputable def matrix_b (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![c, d]]

theorem find_quotient (a b c d : ℝ) (H1 : matrix_a * (matrix_b a b c d) = (matrix_b a b c d) * matrix_a)
  (H2 : 2*b ≠ 3*c) : ((a - d) / (c - 2*b)) = 3 / 2 :=
  sorry

end find_quotient_l239_239117


namespace logs_needed_l239_239491

theorem logs_needed (needed_woodblocks : ℕ) (current_logs : ℕ) (woodblocks_per_log : ℕ) 
  (H1 : needed_woodblocks = 80) 
  (H2 : current_logs = 8) 
  (H3 : woodblocks_per_log = 5) : 
  current_logs * woodblocks_per_log < needed_woodblocks → 
  (needed_woodblocks - current_logs * woodblocks_per_log) / woodblocks_per_log = 8 := by
  sorry

end logs_needed_l239_239491


namespace compute_product_l239_239255

-- Define the conditions
variables {x y : ℝ} (h1 : x - y = 5) (h2 : x^3 - y^3 = 35)

-- Define the theorem to be proved
theorem compute_product (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 190 / 9 := 
sorry

end compute_product_l239_239255


namespace cos_theta_of_triangle_median_l239_239348

theorem cos_theta_of_triangle_median
  (A : ℝ) (a : ℝ) (m : ℝ) (theta : ℝ)
  (area_eq : A = 24)
  (side_eq : a = 12)
  (median_eq : m = 5)
  (area_formula : A = (1/2) * a * m * Real.sin theta) :
  Real.cos theta = 3 / 5 := 
by 
  sorry

end cos_theta_of_triangle_median_l239_239348


namespace find_angle_B_find_area_of_ABC_l239_239496

noncomputable def angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) : ℝ := 
  if b * Real.cos C = -a then Real.pi - 2 * Real.arctan (a / c)
  else 2 * Real.pi / 3

theorem find_angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) :
  angle_B a b c C h1 = 2 * Real.pi / 3 := 
sorry

noncomputable def area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) : ℝ :=
  if position = 1 then /- calculation for BD bisector case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)
  else /- calculation for midpoint case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)

theorem find_area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) (hB : angle_B a b c C h1 = 2 * Real.pi / 3) :
  area_of_ABC a b c C (2 * Real.pi / 3) d position h1 h2 h3 = Real.sqrt 3 := 
sorry

end find_angle_B_find_area_of_ABC_l239_239496


namespace prod_of_real_roots_equation_l239_239385

theorem prod_of_real_roots_equation :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0 ∨ x = -(4 / 7) → (0 * (-(4 / 7)) = 0) :=
by sorry

end prod_of_real_roots_equation_l239_239385


namespace square_roots_equal_49_l239_239404

theorem square_roots_equal_49 (x a : ℝ) (hx1 : (2 * x - 3)^2 = a) (hx2 : (5 - x)^2 = a) (ha_pos: a > 0) : a = 49 := 
by 
  sorry

end square_roots_equal_49_l239_239404


namespace find_quotient_l239_239448

-- Definitions based on given conditions
def remainder : ℕ := 8
def dividend : ℕ := 997
def divisor : ℕ := 23

-- Hypothesis based on the division formula
def quotient_formula (q : ℕ) : Prop :=
  dividend = (divisor * q) + remainder

-- Statement of the problem
theorem find_quotient (q : ℕ) (h : quotient_formula q) : q = 43 :=
sorry

end find_quotient_l239_239448


namespace find_p_q_r_sum_l239_239125

noncomputable def Q (p q r : ℝ) (v : ℂ) : Polynomial ℂ :=
  (Polynomial.C v + 2 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C v + 8 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C (3 * v - 5)).comp Polynomial.X

theorem find_p_q_r_sum (p q r : ℝ) (v : ℂ)
  (h_roots : ∃ v : ℂ, Polynomial.roots (Q p q r v) = {v + 2 * Complex.I, v + 8 * Complex.I, 3 * v - 5}) :
  (p + q + r) = -82 :=
by
  sorry

end find_p_q_r_sum_l239_239125


namespace total_slices_l239_239379

theorem total_slices {slices_per_pizza pizzas : ℕ} (h1 : slices_per_pizza = 2) (h2 : pizzas = 14) : 
  slices_per_pizza * pizzas = 28 :=
by
  -- This is where the proof would go, but we are omitting it as instructed.
  sorry

end total_slices_l239_239379


namespace factor_polynomial_l239_239058

noncomputable def gcd_coeffs : ℕ := Nat.gcd 72 180

theorem factor_polynomial (x : ℝ) (GCD_72_180 : gcd_coeffs = 36)
    (GCD_x5_x9 : ∃ (y: ℝ), x^5 = y ∧ x^9 = y * x^4) :
    72 * x^5 - 180 * x^9 = -36 * x^5 * (5 * x^4 - 2) :=
by
  sorry

end factor_polynomial_l239_239058


namespace find_share_of_A_l239_239811

noncomputable def investment_share_A (initial_investment_A initial_investment_B withdraw_A add_B after_months end_of_year_profit : ℝ) : ℝ :=
  let investment_months_A := (initial_investment_A * after_months) + ((initial_investment_A - withdraw_A) * (12 - after_months))
  let investment_months_B := (initial_investment_B * after_months) + ((initial_investment_B + add_B) * (12 - after_months))
  let total_investment_months := investment_months_A + investment_months_B
  let ratio_A := investment_months_A / total_investment_months
  ratio_A * end_of_year_profit

theorem find_share_of_A : 
  investment_share_A 3000 4000 1000 1000 8 630 = 240 := 
by 
  sorry

end find_share_of_A_l239_239811


namespace gcd_yz_min_value_l239_239308

theorem gcd_yz_min_value (x y z : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) 
  (hxy_gcd : Nat.gcd x y = 224) (hxz_gcd : Nat.gcd x z = 546) : 
  Nat.gcd y z = 14 := 
sorry

end gcd_yz_min_value_l239_239308


namespace cans_in_each_package_of_cat_food_l239_239064

-- Definitions and conditions
def cans_per_package_cat (c : ℕ) := 9 * c
def cans_per_package_dog := 7 * 5
def extra_cans_cat := 55

-- Theorem stating the problem and the answer
theorem cans_in_each_package_of_cat_food (c : ℕ) (h: cans_per_package_cat c = cans_per_package_dog + extra_cans_cat) :
  c = 10 :=
sorry

end cans_in_each_package_of_cat_food_l239_239064


namespace range_of_a_l239_239852

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1) ^ 2 > 4 → x > a) → a ≥ 1 := sorry

end range_of_a_l239_239852


namespace alex_correct_percentage_l239_239271

theorem alex_correct_percentage (y : ℝ) (hy_pos : y > 0) : 
  (5 / 7) * 100 = 71.43 := 
by
  sorry

end alex_correct_percentage_l239_239271


namespace function_monotonic_decreasing_interval_l239_239758

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem function_monotonic_decreasing_interval :
  ∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  ∀ y ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  (x ≤ y → f y ≤ f x) :=
by
  sorry

end function_monotonic_decreasing_interval_l239_239758


namespace identify_conic_section_hyperbola_l239_239015

-- Defining the variables and constants in the Lean environment
variable (x y : ℝ)

-- The given equation in function form
def conic_section_eq : Prop := (x - 3) ^ 2 = 4 * (y + 2) ^ 2 + 25

-- The expected type of conic section (Hyperbola)
def is_hyperbola : Prop := 
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^2 - b * y^2 + c * x + d * y + e = f

-- The theorem statement to prove
theorem identify_conic_section_hyperbola (h : conic_section_eq x y) : is_hyperbola x y := by
  sorry

end identify_conic_section_hyperbola_l239_239015


namespace equivalent_statement_l239_239028

variable (R G : Prop)

theorem equivalent_statement (h : ¬ R → ¬ G) : G → R := by
  intro hG
  by_contra hR
  exact h hR hG

end equivalent_statement_l239_239028


namespace sweater_markup_percentage_l239_239057

-- The wholesale cost W and retail price R
variables (W R : ℝ)

-- The given condition
variable (h : 0.30 * R = 1.40 * W)

-- The theorem to prove
theorem sweater_markup_percentage (h : 0.30 * R = 1.40 * W) : (R - W) / W * 100 = 366.67 :=
by
  -- The solution steps would be placed here, if we were proving.
  sorry

end sweater_markup_percentage_l239_239057


namespace factorization_correct_l239_239797

theorem factorization_correct {x : ℝ} : (x - 15)^2 = x^2 - 30*x + 225 :=
by
  sorry

end factorization_correct_l239_239797


namespace range_of_positive_integers_in_list_H_l239_239268

noncomputable def list_H_lower_bound : Int := -15
noncomputable def list_H_length : Nat := 30

theorem range_of_positive_integers_in_list_H :
  ∃(r : Nat), list_H_lower_bound + list_H_length - 1 = 14 ∧ r = 14 - 1 := 
by
  let upper_bound := list_H_lower_bound + Int.ofNat list_H_length - 1
  use (upper_bound - 1).toNat
  sorry

end range_of_positive_integers_in_list_H_l239_239268


namespace heptagon_triangulation_count_l239_239407

/-- The number of ways to divide a regular heptagon (7-sided polygon) 
    into 5 triangles using non-intersecting diagonals is 4. -/
theorem heptagon_triangulation_count : ∃ (n : ℕ), n = 4 ∧ ∀ (p : ℕ), (p = 7 ∧ (∀ (k : ℕ), k = 5 → (n = 4))) :=
by {
  -- The proof is non-trivial and omitted here
  sorry
}

end heptagon_triangulation_count_l239_239407


namespace maximum_value_frac_l239_239558

-- Let x and y be positive real numbers. Prove that (x + y)^3 / (x^3 + y^3) ≤ 4.
theorem maximum_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^3 / (x^3 + y^3) ≤ 4 := sorry

end maximum_value_frac_l239_239558


namespace total_pages_in_book_l239_239759

-- Define the given conditions
def chapters : Nat := 41
def days : Nat := 30
def pages_per_day : Nat := 15

-- Define the statement to be proven
theorem total_pages_in_book : (days * pages_per_day) = 450 := by
  sorry

end total_pages_in_book_l239_239759


namespace parabola_expression_l239_239897

theorem parabola_expression (a c : ℝ) (h1 : a = 1/4 ∨ a = -1/4) (h2 : ∀ x : ℝ, x = 1 → (a * x^2 + c = 0)) :
  (a = 1/4 ∧ c = -1/4) ∨ (a = -1/4 ∧ c = 1/4) :=
by {
  sorry
}

end parabola_expression_l239_239897


namespace total_chapters_eq_l239_239770

-- Definitions based on conditions
def days : ℕ := 664
def chapters_per_day : ℕ := 332

-- Theorem to prove the total number of chapters in the book is 220448
theorem total_chapters_eq : (chapters_per_day * days = 220448) :=
by
  sorry

end total_chapters_eq_l239_239770


namespace allocation_methods_count_l239_239494

def number_of_allocation_methods (doctors nurses : ℕ) (hospitals : ℕ) (nurseA nurseB : ℕ) :=
  if (doctors = 3) ∧ (nurses = 6) ∧ (hospitals = 3) ∧ (nurseA = 1) ∧ (nurseB = 1) then 684 else 0

theorem allocation_methods_count :
  number_of_allocation_methods 3 6 3 2 2 = 684 :=
by
  sorry

end allocation_methods_count_l239_239494


namespace cat_mouse_position_after_299_moves_l239_239522

-- Definitions based on conditions
def cat_position (move : Nat) : Nat :=
  let active_moves := move - (move / 100)
  active_moves % 4

def mouse_position (move : Nat) : Nat :=
  move % 8

-- Main theorem
theorem cat_mouse_position_after_299_moves :
  cat_position 299 = 0 ∧ mouse_position 299 = 3 :=
by
  sorry

end cat_mouse_position_after_299_moves_l239_239522


namespace find_integer_pairs_l239_239836

theorem find_integer_pairs (a b : ℤ) (h₁ : 1 < a) (h₂ : 1 < b) 
    (h₃ : a ∣ (b + 1)) (h₄ : b ∣ (a^3 - 1)) : 
    ∃ (s : ℤ), (s ≥ 2 ∧ (a, b) = (s, s^3 - 1)) ∨ (s ≥ 3 ∧ (a, b) = (s, s - 1)) :=
  sorry

end find_integer_pairs_l239_239836


namespace solve_for_n_l239_239958

theorem solve_for_n (n : ℕ) (h : (16^n) * (16^n) * (16^n) * (16^n) * (16^n) = 256^5) : n = 2 := by
  sorry

end solve_for_n_l239_239958


namespace smallest_feared_sequence_l239_239143

def is_feared (n : ℕ) : Prop :=
  -- This function checks if a number contains '13' as a contiguous substring.
  sorry

def is_fearless (n : ℕ) : Prop := ¬is_feared n

theorem smallest_feared_sequence : ∃ (n : ℕ) (a : ℕ), 0 < n ∧ a < 100 ∧ is_fearless n ∧ is_fearless (n + 10 * a) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → is_feared (n + k * a)) ∧ n = 1287 := 
by
  sorry

end smallest_feared_sequence_l239_239143


namespace cost_per_can_of_tuna_l239_239132

theorem cost_per_can_of_tuna
  (num_cans : ℕ) -- condition 1
  (num_coupons : ℕ) -- condition 2
  (coupon_discount_cents : ℕ) -- condition 2 detail
  (amount_paid_dollars : ℚ) -- condition 3
  (change_received_dollars : ℚ) -- condition 3 detail
  (cost_per_can_cents: ℚ) : -- the quantity we want to prove
  num_cans = 9 →
  num_coupons = 5 →
  coupon_discount_cents = 25 →
  amount_paid_dollars = 20 →
  change_received_dollars = 5.5 →
  cost_per_can_cents = 175 :=
by
  intros hn hc hcd hap hcr
  sorry

end cost_per_can_of_tuna_l239_239132


namespace notebook_problem_l239_239901

theorem notebook_problem
    (total_notebooks : ℕ)
    (cost_price_A : ℕ)
    (cost_price_B : ℕ)
    (total_cost_price : ℕ)
    (selling_price_A : ℕ)
    (selling_price_B : ℕ)
    (discount_A : ℕ)
    (profit_condition : ℕ)
    (x y m : ℕ) 
    (h1 : total_notebooks = 350)
    (h2 : cost_price_A = 12)
    (h3 : cost_price_B = 15)
    (h4 : total_cost_price = 4800)
    (h5 : selling_price_A = 20)
    (h6 : selling_price_B = 25)
    (h7 : discount_A = 30)
    (h8 : 12 * x + 15 * y = 4800)
    (h9 : x + y = 350)
    (h10 : selling_price_A * m + selling_price_B * m + (x - m) * selling_price_A * 7 / 10 + (y - m) * cost_price_B - total_cost_price ≥ profit_condition):
    x = 150 ∧ m ≥ 128 :=
by
    sorry

end notebook_problem_l239_239901


namespace star_six_three_l239_239916

-- Definition of the operation
def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

-- Statement to prove
theorem star_six_three : star 6 3 = 3 := by
  sorry

end star_six_three_l239_239916


namespace convex_quadrilateral_area_lt_a_sq_l239_239589

theorem convex_quadrilateral_area_lt_a_sq {a x y z t : ℝ} (hx : x < a) (hy : y < a) (hz : z < a) (ht : t < a) :
  (∃ S : ℝ, S < a^2) :=
sorry

end convex_quadrilateral_area_lt_a_sq_l239_239589


namespace problem_proof_l239_239537

variable {α : Type*}
noncomputable def op (a b : ℝ) : ℝ := 1/a + 1/b
theorem problem_proof (a b : ℝ) (h : op a (-b) = 2) : (3 * a * b) / (2 * a - 2 * b) = -3/4 :=
by
  sorry

end problem_proof_l239_239537


namespace nell_more_ace_cards_than_baseball_l239_239017

-- Definitions based on conditions
def original_baseball_cards : ℕ := 239
def original_ace_cards : ℕ := 38
def current_ace_cards : ℕ := 376
def current_baseball_cards : ℕ := 111

-- The statement we need to prove
theorem nell_more_ace_cards_than_baseball :
  current_ace_cards - current_baseball_cards = 265 :=
by
  -- Add the proof here
  sorry

end nell_more_ace_cards_than_baseball_l239_239017


namespace pure_imaginary_complex_number_l239_239362

variable (a : ℝ)

theorem pure_imaginary_complex_number:
  (a^2 + 2*a - 3 = 0) ∧ (a^2 + a - 6 ≠ 0) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l239_239362


namespace fisherman_total_fish_l239_239428

theorem fisherman_total_fish :
  let bass : Nat := 32
  let trout : Nat := bass / 4
  let blue_gill : Nat := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  show bass + trout + blue_gill = 104
  sorry

end fisherman_total_fish_l239_239428


namespace ratio_of_areas_l239_239229

noncomputable def side_length_WXYZ : ℝ := 16

noncomputable def WJ : ℝ := (3/4) * side_length_WXYZ
noncomputable def JX : ℝ := (1/4) * side_length_WXYZ

noncomputable def side_length_JKLM := 4 * Real.sqrt 2

noncomputable def area_JKLM := (side_length_JKLM)^2
noncomputable def area_WXYZ := (side_length_WXYZ)^2

theorem ratio_of_areas : area_JKLM / area_WXYZ = 1 / 8 :=
by
  sorry

end ratio_of_areas_l239_239229


namespace final_price_is_correct_l239_239390

def initial_price : ℝ := 15
def first_discount_rate : ℝ := 0.2
def second_discount_rate : ℝ := 0.25

def first_discount : ℝ := initial_price * first_discount_rate
def price_after_first_discount : ℝ := initial_price - first_discount

def second_discount : ℝ := price_after_first_discount * second_discount_rate
def final_price : ℝ := price_after_first_discount - second_discount

theorem final_price_is_correct :
  final_price = 9 :=
by
  -- The actual proof steps will go here.
  sorry

end final_price_is_correct_l239_239390


namespace function_point_proof_l239_239312

-- Given conditions
def condition (f : ℝ → ℝ) : Prop :=
  f 1 = 3

-- Prove the statement
theorem function_point_proof (f : ℝ → ℝ) (h : condition f) : f (-1) + 1 = 4 :=
by
  -- Adding the conditions here
  sorry -- proof is not required

end function_point_proof_l239_239312


namespace sin_double_angle_plus_pi_over_2_l239_239422

theorem sin_double_angle_plus_pi_over_2 (θ : ℝ) (h : Real.cos θ = -1/3) :
  Real.sin (2 * θ + Real.pi / 2) = -7/9 :=
sorry

end sin_double_angle_plus_pi_over_2_l239_239422


namespace cubic_polynomial_unique_l239_239424

-- Define the polynomial q(x)
def q (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 4

-- State the conditions
theorem cubic_polynomial_unique :
  q 1 = -8 ∧
  q 2 = -10 ∧
  q 3 = -16 ∧
  q 4 = -32 :=
by
  -- Expand the function definition for the given inputs.
  -- Add these expansions in the proof part.
  sorry

end cubic_polynomial_unique_l239_239424


namespace gamesNextMonth_l239_239244

def gamesThisMonth : ℕ := 11
def gamesLastMonth : ℕ := 17
def totalPlannedGames : ℕ := 44

theorem gamesNextMonth :
  (totalPlannedGames - (gamesThisMonth + gamesLastMonth) = 16) :=
by
  unfold totalPlannedGames
  unfold gamesThisMonth
  unfold gamesLastMonth
  sorry

end gamesNextMonth_l239_239244


namespace probability_two_asian_countries_probability_A1_not_B1_l239_239677

-- Scope: Definitions for the problem context
def countries : List String := ["A1", "A2", "A3", "B1", "B2", "B3"]

-- Probability of picking two Asian countries from a pool of six (three Asian, three European)
theorem probability_two_asian_countries : 
  (3 / 15) = (1 / 5) := by
  sorry

-- Probability of picking one country from the Asian group and 
-- one from the European group, including A1 but not B1
theorem probability_A1_not_B1 : 
  (2 / 9) = (2 / 9) := by
  sorry

end probability_two_asian_countries_probability_A1_not_B1_l239_239677


namespace selena_ran_24_miles_l239_239981

theorem selena_ran_24_miles (S J : ℝ) (h1 : S + J = 36) (h2 : J = S / 2) : S = 24 := 
sorry

end selena_ran_24_miles_l239_239981


namespace identify_smart_person_l239_239485

theorem identify_smart_person (F S : ℕ) (h_total : F + S = 30) (h_max_fools : F ≤ 8) : S ≥ 1 :=
by {
  sorry
}

end identify_smart_person_l239_239485


namespace inverse_square_variation_l239_239052

variable (x y : ℝ)

theorem inverse_square_variation (h1 : x = 1) (h2 : y = 3) (h3 : y = 2) : x = 2.25 :=
by
  sorry

end inverse_square_variation_l239_239052


namespace parallel_vectors_sum_coords_l239_239051

theorem parallel_vectors_sum_coords
  (x y : ℝ)
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, x, 3))
  (h_b : b = (-4, 2, y))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  x + y = -7 :=
sorry

end parallel_vectors_sum_coords_l239_239051


namespace projection_of_difference_eq_l239_239456

noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vec_projection (v w : ℝ × ℝ) : ℝ :=
vec_dot (v - w) v / vec_magnitude v

variables (a b : ℝ × ℝ)
  (congruence_cond : vec_magnitude a / vec_magnitude b = Real.cos θ)

theorem projection_of_difference_eq (h : vec_magnitude a / vec_magnitude b = Real.cos θ) :
  vec_projection (a - b) a = (vec_dot a a - vec_dot b b) / vec_magnitude a :=
sorry

end projection_of_difference_eq_l239_239456


namespace functional_expression_y_l239_239621

theorem functional_expression_y (x y : ℝ) (k : ℝ) 
  (h1 : ∀ x, y + 2 = k * x) 
  (h2 : y = 7) 
  (h3 : x = 3) : 
  y = 3 * x - 2 := 
by 
  sorry

end functional_expression_y_l239_239621


namespace wolf_and_nobel_prize_laureates_l239_239210

-- Definitions from the conditions
def num_total_scientists : ℕ := 50
def num_wolf_prize_laureates : ℕ := 31
def num_nobel_prize_laureates : ℕ := 29
def num_no_wolf_prize_and_yes_nobel := 3 -- N_W = N_W'
def num_without_wolf_or_nobel : ℕ := num_total_scientists - num_wolf_prize_laureates - 11 -- Derived from N_W' 

-- The statement to be proved
theorem wolf_and_nobel_prize_laureates :
  ∃ W_N, W_N = num_nobel_prize_laureates - (19 - 3) ∧ W_N = 18 :=
  by
    sorry

end wolf_and_nobel_prize_laureates_l239_239210


namespace half_angle_second_quadrant_l239_239929

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_second_quadrant_l239_239929


namespace frac_3125_over_1024_gt_e_l239_239906

theorem frac_3125_over_1024_gt_e : (3125 : ℝ) / 1024 > Real.exp 1 := sorry

end frac_3125_over_1024_gt_e_l239_239906


namespace sum_of_squares_eq_1850_l239_239329

-- Assuming definitions for the rates
variables (b j s h : ℕ)

-- Condition from Ed's activity
axiom ed_condition : 3 * b + 4 * j + 2 * s + 3 * h = 120

-- Condition from Sue's activity
axiom sue_condition : 2 * b + 3 * j + 4 * s + 3 * h = 150

-- Sum of squares of biking, jogging, swimming, and hiking rates
def sum_of_squares (b j s h : ℕ) : ℕ := b^2 + j^2 + s^2 + h^2

-- Assertion we want to prove
theorem sum_of_squares_eq_1850 :
  ∃ b j s h : ℕ, 3 * b + 4 * j + 2 * s + 3 * h = 120 ∧ 2 * b + 3 * j + 4 * s + 3 * h = 150 ∧ sum_of_squares b j s h = 1850 :=
by
  sorry

end sum_of_squares_eq_1850_l239_239329


namespace least_positive_t_geometric_progression_l239_239256

noncomputable def least_positive_t( α : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) : ℝ :=
  9 - 4 * Real.sqrt 5

theorem least_positive_t_geometric_progression ( α t : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) :
  least_positive_t α h = t ↔
  ∃ r : ℝ, r > 0 ∧
    Real.arcsin (Real.sin α) = α ∧
    Real.arcsin (Real.sin (2 * α)) = 2 * α ∧
    Real.arcsin (Real.sin (7 * α)) = 7 * α ∧
    Real.arcsin (Real.sin (t * α)) = t * α ∧
    (α * r = 2 * α) ∧
    (2 * α * r = 7 * α ) ∧
    (7 * α * r = t * α) :=
sorry

end least_positive_t_geometric_progression_l239_239256


namespace transformed_curve_l239_239471

variables (x y x' y' : ℝ)

def original_curve := (x^2) / 4 - y^2 = 1
def transformation_x := x' = (1/2) * x
def transformation_y := y' = 2 * y

theorem transformed_curve : original_curve x y → transformation_x x x' → transformation_y y y' → x^2 - (y^2) / 4 = 1 := 
sorry

end transformed_curve_l239_239471


namespace nina_widgets_purchase_l239_239284

theorem nina_widgets_purchase (P : ℝ) (h1 : 8 * (P - 1) = 24) (h2 : 24 / P = 6) : true :=
by
  sorry

end nina_widgets_purchase_l239_239284


namespace Peter_buys_more_hot_dogs_than_hamburgers_l239_239835

theorem Peter_buys_more_hot_dogs_than_hamburgers :
  let chicken := 16
  let hamburgers := chicken / 2
  (exists H : Real, 16 + hamburgers + H + H / 2 = 39 ∧ (H - hamburgers = 2)) := sorry

end Peter_buys_more_hot_dogs_than_hamburgers_l239_239835


namespace shark_ratio_l239_239094

theorem shark_ratio (N D : ℕ) (h1 : N = 22) (h2 : D + N = 110) (h3 : ∃ x : ℕ, D = x * N) : 
  (D / N) = 4 :=
by
  -- conditions use only definitions given in the problem.
  sorry

end shark_ratio_l239_239094


namespace geometric_sequence_problem_l239_239261

variable {a : ℕ → ℝ}

theorem geometric_sequence_problem (h1 : a 5 * a 7 = 2) (h2 : a 2 + a 10 = 3) : 
  (a 12 / a 4 = 1 / 2) ∨ (a 12 / a 4 = 2) := 
sorry

end geometric_sequence_problem_l239_239261


namespace selection_assignment_schemes_l239_239665

noncomputable def number_of_selection_schemes (males females : ℕ) : ℕ :=
  if h : males + females < 3 then 0
  else
    let total3 := Nat.choose (males + females) 3
    let all_males := if hM : males < 3 then 0 else Nat.choose males 3
    let all_females := if hF : females < 3 then 0 else Nat.choose females 3
    total3 - all_males - all_females

theorem selection_assignment_schemes :
  number_of_selection_schemes 4 3 = 30 :=
by sorry

end selection_assignment_schemes_l239_239665


namespace area_of_square_on_RS_l239_239536

theorem area_of_square_on_RS (PQ QR PS PS_square PQ_square QR_square : ℝ)
  (hPQ : PQ_square = 25) (hQR : QR_square = 49) (hPS : PS_square = 64)
  (hPQ_eq : PQ_square = PQ^2) (hQR_eq : QR_square = QR^2) (hPS_eq : PS_square = PS^2)
  : ∃ RS_square : ℝ, RS_square = 138 := by
  let PR_square := PQ^2 + QR^2
  let RS_square := PR_square + PS^2
  use RS_square
  sorry

end area_of_square_on_RS_l239_239536


namespace decreasing_function_condition_l239_239154

theorem decreasing_function_condition :
  (∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0) :=
by
  -- Proof outline goes here
  sorry

end decreasing_function_condition_l239_239154


namespace original_number_of_men_l239_239698

theorem original_number_of_men (W : ℝ) (M : ℝ) (total_work : ℝ) :
  (M * W * 11 = (M + 10) * W * 8) → M = 27 :=
by
  sorry

end original_number_of_men_l239_239698


namespace max_correct_questions_prime_score_l239_239347

-- Definitions and conditions
def total_questions := 20
def points_correct := 5
def points_no_answer := 0
def points_wrong := -2

-- Main statement to prove
theorem max_correct_questions_prime_score :
  ∃ (correct : ℕ) (no_answer wrong : ℕ), 
    correct + no_answer + wrong = total_questions ∧ 
    correct * points_correct + no_answer * points_no_answer + wrong * points_wrong = 83 ∧
    correct = 17 :=
sorry

end max_correct_questions_prime_score_l239_239347


namespace gcd_2750_9450_l239_239631

theorem gcd_2750_9450 : Nat.gcd 2750 9450 = 50 := by
  sorry

end gcd_2750_9450_l239_239631


namespace S_calculation_T_calculation_l239_239451

def S (a b : ℕ) : ℕ := 4 * a + 6 * b
def T (a b : ℕ) : ℕ := 5 * a + 3 * b

theorem S_calculation : S 6 3 = 42 :=
by sorry

theorem T_calculation : T 6 3 = 39 :=
by sorry

end S_calculation_T_calculation_l239_239451


namespace class_5_matches_l239_239478

theorem class_5_matches (matches_c1 matches_c2 matches_c3 matches_c4 matches_c5 : ℕ)
  (C1 : matches_c1 = 2)
  (C2 : matches_c2 = 4)
  (C3 : matches_c3 = 4)
  (C4 : matches_c4 = 3) :
  matches_c5 = 3 :=
sorry

end class_5_matches_l239_239478


namespace triangle_angle_sum_l239_239350

-- Definitions of the given angles and relationships
def angle_BAC := 95
def angle_ABC := 55
def angle_ABD := 125

-- We need to express the configuration of points and the measure of angle ACB
noncomputable def angle_ACB (angle_BAC angle_ABC angle_ABD : ℝ) : ℝ :=
  180 - angle_BAC - angle_ABC

-- The formalization of the problem statement in Lean 4
theorem triangle_angle_sum (angle_BAC angle_ABC angle_ABD : ℝ) :
  angle_BAC = 95 → angle_ABC = 55 → angle_ABD = 125 → angle_ACB angle_BAC angle_ABC angle_ABD = 30 :=
by
  intros h_BAC h_ABC h_ABD
  rw [h_BAC, h_ABC, h_ABD]
  sorry

end triangle_angle_sum_l239_239350


namespace temperature_reading_l239_239172

theorem temperature_reading (scale_min scale_max : ℝ) (arrow : ℝ) (h1 : scale_min = -6.0) (h2 : scale_max = -5.5) (h3 : scale_min < arrow) (h4 : arrow < scale_max) : arrow = -5.7 :=
sorry

end temperature_reading_l239_239172


namespace system1_solution_system2_solution_l239_239960

-- Define the first system of equations and its solution
theorem system1_solution (x y : ℝ) : 
    (3 * (x - 1) = y + 5 ∧ 5 * (y - 1) = 3 * (x + 5)) ↔ (x = 5 ∧ y = 7) :=
sorry

-- Define the second system of equations and its solution
theorem system2_solution (x y a : ℝ) :
    (2 * x + 4 * y = a ∧ 7 * x - 2 * y = 3 * a) ↔ 
    (x = (7 / 16) * a ∧ y = (1 / 32) * a) :=
sorry

end system1_solution_system2_solution_l239_239960


namespace right_triangle_x_value_l239_239848

variable (BM MA BC CA x h d : ℝ)

theorem right_triangle_x_value (BM MA BC CA x h d : ℝ)
  (h4 : BM + MA = BC + CA)
  (h5 : BM = x)
  (h6 : BC = h)
  (h7 : CA = d) :
  x = h * d / (2 * h + d) := 
sorry

end right_triangle_x_value_l239_239848


namespace percent_area_square_in_rectangle_l239_239528

theorem percent_area_square_in_rectangle 
  (s : ℝ) (rect_width : ℝ) (rect_length : ℝ) (h1 : rect_width = 2 * s) (h2 : rect_length = 2 * rect_width) : 
  (s^2 / (rect_length * rect_width)) * 100 = 12.5 :=
by
  sorry

end percent_area_square_in_rectangle_l239_239528


namespace tan_degree_identity_l239_239364

theorem tan_degree_identity (k : ℝ) (hk : Real.cos (Real.pi * -80 / 180) = k) : 
  Real.tan (Real.pi * 100 / 180) = - (Real.sqrt (1 - k^2) / k) := 
by 
  sorry

end tan_degree_identity_l239_239364


namespace polynomials_with_conditions_l239_239915

theorem polynomials_with_conditions (n : ℕ) (h_pos : 0 < n) :
  (∃ P : Polynomial ℤ, Polynomial.degree P = n ∧ 
      (∃ (k : Fin n → ℤ), Function.Injective k ∧ (∀ i, P.eval (k i) = n) ∧ P.eval 0 = 0)) ↔ 
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
sorry

end polynomials_with_conditions_l239_239915


namespace parabola_distance_l239_239819

theorem parabola_distance (y : ℝ) (h : y ^ 2 = 24) : |-6 - 1| = 7 :=
by { sorry }

end parabola_distance_l239_239819


namespace perfect_square_iff_l239_239996

theorem perfect_square_iff (x y z : ℕ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  ∃ k : ℕ, 4^x + 4^y + 4^z = k^2 ↔ ∃ b : ℕ, b > 0 ∧ x = 2 * b - 1 + z ∧ y = b + z :=
by
  sorry

end perfect_square_iff_l239_239996


namespace large_circle_radius_l239_239071

noncomputable def radius_of_large_circle (R : ℝ) : Prop :=
  ∃ r : ℝ, (r = 2) ∧
           (R = r + r) ∧
           (r = 2) ∧
           (R - r = 2) ∧
           (R = 4)

theorem large_circle_radius :
  radius_of_large_circle 4 :=
by
  sorry

end large_circle_radius_l239_239071


namespace least_months_for_tripling_debt_l239_239147

theorem least_months_for_tripling_debt (P : ℝ) (r : ℝ) (t : ℕ) : P = 1500 → r = 0.06 → (3 * P < P * (1 + r) ^ t) → t ≥ 20 :=
by
  intros hP hr hI
  rw [hP, hr] at hI
  norm_num at hI
  sorry

end least_months_for_tripling_debt_l239_239147


namespace cars_per_client_l239_239502

-- Define the conditions
def num_cars : ℕ := 18
def selections_per_car : ℕ := 3
def num_clients : ℕ := 18

-- Define the proof problem as a theorem
theorem cars_per_client :
  (num_cars * selections_per_car) / num_clients = 3 :=
sorry

end cars_per_client_l239_239502


namespace roots_polynomial_identity_l239_239788

theorem roots_polynomial_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a * b + b * c + c * a = 22) (h3 : a * b * c = 8) :
  (2 + a) * (2 + b) * (2 + c) = 120 :=
by
  sorry

end roots_polynomial_identity_l239_239788


namespace evaluate_poly_at_2_l239_239577

def my_op (x y : ℕ) : ℕ := (x + 1) * (y + 1)
def star2 (x : ℕ) : ℕ := my_op x x

theorem evaluate_poly_at_2 :
  3 * (star2 2) - 2 * 2 + 1 = 24 :=
by
  sorry

end evaluate_poly_at_2_l239_239577


namespace part1_part2_part3_l239_239292

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem part1 : determinant (-3) (-2) 4 5 = -7 := by
  sorry

theorem part2 (x: ℝ) (h: determinant 2 (-2 * x) 3 (-5 * x) = 2) : x = -1/2 := by
  sorry

theorem part3 (m n x: ℝ) 
  (h1: determinant (8 * m * x - 1) (-8/3 + 2 * x) (3/2) (-3) = 
        determinant 6 (-1) (-n) x) : 
    m = -3/8 ∧ n = -7 := by
  sorry

end part1_part2_part3_l239_239292


namespace minimum_tasks_for_18_points_l239_239703

def task_count (points : ℕ) : ℕ :=
  if points <= 9 then
    (points / 3) * 1
  else if points <= 15 then
    3 + (points - 9 + 2) / 3 * 2
  else
    3 + 4 + (points - 15 + 2) / 3 * 3

theorem minimum_tasks_for_18_points : task_count 18 = 10 := by
  sorry

end minimum_tasks_for_18_points_l239_239703


namespace quad_in_vertex_form_addition_l239_239298

theorem quad_in_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∃ a h k, (4 * x^2 - 8 * x + 3) = a * (x - h) ^ 2 + k) →
  a + h + k = 4 :=
by
  sorry

end quad_in_vertex_form_addition_l239_239298


namespace joan_balloons_l239_239509

def initial_balloons : ℕ := 72
def additional_balloons : ℕ := 23
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem joan_balloons : total_balloons = 95 := by
  sorry

end joan_balloons_l239_239509


namespace line_parabola_intersections_l239_239297

theorem line_parabola_intersections (k : ℝ) :
  ((∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) ↔ k = 0) ∧
  (¬∃ x₁ x₂, x₁ ≠ x₂ ∧ (k * (x₁ - 2) + 1)^2 = 4 * x₁ ∧ (k * (x₂ - 2) + 1)^2 = 4 * x₂) ∧
  (¬∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) :=
by sorry

end line_parabola_intersections_l239_239297


namespace average_speed_round_trip_l239_239872

def time_to_walk_uphill := 30 -- in minutes
def time_to_walk_downhill := 10 -- in minutes
def distance_one_way := 1 -- in km

theorem average_speed_round_trip :
  (2 * distance_one_way) / ((time_to_walk_uphill + time_to_walk_downhill) / 60) = 3 := by
  sorry

end average_speed_round_trip_l239_239872


namespace problem_statement_l239_239880

noncomputable def a : ℝ := Real.tan (1 / 2)
noncomputable def b : ℝ := Real.tan (2 / Real.pi)
noncomputable def c : ℝ := Real.sqrt 3 / Real.pi

theorem problem_statement : a < c ∧ c < b := by
  sorry

end problem_statement_l239_239880


namespace find_sum_lent_l239_239116

theorem find_sum_lent (r t : ℝ) (I : ℝ) (P : ℝ) (h1: r = 0.06) (h2 : t = 8) (h3 : I = P - 520) (h4: I = P * r * t) : P = 1000 := by
  sorry

end find_sum_lent_l239_239116


namespace find_numbers_l239_239984

def is_7_digit (n : ℕ) : Prop := n ≥ 1000000 ∧ n < 10000000
def is_14_digit (n : ℕ) : Prop := n >= 10^13 ∧ n < 10^14

theorem find_numbers (x y z : ℕ) (hx7 : is_7_digit x) (hy7 : is_7_digit y) (hz14 : is_14_digit z) :
  3 * x * y = z ∧ z = 10^7 * x + y → 
  x = 1666667 ∧ y = 3333334 ∧ z = 16666673333334 := 
by
  sorry

end find_numbers_l239_239984


namespace inequality_ay_bz_cx_lt_k_squared_l239_239708

theorem inequality_ay_bz_cx_lt_k_squared
  (a b c x y z k : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hk1 : a + x = k) (hk2 : b + y = k) (hk3 : c + z = k) :
  (a * y + b * z + c * x) < k^2 :=
sorry

end inequality_ay_bz_cx_lt_k_squared_l239_239708


namespace ratio_of_girls_to_boys_l239_239907

-- Define conditions
def num_boys : ℕ := 40
def children_per_counselor : ℕ := 8
def num_counselors : ℕ := 20

-- Total number of children
def total_children : ℕ := num_counselors * children_per_counselor

-- Number of girls
def num_girls : ℕ := total_children - num_boys

-- The ratio of girls to boys
def girls_to_boys_ratio : ℚ := num_girls / num_boys

-- The theorem we need to prove
theorem ratio_of_girls_to_boys : girls_to_boys_ratio = 3 := by
  sorry

end ratio_of_girls_to_boys_l239_239907


namespace factorize_expression_l239_239457

theorem factorize_expression (x : ℝ) : 2 * x ^ 3 - 4 * x ^ 2 - 6 * x = 2 * x * (x - 3) * (x + 1) :=
by
  sorry

end factorize_expression_l239_239457


namespace james_writing_time_l239_239357

theorem james_writing_time (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ):
  pages_per_hour = 10 →
  pages_per_person_per_day = 5 →
  num_people = 2 →
  days_per_week = 7 →
  (5 * 2 * 7) / 10 = 7 :=
by
  intros
  sorry

end james_writing_time_l239_239357


namespace Xiaoming_age_l239_239204

theorem Xiaoming_age (x : ℕ) (h1 : x = x) (h2 : x + 18 = 2 * (x + 6)) : x = 6 :=
sorry

end Xiaoming_age_l239_239204


namespace S8_eq_90_l239_239823

-- Definitions and given conditions
def arithmetic_seq (a : ℕ → ℤ) : Prop := ∃ d, ∀ n, a (n + 1) - a n = d
def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2
def condition_a4 (a : ℕ → ℤ) : Prop := a 4 = 18 - a 5

-- Prove that S₈ = 90
theorem S8_eq_90 (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_seq a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond : condition_a4 a) : S 8 = 90 :=
by
  sorry

end S8_eq_90_l239_239823


namespace circle_x_intercept_of_given_diameter_l239_239395

theorem circle_x_intercept_of_given_diameter (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (10, 8)) : ∃ x : ℝ, ((A.1 + B.1) / 2, (A.2 + B.2) / 2).1 - 6 = 0 :=
by
  -- Sorry to skip the proof
  sorry

end circle_x_intercept_of_given_diameter_l239_239395


namespace sum_of_six_consecutive_integers_l239_239945

theorem sum_of_six_consecutive_integers (m : ℤ) : 
  (m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)) = 6 * m + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l239_239945


namespace remainder_when_divided_by_8_l239_239944

theorem remainder_when_divided_by_8 (x : ℤ) (k : ℤ) (h : x = 72 * k + 19) : x % 8 = 3 :=
by sorry

end remainder_when_divided_by_8_l239_239944


namespace option_c_is_not_equal_l239_239535

theorem option_c_is_not_equal :
  let A := 14 / 12
  let B := 1 + 1 / 6
  let C := 1 + 1 / 2
  let D := 1 + 7 / 42
  let E := 1 + 14 / 84
  A = 7 / 6 ∧ B = 7 / 6 ∧ D = 7 / 6 ∧ E = 7 / 6 ∧ C ≠ 7 / 6 :=
by
  sorry

end option_c_is_not_equal_l239_239535


namespace parallel_lines_a_value_l239_239322

theorem parallel_lines_a_value (a : ℝ) 
  (h1 : ∀ x y : ℝ, x + a * y - 1 = 0 → x = a * (-4 * y - 2)) 
  : a = 2 :=
sorry

end parallel_lines_a_value_l239_239322


namespace sum_of_roots_of_quadratic_l239_239450

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = 6 ∧ c = -9) :
  (-b / a) = -2 :=
by
  rcases h_eq with ⟨ha, hb, hc⟩
  -- Proof goes here, but we can use sorry to skip it
  sorry

end sum_of_roots_of_quadratic_l239_239450


namespace cost_to_produce_program_l239_239900

theorem cost_to_produce_program
  (advertisement_revenue : ℝ)
  (number_of_copies : ℝ)
  (price_per_copy : ℝ)
  (desired_profit : ℝ)
  (total_revenue : ℝ)
  (revenue_from_sales : ℝ)
  (cost_to_produce : ℝ) :
  advertisement_revenue = 15000 →
  number_of_copies = 35000 →
  price_per_copy = 0.5 →
  desired_profit = 8000 →
  total_revenue = advertisement_revenue + desired_profit →
  revenue_from_sales = number_of_copies * price_per_copy →
  total_revenue = revenue_from_sales + cost_to_produce →
  cost_to_produce = 5500 :=
by
  sorry

end cost_to_produce_program_l239_239900


namespace circle_radius_squared_l239_239503

open Real

/-- Prove that the square of the radius of a circle is 200 given the conditions provided. -/

theorem circle_radius_squared {r : ℝ}
  (AB CD : ℝ)
  (BP : ℝ) 
  (APD : ℝ) 
  (hAB : AB = 12)
  (hCD : CD = 9)
  (hBP : BP = 10)
  (hAPD : APD = 45) :
  r^2 = 200 := 
sorry

end circle_radius_squared_l239_239503


namespace numberOfBaseballBoxes_l239_239068

-- Given conditions as Lean definitions and assumptions
def numberOfBasketballBoxes : ℕ := 4
def basketballCardsPerBox : ℕ := 10
def baseballCardsPerBox : ℕ := 8
def cardsGivenToClassmates : ℕ := 58
def cardsLeftAfterGiving : ℕ := 22

def totalBasketballCards : ℕ := numberOfBasketballBoxes * basketballCardsPerBox
def totalCardsBeforeGiving : ℕ := cardsLeftAfterGiving + cardsGivenToClassmates

-- Target number of baseball cards
def totalBaseballCards : ℕ := totalCardsBeforeGiving - totalBasketballCards

-- Prove that the number of baseball boxes is 5
theorem numberOfBaseballBoxes :
  totalBaseballCards / baseballCardsPerBox = 5 :=
sorry

end numberOfBaseballBoxes_l239_239068


namespace min_value_of_m_l239_239785

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, 0 < x → x ≠ ⌊x⌋ → mx < Real.log x) ↔ m = (1 / 2) * Real.log 2 :=
by
  sorry

end min_value_of_m_l239_239785


namespace base_number_eq_2_l239_239315

theorem base_number_eq_2 (x : ℝ) (n : ℕ) (h₁ : x^(2 * n) + x^(2 * n) + x^(2 * n) + x^(2 * n) = 4^28) (h₂ : n = 27) : x = 2 := by
  sorry

end base_number_eq_2_l239_239315


namespace smallest_integer_in_consecutive_set_l239_239704

theorem smallest_integer_in_consecutive_set (n : ℤ) (h : n + 6 < 2 * (n + 3)) : n > 0 := by
  sorry

end smallest_integer_in_consecutive_set_l239_239704


namespace positive_diff_probability_fair_coin_l239_239462

theorem positive_diff_probability_fair_coin :
  let p1 := (Nat.choose 5 3) * (1 / 2)^5
  let p2 := (1 / 2)^5
  p1 - p2 = 9 / 32 :=
by
  sorry

end positive_diff_probability_fair_coin_l239_239462


namespace distance_of_intersection_points_l239_239023

def C1 (x y : ℝ) : Prop := x - y + 4 = 0
def C2 (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

theorem distance_of_intersection_points {A B : ℝ × ℝ} (hA1 : C1 A.fst A.snd) (hA2 : C2 A.fst A.snd)
  (hB1 : C1 B.fst B.snd) (hB2 : C2 B.fst B.snd) : dist A B = Real.sqrt 2 := by
  sorry

end distance_of_intersection_points_l239_239023


namespace min_value_of_a1_plus_a7_l239_239482

variable {a : ℕ → ℝ}
variable {a3 a5 : ℝ}

-- Conditions
def is_positive_geometric_sequence (a : ℕ → ℝ) := 
  ∀ n, a n > 0 ∧ (∃ r, ∀ i, a (i + 1) = a i * r)

def condition (a : ℕ → ℝ) (a3 a5 : ℝ) :=
  a 3 = a3 ∧ a 5 = a5 ∧ a3 * a5 = 64

-- Prove that the minimum value of a1 + a7 is 16
theorem min_value_of_a1_plus_a7
  (h1 : is_positive_geometric_sequence a)
  (h2 : condition a a3 a5) :
  ∃ a1 a7, a 1 = a1 ∧ a 7 = a7 ∧ (∃ (min_sum : ℝ), min_sum = 16 ∧ ∀ sum, sum = a1 + a7 → sum ≥ min_sum) :=
sorry

end min_value_of_a1_plus_a7_l239_239482


namespace ways_to_place_letters_l239_239266

-- defining the conditions of the problem
def num_letters : Nat := 4
def num_mailboxes : Nat := 3

-- the theorem we need to prove
theorem ways_to_place_letters : 
  (num_mailboxes ^ num_letters) = 81 := 
by 
  sorry

end ways_to_place_letters_l239_239266


namespace joshua_miles_ratio_l239_239102

-- Definitions corresponding to conditions
def mitch_macarons : ℕ := 20
def joshua_extra : ℕ := 6
def total_kids : ℕ := 68
def macarons_per_kid : ℕ := 2

-- Variables for unspecified amounts
variable (M : ℕ) -- number of macarons Miles made

-- Calculations for Joshua and Renz's macarons based on given conditions
def joshua_macarons := mitch_macarons + joshua_extra
def renz_macarons := (3 * M) / 4 - 1

-- Total macarons calculation
def total_macarons := mitch_macarons + joshua_macarons + renz_macarons + M

-- Proof statement: Showing the ratio of number of macarons Joshua made to the number of macarons Miles made
theorem joshua_miles_ratio : (total_macarons = total_kids * macarons_per_kid) → (joshua_macarons : ℚ) / (M : ℚ) = 1 / 2 :=
by
  sorry

end joshua_miles_ratio_l239_239102


namespace prove_area_and_sum_l239_239151

-- Define the coordinates of the vertices of the quadrilateral.
variables (a b : ℤ)

-- Define the non-computable requirements related to the problem.
noncomputable def problem_statement : Prop :=
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a > b ∧ (4 * a * b = 32) ∧ (a + b = 5)

theorem prove_area_and_sum : problem_statement := 
sorry

end prove_area_and_sum_l239_239151


namespace original_speed_l239_239714

noncomputable def circumference_feet := 10
noncomputable def feet_to_miles := 5280
noncomputable def seconds_to_hours := 3600
noncomputable def shortened_time := 1 / 18000
noncomputable def speed_increase := 6

theorem original_speed (r : ℝ) (t : ℝ) : 
  r * t = (circumference_feet / feet_to_miles) * seconds_to_hours ∧ 
  (r + speed_increase) * (t - shortened_time) = (circumference_feet / feet_to_miles) * seconds_to_hours
  → r = 6 := 
by
  sorry

end original_speed_l239_239714


namespace rectangle_shaded_area_fraction_l239_239045

-- Defining necessary parameters and conditions
variables {R : Type} [LinearOrderedField R]

noncomputable def shaded_fraction (length width : R) : R :=
  let P : R × R := (0, width / 2)
  let Q : R × R := (length / 2, width)
  let rect_area := length * width
  let tri_area := (1 / 2) * (length / 2) * (width / 2)
  let shaded_area := rect_area - tri_area
  shaded_area / rect_area

-- The theorem stating our desired proof goal
theorem rectangle_shaded_area_fraction (length width : R) (h_length : 0 < length) (h_width : 0 < width) :
  shaded_fraction length width = 7 / 8 := by
  sorry

end rectangle_shaded_area_fraction_l239_239045


namespace range_of_a_l239_239446

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < -1 ↔ x ≤ a) ↔ a < -1 :=
by
  sorry

end range_of_a_l239_239446


namespace area_of_circle_given_circumference_l239_239273

theorem area_of_circle_given_circumference (C : ℝ) (hC : C = 18 * Real.pi) (k : ℝ) :
  ∃ r : ℝ, C = 2 * Real.pi * r ∧ k * Real.pi = Real.pi * r^2 → k = 81 :=
by
  sorry

end area_of_circle_given_circumference_l239_239273


namespace small_barrel_5_tons_l239_239888

def total_oil : ℕ := 95
def large_barrel_capacity : ℕ := 6
def small_barrel_capacity : ℕ := 5

theorem small_barrel_5_tons :
  ∃ (num_large_barrels num_small_barrels : ℕ),
  num_small_barrels = 1 ∧
  total_oil = (num_large_barrels * large_barrel_capacity) + (num_small_barrels * small_barrel_capacity) :=
by
  sorry

end small_barrel_5_tons_l239_239888


namespace coefficient_of_x3_in_expansion_l239_239338

noncomputable def binomial_expansion_coefficient (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem coefficient_of_x3_in_expansion : 
  (∀ k : ℕ, binomial_expansion_coefficient 6 k ≤ binomial_expansion_coefficient 6 3) →
  binomial_expansion_coefficient 6 3 = 20 :=
by
  intro h
  -- skipping the proof
  sorry

end coefficient_of_x3_in_expansion_l239_239338


namespace divisibility_by_seven_l239_239097

theorem divisibility_by_seven : (∃ k : ℤ, (-8)^2019 + (-8)^2018 = 7 * k) :=
sorry

end divisibility_by_seven_l239_239097


namespace charge_increase_percentage_l239_239793

variable (P R G : ℝ)

def charge_relation_1 : Prop := P = 0.45 * R
def charge_relation_2 : Prop := P = 0.90 * G

theorem charge_increase_percentage (h1 : charge_relation_1 P R) (h2 : charge_relation_2 P G) : 
  (R/G - 1) * 100 = 100 :=
by
  sorry

end charge_increase_percentage_l239_239793


namespace sum_end_digit_7_l239_239671

theorem sum_end_digit_7 (n : ℕ) : ¬ (n * (n + 1) ≡ 14 [MOD 20]) :=
by
  intro h
  -- Place where you'd continue the proof, but for now we use sorry
  sorry

end sum_end_digit_7_l239_239671


namespace smallest_possible_a_l239_239029

noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + ↑c

theorem smallest_possible_a
  (a b c : ℕ)
  (r s : ℝ)
  (h_arith_seq : b - a = c - b)
  (h_order_pos : 0 < a ∧ a < b ∧ b < c)
  (h_distinct : r ≠ s)
  (h_rs_2017 : r * s = 2017)
  (h_fr_eq_s : f a b c r = s)
  (h_fs_eq_r : f a b c s = r) :
  a = 1 := sorry

end smallest_possible_a_l239_239029


namespace find_a7_l239_239169

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h3 : a 3 = 1)
  (h_det : a 6 * a 8 - 8 * 8 = 0) :
  a 7 = 8 :=
sorry

end find_a7_l239_239169


namespace tourist_tax_l239_239414

theorem tourist_tax (total_value : ℕ) (non_taxable_amount : ℕ) (tax_rate : ℚ) (tax : ℚ) : 
  total_value = 1720 → 
  non_taxable_amount = 600 → 
  tax_rate = 0.12 → 
  tax = (total_value - non_taxable_amount : ℕ) * tax_rate → 
  tax = 134.40 := 
by 
  intros total_value_eq non_taxable_amount_eq tax_rate_eq tax_eq
  sorry

end tourist_tax_l239_239414


namespace hexagon_arrangements_eq_144_l239_239622

def is_valid_arrangement (arr : (Fin 7 → ℕ)) : Prop :=
  ∀ (i j k : Fin 7),
    (i.val + j.val + k.val = 18) → -- 18 being a derived constant factor (since 3x = 28 + 2G where G ∈ {1, 4, 7} and hence x = 30,34,38/3 respectively make it divisible by 3 sum is 18 always)
    arr i + arr j + arr k = arr ⟨3, sorry⟩ -- arr[3] is the position of G

noncomputable def count_valid_arrangements : ℕ :=
  sorry -- Calculation of 3*48 goes here and respective pairing and permutations.

theorem hexagon_arrangements_eq_144 :
  count_valid_arrangements = 144 :=
sorry

end hexagon_arrangements_eq_144_l239_239622


namespace at_least_one_less_than_equal_one_l239_239141

theorem at_least_one_less_than_equal_one
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 := 
by 
  sorry

end at_least_one_less_than_equal_one_l239_239141


namespace effect_on_revenue_decrease_l239_239992

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q

def new_price (P : ℝ) : ℝ := P * 1.40

def new_quantity (Q : ℝ) : ℝ := Q * 0.65

def new_revenue (P Q : ℝ) : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue_decrease :
  new_revenue P Q = original_revenue P Q * 0.91 →
  new_revenue P Q - original_revenue P Q = original_revenue P Q * -0.09 :=
by
  sorry

end effect_on_revenue_decrease_l239_239992


namespace minimum_value_of_xy_l239_239431

theorem minimum_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy = 64 :=
sorry

end minimum_value_of_xy_l239_239431


namespace horse_running_time_l239_239122

def area_of_square_field : Real := 625
def speed_of_horse_around_field : Real := 25

theorem horse_running_time : (4 : Real) = 
  let side_length := Real.sqrt area_of_square_field
  let perimeter := 4 * side_length
  perimeter / speed_of_horse_around_field :=
by
  sorry

end horse_running_time_l239_239122


namespace relationship_between_abc_l239_239302

noncomputable def a := (4 / 5) ^ (1 / 2)
noncomputable def b := (5 / 4) ^ (1 / 5)
noncomputable def c := (3 / 4) ^ (3 / 4)

theorem relationship_between_abc : c < a ∧ a < b := by
  sorry

end relationship_between_abc_l239_239302


namespace bread_slices_per_friend_l239_239582

theorem bread_slices_per_friend :
  (∀ (slices_per_loaf friends loaves total_slices_per_friend : ℕ),
    slices_per_loaf = 15 →
    friends = 10 →
    loaves = 4 →
    total_slices_per_friend = slices_per_loaf * loaves / friends →
    total_slices_per_friend = 6) :=
by 
  intros slices_per_loaf friends loaves total_slices_per_friend h1 h2 h3 h4
  sorry

end bread_slices_per_friend_l239_239582


namespace markers_needed_total_l239_239346

noncomputable def markers_needed_first_group : ℕ := 10 * 2
noncomputable def markers_needed_second_group : ℕ := 15 * 4
noncomputable def students_last_group : ℕ := 30 - (10 + 15)
noncomputable def markers_needed_last_group : ℕ := students_last_group * 6

theorem markers_needed_total : markers_needed_first_group + markers_needed_second_group + markers_needed_last_group = 110 :=
by
  sorry

end markers_needed_total_l239_239346


namespace peaches_thrown_away_l239_239279

variables (total_peaches fresh_percentage peaches_left : ℕ) (thrown_away : ℕ)
variables (h1 : total_peaches = 250) (h2 : fresh_percentage = 60) (h3 : peaches_left = 135)

theorem peaches_thrown_away :
  thrown_away = (total_peaches * (fresh_percentage / 100)) - peaches_left :=
sorry

end peaches_thrown_away_l239_239279


namespace slab_length_l239_239060

noncomputable def area_of_one_slab (total_area: ℝ) (num_slabs: ℕ) : ℝ :=
  total_area / num_slabs

noncomputable def length_of_one_slab (slab_area : ℝ) : ℝ :=
  Real.sqrt slab_area

theorem slab_length (total_area : ℝ) (num_slabs : ℕ)
  (h_total_area : total_area = 98)
  (h_num_slabs : num_slabs = 50) :
  length_of_one_slab (area_of_one_slab total_area num_slabs) = 1.4 :=
by
  sorry

end slab_length_l239_239060


namespace calc_S_5_minus_S_4_l239_239004

def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 2 * a n - 2

theorem calc_S_5_minus_S_4 {a : ℕ → ℕ} {S : ℕ → ℕ}
  (h : sum_sequence a S) : S 5 - S 4 = 32 :=
by
  sorry

end calc_S_5_minus_S_4_l239_239004


namespace rank_A_second_l239_239693

-- We define the conditions provided in the problem
variables (a b c : ℕ) -- defining the scores of A, B, and C as natural numbers

-- Conditions given
def A_said (a b c : ℕ) := b < a ∧ c < a
def B_said (b c : ℕ) := b > c
def C_said (a b c : ℕ) := a > c ∧ b > c

-- Conditions as hypotheses
variable (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) -- the scores are different
variable (h2 : A_said a b c ∨ B_said b c ∨ C_said a b c) -- exactly one of the statements is incorrect

-- The theorem to prove
theorem rank_A_second : ∃ (rankA : ℕ), rankA = 2 := by
  sorry

end rank_A_second_l239_239693


namespace base6_addition_sum_l239_239628

theorem base6_addition_sum 
  (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : Q ≠ R) 
  (h3 : P ≠ R) 
  (h4 : P < 6) 
  (h5 : Q < 6) 
  (h6 : R < 6) 
  (h7 : 2*R % 6 = P) 
  (h8 : 2*Q % 6 = R)
  : P + Q + R = 7 := 
  sorry

end base6_addition_sum_l239_239628


namespace man_double_son_in_years_l239_239489

-- Definitions of conditions
def son_age : ℕ := 18
def man_age : ℕ := son_age + 20

-- The proof problem statement
theorem man_double_son_in_years :
  ∃ (X : ℕ), (man_age + X = 2 * (son_age + X)) ∧ X = 2 :=
by
  sorry

end man_double_son_in_years_l239_239489


namespace scientific_notation_l239_239573

theorem scientific_notation (a : ℝ) (n : ℤ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 43050000 = a * 10^n) : a = 4.305 ∧ n = 7 :=
by
  sorry

end scientific_notation_l239_239573


namespace extremum_range_a_l239_239818

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - a * x^2 + x

theorem extremum_range_a :
  (∀ x : ℝ, -1 < x ∧ x < 0 → (f a x = 0 → ∃ x0 : ℝ, f a x0 = 0 ∧ -1 < x0 ∧ x0 < 0)) →
  a < -1/5 ∨ a = -1 :=
sorry

end extremum_range_a_l239_239818


namespace vertical_asymptotes_sum_l239_239543

theorem vertical_asymptotes_sum (A B C : ℤ)
  (h : ∀ x : ℝ, x = -1 ∨ x = 2 ∨ x = 3 → x^3 + A * x^2 + B * x + C = 0)
  : A + B + C = -3 :=
sorry

end vertical_asymptotes_sum_l239_239543


namespace cistern_empty_time_l239_239675

noncomputable def time_to_empty_cistern (fill_no_leak_time fill_with_leak_time : ℝ) (filled_cistern : ℝ) : ℝ :=
  let R := filled_cistern / fill_no_leak_time
  let L := (R - filled_cistern / fill_with_leak_time)
  filled_cistern / L

theorem cistern_empty_time :
  time_to_empty_cistern 12 14 1 = 84 :=
by
  unfold time_to_empty_cistern
  simp
  sorry

end cistern_empty_time_l239_239675


namespace mushrooms_weight_change_l239_239100

-- Conditions
variables (x W : ℝ)
variable (initial_weight : ℝ := 100 * x)
variable (dry_weight : ℝ := x)
variable (final_weight_dry : ℝ := 2 * W / 100)

-- Given fresh mushrooms have moisture content of 99%
-- and dried mushrooms have moisture content of 98%
theorem mushrooms_weight_change 
  (h1 : dry_weight = x) 
  (h2 : final_weight_dry = x / 0.02) 
  (h3 : W = x / 0.02) 
  (initial_weight : ℝ := 100 * x) : 
  2 * W = initial_weight / 2 :=
by
  -- This is a placeholder for the proof steps which we skip
  sorry

end mushrooms_weight_change_l239_239100


namespace pandas_increase_l239_239166

theorem pandas_increase 
  (C P : ℕ) -- C: Number of cheetahs 5 years ago, P: Number of pandas 5 years ago
  (h_ratio_5_years_ago : C / P = 1 / 3)
  (h_cheetahs_increase : ∃ z : ℕ, z = 2)
  (h_ratio_now : ∃ k : ℕ, (C + k) / (P + x) = 1 / 3) :
  x = 6 :=
by
  sorry

end pandas_increase_l239_239166


namespace quadratic_equation_solution_l239_239408

theorem quadratic_equation_solution (m : ℝ) :
  (m - 3) * x ^ (m^2 - 7) - x + 3 = 0 → m^2 - 7 = 2 → m ≠ 3 → m = -3 :=
by
  intros h_eq h_power h_nonzero
  sorry

end quadratic_equation_solution_l239_239408


namespace L_shape_area_correct_l239_239290

noncomputable def large_rectangle_area : ℕ := 12 * 7
noncomputable def small_rectangle_area : ℕ := 4 * 3
noncomputable def L_shape_area := large_rectangle_area - small_rectangle_area

theorem L_shape_area_correct : L_shape_area = 72 := by
  -- here goes your solution
  sorry

end L_shape_area_correct_l239_239290


namespace percentage_difference_between_chef_and_dishwasher_l239_239942

theorem percentage_difference_between_chef_and_dishwasher
    (manager_wage : ℝ)
    (dishwasher_wage : ℝ)
    (chef_wage : ℝ)
    (h1 : manager_wage = 6.50)
    (h2 : dishwasher_wage = manager_wage / 2)
    (h3 : chef_wage = manager_wage - 2.60) :
    (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 :=
by
  -- The proof would go here
  sorry

end percentage_difference_between_chef_and_dishwasher_l239_239942


namespace average_class_is_45_6_l239_239096

noncomputable def average_class_score (total_students : ℕ) (top_scorers : ℕ) (top_score : ℕ) 
  (zero_scorers : ℕ) (remaining_students_avg : ℕ) : ℚ :=
  let total_top_score := top_scorers * top_score
  let total_zero_score := zero_scorers * 0
  let remaining_students := total_students - top_scorers - zero_scorers
  let total_remaining_score := remaining_students * remaining_students_avg
  let total_score := total_top_score + total_zero_score + total_remaining_score
  total_score / total_students

theorem average_class_is_45_6 : average_class_score 25 3 95 3 45 = 45.6 := 
by
  -- sorry is used here to skip the proof. Lean will expect a proof here.
  sorry

end average_class_is_45_6_l239_239096


namespace slope_of_given_line_l239_239977

theorem slope_of_given_line : ∀ (x y : ℝ), (4 / x + 5 / y = 0) → (y = (-5 / 4) * x) := 
by 
  intros x y h
  sorry

end slope_of_given_line_l239_239977


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l239_239905

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l239_239905


namespace reciprocal_of_neg_five_l239_239356

theorem reciprocal_of_neg_five : ∃ b : ℚ, (-5) * b = 1 ∧ b = -1/5 :=
by
  sorry

end reciprocal_of_neg_five_l239_239356


namespace calculate_product_N1_N2_l239_239291

theorem calculate_product_N1_N2 : 
  (∃ (N1 N2 : ℝ), 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → 
      (60 * x - 46) / (x^2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) ∧
      N1 * N2 = -1036) :=
  sorry

end calculate_product_N1_N2_l239_239291


namespace largest_number_2013_l239_239742

theorem largest_number_2013 (x y : ℕ) (h1 : x + y = 2013)
    (h2 : y = 5 * (x / 100 + 1)) : max x y = 1913 := by
  sorry

end largest_number_2013_l239_239742


namespace inequality_subtraction_l239_239022

variable (a b : ℝ)

theorem inequality_subtraction (h : a > b) : a - 5 > b - 5 :=
sorry

end inequality_subtraction_l239_239022


namespace value_of_x_y_mn_l239_239311

variables (x y m n : ℝ)

-- Conditions for arithmetic sequence 2, x, y, 3
def arithmetic_sequence_condition_1 : Prop := 2 * x = 2 + y
def arithmetic_sequence_condition_2 : Prop := 2 * y = 3 + x

-- Conditions for geometric sequence 2, m, n, 3
def geometric_sequence_condition_1 : Prop := m^2 = 2 * n
def geometric_sequence_condition_2 : Prop := n^2 = 3 * m

theorem value_of_x_y_mn (h1 : arithmetic_sequence_condition_1 x y) 
                        (h2 : arithmetic_sequence_condition_2 x y) 
                        (h3 : geometric_sequence_condition_1 m n)
                        (h4 : geometric_sequence_condition_2 m n) : 
  x + y + m * n = 11 :=
sorry

end value_of_x_y_mn_l239_239311


namespace max_value_of_x_plus_y_l239_239899

theorem max_value_of_x_plus_y (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : x + y ≤ 5 :=
sorry

end max_value_of_x_plus_y_l239_239899


namespace complex_quadrant_l239_239275

open Complex

-- Let complex number i be the imaginary unit
noncomputable def purely_imaginary (z : ℂ) : Prop := 
  z.re = 0

theorem complex_quadrant (z : ℂ) (a : ℂ) (hz : purely_imaginary z) (h : (2 + I) * z = 1 + a * I ^ 3) :
  (a + z).re > 0 ∧ (a + z).im < 0 :=
by 
  sorry

end complex_quadrant_l239_239275


namespace kernels_popped_in_final_bag_l239_239580

/-- Parker wants to find out what the average percentage of kernels that pop in a bag is.
In the first bag he makes, 60 kernels pop and the bag has 75 kernels.
In the second bag, 42 kernels pop and there are 50 in the bag.
In the final bag, some kernels pop and the bag has 100 kernels.
The average percentage of kernels that pop in a bag is 82%.
How many kernels popped in the final bag?
We prove that given these conditions, the number of popped kernels in the final bag is 82.
-/
noncomputable def kernelsPoppedInFirstBag := 60
noncomputable def totalKernelsInFirstBag := 75
noncomputable def kernelsPoppedInSecondBag := 42
noncomputable def totalKernelsInSecondBag := 50
noncomputable def totalKernelsInFinalBag := 100
noncomputable def averagePoppedPercentage := 82

theorem kernels_popped_in_final_bag (x : ℕ) :
  (kernelsPoppedInFirstBag * 100 / totalKernelsInFirstBag +
   kernelsPoppedInSecondBag * 100 / totalKernelsInSecondBag +
   x * 100 / totalKernelsInFinalBag) / 3 = averagePoppedPercentage →
  x = 82 := 
by
  sorry

end kernels_popped_in_final_bag_l239_239580


namespace volume_of_cylinder_l239_239265

theorem volume_of_cylinder (r h : ℝ) (hr : r = 1) (hh : h = 2) (A : r * h = 4) : (π * r^2 * h = 2 * π) :=
by
  sorry

end volume_of_cylinder_l239_239265


namespace truck_gasoline_rate_l239_239306

theorem truck_gasoline_rate (gas_initial gas_final : ℕ) (dist_supermarket dist_farm_turn dist_farm_final : ℕ) 
    (total_miles gas_used : ℕ) : 
  gas_initial = 12 →
  gas_final = 2 →
  dist_supermarket = 10 →
  dist_farm_turn = 4 →
  dist_farm_final = 6 →
  total_miles = dist_supermarket + dist_farm_turn + dist_farm_final →
  gas_used = gas_initial - gas_final →
  total_miles / gas_used = 2 :=
by sorry

end truck_gasoline_rate_l239_239306


namespace line_equation_problem_l239_239225

theorem line_equation_problem
  (P : ℝ × ℝ)
  (h1 : (P.1 + P.2 - 2 = 0) ∧ (P.1 - P.2 + 4 = 0))
  (l : ℝ × ℝ → Prop)
  (h2 : ∀ A B : ℝ × ℝ, l A → l B → (∃ k, B.2 - A.2 = k * (B.1 - A.1)))
  (h3 : ∀ Q : ℝ × ℝ, l Q → (3 * Q.1 - 2 * Q.2 + 4 = 0)) :
  l P ↔ 3 * P.1 - 2 * P.2 + 9 = 0 := 
sorry

end line_equation_problem_l239_239225


namespace intersection_lines_l239_239515

theorem intersection_lines (a b : ℝ) (h1 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → x = 1/3 * y + a)
                          (h2 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → y = 1/3 * x + b) :
  a + b = 8 / 3 :=
sorry

end intersection_lines_l239_239515


namespace arithmetic_seq_sum_a3_a15_l239_239469

theorem arithmetic_seq_sum_a3_a15 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_eq : a 1 - a 5 + a 9 - a 13 + a 17 = 117) :
  a 3 + a 15 = 234 :=
sorry

end arithmetic_seq_sum_a3_a15_l239_239469


namespace cannot_be_sum_of_six_consecutive_odd_integers_l239_239873

theorem cannot_be_sum_of_six_consecutive_odd_integers (S : ℕ) :
  (S = 90 ∨ S = 150) ->
  ∀ n : ℤ, ¬(S = n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10)) :=
by
  intro h
  intro n
  cases h
  case inl => 
    sorry
  case inr => 
    sorry

end cannot_be_sum_of_six_consecutive_odd_integers_l239_239873


namespace min_max_f_l239_239623

theorem min_max_f (a b x y z t : ℝ) (ha : 0 < a) (hb : 0 < b)
  (hxz : x + z = 1) (hyt : y + t = 1) (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hz : 0 ≤ z) (ht : 0 ≤ t) :
  1 ≤ ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ∧
  ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ≤ 2 :=
sorry

end min_max_f_l239_239623


namespace find_red_chairs_l239_239959

noncomputable def red_chairs := Nat
noncomputable def yellow_chairs := Nat
noncomputable def blue_chairs := Nat

theorem find_red_chairs
    (R Y B : Nat)
    (h1 : Y = 2 * R)
    (h2 : B = Y - 2)
    (h3 : R + Y + B = 18) :
    R = 4 := by
  sorry

end find_red_chairs_l239_239959


namespace combined_weight_l239_239403

theorem combined_weight (mary_weight : ℝ) (jamison_weight : ℝ) (john_weight : ℝ) :
  mary_weight = 160 ∧ jamison_weight = mary_weight + 20 ∧ john_weight = mary_weight + (0.25 * mary_weight) →
  john_weight + mary_weight + jamison_weight = 540 :=
by
  intros h
  obtain ⟨hm, hj, hj'⟩ := h
  rw [hm, hj, hj']
  norm_num
  sorry

end combined_weight_l239_239403


namespace digit_1C3_multiple_of_3_l239_239932

theorem digit_1C3_multiple_of_3 :
  (∃ C : Fin 10, (1 + C.val + 3) % 3 = 0) ∧
  (∀ C : Fin 10, (1 + C.val + 3) % 3 = 0 → (C.val = 2 ∨ C.val = 5 ∨ C.val = 8)) :=
by
  sorry

end digit_1C3_multiple_of_3_l239_239932


namespace max_segment_perimeter_l239_239751

def isosceles_triangle (base height : ℝ) := true -- A realistic definition can define properties of an isosceles triangle

def equal_area_segments (triangle : isosceles_triangle 10 12) (n : ℕ) := true -- A realist definition can define cutting into equal area segments

noncomputable def perimeter_segment (base height : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  1 + Real.sqrt (height^2 + (base / n * k)^2) + Real.sqrt (height^2 + (base / n * (k + 1))^2)

theorem max_segment_perimeter (base height : ℝ) (n : ℕ) (h_base : base = 10) (h_height : height = 12) (h_segments : n = 10) :
  ∃ k, k ∈ Finset.range n ∧ perimeter_segment base height k n = 31.62 :=
by
  sorry

end max_segment_perimeter_l239_239751


namespace contrapositive_example_l239_239483

theorem contrapositive_example (x : ℝ) : (x > 2 → x > 0) ↔ (x ≤ 2 → x ≤ 0) :=
by
  sorry

end contrapositive_example_l239_239483


namespace car_storm_distance_30_l239_239762

noncomputable def car_position (t : ℝ) : ℝ × ℝ :=
  (0, 3/4 * t)

noncomputable def storm_center (t : ℝ) : ℝ × ℝ :=
  (150 - (3/4 / Real.sqrt 2) * t, -(3/4 / Real.sqrt 2) * t)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem car_storm_distance_30 :
  ∃ (t : ℝ), distance (car_position t) (storm_center t) = 30 :=
sorry

end car_storm_distance_30_l239_239762


namespace inequality_proof_l239_239957

variables {a b : ℝ}

theorem inequality_proof :
  a^2 + b^2 - 1 - a^2 * b^2 <= 0 ↔ (a^2 - 1) * (b^2 - 1) >= 0 :=
by sorry

end inequality_proof_l239_239957


namespace relationship_among_neg_a_neg_a3_a2_l239_239940

theorem relationship_among_neg_a_neg_a3_a2 (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 :=
by sorry

end relationship_among_neg_a_neg_a3_a2_l239_239940


namespace probability_blue_face_facing_up_l239_239078

-- Define the context
def octahedron_faces : ℕ := 8
def blue_faces : ℕ := 5
def red_faces : ℕ := 3
def total_faces : ℕ := blue_faces + red_faces

-- The probability calculation theorem
theorem probability_blue_face_facing_up (h : total_faces = octahedron_faces) :
  (blue_faces : ℝ) / (octahedron_faces : ℝ) = 5 / 8 :=
by
  -- Placeholder for proof
  sorry

end probability_blue_face_facing_up_l239_239078


namespace sin_sub_pi_over_3_eq_neg_one_third_l239_239601

theorem sin_sub_pi_over_3_eq_neg_one_third {x : ℝ} (h : Real.cos (x + (π / 6)) = 1 / 3) :
  Real.sin (x - (π / 3)) = -1 / 3 := 
  sorry

end sin_sub_pi_over_3_eq_neg_one_third_l239_239601


namespace cube_volume_l239_239925

theorem cube_volume (length width : ℝ) (h_length : length = 48) (h_width : width = 72) :
  let area := length * width
  let side_length_in_inches := Real.sqrt (area / 6)
  let side_length_in_feet := side_length_in_inches / 12
  let volume := side_length_in_feet ^ 3
  volume = 8 :=
by
  sorry

end cube_volume_l239_239925


namespace quadratic_polynomials_exist_l239_239376

-- Definitions of the polynomials
def p1 (x : ℝ) := (x - 10)^2 - 1
def p2 (x : ℝ) := x^2 - 1
def p3 (x : ℝ) := (x + 10)^2 - 1

-- The theorem to prove
theorem quadratic_polynomials_exist :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ p1 x1 = 0 ∧ p1 x2 = 0) ∧
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ p2 y1 = 0 ∧ p2 y2 = 0) ∧
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ p3 z1 = 0 ∧ p3 z2 = 0) ∧
  (∀ x : ℝ, p1 x + p2 x ≠ 0 ∧ p1 x + p3 x ≠ 0 ∧ p2 x + p3 x ≠ 0) :=
by
  sorry

end quadratic_polynomials_exist_l239_239376


namespace base_eight_seventeen_five_is_one_two_five_l239_239402

def base_eight_to_base_ten (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

theorem base_eight_seventeen_five_is_one_two_five :
  base_eight_to_base_ten 175 = 125 :=
by
  sorry

end base_eight_seventeen_five_is_one_two_five_l239_239402


namespace complex_division_example_l239_239262

-- Given conditions
def i : ℂ := Complex.I

-- The statement we need to prove
theorem complex_division_example : (1 + 3 * i) / (1 + i) = 2 + i :=
by
  sorry

end complex_division_example_l239_239262


namespace Chemistry_marks_l239_239532

theorem Chemistry_marks (english_marks mathematics_marks physics_marks biology_marks : ℕ) (avg_marks : ℝ) (num_subjects : ℕ) (total_marks : ℕ)
  (h1 : english_marks = 72)
  (h2 : mathematics_marks = 60)
  (h3 : physics_marks = 35)
  (h4 : biology_marks = 84)
  (h5 : avg_marks = 62.6)
  (h6 : num_subjects = 5)
  (h7 : total_marks = avg_marks * num_subjects) :
  (total_marks - (english_marks + mathematics_marks + physics_marks + biology_marks) = 62) :=
by
  sorry

end Chemistry_marks_l239_239532


namespace min_n_coloring_property_l239_239286

theorem min_n_coloring_property : ∃ n : ℕ, (∀ (coloring : ℕ → Bool), 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧ coloring a = coloring b ∧ coloring b = coloring c → 2 * a + b = c)) ∧ n = 15 := 
sorry

end min_n_coloring_property_l239_239286


namespace range_of_b_l239_239332

theorem range_of_b (b : ℝ) :
  (∀ x y : ℝ, (x ≠ y) → (y = 1/3 * x^3 + b * x^2 + (b + 2) * x + 3) → (y ≥ 1/3 * x^3 + b * x^2 + (b + 2) * x + 3))
  ↔ (-1 ≤ b ∧ b ≤ 2) :=
sorry

end range_of_b_l239_239332


namespace length_of_train_is_correct_l239_239991

noncomputable def speed_in_m_per_s (speed_in_km_per_hr : ℝ) : ℝ := speed_in_km_per_hr * 1000 / 3600

noncomputable def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

noncomputable def length_of_train (total_distance : ℝ) (length_of_bridge : ℝ) : ℝ := total_distance - length_of_bridge

theorem length_of_train_is_correct :
  ∀ (speed_in_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) (length_of_bridge : ℝ),
  speed_in_km_per_hr = 72 →
  time_to_cross_bridge = 12.199024078073753 →
  length_of_bridge = 134 →
  length_of_train (total_distance (speed_in_m_per_s speed_in_km_per_hr) time_to_cross_bridge) length_of_bridge = 110.98048156147506 :=
by 
  intros speed_in_km_per_hr time_to_cross_bridge length_of_bridge hs ht hl;
  rw [hs, ht, hl];
  sorry

end length_of_train_is_correct_l239_239991


namespace betty_age_l239_239186

def ages (A M B : ℕ) : Prop :=
  A = 2 * M ∧ A = 4 * B ∧ M = A - 22

theorem betty_age (A M B : ℕ) : ages A M B → B = 11 :=
by
  sorry

end betty_age_l239_239186


namespace remainder_correct_l239_239530

noncomputable def P : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^6 
                                  + Polynomial.C 2 * Polynomial.X^5 
                                  - Polynomial.C 3 * Polynomial.X^4 
                                  + Polynomial.C 1 * Polynomial.X^3 
                                  - Polynomial.C 2 * Polynomial.X^2
                                  + Polynomial.C 5 * Polynomial.X 
                                  - Polynomial.C 1

noncomputable def D : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) * 
                                      (Polynomial.X + Polynomial.C 2) * 
                                      (Polynomial.X - Polynomial.C 3)

noncomputable def R : Polynomial ℝ := 17 * Polynomial.X^2 - 52 * Polynomial.X + 38

theorem remainder_correct :
    ∀ (q : Polynomial ℝ), P = D * q + R :=
by sorry

end remainder_correct_l239_239530


namespace triangle_properties_l239_239778

theorem triangle_properties
  (K : ℝ) (α β : ℝ)
  (hK : K = 62.4)
  (hα : α = 70 + 20/60 + 40/3600)
  (hβ : β = 36 + 50/60 + 30/3600) :
  ∃ (a b T : ℝ), 
    a = 16.55 ∧
    b = 30.0 ∧
    T = 260.36 :=
by
  sorry

end triangle_properties_l239_239778


namespace sum_odd_product_even_l239_239748

theorem sum_odd_product_even (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) 
                             (h2 : ∃ m : ℤ, b = 2 * m + 1) 
                             (h3 : ∃ n : ℤ, a + b = 2 * n + 1) : 
  ∃ p : ℤ, a * b = 2 * p := 
  sorry

end sum_odd_product_even_l239_239748


namespace div_polynomial_l239_239461

noncomputable def f (x : ℝ) : ℝ := x^4 + 4*x^3 + 6*x^2 + 4*x + 2
noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ := x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*s*x + t

theorem div_polynomial 
  (p q s t : ℝ) 
  (h : ∀ x : ℝ, f x = 0 → g x p q s t = 0) : 
  (p + q + s) * t = -6 :=
by
  sorry

end div_polynomial_l239_239461


namespace votes_cast_l239_239037

theorem votes_cast (V : ℝ) (h1 : V = 0.33 * V + (0.33 * V + 833)) : V = 2447 := 
by
  sorry

end votes_cast_l239_239037


namespace solve_for_k_l239_239007

theorem solve_for_k (x : ℝ) (k : ℝ) (h₁ : 2 * x - 1 = 3) (h₂ : 3 * x + k = 0) : k = -6 :=
by
  sorry

end solve_for_k_l239_239007


namespace study_group_books_l239_239921

theorem study_group_books (x n : ℕ) (h1 : n = 5 * x - 2) (h2 : n = 4 * x + 3) : x = 5 ∧ n = 23 := by
  sorry

end study_group_books_l239_239921


namespace min_shaded_triangles_l239_239054

-- Definitions (conditions) directly from the problem
def Triangle (n : ℕ) := { x : ℕ // x ≤ n }
def side_length := 8
def smaller_side_length := 1

-- Goal (question == correct answer)
theorem min_shaded_triangles : ∃ (shaded : ℕ), shaded = 15 :=
by {
  sorry
}

end min_shaded_triangles_l239_239054


namespace magic_triangle_largest_S_l239_239871

theorem magic_triangle_largest_S :
  ∃ (S : ℕ) (a b c d e f g : ℕ),
    (10 ≤ a) ∧ (a ≤ 16) ∧
    (10 ≤ b) ∧ (b ≤ 16) ∧
    (10 ≤ c) ∧ (c ≤ 16) ∧
    (10 ≤ d) ∧ (d ≤ 16) ∧
    (10 ≤ e) ∧ (e ≤ 16) ∧
    (10 ≤ f) ∧ (f ≤ 16) ∧
    (10 ≤ g) ∧ (g ≤ 16) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
    (e ≠ f) ∧ (e ≠ g) ∧
    (f ≠ g) ∧
    (S = a + b + c) ∧
    (S = c + d + e) ∧
    (S = e + f + a) ∧
    (S = g + b + c) ∧
    (S = g + d + e) ∧
    (S = g + f + a) ∧
    ((a + b + c) + (c + d + e) + (e + f + a) = 91 - g) ∧
    (S = 26) := sorry

end magic_triangle_largest_S_l239_239871


namespace find_T_l239_239320

theorem find_T (T : ℝ) 
  (h : (1/3) * (1/8) * T = (1/4) * (1/6) * 150) : 
  T = 150 :=
sorry

end find_T_l239_239320


namespace fishing_problem_l239_239510

theorem fishing_problem :
  ∃ (x y : ℕ), 
    (x + y = 70) ∧ 
    (∃ k : ℕ, x = 9 * k) ∧ 
    (∃ m : ℕ, y = 17 * m) ∧ 
    x = 36 ∧ 
    y = 34 := 
by
  sorry

end fishing_problem_l239_239510


namespace sum_squares_bound_l239_239182

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end sum_squares_bound_l239_239182


namespace two_om_2om5_l239_239243

def om (a b : ℕ) : ℕ := a^b - b^a

theorem two_om_2om5 : om 2 (om 2 5) = 79 := by
  sorry

end two_om_2om5_l239_239243


namespace mul_scientific_notation_l239_239606

theorem mul_scientific_notation (a b : ℝ) (c d : ℝ) (h1 : a = 7 * 10⁻¹) (h2 : b = 8 * 10⁻¹) :
  (a * b = 0.56) :=
by
  sorry

end mul_scientific_notation_l239_239606


namespace election_winner_votes_difference_l239_239800

theorem election_winner_votes_difference :
  ∃ W S T F, F = 199 ∧ W = S + 53 ∧ W = T + 79 ∧ W + S + T + F = 979 ∧ (W - F = 105) :=
by
  sorry

end election_winner_votes_difference_l239_239800


namespace correct_proposition_l239_239282

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 + x else 1 - x

def prop_A := ∀ x : ℝ, f (Real.sin x) = -f (Real.sin (-x)) ∧ (∃ T > 0, ∀ x, f (Real.sin (x + T)) = f (Real.sin x))
def prop_B := ∀ x : ℝ, f (Real.sin x) = f (Real.sin (-x)) ∧ ¬(∃ T > 0, ∀ x, f (Real.sin (x + T)) = f (Real.sin x))
def prop_C := ∀ x : ℝ, f (Real.sin (1 / x)) = f (Real.sin (-1 / x)) ∧ ¬(∃ T > 0, ∀ x, f (Real.sin (1 / (x + T))) = f (Real.sin (1 / x)))
def prop_D := ∀ x : ℝ, f (Real.sin (1 / x)) = f (Real.sin (-1 / x)) ∧ (∃ T > 0, ∀ x, f (Real.sin (1 / (x + T))) = f (Real.sin (1 / x)))

theorem correct_proposition :
  (¬ prop_A ∧ ¬ prop_B ∧ prop_C ∧ ¬ prop_D) :=
sorry

end correct_proposition_l239_239282


namespace age_of_first_person_added_l239_239420

theorem age_of_first_person_added :
  ∀ (T A x : ℕ),
    (T = 7 * A) →
    (T + x = 8 * (A + 2)) →
    (T + 15 = 8 * (A - 1)) →
    x = 39 :=
by
  intros T A x h1 h2 h3
  sorry

end age_of_first_person_added_l239_239420


namespace max_min_sum_l239_239039

noncomputable def f : ℝ → ℝ := sorry

-- Define the interval and properties of the function f
def within_interval (x : ℝ) : Prop := -2016 ≤ x ∧ x ≤ 2016
def functional_eq (x1 x2 : ℝ) : Prop := f (x1 + x2) = f x1 + f x2 - 2016
def less_than_2016_proof (x : ℝ) : Prop := x > 0 → f x < 2016

-- Define the minimum and maximum values of the function f
def M : ℝ := sorry
def N : ℝ := sorry

-- Prove that M + N = 4032 given the properties and conditions
theorem max_min_sum : 
  (∀ x1 x2, within_interval x1 → within_interval x2 → functional_eq x1 x2) →
  (∀ x, x > 0 → less_than_2016_proof x) →
  M + N = 4032 :=
by {
  -- Define the formal proof here, placeholder for actual proof
  sorry
}

end max_min_sum_l239_239039


namespace evaluate_expression_evaluate_fraction_l239_239967

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  3 * x^3 + 4 * y^3 = 337 :=
by
  sorry

theorem evaluate_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) 
  (h : 3 * x^3 + 4 * y^3 = 337) :
  (3 * x^3 + 4 * y^3) / 9 = 37 + 4/9 :=
by
  sorry

end evaluate_expression_evaluate_fraction_l239_239967


namespace solve_quadratic_equation_l239_239412

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 6 * x - 3 = 0 ↔ x = 3 + 2 * Real.sqrt 3 ∨ x = 3 - 2 * Real.sqrt 3 :=
by
  sorry

end solve_quadratic_equation_l239_239412


namespace common_ratio_l239_239472

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n, a (n+1) = r * a n)
variable (h1 : a 5 * a 11 = 3)
variable (h2 : a 3 + a 13 = 4)

theorem common_ratio (h_geom : ∀ n, a (n+1) = r * a n) (h1 : a 5 * a 11 = 3) (h2 : a 3 + a 13 = 4) :
  (r = 3 ∨ r = -3) :=
by
  sorry

end common_ratio_l239_239472


namespace minimum_prime_product_l239_239093

noncomputable def is_prime : ℕ → Prop := sorry -- Assume the definition of prime

theorem minimum_prime_product (m n p : ℕ) 
  (hm : is_prime m) 
  (hn : is_prime n) 
  (hp : is_prime p) 
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_sum : m + n = p) : 
  m * n * p = 30 :=
sorry

end minimum_prime_product_l239_239093


namespace polar_eq_parabola_l239_239806

/-- Prove that the curve defined by the polar equation is a parabola. -/
theorem polar_eq_parabola :
  ∀ (r θ : ℝ), r = 1 / (2 * Real.sin θ + Real.cos θ) →
    ∃ (x y : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (x + 2 * y = r^2) :=
by 
  sorry

end polar_eq_parabola_l239_239806


namespace proof_of_acdb_l239_239131

theorem proof_of_acdb
  (x a b c d : ℤ)
  (hx_eq : 7 * x - 8 * x = 20)
  (hx_form : (a + b * Real.sqrt c) / d = x)
  (hints : x = (4 + 2 * Real.sqrt 39) / 7)
  (int_cond : a = 4 ∧ b = 2 ∧ c = 39 ∧ d = 7) :
  a * c * d / b = 546 := by
sorry

end proof_of_acdb_l239_239131


namespace hundredth_power_remainders_l239_239926

theorem hundredth_power_remainders (a : ℤ) : 
  (a % 5 = 0 → a^100 % 125 = 0) ∧ (a % 5 ≠ 0 → a^100 % 125 = 1) :=
by
  sorry

end hundredth_power_remainders_l239_239926


namespace max_area_with_22_matches_l239_239046

-- Definitions based on the conditions
def perimeter := 22

def is_valid_length_width (l w : ℕ) : Prop := l + w = 11

def area (l w : ℕ) : ℕ := l * w

-- Statement of the proof problem
theorem max_area_with_22_matches : 
  ∃ (l w : ℕ), is_valid_length_width l w ∧ (∀ l' w', is_valid_length_width l' w' → area l w ≥ area l' w') ∧ area l w = 30 :=
  sorry

end max_area_with_22_matches_l239_239046


namespace jury_selection_duration_is_two_l239_239564

variable (jury_selection_days : ℕ) (trial_days : ℕ) (deliberation_days : ℕ)

axiom trial_lasts_four_times_jury_selection : trial_days = 4 * jury_selection_days
axiom deliberation_is_six_full_days : deliberation_days = (6 * 24) / 16
axiom john_spends_nineteen_days : jury_selection_days + trial_days + deliberation_days = 19

theorem jury_selection_duration_is_two : jury_selection_days = 2 :=
by
  sorry

end jury_selection_duration_is_two_l239_239564


namespace nth_equation_pattern_l239_239587

theorem nth_equation_pattern (n : ℕ) : 
  (List.range' n (2 * n - 1)).sum = (2 * n - 1) ^ 2 :=
by
  sorry

end nth_equation_pattern_l239_239587


namespace solve_for_a_l239_239032

theorem solve_for_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 16 - 6 * a + a ^ 2) : 
  a = -5 + Real.sqrt 41 ∨ a = -5 - Real.sqrt 41 := by
  sorry

end solve_for_a_l239_239032


namespace compute_r_l239_239031

variables {j p t m n x y r : ℝ}

theorem compute_r
    (h1 : j = 0.75 * p)
    (h2 : j = 0.80 * t)
    (h3 : t = p - r * p / 100)
    (h4 : m = 1.10 * p)
    (h5 : n = 0.70 * m)
    (h6 : j + p + t = m * n)
    (h7 : x = 1.15 * j)
    (h8 : y = 0.80 * n)
    (h9 : x * y = (j + p + t) ^ 2) : r = 6.25 := by
  sorry

end compute_r_l239_239031


namespace range_of_m_l239_239237

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x, (x^2 + 1) * (x^2 - 8 * x - 20) ≤ 0 → (x^2 - 2 * x + (1 - m^2)) ≤ 0) →
  m ≥ 9 := by
  sorry

end range_of_m_l239_239237


namespace principal_amount_l239_239678

theorem principal_amount (A r t : ℝ) (hA : A = 1120) (hr : r = 0.11) (ht : t = 2.4) :
  abs ((A / (1 + r * t)) - 885.82) < 0.01 :=
by
  -- This theorem is stating that given A = 1120, r = 0.11, and t = 2.4,
  -- the principal amount (calculated using the simple interest formula)
  -- is approximately 885.82 with a margin of error less than 0.01.
  sorry

end principal_amount_l239_239678


namespace no_fractions_satisfy_condition_l239_239219

theorem no_fractions_satisfy_condition :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 → Nat.gcd x y = 1 →
    (1.2 : ℚ) * (x : ℚ) / (y : ℚ) = (x + 2 : ℚ) / (y + 2 : ℚ) →
    False :=
by
  intros x y hx hy hrel hcond
  sorry

end no_fractions_satisfy_condition_l239_239219


namespace part_a_part_b_l239_239980

theorem part_a (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 → (3^m - 1) % (2^m) = 0 := by
  sorry

theorem part_b (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 ∨ m = 6 ∨ m = 8 → (31^m - 1) % (2^m) = 0 := by
  sorry

end part_a_part_b_l239_239980


namespace container_capacity_l239_239150

theorem container_capacity 
  (C : ℝ)
  (h1 : 0.75 * C - 0.30 * C = 45) :
  C = 100 := by
  sorry

end container_capacity_l239_239150


namespace proportional_x_y2_y_z2_l239_239985

variable {x y z k m c : ℝ}

theorem proportional_x_y2_y_z2 (h1 : x = k * y^2) (h2 : y = m / z^2) (h3 : x = 2) (hz4 : z = 4) (hz16 : z = 16):
  x = 1/128 :=
by
  sorry

end proportional_x_y2_y_z2_l239_239985


namespace max_largest_int_of_avg_and_diff_l239_239969

theorem max_largest_int_of_avg_and_diff (A B C D E : ℕ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : C ≤ D) (h4 : D ≤ E) 
  (h_avg : (A + B + C + D + E) / 5 = 70) (h_diff : E - A = 10) : E = 340 :=
by
  sorry

end max_largest_int_of_avg_and_diff_l239_239969


namespace choice_of_b_l239_239993

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x - 2)
noncomputable def g (x : ℝ) : ℝ := f (x + 3)

theorem choice_of_b (b : ℝ) :
  (g (g x) = x) ↔ (b = -4) :=
sorry

end choice_of_b_l239_239993


namespace gcd_1043_2295_eq_1_l239_239610

theorem gcd_1043_2295_eq_1 : Nat.gcd 1043 2295 = 1 := by
  sorry

end gcd_1043_2295_eq_1_l239_239610


namespace jewelry_store_total_cost_l239_239523

theorem jewelry_store_total_cost :
  let necklaces_needed := 7
  let rings_needed := 12
  let bracelets_needed := 7
  let necklace_price := 4
  let ring_price := 10
  let bracelet_price := 5
  let necklace_discount := if necklaces_needed >= 6 then 0.15 else if necklaces_needed >= 4 then 0.10 else 0
  let ring_discount := if rings_needed >= 20 then 0.10 else if rings_needed >= 10 then 0.05 else 0
  let bracelet_discount := if bracelets_needed >= 10 then 0.12 else if bracelets_needed >= 7 then 0.08 else 0
  let necklace_cost := necklaces_needed * (necklace_price * (1 - necklace_discount))
  let ring_cost := rings_needed * (ring_price * (1 - ring_discount))
  let bracelet_cost := bracelets_needed * (bracelet_price * (1 - bracelet_discount))
  let total_cost := necklace_cost + ring_cost + bracelet_cost
  total_cost = 170 := by
  -- calculation details omitted
  sorry

end jewelry_store_total_cost_l239_239523


namespace correct_transformation_option_c_l239_239549

theorem correct_transformation_option_c (x : ℝ) (h : (x / 2) - (x / 3) = 1) : 3 * x - 2 * x = 6 :=
by
  sorry

end correct_transformation_option_c_l239_239549


namespace sufficient_not_necessary_condition_l239_239203
open Real

theorem sufficient_not_necessary_condition (m : ℝ) :
  ((m = 0) → ∃ x y : ℝ, (m + 1) * x + (1 - m) * y - 1 = 0 ∧ (m - 1) * x + (2 * m + 1) * y + 4 = 0 ∧ 
  ((m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0 ∨ (m = 1 ∨ m = 0))) :=
by sorry

end sufficient_not_necessary_condition_l239_239203


namespace find_m_l239_239397

theorem find_m 
(x0 m : ℝ)
(h1 : m ≠ 0)
(h2 : x0^2 - x0 + m = 0)
(h3 : (2 * x0)^2 - 2 * x0 + 3 * m = 0)
: m = -2 :=
sorry

end find_m_l239_239397


namespace find_k_l239_239612

def line_p (x y : ℝ) : Prop := y = -2 * x + 3
def line_q (x y k : ℝ) : Prop := y = k * x + 4
def intersection (x y k : ℝ) : Prop := line_p x y ∧ line_q x y k

theorem find_k (k : ℝ) (h_inter : intersection 1 1 k) : k = -3 :=
sorry

end find_k_l239_239612


namespace even_function_x_lt_0_l239_239934

noncomputable def f (x : ℝ) : ℝ :=
if h : x >= 0 then 2^x + 1 else 2^(-x) + 1

theorem even_function_x_lt_0 (x : ℝ) (hx : x < 0) : f x = 2^(-x) + 1 :=
by {
  sorry
}

end even_function_x_lt_0_l239_239934


namespace ten_numbers_property_l239_239849

theorem ten_numbers_property (x : ℕ → ℝ) (h : ∀ i : ℕ, 1 ≤ i → i ≤ 9 → x i + 2 * x (i + 1) = 1) : 
  x 1 + 512 * x 10 = 171 :=
by
  sorry

end ten_numbers_property_l239_239849


namespace no_real_roots_range_k_l239_239571

theorem no_real_roots_range_k (k : ℝ) : (x^2 - 2 * x - k = 0) ∧ (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 := 
by
  sorry

end no_real_roots_range_k_l239_239571


namespace divisor_count_of_45_l239_239208

theorem divisor_count_of_45 : 
  ∃ (n : ℤ), n = 12 ∧ ∀ d : ℤ, d ∣ 45 → (d > 0 ∨ d < 0) := sorry

end divisor_count_of_45_l239_239208


namespace find_range_of_m_l239_239604

-- Statements of the conditions given in the problem
axiom positive_real_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (1 / x + 4 / y = 1)

-- Main statement of the proof problem
theorem find_range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 / x + 4 / y = 1) :
  (∃ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (1 / x + 4 / y = 1) ∧ (x + y / 4 < m^2 - 3 * m)) ↔ (m < -1 ∨ m > 4) := 
sorry

end find_range_of_m_l239_239604


namespace find_fifth_day_income_l239_239164

-- Define the incomes for the first four days
def income_day1 := 45
def income_day2 := 50
def income_day3 := 60
def income_day4 := 65

-- Define the average income over five days
def average_income := 58

-- Expressing the question in terms of a function to determine the fifth day's income
theorem find_fifth_day_income : 
  ∃ (income_day5 : ℕ), 
    (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = average_income 
    ∧ income_day5 = 70 :=
sorry

end find_fifth_day_income_l239_239164


namespace eighteen_mnp_eq_P_np_Q_2mp_l239_239716

theorem eighteen_mnp_eq_P_np_Q_2mp (m n p : ℕ) (P Q : ℕ) (hP : P = 2 ^ m) (hQ : Q = 3 ^ n) :
  18 ^ (m * n * p) = P ^ (n * p) * Q ^ (2 * m * p) :=
by
  sorry

end eighteen_mnp_eq_P_np_Q_2mp_l239_239716


namespace remainder_of_5n_minus_9_l239_239690

theorem remainder_of_5n_minus_9 (n : ℤ) (h : n % 11 = 3) : (5 * n - 9) % 11 = 6 :=
by
  sorry -- Proof is omitted, as per instruction.

end remainder_of_5n_minus_9_l239_239690


namespace values_of_m_l239_239687

theorem values_of_m (m n : ℕ) (hmn : m * n = 900) (hm: m > 1) (hn: n ≥ 1) : 
  (∃ (k : ℕ), ∀ (m : ℕ), (1 < m ∧ (900 / m) ≥ 1 ∧ 900 % m = 0) ↔ k = 25) :=
sorry

end values_of_m_l239_239687


namespace min_seats_occupied_l239_239822

theorem min_seats_occupied (n : ℕ) (h : n = 150) : ∃ k : ℕ, k = 37 ∧ ∀ m : ℕ, m > k → ∃ i : ℕ, i < k ∧ m - k ≥ 2 := sorry

end min_seats_occupied_l239_239822


namespace morning_rowers_count_l239_239911

def number_afternoon_rowers : ℕ := 7
def total_rowers : ℕ := 60

def number_morning_rowers : ℕ :=
  total_rowers - number_afternoon_rowers

theorem morning_rowers_count :
  number_morning_rowers = 53 := by
  sorry

end morning_rowers_count_l239_239911


namespace paper_folding_possible_layers_l239_239289

theorem paper_folding_possible_layers (n : ℕ) : 16 = 2 ^ n :=
by
  sorry

end paper_folding_possible_layers_l239_239289


namespace chiquita_height_l239_239180

theorem chiquita_height (C : ℝ) :
  (C + (C + 2) = 12) → (C = 5) :=
by
  intro h
  sorry

end chiquita_height_l239_239180


namespace coeff_of_linear_term_l239_239144

def quadratic_eqn (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem coeff_of_linear_term :
  ∀ (x : ℝ), (quadratic_eqn x = 0) → (∃ c_b : ℝ, quadratic_eqn x = x^2 + c_b * x + 3 ∧ c_b = -2) :=
by
  sorry

end coeff_of_linear_term_l239_239144


namespace top_card_is_queen_probability_l239_239726

-- Define the conditions of the problem
def standard_deck_size := 52
def number_of_queens := 4

-- Problem statement: The probability that the top card is a Queen
theorem top_card_is_queen_probability : 
  (number_of_queens : ℚ) / standard_deck_size = 1 / 13 := 
sorry

end top_card_is_queen_probability_l239_239726


namespace product_of_digits_l239_239128

theorem product_of_digits (A B : ℕ) (h1 : A + B = 13) (h2 : (10 * A + B) % 4 = 0) : A * B = 42 :=
by
  sorry

end product_of_digits_l239_239128


namespace array_sum_remainder_l239_239954

def entry_value (r c : ℕ) : ℚ :=
  (1 / (2 * 1013) ^ r) * (1 / 1013 ^ c)

def array_sum : ℚ :=
  (1 / (2 * 1013 - 1)) * (1 / (1013 - 1))

def m : ℤ := 1
def n : ℤ := 2046300
def mn_sum : ℤ := m + n

theorem array_sum_remainder :
  (mn_sum % 1013) = 442 :=
by
  sorry

end array_sum_remainder_l239_239954


namespace function_form_l239_239817

theorem function_form (f : ℕ → ℕ) (H : ∀ (x y z : ℕ), x ≠ y → y ≠ z → z ≠ x → (∃ k : ℕ, x + y + z = k^2 ↔ ∃ m : ℕ, f x + f y + f z = m^2)) : ∃ k : ℕ, ∀ n : ℕ, f n = k^2 * n :=
by
  sorry

end function_form_l239_239817


namespace cupboard_cost_price_l239_239316

noncomputable def cost_price_of_cupboard (C : ℝ) : Prop :=
  let SP := 0.88 * C
  let NSP := 1.12 * C
  NSP - SP = 1650

theorem cupboard_cost_price : ∃ (C : ℝ), cost_price_of_cupboard C ∧ C = 6875 := by
  sorry

end cupboard_cost_price_l239_239316


namespace solve_equation_l239_239232

theorem solve_equation :
  ∀ (x : ℝ), 
    x^3 + (Real.log 25 + Real.log 32 + Real.log 53) * x = (Real.log 23 + Real.log 35 + Real.log 52) * x^2 + 1 ↔ 
    x = Real.log 23 ∨ x = Real.log 35 ∨ x = Real.log 52 :=
by
  sorry

end solve_equation_l239_239232


namespace sheila_monthly_savings_l239_239223

-- Define the conditions and the question in Lean
def initial_savings : ℕ := 3000
def family_contribution : ℕ := 7000
def years : ℕ := 4
def final_amount : ℕ := 23248

-- Function to calculate the monthly saving given the conditions
def monthly_savings (initial_savings family_contribution years final_amount : ℕ) : ℕ :=
  (final_amount - (initial_savings + family_contribution)) / (years * 12)

-- The theorem we need to prove in Lean
theorem sheila_monthly_savings :
  monthly_savings initial_savings family_contribution years final_amount = 276 :=
by
  sorry

end sheila_monthly_savings_l239_239223


namespace solution_set_of_inequality_l239_239676

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_2 : f 2 = 1 / 2
axiom f_prime_lt_exp : ∀ x : ℝ, deriv f x < Real.exp x

theorem solution_set_of_inequality :
  {x : ℝ | f x < Real.exp x - 1 / 2} = {x : ℝ | 0 < x} :=
by
  sorry

end solution_set_of_inequality_l239_239676


namespace find_oranges_l239_239739

def A : ℕ := 3
def B : ℕ := 1

theorem find_oranges (O : ℕ) : A + B + O + (A + 4) + 10 * B + 2 * (A + 4) = 39 → O = 4 :=
by 
  intros h
  sorry

end find_oranges_l239_239739


namespace solution_exists_l239_239090

theorem solution_exists (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) :=
by
  sorry

end solution_exists_l239_239090


namespace line_through_center_parallel_to_given_line_l239_239043

def point_in_line (p : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  -a / b

theorem line_through_center_parallel_to_given_line :
  ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -4 ∧
    point_in_line (2, 0) a b c ∧
    slope_of_line a b c = slope_of_line 2 (-1) 1 :=
by
  sorry

end line_through_center_parallel_to_given_line_l239_239043


namespace largest_number_not_sum_of_two_composites_l239_239986

-- Define composite numbers
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define the problem predicate
def cannot_be_expressed_as_sum_of_two_composites (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_number_not_sum_of_two_composites : 
  ∃ n, cannot_be_expressed_as_sum_of_two_composites n ∧ 
  (∀ m, cannot_be_expressed_as_sum_of_two_composites m → m ≤ n) ∧ 
  n = 11 :=
by {
  sorry
}

end largest_number_not_sum_of_two_composites_l239_239986


namespace sum_arithmetic_series_l239_239672

theorem sum_arithmetic_series : 
    let a₁ := 1
    let d := 2
    let n := 9
    let a_n := a₁ + (n - 1) * d
    let S_n := n * (a₁ + a_n) / 2
    a_n = 17 → S_n = 81 :=
by intros
   sorry

end sum_arithmetic_series_l239_239672


namespace distance_between_foci_l239_239948

-- Define the ellipse
def ellipse_eq (x y : ℝ) := 9 * x^2 + 36 * y^2 = 1296

-- Define the semi-major and semi-minor axes
def semi_major_axis := 12
def semi_minor_axis := 6

-- Distance between the foci of the ellipse
theorem distance_between_foci : 
  (∃ x y : ℝ, ellipse_eq x y) → 2 * Real.sqrt (semi_major_axis^2 - semi_minor_axis^2) = 12 * Real.sqrt 3 :=
by
  sorry

end distance_between_foci_l239_239948


namespace man_l239_239505

theorem man's_speed_upstream :
  ∀ (R : ℝ), (R + 1.5 = 11) → (R - 1.5 = 8) :=
by
  intros R h
  sorry

end man_l239_239505


namespace lisa_socks_total_l239_239481

def total_socks (initial : ℕ) (sandra : ℕ) (cousin_ratio : ℕ → ℕ) (mom_extra : ℕ → ℕ) : ℕ :=
  initial + sandra + cousin_ratio sandra + mom_extra initial

def cousin_ratio (sandra : ℕ) : ℕ := sandra / 5
def mom_extra (initial : ℕ) : ℕ := 3 * initial + 8

theorem lisa_socks_total :
  total_socks 12 20 cousin_ratio mom_extra = 80 := by
  sorry

end lisa_socks_total_l239_239481


namespace tapanga_corey_candies_l239_239437

theorem tapanga_corey_candies (corey_candies : ℕ) (tapanga_candies : ℕ) 
                              (h1 : corey_candies = 29) 
                              (h2 : tapanga_candies = corey_candies + 8) : 
                              corey_candies + tapanga_candies = 66 :=
by
  rw [h1, h2]
  sorry

end tapanga_corey_candies_l239_239437


namespace two_digit_sabroso_numbers_l239_239552

theorem two_digit_sabroso_numbers :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ (n + (10 * b + a) = k^2)} =
  {29, 38, 47, 56, 65, 74, 83, 92} :=
sorry

end two_digit_sabroso_numbers_l239_239552


namespace koschei_never_escapes_l239_239384

-- Define a structure for the initial setup
structure Setup where
  koschei_initial_room : Nat -- Initial room of Koschei
  guard_positions : List (Bool) -- Guards' positions, True for West, False for East

-- Example of the required setup:
def initial_setup : Setup :=
  { koschei_initial_room := 1, guard_positions := [true, false, true] }

-- Function to simulate the movement of guards
def move_guards (guards : List Bool) (room : Nat) : List Bool :=
  guards.map (λ g => not g)

-- Function to check if all guards are on the same wall
def all_guards_same_wall (guards : List Bool) : Bool :=
  List.all guards id ∨ List.all guards (λ g => ¬g)

-- Main statement: 
theorem koschei_never_escapes (setup : Setup) :
  ∀ room : Nat, ¬(all_guards_same_wall (move_guards setup.guard_positions room)) :=
  sorry

end koschei_never_escapes_l239_239384


namespace simplify_fraction_l239_239590

theorem simplify_fraction (a b x : ℝ) (h₁ : x = a / b) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (2 * a + b) / (a - 2 * b) = (2 * x + 1) / (x - 2) :=
sorry

end simplify_fraction_l239_239590


namespace tom_catches_48_trout_l239_239294

variable (melanie_tom_catch_ratio : ℕ := 3)
variable (melanie_catch : ℕ := 16)

theorem tom_catches_48_trout (h1 : melanie_catch = 16) (h2 : melanie_tom_catch_ratio = 3) : (melanie_tom_catch_ratio * melanie_catch) = 48 :=
by
  sorry

end tom_catches_48_trout_l239_239294


namespace coords_with_respect_to_origin_l239_239913

/-- Given that the coordinates of point A are (-1, 2), prove that the coordinates of point A
with respect to the origin in the plane rectangular coordinate system xOy are (-1, 2). -/
theorem coords_with_respect_to_origin (A : (ℝ × ℝ)) (h : A = (-1, 2)) : A = (-1, 2) :=
sorry

end coords_with_respect_to_origin_l239_239913


namespace minimum_value_of_quadratic_l239_239349

-- Definition of the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 6 * x + 13

-- Statement of the proof problem
theorem minimum_value_of_quadratic : ∃ (y : ℝ), ∀ x : ℝ, quadratic x >= y ∧ y = 4 := by
  sorry

end minimum_value_of_quadratic_l239_239349


namespace picnic_men_count_l239_239654

variables 
  (M W A C : ℕ)
  (h1 : M + W + C = 200) 
  (h2 : M = W + 20)
  (h3 : A = C + 20)
  (h4 : A = M + W)

theorem picnic_men_count : M = 65 :=
by
  sorry

end picnic_men_count_l239_239654


namespace missed_questions_proof_l239_239466

def num_missed_questions : ℕ := 180

theorem missed_questions_proof (F : ℕ) (h1 : 5 * F + F = 216) : F = 36 ∧ 5 * F = num_missed_questions :=
by {
  sorry
}

end missed_questions_proof_l239_239466


namespace rocky_first_round_knockouts_l239_239111

theorem rocky_first_round_knockouts
  (total_fights : ℕ)
  (knockout_percentage : ℝ)
  (first_round_knockout_percentage : ℝ)
  (h1 : total_fights = 190)
  (h2 : knockout_percentage = 0.50)
  (h3 : first_round_knockout_percentage = 0.20) :
  (total_fights * knockout_percentage * first_round_knockout_percentage = 19) := 
by
  sorry

end rocky_first_round_knockouts_l239_239111


namespace inequality_holds_for_m_l239_239272

theorem inequality_holds_for_m (n : ℕ) (m : ℕ) :
  (∀ a b : ℝ, (0 < a ∧ 0 < b) ∧ (a + b = 2) → (1 / a^n + 1 / b^n ≥ a^m + b^m)) ↔ (m = n ∨ m = n + 1) :=
by
  sorry

end inequality_holds_for_m_l239_239272


namespace ratio_pow_eq_l239_239285

theorem ratio_pow_eq {x y : ℝ} (h : x / y = 7 / 5) : (x^3 / y^2) = 343 / 25 :=
by sorry

end ratio_pow_eq_l239_239285


namespace largest_inscribed_rectangle_area_l239_239061

theorem largest_inscribed_rectangle_area : 
  ∀ (width length : ℝ) (a b : ℝ), 
  width = 8 → length = 12 → 
  (a = (8 / Real.sqrt 3) ∧ b = 2 * a) → 
  (area : ℝ) = (12 * (8 - a)) → 
  area = (96 - 32 * Real.sqrt 3) :=
by
  intros width length a b hw hl htr harea
  sorry

end largest_inscribed_rectangle_area_l239_239061


namespace factorize_expression_l239_239865

theorem factorize_expression (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 :=
by
  sorry

end factorize_expression_l239_239865


namespace pallet_weight_l239_239933

theorem pallet_weight (box_weight : ℕ) (num_boxes : ℕ) (total_weight : ℕ) 
  (h1 : box_weight = 89) (h2 : num_boxes = 3) : total_weight = 267 := by
  sorry

end pallet_weight_l239_239933


namespace standard_equation_of_circle_l239_239668

theorem standard_equation_of_circle :
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ (h - 2) / 2 = k / 1 + 3 / 2 ∧ 
  ((h - 2)^2 + (k + 3)^2 = r^2) ∧ ((h + 2)^2 + (k + 5)^2 = r^2) ∧ 
  h = -1 ∧ k = -2 ∧ r^2 = 10 :=
by
  sorry

end standard_equation_of_circle_l239_239668


namespace solve_for_x_l239_239207

theorem solve_for_x 
  (a b : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P = (3, -1) ∧ (P.2 = 3 + b) ∧ (P.2 = a * 3 + 2)) :
  (a - 1) * 3 = b - 2 :=
by sorry

end solve_for_x_l239_239207


namespace ratio_of_percent_increase_to_decrease_l239_239953

variable (P U V : ℝ)
variable (h1 : P * U = 0.25 * P * V)
variable (h2 : P ≠ 0)

theorem ratio_of_percent_increase_to_decrease (h : U = 0.25 * V) :
  ((V - U) / U) * 100 / 75 = 4 :=
by
  sorry

end ratio_of_percent_increase_to_decrease_l239_239953


namespace fraction_of_time_at_15_mph_l239_239624

theorem fraction_of_time_at_15_mph
  (t1 t2 : ℝ)
  (h : (5 * t1 + 15 * t2) / (t1 + t2) = 10) :
  t2 / (t1 + t2) = 1 / 2 :=
by
  sorry

end fraction_of_time_at_15_mph_l239_239624


namespace Zachary_sold_40_games_l239_239539

theorem Zachary_sold_40_games 
  (R J Z : ℝ)
  (games_Zachary_sold : ℕ)
  (h1 : R = J + 50)
  (h2 : J = 1.30 * Z)
  (h3 : Z = 5 * games_Zachary_sold)
  (h4 : Z + J + R = 770) :
  games_Zachary_sold = 40 :=
by
  sorry

end Zachary_sold_40_games_l239_239539


namespace solve_cubic_equation_l239_239939

theorem solve_cubic_equation :
  ∀ x : ℝ, x^3 = 13 * x + 12 ↔ x = 4 ∨ x = -1 ∨ x = -3 :=
by
  sorry

end solve_cubic_equation_l239_239939


namespace b_share_is_approx_1885_71_l239_239325

noncomputable def investment_problem (x : ℝ) : ℝ := 
  let c_investment := x
  let b_investment := (2 / 3) * c_investment
  let a_investment := 3 * b_investment
  let total_investment := a_investment + b_investment + c_investment
  let b_share := (b_investment / total_investment) * 6600
  b_share

theorem b_share_is_approx_1885_71 (x : ℝ) : abs (investment_problem x - 1885.71) < 0.01 := sorry

end b_share_is_approx_1885_71_l239_239325


namespace price_of_orange_is_60_l239_239727

theorem price_of_orange_is_60
  (x a o : ℕ)
  (h1 : 40 * a + x * o = 540)
  (h2 : a + o = 10)
  (h3 : 40 * a + x * (o - 5) = 240) :
  x = 60 :=
by
  sorry

end price_of_orange_is_60_l239_239727


namespace tyler_brother_age_difference_l239_239079

-- Definitions of Tyler's age and the sum of their ages:
def tyler_age : ℕ := 7
def sum_of_ages (brother_age : ℕ) : Prop := tyler_age + brother_age = 11

-- Proof problem: Prove that Tyler's brother's age minus Tyler's age equals 4 years.
theorem tyler_brother_age_difference (B : ℕ) (h : sum_of_ages B) : B - tyler_age = 4 :=
by
  sorry

end tyler_brother_age_difference_l239_239079


namespace maxwell_distance_l239_239432

-- Define the given conditions
def distance_between_homes : ℝ := 65
def maxwell_speed : ℝ := 2
def brad_speed : ℝ := 3

-- The statement we need to prove
theorem maxwell_distance :
  ∃ (x t : ℝ), 
    x = maxwell_speed * t ∧
    distance_between_homes - x = brad_speed * t ∧
    x = 26 := by sorry

end maxwell_distance_l239_239432


namespace michelle_has_total_crayons_l239_239009

noncomputable def michelle_crayons : ℕ :=
  let type1_crayons_per_box := 5
  let type2_crayons_per_box := 12
  let type1_boxes := 4
  let type2_boxes := 3
  let missing_crayons := 2
  (type1_boxes * type1_crayons_per_box - missing_crayons) + (type2_boxes * type2_crayons_per_box)

theorem michelle_has_total_crayons : michelle_crayons = 54 :=
by
  -- The proof step would go here, but it is omitted according to instructions.
  sorry

end michelle_has_total_crayons_l239_239009


namespace symmetric_line_eq_l239_239781

theorem symmetric_line_eq (x y: ℝ) :
    (∃ (a b: ℝ), 3 * a - b + 2 = 0 ∧ a = 2 - x ∧ b = 2 - y) → 3 * x - y - 6 = 0 :=
by
    intro h
    sorry

end symmetric_line_eq_l239_239781


namespace not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l239_239963

theorem not_divisible_by_5_square_plus_or_minus_1_divisible_by_5 (a : ℤ) (h : a % 5 ≠ 0) :
  (a^2 + 1) % 5 = 0 ∨ (a^2 - 1) % 5 = 0 :=
by
  sorry

end not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l239_239963


namespace swimmers_meet_l239_239743

def time_to_meet (pool_length speed1 speed2 time: ℕ) : ℕ :=
  (time * (speed1 + speed2)) / pool_length

theorem swimmers_meet
  (pool_length : ℕ)
  (speed1 : ℕ)
  (speed2 : ℕ)
  (total_time : ℕ) :
  total_time = 12 * 60 →
  pool_length = 90 →
  speed1 = 3 →
  speed2 = 2 →
  time_to_meet pool_length speed1 speed2 total_time = 20 := by
  sorry

end swimmers_meet_l239_239743


namespace regular_pay_calculation_l239_239088

theorem regular_pay_calculation
  (R : ℝ)  -- defining the regular pay per hour
  (H1 : 40 * R + 20 * R = 180):  -- condition given based on the total actual pay calculation.
  R = 3 := 
by
  -- Skipping the proof
  sorry

end regular_pay_calculation_l239_239088


namespace ramesh_paid_price_l239_239801

-- Define the variables based on the conditions
variable (labelledPrice transportCost installationCost sellingPrice paidPrice : ℝ)

-- Define the specific values given in the problem
def discount : ℝ := 0.20 
def profitRate : ℝ := 0.10 
def actualSellingPrice : ℝ := 24475
def transportAmount : ℝ := 125
def installationAmount : ℝ := 250

-- Define the conditions given in the problem as Lean definitions
def selling_price_no_discount (P : ℝ) : ℝ := (1 + profitRate) * P
def discounted_price (P : ℝ) : ℝ := P * (1 - discount)
def total_cost (P : ℝ) : ℝ :=  discounted_price P + transportAmount + installationAmount

-- The problem is to prove that the price Ramesh paid for the refrigerator is Rs. 18175
theorem ramesh_paid_price : 
  ∀ (labelledPrice : ℝ), 
  selling_price_no_discount labelledPrice = actualSellingPrice → 
  paidPrice = total_cost labelledPrice → 
  paidPrice = 18175 := 
by
  intros labelledPrice h1 h2 
  sorry

end ramesh_paid_price_l239_239801


namespace b_days_solve_l239_239019

-- Definitions from the conditions
variable (b_days : ℝ)
variable (a_rate : ℝ) -- work rate of a
variable (b_rate : ℝ) -- work rate of b

-- Condition 1: a is twice as fast as b
def twice_as_fast_as_b : Prop :=
  a_rate = 2 * b_rate

-- Condition 2: a and b together can complete the work in 3.333333333333333 days
def combined_completion_time : Prop :=
  1 / (a_rate + b_rate) = 10 / 3

-- The number of days b alone can complete the work should satisfy this equation
def b_alone_can_complete_in_b_days : Prop :=
  b_rate = 1 / b_days

-- The actual theorem we want to prove:
theorem b_days_solve (b_rate a_rate : ℝ) (h1 : twice_as_fast_as_b a_rate b_rate) (h2 : combined_completion_time a_rate b_rate) : b_days = 10 :=
by
  sorry

end b_days_solve_l239_239019


namespace first_reduction_percentage_l239_239950

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.6 = P * 0.45 → x = 25 :=
by
  sorry

end first_reduction_percentage_l239_239950


namespace complementary_angle_l239_239050

-- Define the complementary angle condition
def complement (angle : ℚ) := 90 - angle

theorem complementary_angle : complement 30.467 = 59.533 :=
by
  -- Adding sorry to signify the missing proof to ensure Lean builds successfully
  sorry

end complementary_angle_l239_239050


namespace algae_colony_growth_l239_239597

def initial_cells : ℕ := 5
def days : ℕ := 10
def tripling_period : ℕ := 3
def cell_growth_ratio : ℕ := 3

noncomputable def cells_after_n_days (init_cells : ℕ) (day_count : ℕ) (period : ℕ) (growth_ratio : ℕ) : ℕ :=
  let steps := day_count / period
  init_cells * growth_ratio^steps

theorem algae_colony_growth : cells_after_n_days initial_cells days tripling_period cell_growth_ratio = 135 :=
  by sorry

end algae_colony_growth_l239_239597


namespace Luca_weight_loss_per_year_l239_239753

def Barbi_weight_loss_per_month : Real := 1.5
def months_in_a_year : Nat := 12
def Luca_years : Nat := 11
def extra_weight_Luca_lost : Real := 81

theorem Luca_weight_loss_per_year :
  (Barbi_weight_loss_per_month * months_in_a_year + extra_weight_Luca_lost) / Luca_years = 9 := by
  sorry

end Luca_weight_loss_per_year_l239_239753


namespace two_hours_charge_l239_239468

def charge_condition_1 (F A : ℕ) : Prop :=
  F = A + 35

def charge_condition_2 (F A : ℕ) : Prop :=
  F + 4 * A = 350

theorem two_hours_charge (F A : ℕ) (h1 : charge_condition_1 F A) (h2 : charge_condition_2 F A) : 
  F + A = 161 := 
sorry

end two_hours_charge_l239_239468


namespace triangle_angle_bisector_theorem_l239_239020

variable {α : Type*} [LinearOrderedField α]

theorem triangle_angle_bisector_theorem (A B C D : α)
  (h1 : A^2 = (C + D) * (B - (B * D / C)))
  (h2 : B / C = (B * D / C) / D) :
  A^2 = C * B - D * (B * D / C) := 
  by
  sorry

end triangle_angle_bisector_theorem_l239_239020


namespace coupon_savings_difference_l239_239775

theorem coupon_savings_difference {P : ℝ} (hP : P > 200)
  (couponA_savings : ℝ := 0.20 * P) 
  (couponB_savings : ℝ := 50)
  (couponC_savings : ℝ := 0.30 * (P - 200)) :
  (200 ≤ P - 200 + 50 → 200 ≤ P ∧ P ≤ 200 + 400 → 600 - 250 = 350) :=
by
  sorry

end coupon_savings_difference_l239_239775


namespace trapezoid_perimeter_l239_239774

theorem trapezoid_perimeter (a b : ℝ) (h : ∃ c : ℝ, a * b = c^2) :
  ∃ K : ℝ, K = 2 * (a + b + Real.sqrt (a * b)) :=
by
  sorry

end trapezoid_perimeter_l239_239774


namespace quadratic_function_increasing_l239_239260

theorem quadratic_function_increasing (x : ℝ) : ((x - 1)^2 + 2 < (x + 1 - 1)^2 + 2) ↔ (x > 1) := by
  sorry

end quadratic_function_increasing_l239_239260


namespace michael_meets_truck_once_l239_239692

def michael_speed := 5  -- feet per second
def pail_distance := 150  -- feet
def truck_speed := 15  -- feet per second
def truck_stop_time := 20  -- seconds

def initial_michael_position (t : ℕ) : ℕ := t * michael_speed
def initial_truck_position (t : ℕ) : ℕ := pail_distance + t * truck_speed - (t / (truck_speed * truck_stop_time))

def distance (t : ℕ) : ℕ := initial_truck_position t - initial_michael_position t

theorem michael_meets_truck_once :
  ∃ t, (distance t = 0) :=  
sorry

end michael_meets_truck_once_l239_239692


namespace inequality_proof_l239_239504

theorem inequality_proof (a b : Real) (h1 : (1 / a) < (1 / b)) (h2 : (1 / b) < 0) : 
  (b / a) + (a / b) > 2 :=
by
  sorry

end inequality_proof_l239_239504


namespace friends_recycled_pounds_l239_239072

-- Definitions of given conditions
def points_earned : ℕ := 6
def pounds_per_point : ℕ := 8
def zoe_pounds : ℕ := 25

-- Calculation based on given conditions
def total_pounds := points_earned * pounds_per_point
def friends_pounds := total_pounds - zoe_pounds

-- Statement of the proof problem
theorem friends_recycled_pounds : friends_pounds = 23 := by
  sorry

end friends_recycled_pounds_l239_239072


namespace intersecting_circles_l239_239040

theorem intersecting_circles (m c : ℝ)
  (h1 : ∃ (x1 y1 x2 y2 : ℝ), x1 = 1 ∧ y1 = 3 ∧ x2 = m ∧ y2 = 1 ∧ x1 ≠ x2 ∧ y1 ≠ y2)
  (h2 : ∀ (x y : ℝ), (x - y + (c / 2) = 0) → (x = 1 ∨ y = 3)) :
  m + c = 3 :=
sorry

end intersecting_circles_l239_239040


namespace num_positive_integers_which_make_polynomial_prime_l239_239130

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem num_positive_integers_which_make_polynomial_prime :
  (∃! n : ℕ, n > 0 ∧ is_prime (n^3 - 7 * n^2 + 18 * n - 10)) :=
sorry

end num_positive_integers_which_make_polynomial_prime_l239_239130


namespace fruit_juice_conversion_needed_l239_239326

theorem fruit_juice_conversion_needed
  (A_milk_parts B_milk_parts A_fruit_juice_parts B_fruit_juice_parts : ℕ)
  (y : ℕ)
  (x : ℕ)
  (convert_liters : ℕ)
  (A_juice_ratio_milk A_juice_ratio_fruit : ℚ)
  (B_juice_ratio_milk B_juice_ratio_fruit : ℚ) :
  (A_milk_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_milk →
  (A_fruit_juice_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_fruit →
  (B_milk_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_milk →
  (B_fruit_juice_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_fruit →
  (A_juice_ratio_milk * x = A_juice_ratio_fruit * x + y) →
  y = 14 →
  x = 98 :=
by sorry

end fruit_juice_conversion_needed_l239_239326


namespace multiplication_in_S_l239_239912

-- Define the set S as given in the conditions
variable (S : Set ℝ)

-- Condition 1: 1 ∈ S
def condition1 : Prop := 1 ∈ S

-- Condition 2: ∀ a b ∈ S, a - b ∈ S
def condition2 : Prop := ∀ a b : ℝ, a ∈ S → b ∈ S → (a - b) ∈ S

-- Condition 3: ∀ a ∈ S, a ≠ 0 → 1 / a ∈ S
def condition3 : Prop := ∀ a : ℝ, a ∈ S → a ≠ 0 → (1 / a) ∈ S

-- Theorem to prove: ∀ a b ∈ S, ab ∈ S
theorem multiplication_in_S (h1 : condition1 S) (h2 : condition2 S) (h3 : condition3 S) :
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S := 
  sorry

end multiplication_in_S_l239_239912


namespace range_of_m_l239_239626

noncomputable def A := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
noncomputable def B (m : ℝ) := {x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1}

theorem range_of_m (m : ℝ) (h : B m ⊆ A) : -2 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l239_239626


namespace difference_of_sums_1000_l239_239247

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_first_n_odd_not_divisible_by_5 (n : ℕ) : ℕ :=
  (n * n) - 5 * ((n / 5) * ((n / 5) + 1))

theorem difference_of_sums_1000 :
  (sum_first_n_even 1000) - (sum_first_n_odd_not_divisible_by_5 1000) = 51000 :=
by
  sorry

end difference_of_sums_1000_l239_239247


namespace problem_real_numbers_l239_239313

theorem problem_real_numbers (a b c d r : ℝ) 
  (h1 : b + c + d = r * a) 
  (h2 : a + c + d = r * b) 
  (h3 : a + b + d = r * c) 
  (h4 : a + b + c = r * d) : 
  r = 3 ∨ r = -1 :=
sorry

end problem_real_numbers_l239_239313


namespace henry_games_l239_239335

theorem henry_games {N H : ℕ} (hN : N = 7) (hH : H = 4 * N) 
    (h_final: H - 6 = 4 * (N + 6)) : H = 58 :=
by
  -- Proof would be inserted here, but skipped using sorry
  sorry

end henry_games_l239_239335


namespace bryan_travel_hours_per_year_l239_239664

-- Definitions based on the conditions
def minutes_walk_to_bus_station := 5
def minutes_ride_bus := 20
def minutes_walk_to_job := 5
def days_per_year := 365

-- Total time for one-way travel in minutes
def one_way_travel_minutes := minutes_walk_to_bus_station + minutes_ride_bus + minutes_walk_to_job

-- Total daily travel time in minutes
def daily_travel_minutes := one_way_travel_minutes * 2

-- Convert daily travel time from minutes to hours
def daily_travel_hours := daily_travel_minutes / 60

-- Total yearly travel time in hours
def yearly_travel_hours := daily_travel_hours * days_per_year

-- The theorem to prove
theorem bryan_travel_hours_per_year : yearly_travel_hours = 365 :=
by {
  -- The preliminary arithmetic is not the core of the theorem
  sorry
}

end bryan_travel_hours_per_year_l239_239664


namespace find_t_l239_239463

theorem find_t (t : ℝ) (h : (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 3)) : t = -8 := 
by 
  sorry

end find_t_l239_239463


namespace determine_unique_row_weight_free_l239_239574

theorem determine_unique_row_weight_free (t : ℝ) (rows : Fin 10 → ℝ) (unique_row : Fin 10)
  (h_weights_same : ∀ i : Fin 10, i ≠ unique_row → rows i = t) :
  0 = 0 := by
  sorry

end determine_unique_row_weight_free_l239_239574


namespace shepherd_initial_sheep_l239_239158

def sheep_pass_gate (sheep : ℕ) : ℕ :=
  sheep / 2 + 1

noncomputable def shepherd_sheep (initial_sheep : ℕ) : ℕ :=
  (sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate) initial_sheep

theorem shepherd_initial_sheep (initial_sheep : ℕ) (h : shepherd_sheep initial_sheep = 2) :
  initial_sheep = 2 :=
sorry

end shepherd_initial_sheep_l239_239158


namespace number_of_red_socks_l239_239812

-- Definitions:
def red_sock_pairs (R : ℕ) := R
def red_sock_cost (R : ℕ) := 3 * R
def blue_socks_pairs : ℕ := 6
def blue_sock_cost : ℕ := 5
def total_amount_spent := 42

-- Proof Statement
theorem number_of_red_socks (R : ℕ) (h : red_sock_cost R + blue_socks_pairs * blue_sock_cost = total_amount_spent) : 
  red_sock_pairs R = 4 :=
by 
  sorry

end number_of_red_socks_l239_239812


namespace cone_height_l239_239824

theorem cone_height (V : ℝ) (π : ℝ) (r h : ℝ) (sqrt2 : ℝ) :
  V = 9720 * π →
  sqrt2 = Real.sqrt 2 →
  h = r * sqrt2 →
  V = (1/3) * π * r^2 * h →
  h = 38.7 :=
by
  intros
  sorry

end cone_height_l239_239824


namespace find_x_eq_3_plus_sqrt7_l239_239156

variable (x y : ℝ)
variable (h1 : x > y)
variable (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40)
variable (h3 : x * y + x + y = 8)

theorem find_x_eq_3_plus_sqrt7 (h1 : x > y) (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40) (h3 : x * y + x + y = 8) : 
  x = 3 + Real.sqrt 7 :=
sorry

end find_x_eq_3_plus_sqrt7_l239_239156


namespace locus_is_circle_l239_239005

open Complex

noncomputable def circle_center (a b : ℝ) : ℂ := Complex.ofReal (-a / (a^2 + b^2)) + Complex.I * (b / (a^2 + b^2))
noncomputable def circle_radius (a b : ℝ) : ℝ := 1 / Real.sqrt (a^2 + b^2)

theorem locus_is_circle (z0 z1 z : ℂ) (h1 : abs (z1 - z0) = abs z1) (h2 : z0 ≠ 0) (h3 : z1 * z = -1) :
  ∃ (a b : ℝ), z0 = Complex.ofReal a + Complex.I * b ∧
    (∃ c : ℂ, z = c ∧ 
      (c.re + a / (a^2 + b^2))^2 + (c.im - b / (a^2 + b^2))^2 = 1 / (a^2 + b^2)) := by
  sorry

end locus_is_circle_l239_239005


namespace angle_at_7_20_is_100_degrees_l239_239611

def angle_between_hands_at_7_20 : ℝ := 100

theorem angle_at_7_20_is_100_degrees
    (hour_hand_pos : ℝ := 210) -- 7 * 30 degrees
    (minute_hand_pos : ℝ := 120) -- 4 * 30 degrees
    (hour_hand_move_per_minute : ℝ := 0.5) -- 0.5 degrees per minute
    (time_past_7_clock : ℝ := 20) -- 20 minutes
    (adjacent_angle : ℝ := 30) -- angle between adjacent numbers
    : angle_between_hands_at_7_20 = 
      (hour_hand_pos - (minute_hand_pos - hour_hand_move_per_minute * time_past_7_clock)) :=
sorry

end angle_at_7_20_is_100_degrees_l239_239611


namespace chocolate_bar_count_l239_239997

theorem chocolate_bar_count (bar_weight : ℕ) (box_weight : ℕ) (H1 : bar_weight = 125) (H2 : box_weight = 2000) : box_weight / bar_weight = 16 :=
by
  sorry

end chocolate_bar_count_l239_239997


namespace explicit_x_n_formula_l239_239198

theorem explicit_x_n_formula (x y : ℕ → ℕ) (n : ℕ) :
  x 0 = 2 ∧ y 0 = 1 ∧
  (∀ n, x (n + 1) = x n ^ 2 + y n ^ 2) ∧
  (∀ n, y (n + 1) = 2 * x n * y n) →
  x n = (3 ^ (2 ^ n) + 1) / 2 :=
by
  sorry

end explicit_x_n_formula_l239_239198


namespace train_cross_bridge_time_l239_239618

noncomputable def time_to_cross_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (length_of_bridge : ℝ) : ℝ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_mps := speed_kmh * (1000 / 3600)
  total_distance / speed_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 110 72 112 = 11.1 :=
by
  sorry

end train_cross_bridge_time_l239_239618


namespace find_f_2010_l239_239477

def f (x : ℝ) : ℝ := sorry

theorem find_f_2010 (h₁ : ∀ x, f (x + 1) = - f x) (h₂ : f 1 = 4) : f 2010 = -4 :=
by 
  sorry

end find_f_2010_l239_239477


namespace david_overall_average_l239_239663

open Real

noncomputable def english_weighted_average := (74 * 0.20) + (80 * 0.25) + (77 * 0.55)
noncomputable def english_modified := english_weighted_average * 1.5

noncomputable def math_weighted_average := (65 * 0.15) + (75 * 0.25) + (90 * 0.60)
noncomputable def math_modified := math_weighted_average * 2.0

noncomputable def physics_weighted_average := (82 * 0.40) + (85 * 0.60)
noncomputable def physics_modified := physics_weighted_average * 1.2

noncomputable def chemistry_weighted_average := (67 * 0.35) + (89 * 0.65)
noncomputable def chemistry_modified := chemistry_weighted_average * 1.0

noncomputable def biology_weighted_average := (90 * 0.30) + (95 * 0.70)
noncomputable def biology_modified := biology_weighted_average * 1.5

noncomputable def overall_average := (english_modified + math_modified + physics_modified + chemistry_modified + biology_modified) / 5

theorem david_overall_average :
  overall_average = 120.567 :=
by
  -- Proof to be filled in
  sorry

end david_overall_average_l239_239663


namespace solve_problem_l239_239507

open Complex

noncomputable def problem_statement (a : ℝ) : Prop :=
  abs ((a : ℂ) + I) / abs I = 2
  
theorem solve_problem {a : ℝ} : problem_statement a → a = Real.sqrt 3 :=
by
  sorry

end solve_problem_l239_239507


namespace arithmetic_sequence_term_count_l239_239707

theorem arithmetic_sequence_term_count (a d n an : ℕ) (h₀ : a = 5) (h₁ : d = 7) (h₂ : an = 126) (h₃ : an = a + (n - 1) * d) : n = 18 := by
  sorry

end arithmetic_sequence_term_count_l239_239707


namespace divide_segment_mean_proportional_l239_239725

theorem divide_segment_mean_proportional (a : ℝ) (x : ℝ) : 
  ∃ H : ℝ, H > 0 ∧ H < a ∧ H = (a * (Real.sqrt 5 - 1) / 2) :=
sorry

end divide_segment_mean_proportional_l239_239725


namespace sin_45_eq_sqrt2_div_2_l239_239075

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt2_div_2_l239_239075


namespace price_difference_l239_239918

noncomputable def original_price (final_sale_price discount : ℝ) := final_sale_price / (1 - discount)

noncomputable def after_price_increase (price after_increase : ℝ) := price * (1 + after_increase)

theorem price_difference (final_sale_price : ℝ) (discount : ℝ) (price_increase : ℝ) 
    (h1 : final_sale_price = 85) (h2 : discount = 0.15) (h3 : price_increase = 0.25) : 
    after_price_increase final_sale_price price_increase - original_price final_sale_price discount = 6.25 := 
by 
    sorry

end price_difference_l239_239918


namespace picnic_total_persons_l239_239319

-- Definitions based on given conditions
variables (W M A C : ℕ)
axiom cond1 : M = W + 80
axiom cond2 : A = C + 80
axiom cond3 : M = 120

-- Proof problem: Total persons = 240
theorem picnic_total_persons : W + M + A + C = 240 :=
by
  -- Proof will be filled here
  sorry

end picnic_total_persons_l239_239319


namespace fifth_term_geometric_sequence_l239_239133

theorem fifth_term_geometric_sequence (x y : ℚ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x + y
    let a2 := x - y
    let a3 := x / y
    let a4 := x * y
    let r := (x - y)/(x + y)
    (a4 * r = (2 / 3)) :=
by
  -- Proof omitted
  sorry

end fifth_term_geometric_sequence_l239_239133


namespace longer_trip_due_to_red_lights_l239_239163

theorem longer_trip_due_to_red_lights :
  ∀ (num_lights : ℕ) (green_time first_route_base_time red_time_per_light second_route_time : ℕ),
  num_lights = 3 →
  first_route_base_time = 10 →
  red_time_per_light = 3 →
  second_route_time = 14 →
  (first_route_base_time + num_lights * red_time_per_light) - second_route_time = 5 :=
by
  intros num_lights green_time first_route_base_time red_time_per_light second_route_time
  sorry

end longer_trip_due_to_red_lights_l239_239163


namespace deduce_pi_from_cylinder_volume_l239_239183

theorem deduce_pi_from_cylinder_volume 
  (C h V : ℝ) 
  (Circumference : C = 20) 
  (Height : h = 11)
  (VolumeFormula : V = (1 / 12) * C^2 * h) : 
  pi = 3 :=
by 
  -- Carry out the proof
  sorry

end deduce_pi_from_cylinder_volume_l239_239183


namespace number_of_days_A_to_finish_remaining_work_l239_239551

theorem number_of_days_A_to_finish_remaining_work
  (A_days : ℕ) (B_days : ℕ) (B_work_days : ℕ) : 
  A_days = 9 → 
  B_days = 15 → 
  B_work_days = 10 → 
  ∃ d : ℕ, d = 3 :=
by 
  intros hA hB hBw
  sorry

end number_of_days_A_to_finish_remaining_work_l239_239551


namespace least_positive_a_exists_l239_239118

noncomputable def f (x a : ℤ) : ℤ := 5 * x ^ 13 + 13 * x ^ 5 + 9 * a * x

theorem least_positive_a_exists :
  ∃ a : ℕ, (∀ x : ℤ, 65 ∣ f x a) ∧ ∀ b : ℕ, (∀ x : ℤ, 65 ∣ f x b) → a ≤ b :=
sorry

end least_positive_a_exists_l239_239118


namespace shaded_area_l239_239421

theorem shaded_area (r : ℝ) (π : ℝ) (shaded_area : ℝ) (h_r : r = 4) (h_π : π = 3) : shaded_area = 32.5 :=
by
  sorry

end shaded_area_l239_239421


namespace remainder_7547_div_11_l239_239305

theorem remainder_7547_div_11 : 7547 % 11 = 10 :=
by
  sorry

end remainder_7547_div_11_l239_239305


namespace convex_ngon_sides_l239_239452

theorem convex_ngon_sides (n : ℕ) (h : (n * (n - 3)) / 2 = 27) : n = 9 :=
by
  -- Proof omitted
  sorry

end convex_ngon_sides_l239_239452


namespace find_x_l239_239891

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 3) (h2 : y = 27) : x = 108 := by
  sorry

end find_x_l239_239891


namespace relationship_among_a_b_c_l239_239430

theorem relationship_among_a_b_c (a b c : ℝ) (h₁ : a = 0.09) (h₂ : -2 < b ∧ b < -1) (h₃ : 1 < c ∧ c < 2) : b < a ∧ a < c := 
by 
  -- proof will involve but we only need to state this
  sorry

end relationship_among_a_b_c_l239_239430


namespace max_points_of_intersection_l239_239136

open Set

def Point := ℝ × ℝ

structure Circle :=
(center : Point)
(radius : ℝ)

structure Line :=
(coeffs : ℝ × ℝ × ℝ) -- Assume line equation in the form Ax + By + C = 0

def max_intersection_points (circle : Circle) (lines : List Line) : ℕ :=
  let circle_line_intersect_count := 2
  let line_line_intersect_count := 1
  
  let number_of_lines := lines.length
  let pairwise_line_intersections := number_of_lines.choose 2
  
  let circle_and_lines_intersections := circle_line_intersect_count * number_of_lines
  let total_intersections := circle_and_lines_intersections + pairwise_line_intersections

  total_intersections

theorem max_points_of_intersection (c : Circle) (l1 l2 l3 : Line) :
  max_intersection_points c [l1, l2, l3] = 9 :=
by
  sorry

end max_points_of_intersection_l239_239136


namespace boat_speed_in_still_water_l239_239358

theorem boat_speed_in_still_water
  (speed_of_stream : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ)
  (effective_speed : ℝ)
  (boat_speed : ℝ)
  (h1: speed_of_stream = 5)
  (h2: time_downstream = 2)
  (h3: distance_downstream = 54)
  (h4: effective_speed = boat_speed + speed_of_stream)
  (h5: distance_downstream = effective_speed * time_downstream) :
  boat_speed = 22 := by
  sorry

end boat_speed_in_still_water_l239_239358


namespace account_balance_after_transfer_l239_239541

def account_after_transfer (initial_balance transfer_amount : ℕ) : ℕ :=
  initial_balance - transfer_amount

theorem account_balance_after_transfer :
  account_after_transfer 27004 69 = 26935 :=
by
  sorry

end account_balance_after_transfer_l239_239541


namespace calculation_result_l239_239706

theorem calculation_result : 
  (16 = 2^4) → 
  (8 = 2^3) → 
  (4 = 2^2) → 
  (16^6 * 8^3 / 4^10 = 8192) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end calculation_result_l239_239706


namespace solve_2xx_eq_sqrt2_unique_solution_l239_239304

noncomputable def solve_equation_2xx_eq_sqrt2 (x : ℝ) : Prop :=
  2 * x^x = Real.sqrt 2

theorem solve_2xx_eq_sqrt2_unique_solution (x : ℝ) : solve_equation_2xx_eq_sqrt2 x ↔ (x = 1/2 ∨ x = 1/4) ∧ x > 0 :=
by
  sorry

end solve_2xx_eq_sqrt2_unique_solution_l239_239304


namespace rachel_math_homework_l239_239966

def rachel_homework (M : ℕ) (reading : ℕ) (biology : ℕ) (total : ℕ) : Prop :=
reading = 3 ∧ biology = 10 ∧ total = 15 ∧ reading + biology + M = total

theorem rachel_math_homework: ∃ M : ℕ, rachel_homework M 3 10 15 ∧ M = 2 := 
by 
  sorry

end rachel_math_homework_l239_239966


namespace lines_parallel_l239_239458

/--
Given two lines represented by the equations \(2x + my - 2m + 4 = 0\) and \(mx + 2y - m + 2 = 0\), 
prove that the value of \(m\) that makes these two lines parallel is \(m = -2\).
-/
theorem lines_parallel (m : ℝ) : 
    (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0) ∧ (∀ x y : ℝ, m * x + 2 * y - m + 2 = 0) 
    → m = -2 :=
by
  sorry

end lines_parallel_l239_239458


namespace walter_time_spent_at_seals_l239_239345

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end walter_time_spent_at_seals_l239_239345


namespace emma_total_investment_l239_239454

theorem emma_total_investment (X : ℝ) (h : 0.09 * 6000 + 0.11 * (X - 6000) = 980) : X = 10000 :=
sorry

end emma_total_investment_l239_239454


namespace Maria_green_towels_l239_239212

-- Definitions
variable (G : ℕ) -- number of green towels

-- Conditions
def initial_towels := G + 21
def final_towels := initial_towels - 34

-- Theorem statement
theorem Maria_green_towels : final_towels = 22 → G = 35 :=
by
  sorry

end Maria_green_towels_l239_239212


namespace find_x_plus_inv_x_l239_239077

theorem find_x_plus_inv_x (x : ℝ) (hx_pos : 0 < x) (h : x^10 + x^5 + 1/x^5 + 1/x^10 = 15250) :
  x + 1/x = 3 :=
by
  sorry

end find_x_plus_inv_x_l239_239077


namespace value_of_2_star_3_l239_239066

def star (a b : ℕ) : ℕ := a * b ^ 3 - b + 2

theorem value_of_2_star_3 : star 2 3 = 53 :=
by
  -- This is where the proof would go
  sorry

end value_of_2_star_3_l239_239066


namespace rentExpenses_l239_239741

noncomputable def monthlySalary : ℝ := 23000
noncomputable def milkExpenses : ℝ := 1500
noncomputable def groceriesExpenses : ℝ := 4500
noncomputable def educationExpenses : ℝ := 2500
noncomputable def petrolExpenses : ℝ := 2000
noncomputable def miscellaneousExpenses : ℝ := 5200
noncomputable def savings : ℝ := 2300

-- Calculating total non-rent expenses
noncomputable def totalNonRentExpenses : ℝ :=
  milkExpenses + groceriesExpenses + educationExpenses + petrolExpenses + miscellaneousExpenses

-- The rent expenses theorem
theorem rentExpenses : totalNonRentExpenses + savings + 5000 = monthlySalary :=
by sorry

end rentExpenses_l239_239741


namespace solve_for_x_l239_239862

theorem solve_for_x (x : ℝ) (h : 1 - 1 / (1 - x) ^ 3 = 1 / (1 - x)) : x = 1 :=
sorry

end solve_for_x_l239_239862


namespace customer_purchases_90_percent_l239_239746

variable (P Q : ℝ) 

theorem customer_purchases_90_percent (price_increase_expenditure_diff : 
  (1.25 * P * R / 100 * Q = 1.125 * P * Q)) : 
  R = 90 := 
by 
  sorry

end customer_purchases_90_percent_l239_239746


namespace melanie_initial_plums_l239_239476

-- define the conditions as constants
def plums_given_to_sam : ℕ := 3
def plums_left_with_melanie : ℕ := 4

-- define the statement to be proven
theorem melanie_initial_plums : (plums_given_to_sam + plums_left_with_melanie = 7) :=
by
  sorry

end melanie_initial_plums_l239_239476


namespace pipe_filling_time_l239_239095

/-- 
A problem involving two pipes filling and emptying a tank. 
Time taken for the first pipe to fill the tank is proven to be 16.8 minutes.
-/
theorem pipe_filling_time :
  ∃ T : ℝ, (∀ T, let r1 := 1 / T
                let r2 := 1 / 24
                let time_both_pipes_open := 36
                let time_first_pipe_only := 6
                (r1 - r2) * time_both_pipes_open + r1 * time_first_pipe_only = 1) ∧
           T = 16.8 :=
by
  sorry

end pipe_filling_time_l239_239095


namespace percentage_green_shirts_correct_l239_239647

variable (total_students blue_percentage red_percentage other_students : ℕ)

noncomputable def percentage_green_shirts (total_students blue_percentage red_percentage other_students : ℕ) : ℕ :=
  let total_blue_shirts := blue_percentage * total_students / 100
  let total_red_shirts := red_percentage * total_students / 100
  let total_blue_red_other_shirts := total_blue_shirts + total_red_shirts + other_students
  let green_shirts := total_students - total_blue_red_other_shirts
  (green_shirts * 100) / total_students

theorem percentage_green_shirts_correct
  (h1 : total_students = 800) 
  (h2 : blue_percentage = 45)
  (h3 : red_percentage = 23)
  (h4 : other_students = 136) : 
  percentage_green_shirts total_students blue_percentage red_percentage other_students = 15 :=
by
  sorry

end percentage_green_shirts_correct_l239_239647


namespace boys_collected_in_all_l239_239712

-- Definition of the problem’s conditions
variables (solomon juwan levi : ℕ)

-- Given conditions as assumptions
def conditions : Prop :=
  solomon = 66 ∧
  solomon = 3 * juwan ∧
  levi = juwan / 2

-- Total cans collected by all boys
def total_cans (solomon juwan levi : ℕ) : ℕ := solomon + juwan + levi

theorem boys_collected_in_all : ∃ solomon juwan levi : ℕ, 
  conditions solomon juwan levi ∧ total_cans solomon juwan levi = 99 :=
by {
  sorry
}

end boys_collected_in_all_l239_239712


namespace simplify_expression_l239_239038

theorem simplify_expression :
  (0.7264 * 0.4329 * 0.5478) + (0.1235 * 0.3412 * 0.6214) - ((0.1289 * 0.5634 * 0.3921) / (0.3785 * 0.4979 * 0.2884)) - (0.2956 * 0.3412 * 0.6573) = -0.3902 :=
by
  sorry

end simplify_expression_l239_239038


namespace train_speed_is_126_kmh_l239_239252

noncomputable def train_speed_proof : Prop :=
  let length_meters := 560 / 1000           -- Convert length to kilometers
  let time_hours := 16 / 3600               -- Convert time to hours
  let speed := length_meters / time_hours   -- Calculate the speed
  speed = 126                               -- The speed should be 126 km/h

theorem train_speed_is_126_kmh : train_speed_proof := by 
  sorry

end train_speed_is_126_kmh_l239_239252


namespace factorize_polynomial_l239_239030

theorem factorize_polynomial :
  ∀ (x : ℝ), x^4 + 2021 * x^2 + 2020 * x + 2021 = (x^2 + x + 1) * (x^2 - x + 2021) :=
by
  intros x
  sorry

end factorize_polynomial_l239_239030


namespace part1_solution_part2_solution_l239_239696

def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

theorem part1_solution (x : ℝ) : 
  f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := sorry

theorem part2_solution (a : ℝ) :
  (∃ x : ℝ, f x a < 2 * a) ↔ 3 < a := sorry

end part1_solution_part2_solution_l239_239696


namespace ratio_sequences_l239_239197

-- Define positive integers n and k, with k >= n and k - n even.
variables {n k : ℕ} (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0)

-- Define the sets S_N and S_M
def S_N (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_N
def S_M (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_M

-- Main theorem: N / M = 2^(k - n)
theorem ratio_sequences (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0) :
  (S_N n k : ℝ) / (S_M n k : ℝ) = 2^(k - n) := sorry

end ratio_sequences_l239_239197


namespace accurate_scale_l239_239837

-- Definitions for the weights on each scale
variables (a b c d e x : ℝ)

-- Given conditions
def condition1 := c = b - 0.3
def condition2 := d = c - 0.1
def condition3 := e = a - 0.1
def condition4 := c = e - 0.1
def condition5 := 5 * x = a + b + c + d + e

-- Proof statement
theorem accurate_scale 
  (h1 : c = b - 0.3)
  (h2 : d = c - 0.1)
  (h3 : e = a - 0.1)
  (h4 : c = e - 0.1)
  (h5 : 5 * x = a + b + c + d + e) : e = x :=
by
  sorry

end accurate_scale_l239_239837


namespace train_speed_with_coaches_l239_239745

theorem train_speed_with_coaches (V₀ : ℝ) (V₉ V₁₆ : ℝ) (k : ℝ) :
  V₀ = 30 → V₁₆ = 14 → V₉ = 30 - k * (9: ℝ) ^ (1/2: ℝ) ∧ V₁₆ = 30 - k * (16: ℝ) ^ (1/2: ℝ) →
  V₉ = 18 :=
by sorry

end train_speed_with_coaches_l239_239745


namespace negative_values_count_l239_239374

theorem negative_values_count (n : ℕ) : (n < 13) → (n^2 < 150) → ∃ (k : ℕ), k = 12 :=
by
  sorry

end negative_values_count_l239_239374


namespace find_range_of_x_l239_239049

variable (f : ℝ → ℝ) (x : ℝ)

-- Assume f is an increasing function on [-1, 1]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x ≤ y → f x ≤ f y

-- Main theorem statement based on the problem
theorem find_range_of_x (h_increasing : is_increasing_on_interval f (-1) 1)
                        (h_condition : f (x - 1) < f (1 - 3 * x)) :
  0 ≤ x ∧ x < (1 / 2) :=
sorry

end find_range_of_x_l239_239049


namespace math_problem_l239_239441

theorem math_problem (a b c m n : ℝ)
  (h1 : a = -b)
  (h2 : c = -1)
  (h3 : m * n = 1) : 
  (a + b) / 3 + c^2 - 4 * m * n = -3 := 
by 
  -- Proof steps would be here
  sorry

end math_problem_l239_239441


namespace factor_expression_l239_239757

theorem factor_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b)^2 * (b - c)^2 * (c - a)^2 * (a + b + c) :=
sorry

end factor_expression_l239_239757


namespace household_member_count_l239_239563

variable (M : ℕ) -- the number of members in the household

-- Conditions
def slices_per_breakfast := 3
def slices_per_snack := 2
def slices_per_member_daily := slices_per_breakfast + slices_per_snack
def slices_per_loaf := 12
def loaves_last_days := 3
def loaves_given := 5
def total_slices := slices_per_loaf * loaves_given
def daily_consumption := total_slices / loaves_last_days

-- Proof statement
theorem household_member_count : daily_consumption = slices_per_member_daily * M → M = 4 :=
by
  sorry

end household_member_count_l239_239563


namespace value_of_x_plus_2y_l239_239470

theorem value_of_x_plus_2y :
  let x := 3
  let y := 1
  x + 2 * y = 5 :=
by
  sorry

end value_of_x_plus_2y_l239_239470


namespace exists_valid_circle_group_l239_239220

variable {P : Type}
variable (knows : P → P → Prop)

def knows_at_least_three (p : P) : Prop :=
  ∃ (p₁ p₂ p₃ : P), p₁ ≠ p ∧ p₂ ≠ p ∧ p₃ ≠ p ∧ knows p p₁ ∧ knows p p₂ ∧ knows p p₃

def valid_circle_group (G : List P) : Prop :=
  (2 < G.length) ∧ (G.length % 2 = 0) ∧ (∀ i, knows (G.nthLe i sorry) (G.nthLe ((i + 1) % G.length) sorry) ∧ knows (G.nthLe i sorry) (G.nthLe ((i - 1 + G.length) % G.length) sorry))

theorem exists_valid_circle_group (H : ∀ p : P, knows_at_least_three knows p) : 
  ∃ G : List P, valid_circle_group knows G := 
sorry

end exists_valid_circle_group_l239_239220


namespace no_real_b_for_inequality_l239_239694

theorem no_real_b_for_inequality : ¬ ∃ b : ℝ, (∃ x : ℝ, |x^2 + 3 * b * x + 4 * b| = 5 ∧ ∀ y : ℝ, y ≠ x → |y^2 + 3 * b * y + 4 * b| > 5) := sorry

end no_real_b_for_inequality_l239_239694


namespace five_equal_angles_72_degrees_l239_239521

theorem five_equal_angles_72_degrees
  (five_rays : ℝ)
  (equal_angles : ℝ) 
  (sum_angles : five_rays * equal_angles = 360) :
  equal_angles = 72 :=
by
  sorry

end five_equal_angles_72_degrees_l239_239521


namespace total_animals_seen_l239_239870

theorem total_animals_seen (lions_sat : ℕ) (elephants_sat : ℕ) 
                           (buffaloes_sun : ℕ) (leopards_sun : ℕ)
                           (rhinos_mon : ℕ) (warthogs_mon : ℕ) 
                           (h_sat : lions_sat = 3 ∧ elephants_sat = 2)
                           (h_sun : buffaloes_sun = 2 ∧ leopards_sun = 5)
                           (h_mon : rhinos_mon = 5 ∧ warthogs_mon = 3) :
  lions_sat + elephants_sat + buffaloes_sun + leopards_sun + rhinos_mon + warthogs_mon = 20 := by
  sorry

end total_animals_seen_l239_239870


namespace tan_150_deg_l239_239085

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - Real.sqrt 3 / 3 := by
  sorry

end tan_150_deg_l239_239085


namespace jims_investment_l239_239699

theorem jims_investment
  {total_investment : ℝ} 
  (h1 : total_investment = 127000)
  {john_ratio : ℕ} 
  (h2 : john_ratio = 8)
  {james_ratio : ℕ} 
  (h3 : james_ratio = 11)
  {jim_ratio : ℕ} 
  (h4 : jim_ratio = 15)
  {jordan_ratio : ℕ} 
  (h5 : jordan_ratio = 19) :
  jim_ratio / (john_ratio + james_ratio + jim_ratio + jordan_ratio) * total_investment = 35943.40 :=
by {
  sorry
}

end jims_investment_l239_239699


namespace triangle_area_l239_239595

theorem triangle_area (a b : ℝ) (sinC sinA : ℝ) 
  (h1 : a = Real.sqrt 5) 
  (h2 : b = 3) 
  (h3 : sinC = 2 * sinA) : 
  ∃ (area : ℝ), area = 3 := 
by 
  sorry

end triangle_area_l239_239595


namespace polynomial_divisibility_l239_239073

theorem polynomial_divisibility (C D : ℝ) (h : ∀ (ω : ℂ), ω^2 + ω + 1 = 0 → (ω^106 + C * ω + D = 0)) : C + D = -1 :=
by
  -- Add proof here
  sorry

end polynomial_divisibility_l239_239073


namespace asymptote_hyperbola_condition_l239_239486

theorem asymptote_hyperbola_condition : 
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = 4/3 * x ∨ y = -4/3 * x)) ∧
  ¬(∀ x y : ℝ, (y = 4/3 * x ∨ y = -4/3 * x → x^2 / 9 - y^2 / 16 = 1)) :=
by sorry

end asymptote_hyperbola_condition_l239_239486


namespace math_problem_l239_239474

theorem math_problem (a b : ℝ) (h : Real.sqrt (a + 2) + |b - 1| = 0) : (a + b) ^ 2023 = -1 := 
by
  sorry

end math_problem_l239_239474


namespace polynomial_remainder_l239_239367

noncomputable def h (x : ℕ) := x^5 + x^4 + x^3 + x^2 + x + 1

theorem polynomial_remainder (x : ℕ) : (h (x^10)) % (h x) = 5 :=
sorry

end polynomial_remainder_l239_239367


namespace tan_585_eq_one_l239_239923

theorem tan_585_eq_one : Real.tan (585 * Real.pi / 180) = 1 :=
by
  sorry

end tan_585_eq_one_l239_239923


namespace jade_pieces_left_l239_239784

-- Define the initial number of pieces Jade has
def initial_pieces : Nat := 100

-- Define the number of pieces per level
def pieces_per_level : Nat := 7

-- Define the number of levels in the tower
def levels : Nat := 11

-- Define the resulting number of pieces Jade has left after building the tower
def pieces_left : Nat := initial_pieces - (pieces_per_level * levels)

-- The theorem stating that after building the tower, Jade has 23 pieces left
theorem jade_pieces_left : pieces_left = 23 := by
  -- Proof omitted
  sorry

end jade_pieces_left_l239_239784


namespace fourth_angle_of_quadrilateral_l239_239076

theorem fourth_angle_of_quadrilateral (A : ℝ) : 
  (120 + 85 + 90 + A = 360) ↔ A = 65 := 
by
  sorry

end fourth_angle_of_quadrilateral_l239_239076


namespace calculate_bubble_bath_needed_l239_239583

theorem calculate_bubble_bath_needed :
  let double_suites_capacity := 5 * 4
  let rooms_for_couples_capacity := 13 * 2
  let single_rooms_capacity := 14 * 1
  let family_rooms_capacity := 3 * 6
  let total_guests := double_suites_capacity + rooms_for_couples_capacity + single_rooms_capacity + family_rooms_capacity
  let bubble_bath_per_guest := 25
  total_guests * bubble_bath_per_guest = 1950 := by
  let double_suites_capacity := 5 * 4
  let rooms_for_couples_capacity := 13 * 2
  let single_rooms_capacity := 14 * 1
  let family_rooms_capacity := 3 * 6
  let total_guests := double_suites_capacity + rooms_for_couples_capacity + single_rooms_capacity + family_rooms_capacity
  let bubble_bath_per_guest := 25
  sorry

end calculate_bubble_bath_needed_l239_239583


namespace real_function_as_sum_of_symmetric_graphs_l239_239920

theorem real_function_as_sum_of_symmetric_graphs (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ), (∀ x, g x + h x = f x) ∧ (∀ x, g x = g (-x)) ∧ (∀ x, h (1 + x) = h (1 - x)) :=
sorry

end real_function_as_sum_of_symmetric_graphs_l239_239920


namespace modulus_product_l239_239447

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end modulus_product_l239_239447


namespace tournament_committee_count_l239_239490

theorem tournament_committee_count :
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  total_choices = 11568055296 := 
by {
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  have h_total_choices_eq : total_choices = 11568055296 := sorry
  exact h_total_choices_eq
}

end tournament_committee_count_l239_239490


namespace function_property_l239_239391

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem function_property
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_property : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0)
  : f (-4) > f (-6) :=
sorry

end function_property_l239_239391


namespace matrix_power_eq_l239_239042

def MatrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-8, -10]]

def MatrixA : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![201, 200], ![-400, -449]]

theorem matrix_power_eq :
  MatrixC ^ 50 = MatrixA := 
  sorry

end matrix_power_eq_l239_239042


namespace negation_universal_proposition_l239_239245

theorem negation_universal_proposition : 
  (¬ ∀ x : ℝ, x^2 - x < 0) = ∃ x : ℝ, x^2 - x ≥ 0 :=
by
  sorry

end negation_universal_proposition_l239_239245


namespace find_t_l239_239514

noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ :=
  x^4 + p*x^3 + q*x^2 + s*x + t

theorem find_t {p q s t : ℝ}
  (h1 : ∀ r : ℝ, g r p q s t = 0 → r < 0 ∧ Int.mod (round r) 2 = 1)
  (h2 : p + q + s + t = 2047) :
  t = 5715 :=
sorry

end find_t_l239_239514


namespace solve_equation_l239_239123

theorem solve_equation :
  let lhs := ((4 - 3.5 * (15/7 - 6/5)) / 0.16)
  let rhs := ((23/7 - (3/14) / (1/6)) / (3467/84 - 2449/60))
  lhs / 1 = rhs :=
by
  sorry

end solve_equation_l239_239123


namespace prime_sum_divisible_l239_239518

theorem prime_sum_divisible (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q = p + 2) :
  (p ^ q + q ^ p) % (p + q) = 0 :=
by
  sorry

end prime_sum_divisible_l239_239518


namespace tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l239_239547

-- First problem: Given condition and expression to be proved equal to the correct answer.
theorem tan_alpha_eq_two_and_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * Real.pi - α) + Real.cos (Real.pi + α)) / 
  (Real.cos (α - Real.pi) - Real.cos (3 * Real.pi / 2 - α)) = -3 := sorry

-- Second problem: Given expression to be proved simplified to the correct answer.
theorem sin_tan_simplify :
  Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi/180)) = 1 := sorry

end tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l239_239547


namespace sum_of_palindromes_l239_239994

theorem sum_of_palindromes (a b : ℕ) (ha : a > 99) (ha' : a < 1000) (hb : b > 99) (hb' : b < 1000) 
  (hpal_a : ∀ i j k, a = 100*i + 10*j + k → a = 100*k + 10*j + i) 
  (hpal_b : ∀ i j k, b = 100*i + 10*j + k → b = 100*k + 10*j + i) 
  (hprod : a * b = 589185) : a + b = 1534 :=
sorry

end sum_of_palindromes_l239_239994


namespace remove_wallpaper_time_l239_239584

theorem remove_wallpaper_time 
    (total_walls : ℕ := 8)
    (remaining_walls : ℕ := 7)
    (time_for_remaining_walls : ℕ := 14) :
    time_for_remaining_walls / remaining_walls = 2 :=
by
sorry

end remove_wallpaper_time_l239_239584


namespace solve_for_q_l239_239492

theorem solve_for_q
  (n m q : ℚ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m - n) / 66)
  (h3 : 5 / 6 = (q - m) / 150) :
  q = 230 :=
by
  sorry

end solve_for_q_l239_239492


namespace larger_number_l239_239440

/-- The difference of two numbers is 1375 and the larger divided by the smaller gives a quotient of 6 and a remainder of 15. 
Prove that the larger number is 1647. -/
theorem larger_number (L S : ℕ) 
  (h1 : L - S = 1375) 
  (h2 : L = 6 * S + 15) : 
  L = 1647 := 
sorry

end larger_number_l239_239440


namespace correct_statements_count_l239_239903

-- Definitions for each condition
def is_output_correct (stmt : String) : Prop :=
  stmt = "PRINT a, b, c"

def is_input_correct (stmt : String) : Prop :=
  stmt = "INPUT \"x=3\""

def is_assignment_correct_1 (stmt : String) : Prop :=
  stmt = "A=3"

def is_assignment_correct_2 (stmt : String) : Prop :=
  stmt = "A=B ∧ B=C"

-- The main theorem to be proven
theorem correct_statements_count (stmt1 stmt2 stmt3 stmt4 : String) :
  stmt1 = "INPUT a, b, c" → stmt2 = "INPUT x=3" → stmt3 = "3=A" → stmt4 = "A=B=C" →
  (¬ is_output_correct stmt1 ∧ ¬ is_input_correct stmt2 ∧ ¬ is_assignment_correct_1 stmt3 ∧ ¬ is_assignment_correct_2 stmt4) →
  0 = 0 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end correct_statements_count_l239_239903


namespace total_trolls_l239_239735

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end total_trolls_l239_239735


namespace height_difference_l239_239519

variable (h_A h_B h_D h_E h_F h_G : ℝ)

theorem height_difference :
  (h_A - h_D = 4.5) →
  (h_E - h_D = -1.7) →
  (h_F - h_E = -0.8) →
  (h_G - h_F = 1.9) →
  (h_B - h_G = 3.6) →
  (h_A - h_B > 0) :=
by
  intro h_AD h_ED h_FE h_GF h_BG
  sorry

end height_difference_l239_239519


namespace det_of_matrix_l239_239616

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem det_of_matrix :
  determinant_2x2 5 (-2) 3 1 = 11 := by
  sorry

end det_of_matrix_l239_239616


namespace binom_9_5_eq_126_l239_239205

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l239_239205


namespace average_student_headcount_l239_239763

theorem average_student_headcount :
  let count_0304 := 10500
  let count_0405 := 10700
  let count_0506 := 11300
  let total_count := count_0304 + count_0405 + count_0506
  let number_of_terms := 3
  let average := total_count / number_of_terms
  average = 10833 :=
by
  sorry

end average_student_headcount_l239_239763


namespace tangent_line_of_ellipse_l239_239651

theorem tangent_line_of_ellipse 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (x₀ y₀ : ℝ) (hx₀ : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y : ℝ, x₀ * x / a^2 + y₀ * y / b^2 = 1 := 
sorry

end tangent_line_of_ellipse_l239_239651


namespace range_of_a_l239_239947

noncomputable def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

theorem range_of_a {a : ℝ} : is_monotonic (f a) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end range_of_a_l239_239947


namespace sara_initial_quarters_l239_239878

theorem sara_initial_quarters (total_quarters dad_gift initial_quarters : ℕ) (h1 : dad_gift = 49) (h2 : total_quarters = 70) (h3 : total_quarters = initial_quarters + dad_gift) : initial_quarters = 21 :=
by sorry

end sara_initial_quarters_l239_239878


namespace part_I_part_II_l239_239259

def f (x a : ℝ) : ℝ := abs (3 * x + 2) - abs (2 * x + a)

theorem part_I (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = 4 / 3 :=
by
  sorry

theorem part_II (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≤ 0) ↔ (3 ≤ a ∨ a ≤ -7) :=
by
  sorry

end part_I_part_II_l239_239259


namespace parabola_directrix_l239_239803

theorem parabola_directrix (x : ℝ) : ∃ d : ℝ, (∀ x : ℝ, 4 * x ^ 2 - 3 = d) → d = -49 / 16 :=
by
  sorry

end parabola_directrix_l239_239803


namespace problem_lean_statement_l239_239140

def P (x : ℝ) : ℝ := x^2 - 3*x - 9

theorem problem_lean_statement :
  let a := 61
  let b := 109
  let c := 621
  let d := 39
  let e := 20
  a + b + c + d + e = 850 := 
by
  sorry

end problem_lean_statement_l239_239140


namespace circles_internally_tangent_l239_239643

theorem circles_internally_tangent (R r : ℝ) (h1 : R + r = 5) (h2 : R * r = 6) (d : ℝ) (h3 : d = 1) : d = |R - r| :=
by
  -- This allows the logic of the solution to be captured as the theorem we need to prove
  sorry

end circles_internally_tangent_l239_239643


namespace weightlifter_one_hand_l239_239157

theorem weightlifter_one_hand (total_weight : ℕ) (h : total_weight = 20) (even_distribution : total_weight % 2 = 0) : total_weight / 2 = 10 :=
by
  sorry

end weightlifter_one_hand_l239_239157


namespace power_mod_remainder_l239_239553

theorem power_mod_remainder (a b : ℕ) (h1 : a = 3) (h2 : b = 167) :
  (3^167) % 11 = 9 := by
  sorry

end power_mod_remainder_l239_239553


namespace sum_of_numbers_l239_239827

theorem sum_of_numbers (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) (h3 : x^2 - y^2 = 50) : x + y = 5 :=
by
  sorry

end sum_of_numbers_l239_239827


namespace pirate_15_gets_coins_l239_239718

def coins_required_for_pirates : ℕ :=
  Nat.factorial 14 * ((2 ^ 4) * (3 ^ 9)) / 15 ^ 14

theorem pirate_15_gets_coins :
  coins_required_for_pirates = 314928 := 
by sorry

end pirate_15_gets_coins_l239_239718


namespace polynomial_integer_roots_a_value_l239_239236

open Polynomial

theorem polynomial_integer_roots_a_value (α β γ : ℤ) (a : ℤ) :
  (X - C α) * (X - C β) * (X - C γ) = X^3 - 2 * X^2 - 25 * X + C a →
  α + β + γ = 2 →
  α * β + α * γ + β * γ = -25 →
  a = -50 :=
by
  sorry

end polynomial_integer_roots_a_value_l239_239236


namespace find_number_l239_239896

theorem find_number : ∃ n : ℕ, ∃ q : ℕ, ∃ r : ℕ, q = 6 ∧ r = 4 ∧ n = 9 * q + r ∧ n = 58 :=
by
  sorry

end find_number_l239_239896


namespace num_chords_num_triangles_l239_239371

noncomputable def num_points : ℕ := 10

theorem num_chords (n : ℕ) (h : n = num_points) : (n.choose 2) = 45 := by
  sorry

theorem num_triangles (n : ℕ) (h : n = num_points) : (n.choose 3) = 120 := by
  sorry

end num_chords_num_triangles_l239_239371


namespace subtraction_digits_l239_239240

theorem subtraction_digits (a b c : ℕ) (h1 : c - a = 2) (h2 : b = c - 1) (h3 : 100 * a + 10 * b + c - (100 * c + 10 * b + a) = 802) :
a = 0 ∧ b = 1 ∧ c = 2 :=
by {
  -- The detailed proof steps will go here
  sorry
}

end subtraction_digits_l239_239240


namespace final_velocity_l239_239767

variable (u a t : ℝ)

-- Defining the conditions
def initial_velocity := u = 0
def acceleration := a = 1.2
def time := t = 15

-- Statement of the theorem
theorem final_velocity : initial_velocity u ∧ acceleration a ∧ time t → (u + a * t = 18) := by
  sorry

end final_velocity_l239_239767


namespace only_polyC_is_square_of_binomial_l239_239065

-- Defining the polynomials
def polyA (m n : ℤ) : ℤ := (-m + n) * (m - n)
def polyB (a b : ℤ) : ℤ := (1/2 * a + b) * (b - 1/2 * a)
def polyC (x : ℤ) : ℤ := (x + 5) * (x + 5)
def polyD (a b : ℤ) : ℤ := (3 * a - 4 * b) * (3 * b + 4 * a)

-- Proving that only polyC fits the square of a binomial formula
theorem only_polyC_is_square_of_binomial (x : ℤ) :
  (polyC x) = (x + 5) * (x + 5) ∧
  (∀ m n : ℤ, polyA m n ≠ (m - n)^2) ∧
  (∀ a b : ℤ, polyB a b ≠ (1/2 * a + b)^2) ∧
  (∀ a b : ℤ, polyD a b ≠ (3 * a - 4 * b)^2) :=
by
  sorry

end only_polyC_is_square_of_binomial_l239_239065


namespace solution_set_f_x_gt_0_l239_239401

theorem solution_set_f_x_gt_0 (b : ℝ)
  (h_eq : ∀ x : ℝ, (x + 1) * (x - 3) = 0 → b = -2) :
  {x : ℝ | (x - 1)^2 > 0} = {x : ℝ | x ≠ 1} :=
by
  sorry

end solution_set_f_x_gt_0_l239_239401


namespace largest_result_among_expressions_l239_239162

def E1 : ℕ := 992 * 999 + 999
def E2 : ℕ := 993 * 998 + 998
def E3 : ℕ := 994 * 997 + 997
def E4 : ℕ := 995 * 996 + 996

theorem largest_result_among_expressions : E4 > E1 ∧ E4 > E2 ∧ E4 > E3 :=
by sorry

end largest_result_among_expressions_l239_239162


namespace fish_original_count_l239_239295

theorem fish_original_count (F : ℕ) (h : F / 2 - F / 6 = 12) : F = 36 := 
by 
  sorry

end fish_original_count_l239_239295


namespace virginia_initial_eggs_l239_239619

theorem virginia_initial_eggs (final_eggs : ℕ) (taken_eggs : ℕ) (H : final_eggs = 93) (G : taken_eggs = 3) : final_eggs + taken_eggs = 96 := 
by
  -- proof part could go here
  sorry

end virginia_initial_eggs_l239_239619


namespace either_d_or_2d_is_perfect_square_l239_239951

theorem either_d_or_2d_is_perfect_square
  (a c d : ℕ) (hrel_prime : Nat.gcd a c = 1) (hd : ∃ D : ℝ, D = d ∧ (D:ℝ) > 0)
  (hdiam : d^2 = 2 * a^2 + c^2) :
  ∃ m : ℕ, m^2 = d ∨ m^2 = 2 * d :=
by
  sorry

end either_d_or_2d_is_perfect_square_l239_239951


namespace area_ratio_is_correct_l239_239894

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l239_239894


namespace spoons_needed_to_fill_cup_l239_239974

-- Define necessary conditions
def spoon_capacity : Nat := 5
def liter_to_milliliters : Nat := 1000

-- State the problem
theorem spoons_needed_to_fill_cup : liter_to_milliliters / spoon_capacity = 200 := 
by 
  -- Skip the actual proof
  sorry

end spoons_needed_to_fill_cup_l239_239974


namespace carla_glasses_lemonade_l239_239080

theorem carla_glasses_lemonade (time_total : ℕ) (rate : ℕ) (glasses : ℕ) 
  (h1 : time_total = 3 * 60 + 40) 
  (h2 : rate = 20) 
  (h3 : glasses = time_total / rate) : 
  glasses = 11 := 
by 
  -- We'll fill in the proof here in a real scenario
  sorry

end carla_glasses_lemonade_l239_239080


namespace cost_of_nuts_l239_239339

/--
Adam bought 3 kilograms of nuts and 2.5 kilograms of dried fruits at a store. 
One kilogram of nuts costs a certain amount N and one kilogram of dried fruit costs $8. 
His purchases cost $56. Prove that one kilogram of nuts costs $12.
-/
theorem cost_of_nuts (N : ℝ) 
  (h1 : 3 * N + 2.5 * 8 = 56) 
  : N = 12 := by
  sorry

end cost_of_nuts_l239_239339


namespace maximum_area_of_triangle_l239_239003

theorem maximum_area_of_triangle :
  ∃ (b c : ℝ), (a = 2) ∧ (A = 60 * Real.pi / 180) ∧
  (∀ S : ℝ, S = (1/2) * b * c * Real.sin A → S ≤ Real.sqrt 3) :=
by sorry

end maximum_area_of_triangle_l239_239003


namespace least_five_digit_perfect_square_and_cube_l239_239805

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l239_239805


namespace star_j_l239_239234

def star (x y : ℝ) : ℝ := x^3 - x * y

theorem star_j (j : ℝ) : star j (star j j) = 2 * j^3 - j^4 := 
by
  sorry

end star_j_l239_239234


namespace option_d_is_quadratic_equation_l239_239596

theorem option_d_is_quadratic_equation (x y : ℝ) : 
  (x^2 + x - 4 = 0) ↔ (x^2 + x = 4) := 
by
  sorry

end option_d_is_quadratic_equation_l239_239596


namespace soccer_team_arrangements_l239_239910

theorem soccer_team_arrangements : 
  ∃ (n : ℕ), n = 2 * (Nat.factorial 11)^2 := 
sorry

end soccer_team_arrangements_l239_239910


namespace n_prime_of_divisors_l239_239975

theorem n_prime_of_divisors (n k : ℕ) (h₁ : n > 1) 
  (h₂ : ∀ d : ℕ, d ∣ n → (d + k ∣ n) ∨ (d - k ∣ n)) : Prime n :=
  sorry

end n_prime_of_divisors_l239_239975


namespace cobs_count_l239_239686

theorem cobs_count (bushel_weight : ℝ) (ear_weight : ℝ) (num_bushels : ℕ)
  (h1 : bushel_weight = 56) (h2 : ear_weight = 0.5) (h3 : num_bushels = 2) : 
  ((num_bushels * bushel_weight) / ear_weight) = 224 :=
by 
  sorry

end cobs_count_l239_239686


namespace marina_drive_l239_239035

theorem marina_drive (a b c : ℕ) (x : ℕ) 
  (h1 : 1 ≤ a) 
  (h2 : a + b + c ≤ 9)
  (h3 : 90 * (b - a) = 60 * x)
  (h4 : x = 3 * (b - a) / 2) :
  a = 1 ∧ b = 3 ∧ c = 5 ∧ a^2 + b^2 + c^2 = 35 :=
by {
  sorry
}

end marina_drive_l239_239035


namespace sequence_x_values_3001_l239_239427

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l239_239427


namespace percentage_income_spent_on_clothes_l239_239254

-- Define the assumptions
def monthly_income : ℝ := 90000
def household_expenses : ℝ := 0.5 * monthly_income
def medicine_expenses : ℝ := 0.15 * monthly_income
def savings : ℝ := 9000

-- Define the proof statement
theorem percentage_income_spent_on_clothes :
  ∃ (clothes_expenses : ℝ),
    clothes_expenses = monthly_income - household_expenses - medicine_expenses - savings ∧
    (clothes_expenses / monthly_income) * 100 = 25 := 
sorry

end percentage_income_spent_on_clothes_l239_239254


namespace fencing_problem_l239_239233

noncomputable def fencingRequired (L A W F : ℝ) := (A = L * W) → (F = 2 * W + L)

theorem fencing_problem :
  fencingRequired 25 880 35.2 95.4 :=
by
  sorry

end fencing_problem_l239_239233


namespace quadratic_has_single_solution_l239_239761

theorem quadratic_has_single_solution (q : ℚ) (h : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 16 * x + 9 = 0 → q = 64 / 9) := by
  sorry

end quadratic_has_single_solution_l239_239761


namespace lisa_flight_time_l239_239670

theorem lisa_flight_time :
  ∀ (d s : ℕ), (d = 256) → (s = 32) → ((d / s) = 8) :=
by
  intros d s h_d h_s
  sorry

end lisa_flight_time_l239_239670


namespace words_written_first_two_hours_l239_239226

def essay_total_words : ℕ := 1200
def words_per_hour_first_two_hours (W : ℕ) : ℕ := 2 * W
def words_per_hour_next_two_hours : ℕ := 2 * 200

theorem words_written_first_two_hours (W : ℕ) (h : words_per_hour_first_two_hours W + words_per_hour_next_two_hours = essay_total_words) : W = 400 := 
by 
  sorry

end words_written_first_two_hours_l239_239226


namespace inverse_mod_187_l239_239508

theorem inverse_mod_187 : ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 186 ∧ (2 * x) % 187 = 1 :=
by
  use 94
  sorry

end inverse_mod_187_l239_239508


namespace test_score_range_l239_239409

theorem test_score_range
  (mark_score : ℕ) (least_score : ℕ) (highest_score : ℕ)
  (twice_least_score : mark_score = 2 * least_score)
  (mark_fixed : mark_score = 46)
  (highest_fixed : highest_score = 98) :
  (highest_score - least_score) = 75 :=
by
  sorry

end test_score_range_l239_239409


namespace crow_eating_time_l239_239760

theorem crow_eating_time (n : ℕ) (h : ∀ t : ℕ, t = (n / 5) → t = 4) : (4 + (4 / 5) = 4.8) :=
by
  sorry

end crow_eating_time_l239_239760


namespace Donovan_Mitchell_goal_average_l239_239309

theorem Donovan_Mitchell_goal_average 
  (current_avg_pg : ℕ)     -- Donovan's current average points per game.
  (played_games : ℕ)       -- Number of games played so far.
  (required_avg_pg : ℕ)    -- Required average points per game in remaining games.
  (total_games : ℕ)        -- Total number of games in the season.
  (goal_avg_pg : ℕ)        -- Goal average points per game for the entire season.
  (H1 : current_avg_pg = 26)
  (H2 : played_games = 15)
  (H3 : required_avg_pg = 42)
  (H4 : total_games = 20) :
  goal_avg_pg = 30 :=
by
  sorry

end Donovan_Mitchell_goal_average_l239_239309


namespace problem_l239_239196

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom odd_f : ∀ x, f x = -f (-x)
axiom periodic_g : ∀ x, g x = g (x + 2)
axiom f_at_neg1 : f (-1) = 3
axiom g_at_1 : g 1 = 3
axiom g_function : ∀ n : ℕ, g (2 * n * f 1) = n * f (f 1 + g (-1)) + 2

theorem problem : g (-6) + f 0 = 2 :=
by sorry

end problem_l239_239196


namespace folding_hexagon_quadrilateral_folding_hexagon_pentagon_l239_239139

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem folding_hexagon_quadrilateral :
  (sum_of_interior_angles 4 = 360) :=
by
  sorry

theorem folding_hexagon_pentagon :
  (sum_of_interior_angles 5 = 540) :=
by
  sorry

end folding_hexagon_quadrilateral_folding_hexagon_pentagon_l239_239139


namespace vector_addition_l239_239711

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (6, 2)
def vector_b : ℝ × ℝ := (-2, 4)

-- Theorem statement to prove the sum of vector_a and vector_b equals (4, 6)
theorem vector_addition :
  vector_a + vector_b = (4, 6) :=
sorry

end vector_addition_l239_239711


namespace cos_of_angle_complement_l239_239881

theorem cos_of_angle_complement (α : ℝ) (h : 90 - α = 30) : Real.cos α = 1 / 2 :=
by
  sorry

end cos_of_angle_complement_l239_239881


namespace regular_dodecahedron_has_12_faces_l239_239808

-- Define a structure to represent a regular dodecahedron
structure RegularDodecahedron where

-- The main theorem to state that a regular dodecahedron has 12 faces
theorem regular_dodecahedron_has_12_faces (D : RegularDodecahedron) : ∃ faces : ℕ, faces = 12 := by
  sorry

end regular_dodecahedron_has_12_faces_l239_239808


namespace turtles_remaining_proof_l239_239744

noncomputable def turtles_original := 50
noncomputable def turtles_additional := 7 * turtles_original - 6
noncomputable def turtles_total_before_frightened := turtles_original + turtles_additional
noncomputable def turtles_frightened := (3 / 7) * turtles_total_before_frightened
noncomputable def turtles_remaining := turtles_total_before_frightened - turtles_frightened

theorem turtles_remaining_proof : turtles_remaining = 226 := by
  sorry

end turtles_remaining_proof_l239_239744


namespace lines_perpendicular_to_same_plane_are_parallel_l239_239048

variables {Point Line Plane : Type}
variables (a b c : Line) (α β γ : Plane)
variables (perp_line_to_plane : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)
variables (subset_line_in_plane : Line → Plane → Prop)

-- The conditions
axiom a_perp_alpha : perp_line_to_plane a α
axiom b_perp_alpha : perp_line_to_plane b α

-- The statement to prove
theorem lines_perpendicular_to_same_plane_are_parallel :
  parallel_lines a b :=
by sorry

end lines_perpendicular_to_same_plane_are_parallel_l239_239048


namespace triangle_sides_arithmetic_progression_l239_239056

theorem triangle_sides_arithmetic_progression (a d : ℤ) (h : 3 * a = 15) (h1 : a > 0) (h2 : d ≥ 0) :
  (a - d = 5 ∨ a - d = 4 ∨ a - d = 3) ∧ 
  (a = 5) ∧ 
  (a + d = 5 ∨ a + d = 6 ∨ a + d = 7) := 
  sorry

end triangle_sides_arithmetic_progression_l239_239056


namespace find_c_l239_239036

-- Define c and the floor function
def c : ℝ := 13.1

theorem find_c (h : c + ⌊c⌋ = 25.6) : c = 13.1 :=
sorry

end find_c_l239_239036


namespace small_boxes_count_correct_l239_239001

-- Definitions of constants
def feet_per_large_box_seal : ℕ := 4
def feet_per_medium_box_seal : ℕ := 2
def feet_per_small_box_seal : ℕ := 1
def feet_per_box_label : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def total_tape_used : ℕ := 44

-- Definition for the total tape used for large and medium boxes
def tape_used_large_boxes : ℕ := (large_boxes_packed * feet_per_large_box_seal) + (large_boxes_packed * feet_per_box_label)
def tape_used_medium_boxes : ℕ := (medium_boxes_packed * feet_per_medium_box_seal) + (medium_boxes_packed * feet_per_box_label)
def tape_used_large_and_medium_boxes : ℕ := tape_used_large_boxes + tape_used_medium_boxes
def tape_used_small_boxes : ℕ := total_tape_used - tape_used_large_and_medium_boxes

-- The number of small boxes packed
def small_boxes_packed : ℕ := tape_used_small_boxes / (feet_per_small_box_seal + feet_per_box_label)

-- Proof problem statement
theorem small_boxes_count_correct (n : ℕ) (h : small_boxes_packed = n) : n = 5 :=
by
  sorry

end small_boxes_count_correct_l239_239001


namespace age_difference_is_13_l239_239789

variables (A B C X : ℕ)
variables (total_age_A_B total_age_B_C : ℕ)

-- Conditions
def condition1 : Prop := total_age_A_B = total_age_B_C + X
def condition2 : Prop := C = A - 13

-- Theorem statement
theorem age_difference_is_13 (h1: condition1 total_age_A_B total_age_B_C X)
                             (h2: condition2 A C) :
  X = 13 :=
sorry

end age_difference_is_13_l239_239789


namespace average_gpa_of_whole_class_l239_239956

-- Define the conditions
variables (n : ℕ)
def num_students_in_group1 := n / 3
def num_students_in_group2 := 2 * n / 3

def gpa_group1 := 15
def gpa_group2 := 18

-- Lean statement for the proof problem
theorem average_gpa_of_whole_class (hn_pos : 0 < n):
  ((num_students_in_group1 * gpa_group1) + (num_students_in_group2 * gpa_group2)) / n = 17 :=
sorry

end average_gpa_of_whole_class_l239_239956


namespace interest_rate_calculation_l239_239540

theorem interest_rate_calculation :
  let P := 1599.9999999999998
  let A := 1792
  let T := 2 + 2 / 5
  let I := A - P
  I / (P * T) = 0.05 :=
  sorry

end interest_rate_calculation_l239_239540


namespace lower_bound_of_expression_l239_239638

theorem lower_bound_of_expression :
  ∃ L : ℤ, (∀ n : ℤ, ((-1 ≤ n ∧ n ≤ 8) → (L < 4 * n + 7 ∧ 4 * n + 7 < 40))) ∧ L = 1 :=
by {
  sorry
}

end lower_bound_of_expression_l239_239638


namespace joanne_main_job_hours_l239_239869

theorem joanne_main_job_hours (h : ℕ) (earn_main_job : ℝ) (earn_part_time : ℝ) (hours_part_time : ℕ) (days_week : ℕ) (total_weekly_earn : ℝ) :
  earn_main_job = 16.00 →
  earn_part_time = 13.50 →
  hours_part_time = 2 →
  days_week = 5 →
  total_weekly_earn = 775 →
  days_week * earn_main_job * h + days_week * earn_part_time * hours_part_time = total_weekly_earn →
  h = 8 :=
by
  sorry

end joanne_main_job_hours_l239_239869


namespace weight_of_3_moles_of_CaI2_is_881_64_l239_239280

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
noncomputable def weight_3_moles_CaI2 : ℝ := 3 * molar_mass_CaI2

theorem weight_of_3_moles_of_CaI2_is_881_64 :
  weight_3_moles_CaI2 = 881.64 :=
by sorry

end weight_of_3_moles_of_CaI2_is_881_64_l239_239280


namespace parameterized_curve_is_line_l239_239892

theorem parameterized_curve_is_line :
  ∀ (t : ℝ), ∃ (m b : ℝ), y = 5 * ((x - 5) / 3) - 3 → y = (5 * x - 34) / 3 := 
by
  sorry

end parameterized_curve_is_line_l239_239892


namespace smallest_n_l239_239840

-- Define the conditions as predicates
def condition1 (n : ℕ) : Prop := (n + 2018) % 2020 = 0
def condition2 (n : ℕ) : Prop := (n + 2020) % 2018 = 0

-- The main theorem statement using these conditions
theorem smallest_n (n : ℕ) : 
  (∃ n, condition1 n ∧ condition2 n ∧ (∀ m, condition1 m ∧ condition2 m → n ≤ m)) ↔ n = 2030102 := 
by 
    sorry

end smallest_n_l239_239840


namespace largest_even_number_in_sequence_of_six_l239_239330

-- Definitions and conditions
def smallest_even_number (x : ℤ) : Prop :=
  x + (x + 2) + (x+4) + (x+6) + (x + 8) + (x + 10) = 540

def sum_of_squares_of_sequence (x : ℤ) : Prop :=
  x^2 + (x + 2)^2 + (x + 4)^2 + (x + 6)^2 + (x + 8)^2 + (x + 10)^2 = 97920

-- Statement to prove
theorem largest_even_number_in_sequence_of_six (x : ℤ) (h1 : smallest_even_number x) (h2 : sum_of_squares_of_sequence x) : x + 10 = 95 :=
  sorry

end largest_even_number_in_sequence_of_six_l239_239330


namespace daily_profit_1200_impossible_daily_profit_1600_l239_239927

-- Definitions of given conditions
def avg_shirts_sold_per_day : ℕ := 30
def profit_per_shirt : ℕ := 40

-- Function for the number of shirts sold given a price reduction
def shirts_sold (x : ℕ) : ℕ := avg_shirts_sold_per_day + 2 * x

-- Function for the profit per shirt given a price reduction
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt - x

-- Function for the daily profit given a price reduction
def daily_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (shirts_sold x)

-- Proving the desired conditions in Lean

-- Part 1: Prove that reducing the price by 25 yuan results in a daily profit of 1200 yuan
theorem daily_profit_1200 (x : ℕ) : daily_profit x = 1200 ↔ x = 25 :=
by
  { sorry }

-- Part 2: Prove that a daily profit of 1600 yuan is not achievable
theorem impossible_daily_profit_1600 (x : ℕ) : daily_profit x ≠ 1600 :=
by
  { sorry }

end daily_profit_1200_impossible_daily_profit_1600_l239_239927


namespace mrs_hilt_initial_marbles_l239_239978

theorem mrs_hilt_initial_marbles (lost_marble : ℕ) (remaining_marble : ℕ) (h1 : lost_marble = 15) (h2 : remaining_marble = 23) : 
    (remaining_marble + lost_marble) = 38 :=
by
  sorry

end mrs_hilt_initial_marbles_l239_239978


namespace sqrt_sine_tan_domain_l239_239138

open Real

noncomputable def domain_sqrt_sine_tan : Set ℝ :=
  {x | ∃ (k : ℤ), (-π / 2 + 2 * k * π < x ∧ x < π / 2 + 2 * k * π) ∨ x = k * π}

theorem sqrt_sine_tan_domain (x : ℝ) :
  (sin x * tan x ≥ 0) ↔ x ∈ domain_sqrt_sine_tan :=
by
  sorry

end sqrt_sine_tan_domain_l239_239138


namespace percentage_calculation_l239_239609

theorem percentage_calculation : 
  (0.8 * 90) = ((P / 100) * 60.00000000000001 + 30) → P = 70 := by
  sorry

end percentage_calculation_l239_239609


namespace percent_problem_l239_239333

theorem percent_problem (x : ℝ) (h : 0.20 * x = 1000) : 1.20 * x = 6000 := by
  sorry

end percent_problem_l239_239333


namespace motorist_gas_problem_l239_239659

noncomputable def original_price_per_gallon (P : ℝ) : Prop :=
  12 * P = 10 * (P + 0.30)

def fuel_efficiency := 25

def new_distance_travelled (P : ℝ) : ℝ :=
  10 * fuel_efficiency

theorem motorist_gas_problem :
  ∃ P : ℝ, original_price_per_gallon P ∧ P = 1.5 ∧ new_distance_travelled P = 250 :=
by
  use 1.5
  sorry

end motorist_gas_problem_l239_239659


namespace candy_division_l239_239799

theorem candy_division (total_candy : ℕ) (students : ℕ) (per_student : ℕ) 
  (h1 : total_candy = 344) (h2 : students = 43) : 
  total_candy / students = per_student ↔ per_student = 8 := 
by 
  sorry

end candy_division_l239_239799


namespace tim_morning_running_hours_l239_239089

theorem tim_morning_running_hours 
  (runs_per_week : ℕ) 
  (total_hours_per_week : ℕ) 
  (runs_per_day : ℕ → ℕ) 
  (hrs_per_day_morning_evening_equal : ∀ (d : ℕ), runs_per_day d = runs_per_week * total_hours_per_week / runs_per_week) 
  (hrs_per_day : ℕ) 
  (hrs_per_morning : ℕ) 
  (hrs_per_evening : ℕ) 
  : hrs_per_morning = 1 :=
by 
  -- Given conditions
  have hrs_per_day := total_hours_per_week / runs_per_week
  have hrs_per_morning_evening := hrs_per_day / 2
  -- Conclusion
  sorry

end tim_morning_running_hours_l239_239089


namespace part1_part2_l239_239473

/-- Given a triangle ABC with sides opposite to angles A, B, C being a, b, c respectively,
and a sin A sin B + b cos^2 A = 5/3 a,
prove that (1) b / a = 5/3. -/
theorem part1 (a b : ℝ) (A B : ℝ) (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a) :
  b / a = 5 / 3 :=
sorry

/-- Given the previous result b / a = 5/3 and the condition c^2 = a^2 + 8/5 b^2,
prove that (2) angle C = 2π / 3. -/
theorem part2 (a b c : ℝ) (A B C : ℝ)
  (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a)
  (h₂ : c^2 = a^2 + (8 / 5) * b^2)
  (h₃ : b / a = 5 / 3) :
  C = 2 * Real.pi / 3 :=
sorry

end part1_part2_l239_239473


namespace expression_not_defined_at_x_eq_5_l239_239542

theorem expression_not_defined_at_x_eq_5 :
  ∃ x : ℝ, x^3 - 15 * x^2 + 75 * x - 125 = 0 ↔ x = 5 :=
by
  sorry

end expression_not_defined_at_x_eq_5_l239_239542


namespace findInitialVolume_l239_239614

def initialVolume (V : ℝ) : Prop :=
  let newVolume := V + 18
  let initialSugar := 0.27 * V
  let addedSugar := 3.2
  let totalSugar := initialSugar + addedSugar
  let finalSugarPercentage := 0.26536312849162012
  finalSugarPercentage * newVolume = totalSugar 

theorem findInitialVolume : ∃ (V : ℝ), initialVolume V ∧ V = 340 := by
  use 340
  unfold initialVolume
  sorry

end findInitialVolume_l239_239614


namespace printing_time_l239_239425

-- Definitions based on the problem conditions
def printer_rate : ℕ := 25 -- Pages per minute
def total_pages : ℕ := 325 -- Total number of pages to be printed

-- Statement of the problem rewritten as a Lean 4 statement
theorem printing_time : total_pages / printer_rate = 13 := by
  sorry

end printing_time_l239_239425


namespace ratio_of_guests_l239_239639

def bridgette_guests : Nat := 84
def alex_guests : Nat := sorry -- This will be inferred in the theorem
def extra_plates : Nat := 10
def total_asparagus_spears : Nat := 1200
def asparagus_per_plate : Nat := 8

theorem ratio_of_guests (A : Nat) (h1 : total_asparagus_spears / asparagus_per_plate = 150) (h2 : 150 - extra_plates = 140) (h3 : 140 - bridgette_guests = A) : A / bridgette_guests = 2 / 3 :=
by
  sorry

end ratio_of_guests_l239_239639


namespace fraction_equiv_l239_239213

theorem fraction_equiv (x y : ℝ) (h : x / 2 = y / 5) : x / y = 2 / 5 :=
by
  sorry

end fraction_equiv_l239_239213


namespace candy_total_l239_239961

theorem candy_total (n m : ℕ) (h1 : n = 2) (h2 : m = 8) : n * m = 16 :=
by
  -- This will contain the proof
  sorry

end candy_total_l239_239961


namespace find_radius_yz_l239_239737

-- Define the setup for the centers of the circles and their radii
def circle_with_center (c : Type*) (radius : ℝ) : Prop := sorry
def tangent_to (c₁ c₂ : Type*) : Prop := sorry

-- Given conditions
variable (O X Y Z : Type*)
variable (r : ℝ)
variable (Xe_radius : circle_with_center X 1)
variable (O_radius : circle_with_center O 2)
variable (XtangentO : tangent_to X O)
variable (YtangentO : tangent_to Y O)
variable (YtangentX : tangent_to Y X)
variable (YtangentZ : tangent_to Y Z)
variable (ZtangentO : tangent_to Z O)
variable (ZtangentX : tangent_to Z X)
variable (ZtangentY : tangent_to Z Y)

-- The theorem to prove
theorem find_radius_yz :
  r = 8 / 9 := sorry

end find_radius_yz_l239_239737


namespace fewest_fence_posts_l239_239731

def fence_posts (length_wide short_side long_side : ℕ) (post_interval : ℕ) : ℕ :=
  let wide_side_posts := (long_side / post_interval) + 1
  let short_side_posts := (short_side / post_interval)
  wide_side_posts + 2 * short_side_posts

theorem fewest_fence_posts : fence_posts 40 10 100 10 = 19 :=
  by
    -- The proof will be completed here
    sorry

end fewest_fence_posts_l239_239731


namespace setB_can_form_triangle_l239_239373

theorem setB_can_form_triangle : 
  let a := 8
  let b := 6
  let c := 4
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  let a := 8
  let b := 6
  let c := 4
  have h1 : a + b > c := by sorry
  have h2 : a + c > b := by sorry
  have h3 : b + c > a := by sorry
  exact ⟨h1, h2, h3⟩

end setB_can_form_triangle_l239_239373


namespace minibus_children_count_l239_239041

theorem minibus_children_count
  (total_seats : ℕ)
  (seats_with_3_children : ℕ)
  (seats_with_2_children : ℕ)
  (children_per_seat_3 : ℕ)
  (children_per_seat_2 : ℕ)
  (h_seats_count : total_seats = 7)
  (h_seats_distribution : seats_with_3_children = 5 ∧ seats_with_2_children = 2)
  (h_children_per_seat : children_per_seat_3 = 3 ∧ children_per_seat_2 = 2) :
  seats_with_3_children * children_per_seat_3 + seats_with_2_children * children_per_seat_2 = 19 :=
by
  sorry

end minibus_children_count_l239_239041


namespace line_intersection_points_l239_239754

def line_intersects_axes (x y : ℝ) : Prop :=
  (4 * y - 5 * x = 20)

theorem line_intersection_points :
  ∃ p1 p2, line_intersects_axes p1.1 p1.2 ∧ line_intersects_axes p2.1 p2.2 ∧
    (p1 = (-4, 0) ∧ p2 = (0, 5)) :=
by
  sorry

end line_intersection_points_l239_239754


namespace jack_change_l239_239024

def cost_per_sandwich : ℕ := 5
def number_of_sandwiches : ℕ := 3
def payment : ℕ := 20

theorem jack_change : payment - (cost_per_sandwich * number_of_sandwiches) = 5 := 
by
  sorry

end jack_change_l239_239024


namespace overlap_length_in_mm_l239_239398

theorem overlap_length_in_mm {sheets : ℕ} {length_per_sheet : ℝ} {perimeter : ℝ} 
  (h_sheets : sheets = 12)
  (h_length_per_sheet : length_per_sheet = 18)
  (h_perimeter : perimeter = 210) : 
  (length_per_sheet * sheets - perimeter) / sheets * 10 = 5 := by
  sorry

end overlap_length_in_mm_l239_239398


namespace temperature_difference_l239_239167

theorem temperature_difference (T_south T_north : ℝ) (h_south : T_south = 6) (h_north : T_north = -3) :
  T_south - T_north = 9 :=
by 
  -- Proof goes here
  sorry

end temperature_difference_l239_239167


namespace polygon_interior_angle_sum_360_l239_239634

theorem polygon_interior_angle_sum_360 (n : ℕ) (h : (n-2) * 180 = 360) : n = 4 :=
sorry

end polygon_interior_angle_sum_360_l239_239634


namespace sufficient_condition_for_m_l239_239099

variable (x m : ℝ)

def p (x : ℝ) : Prop := abs (x - 4) ≤ 6
def q (x m : ℝ) : Prop := x ≤ 1 + m

theorem sufficient_condition_for_m (h : ∀ x, p x → q x m ∧ ∃ x, ¬p x ∧ q x m) : m ≥ 9 :=
sorry

end sufficient_condition_for_m_l239_239099


namespace ratio_first_term_common_diff_l239_239355

theorem ratio_first_term_common_diff {a d : ℤ} 
  (S_20 : ℤ) (S_10 : ℤ)
  (h1 : S_20 = 10 * (2 * a + 19 * d))
  (h2 : S_10 = 5 * (2 * a + 9 * d))
  (h3 : S_20 = 6 * S_10) :
  a / d = 2 :=
by
  sorry

end ratio_first_term_common_diff_l239_239355


namespace symmetric_function_value_l239_239365

noncomputable def f (x a : ℝ) := (|x - 2| + a) / (Real.sqrt (4 - x^2))

theorem symmetric_function_value :
  ∃ a : ℝ, (∀ x : ℝ, f x a = (|x - 2| + a) / (Real.sqrt (4 - x^2)) ∧ f x a = -f (-x) a) →
  f (a / 2) a = (Real.sqrt 3) / 3 :=
by
  sorry

end symmetric_function_value_l239_239365


namespace period_of_cos_3x_l239_239796

theorem period_of_cos_3x :
  ∃ T : ℝ, (∀ x : ℝ, (Real.cos (3 * (x + T))) = Real.cos (3 * x)) ∧ (T = (2 * Real.pi) / 3) :=
sorry

end period_of_cos_3x_l239_239796


namespace taxi_ride_cost_l239_239222

-- Lean statement
theorem taxi_ride_cost (base_fare : ℝ) (rate1 : ℝ) (rate1_miles : ℝ) (rate2 : ℝ) (total_miles : ℝ) 
  (h_base_fare : base_fare = 2.00)
  (h_rate1 : rate1 = 0.30)
  (h_rate1_miles : rate1_miles = 3)
  (h_rate2 : rate2 = 0.40)
  (h_total_miles : total_miles = 8) :
  let rate1_cost := rate1 * rate1_miles
  let rate2_cost := rate2 * (total_miles - rate1_miles)
  base_fare + rate1_cost + rate2_cost = 4.90 := by
  sorry

end taxi_ride_cost_l239_239222


namespace solve_inequality_range_of_a_l239_239844

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem solve_inequality : {x : ℝ | f x > 5} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 4 / 3} :=
by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (f x < a)) : a ≤ 2 :=
by
  sorry

end solve_inequality_range_of_a_l239_239844


namespace largest_initial_number_l239_239579

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l239_239579


namespace calculation_correct_l239_239500

theorem calculation_correct : -2 + 3 = 1 :=
by
  sorry

end calculation_correct_l239_239500


namespace unique_real_solution_l239_239935

noncomputable def cubic_eq (b x : ℝ) : ℝ :=
  x^3 - b * x^2 - 3 * b * x + b^2 - 2

theorem unique_real_solution (b : ℝ) :
  (∃! x : ℝ, cubic_eq b x = 0) ↔ b = 7 / 4 :=
by
  sorry

end unique_real_solution_l239_239935


namespace leak_takes_3_hours_to_empty_l239_239979

noncomputable def leak_emptying_time (inlet_rate_per_minute: ℕ) (tank_empty_time_with_inlet: ℕ) (tank_capacity: ℕ) : ℕ :=
  let inlet_rate_per_hour := inlet_rate_per_minute * 60
  let effective_empty_rate := tank_capacity / tank_empty_time_with_inlet
  let leak_rate := inlet_rate_per_hour + effective_empty_rate
  tank_capacity / leak_rate

theorem leak_takes_3_hours_to_empty:
  leak_emptying_time 6 12 1440 = 3 := 
sorry

end leak_takes_3_hours_to_empty_l239_239979


namespace cone_height_l239_239566

theorem cone_height (r_sector : ℝ) (θ_sector : ℝ) :
  r_sector = 3 → θ_sector = (2 * Real.pi / 3) → 
  ∃ (h : ℝ), h = 2 * Real.sqrt 2 := 
by 
  intros r_sector_eq θ_sector_eq
  sorry

end cone_height_l239_239566


namespace complement_intersection_l239_239101

open Set

variable (I : Set ℕ) (A B : Set ℕ)

-- Given the universal set and specific sets A and B
def universal_set : Set ℕ := {1,2,3,4,5}
def set_A : Set ℕ := {2,3,5}
def set_B : Set ℕ := {1,2}

-- To prove that the complement of B in I intersects A to be {3,5}
theorem complement_intersection :
  (universal_set \ set_B) ∩ set_A = {3,5} :=
sorry

end complement_intersection_l239_239101


namespace ineq_condition_l239_239548

theorem ineq_condition (a b : ℝ) : (a + 1 > b - 2) ↔ (a > b - 3 ∧ ¬(a > b)) :=
by
  sorry

end ineq_condition_l239_239548


namespace positive_integer_solutions_l239_239498

theorem positive_integer_solutions:
  ∀ (x y : ℕ), (5 * x + y = 11) → (x > 0) → (y > 0) → (x = 1 ∧ y = 6) ∨ (x = 2 ∧ y = 1) :=
by
  sorry

end positive_integer_solutions_l239_239498


namespace hole_depth_l239_239120

theorem hole_depth (height : ℝ) (half_depth : ℝ) (total_depth : ℝ) 
    (h_height : height = 90) 
    (h_half_depth : half_depth = total_depth / 2)
    (h_position : height + half_depth = total_depth - height) : 
    total_depth = 120 := 
by
    sorry

end hole_depth_l239_239120


namespace sum_of_digits_of_4_plus_2_pow_21_l239_239815

theorem sum_of_digits_of_4_plus_2_pow_21 :
  let x := (4 + 2)
  (x^(21) % 100).div 10 + (x^(21) % 100).mod 10 = 6 :=
by
  let x := (4 + 2)
  sorry

end sum_of_digits_of_4_plus_2_pow_21_l239_239815


namespace negation_of_proposition_l239_239747

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1)) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 < 1) := 
sorry

end negation_of_proposition_l239_239747


namespace evaluate_fg_l239_239860

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x - 5

theorem evaluate_fg : f (g 4) = 9 := by
  sorry

end evaluate_fg_l239_239860


namespace protein_percentage_in_mixture_l239_239965

theorem protein_percentage_in_mixture :
  let soybean_meal_weight := 240
  let cornmeal_weight := 40
  let mixture_weight := 280
  let soybean_protein_content := 0.14
  let cornmeal_protein_content := 0.07
  let total_protein := soybean_meal_weight * soybean_protein_content + cornmeal_weight * cornmeal_protein_content
  let protein_percentage := (total_protein / mixture_weight) * 100
  protein_percentage = 13 :=
by
  sorry

end protein_percentage_in_mixture_l239_239965


namespace younger_son_age_in_30_years_l239_239334

theorem younger_son_age_in_30_years
  (age_difference : ℕ)
  (elder_son_current_age : ℕ)
  (younger_son_age_in_30_years : ℕ) :
  age_difference = 10 →
  elder_son_current_age = 40 →
  younger_son_age_in_30_years = elder_son_current_age - age_difference + 30 →
  younger_son_age_in_30_years = 60 :=
by
  intros h_diff h_elder h_calc
  sorry

end younger_son_age_in_30_years_l239_239334


namespace remainder_equality_l239_239893

theorem remainder_equality (P P' : ℕ) (h1 : P = P' + 10) 
  (h2 : P % 10 = 0) (h3 : P' % 10 = 0) : 
  ((P^2 - P'^2) % 10 = 0) :=
by
  sorry

end remainder_equality_l239_239893


namespace sum_of_cubes_l239_239636

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 :=
sorry

end sum_of_cubes_l239_239636


namespace domain_of_log_x_squared_sub_2x_l239_239165

theorem domain_of_log_x_squared_sub_2x (x : ℝ) : x^2 - 2 * x > 0 ↔ x < 0 ∨ x > 2 :=
by
  sorry

end domain_of_log_x_squared_sub_2x_l239_239165


namespace complement_of_A_in_U_l239_239919

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set A
def A : Set ℕ := {3, 4, 5}

-- Statement to prove the complement of A with respect to U
theorem complement_of_A_in_U : U \ A = {1, 2, 6} := 
  by sorry

end complement_of_A_in_U_l239_239919


namespace solution_exists_l239_239074

def valid_grid (grid : List (List Nat)) : Prop :=
  grid = [[2, 3, 6], [6, 3, 2]] ∨
  grid = [[2, 4, 8], [8, 4, 2]]

theorem solution_exists :
  ∃ (grid : List (List Nat)), valid_grid grid := by
  sorry

end solution_exists_l239_239074


namespace matrix_to_system_solution_l239_239769

theorem matrix_to_system_solution :
  ∀ (x y : ℝ),
  (2 * x + y = 5) ∧ (x - 2 * y = 0) →
  3 * x - y = 5 :=
by
  sorry

end matrix_to_system_solution_l239_239769


namespace find_a7_over_b7_l239_239013

-- Definitions of the sequences and the arithmetic properties
variable {a b: ℕ → ℕ}  -- sequences a_n and b_n
variable {S T: ℕ → ℕ}  -- sums of the first n terms

-- Problem conditions
def is_arithmetic_sequence (seq: ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

def sum_of_first_n_terms (seq: ℕ → ℕ) (sum_fn: ℕ → ℕ) : Prop :=
  ∀ n, sum_fn n = n * (seq 1 + seq n) / 2

-- Given conditions
axiom h1: is_arithmetic_sequence a
axiom h2: is_arithmetic_sequence b
axiom h3: sum_of_first_n_terms a S
axiom h4: sum_of_first_n_terms b T
axiom h5: ∀ n, S n / T n = (3 * n + 2) / (2 * n)

-- Main theorem to prove
theorem find_a7_over_b7 : (a 7) / (b 7) = (41 / 26) :=
sorry

end find_a7_over_b7_l239_239013


namespace possible_values_a_possible_values_m_l239_239438

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a + 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem possible_values_a (a : ℝ) : 
  (A ∪ B a = A) → a = 2 ∨ a = 3 := sorry

theorem possible_values_m (m : ℝ) : 
  (A ∩ C m = C m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := sorry

end possible_values_a_possible_values_m_l239_239438


namespace batsman_average_after_17th_inning_l239_239184

theorem batsman_average_after_17th_inning (A : ℝ) (h1 : 16 * A + 200 = 17 * (A + 10)) : 
  A + 10 = 40 := 
by
  sorry

end batsman_average_after_17th_inning_l239_239184


namespace max_sum_unique_digits_expression_equivalent_l239_239976

theorem max_sum_unique_digits_expression_equivalent :
  ∃ (a b c d e : ℕ), (2 * 19 * 53 = 2014) ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (2 * (b + c) * (d + e) = 2014) ∧
    (a + b + c + d + e = 35) ∧ 
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) :=
by
  sorry

end max_sum_unique_digits_expression_equivalent_l239_239976


namespace least_three_digit_product_18_l239_239701

theorem least_three_digit_product_18 : ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ N = 100 * H + 10 * T + U ∧ H * T * U = 18) ∧ ∀ M : ℕ, (100 ≤ M ∧ M ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ M = 100 * H + 10 * T + U ∧ H * T * U = 18)) → N ≤ M :=
    sorry

end least_three_digit_product_18_l239_239701


namespace sequence_to_one_l239_239599

def nextStep (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n - 1

theorem sequence_to_one (n : ℕ) (h : n > 0) :
  ∃ seq : ℕ → ℕ, seq 0 = n ∧ (∀ i, seq (i + 1) = nextStep (seq i)) ∧ (∃ j, seq j = 1) := by
  sorry

end sequence_to_one_l239_239599


namespace condition1_condition2_condition3_l239_239943

-- Condition 1 statement
theorem condition1: (number_of_ways_condition1 : ℕ) = 5520 := by
  -- Expected proof that number_of_ways_condition1 = 5520
  sorry

-- Condition 2 statement
theorem condition2: (number_of_ways_condition2 : ℕ) = 3360 := by
  -- Expected proof that number_of_ways_condition2 = 3360
  sorry

-- Condition 3 statement
theorem condition3: (number_of_ways_condition3 : ℕ) = 360 := by
  -- Expected proof that number_of_ways_condition3 = 360
  sorry

end condition1_condition2_condition3_l239_239943


namespace solve_linear_function_l239_239092

theorem solve_linear_function :
  (∀ (x y : ℤ), (x = -3 ∧ y = -4) ∨ (x = -2 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ 
                      (x = 0 ∧ y = 2) ∨ (x = 1 ∧ y = 4) ∨ (x = 2 ∧ y = 6) →
   ∃ (a b : ℤ), y = a * x + b ∧ a * 1 + b = 4) :=
sorry

end solve_linear_function_l239_239092


namespace print_pages_500_l239_239146

theorem print_pages_500 (cost_per_page cents total_dollars) : 
  cost_per_page = 3 → 
  total_dollars = 15 → 
  cents = 100 * total_dollars → 
  (cents / cost_per_page) = 500 :=
by 
  intros h1 h2 h3
  sorry

end print_pages_500_l239_239146


namespace number_of_registration_methods_l239_239717

theorem number_of_registration_methods
  (students : ℕ) (groups : ℕ) (registration_methods : ℕ)
  (h_students : students = 4) (h_groups : groups = 3) :
  registration_methods = groups ^ students :=
by
  rw [h_students, h_groups]
  exact sorry

end number_of_registration_methods_l239_239717


namespace car_total_travel_time_l239_239107

-- Define the given conditions
def travel_time_ngapara_zipra : ℝ := 60
def travel_time_ningi_zipra : ℝ := 0.8 * travel_time_ngapara_zipra
def speed_limit_zone_fraction : ℝ := 0.25
def speed_reduction_factor : ℝ := 0.5
def travel_time_zipra_varnasi : ℝ := 0.75 * travel_time_ningi_zipra

-- Total adjusted travel time from Ningi to Zipra including speed limit delay
def adjusted_travel_time_ningi_zipra : ℝ :=
  let delayed_time := speed_limit_zone_fraction * travel_time_ningi_zipra * (2 - speed_reduction_factor)
  travel_time_ningi_zipra + delayed_time

-- Total travel time in the day
def total_travel_time : ℝ :=
  travel_time_ngapara_zipra + adjusted_travel_time_ningi_zipra + travel_time_zipra_varnasi

-- Proposition to prove
theorem car_total_travel_time : total_travel_time = 156 :=
by
  -- We skip the proof for now
  sorry

end car_total_travel_time_l239_239107


namespace sally_initial_orange_balloons_l239_239909

def initial_orange_balloons (found_orange : ℝ) (total_orange : ℝ) : ℝ := 
  total_orange - found_orange

theorem sally_initial_orange_balloons : initial_orange_balloons 2.0 11 = 9 := 
by
  sorry

end sally_initial_orange_balloons_l239_239909


namespace einstein_fundraising_l239_239026

def boxes_of_pizza : Nat := 15
def packs_of_potato_fries : Nat := 40
def cans_of_soda : Nat := 25
def price_per_box : ℝ := 12
def price_per_pack : ℝ := 0.3
def price_per_can : ℝ := 2
def goal_amount : ℝ := 500

theorem einstein_fundraising : goal_amount - (boxes_of_pizza * price_per_box + packs_of_potato_fries * price_per_pack + cans_of_soda * price_per_can) = 258 := by
  sorry

end einstein_fundraising_l239_239026


namespace total_money_taken_l239_239755

def individual_bookings : ℝ := 12000
def group_bookings : ℝ := 16000
def returned_due_to_cancellations : ℝ := 1600

def total_taken (individual_bookings : ℝ) (group_bookings : ℝ) (returned_due_to_cancellations : ℝ) : ℝ :=
  (individual_bookings + group_bookings) - returned_due_to_cancellations

theorem total_money_taken :
  total_taken individual_bookings group_bookings returned_due_to_cancellations = 26400 := by
  sorry

end total_money_taken_l239_239755


namespace domain_of_g_l239_239949

theorem domain_of_g : ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 1 ≠ 0 :=
by
  intro t
  sorry

end domain_of_g_l239_239949


namespace product_of_fractions_is_25_div_324_l239_239738

noncomputable def product_of_fractions : ℚ := 
  (10 / 6) * (4 / 20) * (20 / 12) * (16 / 32) * 
  (40 / 24) * (8 / 40) * (60 / 36) * (32 / 64)

theorem product_of_fractions_is_25_div_324 : product_of_fractions = 25 / 324 := 
  sorry

end product_of_fractions_is_25_div_324_l239_239738


namespace range_of_m_l239_239354

-- Define the sets A and B
def setA := {x : ℝ | abs (x - 1) < 2}
def setB (m : ℝ) := {x : ℝ | x >= m}

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), (setA ∩ setB m = setA) → m <= -1 :=
by
  sorry

end range_of_m_l239_239354


namespace amount_borrowed_l239_239555

variable (P : ℝ)
variable (interest_paid : ℝ) -- Interest paid on borrowing
variable (interest_earned : ℝ) -- Interest earned on lending
variable (gain_per_year : ℝ)

variable (h1 : interest_paid = P * 4 * 2 / 100)
variable (h2 : interest_earned = P * 6 * 2 / 100)
variable (h3 : gain_per_year = 160)
variable (h4 : gain_per_year = (interest_earned - interest_paid) / 2)

theorem amount_borrowed : P = 8000 := by
  sorry

end amount_borrowed_l239_239555


namespace johns_weekly_allowance_l239_239392

variable (A : ℝ)

theorem johns_weekly_allowance 
  (h1 : ∃ A : ℝ, A > 0) 
  (h2 : (4/15) * A = 0.75) : 
  A = 2.8125 := 
by 
  -- Proof can be filled in here
  sorry

end johns_weekly_allowance_l239_239392


namespace percentage_is_60_l239_239081

-- Definitions based on the conditions
def fraction_value (x : ℕ) : ℕ := x / 3
def percentage_less_value (x p : ℕ) : ℕ := x - (p * x) / 100

-- Lean statement based on the mathematically equivalent proof problem
theorem percentage_is_60 : ∀ (x p : ℕ), x = 180 → fraction_value x = 60 → percentage_less_value 60 p = 24 → p = 60 :=
by
  intros x p H1 H2 H3
  -- Proof is not required, so we use sorry
  sorry

end percentage_is_60_l239_239081


namespace place_numbers_l239_239277

theorem place_numbers (a b c d : ℕ) (hab : Nat.gcd a b = 1) (hac : Nat.gcd a c = 1) 
  (had : Nat.gcd a d = 1) (hbc : Nat.gcd b c = 1) (hbd : Nat.gcd b d = 1) 
  (hcd : Nat.gcd c d = 1) :
  ∃ (bc ad ab cd abcd : ℕ), 
    bc = b * c ∧ ad = a * d ∧ ab = a * b ∧ cd = c * d ∧ abcd = a * b * c * d ∧
    Nat.gcd bc abcd > 1 ∧ Nat.gcd ad abcd > 1 ∧ Nat.gcd ab abcd > 1 ∧ 
    Nat.gcd cd abcd > 1 ∧
    Nat.gcd ab cd = 1 ∧ Nat.gcd ab ad = 1 ∧ Nat.gcd ab bc = 1 ∧ 
    Nat.gcd cd ad = 1 ∧ Nat.gcd cd bc = 1 ∧ Nat.gcd ad bc = 1 :=
by
  sorry

end place_numbers_l239_239277


namespace viewers_difference_l239_239218

theorem viewers_difference :
  let second_game := 80
  let first_game := second_game - 20
  let third_game := second_game + 15
  let fourth_game := third_game + (third_game / 10)
  let total_last_week := 350
  let total_this_week := first_game + second_game + third_game + fourth_game
  total_this_week - total_last_week = -10 := 
by
  sorry

end viewers_difference_l239_239218


namespace mark_old_bills_l239_239344

noncomputable def old_hourly_wage : ℝ := 40
noncomputable def new_hourly_wage : ℝ := 42
noncomputable def work_hours_per_week : ℝ := 8 * 5
noncomputable def personal_trainer_cost_per_week : ℝ := 100
noncomputable def leftover_after_expenses : ℝ := 980

noncomputable def new_weekly_earnings := new_hourly_wage * work_hours_per_week
noncomputable def total_weekly_spending_after_raise := leftover_after_expenses + personal_trainer_cost_per_week
noncomputable def old_bills_per_week := new_weekly_earnings - total_weekly_spending_after_raise

theorem mark_old_bills : old_bills_per_week = 600 := by
  sorry

end mark_old_bills_l239_239344


namespace pants_cost_l239_239813

theorem pants_cost (P : ℝ) : 
(80 + 3 * P + 300) * 0.90 = 558 → P = 80 :=
by
  sorry

end pants_cost_l239_239813


namespace exists_positive_integer_m_l239_239300

theorem exists_positive_integer_m (n : ℕ) (hn : 0 < n) : ∃ m : ℕ, 0 < m ∧ 7^n ∣ (3^m + 5^m - 1) :=
sorry

end exists_positive_integer_m_l239_239300


namespace frequency_even_numbers_facing_up_l239_239831

theorem frequency_even_numbers_facing_up (rolls : ℕ) (event_occurrences : ℕ) (h_rolls : rolls = 100) (h_event : event_occurrences = 47) : (event_occurrences / (rolls : ℝ)) = 0.47 :=
by
  sorry

end frequency_even_numbers_facing_up_l239_239831


namespace min_value_expr_min_value_achieved_l239_239833

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4*x + 1/x^4 ≥ 5 :=
by
  sorry

theorem min_value_achieved (x : ℝ) : x = 1 → 4*x + 1/x^4 = 5 :=
by
  sorry

end min_value_expr_min_value_achieved_l239_239833


namespace no_base_satisfies_l239_239839

def e : ℕ := 35

theorem no_base_satisfies :
  ∀ (base : ℝ), (1 / 5)^e * (1 / 4)^18 ≠ 1 / 2 * (base)^35 :=
by
  sorry

end no_base_satisfies_l239_239839


namespace no_distinct_natural_numbers_eq_sum_and_cubes_eq_l239_239691

theorem no_distinct_natural_numbers_eq_sum_and_cubes_eq:
  ∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
  → a^3 + b^3 = c^3 + d^3
  → a + b = c + d
  → false := 
by
  intros
  sorry

end no_distinct_natural_numbers_eq_sum_and_cubes_eq_l239_239691


namespace kabulek_four_digits_l239_239174

def isKabulekNumber (N: ℕ) : Prop :=
  let a := N / 100
  let b := N % 100
  (a + b) ^ 2 = N

theorem kabulek_four_digits :
  {N : ℕ | 1000 ≤ N ∧ N < 10000 ∧ isKabulekNumber N} = {2025, 3025, 9801} :=
by sorry

end kabulek_four_digits_l239_239174


namespace least_distinct_values_l239_239895

variable (L : List Nat) (h_len : L.length = 2023) (mode : Nat) 
variable (h_mode_unique : ∀ x ∈ L, L.count x ≤ 15 → x = mode)
variable (h_mode_count : L.count mode = 15)

theorem least_distinct_values : ∃ k, k = 145 ∧ (∀ d ∈ L, List.count d L ≤ 15) :=
by
  sorry

end least_distinct_values_l239_239895


namespace inequality_problem_l239_239529

theorem inequality_problem (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by {
  sorry
}

end inequality_problem_l239_239529


namespace isosceles_triangle_largest_angle_l239_239710

theorem isosceles_triangle_largest_angle (a b c : ℝ) 
  (h1 : a = b)
  (h2 : c + 50 + 50 = 180) : 
  c = 80 :=
by sorry

end isosceles_triangle_largest_angle_l239_239710


namespace solve_for_s_l239_239181

theorem solve_for_s (r s : ℝ) (h1 : 1 < r) (h2 : r < s) (h3 : 1 / r + 1 / s = 3 / 4) (h4 : r * s = 8) : s = 4 :=
sorry

end solve_for_s_l239_239181


namespace average_after_discard_l239_239861

theorem average_after_discard (sum_50 : ℝ) (avg_50 : sum_50 = 2200) (a b : ℝ) (h1 : a = 45) (h2 : b = 55) :
  (sum_50 - (a + b)) / 48 = 43.75 :=
by
  -- Given conditions: sum_50 = 2200, a = 45, b = 55
  -- We need to prove (sum_50 - (a + b)) / 48 = 43.75
  sorry

end average_after_discard_l239_239861


namespace problem1_problem2_l239_239070

theorem problem1 : (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) * Real.sin (40 * Real.pi / 180) = -1 := 
by
  sorry

theorem problem2 (x : ℝ) : 
  (2 * Real.cos x ^ 4 - 2 * Real.cos x ^ 2 + 1 / 2) /
  (2 * Real.tan (Real.pi / 4 - x) * Real.sin (Real.pi / 4 + x) ^ 2) = 
  Real.sin (2 * x) / 4 := 
by
  sorry

end problem1_problem2_l239_239070


namespace find_width_of_room_l239_239688

variable (length : ℕ) (total_carpet_owned : ℕ) (additional_carpet_needed : ℕ)
variable (total_area : ℕ) (width : ℕ)

theorem find_width_of_room
  (h1 : length = 11) 
  (h2 : total_carpet_owned = 16) 
  (h3 : additional_carpet_needed = 149)
  (h4 : total_area = total_carpet_owned + additional_carpet_needed) 
  (h5 : total_area = length * width) :
  width = 15 := by
    sorry

end find_width_of_room_l239_239688


namespace slope_of_line_l239_239952

-- Define the point and the line equation with a generic slope
def point : ℝ × ℝ := (-1, 2)

def line (a : ℝ) := a * (point.fst) + (point.snd) - 4 = 0

-- The main theorem statement
theorem slope_of_line (a : ℝ) (h : line a) : ∃ m : ℝ, m = 2 :=
by
  -- The slope of the line derived from the equation and condition
  sorry

end slope_of_line_l239_239952


namespace cube_sum_gt_l239_239970

variable (a b c d : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
variable (h1 : a + b = c + d)
variable (h2 : a^2 + b^2 > c^2 + d^2)

theorem cube_sum_gt : a^3 + b^3 > c^3 + d^3 := by
  sorry

end cube_sum_gt_l239_239970


namespace extra_amount_spent_on_shoes_l239_239251

theorem extra_amount_spent_on_shoes (total_cost shirt_cost shoes_cost: ℝ) 
  (h1: total_cost = 300) (h2: shirt_cost = 97) 
  (h3: shoes_cost > 2 * shirt_cost)
  (h4: shirt_cost + shoes_cost = total_cost): 
  shoes_cost - 2 * shirt_cost = 9 :=
by
  sorry

end extra_amount_spent_on_shoes_l239_239251


namespace domain_of_f_l239_239882

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 2*x - 3)

theorem domain_of_f : 
  {x : ℝ | (x^2 - 2*x - 3) ≠ 0} = {x : ℝ | x < -1} ∪ {x : ℝ | -1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l239_239882


namespace chlorine_weight_is_35_l239_239423

def weight_Na : Nat := 23
def weight_O : Nat := 16
def molecular_weight : Nat := 74

theorem chlorine_weight_is_35 (Cl : Nat) 
  (h : molecular_weight = weight_Na + Cl + weight_O) : 
  Cl = 35 := by
  -- Proof placeholder
  sorry

end chlorine_weight_is_35_l239_239423


namespace willie_cream_l239_239387

theorem willie_cream : ∀ (total_cream needed_cream: ℕ), total_cream = 300 → needed_cream = 149 → (total_cream - needed_cream) = 151 :=
by
  intros total_cream needed_cream h1 h2
  sorry

end willie_cream_l239_239387


namespace find_b_l239_239857

theorem find_b (b : ℤ) :
  ∃ (r₁ r₂ : ℤ), (r₁ = -9) ∧ (r₁ * r₂ = 36) ∧ (r₁ + r₂ = -b) → b = 13 :=
by {
  sorry
}

end find_b_l239_239857


namespace parallel_vectors_x_value_l239_239363

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

-- Define the condition that vectors are parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- State the problem: if a and b are parallel, then x = 1/2
theorem parallel_vectors_x_value (x : ℝ) (h : is_parallel a (b x)) : x = 1/2 :=
by
  sorry

end parallel_vectors_x_value_l239_239363


namespace value_of_x_minus_y_l239_239336

theorem value_of_x_minus_y 
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : 3 * x - y = 8) :
  x - y = 3 := by
  sorry

end value_of_x_minus_y_l239_239336


namespace correct_total_count_l239_239148

variable (x : ℕ)

-- Define the miscalculation values
def value_of_quarter := 25
def value_of_dime := 10
def value_of_half_dollar := 50
def value_of_nickel := 5

-- Calculate the individual overestimations and underestimations
def overestimation_from_quarters := (value_of_quarter - value_of_dime) * (2 * x)
def underestimation_from_half_dollars := (value_of_half_dollar - value_of_nickel) * x

-- Calculate the net correction needed
def net_correction := overestimation_from_quarters - underestimation_from_half_dollars

theorem correct_total_count :
  net_correction x = 15 * x :=
by
  sorry

end correct_total_count_l239_239148


namespace hh3_eq_2943_l239_239301

-- Define the function h
def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

-- Prove that h(h(3)) = 2943
theorem hh3_eq_2943 : h (h 3) = 2943 :=
by
  sorry

end hh3_eq_2943_l239_239301


namespace sum_tens_units_digit_9_pow_1001_l239_239381

-- Define a function to extract the last two digits of a number
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Define a function to extract the tens digit
def tens_digit (n : ℕ) : ℕ := (last_two_digits n) / 10

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := (last_two_digits n) % 10

-- The main theorem
theorem sum_tens_units_digit_9_pow_1001 :
  tens_digit (9 ^ 1001) + units_digit (9 ^ 1001) = 9 :=
by
  sorry

end sum_tens_units_digit_9_pow_1001_l239_239381


namespace original_fraction_2_7_l239_239525

theorem original_fraction_2_7 (N D : ℚ) : 
  (1.40 * N) / (0.50 * D) = 4 / 5 → N / D = 2 / 7 :=
by
  intro h
  sorry

end original_fraction_2_7_l239_239525
