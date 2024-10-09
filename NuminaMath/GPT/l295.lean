import Mathlib

namespace problem_1_problem_2_l295_29597

-- Definitions for problem (1)
def p (x a : ℝ) := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0

-- Statement for problem (1)
theorem problem_1 (a : ℝ) (h : p 1 a ∧ q x) : 2 < x ∧ x < 3 :=
by 
  sorry

-- Definitions for problem (2)
def neg_p (x a : ℝ) := ¬ (x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0)
def neg_q (x : ℝ) := ¬ (x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0)

-- Statement for problem (2)
theorem problem_2 (a : ℝ) (h : ∀ x, neg_p x a → neg_q x ∧ ¬ (neg_q x → neg_p x a)) : 1 < a ∧ a ≤ 2 :=
by 
  sorry

end problem_1_problem_2_l295_29597


namespace percentage_difference_l295_29502

open scoped Classical

theorem percentage_difference (original_number new_number : ℕ) (h₀ : original_number = 60) (h₁ : new_number = 30) :
  (original_number - new_number) / original_number * 100 = 50 :=
by
      sorry

end percentage_difference_l295_29502


namespace vector_subtraction_l295_29554

/-
Define the vectors we are working with.
-/
def v1 : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-8]]
def v2 : Matrix (Fin 2) (Fin 1) ℤ := ![![2], ![-6]]
def scalar : ℤ := 5
def result : Matrix (Fin 2) (Fin 1) ℤ := ![![-7], ![22]]

/-
The statement of the proof problem.
-/
theorem vector_subtraction : v1 - scalar • v2 = result := 
by
  sorry

end vector_subtraction_l295_29554


namespace gcd_polynomial_multiple_l295_29556

theorem gcd_polynomial_multiple (b : ℤ) (h : b % 2373 = 0) : Int.gcd (b^2 + 13 * b + 40) (b + 5) = 5 :=
by
  sorry

end gcd_polynomial_multiple_l295_29556


namespace bridge_length_correct_l295_29552

noncomputable def length_of_bridge 
  (train_length : ℝ) 
  (time_to_cross : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  (train_speed_kmph * (5 / 18) * time_to_cross) - train_length

theorem bridge_length_correct :
  length_of_bridge 120 31.99744020478362 36 = 199.9744020478362 :=
by
  -- Skipping the proof details
  sorry

end bridge_length_correct_l295_29552


namespace polynomial_square_b_value_l295_29565

theorem polynomial_square_b_value (a b p q : ℝ) (h : (∀ x : ℝ, x^4 + x^3 - x^2 + a * x + b = (x^2 + p * x + q)^2)) : b = 25/64 := by
  sorry

end polynomial_square_b_value_l295_29565


namespace poly_expansion_l295_29517

def poly1 (z : ℝ) := 5 * z^3 + 4 * z^2 - 3 * z + 7
def poly2 (z : ℝ) := 2 * z^4 - z^3 + z - 2
def poly_product (z : ℝ) := 10 * z^7 + 6 * z^6 - 10 * z^5 + 22 * z^4 - 13 * z^3 - 11 * z^2 + 13 * z - 14

theorem poly_expansion (z : ℝ) : poly1 z * poly2 z = poly_product z := by
  sorry

end poly_expansion_l295_29517


namespace solve_equation1_solve_equation2_l295_29594

open Real

theorem solve_equation1 (x : ℝ) : (x^2 - 4 * x + 3 = 0) ↔ (x = 1 ∨ x = 3) := by
  sorry

theorem solve_equation2 (x : ℝ) : (x * (x - 2) = 2 * (2 - x)) ↔ (x = 2 ∨ x = -2) := by
  sorry

end solve_equation1_solve_equation2_l295_29594


namespace empty_set_is_d_l295_29541

open Set

theorem empty_set_is_d : {x : ℝ | x^2 - x + 1 = 0} = ∅ :=
by
  sorry

end empty_set_is_d_l295_29541


namespace find_other_x_intercept_l295_29577

theorem find_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 2 → y = -3) (h_x_intercept : ∀ x, x = 5 → y = 0) : 
  ∃ x, x = -1 ∧ y = 0 := 
sorry

end find_other_x_intercept_l295_29577


namespace rachel_more_than_adam_l295_29529

variable (R J A : ℕ)

def condition1 := R = 75
def condition2 := R = J - 6
def condition3 := R > A
def condition4 := (R + J + A) / 3 = 72

theorem rachel_more_than_adam
  (h1 : condition1 R)
  (h2 : condition2 R J)
  (h3 : condition3 R A)
  (h4 : condition4 R J A) : 
  R - A = 15 := 
by
  sorry

end rachel_more_than_adam_l295_29529


namespace geometric_sum_over_term_l295_29537

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def geometric_term (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem geometric_sum_over_term (a₁ : ℝ) (q : ℝ) (h₁ : q = 3) :
  (geometric_sum a₁ q 4) / (geometric_term a₁ q 4) = 40 / 27 := by
  sorry

end geometric_sum_over_term_l295_29537


namespace range_of_x_for_sqrt_meaningful_l295_29547

theorem range_of_x_for_sqrt_meaningful (x : ℝ) (h : x + 2 ≥ 0) : x ≥ -2 :=
by {
  sorry
}

end range_of_x_for_sqrt_meaningful_l295_29547


namespace rose_initial_rice_l295_29553

theorem rose_initial_rice : 
  ∀ (R : ℝ), (R - 9 / 10 * R - 1 / 4 * (R - 9 / 10 * R) = 0.75) → (R = 10) :=
by
  intro R h
  sorry

end rose_initial_rice_l295_29553


namespace opposite_numbers_l295_29583

theorem opposite_numbers
  (odot otimes : ℝ)
  (x y : ℝ)
  (h1 : 6 * x + odot * y = 3)
  (h2 : 2 * x + otimes * y = -1)
  (h_add : 6 * x + odot * y + (2 * x + otimes * y) = 2) :
  odot + otimes = 0 := by
  sorry

end opposite_numbers_l295_29583


namespace incorrect_propositions_l295_29530

theorem incorrect_propositions :
  ¬ (∀ P : Prop, P → P) ∨
  (¬ (∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x : ℝ, x^2 - x > 0)) ∨
  (∀ (R : Type) (f : R → Prop), (∀ r, f r → ∃ r', f r') = ∃ r, f r ∧ ∃ r', f r') ∨
  (∀ (x : ℝ), x ≠ 3 → abs x = 3 → x = 3) :=
by sorry

end incorrect_propositions_l295_29530


namespace erica_time_is_65_l295_29580

-- Definitions for the conditions
def dave_time : ℕ := 10
def chuck_time : ℕ := 5 * dave_time
def erica_time : ℕ := chuck_time + 3 * chuck_time / 10

-- The proof statement
theorem erica_time_is_65 : erica_time = 65 := by
  sorry

end erica_time_is_65_l295_29580


namespace henry_final_price_l295_29524

-- Definitions based on the conditions in the problem
def price_socks : ℝ := 5
def price_tshirt : ℝ := price_socks + 10
def price_jeans : ℝ := 2 * price_tshirt
def discount_jeans : ℝ := 0.15 * price_jeans
def discounted_price_jeans : ℝ := price_jeans - discount_jeans
def sales_tax_jeans : ℝ := 0.08 * discounted_price_jeans
def final_price_jeans : ℝ := discounted_price_jeans + sales_tax_jeans

-- Statement to prove
theorem henry_final_price : final_price_jeans = 27.54 := by
  sorry

end henry_final_price_l295_29524


namespace row_col_value_2002_2003_l295_29512

theorem row_col_value_2002_2003 :
  let base_num := (2003 - 1)^2 + 1 
  let result := base_num + 2001 
  result = 2002 * 2003 :=
by
  sorry

end row_col_value_2002_2003_l295_29512


namespace probability_of_drawing_two_red_shoes_l295_29548

/-- Given there are 7 red shoes and 3 green shoes, 
    and a total of 10 shoes, if two shoes are drawn randomly,
    prove that the probability of drawing both shoes as red is 7/15. -/
theorem probability_of_drawing_two_red_shoes :
  let total_shoes := 10
  let red_shoes := 7
  let green_shoes := 3
  let total_ways := Nat.choose total_shoes 2
  let red_ways := Nat.choose red_shoes 2
  (1 : ℚ) * red_ways / total_ways = 7 / 15  := by
  sorry

end probability_of_drawing_two_red_shoes_l295_29548


namespace div_ad_bc_l295_29539

theorem div_ad_bc (a b c d : ℤ) (h : (a - c) ∣ (a * b + c * d)) : (a - c) ∣ (a * d + b * c) :=
sorry

end div_ad_bc_l295_29539


namespace product_of_20_random_digits_ends_with_zero_l295_29528

noncomputable def probability_product_ends_in_zero : ℝ := 
  (1 - (9 / 10)^20) +
  (9 / 10)^20 * (1 - (5 / 9)^20) * (1 - (8 / 9)^19)

theorem product_of_20_random_digits_ends_with_zero : 
  abs (probability_product_ends_in_zero - 0.988) < 0.001 :=
by
  sorry

end product_of_20_random_digits_ends_with_zero_l295_29528


namespace double_decker_bus_total_capacity_l295_29509

-- Define conditions for the lower floor seating
def lower_floor_left_seats : Nat := 15
def lower_floor_right_seats : Nat := 12
def lower_floor_priority_seats : Nat := 4

-- Each seat on the left and right side of the lower floor holds 2 people
def lower_floor_left_capacity : Nat := lower_floor_left_seats * 2
def lower_floor_right_capacity : Nat := lower_floor_right_seats * 2
def lower_floor_priority_capacity : Nat := lower_floor_priority_seats * 1

-- Define conditions for the upper floor seating
def upper_floor_left_seats : Nat := 20
def upper_floor_right_seats : Nat := 20
def upper_floor_back_capacity : Nat := 15

-- Each seat on the left and right side of the upper floor holds 3 people
def upper_floor_left_capacity : Nat := upper_floor_left_seats * 3
def upper_floor_right_capacity : Nat := upper_floor_right_seats * 3

-- Total capacity of lower and upper floors
def lower_floor_total_capacity : Nat := lower_floor_left_capacity + lower_floor_right_capacity + lower_floor_priority_capacity
def upper_floor_total_capacity : Nat := upper_floor_left_capacity + upper_floor_right_capacity + upper_floor_back_capacity

-- Assert the total capacity
def bus_total_capacity : Nat := lower_floor_total_capacity + upper_floor_total_capacity

-- Prove that the total bus capacity is 193 people
theorem double_decker_bus_total_capacity : bus_total_capacity = 193 := by
  sorry

end double_decker_bus_total_capacity_l295_29509


namespace value_of_b_minus_a_l295_29576

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2)

theorem value_of_b_minus_a (a b : ℝ) (h1 : ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1 : ℝ) 2) (h2 : ∀ x, f x = 2 * Real.sin (x / 2)) : 
  b - a ≠ 14 * Real.pi / 3 :=
sorry

end value_of_b_minus_a_l295_29576


namespace not_divisible_by_5_l295_29533

theorem not_divisible_by_5 (b : ℕ) : b = 6 ↔ ¬ (5 ∣ (2 * b ^ 3 - 2 * b ^ 2 + 2 * b - 1)) :=
sorry

end not_divisible_by_5_l295_29533


namespace problem1_problem2_l295_29532

-- Definitions based on conditions in the problem
def seq_sum (a : ℕ) (n : ℕ) : ℕ := a * 2^n - 1
def a1 (a : ℕ) : ℕ := seq_sum a 1
def a4 (a : ℕ) : ℕ := seq_sum a 4 - seq_sum a 3

-- Problem statement 1
theorem problem1 (a : ℕ) (h : a = 3) : a1 a = 5 ∧ a4 a = 24 := by 
  sorry

-- Geometric sequence conditions
def is_geometric (a_n : ℕ → ℕ) : Prop :=
  ∃ q ≠ 1, ∀ n, a_n (n + 1) = q * a_n n

-- Definitions for the geometric sequence part
def a_n (a : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 2 * a - 1
  else if n = 2 then 2 * a
  else if n = 3 then 4 * a
  else 0 -- Simplifying for the first few terms only

-- Problem statement 2
theorem problem2 : (∃ a : ℕ, is_geometric (a_n a)) → ∃ a : ℕ, a = 1 := by
  sorry

end problem1_problem2_l295_29532


namespace sum_of_coefficients_l295_29593

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (∀ x : ℝ, (3 * x - 2)^6 = a_0 + a_1 * (2 * x - 1) + a_2 * (2 * x - 1)^2 + a_3 * (2 * x - 1)^3 + a_4 * (2 * x - 1)^4 + a_5 * (2 * x - 1)^5 + a_6 * (2 * x - 1)^6) ->
  a_1 + a_3 + a_5 = -63 / 2 := by
  sorry

end sum_of_coefficients_l295_29593


namespace original_garden_side_length_l295_29527

theorem original_garden_side_length (a : ℝ) (h : (a + 3)^2 = 2 * a^2 + 9) : a = 6 :=
by
  sorry

end original_garden_side_length_l295_29527


namespace parallelogram_area_l295_29514

-- Defining the vectors u and z
def u : ℝ × ℝ := (4, -1)
def z : ℝ × ℝ := (9, -3)

-- Computing the area of parallelogram formed by vectors u and z
def area_parallelogram (u z : ℝ × ℝ) : ℝ :=
  abs (u.1 * (z.2 + u.2) - u.2 * (z.1 + u.1))

-- Lean statement asserting that the area of the parallelogram is 3
theorem parallelogram_area : area_parallelogram u z = 3 := by
  sorry

end parallelogram_area_l295_29514


namespace shirt_pants_outfits_l295_29518

theorem shirt_pants_outfits
  (num_shirts : ℕ) (num_pants : ℕ) (num_formal_pants : ℕ) (num_casual_pants : ℕ) (num_assignee_shirts : ℕ) :
  num_shirts = 5 →
  num_pants = 6 →
  num_formal_pants = 3 →
  num_casual_pants = 3 →
  num_assignee_shirts = 3 →
  (num_casual_pants * num_shirts) + (num_formal_pants * num_assignee_shirts) = 24 :=
by
  intros h_shirts h_pants h_formal h_casual h_assignee
  sorry

end shirt_pants_outfits_l295_29518


namespace probability_three_dice_same_number_is_1_div_36_l295_29575

noncomputable def probability_same_number_three_dice : ℚ :=
  let first_die := 1
  let second_die := 1 / 6
  let third_die := 1 / 6
  first_die * second_die * third_die

theorem probability_three_dice_same_number_is_1_div_36 : probability_same_number_three_dice = 1 / 36 :=
  sorry

end probability_three_dice_same_number_is_1_div_36_l295_29575


namespace total_eggs_michael_has_l295_29589

-- Define the initial number of crates
def initial_crates : ℕ := 6

-- Define the number of crates given to Susan
def crates_given_to_susan : ℕ := 2

-- Define the number of crates bought on Thursday
def crates_bought_thursday : ℕ := 5

-- Define the number of eggs per crate
def eggs_per_crate : ℕ := 30

-- Theorem stating the total number of eggs Michael has now
theorem total_eggs_michael_has :
  (initial_crates - crates_given_to_susan + crates_bought_thursday) * eggs_per_crate = 270 :=
sorry

end total_eggs_michael_has_l295_29589


namespace rearranged_number_divisible_by_27_l295_29585

theorem rearranged_number_divisible_by_27 (n m : ℕ) (hn : m = 3 * n) 
  (hdigits : ∀ a b : ℕ, (a ∈ n.digits 10 ↔ b ∈ m.digits 10)) : 27 ∣ m :=
sorry

end rearranged_number_divisible_by_27_l295_29585


namespace find_f_of_one_l295_29555

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_of_one : f 1 = 2 := 
by
  sorry

end find_f_of_one_l295_29555


namespace exist_elem_not_in_union_l295_29588

-- Assume closed sets
def isClosedSet (S : Set ℝ) : Prop :=
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a - b) ∈ S

-- The theorem to prove
theorem exist_elem_not_in_union {S1 S2 : Set ℝ} (hS1 : isClosedSet S1) (hS2 : isClosedSet S2) :
  S1 ⊂ (Set.univ : Set ℝ) → S2 ⊂ (Set.univ : Set ℝ) → ∃ c : ℝ, c ∉ S1 ∪ S2 :=
by
  intro h1 h2
  sorry

end exist_elem_not_in_union_l295_29588


namespace waiter_initial_tables_l295_29579

theorem waiter_initial_tables
  (T : ℝ)
  (H1 : (T - 12.0) * 8.0 = 256) :
  T = 44.0 :=
sorry

end waiter_initial_tables_l295_29579


namespace cricket_bat_profit_percentage_correct_football_profit_percentage_correct_l295_29573

noncomputable def cricket_bat_selling_price : ℝ := 850
noncomputable def cricket_bat_profit : ℝ := 215
noncomputable def cricket_bat_cost_price : ℝ := cricket_bat_selling_price - cricket_bat_profit
noncomputable def cricket_bat_profit_percentage : ℝ := (cricket_bat_profit / cricket_bat_cost_price) * 100

noncomputable def football_selling_price : ℝ := 120
noncomputable def football_profit : ℝ := 45
noncomputable def football_cost_price : ℝ := football_selling_price - football_profit
noncomputable def football_profit_percentage : ℝ := (football_profit / football_cost_price) * 100

theorem cricket_bat_profit_percentage_correct :
  |cricket_bat_profit_percentage - 33.86| < 1e-2 :=
by sorry

theorem football_profit_percentage_correct :
  football_profit_percentage = 60 :=
by sorry

end cricket_bat_profit_percentage_correct_football_profit_percentage_correct_l295_29573


namespace interval_contains_root_l295_29506

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem interval_contains_root :
  f (-1) < 0 → 
  f 0 < 0 → 
  f 1 < 0 → 
  f 2 > 0 → 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intro h1 h2 h3 h4
  sorry

end interval_contains_root_l295_29506


namespace triangle_angle_A_l295_29586

variable {a b c : ℝ} {A : ℝ}

theorem triangle_angle_A (h : a^2 = b^2 + c^2 - b * c) : A = 2 * Real.pi / 3 :=
by
  sorry

end triangle_angle_A_l295_29586


namespace polynomial_sum_l295_29519

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l295_29519


namespace polynomial_inequality_solution_l295_29581

theorem polynomial_inequality_solution :
  {x : ℝ | x^3 - 4*x^2 - x + 20 > 0} = {x | x < -4} ∪ {x | 1 < x ∧ x < 5} ∪ {x | x > 5} :=
sorry

end polynomial_inequality_solution_l295_29581


namespace maxwell_meets_brad_l295_29568

theorem maxwell_meets_brad :
  ∃ t : ℝ, t = 2 ∧ 
  (∀ distance max_speed brad_speed start_time, 
   distance = 14 ∧ 
   max_speed = 4 ∧ 
   brad_speed = 6 ∧ 
   start_time = 1 → 
   max_speed * (t + start_time) + brad_speed * t = distance) :=
by
  use 1
  sorry

end maxwell_meets_brad_l295_29568


namespace smallest_positive_number_is_x2_l295_29521

noncomputable def x1 : ℝ := 14 - 4 * Real.sqrt 17
noncomputable def x2 : ℝ := 4 * Real.sqrt 17 - 14
noncomputable def x3 : ℝ := 23 - 7 * Real.sqrt 14
noncomputable def x4 : ℝ := 65 - 12 * Real.sqrt 34
noncomputable def x5 : ℝ := 12 * Real.sqrt 34 - 65

theorem smallest_positive_number_is_x2 :
  x2 = 4 * Real.sqrt 17 - 14 ∧
  (0 < x1 ∨ 0 < x2 ∨ 0 < x3 ∨ 0 < x4 ∨ 0 < x5) ∧
  (∀ x : ℝ, (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) → 0 < x → x2 ≤ x) := sorry

end smallest_positive_number_is_x2_l295_29521


namespace find_multiple_l295_29513
-- Importing Mathlib to access any necessary math definitions.

-- Define the constants based on the given conditions.
def Darwin_money : ℝ := 45
def Mia_money : ℝ := 110
def additional_amount : ℝ := 20

-- The Lean theorem which encapsulates the proof problem.
theorem find_multiple (x : ℝ) : 
  Mia_money = x * Darwin_money + additional_amount → x = 2 :=
by
  sorry

end find_multiple_l295_29513


namespace range_of_m_non_perpendicular_tangent_l295_29525

noncomputable def f (m x : ℝ) : ℝ := Real.exp x - m * x

theorem range_of_m_non_perpendicular_tangent (m : ℝ) :
  (∀ x : ℝ, (deriv (f m) x ≠ -2)) → m ≤ 2 :=
by
  sorry

end range_of_m_non_perpendicular_tangent_l295_29525


namespace number_of_children_l295_29545

-- Define the number of adults and their ticket price
def num_adults := 9
def adult_ticket_price := 11

-- Define the children's ticket price and the total cost difference
def child_ticket_price := 7
def cost_difference := 50

-- Define the total cost for adult tickets
def total_adult_cost := num_adults * adult_ticket_price

-- Given the conditions, prove that the number of children is 7
theorem number_of_children : ∃ c : ℕ, total_adult_cost = c * child_ticket_price + cost_difference ∧ c = 7 :=
by
  sorry

end number_of_children_l295_29545


namespace factor_expression_l295_29569

theorem factor_expression (a : ℝ) :
  (9 * a^4 + 105 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 4 * a^2 + 2 * a - 5) =
  (a - 3) * (11 * a^2 * (a + 1) - 2) :=
by
  sorry

end factor_expression_l295_29569


namespace contest_end_time_l295_29563

-- Definitions for the conditions
def start_time_pm : Nat := 15 -- 3:00 p.m. in 24-hour format
def duration_min : Nat := 720

-- Proof that the contest ended at 3:00 a.m.
theorem contest_end_time :
  let end_time := (start_time_pm + (duration_min / 60)) % 24
  end_time = 3 :=
by
  -- This would be the place to provide the proof
  sorry

end contest_end_time_l295_29563


namespace solve_eq_n_fact_plus_n_eq_n_pow_k_l295_29515

theorem solve_eq_n_fact_plus_n_eq_n_pow_k :
  ∀ (n k : ℕ), 0 < n → 0 < k → (n! + n = n^k ↔ (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3)) :=
by
  sorry

end solve_eq_n_fact_plus_n_eq_n_pow_k_l295_29515


namespace Theresa_game_scores_l295_29574

theorem Theresa_game_scores 
  (h_sum_10 : 9 + 5 + 4 + 7 + 6 + 2 + 4 + 8 + 3 + 7 = 55)
  (h_p11 : ∀ p11 : ℕ, p11 < 10 → (55 + p11) % 11 = 0)
  (h_p12 : ∀ p11 p12 : ℕ, p11 < 10 → p12 < 10 → ((55 + p11 + p12) % 12 = 0)) :
  ∃ p11 p12 : ℕ, p11 < 10 ∧ p12 < 10 ∧ (55 + p11) % 11 = 0 ∧ (55 + p11 + p12) % 12 = 0 ∧ p11 * p12 = 0 :=
by
  sorry

end Theresa_game_scores_l295_29574


namespace topsoil_cost_l295_29500

theorem topsoil_cost :
  let cubic_yard_to_cubic_foot := 27
  let cubic_feet_in_5_cubic_yards := 5 * cubic_yard_to_cubic_foot
  let cost_per_cubic_foot := 6
  let total_cost := cubic_feet_in_5_cubic_yards * cost_per_cubic_foot
  total_cost = 810 :=
by
  sorry

end topsoil_cost_l295_29500


namespace father_gave_8_candies_to_Billy_l295_29591

theorem father_gave_8_candies_to_Billy (candies_Billy : ℕ) (candies_Caleb : ℕ) (candies_Andy : ℕ) (candies_father : ℕ) 
  (candies_given_to_Caleb : ℕ) (candies_more_than_Caleb : ℕ) (candies_given_by_father_total : ℕ) :
  (candies_given_to_Caleb = 11) →
  (candies_Caleb = 11) →
  (candies_Andy = 9) →
  (candies_father = 36) →
  (candies_Andy = candies_Caleb + 4) →
  (candies_given_by_father_total = candies_given_to_Caleb + (candies_Andy - 9)) →
  (candies_father - candies_given_by_father_total = 8) →
  candies_Billy = 8 := 
by
  intros
  sorry

end father_gave_8_candies_to_Billy_l295_29591


namespace ram_actual_distance_from_base_l295_29560

def map_distance_between_mountains : ℝ := 312
def actual_distance_between_mountains : ℝ := 136
def ram_map_distance_from_base : ℝ := 28

theorem ram_actual_distance_from_base :
  ram_map_distance_from_base * (actual_distance_between_mountains / map_distance_between_mountains) = 12.205 :=
by sorry

end ram_actual_distance_from_base_l295_29560


namespace initial_soup_weight_l295_29542

theorem initial_soup_weight (W: ℕ) (h: W / 16 = 5): W = 40 :=
by
  sorry

end initial_soup_weight_l295_29542


namespace find_x_l295_29549

theorem find_x :
  ∃ x : ℚ, (1 / 3) * ((x + 8) + (8*x + 3) + (3*x + 9)) = 5*x - 9 ∧ x = 47 / 3 :=
by
  sorry

end find_x_l295_29549


namespace rectangle_area_ratio_l295_29535

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 :=
by
  sorry

end rectangle_area_ratio_l295_29535


namespace train_length_l295_29522

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) (conversion_factor : ℝ) :
  speed_kmh = 54 →
  time_s = 33333333333333336 / 1000000000000000 →
  bridge_length_m = 140 →
  conversion_factor = 1000 / 3600 →
  ∃ (train_length_m : ℝ), 
    speed_kmh * conversion_factor * time_s + bridge_length_m = train_length_m + bridge_length_m :=
by
  intros
  use 360
  sorry

end train_length_l295_29522


namespace Leela_Hotel_all_three_reunions_l295_29564

theorem Leela_Hotel_all_three_reunions
  (A B C : Finset ℕ)
  (hA : A.card = 80)
  (hB : B.card = 90)
  (hC : C.card = 70)
  (hAB : (A ∩ B).card = 30)
  (hAC : (A ∩ C).card = 25)
  (hBC : (B ∩ C).card = 20)
  (hABC : ((A ∪ B ∪ C)).card = 150) : 
  (A ∩ B ∩ C).card = 15 :=
by
  sorry

end Leela_Hotel_all_three_reunions_l295_29564


namespace find_eccentricity_of_ellipse_l295_29510

theorem find_eccentricity_of_ellipse
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (hx : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x, y) ∈ { p | (p.1^2 / a^2 + p.2^2 / b^2 = 1) })
  (hk : ∀ k x1 y1 x2 y2 : ℝ, y1 = k * x1 ∧ y2 = k * x2 → x1 ≠ x2 → (y1 = x1 * k ∧ y2 = x2 * k))  -- intersection points condition
  (hAB_AC : ∀ m n : ℝ, m ≠ 0 → (n - b) / m * (-n - b) / (-m) = -3/4 )
  : ∃ e : ℝ, e = 1/2 :=
sorry

end find_eccentricity_of_ellipse_l295_29510


namespace max_value_of_expression_l295_29596

theorem max_value_of_expression (x y : ℝ) (h : x + y = 4) :
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 ≤ 7225 / 28 :=
sorry

end max_value_of_expression_l295_29596


namespace probability_more_ones_than_sixes_l295_29599

theorem probability_more_ones_than_sixes :
  (∃ (p : ℚ), p = 1673 / 3888 ∧ 
  (∃ (d : Fin 6 → ℕ), 
  (∀ i, d i ≤ 4) ∧ 
  (∃ d1 d6 : ℕ, (1 ≤ d1 + d6 ∧ d1 + d6 ≤ 5 ∧ d1 > d6)))) :=
sorry

end probability_more_ones_than_sixes_l295_29599


namespace original_number_is_15_l295_29551

theorem original_number_is_15 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (N : ℕ) (h4 : 100 * a + 10 * b + c = m)
  (h5 : 100 * a +  10 * b +   c +
        100 * a +   c + 10 * b + 
        100 * b +  10 * a +   c +
        100 * b +   c + 10 * a + 
        100 * c +  10 * a +   b +
        100 * c +   b + 10 * a = 3315) :
  m = 15 :=
sorry

end original_number_is_15_l295_29551


namespace prime_factor_of_difference_l295_29503

theorem prime_factor_of_difference {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (h_neq : A ≠ B) :
  Nat.Prime 2 ∧ (∃ B : ℕ, 20 * B = 20 * B) :=
by
  sorry

end prime_factor_of_difference_l295_29503


namespace arrange_abc_l295_29578

theorem arrange_abc : 
  let a := Real.log 5 / Real.log 0.6
  let b := 2 ^ (4 / 5)
  let c := Real.sin 1
  a < c ∧ c < b := 
by
  sorry

end arrange_abc_l295_29578


namespace product_remainder_mod_5_l295_29590

theorem product_remainder_mod_5 : (2024 * 1980 * 1848 * 1720) % 5 = 0 := by
  sorry

end product_remainder_mod_5_l295_29590


namespace olivia_total_cost_l295_29562

-- Definitions based on conditions given in the problem.
def daily_rate : ℕ := 30 -- daily rate in dollars per day
def mileage_rate : ℕ := 25 -- mileage rate in cents per mile (converted to cents to avoid fractions)
def rental_days : ℕ := 3 -- number of days the car is rented
def miles_driven : ℕ := 500 -- number of miles driven

-- Calculate costs in cents to avoid fractions in the Lean theorem statement.
def daily_rental_cost : ℕ := daily_rate * rental_days * 100
def mileage_cost : ℕ := mileage_rate * miles_driven
def total_cost : ℕ := daily_rental_cost + mileage_cost

-- Final statement to be proved, converting total cost back to dollars.
theorem olivia_total_cost : (total_cost / 100) = 215 := by
  sorry

end olivia_total_cost_l295_29562


namespace boat_speed_in_still_water_l295_29571

variable (B S : ℝ)

-- conditions
def condition1 : Prop := B + S = 6
def condition2 : Prop := B - S = 2

-- question to answer
theorem boat_speed_in_still_water (h1 : condition1 B S) (h2 : condition2 B S) : B = 4 :=
by
  sorry

end boat_speed_in_still_water_l295_29571


namespace least_integer_sol_l295_29572

theorem least_integer_sol (x : ℤ) (h : |(2 : ℤ) * x + 7| ≤ 16) : x ≥ -11 := sorry

end least_integer_sol_l295_29572


namespace brownie_pan_dimensions_l295_29511

def brownie_dimensions (m n : ℕ) : Prop :=
  let numSectionsLength := m - 1
  let numSectionsWidth := n - 1
  let totalPieces := (numSectionsLength + 1) * (numSectionsWidth + 1)
  let interiorPieces := (numSectionsLength - 1) * (numSectionsWidth - 1)
  let perimeterPieces := totalPieces - interiorPieces
  (numSectionsLength = 3) ∧ (numSectionsWidth = 5) ∧ (interiorPieces = 2 * perimeterPieces)

theorem brownie_pan_dimensions :
  ∃ (m n : ℕ), brownie_dimensions m n ∧ m = 6 ∧ n = 12 :=
by
  existsi 6
  existsi 12
  unfold brownie_dimensions
  simp
  exact sorry

end brownie_pan_dimensions_l295_29511


namespace total_fruit_salads_is_1800_l295_29567

def Alaya_fruit_salads := 200
def Angel_fruit_salads := 2 * Alaya_fruit_salads
def Betty_fruit_salads := 3 * Angel_fruit_salads
def Total_fruit_salads := Alaya_fruit_salads + Angel_fruit_salads + Betty_fruit_salads

theorem total_fruit_salads_is_1800 : Total_fruit_salads = 1800 := by
  sorry

end total_fruit_salads_is_1800_l295_29567


namespace locus_of_projection_l295_29544

theorem locus_of_projection {a b c : ℝ} (h : (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2) :
  ∀ (x y : ℝ), (x, y) ∈ ({P : ℝ × ℝ | ∃ a b : ℝ, P = ((a * b^2) / (a^2 + b^2), (a^2 * b) / (a^2 + b^2)) ∧ (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2}) → 
    x^2 + y^2 = c^2 := 
sorry

end locus_of_projection_l295_29544


namespace symmetric_points_addition_l295_29598

theorem symmetric_points_addition 
  (m n : ℝ)
  (A : (ℝ × ℝ)) (B : (ℝ × ℝ))
  (hA : A = (2, m)) 
  (hB : B = (n, -1))
  (symmetry : A.1 = B.1 ∧ A.2 = -B.2) : 
  m + n = 3 :=
by
  sorry

end symmetric_points_addition_l295_29598


namespace marbles_given_by_Joan_l295_29584

def initial_yellow_marbles : ℝ := 86.0
def final_yellow_marbles : ℝ := 111.0

theorem marbles_given_by_Joan :
  final_yellow_marbles - initial_yellow_marbles = 25 := by
  sorry

end marbles_given_by_Joan_l295_29584


namespace journey_divided_into_portions_l295_29570

theorem journey_divided_into_portions
  (total_distance : ℕ)
  (speed : ℕ)
  (time : ℝ)
  (portion_distance : ℕ)
  (portions_covered : ℕ)
  (h1 : total_distance = 35)
  (h2 : speed = 40)
  (h3 : time = 0.7)
  (h4 : portions_covered = 4)
  (distance_covered := speed * time)
  (one_portion_distance := distance_covered / portions_covered)
  (total_portions := total_distance / one_portion_distance) :
  total_portions = 5 := 
sorry

end journey_divided_into_portions_l295_29570


namespace distance_walked_l295_29561

theorem distance_walked (D : ℝ) (t1 t2 : ℝ): 
  (t1 = D / 4) → 
  (t2 = D / 3) → 
  (t2 - t1 = 1 / 2) → 
  D = 6 := 
by
  sorry

end distance_walked_l295_29561


namespace total_votes_l295_29595

theorem total_votes (A B C D E : ℕ)
  (votes_A : ℕ) (votes_B : ℕ) (votes_C : ℕ) (votes_D : ℕ) (votes_E : ℕ)
  (dist_A : votes_A = 38 * A / 100)
  (dist_B : votes_B = 28 * B / 100)
  (dist_C : votes_C = 11 * C / 100)
  (dist_D : votes_D = 15 * D / 100)
  (dist_E : votes_E = 8 * E / 100)
  (redistrib_A : votes_A' = votes_A + 5 * A / 100)
  (redistrib_B : votes_B' = votes_B + 5 * B / 100)
  (redistrib_D : votes_D' = votes_D + 2 * D / 100)
  (total_A : votes_A' = 7320) :
  A = 17023 := 
sorry

end total_votes_l295_29595


namespace barrels_of_pitch_needed_l295_29559

-- Define the basic properties and conditions
def total_length_road := 16
def truckloads_per_mile := 3
def bags_of_gravel_per_truckload := 2
def gravel_to_pitch_ratio := 5
def miles_paved_first_day := 4
def miles_paved_second_day := 2 * miles_paved_first_day - 1
def miles_already_paved := miles_paved_first_day + miles_paved_second_day
def remaining_miles := total_length_road - miles_already_paved
def total_truckloads := truckloads_per_mile * remaining_miles
def total_bags_of_gravel := bags_of_gravel_per_truckload * total_truckloads
def barrels_of_pitch := total_bags_of_gravel / gravel_to_pitch_ratio

-- State the theorem to prove the number of barrels of pitch needed
theorem barrels_of_pitch_needed :
    barrels_of_pitch = 6 :=
by
    sorry

end barrels_of_pitch_needed_l295_29559


namespace pythagorean_theorem_l295_29534

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l295_29534


namespace pictures_per_coloring_book_l295_29526

theorem pictures_per_coloring_book
    (total_colored : ℕ)
    (remaining_pictures : ℕ)
    (two_books : ℕ)
    (h1 : total_colored = 20) 
    (h2 : remaining_pictures = 68) 
    (h3 : two_books = 2) :
  (total_colored + remaining_pictures) / two_books = 44 :=
by
  sorry

end pictures_per_coloring_book_l295_29526


namespace probability_allison_greater_l295_29558

theorem probability_allison_greater (A D S : ℕ) (prob_derek_less_than_4 : ℚ) (prob_sophie_less_than_4 : ℚ) : 
  (A > D) ∧ (A > S) → prob_derek_less_than_4 = 1 / 2 ∧ prob_sophie_less_than_4 = 2 / 3 → 
  (1 / 2 : ℚ) * (2 / 3 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end probability_allison_greater_l295_29558


namespace chimney_base_radius_l295_29536

-- Given conditions
def tinplate_length := 219.8
def tinplate_width := 125.6
def pi_approx := 3.14

def radius_length (circumference : Float) : Float :=
  circumference / (2 * pi_approx)

def radius_width (circumference : Float) : Float :=
  circumference / (2 * pi_approx)

theorem chimney_base_radius :
  radius_length tinplate_length = 35 ∧ radius_width tinplate_width = 20 :=
by 
  sorry

end chimney_base_radius_l295_29536


namespace minimum_draws_divisible_by_3_or_5_l295_29566

theorem minimum_draws_divisible_by_3_or_5 (n : ℕ) (h : n = 90) :
  ∃ k, k = 49 ∧ ∀ (draws : ℕ), draws < k → ¬ (∃ x, 1 ≤ x ∧ x ≤ n ∧ (x % 3 = 0 ∨ x % 5 = 0)) :=
by {
  sorry
}

end minimum_draws_divisible_by_3_or_5_l295_29566


namespace contrapositive_of_proposition_l295_29520

theorem contrapositive_of_proposition (a b : ℝ) : (a > b → a + 1 > b) ↔ (a + 1 ≤ b → a ≤ b) :=
sorry

end contrapositive_of_proposition_l295_29520


namespace multiply_98_102_l295_29508

theorem multiply_98_102 : 98 * 102 = 9996 :=
by sorry

end multiply_98_102_l295_29508


namespace angle_magnification_l295_29538

theorem angle_magnification (α : ℝ) (h : α = 20) : α = 20 := by
  sorry

end angle_magnification_l295_29538


namespace inequality_sqrt_sum_l295_29505

theorem inequality_sqrt_sum (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_sum_l295_29505


namespace determine_F_l295_29546

theorem determine_F (A H S M F : ℕ) (ha : 0 < A) (hh : 0 < H) (hs : 0 < S) (hm : 0 < M) (hf : 0 < F):
  (A * x + H * y = z) →
  (S * x + M * y = z) →
  (F * x = z) →
  (H > A) →
  (A ≠ H) →
  (S ≠ M) →
  (F ≠ A) →
  (F ≠ H) →
  (F ≠ S) →
  (F ≠ M) →
  x = z / F →
  y = ((F - A) / H * z) / z →
  F = (A * F - S * H) / (M - H) := sorry

end determine_F_l295_29546


namespace ball_is_green_probability_l295_29501

noncomputable def probability_green_ball : ℚ :=
  let containerI_red := 8
  let containerI_green := 4
  let containerII_red := 3
  let containerII_green := 5
  let containerIII_red := 4
  let containerIII_green := 6
  let probability_container := (1 : ℚ) / 3
  let probability_green_I := (containerI_green : ℚ) / (containerI_red + containerI_green)
  let probability_green_II := (containerII_green : ℚ) / (containerII_red + containerII_green)
  let probability_green_III := (containerIII_green : ℚ) / (containerIII_red + containerIII_green)
  probability_container * probability_green_I +
  probability_container * probability_green_II +
  probability_container * probability_green_III

theorem ball_is_green_probability :
  probability_green_ball = 187 / 360 :=
by
  -- The detailed proof is omitted and left as an exercise
  sorry

end ball_is_green_probability_l295_29501


namespace soccer_campers_l295_29550

theorem soccer_campers (total_campers : ℕ) (basketball_campers : ℕ) (football_campers : ℕ) (h1 : total_campers = 88) (h2 : basketball_campers = 24) (h3 : football_campers = 32) : 
  total_campers - (basketball_campers + football_campers) = 32 := 
by 
  -- Proof omitted
  sorry

end soccer_campers_l295_29550


namespace intersection_of_A_and_B_l295_29523

def setA (x : ℝ) : Prop := x^2 < 4
def setB : Set ℝ := {0, 1}

theorem intersection_of_A_and_B :
  {x : ℝ | setA x} ∩ setB = setB := by
  sorry

end intersection_of_A_and_B_l295_29523


namespace largest_c_for_minus3_in_range_of_quadratic_l295_29587

theorem largest_c_for_minus3_in_range_of_quadratic (c : ℝ) :
  (∃ x : ℝ, x^2 + 5*x + c = -3) ↔ c ≤ 13/4 :=
sorry

end largest_c_for_minus3_in_range_of_quadratic_l295_29587


namespace cricket_average_l295_29540

theorem cricket_average (A : ℝ) (h : 20 * A + 120 = 21 * (A + 4)) : A = 36 :=
by sorry

end cricket_average_l295_29540


namespace find_constants_eq_l295_29592

theorem find_constants_eq (P Q R : ℚ)
  (h : ∀ x, (x^2 - 5) = P * (x - 4) * (x - 6) + Q * (x - 1) * (x - 6) + R * (x - 1) * (x - 4)) :
  (P = -4 / 15) ∧ (Q = -11 / 6) ∧ (R = 31 / 10) :=
by
  sorry

end find_constants_eq_l295_29592


namespace gcd_75_100_l295_29516

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l295_29516


namespace proposition_four_l295_29507

variables (a b c : Type)

noncomputable def perpend_lines (a b : Type) : Prop := sorry
noncomputable def parallel_lines (a b : Type) : Prop := sorry

theorem proposition_four (a b c : Type) 
  (h1 : perpend_lines a b) (h2 : parallel_lines b c) :
  perpend_lines a c :=
sorry

end proposition_four_l295_29507


namespace rhombus_area_in_rectangle_l295_29543

theorem rhombus_area_in_rectangle :
  ∀ (l w : ℝ), 
  (∀ (A B C D : ℝ), 
    (2 * w = l) ∧ 
    (l * w = 72) →
    let diag1 := w 
    let diag2 := l 
    (1/2 * diag1 * diag2 = 36)) :=
by
  intros
  sorry

end rhombus_area_in_rectangle_l295_29543


namespace minutes_sean_played_each_day_l295_29582

-- Define the given conditions
def t : ℕ := 1512                               -- Total minutes played by Sean and Indira
def i : ℕ := 812                                -- Total minutes played by Indira
def d : ℕ := 14                                 -- Number of days Sean played

-- Define the to-be-proved statement
theorem minutes_sean_played_each_day : (t - i) / d = 50 :=
by
  sorry

end minutes_sean_played_each_day_l295_29582


namespace students_in_first_class_l295_29531

variable (x : ℕ)
variable (avg_marks_first_class : ℕ := 40)
variable (num_students_second_class : ℕ := 28)
variable (avg_marks_second_class : ℕ := 60)
variable (avg_marks_all : ℕ := 54)

theorem students_in_first_class : (40 * x + 60 * 28) / (x + 28) = 54 → x = 12 := 
by 
  sorry

end students_in_first_class_l295_29531


namespace Carson_returned_l295_29504

theorem Carson_returned :
  ∀ (initial_oranges ate_oranges stolen_oranges final_oranges : ℕ), 
  initial_oranges = 60 →
  ate_oranges = 10 →
  stolen_oranges = (initial_oranges - ate_oranges) / 2 →
  final_oranges = 30 →
  final_oranges = (initial_oranges - ate_oranges - stolen_oranges) + 5 :=
by 
  sorry

end Carson_returned_l295_29504


namespace math_problem_l295_29557

theorem math_problem :
  3^(5+2) + 4^(1+3) = 39196 ∧
  2^(9+2) - 3^(4+1) = 3661 ∧
  1^(8+6) + 3^(2+3) = 250 ∧
  6^(5+4) - 4^(5+1) = 409977 → 
  5^(7+2) - 2^(5+3) = 1952869 :=
by
  sorry

end math_problem_l295_29557
