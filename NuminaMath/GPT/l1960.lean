import Mathlib

namespace NUMINAMATH_GPT_volume_of_larger_cube_l1960_196037

theorem volume_of_larger_cube (s : ℝ) (V : ℝ) :
  (∀ (n : ℕ), n = 125 →
    ∀ (v_sm : ℝ), v_sm = 1 →
    V = n * v_sm →
    V = s^3 →
    s = 5 →
    ∀ (sa_large : ℝ), sa_large = 6 * s^2 →
    sa_large = 150 →
    ∀ (sa_sm_total : ℝ), sa_sm_total = n * (6 * v_sm^(2/3)) →
    sa_sm_total = 750 →
    sa_sm_total - sa_large = 600 →
    V = 125) :=
by
  intros n n125 v_sm v1 Vdef Vcube sc5 sa_large sa_large_def sa_large150 sa_sm_total sa_sm_total_def sa_sm_total750 diff600
  simp at *
  sorry

end NUMINAMATH_GPT_volume_of_larger_cube_l1960_196037


namespace NUMINAMATH_GPT_parabola_vertex_relationship_l1960_196025

theorem parabola_vertex_relationship (m x y : ℝ) :
  (y = x^2 - 2*m*x + 2*m^2 - 3*m + 1) → (y = x^2 - 3*x + 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_vertex_relationship_l1960_196025


namespace NUMINAMATH_GPT_correct_option_d_l1960_196091

-- Definitions
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f x = f (-x))
variable (hf_inc : ∀ x y : ℝ, -1 ≤ x → x ≤ 0 → -1 ≤ y → y ≤ 0 → x ≤ y → f x ≤ f y)

-- Theorem statement
theorem correct_option_d :
  f (Real.sin (Real.pi / 12)) > f (Real.tan (Real.pi / 12)) :=
sorry

end NUMINAMATH_GPT_correct_option_d_l1960_196091


namespace NUMINAMATH_GPT_problem1_problem2_l1960_196041

-- Define the main assumptions and the proof problem for Lean 4
theorem problem1 (a : ℝ) (h : a ≠ 0) : (a^2)^3 / (-a)^2 = a^4 := sorry

theorem problem2 (a b : ℝ) : (a + 2 * b) * (a + b) - 3 * a * (a + b) = -2 * a^2 + 2 * b^2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1960_196041


namespace NUMINAMATH_GPT_parameterize_line_l1960_196018

theorem parameterize_line (f : ℝ → ℝ) (t : ℝ) (x y : ℝ)
  (h1 : y = 2 * x - 30)
  (h2 : (x, y) = (f t, 20 * t - 10)) :
  f t = 10 * t + 10 :=
sorry

end NUMINAMATH_GPT_parameterize_line_l1960_196018


namespace NUMINAMATH_GPT_smallest_c_over_a_plus_b_l1960_196026

theorem smallest_c_over_a_plus_b (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  ∃ d : ℝ, d = (c / (a + b)) ∧ d = (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_smallest_c_over_a_plus_b_l1960_196026


namespace NUMINAMATH_GPT_students_in_second_class_l1960_196093

-- Definitions based on the conditions
def students_first_class : ℕ := 30
def avg_mark_first_class : ℕ := 40
def avg_mark_second_class : ℕ := 80
def combined_avg_mark : ℕ := 65

-- Proposition to prove
theorem students_in_second_class (x : ℕ) 
  (h1 : students_first_class * avg_mark_first_class + x * avg_mark_second_class = (students_first_class + x) * combined_avg_mark) : 
  x = 50 :=
sorry

end NUMINAMATH_GPT_students_in_second_class_l1960_196093


namespace NUMINAMATH_GPT_Jordan_length_is_8_l1960_196081

-- Definitions of the conditions given in the problem
def Carol_length := 5
def Carol_width := 24
def Jordan_width := 15

-- Definition to calculate the area of Carol's rectangle
def Carol_area : ℕ := Carol_length * Carol_width

-- Definition to calculate the length of Jordan's rectangle
def Jordan_length (area : ℕ) (width : ℕ) : ℕ := area / width

-- Proposition to prove the length of Jordan's rectangle
theorem Jordan_length_is_8 : Jordan_length Carol_area Jordan_width = 8 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_Jordan_length_is_8_l1960_196081


namespace NUMINAMATH_GPT_fiona_received_59_l1960_196069

theorem fiona_received_59 (Dan_riddles : ℕ) (Andy_riddles : ℕ) (Bella_riddles : ℕ) (Emma_riddles : ℕ) (Fiona_riddles : ℕ)
  (h1 : Dan_riddles = 21)
  (h2 : Andy_riddles = Dan_riddles + 12)
  (h3 : Bella_riddles = Andy_riddles - 7)
  (h4 : Emma_riddles = Bella_riddles / 2)
  (h5 : Fiona_riddles = Andy_riddles + Bella_riddles) :
  Fiona_riddles = 59 :=
by
  sorry

end NUMINAMATH_GPT_fiona_received_59_l1960_196069


namespace NUMINAMATH_GPT_potatoes_left_l1960_196061

theorem potatoes_left (initial_potatoes : ℕ) (potatoes_for_salads : ℕ) (potatoes_for_mashed : ℕ)
  (h1 : initial_potatoes = 52)
  (h2 : potatoes_for_salads = 15)
  (h3 : potatoes_for_mashed = 24) :
  initial_potatoes - (potatoes_for_salads + potatoes_for_mashed) = 13 := by
  sorry

end NUMINAMATH_GPT_potatoes_left_l1960_196061


namespace NUMINAMATH_GPT_cos_A_of_triangle_l1960_196074

theorem cos_A_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : b = Real.sqrt 2 * c)
  (h2 : Real.sin A + Real.sqrt 2 * Real.sin C = 2 * Real.sin B)
  (h3 : a = Real.sin A / Real.sin A * b) -- Sine rule used implicitly

: Real.cos A = Real.sqrt 2 / 4 := by
  -- proof will be skipped, hence 'sorry' included
  sorry

end NUMINAMATH_GPT_cos_A_of_triangle_l1960_196074


namespace NUMINAMATH_GPT_mary_money_left_l1960_196077

def drink_price (p : ℕ) : ℕ := p
def medium_pizza_price (p : ℕ) : ℕ := 2 * p
def large_pizza_price (p : ℕ) : ℕ := 3 * p
def drinks_cost (n : ℕ) (p : ℕ) : ℕ := n * drink_price p
def medium_pizzas_cost (n : ℕ) (p : ℕ) : ℕ := n * medium_pizza_price p
def large_pizza_cost (n : ℕ) (p : ℕ) : ℕ := n * large_pizza_price p
def total_cost (p : ℕ) : ℕ := drinks_cost 5 p + medium_pizzas_cost 2 p + large_pizza_cost 1 p
def money_left (initial_money : ℕ) (p : ℕ) : ℕ := initial_money - total_cost p

theorem mary_money_left (p : ℕ) : money_left 50 p = 50 - 12 * p := sorry

end NUMINAMATH_GPT_mary_money_left_l1960_196077


namespace NUMINAMATH_GPT_abcd_eq_eleven_l1960_196054

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

-- Conditions on a, b, c, d
axiom cond_a : a = Real.sqrt (4 + Real.sqrt (5 + a))
axiom cond_b : b = Real.sqrt (4 - Real.sqrt (5 + b))
axiom cond_c : c = Real.sqrt (4 + Real.sqrt (5 - c))
axiom cond_d : d = Real.sqrt (4 - Real.sqrt (5 - d))

-- Theorem to prove
theorem abcd_eq_eleven : a * b * c * d = 11 :=
by
  sorry

end NUMINAMATH_GPT_abcd_eq_eleven_l1960_196054


namespace NUMINAMATH_GPT_remainder_of_expression_l1960_196056

theorem remainder_of_expression (n : ℤ) : (10 + n^2) % 7 = (3 + n^2) % 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_expression_l1960_196056


namespace NUMINAMATH_GPT_inequality_cannot_hold_l1960_196095

theorem inequality_cannot_hold (a b : ℝ) (ha : a < b) (hb : b < 0) : a^3 ≤ b^3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_cannot_hold_l1960_196095


namespace NUMINAMATH_GPT_D_neither_sufficient_nor_necessary_for_A_l1960_196067

theorem D_neither_sufficient_nor_necessary_for_A 
  (A B C D : Prop) 
  (h1 : A → B) 
  (h2 : ¬(B → A)) 
  (h3 : B ↔ C) 
  (h4 : C → D) 
  (h5 : ¬(D → C)) 
  :
  ¬(D → A) ∧ ¬(A → D) :=
by 
  sorry

end NUMINAMATH_GPT_D_neither_sufficient_nor_necessary_for_A_l1960_196067


namespace NUMINAMATH_GPT_sum_of_digits_of_N_l1960_196090

-- The total number of coins
def total_coins : ℕ := 3081

-- Setting up the equation N^2 = 3081
def N : ℕ := 55 -- Since 55^2 is closest to 3081 and sqrt(3081) ≈ 55

-- Proving the sum of the digits of N is 10
theorem sum_of_digits_of_N : (5 + 5) = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_l1960_196090


namespace NUMINAMATH_GPT_min_sum_of_box_dimensions_l1960_196092

theorem min_sum_of_box_dimensions :
  ∃ (x y z : ℕ), x * y * z = 2541 ∧ (y = x + 3 ∨ x = y + 3) ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 38 :=
sorry

end NUMINAMATH_GPT_min_sum_of_box_dimensions_l1960_196092


namespace NUMINAMATH_GPT_pastries_sold_correctly_l1960_196029

def cupcakes : ℕ := 4
def cookies : ℕ := 29
def total_pastries : ℕ := cupcakes + cookies
def left_over : ℕ := 24
def sold_pastries : ℕ := total_pastries - left_over

theorem pastries_sold_correctly : sold_pastries = 9 :=
by sorry

end NUMINAMATH_GPT_pastries_sold_correctly_l1960_196029


namespace NUMINAMATH_GPT_value_of_composition_l1960_196089

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x - 1

theorem value_of_composition : g (f (1 + 2 * g 3)) = 12 := by
  sorry

end NUMINAMATH_GPT_value_of_composition_l1960_196089


namespace NUMINAMATH_GPT_long_furred_and_brown_dogs_l1960_196058

-- Define the total number of dogs.
def total_dogs : ℕ := 45

-- Define the number of long-furred dogs.
def long_furred_dogs : ℕ := 26

-- Define the number of brown dogs.
def brown_dogs : ℕ := 22

-- Define the number of dogs that are neither long-furred nor brown.
def neither_long_furred_nor_brown_dogs : ℕ := 8

-- Prove that the number of dogs that are both long-furred and brown is 11.
theorem long_furred_and_brown_dogs : 
  (long_furred_dogs + brown_dogs) - (total_dogs - neither_long_furred_nor_brown_dogs) = 11 :=
by
  sorry

end NUMINAMATH_GPT_long_furred_and_brown_dogs_l1960_196058


namespace NUMINAMATH_GPT_ramsey_six_vertices_monochromatic_quadrilateral_l1960_196007

theorem ramsey_six_vertices_monochromatic_quadrilateral :
  ∀ (V : Type) (E : V → V → Prop), (∀ x y : V, x ≠ y → E x y ∨ ¬ E x y) →
  ∃ (u v w x : V), u ≠ v ∧ v ≠ w ∧ w ≠ x ∧ x ≠ u ∧ (E u v = E v w ∧ E v w = E w x ∧ E w x = E x u) :=
by sorry

end NUMINAMATH_GPT_ramsey_six_vertices_monochromatic_quadrilateral_l1960_196007


namespace NUMINAMATH_GPT_kids_in_group_l1960_196088

open Nat

theorem kids_in_group (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 := by
  sorry

end NUMINAMATH_GPT_kids_in_group_l1960_196088


namespace NUMINAMATH_GPT_b_is_perfect_square_l1960_196040

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem b_is_perfect_square (a b : ℕ)
  (h_positive : 0 < a) (h_positive_b : 0 < b)
  (h_gcd_lcm_multiple : (Nat.gcd a b + Nat.lcm a b) % (a + 1) = 0)
  (h_le : b ≤ a) : is_perfect_square b :=
sorry

end NUMINAMATH_GPT_b_is_perfect_square_l1960_196040


namespace NUMINAMATH_GPT_chess_amateurs_play_with_l1960_196042

theorem chess_amateurs_play_with :
  ∃ n : ℕ, ∃ total_players : ℕ, total_players = 6 ∧
  (total_players * (total_players - 1)) / 2 = 12 ∧
  (n = total_players - 1 ∧ n = 5) :=
by
  sorry

end NUMINAMATH_GPT_chess_amateurs_play_with_l1960_196042


namespace NUMINAMATH_GPT_prove_root_property_l1960_196078

-- Define the quadratic equation and its roots
theorem prove_root_property :
  let r := -4 + Real.sqrt 226
  let s := -4 - Real.sqrt 226
  (r + 4) * (s + 4) = -226 :=
by
  -- the proof steps go here (omitted)
  sorry

end NUMINAMATH_GPT_prove_root_property_l1960_196078


namespace NUMINAMATH_GPT_factor_quadratic_expression_l1960_196019

theorem factor_quadratic_expression (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (x + 2 * y) * (5 * x - 4 * y) :=
by
  sorry

end NUMINAMATH_GPT_factor_quadratic_expression_l1960_196019


namespace NUMINAMATH_GPT_part1_part2_l1960_196087

def A (x : ℝ) : Prop := x ^ 2 - 2 * x - 8 < 0
def B (x : ℝ) : Prop := x ^ 2 + 2 * x - 3 > 0
def C (a : ℝ) (x : ℝ) : Prop := x ^ 2 - 3 * a * x + 2 * a ^ 2 < 0

theorem part1 : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 4} := 
by sorry

theorem part2 (a : ℝ) : {x : ℝ | C a x} ⊆ {x : ℝ | A x} ∩ {x : ℝ | B x} ↔ (a = 0 ∨ (1 ≤ a ∧ a ≤ 2)) := 
by sorry

end NUMINAMATH_GPT_part1_part2_l1960_196087


namespace NUMINAMATH_GPT_shirts_bought_by_peter_l1960_196048

-- Define the constants and assumptions
variables (P S x : ℕ)

-- State the conditions given in the problem
def condition1 : P = 6 :=
by sorry

def condition2 : 2 * S = 20 :=
by sorry

def condition3 : 2 * P + x * S = 62 :=
by sorry

-- State the theorem to be proven
theorem shirts_bought_by_peter : x = 5 :=
by sorry

end NUMINAMATH_GPT_shirts_bought_by_peter_l1960_196048


namespace NUMINAMATH_GPT_smallest_circle_radius_polygonal_chain_l1960_196084

theorem smallest_circle_radius_polygonal_chain (l : ℝ) (hl : l = 1) : ∃ (r : ℝ), r = 0.5 := 
sorry

end NUMINAMATH_GPT_smallest_circle_radius_polygonal_chain_l1960_196084


namespace NUMINAMATH_GPT_painting_price_after_new_discount_l1960_196023

namespace PaintingPrice

-- Define the original price and the price Sarah paid
def original_price (x : ℕ) : Prop := x / 5 = 15

-- Define the new discounted price
def new_discounted_price (y x : ℕ) : Prop := y = x * 2 / 3

-- Theorem to prove the final price considering both conditions
theorem painting_price_after_new_discount (x y : ℕ) 
  (h1 : original_price x)
  (h2 : new_discounted_price y x) : y = 50 :=
by
  sorry

end PaintingPrice

end NUMINAMATH_GPT_painting_price_after_new_discount_l1960_196023


namespace NUMINAMATH_GPT_num_even_multiples_of_four_perfect_squares_lt_5000_l1960_196016

theorem num_even_multiples_of_four_perfect_squares_lt_5000 : 
  ∃ (k : ℕ), k = 17 ∧ ∀ (n : ℕ), (0 < n ∧ 16 * n^2 < 5000) ↔ (1 ≤ n ∧ n ≤ 17) :=
by
  sorry

end NUMINAMATH_GPT_num_even_multiples_of_four_perfect_squares_lt_5000_l1960_196016


namespace NUMINAMATH_GPT_unique_solution_condition_l1960_196038

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_condition_l1960_196038


namespace NUMINAMATH_GPT_geometric_sequence_26th_term_l1960_196079

noncomputable def r : ℝ := (8 : ℝ)^(1/6)

noncomputable def a (n : ℕ) (a₁ : ℝ) (r : ℝ) : ℝ := a₁ * r^(n - 1)

theorem geometric_sequence_26th_term :
  (a 26 (a 14 10 r) r = 640) :=
by
  have h₁ : a 14 10 r = 10 := sorry
  have h₂ : r^6 = 8 := sorry
  sorry

end NUMINAMATH_GPT_geometric_sequence_26th_term_l1960_196079


namespace NUMINAMATH_GPT_total_gas_consumed_l1960_196047

def highway_consumption_rate : ℕ := 3
def city_consumption_rate : ℕ := 5

-- Distances driven each day
def day_1_highway_miles : ℕ := 200
def day_1_city_miles : ℕ := 300

def day_2_highway_miles : ℕ := 300
def day_2_city_miles : ℕ := 500

def day_3_highway_miles : ℕ := 150
def day_3_city_miles : ℕ := 350

-- Function to calculate the total consumption for a given day
def daily_consumption (highway_miles city_miles : ℕ) : ℕ :=
  (highway_miles * highway_consumption_rate) + (city_miles * city_consumption_rate)

-- Total consumption over three days
def total_consumption : ℕ :=
  (daily_consumption day_1_highway_miles day_1_city_miles) +
  (daily_consumption day_2_highway_miles day_2_city_miles) +
  (daily_consumption day_3_highway_miles day_3_city_miles)

-- Theorem stating the total consumption over the three days
theorem total_gas_consumed : total_consumption = 7700 := by
  sorry

end NUMINAMATH_GPT_total_gas_consumed_l1960_196047


namespace NUMINAMATH_GPT_possible_values_for_n_l1960_196070

theorem possible_values_for_n (n : ℕ) (h1 : ∀ a b c : ℤ, (a = n-1) ∧ (b = n) ∧ (c = n+1) → 
    (∃ f g : ℤ, f = 2*a - b ∧ g = 2*b - a)) 
    (h2 : ∃ a b c : ℤ, (a = 0 ∨ b = 0 ∨ c = 0) ∧ (a + b + c = 0)) : 
    ∃ k : ℕ, n = 3^k := 
sorry

end NUMINAMATH_GPT_possible_values_for_n_l1960_196070


namespace NUMINAMATH_GPT_perpendicular_lines_intersect_at_point_l1960_196032

theorem perpendicular_lines_intersect_at_point :
  ∀ (d k : ℝ), 
  (∀ x y, 3 * x - 4 * y = d ↔ 8 * x + k * y = d) → 
  (∃ x y, x = 2 ∧ y = -3 ∧ 3 * x - 4 * y = d ∧ 8 * x + k * y = d) → 
  d = -2 :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_intersect_at_point_l1960_196032


namespace NUMINAMATH_GPT_solve_ff_eq_x_l1960_196043

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x - 5

theorem solve_ff_eq_x (x : ℝ) :
  (f (f x) = x) ↔ 
  (x = (5 + 3 * Real.sqrt 5) / 2 ∨
   x = (5 - 3 * Real.sqrt 5) / 2 ∨
   x = (3 + Real.sqrt 41) / 2 ∨ 
   x = (3 - Real.sqrt 41) / 2) := 
by
  sorry

end NUMINAMATH_GPT_solve_ff_eq_x_l1960_196043


namespace NUMINAMATH_GPT_junk_mail_per_house_l1960_196094

theorem junk_mail_per_house (total_junk_mail : ℕ) (houses_per_block : ℕ) 
  (h1 : total_junk_mail = 14) (h2 : houses_per_block = 7) : 
  (total_junk_mail / houses_per_block) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_junk_mail_per_house_l1960_196094


namespace NUMINAMATH_GPT_monotonic_range_of_a_l1960_196009

noncomputable def f (a x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1
noncomputable def f' (a x : ℝ) : ℝ := -3*x^2 + 2*a*x - 1

theorem monotonic_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f' a x ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by 
  sorry

end NUMINAMATH_GPT_monotonic_range_of_a_l1960_196009


namespace NUMINAMATH_GPT_incorrect_statement_l1960_196036

def angles_on_x_axis := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi}
def angles_on_y_axis := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 2 + k * Real.pi}
def angles_on_axes := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi / 2}
def angles_on_y_eq_neg_x := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}

theorem incorrect_statement : ¬ (angles_on_y_eq_neg_x = {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_l1960_196036


namespace NUMINAMATH_GPT_ratio_of_areas_of_similar_triangles_l1960_196031

theorem ratio_of_areas_of_similar_triangles (m1 m2 : ℝ) (med_ratio : m1 / m2 = 1 / Real.sqrt 2) :
    let area_ratio := (m1 / m2) ^ 2
    area_ratio = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_similar_triangles_l1960_196031


namespace NUMINAMATH_GPT_fraction_of_males_l1960_196072

theorem fraction_of_males (M F : ℚ) (h1 : M + F = 1)
  (h2 : (3/4) * M + (5/6) * F = 7/9) :
  M = 2/3 :=
by sorry

end NUMINAMATH_GPT_fraction_of_males_l1960_196072


namespace NUMINAMATH_GPT_car_discount_l1960_196050

variable (P D : ℝ)

theorem car_discount (h1 : 0 < P)
                     (h2 : (P - D) * 1.45 = 1.16 * P) :
                     D = 0.2 * P := by
  sorry

end NUMINAMATH_GPT_car_discount_l1960_196050


namespace NUMINAMATH_GPT_pears_total_l1960_196062

-- Conditions
def keith_initial_pears : ℕ := 47
def keith_given_pears : ℕ := 46
def mike_initial_pears : ℕ := 12

-- Define the remaining pears
def keith_remaining_pears : ℕ := keith_initial_pears - keith_given_pears
def mike_remaining_pears : ℕ := mike_initial_pears

-- Theorem statement
theorem pears_total :
  keith_remaining_pears + mike_remaining_pears = 13 :=
by
  sorry

end NUMINAMATH_GPT_pears_total_l1960_196062


namespace NUMINAMATH_GPT_unattainable_y_l1960_196098

theorem unattainable_y (x : ℝ) (h : x ≠ -4 / 3) :
  ¬ ∃ y : ℝ, y = (2 - x) / (3 * x + 4) ∧ y = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_unattainable_y_l1960_196098


namespace NUMINAMATH_GPT_circular_film_diameter_l1960_196027

-- Definition of the problem conditions
def liquidVolume : ℝ := 576  -- volume of liquid Y in cm^3
def filmThickness : ℝ := 0.2  -- thickness of the film in cm

-- Statement of the theorem to prove the diameter of the film
theorem circular_film_diameter :
  2 * Real.sqrt (2880 / Real.pi) = 2 * Real.sqrt (liquidVolume / (filmThickness * Real.pi)) := by
  sorry

end NUMINAMATH_GPT_circular_film_diameter_l1960_196027


namespace NUMINAMATH_GPT_eval_operations_l1960_196008

def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

theorem eval_operations : star (star 6 8) (hash 3 5) = 26 := by
  sorry

end NUMINAMATH_GPT_eval_operations_l1960_196008


namespace NUMINAMATH_GPT_total_time_in_pool_is_29_minutes_l1960_196003

noncomputable def calculate_total_time_in_pool : ℝ :=
  let jerry := 3             -- Jerry's time in minutes
  let elaine := 2 * jerry    -- Elaine's time in minutes
  let george := elaine / 3    -- George's time in minutes
  let susan := 150 / 60      -- Susan's time in minutes
  let puddy := elaine / 2    -- Puddy's time in minutes
  let frank := elaine / 2    -- Frank's time in minutes
  let estelle := 0.1 * 60    -- Estelle's time in minutes
  let total_excluding_newman := jerry + elaine + george + susan + puddy + frank + estelle
  let newman := total_excluding_newman / 7   -- Newman's average time
  total_excluding_newman + newman

theorem total_time_in_pool_is_29_minutes : 
  calculate_total_time_in_pool = 29 :=
by
  sorry

end NUMINAMATH_GPT_total_time_in_pool_is_29_minutes_l1960_196003


namespace NUMINAMATH_GPT_number_of_incorrect_statements_l1960_196063

-- Conditions
def cond1 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)

def cond2 (x : ℝ) : Prop := x > 5 → x^2 - 4*x - 5 > 0

def cond3 : Prop := ∃ x0 : ℝ, x0^2 + x0 - 1 < 0

def cond3_neg : Prop := ∀ x : ℝ, x^2 + x - 1 ≥ 0

def cond4 (x : ℝ) : Prop := (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)

-- Proof problem
theorem number_of_incorrect_statements : 
  (¬ cond1 (p := true) (q := false)) ∧ (cond2 (x := 6)) ∧ (cond3 → cond3_neg) ∧ (¬ cond4 (x := 0)) → 
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_incorrect_statements_l1960_196063


namespace NUMINAMATH_GPT_moon_speed_conversion_l1960_196046

theorem moon_speed_conversion :
  ∀ (moon_speed_kps : ℝ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ),
  moon_speed_kps = 0.9 →
  seconds_in_minute = 60 →
  minutes_in_hour = 60 →
  (moon_speed_kps * (seconds_in_minute * minutes_in_hour) = 3240) := by
  sorry

end NUMINAMATH_GPT_moon_speed_conversion_l1960_196046


namespace NUMINAMATH_GPT_point_in_second_or_third_quadrant_l1960_196052

theorem point_in_second_or_third_quadrant (k b : ℝ) (h₁ : k < 0) (h₂ : b ≠ 0) : 
  (k < 0 ∧ b > 0) ∨ (k < 0 ∧ b < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_or_third_quadrant_l1960_196052


namespace NUMINAMATH_GPT_power_div_ex_l1960_196039

theorem power_div_ex (a b c : ℕ) (h1 : a = 2^4) (h2 : b = 2^3) (h3 : c = 2^2) :
  ((a^4) * (b^6)) / (c^12) = 1024 := 
sorry

end NUMINAMATH_GPT_power_div_ex_l1960_196039


namespace NUMINAMATH_GPT_system_solution_l1960_196020

theorem system_solution (a x0 : ℝ) (h : a ≠ 0) 
  (h1 : 3 * x0 + 2 * x0 = 15 * a) 
  (h2 : 1 / a * x0 + x0 = 9) 
  : x0 = 6 ∧ a = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_system_solution_l1960_196020


namespace NUMINAMATH_GPT_Carol_width_eq_24_l1960_196030

-- Given conditions
def Carol_length : ℕ := 5
def Jordan_length : ℕ := 2
def Jordan_width : ℕ := 60

-- Required proof: Carol's width is 24 considering equal areas of both rectangles
theorem Carol_width_eq_24 (w : ℕ) (h : Carol_length * w = Jordan_length * Jordan_width) : w = 24 := 
by sorry

end NUMINAMATH_GPT_Carol_width_eq_24_l1960_196030


namespace NUMINAMATH_GPT_intersection_is_correct_l1960_196064

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem intersection_is_correct : A ∩ B = {-1, 2} := 
by 
  -- proof goes here 
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l1960_196064


namespace NUMINAMATH_GPT_total_journey_length_l1960_196051

theorem total_journey_length (y : ℚ)
  (h1 : y * 1 / 4 + 30 + y * 1 / 7 = y) : 
  y = 840 / 17 :=
by 
  sorry

end NUMINAMATH_GPT_total_journey_length_l1960_196051


namespace NUMINAMATH_GPT_f_is_zero_l1960_196065

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_is_zero 
  (H1 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a)
  (H2 : ∀ x : ℝ, |f x| ≤ 1) : ∀ x : ℝ, f x = 0 := 
sorry

end NUMINAMATH_GPT_f_is_zero_l1960_196065


namespace NUMINAMATH_GPT_find_AG_l1960_196012

theorem find_AG (AE CE BD CD AB AG : ℝ) (h1 : AE = 3)
    (h2 : CE = 1) (h3 : BD = 2) (h4 : CD = 2) (h5 : AB = 5) :
    AG = (3 * Real.sqrt 66) / 7 :=
  sorry

end NUMINAMATH_GPT_find_AG_l1960_196012


namespace NUMINAMATH_GPT_max_value_f_l1960_196017

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + x + 1)

theorem max_value_f : ∀ x : ℝ, f x ≤ 4 / 3 :=
sorry

end NUMINAMATH_GPT_max_value_f_l1960_196017


namespace NUMINAMATH_GPT_starling_nests_flying_condition_l1960_196085

theorem starling_nests_flying_condition (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → A ≠ B)
  (h3 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → 
  (∃ d1 d2 : ℝ, d1 < d2 ∧ d1 < d2 → d1 > d2)) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_starling_nests_flying_condition_l1960_196085


namespace NUMINAMATH_GPT_find_m_l1960_196010

theorem find_m (m : ℝ) : (Real.tan (20 * Real.pi / 180) + m * Real.sin (20 * Real.pi / 180) = Real.sqrt 3) → m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1960_196010


namespace NUMINAMATH_GPT_jimmy_points_lost_for_bad_behavior_l1960_196045

theorem jimmy_points_lost_for_bad_behavior (points_per_exam : ℕ) (num_exams : ℕ) (points_needed : ℕ)
  (extra_points_allowed : ℕ) (total_points_earned : ℕ) (current_points : ℕ)
  (h1 : points_per_exam = 20) (h2 : num_exams = 3) (h3 : points_needed = 50)
  (h4 : extra_points_allowed = 5) (h5 : total_points_earned = points_per_exam * num_exams)
  (h6 : current_points = points_needed + extra_points_allowed) :
  total_points_earned - current_points = 5 :=
by
  sorry

end NUMINAMATH_GPT_jimmy_points_lost_for_bad_behavior_l1960_196045


namespace NUMINAMATH_GPT_probability_one_solve_l1960_196080

variables {p1 p2 : ℝ}

theorem probability_one_solve (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) := 
sorry

end NUMINAMATH_GPT_probability_one_solve_l1960_196080


namespace NUMINAMATH_GPT_original_population_l1960_196004

theorem original_population (p: ℝ) :
  (p + 1500) * 0.85 = p - 45 -> p = 8800 :=
by
  sorry

end NUMINAMATH_GPT_original_population_l1960_196004


namespace NUMINAMATH_GPT_first_digit_power_l1960_196006

theorem first_digit_power (n : ℕ) (h : ∃ k : ℕ, 7 * 10^k ≤ 2^n ∧ 2^n < 8 * 10^k) :
  (∃ k' : ℕ, 1 * 10^k' ≤ 5^n ∧ 5^n < 2 * 10^k') :=
sorry

end NUMINAMATH_GPT_first_digit_power_l1960_196006


namespace NUMINAMATH_GPT_days_until_see_grandma_l1960_196066

def hours_in_a_day : ℕ := 24
def hours_until_see_grandma : ℕ := 48

theorem days_until_see_grandma : hours_until_see_grandma / hours_in_a_day = 2 := by
  sorry

end NUMINAMATH_GPT_days_until_see_grandma_l1960_196066


namespace NUMINAMATH_GPT_expected_yield_of_carrots_l1960_196015

def steps_to_feet (steps : ℕ) (step_size : ℕ) : ℕ :=
  steps * step_size

def garden_area (length width : ℕ) : ℕ :=
  length * width

def yield_of_carrots (area : ℕ) (yield_rate : ℚ) : ℚ :=
  area * yield_rate

theorem expected_yield_of_carrots :
  steps_to_feet 18 3 * steps_to_feet 25 3 = 4050 →
  yield_of_carrots 4050 (3 / 4) = 3037.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_yield_of_carrots_l1960_196015


namespace NUMINAMATH_GPT_geometric_sequence_a6_l1960_196013

theorem geometric_sequence_a6
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : a 1 = 1)
  (S3 : S 3 = 7 / 4)
  (sum_S3 : S 3 = a 1 + a 1 * a 2 + a 1 * (a 2)^2) :
  a 6 = 1 / 32 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l1960_196013


namespace NUMINAMATH_GPT_total_animals_correct_l1960_196014

def L := 10

def C := 2 * L + 4

def Merry_lambs := L
def Merry_cows := C
def Merry_pigs (P : ℕ) := P
def Brother_lambs := L + 3

def Brother_chickens (R : ℕ) := R * Brother_lambs
def Brother_goats (Q : ℕ) := 2 * Brother_lambs + Q

def Merry_total (P : ℕ) := Merry_lambs + Merry_cows + Merry_pigs P
def Brother_total (R Q : ℕ) := Brother_lambs + Brother_chickens R + Brother_goats Q

def Total_animals (P R Q : ℕ) := Merry_total P + Brother_total R Q

theorem total_animals_correct (P R Q : ℕ) : 
  Total_animals P R Q = 73 + P + R * 13 + Q := by
  sorry

end NUMINAMATH_GPT_total_animals_correct_l1960_196014


namespace NUMINAMATH_GPT_overlap_area_of_parallelogram_l1960_196073

theorem overlap_area_of_parallelogram (w1 w2 : ℝ) (β : ℝ) (hβ : β = 30) (hw1 : w1 = 2) (hw2 : w2 = 1) : 
  (w1 * (w2 / Real.sin (β * Real.pi / 180))) = 4 :=
by
  sorry

end NUMINAMATH_GPT_overlap_area_of_parallelogram_l1960_196073


namespace NUMINAMATH_GPT_empty_one_container_l1960_196005

theorem empty_one_container (a b c : ℕ) :
  ∃ a' b' c', (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
    (a' = a ∧ b' = b ∧ c' = c ∨
     (a' ≤ a ∧ b' ≤ b ∧ c' ≤ c ∧ (a + b + c = a' + b' + c')) ∧
     (∀ i j, i ≠ j → (i = 1 ∨ i = 2 ∨ i = 3) →
              (j = 1 ∨ j = 2 ∨ j = 3) →
              (if i = 1 then (if j = 2 then a' = a - a ∨ a' = a else (if j = 3 then a' = a - a ∨ a' = a else false))
               else if i = 2 then (if j = 1 then b' = b - b ∨ b' = b else (if j = 3 then b' = b - b ∨ b' = b else false))
               else (if j = 1 then c' = c - c ∨ c' = c else (if j = 2 then c' = c - c ∨ c' = c else false))))) :=
by
  sorry

end NUMINAMATH_GPT_empty_one_container_l1960_196005


namespace NUMINAMATH_GPT_ordered_pair_exists_l1960_196001

theorem ordered_pair_exists :
  ∃ p q : ℝ, 
  (3 + 8 * p = 2 - 3 * q) ∧ (-4 - 6 * p = -3 + 4 * q) ∧ (p = -1/14) ∧ (q = -1/7) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_exists_l1960_196001


namespace NUMINAMATH_GPT_members_even_and_divisible_l1960_196060

structure ClubMember (α : Type) := 
  (friend : α) 
  (enemy : α)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

def can_be_divided_into_two_subclubs (members : List (ClubMember Nat)) : Prop :=
sorry -- Definition of dividing into two subclubs here

theorem members_even_and_divisible (members : List (ClubMember Nat)) :
  is_even members.length ∧ can_be_divided_into_two_subclubs members :=
sorry

end NUMINAMATH_GPT_members_even_and_divisible_l1960_196060


namespace NUMINAMATH_GPT_short_haired_girls_l1960_196096

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end NUMINAMATH_GPT_short_haired_girls_l1960_196096


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1960_196055

theorem geometric_sequence_common_ratio (a : ℕ → ℤ) (q : ℤ)  
  (h1 : a 1 = 3) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h3 : 4 * a 1 + a 3 = 4 * a 2) : 
  q = 2 := 
by {
  -- Proof is omitted here
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1960_196055


namespace NUMINAMATH_GPT_rectangle_semicircle_problem_l1960_196035

/--
Rectangle ABCD and a semicircle with diameter AB are coplanar and have nonoverlapping interiors.
Let R denote the region enclosed by the semicircle and the rectangle.
Line ℓ meets the semicircle, segment AB, and segment CD at distinct points P, V, and S, respectively.
Line ℓ divides region R into two regions with areas in the ratio 3:1.
Suppose that AV = 120, AP = 180, and VB = 240.
Prove the length of DA = 90 * sqrt(6).
-/
theorem rectangle_semicircle_problem (DA : ℝ) (AV AP VB : ℝ) (h₁ : AV = 120) (h₂ : AP = 180) (h₃ : VB = 240) :
  DA = 90 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_rectangle_semicircle_problem_l1960_196035


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_sum_l1960_196075

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)

-- Definitions based on given problem conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) := 
  ∀ n, S n = n * (a 1 + a n) / 2

axiom Sn_2017 : S_n 2017 = 4034

-- Goal: a_3 + a_1009 + a_2015 = 6
theorem arithmetic_sequence_terms_sum :
  arithmetic_sequence a_n →
  sum_first_n_terms S_n a_n →
  S_n 2017 = 4034 → 
  a_n 3 + a_n 1009 + a_n 2015 = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_sum_l1960_196075


namespace NUMINAMATH_GPT_negation_of_positive_x2_plus_2_l1960_196082

theorem negation_of_positive_x2_plus_2 (h : ∀ x : ℝ, x^2 + 2 > 0) : ¬ (∀ x : ℝ, x^2 + 2 > 0) = False := 
by
  sorry

end NUMINAMATH_GPT_negation_of_positive_x2_plus_2_l1960_196082


namespace NUMINAMATH_GPT_dimes_total_l1960_196053

def initial_dimes : ℕ := 9
def added_dimes : ℕ := 7

theorem dimes_total : initial_dimes + added_dimes = 16 := by
  sorry

end NUMINAMATH_GPT_dimes_total_l1960_196053


namespace NUMINAMATH_GPT_angles_bisectors_l1960_196071

theorem angles_bisectors (k : ℤ) : 
    ∃ α : ℤ, α = k * 180 + 135 
  -> 
    (α = (2 * k) * 180 + 135 ∨ α = (2 * k + 1) * 180 + 135) 
  := sorry

end NUMINAMATH_GPT_angles_bisectors_l1960_196071


namespace NUMINAMATH_GPT_number_of_green_fish_l1960_196033

theorem number_of_green_fish (total_fish : ℕ) (blue_fish : ℕ) (orange_fish : ℕ) (green_fish : ℕ)
  (h1 : total_fish = 80)
  (h2 : blue_fish = total_fish / 2)
  (h3 : orange_fish = blue_fish - 15)
  (h4 : green_fish = total_fish - blue_fish - orange_fish)
  : green_fish = 15 :=
by sorry

end NUMINAMATH_GPT_number_of_green_fish_l1960_196033


namespace NUMINAMATH_GPT_average_student_headcount_l1960_196034

variable (headcount_02_03 headcount_03_04 headcount_04_05 headcount_05_06 : ℕ)
variable {h_02_03 : headcount_02_03 = 10900}
variable {h_03_04 : headcount_03_04 = 10500}
variable {h_04_05 : headcount_04_05 = 10700}
variable {h_05_06 : headcount_05_06 = 11300}

theorem average_student_headcount : 
  (headcount_02_03 + headcount_03_04 + headcount_04_05 + headcount_05_06) / 4 = 10850 := 
by 
  sorry

end NUMINAMATH_GPT_average_student_headcount_l1960_196034


namespace NUMINAMATH_GPT_randy_initial_amount_l1960_196099

theorem randy_initial_amount (spend_per_trip: ℤ) (trips_per_month: ℤ) (dollars_left_after_year: ℤ) (total_month_months: ℤ := 12):
  (spend_per_trip = 2 ∧ trips_per_month = 4 ∧ dollars_left_after_year = 104) → spend_per_trip * trips_per_month * total_month_months + dollars_left_after_year = 200 := 
by
  sorry

end NUMINAMATH_GPT_randy_initial_amount_l1960_196099


namespace NUMINAMATH_GPT_sarah_dimes_l1960_196044

theorem sarah_dimes (d n : ℕ) (h1 : d + n = 50) (h2 : 10 * d + 5 * n = 200) : d = 10 :=
sorry

end NUMINAMATH_GPT_sarah_dimes_l1960_196044


namespace NUMINAMATH_GPT_student_score_is_64_l1960_196021

-- Define the total number of questions and correct responses.
def total_questions : ℕ := 100
def correct_responses : ℕ := 88

-- Function to calculate the score based on the grading rule.
def calculate_score (total : ℕ) (correct : ℕ) : ℕ :=
  correct - 2 * (total - correct)

-- The theorem that states the score for the given conditions.
theorem student_score_is_64 :
  calculate_score total_questions correct_responses = 64 :=
by
  sorry

end NUMINAMATH_GPT_student_score_is_64_l1960_196021


namespace NUMINAMATH_GPT_mandy_difference_of_cinnamon_and_nutmeg_l1960_196083

theorem mandy_difference_of_cinnamon_and_nutmeg :
  let cinnamon := 0.6666666666666666
  let nutmeg := 0.5
  let difference := cinnamon - nutmeg
  difference = 0.1666666666666666 :=
by
  sorry

end NUMINAMATH_GPT_mandy_difference_of_cinnamon_and_nutmeg_l1960_196083


namespace NUMINAMATH_GPT_symmetric_about_one_symmetric_about_two_l1960_196000

-- Part 1
theorem symmetric_about_one (rational_num_x : ℚ) (rational_num_r : ℚ) 
(h1 : 3 - 1 = 1 - rational_num_x) (hr1 : r = 3 - 1): 
  rational_num_x = -1 ∧ rational_num_r = 2 := 
by
  sorry

-- Part 2
theorem symmetric_about_two (a b : ℚ) (symmetric_radius : ℚ) 
(h2 : (a + b) / 2 = 2) (condition : |a| = 2 * |b|) : 
  symmetric_radius = 2 / 3 ∨ symmetric_radius = 6 := 
by
  sorry

end NUMINAMATH_GPT_symmetric_about_one_symmetric_about_two_l1960_196000


namespace NUMINAMATH_GPT_integer_difference_divisible_by_n_l1960_196059

theorem integer_difference_divisible_by_n (n : ℕ) (h : n > 0) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_integer_difference_divisible_by_n_l1960_196059


namespace NUMINAMATH_GPT_inequality_imply_positive_a_l1960_196022

theorem inequality_imply_positive_a 
  (a b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h_d_pos : d > 0) 
  (h : a / b > -3 / (2 * d)) : a > 0 :=
sorry

end NUMINAMATH_GPT_inequality_imply_positive_a_l1960_196022


namespace NUMINAMATH_GPT_tan_4x_eq_cos_x_has_9_solutions_l1960_196011

theorem tan_4x_eq_cos_x_has_9_solutions :
  ∃ (s : Finset ℝ), s.card = 9 ∧ ∀ x ∈ s, (0 ≤ x ∧ x ≤ 2 * Real.pi) ∧ (Real.tan (4 * x) = Real.cos x) :=
sorry

end NUMINAMATH_GPT_tan_4x_eq_cos_x_has_9_solutions_l1960_196011


namespace NUMINAMATH_GPT_john_order_cost_l1960_196068

-- Definitions from the problem conditions
def discount_rate : ℝ := 0.10
def item_price : ℝ := 200
def num_items : ℕ := 7
def discount_threshold : ℝ := 1000

-- Final proof statement
theorem john_order_cost : 
  (num_items * item_price) - 
  (if (num_items * item_price) > discount_threshold then 
    discount_rate * ((num_items * item_price) - discount_threshold) 
  else 0) = 1360 := 
sorry

end NUMINAMATH_GPT_john_order_cost_l1960_196068


namespace NUMINAMATH_GPT_math_proof_problem_l1960_196028

variable {a b c : ℝ}

theorem math_proof_problem (h₁ : a * b * c * (a + b) * (b + c) * (c + a) ≠ 0)
  (h₂ : (a + b + c) * (1 / a + 1 / b + 1 / c) = 1007 / 1008) :
  (a * b / ((a + c) * (b + c)) + b * c / ((b + a) * (c + a)) + c * a / ((c + b) * (a + b))) = 2017 := 
sorry

end NUMINAMATH_GPT_math_proof_problem_l1960_196028


namespace NUMINAMATH_GPT_greatest_divisor_condition_gcd_of_numbers_l1960_196024

theorem greatest_divisor_condition (n : ℕ) (h100 : n ∣ 100) (h225 : n ∣ 225) (h150 : n ∣ 150) : n ≤ 25 :=
  sorry

theorem gcd_of_numbers : Nat.gcd (Nat.gcd 100 225) 150 = 25 :=
  sorry

end NUMINAMATH_GPT_greatest_divisor_condition_gcd_of_numbers_l1960_196024


namespace NUMINAMATH_GPT_sum_of_fractions_eq_two_l1960_196002

theorem sum_of_fractions_eq_two : 
  (1 / 2) + (2 / 4) + (4 / 8) + (8 / 16) = 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_fractions_eq_two_l1960_196002


namespace NUMINAMATH_GPT_max_hedgehogs_l1960_196086

theorem max_hedgehogs (S : ℕ) (n : ℕ) (hS : S = 65) (hn : ∀ m, m > n → (m * (m + 1)) / 2 > S) :
  n = 10 := 
sorry

end NUMINAMATH_GPT_max_hedgehogs_l1960_196086


namespace NUMINAMATH_GPT_divides_lcm_condition_l1960_196057

theorem divides_lcm_condition (x y : ℕ) (h₀ : 1 < x) (h₁ : 1 < y)
  (h₂ : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x := 
sorry

end NUMINAMATH_GPT_divides_lcm_condition_l1960_196057


namespace NUMINAMATH_GPT_coord_sum_D_l1960_196049

def is_midpoint (M C D : ℝ × ℝ) := M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem coord_sum_D (M C D : ℝ × ℝ) (h : is_midpoint M C D) (hM : M = (4, 6)) (hC : C = (10, 2)) :
  D.1 + D.2 = 8 :=
sorry

end NUMINAMATH_GPT_coord_sum_D_l1960_196049


namespace NUMINAMATH_GPT_possible_values_for_p_l1960_196076

-- Definitions for the conditions
variables {a b c p : ℝ}

-- Assumptions
def distinct (a b c : ℝ) := ¬(a = b) ∧ ¬(b = c) ∧ ¬(c = a)
def main_eq (a b c p : ℝ) := a + (1 / b) = p ∧ b + (1 / c) = p ∧ c + (1 / a) = p

-- Theorem statement
theorem possible_values_for_p (h1 : distinct a b c) (h2 : main_eq a b c p) : p = 1 ∨ p = -1 := 
sorry

end NUMINAMATH_GPT_possible_values_for_p_l1960_196076


namespace NUMINAMATH_GPT_average_ABC_is_3_l1960_196097

theorem average_ABC_is_3
  (A B C : ℝ)
  (h1 : 2003 * C - 4004 * A = 8008)
  (h2 : 2003 * B + 6006 * A = 10010)
  (h3 : B = 2 * A - 6) : 
  (A + B + C) / 3 = 3 :=
sorry

end NUMINAMATH_GPT_average_ABC_is_3_l1960_196097
