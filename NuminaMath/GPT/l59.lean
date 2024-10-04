import Mathlib

namespace sum_of_distinct_roots_l59_59537

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l59_59537


namespace smallest_distance_l59_59571

open Real

-- Definition for the condition: six points in or on a square of side length 2
def points_in_square (p : Fin 6 → ℝ × ℝ) : Prop :=
  ∀ i, abs (p i).1 ≤ 1 ∧ abs (p i).2 ≤ 1

-- The theorem statement to be proved
theorem smallest_distance (p : Fin 6 → ℝ × ℝ) (h : points_in_square p): ∃ (b : ℝ), 
  b = sqrt 2 ∧ ∃ (i j : Fin 6), i ≠ j ∧ dist (p i) (p j) ≤ b :=
by
  sorry

end smallest_distance_l59_59571


namespace Diane_bakes_160_gingerbreads_l59_59954

-- Definitions
def trays1Count : Nat := 4
def gingerbreads1PerTray : Nat := 25
def trays2Count : Nat := 3
def gingerbreads2PerTray : Nat := 20

def totalGingerbreads : Nat :=
  (trays1Count * gingerbreads1PerTray) + (trays2Count * gingerbreads2PerTray)

-- Problem statement
theorem Diane_bakes_160_gingerbreads :
  totalGingerbreads = 160 := by
  sorry

end Diane_bakes_160_gingerbreads_l59_59954


namespace electronic_items_stock_l59_59992

-- Define the base statements
def all_in_stock (S : Type) (p : S → Prop) : Prop := ∀ x, p x
def some_not_in_stock (S : Type) (p : S → Prop) : Prop := ∃ x, ¬ p x

-- Define the main theorem statement
theorem electronic_items_stock (S : Type) (p : S → Prop) :
  ¬ all_in_stock S p → some_not_in_stock S p :=
by
  intros
  sorry

end electronic_items_stock_l59_59992


namespace eggs_in_each_group_l59_59382

theorem eggs_in_each_group (eggs marbles groups : ℕ) 
  (h_eggs : eggs = 15)
  (h_groups : groups = 3) 
  (h_marbles : marbles = 4) :
  eggs / groups = 5 :=
by sorry

end eggs_in_each_group_l59_59382


namespace how_many_bigger_panda_bears_l59_59076

-- Definitions for the conditions
def four_small_panda_bears_eat_daily : ℕ := 25
def one_small_panda_bear_eats_daily : ℚ := 25 / 4
def each_bigger_panda_bear_eats_daily : ℚ := 40
def total_bamboo_eaten_weekly : ℕ := 2100
def total_bamboo_eaten_daily : ℚ := 2100 / 7

-- The theorem statement to prove
theorem how_many_bigger_panda_bears :
  ∃ B : ℚ, one_small_panda_bear_eats_daily * 4 + each_bigger_panda_bear_eats_daily * B = total_bamboo_eaten_daily := by
  sorry

end how_many_bigger_panda_bears_l59_59076


namespace fraction_of_gasoline_used_l59_59177

-- Define the conditions
def gasoline_per_mile := 1 / 30  -- Gallons per mile
def full_tank := 12  -- Gallons
def speed := 60  -- Miles per hour
def travel_time := 5  -- Hours

-- Total distance traveled
def distance := speed * travel_time  -- Miles

-- Gasoline used
def gasoline_used := distance * gasoline_per_mile  -- Gallons

-- Fraction of the full tank used
def fraction_used := gasoline_used / full_tank

-- The theorem to be proved
theorem fraction_of_gasoline_used :
  fraction_used = 5 / 6 :=
by sorry

end fraction_of_gasoline_used_l59_59177


namespace solve_part_one_solve_part_two_l59_59487

-- Define function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Prove for part (1)
theorem solve_part_one : 
  {x : ℝ | -1 / 3 ≤ x ∧ x ≤ 5} = {x : ℝ | f 2 x ≤ 1} :=
by
  -- Replace the proof with sorry
  sorry

-- Prove for part (2)
theorem solve_part_two :
  {a : ℝ | a = 1 ∨ a = -1} = {a : ℝ | ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4} :=
by
  -- Replace the proof with sorry
  sorry

end solve_part_one_solve_part_two_l59_59487


namespace lcm_924_660_eq_4620_l59_59742

theorem lcm_924_660_eq_4620 : Nat.lcm 924 660 = 4620 := 
by
  sorry

end lcm_924_660_eq_4620_l59_59742


namespace remainder_of_N_mod_D_l59_59069

/-- The given number N and the divisor 252 defined in terms of its prime factors. -/
def N : ℕ := 9876543210123456789
def D : ℕ := 252

/-- The remainders of N modulo 4, 9, and 7 as given in the solution -/
def N_mod_4 : ℕ := 1
def N_mod_9 : ℕ := 0
def N_mod_7 : ℕ := 6

theorem remainder_of_N_mod_D :
  N % D = 27 :=
by
  sorry

end remainder_of_N_mod_D_l59_59069


namespace intersection_A_B_l59_59501

def A := { x : ℝ | x / (x - 1) ≥ 0 }
def B := { y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B : A ∩ B = { y : ℝ | y > 1 } :=
by sorry

end intersection_A_B_l59_59501


namespace simplify_expression_l59_59568

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l59_59568


namespace Sperner_theorem_example_l59_59743

theorem Sperner_theorem_example :
  ∀ (S : Finset (Finset ℕ)), (S.card = 10) →
  (∀ (A B : Finset ℕ), A ∈ S → B ∈ S → A ⊆ B → A = B) → S.card = 252 :=
by sorry

end Sperner_theorem_example_l59_59743


namespace tangent_line_circle_l59_59506

theorem tangent_line_circle (m : ℝ) :
  (∀ x y : ℝ, x - 2*y + m = 0 ↔ (x^2 + y^2 - 4*x + 6*y + 8 = 0)) →
  m = -3 ∨ m = -13 :=
sorry

end tangent_line_circle_l59_59506


namespace resistor_value_l59_59738

-- Definitions based on given conditions
def U : ℝ := 9 -- Volt reading by the voltmeter
def I : ℝ := 2 -- Current reading by the ammeter
def U_total : ℝ := 2 * U -- Total voltage in the series circuit

-- Stating the theorem
theorem resistor_value (R₀ : ℝ) :
  (U_total = I * (2 * R₀)) → R₀ = 9 :=
by
  intro h
  sorry

end resistor_value_l59_59738


namespace mike_bricks_l59_59262

theorem mike_bricks (total_bricks bricks_A bricks_B bricks_other: ℕ) 
  (h1 : bricks_A = 40) 
  (h2 : bricks_B = bricks_A / 2)
  (h3 : total_bricks = 150) 
  (h4 : total_bricks = bricks_A + bricks_B + bricks_other) : bricks_other = 90 := 
by 
  sorry

end mike_bricks_l59_59262


namespace total_spent_l59_59734

-- Constants representing the conditions from the problem
def cost_per_deck : ℕ := 8
def tom_decks : ℕ := 3
def friend_decks : ℕ := 5

-- Theorem stating the total amount spent by Tom and his friend
theorem total_spent : tom_decks * cost_per_deck + friend_decks * cost_per_deck = 64 := by
  sorry

end total_spent_l59_59734


namespace remainder_when_13_plus_y_divided_by_31_l59_59553

theorem remainder_when_13_plus_y_divided_by_31
  (y : ℕ)
  (hy : 7 * y % 31 = 1) :
  (13 + y) % 31 = 22 :=
sorry

end remainder_when_13_plus_y_divided_by_31_l59_59553


namespace draw_ways_l59_59874

open Nat

theorem draw_ways (whiteBalls blackBalls : ℕ) (draws : ℕ) (ways : ℕ) :
  whiteBalls = 6 → blackBalls = 4 → draws = 4 → ways = 
  (choose blackBalls 2) * (choose whiteBalls 2) + 
  (choose blackBalls 3) * (choose whiteBalls 1) + 
  (choose blackBalls 4) → ways = 115 :=
by {
  intros hwhite hblack hdraws hways,
  rw [hwhite, hblack, hdraws, hways],
  sorry,
}

end draw_ways_l59_59874


namespace positive_diff_40_x_l59_59934

theorem positive_diff_40_x
  (x : ℝ)
  (h : (40 + x + 15) / 3 = 35) :
  abs (x - 40) = 10 :=
sorry

end positive_diff_40_x_l59_59934


namespace math_problem_l59_59582

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem math_problem : ((otimes (otimes 2 4) 6) - (otimes 2 (otimes 4 6))) = -23327 / 288 := by sorry

end math_problem_l59_59582


namespace count_integer_b_for_log_b_256_l59_59093

theorem count_integer_b_for_log_b_256 :
  (∃ b : ℕ, b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) ∧ 
  (∀ b : ℕ, (b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) → (b = 2 ∨ b = 4 ∨ b = 16 ∨ b = 256)) :=
by sorry

end count_integer_b_for_log_b_256_l59_59093


namespace cyclists_meet_at_start_l59_59420

theorem cyclists_meet_at_start (T : ℚ) (h1 : T = 5 * 7 * 9 / gcd (5 * 7) (gcd (7 * 9) (9 * 5))) : T = 157.5 :=
by
  sorry

end cyclists_meet_at_start_l59_59420


namespace dense_sets_count_l59_59169

open Finset

def is_dense (A : Finset ℕ) : Prop :=
  (A ⊆ range 50) ∧ (card A > 40) ∧
  (∀ x ∈ range 45, range' x 6 \subseteq A → false)

theorem dense_sets_count :
  {A : Finset ℕ | is_dense A}.card = 495 :=
sorry

end dense_sets_count_l59_59169


namespace probability_of_two_evens_in_five_l59_59624

-- Define the initial set of numbers
def num_set : set ℕ := {1, 2, 3, 4, 5}

-- Define a condition for even numbers in the set
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the count of ways to select two elements from a given set
def combination (n r : ℕ) : ℕ := n.choose r

-- The probability calculation function (stub)
noncomputable def probability_even_in_two_selected (set : set ℕ) : ℚ :=
let total_ways := combination (set.to_finset.card) 2 in
let even_ways := combination (set.filter is_even).to_finset.card 2 in
(even_ways : ℚ) / total_ways

-- The theorem statement to prove
theorem probability_of_two_evens_in_five :
  probability_even_in_two_selected num_set = 1 / 10 :=
by
  sorry

end probability_of_two_evens_in_five_l59_59624


namespace sum_distinct_vars_eq_1716_l59_59542

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l59_59542


namespace taimour_time_to_paint_alone_l59_59031

theorem taimour_time_to_paint_alone (T : ℝ) (h1 : Jamshid_time = T / 2)
  (h2 : (1 / T + 1 / (T / 2)) = 1 / 3) : T = 9 :=
sorry

end taimour_time_to_paint_alone_l59_59031


namespace octagon_has_20_diagonals_l59_59640

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l59_59640


namespace chloe_points_first_round_l59_59617

theorem chloe_points_first_round 
  (P : ℕ)
  (second_round_points : ℕ := 50)
  (lost_points : ℕ := 4)
  (total_points : ℕ := 86)
  (h : P + second_round_points - lost_points = total_points) : 
  P = 40 := 
by 
  sorry

end chloe_points_first_round_l59_59617


namespace triangle_inequality_example_l59_59492

theorem triangle_inequality_example {x : ℝ} (h1: 3 + 4 > x) (h2: abs (3 - 4) < x) : 1 < x ∧ x < 7 :=
  sorry

end triangle_inequality_example_l59_59492


namespace minimum_value_of_expression_l59_59986

theorem minimum_value_of_expression {x y : ℝ} (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ m : ℝ, m = 0.75 ∧ ∀ z : ℝ, (∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y = 1 ∧ z = 2 * x + 3 * y ^ 2) → z ≥ m :=
sorry

end minimum_value_of_expression_l59_59986


namespace remainder_13_plus_y_l59_59551

theorem remainder_13_plus_y :
  (∃ y : ℕ, (0 < y) ∧ (7 * y ≡ 1 [MOD 31])) → (∃ y : ℕ, (13 + y ≡ 22 [MOD 31])) :=
by 
  sorry

end remainder_13_plus_y_l59_59551


namespace combination_exists_l59_59351

theorem combination_exists 
  (S T Ti : ℝ) (x y z : ℝ)
  (h : 3 * S + 4 * T + 2 * Ti = 40) :
  ∃ x y z : ℝ, x * S + y * T + z * Ti = 60 :=
sorry

end combination_exists_l59_59351


namespace sum_of_distinct_roots_l59_59536

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l59_59536


namespace arcsin_one_eq_pi_div_two_l59_59940

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := by
  sorry

end arcsin_one_eq_pi_div_two_l59_59940


namespace intersection_and_perpendicular_line_l59_59200

theorem intersection_and_perpendicular_line :
  ∃ (x y : ℝ), (3 * x + y - 1 = 0) ∧ (x + 2 * y - 7 = 0) ∧ (2 * x - y + 6 = 0) :=
by
  sorry

end intersection_and_perpendicular_line_l59_59200


namespace sum_of_radii_of_tangent_circles_l59_59440

theorem sum_of_radii_of_tangent_circles : 
  ∃ r1 r2 : ℝ, 
    r1 > 0 ∧
    r2 > 0 ∧
    ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
    ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧
    r1 + r2 = 12 :=
by
  sorry

end sum_of_radii_of_tangent_circles_l59_59440


namespace collinear_R_S_T_l59_59447

theorem collinear_R_S_T
    (circle : Type)
    (P : circle)
    (A B C D : circle)
    (E F : Type → Type)
    (angle : ∀ (x y z : circle), ℝ)   -- Placeholder for angles
    (quadrilateral_inscribed_in_circle : ∀ (A B C D : circle), Prop)   -- Placeholder for the condition of the quadrilateral
    (extensions_intersect : ∀ (A B C D : circle) (E F : Type → Type), Prop)   -- Placeholder for extensions intersections
    (diagonals_intersect_at : ∀ (A C B D T : circle), Prop)   -- Placeholder for diagonals intersections
    (P_on_circle : ∀ (P : circle), Prop)        -- Point P is on the circle
    (PE_PF_intersect_again : ∀ (P R S : circle) (E F : Type → Type), Prop)   -- PE and PF intersect the circle again at R and S
    (R S T : circle) :
    quadrilateral_inscribed_in_circle A B C D →
    extensions_intersect A B C D E F →
    P_on_circle P →
    PE_PF_intersect_again P R S E F →
    diagonals_intersect_at A C B D T →
    ∃ collinearity : ∀ (R S T : circle), Prop,
    collinearity R S T := 
by
  intro h1 h2 h3 h4 h5
  sorry

end collinear_R_S_T_l59_59447


namespace sticker_price_of_laptop_l59_59088

variable (x : ℝ)

-- Conditions
noncomputable def price_store_A : ℝ := 0.90 * x - 100
noncomputable def price_store_B : ℝ := 0.80 * x
noncomputable def savings : ℝ := price_store_B x - price_store_A x

-- Theorem statement
theorem sticker_price_of_laptop (x : ℝ) (h : savings x = 20) : x = 800 :=
by
  sorry

end sticker_price_of_laptop_l59_59088


namespace question_1_question_2_l59_59708

variable (m x : ℝ)
def f (x : ℝ) := |x + m|

theorem question_1 (h : f 1 + f (-2) ≥ 5) : 
  m ≤ -2 ∨ m ≥ 3 := sorry

theorem question_2 (hx : x ≠ 0) : 
  f (1 / x) + f (-x) ≥ 2 := sorry

end question_1_question_2_l59_59708


namespace circle_and_tangent_lines_l59_59630

open Real

noncomputable def equation_of_circle_center_on_line (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (x - a)^2 + (y - (a + 1))^2 = 2 ∧ (a = 4) ∧ (b = 5)

noncomputable def tangent_line_through_point (x y : ℝ) : Prop :=
  y = x - 1 ∨ y = (23 / 7) * x - (23 / 7)

theorem circle_and_tangent_lines :
  (∃ (a b : ℝ), (a = 4) ∧ (b = 5) ∧ (∀ x y : ℝ, equation_of_circle_center_on_line x y)) ∧
  (∀ x y : ℝ, tangent_line_through_point x y) := 
  by
  sorry

end circle_and_tangent_lines_l59_59630


namespace sector_properties_l59_59798

noncomputable def central_angle (l R : ℝ) : ℝ := l / R

noncomputable def area_of_sector (l R : ℝ) : ℝ := (1 / 2) * l * R

theorem sector_properties (R l : ℝ) (hR : R = 8) (hl : l = 12) :
  central_angle l R = 3 / 2 ∧ area_of_sector l R = 48 :=
by
  sorry

end sector_properties_l59_59798


namespace right_triangle_sum_of_squares_l59_59826

   theorem right_triangle_sum_of_squares {AB AC BC : ℝ} (h_right: AB^2 + AC^2 = BC^2) (h_hypotenuse: BC = 1) :
     AB^2 + AC^2 + BC^2 = 2 :=
   by
     sorry
   
end right_triangle_sum_of_squares_l59_59826


namespace smallest_b_for_factorable_polynomial_l59_59479

theorem smallest_b_for_factorable_polynomial :
  ∃ (b : ℕ), b > 0 ∧ (∃ (p q : ℤ), x^2 + b * x + 1176 = (x + p) * (x + q) ∧ p * q = 1176 ∧ p + q = b) ∧ 
  (∀ (b' : ℕ), b' > 0 → (∃ (p' q' : ℤ), x^2 + b' * x + 1176 = (x + p') * (x + q') ∧ p' * q' = 1176 ∧ p' + q' = b') → b ≤ b') :=
sorry

end smallest_b_for_factorable_polynomial_l59_59479


namespace geometric_sequence_sum_point_on_line_l59_59416

theorem geometric_sequence_sum_point_on_line
  (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (r : ℝ)
  (h1 : a 1 = t)
  (h2 : ∀ n : ℕ, a (n + 1) = t * r ^ n)
  (h3 : ∀ n : ℕ, S n = t * (1 - r ^ n) / (1 - r))
  (h4 : ∀ n : ℕ, (S n, a (n + 1)) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1})
  : t = 1 :=
by
  sorry

end geometric_sequence_sum_point_on_line_l59_59416


namespace part1_part2_l59_59635

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem part1 : {x : ℝ | f x ≤ 5} = {x : ℝ | -7 / 4 ≤ x ∧ x ≤ 3 / 4} :=
sorry

theorem part2 (h : ∃ x : ℝ, f x < |m - 2|) : m > 6 ∨ m < -2 :=
sorry

end part1_part2_l59_59635


namespace solution_set_inequality_l59_59584

theorem solution_set_inequality (x : ℝ) : (x^2-2*x-3)*(x^2+1) < 0 ↔ -1 < x ∧ x < 3 :=
by
  sorry

end solution_set_inequality_l59_59584


namespace ratio_of_larger_to_smaller_l59_59417

theorem ratio_of_larger_to_smaller (a b : ℝ) (h : a > 0) (h' : b > 0) (h_sum_diff : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l59_59417


namespace find_x_l59_59350

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x+1, -x)

def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x (x : ℝ) (h : perpendicular vector_a (vector_b x)) : x = 1 :=
by sorry

end find_x_l59_59350


namespace max_diff_distance_l59_59369

def hyperbola_right_branch (x y : ℝ) : Prop := 
  (x^2 / 9) - (y^2 / 16) = 1 ∧ x > 0

def circle_1 (x y : ℝ) : Prop := 
  (x + 5)^2 + y^2 = 4

def circle_2 (x y : ℝ) : Prop := 
  (x - 5)^2 + y^2 = 1

theorem max_diff_distance 
  (P M N : ℝ × ℝ) 
  (hp : hyperbola_right_branch P.fst P.snd) 
  (hm : circle_1 M.fst M.snd) 
  (hn : circle_2 N.fst N.snd) :
  |dist P M - dist P N| ≤ 9 := 
sorry

end max_diff_distance_l59_59369


namespace find_x_when_y_equals_two_l59_59677

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l59_59677


namespace Melanie_gumballs_sale_l59_59841

theorem Melanie_gumballs_sale (gumballs : ℕ) (price_per_gumball : ℕ) (total_price : ℕ) :
  gumballs = 4 →
  price_per_gumball = 8 →
  total_price = gumballs * price_per_gumball →
  total_price = 32 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end Melanie_gumballs_sale_l59_59841


namespace pqrs_sum_l59_59530

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l59_59530


namespace dr_reeds_statement_l59_59186

variables (P Q : Prop)

theorem dr_reeds_statement (h : P → Q) : ¬Q → ¬P :=
by sorry

end dr_reeds_statement_l59_59186


namespace payment_ways_l59_59746

-- Define basic conditions and variables
variables {x y z : ℕ}

-- Define the main problem as a Lean statement
theorem payment_ways : 
  ∃ (n : ℕ), n = 9 ∧ 
             (∀ x y z : ℕ, 
              x + y + z ≤ 10 ∧ 
              x + 2 * y + 5 * z = 18 ∧ 
              x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
              (x > 0 ∨ y > 0) ∧ (y > 0 ∨ z > 0) ∧ (z > 0 ∨ x > 0) → 
              n = 9) := 
sorry

end payment_ways_l59_59746


namespace value_of_y_l59_59064

theorem value_of_y (y : ℤ) (h : (2010 + y)^2 = y^2) : y = -1005 :=
sorry

end value_of_y_l59_59064


namespace base3_to_base10_conversion_l59_59945

theorem base3_to_base10_conversion : 
  1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0 = 142 :=
by {
  -- calculations
  sorry
}

end base3_to_base10_conversion_l59_59945


namespace percent_students_with_pets_l59_59176

theorem percent_students_with_pets 
  (total_students : ℕ) (students_with_cats : ℕ) (students_with_dogs : ℕ) (students_with_both : ℕ) (h_total : total_students = 500)
  (h_cats : students_with_cats = 150) (h_dogs : students_with_dogs = 100) (h_both : students_with_both = 40) :
  (students_with_cats + students_with_dogs - students_with_both) * 100 / total_students = 42 := 
by
  sorry

end percent_students_with_pets_l59_59176


namespace pet_store_profit_is_205_l59_59053

def brandon_selling_price : ℤ := 100
def pet_store_selling_price : ℤ := 5 + 3 * brandon_selling_price
def pet_store_profit : ℤ := pet_store_selling_price - brandon_selling_price

theorem pet_store_profit_is_205 :
  pet_store_profit = 205 := by
  sorry

end pet_store_profit_is_205_l59_59053


namespace largest_value_of_n_l59_59464

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l59_59464


namespace gcd_factorial_8_10_l59_59741

theorem gcd_factorial_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_factorial_8_10_l59_59741


namespace arithmetic_geometric_seq_l59_59968

theorem arithmetic_geometric_seq (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_diff : d = 2)
  (h_geom : (a 1)^2 = a 0 * (a 0 + 6)) :
  a 1 = -6 :=
by 
  sorry

end arithmetic_geometric_seq_l59_59968


namespace circle_diameter_l59_59906

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d, d = 4 :=
by
  let r := Real.sqrt 4
  let d := 2 * r
  use d
  simp only [Real.sqrt_eq_rfl, mul_eq_zero, ne.def, not_false_iff]
  linarith
  sorry

end circle_diameter_l59_59906


namespace find_m_and_other_root_l59_59343

theorem find_m_and_other_root (m x_2 : ℝ) :
  (∃ (x_1 : ℝ), x_1 = -1 ∧ x_1^2 + m * x_1 - 5 = 0) →
  m = -4 ∧ ∃ (x_2 : ℝ), x_2 = 5 ∧ x_2^2 + m * x_2 - 5 = 0 :=
by
  sorry

end find_m_and_other_root_l59_59343


namespace solve_quadratic_for_q_l59_59193

-- Define the quadratic equation and the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main theorem statement
theorem solve_quadratic_for_q : ∃ q : ℝ, q ≠ 0 ∧ (discriminant q (-8) 2 = 0) → q = 8 :=
by
  -- Insert the assumptions and proof context here.
  -- However, since we were instructed not to consider the solution steps
  -- the proof is skipped with a "sorry".
  sorry

end solve_quadratic_for_q_l59_59193


namespace octagon_diagonals_l59_59648

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l59_59648


namespace meaningful_fraction_iff_l59_59817

theorem meaningful_fraction_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (2 - x)) ↔ x ≠ 2 := by
  sorry

end meaningful_fraction_iff_l59_59817


namespace three_times_first_number_minus_second_value_l59_59148

theorem three_times_first_number_minus_second_value (x y : ℕ) 
  (h1 : x + y = 48) 
  (h2 : y = 17) : 
  3 * x - y = 76 := 
by 
  sorry

end three_times_first_number_minus_second_value_l59_59148


namespace total_number_of_cars_l59_59509

theorem total_number_of_cars (T A R : ℕ)
  (h1 : T - A = 37)
  (h2 : R ≥ 41)
  (h3 : ∀ x, x ≤ 59 → A = x + 37) :
  T = 133 :=
by
  sorry

end total_number_of_cars_l59_59509


namespace quadratic_grid_fourth_column_l59_59049

theorem quadratic_grid_fourth_column 
  (grid : ℕ → ℕ → ℝ)
  (row_quadratic : ∀ i : ℕ, (∃ a b c : ℝ, ∀ n : ℕ, grid i n = a * n^2 + b * n + c))
  (col_quadratic : ∀ j : ℕ, j ≤ 3 → (∃ a b c : ℝ, ∀ n : ℕ, grid n j = a * n^2 + b * n + c)) :
  ∃ a b c : ℝ, ∀ n : ℕ, grid n 4 = a * n^2 + b * n + c := 
sorry

end quadratic_grid_fourth_column_l59_59049


namespace pyramid_volume_formula_l59_59279

noncomputable def pyramid_volume (a α β : ℝ) : ℝ :=
  (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β)

theorem pyramid_volume_formula (a α β : ℝ) :
  (base_is_isosceles_triangle : Prop) → (lateral_edges_inclined : Prop) → 
  pyramid_volume a α β = (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β) :=
by
  intros c1 c2
  exact sorry

end pyramid_volume_formula_l59_59279


namespace mike_bricks_l59_59263

theorem mike_bricks (total_bricks bricks_A bricks_B bricks_other: ℕ) 
  (h1 : bricks_A = 40) 
  (h2 : bricks_B = bricks_A / 2)
  (h3 : total_bricks = 150) 
  (h4 : total_bricks = bricks_A + bricks_B + bricks_other) : bricks_other = 90 := 
by 
  sorry

end mike_bricks_l59_59263


namespace inequality_1_inequality_2_l59_59060

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem inequality_1 (x : ℝ) : f x > 2 * x ↔ x < -1/2 :=
sorry

theorem inequality_2 (t : ℝ) :
  (∃ x : ℝ, f x > t ^ 2 - t + 1) ↔ (0 < t ∧ t < 1) :=
sorry

end inequality_1_inequality_2_l59_59060


namespace find_speed_l59_59059

theorem find_speed (v : ℝ) (t : ℝ) (h : t = 5 * v^2) (ht : t = 20) : v = 2 :=
by
  sorry

end find_speed_l59_59059


namespace value_of_f_8_minus_f_4_l59_59983

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_8_minus_f_4 :
  -- Conditions
  (∀ x, f (-x) = -f x) ∧              -- odd function
  (∀ x, f (x + 5) = f x) ∧            -- period of 5
  (f 1 = 1) ∧                         -- f(1) = 1
  (f 2 = 3) →                         -- f(2) = 3
  -- Goal
  f 8 - f 4 = -2 :=
sorry

end value_of_f_8_minus_f_4_l59_59983


namespace tangent_line_parallel_x_axis_l59_59457

noncomputable def curve : ℝ → ℝ := λ x, 2 * x^3 - 6 * x

noncomputable def derivative (x : ℝ) : ℝ :=
  (deriv curve) x

theorem tangent_line_parallel_x_axis (x y : ℝ) (h : curve x = y) : 
  (derivative x = 0) ↔ (x = -1 ∧ y = 4) ∨ (x = 1 ∧ y = -4) :=
by
  -- Proof will go here
  sorry

end tangent_line_parallel_x_axis_l59_59457


namespace sufficient_but_not_necessary_condition_for_subset_l59_59636

variable {A B : Set ℕ}
variable {a : ℕ}

theorem sufficient_but_not_necessary_condition_for_subset (hA : A = {1, a}) (hB : B = {1, 2, 3}) :
  (a = 3 → A ⊆ B) ∧ (A ⊆ B → (a = 3 ∨ a = 2)) ∧ ¬(A ⊆ B → a = 3) := by
sorry

end sufficient_but_not_necessary_condition_for_subset_l59_59636


namespace function_range_l59_59289

theorem function_range (f : ℝ → ℝ) (s : Set ℝ) (h : s = Set.Ico (-5 : ℝ) 2) (h_f : ∀ x ∈ s, f x = 3 * x - 1) :
  Set.image f s = Set.Ico (-16 : ℝ) 5 :=
sorry

end function_range_l59_59289


namespace cost_price_of_computer_table_l59_59284

theorem cost_price_of_computer_table (C SP : ℝ) (h1 : SP = 1.25 * C) (h2 : SP = 8340) :
  C = 6672 :=
by
  sorry

end cost_price_of_computer_table_l59_59284


namespace abc_def_ratio_l59_59897

theorem abc_def_ratio (a b c d e f : ℝ)
    (h1 : a / b = 1 / 3)
    (h2 : b / c = 2)
    (h3 : c / d = 1 / 2)
    (h4 : d / e = 3)
    (h5 : e / f = 1 / 8) :
    (a * b * c) / (d * e * f) = 1 / 8 :=
by
  sorry

end abc_def_ratio_l59_59897


namespace y_relationship_l59_59489

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1: y1 = -4 / x1) (h2: y2 = -4 / x2) (h3: y3 = -4 / x3)
  (h4: x1 < 0) (h5: 0 < x2) (h6: x2 < x3) :
  y1 > y3 ∧ y3 > y2 :=
by
  sorry

end y_relationship_l59_59489


namespace solve_for_a_l59_59666

-- Given conditions
variables (a b d : ℕ)
hypotheses
  (h1 : a + b = d)
  (h2 : b + d = 7)
  (h3 : d = 4)

-- Prove that a = 1
theorem solve_for_a : a = 1 :=
by {
  sorry
}

end solve_for_a_l59_59666


namespace sum_fractions_geq_six_l59_59481

variable (x y z : ℝ)
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

theorem sum_fractions_geq_six : 
  (x / y + y / z + z / x + x / z + z / y + y / x) ≥ 6 := 
by
  sorry

end sum_fractions_geq_six_l59_59481


namespace subgroup_equality_example_l59_59777

theorem subgroup_equality_example :
  ∃ (x y : ℤ), x > 0 ∧ H = H_xy ∧ x = 7 ∧ y = 5 := 
by
  let H := Subgroup.closure ({ (3, 8), (4, -1), (5, 4) } : Set (ℤ × ℤ))
  let H_xy := Subgroup.closure ({ (0, 7), (1, 5) } : Set (ℤ × ℤ))
  use 7
  use 5
  split
  { norm_num },  -- For x > 0
  split
  { sorry },    -- Placeholder for proving H = H_xy
  split
  { refl },
  { refl }

end subgroup_equality_example_l59_59777


namespace solve_equation_l59_59134

theorem solve_equation (x : ℚ) :
  (x ≠ -10 ∧ x ≠ -8 ∧ x ≠ -11 ∧ x ≠ -7 ∧ (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7))) → x = -9 :=
by
  split
  sorry

end solve_equation_l59_59134


namespace george_correct_answer_l59_59484

variable (y : ℝ)

theorem george_correct_answer (h : y / 7 = 30) : 70 + y = 280 :=
sorry

end george_correct_answer_l59_59484


namespace sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l59_59255

def is_sum_of_arithmetic_sequence (S : ℕ → ℚ) (a₁ d : ℚ) :=
  ∀ n : ℕ, S n = n * a₁ + (n * (n - 1) / 2) * d

theorem sum_has_minimum_term_then_d_positive
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_min : ∃ n : ℕ, ∀ m : ℕ, S n ≤ S m) :
  d > 0 :=
sorry

theorem Sn_positive_then_increasing_sequence
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_pos : ∀ n : ℕ, S n > 0) :
  (∀ n : ℕ, S n < S (n + 1)) :=
sorry

end sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l59_59255


namespace arnold_plates_count_l59_59931

def arnold_barbell := 45
def mistaken_weight := 600
def actual_weight := 470
def weight_difference_per_plate := 10

theorem arnold_plates_count : 
  ∃ n : ℕ, mistaken_weight - actual_weight = n * weight_difference_per_plate ∧ n = 13 := 
sorry

end arnold_plates_count_l59_59931


namespace maximize_annual_avg_profit_l59_59937

-- Define the problem conditions
def purchase_cost : ℕ := 90000 
def initial_operating_cost : ℕ := 20000
def annual_cost_increase : ℕ := 20000
def annual_income : ℕ := 110000

-- Define the sequence for operating cost
def operating_cost (n : ℕ) : ℕ := 2 * n * 10000 -- in yuan

-- Total operating cost after n years
def total_operating_cost (n : ℕ) : ℕ := n * n * 10000 + n * 10000 -- in yuan

-- Total profit after n years
def total_profit (n : ℕ) : ℕ := 110000 * n - total_operating_cost n - purchase_cost

-- Annual average profit
def annual_avg_profit (n : ℕ) : ℕ := (total_profit n) / n

-- The goal statement
theorem maximize_annual_avg_profit : ∃ n : ℕ, n = 3 ∧ (∀ m : ℕ, m > 0 → annual_avg_profit n ≥ annual_avg_profit m) := sorry

end maximize_annual_avg_profit_l59_59937


namespace sufficient_but_not_necessary_l59_59370

variable (x y : ℝ)

theorem sufficient_but_not_necessary (x_gt_y_gt_zero : x > y ∧ y > 0) : (x / y > 1) :=
by
  sorry

end sufficient_but_not_necessary_l59_59370


namespace find_x_l59_59679

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l59_59679


namespace gcd_1729_1309_eq_7_l59_59594

theorem gcd_1729_1309_eq_7 : Nat.gcd 1729 1309 = 7 := by
  sorry

end gcd_1729_1309_eq_7_l59_59594


namespace largest_n_for_factored_quad_l59_59461

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l59_59461


namespace vikas_rank_among_boys_l59_59244

def vikas_rank_overall := 9
def tanvi_rank_overall := 17
def girls_between := 2
def vikas_rank_top_boys := 4
def vikas_rank_bottom_overall := 18

theorem vikas_rank_among_boys (vikas_rank_overall tanvi_rank_overall girls_between vikas_rank_top_boys vikas_rank_bottom_overall : ℕ) :
  vikas_rank_top_boys = 4 := by
  sorry

end vikas_rank_among_boys_l59_59244


namespace perfect_squares_l59_59404

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l59_59404


namespace landscape_breadth_l59_59319

theorem landscape_breadth (L B : ℕ)
  (h1 : B = 6 * L)
  (h2 : 4200 = (1 / 7 : ℚ) * 6 * L^2) :
  B = 420 := 
  sorry

end landscape_breadth_l59_59319


namespace pure_imaginary_b_eq_two_l59_59215

theorem pure_imaginary_b_eq_two (b : ℝ) : (∃ (im_part : ℝ), (1 + b * Complex.I) / (2 - Complex.I) = im_part * Complex.I) ↔ b = 2 :=
by
  sorry

end pure_imaginary_b_eq_two_l59_59215


namespace coeffs_sum_eq_40_l59_59627

theorem coeffs_sum_eq_40 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (2 * x - 1) ^ 5 = a_0 * x ^ 5 + a_1 * x ^ 4 + a_2 * x ^ 3 + a_3 * x ^ 2 + a_4 * x + a_5) :
  a_2 + a_3 = 40 :=
sorry

end coeffs_sum_eq_40_l59_59627


namespace problem1_problem2_l59_59804

-- Problem 1
theorem problem1 (a b c d : ℝ) (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : (c / a) - (d / b) > 0 := sorry

-- Problem 2
theorem problem2 (a b c d : ℝ) (ha_gt_b : a > b) (hc_gt_d : c > d) : a - d > b - c := sorry

end problem1_problem2_l59_59804


namespace diameter_of_circle_l59_59907

theorem diameter_of_circle (A : ℝ) (h : A = 4 * real.pi) : ∃ d : ℝ, d = 4 :=
  sorry

end diameter_of_circle_l59_59907


namespace range_of_m_l59_59094

variable (f : ℝ → ℝ)

theorem range_of_m (h_inc : ∀ x y : ℝ, x < y → f x < f y) :
  {m : ℝ | f (2 - m) < f (m^2)} = {m | m < -2} ∪ {m | m > 1} :=
by
  sorry

end range_of_m_l59_59094


namespace xiao_ming_final_score_l59_59895

theorem xiao_ming_final_score :
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  (speech_image * weight_speech_image +
   content * weight_content +
   effectiveness * weight_effectiveness) = 8.3 :=
by
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  sorry

end xiao_ming_final_score_l59_59895


namespace range_of_root_difference_l59_59978

variable (a b c d : ℝ)
variable (x1 x2 : ℝ)

def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_of_root_difference
  (h1 : a ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hroot1 : f a b c x1 = 0)
  (hroot2 : f a b c x2 = 0)
  : |x1 - x2| ∈ Set.Ico (Real.sqrt 3 / 3) (2 / 3) := sorry

end range_of_root_difference_l59_59978


namespace max_n_for_factored_polynomial_l59_59468

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l59_59468


namespace arithmetic_sequence_30th_term_l59_59023

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem arithmetic_sequence_30th_term :
  arithmetic_sequence 3 6 30 = 177 :=
by
  -- Proof steps go here
  sorry

end arithmetic_sequence_30th_term_l59_59023


namespace p_necessary_for_q_l59_59984

variable (x : ℝ)

def p := (x - 3) * (|x| + 1) < 0
def q := |1 - x| < 2

theorem p_necessary_for_q : (∀ x, q x → p x) ∧ (∃ x, q x) ∧ (∃ x, ¬(p x ∧ q x)) := by
  sorry

end p_necessary_for_q_l59_59984


namespace sin_squared_sum_eq_one_l59_59115

theorem sin_squared_sum_eq_one (α β γ : ℝ) 
  (h₁ : 0 ≤ α ∧ α ≤ π/2) 
  (h₂ : 0 ≤ β ∧ β ≤ π/2) 
  (h₃ : 0 ≤ γ ∧ γ ≤ π/2) 
  (h₄ : Real.sin α + Real.sin β + Real.sin γ = 1)
  (h₅ : Real.sin α * Real.cos (2 * α) + Real.sin β * Real.cos (2 * β) + Real.sin γ * Real.cos (2 * γ) = -1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1 := 
sorry

end sin_squared_sum_eq_one_l59_59115


namespace population_ratio_l59_59304

variables (Px Py Pz : ℕ)

theorem population_ratio (h1 : Py = 2 * Pz) (h2 : Px = 8 * Py) : Px / Pz = 16 :=
by
  sorry

end population_ratio_l59_59304


namespace num_children_proof_l59_59173

noncomputable def number_of_children (total_persons : ℕ) (total_revenue : ℕ) (adult_price : ℕ) (child_price : ℕ) : ℕ :=
  let adult_tickets := (child_price * total_persons - total_revenue) / (child_price - adult_price)
  let child_tickets := total_persons - adult_tickets
  child_tickets

theorem num_children_proof : number_of_children 280 14000 60 25 = 80 := 
by
  unfold number_of_children
  sorry

end num_children_proof_l59_59173


namespace find_x_l59_59680

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l59_59680


namespace least_number_subtracted_l59_59158

theorem least_number_subtracted (n : ℕ) (h : n = 427398) : ∃ x, x = 8 ∧ (n - x) % 10 = 0 :=
by
  sorry

end least_number_subtracted_l59_59158


namespace minimum_and_maximum_attendees_more_than_one_reunion_l59_59587

noncomputable def minimum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  let total_unique_attendees := oates_attendees + hall_attendees + brown_attendees
  total_unique_attendees - total_guests

noncomputable def maximum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  oates_attendees

theorem minimum_and_maximum_attendees_more_than_one_reunion
  (total_guests oates_attendees hall_attendees brown_attendees : ℕ)
  (H1 : total_guests = 200)
  (H2 : oates_attendees = 60)
  (H3 : hall_attendees = 90)
  (H4 : brown_attendees = 80) :
  minimum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 30 ∧
  maximum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 60 :=
by
  sorry

end minimum_and_maximum_attendees_more_than_one_reunion_l59_59587


namespace solution_of_system_l59_59849

theorem solution_of_system :
  (∀ x : ℝ,
    (2 + x < 6 - 3 * x) ∧ (x ≤ (4 + x) / 2)
    → x < 1) :=
by
  sorry

end solution_of_system_l59_59849


namespace intersection_m_zero_range_of_m_l59_59231

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x : ℝ) (m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

theorem intersection_m_zero : 
  ∀ x : ℝ, A x → B x 0 ↔ (1 ≤ x ∧ x < 3) :=
sorry

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, A x → B x m) ∧ (∃ x : ℝ, B x m ∧ ¬A x) → (m ≤ -2 ∨ m ≥ 4) :=
sorry

end intersection_m_zero_range_of_m_l59_59231


namespace coin_toss_sequences_l59_59327

noncomputable def count_sequences (n m : ℕ) : ℕ := Nat.choose (n + m - 1) (m - 1)

theorem coin_toss_sequences :
  ∃ (seq_count : ℕ), 
    seq_count = (count_sequences 3 3) * (count_sequences 6 4) ∧ seq_count = 840 :=
by
  use ((count_sequences 3 3) * (count_sequences 6 4))
  split
  { sorry, }
  { sorry, }

end coin_toss_sequences_l59_59327


namespace jerusha_and_lottie_earnings_l59_59368

theorem jerusha_and_lottie_earnings :
  let J := 68
  let L := J / 4
  J + L = 85 := 
by
  sorry

end jerusha_and_lottie_earnings_l59_59368


namespace circumference_of_smaller_circle_l59_59138

theorem circumference_of_smaller_circle (C₁ : ℝ) (C₂ : ℝ) (A_diff : ℝ) : 
  C₁ = 352 → 
  A_diff = 4313.735577562732 → 
  ∃ (C : ℝ), C ≈ 263.8935 :=
sorry

end circumference_of_smaller_circle_l59_59138


namespace sequence_recurrence_l59_59555

theorem sequence_recurrence (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : a 2 = 2) (h₃ : ∀ n, n ≥ 1 → a (n + 2) / a n = (a (n + 1) ^ 2 + 1) / (a n ^ 2 + 1)):
  (∀ n, a (n + 1) = a n + 1 / a n) ∧ 63 < a 2008 ∧ a 2008 < 78 :=
by
  sorry

end sequence_recurrence_l59_59555


namespace circle_diameter_l59_59911

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end circle_diameter_l59_59911


namespace compute_value_of_fractions_l59_59574

theorem compute_value_of_fractions (a b c : ℝ) 
  (h1 : (ac / (a + b)) + (ba / (b + c)) + (cb / (c + a)) = 0)
  (h2 : (bc / (a + b)) + (ca / (b + c)) + (ab / (c + a)) = 1) :
  (b / (a + b)) + (c / (b + c)) + (a / (c + a)) = 5 / 2 :=
sorry

end compute_value_of_fractions_l59_59574


namespace proof1_proof2_l59_59214

open Real

noncomputable def problem1 (a b c : ℝ) (A : ℝ) (S : ℝ) :=
  ∃ (a b : ℝ), A = π / 3 ∧ c = 2 ∧ S = sqrt 3 / 2 ∧ S = 1/2 * b * 2 * sin (π / 3) ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos (π / 3) ∧ b = 1 ∧ a = sqrt 3

noncomputable def problem2 (a b c : ℝ) (A B : ℝ) :=
  c = a * cos B ∧ (a + b + c) * (a + b - c) = (2 + sqrt 2) * a * b ∧ 
  B = π / 4 ∧ A = π / 2 → 
  ∃ C, C = π / 4 ∧ C = B

theorem proof1 : problem1 (sqrt 3) 1 2 (π / 3) (sqrt 3 / 2) :=
by
  sorry

theorem proof2 : problem2 (sqrt 3) 1 2 (π / 2) (π / 4) :=
by
  sorry

end proof1_proof2_l59_59214


namespace height_percentage_difference_l59_59046

theorem height_percentage_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.5384615384615385) :
  (H_B - H_A) / H_B * 100 = 35 := 
sorry

end height_percentage_difference_l59_59046


namespace proof_mod_55_l59_59525

theorem proof_mod_55 (M : ℕ) (h1 : M % 5 = 3) (h2 : M % 11 = 9) : M % 55 = 53 := 
  sorry

end proof_mod_55_l59_59525


namespace limit_an_to_a_l59_59380

theorem limit_an_to_a (ε : ℝ) (hε : ε > 0) : 
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
  |(9 - (n^3 : ℝ)) / (1 + 2 * (n^3 : ℝ)) + 1/2| < ε :=
sorry

end limit_an_to_a_l59_59380


namespace find_ab_l59_59030

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 :=
by
  sorry

end find_ab_l59_59030


namespace restaurant_june_production_l59_59612

-- Define the given conditions
def daily_hot_dogs := 60
def daily_pizzas := daily_hot_dogs + 40
def june_days := 30
def daily_total := daily_hot_dogs + daily_pizzas
def june_total := daily_total * june_days

-- The goal is to prove that the total number of pizzas and hot dogs made in June is 4800
theorem restaurant_june_production : june_total = 4800 := by
  -- Sorry to skip proof
  sorry

end restaurant_june_production_l59_59612


namespace sum_alternating_binom_eq_pow_of_two_l59_59056

theorem sum_alternating_binom_eq_pow_of_two :
  (∑ k in Finset.range 51, (-1)^k * Nat.choose 101 (2*k)) = 2^50 :=
sorry

end sum_alternating_binom_eq_pow_of_two_l59_59056


namespace probability_woman_lawyer_aged_36_45_l59_59823

def P_woman : ℝ := 0.40
def P_lawyer_given_woman : ℝ := 0.20
def P_aged_36_45_given_woman_lawyer : ℝ := 0.05

theorem probability_woman_lawyer_aged_36_45 :
  P_woman * P_lawyer_given_woman * P_aged_36_45_given_woman_lawyer = 0.004 :=
by
  sorry

end probability_woman_lawyer_aged_36_45_l59_59823


namespace find_number_l59_59445

theorem find_number (x : ℝ) : 
  ( ((x - 1.9) * 1.5 + 32) / 2.5 = 20 ) → x = 13.9 :=
by
  sorry

end find_number_l59_59445


namespace stamps_cost_l59_59125

theorem stamps_cost (cost_one: ℝ) (cost_three: ℝ) (h: cost_one = 0.34) (h1: cost_three = 3 * cost_one) : 
  2 * cost_one = 0.68 := 
by
  sorry

end stamps_cost_l59_59125


namespace percent_savings_correct_l59_59028

theorem percent_savings_correct :
  let cost_of_package := 9
  let num_of_rolls_in_package := 12
  let cost_per_roll_individually := 1
  let cost_per_roll_in_package := cost_of_package / num_of_rolls_in_package
  let savings_per_roll := cost_per_roll_individually - cost_per_roll_in_package
  let percent_savings := (savings_per_roll / cost_per_roll_individually) * 100
  percent_savings = 25 :=
by
  sorry

end percent_savings_correct_l59_59028


namespace sandy_books_l59_59130

theorem sandy_books (x : ℕ)
  (h1 : 1080 + 840 = 1920)
  (h2 : 16 = 1920 / (x + 55)) :
  x = 65 :=
by
  -- Theorem proof placeholder
  sorry

end sandy_books_l59_59130


namespace masha_wins_l59_59899

def num_matches : Nat := 111

-- Define a function for Masha's optimal play strategy
-- In this problem, we'll denote both players' move range and the condition for winning.
theorem masha_wins (n : Nat := num_matches) (conditions : n > 0 ∧ n % 11 = 0 ∧ (∀ k : Nat, 1 ≤ k ∧ k ≤ 10 → ∃ new_n : Nat, n = k + new_n)) : True :=
  sorry

end masha_wins_l59_59899


namespace smallest_positive_integer_N_l59_59204

theorem smallest_positive_integer_N :
  ∃ N : ℕ, N > 0 ∧ (N % 7 = 5) ∧ (N % 8 = 6) ∧ (N % 9 = 7) ∧ (∀ M : ℕ, M > 0 ∧ (M % 7 = 5) ∧ (M % 8 = 6) ∧ (M % 9 = 7) → N ≤ M) :=
sorry

end smallest_positive_integer_N_l59_59204


namespace a_share_is_approx_560_l59_59603

noncomputable def investment_share (a_invest b_invest c_invest total_months b_share : ℕ) : ℝ :=
  let total_invest := a_invest + b_invest + c_invest
  let total_profit := (b_share * total_invest) / b_invest
  let a_share_ratio := a_invest / total_invest
  (a_share_ratio * total_profit)

theorem a_share_is_approx_560 
  (a_invest : ℕ := 7000) 
  (b_invest : ℕ := 11000) 
  (c_invest : ℕ := 18000) 
  (total_months : ℕ := 8) 
  (b_share : ℕ := 880) : 
  ∃ (a_share : ℝ), abs (a_share - 560) < 1 :=
by
  let a_share := investment_share a_invest b_invest c_invest total_months b_share
  existsi a_share
  sorry

end a_share_is_approx_560_l59_59603


namespace cos_of_theta_l59_59925

theorem cos_of_theta
  (A : ℝ) (a : ℝ) (m : ℝ) (θ : ℝ) 
  (hA : A = 40) 
  (ha : a = 12) 
  (hm : m = 10) 
  (h_area: A = (1/2) * a * m * Real.sin θ) 
  : Real.cos θ = (Real.sqrt 5) / 3 :=
by
  sorry

end cos_of_theta_l59_59925


namespace part1_part2_l59_59800

open Complex

noncomputable def z1 : ℂ := 1 - 2 * I
noncomputable def z2 : ℂ := 4 + 3 * I

theorem part1 : z1 * z2 = 10 - 5 * I := by
  sorry

noncomputable def z : ℂ := -Real.sqrt 2 - Real.sqrt 2 * I

theorem part2 (h_abs_z : abs z = 2)
              (h_img_eq_real : z.im = (3 * z1 - z2).re)
              (h_quadrant : z.re < 0 ∧ z.im < 0) : z = -Real.sqrt 2 - Real.sqrt 2 * I := by
  sorry

end part1_part2_l59_59800


namespace base_six_representation_l59_59482

theorem base_six_representation (b : ℕ) (h₁ : b = 6) :
  625₁₀.toDigits b = [2, 5, 2, 1] ∧ (625₁₀.toDigits b).length = 4 ∧ (625₁₀.toDigits b).head % 2 = 1 :=
by
  sorry

end base_six_representation_l59_59482


namespace find_number_l59_59752

theorem find_number (x : ℝ) : 0.40 * x = 0.80 * 5 + 2 → x = 15 :=
by
  intros h
  sorry

end find_number_l59_59752


namespace power_log_simplification_l59_59429

theorem power_log_simplification (x : ℝ) (h : x > 0) : (16^(Real.log x / Real.log 2))^(1/4) = x :=
by sorry

end power_log_simplification_l59_59429


namespace simplify_expression_l59_59569

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l59_59569


namespace mike_ride_distance_l59_59604

-- Definitions from conditions
def mike_cost (m : ℕ) : ℝ := 2.50 + 0.25 * m
def annie_cost : ℝ := 2.50 + 5.00 + 0.25 * 16

-- Theorem to prove
theorem mike_ride_distance (m : ℕ) (h : mike_cost m = annie_cost) : m = 36 := by
  sorry

end mike_ride_distance_l59_59604


namespace largest_neg_integer_solution_l59_59962

theorem largest_neg_integer_solution 
  (x : ℤ) 
  (h : 34 * x + 6 ≡ 2 [ZMOD 20]) : 
  x = -6 := 
sorry

end largest_neg_integer_solution_l59_59962


namespace count_positive_integers_satisfying_inequality_l59_59951

theorem count_positive_integers_satisfying_inequality :
  ∃ n : ℕ, n = 4 ∧ ∀ x : ℕ, (10 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 50) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) := 
by
  sorry

end count_positive_integers_satisfying_inequality_l59_59951


namespace parabola_coeff_sum_l59_59311

def parabola_vertex_form (a b c : ℚ) : Prop :=
  (∀ y : ℚ, y = 2 → (-3) = a * (y - 2)^2 + b * (y - 2) + c) ∧
  (∀ x y : ℚ, x = 1 ∧ y = -1 → x = a * y^2 + b * y + c) ∧
  (a < 0)  -- Since the parabola opens to the left, implying the coefficient 'a' is positive.

theorem parabola_coeff_sum (a b c : ℚ) :
  parabola_vertex_form a b c → a + b + c = -23 / 9 :=
by
  sorry

end parabola_coeff_sum_l59_59311


namespace remaining_macaroons_weight_is_103_l59_59523

-- Definitions based on the conditions
def coconutMacaroonsInitialCount := 12
def coconutMacaroonWeight := 5
def coconutMacaroonsBags := 4

def almondMacaroonsInitialCount := 8
def almondMacaroonWeight := 8
def almondMacaroonsBags := 2

def whiteChocolateMacaroonsInitialCount := 2
def whiteChocolateMacaroonWeight := 10

def steveAteCoconutMacaroons := coconutMacaroonsInitialCount / coconutMacaroonsBags
def steveAteAlmondMacaroons := (almondMacaroonsInitialCount / almondMacaroonsBags) / 2
def steveAteWhiteChocolateMacaroons := 1

-- Calculation of remaining macaroons weights
def remainingCoconutMacaroonsCount := coconutMacaroonsInitialCount - steveAteCoconutMacaroons
def remainingAlmondMacaroonsCount := almondMacaroonsInitialCount - steveAteAlmondMacaroons
def remainingWhiteChocolateMacaroonsCount := whiteChocolateMacaroonsInitialCount - steveAteWhiteChocolateMacaroons

-- Calculation of total remaining weight
def remainingCoconutMacaroonsWeight := remainingCoconutMacaroonsCount * coconutMacaroonWeight
def remainingAlmondMacaroonsWeight := remainingAlmondMacaroonsCount * almondMacaroonWeight
def remainingWhiteChocolateMacaroonsWeight := remainingWhiteChocolateMacaroonsCount * whiteChocolateMacaroonWeight

def totalRemainingWeight := remainingCoconutMacaroonsWeight + remainingAlmondMacaroonsWeight + remainingWhiteChocolateMacaroonsWeight

-- Statement to be proved
theorem remaining_macaroons_weight_is_103 :
  totalRemainingWeight = 103 := by
  sorry

end remaining_macaroons_weight_is_103_l59_59523


namespace sum_is_odd_square_expression_is_odd_l59_59086

theorem sum_is_odd_square_expression_is_odd (a b c : ℤ) (h : (a + b + c) % 2 = 1) : 
  (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 :=
sorry

end sum_is_odd_square_expression_is_odd_l59_59086


namespace city_division_exists_l59_59692

-- Define the problem conditions and prove the required statement
theorem city_division_exists (squares : Type) (streets : squares → squares → Prop)
  (h_outgoing: ∀ (s : squares), ∃ t u : squares, streets s t ∧ streets s u) :
  ∃ (districts : squares → ℕ), (∀ (s t : squares), districts s ≠ districts t → streets s t ∨ streets t s) ∧
  (∀ (i j : ℕ), i ≠ j → ∀ (s t : squares), districts s = i → districts t = j → streets s t ∨ streets t s) ∧
  (∃ m : ℕ, m = 1014) :=
sorry

end city_division_exists_l59_59692


namespace geom_mean_does_not_exist_l59_59577

theorem geom_mean_does_not_exist (a b : Real) (h1 : a = 2) (h2 : b = -2) : ¬ ∃ g : Real, g^2 = a * b := 
by
  sorry

end geom_mean_does_not_exist_l59_59577


namespace ratio_of_combined_areas_l59_59731

theorem ratio_of_combined_areas :
  let side_length_A := 36
  let side_length_B := 42
  let side_length_C := 48
  let area_A := side_length_A ^ 2
  let area_B := side_length_B ^ 2
  let area_C := side_length_C ^ 2
  let combined_area_AC := area_A + area_C
  (combined_area_AC : ℚ) / area_B = 20 / 7 :=
by {
  let side_length_A := 36
  let side_length_B := 42
  let side_length_C := 48
  let area_A := side_length_A ^ 2
  let area_B := side_length_B ^ 2
  let area_C := side_length_C ^ 2
  let combined_area_AC := area_A + area_C
  sorry
}

end ratio_of_combined_areas_l59_59731


namespace longest_badminton_match_duration_l59_59580

theorem longest_badminton_match_duration :
  let hours := 12
  let minutes := 25
  (hours * 60 + minutes = 745) :=
by
  sorry

end longest_badminton_match_duration_l59_59580


namespace probability_none_given_not_D_l59_59511

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]

-- Define the probabilities for various combinations of risk factors
def P_single (P_single_val : ℝ) : Prop := P_single_val = 0.08
def P_double (P_double_val : ℝ) : Prop := P_double_val = 0.2
def P_all_cond (P_all_val P_two_val : ℝ) : Prop := P_all_val = (1/4) * (P_all_val + P_two_val)
def P_none (P_none_val : ℝ) : Prop := P_none_val = 0.05

-- Define the conditional probability we need to prove
def P_none_given_not_D (P : MeasureTheory.ProbabilityMeasure Ω) := 
  P ({ω : Ω | ω ∉ {D}} ∩ {ω | ¬D ω ∧ ¬E ω ∧ ¬F ω}) = 1/5

theorem probability_none_given_not_D
  (P_single_val P_double_val P_none_val P_all_val P_two_val : ℝ)
  (D E F : Ω → Prop) [MeasurableSet (set_of D)]
  (P : MeasureTheory.ProbabilityMeasure Ω)
  (h1 : P_single P_single_val)
  (h2 : P_double P_double_val)
  (h3 : P_all_cond P_all_val P_two_val)
  (h4 : P_none P_none_val) :
  P_none_given_not_D P := 
by
  sorry

end probability_none_given_not_D_l59_59511


namespace find_k_from_hexadecimal_to_decimal_l59_59100

theorem find_k_from_hexadecimal_to_decimal 
  (k : ℕ) 
  (h : 1 * 6^3 + k * 6 + 5 = 239) : 
  k = 3 := by
  sorry

end find_k_from_hexadecimal_to_decimal_l59_59100


namespace inequality1_solution_inequality2_solution_l59_59387

-- Definitions for the conditions
def cond1 (x : ℝ) : Prop := abs (1 - (2 * x - 1) / 3) ≤ 2
def cond2 (x : ℝ) : Prop := (2 - x) * (x + 3) < 2 - x

-- Lean 4 statement for the proof problem
theorem inequality1_solution (x : ℝ) : cond1 x → -1 ≤ x ∧ x ≤ 5 := by
  sorry

theorem inequality2_solution (x : ℝ) : cond2 x → x > 2 ∨ x < -2 := by
  sorry

end inequality1_solution_inequality2_solution_l59_59387


namespace solve_for_m_l59_59989

theorem solve_for_m (x m : ℝ) (h : (∃ x, (x - 1) / (x - 4) = m / (x - 4))): 
  m = 3 :=
by {
  sorry -- placeholder to indicate where the proof would go
}

end solve_for_m_l59_59989


namespace perfect_square_of_factorials_l59_59892

open Nat

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfect_square_of_factorials :
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  is_perfect_square E3 :=
by
  -- definition of E1, E2, E3, E4, E5 as expressions given conditions
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  
  -- specify that E3 is the perfect square
  show is_perfect_square E3

  sorry

end perfect_square_of_factorials_l59_59892


namespace simplify_expression_eq_l59_59756

theorem simplify_expression_eq (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) : 
  (a - 1/a) / ((a^2 - 2 * a + 1) / a) = (a + 1) / (a - 1) :=
by
  sorry

end simplify_expression_eq_l59_59756


namespace mary_investment_l59_59711

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem mary_investment :
  ∃ (P : ℝ), P = 51346 ∧ compound_interest P 0.10 12 7 = 100000 :=
by
  sorry

end mary_investment_l59_59711


namespace solve_for_x_l59_59814

theorem solve_for_x (x y z : ℕ) 
  (h1 : 3^x * 4^y / 2^z = 59049)
  (h2 : x - y + 2 * z = 10) : 
  x = 10 :=
sorry

end solve_for_x_l59_59814


namespace find_ratio_MH_NH_OH_l59_59101

-- Defining the main problem variables.
variable {A B C O H M N : Type} -- A, B, C are points, O is circumcenter, H is orthocenter, M and N are points on other segments
variables (angleA : ℝ) (AB AC : ℝ)
variables (angleBOC angleBHC : ℝ)
variables (BM CN MH NH OH : ℝ)

-- Conditions: Given constraints from the problem.
axiom angle_A_eq_60 : angleA = 60 -- ∠A = 60°
axiom AB_greater_AC : AB > AC -- AB > AC
axiom circumcenter_property : angleBOC = 120 -- ∠BOC = 120°
axiom orthocenter_property : angleBHC = 120 -- ∠BHC = 120°
axiom BM_eq_CN : BM = CN -- BM = CN

-- Statement of the mathematical proof we need to show.
theorem find_ratio_MH_NH_OH : (MH + NH) / OH = Real.sqrt 3 :=
by
  sorry

end find_ratio_MH_NH_OH_l59_59101


namespace problem_a1_value_l59_59240

theorem problem_a1_value (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (h : ∀ x : ℝ, x^10 = a + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 + a₄ * (x - 1)^4 + a₅ * (x - 1)^5 + a₆ * (x - 1)^6 + a₇ * (x - 1)^7 + a₈ * (x - 1)^8 + a₉ * (x - 1)^9 + a₁₀ * (x - 1)^10) :
  a₁ = 10 :=
sorry

end problem_a1_value_l59_59240


namespace acute_angle_slope_neg_product_l59_59395

   theorem acute_angle_slope_neg_product (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (acute_inclination : ∃ (k : ℝ), k > 0 ∧ y = -a/b): (a * b < 0) :=
   by
     sorry
   
end acute_angle_slope_neg_product_l59_59395


namespace both_owners_count_l59_59773

-- Define the sets and counts as given in the conditions
variable (total_students : ℕ) (rabbit_owners : ℕ) (guinea_pig_owners : ℕ) (both_owners : ℕ)

-- Assume the values given in the problem
axiom total : total_students = 50
axiom rabbits : rabbit_owners = 35
axiom guinea_pigs : guinea_pig_owners = 40

-- The theorem to prove
theorem both_owners_count : both_owners = rabbit_owners + guinea_pig_owners - total_students := by
  sorry

end both_owners_count_l59_59773


namespace ab_plus_cd_is_composite_l59_59524

theorem ab_plus_cd_is_composite 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_eq : a^2 + a * c - c^2 = b^2 + b * d - d^2) : 
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ ab + cd = p * q :=
by
  sorry

end ab_plus_cd_is_composite_l59_59524


namespace candy_bar_calories_l59_59729

theorem candy_bar_calories:
  ∀ (calories_per_candy_bar : ℕ) (num_candy_bars : ℕ), 
  calories_per_candy_bar = 3 → 
  num_candy_bars = 5 → 
  calories_per_candy_bar * num_candy_bars = 15 :=
by
  sorry

end candy_bar_calories_l59_59729


namespace part1_part2_part3_l59_59389

-- Definition of the given expression
def expr (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 2) - (5 * x^2 + 3 * x)

-- Condition 1: Given final result 2x^2 - 4x + 2
def target_expr1 (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 2

-- Condition 2: Given values for a and b by Student B
def student_b_expr (x : ℝ) : ℝ := (5 * x^2 - 3 * x + 2) - (5 * x^2 + 3 * x)

-- Condition 3: Result independent of x
def target_expr3 : ℝ := 2

-- Prove conditions and answers
theorem part1 (a b : ℝ) : (∀ x : ℝ, expr a b x = target_expr1 x) → a = 7 ∧ b = -1 :=
sorry

theorem part2 : (∀ x : ℝ, student_b_expr x = -6 * x + 2) :=
sorry

theorem part3 (a b : ℝ) : (∀ x : ℝ, expr a b x = 2) → a = 5 ∧ b = 3 :=
sorry

end part1_part2_part3_l59_59389


namespace arcsin_one_eq_pi_div_two_l59_59939

noncomputable def arcsin : ℝ → ℝ := sorry -- Define arcsin function

theorem arcsin_one_eq_pi_div_two : arcsin 1 = Real.pi / 2 := sorry

end arcsin_one_eq_pi_div_two_l59_59939


namespace find_B_investment_l59_59751

def A_investment : ℝ := 24000
def C_investment : ℝ := 36000
def C_profit : ℝ := 36000
def total_profit : ℝ := 92000
def B_investment := 32000

theorem find_B_investment (B_investment_unknown : ℝ) :
  (C_investment / C_profit) = ((A_investment + B_investment_unknown + C_investment) / total_profit) →
  B_investment_unknown = B_investment := 
by 
  -- Mathematical equivalence to the given problem
  -- Proof omitted since only the statement is required
  sorry

end find_B_investment_l59_59751


namespace find_sum_of_squares_l59_59526

variable {R : Type*} [Field R]

-- Define the matrix B
def B (e f g h i j : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![e, f, g],
    ![f, h, i],
    ![g, i, j]]

-- B is symmetric
lemma B_symmetric (e f g h i j : R) : (B e f g h i j)ᵀ = B e f g h i j :=
  by simp [B, Matrix.transpose]

-- B is orthogonal
lemma B_orthogonal (e f g h i j : R) (h₁ : (B e f g h i j) ⬝ (B e f g h i j) = 1) : 
  (Matrix.mul (B e f g h i j) (B e f g h i j)) = 1 := 
  by assumption

-- The main theorem
theorem find_sum_of_squares (e f g h i j : ℝ) 
  (h₁ : (B e f g h i j) ⬝ (B e f g h i j) = 1) :
  e^2 + f^2 + g^2 + h^2 + i^2 + j^2 = 3 :=
  by 
    -- Expand Matrix multiplication and use the given conditions
    sorry

end find_sum_of_squares_l59_59526


namespace vincent_total_loads_l59_59151

def loads_wednesday : Nat := 2 + 1 + 3

def loads_thursday : Nat := 2 * loads_wednesday

def loads_friday : Nat := loads_thursday / 2

def loads_saturday : Nat := loads_wednesday / 3

def total_loads : Nat := loads_wednesday + loads_thursday + loads_friday + loads_saturday

theorem vincent_total_loads : total_loads = 20 := by
  -- Proof will be filled in here
  sorry

end vincent_total_loads_l59_59151


namespace extra_flowers_l59_59178

-- Definitions from the conditions
def tulips : Nat := 57
def roses : Nat := 73
def daffodils : Nat := 45
def sunflowers : Nat := 35
def used_flowers : Nat := 181

-- Statement to prove
theorem extra_flowers : (tulips + roses + daffodils + sunflowers) - used_flowers = 29 := by
  sorry

end extra_flowers_l59_59178


namespace length_of_bridge_l59_59303

theorem length_of_bridge (t : ℝ) (s : ℝ) (d : ℝ) : 
  (t = 24 / 60) ∧ (s = 10) ∧ (d = s * t) → d = 4 := by
  sorry

end length_of_bridge_l59_59303


namespace count_proper_subset_pairs_l59_59335

open Finset

-- Define the set S = {1, 2, 3, 4, 5, 6}
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- The main theorem statement
theorem count_proper_subset_pairs : 
  let number_of_pairs := ∑ B in S.powerset.filter (λ b, b ≠ ∅),
                           2^(B.card) - 1
  in
  number_of_pairs = 665 :=
by
  sorry

end count_proper_subset_pairs_l59_59335


namespace quadratic_inequality_iff_abs_a_le_two_l59_59309

-- Definitions from the condition
variable (a : ℝ)
def quadratic_expr (x : ℝ) : ℝ := x^2 + a * x + 1

-- Statement of the problem as a Lean 4 statement
theorem quadratic_inequality_iff_abs_a_le_two :
  (∀ x : ℝ, quadratic_expr a x ≥ 0) ↔ (|a| ≤ 2) := sorry

end quadratic_inequality_iff_abs_a_le_two_l59_59309


namespace probability_of_event_l59_59377

noncomputable def drawing_probability : ℚ := 
  let total_outcomes := 81
  let successful_outcomes :=
    (9 + 9 + 9 + 9 + 9 + 7 + 5 + 3 + 1)
  successful_outcomes / total_outcomes

theorem probability_of_event :
  drawing_probability = 61 / 81 := 
by
  sorry

end probability_of_event_l59_59377


namespace choir_minimum_members_l59_59041

theorem choir_minimum_members (n : ℕ) :
  (∃ k1, n = 8 * k1) ∧ (∃ k2, n = 9 * k2) ∧ (∃ k3, n = 10 * k3) → n = 360 :=
by
  sorry

end choir_minimum_members_l59_59041


namespace most_followers_after_three_weeks_l59_59388

def initial_followers_susy := 100
def initial_followers_sarah := 50
def first_week_gain_susy := 40
def second_week_gain_susy := first_week_gain_susy / 2
def third_week_gain_susy := second_week_gain_susy / 2
def first_week_gain_sarah := 90
def second_week_gain_sarah := first_week_gain_sarah / 3
def third_week_gain_sarah := second_week_gain_sarah / 3

def total_followers_susy := initial_followers_susy + first_week_gain_susy + second_week_gain_susy + third_week_gain_susy
def total_followers_sarah := initial_followers_sarah + first_week_gain_sarah + second_week_gain_sarah + third_week_gain_sarah

theorem most_followers_after_three_weeks : max total_followers_susy total_followers_sarah = 180 :=
by
  sorry

end most_followers_after_three_weeks_l59_59388


namespace parabolas_intersect_at_points_l59_59184

theorem parabolas_intersect_at_points :
  ∃ (x y : ℝ), (y = 3 * x^2 - 5 * x + 1 ∧ y = 4 * x^2 + 3 * x + 1) ↔ ((x = 0 ∧ y = 1) ∨ (x = -8 ∧ y = 233)) := 
sorry

end parabolas_intersect_at_points_l59_59184


namespace inequality_proof_l59_59668

variable {a b c : ℝ}

theorem inequality_proof (h : a > b) : (a / (c^2 + 1)) > (b / (c^2 + 1)) := by
  sorry

end inequality_proof_l59_59668


namespace probability_at_least_one_six_l59_59313

theorem probability_at_least_one_six (h: ℚ) : h = 91 / 216 :=
by 
  sorry

end probability_at_least_one_six_l59_59313


namespace smallest_x_for_palindrome_l59_59427

-- Define the condition for a number to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Mathematically equivalent proof problem statement
theorem smallest_x_for_palindrome : ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 2345) ∧ x = 97 :=
by sorry

end smallest_x_for_palindrome_l59_59427


namespace tan_ratio_l59_59704

-- Given conditions
variables {p q : ℝ} (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3)

-- The theorem we need to prove
theorem tan_ratio (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3) : 
  Real.tan p / Real.tan q = -1 / 3 :=
sorry

end tan_ratio_l59_59704


namespace largest_five_digit_sum_twenty_l59_59886

theorem largest_five_digit_sum_twenty : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000 ∧ (n.digits.sum = 20) ∧ (∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m.digits.sum = 20 → m ≤ n)) ∧ n = 99200 :=
sorry

end largest_five_digit_sum_twenty_l59_59886


namespace length_of_platform_l59_59749

-- Definitions based on the problem conditions
def train_length : ℝ := 300
def platform_crossing_time : ℝ := 39
def signal_pole_crossing_time : ℝ := 18

-- The main theorem statement
theorem length_of_platform : ∀ (L : ℝ), train_length + L = (train_length / signal_pole_crossing_time) * platform_crossing_time → L = 350.13 :=
by
  intro L h
  sorry

end length_of_platform_l59_59749


namespace find_f_4_l59_59669

noncomputable def f (x : ℕ) (a b c : ℕ) : ℕ := 2 * a * x + b * x + c

theorem find_f_4
  (a b c : ℕ)
  (f1 : f 1 a b c = 10)
  (f2 : f 2 a b c = 20) :
  f 4 a b c = 40 :=
sorry

end find_f_4_l59_59669


namespace hyperbola_equation_through_point_l59_59222

theorem hyperbola_equation_through_point
  (hyp_passes_through : ∀ (x y : ℝ), (x, y) = (1, 1) → ∃ (a b t : ℝ), (y^2 / a^2 - x^2 / b^2 = t))
  (asymptotes : ∀ (x y : ℝ), (y / x = Real.sqrt 2 ∨ y / x = -Real.sqrt 2) → ∃ (a b t : ℝ), (a = b * Real.sqrt 2)) :
  ∃ (a b t : ℝ), (2 * (1:ℝ)^2 - (1:ℝ)^2 = 1) :=
by
  sorry

end hyperbola_equation_through_point_l59_59222


namespace most_irregular_acute_triangle_l59_59778

theorem most_irregular_acute_triangle :
  ∃ (α β γ : ℝ), α ≤ β ∧ β ≤ γ ∧ γ ≤ (90:ℝ) ∧ 
  ((β - α ≤ 15) ∧ (γ - β ≤ 15) ∧ (90 - γ ≤ 15)) ∧
  (α + β + γ = 180) ∧ 
  (α = 45 ∧ β = 60 ∧ γ = 75) := sorry

end most_irregular_acute_triangle_l59_59778


namespace octagon_has_20_diagonals_l59_59639

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l59_59639


namespace neg_p_necessary_not_sufficient_neg_q_l59_59371

def p (x : ℝ) := abs x < 1
def q (x : ℝ) := x^2 + x - 6 < 0

theorem neg_p_necessary_not_sufficient_neg_q :
  (¬ (∃ x, p x)) → (¬ (∃ x, q x)) ∧ ¬ ((¬ (∃ x, p x)) → (¬ (∃ x, q x))) :=
by
  sorry

end neg_p_necessary_not_sufficient_neg_q_l59_59371


namespace part1_part2_l59_59977

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (x^2 - 3*x + 2)

theorem part1 (h₁ : a = 1) : ∀ x : ℝ, (0 < x ∧ x < 1/2 ∨ x > 1 → f x 1 > f (1 / 2) 1) ∧ (1 / 2 < x ∧ x < 1 → f x 1 < f 1 1) :=
sorry

theorem part2 (h₂ : ∀ x > 1, f x a ≥ 0) : a ≤ 1 :=
sorry

end part1_part2_l59_59977


namespace solve_for_d_l59_59207

variable (n c b d : ℚ)  -- Alternatively, specify the types if they are required to be specific
variable (H : n = d * c * b / (c - d))

theorem solve_for_d :
  d = n * c / (c * b + n) :=
by
  sorry

end solve_for_d_l59_59207


namespace line_through_point_with_angle_l59_59719

theorem line_through_point_with_angle :
  ∀ (k : ℝ) (x y : ℝ),
    let P := (-1 : ℝ, real.sqrt 3)
    let θ := real.pi / 6
    let L₁ := (λ x : ℝ, real.sqrt 3 * x - 1)
    let L₀ := (λ x : ℝ, k * x + real.sqrt 3)
    ((∀ (x y : ℝ), y - real.sqrt 3 = k * (x + 1)) ∨
     (Λx : ℝ, x = -1)) :=
  sorry

end line_through_point_with_angle_l59_59719


namespace length_of_bridge_is_correct_l59_59165

noncomputable def length_of_inclined_bridge (initial_speed : ℕ) (time : ℕ) (acceleration : ℕ) : ℚ :=
  (1 / 60) * (time * initial_speed + (time * (time - 1)) / 2)

theorem length_of_bridge_is_correct : 
  length_of_inclined_bridge 10 18 1 = 5.55 := 
by
  sorry

end length_of_bridge_is_correct_l59_59165


namespace gcd_a2_13a_36_a_6_eq_6_l59_59629

namespace GCDProblem

variable (a : ℕ)
variable (h : ∃ k, a = 1632 * k)

theorem gcd_a2_13a_36_a_6_eq_6 (ha : ∃ k : ℕ, a = 1632 * k) : 
  Int.gcd (a^2 + 13 * a + 36 : Int) (a + 6 : Int) = 6 := by
  sorry

end GCDProblem

end gcd_a2_13a_36_a_6_eq_6_l59_59629


namespace circle_center_tangent_lines_l59_59761

theorem circle_center_tangent_lines 
    (center : ℝ × ℝ)
    (h1 : 3 * center.1 + 4 * center.2 = 10)
    (h2 : center.1 = 3 * center.2) : 
    center = (30 / 13, 10 / 13) := 
by {
  sorry
}

end circle_center_tangent_lines_l59_59761


namespace rubies_in_treasure_l59_59172

theorem rubies_in_treasure (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) : 
  total_gems - diamonds = 5110 := by
  sorry

end rubies_in_treasure_l59_59172


namespace cupcakes_sold_l59_59206

theorem cupcakes_sold (initial additional final sold : ℕ) (h1 : initial = 14) (h2 : additional = 17) (h3 : final = 25) :
  initial + additional - final = sold :=
by
  sorry

end cupcakes_sold_l59_59206


namespace find_pairs_l59_59455

theorem find_pairs (a b : ℕ) (h1 : a + b = 60) (h2 : Nat.lcm a b = 72) : (a = 24 ∧ b = 36) ∨ (a = 36 ∧ b = 24) := 
sorry

end find_pairs_l59_59455


namespace age_problem_l59_59601

theorem age_problem (A B C D E : ℕ)
  (h1 : A = B + 2)
  (h2 : B = 2 * C)
  (h3 : D = C / 2)
  (h4 : E = D - 3)
  (h5 : A + B + C + D + E = 52) : B = 16 :=
by
  sorry

end age_problem_l59_59601


namespace last_student_calls_out_l59_59757

-- Define the transformation rules as a function
def next_student (n : ℕ) : ℕ :=
  if n < 10 then n + 8 else (n % 10) + 7

-- Define the sequence generation function
noncomputable def student_number : ℕ → ℕ
| 0       => 1  -- the 1st student starts with number 1
| (n + 1) => next_student (student_number n)

-- The main theorem to prove
theorem last_student_calls_out (n : ℕ) : student_number 2013 = 12 :=
sorry

end last_student_calls_out_l59_59757


namespace solve_for_x_l59_59686

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l59_59686


namespace inequality_solution_l59_59869

theorem inequality_solution (x : ℝ) :
  (∃ x, 2 < x ∧ x < 3) ↔ ∃ x, (x-2)*(x-3)/(x^2 + 1) < 0 := by
  sorry

end inequality_solution_l59_59869


namespace find_satisfying_pairs_l59_59329

theorem find_satisfying_pairs (n p : ℕ) (prime_p : Nat.Prime p) :
  n ≤ 2 * p ∧ (p - 1)^n + 1 ≡ 0 [MOD n^2] →
  (n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by sorry

end find_satisfying_pairs_l59_59329


namespace problem1_problem2_l59_59435

variable {n : ℕ}
variable {a b : ℝ}

-- Part 1
theorem problem1 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(2 * n^2 / (n + 2) - n * a) - b| < ε) :
  a = 2 ∧ b = 4 := sorry

-- Part 2
theorem problem2 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(3^n / (3^(n + 1) + (a + 1)^n) - 1/3)| < ε) :
  -4 < a ∧ a < 2 := sorry

end problem1_problem2_l59_59435


namespace arcsin_one_eq_pi_div_two_l59_59938

noncomputable def arcsin : ℝ → ℝ := sorry -- Define arcsin function

theorem arcsin_one_eq_pi_div_two : arcsin 1 = Real.pi / 2 := sorry

end arcsin_one_eq_pi_div_two_l59_59938


namespace find_second_game_points_l59_59618

-- Define Clayton's points for respective games
def first_game_points := 10
def third_game_points := 6

-- Define the points in the second game as P
variable (P : ℕ)

-- Define the points in the fourth game based on the average of first three games
def fourth_game_points := (first_game_points + P + third_game_points) / 3

-- Define the total points over four games
def total_points := first_game_points + P + third_game_points + fourth_game_points

-- Based on the total points, prove P = 14
theorem find_second_game_points (P : ℕ) (h : total_points P = 40) : P = 14 :=
  by
    sorry

end find_second_game_points_l59_59618


namespace Maggie_apples_l59_59839

-- Definition of our problem conditions
def K : ℕ := 28 -- Kelsey's apples
def L : ℕ := 22 -- Layla's apples
def avg : ℕ := 30 -- The average number of apples picked

-- Main statement to prove Maggie's apples
theorem Maggie_apples : (A : ℕ) → (A + K + L) / 3 = avg → A = 40 := by
  intros A h
  -- sorry is added to skip the proof since it's not required here.
  sorry

end Maggie_apples_l59_59839


namespace train_speed_l59_59171

theorem train_speed
    (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
    (h_train : length_train = 250)
    (h_platform : length_platform = 250.04)
    (h_time : time_seconds = 25) :
    (length_train + length_platform) / time_seconds * 3.6 = 72.006 :=
by sorry

end train_speed_l59_59171


namespace largest_n_factorable_l59_59470

theorem largest_n_factorable :
  ∃ n, (∀ A B : ℤ, 6x^2 + n • x + 72 = (6 • x + A) * (x + B)) ∧ 
    (∀ x', 6x' + A = 0 ∨ x' + B = 0) ∧ 
    n = 433 := 
sorry

end largest_n_factorable_l59_59470


namespace inclination_angle_of_focal_chord_l59_59806

theorem inclination_angle_of_focal_chord
  (p : ℝ)
  (h_parabola : ∀ x y : ℝ, y^2 = 2 * p * x → True)
  (h_focal_chord_length : ∀ A B : ℝ, |A - B| = 8 * p → True) :
  ∃ θ : ℝ, (θ = π / 6 ∨ θ = 5 * π / 6) :=
by
  sorry

end inclination_angle_of_focal_chord_l59_59806


namespace employees_without_any_benefit_l59_59126

def employees_total : ℕ := 480
def employees_salary_increase : ℕ := 48
def employees_travel_increase : ℕ := 96
def employees_both_increases : ℕ := 24
def employees_vacation_days : ℕ := 72

theorem employees_without_any_benefit : (employees_total - ((employees_salary_increase + employees_travel_increase + employees_vacation_days) - employees_both_increases)) = 288 :=
by
  sorry

end employees_without_any_benefit_l59_59126


namespace sequence_inequality_l59_59229

theorem sequence_inequality (a : ℕ → ℕ) 
  (h_nonneg : ∀ n, 0 ≤ a n)
  (h_additive : ∀ m n, a (n + m) ≤ a n + a m) 
  (N n : ℕ) 
  (h_N_ge_n : N ≥ n) : 
  a n + a N ≤ n * a 1 + N / n * a n :=
sorry

end sequence_inequality_l59_59229


namespace trapezoid_bases_12_and_16_l59_59136

theorem trapezoid_bases_12_and_16 :
  ∀ (h R : ℝ) (a b : ℝ),
    (R = 10) →
    (h = (a + b) / 2) →
    (∀ k m, ((k = 3/7 * h) ∧ (m = 4/7 * h) ∧ (R^2 = k^2 + (a/2)^2) ∧ (R^2 = m^2 + (b/2)^2))) →
    (a = 12) ∧ (b = 16) :=
by
  intros h R a b hR hMid eqns
  sorry

end trapezoid_bases_12_and_16_l59_59136


namespace recurring_decimal_fraction_l59_59016

theorem recurring_decimal_fraction (h54 : (0.54 : ℝ) = 54 / 99) (h18 : (0.18 : ℝ) = 18 / 99) :
    (0.54 / 0.18 : ℝ) = 3 := 
by
  sorry

end recurring_decimal_fraction_l59_59016


namespace number_of_students_l59_59763

theorem number_of_students (n : ℕ) (h1 : n < 40) (h2 : n % 7 = 3) (h3 : n % 6 = 1) : n = 31 := 
by
  sorry

end number_of_students_l59_59763


namespace ratio_fifth_terms_l59_59003

variable (a_n b_n S_n T_n : ℕ → ℚ)

-- Conditions
variable (h : ∀ n, S_n n / T_n n = (9 * n + 2) / (n + 7))

-- Define the 5th term
def a_5 (S_n : ℕ → ℚ) : ℚ := S_n 9 / 9
def b_5 (T_n : ℕ → ℚ) : ℚ := T_n 9 / 9

-- Prove that the ratio of the 5th terms is 83 / 16
theorem ratio_fifth_terms :
  (a_5 S_n) / (b_5 T_n) = 83 / 16 :=
by
  sorry

end ratio_fifth_terms_l59_59003


namespace polynomial_nonnegative_l59_59357

theorem polynomial_nonnegative (p q : ℝ) (h : q > p^2) :
  ∀ x : ℝ, x^2 + 2 * p * x + q ≥ 0 :=
by
  intro x
  have h2 : x^2 + 2 * p * x + q = (x + p)^2 + (q - p^2) := by sorry
  have h3 : (x + p)^2 ≥ 0 := by sorry
  have h4 : q - p^2 > 0 := h
  have h5 : (x + p)^2 + (q - p^2) ≥ 0 + 0 := by sorry
  linarith

end polynomial_nonnegative_l59_59357


namespace squares_equal_l59_59407

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l59_59407


namespace ali_total_money_l59_59615

-- Definitions based on conditions
def bills_of_5_dollars : ℕ := 7
def bills_of_10_dollars : ℕ := 1
def value_of_5_dollar_bill : ℕ := 5
def value_of_10_dollar_bill : ℕ := 10

-- Prove that Ali's total amount of money is $45
theorem ali_total_money : (bills_of_5_dollars * value_of_5_dollar_bill) + (bills_of_10_dollars * value_of_10_dollar_bill) = 45 := 
by
  sorry

end ali_total_money_l59_59615


namespace value_of_x_l59_59099

theorem value_of_x (x : ℝ) :
  (x^2 - 1 + (x - 1) * I = 0 ∨ x^2 - 1 = 0 ∧ x - 1 ≠ 0) → x = -1 :=
by
  sorry

end value_of_x_l59_59099


namespace ella_last_roll_probability_l59_59241

theorem ella_last_roll_probability :
  let p := ((5/6)^10 * (1/6)) in
  abs (p - 0.027) < 0.001 := sorry

end ella_last_roll_probability_l59_59241


namespace rhombus_area_correct_l59_59822

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct (x : ℝ) (h1 : rhombus_area 7 (abs (8 - x)) = 56) 
    (h2 : x ≠ 8) : x = -8 ∨ x = 24 :=
by
  sorry

end rhombus_area_correct_l59_59822


namespace picture_area_l59_59352

theorem picture_area (x y : ℤ) (hx : 1 < x) (hy : 1 < y) (h : (x + 2) * (y + 4) = 45) : x * y = 15 := by
  sorry

end picture_area_l59_59352


namespace minimum_value_l59_59498

theorem minimum_value :
  ∀ (m n : ℝ), m > 0 → n > 0 → (3 * m + n = 1) → (3 / m + 1 / n) ≥ 16 :=
by
  intros m n hm hn hline
  sorry

end minimum_value_l59_59498


namespace problem_l59_59079

noncomputable def a : Real := 9^(1/3)
noncomputable def b : Real := 3^(2/5)
noncomputable def c : Real := 4^(1/5)

theorem problem (a := 9^(1/3)) (b := 3^(2/5)) (c := 4^(1/5)) : a > b ∧ b > c := by
  sorry

end problem_l59_59079


namespace num_convex_numbers_without_repeats_l59_59805

def is_convex_number (a b c : ℕ) : Prop :=
  a < b ∧ b > c

def is_valid_digit (n : ℕ) : Prop :=
  0 ≤ n ∧ n < 10

def distinct_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem num_convex_numbers_without_repeats : 
  (∃ (numbers : Finset (ℕ × ℕ × ℕ)), 
    (∀ a b c, (a, b, c) ∈ numbers -> is_convex_number a b c ∧ is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ distinct_digits a b c) ∧
    numbers.card = 204) :=
sorry

end num_convex_numbers_without_repeats_l59_59805


namespace part1_part2_l59_59496

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Statement for part (1)
theorem part1 (m : ℝ) : (m > -2) → (∀ x : ℝ, m + f x > 0) :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : (m > 2) ↔ (∀ x : ℝ, m - f x > 0) :=
sorry

end part1_part2_l59_59496


namespace area_of_triangle_l59_59249

theorem area_of_triangle (a c : ℝ) (A : ℝ) (h_a : a = 2) (h_c : c = 2 * Real.sqrt 3) (h_A : A = Real.pi / 6) :
  ∃ (area : ℝ), area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l59_59249


namespace abs_sum_inequality_l59_59901

theorem abs_sum_inequality (x : ℝ) : (|x - 2| + |x + 3| < 7) ↔ (-6 < x ∧ x < 3) :=
sorry

end abs_sum_inequality_l59_59901


namespace smallest_n_l59_59956

-- Define the costs of candies
def cost_purple := 24
def cost_yellow := 30

-- Define the number of candies Lara can buy
def pieces_red := 10
def pieces_green := 16
def pieces_blue := 18
def pieces_yellow := 22

-- Define the total money Lara has equivalently expressed by buying candies
def lara_total_money (n : ℕ) := n * cost_purple

-- Prove the smallest value of n that satisfies the conditions stated
theorem smallest_n : ∀ n : ℕ, 
  (lara_total_money n = 10 * pieces_red * cost_purple) ∧
  (lara_total_money n = 16 * pieces_green * cost_purple) ∧
  (lara_total_money n = 18 * pieces_blue * cost_purple) ∧
  (lara_total_money n = pieces_yellow * cost_yellow) → 
  n = 30 :=
by
  intro
  sorry

end smallest_n_l59_59956


namespace clara_hardcover_books_l59_59187

-- Define the variables and conditions
variables (h p : ℕ)

-- Conditions based on the problem statement
def volumes_total : Prop := h + p = 12
def total_cost (total : ℕ) : Prop := 28 * h + 18 * p = total

-- The theorem to prove
theorem clara_hardcover_books (h p : ℕ) (H1 : volumes_total h p) (H2 : total_cost h p 270) : h = 6 :=
by
  sorry

end clara_hardcover_books_l59_59187


namespace product_of_numbers_l59_59288

theorem product_of_numbers (x y z : ℤ) 
  (h1 : x + y + z = 30) 
  (h2 : x = 3 * ((y + z) - 2))
  (h3 : y = 4 * z - 1) : 
  x * y * z = 294 := 
  sorry

end product_of_numbers_l59_59288


namespace octagon_has_20_diagonals_l59_59653

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l59_59653


namespace function_increasing_l59_59690

noncomputable def f (a x : ℝ) := x^2 + a * x + 1 / x

theorem function_increasing (a : ℝ) :
  (∀ x, (1 / 3) < x → 0 ≤ (2 * x + a - 1 / x^2)) → a ≥ 25 / 3 :=
by
  sorry

end function_increasing_l59_59690


namespace squares_equal_l59_59409

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l59_59409


namespace midpoint_of_segment_l59_59887

theorem midpoint_of_segment :
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint = (4, 1) :=
by
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint = (4, 1)
  sorry

end midpoint_of_segment_l59_59887


namespace tan_of_cos_first_quadrant_l59_59219

-- Define the angle α in the first quadrant and its cosine value
variable (α : ℝ) (h1 : 0 < α ∧ α < π/2) (hcos : Real.cos α = 2 / 3)

-- State the theorem
theorem tan_of_cos_first_quadrant : Real.tan α = Real.sqrt 5 / 2 := 
by
  sorry

end tan_of_cos_first_quadrant_l59_59219


namespace octagon_diagonals_l59_59661

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l59_59661


namespace proof_problem_l59_59932

-- Definitions of points and vectors
def C : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (3, 4)
def N : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (0, 1)

-- Definition of vector operations
def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2)

-- Vectors needed
def AC : ℝ × ℝ := vector_sub C A
def AM : ℝ × ℝ := vector_sub M A
def AN : ℝ × ℝ := vector_sub N A

-- The Lean proof statement
theorem proof_problem :
  (∃ (x y : ℝ), AC = (x * AM.1 + y * AN.1, x * AM.2 + y * AN.2) ∧
     (x, y) = (2 / 3, 1 / 2)) ∧
  (9 * (2 / 3:ℝ) ^ 2 + 16 * (1 / 2:ℝ) ^ 2 = 8) :=
by
  sorry

end proof_problem_l59_59932


namespace distance_between_foci_l59_59075

-- Defining the given ellipse equation 
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 150 * x + 4 * y^2 + 8 * y + 9 = 0

-- Statement to prove the distance between the foci
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 46.2 := 
sorry

end distance_between_foci_l59_59075


namespace sum_first_5n_l59_59993

theorem sum_first_5n (n : ℕ) (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 210) : 
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end sum_first_5n_l59_59993


namespace base_8_subtraction_l59_59785

def subtract_in_base_8 (a b : ℕ) : ℕ := 
  -- Implementing the base 8 subtraction
  sorry

theorem base_8_subtraction : subtract_in_base_8 0o652 0o274 = 0o356 :=
by 
  -- Faking the proof to ensure it can compile.
  sorry

end base_8_subtraction_l59_59785


namespace no_negative_roots_but_at_least_one_positive_root_l59_59188

def f (x : ℝ) : ℝ := x^6 - 3 * x^5 - 6 * x^3 - x + 8

theorem no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → f x ≠ 0) ∧ (∃ x : ℝ, x > 0 ∧ f x = 0) :=
by {
  sorry
}

end no_negative_roots_but_at_least_one_positive_root_l59_59188


namespace largest_n_factorable_l59_59471

theorem largest_n_factorable :
  ∃ n, (∀ A B : ℤ, 6x^2 + n • x + 72 = (6 • x + A) * (x + B)) ∧ 
    (∀ x', 6x' + A = 0 ∨ x' + B = 0) ∧ 
    n = 433 := 
sorry

end largest_n_factorable_l59_59471


namespace find_second_cert_interest_rate_l59_59315

theorem find_second_cert_interest_rate
  (initial_investment : ℝ := 12000)
  (first_term_months : ℕ := 8)
  (first_interest_rate : ℝ := 8 / 100)
  (second_term_months : ℕ := 10)
  (final_amount : ℝ := 13058.40)
  : ∃ s : ℝ, (s = 3.984) := sorry

end find_second_cert_interest_rate_l59_59315


namespace total_number_of_ways_is_144_l59_59379

def count_ways_to_place_letters_on_grid : Nat :=
  16 * 9

theorem total_number_of_ways_is_144 :
  count_ways_to_place_letters_on_grid = 144 :=
  by
    sorry

end total_number_of_ways_is_144_l59_59379


namespace minimum_a_plus_3b_l59_59982

-- Define the conditions
variables (a b : ℝ)
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_eq : a + 3 * b = 1 / a + 3 / b

-- State the theorem
theorem minimum_a_plus_3b (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a + 3 * b = 1 / a + 3 / b) : 
  a + 3 * b ≥ 4 :=
sorry

end minimum_a_plus_3b_l59_59982


namespace problem_1_problem_2_l59_59436

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x - 2)

theorem problem_1 (a b : ℝ) (h1 : f a b 3 - 3 + 12 = 0) (h2 : f a b 4 - 4 + 12 = 0) :
  f a b x = (2 - x) / (x - 2) := sorry

theorem problem_2 (k : ℝ) (h : k > 1) :
  ∀ x, f (-1) 2 x < k ↔ (if 1 < k ∧ k < 2 then (1 < x ∧ x < k) ∨ (2 < x) 
                         else if k = 2 then 1 < x ∧ x ≠ 2 
                         else (1 < x ∧ x < 2) ∨ (k < x)) := sorry

-- Function definition for clarity
noncomputable def f_spec (x : ℝ) : ℝ := (2 - x) / (x - 2)

end problem_1_problem_2_l59_59436


namespace total_pages_in_book_l59_59843

theorem total_pages_in_book (x : ℕ) : 
  (x - (x / 6 + 8) - ((5 * x / 6 - 8) / 5 + 10) - ((4 * x / 6 - 18) / 4 + 12) = 72) → 
  x = 195 :=
by
  sorry

end total_pages_in_book_l59_59843


namespace maxOccursAt2_l59_59809

def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

theorem maxOccursAt2 {m : ℝ} :
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ f m) ∧ 0 ≤ m ∧ m ≤ 2 → (0 < m ∧ m ≤ 2) :=
sorry

end maxOccursAt2_l59_59809


namespace inscribed_square_properties_l59_59422

theorem inscribed_square_properties (r : ℝ) (s : ℝ) (d : ℝ) (A_circle : ℝ) (A_square : ℝ) (total_diagonals : ℝ) (hA_circle : A_circle = 324 * Real.pi) (hr : r = Real.sqrt 324) (hd : d = 2 * r) (hs : s = d / Real.sqrt 2) (hA_square : A_square = s ^ 2) (htotal_diagonals : total_diagonals = 2 * d) :
  A_square = 648 ∧ total_diagonals = 72 :=
by sorry

end inscribed_square_properties_l59_59422


namespace guacamole_servings_l59_59791

theorem guacamole_servings (avocados_per_serving : ℕ) (initial_avocados : ℕ) (additional_avocados : ℕ) (total_avocados : ℕ := initial_avocados + additional_avocados) (servings : ℕ := total_avocados / avocados_per_serving) :
  avocados_per_serving = 3 →
  initial_avocados = 5 →
  additional_avocados = 4 →
  servings = 3 :=
by
  intros h1 h2 h3
  unfold servings total_avocados
  rw [h1, h2, h3]
  norm_num
  rfl
  sorry

end guacamole_servings_l59_59791


namespace floor_sqrt_23_squared_l59_59454

theorem floor_sqrt_23_squared : (Int.floor (Real.sqrt 23))^2 = 16 := 
by
  -- conditions
  have h1 : 4^2 = 16 := by norm_num
  have h2 : 5^2 = 25 := by norm_num
  have h3 : 16 < 23 := by norm_num
  have h4 : 23 < 25 := by norm_num
  -- statement (goal)
  sorry

end floor_sqrt_23_squared_l59_59454


namespace obtuse_angle_of_parallel_vectors_l59_59485

noncomputable def is_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem obtuse_angle_of_parallel_vectors (θ : ℝ) :
  let a := (2, 1 - Real.cos θ)
  let b := (1 + Real.cos θ, 1 / 4)
  is_parallel a b → 90 < θ ∧ θ < 180 → θ = 135 :=
by
  intro ha hb
  sorry

end obtuse_angle_of_parallel_vectors_l59_59485


namespace plane_equation_correct_l59_59894

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p q : Point3D) : Point3D :=
  { x := p.x - q.x, y := p.y - q.y, z := p.z - q.z }

def plane_eq (n : Point3D) (A : Point3D) : Point3D → ℝ :=
  fun P => n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

def is_perpendicular_plane (A B C : Point3D) (D : Point3D → ℝ) : Prop :=
  let BC := vector_sub C B
  D = plane_eq BC A

theorem plane_equation_correct :
  let A := { x := 7, y := -5, z := 1 }
  let B := { x := 5, y := -1, z := -3 }
  let C := { x := 3, y := 0, z := -4 }
  is_perpendicular_plane A B C (fun P => -2 * P.x + P.y - P.z + 20) :=
by
  sorry

end plane_equation_correct_l59_59894


namespace number_of_chords_with_integer_length_l59_59128

theorem number_of_chords_with_integer_length 
(centerP_dist radius : ℝ) 
(h1 : centerP_dist = 12) 
(h2 : radius = 20) : 
  ∃ n : ℕ, n = 9 := 
by 
  sorry

end number_of_chords_with_integer_length_l59_59128


namespace eight_hash_four_eq_ten_l59_59863

def operation (a b : ℚ) : ℚ := a + a / b

theorem eight_hash_four_eq_ten : operation 8 4 = 10 :=
by
  sorry

end eight_hash_four_eq_ten_l59_59863


namespace find_values_of_A_l59_59586

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_values_of_A (A B C : ℕ) :
  sum_of_digits A = B ∧
  sum_of_digits B = C ∧
  A + B + C = 60 →
  (A = 44 ∨ A = 50 ∨ A = 47) :=
by
  sorry

end find_values_of_A_l59_59586


namespace find_original_integer_l59_59341

theorem find_original_integer (a b c d : ℕ) 
    (h1 : (b + c + d) / 3 + 10 = 37) 
    (h2 : (a + c + d) / 3 + 10 = 31) 
    (h3 : (a + b + d) / 3 + 10 = 25) 
    (h4 : (a + b + c) / 3 + 10 = 19) : 
    d = 45 := 
    sorry

end find_original_integer_l59_59341


namespace fish_fishermen_problem_l59_59732

theorem fish_fishermen_problem (h: ℕ) (r: ℕ) (w_h: ℕ) (w_r: ℕ) (claimed_weight: ℕ) (total_real_weight: ℕ) 
  (total_fishermen: ℕ) :
  -- conditions
  (claimed_weight = 60) →
  (total_real_weight = 120) →
  (total_fishermen = 10) →
  (w_h = 30) →
  (w_r < 60 / 7) →
  (h + r = total_fishermen) →
  (2 * w_h * h + r * claimed_weight = claimed_weight * total_fishermen) →
  -- prove the number of regular fishermen
  (r = 7 ∨ r = 8) :=
sorry

end fish_fishermen_problem_l59_59732


namespace inequality_solution_l59_59585

theorem inequality_solution (x : ℝ) :
  x + 1 ≥ -3 ∧ -2 * (x + 3) > 0 ↔ -4 ≤ x ∧ x < -3 :=
by sorry

end inequality_solution_l59_59585


namespace solution_set_of_inequality_g_geq_2_l59_59346

-- Definition of the function f
def f (x a : ℝ) := |x - a|

-- Definition of the function g
def g (x a : ℝ) := f x a + f (x + 2) a

-- Proof Problem I
theorem solution_set_of_inequality (a : ℝ) (x : ℝ) :
  a = -1 → (f x a ≥ 4 - |2 * x - 1|) ↔ (x ≤ -4/3 ∨ x ≥ 4/3) :=
by sorry

-- Proof Problem II
theorem g_geq_2 (a : ℝ) (x : ℝ) :
  (∀ x, f x a ≤ 1 → (0 ≤ x ∧ x ≤ 2)) → a = 1 → g x a ≥ 2 :=
by sorry

end solution_set_of_inequality_g_geq_2_l59_59346


namespace find_a_l59_59664

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 :=
by
  sorry

end find_a_l59_59664


namespace ratio_of_books_l59_59698

theorem ratio_of_books (longest_pages : ℕ) (middle_pages : ℕ) (shortest_pages : ℕ) :
  longest_pages = 396 ∧ middle_pages = 297 ∧ shortest_pages = longest_pages / 4 →
  (middle_pages / shortest_pages = 3) :=
by
  intros h
  obtain ⟨h_longest, h_middle, h_shortest⟩ := h
  sorry

end ratio_of_books_l59_59698


namespace license_plate_count_l59_59503

theorem license_plate_count : (26^3 * 5 * 5 * 4) = 1757600 := 
by 
  sorry

end license_plate_count_l59_59503


namespace find_a_plus_b_l59_59414

theorem find_a_plus_b (a b : ℝ) (h_sum : 2 * a = -6) (h_prod : a^2 - b = 1) : a + b = 5 :=
by {
  -- Proof would go here; we assume the theorem holds true.
  sorry
}

end find_a_plus_b_l59_59414


namespace velocity_division_l59_59106

/--
Given a trapezoidal velocity-time graph with bases V and U,
determine the velocity W that divides the area under the graph into
two regions such that the areas are in the ratio 1:k.
-/
theorem velocity_division (V U k : ℝ) (h_k : k ≠ -1) : 
  ∃ W : ℝ, W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end velocity_division_l59_59106


namespace coefficient_a9_l59_59360

theorem coefficient_a9 (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℤ) :
  (x^2 + x^10 = a0 + a1 * (x + 1) + a2 * (x + 1)^2 + a3 * (x + 1)^3 +
   a4 * (x + 1)^4 + a5 * (x + 1)^5 + a6 * (x + 1)^6 + a7 * (x + 1)^7 +
   a8 * (x + 1)^8 + a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a10 = 1 →
  a9 = -10 :=
by
  sorry

end coefficient_a9_l59_59360


namespace octagon_has_20_diagonals_l59_59638

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l59_59638


namespace cell_division_after_three_hours_l59_59904

theorem cell_division_after_three_hours : (2 ^ 6) = 64 := by
  sorry

end cell_division_after_three_hours_l59_59904


namespace track_length_l59_59449

theorem track_length
  (x : ℕ)
  (run1_Brenda : x / 2 + 80 = a)
  (run2_Sally : x / 2 + 100 = b)
  (run1_ratio : 80 / (x / 2 - 80) = c)
  (run2_ratio : (x / 2 - 100) / (x / 2 + 100) = c)
  : x = 520 :=
by sorry

end track_length_l59_59449


namespace georgie_guacamole_servings_l59_59790

-- Define the conditions
def avocados_needed_per_serving : Nat := 3
def initial_avocados : Nat := 5
def additional_avocados : Nat := 4

-- State the target number of servings Georgie can make
def total_avocados := initial_avocados + additional_avocados
def guacamole_servings := total_avocados / avocados_needed_per_serving

-- Lean 4 statement asserting the number of servings equals 3
theorem georgie_guacamole_servings : guacamole_servings = 3 := by
  sorry

end georgie_guacamole_servings_l59_59790


namespace gcd_of_840_and_1764_l59_59722

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_of_840_and_1764_l59_59722


namespace art_club_activity_l59_59693

theorem art_club_activity (n p s b : ℕ) (h1 : n = 150) (h2 : p = 80) (h3 : s = 60) (h4 : b = 20) :
  (n - (p + s - b) = 30) :=
by
  sorry

end art_club_activity_l59_59693


namespace number_of_diagonals_in_octagon_l59_59646

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l59_59646


namespace find_X_d_minus_Y_d_l59_59850

def digits_in_base_d (X Y d : ℕ) : Prop :=
  2 * d * X + X + Y = d^2 + 8 * d + 2 

theorem find_X_d_minus_Y_d (d X Y : ℕ) (h1 : digits_in_base_d X Y d) (h2 : d > 8) : X - Y = d - 8 :=
by 
  sorry

end find_X_d_minus_Y_d_l59_59850


namespace hyperbola_center_l59_59456

theorem hyperbola_center :
  ∃ (c : ℝ × ℝ), c = (3, 5) ∧
  (9 * (x - c.1)^2 - 36 * (y - c.2)^2 - (1244 - 243 - 1001) = 0) :=
sorry

end hyperbola_center_l59_59456


namespace find_a_b_sum_l59_59412

theorem find_a_b_sum
  (a b : ℝ)
  (h1 : 2 * a = -6)
  (h2 : a ^ 2 - b = 1) :
  a + b = 5 :=
by
  sorry

end find_a_b_sum_l59_59412


namespace sum_of_numbers_l59_59935

theorem sum_of_numbers : 145 + 33 + 29 + 13 = 220 :=
by
  sorry

end sum_of_numbers_l59_59935


namespace arithmetic_sequence_terms_sum_l59_59246

theorem arithmetic_sequence_terms_sum
  (a : ℕ → ℝ)
  (h₁ : ∀ n, a (n+1) = a n + d)
  (h₂ : a 2 = 1 - a 1)
  (h₃ : a 4 = 9 - a 3)
  (h₄ : ∀ n, a n > 0):
  a 4 + a 5 = 27 :=
sorry

end arithmetic_sequence_terms_sum_l59_59246


namespace gcd_poly_l59_59970

-- Defining the conditions as stated in part a:
def is_even_multiple_of_1171 (b : ℤ) : Prop :=
  ∃ k : ℤ, b = 1171 * k * 2

-- Stating the main theorem based on the conditions and required proof in part c:
theorem gcd_poly (b : ℤ) (h : is_even_multiple_of_1171 b) : Int.gcd (3 * b ^ 2 + 47 * b + 79) (b + 17) = 1 := by
  sorry

end gcd_poly_l59_59970


namespace solve_system_l59_59232

-- Define the conditions
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 8) ∧ (2 * x - y = 7)

-- Define the proof problem statement
theorem solve_system : 
  system_of_equations 5 3 :=
by
  -- Proof will be filled in here
  sorry

end solve_system_l59_59232


namespace max_n_for_factored_polynomial_l59_59467

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l59_59467


namespace power_function_properties_l59_59223

theorem power_function_properties (m : ℤ) :
  (m^2 - 2 * m - 2 ≠ 0) ∧ (m^2 + 4 * m < 0) ∧ (m^2 + 4 * m % 2 = 1) → m = -1 := by
  intro h
  sorry

end power_function_properties_l59_59223


namespace simplify_expression_l59_59565

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l59_59565


namespace glucose_in_mixed_solution_l59_59591

def concentration1 := 20 / 100  -- concentration of first solution in grams per cubic centimeter
def concentration2 := 30 / 100  -- concentration of second solution in grams per cubic centimeter
def volume1 := 80               -- volume of first solution in cubic centimeters
def volume2 := 50               -- volume of second solution in cubic centimeters

theorem glucose_in_mixed_solution :
  (concentration1 * volume1) + (concentration2 * volume2) = 31 := by
  sorry

end glucose_in_mixed_solution_l59_59591


namespace paper_clips_in_2_cases_l59_59312

variable (c b : ℕ)

theorem paper_clips_in_2_cases : 2 * (c * b) * 600 = (2 * c * b * 600) := by
  sorry

end paper_clips_in_2_cases_l59_59312


namespace rachel_hw_diff_l59_59272

-- Definitions based on the conditions of the problem
def math_hw_pages := 15
def reading_hw_pages := 6

-- The statement we need to prove, including the conditions
theorem rachel_hw_diff : 
  math_hw_pages - reading_hw_pages = 9 := 
by
  sorry

end rachel_hw_diff_l59_59272


namespace pieces_per_pan_of_brownies_l59_59735

theorem pieces_per_pan_of_brownies (total_guests guests_ala_mode additional_guests total_scoops_per_tub total_tubs_eaten total_pans guests_per_pan second_pan_percentage consumed_pans : ℝ)
    (h1 : total_guests = guests_ala_mode + additional_guests)
    (h2 : total_scoops_per_tub * total_tubs_eaten = guests_ala_mode * 2)
    (h3 : consumed_pans = 1 + second_pan_percentage)
    (h4 : second_pan_percentage = 0.75)
    (h5 : total_guests = guests_per_pan * consumed_pans)
    (h6 : guests_per_pan = 28)
    : total_guests / consumed_pans = 16 :=
by
  have h7 : total_scoops_per_tub * total_tubs_eaten = 48 := by sorry
  have h8 : guests_ala_mode = 24 := by sorry
  have h9 : total_guests = 28 := by sorry
  have h10 : consumed_pans = 1.75 := by sorry
  have h11 : guests_per_pan = 28 := by sorry
  sorry


end pieces_per_pan_of_brownies_l59_59735


namespace range_of_a_for_circle_l59_59857

theorem range_of_a_for_circle (a : ℝ) : 
  -2 < a ∧ a < 2/3 ↔ 
  ∃ (x y : ℝ), (x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1) = 0 :=
sorry

end range_of_a_for_circle_l59_59857


namespace similar_triangles_legs_sum_l59_59592

theorem similar_triangles_legs_sum (a b : ℕ) (h1 : a * b = 18) (h2 : a^2 + b^2 = 25) (bigger_area : ℕ) (smaller_area : ℕ) (hypotenuse : ℕ) 
  (h_similar : bigger_area = 225) 
  (h_smaller_area : smaller_area = 9) 
  (h_hypotenuse : hypotenuse = 5) 
  (h_non_3_4_5 : ¬ (a = 3 ∧ b = 4 ∨ a = 4 ∧ b = 3)) : 
  5 * (a + b) = 45 := 
by sorry

end similar_triangles_legs_sum_l59_59592


namespace find_smallest_M_l59_59480

/-- 
Proof of the smallest real number M such that 
for all real numbers a, b, and c, the following inequality holds:
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)|
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2. 
-/
theorem find_smallest_M (a b c : ℝ) : 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end find_smallest_M_l59_59480


namespace solve_poly_l59_59573

open Real

-- Define the condition as a hypothesis
def prob_condition (x : ℝ) : Prop :=
  arctan (1 / x) + arctan (1 / (x^5)) = π / 6

-- Define the statement to be proven that x satisfies the polynomial equation
theorem solve_poly (x : ℝ) (h : prob_condition x) :
  x^6 - sqrt 3 * x^5 - sqrt 3 * x - 1 = 0 :=
sorry

end solve_poly_l59_59573


namespace average_salary_rest_workers_l59_59245

-- Define the conditions
def total_workers : Nat := 21
def average_salary_all_workers : ℝ := 8000
def number_of_technicians : Nat := 7
def average_salary_technicians : ℝ := 12000

-- Define the task
theorem average_salary_rest_workers :
  let number_of_rest := total_workers - number_of_technicians
  let total_salary_all := average_salary_all_workers * total_workers
  let total_salary_technicians := average_salary_technicians * number_of_technicians
  let total_salary_rest := total_salary_all - total_salary_technicians
  let average_salary_rest := total_salary_rest / number_of_rest
  average_salary_rest = 6000 :=
by
  sorry

end average_salary_rest_workers_l59_59245


namespace units_digit_27_3_sub_17_3_l59_59891

theorem units_digit_27_3_sub_17_3 : 
  (27 ^ 3 - 17 ^ 3) % 10 = 0 :=
sorry

end units_digit_27_3_sub_17_3_l59_59891


namespace product_of_differences_l59_59900

-- Define the context where x and y are real numbers
variables (x y : ℝ)

-- State the theorem to be proved
theorem product_of_differences (x y : ℝ) : 
  (-x + y) * (-x - y) = x^2 - y^2 :=
sorry

end product_of_differences_l59_59900


namespace minimize_costs_l59_59039

def total_books : ℕ := 150000
def handling_fee_per_order : ℕ := 30
def storage_fee_per_1000_copies : ℕ := 40
def evenly_distributed_books : Prop := true --Assuming books are evenly distributed by default

noncomputable def optimal_order_frequency : ℕ := 10
noncomputable def optimal_batch_size : ℕ := 15000

theorem minimize_costs 
  (handling_fee_per_order : ℕ) 
  (storage_fee_per_1000_copies : ℕ) 
  (total_books : ℕ) 
  (evenly_distributed_books : Prop)
  : optimal_order_frequency = 10 ∧ optimal_batch_size = 15000 := sorry

end minimize_costs_l59_59039


namespace largest_n_for_factored_quad_l59_59462

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l59_59462


namespace allie_betty_total_points_product_l59_59694

def score (n : Nat) : Nat :=
  if n % 3 == 0 then 9
  else if n % 2 == 0 then 3
  else if n % 2 == 1 then 1
  else 0

def allie_points : List Nat := [5, 2, 6, 1, 3]
def betty_points : List Nat := [6, 4, 1, 2, 5]

def total_points (rolls: List Nat) : Nat :=
  rolls.foldl (λ acc n => acc + score n) 0

theorem allie_betty_total_points_product : 
  total_points allie_points * total_points betty_points = 391 := by
  sorry

end allie_betty_total_points_product_l59_59694


namespace range_of_a_l59_59210

theorem range_of_a (a : ℝ) (x : ℝ) :
  (¬(x > a) →¬(x^2 + 2*x - 3 > 0)) → (a ≥ 1 ) :=
by
  intro h
  sorry

end range_of_a_l59_59210


namespace investment_time_P_l59_59866

-- Variables and conditions
variables {x : ℕ} {time_P : ℕ}

-- Conditions as seen from the mathematical problem
def investment_P (x : ℕ) := 7 * x
def investment_Q (x : ℕ) := 5 * x
def profit_ratio := 1 / 2
def time_Q := 14

-- Statement of the problem
theorem investment_time_P : 
  (profit_ratio = (investment_P x * time_P) / (investment_Q x * time_Q)) → 
  time_P = 5 := 
sorry

end investment_time_P_l59_59866


namespace correct_statement_B_l59_59299

/-- Define the diameter of a sphere -/
def diameter (d : ℝ) (s : Set (ℝ × ℝ × ℝ)) : Prop :=
∃ x y : ℝ × ℝ × ℝ, x ∈ s ∧ y ∈ s ∧ dist x y = d ∧ ∀ z ∈ s, dist x y ≥ dist x z ∧ dist x y ≥ dist z y

/-- Define that a line segment connects two points on the sphere's surface and passes through the center -/
def connects_diameter (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ) : Prop :=
dist center x = radius ∧ dist center y = radius ∧ (x + y) / 2 = center

/-- A sphere is the set of all points at a fixed distance from the center -/
def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ × ℝ) :=
{x | dist center x = radius}

theorem correct_statement_B (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ):
  (∀ (s : Set (ℝ × ℝ × ℝ)), sphere center radius = s → diameter (2 * radius) s)
  → connects_diameter center radius x y
  → (∃ d : ℝ, diameter d (sphere center radius)) := 
by
  intros
  sorry

end correct_statement_B_l59_59299


namespace total_songs_in_june_l59_59739

-- Define the conditions
def Vivian_daily_songs : ℕ := 10
def Clara_daily_songs : ℕ := Vivian_daily_songs - 2
def Lucas_daily_songs : ℕ := Vivian_daily_songs + 5
def total_play_days_in_june : ℕ := 30 - 8 - 1

-- Total songs listened to in June
def total_songs_Vivian : ℕ := Vivian_daily_songs * total_play_days_in_june
def total_songs_Clara : ℕ := Clara_daily_songs * total_play_days_in_june
def total_songs_Lucas : ℕ := Lucas_daily_songs * total_play_days_in_june

-- The total number of songs listened to by all three
def total_songs_all_three : ℕ := total_songs_Vivian + total_songs_Clara + total_songs_Lucas

-- The proof problem
theorem total_songs_in_june : total_songs_all_three = 693 := by
  -- Placeholder for the proof
  sorry

end total_songs_in_june_l59_59739


namespace number_of_single_rooms_l59_59829

theorem number_of_single_rooms (S : ℕ) : 
  (S + 13 * 2 = 40) ∧ (S * 10 + 13 * 2 * 10 = 400) → S = 14 :=
by 
  sorry

end number_of_single_rooms_l59_59829


namespace solve_for_c_l59_59521

noncomputable def triangle_side_c_proof (a c : ℝ) (C : ℝ) (S : ℝ) : Prop :=
  let b := 5 in -- derived from S and given conditions
  C = π * (2 / 3) ∧ -- 120 degrees in radians
  a = 3 ∧
  S = (15 * real.sqrt 3) / 4 ∧
  (c * c = a * a + b * b - 2 * a * b * real.cos C)

theorem solve_for_c (a : ℝ) (C : ℝ) (S : ℝ) :
  triangle_side_c_proof a 7 C S :=
by
  sorry -- Proof omitted

end solve_for_c_l59_59521


namespace part1_l59_59969

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 3 * a - 10 ≤ x ∧ x < 2 * a + 1}
def Q : Set ℝ := {x | |2 * x - 3| ≤ 7}

-- Define the complement of Q in ℝ
def Q_complement : Set ℝ := {x | x < -2 ∨ x > 5}

-- Define the specific value of a
def a : ℝ := 2

-- Define the specific set P when a = 2
def P_a2 : Set ℝ := {x | -4 ≤ x ∧ x < 5}

-- Define the intersection
def intersection : Set ℝ := {x | -4 ≤ x ∧ x < -2}

theorem part1 : P a ∩ Q_complement = intersection := sorry

end part1_l59_59969


namespace convex_cyclic_quadrilaterals_count_l59_59089

noncomputable def count_cyclic_quadrilaterals_with_perimeter_20 : ℕ :=
  (∑ (a b c d : ℕ) in
    (finset.Icc 1 20 ×ᶠ finset.Icc 1 20 ×ᶠ finset.Icc 1 20 ×ᶠ finset.Icc 1 20),
    if a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 20 then 1 else 0)

theorem convex_cyclic_quadrilaterals_count : count_cyclic_quadrilaterals_with_perimeter_20 = 124 := 
  sorry

end convex_cyclic_quadrilaterals_count_l59_59089


namespace flower_bee_difference_proof_l59_59873

variable (flowers bees : ℕ)

def flowers_bees_difference (flowers bees : ℕ) : ℕ :=
  flowers - bees

theorem flower_bee_difference_proof : flowers_bees_difference 5 3 = 2 :=
by
  sorry

end flower_bee_difference_proof_l59_59873


namespace ab_bc_ca_leq_zero_l59_59486

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end ab_bc_ca_leq_zero_l59_59486


namespace probability_top_face_odd_is_137_over_252_l59_59268

noncomputable def probability_top_face_odd (faces : Fin 6 → ℕ) : ℚ :=
  let n := 1/6 * ((399/420) + (76/420) + (323/420) + (136/420) + (256/420) + (180/420))
  n

theorem probability_top_face_odd_is_137_over_252 :
  probability_top_face_odd (λ i : Fin 6, i.val + 1) = 137 / 252 :=
by 
  sorry

end probability_top_face_odd_is_137_over_252_l59_59268


namespace division_of_repeating_decimals_l59_59014

noncomputable def repeating_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  ⟨n, d⟩

theorem division_of_repeating_decimals :
  let x := repeating_to_fraction 54 99
  let y := repeating_to_fraction 18 99
  (x / y) = (3 : ℚ) :=
by
  -- Proof omitted as requested
  sorry

end division_of_repeating_decimals_l59_59014


namespace final_dollars_final_euros_final_rubles_l59_59771

-- Defining constants and initial amounts
def initial_euros : ℝ := 3000
def initial_dollars : ℝ := 4000
def initial_rubles : ℝ := 240000

def interest_rate_eur : ℝ := 0.021
def interest_rate_usd : ℝ := 0.021
def interest_rate_rub : ℝ := 0.079

def sell_rate_eur_to_rub : ℝ := 60.10
def buy_rate_rub_to_usd : ℝ := 58.90
def sell_rate_usd_to_rub : ℝ := 58.50
def buy_rate_rub_to_eur : ℝ := 61.20

-- Function to compound interest yearly
def compound_interest (principal rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

-- After year 1
def year1_euros : ℝ := compound_interest initial_euros interest_rate_eur 1
def year1_dollars : ℝ := compound_interest initial_dollars interest_rate_usd 1
def year1_rubles : ℝ := compound_interest initial_rubles interest_rate_rub 1

-- Conversion after year 1
def euros_sold : ℝ := 1000
def rubles_from_euros : ℝ := euros_sold * sell_rate_eur_to_rub
def dollars_purchased : ℝ := rubles_from_euros / buy_rate_rub_to_usd

def year2_euros : ℝ :=
  compound_interest (year1_euros - euros_sold) interest_rate_eur 1
def year2_dollars : ℝ :=
  compound_interest (year1_dollars + dollars_purchased) interest_rate_usd 1
def year2_rubles : ℝ :=
  compound_interest year1_rubles interest_rate_rub 1

-- Conversion after year 2
def dollars_sold : ℝ := 2000
def rubles_from_dollars : ℝ := dollars_sold * sell_rate_usd_to_rub
def euros_purchased : ℝ := rubles_from_dollars / buy_rate_rub_to_eur

def year3_euros : ℝ :=
  compound_interest (year2_euros + euros_purchased) interest_rate_eur 1
def year3_dollars : ℝ :=
  compound_interest (year2_dollars - dollars_sold) interest_rate_usd 1
def year3_rubles : ℝ :=
  compound_interest year2_rubles interest_rate_rub 1

-- Statements to prove
theorem final_dollars : round year3_dollars = 3286 := by sorry
theorem final_euros : round year3_euros = 4040 := by sorry
theorem final_rubles : round year3_rubles = 301504 := by sorry

end final_dollars_final_euros_final_rubles_l59_59771


namespace fraction_diff_equals_7_over_12_l59_59884

noncomputable def fraction_diff : ℚ :=
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6)

theorem fraction_diff_equals_7_over_12 : fraction_diff = 7 / 12 := by
  sorry

end fraction_diff_equals_7_over_12_l59_59884


namespace parabola_y_relation_l59_59864

-- Conditions of the problem
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 - 4 * x + c

-- The proof problem statement
theorem parabola_y_relation (c y1 y2 y3 : ℝ) :
  parabola (-4) c = y1 →
  parabola (-2) c = y2 →
  parabola (1 / 2) c = y3 →
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end parabola_y_relation_l59_59864


namespace problem_statements_analysis_l59_59300

theorem problem_statements_analysis:
  (¬∀ a b : ℝ, a + b > a) ∧
  (¬∀ x : ℝ, |x| = -x → x < 0) ∧
  (¬∀ x y : ℝ, |x| = |y| → x = y) ∧
  (∀ p q : ℤ, q ≠ 0 → (p/q : ℚ)).
by
  -- Proof of theorems goes here, but omitted with sorry
  sorry

end problem_statements_analysis_l59_59300


namespace wendy_washing_loads_l59_59294

theorem wendy_washing_loads (shirts sweaters machine_capacity : ℕ) (total_clothes := shirts + sweaters) 
  (loads := total_clothes / machine_capacity) 
  (remainder := total_clothes % machine_capacity) 
  (h_shirts : shirts = 39) 
  (h_sweaters : sweaters = 33) 
  (h_machine_capacity : machine_capacity = 8) : loads = 9 ∧ remainder = 0 := 
by 
  sorry

end wendy_washing_loads_l59_59294


namespace arithmetic_sequence_sum_l59_59181

theorem arithmetic_sequence_sum :
  let first_term := 1
  let common_diff := 2
  let last_term := 33
  let n := (last_term + 1) / common_diff
  (n * (first_term + last_term)) / 2 = 289 :=
by
  sorry

end arithmetic_sequence_sum_l59_59181


namespace minimum_value_problem_l59_59119

theorem minimum_value_problem (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) 
  (hxyz : x + y + z + w = 1) :
  (1 / (x + y) + 1 / (x + z) + 1 / (x + w) + 1 / (y + z) + 1 / (y + w) + 1 / (z + w)) ≥ 18 := 
sorry

end minimum_value_problem_l59_59119


namespace range_of_x_l59_59337

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) 
  (ineq : |a + b| + |a - b| ≥ |a| * |x - 2|) : 
  0 ≤ x ∧ x ≤ 4 :=
  sorry

end range_of_x_l59_59337


namespace sum_of_solutions_l59_59985

theorem sum_of_solutions (x : ℝ) (h : x^2 - 3 * x = 12) : x = 3 := by
  sorry

end sum_of_solutions_l59_59985


namespace tangent_line_of_f_eq_kx_l59_59217

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

theorem tangent_line_of_f_eq_kx (k : ℝ) : 
    (∃ x₀, tangent_line k x₀ = f x₀ ∧ deriv f x₀ = k) → 
    (k = 0 ∨ k = 1 ∨ k = -1) := 
  sorry

end tangent_line_of_f_eq_kx_l59_59217


namespace staff_discount_l59_59442

theorem staff_discount (d : ℝ) (S : ℝ) (h1 : d > 0)
    (h2 : 0.455 * d = (1 - S / 100) * (0.65 * d)) : S = 30 := by
    sorry

end staff_discount_l59_59442


namespace Diane_bakes_160_gingerbreads_l59_59955

-- Definitions
def trays1Count : Nat := 4
def gingerbreads1PerTray : Nat := 25
def trays2Count : Nat := 3
def gingerbreads2PerTray : Nat := 20

def totalGingerbreads : Nat :=
  (trays1Count * gingerbreads1PerTray) + (trays2Count * gingerbreads2PerTray)

-- Problem statement
theorem Diane_bakes_160_gingerbreads :
  totalGingerbreads = 160 := by
  sorry

end Diane_bakes_160_gingerbreads_l59_59955


namespace average_weight_of_eight_boys_l59_59718

theorem average_weight_of_eight_boys :
  let avg16 := 50.25
  let avg24 := 48.55
  let total_weight_16 := 16 * avg16
  let total_weight_all := 24 * avg24
  let W := (total_weight_all - total_weight_16) / 8
  W = 45.15 :=
by
  sorry

end average_weight_of_eight_boys_l59_59718


namespace whale_plankton_consumption_l59_59614

theorem whale_plankton_consumption
  (P : ℕ) -- Amount of plankton consumed in the first hour
  (h1 : ∀ n : ℕ, 1 ≤ n → n ≤ 9 → P + (n - 1) * 3 ∈ ℕ) -- Plankton consumption follows an arithmetic sequence over 9 hours
  (h2 : (finset.range 9).sum (λ n, P + n * 3) = 450) -- Total plankton consumption over 9 hours is 450 kilos
  (h3 : P + 15 = 53) -- On the sixth hour, the whale consumed 53 kilos
  : P = 38 := 
sorry

end whale_plankton_consumption_l59_59614


namespace sum_of_two_numbers_l59_59146

theorem sum_of_two_numbers (x y : ℝ) (h1 : 0.5 * x + 0.3333 * y = 11)
(h2 : max x y = y) (h3 : y = 15) : x + y = 27 :=
by
  -- Skip the proof and add sorry
  sorry

end sum_of_two_numbers_l59_59146


namespace proof_N_union_complement_M_eq_235_l59_59373

open Set

theorem proof_N_union_complement_M_eq_235 :
  let U := ({1,2,3,4,5} : Set ℕ)
  let M := ({1, 4} : Set ℕ)
  let N := ({2, 5} : Set ℕ)
  N ∪ (U \ M) = ({2, 3, 5} : Set ℕ) :=
by
  sorry

end proof_N_union_complement_M_eq_235_l59_59373


namespace sufficient_but_not_necessary_condition_l59_59047

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 + x - 2 > 0) ∧ (∃ y, y < -2 ∧ y^2 + y - 2 > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l59_59047


namespace probability_three_non_red_purple_balls_l59_59438

def total_balls : ℕ := 150
def prob_white : ℝ := 0.15
def prob_green : ℝ := 0.20
def prob_yellow : ℝ := 0.30
def prob_red : ℝ := 0.30
def prob_purple : ℝ := 0.05
def prob_not_red_purple : ℝ := 1 - (prob_red + prob_purple)

theorem probability_three_non_red_purple_balls :
  (prob_not_red_purple * prob_not_red_purple * prob_not_red_purple) = 0.274625 :=
by
  sorry

end probability_three_non_red_purple_balls_l59_59438


namespace smaller_circle_circumference_l59_59137

noncomputable def circumference_of_smaller_circle :=
  let π := Real.pi
  let R := 352 / (2 * π)
  let area_difference := 4313.735577562732
  let R_squared_minus_r_squared := area_difference / π
  let r_squared := R ^ 2 - R_squared_minus_r_squared
  let r := Real.sqrt r_squared
  2 * π * r

theorem smaller_circle_circumference : 
  let circumference_larger := 352
  let area_difference := 4313.735577562732
  circumference_of_smaller_circle = 263.8934 := sorry

end smaller_circle_circumference_l59_59137


namespace number_of_squares_l59_59236

open Int

theorem number_of_squares (n : ℕ) (h : n < 10^7) : 
  (∃ n, 36 ∣ n ∧ n^2 < 10^07) ↔ (n = 87) :=
by sorry

end number_of_squares_l59_59236


namespace balloons_in_each_bag_of_round_balloons_l59_59367

variable (x : ℕ)

-- Definitions based on the problem's conditions
def totalRoundBalloonsBought := 5 * x
def totalLongBalloonsBought := 4 * 30
def remainingRoundBalloons := totalRoundBalloonsBought x - 5
def totalRemainingBalloons := remainingRoundBalloons x + totalLongBalloonsBought

-- Theorem statement based on the question and derived from the conditions and correct answer
theorem balloons_in_each_bag_of_round_balloons : totalRemainingBalloons x = 215 → x = 20 := by
  -- We acknowledge that the proof steps will follow here (omitted as per instructions)
  sorry

end balloons_in_each_bag_of_round_balloons_l59_59367


namespace fraction_sum_is_0_333_l59_59396

theorem fraction_sum_is_0_333 : (3 / 10 : ℝ) + (3 / 100) + (3 / 1000) = 0.333 := 
by
  sorry

end fraction_sum_is_0_333_l59_59396


namespace smallest_integer_divisibility_conditions_l59_59596

theorem smallest_integer_divisibility_conditions :
  ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (900 ∣ n^3) ∧ (1024 ∣ n^4) ∧ n = 120 :=
by
  sorry

end smallest_integer_divisibility_conditions_l59_59596


namespace max_checkers_attacked_l59_59019

def is_adjacent (i j i' j' : ℕ) : Prop :=
  (i' = i + 1 ∧ j' = j) ∨ (i' = i - 1 ∧ j' = j) ∨ (i' = i ∧ j' = j + 1) ∨ (i' = i ∧ j' = j - 1) ∨
  (i' = i + 1 ∧ j' = j + 1) ∨ (i' = i + 1 ∧ j' = j - 1) ∨ (i' = i - 1 ∧ j' = j + 1) ∨ (i' = i - 1 ∧ j' = j - 1)

def is_attacked (P : Finset (ℕ × ℕ)) (i j : ℕ) : Prop :=
  ∃ (i' j' : ℕ), (i', j') ∈ P ∧ is_adjacent i j i' j'

theorem max_checkers_attacked (P : Finset (ℕ × ℕ)) :
  (∀ (i j : ℕ), (i, j) ∈ P → is_attacked P i j) → P.card ≤ 32 := sorry

end max_checkers_attacked_l59_59019


namespace cyclic_inequality_l59_59383

theorem cyclic_inequality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^3 * b^3 * (a * b - a * c - b * c + c^2) +
   b^3 * c^3 * (b * c - b * a - c * a + a^2) +
   c^3 * a^3 * (c * a - c * b - a * b + b^2)) ≥ 0 :=
sorry

end cyclic_inequality_l59_59383


namespace sum_of_distinct_roots_l59_59538

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l59_59538


namespace evaluate_g_at_5_l59_59257

noncomputable def g (x : ℝ) : ℝ := 2 * x ^ 4 - 15 * x ^ 3 + 24 * x ^ 2 - 18 * x - 72

theorem evaluate_g_at_5 : g 5 = -7 := by
  sorry

end evaluate_g_at_5_l59_59257


namespace evaluate_expression_l59_59035

noncomputable def ln (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression : 
  2017 ^ ln (ln 2017) - (ln 2017) ^ ln 2017 = 0 :=
by
  sorry

end evaluate_expression_l59_59035


namespace Kylie_US_coins_left_l59_59252

-- Define the given conditions
def initial_US_coins : ℝ := 15
def Euro_coins : ℝ := 13
def Canadian_coins : ℝ := 8
def US_coins_given_to_Laura : ℝ := 21
def Euro_to_US_rate : ℝ := 1.18
def Canadian_to_US_rate : ℝ := 0.78

-- Define the conversions
def Euro_to_US : ℝ := Euro_coins * Euro_to_US_rate
def Canadian_to_US : ℝ := Canadian_coins * Canadian_to_US_rate
def total_US_before_giving : ℝ := initial_US_coins + Euro_to_US + Canadian_to_US
def US_left_with : ℝ := total_US_before_giving - US_coins_given_to_Laura

-- Statement of the problem to be proven
theorem Kylie_US_coins_left :
  US_left_with = 15.58 := by
  sorry

end Kylie_US_coins_left_l59_59252


namespace smallest_positive_integer_l59_59890
-- Import the required library

-- State the problem in Lean
theorem smallest_positive_integer (x : ℕ) (h : 5 * x ≡ 17 [MOD 31]) : x = 13 :=
sorry

end smallest_positive_integer_l59_59890


namespace diophantine_infinite_solutions_l59_59258

theorem diophantine_infinite_solutions
  (l m n : ℕ) (h_l_positive : l > 0) (h_m_positive : m > 0) (h_n_positive : n > 0)
  (h_gcd_lm_n : gcd (l * m) n = 1) (h_gcd_ln_m : gcd (l * n) m = 1) (h_gcd_mn_l : gcd (m * n) l = 1)
  : ∃ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0 ∧ (x ^ l + y ^ m = z ^ n)) ∧ (∀ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ (a ^ l + b ^ m = c ^ n)) → ∀ d : ℕ, d > 0 → ∃ e f g : ℕ, (e > 0 ∧ f > 0 ∧ g > 0 ∧ (e ^ l + f ^ m = g ^ n))) :=
sorry

end diophantine_infinite_solutions_l59_59258


namespace total_money_l59_59164

theorem total_money (total_coins nickels dimes : ℕ) (val_nickel val_dime : ℕ)
  (h1 : total_coins = 8)
  (h2 : nickels = 2)
  (h3 : total_coins = nickels + dimes)
  (h4 : val_nickel = 5)
  (h5 : val_dime = 10) :
  (nickels * val_nickel + dimes * val_dime) = 70 :=
by
  sorry

end total_money_l59_59164


namespace speedster_convertibles_proof_l59_59432

-- Definitions based on conditions
def total_inventory (T : ℕ) : Prop := 2 / 3 * T = 2 / 3 * T
def not_speedsters (T : ℕ) : Prop := 1 / 3 * T = 60
def speedsters (T : ℕ) (S : ℕ) : Prop := S = 2 / 3 * T
def speedster_convertibles (S : ℕ) (C : ℕ) : Prop := C = 4 / 5 * S

theorem speedster_convertibles_proof (T S C : ℕ) (hT : total_inventory T) (hNS : not_speedsters T) (hS : speedsters T S) (hSC : speedster_convertibles S C) : C = 96 :=
by
  -- Proof goes here
  sorry

end speedster_convertibles_proof_l59_59432


namespace fraction_budget_paid_l59_59190

variable (B : ℝ) (b k : ℝ)

-- Conditions
def condition1 : b = 0.30 * (B - k) := by sorry
def condition2 : k = 0.10 * (B - b) := by sorry

-- Proof that Jenny paid 35% of her budget for her book and snack
theorem fraction_budget_paid :
  b + k = 0.35 * B :=
by
  -- use condition1 and condition2 to prove the theorem
  sorry

end fraction_budget_paid_l59_59190


namespace g_675_eq_42_l59_59256

noncomputable def g : ℕ → ℕ := sorry

axiom gxy : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g15 : g 15 = 18
axiom g45 : g 45 = 24

theorem g_675_eq_42 : g 675 = 42 :=
sorry

end g_675_eq_42_l59_59256


namespace largest_n_for_factored_quad_l59_59463

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l59_59463


namespace solve_equation_l59_59068

theorem solve_equation (x: ℝ) (h : (5 - x)^(1/3) = -5/2) : x = 165/8 :=
by
  -- proof skipped
  sorry

end solve_equation_l59_59068


namespace part1_part2_l59_59211

-- Definitions for the conditions
def A : Set ℝ := {x : ℝ | 2 * x - 4 < 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}
def U : Set ℝ := Set.univ

-- The questions translated as Lean theorems
theorem part1 : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

theorem part2 : (U \ A) ∩ B = {x : ℝ | 2 ≤ x ∧ x < 5} := by
  sorry

end part1_part2_l59_59211


namespace find_values_of_a_and_b_l59_59725

-- Definition of the problem and required conditions:
def symmetric_point (a b : ℝ) : Prop :=
  (a = -2) ∧ (b = -3)

theorem find_values_of_a_and_b (a b : ℝ) 
  (h : (a, -3) = (-2, -3) ∨ (2, b) = (2, -3) ∧ (a = -2)) :
  symmetric_point a b :=
by
  sorry

end find_values_of_a_and_b_l59_59725


namespace solve_for_x_l59_59672

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l59_59672


namespace find_mn_l59_59248

variable (OA OB OC : EuclideanSpace ℝ (Fin 3))
variable (AOC BOC : ℝ)

axiom length_OA : ‖OA‖ = 2
axiom length_OB : ‖OB‖ = 2
axiom length_OC : ‖OC‖ = 2 * Real.sqrt 3
axiom tan_angle_AOC : Real.tan AOC = 3 * Real.sqrt 3
axiom angle_BOC : BOC = Real.pi / 3

theorem find_mn : ∃ m n : ℝ, OC = m • OA + n • OB ∧ m = 5 / 3 ∧ n = 2 * Real.sqrt 3 := by
  sorry

end find_mn_l59_59248


namespace find_a_l59_59991

theorem find_a (a b c : ℚ)
  (h1 : c / b = 4)
  (h2 : b / a = 2)
  (h3 : c = 20 - 7 * b) : a = 10 / 11 :=
by
  sorry

end find_a_l59_59991


namespace percent_value_quarters_l59_59302

noncomputable def value_in_cents (dimes quarters nickels : ℕ) : ℕ := 
  (dimes * 10) + (quarters * 25) + (nickels * 5)

noncomputable def percent_in_quarters (quarters total_value : ℕ) : ℚ := 
  (quarters * 25 : ℚ) / total_value * 100

theorem percent_value_quarters 
  (h_dimes : ℕ := 80) 
  (h_quarters : ℕ := 30) 
  (h_nickels : ℕ := 40) 
  (h_total_value := value_in_cents h_dimes h_quarters h_nickels) : 
  percent_in_quarters h_quarters h_total_value = 42.86 :=
by sorry

end percent_value_quarters_l59_59302


namespace no_real_solutions_l59_59237

theorem no_real_solutions : ∀ x : ℝ, (3 * x - 4) ^ 2 + 3 ≠ -2 * |x - 1| :=
by
  intro x
  have h1 : (3 * x - 4) ^ 2 + 3 ≥ 3 :=
    calc
      (3 * x - 4) ^ 2 + 3 ≥ 0 + 3 : by apply add_le_add_right (pow_two_nonneg (3 * x - 4)) 3
      _ = 3 : by rw zero_add
  have h2 : -2 * |x - 1| ≤ 0 := by nlinarith [abs_nonneg (x - 1)]
  linarith

end no_real_solutions_l59_59237


namespace fifteen_percent_of_x_l59_59098

variables (x : ℝ)

-- Condition: Given x% of 60 is 12
def is_x_percent_of_60 : Prop := (x / 100) * 60 = 12

-- Prove: 15% of x is 3
theorem fifteen_percent_of_x (h : is_x_percent_of_60 x) : (15 / 100) * x = 3 :=
by
  sorry

end fifteen_percent_of_x_l59_59098


namespace base3_to_base10_conversion_l59_59946

theorem base3_to_base10_conversion : 
  1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0 = 142 :=
by {
  -- calculations
  sorry
}

end base3_to_base10_conversion_l59_59946


namespace A_neg10_3_eq_neg1320_l59_59628

noncomputable def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem A_neg10_3_eq_neg1320 : A (-10) 3 = -1320 := 
by
  sorry

end A_neg10_3_eq_neg1320_l59_59628


namespace sum_of_distinct_real_numbers_l59_59547

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l59_59547


namespace area_ratio_l59_59865

theorem area_ratio (l b r : ℝ) (h1 : l = 2 * b) (h2 : 6 * b = 2 * π * r) :
  (l * b) / (π * r ^ 2) = 2 * π / 9 :=
by {
  sorry
}

end area_ratio_l59_59865


namespace probability_of_A_l59_59306

theorem probability_of_A :
  ∀ (P : set (set ℝ) → ℝ) (A B : set ℝ),
  (independent P [A, B]) →
  (0 < P A) →
  (P A = 2 * P B) →
  (P (A ∪ B) = 5 * P (A ∩ B)) →
  P A = 1 / 2 := by
  sorry

end probability_of_A_l59_59306


namespace eliot_votes_l59_59517

theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ)
                    (h1 : randy_votes = 16)
                    (h2 : shaun_votes = 5 * randy_votes)
                    (h3 : eliot_votes = 2 * shaun_votes) :
                    eliot_votes = 160 :=
by {
  -- Proof will be conducted here
  sorry
}

end eliot_votes_l59_59517


namespace no_nat_p_prime_and_p6_plus_6_prime_l59_59783

theorem no_nat_p_prime_and_p6_plus_6_prime (p : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime (p^6 + 6)) : False := 
sorry

end no_nat_p_prime_and_p6_plus_6_prime_l59_59783


namespace exists_universal_accessible_city_l59_59821

-- Define the basic structure for cities and flights
structure Country :=
  (City : Type)
  (accessible : City → City → Prop)

namespace Country

-- Define the properties of accessibility in the country
variables {C : Country}

-- Axiom: Each city is accessible from itself
axiom self_accessible (A : C.City) : C.accessible A A

-- Axiom: For any two cities, there exists a city from which both are accessible
axiom exists_intermediate (P Q : C.City) : ∃ R : C.City, C.accessible R P ∧ C.accessible R Q

-- Definition of the main theorem
theorem exists_universal_accessible_city :
  ∃ U : C.City, ∀ A : C.City, C.accessible U A :=
sorry

end Country

end exists_universal_accessible_city_l59_59821


namespace color_films_count_l59_59441

variables (x y C : ℕ)
variables (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ)))

theorem color_films_count (x y : ℕ) (C : ℕ) (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ))) :
  C = 10 * y :=
sorry

end color_films_count_l59_59441


namespace sum_of_distinct_real_numbers_l59_59544

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l59_59544


namespace verify_inclination_angles_l59_59807

noncomputable def inclination_angle_of_focal_chord (p : ℝ) (θ : ℝ) : Prop :=
  (sqrt(2 * p) * sqrt(2 * p) * sin(θ) * cos(θ) = 8 * p * (sin(θ))/(sin(θ))) → θ = π / 6 ∨ θ = 5 * π / 6

theorem verify_inclination_angles (p : ℝ) (hp : 0 < p):
  inclination_angle_of_focal_chord p (π / 6) ∨ inclination_angle_of_focal_chord p (5 * π / 6) :=
sorry

end verify_inclination_angles_l59_59807


namespace perfect_squares_l59_59405

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l59_59405


namespace surface_area_of_circumscribed_sphere_of_triangular_pyramid_l59_59797

theorem surface_area_of_circumscribed_sphere_of_triangular_pyramid
  (a : ℝ)
  (h₁ : a > 0) : 
  ∃ S, S = (27 * π / 32 * a^2) := 
by
  sorry

end surface_area_of_circumscribed_sphere_of_triangular_pyramid_l59_59797


namespace find_x_when_y_equals_two_l59_59676

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l59_59676


namespace distance_to_post_office_l59_59433

variable (D : ℚ)
variable (rate_to_post : ℚ := 25)
variable (rate_back : ℚ := 4)
variable (total_time : ℚ := 5 + 48 / 60)

theorem distance_to_post_office : (D / rate_to_post + D / rate_back = total_time) → D = 20 := by
  sorry

end distance_to_post_office_l59_59433


namespace train_passes_man_in_4_4_seconds_l59_59924

noncomputable def train_speed_kmph : ℝ := 84
noncomputable def man_speed_kmph : ℝ := 6
noncomputable def train_length_m : ℝ := 110

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def man_speed_mps : ℝ :=
  kmph_to_mps man_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps + man_speed_mps

noncomputable def passing_time : ℝ :=
  train_length_m / relative_speed_mps

theorem train_passes_man_in_4_4_seconds :
  passing_time = 4.4 :=
by
  sorry -- Proof not required, skipping the proof logic

end train_passes_man_in_4_4_seconds_l59_59924


namespace repeating_decimal_eq_l59_59332

-- Defining the repeating decimal as a hypothesis
def repeating_decimal : ℚ := 0.7 + 3/10^2 * (1/(1 - 1/10))
-- We will prove this later by simplifying the fraction
def expected_fraction : ℚ := 11/15

theorem repeating_decimal_eq : repeating_decimal = expected_fraction := 
by
  sorry

end repeating_decimal_eq_l59_59332


namespace area_of_306090_triangle_l59_59774

-- Conditions
def is_306090_triangle (a b c : ℝ) : Prop :=
  a / b = 1 / Real.sqrt 3 ∧ a / c = 1 / 2

-- Given values
def hypotenuse : ℝ := 6

-- To prove
theorem area_of_306090_triangle :
  ∃ (a b c : ℝ), is_306090_triangle a b c ∧ c = hypotenuse ∧ (1 / 2) * a * b = (9 * Real.sqrt 3) / 2 :=
by
  sorry

end area_of_306090_triangle_l59_59774


namespace ac_lt_bc_if_c_lt_zero_l59_59077

variables {a b c : ℝ}
theorem ac_lt_bc_if_c_lt_zero (h : a > b) (h1 : b > c) (h2 : c < 0) : a * c < b * c :=
sorry

end ac_lt_bc_if_c_lt_zero_l59_59077


namespace second_year_undeclared_fraction_l59_59844

def total_students := 12

def fraction_first_year : ℚ := 1 / 4
def fraction_second_year : ℚ := 1 / 2
def fraction_third_year : ℚ := 1 / 6
def fraction_fourth_year : ℚ := 1 / 12

def fraction_undeclared_first_year : ℚ := 4 / 5
def fraction_undeclared_second_year : ℚ := 3 / 4
def fraction_undeclared_third_year : ℚ := 1 / 3
def fraction_undeclared_fourth_year : ℚ := 1 / 6

def students_first_year : ℚ := total_students * fraction_first_year
def students_second_year : ℚ := total_students * fraction_second_year
def students_third_year : ℚ := total_students * fraction_third_year
def students_fourth_year : ℚ := total_students * fraction_fourth_year

def undeclared_first_year : ℚ := students_first_year * fraction_undeclared_first_year
def undeclared_second_year : ℚ := students_second_year * fraction_undeclared_second_year
def undeclared_third_year : ℚ := students_third_year * fraction_undeclared_third_year
def undeclared_fourth_year : ℚ := students_fourth_year * fraction_undeclared_fourth_year

theorem second_year_undeclared_fraction :
  (undeclared_second_year / total_students) = 1 / 3 :=
by
  sorry  -- Proof to be provided

end second_year_undeclared_fraction_l59_59844


namespace range_of_x_l59_59339

theorem range_of_x (x : ℝ) : -2 * x + 3 ≤ 6 → x ≥ -3 / 2 :=
sorry

end range_of_x_l59_59339


namespace eleven_pow_2010_mod_19_l59_59425

theorem eleven_pow_2010_mod_19 : (11 ^ 2010) % 19 = 3 := sorry

end eleven_pow_2010_mod_19_l59_59425


namespace sqrt_product_simplification_l59_59322

theorem sqrt_product_simplification (q : ℝ) : 
  sqrt (42 * q) * sqrt (7 * q) * sqrt (3 * q) = 126 * q * sqrt q := 
by
  sorry

end sqrt_product_simplification_l59_59322


namespace tony_rope_length_l59_59879

-- Definitions based on the conditions in the problem
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss_per_knot : ℝ := 1.2
def number_of_knots : ℕ := 5

-- The final length of the rope after tying all pieces together and losing length per knot
def final_rope_length (lengths : List ℝ) (loss_per_knot : ℝ) (number_of_knots : ℕ) : ℝ :=
  List.sum lengths - (loss_per_knot * number_of_knots)

theorem tony_rope_length :
  final_rope_length rope_lengths knot_loss_per_knot number_of_knots = 35 := by
  sorry

end tony_rope_length_l59_59879


namespace one_over_x_plus_one_over_y_eq_fifteen_l59_59634

theorem one_over_x_plus_one_over_y_eq_fifteen
  (x y : ℝ)
  (h1 : xy > 0)
  (h2 : 1 / xy = 5)
  (h3 : (x + y) / 5 = 0.6) : 
  (1 / x) + (1 / y) = 15 := 
by
  sorry

end one_over_x_plus_one_over_y_eq_fifteen_l59_59634


namespace lcm_of_numbers_l59_59753

theorem lcm_of_numbers (a b c d : ℕ) (h1 : a = 8) (h2 : b = 24) (h3 : c = 36) (h4 : d = 54) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 216 := 
by 
  sorry

end lcm_of_numbers_l59_59753


namespace binom_12_3_equal_220_l59_59942

theorem binom_12_3_equal_220 : Nat.choose 12 3 = 220 := by sorry

end binom_12_3_equal_220_l59_59942


namespace total_population_of_city_l59_59160

theorem total_population_of_city (P : ℝ) (h : 0.85 * P = 85000) : P = 100000 :=
  by
  sorry

end total_population_of_city_l59_59160


namespace find_xy_l59_59493

theorem find_xy (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : (x - y)^2 = 9) : x * y = 3 :=
sorry

end find_xy_l59_59493


namespace find_f_2_l59_59497

def f (a b x : ℝ) := a * x^3 - b * x + 1

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = -1) : f a b 2 = 3 :=
by
  sorry

end find_f_2_l59_59497


namespace largest_four_digit_number_l59_59960

def is_four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N ≤ 9999

def sum_of_digits (N : ℕ) : ℕ :=
  let a := N / 1000
  let b := (N % 1000) / 100
  let c := (N % 100) / 10
  let d := N % 10
  a + b + c + d

def is_divisible (N S : ℕ) : Prop := N % S = 0

theorem largest_four_digit_number :
  ∃ N : ℕ, is_four_digit_number N ∧ is_divisible N (sum_of_digits N) ∧
  (∀ M : ℕ, is_four_digit_number M ∧ is_divisible M (sum_of_digits M) → N ≥ M) ∧ N = 9990 :=
by
  sorry

end largest_four_digit_number_l59_59960


namespace total_students_in_class_l59_59820

theorem total_students_in_class (R S : ℕ)
  (h1 : 2 + 12 * 1 + 12 * 2 + 3 * R = S * 2)
  (h2 : S = 2 + 12 + 12 + R) :
  S = 42 :=
by
  sorry

end total_students_in_class_l59_59820


namespace geometric_sequence_value_of_b_l59_59726

-- Definitions
def is_geometric_sequence (a b c : ℝ) := 
  ∃ r : ℝ, a * r = b ∧ b * r = c

-- Theorem statement
theorem geometric_sequence_value_of_b (b : ℝ) (h : b > 0) 
  (h_seq : is_geometric_sequence 15 b 1) : b = Real.sqrt 15 :=
by
  sorry

end geometric_sequence_value_of_b_l59_59726


namespace exists_colored_right_triangle_l59_59013

theorem exists_colored_right_triangle (color : ℝ × ℝ → ℕ) 
  (h_nonempty_blue  : ∃ p, color p = 0)
  (h_nonempty_green : ∃ p, color p = 1)
  (h_nonempty_red   : ∃ p, color p = 2) :
  ∃ p1 p2 p3 : ℝ × ℝ, 
    (p1 ≠ p2) ∧ (p2 ≠ p3) ∧ (p1 ≠ p3) ∧ 
    ((color p1 = 0) ∧ (color p2 = 1) ∧ (color p3 = 2) ∨ 
     (color p1 = 0) ∧ (color p2 = 2) ∧ (color p3 = 1) ∨ 
     (color p1 = 1) ∧ (color p2 = 0) ∧ (color p3 = 2) ∨ 
     (color p1 = 1) ∧ (color p2 = 2) ∧ (color p3 = 0) ∨ 
     (color p1 = 2) ∧ (color p2 = 0) ∧ (color p3 = 1) ∨ 
     (color p1 = 2) ∧ (color p2 = 1) ∧ (color p3 = 0))
  ∧ ((p1.1 = p2.1 ∧ p2.2 = p3.2) ∨ (p1.2 = p2.2 ∧ p2.1 = p3.1)) :=
sorry

end exists_colored_right_triangle_l59_59013


namespace repeating_decimal_division_l59_59017

theorem repeating_decimal_division :
  (0.\overline{54} / 0.\overline{18}) = 3 :=
by
  have h1 : 0.\overline{54} = 54 / 99 := sorry
  have h2 : 0.\overline{18} = 18 / 99 := sorry
  have h3 : (54 / 99) / (18 / 99) = 54 / 18 := sorry
  have h4 : 54 / 18 = 3 := sorry
  rw [h1, h2, h3, h4]
  exact rfl

end repeating_decimal_division_l59_59017


namespace price_per_unit_l59_59161

theorem price_per_unit (x y : ℝ) 
    (h1 : 2 * x + 3 * y = 690) 
    (h2 : x + 4 * y = 720) : 
    x = 120 ∧ y = 150 := 
by 
    sorry

end price_per_unit_l59_59161


namespace sequence_from_520_to_523_is_0_to_3_l59_59988

theorem sequence_from_520_to_523_is_0_to_3 
  (repeating_pattern : ℕ → ℕ)
  (h_periodic : ∀ n, repeating_pattern (n + 5) = repeating_pattern n) :
  ((repeating_pattern 520, repeating_pattern 521, repeating_pattern 522, repeating_pattern 523) = (repeating_pattern 0, repeating_pattern 1, repeating_pattern 2, repeating_pattern 3)) :=
by {
  sorry
}

end sequence_from_520_to_523_is_0_to_3_l59_59988


namespace binom_divisible_by_4_l59_59150

theorem binom_divisible_by_4 (n : ℕ) : (n ≠ 0) ∧ (¬ (∃ k : ℕ, n = 2^k)) ↔ 4 ∣ n * (Nat.choose (2 * n) n) :=
by
  sorry

end binom_divisible_by_4_l59_59150


namespace households_used_both_brands_l59_59758

theorem households_used_both_brands 
  (total_households : ℕ)
  (neither_AB : ℕ)
  (only_A : ℕ)
  (h3 : ∀ (both : ℕ), ∃ (only_B : ℕ), only_B = 3 * both)
  (h_sum : ∀ (both : ℕ), neither_AB + only_A + both + (3 * both) = total_households) :
  ∃ (both : ℕ), both = 10 :=
by 
  sorry

end households_used_both_brands_l59_59758


namespace cube_convex_hull_half_volume_l59_59109

theorem cube_convex_hull_half_volume : 
  ∃ a : ℝ, 0 <= a ∧ a <= 1 ∧ 4 * (a^3) / 6 + 4 * ((1 - a)^3) / 6 = 1 / 2 :=
by
  sorry

end cube_convex_hull_half_volume_l59_59109


namespace jumping_bug_ways_l59_59913

-- Define the problem with given conditions and required answer
theorem jumping_bug_ways :
  let starting_position := 0
  let ending_position := 3
  let jumps := 5
  let jump_options := [1, -1]
  (∃ (jump_seq : Fin jumps → ℤ), (∀ i, jump_seq i ∈ jump_options ∧ (List.sum (List.ofFn jump_seq) = ending_position)) ∧
  (List.count (-1) (List.ofFn jump_seq) = 1)) →
  (∃ n : ℕ, n = 5) :=
by
  sorry  -- Proof to be completed

end jumping_bug_ways_l59_59913


namespace quadratic_rewrite_de_value_l59_59110

theorem quadratic_rewrite_de_value : 
  ∃ (d e f : ℤ), (d^2 * x^2 + 2 * d * e * x + e^2 + f = 4 * x^2 - 16 * x + 2) → (d * e = -8) :=
by
  sorry

end quadratic_rewrite_de_value_l59_59110


namespace integer_division_condition_l59_59065

theorem integer_division_condition (n : ℕ) (h1 : n > 1): (∃ k : ℕ, 2^n + 1 = k * n^2) → n = 3 :=
by sorry

end integer_division_condition_l59_59065


namespace distinct_primes_eq_1980_l59_59755

theorem distinct_primes_eq_1980 (p q r A : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
    (hne1 : p ≠ q) (hne2 : q ≠ r) (hne3 : p ≠ r) 
    (h1 : 2 * p * q * r + 50 * p * q = A)
    (h2 : 7 * p * q * r + 55 * p * r = A)
    (h3 : 8 * p * q * r + 12 * q * r = A) : 
    A = 1980 := by {
  sorry
}

end distinct_primes_eq_1980_l59_59755


namespace unique_line_equal_intercepts_l59_59410

-- Definitions of the point and line
structure Point where
  x : ℝ
  y : ℝ

def passesThrough (L : ℝ → ℝ) (P : Point) : Prop :=
  L P.x = P.y

noncomputable def hasEqualIntercepts (L : ℝ → ℝ) : Prop :=
  ∃ a, L 0 = a ∧ L a = 0

-- The main theorem statement
theorem unique_line_equal_intercepts (L : ℝ → ℝ) (P : Point) (hP : P.x = 2 ∧ P.y = 1) (h_equal_intercepts : hasEqualIntercepts L) :
  ∃! (L : ℝ → ℝ), passesThrough L P ∧ hasEqualIntercepts L :=
sorry

end unique_line_equal_intercepts_l59_59410


namespace sum_of_inverses_A_B_C_eq_300_l59_59837

theorem sum_of_inverses_A_B_C_eq_300 
  (p q r : ℝ)
  (hroots : ∀ x, (x^3 - 30*x^2 + 105*x - 114 = 0) → (x = p ∨ x = q ∨ x = r))
  (A B C : ℝ)
  (hdecomp : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    (1 / (s^3 - 30*s^2 + 105*s - 114) = A/(s - p) + B/(s - q) + C/(s - r))) :
  (1 / A) + (1 / B) + (1 / C) = 300 :=
sorry

end sum_of_inverses_A_B_C_eq_300_l59_59837


namespace same_solution_m_iff_m_eq_2_l59_59966

theorem same_solution_m_iff_m_eq_2 (m y : ℝ) (h1 : my - 2 = 4) (h2 : y - 2 = 1) : m = 2 :=
by {
  sorry
}

end same_solution_m_iff_m_eq_2_l59_59966


namespace part_I_part_II_l59_59976

noncomputable def f (x : ℝ) (a : ℝ) (omega : ℝ) : ℝ :=
  2 * a * sin(omega * x) * cos(omega * x) + 2 * sqrt 3 * (cos (omega * x))^2 - sqrt 3

theorem part_I (a omega : ℝ) (h_a : 0 < a) (h_omega : 0 < omega)
  (h_max : ∀ x, f x a omega ≤ 2)
  (h_period : ∃ T > 0, ∀ x, f (x + T) a omega = f x a omega) :
  f x a omega = 2 * sin (2 * x + π / 3) :=
sorry

theorem part_II (alpha : ℝ) (h_falpha : f alpha 1 1 = 4 / 3) :
  sin (4 * alpha + π / 6) = -1 / 9 :=
sorry

end part_I_part_II_l59_59976


namespace find_x_l59_59681

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l59_59681


namespace diagonals_of_octagon_l59_59656

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l59_59656


namespace quadratic_completion_l59_59958

theorem quadratic_completion :
  (∀ x : ℝ, (∃ a h k : ℝ, (x ^ 2 - 2 * x - 1 = a * (x - h) ^ 2 + k) ∧ (a = 1) ∧ (h = 1) ∧ (k = -2))) :=
sorry

end quadratic_completion_l59_59958


namespace right_triangle_arithmetic_sequence_side_length_l59_59141

theorem right_triangle_arithmetic_sequence_side_length :
  ∃ (a b c : ℕ), (a < b ∧ b < c) ∧ (b - a = c - b) ∧ (a^2 + b^2 = c^2) ∧ (b = 81) :=
sorry

end right_triangle_arithmetic_sequence_side_length_l59_59141


namespace tea_mixture_price_l59_59250

theorem tea_mixture_price :
  ∀ (price_A price_B : ℝ) (ratio_A ratio_B : ℝ),
  price_A = 65 →
  price_B = 70 →
  ratio_A = 1 →
  ratio_B = 1 →
  (price_A * ratio_A + price_B * ratio_B) / (ratio_A + ratio_B) = 67.5 :=
by
  intros price_A price_B ratio_A ratio_B h1 h2 h3 h4
  sorry

end tea_mixture_price_l59_59250


namespace perfect_squares_l59_59398

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l59_59398


namespace octagon_diagonals_l59_59659

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l59_59659


namespace cost_of_trip_per_student_l59_59376

def raised_fund : ℕ := 50
def contribution_per_student : ℕ := 5
def num_students : ℕ := 20
def remaining_fund : ℕ := 10

theorem cost_of_trip_per_student :
  ((raised_fund - remaining_fund) / num_students) = 2 := by
  sorry

end cost_of_trip_per_student_l59_59376


namespace find_a_l59_59665

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 :=
by
  sorry

end find_a_l59_59665


namespace probability_non_special_number_l59_59142

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k = n

def total_numbers := 200
def non_special_numbers := 182

theorem probability_non_special_number : 
  ((card {n : ℕ | n ≤ total_numbers ∧ ¬ is_square n ∧ ¬ is_cube n ∧ ¬ is_fifth_power n}) : ℚ) / total_numbers = 91 / 100 :=
sorry

end probability_non_special_number_l59_59142


namespace inequality_solution_set_l59_59868

theorem inequality_solution_set :
  {x : ℝ | 2 * x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1 / 2} :=
by
  sorry

end inequality_solution_set_l59_59868


namespace polynomial_relation_l59_59239

theorem polynomial_relation (x y : ℕ) :
  (x = 1 ∧ y = 1) ∨ 
  (x = 2 ∧ y = 4) ∨ 
  (x = 3 ∧ y = 9) ∨ 
  (x = 4 ∧ y = 16) ∨ 
  (x = 5 ∧ y = 25) → 
  y = x^2 := 
by
  sorry

end polynomial_relation_l59_59239


namespace relative_error_comparison_l59_59616

theorem relative_error_comparison :
  let e₁ := 0.05
  let l₁ := 25.0
  let e₂ := 0.4
  let l₂ := 200.0
  let relative_error (e l : ℝ) : ℝ := (e / l) * 100
  (relative_error e₁ l₁ = relative_error e₂ l₂) :=
by
  sorry

end relative_error_comparison_l59_59616


namespace sum_of_fractions_and_decimal_l59_59730

theorem sum_of_fractions_and_decimal :
  (6 / 5 : ℝ) + (1 / 10 : ℝ) + 1.56 = 2.86 :=
by
  sorry

end sum_of_fractions_and_decimal_l59_59730


namespace max_value_of_f_l59_59202

noncomputable def f (x : Real) : Real :=
  9 * Real.sin x + 12 * Real.cos x

theorem max_value_of_f : ∃ M, ∀ x, f x ≤ M ∧ (∃ y, f y = M) := 
  ∃ M, M = 15 ∧ (∀ x, f x ≤ 15) ∧ (f (Real.atan (4/3)) = 15) :=
sorry

end max_value_of_f_l59_59202


namespace paris_hair_count_paris_hair_count_specific_paris_hair_count_increase_l59_59221

theorem paris_hair_count (num_hairs : ℕ) (num_parisians : ℕ) 
  (h_hairs : num_hairs < 300000) (h_parisians : num_parisians = 3000000) : 
  ∃ (k : ℕ), k ≥ 2 ∧ ∃ S : Finset ℕ, S.card ≥ 2 ∧ ∀ x ∈ S, x < 300000 :=
by admit

theorem paris_hair_count_specific (num_hairs : ℕ) (num_parisians : ℕ)
  (h_hairs : num_hairs < 300000) (h_parisians : num_parisians = 3000000) :
  ∃ (k : ℕ), k ≥ 10 ∧ ∃ S : Finset ℕ, S.card ≥ 10 ∧ ∀ x ∈ S, x < 300000 :=
by admit

theorem paris_hair_count_increase (num_hairs : ℕ) (num_parisians : ℕ)
  (h_hairs : num_hairs < 300000) (h_parisians : num_parisians = 3000001) :
  ∃ (k : ℕ), k ≥ 11 ∧ ∃ S : Finset ℕ, S.card ≥ 11 ∧ ∀ x ∈ S, x < 300000 :=
by admit

end paris_hair_count_paris_hair_count_specific_paris_hair_count_increase_l59_59221


namespace gcd_a_b_l59_59622

def a (n : ℤ) : ℤ := n^5 + 6 * n^3 + 8 * n
def b (n : ℤ) : ℤ := n^4 + 4 * n^2 + 3

theorem gcd_a_b (n : ℤ) : ∃ d : ℤ, d = Int.gcd (a n) (b n) ∧ (d = 1 ∨ d = 3) :=
by
  sorry

end gcd_a_b_l59_59622


namespace digit_B_for_divisibility_by_9_l59_59295

theorem digit_B_for_divisibility_by_9 :
  ∃! (B : ℕ), B < 10 ∧ (5 + B + B + 3) % 9 = 0 :=
by
  sorry

end digit_B_for_divisibility_by_9_l59_59295


namespace simplify_expression_l59_59132

theorem simplify_expression (h : 65536 = 2^16) : 
  (√[4](√[3](√(1 / 65536)))) = 1 / 2^(2/3) :=
by
  sorry

end simplify_expression_l59_59132


namespace purple_to_seafoam_valley_ratio_l59_59716

theorem purple_to_seafoam_valley_ratio (azure_skirts : ℕ) (purple_skirts : ℕ) :
  (azure_skirts = 60) → (purple_skirts = 10) → 
  (S = (2 / 3 : ℚ) * 60) → (S = 40) → 
  (purple_skirts / S = 1 / 4) :=
begin
  intros h_azure h_purple h_seafoam_valley1 h_seafoam_valley2,
  -- Proving the ratio
  sorry
end

end purple_to_seafoam_valley_ratio_l59_59716


namespace total_money_divided_l59_59760

noncomputable def children_share_total (A B E : ℕ) :=
  (12 * A = 8 * B ∧ 8 * B = 6 * E ∧ A = 84) → 
  A + B + E = 378

theorem total_money_divided (A B E : ℕ) : children_share_total A B E :=
by
  intros h
  sorry

end total_money_divided_l59_59760


namespace all_fruits_fallen_by_twelfth_day_l59_59875

noncomputable def magical_tree_falling_day : Nat :=
  let total_fruits := 58
  let initial_day_falls := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].foldl (· + ·) 0
  let continuation_falls := [1, 2].foldl (· + ·) 0
  let total_days := initial_day_falls + continuation_falls
  12

theorem all_fruits_fallen_by_twelfth_day :
  magical_tree_falling_day = 12 :=
by
  sorry

end all_fruits_fallen_by_twelfth_day_l59_59875


namespace sufficient_but_not_necessary_not_necessary_l59_59605

-- Conditions
def condition_1 (x : ℝ) : Prop := x > 3
def condition_2 (x : ℝ) : Prop := x^2 - 5 * x + 6 > 0

-- Theorem statement
theorem sufficient_but_not_necessary (x : ℝ) : condition_1 x → condition_2 x :=
sorry

theorem not_necessary (x : ℝ) : condition_2 x → ∃ y : ℝ, ¬ condition_1 y ∧ condition_2 y :=
sorry

end sufficient_but_not_necessary_not_necessary_l59_59605


namespace octagon_has_20_diagonals_l59_59654

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l59_59654


namespace area_of_triangle_ABF_l59_59348

noncomputable def hyperbola (a b : ℝ) := {p : ℝ × ℝ | p.1^2 - p.2^2 / b^2 = 1}

namespace hyperbola_problem

def a : ℝ := 1
def b : ℝ := √3
def c: ℝ := √(a^2 + b^2)

def vertex_right : ℝ × ℝ := (a, 0)
def focus_right : ℝ × ℝ := (c, 0)

def asymptotes (a b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = b * p.1 ∨ p.2 = -b * p.1}

def line_BF : ℝ → ℝ := λ x, b * (x - 2)

def point_B : ℝ × ℝ := (1, -b)

def area_triangle (A B F : ℝ × ℝ) : ℝ := 0.5 * (abs (A.1 * (B.2 - F.2) + B.1 * (F.2 - A.2) + F.1 * (A.2 - B.2)))

theorem area_of_triangle_ABF :
  let A := vertex_right,
      F := focus_right,
      B := point_B
  in area_triangle A B F = √3 / 2 :=
by
  admit

end area_of_triangle_ABF_l59_59348


namespace total_cows_l59_59314

theorem total_cows (n : ℕ) 
  (h₁ : n / 3 + n / 6 + n / 9 + 8 = n) : n = 144 :=
by sorry

end total_cows_l59_59314


namespace angle_of_inclination_vert_line_l59_59853

theorem angle_of_inclination_vert_line (x : ℝ) (h : x = -1) : 
  ∃ θ : ℝ, θ = 90 := 
by
  sorry

end angle_of_inclination_vert_line_l59_59853


namespace subtraction_result_l59_59426

theorem subtraction_result :
  5.3567 - 2.1456 - 1.0211 = 2.1900 := 
sorry

end subtraction_result_l59_59426


namespace wedding_reception_friends_l59_59762

theorem wedding_reception_friends (total_guests bride_couples groom_couples bride_coworkers groom_coworkers bride_relatives groom_relatives: ℕ)
  (h1: total_guests = 400)
  (h2: bride_couples = 40) 
  (h3: groom_couples = 40)
  (h4: bride_coworkers = 10) 
  (h5: groom_coworkers = 10)
  (h6: bride_relatives = 20)
  (h7: groom_relatives = 20)
  : (total_guests - ((bride_couples + groom_couples) * 2 + (bride_coworkers + groom_coworkers) + (bride_relatives + groom_relatives))) = 180 := 
by 
  sorry

end wedding_reception_friends_l59_59762


namespace cookies_per_person_l59_59748

-- Definitions based on conditions
def cookies_total : ℕ := 144
def people_count : ℕ := 6

-- The goal is to prove the number of cookies per person
theorem cookies_per_person : cookies_total / people_count = 24 :=
by
  sorry

end cookies_per_person_l59_59748


namespace parity_of_solutions_l59_59418

theorem parity_of_solutions
  (n m x y : ℤ)
  (hn : Odd n) 
  (hm : Odd m) 
  (h1 : x + 2 * y = n) 
  (h2 : 3 * x - y = m) :
  Odd x ∧ Even y :=
by
  sorry

end parity_of_solutions_l59_59418


namespace hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l59_59179

theorem hexagon_exists_equal_sides_four_equal_angles : 
  ∃ (A B C D E F : Type) (AB BC CD DE EF FA : ℝ) (angle_A angle_B angle_C angle_D angle_E angle_F : ℝ), 
  (AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB) ∧ 
  (angle_A = angle_B ∧ angle_B = angle_E ∧ angle_E = angle_F) ∧ 
  4 * angle_A + angle_C + angle_D = 720 :=
sorry

theorem hexagon_exists_equal_angles_four_equal_sides :
  ∃ (A B C D E F : Type) (AB BC CD DA : ℝ) (angle : ℝ), 
  (angle_A = angle_B ∧ angle_B = angle_C ∧ angle_C = angle_D ∧ angle_D = angle_E ∧ angle_E = angle_F ∧ angle_F = angle_A) ∧ 
  (AB = BC ∧ BC = CD ∧ CD = DA) :=
sorry

end hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l59_59179


namespace solve_equation_l59_59276

theorem solve_equation :
  ∃ x : ℝ, (x + 2) / 4 - (2 * x - 3) / 6 = 2 ∧ x = -12 :=
by
  sorry

end solve_equation_l59_59276


namespace chord_through_P_midpoint_of_ellipse_has_given_line_l59_59108

-- Define the ellipse
def ellipse (x y : ℝ) := 4 * x^2 + 9 * y^2 = 144

-- Define point P
def pointP := (3, 1)

-- Define the problem statement
theorem chord_through_P_midpoint_of_ellipse_has_given_line:
  ∃ (m : ℝ) (c : ℝ), (∀ (x y : ℝ), 4 * x^2 + 9 * y^2 = 144 → x + y = m ∧ 3 * x + y = c) → 
  ∃ (A : ℝ) (B : ℝ), ellipse 3 1 ∧ (A * 4 + B * 3 - 15 = 0) := sorry

end chord_through_P_midpoint_of_ellipse_has_given_line_l59_59108


namespace octagon_diagonals_l59_59649

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l59_59649


namespace endangered_species_count_l59_59293

section BirdsSanctuary

-- Define the given conditions
def pairs_per_species : ℕ := 7
def total_pairs : ℕ := 203

-- Define the result to be proved
theorem endangered_species_count : total_pairs / pairs_per_species = 29 := by
  sorry

end BirdsSanctuary

end endangered_species_count_l59_59293


namespace no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l59_59623

theorem no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime :
  ¬∃ n : ℕ, 2 ≤ n ∧ Nat.Prime (n^4 + n^2 + 1) :=
sorry

end no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l59_59623


namespace inv_func_eval_l59_59505

theorem inv_func_eval (a : ℝ) (h : 8^(1/3) = a) : (fun y => (Real.log y / Real.log 8)) (a + 2) = 2/3 :=
by
  sorry

end inv_func_eval_l59_59505


namespace xiao_ming_math_score_l59_59599

noncomputable def math_score (C M E : ℕ) : ℕ :=
  let A := 94
  let N := 3
  let total_score := A * N
  let T_CE := (A - 1) * 2
  total_score - T_CE

theorem xiao_ming_math_score (C M E : ℕ)
    (h1 : (C + M + E) / 3 = 94)
    (h2 : (C + E) / 2 = 93) :
  math_score C M E = 96 := by
  sorry

end xiao_ming_math_score_l59_59599


namespace unique_solution_cond_l59_59802

open Real

theorem unique_solution_cond (a c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + 2) ↔ c ≠ 4 :=
by sorry

end unique_solution_cond_l59_59802


namespace problem_solution_l59_59078

variable (a : ℝ)

theorem problem_solution (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end problem_solution_l59_59078


namespace arithmetic_sequence_8th_term_l59_59018

theorem arithmetic_sequence_8th_term 
    (a₁ : ℝ) (a₅ : ℝ) (n : ℕ) (a₈ : ℝ) 
    (h₁ : a₁ = 3) 
    (h₂ : a₅ = 78) 
    (h₃ : n = 25) : 
    a₈ = 24.875 := by
  sorry

end arithmetic_sequence_8th_term_l59_59018


namespace positive_difference_l59_59744

theorem positive_difference (a b : ℕ) (h1 : a = (6^2 + 6^2) / 6) (h2 : b = (6^2 * 6^2) / 6) : a < b ∧ b - a = 204 :=
by
  sorry

end positive_difference_l59_59744


namespace find_erased_number_l59_59154

theorem find_erased_number (x : ℕ) (h : 8 * x = 96) : x = 12 := by
  sorry

end find_erased_number_l59_59154


namespace negation_of_proposition_l59_59411

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ ∃ x : ℝ, Real.exp x ≤ x^2 :=
by sorry

end negation_of_proposition_l59_59411


namespace triangle_area_ABC_l59_59325

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (7, 6)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the triangle with the given vertices is 15
theorem triangle_area_ABC : triangle_area A B C = 15 :=
by
  -- Proof goes here
  sorry

end triangle_area_ABC_l59_59325


namespace solve_weights_problem_l59_59709

variable (a b c d : ℕ) 

def weights_problem := 
  a + b = 280 ∧ 
  a + d = 300 ∧ 
  c + d = 290 → 
  b + c = 270

theorem solve_weights_problem (a b c d : ℕ) : weights_problem a b c d :=
 by
  sorry

end solve_weights_problem_l59_59709


namespace simplify_expression_l59_59567

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l59_59567


namespace train_speed_proof_l59_59882

theorem train_speed_proof : 
  ∀ (V_A V_B : ℝ) (T_A T_B : ℝ), 
  T_A = 9 ∧ 
  T_B = 4 ∧ 
  V_B = 90 ∧ 
  (V_A / V_B = T_B / T_A) → 
  V_A = 40 := 
by
  intros V_A V_B T_A T_B h
  obtain ⟨hT_A, hT_B, hV_B, hprop⟩ := h
  sorry

end train_speed_proof_l59_59882


namespace similar_triangle_perimeter_l59_59105

theorem similar_triangle_perimeter 
  (a b c : ℝ) (ha : a = 12) (hb : b = 12) (hc : c = 24) 
  (k : ℝ) (hk : k = 1.5) : 
  (1.5 * a) + (1.5 * b) + (1.5 * c) = 72 :=
by
  sorry

end similar_triangle_perimeter_l59_59105


namespace cards_traded_between_Padma_and_Robert_l59_59559

def total_cards_traded (padma_first_trade padma_second_trade robert_first_trade robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + padma_second_trade + robert_first_trade + robert_second_trade

theorem cards_traded_between_Padma_and_Robert (h1 : padma_first_trade = 2) 
                                            (h2 : robert_first_trade = 10)
                                            (h3 : padma_second_trade = 15)
                                            (h4 : robert_second_trade = 8) :
                                            total_cards_traded 2 15 10 8 = 35 := 
by 
  sorry

end cards_traded_between_Padma_and_Robert_l59_59559


namespace exact_days_two_friends_visit_l59_59328

-- Define the periodicities of Alice, Beatrix, and Claire
def periodicity_alice : ℕ := 1
def periodicity_beatrix : ℕ := 5
def periodicity_claire : ℕ := 7

-- Define the total days to be considered
def total_days : ℕ := 180

-- Define the number of days three friends visit together
def lcm_ab := Nat.lcm periodicity_alice periodicity_beatrix
def lcm_ac := Nat.lcm periodicity_alice periodicity_claire
def lcm_bc := Nat.lcm periodicity_beatrix periodicity_claire
def lcm_abc := Nat.lcm lcm_ab periodicity_claire

-- Define the counts of visitations
def count_ab := total_days / lcm_ab - total_days / lcm_abc
def count_ac := total_days / lcm_ac - total_days / lcm_abc
def count_bc := total_days / lcm_bc - total_days / lcm_abc

-- Finally calculate the number of days exactly two friends visit together
def days_two_friends_visit : ℕ := count_ab + count_ac + count_bc

-- The theorem to prove
theorem exact_days_two_friends_visit : days_two_friends_visit = 51 :=
by 
  -- This is where the actual proof would go
  sorry

end exact_days_two_friends_visit_l59_59328


namespace solve_x_l59_59385

theorem solve_x :
  (1 / 4 - 1 / 6) = 1 / (12 : ℝ) :=
by sorry

end solve_x_l59_59385


namespace largest_n_l59_59183

theorem largest_n (n x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 6 * x + 6 * y + 6 * z - 18 →
  n ≤ 3 := 
by 
  sorry

end largest_n_l59_59183


namespace original_apples_l59_59443

-- Define the conditions
def sells_50_percent (initial remaining : ℕ) : Prop :=
  (initial / 2) = remaining

-- Define the goal
theorem original_apples (remaining : ℕ) (initial : ℕ) (h : sells_50_percent initial remaining) : initial = 10000 :=
by
  sorry

end original_apples_l59_59443


namespace number_of_5_dollar_bills_l59_59558

def total_money : ℤ := 45
def value_of_each_bill : ℤ := 5

theorem number_of_5_dollar_bills : total_money / value_of_each_bill = 9 := by
  sorry

end number_of_5_dollar_bills_l59_59558


namespace slope_of_line_l59_59979

-- Define the point and the line equation with a generic slope
def point : ℝ × ℝ := (-1, 2)

def line (a : ℝ) := a * (point.fst) + (point.snd) - 4 = 0

-- The main theorem statement
theorem slope_of_line (a : ℝ) (h : line a) : ∃ m : ℝ, m = 2 :=
by
  -- The slope of the line derived from the equation and condition
  sorry

end slope_of_line_l59_59979


namespace probability_interval_l59_59002

noncomputable def eventA : Event Prop := sorry
noncomputable def eventB : Event Prop := sorry

axiom PA : P(eventA) = 5 / 6
axiom PB : P(eventB) = 7 / 8
axiom PA_union_B : P(eventA ∪ eventB) = 13 / 16

theorem probability_interval :
  43 / 48 ≤ P(eventA ∩ eventB) ∧ P(eventA ∩ eventB) ≤ 7 / 8 :=
by
  sorry

end probability_interval_l59_59002


namespace domain_shift_l59_59338

noncomputable def domain := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }
noncomputable def shifted_domain := { x : ℝ | 2 ≤ x ∧ x ≤ 5 }

theorem domain_shift (f : ℝ → ℝ) (h : ∀ x, x ∈ domain ↔ (1 ≤ x ∧ x ≤ 4)) :
  ∀ x, x ∈ shifted_domain ↔ ∃ y, (y = x - 1) ∧ y ∈ domain :=
by
  sorry

end domain_shift_l59_59338


namespace determine_c_for_quadratic_eq_l59_59697

theorem determine_c_for_quadratic_eq (x1 x2 c : ℝ) 
  (h1 : x1 + x2 = 2)
  (h2 : x1 * x2 = c)
  (h3 : 7 * x2 - 4 * x1 = 47) : 
  c = -15 :=
sorry

end determine_c_for_quadratic_eq_l59_59697


namespace arithmetic_sequence_l59_59871

theorem arithmetic_sequence (S : ℕ → ℕ) (h : ∀ n, S n = 3 * n * n) :
  (∃ a d : ℕ, ∀ n : ℕ, S n - S (n - 1) = a + (n - 1) * d) ∧
  (∀ n, S n - S (n - 1) = 6 * n - 3) :=
by
  sorry

end arithmetic_sequence_l59_59871


namespace unique_solution_l59_59620

theorem unique_solution : ∀ (x y z : ℕ), 
  x > 0 → y > 0 → z > 0 → 
  x^2 = 2 * (y + z) → 
  x^6 = y^6 + z^6 + 31 * (y^2 + z^2) → 
  (x, y, z) = (2, 1, 1) :=
by sorry

end unique_solution_l59_59620


namespace rick_total_clothes_ironed_l59_59274

def rick_ironing_pieces
  (shirts_per_hour : ℕ)
  (pants_per_hour : ℕ)
  (hours_shirts : ℕ)
  (hours_pants : ℕ) : ℕ :=
  (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants)

theorem rick_total_clothes_ironed :
  rick_ironing_pieces 4 3 3 5 = 27 :=
by
  sorry

end rick_total_clothes_ironed_l59_59274


namespace line_equation_is_correct_l59_59140

noncomputable def line_has_equal_intercepts_and_passes_through_A (p q : ℝ) : Prop :=
(p, q) = (3, 2) ∧ q ≠ 0 ∧ (∃ c : ℝ, p + q = c ∨ 2 * p - 3 * q = 0)

theorem line_equation_is_correct :
  line_has_equal_intercepts_and_passes_through_A 3 2 → 
  (∃ f g : ℝ, 2 * f - 3 * g = 0 ∨ f + g = 5) :=
by
  sorry

end line_equation_is_correct_l59_59140


namespace find_children_tickets_l59_59588

variable (A C S : ℝ)

theorem find_children_tickets 
  (h1 : A + C + S = 600)
  (h2 : 6 * A + 4.5 * C + 5 * S = 3250) :
  C = (350 - S) / 1.5 := 
sorry

end find_children_tickets_l59_59588


namespace triangle_area_is_integer_l59_59733

theorem triangle_area_is_integer (x1 x2 x3 y1 y2 y3 : ℤ) 
  (hx_even : (x1 + x2 + x3) % 2 = 0) 
  (hy_even : (y1 + y2 + y3) % 2 = 0) : 
  ∃ k : ℤ, 
    abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 2 * k := 
sorry

end triangle_area_is_integer_l59_59733


namespace total_number_of_employees_l59_59166
  
def part_time_employees : ℕ := 2041
def full_time_employees : ℕ := 63093
def total_employees : ℕ := part_time_employees + full_time_employees

theorem total_number_of_employees : total_employees = 65134 := by
  sorry

end total_number_of_employees_l59_59166


namespace parabola_directrix_l59_59280

theorem parabola_directrix (x y : ℝ) : 
    (x^2 = (1/2) * y) -> (y = -1/8) :=
sorry

end parabola_directrix_l59_59280


namespace mr_a_net_gain_l59_59712

theorem mr_a_net_gain 
  (initial_value : ℝ)
  (sale_profit_percentage : ℝ)
  (buyback_loss_percentage : ℝ)
  (final_sale_price : ℝ) 
  (buyback_price : ℝ)
  (net_gain : ℝ) :
  initial_value = 12000 →
  sale_profit_percentage = 0.15 →
  buyback_loss_percentage = 0.12 →
  final_sale_price = initial_value * (1 + sale_profit_percentage) →
  buyback_price = final_sale_price * (1 - buyback_loss_percentage) →
  net_gain = final_sale_price - buyback_price →
  net_gain = 1656 :=
by
  sorry

end mr_a_net_gain_l59_59712


namespace walnut_swap_exists_l59_59292

theorem walnut_swap_exists (n : ℕ) (h_n : n = 2021) :
  ∃ k : ℕ, k < n ∧ ∃ a b : ℕ, a < k ∧ k < b :=
by
  sorry

end walnut_swap_exists_l59_59292


namespace correct_statements_truth_of_statements_l59_59453

-- Define basic properties related to factor and divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Given conditions as definitions
def condition_A : Prop := is_factor 4 100
def condition_B1 : Prop := is_divisor 19 133
def condition_B2 : Prop := ¬ is_divisor 19 51
def condition_C1 : Prop := is_divisor 30 90
def condition_C2 : Prop := ¬ is_divisor 30 53
def condition_D1 : Prop := is_divisor 7 21
def condition_D2 : Prop := ¬ is_divisor 7 49
def condition_E : Prop := is_factor 10 200

-- Statement that needs to be proved
theorem correct_statements : 
  (condition_A ∧ 
  (condition_B1 ∧ condition_B2) ∧ 
  condition_E) :=
by sorry -- proof to be inserted

-- Equivalent Lean 4 statement with all conditions encapsulated
theorem truth_of_statements :
  (is_factor 4 100) ∧ 
  (is_divisor 19 133 ∧ ¬ is_divisor 19 51) ∧ 
  is_factor 10 200 :=
by sorry -- proof to be inserted

end correct_statements_truth_of_statements_l59_59453


namespace number_of_diagonals_in_octagon_l59_59644

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l59_59644


namespace perfect_squares_l59_59400

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l59_59400


namespace octagon_diagonals_20_l59_59652

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l59_59652


namespace overtime_hours_proof_l59_59915

-- Define the conditions
variable (regular_pay_rate : ℕ := 3)
variable (regular_hours : ℕ := 40)
variable (overtime_multiplier : ℕ := 2)
variable (total_pay : ℕ := 180)

-- Calculate the regular pay for 40 hours
def regular_pay : ℕ := regular_pay_rate * regular_hours

-- Calculate the extra pay received beyond regular pay
def extra_pay : ℕ := total_pay - regular_pay

-- Calculate overtime pay rate
def overtime_pay_rate : ℕ := overtime_multiplier * regular_pay_rate

-- Calculate the number of overtime hours
def overtime_hours (extra_pay : ℕ) (overtime_pay_rate : ℕ) : ℕ :=
  extra_pay / overtime_pay_rate

-- The theorem to prove
theorem overtime_hours_proof :
  overtime_hours extra_pay overtime_pay_rate = 10 := by
  sorry

end overtime_hours_proof_l59_59915


namespace larry_wins_prob_l59_59113

def probability_larry_wins (pLarry pJulius : ℚ) : ℚ :=
  let r := (1 - pLarry) * (1 - pJulius)
  pLarry * (1 / (1 - r))

theorem larry_wins_prob : probability_larry_wins (2 / 3) (1 / 3) = 6 / 7 :=
by
  -- Definitions for probabilities
  let pLarry := 2 / 3
  let pJulius := 1 / 3
  have r := (1 - pLarry) * (1 - pJulius)
  have S := pLarry * (1 / (1 - r))
  -- Expected result
  have expected := 6 / 7
  -- Prove the result equals the expected
  sorry

end larry_wins_prob_l59_59113


namespace chord_constant_sum_l59_59121

theorem chord_constant_sum (d : ℝ) (h : d = 1/2) :
  ∀ A B : ℝ × ℝ, (A.2 = A.1^2) → (B.2 = B.1^2) →
  (∃ m : ℝ, A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →
  (∃ D : ℝ × ℝ, D = (0, d) ∧ (∃ s : ℝ,
    s = (1 / ((A.1 - D.1)^2 + (A.2 - D.2)^2) + 1 / ((B.1 - D.1)^2 + (B.2 - D.2)^2)) ∧ s = 4)) :=
by 
  sorry

end chord_constant_sum_l59_59121


namespace percentage_increase_in_length_l59_59860

theorem percentage_increase_in_length (L B : ℝ) (hB : 0 < B) (hL : 0 < L) :
  (1 + x / 100) * 1.22 = 1.3542 -> x = 11.016393 :=
by
  sorry

end percentage_increase_in_length_l59_59860


namespace probability_first_prize_l59_59321

-- Define the total number of tickets
def total_tickets : ℕ := 150

-- Define the number of first prizes
def first_prizes : ℕ := 5

-- Define the probability calculation as a theorem
theorem probability_first_prize : (first_prizes : ℚ) / total_tickets = 1 / 30 := 
by sorry  -- Placeholder for the proof

end probability_first_prize_l59_59321


namespace max_cables_used_eq_375_l59_59929

-- Conditions for the problem
def total_employees : Nat := 40
def brand_A_computers : Nat := 25
def brand_B_computers : Nat := 15

-- The main theorem we want to prove
theorem max_cables_used_eq_375 
  (h_employees : total_employees = 40)
  (h_brand_A_computers : brand_A_computers = 25)
  (h_brand_B_computers : brand_B_computers = 15)
  (cables_connectivity : ∀ (a : Fin brand_A_computers) (b : Fin brand_B_computers), Prop)
  (no_initial_connections : ∀ (a : Fin brand_A_computers) (b : Fin brand_B_computers), ¬ cables_connectivity a b)
  (each_brand_B_connected : ∀ (b : Fin brand_B_computers), ∃ (a : Fin brand_A_computers), cables_connectivity a b)
  : ∃ (n : Nat), n = 375 := 
sorry

end max_cables_used_eq_375_l59_59929


namespace simplify_expression_l59_59570

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l59_59570


namespace fraction_given_away_is_three_fifths_l59_59124

variable (initial_bunnies : ℕ) (final_bunnies : ℕ) (kittens_per_bunny : ℕ)

def fraction_given_away (given_away : ℕ) (initial_bunnies : ℕ) : ℚ :=
  given_away / initial_bunnies

theorem fraction_given_away_is_three_fifths 
  (initial_bunnies : ℕ := 30) (final_bunnies : ℕ := 54) (kittens_per_bunny : ℕ := 2)
  (h : final_bunnies = initial_bunnies + kittens_per_bunny * (initial_bunnies - 18)) : 
  fraction_given_away 18 initial_bunnies = 3 / 5 :=
by
  sorry

end fraction_given_away_is_three_fifths_l59_59124


namespace motorcyclist_travel_distances_l59_59317

-- Define the total distance traveled in three days
def total_distance : ℕ := 980

-- Define the total distance traveled in the first two days
def first_two_days_distance : ℕ := 725

-- Define the extra distance traveled on the second day compared to the third day
def second_day_extra : ℕ := 123

-- Define the distances traveled on the first, second, and third days respectively
def day_1_distance : ℕ := 347
def day_2_distance : ℕ := 378
def day_3_distance : ℕ := 255

-- Formalize the theorem statement
theorem motorcyclist_travel_distances :
  total_distance = day_1_distance + day_2_distance + day_3_distance ∧
  first_two_days_distance = day_1_distance + day_2_distance ∧
  day_2_distance = day_3_distance + second_day_extra :=
by 
  sorry

end motorcyclist_travel_distances_l59_59317


namespace oreo_shop_problem_l59_59772

/-- Number of different ways Charlie and Delta can leave the store with 4 products collectively. -/
theorem oreo_shop_problem : 
  let oreo_flavors := 7
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  let charlie_products := 4
  ∑ i in Finset.range (charlie_products+1), 
    (Nat.choose total_flavors (charlie_products - i) * 
     (if i = 0 then 1 else (Nat.choose oreo_flavors i * (Multiset.combos i 1 (charlie_products - i)).card)) ) = 4054 :=
by
  let oreo_flavors := 7
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  let charlie_products := 4
  let case0 := Nat.choose total_flavors 4
  let case1 := Nat.choose total_flavors 3 * oreo_flavors
  let case2 := Nat.choose total_flavors 2 * (Nat.choose oreo_flavors 2 + oreo_flavors)
  let case3 := Nat.choose total_flavors 1 * (Nat.choose oreo_flavors 3 + oreo_flavors * oreo_flavors + oreo_flavors)
  let case4 := (Nat.choose oreo_flavors 4 + oreo_flavors * (oreo_flavors - 1) + (oreo_flavors * (oreo_flavors - 1)) / 2 + oreo_flavors)
  let total_ways := case0 + case1 + case2 + case3 + case4
  have : total_ways = 4054 := by 
    sorry 
  exact this

end oreo_shop_problem_l59_59772


namespace triangle_area_ratios_l59_59027

theorem triangle_area_ratios (K : ℝ) 
  (hCD : ∃ AC, ∃ CD, CD = AC / 4) 
  (hAE : ∃ AB, ∃ AE, AE = AB / 5) 
  (hBF : ∃ BC, ∃ BF, BF = BC / 3) :
  ∃ area_N1N2N3, area_N1N2N3 = (8 / 15) * K :=
by
  sorry

end triangle_area_ratios_l59_59027


namespace initial_fraction_spent_on_clothes_l59_59167

-- Define the conditions and the theorem to be proved
theorem initial_fraction_spent_on_clothes 
  (M : ℝ) (F : ℝ)
  (h1 : M = 249.99999999999994)
  (h2 : (3 / 4) * (4 / 5) * (1 - F) * M = 100) :
  F = 11 / 15 :=
sorry

end initial_fraction_spent_on_clothes_l59_59167


namespace perimeter_of_garden_l59_59390

def area (length width : ℕ) : ℕ := length * width

def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

theorem perimeter_of_garden :
  ∀ (l w : ℕ), area l w = 28 ∧ l = 7 → perimeter l w = 22 := by
  sorry

end perimeter_of_garden_l59_59390


namespace ascending_descending_sixth_ascending_descending_fifth_l59_59944

/-- Consider seven-digit natural numbers in which each of the digits 1, 2, 3, 4, 5, 6, 7 appears exactly once. 
(a) Prove that there are exactly 6 such numbers where the first six digits are in ascending order 
    and the sixth and seventh digits are in descending order. -/
theorem ascending_descending_sixth {α: Type*} [Fintype α] [DecidableEq α] : 
    ∀ (s : Finset ℕ), s.card = 7 → (s = ({1, 2, 3, 4, 5, 6, 7}: Finset ℕ)) → 
    ∃! (l : List ℕ), l.length = 7 ∧ (l.take 6).Ascending ∧ (l.drop 5).Descending :=
by sorry

/-- Consider seven-digit natural numbers in which each of the digits 1, 2, 3, 4, 5, 6, 7 appears exactly once. 
(b) Prove that there are exactly 15 such numbers where the first five digits are in ascending order 
    and the fifth to seventh digits are in descending order. -/
theorem ascending_descending_fifth {α: Type*} [Fintype α] [DecidableEq α] : 
    ∀ (s : Finset ℕ), s.card = 7 → (s = ({1, 2, 3, 4, 5, 6, 7}: Finset ℕ)) → 
    ∃! (l : List ℕ), l.length = 7 ∧ (l.take 5).Ascending ∧ (l.drop 4).Descending :=
by sorry

end ascending_descending_sixth_ascending_descending_fifth_l59_59944


namespace find_time_to_fill_tank_l59_59914

noncomputable def time_to_fill_tanker (TA : ℝ) : Prop :=
  let RB := 1 / 40
  let fill_time := 29.999999999999993
  let half_fill_time := fill_time / 2
  let RAB := (1 / TA) + RB
  (RAB * half_fill_time = 1 / 2) → (TA = 120)

theorem find_time_to_fill_tank : ∃ TA, time_to_fill_tanker TA :=
by
  use 120
  sorry

end find_time_to_fill_tank_l59_59914


namespace product_of_differences_l59_59499

theorem product_of_differences (p q p' q' α β α' β' : ℝ)
  (h1 : α + β = -p) (h2 : α * β = q)
  (h3 : α' + β' = -p') (h4 : α' * β' = q') :
  ((α - α') * (α - β') * (β - α') * (β - β') = (q - q')^2 + (p - p') * (q' * p - p' * q)) :=
sorry

end product_of_differences_l59_59499


namespace range_of_m_for_log_function_domain_l59_59856

theorem range_of_m_for_log_function_domain (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + m > 0) → m > 8 :=
by
  sorry

end range_of_m_for_log_function_domain_l59_59856


namespace eliot_votes_l59_59518

theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ)
                    (h1 : randy_votes = 16)
                    (h2 : shaun_votes = 5 * randy_votes)
                    (h3 : eliot_votes = 2 * shaun_votes) :
                    eliot_votes = 160 :=
by {
  -- Proof will be conducted here
  sorry
}

end eliot_votes_l59_59518


namespace moles_HBr_formed_l59_59476

theorem moles_HBr_formed 
    (moles_CH4 : ℝ) (moles_Br2 : ℝ) (reaction : ℝ) : 
    moles_CH4 = 1 ∧ moles_Br2 = 1 → reaction = 1 :=
by
  intros h
  cases h
  sorry

end moles_HBr_formed_l59_59476


namespace sequence_product_mod_five_l59_59963

theorem sequence_product_mod_five : 
  let seq := List.range 20 |>.map (λ k => 10 * k + 3)
  seq.prod % 5 = 1 := 
by
  sorry

end sequence_product_mod_five_l59_59963


namespace math_problem_l59_59972

theorem math_problem
  (a b c : ℚ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( (a^2 * b^2 / c^2 - 2 / c + 1 / (a^2 * b^2) + 2 * a * b / c^2 - 2 / (a * b * c))
    / (2 / (a * b) - 2 * a * b / c)
    / (101 / c)
  ) = -1 / 202 := 
sorry

end math_problem_l59_59972


namespace polar_circle_equation_l59_59515

theorem polar_circle_equation {r : ℝ} {phi : ℝ} {rho theta : ℝ} :
  (r = 2) → (phi = π / 3) → (rho = 4 * Real.cos (theta - π / 3)) :=
by
  intros hr hphi
  sorry

end polar_circle_equation_l59_59515


namespace buns_problem_l59_59378

theorem buns_problem (N : ℕ) (x y u v : ℕ) 
  (h1 : 3 * x + 5 * y = 25)
  (h2 : 3 * u + 5 * v = 35)
  (h3 : x + y = N)
  (h4 : u + v = N) : 
  N = 7 := 
sorry

end buns_problem_l59_59378


namespace find_integer_l59_59994

noncomputable def least_possible_sum (x y z k : ℕ) : Prop :=
  2 * x = 5 * y ∧ 5 * y = 6 * z ∧ x + k + z = 26

theorem find_integer (x y z : ℕ) (h : least_possible_sum x y z 6) :
  6 = (26 - x - z) :=
  by {
    sorry
  }

end find_integer_l59_59994


namespace minimum_value_of_function_l59_59095

theorem minimum_value_of_function (x : ℝ) (h : x > 1) : 
  (x + (1 / x) + (16 * x) / (x^2 + 1)) ≥ 8 :=
sorry

end minimum_value_of_function_l59_59095


namespace diagonals_of_octagon_l59_59657

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l59_59657


namespace solve_for_a_l59_59667

-- Given conditions
variables (a b d : ℕ)
hypotheses
  (h1 : a + b = d)
  (h2 : b + d = 7)
  (h3 : d = 4)

-- Prove that a = 1
theorem solve_for_a : a = 1 :=
by {
  sorry
}

end solve_for_a_l59_59667


namespace maximum_value_y_eq_x_plus_cosx_in_0_pi_over_2_l59_59475

open Real

theorem maximum_value_y_eq_x_plus_cosx_in_0_pi_over_2 :
  ∃ (x : ℝ), x ∈ Icc 0 (π / 2) ∧ (∀ y ∈ Icc 0 (π / 2), y + cos y ≤ x + cos x) ∧ x = π / 2 := 
sorry

end maximum_value_y_eq_x_plus_cosx_in_0_pi_over_2_l59_59475


namespace ratio_of_areas_l59_59727

theorem ratio_of_areas (s L : ℝ) (h1 : (π * L^2) / (π * s^2) = 9 / 4) : L - s = (1/2) * s :=
by
  sorry

end ratio_of_areas_l59_59727


namespace workshop_total_number_of_workers_l59_59033

theorem workshop_total_number_of_workers
  (average_salary_all : ℝ)
  (average_salary_technicians : ℝ)
  (average_salary_non_technicians : ℝ)
  (num_technicians : ℕ)
  (total_salary_all : ℝ -> ℝ)
  (total_salary_technicians : ℕ -> ℝ)
  (total_salary_non_technicians : ℕ -> ℝ -> ℝ)
  (h1 : average_salary_all = 9000)
  (h2 : average_salary_technicians = 12000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : ∀ W, total_salary_all W = average_salary_all * W )
  (h6 : ∀ n, total_salary_technicians n = n * average_salary_technicians )
  (h7 : ∀ n W, total_salary_non_technicians n W = (W - n) * average_salary_non_technicians)
  (h8 : ∀ W, total_salary_all W = total_salary_technicians num_technicians + total_salary_non_technicians num_technicians W) :
  ∃ W, W = 14 :=
by
  sorry

end workshop_total_number_of_workers_l59_59033


namespace Andy_late_minutes_l59_59050

theorem Andy_late_minutes 
  (school_start : Nat := 8*60) -- 8:00 AM in minutes since midnight
  (normal_travel_time : Nat := 30) -- 30 minutes
  (red_light_stops : Nat := 3 * 4) -- 3 minutes each at 4 lights
  (construction_wait : Nat := 10) -- 10 minutes
  (detour_time : Nat := 7) -- 7 minutes
  (store_stop_time : Nat := 5) -- 5 minutes
  (traffic_delay : Nat := 15) -- 15 minutes
  (departure_time : Nat := 7*60 + 15) -- 7:15 AM in minutes since midnight
  : 34 = departure_time + normal_travel_time + red_light_stops + construction_wait + detour_time + store_stop_time + traffic_delay - school_start := 
by sorry

end Andy_late_minutes_l59_59050


namespace chickens_in_coop_l59_59291

theorem chickens_in_coop (C : ℕ)
  (H1 : ∃ C : ℕ, ∀ R : ℕ, R = 2 * C)
  (H2 : ∃ R : ℕ, ∀ F : ℕ, F = 2 * R - 4)
  (H3 : ∃ F : ℕ, F = 52) :
  C = 14 :=
by sorry

end chickens_in_coop_l59_59291


namespace part1_part2_l59_59226

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem part1 (x : ℝ) : |f (-x)| + |f x| ≥ 4 * |x| := 
by
  sorry

theorem part2 (x a : ℝ) (h : |x - a| < 1 / 2) : |f x - f a| < |a| + 5 / 4 := 
by
  sorry

end part1_part2_l59_59226


namespace edge_length_of_small_cube_l59_59793

-- Define the parameters
def volume_cube : ℕ := 1000
def num_small_cubes : ℕ := 8
def remaining_volume : ℕ := 488

-- Define the main theorem
theorem edge_length_of_small_cube (x : ℕ) :
  (volume_cube - num_small_cubes * x^3 = remaining_volume) → x = 4 := 
by 
  sorry

end edge_length_of_small_cube_l59_59793


namespace new_socks_bought_l59_59267

-- Given conditions:
def initial_socks : ℕ := 11
def socks_thrown_away : ℕ := 4
def final_socks : ℕ := 33

-- Theorem proof statement:
theorem new_socks_bought : (final_socks - (initial_socks - socks_thrown_away)) = 26 :=
by
  sorry

end new_socks_bought_l59_59267


namespace sequence_value_a10_l59_59810

theorem sequence_value_a10 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 2^n) : a 10 = 1023 := by
  sorry

end sequence_value_a10_l59_59810


namespace velocity_divides_trapezoid_area_l59_59107

theorem velocity_divides_trapezoid_area (V U k : ℝ) (h : ℝ) :
  let W := (V^2 + k * U^2) / (k + 1) in 
  W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end velocity_divides_trapezoid_area_l59_59107


namespace multiples_of_5_between_100_and_400_l59_59091

theorem multiples_of_5_between_100_and_400 : 
  ∃ n : ℕ, n = 60 ∧ ∀ k, (100 ≤ 5 * k ∧ 5 * k ≤ 400) ↔ (21 ≤ k ∧ k ≤ 80) :=
by
  sorry

end multiples_of_5_between_100_and_400_l59_59091


namespace ball_bounces_to_less_than_two_feet_l59_59903

noncomputable def bounce_height (n : ℕ) : ℝ := 20 * (3 / 4) ^ n

theorem ball_bounces_to_less_than_two_feet : ∃ k : ℕ, bounce_height k < 2 ∧ k = 7 :=
by
  -- We need to show that bounce_height k < 2 when k = 7
  sorry

end ball_bounces_to_less_than_two_feet_l59_59903


namespace bricks_of_other_types_l59_59265

theorem bricks_of_other_types (A B total other: ℕ) (hA: A = 40) (hB: B = A / 2) (hTotal: total = 150) (hSum: total = A + B + other): 
  other = 90 :=
by sorry

end bricks_of_other_types_l59_59265


namespace object_speed_approx_l59_59816

theorem object_speed_approx :
  ∃ (speed : ℝ), abs (speed - 27.27) < 0.01 ∧
  (∀ (d : ℝ) (t : ℝ)
    (m : ℝ), 
    d = 80 ∧ t = 2 ∧ m = 5280 →
    speed = (d / m) / (t / 3600)) :=
by 
  sorry

end object_speed_approx_l59_59816


namespace votes_for_eliot_l59_59519

theorem votes_for_eliot (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h_randy : randy_votes = 16)
  (h_shaun : shaun_votes = 5 * randy_votes)
  (h_eliot : eliot_votes = 2 * shaun_votes) :
  eliot_votes = 160 :=
by
  sorry

end votes_for_eliot_l59_59519


namespace winning_percentage_is_62_l59_59009

-- Definitions based on given conditions
def candidate_winner_votes : ℕ := 992
def candidate_win_margin : ℕ := 384
def total_votes : ℕ := candidate_winner_votes + (candidate_winner_votes - candidate_win_margin)

-- The key proof statement
theorem winning_percentage_is_62 :
  ((candidate_winner_votes : ℚ) / total_votes) * 100 = 62 := 
sorry

end winning_percentage_is_62_l59_59009


namespace flour_needed_l59_59375

-- Define the given conditions
def F_total : ℕ := 9
def F_added : ℕ := 3

-- State the main theorem to be proven
theorem flour_needed : (F_total - F_added) = 6 := by
  sorry -- Placeholder for the proof

end flour_needed_l59_59375


namespace find_k_l59_59225

theorem find_k (k : ℝ) 
  (h1 : ∀ (r s : ℝ), r + s = -k ∧ r * s = 8 → (r + 3) + (s + 3) = k) : 
  k = 3 :=
by
  sorry

end find_k_l59_59225


namespace sum_of_distinct_real_numbers_l59_59548

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l59_59548


namespace ball_distribution_l59_59846

theorem ball_distribution :
  ∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ x1 x2 x3, f x1 x2 x3 → x1 + x2 + x3 = 10 ∧ x1 ≥ 1 ∧ x2 ≥ 2 ∧ x3 ≥ 3) ∧
    (∃ (count : ℕ), (count = 15) ∧ (∀ x1 x2 x3, f x1 x2 x3 → count = 15)) :=
sorry

end ball_distribution_l59_59846


namespace limit_example_l59_59034

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - 11| ∧ |x - 11| < δ) → |(2 * x^2 - 21 * x - 11) / (x - 11) - 23| < ε :=
by
  sorry

end limit_example_l59_59034


namespace range_of_a_l59_59502

open Set

def real_intervals (a : ℝ) : Prop :=
  let S := {x : ℝ | (x - 2)^2 > 9}
  let T := Ioo a (a + 8)
  S ∪ T = univ → -3 < a ∧ a < -1

theorem range_of_a (a : ℝ) : real_intervals a :=
sorry

end range_of_a_l59_59502


namespace solve_equation_l59_59067

theorem solve_equation :
  ∃ x : ℚ, (x = 165 / 8) ∧ (∛(5 - x) = -(5 / 2)) := 
sorry

end solve_equation_l59_59067


namespace product_equals_eight_l59_59182

theorem product_equals_eight : (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7) = 8 := 
sorry

end product_equals_eight_l59_59182


namespace simplify_radical_expression_l59_59323

variable (q : ℝ)
variable (hq : q > 0)

theorem simplify_radical_expression :
  (sqrt(42 * q) * sqrt(7 * q) * sqrt(3 * q) = 21 * q * sqrt(2 * q)) :=
by
  sorry

end simplify_radical_expression_l59_59323


namespace simplify_and_evaluate_expression_l59_59434

theorem simplify_and_evaluate_expression : 
  ∀ x : ℝ, x = 1 → ( (x^2 - 5) / (x - 3) - 4 / (x - 3) ) = 4 :=
by
  intros x hx
  simp [hx]
  have eq : (1 * 1 - 5) = -4 := by norm_num -- Verify that the expression simplifies correctly
  sorry -- Skip the actual complex proof steps

end simplify_and_evaluate_expression_l59_59434


namespace triangle_altitude_l59_59854

theorem triangle_altitude (A b : ℝ) (h : ℝ) 
  (hA : A = 750) 
  (hb : b = 50) 
  (area_formula : A = (1 / 2) * b * h) : 
  h = 30 :=
  sorry

end triangle_altitude_l59_59854


namespace dense_set_count_l59_59168

-- Define the notion of a dense set based on the given conditions
def isDenseSet (A : Set ℕ) : Prop :=
  (1 ∈ A ∧ 49 ∈ A) ∧ (∀ n ∈ A, n ≥ 1 ∧ n ≤ 49) ∧ (A.card > 40) ∧ (∀ n ∈ A, (n + 1) ∈ A → (n + 2) ∈ A → (n + 3) ∈ A → (n + 4) ∈ A → (n + 5) ∈ A → (n + 6) ∉ A)

-- State the theorem indicating the number of such sets
theorem dense_set_count : ∃ S : Finset (Set ℕ), S.card = 495 ∧ ∀ A ∈ S, isDenseSet A :=
sorry

end dense_set_count_l59_59168


namespace number_of_diagonals_in_octagon_l59_59645

theorem number_of_diagonals_in_octagon :
  let n : ℕ := 8
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 20 := by
  sorry

end number_of_diagonals_in_octagon_l59_59645


namespace find_a_plus_b_l59_59415

theorem find_a_plus_b (a b : ℝ) (h_sum : 2 * a = -6) (h_prod : a^2 - b = 1) : a + b = 5 :=
by {
  -- Proof would go here; we assume the theorem holds true.
  sorry
}

end find_a_plus_b_l59_59415


namespace circle_diameter_l59_59912

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end circle_diameter_l59_59912


namespace sum_f_1_to_1990_l59_59696

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem sum_f_1_to_1990 : (Finset.range 1990).sum f = 1326 :=
  by
  sorry

end sum_f_1_to_1990_l59_59696


namespace predicted_yield_of_rice_l59_59818

theorem predicted_yield_of_rice (x : ℝ) (h : x = 80) : 5 * x + 250 = 650 :=
by {
  sorry -- proof will be given later
}

end predicted_yield_of_rice_l59_59818


namespace pens_count_l59_59145

theorem pens_count (N P : ℕ) (h1 : N = 40) (h2 : P / N = 5 / 4) : P = 50 :=
by
  sorry

end pens_count_l59_59145


namespace negation_exists_negation_proposition_l59_59283

theorem negation_exists (P : ℝ → Prop) :
  (∃ x : ℝ, P x) ↔ ¬ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end negation_exists_negation_proposition_l59_59283


namespace find_n_l59_59203

theorem find_n (n : ℕ) (h1 : 0 < n) : 
  ∃ n, n > 0 ∧ (Real.tan (Real.pi / (2 * n)) + Real.sin (Real.pi / (2 * n)) = n / 3) := 
sorry

end find_n_l59_59203


namespace edge_length_of_cut_off_cube_l59_59794

theorem edge_length_of_cut_off_cube (V_large V_remaining : ℕ) 
  (h1 : V_large = 1000) (h2 : V_remaining = 488) : 
  ∃ x : ℕ, x ^ 3 = 64 ∧ 1000 - 8 * x ^ 3 = V_remaining := by
    use 4
    split
    · norm_num
    · exact h2.symm
appointment

end edge_length_of_cut_off_cube_l59_59794


namespace sum_of_angles_l59_59827

theorem sum_of_angles : 
    ∀ (angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC : ℝ),
    angle1 + angle3 + angle5 = 180 ∧
    angle2 + angle4 + angle6 = 180 ∧
    angleA + angleB + angleC = 180 →
    angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angleA + angleB + angleC = 540 :=
by
  intro angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC
  intro h
  sorry

end sum_of_angles_l59_59827


namespace probability_odd_sum_is_correct_l59_59072

-- Define the set of the first twelve prime numbers.
def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Define the problem statement.
noncomputable def probability_odd_sum : ℚ :=
  let even_prime_count := 1
  let odd_prime_count := 11
  let ways_to_pick_1_even_and_4_odd := (Nat.choose odd_prime_count 4)
  let total_ways := Nat.choose 12 5
  (ways_to_pick_1_even_and_4_odd : ℚ) / total_ways

theorem probability_odd_sum_is_correct :
  probability_odd_sum = 55 / 132 :=
by
  sorry

end probability_odd_sum_is_correct_l59_59072


namespace p_plus_q_l59_59118

-- Define the circles w1 and w2
def circle1 (x y : ℝ) := x^2 + y^2 + 10*x - 20*y - 77 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 10*x - 20*y + 193 = 0

-- Define the line condition
def line (a x y : ℝ) := y = a * x

-- Prove that p + q = 85, where m^2 = p / q and m is the smallest positive a
theorem p_plus_q : ∃ p q : ℕ, (p.gcd q = 1) ∧ (m^2 = (p : ℝ)/(q : ℝ)) ∧ (p + q = 85) :=
  sorry

end p_plus_q_l59_59118


namespace square_87_l59_59775

theorem square_87 : 87^2 = 7569 :=
by
  sorry

end square_87_l59_59775


namespace isosceles_right_triangle_leg_hypotenuse_ratio_l59_59776

theorem isosceles_right_triangle_leg_hypotenuse_ratio (a d k : ℝ) 
  (h_iso : d = a * Real.sqrt 2)
  (h_ratio : k = a / d) : 
  k^2 = 1 / 2 := by sorry

end isosceles_right_triangle_leg_hypotenuse_ratio_l59_59776


namespace price_white_stamp_l59_59384

variable (price_per_white_stamp : ℝ)

theorem price_white_stamp (simon_red_stamps : ℕ)
                          (peter_white_stamps : ℕ)
                          (price_per_red_stamp : ℝ)
                          (money_difference : ℝ)
                          (h1 : simon_red_stamps = 30)
                          (h2 : peter_white_stamps = 80)
                          (h3 : price_per_red_stamp = 0.50)
                          (h4 : money_difference = 1) :
    money_difference = peter_white_stamps * price_per_white_stamp - simon_red_stamps * price_per_red_stamp →
    price_per_white_stamp = 1 / 5 :=
by
  intros
  sorry

end price_white_stamp_l59_59384


namespace student_arrangement_l59_59824

theorem student_arrangement :
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  valid_arrangements = 336 :=
by
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  exact sorry

end student_arrangement_l59_59824


namespace num_diagonals_octagon_l59_59642

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l59_59642


namespace diane_bakes_gingerbreads_l59_59953

open Nat

theorem diane_bakes_gingerbreads :
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  total_gingerbreads1 + total_gingerbreads2 = 160 := 
by
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  exact Eq.refl (total_gingerbreads1 + total_gingerbreads2)

end diane_bakes_gingerbreads_l59_59953


namespace smallest_option_l59_59205

-- Define the problem with the given condition
def x : ℕ := 10

-- Define all the options in the problem
def option_a := 6 / x
def option_b := 6 / (x + 1)
def option_c := 6 / (x - 1)
def option_d := x / 6
def option_e := (x + 1) / 6
def option_f := (x - 2) / 6

-- The proof problem statement to show that option_b is the smallest
theorem smallest_option :
  option_b < option_a ∧ option_b < option_c ∧ option_b < option_d ∧ option_b < option_e ∧ option_b < option_f :=
by
  sorry

end smallest_option_l59_59205


namespace solve_for_x_l59_59673

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l59_59673


namespace number_of_factors_of_x_l59_59707

theorem number_of_factors_of_x (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (h4 : a < b) (h5 : b < c) (h6 : ¬ a = b) (h7 : ¬ b = c) (h8 : ¬ a = c) :
  let x := 2^2 * a^3 * b^2 * c^4
  let num_factors := (2 + 1) * (3 + 1) * (2 + 1) * (4 + 1)
  num_factors = 180 := by
sorry

end number_of_factors_of_x_l59_59707


namespace octagon_has_20_diagonals_l59_59655

-- Conditions
def is_octagon (n : ℕ) : Prop := n = 8

def diagonals_in_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Question to prove == Answer given conditions.
theorem octagon_has_20_diagonals : ∀ n, is_octagon n → diagonals_in_polygon n = 20 := by
  intros n hn
  rw [is_octagon, diagonals_in_polygon]
  rw hn
  norm_num

end octagon_has_20_diagonals_l59_59655


namespace squares_equal_l59_59406

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l59_59406


namespace smallest_number_ending_in_6_moved_front_gives_4_times_l59_59478

theorem smallest_number_ending_in_6_moved_front_gives_4_times (x m n : ℕ) 
  (h1 : n = 10 * x + 6)
  (h2 : 6 * 10^m + x = 4 * n) :
  n = 1538466 :=
by
  sorry

end smallest_number_ending_in_6_moved_front_gives_4_times_l59_59478


namespace surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l59_59198

-- First Problem:
theorem surface_area_cone_first_octant :
  ∃ (surface_area : ℝ), 
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ z^2 = 2*x*y) → surface_area = 16 :=
sorry

-- Second Problem:
theorem surface_area_sphere_inside_cylinder (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 ∧ x^2 + y^2 = R*x) → surface_area = 2 * R^2 * (π - 2) :=
sorry

-- Third Problem:
theorem surface_area_cylinder_inside_sphere (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 = R*x ∧ x^2 + y^2 + z^2 = R^2) → surface_area = 4 * R^2 :=
sorry

end surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l59_59198


namespace constant_k_value_l59_59361

theorem constant_k_value 
  (S : ℕ → ℕ)
  (h : ∀ n : ℕ, S n = 4 * 3^(n + 1) - k) :
  k = 12 :=
sorry

end constant_k_value_l59_59361


namespace max_value_of_n_l59_59459

-- Define the main variables and conditions
noncomputable def max_n : ℕ := 
  let n_pairs := [(1, 72), (2, 36), (3, 24), (4, 18), (6, 12), (8, 9)] 
  max (n_pairs.map (λ p, 6 * p.2 + p.1))

-- Lean theorem statement for the equivalence
theorem max_value_of_n :
  max_n = 433 := by
  sorry

end max_value_of_n_l59_59459


namespace transformed_parabola_l59_59695

theorem transformed_parabola (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 3) → (y = 2 * (x + 1)^2 + 2) :=
by
  sorry

end transformed_parabola_l59_59695


namespace bags_bought_l59_59926

theorem bags_bought (initial_bags : ℕ) (bags_given : ℕ) (final_bags : ℕ) (bags_bought : ℕ) :
  initial_bags = 20 → 
  bags_given = 4 → 
  final_bags = 22 → 
  bags_bought = final_bags - (initial_bags - bags_given) → 
  bags_bought = 6 := 
by
  intros h_initial h_given h_final h_buy
  rw [h_initial, h_given, h_final] at h_buy
  exact h_buy

#check bags_bought

end bags_bought_l59_59926


namespace cost_of_product_l59_59920

theorem cost_of_product (x : ℝ) (a : ℝ) (h : a > 0) :
  (1 + a / 100) * (x / (1 + a / 100)) = x :=
by
  field_simp [ne_of_gt h]
  sorry

end cost_of_product_l59_59920


namespace sum_of_extreme_numbers_is_846_l59_59600

theorem sum_of_extreme_numbers_is_846 :
  let digits := [0, 2, 4, 6]
  let is_valid_hundreds_digit (d : Nat) := d ≠ 0
  let create_three_digit_number (h t u : Nat) := h * 100 + t * 10 + u
  let max_num := create_three_digit_number 6 4 2
  let min_num := create_three_digit_number 2 0 4
  max_num + min_num = 846 := by
  sorry

end sum_of_extreme_numbers_is_846_l59_59600


namespace incorrect_statement_l59_59365

def geom_seq (a r : ℝ) : ℕ → ℝ
| 0       => a
| (n + 1) => r * geom_seq a r n

theorem incorrect_statement
  (a : ℝ) (r : ℝ) (S6 : ℝ)
  (h1 : r = 1 / 2)
  (h2 : S6 = a * (1 - (1 / 2) ^ 6) / (1 - 1 / 2))
  (h3 : S6 = 378) :
  geom_seq a r 2 / S6 ≠ 1 / 8 :=
by 
  have h4 : a = 192 := by sorry
  have h5 : geom_seq 192 (1 / 2) 2 = 192 * (1 / 2) ^ 2 := by sorry
  exact sorry

end incorrect_statement_l59_59365


namespace sum_of_n_values_l59_59021

theorem sum_of_n_values (sum_n : ℕ) : (∀ n : ℕ, 0 < n ∧ 24 % (2 * n - 1) = 0) → sum_n = 3 :=
by
  sorry

end sum_of_n_values_l59_59021


namespace remainder_of_power_modulo_l59_59424

theorem remainder_of_power_modulo : (3^2048) % 11 = 5 := by
  sorry

end remainder_of_power_modulo_l59_59424


namespace probability_multiple_of_3_l59_59326

theorem probability_multiple_of_3 : 
  let outcomes := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
                   (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
                   (3,1), (3,2), (3,3), (3,4), (3,5), (3,6),
                   (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
                   (5,1), (5,2), (5,3), (5,4), (5,5), (5,6),
                   (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)] in
  let multiples_of_3 := [(1,3), (3,1), (1,6), (6,1), (2,3), (3,2),
                         (2,6), (6,2), (3,3), (3,4), (4,3), (3,5),
                         (5,3), (3,6), (6,3), (4,6), (6,4), (5,6),
                         (6,5), (6,6)] in
  (multiples_of_3.length / outcomes.length : ℚ) = 5 / 9 := 
by
  sorry

end probability_multiple_of_3_l59_59326


namespace stream_current_l59_59444

noncomputable def solve_stream_current : Prop :=
  ∃ (r w : ℝ), (24 / (r + w) + 6 = 24 / (r - w)) ∧ (24 / (3 * r + w) + 2 = 24 / (3 * r - w)) ∧ (w = 2)

theorem stream_current : solve_stream_current :=
  sorry

end stream_current_l59_59444


namespace problem_l59_59253

open Real

theorem problem {a : ℝ} (ha : 0 < a) :
  e^a < (λ x, (∑ s in finset.range (nat.succ x), ((a + s) / x) ^ x)) at_top < e^(a + 1) :=
begin
  sorry
end

end problem_l59_59253


namespace parabola_find_a_l59_59578

theorem parabola_find_a (a b c : ℤ) :
  (∀ x y : ℤ, (x, y) ∈ [(1, 4), (-2, 3)] → y = a * x ^ 2 + b * x + c) →
  (∃ x y : ℤ, y = a * (x + 1) ^ 2 + 2 ∧ (x, y) = (-1, 2)) →
  a = 1 := 
by 
  sorry

end parabola_find_a_l59_59578


namespace possible_values_of_m_l59_59828

theorem possible_values_of_m (m : ℕ) (h1 : 3 * m + 15 > 3 * m + 8) 
  (h2 : 3 * m + 8 > 4 * m - 4) (h3 : m > 11) : m = 11 := 
by
  sorry

end possible_values_of_m_l59_59828


namespace number_of_ordered_quadruples_l59_59702

theorem number_of_ordered_quadruples (x1 x2 x3 x4 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h_sum : x1 + x2 + x3 + x4 = 100) : 
  ∃ n : ℕ, n = 156849 := 
by 
  sorry

end number_of_ordered_quadruples_l59_59702


namespace expand_expression_l59_59957

theorem expand_expression (x : ℝ) : 
  (11 * x^2 + 5 * x - 3) * (3 * x^3) = 33 * x^5 + 15 * x^4 - 9 * x^3 :=
by 
  sorry

end expand_expression_l59_59957


namespace expand_expression_l59_59781

variables {R : Type*} [CommRing R] (x : R)

theorem expand_expression : (15 * x^2 + 5) * 3 * x^3 = 45 * x^5 + 15 * x^3 :=
by sorry

end expand_expression_l59_59781


namespace sum_of_interior_edges_l59_59921

-- Define the problem parameters
def width_of_frame : ℝ := 2 -- width of the frame pieces in inches
def exposed_area : ℝ := 30 -- exposed area of the frame in square inches
def outer_edge_length : ℝ := 6 -- one of the outer edge length in inches

-- Define the statement to prove
theorem sum_of_interior_edges :
  ∃ (y : ℝ), (6 * y - 2 * (y - width_of_frame * 2) = exposed_area) ∧
  (2 * (6 - width_of_frame * 2) + 2 * (y - width_of_frame * 2) = 7) :=
sorry

end sum_of_interior_edges_l59_59921


namespace problem1_problem2_l59_59227

-- Define the function f(x)
def f (m x : ℝ) : ℝ := |m * x + 1| + |2 * x - 3|

-- Problem 1: Prove the range of x for f(x) = 4 when m = 2
theorem problem1 (x : ℝ) : f 2 x = 4 ↔ -1 / 2 ≤ x ∧ x ≤ 3 / 2 :=
by
  sorry

-- Problem 2: Prove the range of m given f(1) ≤ (2a^2 + 8) / a for any positive a
theorem problem2 (m : ℝ) (h : ∀ a : ℝ, a > 0 → f m 1 ≤ (2 * a^2 + 8) / a) : -8 ≤ m ∧ m ≤ 6 :=
by
  sorry

end problem1_problem2_l59_59227


namespace quadratic_solution_range_l59_59583

theorem quadratic_solution_range {x : ℝ} 
  (h : x^2 - 6 * x + 8 < 0) : 
  25 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 49 :=
sorry

end quadratic_solution_range_l59_59583


namespace three_digit_divisible_by_11_l59_59715

theorem three_digit_divisible_by_11
  (x y z : ℕ) (h1 : y = x + z) : (100 * x + 10 * y + z) % 11 = 0 :=
by
  sorry

end three_digit_divisible_by_11_l59_59715


namespace solve_for_x_l59_59685

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l59_59685


namespace evaluate_powers_of_i_l59_59189

-- Define complex number "i"
def i := Complex.I

-- Define the theorem to prove
theorem evaluate_powers_of_i : i^44 + i^444 + 3 = 5 := by
  -- use the cyclic property of i to simplify expressions
  sorry

end evaluate_powers_of_i_l59_59189


namespace find_r_over_s_at_2_l59_59721

noncomputable def r (x : ℝ) := 6 * x
noncomputable def s (x : ℝ) := (x + 4) * (x - 1)

theorem find_r_over_s_at_2 :
  r 2 / s 2 = 2 :=
by
  -- The corresponding steps to show this theorem.
  sorry

end find_r_over_s_at_2_l59_59721


namespace poly_eq_zero_or_one_l59_59116

noncomputable def k : ℝ := 2 -- You can replace 2 with any number greater than 1.

theorem poly_eq_zero_or_one (P : ℝ → ℝ) 
  (h1 : k > 1) 
  (h2 : ∀ x : ℝ, P (x ^ k) = (P x) ^ k) : 
  (∀ x, P x = 0) ∨ (∀ x, P x = 1) :=
sorry

end poly_eq_zero_or_one_l59_59116


namespace calc_value_of_fraction_l59_59055

theorem calc_value_of_fraction :
  (10^9 / (2 * 5^2 * 10^3)) = 20000 := by
  sorry

end calc_value_of_fraction_l59_59055


namespace find_red_chairs_l59_59996

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

end find_red_chairs_l59_59996


namespace perimeter_large_star_l59_59723

theorem perimeter_large_star (n m : ℕ) (P : ℕ)
  (triangle_perimeter : ℕ) (quad_perimeter : ℕ) (small_star_perimeter : ℕ)
  (hn : n = 5) (hm : m = 5)
  (h_triangle_perimeter : triangle_perimeter = 7)
  (h_quad_perimeter : quad_perimeter = 18)
  (h_small_star_perimeter : small_star_perimeter = 3) :
  m * quad_perimeter + small_star_perimeter = n * triangle_perimeter + P → P = 58 :=
by 
  -- Placeholder proof
  sorry

end perimeter_large_star_l59_59723


namespace range_of_x_l59_59949

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 * x else 2 * -x

theorem range_of_x {x : ℝ} :
  f (1 - 2 * x) < f 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end range_of_x_l59_59949


namespace sin_cos_product_l59_59353

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin x * Real.cos x = 2 / 5 := by
  sorry

end sin_cos_product_l59_59353


namespace population_density_reduction_l59_59144

theorem population_density_reduction (scale : ℕ) (real_world_population : ℕ) : 
  scale = 1000000 → real_world_population = 1000000000 → 
  real_world_population / (scale ^ 2) < 1 := 
by 
  intros scale_value rw_population_value
  have h1 : scale ^ 2 = 1000000000000 := by sorry
  have h2 : real_world_population / 1000000000000 = 1 / 1000 := by sorry
  sorry

end population_density_reduction_l59_59144


namespace monthly_earnings_l59_59917

theorem monthly_earnings (savings_per_month : ℤ) (total_needed : ℤ) (total_earned : ℤ)
  (H1 : savings_per_month = 500)
  (H2 : total_needed = 45000)
  (H3 : total_earned = 360000) :
  total_earned / (total_needed / savings_per_month) = 4000 := by
  sorry

end monthly_earnings_l59_59917


namespace pqrs_sum_l59_59533

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l59_59533


namespace fresh_fruit_sold_l59_59275

variable (total_fruit frozen_fruit : ℕ)

theorem fresh_fruit_sold (h1 : total_fruit = 9792) (h2 : frozen_fruit = 3513) : 
  total_fruit - frozen_fruit = 6279 :=
by sorry

end fresh_fruit_sold_l59_59275


namespace initial_quantity_of_gummy_worms_l59_59259

theorem initial_quantity_of_gummy_worms (x : ℕ) (h : x / 2^4 = 4) : x = 64 :=
sorry

end initial_quantity_of_gummy_worms_l59_59259


namespace min_value_of_expression_l59_59780

theorem min_value_of_expression : ∃ x : ℝ, (8 - x) * (6 - x) * (8 + x) * (6 + x) ≥ -196 :=
by
  sorry

end min_value_of_expression_l59_59780


namespace ball_radius_and_surface_area_l59_59038

theorem ball_radius_and_surface_area (d h r : ℝ) (radius_eq : d / 2 = 6) (depth_eq : h = 2) 
  (pythagorean : (r - h)^2 + (d / 2)^2 = r^2) :
  r = 10 ∧ (4 * Real.pi * r^2 = 400 * Real.pi) :=
by
  sorry

end ball_radius_and_surface_area_l59_59038


namespace arcsin_one_eq_pi_div_two_l59_59941

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := by
  sorry

end arcsin_one_eq_pi_div_two_l59_59941


namespace power_function_k_values_l59_59359

theorem power_function_k_values (k : ℝ) : (∃ (a : ℝ), (k^2 - k - 5) = a) → (k = 3 ∨ k = -2) :=
by
  intro h
  have h1 : k^2 - k - 5 = 1 := sorry -- Using the condition that it is a power function
  have h2 : k^2 - k - 6 = 0 := by linarith -- Simplify the equation
  exact sorry -- Solve the quadratic equation

end power_function_k_values_l59_59359


namespace minimum_value_ineq_l59_59024

theorem minimum_value_ineq (x : ℝ) (hx : x >= 4) : x + 4 / (x - 1) >= 5 := by
  sorry

end minimum_value_ineq_l59_59024


namespace defective_probability_bayesian_probabilities_l59_59607

noncomputable def output_proportion_A : ℝ := 0.25
noncomputable def output_proportion_B : ℝ := 0.35
noncomputable def output_proportion_C : ℝ := 0.40

noncomputable def defect_rate_A : ℝ := 0.05
noncomputable def defect_rate_B : ℝ := 0.04
noncomputable def defect_rate_C : ℝ := 0.02

noncomputable def probability_defective : ℝ :=
  output_proportion_A * defect_rate_A +
  output_proportion_B * defect_rate_B +
  output_proportion_C * defect_rate_C 

theorem defective_probability :
  probability_defective = 0.0345 := 
  by sorry

noncomputable def P_A_given_defective : ℝ :=
  (output_proportion_A * defect_rate_A) / probability_defective

noncomputable def P_B_given_defective : ℝ :=
  (output_proportion_B * defect_rate_B) / probability_defective

noncomputable def P_C_given_defective : ℝ :=
  (output_proportion_C * defect_rate_C) / probability_defective

theorem bayesian_probabilities :
  P_A_given_defective = 25 / 69 ∧
  P_B_given_defective = 28 / 69 ∧
  P_C_given_defective = 16 / 69 :=
  by sorry

end defective_probability_bayesian_probabilities_l59_59607


namespace select_medical_team_l59_59789

open Nat

theorem select_medical_team : 
  let male_doctors := 5
  let female_doctors := 4
  let selected_doctors := 3
  (male_doctors.choose 1 * female_doctors.choose 2 + male_doctors.choose 2 * female_doctors.choose 1) = 70 :=
by
  sorry

end select_medical_team_l59_59789


namespace squares_equal_l59_59408

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l59_59408


namespace max_value_of_f_l59_59861

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^3 + Real.cos (2 * x) - (Real.cos x)^2 - Real.sin x

theorem max_value_of_f :
  ∃ x : ℝ, f x = 5 / 27 ∧ ∀ y : ℝ, f y ≤ 5 / 27 :=
sorry

end max_value_of_f_l59_59861


namespace odd_numbers_le_twice_switch_pairs_l59_59381

-- Number of odd elements in row n is denoted as numOdd n
def numOdd (n : ℕ) : ℕ := -- Definition of numOdd function
sorry

-- Number of switch pairs in row n is denoted as numSwitchPairs n
def numSwitchPairs (n : ℕ) : ℕ := -- Definition of numSwitchPairs function
sorry

-- Definition of Pascal's Triangle and conditions
def binom (n k : ℕ) : ℕ := if k > n then 0 else if k = 0 ∨ k = n then 1 else binom (n-1) (k-1) + binom (n-1) k

-- Check even or odd
def isOdd (n : ℕ) : Bool := n % 2 = 1

-- Definition of switch pair check
def isSwitchPair (a b : ℕ) : Prop := (isOdd a ∧ ¬isOdd b) ∨ (¬isOdd a ∧ isOdd b)

theorem odd_numbers_le_twice_switch_pairs (n : ℕ) :
  numOdd n ≤ 2 * numSwitchPairs (n-1) :=
sorry

end odd_numbers_le_twice_switch_pairs_l59_59381


namespace max_value_a_l59_59430

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a = 3 :=
sorry

end max_value_a_l59_59430


namespace probability_fewer_heads_than_tails_l59_59889

theorem probability_fewer_heads_than_tails (n : ℕ) (hn : n = 12) (p_heads_tails : ∀ k, k ≤ n → k / 2 = n / 2) :
  (num_fewer : ℚ) × (den_fewer : ℕ) := (793, 2048) := 
by 
  have h_total_outcomes : ℕ := 2^12
  have h_equal_heads_tails : ℕ := nat.choose 12 6
  have p_y : ℚ := (nat.choose 12 6) / h_total_outcomes
  have p_x : ℚ := (1 - p_y) / 2
  exacts [(793, 2048)]

end probability_fewer_heads_than_tails_l59_59889


namespace ratio_of_b_to_a_l59_59759

variable (V A B : ℝ)

def ten_pours_of_a_cup : Prop := 10 * A = V
def five_pours_of_b_cup : Prop := 5 * B = V

theorem ratio_of_b_to_a (h1 : ten_pours_of_a_cup V A) (h2 : five_pours_of_b_cup V B) : B / A = 2 :=
sorry

end ratio_of_b_to_a_l59_59759


namespace move_line_upwards_l59_59281

theorem move_line_upwards (x y : ℝ) :
  (y = -x + 1) → (y + 5 = -x + 6) :=
by
  intro h
  sorry

end move_line_upwards_l59_59281


namespace problem_distribution_l59_59175

theorem problem_distribution:
  let num_problems := 6
  let num_friends := 15
  (num_friends ^ num_problems) = 11390625 :=
by sorry

end problem_distribution_l59_59175


namespace find_y_l59_59354

theorem find_y (x y : ℚ) (h1 : x = 151) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 342200) : 
  y = 342200 / 3354151 :=
by
  sorry

end find_y_l59_59354


namespace minimize_square_sum_l59_59549

theorem minimize_square_sum (x1 x2 x3 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) 
  (h4 : x1 + 3 * x2 + 5 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 ≥ 2000 / 7 :=
sorry

end minimize_square_sum_l59_59549


namespace MarcoScoresAreCorrect_l59_59867

noncomputable def MarcoTestScores : List ℕ := [94, 82, 76, 75, 64]

theorem MarcoScoresAreCorrect : 
  ∀ (scores : List ℕ),
    scores = [82, 76, 75] ∧ 
    (∃ t4 t5, t4 < 95 ∧ t5 < 95 ∧ 82 ≠ t4 ∧ 82 ≠ t5 ∧ 76 ≠ t4 ∧ 76 ≠ t5 ∧ 75 ≠ t4 ∧ 75 ≠ t5 ∧ 
       t4 ≠ t5 ∧
       (82 + 76 + 75 + t4 + t5 = 5 * 85) ∧ 
       (82 + 76 = t4 + t5)) → 
    (scores = [94, 82, 76, 75, 64]) := 
by 
  sorry

end MarcoScoresAreCorrect_l59_59867


namespace standard_robot_weight_l59_59845

variable (S : ℕ) -- Define the variable for the standard robot's weight
variable (MaxWeight : ℕ := 210) -- Define the variable for the maximum weight of a robot, which is 210 pounds
variable (MinWeight : ℕ) -- Define the variable for the minimum weight of the robot

theorem standard_robot_weight (h1 : 2 * MinWeight ≥ MaxWeight) 
                             (h2 : MinWeight = S + 5) 
                             (h3 : MaxWeight = 210) :
  100 ≤ S ∧ S ≤ 105 := 
by
  sorry

end standard_robot_weight_l59_59845


namespace point_equal_distances_l59_59766

theorem point_equal_distances (x y : ℝ) (hx : y = x) (hxy : y - 4 = -x) (hline : x + y = 4) : x = 2 :=
by sorry

end point_equal_distances_l59_59766


namespace minimize_quadratic_l59_59153

theorem minimize_quadratic : ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12*x + 28 ≤ y^2 - 12*y + 28) :=
by
  use 6
  sorry

end minimize_quadratic_l59_59153


namespace number_of_lockers_l59_59870

-- Problem Conditions
def locker_numbers_consecutive_from_one := ∀ (n : ℕ), n ≥ 1
def cost_per_digit := 0.02
def total_cost := 137.94

-- Theorem Statement
theorem number_of_lockers (h1 : locker_numbers_consecutive_from_one) (h2 : cost_per_digit = 0.02) (h3 : total_cost = 137.94) : ∃ n : ℕ, n = 2001 :=
sorry

end number_of_lockers_l59_59870


namespace A_inter_B_domain_l59_59625

def A_domain : Set ℝ := {x : ℝ | x^2 + x - 2 >= 0}
def B_domain : Set ℝ := {x : ℝ | (2*x + 6)/(3 - x) >= 0 ∧ x ≠ -2}

theorem A_inter_B_domain :
  (A_domain ∩ B_domain) = {x : ℝ | (1 <= x ∧ x < 3) ∨ (-3 <= x ∧ x < -2)} :=
by
  sorry

end A_inter_B_domain_l59_59625


namespace dragons_total_games_l59_59933

theorem dragons_total_games (y x : ℕ) (h1 : x = 60 * y / 100) (h2 : (x + 8) = 55 * (y + 11) / 100) : y + 11 = 50 :=
by
  sorry

end dragons_total_games_l59_59933


namespace problem1_problem2_l59_59803

-- Problem 1
theorem problem1 (a b c d : ℝ) (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : (c / a) - (d / b) > 0 := sorry

-- Problem 2
theorem problem2 (a b c d : ℝ) (ha_gt_b : a > b) (hc_gt_d : c > d) : a - d > b - c := sorry

end problem1_problem2_l59_59803


namespace gcd_9011_4379_l59_59740

def a : ℕ := 9011
def b : ℕ := 4379

theorem gcd_9011_4379 : Nat.gcd a b = 1 := by
  sorry

end gcd_9011_4379_l59_59740


namespace rectangle_area_l59_59152

theorem rectangle_area (x : ℝ) (h : (2*x - 3) * (3*x + 4) = 20 * x - 12) : x = 7 / 2 :=
sorry

end rectangle_area_l59_59152


namespace simplify_expression_l59_59566

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l59_59566


namespace solve_quadratic_for_q_l59_59194

-- Define the quadratic equation and the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main theorem statement
theorem solve_quadratic_for_q : ∃ q : ℝ, q ≠ 0 ∧ (discriminant q (-8) 2 = 0) → q = 8 :=
by
  -- Insert the assumptions and proof context here.
  -- However, since we were instructed not to consider the solution steps
  -- the proof is skipped with a "sorry".
  sorry

end solve_quadratic_for_q_l59_59194


namespace determine_base_l59_59779

theorem determine_base (x : ℕ) (h : 2 * x^3 + x + 6 = x^3 + 2 * x + 342) : x = 7 := 
sorry

end determine_base_l59_59779


namespace calculate_entire_surface_area_l59_59768

-- Define the problem parameters
def cube_edge_length : ℝ := 4
def hole_side_length : ℝ := 2

-- Define the function to compute the total surface area
noncomputable def entire_surface_area : ℝ :=
  let original_surface_area := 6 * (cube_edge_length ^ 2)
  let hole_area := 6 * (hole_side_length ^ 2)
  let exposed_internal_area := 6 * 4 * (hole_side_length ^ 2)
  original_surface_area - hole_area + exposed_internal_area

-- Statement of the problem to prove the given conditions
theorem calculate_entire_surface_area : entire_surface_area = 168 := by
  sorry

end calculate_entire_surface_area_l59_59768


namespace binomial_inequality_l59_59714

theorem binomial_inequality (n : ℤ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end binomial_inequality_l59_59714


namespace count_pairs_a_b_l59_59754

-- Define sequence length and target count
def sequenceLength : ℕ := 2016
def targetCount : ℕ := 508536

-- Define the proposition
theorem count_pairs_a_b :
  ∃ (count : ℕ), count = (sequenceLength - 1) * sequenceLength / 2 ∧ count = targetCount := by 
begin
  have h: (sequenceLength - 1) * sequenceLength / 2 = targetCount, 
  {
    calc (2016 - 1) * 2016 / 2 
      = 2015 * 2016 / 2 : by sorry
      = targetCount : by sorry
  },
  use (sequenceLength - 1) * sequenceLength / 2,
  split,
  { exact rfl, },
  { exact h, },
end

end count_pairs_a_b_l59_59754


namespace diagonals_of_octagon_l59_59658

theorem diagonals_of_octagon : 
  ∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20 :=
by 
  intros n h_n
  rw [h_n]
  norm_num
  sorry

end diagonals_of_octagon_l59_59658


namespace minimum_value_l59_59621

open Real

def f (x : ℝ) : ℝ := x / exp(x)

theorem minimum_value : ∃ x ∈ Icc (2:ℝ) 4, ∀ y ∈ Icc (2:ℝ) 4, f(x) ≤ f(y) ∧ f(x) = 2 / exp(2) :=
by
  sorry

end minimum_value_l59_59621


namespace value_b15_l59_59224

def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d
def geometric_sequence (b : ℕ → ℤ) := ∃ q : ℤ, ∀ n : ℕ, b (n+1) = q * b n

theorem value_b15 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n : ℕ, S n = (n * (a 0 + a (n-1)) / 2))
  (h3 : S 9 = -18)
  (h4 : S 13 = -52)
  (h5 : geometric_sequence b)
  (h6 : b 5 = a 5)
  (h7 : b 7 = a 7) : 
  b 15 = -64 :=
sorry

end value_b15_l59_59224


namespace christine_stickers_l59_59451

theorem christine_stickers (stickers_has stickers_needs : ℕ) (h_has : stickers_has = 11) (h_needs : stickers_needs = 19) : 
  stickers_has + stickers_needs = 30 :=
by 
  sorry

end christine_stickers_l59_59451


namespace problem_statement_l59_59488

theorem problem_statement (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + 1 / x^2 = 7 :=
sorry

end problem_statement_l59_59488


namespace xiao_ming_should_choose_store_A_l59_59301

def storeB_cost (x : ℕ) : ℝ := 0.85 * x

def storeA_cost (x : ℕ) : ℝ :=
  if x ≤ 10 then x
  else 0.7 * x + 3

theorem xiao_ming_should_choose_store_A (x : ℕ) (h : x = 22) :
  storeA_cost x < storeB_cost x := by
  sorry

end xiao_ming_should_choose_store_A_l59_59301


namespace set_union_covers_real_line_l59_59230

open Set

def M := {x : ℝ | x < 0 ∨ 2 < x}
def N := {x : ℝ | -Real.sqrt 5 < x ∧ x < Real.sqrt 5}

theorem set_union_covers_real_line : M ∪ N = univ := sorry

end set_union_covers_real_line_l59_59230


namespace axis_of_symmetry_smallest_positive_period_range_of_h_l59_59347

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x + π / 12) ^ 2
noncomputable def g (x : ℝ) : ℝ := 3 + 2 * Real.sin x * Real.cos x
noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f (x + π) = f (x - π) ↔ x = (k * π) / 2 - π / 12 :=
sorry

theorem smallest_positive_period :
  ∃ T : ℝ, (T > 0) ∧ ∀ x : ℝ, h (x + T) = h x ∧ T = π :=
sorry

theorem range_of_h :
  ∀ x : ℝ, 3 ≤ h x ∧ h x ≤ 5 :=
sorry

end axis_of_symmetry_smallest_positive_period_range_of_h_l59_59347


namespace bottom_right_corner_value_l59_59710

variable (a b c x : ℕ)

/--
Conditions:
- The sums of the numbers in each of the four 2x2 grids forming part of the 3x3 grid are equal.
- Known values for corners: a, b, and c.
Conclusion:
- The bottom right corner value x must be 0.
-/

theorem bottom_right_corner_value (S: ℕ) (A B C D E: ℕ) :
  S = a + A + B + C →
  S = A + b + C + D →
  S = B + C + c + E →
  S = C + D + E + x →
  x = 0 :=
by
  sorry

end bottom_right_corner_value_l59_59710


namespace credit_extended_l59_59051

noncomputable def automobile_installment_credit (total_consumer_credit : ℝ) : ℝ :=
  0.43 * total_consumer_credit

noncomputable def extended_by_finance_companies (auto_credit : ℝ) : ℝ :=
  0.25 * auto_credit

theorem credit_extended (total_consumer_credit : ℝ) (h : total_consumer_credit = 465.1162790697675) :
  extended_by_finance_companies (automobile_installment_credit total_consumer_credit) = 50.00 :=
by
  rw [h]
  sorry

end credit_extended_l59_59051


namespace smallest_possible_n_l59_59598

theorem smallest_possible_n (n : ℕ) (h1 : 0 < n) (h2 : 0 < 60) 
  (h3 : (Nat.lcm 60 n) / (Nat.gcd 60 n) = 24) : n = 20 :=
by sorry

end smallest_possible_n_l59_59598


namespace certain_number_divided_by_two_l59_59687

theorem certain_number_divided_by_two (x : ℝ) (h : x / 2 + x + 2 = 62) : x = 40 :=
sorry

end certain_number_divided_by_two_l59_59687


namespace find_sum_of_coefficients_l59_59006

theorem find_sum_of_coefficients (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -(1/2) ∨ x > 1/3)) :
  a + b = -14 := 
sorry

end find_sum_of_coefficients_l59_59006


namespace find_k_value_l59_59965

theorem find_k_value (k : ℝ) (h : 64 / k = 4) : k = 16 :=
by
  sorry

end find_k_value_l59_59965


namespace number_of_bass_caught_l59_59127

/-
Statement:
Given:
1. An eight-pound trout.
2. Two twelve-pound salmon.
3. They need to feed 22 campers with two pounds of fish each.
Prove that the number of two-pound bass caught is 6.
-/

theorem number_of_bass_caught
  (weight_trout : ℕ := 8)
  (weight_salmon : ℕ := 12)
  (num_salmon : ℕ := 2)
  (num_campers : ℕ := 22)
  (required_per_camper : ℕ := 2)
  (weight_bass : ℕ := 2) :
  (num_campers * required_per_camper - (weight_trout + num_salmon * weight_salmon)) / weight_bass = 6 :=
by
  sorry  -- Proof to be completed

end number_of_bass_caught_l59_59127


namespace moving_circle_passes_through_fixed_point_l59_59916
-- We will start by importing the necessary libraries and setting up the problem conditions.

-- Define the parabola y^2 = 8x.
def parabola (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = 8 * p.1

-- Define the line x + 2 = 0.
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.1 = -2

-- Define the fixed point.
def fixed_point : ℝ × ℝ :=
  (2, 0)

-- Define the moving circle passing through the fixed point.
def moving_circle (p : ℝ × ℝ) (c : ℝ × ℝ) :=
  p = fixed_point

-- Bring it all together in the theorem.
theorem moving_circle_passes_through_fixed_point (c : ℝ × ℝ) (p : ℝ × ℝ)
  (h_parabola : parabola c)
  (h_tangent : tangent_line p) :
  moving_circle p c :=
sorry

end moving_circle_passes_through_fixed_point_l59_59916


namespace gummies_remain_l59_59831

theorem gummies_remain
  (initial_candies : ℕ)
  (sibling_candies_per : ℕ)
  (num_siblings : ℕ)
  (best_friend_fraction : ℝ)
  (cousin_fraction : ℝ)
  (kept_candies : ℕ)
  (result : ℕ)
  (h_initial : initial_candies = 500)
  (h_sibling_candies_per : sibling_candies_per = 35)
  (h_num_siblings : num_siblings = 3)
  (h_best_friend_fraction : best_friend_fraction = 0.5)
  (h_cousin_fraction : cousin_fraction = 0.25)
  (h_kept_candies : kept_candies = 50)
  (h_result : result = 99) : 
  (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋ - 
  ⌊cousin_fraction * (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋)⌋ 
  - kept_candies) = result := 
by {
  sorry
}

end gummies_remain_l59_59831


namespace parallel_transitivity_l59_59973

variable (Line Plane : Type)
variable (m n : Line)
variable (α : Plane)

-- Definitions for parallelism
variable (parallel : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Conditions
variable (m_n_parallel : parallel m n)
variable (m_alpha_parallel : parallelLinePlane m α)
variable (n_outside_alpha : ¬ parallelLinePlane n α)

-- Proposition to be proved
theorem parallel_transitivity (m n : Line) (α : Plane) 
  (h1 : parallel m n) 
  (h2 : parallelLinePlane m α) 
  : parallelLinePlane n α :=
sorry 

end parallel_transitivity_l59_59973


namespace octagon_diagonals_l59_59647

theorem octagon_diagonals : 
  let n := 8 in 
  let total_pairs := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 20 :=
by
  sorry

end octagon_diagonals_l59_59647


namespace bricks_of_other_types_l59_59264

theorem bricks_of_other_types (A B total other: ℕ) (hA: A = 40) (hB: B = A / 2) (hTotal: total = 150) (hSum: total = A + B + other): 
  other = 90 :=
by sorry

end bricks_of_other_types_l59_59264


namespace soldiers_arrival_time_l59_59736

open Function

theorem soldiers_arrival_time
    (num_soldiers : ℕ) (distance : ℝ) (car_speed : ℝ) (car_capacity : ℕ) (walk_speed : ℝ) (start_time : ℝ) :
    num_soldiers = 12 →
    distance = 20 →
    car_speed = 20 →
    car_capacity = 4 →
    walk_speed = 4 →
    start_time = 0 →
    ∃ arrival_time, arrival_time = 2 + 36/60 :=
by
  intros
  sorry

end soldiers_arrival_time_l59_59736


namespace number_of_triangles_l59_59662

/-- 
  This statement defines and verifies the number of triangles 
  in the given geometric figure.
-/
theorem number_of_triangles (rectangle : Set ℝ) : 
  (exists lines : Set (List (ℝ × ℝ)), -- assuming a set of lines dividing the rectangle
    let small_right_triangles := 40
    let intermediate_isosceles_triangles := 8
    let intermediate_triangles := 10
    let larger_right_triangles := 20
    let largest_isosceles_triangles := 5
    small_right_triangles + intermediate_isosceles_triangles + intermediate_triangles + larger_right_triangles + largest_isosceles_triangles = 83) :=
sorry

end number_of_triangles_l59_59662


namespace simplify_expression_l59_59563

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l59_59563


namespace smallest_divisor_l59_59724

theorem smallest_divisor (n : ℕ) (h1 : n = 999) :
  ∃ d : ℕ, 2.45 ≤ (999 : ℝ) / d ∧ (999 : ℝ) / d < 2.55 ∧ d = 392 :=
by
  sorry

end smallest_divisor_l59_59724


namespace fraction_solution_l59_59576

theorem fraction_solution (a : ℤ) (h : 0 < a ∧ (a : ℚ) / (a + 36) = 775 / 1000) : a = 124 := 
by
  sorry

end fraction_solution_l59_59576


namespace real_solutions_l59_59784

theorem real_solutions:
  ∀ x: ℝ, 
    (x ≠ 2) ∧ (x ≠ 4) ∧ 
    ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1)) / 
    ((x - 2) * (x - 4) * (x - 2)) = 1 
    → (x = 2 + Real.sqrt 2) ∨ (x = 2 - Real.sqrt 2) :=
by
  sorry

end real_solutions_l59_59784


namespace max_perimeter_of_triangle_l59_59045

theorem max_perimeter_of_triangle (x : ℕ) 
  (h1 : 3 < x) 
  (h2 : x < 15) 
  (h3 : 7 + 8 > x) 
  (h4 : 7 + x > 8) 
  (h5 : 8 + x > 7) :
  x = 14 ∧ 7 + 8 + x = 29 := 
by {
  sorry
}

end max_perimeter_of_triangle_l59_59045


namespace third_term_of_geometric_sequence_l59_59163

theorem third_term_of_geometric_sequence
  (a₁ : ℕ) (a₄ : ℕ)
  (h1 : a₁ = 5)
  (h4 : a₄ = 320) :
  ∃ a₃ : ℕ, a₃ = 80 :=
by
  sorry

end third_term_of_geometric_sequence_l59_59163


namespace sum_distinct_vars_eq_1716_l59_59543

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l59_59543


namespace cyclic_sum_inequality_l59_59689

theorem cyclic_sum_inequality (x y z : ℝ) (hp : x > 0 ∧ y > 0 ∧ z > 0) (h : x + y + z = 3) : 
  (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) < (3 + x * y + y * z + z * x) := by
  sorry

end cyclic_sum_inequality_l59_59689


namespace cards_traded_between_Padma_and_Robert_l59_59560

def total_cards_traded (padma_first_trade padma_second_trade robert_first_trade robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + padma_second_trade + robert_first_trade + robert_second_trade

theorem cards_traded_between_Padma_and_Robert (h1 : padma_first_trade = 2) 
                                            (h2 : robert_first_trade = 10)
                                            (h3 : padma_second_trade = 15)
                                            (h4 : robert_second_trade = 8) :
                                            total_cards_traded 2 15 10 8 = 35 := 
by 
  sorry

end cards_traded_between_Padma_and_Robert_l59_59560


namespace find_extra_factor_l59_59040

theorem find_extra_factor (w : ℕ) (h1 : w > 0) (h2 : w = 156) (h3 : ∃ (k : ℕ), (2^5 * 13^2) ∣ (936 * w))
  : 3 ∣ w := sorry

end find_extra_factor_l59_59040


namespace number_of_girls_l59_59356

def total_students (T : ℕ) :=
  0.40 * T = 300

def girls_at_school (T : ℕ) :=
  0.60 * T = 450

theorem number_of_girls (T : ℕ) (h : total_students T) : girls_at_school T :=
  sorry

end number_of_girls_l59_59356


namespace sum_of_roots_l59_59670

variable {p m n : ℝ}

axiom roots_condition (h : m * n = 4) : m + n = -4

theorem sum_of_roots (h : m * n = 4) : m + n = -4 := 
  roots_condition h

end sum_of_roots_l59_59670


namespace carl_typing_speed_l59_59936

theorem carl_typing_speed (words_per_day: ℕ) (minutes_per_day: ℕ) (total_words: ℕ) (days: ℕ) : 
  words_per_day = total_words / days ∧ 
  minutes_per_day = 4 * 60 ∧ 
  (words_per_day / minutes_per_day) = 50 :=
by 
  sorry

end carl_typing_speed_l59_59936


namespace age_of_17th_student_is_75_l59_59855

variables (T A : ℕ)

def avg_17_students := 17
def avg_5_students := 14
def avg_9_students := 16
def total_17_students := 17 * avg_17_students
def total_5_students := 5 * avg_5_students
def total_9_students := 9 * avg_9_students
def age_17th_student : ℕ := total_17_students - (total_5_students + total_9_students)

theorem age_of_17th_student_is_75 :
  age_17th_student = 75 := by sorry

end age_of_17th_student_is_75_l59_59855


namespace solve_fraction_eq_l59_59386

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x^2 - x - 2) / (x + 2) = x + 3 ↔ x = -4 / 3 :=
by 
  sorry

end solve_fraction_eq_l59_59386


namespace max_handshakes_25_people_l59_59437

-- Define the number of people attending the conference.
def num_people : ℕ := 25

-- Define the combinatorial formula to calculate the maximum number of handshakes.
def max_handshakes (n : ℕ) : ℕ := n.choose 2

-- State the theorem that we need to prove.
theorem max_handshakes_25_people : max_handshakes num_people = 300 :=
by
  -- Proof will be filled in later
  sorry

end max_handshakes_25_people_l59_59437


namespace import_tax_excess_amount_l59_59610

theorem import_tax_excess_amount 
    (tax_rate : ℝ) 
    (tax_paid : ℝ) 
    (total_value : ℝ)
    (X : ℝ) 
    (h1 : tax_rate = 0.07)
    (h2 : tax_paid = 109.2)
    (h3 : total_value = 2560) 
    (eq1 : tax_rate * (total_value - X) = tax_paid) :
    X = 1000 := sorry

end import_tax_excess_amount_l59_59610


namespace g_49_l59_59397

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eqn (x y : ℝ) : g (x^2 * y) = x * g y
axiom g_one_val : g 1 = 6

theorem g_49 : g 49 = 42 := by
  sorry

end g_49_l59_59397


namespace stewart_farm_sheep_count_l59_59285

theorem stewart_farm_sheep_count
  (ratio : ℕ → ℕ → Prop)
  (S H : ℕ)
  (ratio_S_H : ratio S H)
  (one_sheep_seven_horses : ratio 1 7)
  (food_per_horse : ℕ)
  (total_food : ℕ)
  (food_per_horse_val : food_per_horse = 230)
  (total_food_val : total_food = 12880)
  (calc_horses : H = total_food / food_per_horse)
  (calc_sheep : S = H / 7) :
  S = 8 :=
by {
  /- Given the conditions, we need to show that S = 8 -/
  sorry
}

end stewart_farm_sheep_count_l59_59285


namespace rational_coefficients_count_l59_59888

theorem rational_coefficients_count : 
  ∃ n, n = 84 ∧ ∀ k, (0 ≤ k ∧ k ≤ 500) → 
            (k % 3 = 0 ∧ (500 - k) % 2 = 0) → 
            n = 84 :=
by
  sorry

end rational_coefficients_count_l59_59888


namespace correct_calculation_l59_59026

theorem correct_calculation (a b : ℝ) : 
  (a + 2 * a = 3 * a) := by
  sorry

end correct_calculation_l59_59026


namespace age_of_b_l59_59278

variable (A B C : ℕ)

-- Conditions
def avg_abc : Prop := A + B + C = 78
def avg_ac : Prop := A + C = 58

-- Question: Prove that B = 20
theorem age_of_b (h1 : avg_abc A B C) (h2 : avg_ac A C) : B = 20 := 
by sorry

end age_of_b_l59_59278


namespace john_guests_count_l59_59112

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def additional_fractional_guests : ℝ := 0.60
def total_cost_when_wife_gets_her_way : ℕ := 50000

theorem john_guests_count (G : ℕ) :
  venue_cost + cost_per_guest * (1 + additional_fractional_guests) * G = 
  total_cost_when_wife_gets_her_way →
  G = 50 :=
by
  sorry

end john_guests_count_l59_59112


namespace find_x_when_y_equals_two_l59_59678

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l59_59678


namespace alyona_final_balances_l59_59770

noncomputable def finalBalances (initialEuros initialDollars initialRubles : ℕ)
                                (interestRateEuroDollar interestRateRuble : ℚ)
                                (conversionRateEuroToRubles1 conversionRateRublesToDollars1 : ℚ)
                                (conversionRateDollarsToRubles2 conversionRateRublesToEuros2 : ℚ) :
                                ℕ × ℕ × ℕ :=
  sorry

theorem alyona_final_balances :
  finalBalances 3000 4000 240000
                (2.1 / 100) (7.9 / 100)
                60.10 58.90
                58.50 63.20 = (4040, 3286, 301504) :=
  sorry

end alyona_final_balances_l59_59770


namespace proof_problem_l59_59074

theorem proof_problem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 + 2 * |y| = 2 * x * y) :
  (x > 0 → x + y > 3) ∧ (x < 0 → x + y < -3) :=
by
  sorry

end proof_problem_l59_59074


namespace first_discount_percentage_l59_59286

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (additional_discount : ℝ) (first_discount : ℝ) : 
  original_price = 600 → final_price = 513 → additional_discount = 0.05 →
  600 * (1 - first_discount / 100) * (1 - 0.05) = 513 →
  first_discount = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end first_discount_percentage_l59_59286


namespace numbers_not_all_less_than_six_l59_59213

theorem numbers_not_all_less_than_six (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬ (a + 4 / b < 6 ∧ b + 9 / c < 6 ∧ c + 16 / a < 6) :=
sorry

end numbers_not_all_less_than_six_l59_59213


namespace isosceles_trapezoid_ratio_l59_59513

theorem isosceles_trapezoid_ratio (a b h : ℝ) 
  (h1: h = b / 2)
  (h2: a = 1 - ((1 - b) / 2))
  (h3 : 1 = ((a + 1) / 2)^2 + (b / 2)^2) :
  b / a = (-1 + Real.sqrt 7) / 2 := 
sorry

end isosceles_trapezoid_ratio_l59_59513


namespace weight_of_B_l59_59898

theorem weight_of_B (A B C : ℝ) (h1 : (A + B + C) / 3 = 45) (h2 : (A + B) / 2 = 40) (h3 : (B + C) / 2 = 46) : B = 37 :=
by
  sorry

end weight_of_B_l59_59898


namespace rectangle_area_l59_59859

-- Definitions of conditions
def width : ℝ := 5
def length : ℝ := 2 * width

-- The goal is to prove the area is 50 square inches given the length and width
theorem rectangle_area : length * width = 50 := by
  have h_length : length = 2 * width := by rfl
  have h_width : width = 5 := by rfl
  sorry

end rectangle_area_l59_59859


namespace find_k_l59_59228

-- Define the vectors
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 1)
def c : ℝ × ℝ := (-5, 1)

-- Define the condition for parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define the statement to prove
theorem find_k : parallel (a.1 + k * b.1, a.2 + k * b.2) c → k = 1/2 :=
by
  sorry

end find_k_l59_59228


namespace exponential_fixed_point_l59_59858

theorem exponential_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (a^(4-4) + 5 = 6) :=
sorry

end exponential_fixed_point_l59_59858


namespace solve_for_x_l59_59671

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l59_59671


namespace computation_result_l59_59452

theorem computation_result :
  (3 + 6 - 12 + 24 + 48 - 96 + 192 - 384) / (6 + 12 - 24 + 48 + 96 - 192 + 384 - 768) = 1 / 2 :=
by
  sorry

end computation_result_l59_59452


namespace middle_number_is_14_5_l59_59147

theorem middle_number_is_14_5 (x y z : ℝ) (h1 : x + y = 24) (h2 : x + z = 29) (h3 : y + z = 34) : y = 14.5 :=
sorry

end middle_number_is_14_5_l59_59147


namespace convert_base_3_to_base_10_l59_59947

theorem convert_base_3_to_base_10 : 
  (1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) = 142 :=
by
  sorry

end convert_base_3_to_base_10_l59_59947


namespace votes_for_eliot_l59_59520

theorem votes_for_eliot (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h_randy : randy_votes = 16)
  (h_shaun : shaun_votes = 5 * randy_votes)
  (h_eliot : eliot_votes = 2 * shaun_votes) :
  eliot_votes = 160 :=
by
  sorry

end votes_for_eliot_l59_59520


namespace find_natural_triples_l59_59959

theorem find_natural_triples (x y z : ℕ) : 
  (x+1) * (y+1) * (z+1) = 3 * x * y * z ↔ 
  (x, y, z) = (2, 2, 3) ∨ (x, y, z) = (2, 3, 2) ∨ (x, y, z) = (3, 2, 2) ∨
  (x, y, z) = (5, 1, 4) ∨ (x, y, z) = (5, 4, 1) ∨ (x, y, z) = (4, 1, 5) ∨ (x, y, z) = (4, 5, 1) ∨ 
  (x, y, z) = (1, 4, 5) ∨ (x, y, z) = (1, 5, 4) ∨ (x, y, z) = (8, 1, 3) ∨ (x, y, z) = (8, 3, 1) ∨
  (x, y, z) = (3, 1, 8) ∨ (x, y, z) = (3, 8, 1) ∨ (x, y, z) = (1, 3, 8) ∨ (x, y, z) = (1, 8, 3) :=
by {
  sorry
}

end find_natural_triples_l59_59959


namespace rectangle_area_is_30_l59_59997

def Point := (ℤ × ℤ)

def vertices : List Point := [(-5, 1), (1, 1), (1, -4), (-5, -4)]

theorem rectangle_area_is_30 :
  let length := (vertices[1].1 - vertices[0].1).natAbs
  let width := (vertices[0].2 - vertices[2].2).natAbs
  length * width = 30 := by
  sorry

end rectangle_area_is_30_l59_59997


namespace ratio_of_blue_to_red_l59_59556

variable (B : ℕ) -- Number of blue lights

def total_white := 59
def total_colored := total_white - 5
def red_lights := 12
def green_lights := 6

def total_bought := red_lights + green_lights + B

theorem ratio_of_blue_to_red (h : total_bought = total_colored) :
  B / red_lights = 3 :=
by
  sorry

end ratio_of_blue_to_red_l59_59556


namespace explicit_expression_solve_inequality_l59_59216

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := (n^2 - 3*n + 3) * x^(n+1)

theorem explicit_expression (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x)) :
  (∀ n x, f n x = x^3) :=
by
  sorry

theorem solve_inequality (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x))
  (f_eq : ∀ n x, f n x = x^3) :
  ∀ x, (x + 1)^3 + (3 - 2*x)^3 > 0 → x < 4 :=
by
  sorry

end explicit_expression_solve_inequality_l59_59216


namespace jerry_gets_logs_l59_59699

def logs_per_pine_tree : ℕ := 80
def logs_per_maple_tree : ℕ := 60
def logs_per_walnut_tree : ℕ := 100
def logs_per_oak_tree : ℕ := 90
def logs_per_birch_tree : ℕ := 55

def pine_trees_cut : ℕ := 8
def maple_trees_cut : ℕ := 3
def walnut_trees_cut : ℕ := 4
def oak_trees_cut : ℕ := 7
def birch_trees_cut : ℕ := 5

def total_logs : ℕ :=
  pine_trees_cut * logs_per_pine_tree +
  maple_trees_cut * logs_per_maple_tree +
  walnut_trees_cut * logs_per_walnut_tree +
  oak_trees_cut * logs_per_oak_tree +
  birch_trees_cut * logs_per_birch_tree

theorem jerry_gets_logs : total_logs = 2125 :=
by
  sorry

end jerry_gets_logs_l59_59699


namespace total_cookies_and_brownies_l59_59691

-- Define the conditions
def bagsOfCookies : ℕ := 272
def cookiesPerBag : ℕ := 45
def bagsOfBrownies : ℕ := 158
def browniesPerBag : ℕ := 32

-- Define the total cookies, total brownies, and total items
def totalCookies := bagsOfCookies * cookiesPerBag
def totalBrownies := bagsOfBrownies * browniesPerBag
def totalItems := totalCookies + totalBrownies

-- State the theorem to prove
theorem total_cookies_and_brownies : totalItems = 17296 := by
  sorry

end total_cookies_and_brownies_l59_59691


namespace sequence_v5_value_l59_59701

theorem sequence_v5_value (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) - v n)
  (h_v3 : v 3 = 17) (h_v6 : v 6 = 524) : v 5 = 198.625 :=
sorry

end sequence_v5_value_l59_59701


namespace sheila_tue_thu_hours_l59_59847

def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def total_hours_mwf : ℕ := hours_mwf * days_mwf

def weekly_earnings : ℕ := 360
def hourly_rate : ℕ := 10
def earnings_mwf : ℕ := total_hours_mwf * hourly_rate

def earnings_tue_thu : ℕ := weekly_earnings - earnings_mwf
def hours_tue_thu : ℕ := earnings_tue_thu / hourly_rate

theorem sheila_tue_thu_hours : hours_tue_thu = 12 := by
  -- proof omitted
  sorry

end sheila_tue_thu_hours_l59_59847


namespace distance_problem_l59_59919

theorem distance_problem (x y n : ℝ) (h1 : y = 15) (h2 : Real.sqrt ((x - 2) ^ 2 + (15 - 7) ^ 2) = 13) (h3 : x > 2) :
  n = Real.sqrt ((2 + Real.sqrt 105) ^ 2 + 15 ^ 2) := by
  sorry

end distance_problem_l59_59919


namespace fractional_eq_solve_simplify_and_evaluate_l59_59310

-- Question 1: Solve the fractional equation
theorem fractional_eq_solve (x : ℝ) (h1 : (x / (x + 1) = (2 * x) / (3 * x + 3) + 1)) : 
  x = -1.5 := 
sorry

-- Question 2: Simplify and evaluate the expression for x = -1
theorem simplify_and_evaluate (x : ℝ)
  (h2 : x ≠ 0) (h3 : x ≠ 2) (h4 : x ≠ -2) :
  (x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4) / ((x+2) / (x^3 - 4*x)) = 
  (x - 4) / (x - 2) ∧ 
  (x = -1) → ((x - 4) / (x - 2) = (5 / 3)) := 
sorry

end fractional_eq_solve_simplify_and_evaluate_l59_59310


namespace tiles_needed_l59_59876

-- Definitions for the problem
def width_wall : ℕ := 36
def length_wall : ℕ := 72
def width_tile : ℕ := 3
def length_tile : ℕ := 4

-- The area of the wall
def A_wall : ℕ := width_wall * length_wall

-- The area of one tile
def A_tile : ℕ := width_tile * length_tile

-- The number of tiles needed
def number_of_tiles : ℕ := A_wall / A_tile

-- Proof statement
theorem tiles_needed : number_of_tiles = 216 := by
  sorry

end tiles_needed_l59_59876


namespace no_possible_arrangement_of_balloons_l59_59290

/-- 
  There are 10 balloons hanging in a row: blue and green. This statement proves that it is impossible 
  to arrange 10 balloons such that between every two blue balloons, there is an even number of 
  balloons and between every two green balloons, there is an odd number of balloons.
--/

theorem no_possible_arrangement_of_balloons :
  ¬ (∃ (color : Fin 10 → Bool), 
    (∀ i j, i < j ∧ color i = color j ∧ color i = tt → (j - i - 1) % 2 = 0) ∧
    (∀ i j, i < j ∧ color i = color j ∧ color i = ff → (j - i - 1) % 2 = 1)) :=
by
  sorry

end no_possible_arrangement_of_balloons_l59_59290


namespace right_triangle_expression_l59_59512

theorem right_triangle_expression (a c b : ℝ) (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : 
  b^2 = 4 * (a + 1) :=
by
  sorry

end right_triangle_expression_l59_59512


namespace real_root_of_system_l59_59185

theorem real_root_of_system :
  (∃ x : ℝ, x^3 + 9 = 0 ∧ x + 3 = 0) ↔ x = -3 := 
by 
  sorry

end real_root_of_system_l59_59185


namespace unique_friends_count_l59_59036

-- Definitions from conditions
def M : ℕ := 10
def P : ℕ := 20
def G : ℕ := 5
def M_P : ℕ := 4
def M_G : ℕ := 2
def P_G : ℕ := 0
def M_P_G : ℕ := 2

-- Theorem we need to prove
theorem unique_friends_count : (M + P + G - M_P - M_G - P_G + M_P_G) = 31 := by
  sorry

end unique_friends_count_l59_59036


namespace probability_of_red_ball_l59_59998

-- Define the conditions
def num_balls : ℕ := 3
def red_balls : ℕ := 2
def white_balls : ℕ := 1

-- Calculate the probability
def probability_drawing_red_ball : ℚ := red_balls / num_balls

-- The theorem statement to be proven
theorem probability_of_red_ball : probability_drawing_red_ball = 2 / 3 :=
by
  sorry

end probability_of_red_ball_l59_59998


namespace ellipse_correct_l59_59799

noncomputable def ellipse_problem := 
  ∃ a b c : ℝ, 
  (a > b) ∧ (b > 0) ∧ 
  (2 * a = 4) ∧ 
  (c / a = real.sqrt 3 / 2) ∧ 
  (a ^ 2 = b ^ 2 + c ^ 2) ∧ 
  (∃ (x y : ℝ), ((x^2 / 4 + y^2 = 1) ∧ 
  (∃ (k m : ℝ), (4 * k ^ 2 - m ^ 2 + 1 > 0) ∧ 
  (5 * m^2 + 16 * k * m + 12 * k^2 = 0) ∧ 
  ((m = -2 * k ∨ m = -6/5 * k) → 
  (m = -6/5 * k) ∧ 
  (2,0) ≠ (6/5,0)))))

theorem ellipse_correct : ellipse_problem := 
  sorry

end ellipse_correct_l59_59799


namespace algae_free_day_22_l59_59104

def algae_coverage (day : ℕ) : ℝ :=
if day = 25 then 1 else 2 ^ (25 - day)

theorem algae_free_day_22 :
  1 - algae_coverage 22 = 0.875 :=
by
  -- Proof to be filled in
  sorry

end algae_free_day_22_l59_59104


namespace count_multiples_of_five_between_100_and_400_l59_59090

theorem count_multiples_of_five_between_100_and_400 :
  let multiples := {n : ℕ | 100 < n ∧ n < 400 ∧ n % 5 = 0} in
  ∃ (n : ℕ), n = 59 ∧ finset.card (finset.filter (λ x, x % 5 = 0) (finset.Ico 101 400)) = n :=
by sorry

end count_multiples_of_five_between_100_and_400_l59_59090


namespace find_k_l59_59808

noncomputable def arithmetic_sum (n : ℕ) (a1 d : ℚ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_k 
  (a1 d : ℚ) (k : ℕ)
  (h1 : arithmetic_sum (k - 2) a1 d = -4)
  (h2 : arithmetic_sum k a1 d = 0)
  (h3 : arithmetic_sum (k + 2) a1 d = 8) :
  k = 6 :=
by
  sorry

end find_k_l59_59808


namespace sum_of_distinct_roots_l59_59534

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l59_59534


namespace cost_of_each_teddy_bear_is_15_l59_59122

-- Definitions
variable (number_of_toys_cost_10 : ℕ := 28)
variable (cost_per_toy : ℕ := 10)
variable (number_of_teddy_bears : ℕ := 20)
variable (total_amount_in_wallet : ℕ := 580)

-- Theorem statement
theorem cost_of_each_teddy_bear_is_15 :
  (total_amount_in_wallet - (number_of_toys_cost_10 * cost_per_toy)) / number_of_teddy_bears = 15 :=
by
  -- proof goes here
  sorry

end cost_of_each_teddy_bear_is_15_l59_59122


namespace parabola_equation_l59_59085

theorem parabola_equation (x y : ℝ) (hx : x = -2) (hy : y = 3) :
  (y^2 = -(9 / 2) * x) ∨ (x^2 = (4 / 3) * y) :=
by
  sorry

end parabola_equation_l59_59085


namespace quadruple_perimeter_l59_59767

-- Define the rectangle's original and expanded dimensions and perimeters
def original_perimeter (a b : ℝ) := 2 * (a + b)
def new_perimeter (a b : ℝ) := 2 * ((4 * a) + (4 * b))

-- Statement to be proved
theorem quadruple_perimeter (a b : ℝ) : new_perimeter a b = 4 * original_perimeter a b :=
  sorry

end quadruple_perimeter_l59_59767


namespace card_deck_initial_count_l59_59608

theorem card_deck_initial_count 
  (r b : ℕ)
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + (b + 6)) = 1 / 5) : 
  r + b = 24 :=
by
  sorry

end card_deck_initial_count_l59_59608


namespace triangles_sticks_not_proportional_l59_59862

theorem triangles_sticks_not_proportional :
  ∀ (n_triangles n_sticks : ℕ), 
  (∃ k : ℕ, n_triangles = k * n_sticks) 
  ∨ 
  (∃ k : ℕ, n_triangles * n_sticks = k) 
  → False :=
by
  sorry

end triangles_sticks_not_proportional_l59_59862


namespace rational_numbers_countable_l59_59251

theorem rational_numbers_countable : ∃ (f : ℚ → ℕ), Function.Bijective f :=
by
  sorry

end rational_numbers_countable_l59_59251


namespace children_attended_l59_59174

theorem children_attended 
  (x y : ℕ) 
  (h₁ : x + y = 280) 
  (h₂ : 0.60 * x + 0.25 * y = 140) : 
  y = 80 := 
by
  sorry

end children_attended_l59_59174


namespace jovana_bucket_shells_l59_59832

theorem jovana_bucket_shells :
  let a0 := 5.2
  let a1 := a0 + 15.7
  let a2 := a1 + 17.5
  let a3 := a2 - 4.3
  let a4 := 3 * a3
  a4 = 102.3 := 
by
  sorry

end jovana_bucket_shells_l59_59832


namespace find_principal_l59_59029

def r : ℝ := 0.03
def t : ℝ := 3
def I (P : ℝ) : ℝ := P - 1820
def simple_interest (P : ℝ) : ℝ := P * r * t

theorem find_principal (P : ℝ) : simple_interest P = I P -> P = 2000 :=
by
  sorry

end find_principal_l59_59029


namespace max_omega_value_l59_59495

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

def center_of_symmetry (ω φ : ℝ) := 
  ∃ n : ℤ, ω * (-Real.pi / 4) + φ = n * Real.pi

def extremum_point (ω φ : ℝ) :=
  ∃ n' : ℤ, ω * (Real.pi / 4) + φ = n' * Real.pi + Real.pi / 2

def monotonic_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < b → a < y ∧ y < b → x ≤ y → f x ≤ f y

theorem max_omega_value (ω : ℝ) (φ : ℝ) : 
  (ω > 0) →
  (|φ| ≤ Real.pi / 2) →
  center_of_symmetry ω φ →
  extremum_point ω φ →
  monotonic_in_interval (f ω φ) (5 * Real.pi / 18) (2 * Real.pi / 5) →
  ω = 5 :=
by
  sorry

end max_omega_value_l59_59495


namespace solution_to_equation1_solution_to_equation2_l59_59964

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 1)^2 = 4
def equation2 (x : ℝ) : Prop := 3 * x^3 + 4 = -20

-- State the theorems with the correct answers
theorem solution_to_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = -3) :=
by
  sorry

theorem solution_to_equation2 (x : ℝ) : equation2 x ↔ (x = -2) :=
by
  sorry

end solution_to_equation1_solution_to_equation2_l59_59964


namespace lower_limit_b_l59_59358

theorem lower_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : b < 29) 
  (h4 : ∃ min_b max_b, min_b = 4 ∧ max_b ≤ 29 ∧ 3.75 = (16 : ℚ) / (min_b : ℚ) - (7 : ℚ) / (max_b : ℚ)) : 
  b ≥ 4 :=
sorry

end lower_limit_b_l59_59358


namespace perfect_squares_l59_59401

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l59_59401


namespace exist_n_div_k_l59_59528

open Function

theorem exist_n_div_k (k : ℕ) (h1 : k ≥ 1) (h2 : Nat.gcd k 6 = 1) :
  ∃ n : ℕ, n ≥ 0 ∧ k ∣ (2^n + 3^n + 6^n - 1) := 
sorry

end exist_n_div_k_l59_59528


namespace original_price_given_discounts_l59_59928

theorem original_price_given_discounts (p q d : ℝ) (h : d > 0) :
  ∃ x : ℝ, x * (1 + (p - q) / 100 - p * q / 10000) = d :=
by
  sorry

end original_price_given_discounts_l59_59928


namespace mean_computation_l59_59819

theorem mean_computation (x y : ℝ) 
  (h1 : (28 + x + 70 + 88 + 104) / 5 = 67)
  (h2 : (if x < 50 ∧ x < 62 then if y < 62 then ((28 + y) / 2 = 81) else ((62 + x) / 2 = 81) else if y < 50 then ((y + 50) / 2 = 81) else if y < 62 then ((50 + y) / 2 = 81) else ((50 + x) / 2 = 81)) -- conditions for median can be simplified and expanded as necessary
) : (50 + 62 + 97 + 124 + x + y) / 6 = 82.5 :=
sorry

end mean_computation_l59_59819


namespace minimum_disks_needed_l59_59700

theorem minimum_disks_needed :
  ∀ (n_files : ℕ) (disk_space : ℝ) (mb_files_1 : ℕ) (size_file_1 : ℝ) (mb_files_2 : ℕ) (size_file_2 : ℝ) (remaining_files : ℕ) (size_remaining_files : ℝ),
    n_files = 30 →
    disk_space = 1.5 →
    mb_files_1 = 4 →
    size_file_1 = 1.0 →
    mb_files_2 = 10 →
    size_file_2 = 0.6 →
    remaining_files = 16 →
    size_remaining_files = 0.5 →
    ∃ (min_disks : ℕ), min_disks = 13 :=
by
  sorry

end minimum_disks_needed_l59_59700


namespace diamond_comm_l59_59950

def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^2 - b^2

theorem diamond_comm (x y : ℝ) : diamond x y = diamond y x := by
  sorry

end diamond_comm_l59_59950


namespace tony_rope_length_l59_59880

-- Definitions based on the conditions in the problem
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss_per_knot : ℝ := 1.2
def number_of_knots : ℕ := 5

-- The final length of the rope after tying all pieces together and losing length per knot
def final_rope_length (lengths : List ℝ) (loss_per_knot : ℝ) (number_of_knots : ℕ) : ℝ :=
  List.sum lengths - (loss_per_knot * number_of_knots)

theorem tony_rope_length :
  final_rope_length rope_lengths knot_loss_per_knot number_of_knots = 35 := by
  sorry

end tony_rope_length_l59_59880


namespace double_acute_angle_l59_59081

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
sorry

end double_acute_angle_l59_59081


namespace abc_sum_is_twelve_l59_59706

theorem abc_sum_is_twelve
  (f : ℤ → ℤ)
  (a b c : ℕ)
  (h1 : f 1 = 10)
  (h2 : f 0 = 8)
  (h3 : f (-3) = -28)
  (h4 : ∀ x, x > 0 → f x = 2 * a * x + 6)
  (h5 : f 0 = a^2 * b)
  (h6 : ∀ x, x < 0 → f x = 2 * b * x + 2 * c)
  : a + b + c = 12 := sorry

end abc_sum_is_twelve_l59_59706


namespace solution_A_to_B_ratio_l59_59572

def ratio_solution_A_to_B (V_A V_B : ℝ) : Prop :=
  (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B) → V_A / V_B = 5 / 6

theorem solution_A_to_B_ratio (V_A V_B : ℝ) (h : (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B)) :
  V_A / V_B = 5 / 6 :=
sorry

end solution_A_to_B_ratio_l59_59572


namespace problem_solution_l59_59084

theorem problem_solution (x y : ℝ) (h1 : y = x / (3 * x + 1)) (hx : x ≠ 0) (hy : y ≠ 0) :
    (x - y + 3 * x * y) / (x * y) = 6 := by
  sorry

end problem_solution_l59_59084


namespace fraction_reducible_to_17_l59_59590

theorem fraction_reducible_to_17 (m n : ℕ) (h_coprime : Nat.gcd m n = 1)
  (h_reducible : ∃ d : ℕ, d ∣ (3 * m - n) ∧ d ∣ (5 * n + 2 * m)) :
  ∃ k : ℕ, (3 * m - n) / k = 17 ∧ (5 * n + 2 * m) / k = 17 :=
by
  have key : Nat.gcd (3 * m - n) (5 * n + 2 * m) = 17 := sorry
  -- using the result we need to construct our desired k
  use 17 / (Nat.gcd (3 * m - n) (5 * n + 2 * m))
  -- rest of intimate proof here
  sorry

end fraction_reducible_to_17_l59_59590


namespace find_number_l59_59155

theorem find_number (x : ℤ) (h : 3 * (3 * x) = 18) : x = 2 := 
sorry

end find_number_l59_59155


namespace television_screen_horizontal_length_l59_59123

theorem television_screen_horizontal_length :
  ∀ (d : ℝ) (r_l : ℝ) (r_h : ℝ), r_l / r_h = 4 / 3 → d = 27 → 
  let h := (3 / 5) * d
  let l := (4 / 5) * d
  l = 21.6 := by
  sorry

end television_screen_horizontal_length_l59_59123


namespace extreme_point_f_l59_59990

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 1)

theorem extreme_point_f :
  ∃ x : ℝ, (∀ y : ℝ, y ≠ 0 → (Real.exp y * y < 0 ↔ y < x)) ∧ x = 0 :=
by
  sorry

end extreme_point_f_l59_59990


namespace average_speed_l59_59769

-- Define the conditions
def initial_reading : ℕ := 2552
def final_reading : ℕ := 2992
def day1_time : ℕ := 6
def day2_time : ℕ := 8

-- Theorem: Proving the average speed is 31 miles per hour.
theorem average_speed :
  final_reading - initial_reading = 440 ∧ day1_time + day2_time = 14 ∧ 
  (final_reading - initial_reading) / (day1_time + day2_time) = 31 :=
by
  sorry

end average_speed_l59_59769


namespace base_conversion_l59_59393

theorem base_conversion (b : ℕ) (h_pos : b > 0) :
  (1 * 6 ^ 2 + 2 * 6 ^ 1 + 5 * 6 ^ 0 = 2 * b ^ 2 + 2 * b + 1) → b = 4 :=
by
  sorry

end base_conversion_l59_59393


namespace width_of_river_l59_59881

def ferry_problem (v1 v2 W t1 t2 : ℝ) : Prop :=
  v1 * t1 + v2 * t1 = W ∧
  v1 * t1 = 720 ∧
  v2 * t1 = W - 720 ∧
  (v1 * t2 + v2 * t2 = 3 * W) ∧
  v1 * t2 = 2 * W - 400 ∧
  v2 * t2 = W + 400

theorem width_of_river 
  (v1 v2 W t1 t2 : ℝ)
  (h : ferry_problem v1 v2 W t1 t2) :
  W = 1280 :=
by
  sorry

end width_of_river_l59_59881


namespace inequality_holds_l59_59270

theorem inequality_holds : ∀ (n : ℕ), (n - 1)^(n + 1) * (n + 1)^(n - 1) < n^(2 * n) :=
by sorry

end inequality_holds_l59_59270


namespace num_diagonals_octagon_l59_59643

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l59_59643


namespace perfect_square_trinomial_m_value_l59_59218

theorem perfect_square_trinomial_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ y : ℤ, y^2 + my + 9 = (y + a)^2) ↔ (m = 6 ∨ m = -6) :=
by
  sorry

end perfect_square_trinomial_m_value_l59_59218


namespace registration_methods_l59_59504

-- Define the number of students and groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem stating the total number of different registration methods
theorem registration_methods : (num_groups ^ num_students) = 81 := 
by sorry

end registration_methods_l59_59504


namespace number_of_solutions_eq_l59_59703

open Nat

theorem number_of_solutions_eq (n : ℕ) : 
  ∃ N, (∀ (x : ℝ), 1 ≤ x ∧ x ≤ n → x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2) → N = n^2 - n + 1 :=
by sorry

end number_of_solutions_eq_l59_59703


namespace possible_values_of_a_l59_59008

theorem possible_values_of_a :
  ∃ (a : ℤ), (∀ (b c : ℤ), (x : ℤ) → (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → (a = 6 ∨ a = 10) :=
sorry

end possible_values_of_a_l59_59008


namespace num_diagonals_octagon_l59_59641

theorem num_diagonals_octagon : 
  let n := 8
  (n * (n - 3)) / 2 = 20 := 
by
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * 5) / 2 : by rfl
    ... = 40 / 2 : by rfl
    ... = 20 : by rfl

end num_diagonals_octagon_l59_59641


namespace fraction_subtraction_l59_59025

theorem fraction_subtraction (x : ℝ) : (8000 * x - (0.05 / 100 * 8000) = 796) → x = 0.1 :=
by
  sorry

end fraction_subtraction_l59_59025


namespace tony_rope_length_l59_59877

-- Definition of the initial rope lengths and the number of ropes
def ropes : List ℝ := [8, 20, 2, 2, 2, 7]

-- Definition of the length lost per knot
def loss_per_knot : ℝ := 1.2

-- The expected total length of the rope after tying them all together
def total_length_after_knots : ℝ := 35

theorem tony_rope_length : 
  (ropes.sum - (ropes.length - 1) * loss_per_knot) = total_length_after_knots := 
by
  -- define sorry to skip the proof
  sorry

end tony_rope_length_l59_59877


namespace complex_number_in_first_quadrant_l59_59554

-- Definition of the imaginary unit
def i : ℂ := Complex.I

-- Definition of the complex number z
def z : ℂ := i * (1 - i)

-- Coordinates of the complex number z
def z_coords : ℝ × ℝ := (z.re, z.im)

-- Statement asserting that the point corresponding to z lies in the first quadrant
theorem complex_number_in_first_quadrant : z_coords.fst > 0 ∧ z_coords.snd > 0 := 
by
  sorry

end complex_number_in_first_quadrant_l59_59554


namespace solve_for_x_l59_59683

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l59_59683


namespace sum_distinct_vars_eq_1716_l59_59540

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l59_59540


namespace symmetric_polynomial_identity_l59_59191

variable (x y z : ℝ)
def σ1 : ℝ := x + y + z
def σ2 : ℝ := x * y + y * z + z * x
def σ3 : ℝ := x * y * z

theorem symmetric_polynomial_identity : 
  x^3 + y^3 + z^3 = σ1 x y z ^ 3 - 3 * σ1 x y z * σ2 x y z + 3 * σ3 x y z := by
  sorry

end symmetric_polynomial_identity_l59_59191


namespace smallest_positive_integer_solution_l59_59020

theorem smallest_positive_integer_solution : ∃ n : ℕ, 23 * n % 9 = 310 % 9 ∧ n = 8 :=
by
  sorry

end smallest_positive_integer_solution_l59_59020


namespace simplest_form_l59_59336

theorem simplest_form (b : ℝ) (h : b ≠ 2) : 2 - (2 / (2 + b / (2 - b))) = 4 / (4 - b) :=
by sorry

end simplest_form_l59_59336


namespace solve_for_x_l59_59684

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l59_59684


namespace socks_problem_l59_59825

def number_of_same_color_pairs (white black red : ℕ) : ℕ :=
  combinatorics.nat.choose white 2 + combinatorics.nat.choose black 2 + combinatorics.nat.choose red 2

theorem socks_problem : 
  number_of_same_color_pairs 5 4 3 = 19 :=
by
  sorry

end socks_problem_l59_59825


namespace inequality_solution_l59_59277

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

noncomputable def lhs (x : ℝ) := 
  log_b 5 250 + ((4 - (log_b 5 2) ^ 2) / (2 + log_b 5 2))

noncomputable def rhs (x : ℝ) := 
  125 ^ (log_b 5 x) ^ 2 - 24 * x ^ (log_b 5 x)

theorem inequality_solution (x : ℝ) : 
  (lhs x <= rhs x) ↔ (0 < x ∧ x ≤ 1/5) ∨ (5 ≤ x) := 
sorry

end inequality_solution_l59_59277


namespace question1_question2_l59_59349

-- Question 1
theorem question1 (a : ℝ) (h : a = 1 / 2) :
  let A := {x | -1 / 2 < x ∧ x < 2}
  let B := {x | 0 < x ∧ x < 1}
  A ∩ B = {x | 0 < x ∧ x < 1} :=
by
  sorry

-- Question 2
theorem question2 (a : ℝ) :
  let A := {x | a - 1 < x ∧ x < 2 * a + 1}
  let B := {x | 0 < x ∧ x < 1}
  (A ∩ B = ∅) ↔ (a ≤ -1/2 ∨ a ≥ 2) :=
by
  sorry

end question1_question2_l59_59349


namespace total_tickets_used_l59_59129

theorem total_tickets_used :
  let shooting_game_cost := 5
  let carousel_cost := 3
  let jen_games := 2
  let russel_rides := 3
  let jen_total := shooting_game_cost * jen_games
  let russel_total := carousel_cost * russel_rides
  jen_total + russel_total = 19 :=
by
  -- proof goes here
  sorry

end total_tickets_used_l59_59129


namespace probability_is_correct_l59_59606

def num_red : ℕ := 7
def num_green : ℕ := 9
def num_yellow : ℕ := 10
def num_blue : ℕ := 5
def num_purple : ℕ := 3

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue + num_purple

def num_blue_or_purple : ℕ := num_blue + num_purple

-- Probability of selecting a blue or purple jelly bean
def probability_blue_or_purple : ℚ := num_blue_or_purple / total_jelly_beans

theorem probability_is_correct :
  probability_blue_or_purple = 4 / 17 := sorry

end probability_is_correct_l59_59606


namespace lending_rate_is_7_percent_l59_59764

-- Conditions
def principal : ℝ := 5000
def borrowing_rate : ℝ := 0.04  -- 4% p.a. simple interest
def time : ℕ := 2  -- 2 years
def gain_per_year : ℝ := 150

-- Proof of the final statement
theorem lending_rate_is_7_percent :
  let borrowing_interest := principal * borrowing_rate * time / 100
  let interest_per_year := borrowing_interest / 2
  let total_interest_earned_per_year := interest_per_year + gain_per_year
  (total_interest_earned_per_year * 100) / principal = 7 :=
by
  sorry

end lending_rate_is_7_percent_l59_59764


namespace man_swim_upstream_distance_l59_59043

theorem man_swim_upstream_distance (c d : ℝ) (h1 : 15.5 + c ≠ 0) (h2 : 15.5 - c ≠ 0) :
  (15.5 + c) * 2 = 36 ∧ (15.5 - c) * 2 = d → d = 26 := by
  sorry

end man_swim_upstream_distance_l59_59043


namespace restaurant_total_pizzas_and_hotdogs_in_June_l59_59613

theorem restaurant_total_pizzas_and_hotdogs_in_June
  (hotdogs_daily : ℕ)
  (extra_pizzas : ℕ)
  (days_in_June : ℕ)
  (hotdogs_daily = 60)
  (extra_pizzas = 40)
  (days_in_June = 30) :
  (hotdogs_daily + extra_pizzas) * days_in_June = 4800 :=
by
  sorry

end restaurant_total_pizzas_and_hotdogs_in_June_l59_59613


namespace mmobile_additional_line_cost_l59_59557

noncomputable def cost_tmobile (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * 16

noncomputable def cost_mmobile (x : ℕ) (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * x

theorem mmobile_additional_line_cost
  (x : ℕ)
  (ht : cost_tmobile 5 = 98)
  (hm : cost_tmobile 5 - cost_mmobile x 5 = 11) :
  x = 14 :=
by
  sorry

end mmobile_additional_line_cost_l59_59557


namespace combinations_x_eq_2_or_8_l59_59663

theorem combinations_x_eq_2_or_8 (x : ℕ) (h_pos : 0 < x) (h_comb : Nat.choose 10 x = Nat.choose 10 2) : x = 2 ∨ x = 8 :=
sorry

end combinations_x_eq_2_or_8_l59_59663


namespace minimum_quadratic_value_l59_59500

theorem minimum_quadratic_value (h : ℝ) (x : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → (x - h)^2 + 1 ≥ 10) ∧ (∃ x, 1 ≤ x ∧ x ≤ 3 ∧ (x - h)^2 + 1 = 10) 
  ↔ h = -2 ∨ h = 6 :=
by
  sorry

end minimum_quadratic_value_l59_59500


namespace area_of_square_plot_l59_59298

theorem area_of_square_plot (s : ℕ) (price_per_foot total_cost: ℕ)
  (h_price : price_per_foot = 58)
  (h_total_cost : total_cost = 3944) :
  (s * s = 289) :=
by
  sorry

end area_of_square_plot_l59_59298


namespace simplify_sum1_simplify_sum2_l59_59848

theorem simplify_sum1 : 296 + 297 + 298 + 299 + 1 + 2 + 3 + 4 = 1200 := by
  sorry

theorem simplify_sum2 : 457 + 458 + 459 + 460 + 461 + 462 + 463 = 3220 := by
  sorry

end simplify_sum1_simplify_sum2_l59_59848


namespace arithmetic_proof_l59_59902

theorem arithmetic_proof : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end arithmetic_proof_l59_59902


namespace simplify_fraction_144_1008_l59_59133

theorem simplify_fraction_144_1008 :
  (144 : ℤ) / (1008 : ℤ) = (1 : ℤ) / (7 : ℤ) :=
by
  sorry

end simplify_fraction_144_1008_l59_59133


namespace gcd_1617_1225_gcd_2023_111_gcd_589_6479_l59_59199

theorem gcd_1617_1225 : Nat.gcd 1617 1225 = 49 :=
by
  sorry

theorem gcd_2023_111 : Nat.gcd 2023 111 = 1 :=
by
  sorry

theorem gcd_589_6479 : Nat.gcd 589 6479 = 589 :=
by
  sorry

end gcd_1617_1225_gcd_2023_111_gcd_589_6479_l59_59199


namespace positive_number_eq_576_l59_59307

theorem positive_number_eq_576 (x : ℝ) (h : 0 < x) (h_eq : (2 / 3) * x = (25 / 216) * (1 / x)) : x = 5.76 := 
by 
  sorry

end positive_number_eq_576_l59_59307


namespace second_quadrant_set_l59_59782

-- Define the set P of points in the second quadrant
def P : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 }

-- Statement of the problem: Prove that this definition accurately describes the set of all points in the second quadrant
theorem second_quadrant_set :
  P = { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 } :=
by
  sorry

end second_quadrant_set_l59_59782


namespace magnitude_of_z_l59_59096

theorem magnitude_of_z (z : ℂ) (h : z * (1 + 2 * Complex.I) + Complex.I = 0) : 
  Complex.abs z = Real.sqrt (5) / 5 := 
sorry

end magnitude_of_z_l59_59096


namespace local_extrema_l59_59720

-- Defining the function y = 1 + 3x - x^3
def y (x : ℝ) : ℝ := 1 + 3 * x - x ^ 3

-- Statement of the problem to be proved
theorem local_extrema :
  (∃ x : ℝ, x = -1 ∧ y x = -1 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 1) < δ → y z ≥ y (-1)) ∧
  (∃ x : ℝ, x = 1 ∧ y x = 3 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z - 1) < δ → y z ≤ y 1) :=
by sorry

end local_extrema_l59_59720


namespace inscribed_circle_radius_isosceles_triangle_l59_59364

noncomputable def isosceles_triangle_base : ℝ := 30 -- base AC
noncomputable def isosceles_triangle_equal_side : ℝ := 39 -- equal sides AB and BC

theorem inscribed_circle_radius_isosceles_triangle :
  ∀ (AC AB BC: ℝ), 
  AC = isosceles_triangle_base → 
  AB = isosceles_triangle_equal_side →
  BC = isosceles_triangle_equal_side →
  ∃ r : ℝ, r = 10 := 
by
  intros AC AB BC hAC hAB hBC
  sorry

end inscribed_circle_radius_isosceles_triangle_l59_59364


namespace sum_of_distinct_roots_l59_59535

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l59_59535


namespace suff_not_necc_condition_l59_59159

theorem suff_not_necc_condition (x : ℝ) : (x=2) → ((x-2) * (x+5) = 0) ∧ ¬((x-2) * (x+5) = 0 → x=2) :=
by {
  sorry
}

end suff_not_necc_condition_l59_59159


namespace reinforcement_arrival_days_l59_59316

theorem reinforcement_arrival_days (x : ℕ) (h : x = 2000) (provisions_days : ℕ) (provisions_days_initial : provisions_days = 54) 
(reinforcement : ℕ) (reinforcement_val : reinforcement = 1300) (remaining_days : ℕ) (remaining_days_val : remaining_days = 20) 
(total_men : ℕ) (total_men_val : total_men = 3300) (equation : 2000 * (54 - x) = 3300 * 20) : x = 21 := 
by
  have eq1 : 2000 * 54 - 2000 * x = 3300 * 20 := by sorry
  have eq2 : 108000 - 2000 * x = 66000 := by sorry
  have eq3 : 2000 * x = 42000 := by sorry
  have eq4 : x = 21000 / 2000 := by sorry
  have eq5 : x = 21 := by sorry
  sorry

end reinforcement_arrival_days_l59_59316


namespace largest_integral_value_of_y_l59_59961

theorem largest_integral_value_of_y : 
  (1 / 4 : ℝ) < (y / 7 : ℝ) ∧ (y / 7 : ℝ) < (3 / 5 : ℝ) → y ≤ 4 :=
by
  sorry

end largest_integral_value_of_y_l59_59961


namespace complement_intersection_M_N_l59_59260

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x > -1}
def U : Set ℝ := Set.univ

theorem complement_intersection_M_N :
  U \ (M ∩ N) = {x | x ≤ -1} ∪ {x | x ≥ 3} :=
by
  sorry

end complement_intersection_M_N_l59_59260


namespace coeff_x3_l59_59344

noncomputable def M (n : ℕ) : ℝ := (5 * (1:ℝ) - (1:ℝ)^(1/2)) ^ n
noncomputable def N (n : ℕ) : ℝ := 2 ^ n

theorem coeff_x3 (n : ℕ) (h : M n - N n = 240) : 
  (M 3) = 150 := sorry

end coeff_x3_l59_59344


namespace hyperbola_asymptotes_identical_l59_59062

theorem hyperbola_asymptotes_identical (x y M : ℝ) :
  (∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ (y = (b/a) * x ∨ y = -(b/a) * x)) ∧
  (∃ (c d : ℝ), c = 5 ∧ y = (c / d) * x ∨ y = -(c / d) * x) →
  M = (225 / 16) :=
by sorry

end hyperbola_asymptotes_identical_l59_59062


namespace scale_division_l59_59320

theorem scale_division (total_feet : ℕ) (inches_extra : ℕ) (part_length : ℕ) (total_parts : ℕ) :
  total_feet = 6 → inches_extra = 8 → part_length = 20 → 
  total_parts = (6 * 12 + 8) / 20 → total_parts = 4 :=
by
  intros
  sorry

end scale_division_l59_59320


namespace pqrs_sum_l59_59532

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l59_59532


namespace find_base_of_triangle_l59_59786

def triangle_base (area : ℝ) (height : ℝ) (base : ℝ) : Prop :=
  area = (base * height) / 2

theorem find_base_of_triangle : triangle_base 24 8 6 :=
by
  -- Simplification and computation steps are omitted as per the instruction
  sorry

end find_base_of_triangle_l59_59786


namespace algebraic_expression_l59_59192

-- Define a variable x
variable (x : ℝ)

-- State the theorem
theorem algebraic_expression : (5 * x - 3) = 5 * x - 3 :=
by
  sorry

end algebraic_expression_l59_59192


namespace find_contaminated_constant_l59_59597

theorem find_contaminated_constant (contaminated_constant : ℝ) (x : ℝ) 
  (h1 : 2 * (x - 3) - contaminated_constant = x + 1) 
  (h2 : x = 9) : contaminated_constant = 2 :=
  sorry

end find_contaminated_constant_l59_59597


namespace max_n_for_factored_polynomial_l59_59469

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l59_59469


namespace division_of_fractions_l59_59054

theorem division_of_fractions : (4 : ℚ) / (5 / 7) = 28 / 5 := sorry

end division_of_fractions_l59_59054


namespace range_of_a_l59_59001

theorem range_of_a (a : ℝ) (h_decreasing : ∀ x y : ℝ, x < y → (a-1)^x > (a-1)^y) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l59_59001


namespace sum_xyz_eq_10_l59_59209

theorem sum_xyz_eq_10 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * x * y + 3 * x * y * z = 115) : 
  x + y + z = 10 :=
sorry

end sum_xyz_eq_10_l59_59209


namespace line_equation_is_correct_l59_59139

noncomputable def line_has_equal_intercepts_and_passes_through_A (p q : ℝ) : Prop :=
(p, q) = (3, 2) ∧ q ≠ 0 ∧ (∃ c : ℝ, p + q = c ∨ 2 * p - 3 * q = 0)

theorem line_equation_is_correct :
  line_has_equal_intercepts_and_passes_through_A 3 2 → 
  (∃ f g : ℝ, 2 * f - 3 * g = 0 ∨ f + g = 5) :=
by
  sorry

end line_equation_is_correct_l59_59139


namespace oxygen_mass_percentage_is_58_3_l59_59474

noncomputable def C_molar_mass := 12.01
noncomputable def H_molar_mass := 1.01
noncomputable def O_molar_mass := 16.0

noncomputable def molar_mass_C6H8O7 :=
  6 * C_molar_mass + 8 * H_molar_mass + 7 * O_molar_mass

noncomputable def O_mass := 7 * O_molar_mass

noncomputable def oxygen_mass_percentage_C6H8O7 :=
  (O_mass / molar_mass_C6H8O7) * 100

theorem oxygen_mass_percentage_is_58_3 :
  oxygen_mass_percentage_C6H8O7 = 58.3 := by
  sorry

end oxygen_mass_percentage_is_58_3_l59_59474


namespace solution_set_of_inequality_l59_59633

theorem solution_set_of_inequality
  (a b : ℝ)
  (h1 : a < 0) 
  (h2 : b / a = 1) :
  { x : ℝ | (x - 1) * (a * x + b) < 0 } = { x : ℝ | x < -1 } ∪ {x : ℝ | 1 < x} :=
by
  sorry

end solution_set_of_inequality_l59_59633


namespace total_playtime_l59_59833

noncomputable def lena_playtime_minutes : ℕ := 210
noncomputable def brother_playtime_minutes (lena_playtime: ℕ) : ℕ := lena_playtime + 17
noncomputable def sister_playtime_minutes (brother_playtime: ℕ) : ℕ := 2 * brother_playtime

theorem total_playtime
  (lena_playtime : ℕ)
  (brother_playtime : ℕ)
  (sister_playtime : ℕ)
  (h_lena : lena_playtime = lena_playtime_minutes)
  (h_brother : brother_playtime = brother_playtime_minutes lena_playtime)
  (h_sister : sister_playtime = sister_playtime_minutes brother_playtime) :
  lena_playtime + brother_playtime + sister_playtime = 891 := 
  by sorry

end total_playtime_l59_59833


namespace remainder_when_13_plus_y_divided_by_31_l59_59552

theorem remainder_when_13_plus_y_divided_by_31
  (y : ℕ)
  (hy : 7 * y % 31 = 1) :
  (13 + y) % 31 = 22 :=
sorry

end remainder_when_13_plus_y_divided_by_31_l59_59552


namespace decreasing_function_range_l59_59087

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x + 7 * a - 2 else a ^ x

theorem decreasing_function_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 1 / 2) := 
by
  intro a
  sorry

end decreasing_function_range_l59_59087


namespace find_solution_l59_59066

theorem find_solution : ∀ (x : Real), (sqrt[3](5 - x) = -5 / 2) → x = 165 / 8 :=
by
  sorry    -- Proof is omitted

end find_solution_l59_59066


namespace decimal_equivalent_one_half_pow_five_l59_59593

theorem decimal_equivalent_one_half_pow_five :
  (1 / 2) ^ 5 = 0.03125 :=
by sorry

end decimal_equivalent_one_half_pow_five_l59_59593


namespace no_high_quality_triangle_exist_high_quality_quadrilateral_l59_59522

-- Define the necessary predicate for a number being a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the property of being a high-quality triangle
def high_quality_triangle (a b c : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + a)

-- Define the property of non-existence of a high-quality triangle
theorem no_high_quality_triangle (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) : 
  ¬high_quality_triangle a b c := by sorry

-- Define the property of being a high-quality quadrilateral
def high_quality_quadrilateral (a b c d : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + d) ∧ is_perfect_square (d + a)

-- Define the property of existence of a high-quality quadrilateral
theorem exist_high_quality_quadrilateral (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) : 
  high_quality_quadrilateral a b c d := by sorry

end no_high_quality_triangle_exist_high_quality_quadrilateral_l59_59522


namespace option_D_not_equal_l59_59156

def frac1 := (-15 : ℚ) / 12
def fracA := (-30 : ℚ) / 24
def fracB := -1 - (3 : ℚ) / 12
def fracC := -1 - (9 : ℚ) / 36
def fracD := -1 - (5 : ℚ) / 15
def fracE := -1 - (25 : ℚ) / 100

theorem option_D_not_equal :
  fracD ≠ frac1 := 
sorry

end option_D_not_equal_l59_59156


namespace abs_fraction_lt_one_l59_59626

theorem abs_fraction_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) : 
  |(x - y) / (1 - x * y)| < 1 := 
sorry

end abs_fraction_lt_one_l59_59626


namespace probability_both_universities_visited_l59_59927

open ProbabilityTheory

theorem probability_both_universities_visited :
  (∃ (students : Fin 4 → Bool),
    (∃ i, students i = true) ∧ (∃ j, students j = false)) → 
  (1 - ((1 / 2) ^ 4 + (1 / 2) ^ 4)) = 7 / 8 :=
by
  sorry

end probability_both_universities_visited_l59_59927


namespace salt_solution_problem_l59_59092

theorem salt_solution_problem
  (x y : ℝ)
  (h1 : 70 + x + y = 200)
  (h2 : 0.20 * 70 + 0.60 * x + 0.35 * y = 0.45 * 200) :
  x = 122 ∧ y = 8 :=
by
  sorry

end salt_solution_problem_l59_59092


namespace instantaneous_velocity_at_2_l59_59581

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 + t^2

-- State the problem: Prove the instantaneous velocity at t = 2 is 4
theorem instantaneous_velocity_at_2 : (deriv s) 2 = 4 := by
  sorry

end instantaneous_velocity_at_2_l59_59581


namespace find_m3_minus_2mn_plus_n3_l59_59233

theorem find_m3_minus_2mn_plus_n3 (m n : ℝ) (h1 : m^2 = n + 2) (h2 : n^2 = m + 2) (h3 : m ≠ n) : m^3 - 2 * m * n + n^3 = -2 := by
  sorry

end find_m3_minus_2mn_plus_n3_l59_59233


namespace octagon_diagonals_20_l59_59650

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l59_59650


namespace min_val_of_expression_l59_59787

noncomputable def min_val_expr : ℝ → ℝ :=
  λ x, 
  (((Real.sin x + Real.csc x) ^ 2 + (Real.cos x + Real.sec x) ^ 2) * Real.exp (Real.sin x + Real.cos x))

theorem min_val_of_expression : (∀ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.sin x + Real.cos x > 1 → min_val_expr x ≥ 9 * Real.exp (Real.sqrt 2)) ∧
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.sin x + Real.cos x > 1 ∧ min_val_expr x = 9 * Real.exp (Real.sqrt 2)) :=
begin
  sorry
end

end min_val_of_expression_l59_59787


namespace simplify_expression_l59_59564

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l59_59564


namespace T_10_equals_2_pow_45_l59_59366

-- Define the geometric sequence and arithmetic sequence conditions
def geom_sequence (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ n, a (n + 1) = q * a n

def arith_seq (b c d : ℝ) : Prop := 2 * c = b + d

-- Define the initial conditions
def a : ℕ → ℝ := λ n, 2^(n - 1)

theorem T_10_equals_2_pow_45 : 
  geom_sequence a ∧ 
  a 1 = 1 ∧ 
  arith_seq (4 * a 3) (2 * a 4) (a 5) ∧ 
  T n = ∏ i in fin.range n, a i →
  T 10 = 2^45 :=
by
  -- We state the theorem assumptions
  intro h1 h2 h3 h4,
  -- State proof here.
  sorry

end T_10_equals_2_pow_45_l59_59366


namespace circle_diameter_l59_59905

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d, d = 4 :=
by
  let r := Real.sqrt 4
  let d := 2 * r
  use d
  simp only [Real.sqrt_eq_rfl, mul_eq_zero, ne.def, not_false_iff]
  linarith
  sorry

end circle_diameter_l59_59905


namespace triangle_inequality_l59_59220

variable (a b c : ℝ)

theorem triangle_inequality (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0) (h₅ : a + b > c) (h₆ : b + c > a) (h₇ : c + a > b) : 
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 := 
sorry

end triangle_inequality_l59_59220


namespace xiao_zhang_winning_probability_max_expected_value_l59_59157

-- Definitions for the conditions
variables (a b c : ℕ)
variable (h_sum : a + b + c = 6)

-- Main theorem statement 1: Probability of Xiao Zhang winning
theorem xiao_zhang_winning_probability (h_sum : a + b + c = 6) :
  (3 * a + 2 * b + c) / 36 = a / 6 * 3 / 6 + b / 6 * 2 / 6 + c / 6 * 1 / 6 :=
sorry

-- Main theorem statement 2: Maximum expected value of Xiao Zhang's score
theorem max_expected_value (h_sum : a + b + c = 6) :
  (3 * a + 4 * b + 3 * c) / 36 = (1 / 2 + b / 36) →  (a = 0 ∧ b = 6 ∧ c = 0) :=
sorry

end xiao_zhang_winning_probability_max_expected_value_l59_59157


namespace area_of_isosceles_right_triangle_l59_59012

def is_isosceles_right_triangle (X Y Z : Type*) : Prop :=
∃ (XY YZ XZ : ℝ), XY = 6.000000000000001 ∧ XY > YZ ∧ YZ = XZ ∧ XY = YZ * Real.sqrt 2

theorem area_of_isosceles_right_triangle
  {X Y Z : Type*}
  (h : is_isosceles_right_triangle X Y Z) :
  ∃ A : ℝ, A = 9.000000000000002 :=
by
  sorry

end area_of_isosceles_right_triangle_l59_59012


namespace number_of_divisions_l59_59048

-- Definitions
def hour_in_seconds : ℕ := 3600

def is_division (n m : ℕ) : Prop :=
  n * m = hour_in_seconds ∧ n > 0 ∧ m > 0

-- Proof problem statement
theorem number_of_divisions : ∃ (count : ℕ), count = 44 ∧ 
  (∀ (n m : ℕ), is_division n m → ∃ (d : ℕ), d = count) :=
sorry

end number_of_divisions_l59_59048


namespace max_value_of_n_l59_59460

-- Define the main variables and conditions
noncomputable def max_n : ℕ := 
  let n_pairs := [(1, 72), (2, 36), (3, 24), (4, 18), (6, 12), (8, 9)] 
  max (n_pairs.map (λ p, 6 * p.2 + p.1))

-- Lean theorem statement for the equivalence
theorem max_value_of_n :
  max_n = 433 := by
  sorry

end max_value_of_n_l59_59460


namespace intersection_P_Q_l59_59811

open Set

noncomputable def P : Set ℝ := {x | abs (x - 1) < 4}
noncomputable def Q : Set ℝ := {x | ∃ y, y = Real.log (x + 2) }

theorem intersection_P_Q :
  (P ∩ Q) = {x : ℝ | -2 < x ∧ x < 5} :=
by
  sorry

end intersection_P_Q_l59_59811


namespace determine_sum_l59_59170

theorem determine_sum (P R : ℝ) (h : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 78) : 
  P = 2600 :=
sorry

end determine_sum_l59_59170


namespace best_in_district_round_l59_59713

-- Assume a structure that lets us refer to positions
inductive Position
| first
| second
| third
| last

open Position

-- Definitions of the statements
def Eva (p : Position → Prop) := ¬ (p first) ∧ ¬ (p last)
def Mojmir (p : Position → Prop) := ¬ (p last)
def Karel (p : Position → Prop) := p first
def Peter (p : Position → Prop) := p last

-- The main hypothesis
def exactly_one_lie (p : Position → Prop) :=
  (Eva p ∧ Mojmir p ∧ Karel p ∧ ¬ (Peter p)) ∨
  (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∨
  (Eva p ∧ ¬ (Mojmir p) ∧ Karel p ∧ Peter p) ∨
  (¬ (Eva p) ∧ Mojmir p ∧ Karel p ∧ Peter p)

theorem best_in_district_round :
  ∃ (p : Position → Prop),
    (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∧ exactly_one_lie p :=
by
  sorry

end best_in_district_round_l59_59713


namespace max_value_of_n_l59_59458

-- Define the main variables and conditions
noncomputable def max_n : ℕ := 
  let n_pairs := [(1, 72), (2, 36), (3, 24), (4, 18), (6, 12), (8, 9)] 
  max (n_pairs.map (λ p, 6 * p.2 + p.1))

-- Lean theorem statement for the equivalence
theorem max_value_of_n :
  max_n = 433 := by
  sorry

end max_value_of_n_l59_59458


namespace initial_walnuts_l59_59439

theorem initial_walnuts (W : ℕ) (boy_effective : ℕ) (girl_effective : ℕ) (total_walnuts : ℕ) :
  boy_effective = 5 → girl_effective = 3 → total_walnuts = 20 → W + boy_effective + girl_effective = total_walnuts → W = 12 :=
by
  intros h_boy h_girl h_total h_eq
  rw [h_boy, h_girl, h_total] at h_eq
  linarith

end initial_walnuts_l59_59439


namespace solve_inequality_l59_59334

theorem solve_inequality (x : ℝ) : 
  (x ≠ 1) → ( (x^3 - 3*x^2 + 2*x + 1) / (x^2 - 2*x + 1) ≤ 2 ) ↔ 
  (2 - Real.sqrt 3 < x ∧ x < 1) ∨ (1 < x ∧ x < 2 + Real.sqrt 3) := 
sorry

end solve_inequality_l59_59334


namespace find_r_l59_59631

theorem find_r (r : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = 4) → 
  (∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = r^2) →
  (∀ x1 y1 x2 y2: ℝ, 
    (x2 - x1)^2 + (y2 - y1)^2 = 25) →
  (2 + |r| = 5) →
  (r = 3 ∨ r = -3) :=
by
  sorry

end find_r_l59_59631


namespace largest_value_of_n_l59_59466

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l59_59466


namespace square_in_S_l59_59835

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n

def S (n : ℕ) : Prop :=
  is_sum_of_two_squares (n - 1) ∧ is_sum_of_two_squares n ∧ is_sum_of_two_squares (n + 1)

theorem square_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end square_in_S_l59_59835


namespace total_fruits_l59_59930

theorem total_fruits (Mike_fruits Matt_fruits Mark_fruits : ℕ)
  (Mike_receives : Mike_fruits = 3)
  (Matt_receives : Matt_fruits = 2 * Mike_fruits)
  (Mark_receives : Mark_fruits = Mike_fruits + Matt_fruits) :
  Mike_fruits + Matt_fruits + Mark_fruits = 18 := by
  sorry

end total_fruits_l59_59930


namespace sum_of_distinct_real_numbers_l59_59545

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l59_59545


namespace probability_of_square_product_l59_59421

theorem probability_of_square_product :
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9 -- (1,1), (1,4), (2,2), (4,1), (3,3), (9,1), (4,4), (5,5), (6,6)
  favorable_outcomes / total_outcomes = 1 / 8 :=
by
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9
  have h1 : favorable_outcomes / total_outcomes = 1 / 8 := sorry
  exact h1

end probability_of_square_product_l59_59421


namespace union_complement_eq_l59_59372

/-- The universal set U and sets A and B as given in the problem. -/
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

/-- The lean statement of our proof problem. -/
theorem union_complement_eq : A ∪ (U \ B) = {0, 1, 2, 3} := by
  sorry

end union_complement_eq_l59_59372


namespace min_m_n_l59_59851

theorem min_m_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 108 * m = n^3) : m + n = 8 :=
sorry

end min_m_n_l59_59851


namespace pqrs_sum_l59_59531

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l59_59531


namespace slope_angle_range_l59_59688

open Real

theorem slope_angle_range (m : ℝ) (θ : ℝ) (h0 : 0 ≤ θ) (h1 : θ < π) 
    (hslope : tan θ = 1 - m^2) : θ ∈ set.Icc 0 (π / 4) ∪ set.Ioo (π / 2) π :=
by
  -- We are not providing the proof, so adding sorry
  sorry

end slope_angle_range_l59_59688


namespace find_b_l59_59980

theorem find_b (a b c y1 y2 : ℝ) (h1 : y1 = a * 2^2 + b * 2 + c) 
              (h2 : y2 = a * (-2)^2 + b * (-2) + c) 
              (h3 : y1 - y2 = -12) : b = -3 :=
by 
  sorry

end find_b_l59_59980


namespace vector_b_value_l59_59812

theorem vector_b_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, -2)
  2 • a + b = (3, 2) → b = (1, -2) :=
by
  intros
  sorry

end vector_b_value_l59_59812


namespace octagon_diagonals_20_l59_59651

-- Define what an octagon is
def is_octagon (n : ℕ) : Prop := n = 8

-- Define the formula to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in an octagon is 20
theorem octagon_diagonals_20 (n : ℕ) (h : is_octagon n) : num_diagonals n = 20 :=
by
  rw [is_octagon] at h
  rw [h]
  simp [num_diagonals]
  sorry

end octagon_diagonals_20_l59_59651


namespace perfect_squares_l59_59399

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l59_59399


namespace metal_contest_winner_l59_59330

theorem metal_contest_winner (x y : ℕ) (hx : 95 * x + 74 * y = 2831) : x = 15 ∧ y = 19 ∧ 95 * 15 > 74 * 19 := by
  sorry

end metal_contest_winner_l59_59330


namespace range_of_m_l59_59507

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) :=
by
  sorry

end range_of_m_l59_59507


namespace vertex_of_quadratic_l59_59562

theorem vertex_of_quadratic (x : ℝ) : 
  (y : ℝ) = -2 * (x + 1) ^ 2 + 3 →
  (∃ vertex_x vertex_y : ℝ, vertex_x = -1 ∧ vertex_y = 3 ∧ y = -2 * (vertex_x + 1) ^ 2 + vertex_y) :=
by
  intro h
  exists -1, 3
  simp [h]
  sorry

end vertex_of_quadratic_l59_59562


namespace students_with_uncool_parents_l59_59243

def total_students : ℕ := 40
def cool_dads_count : ℕ := 18
def cool_moms_count : ℕ := 20
def both_cool_count : ℕ := 10

theorem students_with_uncool_parents :
  total_students - (cool_dads_count + cool_moms_count - both_cool_count) = 12 :=
by sorry

end students_with_uncool_parents_l59_59243


namespace div_by_20_l59_59271

theorem div_by_20 (n : ℕ) : 20 ∣ (9 ^ (8 * n + 4) - 7 ^ (8 * n + 4)) :=
  sorry

end div_by_20_l59_59271


namespace range_of_m_l59_59796

theorem range_of_m (m : ℝ) : (∃ x y : ℝ, 2 * x^2 - 3 * x + m = 0 ∧ 2 * y^2 - 3 * y + m = 0) → m ≤ 9 / 8 :=
by
  intro h
  -- We need to implement the proof here
  sorry

end range_of_m_l59_59796


namespace silver_value_percentage_l59_59052

theorem silver_value_percentage
  (side_length : ℝ) (weight_per_cubic_inch : ℝ) (price_per_ounce : ℝ) 
  (selling_price : ℝ) (volume : ℝ) (weight : ℝ) (silver_value : ℝ) 
  (percentage_sold : ℝ ) 
  (h1 : side_length = 3) 
  (h2 : weight_per_cubic_inch = 6) 
  (h3 : price_per_ounce = 25)
  (h4 : selling_price = 4455)
  (h5 : volume = side_length^3)
  (h6 : weight = volume * weight_per_cubic_inch)
  (h7 : silver_value = weight * price_per_ounce)
  (h8 : percentage_sold = (selling_price / silver_value) * 100) :
  percentage_sold = 110 :=
by
  sorry

end silver_value_percentage_l59_59052


namespace tony_rope_length_l59_59878

-- Definition of the initial rope lengths and the number of ropes
def ropes : List ℝ := [8, 20, 2, 2, 2, 7]

-- Definition of the length lost per knot
def loss_per_knot : ℝ := 1.2

-- The expected total length of the rope after tying them all together
def total_length_after_knots : ℝ := 35

theorem tony_rope_length : 
  (ropes.sum - (ropes.length - 1) * loss_per_knot) = total_length_after_knots := 
by
  -- define sorry to skip the proof
  sorry

end tony_rope_length_l59_59878


namespace find_a_b_sum_l59_59413

theorem find_a_b_sum
  (a b : ℝ)
  (h1 : 2 * a = -6)
  (h2 : a ^ 2 - b = 1) :
  a + b = 5 :=
by
  sorry

end find_a_b_sum_l59_59413


namespace janet_used_clips_correct_l59_59111

-- Define the initial number of paper clips
def initial_clips : ℕ := 85

-- Define the remaining number of paper clips
def remaining_clips : ℕ := 26

-- Define the number of clips Janet used
def used_clips (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

-- The theorem to state the correctness of the calculation
theorem janet_used_clips_correct : used_clips initial_clips remaining_clips = 59 :=
by
  -- Lean proof goes here
  sorry

end janet_used_clips_correct_l59_59111


namespace binom_26_6_l59_59080

theorem binom_26_6 :
  nat.binomial 26 6 = 290444 :=
by
  have h23_5 : nat.binomial 23 5 = 33649 := rfl
  have h23_6 : nat.binomial 23 6 = 42504 := rfl
  have h23_7 : nat.binomial 23 7 = 53130 := rfl
  have h24_6 : nat.binomial 24 6 = 33649 + 42504 := by rw [h23_5, h23_6]
  have h24_7 : nat.binomial 24 7 = 42504 + 53130 := by rw [h23_6, h23_7]
  have h24_5 : nat.binomial 24 5 = 42504 := by rw [rfl] -- This is provided as a result from previous calculations.
  have h25_6 : nat.binomial 25 6 = 42504 + 76153 := by rw [h24_6]
  have h25_7 : nat.binomial 25 7 = 76153 + 95634 := by rw [h24_6, h24_7]
  rw [h25_6, h25_7]
  sorry -- Proof copy paste ends here and the final steps can be calculated.

end binom_26_6_l59_59080


namespace initial_cats_l59_59842

-- Define the conditions as hypotheses
variables (total_cats now : ℕ) (cats_given : ℕ)

-- State the main theorem
theorem initial_cats:
  total_cats = 31 → cats_given = 14 → (total_cats - cats_given) = 17 :=
by sorry

end initial_cats_l59_59842


namespace largest_is_three_l59_59010

variable (p q r : ℝ)

def cond1 : Prop := p + q + r = 3
def cond2 : Prop := p * q + p * r + q * r = 1
def cond3 : Prop := p * q * r = -6

theorem largest_is_three
  (h1 : cond1 p q r)
  (h2 : cond2 p q r)
  (h3 : cond3 p q r) :
  p = 3 ∨ q = 3 ∨ r = 3 := sorry

end largest_is_three_l59_59010


namespace instantaneous_velocity_at_t2_l59_59611

noncomputable def s (t : ℝ) : ℝ := t^3 - t^2 + 2 * t

theorem instantaneous_velocity_at_t2 : 
  deriv s 2 = 10 := 
by
  sorry

end instantaneous_velocity_at_t2_l59_59611


namespace largest_five_digit_number_is_99200_l59_59885

def largest_five_digit_number_sum_20 (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n < 100000) ∧ (n.digits.sum = 20)

theorem largest_five_digit_number_is_99200 : ∃ n : ℕ, largest_five_digit_number_sum_20 n ∧ n = 99200 :=
by
  sorry

end largest_five_digit_number_is_99200_l59_59885


namespace perfect_squares_lt_10_pow_7_and_multiple_of_36_l59_59235

theorem perfect_squares_lt_10_pow_7_and_multiple_of_36 :
  ∃ (n : ℕ), card { m : ℕ | m > 0 ∧ m ^ 2 < 10^7 ∧ 36 ∣ m ^ 2 } = 87 :=
by
  sorry

end perfect_squares_lt_10_pow_7_and_multiple_of_36_l59_59235


namespace exists_common_plane_l59_59131

-- Definition of the triangular pyramids
structure Pyramid :=
(base_area : ℝ)
(height : ℝ)

-- Function to represent the area of the intersection produced by a horizontal plane at distance x from the table
noncomputable def sectional_area (P : Pyramid) (x : ℝ) : ℝ :=
  P.base_area * (1 - x / P.height) ^ 2

-- Given seven pyramids
variables {P1 P2 P3 P4 P5 P6 P7 : Pyramid}

-- For any three pyramids, there exists a horizontal plane that intersects them in triangles of equal area
axiom triple_intersection:
  ∀ (Pi Pj Pk : Pyramid), ∃ x : ℝ, x ≥ 0 ∧ x ≤ min (Pi.height) (min (Pj.height) (Pk.height)) ∧
    sectional_area Pi x = sectional_area Pj x ∧ sectional_area Pk x = sectional_area Pi x

-- Prove that there exists a plane that intersects all seven pyramids in triangles of equal area
theorem exists_common_plane :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ min P1.height (min P2.height (min P3.height (min P4.height (min P5.height (min P6.height P7.height))))) ∧
    sectional_area P1 x = sectional_area P2 x ∧
    sectional_area P2 x = sectional_area P3 x ∧
    sectional_area P3 x = sectional_area P4 x ∧
    sectional_area P4 x = sectional_area P5 x ∧
    sectional_area P5 x = sectional_area P6 x ∧
    sectional_area P6 x = sectional_area P7 x :=
sorry

end exists_common_plane_l59_59131


namespace brick_width_l59_59162

theorem brick_width (length_courtyard : ℕ) (width_courtyard : ℕ) (num_bricks : ℕ) (brick_length : ℕ) (total_area : ℕ) (brick_area : ℕ) (w : ℕ)
  (h1 : length_courtyard = 1800)
  (h2 : width_courtyard = 1200)
  (h3 : num_bricks = 30000)
  (h4 : brick_length = 12)
  (h5 : total_area = length_courtyard * width_courtyard)
  (h6 : total_area = num_bricks * brick_area)
  (h7 : brick_area = brick_length * w) :
  w = 6 :=
by
  sorry

end brick_width_l59_59162


namespace tg_half_angle_inequality_l59_59308

variable (α β γ : ℝ)

theorem tg_half_angle_inequality 
  (h : α + β + γ = 180) : 
  (Real.tan (α / 2)) * (Real.tan (β / 2)) * (Real.tan (γ / 2)) ≤ (Real.sqrt 3) / 9 := 
sorry

end tg_half_angle_inequality_l59_59308


namespace recreation_proof_l59_59114

noncomputable def recreation_percentage_last_week (W : ℝ) (P : ℝ) :=
  let last_week_spent := (P/100) * W
  let this_week_wages := (70/100) * W
  let this_week_spent := (20/100) * this_week_wages
  this_week_spent = (70/100) * last_week_spent

theorem recreation_proof :
  ∀ (W : ℝ), recreation_percentage_last_week W 20 :=
by
  intros
  sorry

end recreation_proof_l59_59114


namespace usual_time_to_catch_bus_l59_59883

theorem usual_time_to_catch_bus (S T : ℝ) (h : S / (4 / 5 * S) = (T + 3) / T) : T = 12 :=
by 
  sorry

end usual_time_to_catch_bus_l59_59883


namespace repeating_decimal_as_fraction_l59_59333

-- Define repeating decimal 0.7(3) as x
def x := 0.7 + 3 / 10 ^ (2 + n) where n is some natural number

theorem repeating_decimal_as_fraction :
    x = 11 / 15 := sorry

end repeating_decimal_as_fraction_l59_59333


namespace number_of_restaurants_l59_59637

def first_restaurant_meals_per_day := 20
def second_restaurant_meals_per_day := 40
def third_restaurant_meals_per_day := 50
def total_meals_per_week := 770

theorem number_of_restaurants :
  (first_restaurant_meals_per_day * 7) + 
  (second_restaurant_meals_per_day * 7) + 
  (third_restaurant_meals_per_day * 7) = total_meals_per_week → 
  3 = 3 :=
by 
  intros h
  sorry

end number_of_restaurants_l59_59637


namespace solve_first_sales_amount_l59_59918

noncomputable def first_sales_amount
  (S : ℝ) (R : ℝ) (next_sales_royalties : ℝ) (next_sales_amount : ℝ) : Prop :=
  (3 = R * S) ∧ (next_sales_royalties = 0.85 * R * next_sales_amount)

theorem solve_first_sales_amount (S R : ℝ) :
  first_sales_amount S R 9 108 → S = 30.6 :=
by
  intro h
  sorry

end solve_first_sales_amount_l59_59918


namespace circle_diameter_length_l59_59910

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end circle_diameter_length_l59_59910


namespace sqrt_x_minus_2_meaningful_l59_59589

theorem sqrt_x_minus_2_meaningful (x : ℝ) (hx : x = 0 ∨ x = -1 ∨ x = -2 ∨ x = 2) : (x = 2) ↔ (x - 2 ≥ 0) :=
by
  sorry

end sqrt_x_minus_2_meaningful_l59_59589


namespace number_of_chickens_l59_59331

theorem number_of_chickens (c k : ℕ) (h1 : c + k = 120) (h2 : 2 * c + 4 * k = 350) : c = 65 :=
by sorry

end number_of_chickens_l59_59331


namespace function_even_l59_59705

theorem function_even (n : ℤ) (h : 30 ∣ n)
    (h_prop: (1 : ℝ)^n^2 + (-1: ℝ)^n^2 = 2 * ((1: ℝ)^n + (-1: ℝ)^n - 1)) :
    ∀ x : ℝ, (x^n = (-x)^n) :=
by
    sorry

end function_even_l59_59705


namespace max_f_value_l59_59201

noncomputable def f (x : ℝ) : ℝ := 9 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 15 :=
by
  sorry

end max_f_value_l59_59201


namespace fair_game_x_value_l59_59508

theorem fair_game_x_value (x : ℕ) (h : x + 2 * x + 2 * x = 15) : x = 3 := 
by sorry

end fair_game_x_value_l59_59508


namespace sum_between_52_and_53_l59_59117

theorem sum_between_52_and_53 (x y : ℝ) (h1 : y = 4 * (⌊x⌋ : ℝ) + 2) (h2 : y = 5 * (⌊x - 3⌋ : ℝ) + 7) (h3 : ∀ n : ℤ, x ≠ n) :
  52 < x + y ∧ x + y < 53 := 
sorry

end sum_between_52_and_53_l59_59117


namespace remainder_77_pow_77_minus_15_mod_19_l59_59477

theorem remainder_77_pow_77_minus_15_mod_19 : (77^77 - 15) % 19 = 5 := by
  sorry

end remainder_77_pow_77_minus_15_mod_19_l59_59477


namespace quadratic_inequality_solution_set_l59_59083

-- Define the necessary variables and conditions
variable (a b c α β : ℝ)
variable (h1 : 0 < α)
variable (h2 : α < β)
variable (h3 : ∀ x : ℝ, (a * x^2 + b * x + c > 0) ↔ (α < x ∧ x < β))

-- Statement to be proved
theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, ((a + c - b) * x^2 + (b - 2 * a) * x + a > 0) ↔ ((1 / (1 + β) < x) ∧ (x < 1 / (1 + α))) :=
sorry

end quadratic_inequality_solution_set_l59_59083


namespace largest_value_of_n_l59_59465

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l59_59465


namespace circle_diameter_length_l59_59909

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end circle_diameter_length_l59_59909


namespace smallest_value_of_x_l59_59070

theorem smallest_value_of_x (x : ℝ) (hx : |3 * x + 7| = 26) : x = -11 :=
sorry

end smallest_value_of_x_l59_59070


namespace find_xyz_l59_59342

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + x * y * z = 15) :
  x * y * z = 9 / 2 := by
  sorry

end find_xyz_l59_59342


namespace parabola_focus_l59_59795

theorem parabola_focus (F : ℝ × ℝ) :
  (∀ (x y : ℝ), y^2 = 4 * x → (x + 1)^2 + y^2 = ((x - F.1)^2 + (y - F.2)^2)) → 
  F = (1, 0) :=
sorry

end parabola_focus_l59_59795


namespace solve_for_x_l59_59674

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l59_59674


namespace find_x_when_y_equals_two_l59_59675

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l59_59675


namespace find_f500_l59_59527

variable (f : ℕ → ℕ)
variable (h : ∀ x y : ℕ, f (x * y) = f x + f y)
variable (h₁ : f 10 = 16)
variable (h₂ : f 40 = 24)

theorem find_f500 : f 500 = 44 :=
sorry

end find_f500_l59_59527


namespace apples_on_tree_now_l59_59273

-- Definitions based on conditions
def initial_apples : ℕ := 11
def apples_picked : ℕ := 7
def new_apples : ℕ := 2

-- Theorem statement proving the final number of apples on the tree
theorem apples_on_tree_now : initial_apples - apples_picked + new_apples = 6 := 
by 
  sorry

end apples_on_tree_now_l59_59273


namespace range_of_a_l59_59282

noncomputable def f : ℝ → ℝ := sorry
variable (f_even : ∀ x : ℝ, f x = f (-x))
variable (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
variable (a : ℝ) (h : f a ≤ f 2)

theorem range_of_a (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
                   (h : f a ≤ f 2) :
                   a ≤ -2 ∨ a ≥ 2 :=
sorry

end range_of_a_l59_59282


namespace fraction_equation_l59_59813

theorem fraction_equation (P Q : ℕ) (h1 : 4 / 7 = P / 49) (h2 : 4 / 7 = 84 / Q) : P + Q = 175 :=
by
  sorry

end fraction_equation_l59_59813


namespace perfect_squares_l59_59402

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l59_59402


namespace find_square_plot_area_l59_59297

noncomputable def side_length (cost_per_foot : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (4 * cost_per_foot)

noncomputable def area_of_square_plot (cost_per_foot : ℝ) (total_cost : ℝ) : ℝ :=
  let s := side_length cost_per_foot total_cost
  s^2

theorem find_square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) :
  cost_per_foot = 58 → total_cost = 3944 → area_of_square_plot cost_per_foot total_cost = 289 :=
by
  intros h_cost h_total
  rw [area_of_square_plot, side_length]
  rw [h_cost, h_total]
  norm_num
  sorry

end find_square_plot_area_l59_59297


namespace binomial_coeff_12_3_l59_59943

/-- The binomial coefficient is defined as: 
  \binom{n}{k} = \frac{n!}{k!(n-k)!} -/
theorem binomial_coeff_12_3 : Nat.binom 12 3 = 220 := by
  sorry

end binomial_coeff_12_3_l59_59943


namespace sum_of_cubes_l59_59004

theorem sum_of_cubes (a b : ℕ) (h1 : 2 * x = a) (h2 : 3 * x = b) (h3 : b - a = 3) : a^3 + b^3 = 945 := by
  sorry

end sum_of_cubes_l59_59004


namespace nested_sqrt_simplification_l59_59987

theorem nested_sqrt_simplification (y : ℝ) (hy : y ≥ 0) : 
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt y)) = y^(9/4) := 
sorry

end nested_sqrt_simplification_l59_59987


namespace meatballs_fraction_each_son_eats_l59_59872

theorem meatballs_fraction_each_son_eats
  (f1 f2 f3 : ℝ)
  (h1 : ∃ f1 f2 f3, f1 + f2 + f3 = 2)
  (meatballs_initial : ∀ n, n = 3) :
  f1 = 2/3 ∧ f2 = 2/3 ∧ f3 = 2/3 := by
  sorry

end meatballs_fraction_each_son_eats_l59_59872


namespace number_of_incorrect_inequalities_l59_59967

theorem number_of_incorrect_inequalities (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (ite (|a| > |b|) 0 1) + (ite (a < b) 0 1) + (ite (a + b < ab) 0 1) + (ite (a^3 > b^3) 0 1) = 3 :=
sorry

end number_of_incorrect_inequalities_l59_59967


namespace average_marks_l59_59923

variable (P C M : ℕ)

theorem average_marks :
  P = 140 →
  (P + M) / 2 = 90 →
  (P + C) / 2 = 70 →
  (P + C + M) / 3 = 60 :=
by
  intros hP hM hC
  sorry

end average_marks_l59_59923


namespace max_pencils_l59_59830

theorem max_pencils 
  (p : ℕ → ℝ)
  (h_price1 : ∀ n : ℕ, n ≤ 10 → p n = 0.75 * n)
  (h_price2 : ∀ n : ℕ, n > 10 → p n = 0.75 * 10 + 0.65 * (n - 10))
  (budget : ℝ) (h_budget : budget = 10) :
  ∃ n : ℕ, p n ≤ budget ∧ (∀ m : ℕ, p m ≤ budget → m ≤ 13) :=
by {
  sorry
}

end max_pencils_l59_59830


namespace apples_distribution_l59_59234

theorem apples_distribution (total_apples : ℝ) (apples_per_person : ℝ) (number_of_people : ℝ) 
    (h1 : total_apples = 45) (h2 : apples_per_person = 15.0) : number_of_people = 3 :=
by
  sorry

end apples_distribution_l59_59234


namespace find_x_l59_59682

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l59_59682


namespace maximum_marks_l59_59032

noncomputable def passing_mark (M : ℝ) : ℝ := 0.35 * M

theorem maximum_marks (M : ℝ) (h1 : passing_mark M = 210) : M = 600 :=
  by
  sorry

end maximum_marks_l59_59032


namespace female_democrats_count_l59_59419

-- Define the parameters and conditions
variables (F M D_f D_m D_total : ℕ)
variables (h1 : F + M = 840)
variables (h2 : D_total = 1/3 * (F + M))
variables (h3 : D_f = 1/2 * F)
variables (h4 : D_m = 1/4 * M)
variables (h5 : D_total = D_f + D_m)

-- State the theorem
theorem female_democrats_count : D_f = 140 :=
by
  sorry

end female_democrats_count_l59_59419


namespace average_salary_all_workers_l59_59392

-- Definitions based on the conditions
def technicians_avg_salary := 16000
def rest_avg_salary := 6000
def total_workers := 35
def technicians := 7
def rest_workers := total_workers - technicians

-- Prove that the average salary of all workers is 8000
theorem average_salary_all_workers :
  (technicians * technicians_avg_salary + rest_workers * rest_avg_salary) / total_workers = 8000 := by
  sorry

end average_salary_all_workers_l59_59392


namespace tickets_per_friend_l59_59266

-- Defining the conditions
def initial_tickets := 11
def remaining_tickets := 3
def friends := 4

-- Statement to prove
theorem tickets_per_friend (h_tickets_given : initial_tickets - remaining_tickets = 8) : (initial_tickets - remaining_tickets) / friends = 2 :=
by
  sorry

end tickets_per_friend_l59_59266


namespace emily_card_sequence_l59_59063

/--
Emily orders her playing cards continuously in the following sequence:
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, 3, ...

Prove that the 58th card in this sequence is 6.
-/
theorem emily_card_sequence :
  (58 % 13 = 6) := by
  -- The modulo operation determines the position of the card in the cycle
  sorry

end emily_card_sequence_l59_59063


namespace remainder_of_sum_division_l59_59745

def a1 : ℕ := 2101
def a2 : ℕ := 2103
def a3 : ℕ := 2105
def a4 : ℕ := 2107
def a5 : ℕ := 2109
def n : ℕ := 12

theorem remainder_of_sum_division : ((a1 + a2 + a3 + a4 + a5) % n) = 1 :=
by {
  sorry
}

end remainder_of_sum_division_l59_59745


namespace frog_eats_per_day_l59_59073

-- Definition of the constants
def flies_morning : ℕ := 5
def flies_afternoon : ℕ := 6
def escaped_flies : ℕ := 1
def weekly_required_flies : ℕ := 14
def days_in_week : ℕ := 7

-- Prove that the frog eats 2 flies per day
theorem frog_eats_per_day : (flies_morning + flies_afternoon - escaped_flies) * days_in_week + 4 = 14 → (14 / days_in_week = 2) :=
by
  sorry

end frog_eats_per_day_l59_59073


namespace find_nonzero_q_for_quadratic_l59_59196

theorem find_nonzero_q_for_quadratic :
  ∃ (q : ℝ), q ≠ 0 ∧ (∀ (x1 x2 : ℝ), (q * x1^2 - 8 * x1 + 2 = 0 ∧ q * x2^2 - 8 * x2 + 2 = 0) → x1 = x2) ↔ q = 8 :=
by
  sorry

end find_nonzero_q_for_quadratic_l59_59196


namespace repeating_decimal_division_l59_59015

theorem repeating_decimal_division :
  let x := 0 + 54 / 99 in -- 0.545454... = 54/99 = 6/11
  let y := 0 + 18 / 99 in -- 0.181818... = 18/99 = 2/11
  x / y = 3 :=
by
  sorry

end repeating_decimal_division_l59_59015


namespace sqrt_product_simplification_l59_59324

variable (q : ℝ)
variable (hq : q ≥ 0)

theorem sqrt_product_simplification : 
  (Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q)) = 21 * q * Real.sqrt (2 * q) := 
  sorry

end sqrt_product_simplification_l59_59324


namespace sum_distinct_vars_eq_1716_l59_59539

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l59_59539


namespace point_A_in_QuadrantIII_l59_59999

-- Define the Cartesian Point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition for point being in Quadrant III
def inQuadrantIII (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Given point A
def A : Point := { x := -1, y := -2 }

-- The theorem stating that point A lies in Quadrant III
theorem point_A_in_QuadrantIII : inQuadrantIII A :=
  by
    sorry

end point_A_in_QuadrantIII_l59_59999


namespace suzanna_distance_ridden_l59_59852

theorem suzanna_distance_ridden (rate_per_5minutes : ℝ) (time_minutes : ℕ) (total_distance : ℝ) (units_per_interval : ℕ) (interval_distance : ℝ) :
  rate_per_5minutes = 0.75 → time_minutes = 45 → units_per_interval = 5 → interval_distance = 0.75 → total_distance = (time_minutes / units_per_interval) * interval_distance → total_distance = 6.75 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end suzanna_distance_ridden_l59_59852


namespace tangent_line_curve_l59_59362

theorem tangent_line_curve (a b : ℝ)
  (h1 : ∀ (x : ℝ), (x - (x^2 + a*x + b) + 1 = 0) ↔ (a = 1 ∧ b = 1))
  (h2 : ∀ (y : ℝ), (0, y) ∈ { p : ℝ × ℝ | p.2 = 0 ^ 2 + a * 0 + b }) :
  a = 1 ∧ b = 1 :=
by
  sorry

end tangent_line_curve_l59_59362


namespace solve_price_of_meat_l59_59247

def price_of_meat_per_ounce (x : ℕ) : Prop :=
  16 * x - 30 = 8 * x + 18

theorem solve_price_of_meat : ∃ x, price_of_meat_per_ounce x ∧ x = 6 :=
by
  sorry

end solve_price_of_meat_l59_59247


namespace percentage_two_sections_cleared_l59_59363

noncomputable def total_candidates : ℕ := 1200
def pct_cleared_all_sections : ℝ := 0.05
def pct_cleared_none_sections : ℝ := 0.05
def pct_cleared_one_section : ℝ := 0.25
def pct_cleared_four_sections : ℝ := 0.20
def cleared_three_sections : ℕ := 300

theorem percentage_two_sections_cleared :
  (total_candidates - total_candidates * (pct_cleared_all_sections + pct_cleared_none_sections + pct_cleared_one_section + pct_cleared_four_sections) - cleared_three_sections) / total_candidates * 100 = 20 := by
  sorry

end percentage_two_sections_cleared_l59_59363


namespace rectangular_field_area_l59_59305

theorem rectangular_field_area (a b c : ℕ) (h1 : a = 15) (h2 : c = 17)
  (h3 : a * a + b * b = c * c) : a * b = 120 := by
  sorry

end rectangular_field_area_l59_59305


namespace front_wheel_more_revolutions_l59_59394

theorem front_wheel_more_revolutions
  (c_f : ℕ) (c_b : ℕ) (d : ℕ)
  (H1 : c_f = 30) (H2 : c_b = 32) (H3 : d = 2400) :
  let F := d / c_f,
      B := d / c_b in
  F - B = 5 :=
by
  let F := 2400 / 30
  let B := 2400 / 32
  have H4 : F = 80 := by sorry
  have H5 : B = 75 := by sorry
  have H6 : F - B = 5 := by
    calc
      F - B = 80 - 75 : by rw [H4, H5]
          ... = 5 : by norm_num
  exact H6

end front_wheel_more_revolutions_l59_59394


namespace factor_polynomial_l59_59619

theorem factor_polynomial : 
  (x : ℝ) → (x^2 - 6 * x + 9 - 49 * x^4) = (-7 * x^2 + x - 3) * (7 * x^2 + x - 3) :=
by
  sorry

end factor_polynomial_l59_59619


namespace trajectory_moving_circle_l59_59022

theorem trajectory_moving_circle : 
  (∃ P : ℝ × ℝ, (∃ r : ℝ, (P.1 + 1)^2 = r^2 ∧ (P.1 - 2)^2 + P.2^2 = (r + 1)^2) ∧
  P.2^2 = 8 * P.1) :=
sorry

end trajectory_moving_circle_l59_59022


namespace min_tiles_needed_l59_59446

theorem min_tiles_needed : 
  ∀ (tile_length tile_width region_length region_width: ℕ),
  tile_length = 5 → 
  tile_width = 6 → 
  region_length = 3 * 12 → 
  region_width = 4 * 12 → 
  (region_length * region_width) / (tile_length * tile_width) ≤ 58 :=
by
  intros tile_length tile_width region_length region_width h_tile_length h_tile_width h_region_length h_region_width
  sorry

end min_tiles_needed_l59_59446


namespace diophantine_solution_exists_if_prime_divisor_l59_59834

theorem diophantine_solution_exists_if_prime_divisor (b : ℕ) (hb : 0 < b) (gcd_b_6 : Nat.gcd b 6 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 3 / (b : ℚ))) ↔ 
  ∃ p : ℕ, Nat.Prime p ∧ (∃ k : ℕ, p = 6 * k - 1) ∧ p ∣ b := 
by 
  sorry

end diophantine_solution_exists_if_prime_divisor_l59_59834


namespace diameter_of_circle_l59_59908

theorem diameter_of_circle (A : ℝ) (h : A = 4 * real.pi) : ∃ d : ℝ, d = 4 :=
  sorry

end diameter_of_circle_l59_59908


namespace blocks_from_gallery_to_work_l59_59448

theorem blocks_from_gallery_to_work (b_store b_gallery b_already_walked b_more_to_work total_blocks blocks_to_work_from_gallery : ℕ) 
  (h1 : b_store = 11)
  (h2 : b_gallery = 6)
  (h3 : b_already_walked = 5)
  (h4 : b_more_to_work = 20)
  (h5 : total_blocks = b_store + b_gallery + b_more_to_work)
  (h6 : blocks_to_work_from_gallery = total_blocks - b_already_walked - b_store - b_gallery) :
  blocks_to_work_from_gallery = 15 :=
by
  sorry

end blocks_from_gallery_to_work_l59_59448


namespace smallest_positive_period_f_interval_monotonically_decreasing_f_area_of_triangle_ABC_l59_59345

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x) ^ 2 - Real.sqrt 3

theorem smallest_positive_period_f :
  ∃ T > 0, T = π ∧ ∀ x, f (x + T) = f x :=
sorry

theorem interval_monotonically_decreasing_f :
  ∀ k : ℤ, (k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12) → (∀ y z : ℝ, y < z → y ∈ (k * π + π / 12, k * π + 7 * π / 12) → z ∈ (k * π + π / 12, k * π + 7 * π / 12) → f y ≥ f z)
:=
sorry

theorem area_of_triangle_ABC (a b c A B C : ℝ) (h_triangle : Geometry.Triangle a b c A B C)
  (h_a : a = 7)
  (h_f : f (A / 2 - π / 6) = Real.sqrt 3)
  (h_sin_sum : Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14)
  (h_acute : 0 < A ∧ A < π / 2) :
  Geometry.TriangleArea a b c = 10 * Real.sqrt 3 :=
sorry

end smallest_positive_period_f_interval_monotonically_decreasing_f_area_of_triangle_ABC_l59_59345


namespace B_N_Q_collinear_l59_59491

/-- Define point positions -/
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 0⟩
def N : Point := ⟨1, 0⟩

/-- Define the curve C -/
def on_curve_C (P : Point) : Prop :=
  P.x^2 + P.y^2 - 6 * P.x + 1 = 0

/-- Define reflection of point A across the x-axis -/
def reflection_across_x (A : Point) : Point :=
  ⟨A.x, -A.y⟩

/-- Define the condition that line l passes through M and intersects curve C at two distinct points A and B -/
def line_l_condition (A B: Point) (k : ℝ) (hk : k ≠ 0) : Prop :=
  A.y = k * (A.x + 1) ∧ B.y = k * (B.x + 1) ∧ on_curve_C A ∧ on_curve_C B

/-- Main theorem to prove collinearity of B, N, Q -/
theorem B_N_Q_collinear (A B : Point) (k : ℝ) (hk : k ≠ 0)
  (hA : on_curve_C A) (hB : on_curve_C B)
  (h_l : line_l_condition A B k hk) :
  let Q := reflection_across_x A
  (B.x - N.x) * (Q.y - N.y) = (B.y - N.y) * (Q.x - N.x) :=
sorry

end B_N_Q_collinear_l59_59491


namespace function_satisfies_condition_l59_59197

noncomputable def f : ℕ → ℕ := sorry

theorem function_satisfies_condition (f : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → f (n + 1) > (f n + f (f n)) / 2) :
  (∃ b : ℕ, ∀ n : ℕ, (n < b → f n = n) ∧ (n ≥ b → f n = n + 1)) :=
sorry

end function_satisfies_condition_l59_59197


namespace product_of_digits_base8_of_12345_is_0_l59_59423

def base8_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else Nat.digits 8 n 

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_digits_base8_of_12345_is_0 :
  product_of_digits (base8_representation 12345) = 0 := 
sorry

end product_of_digits_base8_of_12345_is_0_l59_59423


namespace greatest_num_consecutive_integers_sum_eq_36_l59_59595

theorem greatest_num_consecutive_integers_sum_eq_36 :
    ∃ a : ℤ, ∃ N : ℕ, N > 0 ∧ (N = 9) ∧ (N * (2 * a + N - 1) = 72) :=
sorry

end greatest_num_consecutive_integers_sum_eq_36_l59_59595


namespace sophia_fraction_of_book_finished_l59_59135

variable (x : ℕ)

theorem sophia_fraction_of_book_finished (h1 : x + (x + 90) = 270) : (x + 90) / 270 = 2 / 3 := by
  sorry

end sophia_fraction_of_book_finished_l59_59135


namespace remainder_13_plus_y_l59_59550

theorem remainder_13_plus_y :
  (∃ y : ℕ, (0 < y) ∧ (7 * y ≡ 1 [MOD 31])) → (∃ y : ℕ, (13 + y ≡ 22 [MOD 31])) :=
by 
  sorry

end remainder_13_plus_y_l59_59550


namespace number_of_girls_l59_59037

theorem number_of_girls (B G : ℕ) (h1 : B + G = 400) 
  (h2 : 0.60 * B = (6 / 10 : ℝ) * B) 
  (h3 : 0.80 * G = (8 / 10 : ℝ) * G) 
  (h4 : (6 / 10 : ℝ) * B + (8 / 10 : ℝ) * G = (65 / 100 : ℝ) * 400) : G = 100 := by
sorry

end number_of_girls_l59_59037


namespace restaurant_table_difference_l59_59602

theorem restaurant_table_difference :
  ∃ (N O : ℕ), N + O = 40 ∧ 6 * N + 4 * O = 212 ∧ (N - O) = 12 :=
by
  sorry

end restaurant_table_difference_l59_59602


namespace train_times_l59_59011

theorem train_times (t x : ℝ) : 
  (30 * t = 360) ∧ (36 * (t - x) = 360) → x = 2 :=
by
  sorry

end train_times_l59_59011


namespace carter_baseball_cards_l59_59374

theorem carter_baseball_cards (m c : ℕ) (h1 : m = 210) (h2 : m = c + 58) : c = 152 := 
by
  sorry

end carter_baseball_cards_l59_59374


namespace quadratic_polynomial_fourth_power_l59_59728

theorem quadratic_polynomial_fourth_power {a b c : ℤ} (h : ∀ x : ℤ, ∃ k : ℤ, ax^2 + bx + c = k^4) : a = 0 ∧ b = 0 :=
sorry

end quadratic_polynomial_fourth_power_l59_59728


namespace tangents_parallel_l59_59149

-- Definitions based on the conditions in part A
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_line (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

def secant_intersection (c1 c2 : Circle) (A : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  sorry

-- Main theorem statement
theorem tangents_parallel 
  (c1 c2 : Circle) (A B C : ℝ × ℝ) 
  (h1 : c1.center ≠ c2.center) 
  (h2 : dist c1.center c2.center = c1.radius + c2.radius) 
  (h3 : (B, C) = secant_intersection c1 c2 A) 
  (h4 : tangent_line c1 B ≠ tangent_line c2 C) :
  tangent_line c1 B = tangent_line c2 C :=
sorry

end tangents_parallel_l59_59149


namespace at_least_one_solves_l59_59082

/--
Given probabilities p1, p2, p3 that individuals A, B, and C solve a problem respectively,
prove that the probability that at least one of them solves the problem is 
1 - (1 - p1) * (1 - p2) * (1 - p3).
-/
theorem at_least_one_solves (p1 p2 p3 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 1 - (1 - p1) * (1 - p2) * (1 - p3) :=
by
  sorry

end at_least_one_solves_l59_59082


namespace tan_C_in_triangle_l59_59102

theorem tan_C_in_triangle
  (A B C : ℝ)
  (cos_A : Real.cos A = 4/5)
  (tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 := 
sorry

end tan_C_in_triangle_l59_59102


namespace perfect_squares_l59_59403

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l59_59403


namespace arith_seq_a1_a7_sum_l59_59514

variable (a : ℕ → ℝ) (d : ℝ)

-- Conditions
def arithmetic_sequence : Prop :=
  ∀ n, a (n + 1) = a n + d

def condition_sum : Prop :=
  a 3 + a 4 + a 5 = 12

-- Equivalent proof problem statement
theorem arith_seq_a1_a7_sum :
  arithmetic_sequence a d →
  condition_sum a →
  a 1 + a 7 = 8 :=
by
  sorry

end arith_seq_a1_a7_sum_l59_59514


namespace valid_bases_for_625_l59_59483

theorem valid_bases_for_625 (b : ℕ) : (b^3 ≤ 625 ∧ 625 < b^4) → ((625 % b) % 2 = 1) ↔ (b = 6 ∨ b = 7 ∨ b = 8) :=
by
  sorry

end valid_bases_for_625_l59_59483


namespace find_fraction_l59_59042

variable (x : ℝ) (f : ℝ)
axiom thirty_percent_of_x : 0.30 * x = 63.0000000000001
axiom fraction_condition : f = 0.40 * x + 12

theorem find_fraction : f = 96 := by
  sorry

end find_fraction_l59_59042


namespace pasture_rent_share_l59_59896

theorem pasture_rent_share (x : ℕ) (H1 : (45 / (10 * x + 60 + 45)) * 245 = 63) : 
  x = 7 :=
by {
  sorry
}

end pasture_rent_share_l59_59896


namespace sum_every_third_odd_integer_l59_59428

theorem sum_every_third_odd_integer (a₁ d n : ℕ) (S : ℕ) 
  (h₁ : a₁ = 201) 
  (h₂ : d = 6) 
  (h₃ : n = 50) 
  (h₄ : S = (n * (2 * a₁ + (n - 1) * d)) / 2) 
  (h₅ : a₁ + (n - 1) * d = 495) 
  : S = 17400 := 
  by sorry

end sum_every_third_odd_integer_l59_59428


namespace gcd_12345_6789_l59_59296

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 :=
by
  sorry

end gcd_12345_6789_l59_59296


namespace ninety_percent_greater_than_eighty_percent_l59_59238

-- Define the constants involved in the problem
def ninety_percent (n : ℕ) : ℝ := 0.90 * n
def eighty_percent (n : ℕ) : ℝ := 0.80 * n

-- Define the problem statement
theorem ninety_percent_greater_than_eighty_percent :
  ninety_percent 40 - eighty_percent 30 = 12 :=
by
  sorry

end ninety_percent_greater_than_eighty_percent_l59_59238


namespace lim_f_iterate_l59_59254

open MeasureTheory

noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (2 * Real.pi * x)) / 2

noncomputable def f_iterate (n : ℕ) : ℝ → ℝ :=
match n with
| 0 => id
| n + 1 => f ∘ f_iterate n

theorem lim_f_iterate (x : ℝ) (hx : x ∈ set.univ) :
  ∀ᶠ (x : ℝ) in (volume : measure ℝ).ae, tendsto (λ n, f_iterate n x) at_top (𝓝 1) :=
sorry

end lim_f_iterate_l59_59254


namespace tangent_line_at_P_l59_59000

noncomputable def tangent_line (x : ℝ) (y : ℝ) := (8 * x - y - 12 = 0)

def curve (x : ℝ) := x^3 - x^2

def derivative (f : ℝ → ℝ) (x : ℝ) := 3 * x^2 - 2 * x

theorem tangent_line_at_P :
    tangent_line 2 4 :=
by
  sorry

end tangent_line_at_P_l59_59000


namespace shortest_distance_to_circle_l59_59579

variable (A O T : Type)
variable (r d : ℝ)
variable [MetricSpace A]
variable [MetricSpace O]
variable [MetricSpace T]

open Real

theorem shortest_distance_to_circle (h : d = (4 / 3) * r) : 
  OA = (5 / 3) * r → shortest_dist = (2 / 3) * r :=
by
  sorry

end shortest_distance_to_circle_l59_59579


namespace range_of_a_l59_59838

def proposition_p (a : ℝ) : Prop :=
  (a + 6) * (a - 7) < 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4 * x + a < 0

def neg_q (a : ℝ) : Prop :=
  a ≥ 4

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ neg_q a) ↔ a ∈ Set.Ioo (-6 : ℝ) (7 : ℝ) ∪ Set.Ici (4 : ℝ) :=
sorry

end range_of_a_l59_59838


namespace number_of_girls_l59_59355

theorem number_of_girls (total_students boys girls : ℕ)
  (h1 : boys = 300)
  (h2 : (girls : ℝ) = 0.6 * total_students)
  (h3 : (boys : ℝ) = 0.4 * total_students) : 
  girls = 450 := by
  sorry

end number_of_girls_l59_59355


namespace triangle_largest_angle_l59_59995

theorem triangle_largest_angle (A B C : ℚ) (sinA sinB sinC : ℚ) 
(h_ratio : sinA / sinB = 3 / 5)
(h_ratio2 : sinB / sinC = 5 / 7)
(h_sum : A + B + C = 180) : C = 120 := 
sorry

end triangle_largest_angle_l59_59995


namespace six_digit_divisibility_by_37_l59_59750

theorem six_digit_divisibility_by_37 (a b c d e f : ℕ) (H : (100 * a + 10 * b + c + 100 * d + 10 * e + f) % 37 = 0) : 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 37 = 0 := 
sorry

end six_digit_divisibility_by_37_l59_59750


namespace cube_mod_35_divisors_l59_59788

theorem cube_mod_35_divisors (a : ℤ) : (35 ∣ a^3 - 1) ↔
  (∃ k : ℤ, a = 35 * k + 1) ∨ 
  (∃ k : ℤ, a = 35 * k + 11) ∨ 
  (∃ k : ℤ, a = 35 * k + 16) :=
by sorry

end cube_mod_35_divisors_l59_59788


namespace kayla_waiting_years_l59_59981

def minimum_driving_age : ℕ := 18
def kimiko_age : ℕ := 26
def kayla_age : ℕ := kimiko_age / 2
def years_until_kayla_can_drive : ℕ := minimum_driving_age - kayla_age

theorem kayla_waiting_years : years_until_kayla_can_drive = 5 :=
by
  sorry

end kayla_waiting_years_l59_59981


namespace point_on_curve_l59_59061

theorem point_on_curve :
  let x := -3 / 4
  let y := 1 / 2
  x^2 = (y^2 - 1) ^ 2 :=
by
  sorry

end point_on_curve_l59_59061


namespace largest_n_factorable_l59_59472

theorem largest_n_factorable :
  ∃ n, (∀ A B : ℤ, 6x^2 + n • x + 72 = (6 • x + A) * (x + B)) ∧ 
    (∀ x', 6x' + A = 0 ∨ x' + B = 0) ∧ 
    n = 433 := 
sorry

end largest_n_factorable_l59_59472


namespace part1_part2_l59_59974

noncomputable def f : ℝ → ℝ 
| x => if 0 ≤ x then 2^x - 1 else -2^(-x) + 1

theorem part1 (x : ℝ) (h : x < 0) : f x = -2^(-x) + 1 := sorry

theorem part2 (a : ℝ) : f a ≤ 3 ↔ a ≤ 2 := sorry

end part1_part2_l59_59974


namespace average_price_of_racket_l59_59922

theorem average_price_of_racket
  (total_amount_made : ℝ)
  (number_of_pairs_sold : ℕ)
  (h1 : total_amount_made = 490) 
  (h2 : number_of_pairs_sold = 50) : 
  (total_amount_made / number_of_pairs_sold : ℝ) = 9.80 := 
  by
  sorry

end average_price_of_racket_l59_59922


namespace find_starting_number_l59_59007

theorem find_starting_number (num_even_ints: ℕ) (end_num: ℕ) (h_num: num_even_ints = 35) (h_end: end_num = 95) : 
  ∃ start_num: ℕ, start_num = 24 ∧ (∀ n: ℕ, (start_num + 2 * n ≤ end_num ∧ n < num_even_ints)) := by
  sorry

end find_starting_number_l59_59007


namespace friend_selling_price_l59_59609

-- Define the conditions
def CP : ℝ := 51136.36
def loss_percent : ℝ := 0.12
def gain_percent : ℝ := 0.20

-- Define the selling prices SP1 and SP2
def SP1 := CP * (1 - loss_percent)
def SP2 := SP1 * (1 + gain_percent)

-- State the theorem
theorem friend_selling_price : SP2 = 54000 := 
by sorry

end friend_selling_price_l59_59609


namespace length_of_CD_l59_59269

theorem length_of_CD (x y u v : ℝ) (R S C D : ℝ → ℝ)
  (h1 : 5 * x = 3 * y)
  (h2 : 7 * u = 4 * v)
  (h3 : u = x + 3)
  (h4 : v = y - 3)
  (h5 : C x + D y = 1) : 
  x + y = 264 :=
by
  sorry

end length_of_CD_l59_59269


namespace probability_neither_perfect_square_cube_fifth_l59_59143

theorem probability_neither_perfect_square_cube_fifth (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) :
  (∑ i in (range (200 + 1)), (if ¬(is_square i ∨ is_cube i ∨ is_power5 i) then 1 else 0)) / 200 = 91 / 100 :=
sorry

end probability_neither_perfect_square_cube_fifth_l59_59143


namespace minimum_value_of_function_l59_59490

theorem minimum_value_of_function (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  ∃ y : ℝ, (∀ z : ℝ, z = (1 / x) + (4 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end minimum_value_of_function_l59_59490


namespace ratio_of_walkway_to_fountain_l59_59510

theorem ratio_of_walkway_to_fountain (n s d : ℝ) (h₀ : n = 10) (h₁ : n^2 * s^2 = 0.40 * (n*s + 2*n*d)^2) : 
  d / s = 1 / 3.44 := 
sorry

end ratio_of_walkway_to_fountain_l59_59510


namespace combined_sleep_hours_l59_59058

def connor_hours : ℕ := 6
def luke_hours : ℕ := connor_hours + 2
def emma_hours : ℕ := connor_hours - 1
def puppy_hours : ℕ := 2 * luke_hours

theorem combined_sleep_hours :
  connor_hours + luke_hours + emma_hours + puppy_hours = 35 := by
  sorry

end combined_sleep_hours_l59_59058


namespace range_of_a_l59_59005

noncomputable def quadratic_inequality_solution_set (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x - 4 < 0

theorem range_of_a :
  {a : ℝ | quadratic_inequality_solution_set a} = {a | -16 < a ∧ a ≤ 0} := 
sorry

end range_of_a_l59_59005


namespace average_value_l59_59180

variable (z : ℝ)

theorem average_value : (0 + 2 * z^2 + 4 * z^2 + 8 * z^2 + 16 * z^2) / 5 = 6 * z^2 :=
by
  sorry

end average_value_l59_59180


namespace option_D_is_div_by_9_l59_59431

-- Define the parameters and expressions
def A (k : ℕ) : ℤ := 6 + 6 * 7^k
def B (k : ℕ) : ℤ := 2 + 7^(k - 1)
def C (k : ℕ) : ℤ := 2 * (2 + 7^(k + 1))
def D (k : ℕ) : ℤ := 3 * (2 + 7^k)

-- Define the main theorem to prove that D is divisible by 9
theorem option_D_is_div_by_9 (k : ℕ) (hk : k > 0) : D k % 9 = 0 :=
sorry

end option_D_is_div_by_9_l59_59431


namespace average_of_remaining_numbers_l59_59391

theorem average_of_remaining_numbers 
  (S S' : ℝ)
  (h1 : S / 12 = 90)
  (h2 : S' = S - 80 - 82) :
  S' / 10 = 91.8 :=
sorry

end average_of_remaining_numbers_l59_59391


namespace find_s_l59_59975

section
variables {a b c p q s : ℕ}

-- Conditions given in the problem
variables (h1 : a + b = p)
variables (h2 : p + c = s)
variables (h3 : s + a = q)
variables (h4 : b + c + q = 18)
variables (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
variables (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0)

-- Statement of the problem
theorem find_s (h1 : a + b = p) (h2 : p + c = s) (h3 : s + a = q) (h4 : b + c + q = 18)
  (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
  (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0) :
  s = 9 :=
sorry
end

end find_s_l59_59975


namespace sequence_formula_l59_59340

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4 ^ n - 1 :=
by
  sorry

end sequence_formula_l59_59340


namespace calculate_expression_l59_59450

theorem calculate_expression : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end calculate_expression_l59_59450


namespace problem_l59_59815

theorem problem (a b : ℚ) (h : a / b = 6 / 5) : (5 * a + 4 * b) / (5 * a - 4 * b) = 5 := 
by 
  sorry

end problem_l59_59815


namespace least_subtract_divisible_by_8_l59_59473

def least_subtracted_to_divisible_by (n : ℕ) (d : ℕ) : ℕ :=
  n % d

theorem least_subtract_divisible_by_8 (n : ℕ) (d : ℕ) (h : n = 964807) (h_d : d = 8) :
  least_subtracted_to_divisible_by n d = 7 :=
by
  sorry

end least_subtract_divisible_by_8_l59_59473


namespace arithmetic_sequence_sum_is_18_l59_59971

variable (a : ℕ → ℕ)

theorem arithmetic_sequence_sum_is_18
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 18 := 
sorry

end arithmetic_sequence_sum_is_18_l59_59971


namespace friends_bought_color_box_l59_59057

variable (total_pencils : ℕ) (pencils_per_box : ℕ) (chloe_pencils : ℕ)

theorem friends_bought_color_box : 
  (total_pencils = 42) → 
  (pencils_per_box = 7) → 
  (chloe_pencils = pencils_per_box) → 
  (total_pencils - chloe_pencils) / pencils_per_box = 5 := 
by 
  intros ht hb hc
  sorry

end friends_bought_color_box_l59_59057


namespace triangle_strike_interval_l59_59893

/-- Jacob strikes the cymbals every 7 beats and the triangle every t beats.
    Given both are struck at the same time every 14 beats, this proves t = 2. -/
theorem triangle_strike_interval :
  ∃ t : ℕ, t ≠ 7 ∧ (∀ n : ℕ, (7 * n % t = 0) → ∃ k : ℕ, 7 * n = 14 * k) ∧ t = 2 :=
by
  use 2
  sorry

end triangle_strike_interval_l59_59893


namespace solve_system_infinite_solutions_l59_59242

theorem solve_system_infinite_solutions (m : ℝ) (h1 : ∀ x y : ℝ, x + m * y = 2) (h2 : ∀ x y : ℝ, m * x + 16 * y = 8) :
  m = 4 :=
sorry

end solve_system_infinite_solutions_l59_59242


namespace range_of_m_l59_59801

noncomputable def prop_p (m : ℝ) : Prop :=
0 < m ∧ m < 1 / 3

noncomputable def prop_q (m : ℝ) : Prop :=
0 < m ∧ m < 15

theorem range_of_m (m : ℝ) : (prop_p m ∧ ¬ prop_q m) ∨ (¬ prop_p m ∧ prop_q m) ↔ 1 / 3 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l59_59801


namespace convert_base_3_to_base_10_l59_59948

theorem convert_base_3_to_base_10 : 
  (1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) = 142 :=
by
  sorry

end convert_base_3_to_base_10_l59_59948


namespace product_of_decimals_l59_59717

theorem product_of_decimals :
  (8 : ℚ) * (1 / 4 : ℚ) * (2 : ℚ) * (1 / 8 : ℚ) = 1 / 2 := by
  sorry

end product_of_decimals_l59_59717


namespace pqrs_sum_l59_59529

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l59_59529


namespace find_a8_l59_59261

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q^(n-1)

noncomputable def sum_geom (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

theorem find_a8 (a_1 q a_2 a_5 a_8 : ℝ) (S : ℕ → ℝ) 
  (Hsum : ∀ n, S n = sum_geom a_1 q n)
  (H1 : 2 * S 9 = S 3 + S 6)
  (H2 : a_2 = geometric_sequence a_1 q 2)
  (H3 : a_5 = geometric_sequence a_1 q 5)
  (H4 : a_2 + a_5 = 4)
  (H5 : a_8 = geometric_sequence a_1 q 8) :
  a_8 = 2 :=
sorry

end find_a8_l59_59261


namespace ralphStartsWith_l59_59561

def ralphEndsWith : ℕ := 15
def ralphLoses : ℕ := 59

theorem ralphStartsWith : (ralphEndsWith + ralphLoses = 74) :=
by
  sorry

end ralphStartsWith_l59_59561


namespace find_nonzero_q_for_quadratic_l59_59195

theorem find_nonzero_q_for_quadratic :
  ∃ (q : ℝ), q ≠ 0 ∧ (∀ (x1 x2 : ℝ), (q * x1^2 - 8 * x1 + 2 = 0 ∧ q * x2^2 - 8 * x2 + 2 = 0) → x1 = x2) ↔ q = 8 :=
by
  sorry

end find_nonzero_q_for_quadratic_l59_59195


namespace outfit_combinations_l59_59747

theorem outfit_combinations 
  (shirts : Fin 5)
  (pants : Fin 6)
  (restricted_shirt : Fin 1)
  (restricted_pants : Fin 2) :
  ∃ total_combinations : ℕ, total_combinations = 28 :=
sorry

end outfit_combinations_l59_59747


namespace sum_of_digits_divisible_by_9_l59_59208

theorem sum_of_digits_divisible_by_9 (N : ℕ) (a b c : ℕ) (hN : N < 10^1962)
  (h1 : N % 9 = 0)
  (ha : a = (N.digits 10).sum)
  (hb : b = (a.digits 10).sum)
  (hc : c = (b.digits 10).sum) :
  c = 9 :=
sorry

end sum_of_digits_divisible_by_9_l59_59208


namespace sum_of_distinct_real_numbers_l59_59546

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l59_59546


namespace calculate_g_at_5_l59_59097

variable {R : Type} [LinearOrderedField R] (g : R → R)
variable (x : R)

theorem calculate_g_at_5 (h : ∀ x : R, g (3 * x - 4) = 5 * x - 7) : g 5 = 8 :=
by
  sorry

end calculate_g_at_5_l59_59097


namespace sum_distinct_vars_eq_1716_l59_59541

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l59_59541


namespace exists_k_consecutive_squareful_numbers_l59_59318

-- Define what it means for a number to be squareful
def is_squareful (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧ m * m ∣ n

-- State the theorem
theorem exists_k_consecutive_squareful_numbers (k : ℕ) : 
  ∃ (a : ℕ), ∀ i, i < k → is_squareful (a + i) :=
sorry

end exists_k_consecutive_squareful_numbers_l59_59318


namespace diane_bakes_gingerbreads_l59_59952

open Nat

theorem diane_bakes_gingerbreads :
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  total_gingerbreads1 + total_gingerbreads2 = 160 := 
by
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  exact Eq.refl (total_gingerbreads1 + total_gingerbreads2)

end diane_bakes_gingerbreads_l59_59952


namespace total_income_per_minute_l59_59575

theorem total_income_per_minute :
  let black_shirt_price := 30
  let black_shirt_quantity := 250
  let white_shirt_price := 25
  let white_shirt_quantity := 200
  let red_shirt_price := 28
  let red_shirt_quantity := 100
  let blue_shirt_price := 25
  let blue_shirt_quantity := 50

  let black_discount := 0.05
  let white_discount := 0.08
  let red_discount := 0.10

  let total_black_income_before_discount := black_shirt_quantity * black_shirt_price
  let total_white_income_before_discount := white_shirt_quantity * white_shirt_price
  let total_red_income_before_discount := red_shirt_quantity * red_shirt_price
  let total_blue_income_before_discount := blue_shirt_quantity * blue_shirt_price

  let total_income_before_discount :=
    total_black_income_before_discount + total_white_income_before_discount + total_red_income_before_discount + total_blue_income_before_discount

  let total_black_discount := black_discount * total_black_income_before_discount
  let total_white_discount := white_discount * total_white_income_before_discount
  let total_red_discount := red_discount * total_red_income_before_discount

  let total_discount :=
    total_black_discount + total_white_discount + total_red_discount

  let total_income_after_discount :=
    total_income_before_discount - total_discount

  let total_minutes := 40
  let total_income_per_minute := total_income_after_discount / total_minutes

  total_income_per_minute = 387.38 := by
  sorry

end total_income_per_minute_l59_59575


namespace binomial_identity_l59_59212

theorem binomial_identity :
  (Nat.choose 16 6 = 8008) → (Nat.choose 16 7 = 11440) → (Nat.choose 16 8 = 12870) →
  Nat.choose 18 8 = 43758 :=
by
  intros h1 h2 h3
  sorry

end binomial_identity_l59_59212


namespace fixed_point_l59_59120

-- Let ABC be a right triangle with ∠B = 90°
variables (A B C D E F P : Point)
variable (h1 : right_triangle A B C ∧ angle ABC = 90)

-- D lies on line CB such that B is between D and C
variable (h2 : on_line D CB ∧ between B D C)

-- E is the midpoint of AD
variable (h3 : midpoint E A D)

-- F is the second intersection point of the circumcircle of ΔACD and ΔBDE
variable (h4 : second_intersection F (circumcircle A C D) (circumcircle B D E))

-- Prove that as D varies, the line EF passes through a fixed point P
theorem fixed_point (D : Point) (hD : condition_to_D D) :
  ∃ P : Point,  ∀ D : Point, condition_to_D D → passes_through (E F) P :=
sorry

end fixed_point_l59_59120


namespace find_divisor_l59_59044

variable {N : ℤ} (k q : ℤ) {D : ℤ}

theorem find_divisor (h1 : N = 158 * k + 50) (h2 : N = D * q + 13) (h3 : D > 13) (h4 : D < 158) :
  D = 37 :=
by 
  sorry

end find_divisor_l59_59044


namespace rabbit_excursion_time_l59_59287

theorem rabbit_excursion_time 
  (line_length : ℝ := 40) 
  (line_speed : ℝ := 3) 
  (rabbit_speed : ℝ := 5) : 
  -- The time calculated for the rabbit to return is 25 seconds
  (line_length / (rabbit_speed - line_speed) + line_length / (rabbit_speed + line_speed)) = 25 :=
by
  -- Placeholder for the proof, to be filled in with a detailed proof later
  sorry

end rabbit_excursion_time_l59_59287


namespace find_C_l59_59516

theorem find_C 
  (m n : ℝ)
  (C : ℝ)
  (h1 : m = 6 * n + C)
  (h2 : m + 2 = 6 * (n + 0.3333333333333333) + C) 
  : C = 0 := by
  sorry

end find_C_l59_59516


namespace octagon_diagonals_l59_59660

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l59_59660


namespace tim_income_less_than_juan_l59_59840

-- Definitions of the conditions
variables {T J M : ℝ}
def mart_income_condition1 (M T : ℝ) : Prop := M = 1.40 * T
def mart_income_condition2 (M J : ℝ) : Prop := M = 0.84 * J

-- The proof goal
theorem tim_income_less_than_juan (T J M : ℝ) 
(h1: mart_income_condition1 M T) 
(h2: mart_income_condition2 M J) : 
T = 0.60 * J :=
by
  sorry

end tim_income_less_than_juan_l59_59840


namespace line_intersects_circle_at_two_points_l59_59632

-- Definitions based on given conditions
def radius (r : ℝ) : Prop := r = 6.5
def distance_from_center_to_line (d : ℝ) : Prop := d = 4.5

-- Theorem statement
theorem line_intersects_circle_at_two_points (r d : ℝ) (hr : radius r) (hd : distance_from_center_to_line d) : 
  d < r → ∃(p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ (∥p1∥ = r ∧ ∥p2∥ = r) := 
by 
  sorry

end line_intersects_circle_at_two_points_l59_59632


namespace circle_tangent_to_ellipse_l59_59737

theorem circle_tangent_to_ellipse {r : ℝ} 
  (h1: ∀ p: ℝ × ℝ, p ≠ (0, 0) → ((p.1 - r)^2 + p.2^2 = r^2 → p.1^2 + 4 * p.2^2 = 8))
  (h2: ∃ p: ℝ × ℝ, p ≠ (0, 0) ∧ ((p.1 - r)^2 + p.2^2 = r^2 ∧ p.1^2 + 4 * p.2^2 = 8)):
  r = Real.sqrt (3 / 2) :=
by
  sorry

end circle_tangent_to_ellipse_l59_59737


namespace person_age_l59_59765

-- Define the conditions
def current_age : ℕ := 18

-- Define the equation based on the person's statement
def age_equation (A Y : ℕ) : Prop := 3 * (A + 3) - 3 * (A - Y) = A

-- Statement to be proven
theorem person_age (Y : ℕ) : 
  age_equation current_age Y → Y = 3 := 
by 
  sorry

end person_age_l59_59765


namespace triangle_shape_l59_59103

open Real

noncomputable def triangle (a b c A B C S : ℝ) :=
  ∃ (a b c A B C S : ℝ),
    a = 2 * sqrt 3 ∧
    A = π / 3 ∧
    S = 2 * sqrt 3 ∧
    (S = (1 / 2) * b * c * sin A) ∧
    (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A) ∧
    (b = 2 ∧ c = 4 ∨ b = 4 ∧ c = 2)

theorem triangle_shape (A B C : ℝ) (h : sin (C - B) = sin (2 * B) - sin A):
    (B = π / 2 ∨ C = B) :=
sorry

end triangle_shape_l59_59103


namespace find_m_l59_59792

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def f' (x : ℝ) : ℝ := -1 / (x^2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * x

theorem find_m (m : ℝ) :
  g 2 m = 1 / (f' 2) →
  m = -2 :=
by
  sorry

end find_m_l59_59792


namespace units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l59_59071

theorem units_digit_2_pow_2010_5_pow_1004_14_pow_1002 :
  (2^2010 * 5^1004 * 14^1002) % 10 = 0 := by
sorry

end units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l59_59071


namespace find_a_range_l59_59494

noncomputable def f (x : ℝ) := (x - 1) / Real.exp x

noncomputable def condition_holds (a : ℝ) : Prop :=
∀ t ∈ (Set.Icc (1/2 : ℝ) 2), f t > t

theorem find_a_range (a : ℝ) (h : condition_holds a) : a > Real.exp 2 + 1/2 := sorry

end find_a_range_l59_59494


namespace min_value_fraction_l59_59836

theorem min_value_fraction (a b : ℝ) (n : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_sum : a + b = 2) : 
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end min_value_fraction_l59_59836
