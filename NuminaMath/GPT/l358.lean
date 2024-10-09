import Mathlib

namespace base_eight_to_ten_l358_35805

theorem base_eight_to_ten (n : Nat) (h : n = 52) : 8 * 5 + 2 = 42 :=
by
  -- Proof will be written here.
  sorry

end base_eight_to_ten_l358_35805


namespace range_of_a4_l358_35819

noncomputable def geometric_sequence (a1 a2 a3 : ℝ) (q : ℝ) (a4 : ℝ) : Prop :=
  ∃ (a1 q : ℝ), 0 < a1 ∧ a1 < 1 ∧ 
                1 < a1 * q ∧ a1 * q < 2 ∧ 
                2 < a1 * q^2 ∧ a1 * q^2 < 4 ∧ 
                a4 = (a1 * q^2) * q ∧ 
                2 * Real.sqrt 2 < a4 ∧ a4 < 16

theorem range_of_a4 (a1 a2 a3 a4 : ℝ) (q : ℝ) (h1 : 0 < a1) (h2 : a1 < 1) 
  (h3 : 1 < a2) (h4 : a2 < 2) (h5 : a2 = a1 * q)
  (h6 : 2 < a3) (h7 : a3 < 4) (h8 : a3 = a1 * q^2) :
  2 * Real.sqrt 2 < a4 ∧ a4 < 16 :=
by
  have hq1 : 2 * q^2 < 1 := sorry    -- Placeholder for necessary inequalities
  have hq2: 1 < q ∧ q < 4 := sorry   -- Placeholder for necessary inequalities
  sorry

end range_of_a4_l358_35819


namespace quad_sin_theorem_l358_35803

-- Define the necessary entities in Lean
structure Quadrilateral (A B C D : Type) :=
(angleB : ℝ)
(angleD : ℝ)
(angleA : ℝ)

-- Define the main theorem
theorem quad_sin_theorem {A B C D : Type} (quad : Quadrilateral A B C D) (AC AD : ℝ) (α : ℝ) :
  quad.angleB = 90 ∧ quad.angleD = 90 ∧ quad.angleA = α → AD = AC * Real.sin α := 
sorry

end quad_sin_theorem_l358_35803


namespace simplify_expression_l358_35846

theorem simplify_expression (a : Int) : 2 * a - a = a :=
by
  sorry

end simplify_expression_l358_35846


namespace molecular_weight_of_3_moles_CaOH2_is_correct_l358_35881

-- Define the atomic weights as given by the conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Define the molecular formula contributions for Ca(OH)2
def molecular_weight_CaOH2 : ℝ :=
  atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H

-- Define the weight of 3 moles of Ca(OH)2 based on the molecular weight
def weight_of_3_moles_CaOH2 : ℝ :=
  3 * molecular_weight_CaOH2

-- Theorem to prove the final result
theorem molecular_weight_of_3_moles_CaOH2_is_correct :
  weight_of_3_moles_CaOH2 = 222.30 := by
  sorry

end molecular_weight_of_3_moles_CaOH2_is_correct_l358_35881


namespace players_taking_all_three_subjects_l358_35859

-- Define the variables for the number of players in each category
def num_players : ℕ := 18
def num_physics : ℕ := 10
def num_biology : ℕ := 7
def num_chemistry : ℕ := 5
def num_physics_biology : ℕ := 3
def num_biology_chemistry : ℕ := 2
def num_physics_chemistry : ℕ := 1

-- Define the proposition we want to prove
theorem players_taking_all_three_subjects :
  ∃ x : ℕ, x = 2 ∧
  num_players = num_physics + num_biology + num_chemistry
                - num_physics_chemistry
                - num_physics_biology
                - num_biology_chemistry
                + x :=
by {
  sorry -- Placeholder for the proof
}

end players_taking_all_three_subjects_l358_35859


namespace invertible_from_c_l358_35887

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the condition for c and the statement to prove
theorem invertible_from_c (c : ℝ) (h : ∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) : c = 3 :=
sorry

end invertible_from_c_l358_35887


namespace non_neg_integer_solutions_l358_35834

theorem non_neg_integer_solutions (a b c : ℕ) :
  (∀ x : ℕ, x^2 - 2 * a * x + b = 0 → x ≥ 0) ∧ 
  (∀ y : ℕ, y^2 - 2 * b * y + c = 0 → y ≥ 0) ∧ 
  (∀ z : ℕ, z^2 - 2 * c * z + a = 0 → z ≥ 0) → 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 0) :=
sorry

end non_neg_integer_solutions_l358_35834


namespace man_walk_time_l358_35832

theorem man_walk_time (speed_kmh : ℕ) (distance_km : ℕ) (time_min : ℕ) 
  (h1 : speed_kmh = 10) (h2 : distance_km = 7) : time_min = 42 :=
by
  sorry

end man_walk_time_l358_35832


namespace gcd_ab_is_22_l358_35811

def a : ℕ := 198
def b : ℕ := 308

theorem gcd_ab_is_22 : Nat.gcd a b = 22 := 
by { sorry }

end gcd_ab_is_22_l358_35811


namespace bus_full_people_could_not_take_l358_35885

-- Definitions of the given conditions
def bus_capacity : ℕ := 80
def first_pickup_people : ℕ := (3 / 5) * bus_capacity
def people_exit_at_second_pickup : ℕ := 25
def people_waiting_at_second_pickup : ℕ := 90

-- The Lean statement to prove the number of people who could not take the bus
theorem bus_full_people_could_not_take (h1 : bus_capacity = 80)
                                       (h2 : first_pickup_people = 48)
                                       (h3 : people_exit_at_second_pickup = 25)
                                       (h4 : people_waiting_at_second_pickup = 90) :
  90 - (80 - (48 - 25)) = 33 :=
by
  sorry

end bus_full_people_could_not_take_l358_35885


namespace find_length_of_AB_l358_35873

variable (A B C : ℝ)
variable (cos_C_div2 BC AC AB : ℝ)
variable (C_gt_0 : 0 < C / 2) (C_lt_pi : C / 2 < Real.pi)

axiom h1 : cos_C_div2 = Real.sqrt 5 / 5
axiom h2 : BC = 1
axiom h3 : AC = 5
axiom h4 : AB = Real.sqrt (BC ^ 2 + AC ^ 2 - 2 * BC * AC * (2 * cos_C_div2 ^ 2 - 1))

theorem find_length_of_AB : AB = 4 * Real.sqrt 2 :=
by
  sorry

end find_length_of_AB_l358_35873


namespace selling_price_correct_l358_35855

noncomputable def cost_price : ℝ := 90.91

noncomputable def profit_rate : ℝ := 0.10

noncomputable def profit : ℝ := profit_rate * cost_price

noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 100.00 := by
  sorry

end selling_price_correct_l358_35855


namespace intersection_proof_l358_35852

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def N : Set ℕ := { x | Real.sqrt (2^x - 1) < 5 }
def expected_intersection : Set ℕ := {1, 2, 3, 4}

theorem intersection_proof : M ∩ N = expected_intersection := by
  sorry

end intersection_proof_l358_35852


namespace ratio_pen_to_pencil_l358_35839

-- Define the costs
def cost_of_pencil (P : ℝ) : ℝ := P
def cost_of_pen (P : ℝ) : ℝ := 4 * P
def total_cost (P : ℝ) : ℝ := cost_of_pencil P + cost_of_pen P

-- The proof that the total cost of the pen and pencil is $6 given the provided ratio
theorem ratio_pen_to_pencil (P : ℝ) (h_total_cost : total_cost P = 6) (h_pen_cost : cost_of_pen P = 4) :
  cost_of_pen P / cost_of_pencil P = 4 :=
by
  -- Proof skipped
  sorry

end ratio_pen_to_pencil_l358_35839


namespace accurate_to_ten_thousandth_l358_35818

/-- Define the original number --/
def original_number : ℕ := 580000

/-- Define the accuracy of the number represented by 5.8 * 10^5 --/
def is_accurate_to_ten_thousandth_place (n : ℕ) : Prop :=
  n = 5 * 100000 + 8 * 10000

/-- The statement to be proven --/
theorem accurate_to_ten_thousandth : is_accurate_to_ten_thousandth_place original_number :=
by
  sorry

end accurate_to_ten_thousandth_l358_35818


namespace find_positive_real_number_solution_l358_35864

theorem find_positive_real_number_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) (hx : x > 0) : x = 15 :=
sorry

end find_positive_real_number_solution_l358_35864


namespace greatest_prime_factor_341_l358_35860

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l358_35860


namespace complex_numbers_count_l358_35883

theorem complex_numbers_count (z : ℂ) (h1 : z^24 = 1) (h2 : ∃ r : ℝ, z^6 = r) : ℕ :=
  sorry -- Proof goes here

end complex_numbers_count_l358_35883


namespace smallest_n_for_convex_100gon_l358_35893

def isConvexPolygon (P : List (Real × Real)) : Prop := sorry -- Assumption for polygon convexity
def canBeIntersectedByTriangles (P : List (Real × Real)) (n : ℕ) : Prop := sorry -- Assumption for intersection by n triangles

theorem smallest_n_for_convex_100gon :
  ∀ (P : List (Real × Real)),
  isConvexPolygon P →
  List.length P = 100 →
  (∀ n, canBeIntersectedByTriangles P n → n ≥ 50) ∧ canBeIntersectedByTriangles P 50 :=
sorry

end smallest_n_for_convex_100gon_l358_35893


namespace icing_two_sides_on_Jack_cake_l358_35801

noncomputable def Jack_cake_icing_two_sides (cake_size : ℕ) : ℕ :=
  let side_cubes := 4 * (cake_size - 2) * 3
  let vertical_edge_cubes := 4 * (cake_size - 2)
  side_cubes + vertical_edge_cubes

-- The statement to be proven
theorem icing_two_sides_on_Jack_cake : Jack_cake_icing_two_sides 5 = 96 :=
by
  sorry

end icing_two_sides_on_Jack_cake_l358_35801


namespace coverable_hook_l358_35824

def is_coverable (m n : ℕ) : Prop :=
  ∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5)

theorem coverable_hook (m n : ℕ) : (∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5))
  ↔ is_coverable m n :=
by
  sorry

end coverable_hook_l358_35824


namespace painted_by_all_three_l358_35896

/-
Statement: Given that 75% of the floor is painted red, 70% painted green, and 65% painted blue,
prove that at least 10% of the floor is painted with all three colors.
-/

def painted_by_red (floor : ℝ) : ℝ := 0.75 * floor
def painted_by_green (floor : ℝ) : ℝ := 0.70 * floor
def painted_by_blue (floor : ℝ) : ℝ := 0.65 * floor

theorem painted_by_all_three (floor : ℝ) :
  ∃ (x : ℝ), x = 0.10 * floor ∧
  (painted_by_red floor) + (painted_by_green floor) + (painted_by_blue floor) ≥ 2 * floor :=
sorry

end painted_by_all_three_l358_35896


namespace michael_scored_times_more_goals_l358_35850

theorem michael_scored_times_more_goals (x : ℕ) (hb : Bruce_goals = 4) (hm : Michael_goals = 4 * x) (ht : Bruce_goals + Michael_goals = 16) : x = 3 := by
  sorry

end michael_scored_times_more_goals_l358_35850


namespace intersection_hyperbola_l358_35879

theorem intersection_hyperbola (t : ℝ) :
  ∃ A B : ℝ, ∀ (x y : ℝ),
  (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 5 = 0) →
  (x^2 / A - y^2 / B = 1) :=
sorry

end intersection_hyperbola_l358_35879


namespace range_of_a_l358_35817

theorem range_of_a 
  (a : ℝ):
  (∀ x : ℝ, |x + 2| + |x - 1| > a^2 - 2 * a) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l358_35817


namespace evaluate_expression_l358_35833

theorem evaluate_expression (x : ℤ) (h1 : 0 ≤ x ∧ x ≤ 2) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x = 0) :
    ( ((4 - x) / (x - 1) - x) / ((x - 2) / (x - 1)) ) = -2 :=
by
    sorry

end evaluate_expression_l358_35833


namespace triangle_inequality_inequality_l358_35814

-- Define a helper function to describe the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

-- Define the main statement
theorem triangle_inequality_inequality (a b c : ℝ) (h_triangle : triangle_inequality a b c):
  a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

end triangle_inequality_inequality_l358_35814


namespace liza_butter_amount_l358_35849

theorem liza_butter_amount (B : ℕ) (h1 : B / 2 + B / 5 + (1 / 3) * ((B - B / 2 - B / 5) / 1) = B - 2) : B = 10 :=
sorry

end liza_butter_amount_l358_35849


namespace find_value_of_fraction_of_x_six_l358_35837

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := (Real.log x) / (Real.log b)

theorem find_value_of_fraction_of_x_six (x : ℝ) (h : log_base (10 * x) 10 + log_base (100 * x ^ 2) 10 = -1) : 
    1 / x ^ 6 = 31622.7766 :=
by
  sorry

end find_value_of_fraction_of_x_six_l358_35837


namespace kopecks_to_rubles_l358_35863

noncomputable def exchangeable_using_coins (total : ℕ) (num_coins : ℕ) : Prop :=
  ∃ (x y z t u v w : ℕ), 
    total = x * 1 + y * 2 + z * 5 + t * 10 + u * 20 + v * 50 + w * 100 ∧ 
    num_coins = x + y + z + t + u + v + w

theorem kopecks_to_rubles (A B : ℕ)
  (h : exchangeable_using_coins A B) : exchangeable_using_coins (100 * B) A :=
sorry

end kopecks_to_rubles_l358_35863


namespace multiple_of_weight_lifted_l358_35872

variable (F : ℝ) (M : ℝ)

theorem multiple_of_weight_lifted 
  (H1: ∀ (B : ℝ), B = 2 * F) 
  (H2: ∀ (B : ℝ), ∀ (W : ℝ), W = 3 * B) 
  (H3: ∃ (B : ℝ), (3 * B = 600)) 
  (H4: M * F = 150) : 
  M = 1.5 :=
by
  sorry

end multiple_of_weight_lifted_l358_35872


namespace quadratic_root_neg3_l358_35886

theorem quadratic_root_neg3 : ∃ x : ℝ, x^2 - 9 = 0 ∧ (x = -3) :=
by
  sorry

end quadratic_root_neg3_l358_35886


namespace chrysler_floors_difference_l358_35865

theorem chrysler_floors_difference (C L : ℕ) (h1 : C = 23) (h2 : C + L = 35) : C - L = 11 := by
  sorry

end chrysler_floors_difference_l358_35865


namespace dog_catches_sheep_in_20_seconds_l358_35854

variable (v_sheep v_dog : ℕ) (d : ℕ)

def relative_speed (v_dog v_sheep : ℕ) := v_dog - v_sheep

def time_to_catch (d v_sheep v_dog : ℕ) : ℕ := d / (relative_speed v_dog v_sheep)

theorem dog_catches_sheep_in_20_seconds
  (h1 : v_sheep = 16)
  (h2 : v_dog = 28)
  (h3 : d = 240) :
  time_to_catch d v_sheep v_dog = 20 := by {
  sorry
}

end dog_catches_sheep_in_20_seconds_l358_35854


namespace value_of_x_l358_35882

theorem value_of_x (x : ℝ) :
  (x^2 - 1 + (x - 1) * I = 0 ∨ x^2 - 1 = 0 ∧ x - 1 ≠ 0) → x = -1 :=
by
  sorry

end value_of_x_l358_35882


namespace find_y_from_eqns_l358_35816

theorem find_y_from_eqns (x y : ℝ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -5) : y = 34 :=
by {
  sorry
}

end find_y_from_eqns_l358_35816


namespace temperature_difference_l358_35838

def highest_temperature : ℝ := 8
def lowest_temperature : ℝ := -1

theorem temperature_difference : highest_temperature - lowest_temperature = 9 := by
  sorry

end temperature_difference_l358_35838


namespace min_value_quadratic_l358_35802

theorem min_value_quadratic :
  ∀ (x : ℝ), (2 * x^2 - 8 * x + 15) ≥ 7 :=
by
  -- We need to show that 2x^2 - 8x + 15 has a minimum value of 7
  sorry

end min_value_quadratic_l358_35802


namespace three_digit_numbers_l358_35868

theorem three_digit_numbers (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999) → 
  (n * n % 1000 = n % 1000) ↔ 
  (n = 625 ∨ n = 376) :=
by 
  sorry

end three_digit_numbers_l358_35868


namespace melissa_points_per_game_l358_35845

theorem melissa_points_per_game (total_points : ℕ) (games : ℕ) (h1 : total_points = 81) 
(h2 : games = 3) : total_points / games = 27 :=
by
  sorry

end melissa_points_per_game_l358_35845


namespace inequality_correct_l358_35831

noncomputable def a : ℝ := Real.exp (-0.5)
def b : ℝ := 0.5
noncomputable def c : ℝ := Real.log 1.5

theorem inequality_correct : a > b ∧ b > c :=
by
  sorry

end inequality_correct_l358_35831


namespace arithmetic_sequence_properties_l358_35895

noncomputable def arithmetic_sequence (n : ℕ) : ℕ :=
  4 * n - 3

noncomputable def sum_of_first_n_terms (n : ℕ) : ℕ :=
  2 * n^2 - n

noncomputable def sum_of_reciprocal_sequence (n : ℕ) : ℝ :=
  n / (4 * n + 1)

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 3 = 9) →
  (arithmetic_sequence 8 = 29) →
  (∀ n, arithmetic_sequence n = 4 * n - 3) ∧
  (∀ n, sum_of_first_n_terms n = 2 * n^2 - n) ∧
  (∀ n, sum_of_reciprocal_sequence n = n / (4 * n + 1)) :=
by
  sorry

end arithmetic_sequence_properties_l358_35895


namespace Morse_code_distinct_symbols_count_l358_35800

theorem Morse_code_distinct_symbols_count :
  let count (n : ℕ) := 2 ^ n
  count 1 + count 2 + count 3 + count 4 + count 5 = 62 :=
by
  sorry

end Morse_code_distinct_symbols_count_l358_35800


namespace percent_apple_juice_in_blend_l358_35898

noncomputable def juice_blend_apple_percentage : ℚ :=
  let apple_juice_per_apple := 9 / 2
  let plum_juice_per_plum := 12 / 3
  let total_apple_juice := 4 * apple_juice_per_apple
  let total_plum_juice := 6 * plum_juice_per_plum
  let total_juice := total_apple_juice + total_plum_juice
  (total_apple_juice / total_juice) * 100

theorem percent_apple_juice_in_blend :
  juice_blend_apple_percentage = 43 :=
by
  sorry

end percent_apple_juice_in_blend_l358_35898


namespace collinear_vectors_l358_35810

theorem collinear_vectors (x : ℝ) :
  (∃ k : ℝ, (2, 4) = (k * 2, k * 4) ∧ (k * 2 = x ∧ k * 4 = 6)) → x = 3 :=
by
  intros h
  sorry

end collinear_vectors_l358_35810


namespace hexagon_inequality_l358_35808

variable {Point : Type*} [MetricSpace Point]

-- Define points A1, A2, A3, A4, A5, A6 in a Metric Space
variables (A1 A2 A3 A4 A5 A6 O : Point)

-- Conditions
def angle_condition (O A1 A2 A3 A4 A5 A6 : Point) : Prop :=
  -- Points form a hexagon where each side is visible from O at 60 degrees
  -- We assume MetricSpace has a function measuring angles such as angle O x y = 60
  true -- A simplified condition; the actual angle measurement needs more geometry setup

def distance_condition_odd (O A1 A3 A5 : Point) : Prop := dist O A1 > dist O A3 ∧ dist O A3 > dist O A5
def distance_condition_even (O A2 A4 A6 : Point) : Prop := dist O A2 > dist O A4 ∧ dist O A4 > dist O A6

-- Question to prove
theorem hexagon_inequality 
  (hc : angle_condition O A1 A2 A3 A4 A5 A6) 
  (ho : distance_condition_odd O A1 A3 A5)
  (he : distance_condition_even O A2 A4 A6) : 
  dist A1 A2 + dist A3 A4 + dist A5 A6 < dist A2 A3 + dist A4 A5 + dist A6 A1 := 
sorry

end hexagon_inequality_l358_35808


namespace TV_cost_difference_l358_35825

def cost_per_square_inch_difference :=
  let first_TV_width := 24
  let first_TV_height := 16
  let first_TV_original_cost_euros := 840
  let first_TV_discount_percent := 0.10
  let first_TV_tax_percent := 0.05
  let exchange_rate_first := 1.20
  let first_TV_area := first_TV_width * first_TV_height

  let discounted_price_first_TV := first_TV_original_cost_euros * (1 - first_TV_discount_percent)
  let total_cost_euros_first_TV := discounted_price_first_TV * (1 + first_TV_tax_percent)
  let total_cost_dollars_first_TV := total_cost_euros_first_TV * exchange_rate_first
  let cost_per_square_inch_first_TV := total_cost_dollars_first_TV / first_TV_area

  let new_TV_width := 48
  let new_TV_height := 32
  let new_TV_original_cost_dollars := 1800
  let new_TV_first_discount_percent := 0.20
  let new_TV_second_discount_percent := 0.15
  let new_TV_tax_percent := 0.08
  let new_TV_area := new_TV_width * new_TV_height

  let price_after_first_discount := new_TV_original_cost_dollars * (1 - new_TV_first_discount_percent)
  let price_after_second_discount := price_after_first_discount * (1 - new_TV_second_discount_percent)
  let total_cost_dollars_new_TV := price_after_second_discount * (1 + new_TV_tax_percent)
  let cost_per_square_inch_new_TV := total_cost_dollars_new_TV / new_TV_area

  let cost_difference_per_square_inch := cost_per_square_inch_first_TV - cost_per_square_inch_new_TV
  cost_difference_per_square_inch

theorem TV_cost_difference :
  cost_per_square_inch_difference = 1.62 := by
  sorry

end TV_cost_difference_l358_35825


namespace find_minimum_value_M_l358_35877

theorem find_minimum_value_M : (∃ (M : ℝ), (∀ (x : ℝ), -x^2 + 2 * x ≤ M) ∧ M = 1) := 
sorry

end find_minimum_value_M_l358_35877


namespace solve_for_x_l358_35892

theorem solve_for_x 
  (y : ℚ) (x : ℚ)
  (h : x / (x - 1) = (y^3 + 2 * y^2 - 2) / (y^3 + 2 * y^2 - 3)) :
  x = (y^3 + 2 * y^2 - 2) / 2 :=
sorry

end solve_for_x_l358_35892


namespace imaginary_unit_root_l358_35871

theorem imaginary_unit_root (a b : ℝ) (h : (Complex.I : ℂ) ^ 2 + a * Complex.I + b = 0) : a + b = 1 := by
  -- Since this is just the statement, we add a sorry to focus on the structure
  sorry

end imaginary_unit_root_l358_35871


namespace swan_count_l358_35806

theorem swan_count (total_birds : ℕ) (fraction_ducks : ℚ):
  fraction_ducks = 5 / 6 →
  total_birds = 108 →
  ∃ (num_swans : ℕ), num_swans = 18 :=
by
  intro h_fraction_ducks h_total_birds
  sorry

end swan_count_l358_35806


namespace circular_patch_radius_l358_35843

theorem circular_patch_radius : 
  let r_cylinder := 3  -- radius of the container in cm
  let h_cylinder := 6  -- height of the container in cm
  let t_patch := 0.2   -- thickness of each patch in cm
  let V := π * r_cylinder^2 * h_cylinder -- Volume of the liquid

  let V_patch := V / 2                  -- Volume of each patch
  let r := 3 * Real.sqrt 15              -- the radius we want to prove

  r^2 * π * t_patch = V_patch           -- the volume equation for one patch
  →

  r = 3 * Real.sqrt 15 := 
by
  sorry

end circular_patch_radius_l358_35843


namespace solve_for_x_and_y_l358_35823

theorem solve_for_x_and_y (x y : ℚ) (h : (1 / 6) + (6 / x) = (14 / x) + (1 / 14) + y) : x = 84 ∧ y = 0 :=
sorry

end solve_for_x_and_y_l358_35823


namespace find_x_l358_35804

def is_mean_twice_mode (l : List ℕ) (mean eq_mode : ℕ) : Prop :=
  l.sum / l.length = eq_mode * 2

theorem find_x (x : ℕ) (h1 : x > 0) (h2 : x ≤ 100)
  (h3 : is_mean_twice_mode [20, x, x, x, x] x (x * 2)) : x = 10 :=
sorry

end find_x_l358_35804


namespace sum_of_infinite_series_l358_35847

noncomputable def infinite_series : ℝ :=
  ∑' k : ℕ, (k^3 : ℝ) / (3^k : ℝ)

theorem sum_of_infinite_series :
  infinite_series = (39/16 : ℝ) :=
sorry

end sum_of_infinite_series_l358_35847


namespace perfect_square_mod_3_l358_35820

theorem perfect_square_mod_3 (k : ℤ) (hk : ∃ m : ℤ, k = m^2) : k % 3 = 0 ∨ k % 3 = 1 :=
by
  sorry

end perfect_square_mod_3_l358_35820


namespace wario_missed_field_goals_wide_right_l358_35869

theorem wario_missed_field_goals_wide_right :
  ∀ (attempts missed_fraction wide_right_fraction : ℕ), 
  attempts = 60 →
  missed_fraction = 1 / 4 →
  wide_right_fraction = 20 / 100 →
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  wide_right = 3 :=
by
  intros attempts missed_fraction wide_right_fraction h1 h2 h3
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  sorry

end wario_missed_field_goals_wide_right_l358_35869


namespace power_inequality_l358_35844

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ (3 / 4) + b ^ (3 / 4) + c ^ (3 / 4) > (a + b + c) ^ (3 / 4) :=
sorry

end power_inequality_l358_35844


namespace transform_binomial_expansion_l358_35836

variable (a b : ℝ)

theorem transform_binomial_expansion (h : (a + b)^4 = a^4 + 4 * a^3 * b + 6 * a^2 * b^2 + 4 * a * b^3 + b^4) :
  (a - b)^4 = a^4 - 4 * a^3 * b + 6 * a^2 * b^2 - 4 * a * b^3 + b^4 :=
by
  sorry

end transform_binomial_expansion_l358_35836


namespace triangle_angle_A_l358_35894

theorem triangle_angle_A (C : ℝ) (c : ℝ) (a : ℝ) 
  (hC : C = 45) (hc : c = Real.sqrt 2) (ha : a = Real.sqrt 3) :
  (∃ A : ℝ, A = 60 ∨ A = 120) :=
by
  sorry

end triangle_angle_A_l358_35894


namespace longest_side_of_enclosure_l358_35812

theorem longest_side_of_enclosure
  (l w : ℝ)
  (h1 : 2 * l + 2 * w = 180)
  (h2 : l * w = 1440) :
  l = 72 ∨ w = 72 :=
by {
  sorry
}

end longest_side_of_enclosure_l358_35812


namespace find_x_l358_35848

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 152) : x = 16 := 
by 
  sorry

end find_x_l358_35848


namespace average_time_for_relay_race_l358_35891

noncomputable def average_leg_time (y_time z_time w_time x_time : ℕ) : ℚ :=
  (y_time + z_time + w_time + x_time) / 4

theorem average_time_for_relay_race :
  let y_time := 58
  let z_time := 26
  let w_time := 2 * z_time
  let x_time := 35
  average_leg_time y_time z_time w_time x_time = 42.75 := by
    sorry

end average_time_for_relay_race_l358_35891


namespace four_digit_numbers_with_three_identical_digits_l358_35880

theorem four_digit_numbers_with_three_identical_digits :
  ∃ n : ℕ, (n = 18) ∧ (∀ x, 1000 ≤ x ∧ x < 10000 → 
  (x / 1000 = 1) ∧ (
    (x % 1000 / 100 = x % 100 / 10) ∧ (x % 1000 / 100 = x % 10))) :=
by
  sorry

end four_digit_numbers_with_three_identical_digits_l358_35880


namespace find_two_numbers_l358_35874

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l358_35874


namespace intersection_x_value_l358_35835

theorem intersection_x_value :
  (∃ x y, y = 3 * x - 7 ∧ y = 48 - 5 * x) → x = 55 / 8 :=
by
  sorry

end intersection_x_value_l358_35835


namespace ball_price_equation_l358_35830

structure BallPrices where
  (x : Real) -- price of each soccer ball in yuan
  (condition1 : ∀ (x : Real), (1500 / (x + 20) - 800 / x = 5))

/-- Prove that the equation follows from the given conditions. -/
theorem ball_price_equation (b : BallPrices) : 1500 / (b.x + 20) - 800 / b.x = 5 := 
by sorry

end ball_price_equation_l358_35830


namespace foot_slide_distance_l358_35876

def ladder_foot_slide (l h_initial h_new x_initial d y: ℝ) : Prop :=
  l = 30 ∧ x_initial = 6 ∧ d = 6 ∧
  h_initial = Real.sqrt (l^2 - x_initial^2) ∧
  h_new = h_initial - d ∧
  (l^2 = h_new^2 + (x_initial + y) ^ 2) → y = 18

theorem foot_slide_distance :
  ladder_foot_slide 30 (Real.sqrt (30^2 - 6^2)) ((Real.sqrt (30^2 - 6^2)) - 6) 6 6 18 :=
by
  sorry

end foot_slide_distance_l358_35876


namespace famous_sentences_correct_l358_35878

def blank_1 : String := "correct_answer_1"
def blank_2 : String := "correct_answer_2"
def blank_3 : String := "correct_answer_3"
def blank_4 : String := "correct_answer_4"
def blank_5 : String := "correct_answer_5"
def blank_6 : String := "correct_answer_6"
def blank_7 : String := "correct_answer_7"
def blank_8 : String := "correct_answer_8"

theorem famous_sentences_correct :
  blank_1 = "correct_answer_1" ∧
  blank_2 = "correct_answer_2" ∧
  blank_3 = "correct_answer_3" ∧
  blank_4 = "correct_answer_4" ∧
  blank_5 = "correct_answer_5" ∧
  blank_6 = "correct_answer_6" ∧
  blank_7 = "correct_answer_7" ∧
  blank_8 = "correct_answer_8" :=
by
  -- The proof details correspond to the part "refer to the correct solution for each blank"
  sorry

end famous_sentences_correct_l358_35878


namespace fractional_equation_solution_l358_35813

theorem fractional_equation_solution (m : ℝ) (x : ℝ) :
  (m + 3) / (x - 1) = 1 → x > 0 → m > -4 ∧ m ≠ -3 :=
by
  sorry

end fractional_equation_solution_l358_35813


namespace largest_of_three_l358_35857

structure RealTriple (x y z : ℝ) where
  h1 : x + y + z = 3
  h2 : x * y + y * z + z * x = -8
  h3 : x * y * z = -18

theorem largest_of_three {x y z : ℝ} (h : RealTriple x y z) : max x (max y z) = Real.sqrt 5 :=
  sorry

end largest_of_three_l358_35857


namespace circle_radius_given_circumference_l358_35829

theorem circle_radius_given_circumference (C : ℝ) (hC : C = 3.14) : ∃ r : ℝ, C = 2 * Real.pi * r ∧ r = 0.5 := 
by
  sorry

end circle_radius_given_circumference_l358_35829


namespace range_of_a_l358_35821

noncomputable def A := { x : ℝ | 0 < x ∧ x < 2 }
noncomputable def B (a : ℝ) := { x : ℝ | 0 < x ∧ x < (2 / a) }

theorem range_of_a (a : ℝ) (h : 0 < a) : (A ∩ (B a)) = A → 0 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l358_35821


namespace fraction_identity_l358_35897

theorem fraction_identity (x y : ℚ) (h : x / y = 5 / 3) : y / (x - y) = 3 / 2 :=
by { sorry }

end fraction_identity_l358_35897


namespace g_9_pow_4_l358_35841

theorem g_9_pow_4 (f g : ℝ → ℝ) (h1 : ∀ x ≥ 1, f (g x) = x^2) (h2 : ∀ x ≥ 1, g (f x) = x^4) (h3 : g 81 = 81) : (g 9)^4 = 81 :=
sorry

end g_9_pow_4_l358_35841


namespace exists_k_seq_zero_to_one_l358_35851

noncomputable def seq (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) := a

theorem exists_k_seq_zero_to_one (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) :
  ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 :=
sorry

end exists_k_seq_zero_to_one_l358_35851


namespace total_amount_l358_35870

theorem total_amount (x y z total : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : y = 27) : total = 117 :=
by
  -- Proof here
  sorry

end total_amount_l358_35870


namespace complex_fraction_value_l358_35862

-- Define the imaginary unit
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_value : (3 : ℂ) / ((1 - i) ^ 2) = (3 / 2) * i := by
  sorry

end complex_fraction_value_l358_35862


namespace smallest_b_for_no_real_root_l358_35866

theorem smallest_b_for_no_real_root :
  ∃ b : ℤ, (b < 8 ∧ b > -8) ∧ (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ -6) ∧ (b = -7) :=
by
  sorry

end smallest_b_for_no_real_root_l358_35866


namespace sequence_50th_term_l358_35809

def sequence_term (n : ℕ) : ℕ × ℕ :=
  (5 + (n - 1), n - 1)

theorem sequence_50th_term :
  sequence_term 50 = (54, 49) :=
by
  sorry

end sequence_50th_term_l358_35809


namespace feet_to_inches_conversion_l358_35853

-- Define the constant equivalence between feet and inches
def foot_to_inches := 12

-- Prove the conversion factor between feet and inches
theorem feet_to_inches_conversion:
  foot_to_inches = 12 :=
by
  sorry

end feet_to_inches_conversion_l358_35853


namespace total_surface_area_of_cuboid_l358_35807

variables (l w h : ℝ)
variables (lw_area wh_area lh_area : ℝ)

def box_conditions :=
  lw_area = l * w ∧
  wh_area = w * h ∧
  lh_area = l * h

theorem total_surface_area_of_cuboid (hc : box_conditions l w h 120 72 60) :
  2 * (120 + 72 + 60) = 504 :=
sorry

end total_surface_area_of_cuboid_l358_35807


namespace common_ratio_is_2_l358_35889

noncomputable def arithmetic_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1)

theorem common_ratio_is_2 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 > 0)
  (h3 : arithmetic_sequence_common_ratio a q) :
  q = 2 :=
sorry

end common_ratio_is_2_l358_35889


namespace coordinate_identification_l358_35858

noncomputable def x1 := (4 * Real.pi) / 5
noncomputable def y1 := -(Real.pi) / 5

noncomputable def x2 := (12 * Real.pi) / 5
noncomputable def y2 := -(3 * Real.pi) / 5

noncomputable def x3 := (4 * Real.pi) / 3
noncomputable def y3 := -(Real.pi) / 3

theorem coordinate_identification :
  (x1, y1) = (4 * Real.pi / 5, -(Real.pi) / 5) ∧
  (x2, y2) = (12 * Real.pi / 5, -(3 * Real.pi) / 5) ∧
  (x3, y3) = (4 * Real.pi / 3, -(Real.pi) / 3) :=
by
  -- proof goes here
  sorry

end coordinate_identification_l358_35858


namespace min_value_of_quadratic_l358_35827

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 8 * x + 15 → y ≥ -1) ∧ (∃ x₀ : ℝ, x₀ = 4 ∧ (x₀^2 - 8 * x₀ + 15 = -1)) :=
by
  sorry

end min_value_of_quadratic_l358_35827


namespace molecular_weight_8_moles_Al2O3_l358_35856

noncomputable def molecular_weight_Al2O3 (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) : ℝ :=
  2 * atomic_weight_Al + 3 * atomic_weight_O

theorem molecular_weight_8_moles_Al2O3
  (atomic_weight_Al : ℝ := 26.98)
  (atomic_weight_O : ℝ := 16.00)
  : molecular_weight_Al2O3 atomic_weight_Al atomic_weight_O * 8 = 815.68 := by
  sorry

end molecular_weight_8_moles_Al2O3_l358_35856


namespace find_theta_l358_35828

theorem find_theta
  (θ : ℝ)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (ha : ∃ k, (2 * Real.cos θ, 2 * Real.sin θ) = (k * 3, k * Real.sqrt 3)) :
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 :=
by
  sorry

end find_theta_l358_35828


namespace tom_jerry_age_ratio_l358_35875

-- Definitions representing the conditions in the problem
variable (t j x : ℕ)

-- Condition 1: Three years ago, Tom was three times as old as Jerry
def condition1 : Prop := t - 3 = 3 * (j - 3)

-- Condition 2: Four years before that, Tom was five times as old as Jerry
def condition2 : Prop := t - 7 = 5 * (j - 7)

-- Question: In how many years will the ratio of their ages be 3:2,
-- asserting that the answer is 21
def ageRatioInYears : Prop := (t + x) / (j + x) = 3 / 2 → x = 21

-- The proposition we need to prove
theorem tom_jerry_age_ratio (h1 : condition1 t j) (h2 : condition2 t j) : ageRatioInYears t j x := 
  sorry
  
end tom_jerry_age_ratio_l358_35875


namespace trajectory_of_center_of_moving_circle_l358_35840

noncomputable def circle_tangency_condition_1 (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 1
noncomputable def circle_tangency_condition_2 (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 9

def ellipse_equation (x y : ℝ) : Prop := x ^ 2 / 4 + y ^ 2 / 3 = 1

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  circle_tangency_condition_1 x y ∧ circle_tangency_condition_2 x y →
  ellipse_equation x y := sorry

end trajectory_of_center_of_moving_circle_l358_35840


namespace relationship_among_three_numbers_l358_35826

theorem relationship_among_three_numbers :
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  b < a ∧ a < c :=
by
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  sorry

end relationship_among_three_numbers_l358_35826


namespace find_b_c_d_l358_35867

def f (x : ℝ) := x^3 + 2 * x^2 + 3 * x + 4
def h (x : ℝ) := x^3 + 6 * x^2 - 8 * x + 16

theorem find_b_c_d :
  (∀ r : ℝ, f r = 0 → h (r^3) = 0) ∧ h (x : ℝ) = x^3 + 6 * x^2 - 8 * x + 16 :=
by 
  -- proof not required
  sorry

end find_b_c_d_l358_35867


namespace number_of_students_in_class_l358_35890

theorem number_of_students_in_class
  (total_stickers : ℕ) (stickers_to_friends : ℕ) (stickers_left : ℝ) (students_each : ℕ → ℝ)
  (n_friends : ℕ) (remaining_stickers : ℝ) :
  total_stickers = 300 →
  stickers_to_friends = (n_friends * (n_friends + 1)) / 2 →
  stickers_left = 7.5 →
  ∀ n, n_friends = 10 →
  remaining_stickers = total_stickers - stickers_to_friends - (students_each n_friends) * (n - n_friends - 1) →
  (∃ n : ℕ, remaining_stickers = 7.5 ∧
              total_stickers - (stickers_to_friends + (students_each (n - n_friends - 1) * (n - n_friends - 1))) = 7.5) :=
by
  sorry

end number_of_students_in_class_l358_35890


namespace coffee_price_l358_35888

theorem coffee_price (C : ℝ) :
  (7 * C) + (8 * 4) = 67 → C = 5 :=
by
  intro h
  sorry

end coffee_price_l358_35888


namespace ratio_3_7_not_possible_l358_35822

theorem ratio_3_7_not_possible (n : ℕ) (h : 30 < n ∧ n < 40) :
  ¬ (∃ k : ℕ, n = 10 * k) :=
by {
  sorry
}

end ratio_3_7_not_possible_l358_35822


namespace find_matrix_A_l358_35884

-- Define the condition that A v = 3 v for all v in R^3
def satisfiesCondition (A : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  ∀ (v : Fin 3 → ℝ), A.mulVec v = 3 • v

theorem find_matrix_A (A : Matrix (Fin 3) (Fin 3) ℝ) :
  satisfiesCondition A → A = 3 • 1 :=
by
  intro h
  sorry

end find_matrix_A_l358_35884


namespace sum_of_constants_l358_35861

theorem sum_of_constants :
  ∃ (a b c d e : ℤ), 1000 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e) ∧ a + b + c + d + e = 92 :=
by
  sorry

end sum_of_constants_l358_35861


namespace wall_number_of_bricks_l358_35842

theorem wall_number_of_bricks (x : ℝ) :
  (∃ x, 6 * ((x / 7) + (x / 11) - 12) = x) →  x = 179 :=
by
  sorry

end wall_number_of_bricks_l358_35842


namespace find_x_coordinate_l358_35815

theorem find_x_coordinate :
  ∃ x : ℝ, (∃ m b : ℝ, (∀ y x : ℝ, y = m * x + b) ∧ 
                     ((3 = m * 10 + b) ∧ 
                      (0 = m * 4 + b)
                     ) ∧ 
                     (-3 = m * x + b) ∧ 
                     (x = -2)) :=
sorry

end find_x_coordinate_l358_35815


namespace a_eq_bn_l358_35899

theorem a_eq_bn (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → ∃ m : ℕ, a - k^n = m * (b - k)) → a = b^n :=
by
  sorry

end a_eq_bn_l358_35899
