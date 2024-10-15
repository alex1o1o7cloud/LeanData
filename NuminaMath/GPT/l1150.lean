import Mathlib

namespace NUMINAMATH_GPT_exists_base_for_part_a_not_exists_base_for_part_b_l1150_115027

theorem exists_base_for_part_a : ∃ b : ℕ, (3 + 4 = b) ∧ (3 * 4 = 1 * b + 5) := 
by
  sorry

theorem not_exists_base_for_part_b : ¬ ∃ b : ℕ, (2 + 3 = b) ∧ (2 * 3 = 1 * b + 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_base_for_part_a_not_exists_base_for_part_b_l1150_115027


namespace NUMINAMATH_GPT_number_whose_square_is_64_l1150_115000

theorem number_whose_square_is_64 (x : ℝ) (h : x^2 = 64) : x = 8 ∨ x = -8 :=
sorry

end NUMINAMATH_GPT_number_whose_square_is_64_l1150_115000


namespace NUMINAMATH_GPT_arithmetic_twelfth_term_l1150_115016

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_twelfth_term_l1150_115016


namespace NUMINAMATH_GPT_four_digit_numbers_count_l1150_115059

theorem four_digit_numbers_count : 
  (∀ d1 d2 d3 d4 : Fin 4, 
    (d1 = 1 ∨ d1 = 2 ∨ d1 = 3) ∧ 
    d2 ≠ d1 ∧ d2 ≠ 0 ∧ 
    d3 ≠ d1 ∧ d3 ≠ d2 ∧ 
    d4 ≠ d1 ∧ d4 ≠ d2 ∧ d4 ≠ d3) →
  3 * 6 = 18 := 
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_count_l1150_115059


namespace NUMINAMATH_GPT_geometric_series_sum_l1150_115021

theorem geometric_series_sum:
  let a := 1
  let r := 5
  let n := 5
  (1 - r^n) / (1 - r) = 781 :=
by
  let a := 1
  let r := 5
  let n := 5
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1150_115021


namespace NUMINAMATH_GPT_smallest_angle_of_triangle_l1150_115098

theorem smallest_angle_of_triangle (x : ℝ) (h : 3 * x + 4 * x + 5 * x = 180) : 3 * x = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_of_triangle_l1150_115098


namespace NUMINAMATH_GPT_solve_fractional_equation_l1150_115065

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 0) : (x + 1) / x = 2 / 3 ↔ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1150_115065


namespace NUMINAMATH_GPT_find_other_leg_length_l1150_115083

theorem find_other_leg_length (a b c : ℝ) (h1 : a = 15) (h2 : b = 5 * Real.sqrt 3) (h3 : c = 2 * (5 * Real.sqrt 3)) (h4 : a^2 + b^2 = c^2)
  (angle_A : ℝ) (h5 : angle_A = Real.pi / 3) (h6 : angle_A ≠ Real.pi / 2) :
  b = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_other_leg_length_l1150_115083


namespace NUMINAMATH_GPT_range_of_a_l1150_115095

open Real

theorem range_of_a (k a : ℝ) : 
  (∀ k : ℝ, ∀ x y : ℝ, k * x - y - k + 2 = 0 → x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) ↔ 
  (a ∈ Set.Ioo (-7 : ℝ) (-2) ∪ Set.Ioi 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1150_115095


namespace NUMINAMATH_GPT_area_ratio_proof_l1150_115009

noncomputable def area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) : ℝ := 
  (a * b) / (c * d)

theorem area_ratio_proof (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  area_ratio a b c d h1 h2 = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_area_ratio_proof_l1150_115009


namespace NUMINAMATH_GPT_three_digit_numbers_count_correct_l1150_115073

def digits : List ℕ := [2, 3, 4, 5, 5, 5, 6, 6]

def three_digit_numbers_count (d : List ℕ) : ℕ := 
  -- To be defined: Full implementation for counting matching three-digit numbers
  sorry

theorem three_digit_numbers_count_correct :
  three_digit_numbers_count digits = 85 :=
sorry

end NUMINAMATH_GPT_three_digit_numbers_count_correct_l1150_115073


namespace NUMINAMATH_GPT_smallest_solution_to_equation_l1150_115069

theorem smallest_solution_to_equation :
  let x := 4 - Real.sqrt 2
  ∃ x, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
       ∀ y, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y :=
  by
    let x := 4 - Real.sqrt 2
    sorry

end NUMINAMATH_GPT_smallest_solution_to_equation_l1150_115069


namespace NUMINAMATH_GPT_sum_of_edges_of_square_l1150_115014

theorem sum_of_edges_of_square (u v w x : ℕ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) 
(hsum : u * x + u * v + v * w + w * x = 15) : u + v + w + x = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_edges_of_square_l1150_115014


namespace NUMINAMATH_GPT_total_seashells_l1150_115037

theorem total_seashells 
  (sally_seashells : ℕ)
  (tom_seashells : ℕ)
  (jessica_seashells : ℕ)
  (h1 : sally_seashells = 9)
  (h2 : tom_seashells = 7)
  (h3 : jessica_seashells = 5) : 
  sally_seashells + tom_seashells + jessica_seashells = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_seashells_l1150_115037


namespace NUMINAMATH_GPT_find_c_d_l1150_115051

theorem find_c_d (y : ℝ) (c d : ℕ) (hy : y^2 + 4*y + 4/y + 1/y^2 = 35)
  (hform : ∃ (c d : ℕ), y = c + Real.sqrt d) : c + d = 42 :=
sorry

end NUMINAMATH_GPT_find_c_d_l1150_115051


namespace NUMINAMATH_GPT_gnuff_tutoring_minutes_l1150_115013

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end NUMINAMATH_GPT_gnuff_tutoring_minutes_l1150_115013


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l1150_115020

open Real

theorem solve_quadratic_1 :
  (∃ x : ℝ, x^2 - 2 * x - 7 = 0) ∧
  (∀ x : ℝ, x^2 - 2 * x - 7 = 0 → x = 1 + 2 * sqrt 2 ∨ x = 1 - 2 * sqrt 2) :=
sorry

theorem solve_quadratic_2 :
  (∃ x : ℝ, 3 * (x - 2)^2 = x * (x - 2)) ∧
  (∀ x : ℝ, 3 * (x - 2)^2 = x * (x - 2) → x = 2 ∨ x = 3) :=
sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l1150_115020


namespace NUMINAMATH_GPT_molecular_weight_4_benzoic_acid_l1150_115057

def benzoic_acid_molecular_weight : Float := (7 * 12.01) + (6 * 1.008) + (2 * 16.00)

def molecular_weight_4_moles_benzoic_acid (molecular_weight : Float) : Float := molecular_weight * 4

theorem molecular_weight_4_benzoic_acid :
  molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight = 488.472 :=
by
  unfold molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight
  -- rest of the proof
  sorry

end NUMINAMATH_GPT_molecular_weight_4_benzoic_acid_l1150_115057


namespace NUMINAMATH_GPT_solve_for_a_l1150_115048

theorem solve_for_a
  (h : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (x^2 - a * x + 2 < 0)) :
  a = 3 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l1150_115048


namespace NUMINAMATH_GPT_division_quotient_example_l1150_115077

theorem division_quotient_example :
  ∃ q : ℕ,
    let dividend := 760
    let divisor := 36
    let remainder := 4
    dividend = divisor * q + remainder ∧ q = 21 :=
by
  sorry

end NUMINAMATH_GPT_division_quotient_example_l1150_115077


namespace NUMINAMATH_GPT_ball_third_bounce_distance_is_correct_l1150_115089

noncomputable def total_distance_third_bounce (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  initial_height + 2 * (initial_height * rebound_ratio) + 2 * (initial_height * rebound_ratio^2)

theorem ball_third_bounce_distance_is_correct : 
  total_distance_third_bounce 80 (2/3) = 257.78 := 
by
  sorry

end NUMINAMATH_GPT_ball_third_bounce_distance_is_correct_l1150_115089


namespace NUMINAMATH_GPT_find_x_l1150_115082

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem find_x (x : ℝ) (hx : x > 0) :
  distance (1, 3) (x, -4) = 15 → x = 1 + Real.sqrt 176 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1150_115082


namespace NUMINAMATH_GPT_problem_solution_l1150_115002

theorem problem_solution
  (x : ℝ) (a b : ℕ) (hx_pos : 0 < x) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_eq : x ^ 2 + 5 * x + 5 / x + 1 / x ^ 2 = 40)
  (h_form : x = a + Real.sqrt b) :
  a + b = 11 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1150_115002


namespace NUMINAMATH_GPT_find_a_l1150_115099

theorem find_a
  (a : ℝ)
  (h1 : ∃ P Q : ℝ × ℝ, (P.1 ^ 2 + P.2 ^ 2 - 2 * P.1 + 4 * P.2 + 1 = 0) ∧ (Q.1 ^ 2 + Q.2 ^ 2 - 2 * Q.1 + 4 * Q.2 + 1 = 0) ∧
                         (a * P.1 + 2 * P.2 + 6 = 0) ∧ (a * Q.1 + 2 * Q.2 + 6 = 0) ∧
                         ((P.1 - 1) * (Q.1 - 1) + (P.2 + 2) * (Q.2 + 2) = 0)) :
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1150_115099


namespace NUMINAMATH_GPT_minimum_slit_length_l1150_115032

theorem minimum_slit_length (circumference : ℝ) (speed_ratio : ℝ) (reliability : ℝ) :
  circumference = 1 → speed_ratio = 2 → (∀ (s : ℝ), (s < 2/3) → (¬ reliable)) → reliability =
    2 / 3 :=
by
  intros hcirc hspeed hrel
  have s := (2 : ℝ) / 3
  sorry

end NUMINAMATH_GPT_minimum_slit_length_l1150_115032


namespace NUMINAMATH_GPT_greatest_value_product_l1150_115010

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def divisible_by (m n : ℕ) : Prop := ∃ k, m = k * n

theorem greatest_value_product (a b : ℕ) : 
    is_prime a → is_prime b → a < 10 → b < 10 → divisible_by (110 + 10 * a + b) 55 → a * b = 15 :=
by
    sorry

end NUMINAMATH_GPT_greatest_value_product_l1150_115010


namespace NUMINAMATH_GPT_solution_unique_l1150_115088

def satisfies_equation (x y : ℝ) : Prop :=
  (x - 7)^2 + (y - 8)^2 + (x - y)^2 = 1 / 3

theorem solution_unique (x y : ℝ) :
  satisfies_equation x y ↔ x = 7 + 1/3 ∧ y = 8 - 1/3 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_unique_l1150_115088


namespace NUMINAMATH_GPT_pyramid_base_edge_length_l1150_115012

theorem pyramid_base_edge_length (height : ℝ) (radius : ℝ) (side_len : ℝ) :
  height = 4 ∧ radius = 3 →
  side_len = (12 * Real.sqrt 14) / 7 :=
by
  intros h
  rcases h with ⟨h1, h2⟩
  sorry

end NUMINAMATH_GPT_pyramid_base_edge_length_l1150_115012


namespace NUMINAMATH_GPT_number_of_cows_on_boat_l1150_115064

-- Definitions based on conditions
def number_of_sheep := 20
def number_of_dogs := 14
def sheep_drowned := 3
def cows_drowned := 2 * sheep_drowned  -- Twice as many cows drowned as did sheep.
def dogs_made_it_shore := number_of_dogs  -- All dogs made it to shore.
def total_animals_shore := 35
def total_sheep_shore := number_of_sheep - sheep_drowned
def total_sheep_cows_shore := total_animals_shore - dogs_made_it_shore
def cows_made_it_shore := total_sheep_cows_shore - total_sheep_shore

-- Theorem stating the problem
theorem number_of_cows_on_boat : 
  (cows_made_it_shore + cows_drowned) = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_cows_on_boat_l1150_115064


namespace NUMINAMATH_GPT_avg_marks_calculation_l1150_115036

theorem avg_marks_calculation (max_score : ℕ)
    (gibi_percent jigi_percent mike_percent lizzy_percent : ℚ)
    (hg : gibi_percent = 0.59) (hj : jigi_percent = 0.55) 
    (hm : mike_percent = 0.99) (hl : lizzy_percent = 0.67)
    (hmax : max_score = 700) :
    ((gibi_percent * max_score + jigi_percent * max_score +
      mike_percent * max_score + lizzy_percent * max_score) / 4 = 490) :=
by
  sorry

end NUMINAMATH_GPT_avg_marks_calculation_l1150_115036


namespace NUMINAMATH_GPT_player_A_wins_even_n_l1150_115068

theorem player_A_wins_even_n (n : ℕ) (hn : n > 0) (even_n : Even n) :
  ∃ strategy_A : ℕ → Bool, 
    ∀ (P Q : ℕ), P % 2 = 0 → (Q + P) % 2 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_player_A_wins_even_n_l1150_115068


namespace NUMINAMATH_GPT_number_of_lamps_bought_l1150_115080

-- Define the given conditions
def price_of_lamp : ℕ := 7
def price_of_bulb : ℕ := price_of_lamp - 4
def bulbs_bought : ℕ := 6
def total_spent : ℕ := 32

-- Define the statement to prove
theorem number_of_lamps_bought : 
  ∃ (L : ℕ), (price_of_lamp * L + price_of_bulb * bulbs_bought = total_spent) ∧ (L = 2) :=
sorry

end NUMINAMATH_GPT_number_of_lamps_bought_l1150_115080


namespace NUMINAMATH_GPT_all_pets_combined_l1150_115060

def Teddy_initial_dogs : Nat := 7
def Teddy_initial_cats : Nat := 8
def Teddy_initial_rabbits : Nat := 6

def Teddy_adopted_dogs : Nat := 2
def Teddy_adopted_rabbits : Nat := 4

def Ben_dogs : Nat := 3 * Teddy_initial_dogs
def Ben_cats : Nat := 2 * Teddy_initial_cats

def Dave_dogs : Nat := (Teddy_initial_dogs + Teddy_adopted_dogs) - 4
def Dave_cats : Nat := Teddy_initial_cats + 13
def Dave_rabbits : Nat := 3 * Teddy_initial_rabbits

def Teddy_current_dogs : Nat := Teddy_initial_dogs + Teddy_adopted_dogs
def Teddy_current_cats : Nat := Teddy_initial_cats
def Teddy_current_rabbits : Nat := Teddy_initial_rabbits + Teddy_adopted_rabbits

def Teddy_total : Nat := Teddy_current_dogs + Teddy_current_cats + Teddy_current_rabbits
def Ben_total : Nat := Ben_dogs + Ben_cats
def Dave_total : Nat := Dave_dogs + Dave_cats + Dave_rabbits

def total_pets_combined : Nat := Teddy_total + Ben_total + Dave_total

theorem all_pets_combined : total_pets_combined = 108 :=
by
  sorry

end NUMINAMATH_GPT_all_pets_combined_l1150_115060


namespace NUMINAMATH_GPT_sum_of_absolute_values_l1150_115041

variables {a : ℕ → ℤ} {S₁₀ S₁₈ : ℤ} {T₁₈ : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

def sum_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem sum_of_absolute_values 
  (h1 : a 0 > 0) 
  (h2 : a 9 * a 10 < 0) 
  (h3 : sum_n_terms a 9 = 36) 
  (h4 : sum_n_terms a 17 = 12) :
  (sum_n_terms a 9) - (sum_n_terms a 17 - sum_n_terms a 9) = 60 :=
sorry

end NUMINAMATH_GPT_sum_of_absolute_values_l1150_115041


namespace NUMINAMATH_GPT_volume_is_six_l1150_115038

-- Define the polygons and their properties
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0)
def rectangle (l w : ℝ) := (l > 0 ∧ w > 0)
def equilateral_triangle (s : ℝ) := (s > 0)

-- The given polygons
def A := right_triangle 1 2 (Real.sqrt 5)
def E := right_triangle 1 2 (Real.sqrt 5)
def F := right_triangle 1 2 (Real.sqrt 5)
def B := rectangle 1 2
def C := rectangle 2 3
def D := rectangle 1 3
def G := equilateral_triangle (Real.sqrt 5)

-- The volume of the polyhedron
-- Assume the largest rectangle C forms the base and a reasonable height
def volume_of_polyhedron : ℝ := 6

theorem volume_is_six : 
  (right_triangle 1 2 (Real.sqrt 5)) → 
  (rectangle 1 2) → 
  (rectangle 2 3) → 
  (rectangle 1 3) → 
  (equilateral_triangle (Real.sqrt 5)) → 
  volume_of_polyhedron = 6 := 
by 
  sorry

end NUMINAMATH_GPT_volume_is_six_l1150_115038


namespace NUMINAMATH_GPT_jesus_squares_l1150_115023

theorem jesus_squares (J : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : linden_squares = 75)
  (h2 : pedro_squares = 200)
  (h3 : pedro_squares = J + linden_squares + 65) : 
  J = 60 := 
by
  sorry

end NUMINAMATH_GPT_jesus_squares_l1150_115023


namespace NUMINAMATH_GPT_simplify_expression_l1150_115049

theorem simplify_expression (a b : ℤ) (h1 : a = 1) (h2 : b = -4) :
  4 * (a^2 * b + a * b^2) - 3 * (a^2 * b - 1) + 2 * a * b^2 - 6 = 89 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1150_115049


namespace NUMINAMATH_GPT_solve_for_k_l1150_115061

theorem solve_for_k : 
  ∃ (k : ℕ), k > 0 ∧ k * k = 2012 * 2012 + 2010 * 2011 * 2013 * 2014 ∧ k = 4048142 :=
sorry

end NUMINAMATH_GPT_solve_for_k_l1150_115061


namespace NUMINAMATH_GPT_ajay_gain_l1150_115018

-- Definitions of the problem conditions as Lean variables/constants.
variables (kg1 kg2 kg_total : ℕ) 
variables (price1 price2 price3 cost1 cost2 total_cost selling_price gain : ℝ)

-- Conditions of the problem.
def conditions : Prop :=
  kg1 = 15 ∧ 
  kg2 = 10 ∧ 
  kg_total = kg1 + kg2 ∧ 
  price1 = 14.5 ∧ 
  price2 = 13 ∧ 
  price3 = 15 ∧ 
  cost1 = kg1 * price1 ∧ 
  cost2 = kg2 * price2 ∧ 
  total_cost = cost1 + cost2 ∧ 
  selling_price = kg_total * price3 ∧ 
  gain = selling_price - total_cost 

-- The theorem for the gain amount proof.
theorem ajay_gain (h : conditions kg1 kg2 kg_total price1 price2 price3 cost1 cost2 total_cost selling_price gain) : 
  gain = 27.50 :=
  sorry

end NUMINAMATH_GPT_ajay_gain_l1150_115018


namespace NUMINAMATH_GPT_find_common_ratio_l1150_115008

-- Define the variables and constants involved.
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)

-- Define the conditions of the problem.
def is_geometric_sequence := ∀ n, a (n + 1) = q * a n
def sum_of_first_n_terms := ∀ n, S n = a 0 * (1 - q^(n + 1)) / (1 - q)
def condition1 := a 5 = 4 * S 4 + 3
def condition2 := a 6 = 4 * S 5 + 3

-- The main statement that needs to be proved.
theorem find_common_ratio
  (h1: is_geometric_sequence a q)
  (h2: sum_of_first_n_terms a S q)
  (h3: condition1 a S)
  (h4: condition2 a S) : 
  q = 5 :=
sorry -- proof to be provided

end NUMINAMATH_GPT_find_common_ratio_l1150_115008


namespace NUMINAMATH_GPT_people_in_third_row_l1150_115055

theorem people_in_third_row (row1_ini row2_ini left_row1 left_row2 total_left : ℕ) (h1 : row1_ini = 24) (h2 : row2_ini = 20) (h3 : left_row1 = row1_ini - 3) (h4 : left_row2 = row2_ini - 5) (h_total : total_left = 54) :
  total_left - (left_row1 + left_row2) = 18 := 
by
  sorry

end NUMINAMATH_GPT_people_in_third_row_l1150_115055


namespace NUMINAMATH_GPT_roots_quadratic_expression_value_l1150_115094

theorem roots_quadratic_expression_value (m n : ℝ) 
  (h1 : m^2 + 2 * m - 2027 = 0)
  (h2 : n^2 + 2 * n - 2027 = 0) :
  (2 * m - m * n + 2 * n) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_expression_value_l1150_115094


namespace NUMINAMATH_GPT_radius_of_circle_from_chord_and_line_l1150_115085

theorem radius_of_circle_from_chord_and_line (r : ℝ) (t θ : ℝ) 
    (param_line : ℝ × ℝ) (param_circle : ℝ × ℝ)
    (chord_length : ℝ) 
    (h1 : param_line = (3 + 3 * t, 1 - 4 * t))
    (h2 : param_circle = (r * Real.cos θ, r * Real.sin θ))
    (h3 : chord_length = 4) 
    : r = Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_radius_of_circle_from_chord_and_line_l1150_115085


namespace NUMINAMATH_GPT_prob_exactly_M_laws_expected_laws_included_l1150_115091

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end NUMINAMATH_GPT_prob_exactly_M_laws_expected_laws_included_l1150_115091


namespace NUMINAMATH_GPT_solve_inequalities_l1150_115063

theorem solve_inequalities (x : ℝ) :
    ((x / 2 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x))) ↔ (-6 ≤ x ∧ x < -3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l1150_115063


namespace NUMINAMATH_GPT_matrix_product_is_correct_l1150_115007

-- Define the matrices A and B
def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 1, 1],
  ![2, 1, 2],
  ![1, 2, 3]
]

def B : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![1, 1, -1],
  ![2, -1, 1],
  ![1, 0, 1]
]

-- Define the expected product matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![6, 2, -1],
  ![6, 1, 1],
  ![8, -1, 4]
]

-- The statement of the problem
theorem matrix_product_is_correct : (A * B) = C := by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_GPT_matrix_product_is_correct_l1150_115007


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1150_115047

theorem boat_speed_in_still_water (B S : ℕ) (h1 : B + S = 13) (h2 : B - S = 5) : B = 9 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1150_115047


namespace NUMINAMATH_GPT_range_of_expr_l1150_115035

theorem range_of_expr (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
by
  sorry

end NUMINAMATH_GPT_range_of_expr_l1150_115035


namespace NUMINAMATH_GPT_original_denominator_l1150_115096

theorem original_denominator (d : ℤ) (h1 : 5 = d + 3) : d = 12 := 
by 
  sorry

end NUMINAMATH_GPT_original_denominator_l1150_115096


namespace NUMINAMATH_GPT_num_men_in_first_group_l1150_115031

variable {x m w : ℝ}

theorem num_men_in_first_group (h1 : x * m + 8 * w = 6 * m + 2 * w)
  (h2 : 2 * m + 3 * w = 0.5 * (x * m + 8 * w)) : 
  x = 3 :=
sorry

end NUMINAMATH_GPT_num_men_in_first_group_l1150_115031


namespace NUMINAMATH_GPT_sum_of_digits_of_fraction_is_nine_l1150_115054

theorem sum_of_digits_of_fraction_is_nine : 
  ∃ (x y : Nat), (4 / 11 : ℚ) = x / 10 + y / 100 + x / 1000 + y / 10000 + (x + y) / 100000 -- and other terms
  ∧ x + y = 9 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_of_fraction_is_nine_l1150_115054


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1150_115081

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ ¬ (|x| > 1 → x > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1150_115081


namespace NUMINAMATH_GPT_total_value_of_gold_is_l1150_115093

-- Definitions based on the conditions
def legacyBars : ℕ := 5
def aleenaBars : ℕ := legacyBars - 2
def valuePerBar : ℝ := 2200
def totalValue : ℝ := (legacyBars + aleenaBars) * valuePerBar

-- Theorem statement
theorem total_value_of_gold_is :
  totalValue = 17600 := by
  -- We add sorry here to skip the proof
  sorry

end NUMINAMATH_GPT_total_value_of_gold_is_l1150_115093


namespace NUMINAMATH_GPT_find_k_l1150_115019

noncomputable def vec_na (x1 k : ℝ) : ℝ × ℝ := (x1 - k/4, 2 * x1^2)
noncomputable def vec_nb (x2 k : ℝ) : ℝ × ℝ := (x2 - k/4, 2 * x2^2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem find_k (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k / 2) 
  (h2 : x1 * x2 = -1) 
  (h3 : dot_product (vec_na x1 k) (vec_nb x2 k) = 0) : 
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1150_115019


namespace NUMINAMATH_GPT_mrs_sheridan_fish_count_l1150_115092

/-
  Problem statement: 
  Prove that the total number of fish Mrs. Sheridan has now is 69, 
  given that she initially had 22 fish and she received 47 more from her sister.
-/

theorem mrs_sheridan_fish_count :
  let initial_fish : ℕ := 22
  let additional_fish : ℕ := 47
  initial_fish + additional_fish = 69 := by
sorry

end NUMINAMATH_GPT_mrs_sheridan_fish_count_l1150_115092


namespace NUMINAMATH_GPT_complementary_event_target_l1150_115084

theorem complementary_event_target (S : Type) (hit miss : S) (shoots : ℕ → S) :
  (∀ n : ℕ, (shoots n = hit ∨ shoots n = miss)) →
  (∃ n : ℕ, shoots n = hit) ↔ (∀ n : ℕ, shoots n ≠ hit) :=
by
sorry

end NUMINAMATH_GPT_complementary_event_target_l1150_115084


namespace NUMINAMATH_GPT_prove_inequality_l1150_115001

theorem prove_inequality (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 :=
  sorry

end NUMINAMATH_GPT_prove_inequality_l1150_115001


namespace NUMINAMATH_GPT_xy_yz_zx_value_l1150_115005

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 9) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + z * x + x^2 = 25) :
  x * y + y * z + z * x = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_xy_yz_zx_value_l1150_115005


namespace NUMINAMATH_GPT_gcd_1458_1479_l1150_115017

def a : ℕ := 1458
def b : ℕ := 1479
def gcd_ab : ℕ := 21

theorem gcd_1458_1479 : Nat.gcd a b = gcd_ab := sorry

end NUMINAMATH_GPT_gcd_1458_1479_l1150_115017


namespace NUMINAMATH_GPT_buffy_less_brittany_by_40_seconds_l1150_115097

/-
The following statement proves that Buffy's breath-holding time was 40 seconds less than Brittany's, 
given the initial conditions about their breath-holding times.
-/
theorem buffy_less_brittany_by_40_seconds 
  (kelly_time : ℕ) 
  (brittany_time : ℕ) 
  (buffy_time : ℕ) 
  (h_kelly : kelly_time = 180) 
  (h_brittany : brittany_time = kelly_time - 20) 
  (h_buffy : buffy_time = 120)
  :
  brittany_time - buffy_time = 40 :=
sorry

end NUMINAMATH_GPT_buffy_less_brittany_by_40_seconds_l1150_115097


namespace NUMINAMATH_GPT_bank_robbery_participants_l1150_115078

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end NUMINAMATH_GPT_bank_robbery_participants_l1150_115078


namespace NUMINAMATH_GPT_contrapositive_equivalence_l1150_115044

variable (Person : Type)
variable (Happy Have : Person → Prop)

theorem contrapositive_equivalence :
  (∀ (x : Person), Happy x → Have x) ↔ (∀ (x : Person), ¬Have x → ¬Happy x) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_equivalence_l1150_115044


namespace NUMINAMATH_GPT_investment_plan_optimization_l1150_115072

-- Define the given conditions.
def max_investment : ℝ := 100000
def max_loss : ℝ := 18000
def max_profit_A_rate : ℝ := 1.0     -- 100%
def max_profit_B_rate : ℝ := 0.5     -- 50%
def max_loss_A_rate : ℝ := 0.3       -- 30%
def max_loss_B_rate : ℝ := 0.1       -- 10%

-- Define the investment amounts.
def invest_A : ℝ := 40000
def invest_B : ℝ := 60000

-- Calculate profit and loss.
def profit : ℝ := (invest_A * max_profit_A_rate) + (invest_B * max_profit_B_rate)
def loss : ℝ := (invest_A * max_loss_A_rate) + (invest_B * max_loss_B_rate)
def total_investment : ℝ := invest_A + invest_B

-- Prove the required statement.
theorem investment_plan_optimization : 
    total_investment ≤ max_investment ∧ loss ≤ max_loss ∧ profit = 70000 :=
by
  simp [total_investment, profit, loss, invest_A, invest_B, 
    max_investment, max_profit_A_rate, max_profit_B_rate, 
    max_loss_A_rate, max_loss_B_rate, max_loss]
  sorry

end NUMINAMATH_GPT_investment_plan_optimization_l1150_115072


namespace NUMINAMATH_GPT_intersection_A_B_l1150_115015

def A : Set ℝ := { x : ℝ | |x - 1| < 2 }
def B : Set ℝ := { x : ℝ | x^2 - x - 2 > 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1150_115015


namespace NUMINAMATH_GPT_distinct_real_roots_range_l1150_115046

theorem distinct_real_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ax^2 + 2 * x + 1 = 0 ∧ ay^2 + 2 * y + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_range_l1150_115046


namespace NUMINAMATH_GPT_drinks_left_for_Seungwoo_l1150_115024

def coke_taken_liters := 35 + 0.5
def cider_taken_liters := 27 + 0.2
def coke_drank_liters := 1 + 0.75

theorem drinks_left_for_Seungwoo :
  (coke_taken_liters - coke_drank_liters) + cider_taken_liters = 60.95 := by
  sorry

end NUMINAMATH_GPT_drinks_left_for_Seungwoo_l1150_115024


namespace NUMINAMATH_GPT_olympic_volunteers_selection_l1150_115042

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem olympic_volunteers_selection :
  (choose 4 3 * choose 3 1) + (choose 4 2 * choose 3 2) + (choose 4 1 * choose 3 3) = 34 := 
by
  sorry

end NUMINAMATH_GPT_olympic_volunteers_selection_l1150_115042


namespace NUMINAMATH_GPT_painter_total_cost_l1150_115070

def south_seq (n : Nat) : Nat :=
  4 + 6 * (n - 1)

def north_seq (n : Nat) : Nat :=
  5 + 6 * (n - 1)

noncomputable def digit_cost (n : Nat) : Nat :=
  String.length (toString n)

noncomputable def total_cost : Nat :=
  let south_cost := (List.range 25).map south_seq |>.map digit_cost |>.sum
  let north_cost := (List.range 25).map north_seq |>.map digit_cost |>.sum
  south_cost + north_cost

theorem painter_total_cost : total_cost = 116 := by
  sorry

end NUMINAMATH_GPT_painter_total_cost_l1150_115070


namespace NUMINAMATH_GPT_apples_final_count_l1150_115033

theorem apples_final_count :
  let initial_apples := 200
  let shared_apples := 5
  let remaining_after_share := initial_apples - shared_apples
  let sister_takes := remaining_after_share / 2
  let half_rounded_down := 97 -- explicitly rounding down since 195 cannot be split exactly
  let remaining_after_sister := remaining_after_share - half_rounded_down
  let received_gift := 7
  let final_count := remaining_after_sister + received_gift
  final_count = 105 :=
by
  sorry

end NUMINAMATH_GPT_apples_final_count_l1150_115033


namespace NUMINAMATH_GPT_three_digit_integers_product_30_l1150_115087

theorem three_digit_integers_product_30 : 
  ∃ (n : ℕ), 
    (100 ≤ n ∧ n < 1000) ∧ 
    (∀ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 → 
    (1 ≤ d1 ∧ d1 ≤ 9) ∧ 
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    d1 * d2 * d3 = 30) ∧ 
    n = 12 :=
sorry

end NUMINAMATH_GPT_three_digit_integers_product_30_l1150_115087


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1150_115066

theorem arithmetic_sequence_15th_term :
  let a1 := 3
  let d := 7
  let n := 15
  a1 + (n - 1) * d = 101 :=
by
  let a1 := 3
  let d := 7
  let n := 15
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1150_115066


namespace NUMINAMATH_GPT_prob1_prob2_prob3_l1150_115028

-- Problem (1)
theorem prob1 (a b : ℝ) :
  ((a / 4 - 1) + 2 * (b / 3 + 2) = 4) ∧ (2 * (a / 4 - 1) + (b / 3 + 2) = 5) →
  a = 12 ∧ b = -3 :=
by { sorry }

-- Problem (2)
theorem prob2 (m n x y a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  (x = 10) ∧ (y = 6) ∧ 
  (5 * a₁ * (m - 3) + 3 * b₁ * (n + 2) = c₁) ∧ (5 * a₂ * (m - 3) + 3 * b₂ * (n + 2) = c₂) →
  (m = 5) ∧ (n = 0) :=
by { sorry }

-- Problem (3)
theorem prob3 (x y z : ℝ) :
  (3 * x - 2 * z + 12 * y = 47) ∧ (2 * x + z + 8 * y = 36) → z = 2 :=
by { sorry }

end NUMINAMATH_GPT_prob1_prob2_prob3_l1150_115028


namespace NUMINAMATH_GPT_integer_solution_exists_l1150_115067

theorem integer_solution_exists : ∃ n : ℤ, (⌊(n^2 : ℚ) / 3⌋ - ⌊(n : ℚ) / 2⌋^2 = 3) ∧ n = 6 := by
  sorry

end NUMINAMATH_GPT_integer_solution_exists_l1150_115067


namespace NUMINAMATH_GPT_kanul_spent_on_machinery_l1150_115086

theorem kanul_spent_on_machinery (total raw_materials cash M : ℝ) 
  (h_total : total = 7428.57) 
  (h_raw_materials : raw_materials = 5000) 
  (h_cash : cash = 0.30 * total) 
  (h_expenditure : total = raw_materials + M + cash) :
  M = 200 := 
by
  sorry

end NUMINAMATH_GPT_kanul_spent_on_machinery_l1150_115086


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1150_115025

theorem solution_set_of_inequality :
  { x : ℝ | (2 * x - 1) / (x + 1) ≤ 1 } = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1150_115025


namespace NUMINAMATH_GPT_intersection_M_N_l1150_115039

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1150_115039


namespace NUMINAMATH_GPT_sum_of_coefficients_l1150_115029

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℝ) 
  (h1 : quadratic a b c 3 = 0) 
  (h2 : quadratic a b c 7 = 0)
  (h3 : ∃ x0, (∀ x, quadratic a b c x ≥ quadratic a b c x0) ∧ quadratic a b c x0 = 20) :
  a + b + c = -105 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1150_115029


namespace NUMINAMATH_GPT_binary_representation_of_fourteen_l1150_115075

theorem binary_representation_of_fourteen :
  (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by
  sorry

end NUMINAMATH_GPT_binary_representation_of_fourteen_l1150_115075


namespace NUMINAMATH_GPT_parabola_find_c_l1150_115053

theorem parabola_find_c (b c : ℝ) 
  (h1 : (1 : ℝ)^2 + b * 1 + c = 2)
  (h2 : (5 : ℝ)^2 + b * 5 + c = 2) : 
  c = 7 := by
  sorry

end NUMINAMATH_GPT_parabola_find_c_l1150_115053


namespace NUMINAMATH_GPT_weight_of_a_l1150_115040

variables (a b c d e : ℝ)

theorem weight_of_a (h1 : (a + b + c) / 3 = 80)
                    (h2 : (a + b + c + d) / 4 = 82)
                    (h3 : e = d + 3)
                    (h4 : (b + c + d + e) / 4 = 81) :
  a = 95 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_a_l1150_115040


namespace NUMINAMATH_GPT_set_intersection_complement_l1150_115022

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {1, 3, 4, 6, 7}

theorem set_intersection_complement :
  A ∩ (U \ B) = {2, 5} := 
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1150_115022


namespace NUMINAMATH_GPT_circle_area_l1150_115050

-- Given conditions
variables {BD AC : ℝ} (BD_pos : BD = 6) (AC_pos : AC = 12)
variables {R : ℝ} (R_pos : R = 15 / 2)

-- Prove that the area of the circles is \(\frac{225}{4}\pi\)
theorem circle_area (BD_pos : BD = 6) (AC_pos : AC = 12) (R : ℝ) (R_pos : R = 15 / 2) : 
        ∃ S, S = (225 / 4) * Real.pi := 
by sorry

end NUMINAMATH_GPT_circle_area_l1150_115050


namespace NUMINAMATH_GPT_number_of_grey_birds_l1150_115058

variable (G : ℕ)

def grey_birds_condition1 := G + 6
def grey_birds_condition2 := G / 2

theorem number_of_grey_birds
  (H1 : G + 6 + G / 2 = 66) :
  G = 40 :=
by
  sorry

end NUMINAMATH_GPT_number_of_grey_birds_l1150_115058


namespace NUMINAMATH_GPT_star_computation_l1150_115004

-- Define the operation ☆
def star (m n : Int) := m^2 - m * n + n

-- Define the main proof problem
theorem star_computation :
  star 3 4 = 1 ∧ star (-1) (star 2 (-3)) = 15 := 
by
  sorry

end NUMINAMATH_GPT_star_computation_l1150_115004


namespace NUMINAMATH_GPT_asymptotes_tangent_to_circle_l1150_115006

theorem asymptotes_tangent_to_circle {m : ℝ} (hm : m > 0) 
  (hyp_eq : ∀ x y : ℝ, y^2 - (x^2 / m^2) = 1) 
  (circ_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * y + 3 = 0) : 
  m = (Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_asymptotes_tangent_to_circle_l1150_115006


namespace NUMINAMATH_GPT_rectangle_perimeter_gt_16_l1150_115056

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_area_gt_perim : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_gt_16_l1150_115056


namespace NUMINAMATH_GPT_regular_pay_correct_l1150_115074

noncomputable def regular_pay_per_hour (total_payment : ℝ) (regular_hours : ℕ) (overtime_hours : ℕ) (overtime_rate : ℝ) : ℝ :=
  let R := total_payment / (regular_hours + overtime_rate * overtime_hours)
  R

theorem regular_pay_correct :
  regular_pay_per_hour 198 40 13 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_regular_pay_correct_l1150_115074


namespace NUMINAMATH_GPT_man_half_father_age_in_years_l1150_115052

theorem man_half_father_age_in_years
  (M F Y : ℕ) 
  (h1: M = (2 * F) / 5) 
  (h2: F = 25) 
  (h3: M + Y = (F + Y) / 2) : 
  Y = 5 := by 
  sorry

end NUMINAMATH_GPT_man_half_father_age_in_years_l1150_115052


namespace NUMINAMATH_GPT_three_inequalities_true_l1150_115034

variables {x y a b : ℝ}
-- Declare the conditions as hypotheses
axiom h₁ : 0 < x
axiom h₂ : 0 < y
axiom h₃ : 0 < a
axiom h₄ : 0 < b
axiom hx : x^2 < a^2
axiom hy : y^2 < b^2

theorem three_inequalities_true : 
  (x^2 + y^2 < a^2 + b^2) ∧ 
  (x^2 * y^2 < a^2 * b^2) ∧ 
  (x^2 / y^2 < a^2 / b^2) :=
sorry

end NUMINAMATH_GPT_three_inequalities_true_l1150_115034


namespace NUMINAMATH_GPT_initial_principal_amount_l1150_115030

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_principal_amount :
  let P := 4410 / (compound_interest 1 0.07 4 2 * compound_interest 1 0.09 2 2)
  abs (P - 3238.78) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_initial_principal_amount_l1150_115030


namespace NUMINAMATH_GPT_inequality_of_pos_reals_l1150_115043

open Real

theorem inequality_of_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤
  (1 / 4) * (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_pos_reals_l1150_115043


namespace NUMINAMATH_GPT_initial_population_l1150_115079

variable (P : ℕ)

theorem initial_population
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℚ := 1.2) :
  (P = 3000) :=
by
  sorry

end NUMINAMATH_GPT_initial_population_l1150_115079


namespace NUMINAMATH_GPT_smallest_x_for_max_f_l1150_115003

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 12)

theorem smallest_x_for_max_f : ∃ x > 0, f x = 2 ∧ ∀ y > 0, (f y = 2 → y ≥ x) :=
sorry

end NUMINAMATH_GPT_smallest_x_for_max_f_l1150_115003


namespace NUMINAMATH_GPT_range_of_m_l1150_115062

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) m, 1 ≤ f x ∧ f x ≤ 10) ↔ 2 ≤ m ∧ m ≤ 5 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1150_115062


namespace NUMINAMATH_GPT_simplify_expression_l1150_115071

theorem simplify_expression (y : ℝ) : 
  (3 * y) ^ 3 - 2 * y * y ^ 2 + y ^ 4 = 25 * y ^ 3 + y ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1150_115071


namespace NUMINAMATH_GPT_part_a_part_b_l1150_115026

-- Define n_mid_condition
def n_mid_condition (n : ℕ) : Prop := n % 2 = 1 ∧ n ∣ 2023^n - 1

-- Part a:
theorem part_a : ∃ (k₁ k₂ : ℕ), k₁ = 3 ∧ k₂ = 9 ∧ n_mid_condition k₁ ∧ n_mid_condition k₂ := by
  sorry

-- Part b:
theorem part_b : ∀ k, k ≥ 1 → n_mid_condition (3^k) := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1150_115026


namespace NUMINAMATH_GPT_problem_statement_problem_statement_2_l1150_115045

noncomputable def A (m : ℝ) : Set ℝ := {x | x > 2^m}
noncomputable def B : Set ℝ := {x | -4 < x - 4 ∧ x - 4 < 4}

theorem problem_statement (m : ℝ) (h1 : m = 2) :
  (A m ∪ B = {x | x > 0}) ∧ (A m ∩ B = {x | 4 < x ∧ x < 8}) :=
by sorry

theorem problem_statement_2 (m : ℝ) (h2 : A m ⊆ {x | x ≤ 0 ∨ 8 ≤ x}) :
  3 ≤ m :=
by sorry

end NUMINAMATH_GPT_problem_statement_problem_statement_2_l1150_115045


namespace NUMINAMATH_GPT_domain_of_f_log2x_is_0_4_l1150_115076

def f : ℝ → ℝ := sorry

-- Given condition: domain of y = f(2x) is (-1, 1)
def dom_f_2x (x : ℝ) : Prop := -1 < 2 * x ∧ 2 * x < 1

-- Conclusion: domain of y = f(log_2 x) is (0, 4)
def dom_f_log2x (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem domain_of_f_log2x_is_0_4 (x : ℝ) :
  (dom_f_2x x) → (dom_f_log2x x) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_log2x_is_0_4_l1150_115076


namespace NUMINAMATH_GPT_calculate_value_l1150_115011

theorem calculate_value : (2200 - 2090)^2 / (144 + 25) = 64 := 
by
  sorry

end NUMINAMATH_GPT_calculate_value_l1150_115011


namespace NUMINAMATH_GPT_tan_A_plus_C_eq_neg_sqrt3_l1150_115090

theorem tan_A_plus_C_eq_neg_sqrt3
  (A B C : Real)
  (hSum : A + B + C = Real.pi)
  (hArithSeq : 2 * B = A + C)
  (hTriangle : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_A_plus_C_eq_neg_sqrt3_l1150_115090
