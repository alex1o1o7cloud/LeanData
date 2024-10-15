import Mathlib

namespace NUMINAMATH_GPT_proof_problem_l467_46767

open Real

def p : Prop := ∀ a : ℝ, a^2017 > -1 → a > -1
def q : Prop := ∀ x : ℝ, x^2 * tan (x^2) > 0

theorem proof_problem : p ∨ q :=
sorry

end NUMINAMATH_GPT_proof_problem_l467_46767


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l467_46786

def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : is_isosceles a b c) (h2 : is_triangle a b c) (h3 : a = 4 ∨ a = 9) (h4 : b = 4 ∨ b = 9) :
  perimeter a b c = 22 :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l467_46786


namespace NUMINAMATH_GPT_sequence_term_1000_l467_46705

open Nat

theorem sequence_term_1000 :
  (∃ b : ℕ → ℤ,
    b 1 = 3010 ∧
    b 2 = 3011 ∧
    (∀ n, 1 ≤ n → b n + b (n + 1) + b (n + 2) = n + 4) ∧
    b 1000 = 3343) :=
sorry

end NUMINAMATH_GPT_sequence_term_1000_l467_46705


namespace NUMINAMATH_GPT_prob_of_B1_selected_prob_of_D1_in_team_l467_46759

noncomputable def total_teams : ℕ := 20

noncomputable def teams_with_B1 : ℕ := 8

noncomputable def teams_with_D1 : ℕ := 12

theorem prob_of_B1_selected : (teams_with_B1 : ℚ) / total_teams = 2 / 5 := by
  sorry

theorem prob_of_D1_in_team : (teams_with_D1 : ℚ) / total_teams = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_prob_of_B1_selected_prob_of_D1_in_team_l467_46759


namespace NUMINAMATH_GPT_prime_exponent_condition_l467_46799

theorem prime_exponent_condition (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n)
  (h : 2^p + 3^p = a^n) : n = 1 :=
sorry

end NUMINAMATH_GPT_prime_exponent_condition_l467_46799


namespace NUMINAMATH_GPT_janet_spends_more_on_piano_l467_46714

-- Condition definitions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℝ := 52

-- Calculations based on conditions
def weekly_cost_clarinet : ℝ := clarinet_hourly_rate * clarinet_hours_per_week
def weekly_cost_piano : ℝ := piano_hourly_rate * piano_hours_per_week
def weekly_difference : ℝ := weekly_cost_piano - weekly_cost_clarinet
def yearly_difference : ℝ := weekly_difference * weeks_per_year

theorem janet_spends_more_on_piano : yearly_difference = 1040 := by
  sorry 

end NUMINAMATH_GPT_janet_spends_more_on_piano_l467_46714


namespace NUMINAMATH_GPT_value_of_x2_minus_y2_l467_46742

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 9 / 17) (h2 : x - y = 1 / 19) : x^2 - y^2 = 9 / 323 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_value_of_x2_minus_y2_l467_46742


namespace NUMINAMATH_GPT_sum_of_remainders_mod_13_l467_46746

theorem sum_of_remainders_mod_13 :
  ∀ (a b c d e : ℤ),
    a ≡ 3 [ZMOD 13] →
    b ≡ 5 [ZMOD 13] →
    c ≡ 7 [ZMOD 13] →
    d ≡ 9 [ZMOD 13] →
    e ≡ 11 [ZMOD 13] →
    (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_13_l467_46746


namespace NUMINAMATH_GPT_kicks_before_break_l467_46769

def total_kicks : ℕ := 98
def kicks_after_break : ℕ := 36
def kicks_needed_to_goal : ℕ := 19

theorem kicks_before_break :
  total_kicks - (kicks_after_break + kicks_needed_to_goal) = 43 := 
by
  -- proof wanted
  sorry

end NUMINAMATH_GPT_kicks_before_break_l467_46769


namespace NUMINAMATH_GPT_number_of_persons_l467_46762

theorem number_of_persons
    (total_amount : ℕ) 
    (amount_per_person : ℕ) 
    (h1 : total_amount = 42900) 
    (h2 : amount_per_person = 1950) :
    total_amount / amount_per_person = 22 :=
by
  sorry

end NUMINAMATH_GPT_number_of_persons_l467_46762


namespace NUMINAMATH_GPT_value_of_M_l467_46756

theorem value_of_M (M : ℝ) (h : 0.2 * M = 500) : M = 2500 :=
by
  sorry

end NUMINAMATH_GPT_value_of_M_l467_46756


namespace NUMINAMATH_GPT_gold_copper_ratio_l467_46797

theorem gold_copper_ratio (G C : ℕ) (h : 19 * G + 9 * C = 17 * (G + C)) : G = 4 * C :=
by
  sorry

end NUMINAMATH_GPT_gold_copper_ratio_l467_46797


namespace NUMINAMATH_GPT_parallelogram_area_l467_46749

theorem parallelogram_area (base height : ℝ) (h_base : base = 14) (h_height : height = 24) :
  base * height = 336 :=
by 
  rw [h_base, h_height]
  sorry

end NUMINAMATH_GPT_parallelogram_area_l467_46749


namespace NUMINAMATH_GPT_number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l467_46703

-- Definition for the number of sides in the polygon
def n : ℕ := 150

-- Definition of the formula for the number of diagonals
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the formula for the sum of interior angles
def sum_of_interior_angles (n : ℕ) : ℕ :=
  180 * (n - 2)

-- Theorem statements to be proved
theorem number_of_diagonals_is_correct : number_of_diagonals n = 11025 := sorry

theorem sum_of_interior_angles_is_correct : sum_of_interior_angles n = 26640 := sorry

end NUMINAMATH_GPT_number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l467_46703


namespace NUMINAMATH_GPT_max_three_cell_corners_l467_46795

-- Define the grid size
def grid_height : ℕ := 7
def grid_width : ℕ := 14

-- Define the concept of a three-cell corner removal
def three_cell_corner (region : ℕ) : ℕ := region / 3

-- Define the problem statement in Lean
theorem max_three_cell_corners : three_cell_corner (grid_height * grid_width) = 32 := by
  sorry

end NUMINAMATH_GPT_max_three_cell_corners_l467_46795


namespace NUMINAMATH_GPT_speed_of_truck_l467_46700

theorem speed_of_truck
  (v : ℝ)                         -- Let \( v \) be the speed of the truck.
  (car_speed : ℝ := 55)           -- Car speed is 55 mph.
  (start_delay : ℝ := 1)          -- Truck starts 1 hour later.
  (catchup_time : ℝ := 6.5)       -- Truck takes 6.5 hours to pass the car.
  (additional_distance_car : ℝ := car_speed * catchup_time)  -- Additional distance covered by the car in 6.5 hours.
  (total_distance_truck : ℝ := car_speed * start_delay + additional_distance_car)  -- Total distance truck must cover to pass the car.
  (truck_distance_eq : v * catchup_time = total_distance_truck)  -- Distance equation for the truck.
  : v = 63.46 :=                -- Prove the truck's speed is 63.46 mph.
by
  -- Original problem solution confirms truck's speed as 63.46 mph. 
  sorry

end NUMINAMATH_GPT_speed_of_truck_l467_46700


namespace NUMINAMATH_GPT_simplify_fraction_l467_46707

theorem simplify_fraction (m : ℝ) (h₁: m ≠ 0) (h₂: m ≠ 1): (m - 1) / m / ((m - 1) / (m * m)) = m := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l467_46707


namespace NUMINAMATH_GPT_triangle_area_l467_46796

theorem triangle_area (a b c : ℝ) (h1: a = 15) (h2: c = 17) (h3: a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l467_46796


namespace NUMINAMATH_GPT_number_is_12_l467_46754

theorem number_is_12 (x : ℝ) (h : 4 * x - 3 = 9 * (x - 7)) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_is_12_l467_46754


namespace NUMINAMATH_GPT_Cara_possible_pairs_l467_46735

-- Define the conditions and the final goal.
theorem Cara_possible_pairs : ∃ p : Nat, p = Nat.choose 7 2 ∧ p = 21 :=
by
  sorry

end NUMINAMATH_GPT_Cara_possible_pairs_l467_46735


namespace NUMINAMATH_GPT_simplify_exponents_l467_46772

variable (x : ℝ)

theorem simplify_exponents (x : ℝ) : (x^5) * (x^2) = x^(7) :=
by
  sorry

end NUMINAMATH_GPT_simplify_exponents_l467_46772


namespace NUMINAMATH_GPT_find_integer_n_l467_46781

theorem find_integer_n : ∃ (n : ℤ), (-90 ≤ n ∧ n ≤ 90) ∧ (Real.sin (n * Real.pi / 180) = Real.cos (456 * Real.pi / 180)) ∧ n = -6 := 
by
  sorry

end NUMINAMATH_GPT_find_integer_n_l467_46781


namespace NUMINAMATH_GPT_total_action_figures_l467_46794

-- Definitions based on conditions
def initial_figures : ℕ := 8
def figures_per_set : ℕ := 5
def added_sets : ℕ := 2
def total_added_figures : ℕ := added_sets * figures_per_set
def total_figures : ℕ := initial_figures + total_added_figures

-- Theorem statement with conditions and expected result
theorem total_action_figures : total_figures = 18 := by
  sorry

end NUMINAMATH_GPT_total_action_figures_l467_46794


namespace NUMINAMATH_GPT_even_function_f_l467_46784

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2^x - 1 else sorry

theorem even_function_f (h_even : ∀ x : ℝ, f x = f (-x)) : f 1 = -1 / 2 := by
  -- proof development skipped
  sorry

end NUMINAMATH_GPT_even_function_f_l467_46784


namespace NUMINAMATH_GPT_trigonometric_identity_l467_46720

theorem trigonometric_identity (α : ℝ) (h : Real.sin (3 * Real.pi - α) = 2 * Real.sin (Real.pi / 2 + α)) : 
  (Real.sin (Real.pi - α) ^ 3 - Real.sin (Real.pi / 2 - α)) / 
  (3 * Real.cos (Real.pi / 2 + α) + 2 * Real.cos (Real.pi + α)) = -3/40 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l467_46720


namespace NUMINAMATH_GPT_fraction_meaningful_l467_46715

theorem fraction_meaningful (a : ℝ) : (∃ x, x = 2 / (a + 1)) ↔ a ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l467_46715


namespace NUMINAMATH_GPT_min_cost_29_disks_l467_46788

theorem min_cost_29_disks
  (price_single : ℕ := 20) 
  (price_pack_10 : ℕ := 111) 
  (price_pack_25 : ℕ := 265) :
  ∃ cost : ℕ, cost ≥ (price_pack_10 + price_pack_10 + price_pack_10) 
              ∧ cost ≤ (price_pack_25 + price_single * 4) 
              ∧ cost = 333 := 
by
  sorry

end NUMINAMATH_GPT_min_cost_29_disks_l467_46788


namespace NUMINAMATH_GPT_find_c_l467_46719

def conditions (c d : ℝ) : Prop :=
  -- The polynomial 6x^3 + 7cx^2 + 3dx + 2c = 0 has three distinct positive roots
  ∃ u v w : ℝ, 0 < u ∧ 0 < v ∧ 0 < w ∧ u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
  (6 * u^3 + 7 * c * u^2 + 3 * d * u + 2 * c = 0) ∧
  (6 * v^3 + 7 * c * v^2 + 3 * d * v + 2 * c = 0) ∧
  (6 * w^3 + 7 * c * w^2 + 3 * d * w + 2 * c = 0) ∧
  -- Sum of the base-2 logarithms of the roots is 6
  Real.log (u * v * w) / Real.log 2 = 6

theorem find_c (c d : ℝ) (h : conditions c d) : c = -192 :=
sorry

end NUMINAMATH_GPT_find_c_l467_46719


namespace NUMINAMATH_GPT_desired_interest_rate_l467_46732

def face_value : Real := 52
def dividend_rate : Real := 0.09
def market_value : Real := 39

theorem desired_interest_rate : (dividend_rate * face_value / market_value) * 100 = 12 := by
  sorry

end NUMINAMATH_GPT_desired_interest_rate_l467_46732


namespace NUMINAMATH_GPT_sum_of_ages_l467_46701

theorem sum_of_ages (a b c : ℕ) (h₁ : a = 20 + b + c) (h₂ : a^2 = 2050 + (b + c)^2) : a + b + c = 80 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l467_46701


namespace NUMINAMATH_GPT_marble_problem_l467_46729

def total_marbles_originally 
  (white_marbles : ℕ := 20) 
  (blue_marbles : ℕ) 
  (red_marbles : ℕ := blue_marbles) 
  (total_left : ℕ := 40)
  (jack_removes : ℕ := 2 * (white_marbles - blue_marbles)) : ℕ :=
  white_marbles + blue_marbles + red_marbles

theorem marble_problem : 
  ∀ (white_marbles : ℕ := 20) 
    (blue_marbles red_marbles : ℕ) 
    (jack_removes total_left : ℕ),
    red_marbles = blue_marbles →
    jack_removes = 2 * (white_marbles - blue_marbles) →
    total_left = total_marbles_originally white_marbles blue_marbles red_marbles - jack_removes →
    total_left = 40 →
    total_marbles_originally white_marbles blue_marbles red_marbles = 50 :=
by
  intros white_marbles blue_marbles red_marbles jack_removes total_left h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_marble_problem_l467_46729


namespace NUMINAMATH_GPT_evaluate_f_at_2_l467_46706

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem evaluate_f_at_2 :
  f 2 = 125 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_2_l467_46706


namespace NUMINAMATH_GPT_stickers_difference_l467_46785

theorem stickers_difference (X : ℕ) :
  let Cindy_initial := X
  let Dan_initial := X
  let Cindy_after := Cindy_initial - 15
  let Dan_after := Dan_initial + 18
  Dan_after - Cindy_after = 33 := by
  sorry

end NUMINAMATH_GPT_stickers_difference_l467_46785


namespace NUMINAMATH_GPT_find_polynomials_l467_46777

-- Define our polynomial P(x)
def polynomial_condition (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, (x-1) * P.eval (x+1) - (x+2) * P.eval x = 0

-- State the theorem
theorem find_polynomials (P : Polynomial ℝ) :
  polynomial_condition P ↔ ∃ a : ℝ, P = Polynomial.C a * (Polynomial.X^3 - Polynomial.X) :=
by
  sorry

end NUMINAMATH_GPT_find_polynomials_l467_46777


namespace NUMINAMATH_GPT_product_of_two_numbers_l467_46718

theorem product_of_two_numbers : 
  ∀ (x y : ℝ), (x + y = 60) ∧ (x - y = 10) → x * y = 875 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l467_46718


namespace NUMINAMATH_GPT_sum_of_angles_l467_46770

theorem sum_of_angles (A B C x y : ℝ) 
  (hA : A = 34) 
  (hB : B = 80) 
  (hC : C = 30)
  (pentagon_angles_sum : A + B + (360 - x) + 90 + (120 - y) = 540) : 
  x + y = 144 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_angles_l467_46770


namespace NUMINAMATH_GPT_volume_of_pyramid_in_cube_l467_46775

structure Cube :=
(side_length : ℝ)

noncomputable def base_triangle_area (side_length : ℝ) : ℝ :=
(1/2) * side_length * side_length

noncomputable def pyramid_volume (triangle_area : ℝ) (height : ℝ) : ℝ :=
(1/3) * triangle_area * height

theorem volume_of_pyramid_in_cube (c : Cube) (h : c.side_length = 2) : 
  pyramid_volume (base_triangle_area c.side_length) c.side_length = 4/3 :=
by {
  sorry
}

end NUMINAMATH_GPT_volume_of_pyramid_in_cube_l467_46775


namespace NUMINAMATH_GPT_value_of_y_plus_10_l467_46724

theorem value_of_y_plus_10 (x y : ℝ) (h1 : 3 * x = (3 / 4) * y) (h2 : x = 20) : y + 10 = 90 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_plus_10_l467_46724


namespace NUMINAMATH_GPT_hyperbola_range_k_l467_46740

theorem hyperbola_range_k (k : ℝ) : (4 + k) * (1 - k) < 0 ↔ k ∈ (Set.Iio (-4) ∪ Set.Ioi 1) := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_range_k_l467_46740


namespace NUMINAMATH_GPT_sum_of_coefficients_l467_46725

-- Given polynomial
def polynomial (x : ℝ) : ℝ := (3 * x - 1) ^ 7

-- Statement
theorem sum_of_coefficients :
  (polynomial 1) = 128 := 
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l467_46725


namespace NUMINAMATH_GPT_horizontal_asymptote_is_3_l467_46708

-- Definitions of the polynomials
noncomputable def p (x : ℝ) : ℝ := 15 * x^5 + 10 * x^4 + 5 * x^3 + 7 * x^2 + 6 * x + 2
noncomputable def q (x : ℝ) : ℝ := 5 * x^5 + 3 * x^4 + 9 * x^3 + 4 * x^2 + 2 * x + 1

-- Statement that we need to prove
theorem horizontal_asymptote_is_3 : 
  (∃ (y : ℝ), (∀ x : ℝ, x ≠ 0 → (p x / q x) = y) ∧ y = 3) :=
  sorry -- The proof is left as an exercise.

end NUMINAMATH_GPT_horizontal_asymptote_is_3_l467_46708


namespace NUMINAMATH_GPT_logarithm_simplification_l467_46728

theorem logarithm_simplification :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 7 / Real.log 9 + 1)) =
  1 - (Real.log 7 / Real.log 1008) :=
sorry

end NUMINAMATH_GPT_logarithm_simplification_l467_46728


namespace NUMINAMATH_GPT_greatest_possible_value_of_x_l467_46723

theorem greatest_possible_value_of_x (x : ℝ) (h : ( (5 * x - 25) / (4 * x - 5) ) ^ 3 + ( (5 * x - 25) / (4 * x - 5) ) = 16):
  x = 5 :=
sorry

end NUMINAMATH_GPT_greatest_possible_value_of_x_l467_46723


namespace NUMINAMATH_GPT_first_year_after_2022_with_digit_sum_5_l467_46783

def sum_of_digits (n : ℕ) : ℕ :=
  (toString n).foldl (λ acc c => acc + c.toNat - '0'.toNat) 0

theorem first_year_after_2022_with_digit_sum_5 :
  ∃ y : ℕ, y > 2022 ∧ sum_of_digits y = 5 ∧ ∀ z : ℕ, z > 2022 ∧ z < y → sum_of_digits z ≠ 5 :=
sorry

end NUMINAMATH_GPT_first_year_after_2022_with_digit_sum_5_l467_46783


namespace NUMINAMATH_GPT_parallel_line_through_P_perpendicular_line_through_P_l467_46709

-- Define the line equations
def line1 (x y : ℝ) : Prop := 2 * x + y - 5 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y = 0
def line_l (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Define the equations for parallel and perpendicular lines through point P
def parallel_line (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 5 = 0

-- Define the point P where the lines intersect
def point_P : (ℝ × ℝ) := (2, 1)

-- Assert the proof statements
theorem parallel_line_through_P : parallel_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry
  
theorem perpendicular_line_through_P : perpendicular_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry

end NUMINAMATH_GPT_parallel_line_through_P_perpendicular_line_through_P_l467_46709


namespace NUMINAMATH_GPT_represent_in_scientific_notation_l467_46751

def million : ℕ := 10^6
def rural_residents : ℝ := 42.39 * million

theorem represent_in_scientific_notation :
  42.39 * 10^6 = 4.239 * 10^7 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_represent_in_scientific_notation_l467_46751


namespace NUMINAMATH_GPT_planet_not_observed_l467_46734

theorem planet_not_observed (k : ℕ) (d : Fin (2*k+1) → Fin (2*k+1) → ℝ) 
  (h_d : ∀ i j : Fin (2*k+1), i ≠ j → d i i = 0 ∧ d i j ≠ d i i) 
  (h_astronomer : ∀ i : Fin (2*k+1), ∃ j : Fin (2*k+1), j ≠ i ∧ ∀ k : Fin (2*k+1), k ≠ i → d i j < d i k) : 
  ∃ i : Fin (2*k+1), ∀ j : Fin (2*k+1), i ≠ j → ∃ l : Fin (2*k+1), (j ≠ l ∧ d l i < d l j) → false :=
  sorry

end NUMINAMATH_GPT_planet_not_observed_l467_46734


namespace NUMINAMATH_GPT_real_return_l467_46790

theorem real_return (n i r: ℝ) (h₁ : n = 0.21) (h₂ : i = 0.10) : 
  (1 + r) = (1 + n) / (1 + i) → r = 0.10 :=
by
  intro h₃
  sorry

end NUMINAMATH_GPT_real_return_l467_46790


namespace NUMINAMATH_GPT_tina_wins_before_first_loss_l467_46752

-- Definitions based on conditions
variable (W : ℕ) -- The number of wins before Tina's first loss

-- Conditions
def win_before_first_loss : W = 10 := by sorry

def total_wins (W : ℕ) := W + 2 * W -- After her first loss, she doubles her wins and loses again
def total_losses : ℕ := 2 -- She loses twice

def career_record_condition (W : ℕ) : Prop := total_wins W - total_losses = 28

-- Proof Problem (Statement)
theorem tina_wins_before_first_loss : career_record_condition W → W = 10 :=
by sorry

end NUMINAMATH_GPT_tina_wins_before_first_loss_l467_46752


namespace NUMINAMATH_GPT_triangle_inequality_internal_point_l467_46787

theorem triangle_inequality_internal_point {A B C P : Type} 
  (x y z p q r : ℝ) 
  (h_distances_from_vertices : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distances_from_sides : p > 0 ∧ q > 0 ∧ r > 0)
  (h_x_y_z_triangle_ineq : x + y > z ∧ y + z > x ∧ z + x > y)
  (h_p_q_r_triangle_ineq : p + q > r ∧ q + r > p ∧ r + p > q) :
  x * y * z ≥ (q + r) * (r + p) * (p + q) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_internal_point_l467_46787


namespace NUMINAMATH_GPT_selling_price_l467_46710

-- Definitions for conditions
variables (CP SP_loss SP_profit : ℝ)
variable (h1 : SP_loss = 0.8 * CP)
variable (h2 : SP_profit = 1.05 * CP)
variable (h3 : SP_profit = 11.8125)

-- Theorem statement to prove
theorem selling_price (h1 : SP_loss = 0.8 * CP) (h2 : SP_profit = 1.05 * CP) (h3 : SP_profit = 11.8125) :
  SP_loss = 9 := 
sorry

end NUMINAMATH_GPT_selling_price_l467_46710


namespace NUMINAMATH_GPT_ryan_chinese_learning_hours_l467_46760

variable (hours_english : ℕ)
variable (days : ℕ)
variable (total_hours : ℕ)

theorem ryan_chinese_learning_hours (h1 : hours_english = 6) 
                                    (h2 : days = 5) 
                                    (h3 : total_hours = 65) : 
                                    total_hours - (hours_english * days) / days = 7 := by
  sorry

end NUMINAMATH_GPT_ryan_chinese_learning_hours_l467_46760


namespace NUMINAMATH_GPT_prescribedDosageLessThanTypical_l467_46722

noncomputable def prescribedDosage : ℝ := 12
noncomputable def bodyWeight : ℝ := 120
noncomputable def typicalDosagePer15Pounds : ℝ := 2
noncomputable def typicalDosage : ℝ := (bodyWeight / 15) * typicalDosagePer15Pounds
noncomputable def percentageDecrease : ℝ := ((typicalDosage - prescribedDosage) / typicalDosage) * 100

theorem prescribedDosageLessThanTypical :
  percentageDecrease = 25 :=
by
  sorry

end NUMINAMATH_GPT_prescribedDosageLessThanTypical_l467_46722


namespace NUMINAMATH_GPT_marie_age_l467_46755

theorem marie_age (L M O : ℕ) (h1 : L = 4 * M) (h2 : O = M + 8) (h3 : L = O) : M = 8 / 3 := by
  sorry

end NUMINAMATH_GPT_marie_age_l467_46755


namespace NUMINAMATH_GPT_orchids_initially_three_l467_46738

-- Define initial number of roses and provided number of orchids in the vase
def initial_roses : ℕ := 9
def added_orchids (O : ℕ) : ℕ := 13
def added_roses : ℕ := 3
def difference := 10

-- Define initial number of orchids that we need to prove
def initial_orchids (O : ℕ) : Prop :=
  added_orchids O - added_roses = difference →
  O = 3

theorem orchids_initially_three :
  initial_orchids O :=
sorry

end NUMINAMATH_GPT_orchids_initially_three_l467_46738


namespace NUMINAMATH_GPT_find_multiplier_l467_46789

theorem find_multiplier (A N : ℕ) (h : A = 32) (eqn : N * (A + 4) - 4 * (A - 4) = A) : N = 4 :=
sorry

end NUMINAMATH_GPT_find_multiplier_l467_46789


namespace NUMINAMATH_GPT_b_should_pay_l467_46721

def TotalRent : ℕ := 725
def Cost_a : ℕ := 12 * 8 * 5
def Cost_b : ℕ := 16 * 9 * 6
def Cost_c : ℕ := 18 * 6 * 7
def Cost_d : ℕ := 20 * 4 * 4
def TotalCost : ℕ := Cost_a + Cost_b + Cost_c + Cost_d
def Payment_b (Cost_b TotalCost TotalRent : ℕ) : ℕ := (Cost_b * TotalRent) / TotalCost

theorem b_should_pay :
  Payment_b Cost_b TotalCost TotalRent = 259 := 
  by
  unfold Payment_b
  -- Leaving the proof body empty as per instructions
  sorry

end NUMINAMATH_GPT_b_should_pay_l467_46721


namespace NUMINAMATH_GPT_exists_n_divides_2022n_minus_n_l467_46711

theorem exists_n_divides_2022n_minus_n (p : ℕ) [hp : Fact (Nat.Prime p)] :
  ∃ n : ℕ, p ∣ (2022^n - n) :=
sorry

end NUMINAMATH_GPT_exists_n_divides_2022n_minus_n_l467_46711


namespace NUMINAMATH_GPT_hotel_P_charge_less_than_G_l467_46758

open Real

variable (G R P : ℝ)

-- Given conditions
def charge_R_eq_2G : Prop := R = 2 * G
def charge_P_eq_R_minus_55percent : Prop := P = R - 0.55 * R

-- Goal: Prove the percentage by which P's charge is less than G's charge is 10%
theorem hotel_P_charge_less_than_G : charge_R_eq_2G G R → charge_P_eq_R_minus_55percent R P → P = 0.9 * G := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_hotel_P_charge_less_than_G_l467_46758


namespace NUMINAMATH_GPT_movie_friends_l467_46779

noncomputable def movie_only (M P G MP MG PG MPG : ℕ) : Prop :=
  let total_M := 20
  let total_P := 20
  let total_G := 5
  let total_students := 31
  (MP = 4) ∧ 
  (MG = 2) ∧ 
  (PG = 0) ∧ (MPG = 2) ∧ 
  (M + MP + MG + MPG = total_M) ∧ 
  (P + MP + PG + MPG = total_P) ∧ 
  (G + MG + PG + MPG = total_G) ∧ 
  (M + P + G + MP + MG + PG + MPG = total_students) ∧ 
  (M = 12)

theorem movie_friends (M P G MP MG PG MPG : ℕ) : movie_only M P G MP MG PG MPG := 
by 
  sorry

end NUMINAMATH_GPT_movie_friends_l467_46779


namespace NUMINAMATH_GPT_power_calculation_l467_46768

theorem power_calculation (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(3*m + 2*n) = 200 := by
  sorry

end NUMINAMATH_GPT_power_calculation_l467_46768


namespace NUMINAMATH_GPT_fraction_expression_l467_46712

theorem fraction_expression : (1 / 3) ^ 3 * (1 / 8) = 1 / 216 :=
by
  sorry

end NUMINAMATH_GPT_fraction_expression_l467_46712


namespace NUMINAMATH_GPT_equilateral_triangle_area_l467_46766

theorem equilateral_triangle_area (A B C P : ℝ × ℝ)
  (hABC : ∃ a b c : ℝ, a = b ∧ b = c ∧ a = dist A B ∧ b = dist B C ∧ c = dist C A)
  (hPA : dist P A = 10)
  (hPB : dist P B = 8)
  (hPC : dist P C = 12) :
  ∃ (area : ℝ), area = 104 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_l467_46766


namespace NUMINAMATH_GPT_find_n_l467_46743

open Nat

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given condition for the proof
def condition (n : ℕ) : Prop := binom (n + 1) 7 - binom n 7 = binom n 8

-- The statement to prove
theorem find_n (n : ℕ) (h : condition n) : n = 14 :=
sorry

end NUMINAMATH_GPT_find_n_l467_46743


namespace NUMINAMATH_GPT_min_value_of_expression_l467_46713

theorem min_value_of_expression 
  (x y : ℝ) 
  (h : 3 * |x - y| + |2 * x - 5| = x + 1) : 
  ∃ (x y : ℝ), 2 * x + y = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_of_expression_l467_46713


namespace NUMINAMATH_GPT_inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l467_46773

noncomputable def inverse_of_half_pow (x : ℝ) : ℝ := Real.log x / Real.log (1 / 2)

theorem inverse_function_of_1_div_2_pow_eq_log_base_1_div_2 (x : ℝ) (hx : 0 < x) :
  inverse_of_half_pow x = Real.log x / Real.log (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l467_46773


namespace NUMINAMATH_GPT_find_fraction_value_l467_46776

variable (a b : ℝ)
variable (h1 : b > a)
variable (h2 : a > 0)
variable (h3 : a / b + b / a = 4)

theorem find_fraction_value (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a / b + b / a = 4) : (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_fraction_value_l467_46776


namespace NUMINAMATH_GPT_pills_needed_for_week_l467_46763

def pill_mg : ℕ := 50 -- Each pill has 50 mg of Vitamin A.
def recommended_daily_mg : ℕ := 200 -- The recommended daily serving of Vitamin A is 200 mg.
def days_in_week : ℕ := 7 -- There are 7 days in a week.

theorem pills_needed_for_week : (recommended_daily_mg / pill_mg) * days_in_week = 28 := 
by 
  sorry

end NUMINAMATH_GPT_pills_needed_for_week_l467_46763


namespace NUMINAMATH_GPT_max_neg_integers_l467_46704

theorem max_neg_integers (
  a b c d e f g h : ℤ
) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_e : e ≠ 0)
  (h_ineq : (a * b^2 + c * d * e^3) * (f * g^2 * h + f^3 - g^2) < 0)
  (h_abs : |d| < |f| ∧ |f| < |h|)
  : ∃ s, s = 5 ∧ ∀ (neg_count : ℕ), neg_count ≤ s := 
sorry

end NUMINAMATH_GPT_max_neg_integers_l467_46704


namespace NUMINAMATH_GPT_weights_divide_three_piles_l467_46736

theorem weights_divide_three_piles (n : ℕ) (h : n > 3) :
  (∃ (k : ℕ), n = 3 * k ∨ n = 3 * k + 2) ↔
  (∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
   A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
   A.sum id = (n * (n + 1)) / 6 ∧ B.sum id = (n * (n + 1)) / 6 ∧ C.sum id = (n * (n + 1)) / 6) :=
sorry

end NUMINAMATH_GPT_weights_divide_three_piles_l467_46736


namespace NUMINAMATH_GPT_sticker_distribution_probability_l467_46764

theorem sticker_distribution_probability :
  let p := 32
  let q := 50050
  p + q = 50082 :=
sorry

end NUMINAMATH_GPT_sticker_distribution_probability_l467_46764


namespace NUMINAMATH_GPT_peter_large_glasses_l467_46739

theorem peter_large_glasses (cost_small cost_large total_money small_glasses change num_large_glasses : ℕ)
    (h1 : cost_small = 3)
    (h2 : cost_large = 5)
    (h3 : total_money = 50)
    (h4 : small_glasses = 8)
    (h5 : change = 1)
    (h6 : total_money - change = 49)
    (h7 : small_glasses * cost_small = 24)
    (h8 : 49 - 24 = 25)
    (h9 : 25 / cost_large = 5) :
  num_large_glasses = 5 :=
by
  sorry

end NUMINAMATH_GPT_peter_large_glasses_l467_46739


namespace NUMINAMATH_GPT_polygon_divided_into_7_triangles_l467_46774

theorem polygon_divided_into_7_triangles (n : ℕ) (h : n - 2 = 7) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_polygon_divided_into_7_triangles_l467_46774


namespace NUMINAMATH_GPT_triangle_inequalities_l467_46747

theorem triangle_inequalities (a b c : ℝ) :
  (∀ n : ℕ, a^n + b^n > c^n ∧ a^n + c^n > b^n ∧ b^n + c^n > a^n) →
  (a = b ∧ a > c) ∨ (a = b ∧ b = c) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequalities_l467_46747


namespace NUMINAMATH_GPT_range_of_m_l467_46731

theorem range_of_m (m : ℝ) :
  (∃ x y, (x^2 / (2*m) + y^2 / (15 - m) = 1) ∧ m > 0 ∧ (15 - m > 0) ∧ (15 - m > 2 * m))
  ∨ (∀ e, (2 < e ∧ e < 3) ∧ ∃ a b x y, (y^2 / 2 - x^2 / (3 * m) = 1) ∧ (4 < (b^2 / a^2) ∧ (b^2 / a^2) < 9)) →
  (¬ (∃ x y, (x^2 / (2*m) + y^2 / (15 - m) = 1) ∧ (∀ e, (2 < e ∧ e < 3) ∧ ∃ a b x y, (y^2 / 2 - x^2 / (3 * m) = 1) ∧ (4 < (b^2 / a^2) ∧ (b^2 / a^2) < 9)))) →
  (0 < m ∧ m ≤ 2) ∨ (5 ≤ m ∧ m < 16/3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l467_46731


namespace NUMINAMATH_GPT_find_y_l467_46793

theorem find_y (x y : ℤ) (h1 : x^2 - 5 * x + 8 = y + 6) (h2 : x = -8) : y = 106 := by
  sorry

end NUMINAMATH_GPT_find_y_l467_46793


namespace NUMINAMATH_GPT_rectangle_perimeter_l467_46761

theorem rectangle_perimeter (breadth length : ℝ) (h1 : length = 3 * breadth) (h2 : length * breadth = 147) : 2 * length + 2 * breadth = 56 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l467_46761


namespace NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l467_46765

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l467_46765


namespace NUMINAMATH_GPT_sector_area_l467_46757

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 4) :
  1/2 * r^2 * α = π / 2 :=
by
  subst h_r
  subst h_α
  sorry

end NUMINAMATH_GPT_sector_area_l467_46757


namespace NUMINAMATH_GPT_gcd_ab_a2b2_l467_46727

theorem gcd_ab_a2b2 (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_ab_a2b2_l467_46727


namespace NUMINAMATH_GPT_corresponding_angles_not_always_equal_l467_46726

theorem corresponding_angles_not_always_equal :
  (∀ α β c : ℝ, (α = β ∧ ¬c = 0) → (∃ x1 x2 y : ℝ, α = x1 ∧ β = x2 ∧ x1 = y * c ∧ x2 = y * c)) → False :=
by
  sorry

end NUMINAMATH_GPT_corresponding_angles_not_always_equal_l467_46726


namespace NUMINAMATH_GPT_negation_of_p_equiv_l467_46744

-- Define the initial proposition p
def p : Prop := ∃ x : ℝ, x^2 - 5*x - 6 < 0

-- State the theorem for the negation of p
theorem negation_of_p_equiv : ¬p ↔ ∀ x : ℝ, x^2 - 5*x - 6 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_equiv_l467_46744


namespace NUMINAMATH_GPT_factorize_polynomial_l467_46717

theorem factorize_polynomial (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
by 
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l467_46717


namespace NUMINAMATH_GPT_equation_of_line_l467_46792

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 4

-- Define the line equation with parameters m and b
def line (m b x : ℝ) : ℝ := m * x + b

-- Define the point of intersection with the parabola on the line x = k
def intersection_point_parabola (k : ℝ) : ℝ := parabola k

-- Define the point of intersection with the line on the line x = k
def intersection_point_line (m b k : ℝ) : ℝ := line m b k

-- Define the vertical distance between the points on x = k
def vertical_distance (k m b : ℝ) : ℝ :=
  abs ((parabola k) - (line m b k))

-- Define the condition that vertical distance is exactly 4 units
def intersection_distance_condition (k m b : ℝ) : Prop :=
  vertical_distance k m b = 4

-- The line passes through point (2, 8)
def passes_through_point (m b : ℝ) : Prop :=
  line m b 2 = 8

-- Non-zero y-intercept condition
def non_zero_intercept (b : ℝ) : Prop := 
  b ≠ 0

-- The final theorem stating the required equation of the line
theorem equation_of_line (m b : ℝ) (h1 : ∃ k, intersection_distance_condition k m b)
  (h2 : passes_through_point m b) (h3 : non_zero_intercept b) : 
  (m = 12 ∧ b = -16) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_l467_46792


namespace NUMINAMATH_GPT_hyperbola_condition_l467_46730

theorem hyperbola_condition (m : ℝ) : 
  (exists a b : ℝ, ¬ a = 0 ∧ ¬ b = 0 ∧ ( ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 )) →
  ( -2 < m ∧ m < -1 ) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l467_46730


namespace NUMINAMATH_GPT_homes_termite_ridden_but_not_collapsing_fraction_l467_46745

variable (H : Type) -- Representing Homes on Gotham Street

def termite_ridden_fraction : ℚ := 1 / 3
def collapsing_fraction_given_termite_ridden : ℚ := 7 / 10

theorem homes_termite_ridden_but_not_collapsing_fraction :
  (termite_ridden_fraction * (1 - collapsing_fraction_given_termite_ridden)) = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_homes_termite_ridden_but_not_collapsing_fraction_l467_46745


namespace NUMINAMATH_GPT_polynomial_equivalence_l467_46771

-- Define the polynomial T in terms of x.
def T (x : ℝ) : ℝ := (x-2)^5 + 5 * (x-2)^4 + 10 * (x-2)^3 + 10 * (x-2)^2 + 5 * (x-2) + 1

-- Define the target polynomial.
def target (x : ℝ) : ℝ := (x-1)^5

-- State the theorem that T is equivalent to target.
theorem polynomial_equivalence (x : ℝ) : T x = target x :=
by
  sorry

end NUMINAMATH_GPT_polynomial_equivalence_l467_46771


namespace NUMINAMATH_GPT_proof_of_area_weighted_sum_of_distances_l467_46750

def area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) 
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ) 
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : Prop :=
  t1 * z1 + t2 * z2 + t3 * z3 + t4 * z4 = t * z

theorem proof_of_area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ)
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : area_weighted_sum_of_distances a b a1 a2 a3 a4 b1 b2 b3 b4 t1 t2 t3 t4 t z1 z2 z3 z4 z h1 h2 h3 h4 rect_area :=
  sorry

end NUMINAMATH_GPT_proof_of_area_weighted_sum_of_distances_l467_46750


namespace NUMINAMATH_GPT_angles_in_triangle_l467_46741

theorem angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 2 * B = 3 * A) (h3 : 5 * A = 2 * C) :
  B = 54 ∧ C = 90 :=
by
  sorry

end NUMINAMATH_GPT_angles_in_triangle_l467_46741


namespace NUMINAMATH_GPT_sqrt_subtraction_l467_46748

theorem sqrt_subtraction :
  (Real.sqrt (49 + 81)) - (Real.sqrt (36 - 9)) = (Real.sqrt 130) - (3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_sqrt_subtraction_l467_46748


namespace NUMINAMATH_GPT_geometric_series_inequality_l467_46716

variables {x y : ℝ}

theorem geometric_series_inequality 
  (hx : |x| < 1) 
  (hy : |y| < 1) :
  (1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y)) :=
sorry

end NUMINAMATH_GPT_geometric_series_inequality_l467_46716


namespace NUMINAMATH_GPT_area_inside_C_outside_A_B_l467_46778

/-- Define the radii of circles A, B, and C --/
def radius_A : ℝ := 1
def radius_B : ℝ := 1
def radius_C : ℝ := 2

/-- Define the condition of tangency and overlap --/
def circles_tangent_at_one_point (r1 r2 : ℝ) : Prop :=
  r1 = r2 

def circle_C_tangent_to_A_B (rA rB rC : ℝ) : Prop :=
  rA = 1 ∧ rB = 1 ∧ rC = 2 ∧ circles_tangent_at_one_point rA rB

/-- Statement to be proved: The area inside circle C but outside circles A and B is 2π --/
theorem area_inside_C_outside_A_B (h : circle_C_tangent_to_A_B radius_A radius_B radius_C) : 
  π * radius_C^2 - π * (radius_A^2 + radius_B^2) = 2 * π :=
by
  sorry

end NUMINAMATH_GPT_area_inside_C_outside_A_B_l467_46778


namespace NUMINAMATH_GPT_thirty_percent_of_forty_percent_of_x_l467_46702

theorem thirty_percent_of_forty_percent_of_x (x : ℝ) (h : 0.12 * x = 24) : 0.30 * 0.40 * x = 24 :=
sorry

end NUMINAMATH_GPT_thirty_percent_of_forty_percent_of_x_l467_46702


namespace NUMINAMATH_GPT_erica_total_earnings_l467_46780

def fishPrice : Nat := 20
def pastCatch : Nat := 80
def todayCatch : Nat := 2 * pastCatch
def pastEarnings := pastCatch * fishPrice
def todayEarnings := todayCatch * fishPrice
def totalEarnings := pastEarnings + todayEarnings

theorem erica_total_earnings : totalEarnings = 4800 := by
  sorry

end NUMINAMATH_GPT_erica_total_earnings_l467_46780


namespace NUMINAMATH_GPT_quadratic_factorization_l467_46733

theorem quadratic_factorization :
  ∃ a b : ℕ, (a > b) ∧ (x^2 - 20 * x + 96 = (x - a) * (x - b)) ∧ (4 * b - a = 20) := sorry

end NUMINAMATH_GPT_quadratic_factorization_l467_46733


namespace NUMINAMATH_GPT_Fred_last_week_l467_46737

-- Definitions from conditions
def Fred_now := 40
def Fred_earned := 21

-- The theorem we need to prove
theorem Fred_last_week :
  Fred_now - Fred_earned = 19 :=
by
  sorry

end NUMINAMATH_GPT_Fred_last_week_l467_46737


namespace NUMINAMATH_GPT_find_product_in_geometric_sequence_l467_46791

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

theorem find_product_in_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 7 * a 13 = 8) : 
  a 3 * a 11 = 4 :=
by sorry

end NUMINAMATH_GPT_find_product_in_geometric_sequence_l467_46791


namespace NUMINAMATH_GPT_smallest_Y_l467_46782

theorem smallest_Y (S : ℕ) (h1 : (∀ d ∈ S.digits 10, d = 0 ∨ d = 1)) (h2 : 18 ∣ S) : 
  (∃ (Y : ℕ), Y = S / 18 ∧ ∀ (S' : ℕ), (∀ d ∈ S'.digits 10, d = 0 ∨ d = 1) → 18 ∣ S' → S' / 18 ≥ Y) → 
  Y = 6172839500 :=
sorry

end NUMINAMATH_GPT_smallest_Y_l467_46782


namespace NUMINAMATH_GPT_no_minimum_of_f_over_M_l467_46753

/-- Define the domain M for the function y = log(3 - 4x + x^2) -/
def domain_M (x : ℝ) : Prop := (x > 3 ∨ x < 1)

/-- Define the function f(x) = 2x + 2 - 3 * 4^x -/
noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * 4^x

/-- The theorem statement:
    Prove that f(x) does not have a minimum value for x in the domain M -/
theorem no_minimum_of_f_over_M : ¬ ∃ x ∈ {x | domain_M x}, ∀ y ∈ {x | domain_M x}, f x ≤ f y := sorry

end NUMINAMATH_GPT_no_minimum_of_f_over_M_l467_46753


namespace NUMINAMATH_GPT_opposite_of_three_minus_one_l467_46798

theorem opposite_of_three_minus_one : -(3 - 1) = -2 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_three_minus_one_l467_46798
