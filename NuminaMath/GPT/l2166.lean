import Mathlib

namespace NUMINAMATH_GPT_pencils_left_l2166_216610

def ashton_boxes : Nat := 3
def pencils_per_box : Nat := 14
def pencils_given_to_brother : Nat := 6
def pencils_given_to_friends : Nat := 12

theorem pencils_left (h₁ : ashton_boxes = 3) 
                     (h₂ : pencils_per_box = 14)
                     (h₃ : pencils_given_to_brother = 6)
                     (h₄ : pencils_given_to_friends = 12) :
  (ashton_boxes * pencils_per_box - pencils_given_to_brother - pencils_given_to_friends) = 24 :=
by
  sorry

end NUMINAMATH_GPT_pencils_left_l2166_216610


namespace NUMINAMATH_GPT_largest_integral_x_l2166_216601

theorem largest_integral_x (x : ℤ) : (1 / 4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 7 / 9 ↔ x = 4 :=
by 
  sorry

end NUMINAMATH_GPT_largest_integral_x_l2166_216601


namespace NUMINAMATH_GPT_ted_age_proof_l2166_216653

theorem ted_age_proof (s t : ℝ) (h1 : t = 3 * s - 20) (h2 : t + s = 78) : t = 53.5 :=
by
  sorry  -- Proof steps are not required, hence using sorry.

end NUMINAMATH_GPT_ted_age_proof_l2166_216653


namespace NUMINAMATH_GPT_sin_double_angle_l2166_216672

theorem sin_double_angle (α : ℝ) (h : Real.tan α = -1/3) : Real.sin (2 * α) = -3/5 := by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_l2166_216672


namespace NUMINAMATH_GPT_flyers_total_l2166_216679

theorem flyers_total (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) 
  (hj : jack_flyers = 120) (hr : rose_flyers = 320) (hl : left_flyers = 796) :
  jack_flyers + rose_flyers + left_flyers = 1236 :=
by {
  sorry
}

end NUMINAMATH_GPT_flyers_total_l2166_216679


namespace NUMINAMATH_GPT_value_of_sum_l2166_216645

theorem value_of_sum (a b c : ℚ) (h1 : 2 * a + 3 * b + c = 27) (h2 : 4 * a + 6 * b + 5 * c = 71) :
  a + b + c = 115 / 9 :=
sorry

end NUMINAMATH_GPT_value_of_sum_l2166_216645


namespace NUMINAMATH_GPT_composite_sum_of_four_integers_l2166_216613

theorem composite_sum_of_four_integers 
  (a b c d : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_eq : a^2 + b^2 + a * b = c^2 + d^2 + c * d) : 
  ∃ n m : ℕ, 1 < a + b + c + d ∧ a + b + c + d = n * m ∧ 1 < n ∧ 1 < m := 
sorry

end NUMINAMATH_GPT_composite_sum_of_four_integers_l2166_216613


namespace NUMINAMATH_GPT_count_valid_pairs_is_7_l2166_216628

def valid_pairs_count : Nat :=
  let pairs := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]
  List.length pairs

theorem count_valid_pairs_is_7 (b c : ℕ) (hb : b > 0) (hc : c > 0) :
  (b^2 - 4 * c ≤ 0) → (c^2 - 4 * b ≤ 0) → valid_pairs_count = 7 :=
by
  sorry

end NUMINAMATH_GPT_count_valid_pairs_is_7_l2166_216628


namespace NUMINAMATH_GPT_inequality_holds_for_all_real_l2166_216619

open Real -- Open the real numbers namespace

theorem inequality_holds_for_all_real (x : ℝ) : 
  2^((sin x)^2) + 2^((cos x)^2) ≥ 2 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_real_l2166_216619


namespace NUMINAMATH_GPT_factorization_of_difference_of_squares_l2166_216617

variable {R : Type} [CommRing R]

theorem factorization_of_difference_of_squares (m : R) : m^2 - 4 = (m + 2) * (m - 2) :=
by sorry

end NUMINAMATH_GPT_factorization_of_difference_of_squares_l2166_216617


namespace NUMINAMATH_GPT_minValue_Proof_l2166_216604

noncomputable def minValue (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) : Prop :=
  ∃ m : ℝ, m = 4.5 ∧ (∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → (1/a + 1/b + 1/c) ≥ 9/2)

theorem minValue_Proof :
  ∀ (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2), 
    minValue x y z h1 h2 h3 h4 := by
  sorry

end NUMINAMATH_GPT_minValue_Proof_l2166_216604


namespace NUMINAMATH_GPT_tied_part_length_l2166_216637

theorem tied_part_length (length_of_each_string : ℕ) (num_strings : ℕ) (total_tied_length : ℕ) 
  (H1 : length_of_each_string = 217) (H2 : num_strings = 3) (H3 : total_tied_length = 627) : 
  (length_of_each_string * num_strings - total_tied_length) / (num_strings - 1) = 12 :=
by
  sorry

end NUMINAMATH_GPT_tied_part_length_l2166_216637


namespace NUMINAMATH_GPT_chicken_legs_baked_l2166_216689

theorem chicken_legs_baked (L : ℕ) (H₁ : 144 / 16 = 9) (H₂ : 224 / 16 = 14) (H₃ : 16 * 9 = 144) :  L = 144 :=
by
  sorry

end NUMINAMATH_GPT_chicken_legs_baked_l2166_216689


namespace NUMINAMATH_GPT_second_person_days_l2166_216650

theorem second_person_days (P1 P2 : ℝ) (h1 : P1 = 1 / 24) (h2 : P1 + P2 = 1 / 8) : 1 / P2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_second_person_days_l2166_216650


namespace NUMINAMATH_GPT_billy_tickets_l2166_216663

theorem billy_tickets (ferris_wheel_rides bumper_car_rides rides_per_ride total_tickets : ℕ) 
  (h1 : ferris_wheel_rides = 7)
  (h2 : bumper_car_rides = 3)
  (h3 : rides_per_ride = 5)
  (h4 : total_tickets = (ferris_wheel_rides + bumper_car_rides) * rides_per_ride) :
  total_tickets = 50 := 
by 
  sorry

end NUMINAMATH_GPT_billy_tickets_l2166_216663


namespace NUMINAMATH_GPT_chenny_friends_l2166_216691

theorem chenny_friends (initial_candies : ℕ) (needed_candies : ℕ) (candies_per_friend : ℕ) (h1 : initial_candies = 10) (h2 : needed_candies = 4) (h3 : candies_per_friend = 2) :
  (initial_candies + needed_candies) / candies_per_friend = 7 :=
by
  sorry

end NUMINAMATH_GPT_chenny_friends_l2166_216691


namespace NUMINAMATH_GPT_factor_polynomial_l2166_216678

theorem factor_polynomial : 
  (x : ℝ) → x^4 - 4 * x^2 + 16 = (x^2 - 4 * x + 4) * (x^2 + 2 * x + 4) :=
by
sorry

end NUMINAMATH_GPT_factor_polynomial_l2166_216678


namespace NUMINAMATH_GPT_trig_expression_equality_l2166_216605

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end NUMINAMATH_GPT_trig_expression_equality_l2166_216605


namespace NUMINAMATH_GPT_positive_value_of_A_l2166_216612

def my_relation (A B k : ℝ) : ℝ := A^2 + k * B^2

theorem positive_value_of_A (A : ℝ) (h1 : ∀ A B, my_relation A B 3 = A^2 + 3 * B^2) (h2 : my_relation A 7 3 = 196) :
  A = 7 := by
  sorry

end NUMINAMATH_GPT_positive_value_of_A_l2166_216612


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l2166_216673

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l2166_216673


namespace NUMINAMATH_GPT_sum_of_six_digits_is_31_l2166_216615

-- Problem constants and definitions
def digits : Set ℕ := {0, 2, 3, 4, 5, 7, 8, 9}

-- Problem conditions expressed as hypotheses
variables (a b c d e f g : ℕ)
variables (h1 : a ∈ digits) (h2 : b ∈ digits) (h3 : c ∈ digits) 
          (h4 : d ∈ digits) (h5 : e ∈ digits) (h6 : f ∈ digits) (h7 : g ∈ digits)
          (h8 : a ≠ b) (h9 : a ≠ c) (h10 : a ≠ d) (h11 : a ≠ e) (h12 : a ≠ f) (h13 : a ≠ g)
          (h14 : b ≠ c) (h15 : b ≠ d) (h16 : b ≠ e) (h17 : b ≠ f) (h18 : b ≠ g)
          (h19 : c ≠ d) (h20 : c ≠ e) (h21 : c ≠ f) (h22 : c ≠ g)
          (h23 : d ≠ e) (h24 : d ≠ f) (h25 : d ≠ g)
          (h26 : e ≠ f) (h27 : e ≠ g) (h28 : f ≠ g)
variable (shared : b = e)
variables (h29 : a + b + c = 24) (h30 : d + e + f + g = 14)

-- Proposition to be proved
theorem sum_of_six_digits_is_31 : a + b + c + d + e + f = 31 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_six_digits_is_31_l2166_216615


namespace NUMINAMATH_GPT_find_r_l2166_216655

theorem find_r : ∃ r : ℕ, (5 + 7 * 8 + 1 * 8^2) = 120 + r ∧ r = 5 := 
by
  use 5
  sorry

end NUMINAMATH_GPT_find_r_l2166_216655


namespace NUMINAMATH_GPT_arc_length_l2166_216676

theorem arc_length (C : ℝ) (theta : ℝ) (hC : C = 100) (htheta : theta = 30) :
  (theta / 360) * C = 25 / 3 :=
by sorry

end NUMINAMATH_GPT_arc_length_l2166_216676


namespace NUMINAMATH_GPT_largest_pies_without_any_ingredients_l2166_216625

-- Define the conditions
def total_pies : ℕ := 60
def pies_with_strawberries : ℕ := total_pies / 4
def pies_with_bananas : ℕ := total_pies * 3 / 8
def pies_with_cherries : ℕ := total_pies / 2
def pies_with_pecans : ℕ := total_pies / 10

-- State the theorem to prove
theorem largest_pies_without_any_ingredients : (total_pies - pies_with_cherries) = 30 := by
  sorry

end NUMINAMATH_GPT_largest_pies_without_any_ingredients_l2166_216625


namespace NUMINAMATH_GPT_problem_proof_l2166_216629

def mixed_to_improper (a b c : ℚ) : ℚ := a + b / c

noncomputable def evaluate_expression : ℚ :=
  100 - (mixed_to_improper 3 1 8) / (mixed_to_improper 2 1 12 - 5 / 8) * (8 / 5 + mixed_to_improper 2 2 3)

theorem problem_proof : evaluate_expression = 636 / 7 := 
  sorry

end NUMINAMATH_GPT_problem_proof_l2166_216629


namespace NUMINAMATH_GPT_problem_statement_l2166_216669

noncomputable def a := 9
noncomputable def b := 729

theorem problem_statement (h1 : ∃ (terms : ℕ), terms = 430)
                          (h2 : ∃ (value : ℕ), value = 3) : a + b = 738 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2166_216669


namespace NUMINAMATH_GPT_average_time_per_stop_l2166_216644

theorem average_time_per_stop (pizzas : ℕ) 
                              (stops_for_two_pizzas : ℕ) 
                              (pizzas_per_stop_for_two : ℕ) 
                              (remaining_pizzas : ℕ) 
                              (total_stops : ℕ) 
                              (total_time : ℕ) 
                              (H1: pizzas = 12) 
                              (H2: stops_for_two_pizzas = 2) 
                              (H3: pizzas_per_stop_for_two = 2) 
                              (H4: remaining_pizzas = pizzas - stops_for_two_pizzas * pizzas_per_stop_for_two)
                              (H5: total_stops = stops_for_two_pizzas + remaining_pizzas)
                              (H6: total_time = 40) :
                              total_time / total_stops = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_time_per_stop_l2166_216644


namespace NUMINAMATH_GPT_dexter_filled_fewer_boxes_with_football_cards_l2166_216623

-- Conditions
def boxes_with_basketball_cards : ℕ := 9
def cards_per_basketball_box : ℕ := 15
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255

-- Definition of the main problem statement
def fewer_boxes_with_football_cards : Prop :=
  let basketball_cards := boxes_with_basketball_cards * cards_per_basketball_box
  let football_cards := total_cards - basketball_cards
  let boxes_with_football_cards := football_cards / cards_per_football_box
  boxes_with_basketball_cards - boxes_with_football_cards = 3

theorem dexter_filled_fewer_boxes_with_football_cards : fewer_boxes_with_football_cards :=
by
  sorry

end NUMINAMATH_GPT_dexter_filled_fewer_boxes_with_football_cards_l2166_216623


namespace NUMINAMATH_GPT_number_of_bricks_is_1800_l2166_216671

-- Define the conditions
def rate_first_bricklayer (x : ℕ) : ℕ := x / 8
def rate_second_bricklayer (x : ℕ) : ℕ := x / 12
def combined_reduced_rate (x : ℕ) : ℕ := (rate_first_bricklayer x + rate_second_bricklayer x - 15)

-- Prove that the number of bricks in the wall is 1800
theorem number_of_bricks_is_1800 :
  ∃ x : ℕ, 5 * combined_reduced_rate x = x ∧ x = 1800 :=
by
  use 1800
  sorry

end NUMINAMATH_GPT_number_of_bricks_is_1800_l2166_216671


namespace NUMINAMATH_GPT_monic_polynomial_root_equivalence_l2166_216667

noncomputable def roots (p : Polynomial ℝ) : List ℝ := sorry

theorem monic_polynomial_root_equivalence :
  let r1 := roots (Polynomial.C (8:ℝ) + Polynomial.X^3 - 3 * Polynomial.X^2)
  let p := Polynomial.C (216:ℝ) + Polynomial.X^3 - 9 * Polynomial.X^2
  r1.map (fun r => 3*r) = roots p :=
by
  sorry

end NUMINAMATH_GPT_monic_polynomial_root_equivalence_l2166_216667


namespace NUMINAMATH_GPT_find_pairs_l2166_216682

-- Define a function that checks if a pair (n, d) satisfies the required conditions
def satisfies_conditions (n d : ℕ) : Prop :=
  ∀ S : ℤ, ∃! (a : ℕ → ℤ), 
    (∀ i : ℕ, i < n → a i ≤ a (i + 1)) ∧                -- Non-decreasing sequence condition
    ((Finset.range n).sum a = S) ∧                  -- Sum of the sequence equals S
    (a n.succ.pred - a 0 = d)                      -- The difference condition

-- The formal statement of the required proof
theorem find_pairs :
  {p : ℕ × ℕ | satisfies_conditions p.fst p.snd} = {(1, 0), (3, 2)} :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l2166_216682


namespace NUMINAMATH_GPT_men_required_l2166_216659

theorem men_required (W M : ℕ) (h1 : M * 20 * W = W) (h2 : (M - 4) * 25 * W = W) : M = 16 := by
  sorry

end NUMINAMATH_GPT_men_required_l2166_216659


namespace NUMINAMATH_GPT_surface_area_increase_factor_l2166_216656

theorem surface_area_increase_factor (n : ℕ) (h : n > 0) : 
  (6 * n^3) / (6 * n^2) = n :=
by {
  sorry -- Proof not required
}

end NUMINAMATH_GPT_surface_area_increase_factor_l2166_216656


namespace NUMINAMATH_GPT_set_equality_implies_sum_zero_l2166_216646

theorem set_equality_implies_sum_zero
  (x y : ℝ)
  (A : Set ℝ := {x, y, x + y})
  (B : Set ℝ := {0, x^2, x * y}) :
  A = B → x + y = 0 :=
by
  sorry

end NUMINAMATH_GPT_set_equality_implies_sum_zero_l2166_216646


namespace NUMINAMATH_GPT_how_many_oxen_c_put_l2166_216621

variables (oxen_a oxen_b months_a months_b rent total_rent c_share x : ℕ)
variable (H : 10 * 7 = oxen_a)
variable (H1 : 12 * 5 = oxen_b)
variable (H2 : 3 * x = months_a)
variable (H3 : 70 + 60 + 3 * x = months_b)
variable (H4 : 280 = total_rent)
variable (H5 : 72 = c_share)

theorem how_many_oxen_c_put : x = 15 :=
  sorry

end NUMINAMATH_GPT_how_many_oxen_c_put_l2166_216621


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_45_l2166_216603

theorem smallest_positive_multiple_of_45 : ∃ (n : ℕ), n > 0 ∧ ∃ (x : ℕ), x > 0 ∧ n = 45 * x ∧ n = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_45_l2166_216603


namespace NUMINAMATH_GPT_total_scoops_l2166_216643

-- Definitions
def single_cone_scoops : ℕ := 1
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

-- Theorem statement: Prove the total number of scoops is 10.
theorem total_scoops : single_cone_scoops + double_cone_scoops + banana_split_scoops + waffle_bowl_scoops = 10 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_scoops_l2166_216643


namespace NUMINAMATH_GPT_least_positive_n_l2166_216606

theorem least_positive_n : ∃ n : ℕ, (1 / (n : ℝ) - 1 / (n + 1 : ℝ) < 1 / 12) ∧ (∀ m : ℕ, (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 12) → n ≤ m) :=
by {
  sorry
}

end NUMINAMATH_GPT_least_positive_n_l2166_216606


namespace NUMINAMATH_GPT_plane_crash_probabilities_eq_l2166_216638

noncomputable def crashing_probability_3_engines (p : ℝ) : ℝ :=
  3 * p^2 * (1 - p) + p^3

noncomputable def crashing_probability_5_engines (p : ℝ) : ℝ :=
  10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

theorem plane_crash_probabilities_eq (p : ℝ) :
  crashing_probability_3_engines p = crashing_probability_5_engines p ↔ p = 0 ∨ p = 1/2 ∨ p = 1 :=
by
  sorry

end NUMINAMATH_GPT_plane_crash_probabilities_eq_l2166_216638


namespace NUMINAMATH_GPT_fraction_covered_by_pepperoni_l2166_216652

theorem fraction_covered_by_pepperoni 
  (d_pizza : ℝ) (n_pepperoni_diameter : ℕ) (n_pepperoni : ℕ) (diameter_pepperoni : ℝ) 
  (radius_pepperoni : ℝ) (radius_pizza : ℝ)
  (area_one_pepperoni : ℝ) (total_area_pepperoni : ℝ) (area_pizza : ℝ)
  (fraction_covered : ℝ)
  (h1 : d_pizza = 16)
  (h2 : n_pepperoni_diameter = 14)
  (h3 : n_pepperoni = 42)
  (h4 : diameter_pepperoni = d_pizza / n_pepperoni_diameter)
  (h5 : radius_pepperoni = diameter_pepperoni / 2)
  (h6 : radius_pizza = d_pizza / 2)
  (h7 : area_one_pepperoni = π * radius_pepperoni ^ 2)
  (h8 : total_area_pepperoni = n_pepperoni * area_one_pepperoni)
  (h9 : area_pizza = π * radius_pizza ^ 2)
  (h10 : fraction_covered = total_area_pepperoni / area_pizza) :
  fraction_covered = 3 / 7 :=
sorry

end NUMINAMATH_GPT_fraction_covered_by_pepperoni_l2166_216652


namespace NUMINAMATH_GPT_ratio_of_areas_l2166_216681

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  -- The problem is to prove the ratio of the areas is 4/9
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l2166_216681


namespace NUMINAMATH_GPT_sqrt_4_eq_2_or_neg2_l2166_216657

theorem sqrt_4_eq_2_or_neg2 (y : ℝ) (h : y^2 = 4) : y = 2 ∨ y = -2 :=
sorry

end NUMINAMATH_GPT_sqrt_4_eq_2_or_neg2_l2166_216657


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_3125_l2166_216683

theorem sum_of_coefficients_eq_3125 
  {b_5 b_4 b_3 b_2 b_1 b_0 : ℤ}
  (h : (2 * x + 3)^5 = b_5 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0) :
  b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 3125 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_3125_l2166_216683


namespace NUMINAMATH_GPT_maximum_ab_is_40_l2166_216633

noncomputable def maximum_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : ℝ :=
  max (a * b) 40

theorem maximum_ab_is_40 {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : maximum_ab a b h₀ h₁ h₂ = 40 := 
by 
  sorry

end NUMINAMATH_GPT_maximum_ab_is_40_l2166_216633


namespace NUMINAMATH_GPT_n_minus_m_l2166_216641

theorem n_minus_m (m n : ℝ) (h1 : m^2 - n^2 = 6) (h2 : m + n = 3) : n - m = -2 :=
by
  sorry

end NUMINAMATH_GPT_n_minus_m_l2166_216641


namespace NUMINAMATH_GPT_Leah_coins_value_in_cents_l2166_216614

theorem Leah_coins_value_in_cents (p n : ℕ) (h₁ : p + n = 15) (h₂ : p = n + 2) : p + 5 * n = 44 :=
by
  sorry

end NUMINAMATH_GPT_Leah_coins_value_in_cents_l2166_216614


namespace NUMINAMATH_GPT_customer_survey_response_l2166_216674

theorem customer_survey_response (N : ℕ)
  (avg_income : ℕ → ℕ)
  (avg_all : avg_income N = 45000)
  (avg_top10 : avg_income 10 = 55000)
  (avg_others : avg_income (N - 10) = 42500) :
  N = 50 := 
sorry

end NUMINAMATH_GPT_customer_survey_response_l2166_216674


namespace NUMINAMATH_GPT_ten_percent_of_number_l2166_216620

theorem ten_percent_of_number (x : ℝ)
  (h : x - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 3.325 :=
sorry

end NUMINAMATH_GPT_ten_percent_of_number_l2166_216620


namespace NUMINAMATH_GPT_rectangle_dimensions_l2166_216636

variable (w l : ℝ)
variable (h1 : l = w + 15)
variable (h2 : 2 * w + 2 * l = 150)

theorem rectangle_dimensions :
  w = 30 ∧ l = 45 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l2166_216636


namespace NUMINAMATH_GPT_sum_second_largest_smallest_l2166_216648

theorem sum_second_largest_smallest (a b c : ℕ) (order_cond : a < b ∧ b < c) : a + b = 21 :=
by
  -- Following the correct answer based on the provided conditions:
  -- 10, 11, and 12 with their ordering, we have the smallest a and the second largest b.
  sorry

end NUMINAMATH_GPT_sum_second_largest_smallest_l2166_216648


namespace NUMINAMATH_GPT_correct_operation_is_d_l2166_216668

theorem correct_operation_is_d (a b : ℝ) : 
  (∀ x y : ℝ, -x * y = -(x * y)) → 
  (∀ x : ℝ, x⁻¹ * (x ^ 2) = x) → 
  (∀ x : ℝ, x ^ 10 / x ^ 4 = x ^ 6) →
  ((a - b) * (-a - b) ≠ a ^ 2 - b ^ 2) ∧ 
  (2 * a ^ 2 * a ^ 3 ≠ 2 * a ^ 6) ∧ 
  ((-a) ^ 10 / (-a) ^ 4 = a ^ 6) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_correct_operation_is_d_l2166_216668


namespace NUMINAMATH_GPT_shirley_boxes_to_cases_l2166_216661

theorem shirley_boxes_to_cases (boxes_sold : Nat) (boxes_per_case : Nat) (cases_needed : Nat) 
      (h1 : boxes_sold = 54) (h2 : boxes_per_case = 6) : cases_needed = 9 :=
by
  sorry

end NUMINAMATH_GPT_shirley_boxes_to_cases_l2166_216661


namespace NUMINAMATH_GPT_ratio_S15_S5_l2166_216626

variable {α : Type*} [LinearOrderedField α]

namespace ArithmeticSequence

def sum_of_first_n_terms (a : α) (d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem ratio_S15_S5
  {a d : α}
  {S5 S10 S15 : α}
  (h1 : S5 = sum_of_first_n_terms a d 5)
  (h2 : S10 = sum_of_first_n_terms a d 10)
  (h3 : S15 = sum_of_first_n_terms a d 15)
  (h_ratio : S5 / S10 = 2 / 3) :
  S15 / S5 = 3 / 2 := 
sorry

end ArithmeticSequence

end NUMINAMATH_GPT_ratio_S15_S5_l2166_216626


namespace NUMINAMATH_GPT_quadrilateral_with_equal_sides_is_rhombus_l2166_216642

theorem quadrilateral_with_equal_sides_is_rhombus (a b c d : ℝ) (h1 : a = b) (h2 : b = c) (h3 : c = d) : a = d :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_with_equal_sides_is_rhombus_l2166_216642


namespace NUMINAMATH_GPT_math_proof_problem_l2166_216698

noncomputable def a_value := 1
noncomputable def b_value := 2

-- Defining the primary conditions
def condition1 (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - 3 * x + 2 > 0) ↔ (x < 1 ∨ x > b)

def condition2 (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - (2 * b - a) * x - 2 * b < 0) ↔ (-1 < x ∧ x < 4)

-- Defining the main goal
theorem math_proof_problem :
  ∃ a b : ℝ, a = a_value ∧ b = b_value ∧ condition1 a b ∧ condition2 a b := 
sorry

end NUMINAMATH_GPT_math_proof_problem_l2166_216698


namespace NUMINAMATH_GPT_opposite_of_fraction_l2166_216687

def opposite_of (x : ℚ) : ℚ := -x

theorem opposite_of_fraction :
  opposite_of (1/2023) = - (1/2023) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_fraction_l2166_216687


namespace NUMINAMATH_GPT_maximum_sum_of_diagonals_of_rhombus_l2166_216684

noncomputable def rhombus_side_length : ℝ := 5
noncomputable def diagonal_bd_max_length : ℝ := 6
noncomputable def diagonal_ac_min_length : ℝ := 6
noncomputable def max_diagonal_sum : ℝ := 14

theorem maximum_sum_of_diagonals_of_rhombus :
  ∀ (s bd ac : ℝ), 
  s = rhombus_side_length → 
  bd ≤ diagonal_bd_max_length → 
  ac ≥ diagonal_ac_min_length → 
  bd + ac ≤ max_diagonal_sum → 
  max_diagonal_sum = 14 :=
by
  sorry

end NUMINAMATH_GPT_maximum_sum_of_diagonals_of_rhombus_l2166_216684


namespace NUMINAMATH_GPT_given_trig_identity_l2166_216688

variable {x : ℂ} {α : ℝ} {n : ℕ}

theorem given_trig_identity (h : x + 1/x = 2 * Real.cos α) : x^n + 1/x^n = 2 * Real.cos (n * α) :=
sorry

end NUMINAMATH_GPT_given_trig_identity_l2166_216688


namespace NUMINAMATH_GPT_how_many_children_got_on_l2166_216640

noncomputable def initial_children : ℝ := 42.5
noncomputable def children_got_off : ℝ := 21.3
noncomputable def final_children : ℝ := 35.8

theorem how_many_children_got_on : initial_children - children_got_off + (final_children - (initial_children - children_got_off)) = final_children := by
  sorry

end NUMINAMATH_GPT_how_many_children_got_on_l2166_216640


namespace NUMINAMATH_GPT_avg_people_moving_per_hour_l2166_216677

theorem avg_people_moving_per_hour (total_people : ℕ) (total_days : ℕ) (hours_per_day : ℕ) (h : total_people = 3000 ∧ total_days = 4 ∧ hours_per_day = 24) : 
  (total_people / (total_days * hours_per_day)).toFloat.round = 31 :=
by
  have h1 : total_people = 3000 := h.1;
  have h2 : total_days = 4 := h.2.1;
  have h3 : hours_per_day = 24 := h.2.2;
  rw [h1, h2, h3];
  sorry

end NUMINAMATH_GPT_avg_people_moving_per_hour_l2166_216677


namespace NUMINAMATH_GPT_find_a_l2166_216600

theorem find_a (a : ℤ) (h₀ : 0 ≤ a ∧ a ≤ 13) (h₁ : 13 ∣ (51 ^ 2016 - a)) : a = 1 := sorry

end NUMINAMATH_GPT_find_a_l2166_216600


namespace NUMINAMATH_GPT_dice_probability_correct_l2166_216690

noncomputable def probability_at_least_one_two_or_three : ℚ :=
  let total_outcomes := 64
  let favorable_outcomes := 64 - 36
  favorable_outcomes / total_outcomes

theorem dice_probability_correct :
  probability_at_least_one_two_or_three = 7 / 16 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_dice_probability_correct_l2166_216690


namespace NUMINAMATH_GPT_a_n_plus_1_is_geometric_general_term_formula_l2166_216622

-- Define the sequence a_n.
def a : ℕ → ℤ
| 0       => 0  -- a_0 is not given explicitly, we start the sequence from 1.
| (n + 1) => if n = 0 then 1 else 2 * a n + 1

-- Prove that the sequence {a_n + 1} is a geometric sequence.
theorem a_n_plus_1_is_geometric : ∃ r : ℤ, ∀ n : ℕ, (a (n + 1) + 1) / (a n + 1) = r := by
  sorry

-- Find the general formula for a_n.
theorem general_term_formula : ∃ f : ℕ → ℤ, ∀ n : ℕ, a n = f n := by
  sorry

end NUMINAMATH_GPT_a_n_plus_1_is_geometric_general_term_formula_l2166_216622


namespace NUMINAMATH_GPT_rectangle_area_l2166_216616

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l2166_216616


namespace NUMINAMATH_GPT_part_i_solution_set_part_ii_minimum_value_l2166_216609

-- Part (I)
theorem part_i_solution_set :
  (∀ (x : ℝ), 1 = 1 ∧ 2 = 2 → |x - 1| + |x + 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) :=
by { sorry }

-- Part (II)
theorem part_ii_minimum_value (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 2 * a * b) :
  |x - a| + |x + b| ≥ 9 / 2 :=
by { sorry }

end NUMINAMATH_GPT_part_i_solution_set_part_ii_minimum_value_l2166_216609


namespace NUMINAMATH_GPT_find_bananas_l2166_216660

theorem find_bananas 
  (bananas apples persimmons : ℕ) 
  (h1 : apples = 4 * bananas) 
  (h2 : persimmons = 3 * bananas) 
  (h3 : apples + persimmons = 210) : 
  bananas = 30 := 
  sorry

end NUMINAMATH_GPT_find_bananas_l2166_216660


namespace NUMINAMATH_GPT_gcd_2197_2208_is_1_l2166_216686

def gcd_2197_2208 : ℕ := Nat.gcd 2197 2208

theorem gcd_2197_2208_is_1 : gcd_2197_2208 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_2197_2208_is_1_l2166_216686


namespace NUMINAMATH_GPT_root_count_sqrt_eq_l2166_216647

open Real

theorem root_count_sqrt_eq (x : ℝ) :
  (∀ y, (y = sqrt (7 - 2 * x)) → y = x * y → (∃ x, x = 7 / 2 ∨ x = 1)) ∧
  (7 - 2 * x ≥ 0) →
  ∃ s, s = 1 ∧ (7 - 2 * s = 0) → x = 1 ∨ x = 7 / 2 :=
sorry

end NUMINAMATH_GPT_root_count_sqrt_eq_l2166_216647


namespace NUMINAMATH_GPT_mona_game_group_size_l2166_216697

theorem mona_game_group_size 
  (x : ℕ)
  (h_conditions: 9 * (x - 1) - 3 = 33) : x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_mona_game_group_size_l2166_216697


namespace NUMINAMATH_GPT_average_marks_is_70_l2166_216624

variable (P C M : ℕ)

-- Condition: The total marks in physics, chemistry, and mathematics is 140 more than the marks in physics
def total_marks_condition : Prop := P + C + M = P + 140

-- Definition of the average marks in chemistry and mathematics
def average_marks_C_M : ℕ := (C + M) / 2

theorem average_marks_is_70 (h : total_marks_condition P C M) : average_marks_C_M C M = 70 :=
sorry

end NUMINAMATH_GPT_average_marks_is_70_l2166_216624


namespace NUMINAMATH_GPT_football_game_spectators_l2166_216631

-- Define the conditions and the proof goals
theorem football_game_spectators 
  (A C : ℕ) 
  (h_condition_1 : 2 * A + 2 * C + 40 = 310) 
  (h_condition_2 : C = A / 2) : 
  A = 90 ∧ C = 45 ∧ (A + C + 20) = 155 := 
by 
  sorry

end NUMINAMATH_GPT_football_game_spectators_l2166_216631


namespace NUMINAMATH_GPT_find_k_l2166_216675

noncomputable def curve (x k : ℝ) : ℝ := x + k * Real.log (1 + x)

theorem find_k (k : ℝ) :
  let y' := (fun x => 1 + k / (1 + x))
  (y' 1 = 2) ∧ ((1 + 2 * 1) = 0) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2166_216675


namespace NUMINAMATH_GPT_total_students_at_concert_l2166_216694

-- Define the number of buses
def num_buses : ℕ := 8

-- Define the number of students per bus
def students_per_bus : ℕ := 45

-- State the theorem with the conditions and expected result
theorem total_students_at_concert : (num_buses * students_per_bus) = 360 := by
  -- Proof is not required as per the instructions; replace with 'sorry'
  sorry

end NUMINAMATH_GPT_total_students_at_concert_l2166_216694


namespace NUMINAMATH_GPT_sum_of_integers_l2166_216693

theorem sum_of_integers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
  (h4 : a * b * c = 343000)
  (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
  a + b + c = 476 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l2166_216693


namespace NUMINAMATH_GPT_line_bisects_circle_l2166_216607

theorem line_bisects_circle
  (C : Type)
  [MetricSpace C]
  (x y : ℝ)
  (h : ∀ {x y : ℝ}, x^2 + y^2 - 2*x - 4*y + 1 = 0) : 
  x - y + 1 = 0 → True :=
by
  intro h_line
  sorry

end NUMINAMATH_GPT_line_bisects_circle_l2166_216607


namespace NUMINAMATH_GPT_solve_quadratic_solution_l2166_216680

theorem solve_quadratic_solution (x : ℝ) : (3 * x^2 - 6 * x = 0) ↔ (x = 0 ∨ x = 2) :=
sorry

end NUMINAMATH_GPT_solve_quadratic_solution_l2166_216680


namespace NUMINAMATH_GPT_rhombus_area_l2166_216695

-- Declare the lengths of the diagonals
def diagonal1 := 6
def diagonal2 := 8

-- Define the area function for a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

-- State the theorem
theorem rhombus_area : area_of_rhombus diagonal1 diagonal2 = 24 := by sorry

end NUMINAMATH_GPT_rhombus_area_l2166_216695


namespace NUMINAMATH_GPT_inequality_first_inequality_second_l2166_216611

theorem inequality_first (x : ℝ) : 4 * x - 2 < 1 - 2 * x → x < 1 / 2 := 
sorry

theorem inequality_second (x : ℝ) : (3 - 2 * x ≥ x - 6) ∧ ((3 * x + 1) / 2 < 2 * x) → 1 < x ∧ x ≤ 3 :=
sorry

end NUMINAMATH_GPT_inequality_first_inequality_second_l2166_216611


namespace NUMINAMATH_GPT_diane_total_loss_l2166_216632

-- Define the starting amount of money Diane had.
def starting_amount : ℤ := 100

-- Define the amount of money Diane won.
def winnings : ℤ := 65

-- Define the amount of money Diane owed at the end.
def debt : ℤ := 50

-- Define the total amount of money Diane had after winnings.
def mid_game_total : ℤ := starting_amount + winnings

-- Define the total amount Diane lost.
def total_loss : ℤ := mid_game_total + debt

-- Theorem stating the total amount Diane lost is 215 dollars.
theorem diane_total_loss : total_loss = 215 := by
  sorry

end NUMINAMATH_GPT_diane_total_loss_l2166_216632


namespace NUMINAMATH_GPT_problem_statement_l2166_216666

variables {Point Line Plane : Type}
variables (l : Line) (α β : Plane)

-- Conditions
def parallel (l : Line) (α : Plane) : Prop := sorry
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def perpendicular_planes (α β : Plane) : Prop := sorry

-- The proof problem
theorem problem_statement (h1 : parallel l α) (h2 : perpendicular l β) : perpendicular_planes α β :=
sorry

end NUMINAMATH_GPT_problem_statement_l2166_216666


namespace NUMINAMATH_GPT_sum_even_deg_coeff_l2166_216634

theorem sum_even_deg_coeff (x : ℕ) : 
  (3 - 2*x)^3 * (2*x + 1)^4 = (3 - 2*x)^3 * (2*x + 1)^4 →
  (∀ (x : ℕ), (3 - 2*x)^3 * (2*1 + 1)^4 =  81 ∧ 
  (3 - 2*(-1))^3 * (2*(-1) + 1)^4 = 125 → 
  (81 + 125) / 2 = 103) :=
by
  sorry

end NUMINAMATH_GPT_sum_even_deg_coeff_l2166_216634


namespace NUMINAMATH_GPT_ellipse_major_minor_ratio_l2166_216692

theorem ellipse_major_minor_ratio (m : ℝ) (x y : ℝ) (h1 : x^2 + y^2 / m = 1) (h2 : 2 * 1 = 4 * Real.sqrt m) 
  : m = 1 / 4 :=
sorry

end NUMINAMATH_GPT_ellipse_major_minor_ratio_l2166_216692


namespace NUMINAMATH_GPT_price_difference_VA_NC_l2166_216699

/-- Define the initial conditions -/
def NC_price : ℝ := 2
def NC_gallons : ℕ := 10
def VA_gallons : ℕ := 10
def total_spent : ℝ := 50

/-- Define the problem to prove the difference in price per gallon between Virginia and North Carolina -/
theorem price_difference_VA_NC (NC_price VA_price total_spent : ℝ) (NC_gallons VA_gallons : ℕ) :
  total_spent = NC_price * NC_gallons + VA_price * VA_gallons →
  VA_price - NC_price = 1 := 
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_price_difference_VA_NC_l2166_216699


namespace NUMINAMATH_GPT_find_roots_square_sum_and_min_y_l2166_216658

-- Definitions from the conditions
def sum_roots (m : ℝ) :=
  -(m + 1)

def product_roots (m : ℝ) :=
  2 * m - 2

def roots_square_sum (m x₁ x₂ : ℝ) :=
  x₁^2 + x₂^2

def y (m : ℝ) :=
  (m - 1)^2 + 4

-- Proof statement
theorem find_roots_square_sum_and_min_y (m x₁ x₂ : ℝ) (h_sum : x₁ + x₂ = sum_roots m)
  (h_prod : x₁ * x₂ = product_roots m) :
  roots_square_sum m x₁ x₂ = (m - 1)^2 + 4 ∧ y m ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_find_roots_square_sum_and_min_y_l2166_216658


namespace NUMINAMATH_GPT_no_solution_integral_pairs_l2166_216685

theorem no_solution_integral_pairs (a b : ℤ) : (1 / (a : ℚ) + 1 / (b : ℚ) = -1 / (a + b : ℚ)) → false :=
by
  sorry

end NUMINAMATH_GPT_no_solution_integral_pairs_l2166_216685


namespace NUMINAMATH_GPT_intersection_line_l2166_216630

-- Define the equations of the circles in Cartesian coordinates.
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + y = 0

-- The theorem to prove.
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → y + 4 * x = 0 :=
by
  sorry

end NUMINAMATH_GPT_intersection_line_l2166_216630


namespace NUMINAMATH_GPT_sum_first_15_terms_l2166_216602

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers

-- Define the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def a1_plus_a15_eq_three (a : ℕ → ℝ) : Prop :=
  a 1 + a 15 = 3

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

theorem sum_first_15_terms (a : ℕ → ℝ) (h_arith: arithmetic_sequence a) (h_sum: a1_plus_a15_eq_three a) :
  sum_first_n_terms a 15 = 22.5 := by
  sorry

end NUMINAMATH_GPT_sum_first_15_terms_l2166_216602


namespace NUMINAMATH_GPT_heather_total_oranges_l2166_216670

--Definition of the problem conditions
def initial_oranges : ℝ := 60.0
def additional_oranges : ℝ := 35.0

--Statement of the theorem
theorem heather_total_oranges : initial_oranges + additional_oranges = 95.0 := by
  sorry

end NUMINAMATH_GPT_heather_total_oranges_l2166_216670


namespace NUMINAMATH_GPT_average_rate_of_change_l2166_216665

noncomputable def f (x : ℝ) : ℝ := x^2 + 1

theorem average_rate_of_change (Δx : ℝ) : 
  (f (1 + Δx) - f 1) / Δx = 2 + Δx := 
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_l2166_216665


namespace NUMINAMATH_GPT_x_intercept_of_line_l2166_216635

theorem x_intercept_of_line :
  (∃ x : ℝ, 5 * x - 7 * 0 = 35 ∧ (x, 0) = (7, 0)) :=
by
  use 7
  simp
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l2166_216635


namespace NUMINAMATH_GPT_expression_value_l2166_216662

theorem expression_value :
  (1 / (3 - (1 / (3 + (1 / (3 - (1 / 3))))))) = (27 / 73) :=
by 
  sorry

end NUMINAMATH_GPT_expression_value_l2166_216662


namespace NUMINAMATH_GPT_min_overlap_percent_l2166_216651

theorem min_overlap_percent
  (M S : ℝ)
  (hM : M = 0.9)
  (hS : S = 0.85) :
  ∃ x, x = 0.75 ∧ (M + S - 1 ≤ x ∧ x ≤ min M S ∧ x = M + S - 1) :=
by
  sorry

end NUMINAMATH_GPT_min_overlap_percent_l2166_216651


namespace NUMINAMATH_GPT_area_of_triangle_FYG_l2166_216696

theorem area_of_triangle_FYG (EF GH : ℝ) 
  (EF_len : EF = 15) 
  (GH_len : GH = 25) 
  (area_trapezoid : 0.5 * (EF + GH) * 10 = 200) 
  (intersection : true) -- Placeholder for intersection condition
  : 0.5 * GH * 3.75 = 46.875 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_FYG_l2166_216696


namespace NUMINAMATH_GPT_f_decreasing_max_k_value_l2166_216618

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_decreasing : ∀ x > 0, ∀ y > 0, x < y → f x > f y := by
  sorry

theorem max_k_value : ∀ x > 0, f x > k / (x + 1) → k ≤ 3 := by
  sorry

end NUMINAMATH_GPT_f_decreasing_max_k_value_l2166_216618


namespace NUMINAMATH_GPT_probability_two_different_colors_l2166_216664

noncomputable def probability_different_colors (total_balls red_balls black_balls : ℕ) : ℚ :=
  let total_ways := (Finset.range total_balls).card.choose 2
  let diff_color_ways := (Finset.range black_balls).card.choose 1 * (Finset.range red_balls).card.choose 1
  diff_color_ways / total_ways

theorem probability_two_different_colors (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ)
  (h_total : total_balls = 5) (h_red : red_balls = 2) (h_black : black_balls = 3) :
  probability_different_colors total_balls red_balls black_balls = 3 / 5 :=
by
  subst h_total
  subst h_red
  subst h_black
  -- Here the proof would follow using the above definitions and reasoning
  sorry

end NUMINAMATH_GPT_probability_two_different_colors_l2166_216664


namespace NUMINAMATH_GPT_find_m_l2166_216608

theorem find_m (n m : ℕ) (h1 : m = 13 * n + 8) (h2 : m = 15 * n) : m = 60 :=
  sorry

end NUMINAMATH_GPT_find_m_l2166_216608


namespace NUMINAMATH_GPT_product_of_two_numbers_l2166_216639

theorem product_of_two_numbers (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 6) : x * y = 616 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2166_216639


namespace NUMINAMATH_GPT_median_eq_range_le_l2166_216654

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end NUMINAMATH_GPT_median_eq_range_le_l2166_216654


namespace NUMINAMATH_GPT_find_m_if_parallel_l2166_216627

-- Given vectors
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b (m : ℝ) : ℝ × ℝ := (m, 2)

-- Parallel condition and the result that m must be -2 or 2
theorem find_m_if_parallel (m : ℝ) (h : ∃ k : ℝ, a m = (k * (b m).fst, k * (b m).snd)) : 
  m = -2 ∨ m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_if_parallel_l2166_216627


namespace NUMINAMATH_GPT_batsman_sixes_l2166_216649

theorem batsman_sixes 
(scorer_runs : ℕ)
(boundaries : ℕ)
(run_contrib : ℕ → ℚ)
(score_by_boundary : ℕ)
(score : ℕ)
(h1 : scorer_runs = 125)
(h2 : boundaries = 5)
(h3 : ∀ (x : ℕ), run_contrib x = (0.60 * scorer_runs : ℚ))
(h4 : score_by_boundary = boundaries * 4)
(h5 : score = scorer_runs - score_by_boundary) : 
∃ (x : ℕ), x = 5 ∧ (scorer_runs = score + (x * 6)) :=
by
  sorry

end NUMINAMATH_GPT_batsman_sixes_l2166_216649
