import Mathlib

namespace NUMINAMATH_GPT_sum_of_squares_geometric_progression_theorem_l917_91755

noncomputable def sum_of_squares_geometric_progression (a₁ q : ℝ) (S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) : ℝ :=
  S₁ * S₂

theorem sum_of_squares_geometric_progression_theorem
  (a₁ q S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) :
  sum_of_squares_geometric_progression a₁ q S₁ S₂ h_q h_S₁ h_S₂ = S₁ * S₂ := sorry

end NUMINAMATH_GPT_sum_of_squares_geometric_progression_theorem_l917_91755


namespace NUMINAMATH_GPT_joe_total_time_l917_91733

variable (r_w t_w : ℝ) 
variable (t_total : ℝ)

-- Given conditions:
def joe_problem_conditions : Prop :=
  (r_w > 0) ∧ 
  (t_w = 9) ∧
  (3 * r_w * (3)) / 2 = r_w * 9 / 2 + 1 / 2

-- The statement to prove:
theorem joe_total_time (h : joe_problem_conditions r_w t_w) : t_total = 13 :=
by { sorry }

end NUMINAMATH_GPT_joe_total_time_l917_91733


namespace NUMINAMATH_GPT_inequality_squares_l917_91749

theorem inequality_squares (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h : a + b + c = 1) :
    (3 / 16) ≤ ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ∧
    ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_squares_l917_91749


namespace NUMINAMATH_GPT_evaluate_f_x_plus_3_l917_91790

def f (x : ℝ) : ℝ := x^2

theorem evaluate_f_x_plus_3 (x : ℝ) : f (x + 3) = x^2 + 6 * x + 9 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_x_plus_3_l917_91790


namespace NUMINAMATH_GPT_factor_expression_l917_91761

theorem factor_expression (a b c : ℝ) :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + ab + bc + ca) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l917_91761


namespace NUMINAMATH_GPT_combined_height_of_trees_l917_91798

noncomputable def growth_rate_A (weeks : ℝ) : ℝ := (weeks / 2) * 50
noncomputable def growth_rate_B (weeks : ℝ) : ℝ := (weeks / 3) * 70
noncomputable def growth_rate_C (weeks : ℝ) : ℝ := (weeks / 4) * 90
noncomputable def initial_height_A : ℝ := 200
noncomputable def initial_height_B : ℝ := 150
noncomputable def initial_height_C : ℝ := 250
noncomputable def total_weeks : ℝ := 16
noncomputable def total_growth_A := growth_rate_A total_weeks
noncomputable def total_growth_B := growth_rate_B total_weeks
noncomputable def total_growth_C := growth_rate_C total_weeks
noncomputable def final_height_A := initial_height_A + total_growth_A
noncomputable def final_height_B := initial_height_B + total_growth_B
noncomputable def final_height_C := initial_height_C + total_growth_C
noncomputable def final_combined_height := final_height_A + final_height_B + final_height_C

theorem combined_height_of_trees :
  final_combined_height = 1733.33 := by
  sorry

end NUMINAMATH_GPT_combined_height_of_trees_l917_91798


namespace NUMINAMATH_GPT_evaluate_expression_l917_91730

theorem evaluate_expression : (733 * 733) - (732 * 734) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l917_91730


namespace NUMINAMATH_GPT_count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l917_91704

noncomputable def count_equally_spaced_integers : ℕ := 
  sorry

theorem count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle:
  count_equally_spaced_integers = 4 :=
sorry

end NUMINAMATH_GPT_count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l917_91704


namespace NUMINAMATH_GPT_base_of_power_expr_l917_91752

-- Defining the power expression as a condition
def power_expr : ℤ := (-4 : ℤ) ^ 3

-- The Lean statement for the proof problem
theorem base_of_power_expr : ∃ b : ℤ, (power_expr = b ^ 3) ∧ (b = -4) := 
sorry

end NUMINAMATH_GPT_base_of_power_expr_l917_91752


namespace NUMINAMATH_GPT_shaded_area_of_square_l917_91728

theorem shaded_area_of_square (side_square : ℝ) (leg_triangle : ℝ) (h1 : side_square = 40) (h2 : leg_triangle = 25) :
  let area_square := side_square ^ 2
  let area_triangle := (1 / 2) * leg_triangle * leg_triangle
  let total_area_triangles := 2 * area_triangle
  let shaded_area := area_square - total_area_triangles
  shaded_area = 975 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_square_l917_91728


namespace NUMINAMATH_GPT_find_expression_l917_91778

def B : ℂ := 3 + 2 * Complex.I
def Q : ℂ := -5 * Complex.I
def R : ℂ := 1 + Complex.I
def T : ℂ := 3 - 4 * Complex.I

theorem find_expression : B * R + Q + T = 4 + Complex.I := by
  sorry

end NUMINAMATH_GPT_find_expression_l917_91778


namespace NUMINAMATH_GPT_go_piece_arrangement_l917_91714

theorem go_piece_arrangement (w b : ℕ) (pieces : List ℕ) 
    (h_w : w = 180) (h_b : b = 181)
    (h_pieces : pieces.length = w + b) 
    (h_black_count : pieces.count 1 = b) 
    (h_white_count : pieces.count 0 = w) :
    ∃ (i j : ℕ), i < j ∧ j < pieces.length ∧ 
    ((j - i - 1 = 178) ∨ (j - i - 1 = 181)) ∧ 
    (pieces.get ⟨i, sorry⟩ = 1) ∧ 
    (pieces.get ⟨j, sorry⟩ = 1) := 
sorry

end NUMINAMATH_GPT_go_piece_arrangement_l917_91714


namespace NUMINAMATH_GPT_base_for_four_digit_even_l917_91723

theorem base_for_four_digit_even (b : ℕ) : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0 → b = 6 :=
by
  sorry

end NUMINAMATH_GPT_base_for_four_digit_even_l917_91723


namespace NUMINAMATH_GPT_initial_price_of_gasoline_l917_91748

theorem initial_price_of_gasoline 
  (P0 : ℝ) 
  (P1 : ℝ := 1.30 * P0)
  (P2 : ℝ := 0.75 * P1)
  (P3 : ℝ := 1.10 * P2)
  (P4 : ℝ := 0.85 * P3)
  (P5 : ℝ := 0.80 * P4)
  (h : P5 = 102.60) : 
  P0 = 140.67 :=
by sorry

end NUMINAMATH_GPT_initial_price_of_gasoline_l917_91748


namespace NUMINAMATH_GPT_monotonically_increasing_range_k_l917_91726

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem monotonically_increasing_range_k :
  (∀ x > 1, deriv (f k) x ≥ 0) → k ≥ 1 :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_range_k_l917_91726


namespace NUMINAMATH_GPT_input_command_is_INPUT_l917_91770

-- Define the commands
def PRINT : String := "PRINT"
def INPUT : String := "INPUT"
def THEN : String := "THEN"
def END : String := "END"

-- Define the properties of each command
def PRINT_is_output (cmd : String) : Prop :=
  cmd = PRINT

def INPUT_is_input (cmd : String) : Prop :=
  cmd = INPUT

def THEN_is_conditional (cmd : String) : Prop :=
  cmd = THEN

def END_is_end (cmd : String) : Prop :=
  cmd = END

-- Theorem stating that INPUT is the command associated with input operation
theorem input_command_is_INPUT : INPUT_is_input INPUT :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_input_command_is_INPUT_l917_91770


namespace NUMINAMATH_GPT_value_of_x2_y2_z2_l917_91707

variable (x y z : ℝ)

theorem value_of_x2_y2_z2 (h1 : x^2 + 3 * y = 4) 
                          (h2 : y^2 - 5 * z = 5) 
                          (h3 : z^2 - 7 * x = -8) : 
                          x^2 + y^2 + z^2 = 20.75 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x2_y2_z2_l917_91707


namespace NUMINAMATH_GPT_bothStoresSaleSameDate_l917_91782

-- Define the conditions
def isBookstoreSaleDay (d : ℕ) : Prop := d % 4 = 0
def isShoeStoreSaleDay (d : ℕ) : Prop := ∃ k : ℕ, d = 5 + 7 * k
def isJulyDay (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 31

-- Define the problem statement
theorem bothStoresSaleSameDate : 
  (∃ d1 d2 : ℕ, isJulyDay d1 ∧ isBookstoreSaleDay d1 ∧ isShoeStoreSaleDay d1 ∧
                 isJulyDay d2 ∧ isBookstoreSaleDay d2 ∧ isShoeStoreSaleDay d2 ∧ d1 ≠ d2) :=
sorry

end NUMINAMATH_GPT_bothStoresSaleSameDate_l917_91782


namespace NUMINAMATH_GPT_carrie_total_sales_l917_91795

theorem carrie_total_sales :
  let tomatoes := 200
  let carrots := 350
  let price_tomato := 1.0
  let price_carrot := 1.50
  (tomatoes * price_tomato + carrots * price_carrot) = 725 := by
  -- let tomatoes := 200
  -- let carrots := 350
  -- let price_tomato := 1.0
  -- let price_carrot := 1.50
  -- show (tomatoes * price_tomato + carrots * price_carrot) = 725
  sorry

end NUMINAMATH_GPT_carrie_total_sales_l917_91795


namespace NUMINAMATH_GPT_geometric_sequence_sum_l917_91772

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 2 = 1 - a 1)
  (h3 : a 4 = 9 - a 3)
  (h4 : ∀ n, a (n + 1) = a n * q) :
  a 4 + a 5 = 27 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l917_91772


namespace NUMINAMATH_GPT_original_cube_volume_l917_91758

theorem original_cube_volume (a : ℕ) (V_cube V_new : ℕ)
  (h1 : V_cube = a^3)
  (h2 : V_new = (a + 2) * a * (a - 2))
  (h3 : V_cube = V_new + 24) :
  V_cube = 216 :=
by
  sorry

end NUMINAMATH_GPT_original_cube_volume_l917_91758


namespace NUMINAMATH_GPT_spend_together_is_85_l917_91785

variable (B D : ℝ)

theorem spend_together_is_85 (h1 : D = 0.70 * B) (h2 : B = D + 15) : B + D = 85 := by
  sorry

end NUMINAMATH_GPT_spend_together_is_85_l917_91785


namespace NUMINAMATH_GPT_additional_grassy_ground_l917_91736

theorem additional_grassy_ground (r₁ r₂ : ℝ) (π : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 23) :
  π * r₂ ^ 2 - π * r₁ ^ 2 = 385 * π :=
  by
  subst h₁ h₂
  sorry

end NUMINAMATH_GPT_additional_grassy_ground_l917_91736


namespace NUMINAMATH_GPT_smaller_circle_radius_l917_91712

open Real

def is_geometric_progression (a b c : ℝ) : Prop :=
  (b / a = c / b)

theorem smaller_circle_radius 
  (B1 B2 : ℝ) 
  (r2 : ℝ) 
  (h1 : B1 + B2 = π * r2^2) 
  (h2 : r2 = 5) 
  (h3 : is_geometric_progression B1 B2 (B1 + B2)) :
  sqrt ((-1 + sqrt (1 + 100 * π)) / (2 * π)) = sqrt (B1 / π) :=
by
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l917_91712


namespace NUMINAMATH_GPT_heidi_and_karl_painting_l917_91745

-- Given conditions
def heidi_paint_rate := 1 / 60 -- Rate at which Heidi paints, in walls per minute
def karl_paint_rate := 2 * heidi_paint_rate -- Rate at which Karl paints, in walls per minute
def painting_time := 20 -- Time spent painting, in minutes

-- Prove the amount of each wall painted
theorem heidi_and_karl_painting :
  (heidi_paint_rate * painting_time = 1 / 3) ∧ (karl_paint_rate * painting_time = 2 / 3) :=
sorry

end NUMINAMATH_GPT_heidi_and_karl_painting_l917_91745


namespace NUMINAMATH_GPT_f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l917_91780

noncomputable def f (x : ℝ) : ℝ := x + 4/x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem f_min_value_pos : ∀ x : ℝ, x > 0 → f x ≥ 4 :=
by
  sorry

theorem f_minimum_at_2 : f 2 = 4 :=
by
  sorry

theorem f_increasing_intervals : (MonotoneOn f {x | x ≤ -2} ∧ MonotoneOn f {x | x ≥ 2}) :=
by
  sorry

end NUMINAMATH_GPT_f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l917_91780


namespace NUMINAMATH_GPT_calc_expr_l917_91787

theorem calc_expr : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 :=
by sorry

end NUMINAMATH_GPT_calc_expr_l917_91787


namespace NUMINAMATH_GPT_rectangle_length_l917_91735

theorem rectangle_length
    (a : ℕ)
    (b : ℕ)
    (area_square : a * a = 81)
    (width_rect : b = 3)
    (area_equal : a * a = b * (27) )
    : b * 27 = 81 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l917_91735


namespace NUMINAMATH_GPT_charity_total_cost_l917_91794

theorem charity_total_cost
  (plates : ℕ)
  (rice_cost_per_plate chicken_cost_per_plate : ℕ)
  (h1 : plates = 100)
  (h2 : rice_cost_per_plate = 10)
  (h3 : chicken_cost_per_plate = 40) :
  plates * (rice_cost_per_plate + chicken_cost_per_plate) / 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_charity_total_cost_l917_91794


namespace NUMINAMATH_GPT_fishermen_total_catch_l917_91779

noncomputable def m : ℕ := 30  -- Mike can catch 30 fish per hour
noncomputable def j : ℕ := 2 * m  -- Jim can catch twice as much as Mike
noncomputable def b : ℕ := j + (j / 2)  -- Bob can catch 50% more than Jim

noncomputable def fish_caught_in_40_minutes : ℕ := (2 * m) / 3 -- Fishermen fish together for 40 minutes (2/3 hour)
noncomputable def fish_caught_by_jim_in_remaining_time : ℕ := j / 3 -- Jim fishes alone for the remaining 20 minutes (1/3 hour)

noncomputable def total_fish_caught : ℕ :=
  fish_caught_in_40_minutes * 3 + fish_caught_by_jim_in_remaining_time

theorem fishermen_total_catch : total_fish_caught = 140 := by
  sorry

end NUMINAMATH_GPT_fishermen_total_catch_l917_91779


namespace NUMINAMATH_GPT_stair_calculation_l917_91747

def already_climbed : ℕ := 74
def left_to_climb : ℕ := 22
def total_stairs : ℕ := 96

theorem stair_calculation :
  already_climbed + left_to_climb = total_stairs :=
by {
  sorry
}

end NUMINAMATH_GPT_stair_calculation_l917_91747


namespace NUMINAMATH_GPT_cookies_per_batch_l917_91756

-- Define the necessary conditions
def total_chips : ℕ := 81
def batches : ℕ := 3
def chips_per_cookie : ℕ := 9

-- Theorem stating the number of cookies per batch
theorem cookies_per_batch : (total_chips / batches) / chips_per_cookie = 3 :=
by
  -- Here would be the proof, but we use sorry as placeholder
  sorry

end NUMINAMATH_GPT_cookies_per_batch_l917_91756


namespace NUMINAMATH_GPT_volume_of_locations_eq_27sqrt6pi_over_8_l917_91743

noncomputable def volumeOfLocationSet : ℝ :=
  let sqrt2_inv := 1 / (2 * Real.sqrt 2)
  let points := [ (sqrt2_inv, sqrt2_inv, sqrt2_inv),
                  (sqrt2_inv, sqrt2_inv, -sqrt2_inv),
                  (sqrt2_inv, -sqrt2_inv, sqrt2_inv),
                  (-sqrt2_inv, sqrt2_inv, sqrt2_inv) ]
  let condition (x y z : ℝ) : Prop :=
    4 * (x^2 + y^2 + z^2) + 3 / 2 ≤ 15
  let r := Real.sqrt (27 / 8)
  let volume := (4/3) * Real.pi * r^3
  volume

theorem volume_of_locations_eq_27sqrt6pi_over_8 :
  volumeOfLocationSet = 27 * Real.sqrt 6 * Real.pi / 8 :=
sorry

end NUMINAMATH_GPT_volume_of_locations_eq_27sqrt6pi_over_8_l917_91743


namespace NUMINAMATH_GPT_range_of_m_and_n_l917_91711

theorem range_of_m_and_n (m n : ℝ) : 
  (2 * 2 - 3 + m > 0) → ¬ (2 + 3 - n ≤ 0) → (m > -1 ∧ n < 5) := by
  intros hA hB
  sorry

end NUMINAMATH_GPT_range_of_m_and_n_l917_91711


namespace NUMINAMATH_GPT_circle_parabola_intersect_l917_91710

theorem circle_parabola_intersect (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = 1 ∧ y = a * x^2 → (x ≠ 0 ∨ y ≠ 0)) ↔ a > 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_parabola_intersect_l917_91710


namespace NUMINAMATH_GPT_find_arithmetic_mean_l917_91713

theorem find_arithmetic_mean (σ μ : ℝ) (hσ : σ = 1.5) (h : 11 = μ - 2 * σ) : μ = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_arithmetic_mean_l917_91713


namespace NUMINAMATH_GPT_ray_total_grocery_bill_l917_91720

noncomputable def meat_cost : ℝ := 5
noncomputable def crackers_cost : ℝ := 3.50
noncomputable def veg_cost_per_bag : ℝ := 2
noncomputable def veg_bags : ℕ := 4
noncomputable def cheese_cost : ℝ := 3.50
noncomputable def discount_rate : ℝ := 0.10

noncomputable def total_grocery_bill : ℝ :=
  let veg_total := veg_cost_per_bag * (veg_bags:ℝ)
  let total_before_discount := meat_cost + crackers_cost + veg_total + cheese_cost
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

theorem ray_total_grocery_bill : total_grocery_bill = 18 :=
  by
  sorry

end NUMINAMATH_GPT_ray_total_grocery_bill_l917_91720


namespace NUMINAMATH_GPT_find_m_l917_91709

theorem find_m (m : ℝ) :
  (∃ x : ℝ, x^2 - m * x + m^2 - 19 = 0 ∧ (x = 2 ∨ x = 3))
  ∧ (∀ x : ℝ, x^2 - m * x + m^2 - 19 = 0 → x ≠ 2 ∧ x ≠ -4) 
  → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l917_91709


namespace NUMINAMATH_GPT_part1_part2_l917_91763

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 2 :=
by
  sorry

theorem part2 (x : ℝ) : (∀ b : ℝ, b ≠ 0 → f x ≥ (|2 * b + 1| - |1 - b|) / |b|) → (x ≤ -1.5 ∨ x ≥ 1.5) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l917_91763


namespace NUMINAMATH_GPT_largest_four_digit_number_prop_l917_91705

theorem largest_four_digit_number_prop :
  ∃ (a b c d : ℕ), a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ (1000 * a + 100 * b + 10 * c + d = 9099) ∧ (c = a + b) ∧ (d = b + c) :=
by
  sorry

end NUMINAMATH_GPT_largest_four_digit_number_prop_l917_91705


namespace NUMINAMATH_GPT_fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l917_91797

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem fermats_little_theorem_poly (X : ℤ) :
  (X + 1) ^ p = X ^ p + 1 := by
    sorry

theorem binom_coeff_divisible_by_prime {k : ℕ} (hkp : 1 ≤ k ∧ k < p) :
  p ∣ Nat.choose p k := by
    sorry

end NUMINAMATH_GPT_fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l917_91797


namespace NUMINAMATH_GPT_solve_for_x_l917_91719

noncomputable def f (x : ℝ) : ℝ := x^3

noncomputable def f_prime (x : ℝ) : ℝ := 3

theorem solve_for_x (x : ℝ) (h : f_prime x = 3) : x = 1 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l917_91719


namespace NUMINAMATH_GPT_num_children_with_dogs_only_l917_91776

-- Defining the given values and constants
def total_children : ℕ := 30
def children_with_cats : ℕ := 12
def children_with_dogs_and_cats : ℕ := 6

-- Define the required proof statement
theorem num_children_with_dogs_only : 
  ∃ (D : ℕ), D + children_with_dogs_and_cats + (children_with_cats - children_with_dogs_and_cats) = total_children ∧ D = 18 :=
by
  sorry

end NUMINAMATH_GPT_num_children_with_dogs_only_l917_91776


namespace NUMINAMATH_GPT_speed_difference_l917_91700

def anna_time_min := 15
def ben_time_min := 25
def distance_miles := 8

def anna_speed_mph := (distance_miles : ℚ) / (anna_time_min / 60 : ℚ)
def ben_speed_mph := (distance_miles : ℚ) / (ben_time_min / 60 : ℚ)

theorem speed_difference : (anna_speed_mph - ben_speed_mph : ℚ) = 12.8 := by {
  sorry
}

end NUMINAMATH_GPT_speed_difference_l917_91700


namespace NUMINAMATH_GPT_difference_of_squares_l917_91764

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- Define the condition for the expression which should hold
def expression_b := (2 * x + y) * (y - 2 * x)

-- The theorem to prove that this expression fits the formula for the difference of squares
theorem difference_of_squares : 
  ∃ a b : ℝ, expression_b x y = a^2 - b^2 := 
by 
  sorry

end NUMINAMATH_GPT_difference_of_squares_l917_91764


namespace NUMINAMATH_GPT_sign_up_ways_l917_91766

theorem sign_up_ways : 
  let num_ways_A := 2
  let num_ways_B := 2
  let num_ways_C := 2
  num_ways_A * num_ways_B * num_ways_C = 8 := 
by 
  -- show the proof (omitted for simplicity)
  sorry

end NUMINAMATH_GPT_sign_up_ways_l917_91766


namespace NUMINAMATH_GPT_intersect_parabolas_l917_91775

theorem intersect_parabolas :
  ∀ (x y : ℝ),
    ((y = 2 * x^2 - 7 * x + 1 ∧ y = 8 * x^2 + 5 * x + 1) ↔ 
     ((x = -2 ∧ y = 23) ∨ (x = 0 ∧ y = 1))) :=
by sorry

end NUMINAMATH_GPT_intersect_parabolas_l917_91775


namespace NUMINAMATH_GPT_inequality_proof_l917_91792

theorem inequality_proof (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l917_91792


namespace NUMINAMATH_GPT_solve_equation_l917_91786

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (x / (x + 1) = 2 / (x^2 - 1)) ↔ (x = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l917_91786


namespace NUMINAMATH_GPT_solve_system_of_equations_l917_91762

theorem solve_system_of_equations :
  ∃ (x y z : ℝ), 
    (2 * y + x - x^2 - y^2 = 0) ∧ 
    (z - x + y - y * (x + z) = 0) ∧ 
    (-2 * y + z - y^2 - z^2 = 0) ∧ 
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 0 ∧ z = 1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l917_91762


namespace NUMINAMATH_GPT_sum_of_x_values_l917_91750

theorem sum_of_x_values :
  (2^(x^2 + 6*x + 9) = 16^(x + 3)) → ∃ x1 x2 : ℝ, x1 + x2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_values_l917_91750


namespace NUMINAMATH_GPT_exp_division_rule_l917_91703

-- The theorem to prove the given problem
theorem exp_division_rule (x : ℝ) (hx : x ≠ 0) :
  x^10 / x^5 = x^5 :=
by sorry

end NUMINAMATH_GPT_exp_division_rule_l917_91703


namespace NUMINAMATH_GPT_fill_tank_with_only_C_l917_91777

noncomputable def time_to_fill_with_only_C (x y z : ℝ) : ℝ := 
  let eq1 := (1 / z - 1 / x) * 2 = 1
  let eq2 := (1 / z - 1 / y) * 4 = 1
  let eq3 := 1 / z * 5 - (1 / x + 1 / y) * 8 = 0
  z

theorem fill_tank_with_only_C (x y z : ℝ) (h1 : (1 / z - 1 / x) * 2 = 1) 
  (h2 : (1 / z - 1 / y) * 4 = 1) (h3 : 1 / z * 5 - (1 / x + 1 / y) * 8 = 0) : 
  time_to_fill_with_only_C x y z = 11 / 6 :=
by
  sorry

end NUMINAMATH_GPT_fill_tank_with_only_C_l917_91777


namespace NUMINAMATH_GPT_value_of_x_minus_y_squared_l917_91769

theorem value_of_x_minus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : (x - y) ^ 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_minus_y_squared_l917_91769


namespace NUMINAMATH_GPT_math_expression_evaluation_l917_91791

theorem math_expression_evaluation :
  36 + (120 / 15) + (15 * 19) - 150 - (450 / 9) = 129 :=
by
  sorry

end NUMINAMATH_GPT_math_expression_evaluation_l917_91791


namespace NUMINAMATH_GPT_intersection_sum_l917_91760

theorem intersection_sum (h j : ℝ → ℝ)
  (H1 : h 3 = 3 ∧ j 3 = 3)
  (H2 : h 6 = 9 ∧ j 6 = 9)
  (H3 : h 9 = 18 ∧ j 9 = 18)
  (H4 : h 12 = 18 ∧ j 12 = 18) :
  ∃ a b : ℕ, h (3 * a) = b ∧ 3 * j a = b ∧ (a + b = 33) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_sum_l917_91760


namespace NUMINAMATH_GPT_required_oranges_for_juice_l917_91708

theorem required_oranges_for_juice (oranges quarts : ℚ) (h : oranges = 36 ∧ quarts = 48) :
  ∃ x, ((oranges / quarts) = (x / 6) ∧ x = 4.5) := 
by sorry

end NUMINAMATH_GPT_required_oranges_for_juice_l917_91708


namespace NUMINAMATH_GPT_john_bike_speed_l917_91732

noncomputable def average_speed_for_bike_ride (swim_distance swim_speed run_distance run_speed bike_distance total_time : ℕ) := 
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

theorem john_bike_speed : average_speed_for_bike_ride 1 5 8 12 (3 / 2) = 18 := by
  sorry

end NUMINAMATH_GPT_john_bike_speed_l917_91732


namespace NUMINAMATH_GPT_seats_empty_l917_91773

def number_of_people : ℕ := 532
def total_seats : ℕ := 750

theorem seats_empty (n : ℕ) (m : ℕ) : m - n = 218 := by
  have number_of_people : ℕ := 532
  have total_seats : ℕ := 750
  sorry

end NUMINAMATH_GPT_seats_empty_l917_91773


namespace NUMINAMATH_GPT_cells_at_day_10_l917_91784

-- Define a function to compute the number of cells given initial cells, tripling rate, intervals, and total time.
def number_of_cells (initial_cells : ℕ) (ratio : ℕ) (interval : ℕ) (total_time : ℕ) : ℕ :=
  let n := total_time / interval + 1
  initial_cells * ratio^(n-1)

-- State the main theorem
theorem cells_at_day_10 :
  number_of_cells 5 3 2 10 = 1215 := by
  sorry

end NUMINAMATH_GPT_cells_at_day_10_l917_91784


namespace NUMINAMATH_GPT_find_n_l917_91724

noncomputable def satisfies_condition (n d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ) : Prop :=
  1 = d₁ ∧ d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ d₄ < d₅ ∧ d₅ < d₆ ∧ d₆ < d₇ ∧ d₇ < n ∧
  (∀ d, d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n → n % d = 0) ∧
  (∀ d, n % d = 0 → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n)

theorem find_n (n : ℕ) : (∃ d₁ d₂ d₃ d₄ d₅ d₆ d₇, satisfies_condition n d₁ d₂ d₃ d₄ d₅ d₆ d₇ ∧ n = d₆^2 + d₇^2 - 1) → (n = 144 ∨ n = 1984) :=
  by
  sorry

end NUMINAMATH_GPT_find_n_l917_91724


namespace NUMINAMATH_GPT_integer_value_expression_l917_91737

theorem integer_value_expression (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  (p = 2 ∧ q = 2) ∨ (p ≠ 2 ∧ q = 2 ∧ pq + p^p + q^q = 3 * (p + q)) :=
sorry

end NUMINAMATH_GPT_integer_value_expression_l917_91737


namespace NUMINAMATH_GPT_total_weight_of_peppers_l917_91757

def green_peppers_weight : Real := 0.3333333333333333
def red_peppers_weight : Real := 0.3333333333333333
def total_peppers_weight : Real := 0.6666666666666666

theorem total_weight_of_peppers :
  green_peppers_weight + red_peppers_weight = total_peppers_weight :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_peppers_l917_91757


namespace NUMINAMATH_GPT_real_solutions_l917_91731

theorem real_solutions :
  ∃ x : ℝ, 
    (x = 9 ∨ x = 5) ∧ 
    (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
     1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10) := 
by 
  sorry  

end NUMINAMATH_GPT_real_solutions_l917_91731


namespace NUMINAMATH_GPT_real_values_of_a_l917_91715

noncomputable def P (x a b : ℝ) : ℝ := x^2 - 2 * a * x + b

theorem real_values_of_a (a b : ℝ) :
  (P 0 a b ≠ 0) →
  (P 1 a b ≠ 0) →
  (P 2 a b ≠ 0) →
  (P 1 a b / P 0 a b = P 2 a b / P 1 a b) →
  (∃ b, P x 1 b = 0) :=
by
  sorry

end NUMINAMATH_GPT_real_values_of_a_l917_91715


namespace NUMINAMATH_GPT_total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l917_91744

-- Define the conditions
def total_people (A B : ℕ) : Prop := A + B = 92
def valid_class_A (A : ℕ) : Prop := 51 < A ∧ A < 55
def total_cost (sets : ℕ) (cost_per_set : ℕ) : ℕ := sets * cost_per_set

-- Prices per set for different ranges of number of sets
def price_per_set (n : ℕ) : ℕ :=
  if n > 90 then 30 else if n > 50 then 40 else 50

-- Question 1
theorem total_amount_for_uniforms (A B : ℕ) (h1 : total_people A B) : total_cost 92 30 = 2760 := sorry

-- Question 2
theorem students_in_classes (A B : ℕ) (h1 : total_people A B) (h2 : valid_class_A A) (h3 : 40 * A + 50 * B = 4080) : A = 52 ∧ B = 40 := sorry

-- Question 3
theorem cost_effective_purchase_plan (A : ℕ) (h1 : 51 < A ∧ A < 55) (B : ℕ) (h2 : 92 - A = B) (h3 : A - 8 + B = 91) :
  ∃ (cost : ℕ), cost = total_cost 91 30 ∧ cost = 2730 := sorry

end NUMINAMATH_GPT_total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l917_91744


namespace NUMINAMATH_GPT_largest_side_of_rectangle_l917_91742

theorem largest_side_of_rectangle (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 1920) : 
    max l w = 101 := 
sorry

end NUMINAMATH_GPT_largest_side_of_rectangle_l917_91742


namespace NUMINAMATH_GPT_ratio_is_9_l917_91799

-- Define the set of numbers
def set_of_numbers := { x : ℕ | ∃ n, n ≤ 8 ∧ x = 10^n }

-- Define the sum of the geometric series excluding the largest element
def sum_of_others : ℕ := (Finset.range 8).sum (λ n => 10^n)

-- Define the largest element
def largest_element := 10^8

-- Define the ratio of the largest element to the sum of the other elements
def ratio := largest_element / sum_of_others

-- Problem statement: The ratio is 9
theorem ratio_is_9 : ratio = 9 := by
  sorry

end NUMINAMATH_GPT_ratio_is_9_l917_91799


namespace NUMINAMATH_GPT_paige_science_problems_l917_91716

variable (S : ℤ)

theorem paige_science_problems (h1 : 43 + S - 44 = 11) : S = 12 :=
by
  sorry

end NUMINAMATH_GPT_paige_science_problems_l917_91716


namespace NUMINAMATH_GPT_river_road_cars_l917_91701

theorem river_road_cars
  (B C : ℕ)
  (h1 : B * 17 = C)
  (h2 : C = B + 80) :
  C = 85 := by
  sorry

end NUMINAMATH_GPT_river_road_cars_l917_91701


namespace NUMINAMATH_GPT_meteorite_weight_possibilities_l917_91788

def valid_meteorite_weight_combinations : ℕ :=
  (2 * (Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2))) + (Nat.factorial 5)

theorem meteorite_weight_possibilities :
  valid_meteorite_weight_combinations = 180 :=
by
  -- Sorry added to skip the proof.
  sorry

end NUMINAMATH_GPT_meteorite_weight_possibilities_l917_91788


namespace NUMINAMATH_GPT_area_of_rectangular_field_l917_91718

-- Define the conditions
def length (b : ℕ) : ℕ := b + 30
def perimeter (b : ℕ) (l : ℕ) : ℕ := 2 * (b + l)

-- Define the main theorem to prove
theorem area_of_rectangular_field (b : ℕ) (l : ℕ) (h1 : l = length b) (h2 : perimeter b l = 540) : 
  l * b = 18000 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l917_91718


namespace NUMINAMATH_GPT_no_prime_solution_l917_91783

theorem no_prime_solution (p : ℕ) (h_prime : Nat.Prime p) : ¬(2^p + p ∣ 3^p + p) := by
  sorry

end NUMINAMATH_GPT_no_prime_solution_l917_91783


namespace NUMINAMATH_GPT_least_cost_of_grass_seed_l917_91739

-- Definitions of the prices and weights
def price_per_bag (size : Nat) : Float :=
  if size = 5 then 13.85
  else if size = 10 then 20.40
  else if size = 25 then 32.25
  else 0.0

-- The conditions for the weights and costs
def valid_weight_range (total_weight : Nat) : Prop :=
  65 ≤ total_weight ∧ total_weight ≤ 80

-- Calculate the total cost given quantities of each bag size
def total_cost (bag5 : Nat) (bag10 : Nat) (bag25 : Nat) : Float :=
  Float.ofNat bag5 * price_per_bag 5 + Float.ofNat bag10 * price_per_bag 10 + Float.ofNat bag25 * price_per_bag 25

-- Correct cost for the minimum possible cost within the given weight range
def min_possible_cost : Float := 98.75

-- Proof statement to be proven
theorem least_cost_of_grass_seed : ∃ (bag5 bag10 bag25 : Nat), 
  valid_weight_range (bag5 * 5 + bag10 * 10 + bag25 * 25) ∧ total_cost bag5 bag10 bag25 = min_possible_cost :=
sorry

end NUMINAMATH_GPT_least_cost_of_grass_seed_l917_91739


namespace NUMINAMATH_GPT_women_count_l917_91702

def total_passengers : Nat := 54
def men : Nat := 18
def children : Nat := 10
def women : Nat := total_passengers - men - children

theorem women_count : women = 26 :=
sorry

end NUMINAMATH_GPT_women_count_l917_91702


namespace NUMINAMATH_GPT_orthodiagonal_quadrilateral_l917_91759

-- Define the quadrilateral sides and their relationships
variables (AB BC CD DA : ℝ)
variables (h1 : AB = 20) (h2 : BC = 70) (h3 : CD = 90)
theorem orthodiagonal_quadrilateral : AB^2 + CD^2 = BC^2 + DA^2 → DA = 60 :=
by
  sorry

end NUMINAMATH_GPT_orthodiagonal_quadrilateral_l917_91759


namespace NUMINAMATH_GPT_area_of_square_same_yarn_l917_91746

theorem area_of_square_same_yarn (a : ℕ) (ha : a = 4) :
  let hexagon_perimeter := 6 * a
  let square_side := hexagon_perimeter / 4
  square_side * square_side = 36 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_same_yarn_l917_91746


namespace NUMINAMATH_GPT_hexagon_angle_E_l917_91717

theorem hexagon_angle_E (A N G L E S : ℝ) 
  (h1 : A = G) 
  (h2 : G = E) 
  (h3 : N + S = 180) 
  (h4 : L = 90) 
  (h_sum : A + N + G + L + E + S = 720) : 
  E = 150 := 
by 
  sorry

end NUMINAMATH_GPT_hexagon_angle_E_l917_91717


namespace NUMINAMATH_GPT_class_gpa_l917_91740

theorem class_gpa (n : ℕ) (h1 : (n / 3) * 60 + (2 * (n / 3)) * 66 = total_gpa) :
  total_gpa / n = 64 :=
by
  sorry

end NUMINAMATH_GPT_class_gpa_l917_91740


namespace NUMINAMATH_GPT_find_k_find_m_l917_91734

-- Condition definitions
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 3)

-- Proof problem statements
theorem find_k (k : ℝ) :
  (3 * a.fst - b.fst) / (a.fst + k * b.fst) = (3 * a.snd - b.snd) / (a.snd + k * b.snd) →
  k = -1 / 3 :=
sorry

theorem find_m (m : ℝ) :
  a.fst * (m * a.fst - b.fst) + a.snd * (m * a.snd - b.snd) = 0 →
  m = -4 / 5 :=
sorry

end NUMINAMATH_GPT_find_k_find_m_l917_91734


namespace NUMINAMATH_GPT_punctures_covered_l917_91768

theorem punctures_covered (P1 P2 P3 : ℝ) (h1 : 0 ≤ P1) (h2 : P1 < P2) (h3 : P2 < P3) (h4 : P3 < 3) :
    ∃ x, x ≤ P1 ∧ x + 2 ≥ P3 := 
sorry

end NUMINAMATH_GPT_punctures_covered_l917_91768


namespace NUMINAMATH_GPT_sqrt_of_mixed_number_l917_91721

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_mixed_number_l917_91721


namespace NUMINAMATH_GPT_find_xy_l917_91765

theorem find_xy (x y : ℝ) (k : ℤ) :
  3 * Real.sin x - 4 * Real.cos x = 4 * y^2 + 4 * y + 6 ↔
  (x = -Real.arccos (-4/5) + (2 * k + 1) * Real.pi ∧ y = -1/2) := by
  sorry

end NUMINAMATH_GPT_find_xy_l917_91765


namespace NUMINAMATH_GPT_a_works_less_than_b_l917_91774

theorem a_works_less_than_b (A B : ℝ) (x y : ℝ)
  (h1 : A = 3 * B)
  (h2 : (A + B) * 22.5 = A * x)
  (h3 : y = 3 * x) :
  y - x = 60 :=
by sorry

end NUMINAMATH_GPT_a_works_less_than_b_l917_91774


namespace NUMINAMATH_GPT_fewer_blue_than_green_l917_91754

-- Definitions for given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def total_buttons : ℕ := 275
def blue_buttons : ℕ := total_buttons - (green_buttons + yellow_buttons)

-- Theorem statement to be proved
theorem fewer_blue_than_green : green_buttons - blue_buttons = 5 :=
by
  -- Proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_fewer_blue_than_green_l917_91754


namespace NUMINAMATH_GPT_incorrect_statement_c_l917_91725

-- Define even function
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

-- Define odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Function definitions
def f1 (x : ℝ) : ℝ := x^4 + x^2
def f2 (x : ℝ) : ℝ := x^3 + x^2

-- Main theorem statement
theorem incorrect_statement_c : ¬ is_odd f2 := sorry

end NUMINAMATH_GPT_incorrect_statement_c_l917_91725


namespace NUMINAMATH_GPT_train_boarding_probability_l917_91793

theorem train_boarding_probability :
  (0.5 / 5) = 1 / 10 :=
by sorry

end NUMINAMATH_GPT_train_boarding_probability_l917_91793


namespace NUMINAMATH_GPT_rhombus_condition_perimeter_rhombus_given_ab_l917_91727

noncomputable def roots_of_quadratic (m : ℝ) : Set ℝ :=
{ x : ℝ | x^2 - m * x + m / 2 - 1 / 4 = 0 }

theorem rhombus_condition (m : ℝ) : 
  (∃ ab ad : ℝ, ab ∈ roots_of_quadratic m ∧ ad ∈ roots_of_quadratic m ∧ ab = ad) ↔ m = 1 :=
by
  sorry

theorem perimeter_rhombus_given_ab (m : ℝ) (ab : ℝ) (ad : ℝ) : 
  ab = 2 →
  (ab ∈ roots_of_quadratic m) →
  (ad ∈ roots_of_quadratic m) →
  ab ≠ ad →
  m = 5 / 2 →
  2 * (ab + ad) = 5 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_condition_perimeter_rhombus_given_ab_l917_91727


namespace NUMINAMATH_GPT_heartsuit_ratio_l917_91738

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_heartsuit_ratio_l917_91738


namespace NUMINAMATH_GPT_Ron_four_times_Maurice_l917_91767

theorem Ron_four_times_Maurice
  (r m : ℕ) (x : ℕ) 
  (h_r : r = 43) 
  (h_m : m = 7) 
  (h_eq : r + x = 4 * (m + x)) : 
  x = 5 := 
by
  sorry

end NUMINAMATH_GPT_Ron_four_times_Maurice_l917_91767


namespace NUMINAMATH_GPT_fleas_difference_l917_91741

-- Define the initial number of fleas and subsequent fleas after each treatment.
def initial_fleas (F : ℝ) := F
def after_first_treatment (F : ℝ) := F * 0.40
def after_second_treatment (F : ℝ) := (after_first_treatment F) * 0.55
def after_third_treatment (F : ℝ) := (after_second_treatment F) * 0.70
def after_fourth_treatment (F : ℝ) := (after_third_treatment F) * 0.80

-- Given condition
axiom final_fleas : initial_fleas 20 = after_fourth_treatment 20

-- Prove the number of fleas before treatment minus the number after treatment is 142
theorem fleas_difference (F : ℝ) (h : initial_fleas F = after_fourth_treatment 20) : 
  F - 20 = 142 :=
by {
  sorry
}

end NUMINAMATH_GPT_fleas_difference_l917_91741


namespace NUMINAMATH_GPT_parity_related_to_phi_not_omega_l917_91751

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem parity_related_to_phi_not_omega (ω : ℝ) (φ : ℝ) (h : 0 < ω) :
  (∃ k : ℤ, φ = k * Real.pi → ∀ x : ℝ, f ω φ (-x) = -f ω φ x) ∧
  (∃ k : ℤ, φ = k * Real.pi + Real.pi / 2 → ∀ x : ℝ, f ω φ (-x) = f ω φ x) :=
sorry

end NUMINAMATH_GPT_parity_related_to_phi_not_omega_l917_91751


namespace NUMINAMATH_GPT_cost_of_candy_car_l917_91722

theorem cost_of_candy_car (starting_amount paid_amount change : ℝ) (h1 : starting_amount = 1.80) (h2 : change = 1.35) (h3 : paid_amount = starting_amount - change) : paid_amount = 0.45 := by
  sorry

end NUMINAMATH_GPT_cost_of_candy_car_l917_91722


namespace NUMINAMATH_GPT_parallel_lines_l917_91781

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, (a-1) * x + 2 * y + 10 = 0) → (∀ x y : ℝ, x + a * y + 3 = 0) → (a = -1 ∨ a = 2) :=
sorry

end NUMINAMATH_GPT_parallel_lines_l917_91781


namespace NUMINAMATH_GPT_extreme_points_inequality_l917_91789

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * Real.log (1 + x)

-- Given m > 0 and f(x) has extreme points x1 and x2 such that x1 < x2
theorem extreme_points_inequality {m x1 x2 : ℝ} (h_m : m > 0)
    (h_extreme1 : x1 = (-1 - Real.sqrt (1 - 2 * m)) / 2)
    (h_extreme2 : x2 = (-1 + Real.sqrt (1 - 2 * m)) / 2)
    (h_order : x1 < x2) :
    2 * f x2 m > -x1 + 2 * x1 * Real.log 2 := sorry

end NUMINAMATH_GPT_extreme_points_inequality_l917_91789


namespace NUMINAMATH_GPT_inscribed_sphere_radius_l917_91796

theorem inscribed_sphere_radius (h1 h2 h3 h4 : ℝ) (S1 S2 S3 S4 V : ℝ)
  (h1_ge : h1 ≥ 1) (h2_ge : h2 ≥ 1) (h3_ge : h3 ≥ 1) (h4_ge : h4 ≥ 1)
  (volume : V = (1/3) * S1 * h1)
  : (∃ r : ℝ, 3 * V = (S1 + S2 + S3 + S4) * r ∧ r = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l917_91796


namespace NUMINAMATH_GPT_factorial_power_of_two_l917_91729

theorem factorial_power_of_two solutions (a b c : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_equation : a.factorial + b.factorial = 2^(c.factorial)) :
  solutions = [(1, 1, 1), (2, 2, 2)] :=
sorry

end NUMINAMATH_GPT_factorial_power_of_two_l917_91729


namespace NUMINAMATH_GPT_line_canonical_eqn_l917_91753

theorem line_canonical_eqn 
  (x y z : ℝ)
  (h1 : x - y + z - 2 = 0)
  (h2 : x - 2*y - z + 4 = 0) :
  ∃ a : ℝ, ∃ b : ℝ, ∃ c : ℝ,
    (a = (x - 8)/3) ∧ (b = (y - 6)/2) ∧ (c = z/(-1)) ∧ (a = b) ∧ (b = c) ∧ (c = a) :=
by sorry

end NUMINAMATH_GPT_line_canonical_eqn_l917_91753


namespace NUMINAMATH_GPT_polar_coordinates_of_point_l917_91771

noncomputable def point_rectangular_to_polar (x y : ℝ) : ℝ × ℝ := 
  let r := Real.sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  (r, θ)

theorem polar_coordinates_of_point :
  point_rectangular_to_polar 1 (-1) = (Real.sqrt 2, 7 * Real.pi / 4) :=
by
  unfold point_rectangular_to_polar
  sorry

end NUMINAMATH_GPT_polar_coordinates_of_point_l917_91771


namespace NUMINAMATH_GPT_goods_train_speed_l917_91706

noncomputable def speed_of_goods_train (train_speed : ℝ) (goods_length : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed_mps := goods_length / passing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  (relative_speed_kmph - train_speed)

theorem goods_train_speed :
  speed_of_goods_train 30 280 9 = 82 :=
by
  sorry

end NUMINAMATH_GPT_goods_train_speed_l917_91706
