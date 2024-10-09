import Mathlib

namespace clock_angle_at_3_15_l194_19487

-- Conditions
def full_circle_degrees : ℕ := 360
def hour_degree : ℕ := full_circle_degrees / 12
def minute_degree : ℕ := full_circle_degrees / 60
def minute_position (m : ℕ) : ℕ := m * minute_degree
def hour_position (h m : ℕ) : ℕ := h * hour_degree + m * (hour_degree / 60)

-- Theorem to prove
theorem clock_angle_at_3_15 : (|minute_position 15 - hour_position 3 15| : ℚ) = 7.5 := by
  sorry

end clock_angle_at_3_15_l194_19487


namespace farmer_owned_land_l194_19457

theorem farmer_owned_land (T : ℝ) (h : 0.10 * T = 720) : 0.80 * T = 5760 :=
by
  sorry

end farmer_owned_land_l194_19457


namespace frequency_of_group5_l194_19495

-- Define the total number of students and the frequencies of each group
def total_students : ℕ := 40
def freq_group1 : ℕ := 12
def freq_group2 : ℕ := 10
def freq_group3 : ℕ := 6
def freq_group4 : ℕ := 8

-- Define the frequency of the fifth group in terms of the above frequencies
def freq_group5 : ℕ := total_students - (freq_group1 + freq_group2 + freq_group3 + freq_group4)

-- The theorem to be proven
theorem frequency_of_group5 : freq_group5 = 4 := by
  -- Proof goes here, skipped with sorry
  sorry

end frequency_of_group5_l194_19495


namespace maximum_utilization_rate_80_l194_19484

noncomputable def maximum_utilization_rate (side_length : ℝ) (AF : ℝ) (BF : ℝ) : ℝ :=
  let area_square := side_length * side_length
  let length_rectangle := side_length
  let width_rectangle := AF / 2
  let area_rectangle := length_rectangle * width_rectangle
  (area_rectangle / area_square) * 100

theorem maximum_utilization_rate_80:
  maximum_utilization_rate 4 2 1 = 80 := by
  sorry

end maximum_utilization_rate_80_l194_19484


namespace rain_probability_weekend_l194_19441

theorem rain_probability_weekend :
  let p_rain_F := 0.60
  let p_rain_S := 0.70
  let p_rain_U := 0.40
  let p_no_rain_F := 1 - p_rain_F
  let p_no_rain_S := 1 - p_rain_S
  let p_no_rain_U := 1 - p_rain_U
  let p_no_rain_all_days := p_no_rain_F * p_no_rain_S * p_no_rain_U
  let p_rain_at_least_one_day := 1 - p_no_rain_all_days
  (p_rain_at_least_one_day * 100 = 92.8) := sorry

end rain_probability_weekend_l194_19441


namespace find_f_pi_l194_19476

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.tan (ω * x + Real.pi / 3)

theorem find_f_pi (ω : ℝ) (h_positive : ω > 0) (h_period : Real.pi / ω = 3 * Real.pi) :
  f (ω := ω) Real.pi = -Real.sqrt 3 :=
by
  -- ω is given to be 1/3 by the condition h_period, substituting that 
  -- directly might be clearer for stating the problem accurately
  have h_omega : ω = 1 / 3 := by
    sorry
  rw [h_omega]
  sorry


end find_f_pi_l194_19476


namespace union_of_A_and_B_l194_19471

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem union_of_A_and_B :
  A ∪ B = {-1, 0, 1, 2, 4} :=
by
  sorry

end union_of_A_and_B_l194_19471


namespace smallest_real_number_l194_19498

theorem smallest_real_number :
  ∃ (x : ℝ), x = -3 ∧ (∀ (y : ℝ), y = 0 ∨ y = (-1/3)^2 ∨ y = -((27:ℝ)^(1/3)) ∨ y = -2 → x ≤ y) := 
by 
  sorry

end smallest_real_number_l194_19498


namespace sum_of_min_value_and_input_l194_19444

def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem sum_of_min_value_and_input : 
  let a := -1
  let b := 3 * a - a ^ 3
  a + b = -3 := 
by
  let a := -1
  let b := 3 * a - a ^ 3
  sorry

end sum_of_min_value_and_input_l194_19444


namespace g_g_g_g_3_l194_19483

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end g_g_g_g_3_l194_19483


namespace diagonals_bisect_in_rhombus_l194_19412

axiom Rhombus : Type
axiom Parallelogram : Type

axiom isParallelogram : Rhombus → Parallelogram
axiom diagonalsBisectEachOther : Parallelogram → Prop

theorem diagonals_bisect_in_rhombus (R : Rhombus) :
  ∀ (P : Parallelogram), isParallelogram R = P → diagonalsBisectEachOther P → diagonalsBisectEachOther (isParallelogram R) :=
by
  sorry

end diagonals_bisect_in_rhombus_l194_19412


namespace calc_expression_l194_19464

theorem calc_expression :
  (8^5 / 8^3) * 3^6 = 46656 := by
  sorry

end calc_expression_l194_19464


namespace octahedron_plane_intersection_l194_19463

theorem octahedron_plane_intersection 
  (s : ℝ) 
  (a b c : ℕ) 
  (ha : Nat.Coprime a c) 
  (hb : ∀ p : ℕ, Prime p → p^2 ∣ b → False) 
  (hs : s = 2) 
  (hangle : ∀ θ, θ = 45 ∧ θ = 45) 
  (harea : ∃ A, A = (s^2 * Real.sqrt 3) / 2 ∧ A = a * Real.sqrt b / c): 
  a + b + c = 11 := 
by 
  sorry

end octahedron_plane_intersection_l194_19463


namespace percentage_of_75_eq_percent_of_450_l194_19419

theorem percentage_of_75_eq_percent_of_450 (x : ℝ) (h : (x / 100) * 75 = 0.025 * 450) : x = 15 := 
sorry

end percentage_of_75_eq_percent_of_450_l194_19419


namespace sum_of_reciprocals_eq_three_l194_19449

theorem sum_of_reciprocals_eq_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1/x + 1/y) = 3 := 
by
  sorry

end sum_of_reciprocals_eq_three_l194_19449


namespace inverse_47_mod_48_l194_19409

theorem inverse_47_mod_48 : ∃ x, x < 48 ∧ x > 0 ∧ 47 * x % 48 = 1 :=
sorry

end inverse_47_mod_48_l194_19409


namespace geom_seq_a4_l194_19491

theorem geom_seq_a4 (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h2 : a 3 = 9)
  (h3 : a 5 = 1) :
  a 4 = 3 ∨ a 4 = -3 :=
by {
  sorry
}

end geom_seq_a4_l194_19491


namespace cost_of_tax_free_item_D_l194_19442

theorem cost_of_tax_free_item_D 
  (P_A P_B P_C : ℝ)
  (H1 : 0.945 * P_A + 1.064 * P_B + 1.18 * P_C = 225)
  (H2 : 0.045 * P_A + 0.12 * P_B + 0.18 * P_C = 30) :
  250 - (0.945 * P_A + 1.064 * P_B + 1.18 * P_C) = 25 := 
by
  -- The proof steps would go here.
  sorry

end cost_of_tax_free_item_D_l194_19442


namespace transfer_deck_l194_19431

-- Define the conditions
variables {k n : ℕ}

-- Assume conditions explicitly
axiom k_gt_1 : k > 1
axiom cards_deck : 2*n = 2*n -- Implicitly states that we have 2n cards

-- Define the problem statement
theorem transfer_deck (k_gt_1 : k > 1) (cards_deck : 2*n = 2*n) : n = k - 1 :=
sorry

end transfer_deck_l194_19431


namespace smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l194_19447

theorem smallest_N_such_that_N_and_N_squared_end_in_same_three_digits :
  ∃ N : ℕ, (N > 0) ∧ (N % 1000 = (N^2 % 1000)) ∧ (1 ≤ N / 100 % 10) ∧ (N = 376) :=
by
  sorry

end smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l194_19447


namespace solve_prime_equation_l194_19421

def is_prime (n : ℕ) : Prop := ∀ k, k < n ∧ k > 1 → n % k ≠ 0

theorem solve_prime_equation (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
  (h : 5 * p = q^3 - r^3) : p = 67 ∧ q = 7 ∧ r = 2 :=
sorry

end solve_prime_equation_l194_19421


namespace reciprocal_eq_self_l194_19478

theorem reciprocal_eq_self (x : ℝ) : (1 / x = x) ↔ (x = 1 ∨ x = -1) :=
sorry

end reciprocal_eq_self_l194_19478


namespace total_flowers_sold_l194_19494

/-
Ginger owns a flower shop, where she sells roses, lilacs, and gardenias.
On Tuesday, she sold three times more roses than lilacs, and half as many gardenias as lilacs.
If she sold 10 lilacs, prove that the total number of flowers sold on Tuesday is 45.
-/

theorem total_flowers_sold
    (lilacs roses gardenias : ℕ)
    (h_lilacs : lilacs = 10)
    (h_roses : roses = 3 * lilacs)
    (h_gardenias : gardenias = lilacs / 2)
    (ht : lilacs + roses + gardenias = 45) :
    lilacs + roses + gardenias = 45 :=
by sorry

end total_flowers_sold_l194_19494


namespace bea_has_max_profit_l194_19499

theorem bea_has_max_profit : 
  let price_bea := 25
  let price_dawn := 28
  let price_carla := 35
  let sold_bea := 10
  let sold_dawn := 8
  let sold_carla := 6
  let cost_bea := 10
  let cost_dawn := 12
  let cost_carla := 15
  let profit_bea := (price_bea * sold_bea) - (cost_bea * sold_bea)
  let profit_dawn := (price_dawn * sold_dawn) - (cost_dawn * sold_dawn)
  let profit_carla := (price_carla * sold_carla) - (cost_carla * sold_carla)
  profit_bea = 150 ∧ profit_dawn = 128 ∧ profit_carla = 120 ∧ ∀ p, p ∈ [profit_bea, profit_dawn, profit_carla] → p ≤ 150 :=
by
  sorry

end bea_has_max_profit_l194_19499


namespace perpendicular_lines_slope_l194_19438

theorem perpendicular_lines_slope {m : ℝ} : 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0 → (m * (-1/2)) = -1) → 
  m = 2 :=
by 
  intros h_perpendicular h_slope
  sorry

end perpendicular_lines_slope_l194_19438


namespace recurring_decimal_to_rational_l194_19400

theorem recurring_decimal_to_rational : 
  (0.125125125 : ℝ) = 125 / 999 :=
sorry

end recurring_decimal_to_rational_l194_19400


namespace final_position_is_negative_one_total_revenue_is_118_yuan_l194_19480

-- Define the distances
def distances : List Int := [9, -3, -6, 4, -8, 6, -3, -6, -4, 10]

-- Define the taxi price per kilometer
def price_per_km : Int := 2

-- Theorem to prove the final position of the taxi relative to Wu Zhong
theorem final_position_is_negative_one : 
  List.sum distances = -1 :=
by 
  sorry -- Proof omitted

-- Theorem to prove the total revenue for the afternoon
theorem total_revenue_is_118_yuan : 
  price_per_km * List.sum (List.map Int.natAbs distances) = 118 :=
by
  sorry -- Proof omitted

end final_position_is_negative_one_total_revenue_is_118_yuan_l194_19480


namespace find_fraction_l194_19401

theorem find_fraction (a b : ℝ) (h₁ : a ≠ b) (h₂ : a / b + (a + 6 * b) / (b + 6 * a) = 2) :
  a / b = 1 / 2 :=
sorry

end find_fraction_l194_19401


namespace problem_ab_value_l194_19462

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x^2 - 4 * x else a * x^2 + b * x

theorem problem_ab_value (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) → a * b = 12 :=
by
  intro h
  let f_eqn := h 1 -- Checking the function equality for x = 1
  sorry

end problem_ab_value_l194_19462


namespace find_g2_l194_19466

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = g x

theorem find_g2 {f g : ℝ → ℝ}
  (h1 : odd_function f)
  (h2 : even_function g)
  (h3 : ∀ x : ℝ, f x + g x = 2^x) :
  g 2 = 17 / 8 :=
sorry

end find_g2_l194_19466


namespace simplify_root_exponentiation_l194_19454

theorem simplify_root_exponentiation : (7 ^ (1 / 3) : ℝ) ^ 6 = 49 := by
  sorry

end simplify_root_exponentiation_l194_19454


namespace optimal_rental_plan_l194_19475

theorem optimal_rental_plan (a b x y : ℕ)
  (h1 : 2 * a + b = 10)
  (h2 : a + 2 * b = 11)
  (h3 : 31 = 3 * x + 4 * y)
  (cost_a : ℕ := 100)
  (cost_b : ℕ := 120) :
  ∃ x y, 3 * x + 4 * y = 31 ∧ cost_a * x + cost_b * y = 940 := by
  sorry

end optimal_rental_plan_l194_19475


namespace mr_callen_total_loss_l194_19433

noncomputable def total_loss : ℤ :=
  let bought_paintings_price := 15 * 60
  let bought_wooden_toys_price := 12 * 25
  let bought_handmade_hats_price := 20 * 15
  let total_bought_price := bought_paintings_price + bought_wooden_toys_price + bought_handmade_hats_price
  let sold_paintings_price := 15 * (60 - (60 * 18 / 100))
  let sold_wooden_toys_price := 12 * (25 - (25 * 25 / 100))
  let sold_handmade_hats_price := 20 * (15 - (15 * 10 / 100))
  let total_sold_price := sold_paintings_price + sold_wooden_toys_price + sold_handmade_hats_price
  total_bought_price - total_sold_price

theorem mr_callen_total_loss : total_loss = 267 := by
  sorry

end mr_callen_total_loss_l194_19433


namespace subtraction_of_fractions_l194_19432

theorem subtraction_of_fractions :
  let S_1 := 3 + 5 + 7
  let S_2 := 2 + 4 + 6
  let S_3 := 2 + 4 + 6
  let S_4 := 3 + 5 + 7
  (S_1 / S_2 - S_3 / S_4) = 9 / 20 :=
by
  let S_1 := 3 + 5 + 7
  let S_2 := 2 + 4 + 6
  let S_3 := 2 + 4 + 6
  let S_4 := 3 + 5 + 7
  sorry

end subtraction_of_fractions_l194_19432


namespace paula_paint_cans_needed_l194_19417

-- Let's define the initial conditions and required computations in Lean.
def initial_rooms : ℕ := 48
def cans_lost : ℕ := 4
def remaining_rooms : ℕ := 36
def large_rooms_to_paint : ℕ := 8
def normal_rooms_to_paint : ℕ := 20
def paint_per_large_room : ℕ := 2 -- as each large room requires twice as much paint

-- Define a function to compute the number of cans required.
def cans_needed (initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room : ℕ) : ℕ :=
  let rooms_lost := initial_rooms - remaining_rooms
  let cans_per_room := rooms_lost / cans_lost
  let total_room_equivalents := large_rooms_to_paint * paint_per_large_room + normal_rooms_to_paint
  total_room_equivalents / cans_per_room

theorem paula_paint_cans_needed : cans_needed initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room = 12 :=
by
  -- The proof would go here
  sorry

end paula_paint_cans_needed_l194_19417


namespace simplify_polynomial_l194_19489

noncomputable def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 4 * r - 3
noncomputable def g (r : ℝ) : ℝ := r^3 + r^2 + 6 * r - 8

theorem simplify_polynomial (r : ℝ) : f r - g r = r^3 - 2 * r + 5 := by
  sorry

end simplify_polynomial_l194_19489


namespace evaluate_expression_l194_19403

theorem evaluate_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ( (1 / a^2 + 1 / b^2)⁻¹ = a^2 * b^2 / (a^2 + b^2) ) :=
by
  sorry

end evaluate_expression_l194_19403


namespace garden_watering_system_pumps_l194_19406

-- Define conditions
def rate := 500 -- gallons per hour
def time := 30 / 60 -- hours, i.e., converting 30 minutes to hours

-- Theorem statement
theorem garden_watering_system_pumps :
  rate * time = 250 := by
  sorry

end garden_watering_system_pumps_l194_19406


namespace geometric_sequence_fourth_term_l194_19467

theorem geometric_sequence_fourth_term (a r T4 : ℝ)
  (h1 : a = 1024)
  (h2 : a * r^5 = 32)
  (h3 : T4 = a * r^3) :
  T4 = 128 :=
by {
  sorry
}

end geometric_sequence_fourth_term_l194_19467


namespace norma_initial_cards_l194_19418

theorem norma_initial_cards (x : ℝ) 
  (H1 : x + 70 = 158) : 
  x = 88 :=
by
  sorry

end norma_initial_cards_l194_19418


namespace hyperbola_foci_l194_19423

-- Define the conditions and the question
def hyperbola_equation (x y : ℝ) : Prop := 
  x^2 - 4 * y^2 - 6 * x + 24 * y - 11 = 0

-- The foci of the hyperbola 
def foci (x1 x2 y1 y2 : ℝ) : Prop := 
  (x1, y1) = (3, 3 + 2 * Real.sqrt 5) ∨ (x2, y2) = (3, 3 - 2 * Real.sqrt 5)

-- The proof statement
theorem hyperbola_foci :
  ∃ x1 x2 y1 y2 : ℝ, hyperbola_equation x1 y1 ∧ foci x1 x2 y1 y2 :=
sorry

end hyperbola_foci_l194_19423


namespace total_snow_volume_l194_19410

theorem total_snow_volume (length width initial_depth additional_depth: ℝ) 
  (h_length : length = 30) 
  (h_width : width = 3) 
  (h_initial_depth : initial_depth = 3 / 4) 
  (h_additional_depth : additional_depth = 1 / 4) : 
  (length * width * initial_depth) + (length * width * additional_depth) = 90 := 
by
  -- proof steps would go here
  sorry

end total_snow_volume_l194_19410


namespace tan_add_pi_div_three_l194_19420

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l194_19420


namespace correct_option_is_C_l194_19435

theorem correct_option_is_C 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (D : Prop)
  (hA : ¬ A)
  (hB : ¬ B)
  (hD : ¬ D)
  (hC : C) :
  C := by
  exact hC

end correct_option_is_C_l194_19435


namespace prime_iff_factorial_mod_l194_19414

theorem prime_iff_factorial_mod (p : ℕ) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end prime_iff_factorial_mod_l194_19414


namespace trapezoid_area_l194_19461

theorem trapezoid_area (base1 base2 height : ℕ) (h_base1 : base1 = 9) (h_base2 : base2 = 11) (h_height : height = 3) :
  (1 / 2 : ℚ) * (base1 + base2 : ℕ) * height = 30 :=
by
  sorry

end trapezoid_area_l194_19461


namespace carnival_activity_order_l194_19485

theorem carnival_activity_order :
  let dodgeball := 3 / 8
  let magic_show := 9 / 24
  let petting_zoo := 1 / 3
  let face_painting := 5 / 12
  let ordered_activities := ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"]
  (face_painting > dodgeball) ∧ (dodgeball = magic_show) ∧ (magic_show > petting_zoo) →
  ordered_activities = ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"] :=
by {
  sorry
}

end carnival_activity_order_l194_19485


namespace remainder_of_55_power_55_plus_55_l194_19496

-- Define the problem statement using Lean

theorem remainder_of_55_power_55_plus_55 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  sorry

end remainder_of_55_power_55_plus_55_l194_19496


namespace relationship_between_a_b_c_l194_19472

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l194_19472


namespace Julia_played_with_kids_on_Monday_l194_19404

theorem Julia_played_with_kids_on_Monday (kids_tuesday : ℕ) (more_kids_monday : ℕ) :
  kids_tuesday = 14 → more_kids_monday = 8 → (kids_tuesday + more_kids_monday = 22) :=
by
  sorry

end Julia_played_with_kids_on_Monday_l194_19404


namespace geometric_progression_product_l194_19459

theorem geometric_progression_product (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
  (h1 : a 3 = a1 * r^2)
  (h2 : a 10 = a1 * r^9)
  (h3 : a1 * r^2 + a1 * r^9 = 3)
  (h4 : a1^2 * r^11 = -5) :
  a 5 * a 8 = -5 :=
by
  sorry

end geometric_progression_product_l194_19459


namespace remainder_of_division_l194_19470

theorem remainder_of_division (x r : ℕ) (h1 : 1620 - x = 1365) (h2 : 1620 = x * 6 + r) : r = 90 :=
sorry

end remainder_of_division_l194_19470


namespace angle_terminal_side_equiv_l194_19448

-- Define the function to check angle equivalence
def angle_equiv (θ₁ θ₂ : ℝ) : Prop := ∃ k : ℤ, θ₁ = θ₂ + k * 360

-- Theorem statement
theorem angle_terminal_side_equiv : angle_equiv 330 (-30) :=
  sorry

end angle_terminal_side_equiv_l194_19448


namespace baseball_cards_per_friend_l194_19440

theorem baseball_cards_per_friend (total_cards : ℕ) (total_friends : ℕ) (h1 : total_cards = 24) (h2 : total_friends = 4) : (total_cards / total_friends) = 6 := 
by
  sorry

end baseball_cards_per_friend_l194_19440


namespace sqrt_floor_eq_l194_19434

theorem sqrt_floor_eq (n : ℕ) (hn : 0 < n) :
    ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by
  sorry

end sqrt_floor_eq_l194_19434


namespace chessboard_game_winner_l194_19474

theorem chessboard_game_winner (m n : ℕ) (initial_position : ℕ × ℕ) :
  (m * n) % 2 = 0 → (∃ A_wins : Prop, A_wins) ∧ 
  (m * n) % 2 = 1 → (∃ B_wins : Prop, B_wins) :=
by
  sorry

end chessboard_game_winner_l194_19474


namespace tangent_circle_equation_l194_19446

theorem tangent_circle_equation :
  (∃ m : Real, ∃ n : Real,
    (∀ x y : Real, (x - m)^2 + (y - n)^2 = 36) ∧ 
    ((m - 0)^2 + (n - 3)^2 = 25) ∧
    n = 6 ∧ (m = 4 ∨ m = -4)) :=
sorry

end tangent_circle_equation_l194_19446


namespace total_price_of_books_l194_19473

theorem total_price_of_books (total_books : ℕ) (math_books : ℕ) (cost_math_book : ℕ) (cost_history_book : ℕ) (remaining_books := total_books - math_books) (total_math_cost := math_books * cost_math_book) (total_history_cost := remaining_books * cost_history_book ) : total_books = 80 → math_books = 27 → cost_math_book = 4 → cost_history_book = 5 → total_math_cost + total_history_cost = 373 :=
by
  intros
  sorry

end total_price_of_books_l194_19473


namespace three_pow_2010_mod_eight_l194_19439

theorem three_pow_2010_mod_eight : (3^2010) % 8 = 1 :=
  sorry

end three_pow_2010_mod_eight_l194_19439


namespace scientific_notation_per_capita_GDP_l194_19405

theorem scientific_notation_per_capita_GDP (GDP : ℝ) (h : GDP = 104000): 
  GDP = 1.04 * 10^5 := 
by
  sorry

end scientific_notation_per_capita_GDP_l194_19405


namespace arrange_BANANA_l194_19429

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l194_19429


namespace max_eggs_l194_19479

theorem max_eggs (x : ℕ) 
  (h1 : x < 200) 
  (h2 : x % 3 = 2) 
  (h3 : x % 4 = 3) 
  (h4 : x % 5 = 4) : 
  x = 179 := 
by
  sorry

end max_eggs_l194_19479


namespace find_C_l194_19482

theorem find_C (A B C D E : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) (h5 : E < 10) 
  (h : 4 * (10 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) + 4) = 400000 + (10000 * A + 1000 * B + 100 * C + 10 * D + E)) : 
  C = 2 :=
sorry

end find_C_l194_19482


namespace probability_two_red_marbles_drawn_l194_19436

/-- A jar contains two red marbles, three green marbles, and ten white marbles and no other marbles.
Two marbles are randomly drawn from this jar without replacement. -/
theorem probability_two_red_marbles_drawn (total_marbles red_marbles green_marbles white_marbles : ℕ)
    (draw_without_replacement : Bool) :
    total_marbles = 15 ∧ red_marbles = 2 ∧ green_marbles = 3 ∧ white_marbles = 10 ∧ draw_without_replacement = true →
    (2 / 15) * (1 / 14) = 1 / 105 :=
by
  intro h
  sorry

end probability_two_red_marbles_drawn_l194_19436


namespace num_arrangements_with_ab_together_l194_19408

theorem num_arrangements_with_ab_together (products : Fin 5 → Type) :
  (∃ A B : Fin 5 → Type, A ≠ B) →
  ∃ (n : ℕ), n = 48 :=
by
  sorry

end num_arrangements_with_ab_together_l194_19408


namespace triangle_length_AX_l194_19469

theorem triangle_length_AX (A B C X : Type*) (AB AC BC AX XB : ℝ)
  (hAB : AB = 70) (hAC : AC = 42) (hBC : BC = 56)
  (h_bisect : ∃ (k : ℝ), AX = 3 * k ∧ XB = 4 * k) :
  AX = 30 := 
by
  sorry

end triangle_length_AX_l194_19469


namespace smallest_value_of_a_l194_19453

theorem smallest_value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : 2 * b = a + c) (h4 : c^2 = a * b) : a = -4 :=
by
  sorry

end smallest_value_of_a_l194_19453


namespace neg_sqrt_17_bounds_l194_19450

theorem neg_sqrt_17_bounds :
  (16 < 17) ∧ (17 < 25) ∧ (16 = 4^2) ∧ (25 = 5^2) ∧ (4 < Real.sqrt 17) ∧ (Real.sqrt 17 < 5) →
  (-5 < -Real.sqrt 17) ∧ (-Real.sqrt 17 < -4) :=
by
  sorry

end neg_sqrt_17_bounds_l194_19450


namespace problem_concentric_circles_chord_probability_l194_19415

open ProbabilityTheory

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h : r1 < r2) : ℝ :=
1/6

theorem problem_concentric_circles_chord_probability :
  probability_chord_intersects_inner_circle 1.5 3 
  (by norm_num) = 1/6 :=
sorry

end problem_concentric_circles_chord_probability_l194_19415


namespace gcd_incorrect_l194_19451

theorem gcd_incorrect (a b c : ℕ) (h : a * b * c = 3000) : gcd (gcd a b) c ≠ 15 := 
sorry

end gcd_incorrect_l194_19451


namespace sum_of_arithmetic_sequence_l194_19416

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : a 5 + a 4 = 18) (hS_def : ∀ n, S n = n * (a 1 + a n) / 2) : S 8 = 72 := 
sorry

end sum_of_arithmetic_sequence_l194_19416


namespace basketball_committee_l194_19428

theorem basketball_committee (total_players guards : ℕ) (choose_committee choose_guard : ℕ) :
  total_players = 12 → guards = 4 → choose_committee = 3 → choose_guard = 1 →
  (guards * ((total_players - guards).choose (choose_committee - choose_guard)) = 112) :=
by
  intros h_tp h_g h_cc h_cg
  rw [h_tp, h_g, h_cc, h_cg]
  simp
  norm_num
  sorry

end basketball_committee_l194_19428


namespace problem_omega_pow_l194_19425

noncomputable def omega : ℂ := Complex.I -- Define a non-real root for x² = 1; an example choice could be i, the imaginary unit.

theorem problem_omega_pow :
  omega^2 = 1 → 
  (1 - omega + omega^2)^6 + (1 + omega - omega^2)^6 = 730 := 
by
  intro h1
  -- proof steps omitted
  sorry

end problem_omega_pow_l194_19425


namespace find_d_l194_19430

variable {x1 x2 k d : ℝ}

axiom h₁ : x1 ≠ x2
axiom h₂ : 4 * x1^2 - k * x1 = d
axiom h₃ : 4 * x2^2 - k * x2 = d
axiom h₄ : x1 + x2 = 2

theorem find_d : d = -12 := by
  sorry

end find_d_l194_19430


namespace coordinates_of_point_P_l194_19452

theorem coordinates_of_point_P :
  ∀ (P : ℝ × ℝ), (P.1, P.2) = -1 ∧ (P.2 = -Real.sqrt 3) :=
by
  sorry

end coordinates_of_point_P_l194_19452


namespace log_arith_example_l194_19458

noncomputable def log10 (x : ℝ) : ℝ := sorry -- Assume the definition of log base 10

theorem log_arith_example : log10 4 + 2 * log10 5 + 8^(2/3) = 6 := 
by
  -- The proof would go here
  sorry

end log_arith_example_l194_19458


namespace Ali_possible_scores_l194_19422

-- Defining the conditions
def categories := 5
def questions_per_category := 3
def correct_answers_points := 12
def total_questions := categories * questions_per_category
def incorrect_answers := total_questions - correct_answers_points

-- Defining the bonuses based on cases

-- All 3 incorrect answers in 1 category
def case_1_bonus := 4
def case_1_total := correct_answers_points + case_1_bonus

-- 3 incorrect answers split into 2 categories
def case_2_bonus := 3
def case_2_total := correct_answers_points + case_2_bonus

-- 3 incorrect answers split into 3 categories
def case_3_bonus := 2
def case_3_total := correct_answers_points + case_3_bonus

theorem Ali_possible_scores : 
  case_1_total = 16 ∧ case_2_total = 15 ∧ case_3_total = 14 :=
by
  -- Skipping the proof here
  sorry

end Ali_possible_scores_l194_19422


namespace greatest_prime_factor_187_l194_19443

theorem greatest_prime_factor_187 : ∃ p : ℕ, Prime p ∧ p ∣ 187 ∧ ∀ q : ℕ, Prime q ∧ q ∣ 187 → p ≥ q := by
  sorry

end greatest_prime_factor_187_l194_19443


namespace initial_unread_messages_correct_l194_19437

-- Definitions based on conditions
def messages_read_per_day := 20
def messages_new_per_day := 6
def duration_in_days := 7
def effective_reading_rate := messages_read_per_day - messages_new_per_day

-- The initial number of unread messages
def initial_unread_messages := duration_in_days * effective_reading_rate

-- The theorem we want to prove
theorem initial_unread_messages_correct :
  initial_unread_messages = 98 :=
sorry

end initial_unread_messages_correct_l194_19437


namespace point_P_on_x_axis_l194_19468

noncomputable def point_on_x_axis (m : ℝ) : ℝ × ℝ := (4, m + 1)

theorem point_P_on_x_axis (m : ℝ) (h : point_on_x_axis m = (4, 0)) : m = -1 := 
by
  sorry

end point_P_on_x_axis_l194_19468


namespace equilateral_prism_lateral_edge_length_l194_19477

theorem equilateral_prism_lateral_edge_length
  (base_side_length : ℝ)
  (h_base : base_side_length = 1)
  (perpendicular_diagonals : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = base_side_length ∧ b = lateral_edge ∧ c = some_diagonal_length ∧ lateral_edge ≠ 0)
  : ∀ lateral_edge : ℝ, lateral_edge = (Real.sqrt 2) / 2 := sorry

end equilateral_prism_lateral_edge_length_l194_19477


namespace calculate_adult_chaperones_l194_19407

theorem calculate_adult_chaperones (students : ℕ) (student_fee : ℕ) (adult_fee : ℕ) (total_fee : ℕ) 
  (h_students : students = 35) 
  (h_student_fee : student_fee = 5) 
  (h_adult_fee : adult_fee = 6) 
  (h_total_fee : total_fee = 199) : 
  ∃ (A : ℕ), 35 * student_fee + A * adult_fee = 199 ∧ A = 4 := 
by
  sorry

end calculate_adult_chaperones_l194_19407


namespace line_eq_l194_19481

theorem line_eq (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 = 5 ∧ y1 = 0 ∧ x2 = 2 ∧ y2 = -5 ∧
    (y - y1) / (x - x1) = (y2 - y1) / (x2 - x1)) →
  5 * x - 3 * y - 25 = 0 :=
sorry

end line_eq_l194_19481


namespace canoe_rental_cost_l194_19456

theorem canoe_rental_cost :
  ∃ (C : ℕ) (K : ℕ), 
  (15 * K + C * (K + 4) = 288) ∧ 
  (3 * K + 12 = 12 * C) ∧ 
  (C = 14) :=
sorry

end canoe_rental_cost_l194_19456


namespace g_function_expression_l194_19488

theorem g_function_expression (f g : ℝ → ℝ) (a : ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, g (-x) = g x) (h3 : ∀ x : ℝ, f x + g x = x^2 + a * x + 2 * a - 1) (h4 : f 1 = 2) :
  ∀ t : ℝ, g t = t^2 + 4 * t - 1 :=
by
  sorry

end g_function_expression_l194_19488


namespace possible_values_2n_plus_m_l194_19413

theorem possible_values_2n_plus_m :
  ∀ (n m : ℤ), 3 * n - m < 5 → n + m > 26 → 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
by sorry

end possible_values_2n_plus_m_l194_19413


namespace find_b_l194_19497

variable {a b c : ℚ}

theorem find_b (h1 : a + b + c = 117) (h2 : a + 8 = 4 * c) (h3 : b - 10 = 4 * c) : b = 550 / 9 := by
  sorry

end find_b_l194_19497


namespace max_largest_integer_l194_19445

theorem max_largest_integer (A B C D E : ℕ) (h₀ : A ≤ B) (h₁ : B ≤ C) (h₂ : C ≤ D) (h₃ : D ≤ E) 
(h₄ : (A + B + C + D + E) = 225) (h₅ : E - A = 10) : E = 215 :=
sorry

end max_largest_integer_l194_19445


namespace num_three_digit_integers_with_odd_factors_l194_19490

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l194_19490


namespace shortest_paths_ratio_l194_19460

theorem shortest_paths_ratio (k n : ℕ) (h : k > 0):
  let paths_along_AB := Nat.choose (k * n + n - 1) (n - 1)
  let paths_along_AD := Nat.choose (k * n + n - 1) k * n - 1
  paths_along_AD = k * paths_along_AB :=
by sorry

end shortest_paths_ratio_l194_19460


namespace exists_m_divisible_l194_19402

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x + 2

-- Define the 100th iterate of f
def f_iter (n : ℕ) : ℕ := 3^n

-- Define the condition that needs to be proven
theorem exists_m_divisible : ∃ m : ℕ, (3^100 * m + (3^100 - 1)) % 1988 = 0 :=
sorry

end exists_m_divisible_l194_19402


namespace bus_speed_excluding_stoppages_l194_19424

variable (v : ℝ) -- Speed of the bus excluding stoppages

-- Conditions
def bus_stops_per_hour := 45 / 60 -- 45 minutes converted to hours
def effective_driving_time := 1 - bus_stops_per_hour -- Effective time driving in an hour

-- Given Condition
def speed_including_stoppages := 12 -- Speed including stoppages in km/hr

theorem bus_speed_excluding_stoppages 
  (h : effective_driving_time * v = speed_including_stoppages) : 
  v = 48 :=
sorry

end bus_speed_excluding_stoppages_l194_19424


namespace my_op_example_l194_19455

def my_op (a b : Int) : Int := a^2 - abs b

theorem my_op_example : my_op (-2) (-1) = 3 := by
  sorry

end my_op_example_l194_19455


namespace sum_of_35_consecutive_squares_div_by_35_l194_19426

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_35_consecutive_squares_div_by_35 (n : ℕ) :
  (sum_of_squares (n + 35) - sum_of_squares n) % 35 = 0 :=
by
  sorry

end sum_of_35_consecutive_squares_div_by_35_l194_19426


namespace pentagon_largest_angle_l194_19486

theorem pentagon_largest_angle
    (P Q : ℝ)
    (hP : P = 55)
    (hQ : Q = 120)
    (R S T : ℝ)
    (hR_eq_S : R = S)
    (hT : T = 2 * R + 20):
    R + S + T + P + Q = 540 → T = 192.5 :=
by
    sorry

end pentagon_largest_angle_l194_19486


namespace fraction_decimal_representation_l194_19492

noncomputable def fraction_as_term_dec : ℚ := 47 / (2^3 * 5^4)

theorem fraction_decimal_representation : fraction_as_term_dec = 0.0094 :=
by
  sorry

end fraction_decimal_representation_l194_19492


namespace remainder_is_4_over_3_l194_19465

noncomputable def original_polynomial (z : ℝ) : ℝ := 3 * z ^ 3 - 4 * z ^ 2 - 14 * z + 3
noncomputable def divisor (z : ℝ) : ℝ := 3 * z + 5
noncomputable def quotient (z : ℝ) : ℝ := z ^ 2 - 3 * z + 1 / 3

theorem remainder_is_4_over_3 :
  ∃ r : ℝ, original_polynomial z = divisor z * quotient z + r ∧ r = 4 / 3 :=
sorry

end remainder_is_4_over_3_l194_19465


namespace both_inequalities_equiv_l194_19493

theorem both_inequalities_equiv (x : ℝ) : (x - 3)/(2 - x) ≥ 0 ↔ (3 - x)/(x - 2) ≥ 0 := by
  sorry

end both_inequalities_equiv_l194_19493


namespace race_result_l194_19411

theorem race_result
    (distance_race : ℕ)
    (distance_diff : ℕ)
    (distance_second_start_diff : ℕ)
    (speed_xm speed_xl : ℕ)
    (h1 : distance_race = 100)
    (h2 : distance_diff = 20)
    (h3 : distance_second_start_diff = 20)
    (xm_wins_first_race : speed_xm * distance_race >= speed_xl * (distance_race - distance_diff)) :
    speed_xm * (distance_race + distance_second_start_diff) >= speed_xl * (distance_race + distance_diff) :=
by
  sorry

end race_result_l194_19411


namespace tan_subtraction_example_l194_19427

noncomputable def tan_subtraction_identity (alpha beta : ℝ) : ℝ :=
  (Real.tan alpha - Real.tan beta) / (1 + Real.tan alpha * Real.tan beta)

theorem tan_subtraction_example (theta : ℝ) (h : Real.tan theta = 1 / 2) :
  Real.tan (π / 4 - theta) = 1 / 3 := 
by
  sorry

end tan_subtraction_example_l194_19427
