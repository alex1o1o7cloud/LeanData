import Mathlib

namespace NUMINAMATH_GPT_average_of_new_sequence_l498_49815

theorem average_of_new_sequence (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end NUMINAMATH_GPT_average_of_new_sequence_l498_49815


namespace NUMINAMATH_GPT_area_of_triangle_PQR_l498_49852

noncomputable def point := ℝ × ℝ

def P : point := (1, 1)
def Q : point := (4, 1)
def R : point := (3, 4)

def triangle_area (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)

theorem area_of_triangle_PQR :
  triangle_area P Q R = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_l498_49852


namespace NUMINAMATH_GPT_fourth_guard_ran_150_meters_l498_49803

def rectangle_width : ℕ := 200
def rectangle_length : ℕ := 300
def total_perimeter : ℕ := 2 * (rectangle_width + rectangle_length)
def three_guards_total_distance : ℕ := 850

def fourth_guard_distance : ℕ := total_perimeter - three_guards_total_distance

theorem fourth_guard_ran_150_meters :
  fourth_guard_distance = 150 :=
by
  -- calculation skipped here
  -- proving fourth_guard_distance as derived being 150 meters
  sorry

end NUMINAMATH_GPT_fourth_guard_ran_150_meters_l498_49803


namespace NUMINAMATH_GPT_shaded_L_area_l498_49851

theorem shaded_L_area 
  (s₁ s₂ s₃ s₄ : ℕ)
  (hA : s₁ = 2)
  (hB : s₂ = 2)
  (hC : s₃ = 3)
  (hD : s₄ = 3)
  (side_ABC : ℕ := 6)
  (area_ABC : ℕ := side_ABC * side_ABC) : 
  area_ABC - (s₁ * s₁ + s₂ * s₂ + s₃ * s₃ + s₄ * s₄) = 10 :=
sorry

end NUMINAMATH_GPT_shaded_L_area_l498_49851


namespace NUMINAMATH_GPT_value_of_p_h_3_l498_49865

-- Define the functions h and p
def h (x : ℝ) : ℝ := 4 * x + 5
def p (x : ℝ) : ℝ := 6 * x - 11

-- Statement to prove
theorem value_of_p_h_3 : p (h 3) = 91 := sorry

end NUMINAMATH_GPT_value_of_p_h_3_l498_49865


namespace NUMINAMATH_GPT_product_modulo_7_l498_49857

theorem product_modulo_7 : (1729 * 1865 * 1912 * 2023) % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_product_modulo_7_l498_49857


namespace NUMINAMATH_GPT_angle_A_and_area_of_triangle_l498_49897

theorem angle_A_and_area_of_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) (h1 : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) 
(h2 : R = 2) (h3 : b^2 + c^2 = 18) :
  A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_angle_A_and_area_of_triangle_l498_49897


namespace NUMINAMATH_GPT_diophantine_solution_exists_if_prime_divisor_l498_49882

theorem diophantine_solution_exists_if_prime_divisor (b : ℕ) (hb : 0 < b) (gcd_b_6 : Nat.gcd b 6 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 3 / (b : ℚ))) ↔ 
  ∃ p : ℕ, Nat.Prime p ∧ (∃ k : ℕ, p = 6 * k - 1) ∧ p ∣ b := 
by 
  sorry

end NUMINAMATH_GPT_diophantine_solution_exists_if_prime_divisor_l498_49882


namespace NUMINAMATH_GPT_sum_of_n_natural_numbers_l498_49869

theorem sum_of_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 1035) : n = 46 :=
sorry

end NUMINAMATH_GPT_sum_of_n_natural_numbers_l498_49869


namespace NUMINAMATH_GPT_hundredth_odd_integer_not_divisible_by_five_l498_49801

def odd_positive_integer (n : ℕ) : ℕ := 2 * n - 1

theorem hundredth_odd_integer_not_divisible_by_five :
  odd_positive_integer 100 = 199 ∧ ¬ (199 % 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_hundredth_odd_integer_not_divisible_by_five_l498_49801


namespace NUMINAMATH_GPT_find_stu_l498_49823

open Complex

theorem find_stu (p q r s t u : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (h1 : p = (q + r) / (s - 3))
  (h2 : q = (p + r) / (t - 3))
  (h3 : r = (p + q) / (u - 3))
  (h4 : s * t + s * u + t * u = 8)
  (h5 : s + t + u = 4) :
  s * t * u = 10 := 
sorry

end NUMINAMATH_GPT_find_stu_l498_49823


namespace NUMINAMATH_GPT_number_of_players_per_game_l498_49861

def total_players : ℕ := 50
def total_games : ℕ := 1225

-- If each player plays exactly one game with each of the other players,
-- there are C(total_players, 2) = total_games games.
theorem number_of_players_per_game : ∃ k : ℕ, k = 2 ∧ (total_players * (total_players - 1)) / 2 = total_games := 
  sorry

end NUMINAMATH_GPT_number_of_players_per_game_l498_49861


namespace NUMINAMATH_GPT_largest_hexagon_angle_l498_49820

theorem largest_hexagon_angle (x : ℝ) : 
  (2 * x + 2 * x + 2 * x + 3 * x + 4 * x + 5 * x = 720) → (5 * x = 200) := by
  sorry

end NUMINAMATH_GPT_largest_hexagon_angle_l498_49820


namespace NUMINAMATH_GPT_jane_percentage_bread_to_treats_l498_49817

variable (T J_b W_b W_t : ℕ) (P : ℕ)

-- Conditions as stated
axiom h1 : J_b = (P * T) / 100
axiom h2 : W_t = T / 2
axiom h3 : W_b = 3 * W_t
axiom h4 : W_b = 90
axiom h5 : J_b + W_b + T + W_t = 225

theorem jane_percentage_bread_to_treats : P = 75 :=
by
-- Proof skeleton
sorry

end NUMINAMATH_GPT_jane_percentage_bread_to_treats_l498_49817


namespace NUMINAMATH_GPT_multiples_count_l498_49826

theorem multiples_count (count_5 count_7 count_35 count_total : ℕ) :
  count_5 = 600 →
  count_7 = 428 →
  count_35 = 85 →
  count_total = count_5 + count_7 - count_35 →
  count_total = 943 :=
by
  sorry

end NUMINAMATH_GPT_multiples_count_l498_49826


namespace NUMINAMATH_GPT_function_intersects_line_at_most_once_l498_49899

variable {α β : Type} [Nonempty α]

def function_intersects_at_most_once (f : α → β) (a : α) : Prop :=
  ∀ (b b' : β), f a = b → f a = b' → b = b'

theorem function_intersects_line_at_most_once {α β : Type} [Nonempty α] (f : α → β) (a : α) :
  function_intersects_at_most_once f a :=
by
  sorry

end NUMINAMATH_GPT_function_intersects_line_at_most_once_l498_49899


namespace NUMINAMATH_GPT_quiz_score_difference_l498_49862

theorem quiz_score_difference :
  let percentage_70 := 0.10
  let percentage_80 := 0.35
  let percentage_90 := 0.30
  let percentage_100 := 0.25
  let mean_score := (percentage_70 * 70) + (percentage_80 * 80) + (percentage_90 * 90) + (percentage_100 * 100)
  let median_score := 90
  mean_score = 87 → median_score - mean_score = 3 :=
by
  sorry

end NUMINAMATH_GPT_quiz_score_difference_l498_49862


namespace NUMINAMATH_GPT_ethanol_in_tank_l498_49866

theorem ethanol_in_tank (capacity fuel_a fuel_b : ℝ)
  (ethanol_a ethanol_b : ℝ)
  (h1 : capacity = 218)
  (h2 : fuel_a = 122)
  (h3 : fuel_b = capacity - fuel_a)
  (h4 : ethanol_a = 0.12)
  (h5 : ethanol_b = 0.16) :
  fuel_a * ethanol_a + fuel_b * ethanol_b = 30 := 
by {
  sorry
}

end NUMINAMATH_GPT_ethanol_in_tank_l498_49866


namespace NUMINAMATH_GPT_chips_per_cookie_l498_49894

theorem chips_per_cookie (total_cookies : ℕ) (uneaten_chips : ℕ) (uneaten_cookies : ℕ) (h1 : total_cookies = 4 * 12) (h2 : uneaten_cookies = total_cookies / 2) (h3 : uneaten_chips = 168) : 
  uneaten_chips / uneaten_cookies = 7 :=
by sorry

end NUMINAMATH_GPT_chips_per_cookie_l498_49894


namespace NUMINAMATH_GPT_bank_deposit_exceeds_1000_on_saturday_l498_49886

theorem bank_deposit_exceeds_1000_on_saturday:
  ∃ n: ℕ, (2 * (3^n - 1) / 2 > 1000) ∧ ((n + 1) % 7 = 0) := by
  sorry

end NUMINAMATH_GPT_bank_deposit_exceeds_1000_on_saturday_l498_49886


namespace NUMINAMATH_GPT_pool_capacity_is_80_percent_l498_49843

noncomputable def current_capacity_percentage (width length depth rate time : ℝ) : ℝ :=
  let total_volume := width * length * depth
  let water_removed := rate * time
  (water_removed / total_volume) * 100

theorem pool_capacity_is_80_percent :
  current_capacity_percentage 50 150 10 60 1000 = 80 :=
by
  sorry

end NUMINAMATH_GPT_pool_capacity_is_80_percent_l498_49843


namespace NUMINAMATH_GPT_total_toys_is_60_l498_49833

def toy_cars : Nat := 20
def toy_soldiers : Nat := 2 * toy_cars
def total_toys : Nat := toy_cars + toy_soldiers

theorem total_toys_is_60 : total_toys = 60 := by
  sorry

end NUMINAMATH_GPT_total_toys_is_60_l498_49833


namespace NUMINAMATH_GPT_power_of_10_digits_l498_49877

theorem power_of_10_digits (n : ℕ) (hn : n > 1) :
  (∃ k : ℕ, (2^(n-1) < 10^k ∧ 10^k < 2^n) ∨ (5^(n-1) < 10^k ∧ 10^k < 5^n)) ∧ ¬((∃ k : ℕ, 2^(n-1) < 10^k ∧ 10^k < 2^n) ∧ (∃ k : ℕ, 5^(n-1) < 10^k ∧ 10^k < 5^n)) :=
sorry

end NUMINAMATH_GPT_power_of_10_digits_l498_49877


namespace NUMINAMATH_GPT_smallest_solution_of_abs_eq_l498_49891

theorem smallest_solution_of_abs_eq (x : ℝ) : 
  (x * |x| = 3 * x + 2 → x ≥ 0 → x = (3 + Real.sqrt 17) / 2) ∧
  (x * |x| = 3 * x + 2 → x < 0 → x = -2) ∧
  (x * |x| = 3 * x + 2 → x = -2 → x = -2) :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_of_abs_eq_l498_49891


namespace NUMINAMATH_GPT_lines_in_4_by_4_grid_l498_49844

-- Definition for the grid and the number of lattice points.
def grid : Nat := 16

-- Theorem stating that the number of different lines passing through at least two points in a 4-by-4 grid of lattice points.
theorem lines_in_4_by_4_grid : 
  (number_of_lines : Nat) → number_of_lines = 40 ↔ grid = 16 := 
by
  -- Calculating number of lines passing through at least two points in a 4-by-4 grid.
  sorry -- proof skipped

end NUMINAMATH_GPT_lines_in_4_by_4_grid_l498_49844


namespace NUMINAMATH_GPT_find_FC_l498_49867

theorem find_FC
  (DC : ℝ) (CB : ℝ) (AD : ℝ)
  (hDC : DC = 9) (hCB : CB = 10)
  (hAB : ∃ (k1 : ℝ), k1 = 1/5 ∧ AB = k1 * AD)
  (hED : ∃ (k2 : ℝ), k2 = 3/4 ∧ ED = k2 * AD) :
  ∃ FC : ℝ, FC = 11.025 :=
by
  sorry

end NUMINAMATH_GPT_find_FC_l498_49867


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l498_49896

theorem molecular_weight_of_one_mole (total_molecular_weight : ℝ) (number_of_moles : ℕ) (h1 : total_molecular_weight = 304) (h2 : number_of_moles = 4) : 
  total_molecular_weight / number_of_moles = 76 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l498_49896


namespace NUMINAMATH_GPT_find_ice_cream_cost_l498_49821

def cost_of_ice_cream (total_paid cost_chapati cost_rice cost_vegetable : ℕ) (n_chapatis n_rice n_vegetables n_ice_cream : ℕ) : ℕ :=
  (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetables * cost_vegetable)) / n_ice_cream

theorem find_ice_cream_cost :
  let total_paid := 1051
  let cost_chapati := 6
  let cost_rice := 45
  let cost_vegetable := 70
  let n_chapatis := 16
  let n_rice := 5
  let n_vegetables := 7
  let n_ice_cream := 6
  cost_of_ice_cream total_paid cost_chapati cost_rice cost_vegetable n_chapatis n_rice n_vegetables n_ice_cream = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_ice_cream_cost_l498_49821


namespace NUMINAMATH_GPT_garage_sale_items_l498_49832

-- Definition of conditions
def is_18th_highest (num_highest: ℕ) : Prop := num_highest = 17
def is_25th_lowest (num_lowest: ℕ) : Prop := num_lowest = 24

-- Theorem statement
theorem garage_sale_items (num_highest num_lowest total_items: ℕ) 
  (h1: is_18th_highest num_highest) (h2: is_25th_lowest num_lowest) :
  total_items = num_highest + num_lowest + 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_garage_sale_items_l498_49832


namespace NUMINAMATH_GPT_current_inventory_l498_49842

noncomputable def initial_books : ℕ := 743
noncomputable def fiction_books : ℕ := 520
noncomputable def nonfiction_books : ℕ := 123
noncomputable def children_books : ℕ := 100

noncomputable def saturday_instore_sales : ℕ := 37
noncomputable def saturday_fiction_sales : ℕ := 15
noncomputable def saturday_nonfiction_sales : ℕ := 12
noncomputable def saturday_children_sales : ℕ := 10
noncomputable def saturday_online_sales : ℕ := 128

noncomputable def sunday_instore_multiplier : ℕ := 2
noncomputable def sunday_online_addition : ℕ := 34

noncomputable def new_shipment : ℕ := 160

noncomputable def current_books := 
  initial_books 
  - (saturday_instore_sales + saturday_online_sales)
  - (sunday_instore_multiplier * saturday_instore_sales + saturday_online_sales + sunday_online_addition)
  + new_shipment

theorem current_inventory : current_books = 502 := by
  sorry

end NUMINAMATH_GPT_current_inventory_l498_49842


namespace NUMINAMATH_GPT_right_handed_players_total_l498_49868

-- Definitions of the given quantities
def total_players : ℕ := 70
def throwers : ℕ := 49
def non_throwers : ℕ := total_players - throwers
def one_third_non_throwers : ℕ := non_throwers / 3
def left_handed_non_throwers : ℕ := one_third_non_throwers
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers

-- The theorem stating the main proof goal
theorem right_handed_players_total (h1 : total_players = 70)
                                   (h2 : throwers = 49)
                                   (h3 : total_players - throwers = non_throwers)
                                   (h4 : non_throwers = 21) -- derived from the above
                                   (h5 : non_throwers / 3 = left_handed_non_throwers)
                                   (h6 : non_throwers - left_handed_non_throwers = right_handed_non_throwers)
                                   (h7 : right_handed_throwers = throwers)
                                   (h8 : total_right_handed = right_handed_throwers + right_handed_non_throwers) :
  total_right_handed = 63 := sorry

end NUMINAMATH_GPT_right_handed_players_total_l498_49868


namespace NUMINAMATH_GPT_length_each_stitch_l498_49800

theorem length_each_stitch 
  (hem_length_feet : ℝ) 
  (stitches_per_minute : ℝ) 
  (hem_time_minutes : ℝ) 
  (hem_length_inches : ℝ) 
  (total_stitches : ℝ) 
  (stitch_length_inches : ℝ) 
  (h1 : hem_length_feet = 3) 
  (h2 : stitches_per_minute = 24) 
  (h3 : hem_time_minutes = 6) 
  (h4 : hem_length_inches = hem_length_feet * 12) 
  (h5 : total_stitches = stitches_per_minute * hem_time_minutes) 
  (h6 : stitch_length_inches = hem_length_inches / total_stitches) :
  stitch_length_inches = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_length_each_stitch_l498_49800


namespace NUMINAMATH_GPT_packages_bought_l498_49827

theorem packages_bought (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 426) (h2 : tshirts_per_package = 6) : 
  (total_tshirts / tshirts_per_package) = 71 :=
by 
  sorry

end NUMINAMATH_GPT_packages_bought_l498_49827


namespace NUMINAMATH_GPT_major_premise_incorrect_l498_49898

theorem major_premise_incorrect (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
    ¬ (∀ x y : ℝ, x < y → a^x < a^y) :=
by {
  sorry
}

end NUMINAMATH_GPT_major_premise_incorrect_l498_49898


namespace NUMINAMATH_GPT_trigonometric_identity_l498_49810

theorem trigonometric_identity :
  (1 - Real.sin (Real.pi / 6)) * (1 - Real.sin (5 * Real.pi / 6)) = 1 / 4 :=
by
  have h1 : Real.sin (5 * Real.pi / 6) = Real.sin (Real.pi - Real.pi / 6) := by sorry
  have h2 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l498_49810


namespace NUMINAMATH_GPT_total_current_ages_l498_49895

theorem total_current_ages (T : ℕ) : (T - 12 = 54) → T = 66 :=
by
  sorry

end NUMINAMATH_GPT_total_current_ages_l498_49895


namespace NUMINAMATH_GPT_smallest_angle_between_radii_l498_49805

theorem smallest_angle_between_radii (n : ℕ) (k : ℕ) (angle_step : ℕ) (angle_smallest : ℕ) 
(h_n : n = 40) 
(h_k : k = 23) 
(h_angle_step : angle_step = k) 
(h_angle_smallest : angle_smallest = 23) : 
angle_smallest = 23 :=
sorry

end NUMINAMATH_GPT_smallest_angle_between_radii_l498_49805


namespace NUMINAMATH_GPT_painters_time_l498_49807

-- Define the initial conditions
def n1 : ℕ := 3
def d1 : ℕ := 2
def W := n1 * d1
def n2 : ℕ := 2
def d2 := W / n2
def d_r := (3 * d2) / 4

-- Theorem statement
theorem painters_time (h : d_r = 9 / 4) : d_r = 9 / 4 := by
  sorry

end NUMINAMATH_GPT_painters_time_l498_49807


namespace NUMINAMATH_GPT_expand_and_simplify_product_l498_49814

theorem expand_and_simplify_product :
  5 * (x + 6) * (x + 2) * (x + 7) = 5 * x^3 + 75 * x^2 + 340 * x + 420 := 
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_product_l498_49814


namespace NUMINAMATH_GPT_ratio_of_height_and_radius_l498_49864

theorem ratio_of_height_and_radius 
  (h r : ℝ) 
  (V_X V_Y : ℝ)
  (hY rY : ℝ)
  (k : ℝ)
  (h_def : V_X = π * r^2 * h)
  (hY_def : hY = k * h)
  (rY_def : rY = k * r)
  (half_filled_VY : V_Y = 1/2 * π * rY^2 * hY)
  (V_X_value : V_X = 2)
  (V_Y_value : V_Y = 64):
  k = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_height_and_radius_l498_49864


namespace NUMINAMATH_GPT_value_of_a3_plus_a5_l498_49834

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_a3_plus_a5 (a : ℕ → α) (S : ℕ → α)
  (h_sequence : arithmetic_sequence a)
  (h_S7 : S 7 = 14)
  (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 3 + a 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a3_plus_a5_l498_49834


namespace NUMINAMATH_GPT_sector_arc_length_l498_49848

theorem sector_arc_length (n : ℝ) (r : ℝ) (l : ℝ) (h1 : n = 90) (h2 : r = 3) (h3 : l = (n * Real.pi * r) / 180) :
  l = (3 / 2) * Real.pi := by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_sector_arc_length_l498_49848


namespace NUMINAMATH_GPT_continuous_polynomial_continuous_cosecant_l498_49812

-- Prove that the function \( f(x) = 2x^2 - 1 \) is continuous on \(\mathbb{R}\)
theorem continuous_polynomial : Continuous (fun x : ℝ => 2 * x^2 - 1) :=
sorry

-- Prove that the function \( g(x) = (\sin x)^{-1} \) is continuous on \(\mathbb{R}\) \setminus \(\{ k\pi \mid k \in \mathbb{Z} \} \)
theorem continuous_cosecant : ∀ x : ℝ, x ∉ Set.range (fun k : ℤ => k * Real.pi) → ContinuousAt (fun x : ℝ => (Real.sin x)⁻¹) x :=
sorry

end NUMINAMATH_GPT_continuous_polynomial_continuous_cosecant_l498_49812


namespace NUMINAMATH_GPT_find_prime_p_l498_49813

theorem find_prime_p (p x y : ℕ) (hp : Nat.Prime p) (hx : x > 0) (hy : y > 0) :
  (p + 49 = 2 * x^2) ∧ (p^2 + 49 = 2 * y^2) ↔ p = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_prime_p_l498_49813


namespace NUMINAMATH_GPT_add_A_to_10_eq_15_l498_49880

theorem add_A_to_10_eq_15 (A : ℕ) (h : A + 10 = 15) : A = 5 :=
sorry

end NUMINAMATH_GPT_add_A_to_10_eq_15_l498_49880


namespace NUMINAMATH_GPT_correct_student_mark_l498_49890

theorem correct_student_mark
  (avg_wrong : ℕ) (num_students : ℕ) (wrong_mark : ℕ) (avg_correct : ℕ)
  (h1 : num_students = 10) (h2 : avg_wrong = 100) (h3 : wrong_mark = 90) (h4 : avg_correct = 92) :
  ∃ (x : ℕ), x = 10 :=
by
  sorry

end NUMINAMATH_GPT_correct_student_mark_l498_49890


namespace NUMINAMATH_GPT_perimeter_of_one_of_the_rectangles_l498_49871

noncomputable def perimeter_of_rectangle (z w : ℕ) : ℕ :=
  2 * z

theorem perimeter_of_one_of_the_rectangles (z w : ℕ) :
  ∃ P, P = perimeter_of_rectangle z w :=
by
  use 2 * z
  sorry

end NUMINAMATH_GPT_perimeter_of_one_of_the_rectangles_l498_49871


namespace NUMINAMATH_GPT_simplify_trig_expr_l498_49802

noncomputable def sin15 := Real.sin (Real.pi / 12)
noncomputable def sin30 := Real.sin (Real.pi / 6)
noncomputable def sin45 := Real.sin (Real.pi / 4)
noncomputable def sin60 := Real.sin (Real.pi / 3)
noncomputable def sin75 := Real.sin (5 * Real.pi / 12)
noncomputable def cos10 := Real.cos (Real.pi / 18)
noncomputable def cos20 := Real.cos (Real.pi / 9)
noncomputable def cos30 := Real.cos (Real.pi / 6)

theorem simplify_trig_expr :
  (sin15 + sin30 + sin45 + sin60 + sin75) / (cos10 * cos20 * cos30) = 5.128 :=
sorry

end NUMINAMATH_GPT_simplify_trig_expr_l498_49802


namespace NUMINAMATH_GPT_martha_saving_l498_49811

-- Definitions for the conditions
def daily_allowance : ℕ := 12
def half_daily_allowance : ℕ := daily_allowance / 2
def quarter_daily_allowance : ℕ := daily_allowance / 4
def days_saving_half : ℕ := 6
def day_saving_quarter : ℕ := 1

-- Statement to be proved
theorem martha_saving :
  (days_saving_half * half_daily_allowance) + (day_saving_quarter * quarter_daily_allowance) = 39 := by
  sorry

end NUMINAMATH_GPT_martha_saving_l498_49811


namespace NUMINAMATH_GPT_measure_of_angle_E_l498_49885

variable (D E F : ℝ)
variable (h1 : E = F)
variable (h2 : F = 3 * D)
variable (h3 : D + E + F = 180)

theorem measure_of_angle_E : E = 540 / 7 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_measure_of_angle_E_l498_49885


namespace NUMINAMATH_GPT_number_one_fourth_less_than_25_percent_more_l498_49856

theorem number_one_fourth_less_than_25_percent_more (x : ℝ) :
  (3 / 4) * x = 1.25 * 80 → x = 133.33 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_number_one_fourth_less_than_25_percent_more_l498_49856


namespace NUMINAMATH_GPT_most_persuasive_method_l498_49884

-- Survey data and conditions
def male_citizens : ℕ := 4258
def male_believe_doping : ℕ := 2360
def female_citizens : ℕ := 3890
def female_believe_framed : ℕ := 2386

def random_division_by_gender : Prop := true -- Represents the random division into male and female groups

-- Proposition to prove
theorem most_persuasive_method : 
  random_division_by_gender → 
  ∃ method : String, method = "Independence Test" := by
  sorry

end NUMINAMATH_GPT_most_persuasive_method_l498_49884


namespace NUMINAMATH_GPT_min_value_g_l498_49853

noncomputable def g (x : ℝ) : ℝ := (6 * x^2 + 11 * x + 17) / (7 * (2 + x))

theorem min_value_g : ∃ x, x ≥ 0 ∧ g x = 127 / 24 :=
by
  sorry

end NUMINAMATH_GPT_min_value_g_l498_49853


namespace NUMINAMATH_GPT_average_transformation_l498_49829

theorem average_transformation (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h_avg : (a_1 + a_2 + a_3 + a_4 + a_5) / 5 = 8) : 
  ((a_1 + 10) + (a_2 - 10) + (a_3 + 10) + (a_4 - 10) + (a_5 + 10)) / 5 = 10 := 
by
  sorry

end NUMINAMATH_GPT_average_transformation_l498_49829


namespace NUMINAMATH_GPT_no_rational_solution_l498_49859

theorem no_rational_solution :
  ¬ ∃ (x y z : ℚ), 
  x + y + z = 0 ∧ x^2 + y^2 + z^2 = 100 := sorry

end NUMINAMATH_GPT_no_rational_solution_l498_49859


namespace NUMINAMATH_GPT_bowls_remaining_l498_49836

-- Definitions based on conditions.
def initial_collection : ℕ := 70
def reward_per_10_bowls : ℕ := 2
def total_customers : ℕ := 20
def customers_bought_20 : ℕ := total_customers / 2
def bowls_bought_per_customer : ℕ := 20
def total_bowls_bought : ℕ := customers_bought_20 * bowls_bought_per_customer
def reward_sets : ℕ := total_bowls_bought / 10
def total_reward_given : ℕ := reward_sets * reward_per_10_bowls

-- Theorem statement to be proved.
theorem bowls_remaining : initial_collection - total_reward_given = 30 :=
by
  sorry

end NUMINAMATH_GPT_bowls_remaining_l498_49836


namespace NUMINAMATH_GPT_toms_nickels_l498_49879

variables (q n : ℕ)

theorem toms_nickels (h1 : q + n = 12) (h2 : 25 * q + 5 * n = 220) : n = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_toms_nickels_l498_49879


namespace NUMINAMATH_GPT_pounds_over_minimum_l498_49872

def cost_per_pound : ℕ := 3
def minimum_purchase : ℕ := 15
def total_spent : ℕ := 105

theorem pounds_over_minimum : 
  (total_spent / cost_per_pound) - minimum_purchase = 20 :=
by
  sorry

end NUMINAMATH_GPT_pounds_over_minimum_l498_49872


namespace NUMINAMATH_GPT_price_on_hot_day_l498_49822

noncomputable def regular_price_P (P : ℝ) : Prop :=
  7 * 32 * (P - 0.75) + 3 * 32 * (1.25 * P - 0.75) = 450

theorem price_on_hot_day (P : ℝ) (h : regular_price_P P) : 1.25 * P = 2.50 :=
by sorry

end NUMINAMATH_GPT_price_on_hot_day_l498_49822


namespace NUMINAMATH_GPT_solution_set_of_abs_x_gt_1_l498_49830

theorem solution_set_of_abs_x_gt_1 (x : ℝ) : |x| > 1 ↔ x > 1 ∨ x < -1 := 
sorry

end NUMINAMATH_GPT_solution_set_of_abs_x_gt_1_l498_49830


namespace NUMINAMATH_GPT_binomial_coeff_sum_abs_l498_49838

theorem binomial_coeff_sum_abs (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ)
  (h : (2 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0):
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coeff_sum_abs_l498_49838


namespace NUMINAMATH_GPT_domino_perfect_play_winner_l498_49847

theorem domino_perfect_play_winner :
  ∀ {PlayerI PlayerII : Type} 
    (legal_move : PlayerI → PlayerII → Prop)
    (initial_move : PlayerI → Prop)
    (next_moves : PlayerII → PlayerI → PlayerII → Prop),
    (∀ pI pII, legal_move pI pII) → 
    (∃ m, initial_move m) → 
    (∀ mI mII, next_moves mII mI mII) → 
    ∃ winner, winner = PlayerI :=
by
  sorry

end NUMINAMATH_GPT_domino_perfect_play_winner_l498_49847


namespace NUMINAMATH_GPT_factor_of_quadratic_l498_49819

theorem factor_of_quadratic (m : ℝ) : (∀ x, (x + 6) * (x + a) = x ^ 2 - mx - 42) → m = 1 :=
by sorry

end NUMINAMATH_GPT_factor_of_quadratic_l498_49819


namespace NUMINAMATH_GPT_find_radii_of_circles_l498_49893

theorem find_radii_of_circles (d : ℝ) (ext_tangent : ℝ) (int_tangent : ℝ)
  (hd : d = 65) (hext : ext_tangent = 63) (hint : int_tangent = 25) :
  ∃ (R r : ℝ), R = 38 ∧ r = 22 :=
by 
  sorry

end NUMINAMATH_GPT_find_radii_of_circles_l498_49893


namespace NUMINAMATH_GPT_recyclable_cans_and_bottles_collected_l498_49881

-- Define the conditions in Lean
def people_at_picnic : ℕ := 90
def soda_cans : ℕ := 50
def plastic_bottles_sparkling_water : ℕ := 50
def glass_bottles_juice : ℕ := 50
def guests_drank_soda : ℕ := people_at_picnic / 2
def guests_drank_sparkling_water : ℕ := people_at_picnic / 3
def juice_consumed : ℕ := (glass_bottles_juice * 4) / 5

-- The theorem statement
theorem recyclable_cans_and_bottles_collected :
  (soda_cans + guests_drank_sparkling_water + juice_consumed) = 120 :=
by
  sorry

end NUMINAMATH_GPT_recyclable_cans_and_bottles_collected_l498_49881


namespace NUMINAMATH_GPT_intersection_points_l498_49808

-- Definition of curve C by the polar equation
def curve_C (ρ : ℝ) (θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Definition of line l by the polar equation
def line_l (ρ : ℝ) (θ : ℝ) (m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 6) = m

-- Proof statement that line l intersects curve C exactly once for specific values of m
theorem intersection_points (m : ℝ) : 
  (∀ ρ θ, curve_C ρ θ → line_l ρ θ m → ρ = 0 ∧ θ = 0) ↔ (m = -1/2 ∨ m = 3/2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l498_49808


namespace NUMINAMATH_GPT_final_price_correct_l498_49840

def original_cost : ℝ := 2.00
def discount : ℝ := 0.57
def final_price : ℝ := 1.43

theorem final_price_correct :
  original_cost - discount = final_price :=
by
  sorry

end NUMINAMATH_GPT_final_price_correct_l498_49840


namespace NUMINAMATH_GPT_pencil_price_units_l498_49860

noncomputable def price_pencil (base_price: ℕ) (extra_cost: ℕ): ℝ :=
  (base_price + extra_cost) / 10000.0

theorem pencil_price_units (base_price: ℕ) (extra_cost: ℕ) (h_base: base_price = 5000) (h_extra: extra_cost = 20) : 
  price_pencil base_price extra_cost = 0.5 := by
  sorry

end NUMINAMATH_GPT_pencil_price_units_l498_49860


namespace NUMINAMATH_GPT_finite_ring_identity_l498_49888

variable {A : Type} [Ring A] [Fintype A]
variables (a b : A)

theorem finite_ring_identity (h : (ab - 1) * b = 0) : b * (ab - 1) = 0 :=
sorry

end NUMINAMATH_GPT_finite_ring_identity_l498_49888


namespace NUMINAMATH_GPT_problem1_problem2_l498_49876

noncomputable def f (x : ℝ) : ℝ :=
let m := (2 * Real.cos x, 1)
let n := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
m.1 * n.1 + m.2 * n.2

theorem problem1 :
  ( ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi ) ∧
  ∀ k : ℤ, ∀ x ∈ Set.Icc ((1 : ℝ) * Real.pi / 6 + k * Real.pi) ((2 : ℝ) * Real.pi / 3 + k * Real.pi),
  f x < f (x + (Real.pi / 3)) :=
sorry

theorem problem2 (A : ℝ) (a b c : ℝ) :
  a ≠ 0 ∧ b = 1 ∧ f A = 2 ∧
  0 < A ∧ A < Real.pi ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2  →
  a = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l498_49876


namespace NUMINAMATH_GPT_no_integer_solution_exists_l498_49831

theorem no_integer_solution_exists : ¬ ∃ (x y z t : ℤ), x^2 + y^2 + z^2 = 8 * t - 1 := 
by sorry

end NUMINAMATH_GPT_no_integer_solution_exists_l498_49831


namespace NUMINAMATH_GPT_smallest_gcd_12a_20b_l498_49863

theorem smallest_gcd_12a_20b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 := sorry

end NUMINAMATH_GPT_smallest_gcd_12a_20b_l498_49863


namespace NUMINAMATH_GPT_fanfan_home_distance_l498_49849

theorem fanfan_home_distance (x y z : ℝ) 
  (h1 : x / 3 = 10) 
  (h2 : x / 3 + y / 2 = 25) 
  (h3 : x / 3 + y / 2 + z = 85) :
  x + y + z = 120 :=
sorry

end NUMINAMATH_GPT_fanfan_home_distance_l498_49849


namespace NUMINAMATH_GPT_shorter_piece_length_l498_49839

theorem shorter_piece_length (x : ℝ) :
  (120 - (2 * x + 15) = x) → x = 35 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l498_49839


namespace NUMINAMATH_GPT_problem_f_symmetry_problem_f_definition_problem_correct_answer_l498_49837

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then Real.log x else Real.log (2 - x)

theorem problem_f_symmetry (x : ℝ) : f (2 - x) = f x := 
sorry

theorem problem_f_definition (x : ℝ) (hx : x ≥ 1) : f x = Real.log x :=
sorry

theorem problem_correct_answer: 
  f (1 / 2) < f 2 ∧ f 2 < f (1 / 3) :=
sorry

end NUMINAMATH_GPT_problem_f_symmetry_problem_f_definition_problem_correct_answer_l498_49837


namespace NUMINAMATH_GPT_total_calories_consumed_l498_49870

-- Definitions for conditions
def calories_per_chip : ℕ := 60 / 10
def extra_calories_per_cheezit := calories_per_chip / 3
def calories_per_cheezit: ℕ := calories_per_chip + extra_calories_per_cheezit
def total_calories_chips : ℕ := 60
def total_calories_cheezits : ℕ := 6 * calories_per_cheezit

-- Main statement to be proved
theorem total_calories_consumed : total_calories_chips + total_calories_cheezits = 108 := by 
  sorry

end NUMINAMATH_GPT_total_calories_consumed_l498_49870


namespace NUMINAMATH_GPT_R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l498_49845

theorem R_H_nonneg_def (H : ℝ) (s t : ℝ) (hH : 0 < H ∧ H ≤ 1) :
  (1 / 2) * (|t| ^ (2 * H) + |s| ^ (2 * H) - |t - s| ^ (2 * H)) ≥ 0 := sorry

theorem R_K_nonneg_def (K : ℝ) (s t : ℝ) (hK : 0 < K ∧ K ≤ 2) :
  (1 / 2 ^ K) * (|t + s| ^ K - |t - s| ^ K) ≥ 0 := sorry

theorem R_HK_nonneg_def (H K : ℝ) (s t : ℝ) (hHK : 0 < H ∧ H ≤ 1 ∧ 0 < K ∧ K ≤ 1) :
  (1 / 2 ^ K) * ( (|t| ^ (2 * H) + |s| ^ (2 * H)) ^ K - |t - s| ^ (2 * H * K) ) ≥ 0 := sorry

end NUMINAMATH_GPT_R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l498_49845


namespace NUMINAMATH_GPT_probability_of_drawing_red_ball_l498_49855

def totalBalls : Nat := 3 + 5 + 2
def redBalls : Nat := 3
def probabilityOfRedBall : ℚ := redBalls / totalBalls

theorem probability_of_drawing_red_ball :
  probabilityOfRedBall = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_red_ball_l498_49855


namespace NUMINAMATH_GPT_nonneg_sol_eq_l498_49809

theorem nonneg_sol_eq {a b c : ℝ} (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c) 
  (h1 : a * (a + b) = b * (b + c)) (h2 : b * (b + c) = c * (c + a)) : 
  a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_nonneg_sol_eq_l498_49809


namespace NUMINAMATH_GPT_find_b_perpendicular_l498_49816

theorem find_b_perpendicular (b : ℝ) : (∀ x y : ℝ, 4 * y - 2 * x = 6 → 5 * y + b * x - 2 = 0 → (1 / 2 : ℝ) * (-(b / 5) : ℝ) = -1) → b = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_b_perpendicular_l498_49816


namespace NUMINAMATH_GPT_no_x_satisfies_arithmetic_mean_l498_49873

theorem no_x_satisfies_arithmetic_mean :
  ¬ ∃ x : ℝ, (3 + 117 + 915 + 138 + 2114 + x) / 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_no_x_satisfies_arithmetic_mean_l498_49873


namespace NUMINAMATH_GPT_pyramid_base_edge_length_l498_49835

-- Prove that the edge-length of the base of the pyramid is as specified
theorem pyramid_base_edge_length
  (r h : ℝ)
  (hemisphere_radius : r = 3)
  (pyramid_height : h = 8)
  (tangency_condition : true) : true :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_edge_length_l498_49835


namespace NUMINAMATH_GPT_screen_width_l498_49887

theorem screen_width
  (A : ℝ) -- Area of the screen
  (h : ℝ) -- Height of the screen
  (w : ℝ) -- Width of the screen
  (area_eq : A = 21) -- Condition 1: Area is 21 sq ft
  (height_eq : h = 7) -- Condition 2: Height is 7 ft
  (area_formula : A = w * h) -- Condition 3: Area formula
  : w = 3 := -- Conclusion: Width is 3 ft
sorry

end NUMINAMATH_GPT_screen_width_l498_49887


namespace NUMINAMATH_GPT_least_value_l498_49804

theorem least_value : ∀ x y : ℝ, (xy + 1)^2 + (x - y)^2 ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_least_value_l498_49804


namespace NUMINAMATH_GPT_oscar_cookie_baking_time_l498_49846

theorem oscar_cookie_baking_time : 
  (1 / 5) + (1 / 6) + (1 / o) - (1 / 4) = (1 / 8) → o = 120 := by
  sorry

end NUMINAMATH_GPT_oscar_cookie_baking_time_l498_49846


namespace NUMINAMATH_GPT_sum_of_sequence_correct_l498_49841

def calculateSumOfSequence : ℚ :=
  (4 / 3) + (7 / 5) + (11 / 8) + (19 / 15) + (35 / 27) + (67 / 52) - 9

theorem sum_of_sequence_correct :
  calculateSumOfSequence = (-17312.5 / 7020) := by
  sorry

end NUMINAMATH_GPT_sum_of_sequence_correct_l498_49841


namespace NUMINAMATH_GPT_number_of_distinct_b_values_l498_49878

theorem number_of_distinct_b_values : 
  ∃ (b : ℝ) (p q : ℤ), (∀ (x : ℝ), x*x + b*x + 12*b = 0) ∧ 
                        p + q = -b ∧ 
                        p * q = 12 * b ∧ 
                        ∃ n : ℤ, 1 ≤ n ∧ n ≤ 15 :=
sorry

end NUMINAMATH_GPT_number_of_distinct_b_values_l498_49878


namespace NUMINAMATH_GPT_middle_guards_hours_l498_49850

def total_hours := 9
def hours_first_guard := 3
def hours_last_guard := 2
def remaining_hours := total_hours - hours_first_guard - hours_last_guard
def num_middle_guards := 2

theorem middle_guards_hours : remaining_hours / num_middle_guards = 2 := by
  sorry

end NUMINAMATH_GPT_middle_guards_hours_l498_49850


namespace NUMINAMATH_GPT_equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l498_49874
open BigOperators

-- First, we define the three equations and their constraints
def equation1_solution (k : ℤ) : ℤ × ℤ := (2 - 5 * k, -1 + 3 * k)
def equation2_solution (k : ℤ) : ℤ × ℤ := (8 - 5 * k, -4 + 3 * k)
def equation3_solution (k : ℤ) : ℤ × ℤ := (16 - 39 * k, -25 + 61 * k)

-- Define the proof that the supposed solutions hold for each equation
theorem equation1_solution_valid (k : ℤ) : 3 * (equation1_solution k).1 + 5 * (equation1_solution k).2 = 1 :=
by
  -- Proof steps would go here
  sorry

theorem equation2_solution_valid (k : ℤ) : 3 * (equation2_solution k).1 + 5 * (equation2_solution k).2 = 4 :=
by
  -- Proof steps would go here
  sorry

theorem equation3_solution_valid (k : ℤ) : 183 * (equation3_solution k).1 + 117 * (equation3_solution k).2 = 3 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l498_49874


namespace NUMINAMATH_GPT_max_value_b_exists_l498_49854

theorem max_value_b_exists :
  ∃ a c : ℝ, ∃ b : ℝ, 
  (∀ x : ℤ, 
  ((x^4 - a * x^3 - b * x^2 - c * x - 2007) = 0) → 
  ∃ r s t : ℤ, r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
  ((x = r) ∨ (x = s) ∨ (x = t))) ∧ 
  (∀ b' : ℝ, b' < b → 
  ¬ ( ∃ a' c' : ℝ, ( ∀ x : ℤ, 
  ((x^4 - a' * x^3 - b' * x^2 - c' * x - 2007) = 0) → 
  ∃ r' s' t' : ℤ, r' ≠ s' ∧ s' ≠ t' ∧ r' ≠ t' ∧ 
  ((x = r') ∨ (x = s') ∨ (x = t') )))) ∧ b = 3343 :=
sorry

end NUMINAMATH_GPT_max_value_b_exists_l498_49854


namespace NUMINAMATH_GPT_bug_total_distance_l498_49883

theorem bug_total_distance :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let final_pos := 0
  let distance1 := |pos1 - pos2|
  let distance2 := |pos2 - pos3|
  let distance3 := |pos3 - final_pos|
  let total_distance := distance1 + distance2 + distance3
  total_distance = 29 := by
    sorry

end NUMINAMATH_GPT_bug_total_distance_l498_49883


namespace NUMINAMATH_GPT_sandra_beignets_16_weeks_l498_49828

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_sandra_beignets_16_weeks_l498_49828


namespace NUMINAMATH_GPT_john_candies_correct_l498_49858

variable (Bob_candies : ℕ) (Mary_candies : ℕ)
          (Sue_candies : ℕ) (Sam_candies : ℕ)
          (Total_candies : ℕ) (John_candies : ℕ)

axiom bob_has : Bob_candies = 10
axiom mary_has : Mary_candies = 5
axiom sue_has : Sue_candies = 20
axiom sam_has : Sam_candies = 10
axiom total_has : Total_candies = 50

theorem john_candies_correct : 
  Bob_candies + Mary_candies + Sue_candies + Sam_candies + John_candies = Total_candies → John_candies = 5 := by
sorry

end NUMINAMATH_GPT_john_candies_correct_l498_49858


namespace NUMINAMATH_GPT_josh_paid_6_dollars_l498_49824

def packs : ℕ := 3
def cheesePerPack : ℕ := 20
def costPerCheese : ℕ := 10 -- cost in cents

theorem josh_paid_6_dollars :
  (packs * cheesePerPack * costPerCheese) / 100 = 6 :=
by
  sorry

end NUMINAMATH_GPT_josh_paid_6_dollars_l498_49824


namespace NUMINAMATH_GPT_min_value_of_frac_l498_49825

open Real

theorem min_value_of_frac (x : ℝ) (hx : x > 0) : 
  ∃ (t : ℝ), t = 2 * sqrt 5 + 2 ∧ (∀ y, y > 0 → (x^2 + 2 * x + 5) / x ≥ t) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_frac_l498_49825


namespace NUMINAMATH_GPT_possible_divisor_of_p_l498_49892

theorem possible_divisor_of_p (p q r s : ℕ)
  (hpq : ∃ x y, p = 40 * x ∧ q = 40 * y ∧ Nat.gcd p q = 40)
  (hqr : ∃ u v, q = 45 * u ∧ r = 45 * v ∧ Nat.gcd q r = 45)
  (hrs : ∃ w z, r = 60 * w ∧ s = 60 * z ∧ Nat.gcd r s = 60)
  (hsp : ∃ t, Nat.gcd s p = 100 * t ∧ 100 ≤ Nat.gcd s p ∧ Nat.gcd s p < 1000) :
  7 ∣ p :=
sorry

end NUMINAMATH_GPT_possible_divisor_of_p_l498_49892


namespace NUMINAMATH_GPT_average_temp_addington_l498_49889

def temperatures : List ℚ := [60, 59, 56, 53, 49, 48, 46]

def average_temp (temps : List ℚ) : ℚ := (temps.sum) / temps.length

theorem average_temp_addington :
  average_temp temperatures = 53 := by
  sorry

end NUMINAMATH_GPT_average_temp_addington_l498_49889


namespace NUMINAMATH_GPT_local_minimum_at_one_l498_49875

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + a^2 * x

theorem local_minimum_at_one (a : ℝ) (hfmin : ∀ x : ℝ, deriv (f a) x = 3 * a * x^2 - 4 * x + a^2) (h1 : f a 1 = f a 1) : a = 1 :=
sorry

end NUMINAMATH_GPT_local_minimum_at_one_l498_49875


namespace NUMINAMATH_GPT_larger_number_of_hcf_and_lcm_factors_l498_49806

theorem larger_number_of_hcf_and_lcm_factors :
  ∃ (a b : ℕ), (∀ d, d ∣ a ∧ d ∣ b → d ≤ 20) ∧ (∃ x y, x * y * 20 = a * b ∧ x * 20 = a ∧ y * 20 = b ∧ x > y ∧ x = 15 ∧ y = 11) → max a b = 300 :=
by sorry

end NUMINAMATH_GPT_larger_number_of_hcf_and_lcm_factors_l498_49806


namespace NUMINAMATH_GPT_from20To25_l498_49818

def canObtain25 (start : ℕ) : Prop :=
  ∃ (steps : ℕ → ℕ), steps 0 = start ∧ (∃ n, steps n = 25) ∧ 
  (∀ i, steps (i+1) = (steps i * 2) ∨ (steps (i+1) = steps i / 10))

theorem from20To25 : canObtain25 20 :=
sorry

end NUMINAMATH_GPT_from20To25_l498_49818
