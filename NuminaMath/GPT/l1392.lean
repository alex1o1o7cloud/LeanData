import Mathlib

namespace locus_of_orthocenter_l1392_139275

theorem locus_of_orthocenter (A_x A_y : ℝ) (h_A : A_x = 0 ∧ A_y = 2)
    (c_r : ℝ) (h_c : c_r = 2) 
    (M_x M_y Q_x Q_y : ℝ)
    (h_circle : Q_x^2 + Q_y^2 = c_r^2)
    (h_tangent : M_x ≠ 0 ∧ (M_y - 2) / M_x = -Q_x / Q_y)
    (h_M_on_tangent : M_x^2 + (M_y - 2)^2 = 4 ∧ M_x ≠ 0)
    (H_x H_y : ℝ)
    (h_orthocenter : (H_x - A_x)^2 + (H_y - A_y + 2)^2 = 4) :
    (H_x^2 + (H_y - 2)^2 = 4) ∧ (H_x ≠ 0) := 
sorry

end locus_of_orthocenter_l1392_139275


namespace N_is_composite_l1392_139247

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Prime N :=
by {
  sorry
}

end N_is_composite_l1392_139247


namespace odd_function_product_nonpositive_l1392_139283

noncomputable def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_product_nonpositive (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) : 
  ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by 
  sorry

end odd_function_product_nonpositive_l1392_139283


namespace farmer_land_l1392_139214

noncomputable def farmer_land_example (A : ℝ) : Prop :=
  let cleared_land := 0.90 * A
  let barley_land := 0.70 * cleared_land
  let potatoes_land := 0.10 * cleared_land
  let corn_land := 0.10 * cleared_land
  let tomatoes_bell_peppers_land := 0.10 * cleared_land
  tomatoes_bell_peppers_land = 90 → A = 1000

theorem farmer_land (A : ℝ) (h_cleared_land : 0.90 * A = cleared_land)
  (h_barley_land : 0.70 * cleared_land = barley_land)
  (h_potatoes_land : 0.10 * cleared_land = potatoes_land)
  (h_corn_land : 0.10 * cleared_land = corn_land)
  (h_tomatoes_bell_peppers_land : 0.10 * cleared_land = 90) :
  A = 1000 :=
by
  sorry

end farmer_land_l1392_139214


namespace smallest_prime_factor_of_2939_l1392_139276

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → p ≤ q

theorem smallest_prime_factor_of_2939 : smallest_prime_factor 2939 13 :=
by
  sorry

end smallest_prime_factor_of_2939_l1392_139276


namespace suji_present_age_l1392_139252

/-- Present ages of Abi and Suji are in the ratio of 5:4. --/
def abi_suji_ratio (abi_age suji_age : ℕ) : Prop := abi_age = 5 * (suji_age / 4)

/-- 3 years hence, the ratio of their ages will be 11:9. --/
def abi_suji_ratio_future (abi_age suji_age : ℕ) : Prop :=
  ((abi_age + 3).toFloat / (suji_age + 3).toFloat) = 11 / 9

theorem suji_present_age (suji_age : ℕ) (abi_age : ℕ) (x : ℕ) 
  (h1 : abi_age = 5 * x) (h2 : suji_age = 4 * x)
  (h3 : abi_suji_ratio_future abi_age suji_age) :
  suji_age = 24 := 
sorry

end suji_present_age_l1392_139252


namespace ratio_65_13_l1392_139277

theorem ratio_65_13 : 65 / 13 = 5 := 
by
  sorry

end ratio_65_13_l1392_139277


namespace probability_crisp_stops_on_dime_l1392_139291

noncomputable def crisp_stops_on_dime_probability : ℚ :=
  let a := (2/3 : ℚ)
  let b := (1/3 : ℚ)
  let a1 := (15/31 : ℚ)
  let b1 := (30/31 : ℚ)
  (2 / 3) * a1 + (1 / 3) * b1

theorem probability_crisp_stops_on_dime :
  crisp_stops_on_dime_probability = 20 / 31 :=
by
  sorry

end probability_crisp_stops_on_dime_l1392_139291


namespace executive_board_elections_l1392_139273

noncomputable def num_candidates : ℕ := 18
noncomputable def num_positions : ℕ := 6
noncomputable def num_former_board_members : ℕ := 8

noncomputable def total_selections := Nat.choose num_candidates num_positions
noncomputable def no_former_board_members_selections := Nat.choose (num_candidates - num_former_board_members) num_positions

noncomputable def valid_selections := total_selections - no_former_board_members_selections

theorem executive_board_elections : valid_selections = 18354 :=
by sorry

end executive_board_elections_l1392_139273


namespace gcd_max_value_l1392_139286

theorem gcd_max_value (x y : ℤ) (h_posx : x > 0) (h_posy : y > 0) (h_sum : x + y = 780) :
  gcd x y ≤ 390 ∧ ∃ x' y', x' > 0 ∧ y' > 0 ∧ x' + y' = 780 ∧ gcd x' y' = 390 := by
  sorry

end gcd_max_value_l1392_139286


namespace largest_a_pow_b_l1392_139271

theorem largest_a_pow_b (a b : ℕ) (h_pos_a : 1 < a) (h_pos_b : 1 < b) (h_eq : a^b * b^a + a^b + b^a = 5329) : 
  a^b = 64 :=
by
  sorry

end largest_a_pow_b_l1392_139271


namespace susan_remaining_spaces_l1392_139254

def susan_first_turn_spaces : ℕ := 15
def susan_second_turn_spaces : ℕ := 7 - 5
def susan_third_turn_spaces : ℕ := 20
def susan_fourth_turn_spaces : ℕ := 0
def susan_fifth_turn_spaces : ℕ := 10 - 8
def susan_sixth_turn_spaces : ℕ := 0
def susan_seventh_turn_roll : ℕ := 6
def susan_seventh_turn_spaces : ℕ := susan_seventh_turn_roll * 2
def susan_total_moved_spaces : ℕ := susan_first_turn_spaces + susan_second_turn_spaces + susan_third_turn_spaces + susan_fourth_turn_spaces + susan_fifth_turn_spaces + susan_sixth_turn_spaces + susan_seventh_turn_spaces
def game_total_spaces : ℕ := 100

theorem susan_remaining_spaces : susan_total_moved_spaces = 51 ∧ (game_total_spaces - susan_total_moved_spaces) = 49 := by
  sorry

end susan_remaining_spaces_l1392_139254


namespace pyramid_inscribed_sphere_radius_l1392_139243

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := 
a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3))

theorem pyramid_inscribed_sphere_radius (a : ℝ) (h1 : a > 0) : 
  inscribed_sphere_radius a = a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3)) :=
by
  sorry

end pyramid_inscribed_sphere_radius_l1392_139243


namespace greatest_divisor_540_180_under_60_l1392_139274

theorem greatest_divisor_540_180_under_60 : ∃ d, d ∣ 540 ∧ d ∣ 180 ∧ d < 60 ∧ ∀ k, k ∣ 540 → k ∣ 180 → k < 60 → k ≤ d :=
by
  sorry

end greatest_divisor_540_180_under_60_l1392_139274


namespace puppies_per_dog_l1392_139298

/--
Chuck breeds dogs. He has 3 pregnant dogs.
They each give birth to some puppies. Each puppy needs 2 shots and each shot costs $5.
The total cost of the shots is $120. Prove that each pregnant dog gives birth to 4 puppies.
-/
theorem puppies_per_dog :
  let num_dogs := 3
  let cost_per_shot := 5
  let shots_per_puppy := 2
  let total_cost := 120
  let cost_per_puppy := shots_per_puppy * cost_per_shot
  let total_puppies := total_cost / cost_per_puppy
  (total_puppies / num_dogs) = 4 := by
  sorry

end puppies_per_dog_l1392_139298


namespace probability_at_least_6_heads_in_8_flips_l1392_139256

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l1392_139256


namespace abs_gt_1_not_sufficient_nor_necessary_l1392_139208

theorem abs_gt_1_not_sufficient_nor_necessary (a : ℝ) :
  ¬((|a| > 1) → (a > 0)) ∧ ¬((a > 0) → (|a| > 1)) :=
by
  sorry

end abs_gt_1_not_sufficient_nor_necessary_l1392_139208


namespace power_of_two_minus_one_divisible_by_seven_power_of_two_plus_one_not_divisible_by_seven_l1392_139269

theorem power_of_two_minus_one_divisible_by_seven (n : ℕ) (hn : 0 < n) : 
  (∃ k : ℕ, 0 < k ∧ n = k * 3) ↔ (7 ∣ 2^n - 1) :=
by sorry

theorem power_of_two_plus_one_not_divisible_by_seven (n : ℕ) (hn : 0 < n) :
  ¬(7 ∣ 2^n + 1) :=
by sorry

end power_of_two_minus_one_divisible_by_seven_power_of_two_plus_one_not_divisible_by_seven_l1392_139269


namespace johnny_tables_l1392_139287

theorem johnny_tables :
  ∀ (T : ℕ),
  (∀ (T : ℕ), 4 * T + 5 * T = 45) →
  T = 5 :=
  sorry

end johnny_tables_l1392_139287


namespace expression_simplified_l1392_139257

noncomputable def expression : ℚ := 1 + 3 / (4 + 5 / 6)

theorem expression_simplified : expression = 47 / 29 :=
by
  sorry

end expression_simplified_l1392_139257


namespace satisfy_equation_l1392_139293

theorem satisfy_equation (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end satisfy_equation_l1392_139293


namespace how_many_kids_joined_l1392_139266

theorem how_many_kids_joined (original_kids : ℕ) (new_kids : ℕ) (h : original_kids = 14) (h1 : new_kids = 36) :
  new_kids - original_kids = 22 :=
by
  sorry

end how_many_kids_joined_l1392_139266


namespace smallest_value_of_x_l1392_139253

theorem smallest_value_of_x (x : ℝ) (h : 6 * x ^ 2 - 37 * x + 48 = 0) : x = 13 / 6 :=
sorry

end smallest_value_of_x_l1392_139253


namespace charlotte_flour_cost_l1392_139217

noncomputable def flour_cost 
  (flour_sugar_eggs_butter_cost blueberry_cost cherry_cost total_cost : ℝ)
  (blueberry_weight oz_per_lb blueberry_cost_per_container cherry_weight cherry_cost_per_bag : ℝ)
  (additional_cost : ℝ) : ℝ :=
  total_cost - (blueberry_cost + additional_cost)

theorem charlotte_flour_cost :
  flour_cost 2.5 13.5 14 18 3 16 2.25 4 14 2.5 = 2 :=
by
  unfold flour_cost
  sorry

end charlotte_flour_cost_l1392_139217


namespace sin_double_angle_identity_l1392_139235

noncomputable def given_tan_alpha (α : ℝ) : Prop := 
  Real.tan α = 1/2

theorem sin_double_angle_identity (α : ℝ) (h : given_tan_alpha α) : 
  Real.sin (2 * α) = 4 / 5 := 
sorry

end sin_double_angle_identity_l1392_139235


namespace integer_b_if_integer_a_l1392_139284

theorem integer_b_if_integer_a (a b : ℤ) (h : 2 * a + a^2 = 2 * b + b^2) : (∃ a' : ℤ, a = a') → ∃ b' : ℤ, b = b' :=
by
-- proof will be filled in here
sorry

end integer_b_if_integer_a_l1392_139284


namespace F_double_reflection_l1392_139231

structure Point where
  x : ℝ
  y : ℝ

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := -1, y := -1 }

theorem F_double_reflection :
  reflect_x (reflect_y F) = { x := 1, y := 1 } :=
  sorry

end F_double_reflection_l1392_139231


namespace vasya_max_earning_l1392_139244

theorem vasya_max_earning (k : ℕ) (h₀: k ≤ 2013) (h₁: 2013 - 2*k % 11 = 0) : k % 11 = 0 → (k ≤ 5) := 
by
  sorry

end vasya_max_earning_l1392_139244


namespace expand_product_l1392_139223

noncomputable def expand_poly (x : ℝ) : ℝ := (x + 3) * (x^2 + 2 * x + 4)

theorem expand_product (x : ℝ) : expand_poly x = x^3 + 5 * x^2 + 10 * x + 12 := 
by 
  -- This will be filled with the proof steps, but for now we use sorry.
  sorry

end expand_product_l1392_139223


namespace aras_current_height_l1392_139296

-- Define the variables and conditions
variables (x : ℝ) (sheas_original_height : ℝ := x) (ars_original_height : ℝ := x)
variables (sheas_growth_factor : ℝ := 0.30) (sheas_current_height : ℝ := 65)
variables (sheas_growth : ℝ := sheas_current_height - sheas_original_height)
variables (aras_growth : ℝ := sheas_growth / 3)

-- Define a theorem for Ara's current height
theorem aras_current_height (h1 : sheas_current_height = (1 + sheas_growth_factor) * sheas_original_height)
                           (h2 : sheas_original_height = ars_original_height) :
                           aras_growth + ars_original_height = 55 :=
by
  sorry

end aras_current_height_l1392_139296


namespace g_10_plus_g_neg10_eq_6_l1392_139297

variable (a b c : ℝ)
noncomputable def g : ℝ → ℝ := λ x => a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + 5

theorem g_10_plus_g_neg10_eq_6 (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 :=
by
  -- Proof goes here
  sorry

end g_10_plus_g_neg10_eq_6_l1392_139297


namespace find_a6_of_arithmetic_seq_l1392_139263

noncomputable def arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_a6_of_arithmetic_seq 
  (a1 d : ℝ) 
  (S3 : ℝ) 
  (h_a1 : a1 = 2) 
  (h_S3 : S3 = 12) 
  (h_sum : S3 = sum_of_arithmetic_sequence 3 a1 d) :
  arithmetic_sequence 6 a1 d = 12 := 
sorry

end find_a6_of_arithmetic_seq_l1392_139263


namespace uncounted_angle_measure_l1392_139227

-- Define the given miscalculated sum
def miscalculated_sum : ℝ := 2240

-- Define the correct sum expression for an n-sided convex polygon
def correct_sum (n : ℕ) : ℝ := (n - 2) * 180

-- State the theorem: 
theorem uncounted_angle_measure (n : ℕ) (h1 : correct_sum n = 2340) (h2 : 2240 < correct_sum n) :
  correct_sum n - miscalculated_sum = 100 := 
by sorry

end uncounted_angle_measure_l1392_139227


namespace positive_integer_is_48_l1392_139241

theorem positive_integer_is_48 (n p : ℕ) (h_prime : Prime p) (h_eq : n = 24 * p) (h_min : n ≥ 48) : n = 48 :=
by
  sorry

end positive_integer_is_48_l1392_139241


namespace maximize_triangle_area_l1392_139299

theorem maximize_triangle_area (m : ℝ) (l : ∀ x y, x + y + m = 0) (C : ∀ x y, x^2 + y^2 + 4 * y = 0) :
  m = 0 ∨ m = 4 :=
sorry

end maximize_triangle_area_l1392_139299


namespace arithmetic_sequence_S2008_l1392_139228

theorem arithmetic_sequence_S2008 (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (h1 : a1 = -2008)
  (h2 : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d)
  (h3 : (S 12 / 12) - (S 10 / 10) = 2) :
  S 2008 = -2008 := 
sorry

end arithmetic_sequence_S2008_l1392_139228


namespace segment_parallel_to_x_axis_l1392_139222

theorem segment_parallel_to_x_axis 
  (f : ℤ → ℤ) 
  (hf : ∀ n, ∃ m, f n = m) 
  (a b : ℤ) 
  (h_dist : ∃ d : ℤ, d * d = (b - a) * (b - a) + (f b - f a) * (f b - f a)) : 
  f a = f b :=
sorry

end segment_parallel_to_x_axis_l1392_139222


namespace ratio_of_P_Q_l1392_139264

theorem ratio_of_P_Q (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    P / (x + 5) + Q / (x^2 - 4 * x) = (x^2 + x + 15) / (x^3 + x^2 - 20 * x)) :
  Q / P = -45 / 2 :=
by
  sorry

end ratio_of_P_Q_l1392_139264


namespace range_of_a_l1392_139251

-- Definitions
def domain_f : Set ℝ := {x : ℝ | x ≤ -4 ∨ x ≥ 4}
def range_g (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ y = x^2 - 2*x + a}

-- Theorem to prove the range of values for a
theorem range_of_a :
  (∀ x : ℝ, x ∈ domain_f ∨ (∃ y : ℝ, ∃ a : ℝ, y ∈ range_g a ∧ x = y)) ↔ (-4 ≤ a ∧ a ≤ -3) :=
sorry

end range_of_a_l1392_139251


namespace two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l1392_139234

theorem two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one
  (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n) 
  (h : 2 ^ p + 3 ^ p = a ^ n) : n = 1 :=
sorry

end two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l1392_139234


namespace minimum_value_expression_l1392_139242

theorem minimum_value_expression :
  ∃ x y : ℝ, (∀ a b : ℝ, (a^2 + 4*a*b + 5*b^2 - 8*a - 6*b) ≥ -41) ∧ (x^2 + 4*x*y + 5*y^2 - 8*x - 6*y) = -41 := 
sorry

end minimum_value_expression_l1392_139242


namespace tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l1392_139233

variable (α β : ℝ)

theorem tan_sub_eq_one_eight (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
sorry

theorem tan_add_eq_neg_four_seven (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -4 / 7 := 
sorry

end tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l1392_139233


namespace determine_sixth_face_l1392_139246

-- Define a cube configuration and corresponding functions
inductive Color
| black
| white

structure Cube where
  faces : Fin 6 → Fin 9 → Color

noncomputable def sixth_face_color (cube : Cube) : Fin 9 → Color := sorry

-- The statement of the theorem proving the coloring of the sixth face
theorem determine_sixth_face (cube : Cube) : 
  (exists f : (Fin 9 → Color), f = sixth_face_color cube) := 
sorry

end determine_sixth_face_l1392_139246


namespace wedding_reception_friends_l1392_139200

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

end wedding_reception_friends_l1392_139200


namespace integral_value_l1392_139209

theorem integral_value (a : ℝ) (h : a = 2) : ∫ x in a..2*Real.exp 1, 1/x = 1 := by
  sorry

end integral_value_l1392_139209


namespace multiply_469160_999999_l1392_139268

theorem multiply_469160_999999 :
  469160 * 999999 = 469159530840 :=
by
  sorry

end multiply_469160_999999_l1392_139268


namespace complex_fraction_simplification_l1392_139211

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) : (2 : ℂ) / (1 + i)^2 = i :=
by 
-- this will be filled when proving the theorem in Lean
sorry

end complex_fraction_simplification_l1392_139211


namespace mass_of_man_l1392_139281

variable (L : ℝ) (B : ℝ) (h : ℝ) (ρ : ℝ)

-- Given conditions
def boatLength := L = 3
def boatBreadth := B = 2
def sinkingDepth := h = 0.018
def waterDensity := ρ = 1000

-- The mass of the man
theorem mass_of_man (L B h ρ : ℝ) (H1 : boatLength L) (H2 : boatBreadth B) (H3 : sinkingDepth h) (H4 : waterDensity ρ) : 
  ρ * L * B * h = 108 := by
  sorry

end mass_of_man_l1392_139281


namespace arithmetic_progression_infinite_kth_powers_l1392_139288

theorem arithmetic_progression_infinite_kth_powers {a d k : ℕ} (ha : a > 0) (hd : d > 0) (hk : k > 0) :
  (∀ n : ℕ, ¬ ∃ b : ℕ, a + n * d = b ^ k) ∨ (∀ b : ℕ, ∃ n : ℕ, a + n * d = b ^ k) :=
sorry

end arithmetic_progression_infinite_kth_powers_l1392_139288


namespace no_full_conspiracies_in_same_lab_l1392_139203

theorem no_full_conspiracies_in_same_lab
(six_conspiracies : Finset (Finset (Fin 10)))
(h_conspiracies : ∀ c ∈ six_conspiracies, c.card = 3)
(h_total : six_conspiracies.card = 6) :
  ∃ (lab1 lab2 : Finset (Fin 10)), lab1 ∩ lab2 = ∅ ∧ lab1 ∪ lab2 = Finset.univ ∧ ∀ c ∈ six_conspiracies, ¬(c ⊆ lab1 ∨ c ⊆ lab2) :=
by
  sorry

end no_full_conspiracies_in_same_lab_l1392_139203


namespace inscribed_rectangle_area_l1392_139213

variables (b h x : ℝ)
variables (h_isosceles_triangle : b > 0 ∧ h > 0 ∧ x > 0 ∧ x < h)

noncomputable def rectangle_area (b h x : ℝ) : ℝ :=
  (b * x / h) * (h - x)

theorem inscribed_rectangle_area :
  rectangle_area b h x = (b * x / h) * (h - x) :=
by
  unfold rectangle_area
  sorry

end inscribed_rectangle_area_l1392_139213


namespace parallel_vectors_m_eq_neg3_l1392_139229

theorem parallel_vectors_m_eq_neg3 {m : ℝ} :
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  (a.1 * b.2 =  a.2 * b.1) → m = -3 :=
by 
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  intro h
  sorry

end parallel_vectors_m_eq_neg3_l1392_139229


namespace barney_no_clean_towels_days_l1392_139210

theorem barney_no_clean_towels_days
  (wash_cycle_weeks : ℕ := 1)
  (total_towels : ℕ := 18)
  (towels_per_day : ℕ := 2)
  (days_per_week : ℕ := 7)
  (missed_laundry_weeks : ℕ := 1) :
  (days_per_week - (total_towels - (days_per_week * towels_per_day * missed_laundry_weeks)) / towels_per_day) = 5 :=
by
  sorry

end barney_no_clean_towels_days_l1392_139210


namespace find_positive_integer_triples_l1392_139204

-- Define the condition for the integer divisibility problem
def is_integer_division (t a b : ℕ) : Prop :=
  (t ^ (a + b) + 1) % (t ^ a + t ^ b + 1) = 0

-- Statement of the theorem
theorem find_positive_integer_triples :
  ∀ (t a b : ℕ), t > 0 → a > 0 → b > 0 → is_integer_division t a b → (t, a, b) = (2, 1, 1) :=
by
  intros t a b t_pos a_pos b_pos h
  sorry

end find_positive_integer_triples_l1392_139204


namespace cuboid_inequality_l1392_139219

theorem cuboid_inequality 
  (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 = 1) : 
  4*a + 4*b + 4*c + 4*a*b + 4*a*c + 4*b*c + 4*a*b*c < 12 := by
  sorry

end cuboid_inequality_l1392_139219


namespace stratified_sampling_third_grade_l1392_139290

theorem stratified_sampling_third_grade 
  (N : ℕ) (N3 : ℕ) (S : ℕ) (x : ℕ)
  (h1 : N = 1600)
  (h2 : N3 = 400)
  (h3 : S = 80)
  (h4 : N3 / N = x / S) :
  x = 20 := 
by {
  sorry
}

end stratified_sampling_third_grade_l1392_139290


namespace find_d_l1392_139265

-- Definitions based on conditions
def f (x : ℝ) (c : ℝ) := 5 * x + c
def g (x : ℝ) (c : ℝ) := c * x + 3

-- The theorem statement
theorem find_d (c d : ℝ) (h₁ : f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry -- Proof is omitted as per the instructions

end find_d_l1392_139265


namespace least_integer_square_double_l1392_139262

theorem least_integer_square_double (x : ℤ) : x^2 = 2 * x + 50 → x = -5 :=
by
  sorry

end least_integer_square_double_l1392_139262


namespace find_f_prime_at_one_l1392_139245

theorem find_f_prime_at_one (a b : ℝ)
  (h1 : ∀ x, f x = a * Real.exp x + b * x) 
  (h2 : f 0 = 1)
  (h3 : ∀ x, deriv f x = a * Real.exp x + b)
  (h4 : deriv f 0 = 0) :
  deriv f 1 = Real.exp 1 - 1 :=
by {
  sorry
}

end find_f_prime_at_one_l1392_139245


namespace mateen_garden_area_l1392_139201

theorem mateen_garden_area :
  ∃ (L W : ℝ), (20 * L = 1000) ∧ (8 * (2 * L + 2 * W) = 1000) ∧ (L * W = 625) :=
by
  sorry

end mateen_garden_area_l1392_139201


namespace extreme_value_at_x_eq_one_l1392_139218

noncomputable def f (x a b: ℝ) : ℝ := x^3 - a * x^2 + b * x + a^2
noncomputable def f_prime (x a b: ℝ) : ℝ := 3 * x^2 - 2 * a * x + b

theorem extreme_value_at_x_eq_one (a b : ℝ) (h_prime : f_prime 1 a b = 0) (h_value : f 1 a b = 10) : a = -4 :=
by 
  sorry -- proof goes here

end extreme_value_at_x_eq_one_l1392_139218


namespace trainB_reaches_in_3_hours_l1392_139237

variable (trainA_speed trainB_speed : ℕ) (x t : ℝ)

-- Given conditions
axiom h1 : trainA_speed = 70
axiom h2 : trainB_speed = 105
axiom h3 : ∀ x t, 70 * x + 70 * 9 = 105 * x + 105 * t

-- Prove that train B takes 3 hours to reach destination after meeting
theorem trainB_reaches_in_3_hours : t = 3 :=
by
  sorry

end trainB_reaches_in_3_hours_l1392_139237


namespace fraction_irreducible_l1392_139215

theorem fraction_irreducible (n : ℤ) : Nat.gcd (18 * n + 3).natAbs (12 * n + 1).natAbs = 1 := 
sorry

end fraction_irreducible_l1392_139215


namespace complete_the_square_l1392_139280

theorem complete_the_square (x : ℝ) :
  (x^2 + 14*x + 60) = ((x + 7) ^ 2 + 11) :=
by
  sorry

end complete_the_square_l1392_139280


namespace sqrt_prime_irrational_l1392_139240

theorem sqrt_prime_irrational (p : ℕ) (hp : Nat.Prime p) : Irrational (Real.sqrt p) :=
by
  sorry

end sqrt_prime_irrational_l1392_139240


namespace range_f_l1392_139212

noncomputable def g (x : ℝ) : ℝ := 30 + 14 * Real.cos x - 7 * Real.cos (2 * x)

noncomputable def z (t : ℝ) : ℝ := 40.5 - 14 * (t - 0.5) ^ 2

noncomputable def u (z : ℝ) : ℝ := (Real.pi / 54) * z

noncomputable def f (x : ℝ) : ℝ := Real.sin (u (z (Real.cos x)))

theorem range_f : ∀ x : ℝ, 0.5 ≤ f x ∧ f x ≤ 1 :=
by
  intro x
  sorry

end range_f_l1392_139212


namespace qin_jiushao_value_l1392_139225

def polynomial (x : ℤ) : ℤ :=
  2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

def step1 (x : ℤ) : ℤ := 2 * x + 5
def step2 (x : ℤ) (v : ℤ) : ℤ := v * x + 8
def step3 (x : ℤ) (v : ℤ) : ℤ := v * x + 7
def step_v3 (x : ℤ) (v : ℤ) : ℤ := v * x - 6

theorem qin_jiushao_value (x : ℤ) (v3 : ℤ) (h1 : x = 3) (h2 : v3 = 130) :
  step_v3 3 (step3 3 (step2 3 (step1 3))) = v3 :=
by {
  sorry
}

end qin_jiushao_value_l1392_139225


namespace polynomial_expansion_l1392_139202

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) =
  -6 * t^5 + 5 * t^4 + 2 * t^3 - 26 * t^2 + 25 * t - 12 :=
by sorry

end polynomial_expansion_l1392_139202


namespace cost_of_apples_and_oranges_correct_l1392_139224

-- Define the initial money jasmine had
def initial_money : ℝ := 100.00

-- Define the remaining money after purchase
def remaining_money : ℝ := 85.00

-- Define the cost of apples and oranges
def cost_of_apples_and_oranges : ℝ := initial_money - remaining_money

-- This is our theorem statement that needs to be proven
theorem cost_of_apples_and_oranges_correct :
  cost_of_apples_and_oranges = 15.00 :=
by
  sorry

end cost_of_apples_and_oranges_correct_l1392_139224


namespace matrix_determinant_eq_16_l1392_139220

theorem matrix_determinant_eq_16 (x : ℝ) :
  (3 * x) * (4 * x) - (2 * x) = 16 ↔ x = 4 / 3 ∨ x = -1 :=
by sorry

end matrix_determinant_eq_16_l1392_139220


namespace sin_double_angle_l1392_139258

theorem sin_double_angle (θ : ℝ) (h : Real.sin (π / 4 + θ) = 1 / 3) : Real.sin (2 * θ) = -7 / 9 :=
by
  sorry

end sin_double_angle_l1392_139258


namespace rectangle_area_change_area_analysis_l1392_139221

noncomputable def original_area (a b : ℝ) : ℝ := a * b

noncomputable def new_area (a b : ℝ) : ℝ := (a - 3) * (b + 3)

theorem rectangle_area_change (a b : ℝ) :
  let S := original_area a b
  let S₁ := new_area a b
  S₁ - S = 3 * (a - b - 3) :=
by
  sorry

theorem area_analysis (a b : ℝ) :
  if a - b - 3 = 0 then new_area a b = original_area a b
  else if a - b - 3 > 0 then new_area a b > original_area a b
  else new_area a b < original_area a b :=
by
  sorry

end rectangle_area_change_area_analysis_l1392_139221


namespace carpet_dimensions_l1392_139270

-- Define the problem parameters
def width_a : ℕ := 50
def width_b : ℕ := 38

-- The dimensions x and y are integral numbers of feet
variables (x y : ℕ)

-- The same length L for both rooms that touches all four walls
noncomputable def length (x y : ℕ) : ℚ := (22 * (x^2 + y^2)) / (x * y)

-- The final theorem to be proven
theorem carpet_dimensions (x y : ℕ) (h : (x^2 + y^2) * 1056 = (x * y) * 48 * (length x y)) : (x = 50) ∧ (y = 25) :=
by
  sorry -- Proof is omitted

end carpet_dimensions_l1392_139270


namespace segment_AB_length_l1392_139261

-- Defining the conditions
def area_ratio (AB CD : ℝ) : Prop := AB / CD = 5 / 2
def length_sum (AB CD : ℝ) : Prop := AB + CD = 280

-- The theorem stating the problem
theorem segment_AB_length (AB CD : ℝ) (h₁ : area_ratio AB CD) (h₂ : length_sum AB CD) : AB = 200 :=
by {
  -- Proof step would be inserted here, but it is omitted as per instructions
  sorry
}

end segment_AB_length_l1392_139261


namespace replace_asterisks_l1392_139294

theorem replace_asterisks (x : ℝ) (h : (x / 20) * (x / 80) = 1) : x = 40 :=
sorry

end replace_asterisks_l1392_139294


namespace number_of_valid_groupings_l1392_139230

-- Definitions based on conditions
def num_guides : ℕ := 2
def num_tourists : ℕ := 6
def total_groupings : ℕ := 2 ^ num_tourists
def invalid_groupings : ℕ := 2  -- All tourists go to one guide either a or b

-- The theorem to prove
theorem number_of_valid_groupings : total_groupings - invalid_groupings = 62 :=
by sorry

end number_of_valid_groupings_l1392_139230


namespace distance_from_P_to_y_axis_l1392_139278

theorem distance_from_P_to_y_axis 
  (x y : ℝ)
  (h1 : (x^2 / 16) + (y^2 / 25) = 1)
  (F1 : ℝ × ℝ := (0, -3))
  (F2 : ℝ × ℝ := (0, 3))
  (h2 : (F1.1 - x)^2 + (F1.2 - y)^2 = 9 ∨ (F2.1 - x)^2 + (F2.2 - y)^2 = 9 
          ∨ (F1.1 - x)^2 + (F1.2 - y)^2 + (F2.1 - x)^2 + (F2.2 - y)^2 = (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) :
  |x| = 16 / 5 :=
by
  sorry

end distance_from_P_to_y_axis_l1392_139278


namespace brother_age_l1392_139206

variables (M B : ℕ)

theorem brother_age (h1 : M = B + 12) (h2 : M + 2 = 2 * (B + 2)) : B = 10 := by
  sorry

end brother_age_l1392_139206


namespace number_of_girls_in_class_l1392_139259

theorem number_of_girls_in_class (B S G : ℕ)
  (h1 : 3 * B = 4 * 18)  -- 3/4 * B = 18
  (h2 : 2 * S = 3 * B)  -- 2/3 * S = B
  (h3 : G = S - B) : G = 12 :=
by
  sorry

end number_of_girls_in_class_l1392_139259


namespace angle_between_hands_230_pm_l1392_139205

def hour_hand_position (hour minute : ℕ) : ℕ := hour % 12 * 5 + minute / 12
def minute_hand_position (minute : ℕ) : ℕ := minute
def divisions_to_angle (divisions : ℕ) : ℕ := divisions * 30

theorem angle_between_hands_230_pm :
    hour_hand_position 2 30 = 2 * 5 + 30 / 12 ∧
    minute_hand_position 30 = 30 ∧
    divisions_to_angle (minute_hand_position 30 / 5 - hour_hand_position 2 30 / 5) = 105 :=
by {
    sorry
}

end angle_between_hands_230_pm_l1392_139205


namespace find_k_for_perfect_square_trinomial_l1392_139255

noncomputable def perfect_square_trinomial (k : ℝ) : Prop :=
∀ x : ℝ, (x^2 - 8*x + k) = (x - 4)^2

theorem find_k_for_perfect_square_trinomial :
  ∃ k : ℝ, perfect_square_trinomial k ∧ k = 16 :=
by
  use 16
  sorry

end find_k_for_perfect_square_trinomial_l1392_139255


namespace interest_rate_l1392_139279

noncomputable def simple_interest (P r t: ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t: ℝ) : ℝ := P * (1 + r / 100) ^ t - P

theorem interest_rate (P r: ℝ) (h1: simple_interest P r 2 = 50) (h2: compound_interest P r 2 = 51.25) : r = 5 :=
by
  sorry

end interest_rate_l1392_139279


namespace trigonometric_relationship_l1392_139289

-- Given conditions
variables (x : ℝ) (a b c : ℝ)

-- Required conditions
variables (h1 : π / 4 < x) (h2 : x < π / 2)
variables (ha : a = Real.sin x)
variables (hb : b = Real.cos x)
variables (hc : c = Real.tan x)

-- Proof goal
theorem trigonometric_relationship : b < a ∧ a < c :=
by
  -- Proof will go here
  sorry

end trigonometric_relationship_l1392_139289


namespace r_needs_35_days_l1392_139226

def work_rate (P Q R: ℚ) : Prop :=
  (P = Q + R) ∧ (P + Q = 1/10) ∧ (Q = 1/28)

theorem r_needs_35_days (P Q R: ℚ) (h: work_rate P Q R) : 1 / R = 35 :=
by 
  sorry

end r_needs_35_days_l1392_139226


namespace total_cards_l1392_139239

def basketball_boxes : ℕ := 12
def cards_per_basketball_box : ℕ := 20
def football_boxes : ℕ := basketball_boxes - 5
def cards_per_football_box : ℕ := 25

theorem total_cards : basketball_boxes * cards_per_basketball_box + football_boxes * cards_per_football_box = 415 := by
  sorry

end total_cards_l1392_139239


namespace fish_population_estimation_l1392_139272

def tagged_fish_day1 := (30, 25, 25) -- (Species A, Species B, Species C)
def tagged_fish_day2 := (40, 35, 25) -- (Species A, Species B, Species C)
def caught_fish_day3 := (60, 50, 30) -- (Species A, Species B, Species C)
def tagged_fish_day3 := (4, 6, 2)    -- (Species A, Species B, Species C)
def caught_fish_day4 := (70, 40, 50) -- (Species A, Species B, Species C)
def tagged_fish_day4 := (5, 7, 3)    -- (Species A, Species B, Species C)

def total_tagged_fish (day1 : (ℕ × ℕ × ℕ)) (day2 : (ℕ × ℕ × ℕ)) :=
  let (a1, b1, c1) := day1
  let (a2, b2, c2) := day2
  (a1 + a2, b1 + b2, c1 + c2)

def average_proportion_tagged (caught3 tagged3 caught4 tagged4 : (ℕ × ℕ × ℕ)) :=
  let (c3a, c3b, c3c) := caught3
  let (t3a, t3b, t3c) := tagged3
  let (c4a, c4b, c4c) := caught4
  let (t4a, t4b, t4c) := tagged4
  ((t3a / c3a + t4a / c4a) / 2,
   (t3b / c3b + t4b / c4b) / 2,
   (t3c / c3c + t4c / c4c) / 2)

def estimate_population (total_tagged average_proportion : (ℕ × ℕ × ℕ)) :=
  let (ta, tb, tc) := total_tagged
  let (pa, pb, pc) := average_proportion
  (ta / pa, tb / pb, tc / pc)

theorem fish_population_estimation :
  let total_tagged := total_tagged_fish tagged_fish_day1 tagged_fish_day2
  let avg_prop := average_proportion_tagged caught_fish_day3 tagged_fish_day3 caught_fish_day4 tagged_fish_day4
  estimate_population total_tagged avg_prop = (1014, 407, 790) :=
by
  sorry

end fish_population_estimation_l1392_139272


namespace power_multiplication_l1392_139295

theorem power_multiplication (x : ℝ) : (-4 * x^3)^2 = 16 * x^6 := 
by 
  sorry

end power_multiplication_l1392_139295


namespace four_digit_numbers_divisible_by_5_l1392_139216

theorem four_digit_numbers_divisible_by_5 : 
  let smallest_4_digit := 1000
  let largest_4_digit := 9999
  let divisible_by_5 (n : ℕ) := ∃ k : ℕ, n = 5 * k
  ∃ n : ℕ, ( ∀ x : ℕ, smallest_4_digit ≤ x ∧ x ≤ largest_4_digit ∧ divisible_by_5 x ↔ (smallest_4_digit + (n-1) * 5 = x) ) ∧ n = 1800 :=
by
  sorry

end four_digit_numbers_divisible_by_5_l1392_139216


namespace problem_1_problem_2_l1392_139232

def simplify_calc : Prop :=
  125 * 3.2 * 25 = 10000

def solve_equation : Prop :=
  ∀ x: ℝ, 24 * (x - 12) = 16 * (x - 4) → x = 28

theorem problem_1 : simplify_calc :=
by
  sorry

theorem problem_2 : solve_equation :=
by
  sorry

end problem_1_problem_2_l1392_139232


namespace kathleen_remaining_money_l1392_139260

-- Define the conditions
def saved_june := 21
def saved_july := 46
def saved_august := 45
def spent_school_supplies := 12
def spent_clothes := 54
def aunt_gift_threshold := 125
def aunt_gift := 25

-- Prove that Kathleen has the correct remaining amount of money
theorem kathleen_remaining_money : 
    (saved_june + saved_july + saved_august) - 
    (spent_school_supplies + spent_clothes) = 46 := 
by
  sorry

end kathleen_remaining_money_l1392_139260


namespace tan_alpha_minus_pi_over_4_l1392_139207

open Real

theorem tan_alpha_minus_pi_over_4
  (α : ℝ)
  (a b : ℝ × ℝ)
  (h1 : a = (cos α, -2))
  (h2 : b = (sin α, 1))
  (h3 : ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) :
  tan (α - π / 4) = -3 := 
sorry

end tan_alpha_minus_pi_over_4_l1392_139207


namespace intersection_of_M_and_N_l1392_139285

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l1392_139285


namespace min_value_of_fraction_l1392_139248

theorem min_value_of_fraction (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + 3 * b = 2) : 
  ∃ m, (∀ (a b : ℝ), a > 0 → b > 0 → a + 3 * b = 2 → 1 / a + 3 / b ≥ m) ∧ m = 8 := 
by
  sorry

end min_value_of_fraction_l1392_139248


namespace sequence_formula_l1392_139292

-- Defining the sequence and the conditions
def bounded_seq (a : ℕ → ℝ) : Prop :=
  ∃ C > 0, ∀ n, |a n| ≤ C

-- Statement of the problem in Lean
theorem sequence_formula (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = 3 * a n - 4) →
  bounded_seq a →
  ∀ n : ℕ, a n = 2 :=
by
  intros h1 h2
  sorry

end sequence_formula_l1392_139292


namespace smallest_side_of_triangle_l1392_139238

theorem smallest_side_of_triangle (A B C : ℝ) (a b c : ℝ) 
  (hA : A = 60) (hC : C = 45) (hb : b = 4) (h_sum : A + B + C = 180) : 
  c = 4 * Real.sqrt 3 - 4 := 
sorry

end smallest_side_of_triangle_l1392_139238


namespace determine_Y_in_arithmetic_sequence_matrix_l1392_139250

theorem determine_Y_in_arithmetic_sequence_matrix :
  (exists a₁ a₂ a₃ a₄ a₅ : ℕ, 
    -- Conditions for the first row (arithmetic sequence with first term 3 and fifth term 15)
    a₁ = 3 ∧ a₅ = 15 ∧ 
    (∃ d₁ : ℕ, a₂ = a₁ + d₁ ∧ a₃ = a₂ + d₁ ∧ a₄ = a₃ + d₁ ∧ a₅ = a₄ + d₁) ∧

    -- Conditions for the fifth row (arithmetic sequence with first term 25 and fifth term 65)
    a₁ = 25 ∧ a₅ = 65 ∧ 
    (∃ d₅ : ℕ, a₂ = a₁ + d₅ ∧ a₃ = a₂ + d₅ ∧ a₄ = a₃ + d₅ ∧ a₅ = a₄ + d₅) ∧

    -- Middle element Y
    a₃ = 27) :=
sorry

end determine_Y_in_arithmetic_sequence_matrix_l1392_139250


namespace points_with_tangent_length_six_l1392_139249

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 4 = 0

-- Define the property of a point having a tangent of length 6 to the circle
def tangent_length_six (h k cx cy r : ℝ) : Prop :=
  (cx - h)^2 + (cy - k)^2 - r^2 = 36

-- Main theorem statement
theorem points_with_tangent_length_six : 
  (∀ x1 y1 : ℝ, (x1 = -4 ∧ y1 = 6) ∨ (x1 = 5 ∧ y1 = -3) → 
    (∃ r1 : ℝ, tangent_length_six x1 y1 (-1) 0 3) ∧ 
    (∃ r2 : ℝ, tangent_length_six x1 y1 2 3 3)) :=
  by 
  sorry

end points_with_tangent_length_six_l1392_139249


namespace shape_formed_is_line_segment_l1392_139236

def point := (ℝ × ℝ)

noncomputable def A : point := (0, 0)
noncomputable def B : point := (0, 4)
noncomputable def C : point := (6, 4)
noncomputable def D : point := (6, 0)

noncomputable def line_eq (p1 p2 : point) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x2 - x1, y2 - y1)

theorem shape_formed_is_line_segment :
  let l1 := line_eq A (1, 1)  -- Line from A at 45°
  let l2 := line_eq B (-1, -1) -- Line from B at -45°
  let l3 := line_eq D (1, -1) -- Line from D at 45°
  let l4 := line_eq C (-1, 5) -- Line from C at -45°
  let intersection1 := (5, 5)  -- Intersection of l1 and l4: solve x = 10 - x
  let intersection2 := (5, -1)  -- Intersection of l2 and l3: solve 4 - x = x - 6
  intersection1.1 = intersection2.1 := 
by
  sorry

end shape_formed_is_line_segment_l1392_139236


namespace geometric_sum_S5_l1392_139267

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)

def geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a_n (n+1) = a_n n * q

theorem geometric_sum_S5 (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom : geometric_sequence a_n)
  (h_cond1 : a_n 2 * a_n 3 = 8 * a_n 1)
  (h_cond2 : (a_n 4 + 2 * a_n 5) / 2 = 20) :
  S 5 = 31 :=
sorry

end geometric_sum_S5_l1392_139267


namespace tan_a_values_l1392_139282

theorem tan_a_values (a : ℝ) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1 / 2 :=
by
  sorry

end tan_a_values_l1392_139282
