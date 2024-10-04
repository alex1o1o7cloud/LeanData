import Mathlib

namespace rectangle_area_with_inscribed_circle_l215_215126

theorem rectangle_area_with_inscribed_circle (w h r : ℝ)
  (hw : ∀ O : ℝ × ℝ, dist O (w/2, h/2) = r)
  (hw_eq_h : w = h) :
  w * h = 2 * r^2 := 
by
  sorry

end rectangle_area_with_inscribed_circle_l215_215126


namespace most_probable_light_l215_215651

theorem most_probable_light (red_duration : ℕ) (yellow_duration : ℕ) (green_duration : ℕ) :
  red_duration = 30 ∧ yellow_duration = 5 ∧ green_duration = 40 →
  (green_duration / (red_duration + yellow_duration + green_duration) > red_duration / (red_duration + yellow_duration + green_duration)) ∧
  (green_duration / (red_duration + yellow_duration + green_duration) > yellow_duration / (red_duration + yellow_duration + green_duration)) :=
by
  sorry

end most_probable_light_l215_215651


namespace positive_integers_powers_of_3_l215_215958

theorem positive_integers_powers_of_3 (n : ℕ) (h : ∀ k : ℤ, ∃ a : ℤ, n ∣ a^3 + a - k) : ∃ b : ℕ, n = 3^b :=
sorry

end positive_integers_powers_of_3_l215_215958


namespace necessary_but_not_sufficient_l215_215688

variables (P Q : Prop)
variables (p : P) (q : Q)

-- Propositions
def quadrilateral_has_parallel_and_equal_sides : Prop := P
def is_rectangle : Prop := Q

-- Necessary but not sufficient condition
theorem necessary_but_not_sufficient (h : P → Q) : ¬(Q → P) :=
by sorry

end necessary_but_not_sufficient_l215_215688


namespace rectangles_in_5x5_grid_l215_215689

theorem rectangles_in_5x5_grid : 
  ∃ n : ℕ, n = 100 ∧ (∀ (grid : Fin 6 → Fin 6 → Prop), 
  (∃ (vlines hlines : Finset (Fin 6)),
   (vlines.card = 2 ∧ hlines.card = 2) ∧
   n = (vlines.card.choose 2) * (hlines.card.choose 2))) :=
by
  sorry

end rectangles_in_5x5_grid_l215_215689


namespace max_neg_square_in_interval_l215_215056

variable (f : ℝ → ℝ)

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, 0 < x → x < y → f x < f y

noncomputable def neg_square_val (x : ℝ) : ℝ :=
  - (f x) ^ 2

theorem max_neg_square_in_interval : 
  (∀ x_1 x_2 : ℝ, f (x_1 + x_2) = f x_1 + f x_2) →
  f 1 = 2 →
  is_increasing f →
  (∀ x : ℝ, f (-x) = - f x) →
  ∃ b ∈ (Set.Icc (-3) (-2)), 
  ∀ x ∈ (Set.Icc (-3) (-2)), neg_square_val f x ≤ neg_square_val f b ∧ neg_square_val f b = -16 := 
sorry

end max_neg_square_in_interval_l215_215056


namespace cost_per_pound_of_mixed_candy_l215_215914

def w1 := 10
def p1 := 8
def w2 := 20
def p2 := 5

theorem cost_per_pound_of_mixed_candy : 
    (w1 * p1 + w2 * p2) / (w1 + w2) = 6 := by
  sorry

end cost_per_pound_of_mixed_candy_l215_215914


namespace bamboo_sections_length_l215_215847

variable {n d : ℕ} (a : ℕ → ℕ)
variable (h_arith : ∀ k, a (k + 1) = a k + d)
variable (h_top : a 1 = 10)
variable (h_sum_last_three : a n + a (n - 1) + a (n - 2) = 114)
variable (h_geom_6 : (a 6) ^ 2 = a 1 * a n)

theorem bamboo_sections_length : n = 16 := 
by 
  sorry

end bamboo_sections_length_l215_215847


namespace bridge_length_correct_l215_215524

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 72
noncomputable def crossing_time : ℝ := 12.399008079353651

-- converting train speed from km/hr to m/s
noncomputable def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- total length the train covers to cross the bridge
noncomputable def total_length : ℝ := train_speed_m_per_s * crossing_time

-- length of the bridge
noncomputable def bridge_length : ℝ := total_length - train_length

theorem bridge_length_correct :
  bridge_length = 137.98 :=
by 
  sorry

end bridge_length_correct_l215_215524


namespace intersection_complement_l215_215676

open Set

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | 2 < x}
def R_complement_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement : M ∩ R_complement_N = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_l215_215676


namespace product_of_t_values_l215_215815

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l215_215815


namespace river_width_l215_215117

variable (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ)

-- Define the given conditions:
def depth_of_river : ℝ := 4
def flow_rate : ℝ := 4
def volume_per_minute_water : ℝ := 10666.666666666666

-- The proposition to prove:
theorem river_width :
  let flow_rate_m_per_min := (flow_rate * 1000) / 60
  let width := volume_per_minute / (flow_rate_m_per_min * depth)
  width = 40 :=
by
  sorry

end river_width_l215_215117


namespace sheep_remain_l215_215345

theorem sheep_remain : ∀ (total_sheep sister_share brother_share : ℕ),
  total_sheep = 400 →
  sister_share = total_sheep / 4 →
  brother_share = (total_sheep - sister_share) / 2 →
  (total_sheep - sister_share - brother_share) = 150 :=
by
  intros total_sheep sister_share brother_share h_total h_sister h_brother
  rw [h_total, h_sister, h_brother]
  sorry

end sheep_remain_l215_215345


namespace complex_division_l215_215125

theorem complex_division :
  (1 - 2 * Complex.I) / (2 + Complex.I) = -Complex.I :=
by sorry

end complex_division_l215_215125


namespace balls_in_boxes_l215_215062

-- Define the conditions
def num_balls : ℕ := 3
def num_boxes : ℕ := 4

-- Define the problem
theorem balls_in_boxes : (num_boxes ^ num_balls) = 64 :=
by
  -- We acknowledge that we are skipping the proof details here
  sorry

end balls_in_boxes_l215_215062


namespace arccos_neg_one_eq_pi_l215_215941

theorem arccos_neg_one_eq_pi : real.arccos (-1) = real.pi :=
by sorry

end arccos_neg_one_eq_pi_l215_215941


namespace lower_base_length_l215_215649

variable (A B C D E : Type)
variable (AD BD BE DE : ℝ)

-- Conditions of the problem
axiom hAD : AD = 12  -- upper base
axiom hBD : BD = 18  -- height
axiom hBE_DE : BE = 2 * DE  -- ratio BE = 2 * DE

-- Define the trapezoid with given lengths and conditions
def trapezoid_exists (A B C D : Type) (AD BD BE DE : ℝ) :=
  AD = 12 ∧ BD = 18 ∧ BE = 2 * DE

-- The length of BC to be proven
def BC : ℝ := 24

-- The theorem to be proven
theorem lower_base_length (h : trapezoid_exists A B C D AD BD BE DE) : BC = 2 * AD :=
by
  sorry

end lower_base_length_l215_215649


namespace find_first_term_of_geometric_progression_l215_215750

theorem find_first_term_of_geometric_progression
  (a_2 : ℝ) (a_3 : ℝ) (a_1 : ℝ) (q : ℝ)
  (h1 : a_2 = a_1 * q)
  (h2 : a_3 = a_1 * q^2)
  (h3 : a_2 = 5)
  (h4 : a_3 = 1) : a_1 = 25 :=
by
  sorry

end find_first_term_of_geometric_progression_l215_215750


namespace percentage_difference_l215_215115

variables (P P' : ℝ)

theorem percentage_difference (h : P' = 1.25 * P) :
  ((P' - P) / P') * 100 = 20 :=
by sorry

end percentage_difference_l215_215115


namespace surface_area_of_cube_l215_215176

-- Define the volume condition
def volume_of_cube (s : ℝ) := s^3 = 125

-- Define the conversion from decimeters to centimeters
def decimeters_to_centimeters (d : ℝ) := d * 10

-- Define the surface area formula for one side of the cube
def surface_area_one_side (s_cm : ℝ) := s_cm^2

-- Prove that given the volume condition, the surface area of one side is 2500 cm²
theorem surface_area_of_cube
  (s : ℝ)
  (h : volume_of_cube s)
  (s_cm : ℝ := decimeters_to_centimeters s) :
  surface_area_one_side s_cm = 2500 :=
by
  sorry

end surface_area_of_cube_l215_215176


namespace height_in_meters_l215_215762

theorem height_in_meters (h: 1 * 100 + 36 = 136) : 1.36 = 1 + 36 / 100 :=
by 
  -- proof steps will go here
  sorry

end height_in_meters_l215_215762


namespace tan_pi_seven_product_eq_sqrt_seven_l215_215061

theorem tan_pi_seven_product_eq_sqrt_seven :
  (Real.tan (Real.pi / 7)) * (Real.tan (2 * Real.pi / 7)) * (Real.tan (3 * Real.pi / 7)) = Real.sqrt 7 :=
by
  sorry

end tan_pi_seven_product_eq_sqrt_seven_l215_215061


namespace nails_remaining_l215_215517

theorem nails_remaining (nails_initial : ℕ) (kitchen_fraction : ℚ) (fence_fraction : ℚ) (nails_used_kitchen : ℕ) (nails_remaining_after_kitchen : ℕ) (nails_used_fence : ℕ) (nails_remaining_final : ℕ) 
  (h1 : nails_initial = 400) 
  (h2 : kitchen_fraction = 0.30) 
  (h3 : nails_used_kitchen = kitchen_fraction * nails_initial) 
  (h4 : nails_remaining_after_kitchen = nails_initial - nails_used_kitchen) 
  (h5 : fence_fraction = 0.70) 
  (h6 : nails_used_fence = fence_fraction * nails_remaining_after_kitchen) 
  (h7 : nails_remaining_final = nails_remaining_after_kitchen - nails_used_fence) :
  nails_remaining_final = 84 := by
sorry

end nails_remaining_l215_215517


namespace arithmetic_sequence_a_100_l215_215580

theorem arithmetic_sequence_a_100 :
  ∀ (a : ℕ → ℕ), 
  (a 1 = 100) → 
  (∀ n : ℕ, a (n + 1) = a n + 2) → 
  a 100 = 298 :=
by
  intros a h1 hrec
  sorry

end arithmetic_sequence_a_100_l215_215580


namespace find_primes_l215_215801

open Int

theorem find_primes (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p ^ x = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by
  sorry

end find_primes_l215_215801


namespace no_solution_exists_l215_215290

theorem no_solution_exists : ¬ ∃ n : ℕ, (n^2 ≡ 1 [MOD 5]) ∧ (n^3 ≡ 3 [MOD 5]) := 
sorry

end no_solution_exists_l215_215290


namespace sum_of_digits_82_l215_215082

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_82 : sum_of_digits 82 = 10 := by
  sorry

end sum_of_digits_82_l215_215082


namespace delta_ratio_l215_215420

theorem delta_ratio 
  (Δx : ℝ) (Δy : ℝ) 
  (y_new : ℝ := (1 + Δx)^2 + 1)
  (y_old : ℝ := 1^2 + 1)
  (Δy_def : Δy = y_new - y_old) :
  Δy / Δx = 2 + Δx :=
by
  sorry

end delta_ratio_l215_215420


namespace convert_angle_l215_215660

theorem convert_angle (α : ℝ) (k : ℤ) :
  -1485 * (π / 180) = α + 2 * k * π ∧ 0 ≤ α ∧ α < 2 * π ∧ k = -10 ∧ α = 7 * π / 4 :=
by
  sorry

end convert_angle_l215_215660


namespace vector_BC_l215_215990

def vector_subtraction (v1 v2 : ℤ × ℤ) : ℤ × ℤ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem vector_BC (BA CA BC : ℤ × ℤ) (hBA : BA = (2, 3)) (hCA : CA = (4, 7)) :
  BC = vector_subtraction BA CA → BC = (-2, -4) :=
by
  intro hBC
  rw [vector_subtraction, hBA, hCA] at hBC
  simpa using hBC

end vector_BC_l215_215990


namespace find_m_range_l215_215554

noncomputable def ellipse_symmetric_points_range (m : ℝ) : Prop :=
  -((2:ℝ) * Real.sqrt (13:ℝ) / 13) < m ∧ m < ((2:ℝ) * Real.sqrt (13:ℝ) / 13)

theorem find_m_range :
  ∃ m : ℝ, ellipse_symmetric_points_range m :=
sorry

end find_m_range_l215_215554


namespace product_mod_32_l215_215009

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215009


namespace A_sym_diff_B_l215_215542

-- Definitions of sets and operations
def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {y | ∃ x : ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x : ℝ, y = -(x-1)^2 + 2}

-- The target equality to prove
theorem A_sym_diff_B : sym_diff A B = (({y | y ≤ 0}) ∪ ({y | y > 2})) :=
by
  sorry

end A_sym_diff_B_l215_215542


namespace number_of_sides_of_polygon_l215_215089

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end number_of_sides_of_polygon_l215_215089


namespace least_number_to_make_divisible_l215_215624

def least_common_multiple (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem least_number_to_make_divisible (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : 
  least_common_multiple a b = 77 → 
  (n % least_common_multiple a b) = 40 →
  c = (least_common_multiple a b - (n % least_common_multiple a b)) →
  c = 37 :=
by
sorry

end least_number_to_make_divisible_l215_215624


namespace largest_stamps_per_page_l215_215336

theorem largest_stamps_per_page (h1 : Nat := 1050) (h2 : Nat := 1260) (h3 : Nat := 1470) :
  Nat.gcd h1 (Nat.gcd h2 h3) = 210 :=
by
  sorry

end largest_stamps_per_page_l215_215336


namespace least_gamma_l215_215585

theorem least_gamma (n : ℕ) (hn : n ≥ 2)
    (x : Fin n → ℝ) (hx : (∀ i, x i > 0) ∧ (∑ i, x i = 1))
    (y : Fin n → ℝ) (hy : (∀ i, 0 ≤ y i ∧ y i ≤ 1/2) ∧ (∑ i, y i = 1)) :
    ∃ γ, γ = 1 / (2 * (n - 1)^(n - 1)) ∧ (∏ i, x i) ≤ γ * (∑ i, x i * y i) := 
by
  sorry

end least_gamma_l215_215585


namespace product_of_t_values_l215_215813

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l215_215813


namespace ashley_cocktail_calories_l215_215271

theorem ashley_cocktail_calories:
  let mango_grams := 150
  let honey_grams := 200
  let water_grams := 300
  let vodka_grams := 100

  let mango_cal_per_100g := 60
  let honey_cal_per_100g := 640
  let vodka_cal_per_100g := 70
  let water_cal_per_100g := 0

  let total_cocktail_grams := mango_grams + honey_grams + water_grams + vodka_grams
  let total_cocktail_calories := (mango_grams * mango_cal_per_100g / 100) +
                                 (honey_grams * honey_cal_per_100g / 100) +
                                 (vodka_grams * vodka_cal_per_100g / 100) +
                                 (water_grams * water_cal_per_100g / 100)
  let caloric_density := total_cocktail_calories / total_cocktail_grams
  let result := 300 * caloric_density
  result = 576 := by
  sorry

end ashley_cocktail_calories_l215_215271


namespace number_of_sides_of_polygon_l215_215091

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end number_of_sides_of_polygon_l215_215091


namespace fernanda_savings_calc_l215_215646

noncomputable def aryan_debt : ℝ := 1200
noncomputable def kyro_debt : ℝ := aryan_debt / 2
noncomputable def aryan_payment : ℝ := 0.60 * aryan_debt
noncomputable def kyro_payment : ℝ := 0.80 * kyro_debt
noncomputable def initial_savings : ℝ := 300
noncomputable def total_payment_received : ℝ := aryan_payment + kyro_payment
noncomputable def total_savings : ℝ := initial_savings + total_payment_received

theorem fernanda_savings_calc : total_savings = 1500 := by
  sorry

end fernanda_savings_calc_l215_215646


namespace union_A_B_l215_215684

open Set

-- Define the sets A and B
def setA : Set ℝ := { x | abs x < 3 }
def setB : Set ℝ := { x | x - 1 ≤ 0 }

-- State the theorem we want to prove
theorem union_A_B : setA ∪ setB = { x : ℝ | x < 3 } :=
by
  -- Skip the proof
  sorry

end union_A_B_l215_215684


namespace earthquake_energy_multiple_l215_215779

theorem earthquake_energy_multiple (E : ℕ → ℝ) (n9 n7 : ℕ)
  (h1 : E n9 = 10 ^ n9) 
  (h2 : E n7 = 10 ^ n7) 
  (hn9 : n9 = 9) 
  (hn7 : n7 = 7) : 
  E n9 / E n7 = 100 := 
by 
  sorry

end earthquake_energy_multiple_l215_215779


namespace function_relation_l215_215054

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 4*x + c

theorem function_relation (c : ℝ) :
  f 1 c > f 0 c ∧ f 0 c > f (-2) c := by
  sorry

end function_relation_l215_215054


namespace product_mod_32_l215_215007

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215007


namespace pages_left_to_read_l215_215532

-- Define the given conditions
def total_pages : ℕ := 563
def pages_read : ℕ := 147

-- Define the proof statement
theorem pages_left_to_read : total_pages - pages_read = 416 :=
by
  -- The proof will be given here
  sorry

end pages_left_to_read_l215_215532


namespace no_int_k_such_that_P_k_equals_8_l215_215465

theorem no_int_k_such_that_P_k_equals_8
    (P : Polynomial ℤ) 
    (a b c d k : ℤ)
    (h0: a ≠ b)
    (h1: a ≠ c)
    (h2: a ≠ d)
    (h3: b ≠ c)
    (h4: b ≠ d)
    (h5: c ≠ d)
    (h6: P.eval a = 5)
    (h7: P.eval b = 5)
    (h8: P.eval c = 5)
    (h9: P.eval d = 5)
    : P.eval k ≠ 8 := by
  sorry

end no_int_k_such_that_P_k_equals_8_l215_215465


namespace diego_can_carry_home_l215_215412

theorem diego_can_carry_home (T W G O A : ℕ) (hT : T = 20) (hW : W = 1) (hG : G = 1) (hO : O = 1) : A = T - (W + G + O) → A = 17 := by
  sorry

end diego_can_carry_home_l215_215412


namespace trapezoid_other_base_possible_lengths_l215_215766

-- Definition of the trapezoid problem.
structure Trapezoid where
  height : ℕ
  leg1 : ℕ
  leg2 : ℕ
  base1 : ℕ

-- The given conditions
def trapezoid_data : Trapezoid :=
{ height := 12, leg1 := 20, leg2 := 15, base1 := 42 }

-- The proof problem in Lean 4 statement
theorem trapezoid_other_base_possible_lengths (t : Trapezoid) :
  t = trapezoid_data → (∃ b : ℕ, (b = 17 ∨ b = 35)) :=
by
  intro h_data_eq
  sorry

end trapezoid_other_base_possible_lengths_l215_215766


namespace ternary_predecessor_l215_215987

theorem ternary_predecessor (M : ℕ) (h : M = 2 * 3^4 + 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) : 
  prev_ternary(M) = "21020" :=
by
  sorry

end ternary_predecessor_l215_215987


namespace car_turns_proof_l215_215265

def turns_opposite_direction (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 180

theorem car_turns_proof
  (angle1 angle2 : ℝ)
  (h1 : (angle1 = 50 ∧ angle2 = 130) ∨ (angle1 = -50 ∧ angle2 = 130) ∨ 
       (angle1 = 50 ∧ angle2 = -130) ∨ (angle1 = 30 ∧ angle2 = -30)) :
  turns_opposite_direction angle1 angle2 ↔ (angle1 = 50 ∧ angle2 = 130) :=
by
  sorry

end car_turns_proof_l215_215265


namespace letter_arrangements_proof_l215_215441

noncomputable def arrangements := 
  ∑ k in Finset.range 6, (Nat.choose 5 k) ^ 3

theorem letter_arrangements_proof :
  (∑ k in Finset.range 6, (Nat.choose 5 k) ^ 3) = arrangements := 
  by 
  sorry

end letter_arrangements_proof_l215_215441


namespace fibonacci_contains_21_l215_215995

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 => 1
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Theorem statement: Proving that 21 is in the Fibonacci sequence
theorem fibonacci_contains_21 : ∃ n, fibonacci n = 21 :=
by
  sorry

end fibonacci_contains_21_l215_215995


namespace knights_wins_33_l215_215450

def sharks_wins : ℕ := sorry
def falcons_wins : ℕ := sorry
def knights_wins : ℕ := sorry
def wolves_wins : ℕ := sorry
def dragons_wins : ℕ := 38 -- Dragons won the most games

-- Condition 1: The Sharks won more games than the Falcons.
axiom sharks_won_more_than_falcons : sharks_wins > falcons_wins

-- Condition 2: The Knights won more games than the Wolves, but fewer than the Dragons.
axiom knights_won_more_than_wolves : knights_wins > wolves_wins
axiom knights_won_less_than_dragons : knights_wins < dragons_wins

-- Condition 3: The Wolves won more than 22 games.
axiom wolves_won_more_than_22 : wolves_wins > 22

-- The possible wins are 24, 27, 33, 36, and 38 and the dragons win 38 (already accounted in dragons_wins)

-- Prove that the Knights won 33 games.
theorem knights_wins_33 : knights_wins = 33 :=
sorry -- proof goes here

end knights_wins_33_l215_215450


namespace chord_length_eq_l215_215249

noncomputable def length_of_chord (radius : ℝ) (distance_to_chord : ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - distance_to_chord^2)

theorem chord_length_eq {radius distance_to_chord : ℝ} (h_radius : radius = 5) (h_distance : distance_to_chord = 4) :
  length_of_chord radius distance_to_chord = 6 :=
by
  sorry

end chord_length_eq_l215_215249


namespace bowling_ball_weight_l215_215129

theorem bowling_ball_weight :
  ∃ (b : ℝ) (c : ℝ),
    8 * b = 5 * c ∧
    4 * c = 100 ∧
    b = 15.625 :=
by 
  sorry

end bowling_ball_weight_l215_215129


namespace five_wednesdays_implies_five_saturdays_in_august_l215_215875

theorem five_wednesdays_implies_five_saturdays_in_august (N : ℕ) (H1 : ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ w ∈ ws, w < 32 ∧ (w % 7 = 3)) (H2 : July_days = 31) (H3 : August_days = 31):
  ∀ w : ℕ, w < 7 → ∃ ws : Finset ℕ, ws.card = 5 ∧ ∀ sat ∈ ws, (sat % 7 = 6) :=
by
  sorry

end five_wednesdays_implies_five_saturdays_in_august_l215_215875


namespace janice_class_girls_l215_215571

theorem janice_class_girls : ∃ (g b : ℕ), (3 * b = 4 * g) ∧ (g + b + 2 = 32) ∧ (g = 13) := by
  sorry

end janice_class_girls_l215_215571


namespace painted_surface_area_of_pyramid_l215_215401

/--
Given 19 unit cubes arranged in a 4-layer pyramid-like structure, where:
- The top layer has 1 cube,
- The second layer has 3 cubes,
- The third layer has 5 cubes,
- The bottom layer has 10 cubes,

Prove that the total painted surface area is 43 square meters.
-/
theorem painted_surface_area_of_pyramid :
  let layer1 := 1 -- top layer
  let layer2 := 3 -- second layer
  let layer3 := 5 -- third layer
  let layer4 := 10 -- bottom layer
  let total_cubes := layer1 + layer2 + layer3 + layer4
  let top_faces := layer1 * 1 + layer2 * 1 + layer3 * 1 + layer4 * 1
  let side_faces_layer1 := layer1 * 5
  let side_faces_layer2 := layer2 * 3
  let side_faces_layer3 := layer3 * 2
  let side_faces := side_faces_layer1 + side_faces_layer2 + side_faces_layer3
  let total_surface_area := top_faces + side_faces
  total_cubes = 19 → total_surface_area = 43 :=
by
  intros
  sorry

end painted_surface_area_of_pyramid_l215_215401


namespace locus_of_P_is_circle_l215_215970

def point_locus_is_circle (x y : ℝ) : Prop :=
  10 * real.sqrt ((x - 1)^2 + (y - 2)^2) = abs (3 * x - 4 * y)

theorem locus_of_P_is_circle :
  ∀ (x y : ℝ), point_locus_is_circle x y → (distance (x, y) (1, 2) = r) := 
sorry

end locus_of_P_is_circle_l215_215970


namespace usual_time_to_school_l215_215756

theorem usual_time_to_school (R : ℝ) (T : ℝ) (h : (17 / 13) * (T - 7) = T) : T = 29.75 :=
sorry

end usual_time_to_school_l215_215756


namespace find_real_roots_of_PQ_l215_215590

noncomputable def P (x b : ℝ) : ℝ := x^2 + x / 2 + b
noncomputable def Q (x c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_real_roots_of_PQ (b c d : ℝ)
  (h: ∀ x : ℝ, P x b * Q x c d = Q (P x b) c d)
  (h_d_zero: d = 0) :
  ∃ x : ℝ, P (Q x c d) b = 0 → x = (-c + Real.sqrt (c^2 + 2)) / 2 ∨ x = (-c - Real.sqrt (c^2 + 2)) / 2 :=
by
  sorry

end find_real_roots_of_PQ_l215_215590


namespace compare_variables_l215_215976

theorem compare_variables (a b c : ℝ) (h1 : a = 2 ^ (1 / 2)) (h2 : b = Real.log 3 / Real.log π) (h3 : c = Real.log (1 / 3) / Real.log 2) : 
  a > b ∧ b > c :=
by
  sorry

end compare_variables_l215_215976


namespace cannot_be_expressed_as_difference_of_squares_l215_215642

theorem cannot_be_expressed_as_difference_of_squares : 
  ¬ ∃ (a b : ℤ), 2006 = a^2 - b^2 :=
sorry

end cannot_be_expressed_as_difference_of_squares_l215_215642


namespace solve_for_y_l215_215361

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end solve_for_y_l215_215361


namespace women_count_l215_215754

def total_passengers : Nat := 54
def men : Nat := 18
def children : Nat := 10
def women : Nat := total_passengers - men - children

theorem women_count : women = 26 :=
sorry

end women_count_l215_215754


namespace product_of_odd_primes_mod_32_l215_215041

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215041


namespace product_of_odd_primes_mod_32_l215_215045

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215045


namespace sample_size_l215_215890

variable (num_classes : ℕ) (papers_per_class : ℕ)

theorem sample_size (h_classes : num_classes = 8) (h_papers : papers_per_class = 12) : 
  num_classes * papers_per_class = 96 := 
by 
  sorry

end sample_size_l215_215890


namespace clara_boxes_l215_215275

theorem clara_boxes (x : ℕ)
  (h1 : 12 * x + 20 * 80 + 16 * 70 = 3320) : x = 50 := by
  sorry

end clara_boxes_l215_215275


namespace triangle_medians_inequality_l215_215575

-- Define the parameters
variables {a b c t_a t_b t_c D : ℝ}

-- Assume the sides and medians of the triangle and the diameter of the circumcircle
axiom sides_of_triangle (a b c : ℝ) : Prop
axiom medians_of_triangle (t_a t_b t_c : ℝ) : Prop
axiom diameter_of_circumcircle (D : ℝ) : Prop

-- The theorem to prove
theorem triangle_medians_inequality
  (h_sides : sides_of_triangle a b c)
  (h_medians : medians_of_triangle t_a t_b t_c)
  (h_diameter : diameter_of_circumcircle D)
  : (a^2 + b^2) / t_c + (b^2 + c^2) / t_a + (c^2 + a^2) / t_b ≤ 6 * D :=
sorry -- proof omitted

end triangle_medians_inequality_l215_215575


namespace part1_solution_part2_solution_l215_215655

noncomputable def part1_expr := (1 / (Real.sqrt 5 + 2)) - (Real.sqrt 3 - 1)^0 - Real.sqrt (9 - 4 * Real.sqrt 5)
theorem part1_solution : part1_expr = 2 := by
  sorry

noncomputable def part2_expr := 2 * Real.sqrt 3 * 612 * (7/2)
theorem part2_solution : part2_expr = 5508 * Real.sqrt 3 := by
  sorry

end part1_solution_part2_solution_l215_215655


namespace min_value_function_l215_215839

theorem min_value_function (x : ℝ) (h : x > 0) : 
  ∃ y, y = (x^2 + x + 25) / x ∧ y ≥ 11 :=
sorry

end min_value_function_l215_215839


namespace number_of_students_in_range_l215_215701

noncomputable def normal_distribution := sorry

theorem number_of_students_in_range 
  (μ : ℝ) (σ : ℝ) (n : ℕ)
  (P_mu_minus_sigma_to_mu_plus_sigma: ℝ)
  (P_mu_minus_3sigma_to_mu_plus_3sigma: ℝ)
  (h1 : μ = 100)
  (h2 : σ = 10)
  (h3 : n = 1000)
  (h4 : P_mu_minus_sigma_to_mu_plus_sigma ≈ 0.6827) 
  (h5 : P_mu_minus_3sigma_to_mu_plus_3sigma ≈ 0.9973) 
: ∃ x : ℕ, x = 840 := 
sorry

end number_of_students_in_range_l215_215701


namespace product_of_odd_primes_mod_32_l215_215044

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215044


namespace arcsin_arccos_eq_l215_215355

theorem arcsin_arccos_eq (x : ℝ) (h : Real.arcsin x + Real.arcsin (2 * x - 1) = Real.arccos x) : x = 1 := by
  sorry

end arcsin_arccos_eq_l215_215355


namespace prob_P_plus_S_one_less_multiple_of_7_l215_215500

theorem prob_P_plus_S_one_less_multiple_of_7 :
  let a b : ℕ := λ x y, x ∈ Finset.range (60+1) ∧ y ∈ Finset.range (60+1) ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 60 ∧ 1 ≤ y ∧ y ≤ 60,
      P : ℕ := ∀ a b, a * b,
      S : ℕ := ∀ a b, a + b,
      m : ℕ := (P + S) + 1,
      all_pairs : ℕ := Nat.choose 60 2,
      valid_pairs : ℕ := 444,
      probability : ℚ := valid_pairs / all_pairs
  in probability = 148 / 590 := sorry

end prob_P_plus_S_one_less_multiple_of_7_l215_215500


namespace person_A_arrives_before_B_l215_215502

variable {a b S : ℝ}

theorem person_A_arrives_before_B (h : a ≠ b) (a_pos : 0 < a) (b_pos : 0 < b) (S_pos : 0 < S) :
  (2 * S / (a + b)) < ((a + b) * S / (2 * a * b)) :=
by
  sorry

end person_A_arrives_before_B_l215_215502


namespace inequality_problem_l215_215152

theorem inequality_problem
  (a b c d e f : ℝ)
  (h_cond : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
by
  sorry

end inequality_problem_l215_215152


namespace func_inequality_l215_215433

noncomputable def f (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given function properties
variables {a b c : ℝ} (h_a : a > 0) (symmetry : ∀ x : ℝ, f a b c (2 + x) = f a b c (2 - x))

theorem func_inequality : f a b c 2 < f a b c 1 ∧ f a b c 1 < f a b c 4 :=
by
  sorry

end func_inequality_l215_215433


namespace JimAgeInXYears_l215_215231

-- Definitions based on conditions
def TomCurrentAge := 37
def JimsAge7YearsAgo := 5 + (TomCurrentAge - 7) / 2

-- We introduce a variable X to represent the number of years into the future.
variable (X : ℕ)

-- Lean 4 statement to prove that Jim will be 27 + X years old in X years from now.
theorem JimAgeInXYears : JimsAge7YearsAgo + 7 + X = 27 + X := 
by
  sorry

end JimAgeInXYears_l215_215231


namespace inverse_function_of_13_l215_215214

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4
noncomputable def f_inv (y : ℝ) : ℝ := (y - 4) / 3

theorem inverse_function_of_13 : f_inv (f_inv 13) = -1 / 3 := by
  sorry

end inverse_function_of_13_l215_215214


namespace neg_alpha_quadrant_l215_215680

theorem neg_alpha_quadrant (α : ℝ) (k : ℤ) 
    (h1 : k * 360 + 180 < α)
    (h2 : α < k * 360 + 270) :
    k * 360 + 90 < -α ∧ -α < k * 360 + 180 :=
by
  sorry

end neg_alpha_quadrant_l215_215680


namespace solution_fractional_equation_l215_215224

noncomputable def solve_fractional_equation : Prop :=
  ∀ x : ℝ, (4/(x-2) = 2/x) ↔ x = -2

theorem solution_fractional_equation :
  solve_fractional_equation :=
by
  sorry

end solution_fractional_equation_l215_215224


namespace factorization_4x2_minus_144_l215_215665

theorem factorization_4x2_minus_144 (x : ℝ) : 4 * x^2 - 144 = 4 * (x - 6) * (x + 6) := 
  sorry

end factorization_4x2_minus_144_l215_215665


namespace problem1_solution_set_problem2_range_of_a_l215_215979

-- Define the functions
def f (x a : ℝ) : ℝ := |2 * x - 1| + |2 * x + a|
def g (x : ℝ) : ℝ := x + 3

-- Problem 1: Proving the solution set when a = -2
theorem problem1_solution_set (x : ℝ) : (f x (-2) < g x) ↔ (0 < x ∧ x < 2) :=
  sorry

-- Problem 2: Proving the range of a
theorem problem2_range_of_a (a : ℝ) : 
  (a > -1) ∧ (∀ x, (x ∈ Set.Icc (-a/2) (1/2) → f x a ≤ g x)) ↔ a ∈ Set.Ioo (-1) (4/3) ∨ a = 4/3 :=
  sorry

end problem1_solution_set_problem2_range_of_a_l215_215979


namespace real_part_zero_implies_a_eq_one_l215_215177

open Complex

theorem real_part_zero_implies_a_eq_one (a : ℝ) : 
  (1 + (1 : ℂ) * I) * (1 + a * I) = 0 ↔ a = 1 := by
  sorry

end real_part_zero_implies_a_eq_one_l215_215177


namespace intersection_count_l215_215437

def M (x y : ℝ) : Prop := y^2 = x - 1
def N (x y m : ℝ) : Prop := y = 2 * x - 2 * m^2 + m - 2

theorem intersection_count (m x y : ℝ) :
  (M x y ∧ N x y m) → (∃ n : ℕ, n = 1 ∨ n = 2) :=
sorry

end intersection_count_l215_215437


namespace cubic_polynomial_a_value_l215_215569

theorem cubic_polynomial_a_value (a b c d y₁ y₂ : ℝ)
  (h₁ : y₁ = a + b + c + d)
  (h₂ : y₂ = -a + b - c + d)
  (h₃ : y₁ - y₂ = -8) : a = -4 :=
by
  sorry

end cubic_polynomial_a_value_l215_215569


namespace first_pack_weight_l215_215260

variable (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
variable (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ)

theorem first_pack_weight (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
    (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ) :
    hiking_rate = 2.5 →
    hours_per_day = 9 →
    days = 7 →
    pounds_per_mile = 0.6 →
    first_resupply_percentage = 0.30 →
    second_resupply_percentage = 0.20 →
    ∃ first_pack : ℝ, first_pack = 47.25 :=
by
  intro h1 h2 h3 h4 h5 h6
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := pounds_per_mile * total_distance
  let first_resupply := total_supplies * first_resupply_percentage
  let second_resupply := total_supplies * second_resupply_percentage
  let first_pack := total_supplies - (first_resupply + second_resupply)
  use first_pack
  sorry

end first_pack_weight_l215_215260


namespace value_of_expression_l215_215686

theorem value_of_expression (x : ℤ) (h : x ^ 2 = 2209) : (x + 2) * (x - 2) = 2205 := 
by
  -- the proof goes here
  sorry

end value_of_expression_l215_215686


namespace contractor_engaged_days_l215_215256

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end contractor_engaged_days_l215_215256


namespace tan_beta_eq_l215_215873

theorem tan_beta_eq
  (a b : ℝ)
  (α β γ : ℝ)
  (h1 : (a + b) / (a - b) = (Real.tan ((α + β) / 2)) / (Real.tan ((α - β) / 2))) 
  (h2 : (α + β) / 2 = 90 - γ / 2) 
  (h3 : (α - β) / 2 = 90 - (β + γ / 2)) 
  : Real.tan β = (2 * b * Real.tan (γ / 2)) / ((a + b) * (Real.tan (γ / 2))^2 + (a - b)) :=
by
  sorry

end tan_beta_eq_l215_215873


namespace minimum_value_f_l215_215141

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_f (x : ℝ) (h : x > 1) : (∃ y, (f y = 3) ∧ ∀ z, z > 1 → f z ≥ 3) :=
by sorry

end minimum_value_f_l215_215141


namespace speed_of_boat_in_still_water_l215_215111

-- Define the given conditions
def speed_of_stream : ℝ := 4  -- Speed of the stream in km/hr
def distance_downstream : ℝ := 60  -- Distance traveled downstream in km
def time_downstream : ℝ := 3  -- Time taken to travel downstream in hours

-- The statement we need to prove
theorem speed_of_boat_in_still_water (V_b : ℝ) (V_d : ℝ) :
  V_d = distance_downstream / time_downstream →
  V_d = V_b + speed_of_stream →
  V_b = 16 :=
by
  intros Vd_eq D_eq
  sorry

end speed_of_boat_in_still_water_l215_215111


namespace arith_prog_sum_eq_l215_215473

variable (a d : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n / 2) * (2 * a 1 + (n - 1) * d 1)

theorem arith_prog_sum_eq (n : ℕ) : 
  S a d (n + 3) - 3 * S a d (n + 2) + 3 * S a d (n + 1) - S a d n = 0 := 
sorry

end arith_prog_sum_eq_l215_215473


namespace detail_understanding_word_meaning_guessing_logical_reasoning_l215_215384

-- Detail Understanding Question
theorem detail_understanding (sentence: String) (s: ∀ x : String, x ∈ ["He hardly watered his new trees,..."] → x = sentence) :
  sentence = "He hardly watered his new trees,..." :=
sorry

-- Word Meaning Guessing Question
theorem word_meaning_guessing (adversity_meaning: String) (meanings: ∀ y : String, y ∈ ["adversity means misfortune or disaster", "lack of water", "sufficient care/attention", "bad weather"] → y = adversity_meaning) :
  adversity_meaning = "adversity means misfortune or disaster" :=
sorry

-- Logical Reasoning Question
theorem logical_reasoning (hope: String) (sentences: ∀ z : String, z ∈ ["The author hopes his sons can withstand the tests of wind and rain in their life journey"] → z = hope) :
  hope = "The author hopes his sons can withstand the tests of wind and rain in their life journey" :=
sorry

end detail_understanding_word_meaning_guessing_logical_reasoning_l215_215384


namespace lloyd_total_hours_worked_l215_215858

-- Conditions
def regular_hours_per_day : ℝ := 7.5
def regular_rate : ℝ := 4.5
def overtime_multiplier : ℝ := 2.5
def total_earnings : ℝ := 67.5

-- Proof problem
theorem lloyd_total_hours_worked :
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_earnings := regular_hours_per_day * regular_rate
  let earnings_from_overtime := total_earnings - regular_earnings
  let hours_of_overtime := earnings_from_overtime / overtime_rate
  let total_hours := regular_hours_per_day + hours_of_overtime
  total_hours = 10.5 :=
by
  sorry

end lloyd_total_hours_worked_l215_215858


namespace sequence_1001st_term_l215_215520

theorem sequence_1001st_term (a b : ℤ) (h1 : b = 2 * a - 3) : 
  ∃ n : ℤ, n = 1001 → (a + 1000 * (20 * a - 30)) = 30003 := 
by 
  sorry

end sequence_1001st_term_l215_215520


namespace length_ab_square_l215_215901

theorem length_ab_square (s a : ℝ) (h_square : s = 2 * a) (h_area : 3000 = 1/2 * (s + (s - 2 * a)) * s) : 
  s = 20 * Real.sqrt 15 :=
by
  sorry

end length_ab_square_l215_215901


namespace blue_pill_cost_correct_l215_215059

-- Defining the conditions
def num_days : Nat := 21
def total_cost : Nat := 672
def red_pill_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost - 2
def daily_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost + red_pill_cost blue_pill_cost

-- The statement to prove
theorem blue_pill_cost_correct : ∃ (y : Nat), daily_cost y * num_days = total_cost ∧ y = 17 :=
by
  sorry

end blue_pill_cost_correct_l215_215059


namespace cubic_roots_number_l215_215462

noncomputable def determinant_cubic (a b c d : ℝ) (x : ℝ) : ℝ :=
  x * (x^2 + a^2) + c * (b * x + a * b) - b * (c * a - b * x)

theorem cubic_roots_number (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ roots : ℕ, (roots = 1 ∨ roots = 3) :=
  sorry

end cubic_roots_number_l215_215462


namespace product_increase_false_l215_215626

theorem product_increase_false (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬ (a * b = a * (10 * b) / 10 ∧ a * (10 * b) / 10 = 10 * (a * b)) :=
by 
  sorry

end product_increase_false_l215_215626


namespace eccentricity_sum_cannot_be_2sqrt2_l215_215216

noncomputable def e1 (a b : ℝ) := Real.sqrt (1 + (b^2) / (a^2))
noncomputable def e2 (a b : ℝ) := Real.sqrt (1 + (a^2) / (b^2))
noncomputable def e1_plus_e2 (a b : ℝ) := e1 a b + e2 a b

theorem eccentricity_sum_cannot_be_2sqrt2 (a b : ℝ) : e1_plus_e2 a b ≠ 2 * Real.sqrt 2 := by
  sorry

end eccentricity_sum_cannot_be_2sqrt2_l215_215216


namespace arccos_neg_one_eq_pi_l215_215942

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l215_215942


namespace sum_of_solutions_l215_215381

theorem sum_of_solutions (a b : ℤ) (h₁ : a = -1) (h₂ : b = -4) (h₃ : ∀ x : ℝ, (16 - 4 * x - x^2 = 0 ↔ -x^2 - 4 * x + 16 = 0)) : 
  (-b / a) = 4 := 
by 
  rw [h₁, h₂]
  norm_num
  sorry

end sum_of_solutions_l215_215381


namespace largest_tile_size_l215_215104

theorem largest_tile_size
  (length width : ℕ)
  (H1 : length = 378)
  (H2 : width = 595) :
  Nat.gcd length width = 7 :=
by
  sorry

end largest_tile_size_l215_215104


namespace cost_of_single_figurine_l215_215643

theorem cost_of_single_figurine (cost_tv : ℕ) (num_tv : ℕ) (num_figurines : ℕ) (total_spent : ℕ) :
  (num_tv = 5) →
  (cost_tv = 50) →
  (num_figurines = 10) →
  (total_spent = 260) →
  ((total_spent - num_tv * cost_tv) / num_figurines = 1) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end cost_of_single_figurine_l215_215643


namespace charlie_widgets_difference_l215_215206

theorem charlie_widgets_difference (w t : ℕ) (hw : w = 3 * t) :
  w * t - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end charlie_widgets_difference_l215_215206


namespace thirty_six_hundredths_is_decimal_l215_215619

namespace thirty_six_hundredths

-- Define the fraction representation of thirty-six hundredths
def fraction_thirty_six_hundredths : ℚ := 36 / 100

-- The problem is to prove that this fraction is equal to 0.36 in decimal form
theorem thirty_six_hundredths_is_decimal : fraction_thirty_six_hundredths = 0.36 := 
sorry

end thirty_six_hundredths

end thirty_six_hundredths_is_decimal_l215_215619


namespace distance_difference_l215_215786

-- Define coordinates as vectors
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def O1 : ℝ × ℝ × ℝ := (0.5, 0.5, 1)
def B_mid : ℝ × ℝ × ℝ := (0.5, 0, 0)

-- Define the Euclidean distance between two 3D points
def euclidean_distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2)

-- Statement of the problem
theorem distance_difference :
  euclidean_distance O1 A - euclidean_distance O1 B_mid = (real.sqrt 6 - real.sqrt 5) / 2 :=
by
  sorry

end distance_difference_l215_215786


namespace inequality_problem_l215_215155

theorem inequality_problem
  (a b c d e f : ℝ)
  (h_cond : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
by
  sorry

end inequality_problem_l215_215155


namespace faster_train_pass_time_l215_215893

-- Defining the conditions
def length_of_train : ℕ := 45 -- length in meters
def speed_of_faster_train : ℕ := 45 -- speed in km/hr
def speed_of_slower_train : ℕ := 36 -- speed in km/hr

-- Define relative speed
def relative_speed := (speed_of_faster_train - speed_of_slower_train) * 5 / 18 -- converting km/hr to m/s

-- Total distance to pass (sum of lengths of both trains)
def total_passing_distance := (2 * length_of_train) -- 2 trains of 45 meters each

-- Calculate the time to pass the slower train
def time_to_pass := total_passing_distance / relative_speed

-- The theorem to prove
theorem faster_train_pass_time : time_to_pass = 36 := by
  -- This is where the proof would be placed
  sorry

end faster_train_pass_time_l215_215893


namespace point_in_third_quadrant_l215_215841

theorem point_in_third_quadrant (x y : ℝ) (h1 : x + y < 0) (h2 : x * y > 0) : x < 0 ∧ y < 0 := 
sorry

end point_in_third_quadrant_l215_215841


namespace product_mod_32_l215_215004

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215004


namespace max_incircle_circumcircle_ratio_l215_215622

theorem max_incircle_circumcircle_ratio (c : ℝ) (α : ℝ) 
  (hα : 0 < α ∧ α < π / 2) :
  let a := c * Real.cos α
  let b := c * Real.sin α
  let R := c / 2
  let r := (a + b - c) / 2
  (r / R <= Real.sqrt 2 - 1) :=
by
  sorry

end max_incircle_circumcircle_ratio_l215_215622


namespace isosceles_triangle_perimeter_l215_215178

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2))
  (h2 : ∃ x y z : ℕ, (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
  a + a + b = 12 :=
by
  sorry

end isosceles_triangle_perimeter_l215_215178


namespace coexistence_of_properties_l215_215133

structure Trapezoid (α : Type _) [Field α] :=
(base1 base2 leg1 leg2 : α)
(height : α)

def isIsosceles {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
T.leg1 = T.leg2

def diagonalsPerpendicular {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
sorry  -- Define this property based on coordinate geometry or vector inner products

def heightsEqual {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
T.base1 = T.base2

def midsegmentEqualHeight {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
(T.base1 + T.base2) / 2 = T.height

theorem coexistence_of_properties (α : Type _) [Field α] (T : Trapezoid α) :
  isIsosceles T → heightsEqual T → midsegmentEqualHeight T → True :=
by sorry

end coexistence_of_properties_l215_215133


namespace angle_relationship_l215_215883

variables {VU VW : ℝ} {x y z : ℝ} (h1 : VU = VW) 
          (angle_UXZ : ℝ) (angle_VYZ : ℝ) (angle_VZX : ℝ)
          (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z)

theorem angle_relationship (h1 : VU = VW) (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z) : 
    x = (y - z) / 2 := 
by 
    sorry

end angle_relationship_l215_215883


namespace infinite_graph_ten_colorable_l215_215584

noncomputable def ten_colorable (G : SimpleGraph (V : Type)) := 
  ∃ (c : V → Fin 10), G.IsColoring c

theorem infinite_graph_ten_colorable (G : SimpleGraph ℕ) (h : ∀ (V' : Finset ℕ), 
  ten_colorable (G.inducedSubgraph V')) : 
  ten_colorable G :=
by
  sorry

end infinite_graph_ten_colorable_l215_215584


namespace area_of_square_with_given_diagonal_l215_215989

-- Definition of the conditions
def diagonal := 12
def s := Real
def area (s : Real) := s^2
def diag_relation (d s : Real) := d^2 = 2 * s^2

-- The proof statement
theorem area_of_square_with_given_diagonal :
  ∃ s : Real, diag_relation diagonal s ∧ area s = 72 :=
by
  sorry

end area_of_square_with_given_diagonal_l215_215989


namespace handbag_monday_price_l215_215909

theorem handbag_monday_price (initial_price : ℝ) (primary_discount : ℝ) (additional_discount : ℝ)
(h_initial_price : initial_price = 250)
(h_primary_discount : primary_discount = 0.4)
(h_additional_discount : additional_discount = 0.1) :
(initial_price - initial_price * primary_discount) - ((initial_price - initial_price * primary_discount) * additional_discount) = 135 := by
  sorry

end handbag_monday_price_l215_215909


namespace arithmetic_sequence_first_term_l215_215332

theorem arithmetic_sequence_first_term (d : ℤ) (a_n a_2 a_9 a_11 : ℤ) 
  (h1 : a_2 = 7) 
  (h2 : a_11 = a_9 + 6)
  (h3 : a_11 = a_n + 10 * d)
  (h4 : a_9 = a_n + 8 * d)
  (h5 : a_2 = a_n + d) :
  a_n = 4 := by
  sorry

end arithmetic_sequence_first_term_l215_215332


namespace exists_x_for_log_eqn_l215_215831

theorem exists_x_for_log_eqn (a : ℝ) (ha : 0 < a) :
  ∃ (x : ℝ), (1 < x) ∧ (Real.log (a * x) / Real.log 10 = 2 * Real.log (x - 1) / Real.log 10) ∧ 
  x = (2 + a + Real.sqrt (a^2 + 4*a)) / 2 := sorry

end exists_x_for_log_eqn_l215_215831


namespace find_functional_l215_215131

noncomputable def functional_equation_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (f x + y) = 2 * x + f (f y - x)

theorem find_functional (f : ℝ → ℝ) :
  functional_equation_solution f → ∃ c : ℝ, ∀ x, f x = x + c := 
by
  sorry

end find_functional_l215_215131


namespace find_m_l215_215836

theorem find_m (a0 a1 a2 a3 a4 a5 a6 m : ℝ) 
  (h1 : (1 + m) ^ 6 = a0 + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  m = 1 ∨ m = -3 := 
  sorry

end find_m_l215_215836


namespace largest_digit_B_divisible_by_4_l215_215189

theorem largest_digit_B_divisible_by_4 :
  ∃ (B : ℕ), B ≤ 9 ∧ ∀ B', (B' ≤ 9 ∧ (4 * 10^5 + B' * 10^4 + 5 * 10^3 + 7 * 10^2 + 8 * 10 + 4) % 4 = 0) → B' ≤ B :=
by
  sorry

end largest_digit_B_divisible_by_4_l215_215189


namespace circle_line_intersection_points_l215_215950

theorem circle_line_intersection_points :
  let circle_eqn : ℝ × ℝ → Prop := fun p => (p.1 - 1)^2 + p.2^2 = 16
  let line_eqn  : ℝ × ℝ → Prop := fun p => p.1 = 4
  ∃ (p₁ p₂ : ℝ × ℝ), 
    circle_eqn p₁ ∧ line_eqn p₁ ∧ circle_eqn p₂ ∧ line_eqn p₂ ∧ p₁ ≠ p₂ 
      → ∀ (p : ℝ × ℝ), circle_eqn p ∧ line_eqn p → 
        p = p₁ ∨ p = p₂ ∧ (p₁ ≠ p ∨ p₂ ≠ p)
 := sorry

end circle_line_intersection_points_l215_215950


namespace logarithmic_inequality_l215_215677

theorem logarithmic_inequality : 
  (a = Real.log 9 / Real.log 2) →
  (b = Real.log 27 / Real.log 3) →
  (c = Real.log 15 / Real.log 5) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  sorry

end logarithmic_inequality_l215_215677


namespace ellipse_perimeter_l215_215972

noncomputable def perimeter_of_triangle (a b : ℝ) (e : ℝ) : ℝ :=
  if (b = 4 ∧ e = 3 / 5 ∧ a = b / (1 - e^2) ^ (1 / 2))
  then 4 * a
  else 0

theorem ellipse_perimeter :
  let a : ℝ := 5
  let b : ℝ := 4
  let e : ℝ := 3 / 5
  4 * a = 20 :=
by
  sorry

end ellipse_perimeter_l215_215972


namespace square_division_possible_l215_215581

theorem square_division_possible :
  ∃ (S a b c : ℕ), 
    S^2 = a^2 + 3 * b^2 + 5 * c^2 ∧ 
    a = 3 ∧ 
    b = 2 ∧ 
    c = 1 :=
  by {
    sorry
  }

end square_division_possible_l215_215581


namespace omega_in_abc_l215_215478

variables {R : Type*}
variables [LinearOrderedField R]
variables {a b c ω x y z : R} 

theorem omega_in_abc 
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ω ≠ a ∧ ω ≠ b ∧ ω ≠ c)
  (h1 : x + y + z = 1)
  (h2 : a^2 * x + b^2 * y + c^2 * z = ω^2)
  (h3 : a^3 * x + b^3 * y + c^3 * z = ω^3)
  (h4 : a^4 * x + b^4 * y + c^4 * z = ω^4):
  ω = a ∨ ω = b ∨ ω = c :=
sorry

end omega_in_abc_l215_215478


namespace geoff_initial_percent_l215_215574

theorem geoff_initial_percent (votes_cast : ℕ) (win_percent : ℝ) (needed_more_votes : ℕ) (initial_votes : ℕ)
  (h1 : votes_cast = 6000)
  (h2 : win_percent = 50.5)
  (h3 : needed_more_votes = 3000)
  (h4 : initial_votes = 31) :
  (initial_votes : ℝ) / votes_cast * 100 = 0.52 :=
by
  sorry

end geoff_initial_percent_l215_215574


namespace set_intersection_A_B_l215_215981

theorem set_intersection_A_B :
  (A : Set ℤ) ∩ (B : Set ℤ) = { -1, 0, 1, 2 } :=
by
  let A := { x : ℤ | x^2 - x - 2 ≤ 0 }
  let B := {x : ℤ | x ∈ Set.univ}
  sorry

end set_intersection_A_B_l215_215981


namespace larger_number_is_37_l215_215092

-- Defining the conditions
def sum_of_two_numbers (a b : ℕ) : Prop := a + b = 62
def one_is_12_more (a b : ℕ) : Prop := a = b + 12

-- Proof statement
theorem larger_number_is_37 (a b : ℕ) (h₁ : sum_of_two_numbers a b) (h₂ : one_is_12_more a b) : a = 37 :=
by
  sorry

end larger_number_is_37_l215_215092


namespace graph_passes_through_fixed_point_l215_215484

theorem graph_passes_through_fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
    ∃ (x y : ℝ), (x = -3) ∧ (y = -1) ∧ (y = a^(x + 3) - 2) :=
by
  sorry

end graph_passes_through_fixed_point_l215_215484


namespace probability_ratio_l215_215069

theorem probability_ratio (a b : ℕ) (h1 : a = 60) (h2 : b = 12)
  (h3 : ∀ n, n ∈ (Finset.range b).image ((λ i, i + 1)) → (∃ c, c = 5))
  (h4 : ∀ k, k = 5) :
  let p := 12 / (Nat.choose 60 5 : ℚ),
      r := 13200 / (Nat.choose 60 5 : ℚ) in
  r / p = 1100 := by
  sorry

end probability_ratio_l215_215069


namespace inequality_solution_l215_215668

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) : (x - (1/x) > 0) ↔ (-1 < x ∧ x < 0) ∨ (1 < x) := 
by
  sorry

end inequality_solution_l215_215668


namespace part1_part2_l215_215543

namespace Proof

def A (a b : ℝ) : ℝ := 3 * a ^ 2 - 4 * a * b
def B (a b : ℝ) : ℝ := a ^ 2 + 2 * a * b

theorem part1 (a b : ℝ) : 2 * A a b - 3 * B a b = 3 * a ^ 2 - 14 * a * b := by
  sorry
  
theorem part2 (a b : ℝ) (h : |3 * a + 1| + (2 - 3 * b) ^ 2 = 0) : A a b - 2 * B a b = 5 / 3 := by
  have ha : a = -1 / 3 := by
    sorry
  have hb : b = 2 / 3 := by
    sorry
  rw [ha, hb]
  sorry

end Proof

end part1_part2_l215_215543


namespace percentage_of_60_eq_15_l215_215487

-- Conditions provided in the problem
def percentage (p : ℚ) : ℚ := p / 100
def num : ℚ := 60
def fraction_of_num (p : ℚ) (n : ℚ) : ℚ := (percentage p) * n

-- Assertion to be proved
theorem percentage_of_60_eq_15 : fraction_of_num 25 num = 15 := 
by 
  show fraction_of_num 25 60 = 15
  sorry

end percentage_of_60_eq_15_l215_215487


namespace range_of_m_l215_215985

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ x y : ℝ, 0 < x → 0 < y → (2 * y / x + 9 * x / (2 * y) ≥ m^2 + m))
  ↔ (-3 ≤ m ∧ m ≤ 2) := sorry

end range_of_m_l215_215985


namespace bookmark_position_second_book_l215_215233

-- Definitions for the conditions
def pages_per_book := 250
def cover_thickness_ratio := 10
def total_books := 2
def distance_bookmarks_factor := 1 / 3

-- Derived constants
def cover_thickness := cover_thickness_ratio * pages_per_book
def total_pages := (pages_per_book * total_books) + (cover_thickness * total_books * 2)
def distance_between_bookmarks := total_pages * distance_bookmarks_factor
def midpoint_pages_within_book := (pages_per_book / 2) + cover_thickness

-- Definitions for bookmarks positions
def first_bookmark_position := midpoint_pages_within_book
def remaining_pages_after_first_bookmark := distance_between_bookmarks - midpoint_pages_within_book
def second_bookmark_position := remaining_pages_after_first_bookmark - cover_thickness

-- Theorem stating the goal
theorem bookmark_position_second_book :
  35 ≤ second_bookmark_position ∧ second_bookmark_position < 36 :=
sorry

end bookmark_position_second_book_l215_215233


namespace alice_paid_24_percent_l215_215493

theorem alice_paid_24_percent (P : ℝ) (h1 : P > 0) :
  let MP := 0.60 * P
  let price_paid := 0.40 * MP
  (price_paid / P) * 100 = 24 :=
by
  sorry

end alice_paid_24_percent_l215_215493


namespace range_of_a_l215_215980

theorem range_of_a (a : ℝ) :
  (∀ (x y z: ℝ), x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) ↔ (a ≤ -2 ∨ a ≥ 4) :=
by
sorry

end range_of_a_l215_215980


namespace abes_age_l215_215613

theorem abes_age (A : ℕ) (h : A + (A - 7) = 29) : A = 18 :=
by
  sorry

end abes_age_l215_215613


namespace product_of_fractions_l215_215529

theorem product_of_fractions : (2 : ℚ) / 9 * (4 : ℚ) / 5 = 8 / 45 :=
by 
  sorry

end product_of_fractions_l215_215529


namespace portion_of_work_done_l215_215213

variable (P W : ℕ)

-- Given conditions
def work_rate_P (P W : ℕ) : ℕ := W / 16
def work_rate_2P (P W : ℕ) : ℕ := 2 * (work_rate_P P W)

-- Lean theorem
theorem portion_of_work_done (h : work_rate_2P P W * 4 = W / 2) : 
    work_rate_2P P W * 4 = W / 2 := 
by 
  sorry

end portion_of_work_done_l215_215213


namespace probability_queen_of_diamonds_l215_215454

/-- 
A standard deck of 52 cards consists of 13 ranks and 4 suits.
We want to prove that the probability the top card is the Queen of Diamonds is 1/52.
-/
theorem probability_queen_of_diamonds 
  (total_cards : ℕ) 
  (queen_of_diamonds : ℕ)
  (h1 : total_cards = 52)
  (h2 : queen_of_diamonds = 1) : 
  (queen_of_diamonds : ℚ) / (total_cards : ℚ) = 1 / 52 := 
by 
  sorry

end probability_queen_of_diamonds_l215_215454


namespace polygon_sides_l215_215083

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end polygon_sides_l215_215083


namespace geometric_sequence_sum_l215_215997

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (ha1 : q ≠ 0)
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 3 + a 4 = (a 1 + a 2) * q^2)
  : a 5 + a 6 = 48 :=
by
  sorry

end geometric_sequence_sum_l215_215997


namespace f_is_even_l215_215078

noncomputable def f (x : ℝ) : ℝ := x ^ 2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := 
by
  intros x
  sorry

end f_is_even_l215_215078


namespace original_cost_of_dress_l215_215932

theorem original_cost_of_dress (x: ℝ) 
  (h1: x / 2 - 10 < x) 
  (h2: x - (x / 2 - 10) = 80) : 
  x = 140 :=
sorry

end original_cost_of_dress_l215_215932


namespace solve_fractional_eq_l215_215225

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) : (4 / (x - 2) = 2 / x) → (x = -2) :=
by 
  sorry

end solve_fractional_eq_l215_215225


namespace scientists_arrival_probability_l215_215892

open Real

theorem scientists_arrival_probability (x y z : ℕ) (n : ℝ) (h : z ≠ 0)
  (hz : ¬ ∃ p : ℕ, Nat.Prime p ∧ p ^ 2 ∣ z)
  (h1 : n = x - y * sqrt z)
  (h2 : ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 120 ∧ 0 ≤ b ∧ b ≤ 120 ∧
    |a - b| ≤ n)
  (h3 : (120 - n)^2 / (120 ^ 2) = 0.7) :
  x + y + z = 202 := sorry

end scientists_arrival_probability_l215_215892


namespace remainder_M_mod_32_l215_215032

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215032


namespace rons_baseball_team_l215_215731

/-- Ron's baseball team scored 270 points in the year. 
    5 players averaged 50 points each, 
    and the remaining players averaged 5 points each.
    Prove that the number of players on the team is 9. -/
theorem rons_baseball_team : (∃ n m : ℕ, 5 * 50 + m * 5 = 270 ∧ n = 5 + m ∧ 5 = 50 ∧ m = 4) :=
sorry

end rons_baseball_team_l215_215731


namespace find_b_l215_215973

noncomputable def circle_center_radius : Prop :=
  let C := (2, 0) -- center
  let r := 2 -- radius
  C.1 = 2 ∧ C.2 = 0 ∧ r = 2

noncomputable def line (b : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, M ≠ N ∧ 
  (M.2 = M.1 + b) ∧ (N.2 = N.1 + b) -- points on the line are M = (x1, x1 + b) and N = (x2, x2 + b)

noncomputable def perpendicular_condition (M N center: ℝ × ℝ) : Prop :=
  (M.1 - center.1) * (N.1 - center.1) + (M.2 - center.2) * (N.2 - center.2) = 0 -- CM ⟂ CN

theorem find_b (b : ℝ) : 
  circle_center_radius ∧
  (∃ M N, line b ∧ perpendicular_condition M N (2, 0)) →
  b = 0 ∨ b = -4 :=
by {
  -- Proof omitted
  sorry
}

end find_b_l215_215973


namespace boys_variance_greater_than_girls_l215_215183

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := (scores.sum / n)
  let squared_diff := scores.map (λ x => (x - mean) ^ 2)
  (squared_diff.sum) / n

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores :=
by
  sorry

end boys_variance_greater_than_girls_l215_215183


namespace factor_is_three_l215_215262

theorem factor_is_three (x f : ℝ) (h1 : 2 * x + 5 = y) (h2 : f * y = 111) (h3 : x = 16):
  f = 3 :=
by
  sorry

end factor_is_three_l215_215262


namespace false_statement_about_circles_l215_215558

variable (P Q : Type) [MetricSpace P] [MetricSpace Q]
variable (p q : ℝ)
variable (dist_PQ : ℝ)

theorem false_statement_about_circles 
  (hA : p - q = dist_PQ → false)
  (hB : p + q = dist_PQ → false)
  (hC : p + q < dist_PQ → false)
  (hD : p - q < dist_PQ → false) : 
  false :=
by sorry

end false_statement_about_circles_l215_215558


namespace arccos_neg_one_eq_pi_l215_215945

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l215_215945


namespace product_mod_32_is_15_l215_215020

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215020


namespace least_ab_value_l215_215675

theorem least_ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h : (1 : ℚ)/a + (1 : ℚ)/(3 * b) = 1 / 6) : a * b = 98 :=
by
  sorry

end least_ab_value_l215_215675


namespace find_value_of_a_l215_215673

theorem find_value_of_a (a : ℤ) (h : ∀ x : ℚ,  x^6 - 33 * x + 20 = (x^2 - x + a) * (x^4 + b * x^3 + c * x^2 + d * x + e)) :
  a = 4 := 
by 
  sorry

end find_value_of_a_l215_215673


namespace find_a_extremum_and_min_value_find_max_k_l215_215978

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x + 1

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a

theorem find_a_extremum_and_min_value :
  (∀ a : ℝ, f' a 0 = 0 → a = -1) ∧
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → f (-1) x ≥ 2) :=
by sorry

theorem find_max_k (k : ℤ) :
  (∀ x : ℝ, 0 < x → k * (Real.exp x - 1) < x * Real.exp x + 1) →
  k ≤ 2 :=
by sorry

end find_a_extremum_and_min_value_find_max_k_l215_215978


namespace product_of_odd_primes_mod_32_l215_215050

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215050


namespace total_arrangements_l215_215397

def total_members : ℕ := 6
def days : ℕ := 3
def people_per_day : ℕ := 2

def A_cannot_on_14 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 14 = 1

def B_cannot_on_16 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 16 = 2

theorem total_arrangements (arrangement : ℕ → ℕ) :
  (∀ arrangement, A_cannot_on_14 arrangement ∧ B_cannot_on_16 arrangement) →
  (total_members.choose 2 * (total_members - 2).choose 2 - 
  2 * (total_members - 1).choose 1 * (total_members - 2).choose 2 +
  (total_members - 2).choose 1 * (total_members - 3).choose 1)
  = 42 := 
by
  sorry

end total_arrangements_l215_215397


namespace PQ_PR_QR_div_l215_215463

theorem PQ_PR_QR_div (p q r : ℝ)
    (midQR : p = 0) (midPR : q = 0) (midPQ : r = 0) :
    (4 * (q ^ 2 + r ^ 2) + 4 * (p ^ 2 + r ^ 2) + 4 * (p ^ 2 + q ^ 2)) / (p ^ 2 + q ^ 2 + r ^ 2) = 8 :=
by {
    sorry
}

end PQ_PR_QR_div_l215_215463


namespace gcd_960_1632_l215_215803

theorem gcd_960_1632 : Int.gcd 960 1632 = 96 := by
  sorry

end gcd_960_1632_l215_215803


namespace innokentiy_games_l215_215663

def games_played_egor := 13
def games_played_nikita := 27
def games_played_innokentiy (N : ℕ) := N - games_played_egor

theorem innokentiy_games (N : ℕ) (h : N = games_played_nikita) : games_played_innokentiy N = 14 :=
by {
  sorry
}

end innokentiy_games_l215_215663


namespace currant_weight_l215_215778

noncomputable def volume_bucket : ℝ := 0.01 -- Volume of bucket in cubic meters
noncomputable def density_water : ℝ := 1000 -- Density of water in kg/m^3
noncomputable def packing_density : ℝ := 0.74 -- Packing density for currants

-- Effective volume occupied by currants
noncomputable def effective_volume : ℝ := volume_bucket * packing_density

-- Weight of the currants
def weight_of_currants : ℝ := density_water * effective_volume

theorem currant_weight :
  weight_of_currants = 7.4 :=
by
  sorry

end currant_weight_l215_215778


namespace monks_mantou_l215_215711

theorem monks_mantou (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + y / 3 = 100) :
  (3 * x + (100 - x) / 3 = 100) ∧ (x + y = 100 ∧ 3 * x + y / 3 = 100) :=
by
  sorry

end monks_mantou_l215_215711


namespace ms_warren_total_distance_l215_215598

-- Conditions as definitions
def running_speed : ℝ := 6 -- mph
def running_time : ℝ := 20 / 60 -- hours

def walking_speed : ℝ := 2 -- mph
def walking_time : ℝ := 30 / 60 -- hours

-- Total distance calculation
def distance_ran : ℝ := running_speed * running_time
def distance_walked : ℝ := walking_speed * walking_time
def total_distance : ℝ := distance_ran + distance_walked

-- Statement to be proved
theorem ms_warren_total_distance : total_distance = 3 := by
  sorry

end ms_warren_total_distance_l215_215598


namespace product_mod_32_is_15_l215_215016

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215016


namespace total_rocks_l215_215181

-- Definitions of variables based on the conditions
variables (igneous shiny_igneous : ℕ) (sedimentary : ℕ) (metamorphic : ℕ) (comet shiny_comet : ℕ)
variables (h1 : 1 / 4 * igneous = 15) (h2 : 1 / 2 * comet = 20)
variables (h3 : comet = 2 * metamorphic) (h4 : igneous = 3 * metamorphic)
variables (h5 : sedimentary = 2 * igneous)

-- The statement to be proved: the total number of rocks is 240
theorem total_rocks (igneous sedimentary metamorphic comet : ℕ) 
  (h1 : igneous = 4 * 15) 
  (h2 : comet = 2 * 20)
  (h3 : comet = 2 * metamorphic) 
  (h4 : igneous = 3 * metamorphic) 
  (h5 : sedimentary = 2 * igneous) : 
  igneous + sedimentary + metamorphic + comet = 240 :=
sorry

end total_rocks_l215_215181


namespace first_digit_base8_of_473_l215_215757

theorem first_digit_base8_of_473 : 
  ∃ (d : ℕ), (d < 8) ∧ (473 = d * 64 + r ∧ r < 64) ∧ 473 = 7 * 64 + 25 :=
sorry

end first_digit_base8_of_473_l215_215757


namespace catalyst_second_addition_is_882_l215_215912

-- Constants for the problem
def lower_bound : ℝ := 500
def upper_bound : ℝ := 1500
def golden_ratio_method : ℝ := 0.618

-- Calculated values
def first_addition : ℝ := lower_bound + golden_ratio_method * (upper_bound - lower_bound)
def second_bound : ℝ := first_addition - lower_bound
def second_addition : ℝ := lower_bound + golden_ratio_method * second_bound

theorem catalyst_second_addition_is_882 :
  lower_bound = 500 → upper_bound = 1500 → golden_ratio_method = 0.618 → second_addition = 882 := by
  -- Proof goes here
  sorry

end catalyst_second_addition_is_882_l215_215912


namespace sum_of_coefficients_of_y_terms_l215_215385

theorem sum_of_coefficients_of_y_terms :
  let p := (5 * x + 3 * y + 2) * (2 * x + 6 * y + 7)
  let expanded_p := 10 * x^2 + 36 * x * y + 39 * x + 18 * y^2 + 33 * y + 14
  (36 + 18 + 33) = 87 := by
  sorry

end sum_of_coefficients_of_y_terms_l215_215385


namespace find_first_term_of_geometric_progression_l215_215749

theorem find_first_term_of_geometric_progression
  (a_2 : ℝ) (a_3 : ℝ) (a_1 : ℝ) (q : ℝ)
  (h1 : a_2 = a_1 * q)
  (h2 : a_3 = a_1 * q^2)
  (h3 : a_2 = 5)
  (h4 : a_3 = 1) : a_1 = 25 :=
by
  sorry

end find_first_term_of_geometric_progression_l215_215749


namespace log9_6_eq_mn_over_2_l215_215986

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log9_6_eq_mn_over_2
  (m n : ℝ)
  (h1 : log_base 7 4 = m)
  (h2 : log_base 4 6 = n) : 
  log_base 9 6 = (m * n) / 2 := by
  sorry

end log9_6_eq_mn_over_2_l215_215986


namespace range_of_a_l215_215974

noncomputable def A := {x : ℝ | x^2 - 2*x - 8 < 0}
noncomputable def B := {x : ℝ | x^2 + 2*x - 3 > 0}
noncomputable def C (a : ℝ) := {x : ℝ | x^2 - 3*a*x + 2*a^2 < 0}

theorem range_of_a (a : ℝ) :
  (C a ⊆ A ∩ B) ↔ (1 ≤ a ∧ a ≤ 2 ∨ a = 0) :=
sorry

end range_of_a_l215_215974


namespace divide_cookie_into_16_equal_parts_l215_215284

def Cookie (n : ℕ) : Type := sorry

theorem divide_cookie_into_16_equal_parts (cookie : Cookie 64) :
  ∃ (slices : List (Cookie 4)), slices.length = 16 ∧ 
  (∀ (slice : Cookie 4), slice ≠ cookie) := 
sorry

end divide_cookie_into_16_equal_parts_l215_215284


namespace _l215_215460

noncomputable theorem compute_b1c1_plus_b2c2_plus_b3c3 :
  ∀ (b1 b2 b3 c1 c2 c3 : ℝ),
    (∀ x : ℝ, x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
      (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3)) →
    b1 * c1 + b2 * c2 + b3 * c3 = 1 :=
begin
  sorry
end

end _l215_215460


namespace geometric_sequence_a5_value_l215_215998

theorem geometric_sequence_a5_value :
  ∃ (a : ℕ → ℝ) (r : ℝ), (a 3)^2 - 4 * a 3 + 3 = 0 ∧ 
                         (a 7)^2 - 4 * a 7 + 3 = 0 ∧ 
                         (a 3) * (a 7) = 3 ∧ 
                         (a 3) + (a 7) = 4 ∧ 
                         a 5 = (a 3 * a 7).sqrt :=
sorry

end geometric_sequence_a5_value_l215_215998


namespace least_score_to_play_final_l215_215572

-- Definitions based on given conditions
def num_teams := 2021

def match_points (outcome : String) : ℕ :=
  match outcome with
  | "win"  => 3
  | "draw" => 1
  | "loss" => 0
  | _      => 0

def brazil_won_first_match : Prop := True

def ties_advantage (bfc_score other_team_score : ℕ) : Prop :=
  bfc_score = other_team_score

-- Theorem statement
theorem least_score_to_play_final (bfc_has_tiebreaker : (bfc_score other_team_score : ℕ) → ties_advantage bfc_score other_team_score)
  (bfc_first_match_won : brazil_won_first_match) :
  ∃ (least_score : ℕ), least_score = 2020 := sorry

end least_score_to_play_final_l215_215572


namespace solve_system_of_equations_l215_215211

theorem solve_system_of_equations (x y : ℝ) (hx : x + y + Real.sqrt (x * y) = 28)
  (hy : x^2 + y^2 + x * y = 336) : (x = 4 ∧ y = 16) ∨ (x = 16 ∧ y = 4) :=
sorry

end solve_system_of_equations_l215_215211


namespace emerson_row_distance_l215_215664

theorem emerson_row_distance (d1 d2 total : ℕ) (h1 : d1 = 6) (h2 : d2 = 18) (h3 : total = 39) :
  15 = total - (d1 + d2) :=
by sorry

end emerson_row_distance_l215_215664


namespace mary_sheep_problem_l215_215346

theorem mary_sheep_problem :
  let initial_sheep := 400
  in let sheep_given_to_sister := initial_sheep / 4
  in let remaining_after_sister := initial_sheep - sheep_given_to_sister
  in let sheep_given_to_brother := remaining_after_sister / 2
  in let sheep_remaining := remaining_after_sister - sheep_given_to_brother
  in sheep_remaining = 150 :=
by
  let initial_sheep := 400
  let sheep_given_to_sister := initial_sheep / 4
  let remaining_after_sister := initial_sheep - sheep_given_to_sister
  let sheep_given_to_brother := remaining_after_sister / 2
  let sheep_remaining := remaining_after_sister - sheep_given_to_brother
  show sheep_remaining = 150 from sorry

end mary_sheep_problem_l215_215346


namespace find_a_l215_215829

theorem find_a 
  {a : ℝ} 
  (h : ∀ x : ℝ, (ax / (x - 1) < 1) ↔ (x < 1 ∨ x > 3)) : 
  a = 2 / 3 := 
sorry

end find_a_l215_215829


namespace product_simplification_l215_215933

theorem product_simplification :
  (10 * (1 / 5) * (1 / 2) * 4 / 2 : ℝ) = 2 :=
by
  sorry

end product_simplification_l215_215933


namespace fraction_transformation_correct_l215_215242

theorem fraction_transformation_correct
  {a b : ℝ} (hb : b ≠ 0) : 
  (2 * a) / (2 * b) = a / b := by
  sorry

end fraction_transformation_correct_l215_215242


namespace point_three_units_away_from_A_is_negative_seven_or_negative_one_l215_215471

-- Defining the point A on the number line
def A : ℤ := -4

-- Definition of the condition where a point is 3 units away from A
def three_units_away (x : ℤ) : Prop := (x = A - 3) ∨ (x = A + 3)

-- The statement to be proved
theorem point_three_units_away_from_A_is_negative_seven_or_negative_one (x : ℤ) :
  three_units_away x → (x = -7 ∨ x = -1) :=
sorry

end point_three_units_away_from_A_is_negative_seven_or_negative_one_l215_215471


namespace product_mod_32_l215_215002

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215002


namespace max_crosses_in_grid_l215_215066

theorem max_crosses_in_grid : ∀ (n : ℕ), n = 16 → (∃ X : ℕ, X = 30 ∧
  ∀ (i j : ℕ), i < n → j < n → 
    (∀ k, k < n → (i ≠ k → X ≠ k)) ∧ 
    (∀ l, l < n → (j ≠ l → X ≠ l))) :=
by
  sorry

end max_crosses_in_grid_l215_215066


namespace tickets_won_in_skee_ball_l215_215855

-- Define the conditions as Lean definitions
def tickets_from_whack_a_mole : ℕ := 8
def ticket_cost_per_candy : ℕ := 5
def candies_bought : ℕ := 3

-- We now state the conjecture (mathematical proof problem) 
-- Prove that the number of tickets won in skee ball is 7.
theorem tickets_won_in_skee_ball :
  (candies_bought * ticket_cost_per_candy) - tickets_from_whack_a_mole = 7 :=
by
  sorry

end tickets_won_in_skee_ball_l215_215855


namespace investment_initial_amount_l215_215289

noncomputable def initialInvestment (final_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  final_amount / interest_rate^years

theorem investment_initial_amount :
  initialInvestment 705.73 1.12 5 = 400.52 := by
  sorry

end investment_initial_amount_l215_215289


namespace least_ab_value_l215_215160

theorem least_ab_value (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : (1 : ℚ) / a + (1 : ℚ) / (3 * b) = 1 / 9) :
  ab = 144 :=
sorry

end least_ab_value_l215_215160


namespace calculate_expression_l215_215100

theorem calculate_expression :
  15^2 + 2 * 15 * 5 + 5^2 + 5^3 = 525 := 
sorry

end calculate_expression_l215_215100


namespace sum_of_ages_l215_215476

theorem sum_of_ages (rose_age mother_age : ℕ) (rose_age_eq : rose_age = 25) (mother_age_eq : mother_age = 75) : 
  rose_age + mother_age = 100 := 
by
  sorry

end sum_of_ages_l215_215476


namespace largest_unsatisfiable_group_l215_215887

theorem largest_unsatisfiable_group :
  ∃ n : ℕ, (∀ a b c : ℕ, n ≠ 6 * a + 9 * b + 20 * c) ∧ (∀ m : ℕ, m > n → ∃ a b c : ℕ, m = 6 * a + 9 * b + 20 * c) ∧ n = 43 :=
by
  sorry

end largest_unsatisfiable_group_l215_215887


namespace polygon_sides_sum_720_l215_215086

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_sum_720_l215_215086


namespace typing_lines_in_10_minutes_l215_215888

def programmers := 10
def total_lines_in_60_minutes := 60
def total_minutes := 60
def target_minutes := 10

theorem typing_lines_in_10_minutes :
  (total_lines_in_60_minutes / total_minutes) * programmers * target_minutes = 100 :=
by sorry

end typing_lines_in_10_minutes_l215_215888


namespace equal_volume_cubes_l215_215442

noncomputable def volume_box : ℝ := 1 -- volume of the cubical box in cubic meters

noncomputable def edge_length_small_cube : ℝ := 0.04 -- edge length of small cubes in meters

noncomputable def number_of_cubes : ℝ := 15624.999999999998 -- number of small cubes

noncomputable def volume_small_cube : ℝ := edge_length_small_cube^3 -- volume of one small cube

theorem equal_volume_cubes : volume_box = volume_small_cube * number_of_cubes :=
  by
  -- Proof goes here
  sorry

end equal_volume_cubes_l215_215442


namespace geometric_sequence_sixth_term_l215_215482

-- Definitions of conditions
def a : ℝ := 512
def r : ℝ := (2 / a)^(1 / 7)

-- The proof statement
theorem geometric_sequence_sixth_term (h : a * r^7 = 2) : 512 * (r^5) = 16 :=
begin
  sorry
end

end geometric_sequence_sixth_term_l215_215482


namespace area_ratio_of_octagon_l215_215475

theorem area_ratio_of_octagon (A : ℝ) (hA : 0 < A) :
  let triangle_ABJ_area := A / 8
  let triangle_ACE_area := A / 2
  triangle_ABJ_area / triangle_ACE_area = 1 / 4 := by
  sorry

end area_ratio_of_octagon_l215_215475


namespace incorrect_statement_D_l215_215498

theorem incorrect_statement_D 
  (population : Set ℕ)
  (time_spent_sample : ℕ → ℕ)
  (sample_size : ℕ)
  (individual : ℕ)
  (h1 : ∀ s, s ∈ population → s ≤ 24)
  (h2 : ∀ i, i < sample_size → population (time_spent_sample i))
  (h3 : sample_size = 300)
  (h4 : ∀ i, i < 300 → time_spent_sample i = individual):
  ¬ (∀ i, i < 300 → time_spent_sample i = individual) :=
sorry

end incorrect_statement_D_l215_215498


namespace rachel_points_product_l215_215707

-- Define the scores in the first 10 games
def scores_first_10_games := [9, 5, 7, 4, 8, 6, 2, 3, 5, 6]

-- Define the conditions as given in the problem
def total_score_first_10_games := scores_first_10_games.sum = 55
def points_scored_in_game_11 (P₁₁ : ℕ) : Prop := P₁₁ < 10 ∧ (55 + P₁₁) % 11 = 0
def points_scored_in_game_12 (P₁₁ P₁₂ : ℕ) : Prop := P₁₂ < 10 ∧ (55 + P₁₁ + P₁₂) % 12 = 0

-- Prove the product of the points scored in eleventh and twelfth games
theorem rachel_points_product : ∃ P₁₁ P₁₂ : ℕ, total_score_first_10_games ∧ points_scored_in_game_11 P₁₁ ∧ points_scored_in_game_12 P₁₁ P₁₂ ∧ P₁₁ * P₁₂ = 0 :=
by 
  sorry -- proof not required

end rachel_points_product_l215_215707


namespace radius_of_circle_l215_215415

theorem radius_of_circle
  (r : ℝ)
  (h1 : ∀ x : ℝ, (x^2 + r = x) → (x^2 - x + r = 0) → ((-1)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
sorry

end radius_of_circle_l215_215415


namespace prove_a_eq_neg2_solve_inequality_for_a_leq0_l215_215557

-- Problem 1: Proving that a = -2 given the solution set of the inequality
theorem prove_a_eq_neg2 (a : ℝ) (h : ∀ x : ℝ, (-1 < x ∧ x < -1/2) ↔ (ax - 1) * (x + 1) > 0) : a = -2 := sorry

-- Problem 2: Solving the inequality (ax-1)(x+1) > 0 for different conditions on a
theorem solve_inequality_for_a_leq0 (a x : ℝ) (h_a_le_0 : a ≤ 0) : 
  (ax - 1) * (x + 1) > 0 ↔ 
    if a < -1 then -1 < x ∧ x < 1/a
    else if a = -1 then false
    else if -1 < a ∧ a < 0 then 1/a < x ∧ x < -1
    else x < -1 := sorry

end prove_a_eq_neg2_solve_inequality_for_a_leq0_l215_215557


namespace problem_l215_215443

variables (y S : ℝ)

theorem problem (h : 5 * (2 * y + 3 * Real.sqrt 3) = S) : 10 * (4 * y + 6 * Real.sqrt 3) = 4 * S :=
sorry

end problem_l215_215443


namespace distribute_balls_into_boxes_l215_215321

theorem distribute_balls_into_boxes : 
  ∀ (balls : ℕ) (boxes : ℕ), balls = 6 ∧ boxes = 3 → boxes^balls = 729 :=
by
  intros balls boxes h
  have hb : balls = 6 := h.1
  have hbox : boxes = 3 := h.2
  rw [hb, hbox]
  show 3^6 = 729
  exact Nat.pow 3 6 -- this would expand to the actual computation
  sorry

end distribute_balls_into_boxes_l215_215321


namespace inequality_proof_l215_215474

theorem inequality_proof (a b c d : ℝ) : 
  (a + b + c + d) * (a * b * (c + d) + (a + b) * c * d) - a * b * c * d ≤ 
  (1 / 2) * (a * (b + d) + b * (c + d) + c * (d + a))^2 :=
by
  sorry

end inequality_proof_l215_215474


namespace arccos_neg_one_eq_pi_l215_215947

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l215_215947


namespace curve_touches_all_Ca_l215_215951

theorem curve_touches_all_Ca (a : ℝ) (h : a > 0) : ∃ C : ℝ → ℝ, ∀ x y, (y - a^2)^2 = x^2 * (a^2 - x^2) → y = C x ∧ C x = 3 * x^2 / 4 :=
sorry

end curve_touches_all_Ca_l215_215951


namespace solution_fractional_equation_l215_215223

noncomputable def solve_fractional_equation : Prop :=
  ∀ x : ℝ, (4/(x-2) = 2/x) ↔ x = -2

theorem solution_fractional_equation :
  solve_fractional_equation :=
by
  sorry

end solution_fractional_equation_l215_215223


namespace max_binary_sequences_len24_l215_215868

/--
  The maximum number of binary sequences of length 24,
  such that any two sequences differ in at least 8 positions,
  is at most 4096.
-/
theorem max_binary_sequences_len24 :
  ∀ (S : set (vector bool 24)), 
    (∀ (x y ∈ S), x ≠ y → dist x y ≥ 8) → 
    S.finite ∧ S.card ≤ 4096 :=
by
  sorry

end max_binary_sequences_len24_l215_215868


namespace number_of_subsets_upper_bound_l215_215344

noncomputable def number_of_subsets_leq (n : ℕ) (λ : ℝ) (x : Fin n → ℝ) :=
  { A : Finset (Fin n) | ∑ i in A, x i ≥ λ }.card

theorem number_of_subsets_upper_bound (n : ℕ) (λ : ℝ) (x : Fin n → ℝ)
  (sum_eq_zero : ∑ i : Fin n, x i = 0)
  (sum_sq_eq_one : ∑ i : Fin n, x i ^ 2 = 1)
  (λ_pos : λ > 0) :
  number_of_subsets_leq n λ x ≤ 2^ (n-3) / λ^2 := by
  sorry

end number_of_subsets_upper_bound_l215_215344


namespace min_value_expression_l215_215589

/-- Prove that for integers a, b, c satisfying 1 ≤ a ≤ b ≤ c ≤ 5, the minimum value of the expression 
  (a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2 is 1.2595. -/
theorem min_value_expression (a b c : ℤ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  ∃ (min_val : ℝ), min_val = ((a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2) ∧ min_val = 1.2595 :=
by
  sorry

end min_value_expression_l215_215589


namespace length_sixth_episode_l215_215334

def length_first_episode : ℕ := 58
def length_second_episode : ℕ := 62
def length_third_episode : ℕ := 65
def length_fourth_episode : ℕ := 71
def length_fifth_episode : ℕ := 79
def total_viewing_time : ℕ := 450

theorem length_sixth_episode :
  length_first_episode + length_second_episode + length_third_episode + length_fourth_episode + length_fifth_episode + 115 = total_viewing_time := by
  sorry

end length_sixth_episode_l215_215334


namespace q_is_false_l215_215326

theorem q_is_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end q_is_false_l215_215326


namespace alpha_plus_beta_l215_215678

theorem alpha_plus_beta (α β : ℝ) 
  (hα : 0 < α ∧ α < Real.pi / 2) 
  (hβ : 0 < β ∧ β < Real.pi / 2)
  (h_sin_alpha : Real.sin α = Real.sqrt 10 / 10)
  (h_cos_beta : Real.cos β = 2 * Real.sqrt 5 / 5) :
  α + β = Real.pi / 4 :=
sorry

end alpha_plus_beta_l215_215678


namespace trig_identity_l215_215307

theorem trig_identity (f : ℝ → ℝ) (ϕ : ℝ) (h₁ : ∀ x, f x = 2 * Real.sin (2 * x + ϕ)) (h₂ : 0 < ϕ) (h₃ : ϕ < π) (h₄ : f 0 = 1) :
  f ϕ = 2 :=
sorry

end trig_identity_l215_215307


namespace train_length_is_900_l215_215120

def train_length_crossing_pole (L V : ℕ) : Prop :=
  L = V * 18

def train_length_crossing_platform (L V : ℕ) : Prop :=
  L + 1050 = V * 39

theorem train_length_is_900 (L V : ℕ) (h1 : train_length_crossing_pole L V) (h2 : train_length_crossing_platform L V) : L = 900 := 
by
  sorry

end train_length_is_900_l215_215120


namespace polynomial_value_at_2_l215_215895

def f (x : ℝ) : ℝ := 2 * x^5 + 4 * x^4 - 2 * x^3 - 3 * x^2 + x

theorem polynomial_value_at_2 : f 2 = 102 := by
  sorry

end polynomial_value_at_2_l215_215895


namespace smallest_prime_10_less_than_perfect_square_l215_215774

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_10_less_than_perfect_square :
  ∃ (a : ℕ), is_prime a ∧ (∃ (n : ℕ), a = n^2 - 10) ∧ (∀ (b : ℕ), is_prime b ∧ (∃ (m : ℕ), b = m^2 - 10) → a ≤ b) ∧ a = 71 := 
by
  sorry

end smallest_prime_10_less_than_perfect_square_l215_215774


namespace remainder_M_mod_32_l215_215037

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215037


namespace equal_angles_l215_215292

variables {P A B C D Q : Point}
variables [circle : Circle]
variables (PA tangent_to circle) (PB tangent_to circle)
variables (PCD secant_to circle)
variables (P_outside_circle : ¬(P ∈ circle))
variables (C_between_P_and_D : between P C D)
variables (on_chord_CD : Q ∈ segment C D)
variables (angle_DAQ_eq_angle_PBC : ∠ D A Q = ∠ P B C)

theorem equal_angles (h : ∠ D A Q = ∠ P B C) : ∠ D B Q = ∠ P A C :=
by sorry

end equal_angles_l215_215292


namespace intersection_M_N_l215_215560

  open Set

  def M : Set ℝ := {x | Real.log x > 0}
  def N : Set ℝ := {x | x^2 ≤ 4}

  theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
  by
    sorry
  
end intersection_M_N_l215_215560


namespace find_fourth_score_l215_215965

theorem find_fourth_score
  (a b c : ℕ) (d : ℕ)
  (ha : a = 70) (hb : b = 80) (hc : c = 90)
  (average_eq : (a + b + c + d) / 4 = 70) :
  d = 40 := 
sorry

end find_fourth_score_l215_215965


namespace birds_on_fence_l215_215879

def number_of_birds_on_fence : ℕ := 20

theorem birds_on_fence (x : ℕ) (h : 2 * x + 10 = 50) : x = number_of_birds_on_fence :=
by
  sorry

end birds_on_fence_l215_215879


namespace bamboo_tube_middle_capacity_l215_215601

-- Definitions and conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem bamboo_tube_middle_capacity:
  ∃ a d, (arithmetic_sequence a d 0 + arithmetic_sequence a d 1 + arithmetic_sequence a d 2 = 3.9) ∧
         (arithmetic_sequence a d 5 + arithmetic_sequence a d 6 + arithmetic_sequence a d 7 + arithmetic_sequence a d 8 = 3) ∧
         (arithmetic_sequence a d 4 = 1) :=
sorry

end bamboo_tube_middle_capacity_l215_215601


namespace num_ways_to_distribute_balls_l215_215317

-- Define the conditions
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- The statement to prove
theorem num_ways_to_distribute_balls : num_boxes ^ num_balls = 729 :=
by {
  -- Proof steps would go here
  sorry
}

end num_ways_to_distribute_balls_l215_215317


namespace roots_of_equation_l215_215748

theorem roots_of_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end roots_of_equation_l215_215748


namespace max_cursed_roads_l215_215708

theorem max_cursed_roads (cities roads N kingdoms : ℕ) (h1 : cities = 1000) (h2 : roads = 2017)
  (h3 : cities = 1 → cities = 1000 → N ≤ 1024 → kingdoms = 7 → True) :
  max_N = 1024 :=
by
  sorry

end max_cursed_roads_l215_215708


namespace number_with_29_proper_divisors_is_720_l215_215792

theorem number_with_29_proper_divisors_is_720
  (n : ℕ) (h1 : n < 1000)
  (h2 : ∀ d, 1 < d ∧ d < n -> ∃ m, n = d * m):
  n = 720 := by
  sorry

end number_with_29_proper_divisors_is_720_l215_215792


namespace sum_of_square_roots_of_consecutive_odd_numbers_l215_215654

theorem sum_of_square_roots_of_consecutive_odd_numbers :
  (Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9)) = 15 :=
by
  sorry

end sum_of_square_roots_of_consecutive_odd_numbers_l215_215654


namespace number_of_valid_rods_l215_215192

def rods : List Nat := List.range' 1 30
def selected_rods : List Nat := [4, 9, 17]

theorem number_of_valid_rods :
  ∃ n, n = 22 ∧ ∀ x ∈ rods, x ∈ (List.range' 5 25).filter (λ y, y ≠ 4 ∧ y ≠ 9 ∧ y ≠ 17) → List.length (List.range' 5 25) - 3 = n := by
  sorry

end number_of_valid_rods_l215_215192


namespace difference_not_divisible_by_1976_l215_215246

theorem difference_not_divisible_by_1976 (A B : ℕ) (hA : 100 ≤ A) (hA' : A < 1000) (hB : 100 ≤ B) (hB' : B < 1000) (h : A ≠ B) :
  ¬ (1976 ∣ (1000 * A + B - (1000 * B + A))) :=
by
  sorry

end difference_not_divisible_by_1976_l215_215246


namespace power_division_result_l215_215392

theorem power_division_result : (-2)^(2014) / (-2)^(2013) = -2 :=
by
  sorry

end power_division_result_l215_215392


namespace product_of_odd_primes_mod_32_l215_215027

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215027


namespace quadratic_other_x_intercept_l215_215423

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → a * x^2 + b * x + c = -3)
  (h_intercept : a * 1^2 + b * 1 + c = 0) : 
  ∃ x0 : ℝ, x0 = 9 ∧ a * x0^2 + b * x0 + c = 0 :=
by
  sorry

end quadratic_other_x_intercept_l215_215423


namespace download_time_l215_215098

theorem download_time (avg_speed : ℤ) (size_A size_B size_C : ℤ) (gb_to_mb : ℤ) (secs_in_min : ℤ) :
  avg_speed = 30 →
  size_A = 450 →
  size_B = 240 →
  size_C = 120 →
  gb_to_mb = 1000 →
  secs_in_min = 60 →
  ( (size_A * gb_to_mb + size_B * gb_to_mb + size_C * gb_to_mb ) / avg_speed ) / secs_in_min = 450 := by
  intros h_avg h_A h_B h_C h_gb h_secs
  sorry

end download_time_l215_215098


namespace find_m_for_unique_solution_l215_215796

theorem find_m_for_unique_solution :
  ∃ m : ℝ, (m = -8 + 2 * Real.sqrt 15 ∨ m = -8 - 2 * Real.sqrt 15) ∧ 
  ∀ x : ℝ, (mx - 2 ≠ 0 → (x + 3) / (mx - 2) = x + 1 ↔ ∃! x : ℝ, (mx - 2) * (x + 1) = (x + 3)) :=
sorry

end find_m_for_unique_solution_l215_215796


namespace largest_even_number_l215_215228

theorem largest_even_number (x : ℤ) 
  (h : x + (x + 2) + (x + 4) = x + 18) : x + 4 = 10 :=
by
  sorry

end largest_even_number_l215_215228


namespace part1_part2_l215_215556

open Real

def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

theorem part1 (m : ℝ) : (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 := 
sorry

theorem part2 : {x : ℝ | x^2 - 8 * x + 15 + f x ≤ 0} = {x : ℝ | 5 - sqrt 3 ≤ x ∧ x ≤ 6} :=
sorry

end part1_part2_l215_215556


namespace contrapositive_example_l215_215367

theorem contrapositive_example (x : ℝ) : (x > 2 → x > 0) ↔ (x ≤ 2 → x ≤ 0) :=
by
  sorry

end contrapositive_example_l215_215367


namespace defense_attorney_mistake_l215_215727

variable (P Q : Prop)

theorem defense_attorney_mistake (h1 : P → Q) (h2 : ¬ (P → Q)) : P ∧ ¬ Q :=
by {
  sorry
}

end defense_attorney_mistake_l215_215727


namespace max_load_per_truck_l215_215072

-- Definitions based on given conditions
def num_trucks : ℕ := 3
def total_boxes : ℕ := 240
def lighter_box_weight : ℕ := 10
def heavier_box_weight : ℕ := 40

-- Proof problem statement
theorem max_load_per_truck :
  (total_boxes / 2) * lighter_box_weight + (total_boxes / 2) * heavier_box_weight = 6000 →
  6000 / num_trucks = 2000 :=
by sorry

end max_load_per_truck_l215_215072


namespace breakfast_time_correct_l215_215286

noncomputable def breakfast_time_calc (x : ℚ) : ℚ :=
  (7 * 60) + (300 / 13)

noncomputable def coffee_time_calc (y : ℚ) : ℚ :=
  (7 * 60) + (420 / 11)

noncomputable def total_breakfast_time : ℚ :=
  coffee_time_calc ((420 : ℚ) / 11) - breakfast_time_calc ((300 : ℚ) / 13)

theorem breakfast_time_correct :
  total_breakfast_time = 15 + (6 / 60) :=
by
  sorry

end breakfast_time_correct_l215_215286


namespace toy_poodle_height_l215_215489

-- Define the heights of the poodles
variables (S M T : ℝ)

-- Conditions
def std_taller_min : Prop := S = M + 8
def min_taller_toy : Prop := M = T + 6
def std_height : Prop := S = 28

-- Goal: How tall is the toy poodle?
theorem toy_poodle_height (h1 : std_taller_min S M)
                          (h2 : min_taller_toy M T)
                          (h3 : std_height S) : T = 14 :=
by 
  sorry

end toy_poodle_height_l215_215489


namespace crucian_vs_bream_l215_215394

-- Define the weights of crucian, bream, and perch
variables (C B P : ℝ)

-- Given conditions
axiom h1 : 6 * C > 10 * B
axiom h2 : 6 * C < 5 * P
axiom h3 : 10 * C > 8 * P

-- Statement to prove
theorem crucian_vs_bream (C B P : ℝ) (h1 : 6 * C > 10 * B) (h2 : 6 * C < 5 * P) (h3 : 10 * C > 8 * P) : 
  2 * C > 3 * B :=
sorry

end crucian_vs_bream_l215_215394


namespace expected_pairs_in_same_row_or_column_l215_215862

noncomputable def grid := Finset (Fin 49)

-- Define a random arrangement of numbers in a 7x7 grid
def random_arrangement (g: grid) : Prop :=
  g.card = 49 -- All numbers from 1 to 49 are present

-- Define the probability that a given pair of numbers is in the same row or column
def same_row_or_column_probability := 1 / 4

-- Define the expected number of pairs
theorem expected_pairs_in_same_row_or_column :
  let total_pairs := (49.choose 2) in
  let probability_same_in_both := same_row_or_column_probability ^ 2 in
  let expected_value := total_pairs * probability_same_in_both in
  expected_value = 73.5 := by
begin
  let total_pairs := 49.choose 2,
  let probability_same_in_both := (1 / 4) ^ 2,
  let expected_value := total_pairs * probability_same_in_both,
  have total_pairs_eq : total_pairs = 1176 := by norm_num,
  have probability_eq : probability_same_in_both = 1 / 16 := by norm_num,
  have expected_value_eq : expected_value = 73.5 := by norm_num,
  exact expected_value_eq,
end

end expected_pairs_in_same_row_or_column_l215_215862


namespace circle_parabola_intersections_l215_215534

theorem circle_parabola_intersections : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, (p.1 ^ 2 + p.2 ^ 2 = 16) ∧ (p.2 = p.1 ^ 2 - 4)) ∧
  points.card = 3 := 
sorry

end circle_parabola_intersections_l215_215534


namespace proof_problem_l215_215825

def p := 8 + 7 = 16
def q := Real.pi > 3

theorem proof_problem :
  (¬p ∧ q) ∧ ((p ∨ q) = true) ∧ ((p ∧ q) = false) ∧ ((¬p) = true) := sorry

end proof_problem_l215_215825


namespace k_plus_a_equals_three_halves_l215_215567

theorem k_plus_a_equals_three_halves :
  ∃ (k a : ℝ), (2 = k * 4 ^ a) ∧ (k + a = 3 / 2) :=
sorry

end k_plus_a_equals_three_halves_l215_215567


namespace range_of_a_l215_215685

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

/-- If A ⊆ B, then the range of values for 'a' satisfies -4 ≤ a ≤ -1 -/
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : -4 ≤ a ∧ a ≤ -1 :=
by
  sorry

end range_of_a_l215_215685


namespace connie_total_markers_l215_215279

theorem connie_total_markers : 2315 + 1028 = 3343 :=
by
  sorry

end connie_total_markers_l215_215279


namespace solve_for_y_l215_215356

theorem solve_for_y (y : ℝ) (h : ∛(5 - 2 / y) = -3) : y = 1/16 :=
sorry

end solve_for_y_l215_215356


namespace sum_of_x_coordinates_l215_215076

def line1 (x : ℝ) : ℝ := -3 * x - 5
def line2 (x : ℝ) : ℝ := 2 * x - 3

def has_x_intersect (line : ℝ → ℝ) (y : ℝ) : Prop := ∃ x : ℝ, line x = y

theorem sum_of_x_coordinates :
  (∃ x1 x2 : ℝ, line1 x1 = 2.2 ∧ line2 x2 = 2.2 ∧ x1 + x2 = 0.2) :=
  sorry

end sum_of_x_coordinates_l215_215076


namespace value_of_b_l215_215299

theorem value_of_b (x y b : ℝ) (h1: 7^(3 * x - 1) * b^(4 * y - 3) = 49^x * 27^y) (h2: x + y = 4) : b = 3 :=
by
  sorry

end value_of_b_l215_215299


namespace inequality_pos_real_l215_215824

theorem inequality_pos_real (
  a b c : ℝ
) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  abc ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧ 
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end inequality_pos_real_l215_215824


namespace mathematical_proof_l215_215547

noncomputable def proof_problem (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : Prop :=
  (1 + x) / y < 2 ∨ (1 + y) / x < 2

theorem mathematical_proof (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : proof_problem x y hx_pos hxy_gt2 :=
by {
  sorry
}

end mathematical_proof_l215_215547


namespace evaluate_f_5_minus_f_neg_5_l215_215697

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x + 3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := 
  by
    sorry

end evaluate_f_5_minus_f_neg_5_l215_215697


namespace gumballs_multiple_purchased_l215_215340

-- Definitions
def joanna_initial : ℕ := 40
def jacques_initial : ℕ := 60
def final_each : ℕ := 250

-- Proof statement
theorem gumballs_multiple_purchased (m : ℕ) :
  (joanna_initial + joanna_initial * m) + (jacques_initial + jacques_initial * m) = 2 * final_each →
  m = 4 :=
by 
  sorry

end gumballs_multiple_purchased_l215_215340


namespace valid_pairs_l215_215799

theorem valid_pairs
  (x y : ℕ)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_div : ∃ k : ℕ, k > 0 ∧ k * (2 * x + 7 * y) = 7 * x + 2 * y) :
  ∃ a : ℕ, a > 0 ∧ (x = a ∧ y = a ∨ x = 4 * a ∧ y = a ∨ x = 19 * a ∧ y = a) :=
by
  sorry

end valid_pairs_l215_215799


namespace fernanda_savings_calculation_l215_215648

theorem fernanda_savings_calculation :
  ∀ (aryan_debt kyro_debt aryan_payment kyro_payment savings total_savings : ℝ),
    aryan_debt = 1200 ∧
    aryan_debt = 2 * kyro_debt ∧
    aryan_payment = (60 / 100) * aryan_debt ∧
    kyro_payment = (80 / 100) * kyro_debt ∧
    savings = 300 ∧
    total_savings = savings + aryan_payment + kyro_payment →
    total_savings = 1500 := by
    sorry

end fernanda_savings_calculation_l215_215648


namespace solve_diff_eqn_l215_215541

noncomputable def diff_eqn (y : ℝ → ℝ) : Prop :=
  ∀ x, deriv y x = 2 + y x

def initial_condition (y : ℝ → ℝ) : Prop :=
  y 0 = 3

def particular_solution (y : ℝ → ℝ) : Prop :=
  y = λ x, 5 * Real.exp x - 2

theorem solve_diff_eqn :
  ∃ y : ℝ → ℝ, diff_eqn y ∧ initial_condition y ∧ particular_solution y :=
sorry

end solve_diff_eqn_l215_215541


namespace largest_value_l215_215322

-- Definition: Given the condition of a quadratic equation
def equation (a : ℚ) : Prop :=
  8 * a^2 + 6 * a + 2 = 0

-- Theorem: Prove the largest value of 3a + 2 is 5/4 given the condition
theorem largest_value (a : ℚ) (h : equation a) : 
  ∃ m, ∀ b, equation b → (3 * b + 2 ≤ m) ∧ (m = 5 / 4) :=
by
  sorry

end largest_value_l215_215322


namespace solve_system1_solve_system2_l215_215872

-- Definitions for the first system of equations
def system1_equation1 (x y : ℚ) := 3 * x - 6 * y = 4
def system1_equation2 (x y : ℚ) := x + 5 * y = 6

-- Definitions for the second system of equations
def system2_equation1 (x y : ℚ) := x / 4 + y / 3 = 3
def system2_equation2 (x y : ℚ) := 3 * (x - 4) - 2 * (y - 1) = -1

-- Lean statement for proving the solution to the first system
theorem solve_system1 :
  ∃ (x y : ℚ), system1_equation1 x y ∧ system1_equation2 x y ∧ x = 8 / 3 ∧ y = 2 / 3 :=
by
  sorry

-- Lean statement for proving the solution to the second system
theorem solve_system2 :
  ∃ (x y : ℚ), system2_equation1 x y ∧ system2_equation2 x y ∧ x = 6 ∧ y = 9 / 2 :=
by
  sorry

end solve_system1_solve_system2_l215_215872


namespace set_properties_proof_l215_215313

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := Icc (-2 : ℝ) 2)
variable (N : Set ℝ := Iic (1 : ℝ))

theorem set_properties_proof :
  (M ∪ N = Iic (2 : ℝ)) ∧
  (M ∩ N = Icc (-2 : ℝ) 1) ∧
  (U \ N = Ioi (1 : ℝ)) := by
  sorry

end set_properties_proof_l215_215313


namespace complement_intersection_l215_215833

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

#check (Set.compl B) ∩ A = {1}

theorem complement_intersection (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 5}) (hB : B = {2, 3, 5}) :
   (U \ B) ∩ A = {1} :=
by
  sorry

end complement_intersection_l215_215833


namespace contractor_engaged_days_l215_215254

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end contractor_engaged_days_l215_215254


namespace part1_part2_l215_215828

def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

theorem part1 (a b c x : ℝ) (h1 : |a - b| > c) : f x a b > c :=
  by sorry

theorem part2 (a : ℝ) (h1 : ∃ (x : ℝ), f x a 1 < 2 - |a - 2|) : 1/2 < a ∧ a < 5/2 :=
  by sorry

end part1_part2_l215_215828


namespace sales_fifth_month_l215_215773

-- Definitions based on conditions
def sales1 : ℝ := 5420
def sales2 : ℝ := 5660
def sales3 : ℝ := 6200
def sales4 : ℝ := 6350
def sales6 : ℝ := 8270
def average_sale : ℝ := 6400

-- Lean proof problem statement
theorem sales_fifth_month :
  sales1 + sales2 + sales3 + sales4 + sales6 + s = 6 * average_sale  →
  s = 6500 :=
by
  sorry

end sales_fifth_month_l215_215773


namespace inequality_proof_l215_215157

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
by 
  sorry

end inequality_proof_l215_215157


namespace fraction_addition_l215_215652

theorem fraction_addition :
  (3/8 : ℚ) / (4/9 : ℚ) + 1/6 = 97/96 := by
  sorry

end fraction_addition_l215_215652


namespace product_of_odd_primes_mod_32_l215_215025

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215025


namespace max_triplets_l215_215094

-- Define conditions as Lean functions and predicates
def points := 1955

def is_valid_triplet (triplets : Finset (Finset ℕ)) : Prop :=
  ∀ t1 t2 ∈ triplets, t1 ≠ t2 → ∃! x, x ∈ t1 ∧ x ∈ t2

-- Define the main theorem statement using the conditions
theorem max_triplets (triplets : Finset (Finset ℕ)) : 
  triplets.card = 977 ∧
  ∀ t ∈ triplets, t.card = 3 ∧ is_valid_triplet triplets :=
sorry

end max_triplets_l215_215094


namespace regression_line_is_y_eq_x_plus_1_l215_215848

def Point : Type := ℝ × ℝ

def A : Point := (1, 2)
def B : Point := (2, 3)
def C : Point := (3, 4)
def D : Point := (4, 5)

def points : List Point := [A, B, C, D]

noncomputable def mean (lst : List ℝ) : ℝ :=
  (lst.foldr (fun x acc => x + acc) 0) / lst.length

noncomputable def regression_line (pts : List Point) : ℝ → ℝ :=
  let xs := pts.map Prod.fst
  let ys := pts.map Prod.snd
  fun x : ℝ => x + 1

theorem regression_line_is_y_eq_x_plus_1 :
  regression_line points = fun x => x + 1 := sorry

end regression_line_is_y_eq_x_plus_1_l215_215848


namespace triangle_inequality_l215_215588

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2*(b + c - a) + b^2*(c + a - b) + c^2*(a + b - c) ≤ 3*a*b*c :=
by
  sorry

end triangle_inequality_l215_215588


namespace inequality_problem_l215_215153

theorem inequality_problem
  (a b c d e f : ℝ)
  (h_cond : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
by
  sorry

end inequality_problem_l215_215153


namespace star_in_S_star_associative_l215_215902

def S (x : ℕ) : Prop :=
  x > 1 ∧ x % 2 = 1

def f (x : ℕ) : ℕ :=
  Nat.log2 x

def star (a b : ℕ) : ℕ :=
  a + 2 ^ (f a) * (b - 3)

theorem star_in_S (a b : ℕ) (h_a : S a) (h_b : S b) : S (star a b) :=
  sorry

theorem star_associative (a b c : ℕ) (h_a : S a) (h_b : S b) (h_c : S c) :
  star (star a b) c = star a (star b c) :=
  sorry

end star_in_S_star_associative_l215_215902


namespace extremum_is_not_unique_l215_215480

-- Define the extremum conditionally in terms of unique extremum within an interval for a function
def isExtremum {α : Type*} [Preorder α] (f : α → ℝ) (x : α) :=
  ∀ y, f y ≤ f x ∨ f x ≤ f y

theorem extremum_is_not_unique (α : Type*) [Preorder α] (f : α → ℝ) :
  ¬ ∀ x, isExtremum f x → (∀ y, isExtremum f y → x = y) :=
by
  sorry

end extremum_is_not_unique_l215_215480


namespace randy_initial_amount_l215_215729

theorem randy_initial_amount (spend_per_trip: ℤ) (trips_per_month: ℤ) (dollars_left_after_year: ℤ) (total_month_months: ℤ := 12):
  (spend_per_trip = 2 ∧ trips_per_month = 4 ∧ dollars_left_after_year = 104) → spend_per_trip * trips_per_month * total_month_months + dollars_left_after_year = 200 := 
by
  sorry

end randy_initial_amount_l215_215729


namespace product_of_odd_primes_mod_32_l215_215026

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215026


namespace probability_B_winning_l215_215874

def P_A : ℝ := 0.2
def P_D : ℝ := 0.5
def P_B : ℝ := 1 - (P_A + P_D)

theorem probability_B_winning : P_B = 0.3 :=
by
  -- Proof steps go here
  sorry

end probability_B_winning_l215_215874


namespace remaining_pictures_l215_215352

theorem remaining_pictures (first_book : ℕ) (second_book : ℕ) (third_book : ℕ) (colored_pictures : ℕ) :
  first_book = 23 → second_book = 32 → third_book = 45 → colored_pictures = 44 →
  (first_book + second_book + third_book - colored_pictures) = 56 :=
by
  sorry

end remaining_pictures_l215_215352


namespace toy_poodle_height_l215_215490

-- Define the heights of the poodles
variables (S M T : ℝ)

-- Conditions
def std_taller_min : Prop := S = M + 8
def min_taller_toy : Prop := M = T + 6
def std_height : Prop := S = 28

-- Goal: How tall is the toy poodle?
theorem toy_poodle_height (h1 : std_taller_min S M)
                          (h2 : min_taller_toy M T)
                          (h3 : std_height S) : T = 14 :=
by 
  sorry

end toy_poodle_height_l215_215490


namespace initial_cupcakes_l215_215065

variable (x : ℕ) -- Define x as the number of cupcakes Robin initially made

-- Define the conditions provided in the problem
def cupcakes_sold := 22
def cupcakes_made := 39
def final_cupcakes := 59

-- Formalize the problem statement: Prove that given the conditions, the initial cupcakes equals 42
theorem initial_cupcakes:
  x - cupcakes_sold + cupcakes_made = final_cupcakes → x = 42 := 
by
  -- Placeholder for the proof
  sorry

end initial_cupcakes_l215_215065


namespace cathy_initial_money_l215_215273

-- Definitions of the conditions
def moneyFromDad : Int := 25
def moneyFromMom : Int := 2 * moneyFromDad
def totalMoneyReceived : Int := moneyFromDad + moneyFromMom
def currentMoney : Int := 87

-- Theorem stating the proof problem
theorem cathy_initial_money (initialMoney : Int) :
  initialMoney + totalMoneyReceived = currentMoney → initialMoney = 12 :=
by
  sorry

end cathy_initial_money_l215_215273


namespace gcd_840_1764_gcd_561_255_l215_215109

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := 
by
  sorry

theorem gcd_561_255 : Nat.gcd 561 255 = 51 :=
by
  sorry

end gcd_840_1764_gcd_561_255_l215_215109


namespace union_of_M_and_N_l215_215312

open Set

noncomputable def M : Set ℝ := {x | x < 1}
noncomputable def N : Set ℝ := {x | 2^x > 1}

theorem union_of_M_and_N : M ∪ N = univ :=
by
  sorry

end union_of_M_and_N_l215_215312


namespace find_alpha_l215_215427

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = - (Real.sqrt 3 / 2)) (h_range : 0 < α ∧ α < Real.pi) : α = 5 * Real.pi / 6 :=
sorry

end find_alpha_l215_215427


namespace remainder_M_mod_32_l215_215034

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215034


namespace solve_quadratic_l215_215210

theorem solve_quadratic (x : ℝ) : 2 * x^2 - x = 2 ↔ x = (1 + Real.sqrt 17) / 4 ∨ x = (1 - Real.sqrt 17) / 4 := by
  sorry

end solve_quadratic_l215_215210


namespace solve_purchase_price_problem_l215_215745

def purchase_price_problem : Prop :=
  ∃ P : ℝ, (0.10 * P + 12 = 35) ∧ (P = 230)

theorem solve_purchase_price_problem : purchase_price_problem :=
  by
    sorry

end solve_purchase_price_problem_l215_215745


namespace solve_y_l215_215359

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end solve_y_l215_215359


namespace min_height_of_box_with_surface_area_condition_l215_215782

theorem min_height_of_box_with_surface_area_condition {x : ℕ}  
(h : 2*x^2 + 4*x*(x + 6) ≥ 150) (hx: x ≥ 5) : (x + 6) = 11 := by
  sorry

end min_height_of_box_with_surface_area_condition_l215_215782


namespace third_side_range_l215_215309

theorem third_side_range (a : ℝ) (h₃ : 0 < a ∧ a ≠ 0) (h₅ : 0 < a ∧ a ≠ 0): 
  (2 < a ∧ a < 8) ↔ (3 - 5 < a ∧ a < 3 + 5) :=
by
  sorry

end third_side_range_l215_215309


namespace credit_limit_l215_215859

theorem credit_limit (paid_tuesday : ℕ) (paid_thursday : ℕ) (remaining_payment : ℕ) (full_payment : ℕ) 
  (h1 : paid_tuesday = 15) 
  (h2 : paid_thursday = 23) 
  (h3 : remaining_payment = 62) 
  (h4 : full_payment = paid_tuesday + paid_thursday + remaining_payment) : 
  full_payment = 100 := 
by
  sorry

end credit_limit_l215_215859


namespace rope_lengths_l215_215747

theorem rope_lengths (joey_len chad_len mandy_len : ℝ) (h1 : joey_len = 56) 
  (h2 : 8 / 3 = joey_len / chad_len) (h3 : 5 / 2 = chad_len / mandy_len) : 
  chad_len = 21 ∧ mandy_len = 8.4 :=
by
  sorry

end rope_lengths_l215_215747


namespace num_digits_abc_l215_215565

theorem num_digits_abc (a b c : ℕ) (n : ℕ) (h_a : 10^(n-1) ≤ a ∧ a < 10^n) (h_b : 10^(n-1) ≤ b ∧ b < 10^n) (h_c : 10^(n-1) ≤ c ∧ c < 10^n) :
  ¬ ((Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 1) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 2)) :=
sorry

end num_digits_abc_l215_215565


namespace box_volume_correct_l215_215261

-- Define the dimensions of the original sheet
def length_original : ℝ := 48
def width_original : ℝ := 36

-- Define the side length of the squares cut from each corner
def side_length_cut : ℝ := 4

-- Define the new dimensions after cutting the squares
def new_length : ℝ := length_original - 2 * side_length_cut
def new_width : ℝ := width_original - 2 * side_length_cut

-- Define the height of the box
def height_box : ℝ := side_length_cut

-- Define the expected volume of the box
def volume_box_expected : ℝ := 4480

-- Prove that the calculated volume is equal to the expected volume
theorem box_volume_correct :
  new_length * new_width * height_box = volume_box_expected := by
  sorry

end box_volume_correct_l215_215261


namespace beaker_filling_l215_215628

theorem beaker_filling (C : ℝ) (hC : 0 < C) :
    let small_beaker_salt := (1/2) * C
    let large_beaker_capacity := 5 * C
    let large_beaker_fresh := large_beaker_capacity / 5
    let large_beaker_total_fill := large_beaker_fresh + small_beaker_salt
    (large_beaker_total_fill / large_beaker_capacity) = 3 / 10 :=
by
    let small_beaker_salt := (1/2) * C
    let large_beaker_capacity := 5 * C
    let large_beaker_fresh := large_beaker_capacity / 5
    let large_beaker_total_fill := large_beaker_fresh + small_beaker_salt
    show (large_beaker_total_fill / large_beaker_capacity) = 3 / 10
    sorry

end beaker_filling_l215_215628


namespace percentage_problem_l215_215696

theorem percentage_problem (x : ℝ) (h : (3 / 8) * x = 141) : (round (0.3208 * x) = 121) :=
by
  sorry

end percentage_problem_l215_215696


namespace trains_clear_time_l215_215374

theorem trains_clear_time :
  ∀ (length_A length_B length_C : ℕ)
    (speed_A_kmph speed_B_kmph speed_C_kmph : ℕ)
    (distance_AB distance_BC : ℕ),
  length_A = 160 ∧ length_B = 320 ∧ length_C = 480 ∧
  speed_A_kmph = 42 ∧ speed_B_kmph = 30 ∧ speed_C_kmph = 48 ∧
  distance_AB = 200 ∧ distance_BC = 300 →
  ∃ (time_clear : ℚ), time_clear = 50.78 :=
by
  intros length_A length_B length_C
         speed_A_kmph speed_B_kmph speed_C_kmph
         distance_AB distance_BC h
  sorry

end trains_clear_time_l215_215374


namespace two_numbers_equal_l215_215948

variables {a b c : ℝ}
variable (h1 : a + b^2 + c^2 = a^2 + b + c^2)
variable (h2 : a^2 + b + c^2 = a^2 + b^2 + c)

theorem two_numbers_equal (h1 : a + b^2 + c^2 = a^2 + b + c^2) (h2 : a^2 + b + c^2 = a^2 + b^2 + c) :
  a = b ∨ a = c ∨ b = c :=
by
  sorry

end two_numbers_equal_l215_215948


namespace inequality_range_a_l215_215239

open Real

theorem inequality_range_a (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 :=
sorry

end inequality_range_a_l215_215239


namespace problem_1_problem_2_l215_215310

def setA (x : ℝ) : Prop := 2 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 3 < x ∧ x ≤ 10
def setC (a : ℝ) (x : ℝ) : Prop := a - 5 < x ∧ x < a

theorem problem_1 (x : ℝ) :
  (setA x ∧ setB x ↔ 3 < x ∧ x < 7) ∧
  (setA x ∨ setB x ↔ 2 ≤ x ∧ x ≤ 10) := 
by sorry

theorem problem_2 (a : ℝ) :
  (∀ x, setC a x → (2 ≤ x ∧ x ≤ 10)) ↔ (7 ≤ a ∧ a ≤ 10) :=
by sorry

end problem_1_problem_2_l215_215310


namespace minimize_sum_find_c_l215_215281

theorem minimize_sum_find_c (a b c d e f : ℕ) (h : a + 2 * b + 6 * c + 30 * d + 210 * e + 2310 * f = 2 ^ 15) 
  (h_min : ∀ a' b' c' d' e' f' : ℕ, a' + 2 * b' + 6 * c' + 30 * d' + 210 * e' + 2310 * f' = 2 ^ 15 → 
  a' + b' + c' + d' + e' + f' ≥ a + b + c + d + e + f) :
  c = 1 :=
sorry

end minimize_sum_find_c_l215_215281


namespace number_and_sum_of_g3_l215_215723

-- Define the function g with its conditions
variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (x * g y - x) = 2 * x * y + g x)

-- Define the problem parameters
def n : ℕ := sorry -- Number of possible values of g(3)
def s : ℝ := sorry -- Sum of all possible values of g(3)

-- The main statement to be proved
theorem number_and_sum_of_g3 : n * s = 0 := sorry

end number_and_sum_of_g3_l215_215723


namespace even_natural_number_factors_count_l215_215802

def is_valid_factor (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a ≤ 3 ∧ 
  0 ≤ b ∧ b ≤ 2 ∧ 
  0 ≤ c ∧ c ≤ 2 ∧ 
  a + b + c ≤ 4

noncomputable def count_valid_factors : ℕ :=
  Nat.card { x : ℕ × ℕ × ℕ // is_valid_factor x.1 x.2.1 x.2.2 }

theorem even_natural_number_factors_count : count_valid_factors = 15 := 
  sorry

end even_natural_number_factors_count_l215_215802


namespace problem_I_problem_II_l215_215434

namespace ProofProblems

def f (x a : ℝ) : ℝ := |x - a| + |x + 5|

theorem problem_I (x : ℝ) : (f x 1) ≥ 2 * |x + 5| ↔ x ≤ -2 := 
by sorry

theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, (f x a) ≥ 8) ↔ (a ≥ 3 ∨ a ≤ -13) := 
by sorry

end ProofProblems

end problem_I_problem_II_l215_215434


namespace right_triangles_sides_l215_215370

theorem right_triangles_sides (a b c p S r DH FC FH: ℝ)
  (h₁ : a = 10)
  (h₂ : b = 10)
  (h₃ : c = 12)
  (h₄ : p = (a + b + c) / 2)
  (h₅ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₆ : r = S / p)
  (h₇ : DH = (c / 2) - r)
  (h₈ : FC = (a * r) / DH)
  (h₉ : FH = Real.sqrt (FC^2 - DH^2))
: FC = 3 ∧ DH = 4 ∧ FH = 5 := by
  sorry

end right_triangles_sides_l215_215370


namespace no_integer_solutions_l215_215963

theorem no_integer_solutions (n : ℕ) (h : 2 ≤ n) :
  ¬ ∃ x y z : ℤ, x^2 + y^2 = z^n :=
sorry

end no_integer_solutions_l215_215963


namespace average_cost_parking_l215_215736

theorem average_cost_parking :
  let cost_first_2_hours := 12.00
  let cost_per_additional_hour := 1.75
  let total_hours := 9
  let total_cost := cost_first_2_hours + cost_per_additional_hour * (total_hours - 2)
  let average_cost_per_hour := total_cost / total_hours
  average_cost_per_hour = 2.69 :=
by
  sorry

end average_cost_parking_l215_215736


namespace fgh_supermarkets_l215_215497

theorem fgh_supermarkets (U C : ℕ) 
  (h1 : U + C = 70) 
  (h2 : U = C + 14) : U = 42 :=
by
  sorry

end fgh_supermarkets_l215_215497


namespace product_of_solutions_of_t_squared_eq_49_l215_215818

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l215_215818


namespace problem_statement_l215_215439

open Real

namespace MathProblem

def p₁ := ∃ x : ℝ, x^2 + x + 1 < 0
def p₂ := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_statement : (¬p₁) ∨ (¬p₂) :=
by
  sorry

end MathProblem

end problem_statement_l215_215439


namespace inequality_proof_l215_215148

theorem inequality_proof
  (a b c d e f : ℝ)
  (h : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
sorry

end inequality_proof_l215_215148


namespace contractor_engagement_days_l215_215252

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end contractor_engagement_days_l215_215252


namespace domain_of_f_when_a_is_3_max_value_of_a_for_inequality_l215_215431

noncomputable def f (x : ℝ) (a : ℝ) := log 2 (|x + 1| + |x - 1| - a)

theorem domain_of_f_when_a_is_3 :
  { x : ℝ | x < -3/2 ∨ x > 3/2 } = { x : ℝ | f x 3 ≠ 0 } :=
by
  sorry

theorem max_value_of_a_for_inequality :
  (∀ x : ℝ, f x a ≥ 2) → a ≤ -2 :=
by
  sorry

end domain_of_f_when_a_is_3_max_value_of_a_for_inequality_l215_215431


namespace eval_polynomial_correct_l215_215130

theorem eval_polynomial_correct (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) (hy_pos : 0 < y) :
  y^3 - 3 * y^2 - 9 * y + 3 = 3 :=
sorry

end eval_polynomial_correct_l215_215130


namespace proof_solution_l215_215587

variable (U : Set ℝ) (A : Set ℝ) (C_U_A : Set ℝ)
variables (a b : ℝ)

noncomputable def proof_problem : Prop :=
  (U = Set.univ) →
  (A = {x | a ≤ x ∧ x ≤ b}) →
  (C_U_A = {x | x > 4 ∨ x < 3}) →
  A = {x | 3 ≤ x ∧ x ≤ 4} ∧ a = 3 ∧ b = 4

theorem proof_solution : proof_problem U A C_U_A a b :=
by
  intro hU hA hCUA
  have hA_eq : A = {x | 3 ≤ x ∧ x ≤ 4} :=
    by { sorry }
  have ha : a = 3 :=
    by { sorry }
  have hb : b = 4 :=
    by { sorry }
  exact ⟨hA_eq, ha, hb⟩

end proof_solution_l215_215587


namespace trapezoid_area_l215_215182

def isosceles_triangle (Δ : Type) (A B C : Δ) : Prop :=
  -- Define the property that triangle ABC is isosceles with AB = AC
  sorry

def similar_triangles (Δ₁ Δ₂ : Type) (A₁ B₁ C₁ : Δ₁) (A₂ B₂ C₂ : Δ₂) : Prop :=
  -- Define the property that triangles Δ₁ and Δ₂ are similar
  sorry

def area (Δ : Type) (A B C : Δ) : ℝ :=
  -- Define the area of a triangle Δ with vertices A, B, and C
  sorry

theorem trapezoid_area
  (Δ : Type)
  {A B C D E : Δ}
  (ABC_is_isosceles : isosceles_triangle Δ A B C)
  (all_similar : ∀ (Δ₁ Δ₂ : Type) (A₁ B₁ C₁ : Δ₁) (A₂ B₂ C₂ : Δ₂), 
    similar_triangles Δ₁ Δ₂ A₁ B₁ C₁ A₂ B₂ C₂ → (area Δ₁ A₁ B₁ C₁ = 1 → area Δ₂ A₂ B₂ C₂ = 1))
  (smallest_triangles_area : area Δ A B C = 50)
  (area_ADE : area Δ A D E = 5) :
  area Δ D B C + area Δ C E B = 45 := 
sorry

end trapezoid_area_l215_215182


namespace factorize_expression_l215_215954

theorem factorize_expression (a : ℝ) : 
  a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end factorize_expression_l215_215954


namespace num_ways_to_distribute_balls_l215_215316

-- Define the conditions
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- The statement to prove
theorem num_ways_to_distribute_balls : num_boxes ^ num_balls = 729 :=
by {
  -- Proof steps would go here
  sorry
}

end num_ways_to_distribute_balls_l215_215316


namespace problem_statement_l215_215364

variable {x y : ℤ}

def is_multiple_of_5 (n : ℤ) : Prop := ∃ m : ℤ, n = 5 * m
def is_multiple_of_10 (n : ℤ) : Prop := ∃ m : ℤ, n = 10 * m

theorem problem_statement (hx : is_multiple_of_5 x) (hy : is_multiple_of_10 y) :
  (is_multiple_of_5 (x + y)) ∧ (x + y ≥ 15) :=
sorry

end problem_statement_l215_215364


namespace num_of_chairs_per_row_l215_215819

theorem num_of_chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (chairs_per_row : ℕ)
  (h1 : total_chairs = 432)
  (h2 : num_rows = 27) :
  total_chairs = num_rows * chairs_per_row ↔ chairs_per_row = 16 :=
by
  sorry

end num_of_chairs_per_row_l215_215819


namespace find_a_2b_3c_l215_215461

noncomputable def a : ℝ := 28
noncomputable def b : ℝ := 32
noncomputable def c : ℝ := -3

def ineq_condition (x : ℝ) : Prop := (x < -3) ∨ (abs (x - 30) ≤ 2)

theorem find_a_2b_3c (a b c : ℝ) (h₁ : a < b)
  (h₂ : ∀ x : ℝ, (x < -3 ∨ abs (x - 30) ≤ 2) ↔ ((x - a)*(x - b)/(x - c) ≤ 0)) :
  a + 2 * b + 3 * c = 83 :=
by
  sorry

end find_a_2b_3c_l215_215461


namespace velocity_at_t4_acceleration_is_constant_l215_215400

noncomputable def s (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 8

def v (t : ℝ) : ℝ := 6 * t - 3

def a : ℝ := 6

theorem velocity_at_t4 : v 4 = 21 := by 
  sorry

theorem acceleration_is_constant : a = 6 := by 
  sorry

end velocity_at_t4_acceleration_is_constant_l215_215400


namespace three_not_divide_thirtyone_l215_215605

theorem three_not_divide_thirtyone : ¬ ∃ q : ℤ, 31 = 3 * q := sorry

end three_not_divide_thirtyone_l215_215605


namespace garden_contains_53_33_percent_tulips_l215_215633

theorem garden_contains_53_33_percent_tulips :
  (∃ (flowers : ℕ) (yellow tulips flowers_in_garden : ℕ) (yellow_flowers blue_flowers yellow_tulips blue_tulips : ℕ),
    flowers_in_garden = yellow_flowers + blue_flowers ∧
    yellow_flowers = 4 * flowers / 5 ∧
    blue_flowers = 1 * flowers / 5 ∧
    yellow_tulips = yellow_flowers / 2 ∧
    blue_tulips = 2 * blue_flowers / 3 ∧
    (yellow_tulips + blue_tulips) = 8 * flowers / 15) →
    0.5333 ∈ ([46.67, 53.33, 60, 75, 80] : List ℝ) := sorry

end garden_contains_53_33_percent_tulips_l215_215633


namespace binary_to_decimal_l215_215405

theorem binary_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5) :=
by
  sorry

end binary_to_decimal_l215_215405


namespace product_of_x_and_y_l215_215579

theorem product_of_x_and_y :
  ∀ (x y : ℝ), (∀ p : ℝ × ℝ, (p = (x, 6) ∨ p = (10, y)) → p.2 = (1 / 2) * p.1) → x * y = 60 :=
by
  intros x y h
  have hx : 6 = (1 / 2) * x := by exact h (x, 6) (Or.inl rfl)
  have hy : y = (1 / 2) * 10 := by exact h (10, y) (Or.inr rfl)
  sorry

end product_of_x_and_y_l215_215579


namespace sum_infinite_geometric_series_l215_215952

theorem sum_infinite_geometric_series :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  (a / (1 - r) = (3 : ℚ) / 8) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  sorry

end sum_infinite_geometric_series_l215_215952


namespace tablet_value_is_2100_compensation_for_m_days_l215_215276

-- Define the given conditions
def monthly_compensation: ℕ := 30
def monthly_tablet_value (x: ℕ) (cash: ℕ): ℕ := x + cash

def daily_compensation (days: ℕ) (x: ℕ) (cash: ℕ): ℕ :=
  days * (x / monthly_compensation + cash / monthly_compensation)

def received_compensation (tablet_value: ℕ) (cash: ℕ): ℕ :=
  tablet_value + cash

-- The proofs we need:
-- Proof that the tablet value is 2100 yuan
theorem tablet_value_is_2100:
  ∀ (x: ℕ) (cash_1 cash_2: ℕ), 
  ((20 * (x / monthly_compensation + 1500 / monthly_compensation)) = (x + 300)) → 
  x = 2100 := sorry

-- Proof that compensation for m days is 120m yuan
theorem compensation_for_m_days (m: ℕ):
  ∀ (x: ℕ), 
  ((x + 1500) / monthly_compensation) = 120 → 
  x = 2100 → 
  m * 120 = 120 * m := sorry

end tablet_value_is_2100_compensation_for_m_days_l215_215276


namespace randy_initial_amount_l215_215730

theorem randy_initial_amount (spend_per_trip: ℤ) (trips_per_month: ℤ) (dollars_left_after_year: ℤ) (total_month_months: ℤ := 12):
  (spend_per_trip = 2 ∧ trips_per_month = 4 ∧ dollars_left_after_year = 104) → spend_per_trip * trips_per_month * total_month_months + dollars_left_after_year = 200 := 
by
  sorry

end randy_initial_amount_l215_215730


namespace find_divisor_l215_215243

-- Define the problem conditions as variables
variables (remainder quotient dividend : ℕ)
variables (D : ℕ)

-- State the problem formally
theorem find_divisor (h0 : remainder = 19) 
                     (h1 : quotient = 61) 
                     (h2 : dividend = 507) 
                     (h3 : dividend = (D * quotient) + remainder) : 
  D = 8 := 
by 
  -- Use the Lean theorem prover to demonstrate the condition
  have h4 : 507 = (D * 61) + 19, from h3,
  -- Simplify and solve for D
  sorry

end find_divisor_l215_215243


namespace weight_of_currants_l215_215777

noncomputable def packing_density : ℝ := 0.74
noncomputable def water_density : ℝ := 1000
noncomputable def bucket_volume : ℝ := 0.01

theorem weight_of_currants :
  (water_density * (packing_density * bucket_volume)) = 7.4 :=
by
  sorry

end weight_of_currants_l215_215777


namespace arithmetic_sequence_fifth_term_l215_215298

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (a6 : a 6 = -3) 
  (S6 : S 6 = 12)
  (h_sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 1 - a 0)) / 2)
  : a 5 = -1 :=
sorry

end arithmetic_sequence_fifth_term_l215_215298


namespace diego_can_carry_home_l215_215411

theorem diego_can_carry_home (T W G O A : ℕ) (hT : T = 20) (hW : W = 1) (hG : G = 1) (hO : O = 1) : A = T - (W + G + O) → A = 17 := by
  sorry

end diego_can_carry_home_l215_215411


namespace remainder_M_mod_32_l215_215033

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215033


namespace caltech_equilateral_triangles_l215_215934

theorem caltech_equilateral_triangles (n : ℕ) (h : n = 900) :
  let total_triangles := (n * (n - 1) / 2) * 2
  let overcounted_triangles := n / 3
  total_triangles - overcounted_triangles = 808800 :=
by
  sorry

end caltech_equilateral_triangles_l215_215934


namespace alice_twice_bob_in_some_years_l215_215122

def alice_age (B : ℕ) : ℕ := B + 10
def future_age_condition (A : ℕ) : Prop := A + 5 = 19
def twice_as_old_condition (A B x : ℕ) : Prop := A + x = 2 * (B + x)

theorem alice_twice_bob_in_some_years :
  ∃ x, ∀ A B,
  alice_age B = A →
  future_age_condition A →
  twice_as_old_condition A B x := by
  sorry

end alice_twice_bob_in_some_years_l215_215122


namespace inequality_solution_set_l215_215751

theorem inequality_solution_set :
  {x : ℝ | (3 * x + 1) / (1 - 2 * x) ≥ 0} = {x : ℝ | -1 / 3 ≤ x ∧ x < 1 / 2} := by
  sorry

end inequality_solution_set_l215_215751


namespace quadrilateral_angle_W_l215_215705

theorem quadrilateral_angle_W (W X Y Z : ℝ) 
  (h₁ : W = 3 * X) 
  (h₂ : W = 4 * Y) 
  (h₃ : W = 6 * Z) 
  (sum_angles : W + X + Y + Z = 360) : 
  W = 1440 / 7 := by
sorry

end quadrilateral_angle_W_l215_215705


namespace equal_roots_implies_c_value_l215_215329

theorem equal_roots_implies_c_value (c : ℝ) 
  (h : ∃ x : ℝ, (x^2 + 6 * x - c = 0) ∧ (2 * x + 6 = 0)) :
  c = -9 :=
sorry

end equal_roots_implies_c_value_l215_215329


namespace point_translation_l215_215350

theorem point_translation :
  ∃ (x y : ℤ), x = -1 ∧ y = -2 ↔ 
  ∃ (x₀ y₀ : ℤ), 
    x₀ = -3 ∧ y₀ = 2 ∧ 
    x = x₀ + 2 ∧ 
    y = y₀ - 4 := by
  sorry

end point_translation_l215_215350


namespace solve_y_l215_215358

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end solve_y_l215_215358


namespace min_value_expression_l215_215304

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 5) :
  ∃ (min_val : ℝ), min_val = ( (x + 1) * (2 * y + 1) ) / (Real.sqrt (x * y)) ∧ min_val = 4 * Real.sqrt 3 :=
by
  sorry

end min_value_expression_l215_215304


namespace dress_design_count_l215_215514

-- Definitions of the given conditions
def number_of_colors : Nat := 4
def number_of_patterns : Nat := 5

-- Statement to prove the total number of unique dress designs
theorem dress_design_count :
  number_of_colors * number_of_patterns = 20 := by
  sorry

end dress_design_count_l215_215514


namespace range_of_a_l215_215425

theorem range_of_a (a x : ℝ) (h_p : a - 4 < x ∧ x < a + 4) (h_q : (x - 2) * (x - 3) > 0) :
  a ≤ -2 ∨ a ≥ 7 :=
sorry

end range_of_a_l215_215425


namespace probability_P_plus_S_mod_7_correct_l215_215499

noncomputable def probability_P_plus_S_mod_7 : ℚ :=
  let n := 60
  let total_ways := (n * (n - 1)) / 2
  let num_special_pairs := total_ways - ((52 * 51) / 2)
  num_special_pairs / total_ways

theorem probability_P_plus_S_mod_7_correct :
  probability_P_plus_S_mod_7 = 148 / 590 :=
by
  rw [probability_P_plus_S_mod_7]
  sorry

end probability_P_plus_S_mod_7_correct_l215_215499


namespace algebra_square_formula_l215_215209

theorem algebra_square_formula (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
sorry

end algebra_square_formula_l215_215209


namespace polygon_sides_sum_720_l215_215088

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_sum_720_l215_215088


namespace probability_two_green_marbles_l215_215114

open Classical

section
variable (num_red num_green num_white num_blue : ℕ)
variable (total_marbles : ℕ := num_red + num_green + num_white + num_blue)

def probability_green_two_draws (num_green : ℕ) (total_marbles : ℕ) : ℚ :=
  (num_green / total_marbles : ℚ) * ((num_green - 1) / (total_marbles - 1))

theorem probability_two_green_marbles :
  probability_green_two_draws 4 (3 + 4 + 8 + 5) = 3 / 95 := by
  sorry
end

end probability_two_green_marbles_l215_215114


namespace product_of_odd_primes_mod_32_l215_215042

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215042


namespace find_y_l215_215325

theorem find_y (x y : ℝ) (h1 : x = 100) (h2 : x^3 * y - 3 * x^2 * y + 3 * x * y = 3000000) : 
  y = 3000000 / (100^3 - 3 * 100^2 + 3 * 100 * 1) :=
by sorry

end find_y_l215_215325


namespace consecutive_odd_integers_l215_215768

theorem consecutive_odd_integers (x : ℤ) (h : x + 4 = 15) : 3 * x - 2 * (x + 4) = 3 :=
by
  sorry

end consecutive_odd_integers_l215_215768


namespace student_count_incorrect_l215_215776

theorem student_count_incorrect :
  ∀ k : ℕ, 2012 ≠ 18 + 17 * k :=
by
  intro k
  sorry

end student_count_incorrect_l215_215776


namespace product_mod_32_is_15_l215_215014

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215014


namespace minimum_x_for_g_maximum_l215_215657

theorem minimum_x_for_g_maximum :
  ∃ x > 0, ∀ k m: ℤ, (x = 1440 * k + 360 ∧ x = 2520 * m + 630) -> x = 7560 :=
by
  sorry

end minimum_x_for_g_maximum_l215_215657


namespace chess_tournament_total_players_l215_215185

theorem chess_tournament_total_players :
  ∃ (n: ℕ), 
    (∀ (players: ℕ) (points: ℕ -> ℕ), 
      (players = n + 15) ∧
      (∀ p, points p = points p / 2 + points p / 2) ∧
      (∀ i < 15, ∀ j < 15, points i = points j / 2) → 
      players = 36) :=
by
  sorry

end chess_tournament_total_players_l215_215185


namespace bases_with_final_digit_one_in_360_l215_215134

theorem bases_with_final_digit_one_in_360 (b : ℕ) (h : 2 ≤ b ∧ b ≤ 9) : ¬(b ∣ 359) :=
by
  sorry

end bases_with_final_digit_one_in_360_l215_215134


namespace find_number_l215_215362

theorem find_number (x : ℕ) (h : x - 263 + 419 = 725) : x = 569 :=
sorry

end find_number_l215_215362


namespace birds_on_fence_l215_215880

def number_of_birds_on_fence : ℕ := 20

theorem birds_on_fence (x : ℕ) (h : 2 * x + 10 = 50) : x = number_of_birds_on_fence :=
by
  sorry

end birds_on_fence_l215_215880


namespace min_xy_min_x_plus_y_l215_215969

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : xy ≥ 4 := sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : x + y ≥ 9 / 2 := sorry

end min_xy_min_x_plus_y_l215_215969


namespace part_a_l215_215984

noncomputable def M (n k : ℕ) : ℕ :=
  Nat.lcm_list (List.range' (n - k + 1) k)

noncomputable def f (n : ℕ) : ℕ :=
  Nat.find_greatest (λ k, (k ≤ n) ∧ ∀ i, 1 ≤ i → i < k → M n i < M n (i+1)) n

theorem part_a (n : ℕ) (hn.pos: 0 < n) : f n < 3 * Nat.sqrt n :=
  sorry

end part_a_l215_215984


namespace lottery_buying_100_may_not_win_l215_215248

-- Definitions corresponding to conditions in a)
def total_tickets : ℕ := 100000
def win_probability : ℝ := 0.01
def tickets_bought : ℕ := 100

-- The main statement proving the question in c)
theorem lottery_buying_100_may_not_win : 
  (∃ (n : ℕ), (n = tickets_bought) ∧ (prob_eventually_does_not_win n = (tickets_bought ≤ total_tickets * win_probability))) :=
by 
  sorry

end lottery_buying_100_may_not_win_l215_215248


namespace find_m_l215_215742

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m + 1)

theorem find_m (m : ℝ) :
  (∀ x > 0, f m x < 0) → m = -2 := by
  sorry

end find_m_l215_215742


namespace remainder_M_mod_32_l215_215036

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215036


namespace polar_coordinates_standard_representation_l215_215576

theorem polar_coordinates_standard_representation :
  ∀ (r θ : ℝ), (r, θ) = (-4, 5 * Real.pi / 6) → (∃ (r' θ' : ℝ), r' > 0 ∧ (r', θ') = (4, 11 * Real.pi / 6))
:= by
  sorry

end polar_coordinates_standard_representation_l215_215576


namespace greatest_difference_areas_l215_215755

theorem greatest_difference_areas (l w l' w' : ℕ) (h₁ : 2*l + 2*w = 120) (h₂ : 2*l' + 2*w' = 120) : 
  l * w ≤ 900 ∧ (l = 30 → w = 30) ∧ l' * w' ≤ 900 ∧ (l' = 30 → w' = 30)  → 
  ∃ (A₁ A₂ : ℕ), (A₁ = l * w ∧ A₂ = l' * w') ∧ (841 = l * w - l' * w') := 
sorry

end greatest_difference_areas_l215_215755


namespace product_of_odd_primes_mod_32_l215_215021

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215021


namespace polynomial_condition_form_l215_215417

theorem polynomial_condition_form (P : Polynomial ℝ) :
  (∀ a b c : ℝ, ab + bc + ca = 0 → P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)) →
  ∃ α β : ℝ, P = α * Polynomial.X ^ 4 + β * Polynomial.X ^ 2 :=
sorry

end polynomial_condition_form_l215_215417


namespace ones_digit_sum_l215_215653

theorem ones_digit_sum : 
  (1 + 2 ^ 2023 + 3 ^ 2023 + 4 ^ 2023 + 5 : ℕ) % 10 = 5 := 
by 
  sorry

end ones_digit_sum_l215_215653


namespace fred_initial_cards_l215_215966

variables {n : ℕ}

theorem fred_initial_cards (h : n - 22 = 18) : n = 40 :=
by {
  sorry
}

end fred_initial_cards_l215_215966


namespace calculate_DA_l215_215187

open Real

-- Definitions based on conditions
def AU := 90
def AN := 180
def UB := 270
def AB := AU + UB
def ratio := 3 / 4

-- Statement of the problem in Lean 
theorem calculate_DA :
  ∃ (p q : ℕ), (q ≠ 0) ∧ (∀ p' q' : ℕ, ¬ (q = p'^2 * q')) ∧ DA = p * sqrt q ∧ p + q = result :=
  sorry

end calculate_DA_l215_215187


namespace vasim_share_l215_215927

theorem vasim_share (x : ℕ) (F V R : ℕ) (h1 : F = 3 * x) (h2 : V = 5 * x) (h3 : R = 11 * x) (h4 : R - F = 2400) : V = 1500 :=
by sorry

end vasim_share_l215_215927


namespace min_ratio_l215_215464

theorem min_ratio (x y : ℕ) 
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (mean : (x + y) = 110) :
  x / y = 1 / 9 :=
  sorry

end min_ratio_l215_215464


namespace second_box_probability_nth_box_probability_l215_215495

noncomputable def P_A1 : ℚ := 2 / 3
noncomputable def P_A2 : ℚ := 5 / 9
noncomputable def P_An (n : ℕ) : ℚ :=
  1 / 2 * (1 / 3) ^ n + 1 / 2

theorem second_box_probability :
  P_A2 = 5 / 9 := by
  sorry

theorem nth_box_probability (n : ℕ) :
  P_An n = 1 / 2 * (1 / 3) ^ n + 1 / 2 := by
  sorry

end second_box_probability_nth_box_probability_l215_215495


namespace xyz_value_l215_215162

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) : 
  x * y * z = 10 :=
by
  sorry

end xyz_value_l215_215162


namespace find_central_angle_l215_215303

variable (L : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions
def arc_length_condition : Prop := L = 200
def radius_condition : Prop := r = 2
def arc_length_formula : Prop := L = r * α

-- Theorem statement
theorem find_central_angle 
  (hL : arc_length_condition L) 
  (hr : radius_condition r) 
  (hf : arc_length_formula L r α) : 
  α = 100 := by
  -- Proof goes here
  sorry

end find_central_angle_l215_215303


namespace sequence_sum_l215_215797

theorem sequence_sum (x y : ℕ) 
  (r : ℚ) 
  (h1 : 4 * r = 1) 
  (h2 : x = 256 * r)
  (h3 : y = x * r): 
  x + y = 80 := 
by 
  sorry

end sequence_sum_l215_215797


namespace coins_amount_correct_l215_215955

-- Definitions based on the conditions
def cost_of_flour : ℕ := 5
def cost_of_cake_stand : ℕ := 28
def amount_given_in_bills : ℕ := 20 + 20
def change_received : ℕ := 10

-- Total cost of items
def total_cost : ℕ := cost_of_flour + cost_of_cake_stand

-- Total money given
def total_money_given : ℕ := total_cost + change_received

-- Amount given in loose coins
def loose_coins_given : ℕ := total_money_given - amount_given_in_bills

-- Proposition statement
theorem coins_amount_correct : loose_coins_given = 3 := by
  sorry

end coins_amount_correct_l215_215955


namespace find_f_l215_215215

theorem find_f
  (d e f : ℝ)
  (vertex_x vertex_y : ℝ)
  (p_x p_y : ℝ)
  (vertex_cond : vertex_x = 3 ∧ vertex_y = -1)
  (point_cond : p_x = 5 ∧ p_y = 1)
  (equation : ∀ y : ℝ, ∃ x : ℝ, x = d * y^2 + e * y + f) :
  f = 7 / 2 :=
by
  sorry

end find_f_l215_215215


namespace age_ratio_correct_l215_215746

noncomputable def RahulDeepakAgeRatio : Prop :=
  let R := 20
  let D := 8
  R / D = 5 / 2

theorem age_ratio_correct (R D : ℕ) (h1 : R + 6 = 26) (h2 : D = 8) : RahulDeepakAgeRatio :=
by
  -- Proof omitted
  sorry

end age_ratio_correct_l215_215746


namespace probability_of_winning_l215_215612

theorem probability_of_winning (P_lose P_tie P_win : ℚ) (h_lose : P_lose = 5/11) (h_tie : P_tie = 1/11)
  (h_total : P_lose + P_win + P_tie = 1) : P_win = 5/11 := 
by
  sorry

end probability_of_winning_l215_215612


namespace tan_value_l215_215670

open Real

theorem tan_value (α : ℝ) (h : sin (5 * π / 6 - α) = sqrt 3 * cos (α + π / 6)) : 
  tan (α + π / 6) = sqrt 3 := 
  sorry

end tan_value_l215_215670


namespace florist_initial_roses_l215_215113

theorem florist_initial_roses : 
  ∀ (R : ℕ), (R - 16 + 19 = 40) → (R = 37) :=
by
  intro R
  intro h
  sorry

end florist_initial_roses_l215_215113


namespace polygon_sides_l215_215084

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end polygon_sides_l215_215084


namespace hall_length_width_difference_l215_215494

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L)
  (h2 : L * W = 450) :
  L - W = 15 :=
sorry

end hall_length_width_difference_l215_215494


namespace divide_money_equally_l215_215244

-- Length of the road built by companies A, B, and total length of the road
def length_A : ℕ := 6
def length_B : ℕ := 10
def total_length : ℕ := 16

-- Money contributed by company C
def money_C : ℕ := 16 * 10^6

-- The equal contribution each company should finance
def equal_contribution := total_length / 3

-- Deviations from the expected length for firms A and B
def deviation_A := length_A - (total_length / 3)
def deviation_B := length_B - (total_length / 3)

-- The ratio based on the deviations to divide the money
def ratio_A := deviation_A * (total_length / (deviation_A + deviation_B))
def ratio_B := deviation_B * (total_length / (deviation_A + deviation_B))

-- The amount of money firms A and B should receive, respectively
def money_A := money_C * ratio_A / total_length
def money_B := money_C * ratio_B / total_length

-- Theorem statement
theorem divide_money_equally : money_A = 2 * 10^6 ∧ money_B = 14 * 10^6 :=
by 
  sorry

end divide_money_equally_l215_215244


namespace fourth_grade_planted_89_l215_215285

-- Define the number of trees planted by the fifth grade
def fifth_grade_trees : Nat := 114

-- Define the condition that the fifth grade planted twice as many trees as the third grade
def third_grade_trees : Nat := fifth_grade_trees / 2

-- Define the condition that the fourth grade planted 32 more trees than the third grade
def fourth_grade_trees : Nat := third_grade_trees + 32

-- Theorem to prove the number of trees planted by the fourth grade is 89
theorem fourth_grade_planted_89 : fourth_grade_trees = 89 := by
  sorry

end fourth_grade_planted_89_l215_215285


namespace find_multiple_of_benjy_peaches_l215_215593

theorem find_multiple_of_benjy_peaches
(martine_peaches gabrielle_peaches : ℕ)
(benjy_peaches : ℕ)
(m : ℕ)
(h1 : martine_peaches = 16)
(h2 : gabrielle_peaches = 15)
(h3 : benjy_peaches = gabrielle_peaches / 3)
(h4 : martine_peaches = m * benjy_peaches + 6) :
m = 2 := by
sorry

end find_multiple_of_benjy_peaches_l215_215593


namespace find_m_l215_215884

def h (x m : ℝ) := x^2 - 3 * x + m
def k (x m : ℝ) := x^2 - 3 * x + 5 * m

theorem find_m (m : ℝ) (h_def : ∀ x, h x m = x^2 - 3 * x + m) (k_def : ∀ x, k x m = x^2 - 3 * x + 5 * m) (key_eq : 3 * h 5 m = 2 * k 5 m) :
  m = 10 / 7 :=
by
  sorry

end find_m_l215_215884


namespace value_of_f_neg2_l215_215294

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - log 3 (x^2 - 3*x + 5) else -(2^(-x) - log 3 ((-x)^2 - 3*(-x) + 5))

theorem value_of_f_neg2 :
  f (-2) = -3 :=
by
  -- Conditions:
  -- f is defined on ℝ
  -- f is an odd function
  -- For x > 0, f(x) = 2^x - log_3(x^2 - 3*x + 5)

  sorry

end value_of_f_neg2_l215_215294


namespace min_vitamins_sold_l215_215348

theorem min_vitamins_sold (n : ℕ) (h1 : n % 11 = 0) (h2 : n % 23 = 0) (h3 : n % 37 = 0) : n = 9361 :=
by
  sorry

end min_vitamins_sold_l215_215348


namespace probability_P_plus_S_is_one_less_than_multiple_of_seven_l215_215501

theorem probability_P_plus_S_is_one_less_than_multiple_of_seven :
  ∀ (a b : ℕ), a ∈ finset.range(1, 61) → b ∈ finset.range(1, 61) → a ≠ b →
  let S := a + b in
  let P := a * b in
  (nat.gcd ((P + S + 1), 7) = 1) →
  (finset.filter (λ (a b : ℕ), (a+1) ∣ 7 ∨ (b+1) ∣ 7) (finset.range(1, 61)).product (finset.range(1, 61)).card) / 1770 = 74 / 295 :=
begin
  sorry
end

end probability_P_plus_S_is_one_less_than_multiple_of_seven_l215_215501


namespace sum_of_first_6_terms_l215_215143

-- Definitions based on given conditions
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + d * (n - 1)

-- The conditions provided in the problem
def condition_1 (a1 d : ℤ) : Prop := arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 4
def condition_2 (a1 d : ℤ) : Prop := arithmetic_sequence a1 d 3 + arithmetic_sequence a1 d 5 = 10

-- The sum of the first 6 terms of the arithmetic sequence
def sum_first_6_terms (a1 d : ℤ) : ℤ := 6 * a1 + 15 * d

-- The theorem to prove
theorem sum_of_first_6_terms (a1 d : ℤ) 
  (h1 : condition_1 a1 d)
  (h2 : condition_2 a1 d) :
  sum_first_6_terms a1 d = 21 := sorry

end sum_of_first_6_terms_l215_215143


namespace shipping_cost_per_unit_l215_215513

noncomputable def fixed_monthly_costs : ℝ := 16500
noncomputable def production_cost_per_component : ℝ := 80
noncomputable def production_quantity : ℝ := 150
noncomputable def selling_price_per_component : ℝ := 193.33

theorem shipping_cost_per_unit :
  ∀ (S : ℝ), (production_quantity * production_cost_per_component + production_quantity * S + fixed_monthly_costs) ≤ (production_quantity * selling_price_per_component) → S ≤ 3.33 :=
by
  intro S
  sorry

end shipping_cost_per_unit_l215_215513


namespace total_number_of_coins_is_324_l215_215095

noncomputable def total_coins (total_sum : ℕ) (coins_20p : ℕ) (coins_25p_value : ℕ) : ℕ :=
    coins_20p + (coins_25p_value / 25)

theorem total_number_of_coins_is_324 (h_sum: 7100 = 71 * 100) (h_coins_20p: 200 * 20 = 4000) :
  total_coins 7100 200 3100 = 324 := by
  sorry

end total_number_of_coins_is_324_l215_215095


namespace DanGreenMarbles_l215_215128

theorem DanGreenMarbles : 
  ∀ (initial_green marbles_taken remaining_green : ℕ), 
  initial_green = 32 →
  marbles_taken = 23 →
  remaining_green = initial_green - marbles_taken →
  remaining_green = 9 :=
by sorry

end DanGreenMarbles_l215_215128


namespace total_ads_clicked_l215_215504

theorem total_ads_clicked (a1 a2 a3 a4 : ℕ) (clicked_ads : ℕ) :
  a1 = 12 →
  a2 = 2 * a1 →
  a3 = a2 + 24 →
  a4 = (3 * a2) / 4 →
  clicked_ads = (2 * (a1 + a2 + a3 + a4)) / 3 →
  clicked_ads = 68 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end total_ads_clicked_l215_215504


namespace product_of_solutions_product_of_all_t_l215_215804

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l215_215804


namespace isosceles_triangle_l215_215845

def shape_of_triangle (A B C : Real) (h : 2 * Real.sin A * Real.cos B = Real.sin C) : Prop :=
  A = B

theorem isosceles_triangle {A B C : Real} (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  shape_of_triangle A B C h := 
  sorry

end isosceles_triangle_l215_215845


namespace simplify_expression_l215_215380

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (18 * x^3) * (4 * x^2) * (1 / (2 * x)^3) = 9 * x^2 :=
by
  sorry

end simplify_expression_l215_215380


namespace contribution_per_student_l215_215994

theorem contribution_per_student (total_contribution : ℝ) (class_funds : ℝ) (num_students : ℕ) 
(h1 : total_contribution = 90) (h2 : class_funds = 14) (h3 : num_students = 19) : 
  (total_contribution - class_funds) / num_students = 4 :=
by
  sorry

end contribution_per_student_l215_215994


namespace base2_to_base4_conversion_l215_215127

theorem base2_to_base4_conversion :
  (2 ^ 8 + 2 ^ 6 + 2 ^ 4 + 2 ^ 3 + 2 ^ 2 + 1) = (1 * 4^3 + 1 * 4^2 + 3 * 4^1 + 1 * 4^0) :=
by 
  sorry

end base2_to_base4_conversion_l215_215127


namespace A_holds_15_l215_215096

def cards : List (ℕ × ℕ) := [(1, 3), (1, 5), (3, 5)]

variables (A_card B_card C_card : ℕ × ℕ)

-- Conditions from the problem
def C_not_35 : Prop := C_card ≠ (3, 5)
def A_says_not_3 (A_card B_card : ℕ × ℕ) : Prop := ¬(A_card.1 = 3 ∧ B_card.1 = 3 ∨ A_card.2 = 3 ∧ B_card.2 = 3)
def B_says_not_1 (B_card C_card : ℕ × ℕ) : Prop := ¬(B_card.1 = 1 ∧ C_card.1 = 1 ∨ B_card.2 = 1 ∧ C_card.2 = 1)

-- Question to prove
theorem A_holds_15 : 
  ∃ (A_card B_card C_card : ℕ × ℕ),
    A_card ∈ cards ∧ B_card ∈ cards ∧ C_card ∈ cards ∧
    A_card ≠ B_card ∧ B_card ≠ C_card ∧ A_card ≠ C_card ∧
    C_not_35 C_card ∧
    A_says_not_3 A_card B_card ∧
    B_says_not_1 B_card C_card ->
    A_card = (1, 5) :=
sorry

end A_holds_15_l215_215096


namespace contractor_engaged_days_l215_215258

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end contractor_engaged_days_l215_215258


namespace age_ratio_in_4_years_l215_215536

variable {p k x : ℕ}

theorem age_ratio_in_4_years (h₁ : p - 8 = 2 * (k - 8)) (h₂ : p - 14 = 3 * (k - 14)) : x = 4 :=
by
  sorry

end age_ratio_in_4_years_l215_215536


namespace grasshopper_jump_is_31_l215_215885

def frog_jump : ℕ := 35
def total_jump : ℕ := 66
def grasshopper_jump := total_jump - frog_jump

theorem grasshopper_jump_is_31 : grasshopper_jump = 31 := 
by
  unfold grasshopper_jump
  sorry

end grasshopper_jump_is_31_l215_215885


namespace period_of_3sin_minus_4cos_l215_215759

theorem period_of_3sin_minus_4cos (x : ℝ) : 
  ∃ T : ℝ, T = 2 * Real.pi ∧ (∀ x, 3 * Real.sin x - 4 * Real.cos x = 3 * Real.sin (x + T) - 4 * Real.cos (x + T)) :=
sorry

end period_of_3sin_minus_4cos_l215_215759


namespace compute_fraction_power_l215_215793

theorem compute_fraction_power (a b : ℕ) (ha : a = 123456) (hb : b = 41152) : (a ^ 5 / b ^ 5) = 243 := by
  sorry

end compute_fraction_power_l215_215793


namespace find_m_plus_n_l215_215982

theorem find_m_plus_n
  (m n : ℝ)
  (l1 : ∀ x y : ℝ, 2 * x + m * y + 2 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + y - 1 = 0)
  (l3 : ∀ x y : ℝ, x + n * y + 1 = 0)
  (parallel_l1_l2 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) → (2 * x + y - 1 = 0))
  (perpendicular_l1_l3 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) ∧ (x + n * y + 1 = 0) → true) :
  m + n = -1 :=
by
  sorry

end find_m_plus_n_l215_215982


namespace possible_values_of_n_l215_215198

theorem possible_values_of_n (n : ℕ) (h1 : 0 < n)
  (h2 : 12 * n^3 = n^4 + 11 * n^2) :
  n = 1 ∨ n = 11 :=
sorry

end possible_values_of_n_l215_215198


namespace benny_turnips_l215_215203

theorem benny_turnips (M B : ℕ) (h1 : M = 139) (h2 : M = B + 26) : B = 113 := 
by 
  sorry

end benny_turnips_l215_215203


namespace diego_apples_weight_l215_215410

-- Definitions based on conditions
def bookbag_capacity : ℕ := 20
def weight_watermelon : ℕ := 1
def weight_grapes : ℕ := 1
def weight_oranges : ℕ := 1

-- Lean statement to check
theorem diego_apples_weight : 
  bookbag_capacity - (weight_watermelon + weight_grapes + weight_oranges) = 17 :=
by
  sorry

end diego_apples_weight_l215_215410


namespace find_length_of_street_l215_215610

-- Definitions based on conditions
def area_street (L : ℝ) : ℝ := L^2
def area_forest (L : ℝ) : ℝ := 3 * (area_street L)
def num_trees (L : ℝ) : ℝ := 4 * (area_forest L)

-- Statement to prove
theorem find_length_of_street (L : ℝ) (h : num_trees L = 120000) : L = 100 := by
  sorry

end find_length_of_street_l215_215610


namespace disjoint_subsets_exist_l215_215903

theorem disjoint_subsets_exist (n : ℕ) (h : 0 < n) 
  (A : Fin (n + 1) → Set (Fin n)) (hA : ∀ i : Fin (n + 1), A i ≠ ∅) :
  ∃ (I J : Finset (Fin (n + 1))), I ≠ ∅ ∧ J ≠ ∅ ∧ Disjoint I J ∧ 
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) :=
sorry

end disjoint_subsets_exist_l215_215903


namespace pamphlet_cost_l215_215469

theorem pamphlet_cost (p : ℝ) 
  (h1 : 9 * p < 10)
  (h2 : 10 * p > 11) : p = 1.11 :=
sorry

end pamphlet_cost_l215_215469


namespace find_x_l215_215840

theorem find_x (x y z : ℝ) (h1 : x ≠ 0) 
  (h2 : x / 3 = z + 2 * y ^ 2) 
  (h3 : x / 6 = 3 * z - y) : 
  x = 168 :=
by
  sorry

end find_x_l215_215840


namespace minimum_m_l215_215428

theorem minimum_m (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 24 * m = n ^ 4) : m ≥ 54 :=
sorry

end minimum_m_l215_215428


namespace number_of_tins_per_day_for_rest_of_week_l215_215191
-- Import necessary library

-- Define conditions as Lean definitions
def d1 : ℕ := 50
def d2 : ℕ := 3 * d1
def d3 : ℕ := d2 - 50
def total_target : ℕ := 500

-- Define what we need to prove
theorem number_of_tins_per_day_for_rest_of_week :
  ∃ (dr : ℕ), d1 + d2 + d3 + 4 * dr = total_target ∧ dr = 50 :=
by
  sorry

end number_of_tins_per_day_for_rest_of_week_l215_215191


namespace point_not_in_plane_l215_215295

def is_in_plane (p0 : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  let (x0, y0, z0) := p0
  let (nx, ny, nz) := n
  let (x, y, z) := p
  (nx * (x - x0) + ny * (y - y0) + nz * (z - z0)) = 0

theorem point_not_in_plane :
  ¬ is_in_plane (1, 2, 3) (1, 1, 1) (-2, 5, 4) :=
by
  sorry

end point_not_in_plane_l215_215295


namespace solve_for_y_l215_215360

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end solve_for_y_l215_215360


namespace line_circle_intersection_l215_215080

-- Define the line and circle in Lean
def line_eq (x y : ℝ) : Prop := x + y - 6 = 0
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 2

-- Define the proof about the intersection
theorem line_circle_intersection :
  (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧
  ∀ (x1 y1 x2 y2 : ℝ), (line_eq x1 y1 ∧ circle_eq x1 y1) → (line_eq x2 y2 ∧ circle_eq x2 y2) → (x1 = x2 ∧ y1 = y2) :=
by {
  sorry
}

end line_circle_intersection_l215_215080


namespace perimeter_pentagon_ABCD_l215_215758

noncomputable def AB : ℝ := 2
noncomputable def BC : ℝ := Real.sqrt 8
noncomputable def CD : ℝ := Real.sqrt 18
noncomputable def DE : ℝ := Real.sqrt 32
noncomputable def AE : ℝ := Real.sqrt 62

theorem perimeter_pentagon_ABCD : 
  AB + BC + CD + DE + AE = 2 + 9 * Real.sqrt 2 + Real.sqrt 62 := by
  -- Note: The proof has been skipped as per instruction.
  sorry

end perimeter_pentagon_ABCD_l215_215758


namespace sheila_earning_per_hour_l215_215105

theorem sheila_earning_per_hour :
  (252 / ((8 * 3) + (6 * 2)) = 7) := 
by
  -- Prove that sheila earns $7 per hour
  
  sorry

end sheila_earning_per_hour_l215_215105


namespace find_function_l215_215196

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 2 = x + y + 1) : 
  ∀ x : ℝ, f x = x + 2 := sorry

end find_function_l215_215196


namespace range_of_expression_l215_215163

theorem range_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) : 
  ∃ (z : Set ℝ), z = Set.Icc (2 / 3) 4 ∧ (4*x^2 + 4*y^2 + (1 - x - y)^2) ∈ z :=
by
  sorry

end range_of_expression_l215_215163


namespace squirrel_acorns_beginning_spring_l215_215636

-- Given conditions as definitions
def total_acorns : ℕ := 210
def months : ℕ := 3
def acorns_per_month : ℕ := total_acorns / months
def acorns_left_per_month : ℕ := 60
def acorns_taken_per_month : ℕ := acorns_per_month - acorns_left_per_month
def total_taken_acorns : ℕ := acorns_taken_per_month * months

-- Prove the final question
theorem squirrel_acorns_beginning_spring : total_taken_acorns = 30 :=
by
  unfold total_acorns months acorns_per_month acorns_left_per_month acorns_taken_per_month total_taken_acorns
  sorry

end squirrel_acorns_beginning_spring_l215_215636


namespace mr_brown_financial_outcome_l215_215596

theorem mr_brown_financial_outcome :
  ∃ (C₁ C₂ : ℝ), (2.40 = 1.25 * C₁) ∧ (2.40 = 0.75 * C₂) ∧ ((2.40 + 2.40) - (C₁ + C₂) = -0.32) :=
by
  sorry

end mr_brown_financial_outcome_l215_215596


namespace sum_of_squares_not_divisible_by_13_l215_215753

theorem sum_of_squares_not_divisible_by_13
  (x y z : ℤ)
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_xz : Int.gcd x z = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_sum : (x + y + z) % 13 = 0)
  (h_prod : (x * y * z) % 13 = 0) :
  (x^2 + y^2 + z^2) % 13 ≠ 0 := by
  sorry

end sum_of_squares_not_divisible_by_13_l215_215753


namespace triangle_properties_l215_215436

theorem triangle_properties
  (K : ℝ) (α β : ℝ)
  (hK : K = 62.4)
  (hα : α = 70 + 20/60 + 40/3600)
  (hβ : β = 36 + 50/60 + 30/3600) :
  ∃ (a b T : ℝ), 
    a = 16.55 ∧
    b = 30.0 ∧
    T = 260.36 :=
by
  sorry

end triangle_properties_l215_215436


namespace ratio_of_b_to_a_l215_215549

open Real

theorem ratio_of_b_to_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a * sin (π / 5) + b * cos (π / 5)) / (a * cos (π / 5) - b * sin (π / 5)) = tan (8 * π / 15) 
  → b / a = sqrt 3 :=
by
  intro h
  sorry

end ratio_of_b_to_a_l215_215549


namespace tank_capacity_l215_215523

theorem tank_capacity (C : ℝ) (h1 : 1/4 * C + 180 = 3/4 * C) : C = 360 :=
sorry

end tank_capacity_l215_215523


namespace arccos_neg_one_eq_pi_proof_l215_215936

noncomputable def arccos_neg_one_eq_pi : Prop :=
  arccos (-1) = π

theorem arccos_neg_one_eq_pi_proof : arccos_neg_one_eq_pi := by
  sorry

end arccos_neg_one_eq_pi_proof_l215_215936


namespace polygon_edges_l215_215635

theorem polygon_edges (n : ℕ) (h1 : (n - 2) * 180 = 4 * 360 + 180) : n = 11 :=
by {
  sorry
}

end polygon_edges_l215_215635


namespace hillary_activities_l215_215527

-- Define the conditions
def swims_every : ℕ := 6
def runs_every : ℕ := 4
def cycles_every : ℕ := 16

-- Define the theorem to prove
theorem hillary_activities : Nat.lcm (Nat.lcm swims_every runs_every) cycles_every = 48 :=
by
  -- Provide a placeholder for the proof
  sorry

end hillary_activities_l215_215527


namespace range_of_a_l215_215712

theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∃ t : ℝ, (5 * t + 1)^2 + (12 * t - 1)^2 = 2 * a * (5 * t + 1)) ↔ (0 < a ∧ a ≤ 17 / 25) := 
sorry

end range_of_a_l215_215712


namespace optimality_theorem_l215_215983

def sequence_1 := "[[[a1, a2], a3], a4]" -- 22 symbols sequence
def sequence_2 := "[[a1, a2], [a3, a4]]" -- 16 symbols sequence

def optimal_sequence := sequence_2

theorem optimality_theorem : optimal_sequence = "[[a1, a2], [a3, a4]]" :=
by
  sorry

end optimality_theorem_l215_215983


namespace remainder_M_mod_32_l215_215038

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215038


namespace prob1_prob2_l215_215393

-- Definitions and conditions for Problem 1
def U : Set ℝ := {x | x ≤ 4}
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof Problem 1: Equivalent Lean proof statement
theorem prob1 : (U \ A) ∩ B = {-3, -2, 3} := by
  sorry

-- Definitions and conditions for Problem 2
def tan_alpha_eq_3 (α : ℝ) : Prop := Real.tan α = 3

-- Proof Problem 2: Equivalent Lean proof statement
theorem prob2 (α : ℝ) (h : tan_alpha_eq_3 α) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 ∧
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = -4 / 5 := by
  sorry

end prob1_prob2_l215_215393


namespace jane_reading_period_l215_215716

theorem jane_reading_period (total_pages pages_per_day : ℕ) (H1 : pages_per_day = 5 + 10) (H2 : total_pages = 105) : 
  total_pages / pages_per_day = 7 :=
by
  sorry

end jane_reading_period_l215_215716


namespace squirrel_spring_acorns_l215_215637

/--
A squirrel had stashed 210 acorns to last him the three winter months. 
It divided the pile into thirds, one for each month, and then took some 
from each third, leaving 60 acorns for each winter month. The squirrel 
combined the ones it took to eat in the first cold month of spring. 
Prove that the number of acorns the squirrel has for the beginning of spring 
is 30.
-/
theorem squirrel_spring_acorns :
  ∀ (initial_acorns acorns_per_month remaining_acorns_per_month acorns_taken_per_month : ℕ),
    initial_acorns = 210 →
    acorns_per_month = initial_acorns / 3 →
    remaining_acorns_per_month = 60 →
    acorns_taken_per_month = acorns_per_month - remaining_acorns_per_month →
    3 * acorns_taken_per_month = 30 :=
by
  intros initial_acorns acorns_per_month remaining_acorns_per_month acorns_taken_per_month
  sorry

end squirrel_spring_acorns_l215_215637


namespace quadruplets_sets_l215_215931

theorem quadruplets_sets (a b c babies: ℕ) (h1: 2 * a + 3 * b + 4 * c = 1200) (h2: b = 5 * c) (h3: a = 2 * b) :
  4 * c = 123 :=
by
  sorry

end quadruplets_sets_l215_215931


namespace price_per_pound_of_rocks_l215_215339

def number_of_rocks : ℕ := 10
def average_weight_per_rock : ℝ := 1.5
def total_amount_made : ℝ := 60

theorem price_per_pound_of_rocks:
  (total_amount_made / (number_of_rocks * average_weight_per_rock)) = 4 := 
by
  sorry

end price_per_pound_of_rocks_l215_215339


namespace similar_triangle_perimeters_l215_215699

theorem similar_triangle_perimeters 
  (h_ratio : ℕ) (h_ratio_eq : h_ratio = 2/3)
  (sum_perimeters : ℕ) (sum_perimeters_eq : sum_perimeters = 50)
  (a b : ℕ)
  (perimeter_ratio : ℕ) (perimeter_ratio_eq : perimeter_ratio = 2/3)
  (hyp1 : a + b = sum_perimeters)
  (hyp2 : a * 3 = b * 2) :
  (a = 20 ∧ b = 30) :=
by
  sorry

end similar_triangle_perimeters_l215_215699


namespace dice_win_properties_l215_215060

theorem dice_win_properties:
  ∃ (A B C : List ℕ),
    A = [1, 4, 4, 4, 4, 4] ∧ 
    B = [2, 2, 2, 5, 5, 5] ∧ 
    C = [3, 3, 3, 3, 3, 6] ∧
    ((Probability (λ a b, a > b) A B > 1/2) ∧
     (Probability (λ b c, b > c) C B > 1/2) ∧
     (Probability (λ c a, a > c) A C > 1/2)) :=
by
  let A := [1, 4, 4, 4, 4, 4]
  let B := [2, 2, 2, 5, 5, 5]
  let C := [3, 3, 3, 3, 3, 6]

  have h1: (Probability (λ a b, a > b) A B > 1/2),
  sorry

  have h2: (Probability (λ b c, b > c) C B > 1/2),
  sorry

  have h3: (Probability (λ c a, a > c) A C > 1/2),
  sorry

  exact ⟨A, B, C, rfl, rfl, rfl, ⟨h1, h2, h3⟩⟩

end dice_win_properties_l215_215060


namespace length_of_second_train_l215_215388

def first_train_length : ℝ := 290
def first_train_speed_kmph : ℝ := 120
def second_train_speed_kmph : ℝ := 80
def cross_time : ℝ := 9

noncomputable def first_train_speed_mps := (first_train_speed_kmph * 1000) / 3600
noncomputable def second_train_speed_mps := (second_train_speed_kmph * 1000) / 3600
noncomputable def relative_speed := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance_covered := relative_speed * cross_time
noncomputable def second_train_length := total_distance_covered - first_train_length

theorem length_of_second_train : second_train_length = 209.95 := by
  sorry

end length_of_second_train_l215_215388


namespace min_adjacent_seat_occupation_l215_215913

def minOccupiedSeats (n : ℕ) : ℕ :=
  n / 3

theorem min_adjacent_seat_occupation (n : ℕ) (h : n = 150) :
  minOccupiedSeats n = 50 :=
by
  -- Placeholder for proof
  sorry

end min_adjacent_seat_occupation_l215_215913


namespace inequality_proof_l215_215146

theorem inequality_proof {a b c d e f : ℝ} (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
sorry

end inequality_proof_l215_215146


namespace eccentricity_of_ellipse_l215_215166

open Real

noncomputable def eccentricity_min (m : ℝ) (h₁ : m > 0) (h₂ : m ≥ 2) : ℝ :=
  if h : m = 2 then (sqrt 6)/3 else 0

theorem eccentricity_of_ellipse (m : ℝ) (h₁ : m > 0) (h₂ : m ≥ 2) :
    eccentricity_min m h₁ h₂ = (sqrt 6)/3 := by
  sorry

end eccentricity_of_ellipse_l215_215166


namespace dogwood_tree_count_l215_215615

def initial_dogwoods : ℕ := 34
def additional_dogwoods : ℕ := 49
def total_dogwoods : ℕ := initial_dogwoods + additional_dogwoods

theorem dogwood_tree_count :
  total_dogwoods = 83 :=
by
  -- omitted proof
  sorry

end dogwood_tree_count_l215_215615


namespace fernanda_savings_calc_l215_215645

noncomputable def aryan_debt : ℝ := 1200
noncomputable def kyro_debt : ℝ := aryan_debt / 2
noncomputable def aryan_payment : ℝ := 0.60 * aryan_debt
noncomputable def kyro_payment : ℝ := 0.80 * kyro_debt
noncomputable def initial_savings : ℝ := 300
noncomputable def total_payment_received : ℝ := aryan_payment + kyro_payment
noncomputable def total_savings : ℝ := initial_savings + total_payment_received

theorem fernanda_savings_calc : total_savings = 1500 := by
  sorry

end fernanda_savings_calc_l215_215645


namespace skyler_total_songs_skyler_success_breakdown_l215_215871

noncomputable def skyler_songs : ℕ :=
  let hit_songs := 25
  let top_100_songs := hit_songs + 10
  let unreleased_songs := hit_songs - 5
  let duets_total := 12
  let duets_top_20 := duets_total / 2
  let duets_not_top_200 := duets_total / 2
  let soundtracks_total := 18
  let soundtracks_extremely := 3
  let soundtracks_moderate := 8
  let soundtracks_lukewarm := 7
  let projects_total := 22
  let projects_global := 1
  let projects_regional := 7
  let projects_overlooked := 14
  hit_songs + top_100_songs + unreleased_songs + duets_total + soundtracks_total + projects_total

theorem skyler_total_songs : skyler_songs = 132 := by
  sorry

theorem skyler_success_breakdown :
  let extremely_successful := 25 + 1
  let successful := 35 + 6 + 3
  let moderately_successful := 8 + 7
  let less_successful := 7 + 14 + 6
  let unreleased := 20
  (extremely_successful, successful, moderately_successful, less_successful, unreleased) =
  (26, 44, 15, 27, 20) := by
  sorry

end skyler_total_songs_skyler_success_breakdown_l215_215871


namespace maximize_distance_l215_215402

noncomputable def maxTotalDistance (x : ℕ) (y : ℕ) (cityMPG highwayMPG : ℝ) (totalGallons : ℝ) : ℝ :=
  let cityDistance := cityMPG * ((x / 100.0) * totalGallons)
  let highwayDistance := highwayMPG * ((y / 100.0) * totalGallons)
  cityDistance + highwayDistance

theorem maximize_distance (x y : ℕ) (hx : x + y = 100) :
  maxTotalDistance x y 7.6 12.2 24.0 = 7.6 * (x / 100.0 * 24.0) + 12.2 * ((100.0 - x) / 100.0 * 24.0) :=
by
  sorry

end maximize_distance_l215_215402


namespace circumference_of_jogging_track_l215_215369

-- Definitions for the given conditions
def speed_deepak : ℝ := 4.5
def speed_wife : ℝ := 3.75
def meet_time : ℝ := 4.32

-- The theorem stating the problem
theorem circumference_of_jogging_track : 
  (speed_deepak + speed_wife) * meet_time = 35.64 :=
by
  sorry

end circumference_of_jogging_track_l215_215369


namespace class_mean_l215_215992

theorem class_mean
  (num_students_1 : ℕ)
  (num_students_2 : ℕ)
  (total_students : ℕ)
  (mean_score_1 : ℚ)
  (mean_score_2 : ℚ)
  (new_mean_score : ℚ)
  (h1 : num_students_1 + num_students_2 = total_students)
  (h2 : total_students = 30)
  (h3 : num_students_1 = 24)
  (h4 : mean_score_1 = 80)
  (h5 : num_students_2 = 6)
  (h6 : mean_score_2 = 85) :
  new_mean_score = 81 :=
by
  sorry

end class_mean_l215_215992


namespace part_a_part_b_l215_215302

variable (p : ℕ)
variable (h1 : prime p)
variable (h2 : p > 3)

theorem part_a : (p + 1) % 4 = 0 ∨ (p - 1) % 4 = 0 :=
sorry

theorem part_b : ¬ ((p + 1) % 5 = 0 ∨ (p - 1) % 5 = 0) :=
sorry

end part_a_part_b_l215_215302


namespace nails_remaining_l215_215518

theorem nails_remaining (nails_initial : ℕ) (kitchen_fraction : ℚ) (fence_fraction : ℚ) (nails_used_kitchen : ℕ) (nails_remaining_after_kitchen : ℕ) (nails_used_fence : ℕ) (nails_remaining_final : ℕ) 
  (h1 : nails_initial = 400) 
  (h2 : kitchen_fraction = 0.30) 
  (h3 : nails_used_kitchen = kitchen_fraction * nails_initial) 
  (h4 : nails_remaining_after_kitchen = nails_initial - nails_used_kitchen) 
  (h5 : fence_fraction = 0.70) 
  (h6 : nails_used_fence = fence_fraction * nails_remaining_after_kitchen) 
  (h7 : nails_remaining_final = nails_remaining_after_kitchen - nails_used_fence) :
  nails_remaining_final = 84 := by
sorry

end nails_remaining_l215_215518


namespace total_time_for_process_l215_215719

-- Given conditions
def cat_resistance_time : ℕ := 20
def walking_distance : ℕ := 64
def walking_rate : ℕ := 8

-- Prove the total time
theorem total_time_for_process : cat_resistance_time + (walking_distance / walking_rate) = 28 := by
  sorry

end total_time_for_process_l215_215719


namespace birds_on_the_fence_l215_215878

theorem birds_on_the_fence (x : ℕ) : 10 + 2 * x = 50 → x = 20 := by
  sorry

end birds_on_the_fence_l215_215878


namespace coprime_squares_l215_215837

theorem coprime_squares (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : ∃ k : ℕ, ab = k^2) : 
  ∃ p q : ℕ, a = p^2 ∧ b = q^2 :=
by
  sorry

end coprime_squares_l215_215837


namespace fruit_basket_apples_oranges_ratio_l215_215703

theorem fruit_basket_apples_oranges_ratio : 
  ∀ (apples oranges : ℕ), 
  apples = 15 ∧ (2 * apples / 3 + 2 * oranges / 3 = 50) → (apples = 15 ∧ oranges = 60) → apples / gcd apples oranges = 1 ∧ oranges / gcd apples oranges = 4 :=
by 
  intros apples oranges h1 h2
  have h_apples : apples = 15 := by exact h2.1
  have h_oranges : oranges = 60 := by exact h2.2
  rw [h_apples, h_oranges]
  sorry

end fruit_basket_apples_oranges_ratio_l215_215703


namespace milkman_A_rent_share_l215_215769

theorem milkman_A_rent_share : 
  let A_cows := 24
  let A_months := 3
  let B_cows := 10
  let B_months := 5
  let C_cows := 35
  let C_months := 4
  let D_cows := 21
  let D_months := 3
  let total_rent := 3250
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + D_cow_months
  let fraction_A := A_cow_months / total_cow_months
  let A_rent_share := total_rent * fraction_A
  A_rent_share = 720 := 
by
  sorry

end milkman_A_rent_share_l215_215769


namespace rectangular_prism_dimensions_l215_215379

theorem rectangular_prism_dimensions 
    (a b c : ℝ) -- edges of the rectangular prism
    (h_increase_volume : (2 * a * b = 90)) -- condition 2: increasing height increases volume by 90 cm³ 
    (h_volume_proportion : (a * (c + 2)) / 2 = (3 / 5) * (a * b * c)) -- condition 3: height change results in 3/5 of original volume
    (h_edge_relation : (a = 5 * b ∨ b = 5 * a ∨ a * b = 45)) -- condition 1: one edge 5 times longer
    : 
    (a = 0.9 ∧ b = 50 ∧ c = 10) ∨ (a = 2 ∧ b = 22.5 ∧ c = 10) ∨ (a = 3 ∧ b = 15 ∧ c = 10) :=
sorry

end rectangular_prism_dimensions_l215_215379


namespace arccos_neg_one_eq_pi_l215_215944

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l215_215944


namespace circle_eq_l215_215200

variable {M : Type*} [EuclideanSpace ℝ M]

def on_line (M : M) : Prop := ∃ x y : ℝ, 2 * x + y = 1

def on_circle (c r : ℝ) (M : M) : Prop := ∃ (x y : ℝ), (x - c)^2 + (y - (-r))^2 = 5

theorem circle_eq (M : M) (hM : on_line M) (h1 : on_circle 1 (sqrt 5) (3, 0)) (h2 : on_circle 1 (sqrt 5) (0, 1)) :
  ∃ c r, (x - c)^2 + (y - r)^2 = 5 := sorry

end circle_eq_l215_215200


namespace final_expression_simplified_l215_215172

variable (a : ℝ)

theorem final_expression_simplified : 
  (2 * a + 6 - 3 * a) / 2 = -a / 2 + 3 := 
by 
sorry

end final_expression_simplified_l215_215172


namespace product_of_odd_primes_mod_32_l215_215022

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215022


namespace initial_apples_l215_215783

-- Define the initial conditions
def r : Nat := 14
def s : Nat := 2 * r
def remaining : Nat := 32
def total_removed : Nat := r + s

-- The proof problem: Prove that the initial number of apples is 74
theorem initial_apples : (total_removed + remaining = 74) :=
by
  sorry

end initial_apples_l215_215783


namespace inequality_proof_l215_215145

theorem inequality_proof {a b c d e f : ℝ} (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
sorry

end inequality_proof_l215_215145


namespace cos_neg_300_eq_half_l215_215631

theorem cos_neg_300_eq_half : Real.cos (-300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_neg_300_eq_half_l215_215631


namespace percentage_increase_l215_215349

theorem percentage_increase (d : ℝ) (v_current v_reduce v_increase t_reduce t_increase : ℝ) (h1 : d = 96)
  (h2 : v_current = 8) (h3 : v_reduce = v_current - 4) (h4 : t_reduce = d / v_reduce) 
  (h5 : t_increase = d / v_increase) (h6 : t_reduce = t_current + 16) (h7 : t_increase = t_current - 16) :
  (v_increase - v_current) / v_current * 100 = 50 := 
sorry

end percentage_increase_l215_215349


namespace relationship_between_M_and_P_l215_215139

def M := {y : ℝ | ∃ x : ℝ, y = x^2 - 4}
def P := {x : ℝ | 2 ≤ x ∧ x ≤ 4}

theorem relationship_between_M_and_P : ∀ y ∈ {y : ℝ | ∃ x ∈ P, y = x^2 - 4}, y ∈ M :=
by
  sorry

end relationship_between_M_and_P_l215_215139


namespace value_added_to_half_is_five_l215_215440

theorem value_added_to_half_is_five (n V : ℕ) (h₁ : n = 16) (h₂ : (1 / 2 : ℝ) * n + V = 13) : V = 5 := 
by 
  sorry

end value_added_to_half_is_five_l215_215440


namespace product_of_roots_eq_negative_forty_nine_l215_215810

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l215_215810


namespace find_integers_l215_215288

-- Problem statement rewritten as a Lean 4 definition
theorem find_integers (a b c : ℤ) (H1 : a = 1) (H2 : b = 2) (H3 : c = 1) : 
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c :=
by
  -- The proof will be presented here
  sorry

end find_integers_l215_215288


namespace mary_sheep_remaining_l215_215347

theorem mary_sheep_remaining : 
  let initial_sheep := 400
  let sheep_given_to_sister := initial_sheep / 4
  let sheep_after_sister := initial_sheep - sheep_given_to_sister
  let sheep_given_to_brother := sheep_after_sister / 2
  let remaining_sheep := sheep_after_sister - sheep_given_to_brother
  remaining_sheep = 150 :=
by
  assume initial_sheep := 400
  have sheep_given_to_sister := initial_sheep / 4
  have sheep_after_sister := initial_sheep - sheep_given_to_sister
  have sheep_given_to_brother := sheep_after_sister / 2
  have remaining_sheep := sheep_after_sister - sheep_given_to_brother
  show remaining_sheep = 150
  sorry

end mary_sheep_remaining_l215_215347


namespace urn_probability_l215_215123

theorem urn_probability :
  ∀ (urn: Finset (ℕ × ℕ)), 
    urn = {(2, 1)} →
    (∀ (n : ℕ) (urn' : Finset (ℕ × ℕ)), n ≤ 5 → urn = urn' → 
      (∃ (r b : ℕ), (r, b) ∈ urn' ∧ urn' = {(r + 1, b), (r, b + 1)} ∨ (r, b) ∈ urn' ∧ urn' = {(r + 1, b), (r, b + 1)}) → 
    ∃ (p : ℚ), p = 8 / 21)
  := by
    sorry

end urn_probability_l215_215123


namespace gcd_12a_20b_min_value_l215_215324

-- Define the conditions
def is_positive_integer (x : ℕ) : Prop := x > 0

def gcd_condition (a b d : ℕ) : Prop := gcd a b = d

-- State the problem
theorem gcd_12a_20b_min_value (a b : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_gcd_ab : gcd_condition a b 10) :
  ∃ (k : ℕ), k = gcd (12 * a) (20 * b) ∧ k = 40 :=
by
  sorry

end gcd_12a_20b_min_value_l215_215324


namespace least_number_to_subtract_l215_215382

theorem least_number_to_subtract (n : ℕ) : 
  ∃ k : ℕ, k = 762429836 % 17 ∧ k = 15 := 
by sorry

end least_number_to_subtract_l215_215382


namespace expand_expression_l215_215798

variable {R : Type} [CommRing R]
variables (x y : R)

theorem expand_expression :
  5 * (3 * x^3 - 4 * x * y + x^2 - y^2) = 15 * x^3 - 20 * x * y + 5 * x^2 - 5 * y^2 :=
by
  sorry

end expand_expression_l215_215798


namespace find_a10_l215_215710

def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

theorem find_a10 
  (a1 d : ℤ)
  (h_condition : a1 + (a1 + 18 * d) = -18) :
  arithmetic_sequence a1 d 10 = -9 := 
by
  sorry

end find_a10_l215_215710


namespace line_slope_product_l215_215390

theorem line_slope_product (x y : ℝ) (h1 : (x, 6) = (x, 6)) (h2 : (10, y) = (10, y)) (h3 : ∀ x, y = (1 / 2) * x) : x * y = 60 :=
sorry

end line_slope_product_l215_215390


namespace remainder_12401_163_l215_215896

theorem remainder_12401_163 :
  let original_number := 12401
  let divisor := 163
  let quotient := 76
  let remainder := 13
  original_number = divisor * quotient + remainder :=
by
  sorry

end remainder_12401_163_l215_215896


namespace digit_difference_l215_215629

variable (X Y : ℕ)

theorem digit_difference (h : 10 * X + Y - (10 * Y + X) = 27) : X - Y = 3 :=
by
  sorry

end digit_difference_l215_215629


namespace triangle_side_lengths_l215_215212

theorem triangle_side_lengths (a b c r : ℕ) (h : a / b / c = 25 / 29 / 36) (hinradius : r = 232) :
  (a = 725 ∧ b = 841 ∧ c = 1044) :=
by
  sorry

end triangle_side_lengths_l215_215212


namespace inequality_holds_for_a_l215_215293

theorem inequality_holds_for_a (a : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < -1 → (x + 1)^2 < Real.logb a (|x|)) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end inequality_holds_for_a_l215_215293


namespace total_money_spent_correct_l215_215594

def money_spent_at_mall : Int := 250

def cost_per_movie : Int := 24
def number_of_movies : Int := 3
def money_spent_at_movies := cost_per_movie * number_of_movies

def cost_per_bag_of_beans : Float := 1.25
def number_of_bags : Int := 20
def money_spent_at_market := cost_per_bag_of_beans * number_of_bags

def total_money_spent := money_spent_at_mall + money_spent_at_movies + money_spent_at_market

theorem total_money_spent_correct : total_money_spent = 347 := by
  sorry

end total_money_spent_correct_l215_215594


namespace nails_remaining_proof_l215_215515

noncomputable
def remaining_nails (initial_nails kitchen_percent fence_percent : ℕ) : ℕ :=
  let kitchen_used := initial_nails * kitchen_percent / 100
  let remaining_after_kitchen := initial_nails - kitchen_used
  let fence_used := remaining_after_kitchen * fence_percent / 100
  let final_remaining := remaining_after_kitchen - fence_used
  final_remaining

theorem nails_remaining_proof :
  remaining_nails 400 30 70 = 84 := by
  sorry

end nails_remaining_proof_l215_215515


namespace product_prices_determined_max_product_A_pieces_l215_215118

theorem product_prices_determined (a b : ℕ) :
  (20 * a + 15 * b = 380) →
  (15 * a + 10 * b = 280) →
  a = 16 ∧ b = 4 :=
by sorry

theorem max_product_A_pieces (x : ℕ) :
  (16 * x + 4 * (100 - x) ≤ 900) →
  x ≤ 41 :=
by sorry

end product_prices_determined_max_product_A_pieces_l215_215118


namespace no_such_m_exists_l215_215714

theorem no_such_m_exists : ¬ ∃ m : ℝ, ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0 :=
sorry

end no_such_m_exists_l215_215714


namespace cost_per_play_l215_215889

-- Conditions
def initial_money : ℝ := 3
def points_per_red_bucket : ℝ := 2
def points_per_green_bucket : ℝ := 3
def rings_per_play : ℕ := 5
def games_played : ℕ := 2
def red_buckets : ℕ := 4
def green_buckets : ℕ := 5
def total_games : ℕ := 3
def total_points : ℝ := 38

-- Point calculations
def points_from_red_buckets : ℝ := red_buckets * points_per_red_bucket
def points_from_green_buckets : ℝ := green_buckets * points_per_green_bucket
def current_points : ℝ := points_from_red_buckets + points_from_green_buckets
def points_needed : ℝ := total_points - current_points

-- Define the theorem statement
theorem cost_per_play :
  (initial_money / (games_played : ℝ)) = 1.50 :=
  sorry

end cost_per_play_l215_215889


namespace find_number_l215_215511

theorem find_number : ∃ x : ℝ, 3550 - (1002 / x) = 3500 ∧ x = 20.04 :=
by
  sorry

end find_number_l215_215511


namespace problem_l215_215830

theorem problem (a b : ℝ) :
  (∀ x : ℝ, 3 * x - 1 ≤ a ∧ 2 * x ≥ 6 - b → -1 ≤ x ∧ x ≤ 2) →
  a + b = 13 := by
  sorry

end problem_l215_215830


namespace fifth_derivative_l215_215287

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 - 7) * Real.log (x - 1)

theorem fifth_derivative :
  ∀ x, (deriv^[5] f) x = 8 * (x ^ 2 - 5 * x - 11) / ((x - 1) ^ 5) :=
by
  sorry

end fifth_derivative_l215_215287


namespace geometric_sequence_problem_l215_215167

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) (h_seq : geometric_sequence a) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 := sorry

end geometric_sequence_problem_l215_215167


namespace complex_number_equality_l215_215669

-- Define the conditions a, b ∈ ℝ and a + i = 1 - bi
theorem complex_number_equality (a b : ℝ) (i : ℂ) (h : a + i = 1 - b * i) : (a + b * i) ^ 8 = 16 :=
  sorry

end complex_number_equality_l215_215669


namespace max_cursed_roads_l215_215709

/--
In the Westeros Empire that started with 1000 cities and 2017 roads,
where initially the graph is connected,
prove that the maximum number of roads that can be cursed to form exactly 7 connected components is 1024.
-/
theorem max_cursed_roads (cities roads components : ℕ) (connected : bool) :
  cities = 1000 ∧ roads = 2017 ∧ connected = tt ∧ components = 7 → 
  ∃ N, N = 1024 :=
by {
  sorry
}

end max_cursed_roads_l215_215709


namespace cats_left_l215_215918

theorem cats_left (siamese house persian sold_first sold_second : ℕ) (h1 : siamese = 23) (h2 : house = 17) (h3 : persian = 29) (h4 : sold_first = 40) (h5 : sold_second = 12) :
  siamese + house + persian - sold_first - sold_second = 17 :=
by sorry

end cats_left_l215_215918


namespace product_of_areas_eq_square_of_volume_l215_215075

theorem product_of_areas_eq_square_of_volume
    (a b c : ℝ)
    (bottom_area : ℝ) (side_area : ℝ) (front_area : ℝ)
    (volume : ℝ)
    (h1 : bottom_area = a * b)
    (h2 : side_area = b * c)
    (h3 : front_area = c * a)
    (h4 : volume = a * b * c) :
    bottom_area * side_area * front_area = volume ^ 2 := by
  -- proof omitted
  sorry

end product_of_areas_eq_square_of_volume_l215_215075


namespace acute_angle_WV_XY_zero_l215_215180

open Real

/-- In triangle XYZ with given angles and side lengths, points R and S on sides,
    midpoints W and V, prove the acute angle WV and XY is 0 degrees. -/
theorem acute_angle_WV_XY_zero :
  ∀ (X Y Z W V R S : ℝ)
    (h₁ : W = (X + Y) / 2)    -- midpoint of XY
    (h₂ : V = (R + S) / 2)    -- midpoint of RS
    (angle_X : 40°)
    (angle_Y : 54°)
    (XY_length : XY = 15)
    (XR_length : XR = 1.5)
    (YS_length : YS = 1.5)
    (angle_Z : angle_Z = 180° - angle_X - angle_Y)
    (RS_length : RS = 7.5),   -- RS as the result of midpoint calculations, $W = 7.5$
    acute_angle WV XY = 0° :=  
by
  sorry

end acute_angle_WV_XY_zero_l215_215180


namespace check_roots_l215_215960

noncomputable def roots_of_quadratic_eq (a b : ℂ) : list ℂ :=
  [(-b + complex.sqrt(b ^ 2 - 4 * a * (3 - 4 * complex.i))) / (2 * a),
   (-b - complex.sqrt(b ^ 2 - 4 * a * (3 - 4 * complex.i))) / (2 * a)]

theorem check_roots :
  ∀ (z : ℂ), z^2 + 2 * z + (3 - 4 * complex.i) = 0 ↔ (z = complex.i ∨ z = -3 - 2 * complex.i) :=
begin
  sorry
end

end check_roots_l215_215960


namespace product_of_roots_eq_negative_forty_nine_l215_215811

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l215_215811


namespace product_of_t_values_l215_215814

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l215_215814


namespace sally_eats_sandwiches_l215_215866

theorem sally_eats_sandwiches
  (saturday_sandwiches : ℕ)
  (bread_per_sandwich : ℕ)
  (total_bread : ℕ)
  (one_sandwich_on_sunday : ℕ)
  (saturday_bread : saturday_sandwiches * bread_per_sandwich = 4)
  (total_bread_consumed : total_bread = 6)
  (bread_on_sundy : bread_per_sandwich = 2) :
  (total_bread - saturday_sandwiches * bread_per_sandwich) / bread_per_sandwich = one_sandwich_on_sunday :=
sorry

end sally_eats_sandwiches_l215_215866


namespace maximum_value_product_cube_expression_l215_215052

theorem maximum_value_product_cube_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^3 - x * y^2 + y^3) * (x^3 - x^2 * z + z^3) * (y^3 - y^2 * z + z^3) ≤ 1 :=
sorry

end maximum_value_product_cube_expression_l215_215052


namespace product_of_odd_primes_mod_32_l215_215029

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215029


namespace omitted_angle_measure_l215_215656

theorem omitted_angle_measure (initial_sum correct_sum : ℝ) (H_initial : initial_sum = 2083) (H_correct : correct_sum = 2160) :
  correct_sum - initial_sum = 77 :=
by sorry

end omitted_angle_measure_l215_215656


namespace axis_of_symmetry_y_range_l215_215407

/-- 
The equation of the curve is given by |x| + y^2 - 3y = 0.
We aim to prove two properties:
1. The axis of symmetry of this curve is x = 0.
2. The range of possible values for y is [0, 3].
-/
noncomputable def curve (x y : ℝ) : ℝ := |x| + y^2 - 3*y

theorem axis_of_symmetry : ∀ x y : ℝ, curve x y = 0 → x = 0 :=
sorry

theorem y_range : ∀ y : ℝ, ∃ x : ℝ, curve x y = 0 → (0 ≤ y ∧ y ≤ 3) :=
sorry

end axis_of_symmetry_y_range_l215_215407


namespace trapezoid_isosceles_l215_215297

noncomputable def is_perpendicular (A B: Point) (line: Line) : Prop := sorry
noncomputable def midpoint (A B: Point) : Point := sorry

theorem trapezoid_isosceles 
    (ABCD : Quadrilateral) (A B C D : Point) 
    (h_trapezoid : ABCD.is_trapezoid AD BC)
    (K : Point) (hK : K = midpoint A C)
    (L : Point) (hL : L = midpoint B D)
    (h_perp_A : is_perpendicular A (Line.mk K L) (Line.mk C D))
    (h_perp_D : is_perpendicular D (Line.mk K L) (Line.mk A B)) :
    ABCD.is_isosceles_trapezoid :=
begin
    sorry
end

end trapezoid_isosceles_l215_215297


namespace physicist_imons_no_entanglement_l215_215634

theorem physicist_imons_no_entanglement (G : SimpleGraph V) :
  (∃ ops : ℕ, ∀ v₁ v₂ : V, ¬G.Adj v₁ v₂) :=
by
  sorry

end physicist_imons_no_entanglement_l215_215634


namespace smallest_integer_satisfying_conditions_l215_215962

-- Define the conditions explicitly as hypotheses
def satisfies_congruence_3_2 (n : ℕ) : Prop :=
  n % 3 = 2

def satisfies_congruence_7_2 (n : ℕ) : Prop :=
  n % 7 = 2

def satisfies_congruence_8_2 (n : ℕ) : Prop :=
  n % 8 = 2

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Define the smallest positive integer satisfying the above conditions
theorem smallest_integer_satisfying_conditions : ∃ (n : ℕ), n > 1 ∧ satisfies_congruence_3_2 n ∧ satisfies_congruence_7_2 n ∧ satisfies_congruence_8_2 n ∧ is_perfect_square n :=
  by
    sorry

end smallest_integer_satisfying_conditions_l215_215962


namespace find_unit_prices_l215_215205

theorem find_unit_prices (price_A price_B : ℕ) 
  (h1 : price_A = price_B + 5) 
  (h2 : 1000 / price_A = 750 / price_B) : 
  price_A = 20 ∧ price_B = 15 := 
by 
  sorry

end find_unit_prices_l215_215205


namespace sum_cos_4x_4y_4z_l215_215857

theorem sum_cos_4x_4y_4z (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) :
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 :=
by
  sorry

end sum_cos_4x_4y_4z_l215_215857


namespace product_of_values_t_squared_eq_49_l215_215809

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l215_215809


namespace toms_restaurant_bill_l215_215106

theorem toms_restaurant_bill (num_adults num_children : ℕ) (meal_cost : ℕ) (total_meals : ℕ) (bill : ℕ) :
  num_adults = 2 ∧ num_children = 5 ∧ meal_cost = 8 ∧ total_meals = num_adults + num_children ∧ bill = total_meals * meal_cost → bill = 56 :=
by sorry

end toms_restaurant_bill_l215_215106


namespace number_of_outliers_l215_215659

def data_set : List ℕ := [10, 24, 36, 36, 42, 45, 45, 46, 58, 64]
def Q1 : ℕ := 36
def Q3 : ℕ := 46
def IQR : ℕ := Q3 - Q1
def low_threshold : ℕ := Q1 - 15
def high_threshold : ℕ := Q3 + 15
def outliers : List ℕ := data_set.filter (λ x => x < low_threshold ∨ x > high_threshold)

theorem number_of_outliers : outliers.length = 3 :=
  by
    -- Proof would go here
    sorry

end number_of_outliers_l215_215659


namespace find_primes_satisfying_equation_l215_215666

theorem find_primes_satisfying_equation :
  {p : ℕ | p.Prime ∧ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p} = {2, 3, 7} :=
by
  sorry

end find_primes_satisfying_equation_l215_215666


namespace product_mod_32_l215_215006

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215006


namespace value_of_expression_l215_215175

theorem value_of_expression (m n : ℝ) (h : m + 2 * n = 1) : 3 * m^2 + 6 * m * n + 6 * n = 3 :=
by
  sorry -- Placeholder for the proof

end value_of_expression_l215_215175


namespace distance_from_point_to_line_l215_215602

open Real

noncomputable def point_to_line_distance (a b c x0 y0 : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2)

theorem distance_from_point_to_line (a b c x0 y0 : ℝ) :
  point_to_line_distance a b c x0 y0 = abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2) :=
by
  sorry

end distance_from_point_to_line_l215_215602


namespace probability_non_smokers_getting_lung_cancer_l215_215639

theorem probability_non_smokers_getting_lung_cancer 
  (overall_lung_cancer : ℝ)
  (smokers_fraction : ℝ)
  (smokers_lung_cancer : ℝ)
  (non_smokers_lung_cancer : ℝ)
  (H1 : overall_lung_cancer = 0.001)
  (H2 : smokers_fraction = 0.2)
  (H3 : smokers_lung_cancer = 0.004)
  (H4 : overall_lung_cancer = smokers_fraction * smokers_lung_cancer + (1 - smokers_fraction) * non_smokers_lung_cancer) :
  non_smokers_lung_cancer = 0.00025 := by
  sorry

end probability_non_smokers_getting_lung_cancer_l215_215639


namespace product_mod_32_is_15_l215_215015

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215015


namespace sum_of_primes_less_than_10_is_17_l215_215237

-- Definition of prime numbers less than 10
def primes_less_than_10 : List ℕ := [2, 3, 5, 7]

-- Sum of the prime numbers less than 10
def sum_primes_less_than_10 : ℕ := List.sum primes_less_than_10

theorem sum_of_primes_less_than_10_is_17 : sum_primes_less_than_10 = 17 := 
by
  sorry

end sum_of_primes_less_than_10_is_17_l215_215237


namespace arccos_neg_one_eq_pi_proof_l215_215938

noncomputable def arccos_neg_one_eq_pi : Prop :=
  arccos (-1) = π

theorem arccos_neg_one_eq_pi_proof : arccos_neg_one_eq_pi := by
  sorry

end arccos_neg_one_eq_pi_proof_l215_215938


namespace arccos_neg_one_eq_pi_l215_215943

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l215_215943


namespace constant_function_on_chessboard_l215_215378

theorem constant_function_on_chessboard
  (f : ℤ × ℤ → ℝ)
  (h_nonneg : ∀ (m n : ℤ), 0 ≤ f (m, n))
  (h_mean : ∀ (m n : ℤ), f (m, n) = (f (m + 1, n) + f (m - 1, n) + f (m, n + 1) + f (m, n - 1)) / 4) :
  ∃ c : ℝ, ∀ (m n : ℤ), f (m, n) = c :=
sorry

end constant_function_on_chessboard_l215_215378


namespace rationalization_sum_l215_215354

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalization_sum : rationalize_denominator = 75 := by
  sorry

end rationalization_sum_l215_215354


namespace yellow_not_greater_than_green_l215_215617

theorem yellow_not_greater_than_green
    (G Y S : ℕ)
    (h1 : G + Y + S = 100)
    (h2 : G + S / 2 = 50)
    (h3 : Y + S / 2 = 50) : ¬ Y > G :=
sorry

end yellow_not_greater_than_green_l215_215617


namespace remainder_M_mod_32_l215_215035

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215035


namespace roots_of_quadratic_eq_l215_215838

noncomputable def r : ℂ := sorry
noncomputable def s : ℂ := sorry

def roots_eq (h : 3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : Prop :=
  (1 / r^3) + (1 / s^3) = 1

theorem roots_of_quadratic_eq (h:3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : roots_eq h :=
sorry

end roots_of_quadratic_eq_l215_215838


namespace intersection_M_N_eq_M_l215_215468

-- Definitions of M and N
def M : Set ℝ := { x : ℝ | x^2 - x < 0 }
def N : Set ℝ := { x : ℝ | abs x < 2 }

-- Proof statement
theorem intersection_M_N_eq_M : M ∩ N = M := 
  sorry

end intersection_M_N_eq_M_l215_215468


namespace range_of_m_l215_215301

variable (m : ℝ)

def p : Prop := m + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : ¬ (p m ∧ q m)) : m ≤ -2 ∨ m > -1 := 
by
  sorry

end range_of_m_l215_215301


namespace distinct_infinite_solutions_l215_215235

theorem distinct_infinite_solutions (n : ℕ) (hn : n > 0) : 
  ∃ p q : ℤ, p + q * Real.sqrt 5 = (9 + 4 * Real.sqrt 5) ^ n ∧ (p * p - 5 * q * q = 1) ∧ 
  ∀ m : ℕ, (m ≠ n → (9 + 4 * Real.sqrt 5) ^ m ≠ (9 + 4 * Real.sqrt 5) ^ n) :=
by
  sorry

end distinct_infinite_solutions_l215_215235


namespace parallel_lines_slope_l215_215662

theorem parallel_lines_slope (k : ℝ) :
  (∀ x : ℝ, 5 * x - 3 = (3 * k) * x + 7 -> ((3 * k) = 5)) -> (k = 5 / 3) :=
by
  -- Posing the conditions on parallel lines
  intro h_eq_slopes
  -- We know 3k = 5, hence k = 5 / 3
  have slope_eq : 3 * k = 5 := by sorry
  -- Therefore k = 5 / 3 follows from the fact 3k = 5
  have k_val : k = 5 / 3 := by sorry
  exact k_val

end parallel_lines_slope_l215_215662


namespace find_c_minus_2d_l215_215368

theorem find_c_minus_2d :
  ∃ (c d : ℕ), (c > d) ∧ (c - 2 * d = 0) ∧ (∀ x : ℕ, (x^2 - 18 * x + 72 = (x - c) * (x - d))) :=
by
  sorry

end find_c_minus_2d_l215_215368


namespace product_of_solutions_product_of_all_t_l215_215805

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l215_215805


namespace connie_total_markers_l215_215280

theorem connie_total_markers : 2315 + 1028 = 3343 :=
by
  sorry

end connie_total_markers_l215_215280


namespace percentage_workday_in_meetings_l215_215531

theorem percentage_workday_in_meetings :
  let workday_minutes := 10 * 60
  let first_meeting := 30
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_minutes := first_meeting + second_meeting + third_meeting
  (total_meeting_minutes * 100) / workday_minutes = 30 :=
by
  sorry

end percentage_workday_in_meetings_l215_215531


namespace quadratic_roots_l215_215961

-- Define the given conditions of the equation
def eqn (z : ℂ) : Prop := z^2 + 2 * z + (3 - 4 * Complex.I) = 0

-- State the theorem to prove that the roots of the equation are 2i and -2 + 2i.
theorem quadratic_roots :
  ∃ z1 z2 : ℂ, (z1 = 2 * Complex.I ∧ z2 = -2 + 2 * Complex.I) ∧ 
  (∀ z : ℂ, eqn z → z = z1 ∨ z = z2) :=
by
  sorry

end quadratic_roots_l215_215961


namespace arccos_neg_one_eq_pi_l215_215940

theorem arccos_neg_one_eq_pi : real.arccos (-1) = real.pi :=
by sorry

end arccos_neg_one_eq_pi_l215_215940


namespace probability_letter_in_MATHEMATICS_l215_215445

theorem probability_letter_in_MATHEMATICS :
  let alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  let mathematics := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']
  (mathematics.length : ℚ) / (alphabet.length : ℚ) = 4 / 13 :=
by
  sorry

end probability_letter_in_MATHEMATICS_l215_215445


namespace minimum_value_am_bn_l215_215904

-- Definitions and conditions
variables {a b m n : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < m) (h₃ : 0 < n)
variables (h₄ : a + b = 1) (h₅ : m * n = 2)

-- Statement of the proof problem
theorem minimum_value_am_bn :
  ∃ c, (∀ a b m n : ℝ, 0 < a → 0 < b → 0 < m → 0 < n → a + b = 1 → m * n = 2 → (am * bn) * (bm * an) ≥ c) ∧ c = 2 :=
sorry

end minimum_value_am_bn_l215_215904


namespace even_increasing_ordering_l215_215140

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Theorem to prove
theorem even_increasing_ordering (h_even : is_even_function f) (h_increasing : is_increasing_on_pos f) : 
  f 1 < f (-2) ∧ f (-2) < f 3 :=
by
  sorry

end even_increasing_ordering_l215_215140


namespace total_days_spent_on_island_l215_215852

noncomputable def first_expedition_weeks := 3
noncomputable def second_expedition_weeks := first_expedition_weeks + 2
noncomputable def last_expedition_weeks := 2 * second_expedition_weeks
noncomputable def total_weeks := first_expedition_weeks + second_expedition_weeks + last_expedition_weeks
noncomputable def total_days := 7 * total_weeks

theorem total_days_spent_on_island : total_days = 126 := by
  sorry

end total_days_spent_on_island_l215_215852


namespace translate_triangle_l215_215077

theorem translate_triangle (A B C A' : (ℝ × ℝ)) (hx_A : A = (2, 1)) (hx_B : B = (4, 3)) 
  (hx_C : C = (0, 2)) (hx_A' : A' = (-1, 5)) : 
  ∃ C' : (ℝ × ℝ), C' = (-3, 6) :=
by 
  sorry

end translate_triangle_l215_215077


namespace contractor_engaged_days_l215_215253

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end contractor_engaged_days_l215_215253


namespace amy_created_albums_l215_215926

theorem amy_created_albums (total_photos : ℕ) (photos_per_album : ℕ) 
  (h1 : total_photos = 180)
  (h2 : photos_per_album = 20) : 
  (total_photos / photos_per_album = 9) :=
by
  sorry

end amy_created_albums_l215_215926


namespace product_mod_32_is_15_l215_215011

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215011


namespace tan_alpha_implies_fraction_l215_215444

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
sorry

end tan_alpha_implies_fraction_l215_215444


namespace four_digit_integer_unique_l215_215259

theorem four_digit_integer_unique (a b c d : ℕ) (h1 : a + b + c + d = 16) (h2 : b + c = 10) (h3 : a - d = 2)
    (h4 : (a - b + c - d) % 11 = 0) : a = 4 ∧ b = 6 ∧ c = 4 ∧ d = 2 := 
  by 
    sorry

end four_digit_integer_unique_l215_215259


namespace maximize_profit_l215_215919

noncomputable def profit (x : ℝ) : ℝ :=
  let selling_price := 10 + 0.5 * x
  let sales_volume := 200 - 10 * x
  (selling_price - 8) * sales_volume

theorem maximize_profit : ∃ x : ℝ, x = 8 → profit x = profit 8 ∧ (∀ y : ℝ, profit y ≤ profit 8) := 
  sorry

end maximize_profit_l215_215919


namespace divided_scale_length_l215_215921

/-
  The problem definition states that we have a scale that is 6 feet 8 inches long, 
  and we need to prove that when the scale is divided into two equal parts, 
  each part is 3 feet 4 inches long.
-/

/-- Given length conditions in feet and inches --/
def total_length_feet : ℕ := 6
def total_length_inches : ℕ := 8

/-- Convert total length to inches --/
def total_length_in_inches := total_length_feet * 12 + total_length_inches

/-- Proof that if a scale is 6 feet 8 inches long and divided into 2 parts, each part is 3 feet 4 inches --/
theorem divided_scale_length :
  (total_length_in_inches / 2) = 40 ∧ (40 / 12 = 3 ∧ 40 % 12 = 4) :=
by
  sorry

end divided_scale_length_l215_215921


namespace exists_x_f_lt_g_l215_215308

noncomputable def f (x : ℝ) := (2 / Real.exp 1) ^ x

noncomputable def g (x : ℝ) := (Real.exp 1 / 3) ^ x

theorem exists_x_f_lt_g : ∃ x : ℝ, f x < g x := by
  sorry

end exists_x_f_lt_g_l215_215308


namespace train_crossing_time_l215_215119

theorem train_crossing_time:
  ∀ (length_train : ℝ) (speed_man_kmph : ℝ) (speed_train_kmph : ℝ),
    length_train = 125 →
    speed_man_kmph = 5 →
    speed_train_kmph = 69.994 →
    (125 / ((69.994 + 5) * (1000 / 3600))) = 6.002 :=
by
  intros length_train speed_man_kmph speed_train_kmph h1 h2 h3
  sorry

end train_crossing_time_l215_215119


namespace product_of_odd_primes_mod_32_l215_215048

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215048


namespace new_average_production_l215_215137

theorem new_average_production (n : ℕ) (average_past today : ℕ) (h₁ : average_past = 70) (h₂ : today = 90) (h₃ : n = 3) : 
  (average_past * n + today) / (n + 1) = 75 := by
  sorry

end new_average_production_l215_215137


namespace possible_values_of_p_l215_215291

theorem possible_values_of_p (a b c : ℝ) (h₁ : (-a + b + c) / a = (a - b + c) / b)
  (h₂ : (a - b + c) / b = (a + b - c) / c) :
  ∃ p ∈ ({-1, 8} : Set ℝ), p = (a + b) * (b + c) * (c + a) / (a * b * c) :=
by sorry

end possible_values_of_p_l215_215291


namespace total_oranges_picked_l215_215376

/-- Michaela needs 20 oranges to get full --/
def oranges_michaela_needs : ℕ := 20

/-- Cassandra needs twice as many oranges as Michaela to get full --/
def oranges_cassandra_needs : ℕ := 2 * oranges_michaela_needs

/-- After both have eaten until they are full, 30 oranges remain --/
def oranges_remaining : ℕ := 30

/-- The total number of oranges eaten by both Michaela and Cassandra --/
def oranges_eaten : ℕ := oranges_michaela_needs + oranges_cassandra_needs

/-- Prove that the total number of oranges picked from the farm is 90 --/
theorem total_oranges_picked : oranges_eaten + oranges_remaining = 90 := by
  sorry

end total_oranges_picked_l215_215376


namespace canada_population_l215_215846

theorem canada_population 
    (M : ℕ) (B : ℕ) (H : ℕ)
    (hM : M = 1000000)
    (hB : B = 2 * M)
    (hH : H = 19 * B) : 
    H = 38000000 := by
  sorry

end canada_population_l215_215846


namespace find_incorrect_statement_l215_215627

def is_opposite (a b : ℝ) := a = -b

theorem find_incorrect_statement :
  ¬∀ (a b : ℝ), (a * b < 0) → is_opposite a b := sorry

end find_incorrect_statement_l215_215627


namespace daisy_dog_toys_l215_215600

theorem daisy_dog_toys (X : ℕ) (lost_toys : ℕ) (total_toys_after_found : ℕ) : 
    (X - lost_toys + (3 + 3) - lost_toys + 5 = total_toys_after_found) → total_toys_after_found = 13 → X = 5 :=
by
  intros h1 h2
  sorry

end daisy_dog_toys_l215_215600


namespace arithmetic_sequence_a9_l215_215586

theorem arithmetic_sequence_a9 (S : ℕ → ℤ) (a : ℕ → ℤ) :
  S 8 = 4 * a 3 → a 7 = -2 → a 9 = -6 := by
  sorry

end arithmetic_sequence_a9_l215_215586


namespace mean_median_difference_l215_215704

open Real

/-- In a class of 100 students, these are the distributions of scores:
  - 10% scored 60 points
  - 30% scored 75 points
  - 25% scored 80 points
  - 20% scored 90 points
  - 15% scored 100 points

Prove that the difference between the mean and the median scores is 1.5. -/
theorem mean_median_difference :
  let total_students := 100 
  let score_60 := 0.10 * total_students
  let score_75 := 0.30 * total_students
  let score_80 := 0.25 * total_students
  let score_90 := 0.20 * total_students
  let score_100 := (100 - (score_60 + score_75 + score_80 + score_90))
  let median := 80
  let mean := (60 * score_60 + 75 * score_75 + 80 * score_80 + 90 * score_90 + 100 * score_100) / total_students
  mean - median = 1.5 :=
by
  sorry

end mean_median_difference_l215_215704


namespace roots_form_parallelogram_l215_215418

theorem roots_form_parallelogram :
  let polynomial := fun (z : ℂ) (a : ℝ) =>
    z^4 - 8*z^3 + 13*a*z^2 - 2*(3*a^2 + 2*a - 4)*z - 2
  let a1 := 7.791
  let a2 := -8.457
  ∀ z1 z2 z3 z4 : ℂ,
    ( (polynomial z1 a1 = 0) ∧ (polynomial z2 a1 = 0) ∧ (polynomial z3 a1 = 0) ∧ (polynomial z4 a1 = 0)
    ∨ (polynomial z1 a2 = 0) ∧ (polynomial z2 a2 = 0) ∧ (polynomial z3 a2 = 0) ∧ (polynomial z4 a2 = 0) )
    → ( (z1 + z2 + z3 + z4) / 4 = 2 )
    → ( Complex.abs (z1 - z2) = Complex.abs (z3 - z4) 
      ∧ Complex.abs (z1 - z3) = Complex.abs (z2 - z4) ) := sorry

end roots_form_parallelogram_l215_215418


namespace correct_article_usage_l215_215277

def sentence : String :=
  "While he was at ____ college, he took part in the march, and was soon thrown into ____ prison."

def rules_for_articles (context : String) (noun : String) : String → Bool
| "the" => noun ≠ "college" ∨ context = "specific"
| ""    => noun = "college" ∨ noun = "prison"
| _     => false

theorem correct_article_usage : 
  rules_for_articles "general" "college" "" ∧ 
  rules_for_articles "general" "prison" "" :=
by
  sorry

end correct_article_usage_l215_215277


namespace inequality_proof_l215_215150

theorem inequality_proof
  (a b c d e f : ℝ)
  (h : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
sorry

end inequality_proof_l215_215150


namespace intersection_complement_l215_215438

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x^2 < 1 }
def B : Set ℝ := { x | x^2 - 2 * x > 0 }

theorem intersection_complement (A B : Set ℝ) : 
  (A ∩ (U \ B)) = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_complement_l215_215438


namespace product_mod_32_is_15_l215_215019

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215019


namespace growth_rate_equation_l215_215897

variable (a x : ℝ)

-- Condition: The number of visitors in March is three times that of January
def visitors_in_march := 3 * a

-- Condition: The average growth rate of visitors in February and March is x
def growth_rate := x

-- Statement to prove
theorem growth_rate_equation 
  (h : (1 + x)^2 = 3) : true :=
by sorry

end growth_rate_equation_l215_215897


namespace initial_fliers_l215_215898

theorem initial_fliers (F : ℕ) (morning_sent afternoon_sent remaining : ℕ) :
  morning_sent = F / 5 → 
  afternoon_sent = (F - morning_sent) / 4 → 
  remaining = F - morning_sent - afternoon_sent → 
  remaining = 1800 → 
  F = 3000 := 
by 
  sorry

end initial_fliers_l215_215898


namespace contractor_engaged_days_l215_215255

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end contractor_engaged_days_l215_215255


namespace solve_for_x_y_l215_215366

theorem solve_for_x_y (x y : ℚ) 
  (h1 : (3 * x + 12 + 2 * y + 18 + 5 * x + 6 * y + (3 * x + y + 16)) / 5 = 60) 
  (h2 : x = 2 * y) : 
  x = 254 / 15 ∧ y = 127 / 15 :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_y_l215_215366


namespace solution_set_of_inequality_l215_215752

theorem solution_set_of_inequality :
  { x : ℝ | ∃ (h : x ≠ 1), 1 / (x - 1) ≥ -1 } = { x : ℝ | x ≤ 0 ∨ 1 < x } :=
by sorry

end solution_set_of_inequality_l215_215752


namespace mod_product_l215_215876

theorem mod_product :
  (105 * 86 * 97) % 25 = 10 :=
by
  sorry

end mod_product_l215_215876


namespace product_mod_32_is_15_l215_215013

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215013


namespace find_a_value_l215_215609

-- Define the conditions
def inverse_variation (a b : ℝ) : Prop := ∃ k : ℝ, a * b^3 = k

-- Define the proof problem
theorem find_a_value
  (a b : ℝ)
  (h1 : inverse_variation a b)
  (h2 : a = 4)
  (h3 : b = 1) :
  ∃ a', a' = 1 / 2 ∧ inverse_variation a' 2 := 
sorry

end find_a_value_l215_215609


namespace interior_and_exterior_angles_of_regular_dodecagon_l215_215623

-- Definition of a regular dodecagon
def regular_dodecagon_sides : ℕ := 12

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Measure of one interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Measure of one exterior angle of a regular polygon (180 degrees supplementary to interior angle)
def one_exterior_angle (n : ℕ) : ℕ := 180 - one_interior_angle n

-- The theorem to prove
theorem interior_and_exterior_angles_of_regular_dodecagon :
  one_interior_angle regular_dodecagon_sides = 150 ∧ one_exterior_angle regular_dodecagon_sides = 30 :=
by
  sorry

end interior_and_exterior_angles_of_regular_dodecagon_l215_215623


namespace fraction_simplification_addition_l215_215741

theorem fraction_simplification_addition :
  (∃ a b : ℕ, 0.4375 = (a : ℚ) / b ∧ Nat.gcd a b = 1 ∧ a + b = 23) :=
by
  sorry

end fraction_simplification_addition_l215_215741


namespace total_students_correct_l215_215269

noncomputable def num_roman_numerals : ℕ := 7
noncomputable def sketches_per_numeral : ℕ := 5
noncomputable def total_students : ℕ := 35

theorem total_students_correct : num_roman_numerals * sketches_per_numeral = total_students := by
  sorry

end total_students_correct_l215_215269


namespace degree_of_g_l215_215563

theorem degree_of_g 
  (f : Polynomial ℤ)
  (g : Polynomial ℤ) 
  (h₁ : f = -9 * Polynomial.X^5 + 4 * Polynomial.X^3 - 2 * Polynomial.X + 6)
  (h₂ : (f + g).degree = 2) :
  g.degree = 5 :=
sorry

end degree_of_g_l215_215563


namespace target_run_correct_l215_215455

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def run_rate_remaining_22_overs : ℝ := 11.363636363636363
def overs_remaining_22 : ℝ := 22

-- Initialize the target run calculation using the given conditions
def runs_first_10_overs := overs_first_10 * run_rate_first_10_overs
def runs_remaining_22_overs := overs_remaining_22 * run_rate_remaining_22_overs
def target_run := runs_first_10_overs + runs_remaining_22_overs 

-- The goal is to prove that the target run is 282
theorem target_run_correct : target_run = 282 := by
  sorry  -- The proof is not required as per the instructions.

end target_run_correct_l215_215455


namespace swimming_pool_time_l215_215522

theorem swimming_pool_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 3)
  (h2 : A + C = 1 / 6)
  (h3 : B + C = 1 / 4.5) :
  1 / (A + B + C) = 2.25 :=
by
  sorry

end swimming_pool_time_l215_215522


namespace g_at_three_l215_215073

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_nonzero_at_zero : g 0 ≠ 0
axiom g_at_one : g 1 = 2

theorem g_at_three : g 3 = 8 := sorry

end g_at_three_l215_215073


namespace product_of_odd_primes_mod_32_l215_215049

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215049


namespace Lisa_photos_l215_215725

variable (a f s : ℕ)

theorem Lisa_photos (h1: a = 10) (h2: f = 3 * a) (h3: s = f - 10) : a + f + s = 60 := by
  sorry

end Lisa_photos_l215_215725


namespace box_surface_area_is_276_l215_215521

-- Define the dimensions of the box
variables {l w h : ℝ}

-- Define the pricing function
def pricing (x y z : ℝ) : ℝ := 0.30 * x + 0.40 * y + 0.50 * z

-- Define the condition for the box fee
def box_fee (x y z : ℝ) (fee : ℝ) := pricing x y z = fee

-- Define the constraint that no faces are squares
def no_square_faces (l w h : ℝ) : Prop := 
  l ≠ w ∧ w ≠ h ∧ h ≠ l

-- Define the surface area calculation
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- The main theorem stating the problem
theorem box_surface_area_is_276 (l w h : ℝ) 
  (H1 : box_fee l w h 8.10 ∧ box_fee w h l 8.10)
  (H2 : box_fee l w h 8.70 ∧ box_fee w h l 8.70)
  (H3 : no_square_faces l w h) : 
  surface_area l w h = 276 := 
sorry

end box_surface_area_is_276_l215_215521


namespace square_area_l215_215929

theorem square_area (l w x : ℝ) (h1 : 2 * (l + w) = 20) (h2 : l = x / 2) (h3 : w = x / 4) :
  x^2 = 1600 / 9 :=
by
  sorry

end square_area_l215_215929


namespace trajectory_C_find_m_l215_215164

noncomputable def trajectory_C_eq (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 7

theorem trajectory_C (x y : ℝ) (hx : trajectory_C_eq x y) :
  (x - 3)^2 + y^2 = 7 := by
  sorry

theorem find_m (m : ℝ) : (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = 3 + m ∧ x1 * x2 + (1/(2:ℝ)) * ((m^2 + 2)/(2:ℝ)) = 0 ∧ x1 * x2 + (x1 - m) * (x2 - m) = 0) → m = 1 ∨ m = 2 := by
  sorry

end trajectory_C_find_m_l215_215164


namespace alley_width_l215_215452

theorem alley_width (L w : ℝ) (k h : ℝ)
    (h1 : k = L / 2)
    (h2 : h = L * (Real.sqrt 3) / 2)
    (h3 : w^2 + (L / 2)^2 = L^2)
    (h4 : w^2 + (L * (Real.sqrt 3) / 2)^2 = L^2):
    w = (Real.sqrt 3) * L / 2 := 
sorry

end alley_width_l215_215452


namespace least_non_lucky_multiple_of_8_l215_215398

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

def lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def multiple_of_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem least_non_lucky_multiple_of_8 : ∃ n > 0, multiple_of_8 n ∧ ¬ lucky n ∧ n = 16 :=
by
  -- Proof goes here.
  sorry

end least_non_lucky_multiple_of_8_l215_215398


namespace bee_honeycomb_path_l215_215860

theorem bee_honeycomb_path (x1 x2 x3 : ℕ) (honeycomb_grid : Prop)
  (shortest_path : ℕ) (honeycomb_property : shortest_path = 100)
  (path_decomposition : x1 + x2 + x3 = 100) : x1 = 50 ∧ x2 + x3 = 50 := 
sorry

end bee_honeycomb_path_l215_215860


namespace s9_s3_ratio_l215_215674

variable {a_n : ℕ → ℝ}
variable {s_n : ℕ → ℝ}
variable {a : ℝ}

-- Conditions
axiom h_s6_s3_ratio : s_n 6 / s_n 3 = 1 / 2

-- Theorem to prove
theorem s9_s3_ratio (h : s_n 3 = a) : s_n 9 / s_n 3 = 3 / 4 := 
sorry

end s9_s3_ratio_l215_215674


namespace problem_1_problem_2_l215_215561

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ :=
( sin θ, cos θ - 2 * sin θ )

def vec_b : ℝ × ℝ :=
( 1, 2 )

theorem problem_1 (θ : ℝ) (h : (cos θ - 2 * sin θ) / sin θ = 2) : tan θ = 1 / 4 :=
by {
  sorry
}

theorem problem_2 (θ : ℝ) (h1 : sin θ ^ 2 + (cos θ - 2 * sin θ) ^ 2 = 5) (h2 : 0 < θ) (h3 : θ < π) : θ = π / 2 ∨ θ = 3 * π / 4 :=
by {
  sorry
}

end problem_1_problem_2_l215_215561


namespace team_A_has_more_uniform_heights_l215_215632

-- Definitions of the conditions
def avg_height_team_A : ℝ := 1.65
def avg_height_team_B : ℝ := 1.65

def variance_team_A : ℝ := 1.5
def variance_team_B : ℝ := 2.4

-- Theorem stating the problem solution
theorem team_A_has_more_uniform_heights :
  variance_team_A < variance_team_B :=
by
  -- Proof omitted
  sorry

end team_A_has_more_uniform_heights_l215_215632


namespace fernanda_savings_calculation_l215_215647

theorem fernanda_savings_calculation :
  ∀ (aryan_debt kyro_debt aryan_payment kyro_payment savings total_savings : ℝ),
    aryan_debt = 1200 ∧
    aryan_debt = 2 * kyro_debt ∧
    aryan_payment = (60 / 100) * aryan_debt ∧
    kyro_payment = (80 / 100) * kyro_debt ∧
    savings = 300 ∧
    total_savings = savings + aryan_payment + kyro_payment →
    total_savings = 1500 := by
    sorry

end fernanda_savings_calculation_l215_215647


namespace largest_integer_mod_l215_215419

theorem largest_integer_mod (a : ℕ) (h₁ : a < 100) (h₂ : a % 5 = 2) : a = 97 :=
by sorry

end largest_integer_mod_l215_215419


namespace cube_painted_probability_l215_215519

theorem cube_painted_probability :
  let length := 20
  let width := 1
  let height := 7
  let total_cubes := length * width * height
  let corner_cubes := 8
  let edge_cubes := 4 * (length - 2) + 8 * (height - 2)
  let face_cubes := (length * height) - edge_cubes - corner_cubes
  let corner_prob := (corner_cubes / total_cubes : Rat) * (3 / 6 : Rat)
  let edge_prob := (edge_cubes / total_cubes : Rat) * (2 / 6 : Rat)
  let face_prob := (face_cubes / total_cubes : Rat) * (1 / 6 : Rat)
  corner_prob + edge_prob + face_prob = 9 / 35 := by
  sorry

end cube_painted_probability_l215_215519


namespace product_mod_32_is_15_l215_215017

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215017


namespace nails_remaining_proof_l215_215516

noncomputable
def remaining_nails (initial_nails kitchen_percent fence_percent : ℕ) : ℕ :=
  let kitchen_used := initial_nails * kitchen_percent / 100
  let remaining_after_kitchen := initial_nails - kitchen_used
  let fence_used := remaining_after_kitchen * fence_percent / 100
  let final_remaining := remaining_after_kitchen - fence_used
  final_remaining

theorem nails_remaining_proof :
  remaining_nails 400 30 70 = 84 := by
  sorry

end nails_remaining_proof_l215_215516


namespace odd_power_divisible_by_sum_l215_215894

theorem odd_power_divisible_by_sum (x y : ℝ) (k : ℕ) (h : k > 0) :
  (x^((2*k - 1)) + y^((2*k - 1))) ∣ (x^(2*k + 1) + y^(2*k + 1)) :=
sorry

end odd_power_divisible_by_sum_l215_215894


namespace translate_line_upwards_l215_215217

-- Define the original line equation
def original_line_eq (x : ℝ) : ℝ := 3 * x - 3

-- Define the translation operation
def translate_upwards (y_translation : ℝ) (line_eq : ℝ → ℝ) (x : ℝ) : ℝ :=
  line_eq x + y_translation

-- Define the proof problem
theorem translate_line_upwards :
  ∀ (x : ℝ), translate_upwards 5 original_line_eq x = 3 * x + 2 :=
by
  intros x
  simp [translate_upwards, original_line_eq]
  sorry

end translate_line_upwards_l215_215217


namespace perfect_square_l215_215101

theorem perfect_square (a b : ℝ) : a^2 + 2 * a * b + b^2 = (a + b)^2 := by
  sorry

end perfect_square_l215_215101


namespace train_crossing_time_is_correct_l215_215562

-- Define the constant values
def train_length : ℝ := 350        -- Train length in meters
def train_speed : ℝ := 20          -- Train speed in m/s
def crossing_time : ℝ := 17.5      -- Time to cross the signal post in seconds

-- Proving the relationship that the time taken for the train to cross the signal post is as calculated
theorem train_crossing_time_is_correct : (train_length / train_speed) = crossing_time :=
by
  sorry

end train_crossing_time_is_correct_l215_215562


namespace fraction_percent_l215_215179

theorem fraction_percent (x : ℝ) (h : x > 0) : ((x / 10 + x / 25) / x) * 100 = 14 :=
by
  sorry

end fraction_percent_l215_215179


namespace determinant_example_l215_215790

def determinant_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

theorem determinant_example : determinant_2x2 5 (-4) 2 3 = 23 := 
by 
  sorry

end determinant_example_l215_215790


namespace product_of_odd_primes_mod_32_l215_215046

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215046


namespace floor_inequality_sqrt_l215_215863

theorem floor_inequality_sqrt (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (⌊ m * Real.sqrt 2 ⌋) * (⌊ n * Real.sqrt 7 ⌋) < (⌊ m * n * Real.sqrt 14 ⌋) := 
by
  sorry

end floor_inequality_sqrt_l215_215863


namespace minimum_nine_points_distance_l215_215573

theorem minimum_nine_points_distance (n : ℕ) : 
  (∀ (p : Fin n → ℝ × ℝ),
    (∀ i, ∃! (four_points : List (Fin n)), 
      List.length four_points = 4 ∧ (∀ j ∈ four_points, dist (p i) (p j) = 1)))
    ↔ n = 9 :=
by 
  sorry

end minimum_nine_points_distance_l215_215573


namespace parabola_focus_coordinates_l215_215735

theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 - 4 * x = 0 → (x, y) = (1, 0) :=
by
  -- Use the equivalence given by the problem
  intros x y h
  sorry

end parabola_focus_coordinates_l215_215735


namespace quadratic_inequality_false_iff_range_of_a_l215_215570

theorem quadratic_inequality_false_iff_range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ (-1 < a ∧ a < 3) :=
sorry

end quadratic_inequality_false_iff_range_of_a_l215_215570


namespace div_polynomial_not_div_l215_215820

theorem div_polynomial_not_div (n : ℕ) : ¬ (n + 2) ∣ (n^3 - 2 * n^2 - 5 * n + 7) := by
  sorry

end div_polynomial_not_div_l215_215820


namespace washing_machine_capacity_l215_215599

def num_shirts : Nat := 19
def num_sweaters : Nat := 8
def num_loads : Nat := 3

theorem washing_machine_capacity :
  (num_shirts + num_sweaters) / num_loads = 9 := by
  sorry

end washing_machine_capacity_l215_215599


namespace overall_gain_loss_percent_zero_l215_215230

theorem overall_gain_loss_percent_zero (CP_A CP_B CP_C SP_A SP_B SP_C : ℝ)
  (h1 : CP_A = 600) (h2 : CP_B = 700) (h3 : CP_C = 800)
  (h4 : SP_A = 450) (h5 : SP_B = 750) (h6 : SP_C = 900) :
  ((SP_A + SP_B + SP_C) - (CP_A + CP_B + CP_C)) / (CP_A + CP_B + CP_C) * 100 = 0 :=
by
  sorry

end overall_gain_loss_percent_zero_l215_215230


namespace product_mod_32_l215_215010

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215010


namespace least_multiple_x_correct_l215_215991

noncomputable def least_multiple_x : ℕ :=
  let x := 20
  let y := 8
  let z := 5
  5 * y

theorem least_multiple_x_correct (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 33) (h5 : 5 * y = 8 * z) : least_multiple_x = 40 :=
by
  sorry

end least_multiple_x_correct_l215_215991


namespace product_of_values_t_squared_eq_49_l215_215808

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l215_215808


namespace katie_sold_4_bead_necklaces_l215_215341

theorem katie_sold_4_bead_necklaces :
  ∃ (B : ℕ), 
    (∃ (G : ℕ), G = 3) ∧ 
    (∃ (C : ℕ), C = 3) ∧ 
    (∃ (T : ℕ), T = 21) ∧ 
    B * 3 + 3 * 3 = 21 :=
sorry

end katie_sold_4_bead_necklaces_l215_215341


namespace probability_of_both_contracts_l215_215743

open Classical

variable (P_A P_B' P_A_or_B P_A_and_B : ℚ)

noncomputable def probability_hardware_contract := P_A = 3 / 4
noncomputable def probability_not_software_contract := P_B' = 5 / 9
noncomputable def probability_either_contract := P_A_or_B = 4 / 5
noncomputable def probability_both_contracts := P_A_and_B = 71 / 180

theorem probability_of_both_contracts {P_A P_B' P_A_or_B P_A_and_B : ℚ} :
  probability_hardware_contract P_A →
  probability_not_software_contract P_B' →
  probability_either_contract P_A_or_B →
  probability_both_contracts P_A_and_B :=
by
  intros
  sorry

end probability_of_both_contracts_l215_215743


namespace number_of_players_knight_moves_friend_not_winner_l215_215184

-- Problem (a)
theorem number_of_players (sum_scores : ℕ) (h : sum_scores = 210) : 
  ∃ x : ℕ, x * (x - 1) = 210 :=
sorry

-- Problem (b)
theorem knight_moves (initial_positions : ℕ) (wrong_guess : ℕ) (correct_answer : ℕ) : 
  initial_positions = 1 ∧ wrong_guess = 64 ∧ correct_answer = 33 → 
  ∃ squares : ℕ, squares = 33 :=
sorry

-- Problem (c)
theorem friend_not_winner (total_scores : ℕ) (num_players : ℕ) (friend_score : ℕ) (avg_score : ℕ) : 
  total_scores = 210 ∧ num_players = 15 ∧ friend_score = 12 ∧ avg_score = 14 → 
  ∃ higher_score : ℕ, higher_score > friend_score :=
sorry

end number_of_players_knight_moves_friend_not_winner_l215_215184


namespace matrix_det_example_l215_215789

variable (A : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A = ![![5, -4], ![2, 3]])

theorem matrix_det_example : Matrix.det A = 23 :=
by
  sorry

end matrix_det_example_l215_215789


namespace find_A_from_conditions_l215_215108

variable (A B C D : ℕ)
variable (h_distinct : A ≠ B) (h_distinct2 : C ≠ D)
variable (h_positive : A > 0) (h_positive2 : B > 0) (h_positive3 : C > 0) (h_positive4 : D > 0)
variable (h_product1 : A * B = 72)
variable (h_product2 : C * D = 72)
variable (h_condition : A - B = C * D)

theorem find_A_from_conditions :
  A = 3 :=
sorry

end find_A_from_conditions_l215_215108


namespace next_four_customers_cases_l215_215616

theorem next_four_customers_cases (total_people : ℕ) (first_eight_cases : ℕ) (last_eight_cases : ℕ) (total_cases : ℕ) :
    total_people = 20 →
    first_eight_cases = 24 →
    last_eight_cases = 8 →
    total_cases = 40 →
    (total_cases - (first_eight_cases + last_eight_cases)) / 4 = 2 :=
by
  intro h1 h2 h3 h4
  -- Fill in the proof steps using h1, h2, h3, and h4
  sorry

end next_four_customers_cases_l215_215616


namespace product_of_values_t_squared_eq_49_l215_215807

theorem product_of_values_t_squared_eq_49 : 
  (∀ t : ℝ, t^2 = 49 → (t = 7 ∨ t = -7) ∧ (7 * -7 = -49)) := 
by {
  intro t,
  intro ht,
  split,
  {
    cases ht with ht1 ht2,
    {
      constructor,
      by { sorry },
      by { sorry }
    },
    by { sorry }
  },
  by {
    sorry
  }
}

end product_of_values_t_squared_eq_49_l215_215807


namespace product_of_odd_primes_mod_32_l215_215023

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215023


namespace range_of_n_l215_215328

theorem range_of_n (x : ℕ) (n : ℝ) : 
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 5 → x - 2 < n + 3) → ∃ n, 0 < n ∧ n ≤ 1 :=
by
  sorry

end range_of_n_l215_215328


namespace inequality_proof_l215_215151

theorem inequality_proof
  (a b c d e f : ℝ)
  (h : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
sorry

end inequality_proof_l215_215151


namespace arccos_neg_one_eq_pi_proof_l215_215937

noncomputable def arccos_neg_one_eq_pi : Prop :=
  arccos (-1) = π

theorem arccos_neg_one_eq_pi_proof : arccos_neg_one_eq_pi := by
  sorry

end arccos_neg_one_eq_pi_proof_l215_215937


namespace junior_titles_in_sample_l215_215512

noncomputable def numberOfJuniorTitlesInSample (totalEmployees: ℕ) (juniorEmployees: ℕ) (sampleSize: ℕ) : ℕ :=
  (juniorEmployees * sampleSize) / totalEmployees

theorem junior_titles_in_sample (totalEmployees juniorEmployees intermediateEmployees seniorEmployees sampleSize : ℕ) 
  (h_total : totalEmployees = 150) 
  (h_junior : juniorEmployees = 90) 
  (h_intermediate : intermediateEmployees = 45) 
  (h_senior : seniorEmployees = 15) 
  (h_sampleSize : sampleSize = 30) : 
  numberOfJuniorTitlesInSample totalEmployees juniorEmployees sampleSize = 18 := by
  sorry

end junior_titles_in_sample_l215_215512


namespace find_prime_triples_l215_215800

theorem find_prime_triples :
  ∀ p x y : ℤ, p.prime ∧ x > 0 ∧ y > 0 ∧ p^x = y^3 + 1 →
  (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) := by
  sorry

end find_prime_triples_l215_215800


namespace find_all_n_l215_215957

theorem find_all_n (n : ℕ) : 
  (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % n = 0) ↔ (∃ j : ℕ, n = 3^j) :=
by 
  -- proof goes here
  sorry

end find_all_n_l215_215957


namespace bobby_jumps_per_second_as_adult_l215_215124

-- Define the conditions as variables
def child_jumps_per_minute : ℕ := 30
def additional_jumps_as_adult : ℕ := 30

-- Theorem statement
theorem bobby_jumps_per_second_as_adult :
  (child_jumps_per_minute + additional_jumps_as_adult) / 60 = 1 :=
by
  -- placeholder for the proof
  sorry

end bobby_jumps_per_second_as_adult_l215_215124


namespace joshInitialMarbles_l215_215721

-- Let n be the number of marbles Josh initially had
variable (n : ℕ)

-- Condition 1: Jack gave Josh 20 marbles
def jackGaveJoshMarbles : ℕ := 20

-- Condition 2: Now Josh has 42 marbles
def joshCurrentMarbles : ℕ := 42

-- Theorem: prove that the number of marbles Josh had initially was 22
theorem joshInitialMarbles : n + jackGaveJoshMarbles = joshCurrentMarbles → n = 22 :=
by
  intros h
  sorry

end joshInitialMarbles_l215_215721


namespace quadratic_roots_real_distinct_l215_215693

theorem quadratic_roots_real_distinct (k : ℝ) (h : k < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + x1 + k - 1 = 0) ∧ (x2^2 + x2 + k - 1 = 0) :=
by
  sorry

end quadratic_roots_real_distinct_l215_215693


namespace percentage_of_students_wearing_blue_shirts_l215_215706

theorem percentage_of_students_wearing_blue_shirts :
  ∀ (total_students red_percent green_percent students_other_colors : ℕ),
  total_students = 800 →
  red_percent = 23 →
  green_percent = 15 →
  students_other_colors = 136 →
  ((total_students - students_other_colors) - (red_percent + green_percent) = 45) :=
by
  intros total_students red_percent green_percent students_other_colors h_total h_red h_green h_other
  have h_other_percent : (students_other_colors * 100 / total_students) = 17 :=
    sorry
  exact sorry

end percentage_of_students_wearing_blue_shirts_l215_215706


namespace necessary_condition_l215_215545

variables (a b : ℝ)

theorem necessary_condition (h : a > b) : a > b - 1 :=
sorry

end necessary_condition_l215_215545


namespace largest_k_no_perpendicular_lines_l215_215194

theorem largest_k_no_perpendicular_lines (n : ℕ) (h : 0 < n) :
  (∃ k, ∀ (l : Fin n → ℝ) (f : Fin n), (∀ i j, i ≠ j → l i ≠ -1 / (l j)) → k = Nat.ceil (n / 2)) :=
sorry

end largest_k_no_perpendicular_lines_l215_215194


namespace daffodil_bulb_cost_l215_215102

theorem daffodil_bulb_cost :
  let total_bulbs := 55
  let crocus_cost := 0.35
  let total_budget := 29.15
  let num_crocus_bulbs := 22
  let total_crocus_cost := num_crocus_bulbs * crocus_cost
  let remaining_budget := total_budget - total_crocus_cost
  let num_daffodil_bulbs := total_bulbs - num_crocus_bulbs
  remaining_budget / num_daffodil_bulbs = 0.65 := 
by
  -- proof to be filled in
  sorry

end daffodil_bulb_cost_l215_215102


namespace derek_walk_time_l215_215582

theorem derek_walk_time (x : ℕ) :
  (∀ y : ℕ, (y = 9) → (∀ d₁ d₂ : ℕ, (d₁ = 20 ∧ d₂ = 60) →
    (20 * x = d₁ * y + d₂))) → x = 12 :=
by
  intro h
  sorry

end derek_walk_time_l215_215582


namespace diego_apples_weight_l215_215409

-- Definitions based on conditions
def bookbag_capacity : ℕ := 20
def weight_watermelon : ℕ := 1
def weight_grapes : ℕ := 1
def weight_oranges : ℕ := 1

-- Lean statement to check
theorem diego_apples_weight : 
  bookbag_capacity - (weight_watermelon + weight_grapes + weight_oranges) = 17 :=
by
  sorry

end diego_apples_weight_l215_215409


namespace total_days_on_island_correct_l215_215853

-- Define the first, second, and third expeditions
def firstExpedition : ℕ := 3

def secondExpedition (a : ℕ) : ℕ := a + 2

def thirdExpedition (b : ℕ) : ℕ := 2 * b

-- Define the total duration in weeks
def totalWeeks : ℕ := firstExpedition + secondExpedition firstExpedition + thirdExpedition (secondExpedition firstExpedition)

-- Define the total days spent on the island
def totalDays (weeks : ℕ) : ℕ := weeks * 7

-- Prove that the total number of days spent is 126
theorem total_days_on_island_correct : totalDays totalWeeks = 126 := 
  by
    sorry

end total_days_on_island_correct_l215_215853


namespace people_after_second_turn_l215_215479

noncomputable def number_of_people_in_front_after_second_turn (formation_size : ℕ) (initial_people : ℕ) (first_turn_people : ℕ) : ℕ := 
  if formation_size = 9 ∧ initial_people = 2 ∧ first_turn_people = 4 then 6 else 0

theorem people_after_second_turn :
  number_of_people_in_front_after_second_turn 9 2 4 = 6 :=
by
  -- Prove the theorem using the conditions and given data
  sorry

end people_after_second_turn_l215_215479


namespace Alan_total_cost_is_84_l215_215640

theorem Alan_total_cost_is_84 :
  let D := 2 * 12
  let A := 12
  let cost_other := 2 * D + A
  let M := 0.4 * cost_other
  2 * D + A + M = 84 := by
    sorry

end Alan_total_cost_is_84_l215_215640


namespace micah_total_strawberries_l215_215204

theorem micah_total_strawberries (eaten saved total : ℕ) 
  (h1 : eaten = 6) 
  (h2 : saved = 18) 
  (h3 : total = eaten + saved) : 
  total = 24 := 
by
  sorry

end micah_total_strawberries_l215_215204


namespace zero_product_property_l215_215692

theorem zero_product_property {a b : ℝ} (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end zero_product_property_l215_215692


namespace ages_of_people_l215_215526

-- Define types
variable (A M B C : ℕ)

-- Define conditions as hypotheses
def conditions : Prop :=
  A = 2 * M ∧
  A = 4 * B ∧
  M = A - 10 ∧
  C = B + 3 ∧
  C = M / 2

-- Define what we want to prove
theorem ages_of_people :
  (conditions A M B C) →
  A = 20 ∧
  M = 10 ∧
  B = 2 ∧
  C = 5 :=
by
  sorry

end ages_of_people_l215_215526


namespace copy_pages_cost_l215_215458

theorem copy_pages_cost :
  (7 : ℕ) * (n : ℕ) = 3500 * 4 / 7 → n = 2000 :=
by
  sorry

end copy_pages_cost_l215_215458


namespace simplification_l215_215606

theorem simplification (a b c : ℤ) :
  (12 * a + 35 * b + 17 * c) + (13 * a - 15 * b + 8 * c) - (8 * a + 28 * b - 25 * c) = 17 * a - 8 * b + 50 * c :=
by
  sorry

end simplification_l215_215606


namespace correct_equation_l215_215386

theorem correct_equation : -(-5) = |-5| :=
by
  -- sorry is used here to skip the actual proof steps which are not required.
  sorry

end correct_equation_l215_215386


namespace product_of_odd_primes_mod_32_l215_215030

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215030


namespace total_price_of_shoes_l215_215335

theorem total_price_of_shoes
  (S J : ℝ) 
  (h1 : 6 * S + 4 * J = 560) 
  (h2 : J = S / 4) :
  6 * S = 480 :=
by 
  -- Begin the proof environment
  sorry -- Placeholder for the actual proof

end total_price_of_shoes_l215_215335


namespace inequality_problem_l215_215154

theorem inequality_problem
  (a b c d e f : ℝ)
  (h_cond : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
by
  sorry

end inequality_problem_l215_215154


namespace diameter_of_circle_given_radius_l215_215568

theorem diameter_of_circle_given_radius (radius: ℝ) (h: radius = 7): 
  2 * radius = 14 :=
by
  rw [h]
  sorry

end diameter_of_circle_given_radius_l215_215568


namespace quadratic_expression_result_l215_215975

theorem quadratic_expression_result (x y : ℚ) 
  (h1 : 4 * x + y = 11) 
  (h2 : x + 4 * y = 15) : 
  13 * x^2 + 14 * x * y + 13 * y^2 = 275.2 := 
by 
  sorry

end quadratic_expression_result_l215_215975


namespace hyperbola_focal_distance_and_asymptotes_l215_215435

-- Define the hyperbola
def hyperbola (y x : ℝ) : Prop := (y^2 / 4) - (x^2 / 3) = 1

-- Prove the properties
theorem hyperbola_focal_distance_and_asymptotes :
  (∀ y x : ℝ, hyperbola y x → ∃ c : ℝ, c = 2 * Real.sqrt 7)
  ∧
  (∀ y x : ℝ, hyperbola y x → (y = (2 * Real.sqrt 3 / 3) * x ∨ y = -(2 * Real.sqrt 3 / 3) * x)) :=
by
  sorry

end hyperbola_focal_distance_and_asymptotes_l215_215435


namespace distance_of_course_l215_215891

-- Definitions
def teamESpeed : ℕ := 20
def teamASpeed : ℕ := teamESpeed + 5

-- Time taken by Team E
variable (tE : ℕ)

-- Distance calculation
def teamEDistance : ℕ := teamESpeed * tE
def teamADistance : ℕ := teamASpeed * (tE - 3)

-- Proof statement
theorem distance_of_course (tE : ℕ) (h : teamEDistance tE = teamADistance tE) : teamEDistance tE = 300 :=
sorry

end distance_of_course_l215_215891


namespace rationalize_denominator_correct_l215_215353

noncomputable def rationalize_denominator (x : ℝ) := 
  (5 : ℝ) / (3 * real.cbrt 7) * (real.cbrt 49) / (real.cbrt 49)

theorem rationalize_denominator_correct : 
  rationalize_denominator (5 / (3 * real.cbrt 7)) = (5 * real.cbrt 49) / (21 : ℝ) 
  ∧ 5 + 49 + 21 = 75 := 
by
  sorry

end rationalize_denominator_correct_l215_215353


namespace infinite_series_correct_l215_215408

noncomputable def infinite_series_sum : ℚ := 
  ∑' n : ℕ, (n+1)^2 * (1/999)^n

theorem infinite_series_correct : infinite_series_sum = 997005 / 996004 :=
  sorry

end infinite_series_correct_l215_215408


namespace smallest_even_n_for_reducible_fraction_l215_215781

theorem smallest_even_n_for_reducible_fraction : 
  ∃ (N: ℕ), (N > 2013) ∧ (N % 2 = 0) ∧ (Nat.gcd (15 * N - 7) (22 * N - 5) > 1) ∧ N = 2144 :=
sorry

end smallest_even_n_for_reducible_fraction_l215_215781


namespace solve_for_y_l215_215357

theorem solve_for_y (y : ℝ) (h : ∛(5 - 2 / y) = -3) : y = 1/16 :=
sorry

end solve_for_y_l215_215357


namespace extreme_value_at_1_l215_215682

theorem extreme_value_at_1 (a b : ℝ) (h1 : (deriv (λ x => x^3 + a * x^2 + b * x + a^2) 1 = 0))
(h2 : (1 + a + b + a^2 = 10)) : a + b = -7 := by
  sorry

end extreme_value_at_1_l215_215682


namespace simplify_fractions_l215_215607

theorem simplify_fractions :
  (20 / 19) * (15 / 28) * (76 / 45) = 95 / 84 :=
by
  sorry

end simplify_fractions_l215_215607


namespace radius_increase_125_surface_area_l215_215488

theorem radius_increase_125_surface_area (r r' : ℝ) 
(increase_surface_area : 4 * π * (r'^2) = 2.25 * 4 * π * r^2) : r' = 1.5 * r :=
by 
  sorry

end radius_increase_125_surface_area_l215_215488


namespace product_of_odd_primes_mod_32_l215_215024

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215024


namespace lines_parallel_lines_perpendicular_l215_215300

-- Definition of lines
def l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a ^ 2 - 1 = 0

-- Parallel condition proof problem
theorem lines_parallel (a : ℝ) : (a = -1) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y →  
        (-(a / 2) = (1 / (1 - a))) ∧ (-3 ≠ -a - 1) :=
by
  intros
  sorry

-- Perpendicular condition proof problem
theorem lines_perpendicular (a : ℝ) : (a = 2 / 3) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y → 
        (- (a / 2) * (1 / (1 - a)) = -1) :=
by
  intros
  sorry

end lines_parallel_lines_perpendicular_l215_215300


namespace solution_set_of_inequality_l215_215886

theorem solution_set_of_inequality :
  {x : ℝ | x^2 * (x - 4) ≥ 0} = {x : ℝ | x = 0 ∨ x ≥ 4} :=
by
  sorry

end solution_set_of_inequality_l215_215886


namespace original_price_of_suit_l215_215219

theorem original_price_of_suit (P : ℝ) (h : 0.96 * P = 144) : P = 150 :=
sorry

end original_price_of_suit_l215_215219


namespace machineB_produces_100_parts_in_40_minutes_l215_215592

-- Define the given conditions
def machineA_rate := 50 / 10 -- Machine A's rate in parts per minute
def machineB_rate := machineA_rate / 2 -- Machine B's rate in parts per minute

-- Machine A produces 50 parts in 10 minutes
def machineA_50_parts_time : ℝ := 10

-- Machine B's time to produce 100 parts (The question)
def machineB_100_parts_time : ℝ := 40

-- Proving that Machine B takes 40 minutes to produce 100 parts
theorem machineB_produces_100_parts_in_40_minutes :
    machineB_100_parts_time = 40 :=
by
  sorry

end machineB_produces_100_parts_in_40_minutes_l215_215592


namespace contractor_engaged_days_l215_215257

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end contractor_engaged_days_l215_215257


namespace probability_of_consecutive_blocks_drawn_l215_215395

theorem probability_of_consecutive_blocks_drawn :
  let total_ways := (Nat.factorial 12)
  let favorable_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5) * (Nat.factorial 3)
  (favorable_ways / total_ways) = 1 / 4620 :=
by
  sorry

end probability_of_consecutive_blocks_drawn_l215_215395


namespace solve_quadratic_inequality_l215_215222

theorem solve_quadratic_inequality :
  ∀ x : ℝ, ((x - 1) * (x - 3) < 0) ↔ (1 < x ∧ x < 3) :=
by
  intro x
  sorry

end solve_quadratic_inequality_l215_215222


namespace driving_time_equation_l215_215267

theorem driving_time_equation :
  ∀ (t : ℝ), (60 * t + 90 * (3.5 - t) = 300) :=
by
  intro t
  sorry

end driving_time_equation_l215_215267


namespace rhombus_condition_perimeter_rhombus_given_ab_l215_215315

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

end rhombus_condition_perimeter_rhombus_given_ab_l215_215315


namespace solve_for_t_l215_215732

open Real

noncomputable def solve_t (t : ℝ) := 4 * (4^t) + sqrt (16 * (16^t))

theorem solve_for_t : ∃ t : ℝ, solve_t t = 32 := 
    exists.intro 1 sorry

end solve_for_t_l215_215732


namespace cube_diagonal_length_l215_215916

theorem cube_diagonal_length
  (side_length : ℝ)
  (h_side_length : side_length = 15) :
  ∃ d : ℝ, d = side_length * Real.sqrt 3 :=
by
  sorry

end cube_diagonal_length_l215_215916


namespace rectangle_perimeter_ratio_l215_215363

theorem rectangle_perimeter_ratio
    (initial_height : ℕ)
    (initial_width : ℕ)
    (H_initial_height : initial_height = 2)
    (H_initial_width : initial_width = 4)
    (fold1_height : ℕ)
    (fold1_width : ℕ)
    (H_fold1_height : fold1_height = initial_height / 2)
    (H_fold1_width : fold1_width = initial_width)
    (fold2_height : ℕ)
    (fold2_width : ℕ)
    (H_fold2_height : fold2_height = fold1_height)
    (H_fold2_width : fold2_width = fold1_width / 2)
    (cut_height : ℕ)
    (cut_width : ℕ)
    (H_cut_height : cut_height = fold2_height)
    (H_cut_width : cut_width = fold2_width) :
    (2 * (cut_height + cut_width)) / (2 * (fold1_height + fold1_width)) = 3 / 5 := 
    by sorry

end rectangle_perimeter_ratio_l215_215363


namespace hot_dogs_served_for_dinner_l215_215920

theorem hot_dogs_served_for_dinner
  (l t : ℕ) 
  (h_cond1 : l = 9) 
  (h_cond2 : t = 11) :
  ∃ d : ℕ, d = t - l ∧ d = 2 := by
  sorry

end hot_dogs_served_for_dinner_l215_215920


namespace number_of_sides_of_polygon_l215_215090

theorem number_of_sides_of_polygon (n : ℕ) : (n - 2) * 180 = 720 → n = 6 :=
by
  sorry

end number_of_sides_of_polygon_l215_215090


namespace cellphone_gifting_l215_215535

theorem cellphone_gifting (n m : ℕ) (h1 : n = 20) (h2 : m = 3) : 
    (Finset.range n).card * (Finset.range (n - 1)).card * (Finset.range (n - 2)).card = 6840 := by
  sorry

end cellphone_gifting_l215_215535


namespace product_of_odd_primes_mod_32_l215_215047

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215047


namespace share_of_B_l215_215604

noncomputable def problem_statement (A B C : ℝ) : Prop :=
  A + B + C = 595 ∧ A = (2/3) * B ∧ B = (1/4) * C

theorem share_of_B (A B C : ℝ) (h : problem_statement A B C) : B = 105 :=
by
  -- Proof omitted
  sorry

end share_of_B_l215_215604


namespace coplanar_vertices_sum_even_l215_215413

theorem coplanar_vertices_sum_even (a b c d e f g h : ℤ) :
  (∃ (a b c d : ℤ), true ∧ (a + b + c + d) % 2 = 0) :=
sorry

end coplanar_vertices_sum_even_l215_215413


namespace each_partner_percentage_l215_215218

-- Defining the conditions as variables
variables (total_profit majority_share combined_amount : ℝ) (num_partners : ℕ)

-- Given conditions
def majority_owner_received_25_percent_of_total : total_profit * 0.25 = majority_share := sorry
def remaining_profit_distribution : total_profit - majority_share = 60000 := sorry
def combined_share_of_three : majority_share + 30000 = combined_amount := sorry
def total_profit_amount : total_profit = 80000 := sorry
def number_of_partners : num_partners = 4 := sorry

-- The theorem to be proven
theorem each_partner_percentage :
  ∃ (percent : ℝ), percent = 25 :=
sorry

end each_partner_percentage_l215_215218


namespace intersection_points_count_l215_215406

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := |3 * x + 6|
def f2 (x : ℝ) : ℝ := -|4 * x - 4|

-- Prove the number of intersection points is 2
theorem intersection_points_count : 
  (∃ x1 y1, (f1 x1 = y1) ∧ (f2 x1 = y1)) ∧ 
  (∃ x2 y2, (f1 x2 = y2) ∧ (f2 x2 = y2) ∧ x1 ≠ x2) :=
sorry

end intersection_points_count_l215_215406


namespace wait_at_least_15_seconds_probability_l215_215650

-- Define the duration of the red light
def red_light_duration : ℕ := 40

-- Define the minimum waiting time for the green light
def min_wait_time : ℕ := 15

-- Define the duration after which pedestrian does not need to wait 15 seconds
def max_arrival_time : ℕ := red_light_duration - min_wait_time

-- Lean statement to prove the required probability
theorem wait_at_least_15_seconds_probability :
  (max_arrival_time : ℝ) / red_light_duration = 5 / 8 :=
by
  -- Proof omitted with sorry
  sorry

end wait_at_least_15_seconds_probability_l215_215650


namespace max_value_function_max_value_expression_l215_215110

theorem max_value_function (x a : ℝ) (hx : x > 0) (ha : a > 2 * x) : ∃ y : ℝ, y = (a^2) / 8 :=
by
  sorry

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 4) : 
   ∃ m : ℝ, m = 4 :=
by
  sorry

end max_value_function_max_value_expression_l215_215110


namespace cylinder_ellipse_major_axis_l215_215116

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ), r = 2 →
  ∀ (minor_axis : ℝ), minor_axis = 2 * r →
  ∀ (major_axis : ℝ), major_axis = 1.4 * minor_axis →
  major_axis = 5.6 :=
by
  intros r hr minor_axis hminor major_axis hmajor
  sorry

end cylinder_ellipse_major_axis_l215_215116


namespace monikaTotalSpending_l215_215595

-- Define the conditions as constants
def mallSpent : ℕ := 250
def movieCost : ℕ := 24
def movieCount : ℕ := 3
def beanCost : ℚ := 1.25
def beanCount : ℕ := 20

-- Define the theorem to prove the total spending
theorem monikaTotalSpending : mallSpent + (movieCost * movieCount) + (beanCost * beanCount) = 347 :=
by
  sorry

end monikaTotalSpending_l215_215595


namespace value_of_a3_l215_215832

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Given conditions
def S (n : ℕ) : ℤ := 2 * (n ^ 2) - 1
def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem value_of_a3 : a 3 = 10 := by
  sorry

end value_of_a3_l215_215832


namespace max_workers_l215_215264

theorem max_workers (S a n : ℕ) (h1 : n > 0) (h2 : S > 0) (h3 : a > 0)
  (h4 : (S:ℚ) / (a * n) > (3 * S:ℚ) / (a * (n + 5))) :
  2 * n + 5 = 9 := 
by
  sorry

end max_workers_l215_215264


namespace graph_transform_l215_215691

-- Define the quadratic function y1 as y = -2x^2 + 4x + 1
def y1 (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

-- Define the quadratic function y2 as y = -2x^2
def y2 (x : ℝ) : ℝ := -2 * x^2

-- Define the transformation function for moving 1 unit to the left and 3 units down
def transform (y : ℝ → ℝ) (x : ℝ) : ℝ := y (x + 1) - 3

-- Statement to prove
theorem graph_transform : ∀ x : ℝ, transform y1 x = y2 x :=
by
  intros x
  sorry

end graph_transform_l215_215691


namespace car_fuel_efficiency_l215_215247

theorem car_fuel_efficiency (distance gallons fuel_efficiency D : ℝ)
  (h₀ : fuel_efficiency = 40)
  (h₁ : gallons = 3.75)
  (h₂ : distance = 150)
  (h_eff : fuel_efficiency = distance / gallons) :
  fuel_efficiency = 40 ∧ (D / fuel_efficiency) = (D / 40) :=
by
  sorry

end car_fuel_efficiency_l215_215247


namespace soldiers_count_l215_215928

-- Statements of conditions and proofs
theorem soldiers_count (n : ℕ) (s : ℕ) :
  (n * n + 30 = s) →
  ((n + 1) * (n + 1) - 50 = s) →
  s = 1975 :=
by
  intros h1 h2
  -- We know from h1 and h2 that there should be a unique solution for s and n that satisfies both
  -- conditions. Our goal is to show that s must be 1975.

  -- Initialize the proof structure
  sorry

end soldiers_count_l215_215928


namespace correct_fraction_transformation_l215_215241

theorem correct_fraction_transformation (a b : ℕ) (a_ne_0 : a ≠ 0) (b_ne_0 : b ≠ 0) :
  (\frac{2a}{2b} = \frac{a}{b}) ∧ 
  (¬(\frac{a^2}{b^2} = \frac{a}{b})) ∧ 
  (¬(\frac{2a + 1}{4b} = \frac{a + 1}{2b})) ∧ 
  (¬(\frac{a + 2}{b + 2} = \frac{a}{b})) := 
by
  sorry

end correct_fraction_transformation_l215_215241


namespace number_of_boxes_of_nectarines_l215_215093

namespace ProofProblem

/-- Define the given conditions: -/
def crates : Nat := 12
def oranges_per_crate : Nat := 150
def nectarines_per_box : Nat := 30
def total_fruit : Nat := 2280

/-- Define the number of oranges: -/
def total_oranges : Nat := crates * oranges_per_crate

/-- Calculate the number of nectarines: -/
def total_nectarines : Nat := total_fruit - total_oranges

/-- Calculate the number of boxes of nectarines: -/
def boxes_of_nectarines : Nat := total_nectarines / nectarines_per_box

-- Theorem stating that given the conditions, the number of boxes of nectarines is 16.
theorem number_of_boxes_of_nectarines :
  boxes_of_nectarines = 16 := by
  sorry

end ProofProblem

end number_of_boxes_of_nectarines_l215_215093


namespace john_writes_book_every_2_months_l215_215720

theorem john_writes_book_every_2_months
    (years_writing : ℕ)
    (average_earnings_per_book : ℕ)
    (total_earnings : ℕ)
    (H1 : years_writing = 20)
    (H2 : average_earnings_per_book = 30000)
    (H3 : total_earnings = 3600000) : 
    (years_writing * 12 / (total_earnings / average_earnings_per_book)) = 2 :=
by
    sorry

end john_writes_book_every_2_months_l215_215720


namespace count_perfect_squares_diff_of_consecutive_squares_l215_215063

-- Define the notion of a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define what it means to be the difference of two consecutive perfect squares
def is_diff_of_consecutive_squares (n : ℕ) : Prop :=
  ∃ b : ℕ, n = (b + 1) * (b + 1) - b * b

-- Define what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Prove the main theorem
theorem count_perfect_squares_diff_of_consecutive_squares :
  {x : ℕ | x < 20000 ∧ is_perfect_square x ∧ is_diff_of_consecutive_squares x}.to_finset.card = 70 :=
by
  sorry

end count_perfect_squares_diff_of_consecutive_squares_l215_215063


namespace quadratic_general_form_l215_215282

theorem quadratic_general_form :
  ∀ x : ℝ, (x - 2) * (x + 3) = 1 → x^2 + x - 7 = 0 :=
by
  intros x h
  sorry

end quadratic_general_form_l215_215282


namespace avg_of_numbers_l215_215760

theorem avg_of_numbers (a b c d : ℕ) (avg : ℕ) (h₁ : a = 6) (h₂ : b = 16) (h₃ : c = 8) (h₄ : d = 22) (h₅ : avg = 13) :
  (a + b + c + d) / 4 = avg := by
  -- Proof here
  sorry

end avg_of_numbers_l215_215760


namespace arccos_neg_one_eq_pi_l215_215946

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l215_215946


namespace product_of_roots_eq_negative_forty_nine_l215_215812

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l215_215812


namespace profit_percentage_is_20_l215_215908

noncomputable def selling_price : ℝ := 200
noncomputable def cost_price : ℝ := 166.67
noncomputable def profit : ℝ := selling_price - cost_price

theorem profit_percentage_is_20 :
  (profit / cost_price) * 100 = 20 := by
  sorry

end profit_percentage_is_20_l215_215908


namespace Jonas_initial_socks_l215_215583

noncomputable def pairsOfSocks(Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                              (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) : ℕ :=
    let individualShoes := Jonas_pairsOfShoes * 2
    let individualPants := Jonas_pairsOfPants * 2
    let individualTShirts := Jonas_tShirts
    let totalWithoutSocks := individualShoes + individualPants + individualTShirts
    let totalToDouble := (totalWithoutSocks + Jonas_pairsOfNewSocks * 2) / 2
    (totalToDouble * 2 - totalWithoutSocks) / 2

theorem Jonas_initial_socks (Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                             (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) 
                             (h1 : Jonas_pairsOfShoes = 5)
                             (h2 : Jonas_pairsOfPants = 10)
                             (h3 : Jonas_tShirts = 10)
                             (h4 : Jonas_pairsOfNewSocks = 35) :
    pairsOfSocks Jonas_pairsOfShoes Jonas_pairsOfPants Jonas_tShirts Jonas_pairsOfNewSocks = 15 :=
by
    subst h1
    subst h2
    subst h3
    subst h4
    sorry

end Jonas_initial_socks_l215_215583


namespace solve_for_x_l215_215070

theorem solve_for_x (i x : ℂ) (h : i^2 = -1) (eq : 3 - 2 * i * x = 5 + 4 * i * x) : x = i / 3 := 
by
  sorry

end solve_for_x_l215_215070


namespace men_absent_l215_215915

theorem men_absent (x : ℕ) (H1 : 10 * 6 = 60) (H2 : (10 - x) * 10 = 60) : x = 4 :=
by
  sorry

end men_absent_l215_215915


namespace proof_ab_greater_ac_l215_215544

theorem proof_ab_greater_ac (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  a * b > a * c :=
by sorry

end proof_ab_greater_ac_l215_215544


namespace alec_votes_l215_215924

theorem alec_votes (class_size : ℕ) (half_class_votes : ℕ) (remaining_interested : ℕ) (fraction_persuaded : ℕ) :
  class_size = 60 →
  half_class_votes = class_size / 2 →
  remaining_interested = 5 →
  fraction_persuaded = (class_size - half_class_votes - remaining_interested) / 5 →
  (3 * class_size) / 4 - (half_class_votes + fraction_persuaded) = 10 :=
by
  intros h_class_size h_half_class_votes h_remaining_interested h_fraction_persuaded
  rw h_class_size at h_half_class_votes h_remaining_interested h_fraction_persuaded
  rw [h_half_class_votes, h_remaining_interested, h_fraction_persuaded]
  sorry

end alec_votes_l215_215924


namespace green_beans_count_l215_215365

def total_beans := 572
def red_beans := (1 / 4) * total_beans
def remaining_after_red := total_beans - red_beans
def white_beans := (1 / 3) * remaining_after_red
def remaining_after_white := remaining_after_red - white_beans
def green_beans := (1 / 2) * remaining_after_white

theorem green_beans_count : green_beans = 143 := by
  sorry

end green_beans_count_l215_215365


namespace product_of_solutions_product_of_all_t_l215_215806

theorem product_of_solutions (t : ℝ) (h : t^2 = 49) : 
  (t = 7 ∨ t = -7) :=
sorry

theorem product_of_all_t (s : Set ℝ) (h : ∀ t ∈ s, t^2 = 49) : 
  ∏ t in s, t = -49 :=
sorry

end product_of_solutions_product_of_all_t_l215_215806


namespace age_difference_l215_215630

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : C + 16 = A := 
by
  sorry

end age_difference_l215_215630


namespace find_purchase_price_l215_215638

noncomputable def purchase_price (total_paid : ℝ) (interest_percent : ℝ) : ℝ :=
    total_paid / (1 + interest_percent)

theorem find_purchase_price :
  purchase_price 130 0.09090909090909092 = 119.09 :=
by
  -- Normally we would provide the full proof here, but it is omitted as per instructions
  sorry

end find_purchase_price_l215_215638


namespace ryan_learning_hours_l215_215953

theorem ryan_learning_hours :
  ∀ (e c s : ℕ) , (e = 6) → (s = 58) → (e = c + 3) → (c = 3) :=
by
  intros e c s he hs hc
  sorry

end ryan_learning_hours_l215_215953


namespace inverse_proportion_expression_and_calculation_l215_215737

theorem inverse_proportion_expression_and_calculation :
  (∃ k : ℝ, (∀ (x y : ℝ), y = k / x) ∧
   (∀ x y : ℝ, y = 400 ∧ x = 0.25 → k = 100) ∧
   (∀ x : ℝ, 200 = 100 / x → x = 0.5)) :=
by
  sorry

end inverse_proportion_expression_and_calculation_l215_215737


namespace find_number_l215_215899

theorem find_number 
  (x : ℝ) 
  (h1 : 3 * (2 * x + 9) = 69) : x = 7 := by
  sorry

end find_number_l215_215899


namespace solve_for_f_1988_l215_215053

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom functional_eq (m n : ℕ+) : f (f m + f n) = m + n

theorem solve_for_f_1988 : f 1988 = 1988 :=
sorry

end solve_for_f_1988_l215_215053


namespace multiplication_simplification_l215_215733

theorem multiplication_simplification :
  let y := 6742
  let z := 397778
  let approx_mult (a b : ℕ) := 60 * a - a
  z = approx_mult y 59 := sorry

end multiplication_simplification_l215_215733


namespace couple_ticket_cost_l215_215375

variable (x : ℝ)

def single_ticket_cost : ℝ := 20
def total_sales : ℝ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16

theorem couple_ticket_cost :
  96 * single_ticket_cost + 16 * x = total_sales →
  x = 22.5 :=
by
  sorry

end couple_ticket_cost_l215_215375


namespace square_side_length_is_10_l215_215481

-- Define the side lengths of the original squares
def side_length1 : ℝ := 8
def side_length2 : ℝ := 6

-- Define the areas of the original squares
def area1 : ℝ := side_length1^2
def area2 : ℝ := side_length2^2

-- Define the total area of the combined squares
def total_area : ℝ := area1 + area2

-- Define the side length of the new square
def side_length_new_square : ℝ := 10

-- Theorem statement to prove that the side length of the new square is 10 cm
theorem square_side_length_is_10 : side_length_new_square^2 = total_area := by
  sorry

end square_side_length_is_10_l215_215481


namespace intersection_of_M_and_N_l215_215311

def M : Set ℕ := {0, 2, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N :
  {x | x ∈ M ∧ x ∈ N} = {0, 4} := by
  sorry

end intersection_of_M_and_N_l215_215311


namespace inheritance_amount_l215_215193

theorem inheritance_amount (x : ℝ)
  (federal_tax_rate : ℝ := 0.25)
  (state_tax_rate : ℝ := 0.15)
  (total_taxes_paid : ℝ := 16000)
  (H : (federal_tax_rate * x) + (state_tax_rate * (1 - federal_tax_rate) * x) = total_taxes_paid) :
  x = 44138 := sorry

end inheritance_amount_l215_215193


namespace Ashis_height_more_than_Babji_height_l215_215270

-- Definitions based on conditions
variables {A B : ℝ}
-- Condition expressing the relationship between Ashis's and Babji's height
def Babji_height (A : ℝ) : ℝ := 0.80 * A

-- The proof problem to show the percentage increase
theorem Ashis_height_more_than_Babji_height :
  B = Babji_height A → (A - B) / B * 100 = 25 :=
sorry

end Ashis_height_more_than_Babji_height_l215_215270


namespace det_condition_l215_215977

theorem det_condition (a b c d : ℤ) 
    (h_exists : ∀ m n : ℤ, ∃ h k : ℤ, a * h + b * k = m ∧ c * h + d * k = n) :
    |a * d - b * c| = 1 :=
sorry

end det_condition_l215_215977


namespace convert_length_convert_area_convert_time_convert_mass_l215_215956

theorem convert_length (cm : ℕ) : cm = 7 → (cm : ℚ) / 100 = 7 / 100 :=
by sorry

theorem convert_area (dm2 : ℕ) : dm2 = 35 → (dm2 : ℚ) / 100 = 7 / 20 :=
by sorry

theorem convert_time (min : ℕ) : min = 45 → (min : ℚ) / 60 = 3 / 4 :=
by sorry

theorem convert_mass (g : ℕ) : g = 2500 → (g : ℚ) / 1000 = 5 / 2 :=
by sorry

end convert_length_convert_area_convert_time_convert_mass_l215_215956


namespace robins_initial_hair_length_l215_215864

variable (L : ℕ)

def initial_length_after_cutting := L - 11
def length_after_growth := initial_length_after_cutting L + 12
def final_length := 17

theorem robins_initial_hair_length : length_after_growth L = final_length → L = 16 := 
by sorry

end robins_initial_hair_length_l215_215864


namespace zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l215_215055

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  x - 1/x - 2 * m * Real.log x

theorem zero_of_f_when_m_is_neg1 : ∃ x > 0, f x (-1) = 0 :=
  by
    use 1
    sorry

theorem monotonicity_of_f_m_gt_neg1 (m : ℝ) (hm : m > -1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x m ≤ f y m) ∨
  (∃ a b : ℝ, 0 < a ∧ a < b ∧
    (∀ x : ℝ, 0 < x ∧ x < a → f x m ≤ f a m) ∧
    (∀ x : ℝ, a < x ∧ x < b → f a m ≥ f x m) ∧
    (∀ x : ℝ, b < x → f b m ≤ f x m)) :=
  by
    cases lt_or_le m 1 with
    | inl hlt =>
        left
        intros x y hx hy hxy
        sorry
    | inr hle =>
        right
        use m - Real.sqrt (m^2 - 1), m + Real.sqrt (m^2 - 1)
        sorry

end zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l215_215055


namespace find_a_l215_215683

def set_A : Set ℝ := { x | abs (x - 1) > 2 }
def set_B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a < 0 }
def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_a (a : ℝ) : (intersection set_A (set_B a)) = { x | 3 < x ∧ x < 5 } → a = 5 :=
by
  sorry

end find_a_l215_215683


namespace exists_m_divisible_by_2k_l215_215142

theorem exists_m_divisible_by_2k {k : ℕ} (h_k : 0 < k) {a : ℤ} (h_a : a % 8 = 3) :
  ∃ m : ℕ, 0 < m ∧ 2^k ∣ (a^m + a + 2) :=
sorry

end exists_m_divisible_by_2k_l215_215142


namespace set_operation_empty_l215_215197

-- Definition of the universal set I, and sets P and Q with the given properties
variable {I : Set ℕ} -- Universal set
variable {P Q : Set ℕ} -- Non-empty sets with P ⊂ Q ⊂ I
variable (hPQ : P ⊂ Q) (hQI : Q ⊂ I)

-- Prove the set operation expression that results in the empty set
theorem set_operation_empty :
  ∃ (P Q : Set ℕ), P ⊂ Q ∧ Q ⊂ I ∧ P ≠ ∅ ∧ Q ≠ ∅ → 
  P ∩ (I \ Q) = ∅ :=
by
  sorry

end set_operation_empty_l215_215197


namespace contractor_engagement_days_l215_215250

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end contractor_engagement_days_l215_215250


namespace no_solution_for_inequalities_l215_215132

theorem no_solution_for_inequalities :
  ¬ ∃ x : ℝ, 3 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 8 * x - 5 := by 
  sorry

end no_solution_for_inequalities_l215_215132


namespace circle_equation_l215_215202

theorem circle_equation :
  ∃ M : ℝ × ℝ, (2 * M.1 + M.2 - 1 = 0) ∧
    (∃ r : ℝ, r ≥ 0 ∧ 
      ((3 - M.1)^2 + (0 - M.2)^2 = r^2) ∧
      ((0 - M.1)^2 + (1 - M.2)^2 = r^2)) ∧
    (∃ x y : ℝ, ((x - 1)^2 + (y + 1)^2 = 5)) := 
sorry

end circle_equation_l215_215202


namespace find_a_from_expansion_l215_215734

theorem find_a_from_expansion :
  (∃ a : ℝ, (∃ c : ℝ, (∃ d : ℝ, (∃ e : ℝ, (20 - 30 * a + 6 * a^2 = -16 ∧ (a = 2 ∨ a = 3))))))
:= sorry

end find_a_from_expansion_l215_215734


namespace robot_material_handling_per_hour_min_num_type_A_robots_l215_215112

-- Definitions and conditions for part 1
def material_handling_robot_B (x : ℕ) := x
def material_handling_robot_A (x : ℕ) := x + 30

def condition_time_handled (x : ℕ) :=
  1000 / material_handling_robot_A x = 800 / material_handling_robot_B x

-- Definitions for part 2
def total_robots := 20
def min_material_handling_per_hour := 2800

def material_handling_total (a b : ℕ) :=
  150 * a + 120 * b

-- Proof problems
theorem robot_material_handling_per_hour :
  ∃ (x : ℕ), material_handling_robot_B x = 120 ∧ material_handling_robot_A x = 150 ∧ condition_time_handled x :=
sorry

theorem min_num_type_A_robots :
  ∀ (a b : ℕ),
  a + b = total_robots →
  material_handling_total a b ≥ min_material_handling_per_hour →
  a ≥ 14 :=
sorry

end robot_material_handling_per_hour_min_num_type_A_robots_l215_215112


namespace range_of_m_l215_215724

theorem range_of_m (m : ℝ) :
  let M := {x : ℝ | x ≤ m}
  let P := {x : ℝ | x ≥ -1}
  (M ∩ P = ∅) → m < -1 :=
by
  sorry

end range_of_m_l215_215724


namespace max_angle_in_hexagon_l215_215486

-- Definition of the problem
theorem max_angle_in_hexagon :
  ∃ (a d : ℕ), a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 ∧ 
               a + 5 * d < 180 ∧ 
               (∀ a d : ℕ, a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 → 
               a + 5*d < 180 → m <= 175) :=
sorry

end max_angle_in_hexagon_l215_215486


namespace aubrey_travel_time_l215_215272

def aubrey_time_to_school (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem aubrey_travel_time :
  aubrey_time_to_school 88 22 = 4 := by
  sorry

end aubrey_travel_time_l215_215272


namespace total_rowing_proof_l215_215767

def morning_rowing := 13
def afternoon_rowing := 21
def total_rowing := 34

theorem total_rowing_proof :
  morning_rowing + afternoon_rowing = total_rowing :=
by
  sorry

end total_rowing_proof_l215_215767


namespace number_of_girls_l215_215186

theorem number_of_girls (B G : ℕ) (h₁ : B = 6 * G / 5) (h₂ : B + G = 440) : G = 200 :=
by {
  sorry -- Proof steps here
}

end number_of_girls_l215_215186


namespace find_a_l215_215698

theorem find_a (a : ℝ) (h : Nat.choose 5 2 * (-a)^3 = 10) : a = -1 :=
by
  sorry

end find_a_l215_215698


namespace unique_solution_l215_215795

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (f x)) + f (f y) = f y + x

-- Define the main theorem
theorem unique_solution (f : ℝ → ℝ) :
  (∀ x y, functional_eq f x y) → (∀ x, f x = x) :=
by
  intros h x
  -- Proof steps would go here
  sorry

end unique_solution_l215_215795


namespace probability_smallest_divides_l215_215620

open Finset

theorem probability_smallest_divides 
  (S : Finset ℕ := {1, 2, 3, 4, 5, 6}) 
  (choosing_fun : Finset ℕ → Finset (Finset ℕ) := λ s, filter (λ t, 3 = card t) (powerset s)) 
  (A : Finset (Finset ℕ) := filter (λ t, ∃ a b c, t = {a, b, c} ∧ a < b ∧ a < c ∧ b % a = 0 ∧ c % a = 0) (choosing_fun S)) :
  (A.card : ℚ) / (choosing_fun S).card = 11 / 20 := 
begin
  sorry
end

end probability_smallest_divides_l215_215620


namespace expression_value_l215_215314

def a : ℤ := 5
def b : ℤ := -3
def c : ℕ := 2

theorem expression_value : (3 * c) / (a + b) + c = 5 := by
  sorry

end expression_value_l215_215314


namespace sqrt_5th_of_x_sqrt_4th_x_l215_215930

theorem sqrt_5th_of_x_sqrt_4th_x (x : ℝ) (hx : 0 < x) : Real.sqrt (x * Real.sqrt (x ^ (1 / 4))) = x ^ (1 / 4) :=
by
  sorry

end sqrt_5th_of_x_sqrt_4th_x_l215_215930


namespace total_days_on_island_correct_l215_215854

-- Define the first, second, and third expeditions
def firstExpedition : ℕ := 3

def secondExpedition (a : ℕ) : ℕ := a + 2

def thirdExpedition (b : ℕ) : ℕ := 2 * b

-- Define the total duration in weeks
def totalWeeks : ℕ := firstExpedition + secondExpedition firstExpedition + thirdExpedition (secondExpedition firstExpedition)

-- Define the total days spent on the island
def totalDays (weeks : ℕ) : ℕ := weeks * 7

-- Prove that the total number of days spent is 126
theorem total_days_on_island_correct : totalDays totalWeeks = 126 := 
  by
    sorry

end total_days_on_island_correct_l215_215854


namespace least_ab_value_l215_215161

theorem least_ab_value (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : (1 : ℚ) / a + (1 : ℚ) / (3 * b) = 1 / 9) :
  ab = 144 :=
sorry

end least_ab_value_l215_215161


namespace percentage_of_360_equals_126_l215_215383

/-- 
  Prove that (126 / 360) * 100 equals 35.
-/
theorem percentage_of_360_equals_126 : (126 / 360 : ℝ) * 100 = 35 := by
  sorry

end percentage_of_360_equals_126_l215_215383


namespace radius_of_spheres_in_cone_l215_215850

-- Given Definitions
def cone_base_radius : ℝ := 6
def cone_height : ℝ := 15
def tangent_spheres (r : ℝ) : Prop :=
  r = (12 * Real.sqrt 29) / 29

-- Problem Statement
theorem radius_of_spheres_in_cone :
  ∃ r : ℝ, tangent_spheres r :=
sorry

end radius_of_spheres_in_cone_l215_215850


namespace book_E_chapters_l215_215726

def total_chapters: ℕ := 97
def chapters_A: ℕ := 17
def chapters_B: ℕ := chapters_A + 5
def chapters_C: ℕ := chapters_B - 7
def chapters_D: ℕ := chapters_C * 2
def chapters_sum : ℕ := chapters_A + chapters_B + chapters_C + chapters_D

theorem book_E_chapters :
  total_chapters - chapters_sum = 13 :=
by
  sorry

end book_E_chapters_l215_215726


namespace noodles_initial_l215_215949

-- Definitions of our conditions
def given_away : ℝ := 12.0
def noodles_left : ℝ := 42.0
def initial_noodles : ℝ := 54.0

-- Theorem statement
theorem noodles_initial (a b : ℝ) (x : ℝ) (h₁ : a = 12.0) (h₂ : b = 42.0) (h₃ : x = a + b) : x = initial_noodles :=
by
  -- Placeholder for the proof
  sorry

end noodles_initial_l215_215949


namespace trains_clear_time_l215_215503

noncomputable def length_train1 : ℝ := 150
noncomputable def length_train2 : ℝ := 165
noncomputable def speed_train1_kmh : ℝ := 80
noncomputable def speed_train2_kmh : ℝ := 65
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * (5/18)
noncomputable def speed_train1 : ℝ := kmh_to_mps speed_train1_kmh
noncomputable def speed_train2 : ℝ := kmh_to_mps speed_train2_kmh
noncomputable def total_distance : ℝ := length_train1 + length_train2
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_time : time_to_clear = 7.82 := 
sorry

end trains_clear_time_l215_215503


namespace range_of_m_l215_215968

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

def tangent_points (m : ℝ) (x₀ : ℝ) : Prop := 
  2 * x₀ ^ 3 - 3 * x₀ ^ 2 + m + 3 = 0

theorem range_of_m (m : ℝ) :
  (∀ x₀, tangent_points m x₀) ∧ m ≠ -2 → (-3 < m ∧ m < -2) :=
sorry

end range_of_m_l215_215968


namespace grasshopper_jump_l215_215485

theorem grasshopper_jump (frog_jump grasshopper_jump : ℕ)
  (h1 : frog_jump = grasshopper_jump + 17)
  (h2 : frog_jump = 53) :
  grasshopper_jump = 36 :=
by
  sorry

end grasshopper_jump_l215_215485


namespace c_work_rate_l215_215763

theorem c_work_rate {A B C : ℚ} (h1 : A + B = 1/6) (h2 : B + C = 1/8) (h3 : C + A = 1/12) : C = 1/48 :=
by
  sorry

end c_work_rate_l215_215763


namespace inequality_proof_l215_215159

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
by 
  sorry

end inequality_proof_l215_215159


namespace find_abc_value_l215_215174

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : a * b = 30)
variable (h5 : b * c = 54)
variable (h6 : c * a = 45)

theorem find_abc_value : a * b * c = 270 := by
  sorry

end find_abc_value_l215_215174


namespace algebra_expression_value_l215_215426

theorem algebra_expression_value (x y : ℝ)
  (h1 : x + y = 3)
  (h2 : x * y = 1) :
  (5 * x + 3) - (2 * x * y - 5 * y) = 16 :=
by
  sorry

end algebra_expression_value_l215_215426


namespace triangle_is_right_l215_215457

theorem triangle_is_right {A B C : ℝ} (h : A + B + C = 180) (h1 : A = B + C) : A = 90 :=
by
  sorry

end triangle_is_right_l215_215457


namespace B_50_l215_215826

noncomputable def B (n : ℕ) (a : ℝ) : ℝ := ∑ i in finset.range n, (3 * (i+1) - 2) * a^(i+1)

theorem B_50 : 
  ∃ a : ℝ, (4 * a - 3 < 3 - 2 * a^2) ∧ y = f (2 * x - 3) ∧ 
  (∀ x : ℝ, y = f (-x + 3)) →
  a = -1 → B 50 a = 75 := 
sorry

end B_50_l215_215826


namespace product_mod_32_l215_215001

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215001


namespace probability_one_person_hits_probability_plane_is_hit_l215_215906
noncomputable def P_A := 0.7
noncomputable def P_B := 0.6

theorem probability_one_person_hits : P_A * (1 - P_B) + (1 - P_A) * P_B = 0.46 :=
by
  sorry

theorem probability_plane_is_hit : 1 - (1 - P_A) * (1 - P_B) = 0.88 :=
by
  sorry

end probability_one_person_hits_probability_plane_is_hit_l215_215906


namespace total_cost_of_rolls_l215_215459

-- Defining the conditions
def price_per_dozen : ℕ := 5
def total_rolls_bought : ℕ := 36
def rolls_per_dozen : ℕ := 12

-- Prove the total cost calculation
theorem total_cost_of_rolls : (total_rolls_bought / rolls_per_dozen) * price_per_dozen = 15 :=
by
  sorry

end total_cost_of_rolls_l215_215459


namespace geometric_sequence_b_value_l215_215614

theorem geometric_sequence_b_value 
  (b : ℝ)
  (h1 : b > 0)
  (h2 : ∃ r : ℝ, 160 * r = b ∧ b * r = 1)
  : b = 4 * Real.sqrt 10 := 
sorry

end geometric_sequence_b_value_l215_215614


namespace scooped_water_amount_l215_215923

variables (x : ℝ)

def initial_water_amount : ℝ := 10
def total_amount : ℝ := initial_water_amount
def alcohol_concentration : ℝ := 0.75

theorem scooped_water_amount (h : x / total_amount = alcohol_concentration) : x = 7.5 :=
by sorry

end scooped_water_amount_l215_215923


namespace max_ab_correct_l215_215546

noncomputable def max_ab (k : ℝ) (a b: ℝ) : ℝ :=
if k = -3 then 9 else sorry

theorem max_ab_correct (k : ℝ) (a b: ℝ)
  (h1 : (-3 ≤ k ∧ k ≤ 1))
  (h2 : a + b = 2 * k)
  (h3 : a^2 + b^2 = k^2 - 2 * k + 3) :
  max_ab k a b = 9 :=
sorry

end max_ab_correct_l215_215546


namespace area_excircle_gteq_four_times_area_l215_215881

-- Define the area function
def area (A B C : Point) : ℝ := sorry -- Area of triangle ABC (this will be implemented later)

-- Define the centers of the excircles (this needs precise definitions and setup)
def excircle_center (A B C : Point) : Point := sorry -- Centers of the excircles of triangle ABC (implementation would follow)

-- Define the area of the triangle formed by the excircle centers
def excircle_area (A B C : Point) : ℝ :=
  let O1 := excircle_center A B C
  let O2 := excircle_center B C A
  let O3 := excircle_center C A B
  area O1 O2 O3

-- Prove the main statement
theorem area_excircle_gteq_four_times_area (A B C : Point) :
  excircle_area A B C ≥ 4 * area A B C :=
by sorry

end area_excircle_gteq_four_times_area_l215_215881


namespace express_308_million_in_scientific_notation_l215_215074

theorem express_308_million_in_scientific_notation :
    (308000000 : ℝ) = 3.08 * (10 ^ 8) :=
by
  sorry

end express_308_million_in_scientific_notation_l215_215074


namespace negation_of_cos_proposition_l215_215559

variable (x : ℝ)

theorem negation_of_cos_proposition (h : ∀ x : ℝ, Real.cos x ≤ 1) : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end negation_of_cos_proposition_l215_215559


namespace total_flowers_sold_l215_215121

def flowers_sold_on_monday : ℕ := 4
def flowers_sold_on_tuesday : ℕ := 8
def flowers_sold_on_friday : ℕ := 2 * flowers_sold_on_monday

theorem total_flowers_sold : flowers_sold_on_monday + flowers_sold_on_tuesday + flowers_sold_on_friday = 20 := by
  sorry

end total_flowers_sold_l215_215121


namespace person_age_l215_215263

variable (x : ℕ) -- Define the variable for age

-- State the condition as a hypothesis
def condition (x : ℕ) : Prop :=
  3 * (x + 3) - 3 * (x - 3) = x

-- State the theorem to be proved
theorem person_age (x : ℕ) (h : condition x) : x = 18 := 
sorry

end person_age_l215_215263


namespace f_at_2018_l215_215679

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_periodic : ∀ x : ℝ, f (x + 6) = f x
axiom f_at_4 : f 4 = 5

theorem f_at_2018 : f 2018 = 5 :=
by
  -- Proof goes here
  sorry

end f_at_2018_l215_215679


namespace sin2alpha_plus_cosalpha_l215_215967

theorem sin2alpha_plus_cosalpha (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) + Real.cos α = (4 + Real.sqrt 5) / 5 :=
by
  sorry

end sin2alpha_plus_cosalpha_l215_215967


namespace geometric_sequence_sixth_term_l215_215483

theorem geometric_sequence_sixth_term (a : ℝ) (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^(7) = 2) :
  a * r^(5) = 16 :=
by
  sorry

end geometric_sequence_sixth_term_l215_215483


namespace arithmetic_sequence_property_l215_215548

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

def condition (S : ℕ → ℝ) : Prop :=
  (S 8 - S 5) * (S 8 - S 4) < 0

-- Theorem to prove
theorem arithmetic_sequence_property {a : ℕ → ℝ} {S : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_cond : condition S) :
  |a 5| > |a 6| := 
sorry

end arithmetic_sequence_property_l215_215548


namespace area_of_concentric_ring_l215_215377

theorem area_of_concentric_ring (r_large : ℝ) (r_small : ℝ) 
  (h1 : r_large = 10) 
  (h2 : r_small = 6) : 
  (π * r_large^2 - π * r_small^2) = 64 * π :=
by {
  sorry
}

end area_of_concentric_ring_l215_215377


namespace when_to_sell_goods_l215_215785

variable (a : ℝ) (currentMonthProfit nextMonthProfitWithStorage : ℝ) 
          (interestRate storageFee thisMonthProfit nextMonthProfit : ℝ)
          (hm1 : interestRate = 0.005)
          (hm2 : storageFee = 5)
          (hm3 : thisMonthProfit = 100)
          (hm4 : nextMonthProfit = 120)
          (hm5 : currentMonthProfit = thisMonthProfit + (a + thisMonthProfit) * interestRate)
          (hm6 : nextMonthProfitWithStorage = nextMonthProfit - storageFee)

theorem when_to_sell_goods :
  (a > 2900 → currentMonthProfit > nextMonthProfitWithStorage) ∧
  (a = 2900 → currentMonthProfit = nextMonthProfitWithStorage) ∧
  (a < 2900 → currentMonthProfit < nextMonthProfitWithStorage) := by
  sorry

end when_to_sell_goods_l215_215785


namespace coefficient_of_x3_in_expansion_l215_215611

theorem coefficient_of_x3_in_expansion :
  (∀ (x : ℝ), (Polynomial.coeff ((Polynomial.C x - 1)^5) 3) = 10) :=
by
  sorry

end coefficient_of_x3_in_expansion_l215_215611


namespace carl_gave_beth_35_coins_l215_215528

theorem carl_gave_beth_35_coins (x : ℕ) (h1 : ∃ n, n = 125) (h2 : ∃ m, m = (125 + x) / 2) (h3 : m = 80) : x = 35 :=
by
  sorry

end carl_gave_beth_35_coins_l215_215528


namespace pesto_calculation_l215_215274

def basil_needed_per_pesto : ℕ := 4
def basil_harvest_per_week : ℕ := 16
def weeks : ℕ := 8
def total_basil_harvested : ℕ := basil_harvest_per_week * weeks
def total_pesto_possible : ℕ := total_basil_harvested / basil_needed_per_pesto

theorem pesto_calculation :
  total_pesto_possible = 32 :=
by
  sorry

end pesto_calculation_l215_215274


namespace roots_squared_sum_l215_215821

theorem roots_squared_sum :
  (∀ x, x^2 + 2 * x - 8 = 0 → (x = x1 ∨ x = x2)) →
  x1 + x2 = -2 ∧ x1 * x2 = -8 →
  x1^2 + x2^2 = 20 :=
by
  intros roots_eq_sum_prod_eq
  sorry

end roots_squared_sum_l215_215821


namespace ticket_cost_proof_l215_215784

def adult_ticket_price : ℕ := 55
def child_ticket_price : ℕ := 28
def senior_ticket_price : ℕ := 42

def num_adult_tickets : ℕ := 4
def num_child_tickets : ℕ := 2
def num_senior_tickets : ℕ := 1

def total_ticket_cost : ℕ :=
  (num_adult_tickets * adult_ticket_price) + (num_child_tickets * child_ticket_price) + (num_senior_tickets * senior_ticket_price)

theorem ticket_cost_proof : total_ticket_cost = 318 := by
  sorry

end ticket_cost_proof_l215_215784


namespace calculate_expression_l215_215530

theorem calculate_expression :
  500 * 1986 * 0.3972 * 100 = 20 * 1986^2 :=
by sorry

end calculate_expression_l215_215530


namespace cos_three_pi_over_four_l215_215538

theorem cos_three_pi_over_four :
  Real.cos (3 * Real.pi / 4) = -1 / Real.sqrt 2 :=
by
  sorry

end cos_three_pi_over_four_l215_215538


namespace distance_between_A_and_B_is_90_l215_215728

variable (A B : Type)
variables (v_A v_B v'_A v'_B : ℝ)
variable (d : ℝ)

-- Conditions
axiom starts_simultaneously : True
axiom speed_ratio : v_A / v_B = 4 / 5
axiom A_speed_decrease : v'_A = 0.75 * v_A
axiom B_speed_increase : v'_B = 1.2 * v_B
axiom distance_when_B_reaches_A : ∃ k : ℝ, k = 30 -- Person A is 30 km away from location B

-- Goal
theorem distance_between_A_and_B_is_90 : d = 90 := by 
  sorry

end distance_between_A_and_B_is_90_l215_215728


namespace quadratic_other_x_intercept_l215_215422

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → (a * x^2 + b * x + c) = -3)
  (h_intercept : ∀ x, x = 1 → (a * x^2 + b * x + c) = 0) : 
  ∃ x : ℝ, x = 9 ∧ (a * x^2 + b * x + c) = 0 :=
sorry

end quadratic_other_x_intercept_l215_215422


namespace product_mod_32_l215_215003

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215003


namespace smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l215_215506

theorem smallest_two_digit_multiple_of_17 : ∃ m, 10 ≤ m ∧ m < 100 ∧ 17 ∣ m ∧ ∀ n, 10 ≤ n ∧ n < 100 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

theorem smallest_four_digit_multiple_of_17 : ∃ m, 1000 ≤ m ∧ m < 10000 ∧ 17 ∣ m ∧ ∀ n, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

end smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l215_215506


namespace find_x2_minus_x1_l215_215057

theorem find_x2_minus_x1 (a x1 x2 d e : ℝ) (h_a : a ≠ 0) (h_d : d ≠ 0) (h_x : x1 ≠ x2) (h_e : e = -d * x1)
  (h_y1 : ∀ x, y1 = a * (x - x1) * (x - x2)) (h_y2 : ∀ x, y2 = d * x + e)
  (h_intersect : ∀ x, y = a * (x - x1) * (x - x2) + (d * x + e)) 
  (h_single_point : ∀ x, y = a * (x - x1)^2) :
  x2 - x1 = d / a :=
sorry

end find_x2_minus_x1_l215_215057


namespace product_mod_32_is_15_l215_215012

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215012


namespace eval_expression_l215_215787

theorem eval_expression : 
  real.cbrt 8 + (1 / 3)⁻¹ - 2 * real.cos (real.pi / 6) + |1 - real.sqrt 3| = 4 :=
by 
  have h1: real.cbrt 8 = 2 := by norm_num,
  have h2: (1 / 3)⁻¹ = 3 := by norm_num,
  have h3: 2 * real.cos (real.pi / 6) = real.sqrt 3 := by norm_num,
  have h4: |1 - real.sqrt 3| = real.sqrt 3 - 1 := 
    by { rw real.abs_of_nonpos (sub_nonpos.mpr (real.sqrt_lt_sqrt (show (1:ℝ)^2 < (real.sqrt 3)^2, by norm_num))) },
  sorry

end eval_expression_l215_215787


namespace product_of_roots_l215_215744

theorem product_of_roots (a b c d : ℝ)
  (h1 : a = 16 ^ (1 / 5))
  (h2 : 16 = 2 ^ 4)
  (h3 : b = 64 ^ (1 / 6))
  (h4 : 64 = 2 ^ 6):
  a * b = 2 * (16 ^ (1 / 5)) := by
  sorry

end product_of_roots_l215_215744


namespace div_fraction_l215_215236

/-- The result of dividing 3/7 by 2 1/2 equals 6/35 -/
theorem div_fraction : (3/7) / (2 + 1/2) = 6/35 :=
by 
  sorry

end div_fraction_l215_215236


namespace find_b_value_l215_215740

-- Definitions based on the problem conditions
def line_bisects_circle (b : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, (c.fst = 4 ∧ c.snd = -1) ∧
                (c.snd = c.fst + b)

-- Theorem statement for the problem
theorem find_b_value : line_bisects_circle (-5) :=
by
  sorry

end find_b_value_l215_215740


namespace triangle_area_l215_215188

-- Define a triangle as a structure with vertices A, B, and C, where the lengths AB, AC, and BC are provided
structure Triangle :=
  (A B C : ℝ)
  (AB AC BC : ℝ)
  (is_isosceles : AB = AC)
  (BC_length : BC = 20)
  (AB_length : AB = 26)

-- Define the length bisector and Pythagorean properties
def bisects_base (t : Triangle) : Prop :=
  ∃ D : ℝ, (t.B - D) = (D - t.C) ∧ 2 * D = t.B + t.C

def pythagorean_theorem_AD (t : Triangle) (D : ℝ) (AD : ℝ) : Prop :=
  t.AB^2 = AD^2 + (t.B - D)^2

-- State the problem as a theorem
theorem triangle_area (t : Triangle) (D : ℝ) (AD : ℝ) (h1 : bisects_base t) (h2 : pythagorean_theorem_AD t D AD) :
  AD = 24 ∧ (1 / 2) * t.BC * AD = 240 :=
sorry

end triangle_area_l215_215188


namespace quadratic_other_x_intercept_l215_215424

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → a * x^2 + b * x + c = -3)
  (h_intercept : a * 1^2 + b * 1 + c = 0) : 
  ∃ x0 : ℝ, x0 = 9 ∧ a * x0^2 + b * x0 + c = 0 :=
by
  sorry

end quadratic_other_x_intercept_l215_215424


namespace shirts_before_buying_l215_215067

-- Define the conditions
variable (new_shirts : ℕ)
variable (total_shirts : ℕ)

-- Define the statement where we need to prove the number of shirts Sarah had before buying the new ones
theorem shirts_before_buying (h₁ : new_shirts = 8) (h₂ : total_shirts = 17) : total_shirts - new_shirts = 9 :=
by
  -- Proof goes here
  sorry

end shirts_before_buying_l215_215067


namespace no_power_of_two_divides_3n_plus_1_l215_215603

theorem no_power_of_two_divides_3n_plus_1 (n : ℕ) (hn : n > 1) : ¬ (2^n ∣ 3^n + 1) := sorry

end no_power_of_two_divides_3n_plus_1_l215_215603


namespace analytic_expression_of_f_max_min_of_f_on_interval_l215_215432

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem analytic_expression_of_f :
  ∀ A ω φ : ℝ, (∀ x, f x = A * Real.sin (ω * x + φ)) →
  A = 2 ∧ ω = 2 ∧ φ = Real.pi / 6 :=
by
  sorry -- Placeholder for the actual proof

theorem max_min_of_f_on_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≤ Real.sqrt 3) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≥ 1) :=
by
  sorry -- Placeholder for the actual proof

end analytic_expression_of_f_max_min_of_f_on_interval_l215_215432


namespace cost_of_figurine_l215_215644

noncomputable def cost_per_tv : ℝ := 50
noncomputable def num_tvs : ℕ := 5
noncomputable def num_figurines : ℕ := 10
noncomputable def total_spent : ℝ := 260

theorem cost_of_figurine : 
  ((total_spent - (num_tvs * cost_per_tv)) / num_figurines) = 1 := 
by
  sorry

end cost_of_figurine_l215_215644


namespace min_value_fractions_l215_215199

theorem min_value_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2 ≤ (5 * z) / (3 * x + y) + (5 * x) / (y + 3 * z) + (2 * y) / (x + z) :=
by sorry

end min_value_fractions_l215_215199


namespace continuity_f_at_3_l215_215467

noncomputable def f (x : ℝ) := if x ≤ 3 then 3 * x^2 - 5 else 18 * x - 32

theorem continuity_f_at_3 : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x - f 3) < ε := by
  intro ε ε_pos
  use 1
  simp
  sorry

end continuity_f_at_3_l215_215467


namespace compare_logs_l215_215551

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem compare_logs : a > b ∧ b > c := by
  -- Proof will be written here, currently placeholder
  sorry

end compare_logs_l215_215551


namespace Triangle_Equality_l215_215448

noncomputable theory
open EuclideanGeometry

variables {A C E B D F : Point}

-- Conditions
axiom Points_on_lines : OnLine B (Segment A C) ∧ OnLine D (Segment A E)
axiom Intersection_property : ∃ F, IntersectAt CD BE F 
axiom Given_condition : distance A B + distance B F = distance A D + distance D F

-- To prove
theorem Triangle_Equality :
  ∀ (P : Triangle),
  P ⟨A, C, E⟩ → 
  Points_on_lines → 
  Intersection_property → 
  Given_condition → 
  distance A C + distance C F = distance A E + distance E F := 
by
  intros P h₁ h₂ h₃ h₄
  sorry

end Triangle_Equality_l215_215448


namespace polygon_sides_l215_215085

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end polygon_sides_l215_215085


namespace quadratic_inequality_solution_l215_215071

theorem quadratic_inequality_solution (x : ℝ) : (-x^2 + 5 * x - 4 < 0) ↔ (1 < x ∧ x < 4) :=
sorry

end quadratic_inequality_solution_l215_215071


namespace qualified_flour_l215_215922

-- Define the acceptable weight range
def acceptable_range (w : ℝ) : Prop :=
  24.75 ≤ w ∧ w ≤ 25.25

-- Define the weight options
def optionA : ℝ := 24.70
def optionB : ℝ := 24.80
def optionC : ℝ := 25.30
def optionD : ℝ := 25.51

-- The statement to be proved
theorem qualified_flour : acceptable_range optionB ∧ ¬acceptable_range optionA ∧ ¬acceptable_range optionC ∧ ¬acceptable_range optionD :=
by
  sorry

end qualified_flour_l215_215922


namespace midpoint_of_AQ_l215_215844

theorem midpoint_of_AQ 
  {A B C P Q M : Type*} [VectorSpace ℝ (V := Type*) ]
  (hP : ∃ t : ℝ, ∀ (CA CB : V), t • CA + (1 - t) • CB = (2 / 3: ℝ) • CA + (1 / 3: ℝ) • CB ∧ P ∈ AB)
  (hQ : Q ∈ Segment B C ∧ ∀(B C : V), Q = (1 / 2) • B + (1 / 2) • C)
  (hM : ∃ t : ℝ, ∀ (AM AQ : V), M ∈ Segment A Q ∧ t • CP = CM) :
  ∃ λ : ℝ, λ = (1 / 2) ∧ ∀ (AQ : V), M = (1 / 2) • A + (1 / 2) • Q := 
sorry

end midpoint_of_AQ_l215_215844


namespace curve_crosses_itself_at_point_l215_215268

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  (2 * t₁^2 + 1 = 2 * t₂^2 + 1) ∧ 
  (2 * t₁^3 - 6 * t₁^2 + 8 = 2 * t₂^3 - 6 * t₂^2 + 8) ∧ 
  2 * t₁^2 + 1 = 1 ∧ 2 * t₁^3 - 6 * t₁^2 + 8 = 8 :=
by
  sorry

end curve_crosses_itself_at_point_l215_215268


namespace determinant_example_l215_215791

def determinant_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

theorem determinant_example : determinant_2x2 5 (-4) 2 3 = 23 := 
by 
  sorry

end determinant_example_l215_215791


namespace consecutive_integers_sum_to_thirty_unique_sets_l215_215690

theorem consecutive_integers_sum_to_thirty_unique_sets :
  (∃ a n : ℕ, a ≥ 3 ∧ n ≥ 2 ∧ n * (2 * a + n - 1) = 60) ↔ ∃! a n : ℕ, a ≥ 3 ∧ n ≥ 2 ∧ n * (2 * a + n - 1) = 60 :=
by
  sorry

end consecutive_integers_sum_to_thirty_unique_sets_l215_215690


namespace jackie_first_tree_height_l215_215715

theorem jackie_first_tree_height
  (h : ℝ)
  (avg_height : (h + 2 * (h / 2) + (h + 200)) / 4 = 800) :
  h = 1000 :=
by
  sorry

end jackie_first_tree_height_l215_215715


namespace popsicle_sticks_sum_l215_215765

-- Define the number of popsicle sticks each person has
def Gino_popsicle_sticks : Nat := 63
def my_popsicle_sticks : Nat := 50

-- Formulate the theorem stating the sum of popsicle sticks
theorem popsicle_sticks_sum : Gino_popsicle_sticks + my_popsicle_sticks = 113 := by
  sorry

end popsicle_sticks_sum_l215_215765


namespace algebra_expression_evaluation_l215_215700

theorem algebra_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 5) : -2 * a^2 - 4 * a + 5 = -7 :=
by
  sorry

end algebra_expression_evaluation_l215_215700


namespace balloons_given_by_mom_l215_215232

def num_balloons_initial : ℕ := 26
def num_balloons_total : ℕ := 60

theorem balloons_given_by_mom :
  (num_balloons_total - num_balloons_initial) = 34 := 
by
  sorry

end balloons_given_by_mom_l215_215232


namespace positive_solution_sqrt_a_sub_b_l215_215794

theorem positive_solution_sqrt_a_sub_b (a b : ℕ) (x : ℝ) 
  (h_eq : x^2 + 14 * x = 32) 
  (h_form : x = Real.sqrt a - b) 
  (h_pos_nat : a > 0 ∧ b > 0) : 
  a + b = 88 := 
by
  sorry

end positive_solution_sqrt_a_sub_b_l215_215794


namespace product_mod_32_l215_215008

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215008


namespace quadratic_other_x_intercept_l215_215421

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → (a * x^2 + b * x + c) = -3)
  (h_intercept : ∀ x, x = 1 → (a * x^2 + b * x + c) = 0) : 
  ∃ x : ℝ, x = 9 ∧ (a * x^2 + b * x + c) = 0 :=
sorry

end quadratic_other_x_intercept_l215_215421


namespace find_x_l215_215539

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉₊ * x = 198) : x = 13.2 :=
by
  sorry

end find_x_l215_215539


namespace find_fraction_l215_215770

theorem find_fraction (x f : ℝ) (h₁ : x = 140) (h₂ : 0.65 * x = f * x - 21) : f = 0.8 :=
by
  sorry

end find_fraction_l215_215770


namespace fraction_of_income_from_tips_l215_215764

theorem fraction_of_income_from_tips 
  (salary tips : ℝ)
  (h1 : tips = (7/4) * salary) 
  (total_income : ℝ)
  (h2 : total_income = salary + tips) :
  (tips / total_income) = (7 / 11) :=
by
  sorry

end fraction_of_income_from_tips_l215_215764


namespace baskets_delivered_l215_215618

theorem baskets_delivered 
  (peaches_per_basket : ℕ := 25)
  (boxes : ℕ := 8)
  (peaches_per_box : ℕ := 15)
  (peaches_eaten : ℕ := 5)
  (peaches_in_boxes := boxes * peaches_per_box) 
  (total_peaches := peaches_in_boxes + peaches_eaten) : 
  total_peaches / peaches_per_basket = 5 :=
by
  sorry

end baskets_delivered_l215_215618


namespace y_coords_diff_of_ellipse_incircle_area_l215_215555

theorem y_coords_diff_of_ellipse_incircle_area
  (x1 y1 x2 y2 : ℝ)
  (F1 F2 : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : F1 = (-4, 0))
  (h4 : F2 = (4, 0))
  (h5 : 4 * (|y1 - y2|) = 20)
  (h6 : ∃ (x : ℝ), (x / 25)^2 + (y1 / 9)^2 = 1 ∧ (x / 25)^2 + (y2 / 9)^2 = 1) :
  |y1 - y2| = 5 :=
sorry

end y_coords_diff_of_ellipse_incircle_area_l215_215555


namespace max_len_sequence_x_l215_215658

theorem max_len_sequence_x :
  ∃ x : ℕ, 3088 < x ∧ x < 3091 :=
sorry

end max_len_sequence_x_l215_215658


namespace jerrys_current_average_score_l215_215338

theorem jerrys_current_average_score (A : ℝ) (h1 : 3 * A + 98 = 4 * (A + 2)) : A = 90 :=
by
  sorry

end jerrys_current_average_score_l215_215338


namespace value_of_x_plus_y_l215_215446

theorem value_of_x_plus_y (x y : ℤ) (h1 : x + 2 = 10) (h2 : y - 1 = 6) : x + y = 15 :=
by
  sorry

end value_of_x_plus_y_l215_215446


namespace consecutive_integers_l215_215342

theorem consecutive_integers (a b c : ℝ)
  (h1 : ∃ k : ℤ, a + b = k ∧ b + c = k + 1 ∧ c + a = k + 2)
  (h2 : ∃ k : ℤ, b + c = 2 * k + 1) :
  ∃ n : ℤ, a = n + 2 ∧ b = n + 1 ∧ c = n := 
sorry

end consecutive_integers_l215_215342


namespace product_mod_32_is_15_l215_215018

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l215_215018


namespace product_of_solutions_of_t_squared_eq_49_l215_215816

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l215_215816


namespace distance_between_first_and_last_tree_l215_215870

theorem distance_between_first_and_last_tree
  (n : ℕ)
  (trees : ℕ)
  (dist_between_first_and_fourth : ℕ)
  (eq_dist : ℕ):
  trees = 6 ∧ dist_between_first_and_fourth = 60 ∧ eq_dist = dist_between_first_and_fourth / 3 ∧ n = (trees - 1) * eq_dist → n = 100 :=
by
  intro h
  sorry

end distance_between_first_and_last_tree_l215_215870


namespace range_of_b_l215_215681

theorem range_of_b :
  (∀ b, (∀ x : ℝ, x ≥ 1 → Real.log (2^x - b) ≥ 0) → b ≤ 1) :=
sorry

end range_of_b_l215_215681


namespace product_of_odd_primes_mod_32_l215_215028

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end product_of_odd_primes_mod_32_l215_215028


namespace percentage_exceed_l215_215447

theorem percentage_exceed (x y : ℝ) (h : y = x + 0.2 * x) :
  (y - x) / x * 100 = 20 :=
by
  -- Proof goes here
  sorry

end percentage_exceed_l215_215447


namespace remainder_M_mod_32_l215_215031

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215031


namespace line_not_in_first_quadrant_l215_215842

theorem line_not_in_first_quadrant (t : ℝ) : 
  (∀ x y : ℝ, ¬ ((0 < x ∧ 0 < y) ∧ (2 * t - 3) * x + y + 6 = 0)) ↔ t ≥ 3 / 2 :=
by
  sorry

end line_not_in_first_quadrant_l215_215842


namespace vulgar_fraction_of_decimal_l215_215103

theorem vulgar_fraction_of_decimal :
  (0.34 : ℚ) = 17 / 50 :=
by 
  norm_num

end vulgar_fraction_of_decimal_l215_215103


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l215_215667

def consecutive_primes (n : Nat) : Prop :=
  -- Define what it means to be 4 consecutive prime numbers
  Nat.Prime n ∧ Nat.Prime (n + 2) ∧ Nat.Prime (n + 6) ∧ Nat.Prime (n + 8)

def sum_of_consecutive_primes (n : Nat) : Nat :=
  n + (n + 2) + (n + 6) + (n + 8)

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ n, n > 10 ∧ consecutive_primes n ∧ sum_of_consecutive_primes n % 5 = 0 ∧ sum_of_consecutive_primes n = 60 :=
by
  sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l215_215667


namespace P_and_S_could_not_be_fourth_l215_215608

-- Define the relationships between the runners using given conditions
variables (P Q R S T U : ℕ)

axiom P_beats_Q : P < Q
axiom Q_beats_R : Q < R
axiom R_beats_S : R < S
axiom T_after_P_before_R : P < T ∧ T < R
axiom U_before_R_after_S : S < U ∧ U < R

-- Prove that P and S could not be fourth
theorem P_and_S_could_not_be_fourth : ¬((Q < U ∧ U < P) ∨ (Q > S ∧ S < P)) :=
by sorry

end P_and_S_could_not_be_fourth_l215_215608


namespace special_collection_books_l215_215917

theorem special_collection_books (loaned_books : ℕ) (returned_percentage : ℝ) (end_of_month_books : ℕ)
    (H1 : loaned_books = 160)
    (H2 : returned_percentage = 0.65)
    (H3 : end_of_month_books = 244) :
    let books_returned := returned_percentage * loaned_books
    let books_not_returned := loaned_books - books_returned
    let original_books := end_of_month_books + books_not_returned
    original_books = 300 :=
by
  sorry

end special_collection_books_l215_215917


namespace marys_birthday_l215_215470

theorem marys_birthday (M : ℝ) (h1 : (3 / 4) * M - (3 / 20) * M = 60) : M = 100 := by
  -- Leave the proof as sorry for now
  sorry

end marys_birthday_l215_215470


namespace product_mod_32_l215_215005

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l215_215005


namespace repaired_shoes_lifespan_l215_215771

-- Definitions of given conditions
def cost_repair : Float := 11.50
def cost_new : Float := 28.00
def lifespan_new : Float := 2.0
def percentage_increase : Float := 21.73913043478261 / 100

-- Cost per year of new shoes
def cost_per_year_new : Float := cost_new / lifespan_new

-- Cost per year of repaired shoes
def cost_per_year_repair (T : Float) : Float := cost_repair / T

-- Theorem statement (goal)
theorem repaired_shoes_lifespan (T : Float) (h : cost_per_year_new = cost_per_year_repair T * (1 + percentage_increase)) : T = 0.6745 :=
by
  sorry

end repaired_shoes_lifespan_l215_215771


namespace total_rent_paid_l215_215058

theorem total_rent_paid
  (weekly_rent : ℕ) (num_weeks : ℕ) 
  (hrent : weekly_rent = 388)
  (hweeks : num_weeks = 1359) :
  weekly_rent * num_weeks = 527292 := 
by
  sorry

end total_rent_paid_l215_215058


namespace monotonic_increasing_interval_l215_215079

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (Real.sqrt (2 * x - x ^ 2))

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 1 ≤ x ∧ x < 2 → ∀ x1 x2, x1 < x2 → f x1 ≤ f x2 :=
by
  sorry

end monotonic_increasing_interval_l215_215079


namespace alec_votes_l215_215925

variable (students totalVotes goalVotes neededVotes : ℕ)

theorem alec_votes (h1 : students = 60)
                   (h2 : goalVotes = 3 * students / 4)
                   (h3 : totalVotes = students / 2 + 5 + (students - (students / 2 + 5)) / 5)
                   (h4 : neededVotes = goalVotes - totalVotes) :
                   neededVotes = 5 :=
by sorry

end alec_votes_l215_215925


namespace total_distance_is_3_miles_l215_215597

-- Define conditions
def running_speed := 6   -- mph
def walking_speed := 2   -- mph
def running_time := 20 / 60   -- hours
def walking_time := 30 / 60   -- hours

-- Define total distance
def total_distance := (running_speed * running_time) + (walking_speed * walking_time)

theorem total_distance_is_3_miles : total_distance = 3 :=
by
  sorry

end total_distance_is_3_miles_l215_215597


namespace digits_C_not_make_1C34_divisible_by_4_l215_215964

theorem digits_C_not_make_1C34_divisible_by_4 :
  ∀ (C : ℕ), (C ≥ 0) ∧ (C ≤ 9) → ¬ (1034 + 100 * C) % 4 = 0 :=
by sorry

end digits_C_not_make_1C34_divisible_by_4_l215_215964


namespace rainbow_nerds_total_l215_215330

theorem rainbow_nerds_total
  (purple yellow green red blue : ℕ)
  (h1 : purple = 10)
  (h2 : yellow = purple + 4)
  (h3 : green = yellow - 2)
  (h4 : red = 3 * green)
  (h5 : blue = red / 2) :
  (purple + yellow + green + red + blue = 90) :=
by
  sorry

end rainbow_nerds_total_l215_215330


namespace probability_draw_l215_215234

theorem probability_draw (pA_win pA_not_lose : ℝ) (h1 : pA_win = 0.3) (h2 : pA_not_lose = 0.8) :
  pA_not_lose - pA_win = 0.5 :=
by 
  sorry

end probability_draw_l215_215234


namespace line_intersects_ellipse_l215_215371

theorem line_intersects_ellipse (k : ℝ) (m : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → (x^2 / 5) + (y^2 / m) = 1 → True) ↔ (1 < m ∧ m < 5) ∨ (5 < m) :=
by
  sorry

end line_intersects_ellipse_l215_215371


namespace concession_stand_total_revenue_l215_215387

theorem concession_stand_total_revenue :
  let hot_dog_price : ℝ := 1.50
  let soda_price : ℝ := 0.50
  let total_items_sold : ℕ := 87
  let hot_dogs_sold : ℕ := 35
  let sodas_sold := total_items_sold - hot_dogs_sold
  let revenue_from_hot_dogs := hot_dogs_sold * hot_dog_price
  let revenue_from_sodas := sodas_sold * soda_price
  revenue_from_hot_dogs + revenue_from_sodas = 78.50 :=
by {
  -- Proof will go here
  sorry
}

end concession_stand_total_revenue_l215_215387


namespace find_x_given_ratio_constant_l215_215165

theorem find_x_given_ratio_constant (x y : ℚ) (k : ℚ)
  (h1 : ∀ x y, (2 * x - 5) / (y + 20) = k)
  (h2 : (2 * 7 - 5) / (6 + 20) = k)
  (h3 : y = 21) :
  x = 499 / 52 :=
by
  sorry

end find_x_given_ratio_constant_l215_215165


namespace jane_mean_score_l215_215717

def quiz_scores : List ℕ := [85, 90, 95, 80, 100]

def total_scores : ℕ := quiz_scores.length

def sum_scores : ℕ := quiz_scores.sum

def mean_score : ℕ := sum_scores / total_scores

theorem jane_mean_score : mean_score = 90 := by
  sorry

end jane_mean_score_l215_215717


namespace prime_quadratic_roots_l215_215695

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_integer_roots (a b c : ℤ) : Prop :=
  ∃ x y : ℤ, (a * x * x + b * x + c = 0) ∧ (a * y * y + b * y + c = 0)

theorem prime_quadratic_roots (p : ℕ) (h_prime : is_prime p)
  (h_roots : has_integer_roots 1 (p : ℤ) (-444 * (p : ℤ))) :
  31 < p ∧ p ≤ 41 :=
sorry

end prime_quadratic_roots_l215_215695


namespace lucy_cardinals_vs_blue_jays_l215_215591

noncomputable def day1_cardinals : ℕ := 3
noncomputable def day1_blue_jays : ℕ := 2
noncomputable def day2_cardinals : ℕ := 3
noncomputable def day2_blue_jays : ℕ := 3
noncomputable def day3_cardinals : ℕ := 4
noncomputable def day3_blue_jays : ℕ := 2

theorem lucy_cardinals_vs_blue_jays :
  (day1_cardinals + day2_cardinals + day3_cardinals) - (day1_blue_jays + day2_blue_jays + day3_blue_jays) = 3 :=
  by sorry

end lucy_cardinals_vs_blue_jays_l215_215591


namespace right_triangle_ratio_proof_l215_215713

-- Declaring the main problem context
noncomputable def right_triangle_ratio : Prop :=
  ∃ (A B C D E F : ℝ × ℝ), 
    ∃ (angle_A angle_B angle_C : ℝ), 
      ∃ (inradius circumradius : ℝ), 
        -- Conditions
        (angle_A + angle_B = π / 2) ∧
        (D = foot_of_altitude A B C) ∧
        (E = intersection_of_angle_bisectors (angle A C D) (angle B C D)) ∧
        (F = intersection_of_angle_bisectors (angle B C D) (angle A C D)) ∧
        -- Computation of inradius and circumradius here 
        (inradius = compute_inradius A B C) ∧
        (circumradius = compute_circumradius C E F) ∧
        -- Proven ratio
        (inradius / circumradius = (sqrt 2) / 2)

-- Placeholder function definitions for conditions
def foot_of_altitude (A B C : ℝ × ℝ) : ℝ × ℝ := sorry
def intersection_of_angle_bisectors (α β : ℝ) : ℝ × ℝ := sorry
def compute_inradius (A B C : ℝ × ℝ) : ℝ := sorry
def compute_circumradius (C E F : ℝ × ℝ) : ℝ := sorry

-- The proof statement of the ratio problem
theorem right_triangle_ratio_proof : right_triangle_ratio := by
  sorry

end right_triangle_ratio_proof_l215_215713


namespace veromont_clicked_ads_l215_215505

def ads_on_first_page := 12
def ads_on_second_page := 2 * ads_on_first_page
def ads_on_third_page := ads_on_second_page + 24
def ads_on_fourth_page := (3 / 4) * ads_on_second_page
def total_ads := ads_on_first_page + ads_on_second_page + ads_on_third_page + ads_on_fourth_page
def ads_clicked := (2 / 3) * total_ads

theorem veromont_clicked_ads : ads_clicked = 68 := 
by
  sorry

end veromont_clicked_ads_l215_215505


namespace product_of_odd_primes_mod_32_l215_215043

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l215_215043


namespace proof_G_eq_BC_eq_D_eq_AB_AC_l215_215996

-- Let's define the conditions of the problem first
variables (A B C O D G F E : Type) [Field A] [Field B] [Field C] [Field O] [Field D] [Field G] [Field F] [Field E]

-- Given triangle ABC with circumcenter O
variable {triangle_ABC: Prop}

-- Given point D on line segment BC
variable (D_on_BC : Prop)

-- Given circle Gamma with diameter OD
variable (circle_Gamma : Prop)

-- Given circles Gamma_1 and Gamma_2 are circumcircles of triangles ABD and ACD respectively
variable (circle_Gamma1 : Prop)
variable (circle_Gamma2 : Prop)

-- Given points F and E as intersection points
variable (intersect_F : Prop)
variable (intersect_E : Prop)

-- Given G as the second intersection point of the circumcircles of triangles BED and DFC
variable (second_intersect_G : Prop)

-- Prove that the condition for point G to be equidistant from points B and C is that point D is equidistant from lines AB and AC
theorem proof_G_eq_BC_eq_D_eq_AB_AC : 
  triangle_ABC ∧ D_on_BC ∧ circle_Gamma ∧ circle_Gamma1 ∧ circle_Gamma2 ∧ intersect_F ∧ intersect_E ∧ second_intersect_G → 
  G_dist_BC ↔ D_dist_AB_AC :=
by
  sorry

end proof_G_eq_BC_eq_D_eq_AB_AC_l215_215996


namespace solution_couples_l215_215722

noncomputable def find_couples (n m k : ℕ) : Prop :=
  ∃ t : ℕ, (n = 2^k - 1 - t ∧ m = (Nat.factorial (2^k)) / 2^(2^k - 1 - t))

theorem solution_couples (k : ℕ) :
  ∃ n m : ℕ, (Nat.factorial (2^k)) = 2^n * m ∧ find_couples n m k :=
sorry

end solution_couples_l215_215722


namespace distribute_balls_into_boxes_l215_215320

theorem distribute_balls_into_boxes : 
  ∀ (balls : ℕ) (boxes : ℕ), balls = 6 ∧ boxes = 3 → boxes^balls = 729 :=
by
  intros balls boxes h
  have hb : balls = 6 := h.1
  have hbox : boxes = 3 := h.2
  rw [hb, hbox]
  show 3^6 = 729
  exact Nat.pow 3 6 -- this would expand to the actual computation
  sorry

end distribute_balls_into_boxes_l215_215320


namespace arccos_neg_one_eq_pi_l215_215939

theorem arccos_neg_one_eq_pi : real.arccos (-1) = real.pi :=
by sorry

end arccos_neg_one_eq_pi_l215_215939


namespace power_division_l215_215404

theorem power_division (a : ℝ) (h : a ≠ 0) : ((-a)^6) / (a^3) = a^3 := by
  sorry

end power_division_l215_215404


namespace possible_last_three_digits_product_l215_215621

def lastThreeDigits (n : ℕ) : ℕ := n % 1000

theorem possible_last_three_digits_product (a b c : ℕ) (ha : a > 1000) (hb : b > 1000) (hc : c > 1000)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (a + c) % 10 = b % 10)
  (h3 : (b + c) % 10 = a % 10) :
  lastThreeDigits (a * b * c) = 0 ∨ lastThreeDigits (a * b * c) = 250 ∨ lastThreeDigits (a * b * c) = 500 ∨ lastThreeDigits (a * b * c) = 750 := 
sorry

end possible_last_three_digits_product_l215_215621


namespace find_length_DY_l215_215869

noncomputable def length_DY : Real :=
    let AE := 2
    let AY := 4 * AE
    let DY  := Real.sqrt (66 + Real.sqrt 5)
    DY

theorem find_length_DY : length_DY = Real.sqrt (66 + Real.sqrt 5) := 
  by
    sorry

end find_length_DY_l215_215869


namespace inequality_proof_l215_215144

theorem inequality_proof {a b c d e f : ℝ} (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
sorry

end inequality_proof_l215_215144


namespace rounding_to_one_decimal_place_l215_215905

def number_to_round : Float := 5.049

def rounded_value : Float := 5.0

theorem rounding_to_one_decimal_place :
  (Float.round (number_to_round * 10) / 10) = rounded_value :=
by
  sorry

end rounding_to_one_decimal_place_l215_215905


namespace percentage_william_land_l215_215389

-- Definitions of the given conditions
def total_tax_collected : ℝ := 3840
def william_tax : ℝ := 480

-- Proof statement
theorem percentage_william_land :
  ((william_tax / total_tax_collected) * 100) = 12.5 :=
by
  sorry

end percentage_william_land_l215_215389


namespace total_seats_theater_l215_215510

theorem total_seats_theater (a1 an d n Sn : ℕ) 
    (h1 : a1 = 12) 
    (h2 : d = 2) 
    (h3 : an = 48) 
    (h4 : an = a1 + (n - 1) * d) 
    (h5 : Sn = n * (a1 + an) / 2) : 
    Sn = 570 := 
sorry

end total_seats_theater_l215_215510


namespace fraction_numerator_l215_215738

theorem fraction_numerator (x : ℕ) (h1 : 4 * x - 4 > 0) (h2 : (x : ℚ) / (4 * x - 4) = 3 / 8) : x = 3 :=
by {
  sorry
}

end fraction_numerator_l215_215738


namespace fraction_exponentiation_example_l215_215507

theorem fraction_exponentiation_example :
  (5/3)^4 = 625/81 :=
by
  sorry

end fraction_exponentiation_example_l215_215507


namespace max_f_value_l215_215672

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℝ := (S_n n : ℝ) / ((n + 32) * S_n (n + 1))

theorem max_f_value : ∃ n : ℕ, f n = 1 / 50 := by
  sorry

end max_f_value_l215_215672


namespace power_complex_l215_215416

theorem power_complex (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : -64 = (-4)^3) (h3 : (a^b)^((3:ℝ) / 2) = a^(b * ((3:ℝ) / 2))) (h4 : (-4:ℂ)^(1/2) = 2 * i) :
  (↑(-64):ℂ) ^ (3/2) = 512 * i :=
by
  sorry

end power_complex_l215_215416


namespace product_of_roots_l215_215466

-- Define the quadratic function in terms of a, b, c
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the conditions
variables (a b c y : ℝ)

-- Given conditions from the problem
def condition_1 := ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2
def condition_2 := quadratic a b c y = 0
def condition_3 := quadratic a b c (4 * y) = 0

-- The statement to be proved
theorem product_of_roots (a b c y : ℝ) 
  (h1: ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2)
  (h2: quadratic a b c y = 0) 
  (h3: quadratic a b c (4 * y) = 0) :
  ∃ x1 x2, (quadratic a b c x = 0 → (x1 = y ∧ x2 = 4 * y) ∨ (x1 = 4 * y ∧ x2 = y)) ∧ x1 * x2 = 4 * y^2 :=
by
  sorry

end product_of_roots_l215_215466


namespace necessary_but_not_sufficient_l215_215391

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 1 = 0) ↔ (x = -1 ∨ x = 1) ∧ (x - 1 = 0) → (x^2 - 1 = 0) ∧ ¬((x^2 - 1 = 0) → (x - 1 = 0)) := 
by sorry

end necessary_but_not_sufficient_l215_215391


namespace distribution_of_balls_l215_215319

theorem distribution_of_balls (n k : ℕ) (h_n : n = 6) (h_k : k = 3) : k^n = 729 := by
  rw [h_n, h_k]
  exact rfl

end distribution_of_balls_l215_215319


namespace inequality_proof_l215_215156

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
by 
  sorry

end inequality_proof_l215_215156


namespace Sally_lost_20_Pokemon_cards_l215_215867

theorem Sally_lost_20_Pokemon_cards (original_cards : ℕ) (received_cards : ℕ) (final_cards : ℕ) (lost_cards : ℕ) 
  (h1 : original_cards = 27) 
  (h2 : received_cards = 41) 
  (h3 : final_cards = 48) 
  (h4 : original_cards + received_cards - lost_cards = final_cards) : 
  lost_cards = 20 := 
sorry

end Sally_lost_20_Pokemon_cards_l215_215867


namespace remainder_M_mod_32_l215_215040

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215040


namespace average_next_seven_consecutive_is_correct_l215_215068

-- Define the sum of seven consecutive integers starting at x.
def sum_seven_consecutive_integers (x : ℕ) : ℕ := 7 * x + 21

-- Define the next sequence of seven integers starting from y + 1.
def average_next_seven_consecutive_integers (x : ℕ) : ℕ :=
  let y := sum_seven_consecutive_integers x
  let start := y + 1
  (start + (start + 1) + (start + 2) + (start + 3) + (start + 4) + (start + 5) + (start + 6)) / 7

-- Problem statement
theorem average_next_seven_consecutive_is_correct (x : ℕ) : 
  average_next_seven_consecutive_integers x = 7 * x + 25 :=
by
  sorry

end average_next_seven_consecutive_is_correct_l215_215068


namespace min_prime_factor_sum_l215_215051

theorem min_prime_factor_sum (x y a b c d : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : 5 * x^7 = 13 * y^11)
  (h4 : x = 13^6 * 5^7) (h5 : a = 13) (h6 : b = 5) (h7 : c = 6) (h8 : d = 7) : 
  a + b + c + d = 31 :=
by
  sorry

end min_prime_factor_sum_l215_215051


namespace incorrect_games_less_than_three_fourths_l215_215453

/-- In a round-robin chess tournament, each participant plays against every other participant exactly once.
A win earns one point, a draw earns half a point, and a loss earns zero points.
We will call a game incorrect if the player who won the game ends up with fewer total points than the player who lost.

1. Prove that incorrect games make up less than 3/4 of the total number of games in the tournament.
2. Prove that in part (1), the number 3/4 cannot be replaced with a smaller number.
--/
theorem incorrect_games_less_than_three_fourths {n : ℕ} (h : n > 1) :
  ∃ m, (∃ (incorrect_games total_games : ℕ), m = incorrect_games ∧ total_games = (n * (n - 1)) / 2 
    ∧ (incorrect_games : ℚ) / total_games < 3 / 4) 
    ∧ (∀ m' : ℚ, m' ≥ 0 → m = incorrect_games ∧ (incorrect_games : ℚ) / total_games < m' → m' ≥ 3 / 4) :=
sorry

end incorrect_games_less_than_three_fourths_l215_215453


namespace perfect_squares_less_than_20000_representable_l215_215064

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the difference of two consecutive perfect squares
def consecutive_difference (b : ℕ) : ℕ :=
  (b + 1) ^ 2 - b ^ 2

-- Define the condition under which the perfect square is less than 20000
def less_than_20000 (n : ℕ) : Prop :=
  n < 20000

-- Define the main problem statement using the above definitions
theorem perfect_squares_less_than_20000_representable :
  ∃ count : ℕ, (∀ n : ℕ, (is_perfect_square n) ∧ (less_than_20000 n) →
  ∃ b : ℕ, n = consecutive_difference b) ∧ count = 69 :=
sorry

end perfect_squares_less_than_20000_representable_l215_215064


namespace focus_of_parabola_l215_215533

theorem focus_of_parabola : 
  ∃(h k : ℚ), ((∀ x : ℚ, -2 * x^2 - 6 * x + 1 = -2 * (x + 3 / 2)^2 + 11 / 2) ∧ 
  (∃ a : ℚ, (a = -2 / 8) ∧ (h = -3/2) ∧ (k = 11/2 + a)) ∧ 
  (h, k) = (-3/2, 43 / 8)) :=
sorry

end focus_of_parabola_l215_215533


namespace range_of_a_part1_range_of_a_part2_l215_215687

theorem range_of_a_part1 (a : ℝ) :
  (∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) → 0 < a ∧ a < 4 :=
sorry

theorem range_of_a_part2 (a : ℝ) :
  ((∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) ∨ (∃ x : ℝ, x^2 - x + a = 0)) ∧ ¬((∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) ∧ (∃ x : ℝ, x^2 - x + a = 0)) →
  a ≤ 0 ∨ (1 / 4) < a ∧ a < 4 :=
sorry

end range_of_a_part1_range_of_a_part2_l215_215687


namespace distribution_of_balls_l215_215318

theorem distribution_of_balls (n k : ℕ) (h_n : n = 6) (h_k : k = 3) : k^n = 729 := by
  rw [h_n, h_k]
  exact rfl

end distribution_of_balls_l215_215318


namespace toy_poodle_height_l215_215492

theorem toy_poodle_height 
  (SP MP TP : ℕ)
  (h1 : SP = MP + 8)
  (h2 : MP = TP + 6)
  (h3 : SP = 28) 
  : TP = 14 := 
    by sorry

end toy_poodle_height_l215_215492


namespace sufficient_but_not_necessary_for_monotonic_l215_215552

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ y → f x ≤ f y

noncomputable def is_sufficient_condition (P Q : Prop) : Prop :=
P → Q

noncomputable def is_not_necessary_condition (P Q : Prop) : Prop :=
¬ Q → ¬ P

noncomputable def is_sufficient_but_not_necessary (P Q : Prop) : Prop :=
is_sufficient_condition P Q ∧ is_not_necessary_condition P Q

theorem sufficient_but_not_necessary_for_monotonic (f : ℝ → ℝ) :
  (∀ x, 0 ≤ deriv f x) → is_monotonically_increasing f :=
sorry

end sufficient_but_not_necessary_for_monotonic_l215_215552


namespace courier_total_travel_times_l215_215661

-- Define the conditions
variables (v1 v2 : ℝ) (t : ℝ)
axiom speed_condition_1 : v1 * (t + 16) = (v1 + v2) * t
axiom speed_condition_2 : v2 * (t + 9) = (v1 + v2) * t
axiom time_condition : t = 12

-- Define the total travel times
def total_travel_time_1 : ℝ := t + 16
def total_travel_time_2 : ℝ := t + 9

-- Proof problem statement
theorem courier_total_travel_times :
  total_travel_time_1 = 28 ∧ total_travel_time_2 = 21 :=
by
  sorry

end courier_total_travel_times_l215_215661


namespace length_of_room_l215_215739

theorem length_of_room (L : ℝ) 
  (h_width : 12 > 0) 
  (h_veranda_width : 2 > 0) 
  (h_area_veranda : (L + 4) * 16 - L * 12 = 140) : 
  L = 19 := 
by
  sorry

end length_of_room_l215_215739


namespace square_areas_l215_215999

theorem square_areas (s1 s2 s3 : ℕ)
  (h1 : s3 = s2 + 1)
  (h2 : s3 = s1 + 2)
  (h3 : s2 = 18)
  (h4 : s1 = s2 - 1) :
  s3^2 = 361 ∧ s2^2 = 324 ∧ s1^2 = 289 :=
by {
sorry
}

end square_areas_l215_215999


namespace raj_snow_removal_volume_l215_215208

theorem raj_snow_removal_volume :
  let length := 30
  let width := 4
  let depth_layer1 := 0.5
  let depth_layer2 := 0.3
  let volume_layer1 := length * width * depth_layer1
  let volume_layer2 := length * width * depth_layer2
  let total_volume := volume_layer1 + volume_layer2
  total_volume = 96 := by
sorry

end raj_snow_removal_volume_l215_215208


namespace total_expense_l215_215935

theorem total_expense (tanya_face_cost : ℕ) (tanya_face_qty : ℕ) (tanya_body_cost : ℕ) (tanya_body_qty : ℕ) 
  (tanya_total_expense : ℕ) (christy_multiplier : ℕ) (christy_total_expense : ℕ) (total_expense : ℕ) :
  tanya_face_cost = 50 →
  tanya_face_qty = 2 →
  tanya_body_cost = 60 →
  tanya_body_qty = 4 →
  tanya_total_expense = tanya_face_qty * tanya_face_cost + tanya_body_qty * tanya_body_cost →
  christy_multiplier = 2 →
  christy_total_expense = christy_multiplier * tanya_total_expense →
  total_expense = christy_total_expense + tanya_total_expense →
  total_expense = 1020 :=
by
  intros
  sorry

end total_expense_l215_215935


namespace angle_in_third_quadrant_l215_215190

theorem angle_in_third_quadrant (θ : ℝ) (h : θ = 2010) : ((θ % 360) > 180 ∧ (θ % 360) < 270) :=
by
  sorry

end angle_in_third_quadrant_l215_215190


namespace find_a2_l215_215553

theorem find_a2 
  (a1 a2 a3 : ℝ)
  (h1 : a1 * a2 * a3 = 15)
  (h2 : (3 / (a1 * 3 * a2)) + (15 / (3 * a2 * 5 * a3)) + (5 / (5 * a3 * a1)) = 3 / 5) :
  a2 = 3 :=
sorry

end find_a2_l215_215553


namespace bug_crawl_distance_l215_215911

-- Define the conditions
def initial_position : ℤ := -2
def first_move : ℤ := -6
def second_move : ℤ := 5

-- Define the absolute difference function (distance on a number line)
def abs_diff (a b : ℤ) : ℤ :=
  abs (b - a)

-- Define the total distance crawled function
def total_distance (p1 p2 p3 : ℤ) : ℤ :=
  abs_diff p1 p2 + abs_diff p2 p3

-- Prove that total distance starting at -2, moving to -6, and then to 5 is 15 units
theorem bug_crawl_distance : total_distance initial_position first_move second_move = 15 := by
  sorry

end bug_crawl_distance_l215_215911


namespace sum_first_15_terms_l215_215578

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

-- Conditions
def a_7 := 1
def a_9 := 5

-- Prove that S_15 = 45
theorem sum_first_15_terms : 
  ∃ (a d : ℤ), 
    (arithmetic_sequence a d 7 = a_7) ∧ 
    (arithmetic_sequence a d 9 = a_9) ∧ 
    (sum_first_n_terms a d 15 = 45) :=
sorry

end sum_first_15_terms_l215_215578


namespace range_of_a_l215_215988

noncomputable def y (a x : ℝ) : ℝ := a * Real.exp x + 3 * x
noncomputable def y_prime (a x : ℝ) : ℝ := a * Real.exp x + 3

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a * Real.exp x + 3 = 0 ∧ a * Real.exp x + 3 * x < 0) → a < -3 :=
by
  sorry

end range_of_a_l215_215988


namespace ryan_learning_hours_l215_215537

theorem ryan_learning_hours (total_hours : ℕ) (chinese_hours : ℕ) (english_hours : ℕ) 
  (h1 : total_hours = 3) (h2 : chinese_hours = 1) : 
  english_hours = 2 :=
by 
  sorry

end ryan_learning_hours_l215_215537


namespace constant_term_in_expansion_is_neg_42_l215_215882

-- Define the general term formula for (x - 1/x)^8
def binomial_term (r : ℕ) : ℤ :=
  (Nat.choose 8 r) * (-1 : ℤ) ^ r

-- Define the constant term in the product expansion
def constant_term : ℤ := 
  binomial_term 4 - 2 * binomial_term 5 

-- Problem statement: Prove the constant term is -42
theorem constant_term_in_expansion_is_neg_42 :
  constant_term = -42 := 
sorry

end constant_term_in_expansion_is_neg_42_l215_215882


namespace quadratic_roots_distinct_l215_215694

-- Define the conditions and the proof structure
theorem quadratic_roots_distinct (k : ℝ) (hk : k < 0) : 
  let a := 1
  let b := 1
  let c := k - 1
  let Δ := b*b - 4*a*c
  in Δ > 0 :=
by
  sorry

end quadratic_roots_distinct_l215_215694


namespace part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l215_215822

noncomputable def A : Set ℝ := { x : ℝ | 3 < x ∧ x < 10 }
noncomputable def B : Set ℝ := { x : ℝ | x^2 - 9 * x + 14 < 0 }
noncomputable def C (m : ℝ) : Set ℝ := { x : ℝ | 5 - m < x ∧ x < 2 * m }

theorem part_I_A_inter_B : A ∩ B = { x : ℝ | 3 < x ∧ x < 7 } :=
sorry

theorem part_I_complement_A_union_B :
  (Aᶜ) ∪ B = { x : ℝ | x < 7 ∨ x ≥ 10 } :=
sorry

theorem part_II_range_of_m :
  {m : ℝ | C m ⊆ A ∩ B} = {m : ℝ | m ≤ 2} :=
sorry

end part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l215_215822


namespace area_of_third_region_l215_215775

theorem area_of_third_region (A B C : ℝ) 
    (hA : A = 24) 
    (hB : B = 13) 
    (hTotal : A + B + C = 48) : 
    C = 11 := 
by 
  sorry

end area_of_third_region_l215_215775


namespace fencing_problem_l215_215900

noncomputable def fencingRequired (L A W F : ℝ) := (A = L * W) → (F = 2 * W + L)

theorem fencing_problem :
  fencingRequired 25 880 35.2 95.4 :=
by
  sorry

end fencing_problem_l215_215900


namespace arithmetic_sequence_general_formula_inequality_satisfaction_l215_215296

namespace Problem

-- Definitions for the sequences and the sum of terms
def a (n : ℕ) : ℕ := sorry -- define based on conditions
def S (n : ℕ) : ℕ := sorry -- sum of first n terms of {a_n}
def b (n : ℕ) : ℕ := 2 * (S (n + 1) - S n) * S n - n * (S (n + 1) + S n)

-- Part 1: Prove the general formula for the arithmetic sequence
theorem arithmetic_sequence_general_formula :
  (∀ n : ℕ, b n = 0) → (∀ n : ℕ, a n = 0 ∨ a n = n) :=
sorry

-- Part 2: Conditions for geometric sequences and inequality
def a_2n_minus_1 (n : ℕ) : ℕ := 2 ^ n
def a_2n (n : ℕ) : ℕ := 3 * 2 ^ (n - 1)
def b_2n (n : ℕ) : ℕ := sorry -- compute based on conditions
def b_2n_minus_1 (n : ℕ) : ℕ := sorry -- compute based on conditions

def b_condition (n : ℕ) : Prop := b_2n n < b_2n_minus_1 n

-- Prove the set of all positive integers n that satisfy the inequality
theorem inequality_satisfaction :
  { n : ℕ | b_condition n } = {1, 2, 3, 4, 5, 6} :=
sorry

end Problem

end arithmetic_sequence_general_formula_inequality_satisfaction_l215_215296


namespace value_of_x_abs_not_positive_l215_215238

theorem value_of_x_abs_not_positive {x : ℝ} : |4 * x - 6| = 0 → x = 3 / 2 :=
by
  sorry

end value_of_x_abs_not_positive_l215_215238


namespace total_days_spent_on_island_l215_215851

noncomputable def first_expedition_weeks := 3
noncomputable def second_expedition_weeks := first_expedition_weeks + 2
noncomputable def last_expedition_weeks := 2 * second_expedition_weeks
noncomputable def total_weeks := first_expedition_weeks + second_expedition_weeks + last_expedition_weeks
noncomputable def total_days := 7 * total_weeks

theorem total_days_spent_on_island : total_days = 126 := by
  sorry

end total_days_spent_on_island_l215_215851


namespace boat_speed_in_still_water_l215_215993

theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11) 
  (h2 : b - s = 3) : b = 7 :=
by
  sorry

end boat_speed_in_still_water_l215_215993


namespace smallest_base10_integer_exists_l215_215099

theorem smallest_base10_integer_exists : ∃ (n a b : ℕ), a > 2 ∧ b > 2 ∧ n = 1 * a + 3 ∧ n = 3 * b + 1 ∧ n = 10 := by
  sorry

end smallest_base10_integer_exists_l215_215099


namespace simple_interest_correct_l215_215508

theorem simple_interest_correct (P R T : ℝ) (hP : P = 400) (hR : R = 12.5) (hT : T = 2) : 
  (P * R * T) / 100 = 50 :=
by
  sorry -- Proof to be provided

end simple_interest_correct_l215_215508


namespace range_of_a_l215_215305

noncomputable def curve_y (a : ℝ) (x : ℝ) : ℝ := (a - 3) * x^3 + Real.log x
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ := x^3 - a * x^2 - 3 * x + 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ deriv (curve_y a) x = 0) ∧
  (∀ x ∈ Set.Icc (1 : ℝ) 2, 0 ≤ deriv (function_f a) x) → a ≤ 0 :=
by sorry

end range_of_a_l215_215305


namespace orthocenter_properties_l215_215343

open euclidean_geometry

variables {V : Type*} [inner_product_space ℝ V]

def orthocenter (A B C : V) : V := 
  sorry -- Details of computing the orthocenter.

def foot_of_altitude (A B C : V) : V := 
  sorry -- Details of computing the foot of the altitude from vertex A to BC.

theorem orthocenter_properties (A B C : V) (H_A H_B H_C : V)
  (H := orthocenter A B C) 
  (P := orthocenter A H_B H_C) 
  (Q := orthocenter C H_A H_B)
  (H_A = foot_of_altitude A B C)
  (H_B = foot_of_altitude B A C)
  (H_C = foot_of_altitude C A B) :
  dist P Q = dist H_C H_A :=
begin
  sorry
end

end orthocenter_properties_l215_215343


namespace bug_total_distance_l215_215910

theorem bug_total_distance :
  let start_position := -2
  let first_stop := -6
  let final_position := 5
  abs(first_stop - start_position) + abs(final_position - first_stop) = 15 :=
by
  sorry

end bug_total_distance_l215_215910


namespace equation_of_l_symmetric_point_l215_215170

/-- Define points O, A, B in the coordinate plane --/
def O := (0, 0)
def A := (2, 0)
def B := (3, 2)

/-- Define midpoint of OA --/
def midpoint_OA := ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

/-- Line l passes through midpoint_OA and B. Prove line l has equation y = x - 1 --/
theorem equation_of_l :
  ∃ l : ℝ → ℝ, (∀ x, l x = x - 1) :=
sorry

/-- Prove the symmetric point of A with respect to line l is (1, 1) --/
theorem symmetric_point :
  ∃ (a b : ℝ), (a, b) = (1, 1) ∧
                (b * (2 - 1)) / (a - 2) = -1 ∧
                b / 2 = (2 + a - 1) / 2 - 1 :=
sorry

end equation_of_l_symmetric_point_l215_215170


namespace triangle_exists_among_single_color_sticks_l215_215496

theorem triangle_exists_among_single_color_sticks
  (red yellow green : ℕ)
  (k y g K Y G : ℕ)
  (hk : k + y > G)
  (hy : y + g > K)
  (hg : g + k > Y)
  (hred : red = 100)
  (hyellow : yellow = 100)
  (hgreen : green = 100) :
  ∃ color : string, ∀ a b c : ℕ, (a = k ∨ a = K) → (b = k ∨ b = K) → (c = k ∨ c = K) → a + b > c :=
sorry

end triangle_exists_among_single_color_sticks_l215_215496


namespace train_length_proof_l215_215525

noncomputable def speed_km_per_hr : ℝ := 108
noncomputable def time_seconds : ℝ := 9
noncomputable def length_of_train : ℝ := 270
noncomputable def km_to_m : ℝ := 1000
noncomputable def hr_to_s : ℝ := 3600

theorem train_length_proof : 
  (speed_km_per_hr * (km_to_m / hr_to_s) * time_seconds) = length_of_train :=
  by
  sorry

end train_length_proof_l215_215525


namespace intersection_of_A_and_B_solve_inequality_l215_215245

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | x^2 - 16 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 ≥ 0}

-- Proof problem 1: Find A ∩ B
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)} :=
sorry

-- Proof problem 2: Solve the inequality with respect to x
theorem solve_inequality (a : ℝ) :
  if a = 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = ∅
  else if a > 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | 1 < x ∧ x < a}
  else
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | a < x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_solve_inequality_l215_215245


namespace expected_value_and_variance_of_Y_l215_215835
open Probability

variables {Ω : Type*} {X Y : Ω → ℝ}

-- Conditions
variables (h1 : ∀ ω, X ω + 2 * Y ω = 4)
          (h2 : ∀ s, MeasureTheory.MeasureSpace.has_distribution (λ ω, X ω) (MeasureTheory.Measure.norm N 1 (2 ^ 2)))

-- Expected value and variance of Y
theorem expected_value_and_variance_of_Y :
  E[Y] = 3 / 2 ∧ var Y = 1 :=
sorry

end expected_value_and_variance_of_Y_l215_215835


namespace determine_xyz_l215_215550

theorem determine_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 23 / 3 := 
by { sorry }

end determine_xyz_l215_215550


namespace students_in_class_l215_215331

theorem students_in_class
  (B : ℕ) (E : ℕ) (G : ℕ)
  (h1 : B = 12)
  (h2 : G + B = 22)
  (h3 : E = 10) :
  G + E + B = 32 :=
by
  sorry

end students_in_class_l215_215331


namespace contractor_engagement_days_l215_215251

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end contractor_engagement_days_l215_215251


namespace length_O_D1_l215_215333

-- Definitions for the setup of the cube and its faces, the center of the sphere, and the intersecting circles
def O : Point := sorry -- Center of the sphere and cube
def radius : ℝ := 10 -- Radius of the sphere

-- Intersection circles with given radii on specific faces of the cube
def r_ADA1D1 : ℝ := 1 -- Radius of the intersection circle on face ADA1D1
def r_A1B1C1D1 : ℝ := 1 -- Radius of the intersection circle on face A1B1C1D1
def r_CDD1C1 : ℝ := 3 -- Radius of the intersection circle on face CDD1C1

-- Distances derived from the problem
def OX1_sq : ℝ := radius^2 - r_ADA1D1^2
def OX2_sq : ℝ := radius^2 - r_A1B1C1D1^2
def OX_sq : ℝ := radius^2 - r_CDD1C1^2

-- To simplify, replace OX1, OX2, and OX with their squared values directly
def OX1_sq_calc : ℝ := 99
def OX2_sq_calc : ℝ := 99
def OX_sq_calc : ℝ := 91

theorem length_O_D1 : (OX1_sq_calc + OX2_sq_calc + OX_sq_calc) = 289 ↔ OD1 = 17 := by
  sorry

end length_O_D1_l215_215333


namespace polygon_sides_sum_720_l215_215087

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_sum_720_l215_215087


namespace calculation_result_l215_215221

theorem calculation_result : 2014 * (1/19 - 1/53) = 68 := by
  sorry

end calculation_result_l215_215221


namespace correct_assignment_statement_l215_215761

noncomputable def is_assignment_statement (stmt : String) : Bool :=
  -- Assume a simplified function that interprets whether the statement is an assignment
  match stmt with
  | "6 = M" => false
  | "M = -M" => true
  | "B = A = 8" => false
  | "x - y = 0" => false
  | _ => false

theorem correct_assignment_statement :
  is_assignment_statement "M = -M" = true :=
by
  rw [is_assignment_statement]
  exact rfl

end correct_assignment_statement_l215_215761


namespace approx_students_between_70_and_110_l215_215702

-- Definitions for the conditions given in the problem
noncomputable def mu : ℝ := 100
noncomputable def sigma_squared : ℝ := 100
noncomputable def sigma : ℝ := real.sqrt sigma_squared
noncomputable def num_students : ℕ := 1000

-- Reference probabilities for the normal distribution
noncomputable def prob_1_std_dev : ℝ := 0.6827
noncomputable def prob_3_std_dev : ℝ := 0.9973

-- Approximate calculation relevant to the problem
noncomputable def prob_70_to_110 : ℝ := (prob_1_std_dev + prob_3_std_dev) / 2
noncomputable def expected_students : ℝ := num_students * prob_70_to_110

-- The formal statement to show number of students scoring between 70 and 110 is approximately 840
theorem approx_students_between_70_and_110 : abs (expected_students - 840) < 1 := 
by
  sorry

end approx_students_between_70_and_110_l215_215702


namespace find_abc_l215_215372

def rearrangements (a b c : ℕ) : List ℕ :=
  [100 * a + 10 * b + c, 100 * a + 10 * c + b, 100 * b + 10 * a + c,
   100 * b + 10 * c + a, 100 * c + 10 * a + b, 100 * c + 10 * b + a]

theorem find_abc (a b c : ℕ) (habc : ℕ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (rearrangements a b c).sum = 2017 + habc →
  habc = 425 :=
by
  sorry

end find_abc_l215_215372


namespace count_triples_not_div_by_4_l215_215135

theorem count_triples_not_div_by_4 :
  {n : ℕ // n = 117 ∧ ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 5 → 1 ≤ b ∧ b ≤ 5 → 1 ≤ c ∧ c ≤ 5 → (a + b) * (a + c) * (b + c) % 4 ≠ 0} :=
sorry

end count_triples_not_div_by_4_l215_215135


namespace find_quotient_l215_215451

-- Constants representing the given conditions
def dividend : ℕ := 690
def divisor : ℕ := 36
def remainder : ℕ := 6

-- Theorem statement
theorem find_quotient : ∃ (quotient : ℕ), dividend = (divisor * quotient) + remainder ∧ quotient = 19 := 
by
  sorry

end find_quotient_l215_215451


namespace equation_of_circle_M_l215_215201

theorem equation_of_circle_M :
  ∃ (M : ℝ × ℝ), 
    (∃ a : ℝ, M = (a, 1 - 2 * a) ∧ (2 * a + 1 - 2 * a - 1) = 0 ) ∧
    (∃ r : ℝ, (M.1 - 3) ^ 2 + (M.2 - 0) ^ 2 = r ^ 2 ∧ (M.1 - 0) ^ 2 + (M.2 - 1) ^ 2 = r ^ 2 ) ∧
    (M = (1, -1) ∧ r = sqrt 5) ∧
    (∀ x y : ℝ, (x-1)^2 + (y+1)^2 = 5) :=
begin
  sorry
end

end equation_of_circle_M_l215_215201


namespace circle_center_is_21_l215_215959

theorem circle_center_is_21 : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 2 * y - 5 = 0 →
                                      ∃ h k : ℝ, h = 2 ∧ k = 1 ∧ (x - h)^2 + (y - k)^2 = 10 :=
by
  intro x y h_eq
  sorry

end circle_center_is_21_l215_215959


namespace problem_1_problem_2_l215_215827

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.log (x + 1) + Real.log (1 - x) + a * (x + 1)

def mono_intervals (a : ℝ) : Set ℝ × Set ℝ := 
  if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) 
  else (∅, ∅)

theorem problem_1 (a : ℝ) (h_pos : a > 0) : 
  mono_intervals a = if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) else (∅, ∅) :=
sorry

theorem problem_2 (h_max : f a 0 = 1) (h_pos : a > 0) : 
  a = 1 :=
sorry

end problem_1_problem_2_l215_215827


namespace numberOfSubsets_of_A_l215_215081

def numberOfSubsets (s : Finset ℕ) : ℕ := 2 ^ (Finset.card s)

theorem numberOfSubsets_of_A : 
  numberOfSubsets ({0, 1} : Finset ℕ) = 4 := 
by 
  sorry

end numberOfSubsets_of_A_l215_215081


namespace female_salmon_returned_l215_215414

/-- The number of female salmon that returned to their rivers is 259378,
    given that the total number of salmon that made the trip is 971639 and
    the number of male salmon that returned is 712261. -/
theorem female_salmon_returned :
  let n := 971639
  let m := 712261
  let f := n - m
  f = 259378 :=
by
  rfl

end female_salmon_returned_l215_215414


namespace not_solution_of_equation_l215_215430

theorem not_solution_of_equation (a : ℝ) (h : a ≠ 0) : ¬ (a^2 * 1^2 + (a + 1) * 1 + 1 = 0) :=
by {
  sorry
}

end not_solution_of_equation_l215_215430


namespace roja_speed_l215_215865

theorem roja_speed (R : ℕ) (h1 : 3 + R = 7) : R = 7 - 3 :=
by sorry

end roja_speed_l215_215865


namespace jerry_boxes_l215_215337

theorem jerry_boxes (boxes_sold boxes_left : ℕ) (h₁ : boxes_sold = 5) (h₂ : boxes_left = 5) : (boxes_sold + boxes_left = 10) :=
by
  sorry

end jerry_boxes_l215_215337


namespace intersection_A_B_l215_215843

def A : Set ℝ := { x | |x| > 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l215_215843


namespace lucy_run_base10_eq_1878_l215_215097

-- Define a function to convert a base-8 numeral to base-10.
def base8_to_base10 (n: Nat) : Nat :=
  (3 * 8^3) + (5 * 8^2) + (2 * 8^1) + (6 * 8^0)

-- Define the base-8 number.
def lucy_run (n : Nat) : Nat := n

-- Prove that the base-10 equivalent of the base-8 number 3526 is 1878.
theorem lucy_run_base10_eq_1878 : base8_to_base10 (lucy_run 3526) = 1878 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end lucy_run_base10_eq_1878_l215_215097


namespace leak_time_to_empty_tank_l215_215207

theorem leak_time_to_empty_tank :
  let rateA := 1 / 2  -- rate at which pipe A fills the tank (tanks per hour)
  let rateB := 2 / 3  -- rate at which pipe B fills the tank (tanks per hour)
  let combined_rate_without_leak := rateA + rateB  -- combined rate without leak
  let combined_rate_with_leak := 1 / 1.75  -- combined rate with leak (tanks per hour)
  let leak_rate := combined_rate_without_leak - combined_rate_with_leak  -- rate of the leak (tanks per hour)
  60 / leak_rate = 100.8 :=  -- time to empty the tank by the leak (minutes)
    by sorry

end leak_time_to_empty_tank_l215_215207


namespace basketball_starting_lineups_l215_215907

theorem basketball_starting_lineups (n_players n_guards n_forwards n_centers : ℕ)
  (h_players : n_players = 12)
  (h_guards : n_guards = 2)
  (h_forwards : n_forwards = 2)
  (h_centers : n_centers = 1) :
  (Nat.choose n_players n_guards) * (Nat.choose (n_players - n_guards) n_forwards) * (Nat.choose (n_players - n_guards - n_forwards) n_centers) = 23760 := by
  sorry

end basketball_starting_lineups_l215_215907


namespace quadratic_function_behavior_l215_215138

theorem quadratic_function_behavior (x : ℝ) (h : x > 2) :
  ∃ y : ℝ, y = - (x - 2)^2 - 7 ∧ ∀ x₁ x₂, x₁ > 2 → x₂ > x₁ → (-(x₂ - 2)^2 - 7) < (-(x₁ - 2)^2 - 7) :=
by
  sorry

end quadratic_function_behavior_l215_215138


namespace probability_all_quitters_from_same_tribe_l215_215220

noncomputable def total_ways_to_choose_quitters : ℕ := Nat.choose 18 3

noncomputable def ways_all_from_tribe (n : ℕ) : ℕ := Nat.choose n 3

noncomputable def combined_ways_same_tribe : ℕ :=
  ways_all_from_tribe 9 + ways_all_from_tribe 9

noncomputable def probability_same_tribe (total : ℕ) (same_tribe : ℕ) : ℚ :=
  same_tribe / total

theorem probability_all_quitters_from_same_tribe :
  probability_same_tribe total_ways_to_choose_quitters combined_ways_same_tribe = 7 / 34 :=
by
  sorry

end probability_all_quitters_from_same_tribe_l215_215220


namespace probability_xi_eq_12_correct_l215_215449

noncomputable def probability_xi_eq_12 : ℚ := (nat.choose 11 9) * (3 / 8) ^ 9 * (5 / 8) ^ 2 * (3 / 8)

theorem probability_xi_eq_12_correct :
  let p_red := (3 : ℚ) / 8,
      p_white := (5 : ℚ) / 8 in
  P(ξ = 12) = (nat.choose 11 9) * p_red ^ 9 * p_white ^ 2 * p_red := 
begin
  let p_red := 3 / 8,
  let p_white := 5 / 8,
  have : P(ξ = 12) = (nat.choose 11 9) * p_red ^ 9 * p_white ^ 2 * p_red,
  sorry
end

end probability_xi_eq_12_correct_l215_215449


namespace third_term_of_sequence_l215_215229

theorem third_term_of_sequence :
  (3 - (1 / 3) = 8 / 3) :=
by
  sorry

end third_term_of_sequence_l215_215229


namespace toy_poodle_height_l215_215491

theorem toy_poodle_height 
  (SP MP TP : ℕ)
  (h1 : SP = MP + 8)
  (h2 : MP = TP + 6)
  (h3 : SP = 28) 
  : TP = 14 := 
    by sorry

end toy_poodle_height_l215_215491


namespace sum_first_n_terms_arithmetic_sequence_eq_l215_215856

open Nat

noncomputable def sum_arithmetic_sequence (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if h: n = 0 then 0 else n * a₁ + (n * (n - 1) * d) / 2

theorem sum_first_n_terms_arithmetic_sequence_eq 
  (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) 
  (h₀ : d ≠ 0)
  (h₁ : a₁ = 4)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₆ = a₁ + 5 * d)
  (h₄ : a₃^2 = a₁ * a₆) :
  sum_arithmetic_sequence a₁ a₃ a₆ d n = (n^2 + 7 * n) / 2 := 
by
  sorry

end sum_first_n_terms_arithmetic_sequence_eq_l215_215856


namespace sum_of_remainders_l215_215266

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) :
  (n % 2 + n % 9) = 3 :=
sorry

end sum_of_remainders_l215_215266


namespace range_of_a_circle_C_intersects_circle_D_l215_215834

/-- Definitions of circles C and D --/
def circle_C_eq (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1
def circle_D_eq (x y m : ℝ) := x^2 + y^2 - 2 * m * x = 0

/-- Condition for the line intersecting Circle C --/
def line_intersects_circle_C (a : ℝ) := (∃ x y : ℝ, circle_C_eq x y ∧ (x + y = a))

/-- Proof of range for a --/
theorem range_of_a (a : ℝ) : line_intersects_circle_C a → (2 - Real.sqrt 2 ≤ a ∧ a ≤ 2 + Real.sqrt 2) :=
sorry

/-- Proposition for point A lying on circle C and satisfying the inequality --/
def point_A_on_circle_C_and_inequality (m : ℝ) (x y : ℝ) :=
  circle_C_eq x y ∧ x^2 + y^2 - (m + Real.sqrt 2 / 2) * x - (m + Real.sqrt 2 / 2) * y ≤ 0

/-- Proof that Circle C intersects Circle D --/
theorem circle_C_intersects_circle_D (m : ℝ) (a : ℝ) : 
  (∀ (x y : ℝ), point_A_on_circle_C_and_inequality m x y) →
  (1 ≤ m ∧
   ∃ (x y : ℝ), (circle_D_eq x y m ∧ (Real.sqrt ((m - 1)^2 + 1) < m + 1 ∧ Real.sqrt ((m - 1)^2 + 1) > m - 1))) :=
sorry

end range_of_a_circle_C_intersects_circle_D_l215_215834


namespace melted_mixture_weight_l215_215509

variable (zinc copper total_weight : ℝ)
variable (ratio_zinc ratio_copper : ℝ := 9 / 11)
variable (weight_zinc : ℝ := 31.5)

theorem melted_mixture_weight :
  (zinc / copper = ratio_zinc / ratio_copper) ∧ (zinc = weight_zinc) →
  (total_weight = zinc + copper) →
  total_weight = 70 := 
sorry

end melted_mixture_weight_l215_215509


namespace problem_1_problem_2_l215_215306

noncomputable def f (x : ℝ) (a : ℝ) := Real.sqrt (a - x^2)

-- First proof problem statement: 
theorem problem_1 (a : ℝ) (x : ℝ) (A B : Set ℝ) (h1 : a = 4) (h2 : A = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) (h3 : B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) : 
  (A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) :=
sorry

-- Second proof problem statement:
theorem problem_2 (a : ℝ) (h : 1 ∈ {y : ℝ | 0 ≤ y ∧ y ≤ Real.sqrt a}) : a ≥ 1 :=
sorry

end problem_1_problem_2_l215_215306


namespace solve_bank_account_problem_l215_215283

noncomputable def bank_account_problem : Prop :=
  ∃ (A E Z : ℝ),
    A > E ∧
    Z > A ∧
    A - E = (1/12) * (A + E) ∧
    Z - A = (1/10) * (Z + A) ∧
    1.10 * A = 1.20 * E + 20 ∧
    1.10 * A + 30 = 1.15 * Z ∧
    E = 2000 / 23

theorem solve_bank_account_problem : bank_account_problem :=
sorry

end solve_bank_account_problem_l215_215283


namespace find_digits_sum_l215_215849

theorem find_digits_sum (a b c : Nat) (ha : 0 <= a ∧ a <= 9) (hb : 0 <= b ∧ b <= 9) 
  (hc : 0 <= c ∧ c <= 9) 
  (h1 : 2 * a = c) 
  (h2 : b = b) : 
  a + b + c = 11 :=
  sorry

end find_digits_sum_l215_215849


namespace division_quotient_l215_215861

theorem division_quotient (dividend divisor remainder quotient : ℕ) 
  (h₁ : dividend = 95) (h₂ : divisor = 15) (h₃ : remainder = 5)
  (h₄ : dividend = divisor * quotient + remainder) : quotient = 6 :=
by
  sorry

end division_quotient_l215_215861


namespace matrix_det_example_l215_215788

variable (A : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A = ![![5, -4], ![2, 3]])

theorem matrix_det_example : Matrix.det A = 23 :=
by
  sorry

end matrix_det_example_l215_215788


namespace product_of_solutions_of_t_squared_eq_49_l215_215817

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l215_215817


namespace inequality_solution_l215_215429

theorem inequality_solution (x : ℝ) (h : ∀ (a b : ℝ) (ha : 0 < a) (hb : 0 < b), x^2 + x < a / b + b / a) : x ∈ Set.Ioo (-2 : ℝ) 1 := 
sorry

end inequality_solution_l215_215429


namespace initial_birds_l215_215477

theorem initial_birds (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end initial_birds_l215_215477


namespace remainder_M_mod_32_l215_215039

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l215_215039


namespace solve_for_s_l215_215323

theorem solve_for_s : ∃ (s t : ℚ), (8 * s + 7 * t = 160) ∧ (s = t - 3) ∧ (s = 139 / 15) := by
  sorry

end solve_for_s_l215_215323


namespace inequality_proof_l215_215149

theorem inequality_proof
  (a b c d e f : ℝ)
  (h : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
sorry

end inequality_proof_l215_215149


namespace solve_system_l215_215472

theorem solve_system :
  (∃ x y : ℝ, 4 * x + y = 5 ∧ 2 * x - 3 * y = 13) ↔ (x = 2 ∧ y = -3) :=
by
  sorry

end solve_system_l215_215472


namespace distance_traveled_l215_215396

def velocity (t : ℝ) : ℝ := t^2 + 1

theorem distance_traveled :
  (∫ t in (0:ℝ)..(3:ℝ), velocity t) = 12 :=
by
  simp [velocity]
  sorry

end distance_traveled_l215_215396


namespace range_a_monotonically_increasing_l215_215566

def g (a x : ℝ) : ℝ := a * x^3 + a * x^2 + x

theorem range_a_monotonically_increasing (a : ℝ) : 
  (∀ x : ℝ, 3 * a * x^2 + 2 * a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 3) := 
sorry

end range_a_monotonically_increasing_l215_215566


namespace maximum_value_of_f_l215_215671

def S (n : ℕ) := (n * (n + 1)) / 2
def f (n : ℕ) := S n / ((n + 32) * S (n + 1))

theorem maximum_value_of_f : ∀ (n : ℕ), f(n) ≤ 1/50 :=
begin
  -- to be proved
  sorry
end

end maximum_value_of_f_l215_215671


namespace AdvancedVowelSoup_l215_215780

noncomputable def AdvancedVowelSoup.sequence_count : ℕ :=
  let total_sequences := 7^7
  let vowel_only_sequences := 5^7
  let consonant_only_sequences := 2^7
  total_sequences - vowel_only_sequences - consonant_only_sequences

theorem AdvancedVowelSoup.valid_sequences : AdvancedVowelSoup.sequence_count = 745290 := by
  sorry

end AdvancedVowelSoup_l215_215780


namespace locus_of_point_is_circle_l215_215971

theorem locus_of_point_is_circle (x y : ℝ) 
  (h : 10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3 * x - 4 * y|) : 
  ∃ (c : ℝ) (r : ℝ), ∀ (x y : ℝ), (x - c)^2 + (y - c)^2 = r^2 := 
sorry

end locus_of_point_is_circle_l215_215971


namespace solve_fractional_eq_l215_215226

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) : (4 / (x - 2) = 2 / x) → (x = -2) :=
by 
  sorry

end solve_fractional_eq_l215_215226


namespace birds_on_the_fence_l215_215877

theorem birds_on_the_fence (x : ℕ) : 10 + 2 * x = 50 → x = 20 := by
  sorry

end birds_on_the_fence_l215_215877


namespace algebraic_identity_l215_215278

variables {R : Type*} [CommRing R] (a b : R)

theorem algebraic_identity : 2 * (a - b) + 3 * b = 2 * a + b :=
by
  sorry

end algebraic_identity_l215_215278


namespace sufficient_but_not_necessary_l215_215107

theorem sufficient_but_not_necessary (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ (¬ (p ∧ q) → p ∨ q → False) :=
by
  sorry

end sufficient_but_not_necessary_l215_215107


namespace alpha_sin_beta_lt_beta_sin_alpha_l215_215351

variable {α β : ℝ}

theorem alpha_sin_beta_lt_beta_sin_alpha (h1 : 0 < α) (h2 : α < β) (h3 : β < Real.pi / 2) : 
  α * Real.sin β < β * Real.sin α := 
by
  sorry

end alpha_sin_beta_lt_beta_sin_alpha_l215_215351


namespace cost_of_dowels_l215_215641

variable (V S : ℝ)

theorem cost_of_dowels 
  (hV : V = 7)
  (h_eq : 0.85 * (V + S) = V + 0.5 * S) :
  S = 3 :=
by
  sorry

end cost_of_dowels_l215_215641


namespace expression_value_l215_215564

theorem expression_value (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 2 * y + 4 * z = 21 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end expression_value_l215_215564


namespace xy_value_l215_215173

theorem xy_value (x y : ℝ) (h : (|x| - 1)^2 + (2 * y + 1)^2 = 0) : xy = 1/2 ∨ xy = -1/2 :=
by {
  sorry
}

end xy_value_l215_215173


namespace points_on_ellipse_l215_215136

theorem points_on_ellipse (u : ℝ) :
  let x := (Real.cos u + Real.sin u)
  let y := (4 * (Real.cos u - Real.sin u))
  (x^2 / 2 + y^2 / 32 = 1) :=
by
  let x := (Real.cos u + Real.sin u)
  let y := (4 * (Real.cos u - Real.sin u))
  sorry

end points_on_ellipse_l215_215136


namespace total_process_time_l215_215718
-- Define the conditions
def resisting_time : ℕ := 20
def distance_walked : ℕ := 64
def walking_rate : ℕ := 8

-- Define the question to prove the total process time
theorem total_process_time : 
  let walking_time := distance_walked / walking_rate in
  let total_time := walking_time + resisting_time in
  total_time = 28 := 
by 
  sorry

end total_process_time_l215_215718


namespace angle_AC_B₁C₁_is_60_l215_215456

-- Redefine the conditions of the problem using Lean definitions
-- We define a regular triangular prism, equilateral triangle condition,
-- and parallel lines relation.

structure TriangularPrism :=
  (A B C A₁ B₁ C₁ : Type)
  (is_regular : Prop) -- Property stating it is a regular triangular prism
  (base_is_equilateral : Prop) -- Property stating the base is an equilateral triangle
  (B₁C₁_parallel_to_BC : Prop) -- Property stating B₁C₁ is parallel to BC

-- Assume a regular triangular prism with the given properties
variable (prism : TriangularPrism)
axiom isRegularPrism : prism.is_regular
axiom baseEquilateral : prism.base_is_equilateral
axiom parallelLines : prism.B₁C₁_parallel_to_BC

-- Define the angle calculation statement in Lean 4
theorem angle_AC_B₁C₁_is_60 :
  ∃ (angle : ℝ), angle = 60 :=
by
  -- Proof is omitted using sorry
  exact ⟨60, sorry⟩

end angle_AC_B₁C₁_is_60_l215_215456


namespace find_c_find_cos_2B_minus_pi_over_4_l215_215823

variable (A B C : Real) (a b c : Real)

-- Given conditions
def conditions (a b c : Real) (A : Real) : Prop :=
  a = 4 * Real.sqrt 3 ∧
  b = 6 ∧
  Real.cos A = -1 / 3

-- Proof of question 1
theorem find_c (h : conditions a b c A) : c = 2 :=
sorry

-- Proof of question 2
theorem find_cos_2B_minus_pi_over_4 (h : conditions a b c A) (B : Real) :
  (angle_opp_b : b = Real.sin B) → -- This is to ensure B is the angle opposite to side b
  Real.cos (2 * B - Real.pi / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end find_c_find_cos_2B_minus_pi_over_4_l215_215823


namespace remainder_of_product_of_odd_primes_mod_32_l215_215000

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l215_215000


namespace bananas_to_oranges_cost_l215_215403

noncomputable def cost_equivalence (bananas apples oranges : ℕ) : Prop :=
  (5 * bananas = 3 * apples) ∧
  (8 * apples = 5 * oranges)

theorem bananas_to_oranges_cost (bananas apples oranges : ℕ) 
  (h : cost_equivalence bananas apples oranges) :
  oranges = 9 :=
by sorry

end bananas_to_oranges_cost_l215_215403


namespace geometric_sequence_a5_l215_215195

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r)
  (h_eqn : ∃ x : ℝ, (x^2 + 7*x + 9 = 0) ∧ (a 3 = x) ∧ (a 7 = x)) :
  a 5 = 3 ∨ a 5 = -3 := 
sorry

end geometric_sequence_a5_l215_215195


namespace problem1_problem2_l215_215171

theorem problem1 (α : ℝ) (h : Real.tan α = -2) : 
  (Real.sin α + 5 * Real.cos α) / (-2 * Real.cos α + Real.sin α) = -3 / 4 :=
sorry

theorem problem2 (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (α - 5 * Real.pi) * Real.sin (3 * Real.pi / 2 - α) = -2 / 5 :=
sorry

end problem1_problem2_l215_215171


namespace standard_deviation_less_than_l215_215227

theorem standard_deviation_less_than:
  ∀ (μ σ : ℝ)
  (h1 : μ = 55)
  (h2 : μ - 3 * σ > 48),
  σ < 7 / 3 :=
by
  intros μ σ h1 h2
  sorry

end standard_deviation_less_than_l215_215227


namespace bus_people_count_l215_215373

-- Define the initial number of people on the bus
def initial_people_on_bus : ℕ := 34

-- Define the number of people who got off the bus
def people_got_off : ℕ := 11

-- Define the number of people who got on the bus
def people_got_on : ℕ := 24

-- Define the final number of people on the bus
def final_people_on_bus : ℕ := (initial_people_on_bus - people_got_off) + people_got_on

-- Theorem: The final number of people on the bus is 47.
theorem bus_people_count : final_people_on_bus = 47 := by
  sorry

end bus_people_count_l215_215373


namespace find_angle_CDB_l215_215577

variables (A B C D E : Type)
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C] [LinearOrderedField D] [LinearOrderedField E]

noncomputable def angle := ℝ -- Define type for angles

variables (AB AD AC ACB ACD : angle)
variables (BAD BEA CDB : ℝ)

-- Define the given angles and conditions in Lean
axiom AB_eq_AD : AB = AD
axiom angle_ACD_eq_angle_ACB : AC = ACD
axiom angle_BAD_eq_140 : BAD = 140
axiom angle_BEA_eq_110 : BEA = 110

theorem find_angle_CDB (AB_eq_AD : AB = AD)
                       (angle_ACD_eq_angle_ACB : AC = ACD)
                       (angle_BAD_eq_140 : BAD = 140)
                       (angle_BEA_eq_110 : BEA = 110) :
                       CDB = 50 :=
by
  sorry

end find_angle_CDB_l215_215577


namespace least_number_to_add_for_divisibility_by_nine_l215_215625

theorem least_number_to_add_for_divisibility_by_nine : ∃ x : ℕ, (4499 + x) % 9 = 0 ∧ x = 1 :=
by
  sorry

end least_number_to_add_for_divisibility_by_nine_l215_215625


namespace find_real_x_l215_215540

theorem find_real_x (x : ℝ) : 
  (2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 6) ↔ (7 / 3 < x ∧ x ≤ 14 / 5) :=
by sorry

end find_real_x_l215_215540


namespace range_of_b_l215_215327

-- Given a function f(x)
def f (b x : ℝ) : ℝ := x^3 - 3 * b * x + 3 * b

-- Derivative of the function f(x)
def f' (b x : ℝ) : ℝ := 3 * x^2 - 3 * b

-- The theorem to prove the range of b
theorem range_of_b (b : ℝ) : (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f' b x = 0) → (0 < b ∧ b < 1) := by
  sorry

end range_of_b_l215_215327


namespace last_episode_broadcast_date_l215_215772

theorem last_episode_broadcast_date :
  ∃ d, d = date.mk 2016 5 29 ∧ d.toMondayBasedWeekday = weekday.sunday :=
sorry

end last_episode_broadcast_date_l215_215772


namespace inequality_proof_l215_215147

theorem inequality_proof {a b c d e f : ℝ} (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
sorry

end inequality_proof_l215_215147


namespace range_of_a_l215_215169

-- Definition of the universal set U
def U : set ℝ := {x | 0 ≤ x}

-- Definition of the set A
def A : set ℝ := {x | x^2 - 2 * x - 3 ≥ 0}

-- Definition of the set B
def B (a : ℝ) : set ℝ := {x | x^2 + a < 0}

-- Lean statement for the theorem
theorem range_of_a :
  ∀ a : ℝ, (U \ A) ∪ B a = U \ A ↔ a ∈ (-9, +∞) :=
by
  sorry

end range_of_a_l215_215169


namespace max_area_triangle_ABO1_l215_215168

-- Definitions of the problem conditions
def l1 := {p : ℝ × ℝ | 2 * p.1 + 5 * p.2 = 1}

def C := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + 4 * p.2 = 4}

def parallel (l1 l2 : ℝ × ℝ → Prop) := 
  ∃ m c1 c2, (∀ p, l1 p ↔ (p.2 = m * p.1 + c1)) ∧ (∀ p, l2 p ↔ (p.2 = m * p.1 + c2))

def intersects (l : ℝ × ℝ → Prop) (C: ℝ × ℝ → Prop) : Prop :=
  ∃ A B, (l A ∧ C A ∧ l B ∧ C B ∧ A ≠ B)

noncomputable def area (A B O : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((A.1 * (B.2 - O.2)) + (B.1 * (O.2 - A.2)) + (O.1 * (A.2 - B.2)))

-- Main statement to prove
theorem max_area_triangle_ABO1 :
  ∀ l2, parallel l1 l2 →
  intersects l2 C →
  ∃ A B, area A B (1, -2) ≤ 9 / 2 := 
sorry

end max_area_triangle_ABO1_l215_215168


namespace trees_to_plant_total_l215_215399

def trees_chopped_first_half := 200
def trees_chopped_second_half := 300
def trees_to_plant_per_tree_chopped := 3

theorem trees_to_plant_total : 
  (trees_chopped_first_half + trees_chopped_second_half) * trees_to_plant_per_tree_chopped = 1500 :=
by
  sorry

end trees_to_plant_total_l215_215399


namespace number_division_l215_215240

theorem number_division (x : ℚ) (h : x / 2 = 100 + x / 5) : x = 1000 / 3 := 
by
  sorry

end number_division_l215_215240


namespace inequality_proof_l215_215158

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
by 
  sorry

end inequality_proof_l215_215158
