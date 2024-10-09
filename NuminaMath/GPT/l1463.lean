import Mathlib

namespace max_value_fraction_squares_l1463_146361

-- Let x and y be positive real numbers
variable (x y : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)

theorem max_value_fraction_squares (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ k, (x + 2 * y)^2 / (x^2 + y^2) ≤ k) ∧ (∀ z, (x + 2 * y)^2 / (x^2 + y^2) ≤ z) → k = 9 / 2 :=
by
  sorry

end max_value_fraction_squares_l1463_146361


namespace previous_painting_price_l1463_146343

-- Define the amount received for the most recent painting
def recentPainting (p : ℕ) := 5 * p - 1000

-- Define the target amount
def target := 44000

-- State that the target amount is achieved by the prescribed function
theorem previous_painting_price : recentPainting 9000 = target :=
by
  sorry

end previous_painting_price_l1463_146343


namespace avg_score_false_iff_unequal_ints_l1463_146330

variable {a b m n : ℕ}

theorem avg_score_false_iff_unequal_ints 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (m_neq_n : m ≠ n) : 
  (∃ a b, (ma + nb) / (m + n) = (a + b)/2) ↔ a ≠ b := 
sorry

end avg_score_false_iff_unequal_ints_l1463_146330


namespace tan_theta_value_l1463_146373

theorem tan_theta_value (θ : ℝ) (h1 : Real.sin θ = 3/5) (h2 : Real.cos θ = -4/5) : 
  Real.tan θ = -3/4 :=
  sorry

end tan_theta_value_l1463_146373


namespace simple_interest_years_l1463_146379

theorem simple_interest_years (SI P : ℝ) (R : ℝ) (T : ℝ) 
  (hSI : SI = 200) 
  (hP : P = 1600) 
  (hR : R = 3.125) : 
  T = 4 :=
by 
  sorry

end simple_interest_years_l1463_146379


namespace cube_surface_area_unchanged_l1463_146377

def cubeSurfaceAreaAfterCornersRemoved
  (original_side : ℕ)
  (corner_side : ℕ)
  (original_surface_area : ℕ)
  (number_of_corners : ℕ)
  (surface_reduction_per_corner : ℕ)
  (new_surface_addition_per_corner : ℕ) : Prop :=
  (original_side * original_side * 6 = original_surface_area) →
  (corner_side * corner_side * 3 = surface_reduction_per_corner) →
  (corner_side * corner_side * 3 = new_surface_addition_per_corner) →
  original_surface_area - (number_of_corners * surface_reduction_per_corner) + (number_of_corners * new_surface_addition_per_corner) = original_surface_area
  
theorem cube_surface_area_unchanged :
  cubeSurfaceAreaAfterCornersRemoved 4 1 96 8 3 3 :=
by
  intro h1 h2 h3
  sorry

end cube_surface_area_unchanged_l1463_146377


namespace integer_pairs_satisfying_equation_l1463_146328

theorem integer_pairs_satisfying_equation:
  ∀ (a b : ℕ), a ≥ 1 → b ≥ 1 → a^(b^2) = b^a ↔ (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end integer_pairs_satisfying_equation_l1463_146328


namespace talent_show_girls_count_l1463_146352

theorem talent_show_girls_count (B G : ℕ) (h1 : B + G = 34) (h2 : G = B + 22) : G = 28 :=
by
  sorry

end talent_show_girls_count_l1463_146352


namespace find_base_l1463_146364

theorem find_base 
  (k : ℕ) 
  (h : 1 * k^2 + 3 * k^1 + 2 * k^0 = 30) : 
  k = 4 :=
  sorry

end find_base_l1463_146364


namespace factor_expression_l1463_146317

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by 
  sorry

end factor_expression_l1463_146317


namespace smallest_integer_value_l1463_146319

theorem smallest_integer_value (y : ℤ) (h : 7 - 3 * y < -8) : y ≥ 6 :=
sorry

end smallest_integer_value_l1463_146319


namespace polar_not_one_to_one_correspondence_l1463_146356

theorem polar_not_one_to_one_correspondence :
  ¬ ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p1 p2 : ℝ × ℝ, f p1 = f p2 → p1 = p2) ∧
  (∀ q : ℝ × ℝ, ∃ p : ℝ × ℝ, q = f p) :=
by
  sorry

end polar_not_one_to_one_correspondence_l1463_146356


namespace minimum_students_exceeds_1000_l1463_146355

theorem minimum_students_exceeds_1000 (n : ℕ) :
  (∃ k : ℕ, k > 1000 ∧ k % 10 = 0 ∧ k % 14 = 0 ∧ k % 18 = 0 ∧ n = k) ↔ n = 1260 :=
sorry

end minimum_students_exceeds_1000_l1463_146355


namespace sum_arithmetic_sequence_min_value_l1463_146310

theorem sum_arithmetic_sequence_min_value (a d : ℤ) 
  (S : ℕ → ℤ) 
  (H1 : S 8 ≤ 6) 
  (H2 : S 11 ≥ 27)
  (H_Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) : 
  S 19 ≥ 133 :=
by
  sorry

end sum_arithmetic_sequence_min_value_l1463_146310


namespace white_animals_count_l1463_146309

-- Definitions
def total : ℕ := 13
def black : ℕ := 6
def white : ℕ := total - black

-- Theorem stating the number of white animals
theorem white_animals_count : white = 7 :=
by {
  -- The proof would go here, but we'll use sorry to skip it.
  sorry
}

end white_animals_count_l1463_146309


namespace tan_double_angle_l1463_146372

open Real

theorem tan_double_angle {θ : ℝ} (h1 : tan (π / 2 - θ) = 4 * cos (2 * π - θ)) (h2 : abs θ < π / 2) : 
  tan (2 * θ) = sqrt 15 / 7 :=
sorry

end tan_double_angle_l1463_146372


namespace triangle_area_relation_l1463_146374

theorem triangle_area_relation :
  let A := (1 / 2) * 5 * 5
  let B := (1 / 2) * 12 * 12
  let C := (1 / 2) * 13 * 13
  A + B = C :=
by
  sorry

end triangle_area_relation_l1463_146374


namespace sum_of_prime_factors_210630_l1463_146302

theorem sum_of_prime_factors_210630 : (2 + 3 + 5 + 7 + 17 + 59) = 93 := by
  -- Proof to be provided
  sorry

end sum_of_prime_factors_210630_l1463_146302


namespace division_remainder_l1463_146387

def p (x : ℝ) := x^5 + 2 * x^3 - x + 4
def a : ℝ := 2
def remainder : ℝ := 50

theorem division_remainder :
  p a = remainder :=
sorry

end division_remainder_l1463_146387


namespace trajectory_of_N_l1463_146362

variables {x y x₀ y₀ : ℝ}

def F : ℝ × ℝ := (1, 0)

def M (x₀ : ℝ) : ℝ × ℝ := (x₀, 0)
def P (y₀ : ℝ) : ℝ × ℝ := (0, y₀)
def N (x y : ℝ) : ℝ × ℝ := (x, y)

def PM (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, -y₀)
def PF (y₀ : ℝ) : ℝ × ℝ := (1, -y₀)

def perpendicular (v1 v2 : ℝ × ℝ) := v1.fst * v2.fst + v1.snd * v2.snd = 0

def MN_eq_2MP (x y x₀ y₀ : ℝ) := ((x - x₀), y) = (2 * (-x₀), 2 * y₀)

theorem trajectory_of_N (h1 : perpendicular (PM x₀ y₀) (PF y₀))
  (h2 : MN_eq_2MP x y x₀ y₀) :
  y^2 = 4*x :=
by
  sorry

end trajectory_of_N_l1463_146362


namespace combined_market_value_two_years_later_l1463_146306

theorem combined_market_value_two_years_later:
  let P_A := 8000
  let P_B := 10000
  let P_C := 12000
  let r_A := 0.20
  let r_B := 0.15
  let r_C := 0.10

  let V_A_year_1 := P_A - r_A * P_A
  let V_A_year_2 := V_A_year_1 - r_A * P_A
  let V_B_year_1 := P_B - r_B * P_B
  let V_B_year_2 := V_B_year_1 - r_B * P_B
  let V_C_year_1 := P_C - r_C * P_C
  let V_C_year_2 := V_C_year_1 - r_C * P_C

  V_A_year_2 + V_B_year_2 + V_C_year_2 = 21400 :=
by
  sorry

end combined_market_value_two_years_later_l1463_146306


namespace find_sequence_index_l1463_146301

theorem find_sequence_index (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) - 3 = a n)
  (h₃ : ∃ n, a n = 2023) : ∃ n, a n = 2023 ∧ n = 675 := 
by 
  sorry

end find_sequence_index_l1463_146301


namespace volume_percentage_correct_l1463_146344

-- Define the initial conditions
def box_length := 8
def box_width := 6
def box_height := 12
def cube_side := 3

-- Calculate the number of cubes along each dimension
def num_cubes_length := box_length / cube_side
def num_cubes_width := box_width / cube_side
def num_cubes_height := box_height / cube_side

-- Calculate volumes
def volume_cube := cube_side ^ 3
def volume_box := box_length * box_width * box_height
def volume_cubes := (num_cubes_length * num_cubes_width * num_cubes_height) * volume_cube

-- Prove the percentage calculation
theorem volume_percentage_correct : (volume_cubes.toFloat / volume_box.toFloat) * 100 = 75 := by
  sorry

end volume_percentage_correct_l1463_146344


namespace find_a_l1463_146384

def system_of_equations (a x y : ℝ) : Prop :=
  y - 2 = a * (x - 4) ∧ (2 * x) / (|y| + y) = Real.sqrt x

def domain_constraints (x y : ℝ) : Prop :=
  y > 0 ∧ x ≥ 0

def valid_a (a : ℝ) : Prop :=
  (∃ x y, domain_constraints x y ∧ system_of_equations a x y)

theorem find_a :
  ∀ a : ℝ, valid_a a ↔
  ((a < 0.5 ∧ ∃ y, y = 2 - 4 * a ∧ y > 0) ∨ 
   (∃ x y, x = 4 ∧ y = 2 ∧ x ≥ 0 ∧ y > 0) ∨
   (0 < a ∧ a ≠ 0.25 ∧ a < 0.5 ∧ ∃ x y, x = (1 - 2 * a) / a ∧ y = (1 - 2 * a) / a)) :=
by sorry

end find_a_l1463_146384


namespace M_necessary_for_N_l1463_146378

open Set

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem M_necessary_for_N : ∀ a : ℝ, a ∈ N → a ∈ M ∧ ¬(a ∈ M → a ∈ N) :=
by
  sorry

end M_necessary_for_N_l1463_146378


namespace ratio_Polly_Willy_l1463_146350

theorem ratio_Polly_Willy (P S W : ℝ) (h1 : P / S = 4 / 5) (h2 : S / W = 5 / 2) :
  P / W = 2 :=
by sorry

end ratio_Polly_Willy_l1463_146350


namespace intersection_of_M_and_N_l1463_146342

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

-- The proof statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := 
  sorry

end intersection_of_M_and_N_l1463_146342


namespace train_speed_in_kph_l1463_146339

noncomputable def speed_of_train (jogger_speed_kph : ℝ) (gap_m : ℝ) (train_length_m : ℝ) (time_s : ℝ) : ℝ :=
let jogger_speed_mps := jogger_speed_kph * (1000 / 3600)
let total_distance_m := gap_m + train_length_m
let speed_mps := total_distance_m / time_s
speed_mps * (3600 / 1000)

theorem train_speed_in_kph :
  speed_of_train 9 240 120 36 = 36 := 
by
  sorry

end train_speed_in_kph_l1463_146339


namespace inequality_am_gm_cauchy_schwarz_equality_iff_l1463_146303

theorem inequality_am_gm_cauchy_schwarz 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

theorem equality_iff (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) 
  ↔ a = b ∧ b = c ∧ c = d :=
sorry

end inequality_am_gm_cauchy_schwarz_equality_iff_l1463_146303


namespace convert_to_cylindrical_l1463_146367

theorem convert_to_cylindrical (x y z : ℝ) (r θ : ℝ) 
  (h₀ : x = 3) 
  (h₁ : y = -3 * Real.sqrt 3) 
  (h₂ : z = 2) 
  (h₃ : r > 0) 
  (h₄ : 0 ≤ θ) 
  (h₅ : θ < 2 * Real.pi) 
  (h₆ : r = Real.sqrt (x^2 + y^2)) 
  (h₇ : x = r * Real.cos θ) 
  (h₈ : y = r * Real.sin θ) : 
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) :=
by
  -- Proof goes here
  sorry

end convert_to_cylindrical_l1463_146367


namespace cost_to_paint_floor_l1463_146316

-- Define the conditions
def length_more_than_breadth_by_200_percent (L B : ℝ) : Prop :=
L = 3 * B

def length_of_floor := 23
def cost_per_sq_meter := 3

-- Prove the cost to paint the floor
theorem cost_to_paint_floor (B : ℝ) (L : ℝ) 
    (h1: length_more_than_breadth_by_200_percent L B) (h2: L = length_of_floor) 
    (rate: ℝ) (h3: rate = cost_per_sq_meter) :
    rate * (L * B) = 529.23 :=
by
  -- intermediate steps would go here
  sorry

end cost_to_paint_floor_l1463_146316


namespace emily_cell_phone_cost_l1463_146390

noncomputable def base_cost : ℝ := 25
noncomputable def included_hours : ℝ := 25
noncomputable def cost_per_text : ℝ := 0.1
noncomputable def cost_per_extra_minute : ℝ := 0.15
noncomputable def cost_per_gigabyte : ℝ := 2

noncomputable def emily_texts : ℝ := 150
noncomputable def emily_hours : ℝ := 26
noncomputable def emily_data : ℝ := 3

theorem emily_cell_phone_cost : 
  let texts_cost := emily_texts * cost_per_text
  let extra_minutes_cost := (emily_hours - included_hours) * 60 * cost_per_extra_minute
  let data_cost := emily_data * cost_per_gigabyte
  base_cost + texts_cost + extra_minutes_cost + data_cost = 55 := by
  sorry

end emily_cell_phone_cost_l1463_146390


namespace example_number_is_not_octal_l1463_146380

-- Define a predicate that checks if a digit is valid in the octal system
def is_octal_digit (d : ℕ) : Prop :=
  d < 8

-- Define a predicate that checks if all digits in a number represented as list of ℕ are valid octal digits
def is_octal_number (n : List ℕ) : Prop :=
  ∀ d ∈ n, is_octal_digit d

-- Example number represented as a list of its digits
def example_number : List ℕ := [2, 8, 5, 3]

-- The statement we aim to prove
theorem example_number_is_not_octal : ¬ is_octal_number example_number := by
  -- Proof goes here
  sorry

end example_number_is_not_octal_l1463_146380


namespace discount_percentage_is_correct_l1463_146320

noncomputable def cost_prices := [540, 660, 780]
noncomputable def markup_percentages := [0.15, 0.20, 0.25]
noncomputable def selling_prices := [496.80, 600, 750]

noncomputable def marked_price (cost : ℝ) (markup : ℝ) : ℝ := cost + (markup * cost)

noncomputable def total_marked_price : ℝ := 
  (marked_price 540 0.15) + (marked_price 660 0.20) + (marked_price 780 0.25)

noncomputable def total_selling_price : ℝ := 496.80 + 600 + 750

noncomputable def overall_discount_percentage : ℝ :=
  ((total_marked_price - total_selling_price) / total_marked_price) * 100

theorem discount_percentage_is_correct : overall_discount_percentage = 22.65 :=
by
  sorry

end discount_percentage_is_correct_l1463_146320


namespace find_a_b_l1463_146397

noncomputable def curve (x a b : ℝ) : ℝ := x^2 + a * x + b

noncomputable def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b (a b : ℝ) :
  (∃ (y : ℝ) (x : ℝ), (y = curve x a b) ∧ tangent_line 0 b ∧ (2 * 0 + a = -1) ∧ (0 - b + 1 = 0)) ->
  a = -1 ∧ b = 1 := 
by
  sorry

end find_a_b_l1463_146397


namespace ana_wins_probability_l1463_146315

noncomputable def probability_ana_wins : ℚ :=
  (1 / 2) ^ 5 / (1 - (1 / 2) ^ 5)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 31 :=
by
  sorry

end ana_wins_probability_l1463_146315


namespace quotient_when_divided_by_44_is_3_l1463_146300

/-
A number, when divided by 44, gives a certain quotient and 0 as remainder.
When dividing the same number by 30, the remainder is 18.
Prove that the quotient in the first division is 3.
-/

theorem quotient_when_divided_by_44_is_3 (N : ℕ) (Q : ℕ) (P : ℕ) 
  (h1 : N % 44 = 0)
  (h2 : N % 30 = 18) :
  N = 44 * Q →
  Q = 3 := 
by
  -- since no proof is required, we use sorry
  sorry

end quotient_when_divided_by_44_is_3_l1463_146300


namespace cos_alpha_value_l1463_146348

theorem cos_alpha_value (α : ℝ) (h₀ : 0 < α ∧ α < 90) (h₁ : Real.sin (α - 45) = - (Real.sqrt 2 / 10)) : 
  Real.cos α = 4 / 5 := 
sorry

end cos_alpha_value_l1463_146348


namespace increase_factor_is_46_8_l1463_146341

-- Definitions for the conditions
def old_plates : ℕ := 26^3 * 10^3
def new_plates_type_A : ℕ := 26^2 * 10^4
def new_plates_type_B : ℕ := 26^4 * 10^2
def average_new_plates := (new_plates_type_A + new_plates_type_B) / 2

-- The Lean 4 statement to prove that the increase factor is 46.8
theorem increase_factor_is_46_8 :
  (average_new_plates : ℚ) / (old_plates : ℚ) = 46.8 := by
  sorry

end increase_factor_is_46_8_l1463_146341


namespace diamond_not_commutative_diamond_not_associative_l1463_146359

noncomputable def diamond (x y : ℝ) : ℝ :=
  x^2 * y / (x + y + 1)

theorem diamond_not_commutative (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x ≠ y → diamond x y ≠ diamond y x :=
by
  intro hxy
  unfold diamond
  intro h
  -- Assume the contradiction leads to this equality not holding
  have eq : x^2 * y * (y + x + 1) = y^2 * x * (x + y + 1) := by
    sorry
  -- Simplify the equation to show the contradiction
  sorry

theorem diamond_not_associative (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (diamond x y) ≠ (diamond y x) → (diamond (diamond x y) z) ≠ (diamond x (diamond y z)) :=
by
  unfold diamond
  intro h
  -- Assume the contradiction leads to this equality not holding
  have eq : (diamond x y)^2 * z / (diamond x y + z + 1) ≠ (x^2 * (diamond y z) / (x + diamond y z + 1)) :=
    by sorry
  -- Simplify the equation to show the contradiction
  sorry

end diamond_not_commutative_diamond_not_associative_l1463_146359


namespace cubic_difference_l1463_146349

theorem cubic_difference (x y : ℤ) (h1 : x + y = 14) (h2 : 3 * x + y = 20) : x^3 - y^3 = -1304 :=
sorry

end cubic_difference_l1463_146349


namespace union_of_A_and_B_l1463_146375

-- Definition of the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := ∅

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = {1, 2} := 
by sorry

end union_of_A_and_B_l1463_146375


namespace min_value_PA_minus_PF_l1463_146396

noncomputable def ellipse_condition : Prop :=
  ∃ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)

noncomputable def focal_property (x y : ℝ) (P : ℝ × ℝ) : Prop :=
  dist P (2, 4) - dist P (1, 0) = 1

theorem min_value_PA_minus_PF :
  ∀ (P : ℝ × ℝ), 
    (∃ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) 
    → ∃ (a b : ℝ), a = 2 ∧ b = 4 ∧ focal_property x y P :=
  sorry

end min_value_PA_minus_PF_l1463_146396


namespace find_f_107_l1463_146336

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = -f x

def piecewise_function (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x / 5

-- Main theorem to prove based on the conditions
theorem find_f_107 (f : ℝ → ℝ)
  (h_periodic : periodic_function f)
  (h_piece : piecewise_function f)
  (h_even : even_function f) : f 107 = 1 / 5 :=
sorry

end find_f_107_l1463_146336


namespace project_assignment_l1463_146327

open Nat

def binom (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem project_assignment :
  let A := 3
  let B := 1
  let C := 2
  let D := 2
  let total_projects := 8
  A + B + C + D = total_projects →
  (binom 8 3) * (binom 5 1) * (binom 4 2) * (binom 2 2) = 1680 :=
by
  intros
  sorry

end project_assignment_l1463_146327


namespace remainder_sum_mod_11_l1463_146346

theorem remainder_sum_mod_11 :
  (72501 + 72502 + 72503 + 72504 + 72505 + 72506 + 72507 + 72508 + 72509 + 72510) % 11 = 5 :=
by
  sorry

end remainder_sum_mod_11_l1463_146346


namespace rowing_distance_l1463_146370

theorem rowing_distance
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (total_time : ℝ)
  (D : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : current_speed = 2)
  (h3 : total_time = 15)
  (h4 : D / (rowing_speed + current_speed) + D / (rowing_speed - current_speed) = total_time) :
  D = 72 := 
sorry

end rowing_distance_l1463_146370


namespace exists_nat_not_in_geom_progressions_l1463_146312

theorem exists_nat_not_in_geom_progressions
  (progressions : Fin 5 → ℕ → ℕ)
  (is_geometric : ∀ i : Fin 5, ∃ a q : ℕ, ∀ n : ℕ, progressions i n = a * q^n) :
  ∃ n : ℕ, ∀ i : Fin 5, ∀ m : ℕ, progressions i m ≠ n :=
by
  sorry

end exists_nat_not_in_geom_progressions_l1463_146312


namespace count_two_digit_prime_with_digit_sum_10_l1463_146391

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l1463_146391


namespace min_reciprocal_sum_l1463_146318

theorem min_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy_sum : x + y = 12) (hxy_neq : x ≠ y) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / x + 1 / y ≥ c) :=
sorry

end min_reciprocal_sum_l1463_146318


namespace fraction_received_l1463_146351

theorem fraction_received (total_money : ℝ) (spent_ratio : ℝ) (spent_amount : ℝ) (remaining_amount : ℝ) (fraction_received : ℝ) :
  total_money = 240 ∧ spent_ratio = 1/5 ∧ spent_amount = spent_ratio * total_money ∧ remaining_amount = 132 ∧ spent_amount + remaining_amount = fraction_received * total_money →
  fraction_received = 3 / 4 :=
by {
  sorry
}

end fraction_received_l1463_146351


namespace xiaoming_grade_is_89_l1463_146358

noncomputable def xiaoming_physical_education_grade
  (extra_activity_score : ℕ) (midterm_score : ℕ) (final_exam_score : ℕ)
  (ratio_extra : ℕ) (ratio_mid : ℕ) (ratio_final : ℕ) : ℝ :=
  (extra_activity_score * ratio_extra + midterm_score * ratio_mid + final_exam_score * ratio_final) / (ratio_extra + ratio_mid + ratio_final)

theorem xiaoming_grade_is_89 :
  xiaoming_physical_education_grade 95 90 85 2 4 4 = 89 := by
    sorry

end xiaoming_grade_is_89_l1463_146358


namespace tangent_line_curve_l1463_146395

theorem tangent_line_curve (x₀ : ℝ) (a : ℝ) :
  (ax₀ + 2 = e^x₀ + 1) ∧ (a = e^x₀) → a = 1 := by
  sorry

end tangent_line_curve_l1463_146395


namespace find_a_plus_b_l1463_146360

theorem find_a_plus_b (a b : ℚ)
  (h1 : 3 = a + b / (2^2 + 1))
  (h2 : 2 = a + b / (1^2 + 1)) :
  a + b = 1 / 3 := 
sorry

end find_a_plus_b_l1463_146360


namespace value_of_expression_l1463_146326

def expression (x y z : ℤ) : ℤ :=
  x^2 + y^2 - z^2 + 2 * x * y + x * y * z

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : z = 1) : 
  expression x y z = -7 := by
  sorry

end value_of_expression_l1463_146326


namespace odd_function_equiv_l1463_146383

noncomputable def odd_function (f : ℝ → ℝ) :=
∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_equiv (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f (x)) ↔ (∀ x : ℝ, f (-(-x)) = -f (-x)) :=
by
  sorry

end odd_function_equiv_l1463_146383


namespace roller_coaster_cars_l1463_146365

theorem roller_coaster_cars
  (people : ℕ)
  (runs : ℕ)
  (seats_per_car : ℕ)
  (people_per_run : ℕ)
  (h1 : people = 84)
  (h2 : runs = 6)
  (h3 : seats_per_car = 2)
  (h4 : people_per_run = people / runs) :
  (people_per_run / seats_per_car) = 7 :=
by
  sorry

end roller_coaster_cars_l1463_146365


namespace tan_315_eq_neg1_l1463_146333

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l1463_146333


namespace megan_eggs_per_meal_l1463_146321

-- Define the initial conditions
def initial_eggs_from_store : Nat := 12
def initial_eggs_from_neighbor : Nat := 12
def eggs_used_for_omelet : Nat := 2
def eggs_used_for_cake : Nat := 4
def meals_to_divide : Nat := 3

-- Calculate various steps
def total_initial_eggs : Nat := initial_eggs_from_store + initial_eggs_from_neighbor
def eggs_after_cooking : Nat := total_initial_eggs - eggs_used_for_omelet - eggs_used_for_cake
def eggs_after_giving_away : Nat := eggs_after_cooking / 2
def eggs_per_meal : Nat := eggs_after_giving_away / meals_to_divide

-- State the theorem to prove the value of eggs_per_meal
theorem megan_eggs_per_meal : eggs_per_meal = 3 := by
  sorry

end megan_eggs_per_meal_l1463_146321


namespace necessarily_positive_expression_l1463_146337

theorem necessarily_positive_expression
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  0 < b + 3 * b^2 := 
sorry

end necessarily_positive_expression_l1463_146337


namespace surface_area_of_solid_l1463_146385

-- Definitions about the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_rectangular_solid (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c ∧ (a * b * c = 399)

-- Main statement of the problem
theorem surface_area_of_solid (a b c : ℕ) (h : is_rectangular_solid a b c) : 
  2 * (a * b + b * c + c * a) = 422 := sorry

end surface_area_of_solid_l1463_146385


namespace smallest_positive_shift_l1463_146334

noncomputable def g : ℝ → ℝ := sorry

theorem smallest_positive_shift
  (H1 : ∀ x, g (x - 20) = g x) : 
  ∃ a > 0, (∀ x, g ((x - a) / 10) = g (x / 10)) ∧ a = 200 :=
sorry

end smallest_positive_shift_l1463_146334


namespace area_of_tangents_l1463_146305

def radius := 3
def segment_length := 6

theorem area_of_tangents (r : ℝ) (l : ℝ) (h1 : r = radius) (h2 : l = segment_length) :
  let R := r * Real.sqrt 2 
  let annulus_area := π * (R ^ 2) - π * (r ^ 2)
  annulus_area = 9 * π :=
by
  sorry

end area_of_tangents_l1463_146305


namespace extremum_point_is_three_l1463_146357

noncomputable def f (x : ℝ) : ℝ := (x - 2) / Real.exp x

theorem extremum_point_is_three {x₀ : ℝ} (h : ∀ x, f x₀ ≤ f x) : x₀ = 3 :=
by
  -- proof goes here
  sorry

end extremum_point_is_three_l1463_146357


namespace sum_of_cubes_of_consecutive_integers_l1463_146393

theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : (n-1)^2 + n^2 + (n+1)^2 = 8450) : 
  (n-1)^3 + n^3 + (n+1)^3 = 446949 := 
sorry

end sum_of_cubes_of_consecutive_integers_l1463_146393


namespace compare_powers_l1463_146388

theorem compare_powers : 2^24 < 10^8 ∧ 10^8 < 5^12 :=
by 
  -- proofs omitted
  sorry

end compare_powers_l1463_146388


namespace average_weight_children_l1463_146382

theorem average_weight_children 
  (n_boys : ℕ)
  (w_boys : ℕ)
  (avg_w_boys : ℕ)
  (n_girls : ℕ)
  (w_girls : ℕ)
  (avg_w_girls : ℕ)
  (h1 : n_boys = 8)
  (h2 : avg_w_boys = 140)
  (h3 : n_girls = 6)
  (h4 : avg_w_girls = 130)
  (h5 : w_boys = n_boys * avg_w_boys)
  (h6 : w_girls = n_girls * avg_w_girls)
  (total_w : ℕ)
  (h7 : total_w = w_boys + w_girls)
  (avg_w : ℚ)
  (h8 : avg_w = total_w / (n_boys + n_girls)) :
  avg_w = 135 :=
by
  sorry

end average_weight_children_l1463_146382


namespace line_segment_length_l1463_146353

theorem line_segment_length (x : ℝ) (h : x > 0) :
  (Real.sqrt ((x - 2)^2 + (6 - 2)^2) = 5) → (x = 5) :=
by
  intro h1
  sorry

end line_segment_length_l1463_146353


namespace contrapositive_equivalence_l1463_146324

theorem contrapositive_equivalence (P Q : Prop) : (P → Q) ↔ (¬ Q → ¬ P) :=
by sorry

end contrapositive_equivalence_l1463_146324


namespace num_zeros_in_decimal_representation_l1463_146389

theorem num_zeros_in_decimal_representation :
  let denom := 2^3 * 5^10
  let frac := (1 : ℚ) / denom
  ∃ n : ℕ, n = 7 ∧ (∃ (a : ℕ) (b : ℕ), frac = a / 10^b ∧ ∃ (k : ℕ), b = n + k + 3) :=
sorry

end num_zeros_in_decimal_representation_l1463_146389


namespace field_trip_students_l1463_146335

theorem field_trip_students 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (total_students : ℕ) 
  (h1 : seats_per_bus = 2) 
  (h2 : buses_needed = 7) 
  (h3 : total_students = seats_per_bus * buses_needed) : 
  total_students = 14 :=
by 
  rw [h1, h2] at h3
  assumption

end field_trip_students_l1463_146335


namespace sum_of_power_of_2_plus_1_divisible_by_3_iff_odd_l1463_146368

theorem sum_of_power_of_2_plus_1_divisible_by_3_iff_odd (n : ℕ) : 
  (3 ∣ (2^n + 1)) ↔ (n % 2 = 1) :=
sorry

end sum_of_power_of_2_plus_1_divisible_by_3_iff_odd_l1463_146368


namespace gcd_lcm_sum_eq_l1463_146369

-- Define the two numbers
def a : ℕ := 72
def b : ℕ := 8712

-- Define the GCD and LCM functions.
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Define the sum of the GCD and LCM.
def sum_gcd_lcm : ℕ := gcd_ab + lcm_ab

-- The theorem we want to prove
theorem gcd_lcm_sum_eq : sum_gcd_lcm = 26160 := by
  -- Details of the proof would go here
  sorry

end gcd_lcm_sum_eq_l1463_146369


namespace inequality_transformation_l1463_146392

variable {a b c d : ℝ}

theorem inequality_transformation
  (h1 : a < b)
  (h2 : b < 0)
  (h3 : c < d)
  (h4 : d < 0) :
  (d / a) < (c / a) :=
by
  sorry

end inequality_transformation_l1463_146392


namespace greatest_integer_less_than_PS_l1463_146323

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ), PQ = 150 → 
  PS = 150 * Real.sqrt 3 → 
  (⌊PS⌋ = 259) := 
by
  intros PQ PS hPQ hPS
  sorry

end greatest_integer_less_than_PS_l1463_146323


namespace sum_of_decimals_l1463_146329

theorem sum_of_decimals :
  (2 / 100 : ℝ) + (5 / 1000) + (8 / 10000) + (6 / 100000) = 0.02586 :=
by
  sorry

end sum_of_decimals_l1463_146329


namespace triangle_minimum_area_l1463_146366

theorem triangle_minimum_area :
  ∃ p q : ℤ, p ≠ 0 ∧ q ≠ 0 ∧ (1 / 2) * |30 * q - 18 * p| = 3 :=
sorry

end triangle_minimum_area_l1463_146366


namespace ruth_weekly_class_hours_l1463_146331

def hours_in_a_day : ℕ := 8
def days_in_a_week : ℕ := 5
def weekly_school_hours := hours_in_a_day * days_in_a_week

def math_class_percentage : ℚ := 0.25
def language_class_percentage : ℚ := 0.30
def science_class_percentage : ℚ := 0.20
def history_class_percentage : ℚ := 0.10

def math_hours := math_class_percentage * weekly_school_hours
def language_hours := language_class_percentage * weekly_school_hours
def science_hours := science_class_percentage * weekly_school_hours
def history_hours := history_class_percentage * weekly_school_hours

def total_class_hours := math_hours + language_hours + science_hours + history_hours

theorem ruth_weekly_class_hours : total_class_hours = 34 := by
  -- Calculation proof logic will go here
  sorry

end ruth_weekly_class_hours_l1463_146331


namespace max_distance_line_l1463_146308

noncomputable def equation_of_line (x y : ℝ) : ℝ := x + 2 * y - 5

theorem max_distance_line (x y : ℝ) : 
  equation_of_line 1 2 = 0 ∧ 
  (∀ (a b c : ℝ), c ≠ 0 → (x = 1 ∧ y = 2 → equation_of_line x y = 0)) ∧ 
  (∀ (L : ℝ → ℝ → ℝ), L 1 2 = 0 → (L = equation_of_line)) :=
sorry

end max_distance_line_l1463_146308


namespace solve_fractional_equation_l1463_146376

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x + 1) / 5 - x / 10 = 2 → x = 6 :=
by
  intros x h
  sorry

end solve_fractional_equation_l1463_146376


namespace symmetric_point_coordinates_l1463_146363

theorem symmetric_point_coordinates (M N : ℝ × ℝ) (x y : ℝ) 
  (hM : M = (-2, 1)) 
  (hN_symmetry : N = (M.1, -M.2)) : N = (-2, -1) :=
by
  sorry

end symmetric_point_coordinates_l1463_146363


namespace consecutive_odd_integers_sum_l1463_146345

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 134) : x + (x + 2) + (x + 4) = 201 := 
by sorry

end consecutive_odd_integers_sum_l1463_146345


namespace EF_length_proof_l1463_146313

noncomputable def length_BD (AB BC : ℝ) : ℝ := Real.sqrt (AB^2 + BC^2)

noncomputable def length_EF (BD AB BC : ℝ) : ℝ :=
  let BE := BD * AB / BD
  let BF := BD * BC / AB
  BE + BF

theorem EF_length_proof : 
  ∀ (AB BC : ℝ), AB = 4 ∧ BC = 3 →
  length_EF (length_BD AB BC) AB BC = 125 / 12 :=
by
  intros AB BC h
  rw [length_BD, length_EF]
  simp
  rw [Real.sqrt_eq_rpow]
  simp
  sorry

end EF_length_proof_l1463_146313


namespace perpendicular_lines_l1463_146314

theorem perpendicular_lines (a : ℝ) :
  (if a ≠ 0 then a^2 ≠ 0 else true) ∧ (a^2 * a + (-1/a) * 2 = -1) → (a = 2 ∨ a = 0) :=
by
  sorry

end perpendicular_lines_l1463_146314


namespace necessary_sufficient_condition_l1463_146394

theorem necessary_sufficient_condition (a : ℝ) :
  (∃ x : ℝ, ax^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ a ≤ 1 := sorry

end necessary_sufficient_condition_l1463_146394


namespace value_of_k_l1463_146381

-- Let k be a real number
variable (k : ℝ)

-- The given condition as a hypothesis
def condition := ∀ x : ℝ, (x + 3) * (x + 2) = k + 3 * x

-- The statement to prove
theorem value_of_k (h : ∀ x : ℝ, (x + 3) * (x + 2) = k + 3 * x) : k = 5 :=
sorry

end value_of_k_l1463_146381


namespace consecutive_even_sum_l1463_146325

theorem consecutive_even_sum (N S : ℤ) (m : ℤ) 
  (hk : 2 * m + 1 > 0) -- k is the number of consecutive even numbers, which is odd
  (h_sum : (2 * m + 1) * N = S) -- The condition of the sum
  (h_even : N % 2 = 0) -- The middle number is even
  : (∃ k : ℤ, k = 2 * m + 1 ∧ k > 0 ∧ (k * N / 2) = S/2 ) := 
  sorry

end consecutive_even_sum_l1463_146325


namespace rabbit_total_apples_90_l1463_146307

-- Define the number of apples each animal places in a basket
def rabbit_apple_per_basket : ℕ := 5
def deer_apple_per_basket : ℕ := 6

-- Define the number of baskets each animal uses
variable (h_r h_d : ℕ)

-- Define the total number of apples collected by both animals
def total_apples : ℕ := rabbit_apple_per_basket * h_r

-- Conditions
axiom deer_basket_count_eq_rabbit : h_d = h_r - 3
axiom same_total_apples : total_apples = deer_apple_per_basket * h_d

-- Goal: Prove that the total number of apples the rabbit collected is 90
theorem rabbit_total_apples_90 : total_apples = 90 := sorry

end rabbit_total_apples_90_l1463_146307


namespace arithmetic_sequence_problem_l1463_146311

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120)
  : 2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_problem_l1463_146311


namespace problem_l1463_146399

theorem problem (m : ℕ) (h : m = 16^2023) : m / 8 = 2^8089 :=
by {
  sorry
}

end problem_l1463_146399


namespace find_coordinates_of_P_l1463_146304

theorem find_coordinates_of_P : 
  ∃ P: ℝ × ℝ, 
  (∃ θ: ℝ, 0 ≤ θ ∧ θ ≤ π ∧ P = (3 * Real.cos θ, 4 * Real.sin θ)) ∧ 
  ∃ m: ℝ, m = 1 ∧ P.fst = P.snd ∧ P = (12/5, 12/5) :=
by {
  sorry -- Proof is omitted as per instruction
}

end find_coordinates_of_P_l1463_146304


namespace number_of_students_l1463_146386

-- Definitions based on the problem conditions
def mini_cupcakes := 14
def donut_holes := 12
def desserts_per_student := 2

-- Total desserts calculation
def total_desserts := mini_cupcakes + donut_holes

-- Prove the number of students
theorem number_of_students : total_desserts / desserts_per_student = 13 :=
by
  -- Proof can be filled in here
  sorry

end number_of_students_l1463_146386


namespace range_of_m_l1463_146332

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 -> (m^2 - m) * 2^x - (1/2)^x < 1) →
  -2 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l1463_146332


namespace find_k_and_b_l1463_146354

noncomputable def setA := {p : ℝ × ℝ | p.2^2 - p.1 - 1 = 0}
noncomputable def setB := {p : ℝ × ℝ | 4 * p.1^2 + 2 * p.1 - 2 * p.2 + 5 = 0}
noncomputable def setC (k b : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + b}

theorem find_k_and_b (k b : ℕ) : 
  (setA ∪ setB) ∩ setC k b = ∅ ↔ (k = 1 ∧ b = 2) := 
sorry

end find_k_and_b_l1463_146354


namespace mildred_total_oranges_l1463_146338

-- Conditions
def initial_oranges : ℕ := 77
def additional_oranges : ℕ := 2

-- Question/Goal
theorem mildred_total_oranges : initial_oranges + additional_oranges = 79 := by
  sorry

end mildred_total_oranges_l1463_146338


namespace silver_status_families_l1463_146347

theorem silver_status_families 
  (goal : ℕ) 
  (remaining : ℕ) 
  (bronze_families : ℕ) 
  (bronze_donation : ℕ) 
  (gold_families : ℕ) 
  (gold_donation : ℕ) 
  (silver_donation : ℕ) 
  (total_raised_so_far : goal - remaining = 700)
  (amount_raised_by_bronze : bronze_families * bronze_donation = 250)
  (amount_raised_by_gold : gold_families * gold_donation = 100)
  (amount_raised_by_silver : 700 - 250 - 100 = 350) :
  ∃ (s : ℕ), s * silver_donation = 350 ∧ s = 7 :=
by
  sorry

end silver_status_families_l1463_146347


namespace y_coord_intersection_with_y_axis_l1463_146340

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + 11

-- Define the point P
def P : ℝ × ℝ := (1, curve 1)

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 3 * x^2

-- Define the tangent line at point P (1, 12)
def tangent_line (x : ℝ) : ℝ := 3 * (x - 1) + 12

-- Proof statement
theorem y_coord_intersection_with_y_axis : 
  tangent_line 0 = 9 :=
by
  -- proof goes here
  sorry

end y_coord_intersection_with_y_axis_l1463_146340


namespace y_pow_x_eq_x_pow_y_l1463_146322

open Real

noncomputable def x (n : ℕ) : ℝ := (1 + 1 / n) ^ n
noncomputable def y (n : ℕ) : ℝ := (1 + 1 / n) ^ (n + 1)

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) : (y n) ^ (x n) = (x n) ^ (y n) :=
by
  sorry

end y_pow_x_eq_x_pow_y_l1463_146322


namespace admission_charge_for_adult_l1463_146398

theorem admission_charge_for_adult 
(admission_charge_per_child : ℝ)
(total_paid : ℝ)
(children_count : ℕ)
(admission_charge_for_adult : ℝ) :
admission_charge_per_child = 0.75 →
total_paid = 3.25 →
children_count = 3 →
admission_charge_for_adult + admission_charge_per_child * children_count = total_paid →
admission_charge_for_adult = 1.00 :=
by
  intros h1 h2 h3 h4
  sorry

end admission_charge_for_adult_l1463_146398


namespace find_original_price_l1463_146371

theorem find_original_price (a b x : ℝ) (h : x * (1 - 0.1) - a = b) : 
  x = (a + b) / (1 - 0.1) :=
sorry

end find_original_price_l1463_146371
