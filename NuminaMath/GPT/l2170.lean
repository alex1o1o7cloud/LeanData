import Mathlib

namespace NUMINAMATH_GPT_average_of_11_numbers_l2170_217069

theorem average_of_11_numbers (a b c d e f g h i j k : ℝ)
  (h_first_6_avg : (a + b + c + d + e + f) / 6 = 98)
  (h_last_6_avg : (f + g + h + i + j + k) / 6 = 65)
  (h_6th_number : f = 318) :
  ((a + b + c + d + e + f + g + h + i + j + k) / 11) = 60 :=
by
  sorry

end NUMINAMATH_GPT_average_of_11_numbers_l2170_217069


namespace NUMINAMATH_GPT_edge_length_of_small_cube_l2170_217043

-- Define the parameters
def volume_cube : ℕ := 1000
def num_small_cubes : ℕ := 8
def remaining_volume : ℕ := 488

-- Define the main theorem
theorem edge_length_of_small_cube (x : ℕ) :
  (volume_cube - num_small_cubes * x^3 = remaining_volume) → x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_edge_length_of_small_cube_l2170_217043


namespace NUMINAMATH_GPT_evaluate_expression_l2170_217006

theorem evaluate_expression :
    123 - (45 * (9 - 6) - 78) + (0 / 1994) = 66 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2170_217006


namespace NUMINAMATH_GPT_twentyfive_percent_in_usd_l2170_217019

variable (X : ℝ)
variable (Y : ℝ) (hY : Y > 0)

theorem twentyfive_percent_in_usd : 0.25 * X * Y = (0.25 : ℝ) * X * Y := by
  sorry

end NUMINAMATH_GPT_twentyfive_percent_in_usd_l2170_217019


namespace NUMINAMATH_GPT_composite_for_infinitely_many_n_l2170_217089

theorem composite_for_infinitely_many_n :
  ∃ᶠ n in at_top, (n > 0) ∧ (n % 6 = 4) → ∃ p, p ≠ 1 ∧ p ≠ n^n + (n+1)^(n+1) :=
sorry

end NUMINAMATH_GPT_composite_for_infinitely_many_n_l2170_217089


namespace NUMINAMATH_GPT_pepper_left_l2170_217046

def initial_pepper : ℝ := 0.25
def used_pepper : ℝ := 0.16
def remaining_pepper : ℝ := 0.09

theorem pepper_left (h1 : initial_pepper = 0.25) (h2 : used_pepper = 0.16) :
  initial_pepper - used_pepper = remaining_pepper :=
by
  sorry

end NUMINAMATH_GPT_pepper_left_l2170_217046


namespace NUMINAMATH_GPT_largest_of_set_l2170_217029

theorem largest_of_set : 
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  c = 2 ∧ (d < b ∧ b < a ∧ a < c) := by
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  sorry

end NUMINAMATH_GPT_largest_of_set_l2170_217029


namespace NUMINAMATH_GPT_cistern_wet_surface_area_l2170_217040

def cistern_length : ℝ := 4
def cistern_width : ℝ := 8
def water_depth : ℝ := 1.25

def area_bottom (l w : ℝ) : ℝ := l * w
def area_pair1 (l h : ℝ) : ℝ := 2 * (l * h)
def area_pair2 (w h : ℝ) : ℝ := 2 * (w * h)
def total_wet_surface_area (l w h : ℝ) : ℝ := area_bottom l w + area_pair1 l h + area_pair2 w h

theorem cistern_wet_surface_area : total_wet_surface_area cistern_length cistern_width water_depth = 62 := 
by 
  sorry

end NUMINAMATH_GPT_cistern_wet_surface_area_l2170_217040


namespace NUMINAMATH_GPT_PQ_ratio_l2170_217015

-- Definitions
def hexagon_area : ℕ := 7
def base_of_triangle : ℕ := 4

-- Conditions
def PQ_bisects_area (A : ℕ) : Prop :=
  A = hexagon_area / 2

def area_below_PQ (U T : ℚ) : Prop :=
  U + T = hexagon_area / 2 ∧ U = 1

def triangle_area (T b : ℚ) : ℚ :=
  1/2 * b * (5/4)

def XQ_QY_ratio (XQ QY : ℚ) : ℚ :=
  XQ / QY

-- Theorem Statement
theorem PQ_ratio (XQ QY : ℕ) (h1 : PQ_bisects_area (hexagon_area / 2))
  (h2 : area_below_PQ 1 (triangle_area (5/2) base_of_triangle))
  (h3 : XQ + QY = base_of_triangle) : XQ_QY_ratio XQ QY = 1 := sorry

end NUMINAMATH_GPT_PQ_ratio_l2170_217015


namespace NUMINAMATH_GPT_James_pays_6_dollars_l2170_217093

-- Defining the conditions
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def friend_share : ℚ := 0.5

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack

-- Total cost calculation
def total_cost : ℚ := total_stickers * cost_per_sticker

-- James' payment calculation
def james_payment : ℚ := total_cost * friend_share

-- Theorem statement to be proven
theorem James_pays_6_dollars : james_payment = 6 := by
  sorry

end NUMINAMATH_GPT_James_pays_6_dollars_l2170_217093


namespace NUMINAMATH_GPT_increasing_on_neg_reals_l2170_217092

variable (f : ℝ → ℝ)

def even_function : Prop := ∀ x : ℝ, f (-x) = f x

def decreasing_on_pos_reals : Prop := ∀ x1 x2 : ℝ, (0 < x1 ∧ 0 < x2 ∧ x1 < x2) → f x1 > f x2

theorem increasing_on_neg_reals
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on_pos_reals f) :
  ∀ x1 x2 : ℝ, (x1 < 0 ∧ x2 < 0 ∧ x1 < x2) → f x1 < f x2 :=
by sorry

end NUMINAMATH_GPT_increasing_on_neg_reals_l2170_217092


namespace NUMINAMATH_GPT_min_value_x2_y2_z2_l2170_217078

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^2 + y^2 + z^2 - 3 * x * y * z = 1) :
  x^2 + y^2 + z^2 ≥ 1 := 
sorry

end NUMINAMATH_GPT_min_value_x2_y2_z2_l2170_217078


namespace NUMINAMATH_GPT_purely_imaginary_complex_l2170_217027

theorem purely_imaginary_complex (a : ℝ) : (a - 2) = 0 → a = 2 :=
by
  intro h
  exact eq_of_sub_eq_zero h

end NUMINAMATH_GPT_purely_imaginary_complex_l2170_217027


namespace NUMINAMATH_GPT_misha_is_lying_l2170_217090

theorem misha_is_lying
  (truth_tellers_scores : Fin 9 → ℕ)
  (h_all_odd : ∀ i, truth_tellers_scores i % 2 = 1)
  (total_scores_truth_tellers : (Fin 9 → ℕ) → ℕ)
  (h_sum_scores : total_scores_truth_tellers truth_tellers_scores = 18) :
  ∀ (misha_score : ℕ), misha_score = 2 → misha_score % 2 = 1 → False :=
by
  intros misha_score hms hmo
  sorry

end NUMINAMATH_GPT_misha_is_lying_l2170_217090


namespace NUMINAMATH_GPT_polina_pizza_combinations_correct_l2170_217007

def polina_pizza_combinations : Nat :=
  let total_toppings := 5
  let possible_combinations := total_toppings * (total_toppings - 1) / 2
  possible_combinations

theorem polina_pizza_combinations_correct :
  polina_pizza_combinations = 10 :=
by
  sorry

end NUMINAMATH_GPT_polina_pizza_combinations_correct_l2170_217007


namespace NUMINAMATH_GPT_odd_nat_numbers_eq_1_l2170_217099

-- Definitions of conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem odd_nat_numbers_eq_1
  (a b c d : ℕ)
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : is_odd a) (h5 : is_odd b) (h6 : is_odd c) (h7 : is_odd d)
  (h8 : a * d = b * c)
  (h9 : is_power_of_two (a + d))
  (h10 : is_power_of_two (b + c)) :
  a = 1 :=
sorry

end NUMINAMATH_GPT_odd_nat_numbers_eq_1_l2170_217099


namespace NUMINAMATH_GPT_fraction_of_tank_used_l2170_217011

theorem fraction_of_tank_used (speed : ℝ) (fuel_efficiency : ℝ) (initial_fuel : ℝ) (time_traveled : ℝ)
  (h_speed : speed = 40) (h_fuel_eff : fuel_efficiency = 1 / 40) (h_initial_fuel : initial_fuel = 12) 
  (h_time : time_traveled = 5) : 
  (speed * time_traveled * fuel_efficiency) / initial_fuel = 5 / 12 :=
by
  -- Here the proof would go, but we add sorry to indicate it's incomplete.
  sorry

end NUMINAMATH_GPT_fraction_of_tank_used_l2170_217011


namespace NUMINAMATH_GPT_ab_value_l2170_217065

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 270) :
  a * b = 216 := by
  sorry

end NUMINAMATH_GPT_ab_value_l2170_217065


namespace NUMINAMATH_GPT_cubic_polynomial_evaluation_l2170_217020

theorem cubic_polynomial_evaluation
  (f : ℚ → ℚ)
  (cubic_f : ∃ a b c d : ℚ, ∀ x, f x = a*x^3 + b*x^2 + c*x + d)
  (h1 : f (-2) = -4)
  (h2 : f 3 = -9)
  (h3 : f (-4) = -16) :
  f 1 = -23 :=
sorry

end NUMINAMATH_GPT_cubic_polynomial_evaluation_l2170_217020


namespace NUMINAMATH_GPT_a_3_def_a_4_def_a_r_recurrence_l2170_217005

-- Define minimally the structure of the problem.
noncomputable def a_r (r : ℕ) : ℕ := -- Definition for minimum phone calls required.
by sorry

-- Assertions for the specific cases provided.
theorem a_3_def : a_r 3 = 3 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_4_def : a_r 4 = 4 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_r_recurrence (r : ℕ) (hr : r ≥ 3) : a_r r ≤ a_r (r - 1) + 2 :=
by
  -- Proof is omitted with sorry.
  sorry

end NUMINAMATH_GPT_a_3_def_a_4_def_a_r_recurrence_l2170_217005


namespace NUMINAMATH_GPT_math_problem_l2170_217035

theorem math_problem 
  (a : ℤ) 
  (h_a : a = -1) 
  (b : ℚ) 
  (h_b : b = 0) 
  (c : ℕ) 
  (h_c : c = 1)
  : a^2024 + 2023 * b - c^2023 = 0 := by
  sorry

end NUMINAMATH_GPT_math_problem_l2170_217035


namespace NUMINAMATH_GPT_three_digit_number_divisible_by_7_l2170_217010

theorem three_digit_number_divisible_by_7 (t : ℕ) :
  (n : ℕ) = 600 + 10 * t + 5 →
  n ≥ 100 ∧ n < 1000 →
  n % 10 = 5 →
  (n / 100) % 10 = 6 →
  n % 7 = 0 →
  n = 665 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_divisible_by_7_l2170_217010


namespace NUMINAMATH_GPT_cube_pyramid_same_volume_height_l2170_217091

theorem cube_pyramid_same_volume_height (h : ℝ) :
  let cube_edge : ℝ := 5
  let pyramid_base_edge : ℝ := 6
  let cube_volume : ℝ := cube_edge ^ 3
  let pyramid_volume : ℝ := (1 / 3) * (pyramid_base_edge ^ 2) * h
  cube_volume = pyramid_volume → h = 125 / 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cube_pyramid_same_volume_height_l2170_217091


namespace NUMINAMATH_GPT_prism_distance_to_plane_l2170_217076

theorem prism_distance_to_plane
  (side_length : ℝ)
  (volume : ℝ)
  (h : ℝ)
  (base_is_square : side_length = 6)
  (volume_formula : volume = (1 / 3) * h * (side_length ^ 2)) :
  h = 8 := 
  by sorry

end NUMINAMATH_GPT_prism_distance_to_plane_l2170_217076


namespace NUMINAMATH_GPT_tom_speed_RB_l2170_217087

/-- Let d be the distance between B and C (in miles).
    Let 2d be the distance between R and B (in miles).
    Let v be Tom’s speed driving from R to B (in mph).
    Given conditions:
    1. Tom's speed from B to C = 20 mph.
    2. Total average speed of the whole journey = 36 mph.
    Prove that Tom's speed driving from R to B is 60 mph. -/
theorem tom_speed_RB
  (d : ℝ) (v : ℝ)
  (h1 : 20 ≠ 0)
  (h2 : 36 ≠ 0)
  (avg_speed : 3 * d / (2 * d / v + d / 20) = 36) :
  v = 60 := 
sorry

end NUMINAMATH_GPT_tom_speed_RB_l2170_217087


namespace NUMINAMATH_GPT_prob_sum_geq_9_prob_tangent_line_prob_isosceles_triangle_l2170_217052

-- Definitions
def fair_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Question 1: Probability that a + b >= 9
theorem prob_sum_geq_9 (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  a + b ≥ 9 → (∃ (valid_outcomes : Finset (ℕ × ℕ)),
    valid_outcomes = {(3, 6), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 3), (6, 4), (6, 5), (6, 6)} ∧
    valid_outcomes.card = 10 ∧
    10 / 36 = 5 / 18) :=
sorry

-- Question 2: Probability that the line ax + by + 5 = 0 is tangent to the circle x^2 + y^2 = 1
theorem prob_tangent_line (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  (∃ (tangent_outcomes : Finset (ℕ × ℕ)),
    tangent_outcomes = {(3, 4), (4, 3)} ∧
    a^2 + b^2 = 25 ∧
    tangent_outcomes.card = 2 ∧
    2 / 36 = 1 / 18) :=
sorry

-- Question 3: Probability that the lengths a, b, and 5 form an isosceles triangle
theorem prob_isosceles_triangle (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  (∃ (isosceles_outcomes : Finset (ℕ × ℕ)),
    isosceles_outcomes = {(1, 5), (2, 5), (3, 3), (3, 5), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 5), (6, 6)} ∧
    isosceles_outcomes.card = 14 ∧
    14 / 36 = 7 / 18) :=
sorry

end NUMINAMATH_GPT_prob_sum_geq_9_prob_tangent_line_prob_isosceles_triangle_l2170_217052


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l2170_217083

theorem arithmetic_sequence_product (b : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ n, b (n + 1) > b n)
  (h2 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l2170_217083


namespace NUMINAMATH_GPT_ratio_KL_eq_3_over_5_l2170_217000

theorem ratio_KL_eq_3_over_5
  (K L : ℤ)
  (h : ∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    (K : ℝ) / (x + 3) + (L : ℝ) / (x^2 - 3 * x) = (x^2 - x + 5) / (x^3 + x^2 - 9 * x)):
  (K : ℝ) / (L : ℝ) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_KL_eq_3_over_5_l2170_217000


namespace NUMINAMATH_GPT_range_of_k_l2170_217075

theorem range_of_k (k : ℝ) : (∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 - 2*k*x + k)) ↔ (k ∈ Set.Iic 0 ∨ k ∈ Set.Ici 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2170_217075


namespace NUMINAMATH_GPT_max_k_constant_for_right_triangle_l2170_217066

theorem max_k_constant_for_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a ≤ b) (h2 : b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3*Real.sqrt 2) * a * b * c :=
by 
  sorry

end NUMINAMATH_GPT_max_k_constant_for_right_triangle_l2170_217066


namespace NUMINAMATH_GPT_meal_center_adults_l2170_217096

theorem meal_center_adults (cans : ℕ) (children_served : ℕ) (adults_served : ℕ) (total_children : ℕ) 
  (initial_cans : cans = 10) 
  (children_per_can : children_served = 7) 
  (adults_per_can : adults_served = 4) 
  (children_to_feed : total_children = 21) : 
  (cans - (total_children / children_served)) * adults_served = 28 := by
  have h1: 3 = total_children / children_served := by
    sorry
  have h2: 7 = cans - 3 := by
    sorry
  have h3: 28 = 7 * adults_served := by
    sorry
  have h4: adults_served = 4 := by
    sorry
  sorry

end NUMINAMATH_GPT_meal_center_adults_l2170_217096


namespace NUMINAMATH_GPT_unique_b_for_unique_solution_l2170_217080

theorem unique_b_for_unique_solution (c : ℝ) (h₁ : c ≠ 0) :
  (∃ b : ℝ, b > 0 ∧ ∃! x : ℝ, x^2 + (b + (2 / b)) * x + c = 0) →
  c = 2 :=
by
  -- sorry will go here to indicate the proof is to be filled in
  sorry

end NUMINAMATH_GPT_unique_b_for_unique_solution_l2170_217080


namespace NUMINAMATH_GPT_south_side_students_count_l2170_217002

variables (N : ℕ)
def students_total := 41
def difference := 3

theorem south_side_students_count (N : ℕ) (h₁ : 2 * N + difference = students_total) : N + difference = 22 :=
sorry

end NUMINAMATH_GPT_south_side_students_count_l2170_217002


namespace NUMINAMATH_GPT_mark_birth_year_proof_l2170_217063

-- Conditions
def current_year := 2021
def janice_age := 21
def graham_age := 2 * janice_age
def mark_age := graham_age + 3
def mark_birth_year (current_year : ℕ) (mark_age : ℕ) := current_year - mark_age

-- Statement to prove
theorem mark_birth_year_proof : 
  mark_birth_year current_year mark_age = 1976 := by
  sorry

end NUMINAMATH_GPT_mark_birth_year_proof_l2170_217063


namespace NUMINAMATH_GPT_smallest_triangle_perimeter_l2170_217059

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def smallest_possible_prime_perimeter : ℕ :=
  31

theorem smallest_triangle_perimeter :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                  a > 5 ∧ b > 5 ∧ c > 5 ∧
                  is_prime a ∧ is_prime b ∧ is_prime c ∧
                  triangle_inequality a b c ∧
                  is_prime (a + b + c) ∧
                  a + b + c = smallest_possible_prime_perimeter :=
sorry

end NUMINAMATH_GPT_smallest_triangle_perimeter_l2170_217059


namespace NUMINAMATH_GPT_rectangle_area_l2170_217037

theorem rectangle_area (x : ℝ) (h1 : (x^2 + (3*x)^2) = (15*Real.sqrt 2)^2) :
  (x * (3 * x)) = 135 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2170_217037


namespace NUMINAMATH_GPT_beads_currently_have_l2170_217001

-- Definitions of the conditions
def friends : Nat := 6
def beads_per_bracelet : Nat := 8
def additional_beads_needed : Nat := 12

-- Theorem statement
theorem beads_currently_have : (beads_per_bracelet * friends - additional_beads_needed) = 36 := by
  sorry

end NUMINAMATH_GPT_beads_currently_have_l2170_217001


namespace NUMINAMATH_GPT_find_a_l2170_217025

theorem find_a (a : ℝ) (x y : ℝ) :
  (x^2 - 4*x + y^2 = 0) →
  ((x - a)^2 + y^2 = 4*((x - 1)^2 + y^2)) →
  a = -2 :=
by
  intros h_circle h_distance
  sorry

end NUMINAMATH_GPT_find_a_l2170_217025


namespace NUMINAMATH_GPT_quadrilateral_side_length_eq_12_l2170_217022

-- Definitions
def EF : ℝ := 7
def FG : ℝ := 15
def GH : ℝ := 7
def HE : ℝ := 12
def EH : ℝ := 12

-- Statement to prove that EH = 12 given the definition and conditions
theorem quadrilateral_side_length_eq_12
  (EF_eq : EF = 7)
  (FG_eq : FG = 15)
  (GH_eq : GH = 7)
  (HE_eq : HE = 12)
  (EH_eq : EH = 12) : 
  EH = 12 :=
sorry

end NUMINAMATH_GPT_quadrilateral_side_length_eq_12_l2170_217022


namespace NUMINAMATH_GPT_volleyball_problem_correct_l2170_217014

noncomputable def volleyball_problem : Nat :=
  let total_players := 16
  let triplets : Finset String := {"Alicia", "Amanda", "Anna"}
  let twins : Finset String := {"Beth", "Brenda"}
  let remaining_players := total_players - triplets.card - twins.card
  let no_triplets_no_twins := Nat.choose remaining_players 6
  let one_triplet_no_twins := triplets.card * Nat.choose remaining_players 5
  let no_triplets_one_twin := twins.card * Nat.choose remaining_players 5
  no_triplets_no_twins + one_triplet_no_twins + no_triplets_one_twin

theorem volleyball_problem_correct : volleyball_problem = 2772 := by
  sorry

end NUMINAMATH_GPT_volleyball_problem_correct_l2170_217014


namespace NUMINAMATH_GPT_triangle_perimeter_l2170_217088

theorem triangle_perimeter :
  let a := 15
  let b := 10
  let c := 12
  (a < b + c) ∧ (b < a + c) ∧ (c < a + b) →
  (a + b + c = 37) :=
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2170_217088


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2170_217031

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h_iso : ¬(4 + 4 > 9 ∧ 4 + 9 > 4 ∧ 9 + 4 > 4))
  (h_ineq : (9 + 9 > 4) ∧ (9 + 4 > 9) ∧ (4 + 9 > 9)) : 2 * b + a = 22 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2170_217031


namespace NUMINAMATH_GPT_odot_computation_l2170_217098

noncomputable def op (a b : ℚ) : ℚ := 
  (a + b) / (1 + a * b)

theorem odot_computation : op 2 (op 3 (op 4 5)) = 7 / 8 := 
  by 
  sorry

end NUMINAMATH_GPT_odot_computation_l2170_217098


namespace NUMINAMATH_GPT_original_proposition_false_converse_false_inverse_false_contrapositive_false_l2170_217081

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop := 
  (a * b ≤ 0) → (a ≤ 0 ∨ b ≤ 0)

-- Define the converse
def converse (a b : ℝ) : Prop := 
  (a ≤ 0 ∨ b ≤ 0) → (a * b ≤ 0)

-- Define the inverse
def inverse (a b : ℝ) : Prop := 
  (a * b > 0) → (a > 0 ∧ b > 0)

-- Define the contrapositive
def contrapositive (a b : ℝ) : Prop := 
  (a > 0 ∧ b > 0) → (a * b > 0)

-- Prove that the original proposition is false
theorem original_proposition_false : ∀ (a b : ℝ), ¬ original_proposition a b :=
by sorry

-- Prove that the converse is false
theorem converse_false : ∀ (a b : ℝ), ¬ converse a b :=
by sorry

-- Prove that the inverse is false
theorem inverse_false : ∀ (a b : ℝ), ¬ inverse a b :=
by sorry

-- Prove that the contrapositive is false
theorem contrapositive_false : ∀ (a b : ℝ), ¬ contrapositive a b :=
by sorry

end NUMINAMATH_GPT_original_proposition_false_converse_false_inverse_false_contrapositive_false_l2170_217081


namespace NUMINAMATH_GPT_value_of_a_l2170_217041

theorem value_of_a (a : ℝ) (h : abs (2 * a + 1) = 3) :
  a = -2 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_value_of_a_l2170_217041


namespace NUMINAMATH_GPT_find_alpha_minus_beta_find_cos_2alpha_minus_beta_l2170_217008

-- Definitions and assumptions
variables (α β : ℝ)
axiom sin_alpha : Real.sin α = (Real.sqrt 5) / 5
axiom sin_beta : Real.sin β = (3 * Real.sqrt 10) / 10
axiom alpha_acute : 0 < α ∧ α < Real.pi / 2
axiom beta_acute : 0 < β ∧ β < Real.pi / 2

-- Statement to prove α - β = -π/4
theorem find_alpha_minus_beta : α - β = -Real.pi / 4 :=
sorry

-- Given α - β = -π/4, statement to prove cos(2α - β) = 3√10 / 10
theorem find_cos_2alpha_minus_beta (h : α - β = -Real.pi / 4) : Real.cos (2 * α - β) = (3 * Real.sqrt 10) / 10 :=
sorry

end NUMINAMATH_GPT_find_alpha_minus_beta_find_cos_2alpha_minus_beta_l2170_217008


namespace NUMINAMATH_GPT_tangential_quadrilateral_difference_l2170_217048

-- Definitions of the conditions given in the problem
def is_cyclic_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the quadrilateral vertices lie on a circle
def is_tangential_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the sides are tangent to a common incircle
def point_tangency (a b c : ℝ) : Prop := sorry

-- Main theorem
theorem tangential_quadrilateral_difference (AB BC CD DA : ℝ) (x y : ℝ) 
  (h1 : is_cyclic_quadrilateral AB BC CD DA)
  (h2 : is_tangential_quadrilateral AB BC CD DA)
  (h3 : AB = 80) (h4 : BC = 140) (h5 : CD = 120) (h6 : DA = 100)
  (h7 : point_tangency x y CD)
  (h8 : x + y = 120) :
  |x - y| = 80 := 
sorry

end NUMINAMATH_GPT_tangential_quadrilateral_difference_l2170_217048


namespace NUMINAMATH_GPT_uncovered_area_l2170_217030

theorem uncovered_area {s₁ s₂ : ℝ} (hs₁ : s₁ = 10) (hs₂ : s₂ = 4) : 
  (s₁^2 - 2 * s₂^2) = 68 := by
  sorry

end NUMINAMATH_GPT_uncovered_area_l2170_217030


namespace NUMINAMATH_GPT_teddy_bears_count_l2170_217060

theorem teddy_bears_count (toys_count : ℕ) (toy_cost : ℕ) (total_money : ℕ) (teddy_bear_cost : ℕ) : 
  toys_count = 28 → 
  toy_cost = 10 → 
  total_money = 580 → 
  teddy_bear_cost = 15 →
  ((total_money - toys_count * toy_cost) / teddy_bear_cost) = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_teddy_bears_count_l2170_217060


namespace NUMINAMATH_GPT_ratio_of_volumes_of_spheres_l2170_217062

theorem ratio_of_volumes_of_spheres (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a / b = 1 / 2 ∧ b / c = 2 / 3) : a^3 / b^3 = 1 / 8 ∧ b^3 / c^3 = 8 / 27 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_of_spheres_l2170_217062


namespace NUMINAMATH_GPT_solve_equation_l2170_217072

theorem solve_equation (x : ℝ) : x^4 + (4 - x)^4 = 272 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 := 
by sorry

end NUMINAMATH_GPT_solve_equation_l2170_217072


namespace NUMINAMATH_GPT_coloring_ways_l2170_217042

-- Definitions for colors
inductive Color
| red
| green

open Color

-- Definition of the coloring function
def color (n : ℕ) : Color := sorry

-- Conditions:
-- 1. Each positive integer is colored either red or green
def condition1 (n : ℕ) : n > 0 → (color n = red ∨ color n = green) := sorry

-- 2. The sum of any two different red numbers is a red number
def condition2 (r1 r2 : ℕ) : r1 ≠ r2 → color r1 = red → color r2 = red → color (r1 + r2) = red := sorry

-- 3. The sum of any two different green numbers is a green number
def condition3 (g1 g2 : ℕ) : g1 ≠ g2 → color g1 = green → color g2 = green → color (g1 + g2) = green := sorry

-- The required theorem
theorem coloring_ways : ∃! (f : ℕ → Color), 
  (∀ n, n > 0 → (f n = red ∨ f n = green)) ∧ 
  (∀ r1 r2, r1 ≠ r2 → f r1 = red → f r2 = red → f (r1 + r2) = red) ∧
  (∀ g1 g2, g1 ≠ g2 → f g1 = green → f g2 = green → f (g1 + g2) = green) :=
sorry

end NUMINAMATH_GPT_coloring_ways_l2170_217042


namespace NUMINAMATH_GPT_last_person_is_knight_l2170_217026

def KnightLiarsGame1 (n : ℕ) : Prop :=
  let m := 10
  let p := 13
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

def KnightLiarsGame2 (n : ℕ) : Prop :=
  let m := 12
  let p := 9
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

theorem last_person_is_knight :
  ∃ n, KnightLiarsGame1 n ∧ KnightLiarsGame2 n :=
by 
  sorry

end NUMINAMATH_GPT_last_person_is_knight_l2170_217026


namespace NUMINAMATH_GPT_normal_pumping_rate_l2170_217018

-- Define the conditions and the proof problem
def pond_capacity : ℕ := 200
def drought_factor : ℚ := 2/3
def fill_time : ℕ := 50

theorem normal_pumping_rate (R : ℚ) :
  (drought_factor * R) * (fill_time : ℚ) = pond_capacity → R = 6 :=
by
  sorry

end NUMINAMATH_GPT_normal_pumping_rate_l2170_217018


namespace NUMINAMATH_GPT_find_numbers_l2170_217038

theorem find_numbers 
  (a b c d : ℝ)
  (h1 : b / c = c / a)
  (h2 : a + b + c = 19)
  (h3 : b - c = c - d)
  (h4 : b + c + d = 12) :
  (a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨ (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2) :=
sorry

end NUMINAMATH_GPT_find_numbers_l2170_217038


namespace NUMINAMATH_GPT_total_points_zach_ben_l2170_217071

theorem total_points_zach_ben (zach_points ben_points : ℝ) (h1 : zach_points = 42.0) (h2 : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
by
  sorry

end NUMINAMATH_GPT_total_points_zach_ben_l2170_217071


namespace NUMINAMATH_GPT_emma_correct_percentage_l2170_217079

theorem emma_correct_percentage (t : ℕ) (lt : t > 0)
  (liam_correct_alone : ℝ := 0.70)
  (liam_overall_correct : ℝ := 0.82)
  (emma_correct_alone : ℝ := 0.85)
  (joint_error_rate : ℝ := 0.05)
  (liam_solved_together_correct : ℝ := liam_overall_correct * t - liam_correct_alone * (t / 2)) :
  (emma_correct_alone * (t / 2) + (1 - joint_error_rate) * liam_solved_together_correct) / t * 100 = 87.15 :=
by
  sorry

end NUMINAMATH_GPT_emma_correct_percentage_l2170_217079


namespace NUMINAMATH_GPT_sum_of_absolute_values_l2170_217084

theorem sum_of_absolute_values (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = n^2 - 4 * n + 2) →
  a 1 = -1 →
  (∀ n, 1 < n → a n = 2 * n - 5) →
  ((abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) +
    abs (a 6) + abs (a 7) + abs (a 8) + abs (a 9) + abs (a 10)) = 66) :=
by
  intros hS a1_eq ha_eq
  sorry

end NUMINAMATH_GPT_sum_of_absolute_values_l2170_217084


namespace NUMINAMATH_GPT_connect_5_points_four_segments_l2170_217039

theorem connect_5_points_four_segments (A B C D E : Type) (h : ∀ (P Q R : Type), P ≠ Q ∧ Q ≠ R ∧ R ≠ P)
: ∃ (n : ℕ), n = 135 := 
  sorry

end NUMINAMATH_GPT_connect_5_points_four_segments_l2170_217039


namespace NUMINAMATH_GPT_typists_retype_time_l2170_217054

theorem typists_retype_time
  (x y : ℕ)
  (h1 : (x / 2) + (y / 2) = 25)
  (h2 : 1 / x + 1 / y = 1 / 12) :
  (x = 20 ∧ y = 30) ∨ (x = 30 ∧ y = 20) :=
by
  sorry

end NUMINAMATH_GPT_typists_retype_time_l2170_217054


namespace NUMINAMATH_GPT_max_units_of_material_A_l2170_217070

theorem max_units_of_material_A (x y z : ℕ) 
    (h1 : 3 * x + 5 * y + 7 * z = 62)
    (h2 : 2 * x + 4 * y + 6 * z = 50) : x ≤ 5 :=
by
    sorry 

end NUMINAMATH_GPT_max_units_of_material_A_l2170_217070


namespace NUMINAMATH_GPT_kyle_and_miles_marbles_l2170_217044

theorem kyle_and_miles_marbles (f k m : ℕ) 
  (h1 : f = 3 * k) 
  (h2 : f = 5 * m) 
  (h3 : f = 15) : 
  k + m = 8 := 
by 
  sorry

end NUMINAMATH_GPT_kyle_and_miles_marbles_l2170_217044


namespace NUMINAMATH_GPT_find_number_l2170_217004

theorem find_number (N : ℝ) (h1 : (3 / 10) * N = 64.8) : N = 216 ∧ (1 / 3) * (1 / 4) * N = 18 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l2170_217004


namespace NUMINAMATH_GPT_total_pairs_is_11_l2170_217057

-- Definitions for the conditions
def soft_lens_price : ℕ := 150
def hard_lens_price : ℕ := 85
def total_sales_last_week : ℕ := 1455

-- Variables
variables (H S : ℕ)

-- Condition that she sold 5 more pairs of soft lenses than hard lenses
def sold_more_soft : Prop := S = H + 5

-- Equation for total sales
def total_sales_eq : Prop := (hard_lens_price * H) + (soft_lens_price * S) = total_sales_last_week

-- Total number of pairs of contact lenses sold
def total_pairs_sold : ℕ := H + S

-- The theorem to prove
theorem total_pairs_is_11 (H S : ℕ) (h1 : sold_more_soft H S) (h2 : total_sales_eq H S) : total_pairs_sold H S = 11 :=
sorry

end NUMINAMATH_GPT_total_pairs_is_11_l2170_217057


namespace NUMINAMATH_GPT_quadrilateral_area_l2170_217068

noncomputable def AreaOfQuadrilateral (AB AC AD : ℝ) : ℝ :=
  let BC := Real.sqrt (AC^2 - AB^2)
  let CD := Real.sqrt (AC^2 - AD^2)
  let AreaABC := (1 / 2) * AB * BC
  let AreaACD := (1 / 2) * AD * CD
  AreaABC + AreaACD

theorem quadrilateral_area :
  AreaOfQuadrilateral 5 13 12 = 60 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l2170_217068


namespace NUMINAMATH_GPT_new_books_count_l2170_217023

-- Defining the conditions
def num_adventure_books : ℕ := 13
def num_mystery_books : ℕ := 17
def num_used_books : ℕ := 15

-- Proving the number of new books Sam bought
theorem new_books_count : (num_adventure_books + num_mystery_books) - num_used_books = 15 :=
by
  sorry

end NUMINAMATH_GPT_new_books_count_l2170_217023


namespace NUMINAMATH_GPT_solve_system_l2170_217013

theorem solve_system : 
  ∃ x y : ℚ, (4 * x + 7 * y = -19) ∧ (4 * x - 5 * y = 17) ∧ x = 1/2 ∧ y = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2170_217013


namespace NUMINAMATH_GPT_ratio_is_one_half_l2170_217094

noncomputable def ratio_of_intercepts (b : ℝ) (hb : b ≠ 0) : ℝ :=
  let s := -b / 8
  let t := -b / 4
  s / t

theorem ratio_is_one_half (b : ℝ) (hb : b ≠ 0) :
  ratio_of_intercepts b hb = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_is_one_half_l2170_217094


namespace NUMINAMATH_GPT_probability_four_or_more_same_value_l2170_217050

theorem probability_four_or_more_same_value :
  let n := 5 -- number of dice
  let d := 10 -- number of sides on each die
  let event := "at least four of the five dice show the same value"
  let probability := (23 : ℚ) / 5000 -- given probability
  n = 5 ∧ d = 10 ∧ event = "at least four of the five dice show the same value" →
  (probability = 23 / 5000) := 
by
  intros
  sorry

end NUMINAMATH_GPT_probability_four_or_more_same_value_l2170_217050


namespace NUMINAMATH_GPT_range_of_a_l2170_217047

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 ≤ x) → ∀ y : ℝ, (1 ≤ y) → (x ≤ y) → (Real.exp (abs (x - a)) ≤ Real.exp (abs (y - a)))) : a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2170_217047


namespace NUMINAMATH_GPT_div_coeff_roots_l2170_217028

theorem div_coeff_roots :
  ∀ (a b c d e : ℝ), (∀ x, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4)
  → (d / e = -25 / 12) :=
by
  intros a b c d e h
  sorry

end NUMINAMATH_GPT_div_coeff_roots_l2170_217028


namespace NUMINAMATH_GPT_fixed_point_of_line_l2170_217051

theorem fixed_point_of_line (k : ℝ) : ∃ (p : ℝ × ℝ), p = (-3, 4) ∧ ∀ (x y : ℝ), (y - 4 = -k * (x + 3)) → (-3, 4) = (x, y) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_line_l2170_217051


namespace NUMINAMATH_GPT_both_miss_probability_l2170_217056

-- Define the probabilities of hitting the target for Persons A and B 
def prob_hit_A : ℝ := 0.85
def prob_hit_B : ℝ := 0.8

-- Calculate the probabilities of missing the target
def prob_miss_A : ℝ := 1 - prob_hit_A
def prob_miss_B : ℝ := 1 - prob_hit_B

-- Prove that the probability of both missing the target is 0.03
theorem both_miss_probability : prob_miss_A * prob_miss_B = 0.03 :=
by
  sorry

end NUMINAMATH_GPT_both_miss_probability_l2170_217056


namespace NUMINAMATH_GPT_distance_AF_l2170_217024

theorem distance_AF (A B C D E F : ℝ×ℝ)
  (h1 : A = (0, 0))
  (h2 : B = (5, 0))
  (h3 : C = (5, 5))
  (h4 : D = (0, 5))
  (h5 : E = (2.5, 5))
  (h6 : ∃ k : ℝ, F = (k, 2 * k) ∧ dist F C = 5) :
  dist A F = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_AF_l2170_217024


namespace NUMINAMATH_GPT_trip_time_maple_to_oak_l2170_217058

noncomputable def total_trip_time (d1 d2 v1 v2 t_break : ℝ) : ℝ :=
  (d1 / v1) + t_break + (d2 / v2)

theorem trip_time_maple_to_oak : 
  total_trip_time 210 210 50 40 0.5 = 5.75 :=
by
  sorry

end NUMINAMATH_GPT_trip_time_maple_to_oak_l2170_217058


namespace NUMINAMATH_GPT_intersection_A_B_l2170_217073

def A (y : ℝ) : Prop := ∃ x : ℝ, y = -x^2 + 2*x - 1
def B (y : ℝ) : Prop := ∃ x : ℝ, y = 2*x + 1

theorem intersection_A_B :
  {y : ℝ | A y} ∩ {y : ℝ | B y} = {y : ℝ | y ≤ 0} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l2170_217073


namespace NUMINAMATH_GPT_am_gm_inequality_l2170_217053

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : (1 + x) * (1 + y) * (1 + z) ≥ 8 :=
sorry

end NUMINAMATH_GPT_am_gm_inequality_l2170_217053


namespace NUMINAMATH_GPT_alice_bob_sum_is_42_l2170_217016

theorem alice_bob_sum_is_42 :
  ∃ (A B : ℕ), 
    (1 ≤ A ∧ A ≤ 60) ∧ 
    (1 ≤ B ∧ B ≤ 60) ∧ 
    Nat.Prime B ∧ B > 10 ∧ 
    (∀ n : ℕ, n < 5 → (A + B) % n ≠ 0) ∧ 
    ∃ k : ℕ, 150 * B + A = k * k ∧ 
    A + B = 42 :=
by 
  sorry

end NUMINAMATH_GPT_alice_bob_sum_is_42_l2170_217016


namespace NUMINAMATH_GPT_union_sets_M_N_l2170_217097

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | -3 < x ∧ x < 2}

-- The proof statement: the union of M and N should be x > -3
theorem union_sets_M_N : (M ∪ N) = {x | x > -3} :=
sorry

end NUMINAMATH_GPT_union_sets_M_N_l2170_217097


namespace NUMINAMATH_GPT_fraction_of_square_above_line_l2170_217034

theorem fraction_of_square_above_line :
  let A := (2, 1)
  let B := (5, 1)
  let C := (5, 4)
  let D := (2, 4)
  let P := (2, 3)
  let Q := (5, 1)
  ∃ f : ℚ, f = 2 / 3 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_fraction_of_square_above_line_l2170_217034


namespace NUMINAMATH_GPT_inequalities_hold_l2170_217032

theorem inequalities_hold (a b c x y z : ℝ) (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧
  (x * y * z ≤ a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_inequalities_hold_l2170_217032


namespace NUMINAMATH_GPT_geometric_sequence_S12_l2170_217085

theorem geometric_sequence_S12 (S : ℕ → ℝ) (S_4_eq : S 4 = 20) (S_8_eq : S 8 = 30) :
  S 12 = 35 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_S12_l2170_217085


namespace NUMINAMATH_GPT_total_amount_shared_l2170_217021

theorem total_amount_shared (z : ℝ) (hz : z = 150) (hy : y = 1.20 * z) (hx : x = 1.25 * y) : 
  x + y + z = 555 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l2170_217021


namespace NUMINAMATH_GPT_total_tickets_correct_l2170_217033

-- Define the initial number of tickets Tate has
def initial_tickets_Tate : ℕ := 32

-- Define the additional tickets Tate buys
def additional_tickets_Tate : ℕ := 2

-- Calculate the total number of tickets Tate has
def total_tickets_Tate : ℕ := initial_tickets_Tate + additional_tickets_Tate

-- Define the number of tickets Peyton has (half of Tate's total tickets)
def tickets_Peyton : ℕ := total_tickets_Tate / 2

-- Calculate the total number of tickets Tate and Peyton have together
def total_tickets_together : ℕ := total_tickets_Tate + tickets_Peyton

-- Prove that the total number of tickets together equals 51
theorem total_tickets_correct : total_tickets_together = 51 := by
  sorry

end NUMINAMATH_GPT_total_tickets_correct_l2170_217033


namespace NUMINAMATH_GPT_avg_diff_noah_liam_l2170_217049

-- Define the daily differences over 14 days
def daily_differences : List ℤ := [5, 0, 15, -5, 10, 10, -10, 5, 5, 10, -5, 15, 0, 5]

-- Define the function to calculate the average difference
def average_daily_difference (daily_diffs : List ℤ) : ℚ :=
  (daily_diffs.sum : ℚ) / daily_diffs.length

-- The proposition we want to prove
theorem avg_diff_noah_liam : average_daily_difference daily_differences = 60 / 14 := by
  sorry

end NUMINAMATH_GPT_avg_diff_noah_liam_l2170_217049


namespace NUMINAMATH_GPT_similar_triangles_perimeter_ratio_l2170_217095

theorem similar_triangles_perimeter_ratio
  (a₁ a₂ s₁ s₂ : ℝ)
  (h₁ : a₁ / a₂ = 1 / 4)
  (h₂ : s₁ / s₂ = 1 / 2) :
  (s₁ / s₂ = 1 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_similar_triangles_perimeter_ratio_l2170_217095


namespace NUMINAMATH_GPT_total_guitars_sold_l2170_217003

theorem total_guitars_sold (total_revenue : ℕ) (price_electric : ℕ) (price_acoustic : ℕ)
  (num_electric_sold : ℕ) (num_acoustic_sold : ℕ) 
  (h1 : total_revenue = 3611) (h2 : price_electric = 479) 
  (h3 : price_acoustic = 339) (h4 : num_electric_sold = 4) 
  (h5 : num_acoustic_sold * price_acoustic + num_electric_sold * price_electric = total_revenue) :
  num_electric_sold + num_acoustic_sold = 9 :=
sorry

end NUMINAMATH_GPT_total_guitars_sold_l2170_217003


namespace NUMINAMATH_GPT_ratio_john_to_total_cost_l2170_217067

noncomputable def cost_first_8_years := 8 * 10000
noncomputable def cost_next_10_years := 10 * 20000
noncomputable def university_tuition := 250000
noncomputable def cost_john_paid := 265000
noncomputable def total_cost := cost_first_8_years + cost_next_10_years + university_tuition

theorem ratio_john_to_total_cost : (cost_john_paid / total_cost : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_john_to_total_cost_l2170_217067


namespace NUMINAMATH_GPT_projectile_max_height_l2170_217082

def h (t : ℝ) : ℝ := -9 * t^2 + 36 * t + 24

theorem projectile_max_height : ∃ t : ℝ, h t = 60 := 
sorry

end NUMINAMATH_GPT_projectile_max_height_l2170_217082


namespace NUMINAMATH_GPT_boys_cannot_score_twice_l2170_217077

-- Define the total number of points in the tournament
def total_points_in_tournament : ℕ := 15

-- Define the number of boys and girls
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 4

-- Define the points scored by boys and girls
axiom points_by_boys : ℕ
axiom points_by_girls : ℕ

-- The conditions
axiom total_points_condition : points_by_boys + points_by_girls = total_points_in_tournament
axiom boys_twice_girls_condition : points_by_boys = 2 * points_by_girls

-- The statement to prove
theorem boys_cannot_score_twice : False :=
  by {
    -- Note: provide a sketch to illustrate that under the given conditions the statement is false
    sorry
  }

end NUMINAMATH_GPT_boys_cannot_score_twice_l2170_217077


namespace NUMINAMATH_GPT_volume_common_solid_hemisphere_cone_l2170_217064

noncomputable def volume_common_solid (r : ℝ) : ℝ := 
  let V_1 := (2/3) * Real.pi * (r^3 - (3 * r / 5)^3)
  let V_2 := Real.pi * ((r / 5)^2) * (r - (r / 15))
  V_1 + V_2

theorem volume_common_solid_hemisphere_cone (r : ℝ) :
  volume_common_solid r = (14 * Real.pi * r^3) / 25 := 
by
  sorry

end NUMINAMATH_GPT_volume_common_solid_hemisphere_cone_l2170_217064


namespace NUMINAMATH_GPT_jasmine_spent_l2170_217012

theorem jasmine_spent 
  (original_cost : ℝ)
  (discount : ℝ)
  (h_original : original_cost = 35)
  (h_discount : discount = 17) : 
  original_cost - discount = 18 := 
by
  sorry

end NUMINAMATH_GPT_jasmine_spent_l2170_217012


namespace NUMINAMATH_GPT_find_speed_l2170_217055

variables (x : ℝ) (V : ℝ)

def initial_speed (x : ℝ) (V : ℝ) : Prop := 
  let time_initial := x / V
  let time_second := (2 * x) / 20
  let total_distance := 3 * x
  let average_speed := 26.25
  average_speed = total_distance / (time_initial + time_second)

theorem find_speed (x : ℝ) (h : initial_speed x V) : V = 70 :=
by sorry

end NUMINAMATH_GPT_find_speed_l2170_217055


namespace NUMINAMATH_GPT_function_domain_l2170_217074

open Set

noncomputable def domain_of_function : Set ℝ :=
  {x | x ≠ 2}

theorem function_domain :
  domain_of_function = {x : ℝ | x ≠ 2} :=
by sorry

end NUMINAMATH_GPT_function_domain_l2170_217074


namespace NUMINAMATH_GPT_value_of_y_when_x_is_neg2_l2170_217045

theorem value_of_y_when_x_is_neg2 :
  ∃ (k b : ℝ), (k + b = 2) ∧ (-k + b = -4) ∧ (∀ x, y = k * x + b) ∧ (x = -2) → (y = -7) := 
sorry

end NUMINAMATH_GPT_value_of_y_when_x_is_neg2_l2170_217045


namespace NUMINAMATH_GPT_smallest_n_l2170_217061

theorem smallest_n (n : ℕ) (hn1 : ∃ k, 5 * n = k^4) (hn2: ∃ m, 4 * n = m^3) : n = 2000 :=
sorry

end NUMINAMATH_GPT_smallest_n_l2170_217061


namespace NUMINAMATH_GPT_math_problem_l2170_217017

/-- Lean translation of the mathematical problem.
Given \(a, b \in \mathbb{R}\) such that \(a^2 + b^2 = a^2 b^2\) and 
\( |a| \neq 1 \) and \( |b| \neq 1 \), prove that 
\[
\frac{a^7}{(1 - a)^2} - \frac{a^7}{(1 + a)^2} = 
\frac{b^7}{(1 - b)^2} - \frac{b^7}{(1 + b)^2}.
\]
-/
theorem math_problem 
  (a b : ℝ) 
  (h1 : a^2 + b^2 = a^2 * b^2) 
  (h2 : |a| ≠ 1) 
  (h3 : |b| ≠ 1) : 
  (a^7 / (1 - a)^2 - a^7 / (1 + a)^2) = 
  (b^7 / (1 - b)^2 - b^7 / (1 + b)^2) := 
by 
  -- Proof is omitted for this exercise.
  sorry

end NUMINAMATH_GPT_math_problem_l2170_217017


namespace NUMINAMATH_GPT_Cd_sum_l2170_217086

theorem Cd_sum : ∀ (C D : ℝ), 
  (∀ x : ℝ, x ≠ 3 → (C / (x-3) + D * (x+2) = (-2 * x^2 + 8 * x + 28) / (x-3))) → 
  (C + D = 20) :=
by
  intros C D h
  sorry

end NUMINAMATH_GPT_Cd_sum_l2170_217086


namespace NUMINAMATH_GPT_simplify_sqrt_l2170_217009

theorem simplify_sqrt (a b : ℝ) (hb : b > 0) : 
  Real.sqrt (20 * a^3 * b^2) = 2 * a * b * Real.sqrt (5 * a) :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_l2170_217009


namespace NUMINAMATH_GPT_sum_x1_x2_range_l2170_217036

variable {x₁ x₂ : ℝ}

-- Definition of x₁ being the real root of the equation x * 2^x = 1
def is_root_1 (x : ℝ) : Prop :=
  x * 2^x = 1

-- Definition of x₂ being the real root of the equation x * log_2 x = 1
def is_root_2 (x : ℝ) : Prop :=
  x * Real.log x / Real.log 2 = 1

theorem sum_x1_x2_range (hx₁ : is_root_1 x₁) (hx₂ : is_root_2 x₂) :
  2 < x₁ + x₂ :=
sorry

end NUMINAMATH_GPT_sum_x1_x2_range_l2170_217036
