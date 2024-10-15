import Mathlib

namespace NUMINAMATH_GPT_inverse_prop_l1644_164498

theorem inverse_prop (a b : ℝ) : (a > b) → (|a| > |b|) :=
sorry

end NUMINAMATH_GPT_inverse_prop_l1644_164498


namespace NUMINAMATH_GPT_quadratic_rewrite_l1644_164466

theorem quadratic_rewrite (a b c x : ℤ) :
  (16 * x^2 - 40 * x - 72 = a^2 * x^2 + 2 * a * b * x + b^2 + c) →
  (a = 4 ∨ a = -4) →
  (2 * a * b = -40) →
  ab = -20 := by
sorry

end NUMINAMATH_GPT_quadratic_rewrite_l1644_164466


namespace NUMINAMATH_GPT_partition_triangle_l1644_164493

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end NUMINAMATH_GPT_partition_triangle_l1644_164493


namespace NUMINAMATH_GPT_probability_red_ball_first_occurrence_l1644_164427

theorem probability_red_ball_first_occurrence 
  (P : ℕ → ℝ) : 
  ∃ (P1 P2 P3 P4 : ℝ),
    P 1 = 0.4 ∧ P 2 = 0.3 ∧ P 3 = 0.2 ∧ P 4 = 0.1 :=
  sorry

end NUMINAMATH_GPT_probability_red_ball_first_occurrence_l1644_164427


namespace NUMINAMATH_GPT_average_rate_first_half_80_l1644_164497

theorem average_rate_first_half_80
    (total_distance : ℝ)
    (average_rate_trip : ℝ)
    (distance_first_half : ℝ)
    (time_first_half : ℝ)
    (time_second_half : ℝ)
    (time_total : ℝ)
    (R : ℝ)
    (H1 : total_distance = 640)
    (H2 : average_rate_trip = 40)
    (H3 : distance_first_half = total_distance / 2)
    (H4 : time_first_half = distance_first_half / R)
    (H5 : time_second_half = 3 * time_first_half)
    (H6 : time_total = time_first_half + time_second_half)
    (H7 : average_rate_trip = total_distance / time_total) :
    R = 80 := 
by 
  -- Given conditions
  sorry

end NUMINAMATH_GPT_average_rate_first_half_80_l1644_164497


namespace NUMINAMATH_GPT_parabola_equation_l1644_164412

-- Definitions of the conditions
def parabola_passes_through (x y : ℝ) : Prop :=
  y^2 = -2 * (3 * x)

def focus_on_line (x y : ℝ) : Prop :=
  3 * x - 2 * y - 6 = 0

theorem parabola_equation (x y : ℝ) (hM : x = -6 ∧ y = 6) (hF : ∃ (x y : ℝ), focus_on_line x y) :
  parabola_passes_through x y = (y^2 = -6 * x) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_equation_l1644_164412


namespace NUMINAMATH_GPT_geometric_progression_first_term_and_ratio_l1644_164445

theorem geometric_progression_first_term_and_ratio (
  b_1 q : ℝ
) :
  b_1 * (1 + q + q^2) = 21 →
  b_1^2 * (1 + q^2 + q^4) = 189 →
  (b_1 = 12 ∧ q = 1/2) ∨ (b_1 = 3 ∧ q = 2) :=
by
  intros hsum hsumsq
  sorry

end NUMINAMATH_GPT_geometric_progression_first_term_and_ratio_l1644_164445


namespace NUMINAMATH_GPT_correct_propositions_l1644_164459

-- Definitions for the propositions
def prop1 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ M) → a ∧ b
def prop2 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ ¬M) → a ∧ ¬b
def prop3 (a b M : Prop) : Prop := (a ∧ b) ∧ (b ∧ M) → a ∧ M
def prop4 (a M N : Prop) : Prop := (a ∧ ¬M) ∧ (a ∧ N) → ¬M ∧ N

-- Proof problem statement
theorem correct_propositions : 
  ∀ (a b M N : Prop), 
    (prop1 a M b = true) ∨ (prop1 a M b = false) ∧ 
    (prop2 a M b = true) ∨ (prop2 a M b = false) ∧ 
    (prop3 a b M = true) ∨ (prop3 a b M = false) ∧ 
    (prop4 a M N = true) ∨ (prop4 a M N = false) → 
    3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l1644_164459


namespace NUMINAMATH_GPT_number_of_children_l1644_164460

def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 16

theorem number_of_children : total_pencils / pencils_per_child = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l1644_164460


namespace NUMINAMATH_GPT_rectangle_length_l1644_164432

theorem rectangle_length (side_square length_rectangle width_rectangle wire_length : ℝ) 
    (h1 : side_square = 12) 
    (h2 : width_rectangle = 6) 
    (h3 : wire_length = 4 * side_square) 
    (h4 : wire_length = 2 * width_rectangle + 2 * length_rectangle) : 
    length_rectangle = 18 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_length_l1644_164432


namespace NUMINAMATH_GPT_smallest_abs_sum_of_products_l1644_164496

noncomputable def g (x : ℝ) : ℝ := x^4 + 16 * x^3 + 69 * x^2 + 112 * x + 64

theorem smallest_abs_sum_of_products :
  (∀ w1 w2 w3 w4 : ℝ, g w1 = 0 ∧ g w2 = 0 ∧ g w3 = 0 ∧ g w4 = 0 → 
   |w1 * w2 + w3 * w4| ≥ 8) ∧ 
  (∃ w1 w2 w3 w4 : ℝ, g w1 = 0 ∧ g w2 = 0 ∧ g w3 = 0 ∧ g w4 = 0 ∧ 
   |w1 * w2 + w3 * w4| = 8) :=
sorry

end NUMINAMATH_GPT_smallest_abs_sum_of_products_l1644_164496


namespace NUMINAMATH_GPT_tan_theta_sqrt3_l1644_164416

theorem tan_theta_sqrt3 (θ : ℝ) 
  (h : Real.cos (40 * (π / 180) - θ) 
     + Real.cos (40 * (π / 180) + θ) 
     + Real.cos (80 * (π / 180) - θ) = 0) 
  : Real.tan θ = -Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_theta_sqrt3_l1644_164416


namespace NUMINAMATH_GPT_number_of_small_gardens_l1644_164441

def totalSeeds : ℕ := 85
def tomatoSeeds : ℕ := 42
def capsicumSeeds : ℕ := 26
def cucumberSeeds : ℕ := 17

def plantedTomatoSeeds : ℕ := 24
def plantedCucumberSeeds : ℕ := 17

def remainingTomatoSeeds : ℕ := tomatoSeeds - plantedTomatoSeeds
def remainingCapsicumSeeds : ℕ := capsicumSeeds
def remainingCucumberSeeds : ℕ := cucumberSeeds - plantedCucumberSeeds

def seedsInSmallGardenTomato : ℕ := 2
def seedsInSmallGardenCapsicum : ℕ := 1
def seedsInSmallGardenCucumber : ℕ := 1

theorem number_of_small_gardens : (remainingTomatoSeeds / seedsInSmallGardenTomato = 9) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_small_gardens_l1644_164441


namespace NUMINAMATH_GPT_xyz_plus_54_l1644_164428

theorem xyz_plus_54 (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y + z = 53) (h2 : y * z + x = 53) (h3 : z * x + y = 53) : 
  x + y + z = 54 := by
  sorry

end NUMINAMATH_GPT_xyz_plus_54_l1644_164428


namespace NUMINAMATH_GPT_followers_after_one_year_l1644_164477

theorem followers_after_one_year :
  let initial_followers := 100000
  let daily_new_followers := 1000
  let unfollowers_per_year := 20000
  let days_per_year := 365
  initial_followers + (daily_new_followers * days_per_year - unfollowers_per_year) = 445000 :=
by
  sorry

end NUMINAMATH_GPT_followers_after_one_year_l1644_164477


namespace NUMINAMATH_GPT_pine_taller_than_maple_l1644_164410

def height_maple : ℚ := 13 + 1 / 4
def height_pine : ℚ := 19 + 3 / 8

theorem pine_taller_than_maple :
  (height_pine - height_maple = 6 + 1 / 8) :=
sorry

end NUMINAMATH_GPT_pine_taller_than_maple_l1644_164410


namespace NUMINAMATH_GPT_max_valid_n_eq_3210_l1644_164450

-- Define the digit sum function S
def digit_sum (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- The condition S(3n) = 3S(n) and all digits of n are distinct
def valid_n (n : ℕ) : Prop :=
  digit_sum (3 * n) = 3 * digit_sum n ∧ (Nat.digits 10 n).Nodup

-- Prove that the maximum value of such n is 3210
theorem max_valid_n_eq_3210 : ∃ n : ℕ, valid_n n ∧ n = 3210 :=
by
  existsi 3210
  sorry

end NUMINAMATH_GPT_max_valid_n_eq_3210_l1644_164450


namespace NUMINAMATH_GPT_value_2_stddevs_less_than_mean_l1644_164491

-- Definitions based on the conditions
def mean : ℝ := 10.5
def stddev : ℝ := 1
def value := mean - 2 * stddev

-- Theorem we aim to prove
theorem value_2_stddevs_less_than_mean : value = 8.5 := by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_value_2_stddevs_less_than_mean_l1644_164491


namespace NUMINAMATH_GPT_cubic_has_one_real_root_l1644_164403

theorem cubic_has_one_real_root :
  (∃ x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0) ∧ ∀ x y : ℝ, (x^3 - 6*x^2 + 9*x - 10 = 0) ∧ (y^3 - 6*y^2 + 9*y - 10 = 0) → x = y :=
by
  sorry

end NUMINAMATH_GPT_cubic_has_one_real_root_l1644_164403


namespace NUMINAMATH_GPT_min_colors_for_distance_six_l1644_164414

/-
Definitions and conditions:
- The board is an infinite checkered paper with a cell side of one unit.
- The distance between two cells is the length of the shortest path of a rook from one cell to another.

Statement:
- Prove that the minimum number of colors needed to color the board such that two cells that are a distance of 6 apart are always painted different colors is 4.
-/

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  |c1.1 - c2.1| + |c1.2 - c2.2|

theorem min_colors_for_distance_six : ∃ (n : ℕ), (∀ (f : cell → ℕ), (∀ c1 c2, rook_distance c1 c2 = 6 → f c1 ≠ f c2) → n ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_min_colors_for_distance_six_l1644_164414


namespace NUMINAMATH_GPT_base_b_equivalence_l1644_164431

theorem base_b_equivalence (b : ℕ) (h : (2 * b + 4) ^ 2 = 5 * b ^ 2 + 5 * b + 4) : b = 12 :=
sorry

end NUMINAMATH_GPT_base_b_equivalence_l1644_164431


namespace NUMINAMATH_GPT_prob_neither_snow_nor_windy_l1644_164419

-- Define the probabilities.
def prob_snow : ℚ := 1 / 4
def prob_windy : ℚ := 1 / 3

-- Define the complementary probabilities.
def prob_not_snow : ℚ := 1 - prob_snow
def prob_not_windy : ℚ := 1 - prob_windy

-- State that the events are independent and calculate the combined probability.
theorem prob_neither_snow_nor_windy :
  prob_not_snow * prob_not_windy = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_prob_neither_snow_nor_windy_l1644_164419


namespace NUMINAMATH_GPT_x_intercept_is_neg_three_halves_l1644_164405

-- Definition of the points
def pointA : ℝ × ℝ := (-1, 1)
def pointB : ℝ × ℝ := (3, 9)

-- Statement of the theorem: The x-intercept of the line passing through the points is -3/2.
theorem x_intercept_is_neg_three_halves (A B : ℝ × ℝ)
    (hA : A = pointA)
    (hB : B = pointB) :
    ∃ x_intercept : ℝ, x_intercept = -3 / 2 := 
by
    sorry

end NUMINAMATH_GPT_x_intercept_is_neg_three_halves_l1644_164405


namespace NUMINAMATH_GPT_find_t_l1644_164443

theorem find_t (s t : ℤ) (h1 : 12 * s + 7 * t = 173) (h2 : s = t - 3) : t = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l1644_164443


namespace NUMINAMATH_GPT_complex_arithmetic_problem_l1644_164482
open Complex

theorem complex_arithmetic_problem : (2 - 3 * Complex.I) * (2 + 3 * Complex.I) + (4 - 5 * Complex.I)^2 = 4 - 40 * Complex.I := by
  sorry

end NUMINAMATH_GPT_complex_arithmetic_problem_l1644_164482


namespace NUMINAMATH_GPT_ratio_of_steps_l1644_164465

-- Defining the conditions of the problem
def andrew_steps : ℕ := 150
def jeffrey_steps : ℕ := 200

-- Stating the theorem that we need to prove
theorem ratio_of_steps : andrew_steps / Nat.gcd andrew_steps jeffrey_steps = 3 ∧ jeffrey_steps / Nat.gcd andrew_steps jeffrey_steps = 4 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_of_steps_l1644_164465


namespace NUMINAMATH_GPT_modulus_of_z_l1644_164449

-- Define the given condition
def condition (z : ℂ) : Prop := (z - 3) * (1 - 3 * Complex.I) = 10

-- State the main theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = 5 :=
sorry

end NUMINAMATH_GPT_modulus_of_z_l1644_164449


namespace NUMINAMATH_GPT_area_of_smallest_square_containing_circle_l1644_164421

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  ∃ (a : ℝ), a = 100 :=
by
  sorry

end NUMINAMATH_GPT_area_of_smallest_square_containing_circle_l1644_164421


namespace NUMINAMATH_GPT_sqrt_calc_l1644_164420

theorem sqrt_calc : Real.sqrt (Real.sqrt (0.00032 ^ (1 / 5))) = 0.669 := by
  sorry

end NUMINAMATH_GPT_sqrt_calc_l1644_164420


namespace NUMINAMATH_GPT_calculate_f_at_8_l1644_164439

def f (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 27 * x^2 - 24 * x - 72

theorem calculate_f_at_8 : f 8 = 952 :=
by sorry

end NUMINAMATH_GPT_calculate_f_at_8_l1644_164439


namespace NUMINAMATH_GPT_reciprocal_neg_sqrt_2_l1644_164437

theorem reciprocal_neg_sqrt_2 : 1 / (-Real.sqrt 2) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_sqrt_2_l1644_164437


namespace NUMINAMATH_GPT_carol_blocks_l1644_164424

theorem carol_blocks (initial_blocks : ℕ) (blocks_lost : ℕ) (final_blocks : ℕ) : 
  initial_blocks = 42 → blocks_lost = 25 → final_blocks = initial_blocks - blocks_lost → final_blocks = 17 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_carol_blocks_l1644_164424


namespace NUMINAMATH_GPT_union_of_A_and_B_intersection_of_A_and_B_l1644_164492

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 4 }
noncomputable def B : Set ℝ := { x | x > 3 ∨ x < 1 }

theorem union_of_A_and_B : A ∪ B = Set.univ :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = { x | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_intersection_of_A_and_B_l1644_164492


namespace NUMINAMATH_GPT_female_employees_count_l1644_164473

theorem female_employees_count (E Male_E Female_E M : ℕ)
  (h1: M = (2 / 5) * E)
  (h2: 200 = (E - Male_E) * (2 / 5))
  (h3: M = (2 / 5) * Male_E + 200) :
  Female_E = 500 := by
{
  sorry
}

end NUMINAMATH_GPT_female_employees_count_l1644_164473


namespace NUMINAMATH_GPT_integers_satisfying_condition_l1644_164463

-- Define the condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Define the theorem stating the proof problem
theorem integers_satisfying_condition :
  {x : ℤ | condition x} = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_integers_satisfying_condition_l1644_164463


namespace NUMINAMATH_GPT_problem1_problem2_l1644_164407

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (3 * x - y)^2 - (3 * x + 2 * y) * (3 * x - 2 * y) = 5 * y^2 - 6 * x * y :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1644_164407


namespace NUMINAMATH_GPT_sum_zero_of_absolute_inequalities_l1644_164472

theorem sum_zero_of_absolute_inequalities 
  (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) :
  a + b + c = 0 := 
  by
    sorry

end NUMINAMATH_GPT_sum_zero_of_absolute_inequalities_l1644_164472


namespace NUMINAMATH_GPT_ping_pong_shaved_head_ping_pong_upset_l1644_164485

noncomputable def probability_shaved_head (pA pB : ℚ) : ℚ :=
  pA^3 + pB^3

noncomputable def probability_upset (pB pA : ℚ) : ℚ :=
  (pB^3) + (3 * (pB^2) * pA) + (6 * (pA^2) * (pB^2))

theorem ping_pong_shaved_head :
  probability_shaved_head (2/3) (1/3) = 1/3 := 
by
  sorry

theorem ping_pong_upset :
  probability_upset (1/3) (2/3) = 11/27 := 
by
  sorry

end NUMINAMATH_GPT_ping_pong_shaved_head_ping_pong_upset_l1644_164485


namespace NUMINAMATH_GPT_harrys_age_l1644_164418

-- Definitions of the ages
variable (Kiarra Bea Job Figaro Harry : ℕ)

-- Given conditions
variable (h1 : Kiarra = 2 * Bea)
variable (h2 : Job = 3 * Bea)
variable (h3 : Figaro = Job + 7)
variable (h4 : Harry = Figaro / 2)
variable (h5 : Kiarra = 30)

-- The statement to prove
theorem harrys_age : Harry = 26 := sorry

end NUMINAMATH_GPT_harrys_age_l1644_164418


namespace NUMINAMATH_GPT_tank_capacity_correct_l1644_164453

-- Define rates and times for each pipe
def rate_a : ℕ := 200 -- in liters per minute
def rate_b : ℕ := 50 -- in liters per minute
def rate_c : ℕ := 25 -- in liters per minute

def time_a : ℕ := 1 -- pipe A open time in minutes
def time_b : ℕ := 2 -- pipe B open time in minutes
def time_c : ℕ := 2 -- pipe C open time in minutes

def cycle_time : ℕ := time_a + time_b + time_c -- total time for one cycle in minutes
def total_time : ℕ := 40 -- total time to fill the tank in minutes

-- Net water added in one cycle
def net_water_in_cycle : ℕ :=
  (rate_a * time_a) + (rate_b * time_b) - (rate_c * time_c)

-- Number of cycles needed to fill the tank
def number_of_cycles : ℕ :=
  total_time / cycle_time

-- Total capacity of the tank
def tank_capacity : ℕ :=
  number_of_cycles * net_water_in_cycle

-- The hypothesis to prove
theorem tank_capacity_correct :
  tank_capacity = 2000 :=
  by
    sorry

end NUMINAMATH_GPT_tank_capacity_correct_l1644_164453


namespace NUMINAMATH_GPT_girls_in_class_l1644_164478

theorem girls_in_class :
  ∀ (x : ℕ), (12 * 84 + 92 * x = 86 * (12 + x)) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_girls_in_class_l1644_164478


namespace NUMINAMATH_GPT_total_boat_licenses_l1644_164438

/-- A state modifies its boat license requirements to include any one of the letters A, M, or S
followed by any six digits. How many different boat licenses can now be issued? -/
theorem total_boat_licenses : 
  let letters := 3
  let digits := 10
  letters * digits^6 = 3000000 := by
  sorry

end NUMINAMATH_GPT_total_boat_licenses_l1644_164438


namespace NUMINAMATH_GPT_cos_pi_over_3_plus_2alpha_l1644_164499

variable (α : ℝ)

theorem cos_pi_over_3_plus_2alpha (h : Real.sin (π / 3 - α) = 1 / 3) :
  Real.cos (π / 3 + 2 * α) = 7 / 9 :=
  sorry

end NUMINAMATH_GPT_cos_pi_over_3_plus_2alpha_l1644_164499


namespace NUMINAMATH_GPT_square_land_perimeter_l1644_164469

theorem square_land_perimeter (a p : ℝ) (h1 : a = p^2 / 16) (h2 : 5*a = 10*p + 45) : p = 36 :=
by sorry

end NUMINAMATH_GPT_square_land_perimeter_l1644_164469


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1644_164400

theorem sufficient_but_not_necessary (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ (¬ (p ∧ q) → p ∨ q → False) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1644_164400


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l1644_164452

theorem perpendicular_line_through_point (a b c : ℝ) (hx : a = 2) (hy : b = -1) (hd : c = 3) :
  ∃ k d : ℝ, (k, d) = (-a / b, (a * 1 + b * (1 - c))) ∧ (b * -1, a * -1 + d, -a) = (1, 2, 3) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l1644_164452


namespace NUMINAMATH_GPT_white_cannot_lose_l1644_164423

-- Define a type to represent the game state
structure Game :=
  (state : Type)
  (white_move : state → state)
  (black_move : state → state)
  (initial : state)

-- Define a type to represent the double chess game conditions
structure DoubleChess extends Game :=
  (double_white_move : state → state)
  (double_black_move : state → state)

-- Define the hypothesis based on the conditions
noncomputable def white_has_no_losing_strategy (g : DoubleChess) : Prop :=
  ∃ s, g.double_white_move (g.double_white_move s) = g.initial

theorem white_cannot_lose (g : DoubleChess) :
  white_has_no_losing_strategy g :=
sorry

end NUMINAMATH_GPT_white_cannot_lose_l1644_164423


namespace NUMINAMATH_GPT_least_multiple_17_gt_500_l1644_164425

theorem least_multiple_17_gt_500 (n : ℕ) (h : (n = 17)) : ∃ m : ℤ, (m * n > 500 ∧ m * n = 510) :=
  sorry

end NUMINAMATH_GPT_least_multiple_17_gt_500_l1644_164425


namespace NUMINAMATH_GPT_length_of_bridge_l1644_164444

noncomputable def convert_speed (km_per_hour : ℝ) : ℝ := km_per_hour * (1000 / 3600)

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (passing_time : ℝ)
  (total_distance_covered : ℝ)
  (bridge_length : ℝ) :
  train_length = 120 →
  train_speed_kmh = 40 →
  passing_time = 25.2 →
  total_distance_covered = convert_speed train_speed_kmh * passing_time →
  bridge_length = total_distance_covered - train_length →
  bridge_length = 160 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1644_164444


namespace NUMINAMATH_GPT_multiplication_value_l1644_164451

theorem multiplication_value : 725143 * 999999 = 725142274857 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_value_l1644_164451


namespace NUMINAMATH_GPT_probability_sum_of_five_l1644_164470

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 4

theorem probability_sum_of_five :
  favorable_outcomes / total_outcomes = 1 / 9 := 
by
  sorry

end NUMINAMATH_GPT_probability_sum_of_five_l1644_164470


namespace NUMINAMATH_GPT_pyramid_sphere_proof_l1644_164486

theorem pyramid_sphere_proof
  (h R_1 R_2 : ℝ) 
  (O_1 O_2 T_1 T_2 : ℝ) 
  (inscription: h > 0 ∧ R_1 > 0 ∧ R_2 > 0) :
  R_1 * R_2 * h^2 = (R_1^2 - O_1 * T_1^2) * (R_2^2 - O_2 * T_2^2) :=
by
  sorry

end NUMINAMATH_GPT_pyramid_sphere_proof_l1644_164486


namespace NUMINAMATH_GPT_pure_imaginary_product_imaginary_part_fraction_l1644_164429

-- Part 1
theorem pure_imaginary_product (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i) :
  (z1 * z2).re = 0 ↔ m = 0 := 
sorry

-- Part 2
theorem imaginary_part_fraction (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i)
  (h4 : z1^2 - 2 * z1 + 2 = 0) :
  (z2 / z1).im = -1 / 2 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_product_imaginary_part_fraction_l1644_164429


namespace NUMINAMATH_GPT_max_weight_American_l1644_164442

noncomputable def max_weight_of_American_swallow (A E : ℕ) : Prop :=
A = 5 ∧ 2 * E + E = 90 ∧ 60 * A + 60 * 2 * A = 600

theorem max_weight_American (A E : ℕ) : max_weight_of_American_swallow A E :=
by
  sorry

end NUMINAMATH_GPT_max_weight_American_l1644_164442


namespace NUMINAMATH_GPT_man_double_son_age_in_2_years_l1644_164475

def present_age_son : ℕ := 25
def age_difference : ℕ := 27
def years_to_double_age : ℕ := 2

theorem man_double_son_age_in_2_years 
  (S : ℕ := present_age_son)
  (M : ℕ := S + age_difference)
  (Y : ℕ := years_to_double_age) : 
  M + Y = 2 * (S + Y) :=
by sorry

end NUMINAMATH_GPT_man_double_son_age_in_2_years_l1644_164475


namespace NUMINAMATH_GPT_populations_equal_in_years_l1644_164406

-- Definitions
def populationX (n : ℕ) : ℤ := 68000 - 1200 * n
def populationY (n : ℕ) : ℤ := 42000 + 800 * n

-- Statement to prove
theorem populations_equal_in_years : ∃ n : ℕ, populationX n = populationY n ∧ n = 13 :=
sorry

end NUMINAMATH_GPT_populations_equal_in_years_l1644_164406


namespace NUMINAMATH_GPT_probability_adjacent_points_l1644_164417

open Finset

-- Define the hexagon points and adjacency relationship
def hexagon_points : Finset ℕ := {0, 1, 2, 3, 4, 5}

def adjacent (a b : ℕ) : Prop :=
  (a = b + 1 ∨ a = b - 1 ∨ (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0))

-- Total number of ways to choose 2 points from 6 points
def total_pairs := (hexagon_points.card.choose 2)

-- Number of pairs that are adjacent
def favorable_pairs := (6 : ℕ) -- Each point has exactly 2 adjacent points, counted twice

-- The probability of selecting two adjacent points
theorem probability_adjacent_points : (favorable_pairs : ℚ) / total_pairs = 2 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_adjacent_points_l1644_164417


namespace NUMINAMATH_GPT_Eddy_travel_time_l1644_164447

theorem Eddy_travel_time :
  ∀ (T_F D_F D_E : ℕ) (S_ratio : ℝ),
    T_F = 4 →
    D_F = 360 →
    D_E = 600 →
    S_ratio = 2.2222222222222223 →
    ((D_F / T_F : ℝ) * S_ratio ≠ 0) →
    D_E / ((D_F / T_F) * S_ratio) = 3 :=
by
  intros T_F D_F D_E S_ratio ht hf hd hs hratio
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_Eddy_travel_time_l1644_164447


namespace NUMINAMATH_GPT_cubic_expression_l1644_164401

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * abc = 1027 :=
sorry

end NUMINAMATH_GPT_cubic_expression_l1644_164401


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l1644_164495

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l1644_164495


namespace NUMINAMATH_GPT_class_percentage_of_girls_l1644_164430

/-
Given:
- Initial number of boys in the class: 11
- Number of girls in the class: 13
- 1 boy is added to the class, resulting in the new total number of boys being 12

Prove:
- The percentage of the class that are girls is 52%.
-/
theorem class_percentage_of_girls (initial_boys : ℕ) (girls : ℕ) (added_boy : ℕ)
  (new_boy_total : ℕ) (total_students : ℕ) (percent_girls : ℕ) (h1 : initial_boys = 11) 
  (h2 : girls = 13) (h3 : added_boy = 1) (h4 : new_boy_total = initial_boys + added_boy) 
  (h5 : total_students = new_boy_total + girls) 
  (h6 : percent_girls = (girls * 100) / total_students) : percent_girls = 52 :=
sorry

end NUMINAMATH_GPT_class_percentage_of_girls_l1644_164430


namespace NUMINAMATH_GPT_trig_identity_l1644_164435

theorem trig_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) : Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1644_164435


namespace NUMINAMATH_GPT_sum_of_digits_div_by_11_in_consecutive_39_l1644_164402

-- Define the sum of digits function for natural numbers.
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement.
theorem sum_of_digits_div_by_11_in_consecutive_39 :
  ∀ (N : ℕ), ∃ k : ℕ, k < 39 ∧ (sum_of_digits (N + k)) % 11 = 0 :=
by sorry

end NUMINAMATH_GPT_sum_of_digits_div_by_11_in_consecutive_39_l1644_164402


namespace NUMINAMATH_GPT_min_value_of_u_l1644_164457

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : a^2 - b + 4 ≤ 0)

theorem min_value_of_u : (∃ (u : ℝ), u = (2*a + 3*b) / (a + b) ∧ u ≥ 14/5) :=
sorry

end NUMINAMATH_GPT_min_value_of_u_l1644_164457


namespace NUMINAMATH_GPT_option_b_is_factorization_l1644_164487

theorem option_b_is_factorization (m : ℝ) :
  m^2 - 1 = (m + 1) * (m - 1) :=
sorry

end NUMINAMATH_GPT_option_b_is_factorization_l1644_164487


namespace NUMINAMATH_GPT_eval_polynomial_at_neg2_l1644_164456

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

-- Statement of the problem, proving that the polynomial equals 11 when x = -2
theorem eval_polynomial_at_neg2 : polynomial (-2) = 11 := by
  sorry

end NUMINAMATH_GPT_eval_polynomial_at_neg2_l1644_164456


namespace NUMINAMATH_GPT_least_value_difference_l1644_164479

noncomputable def least_difference (x : ℝ) : ℝ := 6 - 13/5

theorem least_value_difference (x n m : ℝ) (h1 : 2*x + 5 + 4*x - 3 > x + 15)
                               (h2 : 2*x + 5 + x + 15 > 4*x - 3)
                               (h3 : 4*x - 3 + x + 15 > 2*x + 5)
                               (h4 : x + 15 > 2*x + 5)
                               (h5 : x + 15 > 4*x - 3)
                               (h_m : m = 13/5) (h_n : n = 6)
                               (hx : m < x ∧ x < n) :
  n - m = 17 / 5 :=
  by sorry

end NUMINAMATH_GPT_least_value_difference_l1644_164479


namespace NUMINAMATH_GPT_number_of_zeros_l1644_164489

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then |x| - 2 else 2 * x - 6 + Real.log x

theorem number_of_zeros :
  (∃ x : ℝ, f x = 0) ∧ (∃ y : ℝ, f y = 0) ∧ (∀ z : ℝ, f z = 0 → z = x ∨ z = y) :=
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_l1644_164489


namespace NUMINAMATH_GPT_math_problem_l1644_164434

theorem math_problem (x y : ℕ) (h1 : (x + y * I)^3 = 2 + 11 * I) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y * I = 2 + I :=
sorry

end NUMINAMATH_GPT_math_problem_l1644_164434


namespace NUMINAMATH_GPT_train_length_l1644_164483

theorem train_length (L : ℝ) (h1 : ∀ t1 : ℝ, t1 = 15 → ∀ p1 : ℝ, p1 = 180 → (L + p1) / t1 = v)
(h2 : ∀ t2 : ℝ, t2 = 20 → ∀ p2 : ℝ, p2 = 250 → (L + p2) / t2 = v) : 
L = 30 :=
by
  have h1 := h1 15 rfl 180 rfl
  have h2 := h2 20 rfl 250 rfl
  sorry

end NUMINAMATH_GPT_train_length_l1644_164483


namespace NUMINAMATH_GPT_num_solutions_eq_3_l1644_164436

theorem num_solutions_eq_3 : 
  ∃ (x1 x2 x3 : ℝ), (∀ x : ℝ, 2^x - 2 * (⌊x⌋:ℝ) - 1 = 0 → x = x1 ∨ x = x2 ∨ x = x3) 
  ∧ ¬ ∃ x4, (2^x4 - 2 * (⌊x4⌋:ℝ) - 1 = 0 ∧ x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3) :=
sorry

end NUMINAMATH_GPT_num_solutions_eq_3_l1644_164436


namespace NUMINAMATH_GPT_minimum_value_l1644_164426

noncomputable def problem_statement (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 4 * c^2

theorem minimum_value : ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27), 
  problem_statement a b c h = 180 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1644_164426


namespace NUMINAMATH_GPT_matchsticks_20th_stage_l1644_164404

theorem matchsticks_20th_stage :
  let a1 := 3
  let d := 3
  let a20 := a1 + 19 * d
  a20 = 60 := by
  sorry

end NUMINAMATH_GPT_matchsticks_20th_stage_l1644_164404


namespace NUMINAMATH_GPT_deepak_profit_share_l1644_164471

theorem deepak_profit_share (anand_investment : ℕ) (deepak_investment : ℕ) (total_profit : ℕ) 
  (h₁ : anand_investment = 22500) 
  (h₂ : deepak_investment = 35000) 
  (h₃ : total_profit = 13800) : 
  (14 * total_profit / (9 + 14)) = 8400 := 
by
  sorry

end NUMINAMATH_GPT_deepak_profit_share_l1644_164471


namespace NUMINAMATH_GPT_no_valid_pairs_of_real_numbers_l1644_164455

theorem no_valid_pairs_of_real_numbers :
  ∀ (a b : ℝ), ¬ (∃ (x y : ℤ), 3 * a * x + 7 * b * y = 3 ∧ x^2 + y^2 = 85 ∧ (x % 5 = 0 ∨ y % 5 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_pairs_of_real_numbers_l1644_164455


namespace NUMINAMATH_GPT_opposite_of_neg_six_l1644_164408

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_six_l1644_164408


namespace NUMINAMATH_GPT_installation_time_l1644_164488

-- Definitions (based on conditions)
def total_windows := 14
def installed_windows := 8
def hours_per_window := 8

-- Define what we need to prove
def remaining_windows := total_windows - installed_windows
def total_install_hours := remaining_windows * hours_per_window

theorem installation_time : total_install_hours = 48 := by
  sorry

end NUMINAMATH_GPT_installation_time_l1644_164488


namespace NUMINAMATH_GPT_smallest_integer_is_10_l1644_164481

noncomputable def smallest_integer (a b c : ℕ) : ℕ :=
  if h : (a + b + c = 90) ∧ (2 * b = 3 * a) ∧ (5 * a = 2 * c)
  then a
  else 0

theorem smallest_integer_is_10 (a b c : ℕ) (h₁ : a + b + c = 90) (h₂ : 2 * b = 3 * a) (h₃ : 5 * a = 2 * c) : 
  smallest_integer a b c = 10 :=
sorry

end NUMINAMATH_GPT_smallest_integer_is_10_l1644_164481


namespace NUMINAMATH_GPT_difference_between_numbers_l1644_164490

theorem difference_between_numbers (x y d : ℝ) (h1 : x + y = 10) (h2 : x - y = d) (h3 : x^2 - y^2 = 80) : d = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_difference_between_numbers_l1644_164490


namespace NUMINAMATH_GPT_juice_water_ratio_l1644_164411

theorem juice_water_ratio (V : ℝ) :
  let glass_juice_ratio := (2, 1)
  let mug_volume := 2 * V
  let mug_juice_ratio := (4, 1)
  let glass_juice_vol := (2 / 3) * V
  let glass_water_vol := (1 / 3) * V
  let mug_juice_vol := (8 / 5) * V
  let mug_water_vol := (2 / 5) * V
  let total_juice := glass_juice_vol + mug_juice_vol
  let total_water := glass_water_vol + mug_water_vol
  let ratio := total_juice / total_water
  ratio = 34 / 11 :=
by
  sorry

end NUMINAMATH_GPT_juice_water_ratio_l1644_164411


namespace NUMINAMATH_GPT_sum_of_dimensions_eq_18_sqrt_1_5_l1644_164422

theorem sum_of_dimensions_eq_18_sqrt_1_5 (P Q R : ℝ) (h1 : P * Q = 30) (h2 : P * R = 50) (h3 : Q * R = 90) :
  P + Q + R = 18 * Real.sqrt 1.5 :=
sorry

end NUMINAMATH_GPT_sum_of_dimensions_eq_18_sqrt_1_5_l1644_164422


namespace NUMINAMATH_GPT_sum_of_squares_l1644_164409

theorem sum_of_squares (n m : ℕ) (h : 2 * m = n^2 + 1) : ∃ k : ℕ, m = k^2 + (k - 1)^2 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l1644_164409


namespace NUMINAMATH_GPT_find_f_cos_10_l1644_164494

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) : f (Real.sin x) = Real.cos (3 * x)

theorem find_f_cos_10 : f (Real.cos (10 * Real.pi / 180)) = -1/2 := by
  sorry

end NUMINAMATH_GPT_find_f_cos_10_l1644_164494


namespace NUMINAMATH_GPT_inverse_function_value_l1644_164476

def g (x : ℝ) : ℝ := 4 * x ^ 3 - 5

theorem inverse_function_value (x : ℝ) : g x = -1 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_value_l1644_164476


namespace NUMINAMATH_GPT_percentage_of_copper_first_alloy_l1644_164448

theorem percentage_of_copper_first_alloy :
  ∃ x : ℝ, 
  (66 * x / 100) + (55 * 21 / 100) = 121 * 15 / 100 ∧
  x = 10 := 
sorry

end NUMINAMATH_GPT_percentage_of_copper_first_alloy_l1644_164448


namespace NUMINAMATH_GPT_xy_equation_result_l1644_164446

theorem xy_equation_result (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -5) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -10.528 :=
by
  sorry

end NUMINAMATH_GPT_xy_equation_result_l1644_164446


namespace NUMINAMATH_GPT_functional_equation_solution_l1644_164454

noncomputable def quadratic_polynomial (P : ℝ → ℝ) :=
  ∃ a b c : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x + c

theorem functional_equation_solution (P : ℝ → ℝ) (f : ℝ → ℝ)
  (h_poly : quadratic_polynomial P)
  (h_additive : ∀ x y : ℝ, f (x + y) = f x + f y)
  (h_preserves_poly : ∀ x : ℝ, f (P x) = f x) :
  ∀ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1644_164454


namespace NUMINAMATH_GPT_trajectory_midpoint_l1644_164415

-- Defining the point A(-2, 0)
def A : ℝ × ℝ := (-2, 0)

-- Defining the curve equation
def curve (x y : ℝ) : Prop := 2 * y^2 = x

-- Coordinates of P based on the midpoint formula
def P (x y : ℝ) : ℝ × ℝ := (2 * x + 2, 2 * y)

-- The target trajectory equation
def trajectory_eqn (x y : ℝ) : Prop := x = 4 * y^2 - 1

-- The theorem to be proved
theorem trajectory_midpoint (x y : ℝ) :
  curve (2 * y) (2 * x + 2) → 
  trajectory_eqn x y :=
sorry

end NUMINAMATH_GPT_trajectory_midpoint_l1644_164415


namespace NUMINAMATH_GPT_parallelogram_height_l1644_164467

variable (base height area : ℝ)
variable (h_eq_diag : base = 30)
variable (h_eq_area : area = 600)

theorem parallelogram_height :
  (height = 20) ↔ (base * height = area) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_height_l1644_164467


namespace NUMINAMATH_GPT_pebbles_game_invariant_l1644_164440

/-- 
The game of pebbles is played on an infinite board of lattice points (i, j).
Initially, there is a pebble at (0, 0).
A move consists of removing a pebble from point (i, j) and placing a pebble at each of the points (i+1, j) and (i, j+1) provided both are vacant.
Show that at any stage of the game there is a pebble at some lattice point (a, b) with 0 ≤ a + b ≤ 3. 
-/
theorem pebbles_game_invariant :
  ∀ (board : ℕ × ℕ → Prop) (initial_state : board (0, 0)) (move : (ℕ × ℕ) → Prop → Prop → Prop),
  (∀ (i j : ℕ), board (i, j) → ¬ board (i+1, j) ∧ ¬ board (i, j+1) → board (i+1, j) ∧ board (i, j+1)) →
  ∃ (a b : ℕ), (0 ≤ a + b ∧ a + b ≤ 3) ∧ board (a, b) :=
by
  intros board initial_state move move_rule
  sorry 

end NUMINAMATH_GPT_pebbles_game_invariant_l1644_164440


namespace NUMINAMATH_GPT_find_g_inv_f_8_l1644_164484

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g : ∀ x : ℝ, f_inv (g x) = x^2 - x
axiom g_bijective : Function.Bijective g

theorem find_g_inv_f_8 : g_inv (f 8) = (1 + Real.sqrt 33) / 2 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_find_g_inv_f_8_l1644_164484


namespace NUMINAMATH_GPT_product_abc_l1644_164464

theorem product_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eqn : a * b * c = a * b^3) (h_c_eq_1 : c = 1) :
  a * b * c = a :=
by
  sorry

end NUMINAMATH_GPT_product_abc_l1644_164464


namespace NUMINAMATH_GPT_vanessa_deleted_30_files_l1644_164468

-- Define the initial conditions
def original_files : Nat := 16 + 48
def files_left : Nat := 34

-- Define the number of files deleted
def files_deleted : Nat := original_files - files_left

-- The theorem to prove the number of files deleted
theorem vanessa_deleted_30_files : files_deleted = 30 := by
  sorry

end NUMINAMATH_GPT_vanessa_deleted_30_files_l1644_164468


namespace NUMINAMATH_GPT_number_of_students_in_class_l1644_164462

theorem number_of_students_in_class :
  ∃ a : ℤ, 100 ≤ a ∧ a ≤ 200 ∧ a % 4 = 1 ∧ a % 3 = 2 ∧ a % 7 = 3 ∧ a = 101 := 
sorry

end NUMINAMATH_GPT_number_of_students_in_class_l1644_164462


namespace NUMINAMATH_GPT_Buratino_can_solve_l1644_164433

theorem Buratino_can_solve :
  ∃ (MA TE TI KA : ℕ), MA ≠ TE ∧ MA ≠ TI ∧ MA ≠ KA ∧ TE ≠ TI ∧ TE ≠ KA ∧ TI ≠ KA ∧
  MA * TE * MA * TI * KA = 2016000 :=
by
  -- skip the proof using sorry
  sorry

end NUMINAMATH_GPT_Buratino_can_solve_l1644_164433


namespace NUMINAMATH_GPT_total_number_of_animals_l1644_164461

-- Definitions for the animal types
def heads_per_hen := 2
def legs_per_hen := 8
def heads_per_peacock := 3
def legs_per_peacock := 9
def heads_per_zombie_hen := 6
def legs_per_zombie_hen := 12

-- Given total heads and legs
def total_heads := 800
def total_legs := 2018

-- Proof that the total number of animals is 203
theorem total_number_of_animals : 
  ∀ (H P Z : ℕ), 
    heads_per_hen * H + heads_per_peacock * P + heads_per_zombie_hen * Z = total_heads
    ∧ legs_per_hen * H + legs_per_peacock * P + legs_per_zombie_hen * Z = total_legs 
    → H + P + Z = 203 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_animals_l1644_164461


namespace NUMINAMATH_GPT_coin_problem_l1644_164480

variable (x y S k : ℕ)

theorem coin_problem
  (h1 : x + y = 14)
  (h2 : 2 * x + 5 * y = S)
  (h3 : S = k + 2 * k)
  (h4 : k * 4 = S) :
  y = 4 ∨ y = 8 ∨ y = 12 :=
by
  sorry

end NUMINAMATH_GPT_coin_problem_l1644_164480


namespace NUMINAMATH_GPT_father_son_speed_ratio_l1644_164458

theorem father_son_speed_ratio
  (F S t : ℝ)
  (distance_hallway : ℝ)
  (distance_meet_from_father : ℝ)
  (H1 : distance_hallway = 16)
  (H2 : distance_meet_from_father = 12)
  (H3 : 12 = F * t)
  (H4 : 4 = S * t)
  : F / S = 3 := by
  sorry

end NUMINAMATH_GPT_father_son_speed_ratio_l1644_164458


namespace NUMINAMATH_GPT_zeros_of_f_l1644_164413

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x ^ 2 - 2 * x - 3)

theorem zeros_of_f :
  { x : ℝ | f x = 0 } = {1, -1, 3} :=
sorry

end NUMINAMATH_GPT_zeros_of_f_l1644_164413


namespace NUMINAMATH_GPT_four_digit_greater_than_three_digit_l1644_164474

theorem four_digit_greater_than_three_digit (n m : ℕ) (h₁ : 1000 ≤ n ∧ n ≤ 9999) (h₂ : 100 ≤ m ∧ m ≤ 999) : n > m :=
sorry

end NUMINAMATH_GPT_four_digit_greater_than_three_digit_l1644_164474
