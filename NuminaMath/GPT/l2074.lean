import Mathlib

namespace NUMINAMATH_GPT_no_valid_2011_matrix_l2074_207450

def valid_matrix (A : ℕ → ℕ → ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 2011 →
    (∀ k, 1 ≤ k ∧ k ≤ 4021 →
      (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A i j = k) ∨ (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A j i = k))

theorem no_valid_2011_matrix :
  ¬ ∃ A : ℕ → ℕ → ℕ, (∀ i j, 1 ≤ i ∧ i ≤ 2011 ∧ 1 ≤ j ∧ j ≤ 2011 → 1 ≤ A i j ∧ A i j ≤ 4021) ∧ valid_matrix A :=
by
  sorry

end NUMINAMATH_GPT_no_valid_2011_matrix_l2074_207450


namespace NUMINAMATH_GPT_product_gcd_lcm_is_correct_l2074_207490

-- Define the numbers
def a := 15
def b := 75

-- Definitions related to GCD and LCM
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b
def product_gcd_lcm := gcd_ab * lcm_ab

-- Theorem stating the product of GCD and LCM of a and b is 1125
theorem product_gcd_lcm_is_correct : product_gcd_lcm = 1125 := by
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_is_correct_l2074_207490


namespace NUMINAMATH_GPT_sample_size_l2074_207433

theorem sample_size (F n : ℕ) (FR : ℚ) (h1: F = 36) (h2: FR = 1/4) (h3: FR = F / n) : n = 144 :=
by 
  sorry

end NUMINAMATH_GPT_sample_size_l2074_207433


namespace NUMINAMATH_GPT_swimming_speed_in_still_water_l2074_207444

variable (v : ℝ) -- the person's swimming speed in still water

-- Conditions
variable (water_speed : ℝ := 4) -- speed of the water
variable (time : ℝ := 2) -- time taken to swim 12 km against the current
variable (distance : ℝ := 12) -- distance swam against the current

theorem swimming_speed_in_still_water :
  (v - water_speed) = distance / time → v = 10 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_in_still_water_l2074_207444


namespace NUMINAMATH_GPT_rectangle_equation_l2074_207473

-- Given points in the problem, we define the coordinates
def A : ℝ × ℝ := (5, 5)
def B : ℝ × ℝ := (9, 2)
def C (a : ℝ) : ℝ × ℝ := (a, 13)
def D (b : ℝ) : ℝ × ℝ := (15, b)

-- We need to prove that a - b = 1 given the conditions
theorem rectangle_equation (a b : ℝ) (h1 : C a = (a, 13)) (h2 : D b = (15, b)) (h3 : 15 - a = 4) (h4 : 13 - b = 3) : 
     a - b = 1 := 
sorry

end NUMINAMATH_GPT_rectangle_equation_l2074_207473


namespace NUMINAMATH_GPT_cube_volume_in_pyramid_l2074_207468

noncomputable def pyramid_base_side : ℝ := 2
noncomputable def equilateral_triangle_side : ℝ := 2 * Real.sqrt 2
noncomputable def equilateral_triangle_height : ℝ := Real.sqrt 6
noncomputable def cube_side : ℝ := Real.sqrt 6 / 2
noncomputable def cube_volume : ℝ := (Real.sqrt 6 / 2) ^ 3

theorem cube_volume_in_pyramid : cube_volume = 3 * Real.sqrt 6 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_in_pyramid_l2074_207468


namespace NUMINAMATH_GPT_binomial_cubes_sum_l2074_207492

theorem binomial_cubes_sum (x y : ℤ) :
  let B1 := x^4 + 9 * x * y^3
  let B2 := -(3 * x^3 * y) - 9 * y^4
  (B1 ^ 3 + B2 ^ 3 = x ^ 12 - 729 * y ^ 12) := by
  sorry

end NUMINAMATH_GPT_binomial_cubes_sum_l2074_207492


namespace NUMINAMATH_GPT_simplify_expression_l2074_207484

theorem simplify_expression :
  let a := (1/2)^2
  let b := (1/2)^3
  let c := (1/2)^4
  let d := (1/2)^5
  1 / (1/a + 1/b + 1/c + 1/d) = 1/60 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2074_207484


namespace NUMINAMATH_GPT_exists_positive_int_solutions_l2074_207431

theorem exists_positive_int_solutions (a : ℕ) (ha : a > 2) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = a^2 :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_int_solutions_l2074_207431


namespace NUMINAMATH_GPT_odd_function_sin_cos_product_l2074_207458

-- Prove that if the function f(x) = sin(x + α) - 2cos(x - α) is an odd function, then sin(α) * cos(α) = 2/5
theorem odd_function_sin_cos_product (α : ℝ)
  (hf : ∀ x, Real.sin (x + α) - 2 * Real.cos (x - α) = -(Real.sin (-x + α) - 2 * Real.cos (-x - α))) :
  Real.sin α * Real.cos α = 2 / 5 :=
  sorry

end NUMINAMATH_GPT_odd_function_sin_cos_product_l2074_207458


namespace NUMINAMATH_GPT_shorten_other_side_area_l2074_207485

-- Assuming initial dimensions and given conditions
variable (length1 length2 : ℕ)
variable (new_length : ℕ)
variable (area1 area2 : ℕ)

-- Initial dimensions of the index card
def initial_dimensions (length1 length2 : ℕ) : Prop :=
  length1 = 3 ∧ length2 = 7

-- Area when one side is shortened to a specific new length
def shortened_area (length1 length2 new_length : ℕ) : ℕ :=
  if new_length = length1 - 1 then new_length * length2 else length1 * (length2 - 1)

-- Condition that the area is 15 square inches when one side is shortened
def condition_area_15 (length1 length2 : ℕ) : Prop :=
  (shortened_area length1 length2 (length1 - 1) = 15 ∨
   shortened_area length1 length2 (length2 - 1) = 15)

-- Area when the other side is shortened by 1 inch
def new_area (length1 new_length : ℕ) : ℕ :=
  new_length * (length1 - 1)

-- Proving the final area when the other side is shortened
theorem shorten_other_side_area :
  initial_dimensions length1 length2 →
  condition_area_15 length1 length2 →
  new_area length2 (length2 - 1) = 10 :=
by
  intros hdim hc15
  have hlength1 : length1 = 3 := hdim.1
  have hlength2 : length2 = 7 := hdim.2
  sorry

end NUMINAMATH_GPT_shorten_other_side_area_l2074_207485


namespace NUMINAMATH_GPT_volume_of_cylinder_in_pyramid_l2074_207460

theorem volume_of_cylinder_in_pyramid
  (a α : ℝ)
  (sin_alpha : ℝ := Real.sin α)
  (tan_alpha : ℝ := Real.tan α)
  (sin_pi_four_alpha : ℝ := Real.sin (Real.pi / 4 + α))
  (sqrt_two : ℝ := Real.sqrt 2) :
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3) / (128 * sin_pi_four_alpha^3) =
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3 / (128 * sin_pi_four_alpha^3)) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cylinder_in_pyramid_l2074_207460


namespace NUMINAMATH_GPT_smallest_term_4_in_c_seq_l2074_207464

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

noncomputable def b_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n - 1) + 15

noncomputable def c_seq (n : ℕ) : ℚ :=
  if n = 0 then 0 else (b_seq n) / (a_seq n)

theorem smallest_term_4_in_c_seq : 
  ∀ n : ℕ, n > 0 → c_seq 4 ≤ c_seq n :=
sorry

end NUMINAMATH_GPT_smallest_term_4_in_c_seq_l2074_207464


namespace NUMINAMATH_GPT_derivative_f_l2074_207486

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem derivative_f (x : ℝ) (hx : x ≠ 0) :
  deriv f x = 1 - (1 / (x ^ 2)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_derivative_f_l2074_207486


namespace NUMINAMATH_GPT_find_nat_number_l2074_207437

theorem find_nat_number (N : ℕ) (d : ℕ) (hd : d < 10) (h : N = 5 * d + d) : N = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_nat_number_l2074_207437


namespace NUMINAMATH_GPT_find_petra_age_l2074_207422

namespace MathProof
  -- Definitions of the given conditions
  variables (P M : ℕ)
  axiom sum_of_ages : P + M = 47
  axiom mother_age_relation : M = 2 * P + 14
  axiom mother_actual_age : M = 36

  -- The proof goal which we need to fill later
  theorem find_petra_age : P = 11 :=
  by
    -- Using the axioms we have
    sorry -- Proof steps, which you don't need to fill according to the instructions
end MathProof

end NUMINAMATH_GPT_find_petra_age_l2074_207422


namespace NUMINAMATH_GPT_find_a_plus_b_l2074_207446

variable (a : ℝ) (b : ℝ)
def op (x y : ℝ) : ℝ := x + 2 * y + 3

theorem find_a_plus_b (a b : ℝ) (h1 : op (op (a^3) (a^2)) a = b)
    (h2 : op (a^3) (op (a^2) a) = b) : a + b = 21/8 :=
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l2074_207446


namespace NUMINAMATH_GPT_find_length_of_brick_l2074_207475

-- Definitions given in the problem
def w : ℕ := 4
def h : ℕ := 2
def SA : ℕ := 112
def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

-- Lean 4 statement for the proof problem
theorem find_length_of_brick (l : ℕ) (h w SA : ℕ) (h_w : w = 4) (h_h : h = 2) (h_SA : SA = 112) :
  surface_area l w h = SA → l = 8 := by
  intros H
  simp [surface_area, h_w, h_h, h_SA] at H
  sorry

end NUMINAMATH_GPT_find_length_of_brick_l2074_207475


namespace NUMINAMATH_GPT_domain_shift_l2074_207443

noncomputable def domain := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }
noncomputable def shifted_domain := { x : ℝ | 2 ≤ x ∧ x ≤ 5 }

theorem domain_shift (f : ℝ → ℝ) (h : ∀ x, x ∈ domain ↔ (1 ≤ x ∧ x ≤ 4)) :
  ∀ x, x ∈ shifted_domain ↔ ∃ y, (y = x - 1) ∧ y ∈ domain :=
by
  sorry

end NUMINAMATH_GPT_domain_shift_l2074_207443


namespace NUMINAMATH_GPT_find_a_l2074_207497

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_a (h1 : a ≠ 0) (h2 : f a b c (-1) = 0)
    (h3 : ∀ x : ℝ, x ≤ f a b c x ∧ f a b c x ≤ (1/2) * (x^2 + 1)) :
  a = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2074_207497


namespace NUMINAMATH_GPT_sum_of_any_three_on_line_is_30_l2074_207476

/-- Define the list of numbers from 1 to 19 -/
def numbers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- Define the specific sequence found in the solution -/
def arrangement :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 18,
   17, 16, 15, 14, 13, 12, 11]

/-- Define the function to compute the sum of any three numbers on a straight line -/
def sum_on_line (a b c : ℕ) := a + b + c

theorem sum_of_any_three_on_line_is_30 :
  ∀ i j k : ℕ, 
  i ∈ numbers ∧ j ∈ numbers ∧ k ∈ numbers ∧ (i = 10 ∨ j = 10 ∨ k = 10) →
  sum_on_line i j k = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_any_three_on_line_is_30_l2074_207476


namespace NUMINAMATH_GPT_molly_age_l2074_207416

theorem molly_age : 14 + 6 = 20 := by
  sorry

end NUMINAMATH_GPT_molly_age_l2074_207416


namespace NUMINAMATH_GPT_my_and_mothers_ages_l2074_207436

-- Definitions based on conditions
noncomputable def my_age (x : ℕ) := x
noncomputable def mothers_age (x : ℕ) := 3 * x
noncomputable def sum_of_ages (x : ℕ) := my_age x + mothers_age x

-- Proposition that needs to be proved
theorem my_and_mothers_ages (x : ℕ) (h : sum_of_ages x = 40) :
  my_age x = 10 ∧ mothers_age x = 30 :=
by
  sorry

end NUMINAMATH_GPT_my_and_mothers_ages_l2074_207436


namespace NUMINAMATH_GPT_coordinates_P_wrt_origin_l2074_207482

/-- Define a point P with coordinates we are given. -/
def P : ℝ × ℝ := (-1, 2)

/-- State that the coordinates of P with respect to the origin O are (-1, 2). -/
theorem coordinates_P_wrt_origin : P = (-1, 2) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_coordinates_P_wrt_origin_l2074_207482


namespace NUMINAMATH_GPT_line_intersects_x_axis_between_A_and_B_l2074_207474

theorem line_intersects_x_axis_between_A_and_B (a : ℝ) :
  (∀ x, (x = 1 ∨ x = 3) → (2 * x + (3 - a) = 0)) ↔ 5 ≤ a ∧ a ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_between_A_and_B_l2074_207474


namespace NUMINAMATH_GPT_number_of_dimes_l2074_207425

theorem number_of_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 200) : d = 14 := 
sorry

end NUMINAMATH_GPT_number_of_dimes_l2074_207425


namespace NUMINAMATH_GPT_total_duration_of_running_l2074_207400

-- Definition of conditions
def constant_speed_1 : ℝ := 18
def constant_time_1 : ℝ := 3
def next_distance : ℝ := 70
def average_speed_2 : ℝ := 14

-- Proof statement
theorem total_duration_of_running : 
    let distance_1 := constant_speed_1 * constant_time_1
    let time_2 := next_distance / average_speed_2
    distance_1 = 54 ∧ time_2 = 5 → (constant_time_1 + time_2 = 8) :=
sorry

end NUMINAMATH_GPT_total_duration_of_running_l2074_207400


namespace NUMINAMATH_GPT_find_function_l2074_207442

/-- Any function f : ℝ → ℝ satisfying the two given conditions must be of the form f(x) = cx where |c| ≤ 1. -/
theorem find_function (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, x ≠ 0 → x * (f (x + 1) - f x) = f x)
  (h2 : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ c : ℝ, (∀ x : ℝ, f x = c * x) ∧ |c| ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_find_function_l2074_207442


namespace NUMINAMATH_GPT_smallest_integer_consecutive_set_l2074_207487

theorem smallest_integer_consecutive_set 
(n : ℤ) (h : 7 * n + 21 > 4 * n) : n > -7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_consecutive_set_l2074_207487


namespace NUMINAMATH_GPT_waste_in_scientific_notation_l2074_207478

def water_waste_per_person : ℝ := 0.32
def number_of_people : ℝ := 10^6

def total_daily_waste : ℝ := water_waste_per_person * number_of_people

def scientific_notation (x : ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

theorem waste_in_scientific_notation :
  scientific_notation total_daily_waste ∧ total_daily_waste = 3.2 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_waste_in_scientific_notation_l2074_207478


namespace NUMINAMATH_GPT_cards_probability_l2074_207453

-- Definitions based on conditions
def total_cards := 52
def suits := 4
def cards_per_suit := 13

-- Introducing probabilities for the conditions mentioned
def prob_first := 1
def prob_second := 39 / 52
def prob_third := 26 / 52
def prob_fourth := 13 / 52
def prob_fifth := 26 / 52

-- The problem statement
theorem cards_probability :
  (prob_first * prob_second * prob_third * prob_fourth * prob_fifth) = (3 / 64) :=
by
  sorry

end NUMINAMATH_GPT_cards_probability_l2074_207453


namespace NUMINAMATH_GPT_N2O3_weight_l2074_207445

-- Definitions from the conditions
def molecularWeightN : Float := 14.01
def molecularWeightO : Float := 16.00
def molecularWeightN2O3 : Float := (2 * molecularWeightN) + (3 * molecularWeightO)
def moles : Float := 4

-- The main proof problem statement
theorem N2O3_weight (h1 : molecularWeightN = 14.01)
                    (h2 : molecularWeightO = 16.00)
                    (h3 : molecularWeightN2O3 = (2 * molecularWeightN) + (3 * molecularWeightO))
                    (h4 : moles = 4) :
                    (moles * molecularWeightN2O3) = 304.08 :=
by
  sorry

end NUMINAMATH_GPT_N2O3_weight_l2074_207445


namespace NUMINAMATH_GPT_sum_of_first_2n_terms_l2074_207412

-- Definitions based on conditions
variable (n : ℕ) (S : ℕ → ℝ)

-- Conditions
def condition1 : Prop := S n = 24
def condition2 : Prop := S (3 * n) = 42

-- Statement to be proved
theorem sum_of_first_2n_terms {n : ℕ} (S : ℕ → ℝ) 
    (h1 : S n = 24) (h2 : S (3 * n) = 42) : S (2 * n) = 36 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_2n_terms_l2074_207412


namespace NUMINAMATH_GPT_part1_part2_part3_l2074_207494

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

theorem part1 (x : ℝ) (hx : 0 < x) : f 0 x < x := by sorry

theorem part2 (a x : ℝ) :
  (0 ≤ a ∧ a ≤ 8/9 → 0 = 0) ∧
  (a > 8/9 → 2 = 2) ∧
  (a < 0 → 1 = 1) := by sorry

theorem part3 (a : ℝ) (h : ∀ x > 0, f a x ≥ 0) : 0 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_GPT_part1_part2_part3_l2074_207494


namespace NUMINAMATH_GPT_aaron_ends_up_with_24_cards_l2074_207496

def initial_cards_aaron : Nat := 5
def found_cards_aaron : Nat := 62
def lost_cards_aaron : Nat := 15
def given_cards_to_arthur : Nat := 28

def final_cards_aaron (initial: Nat) (found: Nat) (lost: Nat) (given: Nat) : Nat :=
  initial + found - lost - given

theorem aaron_ends_up_with_24_cards :
  final_cards_aaron initial_cards_aaron found_cards_aaron lost_cards_aaron given_cards_to_arthur = 24 := by
  sorry

end NUMINAMATH_GPT_aaron_ends_up_with_24_cards_l2074_207496


namespace NUMINAMATH_GPT_find_q_l2074_207459

noncomputable def Sn (n : ℕ) (d : ℚ) : ℚ :=
  d^2 * (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def Tn (n : ℕ) (d : ℚ) (q : ℚ) : ℚ :=
  d^2 * (1 - q^n) / (1 - q)

theorem find_q (d : ℚ) (q : ℚ) (hd : d ≠ 0) (hq : 0 < q ∧ q < 1) :
  Sn 3 d / Tn 3 d q = 14 → q = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l2074_207459


namespace NUMINAMATH_GPT_find_n_l2074_207435

-- Defining the conditions.
def condition_one : Prop :=
  ∀ (c d : ℕ), 
  (80 * 2 * c = 320) ∧ (80 * 2 * d = 160)

def condition_two : Prop :=
  ∀ (c d : ℕ), 
  (100 * 3 * c = 450) ∧ (100 * 3 * d = 300)

def condition_three (n : ℕ) : Prop :=
  ∀ (c d : ℕ), 
  (40 * 4 * c = n) ∧ (40 * 4 * d = 160)

-- Statement of the proof problem using the conditions.
theorem find_n : 
  condition_one ∧ condition_two ∧ condition_three 160 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2074_207435


namespace NUMINAMATH_GPT_sin_of_angle_in_first_quadrant_l2074_207404

theorem sin_of_angle_in_first_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 3 / 4) : Real.sin α = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_of_angle_in_first_quadrant_l2074_207404


namespace NUMINAMATH_GPT_find_c_l2074_207477

open Real

def vector := (ℝ × ℝ)

def a : vector := (1, 2)
def b : vector := (2, -3)

def is_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_c (c : vector) : 
  (is_parallel (c.1 + a.1, c.2 + a.2) b) ∧ (is_perpendicular c (a.1 + b.1, a.2 + b.2)) → 
  c = (-7 / 9, -20 / 9) := 
by
  sorry

end NUMINAMATH_GPT_find_c_l2074_207477


namespace NUMINAMATH_GPT_marble_distribution_l2074_207457

theorem marble_distribution (x : ℚ) (total : ℚ) (boy1 : ℚ) (boy2 : ℚ) (boy3 : ℚ) :
  (4 * x + 2) + (2 * x + 1) + (3 * x) = total → total = 62 →
  boy1 = 4 * x + 2 → boy2 = 2 * x + 1 → boy3 = 3 * x →
  boy1 = 254 / 9 ∧ boy2 = 127 / 9 ∧ boy3 = 177 / 9 :=
by
  sorry

end NUMINAMATH_GPT_marble_distribution_l2074_207457


namespace NUMINAMATH_GPT_correct_conclusion_l2074_207439

theorem correct_conclusion :
  ¬ (-(-3)^2 = 9) ∧
  ¬ (-6 / 6 * (1 / 6) = -6) ∧
  ((-3)^2 * abs (-1/3) = 3) ∧
  ¬ (3^2 / 2 = 9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_correct_conclusion_l2074_207439


namespace NUMINAMATH_GPT_alvin_earns_14_dollars_l2074_207427

noncomputable def total_earnings (total_marbles : ℕ) (percent_white percent_black : ℚ)
  (price_white price_black price_colored : ℚ) : ℚ :=
  let white_marbles := percent_white * total_marbles
  let black_marbles := percent_black * total_marbles
  let colored_marbles := total_marbles - white_marbles - black_marbles
  (white_marbles * price_white) + (black_marbles * price_black) + (colored_marbles * price_colored)

theorem alvin_earns_14_dollars :
  total_earnings 100 (20/100) (30/100) 0.05 0.10 0.20 = 14 := by
  sorry

end NUMINAMATH_GPT_alvin_earns_14_dollars_l2074_207427


namespace NUMINAMATH_GPT_find_smaller_number_l2074_207441

theorem find_smaller_number (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h_ratio : u / v = 3 / 5) (h_sum : u + v = 16) : u = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l2074_207441


namespace NUMINAMATH_GPT_expand_product_l2074_207467

theorem expand_product (y : ℝ) (h : y ≠ 0) : 
  (3 / 7) * ((7 / y) - 14 * y^3 + 21) = (3 / y) - 6 * y^3 + 9 := 
by 
  sorry

end NUMINAMATH_GPT_expand_product_l2074_207467


namespace NUMINAMATH_GPT_find_r_k_l2074_207420

theorem find_r_k :
  ∃ r k : ℚ, (∀ t : ℚ, (∃ x y : ℚ, (x = r + 3 * t ∧ y = 2 + k * t) ∧ y = 5 * x - 7)) ∧ 
            r = 9 / 5 ∧ k = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_r_k_l2074_207420


namespace NUMINAMATH_GPT_work_completion_days_l2074_207491

theorem work_completion_days (D : ℕ) 
  (h : 40 * D = 48 * (D - 10)) : D = 60 := 
sorry

end NUMINAMATH_GPT_work_completion_days_l2074_207491


namespace NUMINAMATH_GPT_total_money_left_l2074_207454

theorem total_money_left (david_start john_start emily_start : ℝ) 
  (david_percent_left john_percent_spent emily_percent_spent : ℝ) : 
  (david_start = 3200) → 
  (david_percent_left = 0.65) → 
  (john_start = 2500) → 
  (john_percent_spent = 0.60) → 
  (emily_start = 4000) → 
  (emily_percent_spent = 0.45) → 
  let david_spent := david_start / (1 + david_percent_left)
  let david_remaining := david_start - david_spent
  let john_remaining := john_start * (1 - john_percent_spent)
  let emily_remaining := emily_start * (1 - emily_percent_spent)
  david_remaining + john_remaining + emily_remaining = 4460.61 :=
by
  sorry

end NUMINAMATH_GPT_total_money_left_l2074_207454


namespace NUMINAMATH_GPT_ratio_of_paper_plates_l2074_207479

theorem ratio_of_paper_plates (total_pallets : ℕ) (paper_towels : ℕ) (tissues : ℕ) (paper_cups : ℕ) :
  total_pallets = 20 →
  paper_towels = 20 / 2 →
  tissues = 20 / 4 →
  paper_cups = 1 →
  (total_pallets - (paper_towels + tissues + paper_cups)) / total_pallets = 1 / 5 :=
by
  intros h_total h_towels h_tissues h_cups
  sorry

end NUMINAMATH_GPT_ratio_of_paper_plates_l2074_207479


namespace NUMINAMATH_GPT_largest_number_A_l2074_207469

theorem largest_number_A (A B C : ℕ) (h1: A = 7 * B + C) (h2: B = C) 
  : A ≤ 48 :=
sorry

end NUMINAMATH_GPT_largest_number_A_l2074_207469


namespace NUMINAMATH_GPT_mango_production_l2074_207448

-- Conditions
def num_papaya_trees := 2
def papayas_per_tree := 10
def num_mango_trees := 3
def total_fruits := 80

-- Definition to be proven
def mangos_per_mango_tree : Nat :=
  (total_fruits - num_papaya_trees * papayas_per_tree) / num_mango_trees

theorem mango_production :
  mangos_per_mango_tree = 20 := by
  sorry

end NUMINAMATH_GPT_mango_production_l2074_207448


namespace NUMINAMATH_GPT_snow_white_last_trip_l2074_207430

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end NUMINAMATH_GPT_snow_white_last_trip_l2074_207430


namespace NUMINAMATH_GPT_find_k_solution_l2074_207465

theorem find_k_solution 
  (k : ℝ)
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) : 
  k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_solution_l2074_207465


namespace NUMINAMATH_GPT_math_enthusiast_gender_relation_female_success_probability_l2074_207447

-- Constants and probabilities
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 12
def d : ℕ := 28
def n : ℕ := 100
def P_male_success : ℚ := 3 / 4
def P_female_success : ℚ := 2 / 3
def K_threshold : ℚ := 6.635

-- Computation of K^2
def K_square : ℚ := n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- The first part of the proof comparing K^2 with threshold
theorem math_enthusiast_gender_relation : K_square < K_threshold := sorry

-- The second part calculating given conditions for probability calculation
def P_A : ℚ := (P_male_success ^ 2 * (1 - P_female_success)) + (2 * (1 - P_male_success) * P_male_success * P_female_success)
def P_AB : ℚ := 2 * (1 - P_male_success) * P_male_success * P_female_success
def P_B_given_A : ℚ := P_AB / P_A

theorem female_success_probability : P_B_given_A = 4 / 7 := sorry

end NUMINAMATH_GPT_math_enthusiast_gender_relation_female_success_probability_l2074_207447


namespace NUMINAMATH_GPT_isosceles_right_triangle_C_coordinates_l2074_207411

theorem isosceles_right_triangle_C_coordinates :
  ∃ C : ℝ × ℝ, (let A : ℝ × ℝ := (1, 0)
                let B : ℝ × ℝ := (3, 1) 
                ∃ (x y: ℝ), C = (x, y) ∧ 
                ((x-1)^2 + y^2 = 10) ∧ 
                (((x-3)^2 + (y-1)^2 = 10))) ∨
                ((x = 2 ∧ y = 3) ∨ (x = 4 ∧ y = -1)) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_C_coordinates_l2074_207411


namespace NUMINAMATH_GPT_tens_digit_of_7_pow_2011_l2074_207471

-- Define the conditions for the problem
def seven_power := 7
def exponent := 2011
def modulo := 100

-- Define the target function to find the tens digit
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem formally
theorem tens_digit_of_7_pow_2011 : tens_digit (seven_power ^ exponent % modulo) = 4 := by
  sorry

end NUMINAMATH_GPT_tens_digit_of_7_pow_2011_l2074_207471


namespace NUMINAMATH_GPT_joe_total_toy_cars_l2074_207417

def joe_toy_cars (initial_cars additional_cars : ℕ) : ℕ :=
  initial_cars + additional_cars

theorem joe_total_toy_cars : joe_toy_cars 500 120 = 620 := by
  sorry

end NUMINAMATH_GPT_joe_total_toy_cars_l2074_207417


namespace NUMINAMATH_GPT_min_value_expr_l2074_207419

theorem min_value_expr (x : ℝ) (h : x > -3) : ∃ m, (∀ y > -3, 2 * y + (1 / (y + 3)) ≥ m) ∧ m = 2 * Real.sqrt 2 - 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l2074_207419


namespace NUMINAMATH_GPT_inequality_proof_l2074_207499

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l2074_207499


namespace NUMINAMATH_GPT_simplify_expression_l2074_207434

theorem simplify_expression :
  20 * (14 / 15) * (2 / 18) * (5 / 4) = 70 / 27 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2074_207434


namespace NUMINAMATH_GPT_reynald_soccer_balls_l2074_207413

theorem reynald_soccer_balls (total_balls basketballs_more soccer tennis baseball more_baseballs volleyballs : ℕ) 
(h_total_balls: total_balls = 145) 
(h_basketballs_more: basketballs_more = 5)
(h_tennis: tennis = 2 * soccer)
(h_more_baseballs: more_baseballs = 10)
(h_volleyballs: volleyballs = 30) 
(sum_eq: soccer + (soccer + basketballs_more) + tennis + (soccer + more_baseballs) + volleyballs = total_balls) : soccer = 20 := 
by
  sorry

end NUMINAMATH_GPT_reynald_soccer_balls_l2074_207413


namespace NUMINAMATH_GPT_sin_C_l2074_207406

variable {A B C : ℝ}

theorem sin_C (hA : A = 90) (hcosB : Real.cos B = 3/5) : Real.sin (90 - B) = 3/5 :=
by
  sorry

end NUMINAMATH_GPT_sin_C_l2074_207406


namespace NUMINAMATH_GPT_maximum_z_l2074_207423

theorem maximum_z (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end NUMINAMATH_GPT_maximum_z_l2074_207423


namespace NUMINAMATH_GPT_rahim_books_from_first_shop_l2074_207421

variable (books_first_shop_cost : ℕ)
variable (second_shop_books : ℕ)
variable (second_shop_books_cost : ℕ)
variable (average_price_per_book : ℕ)
variable (number_of_books_first_shop : ℕ)

theorem rahim_books_from_first_shop
  (h₁ : books_first_shop_cost = 581)
  (h₂ : second_shop_books = 20)
  (h₃ : second_shop_books_cost = 594)
  (h₄ : average_price_per_book = 25)
  (h₅ : (books_first_shop_cost + second_shop_books_cost) = (number_of_books_first_shop + second_shop_books) * average_price_per_book) :
  number_of_books_first_shop = 27 :=
sorry

end NUMINAMATH_GPT_rahim_books_from_first_shop_l2074_207421


namespace NUMINAMATH_GPT_marbles_problem_l2074_207451

theorem marbles_problem (initial_marble_tyrone : ℕ) (initial_marble_eric : ℕ) (x : ℝ)
  (h1 : initial_marble_tyrone = 125)
  (h2 : initial_marble_eric = 25)
  (h3 : initial_marble_tyrone - x = 3 * (initial_marble_eric + x)) :
  x = 12.5 := 
sorry

end NUMINAMATH_GPT_marbles_problem_l2074_207451


namespace NUMINAMATH_GPT_problem_statement_l2074_207408

noncomputable def repeating_decimal_to_fraction (n : ℕ) : ℚ :=
  -- Conversion function for repeating two-digit decimals to fractions
  n / 99

theorem problem_statement :
  (repeating_decimal_to_fraction 63) / (repeating_decimal_to_fraction 21) = 3 :=
by
  -- expected simplification and steps skipped
  sorry

end NUMINAMATH_GPT_problem_statement_l2074_207408


namespace NUMINAMATH_GPT_quadratic_complete_square_l2074_207481

theorem quadratic_complete_square (x p q : ℤ) 
  (h_eq : x^2 - 6 * x + 3 = 0) 
  (h_pq_form : x^2 - 6 * x + (p - x)^2 = q) 
  (h_int : ∀ t, t = p + q) : p + q = 3 := sorry

end NUMINAMATH_GPT_quadratic_complete_square_l2074_207481


namespace NUMINAMATH_GPT_negation_of_proposition_divisible_by_2_is_not_even_l2074_207456

theorem negation_of_proposition_divisible_by_2_is_not_even :
  (¬ ∀ n : ℕ, n % 2 = 0 → (n % 2 = 0 → n % 2 = 0))
  ↔ ∃ n : ℕ, n % 2 = 0 ∧ n % 2 ≠ 0 := 
  by
    sorry

end NUMINAMATH_GPT_negation_of_proposition_divisible_by_2_is_not_even_l2074_207456


namespace NUMINAMATH_GPT_jason_stacked_bales_l2074_207414

theorem jason_stacked_bales (initial_bales : ℕ) (final_bales : ℕ) (stored_bales : ℕ) 
  (h1 : initial_bales = 73) (h2 : final_bales = 96) : stored_bales = final_bales - initial_bales := 
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_jason_stacked_bales_l2074_207414


namespace NUMINAMATH_GPT_volume_maximized_at_r_5_h_8_l2074_207407

noncomputable def V (r : ℝ) : ℝ := (Real.pi / 5) * (300 * r - 4 * r^3)

/-- (1) Given that the total construction cost is 12000π yuan, 
express the volume V as a function of the radius r, and determine its domain. -/
def volume_function (r : ℝ) (h : ℝ) (cost : ℝ) : Prop :=
  cost = 12000 * Real.pi ∧
  h = 1 / (5 * r) * (300 - 4 * r^2) ∧
  V r = Real.pi * r^2 * h ∧
  0 < r ∧ r < 5 * Real.sqrt 3

/-- (2) Prove V(r) is maximized when r = 5 and h = 8 -/
theorem volume_maximized_at_r_5_h_8 :
  ∀ (r : ℝ) (h : ℝ) (cost : ℝ), volume_function r h cost → 
  ∃ (r_max : ℝ) (h_max : ℝ), r_max = 5 ∧ h_max = 8 ∧ ∀ x, 0 < x → x < 5 * Real.sqrt 3 → V x ≤ V r_max :=
by
  intros r h cost hvolfunc
  sorry

end NUMINAMATH_GPT_volume_maximized_at_r_5_h_8_l2074_207407


namespace NUMINAMATH_GPT_quadratic_roots_product_sum_l2074_207429

theorem quadratic_roots_product_sum :
  (∀ d e : ℝ, 3 * d^2 + 4 * d - 7 = 0 ∧ 3 * e^2 + 4 * e - 7 = 0 →
   (d + 1) * (e + 1) = - 8 / 3) := by
sorry

end NUMINAMATH_GPT_quadratic_roots_product_sum_l2074_207429


namespace NUMINAMATH_GPT_frame_width_proof_l2074_207409

noncomputable section

-- Define the given conditions
def perimeter_square_opening := 60 -- cm
def perimeter_entire_frame := 180 -- cm

-- Define what we need to prove: the width of the frame
def width_of_frame : ℕ := 5 -- cm

-- Define a function to calculate the side length of a square
def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

-- Define the side length of the square opening
def side_length_opening := side_length_of_square perimeter_square_opening

-- Use the given conditions to calculate the frame's width
-- Given formulas in the solution steps:
--  2 * (3 * side_length + 4 * d) + 2 * (side_length + 2 * d) = perimeter_entire_frame
theorem frame_width_proof (d : ℕ) (perim_square perim_frame : ℕ) :
  perim_square = perimeter_square_opening →
  perim_frame = perimeter_entire_frame →
  2 * (3 * side_length_of_square perim_square + 4 * d) 
  + 2 * (side_length_of_square perim_square + 2 * d) 
  = perim_frame →
  d = width_of_frame := 
by 
  intros h1 h2 h3
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_frame_width_proof_l2074_207409


namespace NUMINAMATH_GPT_probability_one_and_three_painted_faces_l2074_207462

-- Define the conditions of the problem
def side_length := 5
def total_unit_cubes := side_length^3
def painted_faces := 2
def unit_cubes_one_painted_face := 26
def unit_cubes_three_painted_faces := 4

-- Define the probability statement in Lean
theorem probability_one_and_three_painted_faces :
  (unit_cubes_one_painted_face * unit_cubes_three_painted_faces : ℝ) / (total_unit_cubes * (total_unit_cubes - 1) / 2) = 52 / 3875 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_and_three_painted_faces_l2074_207462


namespace NUMINAMATH_GPT_john_streams_hours_per_day_l2074_207428

theorem john_streams_hours_per_day :
  (∃ h : ℕ, (7 - 3) * h * 10 = 160) → 
  (∃ h : ℕ, h = 4) :=
sorry

end NUMINAMATH_GPT_john_streams_hours_per_day_l2074_207428


namespace NUMINAMATH_GPT_opposite_number_113_is_114_l2074_207440

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end NUMINAMATH_GPT_opposite_number_113_is_114_l2074_207440


namespace NUMINAMATH_GPT_geometric_sequence_a5_value_l2074_207455

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n m : ℕ, a n = a 0 * r ^ n)
  (h_condition : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a5_value_l2074_207455


namespace NUMINAMATH_GPT_predict_HCl_formed_l2074_207426

-- Define the initial conditions and chemical reaction constants
def initial_moles_CH4 : ℝ := 3
def initial_moles_Cl2 : ℝ := 6
def volume : ℝ := 2

-- Define the reaction stoichiometry constants
def stoich_CH4_to_HCl : ℝ := 2
def stoich_CH4 : ℝ := 1
def stoich_Cl2 : ℝ := 2

-- Declare the hypothesis that reaction goes to completion
axiom reaction_goes_to_completion : Prop

-- Define the function to calculate the moles of HCl formed
def moles_HCl_formed : ℝ :=
  initial_moles_CH4 * stoich_CH4_to_HCl

-- Prove the predicted amount of HCl formed is 6 moles under the given conditions
theorem predict_HCl_formed : reaction_goes_to_completion → moles_HCl_formed = 6 := by
  sorry

end NUMINAMATH_GPT_predict_HCl_formed_l2074_207426


namespace NUMINAMATH_GPT_nat_numbers_eq_floor_condition_l2074_207438

theorem nat_numbers_eq_floor_condition (a b : ℕ):
  (⌊(a ^ 2 : ℚ) / b⌋₊ + ⌊(b ^ 2 : ℚ) / a⌋₊ = ⌊((a ^ 2 + b ^ 2) : ℚ) / (a * b)⌋₊ + a * b) →
  (b = a ^ 2 + 1) ∨ (a = b ^ 2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_nat_numbers_eq_floor_condition_l2074_207438


namespace NUMINAMATH_GPT_weighted_average_is_correct_l2074_207480

def bag1_pop_kernels := 60
def bag1_total_kernels := 75
def bag2_pop_kernels := 42
def bag2_total_kernels := 50
def bag3_pop_kernels := 25
def bag3_total_kernels := 100
def bag4_pop_kernels := 77
def bag4_total_kernels := 120
def bag5_pop_kernels := 106
def bag5_total_kernels := 150

noncomputable def weighted_average_percentage : ℚ :=
  ((bag1_pop_kernels / bag1_total_kernels * 100 * bag1_total_kernels) +
   (bag2_pop_kernels / bag2_total_kernels * 100 * bag2_total_kernels) +
   (bag3_pop_kernels / bag3_total_kernels * 100 * bag3_total_kernels) +
   (bag4_pop_kernels / bag4_total_kernels * 100 * bag4_total_kernels) +
   (bag5_pop_kernels / bag5_total_kernels * 100 * bag5_total_kernels)) /
  (bag1_total_kernels + bag2_total_kernels + bag3_total_kernels + bag4_total_kernels + bag5_total_kernels)

theorem weighted_average_is_correct : weighted_average_percentage = 60.61 := 
by
  sorry

end NUMINAMATH_GPT_weighted_average_is_correct_l2074_207480


namespace NUMINAMATH_GPT_inequality_abc_l2074_207472

theorem inequality_abc (a b c : ℝ) : a^2 + 4 * b^2 + 8 * c^2 ≥ 3 * a * b + 4 * b * c + 2 * c * a :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l2074_207472


namespace NUMINAMATH_GPT_part1_monotonicity_part2_intersection_l2074_207483

noncomputable def f (a x : ℝ) : ℝ := -x * Real.exp (a * x + 1)

theorem part1_monotonicity (a : ℝ) : 
  ∃ interval : Set ℝ, 
    (∀ x ∈ interval, ∃ interval' : Set ℝ, 
      (∀ x' ∈ interval', f a x' ≤ f a x) ∧ 
      (∀ x' ∈ Set.univ \ interval', f a x' > f a x)) :=
sorry

theorem part2_intersection (a b x_1 x_2 : ℝ) (h1 : a > 0) (h2 : b ≠ 0)
  (h3 : f a x_1 = -b * Real.exp 1) (h4 : f a x_2 = -b * Real.exp 1)
  (h5 : x_1 ≠ x_2) : 
  - (1 / Real.exp 1) < a * b ∧ a * b < 0 ∧ a * (x_1 + x_2) < -2 :=
sorry

end NUMINAMATH_GPT_part1_monotonicity_part2_intersection_l2074_207483


namespace NUMINAMATH_GPT_value_of_X_l2074_207449

def M : ℕ := 2024 / 4
def N : ℕ := M / 2
def X : ℕ := M + N

theorem value_of_X : X = 759 := by
  sorry

end NUMINAMATH_GPT_value_of_X_l2074_207449


namespace NUMINAMATH_GPT_license_plate_calculation_l2074_207488

def license_plate_count : ℕ :=
  let letter_choices := 26^3
  let first_digit_choices := 5
  let remaining_digit_combinations := 5 * 5
  letter_choices * first_digit_choices * remaining_digit_combinations

theorem license_plate_calculation :
  license_plate_count = 455625 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_calculation_l2074_207488


namespace NUMINAMATH_GPT_intersection_setA_setB_l2074_207401

namespace Proof

def setA : Set ℝ := {x | ∃ y : ℝ, y = x + 1}
def setB : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

theorem intersection_setA_setB : (setA ∩ setB) = {y | 0 < y} :=
by
  sorry

end Proof

end NUMINAMATH_GPT_intersection_setA_setB_l2074_207401


namespace NUMINAMATH_GPT_find_a_l2074_207424

-- Definitions of the conditions
def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

-- The proof goal
theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l2074_207424


namespace NUMINAMATH_GPT_fare_collected_from_I_class_l2074_207415

theorem fare_collected_from_I_class (x y : ℕ) 
  (h_ratio_passengers : 4 * x = 4 * x) -- ratio of passengers 1:4
  (h_ratio_fare : 3 * y = 3 * y) -- ratio of fares 3:1
  (h_total_fare : 7 * 3 * x * y = 224000) -- total fare Rs. 224000
  : 3 * x * y = 96000 := 
by
  sorry

end NUMINAMATH_GPT_fare_collected_from_I_class_l2074_207415


namespace NUMINAMATH_GPT_solution_l2074_207402

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry
noncomputable def x7 : ℝ := sorry
noncomputable def x8 : ℝ := sorry

axiom cond1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 + 64 * x8 = 10
axiom cond2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 + 81 * x8 = 40
axiom cond3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 + 100 * x8 = 170

theorem solution : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 + 121 * x8 = 400 := 
by
  sorry

end NUMINAMATH_GPT_solution_l2074_207402


namespace NUMINAMATH_GPT_find_m_l2074_207493

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d 

noncomputable def sum_first_n_terms (a S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

theorem find_m {a S : ℕ → ℤ} (d : ℤ) (m : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : a 1 = 1)
  (h4 : S 3 = a 5)
  (h5 : a m = 2011) :
  m = 1006 :=
sorry

end NUMINAMATH_GPT_find_m_l2074_207493


namespace NUMINAMATH_GPT_fraction_irreducible_iff_l2074_207405

-- Define the condition for natural number n
def is_natural (n : ℕ) : Prop :=
  True  -- All undergraduate natural numbers abide to True

-- Main theorem formalized in Lean 4
theorem fraction_irreducible_iff (n : ℕ) :
  (∃ (g : ℕ), g = 1 ∧ (∃ a b : ℕ, 2 * n * n + 11 * n - 18 = a * g ∧ n + 7 = b * g)) ↔ 
  (n % 3 = 0 ∨ n % 3 = 1) :=
by sorry

end NUMINAMATH_GPT_fraction_irreducible_iff_l2074_207405


namespace NUMINAMATH_GPT_steps_already_climbed_l2074_207432

-- Definitions based on conditions
def total_stair_steps : ℕ := 96
def steps_left_to_climb : ℕ := 22

-- Theorem proving the number of steps already climbed
theorem steps_already_climbed : total_stair_steps - steps_left_to_climb = 74 := by
  sorry

end NUMINAMATH_GPT_steps_already_climbed_l2074_207432


namespace NUMINAMATH_GPT_quad_equiv_proof_l2074_207452

theorem quad_equiv_proof (a b : ℝ) (h : a ≠ 0) (hroot : a * 2019^2 + b * 2019 + 2 = 0) :
  ∃ x : ℝ, a * (x - 1)^2 + b * (x - 1) = -2 ∧ x = 2019 :=
sorry

end NUMINAMATH_GPT_quad_equiv_proof_l2074_207452


namespace NUMINAMATH_GPT_draws_alternate_no_consecutive_same_color_l2074_207403

-- Defining the total number of balls and the count of each color.
def total_balls : ℕ := 15
def white_balls : ℕ := 5
def black_balls : ℕ := 5
def red_balls : ℕ := 5

-- Defining the probability that the draws alternate in colors with no two consecutive balls of the same color.
def probability_no_consecutive_same_color : ℚ := 162 / 1001

theorem draws_alternate_no_consecutive_same_color :
  (white_balls + black_balls + red_balls = total_balls) →
  -- The resulting probability based on the given conditions.
  probability_no_consecutive_same_color = 162 / 1001 := by
  sorry

end NUMINAMATH_GPT_draws_alternate_no_consecutive_same_color_l2074_207403


namespace NUMINAMATH_GPT_specific_time_l2074_207498

theorem specific_time :
  (∀ (s : ℕ), 0 ≤ s ∧ s ≤ 7 → (∃ (t : ℕ), (t ^ 2 + 2 * t) - (3 ^ 2 + 2 * 3) = 20 ∧ t = 5)) :=
  by sorry

end NUMINAMATH_GPT_specific_time_l2074_207498


namespace NUMINAMATH_GPT_min_value_of_a_l2074_207495

theorem min_value_of_a (a : ℝ) (h : ∃ x : ℝ, |x - 1| + |x + a| ≤ 8) : -9 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_l2074_207495


namespace NUMINAMATH_GPT_multiplication_factor_average_l2074_207489

theorem multiplication_factor_average (a : ℕ) (b : ℕ) (c : ℕ) (F : ℝ) 
  (h1 : a = 7) 
  (h2 : b = 26) 
  (h3 : (c : ℝ) = 130) 
  (h4 : (a * b * F : ℝ) = a * c) :
  F = 5 := 
by 
  sorry

end NUMINAMATH_GPT_multiplication_factor_average_l2074_207489


namespace NUMINAMATH_GPT_Ariana_running_time_l2074_207463

theorem Ariana_running_time
  (time_Sadie : ℝ)
  (speed_Sadie : ℝ)
  (speed_Ariana : ℝ)
  (speed_Sarah : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_Sadie := speed_Sadie * time_Sadie)
  (time_Ariana_Sarah := total_time - time_Sadie)
  (distance_Ariana_Sarah := total_distance - distance_Sadie) :
  (6 * (time_Ariana_Sarah - (11 - 6 * (time_Ariana_Sarah / (speed_Ariana + (4 / speed_Sarah)))))
  = (0.5 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_Ariana_running_time_l2074_207463


namespace NUMINAMATH_GPT_inverse_of_congruence_implies_equal_area_l2074_207410

-- Definitions to capture conditions and relationships
def congruent_triangles (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with congruency of two triangles
  sorry

def equal_areas (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with equal areas of two triangles
  sorry

-- Statement to prove the inverse proposition
theorem inverse_of_congruence_implies_equal_area :
  (∀ T1 T2 : Triangle, congruent_triangles T1 T2 → equal_areas T1 T2) →
  (∀ T1 T2 : Triangle, equal_areas T1 T2 → congruent_triangles T1 T2) :=
  sorry

end NUMINAMATH_GPT_inverse_of_congruence_implies_equal_area_l2074_207410


namespace NUMINAMATH_GPT_point_outside_circle_l2074_207466

theorem point_outside_circle
  (radius : ℝ) (distance : ℝ) (h_radius : radius = 8) (h_distance : distance = 10) :
  distance > radius :=
by sorry

end NUMINAMATH_GPT_point_outside_circle_l2074_207466


namespace NUMINAMATH_GPT_gcd_9247_4567_eq_1_l2074_207418

theorem gcd_9247_4567_eq_1 : Int.gcd 9247 4567 = 1 := sorry

end NUMINAMATH_GPT_gcd_9247_4567_eq_1_l2074_207418


namespace NUMINAMATH_GPT_six_times_six_l2074_207461

-- Definitions based on the conditions
def pattern (n : ℕ) : ℕ := n * 6

-- Theorem statement to be proved
theorem six_times_six : pattern 6 = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_six_times_six_l2074_207461


namespace NUMINAMATH_GPT_functional_equation_solution_l2074_207470

theorem functional_equation_solution (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g (f (x + y)) = f x + 2 * (x + y) * g y) : 
  (∀ x : ℝ, f x = 0) ∧ (∀ x : ℝ, g x = 0) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2074_207470
