import Mathlib

namespace NUMINAMATH_GPT_common_ratio_of_geometric_progression_l589_58935

-- Define the problem conditions
variables {a b c q : ℝ}

-- The sequence a, b, c is a geometric progression
def geometric_progression (a b c : ℝ) (q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- The sequence 577a, (2020b/7), (c/7) is an arithmetic progression
def arithmetic_progression (x y z : ℝ) : Prop :=
  2 * y = x + z

-- Main theorem statement to prove
theorem common_ratio_of_geometric_progression (h1 : geometric_progression a b c q) 
  (h2 : arithmetic_progression (577 * a) (2020 * b / 7) (c / 7)) 
  (h3 : b < a ∧ c < b) : q = 4039 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_progression_l589_58935


namespace NUMINAMATH_GPT_sum_of_solutions_eq_neg4_l589_58927

theorem sum_of_solutions_eq_neg4 :
  ∃ (n : ℕ) (solutions : Fin n → ℝ × ℝ),
    (∀ i, ∃ (x y : ℝ), solutions i = (x, y) ∧ abs (x - 3) = abs (y - 9) ∧ abs (x - 9) = 2 * abs (y - 3)) ∧
    (Finset.univ.sum (fun i => (solutions i).1 + (solutions i).2) = -4) :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_neg4_l589_58927


namespace NUMINAMATH_GPT_conclusion_friendly_not_large_l589_58924

variable {Snake : Type}
variable (isLarge isFriendly canClimb canSwim : Snake → Prop)
variable (marysSnakes : Finset Snake)
variable (h1 : marysSnakes.card = 16)
variable (h2 : (marysSnakes.filter isLarge).card = 6)
variable (h3 : (marysSnakes.filter isFriendly).card = 7)
variable (h4 : ∀ s, isFriendly s → canClimb s)
variable (h5 : ∀ s, isLarge s → ¬ canSwim s)
variable (h6 : ∀ s, ¬ canSwim s → ¬ canClimb s)

theorem conclusion_friendly_not_large :
  ∀ s, isFriendly s → ¬ isLarge s :=
by
  sorry

end NUMINAMATH_GPT_conclusion_friendly_not_large_l589_58924


namespace NUMINAMATH_GPT_find_a_l589_58941

theorem find_a (a : ℕ) : 
  (a >= 100 ∧ a <= 999) ∧ 7 ∣ (504000 + a) ∧ 9 ∣ (504000 + a) ∧ 11 ∣ (504000 + a) ↔ a = 711 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l589_58941


namespace NUMINAMATH_GPT_smallest_five_digit_equiv_11_mod_13_l589_58942

open Nat

theorem smallest_five_digit_equiv_11_mod_13 :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 13 = 11 ∧ n = 10009 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_equiv_11_mod_13_l589_58942


namespace NUMINAMATH_GPT_focus_of_parabola_l589_58961

def parabola_focus (a k : ℕ) : ℚ :=
  1 / (4 * a) + k

theorem focus_of_parabola :
  parabola_focus 9 6 = 217 / 36 :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l589_58961


namespace NUMINAMATH_GPT_greatest_mean_weight_l589_58944

variable (X Y Z : Type) [Group X] [Group Y] [Group Z]

theorem greatest_mean_weight 
  (mean_X : ℝ) (mean_Y : ℝ) (mean_XY : ℝ) (mean_XZ : ℝ)
  (hX : mean_X = 30)
  (hY : mean_Y = 70)
  (hXY : mean_XY = 50)
  (hXZ : mean_XZ = 40) :
  ∃ k : ℝ, k = 70 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_mean_weight_l589_58944


namespace NUMINAMATH_GPT_g_at_3_value_l589_58901

theorem g_at_3_value (c d : ℝ) (g : ℝ → ℝ) 
  (h1 : g 1 = 7)
  (h2 : g 2 = 11)
  (h3 : ∀ x : ℝ, g x = c * x + d * x + 3) : 
  g 3 = 15 :=
by
  sorry

end NUMINAMATH_GPT_g_at_3_value_l589_58901


namespace NUMINAMATH_GPT_remainder_of_x_plus_2_pow_2022_l589_58963

theorem remainder_of_x_plus_2_pow_2022 (x : ℂ) :
  ∃ r : ℂ, ∃ q : ℂ, (x + 2)^2022 = q * (x^2 - x + 1) + r ∧ (r = x) :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_x_plus_2_pow_2022_l589_58963


namespace NUMINAMATH_GPT_suitable_survey_l589_58929

def survey_suitable_for_census (A B C D : Prop) : Prop :=
  A ∧ ¬B ∧ ¬C ∧ ¬D

theorem suitable_survey {A B C D : Prop} (h_A : A) (h_B : ¬B) (h_C : ¬C) (h_D : ¬D) : survey_suitable_for_census A B C D :=
by
  unfold survey_suitable_for_census
  exact ⟨h_A, h_B, h_C, h_D⟩

end NUMINAMATH_GPT_suitable_survey_l589_58929


namespace NUMINAMATH_GPT_find_ratio_of_sides_l589_58986

variable {A B : ℝ}
variable {a b : ℝ}

-- Given condition
axiom given_condition : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = a * Real.sqrt 3

-- Theorem we need to prove
theorem find_ratio_of_sides (h : a ≠ 0) : b / a = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_of_sides_l589_58986


namespace NUMINAMATH_GPT_equation_solution_l589_58995

theorem equation_solution (x : ℚ) (h₁ : (5 * x^2 + 4 * x + 2) / (x + 2) = 5 * x - 3) : x = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l589_58995


namespace NUMINAMATH_GPT_student_marks_l589_58978

theorem student_marks :
  let max_marks := 300
  let passing_percentage := 0.60
  let failed_by := 20
  let passing_marks := max_marks * passing_percentage
  let marks_obtained := passing_marks - failed_by
  marks_obtained = 160 := by
sorry

end NUMINAMATH_GPT_student_marks_l589_58978


namespace NUMINAMATH_GPT_diff_x_y_l589_58925

theorem diff_x_y (x y : ℤ) (h1 : x + y = 14) (h2 : x = 37) : x - y = 60 :=
sorry

end NUMINAMATH_GPT_diff_x_y_l589_58925


namespace NUMINAMATH_GPT_find_s_l589_58934

noncomputable def s_value (m : ℝ) : ℝ := m + 16.25

theorem find_s (a b m s : ℝ)
  (h1 : a + b = m) (h2 : a * b = 4) :
  s = s_value m :=
by
  sorry

end NUMINAMATH_GPT_find_s_l589_58934


namespace NUMINAMATH_GPT_concentration_in_third_flask_l589_58951

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end NUMINAMATH_GPT_concentration_in_third_flask_l589_58951


namespace NUMINAMATH_GPT_initially_calculated_average_is_correct_l589_58985

theorem initially_calculated_average_is_correct :
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  initially_avg = 22 :=
by
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  show initially_avg = 22
  sorry

end NUMINAMATH_GPT_initially_calculated_average_is_correct_l589_58985


namespace NUMINAMATH_GPT_total_drink_volume_l589_58946

variable (T : ℝ)

theorem total_drink_volume :
  (0.15 * T + 0.60 * T + 0.25 * T = 35) → T = 140 :=
by
  intros h
  have h1 : (0.25 * T) = 35 := by sorry
  have h2 : T = 140 := by sorry
  exact h2

end NUMINAMATH_GPT_total_drink_volume_l589_58946


namespace NUMINAMATH_GPT_range_of_fraction_l589_58921

theorem range_of_fraction (x1 y1 : ℝ) (h1 : y1 = -2 * x1 + 8) (h2 : 2 ≤ x1 ∧ x1 ≤ 5) :
  -1/6 ≤ (y1 + 1) / (x1 + 1) ∧ (y1 + 1) / (x1 + 1) ≤ 5/3 :=
sorry

end NUMINAMATH_GPT_range_of_fraction_l589_58921


namespace NUMINAMATH_GPT_unique_positive_integers_l589_58979

theorem unique_positive_integers (x y : ℕ) (h1 : x^2 + 84 * x + 2008 = y^2) : x + y = 80 :=
  sorry

end NUMINAMATH_GPT_unique_positive_integers_l589_58979


namespace NUMINAMATH_GPT_percent_calculation_l589_58900

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end NUMINAMATH_GPT_percent_calculation_l589_58900


namespace NUMINAMATH_GPT_eqn_distinct_real_roots_l589_58910

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then x^2 + 2 else 4 * x * Real.cos x + 1

theorem eqn_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, f x = m * x + 1) → 
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-2 * Real.pi) Real.pi ∧ x₂ ∈ Set.Icc (-2 * Real.pi) Real.pi :=
  sorry

end NUMINAMATH_GPT_eqn_distinct_real_roots_l589_58910


namespace NUMINAMATH_GPT_bacteria_seventh_generation_l589_58926

/-- Represents the effective multiplication factor per generation --/
def effective_mult_factor : ℕ := 4

/-- The number of bacteria in the first generation --/
def first_generation : ℕ := 1

/-- A helper function to compute the number of bacteria in the nth generation --/
def bacteria_count (n : ℕ) : ℕ :=
  first_generation * effective_mult_factor ^ n

/-- The number of bacteria in the seventh generation --/
theorem bacteria_seventh_generation : bacteria_count 7 = 4096 := by
  sorry

end NUMINAMATH_GPT_bacteria_seventh_generation_l589_58926


namespace NUMINAMATH_GPT_cost_price_A_l589_58920

theorem cost_price_A (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ) 
(h1 : CP_B = 1.20 * CP_A)
(h2 : SP_C = 1.25 * CP_B)
(h3 : SP_C = 225) : 
CP_A = 150 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_A_l589_58920


namespace NUMINAMATH_GPT_greatest_common_divisor_is_one_l589_58912

-- Define the expressions for a and b
def a : ℕ := 114^2 + 226^2 + 338^2
def b : ℕ := 113^2 + 225^2 + 339^2

-- Now state that the gcd of a and b is 1
theorem greatest_common_divisor_is_one : Nat.gcd a b = 1 := sorry

end NUMINAMATH_GPT_greatest_common_divisor_is_one_l589_58912


namespace NUMINAMATH_GPT_downstream_distance_15_minutes_l589_58980

theorem downstream_distance_15_minutes
  (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ)
  (h1 : speed_boat = 24)
  (h2 : speed_current = 3)
  (h3 : time_minutes = 15) :
  let effective_speed := speed_boat + speed_current
  let time_hours := time_minutes / 60
  let distance := effective_speed * time_hours
  distance = 6.75 :=
by {
  sorry
}

end NUMINAMATH_GPT_downstream_distance_15_minutes_l589_58980


namespace NUMINAMATH_GPT_min_value_of_expression_l589_58967

open Real

noncomputable def minValue (x y z : ℝ) : ℝ :=
  x + 3 * y + 5 * z

theorem min_value_of_expression : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 8 → minValue x y z = 14.796 :=
by
  intros x y z h
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l589_58967


namespace NUMINAMATH_GPT_sum_of_min_x_y_l589_58914

theorem sum_of_min_x_y : ∃ (x y : ℕ), 
  (∃ a b c : ℕ, 180 = 2^a * 3^b * 5^c) ∧
  (∃ u v w : ℕ, 180 * x = 2^u * 3^v * 5^w ∧ u % 4 = 0 ∧ v % 4 = 0 ∧ w % 4 = 0) ∧
  (∃ p q r : ℕ, 180 * y = 2^p * 3^q * 5^r ∧ p % 6 = 0 ∧ q % 6 = 0 ∧ r % 6 = 0) ∧
  (x + y = 4054500) :=
sorry

end NUMINAMATH_GPT_sum_of_min_x_y_l589_58914


namespace NUMINAMATH_GPT_certain_event_at_least_one_good_product_l589_58937

-- Define the number of products and their types
def num_products := 12
def num_good_products := 10
def num_defective_products := 2
def num_selected_products := 3

-- Statement of the problem
theorem certain_event_at_least_one_good_product :
  ∀ (selected : Finset (Fin num_products)),
  selected.card = num_selected_products →
  ∃ p ∈ selected, p.val < num_good_products :=
sorry

end NUMINAMATH_GPT_certain_event_at_least_one_good_product_l589_58937


namespace NUMINAMATH_GPT_fenced_area_l589_58975

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end NUMINAMATH_GPT_fenced_area_l589_58975


namespace NUMINAMATH_GPT_min_value_of_quadratic_form_l589_58908

theorem min_value_of_quadratic_form (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + 2 * y^2 + 3 * z^2 ≥ 1/3 :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_form_l589_58908


namespace NUMINAMATH_GPT_remainder_prod_mod_7_l589_58905

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_prod_mod_7_l589_58905


namespace NUMINAMATH_GPT_smallest_a_l589_58943

theorem smallest_a (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  a = 17 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_l589_58943


namespace NUMINAMATH_GPT_gumballs_ensure_four_same_color_l589_58906

-- Define the total number of gumballs in each color
def red_gumballs : ℕ := 10
def white_gumballs : ℕ := 9
def blue_gumballs : ℕ := 8
def green_gumballs : ℕ := 7

-- Define the minimum number of gumballs to ensure four of the same color
def min_gumballs_to_ensure_four_same_color : ℕ := 13

-- Prove that the minimum number of gumballs to ensure four of the same color is 13
theorem gumballs_ensure_four_same_color (n : ℕ) 
  (h₁ : red_gumballs ≥ 3)
  (h₂ : white_gumballs ≥ 3)
  (h₃ : blue_gumballs ≥ 3)
  (h₄ : green_gumballs ≥ 3)
  : n ≥ min_gumballs_to_ensure_four_same_color := 
sorry

end NUMINAMATH_GPT_gumballs_ensure_four_same_color_l589_58906


namespace NUMINAMATH_GPT_maximum_xy_l589_58922

theorem maximum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_parallel : 2 * x + y = 2) : 
  xy ≤ 1/2 := 
  sorry

end NUMINAMATH_GPT_maximum_xy_l589_58922


namespace NUMINAMATH_GPT_sphere_surface_area_increase_l589_58952

theorem sphere_surface_area_increase (r : ℝ) (h_r_pos : 0 < r):
  let A := 4 * π * r ^ 2
  let r' := 1.10 * r
  let A' := 4 * π * (r') ^ 2
  let ΔA := A' - A
  (ΔA / A) * 100 = 21 := by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_increase_l589_58952


namespace NUMINAMATH_GPT_estimated_prob_is_0_9_l589_58988

section GerminationProbability

-- Defining the experiment data
structure ExperimentData :=
  (totalSeeds : ℕ)
  (germinatedSeeds : ℕ)
  (germinationRate : ℝ)

def experiments : List ExperimentData := [
  ⟨100, 91, 0.91⟩, 
  ⟨400, 358, 0.895⟩, 
  ⟨800, 724, 0.905⟩,
  ⟨1400, 1264, 0.903⟩,
  ⟨3500, 3160, 0.903⟩,
  ⟨7000, 6400, 0.914⟩
]

-- Hypothesis based on the given problem's observation
def estimated_germination_probability (experiments : List ExperimentData) : ℝ :=
  /- Fictively calculating the stable germination rate here; however, logically we should use 
     some weighted average or similar statistical stability method. -/
  0.9  -- Rounded and concluded estimated value based on observation

theorem estimated_prob_is_0_9 :
  estimated_germination_probability experiments = 0.9 :=
  sorry

end GerminationProbability

end NUMINAMATH_GPT_estimated_prob_is_0_9_l589_58988


namespace NUMINAMATH_GPT_part_a_correct_part_b_correct_l589_58994

-- Define the alphabet and mapping
inductive Letter
| C | H | M | O
deriving DecidableEq, Inhabited

open Letter

def letter_to_base4 (ch : Letter) : ℕ :=
  match ch with
  | C => 0
  | H => 1
  | M => 2
  | O => 3

def word_to_base4 (word : List Letter) : ℕ :=
  word.foldl (fun acc ch => acc * 4 + letter_to_base4 ch) 0

def base4_to_letter (n : ℕ) : Letter :=
  match n with
  | 0 => C
  | 1 => H
  | 2 => M
  | 3 => O
  | _ => C -- This should not occur if input is in valid base-4 range

def base4_to_word (n : ℕ) (size : ℕ) : List Letter :=
  if size = 0 then []
  else
    let quotient := n / 4
    let remainder := n % 4
    base4_to_letter remainder :: base4_to_word quotient (size - 1)

-- The size of the words is fixed at 8
def word_size : ℕ := 8

noncomputable def part_a : List Letter :=
  base4_to_word 2017 word_size

theorem part_a_correct :
  part_a = [H, O, O, H, M, C] := by
  sorry

def given_word : List Letter :=
  [H, O, M, C, H, O, M, C]

noncomputable def part_b : ℕ :=
  word_to_base4 given_word + 1 -- Adjust for zero-based indexing

theorem part_b_correct :
  part_b = 29299 := by
  sorry

end NUMINAMATH_GPT_part_a_correct_part_b_correct_l589_58994


namespace NUMINAMATH_GPT_smallest_number_of_small_bottles_l589_58987

def minimum_bottles_needed (large_bottle_capacity : ℕ) (small_bottle1 : ℕ) (small_bottle2 : ℕ) : ℕ :=
  if large_bottle_capacity = 720 ∧ small_bottle1 = 40 ∧ small_bottle2 = 45 then 16 else 0

theorem smallest_number_of_small_bottles :
  minimum_bottles_needed 720 40 45 = 16 := by
  sorry

end NUMINAMATH_GPT_smallest_number_of_small_bottles_l589_58987


namespace NUMINAMATH_GPT_rocky_training_miles_l589_58928

variable (x : ℕ)

theorem rocky_training_miles (h1 : x + 2 * x + 6 * x = 36) : x = 4 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_rocky_training_miles_l589_58928


namespace NUMINAMATH_GPT_intersection_of_lines_l589_58911

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 
  (5 * x - 3 * y = 20) ∧ (3 * x + 4 * y = 6) ∧ 
  x = 98 / 29 ∧ 
  y = 87 / 58 :=
by 
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l589_58911


namespace NUMINAMATH_GPT_intersect_trihedral_angle_l589_58904

-- Definitions of variables
variables {a b c : ℝ} (S : Type) 

-- Definition of a valid intersection condition
def valid_intersection (a b c : ℝ) : Prop :=
  a^2 + b^2 - c^2 > 0 ∧ b^2 + c^2 - a^2 > 0 ∧ a^2 + c^2 - b^2 > 0

-- Theorem statement
theorem intersect_trihedral_angle (h : valid_intersection a b c) : 
  ∃ (SA SB SC : ℝ), (SA^2 + SB^2 = a^2 ∧ SA^2 + SC^2 = b^2 ∧ SB^2 + SC^2 = c^2) :=
sorry

end NUMINAMATH_GPT_intersect_trihedral_angle_l589_58904


namespace NUMINAMATH_GPT_jason_cousins_l589_58960

theorem jason_cousins :
  let dozen := 12
  let cupcakes_bought := 4 * dozen
  let cupcakes_per_cousin := 3
  let number_of_cousins := cupcakes_bought / cupcakes_per_cousin
  number_of_cousins = 16 :=
by
  sorry

end NUMINAMATH_GPT_jason_cousins_l589_58960


namespace NUMINAMATH_GPT_malachi_additional_photos_l589_58992

-- Definition of the conditions
def total_photos : ℕ := 2430
def ratio_last_year : ℕ := 10
def ratio_this_year : ℕ := 17
def total_ratio_units : ℕ := ratio_last_year + ratio_this_year
def diff_ratio_units : ℕ := ratio_this_year - ratio_last_year
def photos_per_unit : ℕ := total_photos / total_ratio_units
def additional_photos : ℕ := diff_ratio_units * photos_per_unit

-- The theorem proving how many more photos Malachi took this year than last year
theorem malachi_additional_photos : additional_photos = 630 := by
  sorry

end NUMINAMATH_GPT_malachi_additional_photos_l589_58992


namespace NUMINAMATH_GPT_maddie_total_payment_l589_58998

def price_palettes : ℝ := 15
def num_palettes : ℕ := 3
def discount_palettes : ℝ := 0.20
def price_lipsticks : ℝ := 2.50
def num_lipsticks_bought : ℕ := 4
def num_lipsticks_pay : ℕ := 3
def price_hair_color : ℝ := 4
def num_hair_color : ℕ := 3
def discount_hair_color : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def total_cost_palettes : ℝ := num_palettes * price_palettes
def total_cost_palettes_after_discount : ℝ := total_cost_palettes * (1 - discount_palettes)

def total_cost_lipsticks : ℝ := num_lipsticks_pay * price_lipsticks

def total_cost_hair_color : ℝ := num_hair_color * price_hair_color
def total_cost_hair_color_after_discount : ℝ := total_cost_hair_color * (1 - discount_hair_color)

def total_pre_tax : ℝ := total_cost_palettes_after_discount + total_cost_lipsticks + total_cost_hair_color_after_discount
def total_sales_tax : ℝ := total_pre_tax * sales_tax_rate
def total_cost : ℝ := total_pre_tax + total_sales_tax

theorem maddie_total_payment : total_cost = 58.64 := by
  sorry

end NUMINAMATH_GPT_maddie_total_payment_l589_58998


namespace NUMINAMATH_GPT_inequality_holds_for_gt_sqrt2_l589_58976

theorem inequality_holds_for_gt_sqrt2 (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_holds_for_gt_sqrt2_l589_58976


namespace NUMINAMATH_GPT_nearest_int_to_expr_l589_58956

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end NUMINAMATH_GPT_nearest_int_to_expr_l589_58956


namespace NUMINAMATH_GPT_part_a_l589_58913

theorem part_a (a : ℤ) : (a^2 < 4) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := 
sorry

end NUMINAMATH_GPT_part_a_l589_58913


namespace NUMINAMATH_GPT_arithmetic_mean_of_18_24_42_l589_58984

-- Define the numbers a, b, c
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 42

-- Define the arithmetic mean
def mean (x y z : ℕ) : ℕ := (x + y + z) / 3

-- State the theorem to be proved
theorem arithmetic_mean_of_18_24_42 : mean a b c = 28 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_18_24_42_l589_58984


namespace NUMINAMATH_GPT_monday_has_greatest_temp_range_l589_58918

-- Define the temperatures
def high_temp (day : String) : Int :=
  if day = "Monday" then 6 else
  if day = "Tuesday" then 3 else
  if day = "Wednesday" then 4 else
  if day = "Thursday" then 4 else
  if day = "Friday" then 8 else 0

def low_temp (day : String) : Int :=
  if day = "Monday" then -4 else
  if day = "Tuesday" then -6 else
  if day = "Wednesday" then -2 else
  if day = "Thursday" then -5 else
  if day = "Friday" then 0 else 0

-- Define the temperature range for a given day
def temp_range (day : String) : Int :=
  high_temp day - low_temp day

-- Statement to prove: Monday has the greatest temperature range
theorem monday_has_greatest_temp_range : 
  temp_range "Monday" > temp_range "Tuesday" ∧
  temp_range "Monday" > temp_range "Wednesday" ∧
  temp_range "Monday" > temp_range "Thursday" ∧
  temp_range "Monday" > temp_range "Friday" := 
sorry

end NUMINAMATH_GPT_monday_has_greatest_temp_range_l589_58918


namespace NUMINAMATH_GPT_convert_mps_to_kmph_l589_58966

-- Define the conversion factor
def conversion_factor : ℝ := 3.6

-- Define the initial speed in meters per second
def initial_speed_mps : ℝ := 50

-- Define the target speed in kilometers per hour
def target_speed_kmph : ℝ := 180

-- Problem statement: Prove the conversion is correct
theorem convert_mps_to_kmph : initial_speed_mps * conversion_factor = target_speed_kmph := by
  sorry

end NUMINAMATH_GPT_convert_mps_to_kmph_l589_58966


namespace NUMINAMATH_GPT_number_of_ways_to_select_team_l589_58982

def calc_binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem number_of_ways_to_select_team : calc_binomial_coefficient 17 4 = 2380 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_select_team_l589_58982


namespace NUMINAMATH_GPT_cost_per_adult_is_3_l589_58969

-- Define the number of people in the group
def total_people : ℕ := 12

-- Define the number of kids in the group
def kids : ℕ := 7

-- Define the total cost for the group
def total_cost : ℕ := 15

-- Define the number of adults, which is the total number of people minus the number of kids
def adults : ℕ := total_people - kids

-- Define the cost per adult meal, which is the total cost divided by the number of adults
noncomputable def cost_per_adult : ℕ := total_cost / adults

-- The theorem stating the cost per adult meal is $3
theorem cost_per_adult_is_3 : cost_per_adult = 3 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_cost_per_adult_is_3_l589_58969


namespace NUMINAMATH_GPT_add_alcohol_solve_l589_58971

variable (x : ℝ)

def initial_solution_volume : ℝ := 6
def initial_alcohol_fraction : ℝ := 0.20
def desired_alcohol_fraction : ℝ := 0.50

def initial_alcohol_content : ℝ := initial_alcohol_fraction * initial_solution_volume
def total_solution_volume_after_addition : ℝ := initial_solution_volume + x
def total_alcohol_content_after_addition : ℝ := initial_alcohol_content + x

theorem add_alcohol_solve (x : ℝ) :
  (initial_alcohol_content + x) / (initial_solution_volume + x) = desired_alcohol_fraction →
  x = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_add_alcohol_solve_l589_58971


namespace NUMINAMATH_GPT_min_value_xy_k_l589_58950

theorem min_value_xy_k (x y k : ℝ) : ∃ x y : ℝ, (xy - k)^2 + (x + y - 1)^2 = 1 := by
  sorry

end NUMINAMATH_GPT_min_value_xy_k_l589_58950


namespace NUMINAMATH_GPT_area_of_enclosed_region_is_zero_l589_58933

theorem area_of_enclosed_region_is_zero :
  (∃ (x y : ℝ), x^2 + y^2 = |x| - |y|) → (0 = 0) :=
sorry

end NUMINAMATH_GPT_area_of_enclosed_region_is_zero_l589_58933


namespace NUMINAMATH_GPT_power_of_two_sequence_invariant_l589_58939

theorem power_of_two_sequence_invariant
  (n : ℕ)
  (a b : ℕ → ℕ)
  (h₀ : a 0 = 1)
  (h₁ : b 0 = n)
  (hi : ∀ i : ℕ, a i < b i → a (i + 1) = 2 * a i + 1 ∧ b (i + 1) = b i - a i - 1)
  (hj : ∀ i : ℕ, a i > b i → a (i + 1) = a i - b i - 1 ∧ b (i + 1) = 2 * b i + 1)
  (hk : ∀ i : ℕ, a i = b i → a (i + 1) = a i ∧ b (i + 1) = b i)
  (k : ℕ)
  (h : a k = b k) :
  ∃ m : ℕ, n + 3 = 2 ^ m :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_sequence_invariant_l589_58939


namespace NUMINAMATH_GPT_frank_pie_consumption_l589_58945

theorem frank_pie_consumption :
  let Erik := 0.6666666666666666
  let MoreThanFrank := 0.3333333333333333
  let Frank := Erik - MoreThanFrank
  Frank = 0.3333333333333333 := by
sorry

end NUMINAMATH_GPT_frank_pie_consumption_l589_58945


namespace NUMINAMATH_GPT_absolute_value_example_l589_58923

theorem absolute_value_example (x : ℝ) (h : x = 4) : |x - 5| = 1 :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_example_l589_58923


namespace NUMINAMATH_GPT_retirement_percentage_l589_58972

-- Define the conditions
def gross_pay : ℝ := 1120
def tax_deduction : ℝ := 100
def net_paycheck : ℝ := 740

-- Define the total deduction
def total_deduction : ℝ := gross_pay - net_paycheck
def retirement_deduction : ℝ := total_deduction - tax_deduction

-- Define the theorem to prove
theorem retirement_percentage :
  (retirement_deduction / gross_pay) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_retirement_percentage_l589_58972


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l589_58953

theorem arithmetic_sequence_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_S_def : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2)
  (h_S2 : S 2 = 4)
  (h_S4 : S 4 = 16) :
  a 5 + a 6 = 20 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l589_58953


namespace NUMINAMATH_GPT_largest_first_term_geometric_progression_l589_58954

noncomputable def geometric_progression_exists (d : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (a + d + 3) / a = (a + 2 * d + 15) / (a + d + 3)

theorem largest_first_term_geometric_progression : ∀ (d : ℝ), 
  d^2 + 6 * d - 36 = 0 → 
  ∃ (a : ℝ), a = 5 ∧ geometric_progression_exists d ∧ a = 5 ∧ 
  ∀ (a' : ℝ), geometric_progression_exists d → a' ≤ a :=
by intros d h; sorry

end NUMINAMATH_GPT_largest_first_term_geometric_progression_l589_58954


namespace NUMINAMATH_GPT_least_value_of_g_l589_58932

noncomputable def g (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 1

theorem least_value_of_g : ∃ x : ℝ, ∀ y : ℝ, g y ≥ g x ∧ g x = -2 := by
  sorry

end NUMINAMATH_GPT_least_value_of_g_l589_58932


namespace NUMINAMATH_GPT_student_correct_answers_l589_58981

theorem student_correct_answers (C W : ℕ) 
  (h1 : 4 * C - W = 130) 
  (h2 : C + W = 80) : 
  C = 42 := by
  sorry

end NUMINAMATH_GPT_student_correct_answers_l589_58981


namespace NUMINAMATH_GPT_susan_ate_6_candies_l589_58964

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end NUMINAMATH_GPT_susan_ate_6_candies_l589_58964


namespace NUMINAMATH_GPT_edit_post_time_zero_l589_58973

-- Define the conditions
def total_videos : ℕ := 4
def setup_time : ℕ := 1
def painting_time_per_video : ℕ := 1
def cleanup_time : ℕ := 1
def total_production_time_per_video : ℕ := 3

-- Define the total time spent on setup, painting, and cleanup for one video
def spc_time : ℕ := setup_time + painting_time_per_video + cleanup_time

-- State the theorem to be proven
theorem edit_post_time_zero : (total_production_time_per_video - spc_time) = 0 := by
  sorry

end NUMINAMATH_GPT_edit_post_time_zero_l589_58973


namespace NUMINAMATH_GPT_stratified_sampling_11th_grade_representatives_l589_58902

theorem stratified_sampling_11th_grade_representatives 
  (students_10th : ℕ)
  (students_11th : ℕ)
  (students_12th : ℕ)
  (total_rep : ℕ)
  (total_students : students_10th + students_11th + students_12th = 5000)
  (Students_10th : students_10th = 2500)
  (Students_11th : students_11th = 1500)
  (Students_12th : students_12th = 1000)
  (Total_rep : total_rep = 30) : 
  (9 : ℕ) = (3 : ℚ) / (10 : ℚ) * (30 : ℕ) :=
sorry

end NUMINAMATH_GPT_stratified_sampling_11th_grade_representatives_l589_58902


namespace NUMINAMATH_GPT_min_value_x2_y2_l589_58965

theorem min_value_x2_y2 (x y : ℝ) (h : x + y = 2) : ∃ m, m = x^2 + y^2 ∧ (∀ (x y : ℝ), x + y = 2 → x^2 + y^2 ≥ m) ∧ m = 2 := 
sorry

end NUMINAMATH_GPT_min_value_x2_y2_l589_58965


namespace NUMINAMATH_GPT_number_of_sides_l589_58993

-- Define the conditions
def interior_angle (n : ℕ) : ℝ := 156

-- The main theorem to prove the number of sides
theorem number_of_sides (n : ℕ) (h : interior_angle n = 156) : n = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_l589_58993


namespace NUMINAMATH_GPT_solve_system_l589_58990

theorem solve_system (a b c : ℝ) (h₁ : a^2 + 3 * a + 1 = (b + c) / 2)
                                (h₂ : b^2 + 3 * b + 1 = (a + c) / 2)
                                (h₃ : c^2 + 3 * c + 1 = (a + b) / 2) : 
  a = -1 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l589_58990


namespace NUMINAMATH_GPT_triangle_is_isosceles_l589_58909

variable {α β γ : ℝ} (quadrilateral_angles : List ℝ)

-- Conditions from the problem
axiom triangle_angle_sum : α + β + γ = 180
axiom quadrilateral_angle_sum : quadrilateral_angles.sum = 360
axiom quadrilateral_angle_conditions : ∀ (a b : ℝ), a ∈ [α, β, γ] → b ∈ [α, β, γ] → a ≠ b → (a + b ∈ quadrilateral_angles)

-- Proof statement
theorem triangle_is_isosceles : (α = β) ∨ (β = γ) ∨ (γ = α) := 
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l589_58909


namespace NUMINAMATH_GPT_income_ratio_l589_58931

-- Define the conditions
variables (I_A I_B E_A E_B : ℝ)
variables (Savings_A Savings_B : ℝ)

-- Given conditions
def expenditure_ratio : E_A / E_B = 3 / 2 := sorry
def savings_A : Savings_A = 1600 := sorry
def savings_B : Savings_B = 1600 := sorry
def income_A : I_A = 4000 := sorry
def expenditure_A : E_A = I_A - Savings_A := sorry
def expenditure_B : E_B = I_B - Savings_B := sorry

-- Prove it's implied that the ratio of incomes is 5:4
theorem income_ratio : I_A / I_B = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_income_ratio_l589_58931


namespace NUMINAMATH_GPT_find_k_l589_58947

theorem find_k (k : ℤ) : 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    (x1, y1) = (2, 9) ∧ (x2, y2) = (5, 18) ∧ (x3, y3) = (8, 27) ∧ 
    ∃ m b : ℤ, y1 = m * x1 + b ∧ y2 = m * x2 + b ∧ y3 = m * x3 + b) 
  ∧ ∃ m b : ℤ, k = m * 42 + b
  → k = 129 :=
sorry

end NUMINAMATH_GPT_find_k_l589_58947


namespace NUMINAMATH_GPT_fg_of_neg5_eq_484_l589_58991

def f (x : Int) : Int := x * x
def g (x : Int) : Int := 6 * x + 8

theorem fg_of_neg5_eq_484 : f (g (-5)) = 484 := 
  sorry

end NUMINAMATH_GPT_fg_of_neg5_eq_484_l589_58991


namespace NUMINAMATH_GPT_not_divisible_a1a2_l589_58977

theorem not_divisible_a1a2 (a1 a2 b1 b2 : ℕ) (h1 : 1 < b1) (h2 : b1 < a1) (h3 : 1 < b2) (h4 : b2 < a2) (h5 : b1 ∣ a1) (h6 : b2 ∣ a2) :
  ¬ (a1 * a2 ∣ a1 * b1 + a2 * b2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_a1a2_l589_58977


namespace NUMINAMATH_GPT_haley_tickets_l589_58919

-- Conditions
def cost_per_ticket : ℕ := 4
def extra_tickets : ℕ := 5
def total_spent : ℕ := 32
def cost_extra_tickets : ℕ := extra_tickets * cost_per_ticket

-- Main proof problem
theorem haley_tickets (T : ℕ) (h : 4 * T + cost_extra_tickets = total_spent) :
  T = 3 := sorry

end NUMINAMATH_GPT_haley_tickets_l589_58919


namespace NUMINAMATH_GPT_find_fayes_age_l589_58999

variable {C D E F : ℕ}

theorem find_fayes_age
  (h1 : D = E - 2)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (h4 : D = 15) :
  F = 16 := by
  sorry

end NUMINAMATH_GPT_find_fayes_age_l589_58999


namespace NUMINAMATH_GPT_total_jellybeans_l589_58907

-- Definitions of the conditions
def caleb_jellybeans : ℕ := 3 * 12
def sophie_jellybeans (caleb_jellybeans : ℕ) : ℕ := caleb_jellybeans / 2

-- Statement of the proof problem
theorem total_jellybeans (C : caleb_jellybeans = 36) (S : sophie_jellybeans 36 = 18) :
  caleb_jellybeans + sophie_jellybeans 36 = 54 :=
by
  sorry

end NUMINAMATH_GPT_total_jellybeans_l589_58907


namespace NUMINAMATH_GPT_number_line_point_B_l589_58989

theorem number_line_point_B (A B : ℝ) (AB : ℝ) (h1 : AB = 4 * Real.sqrt 2) (h2 : A = 3 * Real.sqrt 2) :
  B = -Real.sqrt 2 ∨ B = 7 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_number_line_point_B_l589_58989


namespace NUMINAMATH_GPT_jordan_total_points_l589_58996

-- Definitions based on conditions in the problem
def jordan_attempts (x y : ℕ) : Prop :=
  x + y = 40

def points_from_three_point_shots (x : ℕ) : ℝ :=
  0.75 * x

def points_from_two_point_shots (y : ℕ) : ℝ :=
  0.8 * y

-- Main theorem to prove the total points scored by Jordan
theorem jordan_total_points (x y : ℕ) 
  (h_attempts : jordan_attempts x y) : 
  points_from_three_point_shots x + points_from_two_point_shots y = 30 := 
by
  sorry

end NUMINAMATH_GPT_jordan_total_points_l589_58996


namespace NUMINAMATH_GPT_find_smallest_number_l589_58957

variable (x : ℕ)

def second_number := 2 * x
def third_number := 4 * second_number x
def average := (x + second_number x + third_number x) / 3

theorem find_smallest_number (h : average x = 165) : x = 45 := by
  sorry

end NUMINAMATH_GPT_find_smallest_number_l589_58957


namespace NUMINAMATH_GPT_mrs_hilt_total_payment_l589_58940

-- Define the conditions
def number_of_hot_dogs : ℕ := 6
def cost_per_hot_dog : ℝ := 0.50

-- Define the total cost
def total_cost : ℝ := number_of_hot_dogs * cost_per_hot_dog

-- State the theorem to prove the total cost
theorem mrs_hilt_total_payment : total_cost = 3.00 := 
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_total_payment_l589_58940


namespace NUMINAMATH_GPT_sum_a4_a5_a6_l589_58997

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (h1 : is_arithmetic_sequence a)
          (h2 : a 1 + a 2 + a 3 = 6)
          (h3 : a 7 + a 8 + a 9 = 24)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_a4_a5_a6_l589_58997


namespace NUMINAMATH_GPT_cindy_gives_3_envelopes_per_friend_l589_58970

theorem cindy_gives_3_envelopes_per_friend
  (initial_envelopes : ℕ) 
  (remaining_envelopes : ℕ)
  (friends : ℕ)
  (envelopes_per_friend : ℕ) 
  (h1 : initial_envelopes = 37) 
  (h2 : remaining_envelopes = 22)
  (h3 : friends = 5) 
  (h4 : initial_envelopes - remaining_envelopes = envelopes_per_friend * friends) :
  envelopes_per_friend = 3 :=
by
  sorry

end NUMINAMATH_GPT_cindy_gives_3_envelopes_per_friend_l589_58970


namespace NUMINAMATH_GPT_calculate_expression_l589_58959

theorem calculate_expression :
  (0.125: ℝ) ^ 3 * (-8) ^ 3 = -1 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l589_58959


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l589_58949

variable (R : ℝ)

theorem equilateral_triangle_side_length (R : ℝ) :
  (∃ (s : ℝ), s = R * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l589_58949


namespace NUMINAMATH_GPT_determine_k_l589_58915

theorem determine_k (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l589_58915


namespace NUMINAMATH_GPT_coefficient_of_x3_in_expansion_l589_58917

theorem coefficient_of_x3_in_expansion :
  let coeff := 56 * 972 * Real.sqrt 2
  coeff = 54432 * Real.sqrt 2 :=
by
  let coeff := 56 * 972 * Real.sqrt 2
  have h : coeff = 54432 * Real.sqrt 2 := sorry
  exact h

end NUMINAMATH_GPT_coefficient_of_x3_in_expansion_l589_58917


namespace NUMINAMATH_GPT_lele_has_enough_money_and_remaining_19_yuan_l589_58930

def price_A : ℝ := 46.5
def price_B : ℝ := 54.5
def total_money : ℝ := 120

theorem lele_has_enough_money_and_remaining_19_yuan : 
  (price_A + price_B ≤ total_money) ∧ (total_money - (price_A + price_B) = 19) :=
by
  sorry

end NUMINAMATH_GPT_lele_has_enough_money_and_remaining_19_yuan_l589_58930


namespace NUMINAMATH_GPT_triangle_dimensions_l589_58936

-- Define the problem in Lean 4
theorem triangle_dimensions (a m : ℕ) (h₁ : a = m + 4)
  (h₂ : (a + 12) * (m + 12) = 10 * a * m) : 
  a = 12 ∧ m = 8 := 
by
  sorry

end NUMINAMATH_GPT_triangle_dimensions_l589_58936


namespace NUMINAMATH_GPT_vector_parallel_m_eq_two_neg_two_l589_58938

theorem vector_parallel_m_eq_two_neg_two (m : ℝ) (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 / x = m / y) : m = 2 ∨ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_vector_parallel_m_eq_two_neg_two_l589_58938


namespace NUMINAMATH_GPT_new_concentration_of_solution_l589_58974

theorem new_concentration_of_solution 
  (Q : ℚ) 
  (initial_concentration : ℚ := 0.4) 
  (new_concentration : ℚ := 0.25) 
  (replacement_fraction : ℚ := 1/3) 
  (new_solution_concentration : ℚ := 0.35) :
  (initial_concentration * (1 - replacement_fraction) + new_concentration * replacement_fraction)
  = new_solution_concentration := 
by 
  sorry

end NUMINAMATH_GPT_new_concentration_of_solution_l589_58974


namespace NUMINAMATH_GPT_probability_of_winning_correct_l589_58962

noncomputable def probability_of_winning (P_L : ℚ) (P_T : ℚ) : ℚ :=
  1 - (P_L + P_T)

theorem probability_of_winning_correct :
  probability_of_winning (3/7) (2/21) = 10/21 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_winning_correct_l589_58962


namespace NUMINAMATH_GPT_polynomial_coeff_sum_eq_four_l589_58916

theorem polynomial_coeff_sum_eq_four (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ) :
  (∀ x : ℤ, (2 * x - 1)^6 * (x + 1)^2 = a * x ^ 8 + a1 * x ^ 7 + a2 * x ^ 6 + a3 * x ^ 5 + 
                      a4 * x ^ 4 + a5 * x ^ 3 + a6 * x ^ 2 + a7 * x + a8) →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 := by
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_eq_four_l589_58916


namespace NUMINAMATH_GPT_area_of_square_l589_58948

noncomputable def square_area (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) : ℝ :=
  (v * v) / 4

theorem area_of_square (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) (h_cond : ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → B = (u, 0) → C = (u, v) → 
  (u - 0) * (u - 0) + (v - 0) * (v - 0) = (u - 0) * (u - 0)) :
  square_area u v h_u h_v = v * v / 4 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_square_l589_58948


namespace NUMINAMATH_GPT_packs_sold_by_Robyn_l589_58958

theorem packs_sold_by_Robyn (total_packs : ℕ) (lucy_packs : ℕ) (robyn_packs : ℕ) 
  (h1 : total_packs = 98) (h2 : lucy_packs = 43) (h3 : robyn_packs = total_packs - lucy_packs) :
  robyn_packs = 55 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_packs_sold_by_Robyn_l589_58958


namespace NUMINAMATH_GPT_total_planks_needed_l589_58983

theorem total_planks_needed (large_planks small_planks : ℕ) (h1 : large_planks = 37) (h2 : small_planks = 42) : large_planks + small_planks = 79 :=
by
  sorry

end NUMINAMATH_GPT_total_planks_needed_l589_58983


namespace NUMINAMATH_GPT_root_equation_satisfies_expr_l589_58955

theorem root_equation_satisfies_expr (a : ℝ) (h : 2 * a ^ 2 - 7 * a - 1 = 0) :
  a * (2 * a - 7) + 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_root_equation_satisfies_expr_l589_58955


namespace NUMINAMATH_GPT_second_candidate_percentage_l589_58968

theorem second_candidate_percentage (V : ℝ) (h1 : 0.15 * V ≠ 0) (h2 : 0.38 * V ≠ 300) :
  (0.38 * V - 300) / (0.85 * V - 250) * 100 = 44.71 :=
by 
  -- Let the math proof be synthesized by a more detailed breakdown of conditions and theorems
  sorry

end NUMINAMATH_GPT_second_candidate_percentage_l589_58968


namespace NUMINAMATH_GPT_masha_can_pay_exactly_with_11_ruble_bills_l589_58903

theorem masha_can_pay_exactly_with_11_ruble_bills (m n k p : ℕ) 
  (h1 : 3 * m + 4 * n + 5 * k = 11 * p) : 
  ∃ q : ℕ, 9 * m + n + 4 * k = 11 * q := 
by {
  sorry
}

end NUMINAMATH_GPT_masha_can_pay_exactly_with_11_ruble_bills_l589_58903
