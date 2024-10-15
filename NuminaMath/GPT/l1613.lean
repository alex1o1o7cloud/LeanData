import Mathlib

namespace NUMINAMATH_GPT_player_placing_third_won_against_seventh_l1613_161339

theorem player_placing_third_won_against_seventh :
  ∃ (s : Fin 8 → ℚ),
    -- Condition 1: Scores are different
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    -- Condition 2: Second place score equals the sum of the bottom four scores
    (s 1 = s 4 + s 5 + s 6 + s 7) ∧
    -- Result: Third player won against the seventh player
    (s 2 > s 6) :=
sorry

end NUMINAMATH_GPT_player_placing_third_won_against_seventh_l1613_161339


namespace NUMINAMATH_GPT_image_center_after_reflection_and_translation_l1613_161303

def circle_center_before_translation : ℝ × ℝ := (3, -4)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-x, y)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x, y + d)

theorem image_center_after_reflection_and_translation :
  translate_up (reflect_y_axis circle_center_before_translation) 5 = (-3, 1) :=
by
  -- The detail proof goes here.
  sorry

end NUMINAMATH_GPT_image_center_after_reflection_and_translation_l1613_161303


namespace NUMINAMATH_GPT_mike_spent_on_mower_blades_l1613_161316

theorem mike_spent_on_mower_blades (x : ℝ) 
  (initial_money : ℝ := 101) 
  (cost_of_games : ℝ := 54) 
  (games : ℝ := 9) 
  (price_per_game : ℝ := 6) 
  (h1 : 101 - x = 54) :
  x = 47 := 
by
  sorry

end NUMINAMATH_GPT_mike_spent_on_mower_blades_l1613_161316


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l1613_161356

-- Problem 1
theorem problem1_solution (x : ℝ) : (2 * x - 3) * (x + 1) < 0 ↔ (-1 < x) ∧ (x < 3 / 2) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) : (4 * x - 1) / (x + 2) ≥ 0 ↔ (x < -2) ∨ (x >= 1 / 4) :=
sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l1613_161356


namespace NUMINAMATH_GPT_susan_spent_total_l1613_161360

-- Definitions for the costs and quantities
def pencil_cost : ℝ := 0.25
def pen_cost : ℝ := 0.80
def total_items : ℕ := 36
def pencils_bought : ℕ := 16

-- Question: How much did Susan spend?
theorem susan_spent_total : (pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought)) = 20 :=
by
    -- definition goes here
    sorry

end NUMINAMATH_GPT_susan_spent_total_l1613_161360


namespace NUMINAMATH_GPT_num_children_proof_l1613_161399

noncomputable def number_of_children (total_persons : ℕ) (total_revenue : ℕ) (adult_price : ℕ) (child_price : ℕ) : ℕ :=
  let adult_tickets := (child_price * total_persons - total_revenue) / (child_price - adult_price)
  let child_tickets := total_persons - adult_tickets
  child_tickets

theorem num_children_proof : number_of_children 280 14000 60 25 = 80 := 
by
  unfold number_of_children
  sorry

end NUMINAMATH_GPT_num_children_proof_l1613_161399


namespace NUMINAMATH_GPT_original_rectangle_area_is_56_l1613_161361

-- Conditions
def original_rectangle_perimeter := 30 -- cm
def smaller_rectangle_perimeter := 16 -- cm
def side_length_square := (original_rectangle_perimeter - smaller_rectangle_perimeter) / 2 -- Using the reduction logic

-- Computing the length and width of the original rectangle.
def width_original_rectangle := side_length_square
def length_original_rectangle := smaller_rectangle_perimeter / 2

-- The goal is to prove that the area of the original rectangle is 56 cm^2.

theorem original_rectangle_area_is_56:
  (length_original_rectangle - width_original_rectangle + width_original_rectangle) = 8 -- finding the length
  ∧ (length_original_rectangle * width_original_rectangle) = 56 := by
  sorry

end NUMINAMATH_GPT_original_rectangle_area_is_56_l1613_161361


namespace NUMINAMATH_GPT_simplify_expression_l1613_161355

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 7) + 2 = -x^4 + 3 * x^3 - 6 * x^2 + 7 * x + 2 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1613_161355


namespace NUMINAMATH_GPT_number_of_chairs_is_40_l1613_161331

-- Define the conditions
variables (C : ℕ) -- Total number of chairs
variables (capacity_per_chair : ℕ := 2) -- Each chair's capacity is 2 people
variables (occupied_ratio : ℚ := 3 / 5) -- Ratio of occupied chairs
variables (attendees : ℕ := 48) -- Number of attendees

theorem number_of_chairs_is_40
  (h1 : ∀ c : ℕ, capacity_per_chair * c = attendees)
  (h2 : occupied_ratio * C * capacity_per_chair = attendees) : 
  C = 40 := sorry

end NUMINAMATH_GPT_number_of_chairs_is_40_l1613_161331


namespace NUMINAMATH_GPT_problem_statement_l1613_161323

noncomputable def distance_from_line_to_point (a b : ℝ) : ℝ :=
  abs (1 / 2) / (Real.sqrt (a ^ 2 + b ^ 2))

theorem problem_statement (a b : ℝ) (h1 : a = (1 - 2 * b) / 2) (h2 : b = 1 / 2 - a) :
  distance_from_line_to_point a b ≤ Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1613_161323


namespace NUMINAMATH_GPT_arithmetic_sequence_seventy_fifth_term_l1613_161313

theorem arithmetic_sequence_seventy_fifth_term:
  ∀ (a₁ a₂ d : ℕ), a₁ = 3 → a₂ = 51 → a₂ = a₁ + 24 * d → (3 + 74 * d) = 151 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_seventy_fifth_term_l1613_161313


namespace NUMINAMATH_GPT_value_range_of_f_l1613_161396

-- Define the function f(x) = 2x - x^2
def f (x : ℝ) : ℝ := 2 * x - x^2

-- State the theorem with the given conditions and prove the correct answer
theorem value_range_of_f :
  (∀ y : ℝ, -3 ≤ y ∧ y ≤ 1 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = y) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → -3 ≤ f x ∧ f x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_value_range_of_f_l1613_161396


namespace NUMINAMATH_GPT_intersection_equality_l1613_161376

def setA := {x : ℝ | (x - 1) * (3 - x) < 0}
def setB := {x : ℝ | -3 ≤ x ∧ x ≤ 3}

theorem intersection_equality : setA ∩ setB = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_equality_l1613_161376


namespace NUMINAMATH_GPT_trig_identity_l1613_161371

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1613_161371


namespace NUMINAMATH_GPT_cuberoot_eq_3_implies_cube_eq_19683_l1613_161310

theorem cuberoot_eq_3_implies_cube_eq_19683 (x : ℝ) (h : (x + 6)^(1/3) = 3) : (x + 6)^3 = 19683 := by
  sorry

end NUMINAMATH_GPT_cuberoot_eq_3_implies_cube_eq_19683_l1613_161310


namespace NUMINAMATH_GPT__l1613_161319

noncomputable def polynomial_divides (x : ℂ) (n : ℕ) : Prop :=
  (x - 1) ^ 3 ∣ x ^ (2 * n + 1) - (2 * n + 1) * x ^ (n + 1) + (2 * n + 1) * x ^ n - 1

lemma polynomial_division_theorem : ∀ (n : ℕ), n ≥ 1 → ∀ (x : ℂ), polynomial_divides x n :=
by
  intros n hn x
  unfold polynomial_divides
  sorry

end NUMINAMATH_GPT__l1613_161319


namespace NUMINAMATH_GPT_students_with_uncool_family_l1613_161367

-- Define the conditions as given in the problem.
variables (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool_parents : ℕ)
          (cool_siblings : ℕ) (cool_siblings_and_dads : ℕ)

-- Provide the known values as conditions.
def problem_conditions := 
  total_students = 50 ∧
  cool_dads = 20 ∧
  cool_moms = 25 ∧
  both_cool_parents = 12 ∧
  cool_siblings = 5 ∧
  cool_siblings_and_dads = 3

-- State the problem: prove the number of students with all uncool family members.
theorem students_with_uncool_family : problem_conditions total_students cool_dads cool_moms 
                                            both_cool_parents cool_siblings cool_siblings_and_dads →
                                    (50 - ((20 - 12) + (25 - 12) + 12 + (5 - 3)) = 15) :=
by intros h; cases h; sorry

end NUMINAMATH_GPT_students_with_uncool_family_l1613_161367


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1613_161350

section
variables (a b : ℝ)

theorem simplify_expression1 : -b*(2*a - b) + (a + b)^2 = a^2 + 2*b^2 :=
sorry
end

section
variables (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2)

theorem simplify_expression2 : (1 - (x/(2 + x))) / ((x^2 - 4)/(x^2 + 4*x + 4)) = 2/(x - 2) :=
sorry
end

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1613_161350


namespace NUMINAMATH_GPT_problem_l1613_161305

variable (a b c : ℝ)

theorem problem (h : a^2 * b^2 + 18 * a * b * c > 4 * b^3 + 4 * a^3 * c + 27 * c^2) : a^2 > 3 * b :=
by
  sorry

end NUMINAMATH_GPT_problem_l1613_161305


namespace NUMINAMATH_GPT_total_number_of_flowers_is_correct_l1613_161382

-- Define the conditions
def number_of_pots : ℕ := 544
def flowers_per_pot : ℕ := 32
def total_flowers : ℕ := number_of_pots * flowers_per_pot

-- State the theorem to be proved
theorem total_number_of_flowers_is_correct :
  total_flowers = 17408 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_flowers_is_correct_l1613_161382


namespace NUMINAMATH_GPT_find_a_values_for_eccentricity_l1613_161383

theorem find_a_values_for_eccentricity (a : ℝ) : 
  ( ∃ a : ℝ, ((∀ x y : ℝ, (x^2 / (a+8) + y^2 / 9 = 1)) ∧ (e = 1/2) ) 
  → (a = 4 ∨ a = -5/4)) := 
sorry

end NUMINAMATH_GPT_find_a_values_for_eccentricity_l1613_161383


namespace NUMINAMATH_GPT_find_C_probability_within_r_l1613_161314

noncomputable def probability_density (x y R : ℝ) (C : ℝ) : ℝ :=
if x^2 + y^2 <= R^2 then C * (R - Real.sqrt (x^2 + y^2)) else 0

noncomputable def total_integral (R : ℝ) (C : ℝ) : ℝ :=
∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C

theorem find_C (R : ℝ) (hR : 0 < R) : 
  (∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C) = 1 ↔ 
  C = 3 / (π * R^3) := 
by 
  sorry

theorem probability_within_r (R r : ℝ) 
  (hR : 0 < R) (hr : 0 < r) (hrR : r <= R) (P : ℝ) : 
  (∫ (x : ℝ) in -r..r, ∫ (y : ℝ) in -r..r, probability_density x y R (3 / (π * R^3))) = P ↔ 
  (R = 2 ∧ r = 1 → P = 1 / 2) := 
by 
  sorry

end NUMINAMATH_GPT_find_C_probability_within_r_l1613_161314


namespace NUMINAMATH_GPT_BURN_maps_to_8615_l1613_161344

open List Function

def tenLetterMapping : List (Char × Nat) := 
  [('G', 0), ('R', 1), ('E', 2), ('A', 3), ('T', 4), ('N', 5), ('U', 6), ('M', 7), ('B', 8), ('S', 9)]

def charToDigit (c : Char) : Option Nat :=
  tenLetterMapping.lookup c

def wordToNumber (word : List Char) : Option (List Nat) :=
  word.mapM charToDigit 

theorem BURN_maps_to_8615 :
  wordToNumber ['B', 'U', 'R', 'N'] = some [8, 6, 1, 5] :=
by
  sorry

end NUMINAMATH_GPT_BURN_maps_to_8615_l1613_161344


namespace NUMINAMATH_GPT_beavers_still_working_is_one_l1613_161345

def initial_beavers : Nat := 2
def beavers_swimming : Nat := 1
def still_working_beavers : Nat := initial_beavers - beavers_swimming

theorem beavers_still_working_is_one : still_working_beavers = 1 :=
by
  sorry

end NUMINAMATH_GPT_beavers_still_working_is_one_l1613_161345


namespace NUMINAMATH_GPT_max_value_of_3x_plus_4y_on_curve_C_l1613_161311

theorem max_value_of_3x_plus_4y_on_curve_C :
  ∀ (x y : ℝ),
  (∃ (ρ θ : ℝ), ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (P : ℝ × ℝ) →
  (P = (x, y)) →
  3 * x + 4 * y ≤ Real.sqrt 145 ∧ ∃ (α : ℝ), 0 ≤ α ∧ α < 2 * Real.pi ∧ 3 * x + 4 * y = Real.sqrt 145 := 
by
  intros x y h_exists P hP
  sorry

end NUMINAMATH_GPT_max_value_of_3x_plus_4y_on_curve_C_l1613_161311


namespace NUMINAMATH_GPT_enhanced_inequality_l1613_161301

theorem enhanced_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a^2 / (b + c) + 2 * b^2 / (c + a) + 2 * c^2 / (a + b) ≥ a + b + c + (2 * a - b - c)^2 / (a + b + c)) :=
sorry

end NUMINAMATH_GPT_enhanced_inequality_l1613_161301


namespace NUMINAMATH_GPT_solve_for_a_l1613_161308

open Set

theorem solve_for_a (a : ℝ) :
  let M := ({a^2, a + 1, -3} : Set ℝ)
  let P := ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ)
  M ∩ P = {-3} →
  a = -1 :=
by
  intros M P h
  have hM : M = {a^2, a + 1, -3} := rfl
  have hP : P = {a - 3, 2 * a - 1, a^2 + 1} := rfl
  rw [hM, hP] at h
  sorry

end NUMINAMATH_GPT_solve_for_a_l1613_161308


namespace NUMINAMATH_GPT_simplify_expression_l1613_161358

theorem simplify_expression : 
  (4 * 6 / (12 * 14)) * (8 * 12 * 14 / (4 * 6 * 8)) = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1613_161358


namespace NUMINAMATH_GPT_intersection_eq_l1613_161368

-- Define sets P and Q
def setP := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def setQ := {y : ℝ | ∃ x : ℝ, y = -x + 2}

-- The main theorem statement
theorem intersection_eq: setP ∩ setQ = {y : ℝ | y ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1613_161368


namespace NUMINAMATH_GPT_delta_y_over_delta_x_l1613_161336

def curve (x : ℝ) : ℝ := x^2 + x

theorem delta_y_over_delta_x (Δx Δy : ℝ) 
  (hQ : (2 + Δx, 6 + Δy) = (2 + Δx, curve (2 + Δx)))
  (hP : 6 = curve 2) : 
  (Δy / Δx) = Δx + 5 :=
by
  sorry

end NUMINAMATH_GPT_delta_y_over_delta_x_l1613_161336


namespace NUMINAMATH_GPT_range_of_a_l1613_161369

noncomputable def f (a x : ℝ) := a * x - 1
noncomputable def g (x : ℝ) := -x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ (Set.Icc (-1 : ℝ) 1) → ∃ (x2 : ℝ), x2 ∈ (Set.Icc (0 : ℝ) 2) ∧ f a x1 < g x2) ↔ a ∈ Set.Ioo (-3 : ℝ) 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1613_161369


namespace NUMINAMATH_GPT_probability_of_two_white_balls_l1613_161309

-- Define the total number of balls
def total_balls : ℕ := 11

-- Define the number of white balls
def white_balls : ℕ := 5

-- Define the number of ways to choose 2 out of n (combinations)
def choose (n r : ℕ) : ℕ := n.choose r

-- Define the total combinations of drawing 2 balls out of 11
def total_combinations : ℕ := choose total_balls 2

-- Define the combinations of drawing 2 white balls out of 5
def white_combinations : ℕ := choose white_balls 2

-- Define the probability of drawing 2 white balls
noncomputable def probability_white : ℚ := (white_combinations : ℚ) / (total_combinations : ℚ)

-- Now, state the theorem that states the desired result
theorem probability_of_two_white_balls : probability_white = 2 / 11 := sorry

end NUMINAMATH_GPT_probability_of_two_white_balls_l1613_161309


namespace NUMINAMATH_GPT_one_add_i_cubed_eq_one_sub_i_l1613_161335

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_GPT_one_add_i_cubed_eq_one_sub_i_l1613_161335


namespace NUMINAMATH_GPT_negation_of_p_l1613_161395

open Real

-- Define the original proposition p
def p := ∀ x : ℝ, 0 < x → x^2 > log x

-- State the theorem with its negation
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 0 < x ∧ x^2 ≤ log x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1613_161395


namespace NUMINAMATH_GPT_plates_probability_l1613_161393

noncomputable def number_of_plates := 12
noncomputable def red_plates := 6
noncomputable def light_blue_plates := 3
noncomputable def dark_blue_plates := 3
noncomputable def total_pairs := number_of_plates * (number_of_plates - 1) / 2
noncomputable def red_pairs := red_plates * (red_plates - 1) / 2
noncomputable def light_blue_pairs := light_blue_plates * (light_blue_plates - 1) / 2
noncomputable def dark_blue_pairs := dark_blue_plates * (dark_blue_plates - 1) / 2
noncomputable def mixed_blue_pairs := light_blue_plates * dark_blue_plates
noncomputable def total_satisfying_pairs := red_pairs + light_blue_pairs + dark_blue_pairs + mixed_blue_pairs
noncomputable def desired_probability := (total_satisfying_pairs : ℚ) / total_pairs

theorem plates_probability :
  desired_probability = 5 / 11 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_plates_probability_l1613_161393


namespace NUMINAMATH_GPT_third_factor_of_product_l1613_161329

theorem third_factor_of_product (w : ℕ) (h_w_pos : w > 0) (h_w_168 : w = 168)
  (w_factors : (936 * w) = 2^5 * 3^3 * x)
  (h36_factors : 2^5 ∣ (936 * w)) (h33_factors : 3^3 ∣ (936 * w)) : 
  (936 * w) / (2^5 * 3^3) = 182 :=
by {
  -- This is a placeholder. The actual proof is omitted.
  sorry
}

end NUMINAMATH_GPT_third_factor_of_product_l1613_161329


namespace NUMINAMATH_GPT_minimum_bail_rate_l1613_161348

theorem minimum_bail_rate 
  (distance : ℝ)
  (leak_rate : ℝ)
  (max_water : ℝ)
  (rowing_speed : ℝ)
  (bail_rate : ℝ)
  (time_to_shore : ℝ) :
  distance = 2 ∧
  leak_rate = 15 ∧
  max_water = 60 ∧
  rowing_speed = 3 ∧
  time_to_shore = distance / rowing_speed * 60 →
  bail_rate = (leak_rate * time_to_shore - max_water) / time_to_shore →
  bail_rate = 13.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_minimum_bail_rate_l1613_161348


namespace NUMINAMATH_GPT_cost_price_of_cricket_bat_for_A_l1613_161330

-- Define the cost price of the cricket bat for A as a variable
variable (CP_A : ℝ)

-- Define the conditions given in the problem
def condition1 := CP_A * 1.20 -- B buys at 20% profit
def condition2 := CP_A * 1.20 * 1.25 -- B sells at 25% profit
def totalCost := 231 -- C pays $231

-- The theorem we need to prove
theorem cost_price_of_cricket_bat_for_A : (condition2 = totalCost) → CP_A = 154 := by
  intros h
  sorry

end NUMINAMATH_GPT_cost_price_of_cricket_bat_for_A_l1613_161330


namespace NUMINAMATH_GPT_smallest_n_l1613_161342

theorem smallest_n (n : ℕ) (h1 : n ≥ 1)
  (h2 : ∃ k : ℕ, 2002 * n = k ^ 3)
  (h3 : ∃ m : ℕ, n = 2002 * m ^ 2) :
  n = 2002^5 := sorry

end NUMINAMATH_GPT_smallest_n_l1613_161342


namespace NUMINAMATH_GPT_complex_quadrant_l1613_161391

theorem complex_quadrant (i : ℂ) (hi : i * i = -1) (z : ℂ) (hz : z = 1 / (1 - i)) : 
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l1613_161391


namespace NUMINAMATH_GPT_sufficient_conditions_for_quadratic_l1613_161325

theorem sufficient_conditions_for_quadratic (x : ℝ) : 
  (0 < x ∧ x < 4) ∨ (-2 < x ∧ x < 4) ∨ (-2 < x ∧ x < 3) → x^2 - 2*x - 8 < 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_conditions_for_quadratic_l1613_161325


namespace NUMINAMATH_GPT_scientific_notation_l1613_161359

theorem scientific_notation (n : ℝ) (h : n = 40.9 * 10^9) : n = 4.09 * 10^10 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_l1613_161359


namespace NUMINAMATH_GPT_work_problem_l1613_161318

-- Definition of the conditions and the problem statement
theorem work_problem (P D : ℕ)
  (h1 : ∀ (P : ℕ), ∀ (D : ℕ), (2 * P) * 6 = P * D * 1 / 2) : 
  D = 24 :=
by
  sorry

end NUMINAMATH_GPT_work_problem_l1613_161318


namespace NUMINAMATH_GPT_probability_penny_nickel_heads_l1613_161374

noncomputable def num_outcomes : ℕ := 2^4
noncomputable def num_successful_outcomes : ℕ := 2 * 2

theorem probability_penny_nickel_heads :
  (num_successful_outcomes : ℚ) / num_outcomes = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_penny_nickel_heads_l1613_161374


namespace NUMINAMATH_GPT_sugar_already_put_in_l1613_161387

-- Define the conditions
def totalSugarRequired : Nat := 14
def sugarNeededToAdd : Nat := 12
def sugarAlreadyPutIn (total : Nat) (needed : Nat) : Nat := total - needed

--State the theorem
theorem sugar_already_put_in :
  sugarAlreadyPutIn totalSugarRequired sugarNeededToAdd = 2 := 
  by
    -- Providing 'sorry' as a placeholder for the actual proof
    sorry

end NUMINAMATH_GPT_sugar_already_put_in_l1613_161387


namespace NUMINAMATH_GPT_number_of_students_speaking_two_languages_l1613_161315

variables (G H M GH GM HM GHM N : ℕ)

def students_speaking_two_languages (G H M GH GM HM GHM N : ℕ) : ℕ :=
  G + H + M - (GH + GM + HM) + GHM

theorem number_of_students_speaking_two_languages 
  (h_total : N = 22)
  (h_G : G = 6)
  (h_H : H = 15)
  (h_M : M = 6)
  (h_GHM : GHM = 1)
  (h_students : N = students_speaking_two_languages G H M GH GM HM GHM N): 
  GH + GM + HM = 6 := 
by 
  unfold students_speaking_two_languages at h_students 
  sorry

end NUMINAMATH_GPT_number_of_students_speaking_two_languages_l1613_161315


namespace NUMINAMATH_GPT_instantaneous_velocity_at_2_l1613_161352

def displacement (t : ℝ) : ℝ := 100 * t - 5 * t^2

noncomputable def instantaneous_velocity_at (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  (deriv s) t

theorem instantaneous_velocity_at_2 : instantaneous_velocity_at displacement 2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_2_l1613_161352


namespace NUMINAMATH_GPT_final_percentage_of_alcohol_l1613_161375

theorem final_percentage_of_alcohol (initial_volume : ℝ) (initial_alcohol_percentage : ℝ)
  (removed_alcohol : ℝ) (added_water : ℝ) :
  initial_volume = 15 → initial_alcohol_percentage = 25 →
  removed_alcohol = 2 → added_water = 3 →
  ( ( (initial_alcohol_percentage / 100 * initial_volume - removed_alcohol) / 
    (initial_volume - removed_alcohol + added_water) ) * 100 = 10.9375) :=
by
  intros
  sorry

end NUMINAMATH_GPT_final_percentage_of_alcohol_l1613_161375


namespace NUMINAMATH_GPT_train_length_approx_l1613_161370

noncomputable def length_of_train (distance_km : ℝ) (time_min : ℝ) (time_sec : ℝ) : ℝ :=
  let distance_m := distance_km * 1000 -- Convert km to meters
  let time_s := time_min * 60 -- Convert min to seconds
  let speed := distance_m / time_s -- Speed in meters/second
  speed * time_sec -- Length of train in meters

theorem train_length_approx :
  length_of_train 10 15 10 = 111.1 :=
by
  sorry

end NUMINAMATH_GPT_train_length_approx_l1613_161370


namespace NUMINAMATH_GPT_vehicle_distribution_l1613_161337

theorem vehicle_distribution :
  ∃ B T U : ℕ, 2 * B + 3 * T + U = 18 ∧ ∀ n : ℕ, n ≤ 18 → ∃ t : ℕ, ∃ (u : ℕ), 2 * (n - t) + u = 18 ∧ 2 * Nat.gcd t u + 3 * t + u = 18 ∧
  10 + 8 + 7 + 5 + 4 + 2 + 1 = 37 := by
  sorry

end NUMINAMATH_GPT_vehicle_distribution_l1613_161337


namespace NUMINAMATH_GPT_volume_larger_of_cube_cut_plane_l1613_161381

/-- Define the vertices and the midpoints -/
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def R : Point := ⟨0, 0, 0⟩
def X : Point := ⟨1, 2, 0⟩
def Y : Point := ⟨2, 0, 1⟩

/-- Equation of the plane passing through R, X and Y -/
def plane_eq (p : Point) : Prop :=
p.x - 2 * p.y - 2 * p.z = 0

/-- The volume of the larger of the two solids formed by cutting the cube with the plane -/
noncomputable def volume_larger_solid : ℝ :=
8 - (4/3 - (1/6))

/-- The statement for the given math problem -/
theorem volume_larger_of_cube_cut_plane :
  volume_larger_solid = 41/6 :=
by
  sorry

end NUMINAMATH_GPT_volume_larger_of_cube_cut_plane_l1613_161381


namespace NUMINAMATH_GPT_find_x_l1613_161307

theorem find_x :
  let a := 0.15
  let b := 0.06
  let c := 0.003375
  let d := 0.000216
  let e := 0.0225
  let f := 0.0036
  let g := 0.08999999999999998
  ∃ x, c - (d / e) + x + f = g →
  x = 0.092625 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1613_161307


namespace NUMINAMATH_GPT_condition_M_intersect_N_N_l1613_161334

theorem condition_M_intersect_N_N (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + (y - a)^2 ≤ 1 → y ≥ x^2)) ↔ (a ≥ 5 / 4) :=
sorry

end NUMINAMATH_GPT_condition_M_intersect_N_N_l1613_161334


namespace NUMINAMATH_GPT_B_subscribed_fraction_correct_l1613_161384

-- Define the total capital and the shares of A, C
variables (X : ℝ) (profit : ℝ) (A_share : ℝ) (C_share : ℝ)

-- Define the conditions as given in the problem
def A_capital_share := 1 / 3
def C_capital_share := 1 / 5
def total_profit := 2430
def A_profit_share := 810

-- Define the calculation of B's share
def B_capital_share := 1 - (A_capital_share + C_capital_share)

-- Define the expected correct answer for B's share
def expected_B_share := 7 / 15

-- Theorem statement
theorem B_subscribed_fraction_correct :
  B_capital_share = expected_B_share :=
by
  sorry

end NUMINAMATH_GPT_B_subscribed_fraction_correct_l1613_161384


namespace NUMINAMATH_GPT_total_spent_on_toys_l1613_161324

-- Definition of the costs
def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

-- The theorem to prove the total amount spent on toys
theorem total_spent_on_toys : football_cost + marbles_cost = 12.30 :=
by sorry

end NUMINAMATH_GPT_total_spent_on_toys_l1613_161324


namespace NUMINAMATH_GPT_coloring_connected_circles_diff_colors_l1613_161392

def num_ways_to_color_five_circles : ℕ :=
  36

theorem coloring_connected_circles_diff_colors (A B C D E : Type) (colors : Fin 3) 
  (connected : (A → B → C → D → E → Prop)) : num_ways_to_color_five_circles = 36 :=
by sorry

end NUMINAMATH_GPT_coloring_connected_circles_diff_colors_l1613_161392


namespace NUMINAMATH_GPT_trig_identity_l1613_161373

theorem trig_identity : Real.sin (35 * Real.pi / 6) + Real.cos (-11 * Real.pi / 3) = 0 := by
  sorry

end NUMINAMATH_GPT_trig_identity_l1613_161373


namespace NUMINAMATH_GPT_balance_balls_l1613_161353

theorem balance_balls (R O G B : ℝ) (h₁ : 4 * R = 8 * G) (h₂ : 3 * O = 6 * G) (h₃ : 8 * G = 6 * B) :
  3 * R + 2 * O + 4 * B = (46 / 3) * G :=
by
  -- Using the given conditions to derive intermediate results (included in the detailed proof, not part of the statement)
  sorry

end NUMINAMATH_GPT_balance_balls_l1613_161353


namespace NUMINAMATH_GPT_rectangle_area_given_perimeter_l1613_161362

theorem rectangle_area_given_perimeter (x : ℝ) (h_perim : 8 * x = 160) : (2 * x) * (2 * x) = 1600 := by
  -- Definitions derived from conditions
  let length := 2 * x
  let width := 2 * x
  -- Proof transformed to a Lean statement
  have h1 : length = 40 := by sorry
  have h2 : width = 40 := by sorry
  have h_area : length * width = 1600 := by sorry
  exact h_area

end NUMINAMATH_GPT_rectangle_area_given_perimeter_l1613_161362


namespace NUMINAMATH_GPT_salt_percentage_in_first_solution_l1613_161341

theorem salt_percentage_in_first_solution
    (S : ℝ)
    (h1 : ∀ w : ℝ, w ≥ 0 → ∃ q : ℝ, q = w)  -- One fourth of the first solution was replaced by the second solution
    (h2 : ∀ w1 w2 w3 : ℝ,
            w1 + w2 = w3 →
            (w1 / w3 * S + w2 / w3 * 25 = 16)) :  -- Resulting solution was 16 percent salt by weight
  S = 13 :=   -- Correct answer
sorry

end NUMINAMATH_GPT_salt_percentage_in_first_solution_l1613_161341


namespace NUMINAMATH_GPT_min_value_ge_54_l1613_161388

open Real

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) : ℝ :=
2 * x + 3 * y + 6 * z

theorem min_value_ge_54 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  min_value x y z h1 h2 h3 h4 ≥ 54 :=
sorry

end NUMINAMATH_GPT_min_value_ge_54_l1613_161388


namespace NUMINAMATH_GPT_centroid_of_triangle_l1613_161346

theorem centroid_of_triangle :
  let x1 := 9
  let y1 := -8
  let x2 := -5
  let y2 := 6
  let x3 := 4
  let y3 := -3
  ( (x1 + x2 + x3) / 3 = 8 / 3 ∧ (y1 + y2 + y3) / 3 = -5 / 3 ) :=
by
  let x1 := 9
  let y1 := -8
  let x2 := -5
  let y2 := 6
  let x3 := 4
  let y3 := -3
  have centroid_x : (x1 + x2 + x3) / 3 = 8 / 3 := sorry
  have centroid_y : (y1 + y2 + y3) / 3 = -5 / 3 := sorry
  exact ⟨centroid_x, centroid_y⟩

end NUMINAMATH_GPT_centroid_of_triangle_l1613_161346


namespace NUMINAMATH_GPT_expansion_of_expression_l1613_161354

theorem expansion_of_expression (x : ℝ) :
  let a := 15 * x^2 + 5 - 3 * x
  let b := 3 * x^3
  a * b = 45 * x^5 - 9 * x^4 + 15 * x^3 := by
  sorry

end NUMINAMATH_GPT_expansion_of_expression_l1613_161354


namespace NUMINAMATH_GPT_solve_linear_system_l1613_161312

variable {a b : ℝ}
variables {m n : ℝ}

theorem solve_linear_system
  (h1 : a * 2 - b * 1 = 3)
  (h2 : a * 2 + b * 1 = 5)
  (h3 : a * (m + 2 * n) - 2 * b * n = 6)
  (h4 : a * (m + 2 * n) + 2 * b * n = 10) :
  m = 2 ∧ n = 1 := 
sorry

end NUMINAMATH_GPT_solve_linear_system_l1613_161312


namespace NUMINAMATH_GPT_part1_part2_l1613_161306

theorem part1 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (1 / a^2 + 1 / b^2 ≥ 1 / 2) :=
sorry

theorem part2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = a * b ∧ (|2 * a - 1| + |3 * b - 1| = 2 * Real.sqrt 6 + 3)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1613_161306


namespace NUMINAMATH_GPT_chessboard_polygon_l1613_161328

-- Conditions
variable (A B a b : ℕ)

-- Statement of the theorem
theorem chessboard_polygon (A B a b : ℕ) : A - B = 4 * (a - b) :=
sorry

end NUMINAMATH_GPT_chessboard_polygon_l1613_161328


namespace NUMINAMATH_GPT_positive_m_for_one_root_l1613_161372

theorem positive_m_for_one_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_positive_m_for_one_root_l1613_161372


namespace NUMINAMATH_GPT_max_rectangle_area_l1613_161351

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 48) : x * y ≤ 144 :=
by
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l1613_161351


namespace NUMINAMATH_GPT_initial_walking_speed_l1613_161385

variable (v : ℝ)

theorem initial_walking_speed :
  (13.5 / v - 13.5 / 6 = 27 / 60) → v = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_walking_speed_l1613_161385


namespace NUMINAMATH_GPT_parallel_transitive_l1613_161332

-- Definition of parallel lines
def are_parallel (l1 l2 : Line) : Prop :=
  ∃ (P : Line), l1 = P ∧ l2 = P

-- Theorem stating that if two lines are parallel to the same line, then they are parallel to each other
theorem parallel_transitive (l1 l2 l3 : Line) (h1 : are_parallel l1 l3) (h2 : are_parallel l2 l3) :
  are_parallel l1 l2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_transitive_l1613_161332


namespace NUMINAMATH_GPT_negation_prop_l1613_161300

theorem negation_prop : (¬(∃ x : ℝ, x + 2 ≤ 0)) ↔ (∀ x : ℝ, x + 2 > 0) := 
  sorry

end NUMINAMATH_GPT_negation_prop_l1613_161300


namespace NUMINAMATH_GPT_mean_height_is_68_l1613_161302

/-
Given the heights of the volleyball players:
  heights_50s = [58, 59]
  heights_60s = [60, 61, 62, 65, 65, 66, 67]
  heights_70s = [70, 71, 71, 72, 74, 75, 79, 79]

We need to prove that the mean height of the players is 68 inches.
-/
def heights_50s : List ℕ := [58, 59]
def heights_60s : List ℕ := [60, 61, 62, 65, 65, 66, 67]
def heights_70s : List ℕ := [70, 71, 71, 72, 74, 75, 79, 79]

def total_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s
def number_of_players : ℕ := total_heights.length
def total_height : ℕ := total_heights.sum
def mean_height : ℕ := total_height / number_of_players

theorem mean_height_is_68 : mean_height = 68 := by
  sorry

end NUMINAMATH_GPT_mean_height_is_68_l1613_161302


namespace NUMINAMATH_GPT_platform_length_l1613_161379

theorem platform_length
  (L_train : ℕ) (T_platform : ℕ) (T_pole : ℕ) (P : ℕ)
  (h1 : L_train = 300)
  (h2 : T_platform = 39)
  (h3 : T_pole = 10)
  (h4 : L_train / T_pole * T_platform = L_train + P) :
  P = 870 := 
sorry

end NUMINAMATH_GPT_platform_length_l1613_161379


namespace NUMINAMATH_GPT_john_writes_book_every_2_months_l1613_161390

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

end NUMINAMATH_GPT_john_writes_book_every_2_months_l1613_161390


namespace NUMINAMATH_GPT_intersection_P_Q_equals_P_l1613_161380

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := { y | ∃ x ∈ Set.univ, y = Real.cos x }

theorem intersection_P_Q_equals_P : P ∩ Q = P := by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_equals_P_l1613_161380


namespace NUMINAMATH_GPT_max_buses_in_city_l1613_161326

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end NUMINAMATH_GPT_max_buses_in_city_l1613_161326


namespace NUMINAMATH_GPT_cakes_sold_correct_l1613_161397

def total_cakes_baked_today : Nat := 5
def total_cakes_baked_yesterday : Nat := 3
def cakes_left : Nat := 2

def total_cakes : Nat := total_cakes_baked_today + total_cakes_baked_yesterday
def cakes_sold : Nat := total_cakes - cakes_left

theorem cakes_sold_correct :
  cakes_sold = 6 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cakes_sold_correct_l1613_161397


namespace NUMINAMATH_GPT_system_of_equations_solution_l1613_161394

theorem system_of_equations_solution (x y z : ℝ) :
  (x = 6 + Real.sqrt 29 ∧ y = (5 - 2 * (6 + Real.sqrt 29)) / 3 ∧ z = (4 - (6 + Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) ∨
  (x = 6 - Real.sqrt 29 ∧ y = (5 - 2 * (6 - Real.sqrt 29)) / 3 ∧ z = (4 - (6 - Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) :=
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1613_161394


namespace NUMINAMATH_GPT_plane_equation_and_gcd_l1613_161378

variable (x y z : ℝ)

theorem plane_equation_and_gcd (A B C D : ℤ) (h1 : A = 8) (h2 : B = -6) (h3 : C = 5) (h4 : D = -125) :
    (A * x + B * y + C * z + D = 0 ↔ x = 8 ∧ y = -6 ∧ z = 5) ∧
    Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by sorry

end NUMINAMATH_GPT_plane_equation_and_gcd_l1613_161378


namespace NUMINAMATH_GPT_find_constants_l1613_161321

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 2)

theorem find_constants (a : ℝ) (x : ℝ) (h : x ≠ -2) :
  f a (f a x) = x ∧ a = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1613_161321


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1613_161340

theorem eccentricity_of_hyperbola (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_c : c = Real.sqrt (a^2 + b^2))
  (F1 : ℝ × ℝ := (-c, 0))
  (A B : ℝ × ℝ)
  (slope_of_AB : ∀ (x y : ℝ), y = x + c)
  (asymptotes_eqn : ∀ (x : ℝ), x = a ∨ x = -a)
  (intersections : A = (-(a * c / (a - b)), -(b * c / (a - b))) ∧ B = (-(a * c / (a + b)), (b * c / (a + b))))
  (AB_eq_2BF1 : 2 * (F1 - B) = A - B) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1613_161340


namespace NUMINAMATH_GPT_chandler_weeks_to_buy_bike_l1613_161357

-- Define the given problem conditions as variables/constants
def bike_cost : ℕ := 650
def grandparents_gift : ℕ := 60
def aunt_gift : ℕ := 45
def cousin_gift : ℕ := 25
def weekly_earnings : ℕ := 20
def total_birthday_money : ℕ := grandparents_gift + aunt_gift + cousin_gift

-- Define the total money Chandler will have after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_birthday_money + weekly_earnings * x

-- The main theorem states that Chandler needs 26 weeks to save enough money to buy the bike
theorem chandler_weeks_to_buy_bike : ∃ x : ℕ, total_money_after_weeks x = bike_cost :=
by
  -- Since we know x = 26 from the solution:
  use 26
  sorry

end NUMINAMATH_GPT_chandler_weeks_to_buy_bike_l1613_161357


namespace NUMINAMATH_GPT_fraction_of_q_age_l1613_161389

theorem fraction_of_q_age (P Q : ℕ) (h1 : P / Q = 3 / 4) (h2 : P + Q = 28) : (P - 0) / (Q - 0) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_q_age_l1613_161389


namespace NUMINAMATH_GPT_committee_size_l1613_161365

theorem committee_size (n : ℕ) (h : 2 * n = 6) (p : ℚ) (h_prob : p = 2/5) : n = 3 :=
by
  -- problem conditions
  have h1 : 2 * n = 6 := h
  have h2 : p = 2/5 := h_prob
  -- skip the proof details
  sorry

end NUMINAMATH_GPT_committee_size_l1613_161365


namespace NUMINAMATH_GPT_blue_line_length_correct_l1613_161363

def white_line_length : ℝ := 7.67
def difference_in_length : ℝ := 4.33
def blue_line_length : ℝ := 3.34

theorem blue_line_length_correct :
  white_line_length - difference_in_length = blue_line_length :=
by
  sorry

end NUMINAMATH_GPT_blue_line_length_correct_l1613_161363


namespace NUMINAMATH_GPT_Alice_more_nickels_l1613_161377

-- Define quarters each person has
def Alice_quarters (q : ℕ) : ℕ := 10 * q + 2
def Bob_quarters (q : ℕ) : ℕ := 2 * q + 10

-- Prove that Alice has 40(q - 1) more nickels than Bob
theorem Alice_more_nickels (q : ℕ) : 
  (5 * (Alice_quarters q - Bob_quarters q)) = 40 * (q - 1) :=
by
  sorry

end NUMINAMATH_GPT_Alice_more_nickels_l1613_161377


namespace NUMINAMATH_GPT_calc1_calc2_l1613_161304

-- Problem 1
theorem calc1 : 2 * Real.sqrt 3 - 3 * Real.sqrt 12 + 5 * Real.sqrt 27 = 11 * Real.sqrt 3 := 
by sorry

-- Problem 2
theorem calc2 : (1 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 6) - (2 * Real.sqrt 3 - 1)^2 
              = -2 * Real.sqrt 2 + 4 * Real.sqrt 3 - 13 := 
by sorry

end NUMINAMATH_GPT_calc1_calc2_l1613_161304


namespace NUMINAMATH_GPT_completing_the_square_correct_l1613_161343

theorem completing_the_square_correct :
  (∃ x : ℝ, x^2 - 6 * x + 5 = 0) →
  (∃ x : ℝ, (x - 3)^2 = 4) :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_correct_l1613_161343


namespace NUMINAMATH_GPT_trig_relation_l1613_161322

theorem trig_relation (a b c : ℝ) 
  (h1 : a = Real.sin 2) 
  (h2 : b = Real.cos 2) 
  (h3 : c = Real.tan 2) : c < b ∧ b < a := 
by
  sorry

end NUMINAMATH_GPT_trig_relation_l1613_161322


namespace NUMINAMATH_GPT_find_m_value_l1613_161364

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
    (hf : ∀ x, f x = 3 * x ^ 2 - 1 / x + 4)
    (hg : ∀ x, g x = x ^ 2 - m)
    (hfg : f 3 - g 3 = 5) :
    m = -50 / 3 :=
  sorry

end NUMINAMATH_GPT_find_m_value_l1613_161364


namespace NUMINAMATH_GPT_gcd_8164_2937_l1613_161317

/-- Define the two integers a and b -/
def a : ℕ := 8164
def b : ℕ := 2937

/-- Prove that the greatest common divisor of a and b is 1 -/
theorem gcd_8164_2937 : Nat.gcd a b = 1 :=
  by
  sorry

end NUMINAMATH_GPT_gcd_8164_2937_l1613_161317


namespace NUMINAMATH_GPT_major_axis_length_l1613_161349

/-- Defines the properties of the ellipse we use in this problem. --/
def ellipse (x y : ℝ) : Prop :=
  let f1 := (5, 1 + Real.sqrt 8)
  let f2 := (5, 1 - Real.sqrt 8)
  let tangent_line_at_y := y = 1
  let tangent_line_at_x := x = 1
  tangent_line_at_y ∧ tangent_line_at_x ∧
  ((x - f1.1)^2 + (y - f1.2)^2) + ((x - f2.1)^2 + (y - f2.2)^2) = 4

/-- Proves the length of the major axis of the specific ellipse --/
theorem major_axis_length : ∃ l : ℝ, l = 4 :=
  sorry

end NUMINAMATH_GPT_major_axis_length_l1613_161349


namespace NUMINAMATH_GPT_small_circle_ratio_l1613_161386

theorem small_circle_ratio (a b : ℝ) (ha : 0 < a) (hb : a < b) 
  (h : π * b^2 - π * a^2 = 5 * (π * a^2)) :
  a / b = Real.sqrt 6 / 6 :=
by
  sorry

end NUMINAMATH_GPT_small_circle_ratio_l1613_161386


namespace NUMINAMATH_GPT_find_number_l1613_161347

theorem find_number (x : ℝ) :
  9 * (((x + 1.4) / 3) - 0.7) = 5.4 ↔ x = 2.5 :=
by sorry

end NUMINAMATH_GPT_find_number_l1613_161347


namespace NUMINAMATH_GPT_inequality_proof_l1613_161320

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  2 * (a + b + c) + 9 / (a * b + b * c + c * a)^2 ≥ 7 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1613_161320


namespace NUMINAMATH_GPT_geometric_sequence_general_term_l1613_161333

theorem geometric_sequence_general_term (a : ℕ → ℕ) (q : ℕ) (h_q : q = 4) (h_sum : a 0 + a 1 + a 2 = 21)
  (h_geo : ∀ n, a (n + 1) = a n * q) : ∀ n, a n = 4 ^ n :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_general_term_l1613_161333


namespace NUMINAMATH_GPT_sheets_borrowed_l1613_161366

-- Definitions based on conditions
def total_pages : ℕ := 60  -- Hiram's algebra notes are 60 pages
def total_sheets : ℕ := 30  -- printed on 30 sheets of paper
def average_remaining : ℕ := 23  -- the average of the page numbers on all remaining sheets is 23

-- Let S_total be the sum of all page numbers initially
def S_total := (total_pages * (1 + total_pages)) / 2

-- Let c be the number of consecutive sheets borrowed
-- Let b be the number of sheets before the borrowed sheets
-- Calculate S_borrowed based on problem conditions
def S_borrowed (c b : ℕ) := 2 * c * (b + c) + c

-- Calculate the remaining sum and corresponding mean
def remaining_sum (c b : ℕ) := S_total - S_borrowed c b
def remaining_mean (c : ℕ) := (total_sheets * 2 - 2 * c)

-- The theorem we want to prove
theorem sheets_borrowed (c : ℕ) (h : 1830 - S_borrowed c 10 = 23 * (60 - 2 * c)) : c = 15 :=
  sorry

end NUMINAMATH_GPT_sheets_borrowed_l1613_161366


namespace NUMINAMATH_GPT_Lily_books_on_Wednesday_l1613_161338

noncomputable def booksMike : ℕ := 45

noncomputable def booksCorey : ℕ := 2 * booksMike

noncomputable def booksMikeGivenToLily : ℕ := 13

noncomputable def booksCoreyGivenToLily : ℕ := booksMikeGivenToLily + 5

noncomputable def booksEmma : ℕ := 28

noncomputable def booksEmmaGivenToLily : ℕ := booksEmma / 4

noncomputable def totalBooksLilyGot : ℕ := booksMikeGivenToLily + booksCoreyGivenToLily + booksEmmaGivenToLily

theorem Lily_books_on_Wednesday : totalBooksLilyGot = 38 := by
  sorry

end NUMINAMATH_GPT_Lily_books_on_Wednesday_l1613_161338


namespace NUMINAMATH_GPT_four_letter_list_product_l1613_161398

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def list_product (s : String) : Nat :=
  s.foldl (λ acc c => acc * letter_value c) 1

def target_product : Nat :=
  list_product "TUVW"

theorem four_letter_list_product : 
  ∀ (s1 s2 : String), s1.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') → s2.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') →
  s1.length = 4 → s2.length = 4 →
  list_product s1 = target_product → s2 = "BEHK" :=
by
  sorry

end NUMINAMATH_GPT_four_letter_list_product_l1613_161398


namespace NUMINAMATH_GPT_option_d_necessary_sufficient_l1613_161327

theorem option_d_necessary_sufficient (a : ℝ) : (a ≠ 0) ↔ (∃! x : ℝ, a * x = 1) := 
sorry

end NUMINAMATH_GPT_option_d_necessary_sufficient_l1613_161327
