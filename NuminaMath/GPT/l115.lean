import Mathlib

namespace range_of_a_l115_115560

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → |x| < a) → 1 ≤ a :=
by 
  sorry

end range_of_a_l115_115560


namespace simplify_tan_alpha_l115_115721

noncomputable def f (α : ℝ) : ℝ :=
(Real.sin (Real.pi / 2 + α) + Real.sin (-Real.pi - α)) /
  (3 * Real.cos (2 * Real.pi - α) + Real.cos (3 * Real.pi / 2 - α))

theorem simplify_tan_alpha (α : ℝ) (h : Real.tan α = 1) : f α = 1 := by
  sorry

end simplify_tan_alpha_l115_115721


namespace last_digit_base5_89_l115_115514

theorem last_digit_base5_89 : 
  ∃ (b : ℕ), (89 : ℕ) = b * 5 + 4 :=
by
  -- The theorem above states that there exists an integer b, such that when we compute 89 in base 5, 
  -- its last digit is 4.
  sorry

end last_digit_base5_89_l115_115514


namespace intersection_A_B_l115_115774

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end intersection_A_B_l115_115774


namespace negation_proposition_l115_115090

theorem negation_proposition:
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
by sorry

end negation_proposition_l115_115090


namespace line_ellipse_intersection_product_l115_115088

-- Definitions based on conditions
def line_polar (ρ θ : ℝ) : ℝ := ρ * cos θ + ρ * sin θ - 1

def ellipse_param (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, sin θ)

-- Proof goal
theorem line_ellipse_intersection_product (θ : ℝ) :
  (x : ℝ) (y : ℝ), x = 1 → y = 0 →
  ∃ t₁ t₂ : ℝ, (x = 1 + (sqrt 2) / 2 * t) ∧ (y = (sqrt 2) / 2 * t) ∧ 
    (∀ (x y : ℝ), x = 2 * cos θ → y = sin θ → x^2 + 4 * y^2 = 4) → (|t₁ * t₂| = 6 / 5) :=
begin
  sorry
end

end line_ellipse_intersection_product_l115_115088


namespace totalBasketballs_proof_l115_115074

-- Define the number of balls for Lucca and Lucien
axiom numBallsLucca : ℕ := 100
axiom percBasketballsLucca : ℕ := 10
axiom numBallsLucien : ℕ := 200
axiom percBasketballsLucien : ℕ := 20

-- Calculate the number of basketballs they each have
def basketballsLucca := (percBasketballsLucca * numBallsLucca) / 100
def basketballsLucien := (percBasketballsLucien * numBallsLucien) / 100

-- Total number of basketballs
def totalBasketballs := basketballsLucca + basketballsLucien

theorem totalBasketballs_proof : totalBasketballs = 50 := by
  sorry

end totalBasketballs_proof_l115_115074


namespace stadium_length_in_yards_l115_115288

def length_in_feet := 183
def feet_per_yard := 3

theorem stadium_length_in_yards : length_in_feet / feet_per_yard = 61 := by
  sorry

end stadium_length_in_yards_l115_115288


namespace area_between_chords_is_correct_l115_115404

noncomputable def circle_radius : ℝ := 10
noncomputable def chord_distance_apart : ℝ := 12
noncomputable def area_between_chords : ℝ := 44.73

theorem area_between_chords_is_correct 
    (r : ℝ) (d : ℝ) (A : ℝ) 
    (hr : r = circle_radius) 
    (hd : d = chord_distance_apart) 
    (hA : A = area_between_chords) : 
    ∃ area : ℝ, area = A := by 
  sorry

end area_between_chords_is_correct_l115_115404


namespace intersection_point_of_lines_l115_115742

theorem intersection_point_of_lines : 
  (∃ x y : ℚ, (8 * x - 3 * y = 5) ∧ (5 * x + 2 * y = 20)) ↔ (x = 70 / 31 ∧ y = 135 / 31) :=
sorry

end intersection_point_of_lines_l115_115742


namespace conditional_probability_of_B2_given_A_l115_115092

theorem conditional_probability_of_B2_given_A :
  let P_B1 := 0.55
  let P_B2 := 0.45
  let P_A_given_B1 := 0.9
  let P_A_given_B2 := 0.98
  let P_A := P_A_given_B1 * P_B1 + P_A_given_B2 * P_B2
  (P_A_given_B2 * P_B2 / P_A) = 0.4712 :=
by
  let P_B1 := 0.55
  let P_B2 := 0.45
  let P_A_given_B1 := 0.9
  let P_A_given_B2 := 0.98
  let P_A := P_A_given_B1 * P_B1 + P_A_given_B2 * P_B2
  have h : P_A = 0.936 := by {
    calc
    P_A = 0.9 * 0.55 + 0.98 * 0.45 : by norm_num
    ... = 0.495 + 0.441 : by norm_num
    ... = 0.936 : by norm_num
  }
  calc
  (P_A_given_B2 * P_B2 / P_A) = (0.98 * 0.45 / 0.936) : by norm_num
  ... = 0.4712 : by norm_num

end conditional_probability_of_B2_given_A_l115_115092


namespace probability_intersection_three_elements_l115_115776

theorem probability_intersection_three_elements (U : Finset ℕ) (hU : U = {1, 2, 3, 4, 5}) : 
  ∃ (p : ℚ), p = 5 / 62 :=
by
  sorry

end probability_intersection_three_elements_l115_115776


namespace root_properties_of_cubic_l115_115037

theorem root_properties_of_cubic (z1 z2 : ℂ) (h1 : z1^2 + z1 + 1 = 0) (h2 : z2^2 + z2 + 1 = 0) :
  z1 * z2 = 1 ∧ z1^3 = 1 ∧ z2^3 = 1 :=
by
  -- Proof omitted
  sorry

end root_properties_of_cubic_l115_115037


namespace truck_cargo_solution_l115_115336

def truck_cargo_problem (x : ℝ) (n : ℕ) : Prop :=
  (∀ (x : ℝ) (n : ℕ), x = (x / n - 0.5) * (n + 4)) ∧ (55 ≤ x ∧ x ≤ 64)

theorem truck_cargo_solution :
  ∃ y : ℝ, y = 2.5 :=
sorry

end truck_cargo_solution_l115_115336


namespace least_number_of_trees_l115_115483

theorem least_number_of_trees (n : ℕ) :
  (∃ k₄ k₅ k₆, n = 4 * k₄ ∧ n = 5 * k₅ ∧ n = 6 * k₆) ↔ n = 60 :=
by 
  sorry

end least_number_of_trees_l115_115483


namespace angle_triple_complement_l115_115670

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l115_115670


namespace angle_triple_complement_l115_115651

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l115_115651


namespace even_product_implies_even_factor_l115_115309

theorem even_product_implies_even_factor (a b : ℕ) (h : Even (a * b)) : Even a ∨ Even b :=
by
  sorry

end even_product_implies_even_factor_l115_115309


namespace routeY_is_quicker_l115_115939

noncomputable def timeRouteX : ℝ := 
  8 / 40 

noncomputable def timeRouteY1 : ℝ := 
  6.5 / 50 

noncomputable def timeRouteY2 : ℝ := 
  0.5 / 10

noncomputable def timeRouteY : ℝ := 
  timeRouteY1 + timeRouteY2  

noncomputable def timeDifference : ℝ := 
  (timeRouteX - timeRouteY) * 60 

theorem routeY_is_quicker : 
  timeDifference = 1.2 :=
by
  sorry

end routeY_is_quicker_l115_115939


namespace expand_expression_l115_115201

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115201


namespace sampling_methods_match_l115_115590

inductive SamplingMethod
| simple_random
| stratified
| systematic

open SamplingMethod

def commonly_used_sampling_methods : List SamplingMethod := 
  [simple_random, stratified, systematic]

def option_C : List SamplingMethod := 
  [simple_random, stratified, systematic]

theorem sampling_methods_match : commonly_used_sampling_methods = option_C := by
  sorry

end sampling_methods_match_l115_115590


namespace maximum_value_ratio_l115_115914

theorem maximum_value_ratio (a b : ℝ) (h1 : a + b - 2 ≥ 0) (h2 : b - a - 1 ≤ 0) (h3 : a ≤ 1) :
  ∃ x, x = (a + 2 * b) / (2 * a + b) ∧ x ≤ 7/5 := sorry

end maximum_value_ratio_l115_115914


namespace angle_measure_triple_complement_l115_115622

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l115_115622


namespace problem_statement_l115_115926

def class_of_rem (k : ℕ) : Set ℤ := {n | ∃ m : ℤ, n = 4 * m + k}

theorem problem_statement : (2013 ∈ class_of_rem 1) ∧ 
                            (-2 ∈ class_of_rem 2) ∧ 
                            (∀ x : ℤ, x ∈ class_of_rem 0 ∨ x ∈ class_of_rem 1 ∨ x ∈ class_of_rem 2 ∨ x ∈ class_of_rem 3) ∧ 
                            (∀ a b : ℤ, (∃ k : ℕ, (a ∈ class_of_rem k ∧ b ∈ class_of_rem k)) ↔ (a - b) ∈ class_of_rem 0) :=
by
  -- each of the statements should hold true
  sorry

end problem_statement_l115_115926


namespace Punta_position_l115_115861

theorem Punta_position (N x y p : ℕ) (h1 : N = 36) (h2 : x = y / 4) (h3 : x + y = 35) : p = 8 := by
  sorry

end Punta_position_l115_115861


namespace smallest_value_l115_115016

noncomputable def Q : Polynomial ℝ := X^4 - 2*X^3 + 3*X^2 - 4*X + 5

theorem smallest_value :
  let q1 := Q.eval 1,
      prod_zeros := Q.leading_coeff * Q.coeff 0, -- Using Vieta's formulas
      sum_coeff := Q.coeffs.sum in
  ∃ (p : ℝ), p = min (min q1 (prod_zeros) (sum_coeff)) 
  → p = the_product_of_non_real_zeros Q
:= sorry

end smallest_value_l115_115016


namespace angle_CAD_l115_115432

noncomputable def angle_arc (degree: ℝ) (minute: ℝ) : ℝ :=
  degree + minute / 60

theorem angle_CAD :
  angle_arc 117 23 / 2 + angle_arc 42 37 / 2 = 80 :=
by
  sorry

end angle_CAD_l115_115432


namespace initial_amount_is_825_l115_115857

theorem initial_amount_is_825 (P R : ℝ) 
    (h1 : 956 = P * (1 + 3 * R / 100))
    (h2 : 1055 = P * (1 + 3 * (R + 4) / 100)) : 
    P = 825 := 
by 
  sorry

end initial_amount_is_825_l115_115857


namespace function_decreasing_range_l115_115547

theorem function_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1) ≤ (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1)) ↔ (0 ≤ a ∧ a ≤ 1 / 3) :=
sorry

end function_decreasing_range_l115_115547


namespace carrot_broccoli_ratio_l115_115428

variables (total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings : ℕ)

-- Define the conditions
def is_condition_satisfied :=
  total_earnings = 380 ∧
  broccoli_earnings = 57 ∧
  cauliflower_earnings = 136 ∧
  spinach_earnings = (carrot_earnings / 2 + 16)

-- Define the proof problem that checks the ratio
theorem carrot_broccoli_ratio (h : is_condition_satisfied total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings) :
  ((carrot_earnings + ((carrot_earnings / 2) + 16)) + broccoli_earnings + cauliflower_earnings = total_earnings) →
  (carrot_earnings / broccoli_earnings = 2) :=
sorry

end carrot_broccoli_ratio_l115_115428


namespace intersection_A_B_l115_115538

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l115_115538


namespace sum_from_one_to_twelve_l115_115109

-- Define the sum of an arithmetic series
def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Theorem stating the sum of numbers from 1 to 12
theorem sum_from_one_to_twelve : sum_arithmetic_series 12 1 12 = 78 := by
  sorry

end sum_from_one_to_twelve_l115_115109


namespace problem_statement_l115_115771

noncomputable def f (x : ℝ) : ℝ := 2 / (2019^x + 1) + Real.sin x

noncomputable def f' (x : ℝ) := (deriv f) x

theorem problem_statement :
  f 2018 + f (-2018) + f' 2019 - f' (-2019) = 2 :=
by {
  sorry
}

end problem_statement_l115_115771


namespace angle_measure_triple_complement_l115_115620

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l115_115620


namespace expand_expression_l115_115203

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115203


namespace mabel_marble_ratio_l115_115873

variable (A K M : ℕ)

-- Conditions
def condition1 : Prop := A + 12 = 2 * K
def condition2 : Prop := M = 85
def condition3 : Prop := M = A + 63

-- The main statement to prove
theorem mabel_marble_ratio (h1 : condition1 A K) (h2 : condition2 M) (h3 : condition3 A M) : M / K = 5 :=
by
  sorry

end mabel_marble_ratio_l115_115873


namespace dots_not_visible_l115_115373

def total_dots (n_dice : ℕ) : ℕ := n_dice * 21

def sum_visible_dots (visible : List ℕ) : ℕ := visible.foldl (· + ·) 0

theorem dots_not_visible (visible : List ℕ) (h : visible = [1, 1, 2, 3, 4, 5, 5, 6]) :
  total_dots 4 - sum_visible_dots visible = 57 :=
by
  rw [total_dots, sum_visible_dots]
  simp
  sorry

end dots_not_visible_l115_115373


namespace candy_box_price_increase_l115_115210

theorem candy_box_price_increase
  (C : ℝ) -- Original price of the candy box
  (S : ℝ := 12) -- Original price of a can of soda
  (combined_price : C + S = 16) -- Combined price before increase
  (candy_box_increase : C + 0.25 * C = 1.25 * C) -- Price increase definition
  (soda_increase : S + 0.50 * S = 18) -- New price of soda after increase
  : 1.25 * C = 5 := sorry

end candy_box_price_increase_l115_115210


namespace tan_ratio_proof_l115_115894

theorem tan_ratio_proof (α : ℝ) (h : 5 * Real.sin (2 * α) = Real.sin 2) : 
  Real.tan (α + 1 * Real.pi / 180) / Real.tan (α - 1 * Real.pi / 180) = - 3 / 2 := 
sorry

end tan_ratio_proof_l115_115894


namespace ratio_of_segments_invariant_l115_115231

variable {P E F Oe Of : Point}
variable {circle_e circle_f : Circle}
variable {r_e r_f : ℝ}
variable {angle_PEF angle_PFE : ℝ}

theorem ratio_of_segments_invariant
  (tangent_from_P_to_e : is_tangent P E circle_e)
  (tangent_from_P_to_f : is_tangent P F circle_f)
  (radius_circle_e : circle_e.radius = r_e)
  (radius_circle_f : circle_f.radius = r_f)
  (sin_angle_E : sin (angle P E F) = sin angle_PEF)
  (sin_angle_F : sin (angle P F E) = sin angle_PFE) :
  (2 * r_f * sin_angle_F) / (2 * r_e * sin_angle_E) = (PE / PF) :=
sorry

end ratio_of_segments_invariant_l115_115231


namespace floor_of_fraction_expression_l115_115141

theorem floor_of_fraction_expression : 
  ( ⌊ (2025^3 / (2023 * 2024)) - (2023^3 / (2024 * 2025)) ⌋ ) = 8 :=
sorry

end floor_of_fraction_expression_l115_115141


namespace unique_function_satisfying_conditions_l115_115366

theorem unique_function_satisfying_conditions (f : ℤ → ℤ) :
  (∀ n : ℤ, f (f n) + f n = 2 * n + 3) → 
  (f 0 = 1) → 
  (∀ n : ℤ, f n = n + 1) :=
by
  intro h1 h2
  sorry

end unique_function_satisfying_conditions_l115_115366


namespace inequality_proof_l115_115935

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
by
  sorry

end inequality_proof_l115_115935


namespace probability_of_x_gt_3y_is_correct_l115_115942

noncomputable def probability_x_gt_3y : ℚ :=
  let rectangle_width := 2016
  let rectangle_height := 2017
  let triangle_height := 672 -- 2016 / 3
  let triangle_area := 1 / 2 * rectangle_width * triangle_height
  let rectangle_area := rectangle_width * rectangle_height
  triangle_area / rectangle_area

theorem probability_of_x_gt_3y_is_correct :
  probability_x_gt_3y = 336 / 2017 :=
by
  -- Proof will be filled in later
  sorry

end probability_of_x_gt_3y_is_correct_l115_115942


namespace expand_expression_l115_115184

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115184


namespace ratio_turkeys_to_ducks_l115_115817

theorem ratio_turkeys_to_ducks (chickens ducks turkeys total_birds : ℕ)
  (h1 : chickens = 200)
  (h2 : ducks = 2 * chickens)
  (h3 : total_birds = 1800)
  (h4 : total_birds = chickens + ducks + turkeys) :
  (turkeys : ℚ) / ducks = 3 := by
sorry

end ratio_turkeys_to_ducks_l115_115817


namespace no_odd_multiples_between_1500_and_3000_l115_115232

theorem no_odd_multiples_between_1500_and_3000 :
  ∀ n : ℤ, 1500 ≤ n → n ≤ 3000 → (18 ∣ n) → (24 ∣ n) → (36 ∣ n) → ¬(n % 2 = 1) :=
by
  -- The proof steps would go here, but we skip them according to the instructions.
  sorry

end no_odd_multiples_between_1500_and_3000_l115_115232


namespace equal_probability_of_selection_l115_115081

-- Define the number of total students
def total_students : ℕ := 86

-- Define the number of students to be eliminated through simple random sampling
def eliminated_students : ℕ := 6

-- Define the number of students selected through systematic sampling
def selected_students : ℕ := 8

-- Define the probability calculation
def probability_not_eliminated : ℚ := 80 / 86
def probability_selected : ℚ := 8 / 80
def combined_probability : ℚ := probability_not_eliminated * probability_selected

theorem equal_probability_of_selection :
  combined_probability = 4 / 43 :=
by
  -- We do not need to complete the proof as per instruction
  sorry

end equal_probability_of_selection_l115_115081


namespace angle_triple_complement_l115_115697

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l115_115697


namespace rectangle_k_value_l115_115287

theorem rectangle_k_value (a k : ℝ) (h1 : k > 0) (h2 : 2 * (3 * a + a) = k) (h3 : 3 * a^2 = k) : k = 64 / 3 :=
by
  sorry

end rectangle_k_value_l115_115287


namespace least_three_digit_divisible_3_4_7_is_168_l115_115477

-- Define the function that checks the conditions
def is_least_three_digit_divisible_by_3_4_7 (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ x % 3 = 0 ∧ x % 4 = 0 ∧ x % 7 = 0

-- Define the target value
def least_three_digit_number_divisible_by_3_4_7 : ℕ := 168

-- The theorem we want to prove
theorem least_three_digit_divisible_3_4_7_is_168 :
  ∃ x : ℕ, is_least_three_digit_divisible_by_3_4_7 x ∧ x = least_three_digit_number_divisible_by_3_4_7 := by
  sorry

end least_three_digit_divisible_3_4_7_is_168_l115_115477


namespace problem_D_l115_115902

variable {a b c : ℝ}
variable (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : b ≠ 1) (h₄ : c ≠ 1)

def op ⊕ (a b : ℝ) : ℝ := a ^ (Real.log a / Real.log b)

theorem problem_D :
  (op ⊕ a b) ^ c = op ⊕ (a ^ c) b :=
by
  sorry

end problem_D_l115_115902


namespace angle_is_67_l115_115642

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l115_115642


namespace sallys_change_l115_115587

-- Define the total cost calculation:
def totalCost (numFrames : Nat) (costPerFrame : Nat) : Nat :=
  numFrames * costPerFrame

-- Define the change calculation:
def change (totalAmount : Nat) (amountPaid : Nat) : Nat :=
  amountPaid - totalAmount

-- Define the specific conditions in the problem:
def numFrames := 3
def costPerFrame := 3
def amountPaid := 20

-- Prove that the change Sally gets is $11:
theorem sallys_change : change (totalCost numFrames costPerFrame) amountPaid = 11 := by
  sorry

end sallys_change_l115_115587


namespace spencer_sessions_per_day_l115_115847

theorem spencer_sessions_per_day :
  let jumps_per_minute := 4
  let minutes_per_session := 10
  let jumps_per_session := jumps_per_minute * minutes_per_session
  let total_jumps := 400
  let days := 5
  let jumps_per_day := total_jumps / days
  let sessions_per_day := jumps_per_day / jumps_per_session
  sessions_per_day = 2 :=
by
  sorry

end spencer_sessions_per_day_l115_115847


namespace three_divides_n_of_invertible_diff_l115_115419

theorem three_divides_n_of_invertible_diff
  (n : ℕ)
  (A B : Matrix (Fin n) (Fin n) ℝ)
  (h1 : A * A + B * B = A * B)
  (h2 : Invertible (B * A - A * B)) :
  3 ∣ n :=
sorry

end three_divides_n_of_invertible_diff_l115_115419


namespace vector_b_satisfies_conditions_l115_115931

noncomputable section

def a : ℝ^3 := ![3, 2, 4]

def b : ℝ^3 := ![1, 16, 3]

def dotProduct (u v : ℝ^3) : ℝ := u 0 * v 0 + u 1 * v 1 + u 2 * v 2

def crossProduct (u v : ℝ^3) : ℝ^3 :=
  ![u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0]

theorem vector_b_satisfies_conditions :
  dotProduct a b = 20 ∧ crossProduct a b = ![-20, 5, 2] := by
  sorry

end vector_b_satisfies_conditions_l115_115931


namespace sufficient_condition_transitive_l115_115493

theorem sufficient_condition_transitive
  (C B A : Prop) (h1 : (C → B)) (h2 : (B → A)) : (C → A) :=
  sorry

end sufficient_condition_transitive_l115_115493


namespace infinite_solutions_exists_l115_115434

theorem infinite_solutions_exists :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧
  (x - y + z = 1) ∧ ((x * y) % z = 0) ∧ ((y * z) % x = 0) ∧ ((z * x) % y = 0) ∧
  ∀ n : ℕ, ∃ x y z : ℕ, (n > 0) ∧ (x = n * (n^2 + n - 1)) ∧ (y = (n+1) * (n^2 + n - 1)) ∧ (z = n * (n+1)) := by
  sorry

end infinite_solutions_exists_l115_115434


namespace positive_divisors_3k1_ge_3k_minus_1_l115_115944

theorem positive_divisors_3k1_ge_3k_minus_1 (n : ℕ) (h : 0 < n) :
  (∃ k : ℕ, (3 * k + 1) ∣ n) → (∃ k : ℕ, ¬ (3 * k - 1) ∣ n) :=
  sorry

end positive_divisors_3k1_ge_3k_minus_1_l115_115944


namespace cubic_sum_identity_l115_115050

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 10) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 100 :=
by sorry

end cubic_sum_identity_l115_115050


namespace angle_is_67_l115_115643

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l115_115643


namespace basketball_committee_l115_115855

theorem basketball_committee (total_players guards : ℕ) (choose_committee choose_guard : ℕ) :
  total_players = 12 → guards = 4 → choose_committee = 3 → choose_guard = 1 →
  (guards * ((total_players - guards).choose (choose_committee - choose_guard)) = 112) :=
by
  intros h_tp h_g h_cc h_cg
  rw [h_tp, h_g, h_cc, h_cg]
  simp
  norm_num
  sorry

end basketball_committee_l115_115855


namespace lawn_area_l115_115943

theorem lawn_area (s l : ℕ) (hs: 5 * s = 10) (hl: 5 * l = 50) (hposts: 2 * (s + l) = 24) (hlen: l + 1 = 3 * (s + 1)) :
  s * l = 500 :=
by {
  sorry
}

end lawn_area_l115_115943


namespace increasing_sufficient_not_necessary_l115_115294

-- Conditions
def increasing_function (a : ℝ) (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y
def log2_gt_1 (a : ℝ) := real.log 2 a > 1

-- Statement to be proved
theorem increasing_sufficient_not_necessary {a : ℝ} : 
  (∃ (f : ℝ → ℝ), f = (λ x, a * x) ∧ increasing_function a f) ↔ 
  (log2_gt_1 a) ∧ (∃ b : ℝ, 0 < b ∧ a = b) :=
sorry

end increasing_sufficient_not_necessary_l115_115294


namespace possible_to_form_square_l115_115257

def shape_covers_units : ℕ := 4

theorem possible_to_form_square (shape : ℕ) : ∃ n : ℕ, ∃ k : ℕ, n * n = shape * k :=
by
  use 4
  use 4
  sorry

end possible_to_form_square_l115_115257


namespace part1_part2_l115_115379

def P : Set ℝ := {x | x ≥ 1 / 2 ∧ x ≤ 2}

def Q (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 > 0}

def R (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 = 0}

theorem part1 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ Q a) → a > -1 / 2 :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ R a) → a ≥ -1 / 2 ∧ a ≤ 1 / 2 :=
by
  sorry

end part1_part2_l115_115379


namespace triangle_perimeter_upper_bound_l115_115992

theorem triangle_perimeter_upper_bound (a b : ℕ) (s : ℕ) (h₁ : a = 7) (h₂ : b = 23) 
  (h₃ : 16 < s) (h₄ : s < 30) : 
  ∃ n : ℕ, n = 60 ∧ n > a + b + s := 
by
  sorry

end triangle_perimeter_upper_bound_l115_115992


namespace triangle_hypotenuse_l115_115254

-- Given conditions
variables (ML LN : ℝ)
variables (M N L : Type) -- Vertices of the triangle
variable (sin_N : ℝ)
hypothesis1 : ML = 10
hypothesis2 : sin_N = 5 / 13

-- We aim to prove LN = 26 under these conditions.
theorem triangle_hypotenuse : LN = 26 :=
by
  sorry

end triangle_hypotenuse_l115_115254


namespace sin_cos_product_l115_115558

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l115_115558


namespace determine_g_x2_l115_115422

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem determine_g_x2 (x : ℝ) (h : x^2 ≠ 4) : g (x^2) = (2 * x^2 + 3) / (x^2 - 2) :=
by sorry

end determine_g_x2_l115_115422


namespace marbles_problem_l115_115991

theorem marbles_problem (n : ℕ) :
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0) → 
  n - 10 = 830 :=
sorry

end marbles_problem_l115_115991


namespace min_value_of_inverse_proportional_function_l115_115772

theorem min_value_of_inverse_proportional_function 
  (x y : ℝ) (k : ℝ) 
  (h1 : y = k / x) 
  (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → y ≤ 4) :
  (∀ x, x ≥ 8 → y = -1 / 2) :=
by
  sorry

end min_value_of_inverse_proportional_function_l115_115772


namespace Zhang_Hai_average_daily_delivery_is_37_l115_115983

theorem Zhang_Hai_average_daily_delivery_is_37
  (d1_packages : ℕ) (d1_count : ℕ)
  (d2_packages : ℕ) (d2_count : ℕ)
  (d3_packages : ℕ) (d3_count : ℕ)
  (total_days : ℕ) 
  (h1 : d1_packages = 41) (h2 : d1_count = 1)
  (h3 : d2_packages = 35) (h4 : d2_count = 2)
  (h5 : d3_packages = 37) (h6 : d3_count = 4)
  (h7 : total_days = 7) :
  (d1_count * d1_packages + d2_count * d2_packages + d3_count * d3_packages) / total_days = 37 := 
by sorry

end Zhang_Hai_average_daily_delivery_is_37_l115_115983


namespace ratio_average_speed_l115_115352

-- Define the conditions based on given distances and times
def distanceAB : ℕ := 600
def timeAB : ℕ := 3

def distanceBD : ℕ := 540
def timeBD : ℕ := 2

def distanceAC : ℕ := 460
def timeAC : ℕ := 4

def distanceCE : ℕ := 380
def timeCE : ℕ := 3

-- Define the total distances and times for Eddy and Freddy
def distanceEddy : ℕ := distanceAB + distanceBD
def timeEddy : ℕ := timeAB + timeBD

def distanceFreddy : ℕ := distanceAC + distanceCE
def timeFreddy : ℕ := timeAC + timeCE

-- Define the average speeds for Eddy and Freddy
def averageSpeedEddy : ℚ := distanceEddy / timeEddy
def averageSpeedFreddy : ℚ := distanceFreddy / timeFreddy

-- Prove the ratio of their average speeds is 19:10
theorem ratio_average_speed (h1 : distanceAB = 600) (h2 : timeAB = 3) 
                           (h3 : distanceBD = 540) (h4 : timeBD = 2)
                           (h5 : distanceAC = 460) (h6 : timeAC = 4) 
                           (h7 : distanceCE = 380) (h8 : timeCE = 3):
  averageSpeedEddy / averageSpeedFreddy = 19 / 10 := by sorry

end ratio_average_speed_l115_115352


namespace angle_measure_l115_115688

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l115_115688


namespace solve_trig_eq_l115_115824

noncomputable def rad (d : ℝ) := d * (Real.pi / 180)

theorem solve_trig_eq (z : ℝ) (k : ℤ) :
  (7 * Real.cos (z) ^ 3 - 6 * Real.cos (z) = 3 * Real.cos (3 * z)) ↔
  (z = rad 90 + k * rad 180 ∨
   z = rad 39.2333 + k * rad 180 ∨
   z = rad 140.7667 + k * rad 180) :=
sorry

end solve_trig_eq_l115_115824


namespace hikers_count_l115_115402

theorem hikers_count (B H K : ℕ) (h1 : H = B + 178) (h2 : K = B / 2) (h3 : H + B + K = 920) : H = 474 :=
by
  sorry

end hikers_count_l115_115402


namespace triple_complement_angle_l115_115685

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l115_115685


namespace expand_expression_l115_115200

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115200


namespace harry_morning_routine_l115_115358

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end harry_morning_routine_l115_115358


namespace remove_wallpaper_time_l115_115146

theorem remove_wallpaper_time 
    (total_walls : ℕ := 8)
    (remaining_walls : ℕ := 7)
    (time_for_remaining_walls : ℕ := 14) :
    time_for_remaining_walls / remaining_walls = 2 :=
by
sorry

end remove_wallpaper_time_l115_115146


namespace parking_monthly_charge_l115_115990

theorem parking_monthly_charge :
  ∀ (M : ℕ), (52 * 10 - 12 * M = 100) → M = 35 :=
by
  intro M h
  sorry

end parking_monthly_charge_l115_115990


namespace sector_area_l115_115226

theorem sector_area (θ r arc_length : ℝ) (h_arc_length : arc_length = r * θ) (h_values : θ = 2 ∧ arc_length = 2) :
  1 / 2 * r^2 * θ = 1 := by
  sorry

end sector_area_l115_115226


namespace Petya_can_determine_swap_l115_115585

open Finset

-- Define the cards and their positions
def initial_card_positions : Fin 9 → ℕ
| 0 := 8 | 1 := 1 | 2 := 2
| 3 := 3 | 4 := 5 | 5 := 9
| 6 := 6 | 7 := 7 | 8 := 4

-- Define the gray card positions
def gray_positions : Finset (Fin 9) := {1, 3, 7, 5}.to_finset

-- Define the initial sum of the gray positions
def initial_sum : ℕ := gray_positions.sum initial_card_positions

-- Define a function for sum after swap
def current_sum (f : Fin 9 → ℕ) : ℕ :=
  gray_positions.sum f

-- Problem statement: Petya can determine the exact swap
theorem Petya_can_determine_swap (swap : (Fin 9 → ℕ) → (Fin 9 → ℕ))
    (h_swap : ∃ (i j : Fin 9), i ≠ j ∧ ((λ f, swap f) = (λ f, f ∘ equiv.swap i j))) :
    initial_sum = current_sum initial_card_positions → ∃ f', current_sum (swap initial_card_positions) = current_sum f' → (∃ i j, i ≠ j ∧ (initial_card_positions i = f' j ∧ initial_card_positions j = f' i)) :=
by
  sorry

end Petya_can_determine_swap_l115_115585


namespace triangle_area_50_l115_115741

theorem triangle_area_50 :
  let A := (0, 0)
  let B := (0, 10)
  let C := (-10, 0)
  let base := 10
  let height := 10
  0 + base * height / 2 = 50 := by
sorry

end triangle_area_50_l115_115741


namespace algebraic_expression_evaluation_l115_115591

noncomputable def algebraic_expression (x : ℝ) : ℝ :=
  (x / (x^2 + 2*x + 1) - 1 / (2*x + 2)) / ((x - 1) / (4*x + 4))

noncomputable def substitution_value : ℝ :=
  2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_evaluation :
  algebraic_expression substitution_value = Real.sqrt 2 := by
  sorry

end algebraic_expression_evaluation_l115_115591


namespace total_cost_of_panels_l115_115928

theorem total_cost_of_panels
    (sidewall_width : ℝ)
    (sidewall_height : ℝ)
    (triangle_base : ℝ)
    (triangle_height : ℝ)
    (panel_width : ℝ)
    (panel_height : ℝ)
    (panel_cost : ℝ)
    (total_cost : ℝ)
    (h_sidewall : sidewall_width = 9)
    (h_sidewall_height : sidewall_height = 7)
    (h_triangle_base : triangle_base = 9)
    (h_triangle_height : triangle_height = 6)
    (h_panel_width : panel_width = 10)
    (h_panel_height : panel_height = 15)
    (h_panel_cost : panel_cost = 32)
    (h_total_cost : total_cost = 32) :
    total_cost = panel_cost :=
by
  sorry

end total_cost_of_panels_l115_115928


namespace line_passes_through_quadrants_l115_115758

variables (a b c p : ℝ)

-- Given conditions
def conditions :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  (a + b) / c = p ∧ 
  (b + c) / a = p ∧ 
  (c + a) / b = p

-- Goal statement
theorem line_passes_through_quadrants : conditions a b c p → 
  (∃ x : ℝ, x > 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p < 0) :=
sorry

end line_passes_through_quadrants_l115_115758


namespace square_perimeter_l115_115949

theorem square_perimeter (s : ℝ) (h : s^2 = s) (h₀ : s ≠ 0) : 4 * s = 4 :=
by {
  have s_eq_1 : s = 1 := by {
    field_simp [h],
    exact h₀,
    linarith,
  },
  rw s_eq_1,
  ring,
}

end square_perimeter_l115_115949


namespace solution_l115_115806

noncomputable def problem :=
  let A : (ℝ × ℝ × ℝ) := (a, 0, 0) in
  let B : (ℝ × ℝ × ℝ) := (0, b, 0) in
  let C : (ℝ × ℝ × ℝ) := (0, 0, c) in
  let plane_eq (x y z a b c : ℝ) : Prop := (x / a + y / b + z / c = 1) in
  let distance_eq (a b c : ℝ) : Prop :=
    (1 / (Real.sqrt ((1 / (a^2)) + 1 / (b^2) + 1 / (c^2)))) = 2 in
  let centroid (a b c : ℝ) : ℝ × ℝ × ℝ := (a / 3, b / 3, c / 3) in
  let (p, q, r) := centroid a b c in
  (plane_eq a 0 0 a b c) ∧
  (plane_eq 0 b 0 a b c) ∧
  (plane_eq 0 0 c a b c) ∧
  (distance_eq a b c) → (1 / (p^2) + 1 / (q^2) + 1 / (r^2)) = 2.25

theorem solution : problem :=
sorry

end solution_l115_115806


namespace total_swimming_hours_over_4_weeks_l115_115273

def weekday_swimming_per_day : ℕ := 2  -- Percy swims 2 hours per weekday
def weekday_days_per_week : ℕ := 5     -- Percy swims for 5 days a week
def weekend_swimming_per_week : ℕ := 3 -- Percy swims 3 hours on the weekend
def weeks : ℕ := 4                     -- The number of weeks is 4

-- Define the total swimming hours over 4 weeks
theorem total_swimming_hours_over_4_weeks :
  weekday_swimming_per_day * weekday_days_per_week * weeks + weekend_swimming_per_week * weeks = 52 :=
by
  sorry

end total_swimming_hours_over_4_weeks_l115_115273


namespace parabola_directrix_l115_115748

theorem parabola_directrix (x y : ℝ) :
  (∃ a b c : ℝ, y = (a * x^2 + b * x + c) / 12 ∧ a = 1 ∧ b = -6 ∧ c = 5) →
  y = -10 / 3 :=
by
  sorry

end parabola_directrix_l115_115748


namespace angle_triple_complement_l115_115672

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l115_115672


namespace smallest_x_y_z_sum_l115_115936

theorem smallest_x_y_z_sum :
  ∃ x y z : ℝ, x + 3*y + 6*z = 1 ∧ x*y + 2*x*z + 6*y*z = -8 ∧ x*y*z = 2 ∧ x + y + z = -(8/3) := 
sorry

end smallest_x_y_z_sum_l115_115936


namespace f_f_1_equals_4_l115_115546

noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then x + 1 / (x - 2) else x^2 + 2

theorem f_f_1_equals_4 : f (f 1) = 4 := by sorry

end f_f_1_equals_4_l115_115546


namespace probability_all_same_color_is_correct_l115_115854

-- Definitions of quantities
def yellow_marbles := 3
def green_marbles := 7
def purple_marbles := 5
def total_marbles := yellow_marbles + green_marbles + purple_marbles

-- Calculation of drawing 4 marbles all the same color
def probability_all_yellow : ℚ := (yellow_marbles / total_marbles) * ((yellow_marbles - 1) / (total_marbles - 1)) * ((yellow_marbles - 2) / (total_marbles - 2)) * ((yellow_marbles - 3) / (total_marbles - 3))
def probability_all_green : ℚ := (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) * ((green_marbles - 2) / (total_marbles - 2)) * ((green_marbles - 3) / (total_marbles - 3))
def probability_all_purple : ℚ := (purple_marbles / total_marbles) * ((purple_marbles - 1) / (total_marbles - 1)) * ((purple_marbles - 2) / (total_marbles - 2)) * ((purple_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles all the same color
def total_probability_same_color : ℚ := probability_all_yellow + probability_all_green + probability_all_purple

-- Theorem statement
theorem probability_all_same_color_is_correct : total_probability_same_color = 532 / 4095 :=
by
  sorry

end probability_all_same_color_is_correct_l115_115854


namespace expand_polynomial_eq_l115_115193

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l115_115193


namespace angle_triple_complement_l115_115676

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l115_115676


namespace equal_heights_of_cylinder_and_cone_l115_115723

theorem equal_heights_of_cylinder_and_cone
  (r h : ℝ)
  (hc : h > 0)
  (hr : r > 0)
  (V_cylinder V_cone : ℝ)
  (V_cylinder_eq : V_cylinder = π * r ^ 2 * h)
  (V_cone_eq : V_cone = 1/3 * π * r ^ 2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
h = h := -- Since we are given that the heights are initially the same
sorry

end equal_heights_of_cylinder_and_cone_l115_115723


namespace power_function_decreasing_m_l115_115773

theorem power_function_decreasing_m :
  ∀ (m : ℝ), (m^2 - 5*m - 5) * (2*m + 1) < 0 → m = -1 :=
by
  sorry

end power_function_decreasing_m_l115_115773


namespace smaller_square_area_percentage_is_zero_l115_115331

noncomputable def area_smaller_square_percentage (r : ℝ) : ℝ :=
  let side_length_larger_square := 2 * r
  let x := 0  -- Solution from the Pythagorean step
  let area_larger_square := side_length_larger_square ^ 2
  let area_smaller_square := x ^ 2
  100 * area_smaller_square / area_larger_square

theorem smaller_square_area_percentage_is_zero (r : ℝ) :
    area_smaller_square_percentage r = 0 :=
  sorry

end smaller_square_area_percentage_is_zero_l115_115331


namespace largest_n_under_100000_l115_115473

theorem largest_n_under_100000 (n : ℕ) : 
  n < 100000 ∧ (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 42) % 7 = 0 → n = 99996 :=
by
  sorry

end largest_n_under_100000_l115_115473


namespace inequality_always_true_l115_115218

-- Definitions from the conditions
variables {a b c : ℝ}
variable (h1 : a < b)
variable (h2 : b < c)
variable (h3 : a + b + c = 0)

-- The statement to prove
theorem inequality_always_true : c * a < c * b :=
by
  -- Proof steps go here.
  sorry

end inequality_always_true_l115_115218


namespace solve_eqs_l115_115525

theorem solve_eqs (x y : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 / (2 - y) + y^2 / (2 - x) = 2) :
  x = 1 ∧ y = 1 :=
by
  sorry

end solve_eqs_l115_115525


namespace miles_to_add_per_week_l115_115571

theorem miles_to_add_per_week :
  ∀ (initial_miles_per_week : ℝ) 
    (percentage_increase : ℝ) 
    (total_days : ℕ) 
    (days_in_week : ℕ),
    initial_miles_per_week = 100 →
    percentage_increase = 0.2 →
    total_days = 280 →
    days_in_week = 7 →
    ((initial_miles_per_week * (1 + percentage_increase)) - initial_miles_per_week) / (total_days / days_in_week) = 3 :=
by
  intros initial_miles_per_week percentage_increase total_days days_in_week
  intros h1 h2 h3 h4
  sorry

end miles_to_add_per_week_l115_115571


namespace silver_nitrate_mass_fraction_l115_115731

variable (n : ℝ) (M : ℝ) (m_total : ℝ)
variable (m_agno3 : ℝ) (omega_agno3 : ℝ)

theorem silver_nitrate_mass_fraction 
  (h1 : n = 0.12) 
  (h2 : M = 170) 
  (h3 : m_total = 255)
  (h4 : m_agno3 = n * M) 
  (h5 : omega_agno3 = (m_agno3 * 100) / m_total) : 
  m_agno3 = 20.4 ∧ omega_agno3 = 8 :=
by
  -- insert proof here eventually 
  sorry

end silver_nitrate_mass_fraction_l115_115731


namespace base3_composite_numbers_l115_115843

theorem base3_composite_numbers:
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 12002110 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2210121012 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 121212 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 102102 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 1001 * AB = a * b) :=
by {
  sorry
}

end base3_composite_numbers_l115_115843


namespace min_value_of_a1_plus_a7_l115_115788

variable {a : ℕ → ℝ}
variable {a3 a5 : ℝ}

-- Conditions
def is_positive_geometric_sequence (a : ℕ → ℝ) := 
  ∀ n, a n > 0 ∧ (∃ r, ∀ i, a (i + 1) = a i * r)

def condition (a : ℕ → ℝ) (a3 a5 : ℝ) :=
  a 3 = a3 ∧ a 5 = a5 ∧ a3 * a5 = 64

-- Prove that the minimum value of a1 + a7 is 16
theorem min_value_of_a1_plus_a7
  (h1 : is_positive_geometric_sequence a)
  (h2 : condition a a3 a5) :
  ∃ a1 a7, a 1 = a1 ∧ a 7 = a7 ∧ (∃ (min_sum : ℝ), min_sum = 16 ∧ ∀ sum, sum = a1 + a7 → sum ≥ min_sum) :=
sorry

end min_value_of_a1_plus_a7_l115_115788


namespace collectively_behind_l115_115103

noncomputable def sleep_hours_behind (weeknights weekend nights_ideal: ℕ) : ℕ :=
  let total_sleep := (weeknights * 5) + (weekend * 2)
  let ideal_sleep := nights_ideal * 7
  ideal_sleep - total_sleep

def tom_weeknight := 5
def tom_weekend := 6

def jane_weeknight := 7
def jane_weekend := 9

def mark_weeknight := 6
def mark_weekend := 7

def ideal_night := 8

theorem collectively_behind :
  sleep_hours_behind tom_weeknight tom_weekend ideal_night +
  sleep_hours_behind jane_weeknight jane_weekend ideal_night +
  sleep_hours_behind mark_weeknight mark_weekend ideal_night = 34 :=
by
  sorry

end collectively_behind_l115_115103


namespace angle_measure_triple_complement_l115_115660

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l115_115660


namespace probability_sum_of_digits_eq_10_l115_115962

theorem probability_sum_of_digits_eq_10 (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1): 
  let P := m / n
  let valid_numbers := 120
  let total_numbers := 2020
  (P = valid_numbers / total_numbers) → (m = 6) → (n = 101) → (m + n = 107) :=
by 
  sorry

end probability_sum_of_digits_eq_10_l115_115962


namespace largest_m_for_game_with_2022_grids_l115_115835

variables (n : ℕ) (f : ℕ → ℕ)

/- Definitions using conditions given -/

/-- Definition of the game and the marking process -/
def game (n : ℕ) : ℕ := 
  if n % 4 = 0 then n / 2 + 1
  else if n % 4 = 2 then n / 2 + 1
  else 0

/-- Main theorem statement -/
theorem largest_m_for_game_with_2022_grids : game 2022 = 1011 :=
by sorry

end largest_m_for_game_with_2022_grids_l115_115835


namespace harry_morning_routine_l115_115364

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end harry_morning_routine_l115_115364


namespace alcohol_mixture_l115_115280

theorem alcohol_mixture (y : ℕ) :
  let x_vol := 200 -- milliliters
  let y_conc := 30 / 100 -- 30% alcohol
  let x_conc := 10 / 100 -- 10% alcohol
  let final_conc := 20 / 100 -- 20% target alcohol concentration
  let x_alcohol := x_vol * x_conc -- alcohol in x
  (x_alcohol + y * y_conc) / (x_vol + y) = final_conc ↔ y = 200 :=
by 
  sorry

end alcohol_mixture_l115_115280


namespace proof_problem_l115_115570

def polar_curve_C (ρ : ℝ) : Prop := ρ = 5

def point_P (x y : ℝ) : Prop := x = -3 ∧ y = -3 / 2

def line_l_through_P (x y : ℝ) (k : ℝ) : Prop := y + 3 / 2 = k * (x + 3)

def distance_AB (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64

theorem proof_problem
  (ρ : ℝ) (x y : ℝ) (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : polar_curve_C ρ)
  (h2 : point_P (-3) (-3 / 2))
  (h3 : ∃ k, line_l_through_P x y k)
  (h4 : distance_AB A B) :
  ∃ (x y : ℝ), (x^2 + y^2 = 25) ∧ ((x = -3) ∨ (3 * x + 4 * y + 15 = 0)) := 
sorry

end proof_problem_l115_115570


namespace cube_sum_l115_115066

-- Definitions
variable (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω^2 + ω + 1 = 0) -- nonreal root

-- Theorem statement
theorem cube_sum : (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = -2 :=
by 
  sorry

end cube_sum_l115_115066


namespace find_a_given_coefficient_l115_115563

theorem find_a_given_coefficient (a : ℝ) (h : (a^3 * 10 = 80)) : a = 2 :=
by
  sorry

end find_a_given_coefficient_l115_115563


namespace number_of_girls_l115_115131

-- Define the problem conditions as constants
def total_saplings : ℕ := 44
def teacher_saplings : ℕ := 6
def boy_saplings : ℕ := 4
def girl_saplings : ℕ := 2
def total_students : ℕ := 12
def students_saplings : ℕ := total_saplings - teacher_saplings

-- The proof problem statement
theorem number_of_girls (x y : ℕ) (h1 : x + y = total_students)
  (h2 : boy_saplings * x + girl_saplings * y = students_saplings) :
  y = 5 :=
by
  sorry

end number_of_girls_l115_115131


namespace fraction_of_repeating_decimal_l115_115889

theorem fraction_of_repeating_decimal:
  let a := (4 / 10 : ℝ)
  let r := (1 / 10 : ℝ)
  (∑' n:ℕ, a * r^n) = (4 / 9 : ℝ) := by
  sorry

end fraction_of_repeating_decimal_l115_115889


namespace total_money_l115_115986

theorem total_money (gold_value silver_value cash : ℕ) (num_gold num_silver : ℕ)
  (h_gold : gold_value = 50)
  (h_silver : silver_value = 25)
  (h_num_gold : num_gold = 3)
  (h_num_silver : num_silver = 5)
  (h_cash : cash = 30) :
  gold_value * num_gold + silver_value * num_silver + cash = 305 :=
by
  sorry

end total_money_l115_115986


namespace distance_home_to_school_l115_115118

theorem distance_home_to_school :
  ∃ T D : ℝ, 6 * (T + 7/60) = D ∧ 12 * (T - 8/60) = D ∧ 9 * T = D ∧ D = 2.1 :=
by
  sorry

end distance_home_to_school_l115_115118


namespace expand_polynomial_l115_115172

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l115_115172


namespace solution_l115_115799

noncomputable def problem_statement : Prop :=
  let O := (0, 0, 0)
  let A := (α : ℝ, 0, 0)
  let B := (0, β : ℝ, 0)
  let C := (0, 0, γ : ℝ)
  let plane_equation := λ x y z, (x / α + y / β + z / γ = 1)
  let distance_from_origin := 2
  let centroid := ((α / 3), (β / 3), (γ / 3))
  let p := centroid.1
  let q := centroid.2
  let r := centroid.3
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 2.25

theorem solution : problem_statement := 
by sorry

end solution_l115_115799


namespace triple_complement_angle_l115_115684

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l115_115684


namespace ideal_sleep_hours_l115_115472

open Nat

theorem ideal_sleep_hours 
  (weeknight_sleep : Nat)
  (weekend_sleep : Nat)
  (sleep_deficit : Nat)
  (num_weeknights : Nat := 5)
  (num_weekend_nights : Nat := 2)
  (total_nights : Nat := 7) :
  weeknight_sleep = 5 →
  weekend_sleep = 6 →
  sleep_deficit = 19 →
  ((num_weeknights * weeknight_sleep + num_weekend_nights * weekend_sleep) + sleep_deficit) / total_nights = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end ideal_sleep_hours_l115_115472


namespace find_linear_function_l115_115262

theorem find_linear_function (α : ℝ) (hα : α > 0)
  (f : ℕ+ → ℝ)
  (h : ∀ (k m : ℕ+), α * (m : ℝ) ≤ (k : ℝ) ∧ (k : ℝ) < (α + 1) * (m : ℝ) → f (k + m) = f k + f m)
: ∃ (b : ℝ), ∀ (n : ℕ+), f n = b * (n : ℝ) :=
sorry

end find_linear_function_l115_115262


namespace upstream_speed_proof_l115_115502

-- Definitions based on the conditions in the problem
def speed_in_still_water : ℝ := 25
def speed_downstream : ℝ := 35

-- The speed of the man rowing upstream
def speed_upstream : ℝ := speed_in_still_water - (speed_downstream - speed_in_still_water)

theorem upstream_speed_proof : speed_upstream = 15 := by
  -- Proof is omitted by using sorry
  sorry

end upstream_speed_proof_l115_115502


namespace mars_colony_cost_l115_115282

theorem mars_colony_cost :
  let total_cost := 45000000000
  let number_of_people := 300000000
  total_cost / number_of_people = 150 := 
by sorry

end mars_colony_cost_l115_115282


namespace equilateral_triangle_of_condition_l115_115542

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 0) :
  a = b ∧ b = c := 
sorry

end equilateral_triangle_of_condition_l115_115542


namespace triple_complement_angle_l115_115682

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l115_115682


namespace annie_total_miles_l115_115728

theorem annie_total_miles (initial_gallons : ℕ) (miles_per_gallon : ℕ)
  (initial_trip_miles : ℕ) (purchased_gallons : ℕ) (final_gallons : ℕ)
  (total_miles : ℕ) :
  initial_gallons = 12 →
  miles_per_gallon = 28 →
  initial_trip_miles = 280 →
  purchased_gallons = 6 →
  final_gallons = 5 →
  total_miles = 364 := by
  sorry

end annie_total_miles_l115_115728


namespace marble_probability_l115_115710

theorem marble_probability :
  let red := 4
      blue := 3
      yellow := 6
      green := 2
      total_marbles := red + blue + yellow + green
      favorable_marbles := red + blue + green
  in 
  (favorable_marbles / total_marbles : ℚ) = 3 / 5 :=
by
  sorry

end marble_probability_l115_115710


namespace right_triangle_hypotenuse_l115_115869

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h1 : a + b + c = 60) 
  (h2 : 0.5 * a * b = 120) 
  (h3 : a^2 + b^2 = c^2) : 
  c = 26 :=
by {
  sorry
}

end right_triangle_hypotenuse_l115_115869


namespace books_given_away_l115_115584

theorem books_given_away (original_books : ℝ) (books_left : ℝ) (books_given : ℝ) 
    (h1 : original_books = 54.0) 
    (h2 : books_left = 31) : 
    books_given = original_books - books_left → books_given = 23 :=
by
  sorry

end books_given_away_l115_115584


namespace initial_fund_is_890_l115_115955

-- Given Conditions
def initial_fund (n : ℕ) : ℝ := 60 * n - 10
def bonus_given (n : ℕ) : ℝ := 50 * n
def remaining_fund (initial : ℝ) (bonus : ℝ) : ℝ := initial - bonus

-- Proof problem: Prove that the initial amount equals $890 under the given constraints
theorem initial_fund_is_890 :
  ∃ n : ℕ, 
    initial_fund n = 890 ∧ 
    initial_fund n - bonus_given n = 140 :=
by
  sorry

end initial_fund_is_890_l115_115955


namespace expand_expression_l115_115196

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115196


namespace angle_measure_triple_complement_l115_115657

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l115_115657


namespace volume_surface_area_ratio_l115_115333

theorem volume_surface_area_ratio
  (V : ℕ := 9)
  (S : ℕ := 34)
  (shape_conditions : ∃ n : ℕ, n = 9 ∧ ∃ m : ℕ, m = 2) :
  V / S = 9 / 34 :=
by
  sorry

end volume_surface_area_ratio_l115_115333


namespace angle_triple_complement_l115_115630

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l115_115630


namespace soda_consumption_l115_115370

theorem soda_consumption 
    (dozens : ℕ)
    (people_per_dozen : ℕ)
    (cost_per_box : ℕ)
    (cans_per_box : ℕ)
    (family_members : ℕ)
    (payment_per_member : ℕ)
    (dozens_eq : dozens = 5)
    (people_per_dozen_eq : people_per_dozen = 12)
    (cost_per_box_eq : cost_per_box = 2)
    (cans_per_box_eq : cans_per_box = 10)
    (family_members_eq : family_members = 6)
    (payment_per_member_eq : payment_per_member = 4) :
  (60 * (cans_per_box)) / 60 = 2 :=
by
  -- proof would go here eventually
  sorry

end soda_consumption_l115_115370


namespace angle_measure_l115_115693

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l115_115693


namespace rosie_circles_track_24_l115_115355

-- Definition of the problem conditions
def lou_distance := 3 -- Lou's total distance in miles
def track_length := 1 / 4 -- Length of the circular track in miles
def rosie_speed_factor := 2 -- Rosie runs at twice the speed of Lou

-- Define the number of times Rosie circles the track as a result
def rosie_circles_the_track : Nat :=
  let lou_circles := lou_distance / track_length
  let rosie_distance := lou_distance * rosie_speed_factor
  let rosie_circles := rosie_distance / track_length
  rosie_circles

-- The theorem stating that Rosie circles the track 24 times
theorem rosie_circles_track_24 : rosie_circles_the_track = 24 := by
  sorry

end rosie_circles_track_24_l115_115355


namespace total_potatoes_l115_115722

open Nat

theorem total_potatoes (P T R : ℕ) (h1 : P = 5) (h2 : T = 6) (h3 : R = 48) : P + (R / T) = 13 := by
  sorry

end total_potatoes_l115_115722


namespace rectangle_to_total_height_ratio_l115_115997

theorem rectangle_to_total_height_ratio 
  (total_area : ℕ)
  (width : ℕ)
  (area_per_side : ℕ)
  (height : ℕ)
  (triangle_base : ℕ)
  (triangle_area : ℕ)
  (rect_area : ℕ)
  (total_height : ℕ)
  (ratio : ℚ)
  (h_eqn : 3 * height = area_per_side)
  (h_value : height = total_area / (2 * 3))
  (total_height_eqn : total_height = 2 * height)
  (ratio_eqn : ratio = height / total_height) :
  total_area = 12 → width = 3 → area_per_side = 6 → triangle_base = 3 →
  triangle_area = triangle_base * height / 2 → rect_area = width * height →
  rect_area = area_per_side → ratio = 1 / 2 :=
by
  intros
  sorry

end rectangle_to_total_height_ratio_l115_115997


namespace math_proof_problem_l115_115433

variables {a b c d e f k : ℝ}

theorem math_proof_problem 
  (h1 : a + b + c = d + e + f)
  (h2 : a^2 + b^2 + c^2 = d^2 + e^2 + f^2)
  (h3 : a^3 + b^3 + c^3 ≠ d^3 + e^3 + f^3) :
  (a + b + c + (d + k) + (e + k) + (f + k) = d + e + f + (a + k) + (b + k) + (c + k) ∧
   a^2 + b^2 + c^2 + (d + k)^2 + (e + k)^2 + (f + k)^2 = d^2 + e^2 + f^2 + (a + k)^2 + (b + k)^2 + (c + k)^2 ∧
   a^3 + b^3 + c^3 + (d + k)^3 + (e + k)^3 + (f + k)^3 = d^3 + e^3 + f^3 + (a + k)^3 + (b + k)^3 + (c + k)^3) 
   ∧ 
  (a^4 + b^4 + c^4 + (d + k)^4 + (e + k)^4 + (f + k)^4 ≠ d^4 + e^4 + f^4 + (a + k)^4 + (b + k)^4 + (c + k)^4) := 
  sorry

end math_proof_problem_l115_115433


namespace ratio_of_sides_l115_115276

theorem ratio_of_sides (a b c d : ℝ) 
  (h1 : a / c = 4 / 5) 
  (h2 : b / d = 4 / 5) : b / d = 4 / 5 :=
sorry

end ratio_of_sides_l115_115276


namespace find_divisor_l115_115504

theorem find_divisor (x : ℝ) (h : x / n = 0.01 * (x * n)) : n = 10 :=
sorry

end find_divisor_l115_115504


namespace angle_triple_complement_l115_115650

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l115_115650


namespace selling_price_to_achieve_profit_l115_115326

theorem selling_price_to_achieve_profit :
  ∃ (x : ℝ), let original_price := 210
              let purchase_price := 190
              let avg_sales_initial := 8
              let profit_goal := 280
              (210 - x = 200) ∧
              let profit_per_item := original_price - purchase_price - x
              let avg_sales_quantity := avg_sales_initial + 2 * x
              profit_per_item * avg_sales_quantity = profit_goal := by
  sorry

end selling_price_to_achieve_profit_l115_115326


namespace Danny_caps_vs_wrappers_l115_115516

def park_caps : ℕ := 58
def park_wrappers : ℕ := 25
def beach_caps : ℕ := 34
def beach_wrappers : ℕ := 15
def forest_caps : ℕ := 21
def forest_wrappers : ℕ := 32
def before_caps : ℕ := 12
def before_wrappers : ℕ := 11

noncomputable def total_caps : ℕ := park_caps + beach_caps + forest_caps + before_caps
noncomputable def total_wrappers : ℕ := park_wrappers + beach_wrappers + forest_wrappers + before_wrappers

theorem Danny_caps_vs_wrappers : total_caps - total_wrappers = 42 := by
  sorry

end Danny_caps_vs_wrappers_l115_115516


namespace expand_expression_l115_115167

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l115_115167


namespace isabel_games_problem_l115_115411

noncomputable def prime_sum : ℕ := 83 + 89 + 97

theorem isabel_games_problem (initial_games : ℕ) (X : ℕ) (H1 : initial_games = 90) (H2 : X = prime_sum) : X > initial_games :=
by 
  sorry

end isabel_games_problem_l115_115411


namespace simplify_expression_l115_115521

variable (a b : ℚ)

theorem simplify_expression (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  (2 * a + 1) / (1 - b / (2 * b - 1)) = (2 * a + 1) * (2 * b - 1) / (b - 1) :=
by 
  sorry

end simplify_expression_l115_115521


namespace barber_total_loss_is_120_l115_115503

-- Definitions for the conditions
def haircut_cost : ℕ := 25
def initial_payment_by_customer : ℕ := 50
def flower_shop_change : ℕ := 50
def bakery_change : ℕ := 10
def customer_received_change : ℕ := 25
def counterfeit_50_replacement : ℕ := 50
def counterfeit_10_replacement : ℕ := 10

-- Calculate total loss for the barber
def total_loss : ℕ :=
  let loss_haircut := haircut_cost
  let loss_change_to_customer := customer_received_change
  let loss_given_to_flower_shop := counterfeit_50_replacement
  let loss_given_to_bakery := counterfeit_10_replacement
  let total_loss_before_offset := loss_haircut + loss_change_to_customer + loss_given_to_flower_shop + loss_given_to_bakery
  let real_currency_received := flower_shop_change
  total_loss_before_offset - real_currency_received

-- Proof statement
theorem barber_total_loss_is_120 : total_loss = 120 := by {
  sorry
}

end barber_total_loss_is_120_l115_115503


namespace odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l115_115233

theorem odd_positive_multiples_of_7_with_units_digit_1_lt_200_count : 
  ∃ (count : ℕ), count = 3 ∧
  ∀ n : ℕ, (n % 2 = 1) → (n % 7 = 0) → (n < 200) → (n % 10 = 1) → count = 3 :=
sorry

end odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l115_115233


namespace daniel_sales_tax_l115_115515

theorem daniel_sales_tax :
  let total_cost := 25
  let tax_rate := 0.05
  let tax_free_cost := 18.7
  let tax_paid := 0.3
  exists (taxable_cost : ℝ), 
    18.7 + taxable_cost + 0.05 * taxable_cost = total_cost ∧
    taxable_cost * tax_rate = tax_paid :=
by
  sorry

end daniel_sales_tax_l115_115515


namespace John_reads_50_pages_per_hour_l115_115063

noncomputable def pages_per_hour (reads_daily hours : ℕ) (total_pages total_weeks : ℕ) : ℕ :=
  let days := total_weeks * 7
  let pages_per_day := total_pages / days
  pages_per_day / reads_daily

theorem John_reads_50_pages_per_hour :
  pages_per_hour 2 2800 4 = 50 := by
  sorry

end John_reads_50_pages_per_hour_l115_115063


namespace angle_triple_complement_l115_115695

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l115_115695


namespace at_least_one_not_less_than_two_l115_115559

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x, (x = a + 1/b ∨ x = b + 1/c ∨ x = c + 1/a) ∧ 2 ≤ x :=
by
  sorry

end at_least_one_not_less_than_two_l115_115559


namespace parity_of_E2021_E2022_E2023_l115_115019

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 0
  else seq (n - 2) + seq (n - 3)

theorem parity_of_E2021_E2022_E2023 :
  is_odd (seq 2021) ∧ is_even (seq 2022) ∧ is_odd (seq 2023) :=
by
  sorry

end parity_of_E2021_E2022_E2023_l115_115019


namespace volume_of_prism_l115_115102

theorem volume_of_prism (a b c : ℝ) (h₁ : a * b = 48) (h₂ : b * c = 36) (h₃ : a * c = 50) : 
    (a * b * c = 170) :=
by
  sorry

end volume_of_prism_l115_115102


namespace measure_angle_Z_l115_115256

-- Given conditions
def triangle_condition (X Y Z : ℝ) :=
   X = 78 ∧ Y = 4 * Z - 14

-- Triangle angle sum property
def triangle_angle_sum (X Y Z : ℝ) :=
   X + Y + Z = 180

-- Prove the measure of angle Z
theorem measure_angle_Z (X Y Z : ℝ) (h1 : triangle_condition X Y Z) (h2 : triangle_angle_sum X Y Z) : 
  Z = 23.2 :=
by
  -- Lean will expect proof steps here, ‘sorry’ is used to denote unproven parts.
  sorry

end measure_angle_Z_l115_115256


namespace product_of_reciprocals_plus_one_geq_nine_l115_115376

theorem product_of_reciprocals_plus_one_geq_nine
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hab : a + b = 1) :
  (1 / a + 1) * (1 / b + 1) ≥ 9 :=
sorry

end product_of_reciprocals_plus_one_geq_nine_l115_115376


namespace angle_measure_triple_complement_l115_115616

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l115_115616


namespace woman_lawyer_probability_l115_115496

-- Defining conditions
def total_members : ℝ := 100
def percent_women : ℝ := 0.90
def percent_women_lawyers : ℝ := 0.60

-- Calculating numbers based on the percentages
def number_women : ℝ := percent_women * total_members
def number_women_lawyers : ℝ := percent_women_lawyers * number_women

-- Statement of the problem in Lean 4
theorem woman_lawyer_probability :
  (number_women_lawyers / total_members) = 0.54 :=
by sorry

end woman_lawyer_probability_l115_115496


namespace angle_measure_triple_complement_l115_115624

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l115_115624


namespace simplify_expression_l115_115444

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115444


namespace centroid_plane_distance_l115_115801

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end centroid_plane_distance_l115_115801


namespace common_ratio_of_geometric_seq_l115_115296

variable {α : Type*} [Field α]

-- Definition of the geometric sequence
def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ n

-- Sum of the first three terms of the geometric sequence
def sum_first_three_terms (a q: α) : α :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

theorem common_ratio_of_geometric_seq (a q : α) (h : sum_first_three_terms a q = 3 * a) : q = 1 ∨ q = -2 :=
sorry

end common_ratio_of_geometric_seq_l115_115296


namespace part1_part2_l115_115903

-- Part (1)
theorem part1 (x : ℝ) (m : ℝ) (h : x = 2) : 
  (x / (x - 3) + m / (3 - x) = 3) → m = 5 :=
sorry

-- Part (2)
theorem part2 (x : ℝ) (m : ℝ) :
  (x / (x - 3) + m / (3 - x) = 3) → (x > 0) → (m < 9) ∧ (m ≠ 3) :=
sorry

end part1_part2_l115_115903


namespace simplify_vectors_l115_115947

variables {Point : Type} [AddGroup Point] (A B C D : Point)

def vector (P Q : Point) : Point := Q - P

theorem simplify_vectors :
  vector A B + vector B C - vector A D = vector D C :=
by
  sorry

end simplify_vectors_l115_115947


namespace perimeter_of_one_rectangle_l115_115467

theorem perimeter_of_one_rectangle (s : ℝ) (rectangle_perimeter rectangle_length rectangle_width : ℝ) (h1 : 4 * s = 240) (h2 : rectangle_width = (1/2) * s) (h3 : rectangle_length = s) (h4 : rectangle_perimeter = 2 * (rectangle_length + rectangle_width)) :
  rectangle_perimeter = 180 := 
sorry

end perimeter_of_one_rectangle_l115_115467


namespace fiona_pairs_l115_115369

-- Define the combinatorial calculation using the combination formula
def combination (n k : ℕ) := n.choose k

-- The main theorem stating that the number of pairs from 6 people is 15
theorem fiona_pairs : combination 6 2 = 15 :=
by
  sorry

end fiona_pairs_l115_115369


namespace sequence_23rd_term_is_45_l115_115927

def sequence_game (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * n - 1 else 2 * n + 1

theorem sequence_23rd_term_is_45 :
  sequence_game 23 = 45 :=
by
  -- Proving the 23rd term in the sequence as given by the game rules
  sorry

end sequence_23rd_term_is_45_l115_115927


namespace simplify_expression_l115_115443

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115443


namespace toot_has_vertical_symmetry_l115_115715

def has_vertical_symmetry (letter : Char) : Prop :=
  letter = 'T' ∨ letter = 'O'

def word_has_vertical_symmetry (word : List Char) : Prop :=
  ∀ letter ∈ word, has_vertical_symmetry letter

theorem toot_has_vertical_symmetry : word_has_vertical_symmetry ['T', 'O', 'O', 'T'] :=
  by
    sorry

end toot_has_vertical_symmetry_l115_115715


namespace convert_157_base_10_to_base_7_l115_115740

-- Given
def base_10_to_base_7(n : ℕ) : String := "313"

-- Prove
theorem convert_157_base_10_to_base_7 : base_10_to_base_7 157 = "313" := by
  sorry

end convert_157_base_10_to_base_7_l115_115740


namespace sequence_periodicity_l115_115275

theorem sequence_periodicity (a : ℕ → ℕ) (n : ℕ) (h : ∀ k, a k = 6^k) :
  a (n + 5) % 100 = a n % 100 :=
by sorry

end sequence_periodicity_l115_115275


namespace number_of_zeros_of_f_l115_115464

noncomputable def f (x : ℝ) : ℝ := 2 * x * |Real.log x / Real.log 0.5| - 1

theorem number_of_zeros_of_f : {x : ℝ | f x = 0}.finite ∧ {x : ℝ | f x = 0}.card = 2 :=
by
  sorry

end number_of_zeros_of_f_l115_115464


namespace find_x_l115_115832

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

end find_x_l115_115832


namespace expand_polynomial_l115_115173

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l115_115173


namespace find_acute_angle_of_parallel_vectors_l115_115543

open Real

theorem find_acute_angle_of_parallel_vectors (x : ℝ) (hx1 : (sin x) * (1 / 2 * cos x) = 1 / 4) (hx2 : 0 < x ∧ x < π / 2) : x = π / 4 :=
by
  sorry

end find_acute_angle_of_parallel_vectors_l115_115543


namespace a5_a6_values_b_n_general_formula_minimum_value_T_n_l115_115221

section sequence_problems

def sequence_n (n : ℕ) : ℤ :=
if n = 0 then 1
else if n = 1 then 1
else sequence_n (n - 2) + 2 * (-1)^(n - 2)

def b_sequence (n : ℕ) : ℤ :=
sequence_n (2 * n)

def S_n (n : ℕ) : ℤ :=
(n + 1) * (sequence_n n)

def T_n (n : ℕ) : ℤ :=
(S_n (2 * n) - 18)

theorem a5_a6_values :
  sequence_n 4 = -3 ∧ sequence_n 5 = 5 := by
  sorry

theorem b_n_general_formula (n : ℕ) :
  b_sequence n = 2 * n - 1 := by
  sorry

theorem minimum_value_T_n :
  ∃ n, T_n n = -72 := by
  sorry

end sequence_problems

end a5_a6_values_b_n_general_formula_minimum_value_T_n_l115_115221


namespace polynomial_factor_implies_a_minus_b_l115_115766

theorem polynomial_factor_implies_a_minus_b (a b : ℝ) :
  (∀ x y : ℝ, (x + y - 2) ∣ (x^2 + a * x * y + b * y^2 - 5 * x + y + 6))
  → a - b = 1 :=
by
  intro h
  -- Proof needs to be filled in
  sorry

end polynomial_factor_implies_a_minus_b_l115_115766


namespace complement_inter_section_l115_115265

-- Define the sets M and N
def M : Set ℝ := { x | x^2 - 2*x - 3 >= 0 }
def N : Set ℝ := { x | abs (x - 2) <= 1 }

-- Define the complement of M in ℝ
def compl_M : Set ℝ := { x | -1 < x ∧ x < 3 }

-- Define the expected result set
def expected_set : Set ℝ := { x | 1 <= x ∧ x < 3 }

-- State the theorem to prove
theorem complement_inter_section : compl_M ∩ N = expected_set := by
  sorry

end complement_inter_section_l115_115265


namespace expand_product_l115_115159

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l115_115159


namespace mr_li_age_l115_115849

theorem mr_li_age (xiaofang_age : ℕ) (h1 : xiaofang_age = 5)
  (h2 : ∀ t : ℕ, (t = 3) → ∀ mr_li_age_in_3_years : ℕ, (mr_li_age_in_3_years = xiaofang_age + t + 20)) :
  ∃ mr_li_age : ℕ, mr_li_age = 25 :=
by
  sorry

end mr_li_age_l115_115849


namespace train_cross_time_l115_115968

noncomputable def time_to_cross_pole (length: ℝ) (speed_kmh: ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_time :
  let length := 100
  let speed := 126
  abs (time_to_cross_pole length speed - 2.8571) < 0.0001 :=
by
  let length := 100
  let speed := 126
  have h1 : abs (time_to_cross_pole length speed - 2.8571) < 0.0001
  sorry
  exact h1

end train_cross_time_l115_115968


namespace inequality_always_true_l115_115217

-- Definitions from the conditions
variables {a b c : ℝ}
variable (h1 : a < b)
variable (h2 : b < c)
variable (h3 : a + b + c = 0)

-- The statement to prove
theorem inequality_always_true : c * a < c * b :=
by
  -- Proof steps go here.
  sorry

end inequality_always_true_l115_115217


namespace sandy_money_l115_115969

theorem sandy_money (x : ℝ) (h : 0.70 * x = 210) : x = 300 := by
sorry

end sandy_money_l115_115969


namespace find_real_a_l115_115754

theorem find_real_a (a : ℝ) : 
  (a ^ 2 + 2 * a - 15 = 0) ∧ (a ^ 2 + 4 * a - 5 ≠ 0) → a = 3 :=
by 
  sorry

end find_real_a_l115_115754


namespace xyz_ratio_l115_115026

theorem xyz_ratio (k x y z : ℝ) (h1 : x + k * y + 3 * z = 0)
                                (h2 : 3 * x + k * y - 2 * z = 0)
                                (h3 : 2 * x + 4 * y - 3 * z = 0)
                                (x_ne_zero : x ≠ 0)
                                (y_ne_zero : y ≠ 0)
                                (z_ne_zero : z ≠ 0) :
  (k = 11) → (x * z) / (y ^ 2) = 10 := by
  sorry

end xyz_ratio_l115_115026


namespace root_of_equation_l115_115292

theorem root_of_equation :
  ∀ x : ℝ, (x - 3)^2 = x - 3 ↔ x = 3 ∨ x = 4 :=
by
  sorry

end root_of_equation_l115_115292


namespace lydia_flowers_on_porch_l115_115267

theorem lydia_flowers_on_porch:
  ∀ (total_plants : ℕ) (flowering_percentage : ℚ) (fraction_on_porch : ℚ) (flowers_per_plant : ℕ),
  total_plants = 80 →
  flowering_percentage = 0.40 →
  fraction_on_porch = 1 / 4 →
  flowers_per_plant = 5 →
  let flowering_plants := (total_plants : ℚ) * flowering_percentage in
  let porch_plants := flowering_plants * fraction_on_porch in
  let total_flowers_on_porch := porch_plants * (flowers_per_plant : ℚ) in
  total_flowers_on_porch = 40 :=
by {
  intros total_plants flowering_percentage fraction_on_porch flowers_per_plant,
  intros h1 h2 h3 h4,
  let flowering_plants := (total_plants : ℚ) * flowering_percentage,
  let porch_plants := flowering_plants * fraction_on_porch,
  let total_flowers_on_porch := porch_plants * (flowers_per_plant : ℚ),
  sorry
}

end lydia_flowers_on_porch_l115_115267


namespace four_mutually_acquainted_l115_115920

theorem four_mutually_acquainted (G : SimpleGraph (Fin 9)) 
  (h : ∀ (s : Finset (Fin 9)), s.card = 3 → ∃ (u v : Fin 9), u ∈ s ∧ v ∈ s ∧ G.Adj u v) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (u v : Fin 9), u ∈ s → v ∈ s → G.Adj u v :=
by
  sorry

end four_mutually_acquainted_l115_115920


namespace two_fruits_probability_l115_115007

noncomputable def prob_exactly_two_fruits : ℚ := 10 / 9

theorem two_fruits_probability :
  (∀ (f : ℕ → ℝ), (f 0 = 1/3) ∧ (f 1 = 1/3) ∧ (f 2 = 1/3) ∧
   (∃ f1 f2 f3, f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    (f1 + f2 + f3 = prob_exactly_two_fruits))) :=
sorry

end two_fruits_probability_l115_115007


namespace total_pears_after_giving_away_l115_115003

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17
def carlos_pears : ℕ := 25
def pears_given_away_per_person : ℕ := 5

theorem total_pears_after_giving_away :
  (alyssa_pears + nancy_pears + carlos_pears) - (3 * pears_given_away_per_person) = 69 :=
by
  sorry

end total_pears_after_giving_away_l115_115003


namespace sum_of_squares_mul_l115_115245

theorem sum_of_squares_mul (a b c d : ℝ) :
(a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 :=
by
  sorry

end sum_of_squares_mul_l115_115245


namespace arithmetic_sequence_general_term_geometric_sequence_inequality_l115_115035

-- Sequence {a_n} and its sum S_n
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := (Finset.range n).sum a

-- Sequence {b_n}
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  2 * (S a (n + 1) - S a n) * (S a n) - n * (S a (n + 1) + S a n)

-- Arithmetic sequence and related conditions
theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : ∀ n, b a n = 0) :
  (∀ n, a n = 0) ∨ (∀ n, a n = n) :=
sorry

-- Conditions for sequences and finding the set of positive integers n
theorem geometric_sequence_inequality (a : ℕ → ℤ)
  (h1 : a 1 = 1) (h2 : a 2 = 3)
  (h3 : ∀ n, a (2 * n - 1) = 2^(n-1))
  (h4 : ∀ n, a (2 * n) = 3 * 2^(n-1)) :
  {n : ℕ | b a (2 * n) < b a (2 * n - 1)} = {1, 2, 3, 4, 5, 6} :=
sorry

end arithmetic_sequence_general_term_geometric_sequence_inequality_l115_115035


namespace ceil_sqrt_169_eq_13_l115_115745

theorem ceil_sqrt_169_eq_13 : Int.ceil (Real.sqrt 169) = 13 := by
  sorry

end ceil_sqrt_169_eq_13_l115_115745


namespace smallest_odd_digit_n_l115_115579

theorem smallest_odd_digit_n {n : ℕ} (h : n > 1) : 
  (∀ d ∈ (Nat.digits 10 (9997 * n)), d % 2 = 1) → n = 3335 :=
sorry

end smallest_odd_digit_n_l115_115579


namespace rowing_upstream_speed_l115_115862

theorem rowing_upstream_speed (V_m V_down V_up V_s : ℝ) 
  (hVm : V_m = 40) 
  (hVdown : V_down = 60) 
  (hVdown_eq : V_down = V_m + V_s) 
  (hVup_eq : V_up = V_m - V_s) : 
  V_up = 20 := 
by
  sorry

end rowing_upstream_speed_l115_115862


namespace gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l115_115121

-- GCD as the greatest common divisor
def GCD (a b : ℕ) : ℕ := Nat.gcd a b

-- LCM as the least common multiple
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- First proof problem in Lean 4
theorem gcd_lcm_relation (a b : ℕ) : GCD a b = (a * b) / (LCM a b) :=
  sorry

-- GCD function extended to three arguments
def GCD3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- LCM function extended to three arguments
def LCM3 (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- Second proof problem in Lean 4
theorem gcd3_lcm3_relation (a b c : ℕ) : GCD3 a b c = (a * b * c * LCM3 a b c) / (LCM a b * LCM b c * LCM c a) :=
  sorry

-- Third proof problem in Lean 4
theorem lcm3_gcd3_relation (a b c : ℕ) : LCM3 a b c = (a * b * c * GCD3 a b c) / (GCD a b * GCD b c * GCD c a) :=
  sorry

end gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l115_115121


namespace initial_action_figures_l115_115796

theorem initial_action_figures (x : ℕ) (h1 : x + 2 = 10) : x = 8 := 
by sorry

end initial_action_figures_l115_115796


namespace expand_product_l115_115158

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l115_115158


namespace angle_measure_triple_complement_l115_115617

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l115_115617


namespace C1_cartesian_eq_C2_rect_eq_min_dist_PQ_l115_115409

noncomputable theory

section

-- Conditions
variables (α θ : ℝ)

def C1_param_x (α : ℝ) := (√3) * Real.cos α
def C1_param_y (α : ℝ) := Real.sin α

def C2_polar_eq (ρ θ : ℝ) := ρ * Real.sin (θ + (π / 4)) = 2 * (√2)

-- Cartesian equation for C1
theorem C1_cartesian_eq :
  ∀ (α : ℝ), (C1_param_x α) ^ 2 / 3 + (C1_param_y α) ^ 2 = 1 := 
by
  assume α
  have h1 : (C1_param_x α) ^ 2 = (√3 * Real.cos α) ^ 2 := rfl
  have h2: (C1_param_y α) ^ 2 = (Real.sin α) ^ 2 := rfl
  rw [h1, h2]
  sorry

-- Rectangular equation for C2
theorem C2_rect_eq (ρ θ : ℝ) (hρ : ρ * Real.sin (θ + (π / 4)) = 2 * (√2)) :
  ρ * Real.cos θ + ρ * Real.sin θ = 4 :=
by
  have h3: Real.sin (θ + π / 4) = (Real.sqrt 2 / 2) * Real.sin θ + (Real.sqrt 2 / 2) * Real.cos θ,
  { rw [Real.sin_add, Real.cos_pi_div_four, Real.sin_pi_div_four] }
  rw [h3] at hρ
  sorry

-- Minimum distance |PQ|
theorem min_dist_PQ (α θ : ℝ) (hρ : ρ * Real.sin (θ + (π / 4)) = 2 * (√2)) :
  ∃ P : ℝ × ℝ, ∃ Q : ℝ × ℝ, 
    P = (C1_param_x α, C1_param_y α) ∧
    (Q.1 + Q.2 = 4) ∧
    dist P Q = √2 ∧
    P = (3 / 2, 1 / 2) :=
by
  sorry

end

end C1_cartesian_eq_C2_rect_eq_min_dist_PQ_l115_115409


namespace solution_l115_115800

noncomputable def problem_statement : Prop :=
  let O := (0, 0, 0)
  let A := (α : ℝ, 0, 0)
  let B := (0, β : ℝ, 0)
  let C := (0, 0, γ : ℝ)
  let plane_equation := λ x y z, (x / α + y / β + z / γ = 1)
  let distance_from_origin := 2
  let centroid := ((α / 3), (β / 3), (γ / 3))
  let p := centroid.1
  let q := centroid.2
  let r := centroid.3
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 2.25

theorem solution : problem_statement := 
by sorry

end solution_l115_115800


namespace largest_prime_factor_of_4752_l115_115311

theorem largest_prime_factor_of_4752 : ∃ p : ℕ, nat.prime p ∧ p ∣ 4752 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 4752 → q ≤ p :=
begin
  -- Proof goes here
  sorry
end

end largest_prime_factor_of_4752_l115_115311


namespace amount_distribution_l115_115488

theorem amount_distribution :
  ∃ (P Q R S T : ℝ), 
    (P + Q + R + S + T = 24000) ∧ 
    (R = (3 / 5) * (P + Q)) ∧ 
    (S = 0.45 * 24000) ∧ 
    (T = (1 / 2) * R) ∧ 
    (P + Q = 7000) ∧ 
    (R = 4200) ∧ 
    (S = 10800) ∧ 
    (T = 2100) :=
by
  sorry

end amount_distribution_l115_115488


namespace range_of_ab_l115_115068

noncomputable def f (x : ℝ) : ℝ := abs (2 - x^2)

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : 0 < a * b ∧ a * b < 2 :=
by
  sorry

end range_of_ab_l115_115068


namespace union_of_intervals_l115_115072

open Set

variable {α : Type*}

theorem union_of_intervals : 
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  A ∪ B = Ioo (-1 : ℝ) 2 := 
by
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  sorry

end union_of_intervals_l115_115072


namespace original_profit_percentage_l115_115250

theorem original_profit_percentage (C S : ℝ) 
  (h1 : S - 1.12 * C = 0.5333333333333333 * S) : 
  ((S - C) / C) * 100 = 140 :=
sorry

end original_profit_percentage_l115_115250


namespace angle_measure_triple_complement_l115_115621

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l115_115621


namespace equalize_cheese_pieces_l115_115961

-- Defining the initial masses of the three pieces of cheese
def cheese1 : ℕ := 5
def cheese2 : ℕ := 8
def cheese3 : ℕ := 11

-- State that the fox can cut 1g simultaneously from any two pieces
def can_equalize_masses (cut_action : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ n1 n2 n3 _ : ℕ,
    cut_action cheese1 cheese2 cheese3 ∧
    (n1 = 0 ∧ n2 = 0 ∧ n3 = 0)

-- Introducing the fox's cut action
def cut_action (a b c : ℕ) : Prop :=
  (∃ x : ℕ, x ≥ 0 ∧ a - x ≥ 0 ∧ b - x ≥ 0 ∧ c ≤ cheese3) ∧
  (∃ y : ℕ, y ≥ 0 ∧ a - y ≥ 0 ∧ b ≤ cheese2 ∧ c - y ≥ 0) ∧
  (∃ z : ℕ, z ≥ 0 ∧ a ≤ cheese1 ∧ b - z ≥ 0 ∧ c - z ≥ 0) 

-- The theorem that proves it's possible to equalize the masses
theorem equalize_cheese_pieces : can_equalize_masses cut_action :=
by
  sorry

end equalize_cheese_pieces_l115_115961


namespace expand_expression_l115_115181

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115181


namespace minimum_employees_needed_l115_115501

def min_new_employees (water_pollution: ℕ) (air_pollution: ℕ) (both: ℕ) : ℕ :=
  119 + 34

theorem minimum_employees_needed : min_new_employees 98 89 34 = 153 := 
  by
  sorry

end minimum_employees_needed_l115_115501


namespace min_value_expression_l115_115934

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ m : ℝ, (m = 4 + 6 * Real.sqrt 2) ∧ 
  ∀ a b : ℝ, (0 < a) → (0 < b) → m ≤ (Real.sqrt ((a^2 + b^2) * (2*a^2 + 4*b^2))) / (a * b) :=
by sorry

end min_value_expression_l115_115934


namespace angle_measure_l115_115692

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l115_115692


namespace expand_expression_l115_115165

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l115_115165


namespace christina_speed_l115_115793

theorem christina_speed
  (d v_j v_l t : ℝ)
  (D_l : ℝ)
  (h_d : d = 360)
  (h_v_j : v_j = 5)
  (h_v_l : v_l = 12)
  (h_D_l : D_l = 360)
  (h_t : t = D_l / v_l)
  (h_distance : d = v_j * t + c * t) :
  c = 7 :=
by
  sorry

end christina_speed_l115_115793


namespace hyperbola_eccentricity_sqrt2_l115_115285

noncomputable def isHyperbolaPerpendicularAsymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  let asymptote1 := (1/a : ℝ)
  let asymptote2 := (-1/b : ℝ)
  asymptote1 * asymptote2 = -1

theorem hyperbola_eccentricity_sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  isHyperbolaPerpendicularAsymptotes a b ha hb →
  let e := Real.sqrt (1 + (b^2 / a^2))
  e = Real.sqrt 2 :=
by
  intro h
  sorry

end hyperbola_eccentricity_sqrt2_l115_115285


namespace no_positive_integer_makes_expression_integer_l115_115886

theorem no_positive_integer_makes_expression_integer : 
  ∀ n : ℕ, n > 0 → ¬ ∃ k : ℤ, (n^(3 * n - 2) - 3 * n + 1) = k * (3 * n - 2) := 
by 
  intro n hn
  sorry

end no_positive_integer_makes_expression_integer_l115_115886


namespace find_log2_sum_l115_115895

axiom arithmetic_sequence (a : ℕ → ℤ) (h_arith : ∀ n ≥ 2, a (n + 1) + a (n - 1) = a n ^ 2) : Prop

axiom geometric_sequence (b : ℕ → ℤ) (h_geom : ∀ n ≥ 2, b (n + 1) * b (n - 1) = 2 * b n) : Prop

theorem find_log2_sum (a : ℕ → ℤ) (b : ℕ → ℤ)
  (h_arith : ∀ n ≥ 2, a (n + 1) + a (n - 1) = a n ^ 2)
  (h_geom : ∀ n ≥ 2, b (n + 1) * b (n - 1) = 2 * b n)
  (ha_pos : ∀ n, 0 < a n)
  (hb_pos : ∀ n, 0 < b n) :
  Real.log2 (a 2 + b 2) = 2 := by 
  sorry

end find_log2_sum_l115_115895


namespace find_average_speed_l115_115485

noncomputable def average_speed (distance1 distance2 : ℝ) (time1 time2 : ℝ) : ℝ := 
  (distance1 + distance2) / (time1 + time2)

theorem find_average_speed :
  average_speed 1000 1000 10 4 = 142.86 := by
  sorry

end find_average_speed_l115_115485


namespace sum_of_distinct_values_of_intersections_l115_115211

theorem sum_of_distinct_values_of_intersections (N : ℕ) :
  (∃ v : Finset ℕ, (∀ n ∈ v, 2 ≤ n ∧ n ≤ 10) ∧ (v.card = 9) ∧ (v.sum = 54))
by {
  -- The proof is omitted
  sorry
}

end sum_of_distinct_values_of_intersections_l115_115211


namespace expand_expression_l115_115182

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115182


namespace num_valid_arrangements_l115_115508

-- Define the people and the days of the week
inductive Person := | A | B | C | D | E
inductive DayOfWeek := | Monday | Tuesday | Wednesday | Thursday | Friday

-- Define the arrangement function type
def Arrangement := DayOfWeek → Person

/-- The total number of valid arrangements for 5 people
    (A, B, C, D, E) on duty from Monday to Friday such that:
    - A and B are not on duty on adjacent days,
    - B and C are on duty on adjacent days,
    is 36.
-/
theorem num_valid_arrangements : 
  ∃ (arrangements : Finset (Arrangement)), arrangements.card = 36 ∧
  (∀ (x : Arrangement), x ∈ arrangements →
    (∀ (d1 d2 : DayOfWeek), 
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) →
      ¬(x d1 = Person.A ∧ x d2 = Person.B)) ∧
    (∃ (d1 d2 : DayOfWeek),
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) ∧
      (x d1 = Person.B ∧ x d2 = Person.C)))
  := sorry

end num_valid_arrangements_l115_115508


namespace find_number_of_valid_polynomials_l115_115209

noncomputable def number_of_polynomials_meeting_constraints : Nat :=
  sorry

theorem find_number_of_valid_polynomials : number_of_polynomials_meeting_constraints = 11 :=
  sorry

end find_number_of_valid_polynomials_l115_115209


namespace complement_computation_l115_115970

open Set

theorem complement_computation (U A : Set ℕ) :
  U = {1, 2, 3, 4, 5, 6, 7} → A = {2, 4, 5} →
  U \ A = {1, 3, 6, 7} :=
by
  intros hU hA
  rw [hU, hA]
  ext
  simp
  sorry

end complement_computation_l115_115970


namespace number_of_strawberries_stolen_l115_115046

-- Define the conditions
def daily_harvest := 5
def days_in_april := 30
def strawberries_given_away := 20
def strawberries_left_by_end := 100

-- Calculate total harvested strawberries
def total_harvest := daily_harvest * days_in_april
-- Calculate strawberries after giving away
def remaining_after_giveaway := total_harvest - strawberries_given_away

-- Prove the number of strawberries stolen
theorem number_of_strawberries_stolen : remaining_after_giveaway - strawberries_left_by_end = 30 := by
  sorry

end number_of_strawberries_stolen_l115_115046


namespace triple_complement_angle_l115_115681

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l115_115681


namespace villages_population_equal_l115_115489

def population_x (initial_population rate_decrease : Int) (n : Int) := initial_population - rate_decrease * n
def population_y (initial_population rate_increase : Int) (n : Int) := initial_population + rate_increase * n

theorem villages_population_equal
    (initial_population_x : Int) (rate_decrease_x : Int)
    (initial_population_y : Int) (rate_increase_y : Int)
    (h₁ : initial_population_x = 76000) (h₂ : rate_decrease_x = 1200)
    (h₃ : initial_population_y = 42000) (h₄ : rate_increase_y = 800) :
    ∃ n : Int, population_x initial_population_x rate_decrease_x n = population_y initial_population_y rate_increase_y n ∧ n = 17 :=
by
    sorry

end villages_population_equal_l115_115489


namespace twelve_year_olds_count_l115_115405

theorem twelve_year_olds_count (x y z w : ℕ) 
  (h1 : x + y + z + w = 23)
  (h2 : 10 * x + 11 * y + 12 * z + 13 * w = 253)
  (h3 : z = 3 * w / 2) : 
  z = 6 :=
by sorry

end twelve_year_olds_count_l115_115405


namespace sequence_divisibility_24_l115_115142

theorem sequence_divisibility_24 :
  ∀ (x : ℕ → ℕ), (x 0 = 2) → (x 1 = 3) →
    (∀ n : ℕ, x (n+2) = 7 * x (n+1) - x n + 280) →
    (∀ n : ℕ, (x n * x (n+1) + x (n+1) * x (n+2) + x (n+2) * x (n+3) + 2018) % 24 = 0) :=
by
  intro x h1 h2 h3
  sorry

end sequence_divisibility_24_l115_115142


namespace solution_set_of_inequality_l115_115605

theorem solution_set_of_inequality (x : ℝ) : 
  (x ≠ 0 ∧ (x * (x - 1)) ≤ 0) ↔ 0 < x ∧ x ≤ 1 :=
sorry

end solution_set_of_inequality_l115_115605


namespace cashier_adjustment_l115_115856

-- Define the conditions
variables {y : ℝ}

-- Error calculation given the conditions
def half_dollar_error (y : ℝ) : ℝ := 0.50 * y
def five_dollar_error (y : ℝ) : ℝ := 5 * y
def total_error (y : ℝ) : ℝ := half_dollar_error y + five_dollar_error y

-- Theorem statement
theorem cashier_adjustment (y : ℝ) : total_error y = 5.50 * y :=
sorry

end cashier_adjustment_l115_115856


namespace angle_triple_complement_l115_115652

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l115_115652


namespace total_handshakes_l115_115236

theorem total_handshakes (n : ℕ) (h : n = 10) : ∃ k, k = (n * (n - 1)) / 2 ∧ k = 45 :=
by {
  sorry
}

end total_handshakes_l115_115236


namespace sqrt_three_irrational_l115_115115

theorem sqrt_three_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a:ℝ) / b = Real.sqrt 3 :=
sorry

end sqrt_three_irrational_l115_115115


namespace transaction_gain_per_year_l115_115867

noncomputable def principal : ℝ := 9000
noncomputable def time : ℝ := 2
noncomputable def rate_lending : ℝ := 6
noncomputable def rate_borrowing : ℝ := 4

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def total_interest_earned := simple_interest principal rate_lending time
noncomputable def total_interest_paid := simple_interest principal rate_borrowing time

noncomputable def total_gain := total_interest_earned - total_interest_paid
noncomputable def gain_per_year := total_gain / 2

theorem transaction_gain_per_year : gain_per_year = 180 :=
by
  sorry

end transaction_gain_per_year_l115_115867


namespace find_f_half_l115_115763

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def f_condition (f : R → R) : Prop := ∀ x : R, x < 0 → f x = 1 / (x + 1)

theorem find_f_half (f : R → R) (h_odd : odd_function f) (h_condition : f_condition f) : f (1 / 2) = -2 := by
  sorry

end find_f_half_l115_115763


namespace man_son_age_ratio_is_two_to_one_l115_115863

-- Define the present age of the son
def son_present_age := 33

-- Define the present age of the man
def man_present_age := son_present_age + 35

-- Define the son's age in two years
def son_age_in_two_years := son_present_age + 2

-- Define the man's age in two years
def man_age_in_two_years := man_present_age + 2

-- Define the expected ratio of the man's age to son's age in two years
def ratio := man_age_in_two_years / son_age_in_two_years

-- Theorem statement verifying the ratio
theorem man_son_age_ratio_is_two_to_one : ratio = 2 := by
  -- Note: Proof not required, so we use sorry to denote the missing proof
  sorry

end man_son_age_ratio_is_two_to_one_l115_115863


namespace angle_measure_l115_115689

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l115_115689


namespace average_of_remaining_ten_numbers_l115_115024

theorem average_of_remaining_ten_numbers
  (avg_50 : ℝ)
  (n_50 : ℝ)
  (avg_40 : ℝ)
  (n_40 : ℝ)
  (sum_50 : n_50 * avg_50 = 3800)
  (sum_40 : n_40 * avg_40 = 3200)
  (n_10 : n_50 - n_40 = 10)
  : (3800 - 3200) / 10 = 60 :=
by
  sorry

end average_of_remaining_ten_numbers_l115_115024


namespace flowers_on_porch_l115_115268

theorem flowers_on_porch (total_plants : ℕ) (flowering_percentage : ℝ) (fraction_on_porch : ℝ) (flowers_per_plant : ℕ) (h1 : total_plants = 80) (h2 : flowering_percentage = 0.40) (h3 : fraction_on_porch = 0.25) (h4 : flowers_per_plant = 5) : (total_plants * flowering_percentage * fraction_on_porch * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l115_115268


namespace no_x2_term_imp_a_eq_half_l115_115399

theorem no_x2_term_imp_a_eq_half (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x^2 - 2 * a * x + a^2) = x^3 + (1 - 2 * a) * x^2 + ((a^2 - 2 * a) * x + a^2)) →
  (∀ c : ℝ, (1 - 2 * a) = 0) →
  a = 1 / 2 :=
by
  intros h_prod h_eq
  have h_eq' : 1 - 2 * a = 0 := h_eq 0
  linarith

end no_x2_term_imp_a_eq_half_l115_115399


namespace sequence_formula_l115_115017

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n ^ 2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
by
  sorry

end sequence_formula_l115_115017


namespace intersection_A_B_l115_115541

open Set

-- Conditions given in the problem
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Statement to prove, no proof needed
theorem intersection_A_B : A ∩ B = {1, 2} := 
sorry

end intersection_A_B_l115_115541


namespace first_discount_is_20_percent_l115_115833

-- Define the problem parameters
def original_price : ℝ := 200
def final_price : ℝ := 152
def second_discount : ℝ := 0.05

-- Define the function to compute the price after two discounts
def price_after_discounts (first_discount : ℝ) : ℝ := 
  original_price * (1 - first_discount) * (1 - second_discount)

-- Define the statement that we need to prove
theorem first_discount_is_20_percent : 
  ∃ (first_discount : ℝ), price_after_discounts first_discount = final_price ∧ first_discount = 0.20 :=
by
  sorry

end first_discount_is_20_percent_l115_115833


namespace more_boys_after_initial_l115_115324

theorem more_boys_after_initial (X Y Z : ℕ) (hX : X = 22) (hY : Y = 35) : Z = Y - X :=
by
  sorry

end more_boys_after_initial_l115_115324


namespace number_of_quadruplets_l115_115056

variables (a b c : ℕ)

theorem number_of_quadruplets (h1 : 2 * a + 3 * b + 4 * c = 1200)
                             (h2 : b = 3 * c)
                             (h3 : a = 2 * b) :
  4 * c = 192 :=
by
  sorry

end number_of_quadruplets_l115_115056


namespace mixed_solution_concentration_l115_115850

-- Defining the conditions as given in the question
def weight1 : ℕ := 200
def concentration1 : ℕ := 25
def saltInFirstSolution : ℕ := (concentration1 * weight1) / 100

def weight2 : ℕ := 300
def saltInSecondSolution : ℕ := 60

def totalSalt : ℕ := saltInFirstSolution + saltInSecondSolution
def totalWeight : ℕ := weight1 + weight2

-- Statement of the proof
theorem mixed_solution_concentration :
  ((totalSalt : ℚ) / (totalWeight : ℚ)) * 100 = 22 :=
by
  sorry

end mixed_solution_concentration_l115_115850


namespace greatest_sum_on_circle_l115_115299

theorem greatest_sum_on_circle : 
  ∃ x y : ℤ, x^2 + y^2 = 169 ∧ x ≥ y ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 169 → x' ≥ y' → x + y ≥ x' + y') := 
sorry

end greatest_sum_on_circle_l115_115299


namespace expand_polynomial_l115_115176

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l115_115176


namespace cos_neg_79_pi_over_6_l115_115099

theorem cos_neg_79_pi_over_6 : 
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_neg_79_pi_over_6_l115_115099


namespace tom_final_amount_l115_115608

-- Conditions and definitions from the problem
def initial_amount : ℝ := 74
def spent_percentage : ℝ := 0.15
def earnings : ℝ := 86
def share_percentage : ℝ := 0.60

-- Lean proof statement
theorem tom_final_amount :
  (initial_amount - (spent_percentage * initial_amount)) + (share_percentage * earnings) = 114.5 :=
by
  sorry

end tom_final_amount_l115_115608


namespace compute_floor_expression_l115_115140

theorem compute_floor_expression : 
  (Int.floor (↑(2025^3) / (2023 * 2024 : ℤ) - ↑(2023^3) / (2024 * 2025 : ℤ)) = 8) := 
sorry

end compute_floor_expression_l115_115140


namespace eel_species_count_l115_115021

theorem eel_species_count (sharks eels whales total : ℕ)
    (h_sharks : sharks = 35)
    (h_whales : whales = 5)
    (h_total : total = 55)
    (h_species_sum : sharks + eels + whales = total) : eels = 15 :=
by
  -- Proof goes here
  sorry

end eel_species_count_l115_115021


namespace completing_the_square_correct_l115_115113

theorem completing_the_square_correct :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x h
  sorry

end completing_the_square_correct_l115_115113


namespace boys_meet_once_excluding_start_finish_l115_115839

theorem boys_meet_once_excluding_start_finish 
    (d : ℕ) 
    (h1 : 0 < d) 
    (boy1_speed : ℕ) (boy2_speed : ℕ) 
    (h2 : boy1_speed = 6) (h3 : boy2_speed = 10)
    (relative_speed : ℕ) (h4 : relative_speed = boy1_speed + boy2_speed) 
    (time_to_meet_A_again : ℕ) (h5 : time_to_meet_A_again = d / relative_speed) 
    (boy1_laps_per_sec boy2_laps_per_sec : ℕ) 
    (h6 : boy1_laps_per_sec = boy1_speed / d) 
    (h7 : boy2_laps_per_sec = boy2_speed / d)
    (lcm_laps : ℕ) (h8 : lcm_laps = Nat.lcm 6 10)
    (meetings_per_lap : ℕ) (h9 : meetings_per_lap = lcm_laps / d)
    (total_meetings : ℕ) (h10 : total_meetings = meetings_per_lap * time_to_meet_A_again)
  : total_meetings = 1 := by
  sorry

end boys_meet_once_excluding_start_finish_l115_115839


namespace expand_expression_l115_115197

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115197


namespace nearest_integer_to_sum_l115_115394

theorem nearest_integer_to_sum (x y : ℝ) (h1 : |x| - y = 1) (h2 : |x| * y + x^2 = 2) : Int.ceil (x + y) = 2 :=
sorry

end nearest_integer_to_sum_l115_115394


namespace triangle_segments_l115_115116

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_segments (a : ℕ) (h : a > 0) :
  ¬ triangle_inequality 1 2 3 ∧
  ¬ triangle_inequality 4 5 10 ∧
  triangle_inequality 5 10 13 ∧
  ¬ triangle_inequality (2 * a) (3 * a) (6 * a) :=
by
  -- Proof goes here
  sorry

end triangle_segments_l115_115116


namespace math_club_team_selection_l115_115816

open Nat

-- Lean statement of the problem
theorem math_club_team_selection : 
  (choose 7 3) * (choose 9 3) = 2940 :=
by 
  sorry

end math_club_team_selection_l115_115816


namespace solution_l115_115805

noncomputable def problem :=
  let A : (ℝ × ℝ × ℝ) := (a, 0, 0) in
  let B : (ℝ × ℝ × ℝ) := (0, b, 0) in
  let C : (ℝ × ℝ × ℝ) := (0, 0, c) in
  let plane_eq (x y z a b c : ℝ) : Prop := (x / a + y / b + z / c = 1) in
  let distance_eq (a b c : ℝ) : Prop :=
    (1 / (Real.sqrt ((1 / (a^2)) + 1 / (b^2) + 1 / (c^2)))) = 2 in
  let centroid (a b c : ℝ) : ℝ × ℝ × ℝ := (a / 3, b / 3, c / 3) in
  let (p, q, r) := centroid a b c in
  (plane_eq a 0 0 a b c) ∧
  (plane_eq 0 b 0 a b c) ∧
  (plane_eq 0 0 c a b c) ∧
  (distance_eq a b c) → (1 / (p^2) + 1 / (q^2) + 1 / (r^2)) = 2.25

theorem solution : problem :=
sorry

end solution_l115_115805


namespace degree_of_g_l115_115933

open Polynomial

theorem degree_of_g (f g : Polynomial ℂ) (h1 : f = -3 * X^5 + 4 * X^4 - X^2 + C 2) (h2 : degree (f + g) = 2) : degree g = 5 :=
sorry

end degree_of_g_l115_115933


namespace calculate_expression_l115_115510

theorem calculate_expression : -2 - 2 * Real.sin (Real.pi / 4) + (Real.pi - 3.14) * 0 + (-1) ^ 3 = -3 - Real.sqrt 2 := by 
sorry

end calculate_expression_l115_115510


namespace perpendicular_line_through_center_l115_115456

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * x + y^2 - 3 = 0

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the line we want to prove passes through the center of the circle and is perpendicular to the given line
def wanted_line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Main statement: Prove that the line that passes through the center of the circle and is perpendicular to the given line has the equation x - y + 1 = 0
theorem perpendicular_line_through_center (x y : ℝ) :
  (circle_eq (-1) 0) ∧ (line_eq x y) → wanted_line_eq x y :=
by
  sorry

end perpendicular_line_through_center_l115_115456


namespace sales_tax_difference_l115_115284

theorem sales_tax_difference :
  let price_before_tax := 40
  let tax_rate_8_percent := 0.08
  let tax_rate_7_percent := 0.07
  let sales_tax_8_percent := price_before_tax * tax_rate_8_percent
  let sales_tax_7_percent := price_before_tax * tax_rate_7_percent
  sales_tax_8_percent - sales_tax_7_percent = 0.4 := 
by
  sorry

end sales_tax_difference_l115_115284


namespace lowest_price_eq_195_l115_115499

def cost_per_component : ℕ := 80
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_costs : ℕ := 16500
def num_components : ℕ := 150

theorem lowest_price_eq_195 
  (cost_per_component shipping_cost_per_unit fixed_monthly_costs num_components : ℕ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 5)
  (h3 : fixed_monthly_costs = 16500)
  (h4 : num_components = 150) :
  (fixed_monthly_costs + num_components * (cost_per_component + shipping_cost_per_unit)) / num_components = 195 :=
by
  sorry

end lowest_price_eq_195_l115_115499


namespace remainder_xyz_mod7_condition_l115_115911

-- Define variables and conditions
variables (x y z : ℕ)
theorem remainder_xyz_mod7_condition (hx : x < 7) (hy : y < 7) (hz : z < 7)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 7])
  (h2 : 3 * x + 2 * y + z ≡ 2 [MOD 7])
  (h3 : 2 * x + y + 3 * z ≡ 3 [MOD 7]) :
  (x * y * z % 7) ≡ 1 [MOD 7] := sorry

end remainder_xyz_mod7_condition_l115_115911


namespace area_of_triangle_l115_115918

theorem area_of_triangle {A B C : ℝ} {a b c : ℝ}
  (h1 : b = 2) (h2 : c = 2 * Real.sqrt 2) (h3 : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - C - (1 / 2 * Real.pi / 3)) = Real.sqrt 3 + 1 :=
by
  sorry

end area_of_triangle_l115_115918


namespace principal_amount_borrowed_l115_115967

theorem principal_amount_borrowed (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (R_eq : R = 12) (T_eq : T = 3) (SI_eq : SI = 7200) :
  (SI = (P * R * T) / 100) → P = 20000 :=
by sorry

end principal_amount_borrowed_l115_115967


namespace expand_product_l115_115156

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l115_115156


namespace angle_measure_triple_complement_l115_115615

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l115_115615


namespace problem_statement_l115_115033

variable (f : ℝ → ℝ) 

def prop1 (f : ℝ → ℝ) : Prop := ∃T > 0, T ≠ 3 / 2 ∧ ∀ x, f (x + T) = f x
def prop2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 4) = f (-x + 3 / 4)
def prop3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def prop4 (f : ℝ → ℝ) : Prop := Monotone f

theorem problem_statement (h₁ : ∀ x, f (x + 3 / 2) = -f x)
                          (h₂ : ∀ x, f (x - 3 / 4) = -f (-x - 3 / 4)) : 
                          (¬prop1 f) ∧ (prop2 f) ∧ (prop3 f) ∧ (¬prop4 f) :=
by
  sorry

end problem_statement_l115_115033


namespace contrapositive_equiv_l115_115713

variable {α : Type}  -- Type of elements
variable (P : Set α) (a b : α)

theorem contrapositive_equiv (h : a ∈ P → b ∉ P) : b ∈ P → a ∉ P :=
by
  sorry

end contrapositive_equiv_l115_115713


namespace triangle_area_and_right_angle_l115_115020

-- Define the vertices
def A : Fin 2 → ℝ := ![1, 2]
def B : Fin 2 → ℝ := ![-1, -1]
def C : Fin 2 → ℝ := ![4, -3]

-- Compute vectors
def vectorCA : Fin 2 → ℝ := C - A
def vectorCB : Fin 2 → ℝ := C - B

-- Define the area function for a triangle given two vectors
def triangle_area (v w : Fin 2 → ℝ) : ℝ :=
  0.5 * abs (v 0 * w 1 - v 1 * w 0)

-- Check if vectors are orthogonal
def is_right_angled (v w : Fin 2 → ℝ) : Bool :=
  (v 0 * w 0 + v 1 * w 1) = 0

-- The main theorem
theorem triangle_area_and_right_angle :
  triangle_area vectorCA vectorCB = 19 / 2 ∧ ¬is_right_angled vectorCA vectorCB :=
by
  sorry

end triangle_area_and_right_angle_l115_115020


namespace parallel_lines_l115_115387

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, (a-1) * x + 2 * y + 10 = 0) → (∀ x y : ℝ, x + a * y + 3 = 0) → (a = -1 ∨ a = 2) :=
sorry

end parallel_lines_l115_115387


namespace james_running_increase_l115_115574

theorem james_running_increase (initial_miles_per_week : ℕ) (percent_increase : ℝ) (total_days : ℕ) (days_in_week : ℕ) :
  initial_miles_per_week = 100 →
  percent_increase = 0.2 →
  total_days = 280 →
  days_in_week = 7 →
  ∃ miles_per_week_to_add : ℝ, miles_per_week_to_add = 3 :=
by
  intros
  sorry

end james_running_increase_l115_115574


namespace find_abc_square_sum_l115_115051

theorem find_abc_square_sum (a b c : ℝ) 
  (h1 : a^2 + 3 * b = 9) 
  (h2 : b^2 + 5 * c = -8) 
  (h3 : c^2 + 7 * a = -18) : 
  a^2 + b^2 + c^2 = 20.75 := 
sorry

end find_abc_square_sum_l115_115051


namespace miles_to_add_per_week_l115_115572

theorem miles_to_add_per_week :
  ∀ (initial_miles_per_week : ℝ) 
    (percentage_increase : ℝ) 
    (total_days : ℕ) 
    (days_in_week : ℕ),
    initial_miles_per_week = 100 →
    percentage_increase = 0.2 →
    total_days = 280 →
    days_in_week = 7 →
    ((initial_miles_per_week * (1 + percentage_increase)) - initial_miles_per_week) / (total_days / days_in_week) = 3 :=
by
  intros initial_miles_per_week percentage_increase total_days days_in_week
  intros h1 h2 h3 h4
  sorry

end miles_to_add_per_week_l115_115572


namespace spherical_distance_between_points_l115_115533

noncomputable def spherical_distance (R : ℝ) (α : ℝ) : ℝ :=
  α * R

theorem spherical_distance_between_points 
  (R : ℝ) 
  (α : ℝ) 
  (hR : R > 0) 
  (hα : α = π / 6) : 
  spherical_distance R α = (π / 6) * R :=
by
  rw [hα]
  unfold spherical_distance
  ring

end spherical_distance_between_points_l115_115533


namespace complement_A_l115_115814

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := { x : ℝ | x < 2 }

theorem complement_A :
  (U \ A) = { x : ℝ | x >= 2 } :=
by
  sorry

end complement_A_l115_115814


namespace ratio_of_shaded_area_l115_115481

theorem ratio_of_shaded_area 
  (AC : ℝ) (CB : ℝ) 
  (AB : ℝ := AC + CB) 
  (radius_AC : ℝ := AC / 2) 
  (radius_CB : ℝ := CB / 2)
  (radius_AB : ℝ := AB / 2) 
  (shaded_area : ℝ := (radius_AB ^ 2 * Real.pi / 2) - (radius_AC ^ 2 * Real.pi / 2) - (radius_CB ^ 2 * Real.pi / 2))
  (CD : ℝ := Real.sqrt (AC^2 - radius_CB^2))
  (circle_area : ℝ := CD^2 * Real.pi) :
  (shaded_area / circle_area = 21 / 187) := 
by 
  sorry

end ratio_of_shaded_area_l115_115481


namespace total_games_eq_64_l115_115429

def games_attended : ℕ := 32
def games_missed : ℕ := 32
def total_games : ℕ := games_attended + games_missed

theorem total_games_eq_64 : total_games = 64 := by
  sorry

end total_games_eq_64_l115_115429


namespace minimum_sum_distances_square_l115_115061

noncomputable def minimum_sum_of_distances
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : ℝ :=
(1 + Real.sqrt 2) * d

theorem minimum_sum_distances_square
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : minimum_sum_of_distances A B d h_dist = (1 + Real.sqrt 2) * d := by
sorry

end minimum_sum_distances_square_l115_115061


namespace find_K_l115_115417

noncomputable def cylinder_paint (r h : ℝ) : ℝ := 2 * Real.pi * r * h
noncomputable def cube_surface_area (s : ℝ) : ℝ := 6 * s^2
noncomputable def cube_volume (s : ℝ) : ℝ := s^3

theorem find_K :
  (cylinder_paint 3 4 = 24 * Real.pi) →
  (∃ s, cube_surface_area s = 24 * Real.pi ∧ cube_volume s = 48 / Real.sqrt K) →
  K = 36 / Real.pi^3 :=
by
  sorry

end find_K_l115_115417


namespace expand_product_l115_115157

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l115_115157


namespace jeopardy_episode_length_l115_115795

-- Definitions based on the conditions
def num_episodes_jeopardy : ℕ := 2
def num_episodes_wheel : ℕ := 2
def wheel_twice_jeopardy (J : ℝ) : ℝ := 2 * J
def total_time_watched : ℝ := 120 -- in minutes

-- Condition stating the total time watched in terms of J
def total_watching_time_formula (J : ℝ) : ℝ :=
  num_episodes_jeopardy * J + num_episodes_wheel * (wheel_twice_jeopardy J)

theorem jeopardy_episode_length : ∃ J : ℝ, total_watching_time_formula J = total_time_watched ∧ J = 20 :=
by
  use 20
  simp [total_watching_time_formula, wheel_twice_jeopardy, num_episodes_jeopardy, num_episodes_wheel, total_time_watched]
  sorry

end jeopardy_episode_length_l115_115795


namespace rs_division_l115_115321

theorem rs_division (a b c : ℝ) 
  (h1 : a = 1 / 2 * b)
  (h2 : b = 1 / 2 * c)
  (h3 : a + b + c = 700) : 
  c = 400 :=
sorry

end rs_division_l115_115321


namespace autumn_pencils_l115_115876

theorem autumn_pencils : 
  ∀ (start misplaced broken found bought : ℕ),
  start = 20 →
  misplaced = 7 → 
  broken = 3 → 
  found = 4 → 
  bought = 2 → 
  start - misplaced - broken + found + bought = 16 :=
by 
  intros start misplaced broken found bought h_start h_misplaced h_broken h_found h_bought,
  rw [h_start, h_misplaced, h_broken, h_found, h_bought],
  linarith

end autumn_pencils_l115_115876


namespace find_f_l115_115383

theorem find_f {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2*x := 
by
  sorry

end find_f_l115_115383


namespace largest_common_number_in_arithmetic_sequences_l115_115739

theorem largest_common_number_in_arithmetic_sequences (n : ℕ) :
  (∃ a1 a2 : ℕ, a1 = 5 + 8 * n ∧ a2 = 3 + 9 * n ∧ a1 = a2 ∧ 1 ≤ a1 ∧ a1 ≤ 150) →
  (a1 = 93) :=
by
  sorry

end largest_common_number_in_arithmetic_sequences_l115_115739


namespace triangle_angle_identity_l115_115410

def triangle_angles_arithmetic_sequence (A B C : ℝ) : Prop :=
  A + C = 2 * B

def sum_of_triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = 180

def angle_B_is_60 (B : ℝ) : Prop :=
  B = 60

theorem triangle_angle_identity (A B C a b c : ℝ)
  (h1 : triangle_angles_arithmetic_sequence A B C)
  (h2 : sum_of_triangle_angles A B C)
  (h3 : angle_B_is_60 B) : 
  1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) :=
by 
  sorry

end triangle_angle_identity_l115_115410


namespace total_employees_l115_115981

theorem total_employees (x : Nat) (h1 : x < 13) : 13 + 6 * x = 85 :=
by
  sorry

end total_employees_l115_115981


namespace lcm_24_36_45_l115_115707

open Nat

theorem lcm_24_36_45 : lcm (lcm 24 36) 45 = 360 := by sorry

end lcm_24_36_45_l115_115707


namespace percentage_increase_l115_115337

theorem percentage_increase (W E : ℝ) (P : ℝ) :
  W = 200 →
  E = 204 →
  (∃ P, E = W * (1 + P / 100) * 0.85) →
  P = 20 :=
by
  intros hW hE hP
  -- Proof could be added here.
  sorry

end percentage_increase_l115_115337


namespace mario_meet_speed_l115_115512

noncomputable def Mario_average_speed (x : ℝ) : ℝ :=
  let t1 := x / 5
  let t2 := x / 3
  let t3 := x / 4
  let t4 := x / 10
  let T := t1 + t2 + t3 + t4
  let d_mario := 1.5 * x
  d_mario / T

theorem mario_meet_speed : ∀ (x : ℝ), x > 0 → Mario_average_speed x = 90 / 53 :=
by
  intros
  rw [Mario_average_speed]
  -- You can insert calculations similar to those in the provided solution
  sorry

end mario_meet_speed_l115_115512


namespace problem_A_problem_B_problem_C_problem_D_l115_115478

theorem problem_A (a b: ℝ) (h : b > 0 ∧ a > b) : ¬(1/a > 1/b) := 
by {
  sorry
}

theorem problem_B (a b: ℝ) (h : a < b ∧ b < 0): (a^2 > a*b) := 
by {
  sorry
}

theorem problem_C (a b: ℝ) (h : a > b): ¬(|a| > |b|) := 
by {
  sorry
}

theorem problem_D (a: ℝ) (h : a > 2): (a + 4/(a-2) ≥ 6) := 
by {
  sorry
}

end problem_A_problem_B_problem_C_problem_D_l115_115478


namespace simplify_expression_l115_115449

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l115_115449


namespace flowers_on_porch_l115_115266

-- Definitions based on problem conditions
def total_plants : ℕ := 80
def flowering_percentage : ℝ := 0.40
def fraction_on_porch : ℝ := 0.25
def flowers_per_plant : ℕ := 5

-- Theorem statement
theorem flowers_on_porch (h1 : total_plants = 80)
                         (h2 : flowering_percentage = 0.40)
                         (h3 : fraction_on_porch = 0.25)
                         (h4 : flowers_per_plant = 5) :
    (total_plants * seminal (flowering_percentage * fraction_on_porch) * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l115_115266


namespace product_remainder_l115_115096

theorem product_remainder (a b c d : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 4) (hd : d % 7 = 5) :
  (a * b * c * d) % 7 = 1 :=
by
  sorry

end product_remainder_l115_115096


namespace simplify_to_linear_binomial_l115_115751

theorem simplify_to_linear_binomial (k : ℝ) (x : ℝ) : 
  (-3 * k * x^2 + x - 1) + (9 * x^2 - 4 * k * x + 3 * k) = 
  (1 - 4 * k) * x + (3 * k - 1) → 
  k = 3 := by
  sorry

end simplify_to_linear_binomial_l115_115751


namespace al_bill_cal_probability_l115_115132

-- Let's define the conditions and problem setup
def al_bill_cal_prob : ℚ :=
  let total_ways := 12 * 11 * 10
  let valid_ways := 12 -- This represent the summed valid cases as calculated
  valid_ways / total_ways

theorem al_bill_cal_probability :
  al_bill_cal_prob = 1 / 110 :=
  by
  -- Placeholder for calculation and proof
  sorry

end al_bill_cal_probability_l115_115132


namespace complex_repair_cost_l115_115797

theorem complex_repair_cost
  (charge_tire : ℕ)
  (cost_part_tire : ℕ)
  (num_tires : ℕ)
  (charge_complex : ℕ)
  (num_complex : ℕ)
  (profit_retail : ℕ)
  (fixed_expenses : ℕ)
  (total_profit : ℕ)
  (profit_tire : ℕ := charge_tire - cost_part_tire)
  (total_profit_tire : ℕ := num_tires * profit_tire)
  (total_revenue_complex : ℕ := num_complex * charge_complex)
  (initial_profit : ℕ :=
    total_profit_tire + profit_retail - fixed_expenses)
  (needed_profit_complex : ℕ := total_profit - initial_profit) :
  needed_profit_complex = 100 / num_complex :=
by
  sorry

end complex_repair_cost_l115_115797


namespace intercept_x_parallel_lines_l115_115386

theorem intercept_x_parallel_lines (m : ℝ) 
    (line_l : ∀ x y : ℝ, y + m * (x + 1) = 0) 
    (parallel : ∀ x y : ℝ, y * m - (2 * m + 1) * x = 1) : 
    ∃ x : ℝ, x + 1 = -1 :=
by
  sorry

end intercept_x_parallel_lines_l115_115386


namespace number_of_oxygen_atoms_l115_115890

theorem number_of_oxygen_atoms 
  (M_weight : ℝ)
  (H_weight : ℝ)
  (Cl_weight : ℝ)
  (O_weight : ℝ)
  (MW_formula : M_weight = H_weight + Cl_weight + n * O_weight)
  (M_weight_eq : M_weight = 68)
  (H_weight_eq : H_weight = 1)
  (Cl_weight_eq : Cl_weight = 35.5)
  (O_weight_eq : O_weight = 16)
  : n = 2 := 
  by sorry

end number_of_oxygen_atoms_l115_115890


namespace child_ticket_cost_l115_115859

def cost_of_adult_ticket : ℕ := 22
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 2
def total_family_cost : ℕ := 58
def cost_of_child_ticket : ℕ := 7

theorem child_ticket_cost :
  2 * cost_of_adult_ticket + number_of_children * cost_of_child_ticket = total_family_cost :=
by
  sorry

end child_ticket_cost_l115_115859


namespace purple_marble_probability_l115_115976

theorem purple_marble_probability (P_blue P_green P_purple : ℝ) (h1 : P_blue = 0.35) (h2 : P_green = 0.45) (h3 : P_blue + P_green + P_purple = 1) :
  P_purple = 0.2 := 
by sorry

end purple_marble_probability_l115_115976


namespace periodic_minus_decimal_is_correct_l115_115010

-- Definitions based on conditions

def periodic_63_as_fraction : ℚ := 63 / 99
def decimal_63_as_fraction : ℚ := 63 / 100
def difference : ℚ := periodic_63_as_fraction - decimal_63_as_fraction

-- Lean 4 statement to prove the mathematically equivalent proof problem
theorem periodic_minus_decimal_is_correct :
  difference = 7 / 1100 :=
by
  sorry

end periodic_minus_decimal_is_correct_l115_115010


namespace simplify_expression_l115_115447

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l115_115447


namespace completing_the_square_correct_l115_115114

theorem completing_the_square_correct :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x h
  sorry

end completing_the_square_correct_l115_115114


namespace simplify_expression_l115_115442

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115442


namespace wire_length_l115_115865

theorem wire_length (r_sphere r_cylinder : ℝ) (V_sphere_eq_V_cylinder : (4/3) * π * r_sphere^3 = π * r_cylinder^2 * 144) :
  r_sphere = 12 → r_cylinder = 4 → 144 = 144 := sorry

end wire_length_l115_115865


namespace find_expression_value_l115_115400

theorem find_expression_value (x : ℝ) (h : x + 1/x = 3) : 
  x^10 - 5 * x^6 + x^2 = 8436*x - 338 := 
by {
  sorry
}

end find_expression_value_l115_115400


namespace centroid_sum_of_squares_l115_115804

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end centroid_sum_of_squares_l115_115804


namespace range_of_a_l115_115530

theorem range_of_a (A M : ℝ × ℝ) (a : ℝ) (C : ℝ × ℝ → ℝ) (hA : A = (-3, 0)) 
(hM : C M = 1) (hMA : dist M A = 2 * dist M (0, 0)) :
  a ∈ (Set.Icc (1/2 : ℝ) (3/2) ∪ Set.Icc (-3/2) (-1/2)) :=
sorry

end range_of_a_l115_115530


namespace cost_of_one_dozen_pens_l115_115828

theorem cost_of_one_dozen_pens
  (p q : ℕ)
  (h1 : 3 * p + 5 * q = 240)
  (h2 : p = 5 * q) :
  12 * p = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l115_115828


namespace black_area_fraction_after_three_changes_l115_115727

theorem black_area_fraction_after_three_changes
  (initial_black_area : ℚ)
  (change_factor : ℚ)
  (h1 : initial_black_area = 1)
  (h2 : change_factor = 2 / 3)
  : (change_factor ^ 3) * initial_black_area = 8 / 27 := 
by
  sorry

end black_area_fraction_after_three_changes_l115_115727


namespace num_carnations_l115_115958

-- Define the conditions
def num_roses : ℕ := 5
def total_flowers : ℕ := 10

-- Define the statement we want to prove
theorem num_carnations : total_flowers - num_roses = 5 :=
by {
  -- The proof itself is not required, so we use 'sorry' to indicate incomplete proof
  sorry
}

end num_carnations_l115_115958


namespace percentage_paid_to_A_l115_115840

theorem percentage_paid_to_A (A B : ℝ) (h1 : A + B = 550) (h2 : B = 220) : (A / B) * 100 = 150 := by
  -- Proof omitted
  sorry

end percentage_paid_to_A_l115_115840


namespace linear_decreasing_sequence_l115_115458

theorem linear_decreasing_sequence 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_func1 : y1 = -3 * x1 + 1)
  (h_func2 : y2 = -3 * x2 + 1)
  (h_func3 : y3 = -3 * x3 + 1)
  (hx_seq : x1 < x2 ∧ x2 < x3)
  : y3 < y2 ∧ y2 < y1 := 
sorry

end linear_decreasing_sequence_l115_115458


namespace trapezoid_base_ratio_l115_115030

-- Define the context of the problem
variables (AB CD : ℝ) (h : AB < CD)

-- Define the main theorem to be proved
theorem trapezoid_base_ratio (h : AB / CD = 1 / 2) :
  ∃ (E F G H I J : ℝ), 
    EJ - EI = FI - FH / 5 ∧ -- These points create segments that divide equally as per the conditions 
    FI - FH = GH / 5 ∧
    GH - GI = HI / 5 ∧
    HI - HJ = JI / 5 ∧
    JI - JE = EJ / 5 :=
sorry

end trapezoid_base_ratio_l115_115030


namespace gcd_f_x_l115_115764

-- Define that x is a multiple of 23478
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Define the function f(x)
noncomputable def f (x : ℕ) : ℕ := (2 * x + 3) * (7 * x + 2) * (13 * x + 7) * (x + 13)

-- Assert the proof problem
theorem gcd_f_x (x : ℕ) (h : is_multiple_of x 23478) : Nat.gcd (f x) x = 546 :=
by 
  sorry

end gcd_f_x_l115_115764


namespace Tim_younger_than_Jenny_l115_115307

def Tim_age : Nat := 5
def Rommel_age (T : Nat) : Nat := 3 * T
def Jenny_age (R : Nat) : Nat := R + 2

theorem Tim_younger_than_Jenny :
  let T := Tim_age
  let R := Rommel_age T
  let J := Jenny_age R
  J - T = 12 :=
by
  sorry

end Tim_younger_than_Jenny_l115_115307


namespace coordinates_of_A_l115_115408

/-- The initial point A and the transformations applied to it -/
def initial_point : Prod ℤ ℤ := (-3, 2)

def translate_right (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1 + units, p.2)

def translate_down (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1, p.2 - units)

/-- Proof that the point A' has coordinates (1, -1) -/
theorem coordinates_of_A' : 
  translate_down (translate_right initial_point 4) 3 = (1, -1) :=
by
  sorry

end coordinates_of_A_l115_115408


namespace vegetable_price_l115_115838

theorem vegetable_price (v : ℝ) 
  (beef_cost : ∀ (b : ℝ), b = 3 * v)
  (total_cost : 4 * (3 * v) + 6 * v = 36) : 
  v = 2 :=
by {
  -- The proof would go here.
  sorry
}

end vegetable_price_l115_115838


namespace expand_polynomial_eq_l115_115192

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l115_115192


namespace boat_downstream_distance_l115_115497

theorem boat_downstream_distance 
  (Vb Vr T D U : ℝ)
  (h1 : Vb + Vr = 21)
  (h2 : Vb - Vr = 12)
  (h3 : U = 48)
  (h4 : T = 4)
  (h5 : D = 20) :
  (Vb + Vr) * D = 420 :=
by
  sorry

end boat_downstream_distance_l115_115497


namespace beaver_hid_36_carrots_l115_115892

variable (x y : ℕ)

-- Conditions
def beaverCarrots := 4 * x
def bunnyCarrots := 6 * y

-- Given that both animals hid the same total number of carrots
def totalCarrotsEqual := beaverCarrots x = bunnyCarrots y

-- Bunny used 3 fewer burrows than the beaver
def bunnyBurrows := y = x - 3

-- The goal is to show the beaver hid 36 carrots
theorem beaver_hid_36_carrots (H1 : totalCarrotsEqual x y) (H2 : bunnyBurrows x y) : beaverCarrots x = 36 := by
  sorry

end beaver_hid_36_carrots_l115_115892


namespace marble_ratio_l115_115893

theorem marble_ratio (total_marbles red_marbles dark_blue_marbles : ℕ) (h_total : total_marbles = 63) (h_red : red_marbles = 38) (h_blue : dark_blue_marbles = 6) :
  (total_marbles - red_marbles - dark_blue_marbles) / red_marbles = 1 / 2 := by
  sorry

end marble_ratio_l115_115893


namespace find_solutions_l115_115367

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ Int.gcd (Int.gcd a b) c = 1 ∧
  (a + b + c) ∣ (a^12 + b^12 + c^12) ∧
  (a + b + c) ∣ (a^23 + b^23 + c^23) ∧
  (a + b + c) ∣ (a^11004 + b^11004 + c^11004)

theorem find_solutions :
  (is_solution 1 1 1) ∧ (is_solution 1 1 4) ∧ 
  (∀ a b c : ℕ, is_solution a b c → 
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 4)) := 
sorry

end find_solutions_l115_115367


namespace angle_is_67_l115_115644

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l115_115644


namespace angle_triple_complement_l115_115649

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l115_115649


namespace max_vertices_of_divided_triangle_l115_115340

theorem max_vertices_of_divided_triangle (n : ℕ) (h : n ≥ 1) : 
  (∀ t : ℕ, t = 1000 → exists T : ℕ, T = (n + 2)) :=
by sorry

end max_vertices_of_divided_triangle_l115_115340


namespace blue_part_length_l115_115128

variable (total_length : ℝ) (black_part white_part blue_part : ℝ)

-- Conditions
axiom h1 : black_part = 1 / 8 * total_length
axiom h2 : white_part = 1 / 2 * (total_length - black_part)
axiom h3 : total_length = 8

theorem blue_part_length : blue_part = total_length - black_part - white_part :=
by
  sorry

end blue_part_length_l115_115128


namespace expand_expression_l115_115171

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l115_115171


namespace difference_of_extreme_valid_numbers_l115_115018

theorem difference_of_extreme_valid_numbers :
  ∃ (largest smallest : ℕ),
    (largest = 222210 ∧ smallest = 100002) ∧ 
    (largest % 3 = 0 ∧ smallest % 3 = 0) ∧ 
    (largest ≥ 100000 ∧ largest < 1000000) ∧
    (smallest ≥ 100000 ∧ smallest < 1000000) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10])) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10])) ∧ 
    (∀ d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10], d ∈ [0, 1, 2]) ∧
    (∀ d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10], d ∈ [0, 1, 2]) ∧
    (largest - smallest = 122208) :=
by
  sorry

end difference_of_extreme_valid_numbers_l115_115018


namespace cricket_player_average_l115_115125

theorem cricket_player_average (A : ℝ) (h1 : 10 * A + 84 = 11 * (A + 4)) : A = 40 :=
by
  sorry

end cricket_player_average_l115_115125


namespace intersection_A_B_l115_115539

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l115_115539


namespace angle_measure_triple_complement_l115_115627

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l115_115627


namespace line_intersects_circle_l115_115818

variable (x₀ y₀ r : Real)

theorem line_intersects_circle (h : x₀^2 + y₀^2 > r^2) : 
  ∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = r^2) ∧ (x₀ * p.1 + y₀ * p.2 = r^2) := by
  sorry

end line_intersects_circle_l115_115818


namespace well_rate_correct_l115_115025

noncomputable def well_rate (depth : ℝ) (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  let r := diameter / 2
  let volume := Real.pi * r^2 * depth
  total_cost / volume

theorem well_rate_correct :
  well_rate 14 3 1583.3626974092558 = 15.993 :=
by
  sorry

end well_rate_correct_l115_115025


namespace range_of_x_in_function_l115_115925

theorem range_of_x_in_function (x : ℝ) : (y = 1/(x + 3) → x ≠ -3) :=
sorry

end range_of_x_in_function_l115_115925


namespace original_cost_l115_115718

theorem original_cost (C : ℝ) (h : 670 = C + 0.35 * C) : C = 496.30 :=
by
  -- The proof is omitted
  sorry

end original_cost_l115_115718


namespace gcd_8164_2937_l115_115703

/-- Define the two integers a and b -/
def a : ℕ := 8164
def b : ℕ := 2937

/-- Prove that the greatest common divisor of a and b is 1 -/
theorem gcd_8164_2937 : Nat.gcd a b = 1 :=
  by
  sorry

end gcd_8164_2937_l115_115703


namespace angle_triple_complement_l115_115653

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l115_115653


namespace retailer_profit_percentage_l115_115874

theorem retailer_profit_percentage 
  (CP MP SP : ℝ)
  (hCP : CP = 100)
  (hMP : MP = CP + 0.65 * CP)
  (hSP : SP = MP - 0.25 * MP)
  : ((SP - CP) / CP) * 100 = 23.75 := 
sorry

end retailer_profit_percentage_l115_115874


namespace remainder_1493824_div_4_l115_115316

theorem remainder_1493824_div_4 : 1493824 % 4 = 0 :=
by
  sorry

end remainder_1493824_div_4_l115_115316


namespace Tim_younger_than_Jenny_l115_115306

def Tim_age : Nat := 5
def Rommel_age (T : Nat) : Nat := 3 * T
def Jenny_age (R : Nat) : Nat := R + 2

theorem Tim_younger_than_Jenny :
  let T := Tim_age
  let R := Rommel_age T
  let J := Jenny_age R
  J - T = 12 :=
by
  sorry

end Tim_younger_than_Jenny_l115_115306


namespace area_of_trapezoid_EFGH_l115_115108

noncomputable def length (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def height_FG : ℝ :=
  6 - 2

noncomputable def area_trapezoid (E F G H : ℝ × ℝ) : ℝ :=
  let base1 := length E F
  let base2 := length G H
  let height := height_FG
  1/2 * (base1 + base2) * height

theorem area_of_trapezoid_EFGH :
  area_trapezoid (0, 0) (2, -3) (6, 0) (6, 4) = 2 * (Real.sqrt 13 + 4) :=
by
  sorry

end area_of_trapezoid_EFGH_l115_115108


namespace find_floors_l115_115133

theorem find_floors (a b : ℕ) 
  (h1 : 3 * a + 4 * b = 25)
  (h2 : 2 * a + 3 * b = 18) : 
  a = 3 ∧ b = 4 := 
sorry

end find_floors_l115_115133


namespace dot_product_of_a_and_c_is_4_l115_115778

def vector := (ℝ × ℝ)

def a : vector := (1, -2)
def b : vector := (-3, 2)

def three_a : vector := (3 * 1, 3 * -2)
def two_b_minus_a : vector := (2 * -3 - 1, 2 * 2 - -2)

def c : vector := (-(-three_a.fst + two_b_minus_a.fst), -(-three_a.snd + two_b_minus_a.snd))

def dot_product (u v : vector) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem dot_product_of_a_and_c_is_4 : dot_product a c = 4 := 
by
  sorry

end dot_product_of_a_and_c_is_4_l115_115778


namespace sticks_at_20_l115_115014

-- Define the sequence of sticks used at each stage
def sticks (n : ℕ) : ℕ :=
  if n = 1 then 5
  else if n ≤ 10 then 5 + 3 * (n - 1)
  else 32 + 4 * (n - 11)

-- Prove that the number of sticks at the 20th stage is 68
theorem sticks_at_20 : sticks 20 = 68 := by
  sorry

end sticks_at_20_l115_115014


namespace contrapositive_of_a_gt_1_then_a_sq_gt_1_l115_115827

theorem contrapositive_of_a_gt_1_then_a_sq_gt_1 : 
  (∀ a : ℝ, a > 1 → a^2 > 1) → (∀ a : ℝ, a^2 ≤ 1 → a ≤ 1) :=
by 
  sorry

end contrapositive_of_a_gt_1_then_a_sq_gt_1_l115_115827


namespace find_abc_sum_l115_115065

-- Definitions and statements directly taken from conditions
def Q1 (x y : ℝ) : Prop := y = x^2 + 51/50
def Q2 (x y : ℝ) : Prop := x = y^2 + 23/2
def common_tangent_rational_slope (a b c : ℤ) : Prop :=
  ∃ (x y : ℝ), (a * x + b * y = c) ∧ (Q1 x y ∨ Q2 x y)

theorem find_abc_sum :
  ∃ (a b c : ℕ), 
    gcd (a) (gcd (b) (c)) = 1 ∧
    common_tangent_rational_slope (a) (b) (c) ∧
    a + b + c = 9 :=
  by sorry

end find_abc_sum_l115_115065


namespace min_value_eq_18sqrt3_l115_115225

noncomputable def min_value (x y : ℝ) (h : x + y = 5) : ℝ := 3^x + 3^y

theorem min_value_eq_18sqrt3 {x y : ℝ} (h : x + y = 5) : min_value x y h ≥ 18 * Real.sqrt 3 := 
sorry

end min_value_eq_18sqrt3_l115_115225


namespace arrangements_of_masters_and_apprentices_l115_115960

theorem arrangements_of_masters_and_apprentices : 
  ∃ n : ℕ, n = 48 ∧ 
     let pairs := 3 
     let ways_to_arrange_pairs := pairs.factorial 
     let ways_to_arrange_within_pairs := 2 ^ pairs 
     ways_to_arrange_pairs * ways_to_arrange_within_pairs = n := 
sorry

end arrangements_of_masters_and_apprentices_l115_115960


namespace angle_triple_complement_l115_115636

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l115_115636


namespace average_age_is_27_l115_115595

variables (a b c : ℕ)

def average_age_of_a_and_c (a c : ℕ) := (a + c) / 2

def age_of_b := 23

def average_age_of_a_b_and_c (a b c : ℕ) := (a + b + c) / 3

theorem average_age_is_27 (h1 : average_age_of_a_and_c a c = 29) (h2 : b = age_of_b) :
  average_age_of_a_b_and_c a b c = 27 := by
  sorry

end average_age_is_27_l115_115595


namespace simplify_expression_l115_115437

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115437


namespace fraction_product_equals_12_l115_115965

theorem fraction_product_equals_12 :
  (1 / 3) * (9 / 2) * (1 / 27) * (54 / 1) * (1 / 81) * (162 / 1) * (1 / 243) * (486 / 1) = 12 := 
by
  sorry

end fraction_product_equals_12_l115_115965


namespace evaluate_g_ggg_neg1_l115_115069

def g (y : ℤ) : ℤ := y^3 - 3*y + 1

theorem evaluate_g_ggg_neg1 : g (g (g (-1))) = 6803 := 
by
  sorry

end evaluate_g_ggg_neg1_l115_115069


namespace function_equivalence_l115_115204

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 2020) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (-y) = -g y) ∧ (∀ x : ℝ, f x = g (1 - 2 * x^2) + 1010) :=
sorry

end function_equivalence_l115_115204


namespace right_angled_triangle_exists_l115_115134

def is_triangle (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_right_angled_triangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem right_angled_triangle_exists :
  is_triangle 3 4 5 ∧ is_right_angled_triangle 3 4 5 :=
by
  sorry

end right_angled_triangle_exists_l115_115134


namespace magnitude_of_b_l115_115777

variables (a b : EuclideanSpace ℝ (Fin 3)) (θ : ℝ)

-- Defining the conditions
def vector_a_magnitude : Prop := ‖a‖ = 1
def vector_angle_condition : Prop := θ = Real.pi / 3
def linear_combination_magnitude : Prop := ‖2 • a - b‖ = 2 * Real.sqrt 3
def b_magnitude : Prop := ‖b‖ = 4

-- The statement we want to prove
theorem magnitude_of_b (h1 : vector_a_magnitude a) (h2 : vector_angle_condition θ) (h3 : linear_combination_magnitude a b) : b_magnitude b :=
sorry

end magnitude_of_b_l115_115777


namespace centroid_plane_distance_l115_115802

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end centroid_plane_distance_l115_115802


namespace harry_morning_routine_time_l115_115359

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end harry_morning_routine_time_l115_115359


namespace chef_leftover_potatoes_l115_115327

-- Defining the conditions as variables
def fries_per_potato := 25
def total_potatoes := 15
def fries_needed := 200

-- Calculating the number of potatoes needed.
def potatoes_needed : ℕ :=
  fries_needed / fries_per_potato

-- Calculating the leftover potatoes.
def leftovers : ℕ :=
  total_potatoes - potatoes_needed

-- The theorem statement
theorem chef_leftover_potatoes :
  leftovers = 7 :=
by
  -- the actual proof is omitted.
  sorry

end chef_leftover_potatoes_l115_115327


namespace expand_product_l115_115162

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l115_115162


namespace cos_C_of_triangle_l115_115917

theorem cos_C_of_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hc : c = 4)
  (h_sine_relation : 3 * Real.sin A = 2 * Real.sin B)
  (h_cosine_law : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  Real.cos C = -1/4 :=
by
  sorry

end cos_C_of_triangle_l115_115917


namespace teresa_class_size_l115_115289

theorem teresa_class_size :
  ∃ (a : ℤ), 50 < a ∧ a < 100 ∧ 
  (a % 3 = 2) ∧ 
  (a % 4 = 2) ∧ 
  (a % 5 = 2) ∧ 
  a = 62 := 
by {
  sorry
}

end teresa_class_size_l115_115289


namespace least_three_digit_multiple_of_3_4_7_l115_115474

theorem least_three_digit_multiple_of_3_4_7 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 → m % 3 = 0 ∧ m % 4 = 0 ∧ m % 7 = 0 → n ≤ m :=
  sorry

end least_three_digit_multiple_of_3_4_7_l115_115474


namespace simplify_expression_l115_115448

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l115_115448


namespace value_of_x_l115_115786

theorem value_of_x
  (x : ℝ)
  (h1 : x = 0)
  (h2 : x^2 - 1 ≠ 0) :
  (x = 0) ↔ (x ^ 2 - 1 ≠ 0) :=
by
  sorry

end value_of_x_l115_115786


namespace line_eq_of_midpoint_and_hyperbola_l115_115038

theorem line_eq_of_midpoint_and_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : 9 * (8 : ℝ)^2 - 16 * (3 : ℝ)^2 = 144)
    (h2 : x1 + x2 = 16) (h3 : y1 + y2 = 6) (h4 : 9 * x1^2 - 16 * y1^2 = 144) (h5 : 9 * x2^2 - 16 * y2^2 = 144) :
    3 * (8 : ℝ) - 2 * (3 : ℝ) - 18 = 0 :=
by
  -- The proof steps would go here
  sorry

end line_eq_of_midpoint_and_hyperbola_l115_115038


namespace diameter_increase_l115_115596

theorem diameter_increase (h : 0.628 = π * d) : d = 0.2 := 
sorry

end diameter_increase_l115_115596


namespace leftover_potatoes_l115_115329

theorem leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ)
    (h1 : fries_per_potato = 25) (h2 : total_potatoes = 15) (h3 : required_fries = 200) :
    (total_potatoes - required_fries / fries_per_potato) = 7 :=
sorry

end leftover_potatoes_l115_115329


namespace original_selling_price_is_990_l115_115878

theorem original_selling_price_is_990 
( P : ℝ ) -- original purchase price
( SP_1 : ℝ := 1.10 * P ) -- original selling price
( P_new : ℝ := 0.90 * P ) -- new purchase price
( SP_2 : ℝ := 1.17 * P ) -- new selling price
( h : SP_2 - SP_1 = 63 ) : SP_1 = 990 :=
by {
  -- This is just the statement, proof is not provided
  sorry
}

end original_selling_price_is_990_l115_115878


namespace angle_triple_complement_l115_115633

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l115_115633


namespace expressway_lengths_l115_115999

theorem expressway_lengths (x y : ℕ) (h1 : x + y = 519) (h2 : x = 2 * y - 45) : x = 331 ∧ y = 188 :=
by
  -- Proof omitted
  sorry

end expressway_lengths_l115_115999


namespace recreation_spent_percent_l115_115418

variable (W : ℝ) -- Assume W is the wages last week

-- Conditions
def last_week_spent_on_recreation (W : ℝ) : ℝ := 0.25 * W
def this_week_wages (W : ℝ) : ℝ := 0.70 * W
def this_week_spent_on_recreation (W : ℝ) : ℝ := 0.50 * (this_week_wages W)

-- Proof statement
theorem recreation_spent_percent (W : ℝ) :
  (this_week_spent_on_recreation W / last_week_spent_on_recreation W) * 100 = 140 := by
  sorry

end recreation_spent_percent_l115_115418


namespace values_of_t_l115_115887

theorem values_of_t (x y z t : ℝ) 
  (h1 : 3 * x^2 + 3 * x * z + z^2 = 1)
  (h2 : 3 * y^2 + 3 * y * z + z^2 = 4)
  (h3 : x^2 - x * y + y^2 = t) : 
  t ≤ 10 :=
sorry

end values_of_t_l115_115887


namespace monotonicity_and_extremes_tangent_with_slope_neg3_tangent_passing_through_origin_l115_115384

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 1

theorem monotonicity_and_extremes :
  (∀ x, x < 0 ∨ x > 2 → f' x > 0) ∧ (∀ x, 0 < x ∧ x < 2 → f' x < 0) ∧
  (f(0) = -1) ∧ (f(2) = -5) :=
sorry

theorem tangent_with_slope_neg3 :
  ∃ x, (f' x = -3) ∧ (3 * x + f x = 0) :=
sorry

theorem tangent_passing_through_origin :
  ∃ x₀, (x₀^3 - 3*x₀^2 + 1 = 0) ∧ 
  ((3*x₀ + 1 = 0 ∧ 3 * x₀ + f x₀ = 0) ∨ (15*x₀ + 4 = 0 ∧ 15 * x₀ + 4 * f x₀ = 0)) :=
sorry

end monotonicity_and_extremes_tangent_with_slope_neg3_tangent_passing_through_origin_l115_115384


namespace find_f_3_l115_115883

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ :=
- x^2 + b * x + c

theorem find_f_3 (b c : ℝ) (h1 : quadratic_function b c 2 + quadratic_function b c 4 = 12138)
                       (h2 : 3*b + c = 6079) :
  quadratic_function b c 3 = 6070 := 
by
  sorry

end find_f_3_l115_115883


namespace arc_length_of_sector_l115_115561

theorem arc_length_of_sector (θ r : ℝ) (h1 : θ = 120) (h2 : r = 2) : 
  (θ / 360) * (2 * Real.pi * r) = (4 * Real.pi) / 3 :=
by
  sorry

end arc_length_of_sector_l115_115561


namespace i_pow_2006_l115_115223

-- Definitions based on given conditions
def i : ℂ := Complex.I

-- Cyclic properties of i (imaginary unit)
axiom i_pow_1 : i^1 = i
axiom i_pow_2 : i^2 = -1
axiom i_pow_3 : i^3 = -i
axiom i_pow_4 : i^4 = 1
axiom i_pow_5 : i^5 = i

-- The proof statement
theorem i_pow_2006 : (i^2006 = -1) :=
by
  sorry

end i_pow_2006_l115_115223


namespace longer_piece_length_is_20_l115_115084

-- Define the rope length
def ropeLength : ℕ := 35

-- Define the ratio of the two pieces
def ratioA : ℕ := 3
def ratioB : ℕ := 4
def totalRatio : ℕ := ratioA + ratioB

-- Define the length of each part
def partLength : ℕ := ropeLength / totalRatio

-- Define the length of the longer piece
def longerPieceLength : ℕ := ratioB * partLength

-- Theorem to prove that the length of the longer piece is 20 inches
theorem longer_piece_length_is_20 : longerPieceLength = 20 := by 
  sorry

end longer_piece_length_is_20_l115_115084


namespace true_propositions_count_l115_115602

theorem true_propositions_count (a : ℝ) :
  ((a > -3 → a > -6) ∧ (a > -6 → ¬(a ≤ -3)) ∧ (a ≤ -3 → ¬(a > -6)) ∧ (a ≤ -6 → a ≤ -3)) → 
  2 = 2 := 
by
  sorry

end true_propositions_count_l115_115602


namespace square_perimeter_is_44_8_l115_115001

noncomputable def perimeter_of_congruent_rectangles_division (s : ℝ) (P : ℝ) : ℝ :=
  let rectangle_perimeter := 2 * (s + s / 4)
  if rectangle_perimeter = P then 4 * s else 0

theorem square_perimeter_is_44_8 :
  ∀ (s : ℝ) (P : ℝ), P = 28 → 4 * s = 44.8 → perimeter_of_congruent_rectangles_division s P = 44.8 :=
by intros s P h1 h2
   sorry

end square_perimeter_is_44_8_l115_115001


namespace tim_youth_comparison_l115_115305

theorem tim_youth_comparison :
  let Tim_age : ℕ := 5
  let Rommel_age : ℕ := 3 * Tim_age
  let Jenny_age : ℕ := Rommel_age + 2
  Jenny_age - Tim_age = 12 := 
by 
  sorry

end tim_youth_comparison_l115_115305


namespace quadratic_inequality_solution_l115_115518

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - 2 * k + 4 < 0) ↔ (-6 < k ∧ k < 2) :=
by
  sorry

end quadratic_inequality_solution_l115_115518


namespace union_of_A_and_B_l115_115043

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4} := by
  sorry

end union_of_A_and_B_l115_115043


namespace part1_part2_l115_115790

def pointA : (ℝ × ℝ) := (1, 2)
def pointB : (ℝ × ℝ) := (-2, 3)
def pointC : (ℝ × ℝ) := (8, -5)

-- Definitions of the vectors
def OA : (ℝ × ℝ) := pointA
def OB : (ℝ × ℝ) := pointB
def OC : (ℝ × ℝ) := pointC
def AB : (ℝ × ℝ) := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Part 1: Proving the values of x and y
theorem part1 : ∃ (x y : ℝ), OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2) ∧ x = 2 ∧ y = -3 :=
by
  sorry

-- Part 2: Proving the value of m when vectors are parallel
theorem part2 : ∃ (m : ℝ), ∃ k : ℝ, AB = (k * (m + 8), k * (2 * m - 5)) ∧ m = 1 :=
by
  sorry

end part1_part2_l115_115790


namespace angle_triple_complement_l115_115669

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l115_115669


namespace farmer_apples_after_giving_away_l115_115457

def initial_apples : ℕ := 127
def given_away_apples : ℕ := 88
def remaining_apples : ℕ := 127 - 88

theorem farmer_apples_after_giving_away : remaining_apples = 39 := by
  sorry

end farmer_apples_after_giving_away_l115_115457


namespace magazines_sold_l115_115798

theorem magazines_sold (total_sold : Float) (newspapers_sold : Float) (magazines_sold : Float)
  (h1 : total_sold = 425.0)
  (h2 : newspapers_sold = 275.0) :
  magazines_sold = total_sold - newspapers_sold :=
by
  sorry

#check magazines_sold

end magazines_sold_l115_115798


namespace border_pieces_is_75_l115_115104

-- Definitions based on conditions
def total_pieces : Nat := 500
def trevor_pieces : Nat := 105
def joe_pieces : Nat := 3 * trevor_pieces
def missing_pieces : Nat := 5

-- Number of border pieces
def border_pieces : Nat := total_pieces - missing_pieces - (trevor_pieces + joe_pieces)

-- Theorem statement
theorem border_pieces_is_75 : border_pieces = 75 :=
by
  -- Proof goes here
  sorry

end border_pieces_is_75_l115_115104


namespace trapezoid_area_l115_115087

theorem trapezoid_area (a b H : ℝ) (h_lat1 : a = 10) (h_lat2 : b = 8) (h_height : H = b) : 
∃ S : ℝ, S = 104 :=
by sorry

end trapezoid_area_l115_115087


namespace op_identity_l115_115348

-- Define the operation ⊕ as given by the table
def op (x y : ℕ) : ℕ :=
  match (x, y) with
  | (1, 1) => 4
  | (1, 2) => 1
  | (1, 3) => 2
  | (1, 4) => 3
  | (2, 1) => 1
  | (2, 2) => 3
  | (2, 3) => 4
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 4
  | (3, 3) => 1
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 3
  | (4, 4) => 4
  | _ => 0  -- default case for completeness

-- State the theorem
theorem op_identity : op (op 4 1) (op 2 3) = 3 := by
  sorry

end op_identity_l115_115348


namespace diamonds_in_G15_l115_115880

theorem diamonds_in_G15 (G : ℕ → ℕ) 
  (h₁ : G 1 = 3)
  (h₂ : ∀ n, n ≥ 2 → G (n + 1) = 3 * (2 * (n - 1) + 3) - 3 ) :
  G 15 = 90 := sorry

end diamonds_in_G15_l115_115880


namespace find_p_fifth_plus_3_l115_115070

theorem find_p_fifth_plus_3 (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^4 + 3)) :
  p^5 + 3 = 35 :=
sorry

end find_p_fifth_plus_3_l115_115070


namespace harry_morning_routine_time_l115_115361

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end harry_morning_routine_time_l115_115361


namespace solve_equation_l115_115491

noncomputable def equation_to_solve (x : ℝ) : ℝ :=
  1 / (4^(3*x) - 13 * 4^(2*x) + 51 * 4^x - 60) + 1 / (4^(2*x) - 7 * 4^x + 12)

theorem solve_equation :
  (equation_to_solve (1/2) = 0) ∧ (equation_to_solve (Real.log 6 / Real.log 4) = 0) :=
by {
  sorry
}

end solve_equation_l115_115491


namespace square_perimeter_l115_115950

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by
  sorry

end square_perimeter_l115_115950


namespace polynomial_sum_l115_115423

noncomputable def p (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
noncomputable def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
noncomputable def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = 12 * x - 12 := by
  sorry

end polynomial_sum_l115_115423


namespace angle_measure_triple_complement_l115_115626

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l115_115626


namespace part1_part2_l115_115544

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x)
  else (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 2)

theorem part1 : f (Real.log 3 / Real.log 2 - Real.log 2 / Real.log 2) = 2 / 3 := by
  sorry

theorem part2 : ∃ x : ℝ, f x = -1 / 4 := by
  sorry

end part1_part2_l115_115544


namespace greatest_monthly_drop_is_march_l115_115301

-- Define the price changes for each month
def price_change_january : ℝ := -0.75
def price_change_february : ℝ := 1.50
def price_change_march : ℝ := -3.00
def price_change_april : ℝ := 2.50
def price_change_may : ℝ := -1.00
def price_change_june : ℝ := 0.50
def price_change_july : ℝ := -2.50

-- Prove that the month with the greatest drop in price is March
theorem greatest_monthly_drop_is_march :
  (price_change_march = -3.00) →
  (∀ m, m ≠ price_change_march → m ≥ price_change_march) :=
by
  intros h1 h2
  sorry

end greatest_monthly_drop_is_march_l115_115301


namespace fill_time_60_gallons_ten_faucets_l115_115527

-- Define the problem parameters
def rate_of_five_faucets : ℚ := 150 / 8 -- in gallons per minute

def rate_of_one_faucet : ℚ := rate_of_five_faucets / 5

def rate_of_ten_faucets : ℚ := rate_of_one_faucet * 10

def time_to_fill_60_gallons_minutes : ℚ := 60 / rate_of_ten_faucets

def time_to_fill_60_gallons_seconds : ℚ := time_to_fill_60_gallons_minutes * 60

-- The main theorem to prove
theorem fill_time_60_gallons_ten_faucets : time_to_fill_60_gallons_seconds = 96 := by
  sorry

end fill_time_60_gallons_ten_faucets_l115_115527


namespace expand_expression_l115_115185

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115185


namespace globe_division_l115_115985

theorem globe_division (parallels meridians : ℕ)
  (h_parallels : parallels = 17)
  (h_meridians : meridians = 24) :
  let slices_per_sector := parallels + 1
  let sectors := meridians
  let total_parts := slices_per_sector * sectors
  total_parts = 432 := by
  sorry

end globe_division_l115_115985


namespace a4_is_5_l115_115760

-- Definitions based on the given conditions in the problem
def sum_arith_seq (n a1 d : ℤ) : ℤ := n * a1 + (n * (n-1)) / 2 * d

def S6 : ℤ := 24
def S9 : ℤ := 63

-- The proof problem: we need to prove that a4 = 5 given the conditions
theorem a4_is_5 (a1 d : ℤ) (h_S6 : sum_arith_seq 6 a1 d = S6) (h_S9 : sum_arith_seq 9 a1 d = S9) : 
  a1 + 3 * d = 5 :=
sorry

end a4_is_5_l115_115760


namespace sum_of_roots_eq_4140_l115_115750

open Complex

noncomputable def sum_of_roots : ℝ :=
  let θ0 := 270 / 5;
  let θ1 := (270 + 360) / 5;
  let θ2 := (270 + 2 * 360) / 5;
  let θ3 := (270 + 3 * 360) / 5;
  let θ4 := (270 + 4 * 360) / 5;
  θ0 + θ1 + θ2 + θ3 + θ4

theorem sum_of_roots_eq_4140 : sum_of_roots = 4140 := by
  sorry

end sum_of_roots_eq_4140_l115_115750


namespace factorial_addition_l115_115012

theorem factorial_addition : 
  (Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10)) + 
  (Nat.factorial 11 / (Nat.factorial 6 * Nat.factorial 5)) = 1190 := 
by 
  sorry

end factorial_addition_l115_115012


namespace find_a4_and_s5_l115_115532

def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℚ) (q : ℚ)

axiom condition_1 : a 1 + a 3 = 10
axiom condition_2 : a 4 + a 6 = 1 / 4

theorem find_a4_and_s5 (h_geom : geometric_sequence a q) :
  a 4 = 1 ∧ (a 1 * (1 - q^5) / (1 - q)) = 31 / 2 :=
by
  sorry

end find_a4_and_s5_l115_115532


namespace angle_measure_triple_complement_l115_115623

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l115_115623


namespace lines_intersect_at_single_point_l115_115044

def line1 (a b x y: ℝ) := a * x + 2 * b * y + 3 * (a + b + 1) = 0
def line2 (a b x y: ℝ) := b * x + 2 * (a + b + 1) * y + 3 * a = 0
def line3 (a b x y: ℝ) := (a + b + 1) * x + 2 * a * y + 3 * b = 0

theorem lines_intersect_at_single_point (a b : ℝ) :
  (∃ x y : ℝ, line1 a b x y ∧ line2 a b x y ∧ line3 a b x y) ↔ a + b = -1/2 :=
by
  sorry

end lines_intersect_at_single_point_l115_115044


namespace fractions_equiv_l115_115494

theorem fractions_equiv:
  (8 : ℝ) / (7 * 67) = (0.8 : ℝ) / (0.7 * 67) :=
by
  sorry

end fractions_equiv_l115_115494


namespace hyperbola_condition_sufficiency_l115_115454

theorem hyperbola_condition_sufficiency (k : ℝ) :
  (k > 3) → (∃ x y : ℝ, (x^2)/(3-k) + (y^2)/(k-1) = 1) :=
by
  sorry

end hyperbola_condition_sufficiency_l115_115454


namespace equal_areas_of_parts_l115_115130

theorem equal_areas_of_parts :
  ∀ (S1 S2 S3 S4 : ℝ), 
    S1 = S2 → S2 = S3 → 
    (S1 + S2 = S3 + S4) → 
    (S2 + S3 = S1 + S4) → 
    S1 = S2 ∧ S2 = S3 ∧ S3 = S4 :=
by
  intros S1 S2 S3 S4 h1 h2 h3 h4
  sorry

end equal_areas_of_parts_l115_115130


namespace probability_of_selecting_product_at_least_4_l115_115077

theorem probability_of_selecting_product_at_least_4 : 
  let products := {1, 2, 3, 4, 5} in
  let favorable_products := {x ∈ products | x ≥ 4} in
  let probability := (favorable_products.card : ℝ) / (products.card : ℝ) in
  probability = 2 / 5 :=
by
  let products := {1, 2, 3, 4, 5}
  let favorable_products := {x ∈ products | x ≥ 4}
  have cardinality_products : products.card = 5 := sorry
  have cardinality_favorable_products : favorable_products.card = 2 := sorry
  have h : (2 : ℝ) / (5 : ℝ) = 2 / 5 := by norm_num
  have probability := (favorable_products.card : ℝ) / (products.card : ℝ)
  rw [cardinality_products, cardinality_favorable_products] at probability
  exact (Eq.trans probability h).symm

end probability_of_selecting_product_at_least_4_l115_115077


namespace original_average_l115_115062

theorem original_average (A : ℝ)
  (h1 : (Σ i in (finset.range 15), A) / 15 = A)
  (h2 : (Σ i in (finset.range 15), (A + 10)) / 15 = 50) :
  A = 40 :=
sorry

end original_average_l115_115062


namespace difference_is_cube_sum_1996_impossible_l115_115136

theorem difference_is_cube (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  M - m = (n - 1)^3 := 
by {
  sorry
}

theorem sum_1996_impossible (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  ¬(1996 ∈ {x | m ≤ x ∧ x ≤ M}) := 
by {
  sorry
}

end difference_is_cube_sum_1996_impossible_l115_115136


namespace hcf_of_48_and_64_is_16_l115_115954

theorem hcf_of_48_and_64_is_16
  (lcm_value : Nat)
  (hcf_value : Nat)
  (a : Nat)
  (b : Nat)
  (h_lcm : lcm_value = Nat.lcm a b)
  (hcf_def : hcf_value = Nat.gcd a b)
  (h_lcm_value : lcm_value = 192)
  (h_a : a = 48)
  (h_b : b = 64)
  : hcf_value = 16 := by
  sorry

end hcf_of_48_and_64_is_16_l115_115954


namespace units_digit_of_product_l115_115110

-- Definitions for units digit patterns for powers of 5 and 7
def units_digit (n : ℕ) : ℕ := n % 10

def power5_units_digit := 5
def power7_units_cycle := [7, 9, 3, 1]

-- Statement of the problem
theorem units_digit_of_product :
  units_digit ((5 ^ 3) * (7 ^ 52)) = 5 :=
by
  sorry

end units_digit_of_product_l115_115110


namespace eqn_distinct_real_roots_l115_115545

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then x^2 + 2 else 4 * x * Real.cos x + 1

theorem eqn_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, f x = m * x + 1) → 
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-2 * Real.pi) Real.pi ∧ x₂ ∈ Set.Icc (-2 * Real.pi) Real.pi :=
  sorry

end eqn_distinct_real_roots_l115_115545


namespace sum_of_extreme_values_l115_115264

theorem sum_of_extreme_values (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (5 - Real.sqrt 34) / 3
  let M := (5 + Real.sqrt 34) / 3
  m + M = 10 / 3 :=
by
  sorry

end sum_of_extreme_values_l115_115264


namespace speed_of_second_part_l115_115334

theorem speed_of_second_part
  (total_distance : ℝ)
  (distance_part1 : ℝ)
  (speed_part1 : ℝ)
  (average_speed : ℝ)
  (speed_part2 : ℝ) :
  total_distance = 70 →
  distance_part1 = 35 →
  speed_part1 = 48 →
  average_speed = 32 →
  speed_part2 = 24 :=
by
  sorry

end speed_of_second_part_l115_115334


namespace angle_triple_complement_l115_115696

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l115_115696


namespace count_elements_in_A_l115_115278

variables (a b : ℕ)

def condition1 : Prop := a = 3 * b / 2
def condition2 : Prop := a + b - 1200 = 4500

theorem count_elements_in_A (h1 : condition1 a b) (h2 : condition2 a b) : a = 3420 :=
by sorry

end count_elements_in_A_l115_115278


namespace angle_measure_triple_complement_l115_115614

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l115_115614


namespace actual_time_l115_115300

def digit_in_range (a b : ℕ) : Prop := 
  (a = b + 1 ∨ a = b - 1)

def time_malfunctioned (h m : ℕ) : Prop :=
  digit_in_range 0 (h / 10) ∧ -- tens of hour digit (0 -> 1 or 9)
  digit_in_range 0 (h % 10) ∧ -- units of hour digit (0 -> 1 or 9)
  digit_in_range 5 (m / 10) ∧ -- tens of minute digit (5 -> 4 or 6)
  digit_in_range 9 (m % 10)   -- units of minute digit (9 -> 8 or 0)

theorem actual_time : ∃ h m : ℕ, time_malfunctioned h m ∧ h = 11 ∧ m = 48 :=
by
  sorry

end actual_time_l115_115300


namespace area_of_EPGQ_l115_115080

noncomputable def area_of_region (length_rect width_rect half_length_rect : ℝ) : ℝ :=
  half_length_rect * width_rect

theorem area_of_EPGQ :
  let length_rect := 10.0
  let width_rect := 6.0
  let P_half_length := length_rect / 2
  let Q_half_length := length_rect / 2
  (area_of_region length_rect width_rect P_half_length) = 30.0 :=
by
  sorry

end area_of_EPGQ_l115_115080


namespace Jen_visits_either_but_not_both_l115_115469

-- Define the events and their associated probabilities
def P_Chile : ℝ := 0.30
def P_Madagascar : ℝ := 0.50

-- Define the probability of visiting both assuming independence
def P_both : ℝ := P_Chile * P_Madagascar

-- Define the probability of visiting either but not both
def P_either_but_not_both : ℝ := P_Chile + P_Madagascar - 2 * P_both

-- The problem statement
theorem Jen_visits_either_but_not_both : P_either_but_not_both = 0.65 := by
  /- The proof goes here -/
  sorry

end Jen_visits_either_but_not_both_l115_115469


namespace total_money_is_305_l115_115989

-- Define the worth of each gold coin, silver coin, and the quantity of each type of coin and cash.
def worth_of_gold_coin := 50
def worth_of_silver_coin := 25
def number_of_gold_coins := 3
def number_of_silver_coins := 5
def cash_in_dollars := 30

-- Define the total money calculation based on given conditions.
def total_gold_value := number_of_gold_coins * worth_of_gold_coin
def total_silver_value := number_of_silver_coins * worth_of_silver_coin
def total_value := total_gold_value + total_silver_value + cash_in_dollars

-- The proof statement asserting the total value.
theorem total_money_is_305 : total_value = 305 := by
  -- Proof omitted for brevity.
  sorry

end total_money_is_305_l115_115989


namespace ladder_distance_from_wall_l115_115319

theorem ladder_distance_from_wall (θ : ℝ) (L : ℝ) (d : ℝ) 
  (h_angle : θ = 60) (h_length : L = 19) (h_cos : Real.cos (θ * Real.pi / 180) = 0.5) : 
  d = 9.5 :=
by
  sorry

end ladder_distance_from_wall_l115_115319


namespace company_workers_count_l115_115482

-- Definitions
def num_supervisors := 13
def team_leads_per_supervisor := 3
def workers_per_team_lead := 10

-- Hypothesis
def team_leads := num_supervisors * team_leads_per_supervisor
def workers := team_leads * workers_per_team_lead

-- Theorem to prove
theorem company_workers_count : workers = 390 :=
by
  sorry

end company_workers_count_l115_115482


namespace lcm_24_36_45_l115_115708

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l115_115708


namespace find_a_sequence_formula_l115_115375

variable (f : ℝ → ℝ) (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_f_def : ∀ x ≠ -a, f x = (a * x) / (a + x))
variable (h_f_2 : f 2 = 1) (h_seq_def : ∀ n : ℕ, a_seq (n+1) = f (a_seq n)) (h_a1 : a_seq 1 = 1)

theorem find_a : a = 2 :=
  sorry

theorem sequence_formula : ∀ n : ℕ, a_seq n = 2 / (n + 1) :=
  sorry

end find_a_sequence_formula_l115_115375


namespace simple_interest_years_l115_115784

theorem simple_interest_years (P : ℝ) (hP : P > 0) (R : ℝ := 2.5) (SI : ℝ := P / 5) : 
  ∃ T : ℝ, P * R * T / 100 = SI ∧ T = 8 :=
by
  sorry

end simple_interest_years_l115_115784


namespace not_all_inequalities_hold_l115_115576

theorem not_all_inequalities_hold (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(hlt_a : a < 1) (hlt_b : b < 1) (hlt_c : c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
by
  sorry

end not_all_inequalities_hold_l115_115576


namespace harry_morning_routine_l115_115356

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end harry_morning_routine_l115_115356


namespace angle_measure_triple_complement_l115_115654

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l115_115654


namespace harry_morning_routine_l115_115362

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end harry_morning_routine_l115_115362


namespace angle_triple_complement_l115_115648

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l115_115648


namespace find_k_l115_115526

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def construct_number (k : ℕ) : ℕ :=
  let n := 1000
  let a := (10^(2000 - k) - 1) / 9
  let b := (10^(1001) - 1) / 9
  a * 10^(1001) + k * 10^(1001 - k) - b

theorem find_k : ∀ k : ℕ, (construct_number k > 0) ∧ (isPerfectSquare (construct_number k) ↔ k = 2) := 
by 
  intro k
  sorry

end find_k_l115_115526


namespace probability_four_collinear_dots_l115_115255

noncomputable def probability_collinear_four_dots : ℚ :=
  let total_dots := 25
  let choose_4 := (total_dots.choose 4)
  let successful_outcomes := 60
  successful_outcomes / choose_4

theorem probability_four_collinear_dots :
  probability_collinear_four_dots = 12 / 2530 :=
by
  sorry

end probability_four_collinear_dots_l115_115255


namespace average_speed_of_bus_trip_l115_115978

theorem average_speed_of_bus_trip 
  (v d : ℝ) 
  (h1 : d = 560)
  (h2 : ∀ v > 0, ∀ Δv > 0, (d / v) - (d / (v + Δv)) = 2)
  (h3 : Δv = 10): 
  v = 50 := 
by 
  sorry

end average_speed_of_bus_trip_l115_115978


namespace tan_identity_l115_115765

variable (α β : ℝ)

theorem tan_identity (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : Real.sin (2 * α) = 2 * Real.sin (2 * β)) : 
  Real.tan (α + β) = 3 * Real.tan (α - β) := 
by 
  sorry

end tan_identity_l115_115765


namespace find_number_l115_115093

theorem find_number (x : ℝ) (h : 10 * x = 2 * x - 36) : x = -4.5 :=
sorry

end find_number_l115_115093


namespace expand_product_l115_115163

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l115_115163


namespace arithmetic_lemma_l115_115346

theorem arithmetic_lemma : 45 * 52 + 48 * 45 = 4500 := by
  sorry

end arithmetic_lemma_l115_115346


namespace angle_triple_complement_l115_115668

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l115_115668


namespace problem1_problem2_problem3_l115_115759

noncomputable def f (b a : ℝ) (x : ℝ) := (b - 2^x) / (2^x + a) 

-- (1) Prove values of a and b
theorem problem1 (a b : ℝ) : 
  (f b a 0 = 0) ∧ (f b a (-1) = -f b a 1) → (a = 1 ∧ b = 1) :=
sorry

-- (2) Prove f is decreasing function
theorem problem2 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f b a x₁ - f b a x₂ > 0 :=
sorry

-- (3) Find range of k such that inequality always holds
theorem problem3 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) (k : ℝ) : 
  (∀ t : ℝ, f b a (t^2 - 2*t) + f b a (2*t^2 - k) < 0) → k < -(1/3) :=
sorry

end problem1_problem2_problem3_l115_115759


namespace angle_is_67_l115_115640

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l115_115640


namespace determine_angle_C_in_DEF_l115_115406

def Triangle := Type

structure TriangleProps (T : Triangle) :=
  (right_angle : Prop)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

axiom triangle_ABC : Triangle
axiom triangle_DEF : Triangle

axiom ABC_props : TriangleProps triangle_ABC
axiom DEF_props : TriangleProps triangle_DEF

noncomputable def similar (T1 T2 : Triangle) : Prop := sorry

theorem determine_angle_C_in_DEF
  (h1 : ABC_props.right_angle = true)
  (h2 : ABC_props.angle_A = 30)
  (h3 : DEF_props.right_angle = true)
  (h4 : DEF_props.angle_B = 60)
  (h5 : similar triangle_ABC triangle_DEF) :
  DEF_props.angle_C = 30 :=
sorry

end determine_angle_C_in_DEF_l115_115406


namespace number_of_bottle_caps_put_inside_l115_115916

-- Definitions according to the conditions
def initial_bottle_caps : ℕ := 7
def final_bottle_caps : ℕ := 14
def additional_bottle_caps (initial final : ℕ) := final - initial

-- The main theorem to prove
theorem number_of_bottle_caps_put_inside : additional_bottle_caps initial_bottle_caps final_bottle_caps = 7 :=
by
  sorry

end number_of_bottle_caps_put_inside_l115_115916


namespace remainder_of_2_pow_33_mod_9_l115_115291

theorem remainder_of_2_pow_33_mod_9 : (2 ^ 33) % 9 = 8 :=
by
  sorry

end remainder_of_2_pow_33_mod_9_l115_115291


namespace Jan_height_is_42_l115_115511

-- Given conditions
def Cary_height : ℕ := 72
def Bill_height : ℕ := Cary_height / 2
def Jan_height : ℕ := Bill_height + 6

-- Statement to prove
theorem Jan_height_is_42 : Jan_height = 42 := by
  sorry

end Jan_height_is_42_l115_115511


namespace no_ingredient_pies_max_l115_115885

theorem no_ingredient_pies_max :
  ∃ (total apple blueberry cream chocolate no_ingredient : ℕ),
    total = 48 ∧
    apple = 24 ∧
    blueberry = 16 ∧
    cream = 18 ∧
    chocolate = 12 ∧
    no_ingredient = total - (apple + blueberry + chocolate - min apple blueberry - min apple chocolate - min blueberry chocolate) - cream ∧
    no_ingredient = 10 := sorry

end no_ingredient_pies_max_l115_115885


namespace max_value_frac_l115_115213

theorem max_value_frac (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    ∃ (c : ℝ), c = 1/4 ∧ (∀ (x y z : ℝ), (0 < x) → (0 < y) → (0 < z) → (xyz * (x + y + z)) / ((x + z)^2 * (z + y)^2) ≤ c) := 
by
  sorry

end max_value_frac_l115_115213


namespace angle_triple_complement_l115_115662

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l115_115662


namespace duck_travel_days_l115_115412

theorem duck_travel_days (x : ℕ) (h1 : 40 + 2 * 40 + x = 180) : x = 60 := by
  sorry

end duck_travel_days_l115_115412


namespace buoy_radius_proof_l115_115129

/-
We will define the conditions:
- width: 30 cm
- radius_ice_hole: 15 cm (half of width)
- depth: 12 cm
Then prove the radius of the buoy (r) equals 15.375 cm.
-/
noncomputable def radius_of_buoy : ℝ :=
  let width : ℝ := 30
  let depth : ℝ := 12
  let radius_ice_hole : ℝ := width / 2
  let r : ℝ := (369 / 24)
  r    -- the radius of the buoy

theorem buoy_radius_proof : radius_of_buoy = 15.375 :=
by 
  -- We assert that the above definition correctly computes the radius.
  sorry   -- Actual proof omitted

end buoy_radius_proof_l115_115129


namespace f_of_10_is_20_l115_115039

theorem f_of_10_is_20 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (3 * x + 1) = x^2 + 3 * x + 2) : f 10 = 20 :=
  sorry

end f_of_10_is_20_l115_115039


namespace rooms_in_house_l115_115937

-- define the number of paintings
def total_paintings : ℕ := 32

-- define the number of paintings per room
def paintings_per_room : ℕ := 8

-- define the number of rooms
def number_of_rooms (total_paintings : ℕ) (paintings_per_room : ℕ) : ℕ := total_paintings / paintings_per_room

-- state the theorem
theorem rooms_in_house : number_of_rooms total_paintings paintings_per_room = 4 :=
by sorry

end rooms_in_house_l115_115937


namespace amy_music_files_l115_115005

-- Define the number of total files on the flash drive
def files_on_flash_drive := 48.0

-- Define the number of video files on the flash drive
def video_files := 21.0

-- Define the number of picture files on the flash drive
def picture_files := 23.0

-- Define the number of music files, derived from the conditions
def music_files := files_on_flash_drive - (video_files + picture_files)

-- The theorem we need to prove
theorem amy_music_files : music_files = 4.0 := by
  sorry

end amy_music_files_l115_115005


namespace harry_morning_routine_l115_115363

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end harry_morning_routine_l115_115363


namespace find_f_15_l115_115598

theorem find_f_15
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - 2 * y) + 3 * x ^ 2 + 2) :
  f 15 = 1202 := 
sorry

end find_f_15_l115_115598


namespace angle_triple_complement_l115_115631

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l115_115631


namespace sum_of_distinct_nums_l115_115396

theorem sum_of_distinct_nums (m n p q : ℕ) (hmn : m ≠ n) (hmp : m ≠ p) (hmq : m ≠ q) 
(hnp : n ≠ p) (hnq : n ≠ q) (hpq : p ≠ q) (pos_m : 0 < m) (pos_n : 0 < n) 
(pos_p : 0 < p) (pos_q : 0 < q) (h : (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4) : 
  m + n + p + q = 24 :=
sorry

end sum_of_distinct_nums_l115_115396


namespace distinct_positive_least_sum_seven_integers_prod_2016_l115_115888

theorem distinct_positive_least_sum_seven_integers_prod_2016 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    n1 < n2 ∧ n2 < n3 ∧ n3 < n4 ∧ n4 < n5 ∧ n5 < n6 ∧ n6 < n7 ∧
    (n1 * n2 * n3 * n4 * n5 * n6 * n7) % 2016 = 0 ∧
    n1 + n2 + n3 + n4 + n5 + n6 + n7 = 31 :=
sorry

end distinct_positive_least_sum_seven_integers_prod_2016_l115_115888


namespace simplify_expression_l115_115450

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l115_115450


namespace isosceles_triangle_perimeter_l115_115761

theorem isosceles_triangle_perimeter (a b : ℕ) (h_eq : a = 5 ∨ a = 9) (h_side : b = 9 ∨ b = 5) (h_neq : a ≠ b) : 
  (a + a + b = 19 ∨ a + a + b = 23) :=
by
  sorry

end isosceles_triangle_perimeter_l115_115761


namespace gcd_12a_18b_l115_115909

theorem gcd_12a_18b (a b : ℕ) (h : Nat.gcd a b = 12) : Nat.gcd (12 * a) (18 * b) = 72 :=
sorry

end gcd_12a_18b_l115_115909


namespace g_1993_at_2_l115_115350

def g (x : ℚ) : ℚ := (2 + x) / (1 - 4 * x^2)

def g_n : ℕ → ℚ → ℚ 
| 0     => id
| (n+1) => λ x => g (g_n n x)

theorem g_1993_at_2 : g_n 1993 2 = 65 / 53 := 
  sorry

end g_1993_at_2_l115_115350


namespace angle_triple_complement_l115_115698

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l115_115698


namespace modulus_of_complex_l115_115227

open Complex

theorem modulus_of_complex : ∀ (z : ℂ), z = 3 - 2 * I → Complex.abs z = Real.sqrt 13 :=
by
  intro z
  intro h
  rw [h]
  simp [Complex.abs]
  sorry

end modulus_of_complex_l115_115227


namespace draw_9_cards_ensure_even_product_l115_115320

theorem draw_9_cards_ensure_even_product :
  ∀ (cards : Finset ℕ), (∀ x ∈ cards, 1 ≤ x ∧ x ≤ 16) →
  (cards.card = 9) →
  (∃ (subset : Finset ℕ), subset ⊆ cards ∧ ∃ k ∈ subset, k % 2 = 0) :=
by
  sorry

end draw_9_cards_ensure_even_product_l115_115320


namespace maize_storage_l115_115871

theorem maize_storage (x : ℝ)
  (h1 : 24 * x - 5 + 8 = 27) : x = 1 :=
  sorry

end maize_storage_l115_115871


namespace spend_money_l115_115841

theorem spend_money (n : ℕ) (h : n > 7) : ∃ a b : ℕ, 3 * a + 5 * b = n :=
by
  sorry

end spend_money_l115_115841


namespace probability_of_selecting_storybook_l115_115915

theorem probability_of_selecting_storybook (reference_books storybooks picture_books : ℕ) 
  (h1 : reference_books = 5) (h2 : storybooks = 3) (h3 : picture_books = 2) :
  (storybooks : ℚ) / (reference_books + storybooks + picture_books) = 3 / 10 :=
by {
  sorry
}

end probability_of_selecting_storybook_l115_115915


namespace second_number_l115_115124

theorem second_number (A B : ℝ) (h1 : 0.50 * A = 0.40 * B + 180) (h2 : A = 456) : B = 120 := 
by
  sorry

end second_number_l115_115124


namespace simplify_expression_l115_115439

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115439


namespace op_4_3_equals_23_l115_115048

def op (a b : ℕ) : ℕ := a ^ 2 + a * b + a - b ^ 2

theorem op_4_3_equals_23 : op 4 3 = 23 := by
  -- Proof steps would go here
  sorry

end op_4_3_equals_23_l115_115048


namespace answer_choices_l115_115866

theorem answer_choices (n : ℕ) (h : (n + 1) ^ 4 = 625) : n = 4 :=
by {
  sorry
}

end answer_choices_l115_115866


namespace intersection_is_correct_l115_115549

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | x < 1 }

theorem intersection_is_correct : (A ∩ B) = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_is_correct_l115_115549


namespace bus_stop_time_l115_115523

theorem bus_stop_time (v_no_stop v_with_stop : ℝ) (t_per_hour_minutes : ℝ) (h1 : v_no_stop = 48) (h2 : v_with_stop = 24) : t_per_hour_minutes = 30 := 
sorry

end bus_stop_time_l115_115523


namespace percentage_of_b_l115_115853

variable (a b c p : ℝ)

theorem percentage_of_b (h1 : 0.06 * a = 12) (h2 : p * b = 6) (h3 : c = b / a) : 
  p = 6 / (200 * c) := by
  sorry

end percentage_of_b_l115_115853


namespace tim_youth_comparison_l115_115304

theorem tim_youth_comparison :
  let Tim_age : ℕ := 5
  let Rommel_age : ℕ := 3 * Tim_age
  let Jenny_age : ℕ := Rommel_age + 2
  Jenny_age - Tim_age = 12 := 
by 
  sorry

end tim_youth_comparison_l115_115304


namespace combined_fish_population_l115_115057

theorem combined_fish_population:
  (n1A n2A mA n1B n2B mB : ℕ)
  (h1 : n1A = 50)
  (h2 : n2A = 60)
  (h3 : mA = 4)
  (h4 : n1B = 30)
  (h5 : n2B = 40)
  (h6 : mB = 3) :
  let N_A := (n1A * n2A) / mA in
  let N_B := (n1B * n2B) / mB in
  N_A + N_B = 1150 := by
{
  sorry
}

end combined_fish_population_l115_115057


namespace blue_pill_cost_l115_115027

variable (cost_blue_pill : ℕ) (cost_red_pill : ℕ) (daily_cost : ℕ) 
variable (num_days : ℕ) (total_cost : ℕ)
variable (cost_diff : ℕ)

theorem blue_pill_cost :
  num_days = 21 ∧
  total_cost = 966 ∧
  cost_diff = 4 ∧
  daily_cost = total_cost / num_days ∧
  daily_cost = cost_blue_pill + cost_red_pill ∧
  cost_blue_pill = cost_red_pill + cost_diff ∧
  daily_cost = 46 →
  cost_blue_pill = 25 := by
  sorry

end blue_pill_cost_l115_115027


namespace problem_solution_l115_115219

theorem problem_solution (a b c d : ℕ) (h : 342 * (a * b * c * d + a * b + a * d + c * d + 1) = 379 * (b * c * d + b + d)) :
  (a * 10^3 + b * 10^2 + c * 10 + d) = 1949 :=
by
  sorry

end problem_solution_l115_115219


namespace amount_of_silver_l115_115253

-- Definitions
def total_silver (x : ℕ) : Prop :=
  (x - 4) % 7 = 0 ∧ (x + 8) % 9 = 1

-- Theorem to be proven
theorem amount_of_silver (x : ℕ) (h : total_silver x) : (x - 4)/7 = (x + 8)/9 :=
by sorry

end amount_of_silver_l115_115253


namespace triangle_property_l115_115768

theorem triangle_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_perimeter : a + b + c = 12) (h_inradius : 2 * (a + b + c) = 24) :
    ¬((a^2 + b^2 = c^2) ∨ (a^2 + b^2 > c^2) ∨ (c^2 > a^2 + b^2)) := 
sorry

end triangle_property_l115_115768


namespace sin_cos_product_l115_115557

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l115_115557


namespace sampling_correct_probability_of_event_a_l115_115837

noncomputable def number_of_athletes_from_a : ℕ := 27
noncomputable def number_of_athletes_from_b : ℕ := 9
noncomputable def number_of_athletes_from_c : ℕ := 18
noncomputable def total_athletes_selected : ℕ := 6

-- Calculating the number of athletes to be sampled from each association
def selected_from_a : ℕ := number_of_athletes_from_a * total_athletes_selected / (number_of_athletes_from_a + number_of_athletes_from_b + number_of_athletes_from_c)
def selected_from_b : ℕ := number_of_athletes_from_b * total_athletes_selected / (number_of_athletes_from_a + number_of_athletes_from_b + number_of_athletes_from_c)
def selected_from_c : ℕ := number_of_athletes_from_c * total_athletes_selected / (number_of_athletes_from_a + number_of_athletes_from_b + number_of_athletes_from_c)

theorem sampling_correct :
  selected_from_a = 3 ∧ selected_from_b = 1 ∧ selected_from_c = 2 := by
  sorry

-- Defining the number of combinations
def total_combinations (n k : ℕ) : ℕ := nat.choose n k

def favorable_combinations : ℕ := 9
def total_combinations_sampling : ℕ := total_combinations total_athletes_selected 2

theorem probability_of_event_a :
  favorable_combinations / total_combinations_sampling = (3 / 5 : ℝ) := by
  sorry

end sampling_correct_probability_of_event_a_l115_115837


namespace Mahesh_completes_in_60_days_l115_115581

noncomputable def MaheshWork (W : ℝ) : ℝ :=
    W / 60

variables (W : ℝ)
variables (M R : ℝ)
variables (daysMahesh daysRajesh daysFullRajesh : ℝ)

theorem Mahesh_completes_in_60_days
  (h1 : daysMahesh = 20)
  (h2 : daysRajesh = 30)
  (h3 : daysFullRajesh = 45)
  (hR : R = W / daysFullRajesh)
  (hM : M = (W - R * daysRajesh) / daysMahesh) :
  W / M = 60 :=
by
  sorry

end Mahesh_completes_in_60_days_l115_115581


namespace angle_triple_complement_l115_115632

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l115_115632


namespace angle_triple_complement_l115_115647

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l115_115647


namespace probability_of_selecting_product_not_less_than_4_l115_115078

theorem probability_of_selecting_product_not_less_than_4 :
  let total_products := 5 
  let favorable_outcomes := 2 
  (favorable_outcomes : ℚ) / total_products = 2 / 5 := 
by 
  sorry

end probability_of_selecting_product_not_less_than_4_l115_115078


namespace angle_is_67_l115_115645

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l115_115645


namespace min_side_length_of_square_l115_115896

theorem min_side_length_of_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ s : ℝ, s = 
    if a < (Real.sqrt 2 + 1) * b then 
      a 
    else 
      (Real.sqrt 2 / 2) * (a + b) := 
sorry

end min_side_length_of_square_l115_115896


namespace proof_of_greatest_sum_quotient_remainder_l115_115600

def greatest_sum_quotient_remainder : Prop :=
  ∃ q r : ℕ, 1051 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q + r = 61

theorem proof_of_greatest_sum_quotient_remainder : greatest_sum_quotient_remainder := 
sorry

end proof_of_greatest_sum_quotient_remainder_l115_115600


namespace quadratic_real_roots_range_l115_115783

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + m = 0) → m ≤ 9 / 4 :=
by
  sorry

end quadratic_real_roots_range_l115_115783


namespace A_can_give_C_start_l115_115248

noncomputable def start_A_can_give_C : ℝ :=
  let start_AB := 50
  let start_BC := 157.89473684210532
  start_AB + start_BC

theorem A_can_give_C_start :
  start_A_can_give_C = 207.89473684210532 :=
by
  sorry

end A_can_give_C_start_l115_115248


namespace distance_between_parallel_lines_l115_115829

-- Definitions
def line_eq1 (x y : ℝ) : Prop := 3 * x - 4 * y - 12 = 0
def line_eq2 (x y : ℝ) : Prop := 6 * x - 8 * y + 11 = 0

-- Statement of the problem
theorem distance_between_parallel_lines :
  (∀ x y : ℝ, line_eq1 x y ↔ line_eq2 x y) →
  (∃ d : ℝ, d = 7 / 2) :=
by
  sorry

end distance_between_parallel_lines_l115_115829


namespace PropA_neither_sufficient_nor_necessary_for_PropB_l115_115032

variable (a b : ℤ)

-- Proposition A
def PropA : Prop := a + b ≠ 4

-- Proposition B
def PropB : Prop := a ≠ 1 ∧ b ≠ 3

-- The required statement
theorem PropA_neither_sufficient_nor_necessary_for_PropB : ¬(PropA a b → PropB a b) ∧ ¬(PropB a b → PropA a b) :=
by
  sorry

end PropA_neither_sufficient_nor_necessary_for_PropB_l115_115032


namespace expand_expression_l115_115180

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115180


namespace correct_proposition_l115_115004

theorem correct_proposition :
  (∃ x₀ : ℤ, x₀^2 = 1) ∧ ¬(∃ x₀ : ℤ, x₀^2 < 0) ∧ ¬(∀ x : ℤ, x^2 ≤ 0) ∧ ¬(∀ x : ℤ, x^2 ≥ 1) :=
by
  sorry

end correct_proposition_l115_115004


namespace AM_GM_Inequality_four_vars_l115_115388

theorem AM_GM_Inequality_four_vars (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 :=
by sorry

end AM_GM_Inequality_four_vars_l115_115388


namespace magnitude_difference_l115_115905

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (norm_a : ‖a‖ = 2) (norm_b : ‖b‖ = 1) (norm_a_plus_b : ‖a + b‖ = Real.sqrt 3)

theorem magnitude_difference :
  ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end magnitude_difference_l115_115905


namespace volume_percentage_error_l115_115135

theorem volume_percentage_error (L W H : ℝ) (hL : L > 0) (hW : W > 0) (hH : H > 0) :
  let V_true := L * W * H
  let L_meas := 1.08 * L
  let W_meas := 1.12 * W
  let H_meas := 1.05 * H
  let V_calc := L_meas * W_meas * H_meas
  let percentage_error := ((V_calc - V_true) / V_true) * 100
  percentage_error = 25.424 :=
by
  sorry

end volume_percentage_error_l115_115135


namespace expand_expression_l115_115150

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115150


namespace assignment_plan_count_l115_115568

noncomputable def number_of_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let tasks := ["translation", "tour guide", "etiquette", "driver"]
  let v1 := ["Xiao Zhang", "Xiao Zhao"]
  let v2 := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Condition: Xiao Zhang and Xiao Zhao can only take positions for translation and tour guide
  -- Calculate the number of ways to assign based on the given conditions
  -- 36 is the total number of assignment plans
  36

theorem assignment_plan_count :
  number_of_assignment_plans = 36 :=
  sorry

end assignment_plan_count_l115_115568


namespace common_ratio_of_gp_l115_115487

theorem common_ratio_of_gp (a r : ℝ) (h1 : r ≠ 1) 
  (h2 : (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 343) : r = 6 := 
by
  sorry

end common_ratio_of_gp_l115_115487


namespace acute_angles_sine_relation_l115_115401

theorem acute_angles_sine_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : 2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β) : α < β :=
by
  sorry

end acute_angles_sine_relation_l115_115401


namespace evaluate_expression_at_minus_one_l115_115945

theorem evaluate_expression_at_minus_one :
  ((-1 + 1) * (-1 - 2) + 2 * (-1 + 4) * (-1 - 4)) = -30 := by
  sorry

end evaluate_expression_at_minus_one_l115_115945


namespace geometric_sequence_a2_l115_115948

noncomputable def geometric_sequence_sum (n : ℕ) (a : ℝ) : ℝ :=
  a * (3^n) - 2

theorem geometric_sequence_a2 (a : ℝ) : (∃ a1 a2 a3 : ℝ, 
  a1 = geometric_sequence_sum 1 a ∧ 
  a1 + a2 = geometric_sequence_sum 2 a ∧ 
  a1 + a2 + a3 = geometric_sequence_sum 3 a ∧ 
  a2 = 6 * a ∧ 
  a3 = 18 * a ∧ 
  (6 * a)^2 = (a1) * (a3) ∧ 
  a = 2) →
  a2 = 12 :=
by
  intros h
  sorry

end geometric_sequence_a2_l115_115948


namespace algebraic_expression_no_linear_term_l115_115241

theorem algebraic_expression_no_linear_term (a : ℝ) :
  (∀ x : ℝ, (x + a) * (x - 1/2) = x^2 - a/2 ↔ a = 1/2) :=
by
  sorry

end algebraic_expression_no_linear_term_l115_115241


namespace calculate_train_speed_l115_115522

def speed_train_excluding_stoppages (distance_per_hour_including_stoppages : ℕ) (stoppage_minutes_per_hour : ℕ) : ℕ :=
  let effective_running_time_per_hour := 60 - stoppage_minutes_per_hour
  let effective_running_time_in_hours := effective_running_time_per_hour / 60
  distance_per_hour_including_stoppages / effective_running_time_in_hours

theorem calculate_train_speed :
  speed_train_excluding_stoppages 42 4 = 45 :=
by
  sorry

end calculate_train_speed_l115_115522


namespace prod_one_minus_nonneg_reals_ge_half_l115_115097

theorem prod_one_minus_nonneg_reals_ge_half (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3)
  (h_sum : x1 + x2 + x3 ≤ 1/2) : 
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1 / 2 := 
by
  sorry

end prod_one_minus_nonneg_reals_ge_half_l115_115097


namespace valid_assignment_statement_l115_115714

theorem valid_assignment_statement (S a : ℕ) : (S = a + 1) ∧ ¬(a + 1 = S) ∧ ¬(S - 1 = a) ∧ ¬(S - a = 1) := by
  sorry

end valid_assignment_statement_l115_115714


namespace regular_polygon_sides_l115_115726

theorem regular_polygon_sides (n : ℕ) (h : ∀ n, (n > 2) → (360 / n = 20)) : n = 18 := sorry

end regular_polygon_sides_l115_115726


namespace total_votes_cast_l115_115923

def votes_witch : ℕ := 7
def votes_unicorn : ℕ := 3 * votes_witch
def votes_dragon : ℕ := votes_witch + 25
def votes_total : ℕ := votes_witch + votes_unicorn + votes_dragon

theorem total_votes_cast : votes_total = 60 := by
  sorry

end total_votes_cast_l115_115923


namespace problem_l115_115040

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (x - Real.pi / 2)

theorem problem 
: (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ c, c = (Real.pi / 2) ∧ f c = 0) → (T = Real.pi ∧ c = (Real.pi / 2)) :=
sorry

end problem_l115_115040


namespace area_of_yard_proof_l115_115753

def area_of_yard (L W : ℕ) : ℕ :=
  L * W

theorem area_of_yard_proof (L W : ℕ) (hL : L = 40) (hFence : 2 * W + L = 52) : 
  area_of_yard L W = 240 := 
by 
  sorry

end area_of_yard_proof_l115_115753


namespace horse_distribution_l115_115127

variable (b₁ b₂ b₃ : ℕ) 
variable (a : Matrix (Fin 3) (Fin 3) ℝ)
variable (h1 : a 0 0 > a 0 1 ∧ a 0 0 > a 0 2)
variable (h2 : a 1 1 > a 1 0 ∧ a 1 1 > a 1 2)
variable (h3 : a 2 2 > a 2 0 ∧ a 2 2 > a 2 1)

theorem horse_distribution :
  ∃ n : ℕ, ∀ (b₁ b₂ b₃ : ℕ), min b₁ (min b₂ b₃) > n → 
  ∃ (x1 y1 x2 y2 x3 y3 : ℕ), 3*x1 + y1 = b₁ ∧ 3*x2 + y2 = b₂ ∧ 3*x3 + y3 = b₃ ∧
  y1*a 0 0 > y2*a 0 1 ∧ y1*a 0 0 > y3*a 0 2 ∧
  y2*a 1 1 > y1*a 1 0 ∧ y2*a 1 1 > y3*a 1 2 ∧
  y3*a 2 2 > y1*a 2 0 ∧ y3*a 2 2 > y2*a 2 1 :=
sorry

end horse_distribution_l115_115127


namespace domain_of_f_comp_l115_115564

theorem domain_of_f_comp (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ x^2 - 2 ∧ x^2 - 2 ≤ -1) →
  (∀ x, - (4 : ℝ) / 3 ≤ x ∧ x ≤ -1 → -2 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ -1) :=
by
  sorry

end domain_of_f_comp_l115_115564


namespace unique_handshakes_l115_115971

-- Define the circular arrangement and handshakes conditions
def num_people := 30
def handshakes_per_person := 2

theorem unique_handshakes : 
  (num_people * handshakes_per_person) / 2 = 30 :=
by
  -- Sorry is used here as a placeholder for the proof
  sorry

end unique_handshakes_l115_115971


namespace eval_g_231_l115_115904

def g (a b c : ℤ) : ℚ :=
  (c ^ 2 + a ^ 2) / (c - b)

theorem eval_g_231 : g 2 (-3) 1 = 5 / 4 :=
by
  sorry

end eval_g_231_l115_115904


namespace prob_2_lt_X_le_4_l115_115769

-- Define the PMF of the random variable X
noncomputable def pmf_X (k : ℕ) : ℝ :=
  if h : k ≥ 1 then 1 / (2 ^ k) else 0

-- Define the probability that X lies in the range (2, 4]
noncomputable def P_2_lt_X_le_4 : ℝ :=
  pmf_X 3 + pmf_X 4

-- Theorem stating the probability of x lying in (2, 4) is 3/16.
theorem prob_2_lt_X_le_4 : P_2_lt_X_le_4 = 3 / 16 := 
by
  -- Provide proof here
  sorry

end prob_2_lt_X_le_4_l115_115769


namespace triangle_perimeter_l115_115381

theorem triangle_perimeter
  (a : ℝ) (a_gt_5 : a > 5)
  (ellipse : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 25 = 1)
  (dist_foci : 8 = 2 * 4) :
  4 * Real.sqrt (41) = 4 * Real.sqrt (41) := by
sorry

end triangle_perimeter_l115_115381


namespace reducible_fraction_probability_with_12_is_four_sevenths_l115_115229

open Set

-- Define the set A
def A : Set ℕ := {2, 4, 5, 6, 8, 11, 12, 17}

-- Define the condition that one of the chosen numbers is 12
def chosen_number := 12

-- Define the definition of a fraction being reducible
def is_reducible_fraction (n d : ℕ) : Prop :=
  n.gcd d ≠ 1

-- Calculate the probability of forming a reducible fraction when one number is fixed as 12
def reducible_fraction_probability : ℚ :=
  let total_fractions := 7
  let reducible_fractions := 4
  reducible_fractions / total_fractions

theorem reducible_fraction_probability_with_12_is_four_sevenths :
  reducible_fraction_probability = 4 / 7 :=
by
  sorry

end reducible_fraction_probability_with_12_is_four_sevenths_l115_115229


namespace circle_radius_3_l115_115603

theorem circle_radius_3 :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 2 * y - 7 = 0) → (∃ r : ℝ, r = 3) :=
by
  sorry

end circle_radius_3_l115_115603


namespace angle_triple_complement_l115_115667

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l115_115667


namespace third_chapter_is_24_pages_l115_115975

-- Define the total number of pages in the book
def total_pages : ℕ := 125

-- Define the number of pages in the first chapter
def first_chapter_pages : ℕ := 66

-- Define the number of pages in the second chapter
def second_chapter_pages : ℕ := 35

-- Define the number of pages in the third chapter
def third_chapter_pages : ℕ := total_pages - (first_chapter_pages + second_chapter_pages)

-- Prove that the number of pages in the third chapter is 24
theorem third_chapter_is_24_pages : third_chapter_pages = 24 := by
  sorry

end third_chapter_is_24_pages_l115_115975


namespace bert_spent_fraction_at_hardware_store_l115_115138

variable (f : ℝ)

def initial_money : ℝ := 41.99
def after_hardware (f : ℝ) := (1 - f) * initial_money
def after_dry_cleaners (f : ℝ) := after_hardware f - 7
def after_grocery (f : ℝ) := 0.5 * after_dry_cleaners f

theorem bert_spent_fraction_at_hardware_store 
(h1 : after_grocery f = 10.50) : 
  f = 0.3332 :=
by
  sorry

end bert_spent_fraction_at_hardware_store_l115_115138


namespace x_varies_inversely_l115_115719

theorem x_varies_inversely (y: ℝ) (x: ℝ): (∃ k: ℝ, (∀ y: ℝ, x = k / y ^ 2) ∧ (1 = k / 3 ^ 2)) → x = 0.5625 :=
by
  sorry

end x_varies_inversely_l115_115719


namespace expand_expression_l115_115186

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115186


namespace f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l115_115421

noncomputable def f (x : ℝ) : ℝ := x + 4/x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem f_min_value_pos : ∀ x : ℝ, x > 0 → f x ≥ 4 :=
by
  sorry

theorem f_minimum_at_2 : f 2 = 4 :=
by
  sorry

theorem f_increasing_intervals : (MonotoneOn f {x | x ≤ -2} ∧ MonotoneOn f {x | x ≥ 2}) :=
by
  sorry

end f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l115_115421


namespace range_of_a_l115_115263

noncomputable def f (a x : ℝ) : ℝ := x + (a^2) / (4 * x)
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f a x1 ≥ g x2) → 
  2 * Real.sqrt (Real.exp 1 - 2) ≤ a := sorry

end range_of_a_l115_115263


namespace determine_a_l115_115562

theorem determine_a (a : ℝ) (h : (binom 5 3) * a^3 = 80) : a = 2 := by
  sorry

end determine_a_l115_115562


namespace distance_from_B_to_center_is_74_l115_115858

noncomputable def circle_radius := 10
noncomputable def B_distance (a b : ℝ) := a^2 + b^2

theorem distance_from_B_to_center_is_74 
  (a b : ℝ)
  (hA : a^2 + (b + 6)^2 = 100)
  (hC : (a + 4)^2 + b^2 = 100) :
  B_distance a b = 74 :=
sorry

end distance_from_B_to_center_is_74_l115_115858


namespace deductive_reasoning_l115_115789

theorem deductive_reasoning (
  deductive_reasoning_form : Prop
): ¬(deductive_reasoning_form → true → correct_conclusion) :=
by sorry

end deductive_reasoning_l115_115789


namespace simplify_expression_l115_115445

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115445


namespace expand_polynomial_eq_l115_115190

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l115_115190


namespace angle_triple_complement_l115_115701

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l115_115701


namespace completing_the_square_transformation_l115_115112

theorem completing_the_square_transformation (x : ℝ) : 
  (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro h
  -- Transformation steps will be shown in the proof
  sorry

end completing_the_square_transformation_l115_115112


namespace compute_105_times_95_l115_115347

theorem compute_105_times_95 : (105 * 95 = 9975) :=
by
  sorry

end compute_105_times_95_l115_115347


namespace angle_triple_complement_l115_115694

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l115_115694


namespace prove_ratio_l115_115239

variable (a b c d : ℚ)

-- Conditions
def cond1 : a / b = 5 := sorry
def cond2 : b / c = 1 / 4 := sorry
def cond3 : c / d = 7 := sorry

-- Theorem to prove the final result
theorem prove_ratio (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end prove_ratio_l115_115239


namespace neg_proposition_l115_115956

theorem neg_proposition :
  (¬(∀ x : ℕ, x^3 > x^2)) ↔ (∃ x : ℕ, x^3 ≤ x^2) := 
sorry

end neg_proposition_l115_115956


namespace twenty_five_percent_of_2004_l115_115846

theorem twenty_five_percent_of_2004 : (1 / 4 : ℝ) * 2004 = 501 := by
  sorry

end twenty_five_percent_of_2004_l115_115846


namespace total_cards_after_giveaway_l115_115137

def ben_basketball_boxes := 8
def cards_per_basketball_box := 20
def ben_baseball_boxes := 10
def cards_per_baseball_box := 15
def ben_football_boxes := 12
def cards_per_football_box := 12

def alex_hockey_boxes := 6
def cards_per_hockey_box := 15
def alex_soccer_boxes := 9
def cards_per_soccer_box := 18

def cards_given_away := 175

def total_cards_for_ben := 
  (ben_basketball_boxes * cards_per_basketball_box) + 
  (ben_baseball_boxes * cards_per_baseball_box) + 
  (ben_football_boxes * cards_per_football_box)

def total_cards_for_alex := 
  (alex_hockey_boxes * cards_per_hockey_box) + 
  (alex_soccer_boxes * cards_per_soccer_box)

def total_cards_before_exchange := total_cards_for_ben + total_cards_for_alex

def ben_gives_to_alex := 
  (ben_basketball_boxes * (cards_per_basketball_box / 2)) + 
  (ben_baseball_boxes * (cards_per_baseball_box / 2))

def total_cards_remaining := total_cards_before_exchange - cards_given_away

theorem total_cards_after_giveaway :
  total_cards_before_exchange - cards_given_away = 531 := by
  sorry

end total_cards_after_giveaway_l115_115137


namespace discriminant_zero_l115_115468

theorem discriminant_zero (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -2) (h₃ : c = 1) :
  (b^2 - 4 * a * c) = 0 :=
by
  sorry

end discriminant_zero_l115_115468


namespace total_animal_crackers_eaten_l115_115269

-- Define the context and conditions
def number_of_students : ℕ := 20
def uneaten_students : ℕ := 2
def crackers_per_pack : ℕ := 10

-- Define the statement and prove the question equals the answer given the conditions
theorem total_animal_crackers_eaten : 
  (number_of_students - uneaten_students) * crackers_per_pack = 180 := by
  sorry

end total_animal_crackers_eaten_l115_115269


namespace total_votes_is_correct_l115_115059

-- Definitions and theorem statement
theorem total_votes_is_correct (T : ℝ) 
  (votes_for_A : ℝ) 
  (candidate_A_share : ℝ) 
  (valid_vote_fraction : ℝ) 
  (invalid_vote_fraction : ℝ) 
  (votes_for_A_equals: votes_for_A = 380800) 
  (candidate_A_share_equals: candidate_A_share = 0.80) 
  (valid_vote_fraction_equals: valid_vote_fraction = 0.85) 
  (invalid_vote_fraction_equals: invalid_vote_fraction = 0.15) 
  (valid_vote_computed: votes_for_A = candidate_A_share * valid_vote_fraction * T): 
  T = 560000 := 
by 
  sorry

end total_votes_is_correct_l115_115059


namespace largest_prime_factor_of_4752_l115_115313

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬ (m ∣ n)

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p)

def pf_4752 : ℕ := 4752

theorem largest_prime_factor_of_4752 : largest_prime_factor pf_4752 11 :=
  by
  sorry

end largest_prime_factor_of_4752_l115_115313


namespace angle_triple_complement_l115_115699

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l115_115699


namespace increase_by_1_or_prime_l115_115982

theorem increase_by_1_or_prime (a : ℕ → ℕ) :
  a 0 = 6 →
  (∀ n, a (n + 1) = a n + Nat.gcd (a n) (n + 1)) →
  ∀ n, n < 1000000 → (∃ p, p = 1 ∨ Nat.Prime p ∧ a (n + 1) = a n + p) :=
by
  intro ha0 ha_step
  -- Proof omitted
  sorry

end increase_by_1_or_prime_l115_115982


namespace area_of_right_triangle_l115_115460

theorem area_of_right_triangle (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 10) (h_angle : angle = 45) :
  (1 / 2) * (5 * Real.sqrt 2) * (5 * Real.sqrt 2) = 25 :=
by
  -- Proof goes here
  sorry

end area_of_right_triangle_l115_115460


namespace triple_complement_angle_l115_115678

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l115_115678


namespace angle_measure_triple_complement_l115_115655

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l115_115655


namespace students_not_examined_l115_115836

theorem students_not_examined (boys girls examined : ℕ) (h1 : boys = 121) (h2 : girls = 83) (h3 : examined = 150) : 
  (boys + girls - examined = 54) := by
  sorry

end students_not_examined_l115_115836


namespace largest_possible_sum_l115_115371

theorem largest_possible_sum :
  let a := 12
  let b := 6
  let c := 6
  let d := 12
  a + b = c + d ∧ a + b + 15 = 33 :=
by
  have h1 : 12 + 6 = 6 + 12 := by norm_num
  have h2 : 12 + 6 + 15 = 33 := by norm_num
  exact ⟨h1, h2⟩

end largest_possible_sum_l115_115371


namespace find_x_l115_115391

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ↔ x = 80 / 3 := by
  sorry

end find_x_l115_115391


namespace total_swimming_hours_over_4_weeks_l115_115272

def weekday_swimming_per_day : ℕ := 2  -- Percy swims 2 hours per weekday
def weekday_days_per_week : ℕ := 5     -- Percy swims for 5 days a week
def weekend_swimming_per_week : ℕ := 3 -- Percy swims 3 hours on the weekend
def weeks : ℕ := 4                     -- The number of weeks is 4

-- Define the total swimming hours over 4 weeks
theorem total_swimming_hours_over_4_weeks :
  weekday_swimming_per_day * weekday_days_per_week * weeks + weekend_swimming_per_week * weeks = 52 :=
by
  sorry

end total_swimming_hours_over_4_weeks_l115_115272


namespace largest_k_rooks_l115_115251

noncomputable def rooks_max_k (board_size : ℕ) : ℕ := 
  if board_size = 10 then 16 else 0

theorem largest_k_rooks {k : ℕ} (h : 0 ≤ k ∧ k ≤ 100) :
  k ≤ rooks_max_k 10 := 
sorry

end largest_k_rooks_l115_115251


namespace ratio_of_length_to_breadth_l115_115461

theorem ratio_of_length_to_breadth 
    (breadth : ℝ) (area : ℝ) (h_breadth : breadth = 12) (h_area : area = 432)
    (h_area_formula : area = l * breadth) : 
    l / breadth = 3 :=
sorry

end ratio_of_length_to_breadth_l115_115461


namespace mary_income_more_than_tim_income_l115_115427

variables (J T M : ℝ)
variables (h1 : T = 0.60 * J) (h2 : M = 0.8999999999999999 * J)

theorem mary_income_more_than_tim_income : (M - T) / T * 100 = 50 :=
by
  sorry

end mary_income_more_than_tim_income_l115_115427


namespace crayons_in_new_set_l115_115426

theorem crayons_in_new_set (initial_crayons : ℕ) (half_loss : ℕ) (total_after_purchase : ℕ) (initial_crayons_eq : initial_crayons = 18) (half_loss_eq : half_loss = initial_crayons / 2) (total_eq : total_after_purchase = 29) :
  total_after_purchase - (initial_crayons - half_loss) = 20 :=
by
  sorry

end crayons_in_new_set_l115_115426


namespace ticket_costs_l115_115308

-- Define the conditions
def cost_per_ticket : ℕ := 44
def number_of_tickets : ℕ := 7

-- Define the total cost calculation
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Prove that given the conditions, the total cost is 308
theorem ticket_costs :
  total_cost = 308 :=
by
  -- Proof steps here
  sorry

end ticket_costs_l115_115308


namespace javier_time_outlining_l115_115258

variable (O : ℕ)
variable (W : ℕ := O + 28)
variable (P : ℕ := (O + 28) / 2)
variable (total_time : ℕ := O + W + P)

theorem javier_time_outlining
  (h1 : total_time = 117)
  (h2 : W = O + 28)
  (h3 : P = (O + 28) / 2)
  : O = 30 := by 
  sorry

end javier_time_outlining_l115_115258


namespace points_of_third_l115_115566

noncomputable def points_of_first : ℕ := 11
noncomputable def points_of_second : ℕ := 7
noncomputable def points_of_fourth : ℕ := 2
noncomputable def johns_total_points : ℕ := 38500

theorem points_of_third :
  ∃ x : ℕ, (points_of_first * points_of_second * x * points_of_fourth ∣ johns_total_points) ∧
    (johns_total_points / (points_of_first * points_of_second * points_of_fourth)) = x := 
sorry

end points_of_third_l115_115566


namespace adam_tickets_left_l115_115008

def tickets_left (total_tickets : ℕ) (ticket_cost : ℕ) (total_spent : ℕ) : ℕ :=
  total_tickets - total_spent / ticket_cost

theorem adam_tickets_left :
  tickets_left 13 9 81 = 4 := 
by
  sorry

end adam_tickets_left_l115_115008


namespace area_enclosed_by_abs_eq_l115_115519

theorem area_enclosed_by_abs_eq (x y : ℝ) : 
  (|x| + |3 * y| = 12) → (∃ area : ℝ, area = 96) :=
by
  sorry

end area_enclosed_by_abs_eq_l115_115519


namespace harry_morning_routine_l115_115357

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end harry_morning_routine_l115_115357


namespace num_pairs_divisible_7_l115_115907

theorem num_pairs_divisible_7 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000)
  (divisible : (x^2 + y^2) % 7 = 0) : 
  (∃ k : ℕ, k = 20164) :=
sorry

end num_pairs_divisible_7_l115_115907


namespace probability_same_suit_JQKA_l115_115529

theorem probability_same_suit_JQKA  : 
  let deck_size := 52 
  let prob_J := 4 / deck_size
  let prob_Q_given_J := 1 / (deck_size - 1) 
  let prob_K_given_JQ := 1 / (deck_size - 2)
  let prob_A_given_JQK := 1 / (deck_size - 3)
  prob_J * prob_Q_given_J * prob_K_given_JQ * prob_A_given_JQK = 1 / 1624350 :=
by
  sorry

end probability_same_suit_JQKA_l115_115529


namespace range_of_e_l115_115228

theorem range_of_e (a b c d e : ℝ) (h₁ : a + b + c + d + e = 8) (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end range_of_e_l115_115228


namespace cos_sin_equation_solution_l115_115823

noncomputable def solve_cos_sin_equation (x : ℝ) (n : ℤ) : Prop :=
  let lhs := (Real.cos x) / (Real.sqrt 3)
  let rhs := Real.sqrt ((1 - (Real.cos (2*x)) - 2 * (Real.sin x)^3) / (6 * Real.sin x - 2))
  (lhs = rhs) ∧ (Real.cos x ≥ 0)

theorem cos_sin_equation_solution:
  (∃ (x : ℝ) (n : ℤ), solve_cos_sin_equation x n) ↔ 
  ∃ (n : ℤ), (x = (π / 2) + 2 * π * n) ∨ (x = (π / 6) + 2 * π * n) :=
by
  sorry

end cos_sin_equation_solution_l115_115823


namespace simplify_expression_l115_115440

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115440


namespace problem1_problem2_problem3_problem4_l115_115813

open Set

def M : Set ℝ := { x | x > 3 / 2 }
def N : Set ℝ := { x | x < 1 ∨ x > 3 }
def R := {x : ℝ | 1 ≤ x ∧ x ≤ 3 / 2}

theorem problem1 : M = { x | 2 * x - 3 > 0 } := sorry
theorem problem2 : N = { x | (x - 3) * (x - 1) > 0 } := sorry
theorem problem3 : M ∩ N = { x | x > 3 } := sorry
theorem problem4 : (M ∪ N)ᶜ = R := sorry

end problem1_problem2_problem3_problem4_l115_115813


namespace quadratic_root_expression_value_l115_115781

theorem quadratic_root_expression_value (a : ℝ) 
  (h : a^2 - 2 * a - 3 = 0) : 2 * a^2 - 4 * a + 1 = 7 :=
by
  sorry

end quadratic_root_expression_value_l115_115781


namespace tan_alpha_tan_beta_is_2_l115_115552

theorem tan_alpha_tan_beta_is_2
  (α β : ℝ)
  (h1 : Real.sin α = 2 * Real.sin β)
  (h2 : Real.sin (α + β) * Real.tan (α - β) = 1) :
  Real.tan α * Real.tan β = 2 :=
sorry

end tan_alpha_tan_beta_is_2_l115_115552


namespace sum_moments_equal_l115_115120

theorem sum_moments_equal
  (x1 x2 x3 y1 y2 : ℝ)
  (m1 m2 m3 n1 n2 : ℝ) :
  n1 * y1 + n2 * y2 = m1 * x1 + m2 * x2 + m3 * x3 :=
sorry

end sum_moments_equal_l115_115120


namespace scout_troop_profit_l115_115339

noncomputable def buy_price_per_bar : ℚ := 3 / 4
noncomputable def sell_price_per_bar : ℚ := 2 / 3
noncomputable def num_candy_bars : ℕ := 800

theorem scout_troop_profit :
  num_candy_bars * (sell_price_per_bar : ℚ) - num_candy_bars * (buy_price_per_bar : ℚ) = -66.64 :=
by
  sorry

end scout_troop_profit_l115_115339


namespace angle_measure_l115_115690

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l115_115690


namespace lcm_24_36_45_l115_115709

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l115_115709


namespace k_value_if_divisible_l115_115717

theorem k_value_if_divisible :
  ∀ k : ℤ, (x^2 + k * x - 3) % (x - 1) = 0 → k = 2 :=
by
  intro k
  sorry

end k_value_if_divisible_l115_115717


namespace inequality_solution_set_compare_mn_and_2m_plus_2n_l115_115770

def f (x : ℝ) : ℝ := |x| + |x - 3|

theorem inequality_solution_set :
  {x : ℝ | f x - 5 ≥ x} = { x : ℝ | x ≤ -2 / 3 } ∪ { x : ℝ | x ≥ 8 } :=
sorry

theorem compare_mn_and_2m_plus_2n (m n : ℝ) (hm : ∃ x, m = f x) (hn : ∃ x, n = f x) :
  2 * (m + n) < m * n + 4 :=
sorry

end inequality_solution_set_compare_mn_and_2m_plus_2n_l115_115770


namespace solve_for_x_l115_115451

theorem solve_for_x (x : ℝ) : (x - 55) / 3 = (2 - 3*x + x^2) / 4 → (x = 20 / 3 ∨ x = -11) :=
by
  intro h
  sorry

end solve_for_x_l115_115451


namespace james_running_increase_l115_115573

theorem james_running_increase (initial_miles_per_week : ℕ) (percent_increase : ℝ) (total_days : ℕ) (days_in_week : ℕ) :
  initial_miles_per_week = 100 →
  percent_increase = 0.2 →
  total_days = 280 →
  days_in_week = 7 →
  ∃ miles_per_week_to_add : ℝ, miles_per_week_to_add = 3 :=
by
  intros
  sorry

end james_running_increase_l115_115573


namespace find_m_l115_115924

-- Given definitions and conditions
def is_ellipse (x y m : ℝ) := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (m : ℝ) := Real.sqrt ((m - 4) / m) = 1 / 2

-- Prove that m = 16 / 3 given the conditions
theorem find_m (m : ℝ) (cond1 : is_ellipse 1 1 m) (cond2 : eccentricity m) (cond3 : m > 4) : m = 16 / 3 :=
by
  sorry

end find_m_l115_115924


namespace expand_expression_l115_115202

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115202


namespace evaluate_expression_l115_115147

theorem evaluate_expression :
  (305^2 - 275^2) / 30 = 580 := 
by
  sorry

end evaluate_expression_l115_115147


namespace tap_filling_time_l115_115303

theorem tap_filling_time (T : ℝ) 
  (h_total : (1 / 3) = (1 / T + 1 / 15 + 1 / 6)) : T = 10 := 
sorry

end tap_filling_time_l115_115303


namespace min_pictures_needed_l115_115290

theorem min_pictures_needed (n m : ℕ) (participants : Fin n → Fin m → Prop)
  (h1 : n = 60) (h2 : m ≤ 30)
  (h3 : ∀ (i j : Fin n), ∃ (k : Fin m), participants i k ∧ participants j k) :
  m = 6 :=
sorry

end min_pictures_needed_l115_115290


namespace angle_triple_complement_l115_115637

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l115_115637


namespace find_x_l115_115247

theorem find_x (P0 P1 P2 P3 P4 P5 : ℝ) (y : ℝ) (h1 : P1 = P0 * 1.10)
                                      (h2 : P2 = P1 * 0.85)
                                      (h3 : P3 = P2 * 1.20)
                                      (h4 : P4 = P3 * (1 - x/100))
                                      (h5 : y = 0.15)
                                      (h6 : P5 = P4 * 1.15)
                                      (h7 : P5 = P0) : x = 23 :=
sorry

end find_x_l115_115247


namespace sum_of_integers_between_60_and_460_ending_in_2_is_10280_l115_115368

-- We define the sequence.
def endsIn2Seq : List Int := List.range' 62 (452 + 1 - 62) 10  -- Generates [62, 72, ..., 452]

-- The sum of the sequence.
def sumEndsIn2Seq : Int := endsIn2Seq.sum

-- The theorem to prove the desired sum.
theorem sum_of_integers_between_60_and_460_ending_in_2_is_10280 :
  sumEndsIn2Seq = 10280 := by
  -- Proof is omitted
  sorry

end sum_of_integers_between_60_and_460_ending_in_2_is_10280_l115_115368


namespace angle_measure_triple_complement_l115_115619

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l115_115619


namespace possible_analytical_expression_for_f_l115_115083

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.cos (2 * x))

theorem possible_analytical_expression_for_f :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ x : ℝ, f (x - π/4) = f (-x)) ∧
  (∀ x : ℝ, π/8 < x ∧ x < π/2 → f x < f (x - 1)) :=
by
  sorry

end possible_analytical_expression_for_f_l115_115083


namespace smallest_K_for_triangle_l115_115144

theorem smallest_K_for_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) 
  : ∃ K : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → a + c > b → (a^2 + c^2) / b^2 > K) ∧ K = 1 / 2 :=
by
  sorry

end smallest_K_for_triangle_l115_115144


namespace intersection_of_A_and_B_is_2_l115_115535

-- Define the sets A and B based on the given conditions
def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {2, 3}

-- State the theorem that needs to be proved
theorem intersection_of_A_and_B_is_2 : A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_is_2_l115_115535


namespace expand_polynomial_l115_115178

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l115_115178


namespace average_of_seven_starting_with_d_l115_115821

theorem average_of_seven_starting_with_d (c d : ℕ) (h : d = (c + 3)) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end average_of_seven_starting_with_d_l115_115821


namespace angle_triple_complement_l115_115646

variable {x : ℝ}
-- Condition: Let x be the angle in question
-- Condition: Its complement is 90 - x
-- Condition: The measure of the angle is triple the measure of its complement
axiom triple_complement (h : x = 3 * (90 - x)) : x = 67.5

theorem angle_triple_complement : ∃ x : ℝ, x = 3 * (90 - x) ∧ x = 67.5 :=
by
  use 67.5
  split
  · sorry -- proof that 67.5 = 3 * (90 - 67.5)
  · rfl

end angle_triple_complement_l115_115646


namespace minimum_moves_l115_115058

theorem minimum_moves (n : ℕ) : 
  n > 0 → ∃ k l : ℕ, k + 2 * l ≥ ⌊ (n^2 : ℝ) / 2 ⌋₊ ∧ k + l ≥ ⌊ (n^2 : ℝ) / 3 ⌋₊ :=
by 
  intro hn
  sorry

end minimum_moves_l115_115058


namespace expand_polynomial_eq_l115_115194

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l115_115194


namespace sum_of_midpoints_l115_115295

theorem sum_of_midpoints 
  (a b c d e f : ℝ)
  (h1 : a + b + c = 15)
  (h2 : d + e + f = 15) :
  ((a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15) ∧ 
  ((d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15) :=
by
  sorry

end sum_of_midpoints_l115_115295


namespace quadratic_discriminant_l115_115702

theorem quadratic_discriminant : 
  let a := 4
  let b := -6
  let c := 9
  (b^2 - 4 * a * c = -108) := 
by
  sorry

end quadratic_discriminant_l115_115702


namespace functional_equation_solution_form_l115_115365

noncomputable def functional_equation_problem (f : ℝ → ℝ) :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem functional_equation_solution_form :
  (∀ f : ℝ → ℝ, (functional_equation_problem f) → (∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ 2 + b * x)) :=
by 
  sorry

end functional_equation_solution_form_l115_115365


namespace volume_of_prism_l115_115452

-- Define the conditions
variables {a b c : ℝ}
-- Areas of the faces
def ab := 50
def ac := 72
def bc := 45

-- Theorem stating the volume of the prism
theorem volume_of_prism : a * b * c = 180 * Real.sqrt 5 :=
by
  sorry

end volume_of_prism_l115_115452


namespace numbers_sum_and_difference_l115_115095

variables (a b : ℝ)

theorem numbers_sum_and_difference (h : a / b = -1) : a + b = 0 ∧ (a - b = 2 * b ∨ a - b = -2 * b) :=
by {
  sorry
}

end numbers_sum_and_difference_l115_115095


namespace cost_of_second_type_of_rice_is_22_l115_115972

noncomputable def cost_second_type_of_rice (c1 : ℝ) (w1 : ℝ) (w2 : ℝ) (avg : ℝ) (total_weight : ℝ) : ℝ :=
  ((total_weight * avg) - (w1 * c1)) / w2

theorem cost_of_second_type_of_rice_is_22 :
  cost_second_type_of_rice 16 8 4 18 12 = 22 :=
by
  sorry

end cost_of_second_type_of_rice_is_22_l115_115972


namespace lily_sees_leo_l115_115815

theorem lily_sees_leo : 
  ∀ (d₁ d₂ v₁ v₂ : ℝ), 
  d₁ = 0.75 → 
  d₂ = 0.75 → 
  v₁ = 15 → 
  v₂ = 9 → 
  (d₁ + d₂) / (v₁ - v₂) * 60 = 15 :=
by 
  intros d₁ d₂ v₁ v₂ h₁ h₂ h₃ h₄
  -- skipping the proof with sorry
  sorry

end lily_sees_leo_l115_115815


namespace outfits_count_l115_115479

-- Definitions of the counts of each type of clothing item
def num_blue_shirts : Nat := 6
def num_green_shirts : Nat := 4
def num_pants : Nat := 7
def num_blue_hats : Nat := 9
def num_green_hats : Nat := 7

-- Statement of the problem to prove
theorem outfits_count :
  (num_blue_shirts * num_pants * num_green_hats) + (num_green_shirts * num_pants * num_blue_hats) = 546 :=
by
  sorry

end outfits_count_l115_115479


namespace gingerbread_price_today_is_5_l115_115318

-- Given conditions
variables {x y a b k m : ℤ}

-- Price constraints
axiom price_constraint_yesterday : 9 * x + 7 * y < 100
axiom price_constraint_today1 : 9 * a + 7 * b > 100
axiom price_constraint_today2 : 2 * a + 11 * b < 100

-- Price change constraints
axiom price_change_gingerbread : a = x + k
axiom price_change_pastries : b = y + m
axiom gingerbread_change_range : |k| ≤ 1
axiom pastries_change_range : |m| ≤ 1

theorem gingerbread_price_today_is_5 : a = 5 :=
by
  sorry

end gingerbread_price_today_is_5_l115_115318


namespace exists_pos_int_such_sqrt_not_int_l115_115820

theorem exists_pos_int_such_sqrt_not_int (a b c : ℤ) : ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, k * k = n^3 + a * n^2 + b * n + c :=
by
  sorry

end exists_pos_int_such_sqrt_not_int_l115_115820


namespace union_sets_l115_115230

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} := by
  sorry

end union_sets_l115_115230


namespace smallest_equal_cost_l115_115471

def decimal_cost (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_cost (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem smallest_equal_cost :
  ∃ n : ℕ, n < 200 ∧ decimal_cost n = binary_cost n ∧ (∀ m : ℕ, m < 200 ∧ decimal_cost m = binary_cost m → m ≥ n) :=
by
  -- Proof goes here
  sorry

end smallest_equal_cost_l115_115471


namespace number_of_ordered_pairs_l115_115123

theorem number_of_ordered_pairs (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / 24)) : 
  ∃ n : ℕ, n = 41 :=
by
  sorry

end number_of_ordered_pairs_l115_115123


namespace angle_measure_triple_complement_l115_115629

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l115_115629


namespace angle_triple_complement_l115_115635

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l115_115635


namespace integer_satisfaction_l115_115844

theorem integer_satisfaction (x : ℤ) : 
  (x + 15 ≥ 16 ∧ -3 * x ≥ -15) ↔ (1 ≤ x ∧ x ≤ 5) :=
by 
  sorry

end integer_satisfaction_l115_115844


namespace Jack_hands_in_l115_115415

def num_hundred_bills := 2
def num_fifty_bills := 1
def num_twenty_bills := 5
def num_ten_bills := 3
def num_five_bills := 7
def num_one_bills := 27
def to_leave_in_till := 300

def total_money_in_notes : Nat :=
  (num_hundred_bills * 100) +
  (num_fifty_bills * 50) +
  (num_twenty_bills * 20) +
  (num_ten_bills * 10) +
  (num_five_bills * 5) +
  (num_one_bills * 1)

def money_to_hand_in := total_money_in_notes - to_leave_in_till

theorem Jack_hands_in : money_to_hand_in = 142 := by
  sorry

end Jack_hands_in_l115_115415


namespace brendan_cuts_84_yards_in_week_with_lawnmower_l115_115345

-- Brendan cuts 8 yards per day
def yards_per_day : ℕ := 8

-- The lawnmower increases his efficiency by fifty percent
def efficiency_increase (yards : ℕ) : ℕ :=
  yards + (yards / 2)

-- Calculate total yards cut in 7 days with the lawnmower
def total_yards_in_week (days : ℕ) (daily_yards : ℕ) : ℕ :=
  days * daily_yards

-- Prove the total yards cut in 7 days with the lawnmower is 84
theorem brendan_cuts_84_yards_in_week_with_lawnmower :
  total_yards_in_week 7 (efficiency_increase yards_per_day) = 84 :=
by
  sorry

end brendan_cuts_84_yards_in_week_with_lawnmower_l115_115345


namespace angle_measure_triple_complement_l115_115625

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l115_115625


namespace probability_king_then_ten_l115_115374

-- Define the conditions
def standard_deck_size : ℕ := 52
def num_kings : ℕ := 4
def num_tens : ℕ := 4

-- Define the event probabilities
def prob_first_card_king : ℚ := num_kings / standard_deck_size
def prob_second_card_ten (remaining_deck_size : ℕ) : ℚ := num_tens / remaining_deck_size

-- The theorem statement to be proved
theorem probability_king_then_ten : 
  prob_first_card_king * prob_second_card_ten (standard_deck_size - 1) = 4 / 663 :=
by
  sorry

end probability_king_then_ten_l115_115374


namespace symmetry_proof_l115_115569

-- Define the initial point P and its reflection P' about the x-axis
def P : ℝ × ℝ := (-1, 2)
def P' : ℝ × ℝ := (-1, -2)

-- Define the property of symmetry about the x-axis
def symmetric_about_x_axis (P P' : ℝ × ℝ) : Prop :=
  P'.fst = P.fst ∧ P'.snd = -P.snd

-- The theorem to prove that point P' is symmetric to point P about the x-axis
theorem symmetry_proof : symmetric_about_x_axis P P' :=
  sorry

end symmetry_proof_l115_115569


namespace circle_standard_equation_l115_115957

theorem circle_standard_equation (x y : ℝ) :
  let center_x := 2
  let center_y := -1
  let radius := 3
  (center_x = 2) ∧ (center_y = -1) ∧ (radius = 3) → (x - center_x) ^ 2 + (y - center_y) ^ 2 = radius ^ 2 :=
by
  intros
  sorry

end circle_standard_equation_l115_115957


namespace eulers_formula_l115_115819

-- Definitions related to simply connected polyhedra
def SimplyConnectedPolyhedron (V E F : ℕ) : Prop := true  -- Genus 0 implies it is simply connected

-- Euler's characteristic property for simply connected polyhedra
theorem eulers_formula (V E F : ℕ) (h : SimplyConnectedPolyhedron V E F) : V - E + F = 2 := 
by
  sorry

end eulers_formula_l115_115819


namespace expand_polynomial_eq_l115_115191

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l115_115191


namespace city_rentals_cost_per_mile_l115_115735

theorem city_rentals_cost_per_mile (x : ℝ)
  (h₁ : 38.95 + 150 * x = 41.95 + 150 * 0.29) :
  x = 0.31 :=
by sorry

end city_rentals_cost_per_mile_l115_115735


namespace arithmetic_problem_l115_115834

theorem arithmetic_problem : 987 + 113 - 1000 = 100 :=
by
  sorry

end arithmetic_problem_l115_115834


namespace term_addition_k_to_kplus1_l115_115842

theorem term_addition_k_to_kplus1 (k : ℕ) : 
  (2 * k + 2) + (2 * k + 3) = 4 * k + 5 := 
sorry

end term_addition_k_to_kplus1_l115_115842


namespace find_num_managers_l115_115332

variable (num_associates : ℕ) (avg_salary_managers avg_salary_associates avg_salary_company : ℚ)
variable (num_managers : ℚ)

-- Define conditions based on given problem
def conditions := 
  num_associates = 75 ∧
  avg_salary_managers = 90000 ∧
  avg_salary_associates = 30000 ∧
  avg_salary_company = 40000

-- Proof problem statement
theorem find_num_managers (h : conditions num_associates avg_salary_managers avg_salary_associates avg_salary_company) :
  num_managers = 15 :=
sorry

end find_num_managers_l115_115332


namespace angle_AST_eq_90_l115_115578

open Real Angle

theorem angle_AST_eq_90 
  (ABC : Triangle)
  (h₁ : ABC.isAcute)
  (h₂ : ABC.circumradius = R)
  (D : Point)
  (h₃ : isFootOfAltitude D ABC.A ABC.B ABC.C)
  (T : Point)
  (h₄ : T ∈ Line AD)
  (h₅ : distance A T = 2 * R)
  (h₆ : isBetween D A T)
  (S : Point)
  (h₇ : S = midpointArcBCnotA ABC)
  : ∠ A S T = 90° :=
by
  sorry

end angle_AST_eq_90_l115_115578


namespace expand_expression_l115_115183

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115183


namespace Bill_Sunday_miles_l115_115075

-- Definitions based on problem conditions
def Bill_Saturday (B : ℕ) : ℕ := B
def Bill_Sunday (B : ℕ) : ℕ := B + 4
def Julia_Sunday (B : ℕ) : ℕ := 2 * (B + 4)
def Alex_Total (B : ℕ) : ℕ := B + 2

-- Total miles equation based on conditions
def total_miles (B : ℕ) : ℕ := Bill_Saturday B + Bill_Sunday B + Julia_Sunday B + Alex_Total B

-- Proof statement
theorem Bill_Sunday_miles (B : ℕ) (h : total_miles B = 54) : Bill_Sunday B = 14 :=
by {
  -- calculations and proof would go here if not omitted
  sorry
}

end Bill_Sunday_miles_l115_115075


namespace ms_lee_class_difference_l115_115246

noncomputable def boys_and_girls_difference (ratio_b : ℕ) (ratio_g : ℕ) (total_students : ℕ) : ℕ :=
  let x := total_students / (ratio_b + ratio_g)
  let boys := ratio_b * x
  let girls := ratio_g * x
  girls - boys

theorem ms_lee_class_difference :
  boys_and_girls_difference 3 4 42 = 6 :=
by
  sorry

end ms_lee_class_difference_l115_115246


namespace prove_moles_of_C2H6_l115_115208

def moles_of_CCl4 := 4
def moles_of_Cl2 := 14
def moles_of_C2H6 := 2

theorem prove_moles_of_C2H6
  (h1 : moles_of_Cl2 = 14)
  (h2 : moles_of_CCl4 = 4)
  : moles_of_C2H6 = 2 := 
sorry

end prove_moles_of_C2H6_l115_115208


namespace gain_percent_l115_115913

theorem gain_percent (C S S_d : ℝ) 
  (h1 : 50 * C = 20 * S) 
  (h2 : S_d = S * (1 - 0.15)) : 
  ((S_d - C) / C) * 100 = 112.5 := 
by 
  sorry

end gain_percent_l115_115913


namespace union_A_B_equals_x_lt_3_l115_115930

theorem union_A_B_equals_x_lt_3 :
  let A := { x : ℝ | 3 - x > 0 ∧ x + 2 > 0 }
  let B := { x : ℝ | 3 > 2*x - 1 }
  A ∪ B = { x : ℝ | x < 3 } :=
by
  sorry

end union_A_B_equals_x_lt_3_l115_115930


namespace sallys_change_l115_115589

-- Given conditions
def frames_bought : ℕ := 3
def cost_per_frame : ℕ := 3
def payment : ℕ := 20

-- The statement to prove
theorem sallys_change : payment - (frames_bought * cost_per_frame) = 11 := by
  sorry

end sallys_change_l115_115589


namespace min_value_expression_l115_115932

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (min (a^4 + b^4 + (1 / (a + b)^4)) (a' b' : ℝ) : ((0 < a') ∧ (0 < b'))) = (√2 / 2) :=
sorry

end min_value_expression_l115_115932


namespace probability_of_both_events_l115_115325

theorem probability_of_both_events (P_A P_B P_union : ℝ) (hA : P_A = 0.20) (hB : P_B = 0.30) (h_union : P_union = 0.35) : 
  P_A + P_B - P_union = 0.15 :=
by
  simp [hA, hB, h_union]
  sorry

end probability_of_both_events_l115_115325


namespace polygon_sides_eq_eight_l115_115315

theorem polygon_sides_eq_eight (x : ℕ) (h : x ≥ 3) 
  (h1 : 2 * (x - 2) = 180 * (x - 2) / 90) 
  (h2 : ∀ x, x + 2 * (x - 2) = x * (x - 3) / 2) : 
  x = 8 :=
by
  sorry

end polygon_sides_eq_eight_l115_115315


namespace intersection_A_B_l115_115540

open Set

-- Conditions given in the problem
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Statement to prove, no proof needed
theorem intersection_A_B : A ∩ B = {1, 2} := 
sorry

end intersection_A_B_l115_115540


namespace qingyang_2015_mock_exam_l115_115495

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def problem :=
  U = {1, 2, 3, 4, 5} ∧ A = {2, 3, 4} ∧ B = {2, 5} →
  B ∪ (U \ A) = {1, 2, 5}

theorem qingyang_2015_mock_exam (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) : problem U A B :=
by
  intros
  sorry

end qingyang_2015_mock_exam_l115_115495


namespace arithmetic_sequence_50th_term_l115_115528

theorem arithmetic_sequence_50th_term :
  let a1 := 3
  let d := 2
  let n := 50
  let a_n := a1 + (n - 1) * d
  a_n = 101 :=
by
  sorry

end arithmetic_sequence_50th_term_l115_115528


namespace cosA_sinB_value_l115_115592

theorem cosA_sinB_value (A B : ℝ) (hA1 : 0 < A ∧ A < π / 2) (hB1 : 0 < B ∧ B < π / 2)
  (h_tan_eq : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = Real.sqrt 6 / 6 := sorry

end cosA_sinB_value_l115_115592


namespace reading_speed_increase_factor_l115_115351

-- Define Tom's normal reading speed as a constant rate
def tom_normal_speed := 12 -- pages per hour

-- Define the time period
def hours := 2 -- hours

-- Define the number of pages read in the given time period
def pages_read := 72 -- pages

-- Calculate the expected pages read at normal speed in the given time
def expected_pages := tom_normal_speed * hours -- should be 24 pages

-- Define the calculated factor by which the reading speed has increased
def expected_factor := pages_read / expected_pages -- should be 3

-- Prove that the factor is indeed 3
theorem reading_speed_increase_factor :
  expected_factor = 3 := by
  sorry

end reading_speed_increase_factor_l115_115351


namespace jack_kids_solution_l115_115794

def jack_kids (k : ℕ) : Prop :=
  7 * 3 * k = 63

theorem jack_kids_solution : jack_kids 3 :=
by
  sorry

end jack_kids_solution_l115_115794


namespace bobby_candy_left_l115_115730

theorem bobby_candy_left (initial_candies := 21) (first_eaten := 5) (second_eaten := 9) : 
  initial_candies - first_eaten - second_eaten = 7 :=
by
  -- Proof goes here
  sorry

end bobby_candy_left_l115_115730


namespace inequality_solution_set_l115_115293

theorem inequality_solution_set (x : ℝ) : 3 ≤ abs (5 - 2 * x) ∧ abs (5 - 2 * x) < 9 ↔ (x > -2 ∧ x ≤ 1) ∨ (x ≥ 4 ∧ x < 7) := sorry

end inequality_solution_set_l115_115293


namespace arithmetic_progression_ratio_l115_115738

variable {α : Type*} [LinearOrder α] [Field α]

theorem arithmetic_progression_ratio (a d : α) (h : 15 * a + 105 * d = 3 * (8 * a + 28 * d)) : a / d = 7 / 3 := 
by sorry

end arithmetic_progression_ratio_l115_115738


namespace melanie_total_dimes_l115_115938

/-- Melanie had 7 dimes in her bank. Her dad gave her 8 dimes. Her mother gave her 4 dimes. -/
def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

/-- How many dimes does Melanie have now? -/
theorem melanie_total_dimes : initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end melanie_total_dimes_l115_115938


namespace max_value_of_f_l115_115052

noncomputable def f (x a b : ℝ) := (1 - x ^ 2) * (x ^ 2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (x : ℝ) 
  (h1 : (∀ x, f x 8 15 = (1 - x^2) * (x^2 + 8*x + 15)))
  (h2 : ∀ x, f x a b = f (-(x + 4)) a b) :
  ∃ m, (∀ x, f x 8 15 ≤ m) ∧ m = 16 :=
by
  sorry

end max_value_of_f_l115_115052


namespace meaning_of_sum_of_squares_l115_115089

theorem meaning_of_sum_of_squares (a b : ℝ) : a ^ 2 + b ^ 2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end meaning_of_sum_of_squares_l115_115089


namespace expand_expression_l115_115151

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115151


namespace lesser_number_is_32_l115_115459

variable (x y : ℕ)

theorem lesser_number_is_32 (h1 : y = 2 * x) (h2 : x + y = 96) : x = 32 := 
sorry

end lesser_number_is_32_l115_115459


namespace additional_boys_l115_115323

theorem additional_boys (initial additional total : ℕ) 
  (h1 : initial = 22) 
  (h2 : total = 35) 
  (h3 : additional = total - initial) : 
  additional = 13 := 
by 
  rw [h1, h2]
  simp
  sorry

end additional_boys_l115_115323


namespace total_animals_l115_115054

-- Define the number of pigs and giraffes
def num_pigs : ℕ := 7
def num_giraffes : ℕ := 6

-- Theorem stating the total number of giraffes and pigs
theorem total_animals : num_pigs + num_giraffes = 13 :=
by sorry

end total_animals_l115_115054


namespace sin_cos_product_l115_115554

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l115_115554


namespace shaded_area_l115_115791

theorem shaded_area (r : ℝ) (π : ℝ) (shaded_area : ℝ) (h_r : r = 4) (h_π : π = 3) : shaded_area = 32.5 :=
by
  sorry

end shaded_area_l115_115791


namespace fourth_equation_l115_115583

theorem fourth_equation :
  (5 * 6 * 7 * 8) = (2^4) * 1 * 3 * 5 * 7 :=
by
  sorry

end fourth_equation_l115_115583


namespace largest_integer_x_l115_115207

theorem largest_integer_x (x : ℤ) :
  (x ^ 2 - 11 * x + 28 < 0) → x ≤ 6 := sorry

end largest_integer_x_l115_115207


namespace value_of_x_pow_12_l115_115235

theorem value_of_x_pow_12 (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^12 = 439 := sorry

end value_of_x_pow_12_l115_115235


namespace sequence_12th_term_l115_115395

theorem sequence_12th_term (C : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = C / a n) (h4 : C = 12) : a 12 = 4 :=
sorry

end sequence_12th_term_l115_115395


namespace time_to_cover_length_l115_115006

/-- Define the conditions for the problem -/
def angle_deg : ℝ := 30
def escalator_speed : ℝ := 12
def length_along_incline : ℝ := 160
def person_speed : ℝ := 8

/-- Define the combined speed as the sum of the escalator speed and the person speed -/
def combined_speed : ℝ := escalator_speed + person_speed

/-- Theorem stating the time taken to cover the length of the escalator is 8 seconds -/
theorem time_to_cover_length : (length_along_incline / combined_speed) = 8 := by
  sorry

end time_to_cover_length_l115_115006


namespace expand_expression_l115_115168

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l115_115168


namespace find_positive_integers_n_satisfying_equation_l115_115205

theorem find_positive_integers_n_satisfying_equation :
  ∀ x y z : ℕ,
  x > 0 → y > 0 → z > 0 →
  (x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) →
  (n = 1 ∨ n = 3) :=
by
  sorry

end find_positive_integers_n_satisfying_equation_l115_115205


namespace lambda_is_half_l115_115045

   theorem lambda_is_half
     (a b c : ℝ × ℝ)
     (λ : ℝ)
     (h_a : a = (1, 2))
     (h_b : b = (1, 0))
     (h_c : c = (3, 4))
     (h_parallel : ∃ k : ℝ, a + λ • b = k • c) :
     λ = 1 / 2 := by
   sorry
   
end lambda_is_half_l115_115045


namespace last_three_digits_of_power_l115_115966

theorem last_three_digits_of_power (h : 7^500 ≡ 1 [MOD 1250]) : 7^10000 ≡ 1 [MOD 1250] :=
by
  sorry

end last_three_digits_of_power_l115_115966


namespace angle_measure_triple_complement_l115_115658

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l115_115658


namespace range_of_m_l115_115067

noncomputable def f (x : ℝ) := Real.exp x * (x - 1)
noncomputable def g (m x : ℝ) := m * x

theorem range_of_m :
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (1 : ℝ) 2, f x₁ > g m x₂) ↔ m ∈ Set.Iio (-1/2 : ℝ) :=
sorry

end range_of_m_l115_115067


namespace shopkeeper_percentage_gain_l115_115500

theorem shopkeeper_percentage_gain (false_weight true_weight : ℝ) 
    (h_false_weight : false_weight = 930)
    (h_true_weight : true_weight = 1000) : 
    (true_weight - false_weight) / false_weight * 100 = 7.53 := 
by
  rw [h_false_weight, h_true_weight]
  sorry

end shopkeeper_percentage_gain_l115_115500


namespace tensor_identity_l115_115143

namespace tensor_problem

def otimes (x y : ℝ) : ℝ := x^2 + y

theorem tensor_identity (a : ℝ) : otimes a (otimes a a) = 2 * a^2 + a :=
by sorry

end tensor_problem

end tensor_identity_l115_115143


namespace maximized_area_using_squares_l115_115023

theorem maximized_area_using_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
  by sorry

end maximized_area_using_squares_l115_115023


namespace shortest_side_of_similar_triangle_l115_115338

theorem shortest_side_of_similar_triangle (a1 a2 h1 h2 : ℝ)
  (h1_eq : a1 = 24)
  (h2_eq : h1 = 37)
  (h2_eq' : h2 = 74)
  (h_similar : h2 / h1 = 2)
  (h_a2_eq : a2 = 2 * Real.sqrt 793):
  a2 = 2 * Real.sqrt 793 := by
  sorry

end shortest_side_of_similar_triangle_l115_115338


namespace expand_polynomial_l115_115175

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l115_115175


namespace arithmetic_sequence_inequality_l115_115752

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality
  (a : ℕ → α) (d : α)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_d_pos : d > 0)
  (n : ℕ)
  (h_n_gt_1 : n > 1) :
  a 1 * a (n + 1) < a 2 * a n := 
sorry

end arithmetic_sequence_inequality_l115_115752


namespace calculate_x_l115_115486

theorem calculate_x :
  (422 + 404) ^ 2 - (4 * 422 * 404) = 324 :=
by
  -- proof goes here
  sorry

end calculate_x_l115_115486


namespace sin_double_angle_l115_115898

-- Given Conditions
variable {α : ℝ}
variable (h1 : 0 < α ∧ α < π / 2) -- α is in the first quadrant
variable (h2 : Real.sin α = 3 / 5) -- sin(α) = 3/5

-- Theorem statement
theorem sin_double_angle (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 := 
sorry

end sin_double_angle_l115_115898


namespace initial_percentage_increase_l115_115466

-- Given conditions
def S_original : ℝ := 4000.0000000000005
def S_final : ℝ := 4180
def reduction : ℝ := 5

-- Predicate to prove the initial percentage increase is 10%
theorem initial_percentage_increase (x : ℝ) 
  (hx : (95/100) * (S_original * (1 + x / 100)) = S_final) : 
  x = 10 :=
sorry

end initial_percentage_increase_l115_115466


namespace greg_total_earnings_l115_115906

-- Define the charges and walking times as given
def charge_per_dog : ℕ := 20
def charge_per_minute : ℕ := 1
def one_dog_minutes : ℕ := 10
def two_dogs_minutes : ℕ := 7
def three_dogs_minutes : ℕ := 9
def total_dogs_one : ℕ := 1
def total_dogs_two : ℕ := 2
def total_dogs_three : ℕ := 3

-- Total earnings computation
def earnings_one_dog : ℕ := charge_per_dog + charge_per_minute * one_dog_minutes
def earnings_two_dogs : ℕ := total_dogs_two * charge_per_dog + total_dogs_two * charge_per_minute * two_dogs_minutes
def earnings_three_dogs : ℕ := total_dogs_three * charge_per_dog + total_dogs_three * charge_per_minute * three_dogs_minutes
def total_earnings : ℕ := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

-- The proof: Greg's total earnings should be $171
theorem greg_total_earnings : total_earnings = 171 := by
  -- Placeholder for the proof (not required as per the instructions)
  sorry

end greg_total_earnings_l115_115906


namespace angle_measure_l115_115687

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l115_115687


namespace increasing_sequences_mod_condition_l115_115509

theorem increasing_sequences_mod_condition :
  let sequences := { s : Finset (Fin 1007) // all (λ i, s i - i ∈ {0, 3, ..., 1002}) (Finset.range 5)}
  sequences.card = Nat.choose 338 5
  ∧ 338 % 500 = 338 :=
by
  sorry

end increasing_sequences_mod_condition_l115_115509


namespace find_n_eq_130_l115_115593

theorem find_n_eq_130 
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : 0 < n)
  (h2 : d1 < d2)
  (h3 : d2 < d3)
  (h4 : d3 < d4)
  (h5 : ∀ d, d ∣ n → d = d1 ∨ d = d2 ∨ d = d3 ∨ d = d4 ∨ d ∣ n → ¬(1 < d ∧ d < d1))
  (h6 : n = d1^2 + d2^2 + d3^2 + d4^2) : n = 130 := 
  sorry

end find_n_eq_130_l115_115593


namespace arithmetic_sequence_general_term_absolute_sum_first_19_terms_l115_115042

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) :
  ∀ n : ℕ, a n = 28 - 3 * n := 
sorry

theorem absolute_sum_first_19_terms (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) (an_eq : ∀ n : ℕ, a n = 28 - 3 * n) :
  |a 1| + |a 3| + |a 5| + |a 7| + |a 9| + |a 11| + |a 13| + |a 15| + |a 17| + |a 19| = 150 := 
sorry

end arithmetic_sequence_general_term_absolute_sum_first_19_terms_l115_115042


namespace division_multiplication_relation_l115_115344

theorem division_multiplication_relation (h: 7650 / 306 = 25) :
  25 * 306 = 7650 ∧ 7650 / 25 = 306 := 
by 
  sorry

end division_multiplication_relation_l115_115344


namespace cos_value_given_sin_l115_115899

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (π / 3 - α) = 3 / 5 :=
sorry

end cos_value_given_sin_l115_115899


namespace books_remaining_correct_l115_115610

-- Define the initial number of book donations
def initial_books : ℕ := 300

-- Define the number of people donating and the number of books each donates
def num_people : ℕ := 10
def books_per_person : ℕ := 5

-- Calculate total books donated by all people
def total_donation : ℕ := num_people * books_per_person

-- Define the number of books borrowed by other people
def borrowed_books : ℕ := 140

-- Calculate the total number of books after donations and then subtract the borrowed books
def total_books_remaining : ℕ := initial_books + total_donation - borrowed_books

-- Prove the total number of books remaining is 210
theorem books_remaining_correct : total_books_remaining = 210 := by
  sorry

end books_remaining_correct_l115_115610


namespace globe_division_l115_115984

theorem globe_division (parallels meridians : ℕ) (h_parallels : parallels = 17) (h_meridians : meridians = 24) : 
  (meridians * (parallels + 1)) = 432 :=
by
  rw [h_parallels, h_meridians]
  simp
  sorry

end globe_division_l115_115984


namespace order_of_scores_l115_115743

variables (E L T N : ℝ)

-- Conditions
axiom Lana_condition_1 : L ≠ T
axiom Lana_condition_2 : L ≠ N
axiom Lana_condition_3 : T ≠ N

axiom Tom_condition : ∃ L' E', L' ≠ T ∧ E' > L' ∧ E' ≠ T ∧ E' ≠ L' 

axiom Nina_condition : N < E

-- Proof statement
theorem order_of_scores :
  N < L ∧ L < T :=
sorry

end order_of_scores_l115_115743


namespace no_perfect_squares_l115_115031

theorem no_perfect_squares (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0)
  (h5 : x * y - z * t = x + y) (h6 : x + y = z + t) : ¬(∃ a b : ℕ, a^2 = x * y ∧ b^2 = z * t) := 
by
  sorry

end no_perfect_squares_l115_115031


namespace merchant_marked_price_l115_115864

variable (L C M S : ℝ)

theorem merchant_marked_price :
  (C = 0.8 * L) → (C = 0.8 * S) → (S = 0.8 * M) → (M = 1.25 * L) :=
by
  sorry

end merchant_marked_price_l115_115864


namespace lcm_24_36_45_l115_115706

open Nat

theorem lcm_24_36_45 : lcm (lcm 24 36) 45 = 360 := by sorry

end lcm_24_36_45_l115_115706


namespace value_range_for_inequality_solution_set_l115_115393

-- Define the condition
def condition (a : ℝ) : Prop := a > 0

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |x - 3| < a

-- State the theorem to be proven
theorem value_range_for_inequality_solution_set (a : ℝ) (h: condition a) : (a > 1) ↔ ∃ x : ℝ, inequality x a := 
sorry

end value_range_for_inequality_solution_set_l115_115393


namespace expand_expression_l115_115198

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115198


namespace calculate_a_minus_b_l115_115298

theorem calculate_a_minus_b (a b c : ℝ) (h1 : a - b - c = 3) (h2 : a - b + c = 11) : a - b = 7 :=
by 
  -- The proof would be fleshed out here.
  sorry

end calculate_a_minus_b_l115_115298


namespace angle_measure_triple_complement_l115_115628

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_measure_triple_complement_l115_115628


namespace restaurant_total_tables_l115_115724

theorem restaurant_total_tables (N O : ℕ) (h1 : 6 * N + 4 * O = 212) (h2 : N = O + 12) : N + O = 40 :=
sorry

end restaurant_total_tables_l115_115724


namespace total_money_l115_115987

theorem total_money (gold_value silver_value cash : ℕ) (num_gold num_silver : ℕ)
  (h_gold : gold_value = 50)
  (h_silver : silver_value = 25)
  (h_num_gold : num_gold = 3)
  (h_num_silver : num_silver = 5)
  (h_cash : cash = 30) :
  gold_value * num_gold + silver_value * num_silver + cash = 305 :=
by
  sorry

end total_money_l115_115987


namespace percentage_yield_l115_115498

theorem percentage_yield (market_price annual_dividend : ℝ) (yield : ℝ) 
  (H1 : yield = 0.12)
  (H2 : market_price = 125)
  (H3 : annual_dividend = yield * market_price) :
  (annual_dividend / market_price) * 100 = 12 := 
sorry

end percentage_yield_l115_115498


namespace angle_triple_complement_l115_115666

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l115_115666


namespace min_combined_horses_and_ponies_l115_115484

theorem min_combined_horses_and_ponies : 
  ∀ (P : ℕ), 
  (∃ (P' : ℕ), 
    (P = P' ∧ (∃ (x : ℕ), x = 3 * P' / 10 ∧ x = 3 * P' / 16) ∧
     (∃ (y : ℕ), y = 5 * x / 8) ∧ 
      ∀ (H : ℕ), (H = 3 + P')) → 
  P + (3 + P) = 35) := 
sorry

end min_combined_horses_and_ponies_l115_115484


namespace expand_polynomial_l115_115179

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l115_115179


namespace parameter_conditions_l115_115882

theorem parameter_conditions (p x y : ℝ) :
  (x - p)^2 = 16 * (y - 3 + p) →
  y^2 + ((x - 3) / (|x| - 3))^2 = 1 →
  |x| ≠ 3 →
  p > 3 ∧ 
  ((p ≤ 4 ∨ p ≥ 12) ∧ (p < 19 ∨ 19 < p)) :=
sorry

end parameter_conditions_l115_115882


namespace percentage_of_Indian_women_l115_115567

-- Definitions of conditions
def total_people := 700 + 500 + 800
def indian_men := (20 / 100) * 700
def indian_children := (10 / 100) * 800
def total_indian_people := (21 / 100) * total_people
def indian_women := total_indian_people - indian_men - indian_children

-- Statement of the theorem
theorem percentage_of_Indian_women : 
  (indian_women / 500) * 100 = 40 :=
by
  sorry

end percentage_of_Indian_women_l115_115567


namespace centroid_sum_of_squares_l115_115803

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end centroid_sum_of_squares_l115_115803


namespace expand_expression_l115_115153

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115153


namespace speed_of_train_is_20_l115_115117

def length_of_train := 120 -- in meters
def time_to_cross := 6 -- in seconds

def speed_of_train := length_of_train / time_to_cross -- Speed formula

theorem speed_of_train_is_20 :
  speed_of_train = 20 := by
  sorry

end speed_of_train_is_20_l115_115117


namespace find_x_l115_115392

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ↔ x = 80 / 3 := by
  sorry

end find_x_l115_115392


namespace reimbursement_diff_l115_115609

/-- Let Tom, Emma, and Harry share equally the costs for a group activity.
- Tom paid $95
- Emma paid $140
- Harry paid $165
If Tom and Emma are to reimburse Harry to ensure all expenses are shared equally,
prove that e - t = -45 where e is the amount Emma gives Harry and t is the amount Tom gives Harry.
-/
theorem reimbursement_diff :
  let tom_paid := 95
  let emma_paid := 140
  let harry_paid := 165
  let total_cost := tom_paid + emma_paid + harry_paid
  let equal_share := total_cost / 3
  let t := equal_share - tom_paid
  let e := equal_share - emma_paid
  e - t = -45 :=
by {
  sorry
}

end reimbursement_diff_l115_115609


namespace minimum_value_of_sum_l115_115762

noncomputable def left_focus (a b c : ℝ) : ℝ := -c 

noncomputable def right_focus (a b c : ℝ) : ℝ := c

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
magnitude (q.1 - p.1, q.2 - p.2)

def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1

def P_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_eq P.1 P.2

theorem minimum_value_of_sum (P : ℝ × ℝ) (A : ℝ × ℝ) (F F' : ℝ × ℝ) (a b c : ℝ)
  (h1 : F = (-c, 0)) (h2 : F' = (c, 0)) (h3 : A = (1, 4)) (h4 : 2 * a = 4)
  (h5 : c^2 = a^2 + b^2) (h6 : P_on_hyperbola P) :
  (|distance P F| + |distance P A|) ≥ 9 :=
sorry

end minimum_value_of_sum_l115_115762


namespace total_money_is_305_l115_115988

-- Define the worth of each gold coin, silver coin, and the quantity of each type of coin and cash.
def worth_of_gold_coin := 50
def worth_of_silver_coin := 25
def number_of_gold_coins := 3
def number_of_silver_coins := 5
def cash_in_dollars := 30

-- Define the total money calculation based on given conditions.
def total_gold_value := number_of_gold_coins * worth_of_gold_coin
def total_silver_value := number_of_silver_coins * worth_of_silver_coin
def total_value := total_gold_value + total_silver_value + cash_in_dollars

-- The proof statement asserting the total value.
theorem total_money_is_305 : total_value = 305 := by
  -- Proof omitted for brevity.
  sorry

end total_money_is_305_l115_115988


namespace decagon_area_bisection_ratio_l115_115597

theorem decagon_area_bisection_ratio
  (decagon_area : ℝ := 12)
  (below_PQ_area : ℝ := 6)
  (trapezoid_area : ℝ := 4)
  (b1 : ℝ := 3)
  (b2 : ℝ := 6)
  (h : ℝ := 8/9)
  (XQ : ℝ := 4)
  (QY : ℝ := 2) :
  (XQ / QY = 2) :=
by
  sorry

end decagon_area_bisection_ratio_l115_115597


namespace determine_c_l115_115520

theorem determine_c (c y : ℝ) : (∀ y : ℝ, 3 * (3 + 2 * c * y) = 18 * y + 9) → c = 3 := by
  sorry

end determine_c_l115_115520


namespace ellipse_foci_distance_l115_115513

-- Definitions based on the problem conditions
def ellipse_eq (x y : ℝ) :=
  Real.sqrt (((x - 4)^2) + ((y - 5)^2)) + Real.sqrt (((x + 6)^2) + ((y + 9)^2)) = 22

def focus1 : (ℝ × ℝ) := (4, -5)
def focus2 : (ℝ × ℝ) := (-6, 9)

-- Statement of the problem
noncomputable def distance_between_foci : ℝ :=
  Real.sqrt (((focus1.1 + 6)^2) + ((focus1.2 - 9)^2))

-- Proof statement
theorem ellipse_foci_distance : distance_between_foci = 2 * Real.sqrt 74 := by
  sorry

end ellipse_foci_distance_l115_115513


namespace total_sum_lent_l115_115506

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ)
  (h1 : second_part = 1648)
  (h2 : (x * 3 / 100 * 8) = (second_part * 5 / 100 * 3))
  (h3 : total_sum = x + second_part) :
  total_sum = 2678 := 
  sorry

end total_sum_lent_l115_115506


namespace audrey_dreaming_fraction_l115_115729

theorem audrey_dreaming_fraction
  (total_asleep_time : ℕ) 
  (not_dreaming_time : ℕ)
  (dreaming_time : ℕ)
  (fraction_dreaming : ℚ)
  (h_total_asleep : total_asleep_time = 10)
  (h_not_dreaming : not_dreaming_time = 6)
  (h_dreaming : dreaming_time = total_asleep_time - not_dreaming_time)
  (h_fraction : fraction_dreaming = dreaming_time / total_asleep_time) :
  fraction_dreaming = 2 / 5 := 
by {
  sorry
}

end audrey_dreaming_fraction_l115_115729


namespace expand_product_l115_115160

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l115_115160


namespace lcm_24_36_45_l115_115705

noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem lcm_24_36_45 : lcm_three 24 36 45 = 360 := by
  -- Conditions involves proving the prime factorizations
  have h1 : Nat.factors 24 = [2, 2, 2, 3] := by sorry -- Prime factorization of 24
  have h2 : Nat.factors 36 = [2, 2, 3, 3] := by sorry -- Prime factorization of 36
  have h3 : Nat.factors 45 = [3, 3, 5] := by sorry -- Prime factorization of 45

  -- Least common multiple calculation based on the greatest powers of prime factors
  sorry -- This is where the proof would go

end lcm_24_36_45_l115_115705


namespace angle_triple_complement_l115_115674

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l115_115674


namespace david_pushups_difference_l115_115480

-- Definitions based on conditions
def zachary_pushups : ℕ := 44
def total_pushups : ℕ := 146

-- The number of push-ups David did more than Zachary
def david_more_pushups_than_zachary (D : ℕ) := D - zachary_pushups

-- The theorem we need to prove
theorem david_pushups_difference :
  ∃ D : ℕ, D > zachary_pushups ∧ D + zachary_pushups = total_pushups ∧ david_more_pushups_than_zachary D = 58 :=
by
  -- We leave the proof as an exercise or for further filling.
  sorry

end david_pushups_difference_l115_115480


namespace length_of_one_side_of_regular_octagon_l115_115397

-- Define the conditions of the problem
def is_regular_octagon (n : ℕ) (P : ℝ) (length_of_side : ℝ) : Prop :=
  n = 8 ∧ P = 72 ∧ length_of_side = P / n

-- State the theorem
theorem length_of_one_side_of_regular_octagon : is_regular_octagon 8 72 9 :=
by
  -- The proof is omitted; only the statement is required
  sorry

end length_of_one_side_of_regular_octagon_l115_115397


namespace calculate_neg2_add3_l115_115732

theorem calculate_neg2_add3 : (-2) + 3 = 1 :=
  sorry

end calculate_neg2_add3_l115_115732


namespace rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l115_115000

/-
Via conditions:
1. The rental company owns 100 cars.
2. When the monthly rent for each car is set at 3000 yuan, all cars can be rented out.
3. For every 50 yuan increase in the monthly rent per car, there will be one more car that is not rented out.
4. The maintenance cost for each rented car is 200 yuan per month.
-/

noncomputable def num_rented_cars (rent_per_car : ℕ) : ℕ :=
  if rent_per_car < 3000 then 100 else max 0 (100 - (rent_per_car - 3000) / 50)

noncomputable def monthly_revenue (rent_per_car : ℕ) : ℕ :=
  let cars_rented := num_rented_cars rent_per_car
  let maintenance_cost := 200 * cars_rented
  (rent_per_car - maintenance_cost) * cars_rented

theorem rent_3600_yields_88 : num_rented_cars 3600 = 88 :=
  sorry

theorem optimal_rent_is_4100_and_max_revenue_is_304200 :
  ∃ rent_per_car, rent_per_car = 4100 ∧ monthly_revenue rent_per_car = 304200 :=
  sorry

end rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l115_115000


namespace twelve_hens_lay_48_eggs_in_twelve_days_l115_115302

theorem twelve_hens_lay_48_eggs_in_twelve_days :
  (∀ (hens eggs days : ℕ), hens = 3 → eggs = 3 → days = 3 → eggs / (hens * days) = 1/3) → 
  ∀ (hens days : ℕ), hens = 12 → days = 12 → hens * days * (1/3) = 48 :=
by
  sorry

end twelve_hens_lay_48_eggs_in_twelve_days_l115_115302


namespace average_lecture_minutes_l115_115126

theorem average_lecture_minutes
  (lecture_duration : ℕ)
  (total_audience : ℕ)
  (percent_entire : ℝ)
  (percent_missed : ℝ)
  (percent_half : ℝ)
  (average_minutes : ℝ) :
  lecture_duration = 90 →
  total_audience = 200 →
  percent_entire = 0.30 →
  percent_missed = 0.20 →
  percent_half = 0.40 →
  average_minutes = 56.25 :=
by
  sorry

end average_lecture_minutes_l115_115126


namespace f_12_16_plus_f_16_12_l115_115424

noncomputable def f : ℕ × ℕ → ℕ :=
sorry

axiom ax1 : ∀ (x : ℕ), f (x, x) = x
axiom ax2 : ∀ (x y : ℕ), f (x, y) = f (y, x)
axiom ax3 : ∀ (x y : ℕ), (x + y) * f (x, y) = y * f (x, x + y)

theorem f_12_16_plus_f_16_12 : f (12, 16) + f (16, 12) = 96 :=
by sorry

end f_12_16_plus_f_16_12_l115_115424


namespace polygon_sides_l115_115565

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_l115_115565


namespace angle_measure_l115_115691

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l115_115691


namespace angle_triple_complement_l115_115665

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l115_115665


namespace total_distance_eq_l115_115929

def distance_traveled_by_bus : ℝ := 2.6
def distance_traveled_by_subway : ℝ := 5.98
def total_distance_traveled : ℝ := distance_traveled_by_bus + distance_traveled_by_subway

theorem total_distance_eq : total_distance_traveled = 8.58 := by
  sorry

end total_distance_eq_l115_115929


namespace find_remainder_2500th_term_l115_115881

theorem find_remainder_2500th_term : 
    let seq_position (n : ℕ) := n * (n + 1) / 2 
    let n := ((1 + Int.ofNat 20000).natAbs.sqrt + 1) / 2
    let term_2500 := if seq_position n < 2500 then n + 1 else n
    (term_2500 % 7) = 1 := by 
    sorry

end find_remainder_2500th_term_l115_115881


namespace exists_four_mutually_acquainted_l115_115921

-- Definition of acquaintanceship relation
def Acquaintance (X : Type) := X → X → Prop

-- Given conditions
section
variables {M : Type} [Fintype M] [DecidableRel (Acquaintance M)]
variable h_card : Fintype.card M = 9

-- Condition: For any three men, at least two are mutually acquainted
axiom acquaintance_condition : ∀ (s : Finset M), s.card = 3 → ∃ (x y : M), x ∈ s ∧ y ∈ s ∧ Acquaintance M x y

-- To prove: There exists a subset of four men who are all mutually acquainted
theorem exists_four_mutually_acquainted :
  ∃ (s : Finset M), s.card = 4 ∧ ∀ (x y : M), x ∈ s → y ∈ s → x ≠ y → Acquaintance M x y := sorry
end

end exists_four_mutually_acquainted_l115_115921


namespace proof_correct_option_C_l115_115900

def line := Type
def plane := Type
def perp (m : line) (α : plane) : Prop := sorry
def parallel (n : line) (α : plane) : Prop := sorry
def perpnal (m n: line): Prop := sorry 

variables (m n : line) (α β γ : plane)

theorem proof_correct_option_C : perp m α → parallel n α → perpnal m n := sorry

end proof_correct_option_C_l115_115900


namespace sallys_change_l115_115588

-- Given conditions
def frames_bought : ℕ := 3
def cost_per_frame : ℕ := 3
def payment : ℕ := 20

-- The statement to prove
theorem sallys_change : payment - (frames_bought * cost_per_frame) = 11 := by
  sorry

end sallys_change_l115_115588


namespace profit_equation_correct_l115_115979

theorem profit_equation_correct (x : ℝ) : 
  let original_selling_price := 36
  let purchase_price := 20
  let original_sales_volume := 200
  let price_increase_effect := 5
  let desired_profit := 1200
  let original_profit_per_unit := original_selling_price - purchase_price
  let new_selling_price := original_selling_price + x
  let new_sales_volume := original_sales_volume - price_increase_effect * x
  (original_profit_per_unit + x) * new_sales_volume = desired_profit :=
sorry

end profit_equation_correct_l115_115979


namespace sum_of_all_different_possible_areas_of_cool_rectangles_l115_115725

-- Define the concept of a cool rectangle
def is_cool_rectangle (a b : ℕ) : Prop :=
  a * b = 2 * (2 * a + 2 * b)

-- Define the function to calculate the area of a rectangle
def area (a b : ℕ) : ℕ := a * b

-- Define the set of pairs (a, b) that satisfy the cool rectangle condition
def cool_rectangle_pairs : List (ℕ × ℕ) :=
  [(5, 20), (6, 12), (8, 8)]

-- Calculate the sum of all different possible areas of cool rectangles
def sum_of_cool_rectangle_areas : ℕ :=
  List.sum (cool_rectangle_pairs.map (λ p => area p.fst p.snd))

-- Theorem statement
theorem sum_of_all_different_possible_areas_of_cool_rectangles :
  sum_of_cool_rectangle_areas = 236 :=
by
  -- This is where the proof would go based on the given solution.
  sorry

end sum_of_all_different_possible_areas_of_cool_rectangles_l115_115725


namespace angle_triple_complement_l115_115664

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l115_115664


namespace part1_part2_l115_115073

open Set

variable {U A B : Set ℝ}

def U := Real
def A := {x : ℝ | 0 ≤ x ∧ x ≤ 3 }
def B (m : ℝ) := {x : ℝ | m - 2 ≤ x ∧ x ≤ 2 * m }

theorem part1 (m : ℝ) (hm : m = 3) : 
  A ∩ (U \ B m) = {x : ℝ | 0 ≤ x ∧ x < 1 } := 
by 
  sorry

theorem part2 :
  {m : ℝ | A ∪ B m = B m} = {m : ℝ | 3/2 ≤ m ∧ m ≤ 2} := 
by 
  sorry

end part1_part2_l115_115073


namespace sachin_age_l115_115322

theorem sachin_age (S R : ℕ) (h1 : R = S + 18) (h2 : S * 9 = R * 7) : S = 63 := 
by
  sorry

end sachin_age_l115_115322


namespace initial_number_of_girls_l115_115281

theorem initial_number_of_girls (p : ℝ) (h : (0.4 * p - 2) / p = 0.3) : 0.4 * p = 8 := 
by
  sorry

end initial_number_of_girls_l115_115281


namespace find_y_l115_115952

-- Define the problem conditions
def avg_condition (y : ℝ) : Prop := (15 + 25 + y) / 3 = 23

-- Prove that the value of 'y' satisfying the condition is 29
theorem find_y (y : ℝ) (h : avg_condition y) : y = 29 :=
sorry

end find_y_l115_115952


namespace simplify_expression_l115_115946

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l115_115946


namespace min_value_parabola_l115_115462

theorem min_value_parabola : 
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 4 ∧ (-x^2 + 4 * x - 2) = -2 :=
by
  sorry

end min_value_parabola_l115_115462


namespace ratio_of_b_plus_e_over_c_plus_f_l115_115912

theorem ratio_of_b_plus_e_over_c_plus_f 
  (a b c d e f : ℝ)
  (h1 : a + b = 2 * a + c)
  (h2 : a - 2 * b = 4 * c)
  (h3 : a + b + c = 21)
  (h4 : d + e = 3 * d + f)
  (h5 : d - 2 * e = 5 * f)
  (h6 : d + e + f = 32) :
  (b + e) / (c + f) = -3.99 :=
sorry

end ratio_of_b_plus_e_over_c_plus_f_l115_115912


namespace player_jump_height_to_dunk_l115_115604

/-- Definitions given in the conditions -/
def rim_height : ℕ := 120
def player_height : ℕ := 72
def player_reach_above_head : ℕ := 22

/-- The statement to be proven -/
theorem player_jump_height_to_dunk :
  rim_height - (player_height + player_reach_above_head) = 26 :=
by
  sorry

end player_jump_height_to_dunk_l115_115604


namespace c_linear_combination_of_a_b_l115_115244

-- Definitions of vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, 2)

-- Theorem stating the relationship between vectors a, b, and c
theorem c_linear_combination_of_a_b :
  c = (1 / 2 : ℝ) • a + (-3 / 2 : ℝ) • b :=
  sorry

end c_linear_combination_of_a_b_l115_115244


namespace prob_no_decrease_white_in_A_is_correct_l115_115407

-- Define the conditions of the problem
def bagA_white : ℕ := 3
def bagA_black : ℕ := 5
def bagB_white : ℕ := 4
def bagB_black : ℕ := 6

-- Define the probabilities involved
def prob_draw_black_from_A : ℚ := 5 / 8
def prob_draw_white_from_A : ℚ := 3 / 8
def prob_put_white_back_into_A_conditioned_on_white_drawn : ℚ := 5 / 11

-- Calculate the combined probability
def prob_no_decrease_white_in_A : ℚ := prob_draw_black_from_A + prob_draw_white_from_A * prob_put_white_back_into_A_conditioned_on_white_drawn

-- Prove the probability is as expected
theorem prob_no_decrease_white_in_A_is_correct : prob_no_decrease_white_in_A = 35 / 44 := by
  sorry

end prob_no_decrease_white_in_A_is_correct_l115_115407


namespace sin_cos_product_l115_115553

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l115_115553


namespace determine_parabola_equation_l115_115980

-- Given conditions
variable (p : ℝ) (h_p : p > 0)
variable (x1 x2 : ℝ)
variable (AF BF : ℝ)
variable (h_AF : AF = x1 + p / 2)
variable (h_BF : BF = x2 + p / 2)
variable (h_AF_value : AF = 2)
variable (h_BF_value : BF = 3)

-- Prove the equation of the parabola
theorem determine_parabola_equation (h1 : x1 + x2 = 5 - p)
(h2 : x1 * x2 = p^2 / 4)
(h3 : AF * BF = 6) :
  y^2 = (24/5 : ℝ) * x := 
sorry

end determine_parabola_equation_l115_115980


namespace average_first_six_numbers_l115_115283

theorem average_first_six_numbers (A : ℝ) (h1 : (11 : ℝ) * 9.9 = (6 * A + 6 * 11.4 - 22.5)) : A = 10.5 :=
by sorry

end average_first_six_numbers_l115_115283


namespace range_of_a_if_slope_is_obtuse_l115_115785

theorem range_of_a_if_slope_is_obtuse : 
  ∀ a : ℝ, (a^2 + 2 * a < 0) → -2 < a ∧ a < 0 :=
by
  intro a
  intro h
  sorry

end range_of_a_if_slope_is_obtuse_l115_115785


namespace largest_prime_factor_4752_l115_115312

theorem largest_prime_factor_4752 : ∃ p : ℕ, p = 11 ∧ prime p ∧ (∀ q : ℕ, prime q ∧ q ∣ 4752 → q ≤ 11) :=
by
  sorry

end largest_prime_factor_4752_l115_115312


namespace least_three_digit_divisible_3_4_7_is_168_l115_115476

-- Define the function that checks the conditions
def is_least_three_digit_divisible_by_3_4_7 (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ x % 3 = 0 ∧ x % 4 = 0 ∧ x % 7 = 0

-- Define the target value
def least_three_digit_number_divisible_by_3_4_7 : ℕ := 168

-- The theorem we want to prove
theorem least_three_digit_divisible_3_4_7_is_168 :
  ∃ x : ℕ, is_least_three_digit_divisible_by_3_4_7 x ∧ x = least_three_digit_number_divisible_by_3_4_7 := by
  sorry

end least_three_digit_divisible_3_4_7_is_168_l115_115476


namespace nebraska_more_plates_than_georgia_l115_115825

theorem nebraska_more_plates_than_georgia :
  (26 ^ 2 * 10 ^ 5) - (26 ^ 4 * 10 ^ 2) = 21902400 :=
by
  sorry

end nebraska_more_plates_than_georgia_l115_115825


namespace cuberoot_sum_l115_115011

-- Prove that the sum c + d = 60 for the simplified form of the given expression.
theorem cuberoot_sum :
  let c := 15
  let d := 45
  c + d = 60 :=
by
  sorry

end cuberoot_sum_l115_115011


namespace xyz_identity_l115_115240

theorem xyz_identity (x y z : ℝ) 
  (h1 : x + y + z = 14) 
  (h2 : xy + xz + yz = 32) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 1400 := 
by 
  -- Proof steps will be placed here, use sorry for now
  sorry

end xyz_identity_l115_115240


namespace arithmetic_geometric_sum_l115_115060

theorem arithmetic_geometric_sum (S : ℕ → ℕ) (n : ℕ) 
  (h1 : S n = 48) 
  (h2 : S (2 * n) = 60)
  (h3 : (S (2 * n) - S n) ^ 2 = S n * (S (3 * n) - S (2 * n))) : 
  S (3 * n) = 63 := by
  sorry

end arithmetic_geometric_sum_l115_115060


namespace fraction_to_terminating_decimal_l115_115746

theorem fraction_to_terminating_decimal :
  (45 / (2^2 * 5^3) : ℚ) = 0.09 :=
by sorry

end fraction_to_terminating_decimal_l115_115746


namespace leftover_potatoes_l115_115330

theorem leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ)
    (h1 : fries_per_potato = 25) (h2 : total_potatoes = 15) (h3 : required_fries = 200) :
    (total_potatoes - required_fries / fries_per_potato) = 7 :=
sorry

end leftover_potatoes_l115_115330


namespace calculate_expression_l115_115139

theorem calculate_expression :
  (-3)^4 + (-3)^3 + 3^3 + 3^4 = 162 :=
by
  -- Since all necessary conditions are listed in the problem statement, we honor this structure
  -- The following steps are required logically but are not presently necessary for detailed proof means.
  sorry

end calculate_expression_l115_115139


namespace total_weight_of_7_moles_CaO_l115_115107

/-- Definitions necessary for the problem --/
def atomic_weight_Ca : ℝ := 40.08 -- atomic weight of calcium in g/mol
def atomic_weight_O : ℝ := 16.00 -- atomic weight of oxygen in g/mol
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O -- molecular weight of CaO in g/mol
def number_of_moles_CaO : ℝ := 7 -- number of moles of CaO

/-- The main theorem statement --/
theorem total_weight_of_7_moles_CaO :
  molecular_weight_CaO * number_of_moles_CaO = 392.56 :=
by
  sorry

end total_weight_of_7_moles_CaO_l115_115107


namespace percy_swimming_hours_l115_115270

theorem percy_swimming_hours :
  let weekday_hours_per_day := 2
  let weekdays := 5
  let weekend_hours := 3
  let weeks := 4
  let total_weekday_hours_per_week := weekday_hours_per_day * weekdays
  let total_weekend_hours_per_week := weekend_hours
  let total_hours_per_week := total_weekday_hours_per_week + total_weekend_hours_per_week
  let total_hours_over_weeks := total_hours_per_week * weeks
  total_hours_over_weeks = 64 :=
by
  sorry

end percy_swimming_hours_l115_115270


namespace charlie_and_delta_can_purchase_l115_115974

def num_cookies : ℕ := 7
def num_cupcakes : ℕ := 4
def total_items : ℕ := 4

def ways_purchase (c d : ℕ) : ℕ :=
  if d > c then 0 else
    (Finset.card (Finset.powersetLen c (Finset.range (num_cookies + num_cupcakes)))
    * (if d = 0 then 1
       else (Finset.card (Finset.powersetLen d (Finset.range num_cookies)) + (num_cookies * (num_cookies - 1)) / 2)))

def total_ways : ℕ :=
  ways_purchase 4 0 + ways_purchase 3 1 + ways_purchase 2 2 + ways_purchase 1 3 + ways_purchase 0 4

theorem charlie_and_delta_can_purchase : total_ways = 4054 :=
  by sorry

end charlie_and_delta_can_purchase_l115_115974


namespace alex_ate_more_pears_than_sam_l115_115145

namespace PearEatingContest

def number_of_pears_eaten (Alex Sam : ℕ) : ℕ :=
  Alex - Sam

theorem alex_ate_more_pears_than_sam :
  number_of_pears_eaten 8 2 = 6 := by
  -- proof
  sorry

end PearEatingContest

end alex_ate_more_pears_than_sam_l115_115145


namespace range_of_x_l115_115755

theorem range_of_x (a b c x : ℝ) (h1 : a^2 + 2 * b^2 + 3 * c^2 = 6) (h2 : a + 2 * b + 3 * c > |x + 1|) : -7 < x ∧ x < 5 :=
by
  sorry

end range_of_x_l115_115755


namespace calories_in_300g_lemonade_proof_l115_115425

def g_lemon := 150
def g_sugar := 200
def g_water := 450

def c_lemon_per_100g := 30
def c_sugar_per_100g := 400
def c_water := 0

def total_calories :=
  g_lemon * c_lemon_per_100g / 100 +
  g_sugar * c_sugar_per_100g / 100 +
  g_water * c_water

def total_weight := g_lemon + g_sugar + g_water

def caloric_density := total_calories / total_weight

def calories_in_300g_lemonade := 300 * caloric_density

theorem calories_in_300g_lemonade_proof : calories_in_300g_lemonade = 317 := by
  sorry

end calories_in_300g_lemonade_proof_l115_115425


namespace gcd_of_102_and_238_l115_115106

theorem gcd_of_102_and_238 : Nat.gcd 102 238 = 34 := 
by 
  sorry

end gcd_of_102_and_238_l115_115106


namespace sin_cos_product_l115_115555

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l115_115555


namespace simplify_expression_l115_115441

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115441


namespace triple_complement_angle_l115_115683

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l115_115683


namespace scramble_time_is_correct_l115_115260

-- Define the conditions
def sausages : ℕ := 3
def fry_time_per_sausage : ℕ := 5
def eggs : ℕ := 6
def total_time : ℕ := 39

-- Define the time to scramble each egg
def scramble_time_per_egg : ℕ :=
  let frying_time := sausages * fry_time_per_sausage
  let scrambling_time := total_time - frying_time
  scrambling_time / eggs

-- The theorem stating the main question and desired answer
theorem scramble_time_is_correct : scramble_time_per_egg = 4 := by
  sorry

end scramble_time_is_correct_l115_115260


namespace angle_triple_complement_l115_115675

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l115_115675


namespace star_result_l115_115015

-- Define the operation star
def star (m n p q : ℚ) := (m * p) * (n / q)

-- Given values
def a := (5 : ℚ) / 9
def b := (10 : ℚ) / 6

-- Condition to check
theorem star_result : star 5 9 10 6 = 75 := by
  sorry

end star_result_l115_115015


namespace solve_for_y_in_terms_of_x_l115_115041

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : 2 * x - 7 * y = 5) : y = (2 * x - 5) / 7 :=
sorry

end solve_for_y_in_terms_of_x_l115_115041


namespace expand_expression_l115_115148

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115148


namespace find_x_l115_115237

theorem find_x
  (a b x : ℝ)
  (h1 : a * (x + 2) + b * (x + 2) = 60)
  (h2 : a + b = 12) :
  x = 3 :=
by
  sorry

end find_x_l115_115237


namespace divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l115_115341

theorem divisible_by_6_implies_divisible_by_2 :
  ∀ (n : ℤ), (6 ∣ n) → (2 ∣ n) :=
by sorry

theorem not_divisible_by_2_implies_not_divisible_by_6 :
  ∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n) :=
by sorry

theorem equivalence_of_propositions :
  (∀ (n : ℤ), (6 ∣ n) → (2 ∣ n)) ↔ (∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n)) :=
by sorry


end divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l115_115341


namespace harry_morning_routine_time_l115_115360

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end harry_morning_routine_time_l115_115360


namespace num_red_balls_l115_115922

theorem num_red_balls (x : ℕ) (h : 4 / (4 + x) = 1 / 5) : x = 16 :=
by
  sorry

end num_red_balls_l115_115922


namespace range_of_a_l115_115775

def P (x : ℝ) : Prop := x^2 ≤ 1

def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : ∀ x, (P x ∨ x = a) ↔ P x) : P a :=
by
  sorry

end range_of_a_l115_115775


namespace n_squared_plus_n_plus_1_is_perfect_square_l115_115372

theorem n_squared_plus_n_plus_1_is_perfect_square (n : ℕ) :
  (∃ k : ℕ, n^2 + n + 1 = k^2) ↔ n = 0 :=
by
  sorry

end n_squared_plus_n_plus_1_is_perfect_square_l115_115372


namespace geometric_sequence_a5_l115_115086

theorem geometric_sequence_a5 (α : Type) [LinearOrderedField α] (a : ℕ → α)
  (h1 : ∀ n, a (n + 1) = a n * 2)
  (h2 : ∀ n, a n > 0)
  (h3 : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end geometric_sequence_a5_l115_115086


namespace coordinates_of_P_l115_115868

-- Definitions of conditions
def inFourthQuadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def absEqSeven (x : ℝ) : Prop := |x| = 7
def ysquareEqNine (y : ℝ) : Prop := y^2 = 9

-- Main theorem
theorem coordinates_of_P (x y : ℝ) (hx : absEqSeven x) (hy : ysquareEqNine y) (hq : inFourthQuadrant x y) :
  (x, y) = (7, -3) :=
  sorry

end coordinates_of_P_l115_115868


namespace machine_working_days_l115_115607

variable {V a b c x y z : ℝ} 

noncomputable def machine_individual_times_condition (a b c : ℝ) : Prop :=
  ∀ (x y z : ℝ), (x = a + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (y = b + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (z = (-(c * (a + b)) + c * Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (c > 1)

theorem machine_working_days (h1 : x = (z / c) + a) (h2 : y = (z / c) + b) (h3 : z = c * (z / c)) :
  machine_individual_times_condition a b c :=
by
  sorry

end machine_working_days_l115_115607


namespace number_of_penny_piles_l115_115277

theorem number_of_penny_piles
    (piles_of_quarters : ℕ := 4) 
    (piles_of_dimes : ℕ := 6)
    (piles_of_nickels : ℕ := 9)
    (total_value_in_dollars : ℝ := 21)
    (coins_per_pile : ℕ := 10)
    (quarter_value : ℝ := 0.25)
    (dime_value : ℝ := 0.10)
    (nickel_value : ℝ := 0.05)
    (penny_value : ℝ := 0.01) :
    (total_value_in_dollars - ((piles_of_quarters * coins_per_pile * quarter_value) +
                               (piles_of_dimes * coins_per_pile * dime_value) +
                               (piles_of_nickels * coins_per_pile * nickel_value))) /
                               (coins_per_pile * penny_value) = 5 := 
by
  sorry

end number_of_penny_piles_l115_115277


namespace find_weight_l115_115002

-- Define the weight of each box before taking out 20 kg as W
variable (W : ℚ)

-- Define the condition given in the problem
def condition : Prop := 7 * (W - 20) = 3 * W

-- The proof goal is to prove W = 35 under the given condition
theorem find_weight (h : condition W) : W = 35 := by
  sorry

end find_weight_l115_115002


namespace domain_of_sqrt_2cosx_plus_1_l115_115747

noncomputable def domain_sqrt_2cosx_plus_1 (x : ℝ) : Prop :=
  ∃ (k : ℤ), (2 * k * Real.pi - 2 * Real.pi / 3) ≤ x ∧ x ≤ (2 * k * Real.pi + 2 * Real.pi / 3)

theorem domain_of_sqrt_2cosx_plus_1 :
  (∀ (x: ℝ), 0 ≤ 2 * Real.cos x + 1 ↔ domain_sqrt_2cosx_plus_1 x) :=
by
  sorry

end domain_of_sqrt_2cosx_plus_1_l115_115747


namespace expand_polynomial_l115_115177

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l115_115177


namespace probability_same_class_l115_115505

-- Define the problem conditions
def num_classes : ℕ := 3
def total_scenarios : ℕ := num_classes * num_classes
def same_class_scenarios : ℕ := num_classes

-- Formulate the proof problem
theorem probability_same_class :
  (same_class_scenarios : ℚ) / total_scenarios = 1 / 3 :=
sorry

end probability_same_class_l115_115505


namespace angle_measure_l115_115686

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l115_115686


namespace goldfish_equal_in_seven_months_l115_115994

/-- Define the growth of Alice's goldfish: they triple every month. -/
def alice_goldfish (n : ℕ) : ℕ := 3 * 3 ^ n

/-- Define the growth of Bob's goldfish: they quadruple every month. -/
def bob_goldfish (n : ℕ) : ℕ := 256 * 4 ^ n

/-- The main theorem we want to prove: For Alice and Bob's goldfish count to be equal,
    it takes 7 months. -/
theorem goldfish_equal_in_seven_months : ∃ n : ℕ, alice_goldfish n = bob_goldfish n ∧ n = 7 := 
by
  sorry

end goldfish_equal_in_seven_months_l115_115994


namespace three_angles_difference_is_2pi_over_3_l115_115757

theorem three_angles_difference_is_2pi_over_3 (α β γ : ℝ) 
    (h1 : 0 ≤ α) (h2 : α ≤ β) (h3 : β < γ) (h4 : γ ≤ 2 * Real.pi)
    (h5 : Real.cos α + Real.cos β + Real.cos γ = 0)
    (h6 : Real.sin α + Real.sin β + Real.sin γ = 0) :
    β - α = 2 * Real.pi / 3 :=
sorry

end three_angles_difference_is_2pi_over_3_l115_115757


namespace simplify_expression_l115_115438

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115438


namespace remainder_of_n_mod_1000_l115_115577

-- Definition of the set S
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 15 }

-- Define the number of sets of three non-empty disjoint subsets of S
def num_sets_of_three_non_empty_disjoint_subsets (S : Set ℕ) : ℕ :=
  let total_partitions := 4^15
  let single_empty_partition := 3 * 3^15
  let double_empty_partition := 3 * 2^15
  let all_empty_partition := 1
  total_partitions - single_empty_partition + double_empty_partition - all_empty_partition

-- Compute the result of the number modulo 1000
def result := (num_sets_of_three_non_empty_disjoint_subsets S) % 1000

-- Theorem that states the remainder when n is divided by 1000
theorem remainder_of_n_mod_1000 : result = 406 := by
  sorry

end remainder_of_n_mod_1000_l115_115577


namespace foci_and_directrices_of_ellipse_l115_115028

noncomputable def parametricEllipse
    (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ + 1, 4 * Real.sin θ)

theorem foci_and_directrices_of_ellipse :
  (∀ θ : ℝ, parametricEllipse θ = (x, y)) →
  (∃ (f1 f2 : ℝ × ℝ) (d1 d2 : ℝ → Prop),
    f1 = (1, Real.sqrt 7) ∧
    f2 = (1, -Real.sqrt 7) ∧
    d1 = fun x => x = 1 + 9 / Real.sqrt 7 ∧
    d2 = fun x => x = 1 - 9 / Real.sqrt 7) := sorry

end foci_and_directrices_of_ellipse_l115_115028


namespace symmetric_points_x_axis_l115_115398

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ := (a, 1)) (Q : ℝ × ℝ := (-4, b)) :
  (Q.1 = -P.1 ∧ Q.2 = -P.2) → (a = -4 ∧ b = -1) :=
by {
  sorry
}

end symmetric_points_x_axis_l115_115398


namespace area_of_square_l115_115274

noncomputable def square_area (s : ℝ) : ℝ := s ^ 2

theorem area_of_square
  {E F G H : Type}
  (ABCD : Type)
  (on_segments : E → F → G → H → Prop)
  (EG FH : ℝ)
  (angle_intersection : ℝ)
  (hEG : EG = 7)
  (hFH : FH = 8)
  (hangle : angle_intersection = 30) :
  ∃ s : ℝ, square_area s = 147 / 4 :=
sorry

end area_of_square_l115_115274


namespace angle_triple_complement_l115_115673

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l115_115673


namespace sequence_count_l115_115222

theorem sequence_count :
  ∃ (a : ℕ → ℕ), 
    a 10 = 3 * a 1 ∧ 
    a 2 + a 8 = 2 * a 5 ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 9 → a (i + 1) = 1 + a i ∨ a (i + 1) = 2 + a i) ∧ 
    (∃ n, n = 80) :=
sorry

end sequence_count_l115_115222


namespace coordinates_of_B_l115_115534

noncomputable def B_coordinates := 
  let A : ℝ × ℝ := (-1, -5)
  let a : ℝ × ℝ := (2, 3)
  let AB := (3 * a.1, 3 * a.2)
  let B := (A.1 + AB.1, A.2 + AB.2)
  B

theorem coordinates_of_B : B_coordinates = (5, 4) := 
by 
  sorry

end coordinates_of_B_l115_115534


namespace square_perimeter_l115_115951

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by 
  have h1 : s = 1 := 
  begin
    -- Considering non-zero s, divide both sides by s
    by_contradiction hs,
    have hs' : s ≠ 0 := λ h0, hs (by simp [h0] at h),
    exact hs (by simpa [h, hs'] using h),
  end,
  rw h1,
  norm_num

end square_perimeter_l115_115951


namespace seq_eq_a1_b1_l115_115261

theorem seq_eq_a1_b1 {a b : ℕ → ℝ} 
  (h1 : ∀ n, a (n + 1) = 2 * b n - a n)
  (h2 : ∀ n, b (n + 1) = 2 * a n - b n)
  (h3 : ∀ n, a n > 0) :
  a 1 = b 1 := 
sorry

end seq_eq_a1_b1_l115_115261


namespace expand_polynomial_l115_115174

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l115_115174


namespace ratio_out_of_school_friends_to_classmates_l115_115575

variable (F : ℕ) (classmates : ℕ := 20) (parents : ℕ := 2) (sister : ℕ := 1) (total : ℕ := 33)

theorem ratio_out_of_school_friends_to_classmates (h : classmates + F + parents + sister = total) :
  (F : ℚ) / classmates = 1 / 2 := by
    -- sorry allows this to build even if proof is not provided
    sorry

end ratio_out_of_school_friends_to_classmates_l115_115575


namespace probability_of_ellipse_l115_115079

open ProbabilityTheory MeasureTheory

noncomputable def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop := 
  m^2 < 4

theorem probability_of_ellipse (h : ∀ m, m ∈ set.Icc 1 5) : 
  ℙ (λ m, is_ellipse_with_foci_on_y_axis m) = 1 / 4 :=
sorry

end probability_of_ellipse_l115_115079


namespace find_circle_radius_l115_115749

-- Definitions based on the given conditions
def circle_eq (x y : ℝ) : Prop := (x^2 - 8*x + y^2 - 10*y + 34 = 0)

-- Problem statement
theorem find_circle_radius (x y : ℝ) : circle_eq x y → ∃ r : ℝ, r = Real.sqrt 7 :=
by
  sorry

end find_circle_radius_l115_115749


namespace printer_fraction_l115_115098

noncomputable def basic_computer_price : ℝ := 2000
noncomputable def total_basic_price : ℝ := 2500
noncomputable def printer_price : ℝ := total_basic_price - basic_computer_price -- inferred as 500

noncomputable def enhanced_computer_price : ℝ := basic_computer_price + 500
noncomputable def total_enhanced_price : ℝ := enhanced_computer_price + printer_price -- inferred as 3000

theorem printer_fraction  (h1 : basic_computer_price + printer_price = total_basic_price)
                          (h2 : basic_computer_price = 2000)
                          (h3 : enhanced_computer_price = basic_computer_price + 500) :
  printer_price / total_enhanced_price = 1 / 6 :=
  sorry

end printer_fraction_l115_115098


namespace expand_expression_l115_115187

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115187


namespace sallys_change_l115_115586

-- Define the total cost calculation:
def totalCost (numFrames : Nat) (costPerFrame : Nat) : Nat :=
  numFrames * costPerFrame

-- Define the change calculation:
def change (totalAmount : Nat) (amountPaid : Nat) : Nat :=
  amountPaid - totalAmount

-- Define the specific conditions in the problem:
def numFrames := 3
def costPerFrame := 3
def amountPaid := 20

-- Prove that the change Sally gets is $11:
theorem sallys_change : change (totalCost numFrames costPerFrame) amountPaid = 11 := by
  sorry

end sallys_change_l115_115586


namespace quadratic_one_real_root_positive_n_l115_115053

theorem quadratic_one_real_root_positive_n (n : ℝ) (h : (n ≠ 0)) :
  (∃ x : ℝ, (x^2 - 6*n*x - 9*n) = 0) ∧
  (∀ x y : ℝ, (x^2 - 6*n*x - 9*n) = 0 → (y^2 - 6*n*y - 9*n) = 0 → x = y) ↔
  n = 0 := by
  sorry

end quadratic_one_real_root_positive_n_l115_115053


namespace ice_cream_flavors_l115_115234

-- Definition of the problem setup
def number_of_flavors : ℕ :=
  let scoops := 5
  let dividers := 2
  let total_objects := scoops + dividers
  Nat.choose total_objects dividers

-- Statement of the theorem
theorem ice_cream_flavors : number_of_flavors = 21 := by
  -- The proof of the theorem will use combinatorics to show the result.
  sorry

end ice_cream_flavors_l115_115234


namespace final_silver_tokens_l115_115993

structure TokenCounts :=
  (red : ℕ)
  (blue : ℕ)

def initial_tokens : TokenCounts := { red := 100, blue := 50 }

def exchange_booth1 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red - 3, blue := tokens.blue + 2 }

def exchange_booth2 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red + 1, blue := tokens.blue - 3 }

noncomputable def max_exchanges (initial : TokenCounts) : ℕ × ℕ :=
  let x := 48
  let y := 47
  (x, y)

noncomputable def silver_tokens (x y : ℕ) : ℕ := x + y

theorem final_silver_tokens (x y : ℕ) (tokens : TokenCounts) 
  (hx : tokens.red = initial_tokens.red - 3 * x + y)
  (hy : tokens.blue = initial_tokens.blue + 2 * x - 3 * y) 
  (hx_le : tokens.red >= 3 → false)
  (hy_le : tokens.blue >= 3 → false) : 
  silver_tokens x y = 95 :=
by {
  sorry
}

end final_silver_tokens_l115_115993


namespace factorize_expression_l115_115022

theorem factorize_expression (R : Type*) [CommRing R] (m n : R) : 
  m^2 * n - n = n * (m + 1) * (m - 1) := 
sorry

end factorize_expression_l115_115022


namespace eugene_swim_time_l115_115353

-- Define the conditions
variable (S : ℕ) -- Swim time on Sunday
variable (swim_time_mon : ℕ := 30) -- Swim time on Monday
variable (swim_time_tue : ℕ := 45) -- Swim time on Tuesday
variable (average_swim_time : ℕ := 34) -- Average swim time over three days

-- The total swim time over three days
def total_swim_time := S + swim_time_mon + swim_time_tue

-- The problem statement: Prove that given the conditions, Eugene swam for 27 minutes on Sunday.
theorem eugene_swim_time : total_swim_time S = 3 * average_swim_time → S = 27 := by
  -- Proof process will follow here
  sorry

end eugene_swim_time_l115_115353


namespace expand_expression_l115_115166

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l115_115166


namespace sum_of_inversion_counts_of_all_permutations_l115_115810

noncomputable def sum_of_inversion_counts (n : ℕ) (fixed_val : ℕ) (fixed_pos : ℕ) : ℕ :=
  if n = 6 ∧ fixed_val = 4 ∧ fixed_pos = 3 then 120 else 0

theorem sum_of_inversion_counts_of_all_permutations :
  sum_of_inversion_counts 6 4 3 = 120 :=
by
  sorry

end sum_of_inversion_counts_of_all_permutations_l115_115810


namespace y_days_do_work_l115_115490

theorem y_days_do_work (d : ℝ) (h : (1 / 30) + (1 / d) = 1 / 18) : d = 45 := 
by
  sorry

end y_days_do_work_l115_115490


namespace sqrt_expression_l115_115733

theorem sqrt_expression : 
  (Real.sqrt 75 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 6 * Real.sqrt 2 = 3 * Real.sqrt 3 - Real.sqrt 2) := 
by 
  sorry

end sqrt_expression_l115_115733


namespace Mr_Mayer_purchase_price_l115_115430

theorem Mr_Mayer_purchase_price 
  (P : ℝ) 
  (H1 : (1.30 * 2) * P = 2600) : 
  P = 1000 := 
by
  sorry

end Mr_Mayer_purchase_price_l115_115430


namespace baseball_glove_price_l115_115550

noncomputable def original_price_glove : ℝ := 42.50

theorem baseball_glove_price (cards bat glove_discounted cleats total : ℝ) 
  (h1 : cards = 25) 
  (h2 : bat = 10) 
  (h3 : cleats = 2 * 10)
  (h4 : total = 79) 
  (h5 : glove_discounted = total - (cards + bat + cleats)) 
  (h6 : glove_discounted = 0.80 * original_price_glove) : 
  original_price_glove = 42.50 := by 
  sorry

end baseball_glove_price_l115_115550


namespace diff_of_squares_not_2018_l115_115734

theorem diff_of_squares_not_2018 (a b : ℕ) (h : a > b) : ¬(a^2 - b^2 = 2018) :=
by {
  -- proof goes here
  sorry
}

end diff_of_squares_not_2018_l115_115734


namespace decimal_15_to_binary_l115_115349

theorem decimal_15_to_binary : (15 : ℕ) = (4*1 + 2*1 + 1*1)*2^3 + (4*1 + 2*1 + 1*1)*2^2 + (4*1 + 2*1 + 1*1)*2 + 1 := by
  sorry

end decimal_15_to_binary_l115_115349


namespace building_floors_l115_115977

-- Define the properties of the staircases
def staircaseA_steps : Nat := 104
def staircaseB_steps : Nat := 117
def staircaseC_steps : Nat := 156

-- The problem asks us to show the number of floors, which is the gcd of the steps of all staircases 
theorem building_floors :
  Nat.gcd (Nat.gcd staircaseA_steps staircaseB_steps) staircaseC_steps = 13 :=
by
  sorry

end building_floors_l115_115977


namespace expand_expression_l115_115164

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l115_115164


namespace train_cross_bridge_time_l115_115047

open Nat

-- Defining conditions as per the problem
def train_length : ℕ := 200
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 5 / 18
def total_distance : ℕ := train_length + bridge_length
def time_to_cross : ℕ := total_distance / speed_mps

-- Stating the theorem
theorem train_cross_bridge_time : time_to_cross = 35 := by
  sorry

end train_cross_bridge_time_l115_115047


namespace evaluate_ceiling_neg_cubed_frac_l115_115354

theorem evaluate_ceiling_neg_cubed_frac :
  (Int.ceil ((- (5 : ℚ) / 3) ^ 3 + 1) = -3) :=
sorry

end evaluate_ceiling_neg_cubed_frac_l115_115354


namespace existence_of_E_l115_115036

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 6) + (y^2 / 2) = 1

def point_on_x_axis (E : ℝ × ℝ) : Prop := E.snd = 0

def ea_dot_eb_constant (E A B : ℝ × ℝ) : ℝ :=
  let ea := (A.fst - E.fst, A.snd)
  let eb := (B.fst - E.fst, B.snd)
  ea.fst * eb.fst + ea.snd * eb.snd

noncomputable def E : ℝ × ℝ := (7/3, 0)

noncomputable def const_value : ℝ := (-5/9)

theorem existence_of_E :
  (∃ E, point_on_x_axis E ∧
        (∀ A B, ellipse_eq A.fst A.snd ∧ ellipse_eq B.fst B.snd →
                  ea_dot_eb_constant E A B = const_value)) :=
  sorry

end existence_of_E_l115_115036


namespace family_snails_l115_115335

def total_snails_family (n1 n2 n3 n4 : ℕ) (mother_find : ℕ) : ℕ :=
  n1 + n2 + n3 + mother_find

def first_ducklings_snails (num_ducklings : ℕ) (snails_per_duckling : ℕ) : ℕ :=
  num_ducklings * snails_per_duckling

def remaining_ducklings_snails (num_ducklings : ℕ) (mother_snails : ℕ) : ℕ :=
  num_ducklings * (mother_snails / 2)

def mother_find_snails (snails_group1 : ℕ) (snails_group2 : ℕ) : ℕ :=
  3 * (snails_group1 + snails_group2)

theorem family_snails : 
  ∀ (ducklings : ℕ) (group1_ducklings group2_ducklings : ℕ) 
    (snails1 snails2 : ℕ) 
    (total_ducklings : ℕ), 
    ducklings = 8 →
    group1_ducklings = 3 → 
    group2_ducklings = 3 → 
    snails1 = 5 →
    snails2 = 9 →
    total_ducklings = group1_ducklings + group2_ducklings + 2 →
    total_snails_family 
      (first_ducklings_snails group1_ducklings snails1)
      (first_ducklings_snails group2_ducklings snails2)
      (remaining_ducklings_snails 2 (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)))
      (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)) 
    = 294 :=
by intros; sorry

end family_snails_l115_115335


namespace solve_system1_solve_system2_l115_115082

-- Define the conditions and the proof problem for System 1
theorem solve_system1 (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : 3 * x + 2 * y = 7) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- Define the conditions and the proof problem for System 2
theorem solve_system2 (x y : ℝ) (h1 : x - y = 3) (h2 : (x - y - 3) / 2 - y / 3 = -1) :
  x = 6 ∧ y = 3 := by
  sorry

end solve_system1_solve_system2_l115_115082


namespace sara_disproves_tom_l115_115891

-- Define the type and predicate of cards
inductive Card
| K
| M
| card5
| card7
| card8

open Card

-- Define the conditions
def is_consonant : Card → Prop
| K => true
| M => true
| _ => false

def is_odd : Card → Prop
| card5 => true
| card7 => true
| _ => false

def is_even : Card → Prop
| card8 => true
| _ => false

-- Tom's statement
def toms_statement : Prop :=
  ∀ c, is_consonant c → is_odd c

-- The card Sara turns over (card8) to disprove Tom's statement
theorem sara_disproves_tom : is_even card8 ∧ is_consonant card8 → ¬toms_statement :=
by
  sorry

end sara_disproves_tom_l115_115891


namespace lcm_24_36_45_l115_115704

noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem lcm_24_36_45 : lcm_three 24 36 45 = 360 := by
  -- Conditions involves proving the prime factorizations
  have h1 : Nat.factors 24 = [2, 2, 2, 3] := by sorry -- Prime factorization of 24
  have h2 : Nat.factors 36 = [2, 2, 3, 3] := by sorry -- Prime factorization of 36
  have h3 : Nat.factors 45 = [3, 3, 5] := by sorry -- Prime factorization of 45

  -- Least common multiple calculation based on the greatest powers of prime factors
  sorry -- This is where the proof would go

end lcm_24_36_45_l115_115704


namespace initial_solution_amount_l115_115884

theorem initial_solution_amount (x : ℝ) (h1 : x - 200 + 1000 = 2000) : x = 1200 := by
  sorry

end initial_solution_amount_l115_115884


namespace exponent_arithmetic_proof_l115_115879

theorem exponent_arithmetic_proof :
  ( (6 ^ 6 / 6 ^ 5) ^ 3 * 8 ^ 3 / 4 ^ 3) = 1728 := by
  sorry

end exponent_arithmetic_proof_l115_115879


namespace expand_expression_l115_115199

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115199


namespace calculation1_calculation2_calculation3_calculation4_l115_115940

theorem calculation1 : 72 * 54 + 28 * 54 = 5400 := 
by sorry

theorem calculation2 : 60 * 25 * 8 = 12000 := 
by sorry

theorem calculation3 : 2790 / (250 * 12 - 2910) = 31 := 
by sorry

theorem calculation4 : (100 - 1456 / 26) * 78 = 3432 := 
by sorry

end calculation1_calculation2_calculation3_calculation4_l115_115940


namespace least_three_digit_multiple_of_3_4_7_l115_115475

theorem least_three_digit_multiple_of_3_4_7 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 → m % 3 = 0 ∧ m % 4 = 0 ∧ m % 7 = 0 → n ≤ m :=
  sorry

end least_three_digit_multiple_of_3_4_7_l115_115475


namespace fraction_meaningful_l115_115243

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ¬ (x - 1 = 0) :=
by
  sorry

end fraction_meaningful_l115_115243


namespace expand_polynomial_eq_l115_115195

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l115_115195


namespace tan_half_A_mul_tan_half_C_eq_third_l115_115403

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem tan_half_A_mul_tan_half_C_eq_third (h : a + c = 2 * b) :
  (Real.tan (A / 2)) * (Real.tan (C / 2)) = 1 / 3 :=
sorry

end tan_half_A_mul_tan_half_C_eq_third_l115_115403


namespace min_value_proven_l115_115214

open Real

noncomputable def min_value (x y : ℝ) (h1 : log x + log y = 1) : Prop :=
  2 * x + 5 * y ≥ 20 ∧ (2 * x + 5 * y = 20 ↔ 2 * x = 5 * y ∧ x * y = 10)

theorem min_value_proven (x y : ℝ) (h1 : log x + log y = 1) :
  min_value x y h1 :=
sorry

end min_value_proven_l115_115214


namespace expand_expression_l115_115155

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115155


namespace jack_handing_in_amount_l115_115413

theorem jack_handing_in_amount :
  let total_100_bills := 2 * 100
  let total_50_bills := 1 * 50
  let total_20_bills := 5 * 20
  let total_10_bills := 3 * 10
  let total_5_bills := 7 * 5
  let total_1_bills := 27 * 1
  let total_notes := total_100_bills + total_50_bills + total_20_bills + total_10_bills + total_5_bills + total_1_bills
  let amount_in_till := 300
  let amount_to_hand_in := total_notes - amount_in_till
  amount_to_hand_in = 142 := by
  sorry

end jack_handing_in_amount_l115_115413


namespace increasing_interval_l115_115286

noncomputable def function_y (x : ℝ) : ℝ :=
  2 * (Real.logb (1/2) x) ^ 2 - 2 * Real.logb (1/2) x + 1

theorem increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ {y}, y ≥ x → function_y y ≥ function_y x) ↔ x ∈ Set.Ici (Real.sqrt 2 / 2) :=
by
  sorry

end increasing_interval_l115_115286


namespace quarters_count_l115_115851

noncomputable def num_coins := 12
noncomputable def total_value := 166 -- in cents
noncomputable def min_value := 1 + 5 + 10 + 25 + 50 -- minimum value from one of each type
noncomputable def remaining_value := total_value - min_value
noncomputable def remaining_coins := num_coins - 5

theorem quarters_count :
  ∀ (p n d q h : ℕ), 
  p + n + d + q + h = num_coins ∧
  p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ h ≥ 1 ∧
  (p + 5*n + 10*d + 25*q + 50*h = total_value) → 
  q = 3 := 
by 
  sorry

end quarters_count_l115_115851


namespace find_x_value_l115_115908

theorem find_x_value (x : ℝ) (h1 : Real.sin (π / 2 - x) = -Real.sqrt 3 / 2) (h2 : π < x ∧ x < 2 * π) : x = 7 * π / 6 :=
sorry

end find_x_value_l115_115908


namespace S_5_equals_31_l115_115380

-- Define the sequence sum function S
def S (n : Nat) : Nat := 2^n - 1

-- The theorem to prove that S(5) = 31
theorem S_5_equals_31 : S 5 = 31 :=
by
  rw [S]
  sorry

end S_5_equals_31_l115_115380


namespace emily_mean_seventh_score_l115_115744

theorem emily_mean_seventh_score :
  let a1 := 85
  let a2 := 88
  let a3 := 90
  let a4 := 94
  let a5 := 96
  let a6 := 92
  (a1 + a2 + a3 + a4 + a5 + a6 + a7) / 7 = 91 → a7 = 92 :=
by
  intros
  sorry

end emily_mean_seventh_score_l115_115744


namespace roots_quadratic_expression_l115_115224

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + m - 2023 = 0) (h2 : n^2 + n - 2023 = 0) :
  m^2 + 2 * m + n = 2022 :=
by
  -- proof steps would go here
  sorry

end roots_quadratic_expression_l115_115224


namespace length_of_chord_l115_115897

theorem length_of_chord (r AB : ℝ) (h1 : r = 6) (h2 : 0 < AB) (h3 : AB <= 2 * r) : AB ≠ 14 :=
by
  sorry

end length_of_chord_l115_115897


namespace chef_leftover_potatoes_l115_115328

-- Defining the conditions as variables
def fries_per_potato := 25
def total_potatoes := 15
def fries_needed := 200

-- Calculating the number of potatoes needed.
def potatoes_needed : ℕ :=
  fries_needed / fries_per_potato

-- Calculating the leftover potatoes.
def leftovers : ℕ :=
  total_potatoes - potatoes_needed

-- The theorem statement
theorem chef_leftover_potatoes :
  leftovers = 7 :=
by
  -- the actual proof is omitted.
  sorry

end chef_leftover_potatoes_l115_115328


namespace sum_of_gcd_and_lcm_of_180_and_4620_l115_115711

def gcd_180_4620 : ℕ := Nat.gcd 180 4620
def lcm_180_4620 : ℕ := Nat.lcm 180 4620
def sum_gcd_lcm_180_4620 : ℕ := gcd_180_4620 + lcm_180_4620

theorem sum_of_gcd_and_lcm_of_180_and_4620 :
  sum_gcd_lcm_180_4620 = 13920 :=
by
  sorry

end sum_of_gcd_and_lcm_of_180_and_4620_l115_115711


namespace joy_reading_rate_l115_115064

theorem joy_reading_rate
  (h1 : ∀ t: ℕ, t = 20 → ∀ p: ℕ, p = 8 → ∀ t': ℕ, t' = 60 → ∃ p': ℕ, p' = (p * t') / t)
  (h2 : ∀ t: ℕ, t = 5 * 60 → ∀ p: ℕ, p = 120):
  ∃ r: ℕ, r = 24 :=
by
  sorry

end joy_reading_rate_l115_115064


namespace rita_saving_l115_115435

theorem rita_saving
  (num_notebooks : ℕ)
  (price_per_notebook : ℝ)
  (discount_rate : ℝ) :
  num_notebooks = 7 →
  price_per_notebook = 3 →
  discount_rate = 0.15 →
  (num_notebooks * price_per_notebook) - (num_notebooks * (price_per_notebook * (1 - discount_rate))) = 3.15 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end rita_saving_l115_115435


namespace a_perpendicular_to_a_minus_b_l115_115779

def vector := ℝ × ℝ

def dot_product (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2

def a : vector := (-2, 1)
def b : vector := (-1, 3)

def a_minus_b : vector := (a.1 - b.1, a.2 - b.2) 

theorem a_perpendicular_to_a_minus_b : dot_product a a_minus_b = 0 := by
  sorry

end a_perpendicular_to_a_minus_b_l115_115779


namespace tan_subtraction_l115_115049

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 11) (h₂ : Real.tan β = 5) : 
  Real.tan (α - β) = 3 / 28 := 
  sorry

end tan_subtraction_l115_115049


namespace problem_l115_115377

def a (x : ℕ) : ℕ := 2005 * x + 2006
def b (x : ℕ) : ℕ := 2005 * x + 2007
def c (x : ℕ) : ℕ := 2005 * x + 2008

theorem problem (x : ℕ) : (a x)^2 + (b x)^2 + (c x)^2 - (a x) * (b x) - (a x) * (c x) - (b x) * (c x) = 3 :=
by sorry

end problem_l115_115377


namespace expected_value_of_fair_6_sided_die_l115_115845

noncomputable def fair_die_expected_value : ℝ :=
  (1/6) * 1 + (1/6) * 2 + (1/6) * 3 + (1/6) * 4 + (1/6) * 5 + (1/6) * 6

theorem expected_value_of_fair_6_sided_die : fair_die_expected_value = 3.5 := by
  sorry

end expected_value_of_fair_6_sided_die_l115_115845


namespace relation_among_a_b_c_l115_115808

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 3
noncomputable def c : ℝ := Real.sqrt 6 - Real.sqrt 2

theorem relation_among_a_b_c : a > c ∧ c > b :=
by {
  sorry
}

end relation_among_a_b_c_l115_115808


namespace width_of_field_l115_115119

-- Definitions for the conditions
variables (W L : ℝ) (P : ℝ)
axiom length_condition : L = (7 / 5) * W
axiom perimeter_condition : P = 2 * L + 2 * W
axiom perimeter_value : P = 336

-- Theorem to be proved
theorem width_of_field : W = 70 :=
by
  -- Here will be the proof body
  sorry

end width_of_field_l115_115119


namespace autumn_pencils_l115_115877

-- Define the conditions of the problem.
def initial_pencils := 20
def misplaced_pencils := 7
def broken_pencils := 3
def found_pencils := 4
def bought_pencils := 2

-- Define the number of pencils lost and gained.
def pencils_lost := misplaced_pencils + broken_pencils
def pencils_gained := found_pencils + bought_pencils

-- Define the final number of pencils.
def final_pencils := initial_pencils - pencils_lost + pencils_gained

-- The theorem we want to prove.
theorem autumn_pencils : final_pencils = 16 := by
  sorry

end autumn_pencils_l115_115877


namespace triple_complement_angle_l115_115680

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l115_115680


namespace intersection_A_B_l115_115537

-- Define sets A and B according to the conditions provided
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Define the theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l115_115537


namespace inverse_of_square_positive_is_negative_l115_115831

variable {x : ℝ}

-- Original proposition: ∀ x, x < 0 → x^2 > 0
def original_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 > 0

-- Inverse proposition to be proven: ∀ x, x^2 > 0 → x < 0
def inverse_proposition (x : ℝ) : Prop :=
  x^2 > 0 → x < 0

theorem inverse_of_square_positive_is_negative :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ x : ℝ, x^2 > 0 → x < 0) :=
  sorry

end inverse_of_square_positive_is_negative_l115_115831


namespace completing_the_square_transformation_l115_115111

theorem completing_the_square_transformation (x : ℝ) : 
  (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro h
  -- Transformation steps will be shown in the proof
  sorry

end completing_the_square_transformation_l115_115111


namespace roots_of_f_non_roots_of_g_l115_115492

-- Part (a)

def f (x : ℚ) := x^20 - 123 * x^10 + 1

theorem roots_of_f (a : ℚ) (h : f a = 0) : 
  f (-a) = 0 ∧ f (1/a) = 0 ∧ f (-1/a) = 0 :=
by
  sorry

-- Part (b)

def g (x : ℚ) := x^4 + 3 * x^3 + 4 * x^2 + 2 * x + 1

theorem non_roots_of_g (β : ℚ) (h : g β = 0) : 
  g (-β) ≠ 0 ∧ g (1/β) ≠ 0 ∧ g (-1/β) ≠ 0 :=
by
  sorry

end roots_of_f_non_roots_of_g_l115_115492


namespace neg_P_is_univ_l115_115382

noncomputable def P : Prop :=
  ∃ x0 : ℝ, x0^2 + 2 * x0 + 2 ≤ 0

theorem neg_P_is_univ :
  ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by {
  sorry
}

end neg_P_is_univ_l115_115382


namespace expand_expression_l115_115154

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115154


namespace angle_triple_complement_l115_115663

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l115_115663


namespace compounding_frequency_l115_115830

variable (i : ℝ) (EAR : ℝ)

/-- Given the nominal annual rate (i = 6%) and the effective annual rate (EAR = 6.09%), 
    prove that the frequency of payment (n) is 4. -/
theorem compounding_frequency (h1 : i = 0.06) (h2 : EAR = 0.0609) : 
  ∃ n : ℕ, (1 + i / n)^n - 1 = EAR ∧ n = 4 := sorry

end compounding_frequency_l115_115830


namespace expand_expression_l115_115170

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l115_115170


namespace simplify_expression_l115_115436

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l115_115436


namespace find_number_l115_115094

theorem find_number (x : ℝ) : (10 * x = 2 * x - 36) → (x = -4.5) :=
begin
  intro h,
  sorry
end

end find_number_l115_115094


namespace angle_triple_complement_l115_115671

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l115_115671


namespace depth_of_well_l115_115870

theorem depth_of_well 
  (t1 t2 : ℝ) 
  (d : ℝ) 
  (h1: t1 + t2 = 8) 
  (h2: d = 32 * t1^2) 
  (h3: t2 = d / 1100) 
  : d = 1348 := 
  sorry

end depth_of_well_l115_115870


namespace charles_paints_l115_115872

-- Define the ratio and total work conditions
def ratio_a_to_c (a c : ℕ) := a * 6 = c * 2

def total_work (total : ℕ) := total = 320

-- Define the question, i.e., the amount of work Charles does
theorem charles_paints (a c total : ℕ) (h_ratio : ratio_a_to_c a c) (h_total : total_work total) : 
  (total / (a + c)) * c = 240 :=
by 
  -- We include sorry to indicate the need for proof here
  sorry

end charles_paints_l115_115872


namespace periodic_sequence_implies_integer_l115_115071

open Real

theorem periodic_sequence_implies_integer (α : ℝ) (hα : α > 1) (T : ℕ) (hT : 0 < T) 
  (a : ℕ → ℝ) (h_seq : ∀ n : ℕ, a n = α * floor (α ^ n) - floor (α ^ (n + 1))) 
  (h_periodic : ∀ n : ℕ, a (n + T) = a n) : α ∈ ℤ := 
sorry

end periodic_sequence_implies_integer_l115_115071


namespace angle_triple_complement_l115_115700

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l115_115700


namespace bob_age_is_eleven_l115_115826

/-- 
Susan, Arthur, Tom, and Bob are siblings. Arthur is 2 years older than Susan, 
Tom is 3 years younger than Bob. Susan is 15 years old, 
and the total age of all four family members is 51 years. 
This theorem states that Bob is 11 years old.
-/

theorem bob_age_is_eleven
  (S A T B : ℕ)
  (h1 : A = S + 2)
  (h2 : T = B - 3)
  (h3 : S = 15)
  (h4 : S + A + T + B = 51) : 
  B = 11 :=
  sorry

end bob_age_is_eleven_l115_115826


namespace triangle_problem_l115_115919

-- Define a triangle with given parameters and properties
variables {A B C : ℝ}
variables {a b c : ℝ} (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C) 
variables (h_b2a : b = 2 * a)
variables (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3)

-- Prove the required angles and side length
theorem triangle_problem 
    (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
    (h_b2a : b = 2 * a)
    (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) :

    Real.cos C = -1/2 ∧ C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := 
by 
  sorry

end triangle_problem_l115_115919


namespace inequality_proof_l115_115910

noncomputable def a := (1.01: ℝ) ^ (0.5: ℝ)
noncomputable def b := (1.01: ℝ) ^ (0.6: ℝ)
noncomputable def c := (0.6: ℝ) ^ (0.5: ℝ)

theorem inequality_proof : b > a ∧ a > c := 
by
  sorry

end inequality_proof_l115_115910


namespace percent_daffodils_is_57_l115_115860

-- Condition 1: Four-sevenths of the flowers are yellow
def fraction_yellow : ℚ := 4 / 7

-- Condition 2: Two-thirds of the red flowers are daffodils
def fraction_red_daffodils_given_red : ℚ := 2 / 3

-- Condition 3: Half of the yellow flowers are tulips
def fraction_yellow_tulips_given_yellow : ℚ := 1 / 2

-- Calculate fractions of yellow and red flowers
def fraction_red : ℚ := 1 - fraction_yellow

-- Calculate fractions of daffodils
def fraction_yellow_daffodils : ℚ := fraction_yellow * (1 - fraction_yellow_tulips_given_yellow)
def fraction_red_daffodils : ℚ := fraction_red * fraction_red_daffodils_given_red

-- Total fraction of daffodils
def fraction_daffodils : ℚ := fraction_yellow_daffodils + fraction_red_daffodils

-- Proof statement
theorem percent_daffodils_is_57 :
  fraction_daffodils * 100 = 57 := by
  sorry

end percent_daffodils_is_57_l115_115860


namespace small_cube_edge_length_l115_115105

theorem small_cube_edge_length 
  (m n : ℕ)
  (h1 : 12 % m = 0) 
  (h2 : n = 12 / m) 
  (h3 : 6 * (n - 2)^2 = 12 * (n - 2)) 
  : m = 3 :=
by 
  sorry

end small_cube_edge_length_l115_115105


namespace expand_product_l115_115161

open Polynomial

-- Define the polynomial expressions
def poly1 := (X - 3) * (X + 3)
def poly2 := (X^2 + 9)
def expanded_poly := poly1 * poly2

theorem expand_product : expanded_poly = X^4 - 81 := by
  sorry

end expand_product_l115_115161


namespace fraction_denominator_l115_115055

theorem fraction_denominator (x y Z : ℚ) (h : x / y = 7 / 3) (h2 : (x + y) / Z = 2.5) :
    Z = (4 * y) / 3 :=
by sorry

end fraction_denominator_l115_115055


namespace probability_at_least_two_worth_visiting_l115_115582

theorem probability_at_least_two_worth_visiting :
  let total_caves := 8
  let worth_visiting := 3
  let select_caves := 4
  let worth_select_2 := Nat.choose worth_visiting 2 * Nat.choose (total_caves - worth_visiting) 2
  let worth_select_3 := Nat.choose worth_visiting 3 * Nat.choose (total_caves - worth_visiting) 1
  let total_select := Nat.choose total_caves select_caves
  let probability := (worth_select_2 + worth_select_3) / total_select
  probability = 1 / 2 := sorry

end probability_at_least_two_worth_visiting_l115_115582


namespace find_f2_l115_115385

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 2 = 1 := 
by
  sorry

end find_f2_l115_115385


namespace domain_of_sqrt_tan_minus_one_l115_115455

open Real
open Set

def domain_sqrt_tan_minus_one : Set ℝ := 
  ⋃ k : ℤ, Ico (π/4 + k * π) (π/2 + k * π)

theorem domain_of_sqrt_tan_minus_one :
  {x : ℝ | ∃ y : ℝ, y = sqrt (tan x - 1)} = domain_sqrt_tan_minus_one :=
sorry

end domain_of_sqrt_tan_minus_one_l115_115455


namespace xy_expr_value_l115_115756

variable (x y : ℝ)

-- Conditions
def cond1 : Prop := x - y = 2
def cond2 : Prop := x * y = 3

-- Statement to prove
theorem xy_expr_value (h1 : cond1 x y) (h2 : cond2 x y) : x * y^2 - x^2 * y = -6 :=
by
  sorry

end xy_expr_value_l115_115756


namespace angle_measure_triple_complement_l115_115618

theorem angle_measure_triple_complement (x : ℝ) : 
  (∀ y : ℝ, x = 3 * y ∧ x + y = 90) → x = 67.5 :=
by 
  intro h
  obtain ⟨y, h1, h2⟩ := h y
  sorry

end angle_measure_triple_complement_l115_115618


namespace total_amount_spent_l115_115259

-- Define the prices related to John's Star Wars toy collection
def other_toys_cost : ℕ := 1000
def lightsaber_cost : ℕ := 2 * other_toys_cost

-- Problem statement in Lean: Prove the total amount spent is $3000
theorem total_amount_spent : (other_toys_cost + lightsaber_cost) = 3000 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end total_amount_spent_l115_115259


namespace angle_measure_triple_complement_l115_115656

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l115_115656


namespace race_cars_count_l115_115249

theorem race_cars_count:
  (1 / 7 + 1 / 3 + 1 / 5 = 0.6761904761904762) -> 
  (∀ N : ℕ, (1 / N = 1 / 7 ∨ 1 / N = 1 / 3 ∨ 1 / N = 1 / 5)) -> 
  (1 / 105 = 0.6761904761904762) :=
by
  intro h_sum_probs h_indiv_probs
  sorry

end race_cars_count_l115_115249


namespace monomial_properties_l115_115085

def coefficient (c : ℝ) (x y : ℝ) (k l : ℕ) (m : ℝ) := c = -π / 5
def degree (k l : ℕ) := k = 3 ∧ l = 2 ∧ k + l = 5

theorem monomial_properties : 
  coefficient (-π / 5) x y 3 2 (-π * x^3 * y^2 / 5) ∧ degree 3 2 := 
by 
  sorry

end monomial_properties_l115_115085


namespace fraction_simplification_l115_115736

theorem fraction_simplification (x : ℝ) (h: x ≠ 1) : (5 * x / (x - 1) - 5 / (x - 1)) = 5 := 
sorry

end fraction_simplification_l115_115736


namespace coffee_last_days_l115_115998

theorem coffee_last_days (coffee_weight : ℕ) (cups_per_lb : ℕ) (angie_daily : ℕ) (bob_daily : ℕ) (carol_daily : ℕ) 
  (angie_coffee_weight : coffee_weight = 3) (cups_brewing_rate : cups_per_lb = 40)
  (angie_consumption : angie_daily = 3) (bob_consumption : bob_daily = 2) (carol_consumption : carol_daily = 4) : 
  ((coffee_weight * cups_per_lb) / (angie_daily + bob_daily + carol_daily) = 13) := by
  sorry

end coffee_last_days_l115_115998


namespace product_divisible_by_six_l115_115463

theorem product_divisible_by_six (a : ℤ) : 6 ∣ a * (a + 1) * (2 * a + 1) := 
sorry

end product_divisible_by_six_l115_115463


namespace min_value_at_zero_max_value_a_l115_115809

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - (a * x / (x + 1))

-- Part (I)
theorem min_value_at_zero {a : ℝ} (h : ∀ x, f x a ≥ f 0 a) : a = 1 :=
sorry

-- Part (II)
theorem max_value_a (h : ∀ x > 0, f x a > 0) : a ≤ 1 :=
sorry

end min_value_at_zero_max_value_a_l115_115809


namespace x_intercept_of_perpendicular_line_l115_115310

theorem x_intercept_of_perpendicular_line (x y : ℝ) (b : ℕ) :
  let line1 := 2 * x + 3 * y
  let slope1 := -2/3
  let slope2 := 3/2
  let y_intercept := -1
  let perp_line := slope2 * x + y_intercept
  let x_intercept := 2/3
  line1 = 12 → perp_line = 0 → x = x_intercept :=
by
  sorry

end x_intercept_of_perpendicular_line_l115_115310


namespace initial_books_l115_115100

theorem initial_books (total_books_now : ℕ) (books_added : ℕ) (initial_books : ℕ) :
  total_books_now = 48 → books_added = 10 → initial_books = total_books_now - books_added → initial_books = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_books_l115_115100


namespace ethanol_in_tank_l115_115996

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

end ethanol_in_tank_l115_115996


namespace expand_expression_l115_115169

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l115_115169


namespace quadratic_root_a_l115_115029

theorem quadratic_root_a {a : ℝ} (h : (2 : ℝ) ∈ {x : ℝ | x^2 + 3 * x + a = 0}) : a = -10 :=
by
  sorry

end quadratic_root_a_l115_115029


namespace simplify_polynomial_expression_l115_115822

variable {R : Type*} [CommRing R]

theorem simplify_polynomial_expression (x : R) :
  (2 * x^6 + 3 * x^5 + 4 * x^4 + x^3 + x^2 + x + 20) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 2 * x^2 + 5) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - x^2 + 15 := 
by
  sorry

end simplify_polynomial_expression_l115_115822


namespace part1_part2_l115_115531

section
variable (x y : ℝ)

def A : ℝ := 3 * x^2 + 2 * y^2 - 2 * x * y
def B : ℝ := y^2 - x * y + 2 * x^2

-- Part (1): Prove that 2A - 3B = y^2 - xy
theorem part1 : 2 * A x y - 3 * B x y = y^2 - x * y := 
sorry

-- Part (2): Given |2x - 3| + (y + 2)^2 = 0, prove that 2A - 3B = 7
theorem part2 (h : |2 * x - 3| + (y + 2)^2 = 0) : 2 * A x y - 3 * B x y = 7 :=
sorry

end

end part1_part2_l115_115531


namespace head_start_proofs_l115_115787

def HeadStartAtoB : ℕ := 150
def HeadStartAtoC : ℕ := 310
def HeadStartAtoD : ℕ := 400

def HeadStartBtoC : ℕ := HeadStartAtoC - HeadStartAtoB
def HeadStartCtoD : ℕ := HeadStartAtoD - HeadStartAtoC
def HeadStartBtoD : ℕ := HeadStartAtoD - HeadStartAtoB

theorem head_start_proofs :
  (HeadStartBtoC = 160) ∧
  (HeadStartCtoD = 90) ∧
  (HeadStartBtoD = 250) :=
by
  sorry

end head_start_proofs_l115_115787


namespace fuel_tank_capacity_l115_115507

theorem fuel_tank_capacity (C : ℝ) 
  (h1 : 0.12 * 98 + 0.16 * (C - 98) = 30) : 
  C = 212 :=
by
  sorry

end fuel_tank_capacity_l115_115507


namespace simplify_expression_l115_115446

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l115_115446


namespace log2_bounds_158489_l115_115606

theorem log2_bounds_158489 :
  (2^16 = 65536) ∧ (2^17 = 131072) ∧ (65536 < 158489 ∧ 158489 < 131072) →
  (16 < Real.log 158489 / Real.log 2 ∧ Real.log 158489 / Real.log 2 < 17) ∧ 16 + 17 = 33 :=
by
  intro h
  have h1 : 2^16 = 65536 := h.1
  have h2 : 2^17 = 131072 := h.2.1
  have h3 : 65536 < 158489 := h.2.2.1
  have h4 : 158489 < 131072 := h.2.2.2
  sorry

end log2_bounds_158489_l115_115606


namespace midpoint_fraction_l115_115963

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (a + b) / 2 = 19/24 := by
  sorry

end midpoint_fraction_l115_115963


namespace sufficient_but_not_necessary_condition_l115_115122

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (1 ≤ x ∧ x ≤ 4) ↔ (1 ≤ x^2 ∧ x^2 ≤ 16) :=
by
  sorry

end sufficient_but_not_necessary_condition_l115_115122


namespace angle_triple_complement_l115_115634

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l115_115634


namespace expand_polynomial_eq_l115_115189

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l115_115189


namespace exists_x_y_not_divisible_by_3_l115_115812

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (x^2 + 2 * y^2 = 3^k) ∧ (¬ (x % 3 = 0)) ∧ (¬ (y % 3 = 0)) :=
sorry

end exists_x_y_not_divisible_by_3_l115_115812


namespace proposition_equivalence_l115_115712

open Classical

variable {α : Type*}
variables (P : Set α)

theorem proposition_equivalence:
  (∀ a b, a ∈ P → b ∉ P) ↔ (∀ a b, b ∈ P → a ∉ P) :=
by intros a b; sorry

end proposition_equivalence_l115_115712


namespace terminal_side_angles_l115_115848

theorem terminal_side_angles (k : ℤ) (β : ℝ) :
  β = (Real.pi / 3) + 2 * k * Real.pi → -2 * Real.pi ≤ β ∧ β < 4 * Real.pi :=
by
  sorry

end terminal_side_angles_l115_115848


namespace wall_width_l115_115953

theorem wall_width (V h l w : ℝ) (h_cond : h = 6 * w) (l_cond : l = 42 * w) (vol_cond : 252 * w^3 = 129024) : w = 8 := 
by
  -- Proof is omitted; required to produce lean statement only
  sorry

end wall_width_l115_115953


namespace Jack_hands_in_l115_115416

def num_hundred_bills := 2
def num_fifty_bills := 1
def num_twenty_bills := 5
def num_ten_bills := 3
def num_five_bills := 7
def num_one_bills := 27
def to_leave_in_till := 300

def total_money_in_notes : Nat :=
  (num_hundred_bills * 100) +
  (num_fifty_bills * 50) +
  (num_twenty_bills * 20) +
  (num_ten_bills * 10) +
  (num_five_bills * 5) +
  (num_one_bills * 1)

def money_to_hand_in := total_money_in_notes - to_leave_in_till

theorem Jack_hands_in : money_to_hand_in = 142 := by
  sorry

end Jack_hands_in_l115_115416


namespace max_prob_of_one_six_l115_115212

theorem max_prob_of_one_six (n : ℕ) : 
  let V (x : ℕ) := (x * 5^(x - 1) : ℚ) / 6^x in
  V n ≤ V 5 ∨ V n ≤ V 6 :=
by
  sorry

end max_prob_of_one_six_l115_115212


namespace crocus_bulb_cost_l115_115716

theorem crocus_bulb_cost 
  (space_bulbs : ℕ)
  (crocus_bulbs : ℕ)
  (cost_daffodil_bulb : ℝ)
  (budget : ℝ)
  (purchased_crocus_bulbs : ℕ)
  (total_cost : ℝ)
  (c : ℝ)
  (h_space : space_bulbs = 55)
  (h_cost_daffodil : cost_daffodil_bulb = 0.65)
  (h_budget : budget = 29.15)
  (h_purchased_crocus : purchased_crocus_bulbs = 22)
  (h_total_cost_eq : total_cost = (33:ℕ) * cost_daffodil_bulb)
  (h_eqn : (purchased_crocus_bulbs : ℝ) * c + total_cost = budget) :
  c = 0.35 :=
by 
  sorry

end crocus_bulb_cost_l115_115716


namespace quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l115_115767

-- Case 1
theorem quadratic_function_expression 
  (a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = 3) : 
  by {exact (a = -2 ∧ b = 3)} := sorry

theorem quadratic_function_range 
  (x : ℝ) 
  (h : -1 ≤ x ∧ x ≤ 2) : 
  (-3 ≤ -2*x^2 + 3*x + 2 ∧ -2*x^2 + 3*x + 2 ≤ 25/8) := sorry

-- Case 2
theorem quadratic_function_m_range 
  (m a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = m) 
  (h₃ : a > 0) : 
  m < 1 := sorry

end quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l115_115767


namespace f_f_of_2_l115_115811

def f (x : ℤ) : ℤ := 4 * x ^ 3 - 3 * x + 1

theorem f_f_of_2 : f (f 2) = 78652 := 
by
  sorry

end f_f_of_2_l115_115811


namespace distinct_configurations_l115_115737

/-- 
Define m, n, and the binomial coefficient function.
conditions:
  - integer grid dimensions m and n with m >= 1, n >= 1.
  - initially (m-1)(n-1) coins in the subgrid of size (m-1) x (n-1).
  - legal move conditions for coins.
question:
  - Prove the number of distinct configurations of coins equals the binomial coefficient.
-/
def number_of_distinct_configurations (m n : ℕ) : ℕ :=
  Nat.choose (m + n - 2) (m - 1)

theorem distinct_configurations (m n : ℕ) (h_m : 1 ≤ m) (h_n : 1 ≤ n) :
  number_of_distinct_configurations m n = Nat.choose (m + n - 2) (m - 1) :=
sorry

end distinct_configurations_l115_115737


namespace min_value_expr_l115_115343

theorem min_value_expr (x : ℝ) (h : x > 1) : ∃ m, m = 5 ∧ ∀ y, y = x + 4 / (x - 1) → y ≥ m :=
by
  sorry

end min_value_expr_l115_115343


namespace solve_dfrac_eq_l115_115389

theorem solve_dfrac_eq (x : ℝ) (h : (x / 5) / 3 = 3 / (x / 5)) : x = 15 ∨ x = -15 := by
  sorry

end solve_dfrac_eq_l115_115389


namespace angle_is_67_l115_115638

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l115_115638


namespace Bill_threw_more_sticks_l115_115009

-- Definitions based on the given conditions
def Ted_sticks : ℕ := 10
def Ted_rocks : ℕ := 10
def Ted_double_Bill_rocks (R : ℕ) : Prop := Ted_rocks = 2 * R
def Bill_total_objects (S R : ℕ) : Prop := S + R = 21

-- The theorem stating Bill throws 6 more sticks than Ted
theorem Bill_threw_more_sticks (S R : ℕ) (h1 : Ted_double_Bill_rocks R) (h2 : Bill_total_objects S R) : S - Ted_sticks = 6 :=
by
  -- Definitions and conditions are loaded here
  sorry

end Bill_threw_more_sticks_l115_115009


namespace binom_10_1_eq_10_l115_115013

theorem binom_10_1_eq_10 : Nat.choose 10 1 = 10 := by
  sorry

end binom_10_1_eq_10_l115_115013


namespace inequality_always_true_l115_115216

variable (a b c : ℝ)

theorem inequality_always_true (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end inequality_always_true_l115_115216


namespace angle_measure_triple_complement_l115_115659

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l115_115659


namespace intersection_A_B_l115_115536

-- Define sets A and B according to the conditions provided
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Define the theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l115_115536


namespace ones_digit_of_9_pow_27_l115_115964

-- Definitions representing the cyclical pattern
def ones_digit_of_9_power (n : ℕ) : ℕ :=
  if n % 2 = 1 then 9 else 1

-- The problem statement to be proven
theorem ones_digit_of_9_pow_27 : ones_digit_of_9_power 27 = 9 := 
by
  -- the detailed proof steps are omitted
  sorry

end ones_digit_of_9_pow_27_l115_115964


namespace triple_complement_angle_l115_115679

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l115_115679


namespace coin_probability_l115_115465

theorem coin_probability (a r : ℝ) (h : r < a / 2) :
  let favorable_cells := 3
  let larger_cell_area := 9 * a^2
  let favorable_area_per_cell := (a - 2 * r)^2
  let favorable_area := favorable_cells * favorable_area_per_cell
  let probability := favorable_area / larger_cell_area
  probability = (a - 2 * r)^2 / (3 * a^2) :=
by
  sorry

end coin_probability_l115_115465


namespace find_number_1920_find_number_60_l115_115601

theorem find_number_1920 : 320 * 6 = 1920 :=
by sorry

theorem find_number_60 : (1920 / 7 = 60) :=
by sorry

end find_number_1920_find_number_60_l115_115601


namespace angle_triple_complement_l115_115677

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l115_115677


namespace angle_measure_triple_complement_l115_115661

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l115_115661


namespace intersection_points_count_l115_115580

theorem intersection_points_count (A : ℝ) (hA : A > 0) :
  ((A > 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y) ∧
                              (x ≠ 0 ∨ y ≠ 0)) ∧
  ((A ≤ 1 / 4) → ∃! (x y : ℝ), (y = A * x^2) ∧ (x^2 + y^2 = 4 * y)) :=
by
  sorry

end intersection_points_count_l115_115580


namespace min_value_frac_l115_115220

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (c : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → c ≤ 8 / x + 2 / y) ∧ c = 18 :=
sorry

end min_value_frac_l115_115220


namespace max_and_next_max_values_l115_115206

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log a) / b

theorem max_and_next_max_values :
  let values := [4.0^(1/4), 5.0^(1/5), 16.0^(1/16), 25.0^(1/25)]
  ∃ max2 max1, 
    max1 = 4.0^(1/4) ∧ max2 = 5.0^(1/5) ∧ 
    (∀ x ∈ values, x <= max1) ∧ 
    (∀ x ∈ values, x < max1 → x <= max2) :=
by
  sorry

end max_and_next_max_values_l115_115206


namespace nearest_integer_pow_l115_115314

noncomputable def nearest_integer_to_power : ℤ := 
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_pow : nearest_integer_to_power = 7414 := 
  by
    unfold nearest_integer_to_power
    sorry -- Proof skipped

end nearest_integer_pow_l115_115314


namespace largest_prime_divisor_of_360_is_5_l115_115034

theorem largest_prime_divisor_of_360_is_5 (p : ℕ) (hp₁ : Nat.Prime p) (hp₂ : p ∣ 360) : p ≤ 5 :=
by 
sorry

end largest_prime_divisor_of_360_is_5_l115_115034


namespace range_of_x_l115_115517

-- Define the ceiling function for ease of use.
noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem range_of_x (x : ℝ) (h1 : ceil (2 * x + 1) = 5) (h2 : ceil (2 - 3 * x) = -3) :
  (5 / 3 : ℝ) ≤ x ∧ x < 2 :=
by
  sorry

end range_of_x_l115_115517


namespace tenth_day_is_monday_l115_115594

theorem tenth_day_is_monday (runs_20_mins : ∀ d ∈ [1, 7], d = 1 ∨ d = 6 ∨ d = 7 → True)
                            (total_minutes : 5 * 60 = 300)
                            (first_day_is_saturday : 1 = 6) :
   (10 % 7 = 3) :=
by
  sorry

end tenth_day_is_monday_l115_115594


namespace angle_in_fourth_quadrant_l115_115720

theorem angle_in_fourth_quadrant (θ : ℝ) (hθ : θ = 300) : 270 < θ ∧ θ < 360 :=
by
  -- theta equals 300
  have h1 : θ = 300 := hθ
  -- check that 300 degrees lies between 270 and 360
  sorry

end angle_in_fourth_quadrant_l115_115720


namespace find_p_l115_115852

variable (m n p : ℚ)

theorem find_p (h1 : m = 8 * n + 5) (h2 : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by
  sorry

end find_p_l115_115852


namespace percy_swimming_hours_l115_115271

theorem percy_swimming_hours :
  let weekday_hours_per_day := 2
  let weekdays := 5
  let weekend_hours := 3
  let weeks := 4
  let total_weekday_hours_per_week := weekday_hours_per_day * weekdays
  let total_weekend_hours_per_week := weekend_hours
  let total_hours_per_week := total_weekday_hours_per_week + total_weekend_hours_per_week
  let total_hours_over_weeks := total_hours_per_week * weeks
  total_hours_over_weeks = 64 :=
by
  sorry

end percy_swimming_hours_l115_115271


namespace rectangle_width_to_length_ratio_l115_115252

theorem rectangle_width_to_length_ratio {w : ℕ} 
  (h1 : ∀ (l : ℕ), l = 10)
  (h2 : ∀ (p : ℕ), p = 32)
  (h3 : ∀ (P : ℕ), P = 2 * 10 + 2 * w) :
  (w : ℚ) / 10 = 3 / 5 :=
by
  sorry

end rectangle_width_to_length_ratio_l115_115252


namespace num_ways_to_factor_2210_l115_115551

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_ways_to_factor_2210 : ∃! (a b : ℕ), a * b = 2210 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end num_ways_to_factor_2210_l115_115551


namespace evaluate_expression_l115_115390

theorem evaluate_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(2*x)) / (y^(2*y) * x^(2*x)) = (x / y)^(2 * (y - x)) :=
by
  sorry

end evaluate_expression_l115_115390


namespace symmetry_xOz_A_l115_115792

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetry_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y , z := p.z }

theorem symmetry_xOz_A :
  let A := Point3D.mk 2 (-3) 1
  symmetry_xOz A = Point3D.mk 2 3 1 :=
by
  sorry

end symmetry_xOz_A_l115_115792


namespace value_of_y_l115_115238

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 5) (h2 : x = 7) : y = 54 := by
  sorry

end value_of_y_l115_115238


namespace inequality_always_true_l115_115215

variable (a b c : ℝ)

theorem inequality_always_true (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end inequality_always_true_l115_115215


namespace divisible_by_27_l115_115279

theorem divisible_by_27 (n : ℕ) : 27 ∣ (2^(5*n+1) + 5^(n+2)) :=
by
  sorry

end divisible_by_27_l115_115279


namespace jack_handing_in_amount_l115_115414

theorem jack_handing_in_amount :
  let total_100_bills := 2 * 100
  let total_50_bills := 1 * 50
  let total_20_bills := 5 * 20
  let total_10_bills := 3 * 10
  let total_5_bills := 7 * 5
  let total_1_bills := 27 * 1
  let total_notes := total_100_bills + total_50_bills + total_20_bills + total_10_bills + total_5_bills + total_1_bills
  let amount_in_till := 300
  let amount_to_hand_in := total_notes - amount_in_till
  amount_to_hand_in = 142 := by
  sorry

end jack_handing_in_amount_l115_115414


namespace range_of_a_l115_115242

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x + 4 ≥ 0) ↔ (2 ≤ a ∧ a ≤ 6) := 
sorry

end range_of_a_l115_115242


namespace parallelogram_sides_l115_115076

theorem parallelogram_sides (a b : ℕ): 
  (a = 3 * b) ∧ (2 * a + 2 * b = 24) → (a = 9) ∧ (b = 3) :=
by
  sorry

end parallelogram_sides_l115_115076


namespace max_f_value_range_of_a_l115_115548

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - 4| - a

theorem max_f_value (a : ℝ) : ∃ x, f x a = 5 - a :=
sorry

theorem range_of_a (a : ℝ) : (∃ x, f x a ≥ (4 / a) + 1) ↔ (a = 2 ∨ a < 0) :=
sorry

end max_f_value_range_of_a_l115_115548


namespace angle_is_67_l115_115641

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l115_115641


namespace leo_weight_proof_l115_115782

def Leo_s_current_weight (L K : ℝ) := 
  L + 10 = 1.5 * K ∧ L + K = 170 → L = 98

theorem leo_weight_proof : ∀ (L K : ℝ), L + 10 = 1.5 * K ∧ L + K = 170 → L = 98 := 
by 
  intros L K h
  sorry

end leo_weight_proof_l115_115782


namespace find_distinct_natural_numbers_l115_115613

theorem find_distinct_natural_numbers :
  ∃ (x y : ℕ), x ≥ 10 ∧ y ≠ 1 ∧
  (x * y + x) + (x * y - x) + (x * y * x) + (x * y / x) = 576 :=
by
  sorry

end find_distinct_natural_numbers_l115_115613


namespace tom_buys_oranges_l115_115431

theorem tom_buys_oranges (o a : ℕ) (h₁ : o + a = 7) (h₂ : (90 * o + 60 * a) % 100 = 0) : o = 6 := 
by 
  sorry

end tom_buys_oranges_l115_115431


namespace floor_ceil_sum_l115_115297

theorem floor_ceil_sum (x : ℝ) (h : Int.floor x + Int.ceil x = 7) : x ∈ { x : ℝ | 3 < x ∧ x < 4 } ∪ {3.5} :=
sorry

end floor_ceil_sum_l115_115297


namespace correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l115_115317

theorem correct_division (x : ℝ) : x^6 / x^3 = x^3 := by 
  sorry

theorem incorrect_addition (x : ℝ) : ¬(x^2 + x^3 = 2 * x^5) := by 
  sorry

theorem incorrect_multiplication (x : ℝ) : ¬(x^2 * x^3 = x^6) := by 
  sorry

theorem incorrect_squaring (x : ℝ) : ¬((-x^3) ^ 2 = -x^6) := by 
  sorry

theorem only_correct_operation (x : ℝ) : 
  (x^6 / x^3 = x^3) ∧ ¬(x^2 + x^3 = 2 * x^5) ∧ ¬(x^2 * x^3 = x^6) ∧ ¬((-x^3) ^ 2 = -x^6) := 
  by
    exact ⟨correct_division x, incorrect_addition x, incorrect_multiplication x,
           incorrect_squaring x⟩

end correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l115_115317


namespace expand_expression_l115_115149

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115149


namespace probability_both_red_l115_115973

-- Definitions for the problem conditions
def total_balls := 16
def red_balls := 7
def blue_balls := 5
def green_balls := 4
def first_red_prob := (red_balls : ℚ) / total_balls
def second_red_given_first_red_prob := (red_balls - 1 : ℚ) / (total_balls - 1)

-- The statement to be proved
theorem probability_both_red : (first_red_prob * second_red_given_first_red_prob) = (7 : ℚ) / 40 :=
by 
  -- Proof goes here
  sorry

end probability_both_red_l115_115973


namespace largest_n_l115_115875

noncomputable def a (n : ℕ) (x : ℤ) : ℤ := 2 + (n - 1) * x
noncomputable def b (n : ℕ) (y : ℤ) : ℤ := 3 + (n - 1) * y

theorem largest_n {n : ℕ} (x y : ℤ) :
  a 1 x = 2 ∧ b 1 y = 3 ∧ 3 * a 2 x < 2 * b 2 y ∧ a n x * b n y = 4032 →
  n = 367 :=
sorry

end largest_n_l115_115875


namespace integral_curve_has_inflection_points_l115_115612

theorem integral_curve_has_inflection_points (x y : ℝ) (f : ℝ → ℝ → ℝ) :
  f x y = y - x^2 + 2*x - 2 →
  (∃ y' y'' : ℝ, y' = f x y ∧ y'' = y - x^2 ∧ y'' = 0) ↔ y = x^2 :=
by
  sorry

end integral_curve_has_inflection_points_l115_115612


namespace white_balls_in_bag_l115_115101

theorem white_balls_in_bag : 
  ∀ x : ℕ, (3 + 2 + x ≠ 0) → (2 : ℚ) / (3 + 2 + x) = 1 / 4 → x = 3 :=
by
  intro x
  intro h1
  intro h2
  sorry

end white_balls_in_bag_l115_115101


namespace average_value_of_x_l115_115091

theorem average_value_of_x
  (x : ℝ)
  (h : (5 + 5 + x + 6 + 8) / 5 = 6) :
  x = 6 :=
sorry

end average_value_of_x_l115_115091


namespace parallel_lines_sufficient_not_necessary_l115_115453

theorem parallel_lines_sufficient_not_necessary (a : ℝ) :
  ((a = 3) → (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) → (3 * x + (a - 1) * y - 2 = 0)) ∧ 
  (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) ∧ (3 * x + (a - 1) * y - 2 = 0) → (a = 3 ∨ a = -2))) :=
sorry

end parallel_lines_sufficient_not_necessary_l115_115453


namespace num_distinguishable_large_triangles_l115_115470

-- Given conditions
constant num_colors : ℕ := 8
constant num_corner_triangles : ℕ := 3
constant num_total_positions : ℕ := 4

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Problem Statement
theorem num_distinguishable_large_triangles :
  let corner_color_combinations := binomial num_colors num_corner_triangles
  let corner_color_permutations := corner_color_combinations * Nat.factorial num_corner_triangles
  let center_color_choices := num_colors - num_corner_triangles
  let total_combinations := corner_color_permutations * center_color_choices
  total_combinations = 1680 :=
by
  sorry

end num_distinguishable_large_triangles_l115_115470


namespace minimum_room_size_for_table_l115_115342

theorem minimum_room_size_for_table (S : ℕ) :
  (∃ S, S ≥ 13) := sorry

end minimum_room_size_for_table_l115_115342


namespace expand_expression_l115_115152

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l115_115152


namespace problem_statement_l115_115807

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def geom_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

theorem problem_statement (a₁ q : ℝ) (h : geom_seq a₁ q 6 = 8 * geom_seq a₁ q 3) :
  geom_sum a₁ q 6 / geom_sum a₁ q 3 = 9 :=
by
  -- proof goes here
  sorry

end problem_statement_l115_115807


namespace pencils_per_row_l115_115524

def total_pencils : ℕ := 32
def rows : ℕ := 4

theorem pencils_per_row : total_pencils / rows = 8 := by
  sorry

end pencils_per_row_l115_115524


namespace multiple_optimal_solutions_for_z_l115_115941

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 0 2
def B := Point.mk (-2) (-2)
def C := Point.mk 2 0

def z (a : ℝ) (P : Point) : ℝ := P.y - a * P.x

def maxz_mult_opt_solutions (a : ℝ) : Prop :=
  z a A = z a B ∨ z a A = z a C ∨ z a B = z a C

theorem multiple_optimal_solutions_for_z :
  (maxz_mult_opt_solutions (-1)) ∧ (maxz_mult_opt_solutions 2) :=
by
  sorry

end multiple_optimal_solutions_for_z_l115_115941


namespace ordered_quadruple_solution_exists_l115_115780

theorem ordered_quadruple_solution_exists (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : 
  a^2 * b = c ∧ b * c^2 = a ∧ c * a^2 = b ∧ a + b + c = d → (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3) :=
by
  sorry

end ordered_quadruple_solution_exists_l115_115780


namespace total_tickets_l115_115995

theorem total_tickets (tickets_first_day tickets_second_day tickets_third_day : ℕ) 
  (h1 : tickets_first_day = 5 * 4) 
  (h2 : tickets_second_day = 32)
  (h3 : tickets_third_day = 28) :
  tickets_first_day + tickets_second_day + tickets_third_day = 80 := by
  sorry

end total_tickets_l115_115995


namespace min_value_of_D_l115_115420

noncomputable def D (x a : ℝ) : ℝ :=
  Real.sqrt ((x - a) ^ 2 + (Real.exp x - 2 * Real.sqrt a) ^ 2) + a + 2

theorem min_value_of_D (e : ℝ) (h_e : e = 2.71828) :
  ∀ a : ℝ, ∃ x : ℝ, D x a = Real.sqrt 2 + 1 :=
sorry

end min_value_of_D_l115_115420


namespace range_of_a_l115_115901

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) → (-2 < a ∧ a ≤ 6/5) :=
by
  sorry

end range_of_a_l115_115901


namespace find_n_l115_115599

theorem find_n (e n : ℕ) (h_lcm : Nat.lcm e n = 690) (h_n_not_div_3 : ¬ (3 ∣ n)) (h_e_not_div_2 : ¬ (2 ∣ e)) : n = 230 :=
by
  sorry

end find_n_l115_115599


namespace sin_cos_product_l115_115556

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l115_115556


namespace expand_polynomial_eq_l115_115188

theorem expand_polynomial_eq : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  intro x
  calc
  (x - 3) * (x + 3) * (x^2 + 9) = (x^2 - 9) * (x^2 + 9) : by rw [mul_comm _ _, mul_assoc, mul_self_sub_self]
    ... = x^4 - 81 : by rw [←sub_mul_self]

end expand_polynomial_eq_l115_115188


namespace angle_is_67_l115_115639

theorem angle_is_67.5 (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end angle_is_67_l115_115639


namespace find_x_minus_y_l115_115378

theorem find_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : y^2 = 16) (h3 : x + y > 0) : x - y = 1 ∨ x - y = 9 := 
by sorry

end find_x_minus_y_l115_115378


namespace Betty_will_pay_zero_l115_115611

-- Definitions of the conditions
def Doug_age : ℕ := 40
def Alice_age (D : ℕ) : ℕ := D / 2
def Betty_age (B D A : ℕ) : Prop := B + D + A = 130
def Cost_of_pack_of_nuts (C B : ℕ) : Prop := C = 2 * B
def Decrease_rate : ℕ := 5
def New_cost (C B A : ℕ) : ℕ := max 0 (C - (B - A) * Decrease_rate)
def Total_cost (packs cost_per_pack: ℕ) : ℕ := packs * cost_per_pack

-- The main proposition
theorem Betty_will_pay_zero :
  ∃ B A C, 
    (C = 2 * B) ∧
    (A = Doug_age / 2) ∧
    (B + Doug_age + A = 130) ∧
    (Total_cost 20 (max 0 (C - (B - A) * Decrease_rate)) = 0) :=
by sorry

end Betty_will_pay_zero_l115_115611


namespace greatest_possible_sum_example_sum_case_l115_115959

/-- For integers x and y such that x^2 + y^2 = 50, the greatest possible value of x + y is 10. -/
theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 10 :=
sorry

-- Auxiliary theorem to state that 10 can be achieved
theorem example_sum_case : ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ x + y = 10 :=
sorry

end greatest_possible_sum_example_sum_case_l115_115959
