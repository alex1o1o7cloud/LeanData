import Mathlib

namespace y1_gt_y2_iff_x_gt_neg_one_fifth_l1153_115395

/-- Given y₁ = a^(2x+1), y₂ = a^(-3x), a > 0, and a > 1, y₁ > y₂ if and only if x > -1/5 -/
theorem y1_gt_y2_iff_x_gt_neg_one_fifth (a x : ℝ) (h1 : a > 0) (h2 : a > 1) :
  a^(2*x + 1) > a^(-3*x) ↔ x > -1/5 := by
  sorry

end y1_gt_y2_iff_x_gt_neg_one_fifth_l1153_115395


namespace rectangle_x_value_l1153_115370

/-- Given a rectangle with vertices (x, 1), (1, 1), (1, -4), and (x, -4) and area 30, prove that x = -5 -/
theorem rectangle_x_value (x : ℝ) : 
  let vertices := [(x, 1), (1, 1), (1, -4), (x, -4)]
  let width := 1 - (-4)
  let area := 30
  let length := area / width
  x = 1 - length → x = -5 := by sorry

end rectangle_x_value_l1153_115370


namespace quadratic_function_theorem_l1153_115378

/-- A quadratic function is a polynomial function of degree 2. -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem: if f is quadratic and satisfies the given condition,
    then it has the specific form x^2 - 4x + 4. -/
theorem quadratic_function_theorem (f : ℝ → ℝ) 
    (h1 : IsQuadratic f)
    (h2 : ∀ x, f x + f (x + 1) = 2 * x^2 - 6 * x + 5) :
    ∀ x, f x = x^2 - 4 * x + 4 := by
  sorry

end quadratic_function_theorem_l1153_115378


namespace factor_difference_of_squares_l1153_115306

theorem factor_difference_of_squares (y : ℝ) : 100 - 25 * y^2 = 25 * (2 - y) * (2 + y) := by
  sorry

end factor_difference_of_squares_l1153_115306


namespace A_share_is_175_l1153_115360

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_share_A (initial_A initial_B change_A change_B total_profit : ℚ) : ℚ :=
  let investment_months_A := initial_A * 8 + (initial_A + change_A) * 4
  let investment_months_B := initial_B * 8 + (initial_B + change_B) * 4
  let total_investment_months := investment_months_A + investment_months_B
  (investment_months_A / total_investment_months) * total_profit

/-- Theorem stating that A's share of the profit is 175 given the specified conditions --/
theorem A_share_is_175 :
  calculate_share_A 2000 4000 (-1000) 1000 630 = 175 := by
  sorry

end A_share_is_175_l1153_115360


namespace negation_equivalence_l1153_115331

theorem negation_equivalence (a : ℝ) : 
  (¬∃ x ∈ Set.Icc 1 2, x^2 - a < 0) ↔ (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) := by
  sorry

end negation_equivalence_l1153_115331


namespace Q_no_real_roots_l1153_115373

def Q (x : ℝ) : ℝ := x^6 - 3*x^5 + 6*x^4 - 6*x^3 - x + 8

theorem Q_no_real_roots : ∀ x : ℝ, Q x ≠ 0 := by
  sorry

end Q_no_real_roots_l1153_115373


namespace sin_50_plus_sqrt3_tan_10_equals_1_l1153_115340

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end sin_50_plus_sqrt3_tan_10_equals_1_l1153_115340


namespace minimum_value_implies_b_equals_one_l1153_115307

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + x + b

-- State the theorem
theorem minimum_value_implies_b_equals_one (a : ℝ) :
  (∃ b : ℝ, (f a b 1 = 1) ∧ 
    (∀ x : ℝ, f a b x ≥ 1) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a b x < f a b 1 + ε)) →
  (∃ b : ℝ, b = 1 ∧ (f a b 1 = 1) ∧ 
    (∀ x : ℝ, f a b x ≥ 1) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a b x < f a b 1 + ε)) :=
by sorry

end minimum_value_implies_b_equals_one_l1153_115307


namespace sequence_inequality_l1153_115384

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn1 : a (n + 1) = 0) 
  (h_ineq : ∀ k : ℕ, k ≥ 1 → k ≤ n → a (k - 1) - 2 * a k + a (k + 1) ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → a k ≤ k * (n + 1 - k) / 2 :=
sorry

end sequence_inequality_l1153_115384


namespace product_divisible_by_sum_implies_inequality_l1153_115344

theorem product_divisible_by_sum_implies_inequality (m n : ℕ+) 
  (h : (m + n : ℕ) ∣ (m * n : ℕ)) : 
  (m : ℕ) + n ≤ n^2 := by
sorry

end product_divisible_by_sum_implies_inequality_l1153_115344


namespace function_zeros_and_monotonicity_l1153_115321

theorem function_zeros_and_monotonicity (a : ℝ) : 
  a ≠ 0 →
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2 * a * x^2 - x - 1 = 0) →
  ¬(∀ x y : ℝ, x > 0 → y > 0 → x < y → x^(2-a) > y^(2-a)) →
  1 < a ∧ a ≤ 2 := by
sorry

end function_zeros_and_monotonicity_l1153_115321


namespace yoongi_has_smallest_number_l1153_115325

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 * 3
def yuna_number : ℕ := 5

theorem yoongi_has_smallest_number : 
  yoongi_number < jungkook_number ∧ yoongi_number < yuna_number :=
sorry

end yoongi_has_smallest_number_l1153_115325


namespace inequality_proof_l1153_115303

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_condition : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
sorry

end inequality_proof_l1153_115303


namespace polynomial_identity_l1153_115337

theorem polynomial_identity 
  (a b c x : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a^2 * ((x-b)*(x-c)) / ((a-b)*(a-c)) + 
  b^2 * ((x-c)*(x-a)) / ((b-c)*(b-a)) + 
  c^2 * ((x-a)*(x-b)) / ((c-a)*(c-b)) = x^2 := by
  sorry

end polynomial_identity_l1153_115337


namespace twenty_bananas_equal_twelve_pears_l1153_115346

/-- The cost relationship between bananas, apples, and pears at Hector's Healthy Habits -/
structure FruitCosts where
  banana : ℚ
  apple : ℚ
  pear : ℚ
  banana_apple_ratio : 4 * banana = 3 * apple
  apple_pear_ratio : 5 * apple = 4 * pear

/-- Theorem stating that 20 bananas cost the same as 12 pears -/
theorem twenty_bananas_equal_twelve_pears (c : FruitCosts) : 20 * c.banana = 12 * c.pear := by
  sorry

end twenty_bananas_equal_twelve_pears_l1153_115346


namespace clown_balloons_l1153_115330

/-- The number of balloons the clown blew up initially -/
def initial_balloons : ℕ := sorry

/-- The number of additional balloons the clown blew up -/
def additional_balloons : ℕ := 13

/-- The total number of balloons the clown has now -/
def total_balloons : ℕ := 60

theorem clown_balloons : initial_balloons = 47 := by
  sorry

end clown_balloons_l1153_115330


namespace sum_of_integers_l1153_115389

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
  sorry

end sum_of_integers_l1153_115389


namespace coffee_cost_calculation_l1153_115399

/-- Calculates the weekly coffee cost for a household -/
def weekly_coffee_cost (people : ℕ) (cups_per_day : ℕ) (oz_per_cup : ℚ) (cost_per_oz : ℚ) : ℚ :=
  people * cups_per_day * oz_per_cup * cost_per_oz * 7

theorem coffee_cost_calculation :
  let people : ℕ := 4
  let cups_per_day : ℕ := 2
  let oz_per_cup : ℚ := 1/2
  let cost_per_oz : ℚ := 5/4
  weekly_coffee_cost people cups_per_day oz_per_cup cost_per_oz = 35 := by
  sorry

#eval weekly_coffee_cost 4 2 (1/2) (5/4)

end coffee_cost_calculation_l1153_115399


namespace language_class_selection_probability_l1153_115387

/-- The probability of selecting two students from different language classes -/
theorem language_class_selection_probability
  (total_students : ℕ)
  (french_students : ℕ)
  (spanish_students : ℕ)
  (no_language_students : ℕ)
  (h1 : total_students = 30)
  (h2 : french_students = 22)
  (h3 : spanish_students = 24)
  (h4 : no_language_students = 2)
  (h5 : french_students + spanish_students - (total_students - no_language_students) + no_language_students = total_students) :
  let both_classes := french_students + spanish_students - (total_students - no_language_students)
  let only_french := french_students - both_classes
  let only_spanish := spanish_students - both_classes
  let total_combinations := total_students.choose 2
  let undesirable_outcomes := (only_french.choose 2) + (only_spanish.choose 2)
  (1 : ℚ) - (undesirable_outcomes : ℚ) / (total_combinations : ℚ) = 14 / 15 :=
by sorry

end language_class_selection_probability_l1153_115387


namespace debate_team_combinations_l1153_115386

theorem debate_team_combinations (n : ℕ) (k : ℕ) : n = 7 → k = 4 → Nat.choose n k = 35 := by
  sorry

end debate_team_combinations_l1153_115386


namespace finite_minimal_elements_l1153_115365

def is_minimal {n : ℕ} (A : Set (Fin n → ℕ+)) (a : Fin n → ℕ+) : Prop :=
  a ∈ A ∧ ∀ b ∈ A, (∀ i, b i ≤ a i) → b = a

theorem finite_minimal_elements {n : ℕ} (A : Set (Fin n → ℕ+)) :
  Set.Finite {a ∈ A | is_minimal A a} := by
  sorry

end finite_minimal_elements_l1153_115365


namespace modular_inverse_45_mod_47_l1153_115304

theorem modular_inverse_45_mod_47 :
  ∃ x : ℕ, x ≤ 46 ∧ (45 * x) % 47 = 1 ∧ x = 23 := by
  sorry

end modular_inverse_45_mod_47_l1153_115304


namespace longest_segment_in_cylinder_l1153_115316

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 4) (hh : h = 10) :
  Real.sqrt ((2 * r)^2 + h^2) = Real.sqrt 164 := by
  sorry

end longest_segment_in_cylinder_l1153_115316


namespace parallel_lines_from_perpendicular_to_parallel_planes_l1153_115359

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the parallel relation for lines
variable (parallelLines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicularLinePlane : Line → Plane → Prop)

-- Define the property of two lines being non-coincident
variable (nonCoincident : Line → Line → Prop)

-- Theorem statement
theorem parallel_lines_from_perpendicular_to_parallel_planes 
  (α β : Plane) (a b : Line) :
  parallelPlanes α β →
  nonCoincident a b →
  perpendicularLinePlane a α →
  perpendicularLinePlane b β →
  parallelLines a b :=
by
  sorry

end parallel_lines_from_perpendicular_to_parallel_planes_l1153_115359


namespace intersection_of_A_and_B_l1153_115352

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | Real.log x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l1153_115352


namespace solution_set_theorem_range_of_b_theorem_l1153_115388

-- Define the real number a
variable (a : ℝ)

-- Define the condition that the solution set of (1-a)x^2 - 4x + 6 > 0 is (-3, 1)
def solution_set_condition : Prop :=
  ∀ x : ℝ, ((1 - a) * x^2 - 4 * x + 6 > 0) ↔ (-3 < x ∧ x < 1)

-- Theorem for the first question
theorem solution_set_theorem (h : solution_set_condition a) :
  ∀ x : ℝ, (2 * x^2 + (2 - a) * x - a > 0) ↔ (x < -1 ∨ x > 3/2) :=
sorry

-- Theorem for the second question
theorem range_of_b_theorem (h : solution_set_condition a) :
  ∀ b : ℝ, (∀ x : ℝ, a * x^2 + b * x + 3 ≥ 0) ↔ (-6 ≤ b ∧ b ≤ 6) :=
sorry

end solution_set_theorem_range_of_b_theorem_l1153_115388


namespace acid_concentration_solution_l1153_115398

/-- Represents the acid concentration problem with three flasks of acid and one of water -/
def AcidConcentrationProblem (acid1 acid2 acid3 : ℝ) (concentration1 concentration2 : ℝ) : Prop :=
  let water1 := acid1 / concentration1 - acid1
  let water2 := acid2 / concentration2 - acid2
  let total_water := water1 + water2
  let concentration3 := acid3 / (acid3 + total_water)
  (acid1 = 10) ∧ 
  (acid2 = 20) ∧ 
  (acid3 = 30) ∧ 
  (concentration1 = 0.05) ∧ 
  (concentration2 = 70/300) ∧ 
  (concentration3 = 0.105)

/-- Theorem stating the solution to the acid concentration problem -/
theorem acid_concentration_solution : 
  ∃ (acid1 acid2 acid3 concentration1 concentration2 : ℝ),
  AcidConcentrationProblem acid1 acid2 acid3 concentration1 concentration2 :=
by
  sorry

end acid_concentration_solution_l1153_115398


namespace total_apples_is_eleven_l1153_115385

/-- The number of apples Marin has -/
def marin_apples : ℕ := 9

/-- The number of apples Donald has -/
def donald_apples : ℕ := 2

/-- The total number of apples Marin and Donald have together -/
def total_apples : ℕ := marin_apples + donald_apples

/-- Proof that the total number of apples is 11 -/
theorem total_apples_is_eleven : total_apples = 11 := by
  sorry

end total_apples_is_eleven_l1153_115385


namespace binomial_coefficient_20_19_l1153_115335

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_coefficient_20_19_l1153_115335


namespace love_logic_l1153_115348

-- Define the propositions
variable (B : Prop) -- "I love Betty"
variable (J : Prop) -- "I love Jane"

-- State the theorem
theorem love_logic (h1 : B ∨ J) (h2 : B → J) : J ∧ ¬(B ↔ True) :=
  sorry


end love_logic_l1153_115348


namespace tray_height_l1153_115351

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : 
  side_length = 120 →
  cut_distance = 5 →
  cut_angle = 45 →
  ∃ (height : ℝ), height = 5 * Real.sqrt 3 :=
by sorry

end tray_height_l1153_115351


namespace stating_four_of_a_kind_hands_l1153_115396

/-- Represents the number of distinct values in a standard deck of cards. -/
def num_values : ℕ := 13

/-- Represents the number of distinct suits in a standard deck of cards. -/
def num_suits : ℕ := 4

/-- Represents the total number of cards in a standard deck. -/
def total_cards : ℕ := num_values * num_suits

/-- Represents the number of cards in a hand. -/
def hand_size : ℕ := 5

/-- 
Theorem stating that the number of 5-card hands containing four cards of the same value 
in a standard 52-card deck is equal to 624.
-/
theorem four_of_a_kind_hands : 
  (num_values : ℕ) * (total_cards - num_suits : ℕ) = 624 := by
  sorry


end stating_four_of_a_kind_hands_l1153_115396


namespace line_slope_l1153_115363

/-- The slope of a line given by the equation 3y - (1/2)x = 9 is 1/6 -/
theorem line_slope (x y : ℝ) : 3 * y - (1/2) * x = 9 → (y - 3) / x = 1/6 := by
  sorry

end line_slope_l1153_115363


namespace is_quadratic_equation_l1153_115397

theorem is_quadratic_equation (x : ℝ) : ∃ (a b c : ℝ), a ≠ 0 ∧ 3*(x-1)^2 = 2*(x-1) ↔ a*x^2 + b*x + c = 0 := by
  sorry

end is_quadratic_equation_l1153_115397


namespace circle_power_theorem_l1153_115367

/-- The power of a point with respect to a circle -/
def power (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : ℝ :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 - radius^2

theorem circle_power_theorem (k : ℝ) (hk : k < 0) :
  (∃ p : ℝ × ℝ, power (0, 0) 1 p = k) ∧
  ¬(∀ k : ℝ, k < 0 → ∃ q : ℝ × ℝ, power (0, 0) 1 q = -k) :=
by sorry

end circle_power_theorem_l1153_115367


namespace f_monotone_decreasing_and_even_l1153_115327

def f (x : ℝ) : ℝ := -2 * x^2

theorem f_monotone_decreasing_and_even :
  (∀ x y, x > 0 → y > 0 → x < y → f x > f y) ∧
  (∀ x, x > 0 → f x = f (-x)) :=
by sorry

end f_monotone_decreasing_and_even_l1153_115327


namespace eggs_per_person_l1153_115377

theorem eggs_per_person (mark_eggs : ℕ) (siblings : ℕ) : 
  mark_eggs = 2 * 12 → siblings = 3 → (mark_eggs / (siblings + 1) : ℚ) = 6 := by
  sorry

end eggs_per_person_l1153_115377


namespace pipe_filling_time_l1153_115324

/-- Given two pipes A and B that can fill a tank, this theorem proves the time
    it takes for pipe B to fill the tank alone, given the filling times for
    pipe A alone and both pipes together. -/
theorem pipe_filling_time (fill_time_A fill_time_both : ℝ) 
  (h1 : fill_time_A = 30) 
  (h2 : fill_time_both = 18) : 
  (1 / fill_time_A + 1 / (1 / (1 / fill_time_both - 1 / fill_time_A)))⁻¹ = 45 := by
  sorry

end pipe_filling_time_l1153_115324


namespace number_of_positive_divisors_of_M_l1153_115381

def M : ℕ := 49^6 + 6*49^5 + 15*49^4 + 20*49^3 + 15*49^2 + 6*49 + 1

theorem number_of_positive_divisors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 91 := by
  sorry

end number_of_positive_divisors_of_M_l1153_115381


namespace reciprocal_of_opposite_l1153_115376

theorem reciprocal_of_opposite (x : ℝ) (h : x ≠ 0) : 
  (-(1 / x)) = 1 / (-x) :=
sorry

end reciprocal_of_opposite_l1153_115376


namespace no_geometric_progression_of_2n_plus_1_l1153_115375

theorem no_geometric_progression_of_2n_plus_1 :
  ¬ ∃ (k m n : ℕ), k ≠ m ∧ m ≠ n ∧ k ≠ n ∧
    (2^m + 1)^2 = (2^k + 1) * (2^n + 1) :=
sorry

end no_geometric_progression_of_2n_plus_1_l1153_115375


namespace locus_and_tangent_lines_l1153_115301

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point M on the ellipse
def M : ℝ × ℝ := sorry

-- Define point N as the projection of M on x = 3
def N : ℝ × ℝ := (3, M.2)

-- Define point P
def P : ℝ × ℝ := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector from O to a point
def vector_to (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - O.1, p.2 - O.2)

-- Define the locus E
def E (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point A
def A : ℝ × ℝ := (1, 4)

-- Define the tangent line equations
def tangent_line_1 (x y : ℝ) : Prop := x = 1
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y - 19 = 0

theorem locus_and_tangent_lines :
  ellipse M.1 M.2 ∧
  N = (3, M.2) ∧
  vector_to P = vector_add (vector_to M) (vector_to N) →
  (∀ x y, E x y ↔ (∃ m n, ellipse m n ∧ x = m + 3 ∧ y = 2*n)) ∧
  (∀ x y, (tangent_line_1 x y ∨ tangent_line_2 x y) ↔
    (E x y ∧ (x - A.1)^2 + (y - A.2)^2 = ((x - 3)^2 + y^2))) :=
sorry

end locus_and_tangent_lines_l1153_115301


namespace no_real_solution_for_log_equation_l1153_115366

theorem no_real_solution_for_log_equation :
  ¬ ∃ x : ℝ, (Real.log (x + 5) + Real.log (2 * x - 2) = Real.log (2 * x^2 + x - 10)) ∧ 
  (x + 5 > 0) ∧ (2 * x - 2 > 0) ∧ (2 * x^2 + x - 10 > 0) :=
by sorry

end no_real_solution_for_log_equation_l1153_115366


namespace special_function_at_zero_l1153_115391

/-- A function satisfying f(x + y) = f(x) + f(y) - xy for all real x and y, with f(1) = 1 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - x * y) ∧ (f 1 = 1)

/-- Theorem: For a special function f, f(0) = 0 -/
theorem special_function_at_zero {f : ℝ → ℝ} (hf : special_function f) : f 0 = 0 := by
  sorry

end special_function_at_zero_l1153_115391


namespace circle_radius_three_inches_l1153_115392

theorem circle_radius_three_inches 
  (r : ℝ) 
  (h : r > 0) 
  (h_eq : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2)) : 
  r = 3 := by
sorry

end circle_radius_three_inches_l1153_115392


namespace complement_of_union_M_P_l1153_115369

open Set

-- Define the universal set U as the real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x ≤ 1}

-- Define set P
def P : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_M_P : 
  (M ∪ P)ᶜ = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end complement_of_union_M_P_l1153_115369


namespace system_equation_solution_l1153_115341

theorem system_equation_solution :
  ∃ (x y : ℝ),
    (4 * x + y = 15) ∧
    (x + 4 * y = 18) ∧
    (13 * x^2 + 14 * x * y + 13 * y^2 = 438.6) := by
  sorry

end system_equation_solution_l1153_115341


namespace sum_of_squares_l1153_115393

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end sum_of_squares_l1153_115393


namespace distance_after_one_hour_l1153_115322

/-- The distance between two people moving in opposite directions for 1 hour -/
def distance_between (speed1 speed2 : ℝ) : ℝ :=
  speed1 + speed2

theorem distance_after_one_hour :
  let riya_speed : ℝ := 21
  let priya_speed : ℝ := 22
  distance_between riya_speed priya_speed = 43 := by
  sorry

end distance_after_one_hour_l1153_115322


namespace range_of_3a_minus_b_l1153_115368

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3) 
  (h2 : 2 < a - b ∧ a - b < 4) : 
  (∃ (x y : ℝ), (x = a ∧ y = b) ∧ 3*x - y = 3) ∧ 
  (∃ (x y : ℝ), (x = a ∧ y = b) ∧ 3*x - y = 11) ∧
  (∀ (x y : ℝ), (x = a ∧ y = b) → 3 ≤ 3*x - y ∧ 3*x - y ≤ 11) :=
sorry

end range_of_3a_minus_b_l1153_115368


namespace area_of_specific_quadrilateral_l1153_115356

/-- The area of a quadrilateral can be calculated using the Shoelace formula -/
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let x1 := v1.1
  let y1 := v1.2
  let x2 := v2.1
  let y2 := v2.2
  let x3 := v3.1
  let y3 := v3.2
  let x4 := v4.1
  let y4 := v4.2
  0.5 * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem area_of_specific_quadrilateral :
  quadrilateralArea (2, 1) (4, 3) (7, 1) (4, 6) = 7.5 := by
  sorry

end area_of_specific_quadrilateral_l1153_115356


namespace complement_union_equals_set_l1153_115364

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_equals_set : 
  (U \ (M ∪ N)) = {1, 6} := by sorry

end complement_union_equals_set_l1153_115364


namespace simplify_and_evaluate_l1153_115319

theorem simplify_and_evaluate (m n : ℤ) (h1 : m = 2) (h2 : n = -1^2023) :
  (2*m + n) * (2*m - n) - (2*m - n)^2 + 2*n*(m + n) = -12 := by
  sorry

end simplify_and_evaluate_l1153_115319


namespace train_speed_l1153_115311

-- Define the train length in meters
def train_length : ℝ := 180

-- Define the time to cross in seconds
def crossing_time : ℝ := 12

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem to prove
theorem train_speed :
  (train_length / crossing_time) * ms_to_kmh = 54 := by
  sorry


end train_speed_l1153_115311


namespace winter_solstice_shadow_length_l1153_115312

/-- Given an arithmetic sequence of 12 terms, if the sum of the 1st, 4th, and 7th terms is 37.5
    and the 12th term is 4.5, then the 1st term is 15.5. -/
theorem winter_solstice_shadow_length 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 0 + a 3 + a 6 = 37.5) 
  (h_last : a 11 = 4.5) : 
  a 0 = 15.5 := by
sorry

end winter_solstice_shadow_length_l1153_115312


namespace polynomial_value_theorem_l1153_115333

-- Define the function f
def f (a b c d e : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

-- State the theorem
theorem polynomial_value_theorem (a b c d e : ℝ) :
  f a b c d e (-1) = 2 → 16 * a - 8 * b + 4 * c - 2 * d + e = 2 := by
  sorry

end polynomial_value_theorem_l1153_115333


namespace intersection_of_A_and_B_l1153_115382

def A : Set ℝ := {-1, 0, 3, 5}
def B : Set ℝ := {x : ℝ | x - 2 > 0}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by
  sorry

end intersection_of_A_and_B_l1153_115382


namespace geometric_sequence_product_l1153_115328

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) :
  a 3 * a 7 = 6 → a 2 * a 4 * a 6 * a 8 = 36 := by
  sorry

end geometric_sequence_product_l1153_115328


namespace arithmetic_sequence_eighth_term_l1153_115353

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_sum : a 3 + a 4 + a 5 + a 6 = 20) :
  a 8 = 9 := by
sorry

end arithmetic_sequence_eighth_term_l1153_115353


namespace parallel_line_slope_parallel_line_y_intercept_exists_l1153_115361

/-- Given a line parallel to 3x - 6y = 12, prove its slope is 1/2 --/
theorem parallel_line_slope (a b c : ℝ) (h : ∃ k : ℝ, a * x + b * y = c ∧ k ≠ 0 ∧ 3 * (a / b) = -1 / 2) :
  a / b = 1 / 2 := by sorry

/-- The y-intercept of a line parallel to 3x - 6y = 12 can be any real number --/
theorem parallel_line_y_intercept_exists : ∀ k : ℝ, ∃ (a b c : ℝ), a * x + b * y = c ∧ a / b = 1 / 2 ∧ c / b = k := by sorry

end parallel_line_slope_parallel_line_y_intercept_exists_l1153_115361


namespace system_solution_ratio_l1153_115342

theorem system_solution_ratio :
  ∃ (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0),
    x + 10*y + 5*z = 0 ∧
    2*x + 5*y + 4*z = 0 ∧
    3*x + 6*y + 5*z = 0 ∧
    y*z / (x^2) = -3/49 := by
  sorry

end system_solution_ratio_l1153_115342


namespace min_value_re_z4_over_re_z4_l1153_115310

theorem min_value_re_z4_over_re_z4 (z : ℂ) (h : (z.re : ℝ) ≠ 0) :
  (z^4).re / (z.re^4 : ℝ) ≥ -8 := by sorry

end min_value_re_z4_over_re_z4_l1153_115310


namespace total_paintable_area_is_1200_l1153_115357

/-- Represents the dimensions of a bedroom --/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area of a bedroom --/
def totalWallArea (dim : BedroomDimensions) : ℝ :=
  2 * (dim.length * dim.height + dim.width * dim.height)

/-- Calculates the paintable wall area of a bedroom --/
def paintableWallArea (dim : BedroomDimensions) (nonPaintableArea : ℝ) : ℝ :=
  totalWallArea dim - nonPaintableArea

/-- The main theorem stating the total paintable area of all bedrooms --/
theorem total_paintable_area_is_1200 
  (bedroom1 : BedroomDimensions)
  (bedroom2 : BedroomDimensions)
  (bedroom3 : BedroomDimensions)
  (nonPaintable1 nonPaintable2 nonPaintable3 : ℝ) :
  bedroom1.length = 14 ∧ bedroom1.width = 11 ∧ bedroom1.height = 9 ∧
  bedroom2.length = 13 ∧ bedroom2.width = 12 ∧ bedroom2.height = 9 ∧
  bedroom3.length = 15 ∧ bedroom3.width = 10 ∧ bedroom3.height = 9 ∧
  nonPaintable1 = 50 ∧ nonPaintable2 = 55 ∧ nonPaintable3 = 45 →
  paintableWallArea bedroom1 nonPaintable1 + 
  paintableWallArea bedroom2 nonPaintable2 + 
  paintableWallArea bedroom3 nonPaintable3 = 1200 := by
  sorry

end total_paintable_area_is_1200_l1153_115357


namespace negative_64_to_four_thirds_l1153_115309

theorem negative_64_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by sorry

end negative_64_to_four_thirds_l1153_115309


namespace divisibility_implies_equality_l1153_115317

theorem divisibility_implies_equality (a b : ℕ) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end divisibility_implies_equality_l1153_115317


namespace banana_count_l1153_115347

theorem banana_count (total : ℕ) (apple_multiplier persimmon_multiplier : ℕ) 
  (h1 : total = 210)
  (h2 : apple_multiplier = 4)
  (h3 : persimmon_multiplier = 3) :
  ∃ (banana_count : ℕ), 
    banana_count * (apple_multiplier + persimmon_multiplier) = total ∧ 
    banana_count = 30 := by
  sorry

end banana_count_l1153_115347


namespace fraction_inequality_counterexample_l1153_115350

theorem fraction_inequality_counterexample : 
  ∃ (a b c d : ℝ), (a / b > c / d) ∧ (b / a ≥ d / c) := by
  sorry

end fraction_inequality_counterexample_l1153_115350


namespace min_value_quadratic_l1153_115379

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 1711 ∧ ∀ x, 8 * x^2 - 24 * x + 1729 ≥ m := by
  sorry

end min_value_quadratic_l1153_115379


namespace gcd_459_357_l1153_115383

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l1153_115383


namespace indefinite_integral_proof_l1153_115329

theorem indefinite_integral_proof (x : ℝ) :
  deriv (fun x => (2 - 3*x) * Real.exp (2*x)) = fun x => (1 - 6*x) * Real.exp (2*x) := by
  sorry

end indefinite_integral_proof_l1153_115329


namespace probability_two_red_crayons_l1153_115380

/-- The probability of selecting 2 red crayons from a jar containing 3 red, 2 blue, and 1 green crayon -/
theorem probability_two_red_crayons (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) :
  total = red + blue + green →
  total = 6 →
  red = 3 →
  blue = 2 →
  green = 1 →
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 1 / 5 := by
  sorry

end probability_two_red_crayons_l1153_115380


namespace raisin_distribution_l1153_115345

/-- The number of raisins Bryce received -/
def bryce_raisins : ℕ := 16

/-- The number of raisins Carter received -/
def carter_raisins : ℕ := bryce_raisins - 8

theorem raisin_distribution :
  (bryce_raisins = carter_raisins + 8) ∧
  (carter_raisins = bryce_raisins / 2) :=
by sorry

#check raisin_distribution

end raisin_distribution_l1153_115345


namespace amy_game_score_l1153_115372

theorem amy_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : 
  points_per_treasure = 4 →
  treasures_level1 = 6 →
  treasures_level2 = 2 →
  points_per_treasure * treasures_level1 + points_per_treasure * treasures_level2 = 32 := by
sorry

end amy_game_score_l1153_115372


namespace min_magnitude_in_A_l1153_115339

def a : Fin 3 → ℝ := ![1, 2, 3]
def b : Fin 3 → ℝ := ![1, -1, 1]

def A : Set (Fin 3 → ℝ) :=
  {x | ∃ k : ℤ, x = fun i => a i + k * b i}

theorem min_magnitude_in_A :
  ∃ x ∈ A, ∀ y ∈ A, ‖x‖ ≤ ‖y‖ ∧ ‖x‖ = Real.sqrt 13 :=
sorry

end min_magnitude_in_A_l1153_115339


namespace f_max_value_l1153_115394

/-- The quadratic function f(x) = -2x^2 - 8x + 16 -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 16

/-- The maximum value of f(x) -/
def max_value : ℝ := 24

/-- The x-coordinate where f(x) achieves its maximum value -/
def max_point : ℝ := -2

theorem f_max_value :
  (∀ x : ℝ, f x ≤ max_value) ∧ f max_point = max_value := by sorry

end f_max_value_l1153_115394


namespace least_distance_is_one_thirtyfifth_l1153_115318

-- Define the unit segment [0, 1]
def unit_segment : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define the division points for fifths
def fifth_points : Set ℝ := {x : ℝ | ∃ n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ x = n / 5}

-- Define the division points for sevenths
def seventh_points : Set ℝ := {x : ℝ | ∃ n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ x = n / 7}

-- Define all division points
def all_points : Set ℝ := fifth_points ∪ seventh_points

-- Define the distance between two points
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem least_distance_is_one_thirtyfifth :
  ∃ x y : ℝ, x ∈ all_points ∧ y ∈ all_points ∧ x ≠ y ∧
  distance x y = 1 / 35 ∧
  ∀ a b : ℝ, a ∈ all_points → b ∈ all_points → a ≠ b →
  distance a b ≥ 1 / 35 :=
sorry

end least_distance_is_one_thirtyfifth_l1153_115318


namespace smallest_b_quadratic_inequality_l1153_115338

theorem smallest_b_quadratic_inequality :
  let f : ℝ → ℝ := fun b => -3 * b^2 + 13 * b - 10
  ∃ b_min : ℝ, b_min = -2/3 ∧
    (∀ b : ℝ, f b ≥ 0 → b ≥ b_min) ∧
    f b_min ≥ 0 :=
by sorry

end smallest_b_quadratic_inequality_l1153_115338


namespace john_playing_time_l1153_115305

theorem john_playing_time (beats_per_minute : ℕ) (total_days : ℕ) (total_beats : ℕ) :
  beats_per_minute = 200 →
  total_days = 3 →
  total_beats = 72000 →
  (total_beats / beats_per_minute / 60) / total_days = 2 :=
by
  sorry

end john_playing_time_l1153_115305


namespace cube_volume_from_surface_area_l1153_115308

/-- The volume of a cube with surface area 24 cm² is 8 cm³. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 24 → s^3 = 8 :=
by
  sorry

end cube_volume_from_surface_area_l1153_115308


namespace quadratic_sequence_exists_smallest_n_for_specific_sequence_l1153_115302

/-- A sequence is quadratic if the absolute difference between consecutive terms is the square of their index. -/
def IsQuadraticSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≥ 1 ∧ i ≤ n → |a i - a (i-1)| = i^2

theorem quadratic_sequence_exists (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ IsQuadraticSequence a n :=
sorry

theorem smallest_n_for_specific_sequence :
  (∃ (a : ℕ → ℤ), a 0 = 0 ∧ a 19 = 1996 ∧ IsQuadraticSequence a 19) ∧
  (∀ n : ℕ, n < 19 → ¬∃ (a : ℕ → ℤ), a 0 = 0 ∧ a n = 1996 ∧ IsQuadraticSequence a n) :=
sorry

end quadratic_sequence_exists_smallest_n_for_specific_sequence_l1153_115302


namespace difference_of_cubes_divisible_by_nine_l1153_115349

theorem difference_of_cubes_divisible_by_nine (a b : ℤ) :
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 = 9*k :=
sorry

end difference_of_cubes_divisible_by_nine_l1153_115349


namespace coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l1153_115343

theorem coefficient_x3y5_in_expansion_of_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℕ) * (if k = 3 then 1 else 0)) = 56 := by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l1153_115343


namespace max_value_2sin_l1153_115315

theorem max_value_2sin (f : ℝ → ℝ) (h : f = λ x => 2 * Real.sin x) :
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x ≤ M := by
sorry

end max_value_2sin_l1153_115315


namespace sin_300_cos_0_l1153_115313

theorem sin_300_cos_0 : Real.sin (300 * π / 180) * Real.cos 0 = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_cos_0_l1153_115313


namespace g_4_equals_10_l1153_115323

/-- A function g satisfying xg(y) = yg(x) for all real x and y, and g(12) = 30 -/
def g : ℝ → ℝ :=
  sorry

/-- The property that xg(y) = yg(x) for all real x and y -/
axiom g_property : ∀ x y : ℝ, x * g y = y * g x

/-- The given condition that g(12) = 30 -/
axiom g_12 : g 12 = 30

/-- Theorem stating that g(4) = 10 -/
theorem g_4_equals_10 : g 4 = 10 := by
  sorry

end g_4_equals_10_l1153_115323


namespace sum_of_interior_angles_equal_diagonal_regular_polygon_l1153_115300

/-- A regular polygon with all diagonals equal -/
structure EqualDiagonalRegularPolygon where
  /-- The number of sides of the polygon -/
  sides : ℕ
  /-- The polygon is regular -/
  regular : True
  /-- All diagonals of the polygon are equal -/
  equal_diagonals : True
  /-- The polygon has at least 3 sides -/
  sides_ge_three : sides ≥ 3

/-- The sum of interior angles of a polygon -/
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles of a regular polygon with all diagonals equal
    is either 360° or 540° -/
theorem sum_of_interior_angles_equal_diagonal_regular_polygon
  (p : EqualDiagonalRegularPolygon) :
  sum_of_interior_angles p.sides = 360 ∨ sum_of_interior_angles p.sides = 540 :=
sorry

end sum_of_interior_angles_equal_diagonal_regular_polygon_l1153_115300


namespace sector_area_l1153_115355

/-- Given a circular sector with central angle 1 radian and circumference 6,
    prove that its area is 2. -/
theorem sector_area (θ : Real) (c : Real) (h1 : θ = 1) (h2 : c = 6) :
  let r := c / 3
  (1/2) * r^2 * θ = 2 := by sorry

end sector_area_l1153_115355


namespace volume_ratio_of_cubes_l1153_115334

/-- The ratio of volumes of two cubes -/
theorem volume_ratio_of_cubes (inches_per_foot : ℚ) (edge_length_small : ℚ) (edge_length_large : ℚ) :
  inches_per_foot = 12 →
  edge_length_small = 4 →
  edge_length_large = 2 * inches_per_foot →
  (edge_length_small ^ 3) / (edge_length_large ^ 3) = 1 / 216 := by
  sorry

end volume_ratio_of_cubes_l1153_115334


namespace percentage_increase_proof_l1153_115390

def original_earnings : ℚ := 60
def new_earnings : ℚ := 80

theorem percentage_increase_proof :
  (new_earnings - original_earnings) / original_earnings = 1/3 := by
  sorry

end percentage_increase_proof_l1153_115390


namespace fresh_grape_weight_l1153_115314

/-- Theorem: Weight of fresh grapes given dried grape weight and water content -/
theorem fresh_grape_weight
  (fresh_water_content : ℝ)
  (dried_water_content : ℝ)
  (dried_grape_weight : ℝ)
  (h1 : fresh_water_content = 0.7)
  (h2 : dried_water_content = 0.1)
  (h3 : dried_grape_weight = 33.33333333333333)
  : ∃ (fresh_grape_weight : ℝ),
    fresh_grape_weight * (1 - fresh_water_content) =
    dried_grape_weight * (1 - dried_water_content) ∧
    fresh_grape_weight = 100 := by
  sorry

end fresh_grape_weight_l1153_115314


namespace machine_work_time_l1153_115332

theorem machine_work_time (time_A time_B time_ABC : ℚ) (time_C : ℚ) : 
  time_A = 4 → time_B = 2 → time_ABC = 12/11 → 
  1/time_A + 1/time_B + 1/time_C = 1/time_ABC → 
  time_C = 6 := by sorry

end machine_work_time_l1153_115332


namespace newer_truck_travels_195_miles_l1153_115320

/-- The distance traveled by the older truck in miles -/
def older_truck_distance : ℝ := 150

/-- The percentage increase in distance for the newer truck -/
def newer_truck_percentage : ℝ := 0.30

/-- The distance traveled by the newer truck in miles -/
def newer_truck_distance : ℝ := older_truck_distance * (1 + newer_truck_percentage)

/-- Theorem stating that the newer truck travels 195 miles -/
theorem newer_truck_travels_195_miles :
  newer_truck_distance = 195 := by sorry

end newer_truck_travels_195_miles_l1153_115320


namespace blue_ball_probability_l1153_115358

noncomputable def bag_probabilities (p_red p_yellow p_blue : ℝ) : Prop :=
  p_red + p_yellow + p_blue = 1 ∧ 0 ≤ p_red ∧ 0 ≤ p_yellow ∧ 0 ≤ p_blue

theorem blue_ball_probability :
  ∀ (p_red p_yellow p_blue : ℝ),
    bag_probabilities p_red p_yellow p_blue →
    p_red = 0.48 →
    p_yellow = 0.35 →
    p_blue = 0.17 :=
by sorry

end blue_ball_probability_l1153_115358


namespace library_interval_proof_l1153_115371

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def next_common_day : ℕ := 36

theorem library_interval_proof (x : ℕ) 
  (h1 : x > 0)
  (h2 : x ∣ next_common_day)
  (h3 : x ≠ dance_interval)
  (h4 : x ≠ karate_interval)
  (h5 : ∀ y : ℕ, y > 0 → y ∣ next_common_day → y ≠ dance_interval → y ≠ karate_interval → y ≤ x) :
  x = 18 := by sorry

end library_interval_proof_l1153_115371


namespace range_of_a_l1153_115362

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + a^2 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x y, x < y → (2*a^2 - a)^x < (2*a^2 - a)^y

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
by sorry

end range_of_a_l1153_115362


namespace cubic_polynomial_unique_l1153_115374

/-- A monic cubic polynomial with real coefficients -/
def cubic_polynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_unique 
  (q : ℝ → ℂ) 
  (h_monic : ∀ x, q x = x^3 + (q 1 - 1) * x^2 + (q 1 - q 0 - 1) * x + q 0)
  (h_root : q (5 - 3*I) = 0)
  (h_const : q 0 = 81) :
  ∀ x, q x = x^3 - (79/16)*x^2 - (17/8)*x + 81 := by
sorry

end cubic_polynomial_unique_l1153_115374


namespace inequality_proof_l1153_115336

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_condition : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ 
  (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l1153_115336


namespace quadratic_equation_unique_solution_positive_n_value_l1153_115354

theorem quadratic_equation_unique_solution (n : ℝ) : 
  (∃! x : ℝ, 5 * x^2 + n * x + 45 = 0) → n = 30 ∨ n = -30 :=
by sorry

theorem positive_n_value (n : ℝ) : 
  (∃! x : ℝ, 5 * x^2 + n * x + 45 = 0) → n > 0 → n = 30 :=
by sorry

end quadratic_equation_unique_solution_positive_n_value_l1153_115354


namespace tom_worked_eight_hours_l1153_115326

/-- Represents the number of hours Tom worked on Monday -/
def hours : ℝ := 8

/-- Represents the number of customers Tom served per hour -/
def customers_per_hour : ℝ := 10

/-- Represents the bonus point percentage (20% = 0.20) -/
def bonus_percentage : ℝ := 0.20

/-- Represents the total bonus points Tom earned on Monday -/
def total_bonus_points : ℝ := 16

/-- Proves that Tom worked 8 hours on Monday given the conditions -/
theorem tom_worked_eight_hours :
  hours * customers_per_hour * bonus_percentage = total_bonus_points :=
sorry

end tom_worked_eight_hours_l1153_115326
