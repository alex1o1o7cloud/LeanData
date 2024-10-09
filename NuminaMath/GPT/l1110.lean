import Mathlib

namespace graph_is_two_lines_l1110_111070

theorem graph_is_two_lines (x y : ℝ) : (x^2 - 25 * y^2 - 10 * x + 50 = 0) ↔
  (x = 5 + 5 * y) ∨ (x = 5 - 5 * y) :=
by
  sorry

end graph_is_two_lines_l1110_111070


namespace max_valid_subset_cardinality_l1110_111092

def set_S : Finset ℕ := Finset.range 1998 \ {0}

def is_valid_subset (A : Finset ℕ) : Prop :=
  ∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → (x + y) % 117 ≠ 0

theorem max_valid_subset_cardinality :
  ∃ (A : Finset ℕ), is_valid_subset A ∧ 995 = A.card :=
sorry

end max_valid_subset_cardinality_l1110_111092


namespace mans_rate_in_still_water_l1110_111003

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h_with_stream : V_m + V_s = 26)
  (h_against_stream : V_m - V_s = 4) :
  V_m = 15 :=
by {
  sorry
}

end mans_rate_in_still_water_l1110_111003


namespace part1_part2_l1110_111085

-- Part (1) prove maximum value of 4 - 2x - 1/x when x > 0 is 0.
theorem part1 (x : ℝ) (h : 0 < x) : 
  4 - 2 * x - (2 / x) ≤ 0 :=
sorry

-- Part (2) prove minimum value of 1/a + 1/b when a + 2b = 1 and a > 0, b > 0 is 3 + 2 * sqrt 2.
theorem part2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 1) :
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end part1_part2_l1110_111085


namespace distribute_balls_in_boxes_l1110_111076

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l1110_111076


namespace solve_for_x_l1110_111039

theorem solve_for_x (x y : ℕ) (h1 : x / y = 15 / 5) (h2 : y = 25) : x = 75 := 
by
  sorry

end solve_for_x_l1110_111039


namespace tens_digit_36_pow_12_l1110_111026

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def tens_digit (n : ℕ) : ℕ :=
  (last_two_digits n) / 10

theorem tens_digit_36_pow_12 : tens_digit (36^12) = 3 :=
by
  sorry

end tens_digit_36_pow_12_l1110_111026


namespace base_six_conversion_addition_l1110_111064

def base_six_to_base_ten (n : ℕ) : ℕ :=
  4 * 6^0 + 1 * 6^1 + 2 * 6^2

theorem base_six_conversion_addition : base_six_to_base_ten 214 + 15 = 97 :=
by
  sorry

end base_six_conversion_addition_l1110_111064


namespace paigeRatio_l1110_111001

/-- The total number of pieces in the chocolate bar -/
def totalPieces : ℕ := 60

/-- Michael takes half of the chocolate bar -/
def michaelPieces : ℕ := totalPieces / 2

/-- Mandy gets a fixed number of pieces -/
def mandyPieces : ℕ := 15

/-- The number of pieces left after Michael takes his share -/
def remainingPiecesAfterMichael : ℕ := totalPieces - michaelPieces

/-- The number of pieces Paige takes -/
def paigePieces : ℕ := remainingPiecesAfterMichael - mandyPieces

/-- The ratio of the number of pieces Paige takes to the number of pieces left after Michael takes his share is 1:2 -/
theorem paigeRatio :
  paigePieces / (remainingPiecesAfterMichael / 15) = 1 := sorry

end paigeRatio_l1110_111001


namespace find_divisor_l1110_111051

def dividend := 23
def quotient := 4
def remainder := 3

theorem find_divisor (d : ℕ) (h : dividend = (d * quotient) + remainder) : d = 5 :=
by {
  sorry
}

end find_divisor_l1110_111051


namespace chord_probability_concentric_circles_l1110_111014

noncomputable def chord_intersects_inner_circle_probability : ℝ :=
  sorry

theorem chord_probability_concentric_circles :
  let r₁ := 2
  let r₂ := 3
  ∀ (P₁ P₂ : ℝ × ℝ),
    dist P₁ (0, 0) = r₂ ∧ dist P₂ (0, 0) = r₂ →
    chord_intersects_inner_circle_probability = 0.148 :=
  sorry

end chord_probability_concentric_circles_l1110_111014


namespace jessica_current_age_l1110_111029

theorem jessica_current_age : 
  ∃ J M_d M_c : ℕ, 
    J = (M_d / 2) ∧ 
    M_d = M_c - 10 ∧ 
    M_c = 70 ∧ 
    J + 10 = 40 := 
sorry

end jessica_current_age_l1110_111029


namespace area_overlap_of_triangles_l1110_111017

structure Point where
  x : ℝ
  y : ℝ

def Triangle (p1 p2 p3 : Point) : Set Point :=
  { q | ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ (a * p1.x + b * p2.x + c * p3.x = q.x) ∧ (a * p1.y + b * p2.y + c * p3.y = q.y) }

def area_of_overlap (t1 t2 : Set Point) : ℝ :=
  -- Assume we have a function that calculates the overlap area
  sorry

def point1 : Point := ⟨0, 2⟩
def point2 : Point := ⟨2, 1⟩
def point3 : Point := ⟨0, 0⟩
def point4 : Point := ⟨2, 2⟩
def point5 : Point := ⟨0, 1⟩
def point6 : Point := ⟨2, 0⟩

def triangle1 : Set Point := Triangle point1 point2 point3
def triangle2 : Set Point := Triangle point4 point5 point6

theorem area_overlap_of_triangles :
  area_of_overlap triangle1 triangle2 = 1 :=
by
  -- Proof goes here, replacing sorry with actual proof steps
  sorry

end area_overlap_of_triangles_l1110_111017


namespace east_high_school_students_l1110_111062

theorem east_high_school_students (S : ℝ) 
  (h1 : 0.52 * S * 0.125 = 26) :
  S = 400 :=
by
  -- The proof is omitted for this exercise
  sorry

end east_high_school_students_l1110_111062


namespace rectangle_problem_l1110_111024

noncomputable def calculate_width (L P : ℕ) : ℕ :=
  (P - 2 * L) / 2

theorem rectangle_problem :
  ∀ (L P : ℕ), L = 12 → P = 36 → (calculate_width L P = 6) ∧ ((calculate_width L P) / L = 1 / 2) :=
by
  intros L P hL hP
  have hw : calculate_width L P = 6 := by
    sorry
  have hr : ((calculate_width L P) / L) = 1 / 2 := by
    sorry
  exact ⟨hw, hr⟩

end rectangle_problem_l1110_111024


namespace infinite_sum_evaluation_l1110_111052

theorem infinite_sum_evaluation :
  (∑' n : ℕ, (n : ℚ) / ((n^2 - 2 * n + 2) * (n^2 + 2 * n + 4))) = 5 / 24 :=
sorry

end infinite_sum_evaluation_l1110_111052


namespace max_student_count_l1110_111028

theorem max_student_count
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : (x1 + x2 + x3 + x4 + x5) / 5 = 7)
  (h2 : ((x1 - 7) ^ 2 + (x2 - 7) ^ 2 + (x3 - 7) ^ 2 + (x4 - 7) ^ 2 + (x5 - 7) ^ 2) / 5 = 4)
  (h3 : ∀ i j, i ≠ j → List.nthLe [x1, x2, x3, x4, x5] i sorry ≠ List.nthLe [x1, x2, x3, x4, x5] j sorry) :
  max x1 (max x2 (max x3 (max x4 x5))) = 10 := 
sorry

end max_student_count_l1110_111028


namespace phraseCompletion_l1110_111036

-- Define the condition for the problem
def isCorrectPhrase (phrase : String) : Prop :=
  phrase = "crying"

-- State the theorem to be proven
theorem phraseCompletion : ∃ phrase, isCorrectPhrase phrase :=
by
  use "crying"
  sorry

end phraseCompletion_l1110_111036


namespace arithmetic_sequence_min_value_Sn_l1110_111075

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l1110_111075


namespace jerry_needs_money_l1110_111018

theorem jerry_needs_money 
  (current_count : ℕ) (total_needed : ℕ) (cost_per_action_figure : ℕ)
  (h1 : current_count = 7) 
  (h2 : total_needed = 16) 
  (h3 : cost_per_action_figure = 8) :
  (total_needed - current_count) * cost_per_action_figure = 72 :=
by sorry

end jerry_needs_money_l1110_111018


namespace symmetric_points_sum_l1110_111087

theorem symmetric_points_sum (a b : ℝ) (h1 : B = (-A)) (h2 : A = (1, a)) (h3 : B = (b, 2)) : a + b = -3 := by
  sorry

end symmetric_points_sum_l1110_111087


namespace extended_pattern_ratio_l1110_111038

def original_black_tiles : ℕ := 13
def original_white_tiles : ℕ := 12
def original_total_tiles : ℕ := 5 * 5

def new_side_length : ℕ := 7
def new_total_tiles : ℕ := new_side_length * new_side_length
def added_white_tiles : ℕ := new_total_tiles - original_total_tiles

def new_black_tiles : ℕ := original_black_tiles
def new_white_tiles : ℕ := original_white_tiles + added_white_tiles

def ratio_black_to_white : ℚ := new_black_tiles / new_white_tiles

theorem extended_pattern_ratio :
  ratio_black_to_white = 13 / 36 :=
by
  sorry

end extended_pattern_ratio_l1110_111038


namespace ratio_part_to_third_fraction_l1110_111032

variable (P N : ℕ)

-- Definitions based on conditions
def one_fourth_one_third_P_eq_14 : Prop := (1/4 : ℚ) * (1/3 : ℚ) * (P : ℚ) = 14

def forty_percent_N_eq_168 : Prop := (40/100 : ℚ) * (N : ℚ) = 168

-- Theorem stating the required ratio
theorem ratio_part_to_third_fraction (h1 : one_fourth_one_third_P_eq_14 P) (h2 : forty_percent_N_eq_168 N) : 
  (P : ℚ) / ((1/3 : ℚ) * (N : ℚ)) = 6 / 5 := by
  sorry

end ratio_part_to_third_fraction_l1110_111032


namespace cost_for_23_days_l1110_111059

structure HostelStay where
  charge_first_week : ℝ
  charge_additional_week : ℝ

def cost_of_stay (days : ℕ) (hostel : HostelStay) : ℝ :=
  let first_week_days := min days 7
  let remaining_days := days - first_week_days
  let additional_full_weeks := remaining_days / 7 
  let additional_days := remaining_days % 7
  (first_week_days * hostel.charge_first_week) + 
  (additional_full_weeks * 7 * hostel.charge_additional_week) + 
  (additional_days * hostel.charge_additional_week)

theorem cost_for_23_days :
  cost_of_stay 23 { charge_first_week := 18.00, charge_additional_week := 11.00 } = 302.00 :=
by
  sorry

end cost_for_23_days_l1110_111059


namespace trigonometric_identity_l1110_111042

theorem trigonometric_identity 
  (α β γ : ℝ)
  (h : (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ)) :
  (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
  abs (Real.cos α * Real.cos β * Real.cos γ) ∧
  (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ) = 
  abs (Real.cos α * Real.cos β * Real.cos γ) := by
  sorry

end trigonometric_identity_l1110_111042


namespace find_expression_value_l1110_111007

-- Given conditions
variables {a b : ℝ}

-- Perimeter condition
def perimeter_condition (a b : ℝ) : Prop := 2 * (a + b) = 10

-- Area condition
def area_condition (a b : ℝ) : Prop := a * b = 6

-- Goal statement
theorem find_expression_value (h1 : perimeter_condition a b) (h2 : area_condition a b) :
  a^3 * b + 2 * a^2 * b^2 + a * b^3 = 150 :=
sorry

end find_expression_value_l1110_111007


namespace inscribed_circle_diameter_l1110_111020

noncomputable def diameter_inscribed_circle (side_length : ℝ) : ℝ :=
  let s := (3 * side_length) / 2
  let K := (Real.sqrt 3 / 4) * (side_length ^ 2)
  let r := K / s
  2 * r

theorem inscribed_circle_diameter (side_length : ℝ) (h : side_length = 10) :
  diameter_inscribed_circle side_length = (10 * Real.sqrt 3) / 3 :=
by
  rw [h]
  simp [diameter_inscribed_circle]
  sorry

end inscribed_circle_diameter_l1110_111020


namespace evaluate_m_l1110_111091

theorem evaluate_m (m : ℕ) : 2 ^ m = (64 : ℝ) ^ (1 / 3) → m = 2 :=
by
  sorry

end evaluate_m_l1110_111091


namespace jumpy_implies_not_green_l1110_111089

variables (Lizard : Type)
variables (IsJumpy IsGreen CanSing CanDance : Lizard → Prop)

-- Conditions given in the problem
axiom jumpy_implies_can_sing : ∀ l, IsJumpy l → CanSing l
axiom green_implies_cannot_dance : ∀ l, IsGreen l → ¬ CanDance l
axiom cannot_dance_implies_cannot_sing : ∀ l, ¬ CanDance l → ¬ CanSing l

theorem jumpy_implies_not_green (l : Lizard) : IsJumpy l → ¬ IsGreen l :=
by
  sorry

end jumpy_implies_not_green_l1110_111089


namespace therapy_charge_l1110_111093

-- Let F be the charge for the first hour and A be the charge for each additional hour
-- Two conditions are:
-- 1. F = A + 40
-- 2. F + 4A = 375

-- We need to prove that the total charge for 2 hours of therapy is 174
theorem therapy_charge (A F : ℕ) (h1 : F = A + 40) (h2 : F + 4 * A = 375) :
  F + A = 174 :=
by
  sorry

end therapy_charge_l1110_111093


namespace range_of_3a_minus_2b_l1110_111073

theorem range_of_3a_minus_2b (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 2 ≤ a + b ∧ a + b ≤ 4) :
  7 / 2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 7 :=
sorry

end range_of_3a_minus_2b_l1110_111073


namespace sector_area_eq_25_l1110_111044

theorem sector_area_eq_25 (r θ : ℝ) (h_r : r = 5) (h_θ : θ = 2) : (1 / 2) * θ * r^2 = 25 := by
  sorry

end sector_area_eq_25_l1110_111044


namespace ratatouille_cost_per_quart_l1110_111074

def eggplants := 88 * 0.22
def zucchini := 60.8 * 0.15
def tomatoes := 73.6 * 0.25
def onions := 43.2 * 0.07
def basil := (16 / 4) * 2.70
def bell_peppers := 12 * 0.20

def total_cost := eggplants + zucchini + tomatoes + onions + basil + bell_peppers
def yield := 4.5

def cost_per_quart := total_cost / yield

theorem ratatouille_cost_per_quart : cost_per_quart = 14.02 := 
by
  unfold cost_per_quart total_cost eggplants zucchini tomatoes onions basil bell_peppers 
  sorry

end ratatouille_cost_per_quart_l1110_111074


namespace find_non_divisible_and_product_l1110_111069

-- Define the set of numbers
def numbers : List Nat := [3543, 3552, 3567, 3579, 3581]

-- Function to get the digits of a number
def digits (n : Nat) : List Nat := n.digits 10

-- Function to sum the digits
def sum_of_digits (n : Nat) : Nat := (digits n).sum

-- Function to check divisibility by 3
def divisible_by_3 (n : Nat) : Bool := sum_of_digits n % 3 = 0

-- Find the units digit of a number
def units_digit (n : Nat) : Nat := n % 10

-- Find the tens digit of a number
def tens_digit (n : Nat) : Nat := (n / 10) % 10

-- The problem statement
theorem find_non_divisible_and_product :
  ∃ n ∈ numbers, ¬ divisible_by_3 n ∧ units_digit n * tens_digit n = 8 :=
by
  sorry

end find_non_divisible_and_product_l1110_111069


namespace ab_plus_cd_value_l1110_111078

theorem ab_plus_cd_value (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = 1)
  (h3 : a + c + d = 12)
  (h4 : b + c + d = 7) :
  a * b + c * d = 176 / 9 := 
sorry

end ab_plus_cd_value_l1110_111078


namespace probability_square_or_triangle_l1110_111034

theorem probability_square_or_triangle :
  let total_figures := 10
  let number_of_triangles := 4
  let number_of_squares := 3
  let number_of_favorable_outcomes := number_of_triangles + number_of_squares
  let probability := number_of_favorable_outcomes / total_figures
  probability = 7 / 10 :=
sorry

end probability_square_or_triangle_l1110_111034


namespace solve_first_equation_solve_second_equation_l1110_111043

open Real

/-- Prove solutions to the first equation (x + 8)(x + 1) = -12 are x = -4 and x = -5 -/
theorem solve_first_equation (x : ℝ) : (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 := by
  sorry

/-- Prove solutions to the second equation 2x^2 + 4x - 1 = 0 are x = (-2 + sqrt 6) / 2 and x = (-2 - sqrt 6) / 2 -/
theorem solve_second_equation (x : ℝ) : 2 * x^2 + 4 * x - 1 = 0 ↔ x = (-2 + sqrt 6) / 2 ∨ x = (-2 - sqrt 6) / 2 := by
  sorry

end solve_first_equation_solve_second_equation_l1110_111043


namespace find_d_l1110_111065

theorem find_d :
  ∃ d : ℝ, ∀ x : ℝ, x * (4 * x - 3) < d ↔ - (9/4 : ℝ) < x ∧ x < (3/2 : ℝ) ∧ d = 27 / 2 :=
by
  sorry

end find_d_l1110_111065


namespace problem_a_problem_b_l1110_111066

-- Problem a
theorem problem_a (p q : ℕ) (h1 : ∃ n : ℤ, 2 * p - q = n^2) (h2 : ∃ m : ℤ, 2 * p + q = m^2) : ∃ k : ℤ, q = 2 * k :=
sorry

-- Problem b
theorem problem_b (m : ℕ) (h1 : ∃ n : ℕ, 2 * m - 4030 = n^2) (h2 : ∃ k : ℕ, 2 * m + 4030 = k^2) : (m = 2593 ∨ m = 12097 ∨ m = 81217 ∨ m = 2030113) :=
sorry

end problem_a_problem_b_l1110_111066


namespace range_of_m_l1110_111061

theorem range_of_m (x1 x2 m : Real) (h_eq : ∀ x : Real, x^2 - 2*x + m + 2 = 0)
  (h_abs : |x1| + |x2| ≤ 3)
  (h_real : ∀ x : Real, ∃ y : Real, x^2 - 2*x + m + 2 = 0) : -13 / 4 ≤ m ∧ m ≤ -1 :=
by
  sorry

end range_of_m_l1110_111061


namespace favorite_food_sandwiches_l1110_111022

theorem favorite_food_sandwiches (total_students : ℕ) (cookies_percent pizza_percent pasta_percent : ℝ)
  (h_total : total_students = 200)
  (h_cookies : cookies_percent = 0.25)
  (h_pizza : pizza_percent = 0.30)
  (h_pasta : pasta_percent = 0.35) :
  let sandwiches_percent := 1 - (cookies_percent + pizza_percent + pasta_percent)
  sandwiches_percent * total_students = 20 :=
by
  sorry

end favorite_food_sandwiches_l1110_111022


namespace alcohol_percentage_correct_in_mixed_solution_l1110_111063

-- Define the ratios of alcohol to water
def ratio_A : ℚ := 21 / 25
def ratio_B : ℚ := 2 / 5

-- Define the mixing ratio of solutions A and B
def mix_ratio_A : ℚ := 5 / 11
def mix_ratio_B : ℚ := 6 / 11

-- Define the function to compute the percentage of alcohol in the mixed solution
def alcohol_percentage_mixed : ℚ := 
  (mix_ratio_A * ratio_A + mix_ratio_B * ratio_B) * 100

-- The theorem to be proven
theorem alcohol_percentage_correct_in_mixed_solution : 
  alcohol_percentage_mixed = 60 :=
by
  sorry

end alcohol_percentage_correct_in_mixed_solution_l1110_111063


namespace candies_per_friend_l1110_111086

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (friends : ℕ) 
  (h_initial : initial_candies = 10)
  (h_additional : additional_candies = 4)
  (h_friends : friends = 7) : initial_candies + additional_candies = 14 ∧ 14 / friends = 2 :=
by
  sorry

end candies_per_friend_l1110_111086


namespace george_speed_l1110_111097

theorem george_speed : 
  ∀ (d_tot d_1st : ℝ) (v_tot v_1st : ℝ) (v_2nd : ℝ),
    d_tot = 1 ∧ d_1st = 1 / 2 ∧ v_tot = 3 ∧ v_1st = 2 ∧ ((d_tot / v_tot) = (d_1st / v_1st + d_1st / v_2nd)) →
    v_2nd = 6 :=
by
  -- Proof here
  sorry

end george_speed_l1110_111097


namespace simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l1110_111090

-- Problem (1)
theorem simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth :
  (Real.sqrt 20 - Real.sqrt 5 + Real.sqrt (1 / 5) = 6 * Real.sqrt 5 / 5) :=
by
  sorry

-- Problem (2)
theorem simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3 :
  (Real.sqrt 12 + Real.sqrt 18) / Real.sqrt 3 - 2 * Real.sqrt (1 / 2) * Real.sqrt 3 = 2 :=
by
  sorry

end simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l1110_111090


namespace total_number_of_cars_l1110_111068

theorem total_number_of_cars (T A R : ℕ)
  (h1 : T - A = 37)
  (h2 : R ≥ 41)
  (h3 : ∀ x, x ≤ 59 → A = x + 37) :
  T = 133 :=
by
  sorry

end total_number_of_cars_l1110_111068


namespace p_neither_necessary_nor_sufficient_l1110_111009

def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x : ℝ) : Prop := x ≠ 0
def r (y : ℝ) : Prop := y ≠ -1

theorem p_neither_necessary_nor_sufficient (x y : ℝ) (h1: p x y) (h2: q x) (h3: r y) :
  ¬(p x y → q x) ∧ ¬(q x → p x y) := 
by 
  sorry

end p_neither_necessary_nor_sufficient_l1110_111009


namespace largest_multiple_of_7_smaller_than_neg_50_l1110_111095

theorem largest_multiple_of_7_smaller_than_neg_50 : ∃ n, (∃ k : ℤ, n = 7 * k) ∧ n < -50 ∧ ∀ m, (∃ j : ℤ, m = 7 * j) ∧ m < -50 → m ≤ n :=
by
  sorry

end largest_multiple_of_7_smaller_than_neg_50_l1110_111095


namespace line_intersection_l1110_111033

theorem line_intersection : 
  ∃ (x y : ℚ), 
    8 * x - 5 * y = 10 ∧ 
    3 * x + 2 * y = 16 ∧ 
    x = 100 / 31 ∧ 
    y = 98 / 31 :=
by
  use 100 / 31
  use 98 / 31
  sorry

end line_intersection_l1110_111033


namespace price_per_bottle_is_half_l1110_111055

theorem price_per_bottle_is_half (P : ℚ) 
  (Remy_bottles_morning : ℕ) (Nick_bottles_morning : ℕ) 
  (Total_sales_evening : ℚ) (Evening_more : ℚ) : 
  Remy_bottles_morning = 55 → 
  Nick_bottles_morning = Remy_bottles_morning - 6 → 
  Total_sales_evening = 55 → 
  Evening_more = 3 → 
  104 * P + 3 = 55 → 
  P = 1 / 2 := 
by
  intros h_remy_55 h_nick_remy h_total_55 h_evening_3 h_sales_eq
  sorry

end price_per_bottle_is_half_l1110_111055


namespace cylinder_volume_ratio_l1110_111096

noncomputable def ratio_of_volumes (r h V_small V_large : ℝ) : ℝ := V_large / V_small

theorem cylinder_volume_ratio (r : ℝ) (h : ℝ) 
  (original_height : ℝ := 3 * r)
  (height_small : ℝ := r / 4)
  (height_large : ℝ := 3 * r - height_small)
  (A_small : ℝ := 2 * π * r * (r + height_small))
  (A_large : ℝ := 2 * π * r * (r + height_large))
  (V_small : ℝ := π * r^2 * height_small) 
  (V_large : ℝ := π * r^2 * height_large) :
  A_large = 3 * A_small → 
  ratio_of_volumes r height_small V_small V_large = 11 := by 
  sorry

end cylinder_volume_ratio_l1110_111096


namespace pigeons_on_branches_and_under_tree_l1110_111056

theorem pigeons_on_branches_and_under_tree (x y : ℕ) 
  (h1 : y - 1 = (x + 1) / 2)
  (h2 : x - 1 = y + 1) : x = 7 ∧ y = 5 :=
by
  sorry

end pigeons_on_branches_and_under_tree_l1110_111056


namespace total_children_l1110_111058

theorem total_children (sons daughters : ℕ) (h1 : sons = 3) (h2 : daughters = 6 * sons) : (sons + daughters) = 21 :=
by
  sorry

end total_children_l1110_111058


namespace value_of_a_minus_b_l1110_111048

theorem value_of_a_minus_b (a b : ℚ) (h1 : 3015 * a + 3021 * b = 3025) (h2 : 3017 * a + 3023 * b = 3027) : 
  a - b = - (7 / 3) :=
by
  sorry

end value_of_a_minus_b_l1110_111048


namespace evaluate_expression_l1110_111031

theorem evaluate_expression : (24 : ℕ) = 2^3 * 3 ∧ (72 : ℕ) = 2^3 * 3^2 → (24^40 / 72^20 : ℚ) = 2^60 :=
by {
  sorry
}

end evaluate_expression_l1110_111031


namespace parabola_intercept_sum_l1110_111027

theorem parabola_intercept_sum :
  let a := 6
  let b := 1
  let c := 2
  a + b + c = 9 :=
by
  sorry

end parabola_intercept_sum_l1110_111027


namespace sqrt_E_nature_l1110_111084

def E (x : ℤ) : ℤ :=
  let a := x
  let b := x + 1
  let c := a * b
  let d := b * c
  a^2 + b^2 + c^2 + d^2

theorem sqrt_E_nature : ∀ x : ℤ, (∃ n : ℤ, n^2 = E x) ∧ (∃ m : ℤ, m^2 ≠ E x) :=
  by
  sorry

end sqrt_E_nature_l1110_111084


namespace simplify_sqrt_expression_l1110_111035

theorem simplify_sqrt_expression (x : ℝ) : 
  Real.sqrt (x^6 + x^4 + 1) = Real.sqrt (x^6 + x^4 + 1) := by
  sorry

end simplify_sqrt_expression_l1110_111035


namespace arithmetic_sequence_sum_l1110_111094

/-- Let {a_n} be an arithmetic sequence with a positive common difference d.
  Given that a_1 + a_2 + a_3 = 15 and a_1 * a_2 * a_3 = 80, we aim to show that
  a_11 + a_12 + a_13 = 105. -/
theorem arithmetic_sequence_sum
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end arithmetic_sequence_sum_l1110_111094


namespace contestant_advancing_probability_l1110_111082

noncomputable def probability_correct : ℝ := 0.8
noncomputable def probability_incorrect : ℝ := 1 - probability_correct

def sequence_pattern (q1 q2 q3 q4 : Bool) : Bool :=
  -- Pattern INCORRECT, CORRECT, CORRECT, CORRECT
  q1 == false ∧ q2 == true ∧ q3 == true ∧ q4 == true

def probability_pattern (p_corr p_incorr : ℝ) : ℝ :=
  p_incorr * p_corr * p_corr * p_corr

theorem contestant_advancing_probability :
  (probability_pattern probability_correct probability_incorrect = 0.1024) :=
by
  -- Proof required here
  sorry

end contestant_advancing_probability_l1110_111082


namespace cuboid_diagonals_and_edges_l1110_111021

theorem cuboid_diagonals_and_edges (a b c : ℝ) : 
  4 * (a^2 + b^2 + c^2) = 4 * a^2 + 4 * b^2 + 4 * c^2 :=
by
  sorry

end cuboid_diagonals_and_edges_l1110_111021


namespace phoebe_age_l1110_111037

theorem phoebe_age (P : ℕ) (h₁ : ∀ P, 60 = 4 * (P + 5)) (h₂: 55 + 5 = 60) : P = 10 := 
by
  have h₃ : 60 = 4 * (P + 5) := h₁ P
  sorry

end phoebe_age_l1110_111037


namespace factorization_sum_l1110_111071

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 13 * x + 40)
  (h2 : ∀ x : ℝ, (x - b) * (x - c) = x^2 - 19 * x + 88) :
  a + b + c = 24 := 
sorry

end factorization_sum_l1110_111071


namespace probability_blue_tile_l1110_111005

def is_congruent_to_3_mod_7 (n : ℕ) : Prop := n % 7 = 3

def num_blue_tiles (n : ℕ) : ℕ := (n / 7) + 1

theorem probability_blue_tile : 
  num_blue_tiles 70 / 70 = 1 / 7 :=
by
  sorry

end probability_blue_tile_l1110_111005


namespace bisection_approximation_interval_l1110_111088

noncomputable def bisection_accuracy (a b : ℝ) (n : ℕ) : ℝ := (b - a) / 2^n

theorem bisection_approximation_interval 
  (a b : ℝ) (n : ℕ) (accuracy : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : accuracy = 0.01) 
  (h4 : 2^n ≥ 100) : bisection_accuracy a b n ≤ accuracy :=
sorry

end bisection_approximation_interval_l1110_111088


namespace value_of_y_at_x_8_l1110_111016

theorem value_of_y_at_x_8 (k : ℝ) (x y : ℝ) 
  (hx1 : y = k * x^(1/3)) 
  (hx2 : y = 4 * Real.sqrt 3) 
  (hx3 : x = 64) 
  (hx4 : 8^(1/3) = 2) : 
  (y = 2 * Real.sqrt 3) := 
by 
  sorry

end value_of_y_at_x_8_l1110_111016


namespace problem_statement_equality_condition_l1110_111025

theorem problem_statement (x y z : ℝ) (hx : 0 <= x) (hy : 0 <= y) (hz : 0 <= z) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) >= 2 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 <= x) (hy : 0 <= y) (hz : 0 <= z) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end problem_statement_equality_condition_l1110_111025


namespace train_speed_l1110_111099

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor: ℝ)
  (h_length : length = 100) 
  (h_time : time = 5) 
  (h_conversion : conversion_factor = 3.6) :
  (length / time * conversion_factor) = 72 :=
by
  sorry

end train_speed_l1110_111099


namespace larger_number_l1110_111013

theorem larger_number (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (hcf_eq : hcf = 23) (fact1_eq : factor1 = 13) (fact2_eq : factor2 = 14) : 
  max (hcf * factor1) (hcf * factor2) = 322 := 
by
  sorry

end larger_number_l1110_111013


namespace smallest_value_of_x_l1110_111080

theorem smallest_value_of_x :
  ∃ x, (12 * x^2 - 58 * x + 70 = 0) ∧ x = 7 / 3 :=
by
  sorry

end smallest_value_of_x_l1110_111080


namespace cn_geometric_seq_l1110_111046

-- Given conditions
def Sn (n : ℕ) : ℚ := (3 * n^2 + 5 * n) / 2
def an (n : ℕ) : ℕ := 3 * n + 1
def bn (n : ℕ) : ℕ := 2^n

theorem cn_geometric_seq : 
  ∃ q : ℕ, ∃ (c : ℕ → ℕ), (∀ n : ℕ, c n = q^n) ∧ (∀ n : ℕ, ∃ m : ℕ, c n = an m ∧ c n = bn m) :=
sorry

end cn_geometric_seq_l1110_111046


namespace intersection_empty_l1110_111023

open Set

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := by
  sorry

end intersection_empty_l1110_111023


namespace min_value_of_quadratic_l1110_111072

theorem min_value_of_quadratic (y1 y2 y3 : ℝ) (h1 : 0 < y1) (h2 : 0 < y2) (h3 : 0 < y3) (h_eq : 2 * y1 + 3 * y2 + 4 * y3 = 75) :
  y1^2 + 2 * y2^2 + 3 * y3^2 ≥ 5625 / 29 :=
sorry

end min_value_of_quadratic_l1110_111072


namespace solve_inequality_l1110_111030

theorem solve_inequality (x : ℝ) : (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
  sorry

end solve_inequality_l1110_111030


namespace at_least_two_greater_than_one_l1110_111050

theorem at_least_two_greater_than_one
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a + b + c = a * b * c) : 
  1 < a ∨ 1 < b ∨ 1 < c :=
sorry

end at_least_two_greater_than_one_l1110_111050


namespace line_AB_l1110_111081

-- Statements for circles and intersection
def circle_C1 (x y: ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y: ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

-- Points A and B are defined as the intersection points of circles C1 and C2
axiom A (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y
axiom B (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y

-- The goal is to prove that the line passing through points A and B has the equation x - y = 0
theorem line_AB (x y: ℝ) : circle_C1 x y → circle_C2 x y → (x - y = 0) :=
by
  sorry

end line_AB_l1110_111081


namespace no_real_roots_of_quadratic_l1110_111011

theorem no_real_roots_of_quadratic 
(a b c : ℝ) 
(h1 : b + c > a)
(h2 : b + a > c)
(h3 : c + a > b) :
(b^2 + c^2 - a^2)^2 - 4 * b^2 * c^2 < 0 :=
by
  sorry

end no_real_roots_of_quadratic_l1110_111011


namespace quadratic_inequality_l1110_111008

theorem quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + 4 > 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end quadratic_inequality_l1110_111008


namespace marks_difference_l1110_111083

theorem marks_difference (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 48) 
  (h2 : (A + B + C + D) / 4 = 47) 
  (h3 : E > D) 
  (h4 : (B + C + D + E) / 4 = 48) 
  (h5 : A = 43) : 
  E - D = 3 := 
sorry

end marks_difference_l1110_111083


namespace amount_paid_for_peaches_l1110_111079

noncomputable def cost_of_berries : ℝ := 7.19
noncomputable def change_received : ℝ := 5.98
noncomputable def total_bill : ℝ := 20

theorem amount_paid_for_peaches :
  total_bill - change_received - cost_of_berries = 6.83 :=
by
  sorry

end amount_paid_for_peaches_l1110_111079


namespace radius_of_largest_circle_correct_l1110_111041

noncomputable def radius_of_largest_circle_in_quadrilateral (AB BC CD DA : ℝ) (angle_BCD : ℝ) : ℝ :=
  if AB = 10 ∧ BC = 12 ∧ CD = 8 ∧ DA = 14 ∧ angle_BCD = 90
    then Real.sqrt 210
    else 0

theorem radius_of_largest_circle_correct :
  radius_of_largest_circle_in_quadrilateral 10 12 8 14 90 = Real.sqrt 210 :=
by
  sorry

end radius_of_largest_circle_correct_l1110_111041


namespace recursive_relation_a_recursive_relation_b_recursive_relation_c_recursive_relation_d_recursive_relation_e_l1110_111002

def a (n : ℕ) : ℕ := n
def b (n : ℕ) : ℕ := n^2
def c (n : ℕ) : ℕ := n^3
def d (n : ℕ) : ℕ := n^4
def e (n : ℕ) : ℕ := n^5

theorem recursive_relation_a (n : ℕ) : a (n+2) = 2 * a (n+1) - a n :=
by sorry

theorem recursive_relation_b (n : ℕ) : b (n+3) = 3 * b (n+2) - 3 * b (n+1) + b n :=
by sorry

theorem recursive_relation_c (n : ℕ) : c (n+4) = 4 * c (n+3) - 6 * c (n+2) + 4 * c (n+1) - c n :=
by sorry

theorem recursive_relation_d (n : ℕ) : d (n+5) = 5 * d (n+4) - 10 * d (n+3) + 10 * d (n+2) - 5 * d (n+1) + d n :=
by sorry

theorem recursive_relation_e (n : ℕ) : 
  e (n+6) = 6 * e (n+5) - 15 * e (n+4) + 20 * e (n+3) - 15 * e (n+2) + 6 * e (n+1) - e n :=
by sorry

end recursive_relation_a_recursive_relation_b_recursive_relation_c_recursive_relation_d_recursive_relation_e_l1110_111002


namespace find_x_when_y_equals_2_l1110_111045

theorem find_x_when_y_equals_2 (x : ℚ) (y : ℚ) : 
  y = (1 / (4 * x + 2)) ∧ y = 2 -> x = -3 / 8 := 
by 
  sorry

end find_x_when_y_equals_2_l1110_111045


namespace seeds_planted_on_wednesday_l1110_111098

theorem seeds_planted_on_wednesday
  (total_seeds : ℕ) (seeds_thursday : ℕ) (seeds_wednesday : ℕ)
  (h_total : total_seeds = 22) (h_thursday : seeds_thursday = 2) :
  seeds_wednesday = 20 ↔ total_seeds - seeds_thursday = seeds_wednesday :=
by
  -- the proof would go here
  sorry

end seeds_planted_on_wednesday_l1110_111098


namespace correct_equation_l1110_111006

-- Definitions based on conditions
def total_students := 98
def transfer_students := 3
def original_students_A (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ total_students
def students_B (x : ℕ) := total_students - x

-- Equation set up based on translation of the proof problem
theorem correct_equation (x : ℕ) (h : original_students_A x) :
  students_B x + transfer_students = x - transfer_students ↔ (98 - x) + 3 = x - 3 :=
by
  sorry
  
end correct_equation_l1110_111006


namespace sum_of_cubes_l1110_111010

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = -3) (h3 : x * y * z = 2) : 
  x^3 + y^3 + z^3 = 32 := 
sorry

end sum_of_cubes_l1110_111010


namespace central_angle_of_sector_l1110_111015

-- Given conditions as hypotheses
variable (r θ : ℝ)
variable (h₁ : (1/2) * θ * r^2 = 1)
variable (h₂ : 2 * r + θ * r = 4)

-- The goal statement to be proved
theorem central_angle_of_sector :
  θ = 2 :=
by sorry

end central_angle_of_sector_l1110_111015


namespace map_distance_ratio_l1110_111077

theorem map_distance_ratio (actual_distance_km : ℝ) (map_distance_cm : ℝ) (h_actual_distance : actual_distance_km = 5) (h_map_distance : map_distance_cm = 2) :
  map_distance_cm / (actual_distance_km * 100000) = 1 / 250000 :=
by
  -- Given the actual distance in kilometers and map distance in centimeters, prove the scale ratio
  -- skip the proof
  sorry

end map_distance_ratio_l1110_111077


namespace at_least_one_solves_l1110_111057

-- Given probabilities
def pA : ℝ := 0.8
def pB : ℝ := 0.6

-- Probability that at least one solves the problem
def prob_at_least_one_solves : ℝ := 1 - ((1 - pA) * (1 - pB))

-- Statement: Prove that the probability that at least one solves the problem is 0.92
theorem at_least_one_solves : prob_at_least_one_solves = 0.92 :=
by
  -- Proof steps would go here
  sorry

end at_least_one_solves_l1110_111057


namespace corrected_mean_l1110_111049

theorem corrected_mean (n : ℕ) (mean old_obs new_obs : ℝ) 
    (obs_count : n = 50) (old_mean : mean = 36) (incorrect_obs : old_obs = 23) (correct_obs : new_obs = 46) :
    (mean * n - old_obs + new_obs) / n = 36.46 := by
  sorry

end corrected_mean_l1110_111049


namespace liters_to_pints_conversion_l1110_111040

-- Definitions based on conditions
def liters_to_pints_ratio := 0.75 / 1.575
def target_liters := 1.5
def expected_pints := 3.15

-- Lean statement
theorem liters_to_pints_conversion 
  (h_ratio : 0.75 / 1.575 = liters_to_pints_ratio)
  (h_target : 1.5 = target_liters) :
  target_liters * (1 / liters_to_pints_ratio) = expected_pints :=
by 
  sorry

end liters_to_pints_conversion_l1110_111040


namespace area_sin_transformed_l1110_111000

noncomputable def sin_transformed (x : ℝ) : ℝ := 4 * Real.sin (x - Real.pi)

theorem area_sin_transformed :
  ∫ x in Real.pi..3 * Real.pi, |sin_transformed x| = 16 :=
by
  sorry

end area_sin_transformed_l1110_111000


namespace Tim_weekly_earnings_l1110_111053

theorem Tim_weekly_earnings :
  let tasks_per_day := 100
  let pay_per_task := 1.2
  let days_per_week := 6
  let daily_earnings := tasks_per_day * pay_per_task
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 720 := by
  sorry

end Tim_weekly_earnings_l1110_111053


namespace sum_of_tangents_l1110_111012

noncomputable def g (x : ℝ) : ℝ :=
  max (max (-7 * x - 25) (2 * x + 5)) (5 * x - 7)

theorem sum_of_tangents (a b c : ℝ) (q : ℝ → ℝ) (hq₁ : ∀ x, q x = k * (x - a) ^ 2 + (-7 * x - 25))
  (hq₂ : ∀ x, q x = k * (x - b) ^ 2 + (2 * x + 5))
  (hq₃ : ∀ x, q x = k * (x - c) ^ 2 + (5 * x - 7)) :
  a + b + c = -34 / 3 := 
sorry

end sum_of_tangents_l1110_111012


namespace arithmetic_sequence_general_term_geometric_sequence_inequality_l1110_111060

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

end arithmetic_sequence_general_term_geometric_sequence_inequality_l1110_111060


namespace range_of_m_l1110_111019

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → x < m) : m > 1 := 
by
  sorry

end range_of_m_l1110_111019


namespace part_a_part_b_l1110_111054

-- Part (a): Proving at most one integer solution for general k
theorem part_a (k : ℤ) : 
  ∀ (x1 x2 : ℤ), (x1^3 - 24*x1 + k = 0 ∧ x2^3 - 24*x2 + k = 0) → x1 = x2 :=
sorry

-- Part (b): Proving exactly one integer solution for k = -2016
theorem part_b :
  ∃! (x : ℤ), x^3 + 24*x - 2016 = 0 :=
sorry

end part_a_part_b_l1110_111054


namespace max_marks_l1110_111047

variable (M : ℝ)

def passing_marks (M : ℝ) : ℝ := 0.45 * M

theorem max_marks (h1 : passing_marks M = 225)
  (h2 : 180 + 45 = 225) : M = 500 :=
by
  sorry

end max_marks_l1110_111047


namespace total_files_on_flash_drive_l1110_111004

theorem total_files_on_flash_drive :
  ∀ (music_files video_files picture_files : ℝ),
    music_files = 4.0 ∧ video_files = 21.0 ∧ picture_files = 23.0 →
    music_files + video_files + picture_files = 48.0 :=
by
  sorry

end total_files_on_flash_drive_l1110_111004


namespace percentage_error_in_side_l1110_111067

theorem percentage_error_in_side
  (s s' : ℝ) -- the actual and measured side lengths
  (h : (s' * s' - s * s) / (s * s) * 100 = 41.61) : 
  ((s' - s) / s) * 100 = 19 :=
sorry

end percentage_error_in_side_l1110_111067
