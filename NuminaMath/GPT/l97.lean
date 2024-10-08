import Mathlib

namespace white_pieces_total_l97_97117

theorem white_pieces_total (B W : ℕ) 
  (h_total_pieces : B + W = 300) 
  (h_total_piles : 100 * 3 = B + W) 
  (h_piles_1_white : {n : ℕ | n = 27}) 
  (h_piles_2_3_black : {m : ℕ | m = 42}) 
  (h_piles_3_black_3_white : 15 = 15) :
  W = 158 :=
by
  sorry

end white_pieces_total_l97_97117


namespace wrappers_after_collection_l97_97177

theorem wrappers_after_collection (caps_found : ℕ) (wrappers_found : ℕ) (current_caps : ℕ) (initial_caps : ℕ) : 
  caps_found = 22 → wrappers_found = 30 → current_caps = 17 → initial_caps = 0 → 
  wrappers_found ≥ 30 := 
by 
  intros h1 h2 h3 h4
  -- Solution steps are omitted on purpose
  --- This is where the proof is written
  sorry

end wrappers_after_collection_l97_97177


namespace unique_solution_to_function_equation_l97_97485

theorem unique_solution_to_function_equation (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (2 * n) = 2 * f n)
  (h2 : ∀ n : ℕ, f (2 * n + 1) = 2 * f n + 1) :
  ∀ n : ℕ, f n = n :=
by
  sorry

end unique_solution_to_function_equation_l97_97485


namespace tangent_line_count_l97_97383

noncomputable def circles_tangent_lines (r1 r2 d : ℝ) : ℕ :=
if d = |r1 - r2| then 1 else 0 -- Define the function based on the problem statement

theorem tangent_line_count :
  circles_tangent_lines 4 5 3 = 1 := 
by
  -- Placeholder for the proof, which we are skipping as per instructions
  sorry

end tangent_line_count_l97_97383


namespace max_moves_440_l97_97758

-- Define the set of initial numbers
def initial_numbers : List ℕ := List.range' 1 22

-- Define what constitutes a valid move
def is_valid_move (a b : ℕ) : Prop := b ≥ a + 2

-- Perform the move operation
def perform_move (numbers : List ℕ) (a b : ℕ) : List ℕ :=
  (numbers.erase a).erase b ++ [a + 1, b - 1]

-- Define the maximum number of moves we need to prove
theorem max_moves_440 : ∃ m, m = 440 ∧
  ∀ (moves_done : ℕ) (numbers : List ℕ),
    moves_done <= m → ∃ a b, a ∈ numbers ∧ b ∈ numbers ∧
                             is_valid_move a b ∧
                             numbers = initial_numbers →
                             perform_move numbers a b ≠ numbers
 := sorry

end max_moves_440_l97_97758


namespace problem_quadrilateral_inscribed_in_circle_l97_97109

theorem problem_quadrilateral_inscribed_in_circle
  (r : ℝ)
  (AB BC CD DA : ℝ)
  (h_radius : r = 300 * Real.sqrt 2)
  (h_AB : AB = 300)
  (h_BC : BC = 150)
  (h_CD : CD = 150) :
  DA = 750 :=
sorry

end problem_quadrilateral_inscribed_in_circle_l97_97109


namespace option_C_not_like_terms_l97_97142

theorem option_C_not_like_terms :
  ¬ (2 * (m : ℝ) == 2 * (n : ℝ)) :=
by
  sorry

end option_C_not_like_terms_l97_97142


namespace small_cube_edge_length_l97_97446

theorem small_cube_edge_length 
  (m n : ℕ)
  (h1 : 12 % m = 0) 
  (h2 : n = 12 / m) 
  (h3 : 6 * (n - 2)^2 = 12 * (n - 2)) 
  : m = 3 :=
by 
  sorry

end small_cube_edge_length_l97_97446


namespace increased_cost_per_person_l97_97998

-- Declaration of constants
def initial_cost : ℕ := 30000000000 -- 30 billion dollars in dollars
def people_sharing : ℕ := 300000000 -- 300 million people
def inflation_rate : ℝ := 0.10 -- 10% inflation rate

-- Calculation of increased cost per person
theorem increased_cost_per_person : (initial_cost * (1 + inflation_rate) / people_sharing) = 110 :=
by sorry

end increased_cost_per_person_l97_97998


namespace pebbles_sum_at_12_days_l97_97171

def pebbles_collected (n : ℕ) : ℕ :=
  if n = 0 then 0 else n + pebbles_collected (n - 1)

theorem pebbles_sum_at_12_days : pebbles_collected 12 = 78 := by
  -- This would be the place for the proof, but adding sorry as instructed.
  sorry

end pebbles_sum_at_12_days_l97_97171


namespace range_b_values_l97_97431

theorem range_b_values (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : ∀ x, f x = Real.exp x - 1) 
  (hg : ∀ x, g x = -x^2 + 4*x - 3) 
  (h : f a = g b) : 
  b ∈ Set.univ :=
by sorry

end range_b_values_l97_97431


namespace tan_sum_angles_l97_97607

theorem tan_sum_angles : (Real.tan (17 * Real.pi / 180) + Real.tan (28 * Real.pi / 180)) / (1 - Real.tan (17 * Real.pi / 180) * Real.tan (28 * Real.pi / 180)) = 1 := 
by sorry

end tan_sum_angles_l97_97607


namespace minimum_cable_length_l97_97941

def station_positions : List ℝ := [0, 3, 7, 11, 14]

def total_cable_length (x : ℝ) : ℝ :=
  abs x + abs (x - 3) + abs (x - 7) + abs (x - 11) + abs (x - 14)

theorem minimum_cable_length :
  (∀ x : ℝ, total_cable_length x ≥ 22) ∧ total_cable_length 7 = 22 :=
by
  sorry

end minimum_cable_length_l97_97941


namespace decimalToFrac_l97_97380

theorem decimalToFrac : (145 / 100 : ℚ) = 29 / 20 := by
  sorry

end decimalToFrac_l97_97380


namespace smallest_positive_integer_x_for_cube_l97_97387

theorem smallest_positive_integer_x_for_cube (x : ℕ) (h1 : 1512 = 2^3 * 3^3 * 7) (h2 : ∀ n : ℕ, n > 0 → ∃ k : ℕ, 1512 * n = k^3) : x = 49 :=
sorry

end smallest_positive_integer_x_for_cube_l97_97387


namespace log_increasing_condition_log_increasing_not_necessary_l97_97773

theorem log_increasing_condition (a : ℝ) (h : a > 2) : a > 1 :=
by sorry

theorem log_increasing_not_necessary (a : ℝ) : ∃ b, (b > 1 ∧ ¬(b > 2)) :=
by sorry

end log_increasing_condition_log_increasing_not_necessary_l97_97773


namespace factor_expression_l97_97355

theorem factor_expression (x : ℝ) : 6 * x ^ 3 - 54 * x = 6 * x * (x + 3) * (x - 3) :=
by {
  sorry
}

end factor_expression_l97_97355


namespace a_minus_b_is_neg_seven_l97_97626

-- Definitions for sets
def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 1 < x ∧ x < 4}
def setC : Set ℝ := {x | 1 < x ∧ x < 3}

-- Proving the statement
theorem a_minus_b_is_neg_seven :
  ∀ (a b : ℝ), (∀ x, (x ∈ setC) ↔ (x^2 + a*x + b < 0)) → a - b = -7 :=
by
  intros a b h
  sorry

end a_minus_b_is_neg_seven_l97_97626


namespace compute_expression_l97_97645

theorem compute_expression : 1013^2 - 987^2 - 1007^2 + 993^2 = 24000 := by
  sorry

end compute_expression_l97_97645


namespace max_students_for_distribution_l97_97859

theorem max_students_for_distribution : 
  ∃ (n : Nat), (∀ k, k ∣ 1048 ∧ k ∣ 828 → k ≤ n) ∧ 
               (n ∣ 1048 ∧ n ∣ 828) ∧ 
               n = 4 :=
by
  sorry

end max_students_for_distribution_l97_97859


namespace find_interest_rate_l97_97409

theorem find_interest_rate 
    (P : ℝ) (T : ℝ) (known_rate : ℝ) (diff : ℝ) (R : ℝ) :
    P = 7000 → T = 2 → known_rate = 0.18 → diff = 840 → (P * known_rate * T - (P * (R/100) * T) = diff) → R = 12 :=
by
  intros P_eq T_eq kr_eq diff_eq interest_eq
  simp only [P_eq, T_eq, kr_eq, diff_eq] at interest_eq
-- Solving equation is not required
  sorry

end find_interest_rate_l97_97409


namespace find_constant_a_l97_97824

noncomputable def f (a t : ℝ) : ℝ := (t - 2)^2 - 4 - a

theorem find_constant_a :
  (∃ (a : ℝ),
    (∀ (t : ℝ), -1 ≤ t ∧ t ≤ 1 → |f a t| ≤ 4) ∧ 
    (∃ (t : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ |f a t| = 4)) →
  a = 1 :=
sorry

end find_constant_a_l97_97824


namespace trig_expression_value_l97_97806

theorem trig_expression_value (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : 
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 := 
by
  sorry

end trig_expression_value_l97_97806


namespace range_of_slopes_of_line_AB_l97_97259

variables {x y : ℝ}

/-- (O is the coordinate origin),
    (the parabola y² = 4x),
    (points A and B in the first quadrant),
    (the product of the slopes of lines OA and OB being 1) -/
theorem range_of_slopes_of_line_AB
  (O : ℝ) 
  (A B : ℝ × ℝ)
  (hxA : 0 < A.fst)
  (hyA : 0 < A.snd)
  (hxB : 0 < B.fst)
  (hyB : 0 < B.snd)
  (hA_on_parabola : A.snd^2 = 4 * A.fst)
  (hB_on_parabola : B.snd^2 = 4 * B.fst)
  (h_product_slopes : (A.snd / A.fst) * (B.snd / B.fst) = 1) :
  (0 < (B.snd - A.snd) / (B.fst - A.fst) ∧ (B.snd - A.snd) / (B.fst - A.fst) < 1/2) := 
by
  sorry

end range_of_slopes_of_line_AB_l97_97259


namespace cost_per_pound_peanuts_l97_97639

-- Defining the conditions as needed for our problem
def one_dollar_bills := 7
def five_dollar_bills := 4
def ten_dollar_bills := 2
def twenty_dollar_bills := 1
def change := 4
def pounds_per_day := 3
def days_in_week := 7

-- Calculating the total initial amount of money Frank has
def total_initial_money := (one_dollar_bills * 1) + (five_dollar_bills * 5) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)

-- Calculating the total amount spent on peanuts
def total_spent := total_initial_money - change

-- Calculating the total pounds of peanuts
def total_pounds := pounds_per_day * days_in_week

-- The proof statement
theorem cost_per_pound_peanuts : total_spent / total_pounds = 3 := sorry

end cost_per_pound_peanuts_l97_97639


namespace deficit_percentage_l97_97876

variable (A B : ℝ) -- Actual lengths of the sides of the rectangle
variable (x : ℝ) -- Percentage in deficit
variable (measuredA := A * 1.06) -- One side measured 6% in excess
variable (errorPercent := 0.7) -- Error percent in area
variable (measuredB := B * (1 - x / 100)) -- Other side measured x% in deficit
variable (actualArea := A * B) -- Actual area of the rectangle
variable (calculatedArea := (A * 1.06) * (B * (1 - x / 100))) -- Calculated area with measurement errors
variable (correctArea := actualArea * (1 + errorPercent / 100)) -- Correct area considering the error

theorem deficit_percentage : 
  calculatedArea = correctArea → 
  x = 5 :=
by
  sorry

end deficit_percentage_l97_97876


namespace prove_range_of_p_l97_97417

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x - 1

def A (x : ℝ) : Prop := x > 2
def no_pre_image_in_A (p : ℝ) : Prop := ∀ x, A x → f x ≠ p

theorem prove_range_of_p (p : ℝ) : no_pre_image_in_A p ↔ p > -1 := by
  sorry

end prove_range_of_p_l97_97417


namespace weight_of_5_moles_H₂CO₃_l97_97113

-- Definitions based on the given conditions
def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_H₂CO₃_H : ℕ := 2
def num_H₂CO₃_C : ℕ := 1
def num_H₂CO₃_O : ℕ := 3

def molecular_weight (num_H num_C num_O : ℕ) 
                     (weight_H weight_C weight_O : ℝ) : ℝ :=
  num_H * weight_H + num_C * weight_C + num_O * weight_O

-- Main proof statement
theorem weight_of_5_moles_H₂CO₃ :
  5 * molecular_weight num_H₂CO₃_H num_H₂CO₃_C num_H₂CO₃_O 
                       atomic_weight_H atomic_weight_C atomic_weight_O 
  = 310.12 := by
  sorry

end weight_of_5_moles_H₂CO₃_l97_97113


namespace fraction_taken_by_kiley_l97_97398

-- Define the constants and conditions
def total_crayons : ℕ := 48
def remaining_crayons_after_joe : ℕ := 18

-- Define the main statement to be proven
theorem fraction_taken_by_kiley (f : ℚ) : 
  (48 - (48 * f)) / 2 = 18 → f = 1 / 4 :=
by 
  intro h
  sorry

end fraction_taken_by_kiley_l97_97398


namespace number_2120_in_33rd_group_l97_97356

def last_number_in_group (n : ℕ) := 2 * n * (n + 1)

theorem number_2120_in_33rd_group :
  ∃ n, n = 33 ∧ (last_number_in_group (n - 1) < 2120) ∧ (2120 <= last_number_in_group n) :=
sorry

end number_2120_in_33rd_group_l97_97356


namespace prove_weight_loss_l97_97942

variable (W : ℝ) -- Original weight
variable (x : ℝ) -- Percentage of weight lost

def weight_equation := W - (x / 100) * W + (2 / 100) * W = (89.76 / 100) * W

theorem prove_weight_loss (h : weight_equation W x) : x = 12.24 :=
by
  sorry

end prove_weight_loss_l97_97942


namespace tan_three_theta_l97_97159

theorem tan_three_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_three_theta_l97_97159


namespace minimize_slope_at_one_l97_97980

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * x^2 - (1 / (a * x))

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  4 * a * x - (1 / (a * x^2))

noncomputable def slope_at_one (a : ℝ) : ℝ :=
  f_deriv a 1

theorem minimize_slope_at_one : ∀ a : ℝ, a > 0 → slope_at_one a ≥ 4 ∧ (slope_at_one a = 4 ↔ a = 1 / 2) :=
by 
  sorry

end minimize_slope_at_one_l97_97980


namespace simplify_and_evaluate_expression_l97_97838

theorem simplify_and_evaluate_expression (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) : 
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) :=
by
  sorry

example : (∃ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ (x = 3) ∧ ((x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = 5)) :=
  ⟨3, by norm_num, by norm_num, rfl, by norm_num⟩

end simplify_and_evaluate_expression_l97_97838


namespace distance_center_to_plane_l97_97246

theorem distance_center_to_plane (r : ℝ) (a b : ℝ) (h : a ^ 2 + b ^ 2 = 10 ^ 2) (d : ℝ) : 
  r = 13 → a = 6 → b = 8 → d = 12 := 
by 
  sorry

end distance_center_to_plane_l97_97246


namespace camera_sticker_price_l97_97705

theorem camera_sticker_price (p : ℝ)
  (h1 : p > 0)
  (hx : ∀ x, x = 0.80 * p - 50)
  (hy : ∀ y, y = 0.65 * p)
  (hs : 0.80 * p - 50 = 0.65 * p - 40) :
  p = 666.67 :=
by sorry

end camera_sticker_price_l97_97705


namespace probability_of_rolling_greater_than_five_l97_97119

def probability_of_greater_than_five (dice_faces : Finset ℕ) (greater_than : ℕ) : ℚ := 
  let favorable_outcomes := dice_faces.filter (λ x => x > greater_than)
  favorable_outcomes.card / dice_faces.card

theorem probability_of_rolling_greater_than_five:
  probability_of_greater_than_five ({1, 2, 3, 4, 5, 6} : Finset ℕ) 5 = 1 / 6 :=
by
  sorry

end probability_of_rolling_greater_than_five_l97_97119


namespace animals_total_l97_97433

-- Given definitions and conditions
def ducks : ℕ := 25
def rabbits : ℕ := 8
def chickens := 4 * ducks

-- Proof statement
theorem animals_total (h1 : chickens = 4 * ducks)
                     (h2 : ducks - 17 = rabbits)
                     (h3 : rabbits = 8) :
  chickens + ducks + rabbits = 133 := by
  sorry

end animals_total_l97_97433


namespace problem_inequality_l97_97225

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem problem_inequality (a : ℝ) (m n : ℝ) 
  (h1 : m ∈ Set.Icc 0 2) (h2 : n ∈ Set.Icc 0 2) 
  (h3 : |m - n| ≥ 1) 
  (h4 : f m a / f n a = 1) : 
  1 ≤ a / (Real.exp 1 - 1) ∧ a / (Real.exp 1 - 1) ≤ Real.exp 1 :=
by sorry

end problem_inequality_l97_97225


namespace speed_at_perigee_l97_97612

-- Define the conditions
def semi_major_axis (a : ℝ) := a > 0
def perigee_distance (a : ℝ) := 0.5 * a
def point_P_distance (a : ℝ) := 0.75 * a
def speed_at_P (v1 : ℝ) := v1 > 0

-- Define what we need to prove
theorem speed_at_perigee (a v1 v2 : ℝ) (h1 : semi_major_axis a) (h2 : speed_at_P v1) :
  v2 = (3 / Real.sqrt 5) * v1 :=
sorry

end speed_at_perigee_l97_97612


namespace inequality_bound_l97_97537

theorem inequality_bound (a b c d e p q : ℝ) (hpq : 0 < p ∧ p ≤ q)
  (ha : p ≤ a ∧ a ≤ q) (hb : p ≤ b ∧ b ≤ q) (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
sorry

end inequality_bound_l97_97537


namespace f_sub_f_neg_l97_97056

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + 7 * x

-- State the theorem
theorem f_sub_f_neg : f 3 - f (-3) = 582 :=
by
  -- Definitions and calculations for the proof
  -- (You can complete this part in later proof development)
  sorry

end f_sub_f_neg_l97_97056


namespace parabola_focus_distance_l97_97635

theorem parabola_focus_distance (p : ℝ) (hp : p > 0) (A : ℝ × ℝ)
  (hA_on_parabola : A.2 ^ 2 = 2 * p * A.1)
  (hA_focus_dist : dist A (p / 2, 0) = 12)
  (hA_yaxis_dist : abs A.1 = 9) : p = 6 :=
sorry

end parabola_focus_distance_l97_97635


namespace sqrt_cos_sin_relation_l97_97961

variable {a b c θ : ℝ}

theorem sqrt_cos_sin_relation 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * (Real.cos θ) ^ 2 + b * (Real.sin θ) ^ 2 < c) :
  Real.sqrt a * (Real.cos θ) ^ 2 + Real.sqrt b * (Real.sin θ) ^ 2 < Real.sqrt c :=
sorry

end sqrt_cos_sin_relation_l97_97961


namespace odd_function_behavior_l97_97986

-- Define that f is odd
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Define f for x > 0
def f_pos (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → (f x = (Real.log x / Real.log 2) - 2 * x)

-- Prove that for x < 0, f(x) == -log₂(-x) - 2x
theorem odd_function_behavior (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_pos : f_pos f) :
  ∀ x, x < 0 → f x = -((Real.log (-x)) / (Real.log 2)) - 2 * x := 
by
  sorry -- proof goes here

end odd_function_behavior_l97_97986


namespace square_completion_l97_97074

theorem square_completion (a : ℝ) (h : a^2 + 2 * a - 2 = 0) : (a + 1)^2 = 3 := 
by 
  sorry

end square_completion_l97_97074


namespace unpainted_area_of_five_inch_board_l97_97514

def width1 : ℝ := 5
def width2 : ℝ := 6
def angle : ℝ := 45

theorem unpainted_area_of_five_inch_board : 
  ∃ (area : ℝ), area = 30 :=
by
  sorry

end unpainted_area_of_five_inch_board_l97_97514


namespace find_line_through_intersection_and_perpendicular_l97_97102

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def perpendicular (x y m : ℝ) : Prop := x + 3 * y + 4 = 0 ∧ 3 * x - y + m = 0

theorem find_line_through_intersection_and_perpendicular :
  ∃ m : ℝ, ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ perpendicular x y m → 3 * x - y + 2 = 0 :=
by
  sorry

end find_line_through_intersection_and_perpendicular_l97_97102


namespace pascal_row_12_sum_pascal_row_12_middle_l97_97346

open Nat

/-- Definition of the sum of all numbers in a given row of Pascal's Triangle -/
def pascal_sum (n : ℕ) : ℕ :=
  2^n

/-- Definition of the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Pascal Triangle Row 12 sum -/
theorem pascal_row_12_sum : pascal_sum 12 = 4096 :=
by
  sorry

/-- Pascal Triangle Row 12 middle number -/
theorem pascal_row_12_middle : binomial 12 6 = 924 :=
by
  sorry

end pascal_row_12_sum_pascal_row_12_middle_l97_97346


namespace standing_arrangements_l97_97065

theorem standing_arrangements : ∃ (arrangements : ℕ), arrangements = 2 :=
by
  -- Given that Jia, Yi, Bing, and Ding are four distinct people standing in a row
  -- We need to prove that there are exactly 2 different ways for them to stand such that Jia is not at the far left and Yi is not at the far right
  sorry

end standing_arrangements_l97_97065


namespace gcd_100_450_l97_97008

theorem gcd_100_450 : Int.gcd 100 450 = 50 := 
by sorry

end gcd_100_450_l97_97008


namespace part_1_part_2_l97_97800

variable {a b : ℝ}

theorem part_1 (ha : a > 0) (hb : b > 0) : a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

theorem part_2 (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a * b^2 + a^2 * b :=
sorry

end part_1_part_2_l97_97800


namespace jack_birth_year_l97_97572

theorem jack_birth_year 
  (first_amc8_year : ℕ) 
  (amc8_annual : ℕ → ℕ → ℕ) 
  (jack_age_ninth_amc8 : ℕ) 
  (ninth_amc8_year : amc8_annual first_amc8_year 9 = 1998) 
  (jack_age_in_ninth_amc8 : jack_age_ninth_amc8 = 15)
  : (1998 - jack_age_ninth_amc8 = 1983) := by
  sorry

end jack_birth_year_l97_97572


namespace sum_of_segments_AK_KB_eq_AB_l97_97893

-- Given conditions: length of segment AB is 9 cm
def length_AB : ℝ := 9

-- For any point K on segment AB, prove that AK + KB = AB
theorem sum_of_segments_AK_KB_eq_AB (K : ℝ) (h : 0 ≤ K ∧ K ≤ length_AB) : 
  K + (length_AB - K) = length_AB := by
  sorry

end sum_of_segments_AK_KB_eq_AB_l97_97893


namespace suitable_k_first_third_quadrants_l97_97162

theorem suitable_k_first_third_quadrants (k : ℝ) : 
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
by
  sorry

end suitable_k_first_third_quadrants_l97_97162


namespace original_decimal_number_l97_97436

theorem original_decimal_number (x : ℝ) (h₁ : 0 < x) (h₂ : 100 * x = 9 * (1 / x)) : x = 3 / 10 :=
by
  sorry

end original_decimal_number_l97_97436


namespace max_ab_at_extremum_l97_97583

noncomputable def f (a b x : ℝ) : ℝ := 4*x^3 - a*x^2 - 2*b*x + 2

theorem max_ab_at_extremum (a b : ℝ) (h0: a > 0) (h1 : b > 0) (h2 : ∃ x, f a b x = 4*x^3 - a*x^2 - 2*b*x + 2 ∧ x = 1 ∧ 12*x^2 - 2*a*x - 2*b = 0) :
  ab ≤ 9 := 
sorry  -- proof not required

end max_ab_at_extremum_l97_97583


namespace tens_digit_13_power_1987_l97_97629

theorem tens_digit_13_power_1987 : (13^1987)%100 / 10 = 1 :=
by
  sorry

end tens_digit_13_power_1987_l97_97629


namespace smallest_c_for_defined_expression_l97_97094

theorem smallest_c_for_defined_expression :
  ∃ (c : ℤ), (∀ x : ℝ, x^2 + (c : ℝ) * x + 15 ≠ 0) ∧
             (∀ k : ℤ, (∀ x : ℝ, x^2 + (k : ℝ) * x + 15 ≠ 0) → c ≤ k) ∧
             c = -7 :=
by 
  sorry

end smallest_c_for_defined_expression_l97_97094


namespace smallest_a_divisible_by_1984_l97_97178

theorem smallest_a_divisible_by_1984 :
  ∃ a : ℕ, (∀ n : ℕ, n % 2 = 1 → 1984 ∣ (47^n + a * 15^n)) ∧ a = 1055 := 
by 
  sorry

end smallest_a_divisible_by_1984_l97_97178


namespace non_equivalent_paintings_wheel_l97_97885

theorem non_equivalent_paintings_wheel :
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  (non_single_color_paintings / equivalent_rotation_count) + single_color_cases = 20 :=
by
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  have h1 := (non_single_color_paintings / equivalent_rotation_count) + single_color_cases
  sorry

end non_equivalent_paintings_wheel_l97_97885


namespace problem_statement_l97_97026

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

theorem problem_statement (f : ℝ → ℝ) :
  is_odd_function f →
  (∀ x : ℝ, f (x + 6) = f (x) + 3) →
  f 1 = 1 →
  f 2015 + f 2016 = 2015 :=
by
  sorry

end problem_statement_l97_97026


namespace smallest_integer_base_cube_l97_97445

theorem smallest_integer_base_cube (b : ℤ) (h1 : b > 5) (h2 : ∃ k : ℤ, 1 * b + 2 = k^3) : b = 6 :=
sorry

end smallest_integer_base_cube_l97_97445


namespace trigonometric_proof_l97_97614

theorem trigonometric_proof (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by sorry

end trigonometric_proof_l97_97614


namespace emilia_cartons_total_l97_97476

theorem emilia_cartons_total (strawberries blueberries supermarket : ℕ) (total_needed : ℕ)
  (h1 : strawberries = 2)
  (h2 : blueberries = 7)
  (h3 : supermarket = 33)
  (h4 : total_needed = strawberries + blueberries + supermarket) :
  total_needed = 42 :=
sorry

end emilia_cartons_total_l97_97476


namespace total_days_correct_l97_97690

-- Defining the years and the conditions given.
def year_1999 := 1999
def year_2000 := 2000
def year_2001 := 2001
def year_2002 := 2002

-- Defining the leap year and regular year days
def days_in_regular_year := 365
def days_in_leap_year := 366

-- Noncomputable version to skip the proof
noncomputable def total_days_from_1999_to_2002 : ℕ :=
  3 * days_in_regular_year + days_in_leap_year

-- The theorem stating the problem, which we need to prove
theorem total_days_correct : total_days_from_1999_to_2002 = 1461 := by
  sorry

end total_days_correct_l97_97690


namespace smallest_whole_number_greater_than_sum_is_12_l97_97490

-- Definitions of the mixed numbers as improper fractions
def a : ℚ := 5 / 3
def b : ℚ := 9 / 4
def c : ℚ := 27 / 8
def d : ℚ := 25 / 6

-- The target sum and the required proof statement
theorem smallest_whole_number_greater_than_sum_is_12 : 
  let sum := a + b + c + d
  let smallest_whole_number_greater_than_sum := Nat.ceil sum
  smallest_whole_number_greater_than_sum = 12 :=
by 
  sorry

end smallest_whole_number_greater_than_sum_is_12_l97_97490


namespace simplify_expression_l97_97491

theorem simplify_expression (x y z : ℝ) : - (x - (y - z)) = -x + y - z := by
  sorry

end simplify_expression_l97_97491


namespace find_a_l97_97347

theorem find_a (a b c d : ℕ) (h1 : a + b = d) (h2 : b + c = 6) (h3 : c + d = 7) : a = 1 :=
by
  sorry

end find_a_l97_97347


namespace sum_of_areas_of_circles_l97_97646

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l97_97646


namespace janet_total_owed_l97_97708

def warehouseHourlyWage : ℝ := 15
def managerHourlyWage : ℝ := 20
def numWarehouseWorkers : ℕ := 4
def numManagers : ℕ := 2
def workDaysPerMonth : ℕ := 25
def workHoursPerDay : ℕ := 8
def ficaTaxRate : ℝ := 0.10

theorem janet_total_owed : 
  let warehouseWorkerMonthlyWage := warehouseHourlyWage * workDaysPerMonth * workHoursPerDay
  let managerMonthlyWage := managerHourlyWage * workDaysPerMonth * workHoursPerDay
  let totalMonthlyWages := (warehouseWorkerMonthlyWage * numWarehouseWorkers) + (managerMonthlyWage * numManagers)
  let ficaTaxes := totalMonthlyWages * ficaTaxRate
  let totalAmountOwed := totalMonthlyWages + ficaTaxes
  totalAmountOwed = 22000 := by
  sorry

end janet_total_owed_l97_97708


namespace jack_walked_distance_l97_97985

theorem jack_walked_distance (time_in_hours : ℝ) (rate : ℝ) (expected_distance : ℝ) : 
  time_in_hours = 1 + 15 / 60 ∧ 
  rate = 6.4 →
  expected_distance = 8 → 
  rate * time_in_hours = expected_distance :=
by 
  intros h
  sorry

end jack_walked_distance_l97_97985


namespace combined_mean_correct_l97_97642

section MeanScore

variables (score_first_section mean_first_section : ℝ)
variables (score_second_section mean_second_section : ℝ)
variables (num_first_section num_second_section : ℝ)

axiom mean_first : mean_first_section = 92
axiom mean_second : mean_second_section = 78
axiom ratio_students : num_first_section / num_second_section = 5 / 7

noncomputable def combined_mean_score : ℝ := 
  let total_score := (mean_first_section * num_first_section + mean_second_section * num_second_section)
  let total_students := (num_first_section + num_second_section)
  total_score / total_students

theorem combined_mean_correct : combined_mean_score 92 78 (5 / 7 * num_second_section) num_second_section = 83.8 := by
  sorry

end MeanScore

end combined_mean_correct_l97_97642


namespace investment_return_l97_97682

theorem investment_return 
  (investment1 : ℝ) (investment2 : ℝ) 
  (return1 : ℝ) (combined_return_percent : ℝ) : 
  investment1 = 500 → 
  investment2 = 1500 → 
  return1 = 0.07 → 
  combined_return_percent = 0.085 → 
  (500 * 0.07 + 1500 * r = 2000 * 0.085) → 
  r = 0.09 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end investment_return_l97_97682


namespace solution_set_of_inequality_l97_97429

theorem solution_set_of_inequality (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ x2 → f x2 ≤ f x1) →
  (f 1 = 0) →
  {x : ℝ | f (x - 3) ≥ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
by
  intros h_even h_mono h_f1
  sorry

end solution_set_of_inequality_l97_97429


namespace joe_fish_times_sam_l97_97563

-- Define the number of fish Sam has
def sam_fish : ℕ := 7

-- Define the number of fish Harry has
def harry_fish : ℕ := 224

-- Define the number of times Joe has as many fish as Sam
def joe_times_sam (x : ℕ) : Prop :=
  4 * (sam_fish * x) = harry_fish

-- The theorem to prove Joe has 8 times as many fish as Sam
theorem joe_fish_times_sam : ∃ x, joe_times_sam x ∧ x = 8 :=
by
  sorry

end joe_fish_times_sam_l97_97563


namespace exists_maximum_value_of_f_l97_97147

-- Define the function f(x, y)
noncomputable def f (x y : ℝ) : ℝ := (3 * x * y + 1) * Real.exp (-(x^2 + y^2))

-- Maximum value proof statement
theorem exists_maximum_value_of_f :
  ∃ (x y : ℝ), f x y = (3 / 2) * Real.exp (-1 / 3) :=
sorry

end exists_maximum_value_of_f_l97_97147


namespace sum_of_numbers_l97_97735

theorem sum_of_numbers :
  1357 + 7531 + 3175 + 5713 = 17776 :=
by
  sorry

end sum_of_numbers_l97_97735


namespace greatest_int_less_than_neg_17_div_3_l97_97282

theorem greatest_int_less_than_neg_17_div_3 : 
  ∀ (x : ℚ), x = -17/3 → ⌊x⌋ = -6 :=
by
  sorry

end greatest_int_less_than_neg_17_div_3_l97_97282


namespace men_sent_to_other_project_l97_97733

-- Let the initial number of men be 50
def initial_men : ℕ := 50
-- Let the time to complete the work initially be 10 days
def initial_days : ℕ := 10
-- Calculate the total work in man-days
def total_work : ℕ := initial_men * initial_days

-- Let the total time taken after sending some men to another project be 30 days
def new_days : ℕ := 30
-- Let the number of men sent to another project be x
variable (x : ℕ)
-- Let the new number of men be (initial_men - x)
def new_men : ℕ := initial_men - x

theorem men_sent_to_other_project (x : ℕ):
total_work = new_men x * new_days -> x = 33 :=
by
  sorry

end men_sent_to_other_project_l97_97733


namespace find_phi_increasing_intervals_l97_97553

open Real

-- Defining the symmetry condition
noncomputable def symmetric_phi (x_sym : ℝ) (k : ℤ) (phi : ℝ): Prop :=
  2 * x_sym + phi = k * π + π / 2

-- Finding the value of phi given the conditions
theorem find_phi (x_sym : ℝ) (phi : ℝ) (k : ℤ) 
  (h_sym: symmetric_phi x_sym k phi) (h_phi_bound : -π < phi ∧ phi < 0)
  (h_xsym: x_sym = π / 8) :
  phi = -3 * π / 4 :=
by
  sorry

-- Defining the function and its increasing intervals
noncomputable def f (x : ℝ) (phi : ℝ) : ℝ := sin (2 * x + phi)

-- Finding the increasing intervals of f on the interval [0, π]
theorem increasing_intervals (phi : ℝ) 
  (h_phi: phi = -3 * π / 4) :
  ∀ x, (0 ≤ x ∧ x ≤ π) → 
    (π / 8 ≤ x ∧ x ≤ 5 * π / 8) :=
by
  sorry

end find_phi_increasing_intervals_l97_97553


namespace find_third_number_l97_97761

theorem find_third_number (x : ℕ) (h : 3 * 16 + 3 * 17 + 3 * x + 11 = 170) : x = 20 := by
  sorry

end find_third_number_l97_97761


namespace roots_of_quadratic_l97_97236

theorem roots_of_quadratic :
  ∃ (b c : ℝ), ( ∀ (x : ℝ), x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -2) :=
sorry

end roots_of_quadratic_l97_97236


namespace new_container_volume_l97_97150

-- Define the original volume of the container 
def original_volume : ℝ := 4

-- Define the scale factor of each dimension (quadrupled)
def scale_factor : ℝ := 4

-- Define the new volume, which is original volume * (scale factor ^ 3)
def new_volume (orig_vol : ℝ) (scale : ℝ) : ℝ := orig_vol * (scale ^ 3)

-- The theorem we want to prove
theorem new_container_volume : new_volume original_volume scale_factor = 256 :=
by
  sorry

end new_container_volume_l97_97150


namespace value_after_increase_l97_97124

-- Definition of original number and percentage increase
def original_number : ℝ := 600
def percentage_increase : ℝ := 0.10

-- Theorem stating that after a 10% increase, the value is 660
theorem value_after_increase : original_number * (1 + percentage_increase) = 660 := by
  sorry

end value_after_increase_l97_97124


namespace total_legs_of_camden_dogs_l97_97145

-- Defining the number of dogs Justin has
def justin_dogs : ℕ := 14

-- Defining the number of dogs Rico has
def rico_dogs : ℕ := justin_dogs + 10

-- Defining the number of dogs Camden has
def camden_dogs : ℕ := 3 * rico_dogs / 4

-- Defining the total number of legs Camden's dogs have
def camden_dogs_legs : ℕ := camden_dogs * 4

-- The proof statement
theorem total_legs_of_camden_dogs : camden_dogs_legs = 72 :=
by
  -- skip proof
  sorry

end total_legs_of_camden_dogs_l97_97145


namespace minor_axis_length_l97_97275

theorem minor_axis_length (h : ∀ x y : ℝ, x^2 / 4 + y^2 / 36 = 1) : 
  ∃ b : ℝ, b = 2 ∧ 2 * b = 4 :=
by
  sorry

end minor_axis_length_l97_97275


namespace andrew_age_l97_97067

variable (a g s : ℝ)

theorem andrew_age :
  g = 10 * a ∧ g - s = a + 45 ∧ s = 5 → a = 50 / 9 := by
  sorry

end andrew_age_l97_97067


namespace parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l97_97538

variables {Point Line Plane : Type}
variables (α β : Plane) (ℓ m : Line) (point_on_line_ℓ : Point) (point_on_line_m : Point)

-- Definitions of conditions
def line_perpendicular_to_plane (ℓ : Line) (α : Plane) : Prop := sorry
def line_contained_in_plane (m : Line) (β : Plane) : Prop := sorry
def planes_parallel (α β : Plane) : Prop := sorry
def line_perpendicular_to_line (ℓ m : Line) : Prop := sorry

axiom h1 : line_perpendicular_to_plane ℓ α
axiom h2 : line_contained_in_plane m β

-- Statement of the proof problem
theorem parallel_planes_sufficient_not_necessary_for_perpendicular_lines : 
  (planes_parallel α β → line_perpendicular_to_line ℓ m) ∧ 
  ¬ (line_perpendicular_to_line ℓ m → planes_parallel α β) :=
  sorry

end parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l97_97538


namespace length_of_woods_l97_97128

theorem length_of_woods (area width : ℝ) (h_area : area = 24) (h_width : width = 8) : (area / width) = 3 :=
by
  sorry

end length_of_woods_l97_97128


namespace find_z_given_x4_l97_97188

theorem find_z_given_x4 (k : ℝ) (z : ℝ) (x : ℝ) :
  (7 * 4 = k / 2^3) → (7 * z = k / x^3) → (x = 4) → (z = 0.5) :=
by
  intro h1 h2 h3
  sorry

end find_z_given_x4_l97_97188


namespace solution_set_a_eq_1_find_a_min_value_3_l97_97248

open Real

noncomputable def f (x a : ℝ) := 2 * abs (x + 1) + abs (x - a)

-- The statement for the first question
theorem solution_set_a_eq_1 (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -2 ∨ x ≥ (4 / 3) := 
by sorry

-- The statement for the second question
theorem find_a_min_value_3 (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 3) ∧ (∃ x : ℝ, f x a = 3) ↔ a = 2 ∨ a = -4 := 
by sorry

end solution_set_a_eq_1_find_a_min_value_3_l97_97248


namespace range_of_a_l97_97897

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a

def proposition_q (a : ℝ) : Prop := ∃ (x₀ : ℝ), x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) : proposition_p a ∧ proposition_q a ↔ (a = 1 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_l97_97897


namespace equal_angles_not_necessarily_vertical_l97_97233

-- Define what it means for angles to be vertical
def is_vertical_angle (a b : ℝ) : Prop :=
∃ l1 l2 : ℝ, a = 180 - b ∧ (l1 + l2 == 180 ∨ l1 == 0 ∨ l2 == 0)

-- Define what it means for angles to be equal
def are_equal_angles (a b : ℝ) : Prop := a = b

-- Proposition to be proved
theorem equal_angles_not_necessarily_vertical (a b : ℝ) (h : are_equal_angles a b) : ¬ is_vertical_angle a b :=
by
  sorry

end equal_angles_not_necessarily_vertical_l97_97233


namespace no_arithmetic_progression_exists_l97_97896

theorem no_arithmetic_progression_exists 
  (a : ℕ) (d : ℕ) (a_n : ℕ → ℕ) 
  (h_seq : ∀ n, a_n n = a + n * d) :
  ¬ ∃ (a_n : ℕ → ℕ), (∀ n, a_n (n+1) > a_n n ∧ 
  ∀ n, (a_n n) * (a_n (n+1)) * (a_n (n+2)) * (a_n (n+3)) * (a_n (n+4)) * 
        (a_n (n+5)) * (a_n (n+6)) * (a_n (n+7)) * (a_n (n+8)) * (a_n (n+9)) % 
        ((a_n n) + (a_n (n+1)) + (a_n (n+2)) + (a_n (n+3)) + (a_n (n+4)) + 
         (a_n (n+5)) + (a_n (n+6)) + (a_n (n+7)) + (a_n (n+8)) + (a_n (n+9)) ) = 0 ) := 
sorry

end no_arithmetic_progression_exists_l97_97896


namespace algebraic_expression_value_l97_97584

theorem algebraic_expression_value (a b : ℝ) (h : 4 * b = 3 + 4 * a) :
  a + (a - (a - (a - b) - b) - b) - b = -3 / 2 := by
  sorry

end algebraic_expression_value_l97_97584


namespace unique_positive_integer_appending_digits_eq_sum_l97_97950

-- Define the problem in terms of Lean types and properties
theorem unique_positive_integer_appending_digits_eq_sum :
  ∃! (A : ℕ), (A > 0) ∧ (∃ (B : ℕ), (0 ≤ B ∧ B < 1000) ∧ (1000 * A + B = (A * (A + 1)) / 2)) :=
sorry

end unique_positive_integer_appending_digits_eq_sum_l97_97950


namespace fraction_equality_l97_97080

theorem fraction_equality :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 59 / 61 :=
by
  sorry

end fraction_equality_l97_97080


namespace math_problem_l97_97269

variables {x y z a b c : ℝ}

theorem math_problem
  (h₁ : x / a + y / b + z / c = 4)
  (h₂ : a / x + b / y + c / z = 2) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end math_problem_l97_97269


namespace coffee_shop_distance_l97_97915

theorem coffee_shop_distance (resort_distance mall_distance : ℝ) 
  (coffee_dist : ℝ)
  (h_resort_distance : resort_distance = 400) 
  (h_mall_distance : mall_distance = 700)
  (h_equidistant : ∀ S, (S - resort_distance) ^ 2 + resort_distance ^ 2 = S ^ 2 ∧ 
  (mall_distance - S) ^ 2 + resort_distance ^ 2 = S ^ 2 → coffee_dist = S):
  coffee_dist = 464 := 
sorry

end coffee_shop_distance_l97_97915


namespace number_of_japanese_selectors_l97_97736

theorem number_of_japanese_selectors (F C J : ℕ) (h1 : J = 3 * C) (h2 : C = F + 15) (h3 : J + C + F = 165) : J = 108 :=
by
sorry

end number_of_japanese_selectors_l97_97736


namespace num_divisible_by_7_in_range_l97_97785

theorem num_divisible_by_7_in_range (n : ℤ) (h : 1 ≤ n ∧ n ≤ 2015)
    : (∃ k, 1 ≤ k ∧ k ≤ 335 ∧ 3 ^ (6 * k) + (6 * k) ^ 3 ≡ 0 [MOD 7]) :=
sorry

end num_divisible_by_7_in_range_l97_97785


namespace inverse_of_composed_function_l97_97073

theorem inverse_of_composed_function :
  let f (x : ℝ) := 4 * x + 5
  let g (x : ℝ) := 3 * x - 4
  let k (x : ℝ) := f (g x)
  ∀ y : ℝ, k ( (y + 11) / 12 ) = y :=
by
  sorry

end inverse_of_composed_function_l97_97073


namespace find_m_given_slope_condition_l97_97297

variable (m : ℝ)

theorem find_m_given_slope_condition
  (h : (m - 4) / (3 - 2) = 1) : m = 5 :=
sorry

end find_m_given_slope_condition_l97_97297


namespace michael_has_more_flying_robots_l97_97541

theorem michael_has_more_flying_robots (tom_robots michael_robots : ℕ) (h_tom : tom_robots = 3) (h_michael : michael_robots = 12) :
  michael_robots / tom_robots = 4 :=
by
  sorry

end michael_has_more_flying_robots_l97_97541


namespace jellybeans_problem_l97_97496

theorem jellybeans_problem (n : ℕ) (h : n ≥ 100) (h_mod : n % 13 = 11) : n = 102 :=
sorry

end jellybeans_problem_l97_97496


namespace volume_of_prism_l97_97967

-- Given dimensions a, b, and c, with the following conditions:
variables (a b c : ℝ)
axiom ab_eq_30 : a * b = 30
axiom ac_eq_40 : a * c = 40
axiom bc_eq_60 : b * c = 60

-- The volume of the prism is given by:
theorem volume_of_prism : a * b * c = 120 * Real.sqrt 5 :=
by
  sorry

end volume_of_prism_l97_97967


namespace vector_dot_product_zero_implies_orthogonal_l97_97872

theorem vector_dot_product_zero_implies_orthogonal
  (a b : ℝ → ℝ)
  (h0 : ∀ (x y : ℝ), a x * b y = 0) :
  ¬(a = 0 ∨ b = 0) := 
sorry

end vector_dot_product_zero_implies_orthogonal_l97_97872


namespace required_range_of_a_l97_97992

variable (a : ℝ) (f : ℝ → ℝ)
def function_increasing_on (f : ℝ → ℝ) (a : ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, DifferentiableAt ℝ f x ∧ (deriv f x) ≥ 0

theorem required_range_of_a (h : function_increasing_on (fun x => a * Real.log x + x) a (Set.Icc 2 3)) :
  a ≥ -2 :=
sorry

end required_range_of_a_l97_97992


namespace number_of_sides_of_regular_polygon_l97_97312

theorem number_of_sides_of_regular_polygon (h: ∀ (n: ℕ), (180 * (n - 2) / n) = 135) : ∃ n, n = 8 :=
by
  sorry

end number_of_sides_of_regular_polygon_l97_97312


namespace largest_sum_ABC_l97_97589

noncomputable def max_sum_ABC (A B C : ℕ) : ℕ :=
if A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 then
  A + B + C
else
  0

theorem largest_sum_ABC : ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 ∧ max_sum_ABC A B C = 52 :=
sorry

end largest_sum_ABC_l97_97589


namespace gcd_lcm_product_correct_l97_97672

noncomputable def gcd_lcm_product : ℕ :=
  let a := 90
  let b := 135
  gcd a b * lcm a b

theorem gcd_lcm_product_correct : gcd_lcm_product = 12150 :=
  by
  sorry

end gcd_lcm_product_correct_l97_97672


namespace wilson_fraction_l97_97952

theorem wilson_fraction (N : ℝ) (result : ℝ) (F : ℝ) (h1 : N = 8) (h2 : result = 16 / 3) (h3 : N - F * N = result) : F = 1 / 3 := 
by
  sorry

end wilson_fraction_l97_97952


namespace probability_queen_in_center_after_2004_moves_l97_97732

def initial_probability (n : ℕ) : ℚ :=
if n = 0 then 1
else if n = 1 then 0
else if n % 2 = 0 then (1 : ℚ) / 2^(n / 2)
else (1 - (1 : ℚ) / 2^((n - 1) / 2)) / 2

theorem probability_queen_in_center_after_2004_moves :
  initial_probability 2004 = 1 / 3 + 1 / (3 * 2^2003) :=
sorry

end probability_queen_in_center_after_2004_moves_l97_97732


namespace three_digit_addition_l97_97781

theorem three_digit_addition (a b : ℕ) (h₁ : 307 = 300 + a * 10 + 7) (h₂ : 416 + 10 * (a * 1) + 7 = 700 + b * 10 + 3) (h₃ : (7 + b + 3) % 3 = 0) : a + b = 2 :=
by
  -- mock proof, since solution steps are not considered
  sorry

end three_digit_addition_l97_97781


namespace find_g_5_l97_97060

-- Define the function g and the condition it satisfies
variable {g : ℝ → ℝ}
variable (hg : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x)

-- The proof goal
theorem find_g_5 : g 5 = 206 / 35 :=
by
  -- To be proven using the given condition hg
  sorry

end find_g_5_l97_97060


namespace first_month_sale_l97_97875

def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029
def sale6 : ℕ := 4937
def average_sale : ℕ := 5600

theorem first_month_sale :
  let total_sales := average_sale * 6
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  let sale1 := total_sales - known_sales
  sale1 = 5266 :=
by
  sorry

end first_month_sale_l97_97875


namespace roots_of_equation_l97_97245

theorem roots_of_equation:
  ∀ x : ℝ, (x - 2) * (x - 3) = x - 2 → x = 2 ∨ x = 4 := by
  sorry

end roots_of_equation_l97_97245


namespace calculate_expression_l97_97479

theorem calculate_expression :
  16 * (1/2) * 4 * (1/16) / 2 = 1 := 
by
  sorry

end calculate_expression_l97_97479


namespace sum_of_two_integers_l97_97163

theorem sum_of_two_integers (a b : ℕ) (h₁ : a * b + a + b = 135) (h₂ : Nat.gcd a b = 1) (h₃ : a < 30) (h₄ : b < 30) : a + b = 23 :=
sorry

end sum_of_two_integers_l97_97163


namespace percentage_off_sale_l97_97677

theorem percentage_off_sale (original_price sale_price : ℝ) (h₁ : original_price = 350) (h₂ : sale_price = 140) :
  ((original_price - sale_price) / original_price) * 100 = 60 :=
by
  sorry

end percentage_off_sale_l97_97677


namespace solve_quadratic_eq_solve_cubic_eq_l97_97268

-- Problem 1: 4x^2 - 9 = 0 implies x = ± 3/2
theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2 :=
by sorry

-- Problem 2: 64 * (x + 1)^3 = -125 implies x = -9/4
theorem solve_cubic_eq (x : ℝ) : 64 * (x + 1)^3 = -125 ↔ x = -9/4 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l97_97268


namespace percentage_girls_not_attended_college_l97_97924

-- Definitions based on given conditions
def total_boys : ℕ := 300
def total_girls : ℕ := 240
def percent_boys_not_attended_college : ℚ := 0.30
def percent_class_attended_college : ℚ := 0.70

-- The goal is to prove that the percentage of girls who did not attend college is 30%
theorem percentage_girls_not_attended_college 
  (total_boys : ℕ)
  (total_girls : ℕ)
  (percent_boys_not_attended_college : ℚ)
  (percent_class_attended_college : ℚ)
  (total_students := total_boys + total_girls)
  (boys_not_attended := percent_boys_not_attended_college * total_boys)
  (students_attended := percent_class_attended_college * total_students)
  (students_not_attended := total_students - students_attended)
  (girls_not_attended := students_not_attended - boys_not_attended) :
  (girls_not_attended / total_girls) * 100 = 30 := 
  sorry

end percentage_girls_not_attended_college_l97_97924


namespace cos_36_is_correct_l97_97473

noncomputable def cos_36_eq : Prop :=
  let b := Real.cos (Real.pi * 36 / 180)
  let a := Real.cos (Real.pi * 72 / 180)
  (a = 2 * b^2 - 1) ∧ (b = (1 + Real.sqrt 5) / 4)

theorem cos_36_is_correct : cos_36_eq :=
by sorry

end cos_36_is_correct_l97_97473


namespace Eugene_buys_two_pairs_of_shoes_l97_97418

theorem Eugene_buys_two_pairs_of_shoes :
  let tshirt_price : ℕ := 20
  let pants_price : ℕ := 80
  let shoes_price : ℕ := 150
  let discount_rate : ℕ := 10
  let discounted_price (price : ℕ) := price - (price * discount_rate / 100)
  let total_price (count1 count2 count3 : ℕ) (price1 price2 price3 : ℕ) :=
    (count1 * price1) + (count2 * price2) + (count3 * price3)
  let total_amount_paid : ℕ := 558
  let tshirts_bought : ℕ := 4
  let pants_bought : ℕ := 3
  let amount_left := total_amount_paid - discounted_price (tshirts_bought * tshirt_price + pants_bought * pants_price)
  let shoes_bought := amount_left / discounted_price shoes_price
  shoes_bought = 2 := 
sorry

end Eugene_buys_two_pairs_of_shoes_l97_97418


namespace trapezoid_base_ratio_l97_97955

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end trapezoid_base_ratio_l97_97955


namespace jessicas_score_l97_97673

theorem jessicas_score (average_20 : ℕ) (average_21 : ℕ) (n : ℕ) (jessica_score : ℕ) 
  (h1 : average_20 = 75)
  (h2 : average_21 = 76)
  (h3 : n = 20)
  (h4 : jessica_score = (average_21 * (n + 1)) - (average_20 * n)) :
  jessica_score = 96 :=
by 
  sorry

end jessicas_score_l97_97673


namespace arithmetic_sequence_sum_l97_97089

theorem arithmetic_sequence_sum :
  ∃ x y d : ℕ,
    d = 6
    ∧ x = 3 + d * (3 - 1)
    ∧ y = x + d
    ∧ y + d = 39
    ∧ x + y = 60 :=
by
  sorry

end arithmetic_sequence_sum_l97_97089


namespace certain_number_value_l97_97358

theorem certain_number_value :
  ∃ n : ℚ, 9 - (4 / 6) = 7 + (n / 6) ∧ n = 8 := by
sorry

end certain_number_value_l97_97358


namespace find_x_l97_97441

theorem find_x (x : ℚ) : (8 + 12 + 24) / 3 = (16 + x) / 2 → x = 40 / 3 :=
by
  intro h
  sorry

end find_x_l97_97441


namespace part1_l97_97648

theorem part1 (n : ℕ) (m : ℕ) (h_form : m = 2 ^ (n - 2) * 5 ^ n) (h : 6 * 10 ^ n + m = 25 * m) :
  ∃ k : ℕ, 6 * 10 ^ n + m = 625 * 10 ^ (n - 2) :=
by
  sorry

end part1_l97_97648


namespace votes_for_winner_is_744_l97_97991

variable (V : ℝ) -- Total number of votes cast

-- Conditions
axiom two_candidates : True
axiom winner_received_62_percent : True
axiom winner_won_by_288_votes : 0.62 * V - 0.38 * V = 288

-- Theorem to prove
theorem votes_for_winner_is_744 :
  0.62 * V = 744 :=
by
  sorry

end votes_for_winner_is_744_l97_97991


namespace minimum_a_l97_97575

theorem minimum_a (a : ℝ) : (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (a / x + 4 / y) ≥ 16) → a ≥ 4 :=
by
  intros h
  -- We would provide a detailed mathematical proof here, but we use sorry for now.
  sorry

end minimum_a_l97_97575


namespace half_of_4_pow_2022_is_2_pow_4043_l97_97526

theorem half_of_4_pow_2022_is_2_pow_4043 :
  (4 ^ 2022) / 2 = 2 ^ 4043 :=
by sorry

end half_of_4_pow_2022_is_2_pow_4043_l97_97526


namespace parallel_lines_value_of_m_l97_97295

theorem parallel_lines_value_of_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + m * y - 2 = 0 = (2 * x + (1 - m) * y + 2 = 0)) : 
  m = 1 / 3 :=
by {
  sorry
}

end parallel_lines_value_of_m_l97_97295


namespace car_z_mpg_decrease_l97_97609

theorem car_z_mpg_decrease :
  let mpg_45 := 51
  let mpg_60 := 408 / 10
  let decrease := mpg_45 - mpg_60
  let percentage_decrease := (decrease / mpg_45) * 100
  percentage_decrease = 20 := by
  sorry

end car_z_mpg_decrease_l97_97609


namespace calculate_f_g_f_l97_97321

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 5

theorem calculate_f_g_f : f (g (f 3)) = 630 := by
  sorry

end calculate_f_g_f_l97_97321


namespace black_queen_awake_at_10_l97_97034

-- Define the logical context
def king_awake_at_10 (king_asleep : Prop) : Prop :=
  king_asleep -> false

def king_asleep_at_10 (king_asleep : Prop) : Prop :=
  king_asleep

def queen_awake_at_10 (queen_asleep : Prop) : Prop :=
  queen_asleep -> false

-- Define the main theorem
theorem black_queen_awake_at_10 
  (king_asleep : Prop)
  (queen_asleep : Prop)
  (king_belief : king_asleep ↔ (king_asleep ∧ queen_asleep)) :
  queen_awake_at_10 queen_asleep :=
by
  -- Proof is omitted
  sorry

end black_queen_awake_at_10_l97_97034


namespace find_number_l97_97096

noncomputable def N := 953.87

theorem find_number (h : (0.47 * N - 0.36 * 1412) + 65 = 5) : N = 953.87 := sorry

end find_number_l97_97096


namespace inequality_ge_zero_l97_97213

theorem inequality_ge_zero (x y z : ℝ) : 
  4 * x * (x + y) * (x + z) * (x + y + z) + y^2 * z^2 ≥ 0 := 
sorry

end inequality_ge_zero_l97_97213


namespace ratio_female_democrats_l97_97759

theorem ratio_female_democrats (total_participants male_participants female_participants total_democrats female_democrats : ℕ)
  (h1 : total_participants = 750)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : total_democrats = total_participants / 3)
  (h4 : female_democrats = 125)
  (h5 : total_democrats = male_participants / 4 + female_democrats) :
  (female_democrats / female_participants : ℝ) = 1 / 2 :=
sorry

end ratio_female_democrats_l97_97759


namespace range_of_b_distance_when_b_eq_one_l97_97235

-- Definitions for conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def line (x y b : ℝ) : Prop := y = x + b
def intersect (x y b : ℝ) : Prop := ellipse x y ∧ line x y b

-- Prove the range of b for which there are two distinct intersection points
theorem range_of_b (b : ℝ) : (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ intersect x1 y1 b ∧ intersect x2 y2 b) ↔ (-Real.sqrt 3 < b ∧ b < Real.sqrt 3) :=
by sorry

-- Prove the distance between points A and B when b = 1
theorem distance_when_b_eq_one : 
  ∃ x1 y1 x2 y2, intersect x1 y1 1 ∧ intersect x2 y2 1 ∧ Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 2 / 3 :=
by sorry

end range_of_b_distance_when_b_eq_one_l97_97235


namespace train_or_plane_not_ship_possible_modes_l97_97018

-- Define the probabilities of different modes of transportation
def P_train : ℝ := 0.3
def P_ship : ℝ := 0.2
def P_car : ℝ := 0.1
def P_plane : ℝ := 0.4

-- 1. Proof that probability of train or plane is 0.7
theorem train_or_plane : P_train + P_plane = 0.7 :=
by sorry

-- 2. Proof that probability of not taking a ship is 0.8
theorem not_ship : 1 - P_ship = 0.8 :=
by sorry

-- 3. Proof that if probability is 0.5, the modes are either (ship, train) or (car, plane)
theorem possible_modes (P_value : ℝ) (h1 : P_value = 0.5) :
  (P_ship + P_train = P_value) ∨ (P_car + P_plane = P_value) :=
by sorry

end train_or_plane_not_ship_possible_modes_l97_97018


namespace roots_opposite_sign_eq_magnitude_l97_97742

theorem roots_opposite_sign_eq_magnitude (c d e n : ℝ) (h : ((n+2) * (x^2 + c*x + d)) = (n-2) * (2*x - e)) :
  n = (-4 - 2 * c) / (c - 2) :=
by
  sorry

end roots_opposite_sign_eq_magnitude_l97_97742


namespace James_has_43_Oreos_l97_97316

variable (J : ℕ)
variable (James_Oreos : ℕ)

-- Conditions
def condition1 : Prop := James_Oreos = 4 * J + 7
def condition2 : Prop := J + James_Oreos = 52

-- The statement to prove: James has 43 Oreos given the conditions
theorem James_has_43_Oreos (h1 : condition1 J James_Oreos) (h2 : condition2 J James_Oreos) : James_Oreos = 43 :=
by
  sorry

end James_has_43_Oreos_l97_97316


namespace greatest_x_is_53_l97_97762

-- Define the polynomial expression
def polynomial (x : ℤ) : ℤ := x^2 + 2 * x + 13

-- Define the condition for the expression to be an integer
def isIntegerWhenDivided (x : ℤ) : Prop := (polynomial x) % (x - 5) = 0

-- Define the theorem to prove the greatest integer value of x
theorem greatest_x_is_53 : ∃ x : ℤ, isIntegerWhenDivided x ∧ (∀ y : ℤ, isIntegerWhenDivided y → y ≤ x) ∧ x = 53 :=
by
  sorry

end greatest_x_is_53_l97_97762


namespace quadratic_roots_relation_l97_97953

theorem quadratic_roots_relation (m p q : ℝ) (h_m_ne_zero : m ≠ 0) (h_p_ne_zero : p ≠ 0) (h_q_ne_zero : q ≠ 0) :
  (∀ r1 r2 : ℝ, (r1 + r2 = -q ∧ r1 * r2 = m) → (3 * r1 + 3 * r2 = -m ∧ (3 * r1) * (3 * r2) = p)) →
  p / q = 27 :=
by
  intros h
  sorry

end quadratic_roots_relation_l97_97953


namespace max_cos2_sinx_l97_97488

noncomputable def cos2_sinx (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_cos2_sinx : ∃ x : ℝ, cos2_sinx x = 5 / 4 := 
by
  existsi (Real.arcsin (-1 / 2))
  rw [cos2_sinx]
  -- We need further steps to complete the proof
  sorry

end max_cos2_sinx_l97_97488


namespace optimal_rental_decision_optimal_purchase_decision_l97_97783

-- Definitions of conditions
def monthly_fee_first : ℕ := 50000
def monthly_fee_second : ℕ := 10000
def probability_seizure : ℚ := 0.5
def moving_cost : ℕ := 70000
def months_first_year : ℕ := 12
def months_seizure : ℕ := 4
def months_after_seizure : ℕ := months_first_year - months_seizure
def purchase_cost : ℕ := 2000000
def installment_period : ℕ := 36

-- Proving initial rental decision
theorem optimal_rental_decision :
  let annual_cost_first := monthly_fee_first * months_first_year
  let annual_cost_second := (monthly_fee_second * months_seizure) + (monthly_fee_first * months_after_seizure) + moving_cost
  annual_cost_second < annual_cost_first := 
by
  sorry

-- Proving purchasing decision
theorem optimal_purchase_decision :
  let total_rent_cost_after_seizure := (monthly_fee_second * months_seizure) + moving_cost + (monthly_fee_first * (4 * months_first_year - months_seizure))
  let total_purchase_cost := purchase_cost
  total_purchase_cost < total_rent_cost_after_seizure :=
by
  sorry

end optimal_rental_decision_optimal_purchase_decision_l97_97783


namespace matrix_power_2023_correct_l97_97059

noncomputable def matrix_power_2023 : Matrix (Fin 2) (Fin 2) ℤ :=
  let A := !![1, 0; 2, 1]  -- Define the matrix
  A^2023

theorem matrix_power_2023_correct :
  matrix_power_2023 = !![1, 0; 4046, 1] := by
  sorry

end matrix_power_2023_correct_l97_97059


namespace road_trip_ratio_l97_97438

-- Problem Definitions
variable (x d3 total grand_total : ℕ)
variable (hx1 : total = x + 2 * x + d3 + 2 * (x + 2 * x + d3))
variable (hx2 : d3 = 40)
variable (hx3 : total = 560)
variable (hx4 : grand_total = d3 / x)

-- Proof Statement
theorem road_trip_ratio (hx1 : total = x + 2 * x + d3 + 2 * (x + 2 * x + d3)) 
  (hx2 : d3 = 40) (hx3 : total = 560) : grand_total = 9 / 11 := by
  sorry

end road_trip_ratio_l97_97438


namespace impossible_to_fill_grid_l97_97540

def is_impossible : Prop :=
  ∀ (grid : Fin 3 → Fin 3 → ℕ), 
  (∀ i j, grid i j ≠ grid i (j + 1) ∧ grid i j ≠ grid (i + 1) j) →
  (∀ i, (grid i 0) * (grid i 1) * (grid i 2) = 2005) →
  (∀ j, (grid 0 j) * (grid 1 j) * (grid 2 j) = 2005) →
  (grid 0 0) * (grid 1 1) * (grid 2 2) = 2005 →
  (grid 0 2) * (grid 1 1) * (grid 2 0) = 2005 →
  False

theorem impossible_to_fill_grid : is_impossible :=
  sorry

end impossible_to_fill_grid_l97_97540


namespace living_room_area_l97_97044

theorem living_room_area (L W : ℝ) (percent_covered : ℝ) (expected_area : ℝ) 
  (hL : L = 6.5) (hW : W = 12) (hpercent : percent_covered = 0.85) 
  (hexpected_area : expected_area = 91.76) : 
  (L * W / percent_covered = expected_area) :=
by
  sorry  -- The proof is omitted.

end living_room_area_l97_97044


namespace era_slices_burger_l97_97063

theorem era_slices_burger (slices_per_burger : ℕ) (h : 5 * slices_per_burger = 10) : slices_per_burger = 2 :=
by 
  sorry

end era_slices_burger_l97_97063


namespace length_OD1_l97_97372

-- Define the hypothesis of the problem
noncomputable def sphere_center : Point := sorry -- center O of the sphere
noncomputable def radius_sphere : ℝ := 10 -- radius of the sphere

-- Define face intersection properties
noncomputable def face_AA1D1D_radius : ℝ := 1
noncomputable def face_A1B1C1D1_radius : ℝ := 1
noncomputable def face_CDD1C1_radius : ℝ := 3

-- Define the coordinates of D1 (or in abstract form, we'll assume it is a known point)
noncomputable def segment_OD1 : ℝ := sorry -- Length of OD1 segment to be calculated

-- The main theorem to prove
theorem length_OD1 : 
  -- Given conditions
  (face_AA1D1D_radius = 1) ∧ 
  (face_A1B1C1D1_radius = 1) ∧ 
  (face_CDD1C1_radius = 3) ∧ 
  (radius_sphere = 10) →
  -- Prove the length of segment OD1 is 17
  segment_OD1 = 17 :=
by
  sorry

end length_OD1_l97_97372


namespace polynomial_condition_satisfied_l97_97695

-- Definitions as per conditions:
def p (x : ℝ) : ℝ := x^2 + 1

-- Conditions:
axiom cond1 : p 3 = 10
axiom cond2 : ∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2

-- Theorem to prove:
theorem polynomial_condition_satisfied : (p 3 = 10) ∧ (∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2) :=
by
  apply And.intro cond1
  apply cond2

end polynomial_condition_satisfied_l97_97695


namespace problem_solution_l97_97305

def f (x y : ℝ) : ℝ :=
  (x - y) * x * y * (x + y) * (2 * x^2 - 5 * x * y + 2 * y^2)

theorem problem_solution :
  (∀ x y : ℝ, f x y + f y x = 0) ∧
  (∀ x y : ℝ, f x (x + y) + f y (x + y) = 0) :=
by
  sorry

end problem_solution_l97_97305


namespace artifacts_per_wing_l97_97493

theorem artifacts_per_wing
  (total_wings : ℕ)
  (num_paintings : ℕ)
  (num_artifacts : ℕ)
  (painting_wings : ℕ)
  (large_paintings_wings : ℕ)
  (small_paintings_wings : ℕ)
  (small_paintings_per_wing : ℕ)
  (artifact_wings : ℕ)
  (wings_division : total_wings = painting_wings + artifact_wings)
  (paintings_division : painting_wings = large_paintings_wings + small_paintings_wings)
  (num_large_paintings : large_paintings_wings = 2)
  (num_small_paintings : small_paintings_wings * small_paintings_per_wing = num_paintings - large_paintings_wings)
  (num_artifact_calc : num_artifacts = 8 * num_paintings)
  (artifact_wings_div : artifact_wings = total_wings - painting_wings)
  (artifact_calc : num_artifacts / artifact_wings = 66) :
  num_artifacts / artifact_wings = 66 := 
by
  sorry

end artifacts_per_wing_l97_97493


namespace max_x_plus_2y_l97_97005

theorem max_x_plus_2y {x y : ℝ} (h : x^2 - x * y + y^2 = 1) :
  x + 2 * y ≤ (2 * Real.sqrt 21) / 3 :=
sorry

end max_x_plus_2y_l97_97005


namespace farmer_john_pairs_l97_97768

noncomputable def farmer_john_animals_pairing :
    Nat := 
  let cows := 5
  let pigs := 4
  let horses := 7
  let num_ways_cow_pig_pair := cows * pigs
  let num_ways_horses_remaining := Nat.factorial horses
  num_ways_cow_pig_pair * num_ways_horses_remaining

theorem farmer_john_pairs : farmer_john_animals_pairing = 100800 := 
by
  sorry

end farmer_john_pairs_l97_97768


namespace simplify_and_evaluate_l97_97350

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l97_97350


namespace problem_statement_l97_97434

theorem problem_statement (n : ℕ) : (-1 : ℤ) ^ n * (-1) ^ (2 * n + 1) * (-1) ^ (n + 1) = 1 := 
by
  sorry

end problem_statement_l97_97434


namespace new_container_volume_l97_97322

def volume_of_cube (s : ℝ) : ℝ := s^3

theorem new_container_volume (s : ℝ) (h : volume_of_cube s = 4) : 
  volume_of_cube (2 * s) * volume_of_cube (3 * s) * volume_of_cube (4 * s) = 96 :=
by
  sorry

end new_container_volume_l97_97322


namespace expr_value_l97_97234

variable (a : ℝ)
variable (h : a^2 - 3 * a - 1011 = 0)

theorem expr_value : 2 * a^2 - 6 * a + 1 = 2023 :=
by
  -- insert proof here
  sorry

end expr_value_l97_97234


namespace garden_dimensions_l97_97829

theorem garden_dimensions
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l * w = 600) : 
  w = 10 * Real.sqrt 3 ∧ l = 20 * Real.sqrt 3 :=
by
  sorry

end garden_dimensions_l97_97829


namespace sequence_sum_periodic_l97_97596

theorem sequence_sum_periodic (a : ℕ → ℕ) (a1 a8 : ℕ) :
  a 1 = 11 →
  a 8 = 12 →
  (∀ i, 1 ≤ i → i ≤ 6 → a i + a (i + 1) + a (i + 2) = 50) →
  (a 1 = 11 ∧ a 2 = 12 ∧ a 3 = 27 ∧ a 4 = 11 ∧ a 5 = 12 ∧ a 6 = 27 ∧ a 7 = 11 ∧ a 8 = 12) :=
by
  intros h1 h8 hsum
  sorry

end sequence_sum_periodic_l97_97596


namespace union_sets_l97_97668

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_sets_l97_97668


namespace width_of_rectangle_l97_97090

-- Define the side length of the square and the length of the rectangle.
def side_length_square : ℝ := 12
def length_rectangle : ℝ := 18

-- Calculate the perimeter of the square.
def perimeter_square : ℝ := 4 * side_length_square

-- This definition represents the perimeter of the rectangle made from the same wire.
def perimeter_rectangle : ℝ := perimeter_square

-- Show that the width of the rectangle is 6 cm.
theorem width_of_rectangle : ∃ W : ℝ, 2 * (length_rectangle + W) = perimeter_rectangle ∧ W = 6 :=
by
  use 6
  simp [length_rectangle, perimeter_rectangle, side_length_square]
  norm_num
  sorry

end width_of_rectangle_l97_97090


namespace max_5x_plus_3y_l97_97592

theorem max_5x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 8 * y + 10) : 5 * x + 3 * y ≤ 105 :=
sorry

end max_5x_plus_3y_l97_97592


namespace residue_5_pow_1234_mod_13_l97_97238

theorem residue_5_pow_1234_mod_13 : ∃ k : ℤ, 5^1234 = 13 * k + 12 :=
by
  sorry

end residue_5_pow_1234_mod_13_l97_97238


namespace cos_diff_to_product_l97_97361

open Real

theorem cos_diff_to_product (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := 
  sorry

end cos_diff_to_product_l97_97361


namespace problem_solution_l97_97152

noncomputable def omega : ℂ := sorry -- Choose a suitable representative for ω

variables (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
          (hω : ω^3 = 1 ∧ ω ≠ 1)
          (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω)

theorem problem_solution (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
  (hω : ω^3 = 1 ∧ ω ≠ 1)
  (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 3 :=
sorry

end problem_solution_l97_97152


namespace gain_percent_correct_l97_97857

noncomputable def cycleCP : ℝ := 900
noncomputable def cycleSP : ℝ := 1180
noncomputable def gainPercent : ℝ := (cycleSP - cycleCP) / cycleCP * 100

theorem gain_percent_correct :
  gainPercent = 31.11 := by
  sorry

end gain_percent_correct_l97_97857


namespace Pascal_remaining_distance_l97_97207

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l97_97207


namespace range_of_m_l97_97844

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, abs (x - 3) + abs (x - m) < 5) : -2 < m ∧ m < 8 :=
  sorry

end range_of_m_l97_97844


namespace prove_intersection_points_l97_97001

noncomputable def sqrt5 := Real.sqrt 5

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 5 / 2
def curve2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1
def curve3 (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1
def curve4 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def line (x y : ℝ) : Prop := x + y = sqrt5

theorem prove_intersection_points :
  (∃! (x y : ℝ), curve1 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve3 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve4 x y ∧ line x y) :=
by
  sorry

end prove_intersection_points_l97_97001


namespace arithmetic_sequence_common_difference_and_m_l97_97435

theorem arithmetic_sequence_common_difference_and_m (S : ℕ → ℤ) (a : ℕ → ℤ) (m d : ℕ) 
(h1 : S (m-1) = -2) (h2 : S m = 0) (h3 : S (m+1) = 3) :
  d = 1 ∧ m = 5 :=
by sorry

end arithmetic_sequence_common_difference_and_m_l97_97435


namespace dogs_in_garden_l97_97774

theorem dogs_in_garden (D : ℕ) (ducks : ℕ) (total_feet : ℕ) (dogs_feet : ℕ) (ducks_feet : ℕ) 
  (h1 : ducks = 2) 
  (h2 : total_feet = 28)
  (h3 : dogs_feet = 4)
  (h4 : ducks_feet = 2) 
  (h_eq : dogs_feet * D + ducks_feet * ducks = total_feet) : 
  D = 6 := by
  sorry

end dogs_in_garden_l97_97774


namespace average_class_size_l97_97525

theorem average_class_size 
  (num_3_year_olds : ℕ) 
  (num_4_year_olds : ℕ) 
  (num_5_year_olds : ℕ) 
  (num_6_year_olds : ℕ) 
  (class_size_3_and_4 : num_3_year_olds = 13 ∧ num_4_year_olds = 20) 
  (class_size_5_and_6 : num_5_year_olds = 15 ∧ num_6_year_olds = 22) :
  (num_3_year_olds + num_4_year_olds + num_5_year_olds + num_6_year_olds) / 2 = 35 :=
by
  sorry

end average_class_size_l97_97525


namespace equilateral_right_triangle_impossible_l97_97921
-- Import necessary library

-- Define the conditions and the problem statement
theorem equilateral_right_triangle_impossible :
  ¬(∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A = B ∧ B = C ∧ (A^2 + B^2 = C^2) ∧ (A + B + C = 180)) := sorry

end equilateral_right_triangle_impossible_l97_97921


namespace breadth_of_garden_l97_97151

theorem breadth_of_garden (P L B : ℝ) (hP : P = 1800) (hL : L = 500) : B = 400 :=
by
  sorry

end breadth_of_garden_l97_97151


namespace hyperbola_sufficient_not_necessary_condition_l97_97851

-- Define the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 16 = 1

-- Define the asymptotic line equations of the hyperbola
def asymptotes_eq (x y : ℝ) : Prop :=
  y = 2 * x ∨ y = -2 * x

-- Prove that the equation of the hyperbola is a sufficient but not necessary condition for the asymptotic lines
theorem hyperbola_sufficient_not_necessary_condition :
  (∀ x y : ℝ, hyperbola_eq x y → asymptotes_eq x y) ∧ ¬ (∀ x y : ℝ, asymptotes_eq x y → hyperbola_eq x y) :=
by
  sorry

end hyperbola_sufficient_not_necessary_condition_l97_97851


namespace at_least_one_equation_has_real_roots_l97_97751

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0)

theorem at_least_one_equation_has_real_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  has_two_distinct_real_roots a b c :=
by
  sorry

end at_least_one_equation_has_real_roots_l97_97751


namespace simplify_fraction_l97_97084

theorem simplify_fraction (x : ℚ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by sorry

end simplify_fraction_l97_97084


namespace overlapped_squares_area_l97_97944

/-- 
Theorem: The area of the figure formed by overlapping four identical squares, 
each with an area of \(3 \, \text{cm}^2\), and with an overlapping region 
that double-counts 6 small squares is \(10.875 \, \text{cm}^2\).
-/
theorem overlapped_squares_area (area_of_square : ℝ) (num_squares : ℕ) (overlap_small_squares : ℕ) :
  area_of_square = 3 → 
  num_squares = 4 → 
  overlap_small_squares = 6 →
  ∃ total_area : ℝ, total_area = (num_squares * area_of_square) - (overlap_small_squares * (area_of_square / 16)) ∧
                         total_area = 10.875 :=
by
  sorry

end overlapped_squares_area_l97_97944


namespace trig_problem_l97_97172

variables (θ : ℝ)

theorem trig_problem (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.tan θ + 1 / Real.tan θ = 4 :=
sorry

end trig_problem_l97_97172


namespace value_of_expression_l97_97700

theorem value_of_expression (x y z : ℝ) (h : x * y * z = 1) :
  1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1 :=
sorry

end value_of_expression_l97_97700


namespace abs_sum_inequality_l97_97533

theorem abs_sum_inequality (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := 
sorry

end abs_sum_inequality_l97_97533


namespace find_original_number_l97_97470

def original_four_digit_number (N : ℕ) : Prop :=
  N >= 1000 ∧ N < 10000 ∧ (70000 + N) - (10 * N + 7) = 53208

theorem find_original_number (N : ℕ) (h : original_four_digit_number N) : N = 1865 :=
by
  sorry

end find_original_number_l97_97470


namespace tank_capacity_l97_97981

theorem tank_capacity :
  (∃ (C : ℕ), ∀ (leak_rate inlet_rate net_rate : ℕ),
    leak_rate = C / 6 ∧
    inlet_rate = 6 * 60 ∧
    net_rate = C / 12 ∧
    inlet_rate - leak_rate = net_rate → C = 1440) :=
sorry

end tank_capacity_l97_97981


namespace values_of_2n_plus_m_l97_97262

theorem values_of_2n_plus_m (n m : ℤ) (h1 : 3 * n - m ≤ 4) (h2 : n + m ≥ 27) (h3 : 3 * m - 2 * n ≤ 45) 
  (h4 : n = 8) (h5 : m = 20) : 2 * n + m = 36 := by
  sorry

end values_of_2n_plus_m_l97_97262


namespace sufficient_not_necessary_l97_97129

theorem sufficient_not_necessary (x : ℝ) : (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 1) ∧ ¬((x ≠ 1) → (x^2 - 3 * x + 2 ≠ 0)) :=
by
  sorry

end sufficient_not_necessary_l97_97129


namespace intersection_M_N_l97_97948

open Set

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} :=
by
  sorry

end intersection_M_N_l97_97948


namespace squares_not_all_congruent_l97_97946

/-- Proof that the statement "all squares are congruent to each other" is false. -/
theorem squares_not_all_congruent : ¬(∀ (a b : ℝ), a = b ↔ a = b) :=
by 
  sorry

end squares_not_all_congruent_l97_97946


namespace complex_quadrant_l97_97993

theorem complex_quadrant (i : ℂ) (h_imag : i = Complex.I) :
  let z := (1 + i)⁻¹
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l97_97993


namespace min_height_bounces_l97_97654

noncomputable def geometric_sequence (a r: ℝ) (n: ℕ) : ℝ := 
  a * r^n

theorem min_height_bounces (k : ℕ) : 
  ∀ k, 20 * (2 / 3 : ℝ) ^ k < 3 → k ≥ 7 := 
by
  sorry

end min_height_bounces_l97_97654


namespace custom_op_two_neg_four_l97_97037

-- Define the binary operation *
def custom_op (x y : ℚ) : ℚ := (x * y) / (x + y)

-- Proposition stating 2 * (-4) = 4 using the custom operation
theorem custom_op_two_neg_four : custom_op 2 (-4) = 4 :=
by
  sorry

end custom_op_two_neg_four_l97_97037


namespace factorization_correct_l97_97849

theorem factorization_correct (x : ℝ) : 2 * x^2 - 6 * x - 8 = 2 * (x - 4) * (x + 1) :=
by
  sorry

end factorization_correct_l97_97849


namespace solve_equation1_solve_equation2_l97_97790

-- Proof for equation (1)
theorem solve_equation1 : ∃ x : ℝ, 2 * (2 * x + 1) - (3 * x - 4) = 2 := by
  exists -4
  sorry

-- Proof for equation (2)
theorem solve_equation2 : ∃ y : ℝ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  exists -1
  sorry

end solve_equation1_solve_equation2_l97_97790


namespace taxi_fare_miles_l97_97042

theorem taxi_fare_miles (total_spent : ℝ) (tip : ℝ) (base_fare : ℝ) (additional_fare_rate : ℝ) (base_mile : ℝ) (additional_mile_unit : ℝ) (x : ℝ) :
  (total_spent = 15) →
  (tip = 3) →
  (base_fare = 3) →
  (additional_fare_rate = 0.25) →
  (base_mile = 0.5) →
  (additional_mile_unit = 0.1) →
  (x = base_mile + (total_spent - tip - base_fare) / (additional_fare_rate / additional_mile_unit)) →
  x = 4.1 :=
by
  intros
  sorry

end taxi_fare_miles_l97_97042


namespace polynomial_root_arithmetic_sequence_l97_97461

theorem polynomial_root_arithmetic_sequence :
  (∃ (a d : ℝ), 
    (64 * (a - d)^3 + 144 * (a - d)^2 + 92 * (a - d) + 15 = 0) ∧
    (64 * a^3 + 144 * a^2 + 92 * a + 15 = 0) ∧
    (64 * (a + d)^3 + 144 * (a + d)^2 + 92 * (a + d) + 15 = 0) ∧
    (2 * d = 1)) := sorry

end polynomial_root_arithmetic_sequence_l97_97461


namespace johns_initial_money_l97_97397

theorem johns_initial_money (X : ℝ) 
  (h₁ : (1 / 2) * X + (1 / 3) * X + (1 / 10) * X + 10 = X) : X = 150 :=
sorry

end johns_initial_money_l97_97397


namespace polynomial_div_remainder_l97_97222

theorem polynomial_div_remainder (x : ℝ) : 
  (x^4 % (x^2 + 7*x + 2)) = -315*x - 94 := 
by
  sorry

end polynomial_div_remainder_l97_97222


namespace number_of_solutions_l97_97025

theorem number_of_solutions (x y : ℕ) : (3 * x + 2 * y = 1001) → ∃! (n : ℕ), n = 167 := by
  sorry

end number_of_solutions_l97_97025


namespace factorize_expression_l97_97169

theorem factorize_expression (x : ℝ) : x^2 - 2023 * x = x * (x - 2023) := 
by 
  sorry

end factorize_expression_l97_97169


namespace factor_polynomial_l97_97864

def A (x : ℝ) : ℝ := x^2 + 5 * x + 3
def B (x : ℝ) : ℝ := x^2 + 9 * x + 20
def C (x : ℝ) : ℝ := x^2 + 7 * x - 8

theorem factor_polynomial (x : ℝ) :
  (A x) * (B x) + (C x) = (x^2 + 7 * x + 8) * (x^2 + 7 * x + 14) :=
by
  sorry

end factor_polynomial_l97_97864


namespace tomatoes_picked_l97_97547

theorem tomatoes_picked (original_tomatoes left_tomatoes picked_tomatoes : ℕ)
  (h1 : original_tomatoes = 97)
  (h2 : left_tomatoes = 14)
  (h3 : picked_tomatoes = original_tomatoes - left_tomatoes) :
  picked_tomatoes = 83 :=
by sorry

end tomatoes_picked_l97_97547


namespace smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l97_97755

theorem smallest_four_digit_palindrome_div_by_3_with_odd_first_digit :
  ∃ (n : ℕ), (∃ A B : ℕ, n = 1001 * A + 110 * B ∧ 1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ A % 2 = 1) ∧ 3 ∣ n ∧ n = 1221 :=
by
  sorry

end smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l97_97755


namespace stella_profit_l97_97770

-- Definitions based on the conditions
def number_of_dolls := 6
def price_per_doll := 8
def number_of_clocks := 4
def price_per_clock := 25
def number_of_glasses := 8
def price_per_glass := 6
def number_of_vases := 3
def price_per_vase := 12
def number_of_postcards := 10
def price_per_postcard := 3
def cost_of_merchandise := 250

-- Calculations based on given problem and solution
def revenue_from_dolls := number_of_dolls * price_per_doll
def revenue_from_clocks := number_of_clocks * price_per_clock
def revenue_from_glasses := number_of_glasses * price_per_glass
def revenue_from_vases := number_of_vases * price_per_vase
def revenue_from_postcards := number_of_postcards * price_per_postcard
def total_revenue := revenue_from_dolls + revenue_from_clocks + revenue_from_glasses + revenue_from_vases + revenue_from_postcards
def profit := total_revenue - cost_of_merchandise

-- Main theorem statement
theorem stella_profit : profit = 12 := by
  sorry

end stella_profit_l97_97770


namespace factor_difference_of_squares_l97_97499

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y ^ 2 = (5 - 4 * y) * (5 + 4 * y) :=
by
  sorry

end factor_difference_of_squares_l97_97499


namespace reciprocal_neg_5_l97_97381

theorem reciprocal_neg_5 : ∃ x : ℚ, -5 * x = 1 ∧ x = -1/5 :=
by
  sorry

end reciprocal_neg_5_l97_97381


namespace find_b_and_c_l97_97325

variable (U : Set ℝ) -- Define the universal set U
variable (A : Set ℝ) -- Define the set A
variables (b c : ℝ) -- Variables for coefficients

-- Conditions that U = {2, 3, 5} and A = { x | x^2 + bx + c = 0 }
def cond_universal_set := U = {2, 3, 5}
def cond_set_A := A = { x | x^2 + b * x + c = 0 }

-- Condition for the complement of A w.r.t U being {2}
def cond_complement := (U \ A) = {2}

-- The statement to be proved
theorem find_b_and_c : 
  cond_universal_set U →
  cond_set_A A b c →
  cond_complement U A →
  b = -8 ∧ c = 15 :=
by
  intros
  sorry

end find_b_and_c_l97_97325


namespace initial_red_marbles_l97_97834

variable (r g : ℝ)

def red_green_ratio_initial (r g : ℝ) : Prop := r / g = 5 / 3
def red_green_ratio_new (r g : ℝ) : Prop := (r + 15) / (g - 9) = 3 / 1

theorem initial_red_marbles (r g : ℝ) (h₁ : red_green_ratio_initial r g) (h₂ : red_green_ratio_new r g) : r = 52.5 := sorry

end initial_red_marbles_l97_97834


namespace solution_part_1_solution_part_2_l97_97665

def cost_price_of_badges (x y : ℕ) : Prop :=
  (x - y = 4) ∧ (6 * x = 10 * y)

theorem solution_part_1 (x y : ℕ) :
  cost_price_of_badges x y → x = 10 ∧ y = 6 :=
by
  sorry

def maximizing_profit (m : ℕ) (w : ℕ) : Prop :=
  (10 * m + 6 * (400 - m) ≤ 2800) ∧ (w = m + 800)

theorem solution_part_2 (m : ℕ) :
  maximizing_profit m 900 → m = 100 :=
by
  sorry


end solution_part_1_solution_part_2_l97_97665


namespace tim_same_age_tina_l97_97945

-- Define the ages of Tim and Tina
variables (x y : ℤ)

-- Given conditions
def condition_tim := x + 2 = 2 * (x - 2)
def condition_tina := y + 3 = 3 * (y - 3)

-- The goal is to prove that Tim is the same age as Tina
theorem tim_same_age_tina (htim : condition_tim x) (htina : condition_tina y) : x = y :=
by 
  sorry

end tim_same_age_tina_l97_97945


namespace second_planner_cheaper_l97_97862

theorem second_planner_cheaper (x : ℕ) :
  (∀ x, 250 + 15 * x < 150 + 18 * x → x ≥ 34) :=
by
  intros x h
  sorry

end second_planner_cheaper_l97_97862


namespace base4_arithmetic_l97_97923

theorem base4_arithmetic : 
  ∀ (a b c : ℕ),
  a = 2 * 4^2 + 3 * 4^1 + 1 * 4^0 →
  b = 2 * 4^1 + 4 * 4^0 →
  c = 3 * 4^0 →
  (a * b) / c = 2 * 4^3 + 3 * 4^2 + 1 * 4^1 + 0 * 4^0 :=
by
  intros a b c ha hb hc
  sorry

end base4_arithmetic_l97_97923


namespace no_solution_for_n_eq_neg1_l97_97340

theorem no_solution_for_n_eq_neg1 (x y z : ℝ) : ¬ (∃ x y z, (-1) * x^2 + y = 2 ∧ (-1) * y^2 + z = 2 ∧ (-1) * z^2 + x = 2) :=
by
  sorry

end no_solution_for_n_eq_neg1_l97_97340


namespace opposite_of_neg3_l97_97292

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l97_97292


namespace minimum_button_presses_l97_97242

theorem minimum_button_presses :
  ∃ (r y g : ℕ), 
    2 * y - r = 3 ∧ 2 * g - y = 3 ∧ r + y + g = 9 :=
by sorry

end minimum_button_presses_l97_97242


namespace regular_polygon_sides_l97_97828

theorem regular_polygon_sides (O A B : Type) (angle_OAB : ℝ) 
  (h_angle : angle_OAB = 72) : 
  (360 / angle_OAB = 5) := 
by 
  sorry

end regular_polygon_sides_l97_97828


namespace rectangle_shorter_side_length_l97_97317

theorem rectangle_shorter_side_length (rope_length : ℕ) (long_side : ℕ) : 
  rope_length = 100 → long_side = 28 → 
  ∃ short_side : ℕ, (2 * long_side + 2 * short_side = rope_length) ∧ short_side = 22 :=
by
  sorry

end rectangle_shorter_side_length_l97_97317


namespace rachel_plants_lamps_l97_97428

-- Define the conditions as types
def plants : Type := { fern1 : Prop // true } × { fern2 : Prop // true } × { cactus : Prop // true }
def lamps : Type := { yellow1 : Prop // true } × { yellow2 : Prop // true } × { blue1 : Prop // true } × { blue2 : Prop // true }

-- A function that counts the distribution of plants under lamps
noncomputable def count_ways (p : plants) (l : lamps) : ℕ :=
  -- Here we should define the function that counts the number of configurations, 
  -- but since we are only defining the problem here we'll skip this part.
  sorry

-- The statement to prove
theorem rachel_plants_lamps :
  ∀ (p : plants) (l : lamps), count_ways p l = 14 :=
by
  sorry

end rachel_plants_lamps_l97_97428


namespace distance_between_towns_l97_97181

theorem distance_between_towns 
  (rate1 rate2 : ℕ) (time : ℕ) (distance : ℕ)
  (h_rate1 : rate1 = 48)
  (h_rate2 : rate2 = 42)
  (h_time : time = 5)
  (h_distance : distance = rate1 * time + rate2 * time) : 
  distance = 450 :=
by
  sorry

end distance_between_towns_l97_97181


namespace game_A_greater_game_B_l97_97916

-- Defining the probabilities and independence condition
def P_H := 2 / 3
def P_T := 1 / 3
def independent_tosses := true

-- Game A Probability Definition
def P_A := (P_H ^ 3) + (P_T ^ 3)

-- Game B Probability Definition
def P_B := ((P_H ^ 2) + (P_T ^ 2)) ^ 2

-- Statement to be proved
theorem game_A_greater_game_B : P_A = (27:ℚ) / 81 ∧ P_B = (25:ℚ) / 81 ∧ ((27:ℚ) / 81 - (25:ℚ) / 81 = (2:ℚ) / 81) := 
by
  -- P_A has already been computed: 1/3 = 27/81
  -- P_B has already been computed: 25/81
  sorry

end game_A_greater_game_B_l97_97916


namespace stamps_ratio_l97_97831

theorem stamps_ratio (orig_stamps_P : ℕ) (addie_stamps : ℕ) (final_stamps_P : ℕ) 
  (h₁ : orig_stamps_P = 18) (h₂ : addie_stamps = 72) (h₃ : final_stamps_P = 36) :
  (final_stamps_P - orig_stamps_P) / addie_stamps = 1 / 4 :=
by {
  sorry
}

end stamps_ratio_l97_97831


namespace common_ratio_of_geometric_series_l97_97389

theorem common_ratio_of_geometric_series (a₁ q : ℝ) 
  (S_3 : ℝ) (S_2 : ℝ) 
  (hS3 : S_3 = a₁ * (1 - q^3) / (1 - q)) 
  (hS2 : S_2 = a₁ * (1 - q^2) / (1 - q)) 
  (h_ratio : S_3 / S_2 = 3 / 2) :
  q = 1 ∨ q = -1/2 :=
by
  -- Proof goes here.
  sorry

end common_ratio_of_geometric_series_l97_97389


namespace sum_of_integers_l97_97634

/-- Given two positive integers x and y such that the sum of their squares equals 181 
    and their product equals 90, prove that the sum of these two integers is 19. -/
theorem sum_of_integers (x y : ℤ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_integers_l97_97634


namespace simplify_fractions_l97_97717

-- Define the fractions and their product.
def fraction1 : ℚ := 14 / 3
def fraction2 : ℚ := 9 / -42

-- Define the product of the fractions with scalar multiplication by 5.
def product : ℚ := 5 * fraction1 * fraction2

-- The target theorem to prove the equivalence.
theorem simplify_fractions : product = -5 := 
sorry  -- Proof is omitted

end simplify_fractions_l97_97717


namespace initial_salmons_l97_97674

theorem initial_salmons (x : ℕ) (hx : 10 * x = 5500) : x = 550 := 
by
  sorry

end initial_salmons_l97_97674


namespace freshmen_count_l97_97180

theorem freshmen_count (n : ℕ) : n < 600 ∧ n % 25 = 24 ∧ n % 19 = 10 ↔ n = 574 := 
by sorry

end freshmen_count_l97_97180


namespace incorrect_statement_A_l97_97865

theorem incorrect_statement_A (x_1 x_2 y_1 y_2 : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2*x - 4*y - 4 = 0) ∧
  x_1 = 1 - Real.sqrt 5 ∧
  x_2 = 1 + Real.sqrt 5 ∧
  y_1 = 2 - 2 * Real.sqrt 2 ∧
  y_2 = 2 + 2 * Real.sqrt 2 →
  x_1 + x_2 ≠ -2 := by
  intro h
  sorry

end incorrect_statement_A_l97_97865


namespace cash_sales_amount_l97_97905

-- Definitions for conditions
def total_sales : ℕ := 80
def credit_sales : ℕ := (2 * total_sales) / 5

-- Statement of the proof problem
theorem cash_sales_amount :
  ∃ cash_sales : ℕ, cash_sales = total_sales - credit_sales ∧ cash_sales = 48 :=
by
  sorry

end cash_sales_amount_l97_97905


namespace min_value_of_sum_of_reciprocals_l97_97918

theorem min_value_of_sum_of_reciprocals 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.log (1 / a + 1 / b) / Real.log 4 = Real.log (1 / Real.sqrt (a * b)) / Real.log 2) : 
  1 / a + 1 / b ≥ 4 := 
by 
  sorry

end min_value_of_sum_of_reciprocals_l97_97918


namespace nth_equation_l97_97058

theorem nth_equation (n : ℕ) (h : n > 0) : (1 / n) * ((n^2 + 2 * n) / (n + 1)) - (1 / (n + 1)) = 1 :=
by
  sorry

end nth_equation_l97_97058


namespace point_C_number_l97_97462

theorem point_C_number (B C: ℝ) (h1 : B = 3) (h2 : |C - B| = 2) :
  C = 1 ∨ C = 5 := 
by {
  sorry
}

end point_C_number_l97_97462


namespace duration_of_time_l97_97330

variable (A B C : String)
variable {a1 : A = "Get up at 6:30"}
variable {b1 : B = "School ends at 3:40"}
variable {c1 : C = "It took 30 minutes to do the homework"}

theorem duration_of_time : C = "It took 30 minutes to do the homework" :=
  sorry

end duration_of_time_l97_97330


namespace angle_E_measure_l97_97656

theorem angle_E_measure (H F G E : ℝ) 
  (h1 : E = 2 * F) (h2 : F = 2 * G) (h3 : G = 1.25 * H) 
  (h4 : E + F + G + H = 360) : E = 150 := by
  sorry

end angle_E_measure_l97_97656


namespace exterior_angle_of_polygon_l97_97157

theorem exterior_angle_of_polygon (n : ℕ) (h₁ : (n - 2) * 180 = 1800) (h₂ : n > 2) :
  360 / n = 30 := by
    sorry

end exterior_angle_of_polygon_l97_97157


namespace calculate_shot_cost_l97_97085

theorem calculate_shot_cost :
  let num_pregnant_dogs := 3
  let puppies_per_dog := 4
  let shots_per_puppy := 2
  let cost_per_shot := 5
  let total_puppies := num_pregnant_dogs * puppies_per_dog
  let total_shots := total_puppies * shots_per_puppy
  let total_cost := total_shots * cost_per_shot
  total_cost = 120 :=
by
  sorry

end calculate_shot_cost_l97_97085


namespace count_house_numbers_l97_97439

def isPrime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def twoDigitPrimesBetween40And60 : List ℕ :=
  [41, 43, 47, 53, 59]

theorem count_house_numbers : 
  ∃ n : ℕ, n = 20 ∧ 
  ∀ (AB CD : ℕ), 
  AB ∈ twoDigitPrimesBetween40And60 → 
  CD ∈ twoDigitPrimesBetween40And60 → 
  AB ≠ CD → 
  true :=
by
  sorry

end count_house_numbers_l97_97439


namespace find_interest_rate_l97_97209

-- Conditions
def principal1 : ℝ := 100
def rate1 : ℝ := 0.05
def time1 : ℕ := 48

def principal2 : ℝ := 600
def time2 : ℕ := 4

-- The given interest produced by the first amount
def interest1 : ℝ := principal1 * rate1 * time1

-- The interest produced by the second amount should be the same
def interest2 (rate2 : ℝ) : ℝ := principal2 * rate2 * time2

-- The interest rate to prove
def rate2_correct : ℝ := 0.1

theorem find_interest_rate :
  ∃ rate2 : ℝ, interest2 rate2 = interest1 ∧ rate2 = rate2_correct :=
by
  sorry

end find_interest_rate_l97_97209


namespace fraction_value_l97_97378

variable (x y : ℝ)

theorem fraction_value (h : 1/x - 1/y = 3) : (2 * x + 3 * x * y - 2 * y) / (x - 2 * x * y - y) = 3 / 5 := 
by sorry

end fraction_value_l97_97378


namespace farmer_initial_days_l97_97963

theorem farmer_initial_days 
  (x : ℕ) 
  (plan_daily : ℕ) 
  (actual_daily : ℕ) 
  (extra_days : ℕ) 
  (left_area : ℕ) 
  (total_area : ℕ)
  (h1 : plan_daily = 120) 
  (h2 : actual_daily = 85) 
  (h3 : extra_days = 2) 
  (h4 : left_area = 40) 
  (h5 : total_area = 720): 
  85 * (x + extra_days) + left_area = total_area → x = 6 :=
by
  intros h
  sorry

end farmer_initial_days_l97_97963


namespace length_of_BC_l97_97962

def triangle_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 20

def triangle_area (a b : ℝ) : Prop :=
  (1/2) * a * b * (Real.sqrt 3 / 2) = 10

theorem length_of_BC (a b c : ℝ) (h1 : triangle_perimeter a b c) (h2 : triangle_area a b) : c = 7 :=
  sorry

end length_of_BC_l97_97962


namespace find_pairs_l97_97904
open Nat

theorem find_pairs (x p : ℕ) (hp : p.Prime) (hxp : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (x = 1 ∧ p.Prime) ∨ (x = 2 ∧ p = 2) ∨ (x = 1 ∧ p.Prime) ∨ (x = 3 ∧ p = 3) := 
by
  sorry


end find_pairs_l97_97904


namespace minimum_value_of_a_l97_97228

theorem minimum_value_of_a (A B C : ℝ) (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2 - b * c) 
  (h2 : (1/2) * b * c * (Real.sin A) = (3 * Real.sqrt 3) / 4)
  (h3 : A = Real.arccos (1/2)) :
  a ≥ Real.sqrt 3 := sorry

end minimum_value_of_a_l97_97228


namespace cost_price_A_min_cost_bshelves_l97_97498

-- Define the cost price of type B bookshelf
def costB_bshelf : ℝ := 300

-- Define the cost price of type A bookshelf
def costA_bshelf : ℝ := 1.2 * costB_bshelf

-- Define the total number of bookshelves
def total_bshelves : ℕ := 60

-- Define the condition for type A and type B bookshelves count
def typeBshelves := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves
def typeBshelves_constraints := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves ≤ 2 * typeAshelves

-- Define the equation for the costs
noncomputable def total_cost (typeAshelves : ℕ) : ℝ :=
  360 * typeAshelves + 300 * (total_bshelves - typeAshelves)

-- Define the goal: cost price of type A bookshelf is 360 yuan
theorem cost_price_A : costA_bshelf = 360 :=
by 
  sorry

-- Define the goal: the school should buy 20 type A bookshelves and 40 type B bookshelves to minimize cost
theorem min_cost_bshelves : ∃ typeAshelves : ℕ, typeAshelves = 20 ∧ typeBshelves typeAshelves = 40 :=
by
  sorry

end cost_price_A_min_cost_bshelves_l97_97498


namespace range_of_function_l97_97423

theorem range_of_function : 
  (∀ x, (Real.pi / 4) ≤ x ∧ x ≤ (Real.pi / 2) → 
   1 ≤ (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ∧ 
    (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ≤ 3 / 2) :=
sorry

end range_of_function_l97_97423


namespace length_of_ST_l97_97365

theorem length_of_ST (PQ PS : ℝ) (ST : ℝ) (hPQ : PQ = 8) (hPS : PS = 7) 
  (h_area_eq : (1 / 2) * PQ * (PS * (1 / PS) * 8) = PQ * PS) : 
  ST = 2 * Real.sqrt 65 := 
by
  -- proof steps (to be written)
  sorry

end length_of_ST_l97_97365


namespace integer_type_l97_97041

theorem integer_type (f : ℕ) (h : f = 14) (x : ℕ) (hx : 3150 * f = x * x) : f > 0 :=
by
  sorry

end integer_type_l97_97041


namespace min_stamps_l97_97782

theorem min_stamps : ∃ (x y : ℕ), 5 * x + 7 * y = 35 ∧ x + y = 5 :=
by
  have : ∀ (x y : ℕ), 5 * x + 7 * y = 35 → x + y = 5 → True := sorry
  sorry

end min_stamps_l97_97782


namespace base16_to_base2_bits_l97_97160

theorem base16_to_base2_bits :
  ∀ (n : ℕ), n = 16^4 * 7 + 16^3 * 7 + 16^2 * 7 + 16 * 7 + 7 → (2^18 ≤ n ∧ n < 2^19) → 
  ∃ b : ℕ, b = 19 := 
by
  intros n hn hpow
  sorry

end base16_to_base2_bits_l97_97160


namespace prove_a_zero_l97_97456

-- Define two natural numbers a and b
variables (a b : ℕ)

-- Condition: For every natural number n, 2^n * a + b is a perfect square
def condition := ∀ n : ℕ, ∃ k : ℕ, 2^n * a + b = k^2

-- Statement to prove: a = 0
theorem prove_a_zero (h : condition a b) : a = 0 := sorry

end prove_a_zero_l97_97456


namespace set_representation_l97_97244

theorem set_representation :
  {p : ℕ × ℕ | 2 * p.1 + 3 * p.2 = 16} = {(2, 4), (5, 2), (8, 0)} :=
by
  sorry

end set_representation_l97_97244


namespace six_smallest_distinct_integers_l97_97637

theorem six_smallest_distinct_integers:
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a * b * c * d * e = 999999 ∧ a = 3 ∧
    f = 37 ∨
    a * b * c * d * f = 999999 ∧ a = 3 ∧ e = 13 ∨ 
    a * b * d * f * e = 999999 ∧ c = 9 ∧ 
    a * c * d * e * f = 999999 ∧ b = 7 ∧ 
    b * c * d * e * f = 999999 ∧ a = 3 := 
sorry

end six_smallest_distinct_integers_l97_97637


namespace janet_faster_playtime_l97_97599

theorem janet_faster_playtime 
  (initial_minutes : ℕ)
  (initial_seconds : ℕ)
  (faster_rate : ℝ)
  (initial_time_in_seconds := initial_minutes * 60 + initial_seconds)
  (target_time_in_seconds := initial_time_in_seconds / faster_rate) :
  initial_minutes = 3 →
  initial_seconds = 20 →
  faster_rate = 1.25 →
  target_time_in_seconds = 160 :=
by
  intros h1 h2 h3
  sorry

end janet_faster_playtime_l97_97599


namespace simplify_sqrt_product_l97_97328

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end simplify_sqrt_product_l97_97328


namespace calvin_total_insects_l97_97567

def R : ℕ := 15
def S : ℕ := 2 * R - 8
def C : ℕ := 11 -- rounded from (1/2) * R + 3
def P : ℕ := 3 * S + 7
def B : ℕ := 4 * C - 2
def E : ℕ := 3 * (R + S + C + P + B)
def total_insects : ℕ := R + S + C + P + B + E

theorem calvin_total_insects : total_insects = 652 :=
by
  -- service the proof here.
  sorry

end calvin_total_insects_l97_97567


namespace log_comparison_l97_97204

theorem log_comparison :
  (Real.log 80 / Real.log 20) < (Real.log 640 / Real.log 80) :=
by
  sorry

end log_comparison_l97_97204


namespace problem_a_plus_b_equals_10_l97_97255

theorem problem_a_plus_b_equals_10 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
  (h_equation : 3 * a + 4 * b = 10 * a + b) : a + b = 10 :=
by {
  sorry
}

end problem_a_plus_b_equals_10_l97_97255


namespace wanda_blocks_l97_97165

theorem wanda_blocks (initial_blocks: ℕ) (additional_blocks: ℕ) (total_blocks: ℕ) : 
  initial_blocks = 4 → additional_blocks = 79 → total_blocks = initial_blocks + additional_blocks → total_blocks = 83 :=
by
  intros hi ha ht
  rw [hi, ha] at ht
  exact ht

end wanda_blocks_l97_97165


namespace abs_neg_five_halves_l97_97373

theorem abs_neg_five_halves : abs (-5 / 2) = 5 / 2 := by
  sorry

end abs_neg_five_halves_l97_97373


namespace payment_to_C_l97_97664

theorem payment_to_C (A_days B_days total_payment days_taken : ℕ) 
  (A_work_rate B_work_rate : ℚ)
  (work_fraction_by_A_and_B : ℚ)
  (remaining_work_fraction_by_C : ℚ)
  (C_payment : ℚ) :
  A_days = 6 →
  B_days = 8 →
  total_payment = 3360 →
  days_taken = 3 →
  A_work_rate = 1/6 →
  B_work_rate = 1/8 →
  work_fraction_by_A_and_B = (A_work_rate + B_work_rate) * days_taken →
  remaining_work_fraction_by_C = 1 - work_fraction_by_A_and_B →
  C_payment = total_payment * remaining_work_fraction_by_C →
  C_payment = 420 := 
by
  intros hA hB hTP hD hAR hBR hWF hRWF hCP
  sorry

end payment_to_C_l97_97664


namespace find_x_l97_97908

theorem find_x (x : ℚ) (h : ⌊x⌋ + x = 15/4) : x = 15/4 := by
  sorry

end find_x_l97_97908


namespace work_days_together_l97_97122

variable (d : ℝ) (j : ℝ)

theorem work_days_together (hd : d = 1 / 5) (hj : j = 1 / 9) :
  1 / (d + j) = 45 / 14 := by
  sorry

end work_days_together_l97_97122


namespace exactly_one_gt_one_of_abc_eq_one_l97_97802

theorem exactly_one_gt_one_of_abc_eq_one 
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) 
  (h_sum : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b < 1 ∧ c < 1) ∨ (a < 1 ∧ 1 < b ∧ c < 1) ∨ (a < 1 ∧ b < 1 ∧ 1 < c) :=
sorry

end exactly_one_gt_one_of_abc_eq_one_l97_97802


namespace max_area_of_rectangle_l97_97581

-- Define the parameters and the problem
def perimeter := 150
def half_perimeter := perimeter / 2

theorem max_area_of_rectangle (x : ℕ) (y : ℕ) 
  (h1 : x + y = half_perimeter)
  (h2 : x > 0) (h3 : y > 0) :
  (∃ x y, x * y ≤ 1406) := 
sorry

end max_area_of_rectangle_l97_97581


namespace intersection_M_S_l97_97121

namespace ProofProblem

def M : Set ℕ := { x | 0 < x ∧ x < 4 }

def S : Set ℕ := { 2, 3, 5 }

theorem intersection_M_S :
  M ∩ S = { 2, 3 } := by
  sorry

end ProofProblem

end intersection_M_S_l97_97121


namespace find_xsq_plus_inv_xsq_l97_97219

theorem find_xsq_plus_inv_xsq (x : ℝ) (h : 35 = x^6 + 1/(x^6)) : x^2 + 1/(x^2) = 37 :=
sorry

end find_xsq_plus_inv_xsq_l97_97219


namespace peanuts_remaining_l97_97598

def initial_peanuts := 220
def brock_fraction := 1 / 4
def bonita_fraction := 2 / 5
def carlos_peanuts := 17

noncomputable def peanuts_left := initial_peanuts - (initial_peanuts * brock_fraction + ((initial_peanuts - initial_peanuts * brock_fraction) * bonita_fraction)) - carlos_peanuts

theorem peanuts_remaining : peanuts_left = 82 :=
by
  sorry

end peanuts_remaining_l97_97598


namespace land_tax_calculation_l97_97376

theorem land_tax_calculation
  (area : ℝ)
  (value_per_acre : ℝ)
  (tax_rate : ℝ)
  (total_cadastral_value : ℝ := area * value_per_acre)
  (land_tax : ℝ := total_cadastral_value * tax_rate) :
  area = 15 → value_per_acre = 100000 → tax_rate = 0.003 → land_tax = 4500 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end land_tax_calculation_l97_97376


namespace hyperbola_condition_l97_97465

noncomputable def a_b_sum (a b : ℝ) : ℝ :=
  a + b

theorem hyperbola_condition
  (a b : ℝ)
  (h1 : a^2 - b^2 = 1)
  (h2 : abs (a - b) = 2)
  (h3 : a > b) :
  a_b_sum a b = 1/2 :=
sorry

end hyperbola_condition_l97_97465


namespace arithmetic_sequence_20th_term_l97_97853

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 5
  let n := 20
  let a_n := a + (n - 1) * d
  a_n = 97 := by
  sorry

end arithmetic_sequence_20th_term_l97_97853


namespace range_of_f_l97_97723

noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.cos x - 4 * Real.sin x

theorem range_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 Real.pi → f x ∈ Set.Icc (-5) 3) ∧
  (∀ y : ℝ, y ∈ Set.Icc (-5) 3 → ∃ x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ f x = y) :=
by
  sorry

end range_of_f_l97_97723


namespace vanessa_phone_pictures_l97_97610

theorem vanessa_phone_pictures
  (C : ℕ) (P : ℕ) (hC : C = 7)
  (hAlbums : 5 * 6 = 30)
  (hTotal : 30 = P + C) :
  P = 23 := by
  sorry

end vanessa_phone_pictures_l97_97610


namespace triangle_inequality_l97_97889

theorem triangle_inequality 
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  2 * (a + b + c) * (a * b + b * c + c * a) ≤ (a + b + c) * (a^2 + b^2 + c^2) + 9 * a * b * c :=
by
  sorry

end triangle_inequality_l97_97889


namespace max_value_of_sum_l97_97969

open Real

theorem max_value_of_sum (x y z : ℝ)
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
    (h2 : (1 / x) + (1 / y) + (1 / z) + x + y + z = 0)
    (h3 : (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1) ∧ (z ≤ -1 ∨ z ≥ 1)) :
    x + y + z ≤ 0 := 
sorry

end max_value_of_sum_l97_97969


namespace members_in_both_sets_are_23_l97_97444

variable (U A B : Finset ℕ)
variable (count_U count_A count_B count_neither count_both : ℕ)

theorem members_in_both_sets_are_23 (hU : count_U = 192)
    (hA : count_A = 107) (hB : count_B = 49) (hNeither : count_neither = 59) :
    count_both = 23 :=
by
  sorry

end members_in_both_sets_are_23_l97_97444


namespace total_days_2001_to_2004_l97_97101

def regular_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def num_regular_years : ℕ := 3
def num_leap_years : ℕ := 1

theorem total_days_2001_to_2004 : 
  (num_regular_years * regular_year_days) + (num_leap_years * leap_year_days) = 1461 :=
by
  sorry

end total_days_2001_to_2004_l97_97101


namespace maximum_sum_l97_97028

theorem maximum_sum (a b c d : ℕ) (h₀ : a < b ∧ b < c ∧ c < d)
  (h₁ : (c + d) + (a + b + c) = 2017) : a + b + c + d ≤ 806 :=
sorry

end maximum_sum_l97_97028


namespace annual_income_of_a_l97_97931

-- Definitions based on the conditions
def monthly_income_ratio (a_income b_income : ℝ) : Prop := a_income / b_income = 5 / 2
def income_percentage (part whole : ℝ) : Prop := part / whole = 12 / 100
def c_monthly_income : ℝ := 15000
def b_monthly_income (c_income : ℝ) := c_income + 0.12 * c_income

-- The theorem to prove
theorem annual_income_of_a : ∀ (a_income b_income c_income : ℝ),
  monthly_income_ratio a_income b_income ∧
  b_income = b_monthly_income c_income ∧
  c_income = c_monthly_income →
  (a_income * 12) = 504000 :=
by
  -- Here we do not need to fill out the proof, so we use sorry
  sorry

end annual_income_of_a_l97_97931


namespace train_pass_time_l97_97068

noncomputable def train_length : ℕ := 360
noncomputable def platform_length : ℕ := 140
noncomputable def train_speed_kmh : ℕ := 45

noncomputable def convert_speed_to_mps (speed_kmh : ℕ) : ℚ := 
  (speed_kmh * 1000) / 3600

noncomputable def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

noncomputable def time_to_pass (distance : ℕ) (speed_mps : ℚ) : ℚ :=
  distance / speed_mps

theorem train_pass_time 
  (train_len : ℕ) 
  (platform_len : ℕ) 
  (speed_kmh : ℕ) : 
  time_to_pass (total_distance train_len platform_len) (convert_speed_to_mps speed_kmh) = 40 := 
by 
  sorry

end train_pass_time_l97_97068


namespace probability_top_card_is_joker_l97_97468

def deck_size : ℕ := 54
def joker_count : ℕ := 2

theorem probability_top_card_is_joker :
  (joker_count : ℝ) / (deck_size : ℝ) = 1 / 27 :=
by
  sorry

end probability_top_card_is_joker_l97_97468


namespace largest_of_four_consecutive_odd_numbers_l97_97595

theorem largest_of_four_consecutive_odd_numbers (x : ℤ) : 
  (x % 2 = 1) → 
  ((x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) →
  (x + 6 = 27) :=
by
  sorry

end largest_of_four_consecutive_odd_numbers_l97_97595


namespace find_x_value_l97_97823

theorem find_x_value (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
sorry

end find_x_value_l97_97823


namespace minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l97_97229

theorem minimum_additional_games_to_reach_90_percent (N : ℕ) : 
  (2 + N) * 10 ≥ (5 + N) * 9 ↔ N ≥ 25 := 
sorry

-- An alternative approach to assert directly as exactly 25 by using the condition’s natural number ℕ could be as follows:
theorem hawks_minimum_games_needed_to_win (N : ℕ) : 
  ∀ N, (2 + N) * 10 / (5 + N) ≥ 9 / 10 → N ≥ 25 := 
sorry

end minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l97_97229


namespace find_a_l97_97306

theorem find_a (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 3) (h₂ : 3 / a + 6 / b = 2 / 3) : 
  a = 9 * b / (2 * b - 18) :=
by
  sorry

end find_a_l97_97306


namespace functional_equation_solution_l97_97338

open Nat

theorem functional_equation_solution :
  (∀ (f : ℕ → ℕ), 
    (∀ (x y : ℕ), 0 ≤ y + f x - (Nat.iterate f (f y) x) ∧ (y + f x - (Nat.iterate f (f y) x) ≤ 1)) →
    (∀ n, f n = n + 1)) :=
by
  intro f h
  sorry

end functional_equation_solution_l97_97338


namespace find_x2_plus_y2_l97_97293

theorem find_x2_plus_y2
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := 
by
  sorry

end find_x2_plus_y2_l97_97293


namespace same_terminal_side_l97_97116

theorem same_terminal_side (k : ℤ) : ∃ k : ℤ, (2 * k * Real.pi - Real.pi / 6) = 11 * Real.pi / 6 := by
  sorry

end same_terminal_side_l97_97116


namespace valid_pair_l97_97911

-- Definitions of the animals
inductive Animal
| lion
| tiger
| leopard
| elephant

open Animal

-- Given conditions
def condition1 (selected : Animal → Prop) : Prop :=
  selected lion → selected tiger

def condition2 (selected : Animal → Prop) : Prop :=
  ¬selected leopard → ¬selected tiger

def condition3 (selected : Animal → Prop) : Prop :=
  selected leopard → ¬selected elephant

-- Main theorem to prove
theorem valid_pair (selected : Animal → Prop) (pair : Animal × Animal) :
  (pair = (tiger, leopard)) ↔ 
  (condition1 selected ∧ condition2 selected ∧ condition3 selected) :=
sorry

end valid_pair_l97_97911


namespace mike_laptop_row_division_impossible_l97_97597

theorem mike_laptop_row_division_impossible (total_laptops : ℕ) (num_rows : ℕ) 
(types_ratios : List ℕ)
(H_total : total_laptops = 44)
(H_rows : num_rows = 5) 
(H_ratio : types_ratios = [2, 3, 4]) :
  ¬ (∃ (n : ℕ), (total_laptops = n * num_rows) 
  ∧ (n % (types_ratios.sum) = 0)
  ∧ (∀ (t : ℕ), t ∈ types_ratios → t ≤ n)) := sorry

end mike_laptop_row_division_impossible_l97_97597


namespace xy_product_l97_97032

theorem xy_product (x y : ℝ) (h : x^2 + y^2 - 22*x - 20*y + 221 = 0) : x * y = 110 := 
sorry

end xy_product_l97_97032


namespace henry_money_l97_97082

-- Define the conditions
def initial : ℕ := 11
def birthday : ℕ := 18
def spent : ℕ := 10

-- Define the final amount
def final_amount : ℕ := initial + birthday - spent

-- State the theorem
theorem henry_money : final_amount = 19 := by
  -- Skipping the proof
  sorry

end henry_money_l97_97082


namespace fraction_identity_l97_97636

theorem fraction_identity (a b : ℝ) (h1 : 1/a + 2/b = 1) (h2 : a ≠ -b) : 
  (ab - a)/(a + b) = 1 := 
by 
  sorry

end fraction_identity_l97_97636


namespace blue_string_length_l97_97650

def length_red := 8
def length_white := 5 * length_red
def length_blue := length_white / 8

theorem blue_string_length : length_blue = 5 := by
  sorry

end blue_string_length_l97_97650


namespace unique_solution_exists_l97_97791

theorem unique_solution_exists :
  ∃ (x y : ℝ), x = -13 / 96 ∧ y = 13 / 40 ∧
    (x / Real.sqrt (x^2 + y^2) - 1/x = 7) ∧
    (y / Real.sqrt (x^2 + y^2) + 1/y = 4) :=
by
  sorry

end unique_solution_exists_l97_97791


namespace number_of_candies_picked_up_l97_97174

-- Definitions of the conditions
def num_sides_decagon := 10
def diagonals_from_one_vertex (n : Nat) : Nat := n - 3

-- The theorem stating the number of candies Hyeonsu picked up
theorem number_of_candies_picked_up : diagonals_from_one_vertex num_sides_decagon = 7 := by
  sorry

end number_of_candies_picked_up_l97_97174


namespace chess_tournament_boys_l97_97024

noncomputable def num_boys_in_tournament (n k : ℕ) : Prop :=
  (6 + k * n = (n + 2) * (n + 1) / 2) ∧ (n > 2)

theorem chess_tournament_boys :
  ∃ (n : ℕ), num_boys_in_tournament n (if n = 5 then 3 else if n = 10 then 6 else 0) ∧ (n = 5 ∨ n = 10) :=
by
  sorry

end chess_tournament_boys_l97_97024


namespace gcd_of_60_and_75_l97_97795

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l97_97795


namespace first_number_is_38_l97_97832

theorem first_number_is_38 (x y : ℕ) (h1 : x + 2 * y = 124) (h2 : y = 43) : x = 38 :=
by
  sorry

end first_number_is_38_l97_97832


namespace x_intercept_l97_97148

theorem x_intercept (x y : ℝ) (h : 4 * x - 3 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by {
  sorry
}

end x_intercept_l97_97148


namespace find_n_divides_2_pow_2000_l97_97038

theorem find_n_divides_2_pow_2000 (n : ℕ) (h₁ : n > 2) :
  (1 + n + n * (n - 1) / 2 + n * (n - 1) * (n - 2) / 6) ∣ (2 ^ 2000) →
  n = 3 ∨ n = 7 ∨ n = 23 :=
sorry

end find_n_divides_2_pow_2000_l97_97038


namespace problem1_problem2_l97_97420

theorem problem1 :
  ( (1/2) ^ (-2) - 0.01 ^ (-1) + (-(1 + 1/7)) ^ (0)) = -95 := by
  sorry

theorem problem2 (x : ℝ) :
  (x - 2) * (x + 1) - (x - 1) ^ 2 = x - 3 := by
  sorry

end problem1_problem2_l97_97420


namespace find_supplementary_angle_l97_97329

noncomputable def degree (x : ℝ) : ℝ := x
noncomputable def complementary_angle (x : ℝ) : ℝ := 90 - x
noncomputable def supplementary_angle (x : ℝ) : ℝ := 180 - x

theorem find_supplementary_angle
  (x : ℝ)
  (h1 : degree x / complementary_angle x = 1 / 8) :
  supplementary_angle x = 170 :=
by
  sorry

end find_supplementary_angle_l97_97329


namespace total_time_is_correct_l97_97874

-- Defining the number of items
def chairs : ℕ := 7
def tables : ℕ := 3
def bookshelves : ℕ := 2
def lamps : ℕ := 4

-- Defining the time spent on each type of furniture
def time_per_chair : ℕ := 4
def time_per_table : ℕ := 8
def time_per_bookshelf : ℕ := 12
def time_per_lamp : ℕ := 2

-- Defining the total time calculation
def total_time : ℕ :=
  (chairs * time_per_chair) + 
  (tables * time_per_table) +
  (bookshelves * time_per_bookshelf) +
  (lamps * time_per_lamp)

-- Theorem stating the total time
theorem total_time_is_correct : total_time = 84 :=
by
  -- Skipping the proof details
  sorry

end total_time_is_correct_l97_97874


namespace gcd_expression_infinite_composite_pairs_exists_l97_97079

-- Part (a)
theorem gcd_expression (n : ℕ) (a : ℕ) (b : ℕ) (hn : n > 0) (ha : a > 0) (hb : b > 0) :
  Nat.gcd (n^a + 1) (n^b + 1) ≤ n^(Nat.gcd a b) + 1 :=
by
  sorry

-- Part (b)
theorem infinite_composite_pairs_exists (n : ℕ) (hn : n > 0) :
  ∃ (pairs : ℕ × ℕ → Prop), (∀ a b, pairs (a, b) → a > 1 ∧ b > 1 ∧ ∃ d, d > 1 ∧ a = d ∧ b = dn) ∧
  (∀ a b, pairs (a, b) → Nat.gcd (n^a + 1) (n^b + 1) = n^(Nat.gcd a b) + 1) ∧
  (∀ x y, x > 1 → y > 1 → x ∣ y ∨ y ∣ x → ¬pairs (x, y)) :=
by
  sorry

end gcd_expression_infinite_composite_pairs_exists_l97_97079


namespace unique_x1_sequence_l97_97580

open Nat

theorem unique_x1_sequence (x1 : ℝ) (x : ℕ → ℝ)
  (h₀ : x 1 = x1)
  (h₁ : ∀ n, x (n + 1) = x n * (x n + 1 / (n + 1))) :
  (∃! x1, (0 < x1 ∧ x1 < 1) ∧ 
   (∀ n, 0 < x n ∧ x n < x (n + 1) ∧ x (n + 1) < 1)) := sorry

end unique_x1_sequence_l97_97580


namespace product_ABCD_is_9_l97_97890

noncomputable def A : ℝ := Real.sqrt 2018 + Real.sqrt 2019 + 1
noncomputable def B : ℝ := -Real.sqrt 2018 - Real.sqrt 2019 - 1
noncomputable def C : ℝ := Real.sqrt 2018 - Real.sqrt 2019 + 1
noncomputable def D : ℝ := Real.sqrt 2019 - Real.sqrt 2018 + 1

theorem product_ABCD_is_9 : A * B * C * D = 9 :=
by sorry

end product_ABCD_is_9_l97_97890


namespace minimum_value_l97_97231

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 / x + 1 / y = 1) : 3 * x + 4 * y ≥ 25 :=
sorry

end minimum_value_l97_97231


namespace tax_collection_amount_l97_97602

theorem tax_collection_amount (paid_tax : ℝ) (willam_percentage : ℝ) (total_collected : ℝ) (h_paid: paid_tax = 480) (h_percentage: willam_percentage = 0.3125) :
    total_collected = 1536 :=
by
  sorry

end tax_collection_amount_l97_97602


namespace mass_percentage_of_Ca_in_CaO_is_correct_l97_97578

noncomputable def molarMass_Ca : ℝ := 40.08
noncomputable def molarMass_O : ℝ := 16.00
noncomputable def molarMass_CaO : ℝ := molarMass_Ca + molarMass_O
noncomputable def massPercentageCaInCaO : ℝ := (molarMass_Ca / molarMass_CaO) * 100

theorem mass_percentage_of_Ca_in_CaO_is_correct :
  massPercentageCaInCaO = 71.47 :=
by
  -- This is where the proof would go
  sorry

end mass_percentage_of_Ca_in_CaO_is_correct_l97_97578


namespace tens_digit_of_8_pow_2048_l97_97132

theorem tens_digit_of_8_pow_2048 : (8^2048 % 100) / 10 = 8 := 
by
  sorry

end tens_digit_of_8_pow_2048_l97_97132


namespace min_value_Px_Py_l97_97474

def P (τ : ℝ) : ℝ := (τ + 1)^3

theorem min_value_Px_Py (x y : ℝ) (h : x + y = 0) : P x + P y = 2 :=
sorry

end min_value_Px_Py_l97_97474


namespace divisibility_of_n_l97_97267

theorem divisibility_of_n
  (n : ℕ) (n_gt_1 : n > 1)
  (h : n ∣ (6^n - 1)) : 5 ∣ n :=
by
  sorry

end divisibility_of_n_l97_97267


namespace afternoon_emails_l97_97363

theorem afternoon_emails (A : ℕ) (five_morning_emails : ℕ) (two_more : five_morning_emails + 2 = A) : A = 7 :=
by
  sorry

end afternoon_emails_l97_97363


namespace jack_sees_color_change_l97_97739

noncomputable def traffic_light_cycle := 95    -- Total duration of the traffic light cycle
noncomputable def change_window := 15          -- Duration window where color change occurs
def observation_interval := 5                  -- Length of Jack's observation interval

/-- Probability that Jack sees the color change during his observation. -/
def probability_of_observing_change (cycle: ℕ) (window: ℕ) : ℚ :=
  window / cycle

theorem jack_sees_color_change :
  probability_of_observing_change traffic_light_cycle change_window = 3 / 19 :=
by
  -- We only need the statement for verification
  sorry

end jack_sees_color_change_l97_97739


namespace inequality_holds_l97_97364

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
    a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
by
  sorry

end inequality_holds_l97_97364


namespace number_with_specific_places_l97_97401

theorem number_with_specific_places :
  ∃ (n : Real), 
    (n / 10 % 10 = 6) ∧ -- tens place
    (n / 1 % 10 = 0) ∧  -- ones place
    (n * 10 % 10 = 0) ∧  -- tenths place
    (n * 100 % 10 = 6) →  -- hundredths place
    n = 60.06 :=
by
  sorry

end number_with_specific_places_l97_97401


namespace statement_2_statement_4_l97_97655

-- Definitions and conditions
variables {Point Line Plane : Type}
variable (a b : Line)
variable (α : Plane)

def parallel (l1 l2 : Line) : Prop := sorry  -- Define parallel relation
def perp (l1 l2 : Line) : Prop := sorry  -- Define perpendicular relation
def perp_plane (l : Line) (p : Plane) : Prop := sorry  -- Define line-plane perpendicular relation
def lies_in (l : Line) (p : Plane) : Prop := sorry  -- Define line lies in plane relation

-- Problem statement 2: If a ∥ b and a ⟂ α, then b ⟂ α
theorem statement_2 (h1 : parallel a b) (h2 : perp_plane a α) : perp_plane b α := sorry

-- Problem statement 4: If a ⟂ α and b ⟂ a, then a ∥ b
theorem statement_4 (h1 : perp_plane a α) (h2 : perp b a) : parallel a b := sorry

end statement_2_statement_4_l97_97655


namespace max_rectangle_area_l97_97002

theorem max_rectangle_area (P : ℝ) (hP : 0 < P) : 
  ∃ (x y : ℝ), (2*x + 2*y = P) ∧ (x * y = P ^ 2 / 16) :=
by
  sorry

end max_rectangle_area_l97_97002


namespace tens_digit_of_9_pow_2023_l97_97666

theorem tens_digit_of_9_pow_2023 : (9 ^ 2023) % 100 / 10 = 2 :=
by sorry

end tens_digit_of_9_pow_2023_l97_97666


namespace geometric_sum_5_l97_97214

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = a n * r ∧ a (m + 1) = a m * r

theorem geometric_sum_5 (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) (h3 : ∀ n, 0 < a n) :
  a 3 + a 5 = 5 :=
sorry

end geometric_sum_5_l97_97214


namespace hammerhead_teeth_fraction_l97_97394

theorem hammerhead_teeth_fraction (f : ℚ) : 
  let t := 180 
  let h := f * t
  let w := 2 * (t + h)
  w = 420 → f = (1 : ℚ) / 6 := by
  intros _ 
  sorry

end hammerhead_teeth_fraction_l97_97394


namespace arithmetic_sequence_sum_l97_97709

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ)
variable (h_arith : is_arithmetic_sequence a)
variable (h_sum : a 2 + a 3 + a 10 + a 11 = 48)

-- Goal
theorem arithmetic_sequence_sum : a 6 + a 7 = 24 :=
sorry

end arithmetic_sequence_sum_l97_97709


namespace supply_without_leak_last_for_20_days_l97_97861

variable (C V : ℝ)

-- Condition 1: if there is a 10-liter leak per day, the supply lasts for 15 days
axiom h1 : C = 15 * (V + 10)

-- Condition 2: if there is a 20-liter leak per day, the supply lasts for 12 days
axiom h2 : C = 12 * (V + 20)

-- The problem to prove: without any leak, the tank can supply water to the village for 20 days
theorem supply_without_leak_last_for_20_days (C V : ℝ) (h1 : C = 15 * (V + 10)) (h2 : C = 12 * (V + 20)) : C / V = 20 := 
by 
  sorry

end supply_without_leak_last_for_20_days_l97_97861


namespace simultaneous_eq_solvable_l97_97971

theorem simultaneous_eq_solvable (m : ℝ) : 
  (∃ x y : ℝ, y = m * x + 4 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  sorry

end simultaneous_eq_solvable_l97_97971


namespace function_range_is_correct_l97_97189

noncomputable def function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.log (x^2 - 6 * x + 17) }

theorem function_range_is_correct : function_range = {x : ℝ | x ≤ Real.log 8} :=
by
  sorry

end function_range_is_correct_l97_97189


namespace incorrect_statement_B_l97_97392

variable (a : Nat → Int) (S : Nat → Int)
variable (d : Int)

-- Given conditions
axiom S_5_lt_S_6 : S 5 < S 6
axiom S_6_eq_S_7 : S 6 = S 7
axiom S_7_gt_S_8 : S 7 > S 8
axiom S_n : ∀ n, S n = n * a n

-- Question to prove statement B is incorrect 
theorem incorrect_statement_B : ∃ (d : Int), (S 9 < S 5) :=
by 
  -- Proof goes here
  sorry

end incorrect_statement_B_l97_97392


namespace find_speed_of_man_l97_97714

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
(v_m + v_s = 6) ∧ (v_m - v_s = 8)

theorem find_speed_of_man :
  ∃ v_m v_s : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 7 :=
by
  sorry

end find_speed_of_man_l97_97714


namespace book_original_price_l97_97776

-- Definitions for conditions
def selling_price := 56
def profit_percentage := 75

-- Statement of the theorem
theorem book_original_price : ∃ CP : ℝ, selling_price = CP * (1 + profit_percentage / 100) ∧ CP = 32 :=
by
  sorry

end book_original_price_l97_97776


namespace difference_in_perimeters_of_rectangles_l97_97193

theorem difference_in_perimeters_of_rectangles 
  (l h : ℝ) (hl : l ≥ 0) (hh : h ≥ 0) :
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  difference = 24 :=
by
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  sorry

end difference_in_perimeters_of_rectangles_l97_97193


namespace find_f_of_2_l97_97196

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ (x : ℝ), x > 0 → f (Real.log x / Real.log 2) = 2 ^ x) : f 2 = 16 :=
by
  sorry

end find_f_of_2_l97_97196


namespace spherical_to_rectangular_correct_l97_97495

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) := by
  sorry

end spherical_to_rectangular_correct_l97_97495


namespace percentage_of_720_equals_356_point_4_l97_97513

theorem percentage_of_720_equals_356_point_4 : 
  let part := 356.4
  let whole := 720
  (part / whole) * 100 = 49.5 :=
by
  sorry

end percentage_of_720_equals_356_point_4_l97_97513


namespace add_number_l97_97535

theorem add_number (x : ℕ) (h : 43 + x = 81) : x + 25 = 63 :=
by {
  -- Since this is focusing on the structure and statement no proof steps are required
  sorry
}

end add_number_l97_97535


namespace trigonometric_identity_cos24_cos36_sub_sin24_cos54_l97_97835

theorem trigonometric_identity_cos24_cos36_sub_sin24_cos54  :
  (Real.cos (24 * Real.pi / 180) * Real.cos (36 * Real.pi / 180) - Real.sin (24 * Real.pi / 180) * Real.cos (54 * Real.pi / 180) = 1 / 2) := by
  sorry

end trigonometric_identity_cos24_cos36_sub_sin24_cos54_l97_97835


namespace triangular_number_30_eq_465_perimeter_dots_30_eq_88_l97_97518

-- Definition of the 30th triangular number
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition of the perimeter dots for the triangular number
def perimeter_dots (n : ℕ) : ℕ := n + 2 * (n - 1)

-- Theorem to prove the 30th triangular number is 465
theorem triangular_number_30_eq_465 : triangular_number 30 = 465 := by
  sorry

-- Theorem to prove the perimeter dots for the 30th triangular number is 88
theorem perimeter_dots_30_eq_88 : perimeter_dots 30 = 88 := by
  sorry

end triangular_number_30_eq_465_perimeter_dots_30_eq_88_l97_97518


namespace max_m_for_inequality_min_4a2_9b2_c2_l97_97878

theorem max_m_for_inequality (m : ℝ) : (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 := 
sorry

theorem min_4a2_9b2_c2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (4 * a^2 + 9 * b^2 + c^2) = 36 / 49 ∧ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49 :=
sorry

end max_m_for_inequality_min_4a2_9b2_c2_l97_97878


namespace remainder_when_nm_div_61_l97_97949

theorem remainder_when_nm_div_61 (n m : ℕ) (k j : ℤ):
  n = 157 * k + 53 → m = 193 * j + 76 → (n + m) % 61 = 7 := by
  intros h1 h2
  sorry

end remainder_when_nm_div_61_l97_97949


namespace percentage_loss_is_correct_l97_97077

-- Define the cost price and selling price
def cost_price : ℕ := 2000
def selling_price : ℕ := 1800

-- Define the calculation of loss and percentage loss
def loss (cp sp : ℕ) := cp - sp
def percentage_loss (loss cp : ℕ) := (loss * 100) / cp

-- The goal is to prove that the percentage loss is 10%
theorem percentage_loss_is_correct : percentage_loss (loss cost_price selling_price) cost_price = 10 := by
  sorry

end percentage_loss_is_correct_l97_97077


namespace sum_last_two_digits_is_correct_l97_97845

def fibs : List Nat := [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

def factorial_last_two_digits (n : Nat) : Nat :=
  (Nat.factorial n) % 100

def modified_fib_factorial_series : List Nat :=
  fibs.map (λ k => (factorial_last_two_digits k + 2) % 100)

def sum_last_two_digits : Nat :=
  (modified_fib_factorial_series.sum) % 100

theorem sum_last_two_digits_is_correct :
  sum_last_two_digits = 14 :=
sorry

end sum_last_two_digits_is_correct_l97_97845


namespace cab_driver_income_l97_97901

theorem cab_driver_income (incomes : Fin 5 → ℝ)
  (h1 : incomes 0 = 250)
  (h2 : incomes 1 = 400)
  (h3 : incomes 2 = 750)
  (h4 : incomes 3 = 400)
  (avg_income : (incomes 0 + incomes 1 + incomes 2 + incomes 3 + incomes 4) / 5 = 460) : 
  incomes 4 = 500 :=
sorry

end cab_driver_income_l97_97901


namespace problem1_proof_problem2_proof_l97_97299

noncomputable def problem1_statement : Prop :=
  (2 * Real.sin (Real.pi / 6) - Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4) = 1 / 2)

noncomputable def problem2_statement : Prop :=
  ((-1)^2023 + 2 * Real.sin (Real.pi / 4) - Real.cos (Real.pi / 6) + Real.sin (Real.pi / 3) + Real.tan (Real.pi / 3)^2 = 2 + Real.sqrt 2)

theorem problem1_proof : problem1_statement :=
by
  sorry

theorem problem2_proof : problem2_statement :=
by
  sorry

end problem1_proof_problem2_proof_l97_97299


namespace exponent_calculation_l97_97504

theorem exponent_calculation : (-1 : ℤ) ^ 53 + (2 : ℤ) ^ (5 ^ 3 - 2 ^ 3 + 3 ^ 2) = 2 ^ 126 - 1 :=
by 
  sorry

end exponent_calculation_l97_97504


namespace angle_BAC_in_isosceles_triangle_l97_97197

theorem angle_BAC_in_isosceles_triangle
  (A B C D : Type)
  (AB AC : ℝ)
  (BD DC : ℝ)
  (angle_BDA : ℝ)
  (isosceles_triangle : AB = AC)
  (midpoint_D : BD = DC)
  (external_angle_D : angle_BDA = 120) :
  ∃ (angle_BAC : ℝ), angle_BAC = 60 :=
by
  sorry

end angle_BAC_in_isosceles_triangle_l97_97197


namespace find_smallest_x_l97_97019

theorem find_smallest_x :
  ∃ (x : ℕ), x > 1 ∧ (x^2 % 1000 = x % 1000) ∧ x = 376 := by
  sorry

end find_smallest_x_l97_97019


namespace total_clouds_l97_97796

theorem total_clouds (C B : ℕ) (h1 : C = 6) (h2 : B = 3 * C) : C + B = 24 := by
  sorry

end total_clouds_l97_97796


namespace inequality_abc_l97_97138

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) >= 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_abc_l97_97138


namespace unanswered_questions_count_l97_97814

-- Define the variables: c (correct), w (wrong), u (unanswered)
variables (c w u : ℕ)

-- Define the conditions based on the problem statement.
def total_questions (c w u : ℕ) : Prop := c + w + u = 35
def new_system_score (c u : ℕ) : Prop := 6 * c + 3 * u = 120
def old_system_score (c w : ℕ) : Prop := 5 * c - 2 * w = 55

-- Prove that the number of unanswered questions, u, equals 10
theorem unanswered_questions_count (c w u : ℕ) 
    (h1 : total_questions c w u)
    (h2 : new_system_score c u)
    (h3 : old_system_score c w) : u = 10 :=
by
  sorry

end unanswered_questions_count_l97_97814


namespace polygon_sides_l97_97017

-- Given conditions
def is_interior_angle (angle : ℝ) : Prop :=
  angle = 150

-- The theorem to prove the number of sides
theorem polygon_sides (h : is_interior_angle 150) : ∃ n : ℕ, n = 12 :=
  sorry

end polygon_sides_l97_97017


namespace find_y_l97_97891

theorem find_y (y : ℕ) (h1 : 27 = 3^3) (h2 : 3^9 = 27^y) : y = 3 := 
by 
  sorry

end find_y_l97_97891


namespace factorable_quadratic_l97_97554

theorem factorable_quadratic (b : Int) : 
  (∃ m n p q : Int, 35 * m * p = 35 ∧ m * q + n * p = b ∧ n * q = 35) ↔ (∃ k : Int, b = 2 * k) :=
sorry

end factorable_quadratic_l97_97554


namespace solve_for_x_l97_97879

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l97_97879


namespace expected_profit_may_is_3456_l97_97006

-- Given conditions as definitions
def february_profit : ℝ := 2000
def april_profit : ℝ := 2880
def growth_rate (x : ℝ) : Prop := (2000 * (1 + x)^2 = 2880)

-- The expected profit in May
def expected_may_profit (x : ℝ) : ℝ := april_profit * (1 + x)

-- The theorem to be proved based on the given conditions
theorem expected_profit_may_is_3456 (x : ℝ) (h : growth_rate x) (h_pos : x = (1:ℝ)/5) : 
    expected_may_profit x = 3456 :=
by sorry

end expected_profit_may_is_3456_l97_97006


namespace charity_event_revenue_l97_97630

theorem charity_event_revenue :
  ∃ (f t p : ℕ), f + t = 190 ∧ f * p + t * (p / 3) = 2871 ∧ f * p = 1900 :=
by
  sorry

end charity_event_revenue_l97_97630


namespace sum_abcd_value_l97_97551

theorem sum_abcd_value (a b c d : ℚ) :
  (2 * a + 3 = 2 * b + 5) ∧ 
  (2 * b + 5 = 2 * c + 7) ∧ 
  (2 * c + 7 = 2 * d + 9) ∧ 
  (2 * d + 9 = 2 * (a + b + c + d) + 13) → 
  a + b + c + d = -14 / 3 := 
by
  sorry

end sum_abcd_value_l97_97551


namespace range_of_a_l97_97827

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l97_97827


namespace max_ratio_MO_MF_on_parabola_l97_97757

theorem max_ratio_MO_MF_on_parabola (F M : ℝ × ℝ) : 
  let O := (0, 0)
  let focus := (1 / 2, 0)
  ∀ (M : ℝ × ℝ), (M.snd ^ 2 = 2 * M.fst) →
  F = focus →
  (∃ m > 0, M.fst = m ∧ M.snd ^ 2 = 2 * m) →
  (∃ t, t = m - (1 / 4)) →
  ∃ value, value = (2 * Real.sqrt 3) / 3 ∧
  ∃ rat, rat = dist M O / dist M F ∧
  rat = value := 
by
  admit

end max_ratio_MO_MF_on_parabola_l97_97757


namespace proportional_function_y_decreases_l97_97658

theorem proportional_function_y_decreases (k : ℝ) (h₀ : k ≠ 0) (h₁ : (4 : ℝ) * k = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ :=
by 
  sorry

end proportional_function_y_decreases_l97_97658


namespace percentage_reduction_l97_97706

variable (C S newS newC : ℝ)
variable (P : ℝ)
variable (hC : C = 50)
variable (hS : S = 1.25 * C)
variable (hNewS : newS = S - 10.50)
variable (hGain30 : newS = 1.30 * newC)
variable (hNewC : newC = C - P * C)

theorem percentage_reduction (C S newS newC : ℝ) (hC : C = 50) 
  (hS : S = 1.25 * C) (hNewS : newS = S - 10.50) 
  (hGain30 : newS = 1.30 * newC) 
  (hNewC : newC = C - P * C) : 
  P = 0.20 :=
by
  sorry

end percentage_reduction_l97_97706


namespace outfit_count_l97_97652

theorem outfit_count (shirts pants ties belts : ℕ) (h_shirts : shirts = 8) (h_pants : pants = 5) (h_ties : ties = 4) (h_belts : belts = 2) :
  shirts * pants * (ties + 1) * (belts + 1) = 600 := by
  sorry

end outfit_count_l97_97652


namespace range_of_a_l97_97718

open Real

-- Definitions of the propositions p and q
def p (a : ℝ) : Prop := (2 - a > 0) ∧ (a + 1 > 0)

def discriminant (a : ℝ) : ℝ := 16 + 4 * a

def q (a : ℝ) : Prop := discriminant a ≥ 0

/--
Given propositions p and q defined above,
prove that the range of real number values for a 
such that ¬p ∧ q is true is
- 4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2
--/
theorem range_of_a (a : ℝ) : (¬ p a ∧ q a) → (-4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l97_97718


namespace min_packages_l97_97657

theorem min_packages (p : ℕ) (N : ℕ) :
  (N = 19 * p) →
  (N % 7 = 4) →
  (N % 11 = 1) →
  p = 40 :=
by
  sorry

end min_packages_l97_97657


namespace vampire_daily_needs_l97_97720

theorem vampire_daily_needs :
  (7 * 8) / 2 / 7 = 4 :=
by
  sorry

end vampire_daily_needs_l97_97720


namespace negation_of_proposition_l97_97515

theorem negation_of_proposition :
  (¬ (∃ x : ℝ, x < 0 ∧ x^2 > 0)) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) :=
  sorry

end negation_of_proposition_l97_97515


namespace scientific_notation_of_600_million_l97_97492

theorem scientific_notation_of_600_million : 600000000 = 6 * 10^7 := 
sorry

end scientific_notation_of_600_million_l97_97492


namespace evaluate_g_at_neg2_l97_97447

-- Definition of the polynomial g
def g (x : ℝ) : ℝ := 3 * x^5 - 20 * x^4 + 40 * x^3 - 25 * x^2 - 75 * x + 90

-- Statement to prove using the condition
theorem evaluate_g_at_neg2 : g (-2) = -596 := 
by 
   sorry

end evaluate_g_at_neg2_l97_97447


namespace find_common_ratio_l97_97220

theorem find_common_ratio (a_1 q : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS1 : S 1 = a_1)
  (hS2 : S 2 = a_1 * (1 + q))
  (hS3 : S 3 = a_1 * (1 + q + q^2))
  (ha2 : a 2 = a_1 * q)
  (ha3 : a 3 = a_1 * q^2)
  (hcond : 2 * (S 1 + 2 * a 2) = S 3 + a 3 + S 2 + a 2) :
  q = -1/2 :=
by
  sorry

end find_common_ratio_l97_97220


namespace girl_speed_l97_97565

theorem girl_speed (distance time : ℝ) (h_distance : distance = 96) (h_time : time = 16) : distance / time = 6 :=
by
  sorry

end girl_speed_l97_97565


namespace product_of_w_and_z_l97_97000

variable (EF FG GH HE : ℕ)
variable (w z : ℕ)

-- Conditions from the problem
def parallelogram_conditions : Prop :=
  EF = 42 ∧ FG = 4 * z^3 ∧ GH = 3 * w + 6 ∧ HE = 32 ∧ EF = GH ∧ FG = HE

-- The proof problem proving the requested product given the conditions
theorem product_of_w_and_z (h : parallelogram_conditions EF FG GH HE w z) : (w * z) = 24 :=
by
  sorry

end product_of_w_and_z_l97_97000


namespace cone_lateral_surface_area_l97_97186

theorem cone_lateral_surface_area (a : ℝ) (π : ℝ) (sqrt_3 : ℝ) 
  (h₁ : 0 < a)
  (h_area : (1 / 2) * a^2 * (sqrt_3 / 2) = sqrt_3) :
  π * 1 * 2 = 2 * π :=
by
  sorry

end cone_lateral_surface_area_l97_97186


namespace right_triangle_leg_square_l97_97508

theorem right_triangle_leg_square (a c b : ℕ) (h1 : c = a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = c + a :=
by
  sorry

end right_triangle_leg_square_l97_97508


namespace smallest_three_digit_solution_l97_97869

theorem smallest_three_digit_solution :
  ∃ n : ℕ, 70 * n ≡ 210 [MOD 350] ∧ 100 ≤ n ∧ n = 103 :=
by
  sorry

end smallest_three_digit_solution_l97_97869


namespace person_b_lap_time_l97_97740

noncomputable def lap_time_b (a_lap_time : ℕ) (meet_time : ℕ) : ℕ :=
  let combined_speed := 1 / meet_time
  let a_speed := 1 / a_lap_time
  let b_speed := combined_speed - a_speed
  1 / b_speed

theorem person_b_lap_time 
  (a_lap_time : ℕ) 
  (meet_time : ℕ) 
  (h1 : a_lap_time = 80) 
  (h2 : meet_time = 30) : 
  lap_time_b a_lap_time meet_time = 48 := 
by 
  rw [lap_time_b, h1, h2]
  -- Provided steps to solve the proof, skipped here only for statement
  sorry

end person_b_lap_time_l97_97740


namespace reflected_ray_eq_l97_97935

theorem reflected_ray_eq:
  ∀ (x y : ℝ), 
    (3 * x + 4 * y - 18 = 0) ∧ (3 * x + 2 * y - 12 = 0) →
    63 * x + 16 * y - 174 = 0 :=
by
  intro x y
  intro h
  sorry

end reflected_ray_eq_l97_97935


namespace bananas_to_oranges_l97_97315

theorem bananas_to_oranges (B A O : ℕ) 
    (h1 : 4 * B = 3 * A) 
    (h2 : 7 * A = 5 * O) : 
    28 * B = 15 * O :=
by
  sorry

end bananas_to_oranges_l97_97315


namespace tower_no_knights_l97_97809

-- Define the problem conditions in Lean

variable {T : Type} -- Type for towers
variable {K : Type} -- Type for knights

variable (towers : Fin 9 → T)
variable (knights : Fin 18 → K)

-- Movement of knights: each knight moves to a neighboring tower every hour (either clockwise or counterclockwise)
variable (moves : K → (T → T))

-- Each knight stands watch at each tower exactly once over the course of the night
variable (stands_watch : ∀ k : K, ∀ t : T, ∃ hour : Fin 9, moves k t = towers hour)

-- Condition: at one time (say hour 1), each tower had at least two knights on watch
variable (time1 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 1
variable (cond1 : ∀ i : Fin 9, 2 ≤ time1 1 i)

-- Condition: at another time (say hour 2), exactly five towers each had exactly one knight on watch
variable (time2 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 2
variable (cond2 : ∃ seq : Fin 5 → Fin 9, (∀ i : Fin 5, time2 2 (seq i) = 1) ∧ ∀ j : Fin 4, i ≠ j → 1 ≠ seq j)

-- Prove: there exists a time when one of the towers had no knights at all
theorem tower_no_knights : ∃ hour : Fin 9, ∃ i : Fin 9, moves (knights i) (towers hour) = towers hour ∧ (∀ knight : K, moves knight (towers hour) ≠ towers hour) :=
sorry

end tower_no_knights_l97_97809


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l97_97286

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l97_97286


namespace cost_of_adult_ticket_l97_97270

theorem cost_of_adult_ticket (A : ℝ) (H1 : ∀ (cost_child : ℝ), cost_child = 7) 
                             (H2 : ∀ (num_adults : ℝ), num_adults = 2) 
                             (H3 : ∀ (num_children : ℝ), num_children = 2) 
                             (H4 : ∀ (total_cost : ℝ), total_cost = 58) :
    A = 22 :=
by
  -- You can assume variables for children's cost, number of adults, and number of children
  let cost_child := 7
  let num_adults := 2
  let num_children := 2
  let total_cost := 58
  
  -- Formalize the conditions given
  have H_children_cost : num_children * cost_child = 14 := by simp [cost_child, num_children]
  
  -- Establish the total cost equation
  have H_total_equation : num_adults * A + num_children * cost_child = total_cost := 
    by sorry  -- (Total_equation_proof)
  
  -- Solve for A
  sorry  -- Proof step

end cost_of_adult_ticket_l97_97270


namespace toaster_sales_promotion_l97_97352

theorem toaster_sales_promotion :
  ∀ (p : ℕ) (c₁ c₂ : ℕ) (k : ℕ), 
    (c₁ = 600 ∧ p = 15 ∧ k = p * c₁) ∧ 
    (c₂ = 450 ∧ (p * c₂ = k) ) ∧ 
    (p' = p * 11 / 10) →
    p' = 22 :=
by 
  sorry

end toaster_sales_promotion_l97_97352


namespace Q_at_1_eq_1_l97_97906

noncomputable def Q (x : ℚ) : ℚ := x^4 - 16*x^2 + 16

theorem Q_at_1_eq_1 : Q 1 = 1 := by
  sorry

end Q_at_1_eq_1_l97_97906


namespace roots_quadratic_l97_97557

theorem roots_quadratic (d e : ℝ) (h1 : 3 * d ^ 2 + 5 * d - 2 = 0) (h2 : 3 * e ^ 2 + 5 * e - 2 = 0) :
  (d - 1) * (e - 1) = 2 :=
sorry

end roots_quadratic_l97_97557


namespace trader_sells_cloth_l97_97336

theorem trader_sells_cloth
  (total_SP : ℝ := 4950)
  (profit_per_meter : ℝ := 15)
  (cost_price_per_meter : ℝ := 51)
  (SP_per_meter : ℝ := cost_price_per_meter + profit_per_meter)
  (x : ℝ := total_SP / SP_per_meter) :
  x = 75 :=
by
  sorry

end trader_sells_cloth_l97_97336


namespace sixth_graders_more_than_seventh_l97_97569

def pencil_cost : ℕ := 13
def eighth_graders_total : ℕ := 208
def seventh_graders_total : ℕ := 181
def sixth_graders_total : ℕ := 234

-- Number of students in each grade who bought a pencil
def seventh_graders_count := seventh_graders_total / pencil_cost
def sixth_graders_count := sixth_graders_total / pencil_cost

-- The difference in the number of sixth graders than seventh graders who bought a pencil
theorem sixth_graders_more_than_seventh : sixth_graders_count - seventh_graders_count = 4 :=
by sorry

end sixth_graders_more_than_seventh_l97_97569


namespace subtracted_number_from_32_l97_97943

theorem subtracted_number_from_32 (x : ℕ) (h : 32 - x = 23) : x = 9 := 
by 
  sorry

end subtracted_number_from_32_l97_97943


namespace tan_pi_minus_alpha_l97_97272

theorem tan_pi_minus_alpha (α : ℝ) (h : 3 * Real.sin α = Real.cos α) : Real.tan (π - α) = -1 / 3 :=
by
  sorry

end tan_pi_minus_alpha_l97_97272


namespace C_share_correct_l97_97327

noncomputable def C_share (B_invest: ℝ) (total_profit: ℝ) : ℝ :=
  let A_invest := 3 * B_invest
  let C_invest := (3 * B_invest) * (3/2)
  let total_invest := (3 * B_invest + B_invest + C_invest)
  (C_invest / total_invest) * total_profit

theorem C_share_correct (B_invest total_profit: ℝ) 
  (hA : ∀ x: ℝ, A_invest = 3 * x)
  (hC : ∀ x: ℝ, C_invest = (3 * x) * (3/2)) :
  C_share B_invest 12375 = 6551.47 :=
by
  sorry

end C_share_correct_l97_97327


namespace total_combined_rainfall_l97_97395

theorem total_combined_rainfall :
  let monday_hours := 5
  let monday_rate := 1
  let tuesday_hours := 3
  let tuesday_rate := 1.5
  let wednesday_hours := 4
  let wednesday_rate := 2 * monday_rate
  let thursday_hours := 6
  let thursday_rate := tuesday_rate / 2
  let friday_hours := 2
  let friday_rate := 1.5 * wednesday_rate
  let monday_rain := monday_hours * monday_rate
  let tuesday_rain := tuesday_hours * tuesday_rate
  let wednesday_rain := wednesday_hours * wednesday_rate
  let thursday_rain := thursday_hours * thursday_rate
  let friday_rain := friday_hours * friday_rate
  monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = 28 := by
  sorry

end total_combined_rainfall_l97_97395


namespace determine_phi_l97_97156

variable (ω : ℝ) (varphi : ℝ)

noncomputable def f (ω varphi x: ℝ) : ℝ := Real.sin (ω * x + varphi)

theorem determine_phi
  (hω : ω > 0)
  (hvarphi : 0 < varphi ∧ varphi < π)
  (hx1 : f ω varphi (π/4) = Real.sin (ω * (π / 4) + varphi))
  (hx2 : f ω varphi (5 * π / 4) = Real.sin (ω * (5 * π / 4) + varphi))
  (hsym : ∀ x, f ω varphi x = f ω varphi (π - x))
  : varphi = π / 4 :=
sorry

end determine_phi_l97_97156


namespace disproves_proposition_l97_97601

theorem disproves_proposition (a b : ℤ) (h₁ : a = -4) (h₂ : b = 3) : (a^2 > b^2) ∧ ¬ (a > b) :=
by
  sorry

end disproves_proposition_l97_97601


namespace gcd_consecutive_term_max_l97_97206

def b (n : ℕ) : ℕ := n.factorial + 2^n + n 

theorem gcd_consecutive_term_max (n : ℕ) (hn : n ≥ 0) :
  ∃ m ≤ (n : ℕ), (m = 2) := sorry

end gcd_consecutive_term_max_l97_97206


namespace volume_at_target_temperature_l97_97273

-- Volume expansion relationship
def volume_change_per_degree_rise (ΔT V_real : ℝ) : Prop :=
  ΔT = 2 ∧ V_real = 3

-- Initial conditions
def initial_conditions (V_initial T_initial : ℝ) : Prop :=
  V_initial = 36 ∧ T_initial = 30

-- Target temperature
def target_temperature (T_target : ℝ) : Prop :=
  T_target = 20

-- Theorem stating the volume at the target temperature
theorem volume_at_target_temperature (ΔT V_real T_initial V_initial T_target V_target : ℝ) 
  (h_rel : volume_change_per_degree_rise ΔT V_real)
  (h_init : initial_conditions V_initial T_initial)
  (h_target : target_temperature T_target) :
  V_target = V_initial + V_real * ((T_target - T_initial) / ΔT) :=
by
  -- Insert proof here
  sorry

end volume_at_target_temperature_l97_97273


namespace sphere_volume_l97_97353

theorem sphere_volume (A : ℝ) (d : ℝ) (V : ℝ) : 
    (A = 2 * Real.pi) →  -- Cross-sectional area is 2π cm²
    (d = 1) →            -- Distance from center to cross-section is 1 cm
    (V = 4 * Real.sqrt 3 * Real.pi) :=  -- Volume of sphere is 4√3 π cm³
by 
  intros hA hd
  sorry

end sphere_volume_l97_97353


namespace youngest_child_age_l97_97558

theorem youngest_child_age (total_bill mother_cost twin_age_cost total_age : ℕ) (twin_age youngest_age : ℕ) 
  (h1 : total_bill = 1485) (h2 : mother_cost = 695) (h3 : twin_age_cost = 65) 
  (h4 : total_age = (total_bill - mother_cost) / twin_age_cost)
  (h5 : total_age = 2 * twin_age + youngest_age) :
  youngest_age = 2 :=
by
  -- sorry: Proof to be completed later
  sorry

end youngest_child_age_l97_97558


namespace product_of_three_greater_than_two_or_four_of_others_l97_97071

theorem product_of_three_greater_than_two_or_four_of_others 
  (x : Fin 10 → ℕ) 
  (h_unique : ∀ i j : Fin 10, i ≠ j → x i ≠ x j) 
  (h_positive : ∀ i : Fin 10, 0 < x i) : 
  ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ a b : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ b ≠ i ∧ b ≠ j ∧ b ≠ k → 
      x i * x j * x k > x a * x b) ∨ 
    (∀ a b c d : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ 
      b ≠ i ∧ b ≠ j ∧ b ≠ k ∧ 
      c ≠ i ∧ c ≠ j ∧ c ≠ k ∧ 
      d ≠ i ∧ d ≠ j ∧ d ≠ k → 
      x i * x j * x k > x a * x b * x c * x d) := sorry

end product_of_three_greater_than_two_or_four_of_others_l97_97071


namespace egg_production_l97_97011

theorem egg_production (n_chickens1 n_chickens2 n_eggs1 n_eggs2 n_days1 n_days2 : ℕ)
  (h1 : n_chickens1 = 6) (h2 : n_eggs1 = 30) (h3 : n_days1 = 5) (h4 : n_chickens2 = 10) (h5 : n_days2 = 8) :
  n_eggs2 = 80 :=
sorry

end egg_production_l97_97011


namespace quarters_spent_l97_97271

variable (q_initial q_left q_spent : ℕ)

theorem quarters_spent (h1 : q_initial = 11) (h2 : q_left = 7) : q_spent = q_initial - q_left ∧ q_spent = 4 :=
by
  sorry

end quarters_spent_l97_97271


namespace megatek_manufacturing_percentage_proof_l97_97452

def megatek_employee_percentage
  (total_degrees_in_circle : ℕ)
  (manufacturing_degrees : ℕ) : ℚ :=
  (manufacturing_degrees / total_degrees_in_circle : ℚ) * 100

theorem megatek_manufacturing_percentage_proof (h1 : total_degrees_in_circle = 360)
  (h2 : manufacturing_degrees = 54) :
  megatek_employee_percentage total_degrees_in_circle manufacturing_degrees = 15 := 
by
  sorry

end megatek_manufacturing_percentage_proof_l97_97452


namespace race_track_cost_l97_97341

def toy_car_cost : ℝ := 0.95
def num_toy_cars : ℕ := 4
def total_money : ℝ := 17.80
def money_left : ℝ := 8.00

theorem race_track_cost :
  total_money - num_toy_cars * toy_car_cost - money_left = 6.00 :=
by
  sorry

end race_track_cost_l97_97341


namespace no_solution_abs_eq_l97_97375

theorem no_solution_abs_eq : ∀ y : ℝ, |y - 2| ≠ |y - 1| + |y - 4| :=
by
  intros y
  sorry

end no_solution_abs_eq_l97_97375


namespace find_values_of_c_x1_x2_l97_97301

theorem find_values_of_c_x1_x2 (x₁ x₂ c : ℝ)
    (h1 : x₁ + x₂ = -2)
    (h2 : x₁ * x₂ = c)
    (h3 : x₁^2 + x₂^2 = c^2 - 2 * c) :
    c = -2 ∧ x₁ = -1 + Real.sqrt 3 ∧ x₂ = -1 - Real.sqrt 3 :=
by
  sorry

end find_values_of_c_x1_x2_l97_97301


namespace wheel_stop_probability_l97_97400

theorem wheel_stop_probability 
  (pD pE pG pF : ℚ) 
  (h1 : pD = 1 / 4) 
  (h2 : pE = 1 / 3) 
  (h3 : pG = 1 / 6) 
  (h4 : pD + pE + pG + pF = 1) : 
  pF = 1 / 4 := 
by 
  sorry

end wheel_stop_probability_l97_97400


namespace function_increasing_value_of_a_function_decreasing_value_of_a_l97_97370

-- Part 1: Prove that if \( f(x) = x^3 - ax - 1 \) is increasing on the interval \( (1, +\infty) \), then \( a \leq 3 \)
theorem function_increasing_value_of_a (a : ℝ) :
  (∀ x > 1, 3 * x^2 - a ≥ 0) → a ≤ 3 := by
  sorry

-- Part 2: Prove that if the decreasing interval of \( f(x) = x^3 - ax - 1 \) is \( (-1, 1) \), then \( a = 3 \)
theorem function_decreasing_value_of_a (a : ℝ) :
  (∀ x, -1 < x ∧ x < 1 → 3 * x^2 - a < 0) ∧ (3 * (-1)^2 - a = 0 ∧ 3 * (1)^2 - a = 0) → a = 3 := by
  sorry

end function_increasing_value_of_a_function_decreasing_value_of_a_l97_97370


namespace jen_hours_per_week_l97_97659

theorem jen_hours_per_week (B : ℕ) (h1 : ∀ t : ℕ, t * (B + 7) = 6 * B) : B + 7 = 21 := by
  sorry

end jen_hours_per_week_l97_97659


namespace choose_9_3_eq_84_l97_97913

theorem choose_9_3_eq_84 : Nat.choose 9 3 = 84 :=
by
  sorry

end choose_9_3_eq_84_l97_97913


namespace problem_solution_l97_97842

variable (x : ℝ)

-- Given condition
def condition1 : Prop := (7 / 8) * x = 28

-- The main statement to prove
theorem problem_solution (h : condition1 x) : (x + 16) * (5 / 16) = 15 := by
  sorry

end problem_solution_l97_97842


namespace find_common_ratio_l97_97738

variable {a : ℕ → ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

theorem find_common_ratio (h : geometric_sequence_q a q) : q = 2 :=
by
  sorry

end find_common_ratio_l97_97738


namespace simplify_expression_l97_97285

noncomputable def y := 
  Real.cos (2 * Real.pi / 15) + 
  Real.cos (4 * Real.pi / 15) + 
  Real.cos (8 * Real.pi / 15) + 
  Real.cos (14 * Real.pi / 15)

theorem simplify_expression : 
  y = (-1 + Real.sqrt 61) / 4 := 
sorry

end simplify_expression_l97_97285


namespace water_required_for_reaction_l97_97647

noncomputable def sodium_hydride_reaction (NaH H₂O NaOH H₂ : Type) : Nat :=
  1

theorem water_required_for_reaction :
  let NaH := 2
  let required_H₂O := 2 -- Derived from balanced chemical equation and given condition
  sodium_hydride_reaction Nat Nat Nat Nat = required_H₂O :=
by
  sorry

end water_required_for_reaction_l97_97647


namespace value_of_a0_plus_a8_l97_97215

/-- Theorem stating the value of a0 + a8 from the given polynomial equation -/
theorem value_of_a0_plus_a8 (a_0 a_8 : ℤ) :
  (∀ x : ℤ, (1 + x) ^ 10 = a_0 + a_1 * (1 - x) + a_2 * (1 - x) ^ 2 + 
              a_3 * (1 - x) ^ 3 + a_4 * (1 - x) ^ 4 + a_5 * (1 - x) ^ 5 +
              a_6 * (1 - x) ^ 6 + a_7 * (1 - x) ^ 7 + a_8 * (1 - x) ^ 8 + 
              a_9 * (1 - x) ^ 9 + a_10 * (1 - x) ^ 10) →
  a_0 + a_8 = 1204 :=
by
  sorry

end value_of_a0_plus_a8_l97_97215


namespace integer_solution_of_inequality_l97_97970

theorem integer_solution_of_inequality :
  ∀ (x : ℤ), 0 < (x - 1 : ℚ) * (x - 1) / (x + 1) ∧ (x - 1) * (x - 1) / (x + 1) < 1 →
  x > -1 ∧ x ≠ 1 ∧ x < 3 → 
  x = 2 :=
by
  sorry

end integer_solution_of_inequality_l97_97970


namespace ellipse_equation_l97_97274

open Real

theorem ellipse_equation (x y : ℝ) (h₁ : (- sqrt 15) = x) (h₂ : (5 / 2) = y)
  (h₃ : ∃ (a b : ℝ), (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + 5) 
  ∧ b^2 = 20 ∧ a^2 = 25) :
  (x^2 / 20 + y^2 / 25 = 1) :=
sorry

end ellipse_equation_l97_97274


namespace euler_distance_formula_l97_97996

theorem euler_distance_formula 
  (d R r : ℝ) 
  (h₁ : d = distance_between_centers_of_inscribed_and_circumscribed_circles_of_triangle)
  (h₂ : R = circumradius_of_triangle)
  (h₃ : r = inradius_of_triangle) : 
  d^2 = R^2 - 2 * R * r := 
sorry

end euler_distance_formula_l97_97996


namespace range_of_a_maximum_of_z_l97_97404

-- Problem 1
theorem range_of_a (a b : ℝ) (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) :
  -2 < a ∧ a < 1 :=
sorry

-- Problem 2
theorem maximum_of_z (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 9) :
  ∃ z, z = a * b^2 ∧ z ≤ 27 :=
sorry


end range_of_a_maximum_of_z_l97_97404


namespace cindy_correct_answer_l97_97377

-- Define the conditions given in the problem
def x : ℤ := 272 -- Cindy's miscalculated number

-- The outcome of Cindy's incorrect operation
def cindy_incorrect (x : ℤ) : Prop := (x - 7) = 53 * 5

-- The outcome of Cindy's correct operation
def cindy_correct (x : ℤ) : ℤ := (x - 5) / 7

-- The main theorem to prove
theorem cindy_correct_answer : cindy_incorrect x → cindy_correct x = 38 :=
by
  sorry

end cindy_correct_answer_l97_97377


namespace find_smallest_M_l97_97168

/-- 
Proof of the smallest real number M such that 
for all real numbers a, b, and c, the following inequality holds:
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)|
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2. 
-/
theorem find_smallest_M (a b c : ℝ) : 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end find_smallest_M_l97_97168


namespace describe_set_T_l97_97801

-- Define the conditions for the set of points T
def satisfies_conditions (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y < 7) ∨ (y - 3 = 4 ∧ x < 1)

-- Define the set T based on the conditions
def set_T := {p : ℝ × ℝ | satisfies_conditions p.1 p.2}

-- Statement to prove the geometric description of the set T
theorem describe_set_T :
  (∃ x y, satisfies_conditions x y) → ∃ p1 p2,
  (p1 = (1, t) ∧ t < 7 → satisfies_conditions 1 t) ∧
  (p2 = (t, 7) ∧ t < 1 → satisfies_conditions t 7) ∧
  (p1 ≠ p2) :=
sorry

end describe_set_T_l97_97801


namespace find_a5_l97_97744

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2 + 1) 
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h3 : S 1 = 2) :
  a 5 = 9 :=
sorry

end find_a5_l97_97744


namespace max_expression_value_l97_97608

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l97_97608


namespace sally_picked_peaches_l97_97799

theorem sally_picked_peaches (original_peaches total_peaches picked_peaches : ℕ)
  (h_orig : original_peaches = 13)
  (h_total : total_peaches = 55)
  (h_picked : picked_peaches = total_peaches - original_peaches) :
  picked_peaches = 42 :=
by
  sorry

end sally_picked_peaches_l97_97799


namespace range_of_b_l97_97153

def M := {p : ℝ × ℝ | p.1 ^ 2 + 2 * p.2 ^ 2 = 3}
def N (m b : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

theorem range_of_b (b : ℝ) : (∀ (m : ℝ), (∃ (p : ℝ × ℝ), p ∈ M ∧ p ∈ N m b)) ↔ 
  -Real.sqrt (6) / 2 ≤ b ∧ b ≤ Real.sqrt (6) / 2 :=
by
  sorry

end range_of_b_l97_97153


namespace line_passes_through_fixed_point_min_area_line_eq_l97_97789

section part_one

variable (m x y : ℝ)

def line_eq := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4

theorem line_passes_through_fixed_point :
  ∀ m, line_eq m 3 1 = 0 :=
sorry

end part_one

section part_two

variable (k x y : ℝ)

def line_eq_l1 (k : ℝ) := y = k * (x - 3) + 1

theorem min_area_line_eq :
  line_eq_l1 (-1/3) x y = (x + 3 * y - 6 = 0) :=
sorry

end part_two

end line_passes_through_fixed_point_min_area_line_eq_l97_97789


namespace sum_50th_set_l97_97863

-- Definition of the sequence repeating pattern
def repeating_sequence : List (List Nat) :=
  [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]

-- Definition to get the nth set in the repeating sequence
def nth_set (n : Nat) : List Nat :=
  repeating_sequence.get! ((n - 1) % 4)

-- Definition to sum the elements of a list
def sum_list (l : List Nat) : Nat :=
  l.sum

-- Proposition to prove that the sum of the 50th set is 4
theorem sum_50th_set : sum_list (nth_set 50) = 4 :=
by
  sorry

end sum_50th_set_l97_97863


namespace remainder_1234_mul_2047_mod_600_l97_97797

theorem remainder_1234_mul_2047_mod_600 : (1234 * 2047) % 600 = 198 := by
  sorry

end remainder_1234_mul_2047_mod_600_l97_97797


namespace total_legs_l97_97195

-- Define the number of each type of animal
def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goat : ℕ := 1

-- Define the number of legs per animal
def legs_per_animal : ℕ := 4

-- Define the total number of legs for each type of animal
def horse_legs : ℕ := num_horses * legs_per_animal
def dog_legs : ℕ := num_dogs * legs_per_animal
def cat_legs : ℕ := num_cats * legs_per_animal
def turtle_legs : ℕ := num_turtles * legs_per_animal
def goat_legs : ℕ := num_goat * legs_per_animal

-- Define the problem statement
theorem total_legs : horse_legs + dog_legs + cat_legs + turtle_legs + goat_legs = 72 := by
  -- Sum up all the leg counts
  sorry

end total_legs_l97_97195


namespace jenna_discount_l97_97898

def normal_price : ℝ := 50
def tickets_from_website : ℝ := 2 * normal_price
def scalper_initial_price_per_ticket : ℝ := 2.4 * normal_price
def scalper_total_initial : ℝ := 2 * scalper_initial_price_per_ticket
def friend_discounted_ticket : ℝ := 0.6 * normal_price
def total_price_five_tickets : ℝ := tickets_from_website + scalper_total_initial + friend_discounted_ticket
def amount_paid_by_friends : ℝ := 360

theorem jenna_discount : 
    total_price_five_tickets - amount_paid_by_friends = 10 :=
by
  -- The proof would go here, but we leave it as sorry for now.
  sorry

end jenna_discount_l97_97898


namespace sum_of_possible_values_l97_97748

noncomputable def solution : ℕ :=
  sorry

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| - 4 = 0) : solution = 10 :=
by
  sorry

end sum_of_possible_values_l97_97748


namespace find_a_extremum_and_min_value_find_max_k_l97_97624

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

end find_a_extremum_and_min_value_find_max_k_l97_97624


namespace count_integers_satisfying_sqrt_condition_l97_97391

theorem count_integers_satisfying_sqrt_condition :
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℕ), (3 < Real.sqrt x ∧ Real.sqrt x < 5) → (9 < x ∧ x < 25) :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l97_97391


namespace inequality_solution_l97_97687

theorem inequality_solution (x : ℝ) : 
  (∃ (y : ℝ), y = 1 / (3 ^ x) ∧ y * (y - 2) < 15) ↔ x > - (Real.log 5 / Real.log 3) :=
by 
    sorry

end inequality_solution_l97_97687


namespace intersection_A_complementB_l97_97484

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 5, 7}
def complementB := U \ B

theorem intersection_A_complementB :
  A ∩ complementB = {2, 4, 6} := 
by
  sorry

end intersection_A_complementB_l97_97484


namespace troy_buys_beef_l97_97158

theorem troy_buys_beef (B : ℕ) 
  (veg_pounds : ℕ := 6)
  (veg_cost_per_pound : ℕ := 2)
  (beef_cost_per_pound : ℕ := 3 * veg_cost_per_pound)
  (total_cost : ℕ := 36) :
  6 * veg_cost_per_pound + B * beef_cost_per_pound = total_cost → B = 4 :=
by
  sorry

end troy_buys_beef_l97_97158


namespace initially_calculated_average_l97_97938

theorem initially_calculated_average :
  ∀ (S : ℕ), (S / 10 = 18) →
  ((S - 46 + 26) / 10 = 16) :=
by
  sorry

end initially_calculated_average_l97_97938


namespace trigonometric_expression_value_l97_97988

theorem trigonometric_expression_value :
  4 * Real.cos (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) -
  Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 3 / 4 := sorry

end trigonometric_expression_value_l97_97988


namespace tiles_needed_l97_97411

theorem tiles_needed (S : ℕ) (n : ℕ) (k : ℕ) (N : ℕ) (H1 : S = 18144) 
  (H2 : n * k^2 = S) (H3 : n = (N * (N + 1)) / 2) : n = 2016 := 
sorry

end tiles_needed_l97_97411


namespace smaug_silver_coins_l97_97619

theorem smaug_silver_coins :
  ∀ (num_gold num_copper num_silver : ℕ)
  (value_per_silver value_per_gold conversion_factor value_total : ℕ),
  num_gold = 100 →
  num_copper = 33 →
  value_per_silver = 8 →
  value_per_gold = 3 →
  conversion_factor = value_per_gold * value_per_silver →
  value_total = 2913 →
  (num_gold * conversion_factor + num_silver * value_per_silver + num_copper = value_total) →
  num_silver = 60 :=
by
  intros num_gold num_copper num_silver value_per_silver value_per_gold conversion_factor value_total
  intros h1 h2 h3 h4 h5 h6 h_eq
  sorry

end smaug_silver_coins_l97_97619


namespace haley_collected_cans_l97_97833

theorem haley_collected_cans (C : ℕ) (h : C - 7 = 2) : C = 9 :=
by {
  sorry
}

end haley_collected_cans_l97_97833


namespace ratio_w_to_y_l97_97139

variables {w x y z : ℝ}

theorem ratio_w_to_y
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 9) :
  w / y = 8 :=
by
  sorry

end ratio_w_to_y_l97_97139


namespace concatenated_number_divisible_by_37_l97_97975

theorem concatenated_number_divisible_by_37
  (a b : ℕ) (ha : 100 ≤ a ∧ a ≤ 999) (hb : 100 ≤ b ∧ b ≤ 999)
  (h₁ : a % 37 ≠ 0) (h₂ : b % 37 ≠ 0) (h₃ : (a + b) % 37 = 0) :
  (1000 * a + b) % 37 = 0 :=
sorry

end concatenated_number_divisible_by_37_l97_97975


namespace perpendicular_case_parallel_case_l97_97585

variable (a b : ℝ)

-- Define the lines
def line1 (a b x y : ℝ) := a * x - b * y + 4 = 0
def line2 (a b x y : ℝ) := (a - 1) * x + y + b = 0

-- Define perpendicular condition
def perpendicular (a b : ℝ) := a * (a - 1) - b = 0

-- Define point condition
def passes_through (a b : ℝ) := -3 * a + b + 4 = 0

-- Define parallel condition
def parallel (a b : ℝ) := a * (a - 1) + b = 0

-- Define intercepts equal condition
def intercepts_equal (a b : ℝ) := b = -a

theorem perpendicular_case
    (h1 : perpendicular a b)
    (h2 : passes_through a b) :
    a = 2 ∧ b = 2 :=
sorry

theorem parallel_case
    (h1 : parallel a b)
    (h2 : intercepts_equal a b) :
    a = 2 ∧ b = -2 :=
sorry

end perpendicular_case_parallel_case_l97_97585


namespace large_rectangle_perimeter_correct_l97_97146

def perimeter_of_square (p : ℕ) : ℕ :=
  p / 4

def perimeter_of_rectangle (p : ℕ) (l : ℕ) : ℕ :=
  (p - 2 * l) / 2

def perimeter_of_large_rectangle (side_length_of_square side_length_of_rectangle : ℕ) : ℕ :=
  let height := side_length_of_square + 2 * side_length_of_rectangle
  let width := 3 * side_length_of_square
  2 * (height + width)

theorem large_rectangle_perimeter_correct :
  let side_length_of_square := perimeter_of_square 24
  let side_length_of_rectangle := perimeter_of_rectangle 16 side_length_of_square
  perimeter_of_large_rectangle side_length_of_square side_length_of_rectangle = 52 :=
by
  sorry

end large_rectangle_perimeter_correct_l97_97146


namespace positive_number_condition_l97_97371

theorem positive_number_condition (y : ℝ) (h: 0.04 * y = 16): y = 400 := 
by sorry

end positive_number_condition_l97_97371


namespace sqrt_pow_mul_l97_97241

theorem sqrt_pow_mul (a b : ℝ) : (a = 3) → (b = 5) → (Real.sqrt (a^2 * b^6) = 375) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end sqrt_pow_mul_l97_97241


namespace perpendicular_lines_slope_l97_97004

theorem perpendicular_lines_slope (a : ℝ) :
  (∀ x1 y1 x2 y2: ℝ, y1 = a * x1 - 2 ∧ y2 = x2 + 1 → (a * 1) = -1) → a = -1 :=
by
  sorry

end perpendicular_lines_slope_l97_97004


namespace bus_ride_difference_l97_97477

theorem bus_ride_difference :
  ∀ (Oscar_bus Charlie_bus : ℝ),
  Oscar_bus = 0.75 → Charlie_bus = 0.25 → Oscar_bus - Charlie_bus = 0.50 :=
by
  intros Oscar_bus Charlie_bus hOscar hCharlie
  rw [hOscar, hCharlie]
  norm_num

end bus_ride_difference_l97_97477


namespace hyperbola_asymptote_passing_through_point_l97_97251

theorem hyperbola_asymptote_passing_through_point (a : ℝ) (h_pos : a > 0) :
  (∃ m : ℝ, ∃ b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ (x, y) = (2, 1) ∧ m = 2 / a) → a = 4 :=
by
  sorry

end hyperbola_asymptote_passing_through_point_l97_97251


namespace conditional_probability_B_given_A_l97_97530

/-
Given a box containing 6 balls: 2 red, 2 yellow, and 2 blue.
One ball is drawn with replacement for 3 times.
Let event A be "the color of the ball drawn in the first draw is the same as the color of the ball drawn in the second draw".
Let event B be "the color of the balls drawn in all three draws is the same".
Prove that the conditional probability P(B|A) is 1/3.
-/
noncomputable def total_balls := 6
noncomputable def red_balls := 2
noncomputable def yellow_balls := 2
noncomputable def blue_balls := 2

noncomputable def event_A (n : ℕ) : ℕ := 
  3 * 2 * 2 * total_balls

noncomputable def event_AB (n : ℕ) : ℕ := 
  3 * 2 * 2 * 2

noncomputable def P_B_given_A : ℚ := 
  event_AB total_balls / event_A total_balls

theorem conditional_probability_B_given_A :
  P_B_given_A = 1 / 3 :=
by sorry

end conditional_probability_B_given_A_l97_97530


namespace correct_propositions_l97_97976

noncomputable def proposition1 : Prop :=
  (∀ x : ℝ, x^2 - 3 * x + 2 = 0 -> x = 1) ->
  (∀ x : ℝ, x ≠ 1 -> x^2 - 3 * x + 2 ≠ 0)

noncomputable def proposition2 : Prop :=
  (∀ p q : Prop, p ∨ q -> p ∧ q) ->
  (∀ p q : Prop, p ∧ q -> p ∨ q)

noncomputable def proposition3 : Prop :=
  (∀ p q : Prop, ¬(p ∧ q) -> ¬p ∧ ¬q)

noncomputable def proposition4 : Prop :=
  (∃ x : ℝ, x^2 + x + 1 < 0) ->
  (∀ x : ℝ, x^2 + x + 1 ≥ 0)

theorem correct_propositions :
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l97_97976


namespace flowers_per_basket_l97_97047

-- Definitions derived from the conditions
def initial_flowers : ℕ := 10
def grown_flowers : ℕ := 20
def dead_flowers : ℕ := 10
def baskets : ℕ := 5

-- Theorem stating the equivalence of the problem to its solution
theorem flowers_per_basket :
  (initial_flowers + grown_flowers - dead_flowers) / baskets = 4 :=
by
  sorry

end flowers_per_basket_l97_97047


namespace find_a_l97_97640

theorem find_a (a : ℝ) (h : (a + 3) = 0) : a = -3 :=
by sorry

end find_a_l97_97640


namespace nested_series_sum_l97_97176

theorem nested_series_sum : 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2))))) = 126 :=
by
  sorry

end nested_series_sum_l97_97176


namespace laps_run_l97_97882

theorem laps_run (x : ℕ) (total_distance required_distance lap_length extra_laps : ℕ) (h1 : total_distance = 2400) (h2 : lap_length = 150) (h3 : extra_laps = 4) (h4 : total_distance = lap_length * (x + extra_laps)) : x = 12 :=
by {
  sorry
}

end laps_run_l97_97882


namespace sum_of_roots_quadratic_eq_l97_97482

variable (h : ℝ)
def quadratic_eq_roots (x : ℝ) : Prop := 6 * x^2 - 5 * h * x - 4 * h = 0

theorem sum_of_roots_quadratic_eq (x1 x2 : ℝ) (h : ℝ) 
  (h_roots : quadratic_eq_roots h x1 ∧ quadratic_eq_roots h x2) 
  (h_distinct : x1 ≠ x2) :
  x1 + x2 = 5 * h / 6 := by
sorry

end sum_of_roots_quadratic_eq_l97_97482


namespace total_cards_across_decks_l97_97662

-- Conditions
def DeckA_cards : ℕ := 52
def DeckB_cards : ℕ := 40
def DeckC_cards : ℕ := 50
def DeckD_cards : ℕ := 48

-- Question as a statement
theorem total_cards_across_decks : (DeckA_cards + DeckB_cards + DeckC_cards + DeckD_cards = 190) := by
  sorry

end total_cards_across_decks_l97_97662


namespace root_value_cond_l97_97030

theorem root_value_cond (p q : ℝ) (h₁ : ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = q) (h₂ : q ≠ 0) : p + q = -1 := 
sorry

end root_value_cond_l97_97030


namespace range_of_a_minimize_S_l97_97850

open Real

-- Problem 1: Prove the range of a 
theorem range_of_a (a : ℝ) : (∃ x ≠ 0, x^3 - 3*x^2 + (2 - a)*x = 0) ↔ a > -1 / 4 := sorry

-- Problem 2: Prove the minimizing value of a for the area function S(a)
noncomputable def S (a : ℝ) : ℝ := 
  let α := sorry -- α is the root depending on a (to be determined from the context)
  let β := sorry -- β is the root depending on a (to be determined from the context)
  (1/4 * α^4 - α^3 + (1/2) * (2-a) * α^2) + (1/4 * β^4 - β^3 + (1/2) * (2-a) * β^2)

theorem minimize_S (a : ℝ) : a = 38 - 27 * sqrt 2 → S a = S (38 - 27 * sqrt 2) := sorry

end range_of_a_minimize_S_l97_97850


namespace age_difference_l97_97106

theorem age_difference (h b m : ℕ) (ratio : h = 4 * m ∧ b = 3 * m ∧ 4 * m + 3 * m + 7 * m = 126) : h - b = 9 :=
by
  -- proof will be filled here
  sorry

end age_difference_l97_97106


namespace problem_l97_97669

theorem problem (a b c d : ℝ) (h1 : a - b - c + d = 18) (h2 : a + b - c - d = 6) : (b - d) ^ 2 = 36 :=
by
  sorry

end problem_l97_97669


namespace rate_of_current_in_river_l97_97029

theorem rate_of_current_in_river (b c : ℝ) (h1 : 4 * (b + c) = 24) (h2 : 6 * (b - c) = 24) : c = 1 := by
  sorry

end rate_of_current_in_river_l97_97029


namespace mr_smith_children_l97_97448

noncomputable def gender_probability (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let equal_gender_ways := Nat.choose n (n / 2)
  let favourable_outcomes := total_outcomes - equal_gender_ways
  favourable_outcomes / total_outcomes

theorem mr_smith_children (n : ℕ) (h : n = 8) : 
  gender_probability n = 93 / 128 :=
by
  rw [h]
  sorry

end mr_smith_children_l97_97448


namespace price_per_glass_first_day_l97_97685

theorem price_per_glass_first_day (O P2 P1: ℝ) (H1 : O > 0) (H2 : P2 = 0.2) (H3 : 2 * O * P1 = 3 * O * P2) : P1 = 0.3 :=
by
  sorry

end price_per_glass_first_day_l97_97685


namespace sasha_fractions_l97_97999

theorem sasha_fractions (x y z t : ℕ) 
  (hx : x ≠ y) (hxy : x ≠ z) (hxz : x ≠ t)
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) :
  ∃ (q1 q2 : ℚ), (q1 ≠ q2) ∧ 
    (q1 = x / y ∨ q1 = x / z ∨ q1 = x / t ∨ q1 = y / x ∨ q1 = y / z ∨ q1 = y / t ∨ q1 = z / x ∨ q1 = z / y ∨ q1 = z / t ∨ q1 = t / x ∨ q1 = t / y ∨ q1 = t / z) ∧ 
    (q2 = x / y ∨ q2 = x / z ∨ q2 = x / t ∨ q2 = y / x ∨ q2 = y / z ∨ q2 = y / t ∨ q2 = z / x ∨ q2 = z / y ∨ q2 = z / t ∨ q2 = t / x ∨ q2 = t / y ∨ q2 = t / z) ∧ 
    |q1 - q2| ≤ 11 / 60 := by 
  sorry

end sasha_fractions_l97_97999


namespace only_function_B_has_inverse_l97_97696

-- Definitions based on the problem conditions
def function_A (x : ℝ) : ℝ := 3 - x^2 -- Parabola opening downwards with vertex at (0,3)
def function_B (x : ℝ) : ℝ := x -- Straight line with slope 1 passing through (0,0) and (1,1)
def function_C (x y : ℝ) : Prop := x^2 + y^2 = 4 -- Circle centered at (0,0) with radius 2

-- Theorem stating that only function B has an inverse
theorem only_function_B_has_inverse :
  (∀ y : ℝ, ∃! x : ℝ, function_B x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, function_A x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, ∃ y1 y2 : ℝ, function_C x y1 ∧ function_C x y2 ∧ y1 ≠ y2) :=
  by 
  sorry -- Proof not required

end only_function_B_has_inverse_l97_97696


namespace binomial_expansion_conditions_l97_97290

noncomputable def binomial_expansion (a b : ℝ) (x y : ℝ) (n : ℕ) : ℝ :=
(1 + a*x + b*y)^n

theorem binomial_expansion_conditions
  (a b : ℝ) (n : ℕ) 
  (h1 : (1 + b)^n = 243)
  (h2 : (1 + |a|)^n = 32) :
  a = 1 ∧ b = 2 ∧ n = 5 := by
  sorry

end binomial_expansion_conditions_l97_97290


namespace find_n_given_sum_l97_97989

noncomputable def geometric_sequence_general_term (n : ℕ) : ℝ :=
  if n ≥ 2 then 2^(2 * n - 3) else 0

def b_n (n : ℕ) : ℝ :=
  2 * n - 3

def sum_b_n (n : ℕ) : ℝ :=
  n^2 - 2 * n

theorem find_n_given_sum : ∃ n : ℕ, sum_b_n n = 360 :=
  by { use 20, sorry }

end find_n_given_sum_l97_97989


namespace scientific_notation_of_000000301_l97_97507

/--
Expressing a small number in scientific notation:
Prove that \(0.000000301\) can be written as \(3.01 \times 10^{-7}\).
-/
theorem scientific_notation_of_000000301 :
  0.000000301 = 3.01 * 10 ^ (-7) :=
sorry

end scientific_notation_of_000000301_l97_97507


namespace solve_for_a_l97_97808

theorem solve_for_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
sorry

end solve_for_a_l97_97808


namespace student_score_l97_97956

theorem student_score
    (total_questions : ℕ)
    (correct_responses : ℕ)
    (grading_method : ℕ → ℕ → ℕ)
    (h1 : total_questions = 100)
    (h2 : correct_responses = 92)
    (h3 : grading_method = λ correct incorrect => correct - 2 * incorrect) :
  grading_method correct_responses (total_questions - correct_responses) = 76 :=
by
  -- proof would be here, but is skipped
  sorry

end student_score_l97_97956


namespace find_k_check_divisibility_l97_97675

-- Define the polynomial f(x) as 2x^3 - 8x^2 + kx - 10
def f (x k : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + k * x - 10

-- Define the polynomial g(x) as 2x^3 - 8x^2 + 13x - 10 after finding k = 13
def g (x : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + 13 * x - 10

-- The first proof problem: Finding k
theorem find_k : (f 2 k = 0) → k = 13 := 
sorry

-- The second proof problem: Checking divisibility by 2x^2 - 1
theorem check_divisibility : ¬ (∃ h : ℝ → ℝ, g x = (2 * x^2 - 1) * h x) := 
sorry

end find_k_check_divisibility_l97_97675


namespace binary_to_decimal_and_octal_conversion_l97_97651

-- Definition of the binary number in question
def bin_num : ℕ := 0b1011

-- The expected decimal equivalent
def dec_num : ℕ := 11

-- The expected octal equivalent
def oct_num : ℤ := 0o13

-- Proof problem statement
theorem binary_to_decimal_and_octal_conversion :
  bin_num = dec_num ∧ dec_num = oct_num := 
by 
  sorry

end binary_to_decimal_and_octal_conversion_l97_97651


namespace linda_age_l97_97118

theorem linda_age 
  (J : ℕ)  -- Jane's current age
  (H1 : ∃ J, 2 * J + 3 = 13) -- Linda is 3 more than 2 times the age of Jane
  (H2 : (J + 5) + ((2 * J + 3) + 5) = 28) -- In 5 years, the sum of their ages will be 28
  : 2 * J + 3 = 13 :=
by {
  sorry
}

end linda_age_l97_97118


namespace minimum_value_of_function_l97_97494

theorem minimum_value_of_function (x : ℝ) (hx : x > 5 / 4) : 
  ∃ y, y = 4 * x + 1 / (4 * x - 5) ∧ y = 7 :=
sorry

end minimum_value_of_function_l97_97494


namespace transport_connectivity_l97_97208

-- Define the condition that any two cities are connected by either an air route or a canal.
-- We will formalize this with an inductive type to represent the transport means: AirRoute or Canal.
inductive TransportMeans
| AirRoute : TransportMeans
| Canal : TransportMeans

open TransportMeans

-- Represent cities as a type 'City'
universe u
variable (City : Type u)

-- Connect any two cities by a transport means
variable (connected : City → City → TransportMeans)

-- We want to prove that for any set of cities, 
-- there exists a means of transport such that starting from any city,
-- it is possible to reach any other city using only that means of transport.
theorem transport_connectivity (n : ℕ) (h2 : n ≥ 2) : 
  ∃ (T : TransportMeans), ∀ (c1 c2 : City), connected c1 c2 = T :=
by
  sorry

end transport_connectivity_l97_97208


namespace power_quotient_l97_97590

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l97_97590


namespace root_exists_in_interval_l97_97043

def f (x : ℝ) : ℝ := 2 * x + x - 2

theorem root_exists_in_interval :
  (∃ x ∈ (Set.Ioo 0 1), f x = 0) :=
by
  sorry

end root_exists_in_interval_l97_97043


namespace walking_rate_ratio_l97_97542

theorem walking_rate_ratio :
  let T := 16
  let T' := 12
  (T : ℚ) / (T' : ℚ) = (4 : ℚ) / (3 : ℚ) := 
by
  sorry

end walking_rate_ratio_l97_97542


namespace sum_of_sides_of_regular_pentagon_l97_97914

theorem sum_of_sides_of_regular_pentagon (s : ℝ) (n : ℕ)
    (h : s = 15) (hn : n = 5) : 5 * 15 = 75 :=
sorry

end sum_of_sides_of_regular_pentagon_l97_97914


namespace max_knights_l97_97095

/-- 
On an island with knights who always tell the truth and liars who always lie,
100 islanders seated around a round table where:
  - 50 of them say "both my neighbors are liars,"
  - The other 50 say "among my neighbors, there is exactly one liar."
Prove that the maximum number of knights at the table is 67.
-/
theorem max_knights (K L : ℕ) (h1 : K + L = 100) (h2 : ∃ k, k ≤ 25 ∧ K = 2 * k + (100 - 3 * k) / 2) : K = 67 :=
sorry

end max_knights_l97_97095


namespace cricket_players_count_l97_97719

theorem cricket_players_count (hockey: ℕ) (football: ℕ) (softball: ℕ) (total: ℕ) : 
  hockey = 15 ∧ football = 21 ∧ softball = 19 ∧ total = 77 → ∃ cricket, cricket = 22 := by
  sorry

end cricket_players_count_l97_97719


namespace initial_position_l97_97291

variable (x : Int)

theorem initial_position 
  (h: x - 5 + 4 + 2 - 3 + 1 = 6) : x = 7 := 
  by 
  sorry

end initial_position_l97_97291


namespace fraction_field_planted_l97_97112

-- Define the problem conditions
structure RightTriangle (leg1 leg2 hypotenuse : ℝ) : Prop :=
  (right_angle : ∃ (A B C : ℝ), A = 5 ∧ B = 12 ∧ hypotenuse = 13 ∧ A^2 + B^2 = hypotenuse^2)

structure SquarePatch (shortest_distance : ℝ) : Prop :=
  (distance_to_hypotenuse : shortest_distance = 3)

-- Define the statement
theorem fraction_field_planted (T : RightTriangle 5 12 13) (P : SquarePatch 3) : 
  ∃ (fraction : ℚ), fraction = 7 / 10 :=
by
  sorry

end fraction_field_planted_l97_97112


namespace opposite_neg_two_l97_97040

def opposite (x : Int) : Int := -x

theorem opposite_neg_two : opposite (-2) = 2 := by
  sorry

end opposite_neg_two_l97_97040


namespace value_of_a_b_c_l97_97256

noncomputable def absolute_value (x : ℤ) : ℤ := abs x

theorem value_of_a_b_c (a b c : ℤ)
  (ha : absolute_value a = 1)
  (hb : absolute_value b = 2)
  (hc : absolute_value c = 3)
  (h : a > b ∧ b > c) :
  a + b - c = 2 ∨ a + b - c = 0 :=
by
  sorry

end value_of_a_b_c_l97_97256


namespace value_of_g_neg2_l97_97098

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem value_of_g_neg2 : g (-2) = -3 := 
by sorry

end value_of_g_neg2_l97_97098


namespace pizza_order_cost_l97_97840

def base_cost_per_pizza : ℕ := 10
def cost_per_topping : ℕ := 1
def topping_count_pepperoni : ℕ := 1
def topping_count_sausage : ℕ := 1
def topping_count_black_olive_and_mushroom : ℕ := 2
def tip : ℕ := 5

theorem pizza_order_cost :
  3 * base_cost_per_pizza + (topping_count_pepperoni * cost_per_topping) + (topping_count_sausage * cost_per_topping) + (topping_count_black_olive_and_mushroom * cost_per_topping) + tip = 39 := by
  sorry

end pizza_order_cost_l97_97840


namespace sum_of_five_consecutive_odd_integers_l97_97866

theorem sum_of_five_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 8) = 156) :
  n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 390 :=
by
  sorry

end sum_of_five_consecutive_odd_integers_l97_97866


namespace exists_infinitely_many_gcd_condition_l97_97253

theorem exists_infinitely_many_gcd_condition (a : ℕ → ℕ) (h : ∀ n : ℕ, ∃ m : ℕ, a m = n) :
  ∃ᶠ i in at_top, Nat.gcd (a i) (a (i + 1)) ≤ (3 * i) / 4 :=
sorry

end exists_infinitely_many_gcd_condition_l97_97253


namespace road_construction_days_l97_97310

theorem road_construction_days
  (length_of_road : ℝ)
  (initial_men : ℕ)
  (completed_length : ℝ)
  (completed_days : ℕ)
  (extra_men : ℕ)
  (initial_days : ℕ)
  (remaining_length : ℝ)
  (remaining_days : ℕ)
  (total_men : ℕ) :
  length_of_road = 15 →
  initial_men = 30 →
  completed_length = 2.5 →
  completed_days = 100 →
  extra_men = 45 →
  initial_days = initial_days →
  remaining_length = length_of_road - completed_length →
  remaining_days = initial_days - completed_days →
  total_men = initial_men + extra_men →
  initial_days = 700 :=
by
  intros
  sorry

end road_construction_days_l97_97310


namespace remainder_when_x150_divided_by_x1_4_l97_97621

noncomputable def remainder_div_x150_by_x1_4 (x : ℝ) : ℝ :=
  x^150 % (x-1)^4

theorem remainder_when_x150_divided_by_x1_4 (x : ℝ) :
  remainder_div_x150_by_x1_4 x = -551300 * x^3 + 1665075 * x^2 - 1667400 * x + 562626 :=
by
  sorry

end remainder_when_x150_divided_by_x1_4_l97_97621


namespace speed_in_still_water_l97_97681

theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_up : upstream_speed = 26) (h_down : downstream_speed = 30) :
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end speed_in_still_water_l97_97681


namespace ribbon_initial_amount_l97_97303

theorem ribbon_initial_amount (x : ℕ) (gift_count : ℕ) (ribbon_per_gift : ℕ) (ribbon_left : ℕ)
  (H1 : ribbon_per_gift = 2) (H2 : gift_count = 6) (H3 : ribbon_left = 6)
  (H4 : x = gift_count * ribbon_per_gift + ribbon_left) : x = 18 :=
by
  rw [H1, H2, H3] at H4
  exact H4

end ribbon_initial_amount_l97_97303


namespace sequence_odd_for_all_n_greater_than_1_l97_97243

theorem sequence_odd_for_all_n_greater_than_1 (a : ℕ → ℤ) :
  (a 1 = 2) →
  (a 2 = 7) →
  (∀ n, 2 ≤ n → (-1/2 : ℚ) < (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ∧ (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ≤ (1/2 : ℚ)) →
  ∀ n, 1 < n → Odd (a n) := 
sorry

end sequence_odd_for_all_n_greater_than_1_l97_97243


namespace third_intermission_served_l97_97332

def total_served : ℚ :=  0.9166666666666666
def first_intermission : ℚ := 0.25
def second_intermission : ℚ := 0.4166666666666667

theorem third_intermission_served : first_intermission + second_intermission ≤ total_served →
  (total_served - (first_intermission + second_intermission)) = 0.25 :=
by
  sorry

end third_intermission_served_l97_97332


namespace equivalent_proof_problem_l97_97506

lemma condition_1 (a b : ℝ) (h : b > 0 ∧ 0 > a) : (1 / a) < (1 / b) :=
sorry

lemma condition_2 (a b : ℝ) (h : 0 > a ∧ a > b) : (1 / b) > (1 / a) :=
sorry

lemma condition_4 (a b : ℝ) (h : a > b ∧ b > 0) : (1 / b) > (1 / a) :=
sorry

theorem equivalent_proof_problem (a b : ℝ) :
  (b > 0 ∧ 0 > a → (1 / a) < (1 / b)) ∧
  (0 > a ∧ a > b → (1 / b) > (1 / a)) ∧
  (a > b ∧ b > 0 → (1 / b) > (1 / a)) :=
by {
  exact ⟨condition_1 a b, condition_2 a b, condition_4 a b⟩
}

end equivalent_proof_problem_l97_97506


namespace squirrels_cannot_divide_equally_l97_97837

theorem squirrels_cannot_divide_equally
    (n : ℕ) : ¬ (∃ k, 2022 + n * (n + 1) = 5 * k) :=
by
sorry

end squirrels_cannot_divide_equally_l97_97837


namespace magnitude_of_a_l97_97440

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (theta : ℝ)
variable (hθ : theta = π / 3)
variable (hb : ‖b‖ = 1)
variable (hab : ‖a + 2 • b‖ = 2 * sqrt 3)

theorem magnitude_of_a :
  ‖a‖ = 2 :=
by
  sorry

end magnitude_of_a_l97_97440


namespace hours_per_day_l97_97727

-- Define the parameters
def A1 := 57
def D1 := 12
def H2 := 6
def A2 := 30
def D2 := 19

-- Define the target Equation
theorem hours_per_day :
  A1 * D1 * H = A2 * D2 * H2 → H = 5 :=
by
  sorry

end hours_per_day_l97_97727


namespace sophomores_bought_15_more_markers_l97_97027

theorem sophomores_bought_15_more_markers (f_cost s_cost marker_cost : ℕ) (hf: f_cost = 267) (hs: s_cost = 312) (hm: marker_cost = 3) : 
  (s_cost / marker_cost) - (f_cost / marker_cost) = 15 :=
by
  sorry

end sophomores_bought_15_more_markers_l97_97027


namespace ratio_mercedes_jonathan_l97_97173

theorem ratio_mercedes_jonathan (M : ℝ) (J : ℝ) (D : ℝ) 
  (h1 : J = 7.5) 
  (h2 : D = M + 2) 
  (h3 : M + D = 32) : M / J = 2 :=
by
  sorry

end ratio_mercedes_jonathan_l97_97173


namespace calculate_abs_mul_l97_97497

theorem calculate_abs_mul : |(-3 : ℤ)| * 2 = 6 := 
by 
  -- |(-3)| equals 3 and 3 * 2 equals 6.
  -- The "sorry" is used to complete the statement without proof.
  sorry

end calculate_abs_mul_l97_97497


namespace length_of_train_75_l97_97402

variable (L : ℝ) -- Length of the train in meters

-- Condition 1: The train crosses a bridge of length 150 m in 7.5 seconds
def crosses_bridge (L: ℝ) : Prop := (L + 150) / 7.5 = L / 2.5

-- Condition 2: The train crosses a lamp post in 2.5 seconds
def crosses_lamp (L: ℝ) : Prop := L / 2.5 = L / 2.5

theorem length_of_train_75 (L : ℝ) (h1 : crosses_bridge L) (h2 : crosses_lamp L) : L = 75 := 
by 
  sorry

end length_of_train_75_l97_97402


namespace cube_lateral_surface_area_l97_97003

theorem cube_lateral_surface_area (V : ℝ) (h_V : V = 125) : 
  ∃ A : ℝ, A = 100 :=
by
  sorry

end cube_lateral_surface_area_l97_97003


namespace oranges_per_pack_correct_l97_97287

-- Definitions for the conditions.
def num_trees : Nat := 10
def oranges_per_tree_per_day : Nat := 12
def price_per_pack : Nat := 2
def total_earnings : Nat := 840
def weeks : Nat := 3
def days_per_week : Nat := 7

-- Theorem statement:
theorem oranges_per_pack_correct :
  let oranges_per_day := num_trees * oranges_per_tree_per_day
  let total_days := weeks * days_per_week
  let total_oranges := oranges_per_day * total_days
  let num_packs := total_earnings / price_per_pack
  total_oranges / num_packs = 6 :=
by
  sorry

end oranges_per_pack_correct_l97_97287


namespace smallest_prime_with_composite_reverse_l97_97825

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 ∧ m < n → n % m ≠ 0

def is_composite (n : Nat) : Prop :=
  n > 1 ∧ ∃ m : Nat, m > 1 ∧ m < n ∧ n % m = 0

def reverse_digits (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem smallest_prime_with_composite_reverse :
  ∃ (n : Nat), 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ (n / 10 = 3) ∧ is_composite (reverse_digits n) ∧
  (∀ m : Nat, 10 ≤ m ∧ m < n ∧ (m / 10 = 3) ∧ is_prime m → ¬is_composite (reverse_digits m)) :=
by
  sorry

end smallest_prime_with_composite_reverse_l97_97825


namespace zoo_animal_difference_l97_97729

variable (giraffes non_giraffes : ℕ)

theorem zoo_animal_difference (h1 : giraffes = 300) (h2 : giraffes = 3 * non_giraffes) : giraffes - non_giraffes = 200 :=
by 
  sorry

end zoo_animal_difference_l97_97729


namespace range_of_k_l97_97048

variable (k : ℝ)
def f (x : ℝ) : ℝ := k * x + 1
def g (x : ℝ) : ℝ := x^2 - 1

theorem range_of_k (h : ∀ x : ℝ, f k x > 0 ∨ g x > 0) : k ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) := 
sorry

end range_of_k_l97_97048


namespace labor_union_trees_l97_97110

theorem labor_union_trees (x : ℕ) :
  (∃ t : ℕ, t = 2 * x + 21) ∧ (∃ t' : ℕ, t' = 3 * x - 24) →
  2 * x + 21 = 3 * x - 24 :=
by
  sorry

end labor_union_trees_l97_97110


namespace wait_time_at_least_8_l97_97430

-- Define the conditions
variables (p₀ p : ℝ) (r x : ℝ)

-- Given conditions
def initial_BAC := p₀ = 89
def BAC_after_2_hours := p = 61
def BAC_decrease := p = p₀ * (Real.exp (r * x))
def decrease_in_2_hours := p = 89 * (Real.exp (r * 2))

-- The main goal to prove the time required is at least 8 hours
theorem wait_time_at_least_8 (h1 : p₀ = 89) (h2 : p = 61) (h3 : p = p₀ * Real.exp (r * x)) (h4 : 61 = 89 * Real.exp (2 * r)) : 
  ∃ x, 89 * Real.exp (r * x) < 20 ∧ x ≥ 8 :=
sorry

end wait_time_at_least_8_l97_97430


namespace quadratic_solution_l97_97015

theorem quadratic_solution (m n : ℝ) (h1 : m ≠ 0) (h2 : m * 1^2 + n * 1 - 1 = 0) : m + n = 1 :=
sorry

end quadratic_solution_l97_97015


namespace more_elements_in_set_N_l97_97697

theorem more_elements_in_set_N 
  (M N : Finset ℕ) 
  (h_partition : ∀ x, x ∈ M ∨ x ∈ N) 
  (h_disjoint : ∀ x, x ∈ M → x ∉ N) 
  (h_total_2000 : M.card + N.card = 10^2000 - 10^1999) 
  (h_total_1000 : (10^1000 - 10^999) * (10^1000 - 10^999) < 10^2000 - 10^1999) : 
  N.card > M.card :=
by { sorry }

end more_elements_in_set_N_l97_97697


namespace at_least_one_number_greater_than_16000_l97_97671

theorem at_least_one_number_greater_than_16000 
    (numbers : Fin 20 → ℕ) 
    (h_distinct : Function.Injective numbers)
    (h_square_product : ∀ i : Fin 19, ∃ k : ℕ, numbers i * numbers (i + 1) = k^2)
    (h_first : numbers 0 = 42) :
    ∃ i : Fin 20, numbers i > 16000 :=
by
  sorry

end at_least_one_number_greater_than_16000_l97_97671


namespace exists_rectangle_with_perimeter_divisible_by_4_l97_97385

-- Define the problem conditions in Lean
def square_length : ℕ := 2015

-- Define what it means to cut the square into rectangles with integer sides
def is_rectangle (a b : ℕ) := 1 ≤ a ∧ a ≤ square_length ∧ 1 ≤ b ∧ b ≤ square_length

-- Define the perimeter condition
def perimeter_divisible_by_4 (a b : ℕ) := (2 * a + 2 * b) % 4 = 0

-- Final theorem statement
theorem exists_rectangle_with_perimeter_divisible_by_4 :
  ∃ (a b : ℕ), is_rectangle a b ∧ perimeter_divisible_by_4 a b :=
by {
  sorry -- The proof itself will be filled in to establish the theorem
}

end exists_rectangle_with_perimeter_divisible_by_4_l97_97385


namespace sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l97_97191

theorem sum_of_roots_eq_zero (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 + x2 = 0 :=
by
  sorry

theorem product_of_roots_eq_neg_twentyfive (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 * x2 = -25 :=
by
  sorry

end sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l97_97191


namespace number_of_C_atoms_in_compound_is_4_l97_97202

def atomic_weight_C : ℕ := 12
def atomic_weight_H : ℕ := 1
def atomic_weight_O : ℕ := 16

def molecular_weight : ℕ := 65

def weight_contributed_by_H_O : ℕ := atomic_weight_H + atomic_weight_O -- 17 amu

def weight_contributed_by_C : ℕ := molecular_weight - weight_contributed_by_H_O -- 48 amu

def number_of_C_atoms := weight_contributed_by_C / atomic_weight_C -- The quotient of 48 amu divided by 12 amu per C atom

theorem number_of_C_atoms_in_compound_is_4 : number_of_C_atoms = 4 :=
by
  sorry -- This is where the proof would go, but it's omitted as per instructions.

end number_of_C_atoms_in_compound_is_4_l97_97202


namespace min_b1_b2_sum_l97_97978

def sequence_relation (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (3 * b n + 4073) / (2 + b (n + 1))

theorem min_b1_b2_sum (b : ℕ → ℕ) (h_seq : sequence_relation b) 
  (h_b1_pos : b 1 > 0) (h_b2_pos : b 2 > 0) :
  b 1 + b 2 = 158 :=
sorry

end min_b1_b2_sum_l97_97978


namespace larger_number_eq_1599_l97_97384

theorem larger_number_eq_1599 (L S : ℕ) (h1 : L - S = 1335) (h2 : L = 6 * S + 15) : L = 1599 :=
by 
  sorry

end larger_number_eq_1599_l97_97384


namespace coffee_pods_per_box_l97_97813

theorem coffee_pods_per_box (d k : ℕ) (c e : ℝ) (h1 : d = 40) (h2 : k = 3) (h3 : c = 8) (h4 : e = 32) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end coffee_pods_per_box_l97_97813


namespace sasha_salt_factor_l97_97304

theorem sasha_salt_factor (x y : ℝ) : 
  (y = 2 * x) →
  (x + y = 2 * x + y / 2) →
  (3 * x / (2 * x) = 1.5) :=
by
  intros h₁ h₂
  sorry

end sasha_salt_factor_l97_97304


namespace lara_bought_52_stems_l97_97792

-- Define the conditions given in the problem:
def flowers_given_to_mom : ℕ := 15
def flowers_given_to_grandma : ℕ := flowers_given_to_mom + 6
def flowers_in_vase : ℕ := 16

-- The total number of stems of flowers Lara bought should be:
def total_flowers_bought : ℕ := flowers_given_to_mom + flowers_given_to_grandma + flowers_in_vase

-- The main theorem to prove the total number of flowers Lara bought is 52:
theorem lara_bought_52_stems : total_flowers_bought = 52 := by
  sorry

end lara_bought_52_stems_l97_97792


namespace area_difference_l97_97345

noncomputable def speed_ratio_A_B : ℚ := 3 / 2
noncomputable def side_length : ℝ := 100
noncomputable def perimeter : ℝ := 4 * side_length

noncomputable def distance_A := (3 / 5) * perimeter
noncomputable def distance_B := perimeter - distance_A

noncomputable def EC := distance_A - 2 * side_length
noncomputable def DE := distance_B - side_length

noncomputable def area_ADE := 0.5 * DE * side_length
noncomputable def area_BCE := 0.5 * EC * side_length

theorem area_difference :
  (area_ADE - area_BCE) = 1000 :=
by
  sorry

end area_difference_l97_97345


namespace missing_digit_is_0_l97_97127

/- Define the known digits of the number. -/
def digit1 : ℕ := 6
def digit2 : ℕ := 5
def digit3 : ℕ := 3
def digit4 : ℕ := 4

/- Define the condition that ensures the divisibility by 9. -/
def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

/- The main theorem to prove: the value of the missing digit d is 0. -/
theorem missing_digit_is_0 (d : ℕ) 
  (h : is_divisible_by_9 (digit1 + digit2 + digit3 + digit4 + d)) : 
  d = 0 :=
sorry

end missing_digit_is_0_l97_97127


namespace num_combinations_L_shape_l97_97354

theorem num_combinations_L_shape (n : ℕ) (k : ℕ) (grid_size : ℕ) (L_shape_blocks : ℕ) 
  (h1 : n = 6) (h2 : k = 4) (h3 : grid_size = 36) (h4 : L_shape_blocks = 4) : 
  ∃ (total_combinations : ℕ), total_combinations = 1800 := by
  sorry

end num_combinations_L_shape_l97_97354


namespace remainder_when_dividing_25197631_by_17_l97_97964

theorem remainder_when_dividing_25197631_by_17 :
  25197631 % 17 = 10 :=
by
  sorry

end remainder_when_dividing_25197631_by_17_l97_97964


namespace pages_left_l97_97532

theorem pages_left (total_pages read_fraction : ℕ) (h_total_pages : total_pages = 396) (h_read_fraction : read_fraction = 1/3) : total_pages * (1 - read_fraction) = 264 := 
by
  sorry

end pages_left_l97_97532


namespace point_on_circle_l97_97210

theorem point_on_circle (t : ℝ) : 
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  x^2 + y^2 = 1 :=
by
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  sorry

end point_on_circle_l97_97210


namespace pencil_lead_loss_l97_97737

theorem pencil_lead_loss (L r : ℝ) (h : r = L * 1/10):
  ((9/10 * r^3) * (2/3)) / (r^3) = 3/5 := 
by
  sorry

end pencil_lead_loss_l97_97737


namespace root_monotonicity_l97_97166

noncomputable def f (x : ℝ) := 3^x + 2 / (1 - x)

theorem root_monotonicity
  (x0 : ℝ) (H_root : f x0 = 0)
  (x1 x2 : ℝ) (H1 : x1 > 1) (H2 : x1 < x0) (H3 : x2 > x0) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end root_monotonicity_l97_97166


namespace relationship_among_a_ab_ab2_l97_97427

theorem relationship_among_a_ab_ab2 (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) :
  a < a * b ∧ a * b < a * b^2 :=
by
  sorry

end relationship_among_a_ab_ab2_l97_97427


namespace bus_speed_including_stoppages_l97_97457

theorem bus_speed_including_stoppages
  (speed_without_stoppages : ℝ)
  (stoppage_time : ℝ)
  (remaining_time_ratio : ℝ)
  (h1 : speed_without_stoppages = 12)
  (h2 : stoppage_time = 0.5)
  (h3 : remaining_time_ratio = 1 - stoppage_time) :
  (speed_without_stoppages * remaining_time_ratio) = 6 := 
by
  sorry

end bus_speed_including_stoppages_l97_97457


namespace find_price_of_pants_l97_97721

theorem find_price_of_pants
  (price_jacket : ℕ)
  (num_jackets : ℕ)
  (price_shorts : ℕ)
  (num_shorts : ℕ)
  (num_pants : ℕ)
  (total_cost : ℕ)
  (h1 : price_jacket = 10)
  (h2 : num_jackets = 3)
  (h3 : price_shorts = 6)
  (h4 : num_shorts = 2)
  (h5 : num_pants = 4)
  (h6 : total_cost = 90)
  : (total_cost - (num_jackets * price_jacket + num_shorts * price_shorts)) / num_pants = 12 :=
by sorry

end find_price_of_pants_l97_97721


namespace no_square_ends_in_4444_l97_97528

theorem no_square_ends_in_4444:
  ∀ (a k : ℕ), (a ^ 2 = 1000 * k + 444) → (∃ b m n : ℕ, (b = 500 * n + 38) ∨ (b = 500 * n - 38) → (a = 2 * b) →
  (a ^ 2 ≠ 1000 * m + 4444)) :=
by
  sorry

end no_square_ends_in_4444_l97_97528


namespace max_ab_l97_97954

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ab ≤ 1 / 16 :=
by
  sorry

end max_ab_l97_97954


namespace problem_integer_condition_l97_97451

theorem problem_integer_condition (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 14)
  (h2 : (235935623 * 74^0 + 2 * 74^1 + 6 * 74^2 + 5 * 74^3 + 3 * 74^4 + 9 * 74^5 + 
         5 * 74^6 + 3 * 74^7 + 2 * 74^8 - a) % 15 = 0) : a = 0 :=
by
  sorry

end problem_integer_condition_l97_97451


namespace correct_system_equations_l97_97123

theorem correct_system_equations (x y : ℤ) : 
  (8 * x - y = 3) ∧ (y - 7 * x = 4) ↔ 
    (8 * x - y = 3) ∧ (y - 7 * x = 4) := by
  sorry

end correct_system_equations_l97_97123


namespace woody_savings_l97_97934

-- Definitions from conditions
def console_cost : Int := 282
def weekly_allowance : Int := 24
def saving_weeks : Int := 10

-- Theorem to prove that the amount Woody already has is $42
theorem woody_savings :
  (console_cost - (weekly_allowance * saving_weeks)) = 42 := 
by
  sorry

end woody_savings_l97_97934


namespace lesser_number_l97_97880

theorem lesser_number (x y : ℕ) (h1: x + y = 60) (h2: x - y = 10) : y = 25 :=
sorry

end lesser_number_l97_97880


namespace tan_identity_l97_97105

open Real

-- Definition of conditions
def isPureImaginary (z : Complex) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem tan_identity (theta : ℝ) :
  isPureImaginary ((cos theta - 4/5) + (sin theta - 3/5) * Complex.I) →
  tan (theta - π / 4) = -7 :=
by
  sorry

end tan_identity_l97_97105


namespace studentC_spending_l97_97277

-- Definitions based on the problem conditions

-- Prices of Type A and Type B notebooks, respectively
variables (x y : ℝ)

-- Number of each type of notebook bought by Student A
def studentA : Prop := x + y = 3

-- Number of Type A notebooks bought by Student B
variables (a : ℕ)

-- Total cost and number of notebooks bought by Student B
def studentB : Prop := (x * a + y * (8 - a) = 11)

-- Constraints on the number of Type A and B notebooks bought by Student C
def studentC_notebooks : Prop := ∃ b : ℕ, b = 8 - a ∧ b = a

-- The total amount spent by Student C
def studentC_cost : ℝ := (8 - a) * x + a * y

-- The statement asserting the cost is 13 yuan
theorem studentC_spending (x y : ℝ) (a : ℕ) (hA : studentA x y) (hB : studentB x y a) (hC : studentC_notebooks a) : studentC_cost x y a = 13 := sorry

end studentC_spending_l97_97277


namespace max_correct_answers_l97_97548

theorem max_correct_answers (a b c : ℕ) :
  a + b + c = 50 ∧ 4 * a - c = 99 ∧ b = 50 - a - c ∧ 50 - a - c ≥ 0 →
  a ≤ 29 := by
  sorry

end max_correct_answers_l97_97548


namespace average_pages_per_hour_l97_97422

theorem average_pages_per_hour 
  (P : ℕ) (H : ℕ) (hP : P = 30000) (hH : H = 150) : 
  P / H = 200 := 
by 
  sorry

end average_pages_per_hour_l97_97422


namespace fifth_stack_33_l97_97788

def cups_in_fifth_stack (a d : ℕ) : ℕ :=
a + 4 * d

theorem fifth_stack_33 
  (a : ℕ) 
  (d : ℕ) 
  (h_first_stack : a = 17) 
  (h_pattern : d = 4) : 
  cups_in_fifth_stack a d = 33 := by
  sorry

end fifth_stack_33_l97_97788


namespace average_speed_l97_97644

-- Defining conditions
def speed_first_hour : ℕ := 100  -- The car travels 100 km in the first hour
def speed_second_hour : ℕ := 60  -- The car travels 60 km in the second hour
def total_distance : ℕ := speed_first_hour + speed_second_hour  -- Total distance traveled

def total_time : ℕ := 2  -- Total time taken in hours

-- Stating the theorem
theorem average_speed : total_distance / total_time = 80 := 
by
  sorry

end average_speed_l97_97644


namespace parameterization_of_line_l97_97170

theorem parameterization_of_line : 
  ∀ (r k : ℝ),
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (r, 2) + t • (3, k)) → y = 2 * x - 6) → (r = 4 ∧ k = 6) :=
by
  sorry

end parameterization_of_line_l97_97170


namespace number_of_months_to_fully_pay_off_car_l97_97892

def total_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem number_of_months_to_fully_pay_off_car :
  (total_price - initial_payment) / monthly_payment = 19 :=
by
  sorry

end number_of_months_to_fully_pay_off_car_l97_97892


namespace area_evaluation_l97_97155

noncomputable def radius : ℝ := 6
noncomputable def central_angle : ℝ := 90
noncomputable def p := 18
noncomputable def q := 3
noncomputable def r : ℝ := -27 / 2

theorem area_evaluation :
  p + q + r = 7.5 :=
by
  sorry

end area_evaluation_l97_97155


namespace part1_part2_l97_97257

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem part1 (x : ℝ) : (f x)^2 - (g x)^2 = -4 :=
by sorry

theorem part2 (x y : ℝ) (h1 : f x * f y = 4) (h2 : g x * g y = 8) : 
  g (x + y) / g (x - y) = 3 :=
by sorry

end part1_part2_l97_97257


namespace quadrilateral_area_l97_97678

variable (d : ℝ) (o₁ : ℝ) (o₂ : ℝ)

theorem quadrilateral_area (h₁ : d = 28) (h₂ : o₁ = 8) (h₃ : o₂ = 2) : 
  (1 / 2 * d * o₁) + (1 / 2 * d * o₂) = 140 := 
  by
    rw [h₁, h₂, h₃]
    sorry

end quadrilateral_area_l97_97678


namespace blend_pieces_eq_two_l97_97613

variable (n_silk n_cashmere total_pieces : ℕ)

def luther_line := n_silk = 10 ∧ n_cashmere = n_silk / 2 ∧ total_pieces = 13

theorem blend_pieces_eq_two : luther_line n_silk n_cashmere total_pieces → (n_cashmere - (total_pieces - n_silk) = 2) :=
by
  intros
  sorry

end blend_pieces_eq_two_l97_97613


namespace fraction_subtraction_l97_97852

theorem fraction_subtraction (a b : ℝ) (h1 : 2 * b = 1 + a * b) (h2 : a ≠ 1) (h3 : b ≠ 1) :
  (a + 1) / (a - 1) - (b + 1) / (b - 1) = 2 :=
by
  sorry

end fraction_subtraction_l97_97852


namespace exists_n_satisfying_conditions_l97_97183

open Nat

-- Define that n satisfies the given conditions
theorem exists_n_satisfying_conditions :
  ∃ (n : ℤ), (∃ (k : ℤ), 2 * n + 1 = (2 * k + 1) ^ 2) ∧ 
            (∃ (h : ℤ), 3 * n + 1 = (2 * h + 1) ^ 2) ∧ 
            (40 ∣ n) := by
  sorry

end exists_n_satisfying_conditions_l97_97183


namespace division_remainder_l97_97568

def polynomial (x: ℤ) : ℤ := 3 * x^7 - x^6 - 7 * x^5 + 2 * x^3 + 4 * x^2 - 11
def divisor (x: ℤ) : ℤ := 2 * x - 4

theorem division_remainder : (polynomial 2) = 117 := 
  by 
  -- We state what needs to be proven here formally
  sorry

end division_remainder_l97_97568


namespace find_sample_size_l97_97926

def sports_team (total: Nat) (soccer: Nat) (basketball: Nat) (table_tennis: Nat) : Prop :=
  total = soccer + basketball + table_tennis

def valid_sample_size (total: Nat) (n: Nat) :=
  (n > 0) ∧ (total % n == 0) ∧ (n % 6 == 0)

def systematic_sampling_interval (total: Nat) (n: Nat): Nat :=
  total / n

theorem find_sample_size :
  ∀ (total soccer basketball table_tennis: Nat),
  sports_team total soccer basketball table_tennis →
  total = 36 →
  soccer = 18 →
  basketball = 12 →
  table_tennis = 6 →
  (∃ n, valid_sample_size total n ∧ valid_sample_size (total - 1) (n + 1)) →
  ∃ n, n = 6 := by
  sorry

end find_sample_size_l97_97926


namespace average_speed_l97_97296

theorem average_speed (D T : ℝ) (h1 : D = 100) (h2 : T = 6) : (D / T) = 50 / 3 := by
  sorry

end average_speed_l97_97296


namespace one_thirds_in_nine_halves_l97_97579

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end one_thirds_in_nine_halves_l97_97579


namespace blue_whale_tongue_weight_in_tons_l97_97308

-- Define the conditions
def weight_of_tongue_pounds : ℕ := 6000
def pounds_per_ton : ℕ := 2000

-- Define the theorem stating the question and its answer
theorem blue_whale_tongue_weight_in_tons :
  (weight_of_tongue_pounds / pounds_per_ton) = 3 :=
by sorry

end blue_whale_tongue_weight_in_tons_l97_97308


namespace number_of_zeros_l97_97883

noncomputable def f (x : Real) : Real :=
if x > 0 then -1 + Real.log x
else 3 * x + 4

theorem number_of_zeros : (∃ a b : Real, f a = 0 ∧ f b = 0 ∧ a ≠ b) := 
sorry

end number_of_zeros_l97_97883


namespace triangle_properties_l97_97334

theorem triangle_properties (a b c : ℝ) (h1 : a / b = 5 / 12) (h2 : b / c = 12 / 13) (h3 : a + b + c = 60) :
  (a^2 + b^2 = c^2) ∧ ((1 / 2) * a * b > 100) :=
by
  sorry

end triangle_properties_l97_97334


namespace calculate_expression_l97_97455

theorem calculate_expression : 5 * 7 + 9 * 4 - 36 / 3 + 48 / 4 = 71 := by
  sorry

end calculate_expression_l97_97455


namespace problem_statement_l97_97600

variable {a : ℕ → ℝ} 
variable {a1 d : ℝ}
variable (h_arith : ∀ n, a (n + 1) = a n + d)  -- Arithmetic sequence condition
variable (h_d_nonzero : d ≠ 0)  -- d ≠ 0
variable (h_a1_nonzero : a1 ≠ 0)  -- a1 ≠ 0
variable (h_geom : (a 1) * (a 7) = (a 3) ^ 2)  -- Geometric sequence condition a2 = a 1, a4 = a 3, a8 = a 7

theorem problem_statement :
  (a 0 + a 4 + a 8) / (a 1 + a 2) = 3 :=
by
  sorry

end problem_statement_l97_97600


namespace relationship_not_true_l97_97083

theorem relationship_not_true (a b : ℕ) :
  (b = a + 5 ∨ b = a + 15 ∨ b = a + 29) → ¬(a = b - 9) :=
by
  sorry

end relationship_not_true_l97_97083


namespace fraction_simplification_l97_97144

theorem fraction_simplification (a b : ℝ) : 9 * b / (6 * a + 3) = 3 * b / (2 * a + 1) :=
by sorry

end fraction_simplification_l97_97144


namespace tom_speed_first_part_l97_97615

-- Definitions of conditions in Lean
def total_distance : ℕ := 20
def distance_first_part : ℕ := 10
def speed_second_part : ℕ := 10
def average_speed : ℚ := 10.909090909090908
def distance_second_part := total_distance - distance_first_part

-- Lean statement to prove the speed during the first part of the trip
theorem tom_speed_first_part (v : ℚ) :
  (distance_first_part / v + distance_second_part / speed_second_part) = total_distance / average_speed → v = 12 :=
by
  intro h
  sorry

end tom_speed_first_part_l97_97615


namespace sum_of_transformed_numbers_l97_97812

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end sum_of_transformed_numbers_l97_97812


namespace solve_for_C_and_D_l97_97965

theorem solve_for_C_and_D (C D : ℚ) (h1 : 2 * C + 3 * D + 4 = 31) (h2 : D = C + 2) :
  C = 21 / 5 ∧ D = 31 / 5 :=
by
  sorry

end solve_for_C_and_D_l97_97965


namespace arithmetic_sequence_fifth_term_l97_97217

theorem arithmetic_sequence_fifth_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 6) (h3 : a 3 = 2) (h_arith_seq : ∀ n, a (n + 1) = a n + d) : a 5 = -2 :=
sorry

end arithmetic_sequence_fifth_term_l97_97217


namespace jersey_cost_difference_l97_97570

theorem jersey_cost_difference :
  let jersey_cost := 115
  let tshirt_cost := 25
  jersey_cost - tshirt_cost = 90 :=
by
  -- proof goes here
  sorry

end jersey_cost_difference_l97_97570


namespace tangent_line_at_point_l97_97804

noncomputable def tangent_line_equation (x : ℝ) : Prop :=
  ∀ y : ℝ, y = x * (3 * Real.log x + 1) → (x = 1 ∧ y = 1) → y = 4 * x - 3

theorem tangent_line_at_point : tangent_line_equation 1 :=
sorry

end tangent_line_at_point_l97_97804


namespace euler_sum_of_squares_euler_sum_of_quads_l97_97424

theorem euler_sum_of_squares :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^2 = π^2 / 6 := sorry

theorem euler_sum_of_quads :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^4 = π^4 / 90 := sorry

end euler_sum_of_squares_euler_sum_of_quads_l97_97424


namespace eight_row_triangle_pieces_l97_97972

def unit_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

def connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem eight_row_triangle_pieces : unit_rods 8 + connectors 9 = 153 :=
by
  sorry

end eight_row_triangle_pieces_l97_97972


namespace find_n_given_combination_l97_97747

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem find_n_given_combination : ∃ n : ℕ, binomial_coefficient (n+1) 2 = 21 ↔ n = 6 := by
  sorry

end find_n_given_combination_l97_97747


namespace expression_evaluation_l97_97280

theorem expression_evaluation :
  10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 :=
by
  sorry

end expression_evaluation_l97_97280


namespace tan_cot_theta_l97_97302

theorem tan_cot_theta 
  (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = (Real.sqrt 2) / 3) 
  (h2 : Real.pi / 2 < θ ∧ θ < Real.pi) : 
  Real.tan θ - (1 / Real.tan θ) = - (8 * Real.sqrt 2) / 7 := 
sorry

end tan_cot_theta_l97_97302


namespace range_of_a_l97_97527

theorem range_of_a (a : ℝ) (h : ∀ x, x > a → 2 * x + 2 / (x - a) ≥ 5) : a ≥ 1 / 2 :=
sorry

end range_of_a_l97_97527


namespace power_expression_l97_97314

theorem power_expression : (1 / ((-5)^4)^2) * (-5)^9 = -5 := sorry

end power_expression_l97_97314


namespace mean_squared_sum_l97_97754

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem mean_squared_sum :
  (x + y + z = 30) ∧ 
  (xyz = 125) ∧ 
  ((1 / x + 1 / y + 1 / z) = 3 / 4) 
  → x^2 + y^2 + z^2 = 712.5 :=
by
  intros h
  have h₁ : x + y + z = 30 := h.1
  have h₂ : xyz = 125 := h.2.1
  have h₃ : (1 / x + 1 / y + 1 / z) = 3 / 4 := h.2.2
  sorry

end mean_squared_sum_l97_97754


namespace area_of_triangle_BXC_l97_97227

-- Define a trapezoid ABCD with given conditions
structure Trapezoid :=
  (A B C D X : Type)
  (AB CD : ℝ)
  (area_ABCD : ℝ)
  (intersect_at_X : Prop)

theorem area_of_triangle_BXC (t : Trapezoid) (h1 : t.AB = 24) (h2 : t.CD = 40)
  (h3 : t.area_ABCD = 480) (h4 : t.intersect_at_X) : 
  ∃ (area_BXC : ℝ), area_BXC = 120 :=
by {
  -- skip the proof here by using sorry
  sorry
}

end area_of_triangle_BXC_l97_97227


namespace quadratic_roots_real_or_imaginary_l97_97573

theorem quadratic_roots_real_or_imaginary (a b c d: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) 
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
∃ (A B C: ℝ), (A = a ∨ A = b ∨ A = c ∨ A = d) ∧ (B = a ∨ B = b ∨ B = c ∨ B = d) ∧ (C = a ∨ C = b ∨ C = c ∨ C = d) ∧ 
(A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ 
((1 - 4*B*C ≥ 0 ∧ 1 - 4*C*A ≥ 0 ∧ 1 - 4*A*B ≥ 0) ∨ (1 - 4*B*C < 0 ∧ 1 - 4*C*A < 0 ∧ 1 - 4*A*B < 0)) :=
by
  sorry

end quadratic_roots_real_or_imaginary_l97_97573


namespace ratio_xyz_l97_97566

theorem ratio_xyz (a x y z : ℝ) : 
  5 * x + 4 * y - 6 * z = a ∧
  4 * x - 5 * y + 7 * z = 27 * a ∧
  6 * x + 5 * y - 4 * z = 18 * a →
  (x :ℝ) / (y :ℝ) = 3 / 4 ∧
  (y :ℝ) / (z :ℝ) = 4 / 5 :=
by
  sorry

end ratio_xyz_l97_97566


namespace quadratic_roots_range_l97_97680

variable (a : ℝ)

theorem quadratic_roots_range (h : ∀ b c (eq : b = -a ∧ c = a^2 - 4), ∃ x y, x ≠ y ∧ x^2 + b * x + c = 0 ∧ x > 0 ∧ y^2 + b * y + c = 0) :
  -2 ≤ a ∧ a ≤ 2 :=
by sorry

end quadratic_roots_range_l97_97680


namespace laser_beam_total_distance_l97_97211

theorem laser_beam_total_distance :
  let A := (3, 5)
  let D := (7, 5)
  let D'' := (-7, -5)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  distance A D'' = 10 * Real.sqrt 2 :=
by
  -- definitions and conditions are captured
  sorry -- the proof goes here, no proof is required as per instructions

end laser_beam_total_distance_l97_97211


namespace hannah_monday_run_l97_97333

-- Definitions of the conditions
def ran_on_wednesday : ℕ := 4816
def ran_on_friday : ℕ := 2095
def extra_on_monday : ℕ := 2089

-- Translations to set the total combined distance and the distance ran on Monday
def combined_distance := ran_on_wednesday + ran_on_friday
def ran_on_monday := combined_distance + extra_on_monday

-- A statement to show she ran 9 kilometers on Monday
theorem hannah_monday_run :
  ran_on_monday = 9000 / 1000 * 1000 := sorry

end hannah_monday_run_l97_97333


namespace single_bacteria_colony_days_to_limit_l97_97382

theorem single_bacteria_colony_days_to_limit (n : ℕ) (h : ∀ t : ℕ, t ≤ 21 → (2 ^ t = 2 * 2 ^ (t - 1))) : n = 22 :=
by
  sorry

end single_bacteria_colony_days_to_limit_l97_97382


namespace power_modulo_l97_97408

theorem power_modulo (h : 3 ^ 4 ≡ 1 [MOD 10]) : 3 ^ 2023 ≡ 7 [MOD 10] :=
by
  sorry

end power_modulo_l97_97408


namespace scissors_total_l97_97779

theorem scissors_total (initial_scissors : ℕ) (additional_scissors : ℕ) (h1 : initial_scissors = 54) (h2 : additional_scissors = 22) : 
  initial_scissors + additional_scissors = 76 :=
by
  sorry

end scissors_total_l97_97779


namespace cube_volume_l97_97022

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l97_97022


namespace boy_age_is_10_l97_97054

-- Define the boy's current age as a variable
def boy_current_age := 10

-- Define a condition based on the boy's statement
def boy_statement_condition (x : ℕ) : Prop :=
  x = 2 * (x - 5)

-- The main theorem stating equivalence of the boy's current age to 10 given the condition
theorem boy_age_is_10 (x : ℕ) (h : boy_statement_condition x) : x = boy_current_age := by
  sorry

end boy_age_is_10_l97_97054


namespace parabola_tangent_angle_l97_97161

noncomputable def tangent_slope_angle : Real :=
  let x := (1 / 2 : ℝ)
  let y := x^2
  let slope := (deriv (fun x => x^2)) x
  Real.arctan slope

theorem parabola_tangent_angle :
  tangent_slope_angle = Real.pi / 4 :=
by
sorry

end parabola_tangent_angle_l97_97161


namespace correct_option_l97_97205

theorem correct_option : (-1 - 3 = -4) ∧ ¬(-2 + 8 = 10) ∧ ¬(-2 * 2 = 4) ∧ ¬(-8 / -1 = -1 / 8) :=
by
  sorry

end correct_option_l97_97205


namespace value_of_x_l97_97467

theorem value_of_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 := 
by
  sorry

end value_of_x_l97_97467


namespace find_q_l97_97670

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l97_97670


namespace micah_water_intake_l97_97561

def morning : ℝ := 1.5
def early_afternoon : ℝ := 2 * morning
def late_afternoon : ℝ := 3 * morning
def evening : ℝ := late_afternoon - 0.25 * late_afternoon
def night : ℝ := 2 * evening
def total_water_intake : ℝ := morning + early_afternoon + late_afternoon + evening + night

theorem micah_water_intake :
  total_water_intake = 19.125 := by
  sorry

end micah_water_intake_l97_97561


namespace profit_per_meter_correct_l97_97546

-- Define the conditions
def total_meters := 40
def total_profit := 1400

-- Define the profit per meter calculation
def profit_per_meter := total_profit / total_meters

-- Theorem stating the profit per meter is Rs. 35
theorem profit_per_meter_correct : profit_per_meter = 35 := by
  sorry

end profit_per_meter_correct_l97_97546


namespace f_has_one_zero_l97_97728

noncomputable def f (x : ℝ) : ℝ := 2 * x - 5 - Real.log x

theorem f_has_one_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end f_has_one_zero_l97_97728


namespace taehyung_walks_more_than_minyoung_l97_97603

def taehyung_distance_per_minute : ℕ := 114
def minyoung_distance_per_minute : ℕ := 79
def minutes_per_hour : ℕ := 60

theorem taehyung_walks_more_than_minyoung :
  (taehyung_distance_per_minute * minutes_per_hour) -
  (minyoung_distance_per_minute * minutes_per_hour) = 2100 := by
  sorry

end taehyung_walks_more_than_minyoung_l97_97603


namespace lisa_marbles_l97_97871

def ConnieMarbles : ℕ := 323
def JuanMarbles (ConnieMarbles : ℕ) : ℕ := ConnieMarbles + 175
def MarkMarbles (JuanMarbles : ℕ) : ℕ := 3 * JuanMarbles
def LisaMarbles (MarkMarbles : ℕ) : ℕ := MarkMarbles / 2 - 200

theorem lisa_marbles :
  LisaMarbles (MarkMarbles (JuanMarbles ConnieMarbles)) = 547 := by
  sorry

end lisa_marbles_l97_97871


namespace calculate_A_minus_B_l97_97957

variable (A B : ℝ)
variable (h1 : A + B + B = 814.8)
variable (h2 : 10 * B = A)

theorem calculate_A_minus_B : A - B = 611.1 :=
by
  sorry

end calculate_A_minus_B_l97_97957


namespace hypotenuse_length_l97_97126

theorem hypotenuse_length (a b c : ℝ)
  (h_a : a = 12)
  (h_area : 54 = 1 / 2 * a * b)
  (h_py : c^2 = a^2 + b^2) :
    c = 15 := by
  sorry

end hypotenuse_length_l97_97126


namespace arithmetic_seq_sum_2013_l97_97278

noncomputable def a1 : ℤ := -2013
noncomputable def S (n d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_seq_sum_2013 :
  ∃ d : ℤ, (S 12 d / 12 - S 10 d / 10 = 2) → S 2013 d = -2013 :=
by
  sorry

end arithmetic_seq_sum_2013_l97_97278


namespace range_m_l97_97620

-- Definitions for propositions p and q
def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0)

def q (m : ℝ) : Prop :=
  (m - 1 > 0)

-- Given conditions:
-- 1. p ∨ q is true
-- 2. p ∧ q is false

theorem range_m (m : ℝ) (h1: p m ∨ q m) (h2: ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
by
  sorry

end range_m_l97_97620


namespace number_of_solutions_is_3_l97_97326

noncomputable def count_solutions : Nat :=
  Nat.card {x : Nat // x < 150 ∧ (x + 15) % 45 = 75 % 45}

theorem number_of_solutions_is_3 : count_solutions = 3 := by
  sorry

end number_of_solutions_is_3_l97_97326


namespace find_a8_l97_97252

variable (a : ℕ → ℝ)
variable (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a8 
  (hq : is_geometric_sequence a q)
  (h1 : a 1 * a 3 = 4)
  (h2 : a 9 = 256) : 
  a 8 = 128 ∨ a 8 = -128 :=
by
  sorry

end find_a8_l97_97252


namespace final_selling_price_l97_97623

variable (a : ℝ)

theorem final_selling_price (h : a > 0) : 0.9 * (1.25 * a) = 1.125 * a := 
by
  sorry

end final_selling_price_l97_97623


namespace value_of_f_2014_l97_97125

def f : ℕ → ℕ := sorry

theorem value_of_f_2014 : (∀ n : ℕ, f (f n) + f n = 2 * n + 3) → (f 0 = 1) → (f 2014 = 2015) := by
  intro h₁ h₀
  have h₂ := h₀
  sorry

end value_of_f_2014_l97_97125


namespace max_cities_l97_97131

def city (X : Type) := X

variable (A B C D E : Prop)

-- Conditions as given in the problem
axiom condition1 : A → B
axiom condition2 : D ∨ E
axiom condition3 : B ↔ ¬C
axiom condition4 : C ↔ D
axiom condition5 : E → (A ∧ D)

-- Proof problem: Given the conditions, prove that the maximum set of cities that can be visited is {C, D}
theorem max_cities (h1 : A → B) (h2 : D ∨ E) (h3 : B ↔ ¬C) (h4 : C ↔ D) (h5 : E → (A ∧ D)) : (C ∧ D) ∧ ¬A ∧ ¬B ∧ ¬E :=
by
  -- The core proof would use the constraints to show C and D, and exclude A, B, E
  sorry

end max_cities_l97_97131


namespace mariela_cards_total_l97_97643

theorem mariela_cards_total : 
  let a := 287.0
  let b := 116
  a + b = 403 := 
by
  sorry

end mariela_cards_total_l97_97643


namespace matchsticks_left_l97_97909

def initial_matchsticks : ℕ := 30
def matchsticks_needed_2 : ℕ := 5
def matchsticks_needed_0 : ℕ := 6
def num_2s : ℕ := 3
def num_0s : ℕ := 1

theorem matchsticks_left : 
  initial_matchsticks - (num_2s * matchsticks_needed_2 + num_0s * matchsticks_needed_0) = 9 :=
by sorry

end matchsticks_left_l97_97909


namespace roots_of_quadratic_l97_97007

theorem roots_of_quadratic (p q x1 x2 : ℕ) (hp : p + q = 28) (hroots : ∀ x, x^2 + p * x + q = 0 → (x = x1 ∨ x = x2)) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (x1 = 30 ∧ x2 = 2) ∨ (x1 = 2 ∧ x2 = 30) :=
sorry

end roots_of_quadratic_l97_97007


namespace orchard_tree_growth_problem_l97_97701

theorem orchard_tree_growth_problem
  (T0 : ℕ) (Tn : ℕ) (n : ℕ)
  (h1 : T0 = 1280)
  (h2 : Tn = 3125)
  (h3 : Tn = (5/4 : ℚ) ^ n * T0) :
  n = 4 :=
by
  sorry

end orchard_tree_growth_problem_l97_97701


namespace sufficiency_not_necessity_l97_97052

theorem sufficiency_not_necessity (x y : ℝ) :
  (x > 3 ∧ y > 3) → (x + y > 6 ∧ x * y > 9) ∧ (¬ (x + y > 6 ∧ x * y > 9 → x > 3 ∧ y > 3)) :=
by
  sorry

end sufficiency_not_necessity_l97_97052


namespace nested_fraction_evaluation_l97_97500

def nested_expression := 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))

theorem nested_fraction_evaluation : nested_expression = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l97_97500


namespace expressions_same_type_l97_97683

def same_type_as (e1 e2 : ℕ × ℕ) : Prop :=
  e1 = e2

def exp_of_expr (a_exp b_exp : ℕ) : ℕ × ℕ :=
  (a_exp, b_exp)

def exp_3a2b := exp_of_expr 2 1
def exp_neg_ba2 := exp_of_expr 2 1

theorem expressions_same_type :
  same_type_as exp_neg_ba2 exp_3a2b :=
by
  sorry

end expressions_same_type_l97_97683


namespace hyperbola_eccentricity_l97_97803

theorem hyperbola_eccentricity (a b m n e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_mn : m * n = 2 / 9)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) : e = 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_l97_97803


namespace inequality_proof_l97_97816

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
by 
  sorry

end inequality_proof_l97_97816


namespace find_x_l97_97713

theorem find_x :
  ∃ x : ℝ, x = (1/x) * (-x) - 3*x + 4 ∧ x = 3/4 :=
by
  sorry

end find_x_l97_97713


namespace fraction_of_field_planted_l97_97405

theorem fraction_of_field_planted (AB AC : ℕ) (x : ℕ) (shortest_dist : ℕ) (hypotenuse : ℕ)
  (S : ℕ) (total_area : ℕ) (planted_area : ℕ) :
  AB = 5 ∧ AC = 12 ∧ hypotenuse = 13 ∧ shortest_dist = 2 ∧ x * x = S ∧ 
  total_area = 30 ∧ planted_area = total_area - S →
  (planted_area / total_area : ℚ) = 2951 / 3000 :=
by
  sorry

end fraction_of_field_planted_l97_97405


namespace find_a_l97_97699

noncomputable def a_value_given_conditions : ℝ :=
  let A := 30 * Real.pi / 180
  let C := 105 * Real.pi / 180
  let B := 180 * Real.pi / 180 - A - C
  let b := 8
  let a := (b * Real.sin A) / Real.sin B
  a

theorem find_a :
  a_value_given_conditions = 4 * Real.sqrt 2 :=
by
  -- We assume that the value computation as specified is correct
  -- hence this is just stating the problem.
  sorry

end find_a_l97_97699


namespace profit_is_5000_l97_97765

namespace HorseshoeProfit

-- Defining constants and conditions
def initialOutlay : ℝ := 10000
def costPerSet : ℝ := 20
def sellingPricePerSet : ℝ := 50
def numberOfSets : ℝ := 500

-- Calculating the profit
def profit : ℝ :=
  let revenue := numberOfSets * sellingPricePerSet
  let manufacturingCosts := initialOutlay + (costPerSet * numberOfSets)
  revenue - manufacturingCosts

-- The main theorem: the profit is $5,000
theorem profit_is_5000 : profit = 5000 := by
  sorry

end HorseshoeProfit

end profit_is_5000_l97_97765


namespace find_number_l97_97517

theorem find_number (n : ℝ) (h : n / 0.06 = 16.666666666666668) : n = 1 :=
by
  sorry

end find_number_l97_97517


namespace tshirt_more_expensive_l97_97009

-- Definitions based on given conditions
def jeans_price : ℕ := 30
def socks_price : ℕ := 5
def tshirt_price : ℕ := jeans_price / 2

-- Statement to prove (The t-shirt is $10 more expensive than the socks)
theorem tshirt_more_expensive : (tshirt_price - socks_price) = 10 :=
by
  rw [tshirt_price, socks_price]
  sorry  -- proof steps are omitted

end tshirt_more_expensive_l97_97009


namespace regular_hexagon_interior_angle_deg_l97_97564

theorem regular_hexagon_interior_angle_deg (n : ℕ) (h1 : n = 6) :
  let sum_of_interior_angles : ℕ := (n - 2) * 180
  let each_angle : ℕ := sum_of_interior_angles / n
  each_angle = 120 := by
  sorry

end regular_hexagon_interior_angle_deg_l97_97564


namespace prob_sum_equals_15_is_0_l97_97631

theorem prob_sum_equals_15_is_0 (coin1 coin2 : ℕ) (die_min die_max : ℕ) (age : ℕ)
  (h1 : coin1 = 5) (h2 : coin2 = 15) (h3 : die_min = 1) (h4 : die_max = 6) (h5 : age = 15) :
  ((coin1 = 5 ∨ coin2 = 15) → die_min ≤ ((if coin1 = 5 then 5 else 15) + (die_max - die_min + 1)) ∧ 
   (die_min ≤ 6) ∧ 6 ≤ die_max) → 
  0 = 0 :=
by
  sorry

end prob_sum_equals_15_is_0_l97_97631


namespace commutative_binary_op_no_identity_element_associative_binary_op_l97_97360

def binary_op (x y : ℤ) : ℤ :=
  2 * (x + 2) * (y + 2) - 3

theorem commutative_binary_op (x y : ℤ) : binary_op x y = binary_op y x := by
  sorry

theorem no_identity_element (x e : ℤ) : ¬ (∀ x, binary_op x e = x) := by
  sorry

theorem associative_binary_op (x y z : ℤ) : (binary_op (binary_op x y) z = binary_op x (binary_op y z)) ∨ ¬ (binary_op (binary_op x y) z = binary_op x (binary_op y z)) := by
  sorry

end commutative_binary_op_no_identity_element_associative_binary_op_l97_97360


namespace sum_X_Y_Z_W_eq_156_l97_97756

theorem sum_X_Y_Z_W_eq_156 
  (X Y Z W : ℕ) 
  (h_arith_seq : Y - X = Z - Y)
  (h_geom_seq : Z / Y = 9 / 5)
  (h_W : W = Z^2 / Y) 
  (h_pos : 0 < X ∧ 0 < Y ∧ 0 < Z ∧ 0 < W) :
  X + Y + Z + W = 156 :=
sorry

end sum_X_Y_Z_W_eq_156_l97_97756


namespace highway_extension_l97_97226

theorem highway_extension 
  (current_length : ℕ) 
  (desired_length : ℕ) 
  (first_day_miles : ℕ) 
  (miles_needed : ℕ) 
  (second_day_miles : ℕ) 
  (h1 : current_length = 200) 
  (h2 : desired_length = 650) 
  (h3 : first_day_miles = 50) 
  (h4 : miles_needed = 250) 
  (h5 : second_day_miles = desired_length - current_length - miles_needed - first_day_miles) :
  second_day_miles / first_day_miles = 3 := 
sorry

end highway_extension_l97_97226


namespace arithmetic_sequence_sum_l97_97810

noncomputable def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 + n * d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 = 2)
  (h3 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l97_97810


namespace student_marks_l97_97539

theorem student_marks (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (student_marks : ℕ) : 
  (passing_percentage = 30) → (failed_by = 40) → (max_marks = 400) → 
  student_marks = (max_marks * passing_percentage / 100 - failed_by) → 
  student_marks = 80 :=
by {
  sorry
}

end student_marks_l97_97539


namespace polygon_sides_14_l97_97807

theorem polygon_sides_14 (n : ℕ) (θ : ℝ) 
  (h₀ : (n - 2) * 180 - θ = 2000) :
  n = 14 :=
sorry

end polygon_sides_14_l97_97807


namespace find_f2_l97_97900

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x y : ℝ, x * f y = y * f x) (h10 : f 10 = 30) : f 2 = 6 := 
by
  sorry

end find_f2_l97_97900


namespace retirement_total_correct_l97_97483

-- Definitions of the conditions
def hire_year : Nat := 1986
def hire_age : Nat := 30
def retirement_year : Nat := 2006

-- Calculation of age and years of employment at retirement
def employment_duration : Nat := retirement_year - hire_year
def age_at_retirement : Nat := hire_age + employment_duration

-- The required total of age and years of employment for retirement
def total_required_for_retirement : Nat := age_at_retirement + employment_duration

-- The theorem to be proven
theorem retirement_total_correct :
  total_required_for_retirement = 70 :=
  by 
  sorry

end retirement_total_correct_l97_97483


namespace find_a6_l97_97349

variable {a : ℕ → ℝ}

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def given_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36

theorem find_a6 (d : ℝ) :
  is_arithmetic_sequence a d →
  given_condition a d →
  a 6 = 3 :=
by
  -- The proof would go here
  sorry

end find_a6_l97_97349


namespace largest_lcm_among_given_pairs_l97_97107

theorem largest_lcm_among_given_pairs : 
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by
  sorry

end largest_lcm_among_given_pairs_l97_97107


namespace polynomial_root_problem_l97_97884

theorem polynomial_root_problem (a b c d : ℤ) (r1 r2 r3 r4 : ℕ)
  (h_roots : ∀ x, x^4 + a * x^3 + b * x^2 + c * x + d = (x + r1) * (x + r2) * (x + r3) * (x + r4))
  (h_sum : a + b + c + d = 2009) :
  d = 528 := 
by
  sorry

end polynomial_root_problem_l97_97884


namespace paco_initial_sweet_cookies_l97_97240

theorem paco_initial_sweet_cookies
    (x : ℕ)  -- Paco's initial number of sweet cookies
    (eaten_sweet : ℕ)  -- number of sweet cookies Paco ate
    (left_sweet : ℕ)  -- number of sweet cookies Paco had left
    (h1 : eaten_sweet = 15)  -- Paco ate 15 sweet cookies
    (h2 : left_sweet = 19)  -- Paco had 19 sweet cookies left
    (h3 : x - eaten_sweet = left_sweet)  -- After eating, Paco had 19 sweet cookies left
    : x = 34 :=  -- Paco initially had 34 sweet cookies
sorry

end paco_initial_sweet_cookies_l97_97240


namespace edward_original_lawns_l97_97822

-- Definitions based on conditions
def dollars_per_lawn : ℕ := 4
def lawns_forgotten : ℕ := 9
def dollars_earned : ℕ := 32

-- The original number of lawns to mow
def original_lawns_to_mow (L : ℕ) : Prop :=
  dollars_per_lawn * (L - lawns_forgotten) = dollars_earned

-- The proof problem statement
theorem edward_original_lawns : ∃ L : ℕ, original_lawns_to_mow L ∧ L = 17 :=
by
  sorry

end edward_original_lawns_l97_97822


namespace minimum_value_4x_minus_y_l97_97821

theorem minimum_value_4x_minus_y (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y - 4 ≥ 0) (h3 : x ≤ 4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x' y' : ℝ), (x' - y' ≥ 0) → (x' + y' - 4 ≥ 0) → (x' ≤ 4) → 4 * x' - y' ≥ m :=
by
  sorry

end minimum_value_4x_minus_y_l97_97821


namespace simon_stamps_received_l97_97559

theorem simon_stamps_received (initial_stamps total_stamps received_stamps : ℕ) (h1 : initial_stamps = 34) (h2 : total_stamps = 61) : received_stamps = 27 :=
by
  sorry

end simon_stamps_received_l97_97559


namespace a_eq_3x_or_neg2x_l97_97203

theorem a_eq_3x_or_neg2x (a b x : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 19 * x^3) (h3 : a - b = x) :
    a = 3 * x ∨ a = -2 * x :=
by
  -- The proof will go here
  sorry

end a_eq_3x_or_neg2x_l97_97203


namespace simplify_expr_l97_97649

variable (x y : ℝ)

def expr (x y : ℝ) := (x + y) * (x - y) - y * (2 * x - y)

theorem simplify_expr :
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  expr x y = 2 - 2 * Real.sqrt 6 := by
  sorry

end simplify_expr_l97_97649


namespace arrange_descending_order_l97_97550

noncomputable def a := 8 ^ 0.7
noncomputable def b := 8 ^ 0.9
noncomputable def c := 2 ^ 0.8

theorem arrange_descending_order :
    b > a ∧ a > c := by
  sorry

end arrange_descending_order_l97_97550


namespace find_a2_plus_b2_l97_97760

theorem find_a2_plus_b2 (a b : ℝ) :
  (∀ x, |a * Real.sin x + b * Real.cos x - 1| + |b * Real.sin x - a * Real.cos x| ≤ 11)
  → a^2 + b^2 = 50 :=
by
  sorry

end find_a2_plus_b2_l97_97760


namespace smallest_sum_of_4_numbers_l97_97848

noncomputable def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

noncomputable def not_relatively_prime (a b : ℕ) : Prop :=
  ¬ relatively_prime a b

noncomputable def problem_statement : Prop :=
  ∃ (V1 V2 V3 V4 : ℕ), 
  relatively_prime V1 V3 ∧ 
  relatively_prime V2 V4 ∧ 
  not_relatively_prime V1 V2 ∧ 
  not_relatively_prime V1 V4 ∧ 
  not_relatively_prime V2 V3 ∧ 
  not_relatively_prime V3 V4 ∧ 
  V1 + V2 + V3 + V4 = 60

theorem smallest_sum_of_4_numbers : problem_statement := sorry

end smallest_sum_of_4_numbers_l97_97848


namespace cost_price_of_article_l97_97464

theorem cost_price_of_article (C : ℝ) (SP : ℝ) (C_new : ℝ) (SP_new : ℝ) :
  SP = 1.05 * C →
  C_new = 0.95 * C →
  SP_new = SP - 3 →
  SP_new = 1.045 * C →
  C = 600 :=
by
  intro h1 h2 h3 h4
  -- statement to be proved
  sorry

end cost_price_of_article_l97_97464


namespace total_square_miles_of_plains_l97_97260

-- Defining conditions
def region_east_of_b : ℕ := 200
def region_east_of_a : ℕ := region_east_of_b - 50

-- To test this statement in Lean 4
theorem total_square_miles_of_plains : region_east_of_a + region_east_of_b = 350 := by
  sorry

end total_square_miles_of_plains_l97_97260


namespace pants_to_shirts_ratio_l97_97811

-- Conditions
def shirts : ℕ := 4
def total_clothes : ℕ := 16

-- Given P as the number of pants and S as the number of shorts
variable (P S : ℕ)

-- State the conditions as hypotheses
axiom shorts_half_pants : S = P / 2
axiom total_clothes_condition : 4 + P + S = 16

-- Question: Prove that the ratio of pants to shirts is 2
theorem pants_to_shirts_ratio : P = 2 * shirts :=
by {
  -- insert proof steps here
  sorry
}

end pants_to_shirts_ratio_l97_97811


namespace intersection_A_complement_B_l97_97907

open Set

noncomputable def A : Set ℝ := {2, 3, 4, 5, 6}
noncomputable def B : Set ℝ := {x | x^2 - 8 * x + 12 >= 0}
noncomputable def complement_B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem intersection_A_complement_B :
  A ∩ complement_B = {3, 4, 5} :=
sorry

end intersection_A_complement_B_l97_97907


namespace ratio_of_boys_l97_97399

theorem ratio_of_boys (p : ℚ) (h : p = (3/5) * (1 - p)) : p = 3 / 8 := by
  sorry

end ratio_of_boys_l97_97399


namespace find_f_2_l97_97745

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- The statement to prove: if f is monotonically increasing and satisfies the functional equation
-- for all x, then f(2) = e^2 + 1.
theorem find_f_2
  (h_mono : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2)
  (h_eq : ∀ x : ℝ, f (f x - exp x) = exp 1 + 1) :
  f 2 = exp 2 + 1 := sorry

end find_f_2_l97_97745


namespace overall_gain_percentage_l97_97442

theorem overall_gain_percentage (cost_A cost_B cost_C sp_A sp_B sp_C : ℕ)
  (hA : cost_A = 1000)
  (hB : cost_B = 3000)
  (hC : cost_C = 6000)
  (hsA : sp_A = 2000)
  (hsB : sp_B = 4500)
  (hsC : sp_C = 8000) :
  ((sp_A + sp_B + sp_C - (cost_A + cost_B + cost_C) : ℝ) / (cost_A + cost_B + cost_C) * 100) = 45 :=
by sorry

end overall_gain_percentage_l97_97442


namespace train_length_is_correct_l97_97135

noncomputable def train_length (speed_kmph : ℝ) (time_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time_sec
  total_distance - bridge_length

theorem train_length_is_correct :
  train_length 60 20.99832013438925 240 = 110 :=
by
  sorry

end train_length_is_correct_l97_97135


namespace johns_total_pay_l97_97463

-- Define the given conditions
def lastYearBonus : ℝ := 10000
def CAGR : ℝ := 0.05
def numYears : ℕ := 1
def projectsCompleted : ℕ := 8
def bonusPerProject : ℝ := 2000
def thisYearSalary : ℝ := 200000

-- Define the calculation for the first part of the bonus using the CAGR formula
def firstPartBonus (presentValue : ℝ) (growthRate : ℝ) (years : ℕ) : ℝ :=
  presentValue * (1 + growthRate)^years

-- Define the calculation for the second part of the bonus
def secondPartBonus (numProjects : ℕ) (bonusPerProject : ℝ) : ℝ :=
  numProjects * bonusPerProject

-- Define the total pay calculation
def totalPay (salary : ℝ) (bonus1 : ℝ) (bonus2 : ℝ) : ℝ :=
  salary + bonus1 + bonus2

-- The proof statement, given the conditions, prove the total pay is $226,500
theorem johns_total_pay : totalPay thisYearSalary (firstPartBonus lastYearBonus CAGR numYears) (secondPartBonus projectsCompleted bonusPerProject) = 226500 := 
by
  -- insert proof here
  sorry

end johns_total_pay_l97_97463


namespace first_issue_pages_l97_97230

-- Define the conditions
def total_pages := 220
def pages_third_issue (x : ℕ) := x + 4

-- Statement of the problem
theorem first_issue_pages (x : ℕ) (hx : 3 * x + 4 = total_pages) : x = 72 :=
sorry

end first_issue_pages_l97_97230


namespace train_pass_tree_l97_97693

theorem train_pass_tree
  (L : ℝ) (S : ℝ) (conv_factor : ℝ) 
  (hL : L = 275)
  (hS : S = 90)
  (hconv : conv_factor = 5 / 18) :
  L / (S * conv_factor) = 11 :=
by
  sorry

end train_pass_tree_l97_97693


namespace determine_denominator_of_fraction_l97_97917

theorem determine_denominator_of_fraction (x : ℝ) (h : 57 / x = 0.0114) : x = 5000 :=
by
  sorry

end determine_denominator_of_fraction_l97_97917


namespace probability_diff_color_ball_l97_97638

variable (boxA : List String) (boxB : List String)
def P_A (boxA := ["white", "white", "red", "red", "black"]) (boxB := ["white", "white", "white", "white", "red", "red", "red", "black", "black"]) : ℚ := sorry

theorem probability_diff_color_ball :
  P_A boxA boxB = 29 / 50 :=
sorry

end probability_diff_color_ball_l97_97638


namespace jam_jars_weight_l97_97545

noncomputable def jars_weight 
    (initial_suitcase_weight : ℝ) 
    (perfume_weight_oz : ℝ) (num_perfume : ℕ)
    (chocolate_weight_lb : ℝ)
    (soap_weight_oz : ℝ) (num_soap : ℕ)
    (total_return_weight : ℝ)
    (oz_to_lb : ℝ) : ℝ :=
  initial_suitcase_weight 
  + (num_perfume * perfume_weight_oz) / oz_to_lb 
  + chocolate_weight_lb 
  + (num_soap * soap_weight_oz) / oz_to_lb

theorem jam_jars_weight
    (initial_suitcase_weight : ℝ := 5)
    (perfume_weight_oz : ℝ := 1.2) (num_perfume : ℕ := 5)
    (chocolate_weight_lb : ℝ := 4)
    (soap_weight_oz : ℝ := 5) (num_soap : ℕ := 2)
    (total_return_weight : ℝ := 11)
    (oz_to_lb : ℝ := 16) :
    jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb + (jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb) = 1 :=
by
  sorry

end jam_jars_weight_l97_97545


namespace intersecting_to_quadrilateral_l97_97097

-- Define the geometric solids
inductive GeometricSolid
| cone : GeometricSolid
| sphere : GeometricSolid
| cylinder : GeometricSolid

-- Define a function that checks if intersecting a given solid with a plane can produce a quadrilateral
def can_intersect_to_quadrilateral (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.cone => false
  | GeometricSolid.sphere => false
  | GeometricSolid.cylinder => true

-- State the theorem
theorem intersecting_to_quadrilateral (solid : GeometricSolid) :
  can_intersect_to_quadrilateral solid ↔ solid = GeometricSolid.cylinder :=
sorry

end intersecting_to_quadrilateral_l97_97097


namespace compare_times_l97_97313

variable {v : ℝ} (h_v_pos : 0 < v)

/-- 
  Jones covered a distance of 80 miles on his first trip at speed v.
  On a later trip, he traveled 360 miles at four times his original speed.
  Prove that his new time is (9/8) times his original time.
-/
theorem compare_times :
  let t1 := 80 / v
  let t2 := 360 / (4 * v)
  t2 = (9 / 8) * t1 :=
by
  sorry

end compare_times_l97_97313


namespace tan_ratio_proof_l97_97224

theorem tan_ratio_proof (α : ℝ) (h : 5 * Real.sin (2 * α) = Real.sin 2) : 
  Real.tan (α + 1 * Real.pi / 180) / Real.tan (α - 1 * Real.pi / 180) = - 3 / 2 := 
sorry

end tan_ratio_proof_l97_97224


namespace midpoint_polar_coordinates_l97_97412

noncomputable def polar_midpoint :=
  let A := (10, 7 * Real.pi / 6)
  let B := (10, 11 * Real.pi / 6)
  let A_cartesian := (10 * Real.cos (7 * Real.pi / 6), 10 * Real.sin (7 * Real.pi / 6))
  let B_cartesian := (10 * Real.cos (11 * Real.pi / 6), 10 * Real.sin (11 * Real.pi / 6))
  let midpoint_cartesian := ((A_cartesian.1 + B_cartesian.1) / 2, (A_cartesian.2 + B_cartesian.2) / 2)
  let r := Real.sqrt (midpoint_cartesian.1 ^ 2 + midpoint_cartesian.2 ^ 2)
  let θ := if midpoint_cartesian.1 = 0 then 0 else Real.arctan (midpoint_cartesian.2 / midpoint_cartesian.1)
  (r, θ)

theorem midpoint_polar_coordinates :
  polar_midpoint = (5 * Real.sqrt 3, Real.pi) := by
  sorry

end midpoint_polar_coordinates_l97_97412


namespace find_n_l97_97339

variable (x n : ℝ)

-- Definitions
def positive (x : ℝ) : Prop := x > 0
def equation (x n : ℝ) : Prop := x / n + x / 25 = 0.06 * x

-- Theorem statement
theorem find_n (h1 : positive x) (h2 : equation x n) : n = 50 :=
sorry

end find_n_l97_97339


namespace quadratic_eq1_solution_quadratic_eq2_solution_l97_97294

-- Define the first problem and its conditions
theorem quadratic_eq1_solution :
  ∀ x : ℝ, 4 * x^2 + x - (1 / 2) = 0 ↔ (x = -1 / 2 ∨ x = 1 / 4) :=
by
  -- The proof is omitted
  sorry

-- Define the second problem and its conditions
theorem quadratic_eq2_solution :
  ∀ y : ℝ, (y - 2) * (y + 3) = 6 ↔ (y = -4 ∨ y = 3) :=
by
  -- The proof is omitted
  sorry

end quadratic_eq1_solution_quadratic_eq2_solution_l97_97294


namespace average_of_remaining_two_l97_97374

theorem average_of_remaining_two (S S3 : ℚ) (h1 : S / 5 = 6) (h2 : S3 / 3 = 4) : (S - S3) / 2 = 9 :=
by
  sorry

end average_of_remaining_two_l97_97374


namespace panels_per_home_panels_needed_per_home_l97_97471

theorem panels_per_home (P : ℕ) (total_homes : ℕ) (shortfall : ℕ) (homes_installed : ℕ) :
  total_homes = 20 →
  shortfall = 50 →
  homes_installed = 15 →
  (P - shortfall) / homes_installed = P / total_homes →
  P = 200 :=
by
  intro h1 h2 h3 h4
  sorry

theorem panels_needed_per_home :
  (200 / 20) = 10 :=
by
  sorry

end panels_per_home_panels_needed_per_home_l97_97471


namespace payment_denotation_is_correct_l97_97843

-- Define the initial condition of receiving money
def received_amount : ℤ := 120

-- Define the payment amount
def payment_amount : ℤ := 85

-- The expected payoff
def expected_payment_denotation : ℤ := -85

-- Theorem stating that the payment should be denoted as -85 yuan
theorem payment_denotation_is_correct : (payment_amount = -expected_payment_denotation) :=
by
  sorry

end payment_denotation_is_correct_l97_97843


namespace sum_of_ages_is_29_l97_97198

theorem sum_of_ages_is_29 (age1 age2 age3 : ℕ) (h1 : age1 = 9) (h2 : age2 = 9) (h3 : age3 = 11) :
  age1 + age2 + age3 = 29 := by
  -- skipping the proof
  sorry

end sum_of_ages_is_29_l97_97198


namespace power_evaluation_l97_97777

theorem power_evaluation (x : ℕ) (h1 : 3^x = 81) : 3^(x+2) = 729 := by
  sorry

end power_evaluation_l97_97777


namespace divisibility_equiv_l97_97847

-- Definition of the functions a(n) and b(n)
def a (n : ℕ) := n^5 + 5^n
def b (n : ℕ) := n^5 * 5^n + 1

-- Define a positive integer
variables (n : ℕ) (hn : n > 0)

-- The theorem stating the equivalence
theorem divisibility_equiv : (a n) % 11 = 0 ↔ (b n) % 11 = 0 :=
sorry
 
end divisibility_equiv_l97_97847


namespace fib_divisibility_l97_97769

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

theorem fib_divisibility (m n : ℕ) (hm : 1 ≤ m) (hn : 1 < n) : 
  (fib (m * n - 1) - fib (n - 1) ^ m) % fib n ^ 2 = 0 :=
sorry

end fib_divisibility_l97_97769


namespace expression_value_l97_97616

theorem expression_value (x : ℤ) (h : x = -2) : x ^ 2 + 6 * x - 8 = -16 := 
by 
  rw [h]
  sorry

end expression_value_l97_97616


namespace operation_preserves_remainder_l97_97627

theorem operation_preserves_remainder (N : ℤ) (k : ℤ) (m : ℤ) 
(f : ℤ → ℤ) (hN : N = 6 * k + 3) (hf : f N = 6 * m + 3) : f N % 6 = 3 :=
by
  sorry

end operation_preserves_remainder_l97_97627


namespace mika_initial_stickers_l97_97250

theorem mika_initial_stickers :
  let store_stickers := 26.0
  let birthday_stickers := 20.0 
  let sister_stickers := 6.0 
  let mother_stickers := 58.0 
  let total_stickers := 130.0 
  ∃ x : Real, x + store_stickers + birthday_stickers + sister_stickers + mother_stickers = total_stickers ∧ x = 20.0 := 
by 
  sorry

end mika_initial_stickers_l97_97250


namespace divides_iff_l97_97403

open Int

theorem divides_iff (n m : ℤ) : (9 ∣ (2 * n + 5 * m)) ↔ (9 ∣ (5 * n + 8 * m)) := 
sorry

end divides_iff_l97_97403


namespace count_households_in_apartment_l97_97069

noncomputable def total_households 
  (houses_left : ℕ)
  (houses_right : ℕ)
  (floors_above : ℕ)
  (floors_below : ℕ) 
  (households_per_house : ℕ) : ℕ :=
(houses_left + houses_right) * (floors_above + floors_below) * households_per_house

theorem count_households_in_apartment : 
  ∀ (houses_left houses_right floors_above floors_below households_per_house : ℕ),
  houses_left = 1 →
  houses_right = 6 →
  floors_above = 1 →
  floors_below = 3 →
  households_per_house = 3 →
  total_households houses_left houses_right floors_above floors_below households_per_house = 105 :=
by
  intros houses_left houses_right floors_above floors_below households_per_house hl hr fa fb hh
  rw [hl, hr, fa, fb, hh]
  unfold total_households
  norm_num
  sorry

end count_households_in_apartment_l97_97069


namespace correct_time_fraction_l97_97466

theorem correct_time_fraction :
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  (correct_hours * correct_minutes_per_hour : ℝ) / (hours * minutes_per_hour) = (5 / 36 : ℝ) :=
by
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  sorry

end correct_time_fraction_l97_97466


namespace complement_union_correct_l97_97661

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_correct :
  ((U \ A) ∪ B) = {0, 2, 3, 6} :=
by
  sorry

end complement_union_correct_l97_97661


namespace rate_is_900_l97_97522

noncomputable def rate_per_square_meter (L W : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (L * W)

theorem rate_is_900 :
  rate_per_square_meter 5 4.75 21375 = 900 := by
  sorry

end rate_is_900_l97_97522


namespace det_A_is_neg9_l97_97120

noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 5], ![6, -3]]

theorem det_A_is_neg9 : Matrix.det A = -9 := 
by 
  sorry

end det_A_is_neg9_l97_97120


namespace function_value_corresponds_to_multiple_independent_variables_l97_97386

theorem function_value_corresponds_to_multiple_independent_variables
  {α β : Type*} (f : α → β) :
  ∃ (b : β), ∃ (a1 a2 : α), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
sorry

end function_value_corresponds_to_multiple_independent_variables_l97_97386


namespace speaker_discounted_price_correct_l97_97212

-- Define the initial price and the discount
def initial_price : ℝ := 475.00
def discount : ℝ := 276.00

-- Define the discounted price
def discounted_price : ℝ := initial_price - discount

-- The theorem to prove that the discounted price is 199.00
theorem speaker_discounted_price_correct : discounted_price = 199.00 :=
by
  -- Proof is omitted here, adding sorry to indicate it.
  sorry

end speaker_discounted_price_correct_l97_97212


namespace average_height_of_students_l97_97012

theorem average_height_of_students (x : ℕ) (female_height male_height : ℕ) 
  (female_height_eq : female_height = 170) (male_height_eq : male_height = 185) 
  (ratio : 2 * x = x * 2) : 
  ((2 * x * male_height + x * female_height) / (2 * x + x) = 180) := 
by
  sorry

end average_height_of_students_l97_97012


namespace interest_rate_l97_97351

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r/100)^t - P

theorem interest_rate (P t : ℝ) (diff : ℝ) (r : ℝ) (h : P = 1000) (t_eq : t = 4) 
  (diff_eq : diff = 64.10) : 
  compound_interest P r t - simple_interest P r t = diff → r = 10 :=
by
  sorry

end interest_rate_l97_97351


namespace price_of_two_identical_filters_l97_97066

def price_of_individual_filters (x : ℝ) : Prop :=
  let total_individual := 2 * 14.05 + 19.50 + 2 * x
  total_individual = 87.50 / 0.92

theorem price_of_two_identical_filters
  (h1 : price_of_individual_filters 23.76) :
  23.76 * 2 + 28.10 + 19.50 = 87.50 / 0.92 :=
by sorry

end price_of_two_identical_filters_l97_97066


namespace find_missing_number_l97_97974

theorem find_missing_number (x : ℕ) (h : (1 + x + 23 + 24 + 25 + 26 + 27 + 2) / 8 = 20) : x = 32 := 
by sorry

end find_missing_number_l97_97974


namespace taxi_fare_distance_l97_97702

theorem taxi_fare_distance (x : ℝ) : 
  (8 + if x ≤ 3 then 0 else if x ≤ 8 then 2.15 * (x - 3) else 2.15 * 5 + 2.85 * (x - 8)) + 1 = 31.15 → x = 11.98 :=
by 
  sorry

end taxi_fare_distance_l97_97702


namespace factor_expression_l97_97692

theorem factor_expression (x : ℝ) : 92 * x^3 - 184 * x^6 = 92 * x^3 * (1 - 2 * x^3) :=
by
  sorry

end factor_expression_l97_97692


namespace quadratic_expression_neg_for_all_x_l97_97258

theorem quadratic_expression_neg_for_all_x (m : ℝ) :
  (∀ x : ℝ, m*x^2 + (m-1)*x + (m-1) < 0) ↔ m < -1/3 :=
sorry

end quadratic_expression_neg_for_all_x_l97_97258


namespace find_value_l97_97749

variable (number : ℝ) (V : ℝ)

theorem find_value
  (h1 : number = 8)
  (h2 : 0.75 * number + V = 8) : V = 2 := by
  sorry

end find_value_l97_97749


namespace find_f_expression_find_f_range_l97_97057

noncomputable def y (t x : ℝ) : ℝ := 1 - 2 * t - 2 * t * x + 2 * x ^ 2

noncomputable def f (t : ℝ) : ℝ := 
  if t < -2 then 3 
  else if t > 2 then -4 * t + 3 
  else -t ^ 2 / 2 - 2 * t + 1

theorem find_f_expression (t : ℝ) : 
  f t = if t < -2 then 3 else 
          if t > 2 then -4 * t + 3 
          else - t ^ 2 / 2 - 2 * t + 1 :=
sorry

theorem find_f_range (t : ℝ) (ht : -2 ≤ t ∧ t ≤ 0) : 
  1 ≤ f t ∧ f t ≤ 3 := 
sorry

end find_f_expression_find_f_range_l97_97057


namespace general_formula_arithmetic_sequence_l97_97726

theorem general_formula_arithmetic_sequence :
  (∃ (a_n : ℕ → ℕ) (d : ℕ), d ≠ 0 ∧ 
    (a_2 = a_1 + d) ∧ 
    (a_4 = a_1 + 3 * d) ∧ 
    (a_2^2 = a_1 * a_4) ∧
    (a_5 = a_1 + 4 * d) ∧ 
    (a_6 = a_1 + 5 * d) ∧ 
    (a_5 + a_6 = 11) ∧ 
    ∀ n, a_n = a_1 + (n - 1) * d) → 
  ∀ n, a_n = n := 
sorry

end general_formula_arithmetic_sequence_l97_97726


namespace lunch_to_novel_ratio_l97_97750

theorem lunch_to_novel_ratio 
  (initial_amount : ℕ) 
  (novel_cost : ℕ) 
  (remaining_after_mall : ℕ) 
  (spent_on_lunch : ℕ)
  (h1 : initial_amount = 50) 
  (h2 : novel_cost = 7) 
  (h3 : remaining_after_mall = 29) 
  (h4 : spent_on_lunch = initial_amount - novel_cost - remaining_after_mall) :
  spent_on_lunch / novel_cost = 2 := 
  sorry

end lunch_to_novel_ratio_l97_97750


namespace factorize_expression_l97_97289

theorem factorize_expression (a m n : ℝ) : a * m^2 - 2 * a * m * n + a * n^2 = a * (m - n)^2 :=
by
  sorry

end factorize_expression_l97_97289


namespace guarantee_min_points_l97_97605

-- Define points for positions
def points_for_position (pos : ℕ) : ℕ :=
  if pos = 1 then 6
  else if pos = 2 then 4
  else if pos = 3 then 2
  else 0

-- Define the maximum points
def max_points_per_race := 6
def races := 4
def max_points := max_points_per_race * races

-- Define the condition of no ties
def no_ties := true

-- Define the problem statement
theorem guarantee_min_points (no_ties: true) (h1: points_for_position 1 = 6)
  (h2: points_for_position 2 = 4) (h3: points_for_position 3 = 2)
  (h4: max_points = 24) : 
  ∃ min_points, (min_points = 22) ∧ (∀ points, (points < min_points) → (∃ another_points, (another_points > points))) :=
  sorry

end guarantee_min_points_l97_97605


namespace letter_150_in_pattern_l97_97689

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l97_97689


namespace curtains_length_needed_l97_97947

def room_height_feet : ℕ := 8
def additional_material_inches : ℕ := 5

def height_in_inches : ℕ := room_height_feet * 12

def total_length_curtains : ℕ := height_in_inches + additional_material_inches

theorem curtains_length_needed : total_length_curtains = 101 := by
  sorry

end curtains_length_needed_l97_97947


namespace problem1_problem2_l97_97265

-- Define the sets of balls and boxes
inductive Ball
| ball1 | ball2 | ball3 | ball4

inductive Box
| boxA | boxB | boxC

-- Define the arrangements for the first problem
def arrangements_condition1 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball3 = Box.boxB) ∧
  (∃ b1 b2 b3 : Box, b1 ≠ b2 ∧ b2 ≠ b3 ∧ b3 ≠ b1 ∧ 
    ∃ (f : Ball → Box), 
      (f Ball.ball1 = b1) ∧ (f Ball.ball2 = b2) ∧ (f Ball.ball3 = Box.boxB) ∧ (f Ball.ball4 = b3))

-- Define the proof statement for the first problem
theorem problem1 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition1 arrangement → n = 7) :=
sorry

-- Define the arrangements for the second problem
def arrangements_condition2 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball1 ≠ Box.boxA) ∧
  (arrangement Ball.ball2 ≠ Box.boxB)

-- Define the proof statement for the second problem
theorem problem2 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition2 arrangement → n = 36) :=
sorry

end problem1_problem2_l97_97265


namespace sqrt_eq_l97_97966

noncomputable def sqrt_22500 := 150

theorem sqrt_eq (h : sqrt_22500 = 150) : Real.sqrt 0.0225 = 0.15 :=
sorry

end sqrt_eq_l97_97966


namespace line_always_passes_through_fixed_point_l97_97288

theorem line_always_passes_through_fixed_point :
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = m * x + 2 * m + 1) ∧ (x = -2) ∧ (y = 1) :=
by
  sorry

end line_always_passes_through_fixed_point_l97_97288


namespace envelope_weight_l97_97960

theorem envelope_weight :
  (7.225 * 1000) / 850 = 8.5 :=
by
  sorry

end envelope_weight_l97_97960


namespace factor_tree_X_value_l97_97555

-- Define the constants
def F : ℕ := 5 * 3
def G : ℕ := 7 * 3

-- Define the intermediate values
def Y : ℕ := 5 * F
def Z : ℕ := 7 * G

-- Final value of X
def X : ℕ := Y * Z

-- Prove the value of X
theorem factor_tree_X_value : X = 11025 := by
  sorry

end factor_tree_X_value_l97_97555


namespace least_number_to_add_l97_97873

theorem least_number_to_add (k : ℕ) (h : 1019 % 25 = 19) : (1019 + k) % 25 = 0 ↔ k = 6 :=
by
  sorry

end least_number_to_add_l97_97873


namespace solution_set_of_inequality_l97_97855

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 4 * x - 3 > 0 } = { x : ℝ | 1 < x ∧ x < 3 } := sorry

end solution_set_of_inequality_l97_97855


namespace range_of_a_l97_97562

theorem range_of_a (a x y : ℝ)
  (h1 : x + y = 3 * a + 4)
  (h2 : x - y = 7 * a - 4)
  (h3 : 3 * x - 2 * y < 11) : a < 1 :=
sorry

end range_of_a_l97_97562


namespace derivative_at_1_l97_97421

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1 : (deriv f 1) = 2 * Real.log 2 - 3 := 
sorry

end derivative_at_1_l97_97421


namespace sum_series_eq_l97_97149

theorem sum_series_eq : 
  (∑' k : ℕ, (k + 1) * (1/4)^(k + 1)) = 4 / 9 :=
by sorry

end sum_series_eq_l97_97149


namespace smallest_digit_never_in_units_place_of_odd_numbers_l97_97366

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l97_97366


namespace two_legged_birds_count_l97_97393

def count_birds (b m i : ℕ) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 6 * i = 680 → b = 280

theorem two_legged_birds_count : ∃ b m i : ℕ, count_birds b m i :=
by
  have h1 : count_birds 280 0 20 := sorry
  exact ⟨280, 0, 20, h1⟩

end two_legged_birds_count_l97_97393


namespace smallest_numbers_l97_97927

-- Define the problem statement
theorem smallest_numbers (m n : ℕ) :
  (∃ (m1 n1 m2 n2 : ℕ), 7 * m1^2 - 11 * n1^2 = 1 ∧ 7 * m2^2 - 11 * n2^2 = 5) ↔
  (7 * m^2 - 11 * n^2 = 1) ∨ (7 * m^2 - 11 * n^2 = 5) :=
by
  sorry

end smallest_numbers_l97_97927


namespace large_paintings_count_l97_97469

-- Define the problem conditions
def paint_per_large : Nat := 3
def paint_per_small : Nat := 2
def small_paintings : Nat := 4
def total_paint : Nat := 17

-- Question to find number of large paintings (L)
theorem large_paintings_count :
  ∃ L : Nat, (paint_per_large * L + paint_per_small * small_paintings = total_paint) → L = 3 :=
by
  -- Placeholder for the proof
  sorry

end large_paintings_count_l97_97469


namespace sin_identity_l97_97450

open Real

noncomputable def alpha : ℝ := π  -- since we are considering angles in radians

theorem sin_identity (h1 : sin α = 3/5) (h2 : π/2 < α ∧ α < 3 * π / 2) :
  sin (5 * π / 2 - α) = -4 / 5 :=
by sorry

end sin_identity_l97_97450


namespace amount_invested_l97_97133

variables (P y : ℝ)

-- Conditions
def condition1 : Prop := 800 = P * (2 * y) / 100
def condition2 : Prop := 820 = P * ((1 + y / 100) ^ 2 - 1)

-- The proof we seek
theorem amount_invested (h1 : condition1 P y) (h2 : condition2 P y) : P = 8000 :=
by
  -- Place the proof here
  sorry

end amount_invested_l97_97133


namespace exists_adj_diff_gt_3_max_min_adj_diff_l97_97247
-- Import needed libraries

-- Definition of the given problem and statement of the parts (a) and (b)

-- Part (a)
theorem exists_adj_diff_gt_3 (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∃ i j : Fin 18, adj i j ∧ |arrangement i - arrangement j| > 3) :=
sorry

-- Part (b)
theorem max_min_adj_diff (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∀ i j : Fin 18, adj i j → |arrangement i - arrangement j| ≥ 6) :=
sorry

end exists_adj_diff_gt_3_max_min_adj_diff_l97_97247


namespace sqrt_expr_evaluation_l97_97276

theorem sqrt_expr_evaluation : 
  (Real.sqrt 24) - 3 * (Real.sqrt (1 / 6)) + (Real.sqrt 6) = (5 * Real.sqrt 6) / 2 :=
by
  sorry

end sqrt_expr_evaluation_l97_97276


namespace tangent_line_equation_l97_97798

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 5

def point_A : ℝ × ℝ := (1, -2)

theorem tangent_line_equation :
  ∀ x y : ℝ, (y = 4 * x - 6) ↔ (fderiv ℝ f (point_A.1) x = 4) ∧ (y = f (point_A.1) + 4 * (x - point_A.1)) := by
  sorry

end tangent_line_equation_l97_97798


namespace f_1997_leq_666_l97_97031

noncomputable def f : ℕ+ → ℕ := sorry

axiom f_mn_inequality : ∀ (m n : ℕ+), f (m + n) ≥ f m + f n
axiom f_two : f 2 = 0
axiom f_three_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

theorem f_1997_leq_666 : f 1997 ≤ 666 := sorry

end f_1997_leq_666_l97_97031


namespace order_of_numbers_l97_97187

theorem order_of_numbers (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : 
  -m > n ∧ n > -n ∧ -n > m := 
by
  sorry

end order_of_numbers_l97_97187


namespace ratio_of_ages_l97_97051

theorem ratio_of_ages (age_saras age_kul : ℕ) (h_saras : age_saras = 33) (h_kul : age_kul = 22) : 
  age_saras / Nat.gcd age_saras age_kul = 3 ∧ age_kul / Nat.gcd age_saras age_kul = 2 :=
by
  sorry

end ratio_of_ages_l97_97051


namespace net_profit_is_correct_l97_97099

-- Define the known quantities
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.20
def markup : ℝ := 45

-- Define the derived quantities based on the conditions
def overhead : ℝ := overhead_percentage * purchase_price
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + markup
def net_profit : ℝ := selling_price - total_cost

-- The statement to prove
theorem net_profit_is_correct : net_profit = 45 := by
  sorry

end net_profit_is_correct_l97_97099


namespace sequence_properties_l97_97817

variable {Seq : Nat → ℕ}
-- Given conditions: Sn = an(an + 3) / 6
def Sn (n : ℕ) := Seq n * (Seq n + 3) / 6

theorem sequence_properties :
  (Seq 1 = 3) ∧ (Seq 2 = 9) ∧ (∀ n : ℕ, Seq (n+1) = 3 * (n + 1)) :=
by 
  have h1 : Sn 1 = (Seq 1 * (Seq 1 + 3)) / 6 := rfl
  have h2 : Sn 2 = (Seq 2 * (Seq 2 + 3)) / 6 := rfl
  sorry

end sequence_properties_l97_97817


namespace one_cow_one_bag_days_l97_97523

-- Definitions based on conditions in a)
def cows : ℕ := 60
def bags : ℕ := 75
def days_total : ℕ := 45

-- Main statement for the proof problem
theorem one_cow_one_bag_days : 
  (cows : ℝ) * (bags : ℝ) / (days_total : ℝ) = 1 / 36 := 
by
  sorry   -- Proof placeholder

end one_cow_one_bag_days_l97_97523


namespace medians_formula_l97_97511

noncomputable def ma (a b c : ℝ) : ℝ := (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2))
noncomputable def mb (a b c : ℝ) : ℝ := (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2))
noncomputable def mc (a b c : ℝ) : ℝ := (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2))

theorem medians_formula (a b c : ℝ) :
  ma a b c = (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2)) ∧
  mb a b c = (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2)) ∧
  mc a b c = (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2)) :=
by sorry

end medians_formula_l97_97511


namespace jessica_age_l97_97544

theorem jessica_age 
  (j g : ℚ)
  (h1 : g = 15 * j) 
  (h2 : g - j = 60) : 
  j = 30 / 7 :=
by
  sorry

end jessica_age_l97_97544


namespace S_10_eq_110_l97_97284

-- Conditions
def a (n : ℕ) : ℕ := sorry  -- Assuming general term definition of arithmetic sequence
def S (n : ℕ) : ℕ := sorry  -- Assuming sum definition of arithmetic sequence

axiom a_3_eq_16 : a 3 = 16
axiom S_20_eq_20 : S 20 = 20

-- Prove
theorem S_10_eq_110 : S 10 = 110 :=
  by
  sorry

end S_10_eq_110_l97_97284


namespace vector_solution_l97_97337

theorem vector_solution :
  let u := -6 / 41
  let v := -46 / 41
  let vec1 := (⟨3, -2⟩: ℝ × ℝ)
  let vec2 := (⟨5, -7⟩: ℝ × ℝ)
  let vec3 := (⟨0, 3⟩: ℝ × ℝ)
  let vec4 := (⟨-3, 4⟩: ℝ × ℝ)
  (vec1 + u • vec2 = vec3 + v • vec4) := by
  sorry

end vector_solution_l97_97337


namespace prove_f2_l97_97458

def func_condition (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x ^ 2 - y) + 2 * c * f x * y

theorem prove_f2 (c : ℝ) (f : ℝ → ℝ)
  (hf : func_condition f c) :
  (f 2 = 0 ∨ f 2 = 4) ∧ (2 * (if f 2 = 0 then 4 else if f 2 = 4 then 4 else 0) = 8) :=
by {
  sorry
}

end prove_f2_l97_97458


namespace additional_bags_at_max_weight_l97_97632

/-
Constants representing the problem conditions.
-/
def num_people : Nat := 6
def bags_per_person : Nat := 5
def max_weight_per_bag : Nat := 50
def total_weight_capacity : Nat := 6000

/-
Calculate the total existing luggage weight.
-/
def total_existing_bags : Nat := num_people * bags_per_person
def total_existing_weight : Nat := total_existing_bags * max_weight_per_bag
def remaining_weight_capacity : Nat := total_weight_capacity - total_existing_weight

/-
The proof statement asserting that given the conditions, 
the airplane can hold 90 more bags at maximum weight.
-/
theorem additional_bags_at_max_weight : remaining_weight_capacity / max_weight_per_bag = 90 := by
  sorry

end additional_bags_at_max_weight_l97_97632


namespace trapezoid_shorter_base_l97_97362

theorem trapezoid_shorter_base (a b : ℕ) (mid_segment : ℕ) (longer_base : ℕ) 
    (h1 : mid_segment = 5) (h2 : longer_base = 105) 
    (h3 : mid_segment = (longer_base - a) / 2) : 
  a = 95 := 
by
  sorry

end trapezoid_shorter_base_l97_97362


namespace angle_D_measure_l97_97261

theorem angle_D_measure 
  (A B C D : Type)
  (angleA : ℝ)
  (angleB : ℝ)
  (angleC : ℝ)
  (angleD : ℝ)
  (BD_bisector : ℝ → ℝ) :
  angleA = 85 ∧ angleB = 50 ∧ angleC = 25 ∧ BD_bisector angleB = 25 →
  angleD = 130 :=
by
  intro h
  have hA := h.1
  have hB := h.2.1
  have hC := h.2.2.1
  have hBD := h.2.2.2
  sorry

end angle_D_measure_l97_97261


namespace age_double_in_years_l97_97416

theorem age_double_in_years (S M X: ℕ) (h1: M = S + 22) (h2: S = 20) (h3: M + X = 2 * (S + X)) : X = 2 :=
by 
  sorry

end age_double_in_years_l97_97416


namespace proof_problem_l97_97460

open Set

variable {R : Set ℝ} (A B : Set ℝ) (complement_B : Set ℝ)

-- Defining set A
def setA : Set ℝ := { x | 1 < x ∧ x < 3 }

-- Defining set B based on the given functional relationship
def setB : Set ℝ := { x | 2 < x } 

-- Defining the complement of set B (in the universal set R)
def complementB : Set ℝ := { x | x ≤ 2 }

-- The intersection we need to prove is equivalent to the given answer
def intersection_result : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- The theorem statement (no proof)
theorem proof_problem : setA ∩ complementB = intersection_result := 
by
  sorry

end proof_problem_l97_97460


namespace carolyn_practice_time_l97_97081

theorem carolyn_practice_time :
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  monthly_minutes_total = 1920 :=
by
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  sorry

end carolyn_practice_time_l97_97081


namespace julia_mile_time_l97_97571

variable (x : ℝ)

theorem julia_mile_time
  (h1 : ∀ x, x > 0)
  (h2 : ∀ x, x <= 13)
  (h3 : 65 = 5 * 13)
  (h4 : 50 = 65 - 15)
  (h5 : 50 = 5 * x) :
  x = 10 := by
  sorry

end julia_mile_time_l97_97571


namespace avg_first_six_results_l97_97997

theorem avg_first_six_results (average_11 : ℕ := 52) (average_last_6 : ℕ := 52) (sixth_result : ℕ := 34) :
  ∃ A : ℕ, (6 * A + 6 * average_last_6 - sixth_result = 11 * average_11) ∧ A = 49 :=
by
  sorry

end avg_first_six_results_l97_97997


namespace net_effect_on_sale_value_l97_97013

theorem net_effect_on_sale_value 
  (P Original_Sales_Volume : ℝ) 
  (reduced_by : ℝ := 0.18) 
  (sales_increase : ℝ := 0.88) 
  (additional_tax : ℝ := 0.12) :
  P * Original_Sales_Volume * ((1 - reduced_by) * (1 + additional_tax) * (1 + sales_increase) - 1) = P * Original_Sales_Volume * 0.7184 :=
  by
  sorry

end net_effect_on_sale_value_l97_97013


namespace decimal_to_fraction_l97_97574

theorem decimal_to_fraction (x : ℚ) (h : x = 3.68) : x = 92 / 25 := by
  sorry

end decimal_to_fraction_l97_97574


namespace gcd_45_75_l97_97266

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l97_97266


namespace smallest_m_l97_97249

-- Definitions of lengths and properties of the pieces
variable {lengths : Fin 21 → ℝ} 
variable (h_all_pos : ∀ i, lengths i > 0)
variable (h_total_length : (Finset.univ : Finset (Fin 21)).sum lengths = 21)
variable (h_max_factor : ∀ i j, max (lengths i) (lengths j) ≤ 3 * min (lengths i) (lengths j))

-- Proof statement
theorem smallest_m (m : ℝ) (hm : ∀ i j, max (lengths i) (lengths j) ≤ m * min (lengths i) (lengths j)) : 
  m ≥ 1 := 
sorry

end smallest_m_l97_97249


namespace Julio_limes_expense_l97_97606

/-- Julio's expense on limes after 30 days --/
theorem Julio_limes_expense :
  ((30 * (1 / 2)) / 3) * 1 = 5 := 
by
  sorry

end Julio_limes_expense_l97_97606


namespace graph_passes_through_point_l97_97311

theorem graph_passes_through_point (a : ℝ) (ha : 0 < a) (ha_ne_one : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 3) + 2
  f 3 = 3 := by
  sorry

end graph_passes_through_point_l97_97311


namespace max_intersections_circle_quadrilateral_max_intersections_correct_l97_97453

-- Define the intersection property of a circle and a line segment
def max_intersections_per_side (circle : Type) (line_segment : Type) : ℕ := 2

-- Define a quadrilateral as a shape having four sides
def sides_of_quadrilateral : ℕ := 4

-- The theorem stating the maximum number of intersection points between a circle and a quadrilateral
theorem max_intersections_circle_quadrilateral (circle : Type) (quadrilateral : Type) : Prop :=
  max_intersections_per_side circle quadrilateral * sides_of_quadrilateral = 8

-- Proof is skipped with 'sorry'
theorem max_intersections_correct (circle : Type) (quadrilateral : Type) :
  max_intersections_circle_quadrilateral circle quadrilateral :=
by
  sorry

end max_intersections_circle_quadrilateral_max_intersections_correct_l97_97453


namespace fido_leash_problem_l97_97357

theorem fido_leash_problem
  (r : ℝ) 
  (octagon_area : ℝ := 2 * r^2 * Real.sqrt 2)
  (circle_area : ℝ := Real.pi * r^2)
  (explore_fraction : ℝ := circle_area / octagon_area)
  (a b : ℝ) 
  (h_simplest_form : explore_fraction = (Real.sqrt a) / b * Real.pi)
  (h_a : a = 2)
  (h_b : b = 2) : a * b = 4 :=
by sorry

end fido_leash_problem_l97_97357


namespace infinite_solutions_to_congruence_l97_97368

theorem infinite_solutions_to_congruence :
  ∃ᶠ n in atTop, 3^((n-2)^(n-1)-1) ≡ 1 [MOD 17 * n^2] :=
by
  sorry

end infinite_solutions_to_congruence_l97_97368


namespace simplify_expression_l97_97846

theorem simplify_expression : 
  (1 / ((1 / ((1 / 2)^1)) + (1 / ((1 / 2)^3)) + (1 / ((1 / 2)^4)))) = (1 / 26) := 
by 
  sorry

end simplify_expression_l97_97846


namespace value_of_v_star_star_l97_97763

noncomputable def v_star (v : ℝ) : ℝ :=
  v - v / 3
  
theorem value_of_v_star_star (v : ℝ) (h : v = 8.999999999999998) : v_star (v_star v) = 4.000000000000000 := by
  sorry

end value_of_v_star_star_l97_97763


namespace problem_equiv_l97_97108

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_equiv (x y : ℝ) : dollar ((2 * x + y) ^ 2) ((x - 2 * y) ^ 2) = (3 * x ^ 2 + 8 * x * y - 3 * y ^ 2) ^ 2 := by
  sorry

end problem_equiv_l97_97108


namespace value_of_3b_minus_a_l97_97928

theorem value_of_3b_minus_a :
  ∃ (a b : ℕ), (a > b) ∧ (a >= 0) ∧ (b >= 0) ∧ (∀ x : ℝ, (x - a) * (x - b) = x^2 - 16 * x + 60) ∧ (3 * b - a = 8) := 
sorry

end value_of_3b_minus_a_l97_97928


namespace percentage_of_invalid_votes_l97_97858

theorem percentage_of_invalid_votes:
  ∃ (A B V I VV : ℕ), 
    V = 5720 ∧
    B = 1859 ∧
    A = B + 15 / 100 * V ∧
    VV = A + B ∧
    V = VV + I ∧
    (I: ℚ) / V * 100 = 20 :=
by
  sorry

end percentage_of_invalid_votes_l97_97858


namespace minimum_value_frac_l97_97390

theorem minimum_value_frac (a b : ℝ) (h₁ : 2 * a - b + 2 * 0 = 0) 
  (h₂ : a > 0) (h₃ : b > 0) (h₄ : a + b = 1) : 
  (1 / a) + (1 / b) = 4 :=
sorry

end minimum_value_frac_l97_97390


namespace cost_of_600_candies_l97_97903

-- Definitions based on conditions
def costOfBox : ℕ := 6       -- The cost of one box of 25 candies in dollars
def boxSize   : ℕ := 25      -- The number of candies in one box
def cost (n : ℕ) : ℕ := (n / boxSize) * costOfBox -- The cost function for n candies

-- Theorem to be proven
theorem cost_of_600_candies : cost 600 = 144 :=
by sorry

end cost_of_600_candies_l97_97903


namespace complement_N_subset_M_l97_97459

-- Definitions for the sets M and N
def M : Set ℝ := {x | x * (x - 3) < 0}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- Complement of N in ℝ
def complement_N : Set ℝ := {x | ¬(x < 1 ∨ x ≥ 3)}

-- The theorem stating that complement_N is a subset of M
theorem complement_N_subset_M : complement_N ⊆ M :=
by
  sorry

end complement_N_subset_M_l97_97459


namespace asymptote_slope_l97_97711

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 81 = 1

-- Lean statement to prove slope of asymptotes
theorem asymptote_slope :
  (∀ x y : ℝ, hyperbola x y → (y/x) = 3/4 ∨ (y/x) = -(3/4)) :=
by
  sorry

end asymptote_slope_l97_97711


namespace discount_on_pickles_l97_97201

theorem discount_on_pickles :
  ∀ (meat_weight : ℝ) (meat_price_per_pound : ℝ) (bun_price : ℝ) (lettuce_price : ℝ)
    (tomato_weight : ℝ) (tomato_price_per_pound : ℝ) (pickles_price : ℝ) (total_paid : ℝ) (change : ℝ),
  meat_weight = 2 ∧
  meat_price_per_pound = 3.50 ∧
  bun_price = 1.50 ∧
  lettuce_price = 1.00 ∧
  tomato_weight = 1.5 ∧
  tomato_price_per_pound = 2.00 ∧
  pickles_price = 2.50 ∧
  total_paid = 20.00 ∧
  change = 6 →
  pickles_price - (total_paid - change - (meat_weight * meat_price_per_pound + tomato_weight * tomato_price_per_pound + bun_price + lettuce_price)) = 1 := 
by
  -- Begin the proof here (not required for this task)
  sorry

end discount_on_pickles_l97_97201


namespace fibonacci_polynomial_property_l97_97087

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Define the polynomial P(x) of degree 990
noncomputable def P : ℕ → ℕ :=
  sorry  -- To be defined as a polynomial with specified properties

-- Statement of the problem (theorem)
theorem fibonacci_polynomial_property (P : ℕ → ℕ) (hP : ∀ k, 992 ≤ k → k ≤ 1982 → P k = fibonacci k) :
  P 1983 = fibonacci 1983 - 1 :=
sorry  -- Proof omitted

end fibonacci_polynomial_property_l97_97087


namespace statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l97_97072

-- Statement A
theorem statement_A_incorrect (a b c d : ℝ) (ha : a < b) (hc : c < d) : ¬ (a * c < b * d) := by
  sorry

-- Statement B
theorem statement_B_correct (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) : -1 < a / b ∧ a / b < 3 := by
  sorry

-- Statement C
theorem statement_C_incorrect (m : ℝ) : ¬ (∀ x > 0, x / 2 + 2 / x ≥ m) ∧ (m ≤ 1) := by
  sorry

-- Statement D
theorem statement_D_incorrect : ∃ x : ℝ, (x^2 + 2) + 1 / (x^2 + 2) ≠ 2 := by
  sorry

end statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l97_97072


namespace sandy_siding_cost_l97_97154

theorem sandy_siding_cost
  (wall_length wall_height roof_base roof_height : ℝ)
  (siding_length siding_height siding_cost : ℝ)
  (num_walls num_roof_faces num_siding_sections : ℝ)
  (total_cost : ℝ)
  (h_wall_length : wall_length = 10)
  (h_wall_height : wall_height = 7)
  (h_roof_base : roof_base = 10)
  (h_roof_height : roof_height = 6)
  (h_siding_length : siding_length = 10)
  (h_siding_height : siding_height = 15)
  (h_siding_cost : siding_cost = 35)
  (h_num_walls : num_walls = 2)
  (h_num_roof_faces : num_roof_faces = 1)
  (h_num_siding_sections : num_siding_sections = 2)
  (h_total_cost : total_cost = 70) :
  (siding_cost * num_siding_sections) = total_cost := 
by
  sorry

end sandy_siding_cost_l97_97154


namespace min_balls_to_draw_l97_97684

theorem min_balls_to_draw {red green yellow blue white black : ℕ} 
    (h_red : red = 28) 
    (h_green : green = 20) 
    (h_yellow : yellow = 19) 
    (h_blue : blue = 13) 
    (h_white : white = 11) 
    (h_black : black = 9) :
    ∃ n, n = 76 ∧ 
    (∀ drawn, (drawn < n → (drawn ≤ 14 + 14 + 14 + 13 + 11 + 9)) ∧ (drawn >= n → (∃ c, c ≥ 15))) :=
sorry

end min_balls_to_draw_l97_97684


namespace zoe_total_earnings_l97_97529

theorem zoe_total_earnings
  (weeks : ℕ → ℝ)
  (weekly_hours : ℕ → ℝ)
  (wage_per_hour : ℝ)
  (h1 : weekly_hours 3 = 28)
  (h2 : weekly_hours 2 = 18)
  (h3 : weeks 3 - weeks 2 = 64.40)
  (h_same_wage : ∀ n, weeks n = weekly_hours n * wage_per_hour) :
  weeks 3 + weeks 2 = 296.24 :=
sorry

end zoe_total_earnings_l97_97529


namespace min_distance_between_graphs_l97_97881

noncomputable def minimum_distance (a : ℝ) (h : 1 < a) : ℝ :=
  if h1 : a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a)

theorem min_distance_between_graphs (a : ℝ) (h1 : 1 < a) :
  minimum_distance a h1 = 
  if a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a) :=
by
  intros
  sorry

end min_distance_between_graphs_l97_97881


namespace total_hours_verification_l97_97549

def total_hours_data_analytics : ℕ := 
  let weekly_class_homework_hours := (2 * 3 + 1 * 4 + 4) * 24 
  let lab_project_hours := 8 * 6 + (10 + 14 + 18)
  weekly_class_homework_hours + lab_project_hours

def total_hours_programming : ℕ :=
  let weekly_hours := (2 * 2 + 2 * 4 + 6) * 24
  weekly_hours

def total_hours_statistics : ℕ :=
  let weekly_class_lab_project_hours := (2 * 3 + 1 * 2 + 3) * 24
  let exam_study_hours := 9 * 5
  weekly_class_lab_project_hours + exam_study_hours

def total_hours_all_courses : ℕ :=
  total_hours_data_analytics + total_hours_programming + total_hours_statistics

theorem total_hours_verification : 
    total_hours_all_courses = 1167 := 
by 
    sorry

end total_hours_verification_l97_97549


namespace sandy_savings_percentage_l97_97091

theorem sandy_savings_percentage
  (S : ℝ) -- Sandy's salary last year
  (H1 : 0.10 * S = saved_last_year) -- Last year, Sandy saved 10% of her salary.
  (H2 : 1.10 * S = salary_this_year) -- This year, Sandy made 10% more than last year.
  (H3 : 0.15 * salary_this_year = saved_this_year) -- This year, Sandy saved 15% of her salary.
  : (saved_this_year / saved_last_year) * 100 = 165 := 
by 
  sorry

end sandy_savings_percentage_l97_97091


namespace greatest_integer_sum_of_integers_l97_97936

-- Definition of the quadratic function
def quadratic_expr (n : ℤ) : ℤ := n^2 - 15 * n + 56

-- The greatest integer n such that quadratic_expr n ≤ 0
theorem greatest_integer (n : ℤ) (h : quadratic_expr n ≤ 0) : n ≤ 8 := 
  sorry

-- All integers that satisfy the quadratic inequality
theorem sum_of_integers (sum_n : ℤ) (h : ∀ n : ℤ, 7 ≤ n ∧ n ≤ 8 → quadratic_expr n ≤ 0) 
  (sum_eq : sum_n = 7 + 8) : sum_n = 15 :=
  sorry

end greatest_integer_sum_of_integers_l97_97936


namespace percent_less_l97_97510

theorem percent_less (w u y z : ℝ) (P : ℝ) (hP : P = 0.40)
  (h1 : u = 0.60 * y)
  (h2 : z = 0.54 * y)
  (h3 : z = 1.50 * w) :
  w = (1 - P) * u := 
sorry

end percent_less_l97_97510


namespace complex_problem_l97_97987

open Complex

theorem complex_problem (a b : ℝ) (h : (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I) : a + b = 4 := by
  sorry

end complex_problem_l97_97987


namespace John_to_floor_pushups_l97_97501

theorem John_to_floor_pushups:
  let days_per_week := 5
  let reps_per_day := 1
  let total_reps_per_stage := 15
  let stages := 3 -- number of stages: wall, high elevation, low elevation
  let total_days_needed := stages * total_reps_per_stage
  let total_weeks_needed := total_days_needed / days_per_week
  total_weeks_needed = 9 := by
  -- Here we will define the specifics of the proof later.
  sorry

end John_to_floor_pushups_l97_97501


namespace real_possible_b_values_quadratic_non_real_roots_l97_97552

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l97_97552


namespace find_pairs_l97_97722

theorem find_pairs (a b : ℕ) (q r : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : 0 ≤ r) (h5 : r < a + b)
  (h6 : q^2 + r = 1977) :
  (a, b) = (50, 37) ∨ (a, b) = (50, 7) ∨ (a, b) = (37, 50) ∨ (a, b) = (7, 50) :=
  sorry

end find_pairs_l97_97722


namespace collinear_a_b_l97_97531

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, -2)

-- Definition of collinearity of vectors
def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

-- Statement to prove
theorem collinear_a_b : collinear a b :=
by
  sorry

end collinear_a_b_l97_97531


namespace total_clients_l97_97982

theorem total_clients (V K B N : Nat) (hV : V = 7) (hK : K = 8) (hB : B = 3) (hN : N = 18) :
    V + K - B + N = 30 := by
  sorry

end total_clients_l97_97982


namespace eq_factorial_sum_l97_97419

theorem eq_factorial_sum (k l m n : ℕ) (hk : k > 0) (hl : l > 0) (hm : m > 0) (hn : n > 0) :
  (1 / (Nat.factorial k : ℝ) + 1 / (Nat.factorial l : ℝ) + 1 / (Nat.factorial m : ℝ) = 1 / (Nat.factorial n : ℝ))
  ↔ (k = 3 ∧ l = 3 ∧ m = 3 ∧ n = 2) :=
by
  sorry

end eq_factorial_sum_l97_97419


namespace divisors_of_10_factorial_larger_than_9_factorial_l97_97940

theorem divisors_of_10_factorial_larger_than_9_factorial :
  ∃ n, n = 9 ∧ (∀ d, d ∣ (Nat.factorial 10) → d > (Nat.factorial 9) → d > (Nat.factorial 1) → n = 9) :=
sorry

end divisors_of_10_factorial_larger_than_9_factorial_l97_97940


namespace y_give_z_start_l97_97111

variables (Vx Vy Vz T : ℝ)
variables (D : ℝ)

-- Conditions
def condition1 : Prop := Vx * T = Vy * T + 100
def condition2 : Prop := Vx * T = Vz * T + 200
def condition3 : Prop := T > 0

theorem y_give_z_start (h1 : condition1 Vx Vy T) (h2 : condition2 Vx Vz T) (h3 : condition3 T) : (Vy - Vz) * T = 200 := 
by
  sorry

end y_give_z_start_l97_97111


namespace violates_properties_l97_97481

-- Definitions from conditions
variables {a b c m : ℝ}

-- Conclusion to prove
theorem violates_properties :
  (∀ c : ℝ, ac = bc → (c ≠ 0 → a = b)) ∧ (c = 0 → ac = bc) → False :=
sorry

end violates_properties_l97_97481


namespace rational_pair_exists_l97_97995

theorem rational_pair_exists (a b : ℚ) (h1 : a = 3/2) (h2 : b = 3) : a ≠ b ∧ a + b = a * b :=
by {
  sorry
}

end rational_pair_exists_l97_97995


namespace intersecting_lines_l97_97771

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem intersecting_lines (x y : ℝ) : x ≠ 0 → y ≠ 0 → 
  (diamond x y = diamond y x) ↔ (y = x ∨ y = -x) := 
by
  sorry

end intersecting_lines_l97_97771


namespace paul_taxes_and_fees_l97_97192

theorem paul_taxes_and_fees 
  (hourly_wage: ℝ) 
  (hours_worked : ℕ)
  (spent_on_gummy_bears_percentage : ℝ)
  (final_amount : ℝ)
  (gross_earnings := hourly_wage * hours_worked)
  (taxes_and_fees := gross_earnings - final_amount / (1 - spent_on_gummy_bears_percentage)):
  hourly_wage = 12.50 →
  hours_worked = 40 →
  spent_on_gummy_bears_percentage = 0.15 →
  final_amount = 340 →
  taxes_and_fees / gross_earnings = 0.20 :=
by
  intros
  sorry

end paul_taxes_and_fees_l97_97192


namespace q_zero_iff_arithmetic_l97_97521

-- Definitions of the terms and conditions
variables (A B q : ℝ) (hA : A ≠ 0)
def Sn (n : ℕ) : ℝ := A * n^2 + B * n + q
def arithmetic_sequence (an : ℕ → ℝ) : Prop := ∃ d a1, ∀ n, an n = a1 + n * d

-- The proof statement we need to show
theorem q_zero_iff_arithmetic (an : ℕ → ℝ) :
  (q = 0) ↔ (∃ a1 d, ∀ n, Sn A B 0 n = (d / 2) * n^2 + (a1 - d / 2) * n) :=
sorry

end q_zero_iff_arithmetic_l97_97521


namespace solve_quadratic1_solve_quadratic2_l97_97078

-- For the first quadratic equation: 3x^2 = 6x
theorem solve_quadratic1 (x : ℝ) (h : 3 * x^2 = 6 * x) : x = 0 ∨ x = 2 :=
sorry

-- For the second quadratic equation: x^2 - 6x + 5 = 0
theorem solve_quadratic2 (x : ℝ) (h : x^2 - 6 * x + 5 = 0) : x = 5 ∨ x = 1 :=
sorry

end solve_quadratic1_solve_quadratic2_l97_97078


namespace binary_to_decimal_101101_l97_97200

theorem binary_to_decimal_101101 : 
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 :=
by
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  have h : (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 := sorry
  exact h

end binary_to_decimal_101101_l97_97200


namespace walter_bus_time_l97_97143

noncomputable def walter_schedule : Prop :=
  let wake_up_time := 6  -- Walter gets up at 6:00 a.m.
  let leave_home_time := 7  -- Walter catches the school bus at 7:00 a.m.
  let arrival_home_time := 17  -- Walter arrives home at 5:00 p.m.
  let num_classes := 8  -- Walter has 8 classes
  let class_duration := 45  -- Each class lasts 45 minutes
  let lunch_duration := 40  -- Walter has 40 minutes for lunch
  let additional_activities_hours := 2.5  -- Walter has 2.5 hours of additional activities

  -- Total time calculation
  let total_away_hours := arrival_home_time - leave_home_time
  let total_away_minutes := total_away_hours * 60

  -- School-related activities calculation
  let total_class_minutes := num_classes * class_duration
  let total_additional_activities_minutes := additional_activities_hours * 60
  let total_school_activity_minutes := total_class_minutes + lunch_duration + total_additional_activities_minutes

  -- Time spent on the bus
  let bus_time := total_away_minutes - total_school_activity_minutes
  bus_time = 50

-- Statement to prove
theorem walter_bus_time : walter_schedule :=
  sorry

end walter_bus_time_l97_97143


namespace luke_played_rounds_l97_97426

theorem luke_played_rounds (total_points : ℕ) (points_per_round : ℕ) (result : ℕ)
  (h1 : total_points = 154)
  (h2 : points_per_round = 11)
  (h3 : result = total_points / points_per_round) :
  result = 14 :=
by
  rw [h1, h2] at h3
  exact h3

end luke_played_rounds_l97_97426


namespace max_y_value_l97_97870

theorem max_y_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = (x - y) / (x + 3 * y)) : y ≤ 1 / 3 :=
by
  sorry

end max_y_value_l97_97870


namespace distance_covered_l97_97039

-- Define the rate and time as constants
def rate : ℝ := 4 -- 4 miles per hour
def time : ℝ := 2 -- 2 hours

-- Theorem statement: Verify the distance covered
theorem distance_covered : rate * time = 8 := 
by
  sorry

end distance_covered_l97_97039


namespace father_current_age_is_85_l97_97973

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end father_current_age_is_85_l97_97973


namespace commission_percentage_l97_97710

theorem commission_percentage 
  (total_amount : ℝ) 
  (h1 : total_amount = 800) 
  (commission_first_500 : ℝ) 
  (h2 : commission_first_500 = 0.20 * 500) 
  (excess_amount : ℝ) 
  (h3 : excess_amount = (total_amount - 500)) 
  (commission_excess : ℝ) 
  (h4 : commission_excess = 0.25 * excess_amount) 
  (total_commission : ℝ) 
  (h5 : total_commission = commission_first_500 + commission_excess) 
  : (total_commission / total_amount) * 100 = 21.875 := 
by
  sorry

end commission_percentage_l97_97710


namespace tan_315_eq_neg1_l97_97716

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l97_97716


namespace C_plus_D_l97_97410

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : 
  C + D = -10 := by
  sorry

end C_plus_D_l97_97410


namespace part3_conclusion_l97_97767

-- Definitions and conditions for the problem
def quadratic_function (a x : ℝ) : ℝ := (x - a)^2 + a - 1

-- Part 1: Given condition that (1, 2) lies on the graph of the quadratic function
def part1_condition (a : ℝ) := (quadratic_function a 1) = 2

-- Part 2: Given condition that the function has a minimum value of 2 for 1 ≤ x ≤ 4
def part2_condition (a : ℝ) := ∀ x, 1 ≤ x ∧ x ≤ 4 → quadratic_function a x ≥ 2

-- Part 3: Given condition (m, n) on the graph where m > 0 and m > 2a
def part3_condition (a m n : ℝ) := m > 0 ∧ m > 2 * a ∧ quadratic_function a m = n

-- Conclusion for Part 3: Prove that n > -5/4
theorem part3_conclusion (a m n : ℝ) (h : part3_condition a m n) : n > -5/4 := 
sorry  -- Proof required here

end part3_conclusion_l97_97767


namespace probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l97_97309

-- Definitions of the probabilities
def P_A := 0.24
def P_B := 0.28
def P_C := 0.19
def P_D := 0.16
def P_E := 0.13

-- Prove that the probability of hitting the 10 or 9 rings is 0.52
theorem probability_of_hitting_10_or_9 : P_A + P_B = 0.52 :=
  by sorry

-- Prove that the probability of hitting at least the 7 ring is 0.87
theorem probability_of_hitting_at_least_7 : P_A + P_B + P_C + P_D = 0.87 :=
  by sorry

-- Prove that the probability of hitting less than 8 rings is 0.29
theorem probability_of_hitting_less_than_8 : P_D + P_E = 0.29 :=
  by sorry

end probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l97_97309


namespace select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l97_97712

variable (n m : ℕ) -- n for males, m for females
variable (mc fc : ℕ) -- mc for male captain, fc for female captain

def num_ways_3_males_2_females : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 4 2)

def num_ways_at_least_1_captain : ℕ :=
  (2 * (Nat.choose 8 4)) + (Nat.choose 8 3)

def num_ways_at_least_1_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 6 5)

def num_ways_both_captain_and_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 8 5) - (Nat.choose 5 4)

theorem select_3_males_2_females : num_ways_3_males_2_females = 120 := by
  sorry
  
theorem select_at_least_1_captain : num_ways_at_least_1_captain = 196 := by
  sorry
  
theorem select_at_least_1_female : num_ways_at_least_1_female = 246 := by
  sorry
  
theorem select_both_captain_and_female : num_ways_both_captain_and_female = 191 := by
  sorry

end select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l97_97712


namespace profit_function_expression_l97_97618

def dailySalesVolume (x : ℝ) : ℝ := 300 + 3 * (99 - x)

def profitPerItem (x : ℝ) : ℝ := x - 50

def dailyProfit (x : ℝ) : ℝ := (x - 50) * (300 + 3 * (99 - x))

theorem profit_function_expression (x : ℝ) :
  dailyProfit x = (x - 50) * dailySalesVolume x :=
by sorry

end profit_function_expression_l97_97618


namespace germs_per_dish_l97_97593

theorem germs_per_dish (total_germs : ℝ) (num_dishes : ℝ) 
(h1 : total_germs = 5.4 * 10^6) 
(h2 : num_dishes = 10800) : total_germs / num_dishes = 502 :=
sorry

end germs_per_dish_l97_97593


namespace alexander_total_payment_l97_97951

variable (initialFee : ℝ) (dailyRent : ℝ) (costPerMile : ℝ) (daysRented : ℕ) (milesDriven : ℝ)

def totalCost (initialFee dailyRent costPerMile : ℝ) (daysRented : ℕ) (milesDriven : ℝ) : ℝ :=
  initialFee + (dailyRent * daysRented) + (costPerMile * milesDriven)

theorem alexander_total_payment :
  totalCost 15 30 0.25 3 350 = 192.5 :=
by
  unfold totalCost
  norm_num

end alexander_total_payment_l97_97951


namespace six_digit_start_5_no_12_digit_perfect_square_l97_97359

theorem six_digit_start_5_no_12_digit_perfect_square :
  ∀ (n : ℕ), (500000 ≤ n ∧ n < 600000) → 
  (∀ (m : ℕ), n * 10^6 + m ≠ k^2) :=
by
  sorry

end six_digit_start_5_no_12_digit_perfect_square_l97_97359


namespace rectangle_area_l97_97055

theorem rectangle_area (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 4) :
  l * w = 8 / 9 :=
by
  sorry

end rectangle_area_l97_97055


namespace max_ratio_of_three_digit_to_sum_l97_97061

theorem max_ratio_of_three_digit_to_sum (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9)
  (hc : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c) / (a + b + c) ≤ 100 :=
by sorry

end max_ratio_of_three_digit_to_sum_l97_97061


namespace find_value_of_sum_of_squares_l97_97922

theorem find_value_of_sum_of_squares (x y : ℝ) (h : x^2 + y^2 + x^2 * y^2 - 4 * x * y + 1 = 0) :
  (x + y)^2 = 4 :=
sorry

end find_value_of_sum_of_squares_l97_97922


namespace case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l97_97075

theorem case_one_ellipses_foci_xaxis :
  ∀ (a : ℝ) (e : ℝ), a = 6 ∧ e = 2 / 3 → (∃ (b : ℝ), (b^2 = (a^2 - (e * a)^2) ∧ (a > 0) → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)) ∨ (y^2 / a^2 + x^2 / b^2 = 1)))) :=
by
  sorry

theorem case_two_ellipses_foci_exact :
  ∀ (F1 F2 : ℝ × ℝ), F1 = (-4,0) ∧ F2 = (4,0) ∧ ∀ P : ℝ × ℝ, ((dist P F1) + (dist P F2) = 10) →
  ∃ (a : ℝ) (b : ℝ), a = 5 ∧ b^2 = a^2 - 4^2 → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1))) :=
by
  sorry

end case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l97_97075


namespace complement_of_A_in_reals_l97_97983

open Set

theorem complement_of_A_in_reals :
  (compl {x : ℝ | (x - 1) / (x - 2) ≥ 0}) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end complement_of_A_in_reals_l97_97983


namespace problem_solved_l97_97534

-- Define the function f with the given conditions
def satisfies_conditions(f : ℝ × ℝ × ℝ → ℝ) :=
  (∀ x y z t : ℝ, f (x + t, y + t, z + t) = t + f (x, y, z)) ∧
  (∀ x y z t : ℝ, f (t * x, t * y, t * z) = t * f (x, y, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (y, x, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (x, z, y))

-- We'll state the main result to be proven, without giving the proof
theorem problem_solved (f : ℝ × ℝ × ℝ → ℝ) (h : satisfies_conditions f) : f (2000, 2001, 2002) = 2001 :=
  sorry

end problem_solved_l97_97534


namespace value_of_a_plus_b_l97_97046

theorem value_of_a_plus_b (a b : ℝ) : (|a - 1| + (b + 3)^2 = 0) → (a + b = -2) :=
by
  sorry

end value_of_a_plus_b_l97_97046


namespace ab_div_c_eq_2_l97_97318

variable (a b c : ℝ)

def condition1 (a b c : ℝ) : Prop := a * b - c = 3
def condition2 (a b c : ℝ) : Prop := a * b * c = 18

theorem ab_div_c_eq_2 (h1 : condition1 a b c) (h2 : condition2 a b c) : a * b / c = 2 :=
by sorry

end ab_div_c_eq_2_l97_97318


namespace prime_sum_product_l97_97766

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 91) : p * q = 178 := 
by
  sorry

end prime_sum_product_l97_97766


namespace deans_height_l97_97045

theorem deans_height
  (D : ℕ) 
  (h1 : 10 * D = D + 81) : 
  D = 9 := sorry

end deans_height_l97_97045


namespace seq_form_l97_97704

-- Define the sequence a as a function from natural numbers to natural numbers
def seq (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, 0 < m → 0 < n → ⌊(a m : ℚ) / a n⌋ = ⌊(m : ℚ) / n⌋

-- Define the statement that all sequences satisfying the condition must be of the form k * i
theorem seq_form (a : ℕ → ℕ) : seq a → ∃ k : ℕ, (0 < k) ∧ (∀ n, 0 < n → a n = k * n) := 
by
  intros h
  sorry

end seq_form_l97_97704


namespace derivative_of_f_l97_97826

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x

theorem derivative_of_f :
  ∀ x ≠ 0, deriv f x = ((-x * Real.sin x - Real.cos x) / (x^2)) := sorry

end derivative_of_f_l97_97826


namespace find_z_l97_97520

theorem find_z (x y z : ℝ) (h1 : y = 3 * x - 5) (h2 : z = 3 * x + 3) (h3 : y = 1) : z = 9 := 
by
  sorry

end find_z_l97_97520


namespace smallest_positive_period_of_f_cos_2x0_l97_97746

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sin x * Real.cos x + 2 * (Real.sqrt 3) * (Real.cos x)^2 - Real.sqrt 3

theorem smallest_positive_period_of_f :
  (∃ p > 0, ∀ x, f x = f (x + p)) ∧
  (∀ q > 0, (∀ x, f x = f (x + q)) -> q ≥ Real.pi) :=
sorry

theorem cos_2x0 (x0 : ℝ) (h0 : x0 ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h1 : f (x0 - Real.pi / 12) = 6 / 5) :
  Real.cos (2 * x0) = (3 - 4 * Real.sqrt 3) / 10 :=
sorry

end smallest_positive_period_of_f_cos_2x0_l97_97746


namespace quadratic_polynomial_discriminant_l97_97167

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l97_97167


namespace arithmetic_seq_sum_l97_97140

theorem arithmetic_seq_sum(S : ℕ → ℝ) (d : ℝ) (h1 : S 5 < S 6) 
    (h2 : S 6 = S 7) (h3 : S 7 > S 8) : S 9 < S 5 := 
sorry

end arithmetic_seq_sum_l97_97140


namespace exists_K_p_l97_97323

noncomputable def constant_K_p (p : ℝ) (hp : p > 1) : ℝ :=
  (p * p) / (p - 1)

theorem exists_K_p (p : ℝ) (hp : p > 1) :
  ∃ K_p > 0, ∀ x y : ℝ, |x|^p + |y|^p = 2 → (x - y)^2 ≤ K_p * (4 - (x + y)^2) :=
by
  use constant_K_p p hp
  sorry

end exists_K_p_l97_97323


namespace greater_number_l97_97053

theorem greater_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 12) : a = 26 :=
by
  have h3 : 2 * a = 52 := by linarith
  have h4 : a = 26 := by linarith
  exact h4

end greater_number_l97_97053


namespace initial_HNO3_percentage_is_correct_l97_97556

def initial_percentage_of_HNO3 (P : ℚ) : Prop :=
  let initial_volume := 60
  let added_volume := 18
  let final_volume := 78
  let final_percentage := 50
  (P / 100) * initial_volume + added_volume = (final_percentage / 100) * final_volume

theorem initial_HNO3_percentage_is_correct :
  initial_percentage_of_HNO3 35 :=
by
  sorry

end initial_HNO3_percentage_is_correct_l97_97556


namespace sqrt_90000_eq_300_l97_97223

theorem sqrt_90000_eq_300 : Real.sqrt 90000 = 300 := by
  sorry

end sqrt_90000_eq_300_l97_97223


namespace matrix_power_50_l97_97432

-- Defining the matrix A.
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 1], 
    ![-12, -3]]

-- Statement of the theorem
theorem matrix_power_50 :
  A ^ 50 = ![![301, 50], 
               ![-900, -301]] :=
by
  sorry

end matrix_power_50_l97_97432


namespace cos_pi_over_3_plus_2alpha_correct_l97_97979

noncomputable def cos_pi_over_3_plus_2alpha (α : Real) (h : Real.sin (Real.pi / 3 - α) = 1 / 4) : Real :=
  Real.cos (Real.pi / 3 + 2 * α)

theorem cos_pi_over_3_plus_2alpha_correct (α : Real) (h : Real.sin (Real.pi / 3 - α) = 1 / 4) :
  cos_pi_over_3_plus_2alpha α h = -7 / 8 :=
by
  sorry

end cos_pi_over_3_plus_2alpha_correct_l97_97979


namespace gifts_from_Pedro_l97_97454

theorem gifts_from_Pedro (gifts_from_Emilio gifts_from_Jorge total_gifts : ℕ)
  (h1 : gifts_from_Emilio = 11)
  (h2 : gifts_from_Jorge = 6)
  (h3 : total_gifts = 21) :
  total_gifts - (gifts_from_Emilio + gifts_from_Jorge) = 4 := by
  sorry

end gifts_from_Pedro_l97_97454


namespace Dave_needs_31_gallons_l97_97283

noncomputable def numberOfGallons (numberOfTanks : ℕ) (height : ℝ) (diameter : ℝ) (coveragePerGallon : ℝ) : ℕ :=
  let radius := diameter / 2
  let lateral_surface_area := 2 * Real.pi * radius * height
  let total_surface_area := lateral_surface_area * numberOfTanks
  let gallons_needed := total_surface_area / coveragePerGallon
  Nat.ceil gallons_needed

theorem Dave_needs_31_gallons :
  numberOfGallons 20 24 8 400 = 31 :=
by
  sorry

end Dave_needs_31_gallons_l97_97283


namespace max_n_is_11_l97_97617

noncomputable def max_n (a1 d : ℝ) : ℕ :=
if h : d < 0 then
  11
else
  sorry

theorem max_n_is_11 (d : ℝ) (a1 : ℝ) (c : ℝ) :
  (d / 2) * (22 ^ 2) + (a1 - (d / 2)) * 22 + c ≥ 0 →
  22 = (a1 - (d / 2)) / (- (d / 2)) →
  max_n a1 d = 11 :=
by
  intros h1 h2
  rw [max_n]
  split_ifs
  · exact rfl
  · exact sorry

end max_n_is_11_l97_97617


namespace quadratic_roots_relation_l97_97968

theorem quadratic_roots_relation (a b s p : ℝ) (h : a^2 + b^2 = 15) (h1 : s = a + b) (h2 : p = a * b) : s^2 - 2 * p = 15 :=
by sorry

end quadratic_roots_relation_l97_97968


namespace number_of_ways_to_choose_one_book_l97_97667

-- Defining the conditions
def num_chinese_books : ℕ := 5
def num_math_books : ℕ := 4

-- Statement of the theorem
theorem number_of_ways_to_choose_one_book : num_chinese_books + num_math_books = 9 :=
by
  -- Skipping the proof as instructed
  sorry

end number_of_ways_to_choose_one_book_l97_97667


namespace saving_time_for_downpayment_l97_97503

def annual_salary : ℚ := 150000
def saving_rate : ℚ := 0.10
def house_cost : ℚ := 450000
def downpayment_rate : ℚ := 0.20

theorem saving_time_for_downpayment : 
  (downpayment_rate * house_cost) / (saving_rate * annual_salary) = 6 :=
by
  sorry

end saving_time_for_downpayment_l97_97503


namespace distinct_pairs_count_l97_97686

theorem distinct_pairs_count : 
  (∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, ∃ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ a + b = 40 ∧ p = (a, b)) ∧ s.card = 39) := sorry

end distinct_pairs_count_l97_97686


namespace find_m_l97_97984

-- Define the given equations of the lines
def line1 (m : ℝ) : ℝ × ℝ → Prop := fun p => (3 + m) * p.1 - 4 * p.2 = 5 - 3 * m
def line2 : ℝ × ℝ → Prop := fun p => 2 * p.1 - p.2 = 8

-- Define the condition for parallel lines based on the given equations
def are_parallel (m : ℝ) : Prop := (3 + m) / 4 = 2

-- The main theorem stating the value of m
theorem find_m (m : ℝ) (h1 : ∀ p : ℝ × ℝ, line1 m p) (h2 : ∀ p : ℝ × ℝ, line2 p) (h_parallel : are_parallel m) : m = 5 :=
sorry

end find_m_l97_97984


namespace stream_speed_l97_97367

def upstream_time : ℝ := 4  -- time in hours
def downstream_time : ℝ := 4  -- time in hours
def upstream_distance : ℝ := 32  -- distance in km
def downstream_distance : ℝ := 72  -- distance in km

-- Speed equations based on given conditions
def effective_speed_upstream (vj vs : ℝ) : Prop := vj - vs = upstream_distance / upstream_time
def effective_speed_downstream (vj vs : ℝ) : Prop := vj + vs = downstream_distance / downstream_time

theorem stream_speed (vj vs : ℝ)  
  (h1 : effective_speed_upstream vj vs)
  (h2 : effective_speed_downstream vj vs) : 
  vs = 5 := sorry

end stream_speed_l97_97367


namespace Razorback_tshirt_shop_sales_l97_97348

theorem Razorback_tshirt_shop_sales :
  let tshirt_price := 98
  let hat_price := 45
  let scarf_price := 60
  let tshirts_sold_arkansas := 42
  let hats_sold_arkansas := 32
  let scarves_sold_arkansas := 15
  (tshirts_sold_arkansas * tshirt_price + hats_sold_arkansas * hat_price + scarves_sold_arkansas * scarf_price) = 6456 :=
by
  sorry

end Razorback_tshirt_shop_sales_l97_97348


namespace factor_expression_l97_97698

theorem factor_expression :
  (8 * x ^ 4 + 34 * x ^ 3 - 120 * x + 150) - (-2 * x ^ 4 + 12 * x ^ 3 - 5 * x + 10) 
  = 5 * x * (2 * x ^ 3 + (22 / 5) * x ^ 2 - 23 * x + 28) :=
sorry

end factor_expression_l97_97698


namespace half_abs_diff_squares_l97_97815

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 21) (h₂ : b = 17) :
  (|a^2 - b^2| / 2) = 76 :=
by 
  sorry

end half_abs_diff_squares_l97_97815


namespace probability_of_quitters_from_10_member_tribe_is_correct_l97_97489

noncomputable def probability_quitters_from_10_member_tribe : ℚ :=
  let total_contestants := 18
  let ten_member_tribe := 10
  let total_quitters := 2
  let comb (n k : ℕ) : ℕ := Nat.choose n k
  
  let total_combinations := comb total_contestants total_quitters
  let ten_tribe_combinations := comb ten_member_tribe total_quitters
  
  ten_tribe_combinations / total_combinations

theorem probability_of_quitters_from_10_member_tribe_is_correct :
  probability_quitters_from_10_member_tribe = 5 / 17 :=
  by
    sorry

end probability_of_quitters_from_10_member_tribe_is_correct_l97_97489


namespace small_cubes_with_painted_faces_l97_97929

-- Definitions based on conditions
def large_cube_edge : ℕ := 8
def small_cube_edge : ℕ := 2
def division_factor : ℕ := large_cube_edge / small_cube_edge
def total_small_cubes : ℕ := division_factor ^ 3

-- Proving the number of cubes with specific painted faces.
theorem small_cubes_with_painted_faces :
  (8 : ℤ) = 8 ∧ -- 8 smaller cubes with three painted faces
  (24 : ℤ) = 24 ∧ -- 24 smaller cubes with two painted faces
  (24 : ℤ) = 24 := -- 24 smaller cubes with one painted face
by
  sorry

end small_cubes_with_painted_faces_l97_97929


namespace sequence_a_10_value_l97_97576

theorem sequence_a_10_value : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → (∀ n : ℕ, 0 < n → a (n + 1) - a n = 2) → a 10 = 21 := 
by 
  intros a h1 hdiff
  sorry

end sequence_a_10_value_l97_97576


namespace find_a5_a6_l97_97958

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

-- Given conditions
axiom h1 : a 1 + a 2 = 5
axiom h2 : a 3 + a 4 = 7

-- Arithmetic sequence property
axiom arith_seq : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)

-- The statement we want to prove
theorem find_a5_a6 : a 5 + a 6 = 9 :=
sorry

end ArithmeticSequence

end find_a5_a6_l97_97958


namespace integer_values_of_x_for_positive_star_l97_97342

-- Definition of the operation star
def star (a b : ℕ) : ℚ := (a^2 : ℕ) / b

-- Problem statement
theorem integer_values_of_x_for_positive_star :
  ∃ (count : ℕ), count = 9 ∧ (∀ x : ℕ, (10^2 % x = 0) → (∃ n : ℕ, star 10 x = n)) :=
sorry

end integer_values_of_x_for_positive_star_l97_97342


namespace sound_pressure_level_l97_97830

theorem sound_pressure_level (p_0 p_1 p_2 p_3 : ℝ) (h_p0 : 0 < p_0)
  (L_p : ℝ → ℝ)
  (h_gasoline : 60 ≤ L_p p_1 ∧ L_p p_1 ≤ 90)
  (h_hybrid : 50 ≤ L_p p_2 ∧ L_p p_2 ≤ 60)
  (h_electric : L_p p_3 = 40)
  (h_L_p : ∀ p, L_p p = 20 * Real.log (p / p_0))
  : p_2 ≤ p_1 ∧ p_1 ≤ 100 * p_2 :=
by
  sorry

end sound_pressure_level_l97_97830


namespace train_speeds_l97_97388

noncomputable def c1 : ℝ := sorry  -- speed of the passenger train in km/min
noncomputable def c2 : ℝ := sorry  -- speed of the freight train in km/min
noncomputable def c3 : ℝ := sorry  -- speed of the express train in km/min

def conditions : Prop :=
  (5 / c1 + 5 / c2 = 15) ∧
  (5 / c2 + 5 / c3 = 11) ∧
  (c2 ≤ c1) ∧
  (c3 ≤ 2.5)

-- The theorem to be proved
theorem train_speeds :
  conditions →
  (40 / 60 ≤ c1 ∧ c1 ≤ 50 / 60) ∧ 
  (100 / 3 / 60 ≤ c2 ∧ c2 ≤ 40 / 60) ∧ 
  (600 / 7 / 60 ≤ c3 ∧ c3 ≤ 150 / 60) :=
sorry

end train_speeds_l97_97388


namespace sum_of_number_and_reverse_l97_97088

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l97_97088


namespace extreme_value_h_at_a_zero_range_of_a_l97_97691

noncomputable def f (x : ℝ) : ℝ := 1 - Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x / (a * x + 1)

noncomputable def h (x : ℝ) (a : ℝ) : ℝ := (Real.exp (-x)) * (g x a)

-- Statement for the first proof problem
theorem extreme_value_h_at_a_zero :
  ∀ x : ℝ, h x 0 ≤ 1 / Real.exp 1 :=
sorry

-- Statement for the second proof problem
theorem range_of_a:
  ∀ x : ℝ, (0 ≤ x → x ≤ 1 / 2) → (f x ≤ g x x) :=
sorry

end extreme_value_h_at_a_zero_range_of_a_l97_97691


namespace volume_set_points_sum_l97_97577

-- Defining the problem conditions
def rectangular_parallelepiped_length : ℝ := 5
def rectangular_parallelepiped_width : ℝ := 6
def rectangular_parallelepiped_height : ℝ := 7
def unit_extension : ℝ := 1

-- Defining what we need to prove
theorem volume_set_points_sum :
  let V_box : ℝ := rectangular_parallelepiped_length * rectangular_parallelepiped_width * rectangular_parallelepiped_height
  let V_ext : ℝ := 2 * (unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_width 
                  + unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_height 
                  + unit_extension * rectangular_parallelepiped_width * rectangular_parallelepiped_height)
  let V_cyl : ℝ := 18 * π
  let V_sph : ℝ := (4 / 3) * π
  let V_total : ℝ := V_box + V_ext + V_cyl + V_sph
  let m : ℕ := 1272
  let n : ℕ := 58
  let p : ℕ := 3
  V_total = (m : ℝ) + (n : ℝ) * π / (p : ℝ) ∧ (m + n + p = 1333)
  := by
  sorry

end volume_set_points_sum_l97_97577


namespace meiosis_fertilization_stability_l97_97064

def maintains_chromosome_stability (x : String) : Prop :=
  x = "Meiosis and Fertilization"

theorem meiosis_fertilization_stability :
  maintains_chromosome_stability "Meiosis and Fertilization" :=
by
  sorry

end meiosis_fertilization_stability_l97_97064


namespace weight_shaina_receives_l97_97239

namespace ChocolateProblem

-- Definitions based on conditions
def total_chocolate : ℚ := 60 / 7
def piles : ℚ := 5
def weight_per_pile : ℚ := total_chocolate / piles
def shaina_piles : ℚ := 2

-- Proposition to represent the question and correct answer
theorem weight_shaina_receives : 
  (weight_per_pile * shaina_piles) = 24 / 7 := 
by
  sorry

end ChocolateProblem

end weight_shaina_receives_l97_97239


namespace smallest_sum_l97_97199

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l97_97199


namespace length_of_crease_correct_l97_97641

noncomputable def length_of_crease (theta : ℝ) : ℝ := Real.sqrt (40 + 24 * Real.cos theta)

theorem length_of_crease_correct (theta : ℝ) : 
  length_of_crease theta = Real.sqrt (40 + 24 * Real.cos theta) := 
by 
  sorry

end length_of_crease_correct_l97_97641


namespace sequence_bound_l97_97480

theorem sequence_bound (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_condition : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
sorry

end sequence_bound_l97_97480


namespace jasmine_laps_per_afternoon_l97_97369

-- Defining the conditions
def swims_each_week (days_per_week : ℕ) := days_per_week = 5
def total_weeks := 5
def total_laps := 300

-- Main proof statement
theorem jasmine_laps_per_afternoon (d : ℕ) (l : ℕ) :
  swims_each_week d →
  total_weeks * d = 25 →
  total_laps = 300 →
  300 / 25 = l →
  l = 12 :=
by
  intros
  -- Skipping the proof
  sorry

end jasmine_laps_per_afternoon_l97_97369


namespace measure_4_minutes_with_hourglasses_l97_97092

/-- Prove that it is possible to measure exactly 4 minutes using hourglasses of 9 minutes and 7 minutes and the minimum total time required is 18 minutes -/
theorem measure_4_minutes_with_hourglasses : 
  ∃ (a b : ℕ), (9 * a - 7 * b = 4) ∧ (a + b) * 1 ≤ 2 ∧ (a * 9 ≤ 18 ∧ b * 7 <= 18) :=
by {
  sorry
}

end measure_4_minutes_with_hourglasses_l97_97092


namespace wood_stove_afternoon_burn_rate_l97_97703

-- Conditions extracted as definitions
def morning_burn_rate : ℝ := 2
def morning_duration : ℝ := 4
def initial_wood : ℝ := 30
def final_wood : ℝ := 3
def afternoon_duration : ℝ := 4

-- Theorem statement matching the conditions and correct answer
theorem wood_stove_afternoon_burn_rate :
  let morning_burned := morning_burn_rate * morning_duration
  let total_burned := initial_wood - final_wood
  let afternoon_burned := total_burned - morning_burned
  ∃ R : ℝ, (afternoon_burned = R * afternoon_duration) ∧ (R = 4.75) :=
by
  sorry

end wood_stove_afternoon_burn_rate_l97_97703


namespace calculate_flat_tax_l97_97860

open Real

def price_per_sq_ft (property: String) : Real :=
  if property = "Condo" then 98
  else if property = "BarnHouse" then 84
  else if property = "DetachedHouse" then 102
  else if property = "Townhouse" then 96
  else if property = "Garage" then 60
  else if property = "PoolArea" then 50
  else 0

def area_in_sq_ft (property: String) : Real :=
  if property = "Condo" then 2400
  else if property = "BarnHouse" then 1200
  else if property = "DetachedHouse" then 3500
  else if property = "Townhouse" then 2750
  else if property = "Garage" then 480
  else if property = "PoolArea" then 600
  else 0

def total_value : Real :=
  (price_per_sq_ft "Condo" * area_in_sq_ft "Condo") +
  (price_per_sq_ft "BarnHouse" * area_in_sq_ft "BarnHouse") +
  (price_per_sq_ft "DetachedHouse" * area_in_sq_ft "DetachedHouse") +
  (price_per_sq_ft "Townhouse" * area_in_sq_ft "Townhouse") +
  (price_per_sq_ft "Garage" * area_in_sq_ft "Garage") +
  (price_per_sq_ft "PoolArea" * area_in_sq_ft "PoolArea")

def tax_rate : Real := 0.0125

theorem calculate_flat_tax : total_value * tax_rate = 12697.50 := by
  sorry

end calculate_flat_tax_l97_97860


namespace part_a_l97_97730

-- Define the sequences and their properties
variables {n : ℕ} (h1 : n ≥ 3)
variables (a b : ℕ → ℝ)
variables (h_arith : ∀ k, a (k+1) = a k + d)
variables (h_geom : ∀ k, b (k+1) = b k * q)
variables (h_a1_b1 : a 1 = b 1)
variables (h_an_bn : a n = b n)

-- State the theorem to be proven
theorem part_a (k : ℕ) (h_k : 2 ≤ k ∧ k ≤ n - 1) : a k > b k :=
  sorry

end part_a_l97_97730


namespace candy_distribution_proof_l97_97185

theorem candy_distribution_proof :
  ∀ (candy_total Kate Robert Bill Mary : ℕ),
  candy_total = 20 →
  Kate = 4 →
  Robert = Kate + 2 →
  Bill = Mary - 6 →
  Kate = Bill + 2 →
  Mary > Robert →
  (Mary - Robert = 2) :=
by
  intros candy_total Kate Robert Bill Mary h1 h2 h3 h4 h5 h6
  sorry

end candy_distribution_proof_l97_97185


namespace system1_solution_l97_97587

theorem system1_solution (x y : ℝ) 
  (h1 : x + y = 10^20) 
  (h2 : x - y = 10^19) :
  x = 55 * 10^18 ∧ y = 45 * 10^18 := 
by
  sorry

end system1_solution_l97_97587


namespace greatest_number_of_bouquets_l97_97919

/--
Sara has 42 red flowers, 63 yellow flowers, and 54 blue flowers.
She wants to make bouquets with the same number of each color flower in each bouquet.
Prove that the greatest number of bouquets she can make is 21.
-/
theorem greatest_number_of_bouquets (red yellow blue : ℕ) (h_red : red = 42) (h_yellow : yellow = 63) (h_blue : blue = 54) :
  Nat.gcd (Nat.gcd red yellow) blue = 21 :=
by
  rw [h_red, h_yellow, h_blue]
  sorry

end greatest_number_of_bouquets_l97_97919


namespace seventh_place_is_unspecified_l97_97660

noncomputable def charlie_position : ℕ := 5
noncomputable def emily_position : ℕ := charlie_position + 5
noncomputable def dana_position : ℕ := 10
noncomputable def bob_position : ℕ := dana_position - 2
noncomputable def alice_position : ℕ := emily_position + 3

theorem seventh_place_is_unspecified :
  ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 15 ∧ x ≠ charlie_position ∧ x ≠ emily_position ∧
  x ≠ dana_position ∧ x ≠ bob_position ∧ x ≠ alice_position →
  x = 7 → false := 
by
  sorry

end seventh_place_is_unspecified_l97_97660


namespace janelle_gave_green_marbles_l97_97611

def initial_green_marbles : ℕ := 26
def bags_blue_marbles : ℕ := 6
def marbles_per_bag : ℕ := 10
def total_blue_marbles : ℕ := bags_blue_marbles * marbles_per_bag
def total_marbles_after_gift : ℕ := 72
def blue_marbles_in_gift : ℕ := 8
def final_blue_marbles : ℕ := total_blue_marbles - blue_marbles_in_gift
def final_green_marbles : ℕ := total_marbles_after_gift - final_blue_marbles
def initial_green_marbles_after_gift : ℕ := final_green_marbles
def green_marbles_given : ℕ := initial_green_marbles - initial_green_marbles_after_gift

theorem janelle_gave_green_marbles :
  green_marbles_given = 6 :=
by {
  sorry
}

end janelle_gave_green_marbles_l97_97611


namespace problem_1_problem_2_0_lt_a_lt_1_problem_2_a_gt_1_l97_97049

noncomputable def f (a x : ℝ) := a^(3 * x + 1)
noncomputable def g (a x : ℝ) := (1 / a)^(5 * x - 2)

variables {a x : ℝ}

theorem problem_1 (h : 0 < a ∧ a < 1) : f a x < 1 ↔ x > -1/3 :=
sorry

theorem problem_2_0_lt_a_lt_1 (h : 0 < a ∧ a < 1) : f a x ≥ g a x ↔ x ≤ 1 / 8 :=
sorry

theorem problem_2_a_gt_1 (h : a > 1) : f a x ≥ g a x ↔ x ≥ 1 / 8 :=
sorry

end problem_1_problem_2_0_lt_a_lt_1_problem_2_a_gt_1_l97_97049


namespace units_digit_42_3_plus_27_2_l97_97319

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_42_3_plus_27_2 : units_digit (42^3 + 27^2) = 7 :=
by
  sorry

end units_digit_42_3_plus_27_2_l97_97319


namespace stock_price_return_to_initial_l97_97134

variable (P₀ : ℝ) -- Initial price
variable (y : ℝ) -- Percentage increase during the fourth week

/-- The main theorem stating the required percentage increase in the fourth week -/
theorem stock_price_return_to_initial
  (h1 : P₀ * 1.30 * 0.75 * 1.20 = 117) -- Condition after three weeks
  (h2 : P₃ = P₀) : -- Price returns to initial
  y = -15 := 
by
  sorry

end stock_price_return_to_initial_l97_97134


namespace sum_of_edges_equals_74_l97_97487

def V (pyramid : ℕ) : ℕ := pyramid

def E (pyramid : ℕ) : ℕ := 2 * (V pyramid - 1)

def sum_of_edges (pyramid1 pyramid2 pyramid3 : ℕ) : ℕ :=
  E pyramid1 + E pyramid2 + E pyramid3

theorem sum_of_edges_equals_74 (V₁ V₂ V₃ : ℕ) (h : V₁ + V₂ + V₃ = 40) :
  sum_of_edges V₁ V₂ V₃ = 74 :=
sorry

end sum_of_edges_equals_74_l97_97487


namespace f_log2_9_eq_neg_16_div_9_l97_97216

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x - 2) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2 ^ x

theorem f_log2_9_eq_neg_16_div_9 : f (Real.log 9 / Real.log 2) = -16 / 9 := 
by 
  sorry

end f_log2_9_eq_neg_16_div_9_l97_97216


namespace minimum_value_of_F_l97_97867

noncomputable def F (m n : ℝ) : ℝ := (m - n)^2 + (m^2 - n + 1)^2

theorem minimum_value_of_F : 
  (∀ m n : ℝ, F m n ≥ 9 / 32) ∧ (∃ m n : ℝ, F m n = 9 / 32) :=
by
  sorry

end minimum_value_of_F_l97_97867


namespace lowest_price_per_component_l97_97930

theorem lowest_price_per_component (cost_per_component shipping_per_component fixed_costs num_components : ℕ) 
  (h_cost_per_component : cost_per_component = 80)
  (h_shipping_per_component : shipping_per_component = 5)
  (h_fixed_costs : fixed_costs = 16500)
  (h_num_components : num_components = 150) :
  (cost_per_component + shipping_per_component) * num_components + fixed_costs = 29250 ∧
  29250 / 150 = 195 :=
by
  sorry

end lowest_price_per_component_l97_97930


namespace largest_apartment_size_l97_97912

theorem largest_apartment_size (cost_per_sqft : ℝ) (budget : ℝ) (s : ℝ) 
    (h₁ : cost_per_sqft = 1.20) 
    (h₂ : budget = 600) 
    (h₃ : 1.20 * s = 600) : 
    s = 500 := 
  sorry

end largest_apartment_size_l97_97912


namespace factorize_expression_l97_97786

variable (a : ℝ)

theorem factorize_expression : a^3 + 4 * a^2 + 4 * a = a * (a + 2)^2 := by
  sorry

end factorize_expression_l97_97786


namespace area_of_picture_l97_97895

theorem area_of_picture
  (paper_width : ℝ)
  (paper_height : ℝ)
  (left_margin : ℝ)
  (right_margin : ℝ)
  (top_margin_cm : ℝ)
  (bottom_margin_cm : ℝ)
  (cm_per_inch : ℝ)
  (converted_top_margin : ℝ := top_margin_cm * (1 / cm_per_inch))
  (converted_bottom_margin : ℝ := bottom_margin_cm * (1 / cm_per_inch))
  (picture_width : ℝ := paper_width - left_margin - right_margin)
  (picture_height : ℝ := paper_height - converted_top_margin - converted_bottom_margin)
  (area : ℝ := picture_width * picture_height)
  (h1 : paper_width = 8.5)
  (h2 : paper_height = 10)
  (h3 : left_margin = 1.5)
  (h4 : right_margin = 1.5)
  (h5 : top_margin_cm = 2)
  (h6 : bottom_margin_cm = 2.5)
  (h7 : cm_per_inch = 2.54)
  : area = 45.255925 :=
by sorry

end area_of_picture_l97_97895


namespace cost_of_socks_l97_97379

theorem cost_of_socks (cost_shirt_no_discount cost_pants_no_discount cost_shirt_discounted cost_pants_discounted cost_socks_discounted total_savings team_size socks_cost_no_discount : ℝ) 
    (h1 : cost_shirt_no_discount = 7.5)
    (h2 : cost_pants_no_discount = 15)
    (h3 : cost_shirt_discounted = 6.75)
    (h4 : cost_pants_discounted = 13.5)
    (h5 : cost_socks_discounted = 3.75)
    (h6 : total_savings = 36)
    (h7 : team_size = 12)
    (h8 : 12 * (7.5 + 15 + socks_cost_no_discount) - 12 * (6.75 + 13.5 + 3.75) = 36)
    : socks_cost_no_discount = 4.5 :=
by
  sorry

end cost_of_socks_l97_97379


namespace bisection_method_root_exists_bisection_method_next_calculation_l97_97805

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_method_root_exists :
  (f 0 < 0) → (f 0.5 > 0) → ∃ x0 : ℝ, 0 < x0 ∧ x0 < 0.5 ∧ f x0 = 0 :=
by
  intro h0 h05
  sorry

theorem bisection_method_next_calculation :
  f 0.25 = (0.25)^3 + 3 * 0.25 - 1 :=
by
  calc
    f 0.25 = 0.25^3 + 3 * 0.25 - 1 := rfl

end bisection_method_root_exists_bisection_method_next_calculation_l97_97805


namespace number_of_members_l97_97141

theorem number_of_members (n : ℕ) (h : n * n = 8649) : n = 93 :=
by
  sorry

end number_of_members_l97_97141


namespace average_height_plants_l97_97899

theorem average_height_plants (h1 h3 : ℕ) (h1_eq : h1 = 27) (h3_eq : h3 = 9)
  (prop : ∀ (h2 h4 : ℕ), (h2 = h1 / 3 ∨ h2 = h1 * 3) ∧ (h3 = h2 / 3 ∨ h3 = h2 * 3) ∧ (h4 = h3 / 3 ∨ h4 = h3 * 3)) : 
  ((27 + h2 + 9 + h4) / 4 = 12) :=
by 
  sorry

end average_height_plants_l97_97899


namespace sum_term_addition_l97_97021

theorem sum_term_addition (k : ℕ) (hk : k ≥ 2) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end sum_term_addition_l97_97021


namespace find_b_value_l97_97472

theorem find_b_value (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : b = 8 :=
by
  sorry

end find_b_value_l97_97472


namespace find_focus_of_parabola_l97_97604

-- Define the given parabola equation
def parabola_eqn (x : ℝ) : ℝ := -4 * x^2

-- Define a predicate to check if the point is the focus
def is_focus (x y : ℝ) := x = 0 ∧ y = -1 / 16

theorem find_focus_of_parabola :
  is_focus 0 (parabola_eqn 0) :=
sorry

end find_focus_of_parabola_l97_97604


namespace typist_current_salary_l97_97130

-- Define the initial conditions as given in the problem
def initial_salary : ℝ := 6000
def raise_percentage : ℝ := 0.10
def reduction_percentage : ℝ := 0.05

-- Define the calculations for raised and reduced salaries
def raised_salary := initial_salary * (1 + raise_percentage)
def current_salary := raised_salary * (1 - reduction_percentage)

-- State the theorem to prove the current salary
theorem typist_current_salary : current_salary = 6270 := 
by
  -- Sorry is used to skip proof, overriding with the statement to ensure code builds successfully
  sorry

end typist_current_salary_l97_97130


namespace Problem1_l97_97164

theorem Problem1 (x y : ℝ) (h : x^2 + y^2 = 1) : x^6 + 3*x^2*y^2 + y^6 = 1 := 
by
  sorry

end Problem1_l97_97164


namespace jane_change_l97_97175

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end jane_change_l97_97175


namespace not_perfect_cube_l97_97772

theorem not_perfect_cube (n : ℕ) : ¬ ∃ k : ℕ, k ^ 3 = 2 ^ (2 ^ n) + 1 :=
sorry

end not_perfect_cube_l97_97772


namespace number_of_squares_in_figure_100_l97_97093

theorem number_of_squares_in_figure_100 :
  ∃ (a b c : ℤ), (c = 1) ∧ (a + b + c = 7) ∧ (4 * a + 2 * b + c = 19) ∧ (3 * 100^2 + 3 * 100 + 1 = 30301) :=
sorry

end number_of_squares_in_figure_100_l97_97093


namespace inequality_solution_absolute_inequality_l97_97752

-- Statement for Inequality Solution Problem
theorem inequality_solution (x : ℝ) : |x - 1| + |2 * x + 1| > 3 ↔ (x < -1 ∨ x > 1) := sorry

-- Statement for Absolute Inequality Problem with Bounds
theorem absolute_inequality (a b : ℝ) (ha : -1 ≤ a) (hb : a ≤ 1) (hc : -1 ≤ b) (hd : b ≤ 1) : 
  |1 + (a * b) / 4| > |(a + b) / 2| := sorry

end inequality_solution_absolute_inequality_l97_97752


namespace determine_x_l97_97281

noncomputable def proof_problem (x : ℝ) (y : ℝ) : Prop :=
  y > 0 → 2 * (x * y^2 + x^2 * y + 2 * y^2 + 2 * x * y) / (x + y) > 3 * x^2 * y

theorem determine_x (x : ℝ) : 
  (∀ (y : ℝ), y > 0 → proof_problem x y) ↔ 0 ≤ x ∧ x < (1 + Real.sqrt 13) / 3 := 
sorry

end determine_x_l97_97281


namespace area_of_triangle_l97_97524

open Matrix

def a : Matrix (Fin 2) (Fin 1) ℤ := ![![4], ![-1]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![5]]

theorem area_of_triangle : (abs (a 0 0 * b 1 0 - a 1 0 * b 0 0) : ℚ) / 2 = 23 / 2 :=
by
  -- To be proved (using :ℚ for the cast to rational for division)
  sorry

end area_of_triangle_l97_97524


namespace solve_inner_parentheses_l97_97594

theorem solve_inner_parentheses (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 57 ↔ x = 18 := by
  sorry

end solve_inner_parentheses_l97_97594


namespace general_term_formula_sum_of_b_first_terms_l97_97820

variable (a₁ a₂ : ℝ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Conditions
axiom h1 : a₁ * a₂ = 8
axiom h2 : a₁ + a₂ = 6
axiom increasing_geometric_sequence : ∀ n : ℕ, a (n+1) = a (n) * (a₂ / a₁)
axiom initial_conditions : a 1 = a₁ ∧ a 2 = a₂
axiom b_def : ∀ n, b n = 2 * a n + 3

-- To Prove
theorem general_term_formula : ∀ n: ℕ, a n = 2 ^ (n + 1) :=
sorry

theorem sum_of_b_first_terms (n : ℕ) : T n = 2 ^ (n + 2) - 4 + 3 * n :=
sorry

end general_term_formula_sum_of_b_first_terms_l97_97820


namespace remaining_sum_eq_seven_eighths_l97_97536

noncomputable def sum_series := 
  (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32) + (1 / 64)

noncomputable def removed_terms := 
  (1 / 16) + (1 / 32) + (1 / 64)

theorem remaining_sum_eq_seven_eighths : 
  sum_series - removed_terms = 7 / 8 := by
  sorry

end remaining_sum_eq_seven_eighths_l97_97536


namespace problem_statement_l97_97977

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : (a - c) ^ 3 > (b - c) ^ 3 :=
by
  sorry

end problem_statement_l97_97977


namespace hypotenuse_min_length_l97_97836

theorem hypotenuse_min_length
  (a b l : ℝ)
  (h_area : (1/2) * a * b = 8)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = l)
  (h_min_l : l = 8 + 4 * Real.sqrt 2) :
  Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_min_length_l97_97836


namespace max_product_l97_97586

theorem max_product (a b : ℝ) (h1 : 9 * a ^ 2 + 16 * b ^ 2 = 25) (h2 : a > 0) (h3 : b > 0) :
  a * b ≤ 25 / 24 :=
sorry

end max_product_l97_97586


namespace find_matrix_N_l97_97888

theorem find_matrix_N (N : Matrix (Fin 4) (Fin 4) ℤ)
  (hi : N.mulVec ![1, 0, 0, 0] = ![3, 4, -9, 1])
  (hj : N.mulVec ![0, 1, 0, 0] = ![-1, 6, -3, 2])
  (hk : N.mulVec ![0, 0, 1, 0] = ![8, -2, 5, 0])
  (hl : N.mulVec ![0, 0, 0, 1] = ![1, 0, 7, -1]) :
  N = ![![3, -1, 8, 1],
         ![4, 6, -2, 0],
         ![-9, -3, 5, 7],
         ![1, 2, 0, -1]] := by
  sorry

end find_matrix_N_l97_97888


namespace inequality_1_inequality_2_l97_97939

variable (a b : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom sum_of_cubes_eq_two : a^3 + b^3 = 2

-- Question 1
theorem inequality_1 : (a + b) * (a^5 + b^5) ≥ 4 :=
by
  sorry

-- Question 2
theorem inequality_2 : a + b ≤ 2 :=
by
  sorry

end inequality_1_inequality_2_l97_97939


namespace smallest_integer_cube_ends_in_392_l97_97794

theorem smallest_integer_cube_ends_in_392 : ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 392) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m :=
by 
  sorry

end smallest_integer_cube_ends_in_392_l97_97794


namespace part_I_part_II_l97_97819

-- Problem conditions as definitions
variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a + b = 1)

-- Statement for part (Ⅰ)
theorem part_I : (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Statement for part (Ⅱ)
theorem part_II : (1 / (a ^ 2016)) + (1 / (b ^ 2016)) ≥ 2 ^ 2017 :=
by
  sorry

end part_I_part_II_l97_97819


namespace scientific_notation_of_384000_l97_97035

theorem scientific_notation_of_384000 :
  (384000 : ℝ) = 3.84 * 10^5 :=
by
  sorry

end scientific_notation_of_384000_l97_97035


namespace integer_sequence_count_l97_97298

theorem integer_sequence_count (a₀ : ℕ) (step : ℕ → ℕ) (n : ℕ) 
  (h₀ : a₀ = 5184)
  (h_step : ∀ k, k < n → step k = (a₀ / 4^k))
  (h_stop : a₀ = (4 ^ (n - 1)) * 81) :
  n = 4 := 
sorry

end integer_sequence_count_l97_97298


namespace min_f_in_interval_l97_97263

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) - 2 * sqrt 3 * sin (ω * x / 2) ^ 2 + sqrt 3

theorem min_f_in_interval (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 <= x ∧ x <= π / 2 → f 1 x >= f 1 (π / 3)) :=
by sorry

end min_f_in_interval_l97_97263


namespace didi_total_fund_l97_97218

-- Define the conditions
def cakes : ℕ := 10
def slices_per_cake : ℕ := 8
def price_per_slice : ℕ := 1
def first_business_owner_donation_per_slice : ℚ := 0.5
def second_business_owner_donation_per_slice : ℚ := 0.25

-- Define the proof problem statement
theorem didi_total_fund (h1 : cakes * slices_per_cake = 80)
    (h2 : (80 : ℕ) * price_per_slice = 80)
    (h3 : (80 : ℕ) * first_business_owner_donation_per_slice = 40)
    (h4 : (80 : ℕ) * second_business_owner_donation_per_slice = 20) : 
    (80 : ℕ) + 40 + 20 = 140 := by
  -- The proof itself will be constructed here
  sorry

end didi_total_fund_l97_97218


namespace solution_set_of_xf_x_gt_0_l97_97036

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = - f x
axiom h2 : f 2 = 0
axiom h3 : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x < 0

theorem solution_set_of_xf_x_gt_0 :
  {x : ℝ | x * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by {
  sorry
}

end solution_set_of_xf_x_gt_0_l97_97036


namespace Emilee_earnings_l97_97818

theorem Emilee_earnings (J R_j T R_t E R_e : ℕ) :
  (R_j * J = 35) → 
  (R_t * T = 30) → 
  (R_j * J + R_t * T + R_e * E = 90) → 
  (R_e * E = 25) :=
by
  intros h1 h2 h3
  sorry

end Emilee_earnings_l97_97818


namespace combined_operation_l97_97443

def f (x : ℚ) := (3 / 4) * x
def g (x : ℚ) := (5 / 3) * x

theorem combined_operation (x : ℚ) : g (f x) = (5 / 4) * x :=
by
    unfold f g
    sorry

end combined_operation_l97_97443


namespace cooling_time_condition_l97_97877

theorem cooling_time_condition :
  ∀ (θ0 θ1 θ1' θ0' : ℝ) (t : ℝ), 
    θ0 = 20 → θ1 = 100 → θ1' = 60 → θ0' = 20 →
    let θ := θ0 + (θ1 - θ0) * Real.exp (-t / 4)
    let θ' := θ0' + (θ1' - θ0') * Real.exp (-t / 4)
    (θ - θ' ≤ 10) → (t ≥ 5.52) :=
sorry

end cooling_time_condition_l97_97877


namespace focus_of_parabola_y_eq_8x2_l97_97478

open Real

noncomputable def parabola_focus (a p : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * p))

theorem focus_of_parabola_y_eq_8x2 :
  parabola_focus 8 (1 / 16) = (0, 1 / 32) :=
by
  sorry

end focus_of_parabola_y_eq_8x2_l97_97478


namespace solve_system_l97_97753

theorem solve_system :
  {p : ℝ × ℝ // 
    (p.1 + |p.2| = 3 ∧ 2 * |p.1| - p.2 = 3) ∧
    (p = (2, 1) ∨ p = (0, -3) ∨ p = (-6, 9))} :=
by { sorry }

end solve_system_l97_97753


namespace apogee_reach_second_stage_model_engine_off_time_l97_97020

-- Given conditions
def altitudes := [(0, 0), (1, 24), (2, 96), (4, 386), (5, 514), (6, 616), (9, 850), (13, 994), (14, 1000), (16, 976), (19, 850), (24, 400)]
def second_stage_curve (x : ℝ) : ℝ := -6 * x^2 + 168 * x - 176

-- Proof problems
theorem apogee_reach : (14, 1000) ∈ altitudes :=
sorry  -- Need to prove the inclusion of the apogee point in the table

theorem second_stage_model : 
    second_stage_curve 14 = 1000 ∧ 
    second_stage_curve 16 = 976 ∧ 
    second_stage_curve 19 = 850 ∧ 
    ∃ n, n = 4 :=
sorry  -- Need to prove the analytical expression is correct and n = 4

theorem engine_off_time : 
    ∃ t : ℝ, t = 14 + 5 * Real.sqrt 6 ∧ second_stage_curve t = 100 :=
sorry  -- Need to prove the engine off time calculation

end apogee_reach_second_stage_model_engine_off_time_l97_97020


namespace min_attempts_sufficient_a_l97_97793

theorem min_attempts_sufficient_a (n : ℕ) (h : n > 2)
  (good_batteries bad_batteries : ℕ)
  (h1 : good_batteries = n + 1)
  (h2 : bad_batteries = n)
  (total_batteries := 2 * n + 1) :
  (∃ attempts, attempts = n + 1) := sorry

end min_attempts_sufficient_a_l97_97793


namespace jean_pairs_of_pants_l97_97694

theorem jean_pairs_of_pants
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (number_of_pairs : ℝ)
  (h1 : retail_price = 45)
  (h2 : discount_rate = 0.20)
  (h3 : tax_rate = 0.10)
  (h4 : total_paid = 396)
  (h5 : number_of_pairs = total_paid / ((retail_price * (1 - discount_rate)) * (1 + tax_rate))) :
  number_of_pairs = 10 :=
by
  sorry

end jean_pairs_of_pants_l97_97694


namespace complementary_events_A_B_l97_97100

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def A (n : ℕ) : Prop := is_odd n
def B (n : ℕ) : Prop := is_even n
def C (n : ℕ) : Prop := is_multiple_of_3 n

theorem complementary_events_A_B :
  (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n) ∧ (∀ n, A n ∨ B n) :=
  sorry

end complementary_events_A_B_l97_97100


namespace total_donations_l97_97932

-- Define the conditions
def started_donating_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the proof problem to show the total donation amount equals $432,000
theorem total_donations : (current_age - started_donating_age) * annual_donation = 432000 := 
by
  sorry

end total_donations_l97_97932


namespace greatest_divisor_l97_97780

theorem greatest_divisor (d : ℕ) :
  (1657 % d = 6 ∧ 2037 % d = 5) → d = 127 := by
  sorry

end greatest_divisor_l97_97780


namespace range_of_m_l97_97894

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > g x m) → m < 5 :=
by
  sorry

end range_of_m_l97_97894


namespace eval_diff_squares_l97_97437

theorem eval_diff_squares : 81^2 - 49^2 = 4160 :=
by
  sorry

end eval_diff_squares_l97_97437


namespace no_injective_function_satisfying_conditions_l97_97512

open Real

theorem no_injective_function_satisfying_conditions :
  ¬ ∃ (f : ℝ → ℝ), (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2)
  ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x : ℝ, f (x ^ 2) - (f (a * x + b)) ^ 2 ≥ 1 / 4) :=
by
  sorry

end no_injective_function_satisfying_conditions_l97_97512


namespace solution_set_of_inequality_range_of_a_for_gx_zero_l97_97475

-- Define f(x) and g(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) + abs (x + a)

def g (x : ℝ) (a : ℝ) : ℝ := f x a - abs (3 + a)

-- The first Lean statement
theorem solution_set_of_inequality (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, f x a > 6 ↔ x < -4 ∨ (-3 < x ∧ x < 1) ∨ 2 < x := by
  sorry

-- The second Lean statement
theorem range_of_a_for_gx_zero (a : ℝ) :
  (∃ x : ℝ, g x a = 0) ↔ a ≥ -2 := by
  sorry

end solution_set_of_inequality_range_of_a_for_gx_zero_l97_97475


namespace cos_17pi_over_4_l97_97425

theorem cos_17pi_over_4 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_17pi_over_4_l97_97425


namespace part1_part2_l97_97343

variable (a : ℝ)

-- Proposition A
def propA (a : ℝ) := ∀ x : ℝ, ¬ (x^2 + (2*a-1)*x + a^2 ≤ 0)

-- Proposition B
def propB (a : ℝ) := 0 < a^2 - 1 ∧ a^2 - 1 < 1

theorem part1 (ha : propA a ∨ propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) ∨ (a > 1/4) :=
  sorry

theorem part2 (ha : ¬ propA a) (hb : propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) → (a^3 + 1 < a^2 + a) :=
  sorry

end part1_part2_l97_97343


namespace tan_ratio_alpha_beta_l97_97279

theorem tan_ratio_alpha_beta 
  (α β : ℝ) 
  (h1 : Real.sin (α + β) = 1 / 5) 
  (h2 : Real.sin (α - β) = 3 / 5) : 
  Real.tan α / Real.tan β = -1 :=
sorry

end tan_ratio_alpha_beta_l97_97279


namespace find_a_l97_97237

theorem find_a 
  (a b c : ℚ) 
  (h1 : a + b = c) 
  (h2 : b + c + 2 * b = 11) 
  (h3 : c = 7) :
  a = 17 / 3 :=
by
  sorry

end find_a_l97_97237


namespace longest_segment_CD_l97_97396

theorem longest_segment_CD
  (ABD_angle : ℝ) (ADB_angle : ℝ) (BDC_angle : ℝ) (CBD_angle : ℝ)
  (angle_proof_ABD : ABD_angle = 50)
  (angle_proof_ADB : ADB_angle = 40)
  (angle_proof_BDC : BDC_angle = 35)
  (angle_proof_CBD : CBD_angle = 70) :
  true := 
by
  sorry

end longest_segment_CD_l97_97396


namespace flour_for_recipe_l97_97415

theorem flour_for_recipe (flour_needed shortening_have : ℚ)
  (flour_ratio shortening_ratio : ℚ) 
  (ratio : flour_ratio / shortening_ratio = 5)
  (shortening_used : shortening_ratio = 2 / 3) :
  flour_needed = 10 / 3 := 
by 
  sorry

end flour_for_recipe_l97_97415


namespace range_of_a_l97_97502

def tangent_perpendicular_to_y_axis (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (3 * a * x^2 + 1 / x = 0)

theorem range_of_a : {a : ℝ | tangent_perpendicular_to_y_axis a} = {a : ℝ | a < 0} :=
by
  sorry

end range_of_a_l97_97502


namespace fill_in_the_blank_l97_97190

theorem fill_in_the_blank (x : ℕ) (h : (x - x) + x * x + x / x = 50) : x = 7 :=
sorry

end fill_in_the_blank_l97_97190


namespace ticket_cost_per_ride_l97_97724

theorem ticket_cost_per_ride
  (total_tickets: ℕ) 
  (spent_tickets: ℕ)
  (rides: ℕ)
  (remaining_tickets: ℕ)
  (ride_cost: ℕ)
  (h1: total_tickets = 79)
  (h2: spent_tickets = 23)
  (h3: rides = 8)
  (h4: remaining_tickets = total_tickets - spent_tickets)
  (h5: remaining_tickets = ride_cost * rides):
  ride_cost = 7 :=
by
  sorry

end ticket_cost_per_ride_l97_97724


namespace negation_proposition_l97_97622

theorem negation_proposition :
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - 3 * x + 2 ≤ 0)) =
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 - 3 * x + 2 > 0) := 
sorry

end negation_proposition_l97_97622


namespace cyclists_meet_at_start_point_l97_97086

-- Conditions from the problem
def cyclist1_speed : ℝ := 7 -- speed of the first cyclist in m/s
def cyclist2_speed : ℝ := 8 -- speed of the second cyclist in m/s
def circumference : ℝ := 600 -- circumference of the circular track in meters

-- Relative speed when cyclists move in opposite directions
def relative_speed := cyclist1_speed + cyclist2_speed

-- Prove that they meet at the starting point after 40 seconds
theorem cyclists_meet_at_start_point :
  (circumference / relative_speed) = 40 := by
  -- the proof would go here
  sorry

end cyclists_meet_at_start_point_l97_97086


namespace coefficient_of_x_in_expression_l97_97628

theorem coefficient_of_x_in_expression : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2) + 3 * (x + 4)
  ∃ k : ℤ, (expr = k * x + term) ∧ 
  (∃ coefficient_x : ℤ, coefficient_x = 8) := 
sorry

end coefficient_of_x_in_expression_l97_97628


namespace iron_heating_time_l97_97731

-- Define the conditions as constants
def ironHeatingRate : ℝ := 9 -- degrees Celsius per 20 seconds
def ironCoolingRate : ℝ := 15 -- degrees Celsius per 30 seconds
def coolingTime : ℝ := 180 -- seconds

-- Define the theorem to prove the heating back time
theorem iron_heating_time :
  (coolingTime / 30) * ironCoolingRate = 90 →
  (90 / ironHeatingRate) * 20 = 200 :=
by
  sorry

end iron_heating_time_l97_97731


namespace sphere_tangency_relation_l97_97023

noncomputable def sphere_tangents (r R : ℝ) (h : R > r) :=
  (R >= (2 / (Real.sqrt 3) - 1) * r) ∧
  (∃ x, x = (R * (R + r - Real.sqrt (R^2 + 2 * R * r - r^2 / 3))) /
            (r + Real.sqrt (R^2 + 2 * R * r - r^2 / 3) - R)) 

theorem sphere_tangency_relation (r R: ℝ) (h : R > r) :
  sphere_tangents r R h :=
by
  sorry

end sphere_tangency_relation_l97_97023


namespace polynomial_inequality_l97_97676

noncomputable def F (x a_3 a_2 a_1 k : ℝ) : ℝ :=
  x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + k^4

theorem polynomial_inequality 
  (p k : ℝ) 
  (a_3 a_2 a_1 : ℝ) 
  (h_p : 0 < p) 
  (h_k : 0 < k) 
  (h_roots : ∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧
    F (-x1) a_3 a_2 a_1 k = 0 ∧
    F (-x2) a_3 a_2 a_1 k = 0 ∧
    F (-x3) a_3 a_2 a_1 k = 0 ∧
    F (-x4) a_3 a_2 a_1 k = 0) :
  F p a_3 a_2 a_1 k ≥ (p + k)^4 := 
sorry

end polynomial_inequality_l97_97676


namespace calculate_color_cartridges_l97_97633

theorem calculate_color_cartridges (c b : ℕ) (h1 : 32 * c + 27 * b = 123) (h2 : b ≥ 1) : c = 3 :=
by
  sorry

end calculate_color_cartridges_l97_97633


namespace quadratic_complete_square_l97_97920

theorem quadratic_complete_square (b m : ℝ) (h1 : b > 0)
    (h2 : (x : ℝ) → (x + m)^2 + 8 = x^2 + bx + 20) : b = 4 * Real.sqrt 3 :=
by
  sorry

end quadratic_complete_square_l97_97920


namespace repeating_decimal_ratio_eq_4_l97_97254

-- Definitions for repeating decimals
def rep_dec_36 := 0.36 -- 0.\overline{36}
def rep_dec_09 := 0.09 -- 0.\overline{09}

-- Lean 4 statement of proof problem
theorem repeating_decimal_ratio_eq_4 :
  (rep_dec_36 / rep_dec_09) = 4 :=
sorry

end repeating_decimal_ratio_eq_4_l97_97254


namespace number_of_correct_conclusions_is_two_l97_97959

section AnalogicalReasoning
  variable (a b c : ℝ) (x y : ℂ)

  -- Condition 1: The analogy for distributive property over addition in ℝ and division
  def analogy1 : (c ≠ 0) → ((a + b) * c = a * c + b * c) → (a + b) / c = a / c + b / c := by
    sorry

  -- Condition 2: The analogy for equality of real and imaginary parts in ℂ
  def analogy2 : (x - y = 0) → x = y := by
    sorry

  -- Theorem stating that the number of correct conclusions is 2
  theorem number_of_correct_conclusions_is_two : 2 = 2 := by
    -- which implies that analogy1 and analogy2 are valid, and the other two analogies are not
    sorry

end AnalogicalReasoning

end number_of_correct_conclusions_is_two_l97_97959


namespace ratio_amy_jeremy_l97_97137

variable (Amy Chris Jeremy : ℕ)

theorem ratio_amy_jeremy (h1 : Amy + Jeremy + Chris = 132) (h2 : Jeremy = 66) (h3 : Chris = 2 * Amy) : 
  Amy / Jeremy = 1 / 3 :=
by
  sorry

end ratio_amy_jeremy_l97_97137


namespace meryll_questions_l97_97910

theorem meryll_questions (M P : ℕ) (h1 : (3/5 : ℚ) * M + (2/3 : ℚ) * P = 31) (h2 : P = 15) : M = 35 :=
sorry

end meryll_questions_l97_97910


namespace project_selection_probability_l97_97688

/-- Each employee can randomly select one project from four optional assessment projects. -/
def employees : ℕ := 4

def projects : ℕ := 4

def total_events (e : ℕ) (p : ℕ) : ℕ := p^e

def choose_exactly_one_project_not_selected_probability (e : ℕ) (p : ℕ) : ℚ :=
  (Nat.choose p 2 * Nat.factorial 3) / (p^e : ℚ)

theorem project_selection_probability :
  choose_exactly_one_project_not_selected_probability employees projects = 9 / 16 :=
by
  sorry

end project_selection_probability_l97_97688


namespace perimeter_of_ABFCDE_l97_97560

theorem perimeter_of_ABFCDE {side : ℝ} (h : side = 12) : 
  ∃ perimeter : ℝ, perimeter = 84 :=
by
  sorry

end perimeter_of_ABFCDE_l97_97560


namespace floor_eq_solution_l97_97588

theorem floor_eq_solution (x : ℝ) : 2.5 ≤ x ∧ x < 3.5 → (⌊2 * x + 0.5⌋ = ⌊x + 3⌋) :=
by
  sorry

end floor_eq_solution_l97_97588


namespace inequality1_in_triangle_inequality2_in_triangle_l97_97115

theorem inequality1_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  (13 / 27) * s^2 ≤ a^2 + b^2 + c^2 + (4 / s) * a * b * c ∧ 
  a^2 + b^2 + c^2 + (4 / s) * a * b * c < s^2 / 2 :=
sorry

theorem inequality2_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  s^2 / 4 < a * b + b * c + c * a - (2 / s) * a * b * c ∧ 
  a * b + b * c + c * a - (2 / s) * a * b * c ≤ (7 / 27) * s^2 :=
sorry

end inequality1_in_triangle_inequality2_in_triangle_l97_97115


namespace correct_car_selection_l97_97335

-- Define the production volumes
def production_emgrand : ℕ := 1600
def production_king_kong : ℕ := 6000
def production_freedom_ship : ℕ := 2000

-- Define the total number of cars produced
def total_production : ℕ := production_emgrand + production_king_kong + production_freedom_ship

-- Define the number of cars selected for inspection
def cars_selected_for_inspection : ℕ := 48

-- Calculate the sampling ratio
def sampling_ratio : ℚ := cars_selected_for_inspection / total_production

-- Define the expected number of cars to be selected from each model using the sampling ratio
def cars_selected_emgrand : ℚ := sampling_ratio * production_emgrand
def cars_selected_king_kong : ℚ := sampling_ratio * production_king_kong
def cars_selected_freedom_ship : ℚ := sampling_ratio * production_freedom_ship

theorem correct_car_selection :
  cars_selected_emgrand = 8 ∧ cars_selected_king_kong = 30 ∧ cars_selected_freedom_ship = 10 := by
  sorry

end correct_car_selection_l97_97335


namespace vector_subtraction_l97_97653

-- Definitions of given conditions
def OA : ℝ × ℝ := (2, 1)
def OB : ℝ × ℝ := (-3, 4)

-- Definition of vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_subtraction : vector_sub OB OA = (-5, 3) :=
by 
  -- The proof would go here.
  sorry

end vector_subtraction_l97_97653


namespace jackson_money_l97_97114

theorem jackson_money (W : ℝ) (H1 : 5 * W + W = 150) : 5 * W = 125 :=
by
  sorry

end jackson_money_l97_97114


namespace exponent_calculation_l97_97707

theorem exponent_calculation : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end exponent_calculation_l97_97707


namespace pentagon_square_ratio_l97_97104

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end pentagon_square_ratio_l97_97104


namespace region_of_inequality_l97_97264

theorem region_of_inequality (x y : ℝ) : (x + y - 6 < 0) → y < -x + 6 := by
  sorry

end region_of_inequality_l97_97264


namespace sequence_an_general_formula_sum_bn_formula_l97_97778

variable (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)

axiom seq_Sn_eq_2an_minus_n : ∀ n : ℕ, n > 0 → S n + n = 2 * a n

theorem sequence_an_general_formula (n : ℕ) (h : n > 0) :
  (∀ n > 0, a n + 1 = 2 * (a (n - 1) + 1)) ∧ (a n = 2^n - 1) :=
sorry

theorem sum_bn_formula (n : ℕ) (h : n > 0) :
  (∀ n > 0, b n = n * a n + n) → T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end sequence_an_general_formula_sum_bn_formula_l97_97778


namespace painted_cube_l97_97014

noncomputable def cube_side_length : ℕ :=
  7

theorem painted_cube (painted_faces: ℕ) (one_side_painted_cubes: ℕ) (orig_side_length: ℕ) :
    painted_faces = 6 ∧ one_side_painted_cubes = 54 ∧ (orig_side_length + 2) ^ 2 / 6 = 9 →
    orig_side_length = cube_side_length :=
by
  sorry

end painted_cube_l97_97014


namespace area_ratio_of_octagon_l97_97406

theorem area_ratio_of_octagon (A : ℝ) (hA : 0 < A) :
  let triangle_ABJ_area := A / 8
  let triangle_ACE_area := A / 2
  triangle_ABJ_area / triangle_ACE_area = 1 / 4 := by
  sorry

end area_ratio_of_octagon_l97_97406


namespace domain_of_function_l97_97591

theorem domain_of_function :
  {x : ℝ | (x + 1 ≥ 0) ∧ (2 - x ≠ 0)} = {x : ℝ | -1 ≤ x ∧ x ≠ 2} :=
by {
  sorry
}

end domain_of_function_l97_97591


namespace no_solution_for_conditions_l97_97414

theorem no_solution_for_conditions :
  ∀ (x y : ℝ), 0 < x → 0 < y → x * y = 2^15 → (Real.log x / Real.log 2) * (Real.log y / Real.log 2) = 60 → False :=
by
  intro x y x_pos y_pos h1 h2
  sorry

end no_solution_for_conditions_l97_97414


namespace dry_mixed_fruits_weight_l97_97784

theorem dry_mixed_fruits_weight :
  ∀ (fresh_grapes_weight fresh_apples_weight : ℕ)
    (grapes_water_content fresh_grapes_dry_matter_perc : ℕ)
    (apples_water_content fresh_apples_dry_matter_perc : ℕ),
    fresh_grapes_weight = 400 →
    fresh_apples_weight = 300 →
    grapes_water_content = 65 →
    fresh_grapes_dry_matter_perc = 35 →
    apples_water_content = 84 →
    fresh_apples_dry_matter_perc = 16 →
    (fresh_grapes_weight * fresh_grapes_dry_matter_perc / 100) +
    (fresh_apples_weight * fresh_apples_dry_matter_perc / 100) = 188 := by
  sorry

end dry_mixed_fruits_weight_l97_97784


namespace sum_of_ab_conditions_l97_97775

theorem sum_of_ab_conditions (a b : ℝ) (h : a^3 + b^3 = 1 - 3 * a * b) : a + b = 1 ∨ a + b = -2 := 
by
  sorry

end sum_of_ab_conditions_l97_97775


namespace semicircle_radius_l97_97743

theorem semicircle_radius (P L W : ℝ) (π : Real) (r : ℝ) 
  (hP : P = 144) (hL : L = 48) (hW : W = 24) (hD : ∃ d, d = 2 * r ∧ d = L) :
  r = 48 / (π + 2) := 
by
  sorry

end semicircle_radius_l97_97743


namespace robin_made_more_cupcakes_l97_97324

theorem robin_made_more_cupcakes (initial final sold made: ℕ)
  (h1 : initial = 42)
  (h2 : sold = 22)
  (h3 : final = 59)
  (h4 : initial - sold + made = final) :
  made = 39 :=
  sorry

end robin_made_more_cupcakes_l97_97324


namespace karen_average_speed_l97_97103

noncomputable def total_distance : ℚ := 198
noncomputable def start_time : ℚ := (9 * 60 + 40) / 60
noncomputable def end_time : ℚ := (13 * 60 + 20) / 60
noncomputable def total_time : ℚ := end_time - start_time
noncomputable def average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed :
  average_speed total_distance total_time = 54 := by
  sorry

end karen_average_speed_l97_97103


namespace hyperbola_equation_l97_97887

theorem hyperbola_equation 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_asymptote : (b / a) = (Real.sqrt 3 / 2))
  (c : ℝ) (hc : c = Real.sqrt 7)
  (foci_directrix_condition : a^2 + b^2 = c^2) :
  (∀ x y : ℝ, (x^2 / 4 - y^2 / 3 = 1)) :=
by
  -- We do not provide the proof as per instructions
  sorry

end hyperbola_equation_l97_97887


namespace exists_invisible_square_l97_97320

def invisible (p q : ℤ) : Prop := Int.gcd p q > 1

theorem exists_invisible_square (n : ℤ) (h : 0 < n) : 
  ∃ (a b : ℤ), ∀ i j : ℤ, (0 ≤ i) ∧ (i < n) ∧ (0 ≤ j) ∧ (j < n) → invisible (a + i) (b + j) :=
by {
  sorry
}

end exists_invisible_square_l97_97320


namespace triangle_pentagon_side_ratio_l97_97519

theorem triangle_pentagon_side_ratio :
  let triangle_perimeter := 60
  let pentagon_perimeter := 60
  let triangle_side := triangle_perimeter / 3
  let pentagon_side := pentagon_perimeter / 5
  (triangle_side : ℕ) / (pentagon_side : ℕ) = 5 / 3 :=
by
  sorry

end triangle_pentagon_side_ratio_l97_97519


namespace floral_shop_bouquets_total_l97_97886

theorem floral_shop_bouquets_total (sold_monday_rose : ℕ) (sold_monday_lily : ℕ) (sold_monday_orchid : ℕ)
  (price_monday_rose : ℕ) (price_monday_lily : ℕ) (price_monday_orchid : ℕ)
  (sold_tuesday_rose : ℕ) (sold_tuesday_lily : ℕ) (sold_tuesday_orchid : ℕ)
  (price_tuesday_rose : ℕ) (price_tuesday_lily : ℕ) (price_tuesday_orchid : ℕ)
  (sold_wednesday_rose : ℕ) (sold_wednesday_lily : ℕ) (sold_wednesday_orchid : ℕ)
  (price_wednesday_rose : ℕ) (price_wednesday_lily : ℕ) (price_wednesday_orchid : ℕ)
  (H1 : sold_monday_rose = 12) (H2 : sold_monday_lily = 8) (H3 : sold_monday_orchid = 6)
  (H4 : price_monday_rose = 10) (H5 : price_monday_lily = 15) (H6 : price_monday_orchid = 20)
  (H7 : sold_tuesday_rose = 3 * sold_monday_rose) (H8 : sold_tuesday_lily = 2 * sold_monday_lily)
  (H9 : sold_tuesday_orchid = sold_monday_orchid / 2) (H10 : price_tuesday_rose = 12)
  (H11 : price_tuesday_lily = 18) (H12 : price_tuesday_orchid = 22)
  (H13 : sold_wednesday_rose = sold_tuesday_rose / 3) (H14 : sold_wednesday_lily = sold_tuesday_lily / 4)
  (H15 : sold_wednesday_orchid = 2 * sold_tuesday_orchid / 3) (H16 : price_wednesday_rose = 8)
  (H17 : price_wednesday_lily = 12) (H18 : price_wednesday_orchid = 16) :
  (sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose = 60) ∧
  (sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily = 28) ∧
  (sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid = 11) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose) = 648) ∧
  ((sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily) = 456) ∧
  ((sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 218) ∧
  ((sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose + sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily + sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid) = 99) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose + sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily + sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 1322) :=
  by sorry

end floral_shop_bouquets_total_l97_97886


namespace total_baseball_fans_l97_97486

-- Conditions given
def ratio_YM (Y M : ℕ) : Prop := 2 * Y = 3 * M
def ratio_MR (M R : ℕ) : Prop := 4 * R = 5 * M
def M_value : ℕ := 88

-- Prove total number of baseball fans
theorem total_baseball_fans (Y M R : ℕ) (h1 : ratio_YM Y M) (h2 : ratio_MR M R) (hM : M = M_value) :
  Y + M + R = 330 :=
sorry

end total_baseball_fans_l97_97486


namespace landscape_breadth_l97_97994

theorem landscape_breadth (L B : ℝ) 
  (h1 : B = 6 * L) 
  (h2 : L * B = 29400) : 
  B = 420 :=
by
  sorry

end landscape_breadth_l97_97994


namespace white_ducks_count_l97_97582

theorem white_ducks_count (W : ℕ) : 
  (5 * W + 10 * 7 + 12 * 6 = 157) → W = 3 :=
by
  sorry

end white_ducks_count_l97_97582


namespace value_of_expression_l97_97925

theorem value_of_expression (m a b c d : ℚ) 
  (hm : |m + 1| = 4)
  (hab : a = -b) 
  (hcd : c * d = 1) :
  a + b + 3 * c * d - m = 0 ∨ a + b + 3 * c * d - m = 8 :=
by
  sorry

end value_of_expression_l97_97925


namespace sum_of_product_of_consecutive_numbers_divisible_by_12_l97_97725

theorem sum_of_product_of_consecutive_numbers_divisible_by_12 (a : ℤ) : 
  (a * (a + 1) + (a + 1) * (a + 2) + (a + 2) * (a + 3) + a * (a + 3) + 1) % 12 = 0 :=
by sorry

end sum_of_product_of_consecutive_numbers_divisible_by_12_l97_97725


namespace C_share_l97_97179

theorem C_share (a b c : ℕ) (h1 : a + b + c = 1010)
                (h2 : ∃ k : ℕ, a = 3 * k + 25 ∧ b = 2 * k + 10 ∧ c = 5 * k + 15) : c = 495 :=
by
  -- Sorry is used to skip the proof
  sorry

end C_share_l97_97179


namespace students_at_end_of_year_l97_97221

def students_start : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0
def students_end : ℝ := 28.0

theorem students_at_end_of_year :
  students_start - students_left - students_transferred = students_end := by
  sorry

end students_at_end_of_year_l97_97221


namespace sara_has_total_quarters_l97_97990

-- Define the number of quarters Sara originally had
def original_quarters : ℕ := 21

-- Define the number of quarters Sara's dad gave her
def added_quarters : ℕ := 49

-- Define the total number of quarters Sara has now
def total_quarters : ℕ := original_quarters + added_quarters

-- Prove that the total number of quarters is 70
theorem sara_has_total_quarters : total_quarters = 70 := by
  -- This is where the proof would go
  sorry

end sara_has_total_quarters_l97_97990


namespace multiplication_result_l97_97625

theorem multiplication_result :
  (500 ^ 50) * (2 ^ 100) = 10 ^ 75 :=
by
  sorry

end multiplication_result_l97_97625


namespace valid_cone_from_sector_l97_97010

-- Given conditions
def sector_angle : ℝ := 300
def circle_radius : ℝ := 15

-- Definition of correct option E
def base_radius_E : ℝ := 12
def slant_height_E : ℝ := 15

theorem valid_cone_from_sector :
  ( (sector_angle / 360) * (2 * Real.pi * circle_radius) = 25 * Real.pi ) ∧
  (slant_height_E = circle_radius) ∧
  (base_radius_E = 12) ∧
  (15^2 = 12^2 + 9^2) :=
by
  -- This theorem states that given sector angle and circle radius, the valid option is E
  sorry

end valid_cone_from_sector_l97_97010


namespace convert_3241_quinary_to_septenary_l97_97232

/-- Convert quinary number 3241_(5) to septenary number, yielding 1205_(7). -/
theorem convert_3241_quinary_to_septenary : 
  let quinary := 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0
  let septenary := 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0
  quinary = 446 → septenary = 1205 :=
by
  intros
  -- Quinary to Decimal
  have h₁ : 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0 = 446 := by norm_num
  -- Decimal to Septenary
  have h₂ : 446 = 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0 := by norm_num
  exact sorry

end convert_3241_quinary_to_septenary_l97_97232


namespace smallest_n_divisible_l97_97016

theorem smallest_n_divisible (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end smallest_n_divisible_l97_97016


namespace meaningful_fraction_l97_97741

theorem meaningful_fraction (x : ℝ) : (∃ (f : ℝ), f = 2 / x) ↔ x ≠ 0 :=
by
  sorry

end meaningful_fraction_l97_97741


namespace fraction_of_male_gerbils_is_correct_l97_97449

def total_pets := 90
def total_gerbils := 66
def total_hamsters := total_pets - total_gerbils
def fraction_hamsters_male := 1/3
def total_males := 25
def male_hamsters := fraction_hamsters_male * total_hamsters
def male_gerbils := total_males - male_hamsters
def fraction_gerbils_male := male_gerbils / total_gerbils

theorem fraction_of_male_gerbils_is_correct : fraction_gerbils_male = 17 / 66 := by
  sorry

end fraction_of_male_gerbils_is_correct_l97_97449


namespace solve_equation_l97_97663

theorem solve_equation (x y : ℤ) (eq : (x^2 - y^2)^2 = 16 * y + 1) : 
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ 
  (x = 4 ∧ y = 3) ∨ (x = -4 ∧ y = 3) ∨ 
  (x = 4 ∧ y = 5) ∨ (x = -4 ∧ y = 5) :=
sorry

end solve_equation_l97_97663


namespace matrix_solution_correct_l97_97184

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, -7/3], ![4, -1/3]]

def v1 : Fin 2 → ℚ := ![4, 0]
def v2 : Fin 2 → ℚ := ![2, 3]

def result1 : Fin 2 → ℚ := ![12, 16]
def result2 : Fin 2 → ℚ := ![-1, 7]

theorem matrix_solution_correct :
  (mulVec N v1 = result1) ∧ 
  (mulVec N v2 = result2) := by
  sorry

end matrix_solution_correct_l97_97184


namespace area_of_fourth_rectangle_l97_97839

theorem area_of_fourth_rectangle (A B C D E F G H I J K L : Type) 
  (x y z w : ℕ) (a1 : x * y = 20) (a2 : x * w = 12) (a3 : z * w = 16) : 
  y * w = 16 :=
by sorry

end area_of_fourth_rectangle_l97_97839


namespace seventh_term_in_geometric_sequence_l97_97344

theorem seventh_term_in_geometric_sequence :
  ∃ r, (4 * r^8 = 2097152) ∧ (4 * r^6 = 1048576) :=
by
  sorry

end seventh_term_in_geometric_sequence_l97_97344


namespace nishita_common_shares_l97_97787

def annual_dividend_preferred_shares (num_preferred_shares : ℕ) (par_value : ℕ) (dividend_rate_preferred : ℕ) : ℕ :=
  (dividend_rate_preferred * par_value * num_preferred_shares) / 100

def annual_dividend_common_shares (total_dividend : ℕ) (dividend_preferred : ℕ) : ℕ :=
  total_dividend - dividend_preferred

def number_of_common_shares (annual_dividend_common : ℕ) (par_value : ℕ) (annual_rate_common : ℕ) : ℕ :=
  annual_dividend_common / ((annual_rate_common * par_value) / 100)

theorem nishita_common_shares (total_annual_dividend : ℕ) (num_preferred_shares : ℕ)
                             (par_value : ℕ) (dividend_rate_preferred : ℕ)
                             (semi_annual_rate_common : ℕ) : 
                             (number_of_common_shares (annual_dividend_common_shares total_annual_dividend 
                             (annual_dividend_preferred_shares num_preferred_shares par_value dividend_rate_preferred)) 
                             par_value (semi_annual_rate_common * 2)) = 3000 :=
by
  -- Provide values specific to the problem
  let total_annual_dividend := 16500
  let num_preferred_shares := 1200
  let par_value := 50
  let dividend_rate_preferred := 10
  let semi_annual_rate_common := 3.5
  sorry

end nishita_common_shares_l97_97787


namespace dice_probability_l97_97062

theorem dice_probability :
  let one_digit_prob := 9 / 20
  let two_digit_prob := 11 / 20
  let number_of_dice := 5
  ∃ p : ℚ,
    (number_of_dice.choose 2) * (one_digit_prob ^ 2) * (two_digit_prob ^ 3) = p ∧
    p = 107811 / 320000 :=
by
  sorry

end dice_probability_l97_97062


namespace inequality_solution_l97_97764

theorem inequality_solution (x : ℝ) : x^3 - 9 * x^2 + 27 * x > 0 → (x > 0 ∧ x < 3) ∨ (x > 6) := sorry

end inequality_solution_l97_97764


namespace range_of_a_iff_l97_97413

def cubic_inequality (x : ℝ) : Prop := x^3 + 3 * x^2 - x - 3 > 0

def quadratic_inequality (x a : ℝ) : Prop := x^2 - 2 * a * x - 1 ≤ 0

def integer_solution_condition (x : ℤ) (a : ℝ) : Prop := 
  x^3 + 3 * x^2 - x - 3 > 0 ∧ x^2 - 2 * a * x - 1 ≤ 0

def range_of_a (a : ℝ) : Prop := (3 / 4 : ℝ) ≤ a ∧ a < (4 / 3 : ℝ)

theorem range_of_a_iff : 
  (∃ x : ℤ, integer_solution_condition x a) ↔ range_of_a a := 
sorry

end range_of_a_iff_l97_97413


namespace neg_triangle_obtuse_angle_l97_97070

theorem neg_triangle_obtuse_angle : 
  (¬ ∀ (A B C : ℝ), A + B + C = π → max (max A B) C < π/2) ↔ (∃ (A B C : ℝ), A + B + C = π ∧ min (min A B) C > π/2) :=
by
  sorry

end neg_triangle_obtuse_angle_l97_97070


namespace sam_time_to_cover_distance_l97_97505

/-- Define the total distance between points A and B as the sum of distances from A to C and C to B -/
def distance_A_to_C : ℕ := 600
def distance_C_to_B : ℕ := 400
def speed_sam : ℕ := 50
def distance_A_to_B : ℕ := distance_A_to_C + distance_C_to_B

theorem sam_time_to_cover_distance :
  let time := distance_A_to_B / speed_sam
  time = 20 := 
by
  sorry

end sam_time_to_cover_distance_l97_97505


namespace smallest_tax_amount_is_professional_income_tax_l97_97407

def total_income : ℝ := 50000.00
def professional_deductions : ℝ := 35000.00

def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_expenditure : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

def ndfl_tax : ℝ := (total_income - professional_deductions) * tax_rate_ndfl
def simplified_tax_income : ℝ := total_income * tax_rate_simplified_income
def simplified_tax_income_minus_expenditure : ℝ := (total_income - professional_deductions) * tax_rate_simplified_income_minus_expenditure
def professional_income_tax : ℝ := total_income * tax_rate_professional_income

theorem smallest_tax_amount_is_professional_income_tax : 
  min (min ndfl_tax (min simplified_tax_income simplified_tax_income_minus_expenditure)) professional_income_tax = professional_income_tax := 
sorry

end smallest_tax_amount_is_professional_income_tax_l97_97407


namespace distance_AD_btw_41_and_42_l97_97182

noncomputable def distance_between (x y : ℝ × ℝ) : ℝ :=
  Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

theorem distance_AD_btw_41_and_42 :
  let A := (0, 0)
  let B := (15, 0)
  let C := (15, 5 * Real.sqrt 3)
  let D := (15, 5 * Real.sqrt 3 + 30)

  41 < distance_between A D ∧ distance_between A D < 42 :=
by
  sorry

end distance_AD_btw_41_and_42_l97_97182


namespace john_billed_for_28_minutes_l97_97734

variable (monthlyFee : ℝ) (costPerMinute : ℝ) (totalBill : ℝ)
variable (minutesBilled : ℝ)

def is_billed_correctly (monthlyFee totalBill costPerMinute minutesBilled : ℝ) : Prop :=
  totalBill - monthlyFee = minutesBilled * costPerMinute ∧ minutesBilled = 28

theorem john_billed_for_28_minutes : 
  is_billed_correctly 5 12.02 0.25 28 := 
by
  sorry

end john_billed_for_28_minutes_l97_97734


namespace xn_plus_inv_xn_l97_97136

theorem xn_plus_inv_xn (θ : ℝ) (x : ℝ) (n : ℕ) (h₀ : 0 < θ) (h₁ : θ < π / 2)
  (h₂ : x + 1 / x = -2 * Real.sin θ) (hn_pos : 0 < n) :
  x ^ n + x⁻¹ ^ n = -2 * Real.sin (n * θ) := by
  sorry

end xn_plus_inv_xn_l97_97136


namespace expand_product_l97_97715

theorem expand_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * (7 / x^3 - 14 * x^4) = 3 / x^3 - 6 * x^4 :=
by
  sorry

end expand_product_l97_97715


namespace value_of_x_l97_97050

theorem value_of_x (x : ℕ) (h : x + (10 * x + x) = 12) : x = 1 := by
  sorry

end value_of_x_l97_97050


namespace part1_minimum_b_over_a_l97_97307

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

-- Prove part 1
theorem part1 (x : ℝ) : (0 < x ∧ x < 1 → (f x 1 / (1/x - 1) > 0)) ∧ (1 < x → (f x 1 / (1/x - 1) < 0)) := sorry

-- Prove part 2
lemma part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) (ha : a ≠ 0) : ∃ x > 0, f x a = b - a := sorry

theorem minimum_b_over_a (a : ℝ) (ha : a ≠ 0) (h : ∀ x > 0, f x a ≤ b - a) : b/a ≥ 0 := sorry

end part1_minimum_b_over_a_l97_97307


namespace hispanic_population_in_west_l97_97937

theorem hispanic_population_in_west (p_NE p_MW p_South p_West : ℕ)
  (h_NE : p_NE = 4)
  (h_MW : p_MW = 5)
  (h_South : p_South = 12)
  (h_West : p_West = 20) :
  ((p_West : ℝ) / (p_NE + p_MW + p_South + p_West : ℝ)) * 100 = 49 :=
by sorry

end hispanic_population_in_west_l97_97937


namespace rabbit_speed_correct_l97_97300

-- Define the conditions given in the problem
def rabbit_speed (x : ℝ) : Prop :=
2 * (2 * x + 4) = 188

-- State the main theorem using the defined conditions
theorem rabbit_speed_correct : ∃ x : ℝ, rabbit_speed x ∧ x = 45 :=
by
  sorry

end rabbit_speed_correct_l97_97300


namespace problem_equivalence_of_angles_l97_97076

noncomputable def ctg (x : ℝ) : ℝ := 1 / (Real.tan x)

theorem problem_equivalence_of_angles
  (a b c t S ω : ℝ)
  (hS : S = Real.sqrt ((a^2 + b^2 + c^2)^2 + (4 * t)^2))
  (h1 : ctg ω = (a^2 + b^2 + c^2) / (4 * t))
  (h2 : Real.cos ω = (a^2 + b^2 + c^2) / S)
  (h3 : Real.sin ω = (4 * t) / S) :
  True :=
sorry

end problem_equivalence_of_angles_l97_97076


namespace find_n_solution_l97_97033

theorem find_n_solution : ∃ n : ℤ, (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n : ℝ) / (n + 1 : ℝ) = 3) :=
by
  use 0
  sorry

end find_n_solution_l97_97033


namespace solve_3_pow_n_plus_55_eq_m_squared_l97_97516

theorem solve_3_pow_n_plus_55_eq_m_squared :
  ∃ (n m : ℕ), 3^n + 55 = m^2 ∧ ((n = 2 ∧ m = 8) ∨ (n = 6 ∧ m = 28)) :=
by
  sorry

end solve_3_pow_n_plus_55_eq_m_squared_l97_97516


namespace T_100_gt_T_99_l97_97543

-- Definition: T(n) denotes the number of ways to place n objects of weights 1, 2, ..., n on a balance such that the sum of the weights in each pan is the same.
def T (n : ℕ) : ℕ := sorry

-- Theorem we need to prove
theorem T_100_gt_T_99 : T 100 > T 99 := 
sorry

end T_100_gt_T_99_l97_97543


namespace johns_apartment_number_l97_97902

theorem johns_apartment_number (car_reg : Nat) (apartment_num : Nat) 
  (h_car_reg_sum : car_reg = 834205) 
  (h_car_digits : (8 + 3 + 4 + 2 + 0 + 5 = 22)) 
  (h_apartment_digits : ∃ (d1 d2 d3 : Nat), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 + d2 + d3 = 22) :
  apartment_num = 985 :=
by
  sorry

end johns_apartment_number_l97_97902


namespace digit_solve_l97_97194

theorem digit_solve : ∀ (D : ℕ), D < 10 → (D * 9 + 6 = D * 10 + 3) → D = 3 :=
by
  intros D hD h
  sorry

end digit_solve_l97_97194


namespace range_u_of_given_condition_l97_97509

theorem range_u_of_given_condition (x y : ℝ) (h : x^2 / 3 + y^2 = 1) :
  1 ≤ |2 * x + y - 4| + |3 - x - 2 * y| ∧ |2 * x + y - 4| + |3 - x - 2 * y| ≤ 13 := 
sorry

end range_u_of_given_condition_l97_97509


namespace radius_of_arch_bridge_l97_97331

theorem radius_of_arch_bridge :
  ∀ (AB CD AD r : ℝ),
    AB = 12 →
    CD = 4 →
    AD = AB / 2 →
    r^2 = AD^2 + (r - CD)^2 →
    r = 6.5 :=
by
  intros AB CD AD r hAB hCD hAD h_eq
  sorry

end radius_of_arch_bridge_l97_97331


namespace range_of_a_min_value_ab_range_of_y_l97_97868
-- Import the necessary Lean library 

-- Problem 1
theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) → (-2 ≤ a ∧ a ≤ 1) := 
sorry

-- Problem 2
theorem min_value_ab (a b : ℝ) (h₁ : a + b = 1) : 
  (∀ x, |x - 1| + |x - 3| ≥ a^2 + a) → 
  (min ((1 : ℝ) / (4 * |b|) + |b| / a) = 3 / 4 ∧ (a = 2)) :=
sorry

-- Problem 3
theorem range_of_y (a : ℝ) (y : ℝ) (h₁ : a ∈ Set.Ici (2 : ℝ)) : 
  y = (2 * a) / (a^2 + 1) → 0 < y ∧ y ≤ (4 / 5) :=
sorry

end range_of_a_min_value_ab_range_of_y_l97_97868


namespace grain_output_scientific_notation_l97_97679

theorem grain_output_scientific_notation :
    682.85 * 10^6 = 6.8285 * 10^8 := 
by sorry

end grain_output_scientific_notation_l97_97679


namespace dig_site_date_l97_97841

theorem dig_site_date (S1 S2 S3 S4 : ℕ) (S2_bc : S2 = 852) 
  (h1 : S1 = S2 - 352) 
  (h2 : S3 = S1 + 3700) 
  (h3 : S4 = 2 * S3) : 
  S4 = 6400 :=
by sorry

end dig_site_date_l97_97841


namespace investment_return_l97_97856

theorem investment_return (y_r : ℝ) :
  (500 + 1500) * 0.085 = 500 * 0.07 + 1500 * y_r → y_r = 0.09 :=
by
  sorry

end investment_return_l97_97856


namespace total_cookies_dropped_throughout_entire_baking_process_l97_97933

def initially_baked_by_alice := 74 + 45 + 15
def initially_baked_by_bob := 7 + 32 + 18

def initially_dropped_by_alice := 5 + 8
def initially_dropped_by_bob := 10 + 6

def additional_baked_by_alice := 5 + 4 + 12
def additional_baked_by_bob := 22 + 36 + 14

def edible_cookies := 145

theorem total_cookies_dropped_throughout_entire_baking_process :
  initially_baked_by_alice + initially_baked_by_bob +
  additional_baked_by_alice + additional_baked_by_bob -
  edible_cookies = 139 := by
  sorry

end total_cookies_dropped_throughout_entire_baking_process_l97_97933


namespace fraction_sum_reciprocal_ge_two_l97_97854

theorem fraction_sum_reciprocal_ge_two (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a / b) + (b / a) ≥ 2 :=
sorry

end fraction_sum_reciprocal_ge_two_l97_97854
