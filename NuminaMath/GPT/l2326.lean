import Mathlib

namespace polynomial_evaluation_l2326_232617

theorem polynomial_evaluation (x : ℤ) (h : x = 2) : 3 * x^2 + 5 * x - 2 = 20 := by
  sorry

end polynomial_evaluation_l2326_232617


namespace quadrilateral_angles_combinations_pentagon_angles_combination_l2326_232653

-- Define angle types
inductive AngleType
| acute
| right
| obtuse

open AngleType

-- Define predicates for sum of angles in a quadrilateral and pentagon
def quadrilateral_sum (angles : List AngleType) : Bool :=
  match angles with
  | [right, right, right, right] => true
  | [right, right, acute, obtuse] => true
  | [right, acute, obtuse, obtuse] => true
  | [right, acute, acute, obtuse] => true
  | [acute, obtuse, obtuse, obtuse] => true
  | [acute, acute, obtuse, obtuse] => true
  | [acute, acute, acute, obtuse] => true
  | _ => false

def pentagon_sum (angles : List AngleType) : Prop :=
  -- Broad statement, more complex combinations possible
  ∃ a b c d e : ℕ, (a + b + c + d + e = 540) ∧
    (a < 90 ∨ a = 90 ∨ a > 90) ∧
    (b < 90 ∨ b = 90 ∨ b > 90) ∧
    (c < 90 ∨ c = 90 ∨ c > 90) ∧
    (d < 90 ∨ d = 90 ∨ d > 90) ∧
    (e < 90 ∨ e = 90 ∨ e > 90)

-- Prove the possible combinations for a quadrilateral and a pentagon
theorem quadrilateral_angles_combinations {angles : List AngleType} :
  quadrilateral_sum angles = true :=
sorry

theorem pentagon_angles_combination :
  ∃ angles : List AngleType, pentagon_sum angles :=
sorry

end quadrilateral_angles_combinations_pentagon_angles_combination_l2326_232653


namespace find_cost_price_of_radio_l2326_232685

def cost_price_of_radio
  (profit_percent: ℝ) (overhead_expenses: ℝ) (selling_price: ℝ) (C: ℝ) : Prop :=
  profit_percent = ((selling_price - (C + overhead_expenses)) / C) * 100

theorem find_cost_price_of_radio :
  cost_price_of_radio 21.457489878542503 15 300 234.65 :=
by
  sorry

end find_cost_price_of_radio_l2326_232685


namespace liquid_x_percentage_l2326_232631

theorem liquid_x_percentage 
  (percentage_a : ℝ) (percentage_b : ℝ)
  (weight_a : ℝ) (weight_b : ℝ)
  (h1 : percentage_a = 0.8)
  (h2 : percentage_b = 1.8)
  (h3 : weight_a = 400)
  (h4 : weight_b = 700) :
  (weight_a * (percentage_a / 100) + weight_b * (percentage_b / 100)) / (weight_a + weight_b) * 100 = 1.44 := 
by
  sorry

end liquid_x_percentage_l2326_232631


namespace greatest_possible_value_of_x_l2326_232650

theorem greatest_possible_value_of_x
    (x : ℕ)
    (h1 : x > 0)
    (h2 : x % 4 = 0)
    (h3 : x^3 < 8000) :
    x ≤ 16 :=
    sorry

end greatest_possible_value_of_x_l2326_232650


namespace sum_of_digits_in_rectangle_l2326_232639

theorem sum_of_digits_in_rectangle :
  ∃ A B C : ℕ,
    (4 + A + 1 + B = 12) ∧
    (4 + A + 1 + B = 6 + 6) ∧
    (C + 1 + 6 + C = 11) ∧
    (1 + B + 2 + C = 9) ∧
    (A + 8 + 8 = 8) ∧
    (A + 8 + B + 2 + C = 13) ∧
    (9 = 4 + A + 1 + B) ∧
    (B + 2 + C = 9) ∧    
    B = 5 ∧ A = 2 ∧ C = 6 :=
sorry

end sum_of_digits_in_rectangle_l2326_232639


namespace inequality_solution_range_l2326_232646

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end inequality_solution_range_l2326_232646


namespace l_shape_area_is_42_l2326_232633

-- Defining the dimensions of the larger rectangle
def large_rect_length : ℕ := 10
def large_rect_width : ℕ := 7

-- Defining the smaller rectangle dimensions based on the given conditions
def small_rect_length : ℕ := large_rect_length - 3
def small_rect_width : ℕ := large_rect_width - 3

-- Defining the areas of the rectangles
def large_rect_area : ℕ := large_rect_length * large_rect_width
def small_rect_area : ℕ := small_rect_length * small_rect_width

-- Defining the area of the "L" shape
def l_shape_area : ℕ := large_rect_area - small_rect_area

-- The theorem to prove
theorem l_shape_area_is_42 : l_shape_area = 42 :=
by
  sorry

end l_shape_area_is_42_l2326_232633


namespace divisible_by_11_l2326_232605

theorem divisible_by_11 (k : ℕ) (h : 0 ≤ k ∧ k ≤ 9) :
  (9 + 4 + 5 + k + 3 + 1 + 7) - 2 * (4 + k + 1) ≡ 0 [MOD 11] → k = 8 :=
by
  sorry

end divisible_by_11_l2326_232605


namespace tan_value_l2326_232620

theorem tan_value (θ : ℝ) (h : Real.sin (12 * Real.pi / 5 + θ) + 2 * Real.sin (11 * Real.pi / 10 - θ) = 0) :
  Real.tan (2 * Real.pi / 5 + θ) = 2 :=
by
  sorry

end tan_value_l2326_232620


namespace maxRegions_formula_l2326_232675

-- Define the maximum number of regions in the plane given by n lines
def maxRegions (n: ℕ) : ℕ := (n^2 + n + 2) / 2

-- Main theorem to prove
theorem maxRegions_formula (n : ℕ) : maxRegions n = (n^2 + n + 2) / 2 := by 
  sorry

end maxRegions_formula_l2326_232675


namespace tan_alpha_add_pi_over_3_l2326_232618

theorem tan_alpha_add_pi_over_3 (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5) 
  (h2 : Real.tan (β - π / 3) = 1 / 4) : 
  Real.tan (α + π / 3) = 7 / 23 := 
by
  sorry

end tan_alpha_add_pi_over_3_l2326_232618


namespace arithmetic_sequence_sum_l2326_232669

/-- Given an arithmetic sequence {a_n} and the first term a_1 = -2010, 
and given that the average of the first 2009 terms minus the average of the first 2007 terms equals 2,
prove that the sum of the first 2011 terms S_2011 equals 0. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith_seq : ∃ d, ∀ n, a n = a 1 + (n - 1) * d)
  (h_Sn : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d)
  (h_a1 : a 1 = -2010)
  (h_avg_diff : (S 2009) / 2009 - (S 2007) / 2007 = 2) :
  S 2011 = 0 := 
sorry

end arithmetic_sequence_sum_l2326_232669


namespace thor_hammer_weight_exceeds_2000_l2326_232670

/--  The Mighty Thor uses a hammer that doubles in weight each day as he trains.
      Starting on the first day with a hammer that weighs 7 pounds, prove that
      on the 10th day the hammer's weight exceeds 2000 pounds. 
-/
theorem thor_hammer_weight_exceeds_2000 :
  ∃ n : ℕ, 7 * 2^(n - 1) > 2000 ∧ n = 10 :=
by
  sorry

end thor_hammer_weight_exceeds_2000_l2326_232670


namespace Jack_minimum_cars_per_hour_l2326_232654

theorem Jack_minimum_cars_per_hour (J : ℕ) (h1 : 2 * 8 + 8 * J ≥ 40) : J ≥ 3 :=
by {
  -- The statement of the theorem directly follows
  sorry
}

end Jack_minimum_cars_per_hour_l2326_232654


namespace calculate_new_volume_l2326_232661

noncomputable def volume_of_sphere_with_increased_radius
  (initial_surface_area : ℝ) (radius_increase : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((Real.sqrt (initial_surface_area / (4 * Real.pi)) + radius_increase) ^ 3)

theorem calculate_new_volume :
  volume_of_sphere_with_increased_radius 400 (2) = 2304 * Real.pi :=
by
  sorry

end calculate_new_volume_l2326_232661


namespace range_of_m_l2326_232666

theorem range_of_m (a b m : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_eq : a * b = a + b + 3) (h_ineq : a * b ≥ m) : m ≤ 9 :=
sorry

end range_of_m_l2326_232666


namespace cos_240_eq_negative_half_l2326_232698

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l2326_232698


namespace range_of_m_l2326_232681

-- Definition of propositions p and q
def p (m : ℝ) : Prop := (2 * m - 3)^2 - 4 > 0
def q (m : ℝ) : Prop := m > 2

-- The main theorem stating the range of values for m
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ (m < 1/2 ∨ (2 < m ∧ m ≤ 5/2)) :=
by
  sorry

end range_of_m_l2326_232681


namespace x_coordinate_of_tangent_point_l2326_232625

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem x_coordinate_of_tangent_point 
  (a : ℝ) 
  (h_even : ∀ x : ℝ, f x a = f (-x) a)
  (h_slope : ∃ m : ℝ, Real.exp m - Real.exp (-m) = 3 / 2) : 
  ∃ m : ℝ, m = Real.log 2 := 
by
  sorry

end x_coordinate_of_tangent_point_l2326_232625


namespace marbles_game_winning_strategy_l2326_232609

theorem marbles_game_winning_strategy :
  ∃ k : ℕ, 1 < k ∧ k < 1024 ∧ (k = 4 ∨ k = 24 ∨ k = 40) := sorry

end marbles_game_winning_strategy_l2326_232609


namespace complex_sum_zero_l2326_232628

noncomputable def complexSum {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^(15) + ω^(18) + ω^(21) + ω^(24) + ω^(27) + ω^(30) +
  ω^(33) + ω^(36) + ω^(39) + ω^(42) + ω^(45)

theorem complex_sum_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : complexSum h1 h2 = 0 :=
by
  sorry

end complex_sum_zero_l2326_232628


namespace imaginary_part_of_complex_number_l2326_232603

theorem imaginary_part_of_complex_number :
  let z := (1 + Complex.I)^2 * (2 + Complex.I)
  Complex.im z = 4 :=
by
  sorry

end imaginary_part_of_complex_number_l2326_232603


namespace geometric_progression_fourth_term_l2326_232607

theorem geometric_progression_fourth_term :
  let a1 := 3^(1/2)
  let a2 := 3^(1/3)
  let a3 := 3^(1/6)
  let r  := a3 / a2    -- Common ratio of the geometric sequence
  let a4 := a3 * r     -- Fourth term in the geometric sequence
  a4 = 1 := by
  sorry

end geometric_progression_fourth_term_l2326_232607


namespace smallest_possible_N_l2326_232606

theorem smallest_possible_N :
  ∀ (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0),
  p + q + r + s + t = 4020 →
  (∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1342) :=
by
  intros p q r s t hp hq hr hs ht h
  use 1342
  sorry

end smallest_possible_N_l2326_232606


namespace pam_bags_l2326_232689

-- Definitions
def gerald_bag_apples : ℕ := 40
def pam_bag_apples : ℕ := 3 * gerald_bag_apples
def pam_total_apples : ℕ := 1200

-- Theorem stating that the number of Pam's bags is 10
theorem pam_bags : pam_total_apples / pam_bag_apples = 10 := by
  sorry

end pam_bags_l2326_232689


namespace g_five_eq_one_l2326_232608

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one (hx : ∀ x y : ℝ, g (x * y) = g x * g y) (h1 : g 1 ≠ 0) : g 5 = 1 :=
sorry

end g_five_eq_one_l2326_232608


namespace find_k_value_l2326_232600

theorem find_k_value (k : ℝ) (hx : ∃ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0) :
  k = -1 :=
sorry

end find_k_value_l2326_232600


namespace original_number_is_two_l2326_232623

theorem original_number_is_two (x : ℝ) (hx : 0 < x) (h : x^2 = 8 * (1 / x)) : x = 2 :=
  sorry

end original_number_is_two_l2326_232623


namespace approx_d_l2326_232663

noncomputable def close_approx_d : ℝ :=
  let d := (69.28 * (0.004)^3 - Real.log 27) / (0.03 * Real.cos (55 * Real.pi / 180))
  d

theorem approx_d : |close_approx_d + 191.297| < 0.001 :=
  by
    -- Proof goes here.
    sorry

end approx_d_l2326_232663


namespace difference_q_r_l2326_232684

theorem difference_q_r (x : ℝ) (p q r : ℝ) 
  (h1 : 7 * x - 3 * x = 3600) 
  (h2 : q = 7 * x) 
  (h3 : r = 12 * x) :
  r - q = 4500 := 
sorry

end difference_q_r_l2326_232684


namespace race_placement_l2326_232664

def finished_places (nina zoey sam liam vince : ℕ) : Prop :=
  nina = 12 ∧
  sam = nina + 1 ∧
  zoey = nina - 2 ∧
  liam = zoey - 3 ∧
  vince = liam + 2 ∧
  vince = nina - 3

theorem race_placement (nina zoey sam liam vince : ℕ) :
  finished_places nina zoey sam liam vince →
  nina = 12 →
  sam = 13 →
  zoey = 10 →
  liam = 7 →
  vince = 5 →
  (8 ≠ sam ∧ 8 ≠ nina ∧ 8 ≠ zoey ∧ 8 ≠ liam ∧ 8 ≠ jodi ∧ 8 ≠ vince) := by
  sorry

end race_placement_l2326_232664


namespace symmetric_intersection_points_eq_y_axis_l2326_232672

theorem symmetric_intersection_points_eq_y_axis (k : ℝ) :
  (∀ x y : ℝ, (y = k * x + 1) ∧ (x^2 + y^2 + k * x - y - 9 = 0) → (∃ x' : ℝ, y = k * (-x') + 1 ∧ (x'^2 + y^2 + k * x' - y - 9 = 0) ∧ x' = -x)) →
  k = 0 :=
by
  sorry

end symmetric_intersection_points_eq_y_axis_l2326_232672


namespace triangle_area_l2326_232626

theorem triangle_area (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) 
  (h₄ : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * a * b = 30 := 
by 
  rw [h₁, h₂]
  norm_num

end triangle_area_l2326_232626


namespace sandy_potatoes_l2326_232688

theorem sandy_potatoes (n_total n_nancy n_sandy : ℕ) 
  (h_total : n_total = 13) 
  (h_nancy : n_nancy = 6) 
  (h_sum : n_total = n_nancy + n_sandy) : 
  n_sandy = 7 :=
by
  sorry

end sandy_potatoes_l2326_232688


namespace complex_expression_eq_l2326_232674

-- Define the complex numbers
def c1 : ℂ := 6 - 3 * Complex.I
def c2 : ℂ := 2 - 7 * Complex.I

-- Define the scale
def scale : ℂ := 3

-- State the theorem
theorem complex_expression_eq : (c1 + scale * c2) = 12 - 24 * Complex.I :=
by
  -- This is the statement only; the proof is omitted with sorry.
  sorry

end complex_expression_eq_l2326_232674


namespace quadrilateral_area_proof_l2326_232660

noncomputable def quadrilateral_area_statement : Prop :=
  ∀ (a b : ℤ), a > b ∧ b > 0 ∧ 8 * (a - b) * (a - b) = 32 → a + b = 4

theorem quadrilateral_area_proof : quadrilateral_area_statement :=
sorry

end quadrilateral_area_proof_l2326_232660


namespace repeating_decimal_base4_sum_l2326_232614

theorem repeating_decimal_base4_sum (a b : ℕ) (hrelprime : Int.gcd a b = 1)
  (h4_rep : ((12 : ℚ) / (44 : ℚ)) = (a : ℚ) / (b : ℚ)) : a + b = 7 :=
sorry

end repeating_decimal_base4_sum_l2326_232614


namespace trivia_team_students_l2326_232668

def total_students (not_picked groups students_per_group: ℕ) :=
  not_picked + groups * students_per_group

theorem trivia_team_students (not_picked groups students_per_group: ℕ) (h_not_picked: not_picked = 10) (h_groups: groups = 8) (h_students_per_group: students_per_group = 6) :
  total_students not_picked groups students_per_group = 58 :=
by
  sorry

end trivia_team_students_l2326_232668


namespace birds_initial_count_l2326_232604

theorem birds_initial_count (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end birds_initial_count_l2326_232604


namespace equation_solution_l2326_232679

theorem equation_solution :
  ∃ x : ℝ, (3 * (x + 2) = x * (x + 2)) ↔ (x = -2 ∨ x = 3) :=
by
  sorry

end equation_solution_l2326_232679


namespace circle_diameter_l2326_232694

theorem circle_diameter (r : ℝ) (h : r = 4) : 2 * r = 8 := sorry

end circle_diameter_l2326_232694


namespace final_tree_count_l2326_232687

def current_trees : ℕ := 7
def monday_trees : ℕ := 3
def tuesday_trees : ℕ := 2
def wednesday_trees : ℕ := 5
def thursday_trees : ℕ := 1
def friday_trees : ℕ := 6
def saturday_trees : ℕ := 4
def sunday_trees : ℕ := 3

def total_trees_planted : ℕ := monday_trees + tuesday_trees + wednesday_trees + thursday_trees + friday_trees + saturday_trees + sunday_trees

theorem final_tree_count :
  current_trees + total_trees_planted = 31 :=
by
  sorry

end final_tree_count_l2326_232687


namespace dealer_pricing_l2326_232673

theorem dealer_pricing
  (cost_price : ℝ)
  (discount : ℝ := 0.10)
  (profit : ℝ := 0.20)
  (num_articles_sold : ℕ := 45)
  (num_articles_cost : ℕ := 40)
  (selling_price_per_article : ℝ := (num_articles_cost : ℝ) / num_articles_sold)
  (actual_cost_price_per_article : ℝ := selling_price_per_article / (1 + profit))
  (listed_price_per_article : ℝ := selling_price_per_article / (1 - discount)) :
  100 * ((listed_price_per_article - actual_cost_price_per_article) / actual_cost_price_per_article) = 33.33 := by
  sorry

end dealer_pricing_l2326_232673


namespace maximum_bugs_on_board_l2326_232645

-- Definition of the problem board size, bug movement directions, and non-collision rule
def board_size := 10
inductive Direction
| up | down | left | right

-- The main theorem stating the maximum number of bugs on the board
theorem maximum_bugs_on_board (bugs : List (Nat × Nat × Direction)) :
  (∀ (x y : Nat) (d : Direction) (bug : Nat × Nat × Direction), 
    bug = (x, y, d) → 
    x < board_size ∧ y < board_size ∧ 
    (∀ (c : Nat × Nat × Direction), 
      c ∈ bugs → bug ≠ c → bug.1 ≠ c.1 ∨ bug.2 ≠ c.2)) →
  List.length bugs <= 40 :=
sorry

end maximum_bugs_on_board_l2326_232645


namespace validate_expression_l2326_232656

-- Define the expression components
def a := 100
def b := 6
def c := 7
def d := 52
def e := 8
def f := 9

-- Define the expression using the given numbers and operations
def expression := (a - b) * c - d + e + f

-- The theorem statement asserting that the expression evaluates to 623
theorem validate_expression : expression = 623 := 
by
  -- Proof would go here
  sorry

end validate_expression_l2326_232656


namespace square_area_l2326_232621

theorem square_area (x : ℝ) 
  (h1 : 5 * x - 18 = 27 - 4 * x) 
  (side_length : ℝ := 5 * x - 18) : 
  side_length ^ 2 = 49 := 
by 
  sorry

end square_area_l2326_232621


namespace Rebecca_eggs_l2326_232657

/-- Rebecca has 6 marbles -/
def M : ℕ := 6

/-- Rebecca has 14 more eggs than marbles -/
def E : ℕ := M + 14

/-- Rebecca has 20 eggs -/
theorem Rebecca_eggs : E = 20 := by
  sorry

end Rebecca_eggs_l2326_232657


namespace wings_area_l2326_232643

-- Define the areas of the two cut triangles
def A1 : ℕ := 4
def A2 : ℕ := 9

-- Define the area of the wings (remaining two triangles)
def W : ℕ := 12

-- The proof goal
theorem wings_area (A1 A2 : ℕ) (W : ℕ) : A1 = 4 → A2 = 9 → W = 12 → A1 + A2 = 13 → W = 12 :=
by
  intros hA1 hA2 hW hTotal
  -- Sorry is used as a placeholder for the proof steps
  sorry

end wings_area_l2326_232643


namespace polynomial_factor_implies_a_minus_b_l2326_232630

theorem polynomial_factor_implies_a_minus_b (a b : ℝ) :
  (∀ x y : ℝ, (x + y - 2) ∣ (x^2 + a * x * y + b * y^2 - 5 * x + y + 6))
  → a - b = 1 :=
by
  intro h
  -- Proof needs to be filled in
  sorry

end polynomial_factor_implies_a_minus_b_l2326_232630


namespace aira_rubber_bands_l2326_232678

variable (S A J : ℕ)

-- Conditions
def conditions (S A J : ℕ) : Prop :=
  S = A + 5 ∧ A = J - 1 ∧ S + A + J = 18

-- Proof problem
theorem aira_rubber_bands (S A J : ℕ) (h : conditions S A J) : A = 4 :=
by
  -- introduce the conditions
  obtain ⟨h₁, h₂, h₃⟩ := h
  -- use sorry to skip the proof
  sorry

end aira_rubber_bands_l2326_232678


namespace skittles_taken_away_l2326_232624

def initial_skittles : ℕ := 25
def remaining_skittles : ℕ := 18

theorem skittles_taken_away (C_initial C_remaining : ℕ) (h1 : C_initial = 25) (h2 : C_remaining = 18) :
  (C_initial - C_remaining = 7) :=
by
  sorry

end skittles_taken_away_l2326_232624


namespace smallest_base_10_integer_exists_l2326_232683

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l2326_232683


namespace leila_savings_l2326_232636

theorem leila_savings (S : ℝ) (h : (1 / 4) * S = 20) : S = 80 :=
by
  sorry

end leila_savings_l2326_232636


namespace mosquitoes_required_l2326_232676

theorem mosquitoes_required
  (blood_loss_to_cause_death : Nat)
  (drops_per_mosquito_A : Nat)
  (drops_per_mosquito_B : Nat)
  (drops_per_mosquito_C : Nat)
  (n : Nat) :
  blood_loss_to_cause_death = 15000 →
  drops_per_mosquito_A = 20 →
  drops_per_mosquito_B = 25 →
  drops_per_mosquito_C = 30 →
  75 * n = blood_loss_to_cause_death →
  n = 200 := by
  sorry

end mosquitoes_required_l2326_232676


namespace binomial_expansion_constant_term_l2326_232647

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) :
  (∃ c : ℝ, (3 * x^2 - (1 / (2 * x^3)))^5 = c ∧ c = 135 / 2) :=
by
  sorry

end binomial_expansion_constant_term_l2326_232647


namespace correct_answer_l2326_232677

theorem correct_answer (x : ℤ) (h : (x - 11) / 5 = 31) : (x - 5) / 11 = 15 :=
by
  sorry

end correct_answer_l2326_232677


namespace marked_price_percentage_l2326_232696

variable (L P M S : ℝ)

-- Conditions
def original_list_price := 100               -- L = 100
def purchase_price := 70                     -- P = 70
def required_profit_price := 91              -- S = 91
def final_selling_price (M : ℝ) := 0.85 * M  -- S = 0.85M

-- Question: What percentage of the original list price should the marked price be?
theorem marked_price_percentage :
  L = original_list_price →
  P = purchase_price →
  S = required_profit_price →
  final_selling_price M = S →
  M = 107.06 := sorry

end marked_price_percentage_l2326_232696


namespace flowchart_structure_correct_l2326_232611

-- Definitions based on conditions
def flowchart_typically_has_one_start : Prop :=
  ∃ (start : Nat), start = 1

def flowchart_typically_has_one_or_more_ends : Prop :=
  ∃ (ends : Nat), ends ≥ 1

-- Theorem for the correct statement
theorem flowchart_structure_correct :
  (flowchart_typically_has_one_start ∧ flowchart_typically_has_one_or_more_ends) →
  (∃ (start : Nat) (ends : Nat), start = 1 ∧ ends ≥ 1) :=
by
  sorry

end flowchart_structure_correct_l2326_232611


namespace omicron_variant_diameter_in_scientific_notation_l2326_232692

/-- Converting a number to scientific notation. -/
def to_scientific_notation (d : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  d = a * 10 ^ n

theorem omicron_variant_diameter_in_scientific_notation :
  to_scientific_notation 0.00000011 1.1 (-7) :=
by
  sorry

end omicron_variant_diameter_in_scientific_notation_l2326_232692


namespace beth_comic_books_percentage_l2326_232629

/-- Definition of total books Beth owns -/
def total_books : ℕ := 120

/-- Definition of percentage novels in her collection -/
def percentage_novels : ℝ := 0.65

/-- Definition of number of graphic novels in her collection -/
def graphic_novels : ℕ := 18

/-- Calculation of the percentage of comic books she owns -/
theorem beth_comic_books_percentage (total_books : ℕ) (percentage_novels : ℝ) (graphic_novels : ℕ) : 
  (100 * ((total_books * (1 - percentage_novels) - graphic_novels) / total_books) = 20) :=
by
  let non_novel_books := total_books * (1 - percentage_novels)
  let comic_books := non_novel_books - graphic_novels
  let percentage_comic_books := 100 * (comic_books / total_books)
  have h : percentage_comic_books = 20 := sorry
  assumption

end beth_comic_books_percentage_l2326_232629


namespace knicks_equal_knocks_l2326_232612

theorem knicks_equal_knocks :
  (8 : ℝ) / (3 : ℝ) * (5 : ℝ) / (6 : ℝ) * (30 : ℝ) = 66 + 2 / 3 :=
by
  sorry

end knicks_equal_knocks_l2326_232612


namespace Maria_soap_cost_l2326_232641
-- Import the entire Mathlib library
  
theorem Maria_soap_cost (soap_last_months : ℕ) (cost_per_bar : ℝ) (months_in_year : ℕ):
  (soap_last_months = 2) -> 
  (cost_per_bar = 8.00) ->
  (months_in_year = 12) -> 
  (months_in_year / soap_last_months * cost_per_bar = 48.00) := 
by
  intros h_soap_last h_cost h_year
  sorry

end Maria_soap_cost_l2326_232641


namespace probability_A1_selected_probability_neither_A2_B2_selected_l2326_232659

-- Define the set of students
structure Student := (id : String) (gender : String)

def students : List Student :=
  [⟨"A1", "M"⟩, ⟨"A2", "M"⟩, ⟨"A3", "M"⟩, ⟨"A4", "M"⟩, ⟨"B1", "F"⟩, ⟨"B2", "F"⟩, ⟨"B3", "F"⟩]

-- Define the conditions
def males := students.filter (λ s => s.gender = "M")
def females := students.filter (λ s => s.gender = "F")

def possible_pairs : List (Student × Student) :=
  List.product males females

-- Prove the probability of selecting A1
theorem probability_A1_selected : (3 : ℚ) / (12 : ℚ) = (1 : ℚ) / (4 : ℚ) :=
by
  sorry

-- Prove the probability that neither A2 nor B2 are selected
theorem probability_neither_A2_B2_selected : (11 : ℚ) / (12 : ℚ) = (11 : ℚ) / (12 : ℚ) :=
by
  sorry

end probability_A1_selected_probability_neither_A2_B2_selected_l2326_232659


namespace c_share_l2326_232615

theorem c_share (A B C : ℕ) (h1 : A + B + C = 364) (h2 : A = B / 2) (h3 : B = C / 2) : 
  C = 208 := by
  -- Proof omitted
  sorry

end c_share_l2326_232615


namespace part_a_solution_exists_l2326_232671

theorem part_a_solution_exists : ∃ (x y : ℕ), x^2 - y^2 = 31 ∧ x = 16 ∧ y = 15 := 
by 
  sorry

end part_a_solution_exists_l2326_232671


namespace original_number_eq_9999876_l2326_232697

theorem original_number_eq_9999876 (x : ℕ) (h : x + 9876 = 10 * x + 9 + 876) : x = 999 :=
by {
  -- Simplify the equation and solve for x
  sorry
}

end original_number_eq_9999876_l2326_232697


namespace Sammy_has_8_bottle_caps_l2326_232686

def Billie_caps : Nat := 2
def Janine_caps (B : Nat) : Nat := 3 * B
def Sammy_caps (J : Nat) : Nat := J + 2

theorem Sammy_has_8_bottle_caps : 
  Sammy_caps (Janine_caps Billie_caps) = 8 := 
by
  sorry

end Sammy_has_8_bottle_caps_l2326_232686


namespace f_at_1_l2326_232642

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom fg_eq : ∀ x : ℝ, f x + g x = x^3 - x^2 + 1

theorem f_at_1 : f 1 = 1 := by
  sorry

end f_at_1_l2326_232642


namespace women_lawyers_percentage_l2326_232665

-- Define the conditions of the problem
variable {T : ℝ} (h1 : 0.80 * T = 0.80 * T)                          -- Placeholder for group size, not necessarily used directly
variable (h2 : 0.32 = 0.80 * L)                                       -- Given condition of the problem: probability of selecting a woman lawyer

-- Define the theorem to be proven
theorem women_lawyers_percentage (h2 : 0.32 = 0.80 * L) : L = 0.4 :=
by
  sorry

end women_lawyers_percentage_l2326_232665


namespace rotation_of_unit_circle_l2326_232693

open Real

noncomputable def rotated_coordinates (θ : ℝ) : ℝ × ℝ :=
  ( -sin θ, cos θ )

theorem rotation_of_unit_circle (θ : ℝ) (k : ℤ) (h : θ ≠ k * π + π / 2) :
  let A := (cos θ, sin θ)
  let O := (0, 0)
  let B := rotated_coordinates (θ)
  B = (-sin θ, cos θ) :=
sorry

end rotation_of_unit_circle_l2326_232693


namespace vector_add_sub_eq_l2326_232634

-- Define the vectors involved in the problem
def v1 : ℝ×ℝ×ℝ := (4, -3, 7)
def v2 : ℝ×ℝ×ℝ := (-1, 5, 2)
def v3 : ℝ×ℝ×ℝ := (2, -4, 9)

-- Define the result of the given vector operations
def result : ℝ×ℝ×ℝ := (1, 6, 0)

-- State the theorem we want to prove
theorem vector_add_sub_eq :
  v1 + v2 - v3 = result :=
sorry

end vector_add_sub_eq_l2326_232634


namespace trays_needed_to_refill_l2326_232652

theorem trays_needed_to_refill (initial_ice_cubes used_ice_cubes tray_capacity : ℕ)
  (h_initial: initial_ice_cubes = 130)
  (h_used: used_ice_cubes = (initial_ice_cubes * 8 / 10))
  (h_tray_capacity: tray_capacity = 14) :
  (initial_ice_cubes + tray_capacity - 1) / tray_capacity = 10 :=
by
  sorry

end trays_needed_to_refill_l2326_232652


namespace apples_total_l2326_232658

theorem apples_total (apples_per_person : ℝ) (number_of_people : ℝ) (h_apples : apples_per_person = 15.0) (h_people : number_of_people = 3.0) : 
  apples_per_person * number_of_people = 45.0 := by
  sorry

end apples_total_l2326_232658


namespace discount_percentage_l2326_232648

theorem discount_percentage (P D : ℝ) 
  (h1 : P > 0)
  (h2 : D = (1 - 0.28000000000000004 / 0.60)) :
  D = 0.5333333333333333 :=
by
  sorry

end discount_percentage_l2326_232648


namespace find_b_l2326_232638

theorem find_b (h1 : 2.236 = 1 + (b - 1) * 0.618) 
               (h2 : 2.236 = b - (b - 1) * 0.618) : 
               b = 3 ∨ b = 4.236 := 
by
  sorry

end find_b_l2326_232638


namespace necessary_and_sufficient_conditions_l2326_232602

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - x^2

-- Define the domain of x
def dom_x (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

theorem necessary_and_sufficient_conditions {a : ℝ} (ha : a > 0) :
  (∀ x : ℝ, dom_x x → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end necessary_and_sufficient_conditions_l2326_232602


namespace value_of_a_for_perfect_square_trinomial_l2326_232690

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 2 * a * x + 9 = (x + b)^2) → (a = 3 ∨ a = -3) :=
by
  sorry

end value_of_a_for_perfect_square_trinomial_l2326_232690


namespace greatest_perimeter_among_four_pieces_l2326_232691

/--
Given an isosceles triangle with a base of 12 inches and a height of 15 inches,
the greatest perimeter among the four pieces of equal area obtained by cutting
the triangle into four smaller triangles is approximately 33.43 inches.
-/
theorem greatest_perimeter_among_four_pieces :
  let base : ℝ := 12
  let height : ℝ := 15
  ∃ (P : ℝ), P = (3 + Real.sqrt (225 + 4) + Real.sqrt (225 + 9)) ∧ abs (P - 33.43) < 0.01 := sorry

end greatest_perimeter_among_four_pieces_l2326_232691


namespace largest_n_divisible_103_l2326_232601

theorem largest_n_divisible_103 (n : ℕ) (h1 : n < 103) (h2 : 103 ∣ (n^3 - 1)) : n = 52 :=
sorry

end largest_n_divisible_103_l2326_232601


namespace fill_pool_time_l2326_232627

theorem fill_pool_time 
  (pool_volume : ℕ) (num_hoses : ℕ) (flow_rate_per_hose : ℕ)
  (H_pool_volume : pool_volume = 36000)
  (H_num_hoses : num_hoses = 6)
  (H_flow_rate_per_hose : flow_rate_per_hose = 3) :
  (pool_volume : ℚ) / (num_hoses * flow_rate_per_hose * 60) = 100 / 3 :=
by sorry

end fill_pool_time_l2326_232627


namespace JacksonsGrade_l2326_232651

theorem JacksonsGrade : 
  let hours_playing_video_games := 12
  let hours_studying := (1 / 3) * hours_playing_video_games
  let hours_kindness := (1 / 4) * hours_playing_video_games
  let grade_initial := 0
  let grade_per_hour_studying := 20
  let grade_per_hour_kindness := 40
  let grade_from_studying := grade_per_hour_studying * hours_studying
  let grade_from_kindness := grade_per_hour_kindness * hours_kindness
  let total_grade := grade_initial + grade_from_studying + grade_from_kindness
  total_grade = 200 :=
by
  -- Proof goes here
  sorry

end JacksonsGrade_l2326_232651


namespace garden_perimeter_ratio_l2326_232667

theorem garden_perimeter_ratio (side_length : ℕ) (tripled_side_length : ℕ) (original_perimeter : ℕ) (new_perimeter : ℕ) (ratio : ℚ) :
  side_length = 50 →
  tripled_side_length = 3 * side_length →
  original_perimeter = 4 * side_length →
  new_perimeter = 4 * tripled_side_length →
  ratio = original_perimeter / new_perimeter →
  ratio = 1 / 3 :=
by
  sorry

end garden_perimeter_ratio_l2326_232667


namespace smallest_base10_integer_l2326_232622

theorem smallest_base10_integer (X Y : ℕ) (hX : X < 6) (hY : Y < 8) (h : 7 * X = 9 * Y) :
  63 = 7 * X ∧ 63 = 9 * Y :=
by
  -- Proof steps would go here
  sorry

end smallest_base10_integer_l2326_232622


namespace a_greater_than_b_c_less_than_a_l2326_232619

-- Condition 1: Definition of box dimensions
def Box := (Nat × Nat × Nat)

-- Condition 2: Dimension comparisons
def le_box (a b : Box) : Prop :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  (a1 ≤ b1 ∨ a1 ≤ b2 ∨ a1 ≤ b3) ∧ (a2 ≤ b1 ∨ a2 ≤ b2 ∨ a2 ≤ b3) ∧ (a3 ≤ b1 ∨ a3 ≤ b2 ∨ a3 ≤ b3)

def lt_box (a b : Box) : Prop := le_box a b ∧ ¬(a = b)

-- Condition 3: Box dimensions
def A : Box := (6, 5, 3)
def B : Box := (5, 4, 1)
def C : Box := (3, 2, 2)

-- Equivalent Problem 1: Prove A > B
theorem a_greater_than_b : lt_box B A :=
by
  -- theorem proof here
  sorry

-- Equivalent Problem 2: Prove C < A
theorem c_less_than_a : lt_box C A :=
by
  -- theorem proof here
  sorry

end a_greater_than_b_c_less_than_a_l2326_232619


namespace incorrect_inequality_l2326_232635

theorem incorrect_inequality (m n : ℝ) (a : ℝ) (hmn : m > n) (hm1 : m > 1) (hn1 : n > 1) (ha0 : 0 < a) (ha1 : a < 1) : 
  ¬ (a^m > a^n) :=
sorry

end incorrect_inequality_l2326_232635


namespace calculate_entire_surface_area_l2326_232655

-- Define the problem parameters
def cube_edge_length : ℝ := 4
def hole_side_length : ℝ := 2

-- Define the function to compute the total surface area
noncomputable def entire_surface_area : ℝ :=
  let original_surface_area := 6 * (cube_edge_length ^ 2)
  let hole_area := 6 * (hole_side_length ^ 2)
  let exposed_internal_area := 6 * 4 * (hole_side_length ^ 2)
  original_surface_area - hole_area + exposed_internal_area

-- Statement of the problem to prove the given conditions
theorem calculate_entire_surface_area : entire_surface_area = 168 := by
  sorry

end calculate_entire_surface_area_l2326_232655


namespace barber_loss_is_25_l2326_232610

-- Definition of conditions
structure BarberScenario where
  haircut_cost : ℕ
  counterfeit_bill : ℕ
  real_change : ℕ
  change_given : ℕ
  real_bill_given : ℕ

def barberScenario_example : BarberScenario :=
  { haircut_cost := 15,
    counterfeit_bill := 20,
    real_change := 20,
    change_given := 5,
    real_bill_given := 20 }

-- Lean 4 problem statement
theorem barber_loss_is_25 (b : BarberScenario) : 
  b.haircut_cost = 15 ∧
  b.counterfeit_bill = 20 ∧
  b.real_change = 20 ∧
  b.change_given = 5 ∧
  b.real_bill_given = 20 → (15 + 5 + 20 - 20 + 5 = 25) :=
by
  intro h
  cases' h with h1 h23
  sorry

end barber_loss_is_25_l2326_232610


namespace andy_wrong_questions_l2326_232640

/-- Andy, Beth, Charlie, and Daniel take a test. Andy and Beth together get the same number of 
    questions wrong as Charlie and Daniel together. Andy and Daniel together get four more 
    questions wrong than Beth and Charlie do together. Charlie gets five questions wrong. 
    Prove that Andy gets seven questions wrong. -/
theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 4) (h3 : c = 5) :
  a = 7 :=
by
  sorry

end andy_wrong_questions_l2326_232640


namespace find_z_proportional_l2326_232644

theorem find_z_proportional (k : ℝ) (y x z : ℝ) 
  (h₁ : y = 8) (h₂ : x = 2) (h₃ : z = 4) (relationship : y = (k * x^2) / z)
  (y' x' z' : ℝ) (h₄ : y' = 72) (h₅ : x' = 4) : 
  z' = 16 / 9 := by
  sorry

end find_z_proportional_l2326_232644


namespace sum_of_five_consecutive_integers_l2326_232699

theorem sum_of_five_consecutive_integers (n : ℤ) :
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 5 * n :=
by
  sorry

end sum_of_five_consecutive_integers_l2326_232699


namespace tan_10pi_minus_theta_l2326_232613

open Real

theorem tan_10pi_minus_theta (θ : ℝ) (h1 : π < θ) (h2 : θ < 2 * π) (h3 : cos (θ - 9 * π) = -3 / 5) : 
  tan (10 * π - θ) = -4 / 3 := 
sorry

end tan_10pi_minus_theta_l2326_232613


namespace age_solution_l2326_232680

noncomputable def age_problem : Prop :=
  ∃ (m s x : ℕ),
  (m - 3 = 2 * (s - 3)) ∧
  (m - 5 = 3 * (s - 5)) ∧
  (m + x) * 2 = 3 * (s + x) ∧
  x = 1

theorem age_solution : age_problem :=
  by
    sorry

end age_solution_l2326_232680


namespace ratio_length_to_breadth_l2326_232616

-- Definitions of the given conditions
def length_landscape : ℕ := 120
def area_playground : ℕ := 1200
def ratio_playground_to_landscape : ℕ := 3

-- Property that the area of the playground is 1/3 of the area of the landscape
def total_area_landscape (area_playground : ℕ) (ratio_playground_to_landscape : ℕ) : ℕ :=
  area_playground * ratio_playground_to_landscape

-- Calculation that breadth of the landscape
def breadth_landscape (length_landscape total_area_landscape : ℕ) : ℕ :=
  total_area_landscape / length_landscape

-- The proof statement for the ratio of length to breadth
theorem ratio_length_to_breadth (length_landscape area_playground : ℕ) (ratio_playground_to_landscape : ℕ)
  (h1 : length_landscape = 120)
  (h2 : area_playground = 1200)
  (h3 : ratio_playground_to_landscape = 3)
  (h4 : total_area_landscape area_playground ratio_playground_to_landscape = 3600)
  (h5 : breadth_landscape length_landscape (total_area_landscape area_playground ratio_playground_to_landscape) = 30) :
  length_landscape / breadth_landscape length_landscape (total_area_landscape area_playground ratio_playground_to_landscape) = 4 :=
by
  sorry


end ratio_length_to_breadth_l2326_232616


namespace average_GPA_school_l2326_232662

theorem average_GPA_school (GPA6 GPA7 GPA8 : ℕ) (h1 : GPA6 = 93) (h2 : GPA7 = GPA6 + 2) (h3 : GPA8 = 91) : ((GPA6 + GPA7 + GPA8) / 3) = 93 :=
by
  sorry

end average_GPA_school_l2326_232662


namespace average_next_seven_l2326_232649

variable (c : ℕ) (h : c > 0)

theorem average_next_seven (d : ℕ) (h1 : d = (2 * c + 3)) 
  : (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = 2 * c + 6 := by
  sorry

end average_next_seven_l2326_232649


namespace original_agreed_amount_l2326_232632

theorem original_agreed_amount (months: ℕ) (cash: ℚ) (uniform_price: ℚ) (received_total: ℚ) (full_year: ℚ) :
  months = 9 →
  cash = 300 →
  uniform_price = 300 →
  received_total = 600 →
  full_year = (12: ℚ) →
  ((months / full_year) * (cash + uniform_price) = received_total) →
  cash + uniform_price = 800 := 
by
  intros h_months h_cash h_uniform h_received h_year h_proportion
  sorry

end original_agreed_amount_l2326_232632


namespace min_distance_is_18_l2326_232682

noncomputable def minimize_distance (a b c d : ℝ) : ℝ := (a - c) ^ 2 + (b - d) ^ 2

theorem min_distance_is_18 (a b c d : ℝ) (h1 : b = a - 2 * Real.exp a) (h2 : c + d = 4) :
  minimize_distance a b c d = 18 :=
sorry

end min_distance_is_18_l2326_232682


namespace triangle_inequality_condition_l2326_232695

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end triangle_inequality_condition_l2326_232695


namespace range_of_a_l2326_232637

theorem range_of_a :
  (∀ x : ℝ, abs (x - a) < 1 ↔ (1 / 2 < x ∧ x < 3 / 2)) → (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by sorry

end range_of_a_l2326_232637
