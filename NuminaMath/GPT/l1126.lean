import Mathlib

namespace total_seats_theater_l1126_112685

theorem total_seats_theater (a1 an d n Sn : ℕ) 
    (h1 : a1 = 12) 
    (h2 : d = 2) 
    (h3 : an = 48) 
    (h4 : an = a1 + (n - 1) * d) 
    (h5 : Sn = n * (a1 + an) / 2) : 
    Sn = 570 := 
sorry

end total_seats_theater_l1126_112685


namespace solution_set_inequality_l1126_112624

theorem solution_set_inequality (x : ℝ) (h1 : x < -3) (h2 : x < 2) : x < -3 :=
by
  exact h1

end solution_set_inequality_l1126_112624


namespace smallest_multiple_14_15_16_l1126_112604

theorem smallest_multiple_14_15_16 : 
  Nat.lcm (Nat.lcm 14 15) 16 = 1680 := by
  sorry

end smallest_multiple_14_15_16_l1126_112604


namespace find_y_l1126_112679

theorem find_y (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := 
by sorry

end find_y_l1126_112679


namespace nonagon_diagonals_l1126_112646

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nonagon_diagonals : number_of_diagonals 9 = 27 := 
by
  sorry

end nonagon_diagonals_l1126_112646


namespace value_of_m_l1126_112659

theorem value_of_m (m : ℤ) (h : m + 1 = - (-2)) : m = 1 :=
sorry

end value_of_m_l1126_112659


namespace distance_between_lines_is_sqrt2_l1126_112613

noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  |c1 - c2| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_is_sqrt2 :
  distance_between_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 := 
by 
  sorry

end distance_between_lines_is_sqrt2_l1126_112613


namespace probability_blue_face_up_l1126_112606

-- Definitions of the conditions
def dodecahedron_faces : ℕ := 12
def blue_faces : ℕ := 10
def red_faces : ℕ := 2

-- Expected probability
def probability_blue_face : ℚ := 5 / 6

-- Theorem to prove the probability of rolling a blue face on a dodecahedron
theorem probability_blue_face_up (total_faces blue_count red_count : ℕ)
    (h1 : total_faces = dodecahedron_faces)
    (h2 : blue_count = blue_faces)
    (h3 : red_count = red_faces) :
  blue_count / total_faces = probability_blue_face :=
by sorry

end probability_blue_face_up_l1126_112606


namespace max_value_of_f_l1126_112628

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_of_f : ∃ M : ℝ, M = 1 / 3 ∧ ∀ x : ℝ, f x ≤ M :=
by
  sorry

end max_value_of_f_l1126_112628


namespace area_of_quadrilateral_l1126_112683

theorem area_of_quadrilateral (d h1 h2 : ℝ) (h1_pos : h1 = 9) (h2_pos : h2 = 6) (d_pos : d = 30) : 
  let area1 := (1/2 : ℝ) * d * h1
  let area2 := (1/2 : ℝ) * d * h2
  (area1 + area2) = 225 :=
by
  sorry

end area_of_quadrilateral_l1126_112683


namespace tailor_time_calculation_l1126_112625

-- Define the basic quantities and their relationships
def time_ratio_shirt : ℕ := 1
def time_ratio_pants : ℕ := 2
def time_ratio_jacket : ℕ := 3

-- Given conditions
def shirts_made := 2
def pants_made := 3
def jackets_made := 4
def total_time_initial : ℝ := 10

-- Unknown time per shirt
noncomputable def time_per_shirt := total_time_initial / (shirts_made * time_ratio_shirt 
  + pants_made * time_ratio_pants 
  + jackets_made * time_ratio_jacket)

-- Future quantities
def future_shirts := 14
def future_pants := 10
def future_jackets := 2

-- Calculate the future total time required
noncomputable def future_time_required := (future_shirts * time_ratio_shirt 
  + future_pants * time_ratio_pants 
  + future_jackets * time_ratio_jacket) * time_per_shirt

-- State the theorem to prove
theorem tailor_time_calculation : future_time_required = 20 := by
  sorry

end tailor_time_calculation_l1126_112625


namespace total_bottle_caps_in_collection_l1126_112669

-- Statements of given conditions
def small_box_caps : ℕ := 35
def large_box_caps : ℕ := 75
def num_small_boxes : ℕ := 7
def num_large_boxes : ℕ := 3
def individual_caps : ℕ := 23

-- Theorem statement that needs to be proved
theorem total_bottle_caps_in_collection :
  small_box_caps * num_small_boxes + large_box_caps * num_large_boxes + individual_caps = 493 :=
by sorry

end total_bottle_caps_in_collection_l1126_112669


namespace express_y_in_terms_of_x_l1126_112631

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + 3 * y = 4) : y = (4 - 2 * x) / 3 := 
by
  sorry

end express_y_in_terms_of_x_l1126_112631


namespace initial_percentage_proof_l1126_112670

-- Defining the initial percentage of water filled in the container
def initial_percentage (capacity add amount_filled : ℕ) : ℕ :=
  (amount_filled * 100) / capacity

-- The problem constraints
theorem initial_percentage_proof : initial_percentage 120 48 (3 * 120 / 4 - 48) = 35 := by
  -- We need to show that the initial percentage is 35%
  sorry

end initial_percentage_proof_l1126_112670


namespace contribution_is_6_l1126_112621

-- Defining the earnings of each friend
def earning_1 : ℕ := 18
def earning_2 : ℕ := 22
def earning_3 : ℕ := 30
def earning_4 : ℕ := 35
def earning_5 : ℕ := 45

-- Defining the modified contribution for the highest earner
def modified_earning_5 : ℕ := 40

-- Calculate the total adjusted earnings
def total_earnings : ℕ := earning_1 + earning_2 + earning_3 + earning_4 + modified_earning_5

-- Calculate the equal share each friend should receive
def equal_share : ℕ := total_earnings / 5

-- Calculate the contribution needed from the friend who earned $35 to match the equal share
def contribution_from_earning_4 : ℕ := earning_4 - equal_share

-- Stating the proof problem
theorem contribution_is_6 : contribution_from_earning_4 = 6 := by
  sorry

end contribution_is_6_l1126_112621


namespace break_even_production_volume_l1126_112611

theorem break_even_production_volume :
  ∃ Q : ℝ, 300 = 100 + 100000 / Q ∧ Q = 500 :=
by
  use 500
  sorry

end break_even_production_volume_l1126_112611


namespace common_difference_l1126_112632

def arith_seq_common_difference (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference {a : ℕ → ℤ} (h₁ : a 5 = 3) (h₂ : a 6 = -2) : arith_seq_common_difference a (-5) :=
by
  intros n
  cases n with
  | zero => sorry -- base case: a 1 = a 0 + (-5), requires additional initial condition
  | succ n' => sorry -- inductive step

end common_difference_l1126_112632


namespace percent_within_one_standard_deviation_l1126_112691

variable (m d : ℝ)
variable (distribution : ℝ → ℝ)
variable (symmetric_about_mean : ∀ x, distribution (m + x) = distribution (m - x))
variable (percent_less_than_m_plus_d : distribution (m + d) = 0.84)

theorem percent_within_one_standard_deviation :
  distribution (m + d) - distribution (m - d) = 0.68 :=
sorry

end percent_within_one_standard_deviation_l1126_112691


namespace Murtha_pebbles_l1126_112676

-- Definition of the geometric series sum formula
noncomputable def sum_geometric_series (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Constants for the problem
def a : ℕ := 1
def r : ℕ := 2
def n : ℕ := 10

-- The theorem to be proven
theorem Murtha_pebbles : sum_geometric_series a r n = 1023 :=
by
  -- Our condition setup implies the formula
  sorry

end Murtha_pebbles_l1126_112676


namespace area_of_circle_diameter_7_5_l1126_112602

theorem area_of_circle_diameter_7_5 :
  ∃ (A : ℝ), (A = 14.0625 * Real.pi) ↔ (∃ (d : ℝ), d = 7.5 ∧ A = Real.pi * (d / 2) ^ 2) :=
by
  sorry

end area_of_circle_diameter_7_5_l1126_112602


namespace smallest_positive_number_is_option_B_l1126_112639

theorem smallest_positive_number_is_option_B :
  let A := 8 - 2 * Real.sqrt 17
  let B := 2 * Real.sqrt 17 - 8
  let C := 25 - 7 * Real.sqrt 5
  let D := 40 - 9 * Real.sqrt 2
  let E := 9 * Real.sqrt 2 - 40
  0 < B ∧ (A ≤ 0 ∨ B < A) ∧ (C ≤ 0 ∨ B < C) ∧ (D ≤ 0 ∨ B < D) ∧ (E ≤ 0 ∨ B < E) :=
by
  sorry

end smallest_positive_number_is_option_B_l1126_112639


namespace count_total_shells_l1126_112698

theorem count_total_shells 
  (purple_shells : ℕ := 13)
  (pink_shells : ℕ := 8)
  (yellow_shells : ℕ := 18)
  (blue_shells : ℕ := 12)
  (orange_shells : ℕ := 14) :
  purple_shells + pink_shells + yellow_shells + blue_shells + orange_shells = 65 :=
by
  -- Calculation
  sorry

end count_total_shells_l1126_112698


namespace claire_photos_eq_10_l1126_112607

variable (C L R : Nat)

theorem claire_photos_eq_10
  (h1: L = 3 * C)
  (h2: R = C + 20)
  (h3: L = R)
  : C = 10 := by
  sorry

end claire_photos_eq_10_l1126_112607


namespace clothing_probability_l1126_112600

/-- I have a drawer with 6 shirts, 8 pairs of shorts, 7 pairs of socks, and 3 jackets in it.
    If I reach in and randomly remove four articles of clothing, what is the probability that 
    I get one shirt, one pair of shorts, one pair of socks, and one jacket? -/
theorem clothing_probability :
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  (favorable_combinations : ℚ) / total_combinations = 144 / 1815 :=
by
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  suffices (favorable_combinations : ℚ) / total_combinations = 144 / 1815
  by
    sorry
  sorry

end clothing_probability_l1126_112600


namespace max_b_lattice_free_line_l1126_112643

theorem max_b_lattice_free_line : 
  ∃ b : ℚ, (∀ (m : ℚ), (1 / 3) < m ∧ m < b → 
  ∀ x : ℤ, 0 < x ∧ x ≤ 150 → ¬ (∃ y : ℤ, y = m * x + 4)) ∧ 
  b = 50 / 147 :=
sorry

end max_b_lattice_free_line_l1126_112643


namespace isosceles_triangle_base_length_l1126_112644

-- Define the isosceles triangle problem
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  perimeter : ℝ
  isIsosceles : (side1 = side2 ∨ side1 = base ∨ side2 = base)
  sideLengthCondition : (side1 = 3 ∨ side2 = 3 ∨ base = 3)
  perimeterCondition : side1 + side2 + base = 13
  triangleInequality1 : side1 + side2 > base
  triangleInequality2 : side1 + base > side2
  triangleInequality3 : side2 + base > side1

-- Define the theorem to prove
theorem isosceles_triangle_base_length (T : IsoscelesTriangle) :
  T.base = 3 := by
  sorry

end isosceles_triangle_base_length_l1126_112644


namespace bus_speed_excluding_stoppages_l1126_112626

theorem bus_speed_excluding_stoppages (v : ℕ): (45 : ℝ) = (5 / 6 * v) → v = 54 :=
by
  sorry

end bus_speed_excluding_stoppages_l1126_112626


namespace factor_quadratic_l1126_112674

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 16 * x^2 - 56 * x + 49

-- The goal is to prove that the quadratic expression is equal to (4x - 7)^2
theorem factor_quadratic (x : ℝ) : quadratic_expr x = (4 * x - 7)^2 :=
by
  sorry

end factor_quadratic_l1126_112674


namespace problem1_problem2_l1126_112678
noncomputable section

-- Problem (1) Lean Statement
theorem problem1 : |-4| - (2021 - Real.pi)^0 + (Real.cos (Real.pi / 3))⁻¹ - (-Real.sqrt 3)^2 = 2 :=
by 
  sorry

-- Problem (2) Lean Statement
theorem problem2 (a : ℝ) (h : a ≠ 2 ∧ a ≠ -2) : 
  (1 + 4 / (a^2 - 4)) / (a / (a + 2)) = a / (a - 2) := 
by 
  sorry

end problem1_problem2_l1126_112678


namespace tetrahedron_pairs_l1126_112663

theorem tetrahedron_pairs (tetra_edges : ℕ) (h_tetra : tetra_edges = 6) :
  ∀ (num_pairs : ℕ), num_pairs = (tetra_edges * (tetra_edges - 1)) / 2 → num_pairs = 15 :=
by
  sorry

end tetrahedron_pairs_l1126_112663


namespace exponentiation_rule_l1126_112692

theorem exponentiation_rule (a m : ℕ) (h : (a^2)^m = a^6) : m = 3 :=
by
  sorry

end exponentiation_rule_l1126_112692


namespace problem_solution_l1126_112665

theorem problem_solution (x : ℝ) (h : x * Real.log 4 / Real.log 3 = 1) : 
  2^x + 4^(-x) = 1 / 3 + Real.sqrt 3 :=
by 
  sorry

end problem_solution_l1126_112665


namespace part_a_l1126_112641

theorem part_a (b c: ℤ) : ∃ (n : ℕ) (a : ℕ → ℤ), 
  (a 0 = b) ∧ (a n = c) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → |a i - a (i - 1)| = i^2) :=
sorry

end part_a_l1126_112641


namespace class_a_winning_probability_best_of_three_l1126_112634

theorem class_a_winning_probability_best_of_three :
  let p := (3 : ℚ) / 5
  let win_first_two := p * p
  let win_first_and_third := p * ((1 - p) * p)
  let win_last_two := (1 - p) * (p * p)
  p * p + p * ((1 - p) * p) + (1 - p) * (p * p) = 81 / 125 :=
by
  sorry

end class_a_winning_probability_best_of_three_l1126_112634


namespace pupils_like_both_l1126_112655

theorem pupils_like_both (total_pupils : ℕ) (likes_pizza : ℕ) (likes_burgers : ℕ)
  (total := 200) (P := 125) (B := 115) :
  (P + B - total_pupils) = 40 :=
by
  sorry

end pupils_like_both_l1126_112655


namespace sqrt_cubic_sqrt_decimal_l1126_112614

theorem sqrt_cubic_sqrt_decimal : 
  (Real.sqrt (0.0036 : ℝ))^(1/3) = 0.3912 :=
sorry

end sqrt_cubic_sqrt_decimal_l1126_112614


namespace domain_of_function_l1126_112656

theorem domain_of_function :
  { x : ℝ | x + 2 ≥ 0 ∧ x - 1 ≠ 0 } = { x : ℝ | x ≥ -2 ∧ x ≠ 1 } :=
by
  sorry

end domain_of_function_l1126_112656


namespace number_of_solutions_l1126_112649

theorem number_of_solutions : 
  ∃ n : ℕ, n = 5 ∧ (∃ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ 4 * x + 5 * y = 98) :=
sorry

end number_of_solutions_l1126_112649


namespace math_problem_l1126_112697

/-- The proof problem: Calculate -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11. -/
theorem math_problem : -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11 :=
by
  sorry

end math_problem_l1126_112697


namespace liams_numbers_l1126_112635

theorem liams_numbers (x y : ℤ) 
  (h1 : 3 * x + 2 * y = 75)
  (h2 : x = 15)
  (h3 : ∃ k : ℕ, x * y = 5 * k) : 
  y = 15 := 
by
  sorry

end liams_numbers_l1126_112635


namespace new_rectangle_area_l1126_112672

theorem new_rectangle_area (a b : ℝ) : 
  let base := b + 2 * a
  let height := b - a
  let area := base * height
  area = b^2 + b * a - 2 * a^2 :=
by
  let base := b + 2 * a
  let height := b - a
  let area := base * height
  show area = b^2 + b * a - 2 * a^2
  sorry

end new_rectangle_area_l1126_112672


namespace inequality_proof_l1126_112688

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l1126_112688


namespace train_A_reaches_destination_in_6_hours_l1126_112673

noncomputable def t : ℕ := 
  let tA := 110
  let tB := 165
  let tB_time := 4
  (tB * tB_time) / tA

theorem train_A_reaches_destination_in_6_hours :
  t = 6 := by
  sorry

end train_A_reaches_destination_in_6_hours_l1126_112673


namespace monotonically_increasing_function_l1126_112645

open Function

theorem monotonically_increasing_function (f : ℝ → ℝ) (h_mono : ∀ x y, x < y → f x < f y) (t : ℝ) (h_t : t ≠ 0) :
    f (t^2 + t) > f t :=
by
  sorry

end monotonically_increasing_function_l1126_112645


namespace expected_BBR_sequences_l1126_112618

theorem expected_BBR_sequences :
  let total_cards := 52
  let black_cards := 26
  let red_cards := 26
  let probability_of_next_black := (25 / 51)
  let probability_of_third_red := (26 / 50)
  let probability_of_BBR := probability_of_next_black * probability_of_third_red
  let possible_start_positions := 26
  let expected_BBR := possible_start_positions * probability_of_BBR
  expected_BBR = (338 / 51) :=
by
  sorry

end expected_BBR_sequences_l1126_112618


namespace cows_with_no_spots_l1126_112633

-- Definitions of conditions
def total_cows : Nat := 140
def cows_with_red_spot : Nat := (40 * total_cows) / 100
def cows_without_red_spot : Nat := total_cows - cows_with_red_spot
def cows_with_blue_spot : Nat := (25 * cows_without_red_spot) / 100

-- Theorem statement asserting the number of cows with no spots
theorem cows_with_no_spots : (total_cows - cows_with_red_spot - cows_with_blue_spot) = 63 := by
  -- Proof would go here
  sorry

end cows_with_no_spots_l1126_112633


namespace shifted_parabola_sum_l1126_112650

theorem shifted_parabola_sum (a b c : ℝ) :
  (∃ (a b c : ℝ), ∀ x : ℝ, 3 * x^2 + 2 * x - 5 = 3 * (x - 6)^2 + 2 * (x - 6) - 5 → y = a * x^2 + b * x + c) → a + b + c = 60 :=
sorry

end shifted_parabola_sum_l1126_112650


namespace infinite_equal_pairs_l1126_112667

theorem infinite_equal_pairs
  (a : ℤ → ℝ)
  (h : ∀ k : ℤ, a k = 1/4 * (a (k - 1) + a (k + 1)))
  (k p : ℤ) (hne : k ≠ p) (heq : a k = a p) :
  ∃ infinite_pairs : ℕ → (ℤ × ℤ), 
  (∀ n : ℕ, (infinite_pairs n).1 ≠ (infinite_pairs n).2) ∧
  (∀ n : ℕ, a (infinite_pairs n).1 = a (infinite_pairs n).2) :=
sorry

end infinite_equal_pairs_l1126_112667


namespace inscribed_square_area_l1126_112622

theorem inscribed_square_area :
  (∃ (t : ℝ), (2*t)^2 = 4 * (t^2) ∧ ∀ (x y : ℝ), (x = t ∧ y = t ∨ x = -t ∧ y = t ∨ x = t ∧ y = -t ∨ x = -t ∧ y = -t) 
  → (x^2 / 4 + y^2 / 8 = 1) ) 
  → (∃ (a : ℝ), a = 32 / 3) := 
by
  sorry

end inscribed_square_area_l1126_112622


namespace odd_function_ln_negx_l1126_112687

theorem odd_function_ln_negx (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_positive : ∀ x, x > 0 → f x = Real.log x) :
  ∀ x, x < 0 → f x = -Real.log (-x) :=
by 
  intros x hx_neg
  have hx_pos : -x > 0 := by linarith
  rw [← h_positive (-x) hx_pos, h_odd x]
  sorry

end odd_function_ln_negx_l1126_112687


namespace intersection_of_A_and_B_l1126_112610

def A : Set ℤ := {-3, -1, 2, 6}
def B : Set ℤ := {x | x > 0}

theorem intersection_of_A_and_B : A ∩ B = {2, 6} :=
by
  sorry

end intersection_of_A_and_B_l1126_112610


namespace calculate_product_l1126_112660

theorem calculate_product : 3^6 * 4^3 = 46656 := by
  sorry

end calculate_product_l1126_112660


namespace evaluate_expression_at_2_l1126_112601

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := 2 * x - 3

theorem evaluate_expression_at_2 : f (g 2) + g (f 2) = 331 / 20 :=
by
  sorry

end evaluate_expression_at_2_l1126_112601


namespace friend_c_spent_26_l1126_112603

theorem friend_c_spent_26 :
  let you_spent := 12
  let friend_a_spent := you_spent + 4
  let friend_b_spent := friend_a_spent - 3
  let friend_c_spent := friend_b_spent * 2
  friend_c_spent = 26 :=
by
  sorry

end friend_c_spent_26_l1126_112603


namespace farmer_tomatoes_l1126_112675

theorem farmer_tomatoes (t p l : ℕ) (H1 : t = 97) (H2 : p = 83) : l = t - p → l = 14 :=
by {
  sorry
}

end farmer_tomatoes_l1126_112675


namespace jake_first_test_score_l1126_112693

theorem jake_first_test_score 
  (avg_score : ℕ)
  (n_tests : ℕ)
  (second_test_extra : ℕ)
  (third_test_score : ℕ)
  (x : ℕ) : 
  avg_score = 75 → 
  n_tests = 4 → 
  second_test_extra = 10 → 
  third_test_score = 65 →
  (x + (x + second_test_extra) + third_test_score + third_test_score) / n_tests = avg_score →
  x = 80 := by
  intros h1 h2 h3 h4 h5
  sorry

end jake_first_test_score_l1126_112693


namespace average_marks_math_chem_l1126_112616

-- Definitions to capture the conditions
variables (M P C : ℕ)
variable (cond1 : M + P = 32)
variable (cond2 : C = P + 20)

-- The theorem to prove
theorem average_marks_math_chem (M P C : ℕ) 
  (cond1 : M + P = 32) 
  (cond2 : C = P + 20) : 
  (M + C) / 2 = 26 := 
sorry

end average_marks_math_chem_l1126_112616


namespace initial_number_of_machines_l1126_112642

theorem initial_number_of_machines
  (x : ℕ)
  (h1 : x * 270 = 1080)
  (h2 : 20 * 3600 = 144000)
  (h3 : ∀ y, (20 * y * 4 = 3600) → y = 45) :
  x = 6 :=
by
  sorry

end initial_number_of_machines_l1126_112642


namespace sum_of_squares_of_sides_l1126_112681

-- Definition: A cyclic quadrilateral with perpendicular diagonals inscribed in a circle
structure CyclicQuadrilateral (R : ℝ) :=
  (m n k t : ℝ) -- sides of the quadrilateral
  (perpendicular_diagonals : true) -- diagonals are perpendicular (trivial placeholder)
  (radius : ℝ := R) -- Radius of the circumscribed circle

-- The theorem to prove: The sum of the squares of the sides of the quadrilateral is 8R^2
theorem sum_of_squares_of_sides (R : ℝ) (quad : CyclicQuadrilateral R) :
  quad.m ^ 2 + quad.n ^ 2 + quad.k ^ 2 + quad.t ^ 2 = 8 * R^2 := 
by sorry

end sum_of_squares_of_sides_l1126_112681


namespace geometric_mean_of_roots_l1126_112638

theorem geometric_mean_of_roots (x : ℝ) (h : x^2 = (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1)) : x = 1 ∨ x = -1 := 
by
  sorry

end geometric_mean_of_roots_l1126_112638


namespace concert_cost_l1126_112680

noncomputable def ticket_price : ℝ := 50.0
noncomputable def processing_fee_rate : ℝ := 0.15
noncomputable def parking_fee : ℝ := 10.0
noncomputable def entrance_fee : ℝ := 5.0
def number_of_people : ℕ := 2

noncomputable def processing_fee_per_ticket : ℝ := processing_fee_rate * ticket_price
noncomputable def total_cost_per_ticket : ℝ := ticket_price + processing_fee_per_ticket
noncomputable def total_ticket_cost : ℝ := number_of_people * total_cost_per_ticket
noncomputable def total_cost_with_parking : ℝ := total_ticket_cost + parking_fee
noncomputable def total_entrance_fee : ℝ := number_of_people * entrance_fee
noncomputable def total_cost : ℝ := total_cost_with_parking + total_entrance_fee

theorem concert_cost : total_cost = 135.0 := by
  sorry

end concert_cost_l1126_112680


namespace nina_earnings_l1126_112647

/-- 
Problem: Calculate the total earnings from selling various types of jewelry.
Conditions:
- Necklace price: $25 each
- Bracelet price: $15 each
- Earring price: $10 per pair
- Complete jewelry ensemble price: $45 each
- Number of necklaces sold: 5
- Number of bracelets sold: 10
- Number of earrings sold: 20
- Number of complete jewelry ensembles sold: 2
Question: How much money did Nina make over the weekend?
Answer: Nina made $565.00
-/
theorem nina_earnings
  (necklace_price : ℕ)
  (bracelet_price : ℕ)
  (earring_price : ℕ)
  (ensemble_price : ℕ)
  (necklaces_sold : ℕ)
  (bracelets_sold : ℕ)
  (earrings_sold : ℕ)
  (ensembles_sold : ℕ) :
  necklace_price = 25 → 
  bracelet_price = 15 → 
  earring_price = 10 → 
  ensemble_price = 45 → 
  necklaces_sold = 5 → 
  bracelets_sold = 10 → 
  earrings_sold = 20 → 
  ensembles_sold = 2 →
  (necklace_price * necklaces_sold) + 
  (bracelet_price * bracelets_sold) + 
  (earring_price * earrings_sold) +
  (ensemble_price * ensembles_sold) = 565 := by
  sorry

end nina_earnings_l1126_112647


namespace common_ratio_of_geometric_series_l1126_112682

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end common_ratio_of_geometric_series_l1126_112682


namespace isosceles_triangle_base_length_l1126_112627

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l1126_112627


namespace remainder_of_N_mod_45_l1126_112629

def concatenated_num_from_1_to_52 : ℕ := 
  -- This represents the concatenated number from 1 to 52.
  -- We define here in Lean as a placeholder 
  -- since Lean cannot concatenate numbers directly.
  12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152

theorem remainder_of_N_mod_45 : 
  concatenated_num_from_1_to_52 % 45 = 37 := 
sorry

end remainder_of_N_mod_45_l1126_112629


namespace adam_earnings_l1126_112696

theorem adam_earnings
  (earn_per_lawn : ℕ) (total_lawns : ℕ) (forgot_lawns : ℕ)
  (h1 : earn_per_lawn = 9) (h2 : total_lawns = 12) (h3 : forgot_lawns = 8) :
  (total_lawns - forgot_lawns) * earn_per_lawn = 36 :=
by
  sorry

end adam_earnings_l1126_112696


namespace man_saves_percentage_of_salary_l1126_112617

variable (S : ℝ) (P : ℝ) (S_s : ℝ)

def problem_statement (S : ℝ) (S_s : ℝ) (P : ℝ) : Prop :=
  S_s = S - 1.2 * (S - (P / 100) * S)

theorem man_saves_percentage_of_salary
  (h1 : S = 6250)
  (h2 : S_s = 250) :
  problem_statement S S_s 20 :=
by
  sorry

end man_saves_percentage_of_salary_l1126_112617


namespace greatest_possible_d_l1126_112620

noncomputable def point_2d_units_away_origin (d : ℝ) : Prop :=
  2 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d + 5)^2)

theorem greatest_possible_d : 
  ∃ d : ℝ, point_2d_units_away_origin d ∧ d = (5 + Real.sqrt 244) / 3 :=
sorry

end greatest_possible_d_l1126_112620


namespace ThreeDigitEvenNumbersCount_l1126_112668

theorem ThreeDigitEvenNumbersCount : 
  let a := 100
  let max := 998
  let d := 2
  let n := (max - a) / d + 1
  100 < 999 ∧ 100 % 2 = 0 ∧ max % 2 = 0 
  → d > 0 
  → n = 450 :=
by
  sorry

end ThreeDigitEvenNumbersCount_l1126_112668


namespace numerator_of_fraction_l1126_112666

-- Define the conditions
def y_pos (y : ℝ) : Prop := y > 0

-- Define the equation
def equation (x y : ℝ) : Prop := x + (3 * y) / 10 = (1 / 2) * y

-- Prove that x = (1/5) * y given the conditions
theorem numerator_of_fraction {y x : ℝ} (h1 : y_pos y) (h2 : equation x y) : x = (1/5) * y :=
  sorry

end numerator_of_fraction_l1126_112666


namespace simplify_expression_l1126_112652

theorem simplify_expression (n : ℕ) : 
  (3^(n + 3) - 3 * 3^n) / (3 * 3^(n + 2)) = 8 / 3 := 
sorry

end simplify_expression_l1126_112652


namespace range_of_a1_l1126_112651

theorem range_of_a1 (a : ℕ → ℕ) (S : ℕ → ℕ) (h_seq : ∀ n, 12 * S n = 4 * a (n + 1) + 5^n - 13)
  (h_S4 : ∀ n, S n ≤ S 4):
  13 / 48 ≤ a 1 ∧ a 1 ≤ 59 / 64 :=
sorry

end range_of_a1_l1126_112651


namespace M_inter_N_is_empty_l1126_112689

-- Definition conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | (x - 1) / x < 0}

-- Theorem statement
theorem M_inter_N_is_empty : M ∩ N = ∅ := by
  sorry

end M_inter_N_is_empty_l1126_112689


namespace identify_quadratic_equation_l1126_112615

theorem identify_quadratic_equation :
  (¬(∃ x y : ℝ, x^2 - 2*x*y + y^2 = 0) ∧  -- Condition A is not a quadratic equation
   ¬(∃ x : ℝ, x*(x + 3) = x^2 - 1) ∧      -- Condition B is not a quadratic equation
   (∃ x : ℝ, x^2 - 2*x - 3 = 0) ∧         -- Condition C is a quadratic equation
   ¬(∃ x : ℝ, x + (1/x) = 0)) →           -- Condition D is not a quadratic equation
  (true) := sorry

end identify_quadratic_equation_l1126_112615


namespace larger_angle_of_nonagon_l1126_112636

theorem larger_angle_of_nonagon : 
  ∀ (n : ℕ) (x : ℝ), 
  n = 9 → 
  (∃ a b : ℕ, a + b = n ∧ a * x + b * (3 * x) = 180 * (n - 2)) → 
  3 * (180 * (n - 2) / 15) = 252 :=
by
  sorry

end larger_angle_of_nonagon_l1126_112636


namespace square_area_l1126_112694

theorem square_area (side_length : ℝ) (h : side_length = 11) : side_length * side_length = 121 := 
by 
  simp [h]
  sorry

end square_area_l1126_112694


namespace simplify_sqrt_25000_l1126_112619

theorem simplify_sqrt_25000 : Real.sqrt 25000 = 50 * Real.sqrt 10 := 
by
  sorry

end simplify_sqrt_25000_l1126_112619


namespace exists_prime_q_l1126_112661

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) :
  ∃ q, Nat.Prime q ∧ ∀ n, ¬ (q ∣ n^p - p) := by
  sorry

end exists_prime_q_l1126_112661


namespace roots_of_quadratic_eq_l1126_112662

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l1126_112662


namespace triangles_with_positive_integer_area_count_l1126_112677

theorem triangles_with_positive_integer_area_count :
  let points := { p : (ℕ × ℕ) // 41 * p.1 + p.2 = 2017 }
  ∃ count, count = 600 ∧ ∀ (P Q : points), P ≠ Q →
    let area := (P.val.1 * Q.val.2 - Q.val.1 * P.val.2 : ℤ)
    0 < area ∧ (area % 2 = 0) := sorry

end triangles_with_positive_integer_area_count_l1126_112677


namespace radius_of_circle_l1126_112690

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

-- Prove that given the circle's equation, the radius is 1
theorem radius_of_circle (x y : ℝ) :
  circle_equation x y → ∃ (r : ℝ), r = 1 :=
by
  sorry

end radius_of_circle_l1126_112690


namespace a_5_value_l1126_112699

noncomputable def seq : ℕ → ℤ
| 0       => 1
| (n + 1) => (seq n) ^ 2 - 1

theorem a_5_value : seq 4 = -1 :=
by
  sorry

end a_5_value_l1126_112699


namespace general_term_arithmetic_sequence_l1126_112653

theorem general_term_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (a1 : a 1 = -1) 
  (d : ℤ) 
  (h : d = 4) : 
  ∀ n : ℕ, a n = 4 * n - 5 :=
by
  sorry

end general_term_arithmetic_sequence_l1126_112653


namespace Emily_beads_l1126_112612

-- Define the conditions and question
theorem Emily_beads (n k : ℕ) (h1 : k = 4) (h2 : n = 5) : n * k = 20 := by
  -- Sorry: this is a placeholder for the actual proof
  sorry

end Emily_beads_l1126_112612


namespace original_area_of_triangle_l1126_112608

theorem original_area_of_triangle (A : ℝ) (h1 : 4 * A * 16 = 64) : A = 4 :=
by
  sorry

end original_area_of_triangle_l1126_112608


namespace returning_players_count_l1126_112686

def total_players_in_team (groups : ℕ) (players_per_group : ℕ): ℕ := groups * players_per_group
def returning_players (total_players : ℕ) (new_players : ℕ): ℕ := total_players - new_players

theorem returning_players_count
    (new_players : ℕ)
    (groups : ℕ)
    (players_per_group : ℕ)
    (total_players : ℕ := total_players_in_team groups players_per_group)
    (returning_players_count : ℕ := returning_players total_players new_players):
    new_players = 4 ∧
    groups = 2 ∧
    players_per_group = 5 → 
    returning_players_count = 6 := by
    intros h
    sorry

end returning_players_count_l1126_112686


namespace johns_quarters_l1126_112671

variable (x : ℕ)  -- Number of quarters John has

def number_of_dimes : ℕ := x + 3  -- Number of dimes
def number_of_nickels : ℕ := x - 6  -- Number of nickels

theorem johns_quarters (h : x + (x + 3) + (x - 6) = 63) : x = 22 :=
by
  sorry

end johns_quarters_l1126_112671


namespace cake_recipe_l1126_112609

theorem cake_recipe (flour : ℕ) (milk_per_200ml : ℕ) (egg_per_200ml : ℕ) (total_flour : ℕ)
  (h1 : milk_per_200ml = 60)
  (h2 : egg_per_200ml = 1)
  (h3 : total_flour = 800) :
  (total_flour / 200 * milk_per_200ml = 240) ∧ (total_flour / 200 * egg_per_200ml = 4) :=
by
  sorry

end cake_recipe_l1126_112609


namespace correct_operation_l1126_112648

-- Define the conditions
def cond1 (m : ℝ) : Prop := m^2 + m^3 ≠ m^5
def cond2 (m : ℝ) : Prop := m^2 * m^3 = m^5
def cond3 (m : ℝ) : Prop := (m^2)^3 = m^6

-- Main statement that checks the correct operation
theorem correct_operation (m : ℝ) : cond1 m → cond2 m → cond3 m → (m^2 * m^3 = m^5) :=
by
  intros h1 h2 h3
  exact h2

end correct_operation_l1126_112648


namespace other_root_is_neg_2_l1126_112637

theorem other_root_is_neg_2 (k : ℝ) (h : Polynomial.eval 0 (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) : 
  ∃ t : ℝ, (Polynomial.eval t (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) ∧ t = -2 :=
by
  sorry

end other_root_is_neg_2_l1126_112637


namespace prove_math_problem_l1126_112654

noncomputable def ellipse_foci : Prop := 
  ∃ (a b : ℝ), 
  a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ),
  (x^2 / a^2 + y^2 / b^2 = 1) → 
  a = 2 ∧ b^2 = 3)

noncomputable def intersect_and_rhombus : Prop :=
  ∃ (m : ℝ) (t : ℝ),
  (3 * m^2 + 4) > 0 ∧ 
  t = 1 / (3 * m^2 + 4) ∧ 
  0 < t ∧ t < 1 / 4

theorem prove_math_problem : ellipse_foci ∧ intersect_and_rhombus :=
by sorry

end prove_math_problem_l1126_112654


namespace positive_integer_a_l1126_112657

theorem positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ (k : ℤ), (2 * a + 8) = k * (a + 1)) :
  a = 1 ∨ a = 2 ∨ a = 5 :=
by sorry

end positive_integer_a_l1126_112657


namespace g_one_fourth_l1126_112664

noncomputable def g : ℝ → ℝ := sorry

theorem g_one_fourth :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1) ∧  -- g(x) is defined for 0 ≤ x ≤ 1
  g 0 = 0 ∧                                    -- g(0) = 0
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧ -- g is non-decreasing
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧ -- symmetric property
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2)   -- scaling property
  → g (1/4) = 1/2 :=
sorry

end g_one_fourth_l1126_112664


namespace greatest_possible_value_of_n_l1126_112684

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 :=
by
  sorry

end greatest_possible_value_of_n_l1126_112684


namespace negation_of_universal_prop_l1126_112623

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end negation_of_universal_prop_l1126_112623


namespace number_of_classmates_ate_cake_l1126_112640

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l1126_112640


namespace estimate_height_of_student_l1126_112630

theorem estimate_height_of_student
  (x_values : List ℝ)
  (y_values : List ℝ)
  (h_sum_x : x_values.sum = 225)
  (h_sum_y : y_values.sum = 1600)
  (h_length : x_values.length = 10 ∧ y_values.length = 10)
  (b : ℝ := 4) :
  ∃ a : ℝ, ∀ x : ℝ, x = 24 → (b * x + a = 166) :=
by
  have avg_x := (225 / 10 : ℝ)
  have avg_y := (1600 / 10 : ℝ)
  have a := avg_y - b * avg_x
  use a
  intro x h
  rw [h]
  sorry

end estimate_height_of_student_l1126_112630


namespace validate_triangle_count_l1126_112605

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l1126_112605


namespace saree_sale_price_l1126_112695

def initial_price : Real := 150
def discount1 : Real := 0.20
def tax1 : Real := 0.05
def discount2 : Real := 0.15
def tax2 : Real := 0.04
def discount3 : Real := 0.10
def tax3 : Real := 0.03
def final_price : Real := 103.25

theorem saree_sale_price :
  let price_after_discount1 : Real := initial_price * (1 - discount1)
  let price_after_tax1 : Real := price_after_discount1 * (1 + tax1)
  let price_after_discount2 : Real := price_after_tax1 * (1 - discount2)
  let price_after_tax2 : Real := price_after_discount2 * (1 + tax2)
  let price_after_discount3 : Real := price_after_tax2 * (1 - discount3)
  let price_after_tax3 : Real := price_after_discount3 * (1 + tax3)
  abs (price_after_tax3 - final_price) < 0.01 :=
by
  sorry

end saree_sale_price_l1126_112695


namespace find_side_b_of_triangle_l1126_112658

theorem find_side_b_of_triangle
  (A B : Real) (a b : Real)
  (hA : A = Real.pi / 6)
  (hB : B = Real.pi / 4)
  (ha : a = 2) :
  b = 2 * Real.sqrt 2 :=
sorry

end find_side_b_of_triangle_l1126_112658
