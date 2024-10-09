import Mathlib

namespace modulus_of_z_equals_two_l1665_166577

namespace ComplexProblem

open Complex

-- Definition and conditions of the problem
def satisfies_condition (z : ℂ) : Prop :=
  (z + I) * (1 + I) = 1 - I

-- Statement that needs to be proven
theorem modulus_of_z_equals_two (z : ℂ) (h : satisfies_condition z) : abs z = 2 :=
sorry

end ComplexProblem

end modulus_of_z_equals_two_l1665_166577


namespace bird_families_flew_to_Asia_l1665_166565

-- Variables/Parameters
variable (A : ℕ) (X : ℕ)
axiom hA : A = 47
axiom hX : X = A + 47

-- Theorem Statement
theorem bird_families_flew_to_Asia : X = 94 :=
by
  sorry

end bird_families_flew_to_Asia_l1665_166565


namespace fraction_simplification_l1665_166512

theorem fraction_simplification (x : ℝ) (h : x = Real.sqrt 2) : 
  ( (x^2 - 1) / (x^2 - x) - 1) = Real.sqrt 2 / 2 :=
by 
  sorry

end fraction_simplification_l1665_166512


namespace simplify_expression_l1665_166532

theorem simplify_expression (x : ℝ) :
  (3 * x)^3 - (4 * x^2) * (2 * x^3) = 27 * x^3 - 8 * x^5 :=
by
  sorry

end simplify_expression_l1665_166532


namespace average_of_two_integers_l1665_166524

theorem average_of_two_integers {A B C D : ℕ} (h1 : A + B + C + D = 200) (h2 : C ≤ 130) : (A + B) / 2 = 35 :=
by
  sorry

end average_of_two_integers_l1665_166524


namespace lele_dongdong_meet_probability_l1665_166566

-- Define the conditions: distances and speeds
def segment_length : ℕ := 500
def n : ℕ := sorry
def d : ℕ := segment_length * n
def lele_speed : ℕ := 18
def dongdong_speed : ℕ := 24

-- Define times to traverse distance d
def t_L : ℚ := d / lele_speed
def t_D : ℚ := d / dongdong_speed

-- Define the time t when they meet
def t : ℚ := d / (lele_speed + dongdong_speed)

-- Define the maximum of t_L and t_D
def max_t_L_t_D : ℚ := max t_L t_D

-- Define the probability they meet on their way
def P_meet : ℚ := t / max_t_L_t_D

-- The theorem to prove the probability of meeting is 97/245
theorem lele_dongdong_meet_probability : P_meet = 97 / 245 :=
sorry

end lele_dongdong_meet_probability_l1665_166566


namespace smallest_number_divisible_by_20_and_36_l1665_166548

-- Define the conditions that x must be divisible by both 20 and 36
def divisible_by (x n : ℕ) : Prop := ∃ m : ℕ, x = n * m

-- Define the problem statement
theorem smallest_number_divisible_by_20_and_36 : 
  ∃ x : ℕ, divisible_by x 20 ∧ divisible_by x 36 ∧ 
  (∀ y : ℕ, (divisible_by y 20 ∧ divisible_by y 36) → y ≥ x) ∧ x = 180 := 
by
  sorry

end smallest_number_divisible_by_20_and_36_l1665_166548


namespace fraction_of_lollipops_given_to_emily_is_2_3_l1665_166517

-- Given conditions as definitions
def initial_lollipops := 42
def kept_lollipops := 4
def lou_received := 10

-- The fraction of lollipops given to Emily
def fraction_given_to_emily : ℚ :=
  have emily_received : ℚ := initial_lollipops - (kept_lollipops + lou_received)
  have total_lollipops : ℚ := initial_lollipops
  emily_received / total_lollipops

-- The proof statement assert that fraction_given_to_emily is equal to 2/3
theorem fraction_of_lollipops_given_to_emily_is_2_3 : fraction_given_to_emily = 2 / 3 := by
  sorry

end fraction_of_lollipops_given_to_emily_is_2_3_l1665_166517


namespace problem_solution_l1665_166582

open Real

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (∃ (C₁ : ℝ), (2 : ℝ)^x + (4 : ℝ)^y = C₁ ∧ C₁ = 2 * sqrt 2) ∧
  (∃ (C₂ : ℝ), 1 / x + 2 / y = C₂ ∧ C₂ = 9) ∧
  (∃ (C₃ : ℝ), x^2 + 4 * y^2 = C₃ ∧ C₃ = 1 / 2) :=
by
  sorry

end problem_solution_l1665_166582


namespace find_original_number_l1665_166591

-- Let x be the original number
def maria_operations (x : ℤ) : Prop :=
  (3 * (x - 3) + 3) / 3 = 10

theorem find_original_number (x : ℤ) (h : maria_operations x) : x = 12 :=
by
  sorry

end find_original_number_l1665_166591


namespace leah_coins_worth_89_cents_l1665_166588

variables (p n d : ℕ)

theorem leah_coins_worth_89_cents (h1 : p + n + d = 15) (h2 : d - 1 = n) : 
  1 * p + 5 * n + 10 * d = 89 := 
sorry

end leah_coins_worth_89_cents_l1665_166588


namespace student_marks_l1665_166557

theorem student_marks
(M P C : ℕ) -- the marks of Mathematics, Physics, and Chemistry are natural numbers
(h1 : C = P + 20)  -- Chemistry is 20 marks more than Physics
(h2 : (M + C) / 2 = 30)  -- The average marks in Mathematics and Chemistry is 30
: M + P = 40 := 
sorry

end student_marks_l1665_166557


namespace average_attendance_l1665_166569

def monday_attendance := 10
def tuesday_attendance := 15
def wednesday_attendance := 10
def thursday_attendance := 10
def friday_attendance := 10
def total_days := 5

theorem average_attendance :
  (monday_attendance + tuesday_attendance + wednesday_attendance + thursday_attendance + friday_attendance) / total_days = 11 :=
by
  sorry

end average_attendance_l1665_166569


namespace Brazil_wins_10_l1665_166590

/-- In the year 3000, the World Hockey Championship will follow new rules: 12 points will be awarded for a win, 
5 points will be deducted for a loss, and no points will be awarded for a draw. If the Brazilian team plays 
38 matches, scores 60 points, and loses at least once, then the number of wins they can achieve is 10. 
List all possible scenarios and justify why there cannot be any others. -/
theorem Brazil_wins_10 (x y z : ℕ) 
    (h1: x + y + z = 38) 
    (h2: 12 * x - 5 * y = 60) 
    (h3: y ≥ 1)
    (h4: z ≥ 0): 
  x = 10 :=
by
  sorry

end Brazil_wins_10_l1665_166590


namespace expected_balls_in_original_positions_after_transpositions_l1665_166510

theorem expected_balls_in_original_positions_after_transpositions :
  let num_balls := 7
  let first_swap_probability := 2 / 7
  let second_swap_probability := 1 / 7
  let third_swap_probability := 1 / 7
  let original_position_probability := (2 / 343) + (125 / 343)
  let expected_balls := num_balls * original_position_probability
  expected_balls = 889 / 343 := 
sorry

end expected_balls_in_original_positions_after_transpositions_l1665_166510


namespace product_form_l1665_166516

theorem product_form (a b c d : ℤ) :
    (a^2 - 7*b^2) * (c^2 - 7*d^2) = (a*c + 7*b*d)^2 - 7*(a*d + b*c)^2 :=
by sorry

end product_form_l1665_166516


namespace expression_equivalence_l1665_166599

theorem expression_equivalence (a b : ℝ) :
  let P := a + b
  let Q := a - b
  (P + Q)^2 / (P - Q)^2 - (P - Q)^2 / (P + Q)^2 = (a^2 + b^2) * (a^2 - b^2) / (a^2 * b^2) :=
by
  sorry

end expression_equivalence_l1665_166599


namespace area_inside_circle_outside_square_is_zero_l1665_166514

theorem area_inside_circle_outside_square_is_zero 
  (side_length : ℝ) (circle_radius : ℝ)
  (h_square_side : side_length = 2) (h_circle_radius : circle_radius = 1) : 
  (π * circle_radius^2) - (side_length^2) = 0 := 
by 
  sorry

end area_inside_circle_outside_square_is_zero_l1665_166514


namespace max_sum_is_38_l1665_166525

-- Definition of the problem variables and conditions
def number_set : Set ℤ := {2, 3, 8, 9, 14, 15}
variable (a b c d e : ℤ)

-- Conditions translated to Lean
def condition1 : Prop := b = c
def condition2 : Prop := a = d

-- Sum condition to find maximum sum
def max_combined_sum : ℤ := a + b + e

theorem max_sum_is_38 : 
  ∃ a b c d e, 
    {a, b, c, d, e} ⊆ number_set ∧
    b = c ∧ 
    a = d ∧ 
    a + b + e = 38 :=
sorry

end max_sum_is_38_l1665_166525


namespace sum_of_interior_angles_of_pentagon_l1665_166509

theorem sum_of_interior_angles_of_pentagon :
  let n := 5
  let angleSum := 180 * (n - 2)
  angleSum = 540 :=
by
  sorry

end sum_of_interior_angles_of_pentagon_l1665_166509


namespace ratio_of_Katie_to_Cole_l1665_166542

variable (K C : ℕ)

theorem ratio_of_Katie_to_Cole (h1 : 3 * K = 84) (h2 : C = 7) : K / C = 4 :=
by
  sorry

end ratio_of_Katie_to_Cole_l1665_166542


namespace rise_in_water_level_l1665_166587

noncomputable def edge : ℝ := 15.0
noncomputable def base_length : ℝ := 20.0
noncomputable def base_width : ℝ := 15.0
noncomputable def volume_cube : ℝ := edge ^ 3
noncomputable def base_area : ℝ := base_length * base_width

theorem rise_in_water_level :
  (volume_cube / base_area) = 11.25 :=
by
  sorry

end rise_in_water_level_l1665_166587


namespace sum_of_fractions_l1665_166559

theorem sum_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (7 : ℚ) / 9
  a + b = 83 / 72 := 
by
  sorry

end sum_of_fractions_l1665_166559


namespace symmetric_colors_different_at_8281_div_2_l1665_166541

def is_red (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ n = 81 * x + 100 * y

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n

theorem symmetric_colors_different_at_8281_div_2 :
  ∃ n : ℕ, (is_red n ∧ is_blue (8281 - n)) ∨ (is_blue n ∧ is_red (8281 - n)) ∧ 2 * n = 8281 :=
by
  sorry

end symmetric_colors_different_at_8281_div_2_l1665_166541


namespace second_tap_empty_time_l1665_166523

theorem second_tap_empty_time :
  ∃ T : ℝ, (1 / 4 - 1 / T = 3 / 28) → T = 7 :=
by
  sorry

end second_tap_empty_time_l1665_166523


namespace find_common_difference_l1665_166551

variable {aₙ : ℕ → ℝ}
variable {Sₙ : ℕ → ℝ}

-- Condition that the sum of the first n terms of the arithmetic sequence is S_n
def is_arith_seq (aₙ : ℕ → ℝ) (Sₙ : ℕ → ℝ) : Prop :=
  ∀ n, Sₙ n = (n * (aₙ 0 + (aₙ (n - 1))) / 2)

-- Condition given in the problem
def problem_condition (Sₙ : ℕ → ℝ) : Prop :=
  2 * Sₙ 3 - 3 * Sₙ 2 = 12

theorem find_common_difference (h₀ : is_arith_seq aₙ Sₙ) (h₁ : problem_condition Sₙ) : 
  ∃ d : ℝ, d = 4 := 
sorry

end find_common_difference_l1665_166551


namespace abc_value_l1665_166594

theorem abc_value (a b c : ℂ) (h1 : 2 * a * b + 3 * b = -21)
                   (h2 : 2 * b * c + 3 * c = -21)
                   (h3 : 2 * c * a + 3 * a = -21) :
                   a * b * c = 105.75 := 
sorry

end abc_value_l1665_166594


namespace circle_standard_equation_l1665_166567

theorem circle_standard_equation (x y : ℝ) (center : ℝ × ℝ) (radius : ℝ) 
  (h_center : center = (2, -1)) (h_radius : radius = 2) :
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 ↔ (x - 2) ^ 2 + (y + 1) ^ 2 = 4 := by
  sorry

end circle_standard_equation_l1665_166567


namespace bob_total_profit_l1665_166576

/-- Define the cost of each dog --/
def dog_cost : ℝ := 250.0

/-- Define the number of dogs Bob bought --/
def number_of_dogs : ℕ := 2

/-- Define the total cost of the dogs --/
def total_cost_for_dogs : ℝ := dog_cost * number_of_dogs

/-- Define the selling price of each puppy --/
def puppy_selling_price : ℝ := 350.0

/-- Define the number of puppies --/
def number_of_puppies : ℕ := 6

/-- Define the total revenue from selling the puppies --/
def total_revenue_from_puppies : ℝ := puppy_selling_price * number_of_puppies

/-- Define Bob's total profit from selling the puppies --/
def total_profit : ℝ := total_revenue_from_puppies - total_cost_for_dogs

/-- The theorem stating that Bob's total profit is $1600.00 --/
theorem bob_total_profit : total_profit = 1600.0 := 
by
  /- We leave the proof out as we just need the statement -/
  sorry

end bob_total_profit_l1665_166576


namespace even_three_digit_numbers_less_than_600_l1665_166573

def count_even_three_digit_numbers : ℕ :=
  let hundreds_choices := 5
  let tens_choices := 6
  let units_choices := 3
  hundreds_choices * tens_choices * units_choices

theorem even_three_digit_numbers_less_than_600 : count_even_three_digit_numbers = 90 := by
  -- sorry ensures that the statement type checks even without the proof.
  sorry

end even_three_digit_numbers_less_than_600_l1665_166573


namespace initial_deck_card_count_l1665_166508

-- Define the initial conditions
def initial_red_probability (r b : ℕ) : Prop := r * 4 = r + b
def added_black_probability (r b : ℕ) : Prop := r * 5 = 4 * r + 6

theorem initial_deck_card_count (r b : ℕ) (h1 : initial_red_probability r b) (h2 : added_black_probability r b) : r + b = 24 := 
by sorry

end initial_deck_card_count_l1665_166508


namespace sum_medians_is_64_l1665_166580

noncomputable def median (l: List ℝ) : ℝ := sorry  -- Placeholder for median calculation

open List

/-- Define the scores for players A and B as lists of real numbers -/
def player_a_scores : List ℝ := sorry
def player_b_scores : List ℝ := sorry

/-- Prove that the sum of the medians of the scores lists is 64 -/
theorem sum_medians_is_64 : median player_a_scores + median player_b_scores = 64 := sorry

end sum_medians_is_64_l1665_166580


namespace prime_gt3_43_divides_expression_l1665_166503

theorem prime_gt3_43_divides_expression {p : ℕ} (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (7^p - 6^p - 1) % 43 = 0 := 
  sorry

end prime_gt3_43_divides_expression_l1665_166503


namespace original_number_of_workers_l1665_166549

-- Definitions of the conditions given in the problem
def workers_days (W : ℕ) : ℕ := 35
def additional_workers : ℕ := 10
def reduced_days : ℕ := 10

-- The main theorem we need to prove
theorem original_number_of_workers (W : ℕ) (A : ℕ) 
  (h1 : W * workers_days W = (W + additional_workers) * (workers_days W - reduced_days)) :
  W = 25 :=
by
  sorry

end original_number_of_workers_l1665_166549


namespace maximum_sequence_length_l1665_166585

theorem maximum_sequence_length
  (seq : List ℚ) 
  (h1 : ∀ i : ℕ, i + 2 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2)) < 0)
  (h2 : ∀ i : ℕ, i + 3 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2) + seq.get! (i+3)) > 0) 
  : seq.length ≤ 5 := 
sorry

end maximum_sequence_length_l1665_166585


namespace arithmetic_geometric_mean_l1665_166545

theorem arithmetic_geometric_mean (a b : ℝ) 
  (h1 : (a + b) / 2 = 20) 
  (h2 : Real.sqrt (a * b) = Real.sqrt 135) : 
  a^2 + b^2 = 1330 :=
by
  sorry

end arithmetic_geometric_mean_l1665_166545


namespace y_relationship_l1665_166562

variable (a c : ℝ) (h_a : a < 0)

def f (x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

theorem y_relationship (y1 y2 y3 : ℝ)
  (h1 : y1 = f a c (Real.sqrt 5))
  (h2 : y2 = f a c 0)
  (h3 : y3 = f a c 4) :
  y2 < y3 ∧ y3 < y1 :=
  sorry

end y_relationship_l1665_166562


namespace tennis_balls_ordered_l1665_166522

variables (W Y : ℕ)
def original_eq (W Y : ℕ) := W = Y
def ratio_condition (W Y : ℕ) := W / (Y + 90) = 8 / 13
def total_tennis_balls (W Y : ℕ) := W + Y = 288

theorem tennis_balls_ordered (W Y : ℕ) (h1 : original_eq W Y) (h2 : ratio_condition W Y) : total_tennis_balls W Y :=
sorry

end tennis_balls_ordered_l1665_166522


namespace distance_X_X_l1665_166560

/-
  Define the vertices of the triangle XYZ
-/
def X : ℝ × ℝ := (2, -4)
def Y : ℝ × ℝ := (-1, 2)
def Z : ℝ × ℝ := (5, 1)

/-
  Define the reflection of point X over the y-axis
-/
def X' : ℝ × ℝ := (-2, -4)

/-
  Prove that the distance between X and X' is 4 units.
-/
theorem distance_X_X' : (Real.sqrt (((-2) - 2) ^ 2 + ((-4) - (-4)) ^ 2)) = 4 := by
  sorry

end distance_X_X_l1665_166560


namespace vector_rotation_correct_l1665_166511

def vector_rotate_z_90 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := v
  ( -y, x, z )

theorem vector_rotation_correct :
  vector_rotate_z_90 (3, -1, 4) = (-3, 0, 4) := 
by 
  sorry

end vector_rotation_correct_l1665_166511


namespace range_of_a_l1665_166537

theorem range_of_a (a : ℝ) : ({x : ℝ | a - 4 < x ∧ x < a + 4} ⊆ {x : ℝ | 1 < x ∧ x < 3}) → (-1 ≤ a ∧ a ≤ 5) := by
  sorry

end range_of_a_l1665_166537


namespace arithmetic_seq_solution_l1665_166534

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Definition of arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of first n terms of arithmetic sequence
def sum_arithmetic_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) / 2 * (a 0 + a n)

-- Given conditions
def given_conditions (a : ℕ → ℝ) : Prop :=
  a 0 + a 4 + a 8 = 27

-- Main theorem to be proved
theorem arithmetic_seq_solution (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (ha : arithmetic_seq a d)
  (hs : sum_arithmetic_seq S a)
  (h_given : given_conditions a) :
  a 4 = 9 ∧ S 8 = 81 :=
sorry

end arithmetic_seq_solution_l1665_166534


namespace number_of_positive_area_triangles_l1665_166563

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l1665_166563


namespace ratio_of_sides_l1665_166529

theorem ratio_of_sides (a b : ℝ) (h1 : a + b = 3 * a) (h2 : a + b - Real.sqrt (a^2 + b^2) = (1 / 3) * b) : a / b = 1 / 2 :=
sorry

end ratio_of_sides_l1665_166529


namespace movie_theorem_l1665_166539

variables (A B C D : Prop)

theorem movie_theorem 
  (h1 : (A → B))
  (h2 : (B → C))
  (h3 : (C → A))
  (h4 : (D → B)) 
  : ¬D := 
by
  sorry

end movie_theorem_l1665_166539


namespace chicken_legs_baked_l1665_166500

theorem chicken_legs_baked (L : ℕ) (H₁ : 144 / 16 = 9) (H₂ : 224 / 16 = 14) (H₃ : 16 * 9 = 144) :  L = 144 :=
by
  sorry

end chicken_legs_baked_l1665_166500


namespace diane_stamp_combinations_l1665_166506

/-- Define the types of stamps Diane has --/
def diane_stamps : List ℕ := [1, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8]

/-- Define the condition for the correct number of different arrangements to sum exactly to 12 cents -/
noncomputable def count_arrangements (stamps : List ℕ) (sum : ℕ) : ℕ :=
  -- Implementation of the counting function goes here
  sorry

/-- Prove that the number of distinct arrangements to make exactly 12 cents is 13 --/
theorem diane_stamp_combinations : count_arrangements diane_stamps 12 = 13 :=
  sorry

end diane_stamp_combinations_l1665_166506


namespace probability_quadratic_real_roots_l1665_166526

noncomputable def probability_real_roots : ℝ := 3 / 4

theorem probability_quadratic_real_roots :
  (∀ a b : ℝ, -π ≤ a ∧ a ≤ π ∧ -π ≤ b ∧ b ≤ π →
  (∃ x : ℝ, x^2 + 2*a*x - b^2 + π = 0) ↔ a^2 + b^2 ≥ π) →
  (probability_real_roots = 3 / 4) :=
sorry

end probability_quadratic_real_roots_l1665_166526


namespace speed_of_first_boy_l1665_166544

-- Variables for speeds and time
variables (v : ℝ) (t : ℝ) (d : ℝ)

-- Given conditions
def initial_conditions := 
  v > 0 ∧ 
  7.5 > 0 ∧ 
  t = 10 ∧ 
  d = 20

-- Theorem statement with the conditions and the expected answer
theorem speed_of_first_boy
  (h : initial_conditions v t d) : 
  v = 9.5 :=
sorry

end speed_of_first_boy_l1665_166544


namespace divisibility_by_11_l1665_166519

theorem divisibility_by_11 (m n : ℤ) (h : (5 * m + 3 * n) % 11 = 0) : (9 * m + n) % 11 = 0 := by
  sorry

end divisibility_by_11_l1665_166519


namespace tank_capacity_l1665_166558

-- Define the initial fullness of the tank and the total capacity
def initial_fullness (w c : ℝ) : Prop :=
  w = c / 5

-- Define the fullness of the tank after adding 5 liters
def fullness_after_adding (w c : ℝ) : Prop :=
  (w + 5) / c = 2 / 7

-- The main theorem: if both conditions hold, c must equal to 35/3
theorem tank_capacity (w c : ℝ) (h1 : initial_fullness w c) (h2 : fullness_after_adding w c) : 
  c = 35 / 3 :=
sorry

end tank_capacity_l1665_166558


namespace minimum_value_expression_l1665_166586

theorem minimum_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  (∃ a b c : ℝ, (b > c ∧ c > a) ∧ b ≠ 0 ∧ (a + b) = b - c ∧ (b - c) = c - a ∧ (a - c) = 0 ∧
   ∀ x y z : ℝ, (x = a + b ∧ y = b - c ∧ z = c - a) → 
    (x^2 + y^2 + z^2) / b^2 = 4/3) :=
  sorry

end minimum_value_expression_l1665_166586


namespace area_percentage_l1665_166592

theorem area_percentage (D_S D_R : ℝ) (h : D_R = 0.8 * D_S) : 
  let R_S := D_S / 2
  let R_R := D_R / 2
  let A_S := π * R_S^2
  let A_R := π * R_R^2
  (A_R / A_S) * 100 = 64 := 
by
  sorry

end area_percentage_l1665_166592


namespace conic_section_is_ellipse_l1665_166520

open Real

def is_conic_section_ellipse (x y : ℝ) (k : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  sqrt ((x - p1.1) ^ 2 + (y - p1.2) ^ 2) + sqrt ((x - p2.1) ^ 2 + (y - p2.2) ^ 2) = k

theorem conic_section_is_ellipse :
  is_conic_section_ellipse 2 (-2) 12 (2, -2) (-3, 5) :=
by
  sorry

end conic_section_is_ellipse_l1665_166520


namespace magician_inequality_l1665_166555

theorem magician_inequality (N : ℕ) : 
  (N - 1) * 10^(N - 2) ≥ 10^N → N ≥ 101 :=
by
  sorry

end magician_inequality_l1665_166555


namespace jill_total_tax_percentage_l1665_166596

theorem jill_total_tax_percentage (spent_clothing_percent spent_food_percent spent_other_percent tax_clothing_percent tax_food_percent tax_other_percent : ℝ)
  (h1 : spent_clothing_percent = 0.5)
  (h2 : spent_food_percent = 0.25)
  (h3 : spent_other_percent = 0.25)
  (h4 : tax_clothing_percent = 0.1)
  (h5 : tax_food_percent = 0)
  (h6 : tax_other_percent = 0.2) :
  ((spent_clothing_percent * tax_clothing_percent + spent_food_percent * tax_food_percent + spent_other_percent * tax_other_percent) * 100) = 10 :=
by
  sorry

end jill_total_tax_percentage_l1665_166596


namespace music_tool_cost_l1665_166521

namespace BandCost

def trumpet_cost : ℝ := 149.16
def song_book_cost : ℝ := 4.14
def total_spent : ℝ := 163.28

theorem music_tool_cost : (total_spent - (trumpet_cost + song_book_cost)) = 9.98 :=
by
  sorry

end music_tool_cost_l1665_166521


namespace arith_seq_sum_l1665_166547

theorem arith_seq_sum (a₃ a₄ a₅ : ℤ) (h₁ : a₃ = 7) (h₂ : a₄ = 11) (h₃ : a₅ = 15) :
  let d := a₄ - a₃;
  let a := a₄ - 3 * d;
  (6 / 2 * (2 * a + 5 * d)) = 54 :=
by
  sorry

end arith_seq_sum_l1665_166547


namespace fraction_of_oranges_is_correct_l1665_166504

variable (O P A : ℕ)
variable (total_fruit : ℕ := 56)

theorem fraction_of_oranges_is_correct:
  (A = 35) →
  (P = O / 2) →
  (A = 5 * P) →
  (O + P + A = total_fruit) →
  (O / total_fruit = 1 / 4) :=
by
  -- proof to be filled in 
  sorry

end fraction_of_oranges_is_correct_l1665_166504


namespace diamond_more_olivine_l1665_166536

theorem diamond_more_olivine :
  ∃ A O D : ℕ, A = 30 ∧ O = A + 5 ∧ A + O + D = 111 ∧ D - O = 11 :=
by
  sorry

end diamond_more_olivine_l1665_166536


namespace find_C_l1665_166513

variable (A B C : ℚ)

def condition1 := A + B + C = 350
def condition2 := A + C = 200
def condition3 := B + C = 350

theorem find_C : condition1 A B C → condition2 A C → condition3 B C → C = 200 :=
by
  sorry

end find_C_l1665_166513


namespace average_rate_of_change_l1665_166501

noncomputable def f (x : ℝ) : ℝ := x^2 + 1

theorem average_rate_of_change (Δx : ℝ) : 
  (f (1 + Δx) - f 1) / Δx = 2 + Δx := 
by
  sorry

end average_rate_of_change_l1665_166501


namespace total_rankings_l1665_166505

-- Defines the set of players
inductive Player
| P : Player
| Q : Player
| R : Player
| S : Player

-- Defines a function to count the total number of ranking sequences
def total_possible_rankings (p : Player → Player → Prop) : Nat := 
  4 * 2 * 2

-- Problem statement
theorem total_rankings : ∃ t : Player → Player → Prop, total_possible_rankings t = 16 :=
by
  sorry

end total_rankings_l1665_166505


namespace part_one_min_f_value_part_two_range_a_l1665_166556

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x + a|

theorem part_one_min_f_value (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≥ (3/2) :=
  sorry

theorem part_two_range_a (a : ℝ) : (11/2 < a) ∧ (a < 4.5) :=
  sorry

end part_one_min_f_value_part_two_range_a_l1665_166556


namespace greatest_A_satisfies_condition_l1665_166543

theorem greatest_A_satisfies_condition :
  ∃ (A : ℝ), A = 64 ∧ ∀ (s : Fin₇ → ℝ), (∀ i, 1 ≤ s i ∧ s i ≤ A) →
  ∃ (i j : Fin₇), i ≠ j ∧ (1 / 2 ≤ s i / s j ∧ s i / s j ≤ 2) :=
by 
  sorry

end greatest_A_satisfies_condition_l1665_166543


namespace infinite_squares_form_l1665_166533

theorem infinite_squares_form (k : ℕ) (hk : 0 < k) : ∃ f : ℕ → ℕ, ∀ n, ∃ a, a^2 = f n * 2^k - 7 :=
by
  sorry

end infinite_squares_form_l1665_166533


namespace least_positive_x_l1665_166553

theorem least_positive_x (x : ℕ) (h : (2 * x + 45)^2 % 43 = 0) : x = 42 :=
  sorry

end least_positive_x_l1665_166553


namespace scientific_notation_28400_is_correct_l1665_166538

theorem scientific_notation_28400_is_correct : (28400 : ℝ) = 2.84 * 10^4 := 
by 
  sorry

end scientific_notation_28400_is_correct_l1665_166538


namespace one_twenty_percent_of_number_l1665_166540

theorem one_twenty_percent_of_number (x : ℝ) (h : 0.20 * x = 300) : 1.20 * x = 1800 :=
by 
sorry

end one_twenty_percent_of_number_l1665_166540


namespace max_principals_in_8_years_l1665_166527

theorem max_principals_in_8_years 
  (years_in_term : ℕ)
  (terms_in_given_period : ℕ)
  (term_length : ℕ)
  (term_length_eq : term_length = 4)
  (given_period : ℕ)
  (given_period_eq : given_period = 8) :
  terms_in_given_period = given_period / term_length :=
by
  rw [term_length_eq, given_period_eq]
  sorry

end max_principals_in_8_years_l1665_166527


namespace full_price_ticket_revenue_l1665_166530

theorem full_price_ticket_revenue 
  (f h p : ℕ)
  (h1 : f + h = 160)
  (h2 : f * p + h * (p / 3) = 2400) :
  f * p = 400 := 
sorry

end full_price_ticket_revenue_l1665_166530


namespace simplify_expression_1_simplify_expression_2_l1665_166571

theorem simplify_expression_1 (x y : ℝ) :
  x^2 + 5*y - 4*x^2 - 3*y = -3*x^2 + 2*y :=
sorry

theorem simplify_expression_2 (a b : ℝ) :
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b :=
sorry

end simplify_expression_1_simplify_expression_2_l1665_166571


namespace sin_60_eq_sqrt_three_div_two_l1665_166578

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l1665_166578


namespace sufficient_not_necessary_implies_a_lt_1_l1665_166593

theorem sufficient_not_necessary_implies_a_lt_1 {x a : ℝ} (h : ∀ x : ℝ, x > 1 → x > a ∧ ¬(x > a → x > 1)) : a < 1 :=
sorry

end sufficient_not_necessary_implies_a_lt_1_l1665_166593


namespace find_total_children_l1665_166561

-- Define conditions as a Lean structure
structure SchoolDistribution where
  B : ℕ     -- Total number of bananas
  C : ℕ     -- Total number of children
  absent : ℕ := 160      -- Number of absent children (constant)
  bananas_per_child : ℕ := 2 -- Bananas per child originally (constant)
  bananas_extra : ℕ := 2      -- Extra bananas given to present children (constant)

-- Define the theorem we want to prove
theorem find_total_children (dist : SchoolDistribution) 
  (h1 : dist.B = 2 * dist.C) 
  (h2 : dist.B = 4 * (dist.C - dist.absent)) :
  dist.C = 320 := by
  sorry

end find_total_children_l1665_166561


namespace decagon_diagonals_l1665_166515

-- Definition of the number of diagonals in a polygon with n sides.
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The proof problem statement
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l1665_166515


namespace symmetric_codes_count_l1665_166589

def isSymmetric (grid : List (List Bool)) : Prop :=
  -- condition for symmetry: rotational and reflectional symmetry
  sorry

def isValidCode (grid : List (List Bool)) : Prop :=
  -- condition for valid scanning code with at least one black and one white
  sorry

noncomputable def numberOfSymmetricCodes : Nat :=
  -- function to count the number of symmetric valid codes
  sorry

theorem symmetric_codes_count :
  numberOfSymmetricCodes = 62 := 
  sorry

end symmetric_codes_count_l1665_166589


namespace prove_a_eq_neg2_solve_inequality_for_a_leq0_l1665_166579

-- Problem 1: Proving that a = -2 given the solution set of the inequality
theorem prove_a_eq_neg2 (a : ℝ) (h : ∀ x : ℝ, (-1 < x ∧ x < -1/2) ↔ (ax - 1) * (x + 1) > 0) : a = -2 := sorry

-- Problem 2: Solving the inequality (ax-1)(x+1) > 0 for different conditions on a
theorem solve_inequality_for_a_leq0 (a x : ℝ) (h_a_le_0 : a ≤ 0) : 
  (ax - 1) * (x + 1) > 0 ↔ 
    if a < -1 then -1 < x ∧ x < 1/a
    else if a = -1 then false
    else if -1 < a ∧ a < 0 then 1/a < x ∧ x < -1
    else x < -1 := sorry

end prove_a_eq_neg2_solve_inequality_for_a_leq0_l1665_166579


namespace smallest_e_value_l1665_166568

noncomputable def poly := (1, -3, 7, -2/5)

theorem smallest_e_value (a b c d e : ℤ) 
  (h_poly_eq : a * (1)^4 + b * (1)^3 + c * (1)^2 + d * (1) + e = 0)
  (h_poly_eq_2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h_poly_eq_3 : a * 7^4 + b * 7^3 + c * 7^2 + d * 7 + e = 0)
  (h_poly_eq_4 : a * (-2/5)^4 + b * (-2/5)^3 + c * (-2/5)^2 + d * (-2/5) + e = 0)
  (h_e_positive : e > 0) :
  e = 42 :=
sorry

end smallest_e_value_l1665_166568


namespace find_x_l1665_166574

theorem find_x 
  (x : ℝ)
  (h : 0.4 * x + (0.6 * 0.8) = 0.56) : 
  x = 0.2 := sorry

end find_x_l1665_166574


namespace range_of_a_l1665_166550

theorem range_of_a (a x : ℝ) (h : x - a = 1 - 2*x) (non_neg_x : x ≥ 0) : a ≥ -1 := by
  sorry

end range_of_a_l1665_166550


namespace customer_survey_response_l1665_166502

theorem customer_survey_response (N : ℕ)
  (avg_income : ℕ → ℕ)
  (avg_all : avg_income N = 45000)
  (avg_top10 : avg_income 10 = 55000)
  (avg_others : avg_income (N - 10) = 42500) :
  N = 50 := 
sorry

end customer_survey_response_l1665_166502


namespace evaluate_expr_at_2_l1665_166564

def expr (x : ℝ) : ℝ := (2 * x + 3) * (2 * x - 3) + (x - 2) ^ 2 - 3 * x * (x - 1)

theorem evaluate_expr_at_2 : expr 2 = 1 :=
by
  sorry

end evaluate_expr_at_2_l1665_166564


namespace geometric_first_term_l1665_166570

-- Define the conditions
def is_geometric_series (first_term : ℝ) (r : ℝ) (sum : ℝ) : Prop :=
  sum = first_term / (1 - r)

-- Define the main theorem
theorem geometric_first_term (r : ℝ) (sum : ℝ) (first_term : ℝ) 
  (h_r : r = 1/4) (h_S : sum = 80) (h_sum_formula : is_geometric_series first_term r sum) : 
  first_term = 60 :=
by
  sorry

end geometric_first_term_l1665_166570


namespace ratio_of_triangle_areas_l1665_166554

theorem ratio_of_triangle_areas 
  (r s : ℝ) (n : ℝ)
  (h_ratio : 3 * s = r) 
  (h_area : (3 / 2) * n = 1 / 2 * r * ((3 * n * 2) / r)) :
  3 / 3 = n :=
by
  sorry

end ratio_of_triangle_areas_l1665_166554


namespace biff_hourly_earnings_l1665_166584

theorem biff_hourly_earnings:
  let ticket_cost := 11
  let drinks_snacks_cost := 3
  let headphones_cost := 16
  let wifi_cost_per_hour := 2
  let bus_ride_hours := 3
  let total_non_wifi_expenses := ticket_cost + drinks_snacks_cost + headphones_cost
  let total_wifi_cost := bus_ride_hours * wifi_cost_per_hour
  let total_expenses := total_non_wifi_expenses + total_wifi_cost
  ∀ (x : ℝ), 3 * x = total_expenses → x = 12 :=
by sorry -- Proof skipped

end biff_hourly_earnings_l1665_166584


namespace isosceles_triangle_base_angles_l1665_166552

theorem isosceles_triangle_base_angles (a b : ℝ) (h1 : a + b + b = 180)
  (h2 : a = 110) : b = 35 :=
by 
  sorry

end isosceles_triangle_base_angles_l1665_166552


namespace similar_triangle_legs_l1665_166535

theorem similar_triangle_legs (y : ℝ) 
  (h1 : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 15 ∧ b = 12)
  (h2 : ∃ u v w : ℝ, u^2 + v^2 = w^2 ∧ u = y ∧ v = 9) 
  (h3 : ∀ (a b c u v w : ℝ), (a^2 + b^2 = c^2 ∧ u^2 + v^2 = w^2 ∧ a/u = b/v) → (a = b → u = v)) 
  : y = 11.25 := 
  by 
    sorry

end similar_triangle_legs_l1665_166535


namespace compute_f_1_g_3_l1665_166581

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x + 2

theorem compute_f_1_g_3 : f (1 + g 3) = 7 := 
by
  -- Proof goes here
  sorry

end compute_f_1_g_3_l1665_166581


namespace monotonic_function_a_ge_one_l1665_166546

theorem monotonic_function_a_ge_one (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2 * x + a) ≥ 0) → a ≥ 1 :=
by
  intros h
  sorry

end monotonic_function_a_ge_one_l1665_166546


namespace sales_fifth_month_l1665_166583

-- Definitions based on conditions
def sales1 : ℝ := 5420
def sales2 : ℝ := 5660
def sales3 : ℝ := 6200
def sales4 : ℝ := 6350
def sales6 : ℝ := 8270
def average_sale : ℝ := 6400

-- Lean proof problem statement
theorem sales_fifth_month :
  sales1 + sales2 + sales3 + sales4 + sales6 + s = 6 * average_sale  →
  s = 6500 :=
by
  sorry

end sales_fifth_month_l1665_166583


namespace can_divide_2007_triangles_can_divide_2008_triangles_l1665_166572

theorem can_divide_2007_triangles :
  ∃ k : ℕ, 2007 = 9 + 3 * k :=
by
  sorry

theorem can_divide_2008_triangles :
  ∃ m : ℕ, 2008 = 4 + 3 * m :=
by
  sorry

end can_divide_2007_triangles_can_divide_2008_triangles_l1665_166572


namespace number_of_pairs_l1665_166528

theorem number_of_pairs (h : ∀ (a : ℝ) (b : ℕ), 0 < a → 2 ≤ b ∧ b ≤ 200 → (Real.log a / Real.log b) ^ 2017 = Real.log (a ^ 2017) / Real.log b) :
  ∃ n, n = 597 ∧ ∀ b : ℕ, 2 ≤ b ∧ b ≤ 200 → 
    ∃ a1 a2 a3 : ℝ, 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 
      (Real.log a1 / Real.log b) = 0 ∧ 
      (Real.log a2 / Real.log b) = 2017^((1:ℝ)/2016) ∧ 
      (Real.log a3 / Real.log b) = -2017^((1:ℝ)/2016) :=
sorry

end number_of_pairs_l1665_166528


namespace greatest_possible_grapes_thrown_out_l1665_166531

theorem greatest_possible_grapes_thrown_out (n : ℕ) : 
  n % 7 ≤ 6 := by 
  sorry

end greatest_possible_grapes_thrown_out_l1665_166531


namespace find_initial_number_l1665_166595

theorem find_initial_number (N : ℕ) (k : ℤ) (h : N - 3 = 15 * k) : N = 18 := 
by
  sorry

end find_initial_number_l1665_166595


namespace number_of_people_per_cubic_yard_l1665_166518

-- Lean 4 statement

variable (P : ℕ) -- Number of people per cubic yard

def city_population_9000 := 9000 * P
def city_population_6400 := 6400 * P

theorem number_of_people_per_cubic_yard :
  city_population_9000 - city_population_6400 = 208000 →
  P = 80 :=
by
  sorry

end number_of_people_per_cubic_yard_l1665_166518


namespace pie_shop_earnings_l1665_166575

-- Define the conditions
def price_per_slice : ℕ := 3
def slices_per_pie : ℕ := 10
def number_of_pies : ℕ := 6

-- Calculate the total slices
def total_slices : ℕ := number_of_pies * slices_per_pie

-- Calculate the total earnings
def total_earnings : ℕ := total_slices * price_per_slice

-- State the theorem
theorem pie_shop_earnings : total_earnings = 180 :=
by
  -- Proof can be skipped with a sorry
  sorry

end pie_shop_earnings_l1665_166575


namespace first_quadrant_solution_l1665_166597

theorem first_quadrant_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 2 ∧ c * x + y = 3 ∧ 0 < x ∧ 0 < y) ↔ -1 < c ∧ c < 3 / 2 :=
by
  sorry

end first_quadrant_solution_l1665_166597


namespace twigs_per_branch_l1665_166507

/-- Definitions -/
def total_branches : ℕ := 30
def total_leaves : ℕ := 12690
def percentage_4_leaves : ℝ := 0.30
def leaves_per_twig_4_leaves : ℕ := 4
def percentage_5_leaves : ℝ := 0.70
def leaves_per_twig_5_leaves : ℕ := 5

/-- Given conditions translated to Lean -/
def hypothesis (T : ℕ) : Prop :=
  (percentage_4_leaves * T * leaves_per_twig_4_leaves) +
  (percentage_5_leaves * T * leaves_per_twig_5_leaves) = total_leaves

/-- The main theorem to prove -/
theorem twigs_per_branch
  (T : ℕ)
  (h : hypothesis T) :
  (T / total_branches) = 90 :=
sorry

end twigs_per_branch_l1665_166507


namespace simplify_and_evaluate_l1665_166598

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  ((m ^ 2 - 9) / (m ^ 2 - 6 * m + 9) - 3 / (m - 3)) / (m ^ 2 / (m - 3)) = Real.sqrt 2 / 2 :=
by {
  -- Proof goes here
  sorry
}

end simplify_and_evaluate_l1665_166598
