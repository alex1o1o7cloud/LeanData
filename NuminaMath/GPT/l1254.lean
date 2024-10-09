import Mathlib

namespace max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l1254_125483

noncomputable def y (x : ℝ) (a b : ℝ) : ℝ := (Real.cos x)^2 - a * (Real.sin x) + b

theorem max_min_conditions (a b : ℝ) :
  (∃ x : ℝ, y x a b = 0 ∧ (∀ x' : ℝ, y x' a b ≤ 0)) ∧ 
  (∃ x : ℝ, y x a b = -4 ∧ (∀ x' : ℝ, y x' a b ≥ -4)) ↔ 
  (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = -2) := sorry

theorem x_values_for_max_min_a2 (k : ℤ) :
  (∀ x, y x 2 (-2) = 0 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x 2 (-2)) = -4 ↔ x = Real.pi / 2 + 2 * Real.pi * k) := sorry

theorem x_values_for_max_min_aneg2 (k : ℤ) :
  (∀ x, y x (-2) (-2) = 0 ↔ x = Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x (-2) (-2)) = -4 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) := sorry

end max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l1254_125483


namespace gift_cost_l1254_125400

theorem gift_cost (C F : ℕ) (hF : F = 15) (h_eq : C / (F - 4) = C / F + 12) : C = 495 :=
by
  -- Using the conditions given, we need to show that C computes to 495.
  -- Details are skipped using sorry.
  sorry

end gift_cost_l1254_125400


namespace find_PQ_length_l1254_125442

-- Defining the problem parameters
variables {X Y Z P Q R : Type}
variables (dXY dXZ dPQ dPR : ℝ)
variable (angle_common : ℝ)

-- Conditions:
def angle_XYZ_PQR_common : Prop :=
  angle_common = 150 ∧ 
  dXY = 10 ∧
  dXZ = 20 ∧
  dPQ = 5 ∧
  dPR = 12

-- Question: Prove PQ = 2.5 given the conditions
theorem find_PQ_length
  (h : angle_XYZ_PQR_common dXY dXZ dPQ dPR angle_common) :
  dPQ = 2.5 :=
sorry

end find_PQ_length_l1254_125442


namespace john_needs_29_planks_for_house_wall_l1254_125421

def total_number_of_planks (large_planks small_planks : ℕ) : ℕ :=
  large_planks + small_planks

theorem john_needs_29_planks_for_house_wall :
  total_number_of_planks 12 17 = 29 :=
by
  sorry

end john_needs_29_planks_for_house_wall_l1254_125421


namespace number_of_dodge_trucks_l1254_125453

theorem number_of_dodge_trucks (V T F D : ℕ) (h1 : V = 5)
  (h2 : T = 2 * V) 
  (h3 : F = 2 * T)
  (h4 : F = D / 3) :
  D = 60 := 
by
  sorry

end number_of_dodge_trucks_l1254_125453


namespace range_of_m_l1254_125427

theorem range_of_m (m : ℝ) : (-1 : ℝ) ≤ m ∧ m ≤ 3 ∧ ∀ x y : ℝ, x - ((m^2) - 2 * m + 4) * y - 6 > 0 → (x, y) ≠ (-1, -1) := 
by sorry

end range_of_m_l1254_125427


namespace grasshopper_opposite_corner_moves_l1254_125457

noncomputable def grasshopper_jump_count : ℕ :=
  Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9)

theorem grasshopper_opposite_corner_moves :
  grasshopper_jump_count = Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9) :=
by
  -- The detailed proof would go here.
  sorry

end grasshopper_opposite_corner_moves_l1254_125457


namespace find_M_value_l1254_125436

-- Statements of the problem conditions and the proof goal
theorem find_M_value (a b c M : ℤ) (h1 : a + b + c = 75) (h2 : a + 4 = M) (h3 : b - 5 = M) (h4 : 3 * c = M) : M = 31 := 
by
  sorry

end find_M_value_l1254_125436


namespace inequality_proof_problem_l1254_125432

theorem inequality_proof_problem (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) ≤ 1 / a) :=
sorry

end inequality_proof_problem_l1254_125432


namespace minimum_value_l1254_125437

noncomputable def polynomial_expr (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5

theorem minimum_value : ∃ x y : ℝ, (polynomial_expr x y = 8) := 
sorry

end minimum_value_l1254_125437


namespace number_of_books_is_8_l1254_125408

def books_and_albums (x y p_a p_b : ℕ) : Prop :=
  (x * p_b = 1056) ∧ (p_b = p_a + 100) ∧ (x = y + 6)

theorem number_of_books_is_8 (y p_a p_b : ℕ) (h : books_and_albums 8 y p_a p_b) : 8 = 8 :=
by
  sorry

end number_of_books_is_8_l1254_125408


namespace percentage_of_books_returned_l1254_125412

theorem percentage_of_books_returned
  (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) (returned_books_percentage : ℚ) 
  (h1 : initial_books = 75) 
  (h2 : end_books = 68) 
  (h3 : loaned_books = 20)
  (h4 : returned_books_percentage = (end_books - (initial_books - loaned_books)) * 100 / loaned_books):
  returned_books_percentage = 65 := 
by
  sorry

end percentage_of_books_returned_l1254_125412


namespace net_population_change_l1254_125451

theorem net_population_change (P : ℝ) : 
  let P1 := P * (6/5)
  let P2 := P1 * (7/10)
  let P3 := P2 * (6/5)
  let P4 := P3 * (7/10)
  (P4 / P - 1) * 100 = -29 := 
by
  sorry

end net_population_change_l1254_125451


namespace solve_equation_l1254_125425

theorem solve_equation : ∀ x : ℝ, 2 * x - 6 = 3 * x * (x - 3) ↔ (x = 3 ∨ x = 2 / 3) := by sorry

end solve_equation_l1254_125425


namespace length_AC_l1254_125417
open Real

-- Define the conditions and required proof
theorem length_AC (AB DC AD : ℝ) (h1 : AB = 17) (h2 : DC = 25) (h3 : AD = 8) : 
  abs (sqrt ((AD + DC - AD)^2 + (DC - sqrt (AB^2 - AD^2))^2) - 33.6) < 0.1 := 
  by
  -- The proof is omitted for brevity
  sorry

end length_AC_l1254_125417


namespace chocolate_eggs_weeks_l1254_125448

theorem chocolate_eggs_weeks (e: ℕ) (d: ℕ) (w: ℕ) (total: ℕ) (weeks: ℕ) 
    (initialEggs : e = 40)
    (dailyEggs : d = 2)
    (schoolDays : w = 5)
    (totalWeeks : weeks = total):
    total = e / (d * w) := by
sorry

end chocolate_eggs_weeks_l1254_125448


namespace lowest_score_of_14_scores_l1254_125414

theorem lowest_score_of_14_scores (mean_14 : ℝ) (new_mean_12 : ℝ) (highest_score : ℝ) (lowest_score : ℝ) :
  mean_14 = 85 ∧ new_mean_12 = 88 ∧ highest_score = 105 → lowest_score = 29 :=
by
  sorry

end lowest_score_of_14_scores_l1254_125414


namespace perfect_square_x4_x3_x2_x1_1_eq_x0_l1254_125458

theorem perfect_square_x4_x3_x2_x1_1_eq_x0 :
  ∀ x : ℤ, ∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2 ↔ x = 0 :=
by sorry

end perfect_square_x4_x3_x2_x1_1_eq_x0_l1254_125458


namespace arithmetic_sequence_ratio_l1254_125471

def arithmetic_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio :
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sum1 / sum2 = 1683 / 1300 :=
by {
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sorry
}

end arithmetic_sequence_ratio_l1254_125471


namespace bottles_more_than_apples_l1254_125407

-- Definitions given in the conditions
def apples : ℕ := 36
def regular_soda_bottles : ℕ := 80
def diet_soda_bottles : ℕ := 54

-- Theorem statement representing the question
theorem bottles_more_than_apples : (regular_soda_bottles + diet_soda_bottles) - apples = 98 :=
by
  sorry

end bottles_more_than_apples_l1254_125407


namespace cole_drive_time_l1254_125493

theorem cole_drive_time (D : ℝ) (T_work T_home : ℝ) 
  (h1 : T_work = D / 75) 
  (h2 : T_home = D / 105)
  (h3 : T_work + T_home = 4) : 
  T_work * 60 = 140 := 
by sorry

end cole_drive_time_l1254_125493


namespace slope_AA_l1254_125482

-- Define the points and conditions
variable (a b c d e f : ℝ)

-- Assumptions
#check (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
#check (a ≠ b ∧ c ≠ d ∧ e ≠ f)
#check (a+2 > 0 ∧ b > 0 ∧ c+2 > 0 ∧ d > 0 ∧ e+2 > 0 ∧ f > 0)

-- Main Statement
theorem slope_AA'_not_negative_one
    (H1: a > 0) (H2: b > 0) (H3: c > 0) (H4: d > 0)
    (H5: e > 0) (H6: f > 0) 
    (H7: a ≠ b) (H8: c ≠ d) (H9: e ≠ f)
    (H10: a + 2 > 0) (H11: c + 2 > 0) (H12: e + 2 > 0) : 
    (a ≠ b) → (c ≠ d) → (e ≠ f) → ¬( (a + 2 - b) / (b - a) = -1 ) :=
by
  sorry

end slope_AA_l1254_125482


namespace chocolate_bar_cost_l1254_125416

theorem chocolate_bar_cost 
  (x : ℝ)  -- cost of each bar in dollars
  (total_bars : ℕ)  -- total number of bars in the box
  (sold_bars : ℕ)  -- number of bars sold
  (amount_made : ℝ)  -- amount made in dollars
  (h1 : total_bars = 9)  -- condition: total bars in the box is 9
  (h2 : sold_bars = total_bars - 3)  -- condition: Wendy sold all but 3 bars
  (h3 : amount_made = 18)  -- condition: Wendy made $18
  (h4 : amount_made = sold_bars * x)  -- condition: amount made from selling sold bars
  : x = 3 := 
sorry

end chocolate_bar_cost_l1254_125416


namespace jogging_friends_probability_l1254_125461

theorem jogging_friends_probability
  (n p q r : ℝ)
  (h₀ : 1 > 0) -- Positive integers condition
  (h₁ : n = p - q * Real.sqrt r)
  (h₂ : ∀ prime, ¬ (r ∣ prime ^ 2)) -- r is not divisible by the square of any prime
  (h₃ : (60 - n)^2 = 1800) -- Derived from 50% meeting probability
  (h₄ : p = 60) -- Identified values from solution
  (h₅ : q = 30)
  (h₆ : r = 2) : 
  p + q + r = 92 :=
by
  sorry

end jogging_friends_probability_l1254_125461


namespace smallest_k_for_inequality_l1254_125429

theorem smallest_k_for_inequality : 
  ∃ k : ℕ,  k > 0 ∧ ( (k-10) ^ 5026 ≥ 2013 ^ 2013 ) ∧ 
  (∀ m : ℕ, m > 0 ∧ ((m-10) ^ 5026) ≥ 2013 ^ 2013 → m ≥ 55) :=
sorry

end smallest_k_for_inequality_l1254_125429


namespace q_zero_l1254_125473

noncomputable def q (x : ℝ) : ℝ := sorry -- Definition of the polynomial q(x) is required here.

theorem q_zero : 
  (∀ n : ℕ, n ≤ 7 → q (3^n) = 1 / 3^n) →
  q 0 = 0 :=
by 
  sorry

end q_zero_l1254_125473


namespace prove_angle_BFD_l1254_125474

def given_conditions (A : ℝ) (AFG AGF : ℝ) : Prop :=
  A = 40 ∧ AFG = AGF

theorem prove_angle_BFD (A AFG AGF BFD : ℝ) (h1 : given_conditions A AFG AGF) : BFD = 110 :=
  by
  -- Utilize the conditions h1 stating that A = 40 and AFG = AGF
  sorry

end prove_angle_BFD_l1254_125474


namespace project_presentation_periods_l1254_125419

def students : ℕ := 32
def period_length : ℕ := 40
def presentation_time_per_student : ℕ := 5

theorem project_presentation_periods : 
  (students * presentation_time_per_student) / period_length = 4 := by
  sorry

end project_presentation_periods_l1254_125419


namespace batsman_average_46_innings_l1254_125476

theorem batsman_average_46_innings {hs ls t_44 : ℕ} (h_diff: hs - ls = 180) (h_avg_44: t_44 = 58 * 44) (h_hiscore: hs = 194) : 
  (t_44 + hs + ls) / 46 = 60 := 
sorry

end batsman_average_46_innings_l1254_125476


namespace correct_answer_is_option_d_l1254_125459

def is_quadratic (eq : String) : Prop :=
  eq = "a*x^2 + b*x + c = 0"

def OptionA : String := "1/x^2 + x - 1 = 0"
def OptionB : String := "3x + 1 = 5x + 4"
def OptionC : String := "x^2 + y = 0"
def OptionD : String := "x^2 - 2x + 1 = 0"

theorem correct_answer_is_option_d :
  is_quadratic OptionD :=
by
  sorry

end correct_answer_is_option_d_l1254_125459


namespace tim_prank_combinations_l1254_125415

def number_of_combinations : Nat :=
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations = 60 :=
by
  sorry

end tim_prank_combinations_l1254_125415


namespace range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l1254_125428

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 + a * Real.sin x - Real.cos x ^ 2

theorem range_of_f_when_a_neg_2_is_0_to_4_and_bounded :
  (∀ x : ℝ, 0 ≤ f (-2) x ∧ f (-2) x ≤ 4) :=
sorry

theorem range_of_a_if_f_bounded_by_4 :
  (∀ x : ℝ, abs (f a x) ≤ 4) → (-2 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l1254_125428


namespace range_of_f_l1254_125487

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem range_of_f :
  (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → 0 ≤ f x ∧ f x ≤ 3) := sorry

end range_of_f_l1254_125487


namespace count_perfect_cubes_between_10_and_2000_l1254_125466

theorem count_perfect_cubes_between_10_and_2000 : 
  (∃ n_min n_max, n_min^3 ≥ 10 ∧ n_max^3 ≤ 2000 ∧ 
  (n_max - n_min + 1 = 10)) := 
sorry

end count_perfect_cubes_between_10_and_2000_l1254_125466


namespace num_ordered_pairs_l1254_125463

open Real 

-- Define the conditions
def eq_condition (x y : ℕ) : Prop :=
  x * (sqrt y) + y * (sqrt x) + (sqrt (2006 * x * y)) - (sqrt (2006 * x)) - (sqrt (2006 * y)) - 2006 = 0

-- Define the main problem statement
theorem num_ordered_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (x y : ℕ), eq_condition x y → x * y = 2006) :=
by
  sorry

end num_ordered_pairs_l1254_125463


namespace solution_l1254_125423

noncomputable def problem (x : ℝ) (h : x ≠ 3) : ℝ :=
  (3 * x / (x - 3)) + ((x + 6) / (3 - x))

theorem solution (x : ℝ) (h : x ≠ 3) : problem x h = 2 :=
by
  sorry

end solution_l1254_125423


namespace ratio_of_radii_of_circles_l1254_125443

theorem ratio_of_radii_of_circles 
  (a b : ℝ) 
  (h1 : a = 6) 
  (h2 : b = 8) 
  (h3 : ∃ (c : ℝ), c = Real.sqrt (a^2 + b^2)) 
  (h4 : ∃ (r R : ℝ), R = c / 2 ∧ r = 24 / (a + b + c)) : R / r = 5 / 2 :=
by
  sorry

end ratio_of_radii_of_circles_l1254_125443


namespace collinear_vectors_parallel_right_angle_triangle_abc_l1254_125405

def vec_ab (k : ℝ) : ℝ × ℝ := (2 - k, -1)
def vec_ac (k : ℝ) : ℝ × ℝ := (1, k)

-- Prove that if vectors AB and AC are collinear, then k = 1 ± √2
theorem collinear_vectors_parallel (k : ℝ) :
  (2 - k) * k - 1 = 0 ↔ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2 :=
by
  sorry

def vec_bc (k : ℝ) : ℝ × ℝ := (k - 1, k + 1)

-- Prove that if triangle ABC is right-angled, then k = 1 or k = -1 ± √2
theorem right_angle_triangle_abc (k : ℝ) :
  ( (2 - k) * 1 + (-1) * k = 0 ∨ (k - 1) * 1 + (k + 1) * k = 0 ) ↔ 
  k = 1 ∨ k = -1 + Real.sqrt 2 ∨ k = -1 - Real.sqrt 2 :=
by
  sorry

end collinear_vectors_parallel_right_angle_triangle_abc_l1254_125405


namespace find_x_l1254_125413

variable {a b x : ℝ}
variable (h₁ : b ≠ 0)
variable (h₂ : (3 * a) ^ (2 * b) = a ^ b * x ^ b)

theorem find_x (h₁ : b ≠ 0) (h₂ : (3 * a) ^ (2 * b) = a ^ b * x ^ b) : 
  x = 9 * a :=
by
  sorry

end find_x_l1254_125413


namespace inequalities_not_hold_range_a_l1254_125456

theorem inequalities_not_hold_range_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 - a * x + 1 ≤ 0) ∧ (¬ ∀ x : ℝ, a * x^2 + x - 1 > 0) ↔ (-2 < a ∧ a ≤ -1 / 4) :=
by
  sorry

end inequalities_not_hold_range_a_l1254_125456


namespace six_positive_integers_solution_count_l1254_125498

theorem six_positive_integers_solution_count :
  ∃ (S : Finset (Finset ℕ)) (n : ℕ) (a b c x y z : ℕ), 
  a ≥ b → b ≥ c → x ≥ y → y ≥ z → 
  a + b + c = x * y * z → 
  x + y + z = a * b * c → 
  S.card = 7 := by
    sorry

end six_positive_integers_solution_count_l1254_125498


namespace three_digit_numbers_last_three_digits_of_square_l1254_125465

theorem three_digit_numbers_last_three_digits_of_square (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n^2 % 1000) = n ↔ n = 376 ∨ n = 625 := 
sorry

end three_digit_numbers_last_three_digits_of_square_l1254_125465


namespace part_I_part_II_l1254_125452

variable (a b c : ℝ)

theorem part_I (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : a + b + c = 4 :=
sorry

theorem part_II (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8/7 :=
sorry

end part_I_part_II_l1254_125452


namespace find_number_l1254_125422

theorem find_number (N M : ℕ) 
  (h1 : N + M = 3333) (h2 : N - M = 693) :
  N = 2013 :=
sorry

end find_number_l1254_125422


namespace ad_eb_intersect_on_altitude_l1254_125477

open EuclideanGeometry

variables {A B C D E F G K L C1 : Point}

-- Definitions for the problem
variables (triangleABC : Triangle A B C)
  (squareAEFC : Square A E F C)
  (squareBDGC : Square B D G C)
  (altitudeCC1 : Line C C1)
  (lineDA : Line A D)
  (lineEB : Line B E)

-- Definition of intersection
def intersects_on_altitude (pt : Point) : Prop :=
  pt ∈ lineDA ∧ pt ∈ lineEB ∧ pt ∈ altitudeCC1

-- The theorem to be proved
theorem ad_eb_intersect_on_altitude : 
  ∃ pt : Point, intersects_on_altitude lineDA lineEB altitudeCC1 pt := 
sorry

end ad_eb_intersect_on_altitude_l1254_125477


namespace adi_baller_prob_l1254_125404

theorem adi_baller_prob (a b : ℕ) (p : ℝ) (h_prime: Nat.Prime a) (h_pos_b: 0 < b)
  (h_p: p = (1 / 2) ^ (1 / 35)) : a + b = 37 :=
sorry

end adi_baller_prob_l1254_125404


namespace continuity_at_2_l1254_125497

theorem continuity_at_2 (f : ℝ → ℝ) (x0 : ℝ) (hf : ∀ x, f x = -4 * x ^ 2 - 8) :
  x0 = 2 → ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x0| < δ → |f x + 24| < ε := by
  sorry

end continuity_at_2_l1254_125497


namespace polar_equation_C1_intersection_C2_C1_distance_l1254_125409

noncomputable def parametric_to_cartesian (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 + 2 * Real.cos α ∧ y = 4 + 2 * Real.sin α

noncomputable def cartesian_to_polar (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = 4

noncomputable def polar_equation_of_C1 (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 16 = 0

noncomputable def C2_line_polar (θ : ℝ) : Prop :=
  θ = Real.pi / 4

theorem polar_equation_C1 (α : ℝ) (ρ θ : ℝ) :
  parametric_to_cartesian α →
  cartesian_to_polar (2 + 2 * Real.cos α) (4 + 2 * Real.sin α) →
  polar_equation_of_C1 ρ θ :=
by
  sorry

theorem intersection_C2_C1_distance (ρ θ : ℝ) (t1 t2 : ℝ) :
  C2_line_polar θ →
  polar_equation_of_C1 ρ θ →
  (t1 + t2 = 6 * Real.sqrt 2) ∧ (t1 * t2 = 16) →
  |t1 - t2| = 2 * Real.sqrt 2 :=
by
  sorry

end polar_equation_C1_intersection_C2_C1_distance_l1254_125409


namespace perfect_square_trinomial_l1254_125435

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x + a)^2) ∨ (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x - a)^2)) ↔ m = 5 ∨ m = -3 :=
sorry

end perfect_square_trinomial_l1254_125435


namespace roller_coaster_people_l1254_125444

def num_cars : ℕ := 7
def seats_per_car : ℕ := 2
def num_runs : ℕ := 6
def total_seats_per_run : ℕ := num_cars * seats_per_car
def total_people : ℕ := total_seats_per_run * num_runs

theorem roller_coaster_people:
  total_people = 84 := 
by
  sorry

end roller_coaster_people_l1254_125444


namespace simplify_expression_l1254_125455

variable (a b : ℤ)

theorem simplify_expression : 
  (15 * a + 45 * b) + (21 * a + 32 * b) - (12 * a + 40 * b) = 24 * a + 37 * b := 
    by sorry

end simplify_expression_l1254_125455


namespace total_bushels_needed_l1254_125401

def cows := 5
def sheep := 4
def chickens := 8
def pigs := 6
def horses := 2

def cow_bushels := 3.5
def sheep_bushels := 1.75
def chicken_bushels := 1.25
def pig_bushels := 4.5
def horse_bushels := 5.75

theorem total_bushels_needed
  (cows : ℕ) (sheep : ℕ) (chickens : ℕ) (pigs : ℕ) (horses : ℕ)
  (cow_bushels: ℝ) (sheep_bushels: ℝ) (chicken_bushels: ℝ) (pig_bushels: ℝ) (horse_bushels: ℝ) :
  cows * cow_bushels + sheep * sheep_bushels + chickens * chicken_bushels + pigs * pig_bushels + horses * horse_bushels = 73 :=
by
  -- Skipping the proof
  sorry

end total_bushels_needed_l1254_125401


namespace values_of_cos_0_45_l1254_125402

-- Define the interval and the condition for the cos function
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 360
def cos_condition (x : ℝ) : Prop := Real.cos x = 0.45

-- Final theorem statement
theorem values_of_cos_0_45 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), in_interval x ∧ cos_condition x ↔ x = 1 ∨ x = 2 := 
sorry

end values_of_cos_0_45_l1254_125402


namespace coin_flip_probability_l1254_125481

open Classical

noncomputable section

theorem coin_flip_probability :
  let total_outcomes := 2^10
  let exactly_five_heads_tails := Nat.choose 10 5 / total_outcomes
  let even_heads_probability := 1/2
  (even_heads_probability * (1 - exactly_five_heads_tails) / 2 = 193 / 512) :=
by
  sorry

end coin_flip_probability_l1254_125481


namespace largest_number_divisible_by_48_is_9984_l1254_125454

def largest_divisible_by_48 (n : ℕ) := ∀ m ≥ n, m % 48 = 0 → m ≤ 9999

theorem largest_number_divisible_by_48_is_9984 :
  largest_divisible_by_48 9984 ∧ 9999 / 10^3 = 9 ∧ 48 ∣ 9984 ∧ 9984 < 10000 :=
by
  sorry

end largest_number_divisible_by_48_is_9984_l1254_125454


namespace find_k_l1254_125446

theorem find_k : 
  (∃ y, -3 * (-15) + y = k ∧ 0.3 * (-15) + y = 10) → k = 59.5 :=
by
  sorry

end find_k_l1254_125446


namespace scientific_notation_periodicals_l1254_125462

theorem scientific_notation_periodicals :
  (56000000 : ℝ) = 5.6 * 10^7 := by
sorry

end scientific_notation_periodicals_l1254_125462


namespace multiplication_counts_l1254_125489

open Polynomial

noncomputable def horner_multiplications (n : ℕ) : ℕ := n

noncomputable def direct_summation_multiplications (n : ℕ) : ℕ := n * (n + 1) / 2

theorem multiplication_counts (P : Polynomial ℝ) (x₀ : ℝ) (n : ℕ)
  (h_degree : P.degree = n) :
  horner_multiplications n = n ∧ direct_summation_multiplications n = (n * (n + 1)) / 2 :=
by
  sorry

end multiplication_counts_l1254_125489


namespace average_infection_rate_l1254_125488

theorem average_infection_rate (x : ℕ) : 
  1 + x + x * (1 + x) = 81 :=
sorry

end average_infection_rate_l1254_125488


namespace woman_work_rate_l1254_125491

theorem woman_work_rate (W : ℝ) :
  (1 / 6) + W + (1 / 9) = (1 / 3) → W = (1 / 18) :=
by
  intro h
  sorry

end woman_work_rate_l1254_125491


namespace eval_expression_l1254_125460

theorem eval_expression : (500 * 500) - (499 * 501) = 1 := by
  sorry

end eval_expression_l1254_125460


namespace maria_towels_l1254_125469

theorem maria_towels (green_towels white_towels given_towels : ℕ) (bought_green : green_towels = 40) 
(bought_white : white_towels = 44) (gave_mother : given_towels = 65) : 
  green_towels + white_towels - given_towels = 19 := by
sorry

end maria_towels_l1254_125469


namespace diamond_evaluation_l1254_125495

def diamond (X Y : ℚ) : ℚ := (2 * X + 3 * Y) / 5

theorem diamond_evaluation : diamond (diamond 3 15) 6 = 192 / 25 := 
by
  sorry

end diamond_evaluation_l1254_125495


namespace simplify_expression_is_one_fourth_l1254_125478

noncomputable def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
noncomputable def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
noncomputable def simplified_expression : ℝ := (fourth_root 81 - square_root 12.25) ^ 2

theorem simplify_expression_is_one_fourth : simplified_expression = 1 / 4 := 
by
  sorry

end simplify_expression_is_one_fourth_l1254_125478


namespace bronson_cost_per_bushel_is_12_l1254_125410

noncomputable def cost_per_bushel 
  (sale_price_per_apple : ℝ := 0.40)
  (apples_per_bushel : ℕ := 48)
  (profit_from_100_apples : ℝ := 15)
  (number_of_apples_sold : ℕ := 100) 
  : ℝ :=
  let revenue := number_of_apples_sold * sale_price_per_apple
  let cost := revenue - profit_from_100_apples
  let number_of_bushels := (number_of_apples_sold : ℝ) / apples_per_bushel
  cost / number_of_bushels

theorem bronson_cost_per_bushel_is_12 :
  cost_per_bushel = 12 :=
by
  sorry

end bronson_cost_per_bushel_is_12_l1254_125410


namespace sequence_constant_l1254_125486

theorem sequence_constant
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : ∀ n, Nat.Prime (Int.natAbs (a n)))
  (h2 : ∀ n, a (n + 2) = a (n + 1) + a n + d) :
  ∃ c : ℤ, ∀ n, a n = c :=
by
  sorry

end sequence_constant_l1254_125486


namespace color_plane_with_two_colors_l1254_125490

/-- Given a finite set of circles that divides the plane into regions, we can color the plane such that no two adjacent regions have the same color. -/
theorem color_plane_with_two_colors (circles : Finset (Set ℝ)) :
  (∀ (r1 r2 : Set ℝ), (r1 ∩ r2).Nonempty → ∃ (coloring : Set ℝ → Bool), (coloring r1 ≠ coloring r2)) :=
  sorry

end color_plane_with_two_colors_l1254_125490


namespace quadratic_solution_range_l1254_125496

theorem quadratic_solution_range (t : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x - t = 0 ∧ -1 < x ∧ x < 4) ↔ (-1 ≤ t ∧ t < 8) := 
sorry

end quadratic_solution_range_l1254_125496


namespace sin_theta_correct_l1254_125450

noncomputable def sin_theta (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) : Real :=
  -Real.sqrt 2 / 2

theorem sin_theta_correct (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan (Real.arctan (-a)) = -a) : sin_theta a h1 h2 = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_theta_correct_l1254_125450


namespace find_m_l1254_125431

variable (a : ℝ) (m : ℝ)

theorem find_m (h : a^(m + 1) * a^(2 * m - 1) = a^9) : m = 3 := 
by
  sorry

end find_m_l1254_125431


namespace series_sum_eq_five_l1254_125430

open Nat Real

noncomputable def sum_series : ℝ := ∑' (n : ℕ), (2 * n ^ 2 - n) / (n * (n + 1) * (n + 2))

theorem series_sum_eq_five : sum_series = 5 :=
sorry

end series_sum_eq_five_l1254_125430


namespace average_children_in_families_with_children_l1254_125424

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l1254_125424


namespace supplementary_angles_difference_l1254_125441

theorem supplementary_angles_difference 
  (x : ℝ) 
  (h1 : 5 * x + 3 * x = 180) 
  (h2 : 0 < x) : 
  abs (5 * x - 3 * x) = 45 :=
by sorry

end supplementary_angles_difference_l1254_125441


namespace each_friend_eats_six_slices_l1254_125447

-- Definitions
def slices_per_loaf : ℕ := 15
def loaves_bought : ℕ := 4
def friends : ℕ := 10
def total_slices : ℕ := loaves_bought * slices_per_loaf
def slices_per_friend : ℕ := total_slices / friends

-- Theorem to prove
theorem each_friend_eats_six_slices (h1 : slices_per_loaf = 15) (h2 : loaves_bought = 4) (h3 : friends = 10) : slices_per_friend = 6 :=
by
  sorry

end each_friend_eats_six_slices_l1254_125447


namespace meaningful_sqrt_range_l1254_125440

theorem meaningful_sqrt_range (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
sorry

end meaningful_sqrt_range_l1254_125440


namespace probability_of_black_ball_l1254_125470

theorem probability_of_black_ball 
  (p_red : ℝ)
  (p_white : ℝ)
  (h_red : p_red = 0.43)
  (h_white : p_white = 0.27)
  : (1 - p_red - p_white) = 0.3 :=
by 
  sorry

end probability_of_black_ball_l1254_125470


namespace least_five_digit_congruent_to_7_mod_18_l1254_125418

theorem least_five_digit_congruent_to_7_mod_18 :
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < n → m % 18 ≠ 7 :=
by
  sorry

end least_five_digit_congruent_to_7_mod_18_l1254_125418


namespace polynomial_expansion_l1254_125403

-- Define the polynomial expressions
def poly1 (s : ℝ) : ℝ := 3 * s^3 - 4 * s^2 + 5 * s - 2
def poly2 (s : ℝ) : ℝ := 2 * s^2 - 3 * s + 4

-- Define the expanded form of the product of the two polynomials
def expanded_poly (s : ℝ) : ℝ :=
  6 * s^5 - 17 * s^4 + 34 * s^3 - 35 * s^2 + 26 * s - 8

-- The theorem to prove the equivalence
theorem polynomial_expansion (s : ℝ) :
  (poly1 s) * (poly2 s) = expanded_poly s :=
sorry -- proof goes here

end polynomial_expansion_l1254_125403


namespace hyeoncheol_initial_money_l1254_125484

theorem hyeoncheol_initial_money
  (X : ℕ)
  (h1 : X / 2 / 2 = 1250) :
  X = 5000 :=
sorry

end hyeoncheol_initial_money_l1254_125484


namespace minimum_cost_for_18_oranges_l1254_125475

noncomputable def min_cost_oranges (x y : ℕ) : ℕ :=
  10 * x + 30 * y

theorem minimum_cost_for_18_oranges :
  (∃ x y : ℕ, 3 * x + 7 * y = 18 ∧ min_cost_oranges x y = 60) ∧ (60 / 18 = 10 / 3) :=
sorry

end minimum_cost_for_18_oranges_l1254_125475


namespace max_ladder_height_reached_l1254_125434

def distance_from_truck_to_building : ℕ := 5
def ladder_extension : ℕ := 13

theorem max_ladder_height_reached :
  (ladder_extension ^ 2 - distance_from_truck_to_building ^ 2) = 144 :=
by
  -- This is where the proof should go
  sorry

end max_ladder_height_reached_l1254_125434


namespace evening_minivans_l1254_125472

theorem evening_minivans (total_minivans afternoon_minivans : ℕ) (h_total : total_minivans = 5) 
(h_afternoon : afternoon_minivans = 4) : total_minivans - afternoon_minivans = 1 := 
by
  sorry

end evening_minivans_l1254_125472


namespace find_k_value_l1254_125468

variable {a : ℕ → ℕ} {S : ℕ → ℕ} 

axiom sum_of_first_n_terms (n : ℕ) (hn : n > 0) : S n = a n / n
axiom exists_Sk_inequality (k : ℕ) (hk : k > 0) : 1 < S k ∧ S k < 9

theorem find_k_value 
  (k : ℕ) (hk : k > 0) (hS : S k = a k / k) (hSk : 1 < S k ∧ S k < 9)
  (h_cond : ∀ n > 0, S n = n * S n ∧ S (n - 1) = S n * (n - 1)) : 
  k = 4 :=
sorry

end find_k_value_l1254_125468


namespace range_of_a_for_false_proposition_l1254_125464

theorem range_of_a_for_false_proposition :
  ∀ a : ℝ, (¬ ∃ x : ℝ, a * x ^ 2 + a * x + 1 ≤ 0) ↔ (0 ≤ a ∧ a < 4) :=
by sorry

end range_of_a_for_false_proposition_l1254_125464


namespace correct_calculation_l1254_125494

variable (a : ℝ)

theorem correct_calculation (a : ℝ) : (2 * a)^2 / (4 * a) = a := by
  sorry

end correct_calculation_l1254_125494


namespace correct_answer_l1254_125467

variables (A B : polynomial ℝ) (a : ℝ)

theorem correct_answer (hB : B = 3 * a^2 - 5 * a - 7) (hMistake : A - 2 * B = -2 * a^2 + 3 * a + 6) :
  A + 2 * B = 10 * a^2 - 17 * a - 22 :=
by
  sorry

end correct_answer_l1254_125467


namespace quadratic_distinct_roots_l1254_125406

theorem quadratic_distinct_roots (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ x = 2 * p ∨ x = p + q) →
  (p = 2 / 3 ∧ q = -8 / 3) :=
by sorry

end quadratic_distinct_roots_l1254_125406


namespace rational_iff_geometric_progression_l1254_125439

theorem rational_iff_geometric_progression :
  (∃ x a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + a)*(x + c) = (x + b)^2) ↔
  (∃ x : ℚ, ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + (a : ℚ))*(x + (c : ℚ)) = (x + (b : ℚ))^2) :=
sorry

end rational_iff_geometric_progression_l1254_125439


namespace triangle_area_l1254_125438

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 180 :=
by
  sorry

end triangle_area_l1254_125438


namespace option_A_equal_l1254_125480

theorem option_A_equal : (-2: ℤ)^(3: ℕ) = ((-2: ℤ)^(3: ℕ)) :=
by
  sorry

end option_A_equal_l1254_125480


namespace n_is_prime_l1254_125485

variable {n : ℕ}

theorem n_is_prime (hn : n > 1) (hd : ∀ d : ℕ, d > 0 ∧ d ∣ n → d + 1 ∣ n + 1) :
  Prime n := 
sorry

end n_is_prime_l1254_125485


namespace powers_of_two_diff_div_by_1987_l1254_125492

theorem powers_of_two_diff_div_by_1987 :
  ∃ a b : ℕ, a > b ∧ 1987 ∣ (2^a - 2^b) :=
by sorry

end powers_of_two_diff_div_by_1987_l1254_125492


namespace percentage_increase_l1254_125449

theorem percentage_increase
  (initial_earnings new_earnings : ℝ)
  (h_initial : initial_earnings = 55)
  (h_new : new_earnings = 60) :
  ((new_earnings - initial_earnings) / initial_earnings * 100) = 9.09 :=
by
  sorry

end percentage_increase_l1254_125449


namespace units_digit_of_sum_is_4_l1254_125445

-- Definitions and conditions based on problem
def base_8_add (a b : List Nat) : List Nat :=
    sorry -- Function to perform addition in base 8, returning result as a list of digits

def units_digit (a : List Nat) : Nat :=
    a.headD 0  -- Function to get the units digit of the result

-- The list representation for the digits of 65 base 8 and 37 base 8
def sixty_five_base8 := [6, 5]
def thirty_seven_base8 := [3, 7]

-- The theorem that asserts the final result
theorem units_digit_of_sum_is_4 : units_digit (base_8_add sixty_five_base8 thirty_seven_base8) = 4 :=
    sorry

end units_digit_of_sum_is_4_l1254_125445


namespace distance_between_centers_same_side_distance_between_centers_opposite_side_l1254_125411

open Real

noncomputable def distance_centers_same_side (r : ℝ) : ℝ := (r * (sqrt 6 + sqrt 2)) / 2

noncomputable def distance_centers_opposite_side (r : ℝ) : ℝ := (r * (sqrt 6 - sqrt 2)) / 2

theorem distance_between_centers_same_side (r : ℝ):
  ∃ dist, dist = distance_centers_same_side r :=
sorry

theorem distance_between_centers_opposite_side (r : ℝ):
  ∃ dist, dist = distance_centers_opposite_side r :=
sorry

end distance_between_centers_same_side_distance_between_centers_opposite_side_l1254_125411


namespace range_of_m_l1254_125499

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0)
  (h_equation : (2 / x) + (1 / y) = 1 / 3)
  (h_inequality : x + 2 * y > m^2 - 2 * m) : 
  -4 < m ∧ m < 6 := 
sorry

end range_of_m_l1254_125499


namespace Isaiah_types_more_l1254_125479

theorem Isaiah_types_more (Micah_rate Isaiah_rate : ℕ) (h_Micah : Micah_rate = 20) (h_Isaiah : Isaiah_rate = 40) :
  (Isaiah_rate * 60 - Micah_rate * 60) = 1200 :=
by
  -- Here we assume we need to prove this theorem
  sorry

end Isaiah_types_more_l1254_125479


namespace trigonometric_identity_l1254_125420

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 := 
by
  sorry

end trigonometric_identity_l1254_125420


namespace geometric_sequence_a10_a11_l1254_125433

noncomputable def a (n : ℕ) : ℝ := sorry  -- define the geometric sequence {a_n}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q^m

variables (a : ℕ → ℝ) (q : ℝ)

-- Conditions given in the problem
axiom h1 : a 1 + a 5 = 5
axiom h2 : a 4 + a 5 = 15
axiom geom_seq : is_geometric_sequence a q

theorem geometric_sequence_a10_a11 : a 10 + a 11 = 135 :=
by {
  sorry
}

end geometric_sequence_a10_a11_l1254_125433


namespace find_b_l1254_125426

theorem find_b (a b : ℝ) (h1 : 2 * a + b = 6) (h2 : -2 * a + b = 2) : b = 4 :=
sorry

end find_b_l1254_125426
