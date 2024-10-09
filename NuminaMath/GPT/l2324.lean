import Mathlib

namespace min_segments_of_polyline_l2324_232410

theorem min_segments_of_polyline (n : ℕ) (h : n ≥ 2) : 
  ∃ s : ℕ, s = 2 * n - 2 := sorry

end min_segments_of_polyline_l2324_232410


namespace cos_periodicity_even_function_property_l2324_232492

theorem cos_periodicity_even_function_property (n : ℤ) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) (h_range : -180 ≤ n ∧ n ≤ 180) : n = 43 :=
by
  sorry

end cos_periodicity_even_function_property_l2324_232492


namespace find_nonzero_q_for_quadratic_l2324_232416

theorem find_nonzero_q_for_quadratic :
  ∃ (q : ℝ), q ≠ 0 ∧ (∀ (x1 x2 : ℝ), (q * x1^2 - 8 * x1 + 2 = 0 ∧ q * x2^2 - 8 * x2 + 2 = 0) → x1 = x2) ↔ q = 8 :=
by
  sorry

end find_nonzero_q_for_quadratic_l2324_232416


namespace complex_multiplication_example_l2324_232475

def imaginary_unit (i : ℂ) : Prop := i^2 = -1

theorem complex_multiplication_example (i : ℂ) (h : imaginary_unit i) :
  (3 + i) * (1 - 2 * i) = 5 - 5 * i := 
by
  sorry

end complex_multiplication_example_l2324_232475


namespace max_volume_small_cube_l2324_232487

theorem max_volume_small_cube (a : ℝ) (h : a = 2) : (a^3 = 8) := by
  sorry

end max_volume_small_cube_l2324_232487


namespace parabola_focus_coordinates_l2324_232426

theorem parabola_focus_coordinates (x y : ℝ) (h : x = 2 * y^2) : (x, y) = (1/8, 0) :=
sorry

end parabola_focus_coordinates_l2324_232426


namespace correct_mark_l2324_232423

theorem correct_mark (x : ℕ) (S_Correct S_Wrong : ℕ) (n : ℕ) :
  n = 26 →
  S_Wrong = S_Correct + (83 - x) →
  (S_Wrong : ℚ) / n = (S_Correct : ℚ) / n + 1 / 2 →
  x = 70 :=
by
  intros h1 h2 h3
  sorry

end correct_mark_l2324_232423


namespace power_binary_representation_zero_digit_l2324_232407

theorem power_binary_representation_zero_digit
  (a n s : ℕ) (ha : a > 1) (hn : n > 1) (hs : s > 0) :
  a ^ n ≠ 2 ^ s - 1 :=
by
  sorry

end power_binary_representation_zero_digit_l2324_232407


namespace factorization_of_difference_of_squares_l2324_232469

theorem factorization_of_difference_of_squares (m n : ℝ) : m^2 - n^2 = (m + n) * (m - n) := 
by sorry

end factorization_of_difference_of_squares_l2324_232469


namespace product_first_8_terms_l2324_232427

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a_2 : a 2 = 3 := sorry
def a_7 : a 7 = 1 := sorry

-- Proof statement
theorem product_first_8_terms (h_geom : is_geometric_sequence a q) 
  (h_a2 : a 2 = 3) 
  (h_a7 : a 7 = 1) : 
  (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 = 81) :=
sorry

end product_first_8_terms_l2324_232427


namespace find_x2_plus_y2_l2324_232477

-- Given conditions as definitions in Lean
variable {x y : ℝ}
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : x * y + x + y = 71)
variable (h4 : x^2 * y + x * y^2 = 880)

-- The statement to be proved
theorem find_x2_plus_y2 : x^2 + y^2 = 146 :=
by
  sorry

end find_x2_plus_y2_l2324_232477


namespace binary_to_decimal_l2324_232433

theorem binary_to_decimal : (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end binary_to_decimal_l2324_232433


namespace interval_representation_l2324_232434

def S : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem interval_representation : S = Set.Ioc (-1) 3 :=
sorry

end interval_representation_l2324_232434


namespace fraction_decomposition_l2324_232486

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ -1 ∧ x ≠ 2  →
    7 * x - 18 = A * (3 * x + 1) + B * (x - 2))
  ↔ (A = -4 / 7 ∧ B = 61 / 7) :=
by
  sorry

end fraction_decomposition_l2324_232486


namespace stacy_savings_for_3_pairs_l2324_232417

-- Define the cost per pair of shorts
def cost_per_pair : ℕ := 10

-- Define the discount percentage as a decimal
def discount_percentage : ℝ := 0.1

-- Function to calculate the total cost without discount for n pairs
def total_cost_without_discount (n : ℕ) : ℕ := cost_per_pair * n

-- Function to calculate the total cost with discount for n pairs
noncomputable def total_cost_with_discount (n : ℕ) : ℝ :=
  if n >= 3 then
    let discount := discount_percentage * (cost_per_pair * n : ℝ)
    (cost_per_pair * n : ℝ) - discount
  else
    cost_per_pair * n

-- Function to calculate the savings for buying n pairs at once compared to individually
noncomputable def savings (n : ℕ) : ℝ :=
  (total_cost_without_discount n : ℝ) - total_cost_with_discount n

-- Proof statement
theorem stacy_savings_for_3_pairs : savings 3 = 3 := by
  sorry

end stacy_savings_for_3_pairs_l2324_232417


namespace fraction_to_decimal_l2324_232406

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l2324_232406


namespace hyperbola_eccentricity_l2324_232455

theorem hyperbola_eccentricity 
  (p1 p2 : ℝ × ℝ)
  (asymptote_passes_through_p1 : p1 = (1, 2))
  (hyperbola_passes_through_p2 : p2 = (2 * Real.sqrt 2, 4)) :
  ∃ e : ℝ, e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l2324_232455


namespace determine_values_of_x_l2324_232404

variable (x : ℝ)

theorem determine_values_of_x (h1 : 1/x < 3) (h2 : 1/x > -4) : x > 1/3 ∨ x < -1/4 := 
  sorry


end determine_values_of_x_l2324_232404


namespace avg_A_lt_avg_B_combined_avg_eq_6_6_l2324_232464

-- Define the scores for A and B
def scores_A := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

-- Define the average score function
def average (scores : List ℚ) : ℚ := (scores.sum : ℚ) / scores.length

-- Define the mean for the combined data
def combined_average : ℚ :=
  (average scores_A * scores_A.length + average scores_B * scores_B.length) / 
  (scores_A.length + scores_B.length)

-- Specify the variances given in the problem
def variance_A := 2.25
def variance_B := 4.41

-- Claim the average score of A is smaller than the average score of B
theorem avg_A_lt_avg_B : average scores_A < average scores_B := by sorry

-- Claim the average score of these 20 data points is 6.6
theorem combined_avg_eq_6_6 : combined_average = 6.6 := by sorry

end avg_A_lt_avg_B_combined_avg_eq_6_6_l2324_232464


namespace middle_digit_is_zero_l2324_232481

noncomputable def N_in_base8 (a b c : ℕ) : ℕ := 512 * a + 64 * b + 8 * c
noncomputable def N_in_base10 (a b c : ℕ) : ℕ := 100 * b + 10 * c + a

theorem middle_digit_is_zero (a b c : ℕ) (h : N_in_base8 a b c = N_in_base10 a b c) :
  b = 0 :=
by 
  sorry

end middle_digit_is_zero_l2324_232481


namespace ratio_rate_down_to_up_l2324_232436

theorem ratio_rate_down_to_up 
  (rate_up : ℝ) (time_up : ℝ) (distance_down : ℝ) (time_down_eq_time_up : time_down = time_up) :
  (time_up = 2) → 
  (rate_up = 3) →
  (distance_down = 9) → 
  (time_down = time_up) →
  (distance_down / time_down / rate_up = 1.5) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_rate_down_to_up_l2324_232436


namespace reading_time_equal_l2324_232457

/--
  Alice, Bob, and Chandra are reading a 760-page book. Alice reads a page in 20 seconds, 
  Bob reads a page in 45 seconds, and Chandra reads a page in 30 seconds. Prove that if 
  they divide the book into three sections such that each reads for the same length of 
  time, then each person will read for 7200 seconds.
-/
theorem reading_time_equal 
  (rate_A : ℝ := 1/20) 
  (rate_B : ℝ := 1/45) 
  (rate_C : ℝ := 1/30) 
  (total_pages : ℝ := 760) : 
  ∃ t : ℝ, t = 7200 ∧ 
    (t * rate_A + t * rate_B + t * rate_C = total_pages) := 
by
  sorry  -- proof to be provided

end reading_time_equal_l2324_232457


namespace probability_triangle_or_circle_l2324_232478

theorem probability_triangle_or_circle (total_figures triangles circles : ℕ) 
  (h1 : total_figures = 10) 
  (h2 : triangles = 4) 
  (h3 : circles = 3) : 
  (triangles + circles) / total_figures = 7 / 10 :=
by
  sorry

end probability_triangle_or_circle_l2324_232478


namespace gcd_lcm_product_eq_abc_l2324_232443

theorem gcd_lcm_product_eq_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  let D := Nat.gcd (Nat.gcd a b) c
  let m := Nat.lcm (Nat.lcm a b) c
  D * m = a * b * c :=
by
  sorry

end gcd_lcm_product_eq_abc_l2324_232443


namespace matrix_power_minus_l2324_232422

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![3, 4],
    ![0, 2]
  ]

theorem matrix_power_minus :
  B^15 - 3 • B^14 = ![
    ![0, 8192],
    ![0, -8192]
  ] :=
by
  sorry

end matrix_power_minus_l2324_232422


namespace expected_value_of_die_is_475_l2324_232462

-- Define the given probabilities
def prob_1 : ℚ := 1 / 12
def prob_2 : ℚ := 1 / 12
def prob_3 : ℚ := 1 / 6
def prob_4 : ℚ := 1 / 12
def prob_5 : ℚ := 1 / 12
def prob_6 : ℚ := 7 / 12

-- Define the expected value calculation
def expected_value := 
  prob_1 * 1 + prob_2 * 2 + prob_3 * 3 +
  prob_4 * 4 + prob_5 * 5 + prob_6 * 6

-- The problem statement to prove
theorem expected_value_of_die_is_475 : expected_value = 4.75 := by
  sorry

end expected_value_of_die_is_475_l2324_232462


namespace total_amount_shared_l2324_232468

theorem total_amount_shared (jane mike nora total : ℝ) 
  (h1 : jane = 30) 
  (h2 : jane / 2 = mike / 3) 
  (h3 : mike / 3 = nora / 8) 
  (h4 : total = jane + mike + nora) : 
  total = 195 :=
by
  sorry

end total_amount_shared_l2324_232468


namespace probability_neither_red_nor_purple_l2324_232420

theorem probability_neither_red_nor_purple (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) : 
  total_balls = 60 →
  white_balls = 22 →
  green_balls = 18 →
  yellow_balls = 2 →
  red_balls = 15 →
  purple_balls = 3 →
  (total_balls - red_balls - purple_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end probability_neither_red_nor_purple_l2324_232420


namespace find_range_of_m_l2324_232471

def proposition_p (m : ℝ) : Prop := 0 < m ∧ m < 1/3
def proposition_q (m : ℝ) : Prop := 0 < m ∧ m < 15
def proposition_r (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def proposition_s (m : ℝ) : Prop := proposition_p m ∧ proposition_q m = False
def range_of_m (m : ℝ) : Prop := 1/3 ≤ m ∧ m < 15

theorem find_range_of_m (m : ℝ) : proposition_r m ∧ proposition_s m → range_of_m m := by
  sorry

end find_range_of_m_l2324_232471


namespace arithmetic_sequence_general_formula_l2324_232421

theorem arithmetic_sequence_general_formula
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h2 : a 4 - a 2 = 4)
  (h3 : S 3 = 9)
  : ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end arithmetic_sequence_general_formula_l2324_232421


namespace gcd_of_n13_minus_n_l2324_232467

theorem gcd_of_n13_minus_n : 
  ∀ n : ℤ, n ≠ 0 → 2730 ∣ (n ^ 13 - n) :=
by sorry

end gcd_of_n13_minus_n_l2324_232467


namespace minimum_value_of_function_l2324_232460

theorem minimum_value_of_function (x : ℝ) (hx : x > 4) : 
    (∃ y : ℝ, y = x + 9 / (x - 4) ∧ (∀ z : ℝ, (∃ w : ℝ, w > 4 ∧ z = w + 9 / (w - 4)) → z ≥ 10) ∧ y = 10) :=
sorry

end minimum_value_of_function_l2324_232460


namespace roots_difference_is_one_l2324_232444

noncomputable def quadratic_eq (p : ℝ) :=
  ∃ (α β : ℝ), (α ≠ β) ∧ (α - β = 1) ∧ (α ^ 2 - p * α + (p ^ 2 - 1) / 4 = 0) ∧ (β ^ 2 - p * β + (p ^ 2 - 1) / 4 = 0)

theorem roots_difference_is_one (p : ℝ) : quadratic_eq p :=
  sorry

end roots_difference_is_one_l2324_232444


namespace can_transfer_increase_average_l2324_232401

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end can_transfer_increase_average_l2324_232401


namespace robot_material_handling_per_hour_min_num_type_A_robots_l2324_232441

-- Definitions and conditions for part 1
def material_handling_robot_B (x : ℕ) := x
def material_handling_robot_A (x : ℕ) := x + 30

def condition_time_handled (x : ℕ) :=
  1000 / material_handling_robot_A x = 800 / material_handling_robot_B x

-- Definitions for part 2
def total_robots := 20
def min_material_handling_per_hour := 2800

def material_handling_total (a b : ℕ) :=
  150 * a + 120 * b

-- Proof problems
theorem robot_material_handling_per_hour :
  ∃ (x : ℕ), material_handling_robot_B x = 120 ∧ material_handling_robot_A x = 150 ∧ condition_time_handled x :=
sorry

theorem min_num_type_A_robots :
  ∀ (a b : ℕ),
  a + b = total_robots →
  material_handling_total a b ≥ min_material_handling_per_hour →
  a ≥ 14 :=
sorry

end robot_material_handling_per_hour_min_num_type_A_robots_l2324_232441


namespace kara_uses_28_cups_of_sugar_l2324_232479

theorem kara_uses_28_cups_of_sugar (S W : ℕ) (h1 : S + W = 84) (h2 : S * 2 = W) : S = 28 :=
by sorry

end kara_uses_28_cups_of_sugar_l2324_232479


namespace eval_poly_at_2_l2324_232442

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem eval_poly_at_2 :
  f 2 = 123 :=
by
  sorry

end eval_poly_at_2_l2324_232442


namespace sum_first_19_terms_l2324_232480

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a₀ a₃ a₁₇ a₁₀ : ℝ)

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ a₀ d, ∀ n, a n = a₀ + n * d

noncomputable def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_first_19_terms (h1 : is_arithmetic_sequence a)
                          (h2 : a 3 + a 17 = 10)
                          (h3 : sum_first_n_terms S a) :
  S 19 = 95 :=
sorry

end sum_first_19_terms_l2324_232480


namespace expression_value_l2324_232411

-- Define the difference of squares identity
lemma diff_of_squares (x y : ℤ) : x^2 - y^2 = (x + y) * (x - y) :=
by sorry

-- Define the specific values for x and y
def x := 7
def y := 3

-- State the theorem to be proven
theorem expression_value : ((x^2 - y^2)^2) = 1600 :=
by sorry

end expression_value_l2324_232411


namespace boat_man_mass_l2324_232419

theorem boat_man_mass (L B h : ℝ) (rho g : ℝ): 
  L = 3 → B = 2 → h = 0.015 → rho = 1000 → g = 9.81 → (rho * L * B * h * g) / g = 9 :=
by
  intros
  simp_all
  sorry

end boat_man_mass_l2324_232419


namespace evaluate_expression_l2324_232414

theorem evaluate_expression :
  (π - 2023) ^ 0 + |(-9)| - 3 ^ 2 = 1 :=
by
  sorry

end evaluate_expression_l2324_232414


namespace original_cost_of_each_magazine_l2324_232495

-- Definitions and conditions
def magazine_cost (C : ℝ) : Prop :=
  let total_magazines := 10
  let sell_price := 3.50
  let gain := 5
  let total_revenue := total_magazines * sell_price
  let total_cost := total_revenue - gain
  C = total_cost / total_magazines

-- Goal to prove
theorem original_cost_of_each_magazine : ∃ C : ℝ, magazine_cost C ∧ C = 3 :=
by
  sorry

end original_cost_of_each_magazine_l2324_232495


namespace find_slope_l2324_232405

theorem find_slope (k : ℝ) : (∃ x : ℝ, (y = k * x + 2) ∧ (y = 0) ∧ (abs x = 4)) ↔ (k = 1/2 ∨ k = -1/2) := by
  sorry

end find_slope_l2324_232405


namespace postal_code_permutations_l2324_232497

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def multiplicity_permutations (n : Nat) (repetitions : List Nat) : Nat :=
  factorial n / List.foldl (λ acc k => acc * factorial k) 1 repetitions

theorem postal_code_permutations : multiplicity_permutations 4 [2, 1, 1] = 12 :=
by
  unfold multiplicity_permutations
  unfold factorial
  sorry

end postal_code_permutations_l2324_232497


namespace sum_cubed_identity_l2324_232439

theorem sum_cubed_identity
  (p q r : ℝ)
  (h1 : p + q + r = 5)
  (h2 : pq + pr + qr = 7)
  (h3 : pqr = -10) :
  p^3 + q^3 + r^3 = -10 := 
by
  sorry

end sum_cubed_identity_l2324_232439


namespace part1_part2_i_part2_ii_l2324_232425

def equation1 (x : ℝ) : Prop := 3 * x - 2 = 0
def equation2 (x : ℝ) : Prop := 2 * x - 3 = 0
def equation3 (x : ℝ) : Prop := x - (3 * x + 1) = -7

def inequality1 (x : ℝ) : Prop := -x + 2 > x - 5
def inequality2 (x : ℝ) : Prop := 3 * x - 1 > -x + 2

def sys_ineq (x m : ℝ) : Prop := x + m < 2 * x ∧ x - 2 < m

def equation4 (x : ℝ) : Prop := (2 * x - 1) / 3 = -3

theorem part1 : 
  ∀ (x : ℝ), inequality1 x → inequality2 x → equation2 x → equation3 x :=
by sorry

theorem part2_i :
  ∀ (m : ℝ), (∃ (x : ℝ), equation4 x ∧ sys_ineq x m) → -6 < m ∧ m < -4 :=
by sorry

theorem part2_ii :
  ∀ (m : ℝ), ¬ (sys_ineq 1 m ∧ sys_ineq 2 m) → m ≥ 2 ∨ m ≤ -1 :=
by sorry

end part1_part2_i_part2_ii_l2324_232425


namespace find_mistake_l2324_232451

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l2324_232451


namespace larger_tablet_diagonal_length_l2324_232485

theorem larger_tablet_diagonal_length :
  ∀ (d : ℝ), (d^2 / 2 = 25 / 2 + 5.5) → d = 6 :=
by
  intro d
  sorry

end larger_tablet_diagonal_length_l2324_232485


namespace compute_expression_l2324_232429

theorem compute_expression : 3 * 3^4 - 9^19 / 9^17 = 162 := by
  sorry

end compute_expression_l2324_232429


namespace gold_coins_percentage_l2324_232453

-- Definitions for conditions
def percent_beads : Float := 0.30
def percent_sculptures : Float := 0.10
def percent_silver_coins : Float := 0.30

-- Definitions derived from conditions
def percent_coins : Float := 1.0 - percent_beads - percent_sculptures
def percent_gold_coins_among_coins : Float := 1.0 - percent_silver_coins

-- Theorem statement
theorem gold_coins_percentage : percent_gold_coins_among_coins * percent_coins = 0.42 :=
by
sorry

end gold_coins_percentage_l2324_232453


namespace trig_identity_l2324_232403

open Real

theorem trig_identity (α : ℝ) (h_tan : tan α = 2) (h_quad : 0 < α ∧ α < π / 2) :
  sin (2 * α) + cos α = (4 + sqrt 5) / 5 :=
sorry

end trig_identity_l2324_232403


namespace find_number_l2324_232450

theorem find_number (number : ℝ) (h : 0.001 * number = 0.24) : number = 240 :=
sorry

end find_number_l2324_232450


namespace Steven_has_16_apples_l2324_232483

variable (Jake_Peaches Steven_Peaches Jake_Apples Steven_Apples : ℕ)

theorem Steven_has_16_apples
  (h1 : Jake_Peaches = Steven_Peaches - 6)
  (h2 : Steven_Peaches = 17)
  (h3 : Steven_Peaches = Steven_Apples + 1)
  (h4 : Jake_Apples = Steven_Apples + 8) :
  Steven_Apples = 16 := by
  sorry

end Steven_has_16_apples_l2324_232483


namespace fraction_inequality_l2324_232431

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < (b + m) / (a + m) := 
sorry

end fraction_inequality_l2324_232431


namespace quadratic_function_value_at_neg_one_l2324_232488

theorem quadratic_function_value_at_neg_one (b c : ℝ) 
  (h1 : (1:ℝ) ^ 2 + b * 1 + c = 0) 
  (h2 : (3:ℝ) ^ 2 + b * 3 + c = 0) : 
  ((-1:ℝ) ^ 2 + b * (-1) + c = 8) :=
by
  sorry

end quadratic_function_value_at_neg_one_l2324_232488


namespace expression_simplification_l2324_232454

theorem expression_simplification (a b : ℤ) : 
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by
  sorry

end expression_simplification_l2324_232454


namespace part1_part2_l2324_232400

-- Part (1)  
theorem part1 (m : ℝ) : (∀ x : ℝ, 1 < x ∧ x < 3 → 2 * m < x ∧ x < 1 - m) ↔ (m ≤ -2) :=
sorry

-- Part (2)
theorem part2 (m : ℝ) : (∀ x : ℝ, (1 < x ∧ x < 3) → ¬ (2 * m < x ∧ x < 1 - m)) ↔ (0 ≤ m) :=
sorry

end part1_part2_l2324_232400


namespace add_to_fraction_l2324_232447

theorem add_to_fraction (x : ℕ) :
  (3 + x) / (11 + x) = 5 / 9 ↔ x = 7 :=
by
  sorry

end add_to_fraction_l2324_232447


namespace min_disks_required_l2324_232440

-- Define the initial conditions
def num_files : ℕ := 40
def disk_capacity : ℕ := 2 -- capacity in MB
def num_files_1MB : ℕ := 5
def num_files_0_8MB : ℕ := 15
def num_files_0_5MB : ℕ := 20
def size_1MB : ℕ := 1
def size_0_8MB : ℕ := 8/10 -- 0.8 MB
def size_0_5MB : ℕ := 1/2 -- 0.5 MB

-- Define the mathematical problem
theorem min_disks_required :
  (num_files_1MB * size_1MB + num_files_0_8MB * size_0_8MB + num_files_0_5MB * size_0_5MB) / disk_capacity ≤ 15 := by
  sorry

end min_disks_required_l2324_232440


namespace consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l2324_232446

-- 6(a): Prove that the product of two consecutive integers is either divisible by 6 or gives a remainder of 2 when divided by 18.
theorem consecutive_integers_product (n : ℕ) : n * (n + 1) % 18 = 0 ∨ n * (n + 1) % 18 = 2 := 
sorry

-- 6(b): Prove that there does not exist an integer n such that the number 3n + 1 is the product of two consecutive integers.
theorem no_3n_plus_1_product_consecutive : ¬ ∃ n : ℕ, ∃ m : ℕ, 3 * m + 1 = m * (m + 1) := 
sorry

-- 6(c): Prove that for no integer n, the number n^3 + 5n + 4 can be the product of two consecutive integers.
theorem no_n_cubed_plus_5n_plus_4_product_consecutive : ¬ ∃ n : ℕ, ∃ m : ℕ, n^3 + 5 * n + 4 = m * (m + 1) := 
sorry

-- 6(d): Prove that none of the numbers resulting from the rearrangement of the digits in 23456780 is the product of two consecutive integers.
def is_permutation (m : ℕ) (n : ℕ) : Prop := 
-- This function definition should check that m is a permutation of the digits of n
sorry

theorem no_permutation_23456780_product_consecutive : 
  ∀ m : ℕ, is_permutation m 23456780 → ¬ ∃ n : ℕ, m = n * (n + 1) := 
sorry

end consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l2324_232446


namespace quadratic_eq_solutions_l2324_232493

open Real

theorem quadratic_eq_solutions (x : ℝ) :
  (2 * x + 1) ^ 2 = (2 * x + 1) * (x - 1) ↔ x = -1 / 2 ∨ x = -2 :=
by sorry

end quadratic_eq_solutions_l2324_232493


namespace find_three_digit_numbers_l2324_232465
open Nat

theorem find_three_digit_numbers (n : ℕ) (h1 : 100 ≤ n) (h2 : n < 1000) (h3 : ∀ (k : ℕ), n^k % 1000 = n % 1000) : n = 625 ∨ n = 376 :=
sorry

end find_three_digit_numbers_l2324_232465


namespace factor_1_factor_2_triangle_is_isosceles_l2324_232432

-- Factorization problems
theorem factor_1 (x y : ℝ) : 
  (x^2 - x * y + 4 * x - 4 * y) = ((x - y) * (x + 4)) :=
sorry

theorem factor_2 (x y : ℝ) : 
  (x^2 - y^2 + 4 * y - 4) = ((x + y - 2) * (x - y + 2)) :=
sorry

-- Triangle shape problem
theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - a * c - b^2 + b * c = 0) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end factor_1_factor_2_triangle_is_isosceles_l2324_232432


namespace inheritance_amount_l2324_232452

theorem inheritance_amount (x : ℝ)
  (federal_tax_rate : ℝ := 0.25)
  (state_tax_rate : ℝ := 0.15)
  (total_taxes_paid : ℝ := 16000)
  (H : (federal_tax_rate * x) + (state_tax_rate * (1 - federal_tax_rate) * x) = total_taxes_paid) :
  x = 44138 := sorry

end inheritance_amount_l2324_232452


namespace monthly_income_l2324_232445

-- Define the conditions
variable (I : ℝ) -- Total monthly income
variable (remaining : ℝ) -- Remaining amount before donation
variable (remaining_after_donation : ℝ) -- Amount after donation

-- Conditions
def condition1 : Prop := remaining = I - 0.63 * I - 1500
def condition2 : Prop := remaining_after_donation = remaining - 0.05 * remaining
def condition3 : Prop := remaining_after_donation = 35000

-- Theorem to prove the total monthly income
theorem monthly_income (h1 : condition1 I remaining) (h2 : condition2 remaining remaining_after_donation) (h3 : condition3 remaining_after_donation) : I = 103600 := 
by sorry

end monthly_income_l2324_232445


namespace find_value_l2324_232470

theorem find_value (x : ℝ) (h : x^2 - 2 * x = 1) : 2023 + 6 * x - 3 * x^2 = 2020 := 
by 
sorry

end find_value_l2324_232470


namespace parabola_focus_distance_l2324_232490

theorem parabola_focus_distance (p : ℝ) : 
  (∀ (y : ℝ), y^2 = 2 * p * 4 → abs (4 + p / 2) = 5) → 
  p = 2 :=
by
  sorry

end parabola_focus_distance_l2324_232490


namespace ellipse_standard_equation_l2324_232430

theorem ellipse_standard_equation :
  ∃ (a b c : ℝ),
    2 * a = 10 ∧
    c / a = 3 / 5 ∧
    b^2 = a^2 - c^2 ∧
    (∀ x y : ℝ, (x^2 / 16) + (y^2 / 25) = 1) :=
by
  sorry

end ellipse_standard_equation_l2324_232430


namespace water_tank_full_capacity_l2324_232496

-- Define the conditions
variable {C x : ℝ}
variable (h1 : x / C = 1 / 3)
variable (h2 : (x + 6) / C = 1 / 2)

-- Prove that C = 36
theorem water_tank_full_capacity : C = 36 :=
by
  sorry

end water_tank_full_capacity_l2324_232496


namespace quadratic_inequality_solution_l2324_232402

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ -8/3 < k ∧ k < 6 :=
by
  sorry

end quadratic_inequality_solution_l2324_232402


namespace fraction_product_cube_l2324_232428

theorem fraction_product_cube :
  ((5 : ℚ) / 8)^3 * ((4 : ℚ) / 9)^3 = (125 : ℚ) / 5832 :=
by
  sorry

end fraction_product_cube_l2324_232428


namespace sum_of_interior_angles_of_polygon_l2324_232484

theorem sum_of_interior_angles_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 36) :
  ∃ interior_sum : ℝ, interior_sum = 1440 :=
by
  sorry

end sum_of_interior_angles_of_polygon_l2324_232484


namespace sum_of_squares_l2324_232498

theorem sum_of_squares (a b : ℝ) (h1 : (a + b)^2 = 11) (h2 : (a - b)^2 = 5) : a^2 + b^2 = 8 := 
sorry

end sum_of_squares_l2324_232498


namespace abc_min_value_l2324_232408

open Real

theorem abc_min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_sum : a + b + c = 1) (h_bound : a ≤ b ∧ b ≤ c ∧ c ≤ 3 * a) :
  3 * a * a * (1 - 4 * a) = (9/343) := 
sorry

end abc_min_value_l2324_232408


namespace nonneg_triple_inequality_l2324_232409

theorem nonneg_triple_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/3) * (a + b + c)^2 ≥ a * Real.sqrt (b * c) + b * Real.sqrt (c * a) + c * Real.sqrt (a * b) :=
by
  sorry

end nonneg_triple_inequality_l2324_232409


namespace simplify_fraction_l2324_232476

theorem simplify_fraction :
  (45 * (14 / 25) * (1 / 18) * (5 / 11) : ℚ) = 7 / 11 := 
by sorry

end simplify_fraction_l2324_232476


namespace meaningful_expression_condition_l2324_232435

theorem meaningful_expression_condition (x : ℝ) : (x > 1) ↔ (∃ y : ℝ, y = 2 / Real.sqrt (x - 1)) :=
by
  sorry

end meaningful_expression_condition_l2324_232435


namespace rational_root_of_polynomial_l2324_232474

-- Polynomial definition
def P (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 + 4 * x^2 + 6 * x - 8

-- Theorem statement
theorem rational_root_of_polynomial : ∀ x : ℚ, P x = 0 ↔ x = -1 :=
by
  sorry

end rational_root_of_polynomial_l2324_232474


namespace miley_discount_rate_l2324_232415

theorem miley_discount_rate :
  let cost_per_cellphone := 800
  let number_of_cellphones := 2
  let amount_paid := 1520
  let total_cost_without_discount := cost_per_cellphone * number_of_cellphones
  let discount_amount := total_cost_without_discount - amount_paid
  let discount_rate := (discount_amount / total_cost_without_discount) * 100
  discount_rate = 5 := by
    sorry

end miley_discount_rate_l2324_232415


namespace initial_cloves_l2324_232489

theorem initial_cloves (used_cloves left_cloves initial_cloves : ℕ) (h1 : used_cloves = 86) (h2 : left_cloves = 7) : initial_cloves = 93 :=
by
  sorry

end initial_cloves_l2324_232489


namespace arithmetic_sequence_eighth_term_l2324_232438

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Specify the given conditions
def a1 : ℚ := 10 / 11
def a15 : ℚ := 8 / 9

-- Prove that the eighth term is equal to 89 / 99
theorem arithmetic_sequence_eighth_term :
  ∃ d : ℚ, arithmetic_sequence a1 d 15 = a15 →
             arithmetic_sequence a1 d 8 = 89 / 99 :=
by
  sorry

end arithmetic_sequence_eighth_term_l2324_232438


namespace Jason_spent_correct_amount_l2324_232413

namespace MusicStore

def costFlute : Real := 142.46
def costMusicStand : Real := 8.89
def costSongBook : Real := 7.00
def totalCost : Real := 158.35

theorem Jason_spent_correct_amount :
  costFlute + costMusicStand + costSongBook = totalCost :=
sorry

end MusicStore

end Jason_spent_correct_amount_l2324_232413


namespace guilty_D_l2324_232437

def isGuilty (A B C D : Prop) : Prop :=
  ¬A ∧ (B → ∃! x, x ≠ A ∧ (x = C ∨ x = D)) ∧ (C → ∃! x₁ x₂, x₁ ≠ x₂ ∧ x₁ ≠ A ∧ x₂ ≠ A ∧ ((x₁ = B ∨ x₁ = D) ∧ (x₂ = B ∨ x₂ = D))) ∧ (¬A ∨ B ∨ C ∨ D)

theorem guilty_D (A B C D : Prop) (h : isGuilty A B C D) : D :=
by
  sorry

end guilty_D_l2324_232437


namespace result_after_subtraction_l2324_232466

-- Define the conditions
def x : ℕ := 40
def subtract_value : ℕ := 138

-- The expression we will evaluate
def result (x : ℕ) : ℕ := 6 * x - subtract_value

-- The theorem stating the evaluated result
theorem result_after_subtraction : result 40 = 102 :=
by
  unfold result
  rw [← Nat.mul_comm]
  simp
  sorry -- Proof placeholder

end result_after_subtraction_l2324_232466


namespace concrete_volume_is_six_l2324_232473

def to_yards (feet : ℕ) (inches : ℕ) : ℚ :=
  feet * (1 / 3) + inches * (1 / 36)

def sidewalk_volume (width_feet : ℕ) (length_feet : ℕ) (thickness_inches : ℕ) : ℚ :=
  to_yards width_feet 0 * to_yards length_feet 0 * to_yards 0 thickness_inches

def border_volume (border_width_feet : ℕ) (border_thickness_inches : ℕ) (sidewalk_length_feet : ℕ) : ℚ :=
  to_yards (2 * border_width_feet) 0 * to_yards sidewalk_length_feet 0 * to_yards 0 border_thickness_inches

def total_concrete_volume (sidewalk_width_feet : ℕ) (sidewalk_length_feet : ℕ) (sidewalk_thickness_inches : ℕ)
  (border_width_feet : ℕ) (border_thickness_inches : ℕ) : ℚ :=
  sidewalk_volume sidewalk_width_feet sidewalk_length_feet sidewalk_thickness_inches +
  border_volume border_width_feet border_thickness_inches sidewalk_length_feet

def volume_in_cubic_yards (w1_feet : ℕ) (l1_feet : ℕ) (t1_inches : ℕ) (w2_feet : ℕ) (t2_inches : ℕ) : ℚ :=
  total_concrete_volume w1_feet l1_feet t1_inches w2_feet t2_inches

theorem concrete_volume_is_six :
  -- conditions
  volume_in_cubic_yards 4 80 4 1 2 = 6 :=
by
  -- Proof omitted
  sorry

end concrete_volume_is_six_l2324_232473


namespace part1_part2_l2324_232494

-- Part (1)
theorem part1 (x y : ℚ) 
  (h1 : 2022 * x + 2020 * y = 2021)
  (h2 : 2023 * x + 2021 * y = 2022) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

-- Part (2)
theorem part2 (x y a b : ℚ)
  (ha : a ≠ b) 
  (h1 : (a + 1) * x + (a - 1) * y = a)
  (h2 : (b + 1) * x + (b - 1) * y = b) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

end part1_part2_l2324_232494


namespace rationalize_denominator_l2324_232418

theorem rationalize_denominator 
  (cbrt32_eq_2cbrt4 : (32:ℝ)^(1/3) = 2 * (4:ℝ)^(1/3))
  (cbrt16_eq_2cbrt2 : (16:ℝ)^(1/3) = 2 * (2:ℝ)^(1/3))
  (cbrt64_eq_4 : (64:ℝ)^(1/3) = 4) :
  1 / ((4:ℝ)^(1/3) + (32:ℝ)^(1/3)) = ((2:ℝ)^(1/3)) / 6 :=
  sorry

end rationalize_denominator_l2324_232418


namespace parallel_lines_l2324_232449

theorem parallel_lines :
  (∃ m: ℚ, (∀ x y: ℚ, (4 * y - 3 * x = 16 → y = m * x + (16 / 4)) ∧
                      (-3 * x - 4 * y = 15 → y = -m * x - (15 / 4)) ∧
                      (4 * y + 3 * x = 16 → y = -m * x + (16 / 4)) ∧
                      (3 * y + 4 * x = 15) → False)) :=
sorry

end parallel_lines_l2324_232449


namespace determine_c_l2324_232448

theorem determine_c (c : ℝ) :
  let vertex_x := -(-10 / (2 * 1))
  let vertex_y := c - ((-10)^2 / (4 * 1))
  ((5 - 0)^2 + (vertex_y - 0)^2 = 10^2)
  → (c = 25 + 5 * Real.sqrt 3 ∨ c = 25 - 5 * Real.sqrt 3) :=
by
  sorry

end determine_c_l2324_232448


namespace set_intersection_complement_l2324_232456

-- Definitions corresponding to conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | x > 1}

-- Statement to prove
theorem set_intersection_complement : A ∩ (U \ B) = {x | -1 < x ∧ x ≤ 1} := by
  sorry

end set_intersection_complement_l2324_232456


namespace trains_pass_each_other_l2324_232424

noncomputable def time_to_pass (speed1 speed2 distance : ℕ) : ℚ :=
  (distance : ℚ) / ((speed1 + speed2) : ℚ) * 60

theorem trains_pass_each_other :
  time_to_pass 60 80 100 = 42.86 := sorry

end trains_pass_each_other_l2324_232424


namespace complement_of_angle_l2324_232412

variable (α : ℝ)

axiom given_angle : α = 63 + 21 / 60

theorem complement_of_angle :
  90 - α = 26 + 39 / 60 :=
by
  sorry

end complement_of_angle_l2324_232412


namespace John_avg_speed_l2324_232463

theorem John_avg_speed :
  ∀ (initial final : ℕ) (time : ℕ),
    initial = 27372 →
    final = 27472 →
    time = 4 →
    ((final - initial) / time) = 25 :=
by
  intros initial final time h_initial h_final h_time
  sorry

end John_avg_speed_l2324_232463


namespace seeds_in_big_garden_l2324_232458

-- Definitions based on conditions
def total_seeds : ℕ := 42
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 2
def seeds_planted_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden

-- Proof statement
theorem seeds_in_big_garden : total_seeds - seeds_planted_in_small_gardens = 36 :=
sorry

end seeds_in_big_garden_l2324_232458


namespace inequalities_always_true_l2324_232499

variables {x y a b : Real}

/-- All given conditions -/
def conditions (x y a b : Real) :=
  x < a ∧ y < b ∧ x < 0 ∧ y < 0 ∧ a > 0 ∧ b > 0

theorem inequalities_always_true {x y a b : Real} (h : conditions x y a b) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ((x + y) / (x - y) < (a + b) / (a - b)) :=
sorry

end inequalities_always_true_l2324_232499


namespace wall_width_l2324_232482

theorem wall_width (brick_length brick_height brick_depth : ℝ)
    (wall_length wall_height : ℝ)
    (num_bricks : ℝ)
    (total_bricks_volume : ℝ)
    (total_wall_volume : ℝ) :
    brick_length = 25 →
    brick_height = 11.25 →
    brick_depth = 6 →
    wall_length = 800 →
    wall_height = 600 →
    num_bricks = 6400 →
    total_bricks_volume = num_bricks * (brick_length * brick_height * brick_depth) →
    total_wall_volume = wall_length * wall_height * (total_bricks_volume / (brick_length * brick_height * brick_depth)) →
    (total_bricks_volume / (wall_length * wall_height) = 22.5) :=
by
  intros
  sorry -- proof not required

end wall_width_l2324_232482


namespace ryan_weekly_commuting_time_l2324_232472

-- Define Ryan's commuting conditions
def bike_time (biking_days: Nat) : Nat := biking_days * 30
def bus_time (bus_days: Nat) : Nat := bus_days * 40
def friend_time (friend_days: Nat) : Nat := friend_days * 10

-- Calculate total commuting time per week
def total_commuting_time (biking_days bus_days friend_days: Nat) : Nat := 
  bike_time biking_days + bus_time bus_days + friend_time friend_days

-- Given conditions
def biking_days : Nat := 1
def bus_days : Nat := 3
def friend_days : Nat := 1

-- Formal statement to prove
theorem ryan_weekly_commuting_time : 
  total_commuting_time biking_days bus_days friend_days = 160 := by 
  sorry

end ryan_weekly_commuting_time_l2324_232472


namespace math_problem_l2324_232491

noncomputable def is_solution (x : ℝ) : Prop :=
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12

theorem math_problem :
  (is_solution ((7 + Real.sqrt 153) / 2)) ∧ (is_solution ((7 - Real.sqrt 153) / 2)) := 
by
  sorry

end math_problem_l2324_232491


namespace inequality_proof_l2324_232459

theorem inequality_proof
  (a b x y z : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (x_pos : 0 < x) 
  (y_pos : 0 < y) 
  (z_pos : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_proof_l2324_232459


namespace ceil_sqrt_sum_l2324_232461

theorem ceil_sqrt_sum :
  ⌈Real.sqrt 3⌉₊ + ⌈Real.sqrt 27⌉₊ + ⌈Real.sqrt 243⌉₊ = 24 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 := by sorry
  have h3 : 15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 := by sorry
  sorry

end ceil_sqrt_sum_l2324_232461
