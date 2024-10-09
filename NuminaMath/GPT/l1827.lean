import Mathlib

namespace ratio_of_pens_to_pencils_l1827_182752

-- Define the conditions
def total_items : ℕ := 13
def pencils : ℕ := 4
def eraser : ℕ := 1
def pens : ℕ := total_items - pencils - eraser

-- Prove the ratio of pens to pencils is 2:1
theorem ratio_of_pens_to_pencils : pens = 2 * pencils :=
by
  -- indicate that the proof is omitted
  sorry

end ratio_of_pens_to_pencils_l1827_182752


namespace cylinder_volume_ratio_l1827_182784

noncomputable def volume_of_cylinder (height : ℝ) (circumference : ℝ) : ℝ := 
  let r := circumference / (2 * Real.pi)
  Real.pi * r^2 * height

theorem cylinder_volume_ratio :
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  max V1 V2 / min V1 V2 = 5 / 3 :=
by
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  have hV1 : V1 = 90 / Real.pi := sorry
  have hV2 : V2 = 150 / Real.pi := sorry
  sorry

end cylinder_volume_ratio_l1827_182784


namespace fraction_of_book_finished_l1827_182765

variables (x y : ℝ)

theorem fraction_of_book_finished (h1 : x = y + 90) (h2 : x + y = 270) : x / 270 = 2 / 3 :=
by sorry

end fraction_of_book_finished_l1827_182765


namespace problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l1827_182793

theorem problem1421_part1 (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ)
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_yellow : yellow_balls = 15) :
  (red_balls < yellow_balls) := by 
  sorry  -- Solution Proof for Part 1

theorem problem1421_part2 (total_balls : ℕ) (red_balls : ℕ) (h_total : total_balls = 20) 
  (h_red : red_balls = 5) :
  (red_balls / total_balls = 1 / 4) := by 
  sorry  -- Solution Proof for Part 2

theorem problem1421_part3 (red_balls total_balls m : ℕ) (h_red : red_balls = 5) 
  (h_total : total_balls = 20) :
  ((red_balls + m) / (total_balls + m) = 3 / 4) → (m = 40) := by 
  sorry  -- Solution Proof for Part 3

theorem problem1421_part4 (total_balls red_balls additional_balls x : ℕ) 
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_additional : additional_balls = 18):
  (total_balls + additional_balls = 38) → ((red_balls + x) / 38 = 1 / 2) → 
  (x = 14) ∧ ((additional_balls - x) = 4) := by 
  sorry  -- Solution Proof for Part 4

end problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l1827_182793


namespace alarm_prob_l1827_182779

theorem alarm_prob (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.90) : 
  (1 - (1 - pA) * (1 - pB)) = 0.98 :=
by 
  sorry

end alarm_prob_l1827_182779


namespace optionD_is_deductive_l1827_182722

-- Conditions related to the reasoning options
inductive ReasoningProcess where
  | optionA : ReasoningProcess
  | optionB : ReasoningProcess
  | optionC : ReasoningProcess
  | optionD : ReasoningProcess

-- Definitions matching the equivalent Lean problem
def isDeductiveReasoning (rp : ReasoningProcess) : Prop :=
  match rp with
  | ReasoningProcess.optionA => False
  | ReasoningProcess.optionB => False
  | ReasoningProcess.optionC => False
  | ReasoningProcess.optionD => True

-- The proposition we need to prove
theorem optionD_is_deductive :
  isDeductiveReasoning ReasoningProcess.optionD = True := by
  sorry

end optionD_is_deductive_l1827_182722


namespace river_current_speed_l1827_182728

noncomputable section

variables {d r w : ℝ}

def time_equation_normal_speed (d r w : ℝ) : Prop :=
  (d / (r + w)) + 4 = (d / (r - w))

def time_equation_tripled_speed (d r w : ℝ) : Prop :=
  (d / (3 * r + w)) + 2 = (d / (3 * r - w))

theorem river_current_speed (d r : ℝ) (h1 : time_equation_normal_speed d r w) (h2 : time_equation_tripled_speed d r w) : w = 2 :=
sorry

end river_current_speed_l1827_182728


namespace common_tangent_at_point_l1827_182780

theorem common_tangent_at_point (x₀ b : ℝ) 
  (h₁ : 6 * x₀^2 = 6 * x₀) 
  (h₂ : 1 + 2 * x₀^3 = 3 * x₀^2 - b) :
  b = 0 ∨ b = -1 :=
sorry

end common_tangent_at_point_l1827_182780


namespace combined_value_of_cookies_l1827_182710

theorem combined_value_of_cookies
  (total_boxes_sold : ℝ)
  (plain_boxes_sold : ℝ)
  (price_chocolate_chip : ℝ)
  (price_plain : ℝ)
  (h1 : total_boxes_sold = 1585)
  (h2 : plain_boxes_sold = 793.375)
  (h3 : price_chocolate_chip = 1.25)
  (h4 : price_plain = 0.75) :
  (plain_boxes_sold * price_plain) + ((total_boxes_sold - plain_boxes_sold) * price_chocolate_chip) = 1584.5625 :=
by
  sorry

end combined_value_of_cookies_l1827_182710


namespace area_of_inscribed_triangle_l1827_182712

-- Define the square with a given diagonal
def diagonal (d : ℝ) : Prop := d = 16
def side_length_of_square (s : ℝ) : Prop := s = 8 * Real.sqrt 2
def side_length_of_equilateral_triangle (a : ℝ) : Prop := a = 8 * Real.sqrt 2

-- Define the area of the equilateral triangle
def area_of_equilateral_triangle (area : ℝ) : Prop :=
  area = 32 * Real.sqrt 3

-- The theorem: Given the above conditions, prove the area of the equilateral triangle
theorem area_of_inscribed_triangle (d s a area : ℝ) 
  (h1 : diagonal d) 
  (h2 : side_length_of_square s) 
  (h3 : side_length_of_equilateral_triangle a) 
  (h4 : s = a) : 
  area_of_equilateral_triangle area :=
sorry

end area_of_inscribed_triangle_l1827_182712


namespace solve_for_x_l1827_182702

theorem solve_for_x (x : ℝ) : 
  (x - 35) / 3 = (3 * x + 10) / 8 → x = -310 := by
  sorry

end solve_for_x_l1827_182702


namespace shortest_distance_from_curve_to_line_l1827_182787

noncomputable def curve (x : ℝ) : ℝ := Real.log (2 * x - 1)

def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem shortest_distance_from_curve_to_line : 
  ∃ (x y : ℝ), y = curve x ∧ line x y ∧ 
  (∀ (x₀ y₀ : ℝ), y₀ = curve x₀ → ∃ (x₀ y₀ : ℝ), 
    y₀ = curve x₀ ∧ d = Real.sqrt 5) :=
sorry

end shortest_distance_from_curve_to_line_l1827_182787


namespace elasticity_ratio_is_correct_l1827_182749

-- Definitions of the given elasticities
def e_OGBR_QN : ℝ := 1.27
def e_OGBR_PN : ℝ := 0.76

-- Theorem stating the ratio of elasticities equals 1.7
theorem elasticity_ratio_is_correct : (e_OGBR_QN / e_OGBR_PN) = 1.7 := sorry

end elasticity_ratio_is_correct_l1827_182749


namespace systematic_sample_first_segment_number_l1827_182789

theorem systematic_sample_first_segment_number :
  ∃ a_1 : ℕ, ∀ d k : ℕ, k = 5 → a_1 + (59 - 1) * k = 293 → a_1 = 3 :=
by
  sorry

end systematic_sample_first_segment_number_l1827_182789


namespace function_additive_of_tangential_property_l1827_182756

open Set

variable {f : ℝ → ℝ}

def is_tangential_quadrilateral_sides (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ (a + c = b + d)

theorem function_additive_of_tangential_property
  (h : ∀ (a b c d : ℝ), is_tangential_quadrilateral_sides a b c d → f (a + b + c + d) = f a + f b + f c + f d) :
  ∀ (x y : ℝ), 0 < x → 0 < y → f (x + y) = f x + f y :=
by
  sorry

end function_additive_of_tangential_property_l1827_182756


namespace annual_expenditure_l1827_182744

theorem annual_expenditure (x y : ℝ) (h1 : y = 0.8 * x + 0.1) (h2 : x = 15) : y = 12.1 :=
by
  sorry

end annual_expenditure_l1827_182744


namespace total_votes_polled_l1827_182760

theorem total_votes_polled (V: ℝ) (h: 0 < V) (h1: 0.70 * V - 0.30 * V = 320) : V = 800 :=
sorry

end total_votes_polled_l1827_182760


namespace base_3_is_most_economical_l1827_182754

theorem base_3_is_most_economical (m d : ℕ) (h : d ≥ 1) (h_m_div_d : m % d = 0) :
  3^(m / 3) ≥ d^(m / d) :=
sorry

end base_3_is_most_economical_l1827_182754


namespace max_value_y_l1827_182714

theorem max_value_y (x y : ℕ) (h₁ : 9 * (x + y) > 17 * x) (h₂ : 15 * x < 8 * (x + y)) :
  y ≤ 112 :=
sorry

end max_value_y_l1827_182714


namespace simultaneous_equations_solution_l1827_182723

theorem simultaneous_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 20) ∧ (9 * x - 8 * y = 36) ∧ (x = 76 / 15) ∧ (y = 18 / 15) :=
by
  sorry

end simultaneous_equations_solution_l1827_182723


namespace sum_of_consecutive_integers_product_384_l1827_182759

theorem sum_of_consecutive_integers_product_384 :
  ∃ (a : ℤ), a * (a + 1) * (a + 2) = 384 ∧ a + (a + 1) + (a + 2) = 24 :=
by
  sorry

end sum_of_consecutive_integers_product_384_l1827_182759


namespace ratio_correct_l1827_182753

def my_age : ℕ := 35
def son_age_next_year : ℕ := 8
def son_age_now : ℕ := son_age_next_year - 1
def ratio_of_ages : ℕ := my_age / son_age_now

theorem ratio_correct : ratio_of_ages = 5 :=
by
  -- Add proof here
  sorry

end ratio_correct_l1827_182753


namespace yellow_marbles_count_l1827_182770

theorem yellow_marbles_count 
  (total_marbles red_marbles blue_marbles : ℕ) 
  (h_total : total_marbles = 85) 
  (h_red : red_marbles = 14) 
  (h_blue : blue_marbles = 3 * red_marbles) :
  (total_marbles - (red_marbles + blue_marbles)) = 29 :=
by
  sorry

end yellow_marbles_count_l1827_182770


namespace abs_iff_neg_one_lt_x_lt_one_l1827_182785

theorem abs_iff_neg_one_lt_x_lt_one (x : ℝ) : |x| < 1 ↔ -1 < x ∧ x < 1 :=
by
  sorry

end abs_iff_neg_one_lt_x_lt_one_l1827_182785


namespace solution_set_quadratic_inequality_l1827_182733

theorem solution_set_quadratic_inequality :
  {x : ℝ | (x^2 - 3*x + 2) < 0} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end solution_set_quadratic_inequality_l1827_182733


namespace digits_of_result_l1827_182792

theorem digits_of_result 
  (u1 u2 t1 t2 h1 h2 : ℕ) 
  (hu_condition : u1 = u2 + 6)
  (units_column : u1 - u2 = 5)
  (tens_column : t1 - t2 = 9)
  (no_borrowing : u2 < u1) 
  : (h1, u1 - u2) = (4, 5) := 
sorry

end digits_of_result_l1827_182792


namespace points_connected_l1827_182799

theorem points_connected (m l : ℕ) (h1 : l < m) (h2 : Even (l * m)) :
  ∃ points : Finset (ℕ × ℕ), ∀ p ∈ points, (∃ q, q ∈ points ∧ (p ≠ q → p.snd = q.snd → p.fst = q.fst)) :=
sorry

end points_connected_l1827_182799


namespace proof_problem_l1827_182735

variable (a b c d x : ℤ)

-- Conditions
axiom condition1 : a - b = c + d + x
axiom condition2 : a + b = c - d - 3
axiom condition3 : a - c = 3
axiom answer_eq : x = 9

-- Proof statement
theorem proof_problem : (a - b) = (c + d + 9) :=
by
  sorry

end proof_problem_l1827_182735


namespace solve_equation1_solve_equation2_l1827_182775

theorem solve_equation1 (x : ℝ) (h : 4 * x^2 - 81 = 0) : x = 9/2 ∨ x = -9/2 := 
sorry

theorem solve_equation2 (x : ℝ) (h : 8 * (x + 1)^3 = 27) : x = 1/2 := 
sorry

end solve_equation1_solve_equation2_l1827_182775


namespace total_ages_l1827_182742

variable (Bill_age Caroline_age : ℕ)
variable (h1 : Bill_age = 2 * Caroline_age - 1) (h2 : Bill_age = 17)

theorem total_ages : Bill_age + Caroline_age = 26 :=
by
  sorry

end total_ages_l1827_182742


namespace fraction_problem_l1827_182777

-- Definitions given in the conditions
variables {p q r s : ℚ}
variables (h₁ : p / q = 8)
variables (h₂ : r / q = 5)
variables (h₃ : r / s = 3 / 4)

-- Statement to prove
theorem fraction_problem : s / p = 5 / 6 :=
by
  sorry

end fraction_problem_l1827_182777


namespace no_integers_p_q_l1827_182713

theorem no_integers_p_q :
  ¬ ∃ p q : ℤ, ∀ x : ℤ, 3 ∣ (x^2 + p * x + q) :=
by
  sorry

end no_integers_p_q_l1827_182713


namespace arithmetic_seq_sum_l1827_182704

theorem arithmetic_seq_sum (a : ℕ → ℤ) (h_arith_seq : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) (h_a5 : a 5 = 15) : a 2 + a 4 + a 6 + a 8 = 60 := 
by
  sorry

end arithmetic_seq_sum_l1827_182704


namespace ratio_of_distances_l1827_182766

/-- 
  Given two points A and B moving along intersecting lines with constant,
  but different velocities v_A and v_B respectively, prove that there exists a 
  point P such that at any moment in time, the ratio of distances AP to BP equals 
  the ratio of their velocities.
-/
theorem ratio_of_distances (A B : ℝ → ℝ × ℝ) (v_A v_B : ℝ)
  (intersecting_lines : ∃ t, A t = B t)
  (diff_velocities : v_A ≠ v_B) :
  ∃ P : ℝ × ℝ, ∀ t, (dist P (A t) / dist P (B t)) = v_A / v_B := 
sorry

end ratio_of_distances_l1827_182766


namespace min_value_frac_sum_l1827_182731

open Real

theorem min_value_frac_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 2 * a + b = 1) :
  1 / a + 2 / b = 8 :=
sorry

end min_value_frac_sum_l1827_182731


namespace selection_methods_l1827_182757

theorem selection_methods (students : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) (h1 : students = 8) (h2 : boys = 6) (h3 : girls = 2) (h4 : selected = 4) : 
  ∃ methods, methods = 40 :=
by
  have h5 : students = boys + girls := by linarith
  sorry

end selection_methods_l1827_182757


namespace more_silverfish_than_goldfish_l1827_182701

variable (n G S R : ℕ)

-- Condition 1: If the cat eats all the goldfish, the number of remaining fish is \(\frac{2}{3}\)n - 1
def condition1 := n - G = (2 * n) / 3 - 1

-- Condition 2: If the cat eats all the redfish, the number of remaining fish is \(\frac{2}{3}\)n + 4
def condition2 := n - R = (2 * n) / 3 + 4

-- The goal: Silverfish are more numerous than goldfish by 2
theorem more_silverfish_than_goldfish (h1 : condition1 n G) (h2 : condition2 n R) :
  S = (n / 3) + 3 → G = (n / 3) + 1 → S - G = 2 :=
by
  sorry

end more_silverfish_than_goldfish_l1827_182701


namespace coin_toss_probability_l1827_182774

-- Define the sample space of the coin toss
inductive Coin
| heads : Coin
| tails : Coin

-- Define the probability function
def probability (outcome : Coin) : ℝ :=
  match outcome with
  | Coin.heads => 0.5
  | Coin.tails => 0.5

-- The theorem to be proved: In a fair coin toss, the probability of getting "heads" or "tails" is 0.5
theorem coin_toss_probability (outcome : Coin) : probability outcome = 0.5 :=
sorry

end coin_toss_probability_l1827_182774


namespace A_holds_15_l1827_182762

def cards : List (ℕ × ℕ) := [(1, 3), (1, 5), (3, 5)]

variables (A_card B_card C_card : ℕ × ℕ)

-- Conditions from the problem
def C_not_35 : Prop := C_card ≠ (3, 5)
def A_says_not_3 (A_card B_card : ℕ × ℕ) : Prop := ¬(A_card.1 = 3 ∧ B_card.1 = 3 ∨ A_card.2 = 3 ∧ B_card.2 = 3)
def B_says_not_1 (B_card C_card : ℕ × ℕ) : Prop := ¬(B_card.1 = 1 ∧ C_card.1 = 1 ∨ B_card.2 = 1 ∧ C_card.2 = 1)

-- Question to prove
theorem A_holds_15 : 
  ∃ (A_card B_card C_card : ℕ × ℕ),
    A_card ∈ cards ∧ B_card ∈ cards ∧ C_card ∈ cards ∧
    A_card ≠ B_card ∧ B_card ≠ C_card ∧ A_card ≠ C_card ∧
    C_not_35 C_card ∧
    A_says_not_3 A_card B_card ∧
    B_says_not_1 B_card C_card ->
    A_card = (1, 5) :=
sorry

end A_holds_15_l1827_182762


namespace estimate_larger_than_difference_l1827_182755

variable {x y : ℝ}

theorem estimate_larger_than_difference (h1 : x > y) (h2 : y > 0) :
    ⌈x⌉ - ⌊y⌋ > x - y := by
  sorry

end estimate_larger_than_difference_l1827_182755


namespace trigonometric_ineq_l1827_182737

theorem trigonometric_ineq (h₁ : (Real.pi / 4) < 1.5) (h₂ : 1.5 < (Real.pi / 2)) : 
  Real.cos 1.5 < Real.sin 1.5 ∧ Real.sin 1.5 < Real.tan 1.5 := 
sorry

end trigonometric_ineq_l1827_182737


namespace no_daily_coverage_l1827_182715

theorem no_daily_coverage (ranks : Nat → Nat)
  (h_ranks_ordered : ∀ i, ranks (i+1) ≥ 3 * ranks i)
  (h_cycle : ∀ i, ∃ N : Nat, ranks i = N ∧ ∃ k : Nat, k = N ∧ ∀ m, m % (2 * N) < N → (¬ ∃ j, ranks j ≤ N))
  : ¬ (∀ d : Nat, ∃ j : Nat, (∃ k : Nat, d % (2 * (ranks j)) < ranks j))
  := sorry

end no_daily_coverage_l1827_182715


namespace distinct_intersection_points_l1827_182761

theorem distinct_intersection_points :
  let S1 := { p : ℝ × ℝ | (p.1 + p.2 - 7) * (2 * p.1 - 3 * p.2 + 9) = 0 }
  let S2 := { p : ℝ × ℝ | (p.1 - p.2 - 2) * (4 * p.1 + 3 * p.2 - 18) = 0 }
  ∃! (p1 p2 p3 : ℝ × ℝ), p1 ∈ S1 ∧ p1 ∈ S2 ∧ p2 ∈ S1 ∧ p2 ∈ S2 ∧ p3 ∈ S1 ∧ p3 ∈ S2 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end distinct_intersection_points_l1827_182761


namespace prime_p_perfect_cube_l1827_182705

theorem prime_p_perfect_cube (p : ℕ) (hp : Nat.Prime p) (h : ∃ n : ℕ, 13 * p + 1 = n^3) :
  p = 2 ∨ p = 211 :=
by
  sorry

end prime_p_perfect_cube_l1827_182705


namespace remainder_avg_is_correct_l1827_182743

-- Definitions based on the conditions
variables (total_avg : ℝ) (first_part_avg : ℝ) (second_part_avg : ℝ) (first_part_percent : ℝ) (second_part_percent : ℝ)

-- The conditions stated mathematically
def overall_avg_contribution 
  (remainder_avg : ℝ) : Prop :=
  first_part_percent * first_part_avg + 
  second_part_percent * second_part_avg + 
  (1 - first_part_percent - second_part_percent) * remainder_avg =  total_avg
  
-- The question
theorem remainder_avg_is_correct : overall_avg_contribution 75 80 65 0.25 0.50 90 := sorry

end remainder_avg_is_correct_l1827_182743


namespace ordering_of_powers_l1827_182782

theorem ordering_of_powers :
  (3:ℕ)^15 < 10^9 ∧ 10^9 < (5:ℕ)^13 :=
by
  sorry

end ordering_of_powers_l1827_182782


namespace closest_multiple_of_12_l1827_182797

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

-- Define the closest multiple of 4 to 2050 (2048 and 2052)
def closest_multiple_of_4 (n m : ℕ) : ℕ :=
if n % 4 < m % 4 then n - (n % 4)
else m + (4 - (m % 4))

-- Define the conditions for being divisible by both 3 and 4
def is_multiple_of_12 (n : ℕ) : Prop := is_multiple_of n 12

-- Theorem statement
theorem closest_multiple_of_12 (n m : ℕ) (h : n = 2050) (hm : m = 2052) :
  is_multiple_of_12 m :=
sorry

end closest_multiple_of_12_l1827_182797


namespace solve_equation_l1827_182709

noncomputable def a := 3 + Real.sqrt 8
noncomputable def b := 3 - Real.sqrt 8

theorem solve_equation (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 6) ↔ (x = 2 ∨ x = -2) := 
  by
  sorry

end solve_equation_l1827_182709


namespace suitable_storage_temp_l1827_182750

theorem suitable_storage_temp : -5 ≤ -1 ∧ -1 ≤ 1 := by {
  sorry
}

end suitable_storage_temp_l1827_182750


namespace max_stamps_l1827_182745

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h_price : price_per_stamp = 37) (h_total : total_money = 4000) : 
  ∃ max_stamps : ℕ, max_stamps = 108 ∧ max_stamps * price_per_stamp ≤ total_money ∧ ∀ n : ℕ, n * price_per_stamp ≤ total_money → n ≤ max_stamps :=
by
  sorry

end max_stamps_l1827_182745


namespace max_value_of_expression_l1827_182717

theorem max_value_of_expression (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) : 
  (∃ x : ℝ, x = 3 → 
    ∀ A : ℝ, A = (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3)) → 
      A ≤ x) :=
by
  sorry

end max_value_of_expression_l1827_182717


namespace work_completion_l1827_182703

noncomputable def efficiency (p q: ℕ) := q = 3 * p / 5

theorem work_completion (p q : ℕ) (h1 : efficiency p q) (h2: p * 24 = 100) :
  2400 / (p + q) = 15 :=
by 
  sorry

end work_completion_l1827_182703


namespace largest_number_of_pangs_largest_number_of_pangs_possible_l1827_182719

theorem largest_number_of_pangs (x y z : ℕ) 
  (hx : x ≥ 2) 
  (hy : y ≥ 3) 
  (hcost : 3 * x + 4 * y + 9 * z = 100) : 
  z ≤ 9 :=
by sorry

theorem largest_number_of_pangs_possible (x y z : ℕ) 
  (hx : x = 2) 
  (hy : y = 3) 
  (hcost : 3 * x + 4 * y + 9 * z = 100) : 
  z = 9 :=
by sorry

end largest_number_of_pangs_largest_number_of_pangs_possible_l1827_182719


namespace rental_lower_amount_eq_50_l1827_182727

theorem rental_lower_amount_eq_50 (L : ℝ) (total_rent : ℝ) (reduction : ℝ) (rooms_changed : ℕ) (diff_per_room : ℝ)
  (h1 : total_rent = 400)
  (h2 : reduction = 0.25 * total_rent)
  (h3 : rooms_changed = 10)
  (h4 : diff_per_room = reduction / ↑rooms_changed)
  (h5 : 60 - L = diff_per_room) :
  L = 50 :=
  sorry

end rental_lower_amount_eq_50_l1827_182727


namespace fraction_power_multiplication_l1827_182741

theorem fraction_power_multiplication :
  ( (1 / 3) ^ 4 * (1 / 5) = 1 / 405 ) :=
by
  sorry

end fraction_power_multiplication_l1827_182741


namespace found_bottle_caps_is_correct_l1827_182730

def initial_bottle_caps : ℕ := 6
def total_bottle_caps : ℕ := 28

theorem found_bottle_caps_is_correct : total_bottle_caps - initial_bottle_caps = 22 := by
  sorry

end found_bottle_caps_is_correct_l1827_182730


namespace min_value_polynomial_expression_at_k_eq_1_is_0_l1827_182791

-- Definition of the polynomial expression
def polynomial_expression (k x y : ℝ) : ℝ :=
  3 * x^2 - 4 * k * x * y + (2 * k^2 + 1) * y^2 - 6 * x - 2 * y + 4

-- Proof statement
theorem min_value_polynomial_expression_at_k_eq_1_is_0 :
  (∀ x y : ℝ, polynomial_expression 1 x y ≥ 0) ∧ (∃ x y : ℝ, polynomial_expression 1 x y = 0) :=
by
  -- Expected proof here. For now, we indicate sorry to skip the proof.
  sorry

end min_value_polynomial_expression_at_k_eq_1_is_0_l1827_182791


namespace inequality_correct_l1827_182738

variable (a b : ℝ)

theorem inequality_correct (h : a < b) : 2 - a > 2 - b :=
by
  sorry

end inequality_correct_l1827_182738


namespace multiply_polynomials_l1827_182751

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l1827_182751


namespace neon_signs_blink_together_l1827_182773

-- Define the time intervals for the blinks
def blink_interval1 : ℕ := 7
def blink_interval2 : ℕ := 11
def blink_interval3 : ℕ := 13

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- State the theorem
theorem neon_signs_blink_together : Nat.lcm (Nat.lcm blink_interval1 blink_interval2) blink_interval3 = 1001 := by
  sorry

end neon_signs_blink_together_l1827_182773


namespace initial_red_marbles_l1827_182739

theorem initial_red_marbles (R : ℕ) (blue_marbles_initial : ℕ) (red_marbles_removed : ℕ) :
  blue_marbles_initial = 30 →
  red_marbles_removed = 3 →
  (R - red_marbles_removed) + (blue_marbles_initial - 4 * red_marbles_removed) = 35 →
  R = 20 :=
by
  intros h_blue h_red h_total
  sorry

end initial_red_marbles_l1827_182739


namespace percent_profit_l1827_182718

theorem percent_profit (C S : ℝ) (h : 60 * C = 50 * S):
  (((S - C) / C) * 100) = 20 :=
by 
  sorry

end percent_profit_l1827_182718


namespace ratio_a6_b6_l1827_182788

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence a
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Define the nth term of sequence b
noncomputable def S_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence a
noncomputable def T_n (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of sequence b

axiom condition (n : ℕ) : S_n n / T_n n = (2 * n) / (3 * n + 1)

theorem ratio_a6_b6 : a_n 6 / b_n 6 = 11 / 17 :=
by
  sorry

end ratio_a6_b6_l1827_182788


namespace inequality_proof_l1827_182708

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 :=
by
  sorry

end inequality_proof_l1827_182708


namespace f_values_sum_l1827_182746

noncomputable def f : ℝ → ℝ := sorry

-- defining the properties
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- given conditions
axiom f_odd : is_odd f
axiom f_periodic : is_periodic f 2

-- statement to prove
theorem f_values_sum : f 1 + f 2 + f 3 = 0 :=
by
  sorry

end f_values_sum_l1827_182746


namespace factorize_one_factorize_two_l1827_182734

variable (m x y : ℝ)

-- Problem statement for Question 1
theorem factorize_one (m : ℝ) : 
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := 
sorry

-- Problem statement for Question 2
theorem factorize_two (x y : ℝ) : 
  (x + y)^2 - 4 * (x + y) + 4 = (x + y - 2)^2 := 
sorry

end factorize_one_factorize_two_l1827_182734


namespace prime_q_exists_l1827_182763

theorem prime_q_exists (p : ℕ) (pp : Nat.Prime p) : 
  ∃ q, Nat.Prime q ∧ (∀ n, n > 0 → ¬ q ∣ n ^ p - p) := 
sorry

end prime_q_exists_l1827_182763


namespace binary_subtraction_result_l1827_182781

theorem binary_subtraction_result :
  let x := 0b1101101 -- binary notation for 109
  let y := 0b11101   -- binary notation for 29
  let z := 0b101010  -- binary notation for 42
  let product := x * y
  let result := product - z
  result = 0b10000010001 := -- binary notation for 3119
by
  sorry

end binary_subtraction_result_l1827_182781


namespace geometric_sequence_sum_inverse_equals_l1827_182747

variable (a : ℕ → ℝ)
variable (n : ℕ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃(r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum_inverse_equals (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 = 15 / 8)
  (h_prod : a 6 * a 7 = -9 / 8) :
  (1 / a 5) + (1 / a 6) + (1 / a 7) + (1 / a 8) = -5 / 3 :=
by
  sorry

end geometric_sequence_sum_inverse_equals_l1827_182747


namespace part1_part2a_part2b_l1827_182778

-- Definitions and conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-3, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def scalar_mul (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def collinear (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Proof statements

-- Part 1: Verify the dot product computation
theorem part1 : dot_product (vector_add vector_a vector_b) (vector_sub vector_a vector_b) = -8 := by
  sorry

-- Part 2a: Verify the value of k for parallel vectors
theorem part2a : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (vector_sub vector_a (scalar_mul 3 vector_b)) := by
  sorry

-- Part 2b: Verify antiparallel direction
theorem part2b : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (scalar_mul (-1) (vector_sub vector_a (scalar_mul 3 vector_b))) := by
  sorry

end part1_part2a_part2b_l1827_182778


namespace excircle_problem_l1827_182795

-- Define the data structure for a triangle with incenter and excircle properties
structure TriangleWithIncenterAndExcircle (α : Type) [LinearOrderedField α] :=
  (A B C I X : α)
  (is_incenter : Boolean)  -- condition for point I being the incenter
  (is_excircle_center_opposite_A : Boolean)  -- condition for point X being the excircle center opposite A
  (I_A_I : I ≠ A)
  (X_A_X : X ≠ A)

-- Define the problem statement
theorem excircle_problem
  (α : Type) [LinearOrderedField α]
  (T : TriangleWithIncenterAndExcircle α)
  (h_incenter : T.is_incenter)
  (h_excircle_center : T.is_excircle_center_opposite_A)
  (h_not_eq_I : T.I ≠ T.A)
  (h_not_eq_X : T.X ≠ T.A)
  : 
    (T.I * T.X = T.A * T.B) ∧ 
    (T.I * (T.B * T.C) = T.X * (T.B * T.C)) :=
by
  sorry

end excircle_problem_l1827_182795


namespace percentage_of_other_investment_l1827_182748

theorem percentage_of_other_investment (investment total_interest interest_5 interest_other percentage_other : ℝ) 
  (h1 : investment = 18000)
  (h2 : interest_5 = 6000 * 0.05)
  (h3 : total_interest = 660)
  (h4 : percentage_other / 100 * (investment - 6000) = 360) : 
  percentage_other = 3 :=
by
  sorry

end percentage_of_other_investment_l1827_182748


namespace total_phd_time_l1827_182706

-- Definitions for the conditions
def acclimation_period : ℕ := 1
def basics_period : ℕ := 2
def research_period := basics_period + (3 * basics_period / 4)
def dissertation_period := acclimation_period / 2

-- Main statement to prove
theorem total_phd_time : acclimation_period + basics_period + research_period + dissertation_period = 7 := by
  -- Here should be the proof (skipped with sorry)
  sorry

end total_phd_time_l1827_182706


namespace smallest_natrural_number_cube_ends_888_l1827_182736

theorem smallest_natrural_number_cube_ends_888 :
  ∃ n : ℕ, (n^3 % 1000 = 888) ∧ (∀ m : ℕ, (m^3 % 1000 = 888) → n ≤ m) := 
sorry

end smallest_natrural_number_cube_ends_888_l1827_182736


namespace calculate_sum_of_squares_l1827_182769

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end calculate_sum_of_squares_l1827_182769


namespace max_students_l1827_182725

theorem max_students (A B C : ℕ) (A_left B_left C_left : ℕ)
  (hA : A = 38) (hB : B = 78) (hC : C = 128)
  (hA_left : A_left = 2) (hB_left : B_left = 6) (hC_left : C_left = 20) :
  gcd (A - A_left) (gcd (B - B_left) (C - C_left)) = 36 :=
by {
  sorry
}

end max_students_l1827_182725


namespace largest_sum_fraction_l1827_182758

open Rat

theorem largest_sum_fraction :
  let a := (2:ℚ) / 5
  let c1 := (1:ℚ) / 6
  let c2 := (1:ℚ) / 3
  let c3 := (1:ℚ) / 7
  let c4 := (1:ℚ) / 8
  let c5 := (1:ℚ) / 9
  max (a + c1) (max (a + c2) (max (a + c3) (max (a + c4) (a + c5)))) = a + c2
  ∧ a + c2 = (11:ℚ) / 15 := by
  sorry

end largest_sum_fraction_l1827_182758


namespace relationship_of_squares_and_products_l1827_182740

theorem relationship_of_squares_and_products (a b x : ℝ) (h1 : b < x) (h2 : x < a) (h3 : a < 0) : 
  x^2 > ax ∧ ax > b^2 :=
by
  sorry

end relationship_of_squares_and_products_l1827_182740


namespace ratio_income_to_expenditure_l1827_182776

theorem ratio_income_to_expenditure (I E S : ℕ) 
  (h1 : I = 10000) 
  (h2 : S = 3000) 
  (h3 : S = I - E) : I / Nat.gcd I E = 10 ∧ E / Nat.gcd I E = 7 := by 
  sorry

end ratio_income_to_expenditure_l1827_182776


namespace find_circle_center_l1827_182790

theorem find_circle_center :
  ∀ x y : ℝ,
  (x^2 + 4*x + y^2 - 6*y = 20) →
  (x + 2, y - 3) = (-2, 3) := by
  sorry

end find_circle_center_l1827_182790


namespace not_square_or_cube_l1827_182794

theorem not_square_or_cube (n : ℕ) (h : n > 1) : 
  ¬ (∃ a : ℕ, 2^n - 1 = a^2) ∧ ¬ (∃ a : ℕ, 2^n - 1 = a^3) :=
by
  sorry

end not_square_or_cube_l1827_182794


namespace people_remaining_on_bus_l1827_182764

theorem people_remaining_on_bus
  (students_left : ℕ) (students_right : ℕ) (students_back : ℕ)
  (students_aisle : ℕ) (teachers : ℕ) (bus_driver : ℕ) 
  (students_off1 : ℕ) (teachers_off1 : ℕ)
  (students_off2 : ℕ) (teachers_off2 : ℕ)
  (students_off3 : ℕ) :
  students_left = 42 ∧ students_right = 38 ∧ students_back = 5 ∧
  students_aisle = 15 ∧ teachers = 2 ∧ bus_driver = 1 ∧
  students_off1 = 14 ∧ teachers_off1 = 1 ∧
  students_off2 = 18 ∧ teachers_off2 = 1 ∧
  students_off3 = 5 →
  (students_left + students_right + students_back + students_aisle + teachers + bus_driver) -
  (students_off1 + teachers_off1 + students_off2 + teachers_off2 + students_off3) = 64 :=
by {
  sorry
}

end people_remaining_on_bus_l1827_182764


namespace A_minus_B_l1827_182796

def A : ℕ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℕ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem A_minus_B : A - B = 128 := by
  sorry

end A_minus_B_l1827_182796


namespace find_some_number_l1827_182786

theorem find_some_number (some_number : ℝ) :
  (0.0077 * some_number) / (0.04 * 0.1 * 0.007) = 990.0000000000001 → 
  some_number = 3.6 :=
by
  intro h
  sorry

end find_some_number_l1827_182786


namespace donald_paul_ratio_l1827_182798

-- Let P be the number of bottles Paul drinks in one day.
-- Let D be the number of bottles Donald drinks in one day.
def paul_bottles (P : ℕ) := P = 3
def donald_bottles (D : ℕ) := D = 9

theorem donald_paul_ratio (P D : ℕ) (hP : paul_bottles P) (hD : donald_bottles D) : D / P = 3 :=
by {
  -- Insert proof steps here using the conditions.
  sorry
}

end donald_paul_ratio_l1827_182798


namespace min_value_frac_l1827_182700

theorem min_value_frac (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) (h3 : a * c = 4) : 
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, y = (1 / c + 9 / a) → y ≥ x :=
by sorry

end min_value_frac_l1827_182700


namespace find_b_l1827_182783

theorem find_b (a b c : ℝ) (h1 : a + b + c = 120) (h2 : a + 5 = b - 5) (h3 : b - 5 = c^2) : b = 61.25 :=
by {
  sorry
}

end find_b_l1827_182783


namespace icosahedron_edge_probability_l1827_182772

theorem icosahedron_edge_probability :
  let vertices := 12
  let total_pairs := vertices * (vertices - 1) / 2
  let edges := 30
  let probability := edges.toFloat / total_pairs.toFloat
  probability = 5 / 11 :=
by
  sorry

end icosahedron_edge_probability_l1827_182772


namespace two_point_form_eq_l1827_182720

theorem two_point_form_eq (x y : ℝ) : 
  let A := (5, 6)
  let B := (-1, 2)
  (y - 6) / (2 - 6) = (x - 5) / (-1 - 5) := 
  sorry

end two_point_form_eq_l1827_182720


namespace min_value_sin_sq_l1827_182707

theorem min_value_sin_sq (A B : ℝ) (h : A + B = π / 2) :
  4 / (Real.sin A)^2 + 9 / (Real.sin B)^2 ≥ 25 :=
sorry

end min_value_sin_sq_l1827_182707


namespace vershoks_per_arshin_l1827_182768

theorem vershoks_per_arshin (plank_length_arshins : ℝ) (plank_width_vershoks : ℝ) 
    (room_side_length_arshins : ℝ) (total_planks : ℕ) (n : ℝ)
    (h1 : plank_length_arshins = 6) (h2 : plank_width_vershoks = 6)
    (h3 : room_side_length_arshins = 12) (h4 : total_planks = 64) 
    (h5 : (total_planks : ℝ) * (plank_length_arshins * (plank_width_vershoks / n)) = room_side_length_arshins^2) :
    n = 16 :=
by {
  sorry
}

end vershoks_per_arshin_l1827_182768


namespace cost_of_3000_pencils_l1827_182729

-- Define the cost per box and the number of pencils per box
def cost_per_box : ℝ := 36
def pencils_per_box : ℕ := 120

-- Define the number of pencils to buy
def pencils_to_buy : ℕ := 3000

-- Define the total cost to prove
def total_cost_to_prove : ℝ := 900

-- The theorem to prove
theorem cost_of_3000_pencils : 
  (cost_per_box / pencils_per_box) * pencils_to_buy = total_cost_to_prove :=
by
  sorry

end cost_of_3000_pencils_l1827_182729


namespace relationship_f_2011_2014_l1827_182716

noncomputable def quadratic_func : Type := ℝ → ℝ

variable (f : quadratic_func)

-- The function is symmetric about x = 2013
axiom symmetry (x : ℝ) : f (2013 + x) = f (2013 - x)

-- The function opens upward (convexity)
axiom opens_upward (a b : ℝ) : f ((a + b) / 2) ≤ (f a + f b) / 2

theorem relationship_f_2011_2014 :
  f 2011 > f 2014 := 
sorry

end relationship_f_2011_2014_l1827_182716


namespace average_white_paper_per_ton_trees_saved_per_ton_l1827_182711

-- Define the given conditions
def waste_paper_tons : ℕ := 5
def produced_white_paper_tons : ℕ := 4
def saved_trees : ℕ := 40

-- State the theorems that need to be proved
theorem average_white_paper_per_ton :
  (produced_white_paper_tons : ℚ) / waste_paper_tons = 0.8 := 
sorry

theorem trees_saved_per_ton :
  (saved_trees : ℚ) / waste_paper_tons = 8 := 
sorry

end average_white_paper_per_ton_trees_saved_per_ton_l1827_182711


namespace rate_of_change_area_at_t4_l1827_182771

variable (t : ℝ)

def a (t : ℝ) : ℝ := 2 * t + 1

def b (t : ℝ) : ℝ := 3 * t + 2

def S (t : ℝ) : ℝ := a t * b t

theorem rate_of_change_area_at_t4 :
  (deriv S 4) = 55 := by
  sorry

end rate_of_change_area_at_t4_l1827_182771


namespace simplify_tangent_expression_l1827_182726

theorem simplify_tangent_expression :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end simplify_tangent_expression_l1827_182726


namespace range_of_a_no_real_roots_l1827_182721

theorem range_of_a_no_real_roots (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 + ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_no_real_roots_l1827_182721


namespace ab_is_square_l1827_182724

theorem ab_is_square (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_main : a + b = b * (a - c)) (h_prime : ∃ p : ℕ, Prime p ∧ c + 1 = p^2) :
  ∃ k : ℕ, a + b = k^2 :=
by
  sorry

end ab_is_square_l1827_182724


namespace students_passed_in_dixon_lecture_l1827_182767

theorem students_passed_in_dixon_lecture :
  let ratio_collins := 18 / 30
  let students_dixon := 45
  ∃ y, ratio_collins = y / students_dixon ∧ y = 27 :=
by
  sorry

end students_passed_in_dixon_lecture_l1827_182767


namespace emily_and_eli_probability_l1827_182732

noncomputable def probability_same_number : ℚ :=
  let count_multiples (n k : ℕ) := (k - 1) / n
  let emily_count := count_multiples 20 250
  let eli_count := count_multiples 30 250
  let common_lcm := Nat.lcm 20 30
  let common_count := count_multiples common_lcm 250
  common_count / (emily_count * eli_count : ℚ)

theorem emily_and_eli_probability :
  let probability := probability_same_number
  probability = 1 / 24 :=
by
  sorry

end emily_and_eli_probability_l1827_182732
