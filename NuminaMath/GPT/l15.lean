import Mathlib

namespace angle_inclusion_l15_1593

-- Defining the sets based on the given conditions
def M : Set ℝ := { x | 0 < x ∧ x ≤ 90 }
def N : Set ℝ := { x | 0 < x ∧ x < 90 }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 90 }

-- The proof statement
theorem angle_inclusion : N ⊆ M ∧ M ⊆ P :=
by
  sorry

end angle_inclusion_l15_1593


namespace complement_of_A_in_U_l15_1557

def U : Set ℕ := {1, 2, 3, 4}

def satisfies_inequality (x : ℕ) : Prop := x^2 - 5 * x + 4 < 0

def A : Set ℕ := {x | satisfies_inequality x}

theorem complement_of_A_in_U : U \ A = {1, 4} :=
by
  -- Proof omitted.
  sorry

end complement_of_A_in_U_l15_1557


namespace complex_pure_imaginary_is_x_eq_2_l15_1524

theorem complex_pure_imaginary_is_x_eq_2
  (x : ℝ)
  (z : ℂ)
  (h : z = ⟨x^2 - 3 * x + 2, x - 1⟩)
  (pure_imaginary : z.re = 0) :
  x = 2 :=
by
  sorry

end complex_pure_imaginary_is_x_eq_2_l15_1524


namespace average_age_of_9_l15_1584

theorem average_age_of_9 : 
  ∀ (avg_20 avg_5 age_15 : ℝ),
  avg_20 = 15 →
  avg_5 = 14 →
  age_15 = 86 →
  (9 * (69/9)) = 7.67 :=
by
  intros avg_20 avg_5 age_15 avg_20_val avg_5_val age_15_val
  -- The proof is skipped
  sorry

end average_age_of_9_l15_1584


namespace elder_person_age_l15_1565

-- Definitions based on conditions
variables (y e : ℕ) 

-- Given conditions
def condition1 : Prop := e = y + 20
def condition2 : Prop := e - 5 = 5 * (y - 5)

-- Theorem stating the required proof problem
theorem elder_person_age (h1 : condition1 y e) (h2 : condition2 y e) : e = 30 :=
by
  sorry

end elder_person_age_l15_1565


namespace hunter_saw_32_frogs_l15_1536

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end hunter_saw_32_frogs_l15_1536


namespace inequality_proof_l15_1591

theorem inequality_proof (a b c : ℝ) (hp : 0 < a ∧ 0 < b ∧ 0 < c) (hd : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    (bc / a + ac / b + ab / c > a + b + c) :=
by
  sorry

end inequality_proof_l15_1591


namespace hyperbola_sum_l15_1570

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := -4
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 53
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_sum : h + k + a + b = 3 + Real.sqrt 37 :=
by
  -- sorry is used to skip the proof as per the instruction
  sorry
  -- exact calc
  --   h + k + a + b = 3 + (-4) + 4 + Real.sqrt 37 : by simp
  --             ... = 3 + Real.sqrt 37 : by simp

end hyperbola_sum_l15_1570


namespace ratio_doctors_lawyers_l15_1527

theorem ratio_doctors_lawyers (d l : ℕ) (h1 : (45 * d + 60 * l) / (d + l) = 50) (h2 : d + l = 50) : d = 2 * l :=
by
  sorry

end ratio_doctors_lawyers_l15_1527


namespace max_value_sin_cos_combination_l15_1571

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end max_value_sin_cos_combination_l15_1571


namespace NaOH_combined_l15_1595

theorem NaOH_combined (n : ℕ) (h : n = 54) : 
  (2 * n) / 2 = 54 :=
by
  sorry

end NaOH_combined_l15_1595


namespace geometric_sequence_a5_l15_1586

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6)
  (h2 : a 3 + a 5 + a 7 = 78)
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  a 5 = 18 :=
by sorry

end geometric_sequence_a5_l15_1586


namespace total_first_tier_college_applicants_l15_1513

theorem total_first_tier_college_applicants
  (total_students : ℕ)
  (sample_size : ℕ)
  (sample_applicants : ℕ)
  (total_applicants : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sample_size = 150)
  (h3 : sample_applicants = 60)
  : total_applicants = 400 :=
sorry

end total_first_tier_college_applicants_l15_1513


namespace blocks_left_l15_1579

/-- Problem: Randy has 78 blocks. He uses 19 blocks to build a tower. Prove that he has 59 blocks left. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) (remaining_blocks : ℕ) : initial_blocks = 78 → used_blocks = 19 → remaining_blocks = initial_blocks - used_blocks → remaining_blocks = 59 :=
by
  sorry

end blocks_left_l15_1579


namespace difference_of_two_numbers_l15_1562

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end difference_of_two_numbers_l15_1562


namespace sum_of_squares_of_roots_l15_1550

theorem sum_of_squares_of_roots (a b : ℝ) (x₁ x₂ : ℝ)
  (h₁ : x₁^2 - (3 * a + b) * x₁ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0)
  (h₂ : x₂^2 - (3 * a + b) * x₂ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0) :
  x₁^2 + x₂^2 = 5 * (a^2 + b^2) := 
by
  sorry

end sum_of_squares_of_roots_l15_1550


namespace truth_prob_l15_1511

-- Define the probabilities
def prob_A := 0.80
def prob_B := 0.60
def prob_C := 0.75

-- The problem statement
theorem truth_prob :
  prob_A * prob_B * prob_C = 0.27 :=
by
  -- Proof would go here
  sorry

end truth_prob_l15_1511


namespace wash_time_difference_l15_1521

def C := 30
def T := 2 * C
def total_time := 135

theorem wash_time_difference :
  ∃ S, C + T + S = total_time ∧ T - S = 15 :=
by
  sorry

end wash_time_difference_l15_1521


namespace find_A_l15_1547

theorem find_A (A B : ℕ) (h1 : 15 = 3 * A) (h2 : 15 = 5 * B) : A = 5 := 
by 
  sorry

end find_A_l15_1547


namespace complement_of_M_l15_1502

-- Definitions:
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Assertion:
theorem complement_of_M :
  (U \ M) = {x | x ≤ -1} ∪ {x | 2 < x} :=
by sorry

end complement_of_M_l15_1502


namespace sum_of_cube_faces_l15_1517

-- Define the cube numbers as consecutive integers starting from 15.
def cube_faces (faces : List ℕ) : Prop :=
  faces = [15, 16, 17, 18, 19, 20]

-- Define the condition that the sum of numbers on opposite faces is the same.
def opposite_faces_condition (pairs : List (ℕ × ℕ)) : Prop :=
  ∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) = 35

theorem sum_of_cube_faces : ∃ faces : List ℕ, cube_faces faces ∧ (∃ pairs : List (ℕ × ℕ), opposite_faces_condition pairs ∧ faces.sum = 105) :=
by
  sorry

end sum_of_cube_faces_l15_1517


namespace tangent_line_at_one_e_l15_1501

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_one_e : ∀ (x y : ℝ), (x, y) = (1, Real.exp 1) → (y = 2 * Real.exp x * x - Real.exp 1) :=
by
  intro x y h
  sorry

end tangent_line_at_one_e_l15_1501


namespace total_money_l15_1526

theorem total_money (n : ℕ) (h1 : n * 3 = 36) :
  let one_rupee := n * 1
  let five_rupee := n * 5
  let ten_rupee := n * 10
  (one_rupee + five_rupee + ten_rupee) = 192 :=
by
  -- Note: The detailed calculations would go here in the proof
  -- Since we don't need to provide the proof, we add sorry to indicate the omitted part
  sorry

end total_money_l15_1526


namespace solve_expression_l15_1509

theorem solve_expression :
  2^3 + 2 * 5 - 3 + 6 = 21 :=
by
  sorry

end solve_expression_l15_1509


namespace odd_checkerboard_cannot_be_covered_by_dominoes_l15_1523

theorem odd_checkerboard_cannot_be_covered_by_dominoes 
    (m n : ℕ) (h : (m * n) % 2 = 1) :
    ¬ ∃ (dominos : Finset (Fin 2 × Fin 2)),
    ∀ {i j : Fin 2}, (i, j) ∈ dominos → 
    ((i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0)) ∧ 
    dominos.card = (m * n) / 2 := sorry

end odd_checkerboard_cannot_be_covered_by_dominoes_l15_1523


namespace cosine_of_negative_three_pi_over_two_l15_1508

theorem cosine_of_negative_three_pi_over_two : 
  Real.cos (-3 * Real.pi / 2) = 0 := 
by sorry

end cosine_of_negative_three_pi_over_two_l15_1508


namespace arithmetic_sequence_first_term_l15_1525

theorem arithmetic_sequence_first_term (d : ℤ) (a_n a_2 a_9 a_11 : ℤ) 
  (h1 : a_2 = 7) 
  (h2 : a_11 = a_9 + 6)
  (h3 : a_11 = a_n + 10 * d)
  (h4 : a_9 = a_n + 8 * d)
  (h5 : a_2 = a_n + d) :
  a_n = 4 := by
  sorry

end arithmetic_sequence_first_term_l15_1525


namespace projection_of_vector_a_on_b_l15_1543

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / norm_b

theorem projection_of_vector_a_on_b
  (a b : ℝ × ℝ) 
  (ha : Real.sqrt (a.1^2 + a.2^2) = 1)
  (hb : Real.sqrt (b.1^2 + b.2^2) = 2)
  (theta : ℝ)
  (h_theta : theta = Real.pi * (5/6)) -- 150 degrees in radians
  (h_cos_theta : Real.cos theta = -(Real.sqrt 3 / 2)) :
  vector_projection a b = -Real.sqrt 3 / 2 := 
by
  sorry

end projection_of_vector_a_on_b_l15_1543


namespace typing_cost_equation_l15_1596

def typing_cost (x : ℝ) : ℝ :=
  200 * x + 80 * 3 + 20 * 6

theorem typing_cost_equation (x : ℝ) (h : typing_cost x = 1360) : x = 5 :=
by
  sorry

end typing_cost_equation_l15_1596


namespace last_card_in_box_l15_1597

-- Define the zigzag pattern
def card_position (n : Nat) : Nat :=
  let cycle_pos := n % 12
  if cycle_pos = 0 then
    12
  else
    cycle_pos

def box_for_card (pos : Nat) : Nat :=
  if pos ≤ 7 then
    pos
  else
    14 - pos

theorem last_card_in_box : box_for_card (card_position 2015) = 3 := by
  sorry

end last_card_in_box_l15_1597


namespace invalid_votes_percentage_l15_1539

theorem invalid_votes_percentage (total_votes : ℕ) (valid_votes_candidate2 : ℕ) (valid_votes_percentage_candidate1 : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_valid_votes_candidate2 : valid_votes_candidate2 = 2700)
  (h_valid_votes_percentage_candidate1 : valid_votes_percentage_candidate1 = 55) :
  ((total_votes - (valid_votes_candidate2 * 100 / (100 - valid_votes_percentage_candidate1))) * 100 / total_votes) = 20 :=
by sorry

end invalid_votes_percentage_l15_1539


namespace largest_integer_condition_l15_1528

theorem largest_integer_condition (x : ℤ) : (x/3 + 3/4 : ℚ) < 7/3 → x ≤ 4 :=
by
  sorry

end largest_integer_condition_l15_1528


namespace min_value_of_fraction_l15_1520

theorem min_value_of_fraction (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 := 
sorry

end min_value_of_fraction_l15_1520


namespace area_of_trapezium_l15_1575

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
by sorry

end area_of_trapezium_l15_1575


namespace total_marks_eq_300_second_candidate_percentage_l15_1590

-- Defining the conditions
def percentage_marks (total_marks : ℕ) : ℕ := 40
def fail_by (fail_marks : ℕ) : ℕ := 40
def passing_marks : ℕ := 160

-- The number of total marks in the exam computed from conditions
theorem total_marks_eq_300 : ∃ T, 0.40 * T = 120 :=
by
  use 300
  sorry

-- The percentage of marks the second candidate gets
theorem second_candidate_percentage : ∃ percent, percent = (180 / 300) * 100 :=
by
  use 60
  sorry

end total_marks_eq_300_second_candidate_percentage_l15_1590


namespace complement_union_M_N_correct_l15_1552

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define the set N
def N : Set ℕ := {5, 6, 7}

-- Define the union of M and N
def union_M_N : Set ℕ := M ∪ N

-- Define the complement of the union of M and N in U
def complement_union_M_N : Set ℕ := U \ union_M_N

-- Main theorem statement to prove
theorem complement_union_M_N_correct : complement_union_M_N = {2, 4, 8} :=
by
  sorry

end complement_union_M_N_correct_l15_1552


namespace square_not_covered_by_circles_l15_1566

noncomputable def area_uncovered_by_circles : Real :=
  let side_length := 2
  let square_area := (side_length^2 : Real)
  let radius := 1
  let circle_area := Real.pi * radius^2
  let quarter_circle_area := circle_area / 4
  let total_circles_area := 4 * quarter_circle_area
  square_area - total_circles_area

theorem square_not_covered_by_circles :
  area_uncovered_by_circles = 4 - Real.pi := sorry

end square_not_covered_by_circles_l15_1566


namespace number_of_digits_in_expression_l15_1540

theorem number_of_digits_in_expression : 
  (Nat.digits 10 (2^12 * 5^8)).length = 10 := 
by
  sorry

end number_of_digits_in_expression_l15_1540


namespace geometric_sequence_common_ratio_l15_1599

theorem geometric_sequence_common_ratio
  (a_n : ℕ → ℝ)
  (q : ℝ)
  (h1 : a_n 3 = 7)
  (h2 : a_n 1 + a_n 2 + a_n 3 = 21) :
  q = 1 ∨ q = -1 / 2 :=
sorry

end geometric_sequence_common_ratio_l15_1599


namespace geometric_sequence_fourth_term_l15_1506

theorem geometric_sequence_fourth_term
  (a₁ a₅ : ℕ)
  (r : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₅ = 2187)
  (h₃ : a₅ = a₁ * r ^ 4) :
  a₁ * r ^ 3 = 2187 :=
by {
  sorry
}

end geometric_sequence_fourth_term_l15_1506


namespace mouse_grasshopper_diff_l15_1568

def grasshopper_jump: ℕ := 19
def frog_jump: ℕ := grasshopper_jump + 10
def mouse_jump: ℕ := frog_jump + 20

theorem mouse_grasshopper_diff:
  (mouse_jump - grasshopper_jump) = 30 :=
by
  sorry

end mouse_grasshopper_diff_l15_1568


namespace mean_median_sum_is_11_l15_1585

theorem mean_median_sum_is_11 (m n : ℕ) (h1 : m + 5 < n)
  (h2 : (m + (m + 3) + (m + 5) + n + (n + 1) + (2 * n - 1)) / 6 = n)
  (h3 : (m + 5 + n) / 2 = n) : m + n = 11 := by
  sorry

end mean_median_sum_is_11_l15_1585


namespace abc_not_all_positive_l15_1515

theorem abc_not_all_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ac > 0) (h3 : abc > 0) : 
  ¬(a > 0 ∧ b > 0 ∧ c > 0) ↔ (a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0) := 
by 
sorry

end abc_not_all_positive_l15_1515


namespace min_convex_cover_area_l15_1546

-- Define the dimensions of the box and the hole
def box_side := 5
def hole_side := 1

-- Define a function to represent the minimum area convex cover
def min_area_convex_cover (box_side hole_side : ℕ) : ℕ :=
  5 -- As given in the problem, the minimum area is concluded to be 5.

-- Theorem to state that the minimum area of the convex cover is 5
theorem min_convex_cover_area : min_area_convex_cover box_side hole_side = 5 :=
by
  -- Proof of the theorem
  sorry

end min_convex_cover_area_l15_1546


namespace gcd_30_45_is_15_l15_1533

theorem gcd_30_45_is_15 : Nat.gcd 30 45 = 15 := by
  sorry

end gcd_30_45_is_15_l15_1533


namespace part1_part2_l15_1594

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k)
noncomputable def f_prime (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k) + Real.exp x / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f_prime x k - 2 * (f x k + Real.exp x)
noncomputable def phi (x : ℝ) : ℝ := Real.exp x / x

theorem part1 (h : f_prime 1 k = 0) : k = -1 := sorry

theorem part2 (t : ℝ) (h_g_le_phi : ∀ x > 0, g x (-1) ≤ t * phi x) : t ≥ 1 + 1 / Real.exp 2 := sorry

end part1_part2_l15_1594


namespace rectangles_in_grid_at_least_three_cells_l15_1583

theorem rectangles_in_grid_at_least_three_cells :
  let number_of_rectangles (n : ℕ) := (n + 1).choose 2 * (n + 1).choose 2
  let single_cell_rectangles (n : ℕ) := n * n
  let one_by_two_or_two_by_one_rectangles (n : ℕ) := n * (n - 1) * 2
  let total_rectangles (n : ℕ) := number_of_rectangles n - (single_cell_rectangles n + one_by_two_or_two_by_one_rectangles n)
  total_rectangles 6 = 345 :=
by
  sorry

end rectangles_in_grid_at_least_three_cells_l15_1583


namespace milk_production_l15_1531

theorem milk_production (y : ℕ) (hcows : y > 0) (hcans : y + 2 > 0) (hdays : y + 3 > 0) :
  let daily_production_per_cow := (y + 2 : ℕ) / (y * (y + 3) : ℕ)
  let total_daily_production := (y + 4 : ℕ) * daily_production_per_cow
  let required_days := (y + 6 : ℕ) / total_daily_production
  required_days = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by
  sorry

end milk_production_l15_1531


namespace tammy_speed_second_day_l15_1500

theorem tammy_speed_second_day :
  ∀ (v1 t1 v2 t2 : ℝ), 
    t1 + t2 = 14 →
    t2 = t1 - 2 →
    v2 = v1 + 0.5 →
    v1 * t1 + v2 * t2 = 52 →
    v2 = 4 :=
by
  intros v1 t1 v2 t2 h1 h2 h3 h4
  sorry

end tammy_speed_second_day_l15_1500


namespace planeThroughPointAndLine_l15_1551

theorem planeThroughPointAndLine :
  ∃ A B C D : ℤ, (A = -3 ∧ B = -4 ∧ C = -4 ∧ D = 14) ∧ 
  (∀ x y z : ℝ, x = 2 ∧ y = -3 ∧ z = 5 ∨ (∃ t : ℝ, x = 4 * t + 2 ∧ y = -5 * t - 1 ∧ z = 2 * t + 3) → A * x + B * y + C * z + D = 0) :=
sorry

end planeThroughPointAndLine_l15_1551


namespace measure_of_angle_A_range_of_b2_add_c2_div_a2_l15_1569

variable {A B C a b c : ℝ}
variable {S : ℝ}

theorem measure_of_angle_A
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) : 
  A = 2 * Real.pi / 3 :=
by
  sorry

theorem range_of_b2_add_c2_div_a2
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : A = 2 * Real.pi / 3) : 
  2 / 3 ≤ (b ^ 2 + c ^ 2) / a ^ 2 ∧ (b ^ 2 + c ^ 2) / a ^ 2 < 1 :=
by
  sorry

end measure_of_angle_A_range_of_b2_add_c2_div_a2_l15_1569


namespace initial_caterpillars_l15_1514

theorem initial_caterpillars (C : ℕ) 
    (hatch_eggs : C + 4 - 8 = 10) : C = 14 :=
by
  sorry

end initial_caterpillars_l15_1514


namespace smallest_k_l15_1512

-- Define p as the largest prime number with 2023 digits
def p : ℕ := sorry -- This represents the largest prime number with 2023 digits

-- Define the target k
def k : ℕ := 1

-- The theorem stating that k is the smallest positive integer such that p^2 - k is divisible by 30
theorem smallest_k (p_largest_prime : ∀ m : ℕ, m ≤ p → Nat.Prime m → m = p) 
  (p_digits : 10^2022 ≤ p ∧ p < 10^2023) : 
  ∀ n : ℕ, n > 0 → (p^2 - n) % 30 = 0 → n = k :=
by 
  sorry

end smallest_k_l15_1512


namespace functional_equation_solution_exists_l15_1587

theorem functional_equation_solution_exists (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  intro h
  sorry

end functional_equation_solution_exists_l15_1587


namespace solve_for_x_l15_1567

theorem solve_for_x (x : ℚ) (h : (2 * x + 18) / (x - 6) = (2 * x - 4) / (x + 10)) : x = -26 / 9 :=
sorry

end solve_for_x_l15_1567


namespace john_income_increase_l15_1578

theorem john_income_increase :
  let initial_job_income := 60
  let initial_freelance_income := 40
  let initial_online_sales_income := 20

  let new_job_income := 120
  let new_freelance_income := 60
  let new_online_sales_income := 35

  let weeks_per_month := 4

  let initial_monthly_income := (initial_job_income + initial_freelance_income + initial_online_sales_income) * weeks_per_month
  let new_monthly_income := (new_job_income + new_freelance_income + new_online_sales_income) * weeks_per_month
  
  let percentage_increase := 100 * (new_monthly_income - initial_monthly_income) / initial_monthly_income

  percentage_increase = 79.17 := by
  sorry

end john_income_increase_l15_1578


namespace fraction_of_board_shaded_is_one_fourth_l15_1577

def totalArea : ℕ := 16
def shadedTopLeft : ℕ := 4
def shadedBottomRight : ℕ := 4
def fractionShaded (totalArea shadedTopLeft shadedBottomRight : ℕ) : ℚ :=
  (shadedTopLeft + shadedBottomRight) / totalArea

theorem fraction_of_board_shaded_is_one_fourth :
  fractionShaded totalArea shadedTopLeft shadedBottomRight = 1 / 4 := by
  sorry

end fraction_of_board_shaded_is_one_fourth_l15_1577


namespace combinations_of_eight_choose_three_is_fifty_six_l15_1522

theorem combinations_of_eight_choose_three_is_fifty_six :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end combinations_of_eight_choose_three_is_fifty_six_l15_1522


namespace set_intersection_l15_1581

def A := {x : ℝ | x^2 - 3*x ≥ 0}
def B := {x : ℝ | x < 1}
def intersection := {x : ℝ | x ≤ 0}

theorem set_intersection : A ∩ B = intersection :=
  sorry

end set_intersection_l15_1581


namespace water_temp_increase_per_minute_l15_1505

theorem water_temp_increase_per_minute :
  ∀ (initial_temp final_temp total_time pasta_time mixing_ratio : ℝ),
    initial_temp = 41 →
    final_temp = 212 →
    total_time = 73 →
    pasta_time = 12 →
    mixing_ratio = (1 / 3) →
    ((final_temp - initial_temp) / (total_time - pasta_time - (mixing_ratio * pasta_time)) = 3) :=
by
  intros initial_temp final_temp total_time pasta_time mixing_ratio
  sorry

end water_temp_increase_per_minute_l15_1505


namespace correct_statement_B_l15_1518

def flowchart_start_points : Nat := 1
def flowchart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

def program_flowchart_start_points : Nat := 1
def program_flowchart_end_points : Nat := 1

def structure_chart_start_points : Nat := 1
def structure_chart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

theorem correct_statement_B :
  (program_flowchart_start_points = 1 ∧ program_flowchart_end_points = 1) :=
by 
  sorry

end correct_statement_B_l15_1518


namespace trigonometric_identity_l15_1592

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 8 :=
by 
  sorry

end trigonometric_identity_l15_1592


namespace gcd_of_g_and_y_l15_1532

noncomputable def g (y : ℕ) := (3 * y + 5) * (8 * y + 3) * (16 * y + 9) * (y + 16)

theorem gcd_of_g_and_y (y : ℕ) (hy : y % 46896 = 0) : Nat.gcd (g y) y = 2160 :=
by
  -- Proof to be written here
  sorry

end gcd_of_g_and_y_l15_1532


namespace slower_bike_longer_time_by_1_hour_l15_1574

/-- Speed of the slower bike in kmph -/
def speed_slow : ℕ := 60

/-- Speed of the faster bike in kmph -/
def speed_fast : ℕ := 64

/-- Distance both bikes travel in km -/
def distance : ℕ := 960

/-- Time taken to travel the distance by a bike going at a certain speed -/
def time (speed : ℕ) : ℕ :=
  distance / speed

/-- Proof that the slower bike takes 1 hour longer to cover the distance compared to the faster bike -/
theorem slower_bike_longer_time_by_1_hour : 
  (time speed_slow) = (time speed_fast) + 1 := by
sorry

end slower_bike_longer_time_by_1_hour_l15_1574


namespace trapezoid_angles_l15_1598

-- Definition of the problem statement in Lean 4
theorem trapezoid_angles (A B C D : ℝ) (h1 : A = 60) (h2 : B = 130)
  (h3 : A + D = 180) (h4 : B + C = 180) (h_sum : A + B + C + D = 360) :
  C = 50 ∧ D = 120 :=
by
  sorry

end trapezoid_angles_l15_1598


namespace sturdy_square_impossible_l15_1559

def size : ℕ := 6
def dominos_used : ℕ := 18
def cells_per_domino : ℕ := 2
def total_cells : ℕ := size * size
def dividing_lines : ℕ := 10

def is_sturdy_square (grid_size : ℕ) (domino_count : ℕ) : Prop :=
  grid_size * grid_size = domino_count * cells_per_domino ∧ 
  ∀ line : ℕ, line < dividing_lines → ∃ domino : ℕ, domino < domino_count

theorem sturdy_square_impossible 
    (grid_size : ℕ) (domino_count : ℕ)
    (h1 : grid_size = size) (h2 : domino_count = dominos_used)
    (h3 : cells_per_domino = 2) (h4 : dividing_lines = 10) : 
  ¬ is_sturdy_square grid_size domino_count :=
by
  cases h1
  cases h2
  cases h3
  cases h4
  sorry

end sturdy_square_impossible_l15_1559


namespace intersection_M_N_l15_1541

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | x > 0 ∧ x < 2}

theorem intersection_M_N : M ∩ N = {1} :=
by {
  sorry
}

end intersection_M_N_l15_1541


namespace maximum_area_of_triangle_ABQ_l15_1573

open Real

structure Point3D where
  x : ℝ
  y : ℝ

def circle_C (Q : Point3D) : Prop := (Q.x - 3)^2 + (Q.y - 4)^2 = 4

def A := Point3D.mk 1 0
def B := Point3D.mk (-1) 0

noncomputable def area_triangle (P Q R : Point3D) : ℝ :=
  (1 / 2) * abs ((P.x * (Q.y - R.y)) + (Q.x * (R.y - P.y)) + (R.x * (P.y - Q.y)))

theorem maximum_area_of_triangle_ABQ : ∀ (Q : Point3D), circle_C Q → area_triangle A B Q ≤ 6 := by
  sorry

end maximum_area_of_triangle_ABQ_l15_1573


namespace fraction_of_time_spent_covering_initial_distance_l15_1538

variables (D T : ℝ) (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40)

theorem fraction_of_time_spent_covering_initial_distance (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40) :
  ((2 / 3) * D / 80) / T = 1 / 2 :=
by
  sorry

end fraction_of_time_spent_covering_initial_distance_l15_1538


namespace next_ten_winners_each_receive_160_l15_1564

def total_prize_money : ℕ := 2400

def first_winner_amount : ℕ := total_prize_money / 3

def remaining_amount : ℕ := total_prize_money - first_winner_amount

def each_of_ten_winners_receive : ℕ := remaining_amount / 10

theorem next_ten_winners_each_receive_160 : each_of_ten_winners_receive = 160 := by
  sorry

end next_ten_winners_each_receive_160_l15_1564


namespace sqrt_four_eq_plus_minus_two_l15_1553

theorem sqrt_four_eq_plus_minus_two : ∃ y : ℤ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  -- Proof goes here
  sorry

end sqrt_four_eq_plus_minus_two_l15_1553


namespace polynomial_difference_l15_1560

theorem polynomial_difference (a : ℝ) :
  (6 * a^2 - 5 * a + 3) - (5 * a^2 + 2 * a - 1) = a^2 - 7 * a + 4 :=
by
  sorry

end polynomial_difference_l15_1560


namespace augmented_matrix_solution_l15_1503

theorem augmented_matrix_solution (m n : ℝ) (x y : ℝ)
  (h1 : m * x = 6) (h2 : 3 * y = n) (hx : x = -3) (hy : y = 4) :
  m + n = 10 :=
by
  sorry

end augmented_matrix_solution_l15_1503


namespace price_of_basketball_l15_1561

-- Problem definitions based on conditions
def price_of_soccer_ball (x : ℝ) : Prop :=
  let price_of_basketball := 2 * x
  x + price_of_basketball = 186

theorem price_of_basketball (x : ℝ) (h : price_of_soccer_ball x) : 2 * x = 124 :=
by
  sorry

end price_of_basketball_l15_1561


namespace length_of_second_offset_l15_1572

theorem length_of_second_offset (d₁ d₂ h₁ A : ℝ) (h_d₁ : d₁ = 30) (h_h₁ : h₁ = 9) (h_A : A = 225):
  ∃ h₂, (A = (1/2) * d₁ * h₁ + (1/2) * d₁ * h₂) → h₂ = 6 := by
  sorry

end length_of_second_offset_l15_1572


namespace find_a_l15_1548

variable (a x y : ℝ)

theorem find_a (h1 : x / (2 * y) = 3 / 2) (h2 : (a * x + 6 * y) / (x - 2 * y) = 27) : a = 7 :=
sorry

end find_a_l15_1548


namespace duty_person_C_l15_1504

/-- Given amounts of money held by three persons and a total custom duty,
    prove that the duty person C should pay is 17 when payments are proportional. -/
theorem duty_person_C (money_A money_B money_C total_duty : ℕ) (total_money : ℕ)
  (hA : money_A = 560) (hB : money_B = 350) (hC : money_C = 180) (hD : total_duty = 100)
  (hT : total_money = money_A + money_B + money_C) :
  total_duty * money_C / total_money = 17 :=
by
  -- proof goes here
  sorry

end duty_person_C_l15_1504


namespace wire_pieces_difference_l15_1507

theorem wire_pieces_difference (L1 L2 : ℝ) (H1 : L1 = 14) (H2 : L2 = 16) : L2 - L1 = 2 :=
by
  rw [H1, H2]
  norm_num

end wire_pieces_difference_l15_1507


namespace band_member_earnings_l15_1535

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end band_member_earnings_l15_1535


namespace mark_egg_supply_in_a_week_l15_1588

def dozen := 12
def eggs_per_day_store1 := 5 * dozen
def eggs_per_day_store2 := 30
def daily_eggs_supplied := eggs_per_day_store1 + eggs_per_day_store2
def days_per_week := 7

theorem mark_egg_supply_in_a_week : daily_eggs_supplied * days_per_week = 630 := by
  sorry

end mark_egg_supply_in_a_week_l15_1588


namespace trajectory_of_C_l15_1563

-- Definitions of points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 3)

-- Definition of point C as a linear combination of points A and B
def C (α β : ℝ) : ℝ × ℝ := (α * A.1 + β * B.1, α * A.2 + β * B.2)

-- The main theorem statement to prove the equation of the trajectory of point C
theorem trajectory_of_C (x y α β : ℝ)
  (h_cond : α + β = 1)
  (h_C : (x, y) = C α β) : 
  x + 2*y = 5 := 
sorry -- Proof to be skipped

end trajectory_of_C_l15_1563


namespace keiko_speed_calc_l15_1555

noncomputable def keiko_speed (r : ℝ) (time_diff : ℝ) : ℝ :=
  let circumference_diff := 2 * Real.pi * 8
  circumference_diff / time_diff

theorem keiko_speed_calc (r : ℝ) (time_diff : ℝ) :
  keiko_speed r 48 = Real.pi / 3 := by
  sorry

end keiko_speed_calc_l15_1555


namespace isosceles_triangle_base_length_l15_1519

def is_isosceles (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_base_length
  (x y : ℝ)
  (h1 : 2 * x + 2 * y = 16)
  (h2 : 4^2 + y^2 = x^2)
  (h3 : is_isosceles x x (2 * y) ) :
  2 * y = 6 := 
by
  sorry

end isosceles_triangle_base_length_l15_1519


namespace rita_money_left_l15_1516

theorem rita_money_left :
  let initial_amount : ℝ := 400
  let cost_short_dresses : ℝ := 5 * (20 - 0.1 * 20)
  let cost_pants : ℝ := 2 * 15
  let cost_jackets : ℝ := 2 * (30 - 0.15 * 30) + 2 * 30
  let cost_skirts : ℝ := 2 * 18 * 0.8
  let cost_tshirts : ℝ := 2 * 8
  let cost_transportation : ℝ := 5
  let total_spent : ℝ := cost_short_dresses + cost_pants + cost_jackets + cost_skirts + cost_tshirts + cost_transportation
  let money_left : ℝ := initial_amount - total_spent
  money_left = 119.2 :=
by 
  sorry

end rita_money_left_l15_1516


namespace union_of_sets_l15_1529

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 3, 6}

theorem union_of_sets : A ∪ B = {1, 2, 3, 5, 6} :=
by sorry

end union_of_sets_l15_1529


namespace age_of_other_man_l15_1544

variables (A M : ℝ)

theorem age_of_other_man 
  (avg_age_of_men : ℝ)
  (replaced_man_age : ℝ)
  (avg_age_of_women : ℝ)
  (total_age_6_men : 6 * avg_age_of_men = 6 * (avg_age_of_men + 3) - replaced_man_age - M + 2 * avg_age_of_women) :
  M = 44 :=
by
  sorry

end age_of_other_man_l15_1544


namespace contrapositive_l15_1537

theorem contrapositive (p q : Prop) : (p → q) → (¬q → ¬p) :=
by
  sorry

end contrapositive_l15_1537


namespace runway_trip_time_l15_1530

-- Define the conditions
def num_models := 6
def num_bathing_suit_outfits := 2
def num_evening_wear_outfits := 3
def total_time_minutes := 60

-- Calculate the total number of outfits per model
def total_outfits_per_model := num_bathing_suit_outfits + num_evening_wear_outfits

-- Calculate the total number of runway trips
def total_runway_trips := num_models * total_outfits_per_model

-- State the goal: Time per runway trip
def time_per_runway_trip := total_time_minutes / total_runway_trips

theorem runway_trip_time : time_per_runway_trip = 2 := by
  sorry

end runway_trip_time_l15_1530


namespace sector_central_angle_l15_1542

theorem sector_central_angle (r l α : ℝ) 
  (h1 : 2 * r + l = 6) 
  (h2 : 0.5 * l * r = 2) :
  α = l / r → α = 4 ∨ α = 1 :=
sorry

end sector_central_angle_l15_1542


namespace gas_pressure_inversely_proportional_l15_1549

variable {T : Type} [Nonempty T]

theorem gas_pressure_inversely_proportional
  (P : T → ℝ) (V : T → ℝ)
  (h_inv : ∀ t, P t * V t = 24) -- Given that pressure * volume = k where k = 24
  (t₀ t₁ : T)
  (hV₀ : V t₀ = 3) (hP₀ : P t₀ = 8) -- Initial condition: volume = 3 liters, pressure = 8 kPa
  (hV₁ : V t₁ = 6) -- New condition: volume = 6 liters
  : P t₁ = 4 := -- We need to prove that the new pressure is 4 kPa
by 
  sorry

end gas_pressure_inversely_proportional_l15_1549


namespace emily_card_sequence_l15_1510

/--
Emily orders her playing cards continuously in the following sequence:
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, 3, ...

Prove that the 58th card in this sequence is 6.
-/
theorem emily_card_sequence :
  (58 % 13 = 6) := by
  -- The modulo operation determines the position of the card in the cycle
  sorry

end emily_card_sequence_l15_1510


namespace intersection_M_N_l15_1558

def M : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def N : Set ℝ := {x | -4 < x ∧ x ≤ 2}
def intersection : Set ℝ := {-1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l15_1558


namespace infection_equation_correct_l15_1554

theorem infection_equation_correct (x : ℝ) :
  1 + x + x * (x + 1) = 196 :=
sorry

end infection_equation_correct_l15_1554


namespace overtime_pay_rate_ratio_l15_1545

noncomputable def regular_pay_rate : ℕ := 3
noncomputable def regular_hours : ℕ := 40
noncomputable def total_pay : ℕ := 180
noncomputable def overtime_hours : ℕ := 10

theorem overtime_pay_rate_ratio : 
  (total_pay - (regular_hours * regular_pay_rate)) / overtime_hours / regular_pay_rate = 2 := by
  sorry

end overtime_pay_rate_ratio_l15_1545


namespace smallest_altitude_leq_three_l15_1576

theorem smallest_altitude_leq_three (a b c : ℝ) (r : ℝ) 
  (ha : a = max a (max b c)) 
  (r_eq : r = 1) 
  (area_eq : ∀ (S : ℝ), S = (a + b + c) / 2 ∧ S = a * h / 2) :
  ∃ h : ℝ, h ≤ 3 :=
by
  sorry

end smallest_altitude_leq_three_l15_1576


namespace solve_for_y_l15_1534

-- Define the condition
def condition (y : ℤ) : Prop := 7 - y = 13

-- Prove that if the condition is met, then y = -6
theorem solve_for_y (y : ℤ) (h : condition y) : y = -6 :=
by {
  sorry
}

end solve_for_y_l15_1534


namespace equivalent_expression_l15_1580

theorem equivalent_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
sorry

end equivalent_expression_l15_1580


namespace problem_statement_l15_1556

def A : ℕ := 9 * 10 * 10 * 5
def B : ℕ := 9 * 10 * 10 * 2 / 3

theorem problem_statement : A + B = 5100 := by
  sorry

end problem_statement_l15_1556


namespace jill_more_than_jake_l15_1589

-- Definitions from conditions
def jill_peaches := 12
def steven_peaches := jill_peaches + 15
def jake_peaches := steven_peaches - 16

-- Theorem to prove the question == answer given conditions
theorem jill_more_than_jake : jill_peaches - jake_peaches = 1 :=
by
  -- Proof steps would be here, but for the statement requirement we put sorry
  sorry

end jill_more_than_jake_l15_1589


namespace keith_spent_on_cards_l15_1582

theorem keith_spent_on_cards :
  let digimon_card_cost := 4.45
  let num_digimon_packs := 4
  let baseball_card_cost := 6.06
  let total_spent := num_digimon_packs * digimon_card_cost + baseball_card_cost
  total_spent = 23.86 :=
by
  sorry

end keith_spent_on_cards_l15_1582
