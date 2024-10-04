import Mathlib

namespace problem1_problem2_problem3_problem4_l141_141477

-- Problem 1
theorem problem1 : ∃ n : ℕ, n = 3^4 ∧ n = 81 :=
by
  sorry

-- Problem 2
theorem problem2 : ∃ n : ℕ, n = (Nat.choose 4 2) * 6 ∧ n = 36 :=
by
  sorry

-- Problem 3
theorem problem3 : ∃ n : ℕ, n = Nat.choose 4 2 ∧ n = 6 :=
by
  sorry

-- Problem 4
theorem problem4 : ∃ n : ℕ, n = 1 + (Nat.choose 4 1 + Nat.choose 4 2 / 2) + 6 ∧ n = 14 :=
by
  sorry

end problem1_problem2_problem3_problem4_l141_141477


namespace cost_in_chinese_yuan_l141_141783

theorem cost_in_chinese_yuan
  (usd_to_nad : ℝ := 8)
  (usd_to_cny : ℝ := 5)
  (sculpture_cost_nad : ℝ := 160) :
  sculpture_cost_nad / usd_to_nad * usd_to_cny = 100 := 
by
  sorry

end cost_in_chinese_yuan_l141_141783


namespace range_of_a_add_b_l141_141106

-- Define the problem and assumptions
variables (a b : ℝ)
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom ab_eq_a_add_b_add_3 : a * b = a + b + 3

-- Define the theorem to prove
theorem range_of_a_add_b : a + b ≥ 6 :=
sorry

end range_of_a_add_b_l141_141106


namespace base5_addition_correct_l141_141793

-- Definitions to interpret base-5 numbers
def base5_to_base10 (n : List ℕ) : ℕ :=
  n.reverse.foldl (λ acc d => acc * 5 + d) 0

-- Conditions given in the problem
def num1 : ℕ := base5_to_base10 [2, 0, 1, 4]  -- (2014)_5 in base-10
def num2 : ℕ := base5_to_base10 [2, 2, 3]    -- (223)_5 in base-10

-- Statement to prove
theorem base5_addition_correct :
  base5_to_base10 ([2, 0, 1, 4]) + base5_to_base10 ([2, 2, 3]) = base5_to_base10 ([2, 2, 4, 2]) :=
by
  -- Proof goes here
  sorry

#print axioms base5_addition_correct

end base5_addition_correct_l141_141793


namespace river_depth_conditions_l141_141337

noncomputable def depth_beginning_may : ℝ := 15
noncomputable def depth_increase_june : ℝ := 11.25

theorem river_depth_conditions (d k : ℝ)
  (h1 : ∃ d, d = depth_beginning_may) 
  (h2 : 1.5 * d + k = 45)
  (h3 : k = 0.75 * d) :
  d = depth_beginning_may ∧ k = depth_increase_june :=
by
  have H : d = 15 := sorry
  have K : k = 11.25 := sorry
  exact ⟨H, K⟩

end river_depth_conditions_l141_141337


namespace even_three_digit_numbers_sum_tens_units_14_l141_141733

theorem even_three_digit_numbers_sum_tens_units_14 : 
  ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n % 2 = 0) ∧
  (let t := (n / 10) % 10 in let u := n % 10 in t + u = 14) ∧
  n = 18 := sorry

end even_three_digit_numbers_sum_tens_units_14_l141_141733


namespace prob_two_hits_is_correct_l141_141555

section ring_toss_game

def prob_hit_M : ℝ := 3 / 4
def prob_hit_N : ℝ := 2 / 3
def prob_miss_M : ℝ := 1 - prob_hit_M
def prob_miss_N : ℝ := 1 - prob_hit_N

def scenario1_prob : ℝ := prob_hit_M * (2 * prob_hit_N * prob_miss_N)
def scenario2_prob : ℝ := prob_miss_M * (prob_hit_N * prob_hit_N)

def prob_hit_two_times : ℝ := scenario1_prob + scenario2_prob

theorem prob_two_hits_is_correct : prob_hit_two_times = 4 / 9 := by
  sorry

end ring_toss_game

end prob_two_hits_is_correct_l141_141555


namespace a14_eq_33_l141_141126

variable {a : ℕ → ℝ}
variables (d : ℝ) (a1 : ℝ)

-- Defining the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℝ := a1 + n * d

-- Given conditions
axiom a5_eq_6 : arithmetic_sequence 4 = 6
axiom a8_eq_15 : arithmetic_sequence 7 = 15

-- Theorem statement
theorem a14_eq_33 : arithmetic_sequence 13 = 33 :=
by
  -- Proof skipped
  sorry

end a14_eq_33_l141_141126


namespace remainder_product_div_6_l141_141093

theorem remainder_product_div_6 :
  (3 * 7 * 13 * 17 * 23 * 27 * 33 * 37 * 43 * 47 * 53 * 57 * 63 * 67 * 73 * 77 * 83 * 87 * 93 * 97 
   * 103 * 107 * 113 * 117 * 123 * 127 * 133 * 137 * 143 * 147 * 153 * 157 * 163 * 167 * 173 
   * 177 * 183 * 187 * 193 * 197) % 6 = 3 := 
by 
  -- basic info about modulo arithmetic and properties of sequences
  sorry

end remainder_product_div_6_l141_141093


namespace cos_alpha_minus_pi_l141_141098

theorem cos_alpha_minus_pi (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 4) : 
  Real.cos (α - Real.pi) = -5 / 8 :=
sorry

end cos_alpha_minus_pi_l141_141098


namespace ammonium_iodide_requirement_l141_141528

theorem ammonium_iodide_requirement :
  ∀ (NH4I KOH NH3 KI H2O : ℕ),
  (NH4I + KOH = NH3 + KI + H2O) → 
  (NH4I = 3) →
  (KOH = 3) →
  (NH3 = 3) →
  (KI = 3) →
  (H2O = 3) →
  NH4I = 3 :=
by
  intros NH4I KOH NH3 KI H2O reaction_balanced NH4I_req KOH_req NH3_prod KI_prod H2O_prod
  exact NH4I_req

end ammonium_iodide_requirement_l141_141528


namespace smallest_base10_integer_l141_141658

theorem smallest_base10_integer :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (2 * a + 2 = 3 * b + 3) ∧ (2 * a + 2 = 18) :=
by
  existsi 8 -- assign specific solutions to a
  existsi 5 -- assign specific solutions to b
  exact sorry -- follows from the validations done above

end smallest_base10_integer_l141_141658


namespace inverse_prop_l141_141289

theorem inverse_prop (x : ℝ) : x < 0 → x^2 > 0 :=
by
  sorry

end inverse_prop_l141_141289


namespace rectangle_length_l141_141314

theorem rectangle_length (P W : ℝ) (hP : P = 40) (hW : W = 8) : ∃ L : ℝ, 2 * (L + W) = P ∧ L = 12 := 
by 
  sorry

end rectangle_length_l141_141314


namespace pow_mult_same_base_l141_141836

theorem pow_mult_same_base (a b : ℕ) : 10^a * 10^b = 10^(a + b) := by 
  sorry

example : 10^655 * 10^652 = 10^1307 :=
  pow_mult_same_base 655 652

end pow_mult_same_base_l141_141836


namespace relationship_among_a_b_c_l141_141995

noncomputable def a : ℝ := Real.log 2 / Real.log 0.3
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.4 * Real.log 0.3)

theorem relationship_among_a_b_c : a < c ∧ c < b := by
  sorry

end relationship_among_a_b_c_l141_141995


namespace sin_cos_product_l141_141534

variable (α : ℝ)

theorem sin_cos_product (h : cos α - sin α = 1 / 2) : sin α * cos α = 3 / 8 := by
  sorry

end sin_cos_product_l141_141534


namespace area_enclosed_by_trajectory_of_P_l141_141265

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Definition of fixed points A and B
def A : Point := { x := -3, y := 0 }
def B : Point := { x := 3, y := 0 }

-- Condition for the ratio of distances
def ratio_condition (P : Point) : Prop :=
  ((P.x + 3)^2 + P.y^2) / ((P.x - 3)^2 + P.y^2) = 1 / 4

-- Definition of a circle based on the derived condition in the solution
def circle_eq (P : Point) : Prop :=
  (P.x + 5)^2 + P.y^2 = 16

-- Theorem stating the area enclosed by the trajectory of point P is 16π
theorem area_enclosed_by_trajectory_of_P : 
  (∀ P : Point, ratio_condition P → circle_eq P) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  sorry

end area_enclosed_by_trajectory_of_P_l141_141265


namespace apples_problem_l141_141902

variable (K A : ℕ)

theorem apples_problem (K A : ℕ) (h1 : K + (3 / 4) * K + 600 = 2600) (h2 : A + (3 / 4) * A + 600 = 2600) :
  K = 1142 ∧ A = 1142 :=
by
  sorry

end apples_problem_l141_141902


namespace pencil_length_eq_eight_l141_141117

theorem pencil_length_eq_eight (L : ℝ) 
  (h1 : (1/8) * L + (1/2) * ((7/8) * L) + (7/2) = L) : 
  L = 8 :=
by
  sorry

end pencil_length_eq_eight_l141_141117


namespace perimeter_correct_l141_141411

open EuclideanGeometry

noncomputable def perimeter_of_figure : ℝ := 
  let AB : ℝ := 6
  let BC : ℝ := AB
  let AD : ℝ := AB / 2
  let DC : ℝ := AD
  let DE : ℝ := AD
  let EA : ℝ := DE
  let EF : ℝ := EA / 2
  let FG : ℝ := EF
  let GH : ℝ := FG / 2
  let HJ : ℝ := GH
  let JA : ℝ := HJ
  AB + BC + DC + DE + EF + FG + GH + HJ + JA

theorem perimeter_correct : perimeter_of_figure = 23.25 :=
by
  -- proof steps would go here, but are not required for this problem transformation
  sorry

end perimeter_correct_l141_141411


namespace factorize_x_squared_sub_xy_l141_141086

theorem factorize_x_squared_sub_xy (x y : ℝ) : x^2 - x * y = x * (x - y) :=
sorry

end factorize_x_squared_sub_xy_l141_141086


namespace jezebel_total_flower_cost_l141_141245

theorem jezebel_total_flower_cost :
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  (red_rose_count * red_rose_cost + sunflower_count * sunflower_cost = 45) :=
by
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  sorry

end jezebel_total_flower_cost_l141_141245


namespace parabola_has_one_x_intercept_l141_141385

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end parabola_has_one_x_intercept_l141_141385


namespace shaded_area_first_quadrant_l141_141302

/-- Two concentric circles with radii 15 and 9. Prove that the area of the shaded region 
within the first quadrant is 36π. -/
theorem shaded_area_first_quadrant (r_big r_small : ℝ) (h_big : r_big = 15) (h_small : r_small = 9) : 
  (π * (r_big ^ 2 - r_small ^ 2)) / 4 = 36 * π := 
by
  sorry

end shaded_area_first_quadrant_l141_141302


namespace compare_subtract_one_l141_141737

theorem compare_subtract_one (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end compare_subtract_one_l141_141737


namespace calculate_x_l141_141270

variable (a b x : ℝ)
variable (h1 : r = (3 * a) ^ (3 * b))
variable (h2 : r = a ^ b * x ^ b)
variable (h3 : x > 0)

theorem calculate_x (a b x : ℝ) (h1 : r = (3 * a) ^ (3 * b)) (h2 : r = a ^ b * x ^ b) (h3 : x > 0) : x = 27 * a ^ 2 := by
  sorry

end calculate_x_l141_141270


namespace proof_product_eq_l141_141174

theorem proof_product_eq (a b c d : ℚ) (h1 : 2 * a + 3 * b + 5 * c + 7 * d = 42)
    (h2 : 4 * (d + c) = b) (h3 : 2 * b + 2 * c = a) (h4 : c - 2 = d) :
    a * b * c * d = -26880 / 729 := by
  sorry

end proof_product_eq_l141_141174


namespace segment_distance_sum_l141_141239

theorem segment_distance_sum
  (AB_len : ℝ) (A'B'_len : ℝ) (D_midpoint : AB_len / 2 = 4)
  (D'_midpoint : A'B'_len / 2 = 6) (x : ℝ) (y : ℝ)
  (x_val : x = 3) :
  x + y = 10 :=
by sorry

end segment_distance_sum_l141_141239


namespace brianna_more_chocolates_than_alix_l141_141430

def Nick_ClosetA : ℕ := 10
def Nick_ClosetB : ℕ := 6
def Alix_ClosetA : ℕ := 3 * Nick_ClosetA
def Alix_ClosetB : ℕ := 3 * Nick_ClosetA
def Mom_Takes_From_AlixA : ℚ := (1/4:ℚ) * Alix_ClosetA
def Brianna_ClosetA : ℚ := 2 * (Nick_ClosetA + Alix_ClosetA - Mom_Takes_From_AlixA)
def Brianna_ClosetB_after : ℕ := 18
def Brianna_ClosetB : ℚ := Brianna_ClosetB_after / (0.8:ℚ)

def Brianna_Total : ℚ := Brianna_ClosetA + Brianna_ClosetB
def Alix_Total : ℚ := Alix_ClosetA + Alix_ClosetB
def Difference : ℚ := Brianna_Total - Alix_Total

theorem brianna_more_chocolates_than_alix : Difference = 35 := by
  sorry

end brianna_more_chocolates_than_alix_l141_141430


namespace batteries_manufactured_l141_141486

theorem batteries_manufactured (gather_time create_time : Nat) (robots : Nat) (hours : Nat) (total_batteries : Nat) :
  gather_time = 6 →
  create_time = 9 →
  robots = 10 →
  hours = 5 →
  total_batteries = (hours * 60 / (gather_time + create_time)) * robots →
  total_batteries = 200 :=
by
  intros h_gather h_create h_robots h_hours h_batteries
  simp [h_gather, h_create, h_robots, h_hours] at h_batteries
  exact h_batteries

end batteries_manufactured_l141_141486


namespace coordinates_P_correct_l141_141757

noncomputable def coordinates_of_P : ℝ × ℝ :=
  let x_distance_to_y_axis : ℝ := 5
  let y_distance_to_x_axis : ℝ := 4
  -- x-coordinate must be negative, y-coordinate must be positive
  let x_coord : ℝ := -x_distance_to_y_axis
  let y_coord : ℝ := y_distance_to_x_axis
  (x_coord, y_coord)

theorem coordinates_P_correct:
  coordinates_of_P = (-5, 4) :=
by
  sorry

end coordinates_P_correct_l141_141757


namespace arithmetic_mean_eq_l141_141651

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l141_141651


namespace general_form_line_eq_line_passes_fixed_point_l141_141101

-- (Ⅰ) Prove that if m = 1/2 and point P (1/2, 2), the general form equation of line l is 2x - y + 1 = 0
theorem general_form_line_eq (m n : ℝ) (h1 : m = 1/2) (h2 : n = 1 / (1 - m)) (h3 : n = 2) (P : (ℝ × ℝ)) (hP : P = (1/2, 2)) :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 1 := sorry

-- (Ⅱ) Prove that if point P(m,n) is on the line l0, then the line mx + (n-1)y + n + 5 = 0 passes through a fixed point, coordinates (1,1)
theorem line_passes_fixed_point (m n : ℝ) (h1 : m + 2 * n + 4 = 0) :
  ∀ (x y : ℝ), (m * x + (n - 1) * y + n + 5 = 0) ↔ (x = 1) ∧ (y = 1) := sorry

end general_form_line_eq_line_passes_fixed_point_l141_141101


namespace arithmetic_mean_of_fractions_l141_141648

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l141_141648


namespace find_a5_l141_141553

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: The sum of the first n terms of the sequence {a_n} is represented by S_n = 2a_n - 1 (n ∈ ℕ)
axiom sum_of_terms (n : ℕ) : S n = 2 * (a n) - 1

-- Prove that a_5 = 16
theorem find_a5 : a 5 = 16 :=
  sorry

end find_a5_l141_141553


namespace incenter_ineq_l141_141874

open Real

-- Definitions of the incenter and angle bisector intersection points
def incenter (A B C : Point) : Point := sorry
def angle_bisector_intersect (A B C I : Point) (angle_vertex : Point) : Point := sorry
def AI (A I : Point) : ℝ := sorry
def AA' (A A' : Point) : ℝ := sorry
def BI (B I : Point) : ℝ := sorry
def BB' (B B' : Point) : ℝ := sorry
def CI (C I : Point) : ℝ := sorry
def CC' (C C' : Point) : ℝ := sorry

-- Statement of the problem
theorem incenter_ineq 
    (A B C I A' B' C' : Point)
    (h1 : I = incenter A B C)
    (h2 : A' = angle_bisector_intersect A B C I A)
    (h3 : B' = angle_bisector_intersect A B C I B)
    (h4 : C' = angle_bisector_intersect A B C I C) :
    (1/4 : ℝ) < (AI A I * BI B I * CI C I) / (AA' A A' * BB' B B' * CC' C C') ∧ 
    (AI A I * BI B I * CI C I) / (AA' A A' * BB' B B' * CC' C C') ≤ (8/27 : ℝ) :=
sorry

end incenter_ineq_l141_141874


namespace incorrect_statements_are_1_2_4_l141_141826

theorem incorrect_statements_are_1_2_4:
    let statements := ["Inductive reasoning and analogical reasoning both involve reasoning from specific to general.",
                       "When making an analogy, it is more appropriate to use triangles in a plane and parallelepipeds in space as the objects of analogy.",
                       "'All multiples of 9 are multiples of 3, if a number m is a multiple of 9, then m must be a multiple of 3' is an example of syllogistic reasoning.",
                       "In deductive reasoning, as long as it follows the form of deductive reasoning, the conclusion is always correct."]
    let incorrect_statements := {1, 2, 4}
    incorrect_statements = {i | i ∈ [1, 2, 3, 4] ∧
                             ((i = 1 → ¬(∃ s, s ∈ statements ∧ s = statements[0])) ∧ 
                              (i = 2 → ¬(∃ s, s ∈ statements ∧ s = statements[1])) ∧ 
                              (i = 3 → ∃ s, s ∈ statements ∧ s = statements[2]) ∧ 
                              (i = 4 → ¬(∃ s, s ∈ statements ∧ s = statements[3])))} :=
by
  sorry

end incorrect_statements_are_1_2_4_l141_141826


namespace find_line_equation_l141_141090

variable (x y : ℝ)

theorem find_line_equation (hx : x = -5) (hy : y = 2)
  (line_through_point : ∃ a b c : ℝ, a * x + b * y + c = 0)
  (x_intercept_twice_y_intercept : ∀ a b c : ℝ, c ≠ 0 → b ≠ 0 → (a / c) = 2 * (c / b)) :
  ∃ a b c : ℝ, (a * x + b * y + c = 0 ∧ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 1 ∧ b = 2 ∧ c = 1)) :=
sorry

end find_line_equation_l141_141090


namespace tangent_lengths_identity_l141_141770

theorem tangent_lengths_identity
  (a b c BC AC AB : ℝ)
  (sqrt_a sqrt_b sqrt_c : ℝ)
  (h1 : sqrt_a^2 = a)
  (h2 : sqrt_b^2 = b)
  (h3 : sqrt_c^2 = c) :
  a * BC + c * AB - b * AC = BC * AC * AB :=
sorry

end tangent_lengths_identity_l141_141770


namespace line_equation_through_point_with_intercepts_conditions_l141_141089

theorem line_equation_through_point_with_intercepts_conditions :
  ∃ (a b : ℚ) (m c : ℚ), 
    (-5) * m + c = 2 ∧ -- The line passes through A(-5, 2)
    a = 2 * b ∧       -- x-intercept is twice the y-intercept
    (a * m + c = 0 ∨ ((1/m)*a + (1/m)^2 * c+1 = 0)) :=         -- Equations of the line
sorry

end line_equation_through_point_with_intercepts_conditions_l141_141089


namespace sum_of_reflection_midpoint_coordinates_l141_141914

theorem sum_of_reflection_midpoint_coordinates (P R : ℝ × ℝ) (M : ℝ × ℝ) (P' R' M' : ℝ × ℝ) :
  P = (2, 1) → R = (12, 15) → 
  M = ((P.fst + R.fst) / 2, (P.snd + R.snd) / 2) →
  P' = (-P.fst, P.snd) → R' = (-R.fst, R.snd) →
  M' = ((P'.fst + R'.fst) / 2, (P'.snd + R'.snd) / 2) →
  (M'.fst + M'.snd) = 1 := 
by 
  intros
  sorry

end sum_of_reflection_midpoint_coordinates_l141_141914


namespace solve_inequality_l141_141921

noncomputable def inequality_statement (x : ℝ) : Prop :=
  2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 20

theorem solve_inequality (x : ℝ) :
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) →
  (inequality_statement x ↔ (x < -3 ∨ (-2 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ 8 < x)) :=
by sorry

end solve_inequality_l141_141921


namespace A_finishes_job_in_12_days_l141_141827

variable (A B : ℝ)

noncomputable def work_rate_A_and_B := (1 / 40)
noncomputable def work_rate_A := (1 / A)
noncomputable def work_rate_B := (1 / B)

theorem A_finishes_job_in_12_days
  (h1 : work_rate_A + work_rate_B = work_rate_A_and_B)
  (h2 : 10 * work_rate_A_and_B = 1 / 4)
  (h3 : 9 * work_rate_A = 3 / 4) :
  A = 12 :=
  sorry

end A_finishes_job_in_12_days_l141_141827


namespace trains_meet_in_32_seconds_l141_141039

noncomputable def length_first_train : ℕ := 400
noncomputable def length_second_train : ℕ := 200
noncomputable def initial_distance : ℕ := 200

noncomputable def speed_first_train : ℕ := 15
noncomputable def speed_second_train : ℕ := 10

noncomputable def relative_speed : ℕ := speed_first_train + speed_second_train
noncomputable def total_distance : ℕ := length_first_train + length_second_train + initial_distance
noncomputable def time_to_meet := total_distance / relative_speed

theorem trains_meet_in_32_seconds : time_to_meet = 32 := by
  sorry

end trains_meet_in_32_seconds_l141_141039


namespace percent_of_a_is_b_l141_141233

variable {a b c : ℝ}

theorem percent_of_a_is_b (h1 : c = 0.25 * a) (h2 : c = 0.10 * b) : b = 2.5 * a :=
by sorry

end percent_of_a_is_b_l141_141233


namespace max_value_of_trig_expression_l141_141357

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end max_value_of_trig_expression_l141_141357


namespace area_in_sq_yds_l141_141328

-- Definitions based on conditions
def side_length_ft : ℕ := 9
def sq_ft_per_sq_yd : ℕ := 9

-- Statement to prove
theorem area_in_sq_yds : (side_length_ft * side_length_ft) / sq_ft_per_sq_yd = 9 :=
by
  sorry

end area_in_sq_yds_l141_141328


namespace margo_total_distance_l141_141568

theorem margo_total_distance (time_to_friend : ℝ) (time_back_home : ℝ) (average_rate : ℝ)
  (total_time_hours : ℝ) (total_miles : ℝ) :
  time_to_friend = 12 / 60 ∧
  time_back_home = 24 / 60 ∧
  total_time_hours = (12 / 60) + (24 / 60) ∧
  average_rate = 3 ∧
  total_miles = average_rate * total_time_hours →
  total_miles = 1.8 :=
by
  sorry

end margo_total_distance_l141_141568


namespace neg_p_equiv_l141_141110

theorem neg_p_equiv :
  (¬ (∀ x : ℝ, x > 0 → x - Real.log x > 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0 - Real.log x_0 ≤ 0) :=
by
  sorry

end neg_p_equiv_l141_141110


namespace katya_solves_enough_l141_141261

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l141_141261


namespace solve_for_x_l141_141919

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l141_141919


namespace ellipse_standard_equation_and_line_l141_141726

theorem ellipse_standard_equation_and_line (a b c : ℝ) (P : ℝ × ℝ)
    (h1: a = 2 * c) (h2: 2 * b = 2 * sqrt 3) (h3 : a^2 = b^2 + c^2) (h4: P = (0, 2))
    : ( ∃ eq : Prop, eq = (∃ x y : ℝ, x^2 / 4 + y^2 / 3 = 1) ) ∧ 
      ( ∃ k : ℝ, (1 + (k^2)) * x^2 + 4 * k * 2 * x + 4 = 0 ∧ 
          (16 - 12 * k^2) / (3 + 4 * k^2) = 2 ∧ 
          (y = k * x + 2) ) :=
by
  sorry

end ellipse_standard_equation_and_line_l141_141726


namespace lcm_12_18_l141_141976

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l141_141976


namespace part1_union_part1_complement_part2_intersect_l141_141217

namespace MathProof

open Set Real

def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }
def R : Set ℝ := univ  -- the set of all real numbers

theorem part1_union :
  A ∪ B = { x | 1 ≤ x ∧ x < 10 } :=
sorry

theorem part1_complement :
  R \ B = { x | x ≤ 2 ∨ x ≥ 10 } :=
sorry

theorem part2_intersect (a : ℝ) :
  (A ∩ C a ≠ ∅) → a > 1 :=
sorry

end MathProof

end part1_union_part1_complement_part2_intersect_l141_141217


namespace distance_to_Tianbo_Mountain_l141_141082

theorem distance_to_Tianbo_Mountain : ∀ (x y : ℝ), 
  (x ≠ 0) ∧ 
  (y = 3) ∧ 
  (∀ v, v = (4 * y + x) * ((2 * x - 8) / v)) ∧ 
  (2 * (y * x) = 8 * y + x^2 - 4 * x) 
  → 
  (x + y = 9) := 
by
  sorry

end distance_to_Tianbo_Mountain_l141_141082


namespace find_original_six_digit_number_l141_141484

theorem find_original_six_digit_number (N x y : ℕ) (h1 : N = 10 * x + y) (h2 : N - x = 654321) (h3 : 0 ≤ y ∧ y ≤ 9) :
  N = 727023 :=
sorry

end find_original_six_digit_number_l141_141484


namespace find_x_in_isosceles_triangle_l141_141123

def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)

def triangle_inequality (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem find_x_in_isosceles_triangle (x : ℝ) :
  is_isosceles (x + 3) (2 * x + 1) 11 ∧ triangle_inequality (x + 3) (2 * x + 1) 11 →
  (x = 8) ∨ (x = 5) :=
sorry

end find_x_in_isosceles_triangle_l141_141123


namespace remainder_of_12_pow_2012_mod_5_l141_141092

theorem remainder_of_12_pow_2012_mod_5 : (12 ^ 2012) % 5 = 1 :=
by
  sorry

end remainder_of_12_pow_2012_mod_5_l141_141092


namespace system1_solution_system2_solution_l141_141797

-- System (1)
theorem system1_solution (x y : ℝ) (h1 : x + y = 1) (h2 : 3 * x + y = 5) : x = 2 ∧ y = -1 := sorry

-- System (2)
theorem system2_solution (x y : ℝ) (h1 : 3 * (x - 1) + 4 * y = 1) (h2 : 2 * x + 3 * (y + 1) = 2) : x = 16 ∧ y = -11 := sorry

end system1_solution_system2_solution_l141_141797


namespace num_positive_solutions_eq_32_l141_141201

theorem num_positive_solutions_eq_32 : 
  ∃ n : ℕ, (∀ x y : ℕ, 4 * x + 7 * y = 888 → x > 0 ∧ y > 0) ∧ n = 32 :=
sorry

end num_positive_solutions_eq_32_l141_141201


namespace tangent_line_at_origin_l141_141208

/-- 
The curve is given by y = exp x.
The tangent line to this curve that passes through the origin (0, 0) 
has the equation y = exp 1 * x.
-/
theorem tangent_line_at_origin :
  ∀ (x y : ℝ), y = Real.exp x → (∃ k : ℝ, ∀ x, y = k * x ∧ k = Real.exp 1) :=
by
  sorry

end tangent_line_at_origin_l141_141208


namespace whales_last_year_eq_4000_l141_141296

variable (W : ℕ) (last_year this_year next_year : ℕ)

theorem whales_last_year_eq_4000
    (h1 : this_year = 2 * last_year)
    (h2 : next_year = this_year + 800)
    (h3 : next_year = 8800) :
    last_year = 4000 := by
  sorry

end whales_last_year_eq_4000_l141_141296


namespace quadratic_root_l141_141120

theorem quadratic_root (a b c : ℝ) (h : 9 * a - 3 * b + c = 0) : 
  a * (-3)^2 + b * (-3) + c = 0 :=
by
  sorry

end quadratic_root_l141_141120


namespace solve_equation_l141_141439

theorem solve_equation : ∀ x : ℝ, 2 * (3 * x - 1) = 7 - (x - 5) → x = 2 :=
by
  intro x h
  sorry

end solve_equation_l141_141439


namespace game_result_l141_141007

def g (m : ℕ) : ℕ :=
  if m % 3 = 0 then 8
  else if m = 2 ∨ m = 3 ∨ m = 5 then 3
  else if m % 2 = 0 then 1
  else 0

def jack_sequence : List ℕ := [2, 5, 6, 4, 3]
def jill_sequence : List ℕ := [1, 6, 3, 2, 5]

def calculate_score (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc + g x) 0

theorem game_result : calculate_score jack_sequence * calculate_score jill_sequence = 420 :=
by
  sorry

end game_result_l141_141007


namespace total_pure_acid_in_mixture_l141_141002

-- Definitions of the conditions
def solution1_volume : ℝ := 8
def solution1_concentration : ℝ := 0.20
def solution2_volume : ℝ := 5
def solution2_concentration : ℝ := 0.35

-- Proof statement
theorem total_pure_acid_in_mixture :
  solution1_volume * solution1_concentration + solution2_volume * solution2_concentration = 3.35 := by
  sorry

end total_pure_acid_in_mixture_l141_141002


namespace proposition_1_proposition_4_l141_141704

-- Definitions
variable {a b c : Type} (Line : Type) (Plane : Type)
variable (a b c : Line) (γ : Plane)

-- Given conditions
variable (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Propositions to prove
theorem proposition_1 (H1 : parallel a b) (H2 : parallel b c) : parallel a c := sorry

theorem proposition_4 (H3 : perpendicular a γ) (H4 : perpendicular b γ) : parallel a b := sorry

end proposition_1_proposition_4_l141_141704


namespace correct_operation_l141_141051

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l141_141051


namespace isosceles_triangle_angle_l141_141508

theorem isosceles_triangle_angle (A B C : ℝ) (h_iso : A = C)
  (h_obtuse : B = 1.4 * 90) (h_sum : A + B + C = 180) :
  A = 27 :=
by
  have h1 : B = 126 from h_obtuse
  have h2 : A + C = 54 := by linarith [h1, h_sum]
  have h3 : 2 * A = 54 := by linarith [h_iso, h2]
  exact eq_div_of_mul_eq two_ne_zero h3

end isosceles_triangle_angle_l141_141508


namespace ratio_of_combined_area_to_combined_perimeter_l141_141303

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def equilateral_triangle_perimeter (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_combined_area_to_combined_perimeter :
  (equilateral_triangle_area 6 + equilateral_triangle_area 8) / 
  (equilateral_triangle_perimeter 6 + equilateral_triangle_perimeter 8) = (25 * Real.sqrt 3) / 42 :=
by
  sorry

end ratio_of_combined_area_to_combined_perimeter_l141_141303


namespace arithmetic_and_geometric_mean_l141_141144

theorem arithmetic_and_geometric_mean (a b : ℝ) (h1 : a + b = 40) (h2 : a * b = 100) : a^2 + b^2 = 1400 := by
  sorry

end arithmetic_and_geometric_mean_l141_141144


namespace hannah_jerry_difference_l141_141732

-- Define the calculations of Hannah (H) and Jerry (J)
def H : Int := 10 - (3 * 4)
def J : Int := 10 - 3 + 4

-- Prove that H - J = -13
theorem hannah_jerry_difference : H - J = -13 := by
  sorry

end hannah_jerry_difference_l141_141732


namespace value_of_f_sin_7pi_over_6_l141_141376

def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x

theorem value_of_f_sin_7pi_over_6 :
  f (Real.sin (7 * Real.pi / 6)) = 0 :=
by
  sorry

end value_of_f_sin_7pi_over_6_l141_141376


namespace find_pairs_l141_141567

theorem find_pairs (a b : ℕ) (h : a + b + a * b = 1000) : 
  (a = 6 ∧ b = 142) ∨ (a = 142 ∧ b = 6) ∨
  (a = 10 ∧ b = 90) ∨ (a = 90 ∧ b = 10) ∨
  (a = 12 ∧ b = 76) ∨ (a = 76 ∧ b = 12) :=
by sorry

end find_pairs_l141_141567


namespace bug_paths_in_hypercube_l141_141904

theorem bug_paths_in_hypercube :
  let vertices := Fin 2 → Fin 4
  ∃ (f : Fin 4 → vertices), 
    f 0 = (λ i, 0) ∧
    f 4 = (λ i, 1) ∧ 
    (∀ i, ∃ j : Fin 4, f (i + 1) = λ k, if j = k then 1 else f i k) ∧
    (∏ i, f (4 - i) = 4!) :=
by 
  -- declaration of vertices
  sorry

end bug_paths_in_hypercube_l141_141904


namespace alyssa_final_money_l141_141694

-- Definitions based on conditions
def weekly_allowance : Int := 8
def spent_on_movies : Int := weekly_allowance / 2
def earnings_from_washing_car : Int := 8

-- The statement to prove
def final_amount : Int := (weekly_allowance - spent_on_movies) + earnings_from_washing_car

-- The theorem expressing the problem
theorem alyssa_final_money : final_amount = 12 := by
  sorry

end alyssa_final_money_l141_141694


namespace gcd_m_n_is_one_l141_141820

/-- Definition of m -/
def m : ℕ := 130^2 + 241^2 + 352^2

/-- Definition of n -/
def n : ℕ := 129^2 + 240^2 + 353^2 + 2^3

/-- Proof statement: The greatest common divisor of m and n is 1 -/
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l141_141820


namespace arithmetic_mean_eq_l141_141653

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l141_141653


namespace number_of_girls_l141_141460

variable (total_children : ℕ) (boys : ℕ)

theorem number_of_girls (h1 : total_children = 117) (h2 : boys = 40) : 
  total_children - boys = 77 := by
  sorry

end number_of_girls_l141_141460


namespace homework_total_time_l141_141434

theorem homework_total_time :
  ∀ (j g p : ℕ),
  j = 18 →
  g = j - 6 →
  p = 2 * g - 4 →
  j + g + p = 50 :=
by
  intros j g p h1 h2 h3
  sorry

end homework_total_time_l141_141434


namespace polynomial_roots_l141_141361

theorem polynomial_roots :
  ∀ x : ℝ, (4 * x^4 - 28 * x^3 + 53 * x^2 - 28 * x + 4 = 0) ↔ (x = 4 ∨ x = 2 ∨ x = 1/4 ∨ x = 1/2) := 
by
  sorry

end polynomial_roots_l141_141361


namespace fraction_meaningful_cond_l141_141168

theorem fraction_meaningful_cond (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) := 
by
  sorry

end fraction_meaningful_cond_l141_141168


namespace triangle_neg3_4_l141_141710

def triangle (a b : ℚ) : ℚ := -a + b

theorem triangle_neg3_4 : triangle (-3) 4 = 7 := 
by 
  sorry

end triangle_neg3_4_l141_141710


namespace sum_of_coefficients_no_y_l141_141929

-- Defining the problem conditions
def expansion (a b c : ℤ) (n : ℕ) : ℤ := (a - b + c)^n

-- Summing the coefficients of the terms that do not contain y
noncomputable def coefficients_sum (a b : ℤ) (n : ℕ) : ℤ :=
  (a - b)^n

theorem sum_of_coefficients_no_y (n : ℕ) (h : 0 < n) : 
  coefficients_sum 4 3 n = 1 :=
by
  sorry

end sum_of_coefficients_no_y_l141_141929


namespace tom_tickets_l141_141962

theorem tom_tickets :
  let tickets_whack_a_mole := 32
  let tickets_skee_ball := 25
  let tickets_spent_on_hat := 7
  let total_tickets := tickets_whack_a_mole + tickets_skee_ball
  let tickets_left := total_tickets - tickets_spent_on_hat
  tickets_left = 50 :=
by
  sorry

end tom_tickets_l141_141962


namespace find_y_l141_141662

theorem find_y (x y : ℕ) (h1 : x % y = 7) (h2 : (x : ℚ) / y = 86.1) (h3 : Nat.Prime (x + y)) : y = 70 :=
sorry

end find_y_l141_141662


namespace find_range_of_m_l141_141881

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 7
def B (m x : ℝ) : Prop := m + 1 < x ∧ x < 2 * m - 1

theorem find_range_of_m (m : ℝ) : 
  (∀ x, B m x → A x) ∧ (∃ x, B m x) → 2 < m ∧ m ≤ 4 :=
by
  sorry

end find_range_of_m_l141_141881


namespace benny_missed_games_l141_141342

theorem benny_missed_games (total_games attended_games missed_games : ℕ)
  (H1 : total_games = 39)
  (H2 : attended_games = 14)
  (H3 : missed_games = total_games - attended_games) :
  missed_games = 25 :=
by
  sorry

end benny_missed_games_l141_141342


namespace bus_driver_total_compensation_l141_141322

-- Definitions of conditions
def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def overtime_rate : ℝ := regular_rate * 1.75
def total_hours : ℝ := 65
def total_compensation : ℝ := (regular_rate * regular_hours) + (overtime_rate * (total_hours - regular_hours))

-- Theorem stating the total compensation
theorem bus_driver_total_compensation : total_compensation = 1340 :=
by
  sorry

end bus_driver_total_compensation_l141_141322


namespace area_of_triangle_AMN_is_correct_l141_141895

noncomputable def area_triangle_AMN : ℝ :=
  let A := (120 + 56 * Real.sqrt 3) / 3
  let M := (12 + 20 * Real.sqrt 3) / 3
  let N := 4 * Real.sqrt 3 + 20
  (A * N) / 2

theorem area_of_triangle_AMN_is_correct :
  area_triangle_AMN = (224 * Real.sqrt 3 + 240) / 3 := sorry

end area_of_triangle_AMN_is_correct_l141_141895


namespace bed_width_is_4_feet_l141_141850

def total_bags : ℕ := 16
def soil_per_bag : ℕ := 4
def bed_length : ℝ := 8
def bed_height : ℝ := 1
def num_beds : ℕ := 2

theorem bed_width_is_4_feet :
  (total_bags * soil_per_bag / num_beds) = (bed_length * 4 * bed_height) :=
by
  sorry

end bed_width_is_4_feet_l141_141850


namespace min_num_cuboids_l141_141584

/-
Definitions based on the conditions:
- Dimensions of the cuboid are given as 3 cm, 4 cm, and 5 cm.
- We need to find the Least Common Multiple (LCM) of these dimensions.
- Calculate the volume of the smallest cube.
- Calculate the volume of the given cuboid.
- Find the number of such cuboids needed to form the cube.
-/
def cuboid_length : ℤ := 3
def cuboid_width : ℤ := 4
def cuboid_height : ℤ := 5

noncomputable def lcm_3_4_5 : ℤ := Int.lcm (Int.lcm cuboid_length cuboid_width) cuboid_height

noncomputable def cube_side_length : ℤ := lcm_3_4_5
noncomputable def cube_volume : ℤ := cube_side_length * cube_side_length * cube_side_length
noncomputable def cuboid_volume : ℤ := cuboid_length * cuboid_width * cuboid_height

noncomputable def num_cuboids : ℤ := cube_volume / cuboid_volume

theorem min_num_cuboids :
  num_cuboids = 3600 := by
  sorry

end min_num_cuboids_l141_141584


namespace wall_volume_is_128512_l141_141807

noncomputable def wall_volume (width : ℝ) (height : ℝ) (length : ℝ) : ℝ :=
  width * height * length

theorem wall_volume_is_128512 : 
  ∀ (w : ℝ) (h : ℝ) (l : ℝ), 
  h = 6 * w ∧ l = 7 * h ∧ w = 8 → 
  wall_volume w h l = 128512 := 
by
  sorry

end wall_volume_is_128512_l141_141807


namespace linda_original_amount_l141_141550

-- Define the original amount of money Lucy and Linda have
variables (L : ℕ) (lucy_initial : ℕ := 20)

-- Condition: If Lucy gives Linda $5, they have the same amount of money.
def condition := (lucy_initial - 5) = (L + 5)

-- Theorem: The original amount of money that Linda had
theorem linda_original_amount (h : condition L) : L = 10 := 
sorry

end linda_original_amount_l141_141550


namespace maximal_possible_degree_difference_l141_141951

theorem maximal_possible_degree_difference (n_vertices : ℕ) (n_edges : ℕ) (disjoint_edge_pairs : ℕ) 
    (h1 : n_vertices = 30) (h2 : n_edges = 105) (h3 : disjoint_edge_pairs = 4822) : 
    ∃ (max_diff : ℕ), max_diff = 22 :=
by
  sorry

end maximal_possible_degree_difference_l141_141951


namespace range_of_x_l141_141532

theorem range_of_x (x : ℝ) : (x^2 - 9*x + 14 < 0) ∧ (2*x + 3 > 0) ↔ (2 < x) ∧ (x < 7) := 
by 
  sorry

end range_of_x_l141_141532


namespace bowling_ball_weight_l141_141157

variable {b c : ℝ}

theorem bowling_ball_weight :
  (10 * b = 4 * c) ∧ (3 * c = 108) → b = 14.4 :=
by
  sorry

end bowling_ball_weight_l141_141157


namespace max_value_l141_141042

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 18 * x - 5

theorem max_value : ∃ x : ℝ, quadratic_function x = 22 :=
sorry

end max_value_l141_141042


namespace rectangular_prism_volume_l141_141684

theorem rectangular_prism_volume
  (l w h : ℝ)
  (h1 : l * w = 15)
  (h2 : w * h = 10)
  (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end rectangular_prism_volume_l141_141684


namespace expected_value_winnings_l141_141502

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def winnings_heads : ℚ := 4
def loss_tails : ℚ := -3

theorem expected_value_winnings : 
  (probability_heads * winnings_heads + probability_tails * loss_tails) = -1 / 5 := 
by
  -- calculation steps and proof would go here
  sorry

end expected_value_winnings_l141_141502


namespace factor_expression_l141_141714

theorem factor_expression (x : ℝ) : (45 * x^3 - 135 * x^7) = 45 * x^3 * (1 - 3 * x^4) :=
by
  sorry

end factor_expression_l141_141714


namespace katya_solves_enough_l141_141262

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l141_141262


namespace max_abs_x_minus_2y_plus_1_l141_141533

theorem max_abs_x_minus_2y_plus_1 (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) :
  |x - 2 * y + 1| ≤ 5 :=
sorry

end max_abs_x_minus_2y_plus_1_l141_141533


namespace jezebel_total_cost_l141_141244

theorem jezebel_total_cost :
  let red_rose_count := 2 * 12,
      sunflower_count := 3,
      red_rose_cost := 1.50,
      sunflower_cost := 3,
      total_cost := (red_rose_count * red_rose_cost) + (sunflower_count * sunflower_cost)
  in
  total_cost = 45 := 
by
  sorry

end jezebel_total_cost_l141_141244


namespace combined_cost_price_l141_141462

def cost_price_A : ℕ := (120 + 60) / 2
def cost_price_B : ℕ := (200 + 100) / 2
def cost_price_C : ℕ := (300 + 180) / 2

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C

theorem combined_cost_price :
  total_cost_price = 480 := by
  sorry

end combined_cost_price_l141_141462


namespace xy_solution_l141_141037

theorem xy_solution (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 72) : x * y = -8 := by
  sorry

end xy_solution_l141_141037


namespace solve_fraction_inequality_l141_141155

theorem solve_fraction_inequality :
  { x : ℝ | x / (x + 5) ≥ 0 } = { x : ℝ | x < -5 } ∪ { x : ℝ | x ≥ 0 } := by
  sorry

end solve_fraction_inequality_l141_141155


namespace fraction_simplification_l141_141047

theorem fraction_simplification (a b : ℝ) : 9 * b / (6 * a + 3) = 3 * b / (2 * a + 1) :=
by sorry

end fraction_simplification_l141_141047


namespace find_b_num_days_worked_l141_141666

noncomputable def a_num_days_worked := 6
noncomputable def b_num_days_worked := 9  -- This is what we want to verify
noncomputable def c_num_days_worked := 4

noncomputable def c_daily_wage := 105
noncomputable def wage_ratio_a := 3
noncomputable def wage_ratio_b := 4
noncomputable def wage_ratio_c := 5

-- Helper to find daily wages for a and b given the ratio and c's wage
noncomputable def x := c_daily_wage / wage_ratio_c
noncomputable def a_daily_wage := wage_ratio_a * x
noncomputable def b_daily_wage := wage_ratio_b * x

-- Calculate total earnings
noncomputable def a_total_earning := a_num_days_worked * a_daily_wage
noncomputable def c_total_earning := c_num_days_worked * c_daily_wage
noncomputable def total_earning := 1554
noncomputable def b_total_earning := b_num_days_worked * b_daily_wage

theorem find_b_num_days_worked : total_earning = a_total_earning + b_total_earning + c_total_earning → b_num_days_worked = 9 := by
  sorry

end find_b_num_days_worked_l141_141666


namespace fixed_point_of_line_l141_141221

theorem fixed_point_of_line (a : ℝ) (x y : ℝ)
  (h : ∀ a : ℝ, a * x + y + 1 = 0) :
  x = 0 ∧ y = -1 := 
by
  sorry

end fixed_point_of_line_l141_141221


namespace distance_relation_possible_l141_141556

-- Define a structure representing points in 2D space
structure Point where
  x : ℤ
  y : ℤ

-- Define the artificial geometry distance function (Euclidean distance)
def varrho (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

-- Define the non-collinearity condition for points A, B, and C
def non_collinear (A B C : Point) : Prop :=
  ¬(A.x = B.x ∧ B.x = C.x) ∧ ¬(A.y = B.y ∧ B.y = C.y)

theorem distance_relation_possible :
  ∃ (A B C : Point), non_collinear A B C ∧ varrho A C ^ 2 + varrho B C ^ 2 = varrho A B ^ 2 :=
by
  sorry

end distance_relation_possible_l141_141556


namespace inequality_solution_set_l141_141600

theorem inequality_solution_set :
  {x : ℝ | (x / (x ^ 2 - 8 * x + 15) ≥ 2) ∧ (x ^ 2 - 8 * x + 15 ≠ 0)} =
  {x : ℝ | (5 / 2 ≤ x ∧ x < 3) ∨ (5 < x ∧ x ≤ 6)} :=
by
  -- The proof is omitted
  sorry

end inequality_solution_set_l141_141600


namespace jane_last_day_vases_l141_141761

theorem jane_last_day_vases (vases_per_day : ℕ) (total_vases : ℕ) (days : ℕ) (day_arrange_total: days = 17) (vases_per_day_is_25 : vases_per_day = 25) (total_vases_is_378 : total_vases = 378) :
  (vases_per_day * (days - 1) >= total_vases) → (total_vases - vases_per_day * (days - 1)) = 0 :=
by
  intros h
  -- adding this line below to match condition ": (total_vases - vases_per_day * (days - 1)) = 0"
  sorry

end jane_last_day_vases_l141_141761


namespace max_mn_square_proof_l141_141081

noncomputable def max_mn_square (m n : ℕ) : ℕ :=
m^2 + n^2

theorem max_mn_square_proof (m n : ℕ) (h1 : 1 ≤ m ∧ m ≤ 2005) (h2 : 1 ≤ n ∧ n ≤ 2005) (h3 : (n^2 + 2 * m * n - 2 * m^2)^2 = 1) : 
max_mn_square m n ≤ 702036 :=
sorry

end max_mn_square_proof_l141_141081


namespace staircase_steps_eq_twelve_l141_141580

theorem staircase_steps_eq_twelve (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → (n = 12) :=
by
  intro h
  sorry

end staircase_steps_eq_twelve_l141_141580


namespace blue_eyed_blonds_greater_than_population_proportion_l141_141403

variables {G_B Γ B N : ℝ}

theorem blue_eyed_blonds_greater_than_population_proportion (h : G_B / Γ > B / N) : G_B / B > Γ / N :=
sorry

end blue_eyed_blonds_greater_than_population_proportion_l141_141403


namespace lcm_12_18_l141_141978

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l141_141978


namespace joe_total_paint_used_l141_141668

-- Conditions
def initial_paint : ℕ := 360
def paint_first_week : ℕ := initial_paint * 1 / 4
def remaining_paint_after_first_week : ℕ := initial_paint - paint_first_week
def paint_second_week : ℕ := remaining_paint_after_first_week * 1 / 6

-- Theorem statement
theorem joe_total_paint_used : paint_first_week + paint_second_week = 135 := by
  sorry

end joe_total_paint_used_l141_141668


namespace slope_of_intersection_line_is_one_l141_141817

open Real

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 2 * y + 4 = 0

-- The statement to prove that the slope of the line through the intersection points is 1
theorem slope_of_intersection_line_is_one :
  ∃ m : ℝ, (∀ x y : ℝ, circle1 x y → circle2 x y → (y = m * x + b)) ∧ m = 1 :=
by
  sorry

end slope_of_intersection_line_is_one_l141_141817


namespace parallelogram_area_ratio_l141_141491

theorem parallelogram_area_ratio (
  AB CD BC AD AP CQ BP DQ: ℝ)
  (h1 : AB = 13)
  (h2 : CD = 13)
  (h3 : BC = 15)
  (h4 : AD = 15)
  (h5 : AP = 10 / 3)
  (h6 : CQ = 10 / 3)
  (h7 : BP = 29 / 3)
  (h8 : DQ = 29 / 3)
  : ((area_APDQ / area_BPCQ) = 19) :=
sorry

end parallelogram_area_ratio_l141_141491


namespace boat_distance_against_stream_l141_141125

/-- 
  Given:
  1. The boat goes 13 km along the stream in one hour.
  2. The speed of the boat in still water is 11 km/hr.

  Prove:
  The distance the boat goes against the stream in one hour is 9 km.
-/
theorem boat_distance_against_stream (v_s : ℝ) (distance_along_stream time : ℝ) (v_still : ℝ) :
  distance_along_stream = 13 ∧ time = 1 ∧ v_still = 11 ∧ (v_still + v_s) = 13 → 
  (v_still - v_s) * time = 9 := by
  sorry

end boat_distance_against_stream_l141_141125


namespace problem1_problem2_l141_141991

variables {a b : ℝ}

-- Given conditions
def condition1 : a + b = 2 := sorry
def condition2 : a * b = -1 := sorry

-- Proof for a^2 + b^2 = 6
theorem problem1 (h1 : a + b = 2) (h2 : a * b = -1) : a^2 + b^2 = 6 :=
by sorry

-- Proof for (a - b)^2 = 8
theorem problem2 (h1 : a + b = 2) (h2 : a * b = -1) : (a - b)^2 = 8 :=
by sorry

end problem1_problem2_l141_141991


namespace alyssa_final_money_l141_141693

-- Definitions based on conditions
def weekly_allowance : Int := 8
def spent_on_movies : Int := weekly_allowance / 2
def earnings_from_washing_car : Int := 8

-- The statement to prove
def final_amount : Int := (weekly_allowance - spent_on_movies) + earnings_from_washing_car

-- The theorem expressing the problem
theorem alyssa_final_money : final_amount = 12 := by
  sorry

end alyssa_final_money_l141_141693


namespace car_and_truck_arrival_time_simultaneous_l141_141670

theorem car_and_truck_arrival_time_simultaneous {t_car t_truck : ℕ} 
    (h1 : t_car = 8 * 60 + 16) -- Car leaves at 08:16
    (h2 : t_truck = 9 * 60) -- Truck leaves at 09:00
    (h3 : t_car_arrive = 10 * 60 + 56) -- Car arrives at 10:56
    (h4 : t_truck_arrive = 12 * 60 + 20) -- Truck arrives at 12:20
    (h5 : t_truck_exit = t_car_exit + 2) -- Truck leaves tunnel 2 minutes after car
    : (t_car_exit + t_car_tunnel_time = 10 * 60) ∧ (t_truck_exit + t_truck_tunnel_time = 10 * 60) :=
  sorry

end car_and_truck_arrival_time_simultaneous_l141_141670


namespace sum_of_arithmetic_series_l141_141107

theorem sum_of_arithmetic_series (a1 an : ℕ) (d n : ℕ) (s : ℕ) :
  a1 = 2 ∧ an = 100 ∧ d = 2 ∧ n = (an - a1) / d + 1 ∧ s = n * (a1 + an) / 2 → s = 2550 :=
by
  sorry

end sum_of_arithmetic_series_l141_141107


namespace factorization_of_expression_l141_141083

theorem factorization_of_expression (x y : ℝ) : x^2 - x * y = x * (x - y) := 
by
  sorry

end factorization_of_expression_l141_141083


namespace smallest_positive_multiple_l141_141937

/-- Prove that the smallest positive multiple of 15 that is 7 more than a multiple of 65 is 255. -/
theorem smallest_positive_multiple : 
  ∃ n : ℕ, n > 0 ∧ n % 15 = 0 ∧ n % 65 = 7 ∧ n = 255 :=
sorry

end smallest_positive_multiple_l141_141937


namespace number_divisors_l141_141449

theorem number_divisors (p : ℕ) (h : p = 2^56 - 1) : ∃ x y : ℕ, 95 ≤ x ∧ x ≤ 105 ∧ 95 ≤ y ∧ y ≤ 105 ∧ p % x = 0 ∧ p % y = 0 ∧ x = 101 ∧ y = 127 :=
by {
  sorry
}

end number_divisors_l141_141449


namespace chef_earns_2_60_less_l141_141074

/--
At Joe's Steakhouse, the hourly wage for a chef is 20% greater than that of a dishwasher,
and the hourly wage of a dishwasher is half as much as the hourly wage of a manager.
If a manager's wage is $6.50 per hour, prove that a chef earns $2.60 less per hour than a manager.
-/
theorem chef_earns_2_60_less {w_manager w_dishwasher w_chef : ℝ} 
  (h1 : w_dishwasher = w_manager / 2)
  (h2 : w_chef = w_dishwasher * 1.20)
  (h3 : w_manager = 6.50) :
  w_manager - w_chef = 2.60 :=
by
  sorry

end chef_earns_2_60_less_l141_141074


namespace kamal_twice_age_in_future_l141_141010

theorem kamal_twice_age_in_future :
  ∃ x : ℕ, (K = 40) ∧ (K - 8 = 4 * (S - 8)) ∧ (K + x = 2 * (S + x)) :=
by {
  sorry 
}

end kamal_twice_age_in_future_l141_141010


namespace pos_int_fraction_iff_l141_141739

theorem pos_int_fraction_iff (p : ℕ) (hp : p > 0) : (∃ k : ℕ, 4 * p + 11 = k * (2 * p - 7)) ↔ (p = 4 ∨ p = 5) := 
sorry

end pos_int_fraction_iff_l141_141739


namespace average_tickets_per_day_l141_141194

def total_revenue : ℕ := 960
def price_per_ticket : ℕ := 4
def number_of_days : ℕ := 3

theorem average_tickets_per_day :
  (total_revenue / price_per_ticket) / number_of_days = 80 := 
sorry

end average_tickets_per_day_l141_141194


namespace tan_15_degrees_theta_range_valid_max_f_value_l141_141128

-- Define the dot product condition
def dot_product_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  AB * BC * (Real.cos θ) = 6

-- Define the sine inequality condition
def sine_inequality_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  6 * (2 - Real.sqrt 3) ≤ AB * BC * (Real.sin θ) ∧ AB * BC * (Real.sin θ) ≤ 6 * Real.sqrt 3

-- Define the maximum value function
noncomputable def f (θ : ℝ) : ℝ :=
  (1 - Real.sqrt 2 * Real.cos (2 * θ - Real.pi / 4)) / (Real.sin θ)

-- Proof that tan 15 degrees is equal to 2 - sqrt(3)
theorem tan_15_degrees : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 := 
  by sorry

-- Proof for the range of θ
theorem theta_range_valid (AB BC : ℝ) (θ : ℝ) 
  (h1 : dot_product_condition AB BC θ)
  (h2 : sine_inequality_condition AB BC θ) : 
  (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3) := 
  by sorry

-- Proof for the maximum value of the function
theorem max_f_value (θ : ℝ) 
  (h : (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3)) : 
  f θ ≤ Real.sqrt 3 - 1 := 
  by sorry

end tan_15_degrees_theta_range_valid_max_f_value_l141_141128


namespace sample_size_proof_l141_141932

-- Conditions
def investigate_height_of_students := "To investigate the height of junior high school students in Rui State City in early 2016, 200 students were sampled for the survey."

-- Definition of sample size based on the condition
def sample_size_condition (students_sampled : ℕ) : ℕ := students_sampled

-- Prove the sample size is 200 given the conditions
theorem sample_size_proof : sample_size_condition 200 = 200 := 
by
  sorry

end sample_size_proof_l141_141932


namespace divisor_of_136_l141_141432

theorem divisor_of_136 (d : ℕ) (h : 136 = 9 * d + 1) : d = 15 := 
by {
  -- Since the solution steps are skipped, we use sorry to indicate a placeholder.
  sorry
}

end divisor_of_136_l141_141432


namespace earnings_difference_l141_141054

noncomputable def investment_ratio_a : ℕ := 3
noncomputable def investment_ratio_b : ℕ := 4
noncomputable def investment_ratio_c : ℕ := 5

noncomputable def return_ratio_a : ℕ := 6
noncomputable def return_ratio_b : ℕ := 5
noncomputable def return_ratio_c : ℕ := 4

noncomputable def total_earnings : ℕ := 2900

noncomputable def earnings_a (x y : ℕ) : ℚ := (investment_ratio_a * return_ratio_a * x * y) / 100
noncomputable def earnings_b (x y : ℕ) : ℚ := (investment_ratio_b * return_ratio_b * x * y) / 100

theorem earnings_difference (x y : ℕ) (h : (investment_ratio_a * return_ratio_a * x * y + investment_ratio_b * return_ratio_b * x * y + investment_ratio_c * return_ratio_c * x * y) / 100 = total_earnings) :
  earnings_b x y - earnings_a x y = 100 := by
  sorry

end earnings_difference_l141_141054


namespace shaded_area_percentage_l141_141470

theorem shaded_area_percentage (n_shaded : ℕ) (n_total : ℕ) (hn_shaded : n_shaded = 21) (hn_total : n_total = 36) :
  ((n_shaded : ℚ) / (n_total : ℚ)) * 100 = 58.33 :=
by
  sorry

end shaded_area_percentage_l141_141470


namespace find_m_l141_141179

theorem find_m (m : ℕ) : 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ↔ m = 14 := by
  sorry

end find_m_l141_141179


namespace multiply_same_exponents_l141_141075

theorem multiply_same_exponents (x : ℝ) : (x^3) * (x^3) = x^6 :=
by sorry

end multiply_same_exponents_l141_141075


namespace num_men_employed_l141_141483

noncomputable def original_number_of_men (M : ℕ) : Prop :=
  let total_work_original := M * 5
  let total_work_actual := (M - 8) * 15
  total_work_original = total_work_actual

theorem num_men_employed (M : ℕ) (h : original_number_of_men M) : M = 12 :=
by sorry

end num_men_employed_l141_141483


namespace product_of_x1_to_x13_is_zero_l141_141286

theorem product_of_x1_to_x13_is_zero
  (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ)
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 :=
sorry

end product_of_x1_to_x13_is_zero_l141_141286


namespace six_people_mutual_known_l141_141476

/-!
Proof Problem:
Given:
1. There are 512 people at the meeting.
2. Under every six people, there is always at least two who know each other.
Prove:
There must be six people at this gathering who all mutually know each other.
-/

theorem six_people_mutual_known (n : ℕ) (h : n = 512)
  (H : ∀ s : Finset (Fin n), s.card = 6 → ∃ x y : Fin n, x ≠ y ∧ x ∈ s ∧ y ∈ s ∧ x.adj y) :
  ∃ t : Finset (Fin n), t.card = 6 ∧ ∀ x y : Fin n, x ∈ t ∧ y ∈ t → x.adj y := by
  sorry

end six_people_mutual_known_l141_141476


namespace number_of_shelves_l141_141930

theorem number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h_total_books : total_books = 14240) (h_books_per_shelf : books_per_shelf = 8) : total_books / books_per_shelf = 1780 :=
by 
  -- Proof goes here.
  sorry

end number_of_shelves_l141_141930


namespace cos_alpha_value_l141_141723

open Real

theorem cos_alpha_value (α : ℝ) (h1 : sin (α + π / 3) = 3 / 5) (h2 : π / 6 < α ∧ α < 5 * π / 6) :
  cos α = (3 * sqrt 3 - 4) / 10 :=
by
  sorry

end cos_alpha_value_l141_141723


namespace polynomial_solution_l141_141565

noncomputable def roots (a b c : ℤ) : Set ℝ :=
  { x : ℝ | a * x ^ 2 + b * x + c = 0 }

theorem polynomial_solution :
  let x1 := (1 + Real.sqrt 13) / 2
  let x2 := (1 - Real.sqrt 13) / 2
  x1 ∈ roots 1 (-1) (-3) → x2 ∈ roots 1 (-1) (-3) →
  ((x1^5 - 20) * (3*x2^4 - 2*x2 - 35) = -1063) :=
by
  sorry

end polynomial_solution_l141_141565


namespace find_fourth_number_l141_141948

theorem find_fourth_number : 
  ∃ (x : ℝ), (217 + 2.017 + 0.217 + x = 221.2357) ∧ (x = 2.0017) :=
by
  sorry

end find_fourth_number_l141_141948


namespace binkie_gemstones_l141_141707

variables (F B S : ℕ)

theorem binkie_gemstones :
  (B = 4 * F) →
  (S = (1 / 2 : ℝ) * F - 2) →
  (S = 1) →
  B = 24 :=
by
  sorry

end binkie_gemstones_l141_141707


namespace arithmetic_mean_of_fractions_l141_141645

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l141_141645


namespace similarity_ratio_of_polygons_l141_141933

theorem similarity_ratio_of_polygons (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : a / (b : ℚ) = 3 / 5 :=
by 
  sorry

end similarity_ratio_of_polygons_l141_141933


namespace total_weight_correct_l141_141965

-- Define the constant variables as per the conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def fill_percentage : ℝ := 0.7
def jug1_density : ℝ := 4
def jug2_density : ℝ := 5

-- Define the volumes of sand in each jug
def jug1_sand_volume : ℝ := fill_percentage * jug1_capacity
def jug2_sand_volume : ℝ := fill_percentage * jug2_capacity

-- Define the weights of sand in each jug
def jug1_weight : ℝ := jug1_sand_volume * jug1_density
def jug2_weight : ℝ := jug2_sand_volume * jug2_density

-- State the theorem that combines the weights
theorem total_weight_correct : jug1_weight + jug2_weight = 16.1 := sorry

end total_weight_correct_l141_141965


namespace problem_statement_l141_141220

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem problem_statement : f (g 5) - g (f 5) = 63 :=
by
  sorry

end problem_statement_l141_141220


namespace sum_of_roots_l141_141703
-- Import Mathlib to cover all necessary functionality.

-- Define the function representing the given equation.
def equation (x : ℝ) : ℝ :=
  (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- State the theorem to be proved.
theorem sum_of_roots : (6 : ℝ) + (-4 / 3) = 14 / 3 :=
by
  sorry

end sum_of_roots_l141_141703


namespace tangent_line_through_P_is_correct_l141_141067

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the tangent line equation to prove
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 10 = 0

-- Problem statement in Lean 4
theorem tangent_line_through_P_is_correct :
  C P.1 P.2 → tangent_line P.1 P.2 :=
by
  intros hC
  sorry

end tangent_line_through_P_is_correct_l141_141067


namespace binomial_expectation_variance_l141_141725

open ProbabilityTheory

-- Conditions
def binomial_pmf (n : ℕ) (p : ℝ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.ofMeasure (binomial n p)

-- Binomial random variable X
def X : ProbabilityMassFunction ℕ := binomial_pmf 10 0.6

-- Statement
theorem binomial_expectation_variance :
  let E_X := X.μf.sum (fun k => k * X p k) in -- Expectation
  let D_X := X.μf.sum (fun k => (k - E_X)^2 * X p k) in -- Variance
  E_X = 6 ∧ D_X = 2.4 :=
sorry

end binomial_expectation_variance_l141_141725


namespace monitor_width_l141_141427

theorem monitor_width (d w h : ℝ) (h_ratio : w / h = 16 / 9) (h_diag : d = 24) :
  w = 384 / Real.sqrt 337 :=
by
  sorry

end monitor_width_l141_141427


namespace root_of_quadratic_l141_141104

theorem root_of_quadratic (m : ℝ) (h : 3*1^2 - 1 + m = 0) : m = -2 :=
by {
  sorry
}

end root_of_quadratic_l141_141104


namespace negation_of_prop_l141_141448

-- Define the original proposition
def prop (x : ℝ) : Prop := x^2 - x + 2 ≥ 0

-- State the negation of the original proposition
theorem negation_of_prop : (¬ ∀ x : ℝ, prop x) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := 
by
  sorry

end negation_of_prop_l141_141448


namespace wasting_water_notation_l141_141744

theorem wasting_water_notation (saving_wasting : ℕ → ℤ)
  (h_pos : saving_wasting 30 = 30) :
  saving_wasting 10 = -10 :=
by
  sorry

end wasting_water_notation_l141_141744


namespace solution_set_of_inequality_system_l141_141454

theorem solution_set_of_inequality_system (x : ℝ) : (x + 1 > 0) ∧ (-2 * x ≤ 6) ↔ (x > -1) := 
by 
  sorry

end solution_set_of_inequality_system_l141_141454


namespace certain_number_minus_15_l141_141310

theorem certain_number_minus_15 (n : ℕ) (h : n / 10 = 6) : n - 15 = 45 :=
sorry

end certain_number_minus_15_l141_141310


namespace equal_playtime_l141_141301

theorem equal_playtime (children : ℕ) (total_minutes : ℕ) (simultaneous_players : ℕ) (equal_playtime_per_child : ℕ)
  (h1 : children = 12) (h2 : total_minutes = 120) (h3 : simultaneous_players = 2) (h4 : equal_playtime_per_child = (simultaneous_players * total_minutes) / children) :
  equal_playtime_per_child = 20 := 
by sorry

end equal_playtime_l141_141301


namespace find_cost_price_l141_141828

theorem find_cost_price (C : ℝ) (h1 : 1.12 * C + 18 = 1.18 * C) : C = 300 :=
by
  sorry

end find_cost_price_l141_141828


namespace sufficient_condition_p_or_q_false_p_and_q_false_l141_141872

variables (p q : Prop)

theorem sufficient_condition_p_or_q_false_p_and_q_false :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ¬ ( (¬ (p ∧ q)) → ¬ (p ∨ q)) :=
by 
  -- Proof: If ¬ (p ∨ q), then (p ∨ q) is false, which means (p ∧ q) must also be false.
  -- The other direction would mean if at least one of p or q is false, then (p ∨ q) is false,
  -- which is not necessarily true. Therefore, it's not a necessary condition.
  sorry

end sufficient_condition_p_or_q_false_p_and_q_false_l141_141872


namespace direct_proportion_k_l141_141889

theorem direct_proportion_k (k x : ℝ) : ((k-1) * x + k^2 - 1 = 0) ∧ (k ≠ 1) ↔ k = -1 := 
sorry

end direct_proportion_k_l141_141889


namespace students_left_is_31_l141_141020

-- Define the conditions based on the problem statement
def total_students : ℕ := 124
def checked_out_early : ℕ := 93

-- Define the theorem that states the problem we want to prove
theorem students_left_is_31 :
  total_students - checked_out_early = 31 :=
by
  -- Proof would go here
  sorry

end students_left_is_31_l141_141020


namespace percentage_calculation_l141_141741

theorem percentage_calculation
  (x : ℝ)
  (hx : x = 16)
  (h : 0.15 * 40 - (P * x) = 2) :
  P = 0.25 := by
  sorry

end percentage_calculation_l141_141741


namespace actual_distance_traveled_l141_141743

theorem actual_distance_traveled 
  (D : ℝ) (t : ℝ)
  (h1 : 8 * t = D)
  (h2 : 12 * t = D + 20) : 
  D = 40 :=
by
  sorry

end actual_distance_traveled_l141_141743


namespace find_f_2022_l141_141805

-- Define a function f that satisfies the given conditions
def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f(a) + 2 * f(b)) / 3
axiom f_1 : f 1 = 1
axiom f_4 : f 4 = 7

-- The main theorem to prove
theorem find_f_2022 : f 2022 = 4043 :=
by
  sorry

end find_f_2022_l141_141805


namespace factorial_division_l141_141518

theorem factorial_division (N : Nat) (h : N ≥ 2) : 
  (Nat.factorial (2 * N)) / ((Nat.factorial (N + 2)) * (Nat.factorial (N - 2))) = 
  (List.prod (List.range' (N + 3) (2 * N - (N + 2) + 1))) / (Nat.factorial (N - 1)) :=
sorry

end factorial_division_l141_141518


namespace advertisements_shown_l141_141523

theorem advertisements_shown (advertisement_duration : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) :
  advertisement_duration = 3 →
  cost_per_minute = 4000 →
  total_cost = 60000 →
  total_cost / (advertisement_duration * cost_per_minute) = 5 :=
by
  sorry

end advertisements_shown_l141_141523


namespace cyclists_meeting_l141_141720

-- Define the velocities of the cyclists and the time variable
variables (v₁ v₂ t : ℝ)

-- Define the conditions for the problem
def condition1 : Prop := v₁ * t = v₂ * (2/3)
def condition2 : Prop := v₂ * t = v₁ * 1.5

-- Define the main theorem to be proven
theorem cyclists_meeting (h1 : condition1 v₁ v₂ t) (h2 : condition2 v₁ v₂ t) :
  t = 1 ∧ (v₁ / v₂ = 3 / 2) :=
by sorry

end cyclists_meeting_l141_141720


namespace side_error_percentage_l141_141506

theorem side_error_percentage (S S' : ℝ) (h1: S' = S * Real.sqrt 1.0609) : 
  (S' / S - 1) * 100 = 3 :=
by
  sorry

end side_error_percentage_l141_141506


namespace range_of_xy_l141_141996

theorem range_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y + x * y = 30) :
  12 < x * y ∧ x * y < 870 :=
by sorry

end range_of_xy_l141_141996


namespace average_weight_increase_l141_141281

theorem average_weight_increase
  (A : ℝ) -- Average weight of the two persons
  (w1 : ℝ) (h1 : w1 = 65) -- One person's weight is 65 kg 
  (w2 : ℝ) (h2 : w2 = 74) -- The new person's weight is 74 kg
  :
  ((A * 2 - w1 + w2) / 2 - A = 4.5) :=
by
  simp [h1, h2]
  sorry

end average_weight_increase_l141_141281


namespace arithmetic_mean_of_fractions_l141_141628

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l141_141628


namespace length_of_intersection_segment_l141_141759

-- Define the polar coordinates conditions
def curve_1 (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def curve_2 (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

-- Convert polar equations to Cartesian coordinates
def curve_1_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4 * y
def curve_2_cartesian (x y : ℝ) : Prop := x = 1

-- Define the intersection points and the segment length function
def segment_length (y1 y2 : ℝ) : ℝ := abs (y1 - y2)

-- The statement to prove
theorem length_of_intersection_segment :
  (curve_1_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_1_cartesian 1 (2 - Real.sqrt 3)) →
  (curve_2_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_2_cartesian 1 (2 - Real.sqrt 3)) →
  segment_length (2 + Real.sqrt 3) (2 - Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end length_of_intersection_segment_l141_141759


namespace simplify_expression_l141_141581

variable {R : Type} [AddCommGroup R] [Module ℤ R]

theorem simplify_expression (a b : R) :
  (25 • a + 70 • b) + (15 • a + 34 • b) - (12 • a + 55 • b) = 28 • a + 49 • b :=
by sorry

end simplify_expression_l141_141581


namespace sum_of_8x8_array_l141_141897

theorem sum_of_8x8_array : 
  ∀ (n : ℕ), (n + 1) * (n + 1) = 64 ∧ 
  (16 * (n / 4) + (15 * 16) / 2 = 560) → 
  ∑ i in Finset.range 64, i = 1984 :=
by
  intros n h
  sorry

end sum_of_8x8_array_l141_141897


namespace arithmetic_mean_of_fractions_l141_141624

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l141_141624


namespace solve_equation_l141_141917

-- Define the equation to be proven
def equation (x : ℚ) : Prop :=
  (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the theorem
theorem solve_equation : equation (-2 / 11) :=
by
  -- Introduce the equation and the solution to be proven
  unfold equation

  -- Simplify the equation to verify the solution
  sorry


end solve_equation_l141_141917


namespace player_B_wins_in_least_steps_l141_141607

noncomputable def least_steps_to_win (n : ℕ) : ℕ :=
  n

theorem player_B_wins_in_least_steps (n : ℕ) (h_n : n > 0) :
  ∃ k, k = least_steps_to_win n ∧ k = n := by
  sorry

end player_B_wins_in_least_steps_l141_141607


namespace max_value_quadratic_expression_l141_141040

theorem max_value_quadratic_expression : ∃ x : ℝ, -3 * x^2 + 18 * x - 5 ≤ 22 ∧ ∀ y : ℝ, -3 * y^2 + 18 * y - 5 ≤ 22 := 
by 
  sorry

end max_value_quadratic_expression_l141_141040


namespace convert_to_base_8_l141_141078

theorem convert_to_base_8 (n : ℕ) (hn : n = 3050) : 
  ∃ d1 d2 d3 d4 : ℕ, d1 = 5 ∧ d2 = 7 ∧ d3 = 5 ∧ d4 = 2 ∧ n = d1 * 8^3 + d2 * 8^2 + d3 * 8^1 + d4 * 8^0 :=
by 
  use 5, 7, 5, 2
  sorry

end convert_to_base_8_l141_141078


namespace union_of_sets_l141_141880

open Set

noncomputable def A (a : ℝ) : Set ℝ := {1, 2^a}
noncomputable def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) (h₁ : A a ∩ B a b = {1 / 2}) :
  A a ∪ B a b = {-1, 1 / 2, 1} :=
by
  sorry

end union_of_sets_l141_141880


namespace bank_exceeds_50_dollars_l141_141903

theorem bank_exceeds_50_dollars (a : ℕ := 5) (r : ℕ := 2) :
  ∃ n : ℕ, 5 * (2 ^ n - 1) > 5000 ∧ (n ≡ 9 [MOD 7]) :=
by
  sorry

end bank_exceeds_50_dollars_l141_141903


namespace smith_family_seating_problem_l141_141587

theorem smith_family_seating_problem :
  let total_children := 8
  let boys := 4
  let girls := 4
  (total_children.factorial - (boys.factorial * girls.factorial)) = 39744 :=
by
  sorry

end smith_family_seating_problem_l141_141587


namespace floor_div_eq_floor_div_floor_l141_141436

theorem floor_div_eq_floor_div_floor {α : ℝ} {d : ℕ} (h₁ : 0 < α) : 
  (⌊α / d⌋ = ⌊⌊α⌋ / d⌋) := 
sorry

end floor_div_eq_floor_div_floor_l141_141436


namespace problem_statement_l141_141854

variable (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ)

theorem problem_statement
  (h1 : y1 + 4 * y2 + 9 * y3 + 16 * y4 + 25 * y5 + 36 * y6 + 49 * y7 + 64 * y8 = 3)
  (h2 : 4 * y1 + 9 * y2 + 16 * y3 + 25 * y4 + 36 * y5 + 49 * y6 + 64 * y7 + 81 * y8 = 15)
  (h3 : 9 * y1 + 16 * y2 + 25 * y3 + 36 * y4 + 49 * y5 + 64 * y6 + 81 * y7 + 100 * y8 = 140) :
  16 * y1 + 25 * y2 + 36 * y3 + 49 * y4 + 64 * y5 + 81 * y6 + 100 * y7 + 121 * y8 = 472 := by
  sorry

end problem_statement_l141_141854


namespace unique_solution_condition_l141_141920

noncomputable def unique_solution_system (a b c x y z : ℝ) : Prop :=
  (a * x + b * y - b * z = c) ∧ 
  (a * y + b * x - b * z = c) ∧ 
  (a * z + b * y - b * x = c) → 
  (x = y ∧ y = z ∧ x = c / a)

theorem unique_solution_condition (a b c x y z : ℝ) 
  (h1 : a * x + b * y - b * z = c)
  (h2 : a * y + b * x - b * z = c)
  (h3 : a * z + b * y - b * x = c)
  (ha : a ≠ 0)
  (ha_b : a ≠ b)
  (ha_b' : a + b ≠ 0) :
  unique_solution_system a b c x y z :=
by 
  sorry

end unique_solution_condition_l141_141920


namespace selection_methods_count_l141_141023

-- Define the number of female students
def num_female_students : ℕ := 3

-- Define the number of male students
def num_male_students : ℕ := 2

-- Define the total number of different selection methods
def total_selection_methods : ℕ := num_female_students + num_male_students

-- Prove that the total number of different selection methods is 5
theorem selection_methods_count : total_selection_methods = 5 := by
  sorry

end selection_methods_count_l141_141023


namespace range_of_f_l141_141400

noncomputable def f (θ x : ℝ) : ℝ := 2 * sin (x + 2 * θ) * cos x

theorem range_of_f (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) (h : f θ 0 = 2) : 
  ∀ x, 0 ≤ f θ x ∧ f θ x ≤ 2 :=
sorry

end range_of_f_l141_141400


namespace speed_of_A_is_7_l141_141058

theorem speed_of_A_is_7
  (x : ℝ)
  (h1 : ∀ t : ℝ, t = 1)
  (h2 : ∀ y : ℝ, y = 3)
  (h3 : ∀ n : ℕ, n = 10)
  (h4 : x + 3 = 10) :
  x = 7 := by
  sorry

end speed_of_A_is_7_l141_141058


namespace percentage_of_failed_candidates_l141_141945

theorem percentage_of_failed_candidates
(total_candidates : ℕ)
(girls : ℕ)
(passed_boys_percentage : ℝ)
(passed_girls_percentage : ℝ)
(h1 : total_candidates = 2000)
(h2 : girls = 900)
(h3 : passed_boys_percentage = 0.28)
(h4 : passed_girls_percentage = 0.32)
: (total_candidates - (passed_boys_percentage * (total_candidates - girls) + passed_girls_percentage * girls)) / total_candidates * 100 = 70.2 :=
by
  sorry

end percentage_of_failed_candidates_l141_141945


namespace frustum_volume_correct_l141_141503

-- Definitions of pyramids and their properties
structure Pyramid :=
  (base_edge : ℕ)
  (altitude : ℕ)
  (volume : ℚ)

-- Definition of the original pyramid and smaller pyramid
def original_pyramid : Pyramid := {
  base_edge := 20,
  altitude := 10,
  volume := (1 / 3 : ℚ) * (20 ^ 2) * 10
}

def smaller_pyramid : Pyramid := {
  base_edge := 8,
  altitude := 5,
  volume := (1 / 3 : ℚ) * (8 ^ 2) * 5
}

-- Definition and calculation of the volume of the frustum 
def volume_frustum (p1 p2 : Pyramid) : ℚ :=
  p1.volume - p2.volume

-- Main theorem to be proved
theorem frustum_volume_correct :
  volume_frustum original_pyramid smaller_pyramid = 992 := by
  sorry

end frustum_volume_correct_l141_141503


namespace arithmetic_mean_of_fractions_l141_141623

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l141_141623


namespace equal_segments_l141_141771

-- Given a triangle ABC and D as the foot of the bisector from B
variables (A B C D E F : Point) (ABC : Triangle A B C) (Dfoot : BisectorFoot B A C D) 

-- Given that the circumcircles of triangles ABD and BCD intersect sides AB and BC at E and F respectively
variables (circABD : Circumcircle A B D) (circBCD : Circumcircle B C D)
variables (intersectAB : Intersect circABD A B E) (intersectBC : Intersect circBCD B C F)

-- The theorem to prove that AE = CF
theorem equal_segments : AE = CF :=
by
  sorry

end equal_segments_l141_141771


namespace sum_of_digits_n_l141_141830

-- Helper function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem sum_of_digits_n :
  ∃ n, n = Int.gcd (4665 - 1305) (Int.gcd (6905 - 4665) (6905 - 1305)) ∧ sum_of_digits n = 4 :=
by
  have h : 4665 - 1305 = 3360 := by norm_num
  have h1 : 6905 - 4665 = 2240 := by norm_num
  have h2 : 6905 - 1305 = 5600 := by norm_num
  let n := Int.gcd 3360 (Int.gcd 2240 5600)
  use n
  split
  · suffices : n = 1120, from this
    rw [h, h1, h2]
    exact Eq.symm (Int.gcd_gcd_1120 3360 2240 5600)
  · unfold sum_of_digits
    norm_num
    exact rfl

end sum_of_digits_n_l141_141830


namespace chess_tournament_l141_141750

theorem chess_tournament (n : ℕ) (h : (n * (n - 1)) / 2 - ((n - 3) * (n - 4)) / 2 = 130) : n = 19 :=
sorry

end chess_tournament_l141_141750


namespace polygons_after_cuts_l141_141699

theorem polygons_after_cuts (initial_polygons : ℕ) (cuts : ℕ) 
  (initial_vertices : ℕ) (max_vertices_added_per_cut : ℕ) :
  (initial_polygons = 10) →
  (cuts = 51) →
  (initial_vertices = 100) →
  (max_vertices_added_per_cut = 4) →
  ∃ p, (p < 5 ∧ p ≥ 3) :=
by
  intros h_initial_polygons h_cuts h_initial_vertices h_max_vertices_added_per_cut
  -- proof steps would go here
  sorry

end polygons_after_cuts_l141_141699


namespace find_value_of_expression_l141_141214

variables (a b c : ℝ)

theorem find_value_of_expression
  (h1 : a ^ 4 * b ^ 3 * c ^ 5 = 18)
  (h2 : a ^ 3 * b ^ 5 * c ^ 4 = 8) :
  a ^ 5 * b * c ^ 6 = 81 / 2 :=
sorry

end find_value_of_expression_l141_141214


namespace probability_rain_one_day_at_least_l141_141299

open ProbabilityTheory

-- Define the given conditions
variables {P_A P_B P_B_given_A P_B_given_not_A : ℚ}
variables (h1 : P_A = 0.4)
variables (h2 : P_B = 0.3)
variables (h3 : P_B_given_A = 2 * P_B_given_not_A)

-- Define the statement that needs to be proven
theorem probability_rain_one_day_at_least (h1 : P_A = 0.4) (h2 : P_B = 0.3) (h3 : P_B_given_A = 2 * P_B_given_not_A) :
  let P_not_A := 1 - P_A,
      P_B_given_not_A := P_B / (P_A * 2 + (1 - P_A)),
      P_not_B_given_A := 1 - P_B_given_A,
      P_not_B_given_not_A := 1 - P_B_given_not_A,
      P_not_A_and_not_B := P_not_A * P_not_B_given_not_A in
  1 - P_not_A_and_not_B = 37 / 70 := by
  let P_not_A := 1 - 0.4
  let P_B_given_not_A := 0.3 / (0.4 * 2 + (1 - 0.4))
  let P_not_B_given_A := 1 - 2 * P_B_given_not_A
  let P_not_B_given_not_A := 1 - P_B_given_not_A
  let P_not_A_and_not_B := P_not_A * P_not_B_given_not_A
  show 1 - P_not_A_and_not_B = 37 / 70

end probability_rain_one_day_at_least_l141_141299


namespace not_covered_by_homothetic_polygons_l141_141573

structure Polygon :=
  (vertices : Set (ℝ × ℝ))

def homothetic (M : Polygon) (k : ℝ) (O : ℝ × ℝ) : Polygon :=
  {
    vertices := {p | ∃ (q : ℝ × ℝ) (hq : q ∈ M.vertices), p = (O.1 + k * (q.1 - O.1), O.2 + k * (q.2 - O.2))}
  }

theorem not_covered_by_homothetic_polygons (M : Polygon) (k : ℝ) (h : 0 < k ∧ k < 1)
  (O1 O2 : ℝ × ℝ) :
  ¬ (∀ p ∈ M.vertices, p ∈ (homothetic M k O1).vertices ∨ p ∈ (homothetic M k O2).vertices) := by
  sorry

end not_covered_by_homothetic_polygons_l141_141573


namespace M_is_correct_ab_property_l141_141722

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 1|
def M : Set ℝ := {x | f x < 4}

theorem M_is_correct : M = {x | -2 < x ∧ x < 2} :=
sorry

theorem ab_property (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 2 * |a + b| < |4 + a * b| :=
sorry

end M_is_correct_ab_property_l141_141722


namespace arithmetic_mean_of_fractions_l141_141629

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l141_141629


namespace problem1_l141_141672

theorem problem1 : 20 + (-14) - (-18) + 13 = 37 :=
by
  sorry

end problem1_l141_141672


namespace solve_equation_l141_141916

-- Define the equation to be proven
def equation (x : ℚ) : Prop :=
  (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the theorem
theorem solve_equation : equation (-2 / 11) :=
by
  -- Introduce the equation and the solution to be proven
  unfold equation

  -- Simplify the equation to verify the solution
  sorry


end solve_equation_l141_141916


namespace hill_height_l141_141008

theorem hill_height (h : ℝ) (time_up : ℝ := h / 9) (time_down : ℝ := h / 12) (total_time : ℝ := time_up + time_down) (time_cond : total_time = 175) : h = 900 :=
by 
  sorry

end hill_height_l141_141008


namespace find_omega_and_range_l141_141109

noncomputable def f (ω : ℝ) (x : ℝ) := (Real.sin (ω * x))^2 + (Real.sqrt 3) * (Real.sin (ω * x)) * (Real.sin (ω * x + Real.pi / 2))

theorem find_omega_and_range :
  ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x = (Real.sin (2 * ω * x - Real.pi / 6) + 1/2)) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
      f 1 x ∈ Set.Icc ((1 - Real.sqrt 3) / 2) (3 / 2)) :=
by
  sorry

end find_omega_and_range_l141_141109


namespace find_m_l141_141215

noncomputable def polynomial (x : ℝ) (m : ℝ) := 4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1

theorem find_m (m : ℝ) : 
  ∀ x : ℝ, (4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1 = (4 - 2 * m) * x^2 - 4 * x + 6)
  → (4 - 2 * m = 0) → (m = 2) :=
by
  intros x h1 h2
  sorry

end find_m_l141_141215


namespace quadratic_positive_intervals_l141_141122

-- Problem setup
def quadratic (x : ℝ) : ℝ := x^2 - x - 6

-- Define the roots of the quadratic function
def is_root (a b : ℝ) (f : ℝ → ℝ) := f a = 0 ∧ f b = 0

-- Proving the intervals where the quadratic function is greater than 0
theorem quadratic_positive_intervals :
  is_root (-2) 3 quadratic →
  { x : ℝ | quadratic x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 3 } :=
by
  sorry

end quadratic_positive_intervals_l141_141122


namespace number_of_equilateral_triangles_l141_141121

noncomputable def parabola_equilateral_triangles (y x : ℝ) : Prop :=
  y^2 = 4 * x

theorem number_of_equilateral_triangles : ∃ n : ℕ, n = 2 ∧
  ∀ (a b c d e : ℝ), 
    (parabola_equilateral_triangles (a - 1) b) ∧ 
    (parabola_equilateral_triangles (c - 1) d) ∧ 
    ((a = e ∧ b = 0) ∨ (c = e ∧ d = 0)) → n = 2 :=
by 
  sorry

end number_of_equilateral_triangles_l141_141121


namespace floor_sum_equality_l141_141907

theorem floor_sum_equality (a b n x y : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
  (hcoprime : Nat.gcd a b = 1) (heq : a * x + b * y = a ^ n + b ^ n) :
  Int.floor (x / b) + Int.floor (y / a) = Int.floor ((a ^ (n - 1)) / b) + Int.floor ((b ^ (n - 1)) / a) := sorry

end floor_sum_equality_l141_141907


namespace part_I_part_II_l141_141540

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x

theorem part_I (a : ℝ) (h_a : a ≠ 0) :
  (∃ x : ℝ, (x * (f a (1/x))) = 4 * x - 3 ∧ ∀ y, x = y → (x * (f a (1/x))) = 4 * x - 3) →
  a = 2 :=
sorry

noncomputable def f2 (x : ℝ) : ℝ := 2 / x - x

theorem part_II : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f2 x1 > f2 x2 :=
sorry

end part_I_part_II_l141_141540


namespace sum_of_squares_fraction_l141_141728

variable {x1 x2 x3 y1 y2 y3 : ℝ}

theorem sum_of_squares_fraction :
  x1 + x2 + x3 = 0 → y1 + y2 + y3 = 0 → x1 * y1 + x2 * y2 + x3 * y3 = 0 →
  (x1^2 / (x1^2 + x2^2 + x3^2)) + (y1^2 / (y1^2 + y2^2 + y3^2)) = 2 / 3 :=
by
  intros h1 h2 h3
  sorry

end sum_of_squares_fraction_l141_141728


namespace quadrilateral_divided_similarity_iff_trapezoid_l141_141468

noncomputable def convex_quadrilateral (A B C D : Type) : Prop := sorry
noncomputable def is_trapezoid (A B C D : Type) : Prop := sorry
noncomputable def similar_quadrilaterals (E F A B C D : Type) : Prop := sorry

theorem quadrilateral_divided_similarity_iff_trapezoid {A B C D E F : Type}
  (h1 : convex_quadrilateral A B C D)
  (h2 : similar_quadrilaterals E F A B C D): 
  is_trapezoid A B C D ↔ similar_quadrilaterals E F A B C D :=
sorry

end quadrilateral_divided_similarity_iff_trapezoid_l141_141468


namespace proportion_solution_l141_141847

theorem proportion_solution (x : ℝ) : (x ≠ 0) → (1 / 3 = 5 / (3 * x)) → x = 5 :=
by
  intro hnx hproportion
  sorry

end proportion_solution_l141_141847


namespace neither_odd_nor_even_and_min_value_at_one_l141_141377

def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem neither_odd_nor_even_and_min_value_at_one :
  (∀ x, f (-x) ≠ f x ∧ f (-x) ≠ - f x) ∧ ∃ x, x = 1 ∧ ∀ y, f y ≥ f x :=
by
  sorry

end neither_odd_nor_even_and_min_value_at_one_l141_141377


namespace nat_solution_unique_l141_141205

theorem nat_solution_unique (n : ℕ) (h : 2 * n - 1 / n^5 = 3 - 2 / n) : 
  n = 1 :=
sorry

end nat_solution_unique_l141_141205


namespace part1_solution_set_of_inequality_part2_range_of_m_l141_141877

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1_solution_set_of_inequality :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} :=
by
  sorry

theorem part2_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x > 6 * m ^ 2 - 4 * m) ↔ -1/3 < m ∧ m < 1 :=
by
  sorry

end part1_solution_set_of_inequality_part2_range_of_m_l141_141877


namespace sum_reciprocals_squares_l141_141927

theorem sum_reciprocals_squares {a b : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a * b = 11) :
  (1 / (a: ℚ)^2) + (1 / (b: ℚ)^2) = 122 / 121 := 
sorry

end sum_reciprocals_squares_l141_141927


namespace no_real_solutions_l141_141200

theorem no_real_solutions (x : ℝ) (h_nonzero : x ≠ 0) (h_pos : 0 < x):
  (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 12 * x^9 :=
by
-- Proof will go here.
sorry

end no_real_solutions_l141_141200


namespace range_of_m_l141_141379

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 := 
sorry

end range_of_m_l141_141379


namespace cyclic_quadrilateral_fourth_side_length_l141_141953

theorem cyclic_quadrilateral_fourth_side_length
  (r : ℝ) (a b c d : ℝ) (r_eq : r = 300 * Real.sqrt 2) (a_eq : a = 300) (b_eq : b = 400)
  (c_eq : c = 300) :
  d = 500 := 
by 
  sorry

end cyclic_quadrilateral_fourth_side_length_l141_141953


namespace cos_value_of_geometric_sequence_l141_141373

theorem cos_value_of_geometric_sequence (a : ℕ → ℝ) (r : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * r)
  (h2 : a 1 * a 13 + 2 * (a 7) ^ 2 = 5 * Real.pi) :
  Real.cos (a 2 * a 12) = 1 / 2 := 
sorry

end cos_value_of_geometric_sequence_l141_141373


namespace solve_quadratic_l141_141163

theorem solve_quadratic (x : ℝ) : (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3) :=
by
  sorry

end solve_quadratic_l141_141163


namespace slices_dinner_l141_141849

variable (lunch_slices : ℕ) (total_slices : ℕ)
variable (h1 : lunch_slices = 7) (h2 : total_slices = 12)

theorem slices_dinner : total_slices - lunch_slices = 5 :=
by sorry

end slices_dinner_l141_141849


namespace boat_upstream_speed_l141_141317

variable (Vb Vc : ℕ)

def boat_speed_upstream (Vb Vc : ℕ) : ℕ := Vb - Vc

theorem boat_upstream_speed (hVb : Vb = 50) (hVc : Vc = 20) : boat_speed_upstream Vb Vc = 30 :=
by sorry

end boat_upstream_speed_l141_141317


namespace number_of_students_speaking_two_languages_l141_141752

variables (G H M GH GM HM GHM N : ℕ)

def students_speaking_two_languages (G H M GH GM HM GHM N : ℕ) : ℕ :=
  G + H + M - (GH + GM + HM) + GHM

theorem number_of_students_speaking_two_languages 
  (h_total : N = 22)
  (h_G : G = 6)
  (h_H : H = 15)
  (h_M : M = 6)
  (h_GHM : GHM = 1)
  (h_students : N = students_speaking_two_languages G H M GH GM HM GHM N): 
  GH + GM + HM = 6 := 
by 
  unfold students_speaking_two_languages at h_students 
  sorry

end number_of_students_speaking_two_languages_l141_141752


namespace archer_score_below_8_probability_l141_141595

theorem archer_score_below_8_probability :
  ∀ (p10 p9 p8 : ℝ), p10 = 0.2 → p9 = 0.3 → p8 = 0.3 → 
  (1 - (p10 + p9 + p8) = 0.2) :=
by
  intros p10 p9 p8 hp10 hp9 hp8
  rw [hp10, hp9, hp8]
  sorry

end archer_score_below_8_probability_l141_141595


namespace arithmetic_mean_of_fractions_l141_141647

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l141_141647


namespace james_monthly_earnings_l141_141560

theorem james_monthly_earnings :
  let initial_subscribers := 150
  let gifted_subscribers := 50
  let rate_per_subscriber := 9
  let total_subscribers := initial_subscribers + gifted_subscribers
  let total_earnings := total_subscribers * rate_per_subscriber
  total_earnings = 1800 := by
  sorry

end james_monthly_earnings_l141_141560


namespace sum_of_roots_quadratic_l141_141738

theorem sum_of_roots_quadratic :
  ∀ (a b : ℝ), (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a + b = 1) :=
by
  intro a b
  intros
  sorry

end sum_of_roots_quadratic_l141_141738


namespace max_value_2x_minus_y_l141_141882

theorem max_value_2x_minus_y (x y : ℝ) (h₁ : x + y - 1 < 0) (h₂ : x - y ≤ 0) (h₃ : 0 ≤ x) :
  ∃ z, (z = 2 * x - y) ∧ (z ≤ (1 / 2)) :=
sorry

end max_value_2x_minus_y_l141_141882


namespace arithmetic_mean_of_fractions_l141_141646

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l141_141646


namespace find_large_number_l141_141210

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 50000) 
  (h2 : L = 13 * S + 317) : 
  L = 54140 := 
sorry

end find_large_number_l141_141210


namespace dvd_player_movie_ratio_l141_141800

theorem dvd_player_movie_ratio (M D : ℝ) (h1 : D = M + 63) (h2 : D = 81) : D / M = 4.5 :=
by
  sorry

end dvd_player_movie_ratio_l141_141800


namespace boy_speed_in_kmph_l141_141678

-- Define the conditions
def side_length : ℕ := 35
def time_seconds : ℕ := 56

-- Perimeter of the square field
def perimeter : ℕ := 4 * side_length

-- Speed in meters per second
def speed_mps : ℚ := perimeter / time_seconds

-- Speed in kilometers per hour
def speed_kmph : ℚ := speed_mps * (3600 / 1000)

-- Theorem stating the boy's speed is 9 km/hr
theorem boy_speed_in_kmph : speed_kmph = 9 :=
by
  sorry

end boy_speed_in_kmph_l141_141678


namespace max_value_of_p_l141_141014

theorem max_value_of_p
  (p q r s : ℕ)
  (h1 : p < 3 * q)
  (h2 : q < 4 * r)
  (h3 : r < 5 * s)
  (h4 : s < 90)
  (h5 : 0 < s)
  (h6 : 0 < r)
  (h7 : 0 < q)
  (h8 : 0 < p):
  p ≤ 5324 :=
by
  sorry

end max_value_of_p_l141_141014


namespace find_x_in_triangle_l141_141127

theorem find_x_in_triangle 
  (P Q R S: Type) 
  (PQS_is_straight: PQS) 
  (angle_PQR: ℝ)
  (h1: angle_PQR = 110) 
  (angle_RQS : ℝ)
  (h2: angle_RQS = 70)
  (angle_QRS : ℝ)
  (h3: angle_QRS = 3 * angle_x)
  (angle_QSR : ℝ)
  (h4: angle_QSR = angle_x + 14) 
  (triangle_angles_sum : ∀ (a b c: ℝ), a + b + c = 180) : 
  angle_x = 24 :=
by
  sorry

end find_x_in_triangle_l141_141127


namespace average_weight_of_24_boys_l141_141298

theorem average_weight_of_24_boys (A : ℝ) : 
  (24 * A + 8 * 45.15) / 32 = 48.975 → A = 50.25 :=
by
  intro h
  sorry

end average_weight_of_24_boys_l141_141298


namespace lcm_12_18_l141_141975

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l141_141975


namespace part_a_part_b_part_c_l141_141832

section conference_news

variables (α : Type) [Fintype α] [DecidableEq α]

/-- The number of scientists --/
constant num_scientists : ℕ
/-- The number of scientists who initially know the news --/
constant num_know_news : ℕ

/-- The probability that after the coffee break the number of scientists who know the news is 13 is 0 --/
theorem part_a (h1: num_scientists = 18) (h2: num_know_news = 10) :
  (0 : ℝ) = 0 := by sorry

/-- The probability that after the coffee break the number of scientists who know the news is 14 is 1120 / 2431 --/
theorem part_b (h1: num_scientists = 18) (h2: num_know_news = 10) :
  let probability := (1120 : ℝ) / 2431 in probability = 1120 / 2431 := by sorry

/-- The expected number of scientists knowing the news after the coffee break is 14.7 --/
theorem part_c (h1: num_scientists = 18) (h2: num_know_news = 10) :
  let expected_value := (14.7 : ℝ) in expected_value = 147 / 10 := by sorry

end conference_news

end part_a_part_b_part_c_l141_141832


namespace smallest_fraction_numerator_l141_141959

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (4 * b < 9 * a) ∧ 
  (∀ (a' b' : ℕ), a' ≥ 10 ∧ a' ≤ 99 ∧ b' ≥ 10 ∧ b' ≤ 99 ∧ (4 * b' < 9 * a') → b * a' ≥ a * b') ∧ a = 41 :=
sorry

end smallest_fraction_numerator_l141_141959


namespace arithmetic_mean_of_fractions_l141_141643

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l141_141643


namespace initial_average_customers_l141_141192

theorem initial_average_customers (x A : ℕ) (h1 : x = 1) (h2 : (A + 120) / 2 = 90) : A = 60 := by
  sorry

end initial_average_customers_l141_141192


namespace repeating_decimal_sum_l141_141594

/--
The number 3.17171717... can be written as a reduced fraction x/y where x = 314 and y = 99.
We aim to prove that the sum of x and y is 413.
-/
theorem repeating_decimal_sum : 
  let x := 314
  let y := 99
  (x + y) = 413 := 
by
  sorry

end repeating_decimal_sum_l141_141594


namespace katya_needs_at_least_ten_l141_141255

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l141_141255


namespace principal_amount_borrowed_l141_141183

theorem principal_amount_borrowed 
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 9000) 
  (h2 : R = 0.12) 
  (h3 : T = 3) 
  (h4 : SI = P * R * T) : 
  P = 25000 :=
sorry

end principal_amount_borrowed_l141_141183


namespace solve_for_x_l141_141232

theorem solve_for_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : 6 * x^3 + 18 * x^2 * y * z = 3 * x^4 + 6 * x^3 * y * z) : 
  x = 2 := 
by sorry

end solve_for_x_l141_141232


namespace smallest_fraction_numerator_l141_141956

theorem smallest_fraction_numerator (a b : ℕ) (h1 : 10 ≤ a) (h2 : a ≤ 99) (h3 : 10 ≤ b) (h4 : b ≤ 99) (h5 : 9 * a > 4 * b) (smallest : ∀ c d, 10 ≤ c → c ≤ 99 → 10 ≤ d → d ≤ 99 → 9 * c > 4 * d → (a * d ≤ b * c) → a * d ≤ 41 * 92) :
  a = 41 :=
by
  sorry

end smallest_fraction_numerator_l141_141956


namespace sum_of_digits_is_base_6_l141_141885

def is_valid_digit (x : ℕ) : Prop := x > 0 ∧ x < 6 
def distinct_3 (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a  

theorem sum_of_digits_is_base_6 :
  ∃ (S H E : ℕ), is_valid_digit S ∧ is_valid_digit H ∧ is_valid_digit E
  ∧ distinct_3 S H E 
  ∧ (E + E) % 6 = S 
  ∧ (S + H) % 6 = E 
  ∧ (S + H + E) % 6 = 11 % 6 :=
by 
  sorry

end sum_of_digits_is_base_6_l141_141885


namespace initial_oranges_count_l141_141461

theorem initial_oranges_count
  (initial_apples : ℕ := 50)
  (apple_cost : ℝ := 0.80)
  (orange_cost : ℝ := 0.50)
  (total_earnings : ℝ := 49)
  (remaining_apples : ℕ := 10)
  (remaining_oranges : ℕ := 6)
  : initial_oranges = 40 := 
by
  sorry

end initial_oranges_count_l141_141461


namespace symm_y_axis_l141_141030

noncomputable def f (x : ℝ) : ℝ := abs x

theorem symm_y_axis (x : ℝ) : f (-x) = f (x) := by
  sorry

end symm_y_axis_l141_141030


namespace prob_chemistry_prob_biology_union_history_prob_chemistry_union_geography_l141_141180

variables (students : Type) [fintype students] [decidable_eq students]
variables (physics chemistry biology politics history geography : set students)
variables (n : ℕ) (total_students := 1000)

-- Students counts based on conditions
variables (h_total : fintype.card students = total_students)
variables (h_physics : fintype.card physics = 300)
variables (h_chemistry : fintype.card chemistry = 200)
variables (h_biology : fintype.card biology = 100)
variables (h_politics : fintype.card politics = 200)
variables (h_history : fintype.card history = 100)
variables (h_geography : fintype.card geography = 100)

namespace high_school_probabilities

-- Probability of selecting a student from a set
def P (s : set students) : ℚ := fintype.card s / total_students

-- Problem 1: Prove P(B) = 1/5
theorem prob_chemistry : P chemistry = 1/5 :=
sorry

-- Problem 2: Prove P(C ∪ E) = 1/5
theorem prob_biology_union_history : P (biology ∪ history) = 1/5 :=
sorry

-- Problem 3: Prove P(B ∪ F) = 3/10
theorem prob_chemistry_union_geography : P (chemistry ∪ geography) = 3/10 :=
sorry

end high_school_probabilities

end prob_chemistry_prob_biology_union_history_prob_chemistry_union_geography_l141_141180


namespace probability_both_meat_given_same_l141_141017

open ProbabilityTheory

-- Definition of the problem conditions
def total_dumplings : Finset (Fin 5) := Finset.univ
def meat_dumplings : Finset (Fin 5) := {0, 1} -- using the first 2 as meat filled
def red_bean_paste_dumplings : Finset (Fin 5) := {2, 3, 4} -- the remaining are red bean filled

def event_same_filling (x y : Fin 5) : Prop :=
  (x ∈ meat_dumplings ∧ y ∈ meat_dumplings) ∨ (x ∈ red_bean_paste_dumplings ∧ y ∈ red_bean_paste_dumplings)

def event_both_meat (x y : Fin 5) : Prop :=
  x ∈ meat_dumplings ∧ y ∈ meat_dumplings

-- Probability calculations
noncomputable def probability_same_filling : ℚ :=
  (Finset.card ((total_dumplings.product total_dumplings).filter (λ p, event_same_filling p.1 p.2))).toRat /
  (Finset.card (total_dumplings.product total_dumplings)).toRat

noncomputable def probability_both_meat : ℚ :=
  (Finset.card ((total_dumplings.product total_dumplings).filter (λ p, event_both_meat p.1 p.2))).toRat /
  (Finset.card (total_dumplings.product total_dumplings)).toRat

noncomputable def conditional_probability_both_meat_given_same_filling : ℚ :=
  probability_both_meat / probability_same_filling

-- The main theorem statement
theorem probability_both_meat_given_same : 
  conditional_probability_both_meat_given_same_filling = 1 / 4 :=
by
  sorry

end probability_both_meat_given_same_l141_141017


namespace oxen_count_l141_141480

theorem oxen_count (B C O : ℕ) (H1 : 3 * B = 4 * C) (H2 : 3 * B = 2 * O) (H3 : 15 * B + 24 * C + O * O = 33 * B + (3 / 2) * O * B) (H4 : 24 * B = 48) (H5 : 60 * C + 30 * B + 18 * (O * (3 / 2) * B) = 108 * B + (3 / 2) * O * B * 18)
: O = 8 :=
by 
  sorry

end oxen_count_l141_141480


namespace soda_cost_l141_141333

variable {b s f : ℕ}

theorem soda_cost :
    5 * b + 3 * s + 2 * f = 520 ∧
    3 * b + 2 * s + f = 340 →
    s = 80 :=
by
  sorry

end soda_cost_l141_141333


namespace min_y_in_quadratic_l141_141173

theorem min_y_in_quadratic (x : ℝ) : ∃ y : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ y', (y' = x^2 + 16 * x + 20) → y ≤ y' := 
sorry

end min_y_in_quadratic_l141_141173


namespace insurance_covers_90_percent_l141_141129

-- We firstly define the variables according to the conditions.
def adoption_fee : ℕ := 150
def training_cost_per_week : ℕ := 250
def training_weeks : ℕ := 12
def certification_cost : ℕ := 3000
def total_out_of_pocket_cost : ℕ := 3450

-- We now compute intermediate results based on the conditions provided.
def total_training_cost : ℕ := training_cost_per_week * training_weeks
def out_of_pocket_cert_cost : ℕ := total_out_of_pocket_cost - adoption_fee - total_training_cost
def insurance_coverage_amount : ℕ := certification_cost - out_of_pocket_cert_cost
def insurance_coverage_percentage : ℕ := (insurance_coverage_amount * 100) / certification_cost

-- Now, we state the theorem that needs to be proven.
theorem insurance_covers_90_percent : insurance_coverage_percentage = 90 := by
  sorry

end insurance_covers_90_percent_l141_141129


namespace lcm_12_18_l141_141977

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l141_141977


namespace find_y_l141_141891

theorem find_y (x y : ℤ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 :=
by
  subst h1
  have h : 3 * 4 + 2 * y = 30 := by rw [h2]
  linarith

end find_y_l141_141891


namespace factorization_of_expression_l141_141084

theorem factorization_of_expression (x y : ℝ) : x^2 - x * y = x * (x - y) := 
by
  sorry

end factorization_of_expression_l141_141084


namespace amount_earned_from_each_family_l141_141598

theorem amount_earned_from_each_family
  (goal : ℕ) (earn_from_fifteen_families : ℕ) (additional_needed : ℕ) (three_families : ℕ) 
  (earn_from_three_families_total : ℕ) (per_family_earn : ℕ) :
  goal = 150 →
  earn_from_fifteen_families = 75 →
  additional_needed = 45 →
  three_families = 3 →
  earn_from_three_families_total = (goal - additional_needed) - earn_from_fifteen_families →
  per_family_earn = earn_from_three_families_total / three_families →
  per_family_earn = 10 :=
by
  sorry

end amount_earned_from_each_family_l141_141598


namespace confectioner_customers_l141_141181

theorem confectioner_customers (x : ℕ) (h : 0 < x) :
  (49 * (392 / x - 6) = 392) → x = 28 :=
by
sorry

end confectioner_customers_l141_141181


namespace cups_of_flour_per_pound_of_pasta_l141_141569

-- Definitions from conditions
def pounds_of_pasta_per_rack : ℕ := 3
def racks_owned : ℕ := 3
def additional_rack_needed : ℕ := 1
def cups_per_bag : ℕ := 8
def bags_used : ℕ := 3

-- Derived definitions from above conditions
def total_cups_of_flour : ℕ := bags_used * cups_per_bag  -- 24 cups
def total_racks_needed : ℕ := racks_owned + additional_rack_needed  -- 4 racks
def total_pounds_of_pasta : ℕ := total_racks_needed * pounds_of_pasta_per_rack  -- 12 pounds

theorem cups_of_flour_per_pound_of_pasta (x : ℕ) :
  (total_cups_of_flour / total_pounds_of_pasta) = x → x = 2 :=
by
  intro h
  sorry

end cups_of_flour_per_pound_of_pasta_l141_141569


namespace extremum_at_x_1_max_integer_k_l141_141869

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * Real.log x - (a + 1) * x

theorem extremum_at_x_1 (a : ℝ) : (∀ x : ℝ, 0 < x → ((Real.log x - 1 / x - a = 0) ↔ x = 1))
  → a = -1 ∧
  (∀ x : ℝ, 0 < x → (Real.log x - 1 / x + 1) < 0 → f x (-1) < f 1 (-1) ∧
  (Real.log x - 1 / x + 1) > 0 → f 1 (-1) < f x (-1)) :=
sorry

theorem max_integer_k (k : ℤ) :
  (∀ x : ℝ, 0 < x → (f x 1 > k))
  → k ≤ -4 :=
sorry

end extremum_at_x_1_max_integer_k_l141_141869


namespace felicity_collecting_weeks_l141_141204

-- Define the conditions
def fort_total_sticks : ℕ := 400
def fort_completion_percent : ℝ := 0.60
def store_visits_per_week : ℕ := 3

-- Define the proof problem
theorem felicity_collecting_weeks :
  let collected_sticks := (fort_completion_percent * fort_total_sticks).to_nat
  in collected_sticks / store_visits_per_week = 80 := by
  -- This will be proven in the proof section, currently left as sorry
  sorry

end felicity_collecting_weeks_l141_141204


namespace part1_part2_l141_141778

-- Definition of the function
def f (a x : ℝ) := |x - a|

-- Proof statement for question 1
theorem part1 (a : ℝ)
  (h : ∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) :
  a = 3 := by
  sorry

-- Auxiliary function for question 2
def g (a x : ℝ) := f a (2 * x) + f a (x + 2)

-- Proof statement for question 2
theorem part2 (m : ℝ)
  (h : ∀ x : ℝ, g 3 x ≥ m) :
  m ≤ 1/2 := by
  sorry

end part1_part2_l141_141778


namespace area_triangle_l141_141557

theorem area_triangle (A B C: ℝ) (AB AC : ℝ) (h1 : Real.sin A = 4 / 5) (h2 : AB * AC * Real.cos A = 6) :
  (1 / 2) * AB * AC * Real.sin A = 4 :=
by
  sorry

end area_triangle_l141_141557


namespace not_p_and_p_l141_141021

theorem not_p_and_p (p : Prop) : ¬ (p ∧ ¬ p) :=
by 
  sorry

end not_p_and_p_l141_141021


namespace syllogism_example_l141_141052

-- Definitions based on the conditions
def is_even (n : ℕ) := n % 2 = 0
def is_divisible_by_2 (n : ℕ) := n % 2 = 0

-- Given conditions:
axiom even_implies_divisible_by_2 : ∀ n : ℕ, is_even n → is_divisible_by_2 n
axiom h2012_is_even : is_even 2012

-- Proving the conclusion and the syllogism pattern
theorem syllogism_example : is_divisible_by_2 2012 :=
by
  apply even_implies_divisible_by_2
  apply h2012_is_even

end syllogism_example_l141_141052


namespace amount_spent_on_belt_correct_l141_141071

variable (budget shirt pants coat socks shoes remaining : ℕ)

-- Given conditions
def initial_budget : ℕ := 200
def spent_shirt : ℕ := 30
def spent_pants : ℕ := 46
def spent_coat : ℕ := 38
def spent_socks : ℕ := 11
def spent_shoes : ℕ := 41
def remaining_amount : ℕ := 16

-- The amount spent on the belt
def amount_spent_on_belt : ℕ :=
  budget - remaining - (shirt + pants + coat + socks + shoes)

-- The theorem statement we need to prove
theorem amount_spent_on_belt_correct :
  initial_budget = budget →
  spent_shirt = shirt →
  spent_pants = pants →
  spent_coat = coat →
  spent_socks = socks →
  spent_shoes = shoes →
  remaining_amount = remaining →
  amount_spent_on_belt budget shirt pants coat socks shoes remaining = 18 := by
    simp [initial_budget, spent_shirt, spent_pants, spent_coat, spent_socks, spent_shoes, remaining_amount, amount_spent_on_belt]
    sorry

end amount_spent_on_belt_correct_l141_141071


namespace general_term_formula_l141_141727

variable (a : ℕ → ℤ) -- A sequence of integers 
variable (d : ℤ) -- The common difference 

-- Conditions provided
axiom h1 : a 1 = 6
axiom h2 : a 3 + a 5 = 0
axiom h_arithmetic : ∀ n, a (n + 1) = a n + d -- Arithmetic progression condition

-- The general term formula we need to prove
theorem general_term_formula : ∀ n, a n = 8 - 2 * n := 
by 
  sorry -- Proof goes here


end general_term_formula_l141_141727


namespace playground_girls_l141_141458

theorem playground_girls (total_children boys girls : ℕ) (h1 : boys = 40) (h2 : total_children = 117) (h3 : total_children = boys + girls) : girls = 77 := 
by 
  sorry

end playground_girls_l141_141458


namespace correctly_calculated_value_l141_141229

theorem correctly_calculated_value : 
  ∃ x : ℝ, (x + 4 = 40) ∧ (x / 4 = 9) :=
sorry

end correctly_calculated_value_l141_141229


namespace bears_total_l141_141213

-- Define the number of each type of bear
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27
def polar_bears : ℕ := 12
def grizzly_bears : ℕ := 18

-- Define the total number of bears
def total_bears : ℕ := brown_bears + white_bears + black_bears + polar_bears + grizzly_bears

-- The theorem stating the total number of bears is 96
theorem bears_total : total_bears = 96 :=
by
  -- The proof is omitted here
  sorry

end bears_total_l141_141213


namespace sum_of_A_and_B_l141_141815

theorem sum_of_A_and_B (A B : ℕ) (h1 : A ≠ B) (h2 : A < 10) (h3 : B < 10) :
  (10 * A + B) * 6 = 111 * B → A + B = 11 :=
by
  intros h
  sorry

end sum_of_A_and_B_l141_141815


namespace program_total_cost_l141_141475

-- Define the necessary variables and constants
def ms_to_s : Float := 0.001
def os_overhead : Float := 1.07
def cost_per_ms : Float := 0.023
def mount_cost : Float := 5.35
def time_required : Float := 1.5

-- Calculate components of the total cost
def total_cost_for_computer_time := (time_required * 1000) * cost_per_ms
def total_cost := os_overhead + total_cost_for_computer_time + mount_cost

-- State the theorem
theorem program_total_cost : total_cost = 40.92 := by
  sorry

end program_total_cost_l141_141475


namespace perimeter_of_square_with_area_625_cm2_l141_141497

noncomputable def side_length (a : ℝ) : ℝ := 
  real.sqrt a

noncomputable def perimeter (s : ℝ) : ℝ :=
  4 * s

theorem perimeter_of_square_with_area_625_cm2 :
  perimeter (side_length 625) = 100 :=
by
  sorry

end perimeter_of_square_with_area_625_cm2_l141_141497


namespace percentage_of_female_employees_l141_141124

theorem percentage_of_female_employees (E : ℕ) (hE : E = 1400) 
  (pct_computer_literate : ℚ) (hpct : pct_computer_literate = 0.62)
  (female_computer_literate : ℕ) (hfcl : female_computer_literate = 588)
  (pct_male_computer_literate : ℚ) (hmcl : pct_male_computer_literate = 0.5) :
  100 * (840 / 1400) = 60 := 
by
  sorry

end percentage_of_female_employees_l141_141124


namespace percent_c_of_b_l141_141119

variable (a b c : ℝ)

theorem percent_c_of_b (h1 : c = 0.20 * a) (h2 : b = 2 * a) : 
  ∃ x : ℝ, c = (x / 100) * b ∧ x = 10 :=
by
  sorry

end percent_c_of_b_l141_141119


namespace find_coefficients_sum_l141_141363

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 : ℝ) (h : ∀ x : ℝ, x^3 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3) :
  a_1 + a_2 + a_3 = 19 :=
by
  sorry

end find_coefficients_sum_l141_141363


namespace complement_intersection_l141_141381

open Finset

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 3, 4}
def B : Finset ℕ := {3, 5}

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 2, 4, 5} :=
by sorry

end complement_intersection_l141_141381


namespace muffins_sold_in_afternoon_l141_141766

variable (total_muffins : ℕ)
variable (morning_muffins : ℕ)
variable (remaining_muffins : ℕ)

theorem muffins_sold_in_afternoon 
  (h1 : total_muffins = 20) 
  (h2 : morning_muffins = 12) 
  (h3 : remaining_muffins = 4) : 
  (total_muffins - remaining_muffins - morning_muffins) = 4 := 
by
  sorry

end muffins_sold_in_afternoon_l141_141766


namespace solve_inequality_l141_141928

theorem solve_inequality (x : ℝ) : 2 * x^2 - x - 1 > 0 ↔ x < -1/2 ∨ x > 1 :=
by
  sorry

end solve_inequality_l141_141928


namespace positive_two_digit_integers_remainder_4_div_9_l141_141883

theorem positive_two_digit_integers_remainder_4_div_9 : ∃ (n : ℕ), 
  (10 ≤ 9 * n + 4) ∧ (9 * n + 4 < 100) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 10 ∧ ∀ m, 1 ≤ m ∧ m ≤ 10 → n = k) :=
by
  sorry

end positive_two_digit_integers_remainder_4_div_9_l141_141883


namespace hemisphere_surface_area_l141_141669

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (area_base : ℝ) (surface_area_sphere : ℝ) (Q : ℝ) : 
  area_base = 3 ∧ surface_area_sphere = 4 * π * r^2 → Q = 9 :=
by
  sorry

end hemisphere_surface_area_l141_141669


namespace Isabel_reading_pages_l141_141005

def pages_of_math_homework : ℕ := 2
def problems_per_page : ℕ := 5
def total_problems : ℕ := 30

def math_problems : ℕ := pages_of_math_homework * problems_per_page
def reading_problems : ℕ := total_problems - math_problems

theorem Isabel_reading_pages : (reading_problems / problems_per_page) = 4 :=
by
  sorry

end Isabel_reading_pages_l141_141005


namespace rope_cut_prob_l141_141184

theorem rope_cut_prob (x : ℝ) (hx : 0 < x) : 
  (∃ (a b : ℝ), a + b = 1 ∧ min a b ≤ max a b / x) → 
  (1 / (x + 1) * 2) = 2 / (x + 1) :=
sorry

end rope_cut_prob_l141_141184


namespace like_apple_orange_mango_l141_141318

theorem like_apple_orange_mango (A B C: ℕ) 
  (h1: A = 40) 
  (h2: B = 7) 
  (h3: C = 10) 
  (total: ℕ) 
  (h_total: total = 47) 
: ∃ x: ℕ, 40 + (10 - x) + x = 47 ∧ x = 3 := 
by 
  sorry

end like_apple_orange_mango_l141_141318


namespace smallest_integer_to_multiply_y_to_make_perfect_square_l141_141136

noncomputable def y : ℕ :=
  3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_multiply_y_to_make_perfect_square :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (k * y) = m^2) ∧ k = 3 := by
  sorry

end smallest_integer_to_multiply_y_to_make_perfect_square_l141_141136


namespace union_of_A_and_B_l141_141776

open Set

theorem union_of_A_and_B : 
  let A := {x : ℝ | x + 2 > 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = Real.cos x}
  A ∪ B = {z : ℝ | z > -2} := 
by
  intros
  sorry

end union_of_A_and_B_l141_141776


namespace positive_integer_solutions_l141_141981

theorem positive_integer_solutions (n x y z : ℕ) (h1 : n > 1) (h2 : n^z < 2001) (h3 : n^x + n^y = n^z) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ x = k ∧ y = k ∧ z = k + 1) :=
sorry

end positive_integer_solutions_l141_141981


namespace rate_per_meter_for_fencing_l141_141444

/-- The length of a rectangular plot is 10 meters more than its width. 
    The cost of fencing the plot along its perimeter at a certain rate per meter is Rs. 1430. 
    The perimeter of the plot is 220 meters. 
    Prove that the rate per meter for fencing the plot is 6.5 Rs. 
 -/
theorem rate_per_meter_for_fencing (width length perimeter cost : ℝ)
  (h_length : length = width + 10)
  (h_perimeter : perimeter = 2 * (width + length))
  (h_perimeter_value : perimeter = 220)
  (h_cost : cost = 1430) :
  (cost / perimeter) = 6.5 := by
  sorry

end rate_per_meter_for_fencing_l141_141444


namespace shop_makes_off_each_jersey_l141_141279

theorem shop_makes_off_each_jersey :
  ∀ (T : ℝ) (jersey_earnings : ℝ),
  (T = 25) →
  (jersey_earnings = T + 90) →
  jersey_earnings = 115 := by
  intros T jersey_earnings ht hj
  sorry

end shop_makes_off_each_jersey_l141_141279


namespace S6_equals_63_l141_141536

variable {S : ℕ → ℕ}

-- Define conditions
axiom S_n_geometric_sequence (a : ℕ → ℕ) (n : ℕ) : n ≥ 1 → S n = (a 0) * ((a 1)^(n) -1) / (a 1 - 1)
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- State theorem
theorem S6_equals_63 : S 6 = 63 := by
  sorry

end S6_equals_63_l141_141536


namespace grover_total_profit_is_15_l141_141543

theorem grover_total_profit_is_15 
  (boxes : ℕ) 
  (masks_per_box : ℕ) 
  (price_per_mask : ℝ) 
  (cost_of_boxes : ℝ) 
  (total_profit : ℝ)
  (hb : boxes = 3)
  (hm : masks_per_box = 20)
  (hp : price_per_mask = 0.5)
  (hc : cost_of_boxes = 15)
  (htotal : total_profit = (boxes * masks_per_box) * price_per_mask - cost_of_boxes) :
  total_profit = 15 :=
sorry

end grover_total_profit_is_15_l141_141543


namespace arithmetic_mean_of_fractions_l141_141621

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l141_141621


namespace white_chocolate_bars_sold_l141_141451

theorem white_chocolate_bars_sold (W D : ℕ) (h1 : D = 15) (h2 : W / D = 4 / 3) : W = 20 :=
by
  -- This is where the proof would go.
  sorry

end white_chocolate_bars_sold_l141_141451


namespace product_is_zero_l141_141350

theorem product_is_zero (b : ℤ) (h : b = 3) :
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b * (b + 1) * (b + 2) = 0 :=
by {
  -- Substituting b = 3
  -- (3-5) * (3-4) * (3-3) * (3-2) * (3-1) * 3 * (3+1) * (3+2)
  -- = (-2) * (-1) * 0 * 1 * 2 * 3 * 4 * 5
  -- = 0
  sorry
}

end product_is_zero_l141_141350


namespace marble_probability_l141_141182

theorem marble_probability :
  let total_marbles := 4 + 5 + 11
  let prob_red := 4 / total_marbles
  let prob_green := 5 / (total_marbles - 1)
  let prob_white := 11 / (total_marbles - 2)
  (prob_red * prob_green * prob_white) = 11 / 342 :=
by {
  let total_marbles := 4 + 5 + 11
  let prob_red := 4 / total_marbles
  let prob_green := 5 / (total_marbles - 1)
  let prob_white := 11 / (total_marbles - 2)
  rw [show total_marbles = 20 from rfl],
  rw [show prob_red = 4 / 20 from rfl],
  rw [show prob_green = 5 / 19 from rfl],
  rw [show prob_white = 11 / 18 from rfl],
  norm_num,
  ring,
  norm_num,
  exact rfl
}

end marble_probability_l141_141182


namespace hamburger_price_l141_141365

theorem hamburger_price (P : ℝ) 
    (h1 : 2 * 4 + 2 * 2 = 12) 
    (h2 : 12 * P + 4 * P = 50) : 
    P = 3.125 := 
by
  -- sorry added to skip the proof.
  sorry

end hamburger_price_l141_141365


namespace cubic_yard_to_cubic_meter_l141_141228

/-- Define the conversion from yards to meters. -/
def yard_to_meter : ℝ := 0.9144

/-- Theorem stating how many cubic meters are in one cubic yard. -/
theorem cubic_yard_to_cubic_meter :
  (yard_to_meter ^ 3 : ℝ) = 0.7636 :=
by
  sorry

end cubic_yard_to_cubic_meter_l141_141228


namespace area_of_triangle_ABC_l141_141968

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 2 }
def B : Point := { x := 6, y := 0 }
def C : Point := { x := 4, y := 7 }

def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * abs (P1.x * (P2.y - P3.y) +
             P2.x * (P3.y - P1.y) +
             P3.x * (P1.y - P2.y))

theorem area_of_triangle_ABC : triangle_area A B C = 19 :=
by
  sorry

end area_of_triangle_ABC_l141_141968


namespace a_and_b_work_together_l141_141175
noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem a_and_b_work_together (A_days B_days : ℕ) (hA : A_days = 32) (hB : B_days = 32) :
  (1 / work_rate A_days + 1 / work_rate B_days) = 16 := by
  sorry

end a_and_b_work_together_l141_141175


namespace xy_pos_iff_div_pos_ab_leq_mean_sq_l141_141046

-- Definition for question 1
theorem xy_pos_iff_div_pos (x y : ℝ) : 
  (x * y > 0) ↔ (x / y > 0) :=
sorry

-- Definition for question 3
theorem ab_leq_mean_sq (a b : ℝ) : 
  a * b ≤ ((a + b) / 2) ^ 2 :=
sorry

end xy_pos_iff_div_pos_ab_leq_mean_sq_l141_141046


namespace circle_symmetric_about_line_l141_141158

-- The main proof statement
theorem circle_symmetric_about_line (x y : ℝ) (k : ℝ) :
  (x - 1)^2 + (y - 1)^2 = 2 ∧ y = k * x + 3 → k = -2 :=
by
  sorry

end circle_symmetric_about_line_l141_141158


namespace perimeter_of_region_is_70_l141_141924

-- Define the given conditions
def area_of_region (total_area : ℝ) (num_squares : ℕ) : Prop :=
  total_area = 392 ∧ num_squares = 8

def side_length_of_square (area : ℝ) (side_length : ℝ) : Prop :=
  area = side_length^2 ∧ side_length = 7

def perimeter_of_region (num_squares : ℕ) (side_length : ℝ) (perimeter : ℝ) : Prop :=
  perimeter = 8 * side_length + 2 * side_length ∧ perimeter = 70

-- Statement to prove
theorem perimeter_of_region_is_70 :
  ∀ (total_area : ℝ) (num_squares : ℕ), 
    area_of_region total_area num_squares →
    ∃ (side_length : ℝ) (perimeter : ℝ), 
      side_length_of_square (total_area / num_squares) side_length ∧
      perimeter_of_region num_squares side_length perimeter :=
by {
  sorry
}

end perimeter_of_region_is_70_l141_141924


namespace perfect_score_l141_141493

theorem perfect_score (P : ℕ) (h : 3 * P = 63) : P = 21 :=
by
  -- Proof to be provided
  sorry

end perfect_score_l141_141493


namespace inequality_solution_set_l141_141162

theorem inequality_solution_set (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end inequality_solution_set_l141_141162


namespace unique_increasing_seq_l141_141591

noncomputable def unique_seq (a : ℕ → ℕ) (r : ℝ) : Prop :=
∀ (b : ℕ → ℕ), (∀ n, b n = 3 * n - 2 → ∑' n, r ^ (b n) = 1 / 2 ) → (∀ n, a n = b n)

theorem unique_increasing_seq {r : ℝ} 
  (hr : 0.4 < r ∧ r < 0.5) 
  (hc : r^3 + 2*r = 1):
  ∃ a : ℕ → ℕ, (∀ n, a n = 3 * n - 2) ∧ (∑'(n), r^(a n) = 1/2) ∧ unique_seq a r :=
by
  sorry

end unique_increasing_seq_l141_141591


namespace exists_m_for_n_divides_2_pow_m_plus_m_l141_141362

theorem exists_m_for_n_divides_2_pow_m_plus_m (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, 0 < m ∧ n ∣ 2^m + m :=
sorry

end exists_m_for_n_divides_2_pow_m_plus_m_l141_141362


namespace shaniqua_style_income_correct_l141_141147

def shaniqua_income_per_style (haircut_income : ℕ) (total_income : ℕ) (number_of_haircuts : ℕ) (number_of_styles : ℕ) : ℕ :=
  (total_income - (number_of_haircuts * haircut_income)) / number_of_styles

theorem shaniqua_style_income_correct :
  shaniqua_income_per_style 12 221 8 5 = 25 :=
by
  sorry

end shaniqua_style_income_correct_l141_141147


namespace min_value_y_l141_141172

theorem min_value_y : ∃ x : ℝ, (∀ y : ℝ, y = x^2 + 16 * x + 20 → y ≥ -44) :=
begin
  use -8,
  intro y,
  intro hy,
  suffices : y = (x + 8)^2 - 44,
  { rw this,
    exact sub_nonneg_of_le (sq_nonneg (x + 8)) },
  sorry
end

end min_value_y_l141_141172


namespace workers_contribution_l141_141472

theorem workers_contribution (W C : ℕ) 
    (h1 : W * C = 300000) 
    (h2 : W * (C + 50) = 325000) : 
    W = 500 :=
by
    sorry

end workers_contribution_l141_141472


namespace probability_collinear_dots_l141_141405

theorem probability_collinear_dots 
  (rows : ℕ) (cols : ℕ) (total_dots : ℕ) (collinear_sets : ℕ) (total_ways : ℕ) : 
  rows = 5 → cols = 4 → total_dots = 20 → collinear_sets = 20 → total_ways = 4845 → 
  (collinear_sets : ℚ) / total_ways = 4 / 969 :=
by
  intros hrows hcols htotal_dots hcollinear_sets htotal_ways
  sorry

end probability_collinear_dots_l141_141405


namespace part1_part2_l141_141102

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a * real.log x) / x + x
noncomputable def g (x : ℝ) : ℝ := 2 * x - 1 / x

theorem part1 (a : ℝ): 
    (∃! x : ℝ, 1 ≤ x ∧ x ≤ real.exp 1 ∧ f a x = g x) ↔ 
    a ∈ set.Iic 1 ∪ set.Ioi ((real.exp 2 - 1) / 2) := sorry

theorem part2 (a : ℝ): 
    (∃ t0 ∈ set.Icc 1 (real.exp 1), ∀ x ≥ t0, x * f a x > x^2 + 2 * (a + 1) / x + 2 * x) ↔ 
    a ∈ set.Iio (-2) ∪ set.Ioi ((real.exp 2 + 1) / (real.exp 1 - 1)) := sorry

end part1_part2_l141_141102


namespace time_for_B_alone_l141_141176

theorem time_for_B_alone (W_A W_B : ℝ) (h1 : W_A = 2 * W_B) (h2 : W_A + W_B = 1/6) : 1 / W_B = 18 := by
  sorry

end time_for_B_alone_l141_141176


namespace jasmine_first_exceed_500_l141_141901

theorem jasmine_first_exceed_500 {k : ℕ} (initial : ℕ) (factor : ℕ) :
  initial = 5 → factor = 4 → (5 * 4^k > 500) → k = 4 :=
by
  sorry

end jasmine_first_exceed_500_l141_141901


namespace syllogism_error_l141_141521

-- Definitions based on conditions from a)
def major_premise (a: ℝ) : Prop := a^2 > 0

def minor_premise (a: ℝ) : Prop := true

-- Theorem stating that the conclusion does not necessarily follow
theorem syllogism_error (a : ℝ) (h_minor : minor_premise a) : ¬major_premise 0 :=
by
  sorry

end syllogism_error_l141_141521


namespace hispanic_population_in_west_l141_141961

theorem hispanic_population_in_west (p_NE p_MW p_South p_West : ℕ)
  (h_NE : p_NE = 4)
  (h_MW : p_MW = 5)
  (h_South : p_South = 12)
  (h_West : p_West = 20) :
  ((p_West : ℝ) / (p_NE + p_MW + p_South + p_West : ℝ)) * 100 = 49 :=
by sorry

end hispanic_population_in_west_l141_141961


namespace total_highlighters_l141_141829

-- Define the number of highlighters of each color
def pink_highlighters : ℕ := 10
def yellow_highlighters : ℕ := 15
def blue_highlighters : ℕ := 8

-- Prove the total number of highlighters
theorem total_highlighters : pink_highlighters + yellow_highlighters + blue_highlighters = 33 :=
by
  sorry

end total_highlighters_l141_141829


namespace monotonic_increasing_quadratic_l141_141447

theorem monotonic_increasing_quadratic (b : ℝ) (c : ℝ) :
  (∀ x y : ℝ, (0 ≤ x → x ≤ y → (x^2 + b*x + c) ≤ (y^2 + b*y + c))) ↔ (b ≥ 0) :=
sorry  -- Proof is omitted

end monotonic_increasing_quadratic_l141_141447


namespace pi_bounds_l141_141274

theorem pi_bounds : 
  3.14 < Real.pi ∧ Real.pi < 3.142 ∧
  9.86 < Real.pi ^ 2 ∧ Real.pi ^ 2 < 9.87 := sorry

end pi_bounds_l141_141274


namespace tangent_line_eq_l141_141835

theorem tangent_line_eq (x y : ℝ) (h : y = e^(-5 * x) + 2) :
  ∀ (t : ℝ), t = 0 → y = 3 → y = -5 * x + 3 :=
by
  sorry

end tangent_line_eq_l141_141835


namespace geometry_intersection_ellipse_proof_l141_141790

noncomputable def intersection_is_ellipse (R : ℝ) (φ : ℝ) : Prop :=
  ∀ (cylinder : Cylinder ℝ) (plane : Plane ℝ),
  (plane.intersects_cylinder_lateral_surface cylinder ∧ 
   ¬ plane.is_perpendicular_to_cylinder_axis cylinder ∧ 
   ¬ plane.intersects_cylinder_bases cylinder)
   → 
  (plane.intersection_with_cylinder cylinder).is_ellipse ∧
  (plane.intersection_with_cylinder cylinder).major_diameter = 2 * R ∧
  (plane.intersection_with_cylinder cylinder).minor_diameter = 2 * R * Real.cos φ

theorem geometry_intersection_ellipse_proof {R φ : ℝ} (cylinder : Cylinder ℝ) (plane : Plane ℝ)
  (h_plane_intersects : plane.intersects_cylinder_lateral_surface cylinder)
  (h_plane_not_perpendicular : ¬ plane.is_perpendicular_to_cylinder_axis cylinder)
  (h_plane_not_intersects_bases : ¬ plane.intersects_cylinder_bases cylinder) :
  (plane.intersection_with_cylinder cylinder).is_ellipse ∧
  (plane.intersection_with_cylinder cylinder).major_diameter = 2 * R ∧
  (plane.intersection_with_cylinder cylinder).minor_diameter = 2 * R * Real.cos φ :=
by
  sorry

end geometry_intersection_ellipse_proof_l141_141790


namespace compute_fraction_l141_141271

noncomputable def distinct_and_sum_zero (w x y z : ℝ) : Prop :=
w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ w + x + y + z = 0

theorem compute_fraction (w x y z : ℝ) (h : distinct_and_sum_zero w x y z) :
  (w * y + x * z) / (w^2 + x^2 + y^2 + z^2) = -1 / 2 :=
sorry

end compute_fraction_l141_141271


namespace allowance_is_14_l141_141719

def initial := 11
def spent := 3
def final := 22

def allowance := final - (initial - spent)

theorem allowance_is_14 : allowance = 14 := by
  -- proof goes here
  sorry

end allowance_is_14_l141_141719


namespace arrange_numbers_l141_141335

noncomputable def a := (10^100)^10
noncomputable def b := 10^(10^10)
noncomputable def c := Nat.factorial 1000000
noncomputable def d := (Nat.factorial 100)^10

theorem arrange_numbers :
  a < d ∧ d < c ∧ c < b := 
sorry

end arrange_numbers_l141_141335


namespace lines_intersect_l141_141682

variables {s v : ℝ}

def line1 (s : ℝ) : ℝ × ℝ :=
  (3 - 2 * s, 4 + 3 * s)

def line2 (v : ℝ) : ℝ × ℝ :=
  (1 - 3 * v, 5 + 2 * v)

theorem lines_intersect :
  ∃ s v : ℝ, line1 s = line2 v ∧ line1 s = (25 / 13, 73 / 13) :=
by
  sorry

end lines_intersect_l141_141682


namespace edric_hourly_rate_l141_141712

-- Define conditions
def edric_monthly_salary : ℝ := 576
def edric_weekly_hours : ℝ := 8 * 6 -- 48 hours
def average_weeks_per_month : ℝ := 4.33
def edric_monthly_hours : ℝ := edric_weekly_hours * average_weeks_per_month -- Approx 207.84 hours

-- Define the expected result
def edric_expected_hourly_rate : ℝ := 2.77

-- Proof statement
theorem edric_hourly_rate :
  edric_monthly_salary / edric_monthly_hours = edric_expected_hourly_rate :=
by
  sorry

end edric_hourly_rate_l141_141712


namespace mean_of_three_l141_141446

variable (p q r : ℚ)

theorem mean_of_three (h1 : (p + q) / 2 = 13)
                      (h2 : (q + r) / 2 = 16)
                      (h3 : (r + p) / 2 = 7) :
                      (p + q + r) / 3 = 12 :=
by
  sorry

end mean_of_three_l141_141446


namespace determine_x_l141_141369

variable {m x : ℝ}

theorem determine_x (h₁ : m > 25)
    (h₂ : ((m / 100) * m = (m - 20) / 100 * (m + x))) : 
    x = 20 * m / (m - 20) := 
sorry

end determine_x_l141_141369


namespace tallest_building_height_l141_141601

theorem tallest_building_height :
  ∃ H : ℝ, H + (1/2) * H + (1/4) * H + (1/20) * H = 180 ∧ H = 100 := by
  sorry

end tallest_building_height_l141_141601


namespace sum_of_squares_l141_141034

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + a * c + b * c = 131) (h2 : a + b + c = 22) : a^2 + b^2 + c^2 = 222 :=
by
  sorry

end sum_of_squares_l141_141034


namespace solve_quadratic_equation_l141_141165

theorem solve_quadratic_equation :
  ∀ (x : ℝ), ((x - 2) * (x + 3) = 0) ↔ (x = 2 ∨ x = -3) :=
by
  intro x
  sorry

end solve_quadratic_equation_l141_141165


namespace part1_part2_l141_141223

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * (Real.sin x) ^ 2

theorem part1 : f (Real.pi / 6) = 1 / 2 :=
by
  sorry

theorem part2 : 
  ∃ (M m : ℝ), 
  (∀ x ∈ set.Icc (-Real.pi / 4) (Real.pi / 6), f x ≤ M) ∧
  (∀ x ∈ set.Icc (-Real.pi / 4) (Real.pi / 6), f x ≥ m) ∧
  (∀ x ∈ set.Icc (-Real.pi / 4) (Real.pi / 6), M = f x → x = 0) ∧
  (∀ x ∈ set.Icc (-Real.pi / 4) (Real.pi / 6), m = f x → x = -Real.pi / 4) :=
by
  sorry

end part1_part2_l141_141223


namespace lights_ratio_l141_141409

theorem lights_ratio (M S L : ℕ) (h1 : M = 12) (h2 : S = M + 10) (h3 : 118 = (S * 1) + (M * 2) + (L * 3)) :
  L = 24 ∧ L / M = 2 :=
by
  sorry

end lights_ratio_l141_141409


namespace percent_value_in_quarters_l141_141665

theorem percent_value_in_quarters (dimes quarters : ℕ) (dime_value quarter_value : ℕ) (dime_count quarter_count : ℕ) :
  dimes = 50 →
  quarters = 20 →
  dime_value = 10 →
  quarter_value = 25 →
  dime_count = dimes * dime_value →
  quarter_count = quarters * quarter_value →
  (quarter_count : ℚ) / (dime_count + quarter_count) * 100 = 50 :=
by
  intros
  sorry

end percent_value_in_quarters_l141_141665


namespace simplify_expr1_simplify_expr2_l141_141796

-- Proof problem for the first expression
theorem simplify_expr1 (x y : ℤ) : (2 - x + 3 * y + 8 * x - 5 * y - 6) = (7 * x - 2 * y -4) := 
by 
   -- Proving steps would go here
   sorry

-- Proof problem for the second expression
theorem simplify_expr2 (a b : ℤ) : (15 * a^2 * b - 12 * a * b^2 + 12 - 4 * a^2 * b - 18 + 8 * a * b^2) = (11 * a^2 * b - 4 * a * b^2 - 6) := 
by 
   -- Proving steps would go here
   sorry

end simplify_expr1_simplify_expr2_l141_141796


namespace arithmetic_mean_of_fractions_l141_141627

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l141_141627


namespace smallest_value_of_a_squared_plus_b_l141_141263

theorem smallest_value_of_a_squared_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → a * x^3 + b * y^2 ≥ x * y - 1) :
    a^2 + b = 2 / (3 * Real.sqrt 3) :=
by
  sorry

end smallest_value_of_a_squared_plus_b_l141_141263


namespace arithmetic_mean_of_fractions_l141_141617

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l141_141617


namespace katya_solves_enough_l141_141259

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l141_141259


namespace arctan_sum_is_pi_over_4_l141_141003

open Real

theorem arctan_sum_is_pi_over_4 (a b c : ℝ) (h1 : b = c) (h2 : c / (a + b) + a / (b + c) = 1) :
  arctan (c / (a + b)) + arctan (a / (b + c)) = π / 4 :=
by 
  sorry

end arctan_sum_is_pi_over_4_l141_141003


namespace tomatoes_for_5_liters_l141_141605

theorem tomatoes_for_5_liters (kg_per_3_liters : ℝ) (liters_needed : ℝ) :
  (kg_per_3_liters = 69 / 3) → (liters_needed = 5) → (kg_per_3_liters * liters_needed = 115) := 
by
  intros h1 h2
  sorry

end tomatoes_for_5_liters_l141_141605


namespace find_change_l141_141250

def initial_amount : ℝ := 1.80
def cost_of_candy_bar : ℝ := 0.45
def change : ℝ := 1.35

theorem find_change : initial_amount - cost_of_candy_bar = change :=
by sorry

end find_change_l141_141250


namespace not_axiom_l141_141971

theorem not_axiom (P Q R S : Prop)
  (B : P -> Q -> R -> S)
  (C : P -> Q)
  (D : P -> R)
  : ¬ (P -> Q -> S) :=
sorry

end not_axiom_l141_141971


namespace combined_time_in_pool_l141_141339

theorem combined_time_in_pool : 
    ∀ (Jerry_time Elaine_time George_time Kramer_time : ℕ), 
    Jerry_time = 3 →
    Elaine_time = 2 * Jerry_time →
    George_time = Elaine_time / 3 →
    Kramer_time = 0 →
    Jerry_time + Elaine_time + George_time + Kramer_time = 11 :=
by 
  intros Jerry_time Elaine_time George_time Kramer_time hJerry hElaine hGeorge hKramer
  sorry

end combined_time_in_pool_l141_141339


namespace clark_paid_correct_amount_l141_141853

-- Definitions based on the conditions
def cost_per_part : ℕ := 80
def number_of_parts : ℕ := 7
def total_discount : ℕ := 121

-- Given conditions
def total_cost_without_discount : ℕ := cost_per_part * number_of_parts
def expected_total_cost_after_discount : ℕ := 439

-- Theorem to prove the amount Clark paid after the discount is correct
theorem clark_paid_correct_amount : total_cost_without_discount - total_discount = expected_total_cost_after_discount := by
  sorry

end clark_paid_correct_amount_l141_141853


namespace polynomial_coeff_sums_l141_141293

theorem polynomial_coeff_sums (g h : ℤ) (d : ℤ) :
  (7 * d^2 - 3 * d + g) * (3 * d^2 + h * d - 8) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d - 16 →
  g + h = -3 :=
by
  sorry

end polynomial_coeff_sums_l141_141293


namespace arithmetic_mean_of_fractions_l141_141615

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l141_141615


namespace katya_minimum_problems_l141_141251

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l141_141251


namespace cost_in_chinese_yuan_l141_141782

theorem cost_in_chinese_yuan
  (usd_to_nad : ℝ := 8)
  (usd_to_cny : ℝ := 5)
  (sculpture_cost_nad : ℝ := 160) :
  sculpture_cost_nad / usd_to_nad * usd_to_cny = 100 := 
by
  sorry

end cost_in_chinese_yuan_l141_141782


namespace exists_points_irrational_distance_rational_area_l141_141212

noncomputable def points := λ (n : ℕ), fin n → (ℝ × ℝ)

theorem exists_points_irrational_distance_rational_area (n : ℕ) (hn : 3 ≤ n) :
  ∃ (P : points n), 
    (∀ i j, i ≠ j → irrational (dist (P i) (P j))) ∧ 
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ∃ (a : ℚ), area (P i) (P j) (P k) = a) :=
sorry

end exists_points_irrational_distance_rational_area_l141_141212


namespace incorrect_regression_statement_incorrect_statement_proof_l141_141053

-- Define the regression equation and the statement about y and x
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- Proof statement: given the regression equation, show that when x increases by one unit, y decreases by 5 units on average
theorem incorrect_regression_statement : 
  (regression_equation (x + 1) = regression_equation x + (-5)) :=
by sorry

-- Proof statement: prove that the statement "when the variable x increases by one unit, y increases by 5 units on average" is incorrect
theorem incorrect_statement_proof :
  ¬ (regression_equation (x + 1) = regression_equation x + 5) :=
by sorry  

end incorrect_regression_statement_incorrect_statement_proof_l141_141053


namespace min_value_inequality_l141_141013

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end min_value_inequality_l141_141013


namespace compare_magnitudes_l141_141906

noncomputable
def f (x : ℝ) : ℝ := Real.cos (Real.cos x)

noncomputable
def g (x : ℝ) : ℝ := Real.sin (Real.sin x)

theorem compare_magnitudes : ∀ x : ℝ, f x > g x :=
by
  sorry

end compare_magnitudes_l141_141906


namespace multiples_33_between_1_and_300_l141_141456

theorem multiples_33_between_1_and_300 : ∃ (x : ℕ), (∀ n : ℕ, n ≤ 300 → n % x = 0 → n / x ≤ 33) ∧ x = 9 :=
by
  sorry

end multiples_33_between_1_and_300_l141_141456


namespace linda_original_amount_l141_141548

theorem linda_original_amount (L L2 : ℕ) 
  (h1 : L = 20) 
  (h2 : L - 5 = L2) : 
  L2 + 5 = 15 := 
sorry

end linda_original_amount_l141_141548


namespace find_two_digit_integers_l141_141382

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem find_two_digit_integers
    (a b : ℕ) :
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 
    (a = b + 12 ∨ b = a + 12) ∧
    (a / 10 = b / 10 ∨ a % 10 = b % 10) ∧
    (sum_of_digits a = sum_of_digits b + 3 ∨ sum_of_digits b = sum_of_digits a + 3) :=
sorry

end find_two_digit_integers_l141_141382


namespace sqrt_range_l141_141552

theorem sqrt_range (x : ℝ) (h : 5 - x ≥ 0) : x ≤ 5 :=
sorry

end sqrt_range_l141_141552


namespace only_other_list_with_same_product_l141_141524

-- Assigning values to letters
def letter_value (ch : Char) : ℕ :=
  match ch with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7 | 'H' => 8
  | 'I' => 9 | 'J' => 10| 'K' => 11| 'L' => 12| 'M' => 13| 'N' => 14| 'O' => 15| 'P' => 16
  | 'Q' => 17| 'R' => 18| 'S' => 19| 'T' => 20| 'U' => 21| 'V' => 22| 'W' => 23| 'X' => 24
  | 'Y' => 25| 'Z' => 26| _ => 0

-- Define the product function for a list of 4 letters
def product_of_list (lst : List Char) : ℕ :=
  lst.map letter_value |> List.prod

-- Define the specific lists
def BDFH : List Char := ['B', 'D', 'F', 'H']
def BCDH : List Char := ['B', 'C', 'D', 'H']

-- The main statement to prove
theorem only_other_list_with_same_product : 
  product_of_list BCDH = product_of_list BDFH :=
by
  -- Sorry is a placeholder for the proof
  sorry

end only_other_list_with_same_product_l141_141524


namespace joan_kittens_count_correct_l141_141246

def joan_initial_kittens : Nat := 8
def kittens_from_friends : Nat := 2
def joan_total_kittens (initial: Nat) (added: Nat) : Nat := initial + added

theorem joan_kittens_count_correct : joan_total_kittens joan_initial_kittens kittens_from_friends = 10 := 
by
  sorry

end joan_kittens_count_correct_l141_141246


namespace cubic_feet_per_bag_l141_141955

-- Definitions
def length_bed := 8 -- in feet
def width_bed := 4 -- in feet
def height_bed := 1 -- in feet
def number_of_beds := 2
def number_of_bags := 16

-- Theorem statement
theorem cubic_feet_per_bag : 
  (length_bed * width_bed * height_bed * number_of_beds) / number_of_bags = 4 :=
by
  sorry

end cubic_feet_per_bag_l141_141955


namespace evaluate_expression_l141_141659

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := 
  sorry

end evaluate_expression_l141_141659


namespace part1_part2_l141_141108

/- Define the function f(x) = |x-1| + |x-a| -/
def f (x a : ℝ) := abs (x - 1) + abs (x - a)

/- Part 1: Prove that if f(x) ≥ 2 implies the solution set {x | x ≤ 1/2 or x ≥ 5/2}, then a = 2 -/
theorem part1 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 → (x ≤ 1/2 ∨ x ≥ 5/2)) : a = 2 :=
  sorry

/- Part 2: Prove that for all x ∈ ℝ, f(x) + |x-1| ≥ 1 implies a ∈ [2, +∞) -/
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a + abs (x - 1) ≥ 1) : 2 ≤ a :=
  sorry

end part1_part2_l141_141108


namespace next_in_step_distance_l141_141414

theorem next_in_step_distance
  (jack_stride jill_stride : ℕ)
  (h1 : jack_stride = 64)
  (h2 : jill_stride = 56) :
  Nat.lcm jack_stride jill_stride = 448 := by
  sorry

end next_in_step_distance_l141_141414


namespace latus_rectum_of_parabola_l141_141529

theorem latus_rectum_of_parabola :
  (∃ p : ℝ, ∀ x y : ℝ, y = - (1 / 6) * x^2 → y = p ∧ p = 3 / 2) :=
sorry

end latus_rectum_of_parabola_l141_141529


namespace lcm_12_18_l141_141979

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l141_141979


namespace transportation_degrees_l141_141942

theorem transportation_degrees
  (salaries : ℕ) (r_and_d : ℕ) (utilities : ℕ) (equipment : ℕ) (supplies : ℕ) (total_degrees : ℕ)
  (h_salaries : salaries = 60)
  (h_r_and_d : r_and_d = 9)
  (h_utilities : utilities = 5)
  (h_equipment : equipment = 4)
  (h_supplies : supplies = 2)
  (h_total_degrees : total_degrees = 360) :
  (total_degrees * (100 - (salaries + r_and_d + utilities + equipment + supplies)) / 100 = 72) :=
by {
  sorry
}

end transportation_degrees_l141_141942


namespace factor_expression_l141_141526

theorem factor_expression (x : ℝ) : 100 * x ^ 23 + 225 * x ^ 46 = 25 * x ^ 23 * (4 + 9 * x ^ 23) :=
by
  -- Proof steps will go here
  sorry

end factor_expression_l141_141526


namespace find_weight_of_A_l141_141027

theorem find_weight_of_A 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 5) 
  (h4 : (B + C + D + E) / 4 = 79) 
  : A = 77 := 
sorry

end find_weight_of_A_l141_141027


namespace slices_with_both_toppings_l141_141320

theorem slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices : ℕ)
    (all_have_topping : total_slices = 24)
    (pepperoni_cond: pepperoni_slices = 14)
    (mushroom_cond: mushroom_slices = 16)
    (at_least_one_topping : total_slices = pepperoni_slices + mushroom_slices - slices_with_both):
    slices_with_both = 6 := by
  sorry

end slices_with_both_toppings_l141_141320


namespace robin_bobin_can_meet_prescription_l141_141404

def large_gr_pill : ℝ := 11
def medium_gr_pill : ℝ := -1.1
def small_gr_pill : ℝ := -0.11
def prescribed_gr : ℝ := 20.13

theorem robin_bobin_can_meet_prescription :
  ∃ (large : ℕ) (medium : ℕ) (small : ℕ), large ≥ 1 ∧ medium ≥ 1 ∧ small ≥ 1 ∧
  large_gr_pill * large + medium_gr_pill * medium + small_gr_pill * small = prescribed_gr :=
sorry

end robin_bobin_can_meet_prescription_l141_141404


namespace simplify_and_evaluate_l141_141149

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3)) - (x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  sorry

end simplify_and_evaluate_l141_141149


namespace arithmetic_mean_of_fractions_l141_141634

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l141_141634


namespace simplify_and_evaluate_expr_l141_141151

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l141_141151


namespace multiply_fractions_l141_141343

theorem multiply_fractions :
  (2/3) * (4/7) * (9/11) * (5/8) = 15/77 :=
by
  -- It is just a statement, no need for the proof steps here
  sorry

end multiply_fractions_l141_141343


namespace lcm_12_18_l141_141974

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l141_141974


namespace reflect_y_axis_l141_141241

theorem reflect_y_axis (x y z : ℝ) : (x, y, z) = (1, -2, 3) → (-x, y, -z) = (-1, -2, -3) :=
by
  intros
  sorry

end reflect_y_axis_l141_141241


namespace smallest_add_to_2002_l141_141845

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def next_palindrome_after (n : ℕ) : ℕ :=
  -- a placeholder function for the next palindrome calculation
  -- implementation logic is skipped
  2112

def smallest_add_to_palindrome (n target : ℕ) : ℕ :=
  target - n

theorem smallest_add_to_2002 :
  let target := next_palindrome_after 2002
  ∃ k, is_palindrome (2002 + k) ∧ (2002 < 2002 + k) ∧ target = 2002 + k ∧ k = 110 := 
by
  use 110
  sorry

end smallest_add_to_2002_l141_141845


namespace problem_l141_141970

theorem problem
    (a b c d : ℕ)
    (h1 : a = b + 7)
    (h2 : b = c + 15)
    (h3 : c = d + 25)
    (h4 : d = 90) :
  a = 137 := by
  sorry

end problem_l141_141970


namespace multiplicative_inverse_l141_141440

def A : ℕ := 123456
def B : ℕ := 162738
def N : ℕ := 503339
def modulo : ℕ := 1000000

theorem multiplicative_inverse :
  (A * B * N) % modulo = 1 :=
by
  -- placeholder for proof
  sorry

end multiplicative_inverse_l141_141440


namespace celina_total_expenditure_l141_141858

theorem celina_total_expenditure :
  let hoodie_cost := 80
  let flashlight_cost := 0.20 * hoodie_cost
  let boots_original_cost := 110
  let boots_discounted_cost := boots_original_cost - 0.10 * boots_original_cost
  let total_cost := hoodie_cost + flashlight_cost + boots_discounted_cost
  total_cost = 195 :=
by
  -- Definitions
  let hoodie_cost := 80
  let flashlight_cost := 0.20 * hoodie_cost
  let boots_original_cost := 110
  let boots_discounted_cost := boots_original_cost - 0.10 * boots_original_cost
  let total_cost := hoodie_cost + flashlight_cost + boots_discounted_cost
  -- Assertion
  have h : total_cost = 195 := sorry
  exact h

end celina_total_expenditure_l141_141858


namespace system_of_equations_solution_l141_141209

/-- Integer solutions to the system of equations:
    \begin{cases}
        xz - 2yt = 3 \\
        xt + yz = 1
    \end{cases}
-/
theorem system_of_equations_solution :
  ∃ (x y z t : ℤ), 
    x * z - 2 * y * t = 3 ∧ 
    x * t + y * z = 1 ∧
    ((x = 1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
     (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
     (x = 3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
     (x = -3 ∧ y = -1 ∧ z = -1 ∧ t = 0)) :=
by {
  sorry
}

end system_of_equations_solution_l141_141209


namespace n_times_s_eq_neg_two_l141_141135

-- Define existence of function g
variable (g : ℝ → ℝ)

-- The given condition for the function g: ℝ -> ℝ
axiom g_cond : ∀ x y : ℝ, g (g x - y) = 2 * g x + g (g y - g (-x)) + y

-- Define n and s as per the conditions mentioned in the problem
def n : ℕ := 1 -- Based on the solution, there's only one possible value
def s : ℝ := -2 -- Sum of all possible values

-- The main statement to prove
theorem n_times_s_eq_neg_two : (n * s) = -2 := by
  sorry

end n_times_s_eq_neg_two_l141_141135


namespace solve_quadratic_equation_l141_141166

theorem solve_quadratic_equation :
  ∀ (x : ℝ), ((x - 2) * (x + 3) = 0) ↔ (x = 2 ∨ x = -3) :=
by
  intro x
  sorry

end solve_quadratic_equation_l141_141166


namespace icing_time_is_30_l141_141859

def num_batches : Nat := 4
def baking_time_per_batch : Nat := 20
def total_time : Nat := 200

def baking_time_total : Nat := num_batches * baking_time_per_batch
def icing_time_total : Nat := total_time - baking_time_total
def icing_time_per_batch : Nat := icing_time_total / num_batches

theorem icing_time_is_30 :
  icing_time_per_batch = 30 := by
  sorry

end icing_time_is_30_l141_141859


namespace ax_product_zero_l141_141287

theorem ax_product_zero 
  {a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ} 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) *
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) *
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := 
sorry

end ax_product_zero_l141_141287


namespace Julia_total_payment_l141_141419

namespace CarRental

def daily_rate : ℝ := 30
def mileage_rate : ℝ := 0.25
def num_days : ℝ := 3
def num_miles : ℝ := 500

def daily_cost : ℝ := daily_rate * num_days
def mileage_cost : ℝ := mileage_rate * num_miles
def total_cost : ℝ := daily_cost + mileage_cost

theorem Julia_total_payment : total_cost = 215 := by
  sorry

end CarRental

end Julia_total_payment_l141_141419


namespace carrots_picked_next_day_l141_141516

-- Definitions based on conditions
def initial_carrots : Nat := 48
def carrots_thrown_away : Nat := 45
def total_carrots_next_day : Nat := 45

-- The proof problem statement
theorem carrots_picked_next_day : 
  (initial_carrots - carrots_thrown_away + x = total_carrots_next_day) → (x = 42) :=
by 
  sorry

end carrots_picked_next_day_l141_141516


namespace arithmetic_sequence_sum_l141_141455

/-- Let {a_n} be an arithmetic sequence and S_n the sum of its first n terms.
   Given a_1 - a_5 - a_10 - a_15 + a_19 = 2, prove that S_19 = -38. --/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
  S 19 = -38 := 
sorry

end arithmetic_sequence_sum_l141_141455


namespace x_intercepts_of_parabola_l141_141391

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end x_intercepts_of_parabola_l141_141391


namespace number_is_580_l141_141675

noncomputable def find_number (x : ℝ) : Prop :=
  0.20 * x = 116

theorem number_is_580 (x : ℝ) (h : find_number x) : x = 580 :=
  by sorry

end number_is_580_l141_141675


namespace more_than_half_remains_l141_141323

def cubic_block := { n : ℕ // n > 0 }

noncomputable def total_cubes (b : cubic_block) : ℕ := b.val ^ 3

noncomputable def outer_layer_cubes (b : cubic_block) : ℕ := 6 * (b.val ^ 2) - 12 * b.val + 8

noncomputable def remaining_cubes (b : cubic_block) : ℕ := total_cubes b - outer_layer_cubes b

theorem more_than_half_remains (b : cubic_block) (h : b.val = 10) : remaining_cubes b > total_cubes b / 2 :=
by
  sorry

end more_than_half_remains_l141_141323


namespace smallest_fraction_numerator_l141_141957

theorem smallest_fraction_numerator (a b : ℕ) (h1 : 10 ≤ a) (h2 : a ≤ 99) (h3 : 10 ≤ b) (h4 : b ≤ 99) (h5 : 9 * a > 4 * b) (smallest : ∀ c d, 10 ≤ c → c ≤ 99 → 10 ≤ d → d ≤ 99 → 9 * c > 4 * d → (a * d ≤ b * c) → a * d ≤ 41 * 92) :
  a = 41 :=
by
  sorry

end smallest_fraction_numerator_l141_141957


namespace sum_of_roots_eq_l141_141701

noncomputable def polynomial_sum_of_roots : ℚ :=
  let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  let roots := [(-4) / 3, 6]
  roots.sum

theorem sum_of_roots_eq :
  let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
  ∑ root in [(-4) / 3, 6], root = 14 / 3 :=
by
  sorry

end sum_of_roots_eq_l141_141701


namespace tank_capacity_correctness_l141_141890

noncomputable def tankCapacity : ℝ := 77.65

theorem tank_capacity_correctness (T : ℝ) 
  (h_initial: T * (5 / 8) + 11 = T * (23 / 30)) : 
  T = tankCapacity := 
by
  sorry

end tank_capacity_correctness_l141_141890


namespace percentage_of_360_equals_115_2_l141_141045

theorem percentage_of_360_equals_115_2 (p : ℝ) (h : (p / 100) * 360 = 115.2) : p = 32 :=
by
  sorry

end percentage_of_360_equals_115_2_l141_141045


namespace largest_number_l141_141663

noncomputable def a : ℝ := 8.12331
noncomputable def b : ℝ := 8.123 + 3 / 10000 * ∑' n, 1 / (10 : ℝ)^n
noncomputable def c : ℝ := 8.12 + 331 / 100000 * ∑' n, 1 / (1000 : ℝ)^n
noncomputable def d : ℝ := 8.1 + 2331 / 1000000 * ∑' n, 1 / (10000 : ℝ)^n
noncomputable def e : ℝ := 8 + 12331 / 100000 * ∑' n, 1 / (10000 : ℝ)^n

theorem largest_number : (b > a) ∧ (b > c) ∧ (b > d) ∧ (b > e) := by
  sorry

end largest_number_l141_141663


namespace second_marble_orange_probability_l141_141511

noncomputable def prob_second_orange (BagX_red : ℕ) (BagX_green : ℕ)
                                     (BagY_orange : ℕ) (BagY_purple : ℕ)
                                     (BagZ_orange : ℕ) (BagZ_purple : ℕ) : ℚ :=
  let P_red_from_X := BagX_red / (BagX_red + BagX_green : ℚ)
  let P_green_from_X := BagX_green / (BagX_red + BagX_green : ℚ)
  let P_orange_from_Y := BagY_orange / (BagY_orange + BagY_purple : ℚ)
  let P_orange_from_Z := BagZ_orange / (BagZ_orange + BagZ_purple : ℚ)
  (P_red_from_X * P_orange_from_Y) + (P_green_from_X * P_orange_from_Z)

theorem second_marble_orange_probability : 
  prob_second_orange 5 3 7 5 4 6 = 247 / 480 := 
by sorry

end second_marble_orange_probability_l141_141511


namespace smallest_fraction_numerator_l141_141958

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (4 * b < 9 * a) ∧ 
  (∀ (a' b' : ℕ), a' ≥ 10 ∧ a' ≤ 99 ∧ b' ≥ 10 ∧ b' ≤ 99 ∧ (4 * b' < 9 * a') → b * a' ≥ a * b') ∧ a = 41 :=
sorry

end smallest_fraction_numerator_l141_141958


namespace combined_total_time_l141_141340

def jerry_time : ℕ := 3
def elaine_time : ℕ := 2 * jerry_time
def george_time : ℕ := elaine_time / 3
def kramer_time : ℕ := 0
def total_time : ℕ := jerry_time + elaine_time + george_time + kramer_time

theorem combined_total_time : total_time = 11 := by
  unfold total_time jerry_time elaine_time george_time kramer_time
  rfl

end combined_total_time_l141_141340


namespace fixed_monthly_fee_l141_141515

theorem fixed_monthly_fee (x y z : ℝ) 
  (h1 : x + y = 18.50) 
  (h2 : x + y + 3 * z = 23.45) : 
  x = 7.42 := 
by 
  sorry

end fixed_monthly_fee_l141_141515


namespace find_children_and_coins_l141_141351

def condition_for_child (k m remaining_coins : ℕ) : Prop :=
  ∃ (received_coins : ℕ), (received_coins = k + remaining_coins / 7 ∧ received_coins * 7 = 7 * k + remaining_coins)

def valid_distribution (n m : ℕ) : Prop :=
  ∀ k (hk : 1 ≤ k ∧ k ≤ n),
  ∃ remaining_coins,
    condition_for_child k m remaining_coins

theorem find_children_and_coins :
  ∃ n m, valid_distribution n m ∧ n = 6 ∧ m = 36 :=
sorry

end find_children_and_coins_l141_141351


namespace number_of_people_per_van_l141_141863

theorem number_of_people_per_van (num_students : ℕ) (num_adults : ℕ) (num_vans : ℕ) (total_people : ℕ) (people_per_van : ℕ) :
  num_students = 40 →
  num_adults = 14 →
  num_vans = 6 →
  total_people = num_students + num_adults →
  people_per_van = total_people / num_vans →
  people_per_van = 9 :=
by
  intros h_students h_adults h_vans h_total h_div
  sorry

end number_of_people_per_van_l141_141863


namespace simplify_and_evaluate_expr_l141_141150

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l141_141150


namespace find_original_number_l141_141825

noncomputable def original_number (x : ℝ) : Prop :=
  1000 * x = 3 / x

theorem find_original_number (x : ℝ) (h : original_number x) : x = (Real.sqrt 30) / 100 :=
sorry

end find_original_number_l141_141825


namespace find_original_number_l141_141824

noncomputable def original_number (x : ℝ) : Prop :=
  1000 * x = 3 / x

theorem find_original_number (x : ℝ) (h : original_number x) : x = (Real.sqrt 30) / 100 :=
sorry

end find_original_number_l141_141824


namespace find_parabola_vertex_l141_141985

-- Define the parabola with specific roots.
def parabola (x : ℝ) : ℝ := -x^2 + 2 * x + 24

-- Define the vertex of the parabola.
def vertex : ℝ × ℝ := (1, 25)

-- Prove that the vertex of the parabola is indeed at (1, 25).
theorem find_parabola_vertex : vertex = (1, 25) :=
  sorry

end find_parabola_vertex_l141_141985


namespace area_ratio_of_squares_l141_141022

theorem area_ratio_of_squares (hA : ∃ sA : ℕ, 4 * sA = 16)
                             (hB : ∃ sB : ℕ, 4 * sB = 20)
                             (hC : ∃ sC : ℕ, 4 * sC = 40) :
  (∃ aB aC : ℕ, aB = sB * sB ∧ aC = sC * sC ∧ aB * 4 = aC) := by
  sorry

end area_ratio_of_squares_l141_141022


namespace crayons_count_l141_141024

def crayons_per_box : ℕ := 8
def number_of_boxes : ℕ := 10
def total_crayons : ℕ := crayons_per_box * number_of_boxes

theorem crayons_count : total_crayons = 80 := by
  sorry

end crayons_count_l141_141024


namespace arithmetic_mean_of_fractions_l141_141620

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l141_141620


namespace minutes_before_4_angle_same_as_4_l141_141114

def hour_hand_angle_at_4 := 120
def minute_hand_angle_at_4 := 0
def minute_hand_angle_per_minute := 6
def hour_hand_angle_per_minute := 0.5

theorem minutes_before_4_angle_same_as_4 :
  ∃ m : ℚ, abs (hour_hand_angle_at_4 - 5.5 * m) = hour_hand_angle_at_4 ∧ 
           (60 - m) = 21 + 9 / 11 := by
  sorry

end minutes_before_4_angle_same_as_4_l141_141114


namespace square_perimeter_l141_141588

-- We define a structure for a square with an area as a condition.
structure Square (s : ℝ) :=
(area_eq : s ^ 2 = 400)

-- The theorem states that given the area of the square is 400 square meters,
-- the perimeter of the square is 80 meters.
theorem square_perimeter (s : ℝ) (sq : Square s) : 4 * s = 80 :=
by
  -- proof omitted
  sorry

end square_perimeter_l141_141588


namespace equation_transformation_l141_141169

theorem equation_transformation (x y: ℝ) (h : 2 * x - 3 * y = 6) : 
  y = (2 * x - 6) / 3 := 
by
  sorry

end equation_transformation_l141_141169


namespace quoted_value_of_stock_l141_141696

theorem quoted_value_of_stock (D Y Q : ℝ) (h1 : D = 8) (h2 : Y = 10) (h3 : Y = (D / Q) * 100) : Q = 80 :=
by 
  -- Insert proof here
  sorry

end quoted_value_of_stock_l141_141696


namespace sqrt_sine_tan_domain_l141_141087

open Real

noncomputable def domain_sqrt_sine_tan : Set ℝ :=
  {x | ∃ (k : ℤ), (-π / 2 + 2 * k * π < x ∧ x < π / 2 + 2 * k * π) ∨ x = k * π}

theorem sqrt_sine_tan_domain (x : ℝ) :
  (sin x * tan x ≥ 0) ↔ x ∈ domain_sqrt_sine_tan :=
by
  sorry

end sqrt_sine_tan_domain_l141_141087


namespace difference_between_min_and_max_l141_141905

noncomputable 
def minValue (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ := 0

noncomputable
def maxValue (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ := 1.5

theorem difference_between_min_and_max (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  maxValue x z hx hz - minValue x z hx hz = 1.5 :=
by
  sorry

end difference_between_min_and_max_l141_141905


namespace i_pow_2016_eq_one_l141_141834
open Complex

theorem i_pow_2016_eq_one : (Complex.I ^ 2016) = 1 := by
  have h : Complex.I ^ 4 = 1 :=
    by rw [Complex.I_pow_four]
  exact sorry

end i_pow_2016_eq_one_l141_141834


namespace shop_profit_correct_l141_141417

def profit_per_tire_repair : ℕ := 20 - 5
def total_tire_repairs : ℕ := 300
def profit_per_complex_repair : ℕ := 300 - 50
def total_complex_repairs : ℕ := 2
def retail_profit : ℕ := 2000
def fixed_expenses : ℕ := 4000

theorem shop_profit_correct :
  profit_per_tire_repair * total_tire_repairs +
  profit_per_complex_repair * total_complex_repairs +
  retail_profit - fixed_expenses = 3000 :=
by
  sorry

end shop_profit_correct_l141_141417


namespace mutually_exclusive_not_complementary_l141_141791

open Probability -- Using probability namespace

-- Definitions for the events
def red_card (c : char) : Prop :=
  c = 'r' -- red card event

def yellow_card (c : char) : Prop :=
  c = 'y' -- yellow card event

def blue_card (c : char) : Prop :=
  c = 'b' -- blue card event

def person_A_gets_red (A : char) : Prop :=
  red_card A -- Person A gets red card 

def person_B_gets_red (B : char) : Prop :=
  red_card B -- Person B gets red card 

-- Lean statement for the equivalent proof problem
theorem mutually_exclusive_not_complementary :
  ∀ (A B C : char), 
  (red_card A ∨ yellow_card A ∨ blue_card A) ∧
  (red_card B ∨ yellow_card B ∨ blue_card B) ∧
  (red_card C ∨ yellow_card C ∨ blue_card C) ∧
  (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C) →
  (person_A_gets_red A → ¬ person_B_gets_red B) ∧ 
  (¬ person_A_gets_red A → (person_B_gets_red B ∨ ¬ person_B_gets_red B)) := 
by 
  sorry

end mutually_exclusive_not_complementary_l141_141791


namespace katya_needs_at_least_ten_l141_141258

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l141_141258


namespace vectors_not_coplanar_l141_141189

def vector_a : Fin 3 → ℤ := ![1, 5, 2]
def vector_b : Fin 3 → ℤ := ![-1, 1, -1]
def vector_c : Fin 3 → ℤ := ![1, 1, 1]

def scalar_triple_product (a b c : Fin 3 → ℤ) : ℤ :=
  a 0 * (b 1 * c 2 - b 2 * c 1) -
  a 1 * (b 0 * c 2 - b 2 * c 0) +
  a 2 * (b 0 * c 1 - b 1 * c 0)

theorem vectors_not_coplanar :
  scalar_triple_product vector_a vector_b vector_c ≠ 0 :=
by
  sorry

end vectors_not_coplanar_l141_141189


namespace arithmetic_mean_of_fractions_l141_141626

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l141_141626


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l141_141614

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l141_141614


namespace house_vs_trailer_payment_difference_l141_141416

-- Definitions based on given problem conditions
def house_cost : ℝ := 480000
def trailer_cost : ℝ := 120000
def loan_term_years : ℝ := 20
def months_in_year : ℝ := 12
def total_months : ℝ := loan_term_years * months_in_year

-- Monthly payment calculations
def house_monthly_payment : ℝ := house_cost / total_months
def trailer_monthly_payment : ℝ := trailer_cost / total_months

-- The theorem we need to prove
theorem house_vs_trailer_payment_difference :
  house_monthly_payment - trailer_monthly_payment = 1500 := 
by
  sorry

end house_vs_trailer_payment_difference_l141_141416


namespace roots_sum_of_quadratic_l141_141597

theorem roots_sum_of_quadratic :
  ∀ x1 x2 : ℝ, (Polynomial.eval x1 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              (Polynomial.eval x2 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              x1 + x2 = -2 :=
by
  intros x1 x2 h1 h2
  sorry

end roots_sum_of_quadratic_l141_141597


namespace market_value_of_stock_l141_141474

def face_value : ℝ := 100
def dividend_percentage : ℝ := 0.13
def yield : ℝ := 0.08

theorem market_value_of_stock : 
  (dividend_percentage * face_value / yield) * 100 = 162.50 :=
by
  sorry

end market_value_of_stock_l141_141474


namespace right_triangle_area_l141_141406

theorem right_triangle_area (hypotenuse : ℝ) (angle_ratio : ℝ) (a1 a2 : ℝ)
  (h1 : hypotenuse = 10)
  (h2 : angle_ratio = 5 / 4)
  (h3 : a1 = 50)
  (h4 : a2 = 40) :
  let A := hypotenuse * Real.sin (a2 * Real.pi / 180)
  let B := hypotenuse * Real.sin (a1 * Real.pi / 180)
  let area := 0.5 * A * B
  area = 24.63156 := by sorry

end right_triangle_area_l141_141406


namespace least_zorgs_to_drop_more_points_than_eating_l141_141898

theorem least_zorgs_to_drop_more_points_than_eating :
  ∃ (n : ℕ), (∀ m < n, m * (m + 1) / 2 ≤ 20 * m) ∧ n * (n + 1) / 2 > 20 * n :=
sorry

end least_zorgs_to_drop_more_points_than_eating_l141_141898


namespace johns_new_weekly_earnings_l141_141249

-- Define the original weekly earnings and the percentage increase as given conditions:
def original_weekly_earnings : ℕ := 60
def percentage_increase : ℕ := 50

-- Prove that John's new weekly earnings after the raise is 90 dollars:
theorem johns_new_weekly_earnings : original_weekly_earnings + (percentage_increase * original_weekly_earnings / 100) = 90 := by
sorry

end johns_new_weekly_earnings_l141_141249


namespace speed_in_still_water_l141_141060

-- Define the given conditions
def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

-- State the theorem to be proven
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 40 := by
  -- Proof omitted
  sorry

end speed_in_still_water_l141_141060


namespace equation_of_curve_E_equation_of_line_l_through_origin_intersecting_E_l141_141870

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

def is_ellipse (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1) ^ 2 + P.2 ^ 2) + Real.sqrt ((P.1 - F₂.1) ^ 2 + P.2 ^ 2) = 4

theorem equation_of_curve_E :
  ∀ P : ℝ × ℝ, is_ellipse P ↔ (P.1 ^ 2 / 4 + P.2 ^ 2 = 1) :=
sorry

def intersects_at_origin (C D : ℝ × ℝ) : Prop :=
  C.1 * D.1 + C.2 * D.2 = 0

theorem equation_of_line_l_through_origin_intersecting_E :
  ∀ (l : ℝ → ℝ) (C D : ℝ × ℝ),
    (l 0 = -2) →
    (∀ P : ℝ × ℝ, is_ellipse P ↔ (P.1, P.2) = (C.1, l C.1) ∨ (P.1, P.2) = (D.1, l D.1)) →
    intersects_at_origin C D →
    (∀ x, l x = 2 * x - 2) ∨ (∀ x, l x = -2 * x - 2) :=
sorry

end equation_of_curve_E_equation_of_line_l_through_origin_intersecting_E_l141_141870


namespace compare_abc_l141_141992

noncomputable def a : ℝ := 2 + (1 / 5) * Real.log 2
noncomputable def b : ℝ := 1 + Real.exp (0.2 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.1 * Real.log 2)

theorem compare_abc : a < c ∧ c < b := by
  sorry

end compare_abc_l141_141992


namespace combined_total_time_l141_141341

def jerry_time : ℕ := 3
def elaine_time : ℕ := 2 * jerry_time
def george_time : ℕ := elaine_time / 3
def kramer_time : ℕ := 0
def total_time : ℕ := jerry_time + elaine_time + george_time + kramer_time

theorem combined_total_time : total_time = 11 := by
  unfold total_time jerry_time elaine_time george_time kramer_time
  rfl

end combined_total_time_l141_141341


namespace least_number_of_cookies_l141_141141

theorem least_number_of_cookies :
  ∃ x : ℕ, x % 6 = 4 ∧ x % 5 = 3 ∧ x % 8 = 6 ∧ x % 9 = 7 ∧ x = 208 :=
by
  sorry

end least_number_of_cookies_l141_141141


namespace teal_bakery_revenue_l141_141408

theorem teal_bakery_revenue :
    let pumpkin_pies := 4
    let pumpkin_pie_slices := 8
    let pumpkin_slice_price := 5
    let custard_pies := 5
    let custard_pie_slices := 6
    let custard_slice_price := 6
    let total_pumpkin_slices := pumpkin_pies * pumpkin_pie_slices
    let total_custard_slices := custard_pies * custard_pie_slices
    let pumpkin_revenue := total_pumpkin_slices * pumpkin_slice_price
    let custard_revenue := total_custard_slices * custard_slice_price
    let total_revenue := pumpkin_revenue + custard_revenue
    total_revenue = 340 :=
by
  sorry

end teal_bakery_revenue_l141_141408


namespace find_z_when_y_is_6_l141_141278

variable {y z : ℚ}

/-- Condition: y^4 varies inversely with √[4]{z}. -/
def inverse_variation (k : ℚ) (y z : ℚ) : Prop :=
  y^4 * z^(1/4) = k

/-- Given constant k based on y = 3 and z = 16. -/
def k_value : ℚ := 162

theorem find_z_when_y_is_6
  (h_inv : inverse_variation k_value 3 16)
  (h_y : y = 6) :
  z = 1 / 4096 := 
sorry

end find_z_when_y_is_6_l141_141278


namespace total_earnings_correct_l141_141561

-- Define the earnings of Terrence
def TerrenceEarnings : ℕ := 30

-- Define the difference in earnings between Jermaine and Terrence
def JermaineEarningsDifference : ℕ := 5

-- Define the earnings of Jermaine
def JermaineEarnings : ℕ := TerrenceEarnings + JermaineEarningsDifference

-- Define the earnings of Emilee
def EmileeEarnings : ℕ := 25

-- Define the total earnings
def TotalEarnings : ℕ := TerrenceEarnings + JermaineEarnings + EmileeEarnings

theorem total_earnings_correct : TotalEarnings = 90 := by
  sorry

end total_earnings_correct_l141_141561


namespace volume_of_prism_l141_141685

theorem volume_of_prism (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 :=
by
  sorry

end volume_of_prism_l141_141685


namespace largest_divisor_n4_n2_l141_141711

theorem largest_divisor_n4_n2 (n : ℤ) : (6 : ℤ) ∣ (n^4 - n^2) :=
sorry

end largest_divisor_n4_n2_l141_141711


namespace incorrect_statement_is_A_l141_141345

theorem incorrect_statement_is_A :
  (∀ (w h : ℝ), w * (2 * h) ≠ 3 * (w * h)) ∧
  (∀ (s : ℝ), (2 * s) ^ 2 = 4 * (s ^ 2)) ∧
  (∀ (s : ℝ), (2 * s) ^ 3 = 8 * (s ^ 3)) ∧
  (∀ (w h : ℝ), (w / 2) * (3 * h) = (3 / 2) * (w * h)) ∧
  (∀ (l w : ℝ), (2 * l) * (3 * w) = 6 * (l * w)) →
  ∃ (incorrect_statement : String), incorrect_statement = "A" := 
by 
  sorry

end incorrect_statement_is_A_l141_141345


namespace katya_minimum_problems_l141_141252

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l141_141252


namespace train_pass_time_l141_141068

theorem train_pass_time (train_length : ℕ) (platform_length : ℕ) (speed : ℕ) (h1 : train_length = 50) (h2 : platform_length = 100) (h3 : speed = 15) : 
  (train_length + platform_length) / speed = 10 :=
by
  sorry

end train_pass_time_l141_141068


namespace f_fraction_neg_1987_1988_l141_141593

-- Define the function f and its properties
def f : ℚ → ℝ := sorry

axiom functional_eq (x y : ℚ) : f (x + y) = f x * f y - f (x * y) + 1
axiom not_equal_f : f 1988 ≠ f 1987

-- Prove the desired equality
theorem f_fraction_neg_1987_1988 : f (-1987 / 1988) = 1 / 1988 :=
by
  sorry

end f_fraction_neg_1987_1988_l141_141593


namespace fraction_equivalence_l141_141392

variable {m n p q : ℚ}

theorem fraction_equivalence
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5) :
  m / q = 1 :=
by {
  sorry
}

end fraction_equivalence_l141_141392


namespace insert_digits_identical_l141_141816

theorem insert_digits_identical (A B : List Nat) (hA : A.length = 2007) (hB : B.length = 2007)
  (hErase : ∃ (C : List Nat) (erase7A : List Nat → List Nat) (erase7B : List Nat → List Nat),
    (erase7A A = C) ∧ (erase7B B = C) ∧ (C.length = 2000)) :
  ∃ (D : List Nat) (insert7A : List Nat → List Nat) (insert7B : List Nat → List Nat),
    (insert7A A = D) ∧ (insert7B B = D) ∧ (D.length = 2014) := sorry

end insert_digits_identical_l141_141816


namespace equation1_solution_equation2_solution_l141_141156

theorem equation1_solution (x : ℝ) (h : 2 * (x - 1) = 2 - 5 * (x + 2)) : x = -6 / 7 :=
sorry

theorem equation2_solution (x : ℝ) (h : (5 * x + 1) / 2 - (6 * x + 2) / 4 = 1) : x = 1 :=
sorry

end equation1_solution_equation2_solution_l141_141156


namespace find_angle_B_l141_141760

variables {A B C a b c : ℝ} (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3)

theorem find_angle_B (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3) : B = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_B_l141_141760


namespace icing_cubes_count_l141_141680

theorem icing_cubes_count :
  let n := 5
  let total_cubes := n * n * n
  let side_faces := 4
  let cubes_per_edge_per_face := (n - 2) * (n - 1)
  let shared_edges := 4
  let icing_cubes := (side_faces * cubes_per_edge_per_face) / 2
  icing_cubes = 32 := sorry

end icing_cubes_count_l141_141680


namespace parabola_eq_exists_minimum_area_triangle_l141_141100

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y
noncomputable def circle (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 9 / 4
noncomputable def is_on_parabola (a b : ℝ) : Prop := a^2 = 2 * p * b
noncomputable def passes_through_origin (a b : ℝ) : Prop := (0 - a)^2 + (0 - b)^2 = 9 / 4
noncomputable def tangent_to_directrix (a b p : ℝ) : Prop := -- Define tangent condition here if necessary

theorem parabola_eq_exists
  (a b p : ℝ)
  (h1 : is_on_parabola a b)
  (h2 : passes_through_origin a b)
  (h3 : tangent_to_directrix a b p) :
  ∃ (x y : ℝ), parabola x y :=
begin
  sorry
end

noncomputable def minimum_area (a b p : ℝ) : ℝ := -- Define minimum area calculation here if necessary
noncomputable def line_eq (x : ℝ) : ℝ := -sqrt 6/3 * x + sqrt 2

theorem minimum_area_triangle
  (a b p : ℝ)
  (h1 : is_on_parabola a b)
  (h2 : passes_through_origin a b)
  (h3 : tangent_to_directrix a b p) :
  minimum_area a b p = 9 * sqrt 3 / 4 ∧ line_eq = (λ x, -√6/3 * x + √2) :=
begin
  sorry
end

end parabola_eq_exists_minimum_area_triangle_l141_141100


namespace ending_number_divisible_by_six_l141_141297

theorem ending_number_divisible_by_six (first_term : ℕ) (n : ℕ) (common_difference : ℕ) (sequence_length : ℕ) 
  (start : first_term = 12) 
  (diff : common_difference = 6)
  (num_terms : sequence_length = 11) :
  first_term + (sequence_length - 1) * common_difference = 72 := by
  sorry

end ending_number_divisible_by_six_l141_141297


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l141_141608

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l141_141608


namespace new_person_weight_l141_141946

theorem new_person_weight
  (initial_weight : ℝ)
  (average_increase : ℝ)
  (num_people : ℕ)
  (weight_replace : ℝ)
  (total_increase : ℝ)
  (W : ℝ)
  (h1 : num_people = 10)
  (h2 : average_increase = 3.5)
  (h3 : weight_replace = 65)
  (h4 : total_increase = num_people * average_increase)
  (h5 : total_increase = 35)
  (h6 : W = weight_replace + total_increase) :
  W = 100 := sorry

end new_person_weight_l141_141946


namespace consecutive_integers_equation_l141_141410

theorem consecutive_integers_equation
  (X Y : ℕ)
  (h_consecutive : Y = X + 1)
  (h_equation : 2 * X^2 + 4 * X + 5 * Y + 3 = (X + Y)^2 + 9 * (X + Y) + 4) :
  X + Y = 15 := by
  sorry

end consecutive_integers_equation_l141_141410


namespace shaded_percentage_six_by_six_grid_l141_141471

theorem shaded_percentage_six_by_six_grid (total_squares shaded_squares : ℕ)
    (h_total : total_squares = 36) (h_shaded : shaded_squares = 21) : 
    (shaded_squares.to_rat / total_squares.to_rat) * 100 = 58.33 := 
by
  sorry

end shaded_percentage_six_by_six_grid_l141_141471


namespace divide_fractions_l141_141544

theorem divide_fractions : (3 / 8) / (1 / 4) = 3 / 2 :=
by sorry

end divide_fractions_l141_141544


namespace gcf_252_96_l141_141305

theorem gcf_252_96 : Int.gcd 252 96 = 12 := by
  sorry

end gcf_252_96_l141_141305


namespace randy_gave_sally_l141_141575

-- Define the given conditions
def initial_amount_randy : ℕ := 3000
def smith_contribution : ℕ := 200
def amount_kept_by_randy : ℕ := 2000

-- The total amount Randy had after Smith's contribution
def total_amount_randy : ℕ := initial_amount_randy + smith_contribution

-- The amount of money Randy gave to Sally
def amount_given_to_sally : ℕ := total_amount_randy - amount_kept_by_randy

-- The theorem statement: Given the conditions, prove that Randy gave Sally $1,200
theorem randy_gave_sally : amount_given_to_sally = 1200 :=
by
  sorry

end randy_gave_sally_l141_141575


namespace identity_proof_l141_141522

theorem identity_proof :
  ∀ (x : ℝ), 
    x ≠ 2 →
    (x^2 + x + 1) ≠ 0 →
    ((x + 3) ^ 2 / ((x - 2) * (x^2 + x + 1)) = 
     (25 / 7) / (x - 2) + (-18 / 7 * x - 19 / 7) / (x^2 + x + 1)) :=
by
  intro x
  intros hx1 hx2
  -- proof goes here
  sorry

end identity_proof_l141_141522


namespace Cody_initial_money_l141_141517

-- Define the conditions
def initial_money (x : ℕ) : Prop :=
  x + 9 - 19 = 35

-- Define the theorem we need to prove
theorem Cody_initial_money : initial_money 45 :=
by
  -- Add a placeholder for the proof
  sorry

end Cody_initial_money_l141_141517


namespace product_of_solutions_eq_neg_nine_product_of_solutions_l141_141983

theorem product_of_solutions_eq_neg_nine :
  ∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions :
  (∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → (∃ (a b : ℝ), x = a ∨ x = b ∧ a * b = -9)) :=
by
  sorry

end product_of_solutions_eq_neg_nine_product_of_solutions_l141_141983


namespace side_length_of_S2_l141_141577

-- Define our context and the statements we need to work with
theorem side_length_of_S2
  (r s : ℕ)
  (h1 : 2 * r + s = 2450)
  (h2 : 2 * r + 3 * s = 4000) : 
  s = 775 :=
sorry

end side_length_of_S2_l141_141577


namespace part1_part2_l141_141856

-- Statement for Part 1
theorem part1 : 
  ∃ (a : Fin 8 → ℕ), 
    (∀ i : Fin 8, 1 ≤ a i ∧ a i ≤ 8) ∧ 
    (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
    (∀ i : Fin 8, (a i + a (i + 1) + a (i + 2)) > 11) := sorry

-- Statement for Part 2
theorem part2 : 
  ¬ ∃ (a : Fin 8 → ℕ), 
    (∀ i : Fin 8, 1 ≤ a i ∧ a i ≤ 8) ∧ 
    (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
    (∀ i : Fin 8, (a i + a (i + 1) + a (i + 2)) > 13) := sorry

end part1_part2_l141_141856


namespace range_f_range_a_l141_141375

def f (x : ℝ) := abs (x + 2) - abs (x - 1)

def g (a : ℝ) (s : ℝ) := (a * s^2 - 3 * s + 3) / s

theorem range_f :
  set.range f = set.Icc (-3 : ℝ) 3 := sorry

theorem range_a (a : ℝ) :
  ( ∀ (s : ℝ) (t : ℝ), s > 0 → g a s ≥ f t ) → a ≥ 3 := sorry

end range_f_range_a_l141_141375


namespace cinnamon_balls_required_l141_141006

theorem cinnamon_balls_required 
  (num_family_members : ℕ) 
  (cinnamon_balls_per_day : ℕ) 
  (num_days : ℕ) 
  (h_family : num_family_members = 5) 
  (h_balls_per_day : cinnamon_balls_per_day = 5) 
  (h_days : num_days = 10) : 
  num_family_members * cinnamon_balls_per_day * num_days = 50 := by
  sorry

end cinnamon_balls_required_l141_141006


namespace product_of_solutions_l141_141984

theorem product_of_solutions (x : ℝ) (h : |x| = 3 * (|x| - 2)) : (subs := (|x| == 3 ->  (x = 3)  ∨ (x = -3)): 
solution ( ∀ x:solution ∧ x₁ * x₂= 3 * (-3) )  : -9)   := 
sorry

end product_of_solutions_l141_141984


namespace chris_mixed_raisins_l141_141967

-- Conditions
variables (R C : ℝ)

-- 1. Chris mixed some pounds of raisins with 3 pounds of nuts.
-- 2. A pound of nuts costs 3 times as much as a pound of raisins.
-- 3. The total cost of the raisins was 0.25 of the total cost of the mixture.

-- Problem statement: Prove that R = 3 given the conditions
theorem chris_mixed_raisins :
  R * C = 0.25 * (R * C + 3 * 3 * C) → R = 3 :=
by
  sorry

end chris_mixed_raisins_l141_141967


namespace maximum_value_of_function_l141_141746

theorem maximum_value_of_function (a : ℕ) (ha : 0 < a) : 
  ∃ x : ℝ, x + Real.sqrt (13 - 2 * a * x) = 7 :=
by
  sorry

end maximum_value_of_function_l141_141746


namespace ratio_of_dogs_to_cats_l141_141412

theorem ratio_of_dogs_to_cats (D C : ℕ) (hC : C = 40) (h : D + 20 = 2 * C) :
  D / Nat.gcd D C = 3 ∧ C / Nat.gcd D C = 2 :=
by
  sorry

end ratio_of_dogs_to_cats_l141_141412


namespace parabola_directrix_correct_l141_141353

noncomputable def parabola_directrix : Prop :=
  let eqn := λ x : ℝ, -3 * x^2 + 6 * x - 5
  let directrix := -23/12
  ∀ x : ℝ, eqn x = y → y = directrix

theorem parabola_directrix_correct : parabola_directrix :=
begin
  sorry
end

end parabola_directrix_correct_l141_141353


namespace value_of_fraction_l141_141394

-- Lean 4 statement
theorem value_of_fraction (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by
  sorry

end value_of_fraction_l141_141394


namespace number_of_possible_a_values_l141_141029

-- Define the function f(x)
def f (a x : ℝ) := abs (x + 1) + abs (a * x + 1)

-- Define the condition for the minimum value
def minimum_value_of_f (a : ℝ) := ∃ x : ℝ, f a x = (3 / 2)

-- The proof problem statement
theorem number_of_possible_a_values : 
  (∃ (a1 a2 a3 a4 : ℝ),
    minimum_value_of_f a1 ∧
    minimum_value_of_f a2 ∧
    minimum_value_of_f a3 ∧
    minimum_value_of_f a4 ∧
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :=
sorry

end number_of_possible_a_values_l141_141029


namespace a_share_correct_l141_141667

-- Investment periods for each individual in months
def investment_a := 12
def investment_b := 6
def investment_c := 4
def investment_d := 9
def investment_e := 7
def investment_f := 5

-- Investment multiplier for each individual
def multiplier_b := 2
def multiplier_c := 3
def multiplier_d := 4
def multiplier_e := 5
def multiplier_f := 6

-- Total annual gain
def total_gain := 38400

-- Calculate individual shares
def share_a (x : ℝ) := x * investment_a
def share_b (x : ℝ) := multiplier_b * x * investment_b
def share_c (x : ℝ) := multiplier_c * x * investment_c
def share_d (x : ℝ) := multiplier_d * x * investment_d
def share_e (x : ℝ) := multiplier_e * x * investment_e
def share_f (x : ℝ) := multiplier_f * x * investment_f

-- Calculate total investment
def total_investment (x : ℝ) :=
  share_a x + share_b x + share_c x + share_d x + share_e x + share_f x

-- Prove that a's share of the annual gain is Rs. 3360
theorem a_share_correct : 
  ∃ x : ℝ, (12 * x / total_investment x) * total_gain = 3360 := 
sorry

end a_share_correct_l141_141667


namespace calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l141_141989

theorem calculation_a_squared_plus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  a^2 + b^2 = 6 := by
  sorry

theorem calculation_a_minus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  (a - b)^2 = 8 := by
  sorry

end calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l141_141989


namespace inequality_solution_l141_141152

theorem inequality_solution (x : ℝ) : 
  (x / (x + 5) ≥ 0) ↔ (x ∈ (Set.Iio (-5)).union (Set.Ici 0)) :=
by
  sorry

end inequality_solution_l141_141152


namespace triangle_area_l141_141758

/-- Given a triangle ABC with BC = 12 cm and AD perpendicular to BC with AD = 15 cm,
    prove that the area of triangle ABC is 90 square centimeters. -/
theorem triangle_area {BC AD : ℝ} (hBC : BC = 12) (hAD : AD = 15) :
  (1 / 2) * BC * AD = 90 := by
  sorry

end triangle_area_l141_141758


namespace arithmetic_mean_of_fractions_l141_141616

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l141_141616


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l141_141613

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l141_141613


namespace symmetric_point_origin_l141_141590

def Point := (ℝ × ℝ × ℝ)

def symmetric_point (P : Point) (O : Point) : Point :=
  let (x, y, z) := P
  let (ox, oy, oz) := O
  (2 * ox - x, 2 * oy - y, 2 * oz - z)

theorem symmetric_point_origin :
  symmetric_point (1, 3, 5) (0, 0, 0) = (-1, -3, -5) :=
by sorry

end symmetric_point_origin_l141_141590


namespace max_value_trig_expression_l141_141355

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end max_value_trig_expression_l141_141355


namespace geometric_sum_s9_l141_141747

variable (S : ℕ → ℝ)

theorem geometric_sum_s9
  (h1 : S 3 = 7)
  (h2 : S 6 = 63) :
  S 9 = 511 :=
by
  sorry

end geometric_sum_s9_l141_141747


namespace circles_intersect_at_circumcenter_l141_141421

-- Definitions and Properties
variables {A B C D E F O : Type}
variables (is_midpoint : ∀ {X Y M}, M = (X + Y) / 2)
variables (triangle_ABC : (A B C : Type))
variables (mid_AB : D = (A + B) / 2)
variables (mid_BC : E = (B + C) / 2)
variables (mid_AC : F = (A + C) / 2)
variables (circumcenter : ∀ {A B C}, O = circumcenter A B C)
variables (circle_k1 : k_1 contains_points [A, D, F])
variables (circle_k2 : k_2 contains_points [B, E, D])
variables (circle_k3 : k_3 contains_points [C, F, E])

-- Theorem Statement
theorem circles_intersect_at_circumcenter :
  O ∈ k_1 ∧ O ∈ k_2 ∧ O ∈ k_3 :=
by
  sorry

end circles_intersect_at_circumcenter_l141_141421


namespace problem_1_problem_2_l141_141368

theorem problem_1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : (a + b) * (a^5 + b^5) ≥ 4 :=
sorry

theorem problem_2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : a + b ≤ 2 :=
sorry

end problem_1_problem_2_l141_141368


namespace max_distance_difference_l141_141913

-- Given definitions and conditions
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 15 = 1
def circle1 (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

-- Main theorem to prove the maximum value of |PM| - |PN|
theorem max_distance_difference (P M N : ℝ × ℝ) :
  hyperbola P.1 P.2 →
  circle1 M.1 M.2 →
  circle2 N.1 N.2 →
  ∃ max_val : ℝ, max_val = 5 :=
by
  -- Proof skipped, only statement is required
  sorry

end max_distance_difference_l141_141913


namespace seven_line_intersections_twenty_one_line_intersections_l141_141203

-- Problem 1

theorem seven_line_intersections : 
  let k_values : List ℝ := [0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9]
  (lines : Set (ℝ → ℝ)) := 
  {f | ∃ k ∈ k_values, ∀ x, f x = - k * x - k^3} 
  (intersections : Set (ℝ × ℝ)) :=
  {p | ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ ∃ x, p = (x, l1 x) ∧ p = (x, l2 x)} 
  (intersections ∧ ∀ p1 p2 ∈ intersections, p1 = p2) →
  intersections.card = 11 :=
sorry

-- Problem 2

theorem twenty_one_line_intersections : 
  let k_values : List ℝ := [0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 
                            0.6, -0.6, 0.7, -0.7, 0.8, -0.8, 0.9, -0.9, 1.0, -1.0]
  (lines : Set (ℝ → ℝ)) := 
  {f | ∃ k ∈ k_values, ∀ x, f x = - k * x - k^3} 
  (intersections : Set (ℝ × ℝ)) :=
  {p | ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ ∃ x, p = (x, l1 x) ∧ p = (x, l2 x)} 
  (intersections ∧ ∀ p1 p2 ∈ intersections, p1 = p2) →
  intersections.card = 110 :=
sorry

end seven_line_intersections_twenty_one_line_intersections_l141_141203


namespace john_finish_work_alone_in_48_days_l141_141247

variable {J R : ℝ}

theorem john_finish_work_alone_in_48_days
  (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 2 / 3)
  (h3 : 16 * J = 1 / 3) :
  1 / J = 48 := 
by
  sorry

end john_finish_work_alone_in_48_days_l141_141247


namespace largest_n_crates_same_orange_count_l141_141185

theorem largest_n_crates_same_orange_count :
  ∀ (num_crates : ℕ) (min_oranges max_oranges : ℕ),
    num_crates = 200 →
    min_oranges = 100 →
    max_oranges = 130 →
    (∃ (n : ℕ), n = 7 ∧ (∃ (distribution : ℕ → ℕ), 
      (∀ x, min_oranges ≤ x ∧ x ≤ max_oranges) ∧ 
      (∀ x, distribution x ≤ num_crates ∧ 
          ∃ y, distribution y ≥ n))) := sorry

end largest_n_crates_same_orange_count_l141_141185


namespace find_value_of_y_l141_141740

theorem find_value_of_y (x y : ℕ) 
    (h1 : 2^x - 2^y = 3 * 2^12) 
    (h2 : x = 14) : 
    y = 13 := 
by
  sorry

end find_value_of_y_l141_141740


namespace sum_ages_divya_nacho_l141_141001

theorem sum_ages_divya_nacho
  (divya_current_age : ℕ)
  (h1 : divya_current_age = 5)
  (h2 : ∀ n, Nacho_in_5_years n = 3 * (divya_current_age + 5)) :
  (divya_current_age + Nacho_current_age) = 40 :=
by
  let divya_age_5_years := divya_current_age + 5
  have nacho_age_5_years := 3 * divya_age_5_years
  let nacho_current_age := nacho_age_5_years - 5
  exact sorry

end sum_ages_divya_nacho_l141_141001


namespace halfway_between_l141_141031

theorem halfway_between (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/15) : (a + b) / 2 = 3 / 40 := by
  -- proofs go here
  sorry

end halfway_between_l141_141031


namespace solve_for_x_l141_141396

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x + 4 * x = 12 + 9 + 6 → x = 3 :=
by
  sorry

end solve_for_x_l141_141396


namespace lcm_six_ten_fifteen_is_30_l141_141531

-- Define the numbers and their prime factorizations
def six := 6
def ten := 10
def fifteen := 15

noncomputable def lcm_six_ten_fifteen : ℕ :=
  Nat.lcm (Nat.lcm six ten) fifteen

-- The theorem to prove the LCM
theorem lcm_six_ten_fifteen_is_30 : lcm_six_ten_fifteen = 30 :=
  sorry

end lcm_six_ten_fifteen_is_30_l141_141531


namespace unique_third_rectangle_exists_l141_141466

-- Define the given rectangles.
def rect1_length : ℕ := 3
def rect1_width : ℕ := 8
def rect2_length : ℕ := 2
def rect2_width : ℕ := 5

-- Define the areas of the given rectangles.
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width

-- Define the total area covered by the two given rectangles.
def total_area_without_third : ℕ := area_rect1 + area_rect2

-- We need to prove that there exists one unique configuration for the third rectangle.
theorem unique_third_rectangle_exists (a b : ℕ) : 
  (total_area_without_third + a * b = 34) → 
  (a * b = 4) → 
  (a = 4 ∧ b = 1 ∨ a = 1 ∧ b = 4) :=
by sorry

end unique_third_rectangle_exists_l141_141466


namespace profit_percentage_is_33_point_33_l141_141893

variable (C S : ℝ)

-- Initial condition based on the problem statement
axiom cost_eq_sell : 20 * C = 15 * S

-- Statement to prove
theorem profit_percentage_is_33_point_33 (h : 20 * C = 15 * S) : (S - C) / C * 100 = 33.33 := 
sorry

end profit_percentage_is_33_point_33_l141_141893


namespace Mary_sleep_hours_for_avg_score_l141_141428

def sleep_score_inverse_relation (sleep1 score1 sleep2 score2 : ℝ) : Prop :=
  sleep1 * score1 = sleep2 * score2

theorem Mary_sleep_hours_for_avg_score (h1 s1 s2 : ℝ) (h_eq : h1 = 6) (s1_eq : s1 = 60)
  (avg_score_cond : (s1 + s2) / 2 = 75) :
  ∃ h2 : ℝ, sleep_score_inverse_relation h1 s1 h2 s2 ∧ h2 = 4 := 
by
  sorry

end Mary_sleep_hours_for_avg_score_l141_141428


namespace jill_total_trip_duration_is_101_l141_141763

def first_bus_wait_time : Nat := 12
def first_bus_ride_time : Nat := 30
def first_bus_delay_time : Nat := 5

def walk_time_to_train : Nat := 10
def train_wait_time : Nat := 8
def train_ride_time : Nat := 20
def train_delay_time : Nat := 3

def second_bus_wait_time : Nat := 20
def second_bus_ride_time : Nat := 6

def route_b_combined_time := (second_bus_wait_time + second_bus_ride_time) / 2

def total_trip_duration : Nat := 
  first_bus_wait_time + first_bus_ride_time + first_bus_delay_time +
  walk_time_to_train + train_wait_time + train_ride_time + train_delay_time +
  route_b_combined_time

theorem jill_total_trip_duration_is_101 : total_trip_duration = 101 := by
  sorry

end jill_total_trip_duration_is_101_l141_141763


namespace number_of_minutes_away_l141_141331

noncomputable def time_away (n : ℚ) :=
  |(150 : ℚ) - (11 * n / 2)| = 120

theorem number_of_minutes_away 
  (n₁ n₂ : ℚ) 
  (h₁ : time_away n₁) 
  (h₂ : time_away n₂) 
  (h_neq : n₁ ≠ n₂):
  n₂ - n₁ = 480 / 11 := 
sorry

end number_of_minutes_away_l141_141331


namespace candy_bar_cost_l141_141514

-- Definitions of conditions
def soft_drink_cost : ℕ := 4
def num_soft_drinks : ℕ := 2
def num_candy_bars : ℕ := 5
def total_cost : ℕ := 28

-- Proof Statement
theorem candy_bar_cost : (total_cost - num_soft_drinks * soft_drink_cost) / num_candy_bars = 4 := by
  sorry

end candy_bar_cost_l141_141514


namespace common_area_of_rectangle_and_circle_eqn_l141_141695

theorem common_area_of_rectangle_and_circle_eqn :
  let rect_length := 8
  let rect_width := 4
  let circle_radius := 3
  let common_area := (3^2 * 2 * Real.pi / 4) - 2 * Real.sqrt 5  
  common_area = (9 * Real.pi / 2) - 2 * Real.sqrt 5 := 
sorry

end common_area_of_rectangle_and_circle_eqn_l141_141695


namespace correct_operation_l141_141312

theorem correct_operation (a : ℕ) :
  (a^2 * a^3 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^6 / a^2 = a^3) ∧ ¬(3 * a^2 - 2 * a = a^2) :=
by
  sorry

end correct_operation_l141_141312


namespace someone_received_grade_D_or_F_l141_141236

theorem someone_received_grade_D_or_F (m x : ℕ) (hboys : ∃ n : ℕ, n = m + 3) 
  (hgrades_B : ∃ k : ℕ, k = x + 2) (hgrades_C : ∃ l : ℕ, l = 2 * (x + 2)) :
  ∃ p : ℕ, p = 1 ∨ p = 2 :=
by
  sorry

end someone_received_grade_D_or_F_l141_141236


namespace arithmetic_mean_of_fractions_l141_141640

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l141_141640


namespace find_a_for_perfect_square_trinomial_l141_141539

theorem find_a_for_perfect_square_trinomial (a : ℝ) :
  (∃ b : ℝ, x^2 - 8*x + a = (x - b)^2) ↔ a = 16 :=
by sorry

end find_a_for_perfect_square_trinomial_l141_141539


namespace boat_speed_in_still_water_l141_141237

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 13) (h2 : B - S = 9) : B = 11 :=
by
  sorry

end boat_speed_in_still_water_l141_141237


namespace batteries_manufactured_l141_141485

theorem batteries_manufactured (gather_time create_time : Nat) (robots : Nat) (hours : Nat) (total_batteries : Nat) :
  gather_time = 6 →
  create_time = 9 →
  robots = 10 →
  hours = 5 →
  total_batteries = (hours * 60 / (gather_time + create_time)) * robots →
  total_batteries = 200 :=
by
  intros h_gather h_create h_robots h_hours h_batteries
  simp [h_gather, h_create, h_robots, h_hours] at h_batteries
  exact h_batteries

end batteries_manufactured_l141_141485


namespace bus_A_speed_l141_141855

-- Define the conditions
variables (v_A v_B : ℝ)
axiom equation1 : v_A - v_B = 15
axiom equation2 : v_A + v_B = 75

-- The main theorem we want to prove
theorem bus_A_speed : v_A = 45 :=
by {
  sorry
}

end bus_A_speed_l141_141855


namespace problem_prove_a5_b5_c5_l141_141735

theorem problem_prove_a5_b5_c5 :
  ∀ (a b c : ℝ), 
    a + b + c = 1 ∧ 
    a^2 + b^2 + c^2 = 3 ∧ 
    a^3 + b^3 + c^3 = 4 → 
    a^5 + b^5 + c^5 = 11 / 3 :=
by
  intros a b c h
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end problem_prove_a5_b5_c5_l141_141735


namespace range_of_a_l141_141402

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (4 * (x - 1) > 3 * x - 1) ∧ (5 * x > 3 * x + 2 * a) ↔ (x > 3)) ↔ (a ≤ 3) :=
by
  sorry

end range_of_a_l141_141402


namespace find_angle_C_l141_141558

theorem find_angle_C (a b c : ℝ) (h : a ^ 2 + b ^ 2 - c ^ 2 + a * b = 0) : 
  C = 2 * pi / 3 := 
sorry

end find_angle_C_l141_141558


namespace star_operation_result_l141_141364

def set_minus (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∉ B}

def set_star (A B : Set ℝ) : Set ℝ :=
  set_minus A B ∪ set_minus B A

def A : Set ℝ := { y : ℝ | y ≥ 0 }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 3 }

theorem star_operation_result :
  set_star A B = {x : ℝ | (-3 ≤ x ∧ x < 0) ∨ (x > 3)} :=
  sorry

end star_operation_result_l141_141364


namespace cheolsu_weight_l141_141700

variable (C M : ℝ)

theorem cheolsu_weight:
  (C = (2/3) * M) →
  (C + 72 = 2 * M) →
  C = 36 :=
by
  intros h1 h2
  sorry

end cheolsu_weight_l141_141700


namespace sculpture_cost_in_chinese_yuan_l141_141784

theorem sculpture_cost_in_chinese_yuan
  (usd_to_nad : ℝ)
  (usd_to_cny : ℝ)
  (cost_nad : ℝ)
  (h1 : usd_to_nad = 8)
  (h2 : usd_to_cny = 5)
  (h3 : cost_nad = 160) :
  (cost_nad / usd_to_nad) * usd_to_cny = 100 :=
by
  sorry

end sculpture_cost_in_chinese_yuan_l141_141784


namespace original_price_l141_141851

variable (a : ℝ)

theorem original_price (h : 0.6 * x = a) : x = (5 / 3) * a :=
sorry

end original_price_l141_141851


namespace distinct_triple_identity_l141_141138

theorem distinct_triple_identity (p q r : ℝ) 
  (h1 : p ≠ q) 
  (h2 : q ≠ r) 
  (h3 : r ≠ p)
  (h : (p / (q - r)) + (q / (r - p)) + (r / (p - q)) = 3) : 
  (p^2 / (q - r)^2) + (q^2 / (r - p)^2) + (r^2 / (p - q)^2) = 3 :=
by 
  sorry

end distinct_triple_identity_l141_141138


namespace solve_eqn_l141_141973

noncomputable def root_expr (a b k x : ℝ) : ℝ := Real.sqrt ((a + b * Real.sqrt k)^x)

theorem solve_eqn: {x : ℝ | root_expr 3 2 2 x + root_expr 3 (-2) 2 x = 6} = {2, -2} :=
by
  sorry

end solve_eqn_l141_141973


namespace sequence_formula_l141_141413

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 2 = 3 / 2) (h2 : a 3 = 7 / 3) 
  (h3 : ∀ n : ℕ, ∃ r : ℚ, (∀ m : ℕ, m ≥ 2 → (m * a m + 1) / (n * a n + 1) = r ^ (m - n))) :
  a n = (2^n - 1) / n := 
sorry

end sequence_formula_l141_141413


namespace number_of_girls_l141_141459

variable (total_children : ℕ) (boys : ℕ)

theorem number_of_girls (h1 : total_children = 117) (h2 : boys = 40) : 
  total_children - boys = 77 := by
  sorry

end number_of_girls_l141_141459


namespace zero_of_log_function_l141_141811

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem zero_of_log_function : ∃ x : ℝ, log_base_2 (3 - 2 * x) = 0 ↔ x = 1 :=
by
  -- We define log_base_2(3 - 2 * x) and then find x for which the equation equals zero
  sorry

end zero_of_log_function_l141_141811


namespace arithmetic_mean_of_fractions_l141_141625

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l141_141625


namespace katya_solves_enough_l141_141260

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l141_141260


namespace no_value_of_b_l141_141080

theorem no_value_of_b (b : ℤ) : ¬ ∃ (n : ℤ), 2 * b^2 + 3 * b + 2 = n^2 := 
sorry

end no_value_of_b_l141_141080


namespace sufficient_condition_for_having_skin_l141_141586

theorem sufficient_condition_for_having_skin (H_no_skin_no_hair : ¬skin → ¬hair) :
  (hair → skin) :=
sorry

end sufficient_condition_for_having_skin_l141_141586


namespace time_for_A_l141_141679

theorem time_for_A (A B C : ℝ) 
  (h1 : 1/B + 1/C = 1/3) 
  (h2 : 1/A + 1/C = 1/2) 
  (h3 : 1/B = 1/30) : 
  A = 5/2 := 
by
  sorry

end time_for_A_l141_141679


namespace magnitude_of_b_l141_141227

open Real

noncomputable def a : ℝ × ℝ := (-sqrt 3, 1)

theorem magnitude_of_b (b : ℝ × ℝ)
    (h1 : (a.1 + 2 * b.1, a.2 + 2 * b.2) = (a.1, a.2))
    (h2 : (a.1 + b.1, a.2 + b.2) = (b.1, b.2)) :
    sqrt (b.1 ^ 2 + b.2 ^ 2) = sqrt 2 :=
sorry

end magnitude_of_b_l141_141227


namespace ratio_of_red_to_blue_marbles_l141_141603

theorem ratio_of_red_to_blue_marbles:
  ∀ (R B : ℕ), 
    R + B = 30 →
    2 * (20 - B) = 10 →
    B = 15 → 
    R = 15 →
    R / B = 1 :=
by intros R B h₁ h₂ h₃ h₄
   sorry

end ratio_of_red_to_blue_marbles_l141_141603


namespace problem1_problem2_l141_141478

variable {a b : ℝ}

theorem problem1 (h : a ≠ b) : 
  ((b / (a - b)) - (a / (a - b))) = -1 := 
by
  sorry

theorem problem2 (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) : 
  ((a^2 - a * b)/(a^2) / ((a / b) - (b / a))) = (b / (a + b)) := 
by
  sorry

end problem1_problem2_l141_141478


namespace evaluate_N_l141_141713

theorem evaluate_N (N : ℕ) :
    988 + 990 + 992 + 994 + 996 = 5000 - N → N = 40 :=
by
  sorry

end evaluate_N_l141_141713


namespace problem1_problem2_l141_141990

variables {a b : ℝ}

-- Given conditions
def condition1 : a + b = 2 := sorry
def condition2 : a * b = -1 := sorry

-- Proof for a^2 + b^2 = 6
theorem problem1 (h1 : a + b = 2) (h2 : a * b = -1) : a^2 + b^2 = 6 :=
by sorry

-- Proof for (a - b)^2 = 8
theorem problem2 (h1 : a + b = 2) (h2 : a * b = -1) : (a - b)^2 = 8 :=
by sorry

end problem1_problem2_l141_141990


namespace percentage_charge_l141_141689

def car_cost : ℝ := 14600
def initial_savings : ℝ := 14500
def trip_charge : ℝ := 1.5
def number_of_trips : ℕ := 40
def grocery_value : ℝ := 800
def final_savings_needed : ℝ := car_cost - initial_savings

-- The amount earned from trips
def amount_from_trips : ℝ := number_of_trips * trip_charge

-- The amount needed from percentage charge on groceries
def amount_from_percentage (P: ℝ) : ℝ := grocery_value * P

-- The required amount from percentage charge on groceries
def required_amount_from_percentage : ℝ := final_savings_needed - amount_from_trips

theorem percentage_charge (P: ℝ) (h: amount_from_percentage P = required_amount_from_percentage) : P = 0.05 :=
by 
  -- Proof follows from the given condition that amount_from_percentage P = required_amount_from_percentage
  sorry

end percentage_charge_l141_141689


namespace driver_speed_ratio_l141_141818

theorem driver_speed_ratio (V1 V2 x : ℝ) (h : V1 > 0 ∧ V2 > 0 ∧ x > 0)
  (meet_halfway : ∀ t1 t2, t1 = x / (2 * V1) ∧ t2 = x / (2 * V2))
  (earlier_start : ∀ t1 t2, t1 = t2 + x / (2 * (V1 + V2))) :
  V2 / V1 = (1 + Real.sqrt 5) / 2 := by
  sorry

end driver_speed_ratio_l141_141818


namespace ratio_of_falls_l141_141585

variable (SteveFalls : ℕ) (StephFalls : ℕ) (SonyaFalls : ℕ)
variable (H1 : SteveFalls = 3)
variable (H2 : StephFalls = SteveFalls + 13)
variable (H3 : SonyaFalls = 6)

theorem ratio_of_falls : SonyaFalls / (StephFalls / 2) = 3 / 4 := by
  sorry

end ratio_of_falls_l141_141585


namespace solve_for_x_l141_141918

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l141_141918


namespace factorization_problem_l141_141802

theorem factorization_problem 
  (C D : ℤ)
  (h1 : 15 * y ^ 2 - 76 * y + 48 = (C * y - 16) * (D * y - 3))
  (h2 : C * D = 15)
  (h3 : C * (-3) + D * (-16) = -76)
  (h4 : (-16) * (-3) = 48) : 
  C * D + C = 20 :=
by { sorry }

end factorization_problem_l141_141802


namespace candidate_failed_by_45_marks_l141_141407

-- Define the main parameters
def passing_percentage : ℚ := 45 / 100
def candidate_marks : ℝ := 180
def maximum_marks : ℝ := 500
def passing_marks : ℝ := passing_percentage * maximum_marks
def failing_marks : ℝ := passing_marks - candidate_marks

-- State the theorem to be proved
theorem candidate_failed_by_45_marks : failing_marks = 45 := by
  sorry

end candidate_failed_by_45_marks_l141_141407


namespace geometric_sequence_a8_eq_pm1_l141_141004

variable {R : Type*} [LinearOrderedField R]

theorem geometric_sequence_a8_eq_pm1 :
  ∀ (a : ℕ → R), (∀ n : ℕ, ∃ r : R, r ≠ 0 ∧ a n = a 0 * r ^ n) → 
  (a 4 + a 12 = -3) ∧ (a 4 * a 12 = 1) → 
  (a 8 = 1 ∨ a 8 = -1) := by
  sorry

end geometric_sequence_a8_eq_pm1_l141_141004


namespace melanie_attended_games_l141_141036

-- Define the total number of football games and the number of games missed by Melanie.
def total_games := 7
def missed_games := 4

-- Define what we need to prove: the number of games attended by Melanie.
theorem melanie_attended_games : total_games - missed_games = 3 := 
by
  sorry

end melanie_attended_games_l141_141036


namespace smallest_unit_of_money_correct_l141_141056

noncomputable def smallest_unit_of_money (friends : ℕ) (total_bill paid_amount : ℚ) : ℚ :=
  if (total_bill % friends : ℚ) = 0 then
    total_bill / friends
  else
    1 % 100

theorem smallest_unit_of_money_correct :
  smallest_unit_of_money 9 124.15 124.11 = 1 % 100 := 
by
  sorry

end smallest_unit_of_money_correct_l141_141056


namespace min_value_of_diff_squares_l141_141424

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))

theorem min_value_of_diff_squares (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  ∃ minimum_value, minimum_value = 36 ∧ ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → (C x y z)^2 - (D x y z)^2 ≥ minimum_value :=
sorry

end min_value_of_diff_squares_l141_141424


namespace adam_change_l141_141688

-- Defining the given amount Adam has and the cost of the airplane.
def amountAdamHas : ℝ := 5.00
def costOfAirplane : ℝ := 4.28

-- Statement of the theorem to be proven.
theorem adam_change : amountAdamHas - costOfAirplane = 0.72 := by
  sorry

end adam_change_l141_141688


namespace janet_roses_l141_141762

def total_flowers (used_flowers extra_flowers : Nat) : Nat :=
  used_flowers + extra_flowers

def number_of_roses (total tulips : Nat) : Nat :=
  total - tulips

theorem janet_roses :
  ∀ (used_flowers extra_flowers tulips : Nat),
  used_flowers = 11 → extra_flowers = 4 → tulips = 4 →
  number_of_roses (total_flowers used_flowers extra_flowers) tulips = 11 :=
by
  intros used_flowers extra_flowers tulips h_used h_extra h_tulips
  rw [h_used, h_extra, h_tulips]
  -- proof steps skipped
  sorry

end janet_roses_l141_141762


namespace arithmetic_mean_of_fractions_l141_141639

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l141_141639


namespace number_of_eggs_left_l141_141513

theorem number_of_eggs_left (initial_eggs : ℕ) (eggs_eaten_morning : ℕ) (eggs_eaten_afternoon : ℕ) (eggs_left : ℕ) :
    initial_eggs = 20 → eggs_eaten_morning = 4 → eggs_eaten_afternoon = 3 → eggs_left = initial_eggs - (eggs_eaten_morning + eggs_eaten_afternoon) → eggs_left = 13 :=
by
  intros h_initial h_morning h_afternoon h_calc
  rw [h_initial, h_morning, h_afternoon] at h_calc
  norm_num at h_calc
  exact h_calc

end number_of_eggs_left_l141_141513


namespace total_sum_of_k_p_l141_141103

def sum_of_indices (n : ℕ) (h1 : n ≡ 1 [MOD 4]) (h2 : n > 1) : ℕ :=
  (1 / 2 * (n - 1) * (Nat.factorial n)).to_nat -- representing the correct answer

theorem total_sum_of_k_p (n : ℕ) (h1 : n ≡ 1 [MOD 4]) (h2 : n > 1) :
  ∑ P in Finset.univ, k_p P = sum_of_indices n h1 h2 :=
sorry

end total_sum_of_k_p_l141_141103


namespace problem_statement_l141_141887

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 3 = 973 := 
by
  sorry

end problem_statement_l141_141887


namespace equivalent_operation_l141_141311

theorem equivalent_operation (x : ℚ) : 
  (x * (5 / 6) / (2 / 7)) = x * (35 / 12) :=
by
  sorry

end equivalent_operation_l141_141311


namespace arithmetic_mean_of_fractions_l141_141635

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l141_141635


namespace octahedron_tetrahedron_volume_ratio_l141_141327

theorem octahedron_tetrahedron_volume_ratio (s : ℝ) :
  let V_T := (s^3 * Real.sqrt 2) / 12
  let a := s / 2
  let V_O := (a^3 * Real.sqrt 2) / 3
  V_O / V_T = 1 / 2 :=
by
  sorry

end octahedron_tetrahedron_volume_ratio_l141_141327


namespace fixed_point_parabola_l141_141564

theorem fixed_point_parabola (t : ℝ) : 4 * 3^2 + t * 3 - t^2 - 3 * t = 36 := by
  sorry

end fixed_point_parabola_l141_141564


namespace sculpture_cost_NAD_to_CNY_l141_141787

def NAD_to_USD (nad : ℕ) : ℕ := nad / 8
def USD_to_CNY (usd : ℕ) : ℕ := usd * 5

theorem sculpture_cost_NAD_to_CNY (nad : ℕ) : (nad = 160) → (USD_to_CNY (NAD_to_USD nad) = 100) :=
by
  intro h1
  rw [h1]
  -- NAD_to_USD 160 = 160 / 8
  have h2 : NAD_to_USD 160 = 20 := rfl
  -- USD_to_CNY 20 = 20 * 5
  have h3 : USD_to_CNY 20 = 100 := rfl
  -- Concluding the theorem
  rw [h2, h3]
  reflexivity

end sculpture_cost_NAD_to_CNY_l141_141787


namespace find_y_l141_141892

theorem find_y (x y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 3) (h2 : x = -5) : y = 44 := by
  sorry

end find_y_l141_141892


namespace count_integers_congruent_mod_l141_141734

theorem count_integers_congruent_mod (n : ℕ) (h₁ : n < 1200) (h₂ : n ≡ 3 [MOD 7]) : 
  ∃ (m : ℕ), (m = 171) :=
by
  sorry

end count_integers_congruent_mod_l141_141734


namespace proof_tan_alpha_proof_exp_l141_141218

-- Given conditions
variables (α : ℝ) (h_condition1 : Real.tan (α + Real.pi / 4) = - 1 / 2) (h_condition2 : Real.pi / 2 < α ∧ α < Real.pi)

-- To prove
theorem proof_tan_alpha :
  Real.tan α = -3 :=
sorry -- proof goes here

theorem proof_exp :
  (Real.sin (2 * α) - 2 * Real.cos α ^ 2) / Real.sin (α - Real.pi / 4) = - 2 * Real.sqrt 5 / 5 :=
sorry -- proof goes here

end proof_tan_alpha_proof_exp_l141_141218


namespace set_intersection_l141_141542

open Set

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def intersection : Set ℕ := {1, 3}

theorem set_intersection : M ∩ N = intersection := by
  sorry

end set_intersection_l141_141542


namespace shirts_made_today_l141_141960

def shirts_per_minute : ℕ := 8
def working_minutes : ℕ := 2

theorem shirts_made_today (h1 : shirts_per_minute = 8) (h2 : working_minutes = 2) : shirts_per_minute * working_minutes = 16 := by
  sorry

end shirts_made_today_l141_141960


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l141_141609

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l141_141609


namespace identical_machine_production_l141_141438

-- Definitions based on given conditions
def machine_production_rate (machines : ℕ) (rate : ℕ) :=
  rate / machines

def bottles_in_minute (machines : ℕ) (rate_per_machine : ℕ) :=
  machines * rate_per_machine

def total_bottles (bottle_rate_per_minute : ℕ) (minutes : ℕ) :=
  bottle_rate_per_minute * minutes

-- Theorem to prove based on the question == answer given conditions
theorem identical_machine_production :
  ∀ (machines_initial machines_final : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ),
    machines_initial = 6 →
    machines_final = 12 →
    bottles_per_minute = 270 →
    minutes = 4 →
    total_bottles (bottles_in_minute machines_final (machine_production_rate machines_initial bottles_per_minute)) minutes = 2160 := by
  intros
  sorry

end identical_machine_production_l141_141438


namespace fourth_student_number_systematic_sampling_l141_141751

theorem fourth_student_number_systematic_sampling :
  ∀ (students : Finset ℕ), students = Finset.range 55 →
  ∀ (sample_size : ℕ), sample_size = 4 →
  ∀ (numbers_in_sample : Finset ℕ),
  numbers_in_sample = {3, 29, 42} →
  ∃ (fourth_student : ℕ), fourth_student = 44 :=
  by sorry

end fourth_student_number_systematic_sampling_l141_141751


namespace cos_double_angle_l141_141096

theorem cos_double_angle (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = -7 / 8 :=
sorry

end cos_double_angle_l141_141096


namespace yoki_cans_collected_l141_141348

theorem yoki_cans_collected (total_cans LaDonna_cans Prikya_cans Avi_cans : ℕ) (half_Avi_cans Yoki_cans : ℕ) 
    (h1 : total_cans = 85) 
    (h2 : LaDonna_cans = 25) 
    (h3 : Prikya_cans = 2 * LaDonna_cans - 3) 
    (h4 : Avi_cans = 8) 
    (h5 : half_Avi_cans = Avi_cans / 2) 
    (h6 : total_cans = LaDonna_cans + Prikya_cans + half_Avi_cans + Yoki_cans) :
    Yoki_cans = 9 := sorry

end yoki_cans_collected_l141_141348


namespace inequality_solution_l141_141153

theorem inequality_solution (x : ℝ) : 
  (x / (x + 5) ≥ 0) ↔ (x ∈ (Set.Iio (-5)).union (Set.Ici 0)) :=
by
  sorry

end inequality_solution_l141_141153


namespace max_value_of_trig_expression_l141_141359

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end max_value_of_trig_expression_l141_141359


namespace solve_system_of_equations_l141_141275

theorem solve_system_of_equations :
  ∃ x y z : ℚ, 
    (y * z = 3 * y + 2 * z - 8) ∧
    (z * x = 4 * z + 3 * x - 8) ∧
    (x * y = 2 * x + y - 1) ∧
    ((x = 2 ∧ y = 3 ∧ z = 1) ∨ 
     (x = 3 ∧ y = 5 / 2 ∧ z = -1)) := 
by
  sorry

end solve_system_of_equations_l141_141275


namespace playground_girls_l141_141457

theorem playground_girls (total_children boys girls : ℕ) (h1 : boys = 40) (h2 : total_children = 117) (h3 : total_children = boys + girls) : girls = 77 := 
by 
  sorry

end playground_girls_l141_141457


namespace g_at_5_l141_141443

def g : ℝ → ℝ := sorry

axiom g_property : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2

theorem g_at_5 : g 5 = -20 :=
by {
  apply sorry
}

end g_at_5_l141_141443


namespace evaluate_expression_l141_141671

-- Introduce the expression as a Lean definition
def expression := (- (1 / 2))⁻¹ + (Real.pi - 3)^0 + abs (1 - Real.sqrt 2) + Real.sin (Real.pi / 4) * Real.sin (Real.pi / 6)

-- State the theorem to be proven
theorem evaluate_expression : expression = (5 * Real.sqrt 2) / 4 - 2 := 
by 
  sorry

end evaluate_expression_l141_141671


namespace green_marbles_count_l141_141321

-- Conditions
def num_red_marbles : ℕ := 2

def probability_of_two_reds (G : ℕ) : ℝ :=
  (num_red_marbles / (num_red_marbles + G)) * ((num_red_marbles - 1) / (num_red_marbles + G - 1))

-- Problem Statement
theorem green_marbles_count (G : ℕ) (h : probability_of_two_reds G = 0.1) : G = 3 :=
sorry

end green_marbles_count_l141_141321


namespace teams_equation_l141_141602

theorem teams_equation (x : ℕ) (h1 : 100 = x + 4*x - 10) : 4 * x + x - 10 = 100 :=
by
  sorry

end teams_equation_l141_141602


namespace decimal_2_09_is_209_percent_l141_141674

-- Definition of the conversion from decimal to percentage
def decimal_to_percentage (x : ℝ) := x * 100

-- Theorem statement
theorem decimal_2_09_is_209_percent : decimal_to_percentage 2.09 = 209 :=
by sorry

end decimal_2_09_is_209_percent_l141_141674


namespace polar_to_rectangular_inequality_range_l141_141332

-- Part A: Transforming a polar coordinate equation to a rectangular coordinate equation
theorem polar_to_rectangular (ρ θ : ℝ) : 
  (ρ^2 * Real.cos θ - ρ = 0) ↔ ((ρ = 0 ∧ 0 = 1) ∨ (ρ ≠ 0 ∧ Real.cos θ = 1 / ρ)) := 
sorry

-- Part B: Determining range for an inequality
theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2-x| + |x+1| ≤ a) ↔ (a ≥ 9) := 
sorry

end polar_to_rectangular_inequality_range_l141_141332


namespace triangle_first_side_length_l141_141360

theorem triangle_first_side_length (x : ℕ) (h1 : x + 20 + 30 = 55) : x = 5 :=
by
  sorry

end triangle_first_side_length_l141_141360


namespace solve_prime_equation_l141_141582

theorem solve_prime_equation (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) 
(h_eq : p^3 - q^3 = 5 * r) : p = 7 ∧ q = 2 ∧ r = 67 := 
sorry

end solve_prime_equation_l141_141582


namespace children_play_time_equal_l141_141235

-- Definitions based on the conditions in the problem
def totalChildren := 7
def totalPlayingTime := 140
def playersAtATime := 2

-- The statement to be proved
theorem children_play_time_equal :
  (playersAtATime * totalPlayingTime) / totalChildren = 40 := by
sorry

end children_play_time_equal_l141_141235


namespace angle_measure_of_three_times_complementary_l141_141852

def is_complementary (α β : ℝ) : Prop := α + β = 90

def three_times_complement (α : ℝ) : Prop := 
  ∃ β : ℝ, is_complementary α β ∧ α = 3 * β

theorem angle_measure_of_three_times_complementary :
  ∀ α : ℝ, three_times_complement α → α = 67.5 :=
by sorry

end angle_measure_of_three_times_complementary_l141_141852


namespace max_k_constant_for_right_triangle_l141_141775

theorem max_k_constant_for_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a ≤ b) (h2 : b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3*Real.sqrt 2) * a * b * c :=
by 
  sorry

end max_k_constant_for_right_triangle_l141_141775


namespace average_salary_associates_l141_141841

theorem average_salary_associates :
  ∃ (A : ℝ), 
    let total_salary_managers := 15 * 90000
    let total_salary_company := 90 * 40000
    let total_salary_associates := 75 * A
    total_salary_managers + total_salary_associates = total_salary_company ∧
    A = 30000 :=
begin
  use 30000,
  let total_salary_managers := 15 * 90000,
  let total_salary_company := 90 * 40000,
  let total_salary_associates := 75 * 30000,
  split,
  { exact eq.trans (by ring) (by rfl) },
  { exact rfl }
end

end average_salary_associates_l141_141841


namespace election_including_past_officers_l141_141798

def election_problem (total_candidates past_officers: ℕ) (num_positions : ℕ) (num_includes: ℕ) : ℕ :=
  ∑ k in finset.range (num_includes + 1), (nat.choose past_officers k) * (nat.choose (total_candidates - past_officers) (num_positions - k))

theorem election_including_past_officers 
  (total_candidates : ℕ) (past_officers: ℕ) (num_positions : ℕ) (min_past_officers num_includes: ℕ) 
  (total_candidates = 20) 
  (past_officers = 5) 
  (num_positions = 6) 
  (min_past_officers = 1)
  (num_includes = 3) : 
  election_problem total_candidates past_officers num_positions num_includes = 33215 := 
sorry

end election_including_past_officers_l141_141798


namespace radio_selling_price_l141_141326

noncomputable def sellingPrice (costPrice : ℝ) (lossPercentage : ℝ) : ℝ :=
  costPrice - (lossPercentage / 100 * costPrice)

theorem radio_selling_price :
  sellingPrice 490 5 = 465.5 :=
by
  sorry

end radio_selling_price_l141_141326


namespace find_arc_length_of_sector_l141_141833

variable (s r p : ℝ)
variable (h_s : s = 4)
variable (h_r : r = 2)
variable (h_area : 2 * s = r * p)

theorem find_arc_length_of_sector 
  (h_s : s = 4) (h_r : r = 2) (h_area : 2 * s = r * p) :
  p = 4 :=
sorry

end find_arc_length_of_sector_l141_141833


namespace number_of_women_bathing_suits_correct_l141_141676

def men_bathing_suits : ℕ := 14797
def total_bathing_suits : ℕ := 19766

def women_bathing_suits : ℕ :=
  total_bathing_suits - men_bathing_suits

theorem number_of_women_bathing_suits_correct :
  women_bathing_suits = 19669 := by
  -- proof goes here
  sorry

end number_of_women_bathing_suits_correct_l141_141676


namespace find_g3_value_l141_141272

def g (n : ℕ) : ℕ :=
  if n < 5 then 2 * n ^ 2 + 3 else 4 * n + 1

theorem find_g3_value : g (g (g 3)) = 341 := by
  sorry

end find_g3_value_l141_141272


namespace problem1_problem2_l141_141374

noncomputable def f (x a b : ℝ) : ℝ := x^2 - (a+1)*x + b

theorem problem1 (h : ∀ x : ℝ, f x (-4) (-10) < 0 ↔ -5 < x ∧ x < 2) : f x (-4) (-10) < 0 :=
sorry

theorem problem2 (a : ℝ) : 
  (a > 1 → ∀ x : ℝ, f x a a > 0 ↔ x < 1 ∨ x > a) ∧
  (a = 1 → ∀ x : ℝ, f x a a > 0 ↔ x ≠ 1) ∧
  (a < 1 → ∀ x : ℝ, f x a a > 0 ↔ x < a ∨ x > 1) :=
sorry

end problem1_problem2_l141_141374


namespace find_prime_triplet_l141_141352

def is_geometric_sequence (x y z : ℕ) : Prop :=
  (y^2 = x * z)

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_prime_triplet :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  is_geometric_sequence (a + 1) (b + 1) (c + 1) ∧
  (a = 17 ∧ b = 23 ∧ c = 31) :=
by
  sorry

end find_prime_triplet_l141_141352


namespace prove_perpendicular_planes_l141_141145

-- Defining the non-coincident lines m and n
variables {m n : Set Point} {α β : Set Point}

-- Lines and plane relationship definitions
def parallel (x y : Set Point) : Prop := sorry
def perpendicular (x y : Set Point) : Prop := sorry
def subset (x y : Set Point) : Prop := sorry

-- Given conditions
axiom h1 : parallel m n
axiom h2 : subset m α
axiom h3 : perpendicular n β

-- Prove that α is perpendicular to β
theorem prove_perpendicular_planes :
  perpendicular α β :=
  sorry

end prove_perpendicular_planes_l141_141145


namespace arithmetic_sequence_a1_value_l141_141755

theorem arithmetic_sequence_a1_value (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 3 = -6) 
  (h2 : a 7 = a 5 + 4) 
  (h_seq : ∀ n, a (n+1) = a n + d) : 
  a 1 = -10 := 
by
  sorry

end arithmetic_sequence_a1_value_l141_141755


namespace arithmetic_mean_eq_l141_141656

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l141_141656


namespace katya_needs_at_least_ten_l141_141256

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l141_141256


namespace recruits_line_l141_141452

theorem recruits_line
  (x y z : ℕ) 
  (hx : x + y + z + 3 = 211) 
  (hx_peter : x = 50) 
  (hy_nikolai : y = 100) 
  (hz_denis : z = 170) 
  (hxy_ratio : x = 4 * z) : 
  x + y + z + 3 = 211 :=
by
  sorry

end recruits_line_l141_141452


namespace veronica_pits_cherries_in_2_hours_l141_141463

theorem veronica_pits_cherries_in_2_hours :
  ∀ (pounds_cherries : ℕ) (cherries_per_pound : ℕ)
    (time_first_pound : ℕ) (cherries_first_pound : ℕ)
    (time_second_pound : ℕ) (cherries_second_pound : ℕ)
    (time_third_pound : ℕ) (cherries_third_pound : ℕ)
    (minutes_per_hour : ℕ),
  pounds_cherries = 3 →
  cherries_per_pound = 80 →
  time_first_pound = 10 →
  cherries_first_pound = 20 →
  time_second_pound = 8 →
  cherries_second_pound = 20 →
  time_third_pound = 12 →
  cherries_third_pound = 20 →
  minutes_per_hour = 60 →
  ((time_first_pound / cherries_first_pound * cherries_per_pound) + 
   (time_second_pound / cherries_second_pound * cherries_per_pound) + 
   (time_third_pound / cherries_third_pound * cherries_per_pound)) / minutes_per_hour = 2 :=
by
  intros pounds_cherries cherries_per_pound
         time_first_pound cherries_first_pound
         time_second_pound cherries_second_pound
         time_third_pound cherries_third_pound
         minutes_per_hour
         pounds_eq cherries_eq
         time1_eq cherries1_eq
         time2_eq cherries2_eq
         time3_eq cherries3_eq
         mins_eq

  -- You would insert the proof here
  sorry

end veronica_pits_cherries_in_2_hours_l141_141463


namespace neg_exists_is_forall_l141_141291

theorem neg_exists_is_forall: 
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by
  sorry

end neg_exists_is_forall_l141_141291


namespace arithmetic_sequence_common_difference_divisible_by_p_l141_141768

theorem arithmetic_sequence_common_difference_divisible_by_p 
  (n : ℕ) (a : ℕ → ℕ) (h1 : n ≥ 2021) (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) 
  (h3 : a 1 > 2021) (h4 : ∀ i, 1 ≤ i → i ≤ n → Nat.Prime (a i)) : 
  ∀ p, Nat.Prime p → p < 2021 → ∃ d, (∀ m, 2 ≤ m → a m = a 1 + (m - 1) * d) ∧ p ∣ d := 
sorry

end arithmetic_sequence_common_difference_divisible_by_p_l141_141768


namespace sword_length_difference_l141_141195

def christopher_sword := 15.0
def jameson_sword := 2 * christopher_sword + 3
def june_sword := jameson_sword + 5
def average_length := (christopher_sword + jameson_sword + june_sword) / 3
def laura_sword := average_length - 0.1 * average_length
def difference := june_sword - laura_sword

theorem sword_length_difference :
  difference = 12.197 := 
sorry

end sword_length_difference_l141_141195


namespace number_of_tangent_lines_l141_141133

def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 + a

def on_line (a x y : ℝ) : Prop := 3 * x + y = a + 1

theorem number_of_tangent_lines (a m : ℝ) (h1 : on_line a m (a + 1 - 3 * m)) :
  ∃ n : ℤ, n = 1 ∨ n = 2 :=
sorry

end number_of_tangent_lines_l141_141133


namespace turkey_weight_l141_141908

theorem turkey_weight (total_time_minutes roast_time_per_pound number_of_turkeys : ℕ) 
  (h1 : total_time_minutes = 480) 
  (h2 : roast_time_per_pound = 15)
  (h3 : number_of_turkeys = 2) : 
  (total_time_minutes / number_of_turkeys) / roast_time_per_pound = 16 :=
by
  sorry

end turkey_weight_l141_141908


namespace inequality_holds_l141_141527

variable (a t1 t2 t3 t4 : ℝ)

theorem inequality_holds
  (a_pos : 0 < a)
  (h_a_le : a ≤ 7/9)
  (t1_pos : 0 < t1)
  (t2_pos : 0 < t2)
  (t3_pos : 0 < t3)
  (t4_pos : 0 < t4)
  (h_prod : t1 * t2 * t3 * t4 = a^4) :
  (1 / Real.sqrt (1 + t1) + 1 / Real.sqrt (1 + t2) + 1 / Real.sqrt (1 + t3) + 1 / Real.sqrt (1 + t4)) ≤ (4 / Real.sqrt (1 + a)) :=
by
  sorry 

end inequality_holds_l141_141527


namespace find_y_value_l141_141277

def op (a b : ℤ) : ℤ := 4 * a + 2 * b

theorem find_y_value : ∃ y : ℤ, op 3 (op 4 y) = -14 ∧ y = -29 / 2 := sorry

end find_y_value_l141_141277


namespace emily_widgets_production_l141_141431

variable (w t : ℕ) (work_hours_monday work_hours_tuesday production_monday production_tuesday : ℕ)

theorem emily_widgets_production :
  (w = 2 * t) → 
  (work_hours_monday = t) →
  (work_hours_tuesday = t - 3) →
  (production_monday = w * work_hours_monday) → 
  (production_tuesday = (w + 6) * work_hours_tuesday) →
  (production_monday - production_tuesday) = 18 :=
by
  intros hw hwm hwmt hpm hpt
  sorry

end emily_widgets_production_l141_141431


namespace graph_crosses_x_axis_at_origin_l141_141519

-- Let g(x) be a quadratic function defined as ax^2 + bx
def g (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Define the conditions a ≠ 0 and b ≠ 0
axiom a_ne_0 (a : ℝ) : a ≠ 0
axiom b_ne_0 (b : ℝ) : b ≠ 0

-- The problem statement
theorem graph_crosses_x_axis_at_origin (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  ∃ x : ℝ, g a b x = 0 ∧ ∀ x', g a b x' = 0 → x' = 0 ∨ x' = -b / a :=
sorry

end graph_crosses_x_axis_at_origin_l141_141519


namespace tan_two_alpha_l141_141721

theorem tan_two_alpha (α β : ℝ) (h₁ : Real.tan (α - β) = -3/2) (h₂ : Real.tan (α + β) = 3) :
  Real.tan (2 * α) = 3/11 := 
sorry

end tan_two_alpha_l141_141721


namespace proof1_proof2_l141_141806

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  |a * x - 2| - |x + 2|

-- Statement for proof 1
theorem proof1 (x : ℝ)
  (a : ℝ) (h : a = 2) (hx : f 2 x ≤ 1) : -1/3 ≤ x ∧ x ≤ 5 :=
sorry

-- Statement for proof 2
theorem proof2 (a : ℝ)
  (h : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) : a = 1 ∨ a = -1 :=
sorry

end proof1_proof2_l141_141806


namespace number_of_eggs_left_l141_141512

theorem number_of_eggs_left (initial_eggs : ℕ) (eggs_eaten_morning : ℕ) (eggs_eaten_afternoon : ℕ) (eggs_left : ℕ) :
    initial_eggs = 20 → eggs_eaten_morning = 4 → eggs_eaten_afternoon = 3 → eggs_left = initial_eggs - (eggs_eaten_morning + eggs_eaten_afternoon) → eggs_left = 13 :=
by
  intros h_initial h_morning h_afternoon h_calc
  rw [h_initial, h_morning, h_afternoon] at h_calc
  norm_num at h_calc
  exact h_calc

end number_of_eggs_left_l141_141512


namespace parallel_lines_value_of_a_l141_141111

theorem parallel_lines_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, ax + (a+2)*y + 2 = 0 → x + a*y + 1 = 0 → ∀ m n : ℝ, ax + (a + 2)*n + 2 = 0 → x + a*n + 1 = 0) →
  a = -1 := 
sorry

end parallel_lines_value_of_a_l141_141111


namespace remainder_eq_four_l141_141661

theorem remainder_eq_four {x : ℤ} (h : x % 61 = 24) : x % 5 = 4 :=
sorry

end remainder_eq_four_l141_141661


namespace total_sales_l141_141579

-- Define sales of Robyn and Lucy
def Robyn_sales : Nat := 47
def Lucy_sales : Nat := 29

-- Prove total sales
theorem total_sales : Robyn_sales + Lucy_sales = 76 :=
by
  sorry

end total_sales_l141_141579


namespace alyssa_money_after_movies_and_carwash_l141_141691

theorem alyssa_money_after_movies_and_carwash : 
  ∀ (allowance spent earned : ℕ), 
  allowance = 8 → 
  spent = allowance / 2 → 
  earned = 8 → 
  (allowance - spent + earned = 12) := 
by 
  intros allowance spent earned h_allowance h_spent h_earned 
  rw [h_allowance, h_spent, h_earned] 
  simp 
  sorry

end alyssa_money_after_movies_and_carwash_l141_141691


namespace find_f2_plus_fneg2_l141_141546

def f (x a: ℝ) := (x + a)^3

theorem find_f2_plus_fneg2 (a : ℝ)
  (h_cond : ∀ x : ℝ, f (1 + x) a = -f (1 - x) a) :
  f 2 (-1) + f (-2) (-1) = -26 :=
by
  sorry

end find_f2_plus_fneg2_l141_141546


namespace reflection_matrix_condition_l141_141202

noncomputable def reflection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![-(3/4 : ℝ), 1/4]]

noncomputable def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem reflection_matrix_condition (a b : ℝ) :
  (reflection_matrix a b)^2 = identity_matrix ↔ a = -(1/4) ∧ b = -(3/4) :=
  by
  sorry

end reflection_matrix_condition_l141_141202


namespace set_addition_example_1_inequality_Snk_exists_self_generating_and_basis_set_l141_141777

-- Definitions
def add_sets (A B : set ℝ) : set ℝ := {c | ∃ a b, a ∈ A ∧ b ∈ B ∧ c = a + b}

-- Question (1)
theorem set_addition_example_1 : add_sets {0, 1, 2} {-1, 3} = {-1, 0, 1, 3, 4, 5} :=
sorry

-- Question (2)
noncomputable def a_n (n : ℕ) : ℝ := (2 / 3) * n
noncomputable def S_n (n : ℕ) : ℝ := n^2

theorem inequality_Snk (m n k : ℕ) (h1 : m + n = 3 * k) (h2 : m ≠ n) : 
  S_n m + S_n n - (9 / 2) * S_n k > 0 :=
sorry

-- Question (3)
-- Definitions for self-generating set and N* basis set
def is_self_generating_set (A : set ℤ) : Prop :=
∀ a ∈ A, ∃ b c ∈ A, a = b + c

def is_N_star_basis_set (A : set ℤ) : Prop :=
∀ n : ℕ, n > 0 → ∃ S : finset ℤ, (∀ x ∈ S, x ∈ A ∧ x ≠ 0) ∧ S.sum id = n

theorem exists_self_generating_and_basis_set : 
  ∃ A : set ℤ, is_self_generating_set A ∧ is_N_star_basis_set A :=
sorry

end set_addition_example_1_inequality_Snk_exists_self_generating_and_basis_set_l141_141777


namespace intersection_of_A_and_B_l141_141873

-- Define sets A and B
def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The proof statement
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l141_141873


namespace cost_price_of_watch_l141_141070

theorem cost_price_of_watch (C : ℝ) (h1 : ∃ C, 0.91 * C + 220 = 1.04 * C) : C = 1692.31 :=
sorry  -- proof to be provided

end cost_price_of_watch_l141_141070


namespace some_employee_not_team_leader_l141_141510

variables (Employee : Type) (isTeamLeader : Employee → Prop) (meetsDeadline : Employee → Prop)

-- Conditions
axiom some_employee_not_meets_deadlines : ∃ e : Employee, ¬ meetsDeadline e
axiom all_team_leaders_meet_deadlines : ∀ e : Employee, isTeamLeader e → meetsDeadline e

-- Theorem to prove
theorem some_employee_not_team_leader : ∃ e : Employee, ¬ isTeamLeader e :=
sorry

end some_employee_not_team_leader_l141_141510


namespace factorization_identity_l141_141972

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 2 * x^2 - 2

-- Define the factorized form
def factorized_expr (x : ℝ) : ℝ := 2 * (x + 1) * (x - 1)

-- The theorem stating the equality
theorem factorization_identity (x : ℝ) : initial_expr x = factorized_expr x := 
by sorry

end factorization_identity_l141_141972


namespace fib_seventh_term_l141_141899

-- Defining the Fibonacci sequence
def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib n + fib (n + 1)

-- Proving the value of the 7th term given 
-- fib(5) = 5 and fib(6) = 8
theorem fib_seventh_term : fib 7 = 13 :=
by {
    -- Conditions have been used in the definition of Fibonacci sequence
    sorry
}

end fib_seventh_term_l141_141899


namespace range_of_a_l141_141875

noncomputable def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def is_increasing_on_nonneg (f : ℝ → ℝ) :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a
  {f : ℝ → ℝ}
  (hf_even : is_even f)
  (hf_increasing : is_increasing_on_nonneg f)
  (hf_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f (a * x + 1) ≤ f (x - 3)) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l141_141875


namespace tree_height_is_12_l141_141954

-- Let h be the height of the tree in meters.
def height_of_tree (h : ℝ) : Prop :=
  ∃ h, (h / 8 = 150 / 100) → h = 12

theorem tree_height_is_12 : ∃ h : ℝ, height_of_tree h :=
by {
  sorry
}

end tree_height_is_12_l141_141954


namespace find_a_value_l141_141730

theorem find_a_value (a x : ℝ) (h1 : 6 * (x + 8) = 18 * x) (h2 : 6 * x - 2 * (a - x) = 2 * a + x) : a = 7 :=
by
  sorry

end find_a_value_l141_141730


namespace value_after_increase_l141_141062

-- Definition of original number and percentage increase
def original_number : ℝ := 600
def percentage_increase : ℝ := 0.10

-- Theorem stating that after a 10% increase, the value is 660
theorem value_after_increase : original_number * (1 + percentage_increase) = 660 := by
  sorry

end value_after_increase_l141_141062


namespace original_people_count_l141_141142

theorem original_people_count (x : ℕ) 
  (H1 : (x - x / 3) / 2 = 15) : x = 45 := by
  sorry

end original_people_count_l141_141142


namespace pumpkin_patch_pie_filling_l141_141065

def pumpkin_cans (small_pumpkins : ℕ) (large_pumpkins : ℕ) (sales : ℕ) (small_price : ℕ) (large_price : ℕ) : ℕ :=
  let remaining_small_pumpkins := small_pumpkins
  let remaining_large_pumpkins := large_pumpkins
  let small_cans := remaining_small_pumpkins / 2
  let large_cans := remaining_large_pumpkins
  small_cans + large_cans

#eval pumpkin_cans 50 33 120 3 5 -- This evaluates the function with the given data to ensure the logic matches the question

theorem pumpkin_patch_pie_filling : pumpkin_cans 50 33 120 3 5 = 58 := by sorry

end pumpkin_patch_pie_filling_l141_141065


namespace arithmetic_mean_of_fractions_l141_141636

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l141_141636


namespace min_value_x_y_l141_141994

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1 / x + 4 / y + 8) : 
  x + y ≥ 9 :=
sorry

end min_value_x_y_l141_141994


namespace intersection_N_complement_M_l141_141729

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}
def CU_M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | (1 - x) / (x - 3) > 0}

theorem intersection_N_complement_M :
  N ∩ CU_M = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end intersection_N_complement_M_l141_141729


namespace cos_double_angle_l141_141886

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -3/5) : Real.cos (2 * α) = -7/25 :=
by
  sorry

end cos_double_angle_l141_141886


namespace intersection_of_M_and_complement_N_l141_141425

def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | 2 * x < 2 }
def complement_N : Set ℝ := { x | x ≥ 1 }

theorem intersection_of_M_and_complement_N : M ∩ complement_N = { x | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_M_and_complement_N_l141_141425


namespace sum_of_quotient_and_remainder_is_184_l141_141309

theorem sum_of_quotient_and_remainder_is_184 
  (q r : ℕ)
  (h1 : 23 * 17 + 19 = q)
  (h2 : q * 10 = r)
  (h3 : r / 23 = 178)
  (h4 : r % 23 = 6) :
  178 + 6 = 184 :=
by
  -- Inform Lean that we are skipping the proof
  sorry

end sum_of_quotient_and_remainder_is_184_l141_141309


namespace x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l141_141589

theorem x_squared_eq_y_squared_iff_x_eq_y_or_neg_y (x y : ℝ) : 
  (x^2 = y^2) ↔ (x = y ∨ x = -y) := by
  sorry

theorem x_squared_eq_y_squared_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 = y^2 → x = y) ↔ false := by
  sorry

end x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l141_141589


namespace esteban_exercise_each_day_l141_141909

theorem esteban_exercise_each_day (natasha_daily : ℕ) (natasha_days : ℕ) (esteban_days : ℕ) (total_hours : ℕ) :
  let total_minutes := total_hours * 60
  let natasha_total := natasha_daily * natasha_days
  let esteban_total := total_minutes - natasha_total
  esteban_days ≠ 0 →
  natasha_daily = 30 →
  natasha_days = 7 →
  esteban_days = 9 →
  total_hours = 5 →
  esteban_total / esteban_days = 10 := 
by
  intros
  sorry

end esteban_exercise_each_day_l141_141909


namespace determine_event_C_l141_141167

variable (A B C : Prop)
variable (Tallest Shortest : Prop)
variable (Running LongJump ShotPut : Prop)

variables (part_A_Running part_A_LongJump part_A_ShotPut
           part_B_Running part_B_LongJump part_B_ShotPut
           part_C_Running part_C_LongJump part_C_ShotPut : Prop)

variable (not_tallest_A : ¬Tallest → A)
variable (not_tallest_ShotPut : Tallest → ¬ShotPut)
variable (shortest_LongJump : Shortest → LongJump)
variable (not_shortest_B : ¬Shortest → B)
variable (not_running_B : ¬Running → B)

theorem determine_event_C :
  (¬Tallest → A) →
  (Tallest → ¬ShotPut) →
  (Shortest → LongJump) →
  (¬Shortest → B) →
  (¬Running → B) →
  part_C_Running :=
by
  intros h1 h2 h3 h4 h5
  sorry

end determine_event_C_l141_141167


namespace tension_limit_l141_141947

theorem tension_limit (M m g : ℝ) (hM : 0 < M) (hg : 0 < g) :
  (∀ T, (T = Mg ↔ m = 0) → (∀ ε, 0 < ε → ∃ m₀, m > m₀ → |T - 2 * M * g| < ε)) :=
by 
  sorry

end tension_limit_l141_141947


namespace rods_in_one_mile_l141_141219

theorem rods_in_one_mile (mile_to_furlong : ℕ) (furlong_to_rod : ℕ) (mile_eq : 1 = 8 * mile_to_furlong) (furlong_eq: 1 = 50 * furlong_to_rod) : 
  (1 * 8 * 50 = 400) :=
by
  sorry

end rods_in_one_mile_l141_141219


namespace locus_of_moving_point_l141_141772

open Real

theorem locus_of_moving_point
  (M N P Q T E : ℝ × ℝ)
  (a b : ℝ)
  (h_ellipse_M : M.1^2 / 48 + M.2^2 / 16 = 1)
  (h_P : P = (-M.1, M.2))
  (h_Q : Q = (-M.1, -M.2))
  (h_T : T = (M.1, -M.2))
  (h_ellipse_N : N.1^2 / 48 + N.2^2 / 16 = 1)
  (h_perp : (M.1 - N.1) * (M.1 + N.1) + (M.2 - N.2) * (M.2 + N.2) = 0)
  (h_intersection : ∃ x y : ℝ, (y - Q.2) = (N.2 - Q.2)/(N.1 - Q.1) * (x - Q.1) ∧ (y - P.2) = (T.2 - P.2)/(T.1 - P.1) * (x - P.1) ∧ E = (x, y)) : 
  (E.1^2 / 12 + E.2^2 / 4 = 1) :=
  sorry

end locus_of_moving_point_l141_141772


namespace directrix_of_parabola_l141_141716

noncomputable def parabola_directrix (x : ℝ) : ℝ := 4 * x^2 + 4 * x + 1

theorem directrix_of_parabola :
  ∃ (y : ℝ) (x : ℝ), parabola_directrix x = y ∧ y = 4 * (x + 1/2)^2 + 3/4 ∧ y - 1/16 = 11/16 :=
by
  sorry

end directrix_of_parabola_l141_141716


namespace cubic_coefficient_determination_l141_141969

def f (x : ℚ) (A B C D : ℚ) : ℚ := A*x^3 + B*x^2 + C*x + D

theorem cubic_coefficient_determination {A B C D : ℚ}
  (h1 : f 1 A B C D = 0)
  (h2 : f (2/3) A B C D = -4)
  (h3 : f (4/5) A B C D = -16/5) :
  A = 15 ∧ B = -37 ∧ C = 30 ∧ D = -8 :=
  sorry

end cubic_coefficient_determination_l141_141969


namespace jasmine_percent_after_addition_l141_141057

-- Variables definition based on the problem
def original_volume : ℕ := 90
def original_jasmine_percent : ℚ := 0.05
def added_jasmine : ℕ := 8
def added_water : ℕ := 2

-- Total jasmine amount calculation in original solution
def original_jasmine_amount : ℚ := original_jasmine_percent * original_volume

-- New total jasmine amount after addition
def new_jasmine_amount : ℚ := original_jasmine_amount + added_jasmine

-- New total volume calculation after addition
def new_total_volume : ℕ := original_volume + added_jasmine + added_water

-- New jasmine percent in the solution
def new_jasmine_percent : ℚ := (new_jasmine_amount / new_total_volume) * 100

-- The proof statement
theorem jasmine_percent_after_addition : new_jasmine_percent = 12.5 :=
by
  sorry

end jasmine_percent_after_addition_l141_141057


namespace cos_product_inequality_l141_141574

theorem cos_product_inequality : (1 / 8 : ℝ) < (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) ∧
    (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
by
  sorry

end cos_product_inequality_l141_141574


namespace positive_integer_solutions_l141_141980

theorem positive_integer_solutions (n x y z : ℕ) (h1 : n > 1) (h2 : n^z < 2001) (h3 : n^x + n^y = n^z) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ x = k ∧ y = k ∧ z = k + 1) :=
sorry

end positive_integer_solutions_l141_141980


namespace siding_cost_l141_141794

noncomputable def front_wall_width : ℝ := 10
noncomputable def front_wall_height : ℝ := 8
noncomputable def triangle_base : ℝ := 10
noncomputable def triangle_height : ℝ := 4
noncomputable def panel_area : ℝ := 100
noncomputable def panel_cost : ℝ := 30

theorem siding_cost :
  let front_wall_area := front_wall_width * front_wall_height
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let total_area := front_wall_area + triangle_area
  let panels_needed := total_area / panel_area
  let total_cost := panels_needed * panel_cost
  total_cost = 30 := sorry

end siding_cost_l141_141794


namespace parabola_has_one_x_intercept_l141_141383

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end parabola_has_one_x_intercept_l141_141383


namespace find_f_2022_l141_141804

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

variables (f : ℝ → ℝ)
  (h_condition : satisfies_condition f)
  (h_f1 : f 1 = 1)
  (h_f4 : f 4 = 7)

theorem find_f_2022 : f 2022 = 4043 :=
  sorry

end find_f_2022_l141_141804


namespace width_of_first_sheet_l141_141028

theorem width_of_first_sheet (w : ℝ) (h : 2 * (w * 17) = 2 * (8.5 * 11) + 100) : w = 287 / 34 :=
by
  sorry

end width_of_first_sheet_l141_141028


namespace find_x_plus_y_l141_141393

theorem find_x_plus_y (x y : ℝ) (hx : abs x - x + y = 6) (hy : x + abs y + y = 16) : x + y = 10 :=
sorry

end find_x_plus_y_l141_141393


namespace john_finishes_ahead_l141_141248

noncomputable def InitialDistanceBehind : ℝ := 12
noncomputable def JohnSpeed : ℝ := 4.2
noncomputable def SteveSpeed : ℝ := 3.7
noncomputable def PushTime : ℝ := 28

theorem john_finishes_ahead :
  (JohnSpeed * PushTime - InitialDistanceBehind) - (SteveSpeed * PushTime) = 2 := by
  sorry

end john_finishes_ahead_l141_141248


namespace rem_l141_141923

def rem' (x y : ℚ) : ℚ := x - y * (⌊ x / (2 * y) ⌋)

theorem rem'_value : rem' (5 / 9 : ℚ) (-3 / 7) = 62 / 63 := by
  sorry

end rem_l141_141923


namespace count_two_digit_integers_with_remainder_4_when_divided_by_9_l141_141884

theorem count_two_digit_integers_with_remainder_4_when_divided_by_9 :
  ∃ (count : ℕ), count = 10 ∧ 
    ∃ (n : ℕ → ℕ), 
      ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 10 → 
        let k := n i in 10 ≤ k ∧ k < 100 ∧ k % 9 = 4 :=
begin
  sorry
end

end count_two_digit_integers_with_remainder_4_when_divided_by_9_l141_141884


namespace probability_correct_l141_141837

-- Define the balls
inductive BallColor
| Red | White | Black

open BallColor

-- Define the bag containing the balls with specified counts
def bag : List BallColor := [Red, White, White, Black, Black, Black]

-- Function to count the number of satisfying pairs
def count_satisfying_pairs (l : List BallColor) : Nat :=
  let pairs := l.combinations 2
  pairs.count (λ p, p.head = White ∧ p.tail.head = Black ∨ p.head = Black ∧ p.tail.head = White)

-- Total combinations when drawing 2 balls out of 6
def total_combinations (l : List BallColor) : Nat :=
  l.combinations 2 |>.length

-- Compute the probability
def probability_one_white_one_black : ℚ :=
  (count_satisfying_pairs bag : ℚ) / (total_combinations bag : ℚ)

theorem probability_correct :
  probability_one_white_one_black = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end probability_correct_l141_141837


namespace anne_distance_diff_l141_141191

def track_length := 300
def min_distance := 100

-- Define distances functions as described
def distance_AB (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Beth over time 
def distance_AC (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Carmen over time 

theorem anne_distance_diff (Anne_speed Beth_speed Carmen_speed : ℝ) 
  (hneA : Anne_speed ≠ Beth_speed)
  (hneC : Anne_speed ≠ Carmen_speed) :
  ∃ α ≥ 0, min_distance ≤ distance_AB α ∧ min_distance ≤ distance_AC α :=
sorry

end anne_distance_diff_l141_141191


namespace problem_statement_l141_141307

-- Define what it means for a number's tens and ones digits to have a sum of 13
def sum_of_tens_and_ones_equals (n : ℕ) (s : ℕ) : Prop :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit = s

-- State the theorem with the given conditions and correct answer
theorem problem_statement : sum_of_tens_and_ones_equals (6^11) 13 :=
sorry

end problem_statement_l141_141307


namespace T_five_three_l141_141709

def T (a b : ℤ) : ℤ := 4 * a + 6 * b + 2

theorem T_five_three : T 5 3 = 40 := by
  sorry

end T_five_three_l141_141709


namespace percentage_apples_sold_l141_141842

noncomputable def original_apples : ℝ := 750
noncomputable def remaining_apples : ℝ := 300

theorem percentage_apples_sold (A P : ℝ) (h1 : A = 750) (h2 : A * (1 - P / 100) = 300) : 
  P = 60 :=
by
  sorry

end percentage_apples_sold_l141_141842


namespace part1_part2_l141_141554

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if 2a sin B = sqrt(3) b and A is an acute angle, then A = 60 degrees. -/
theorem part1 {a b : ℝ} {A B : ℝ} (h1 : 2 * a * Real.sin B = Real.sqrt 3 * b)
  (h2 : 0 < A ∧ A < Real.pi / 2) : A = Real.pi / 3 :=
sorry

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if b = 5, c = sqrt(5), and cos C = 9 / 10, then a = 4 or a = 5. -/
theorem part2 {a b c : ℝ} {C : ℝ} (h1 : b = 5) (h2 : c = Real.sqrt 5) 
  (h3 : Real.cos C = 9 / 10) : a = 4 ∨ a = 5 :=
sorry

end part1_part2_l141_141554


namespace cost_price_of_book_l141_141677

theorem cost_price_of_book
(marked_price : ℝ)
(list_price : ℝ)
(cost_price : ℝ)
(h1 : marked_price = 69.85)
(h2 : list_price = marked_price * 0.85)
(h3 : list_price = cost_price * 1.25) :
cost_price = 65.75 :=
by
  sorry

end cost_price_of_book_l141_141677


namespace inequality_x_y_l141_141131

theorem inequality_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : x + y ≥ 2 := 
  sorry

end inequality_x_y_l141_141131


namespace race_problem_l141_141481

theorem race_problem (a_speed b_speed : ℕ) (A B : ℕ) (finish_dist : ℕ)
  (h1 : finish_dist = 3000)
  (h2 : A = finish_dist - 500)
  (h3 : B = finish_dist - 600)
  (h4 : A / a_speed = B / b_speed)
  (h5 : a_speed / b_speed = 25 / 24) :
  B - ((500 * b_speed) / a_speed) = 120 :=
by
  sorry

end race_problem_l141_141481


namespace math_majors_consecutive_probability_l141_141465

noncomputable def probability_math_majors_consecutive (n : ℕ) :=
  -- Number of favorable arrangements: 7! * 5!
  (Nat.factorial 7 * Nat.factorial 5) /
  -- Total arrangements: 11!
  (Nat.factorial 11)

theorem math_majors_consecutive_probability :
  probability_math_majors_consecutive 12 = 1 / 66 := 
by
  sorry

end math_majors_consecutive_probability_l141_141465


namespace katya_needs_at_least_ten_l141_141257

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l141_141257


namespace norma_total_cards_l141_141781

theorem norma_total_cards (initial_cards : ℝ) (additional_cards : ℝ) (total_cards : ℝ) 
  (h1 : initial_cards = 88) (h2 : additional_cards = 70) : total_cards = 158 :=
by
  sorry

end norma_total_cards_l141_141781


namespace max_value_of_trig_expression_l141_141358

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end max_value_of_trig_expression_l141_141358


namespace solve_abs_eq_l141_141025

theorem solve_abs_eq {x : ℝ} (h₁ : x ≠ 3 ∧ (x >= 3 ∨ x < 3)) :
  (|x - 3| = 5 - 2 * x) ↔ x = 2 :=
by
  split;
  intro h;
  sorry

end solve_abs_eq_l141_141025


namespace find_line_equation_l141_141091

variable (x y : ℝ)

theorem find_line_equation (hx : x = -5) (hy : y = 2)
  (line_through_point : ∃ a b c : ℝ, a * x + b * y + c = 0)
  (x_intercept_twice_y_intercept : ∀ a b c : ℝ, c ≠ 0 → b ≠ 0 → (a / c) = 2 * (c / b)) :
  ∃ a b c : ℝ, (a * x + b * y + c = 0 ∧ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 1 ∧ b = 2 ∧ c = 1)) :=
sorry

end find_line_equation_l141_141091


namespace cookies_none_of_ingredients_l141_141690

theorem cookies_none_of_ingredients (c : ℕ) (o : ℕ) (r : ℕ) (a : ℕ) (total_cookies : ℕ) :
  total_cookies = 48 ∧ c = total_cookies / 3 ∧ o = (3 * total_cookies + 4) / 5 ∧ r = total_cookies / 2 ∧ a = total_cookies / 8 → 
  ∃ n, n = 19 ∧ (∀ k, k = total_cookies - max c (max o (max r a)) → k ≤ n) :=
by sorry

end cookies_none_of_ingredients_l141_141690


namespace solve_for_x_l141_141016

theorem solve_for_x (x : ℚ) (h : 10 * x = x + 20) : x = 20 / 9 :=
  sorry

end solve_for_x_l141_141016


namespace factorization_correct_l141_141940

theorem factorization_correct (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 :=
by
  sorry

end factorization_correct_l141_141940


namespace independent_events_l141_141230

open ProbabilityTheory

variable {Ω : Type*} {P : Measure Ω}

def A : Set Ω := sorry
def B : Set Ω := sorry

noncomputable def P_A : ℝ := 1 - 2/3
noncomputable def P_B : ℝ := 1/3
noncomputable def P_AB : ℝ := 1/9

theorem independent_events :
  P (A ∩ B) = P A * P B :=
by
  have hA : P A = 1 - P (Aᶜ), by sorry
  sorry

end independent_events_l141_141230


namespace arithmetic_series_sum_is_1620_l141_141077

open ArithmeticSeries

/-- Conditions for the arithmetic series -/
def a1 : ℚ := 10
def an : ℚ := 30
def d : ℚ := 1 / 4

/--Calculating the number of terms in the series -/
def n : ℕ := ((an - a1) / d).to_nat + 1

/--Proving the sum of the arithmetic series is 1620 -/
theorem arithmetic_series_sum_is_1620 : 
  (arithmetic_series_sum a1 an d n) = 1620 :=
by
  sorry

end arithmetic_series_sum_is_1620_l141_141077


namespace math_problem_l141_141197

noncomputable def f (x : ℝ) : ℝ := sorry

theorem math_problem (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (1 - x) = f (1 + x))
  (h2 : ∀ x : ℝ, f (-x) = -f x)
  (h3 : ∀ {x y : ℝ}, (0 ≤ x → x < y → y ≤ 1 → f x < f y)) :
  (f 0 = 0) ∧ 
  (∀ x : ℝ, f (x + 2) = f (-x)) ∧ 
  (∀ x : ℝ, x = -1 ∨ ∀ ε > 0, ε ≠ (x + 1))
:= sorry

end math_problem_l141_141197


namespace interest_difference_l141_141316

noncomputable def difference_between_interest (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) : ℝ :=
  let SI := P * R * T / 100
  let CI := P * (1 + (R / (n*100)))^(n * T) - P
  CI - SI

theorem interest_difference (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) (hP : P = 1200) (hR : R = 10) (hT : T = 1) (hn : n = 2) :
  difference_between_interest P R T n = -59.25 := by
  sorry

end interest_difference_l141_141316


namespace linda_original_amount_l141_141547

theorem linda_original_amount (L L2 : ℕ) 
  (h1 : L = 20) 
  (h2 : L - 5 = L2) : 
  L2 + 5 = 15 := 
sorry

end linda_original_amount_l141_141547


namespace find_triangle_sides_l141_141900

noncomputable def side_lengths (k c d : ℕ) : Prop :=
  let p1 := 26
  let p2 := 32
  let p3 := 30
  (2 * k = 6) ∧ (2 * k + 6 * c = p3) ∧ (2 * c + 2 * d = p1)

theorem find_triangle_sides (k c d : ℕ) (h1 : side_lengths k c d) : k = 3 ∧ c = 4 ∧ d = 5 := 
  sorry

end find_triangle_sides_l141_141900


namespace find_x_for_collinear_vectors_l141_141749

noncomputable def collinear_vectors (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem find_x_for_collinear_vectors : ∀ (x : ℝ), collinear_vectors (2, -3) (x, 6) → x = -4 := by
  intros x h
  sorry

end find_x_for_collinear_vectors_l141_141749


namespace watch_correction_needed_l141_141069

def watch_loses_rate : ℚ := 15 / 4  -- rate of loss per day in minutes
def initial_set_time : ℕ := 15  -- March 15th at 10 A.M.
def report_time : ℕ := 24  -- March 24th at 4 P.M.
def correction (loss_rate per_day min_hrs : ℚ) (days_hrs : ℚ) : ℚ :=
  (days_hrs * (loss_rate / (per_day * min_hrs)))

theorem watch_correction_needed :
  correction watch_loses_rate 24 60 (222) = 34.6875 := 
sorry

end watch_correction_needed_l141_141069


namespace max_ratio_of_mean_70_l141_141423

theorem max_ratio_of_mean_70 (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hmean : (x + y) / 2 = 70) : (x / y ≤ 99 / 41) :=
sorry

end max_ratio_of_mean_70_l141_141423


namespace condition_sufficient_but_not_necessary_l141_141116

theorem condition_sufficient_but_not_necessary (a : ℝ) : (a > 9 → (1 / a < 1 / 9)) ∧ ¬(1 / a < 1 / 9 → a > 9) :=
by 
  sorry

end condition_sufficient_but_not_necessary_l141_141116


namespace line_equation_through_point_with_intercepts_conditions_l141_141088

theorem line_equation_through_point_with_intercepts_conditions :
  ∃ (a b : ℚ) (m c : ℚ), 
    (-5) * m + c = 2 ∧ -- The line passes through A(-5, 2)
    a = 2 * b ∧       -- x-intercept is twice the y-intercept
    (a * m + c = 0 ∨ ((1/m)*a + (1/m)^2 * c+1 = 0)) :=         -- Equations of the line
sorry

end line_equation_through_point_with_intercepts_conditions_l141_141088


namespace sequence_count_zeros_ones_15_l141_141115

-- Definition of the problem
def count_sequences (n : Nat) : Nat := sorry -- Function calculating the number of valid sequences

-- The theorem stating that for sequence length 15, the number of such sequences is 266
theorem sequence_count_zeros_ones_15 : count_sequences 15 = 266 := 
by {
  sorry -- Proof goes here
}

end sequence_count_zeros_ones_15_l141_141115


namespace sum_of_coordinates_reflection_l141_141788

def point (x y : ℝ) : Type := (x, y)

variable (y : ℝ)

theorem sum_of_coordinates_reflection :
  let A := point 3 y in
  let B := point 3 (-y) in
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  sorry

end sum_of_coordinates_reflection_l141_141788


namespace solve_system_1_solve_system_2_l141_141583

theorem solve_system_1 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : 3 * x + 2 * y = 8) : x = 2 ∧ y = 1 :=
by {
  sorry
}

theorem solve_system_2 (x y : ℤ) (h1 : 2 * x + 3 * y = 7) (h2 : 3 * x - 2 * y = 4) : x = 2 ∧ y = 1 :=
by {
  sorry
}

end solve_system_1_solve_system_2_l141_141583


namespace part1_part2_l141_141879

-- Define set A and set B for m = 3
def setA : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def setB_m3 : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Define the complement of B in ℝ and the intersection of complements
def complB_m3 : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def intersection_complB_A : Set ℝ := complB_m3 ∩ setA

-- Verify that the intersection of the complement of B and A equals the given set
theorem part1 : intersection_complB_A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  sorry

-- Define set A and the intersection of A and B
def setA' : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def setAB : Set ℝ := {x | -1 < x ∧ x < 4}

-- Given A ∩ B = {x | -1 < x < 4}, determine m such that B = {x | -1 < x < 4}
theorem part2 : ∃ m : ℝ, (setA' ∩ {x | x^2 - 2 * x - m < 0} = setAB) ∧ m = 8 :=
by
  sorry

end part1_part2_l141_141879


namespace alfred_gain_percent_l141_141315

-- Definitions based on the conditions
def purchase_price : ℝ := 4700
def repair_costs : ℝ := 800
def selling_price : ℝ := 6000

-- Lean statement to prove gain percent
theorem alfred_gain_percent :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 := by
  sorry

end alfred_gain_percent_l141_141315


namespace remainder_of_a_sq_plus_five_mod_seven_l141_141494

theorem remainder_of_a_sq_plus_five_mod_seven (a : ℕ) (h : a % 7 = 4) : (a^2 + 5) % 7 = 0 := 
by 
  sorry

end remainder_of_a_sq_plus_five_mod_seven_l141_141494


namespace max_value_l141_141043

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 18 * x - 5

theorem max_value : ∃ x : ℝ, quadratic_function x = 22 :=
sorry

end max_value_l141_141043


namespace value_of_c_l141_141742

theorem value_of_c
    (x y c : ℝ)
    (h1 : 3 * x - 5 * y = 5)
    (h2 : x / (x + y) = c)
    (h3 : x - y = 2.999999999999999) :
    c = 0.7142857142857142 :=
by
    sorry

end value_of_c_l141_141742


namespace experiment_implies_101_sq_1_equals_10200_l141_141571

theorem experiment_implies_101_sq_1_equals_10200 :
    (5^2 - 1 = 24) →
    (7^2 - 1 = 48) →
    (11^2 - 1 = 120) →
    (13^2 - 1 = 168) →
    (101^2 - 1 = 10200) :=
by
  repeat { intro }
  sorry

end experiment_implies_101_sq_1_equals_10200_l141_141571


namespace circle_area_x2_y2_eq_102_l141_141304

theorem circle_area_x2_y2_eq_102 :
  ∀ (x y : ℝ), (x + 9)^2 + (y - 3)^2 = 102 → π * 102 = 102 * π :=
by
  intros
  sorry

end circle_area_x2_y2_eq_102_l141_141304


namespace arithmetic_mean_of_fractions_l141_141630

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l141_141630


namespace y_exceeds_x_by_35_percent_l141_141177

theorem y_exceeds_x_by_35_percent {x y : ℝ} (h : x = 0.65 * y) : ((y - x) / x) * 100 = 35 :=
by
  sorry

end y_exceeds_x_by_35_percent_l141_141177


namespace sculpture_cost_in_chinese_yuan_l141_141785

theorem sculpture_cost_in_chinese_yuan
  (usd_to_nad : ℝ)
  (usd_to_cny : ℝ)
  (cost_nad : ℝ)
  (h1 : usd_to_nad = 8)
  (h2 : usd_to_cny = 5)
  (h3 : cost_nad = 160) :
  (cost_nad / usd_to_nad) * usd_to_cny = 100 :=
by
  sorry

end sculpture_cost_in_chinese_yuan_l141_141785


namespace generate_one_fifth_from_zero_point_one_generate_rationals_between_zero_and_one_l141_141604

variable {n m : ℚ} 

/-- Starting from the number set containing 0.1, prove we can generate the number 1/5 using an averaging process. -/
theorem generate_one_fifth_from_zero_point_one :
  (∀ a b ∈ {0.1}, (a + b) / 2 ∉ {0.1}) →
  (∀ s : set ℚ, s = {0.1} ∨ ∀ p q ∈ s, (p + q) / 2 ∈ s) →
  (∃ s : set ℚ, 1/5 ∈ s ∧ {0.1} ⊆ s ∧ ∀ p q ∈ s, (p + q) / 2 ∈ s) :=
sorry

/-- Starting from the number set containing 0.1, prove we can generate any rational number between 0 and 1 using an averaging process. -/
theorem generate_rationals_between_zero_and_one :
  (∀ a b ∈ {0.1}, (a + b) / 2 ∉ {0.1}) →
  (∀ s : set ℚ, s = {0.1} ∨ ∀ p q ∈ s, (p + q) / 2 ∈ s) →
  (∀ r : ℚ, 0 < r ∧ r < 1 → ∃ s : set ℚ, r ∈ s ∧ {0.1} ⊆ s ∧ ∀ p q ∈ s, (p + q) / 2 ∈ s) :=
sorry

end generate_one_fifth_from_zero_point_one_generate_rationals_between_zero_and_one_l141_141604


namespace divisible_by_square_of_k_l141_141774

theorem divisible_by_square_of_k (a b l : ℕ) (k : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : a % 2 = 1) (h4 : b % 2 = 1) (h5 : a + b = 2 ^ l) : k = 1 ↔ k^2 ∣ a^k + b^k := 
sorry

end divisible_by_square_of_k_l141_141774


namespace unique_measures_of_A_l141_141290

theorem unique_measures_of_A : 
  ∃ n : ℕ, n = 17 ∧ 
    (∀ A B : ℕ, 
      (A > 0) ∧ (B > 0) ∧ (A + B = 180) ∧ (∃ k : ℕ, A = k * B) → 
      ∃! A : ℕ, A > 0 ∧ (A + B = 180)) :=
sorry

end unique_measures_of_A_l141_141290


namespace hannah_total_spending_l141_141112

def sweatshirt_price : ℕ := 15
def sweatshirt_quantity : ℕ := 3
def t_shirt_price : ℕ := 10
def t_shirt_quantity : ℕ := 2
def socks_price : ℕ := 5
def socks_quantity : ℕ := 4
def jacket_price : ℕ := 50
def discount_rate : ℚ := 0.10

noncomputable def total_cost_before_discount : ℕ :=
  (sweatshirt_quantity * sweatshirt_price) +
  (t_shirt_quantity * t_shirt_price) +
  (socks_quantity * socks_price) +
  jacket_price

noncomputable def total_cost_after_discount : ℚ :=
  total_cost_before_discount - (discount_rate * total_cost_before_discount)

theorem hannah_total_spending : total_cost_after_discount = 121.50 := by
  sorry

end hannah_total_spending_l141_141112


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l141_141610

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l141_141610


namespace destiny_cookies_divisible_l141_141198

theorem destiny_cookies_divisible (C : ℕ) (h : C % 6 = 0) : ∃ k : ℕ, C = 6 * k :=
by {
  sorry
}

end destiny_cookies_divisible_l141_141198


namespace remainder_of_towers_l141_141950

open Nat

def count_towers (m : ℕ) : ℕ :=
  match m with
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 18
  | 5 => 54
  | 6 => 162
  | _ => 0

theorem remainder_of_towers : (count_towers 6) % 100 = 62 :=
  by
  sorry

end remainder_of_towers_l141_141950


namespace product_discount_rate_l141_141482

theorem product_discount_rate (cost_price marked_price : ℝ) (desired_profit_rate : ℝ) :
  cost_price = 200 → marked_price = 300 → desired_profit_rate = 0.2 →
  (∃ discount_rate : ℝ, discount_rate = 0.8 ∧ marked_price * discount_rate = cost_price * (1 + desired_profit_rate)) :=
by
  intros
  sorry

end product_discount_rate_l141_141482


namespace max_area_of_equilateral_triangle_in_rectangle_l141_141566

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  if h : a ≤ b then
    (a^2 * Real.sqrt 3) / 4
  else
    (b^2 * Real.sqrt 3) / 4

theorem max_area_of_equilateral_triangle_in_rectangle :
  maxEquilateralTriangleArea 12 14 = 36 * Real.sqrt 3 :=
by
  sorry

end max_area_of_equilateral_triangle_in_rectangle_l141_141566


namespace monomials_like_terms_l141_141105

theorem monomials_like_terms (a b : ℝ) (m n : ℤ) 
  (h1 : 2 * (a^4) * (b^(-2 * m + 7)) = 3 * (a^(2 * m)) * (b^(n + 2))) :
  m + n = 3 := 
by {
  -- Our proof will be placed here
  sorry
}

end monomials_like_terms_l141_141105


namespace sum_of_their_ages_now_l141_141000

variable (Nacho Divya : ℕ)

-- Conditions
def divya_current_age := 5
def nacho_in_5_years := 3 * (divya_current_age + 5)

-- Definition to determine current age of Nacho
def nacho_current_age := nacho_in_5_years - 5

-- Sum of current ages
def sum_of_ages := divya_current_age + nacho_current_age

-- Theorem to prove the sum of their ages now is 30
theorem sum_of_their_ages_now : sum_of_ages = 30 :=
by
  sorry

end sum_of_their_ages_now_l141_141000


namespace find_intersection_complement_find_value_m_l141_141999

-- (1) Problem Statement
theorem find_intersection_complement (A : Set ℝ) (B : Set ℝ) (x : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - 3 < 0}) →
  (x ∈ A ∩ (Bᶜ : Set ℝ)) ↔ (x = -1 ∨ 3 ≤ x ∧ x ≤ 5) :=
by
  sorry

-- (2) Problem Statement
theorem find_value_m (A B : Set ℝ) (m : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - m < 0}) →
  (A ∩ B = {x | -1 ≤ x ∧ x < 4}) →
  m = 8 :=
by
  sorry

end find_intersection_complement_find_value_m_l141_141999


namespace sequence_first_last_four_equal_l141_141773

theorem sequence_first_last_four_equal (S : List ℕ) (n : ℕ)
  (hS : S.length = n)
  (h_max : ∀ T : List ℕ, (∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                        (S.drop i).take 5 ≠ (S.drop j).take 5) → T.length ≤ n)
  (h_distinct : ∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                (S.drop i).take 5 ≠ (S.drop j).take 5) :
  (S.take 4 = S.drop (n-4)) :=
by
  sorry

end sequence_first_last_four_equal_l141_141773


namespace combined_time_in_pool_l141_141338

theorem combined_time_in_pool : 
    ∀ (Jerry_time Elaine_time George_time Kramer_time : ℕ), 
    Jerry_time = 3 →
    Elaine_time = 2 * Jerry_time →
    George_time = Elaine_time / 3 →
    Kramer_time = 0 →
    Jerry_time + Elaine_time + George_time + Kramer_time = 11 :=
by 
  intros Jerry_time Elaine_time George_time Kramer_time hJerry hElaine hGeorge hKramer
  sorry

end combined_time_in_pool_l141_141338


namespace probability_floor_log_10_eq_approx_l141_141143

noncomputable def probability_floor_log_eq : ℝ :=
  probability ((uniform (0, 1)) × (uniform (0, 1))) 
  {xy | ∃ n : ℤ, (xy.1 ∈ set.Ico (10^(n-1)) (10^n))
                ∧ (xy.2 ∈ set.Ico (10^(n-1)) (10^n))}

theorem probability_floor_log_10_eq_approx :
  probability_floor_log_eq ≈ 0.81818 := 
sorry

end probability_floor_log_10_eq_approx_l141_141143


namespace factorize_x_squared_sub_xy_l141_141085

theorem factorize_x_squared_sub_xy (x y : ℝ) : x^2 - x * y = x * (x - y) :=
sorry

end factorize_x_squared_sub_xy_l141_141085


namespace max_value_trig_expression_l141_141356

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end max_value_trig_expression_l141_141356


namespace frozen_yogurt_combinations_l141_141489

theorem frozen_yogurt_combinations :
  (5 * (nat.choose 7 3) = 175) :=
by
  have x := nat.choose 7 3,
  calc
    5 * x = 5 * 35 : by rw nat.choose_eq_factorial_div_factorial
    ...    = 175   : by norm_num

end frozen_yogurt_combinations_l141_141489


namespace find_M_l141_141718

theorem find_M : ∃ M : ℕ, M > 0 ∧ 18 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2 ∧ M = 54 := by
  use 54
  sorry

end find_M_l141_141718


namespace abs_value_x_minus_2_plus_x_plus_3_ge_4_l141_141294

theorem abs_value_x_minus_2_plus_x_plus_3_ge_4 :
  ∀ x : ℝ, (|x - 2| + |x + 3| ≥ 4) ↔ (x ≤ - (5 / 2)) := 
sorry

end abs_value_x_minus_2_plus_x_plus_3_ge_4_l141_141294


namespace value_of_a_minus_b_l141_141551

theorem value_of_a_minus_b (a b : ℝ) :
  (∀ x, - (1 / 2 : ℝ) < x ∧ x < (1 / 3 : ℝ) → ax^2 + bx + 2 > 0) → a - b = -10 := by
sorry

end value_of_a_minus_b_l141_141551


namespace remainder_when_160_divided_by_k_l141_141094

-- Define k to be a positive integer
def positive_integer (n : ℕ) := n > 0

-- Given conditions in the problem
def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def problem_condition (k : ℕ) := positive_integer k ∧ (120 % (k * k) = 12)

-- Prove the main statement
theorem remainder_when_160_divided_by_k (k : ℕ) (h : problem_condition k) : 160 % k = 4 := 
sorry  -- Proof here

end remainder_when_160_divided_by_k_l141_141094


namespace john_unanswered_problems_is_9_l141_141009

variables (x y z : ℕ)

theorem john_unanswered_problems_is_9 (h1 : 5 * x + 2 * z = 93)
                                      (h2 : 4 * x - y = 54)
                                      (h3 : x + y + z = 30) : 
  z = 9 :=
by 
  sorry

end john_unanswered_problems_is_9_l141_141009


namespace find_sales_tax_percentage_l141_141064

noncomputable def salesTaxPercentage (price_with_tax : ℝ) (price_difference : ℝ) : ℝ :=
  (price_difference * 100) / (price_with_tax - price_difference)

theorem find_sales_tax_percentage :
  salesTaxPercentage 2468 161.46 = 7 := by
  sorry

end find_sales_tax_percentage_l141_141064


namespace ax_product_zero_l141_141288

theorem ax_product_zero 
  {a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ} 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) *
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) *
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := 
sorry

end ax_product_zero_l141_141288


namespace solve_quadratic_l141_141164

theorem solve_quadratic (x : ℝ) : (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3) :=
by
  sorry

end solve_quadratic_l141_141164


namespace simplify_and_evaluate_l141_141795

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) :
  ( ((a + 7) / (a - 1) - 2 / (a + 1)) / ((a^2 + 3 * a) / (a^2 - 1)) = -1/2 ) :=
by
  sorry

end simplify_and_evaluate_l141_141795


namespace solution_set_l141_141866

/-- Definition: integer solutions (a, b, c) with c ≤ 94 that satisfy the equation -/
def int_solutions (a b c : ℤ) : Prop :=
  c ≤ 94 ∧ (a + Real.sqrt c)^2 + (b + Real.sqrt c)^2 = 60 + 20 * Real.sqrt c

/-- Proposition: The integer solutions (a, b, c) that satisfy the equation are exactly these -/
theorem solution_set :
  { (a, b, c) : ℤ × ℤ × ℤ  | int_solutions a b c } =
  { (3, 7, 41), (4, 6, 44), (5, 5, 45), (6, 4, 44), (7, 3, 41) } :=
by
  sorry

end solution_set_l141_141866


namespace clock_angle_l141_141113

theorem clock_angle :
  ∃ (M : ℚ), 
    (M = 21 + 9 / 11 ∨ M ≈ 21.82) ∧ 
    let angle_4_00 := 120 in 
    ∃ (M1 : ℚ), 120 - 5.5 * M1 = 120 ∧ 
                4 * 30 + 0.5 * M - 6 * M = angle_4_00 ∧ 
                60 - M = M1 :=
sorry

end clock_angle_l141_141113


namespace value_in_box_l141_141660

theorem value_in_box (x : ℤ) (h : 5 + x = 10 + 20) : x = 25 := by
  sorry

end value_in_box_l141_141660


namespace proof_problem_l141_141708

noncomputable def f (x : ℝ) : ℝ :=
  Real.log ((1 + Real.sqrt x) / (1 - Real.sqrt x))

theorem proof_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  f ( (5 * x + 2 * x^2) / (1 + 5 * x + 3 * x^2) ) = Real.sqrt 5 * f x :=
by
  sorry

end proof_problem_l141_141708


namespace geometric_sequence_fourth_term_l141_141717

theorem geometric_sequence_fourth_term (a₁ a₂ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 1/3) :
    ∃ a₄ : ℝ, a₄ = 1/243 :=
sorry

end geometric_sequence_fourth_term_l141_141717


namespace angle_of_skew_lines_in_range_l141_141442

noncomputable def angle_between_skew_lines (θ : ℝ) (θ_range : 0 < θ ∧ θ ≤ 90) : Prop :=
  θ ∈ (Set.Ioc 0 90)

-- We assume the existence of such an angle θ formed by two skew lines
theorem angle_of_skew_lines_in_range (θ : ℝ) (h_skew : true) : angle_between_skew_lines θ (⟨sorry, sorry⟩) :=
  sorry

end angle_of_skew_lines_in_range_l141_141442


namespace smallest_number_divisible_l141_141831

theorem smallest_number_divisible (n : ℕ) 
    (h1 : (n - 20) % 15 = 0) 
    (h2 : (n - 20) % 30 = 0)
    (h3 : (n - 20) % 45 = 0)
    (h4 : (n - 20) % 60 = 0) : 
    n = 200 :=
sorry

end smallest_number_divisible_l141_141831


namespace Michael_pizza_fraction_l141_141606

theorem Michael_pizza_fraction (T : ℚ) (L : ℚ) (total : ℚ) (M : ℚ) 
  (hT : T = 1 / 2) (hL : L = 1 / 6) (htotal : total = 1) (hM : total - (T + L) = M) :
  M = 1 / 3 := 
sorry

end Michael_pizza_fraction_l141_141606


namespace length_of_train_l141_141329

theorem length_of_train (V L : ℝ) (h1 : L = V * 18) (h2 : L + 250 = V * 33) : L = 300 :=
by
  sorry

end length_of_train_l141_141329


namespace arithmetic_mean_of_fractions_l141_141633

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l141_141633


namespace range_of_a_l141_141876

open Real

noncomputable def f (x a : ℝ) : ℝ := (exp x / 2) - (a / exp x)

def condition (x₁ x₂ a : ℝ) : Prop :=
  x₁ ≠ x₂ ∧ 1 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1 ≤ x₂ ∧ x₂ ≤ 2 ∧ ((abs (f x₁ a) - abs (f x₂ a)) * (x₁ - x₂) > 0)

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), condition x₁ x₂ a) ↔ (- (exp 2) / 2 ≤ a ∧ a ≤ (exp 2) / 2) :=
by
  sorry

end range_of_a_l141_141876


namespace combined_books_total_l141_141420

def keith_books : ℕ := 20
def jason_books : ℕ := 21
def amanda_books : ℕ := 15
def sophie_books : ℕ := 30

def total_books := keith_books + jason_books + amanda_books + sophie_books

theorem combined_books_total : total_books = 86 := 
by sorry

end combined_books_total_l141_141420


namespace geometric_sequence_a3_l141_141238

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3 
  (a : ℕ → ℝ) (h1 : a 1 = -2) (h5 : a 5 = -8)
  (h : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 3 = -4 :=
sorry

end geometric_sequence_a3_l141_141238


namespace find_A_find_b_and_c_l141_141559

open Real

variable {a b c A B C : ℝ}

-- Conditions for the problem
axiom triangle_sides : ∀ {A B C : ℝ}, a > 0
axiom sine_law_condition : b * sin B + c * sin C - sqrt 2 * b * sin C = a * sin A
axiom degrees_60 : B = π / 3
axiom side_a : a = 2

theorem find_A : A = π / 4 :=
by sorry

theorem find_b_and_c (h : A = π / 4) (hB : B = π / 3) (ha : a = 2) : b = sqrt 6 ∧ c = 1 + sqrt 3 :=
by sorry

end find_A_find_b_and_c_l141_141559


namespace problem_solution_l141_141987

def tens_digit_is_odd (n : ℕ) : Bool :=
  let m := (n * n + n) / 10 % 10
  m % 2 = 1

def count_tens_digit_odd : ℕ :=
  List.range 50 |>.filter tens_digit_is_odd |>.length

theorem problem_solution : count_tens_digit_odd = 25 :=
  sorry

end problem_solution_l141_141987


namespace total_spent_is_195_l141_141857

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def boots_discounted_cost : ℝ := boots_original_cost * (1 - boots_discount)
def total_cost : ℝ := hoodie_cost + flashlight_cost + boots_discounted_cost

theorem total_spent_is_195 : total_cost = 195 := by
  sorry

end total_spent_is_195_l141_141857


namespace abc_relationship_l141_141912

variable (x y : ℝ)

def parabola (x : ℝ) : ℝ :=
  x^2 + x + 2

def a := parabola 2
def b := parabola (-1)
def c := parabola 3

theorem abc_relationship : c > a ∧ a > b := by
  sorry

end abc_relationship_l141_141912


namespace find_sum_x1_x2_l141_141998

-- Define sets A and B with given properties
def set_A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
def set_B (x1 x2 : ℝ) : Set ℝ := {x | x1 ≤ x ∧ x ≤ x2}

-- Conditions of union and intersection
def union_condition (x1 x2 : ℝ) : Prop := set_A ∪ set_B x1 x2 = {x | x > -2}
def intersection_condition (x1 x2 : ℝ) : Prop := set_A ∩ set_B x1 x2 = {x | 1 < x ∧ x ≤ 3}

-- Main theorem to prove
theorem find_sum_x1_x2 (x1 x2 : ℝ) (h_union : union_condition x1 x2) (h_intersect : intersection_condition x1 x2) :
  x1 + x2 = 2 :=
sorry

end find_sum_x1_x2_l141_141998


namespace complex_subtraction_l141_141372

def z1 : ℂ := 3 + (1 : ℂ)
def z2 : ℂ := 2 - (1 : ℂ)

theorem complex_subtraction : z1 - z2 = 1 + 2 * (1 : ℂ) :=
by
  sorry

end complex_subtraction_l141_141372


namespace min_buses_needed_l141_141495

theorem min_buses_needed (n : ℕ) : 325 / 45 ≤ n ∧ n < 325 / 45 + 1 ↔ n = 8 :=
by
  sorry

end min_buses_needed_l141_141495


namespace find_mistaken_number_l141_141563

theorem find_mistaken_number : 
  ∃! x : ℕ, (x ∈ {n : ℕ | n ≥ 10 ∧ n < 100 ∧ (n % 10 = 5 ∨ n % 10 = 0)} ∧ 
  (10 + 15 + 20 + 25 + 30 + 35 + 40 + 45 + 50 + 55 + 60 + 65 + 70 + 75 + 80 + 85 + 90 + 95) + 2 * x = 1035) :=
sorry

end find_mistaken_number_l141_141563


namespace value_of_adams_collection_l141_141187

theorem value_of_adams_collection (num_coins : ℕ) (coins_value : ℕ) (total_value_4coins : ℕ) (h1 : num_coins = 20) (h2 : total_value_4coins = 16) (h3 : ∀ k, k = 4 → coins_value = total_value_4coins / k) : 
  num_coins * coins_value = 80 := 
by {
  sorry
}

end value_of_adams_collection_l141_141187


namespace trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l141_141366

open Real

theorem trig_cos2_minus_sin2_eq_neg_sqrt5_div3 (α : ℝ) (hα1 : 0 < α ∧ α < π) (hα2 : sin α + cos α = sqrt 3 / 3) :
  cos α ^ 2 - sin α ^ 2 = - sqrt 5 / 3 := 
  sorry

end trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l141_141366


namespace combined_length_of_legs_is_ten_l141_141292

-- Define the conditions given in the problem.
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * Real.sqrt 2

def hypotenuse_length (c : ℝ) : Prop :=
  c = 7.0710678118654755

def perimeter_condition (a b c perimeter : ℝ) : Prop :=
  perimeter = a + b + c ∧ perimeter = 10 + c

-- Prove the combined length of the two legs is 10.
theorem combined_length_of_legs_is_ten :
  ∃ (a b c : ℝ), is_isosceles_right_triangle a b c →
  hypotenuse_length c →
  ∀ perimeter : ℝ, perimeter_condition a b c perimeter →
  2 * a = 10 :=
by
  sorry

end combined_length_of_legs_is_ten_l141_141292


namespace min_cos_C_l141_141538

theorem min_cos_C (a b c : ℝ) (A B C : ℝ) (h1 : a^2 + b^2 = (5 / 2) * c^2) 
  (h2 : ∃ (A B C : ℝ), a ≠ b ∧ 
    c = (a ^ 2 + b ^ 2 - 2 * a * b * (Real.cos C))) : 
  ∃ (C : ℝ), Real.cos C = 3 / 5 :=
by
  sorry

end min_cos_C_l141_141538


namespace solve_fraction_inequality_l141_141154

theorem solve_fraction_inequality :
  { x : ℝ | x / (x + 5) ≥ 0 } = { x : ℝ | x < -5 } ∪ { x : ℝ | x ≥ 0 } := by
  sorry

end solve_fraction_inequality_l141_141154


namespace find_number_l141_141922

-- Definitions based on conditions
def condition (x : ℝ) : Prop := (x - 5) / 3 = 4

-- The target theorem to prove
theorem find_number (x : ℝ) (h : condition x) : x = 17 :=
sorry

end find_number_l141_141922


namespace parabola_has_one_x_intercept_l141_141384

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end parabola_has_one_x_intercept_l141_141384


namespace compound_interest_rate_l141_141073

theorem compound_interest_rate :
  ∀ (A P : ℝ) (t : ℕ),
  A = 4840.000000000001 ->
  P = 4000 ->
  t = 2 ->
  A = P * (1 + 0.1)^t :=
by
  intros A P t hA hP ht
  rw [hA, hP, ht]
  norm_num
  sorry

end compound_interest_rate_l141_141073


namespace nesting_doll_height_l141_141139

variable (H₀ : ℝ) (n : ℕ)

theorem nesting_doll_height (H₀ : ℝ) (Hₙ : ℝ) (H₁ : H₀ = 243) (H₂ : ∀ n : ℕ, Hₙ = H₀ * (2 / 3) ^ n) (H₃ : Hₙ = 32) : n = 4 :=
by
  sorry

end nesting_doll_height_l141_141139


namespace geometric_progression_solution_l141_141596

theorem geometric_progression_solution 
  (b₁ q : ℝ)
  (h₁ : b₁^3 * q^3 = 1728)
  (h₂ : b₁ * (1 + q + q^2) = 63) :
  (b₁ = 3 ∧ q = 4) ∨ (b₁ = 48 ∧ q = 1/4) :=
  sorry

end geometric_progression_solution_l141_141596


namespace solve_for_x_l141_141944

theorem solve_for_x (x : ℝ) (h : 1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l141_141944


namespace find_y_l141_141715

theorem find_y (y : ℝ) : 
  2 ≤ y / (3 * y - 4) ∧ y / (3 * y - 4) < 5 ↔ y ∈ Set.Ioc (10 / 7) (8 / 5) := 
sorry

end find_y_l141_141715


namespace Jane_remaining_time_l141_141664

noncomputable def JaneRate : ℚ := 1 / 4
noncomputable def RoyRate : ℚ := 1 / 5
noncomputable def workingTime : ℚ := 2
noncomputable def cakeFractionCompletedTogether : ℚ := (JaneRate + RoyRate) * workingTime
noncomputable def remainingCakeFraction : ℚ := 1 - cakeFractionCompletedTogether
noncomputable def timeForJaneToCompleteRemainingCake : ℚ := remainingCakeFraction / JaneRate

theorem Jane_remaining_time :
  timeForJaneToCompleteRemainingCake = 2 / 5 :=
by
  sorry

end Jane_remaining_time_l141_141664


namespace triangle_area_l141_141330

/-- Define the area of a triangle with one side of length 13, an opposite angle of 60 degrees, and side ratio 4:3. -/
theorem triangle_area (a b c : ℝ) (A : ℝ) (S : ℝ) 
  (h_a : a = 13)
  (h_A : A = Real.pi / 3)
  (h_bc_ratio : b / c = 4 / 3)
  (h_cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h_area : S = 1 / 2 * b * c * Real.sin A) :
  S = 39 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l141_141330


namespace arithmetic_mean_of_fractions_l141_141618

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l141_141618


namespace sum_of_coefficients_l141_141284

theorem sum_of_coefficients 
  (A B C : ℤ)
  (h : ∀ x : ℂ, x ∈ {(-1 : ℂ), 3, 4} → x^3 + (A : ℂ) * x^2 + (B : ℂ) * x + (C : ℂ) = 0) : 
  A + B + C = 11 := 
sorry

end sum_of_coefficients_l141_141284


namespace arithmetic_mean_eq_l141_141652

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l141_141652


namespace arithmetic_mean_of_fractions_l141_141644

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l141_141644


namespace frame_width_proof_l141_141803

noncomputable section

-- Define the given conditions
def perimeter_square_opening := 60 -- cm
def perimeter_entire_frame := 180 -- cm

-- Define what we need to prove: the width of the frame
def width_of_frame : ℕ := 5 -- cm

-- Define a function to calculate the side length of a square
def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

-- Define the side length of the square opening
def side_length_opening := side_length_of_square perimeter_square_opening

-- Use the given conditions to calculate the frame's width
-- Given formulas in the solution steps:
--  2 * (3 * side_length + 4 * d) + 2 * (side_length + 2 * d) = perimeter_entire_frame
theorem frame_width_proof (d : ℕ) (perim_square perim_frame : ℕ) :
  perim_square = perimeter_square_opening →
  perim_frame = perimeter_entire_frame →
  2 * (3 * side_length_of_square perim_square + 4 * d) 
  + 2 * (side_length_of_square perim_square + 2 * d) 
  = perim_frame →
  d = width_of_frame := 
by 
  intros h1 h2 h3
  -- The proof will go here
  sorry

end frame_width_proof_l141_141803


namespace triangle_TSR_area_l141_141915

noncomputable def TriangleGeometry :=
  let PQR : ∀ (P Q R : ℝ × ℝ), Triangle P Q R
  let PQ := 3
  let QR := 4
  let PR := 5
  let S := midpoint PQ
  let PT := 3
  let T := PQR.point_on_PQ PQ
  let R := midpoint TX
  /- Prove that the area of triangle TSR is 1.125 given the conditions -/
  theorem triangle_TSR_area :
    ∀ (P Q R S T: ℝ × ℝ),
      is_right_triangle PQR →
      let TSR := mkTriangle T S R
      ⁇
      expect_area TSR = 1.125 :=
  sorry

end triangle_TSR_area_l141_141915


namespace probability_of_first_good_product_on_third_try_l141_141190

-- Define the problem parameters
def pass_rate : ℚ := 3 / 4
def failure_rate : ℚ := 1 / 4
def epsilon := 3

-- The target probability statement
theorem probability_of_first_good_product_on_third_try :
  (failure_rate * failure_rate * pass_rate) = ((1 / 4) ^ 2 * (3 / 4)) :=
by
  sorry

end probability_of_first_good_product_on_third_try_l141_141190


namespace smallest_solution_l141_141211

theorem smallest_solution (x : ℝ) (h : x * |x| = 2 * x + 1) : x = -1 := 
by
  sorry

end smallest_solution_l141_141211


namespace total_money_made_l141_141687

def dvd_price : ℕ := 240
def dvd_quantity : ℕ := 8
def washing_machine_price : ℕ := 898

theorem total_money_made : dvd_price * dvd_quantity + washing_machine_price = 240 * 8 + 898 :=
by
  sorry

end total_money_made_l141_141687


namespace problem_statement_l141_141269

theorem problem_statement (x : ℝ) (h : x^3 - 3 * x = 7) : x^7 + 27 * x^2 = 76 * x^2 + 270 * x + 483 :=
sorry

end problem_statement_l141_141269


namespace inverse_proportion_function_range_m_l141_141380

theorem inverse_proportion_function_range_m
  (x1 x2 y1 y2 m : ℝ)
  (h_func_A : y1 = (5 * m - 2) / x1)
  (h_func_B : y2 = (5 * m - 2) / x2)
  (h_x : x1 < x2)
  (h_x_neg : x2 < 0)
  (h_y : y1 < y2) :
  m < 2 / 5 :=
sorry

end inverse_proportion_function_range_m_l141_141380


namespace arithmetic_mean_of_fractions_l141_141622

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l141_141622


namespace dropped_student_score_l141_141280

theorem dropped_student_score (total_students : ℕ) (remaining_students : ℕ) (initial_average : ℝ) (new_average : ℝ) (x : ℝ) 
  (h1 : total_students = 16) 
  (h2 : remaining_students = 15) 
  (h3 : initial_average = 62.5) 
  (h4 : new_average = 63.0) 
  (h5 : total_students * initial_average - remaining_students * new_average = x) : 
  x = 55 := 
sorry

end dropped_student_score_l141_141280


namespace midpoint_trajectory_l141_141234

theorem midpoint_trajectory (x y : ℝ) (h : ∃ (xₚ yₚ : ℝ), yₚ = 2 * xₚ^2 + 1 ∧ y = 4 * (xₚ / 2) ^ 2) : y = 4 * x ^ 2 :=
sorry

end midpoint_trajectory_l141_141234


namespace sum_eq_24_of_greatest_power_l141_141860

theorem sum_eq_24_of_greatest_power (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_b_gt_1 : b > 1) (h_a_pow_b_lt_500 : a^b < 500)
  (h_greatest : ∀ (x y : ℕ), (0 < x) → (0 < y) → (y > 1) → (x^y < 500) → (x^y ≤ a^b)) : a + b = 24 :=
  sorry

end sum_eq_24_of_greatest_power_l141_141860


namespace man_speed_with_stream_l141_141325

-- Define the man's rate in still water
def man_rate_in_still_water : ℝ := 6

-- Define the man's rate against the stream
def man_rate_against_stream (stream_speed : ℝ) : ℝ :=
  man_rate_in_still_water - stream_speed

-- The given condition that the man's rate against the stream is 10 km/h
def man_rate_against_condition : Prop := ∃ (stream_speed : ℝ), man_rate_against_stream stream_speed = 10

-- We aim to prove that the man's speed with the stream is 10 km/h
theorem man_speed_with_stream (stream_speed : ℝ) (h : man_rate_against_stream stream_speed = 10) :
  man_rate_in_still_water + stream_speed = 10 := by
  sorry

end man_speed_with_stream_l141_141325


namespace problem_statement_l141_141132

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 1988 a b α β = 3) : f 2013 a b α β = 5 :=
by 
  sorry

end problem_statement_l141_141132


namespace hexagon_monochromatic_triangles_l141_141681

theorem hexagon_monochromatic_triangles :
  let hexagon_edges := 15 -- $\binom{6}{2}$
  let monochromatic_tri_prob := (1 / 3) -- Prob of one triangle being monochromatic
  let combinations := 20 -- $\binom{6}{3}$, total number of triangles in K_6
  let exactly_two_monochromatic := (combinations.choose 2) * (monochromatic_tri_prob ^ 2) * ((2 / 3) ^ 18)
  (exactly_two_monochromatic = 49807360 / 3486784401) := sorry

end hexagon_monochromatic_triangles_l141_141681


namespace existence_of_same_remainder_mod_36_l141_141130

theorem existence_of_same_remainder_mod_36
  (a : Fin 7 → ℕ) :
  ∃ (i j k l : Fin 7), i < j ∧ k < l ∧ (a i)^2 + (a j)^2 % 36 = (a k)^2 + (a l)^2 % 36 := by
  sorry

end existence_of_same_remainder_mod_36_l141_141130


namespace sin_alpha_l141_141736

variable (α : Real)
variable (hcos : Real.cos α = 3 / 5)
variable (htan : Real.tan α < 0)

theorem sin_alpha (α : Real) (hcos : Real.cos α = 3 / 5) (htan : Real.tan α < 0) :
  Real.sin α = -4 / 5 :=
sorry

end sin_alpha_l141_141736


namespace integer_solutions_l141_141199

theorem integer_solutions (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intros h
  sorry

end integer_solutions_l141_141199


namespace parabola_directrix_l141_141283

theorem parabola_directrix (y : ℝ) : 
  x = -((1:ℝ)/4)*y^2 → x = 1 :=
by 
  sorry

end parabola_directrix_l141_141283


namespace arithmetic_mean_eq_l141_141655

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l141_141655


namespace probability_both_meat_given_same_l141_141018

open ProbabilityTheory

-- Definition of the problem conditions
def total_dumplings : Finset (Fin 5) := Finset.univ
def meat_dumplings : Finset (Fin 5) := {0, 1} -- using the first 2 as meat filled
def red_bean_paste_dumplings : Finset (Fin 5) := {2, 3, 4} -- the remaining are red bean filled

def event_same_filling (x y : Fin 5) : Prop :=
  (x ∈ meat_dumplings ∧ y ∈ meat_dumplings) ∨ (x ∈ red_bean_paste_dumplings ∧ y ∈ red_bean_paste_dumplings)

def event_both_meat (x y : Fin 5) : Prop :=
  x ∈ meat_dumplings ∧ y ∈ meat_dumplings

-- Probability calculations
noncomputable def probability_same_filling : ℚ :=
  (Finset.card ((total_dumplings.product total_dumplings).filter (λ p, event_same_filling p.1 p.2))).toRat /
  (Finset.card (total_dumplings.product total_dumplings)).toRat

noncomputable def probability_both_meat : ℚ :=
  (Finset.card ((total_dumplings.product total_dumplings).filter (λ p, event_both_meat p.1 p.2))).toRat /
  (Finset.card (total_dumplings.product total_dumplings)).toRat

noncomputable def conditional_probability_both_meat_given_same_filling : ℚ :=
  probability_both_meat / probability_same_filling

-- The main theorem statement
theorem probability_both_meat_given_same : 
  conditional_probability_both_meat_given_same_filling = 1 / 4 :=
by
  sorry

end probability_both_meat_given_same_l141_141018


namespace bad_carrots_l141_141467

-- Conditions
def carrots_picked_by_vanessa := 17
def carrots_picked_by_mom := 14
def good_carrots := 24
def total_carrots := carrots_picked_by_vanessa + carrots_picked_by_mom

-- Question and Proof
theorem bad_carrots :
  total_carrots - good_carrots = 7 :=
by
  -- Placeholder for proof
  sorry

end bad_carrots_l141_141467


namespace sequence_formula_and_sum_l141_141216

def arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) :=
  ∀ m n k, m < n → n < k → a n^2 = a m * a k

def Sn (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sequence_formula_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 4 * n - 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ (∀ n, S n = (n * (4 * n)) / 2) → ∃ n > 0, S n > 60 * n + 800 ∧ n = 41) ∧
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ (∀ n, S n = 2 * n) → ∀ n > 0, ¬ (S n > 60 * n + 800)) :=
by sorry

end sequence_formula_and_sum_l141_141216


namespace days_at_grandparents_l141_141011

theorem days_at_grandparents
  (total_vacation_days : ℕ)
  (travel_to_gp : ℕ)
  (travel_to_brother : ℕ)
  (days_at_brother : ℕ)
  (travel_to_sister : ℕ)
  (days_at_sister : ℕ)
  (travel_home : ℕ)
  (total_days : total_vacation_days = 21) :
  total_vacation_days - (travel_to_gp + travel_to_brother + days_at_brother + travel_to_sister + days_at_sister + travel_home) = 5 :=
by
  sorry -- proof to be constructed

end days_at_grandparents_l141_141011


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l141_141612

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l141_141612


namespace no_integer_roots_l141_141398

theorem no_integer_roots (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) : ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end no_integer_roots_l141_141398


namespace complement_is_empty_l141_141097

def U : Set ℕ := {1, 3}
def A : Set ℕ := {1, 3}

theorem complement_is_empty : (U \ A) = ∅ := 
by 
  sorry

end complement_is_empty_l141_141097


namespace perpendicular_lines_solve_b_l141_141867

theorem perpendicular_lines_solve_b (b : ℝ) : (∀ x y : ℝ, y = 3 * x + 7 →
                                                    ∃ y1 : ℝ, y1 = ( - b / 4 ) * x + 3 ∧
                                                               3 * ( - b / 4 ) = -1) → 
                                               b = 4 / 3 :=
by
  sorry

end perpendicular_lines_solve_b_l141_141867


namespace algorithm_output_is_127_l141_141754
-- Import the entire Mathlib library

-- Define the possible values the algorithm can output
def possible_values : List ℕ := [15, 31, 63, 127]

-- Define the property where the value is of the form 2^n - 1
def is_exp2_minus_1 (x : ℕ) := ∃ n : ℕ, x = 2^n - 1

-- Define the main theorem to prove the algorithm's output is 127
theorem algorithm_output_is_127 : (∀ x ∈ possible_values, is_exp2_minus_1 x) →
                                      ∃ n : ℕ, 127 = 2^n - 1 :=
by
  -- Define the conditions and the proof steps are left out
  sorry

end algorithm_output_is_127_l141_141754


namespace x_intercepts_of_parabola_l141_141389

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end x_intercepts_of_parabola_l141_141389


namespace mean_of_remaining_four_numbers_l141_141925

theorem mean_of_remaining_four_numbers (a b c d : ℝ) 
  (h_mean_five : (a + b + c + d + 120) / 5 = 100) : 
  (a + b + c + d) / 4 = 95 :=
by
  sorry

end mean_of_remaining_four_numbers_l141_141925


namespace boat_trip_l141_141934

variable {v v_T : ℝ}

theorem boat_trip (d_total t_total : ℝ) (h1 : d_total = 10) (h2 : t_total = 5) (h3 : 2 / (v - v_T) = 3 / (v + v_T)) :
  v_T = 5 / 12 ∧ (5 / (v - v_T)) = 3 ∧ (5 / (v + v_T)) = 2 :=
by
  have h4 : 1 / (d_total / t_total) = v - v_T := sorry
  have h5 : 1 / (d_total / t_total) = v + v_T := sorry
  have h6 : v = 5 * v_T := sorry
  have h7 : v_T = 5 / 12 := sorry
  have t_upstream : 5 / (v - v_T) = 3 := sorry
  have t_downstream : 5 / (v + v_T) = 2 := sorry
  exact ⟨h7, t_upstream, t_downstream⟩

end boat_trip_l141_141934


namespace arithmetic_mean_of_fractions_l141_141632

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l141_141632


namespace C_plus_D_l141_141266

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : 
  C + D = -10 := by
  sorry

end C_plus_D_l141_141266


namespace man_speed_upstream_l141_141952

def man_speed_still_water : ℕ := 50
def speed_downstream : ℕ := 80

theorem man_speed_upstream : (man_speed_still_water - (speed_downstream - man_speed_still_water)) = 20 :=
by
  sorry

end man_speed_upstream_l141_141952


namespace spinner_prime_probability_l141_141196

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def spinner_sections : List ℕ := [2, 3, 5, 7, 4, 6, 8, 9]

def no_of_prime_numbers : ℕ := (spinner_sections.filter is_prime).length

def probability_prime : ℚ := (no_of_prime_numbers : ℚ) / (spinner_sections.length : ℚ)

theorem spinner_prime_probability : probability_prime = 1 / 2 := by
  rw [probability_prime, no_of_prime_numbers]
  simp only [spinner_sections, List.filter_eq_self.mpr, List.length]
  norm_num
  sorry

end spinner_prime_probability_l141_141196


namespace parity_of_f_find_a_l141_141535

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x + a * Real.exp (-x)

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a ↔ a = 1 ∨ a = -1) ∧
  (∀ x : ℝ, f (-x) a = -f x a ↔ a = -1) ∧
  (∀ x : ℝ, ¬(f (-x) a = f x a) ∧ ¬(f (-x) a = -f x a) ↔ ¬(a = 1 ∨ a = -1)) :=
by
  sorry

theorem find_a (h : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x a ≥ f 0 a) : 
  a = 1 :=
by
  sorry

end parity_of_f_find_a_l141_141535


namespace wrapping_cube_wrapping_prism_a_wrapping_prism_b_l141_141936

theorem wrapping_cube (ways_cube : ℕ) :
  ways_cube = 3 :=
  sorry

theorem wrapping_prism_a (ways_prism_a : ℕ) (a : ℝ) :
  (ways_prism_a = 5) ↔ (a > 0) :=
  sorry

theorem wrapping_prism_b (ways_prism_b : ℕ) (b : ℝ) :
  (ways_prism_b = 7) ↔ (b > 0) :=
  sorry

end wrapping_cube_wrapping_prism_a_wrapping_prism_b_l141_141936


namespace min_value_y_l141_141171

theorem min_value_y (x : ℝ) : ∃ (x : ℝ), y = x^2 + 16 * x + 20 → y ≥ -44 :=
begin
  sorry
end

end min_value_y_l141_141171


namespace mary_travel_time_l141_141429

noncomputable def ambulance_speed : ℝ := 60
noncomputable def don_speed : ℝ := 30
noncomputable def don_time : ℝ := 0.5

theorem mary_travel_time : (don_speed * don_time) / ambulance_speed * 60 = 15 := by
  sorry

end mary_travel_time_l141_141429


namespace remainder_when_divided_by_22_l141_141939

theorem remainder_when_divided_by_22 (n : ℤ) (h : (2 * n) % 11 = 2) : n % 22 = 1 :=
by
  sorry

end remainder_when_divided_by_22_l141_141939


namespace problem_part1_problem_part2_area_height_l141_141099

theorem problem_part1 (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) : 
  x * y ^ 2 - x ^ 2 * y = -32 * Real.sqrt 2 := 
  sorry

theorem problem_part2_area_height (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) :
  let side_length := Real.sqrt 12
  let area := (1 / 2) * x * y
  let height := area / side_length
  area = 4 ∧ height = (2 * Real.sqrt 3) / 3 := 
  sorry

end problem_part1_problem_part2_area_height_l141_141099


namespace arithmetic_mean_of_fractions_l141_141637

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l141_141637


namespace sum_of_roots_of_equation_l141_141702

theorem sum_of_roots_of_equation :
  let f := (3 * (x: ℝ) + 4) * (x - 5) + (3 * x + 4) * (x - 7)
  ∀ x : ℝ, f = 0 → ∑ (roots : ℝ) in {x | f x = 0}, x = -2 :=
by {
  let f := (3 * (x: ℝ) + 4) * (x - 5) + (3 * x + 4) * (x - 7),
  sorry
}

end sum_of_roots_of_equation_l141_141702


namespace scientific_notation_conversion_l141_141336

theorem scientific_notation_conversion :
  (6.1 * 10^9 = (6.1 : ℝ) * 10^8) :=
sorry

end scientific_notation_conversion_l141_141336


namespace intersection_complement_l141_141541

-- Declare variables for sets
variable (I A B : Set ℤ)

-- Define the universal set I
def universal_set : Set ℤ := { x | -3 < x ∧ x < 3 }

-- Define sets A and B
def set_A : Set ℤ := { -2, 0, 1 }
def set_B : Set ℤ := { -1, 0, 1, 2 }

-- Main theorem statement
theorem intersection_complement
  (hI : I = universal_set)
  (hA : A = set_A)
  (hB : B = set_B) :
  B ∩ (I \ A) = { -1, 2 } :=
sorry

end intersection_complement_l141_141541


namespace ab_value_l141_141792

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 * b^2 + a^2 * b^3 = 20) : ab = 2 ∨ ab = -2 :=
by
  sorry

end ab_value_l141_141792


namespace mutually_exclusive_events_l141_141095

-- Define the bag, balls, and events
def bag := (5, 3) -- (red balls, white balls)

def draws (r w : Nat) := (r + w = 3)

def event_A (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.1 = 3 -- At least one red ball and all red balls
def event_B (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.2 = 3 -- At least one red ball and all white balls
def event_C (draw : ℕ × ℕ) := draw.1 ≥ 1 ∧ draw.2 ≥ 1 -- At least one red ball and at least one white ball
def event_D (draw : ℕ × ℕ) := (draw.1 = 1 ∨ draw.1 = 2) ∧ draws draw.1 draw.2 -- Exactly one red ball and exactly two red balls

theorem mutually_exclusive_events : 
  ∀ draw : ℕ × ℕ, 
  (event_A draw ∨ event_B draw ∨ event_C draw ∨ event_D draw) → 
  (event_D draw ↔ (draw.1 = 1 ∧ draw.2 = 2) ∨ (draw.1 = 2 ∧ draw.2 = 1)) :=
by
  sorry

end mutually_exclusive_events_l141_141095


namespace euler_totient_divisibility_l141_141767

def euler_totient (n : ℕ) : ℕ := sorry

theorem euler_totient_divisibility (n : ℕ) (hn : 0 < n) : 2^(n * (n + 1)) ∣ 32 * euler_totient (2^(2^n) - 1) := 
sorry

end euler_totient_divisibility_l141_141767


namespace fixed_point_on_line_l141_141445

theorem fixed_point_on_line (m x y : ℝ) (h : ∀ m : ℝ, m * x - y + 2 * m + 1 = 0) : 
  (x = -2 ∧ y = 1) :=
sorry

end fixed_point_on_line_l141_141445


namespace river_current_speed_l141_141061

noncomputable section

variables {d r w : ℝ}

def time_equation_normal_speed (d r w : ℝ) : Prop :=
  (d / (r + w)) + 4 = (d / (r - w))

def time_equation_tripled_speed (d r w : ℝ) : Prop :=
  (d / (3 * r + w)) + 2 = (d / (3 * r - w))

theorem river_current_speed (d r : ℝ) (h1 : time_equation_normal_speed d r w) (h2 : time_equation_tripled_speed d r w) : w = 2 :=
sorry

end river_current_speed_l141_141061


namespace meeting_probability_of_C_and_D_l141_141910

open Finset

def num_paths (steps right_steps : ℕ) : ℕ :=
  Nat.choose steps right_steps

noncomputable def meet_probability : ℝ :=
  ∑ i in range 5, 
    (num_paths 5 i : ℝ) / 2^5 * (num_paths 5 (i + 1) : ℝ) / 2^5

theorem meeting_probability_of_C_and_D : meet_probability = 0.049 :=
by
  sorry

end meeting_probability_of_C_and_D_l141_141910


namespace time_to_traverse_nth_mile_l141_141063

theorem time_to_traverse_nth_mile (n : ℕ) (n_pos : n > 1) :
  let k := (1 / 2 : ℝ)
  let s_n := k / ((n-1) * (2 ^ (n-2)))
  let t_n := 1 / s_n
  t_n = 2 * (n-1) * 2^(n-2) := 
by sorry

end time_to_traverse_nth_mile_l141_141063


namespace painted_pictures_in_june_l141_141697

theorem painted_pictures_in_june (J : ℕ) (h1 : J + (J + 2) + 9 = 13) : J = 1 :=
by
  -- Given condition translates to J + J + 2 + 9 = 13
  -- Simplification yields 2J + 11 = 13
  -- Solving 2J + 11 = 13 gives J = 1
  sorry

end painted_pictures_in_june_l141_141697


namespace find_value_l141_141993

theorem find_value (x : ℝ) (h : x^2 - 2 * x = 1) : 2023 + 6 * x - 3 * x^2 = 2020 := 
by 
sorry

end find_value_l141_141993


namespace problem1_problem2_1_problem2_2_l141_141370

-- Define the quadratic function and conditions
def quadratic (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

-- Problem 1: Expression of the quadratic function given vertex
theorem problem1 (b c : ℝ) : (quadratic 2 b c = 0) ∧ (∀ x : ℝ, quadratic x b c = (x - 2)^2) ↔ (b = -4) ∧ (c = 4) := sorry

-- Problem 2.1: Given n < -5 and y1 = y2, range of b + c
theorem problem2_1 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : quadratic (3*n - 4) b c = y1)
  (h3 : quadratic (5*n + 6) b c = y2) (h4 : y1 = y2) : b + c < -38 := sorry

-- Problem 2.2: Given n < -5 and c > 0, compare values of y1 and y2
theorem problem2_2 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : c > 0) 
  (h3 : quadratic (3*n - 4) b c = y1) (h4 : quadratic (5*n + 6) b c = y2) : y1 < y2 := sorry

end problem1_problem2_1_problem2_2_l141_141370


namespace parabola_x_intercepts_count_l141_141387

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end parabola_x_intercepts_count_l141_141387


namespace parabola_standard_eq_line_m_tangent_l141_141724

open Real

variables (p k : ℝ) (x y : ℝ)

-- Definitions based on conditions
def parabola_equation (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 2 * p * y
def line_m (k : ℝ) : Prop := ∀ x y : ℝ, y = k * x + 6

-- Problem statement
theorem parabola_standard_eq (p : ℝ) (hp : p = 2) :
  parabola_equation p ↔ (∀ x y : ℝ, x^2 = 4 * y) :=
sorry

theorem line_m_tangent (k : ℝ) (x1 x2 : ℝ)
  (hpq : x1 + x2 = 4 * k ∧ x1 * x2 = -24)
  (hk : k = 1/2 ∨ k = -1/2) :
  line_m k ↔ ((k = 1/2 ∧ ∀ x y : ℝ, y = 1/2 * x + 6) ∨ (k = -1/2 ∧ ∀ x y : ℝ, y = -1/2 * x + 6)) :=
sorry

end parabola_standard_eq_line_m_tangent_l141_141724


namespace percentage_of_students_on_trip_l141_141118

variable (students : ℕ) -- Total number of students at the school
variable (students_trip_and_more_than_100 : ℕ) -- Number of students who went to the camping trip and took more than $100
variable (percent_trip_and_more_than_100 : ℚ) -- Percent of students who went to camping trip and took more than $100

-- Given Conditions
def cond1 : students_trip_and_more_than_100 = (percent_trip_and_more_than_100 * students) := 
  by
    sorry  -- This will represent the first condition: 18% of students went to a camping trip and took more than $100.

variable (percent_did_not_take_more_than_100 : ℚ) -- Percent of students who went to camping trip and did not take more than $100

-- second condition
def cond2 : percent_did_not_take_more_than_100 = 0.75 := 
  by
    sorry  -- Represent the second condition: 75% of students who went to the camping trip did not take more than $100.

-- Prove
theorem percentage_of_students_on_trip : 
  (students_trip_and_more_than_100 / (0.25 * students)) * 100 = (72 : ℚ) := 
  by
    sorry

end percentage_of_students_on_trip_l141_141118


namespace chinese_chess_draw_probability_l141_141313

theorem chinese_chess_draw_probability (pMingNotLosing : ℚ) (pDongLosing : ℚ) : 
    pMingNotLosing = 3/4 → 
    pDongLosing = 1/2 → 
    (pMingNotLosing - (1 - pDongLosing)) = 1/4 :=
by
  intros
  sorry

end chinese_chess_draw_probability_l141_141313


namespace hari_contribution_l141_141055

theorem hari_contribution (c_p: ℕ) (m_p: ℕ) (ratio_p: ℕ) 
                          (m_h: ℕ) (ratio_h: ℕ) (profit_ratio_p: ℕ) (profit_ratio_h: ℕ) 
                          (c_h: ℕ) : 
  (c_p = 3780) → 
  (m_p = 12) → 
  (ratio_p = 2) → 
  (m_h = 7) → 
  (ratio_h = 3) → 
  (profit_ratio_p = 2) →
  (profit_ratio_h = 3) →
  (c_p * m_p * profit_ratio_h) = (c_h * m_h * profit_ratio_p) → 
  c_h = 9720 :=
by
  intros
  sorry

end hari_contribution_l141_141055


namespace compressor_stations_valid_l141_141931

def compressor_stations : Prop :=
  ∃ (x y z a : ℝ),
    x + y = 3 * z ∧  -- condition 1
    z + y = x + a ∧  -- condition 2
    x + z = 60 ∧     -- condition 3
    0 < a ∧ a < 60 ∧ -- condition 4
    a = 42 ∧         -- specific value for a
    x = 33 ∧         -- expected value for x
    y = 48 ∧         -- expected value for y
    z = 27           -- expected value for z

theorem compressor_stations_valid : compressor_stations := 
  by sorry

end compressor_stations_valid_l141_141931


namespace similar_triangle_perimeter_l141_141334

theorem similar_triangle_perimeter :
  ∀ (a b c : ℝ), a = 7 ∧ b = 7 ∧ c = 12 →
  ∀ (d : ℝ), d = 30 →
  ∃ (p : ℝ), p = 65 ∧ 
  (∃ a' b' c' : ℝ, (a' = 17.5 ∧ b' = 17.5 ∧ c' = d) ∧ p = a' + b' + c') :=
by sorry

end similar_triangle_perimeter_l141_141334


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l141_141611

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l141_141611


namespace find_a_maximize_profit_sets_sold_after_increase_l141_141839

variable (a x m : ℕ)

-- Condition for finding 'a'
def condition_for_a (a : ℕ) : Prop :=
  600 * (a - 110) = 160 * a

-- The equation after solving
def solution_for_a (a : ℕ) : Prop :=
  a = 150

theorem find_a : condition_for_a a → solution_for_a a :=
sorry

-- Profit maximization constraints
def condition_for_max_profit (x : ℕ) : Prop :=
  x + 5 * x + 20 ≤ 200

-- Total number of items purchased
def total_items_purchased (x : ℕ) : ℕ :=
  x + 5 * x + 20

-- Profit expression
def profit (x : ℕ) : ℕ :=
  215 * x + 600

-- Maximized profit
def maximum_profit (W : ℕ) : Prop :=
  W = 7050

theorem maximize_profit (x : ℕ) (W : ℕ) :
  condition_for_max_profit x → x ≤ 30 → total_items_purchased x ≤ 200 → maximum_profit W → x = 30 :=
sorry

-- Condition for sets sold after increase
def condition_for_sets_sold (a m : ℕ) : Prop :=
  let new_table_price := 160
  let new_chair_price := 50
  let profit_m_after_increase := (500 - new_table_price - 4 * new_chair_price) * m +
                                (30 - m) * (270 - new_table_price) +
                                (170 - 4 * m) * (70 - new_chair_price)
  profit_m_after_increase + 2250 = 7050 - 2250

-- Solved for 'm'
def quantity_of_sets_sold (m : ℕ) : Prop :=
  m = 20

theorem sets_sold_after_increase (a m : ℕ) :
  condition_for_sets_sold a m → quantity_of_sets_sold m :=
sorry

end find_a_maximize_profit_sets_sold_after_increase_l141_141839


namespace total_spent_on_video_games_l141_141562

theorem total_spent_on_video_games (cost_basketball cost_racing : ℝ) (h_ball : cost_basketball = 5.20) (h_race : cost_racing = 4.23) : 
  cost_basketball + cost_racing = 9.43 :=
by
  sorry

end total_spent_on_video_games_l141_141562


namespace arithmetic_sequence_common_difference_l141_141756

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 5 = 3) (h2 : a_n 6 = -2) : a_n 6 - a_n 5 = -5 :=
by
  sorry

end arithmetic_sequence_common_difference_l141_141756


namespace insulation_cost_per_sq_ft_l141_141848

theorem insulation_cost_per_sq_ft 
  (l w h : ℤ) 
  (surface_area : ℤ := (2 * l * w) + (2 * l * h) + (2 * w * h))
  (total_cost : ℤ)
  (cost_per_sq_ft : ℤ := total_cost / surface_area)
  (h_l : l = 3)
  (h_w : w = 5)
  (h_h : h = 2)
  (h_total_cost : total_cost = 1240) :
  cost_per_sq_ft = 20 := 
by
  sorry

end insulation_cost_per_sq_ft_l141_141848


namespace race_duration_l141_141453

theorem race_duration 
  (lap_distance : ℕ) (laps : ℕ)
  (award_per_hundred_meters : ℝ) (earn_rate_per_minute : ℝ)
  (total_distance : ℕ) (total_award : ℝ) (duration : ℝ) :
  lap_distance = 100 →
  laps = 24 →
  award_per_hundred_meters = 3.5 →
  earn_rate_per_minute = 7 →
  total_distance = lap_distance * laps →
  total_award = (total_distance / 100) * award_per_hundred_meters →
  duration = total_award / earn_rate_per_minute →
  duration = 12 := 
by 
  intros;
  sorry

end race_duration_l141_141453


namespace chord_intercept_min_value_l141_141997

noncomputable def minimum_value_of_fraction (a b : ℝ) : ℝ :=
  if a > 0 ∧ b > 0 ∧ a + b = 1 then (2 / a + 3 / b) else 0

theorem chord_intercept_min_value : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + b = 1) ∧ minimum_value_of_fraction a b = 5 + 2 * Real.sqrt 6 :=
begin
    use [1 - Real.sqrt 6, Real.sqrt 6],
    split,
    { exact sub_pos_of_lt (lt_sqrt.2 (show 1 < 6, by norm_num)) },
    split,
    { exact sqrt_pos.2 (by norm_num) },
    split,
    { norm_num },
    { rw minimum_value_of_fraction,
      split_ifs,
      norm_num,
      exact congr_arg (fun x => 5 + x) (Real.sqrt_mul (two_ne_zero'.ne.symm) (sqrt_nonneg _).symm) },
end

end chord_intercept_min_value_l141_141997


namespace inequality_satisfaction_l141_141231

theorem inequality_satisfaction (a b : ℝ) (h : a < 0) : (a < b) ∧ (a^2 + b^2 > 2) :=
by
  sorry

end inequality_satisfaction_l141_141231


namespace parallelogram_height_l141_141530

theorem parallelogram_height
  (A b : ℝ)
  (h : ℝ)
  (h_area : A = 120)
  (h_base : b = 12)
  (h_formula : A = b * h) : h = 10 :=
by 
  sorry

end parallelogram_height_l141_141530


namespace find_x_for_f_eq_f_inv_l141_141865

noncomputable def f (x : ℝ) : ℝ := 4 * x - 9
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem find_x_for_f_eq_f_inv :
  ∃ x : ℝ, f x = f_inv x ∧ x = 3 :=
by
  use 3
  split
  . show f 3 = f_inv 3
    rw [f, f_inv]
    norm_num
  . show 3 = 3
    rfl

end find_x_for_f_eq_f_inv_l141_141865


namespace volume_of_prism_l141_141686

theorem volume_of_prism (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 :=
by
  sorry

end volume_of_prism_l141_141686


namespace numDifferentSignals_l141_141019

-- Number of indicator lights in a row
def numLights : Nat := 6

-- Number of lights that light up each time
def lightsLit : Nat := 3

-- Number of colors each light can show
def numColors : Nat := 3

-- Function to calculate number of different signals
noncomputable def calculateSignals (n m k : Nat) : Nat :=
  -- Number of possible arrangements of "adjacent, adjacent, separate" and "separate, adjacent, adjacent"
  let arrangements := 4 + 4
  -- Number of color combinations for the lit lights
  let colors := k * k * k
  arrangements * colors

-- Theorem stating the total number of different signals is 324
theorem numDifferentSignals : calculateSignals numLights lightsLit numColors = 324 := 
by
  sorry

end numDifferentSignals_l141_141019


namespace percentage_proof_l141_141745

theorem percentage_proof (a : ℝ) (paise : ℝ) (x : ℝ) (h1: paise = 85) (h2: a = 170) : 
  (x/100) * a = paise ↔ x = 50 := 
by
  -- The setup includes:
  -- paise = 85
  -- a = 170
  -- We prove that x% of 170 equals 85 if and only if x = 50.
  sorry

end percentage_proof_l141_141745


namespace simplified_value_l141_141308

theorem simplified_value :
  (245^2 - 205^2) / 40 = 450 := by
  sorry

end simplified_value_l141_141308


namespace simplify_and_evaluate_l141_141148

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3)) - (x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  sorry

end simplify_and_evaluate_l141_141148


namespace modulusOfComplexNumber_proof_l141_141222

noncomputable def complexNumber {a : ℝ} (h : (a - 1) + 1 * Complex.I = (0 : ℂ)) : ℂ :=
  (2 + Real.sqrt 2 * Complex.I) / (a - Complex.I)

theorem modulusOfComplexNumber_proof (a : ℝ) (h : (a - 1) + 1 * Complex.I = (0 : ℂ)) : Complex.abs (complexNumber h) = Real.sqrt 3 := by
  sorry

end modulusOfComplexNumber_proof_l141_141222


namespace friends_count_l141_141843

theorem friends_count (n : ℕ) (average_rent : ℝ) (new_average_rent : ℝ) (original_rent : ℝ) (increase_percent : ℝ)
  (H1 : average_rent = 800)
  (H2 : new_average_rent = 870)
  (H3 : original_rent = 1400)
  (H4 : increase_percent = 0.20) :
  n = 4 :=
by
  -- Define the initial total rent
  let initial_total_rent := n * average_rent
  -- Define the increased rent for one person
  let increased_rent := original_rent * (1 + increase_percent)
  -- Define the new total rent
  let new_total_rent := initial_total_rent - original_rent + increased_rent
  -- Set up the new average rent equation
  have rent_equation := new_total_rent = n * new_average_rent
  sorry

end friends_count_l141_141843


namespace range_of_a_l141_141015

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_a (h_odd : ∀ x, f (-x) = -f x) 
  (h_period : ∀ x, f (x + 3) = f x)
  (h1 : f 1 > 1) 
  (h2018 : f 2018 = (a : ℝ) ^ 2 - 5) : 
  -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l141_141015


namespace battery_life_in_standby_l141_141780

noncomputable def remaining_battery_life (b_s : ℝ) (b_a : ℝ) (t_total : ℝ) (t_active : ℝ) : ℝ :=
  let standby_rate := 1 / b_s
  let active_rate := 1 / b_a
  let standby_time := t_total - t_active
  let consumption_active := t_active * active_rate
  let consumption_standby := standby_time * standby_rate
  let total_consumption := consumption_active + consumption_standby
  let remaining_battery := 1 - total_consumption
  remaining_battery * b_s

theorem battery_life_in_standby :
  remaining_battery_life 30 4 10 1.5 = 10.25 := sorry

end battery_life_in_standby_l141_141780


namespace infinite_sequence_no_square_factors_l141_141437

/-
  Prove that there exist infinitely many positive integers \( n_1 < n_2 < \cdots \)
  such that for all \( i \neq j \), \( n_i + n_j \) has no square factors other than 1.
-/

theorem infinite_sequence_no_square_factors :
  ∃ (n : ℕ → ℕ), (∀ (i j : ℕ), i ≠ j → ∀ p : ℕ, p ≠ 1 → p^2 ∣ (n i + n j) → false) ∧
    ∀ k : ℕ, n k < n (k + 1) :=
sorry

end infinite_sequence_no_square_factors_l141_141437


namespace domain_of_f_l141_141801

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.sqrt (-x^2 + x + 2)

theorem domain_of_f :
  {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l141_141801


namespace maggie_goldfish_fraction_l141_141426

theorem maggie_goldfish_fraction :
  ∀ (x : ℕ), 3*x / 5 + 20 = x → (x / 100 : ℚ) = 1 / 2 :=
by
  sorry

end maggie_goldfish_fraction_l141_141426


namespace partial_fraction_decomposition_l141_141525

theorem partial_fraction_decomposition :
  ∃ x y z : ℕ, 77 * x + 55 * y + 35 * z = 674 ∧ x + y + z = 14 :=
begin
  sorry
end

end partial_fraction_decomposition_l141_141525


namespace hyperbola_through_focus_and_asymptotes_l141_141159

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 1

def asymptotes_holds (x y : ℝ) : Prop :=
  (x + y = 0) ∨ (x - y = 0)

theorem hyperbola_through_focus_and_asymptotes :
  hyperbola parabola_focus.1 parabola_focus.2 ∧ asymptotes_holds parabola_focus.1 parabola_focus.2 :=
sorry

end hyperbola_through_focus_and_asymptotes_l141_141159


namespace y_sequence_integer_7_l141_141705

noncomputable def y_sequence (n : ℕ) : ℝ :=
  match n with
  | 0     => 0    -- Not used, 0 index case
  | 1     => (2:ℝ)^(1/3)
  | k + 1 => (y_sequence k)^(2^(1/3))

theorem y_sequence_integer_7 : ∃ n : ℕ, (∀ m < n, ¬ ∃ k : ℤ, y_sequence m = k) ∧ (∃ k : ℤ, y_sequence n = k) ∧ n = 7 :=
by {
  sorry
}

end y_sequence_integer_7_l141_141705


namespace contradiction_proof_example_l141_141464

theorem contradiction_proof_example (a b : ℝ) (h: a ≤ b → False) : a > b :=
by sorry

end contradiction_proof_example_l141_141464


namespace problem_solution_l141_141673

theorem problem_solution (a b c : ℝ)
  (h₁ : 10 = (6 / 100) * a)
  (h₂ : 6 = (10 / 100) * b)
  (h₃ : c = b / a) : c = 0.36 :=
by sorry

end problem_solution_l141_141673


namespace determine_x_l141_141225

/-
  Determine \( x \) when \( y = 19 \)
  given the ratio of \( 5x - 3 \) to \( y + 10 \) is constant,
  and when \( x = 3 \), \( y = 4 \).
-/

theorem determine_x (x y k : ℚ) (h1 : ∀ x y, (5 * x - 3) / (y + 10) = k)
  (h2 : 5 * 3 - 3 / (4 + 10) = k) : x = 39 / 7 :=
sorry

end determine_x_l141_141225


namespace city_map_representation_l141_141911

-- Given conditions
def scale (x : ℕ) : ℕ := x * 6
def cm_represents_km(cm : ℕ) : ℕ := scale cm
def fifteen_cm := 15
def ninety_km := 90

-- Given condition: 15 centimeters represents 90 kilometers
axiom representation : cm_represents_km fifteen_cm = ninety_km

-- Proof statement: A 20-centimeter length represents 120 kilometers
def twenty_cm := 20
def correct_answer := 120

theorem city_map_representation : cm_represents_km twenty_cm = correct_answer := by
  sorry

end city_map_representation_l141_141911


namespace average_of_first_40_results_l141_141799

theorem average_of_first_40_results 
  (A : ℝ)
  (avg_other_30 : ℝ := 40)
  (avg_all_70 : ℝ := 34.285714285714285) : A = 30 :=
by 
  let sum1 := A * 40
  let sum2 := avg_other_30 * 30
  let combined_sum := sum1 + sum2
  let combined_avg := combined_sum / 70
  have h1 : combined_avg = avg_all_70 := by sorry
  have h2 : combined_avg = 34.285714285714285 := by sorry
  have h3 : combined_sum = (A * 40) + (40 * 30) := by sorry
  have h4 : (A * 40) + 1200 = 2400 := by sorry
  have h5 : A * 40 = 1200 := by sorry
  have h6 : A = 1200 / 40 := by sorry
  have h7 : A = 30 := by sorry
  exact h7

end average_of_first_40_results_l141_141799


namespace horizontal_distance_is_0_65_l141_141492

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 4

-- Calculate the horizontal distance between two points on the parabola given their y-coordinates and prove it equals to 0.65
theorem horizontal_distance_is_0_65 :
  ∃ (x1 x2 : ℝ), 
    parabola x1 = 10 ∧ parabola x2 = 0 ∧ abs (x1 - x2) = 0.65 :=
sorry

end horizontal_distance_is_0_65_l141_141492


namespace intersection_point_l141_141012

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := (x^2 - 4*x + 4) / (2*x - 6)
def g (x : ℝ) := (-2*x^2 + 6*x - 4) / (x - 3)

-- Conditions:
-- The graphs of f(x) and g(x) have the same vertical asymptote (x = 3)
-- The oblique asymptotes of f(x) and g(x) are perpendicular and intersect at the origin
-- The graphs of f(x) and g(x) have an intersection point at x = 1

-- The proof will show that the other intersection point is at x = 2
-- and evaluates f(2)

theorem intersection_point (h1 : f 1 = g 1) : 
  ∃ x : ℝ, x ≠ 1 ∧ f x = g x ∧ x = 2 ∧ f 2 = 0 := 
by
  sorry

end intersection_point_l141_141012


namespace quadratic_factor_conditions_l141_141450

theorem quadratic_factor_conditions (b : ℤ) :
  (∃ m n p q : ℤ, m * p = 15 ∧ n * q = 75 ∧ mq + np = b) → ∃ (c : ℤ), b = c :=
sorry

end quadratic_factor_conditions_l141_141450


namespace original_number_l141_141823

theorem original_number 
  (x : ℝ)
  (h₁ : 0 < x)
  (h₂ : 1000 * x = 3 * (1 / x)) : 
  x = (Real.sqrt 30) / 100 :=
sorry

end original_number_l141_141823


namespace gold_weight_l141_141505

theorem gold_weight:
  ∀ (G C A : ℕ), 
  C = 9 → 
  (A = (4 * G + C) / 5) → 
  A = 17 → 
  G = 19 :=
by
  intros G C A hc ha h17
  sorry

end gold_weight_l141_141505


namespace correct_operation_l141_141049

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := 
by 
  sorry

end correct_operation_l141_141049


namespace length_of_rectangle_l141_141397

-- Definitions based on conditions:
def side_length_square : ℝ := 4
def width_rectangle : ℝ := 8
def area_square (side : ℝ) : ℝ := side * side
def area_rectangle (width length : ℝ) : ℝ := width * length

-- The goal is to prove the length of the rectangle
theorem length_of_rectangle :
  (area_square side_length_square) = (area_rectangle width_rectangle 2) :=
by
  sorry

end length_of_rectangle_l141_141397


namespace find_vector_v_l141_141769

def vector3 := ℝ × ℝ × ℝ

def cross_product (u v : vector3) : vector3 :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1  - u.1   * v.2.2,
   u.1   * v.2.1 - u.2.1 * v.1)

def a : vector3 := (1, 2, 1)
def b : vector3 := (2, 0, -1)
def v : vector3 := (3, 2, 0)
def b_cross_a : vector3 := (2, 3, 4)
def a_cross_b : vector3 := (-2, 3, -4)

theorem find_vector_v :
  cross_product v a = b_cross_a ∧ cross_product v b = a_cross_b :=
sorry

end find_vector_v_l141_141769


namespace lori_beanie_babies_times_l141_141779

theorem lori_beanie_babies_times (l s : ℕ) (h1 : l = 300) (h2 : l + s = 320) : l = 15 * s :=
by
  sorry

end lori_beanie_babies_times_l141_141779


namespace range_of_k_find_k_value_l141_141032

open Real

noncomputable def quadratic_eq_has_real_roots (a b c : ℝ) (disc : ℝ) : Prop :=
  disc > 0

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

variables {a : ℝ} {b : ℝ} {c : ℝ} {k : ℝ}
variables (alpha beta : ℝ)

-- Given conditions
def quadratic_eq_with_k := (a = 1) ∧ (b = 2) ∧ (c = 3 - k)
def two_distinct_real_roots := quadratic_eq_has_real_roots 1 2 (3 - k) (discriminant 1 2 (3 - k))
def product_of_roots := alpha * beta = 3 - k
def given_condition := k^2 = alpha * beta + 3 * k

-- Proofs to be done
theorem range_of_k : quadratic_eq_with_k k → two_distinct_real_roots k → k > 2 :=
by
  intro h1 h2
  sorry

theorem find_k_value : quadratic_eq_with_k k → k > 2 → given_condition k alpha beta → k = 3 :=
by
  intro h1 h2 h3
  sorry

end range_of_k_find_k_value_l141_141032


namespace find_C_l141_141504

theorem find_C (A B C : ℝ) 
  (h1 : A + B + C = 600) 
  (h2 : A + C = 250) 
  (h3 : B + C = 450) : 
  C = 100 :=
sorry

end find_C_l141_141504


namespace average_salary_of_associates_l141_141840

theorem average_salary_of_associates 
  (num_managers : ℕ) (num_associates : ℕ)
  (avg_salary_managers : ℝ) (avg_salary_company : ℝ)
  (H_num_managers : num_managers = 15)
  (H_num_associates : num_associates = 75)
  (H_avg_salary_managers : avg_salary_managers = 90000)
  (H_avg_salary_company : avg_salary_company = 40000) :
  ∃ (A : ℝ), (num_managers * avg_salary_managers + num_associates * A) / (num_managers + num_associates) = avg_salary_company ∧ A = 30000 := by
  sorry

end average_salary_of_associates_l141_141840


namespace perimeter_of_square_l141_141498

-- Defining the square with area
structure Square where
  side_length : ℝ
  area : ℝ

-- Defining a constant square with given area 625
def givenSquare : Square := 
  { side_length := 25, -- will square root the area of 625
    area := 625 }

-- Defining the function to calculate the perimeter of the square
noncomputable def perimeter (s : Square) : ℝ :=
  4 * s.side_length

-- The theorem stating that the perimeter of the given square with area 625 is 100
theorem perimeter_of_square : perimeter givenSquare = 100 := 
sorry

end perimeter_of_square_l141_141498


namespace fruit_eating_orders_l141_141079

theorem fruit_eating_orders:
  let apples := 4
  let oranges := 3
  let bananas := 2
  let days := 7
  ∀ (orderings : List (List ℕ)) (valid_ordering : List ℕ → Prop),
  (∀ order ∈ orderings, valid_ordering order) →
  (valid_ordering = λ order, 
    order.length = days ∧ 
    (∀ i, i < days → order.nth i ≠ some 0 ∨ 
    (∃ j, j ≠ i ∧ j < days ∧ order.nth j ≠ some 0 ∧ order.nth j ≠ some 1))) → 
  (orderings.length = 150) :=
begin
  intros orderings valid_ordering valid_property valid_spec,
  sorry,
end

end fruit_eating_orders_l141_141079


namespace num_ways_to_remove_blocks_l141_141273

-- Definitions based on the problem conditions
def stack_blocks := 85
def block_layers := [1, 4, 16, 64]

-- Theorem statement
theorem num_ways_to_remove_blocks : 
  (∃ f : (ℕ → ℕ), 
    (∀ n, f n = if n = 0 then 1 else if n ≤ 4 then n * f (n - 1) + 3 * (f (n - 1) - 1) else 4^3 * 16) ∧ 
    f 5 = 3384) := sorry

end num_ways_to_remove_blocks_l141_141273


namespace parabola_arc_length_exceeds_4_l141_141846

noncomputable def parabola_arc_length (k : ℝ) : ℝ :=
  2 * ∫ x in (0 : ℝ)..(sqrt (2 * k - 1) / k), sqrt (1 + 4 * k^2 * x^2)

theorem parabola_arc_length_exceeds_4 :
  ∃ k : ℝ, k > 0 ∧ parabola_arc_length k > 4 :=
begin
  sorry
end

end parabola_arc_length_exceeds_4_l141_141846


namespace minimum_value_inequality_l141_141134

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 1) * (y^2 + 5 * y + 1) * (z^2 + 5 * y + 1) / (x * y * z) ≥ 343 :=
by sorry

end minimum_value_inequality_l141_141134


namespace joan_friends_kittens_l141_141418

theorem joan_friends_kittens (initial_kittens final_kittens friends_kittens : ℕ) 
  (h1 : initial_kittens = 8) 
  (h2 : final_kittens = 10) 
  (h3 : friends_kittens = 2) : 
  final_kittens - initial_kittens = friends_kittens := 
by 
  -- Sorry is used here as a placeholder to indicate where the proof would go.
  sorry

end joan_friends_kittens_l141_141418


namespace vertex_closer_to_Q_than_P_l141_141242

open Metric

theorem vertex_closer_to_Q_than_P
  {α : Type*} [MetricSpace α]
  {polygon : Set α} (h_convex : Convex ℝ polygon)
  {P Q : α} (hP_in : P ∈ polygon) (hQ_in : Q ∈ polygon) :
  ∃ (V ∈ polygon), dist V Q < dist V P := 
sorry

end vertex_closer_to_Q_than_P_l141_141242


namespace discount_per_person_correct_l141_141072

noncomputable def price_per_person : ℕ := 147
noncomputable def total_people : ℕ := 2
noncomputable def total_cost_with_discount : ℕ := 266

theorem discount_per_person_correct :
  let total_cost_without_discount := price_per_person * total_people
  let total_discount := total_cost_without_discount - total_cost_with_discount
  let discount_per_person := total_discount / total_people
  discount_per_person = 14 := by
  sorry

end discount_per_person_correct_l141_141072


namespace div_remainder_l141_141469

theorem div_remainder (x : ℕ) (h : x = 2^40) : 
  (2^160 + 160) % (2^80 + 2^40 + 1) = 159 :=
by
  sorry

end div_remainder_l141_141469


namespace number_of_cheesecakes_in_fridge_l141_141838

section cheesecake_problem

def cheesecakes_on_display : ℕ := 10
def cheesecakes_sold : ℕ := 7
def cheesecakes_left_to_be_sold : ℕ := 18

def cheesecakes_in_fridge (total_display : ℕ) (sold : ℕ) (left : ℕ) : ℕ :=
  left - (total_display - sold)

theorem number_of_cheesecakes_in_fridge :
  cheesecakes_in_fridge cheesecakes_on_display cheesecakes_sold cheesecakes_left_to_be_sold = 15 :=
by
  sorry

end cheesecake_problem

end number_of_cheesecakes_in_fridge_l141_141838


namespace complementary_angles_not_obtuse_l141_141295

-- Define the concept of complementary angles.
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

-- Define that neither angle should be obtuse.
def not_obtuse (a b : ℝ) : Prop :=
  a < 90 ∧ b < 90

-- Proof problem statement
theorem complementary_angles_not_obtuse (a b : ℝ) (ha : a < 90) (hb : b < 90) (h_comp : is_complementary a b) : 
  not_obtuse a b :=
by
  sorry

end complementary_angles_not_obtuse_l141_141295


namespace find_fifth_month_sale_l141_141324

theorem find_fifth_month_sale (
  a1 a2 a3 a4 a6 : ℕ
) (avg_sales : ℕ)
  (h1 : a1 = 5420)
  (h2 : a2 = 5660)
  (h3 : a3 = 6200)
  (h4 : a4 = 6350)
  (h6 : a6 = 7070)
  (avg_condition : avg_sales = 6200)
  (total_condition : (a1 + a2 + a3 + a4 + a6 + (6500)) / 6 = avg_sales)
  : (∃ a5 : ℕ, a5 = 6500 ∧ (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sales) :=
by {
  sorry
}

end find_fifth_month_sale_l141_141324


namespace equation_has_solution_iff_l141_141809

open Real

theorem equation_has_solution_iff (a : ℝ) : 
  (∃ x : ℝ, (1/3)^|x| + a - 1 = 0) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end equation_has_solution_iff_l141_141809


namespace least_boxes_l141_141813
-- Definitions and conditions
def isPerfectCube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

def isFactor (a b : ℕ) : Prop := ∃ k, a * k = b

def numBoxes (N boxSize : ℕ) : ℕ := N / boxSize

-- Specific conditions for our problem
theorem least_boxes (N : ℕ) (boxSize : ℕ) 
  (h1 : N ≠ 0) 
  (h2 : isPerfectCube N)
  (h3 : isFactor boxSize N)
  (h4 : boxSize = 45): 
  numBoxes N boxSize = 75 :=
by
  sorry

end least_boxes_l141_141813


namespace kim_monthly_expenses_l141_141765

-- Define the conditions

def initial_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def payback_period : ℕ := 10

-- Define the proof statement
theorem kim_monthly_expenses :
  ∃ (E : ℝ), 
    (payback_period * (monthly_revenue - E) = initial_cost) → (E = 1500) :=
by
  sorry

end kim_monthly_expenses_l141_141765


namespace arithmetic_mean_of_fractions_l141_141649

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l141_141649


namespace multiply_same_exponents_l141_141076

theorem multiply_same_exponents (x : ℝ) : (x^3) * (x^3) = x^6 :=
by sorry

end multiply_same_exponents_l141_141076


namespace number_of_girls_in_school_l141_141844

theorem number_of_girls_in_school (total_students : ℕ) (sample_size : ℕ) (x : ℕ) :
  total_students = 2400 →
  sample_size = 200 →
  2 * x + 10 = sample_size →
  (95 / 200 : ℚ) * (total_students : ℚ) = 1140 :=
by
  intros h_total h_sample h_sampled
  rw [h_total, h_sample] at *
  sorry

end number_of_girls_in_school_l141_141844


namespace integral_circle_minus_x_l141_141349

open Set Filter

theorem integral_circle_minus_x : (∫ x in 0..1, (sqrt (1 - (x - 1) ^ 2) - x)) = (Real.pi / 4 - 1 / 2) := by
  sorry

end integral_circle_minus_x_l141_141349


namespace min_AP_l141_141264

noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B' : ℝ × ℝ := (8, 6)
def parabola (P' : ℝ × ℝ) : Prop := P'.2^2 = 8 * P'.1

theorem min_AP'_plus_BP' : 
  ∃ P' : ℝ × ℝ, parabola P' ∧ (dist A P' + dist B' P' = 12) := 
sorry

end min_AP_l141_141264


namespace isosceles_triangle_smallest_angle_l141_141507

-- Given conditions:
-- 1. The triangle is isosceles
-- 2. One angle is 40% larger than the measure of a right angle

theorem isosceles_triangle_smallest_angle :
  ∃ (A B C : ℝ), 
  A + B + C = 180 ∧ 
  (A = B ∨ A = C ∨ B = C) ∧ 
  (∃ (large_angle : ℝ), large_angle = 90 + 0.4 * 90 ∧ (A = large_angle ∨ B = large_angle ∨ C = large_angle)) →
  (A = 27 ∨ B = 27 ∨ C = 27) := sorry

end isosceles_triangle_smallest_angle_l141_141507


namespace arithmetic_sum_S9_l141_141267

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable (S : ℕ → ℝ) -- Define the sum of the first n terms
variable (d : ℝ) -- Define the common difference
variable (a_1 : ℝ) -- Define the first term of the sequence

-- Assume the arithmetic sequence properties
axiom arith_seq_def : ∀ n, a (n + 1) = a_1 + n * d

-- Define the sum of the first n terms
axiom sum_first_n_terms : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom given_condition : a 1 + a 7 = 15 - a 4

theorem arithmetic_sum_S9 : S 9 = 45 :=
by
  -- Proof omitted
  sorry

end arithmetic_sum_S9_l141_141267


namespace volume_PABCD_l141_141576

noncomputable def volume_of_pyramid (AB BC : ℝ) (PA : ℝ) : ℝ :=
  (1 / 3) * (AB * BC) * PA

theorem volume_PABCD (AB BC : ℝ) (h_AB : AB = 10) (h_BC : BC = 5)
  (PA : ℝ) (h_PA : PA = 2 * BC) :
  volume_of_pyramid AB BC PA = 500 / 3 :=
by
  subst h_AB
  subst h_BC
  subst h_PA
  -- At this point, we assert that everything simplifies correctly.
  -- This fill in the details for the correct expressions.
  sorry

end volume_PABCD_l141_141576


namespace rectangular_prism_volume_l141_141683

theorem rectangular_prism_volume
  (l w h : ℝ)
  (h1 : l * w = 15)
  (h2 : w * h = 10)
  (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end rectangular_prism_volume_l141_141683


namespace num_intersections_circle_line_eq_two_l141_141160

theorem num_intersections_circle_line_eq_two :
  ∃ (points : Finset (ℝ × ℝ)), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25 ∧ p.1 = 3} = points ∧ points.card = 2 :=
by
  sorry

end num_intersections_circle_line_eq_two_l141_141160


namespace karsyn_total_payment_l141_141764

-- Define the initial price of the phone
def initial_price : ℝ := 600

-- Define the discounted rate for the phone
def discount_rate_phone : ℝ := 0.20

-- Define the prices for additional items
def phone_case_price : ℝ := 25
def screen_protector_price : ℝ := 15

-- Define the discount rates
def discount_rate_125 : ℝ := 0.05
def discount_rate_150 : ℝ := 0.10
def final_discount_rate : ℝ := 0.03

-- Define the tax rate and fee
def exchange_rate_fee : ℝ := 0.02

noncomputable def total_payment (initial_price : ℝ) (discount_rate_phone : ℝ) 
  (phone_case_price : ℝ) (screen_protector_price : ℝ) (discount_rate_125 : ℝ) 
  (discount_rate_150 : ℝ) (final_discount_rate : ℝ) (exchange_rate_fee : ℝ) : ℝ :=
  let discounted_phone_price := initial_price * discount_rate_phone
  let additional_items_price := phone_case_price + screen_protector_price
  let total_before_discounts := discounted_phone_price + additional_items_price
  let total_after_first_discount := total_before_discounts * (1 - discount_rate_125)
  let total_after_second_discount := total_after_first_discount * (1 - discount_rate_150)
  let total_after_all_discounts := total_after_second_discount * (1 - final_discount_rate)
  let total_with_exchange_fee := total_after_all_discounts * (1 + exchange_rate_fee)
  total_with_exchange_fee

theorem karsyn_total_payment :
  total_payment initial_price discount_rate_phone phone_case_price screen_protector_price 
    discount_rate_125 discount_rate_150 final_discount_rate exchange_rate_fee = 135.35 := 
  by 
  -- Specify proof steps here
  sorry

end karsyn_total_payment_l141_141764


namespace find_rate_l141_141963

-- Definitions of conditions
def Principal : ℝ := 2500
def Amount : ℝ := 3875
def Time : ℝ := 12

-- Main statement we are proving
theorem find_rate (P : ℝ) (A : ℝ) (T : ℝ) (R : ℝ) 
    (hP : P = Principal) 
    (hA : A = Amount) 
    (hT : T = Time) 
    (hR : R = (A - P) * 100 / (P * T)) : R = 55 / 12 := 
by 
  sorry

end find_rate_l141_141963


namespace value_of_m_l141_141178

theorem value_of_m :
  ∃ m : ℕ, 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ∧ m = 14 :=
begin
  -- Solution steps would be here
  sorry
end

end value_of_m_l141_141178


namespace rectangle_to_cylinder_max_volume_ratio_l141_141344

/-- Given a rectangle with a perimeter of 12 and converting it into a cylinder 
with the height being the same as the width of the rectangle, prove that the 
ratio of the circumference of the cylinder's base to its height when the volume 
is maximized is 2:1. -/
theorem rectangle_to_cylinder_max_volume_ratio : 
  ∃ (x : ℝ), (2 * x + 2 * (6 - x)) = 12 → 2 * (6 - x) / x = 2 :=
sorry

end rectangle_to_cylinder_max_volume_ratio_l141_141344


namespace Angie_age_ratio_l141_141698

-- Define Angie's age as a variable
variables (A : ℕ)

-- Give the condition
def Angie_age_condition := A + 4 = 20

-- State the theorem to be proved
theorem Angie_age_ratio (h : Angie_age_condition A) : (A : ℚ) / (A + 4) = 4 / 5 := 
sorry

end Angie_age_ratio_l141_141698


namespace sum_of_proper_divisors_less_than_100_of_780_l141_141347

def is_divisor (n d : ℕ) : Bool :=
  d ∣ n

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d ∣ n ∧ d < n)

def proper_divisors_less_than (n bound : ℕ) : List ℕ :=
  (proper_divisors n).filter (λ d => d < bound)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => acc + x) 0

theorem sum_of_proper_divisors_less_than_100_of_780 :
  sum_list (proper_divisors_less_than 780 100) = 428 :=
by
  sorry

end sum_of_proper_divisors_less_than_100_of_780_l141_141347


namespace katya_minimum_problems_l141_141254

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l141_141254


namespace third_shot_scores_l141_141038

noncomputable theory

open Finset

def shooters_scores (a b : Fin 5 → ℕ) : Prop :=
(∀ i, a i ∈ {10, 9, 8, 5, 4, 3, 2}.erase 7) ∧
(∀ i, b i ∈ {10, 9, 8, 5, 4, 3, 2}.erase 6 ∧ b i ∉ {a i}) ∧
(a 0 + a 1 + a 2 = b 0 + b 1 + b 2) ∧
(a 2 + a 3 + a 4 = 3 * (b 2 + b 3 + b 4))

theorem third_shot_scores (a b : Fin 5 → ℕ) (h : shooters_scores a b) :
  a 2 = 10 ∧ b 2 = 2 :=
sorry

end third_shot_scores_l141_141038


namespace div_polynomial_l141_141888

noncomputable def f (x : ℝ) : ℝ := x^4 + 4*x^3 + 6*x^2 + 4*x + 2
noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ := x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*s*x + t

theorem div_polynomial 
  (p q s t : ℝ) 
  (h : ∀ x : ℝ, f x = 0 → g x p q s t = 0) : 
  (p + q + s) * t = -6 :=
by
  sorry

end div_polynomial_l141_141888


namespace arithmetic_mean_of_fractions_l141_141641

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l141_141641


namespace cost_of_each_skin_l141_141935

theorem cost_of_each_skin
  (total_value : ℕ)
  (overall_profit : ℚ)
  (profit_first : ℚ)
  (profit_second : ℚ)
  (total_sell : ℕ)
  (equality : (1 : ℚ) + profit_first ≠ 0 ∧ (1 : ℚ) + profit_second ≠ 0) :
  total_value = 2250 → overall_profit = 0.4 → profit_first = 0.25 → profit_second = -0.5 →
  total_sell = 3150 →
  ∃ x y : ℚ, x = 2700 ∧ y = -450 :=
by
  sorry

end cost_of_each_skin_l141_141935


namespace count_valid_three_digit_numbers_l141_141545

theorem count_valid_three_digit_numbers : 
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  valid_numbers = 90 :=
by
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  have h1 : valid_numbers = 900 - 810 := by rfl
  have h2 : 900 - 810 = 90 := by norm_num
  exact h1.trans h2

end count_valid_three_digit_numbers_l141_141545


namespace function_three_distinct_zeros_l141_141894

theorem function_three_distinct_zeros (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - 3 * a * x + a) ∧ (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) →
  a > 1/4 :=
by
  sorry

end function_three_distinct_zeros_l141_141894


namespace find_x_eq_3_l141_141864

noncomputable def f (x : ℝ) : ℝ := 4 * x - 9
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem find_x_eq_3 : ∃ x : ℝ, f x = f_inv x ∧ x = 3 :=
by
  sorry

end find_x_eq_3_l141_141864


namespace solution_set_of_inequality_l141_141401

theorem solution_set_of_inequality (a t : ℝ) (h1 : ∀ x : ℝ, x^2 - 2 * a * x + a > 0) : 
  a > 0 ∧ a < 1 → (a^(2*t + 1) < a^(t^2 + 2*t - 3) ↔ -2 < t ∧ t < 2) :=
by
  intro ha
  have h : (0 < a ∧ a < 1) := sorry
  exact sorry

end solution_set_of_inequality_l141_141401


namespace expected_value_correct_prob_abs_diff_ge_1_correct_l141_141059

/-- Probability distribution for a single die roll -/
def prob_score (n : ℕ) : ℚ :=
  if n = 1 then 1/2 else if n = 2 then 1/3 else if n = 3 then 1/6 else 0

/-- Expected value based on the given probability distribution -/
def expected_value : ℚ := 
  (1 * prob_score 1) + (2 * prob_score 2) + (3 * prob_score 3)

/-- Proving the expected value calculation -/
theorem expected_value_correct : expected_value = 7/6 :=
  by sorry

/-- Calculate the probability of score difference being at least 1 between two players -/
def prob_abs_diff_ge_1 (x y : ℕ) : ℚ :=
  -- Implementation would involve detailed probability combinations that result in diff >= 1
  sorry

/-- Prove the probability of |x - y| being at least 1 -/
theorem prob_abs_diff_ge_1_correct : 
  ∀ (x y : ℕ), prob_abs_diff_ge_1 x y < 1 :=
  by sorry

end expected_value_correct_prob_abs_diff_ge_1_correct_l141_141059


namespace solve_abs_eq_l141_141026

theorem solve_abs_eq (x : ℝ) : |x - 3| = 5 - 2x ↔ (x = 2 ∨ x = 8/3) :=
by sorry

end solve_abs_eq_l141_141026


namespace product_of_x1_to_x13_is_zero_l141_141285

theorem product_of_x1_to_x13_is_zero
  (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ)
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 :=
sorry

end product_of_x1_to_x13_is_zero_l141_141285


namespace compute_expression_l141_141268

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := 2 * x
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

theorem compute_expression : 
  f (g_inv (f_inv (f_inv (g (f 15))))) = 18 := by
  sorry

end compute_expression_l141_141268


namespace linda_original_amount_l141_141549

-- Define the original amount of money Lucy and Linda have
variables (L : ℕ) (lucy_initial : ℕ := 20)

-- Condition: If Lucy gives Linda $5, they have the same amount of money.
def condition := (lucy_initial - 5) = (L + 5)

-- Theorem: The original amount of money that Linda had
theorem linda_original_amount (h : condition L) : L = 10 := 
sorry

end linda_original_amount_l141_141549


namespace second_planner_cheaper_l141_141433

theorem second_planner_cheaper (x : ℕ) :
  (∀ x, 250 + 15 * x < 150 + 18 * x → x ≥ 34) :=
by
  intros x h
  sorry

end second_planner_cheaper_l141_141433


namespace arithmetic_mean_of_fractions_l141_141638

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l141_141638


namespace ratio_of_side_lengths_l141_141161

theorem ratio_of_side_lengths (a b c : ℕ) (h : a * a * b * b = 18 * c * c * 50 * c * c) :
  (12 = 1800000) ->  (15 = 1500) -> (10 > 0):=
by
  sorry

end ratio_of_side_lengths_l141_141161


namespace power_inequality_l141_141572

theorem power_inequality 
( a b : ℝ )
( h1 : 0 < a )
( h2 : 0 < b )
( h3 : a ^ 1999 + b ^ 2000 ≥ a ^ 2000 + b ^ 2001 ) :
  a ^ 2000 + b ^ 2000 ≤ 2 :=
sorry

end power_inequality_l141_141572


namespace hyperbola_focus_coordinates_l141_141861

theorem hyperbola_focus_coordinates : 
  ∃ (x y : ℝ), -2 * x^2 + 3 * y^2 + 8 * x - 18 * y - 8 = 0 ∧ (x, y) = (2, 7.5) :=
sorry

end hyperbola_focus_coordinates_l141_141861


namespace change_in_nickels_l141_141140

theorem change_in_nickels (cost_bread cost_cheese given_amount : ℝ) (quarters dimes : ℕ) (nickel_value : ℝ) 
  (h1 : cost_bread = 4.2) (h2 : cost_cheese = 2.05) (h3 : given_amount = 7.0)
  (h4 : quarters = 1) (h5 : dimes = 1) (hnickel_value : nickel_value = 0.05) : 
  ∃ n : ℕ, n = 8 :=
by
  sorry

end change_in_nickels_l141_141140


namespace books_in_series_l141_141814

theorem books_in_series (books_watched : ℕ) (movies_watched : ℕ) (read_more_movies_than_books : books_watched + 3 = movies_watched) (watched_movies : movies_watched = 19) : books_watched = 16 :=
by sorry

end books_in_series_l141_141814


namespace sum_of_coordinates_reflection_l141_141789

theorem sum_of_coordinates_reflection (y : ℝ) :
  let A := (3, y)
  let B := (3, -y)
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  let A := (3, y)
  let B := (3, -y)
  sorry

end sum_of_coordinates_reflection_l141_141789


namespace calculate_expression_l141_141966

theorem calculate_expression : 
  (12 * 0.5 * 3 * 0.0625 - 1.5) = -3 / 8 := 
by 
  sorry 

end calculate_expression_l141_141966


namespace roger_individual_pouches_per_pack_l141_141146

variable (members : ℕ) (coaches : ℕ) (helpers : ℕ) (packs : ℕ)

-- Given conditions
def total_people (members coaches helpers : ℕ) : ℕ := members + coaches + helpers
def pouches_per_pack (total_people packs : ℕ) : ℕ := total_people / packs

-- Specific values from the problem
def roger_total_people : ℕ := total_people 13 3 2
def roger_packs : ℕ := 3

-- The problem statement to prove:
theorem roger_individual_pouches_per_pack : pouches_per_pack roger_total_people roger_packs = 6 :=
by
  sorry

end roger_individual_pouches_per_pack_l141_141146


namespace arithmetic_mean_eq_l141_141654

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l141_141654


namespace original_number_l141_141822

theorem original_number 
  (x : ℝ)
  (h₁ : 0 < x)
  (h₂ : 1000 * x = 3 * (1 / x)) : 
  x = (Real.sqrt 30) / 100 :=
sorry

end original_number_l141_141822


namespace max_value_quadratic_expression_l141_141041

theorem max_value_quadratic_expression : ∃ x : ℝ, -3 * x^2 + 18 * x - 5 ≤ 22 ∧ ∀ y : ℝ, -3 * y^2 + 18 * y - 5 ≤ 22 := 
by 
  sorry

end max_value_quadratic_expression_l141_141041


namespace arithmetic_mean_of_fractions_l141_141631

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l141_141631


namespace total_batteries_produced_l141_141488

def time_to_gather_materials : ℕ := 6 -- in minutes
def time_to_create_battery : ℕ := 9   -- in minutes
def num_robots : ℕ := 10
def total_time : ℕ := 5 * 60 -- in minutes (5 hours * 60 minutes/hour)

theorem total_batteries_produced :
  total_time / (time_to_gather_materials + time_to_create_battery) * num_robots = 200 :=
by
  -- Placeholder for the proof steps
  sorry

end total_batteries_produced_l141_141488


namespace homework_total_time_l141_141435

theorem homework_total_time :
  ∀ (j g p : ℕ),
  j = 18 →
  g = j - 6 →
  p = 2 * g - 4 →
  j + g + p = 50 :=
by
  intros j g p h1 h2 h3
  sorry

end homework_total_time_l141_141435


namespace correct_operation_l141_141050

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l141_141050


namespace parabola_sum_vertex_point_l141_141926

theorem parabola_sum_vertex_point
  (a b c : ℝ)
  (h_vertex : ∀ y : ℝ, y = -6 → x = a * (y + 6)^2 + 8)
  (h_point : x = a * ((-4) + 6)^2 + 8)
  (ha : a = 0.5)
  (hb : b = 6)
  (hc : c = 26) :
  a + b + c = 32.5 :=
by
  sorry

end parabola_sum_vertex_point_l141_141926


namespace find_k_l141_141206

-- Auxiliary function to calculate the product of the digits of a number
def productOfDigits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (λ acc d => acc * d) 1

theorem find_k (k : ℕ) (h1 : 0 < k) (h2 : productOfDigits k = (25 * k) / 8 - 211) : 
  k = 72 ∨ k = 88 :=
by
  sorry

end find_k_l141_141206


namespace arithmetic_mean_of_fractions_l141_141619

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l141_141619


namespace x_intercepts_of_parabola_l141_141390

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end x_intercepts_of_parabola_l141_141390


namespace quadratic_root_conditions_l141_141033

theorem quadratic_root_conditions (a b : ℝ)
    (h1 : ∃ k : ℝ, ∀ x : ℝ, x^2 + 2 * x + 3 - k = 0)
    (h2 : ∀ α β : ℝ, α * β = 3 - k ∧ k^2 = α * β + 3 * k) : 
    k = 3 := 
sorry

end quadratic_root_conditions_l141_141033


namespace arithmetic_geometric_ratio_l141_141871

variables {a : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem arithmetic_geometric_ratio {a : ℕ → ℝ} {d : ℝ} (h1 : is_arithmetic_sequence a d)
  (h2 : a 9 ≠ a 3) (h3 : is_geometric_sequence (a 1) (a 3) (a 9)):
  (a 2 + a 4 + a 10) / (a 1 + a 3 + a 9) = 16 / 13 :=
sorry

end arithmetic_geometric_ratio_l141_141871


namespace tan_add_formula_l141_141367

noncomputable def tan_subtract (a b : ℝ) : ℝ := (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b)
noncomputable def tan_add (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

theorem tan_add_formula (α : ℝ) (hf : tan_subtract α (Real.pi / 4) = 1 / 4) :
  tan_add α (Real.pi / 4) = -4 :=
by
  sorry

end tan_add_formula_l141_141367


namespace product_of_solutions_l141_141982

theorem product_of_solutions :
  let solutions := {x : ℝ | |x| = 3 * (|x| - 2)} in
  ∏ x in solutions, x = -9 := by
  sorry

end product_of_solutions_l141_141982


namespace alyssa_money_after_movies_and_carwash_l141_141692

theorem alyssa_money_after_movies_and_carwash : 
  ∀ (allowance spent earned : ℕ), 
  allowance = 8 → 
  spent = allowance / 2 → 
  earned = 8 → 
  (allowance - spent + earned = 12) := 
by 
  intros allowance spent earned h_allowance h_spent h_earned 
  rw [h_allowance, h_spent, h_earned] 
  simp 
  sorry

end alyssa_money_after_movies_and_carwash_l141_141692


namespace range_of_a_l141_141399

noncomputable def f (x a : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x
noncomputable def f' (x a : ℝ) : ℝ := 1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ 0) ↔ -1 / 3 ≤ a ∧ a ≤ 1 / 3 :=
sorry

end range_of_a_l141_141399


namespace platform_length_l141_141501

theorem platform_length (train_length : ℕ) (pole_time : ℕ) (platform_time : ℕ) (V : ℕ) (L : ℕ)
  (h_train_length : train_length = 500)
  (h_pole_time : pole_time = 50)
  (h_platform_time : platform_time = 100)
  (h_speed : V = train_length / pole_time)
  (h_platform_distance : V * platform_time = train_length + L) : 
  L = 500 := 
sorry

end platform_length_l141_141501


namespace max_odd_integers_chosen_l141_141188

theorem max_odd_integers_chosen (a b c d e f : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h_prod_even : a * b * c * d * e * f % 2 = 0) : 
  (∀ n : ℕ, n = 5 → ∃ a b c d e, (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1) ∧ f % 2 = 0) :=
sorry

end max_odd_integers_chosen_l141_141188


namespace find_BM_length_l141_141240

variables (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)

-- Conditions
def condition1 : Prop := MA + (BC - BM) = 2 * CA
def condition2 : Prop := MA = x
def condition3 : Prop := CA = d
def condition4 : Prop := BC = h

-- The proof problem statement
theorem find_BM_length (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)
  (h1 : condition1 MA CA BC BM)
  (h2 : condition2 MA x)
  (h3 : condition3 CA d)
  (h4 : condition4 BC h) :
  BM = 2 * d :=
sorry

end find_BM_length_l141_141240


namespace prob_A_more_than_B_in_one_round_prob_A_more_than_B_in_at_least_two_of_three_rounds_l141_141896

def prob_A_hitting_8 := 0.6
def prob_A_hitting_9 := 0.3
def prob_A_hitting_10 := 0.1

def prob_B_hitting_8 := 0.4
def prob_B_hitting_9 := 0.4
def prob_B_hitting_10 := 0.2

-- Part (I): Probability that A hits more rings than B in a single round
theorem prob_A_more_than_B_in_one_round :
  let P_A := prob_A_hitting_9 * prob_B_hitting_8 + 
              prob_A_hitting_10 * prob_B_hitting_8 +
              prob_A_hitting_10 * prob_B_hitting_9 in
  P_A = 0.2 := by sorry

-- Part (II): Probability that in three rounds, A hits more rings than B in at least two rounds
theorem prob_A_more_than_B_in_at_least_two_of_three_rounds :
  let P_A := 0.2 in
  let P_C1 := 3 * P_A ^ 2 * (1 - P_A) in
  let P_C2 := P_A ^ 3 in
  P_C1 + P_C2 = 0.104 := by sorry

end prob_A_more_than_B_in_one_round_prob_A_more_than_B_in_at_least_two_of_three_rounds_l141_141896


namespace fraction_of_network_advertisers_l141_141943

theorem fraction_of_network_advertisers 
  (total_advertisers : ℕ := 20) 
  (percentage_from_uni_a : ℝ := 0.75)
  (advertisers_from_uni_a := total_advertisers * percentage_from_uni_a) :
  (advertisers_from_uni_a / total_advertisers) = (3 / 4) :=
by
  sorry

end fraction_of_network_advertisers_l141_141943


namespace remainder_of_sum_l141_141748

theorem remainder_of_sum (x y z : ℕ) (h1 : x % 15 = 6) (h2 : y % 15 = 9) (h3 : z % 15 = 3) : 
  (x + y + z) % 15 = 3 := 
  sorry

end remainder_of_sum_l141_141748


namespace tangent_parallel_to_line_l141_141810

def f (x : ℝ) : ℝ := x ^ 3 + x - 2

theorem tangent_parallel_to_line (x y : ℝ) : 
  (y = 4 * x - 1) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by
  sorry

end tangent_parallel_to_line_l141_141810


namespace seating_arrangements_l141_141868

theorem seating_arrangements (p : Fin 5 → Fin 5 → Prop) :
  (∃! i j : Fin 5, p i j ∧ i = j) →
  (∃! i j : Fin 5, p i j ∧ i ≠ j) →
  ∃ ways : ℕ,
  ways = 20 :=
by
  sorry

end seating_arrangements_l141_141868


namespace solve_quadratic_inequalities_find_values_a_c_l141_141878

open Polynomial

theorem solve_quadratic_inequalities :
  (∀ x : ℝ, -6 * x^2 + (6 + b) * x - b ≥ 0) ↔
    (if b > 6 then (1, b / 6) else if b = 6 then {1} else (b / 6, 1)) :=
by sorry

theorem find_values_a_c (a c : ℝ) :
  (∀ x : ℝ, a * x^2 + 5 * x + c > 0 ↔ (1 / 3 < x ∧ x < 1 / 2)) →
  a = -6 ∧ c = -1 :=
by sorry

end solve_quadratic_inequalities_find_values_a_c_l141_141878


namespace total_batteries_produced_l141_141487

def time_to_gather_materials : ℕ := 6 -- in minutes
def time_to_create_battery : ℕ := 9   -- in minutes
def num_robots : ℕ := 10
def total_time : ℕ := 5 * 60 -- in minutes (5 hours * 60 minutes/hour)

theorem total_batteries_produced :
  total_time / (time_to_gather_materials + time_to_create_battery) * num_robots = 200 :=
by
  -- Placeholder for the proof steps
  sorry

end total_batteries_produced_l141_141487


namespace interest_rate_correct_l141_141306

theorem interest_rate_correct :
  let SI := 155
  let P := 810
  let T := 4
  let R := SI * 100 / (P * T)
  R = 155 * 100 / (810 * 4) := 
sorry

end interest_rate_correct_l141_141306


namespace part1_part2_a_part2_b_part2_c_l141_141378

noncomputable def f (x a : ℝ) := Real.exp x - x - a

theorem part1 (x : ℝ) : f x 0 > x := 
by 
  -- here would be the proof
  sorry

theorem part2_a (a : ℝ) : a > 1 → ∃ z₁ z₂ : ℝ, f z₁ a = 0 ∧ f z₂ a = 0 ∧ z₁ ≠ z₂ := 
by 
  -- here would be the proof
  sorry

theorem part2_b (a : ℝ) : a < 1 → ¬ (∃ z : ℝ, f z a = 0) := 
by 
  -- here would be the proof
  sorry

theorem part2_c : f 0 1 = 0 := 
by 
  -- here would be the proof
  sorry

end part1_part2_a_part2_b_part2_c_l141_141378


namespace negation_statement_l141_141808

theorem negation_statement (h : ∀ x : ℝ, |x - 2| + |x - 4| > 3) : 
  ∃ x0 : ℝ, |x0 - 2| + |x0 - 4| ≤ 3 :=
sorry

end negation_statement_l141_141808


namespace find_subtracted_number_l141_141509

theorem find_subtracted_number 
  (a : ℕ) (b : ℕ) (g : ℕ) (n : ℕ) 
  (h1 : a = 2) 
  (h2 : b = 3 * a) 
  (h3 : g = 2 * b - n) 
  (h4 : g = 8) : n = 4 :=
by 
  sorry

end find_subtracted_number_l141_141509


namespace percentage_of_x_is_y_l141_141395

theorem percentage_of_x_is_y (x y : ℝ) (h : 0.5 * (x - y) = 0.4 * (x + y)) : y = 0.1111 * x := 
sorry

end percentage_of_x_is_y_l141_141395


namespace num_games_played_l141_141812

theorem num_games_played (n : ℕ) (h : n = 14) : (n.choose 2) = 91 :=
by
  sorry

end num_games_played_l141_141812


namespace find_2016th_smallest_n_l141_141422

def a_n (n : ℕ) : ℕ := sorry -- Placeholder for the actual relevant function

def satisfies_condition (n : ℕ) : Prop :=
  a_n n ≡ 1 [MOD 5]

theorem find_2016th_smallest_n :
  (∃ n, ∀ m < 2016, satisfies_condition m → satisfies_condition n ∧ n < m)
  ∧ 
  satisfies_condition 2016 → 
  2016 = 475756 :=
sorry

end find_2016th_smallest_n_l141_141422


namespace length_of_angle_bisector_l141_141599

theorem length_of_angle_bisector (AB AC : ℝ) (angleBAC : ℝ) (AD : ℝ) :
  AB = 6 → AC = 3 → angleBAC = 60 → AD = 2 * Real.sqrt 3 :=
by
  intro hAB hAC hAngleBAC
  -- Consider adding proof steps here in the future
  sorry

end length_of_angle_bisector_l141_141599


namespace compare_a_b_c_l141_141537

def a : ℝ := 2^(1/2)
def b : ℝ := 3^(1/3)
def c : ℝ := 5^(1/5)

theorem compare_a_b_c : b > a ∧ a > c :=
  by
  sorry

end compare_a_b_c_l141_141537


namespace sequence_sum_l141_141371

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 0 < a n)
  → (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) 
  → (∀ n : ℕ, a (n+1)^2 = a n * a (n+2))
  → S 3 = 13
  → a 1 = 1
  → (a 3 + a 4) / (a 1 + a 2) = 9 :=
sorry

end sequence_sum_l141_141371


namespace new_average_of_remaining_numbers_l141_141706

theorem new_average_of_remaining_numbers (sum_12 avg_12 n1 n2 : ℝ) 
  (h1 : avg_12 = 90)
  (h2 : sum_12 = 1080)
  (h3 : n1 = 80)
  (h4 : n2 = 85)
  : (sum_12 - n1 - n2) / 10 = 91.5 := 
by
  sorry

end new_average_of_remaining_numbers_l141_141706


namespace first_competitor_hotdogs_l141_141753

theorem first_competitor_hotdogs (x y z : ℕ) (h1 : y = 3 * x) (h2 : z = 2 * y) (h3 : z * 5 = 300) : x = 10 :=
sorry

end first_competitor_hotdogs_l141_141753


namespace banker_l141_141282

-- Define the given conditions
def present_worth : ℝ := 400
def interest_rate : ℝ := 0.10
def time_period : ℕ := 3

-- Define the amount due in the future
def amount_due (PW : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PW * (1 + r) ^ n

-- Define the banker's gain
def bankers_gain (A PW : ℝ) : ℝ :=
  A - PW

-- State the theorem we need to prove
theorem banker's_gain_is_correct :
  bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 132.4 :=
by sorry

end banker_l141_141282


namespace area_ratio_of_region_A_and_C_l141_141578

theorem area_ratio_of_region_A_and_C
  (pA : ℕ) (pC : ℕ) 
  (hA : pA = 16)
  (hC : pC = 24) :
  let sA := pA / 4
  let sC := pC / 6
  let areaA := sA * sA
  let areaC := (3 * Real.sqrt 3 / 2) * sC * sC
  (areaA / areaC) = (2 * Real.sqrt 3 / 9) :=
by
  sorry

end area_ratio_of_region_A_and_C_l141_141578


namespace fraction_pow_zero_l141_141819

theorem fraction_pow_zero :
  let a := 7632148
  let b := -172836429
  (a / b ≠ 0) → (a / b)^0 = 1 := by
  sorry

end fraction_pow_zero_l141_141819


namespace smallest_positive_m_l141_141346

theorem smallest_positive_m (m : ℕ) (h : ∃ n : ℤ, m^3 - 90 = n * (m + 9)) : m = 12 :=
by
  sorry

end smallest_positive_m_l141_141346


namespace probability_both_red_l141_141243

def initial_red_buttons : ℕ := 5
def initial_blue_buttons : ℕ := 10
def total_initial_buttons : ℕ := initial_red_buttons + initial_blue_buttons

def final_buttons_in_A : ℕ := (3 * total_initial_buttons) / 5

def removed_buttons : ℕ := total_initial_buttons - final_buttons_in_A

-- Carla removes the same number of red and blue buttons
axiom removed_red_buttons : ℕ 
axiom removed_blue_buttons : ℕ 

-- After removal, Jar A has the remaining buttons
def remaining_red_buttons_in_A : ℕ := initial_red_buttons - removed_red_buttons
def remaining_blue_buttons_in_A : ℕ := initial_blue_buttons - removed_blue_buttons
def total_remaining_buttons_in_A : ℕ := remaining_red_buttons_in_A + remaining_blue_buttons_in_A

-- Jar B would contain the removed buttons
def red_buttons_in_B : ℕ := removed_red_buttons
def blue_buttons_in_B : ℕ := removed_blue_buttons
def total_buttons_in_B : ℕ := red_buttons_in_B + blue_buttons_in_B

-- Probabilities of drawing red buttons
def prob_red_A : ℚ := remaining_red_buttons_in_A / total_remaining_buttons_in_A
def prob_red_B : ℚ := red_buttons_in_B / total_buttons_in_B

-- Assertion that the final probability that both selected buttons are red is 1/9
theorem probability_both_red :
  total_initial_buttons = 15 → 
  final_buttons_in_A = 9 → 
  total_remaining_buttons_in_A = 9 → 
  removed_buttons = 6 → 
  red_buttons_in_B = removed_red_buttons → 
  blue_buttons_in_B = removed_blue_buttons → 
  removed_red_buttons + removed_blue_buttons = 6 → 
  prob_red_A = 1 / 3 → 
  prob_red_B = 1 / 3 → 
  prob_red_A * prob_red_B = 1 / 9 :=
  by
    intros,
    sorry

end probability_both_red_l141_141243


namespace max_value_trig_expression_l141_141354

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end max_value_trig_expression_l141_141354


namespace min_N_of_block_viewed_l141_141496

theorem min_N_of_block_viewed (x y z N : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_factor : (x - 1) * (y - 1) * (z - 1) = 231) : 
  N = x * y * z ∧ N = 384 :=
by {
  sorry 
}

end min_N_of_block_viewed_l141_141496


namespace half_abs_diff_squares_l141_141170

theorem half_abs_diff_squares (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  (1 / 2) * |a^2 - b^2| = 112.5 :=
sorry

end half_abs_diff_squares_l141_141170


namespace gather_all_candies_l141_141035

theorem gather_all_candies (n : ℕ) (h₁ : n ≥ 4) (candies : ℕ) (h₂ : candies ≥ 4)
    (plates : Fin n → ℕ) :
    ∃ plate : Fin n, ∀ i : Fin n, i ≠ plate → plates i = 0 :=
sorry

end gather_all_candies_l141_141035


namespace sculpture_cost_NAD_to_CNY_l141_141786

def NAD_to_USD (nad : ℕ) : ℕ := nad / 8
def USD_to_CNY (usd : ℕ) : ℕ := usd * 5

theorem sculpture_cost_NAD_to_CNY (nad : ℕ) : (nad = 160) → (USD_to_CNY (NAD_to_USD nad) = 100) :=
by
  intro h1
  rw [h1]
  -- NAD_to_USD 160 = 160 / 8
  have h2 : NAD_to_USD 160 = 20 := rfl
  -- USD_to_CNY 20 = 20 * 5
  have h3 : USD_to_CNY 20 = 100 := rfl
  -- Concluding the theorem
  rw [h2, h3]
  reflexivity

end sculpture_cost_NAD_to_CNY_l141_141786


namespace probability_no_shaded_in_2_by_2004_l141_141949

noncomputable def probability_no_shaded_rectangle (total_rectangles shaded_rectangles : Nat) : ℚ :=
  1 - (shaded_rectangles : ℚ) / (total_rectangles : ℚ)

theorem probability_no_shaded_in_2_by_2004 :
  let rows := 2
  let cols := 2004
  let total_rectangles := (cols + 1) * cols / 2 * rows
  let shaded_rectangles := 501 * 2507 
  probability_no_shaded_rectangle total_rectangles shaded_rectangles = 1501 / 4008 :=
by
  sorry

end probability_no_shaded_in_2_by_2004_l141_141949


namespace car_speed_l141_141941

theorem car_speed (v : ℝ) (hv : (1 / v * 3600) = (1 / 40 * 3600) + 10) : v = 36 := 
by
  sorry

end car_speed_l141_141941


namespace house_trailer_payment_difference_l141_141415

-- Define the costs and periods
def cost_house : ℕ := 480000
def cost_trailer : ℕ := 120000
def loan_period_years : ℕ := 20
def months_per_year : ℕ := 12

-- Calculate total months
def total_months : ℕ := loan_period_years * months_per_year

-- Calculate monthly payments
def monthly_payment_house : ℕ := cost_house / total_months
def monthly_payment_trailer : ℕ := cost_trailer / total_months

-- Theorem stating the difference in monthly payments
theorem house_trailer_payment_difference :
  monthly_payment_house - monthly_payment_trailer = 1500 := by sorry

end house_trailer_payment_difference_l141_141415


namespace difference_of_squares_example_l141_141821

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 123) (h2 : b = 23) : a^2 - b^2 = 14600 :=
by
  rw [h1, h2]
  sorry

end difference_of_squares_example_l141_141821


namespace fourth_term_of_geometric_sequence_l141_141592

theorem fourth_term_of_geometric_sequence (a₁ : ℝ) (a₆ : ℝ) (a₄ : ℝ) (r : ℝ)
  (h₁ : a₁ = 1000)
  (h₂ : a₆ = a₁ * r^5)
  (h₃ : a₆ = 125)
  (h₄ : a₄ = a₁ * r^3) : 
  a₄ = 125 :=
sorry

end fourth_term_of_geometric_sequence_l141_141592


namespace minimum_value_function_l141_141986

theorem minimum_value_function :
  ∀ x : ℝ, x ≥ 0 → (∃ y : ℝ, y = (3 * x^2 + 9 * x + 20) / (7 * (2 + x)) ∧
    (∀ z : ℝ, z ≥ 0 → (3 * z^2 + 9 * z + 20) / (7 * (2 + z)) ≥ y)) ∧
    (∃ x0 : ℝ, x0 = 0 ∧ y = (3 * x0^2 + 9 * x0 + 20) / (7 * (2 + x0)) ∧ y = 10 / 7) :=
by
  sorry

end minimum_value_function_l141_141986


namespace dice_sum_probability_l141_141938

theorem dice_sum_probability (n : ℕ) (h : ∃ k : ℕ, (8 : ℕ) * k + k = 12) : n = 330 :=
sorry

end dice_sum_probability_l141_141938


namespace power_mod_result_l141_141657

-- Define the modulus and base
def mod : ℕ := 8
def base : ℕ := 7
def exponent : ℕ := 202

-- State the theorem
theorem power_mod_result :
  (base ^ exponent) % mod = 1 :=
by
  sorry

end power_mod_result_l141_141657


namespace min_points_condition_met_l141_141044

noncomputable def min_points_on_circle (L : ℕ) : ℕ := 1304

theorem min_points_condition_met (L : ℕ) (hL : L = 1956) :
  (∀ (points : ℕ → ℕ), (∀ n, points n ≠ points (n + 1) ∧ points n ≠ points (n + 2)) ∧ (∀ n, points n < L)) →
  min_points_on_circle L = 1304 :=
by
  -- Proof steps omitted
  sorry

end min_points_condition_met_l141_141044


namespace arithmetic_mean_of_fractions_l141_141642

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l141_141642


namespace arithmetic_mean_eq_l141_141650

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l141_141650


namespace parabola_x_intercepts_count_l141_141388

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end parabola_x_intercepts_count_l141_141388


namespace parabola_x_intercepts_count_l141_141386

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end parabola_x_intercepts_count_l141_141386


namespace percentage_reduction_in_price_l141_141570

noncomputable def original_price_per_mango : ℝ := 416.67 / 125

noncomputable def original_num_mangoes : ℝ := 360 / original_price_per_mango

def additional_mangoes : ℝ := 12

noncomputable def new_num_mangoes : ℝ := original_num_mangoes + additional_mangoes

noncomputable def new_price_per_mango : ℝ := 360 / new_num_mangoes

noncomputable def percentage_reduction : ℝ := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100

theorem percentage_reduction_in_price : percentage_reduction = 10 := by
  sorry

end percentage_reduction_in_price_l141_141570


namespace correct_operation_l141_141048

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := 
by 
  sorry

end correct_operation_l141_141048


namespace odd_f_even_g_fg_eq_g_increasing_min_g_sum_l141_141224

noncomputable def f (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x - (2:ℝ)^(-x))
noncomputable def g (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x + (2:ℝ)^(-x))

theorem odd_f (x : ℝ) : f (-x) = -f (x) := sorry
theorem even_g (x : ℝ) : g (-x) = g (x) := sorry
theorem fg_eq (x : ℝ) : f (x) + g (x) = (2:ℝ)^x := sorry
theorem g_increasing (x : ℝ) : x ≥ 0 → ∀ y, 0 ≤ y ∧ y < x → g y < g x := sorry
theorem min_g_sum (x : ℝ) : ∃ t, t ≥ 2 ∧ (g x + g (2 * x) = 2) := sorry

end odd_f_even_g_fg_eq_g_increasing_min_g_sum_l141_141224


namespace hire_charges_paid_by_B_l141_141473

theorem hire_charges_paid_by_B (total_cost : ℝ) (hours_A hours_B hours_C : ℝ) (b_payment : ℝ) :
  total_cost = 720 ∧ hours_A = 9 ∧ hours_B = 10 ∧ hours_C = 13 ∧ b_payment = (total_cost / (hours_A + hours_B + hours_C)) * hours_B → b_payment = 225 :=
by
  sorry

end hire_charges_paid_by_B_l141_141473


namespace problem_I_l141_141479

theorem problem_I (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : 
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 := 
by
  sorry

end problem_I_l141_141479


namespace one_meter_eq_jumps_l141_141441

theorem one_meter_eq_jumps 
  (x y a b p q s t : ℝ) 
  (h1 : x * hops = y * skips)
  (h2 : a * jumps = b * hops)
  (h3 : p * skips = q * leaps)
  (h4 : s * leaps = t * meters) :
  1 * meters = (sp * x * a / (tq * y * b)) * jumps :=
sorry

end one_meter_eq_jumps_l141_141441


namespace ratio_of_perimeter_to_b_l141_141499

theorem ratio_of_perimeter_to_b (b : ℝ) (hb : b ≠ 0) :
  let p1 := (-2*b, -2*b)
  let p2 := (2*b, -2*b)
  let p3 := (2*b, 2*b)
  let p4 := (-2*b, 2*b)
  let l := (y = b * x)
  let d1 := 4*b
  let d2 := 4*b
  let d3 := 4*b
  let d4 := 4*b*Real.sqrt 2
  let perimeter := d1 + d2 + d3 + d4
  let ratio := perimeter / b
  ratio = 12 + 4 * Real.sqrt 2 := by
  -- Placeholder for proof
  sorry

end ratio_of_perimeter_to_b_l141_141499


namespace train_speed_is_72_kmh_l141_141500

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 175
noncomputable def crossing_time : ℝ := 14.248860091192705

theorem train_speed_is_72_kmh :
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

end train_speed_is_72_kmh_l141_141500


namespace calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l141_141988

theorem calculation_a_squared_plus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  a^2 + b^2 = 6 := by
  sorry

theorem calculation_a_minus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  (a - b)^2 = 8 := by
  sorry

end calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l141_141988


namespace flag_count_l141_141862

-- Definitions based on the conditions
def colors : ℕ := 3
def stripes : ℕ := 3

-- The main statement
theorem flag_count : colors ^ stripes = 27 :=
by
  sorry

end flag_count_l141_141862


namespace find_number_l141_141276

theorem find_number (x : ℝ) (h : (x - 5) / 3 = 4) : x = 17 :=
by {
  sorry
}

end find_number_l141_141276


namespace find_constants_l141_141207

theorem find_constants (P Q R : ℤ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (2 * x^2 - 5 * x + 6) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) →
  P = -6 ∧ Q = 8 ∧ R = -5 :=
by
  sorry

end find_constants_l141_141207


namespace partnership_total_profit_l141_141186

theorem partnership_total_profit
  (total_capital : ℝ)
  (A_share : ℝ := 1/3)
  (B_share : ℝ := 1/4)
  (C_share : ℝ := 1/5)
  (D_share : ℝ := 1 - (A_share + B_share + C_share))
  (A_profit : ℝ := 805)
  (A_capital : ℝ := total_capital * A_share)
  (total_capital_positive : 0 < total_capital)
  (shares_add_up : A_share + B_share + C_share + D_share = 1) :
  (A_profit / (total_capital * A_share)) * total_capital = 2415 :=
by
  -- Proof will go here.
  sorry

end partnership_total_profit_l141_141186


namespace M_inter_N_is_5_l141_141731

/-- Define the sets M and N. -/
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {2, 5, 8}

/-- Prove the intersection of M and N is {5}. -/
theorem M_inter_N_is_5 : M ∩ N = {5} :=
by
  sorry

end M_inter_N_is_5_l141_141731


namespace seventh_grade_male_students_l141_141066

theorem seventh_grade_male_students:
  ∃ x : ℤ, (48 = x + (4*x)/5 + 3) ∧ x = 25 :=
by
  sorry

end seventh_grade_male_students_l141_141066


namespace katya_minimum_problems_l141_141253

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l141_141253


namespace range_of_b_div_a_l141_141300

theorem range_of_b_div_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
(h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end range_of_b_div_a_l141_141300


namespace red_mushrooms_bill_l141_141964

theorem red_mushrooms_bill (R : ℝ) : 
  (2/3) * R + 6 + 3 = 17 → R = 12 :=
by
  intro h
  sorry

end red_mushrooms_bill_l141_141964


namespace find_divisor_l141_141490

theorem find_divisor :
  ∃ d : ℕ, (d = 859560) ∧ ∃ n : ℕ, (n + 859622) % d = 0 ∧ n = 859560 :=
by
  sorry

end find_divisor_l141_141490


namespace spherical_to_rectangular_coordinates_l141_141520

theorem spherical_to_rectangular_coordinates :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 5 / 2
:= by
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  have hx : x = (5 * Real.sqrt 6) / 4 := sorry
  have hy : y = (5 * Real.sqrt 6) / 4 := sorry
  have hz : z = 5 / 2 := sorry
  exact ⟨hx, hy, hz⟩

end spherical_to_rectangular_coordinates_l141_141520


namespace functional_square_for_all_n_l141_141137

theorem functional_square_for_all_n (f : ℕ → ℕ) :
  (∀ m n : ℕ, ∃ k : ℕ, (f m + n) * (m + f n) = k ^ 2) ↔ ∃ c : ℕ, ∀ n : ℕ, f n = n + c := 
sorry

end functional_square_for_all_n_l141_141137


namespace unique_solution_values_l141_141226

theorem unique_solution_values (a : ℝ) :
  (∃! x : ℝ, a * x^2 - x + 1 = 0) ↔ (a = 0 ∨ a = 1 / 4) :=
by
  sorry

end unique_solution_values_l141_141226


namespace blueberries_count_l141_141319

theorem blueberries_count (total_berries raspberries blackberries blueberries : ℕ)
  (h1 : total_berries = 42)
  (h2 : raspberries = total_berries / 2)
  (h3 : blackberries = total_berries / 3)
  (h4 : blueberries = total_berries - raspberries - blackberries) :
  blueberries = 7 :=
sorry

end blueberries_count_l141_141319


namespace composite_quotient_l141_141193

def first_eight_composites := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) := l.foldl (· * ·) 1

theorem composite_quotient :
  let numerator := product first_eight_composites
  let denominator := product next_eight_composites
  numerator / denominator = (1 : ℚ)/(1430 : ℚ) :=
by
  sorry

end composite_quotient_l141_141193
