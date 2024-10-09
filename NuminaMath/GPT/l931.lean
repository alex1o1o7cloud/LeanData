import Mathlib

namespace temperature_conversion_l931_93180

noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ :=
  (c * (9 / 5)) + 32

theorem temperature_conversion (c : ℝ) (hf : c = 60) :
  celsius_to_fahrenheit c = 140 :=
by {
  rw [hf, celsius_to_fahrenheit];
  norm_num
}

end temperature_conversion_l931_93180


namespace solve_inequality_l931_93175

theorem solve_inequality (x : Real) : 
  x^2 - 48 * x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 :=
by
  sorry

end solve_inequality_l931_93175


namespace complement_S_union_T_eq_l931_93196

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3 * x - 4 ≤ 0}
noncomputable def complement_S := {x : ℝ | x ≤ -2}

theorem complement_S_union_T_eq : (complement_S ∪ T) = {x : ℝ | x ≤ 1} := by 
  sorry

end complement_S_union_T_eq_l931_93196


namespace sequence_arithmetic_l931_93179

-- Define the sequence and sum conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (p : ℝ)

-- We are given that the sum of the first n terms is Sn = n * p * a_n
axiom sum_condition (n : ℕ) (hpos : n > 0) : S n = n * p * a n

-- Also, given that a_1 ≠ a_2
axiom a1_ne_a2 : a 1 ≠ a 2

-- Define what we need to prove
theorem sequence_arithmetic (n : ℕ) (hn : n ≥ 2) :
  ∃ (a2 : ℝ), p = 1/2 ∧ a n = (n-1) * a2 :=
by
  sorry

end sequence_arithmetic_l931_93179


namespace directrix_of_parabola_l931_93107

theorem directrix_of_parabola : 
  let y := 3 * x^2 - 6 * x + 1
  y = -25 / 12 :=
sorry

end directrix_of_parabola_l931_93107


namespace five_natural_numbers_increase_15_times_l931_93168

noncomputable def prod_of_decreased_factors_is_15_times_original (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  (a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * (a1 * a2 * a3 * a4 * a5)

theorem five_natural_numbers_increase_15_times {a1 a2 a3 a4 a5 : ℕ} :
  a1 * a2 * a3 * a4 * a5 = 48 → prod_of_decreased_factors_is_15_times_original a1 a2 a3 a4 a5 :=
by
  sorry

end five_natural_numbers_increase_15_times_l931_93168


namespace team_total_score_is_correct_l931_93129

-- Define the total number of team members
def total_members : ℕ := 30

-- Define the number of members who didn't show up
def members_absent : ℕ := 8

-- Define the score per member
def score_per_member : ℕ := 4

-- Define the points deducted per incorrect answer
def points_per_incorrect_answer : ℕ := 2

-- Define the total number of incorrect answers
def total_incorrect_answers : ℕ := 6

-- Define the bonus multiplier
def bonus_multiplier : ℝ := 1.5

-- Define the total score calculation
def total_score_calculation (total_members : ℕ) (members_absent : ℕ) (score_per_member : ℕ)
  (points_per_incorrect_answer : ℕ) (total_incorrect_answers : ℕ) (bonus_multiplier : ℝ) : ℝ :=
  let members_present := total_members - members_absent
  let initial_score := members_present * score_per_member
  let total_deductions := total_incorrect_answers * points_per_incorrect_answer
  let final_score := initial_score - total_deductions
  final_score * bonus_multiplier

-- Prove that the total score is 114 points
theorem team_total_score_is_correct : total_score_calculation total_members members_absent score_per_member
  points_per_incorrect_answer total_incorrect_answers bonus_multiplier = 114 :=
by
  sorry

end team_total_score_is_correct_l931_93129


namespace product_of_roots_l931_93165

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 9 * x + 20

-- The main statement for the Lean theorem
theorem product_of_roots : (∃ x₁ x₂ : ℝ, quadratic x₁ = 0 ∧ quadratic x₂ = 0 ∧ x₁ * x₂ = 20) :=
by
  sorry

end product_of_roots_l931_93165


namespace vector_magnitude_l931_93177

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l931_93177


namespace paths_from_A_to_B_l931_93174

def path_count_A_to_B : Nat :=
  let red_to_blue_ways := [2, 3]  -- 2 ways to first blue, 3 ways to second blue
  let blue_to_green_ways_first := 4 * 2  -- Each of the 2 green arrows from first blue, 4 ways each
  let blue_to_green_ways_second := 5 * 2 -- Each of the 2 green arrows from second blue, 5 ways each
  let green_to_B_ways_first := 2 * blue_to_green_ways_first  -- Each of the first green, 2 ways each
  let green_to_B_ways_second := 3 * blue_to_green_ways_second  -- Each of the second green, 3 ways each
  green_to_B_ways_first + green_to_B_ways_second  -- Total paths from green arrows to B

theorem paths_from_A_to_B : path_count_A_to_B = 46 := by
  sorry

end paths_from_A_to_B_l931_93174


namespace hcf_462_5_1_l931_93136

theorem hcf_462_5_1 (a b c : ℕ) (h₁ : a = 462) (h₂ : b = 5) (h₃ : c = 2310) (h₄ : Nat.lcm a b = c) : Nat.gcd a b = 1 := by
  sorry

end hcf_462_5_1_l931_93136


namespace solve_pairs_l931_93115

theorem solve_pairs (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (m, n) = (6, 3) ∨ (m, n) = (9, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end solve_pairs_l931_93115


namespace ketchup_bottles_count_l931_93159

def ratio_ketchup_mustard_mayo : Nat × Nat × Nat := (3, 3, 2)
def num_mayo_bottles : Nat := 4

theorem ketchup_bottles_count 
  (r : Nat × Nat × Nat)
  (m : Nat)
  (h : r = ratio_ketchup_mustard_mayo)
  (h2 : m = num_mayo_bottles) :
  ∃ k : Nat, k = 6 := by
sorry

end ketchup_bottles_count_l931_93159


namespace find_two_digit_number_l931_93182

def tens_digit (n: ℕ) := n / 10
def unit_digit (n: ℕ) := n % 10
def is_required_number (n: ℕ) : Prop :=
  tens_digit n + 2 = unit_digit n ∧ n < 30 ∧ 10 ≤ n

theorem find_two_digit_number (n : ℕ) :
  is_required_number n → n = 13 ∨ n = 24 :=
by
  -- Proof placeholder
  sorry

end find_two_digit_number_l931_93182


namespace part_a_part_b_l931_93103

-- Part (a): Prove that for N = a^2 + 2, the equation has positive integral solutions for infinitely many a.
theorem part_a (N : ℕ) (a : ℕ) (x y z t : ℕ) (hx : x = a * (a^2 + 2)) (hy : y = a) (hz : z = 1) (ht : t = 1) :
  (∃ (N : ℕ), ∀ (a : ℕ), ∃ (x y z t : ℕ),
    x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0) :=
sorry

-- Part (b): Prove that for N = 4^k(8m + 7), the equation has no positive integral solutions.
theorem part_b (N : ℕ) (k m : ℕ) (x y z t : ℕ) (hN : N = 4^k * (8 * m + 7)) :
  ¬ (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N) :=
sorry

end part_a_part_b_l931_93103


namespace correct_equation_is_x2_sub_10x_add_9_l931_93176

-- Define the roots found by Student A and Student B
def roots_A := (8, 2)
def roots_B := (-9, -1)

-- Define the incorrect equation by student A from given roots
def equation_A (x : ℝ) := x^2 - 10 * x + 16

-- Define the incorrect equation by student B from given roots
def equation_B (x : ℝ) := x^2 + 10 * x + 9

-- Define the correct quadratic equation
def correct_quadratic_equation (x : ℝ) := x^2 - 10 * x + 9

-- Theorem stating that the correct quadratic equation balances the errors of both students
theorem correct_equation_is_x2_sub_10x_add_9 :
  ∃ (eq_correct : ℝ → ℝ), 
    eq_correct = correct_quadratic_equation :=
by
  -- proof will go here
  sorry

end correct_equation_is_x2_sub_10x_add_9_l931_93176


namespace smallest_n_for_partition_condition_l931_93147

theorem smallest_n_for_partition_condition :
  ∃ n : ℕ, n = 4 ∧ ∀ T, (T = {i : ℕ | 2 ≤ i ∧ i ≤ n}) →
  (∀ A B, (T = A ∪ B ∧ A ∩ B = ∅) →
   (∃ a b c, (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B) ∧ (a + b = c))) := sorry

end smallest_n_for_partition_condition_l931_93147


namespace min_distance_l931_93198

variables {P Q : ℝ × ℝ}

def line (P : ℝ × ℝ) : Prop := 3 * P.1 + 4 * P.2 + 5 = 0
def circle (Q : ℝ × ℝ) : Prop := (Q.1 - 2) ^ 2 + (Q.2 - 2) ^ 2 = 4

theorem min_distance (P : ℝ × ℝ) (Q : ℝ × ℝ) (hP : line P) (hQ : circle Q) :
  ∃ d : ℝ, d = dist P Q ∧ d = 9 / 5 := sorry

end min_distance_l931_93198


namespace algebraic_expression_opposite_l931_93188

theorem algebraic_expression_opposite (a b x : ℝ) (h : b^2 * x^2 + |a| = -(b^2 * x^2 + |a|)) : a * b = 0 :=
by 
  sorry

end algebraic_expression_opposite_l931_93188


namespace peaches_eaten_correct_l931_93112

-- Given conditions
def total_peaches : ℕ := 18
def initial_ripe_peaches : ℕ := 4
def peaches_ripen_per_day : ℕ := 2
def days_passed : ℕ := 5
def ripe_unripe_difference : ℕ := 7

-- Definitions derived from conditions
def ripe_peaches_after_days := initial_ripe_peaches + peaches_ripen_per_day * days_passed
def unripe_peaches_initial := total_peaches - initial_ripe_peaches
def unripe_peaches_after_days := unripe_peaches_initial - peaches_ripen_per_day * days_passed
def actual_ripe_peaches_needed := unripe_peaches_after_days + ripe_unripe_difference
def peaches_eaten := ripe_peaches_after_days - actual_ripe_peaches_needed

-- Prove that the number of peaches eaten is equal to 3
theorem peaches_eaten_correct : peaches_eaten = 3 := by
  sorry

end peaches_eaten_correct_l931_93112


namespace fraction_checked_by_worker_y_l931_93111

variable (P : ℝ) -- Total number of products
variable (f_X f_Y : ℝ) -- Fraction of products checked by worker X and Y
variable (dx : ℝ) -- Defective rate for worker X
variable (dy : ℝ) -- Defective rate for worker Y
variable (dt : ℝ) -- Total defective rate

-- Conditions
axiom f_sum : f_X + f_Y = 1
axiom dx_val : dx = 0.005
axiom dy_val : dy = 0.008
axiom dt_val : dt = 0.0065

-- Proof
theorem fraction_checked_by_worker_y : f_Y = 1 / 2 :=
by
  sorry

end fraction_checked_by_worker_y_l931_93111


namespace Patriots_won_30_games_l931_93164

def Tigers_won_more_games_than_Eagles (games_tigers games_eagles : ℕ) : Prop :=
games_tigers > games_eagles

def Patriots_won_more_than_Cubs_less_than_Mounties (games_patriots games_cubs games_mounties : ℕ) : Prop :=
games_cubs < games_patriots ∧ games_patriots < games_mounties

def Cubs_won_more_than_20_games (games_cubs : ℕ) : Prop :=
games_cubs > 20

theorem Patriots_won_30_games (games_tigers games_eagles games_patriots games_cubs games_mounties : ℕ)  :
  Tigers_won_more_games_than_Eagles games_tigers games_eagles →
  Patriots_won_more_than_Cubs_less_than_Mounties games_patriots games_cubs games_mounties →
  Cubs_won_more_than_20_games games_cubs →
  ∃ games_patriots, games_patriots = 30 := 
by
  sorry

end Patriots_won_30_games_l931_93164


namespace percentage_of_men_l931_93148

theorem percentage_of_men (E M W : ℝ) 
  (h1 : M + W = E)
  (h2 : 0.5 * M + 0.1666666666666669 * W = 0.4 * E)
  (h3 : W = E - M) : 
  (M / E = 0.70) :=
by
  sorry

end percentage_of_men_l931_93148


namespace sphere_radius_vol_eq_area_l931_93134

noncomputable def volume (r : ℝ) : ℝ := (4/3) * Real.pi * r ^ 3
noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2

theorem sphere_radius_vol_eq_area (r : ℝ) :
  volume r = surface_area r → r = 3 :=
by
  sorry

end sphere_radius_vol_eq_area_l931_93134


namespace heather_average_balance_l931_93158

theorem heather_average_balance :
  let balance_J := 150
  let balance_F := 250
  let balance_M := 100
  let balance_A := 200
  let balance_May := 300
  let total_balance := balance_J + balance_F + balance_M + balance_A + balance_May
  let avg_balance := total_balance / 5
  avg_balance = 200 :=
by
  sorry

end heather_average_balance_l931_93158


namespace eval_expression_l931_93185

open Real

theorem eval_expression :
  (0.8^5 - (0.5^6 / 0.8^4) + 0.40 + 0.5^3 - log 0.3 + sin (π / 6)) = 2.51853302734375 :=
  sorry

end eval_expression_l931_93185


namespace problem_statement_l931_93195

theorem problem_statement (a b c : ℝ) (h1 : a ∈ Set.Ioi 0) (h2 : b ∈ Set.Ioi 0) (h3 : c ∈ Set.Ioi 0) (h4 : a^2 + b^2 + c^2 = 3) : 
  1 / (2 - a) + 1 / (2 - b) + 1 / (2 - c) ≥ 3 := 
sorry

end problem_statement_l931_93195


namespace part1_part2_l931_93144

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- Part 1: Prove the range of k such that f(x) < k * x for all x
theorem part1 (k : ℝ) : (∀ x : ℝ, x > 0 → f x < k * x) ↔ k > 1 / (2 * Real.exp 1) :=
by sorry

-- Part 2: Define the function g(x) = f(x) - k * x and prove the range of k for which g(x) has two zeros in the interval [1/e, e^2]
noncomputable def g (x k : ℝ) : ℝ := f x - k * x

theorem part2 (k : ℝ) : (∃ x1 x2 : ℝ, 1 / Real.exp 1 ≤ x1 ∧ x1 ≤ Real.exp 2 ∧
                                 1 / Real.exp 1 ≤ x2 ∧ x2 ≤ Real.exp 2 ∧
                                 g x1 k = 0 ∧ g x2 k = 0 ∧ x1 ≠ x2)
                               ↔ 2 / (Real.exp 4) ≤ k ∧ k < 1 / (2 * Real.exp 1) :=
by sorry

end part1_part2_l931_93144


namespace necessary_and_sufficient_condition_l931_93102

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (a^2 + 4 * a - 5 > 0) ↔ (|a + 2| > 3) := sorry

end necessary_and_sufficient_condition_l931_93102


namespace diameter_of_circle_l931_93139

theorem diameter_of_circle (a b : ℕ) (r : ℝ) (h_a : a = 6) (h_b : b = 8) (h_triangle : a^2 + b^2 = r^2) : r = 10 :=
by 
  rw [h_a, h_b] at h_triangle
  sorry

end diameter_of_circle_l931_93139


namespace find_D_l931_93120

-- Definitions
variable (A B C D E F : ℕ)

-- Conditions
axiom sum_AB : A + B = 16
axiom sum_BC : B + C = 12
axiom sum_EF : E + F = 8
axiom total_sum : A + B + C + D + E + F = 18

-- Theorem statement
theorem find_D : D = 6 :=
by
  sorry

end find_D_l931_93120


namespace general_equation_M_range_distance_D_to_l_l931_93190

noncomputable def parametric_to_general (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  x^2 + y^2 / 4 = 1

noncomputable def distance_range (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  let l := x + y - 4
  let d := |x + 2 * y - 4| / Real.sqrt 2
  let min_dist := (4 * Real.sqrt 2 - Real.sqrt 10) / 2
  let max_dist := (4 * Real.sqrt 2 + Real.sqrt 10) / 2
  min_dist ≤ d ∧ d ≤ max_dist

theorem general_equation_M (θ : ℝ) : parametric_to_general θ := sorry

theorem range_distance_D_to_l (θ : ℝ) : distance_range θ := sorry

end general_equation_M_range_distance_D_to_l_l931_93190


namespace equivalence_of_statements_l931_93171

theorem equivalence_of_statements 
  (Q P : Prop) :
  (Q → ¬ P) ↔ (P → ¬ Q) := sorry

end equivalence_of_statements_l931_93171


namespace monotonicity_intervals_f_above_g_l931_93178

noncomputable def f (x m : ℝ) := (Real.exp x) / (x^2 - m * x + 1)

theorem monotonicity_intervals (m : ℝ) (h : m ∈ Set.Ioo (-2 : ℝ) 2) :
  (m = 0 → ∀ x y : ℝ, x ≤ y → f x m ≤ f y m) ∧ 
  (0 < m ∧ m < 2 → ∀ x : ℝ, (x < 1 → f x m < f (x + 1) m) ∧
    (1 < x ∧ x < m + 1 → f x m > f (x + 1) m) ∧
    (x > m + 1 → f x m < f (x + 1) m)) ∧
  (-2 < m ∧ m < 0 → ∀ x : ℝ, (x < m + 1 → f x m < f (x + 1) m) ∧
    (m + 1 < x ∧ x < 1 → f x m > f (x + 1) m) ∧
    (x > 1 → f x m < f (x + 1) m)) :=
sorry

theorem f_above_g (m : ℝ) (hm : m ∈ Set.Ioo (0 : ℝ) (1/2 : ℝ)) (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (m + 1)) :
  f x m > x :=
sorry

end monotonicity_intervals_f_above_g_l931_93178


namespace minimum_area_rectangle_l931_93117

noncomputable def minimum_rectangle_area (a : ℝ) : ℝ :=
  if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
  else if a < 1 / 2 then 1 - 2 * a
  else 0

theorem minimum_area_rectangle (a : ℝ) :
  minimum_rectangle_area a =
    if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
    else if a < 1 / 2 then 1 - 2 * a
    else 0 :=
by
  sorry

end minimum_area_rectangle_l931_93117


namespace part1_part2_l931_93137

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 9 → x ∈ Set.Icc (-2 : ℝ) 4 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = -x^2 + a) →
  (a ∈ Set.Icc (19 / 4) (7 : ℝ)) :=
sorry

end part1_part2_l931_93137


namespace quadratic_equation_root_condition_l931_93197

theorem quadratic_equation_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, (a - 1) * x1^2 - 4 * x1 - 1 = 0 ∧ (a - 1) * x2^2 - 4 * x2 - 1 = 0) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end quadratic_equation_root_condition_l931_93197


namespace natasha_dimes_l931_93104

theorem natasha_dimes (n : ℕ) :
  100 < n ∧ n < 200 ∧
  n % 3 = 2 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2 ↔ n = 182 := by
sorry

end natasha_dimes_l931_93104


namespace max_2b_div_a_l931_93122

theorem max_2b_div_a (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : 
  ∃ max_val, max_val = (2 * b) / a ∧ max_val = (32 / 3) :=
by
  sorry

end max_2b_div_a_l931_93122


namespace minimum_value_at_x_eq_3_l931_93187

theorem minimum_value_at_x_eq_3 (b : ℝ) : 
  ∃ m : ℝ, (∀ x : ℝ, 3 * x^2 - 18 * x + b ≥ m) ∧ (3 * 3^2 - 18 * 3 + b = m) :=
by
  sorry

end minimum_value_at_x_eq_3_l931_93187


namespace radius_of_base_of_cone_is_3_l931_93161

noncomputable def radius_of_base_of_cone (θ R : ℝ) : ℝ :=
  ((θ / 360) * 2 * Real.pi * R) / (2 * Real.pi)

theorem radius_of_base_of_cone_is_3 :
  radius_of_base_of_cone 120 9 = 3 := 
by 
  simp [radius_of_base_of_cone]
  sorry

end radius_of_base_of_cone_is_3_l931_93161


namespace white_tshirts_l931_93149

theorem white_tshirts (packages shirts_per_package : ℕ) (h1 : packages = 71) (h2 : shirts_per_package = 6) : packages * shirts_per_package = 426 := 
by 
  sorry

end white_tshirts_l931_93149


namespace probability_of_drawing_white_ball_l931_93132

theorem probability_of_drawing_white_ball 
  (total_balls : ℕ) (white_balls : ℕ) 
  (h_total : total_balls = 9) (h_white : white_balls = 4) : 
  (white_balls : ℚ) / total_balls = 4 / 9 := 
by 
  sorry

end probability_of_drawing_white_ball_l931_93132


namespace triangle_right_hypotenuse_l931_93191

theorem triangle_right_hypotenuse (c : ℝ) (a : ℝ) (h₀ : c = 4) (h₁ : 0 < a) (h₂ : a^2 + b^2 = c^2) :
  a ≤ 2 * Real.sqrt 2 :=
sorry

end triangle_right_hypotenuse_l931_93191


namespace function_form_l931_93169

def satisfies_condition (f : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, m > 0 → n > 0 → ⌊ (f (m * n) : ℚ) / n ⌋ = f m

theorem function_form (f : ℕ → ℤ) (h : satisfies_condition f) :
  ∃ r : ℝ, ∀ n : ℕ, 
    (f n = ⌊ (r * n : ℝ) ⌋) ∨ (f n = ⌈ (r * n : ℝ) ⌉ - 1) := 
  sorry

end function_form_l931_93169


namespace stamps_initial_count_l931_93160

theorem stamps_initial_count (total_stamps stamps_received initial_stamps : ℕ) 
  (h1 : total_stamps = 61)
  (h2 : stamps_received = 27)
  (h3 : initial_stamps = total_stamps - stamps_received) :
  initial_stamps = 34 :=
sorry

end stamps_initial_count_l931_93160


namespace choir_members_max_l931_93109

-- Define the conditions and the proof for the equivalent problem.
theorem choir_members_max (c s y : ℕ) (h1 : c < 120) (h2 : s * y + 3 = c) (h3 : (s - 1) * (y + 2) = c) : c = 120 := by
  sorry

end choir_members_max_l931_93109


namespace crude_oil_mixture_l931_93119

theorem crude_oil_mixture (x y : ℝ) 
  (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 0.55 * 50) : 
  y = 30 :=
by
  sorry

end crude_oil_mixture_l931_93119


namespace fractions_proper_or_improper_l931_93167

theorem fractions_proper_or_improper : 
  ∀ (a b : ℚ), (∃ p q : ℚ, a = p / q ∧ p < q) ∨ (∃ r s : ℚ, a = r / s ∧ r ≥ s) :=
by 
  sorry

end fractions_proper_or_improper_l931_93167


namespace angle_between_lines_is_arctan_one_third_l931_93143

theorem angle_between_lines_is_arctan_one_third
  (l1 : ∀ x y : ℝ, 2 * x - y + 1 = 0)
  (l2 : ∀ x y : ℝ, x - y - 2 = 0)
  : ∃ θ : ℝ, θ = Real.arctan (1 / 3) := 
sorry

end angle_between_lines_is_arctan_one_third_l931_93143


namespace intersection_of_sets_l931_93146

def setA (x : ℝ) : Prop := x^2 - 4 * x - 5 > 0

def setB (x : ℝ) : Prop := 4 - x^2 > 0

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end intersection_of_sets_l931_93146


namespace yellow_yellow_pairs_count_l931_93199

def num_blue_students : ℕ := 75
def num_yellow_students : ℕ := 105
def total_pairs : ℕ := 90
def blue_blue_pairs : ℕ := 30

theorem yellow_yellow_pairs_count :
  -- number of pairs where both students are wearing yellow shirts is 45.
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 45 :=
by
  sorry

end yellow_yellow_pairs_count_l931_93199


namespace go_games_l931_93186

theorem go_games (total_go_balls : ℕ) (go_balls_per_game : ℕ) (h_total : total_go_balls = 901) (h_game : go_balls_per_game = 53) : (total_go_balls / go_balls_per_game) = 17 := by
  sorry

end go_games_l931_93186


namespace relatively_prime_ratios_l931_93156

theorem relatively_prime_ratios (r s : ℕ) (h_coprime: Nat.gcd r s = 1) 
  (h_cond: (r : ℝ) / s = 2 * (Real.sqrt 2 + Real.sqrt 10) / (5 * Real.sqrt (3 + Real.sqrt 5))) :
  r = 4 ∧ s = 5 :=
by
  sorry

end relatively_prime_ratios_l931_93156


namespace inequality_proof_l931_93170

variables {a1 a2 a3 b1 b2 b3 : ℝ}

theorem inequality_proof (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) 
                         (h4 : 0 < b1) (h5 : 0 < b2) (h6 : 0 < b3):
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 
  ≥ 4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) := 
sorry

end inequality_proof_l931_93170


namespace total_birds_remaining_l931_93157

theorem total_birds_remaining (grey_birds_in_cage : ℕ) (white_birds_next_to_cage : ℕ) :
  (grey_birds_in_cage = 40) →
  (white_birds_next_to_cage = grey_birds_in_cage + 6) →
  (1/2 * grey_birds_in_cage = 20) →
  (1/2 * grey_birds_in_cage + white_birds_next_to_cage = 66) :=
by 
  intros h_grey_birds h_white_birds h_grey_birds_freed
  sorry

end total_birds_remaining_l931_93157


namespace exists_representation_of_77_using_fewer_sevens_l931_93123

-- Definition of the problem
def represent_77 (expr : String) : Prop :=
  ∀ n : ℕ, expr = "77" ∨ 
             expr = "(77 - 7) + 7" ∨ 
             expr = "(10 * 7) + 7" ∨ 
             expr = "(70 + 7)" ∨ 
             expr = "(7 * 11)" ∨ 
             expr = "7 + 7 * 7 + (7 / 7)"

-- The proof statement
theorem exists_representation_of_77_using_fewer_sevens : ∃ expr : String, represent_77 expr ∧ String.length expr < 3 := 
sorry

end exists_representation_of_77_using_fewer_sevens_l931_93123


namespace find_range_of_m_l931_93130

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 7
def B (m x : ℝ) : Prop := m + 1 < x ∧ x < 2 * m - 1

theorem find_range_of_m (m : ℝ) : 
  (∀ x, B m x → A x) ∧ (∃ x, B m x) → 2 < m ∧ m ≤ 4 :=
by
  sorry

end find_range_of_m_l931_93130


namespace second_discount_percentage_l931_93113

-- Define the original price as P
variables {P : ℝ} (hP : P > 0)

-- Define the price increase by 34%
def price_after_increase (P : ℝ) := 1.34 * P

-- Define the first discount of 10%
def price_after_first_discount (P : ℝ) := 0.90 * (price_after_increase P)

-- Define the second discount percentage as D (in decimal form)
variables {D : ℝ}

-- Define the price after the second discount
def price_after_second_discount (P D : ℝ) := (1 - D) * (price_after_first_discount P)

-- Define the overall percentage gain of 2.51%
def final_price (P : ℝ) := 1.0251 * P

-- The main theorem to prove
theorem second_discount_percentage (hP : P > 0) (hD : 0 ≤ D ∧ D ≤ 1) :
  price_after_second_discount P D = final_price P ↔ D = 0.1495 :=
by
  sorry

end second_discount_percentage_l931_93113


namespace proof_problem_l931_93189

/-- Definition of the problem -/
def problem_statement : Prop :=
  ∃(a b c : ℝ) (A B C : ℝ) (D : ℝ),
    -- Conditions:
    ((b ^ 2 = a * c) ∧
     (2 * Real.cos (A - C) - 2 * Real.cos B = 1) ∧
     (D = 5) ∧
     -- Questions:
     (B = Real.pi / 3) ∧
     (∀ (AC CD : ℝ), (a = b ∧ b = c) → -- Equilateral triangle
       (AC * CD = (1/2) * (5 * AC - AC ^ 2) ∧
       (0 < AC * CD ∧ AC * CD ≤ 25/8))))

-- Lean 4 statement
theorem proof_problem : problem_statement := sorry

end proof_problem_l931_93189


namespace washing_machine_capacity_l931_93127

-- Define the conditions:
def shirts : ℕ := 39
def sweaters : ℕ := 33
def loads : ℕ := 9
def total_clothes : ℕ := shirts + sweaters -- which is 72

-- Define the statement to be proved:
theorem washing_machine_capacity : ∃ x : ℕ, loads * x = total_clothes ∧ x = 8 :=
by
  -- proof to be completed
  sorry

end washing_machine_capacity_l931_93127


namespace geometric_progression_sum_ratio_l931_93150

theorem geometric_progression_sum_ratio (a : ℝ) (r n : ℕ) (hn : r = 3)
  (h : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28) : n = 6 :=
by
  -- Place the steps of the proof here, which are not required as per instructions.
  sorry

end geometric_progression_sum_ratio_l931_93150


namespace alicia_total_payment_l931_93101

def daily_rent_cost : ℕ := 30
def miles_cost_per_mile : ℝ := 0.25
def rental_days : ℕ := 5
def driven_miles : ℕ := 500

def total_cost (daily_rent_cost : ℕ) (rental_days : ℕ)
               (miles_cost_per_mile : ℝ) (driven_miles : ℕ) : ℝ :=
  (daily_rent_cost * rental_days) + (miles_cost_per_mile * driven_miles)

theorem alicia_total_payment :
  total_cost daily_rent_cost rental_days miles_cost_per_mile driven_miles = 275 := by
  sorry

end alicia_total_payment_l931_93101


namespace cost_of_one_shirt_l931_93194

theorem cost_of_one_shirt
  (J S : ℝ)
  (h1 : 3 * J + 2 * S = 69)
  (h2 : 2 * J + 3 * S = 76) :
  S = 18 :=
by
  sorry

end cost_of_one_shirt_l931_93194


namespace electric_guitar_count_l931_93142

theorem electric_guitar_count (E A : ℤ) (h1 : E + A = 9) (h2 : 479 * E + 339 * A = 3611) (hE_nonneg : E ≥ 0) (hA_nonneg : A ≥ 0) : E = 4 :=
by
  sorry

end electric_guitar_count_l931_93142


namespace regression_shows_positive_correlation_l931_93173

-- Define the regression equations as constants
def reg_eq_A (x : ℝ) : ℝ := -2.1 * x + 1.8
def reg_eq_B (x : ℝ) : ℝ := 1.2 * x + 1.5
def reg_eq_C (x : ℝ) : ℝ := -0.5 * x + 2.1
def reg_eq_D (x : ℝ) : ℝ := -0.6 * x + 3

-- Define the condition for positive correlation
def positive_correlation (b : ℝ) : Prop := b > 0

-- The theorem statement to prove
theorem regression_shows_positive_correlation : 
  positive_correlation 1.2 := 
by
  sorry

end regression_shows_positive_correlation_l931_93173


namespace average_annual_growth_rate_sales_revenue_2018_l931_93181

-- Define the conditions as hypotheses
def initial_sales := 200000
def final_sales := 800000
def years := 2
def growth_rate := 1.0 -- representing 100%

theorem average_annual_growth_rate (x : ℝ) :
  (initial_sales : ℝ) * (1 + x)^years = final_sales → x = 1 :=
by
  intro h1
  sorry

theorem sales_revenue_2018 (x : ℝ) (revenue_2017 : ℝ) :
  x = 1 → revenue_2017 = final_sales → revenue_2017 * (1 + x) = 1600000 :=
by
  intros h1 h2
  sorry

end average_annual_growth_rate_sales_revenue_2018_l931_93181


namespace probability_same_color_l931_93128

theorem probability_same_color (pairs : ℕ) (total_shoes : ℕ) (select_shoes : ℕ)
  (h_pairs : pairs = 6) 
  (h_total_shoes : total_shoes = 12) 
  (h_select_shoes : select_shoes = 2) : 
  (Nat.choose total_shoes select_shoes > 0) → 
  (Nat.div (pairs * (Nat.choose 2 2)) (Nat.choose total_shoes select_shoes) = 1/11) :=
by
  sorry

end probability_same_color_l931_93128


namespace area_S4_is_3_125_l931_93106

theorem area_S4_is_3_125 (S_1 : Type) (area_S1 : ℝ) 
  (hS1 : area_S1 = 25)
  (bisect_and_construct : ∀ (S : Type) (area : ℝ),
    ∃ S' : Type, ∃ area' : ℝ, area' = area / 2) :
  ∃ S_4 : Type, ∃ area_S4 : ℝ, area_S4 = 3.125 :=
by
  sorry

end area_S4_is_3_125_l931_93106


namespace intersection_S_T_l931_93183

def setS (x : ℝ) : Prop := (x - 1) * (x - 3) ≥ 0
def setT (x : ℝ) : Prop := x > 0

theorem intersection_S_T : {x : ℝ | setS x} ∩ {x : ℝ | setT x} = {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (3 ≤ x)} := 
sorry

end intersection_S_T_l931_93183


namespace subcommittee_has_teacher_l931_93100

def total_combinations (n k : ℕ) : ℕ := Nat.choose n k

def teacher_subcommittee_count : ℕ := total_combinations 12 5 - total_combinations 7 5

theorem subcommittee_has_teacher : teacher_subcommittee_count = 771 := 
by
  sorry

end subcommittee_has_teacher_l931_93100


namespace largest_three_digit_divisible_by_13_l931_93152

theorem largest_three_digit_divisible_by_13 :
  ∃ n, (n ≤ 999 ∧ n ≥ 100 ∧ 13 ∣ n) ∧ (∀ m, m ≤ 999 ∧ m ≥ 100 ∧ 13 ∣ m → m ≤ 987) :=
by
  sorry

end largest_three_digit_divisible_by_13_l931_93152


namespace ellipse_condition_l931_93135

theorem ellipse_condition (x y m : ℝ) :
  (1 < m ∧ m < 3) → (∀ x y, (∃ k1 k2: ℝ, k1 > 0 ∧ k2 > 0 ∧ k1 ≠ k2 ∧ (x^2 / k1 + y^2 / k2 = 1 ↔ (1 < m ∧ m < 3 ∧ m ≠ 2)))) :=
by 
  sorry

end ellipse_condition_l931_93135


namespace tickets_used_63_l931_93192

def rides_ferris_wheel : ℕ := 5
def rides_bumper_cars : ℕ := 4
def cost_per_ride : ℕ := 7
def total_rides : ℕ := rides_ferris_wheel + rides_bumper_cars
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_63 : total_tickets_used = 63 := by
  unfold total_tickets_used
  unfold total_rides
  unfold rides_ferris_wheel
  unfold rides_bumper_cars
  unfold cost_per_ride
  -- proof goes here
  sorry

end tickets_used_63_l931_93192


namespace athlete_heartbeats_l931_93131

def heart_beats_per_minute : ℕ := 120
def running_pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 30
def total_heartbeats : ℕ := 21600

theorem athlete_heartbeats :
  (running_pace_minutes_per_mile * race_distance_miles * heart_beats_per_minute) = total_heartbeats :=
by
  sorry

end athlete_heartbeats_l931_93131


namespace ann_frosting_time_l931_93163

theorem ann_frosting_time (time_normal time_sprained n : ℕ) (h1 : time_normal = 5) (h2 : time_sprained = 8) (h3 : n = 10) : 
  ((time_sprained * n) - (time_normal * n)) = 30 := 
by 
  sorry

end ann_frosting_time_l931_93163


namespace salary_reduction_l931_93105

noncomputable def percentageIncrease : ℝ := 16.27906976744186 / 100

theorem salary_reduction (S R : ℝ) (P : ℝ) (h1 : R = S * (1 - P / 100)) (h2 : S = R * (1 + percentageIncrease)) : P = 14 :=
by
  sorry

end salary_reduction_l931_93105


namespace distance_traveled_l931_93162

theorem distance_traveled (speed time : ℕ) (h_speed : speed = 20) (h_time : time = 8) : 
  speed * time = 160 := 
by
  -- Solution proof goes here
  sorry

end distance_traveled_l931_93162


namespace P_works_alone_l931_93153

theorem P_works_alone (P : ℝ) (hP : 2 * (1 / P + 1 / 15) + 0.6 * (1 / P) = 1) : P = 3 :=
by sorry

end P_works_alone_l931_93153


namespace unique_surjective_f_l931_93114

-- Define the problem conditions
variable (f : ℕ → ℕ)

-- Define that f is surjective
axiom surjective_f : Function.Surjective f

-- Define condition that for every m, n and prime p
axiom condition_f : ∀ m n : ℕ, ∀ p : ℕ, Nat.Prime p → (p ∣ f (m + n) ↔ p ∣ f m + f n)

-- The theorem we need to prove: the only surjective function f satisfying the condition is the identity function
theorem unique_surjective_f : ∀ x : ℕ, f x = x :=
by
  sorry

end unique_surjective_f_l931_93114


namespace double_increase_divide_l931_93121

theorem double_increase_divide (x : ℤ) (h : (2 * x + 7) / 5 = 17) : x = 39 := by
  sorry

end double_increase_divide_l931_93121


namespace percentage_died_by_bombardment_l931_93126

def initial_population : ℕ := 4675
def remaining_population : ℕ := 3553
def left_percentage : ℕ := 20

theorem percentage_died_by_bombardment (x : ℕ) (h : initial_population * (100 - x) / 100 * 8 / 10 = remaining_population) : 
  x = 5 :=
by
  sorry

end percentage_died_by_bombardment_l931_93126


namespace trig_identity_solution_l931_93133

theorem trig_identity_solution
  (α : ℝ) (β : ℝ)
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan β = -1 / 3) :
  (3 * Real.sin α * Real.cos β - Real.sin β * Real.cos α) / (Real.cos α * Real.cos β + 2 * Real.sin α * Real.sin β) = 11 / 4 :=
by
  sorry

end trig_identity_solution_l931_93133


namespace square_of_binomial_l931_93125

-- Define a condition that the given term is the square of a binomial.
theorem square_of_binomial (a b: ℝ) : (a + b) * (a + b) = (a + b) ^ 2 :=
by {
  -- The proof is omitted.
  sorry
}

end square_of_binomial_l931_93125


namespace height_inradius_ratio_is_7_l931_93172

-- Definitions of geometric entities and given conditions.
variable (h r : ℝ)
variable (cos_theta : ℝ)
variable (cos_theta_eq : cos_theta = 1 / 6)

-- Theorem statement: Ratio of height to inradius is 7 given the cosine condition.
theorem height_inradius_ratio_is_7
  (h r : ℝ)
  (cos_theta : ℝ)
  (cos_theta_eq : cos_theta = 1 / 6)
  (prism_def : true) -- Added to mark the geometric nature properly
: h / r = 7 :=
sorry  -- Placeholder for the actual proof.

end height_inradius_ratio_is_7_l931_93172


namespace mass_of_three_packages_l931_93140

noncomputable def total_mass {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : ℝ := 
  x + y + z

theorem mass_of_three_packages {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : total_mass h1 h2 h3 = 175 :=
by
  sorry

end mass_of_three_packages_l931_93140


namespace inequality_solution_l931_93116

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 4) > 2 / x + 12 / 5 ↔ x < 0 :=
by
  sorry

end inequality_solution_l931_93116


namespace probability_at_least_one_l931_93118

variable (p_A p_B : ℚ) (hA : p_A = 1 / 4) (hB : p_B = 2 / 5)

theorem probability_at_least_one (h : p_A * (1 - p_B) + (1 - p_A) * p_B + p_A * p_B = 11 / 20) : 
  (1 - (1 - p_A) * (1 - p_B) = 11 / 20) :=
by
  rw [hA, hB,←h]
  sorry

end probability_at_least_one_l931_93118


namespace shortest_side_of_triangle_l931_93108

theorem shortest_side_of_triangle 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) 
  (h_inequal : a^2 + b^2 > 5 * c^2) :
  c < a ∧ c < b := 
by 
  sorry

end shortest_side_of_triangle_l931_93108


namespace claire_photos_l931_93138

theorem claire_photos (C L R : ℕ) 
  (h1 : L = 3 * C) 
  (h2 : R = C + 12)
  (h3 : L = R) : C = 6 := 
by
  sorry

end claire_photos_l931_93138


namespace distance_from_point_to_directrix_l931_93184

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l931_93184


namespace max_correct_answers_l931_93151

theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 80) (h2 : 5 * a - 2 * c = 150) : a ≤ 44 :=
by
  sorry

end max_correct_answers_l931_93151


namespace line_through_point_equal_distance_l931_93110

noncomputable def line_equation (x0 y0 a b c x1 y1 : ℝ) : Prop :=
  (a * x0 + b * y0 + c = 0) ∧ (a * x1 + b * y1 + c = 0)

theorem line_through_point_equal_distance (A B : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ (a b c : ℝ), 
    line_equation P.1 P.2 a b c A.1 A.2 ∧ 
    line_equation P.1 P.2 a b c B.1 B.2 ∧
    (a = 2) ∧ (b = 3) ∧ (c = -18) ∨
    (a = 2) ∧ (b = -1) ∧ (c = -2)
:=
sorry

end line_through_point_equal_distance_l931_93110


namespace find_a_of_pure_imaginary_z_l931_93155

-- Definition of a pure imaginary number
def pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Main theorem statement
theorem find_a_of_pure_imaginary_z (a : ℝ) (z : ℂ) (hz : pure_imaginary z) (h : (2 - I) * z = 4 + 2 * a * I) : a = 4 :=
by
  sorry

end find_a_of_pure_imaginary_z_l931_93155


namespace smallest_lcm_l931_93124

theorem smallest_lcm (k l : ℕ) (hk : k ≥ 1000) (hl : l ≥ 1000) (huk : k < 10000) (hul : l < 10000) (hk_pos : 0 < k) (hl_pos : 0 < l) (h_gcd: Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
by
  sorry

end smallest_lcm_l931_93124


namespace polygon_interior_exterior_relation_l931_93166

theorem polygon_interior_exterior_relation (n : ℕ) (h1 : (n-2) * 180 = 2 * 360) : n = 6 :=
by sorry

end polygon_interior_exterior_relation_l931_93166


namespace georgie_ghost_enter_exit_diff_window_l931_93145

theorem georgie_ghost_enter_exit_diff_window (n : ℕ) (h : n = 8) :
    (∃ enter exit, enter ≠ exit ∧ 1 ≤ enter ∧ enter ≤ n ∧ 1 ≤ exit ∧ exit ≤ n) ∧
    (∃ W : ℕ, W = (n * (n - 1))) :=
sorry

end georgie_ghost_enter_exit_diff_window_l931_93145


namespace power_function_value_l931_93154

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_value (α : ℝ) (h : 2 ^ α = (Real.sqrt 2) / 2) : f 4 α = 1 / 2 := 
by 
  sorry

end power_function_value_l931_93154


namespace largest_sum_of_distinct_factors_of_1764_l931_93193

theorem largest_sum_of_distinct_factors_of_1764 :
  ∃ (A B C : ℕ), A * B * C = 1764 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A + B + C = 33 :=
by
  sorry

end largest_sum_of_distinct_factors_of_1764_l931_93193


namespace evaluate_imaginary_expression_l931_93141

theorem evaluate_imaginary_expression (i : ℂ) (h_i2 : i^2 = -1) (h_i4 : i^4 = 1) :
  i^14 + i^19 + i^24 + i^29 + 3 * i^34 + 2 * i^39 = -3 - 2 * i :=
by sorry

end evaluate_imaginary_expression_l931_93141
