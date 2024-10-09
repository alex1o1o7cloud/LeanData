import Mathlib

namespace monotonicity_and_extrema_of_f_l1049_104916

noncomputable def f (x : ℝ) : ℝ := 3 * x + 2

theorem monotonicity_and_extrema_of_f :
  (∀ (x_1 x_2 : ℝ), x_1 ∈ Set.Icc (-1 : ℝ) 2 → x_2 ∈ Set.Icc (-1 : ℝ) 2 → x_1 < x_2 → f x_1 < f x_2) ∧ 
  (f (-1) = -1) ∧ 
  (f 2 = 8) :=
by
  sorry

end monotonicity_and_extrema_of_f_l1049_104916


namespace volume_of_prism_l1049_104985

theorem volume_of_prism (a b c : ℝ) (h₁ : a * b = 48) (h₂ : b * c = 36) (h₃ : a * c = 50) : 
    (a * b * c = 170) :=
by
  sorry

end volume_of_prism_l1049_104985


namespace find_matrix_M_l1049_104955

-- Define the given matrix with real entries
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 2], ![-1, 0]]

-- Define the function for matrix operations
def M_calc (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (M * M * M) - (M * M) + (2 • M)

-- Define the target matrix
def target_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 3], ![-2, 0]]

-- Problem statement: The matrix M should satisfy the given matrix equation
theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) :
  M_calc M = target_matrix ↔ M = matrix_M :=
sorry

end find_matrix_M_l1049_104955


namespace rectangle_length_l1049_104918

theorem rectangle_length {b l : ℝ} (h1 : 2 * (l + b) = 5 * b) (h2 : l * b = 216) : l = 18 := by
    sorry

end rectangle_length_l1049_104918


namespace students_count_rental_cost_l1049_104936

theorem students_count (k m : ℕ) (n : ℕ) 
  (h1 : n = 35 * k)
  (h2 : n = 55 * (m - 1) + 45) : 
  n = 175 := 
by {
  sorry
}

theorem rental_cost (x y : ℕ) 
  (total_buses : x + y = 4)
  (cost_limit : 35 * x + 55 * y ≤ 1500) : 
  320 * x + 400 * y = 1440 := 
by {
  sorry 
}

end students_count_rental_cost_l1049_104936


namespace intersection_nonempty_l1049_104921

open Nat

theorem intersection_nonempty (a : ℕ) (ha : a ≥ 2) :
  ∃ (b : ℕ), b = 1 ∨ b = a ∧
  ∃ y, (∃ x, y = a^x ∧ x ≥ 1) ∧
       (∃ x, y = (a + 1)^x + b ∧ x ≥ 1) :=
by sorry

end intersection_nonempty_l1049_104921


namespace intersection_M_N_union_complements_M_N_l1049_104942

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_M_N :
  M ∩ N = {x | 1 ≤ x ∧ x < 5} :=
by {
  sorry
}

theorem union_complements_M_N :
  (compl M) ∪ (compl N) = {x | x < 1 ∨ x ≥ 5} :=
by {
  sorry
}

end intersection_M_N_union_complements_M_N_l1049_104942


namespace commute_times_abs_diff_l1049_104939

def commute_times_avg (x y : ℝ) : Prop := (x + y + 7 + 8 + 9) / 5 = 8
def commute_times_var (x y : ℝ) : Prop := ((x - 8)^2 + (y - 8)^2 + (7 - 8)^2 + (8 - 8)^2 + (9 - 8)^2) / 5 = 4

theorem commute_times_abs_diff (x y : ℝ) (h_avg : commute_times_avg x y) (h_var : commute_times_var x y) :
  |x - y| = 6 :=
sorry

end commute_times_abs_diff_l1049_104939


namespace puppies_per_cage_l1049_104908

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (remaining_puppies : ℕ)
  (cages : ℕ)
  (puppies_per_cage : ℕ)
  (h1 : initial_puppies = 78)
  (h2 : sold_puppies = 30)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 6)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 8 := by
  sorry

end puppies_per_cage_l1049_104908


namespace solve_for_a_and_b_l1049_104913

theorem solve_for_a_and_b (a b : ℤ) (h1 : 5 + a = 6 - b) (h2 : 6 + b = 9 + a) : 5 - a = 6 := 
sorry

end solve_for_a_and_b_l1049_104913


namespace part_a_cube_edge_length_part_b_cube_edge_length_l1049_104915

-- Part (a)
theorem part_a_cube_edge_length (small_cubes : ℕ) (edge_length_original : ℤ) :
  small_cubes = 512 → edge_length_original^3 = small_cubes → edge_length_original = 8 :=
by
  intros h1 h2
  sorry

-- Part (b)
theorem part_b_cube_edge_length (small_cubes_internal : ℕ) (edge_length_inner : ℤ) (edge_length_original : ℤ) :
  small_cubes_internal = 512 →
  edge_length_inner^3 = small_cubes_internal → 
  edge_length_original = edge_length_inner + 2 →
  edge_length_original = 10 :=
by
  intros h1 h2 h3
  sorry

end part_a_cube_edge_length_part_b_cube_edge_length_l1049_104915


namespace trivia_team_students_per_group_l1049_104925

theorem trivia_team_students_per_group (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total_students = 58) (h2 : not_picked = 10) (h3 : num_groups = 8) :
  (total_students - not_picked) / num_groups = 6 :=
by
  sorry

end trivia_team_students_per_group_l1049_104925


namespace avg_visitors_other_days_l1049_104992

-- Definitions for average visitors on Sundays and average visitors over the month
def avg_visitors_on_sundays : ℕ := 600
def avg_visitors_over_month : ℕ := 300
def days_in_month : ℕ := 30

-- Given conditions
def num_sundays_in_month : ℕ := 5
def total_days : ℕ := days_in_month
def total_visitors_over_month : ℕ := avg_visitors_over_month * days_in_month

-- Goal: Calculate the average number of visitors on other days (Monday to Saturday)
theorem avg_visitors_other_days :
  (avg_visitors_on_sundays * num_sundays_in_month + (total_days - num_sundays_in_month) * 240) = total_visitors_over_month :=
by
  -- Proof expected here, but skipped according to the instructions
  sorry

end avg_visitors_other_days_l1049_104992


namespace non_degenerate_ellipse_l1049_104983

theorem non_degenerate_ellipse (k : ℝ) : 
    (∃ x y : ℝ, x^2 + 9 * y^2 - 6 * x + 18 * y = k) ↔ k > -18 :=
sorry

end non_degenerate_ellipse_l1049_104983


namespace blueBirdChessTeam72_l1049_104957

def blueBirdChessTeamArrangements : Nat :=
  let boys_girls_ends := 3 * 3 + 3 * 3
  let alternate_arrangements := 2 * 2
  boys_girls_ends * alternate_arrangements

theorem blueBirdChessTeam72 : blueBirdChessTeamArrangements = 72 := by
  unfold blueBirdChessTeamArrangements
  sorry

end blueBirdChessTeam72_l1049_104957


namespace find_a_l1049_104981

-- Define the lines as given
def line1 (x y : ℝ) := 2 * x + y - 5 = 0
def line2 (x y : ℝ) := x - y - 1 = 0
def line3 (a x y : ℝ) := a * x + y - 3 = 0

-- Define the condition that they intersect at a single point
def lines_intersect_at_point (x y a : ℝ) := line1 x y ∧ line2 x y ∧ line3 a x y

-- To prove: If lines intersect at a certain point, then a = 1
theorem find_a (a : ℝ) : (∃ x y, lines_intersect_at_point x y a) → a = 1 :=
by
  sorry

end find_a_l1049_104981


namespace nabla_four_seven_l1049_104969

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem nabla_four_seven : nabla 4 7 = 11 / 29 :=
by
  sorry

end nabla_four_seven_l1049_104969


namespace arithmetic_sequence_a7_l1049_104924

variable {a : ℕ → ℚ}

def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (h_arith : isArithmeticSequence a) (h_a1 : a 1 = 2) (h_a3_a5 : a 3 + a 5 = 8) :
  a 7 = 6 :=
sorry

end arithmetic_sequence_a7_l1049_104924


namespace find_x_squared_plus_one_over_x_squared_l1049_104995

theorem find_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 4) : x^2 + 1/x^2 = 14 := by
  sorry

end find_x_squared_plus_one_over_x_squared_l1049_104995


namespace factorize1_factorize2_factorize3_l1049_104901

-- Proof problem 1: Prove m^2 + 4m + 4 = (m + 2)^2
theorem factorize1 (m : ℝ) : m^2 + 4 * m + 4 = (m + 2)^2 :=
sorry

-- Proof problem 2: Prove a^2 b - 4ab^2 + 3b^3 = b(a-b)(a-3b)
theorem factorize2 (a b : ℝ) : a^2 * b - 4 * a * b^2 + 3 * b^3 = b * (a - b) * (a - 3 * b) :=
sorry

-- Proof problem 3: Prove (x^2 + y^2)^2 - 4x^2 y^2 = (x + y)^2 (x - y)^2
theorem factorize3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

end factorize1_factorize2_factorize3_l1049_104901


namespace Batman_game_cost_l1049_104938

theorem Batman_game_cost (football_cost strategy_cost total_spent batman_cost : ℝ)
  (h₁ : football_cost = 14.02)
  (h₂ : strategy_cost = 9.46)
  (h₃ : total_spent = 35.52)
  (h₄ : total_spent = football_cost + strategy_cost + batman_cost) :
  batman_cost = 12.04 := by
  sorry

end Batman_game_cost_l1049_104938


namespace triangle_sides_consecutive_obtuse_l1049_104966

/-- Given the sides of a triangle are consecutive natural numbers 
    and the largest angle is obtuse, 
    the lengths of the sides in ascending order are 2, 3, 4. -/
theorem triangle_sides_consecutive_obtuse 
    (x : ℕ) (hx : x > 1) 
    (cos_alpha_neg : (x - 4) < 0) 
    (x_lt_4 : x < 4) :
    (x = 3) → (∃ a b c : ℕ, a < b ∧ b < c ∧ a + b > c ∧ a = 2 ∧ b = 3 ∧ c = 4) :=
by
  intro hx3
  use 2, 3, 4
  repeat {split}
  any_goals {linarith}
  all_goals {sorry}

end triangle_sides_consecutive_obtuse_l1049_104966


namespace loss_percent_l1049_104923

theorem loss_percent (cost_price selling_price loss_percent : ℝ) 
  (h_cost_price : cost_price = 600)
  (h_selling_price : selling_price = 550)
  (h_loss_percent : loss_percent = 8.33) : 
  (loss_percent = ((cost_price - selling_price) / cost_price) * 100) := 
by
  rw [h_cost_price, h_selling_price]
  sorry

end loss_percent_l1049_104923


namespace expand_and_simplify_l1049_104934

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (14 / x^3 + 15 * x - 6 * x^5) = (6 / x^3) + (45 * x / 7) - (18 * x^5 / 7) :=
by
  sorry

end expand_and_simplify_l1049_104934


namespace student_distribution_l1049_104990

theorem student_distribution (a b : ℕ) (h1 : a + b = 81) (h2 : a = b - 9) : a = 36 ∧ b = 45 := 
by
  sorry

end student_distribution_l1049_104990


namespace solve_equation_l1049_104991

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x) / (x - 1) = 2 + 1 / (x - 1) → x = -1 :=
by
  sorry

end solve_equation_l1049_104991


namespace a_2015_eq_neg6_l1049_104902

noncomputable def a : ℕ → ℤ
| 0 => 3
| 1 => 6
| (n+2) => a (n+1) - a n

theorem a_2015_eq_neg6 : a 2015 = -6 := 
by 
  sorry

end a_2015_eq_neg6_l1049_104902


namespace ali_ate_half_to_percent_l1049_104993

theorem ali_ate_half_to_percent : (1 / 2 : ℚ) * 100 = 50 := by
  sorry

end ali_ate_half_to_percent_l1049_104993


namespace arithmetic_sequence_problem_l1049_104904

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ A, ∀ n : ℕ, a n = A * (q ^ (n - 1))

theorem arithmetic_sequence_problem
  (q : ℝ) 
  (h1 : q > 1)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h_seq : is_arithmetic_sequence a q) : 
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := 
by 
  sorry

end arithmetic_sequence_problem_l1049_104904


namespace arithmetic_sequence_common_difference_l1049_104927

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) :
  (a 5 = 8) → (a 1 + a 2 + a 3 = 6) → (∀ n : ℕ, a (n + 1) = a 1 + n * d) → d = 2 :=
by
  intros ha5 hsum harr
  sorry

end arithmetic_sequence_common_difference_l1049_104927


namespace find_a_20_l1049_104972

-- Definitions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ (a₀ d : ℤ), ∀ n, a n = a₀ + n * d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 0 + a (n - 1)) / 2

-- Conditions and question
theorem find_a_20 (a S : ℕ → ℤ) (a₀ d : ℤ) :
  arithmetic_seq a ∧ sum_first_n a S ∧ 
  S 6 = 8 * (S 3) ∧ a 3 - a 5 = 8 → a 20 = -74 :=
by
  sorry

end find_a_20_l1049_104972


namespace gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l1049_104977

theorem gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1 :
  Int.gcd (97 ^ 10 + 1) (97 ^ 10 + 97 ^ 3 + 1) = 1 := sorry

end gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l1049_104977


namespace square_perimeter_ratio_l1049_104974

theorem square_perimeter_ratio (x y : ℝ)
(h : (x / y) ^ 2 = 16 / 25) : (4 * x) / (4 * y) = 4 / 5 :=
by sorry

end square_perimeter_ratio_l1049_104974


namespace find_geometric_sequence_l1049_104967

def geometric_sequence (b1 b2 b3 b4 : ℤ) :=
  ∃ q : ℤ, b2 = b1 * q ∧ b3 = b1 * q^2 ∧ b4 = b1 * q^3

theorem find_geometric_sequence :
  ∃ b1 b2 b3 b4 : ℤ, 
    geometric_sequence b1 b2 b3 b4 ∧
    (b1 + b4 = -49) ∧
    (b2 + b3 = 14) ∧ 
    ((b1, b2, b3, b4) = (7, -14, 28, -56) ∨ (b1, b2, b3, b4) = (-56, 28, -14, 7)) :=
by
  sorry

end find_geometric_sequence_l1049_104967


namespace optimal_solution_range_l1049_104980

theorem optimal_solution_range (a : ℝ) (x y : ℝ) :
  (x + y - 4 ≥ 0) → (2 * x - y - 5 ≤ 0) → (x = 1) → (y = 3) →
  (-2 < a) ∧ (a < 1) :=
by
  intros h1 h2 hx hy
  sorry

end optimal_solution_range_l1049_104980


namespace double_grandfather_pension_l1049_104945

-- Define the total family income and individual contributions
def total_income (masha mother father grandfather : ℝ) : ℝ :=
  masha + mother + father + grandfather

-- Define the conditions provided in the problem
variables
  (masha mother father grandfather : ℝ)
  (cond1 : 2 * masha = total_income masha mother father grandfather * 1.05)
  (cond2 : 2 * mother = total_income masha mother father grandfather * 1.15)
  (cond3 : 2 * father = total_income masha mother father grandfather * 1.25)

-- Define the statement to be proved
theorem double_grandfather_pension :
  2 * grandfather = total_income masha mother father grandfather * 1.55 :=
by
  -- Proof placeholder
  sorry

end double_grandfather_pension_l1049_104945


namespace common_solution_l1049_104933

theorem common_solution (x : ℚ) : 
  (8 * x^2 + 7 * x - 1 = 0) ∧ (40 * x^2 + 89 * x - 9 = 0) → x = 1 / 8 :=
by { sorry }

end common_solution_l1049_104933


namespace num_boys_l1049_104976

variable (B G : ℕ)

def ratio_boys_girls (B G : ℕ) : Prop := B = 7 * G
def total_students (B G : ℕ) : Prop := B + G = 48

theorem num_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : 
  B = 42 :=
by
  sorry

end num_boys_l1049_104976


namespace total_junk_mail_l1049_104950

-- Definitions for conditions
def houses_per_block : Nat := 17
def pieces_per_house : Nat := 4
def blocks : Nat := 16

-- Theorem stating that the mailman gives out 1088 pieces of junk mail in total
theorem total_junk_mail : houses_per_block * pieces_per_house * blocks = 1088 := by
  sorry

end total_junk_mail_l1049_104950


namespace average_of_remaining_numbers_l1049_104903

theorem average_of_remaining_numbers (s : ℝ) (a b c d e f : ℝ)
  (h1: (a + b + c + d + e + f) / 6 = 3.95)
  (h2: (a + b) / 2 = 4.4)
  (h3: (c + d) / 2 = 3.85) :
  ((e + f) / 2 = 3.6) :=
by
  sorry

end average_of_remaining_numbers_l1049_104903


namespace compound_interest_calculation_l1049_104912

noncomputable def compoundInterest (P r t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simpleInterest (P r t : ℝ) : ℝ :=
  P * r * t

theorem compound_interest_calculation :
  ∃ P : ℝ, simpleInterest P 0.10 2 = 600 ∧ compoundInterest P 0.10 2 = 630 :=
by
  sorry

end compound_interest_calculation_l1049_104912


namespace shifted_parabola_sum_l1049_104940

theorem shifted_parabola_sum :
  let f (x : ℝ) := 3 * x^2 - 2 * x + 5
  let g (x : ℝ) := 3 * (x - 3)^2 - 2 * (x - 3) + 5
  let a := 3
  let b := -20
  let c := 38
  a + b + c = 21 :=
by
  sorry

end shifted_parabola_sum_l1049_104940


namespace power_function_properties_l1049_104944

theorem power_function_properties (m : ℤ) :
  (m^2 - 2 * m - 2 ≠ 0) ∧ (m^2 + 4 * m < 0) ∧ (m^2 + 4 * m % 2 = 1) → m = -1 := by
  intro h
  sorry

end power_function_properties_l1049_104944


namespace frank_maze_time_l1049_104951

theorem frank_maze_time 
    (n mazes : ℕ)
    (avg_time_per_maze completed_time total_allowable_time remaining_maze_time extra_time_inside current_time : ℕ) 
    (h1 : mazes = 5)
    (h2 : avg_time_per_maze = 60)
    (h3 : completed_time = 200)
    (h4 : total_allowable_time = mazes * avg_time_per_maze)
    (h5 : total_allowable_time = 300)
    (h6 : remaining_maze_time = total_allowable_time - completed_time) 
    (h7 : extra_time_inside = 55)
    (h8 : current_time + extra_time_inside ≤ remaining_maze_time) :
  current_time = 45 :=
by
  sorry

end frank_maze_time_l1049_104951


namespace sufficient_condition_l1049_104914

theorem sufficient_condition (a b : ℝ) (h : a > b ∧ b > 0) : a + a^2 > b + b^2 :=
by
  sorry

end sufficient_condition_l1049_104914


namespace pyramid_base_side_length_l1049_104961

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ)
  (hA : A = 200)
  (hh : h = 40)
  (hface : A = (1 / 2) * s * h) : 
  s = 10 :=
by
  sorry

end pyramid_base_side_length_l1049_104961


namespace find_circle_center_l1049_104948

def circle_center_eq : Prop :=
  ∃ (x y : ℝ), (x^2 - 6 * x + y^2 + 2 * y - 12 = 0) ∧ (x = 3) ∧ (y = -1)

theorem find_circle_center : circle_center_eq :=
sorry

end find_circle_center_l1049_104948


namespace new_cases_first_week_l1049_104975

theorem new_cases_first_week
  (X : ℕ)
  (second_week_cases : X / 2 = X / 2)
  (third_week_cases : X / 2 + 2000 = (X / 2) + 2000)
  (total_cases : X + X / 2 + (X / 2 + 2000) = 9500) :
  X = 3750 := 
by sorry

end new_cases_first_week_l1049_104975


namespace sum_digits_10_pow_100_minus_100_l1049_104958

open Nat

/-- Define the condition: 10^100 - 100 as an expression. -/
def subtract_100_from_power_10 (n : ℕ) : ℕ :=
  10^n - 100

/-- Sum the digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The goal is to prove the sum of the digits of 10^100 - 100 equals 882. -/
theorem sum_digits_10_pow_100_minus_100 :
  sum_of_digits (subtract_100_from_power_10 100) = 882 :=
by
  sorry

end sum_digits_10_pow_100_minus_100_l1049_104958


namespace irrational_number_l1049_104964

noncomputable def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number : 
  is_rational (Real.sqrt 4) ∧ 
  is_rational (22 / 7 : ℝ) ∧ 
  is_rational (1.0101 : ℝ) ∧ 
  ¬ is_rational (Real.pi / 3) 
  :=
sorry

end irrational_number_l1049_104964


namespace missing_angle_in_convex_polygon_l1049_104928

theorem missing_angle_in_convex_polygon (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 5) 
  (h2 : 180 * (n - 2) - 3 * x = 3330) : 
  x = 54 := 
by 
  sorry

end missing_angle_in_convex_polygon_l1049_104928


namespace certain_number_eq_neg17_l1049_104926

theorem certain_number_eq_neg17 (x : Int) : 47 + x = 30 → x = -17 := by
  intro h
  have : x = 30 - 47 := by
    sorry  -- This is just to demonstrate the proof step. Actual manipulation should prove x = -17
  simp [this]

end certain_number_eq_neg17_l1049_104926


namespace number_of_girls_joined_l1049_104920

-- Define the initial conditions
def initial_girls := 18
def initial_boys := 15
def boys_quit := 4
def total_children_after_changes := 36

-- Define the changes
def boys_after_quit := initial_boys - boys_quit
def girls_after_changes := total_children_after_changes - boys_after_quit
def girls_joined := girls_after_changes - initial_girls

-- State the theorem
theorem number_of_girls_joined :
  girls_joined = 7 :=
by
  sorry

end number_of_girls_joined_l1049_104920


namespace repeating_block_length_of_three_elevens_l1049_104982

def smallest_repeating_block_length (x : ℚ) : ℕ :=
  sorry -- Definition to compute smallest repeating block length if needed

theorem repeating_block_length_of_three_elevens :
  smallest_repeating_block_length (3 / 11) = 2 :=
sorry

end repeating_block_length_of_three_elevens_l1049_104982


namespace trig_identity_l1049_104929

theorem trig_identity (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) 
  : Real.cos (5 / 6 * π + α) + (Real.cos (4 * π / 3 + α))^2 = (2 - Real.sqrt 3) / 3 := 
sorry

end trig_identity_l1049_104929


namespace problem_inequality_l1049_104952

variable (a b : ℝ)

theorem problem_inequality (h1 : a < 0) (h2 : -1 < b) (h3 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end problem_inequality_l1049_104952


namespace remainder_8_pow_2023_mod_5_l1049_104970

theorem remainder_8_pow_2023_mod_5 :
  8 ^ 2023 % 5 = 2 :=
by
  sorry

end remainder_8_pow_2023_mod_5_l1049_104970


namespace probability_at_least_two_worth_visiting_l1049_104986

theorem probability_at_least_two_worth_visiting :
  let total_caves := 8
  let worth_visiting := 3
  let select_caves := 4
  let worth_select_2 := Nat.choose worth_visiting 2 * Nat.choose (total_caves - worth_visiting) 2
  let worth_select_3 := Nat.choose worth_visiting 3 * Nat.choose (total_caves - worth_visiting) 1
  let total_select := Nat.choose total_caves select_caves
  let probability := (worth_select_2 + worth_select_3) / total_select
  probability = 1 / 2 := sorry

end probability_at_least_two_worth_visiting_l1049_104986


namespace tan_double_angle_l1049_104956

theorem tan_double_angle (α : ℝ) (h : Real.tan (π - α) = 2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

end tan_double_angle_l1049_104956


namespace union_M_N_l1049_104909

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_M_N : M ∪ N = {x | -1 < x ∧ x < 3} := 
by 
  sorry

end union_M_N_l1049_104909


namespace jack_needs_more_money_l1049_104906

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end jack_needs_more_money_l1049_104906


namespace k_less_than_two_l1049_104989

theorem k_less_than_two
    (x : ℝ)
    (k : ℝ)
    (y : ℝ)
    (h : y = (2 - k) / x)
    (h1 : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) : k < 2 :=
by
  sorry

end k_less_than_two_l1049_104989


namespace large_cube_side_length_painted_blue_l1049_104907

   theorem large_cube_side_length_painted_blue (n : ℕ) (h : 6 * n^2 = (1 / 3) * 6 * n^3) : n = 3 :=
   by
     sorry
   
end large_cube_side_length_painted_blue_l1049_104907


namespace binomial_coeffs_not_arith_seq_l1049_104917

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def are_pos_integer (n : ℕ) : Prop := n > 0

def is_arith_seq (a b c d : ℕ) : Prop := 
  2 * b = a + c ∧ 2 * c = b + d 

theorem binomial_coeffs_not_arith_seq (n r : ℕ) : 
  are_pos_integer n → are_pos_integer r → n ≥ r + 3 → ¬ is_arith_seq (binomial n r) (binomial n (r+1)) (binomial n (r+2)) (binomial n (r+3)) :=
by
  sorry

end binomial_coeffs_not_arith_seq_l1049_104917


namespace total_passengers_transportation_l1049_104937

theorem total_passengers_transportation : 
  let passengers_one_way := 100
  let passengers_return := 60
  let first_trip_total := passengers_one_way + passengers_return
  let additional_trips := 3
  let additional_trips_total := additional_trips * first_trip_total
  let total_passengers := first_trip_total + additional_trips_total
  total_passengers = 640 := 
by
  sorry

end total_passengers_transportation_l1049_104937


namespace calculate_result_l1049_104965

theorem calculate_result (x : ℝ) : (-x^3)^3 = -x^9 :=
by {
  sorry  -- Proof not required per instructions
}

end calculate_result_l1049_104965


namespace problem1_problem2_l1049_104962

noncomputable def f (x a b c : ℝ) : ℝ := abs (x + a) + abs (x - b) + c

theorem problem1 (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : ∃ x, f x a b c = 4) : a + b + c = 4 :=
sorry

theorem problem2 (a b c : ℝ) (h : a + b + c = 4) : (1 / a) + (1 / b) + (1 / c) ≥ 9 / 4 :=
sorry

end problem1_problem2_l1049_104962


namespace tan_3theta_l1049_104954

theorem tan_3theta (θ : ℝ) (h : Real.tan θ = 3 / 4) : Real.tan (3 * θ) = -12.5 :=
sorry

end tan_3theta_l1049_104954


namespace area_of_isosceles_triangle_l1049_104941

open Real

theorem area_of_isosceles_triangle 
  (PQ PR QR : ℝ) (PQ_eq_PR : PQ = PR) (PQ_val : PQ = 13) (QR_val : QR = 10) : 
  1 / 2 * QR * sqrt (PQ^2 - (QR / 2)^2) = 60 := 
by 
sorry

end area_of_isosceles_triangle_l1049_104941


namespace truncated_quadrilateral_pyramid_exists_l1049_104911

theorem truncated_quadrilateral_pyramid_exists :
  ∃ (x y z u r s t : ℤ),
    x = 4 * r * t ∧
    y = 4 * s * t ∧
    z = (r - s)^2 - 2 * t^2 ∧
    u = (r - s)^2 + 2 * t^2 ∧
    (x - y)^2 + 2 * z^2 = 2 * u^2 :=
by
  sorry

end truncated_quadrilateral_pyramid_exists_l1049_104911


namespace find_k_l1049_104900

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) ∧ 
  (∀ x : ℝ, y = k * x + 4) ∧ 
  (1, 5) ∈ { p | ∃ x, p = (x, 2 * x + 3) } ∧ 
  (1, 5) ∈ { q | ∃ x, q = (x, k * x + 4) } → 
  k = 1 :=
by
  sorry

end find_k_l1049_104900


namespace necessary_condition_for_positive_on_interval_l1049_104953

theorem necessary_condition_for_positive_on_interval (a b : ℝ) (h : a + 2 * b > 0) :
  (∀ x, 0 ≤ x → x ≤ 1 → (a * x + b) > 0) ↔ ∃ c, 0 < c ∧ c ≤ 1 ∧ a + 2 * b > 0 ∧ ¬∀ d, 0 < d ∧ d ≤ 1 → a * d + b > 0 := 
by 
  sorry

end necessary_condition_for_positive_on_interval_l1049_104953


namespace simplify_fraction_lemma_l1049_104994

noncomputable def simplify_fraction (a : ℝ) (h : a ≠ 5) : ℝ :=
  (a^2 - 5 * a) / (a - 5)

theorem simplify_fraction_lemma (a : ℝ) (h : a ≠ 5) : simplify_fraction a h = a := by
  sorry

end simplify_fraction_lemma_l1049_104994


namespace frog_jump_problem_l1049_104910

theorem frog_jump_problem (A B C : ℝ) (PA PB PC : ℝ) 
  (H1: PA' = (PB + PC) / 2)
  (H2: jump_distance_B = 60)
  (H3: jump_distance_B = 2 * abs ((PB - (PB + PC) / 2))) :
  third_jump_distance = 30 := sorry

end frog_jump_problem_l1049_104910


namespace find_a_plus_b_l1049_104968

theorem find_a_plus_b {f : ℝ → ℝ} (a b : ℝ) :
  (∀ x, f x = x^3 + 3*x^2 + 6*x + 14) →
  f a = 1 →
  f b = 19 →
  a + b = -2 :=
by
  sorry

end find_a_plus_b_l1049_104968


namespace average_income_QR_l1049_104996

theorem average_income_QR 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (P + R) / 2 = 6200)
  (h3 : P = 3000) :
  (Q + R) / 2 = 5250 :=
  sorry

end average_income_QR_l1049_104996


namespace tank_capacity_l1049_104935

theorem tank_capacity (w c : ℝ) (h1 : w / c = 1 / 6) (h2 : (w + 5) / c = 1 / 3) : c = 30 :=
by
  sorry

end tank_capacity_l1049_104935


namespace proof_x_squared_minus_y_squared_l1049_104971

theorem proof_x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 9 / 14) (h2 : x - y = 3 / 14) :
  x^2 - y^2 = 27 / 196 := by
  sorry

end proof_x_squared_minus_y_squared_l1049_104971


namespace tony_water_intake_l1049_104959

theorem tony_water_intake (yesterday water_two_days_ago : ℝ) 
    (h1 : yesterday = 48) (h2 : yesterday = 0.96 * water_two_days_ago) :
    water_two_days_ago = 50 :=
by
  sorry

end tony_water_intake_l1049_104959


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l1049_104932

def consecutive_primes (n : Nat) : Prop :=
  -- Define what it means to be 4 consecutive prime numbers
  Nat.Prime n ∧ Nat.Prime (n + 2) ∧ Nat.Prime (n + 6) ∧ Nat.Prime (n + 8)

def sum_of_consecutive_primes (n : Nat) : Nat :=
  n + (n + 2) + (n + 6) + (n + 8)

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ n, n > 10 ∧ consecutive_primes n ∧ sum_of_consecutive_primes n % 5 = 0 ∧ sum_of_consecutive_primes n = 60 :=
by
  sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l1049_104932


namespace rectangle_length_l1049_104987

theorem rectangle_length (L W : ℝ) 
  (h1 : L + W = 23) 
  (h2 : L^2 + W^2 = 289) : 
  L = 15 :=
by 
  sorry

end rectangle_length_l1049_104987


namespace rectangle_square_ratio_l1049_104979

theorem rectangle_square_ratio (s x y : ℝ) (h1 : 0.1 * s ^ 2 = 0.25 * x * y) (h2 : y = s / 4) :
  x / y = 6 := 
sorry

end rectangle_square_ratio_l1049_104979


namespace odd_function_iff_a2_b2_zero_l1049_104998

noncomputable def f (x a b : ℝ) : ℝ := x * |x - a| + b

theorem odd_function_iff_a2_b2_zero {a b : ℝ} :
  (∀ x, f x a b = - f (-x) a b) ↔ a^2 + b^2 = 0 := by
  sorry

end odd_function_iff_a2_b2_zero_l1049_104998


namespace largest_and_next_largest_difference_l1049_104931

theorem largest_and_next_largest_difference (a b c : ℕ) (h1: a = 10) (h2: b = 11) (h3: c = 12) : 
  let largest := max a (max b c)
  let next_largest := min (max a b) (max (min a b) c)
  largest - next_largest = 1 :=
by
  -- Proof to be filled in for verification
  sorry

end largest_and_next_largest_difference_l1049_104931


namespace ratio_of_dogs_to_cats_l1049_104943

theorem ratio_of_dogs_to_cats (D C : ℕ) (hC : C = 40) (h : D + 20 = 2 * C) :
  D / Nat.gcd D C = 3 ∧ C / Nat.gcd D C = 2 :=
by
  sorry

end ratio_of_dogs_to_cats_l1049_104943


namespace comb_identity_l1049_104960

theorem comb_identity (n : Nat) (h : 0 < n) (h_eq : Nat.choose n 2 = Nat.choose (n-1) 2 + Nat.choose (n-1) 3) : n = 5 := by
  sorry

end comb_identity_l1049_104960


namespace initial_speed_100kmph_l1049_104922

theorem initial_speed_100kmph (v x : ℝ) (h1 : 0 < v) (h2 : 100 - x = v / 2) 
  (h3 : (80 - x) / (v - 10) - 20 / (v - 20) = 1 / 12) : v = 100 :=
by 
  sorry

end initial_speed_100kmph_l1049_104922


namespace salad_dressing_percentage_l1049_104988

variable (P Q : ℝ) -- P and Q are the amounts of dressings P and Q in grams

-- Conditions
variable (h1 : 0.3 * P + 0.1 * Q = 12) -- The combined vinegar percentage condition
variable (h2 : P + Q = 100)            -- The total weight condition

-- Statement to prove
theorem salad_dressing_percentage (P_percent : ℝ) 
    (h1 : 0.3 * P + 0.1 * Q = 12) (h2 : P + Q = 100) : 
    P / (P + Q) * 100 = 10 :=
sorry

end salad_dressing_percentage_l1049_104988


namespace Liam_cycling_speed_l1049_104978

theorem Liam_cycling_speed :
  ∀ (Eugene_speed Claire_speed Liam_speed : ℝ),
    Eugene_speed = 6 →
    Claire_speed = (3/4) * Eugene_speed →
    Liam_speed = (4/3) * Claire_speed →
    Liam_speed = 6 :=
by
  intros
  sorry

end Liam_cycling_speed_l1049_104978


namespace find_prime_pair_l1049_104973

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 + p * x^3 - q = 0

theorem find_prime_pair :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ has_integer_root p q ∧ p = 2 ∧ q = 3 := by
  sorry

end find_prime_pair_l1049_104973


namespace find_f_of_2_l1049_104999

theorem find_f_of_2 : ∃ (f : ℤ → ℤ), (∀ x : ℤ, f (x+1) = x^2 - 1) ∧ f 2 = 0 :=
by
  sorry

end find_f_of_2_l1049_104999


namespace arithmetic_mean_first_n_positive_integers_l1049_104963

theorem arithmetic_mean_first_n_positive_integers (n : ℕ) (Sn : ℕ) (h : Sn = n * (n + 1) / 2) : 
  (Sn / n) = (n + 1) / 2 := by
  -- proof steps would go here
  sorry

end arithmetic_mean_first_n_positive_integers_l1049_104963


namespace condo_floors_l1049_104984

theorem condo_floors (F P : ℕ) (h1: 12 * F + 2 * P = 256) (h2 : P = 2) : F + P = 23 :=
by
  sorry

end condo_floors_l1049_104984


namespace gum_pieces_per_package_l1049_104930

theorem gum_pieces_per_package (packages : ℕ) (extra : ℕ) (total : ℕ) (pieces_per_package : ℕ) :
    packages = 43 → extra = 8 → total = 997 → 43 * pieces_per_package + extra = total → pieces_per_package = 23 :=
by
  intros hpkg hextra htotal htotal_eq
  sorry

end gum_pieces_per_package_l1049_104930


namespace relay_race_total_time_l1049_104947

noncomputable def mary_time (susan_time : ℕ) : ℕ := 2 * susan_time
noncomputable def susan_time (jen_time : ℕ) : ℕ := jen_time + 10
def jen_time : ℕ := 30
noncomputable def tiffany_time (mary_time : ℕ) : ℕ := mary_time - 7

theorem relay_race_total_time :
  let mary_time := mary_time (susan_time jen_time)
  let susan_time := susan_time jen_time
  let tiffany_time := tiffany_time mary_time
  mary_time + susan_time + jen_time + tiffany_time = 223 := by
  sorry

end relay_race_total_time_l1049_104947


namespace least_four_digit_multiple_3_5_7_l1049_104946

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end least_four_digit_multiple_3_5_7_l1049_104946


namespace A_fraction_simplification_l1049_104919

noncomputable def A : ℚ := 
  ((3/8) * (13/5)) / ((5/2) * (6/5)) +
  ((5/8) * (8/5)) / (3 * (6/5) * (25/6)) +
  (20/3) * (3/25) +
  28 +
  (1 / 9) / 7 +
  (1/5) / (9 * 22)

theorem A_fraction_simplification :
  let num := 1901
  let denom := 3360
  (A = num / denom) :=
sorry

end A_fraction_simplification_l1049_104919


namespace real_solution_count_l1049_104949

/-- Given \( \lfloor x \rfloor \) is the greatest integer less than or equal to \( x \),
prove that the number of real solutions to the equation \( 9x^2 - 36\lfloor x \rfloor + 20 = 0 \) is 2. --/
theorem real_solution_count (x : ℝ) (h : ⌊x⌋ = Int.floor x) :
  ∃ (S : Finset ℝ), S.card = 2 ∧ ∀ a ∈ S, 9 * a^2 - 36 * ⌊a⌋ + 20 = 0 :=
sorry

end real_solution_count_l1049_104949


namespace parabola_focus_coordinates_l1049_104997

theorem parabola_focus_coordinates (x y p : ℝ) (h : y^2 = 8 * x) : 
  p = 2 → (p, 0) = (2, 0) := 
by 
  sorry

end parabola_focus_coordinates_l1049_104997


namespace find_q_l1049_104905

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 6) : q = 3 + Real.sqrt 3 :=
by
  sorry

end find_q_l1049_104905
